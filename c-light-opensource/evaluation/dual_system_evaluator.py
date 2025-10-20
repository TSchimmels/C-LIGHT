"""
Dual System Evaluator
Compares System A (Mixtral) vs System B (SEAL) performance
"""

import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import numpy as np

from ..core.base_types import Paper, QueryResult, KnowledgeDomain
from ..system_a.llm_rag import MixtralRAG
from ..system_b.seal_rag import SEALBasedRAG

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    """A query for evaluation"""
    query_id: str
    question: str
    ground_truth: Optional[str] = None  # If known
    expected_answer: Optional[str] = None
    domains: List[KnowledgeDomain] = field(default_factory=list)


@dataclass
class SystemComparison:
    """Comparison of both systems on a single query"""
    query_id: str
    question: str
    timestamp: datetime

    # System A (Mixtral)
    system_a_answer: str
    system_a_confidence: float
    system_a_latency: float
    system_a_sources: int

    # System B (SEAL)
    system_b_answer: str
    system_b_confidence: float
    system_b_latency: float
    system_b_sources: int

    # User ratings (if provided)
    system_a_rating: Optional[int] = None  # 1-5
    system_b_rating: Optional[int] = None  # 1-5

    # Winner (if rated)
    winner: Optional[str] = None  # 'system_a', 'system_b', or 'tie'

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class DualSystemEvaluator:
    """
    Evaluates and compares System A (Mixtral) and System B (SEAL)

    Key metrics:
    1. Accuracy - how often each system gets it right
    2. Latency - response time
    3. Confidence calibration - does confidence match accuracy?
    4. Improvement over time - System B should improve, System A should plateau
    5. Domain performance - which system is better in which domains?
    6. Citation quality - do answers cite papers correctly?
    """

    def __init__(
        self,
        system_a: MixtralRAG,
        system_b: SEALBasedRAG,
        storage_path: str = "/mnt/nvme/c-light/evaluation"
    ):
        self.system_a = system_a
        self.system_b = system_b

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Comparison history
        self.comparisons: List[SystemComparison] = []

        # Statistics
        self.stats = {
            'system_a': {
                'queries': 0,
                'avg_rating': 0.0,
                'wins': 0,
                'avg_latency': 0.0,
                'avg_confidence': 0.0
            },
            'system_b': {
                'queries': 0,
                'avg_rating': 0.0,
                'wins': 0,
                'avg_latency': 0.0,
                'avg_confidence': 0.0
            },
            'ties': 0
        }

        # Load existing comparisons
        self._load_comparisons()

        logger.info("Dual system evaluator initialized")

    def compare_query(
        self,
        question: str,
        domains: Optional[List[KnowledgeDomain]] = None,
        ground_truth: Optional[str] = None
    ) -> SystemComparison:
        """
        Run same query through both systems and compare

        Args:
            question: Question to ask
            domains: Optional domain filter
            ground_truth: Optional correct answer for automatic evaluation

        Returns:
            SystemComparison with results from both systems
        """
        logger.info(f"Comparing systems on: {question[:60]}...")

        # Run System A
        start_a = time.time()
        result_a = self.system_a.query(question, domains)
        latency_a = time.time() - start_a

        # Run System B
        start_b = time.time()
        result_b = self.system_b.query(question, domains)
        latency_b = time.time() - start_b

        # Create comparison
        comparison = SystemComparison(
            query_id=result_a.metadata.get('query_id', str(time.time())),
            question=question,
            timestamp=datetime.now(),
            system_a_answer=result_a.answer,
            system_a_confidence=result_a.confidence,
            system_a_latency=latency_a,
            system_a_sources=result_a.sources_count,
            system_b_answer=result_b.answer,
            system_b_confidence=result_b.confidence,
            system_b_latency=latency_b,
            system_b_sources=result_b.sources_count
        )

        # If ground truth provided, do automatic evaluation
        if ground_truth:
            comparison.system_a_rating = self._evaluate_answer(
                result_a.answer, ground_truth
            )
            comparison.system_b_rating = self._evaluate_answer(
                result_b.answer, ground_truth
            )
            comparison.winner = self._determine_winner(comparison)

        # Store comparison
        self.comparisons.append(comparison)
        self._save_comparison(comparison)
        self._update_stats(comparison)

        logger.info(f"✓ Comparison complete")
        logger.info(f"  System A: {latency_a:.2f}s, confidence={result_a.confidence:.2f}")
        logger.info(f"  System B: {latency_b:.2f}s, confidence={result_b.confidence:.2f}")

        return comparison

    def add_user_ratings(
        self,
        query_id: str,
        system_a_rating: int,
        system_b_rating: int
    ):
        """
        Add user ratings for a comparison

        Args:
            query_id: Query to rate
            system_a_rating: 1-5 rating for System A
            system_b_rating: 1-5 rating for System B
        """
        # Find comparison
        comparison = None
        for c in self.comparisons:
            if c.query_id == query_id:
                comparison = c
                break

        if not comparison:
            logger.warning(f"Query {query_id} not found")
            return

        # Update ratings
        comparison.system_a_rating = system_a_rating
        comparison.system_b_rating = system_b_rating
        comparison.winner = self._determine_winner(comparison)

        # Also provide feedback to System B (for learning)
        self.system_b.provide_feedback(
            query_id=query_id,
            rating=system_b_rating
        )

        # Update stats
        self._update_stats(comparison)
        self._save_comparison(comparison)

        logger.info(f"Added ratings for {query_id}: A={system_a_rating}, B={system_b_rating}")

    def _determine_winner(self, comparison: SystemComparison) -> str:
        """Determine which system won"""
        if comparison.system_a_rating is None or comparison.system_b_rating is None:
            return None

        if comparison.system_a_rating > comparison.system_b_rating:
            return 'system_a'
        elif comparison.system_b_rating > comparison.system_a_rating:
            return 'system_b'
        else:
            return 'tie'

    def _evaluate_answer(self, answer: str, ground_truth: str) -> int:
        """
        Automatically evaluate answer against ground truth

        Returns rating 1-5
        """
        answer_lower = answer.lower()
        truth_lower = ground_truth.lower()

        # Extract key concepts from ground truth
        truth_words = set(truth_lower.split())
        answer_words = set(answer_lower.split())

        # Calculate overlap
        overlap = len(truth_words & answer_words) / max(len(truth_words), 1)

        # Convert to rating
        if overlap > 0.8:
            return 5  # Excellent
        elif overlap > 0.6:
            return 4  # Good
        elif overlap > 0.4:
            return 3  # Acceptable
        elif overlap > 0.2:
            return 2  # Poor
        else:
            return 1  # Terrible

    def _update_stats(self, comparison: SystemComparison):
        """Update running statistics"""
        # System A
        self.stats['system_a']['queries'] += 1
        self.stats['system_a']['avg_latency'] = (
            (self.stats['system_a']['avg_latency'] * (self.stats['system_a']['queries'] - 1) +
             comparison.system_a_latency) / self.stats['system_a']['queries']
        )
        self.stats['system_a']['avg_confidence'] = (
            (self.stats['system_a']['avg_confidence'] * (self.stats['system_a']['queries'] - 1) +
             comparison.system_a_confidence) / self.stats['system_a']['queries']
        )

        # System B
        self.stats['system_b']['queries'] += 1
        self.stats['system_b']['avg_latency'] = (
            (self.stats['system_b']['avg_latency'] * (self.stats['system_b']['queries'] - 1) +
             comparison.system_b_latency) / self.stats['system_b']['queries']
        )
        self.stats['system_b']['avg_confidence'] = (
            (self.stats['system_b']['avg_confidence'] * (self.stats['system_b']['queries'] - 1) +
             comparison.system_b_confidence) / self.stats['system_b']['queries']
        )

        # Update ratings if available
        if comparison.system_a_rating:
            rated_a = len([c for c in self.comparisons if c.system_a_rating])
            total_rating_a = sum(c.system_a_rating for c in self.comparisons if c.system_a_rating)
            self.stats['system_a']['avg_rating'] = total_rating_a / max(rated_a, 1)

        if comparison.system_b_rating:
            rated_b = len([c for c in self.comparisons if c.system_b_rating])
            total_rating_b = sum(c.system_b_rating for c in self.comparisons if c.system_b_rating)
            self.stats['system_b']['avg_rating'] = total_rating_b / max(rated_b, 1)

        # Update wins
        if comparison.winner == 'system_a':
            self.stats['system_a']['wins'] += 1
        elif comparison.winner == 'system_b':
            self.stats['system_b']['wins'] += 1
        elif comparison.winner == 'tie':
            self.stats['ties'] += 1

    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get summary of all comparisons"""
        rated_comparisons = [c for c in self.comparisons if c.winner]

        return {
            'total_comparisons': len(self.comparisons),
            'rated_comparisons': len(rated_comparisons),
            'system_a': {
                **self.stats['system_a'],
                'win_rate': self.stats['system_a']['wins'] / max(len(rated_comparisons), 1)
            },
            'system_b': {
                **self.stats['system_b'],
                'win_rate': self.stats['system_b']['wins'] / max(len(rated_comparisons), 1)
            },
            'ties': self.stats['ties'],
            'tie_rate': self.stats['ties'] / max(len(rated_comparisons), 1)
        }

    def analyze_improvement_over_time(
        self,
        window_size: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze if System B is improving relative to System A

        System B should improve over time, System A should plateau

        Args:
            window_size: Number of queries to use for rolling average

        Returns:
            Analysis of improvement trends
        """
        rated = [c for c in self.comparisons if c.system_a_rating and c.system_b_rating]

        if len(rated) < window_size * 2:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least {window_size * 2} rated queries'
            }

        # Split into early and recent windows
        early_window = rated[:window_size]
        recent_window = rated[-window_size:]

        # System A performance (should be flat)
        system_a_early = np.mean([c.system_a_rating for c in early_window])
        system_a_recent = np.mean([c.system_a_rating for c in recent_window])
        system_a_change = system_a_recent - system_a_early

        # System B performance (should improve)
        system_b_early = np.mean([c.system_b_rating for c in early_window])
        system_b_recent = np.mean([c.system_b_rating for c in recent_window])
        system_b_change = system_b_recent - system_b_early

        # Win rates
        system_a_wins_early = len([c for c in early_window if c.winner == 'system_a']) / len(early_window)
        system_a_wins_recent = len([c for c in recent_window if c.winner == 'system_a']) / len(recent_window)

        system_b_wins_early = len([c for c in early_window if c.winner == 'system_b']) / len(early_window)
        system_b_wins_recent = len([c for c in recent_window if c.winner == 'system_b']) / len(recent_window)

        return {
            'system_a': {
                'early_rating': system_a_early,
                'recent_rating': system_a_recent,
                'change': system_a_change,
                'early_win_rate': system_a_wins_early,
                'recent_win_rate': system_a_wins_recent,
                'improving': system_a_change > 0.1
            },
            'system_b': {
                'early_rating': system_b_early,
                'recent_rating': system_b_recent,
                'change': system_b_change,
                'early_win_rate': system_b_wins_early,
                'recent_win_rate': system_b_wins_recent,
                'improving': system_b_change > 0.1
            },
            'hypothesis_confirmed': system_b_change > system_a_change and system_b_change > 0.2,
            'message': self._generate_improvement_message(
                system_a_change, system_b_change,
                system_b_wins_recent, system_a_wins_recent
            )
        }

    def _generate_improvement_message(
        self,
        system_a_change: float,
        system_b_change: float,
        system_b_win_rate: float,
        system_a_win_rate: float
    ) -> str:
        """Generate human-readable improvement message"""
        if system_b_change > 0.2 and system_b_change > system_a_change:
            if system_b_win_rate > system_a_win_rate:
                return "✓ System B (SEAL) is learning and now outperforms System A (Mixtral)"
            else:
                return "System B is improving but hasn't surpassed System A yet"
        elif system_b_change > 0:
            return "System B is improving slowly"
        elif system_a_change > system_b_change:
            return "⚠ System A is performing better - System B may need tuning"
        else:
            return "Neither system showing clear improvement"

    def compare_by_domain(self) -> Dict[KnowledgeDomain, Dict[str, Any]]:
        """Compare systems by domain"""
        domain_performance = defaultdict(lambda: {
            'system_a_ratings': [],
            'system_b_ratings': []
        })

        for comparison in self.comparisons:
            if comparison.system_a_rating and comparison.system_b_rating:
                # Note: We'd need to track domains per query
                # For now, this is a placeholder
                pass

        # Calculate per-domain stats
        results = {}
        for domain, data in domain_performance.items():
            if data['system_a_ratings'] and data['system_b_ratings']:
                results[domain] = {
                    'system_a_avg': np.mean(data['system_a_ratings']),
                    'system_b_avg': np.mean(data['system_b_ratings']),
                    'better_system': 'system_b' if np.mean(data['system_b_ratings']) > np.mean(data['system_a_ratings']) else 'system_a'
                }

        return results

    def analyze_confidence_calibration(self) -> Dict[str, Any]:
        """
        Check if confidence scores match actual performance

        Well-calibrated: high confidence → high rating, low confidence → low rating
        """
        rated = [c for c in self.comparisons if c.system_a_rating and c.system_b_rating]

        if not rated:
            return {'status': 'insufficient_data'}

        # System A calibration
        system_a_high_conf = [c for c in rated if c.system_a_confidence > 0.7]
        system_a_low_conf = [c for c in rated if c.system_a_confidence < 0.3]

        system_a_high_conf_accuracy = (
            np.mean([c.system_a_rating for c in system_a_high_conf]) / 5.0
            if system_a_high_conf else 0.0
        )
        system_a_low_conf_accuracy = (
            np.mean([c.system_a_rating for c in system_a_low_conf]) / 5.0
            if system_a_low_conf else 0.0
        )

        # System B calibration
        system_b_high_conf = [c for c in rated if c.system_b_confidence > 0.7]
        system_b_low_conf = [c for c in rated if c.system_b_confidence < 0.3]

        system_b_high_conf_accuracy = (
            np.mean([c.system_b_rating for c in system_b_high_conf]) / 5.0
            if system_b_high_conf else 0.0
        )
        system_b_low_conf_accuracy = (
            np.mean([c.system_b_rating for c in system_b_low_conf]) / 5.0
            if system_b_low_conf else 0.0
        )

        return {
            'system_a': {
                'high_confidence_accuracy': system_a_high_conf_accuracy,
                'low_confidence_accuracy': system_a_low_conf_accuracy,
                'calibration_gap': system_a_high_conf_accuracy - system_a_low_conf_accuracy,
                'well_calibrated': system_a_high_conf_accuracy - system_a_low_conf_accuracy > 0.2
            },
            'system_b': {
                'high_confidence_accuracy': system_b_high_conf_accuracy,
                'low_confidence_accuracy': system_b_low_conf_accuracy,
                'calibration_gap': system_b_high_conf_accuracy - system_b_low_conf_accuracy,
                'well_calibrated': system_b_high_conf_accuracy - system_b_low_conf_accuracy > 0.2
            }
        }

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive comparison report

        Args:
            output_path: Optional path to save report

        Returns:
            Report as markdown string
        """
        summary = self.get_comparison_summary()
        improvement = self.analyze_improvement_over_time()
        calibration = self.analyze_confidence_calibration()

        report = f"""# Dual System Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

Total comparisons: {summary['total_comparisons']}
Rated comparisons: {summary['rated_comparisons']}

### System A (Mixtral LLM-based RAG)
- **Win Rate**: {summary['system_a']['win_rate']:.1%}
- **Average Rating**: {summary['system_a']['avg_rating']:.2f}/5.0
- **Average Latency**: {summary['system_a']['avg_latency']:.2f}s
- **Average Confidence**: {summary['system_a']['avg_confidence']:.2f}

### System B (SEAL Self-Learning RAG)
- **Win Rate**: {summary['system_b']['win_rate']:.1%}
- **Average Rating**: {summary['system_b']['avg_rating']:.2f}/5.0
- **Average Latency**: {summary['system_b']['avg_latency']:.2f}s
- **Average Confidence**: {summary['system_b']['avg_confidence']:.2f}

### Ties
- **Tie Rate**: {summary['tie_rate']:.1%}

---

## Improvement Over Time

{improvement.get('message', 'Analyzing...')}

### System A
- Early rating: {improvement['system_a']['early_rating']:.2f}
- Recent rating: {improvement['system_a']['recent_rating']:.2f}
- Change: {improvement['system_a']['change']:+.2f}
- Improving: {improvement['system_a']['improving']}

### System B
- Early rating: {improvement['system_b']['early_rating']:.2f}
- Recent rating: {improvement['system_b']['recent_rating']:.2f}
- Change: {improvement['system_b']['change']:+.2f}
- Improving: {improvement['system_b']['improving']}

**Hypothesis (System B surpasses System A after 2-3 months)**: {'✓ CONFIRMED' if improvement.get('hypothesis_confirmed') else '⏳ Not yet confirmed'}

---

## Confidence Calibration

Well-calibrated systems have high accuracy when confident, low accuracy when uncertain.

### System A
- High confidence (>0.7) accuracy: {calibration['system_a']['high_confidence_accuracy']:.1%}
- Low confidence (<0.3) accuracy: {calibration['system_a']['low_confidence_accuracy']:.1%}
- Calibration gap: {calibration['system_a']['calibration_gap']:.2f}
- Well calibrated: {calibration['system_a']['well_calibrated']}

### System B
- High confidence (>0.7) accuracy: {calibration['system_b']['high_confidence_accuracy']:.1%}
- Low confidence (<0.3) accuracy: {calibration['system_b']['low_confidence_accuracy']:.1%}
- Calibration gap: {calibration['system_b']['calibration_gap']:.2f}
- Well calibrated: {calibration['system_b']['well_calibrated']}

---

## Recommendations

"""

        # Add recommendations
        if summary['system_a']['win_rate'] > 0.6:
            report += "- System A is currently performing better - stick with Mixtral for now\n"
        elif summary['system_b']['win_rate'] > 0.6:
            report += "- ✓ System B has surpassed System A - SEAL is working!\n"
        else:
            report += "- Systems are roughly equal - continue collecting feedback\n"

        if improvement.get('hypothesis_confirmed'):
            report += "- ✓ Learning hypothesis confirmed - System B is improving over time\n"

        if not calibration['system_b']['well_calibrated']:
            report += "- System B confidence needs calibration - consider adjusting edge weight learning\n"

        if summary['rated_comparisons'] < 100:
            report += f"- Need more feedback - only {summary['rated_comparisons']} rated queries (target: 100+)\n"

        # Save if path provided
        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def _save_comparison(self, comparison: SystemComparison):
        """Save comparison to disk"""
        filename = f"{comparison.query_id}.json"
        filepath = self.storage_path / filename

        with open(filepath, 'w') as f:
            json.dump(comparison.to_dict(), f, indent=2)

    def _load_comparisons(self):
        """Load existing comparisons from disk"""
        if not self.storage_path.exists():
            return

        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    comparison = SystemComparison(**data)
                    self.comparisons.append(comparison)

            except Exception as e:
                logger.error(f"Error loading comparison from {filepath}: {e}")

        # Recalculate stats
        for comparison in self.comparisons:
            self._update_stats(comparison)

        logger.info(f"Loaded {len(self.comparisons)} existing comparisons")

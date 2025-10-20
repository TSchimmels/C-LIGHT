"""
Feedback Collection System
Tracks query success/failure to drive learning
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QueryFeedback:
    """Feedback for a single query"""
    query_id: str
    timestamp: datetime

    # Query details
    question: str
    answer: str

    # Feedback
    rating: int  # 1-5 scale (1=terrible, 5=excellent)
    correct_answer: Optional[str] = None  # User can provide correct answer

    # System state at query time
    papers_retrieved: List[str] = field(default_factory=list)
    causal_path: List[str] = field(default_factory=list)
    confidence: float = 0.0

    # Metadata
    domains: List[str] = field(default_factory=list)
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryFeedback':
        """Load from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class FeedbackCollector:
    """
    Collects and manages feedback for self-learning

    This is the core of SEAL's learning loop:
    1. User queries system
    2. System provides answer
    3. User rates answer (1-5) or provides correct answer
    4. System learns from feedback
    """

    def __init__(self, storage_path: str = "/mnt/nvme/c-light/seal/feedback"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory feedback store
        self.feedback_history: List[QueryFeedback] = []
        self.query_lookup: Dict[str, QueryFeedback] = {}

        # Statistics
        self.stats = {
            'total_queries': 0,
            'total_feedback': 0,
            'avg_rating': 0.0,
            'ratings_by_score': defaultdict(int)
        }

        # Load existing feedback
        self._load_feedback()

        logger.info(f"Feedback collector initialized: {len(self.feedback_history)} existing feedback entries")

    def record_query(
        self,
        query_id: str,
        question: str,
        answer: str,
        papers_retrieved: List[str],
        causal_path: List[str],
        confidence: float,
        domains: List[str] = None
    ):
        """
        Record a query (before feedback is received)

        Args:
            query_id: Unique ID for this query
            question: User's question
            answer: System's answer
            papers_retrieved: Papers used in answer
            causal_path: Causal path through knowledge graph
            confidence: System's confidence score
            domains: Relevant domains
        """
        feedback = QueryFeedback(
            query_id=query_id,
            timestamp=datetime.now(),
            question=question,
            answer=answer,
            rating=0,  # Not rated yet
            papers_retrieved=papers_retrieved,
            causal_path=causal_path,
            confidence=confidence,
            domains=domains or []
        )

        self.query_lookup[query_id] = feedback
        self.stats['total_queries'] += 1

    def add_feedback(
        self,
        query_id: str,
        rating: int,
        correct_answer: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Add user feedback for a query

        Args:
            query_id: Query to rate
            rating: 1-5 rating (1=terrible, 5=excellent)
            correct_answer: Optional correct answer if system was wrong
            user_id: Optional user identifier

        Returns:
            True if feedback recorded successfully
        """
        if query_id not in self.query_lookup:
            logger.warning(f"Query {query_id} not found")
            return False

        if not (1 <= rating <= 5):
            logger.warning(f"Invalid rating: {rating}, must be 1-5")
            return False

        # Update feedback
        feedback = self.query_lookup[query_id]
        feedback.rating = rating
        feedback.correct_answer = correct_answer
        feedback.user_id = user_id

        # Add to history
        self.feedback_history.append(feedback)

        # Update stats
        self.stats['total_feedback'] += 1
        self.stats['ratings_by_score'][rating] += 1
        self._update_avg_rating()

        # Save to disk
        self._save_feedback(feedback)

        logger.info(f"Recorded feedback for query {query_id}: rating={rating}")

        return True

    def get_feedback(self, query_id: str) -> Optional[QueryFeedback]:
        """Get feedback for a specific query"""
        return self.query_lookup.get(query_id)

    def get_recent_feedback(self, n: int = 100) -> List[QueryFeedback]:
        """Get N most recent feedback entries"""
        return sorted(
            self.feedback_history,
            key=lambda x: x.timestamp,
            reverse=True
        )[:n]

    def get_feedback_by_rating(self, rating: int) -> List[QueryFeedback]:
        """Get all feedback with specific rating"""
        return [f for f in self.feedback_history if f.rating == rating]

    def get_failed_queries(self, threshold: int = 2) -> List[QueryFeedback]:
        """Get queries with rating <= threshold (failures)"""
        return [f for f in self.feedback_history if f.rating <= threshold]

    def get_successful_queries(self, threshold: int = 4) -> List[QueryFeedback]:
        """Get queries with rating >= threshold (successes)"""
        return [f for f in self.feedback_history if f.rating >= threshold]

    def get_feedback_for_domain(self, domain: str) -> List[QueryFeedback]:
        """Get feedback for specific domain"""
        return [
            f for f in self.feedback_history
            if domain in f.domains
        ]

    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze feedback to find patterns
        What types of queries fail? What domains are weak?
        """
        if not self.feedback_history:
            return {}

        # Domain performance
        domain_ratings = defaultdict(list)
        for feedback in self.feedback_history:
            for domain in feedback.domains:
                domain_ratings[domain].append(feedback.rating)

        domain_performance = {
            domain: {
                'avg_rating': sum(ratings) / len(ratings),
                'count': len(ratings),
                'success_rate': len([r for r in ratings if r >= 4]) / len(ratings)
            }
            for domain, ratings in domain_ratings.items()
        }

        # Path length analysis
        path_lengths = [len(f.causal_path) for f in self.feedback_history if f.causal_path]

        # Confidence calibration
        high_conf_correct = [
            f for f in self.feedback_history
            if f.confidence > 0.7 and f.rating >= 4
        ]
        high_conf_wrong = [
            f for f in self.feedback_history
            if f.confidence > 0.7 and f.rating <= 2
        ]

        return {
            'total_feedback': len(self.feedback_history),
            'avg_rating': self.stats['avg_rating'],
            'domain_performance': domain_performance,
            'avg_path_length': sum(path_lengths) / max(len(path_lengths), 1),
            'confidence_calibration': {
                'high_conf_correct': len(high_conf_correct),
                'high_conf_wrong': len(high_conf_wrong),
                'calibration_accuracy': len(high_conf_correct) / max(len(high_conf_correct) + len(high_conf_wrong), 1)
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self.stats,
            'success_rate': len(self.get_successful_queries()) / max(len(self.feedback_history), 1),
            'failure_rate': len(self.get_failed_queries()) / max(len(self.feedback_history), 1)
        }

    def _update_avg_rating(self):
        """Update average rating"""
        if self.feedback_history:
            self.stats['avg_rating'] = sum(f.rating for f in self.feedback_history) / len(self.feedback_history)

    def _save_feedback(self, feedback: QueryFeedback):
        """Save feedback to disk"""
        filename = f"{feedback.query_id}.json"
        filepath = self.storage_path / filename

        with open(filepath, 'w') as f:
            json.dump(feedback.to_dict(), f, indent=2)

    def _load_feedback(self):
        """Load existing feedback from disk"""
        if not self.storage_path.exists():
            return

        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    feedback = QueryFeedback.from_dict(data)

                    if feedback.rating > 0:  # Only load rated queries
                        self.feedback_history.append(feedback)
                        self.query_lookup[feedback.query_id] = feedback
                        self.stats['ratings_by_score'][feedback.rating] += 1

            except Exception as e:
                logger.error(f"Error loading feedback from {filepath}: {e}")

        self.stats['total_feedback'] = len(self.feedback_history)
        self._update_avg_rating()

    def export_training_data(self) -> List[Dict[str, Any]]:
        """
        Export feedback as training data for pattern learning

        Returns:
            List of training examples with features and labels
        """
        training_data = []

        for feedback in self.feedback_history:
            if feedback.rating == 0:  # Skip unrated
                continue

            example = {
                'question': feedback.question,
                'answer': feedback.answer,
                'correct_answer': feedback.correct_answer,
                'causal_path': feedback.causal_path,
                'papers': feedback.papers_retrieved,
                'domains': feedback.domains,
                'label': 1 if feedback.rating >= 4 else 0,  # Binary: success or failure
                'rating': feedback.rating
            }
            training_data.append(example)

        return training_data

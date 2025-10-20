"""
Pattern Learner for SEAL System
Discovers new causal extraction patterns from feedback
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import json
from pathlib import Path

from ..core.base_types import Paper, CausalRelation, KnowledgeDomain

logger = logging.getLogger(__name__)


@dataclass
class CausalPattern:
    """A learned pattern for extracting causal relationships"""
    pattern_id: str
    regex: str
    relation_type: str  # causes, increases, decreases, etc.

    # Learning statistics
    successes: int = 0
    failures: int = 0
    confidence: float = 0.5  # Updated based on performance

    # Example matches (for debugging/analysis)
    examples: List[str] = field(default_factory=list)

    def update_stats(self, success: bool):
        """Update pattern performance stats"""
        if success:
            self.successes += 1
        else:
            self.failures += 1

        # Update confidence (Bayesian update)
        total = self.successes + self.failures
        if total > 0:
            self.confidence = (self.successes + 1) / (total + 2)  # +1/+2 for smoothing

    def get_performance(self) -> float:
        """Get success rate"""
        total = self.successes + self.failures
        return self.successes / max(total, 1)


class PatternLearner:
    """
    Learns new causal extraction patterns from feedback

    Key idea: When users provide correct answers, analyze the text to discover
    new linguistic patterns that indicate causation.

    Example:
    - User says answer is wrong
    - User provides correct answer: "X modulates Y through Z"
    - System learns: "modulates...through" is a causal pattern
    - Applies this pattern to future papers
    """

    def __init__(self, storage_path: str = "/mnt/nvme/c-light/seal/patterns"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initial patterns (from rule-based extractor)
        self.patterns: Dict[str, CausalPattern] = {}
        self._initialize_base_patterns()

        # Pattern discovery
        self.candidate_patterns: Dict[str, int] = defaultdict(int)  # pattern -> count

        # Load saved patterns
        self._load_patterns()

        logger.info(f"Pattern learner initialized with {len(self.patterns)} patterns")

    def _initialize_base_patterns(self):
        """Initialize with basic patterns"""
        base_patterns = [
            # Strong causal
            ("causes", r'(\w+(?:\s+\w+){0,3})\s+causes\s+(\w+(?:\s+\w+){0,3})', 0.9),
            ("leads_to", r'(\w+(?:\s+\w+){0,3})\s+leads to\s+(\w+(?:\s+\w+){0,3})', 0.85),
            ("results_in", r'(\w+(?:\s+\w+){0,3})\s+results in\s+(\w+(?:\s+\w+){0,3})', 0.8),
            ("induces", r'(\w+(?:\s+\w+){0,3})\s+induces\s+(\w+(?:\s+\w+){0,3})', 0.8),

            # Moderate causal
            ("increases", r'(\w+(?:\s+\w+){0,3})\s+increases\s+(\w+(?:\s+\w+){0,3})', 0.7),
            ("decreases", r'(\w+(?:\s+\w+){0,3})\s+decreases\s+(\w+(?:\s+\w+){0,3})', 0.7),
            ("enhances", r'(\w+(?:\s+\w+){0,3})\s+enhances\s+(\w+(?:\s+\w+){0,3})', 0.65),
            ("modulates", r'(\w+(?:\s+\w+){0,3})\s+modulates\s+(\w+(?:\s+\w+){0,3})', 0.6),

            # Weaker
            ("affects", r'(\w+(?:\s+\w+){0,3})\s+affects\s+(\w+(?:\s+\w+){0,3})', 0.5),
        ]

        for pattern_id, regex, confidence in base_patterns:
            self.patterns[pattern_id] = CausalPattern(
                pattern_id=pattern_id,
                regex=regex,
                relation_type=pattern_id,
                confidence=confidence
            )

    def extract_relations(
        self,
        paper: Paper,
        use_learned: bool = True
    ) -> List[CausalRelation]:
        """
        Extract causal relations using current patterns

        Args:
            paper: Paper to extract from
            use_learned: Whether to use learned patterns (vs. just base patterns)

        Returns:
            List of extracted CausalRelation objects
        """
        relations = []
        text = paper.abstract + " " + (paper.full_text or "")

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        patterns_to_use = self.patterns if use_learned else {
            k: v for k, v in self.patterns.items()
            if v.successes + v.failures == 0  # Only base patterns
        }

        for sentence in sentences:
            for pattern_id, pattern in patterns_to_use.items():
                matches = re.finditer(pattern.regex, sentence, re.IGNORECASE)

                for match in matches:
                    source = self._clean_entity(match.group(1))
                    target = self._clean_entity(match.group(2))

                    if not source or not target or len(source) < 3 or len(target) < 3:
                        continue

                    relation = CausalRelation(
                        source=source,
                        target=target,
                        relation_type=pattern.relation_type,
                        evidence_text=sentence,
                        paper_id=paper.paper_id,
                        confidence=pattern.confidence,
                        domains=paper.domains
                    )
                    relations.append(relation)

                    # Store example
                    if len(pattern.examples) < 10:
                        pattern.examples.append(sentence)

        return relations

    def learn_from_feedback(
        self,
        question: str,
        system_answer: str,
        correct_answer: str,
        papers: List[Paper],
        rating: int
    ):
        """
        Learn from user feedback

        When a query succeeds/fails, update pattern performance.
        When user provides correct answer, try to discover new patterns.

        Args:
            question: User's question
            system_answer: System's answer
            correct_answer: Correct answer (if provided)
            papers: Papers that were used
            rating: User rating (1-5)
        """
        success = rating >= 4

        # If user provided correct answer, try to discover new patterns
        if correct_answer:
            self._discover_patterns_from_answer(
                question,
                correct_answer,
                papers
            )

        # Update pattern performance based on success/failure
        # This is tricky: we need to figure out which patterns were used
        # For now, boost/penalize all patterns proportionally
        for pattern in self.patterns.values():
            # Small update to all patterns
            pattern.update_stats(success)

    def _discover_patterns_from_answer(
        self,
        question: str,
        correct_answer: str,
        papers: List[Paper]
    ):
        """
        Discover new causal patterns from correct answer

        Strategy:
        1. Find key concepts in correct answer
        2. Search for those concepts in papers
        3. Extract surrounding text patterns
        4. Identify potential causal connectors
        """
        # Extract concept pairs from correct answer
        concepts = self._extract_concepts(correct_answer)

        if len(concepts) < 2:
            return

        # Look for these concepts co-occurring in papers
        for paper in papers:
            text = paper.abstract + " " + (paper.full_text or "")

            for i in range(len(concepts) - 1):
                concept1 = concepts[i]
                concept2 = concepts[i + 1]

                # Find sentences containing both concepts
                sentences = re.split(r'(?<=[.!?])\s+', text)

                for sentence in sentences:
                    if concept1.lower() in sentence.lower() and concept2.lower() in sentence.lower():
                        # Found a sentence with both concepts!
                        # Extract the connecting words
                        connector = self._extract_connector(sentence, concept1, concept2)

                        if connector:
                            self.candidate_patterns[connector] += 1

                            # If this pattern appears frequently, promote it
                            if self.candidate_patterns[connector] >= 3:
                                self._promote_candidate_pattern(connector)

    def _extract_connector(self, sentence: str, concept1: str, concept2: str) -> Optional[str]:
        """
        Extract the words connecting two concepts in a sentence

        Example: "dopamine increases motivation"
        Returns: "increases"
        """
        # Find positions of concepts
        lower_sent = sentence.lower()
        pos1 = lower_sent.find(concept1.lower())
        pos2 = lower_sent.find(concept2.lower())

        if pos1 == -1 or pos2 == -1:
            return None

        # Extract text between concepts
        start = min(pos1, pos2) + len(concept1)
        end = max(pos1, pos2)

        between = sentence[start:end].strip()

        # Extract verb/connector
        words = between.split()
        if 1 <= len(words) <= 5:  # Reasonable connector length
            return between.lower()

        return None

    def _promote_candidate_pattern(self, connector: str):
        """
        Promote a candidate pattern to a real pattern

        Args:
            connector: The connecting phrase (e.g., "increases", "modulates through")
        """
        pattern_id = f"learned_{connector.replace(' ', '_')}"

        if pattern_id in self.patterns:
            return  # Already exists

        # Create regex pattern
        regex = rf'(\w+(?:\s+\w+){{0,3}})\s+{re.escape(connector)}\s+(\w+(?:\s+\w+){{0,3}})'

        # Determine relation type from connector
        relation_type = self._infer_relation_type(connector)

        new_pattern = CausalPattern(
            pattern_id=pattern_id,
            regex=regex,
            relation_type=relation_type,
            confidence=0.5,  # Start with neutral confidence
            successes=self.candidate_patterns[connector],  # Bootstrap with discovery count
            failures=0
        )

        self.patterns[pattern_id] = new_pattern

        logger.info(f"🎓 Learned new pattern: '{connector}' (seen {self.candidate_patterns[connector]} times)")

        # Save patterns
        self._save_patterns()

    def _infer_relation_type(self, connector: str) -> str:
        """Infer relation type from connector phrase"""
        increases_words = ['increase', 'enhance', 'boost', 'promote', 'elevate', 'amplify']
        decreases_words = ['decrease', 'reduce', 'inhibit', 'suppress', 'lower', 'diminish']
        causes_words = ['cause', 'trigger', 'induce', 'produce', 'generate', 'lead to']

        connector_lower = connector.lower()

        if any(word in connector_lower for word in increases_words):
            return 'increases'
        elif any(word in connector_lower for word in decreases_words):
            return 'decreases'
        elif any(word in connector_lower for word in causes_words):
            return 'causes'
        else:
            return 'affects'

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple: extract noun phrases
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        concepts = [w for w in words if w not in stop_words and len(w) > 3]
        return concepts

    def _clean_entity(self, text: str) -> str:
        """Clean entity text"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^(the|a|an)\s+', '', text, flags=re.IGNORECASE)
        text = text.rstrip('.,;:')
        return text.lower()

    def get_top_patterns(self, n: int = 10) -> List[CausalPattern]:
        """Get best performing patterns"""
        return sorted(
            self.patterns.values(),
            key=lambda p: p.get_performance(),
            reverse=True
        )[:n]

    def get_learned_patterns(self) -> List[CausalPattern]:
        """Get patterns learned from feedback (not base patterns)"""
        return [
            p for p in self.patterns.values()
            if p.pattern_id.startswith('learned_')
        ]

    def get_stats(self) -> Dict[str, any]:
        """Get pattern learning statistics"""
        learned = self.get_learned_patterns()

        return {
            'total_patterns': len(self.patterns),
            'learned_patterns': len(learned),
            'candidate_patterns': len(self.candidate_patterns),
            'avg_pattern_confidence': sum(p.confidence for p in self.patterns.values()) / max(len(self.patterns), 1),
            'best_pattern': max(self.patterns.values(), key=lambda p: p.get_performance()).pattern_id if self.patterns else None
        }

    def _save_patterns(self):
        """Save learned patterns to disk"""
        patterns_data = {
            pid: {
                'regex': p.regex,
                'relation_type': p.relation_type,
                'successes': p.successes,
                'failures': p.failures,
                'confidence': p.confidence,
                'examples': p.examples
            }
            for pid, p in self.patterns.items()
            if pid.startswith('learned_')  # Only save learned patterns
        }

        with open(self.storage_path / "learned_patterns.json", 'w') as f:
            json.dump(patterns_data, f, indent=2)

    def _load_patterns(self):
        """Load saved patterns from disk"""
        filepath = self.storage_path / "learned_patterns.json"

        if not filepath.exists():
            return

        try:
            with open(filepath, 'r') as f:
                patterns_data = json.load(f)

            for pattern_id, data in patterns_data.items():
                pattern = CausalPattern(
                    pattern_id=pattern_id,
                    regex=data['regex'],
                    relation_type=data['relation_type'],
                    successes=data['successes'],
                    failures=data['failures'],
                    confidence=data['confidence'],
                    examples=data.get('examples', [])
                )
                self.patterns[pattern_id] = pattern

            logger.info(f"Loaded {len(patterns_data)} learned patterns from disk")

        except Exception as e:
            logger.error(f"Error loading patterns: {e}")

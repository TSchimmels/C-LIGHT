"""
Causal Relation Extractor for C-LIGHT
Extracts causal relationships from scientific text
"""

import re
from typing import List, Tuple
import logging

from ..core.base_types import CausalRelation, Paper, KnowledgeDomain

logger = logging.getLogger(__name__)


class CausalRelationExtractor:
    """
    Extracts causal relationships from paper text using pattern matching
    Can be extended with ML models later
    """

    # Causal patterns (ordered by strength)
    STRONG_CAUSAL_PATTERNS = [
        (r'(\w+(?:\s+\w+){0,3})\s+causes\s+(\w+(?:\s+\w+){0,3})', 'causes', 0.9),
        (r'(\w+(?:\s+\w+){0,3})\s+leads to\s+(\w+(?:\s+\w+){0,3})', 'causes', 0.85),
        (r'(\w+(?:\s+\w+){0,3})\s+results in\s+(\w+(?:\s+\w+){0,3})', 'causes', 0.8),
        (r'(\w+(?:\s+\w+){0,3})\s+induces\s+(\w+(?:\s+\w+){0,3})', 'causes', 0.8),
        (r'(\w+(?:\s+\w+){0,3})\s+triggers\s+(\w+(?:\s+\w+){0,3})', 'causes', 0.75),
    ]

    MODERATE_CAUSAL_PATTERNS = [
        (r'(\w+(?:\s+\w+){0,3})\s+increases\s+(\w+(?:\s+\w+){0,3})', 'increases', 0.7),
        (r'(\w+(?:\s+\w+){0,3})\s+decreases\s+(\w+(?:\s+\w+){0,3})', 'decreases', 0.7),
        (r'(\w+(?:\s+\w+){0,3})\s+enhances\s+(\w+(?:\s+\w+){0,3})', 'enhances', 0.65),
        (r'(\w+(?:\s+\w+){0,3})\s+reduces\s+(\w+(?:\s+\w+){0,3})', 'reduces', 0.65),
        (r'(\w+(?:\s+\w+){0,3})\s+modulates\s+(\w+(?:\s+\w+){0,3})', 'modulates', 0.6),
        (r'(\w+(?:\s+\w+){0,3})\s+affects\s+(\w+(?:\s+\w+){0,3})', 'affects', 0.5),
    ]

    WEAK_CAUSAL_PATTERNS = [
        (r'(\w+(?:\s+\w+){0,3})\s+is associated with\s+(\w+(?:\s+\w+){0,3})', 'correlates', 0.4),
        (r'(\w+(?:\s+\w+){0,3})\s+correlates with\s+(\w+(?:\s+\w+){0,3})', 'correlates', 0.35),
        (r'(\w+(?:\s+\w+){0,3})\s+relates to\s+(\w+(?:\s+\w+){0,3})', 'relates', 0.3),
    ]

    def __init__(self):
        self.all_patterns = (
            self.STRONG_CAUSAL_PATTERNS +
            self.MODERATE_CAUSAL_PATTERNS +
            self.WEAK_CAUSAL_PATTERNS
        )

    def extract(self, paper: Paper) -> List[CausalRelation]:
        """
        Extract causal relations from a paper

        Args:
            paper: Paper object with abstract (and optionally full_text)

        Returns:
            List of CausalRelation objects
        """
        relations = []

        # Extract from abstract (always available)
        if paper.abstract:
            relations.extend(self._extract_from_text(
                paper.abstract,
                paper.paper_id,
                paper.domains
            ))

        # Extract from full text if available
        if paper.full_text:
            relations.extend(self._extract_from_text(
                paper.full_text,
                paper.paper_id,
                paper.domains
            ))

        logger.info(f"Extracted {len(relations)} causal relations from {paper.paper_id}")

        return relations

    def _extract_from_text(
        self,
        text: str,
        paper_id: str,
        domains: List[KnowledgeDomain]
    ) -> List[CausalRelation]:
        """Extract relations from a text string"""
        relations = []

        # Split into sentences for better context
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Try each pattern
            for pattern, relation_type, confidence in self.all_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)

                for match in matches:
                    source = match.group(1).strip()
                    target = match.group(2).strip()

                    # Clean up entities
                    source = self._clean_entity(source)
                    target = self._clean_entity(target)

                    # Skip if too generic or empty
                    if not source or not target:
                        continue
                    if len(source) < 3 or len(target) < 3:
                        continue

                    # Create relation
                    relation = CausalRelation(
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        evidence_text=sentence,
                        paper_id=paper_id,
                        confidence=confidence,
                        domains=domains
                    )

                    relations.append(relation)

        return relations

    def _clean_entity(self, text: str) -> str:
        """Clean and normalize entity text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove leading articles
        text = re.sub(r'^(the|a|an)\s+', '', text, flags=re.IGNORECASE)

        # Remove trailing punctuation
        text = text.rstrip('.,;:')

        # Lowercase for consistency
        text = text.lower()

        return text

    def extract_with_study_metadata(
        self,
        paper: Paper,
        study_design: Optional[str] = None,
        sample_size: Optional[int] = None
    ) -> List[CausalRelation]:
        """
        Extract relations and add study metadata
        This increases the weight of relations from better studies
        """
        relations = self.extract(paper)

        # Add study metadata to each relation
        for relation in relations:
            relation.study_design = study_design
            relation.sample_size = sample_size

        return relations

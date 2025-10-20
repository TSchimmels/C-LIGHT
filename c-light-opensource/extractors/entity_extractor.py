"""
Entity Extractor for C-LIGHT
Extracts named entities from scientific text
"""

import re
from collections import defaultdict
from typing import List, Dict, Set
import logging

from ..core.base_types import Entity, Paper

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts entities from paper text
    Focuses on cognitive science relevant entities
    """

    # Entity patterns by type
    ENTITY_PATTERNS = {
        'neurotransmitter': [
            r'\b(dopamine|serotonin|norepinephrine|epinephrine|acetylcholine|GABA|glutamate|histamine|adenosine)\b',
        ],
        'brain_region': [
            r'\b(prefrontal cortex|amygdala|hippocampus|nucleus accumbens|VTA|ventral tegmental area|striatum|thalamus|hypothalamus|cerebellum|cortex)\b',
        ],
        'molecule': [
            r'\b([A-Z][a-z]*\d+[A-Za-z]*)\b',  # e.g., Nav1.6, KATP
            r'\b(ATP|ADP|cAMP|cGMP|NAD|NADH)\b',
        ],
        'gene': [
            r'\b([A-Z]{2,}[0-9]+[A-Z]*)\b',  # e.g., NR3C1, FKBP5
        ],
        'drug': [
            r'\b(caffeine|alcohol|ethanol|nicotine|THC|CBD|SSRI|benzodiazepine|statin|metformin)\b',
        ],
        'metabolite': [
            r'\b(glucose|ketones?|beta-hydroxybutyrate|lactate|pyruvate|acetyl-CoA)\b',
        ],
        'ion': [
            r'\b(sodium|potassium|calcium|magnesium|chloride|Na\+|K\+|Ca2\+|Mg2\+)\b',
        ],
        'protein': [
            r'\b(receptor|channel|enzyme|kinase|phosphatase|transporter|pump)\b',
        ],
        'psychological_construct': [
            r'\b(attention|memory|cognition|consciousness|emotion|mood|anxiety|depression|stress|personality|motivation|reward)\b',
        ],
        'intervention': [
            r'\b(fasting|diet|exercise|meditation|therapy|training|supplementation)\b',
        ],
        'outcome': [
            r'\b(performance|improvement|decline|enhancement|impairment|change)\b',
        ],
    }

    def __init__(self):
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def extract(self, paper: Paper) -> List[Entity]:
        """
        Extract entities from a paper

        Args:
            paper: Paper object

        Returns:
            List of Entity objects
        """
        # Count entity mentions by type and name
        entity_mentions: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        # Extract from abstract
        if paper.abstract:
            self._extract_from_text(paper.abstract, paper.paper_id, entity_mentions)

        # Extract from full text if available
        if paper.full_text:
            self._extract_from_text(paper.full_text, paper.paper_id, entity_mentions)

        # Convert to Entity objects
        entities = []
        for entity_type, mentions in entity_mentions.items():
            for name, paper_ids in mentions.items():
                entity = Entity(
                    name=name,
                    entity_type=entity_type,
                    mentions=len(paper_ids),
                    papers=paper_ids
                )
                entities.append(entity)

        logger.info(f"Extracted {len(entities)} entities from {paper.paper_id}")

        return entities

    def _extract_from_text(
        self,
        text: str,
        paper_id: str,
        entity_mentions: Dict[str, Dict[str, Set[str]]]
    ):
        """Extract entities from text and update mentions dict"""
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    name = match.group(0).lower()
                    # Clean and normalize
                    name = self._normalize_entity(name)
                    if name:
                        entity_mentions[entity_type][name].add(paper_id)

    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove trailing punctuation
        text = text.rstrip('.,;:')

        if len(text) < 2:
            return ""

        return text

    def extract_domain_specific(
        self,
        paper: Paper,
        entity_types: List[str]
    ) -> List[Entity]:
        """
        Extract only specific entity types
        Useful for domain-focused extraction
        """
        # Temporarily filter patterns
        original_patterns = self.compiled_patterns
        self.compiled_patterns = {
            k: v for k, v in original_patterns.items()
            if k in entity_types
        }

        entities = self.extract(paper)

        # Restore patterns
        self.compiled_patterns = original_patterns

        return entities


class TemporalEntityExtractor(EntityExtractor):
    """
    Extended entity extractor for temporal pharmacology
    Extracts timing information about substances
    """

    TEMPORAL_PATTERNS = {
        'onset': r'onset.*?(\d+(?:\.\d+)?)\s*(min|minutes|hr|hours|h)',
        'peak': r'peak.*?(\d+(?:\.\d+)?)\s*(min|minutes|hr|hours|h)',
        'half_life': r'half[- ]life.*?(\d+(?:\.\d+)?)\s*(min|minutes|hr|hours|h)',
        'duration': r'duration.*?(\d+(?:\.\d+)?)\s*(min|minutes|hr|hours|h)',
    }

    def extract_temporal_info(self, paper: Paper) -> Dict[str, Dict[str, str]]:
        """
        Extract temporal information about substances

        Returns:
            Dict mapping substance names to temporal properties
        """
        temporal_info = defaultdict(dict)

        text = (paper.abstract or "") + " " + (paper.full_text or "")

        for property_name, pattern in self.TEMPORAL_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                unit = match.group(2)
                # Store as string for now, can be parsed later
                temporal_info[property_name] = f"{value} {unit}"

        return dict(temporal_info)

"""
Base types and data structures for C-LIGHT
Completely standalone - no external dependencies required
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Set, Any


class KnowledgeDomain(Enum):
    """Research domains covered by C-LIGHT"""
    # Core Sciences
    NEUROSCIENCE = "neuroscience"
    PSYCHOLOGY = "psychology"
    COGNITIVE_SCIENCE = "cognitive_science"

    # Molecular & Cellular
    MOLECULAR_BIOLOGY = "molecular_biology"
    PHARMACOLOGY = "pharmacology"
    EPIGENETICS = "epigenetics"

    # Brain & Development
    BRAIN_ORGANOIDS = "brain_organoids"
    TISSUE_ENGINEERING = "tissue_engineering"
    DEVELOPMENTAL_BIOLOGY = "developmental_biology"

    # Metabolism & Nutrition
    METABOLIC_COGNITION = "metabolic_cognition"
    NUTRITIONAL_NEUROSCIENCE = "nutritional_neuroscience"
    EXERCISE_PHYSIOLOGY = "exercise_physiology"

    # Behavioral & Social
    BEHAVIORAL_NEUROSCIENCE = "behavioral_neuroscience"
    SOCIAL_PSYCHOLOGY = "social_psychology"
    POLITICAL_PSYCHOLOGY = "political_psychology"
    CULTURAL_NEUROSCIENCE = "cultural_neuroscience"

    # Digital & Manipulation
    DIGITAL_PSYCHOLOGY = "digital_psychology"
    SOCIAL_MEDIA_EFFECTS = "social_media_effects"
    PERSUASIVE_TECHNOLOGY = "persuasive_technology"

    # Advanced Topics
    QUANTUM_BIOLOGY = "quantum_biology"
    CONSCIOUSNESS_STUDIES = "consciousness_studies"
    EMF_BIOLOGY = "emf_biology"

    # Clinical
    SLEEP_SCIENCE = "sleep_science"
    MICROBIOME = "microbiome"


class ProcessingStatus(Enum):
    """Paper processing status"""
    DOWNLOADED = "downloaded"
    EXTRACTED = "extracted"
    EMBEDDED = "embedded"
    IN_GRAPH = "in_graph"
    FAILED = "failed"


@dataclass
class Paper:
    """Structured representation of a scientific paper"""
    # Identifiers
    paper_id: str  # DOI or ArXiv ID
    title: str
    authors: List[str]

    # Content
    abstract: str
    full_text: Optional[str] = None

    # Metadata
    categories: List[str] = field(default_factory=list)
    domains: List[KnowledgeDomain] = field(default_factory=list)
    published_date: Optional[datetime] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    # Processing
    status: ProcessingStatus = ProcessingStatus.DOWNLOADED
    embedding: Optional[List[float]] = None

    # Source
    pdf_url: Optional[str] = None
    source: str = "arxiv"  # arxiv, pubmed, biorxiv, etc.

    # Citations
    citation_count: int = 0
    references: List[str] = field(default_factory=list)

    # Quality metrics
    study_design: Optional[str] = None  # RCT, observational, meta-analysis
    sample_size: Optional[int] = None

    def __hash__(self):
        return hash(self.paper_id)

    def __eq__(self, other):
        if not isinstance(other, Paper):
            return False
        return self.paper_id == other.paper_id


@dataclass
class Entity:
    """Extracted entity from papers"""
    name: str
    entity_type: str  # gene, protein, drug, concept, intervention, etc.
    mentions: int = 0
    papers: Set[str] = field(default_factory=set)
    aliases: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.name, self.entity_type))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name == other.name and self.entity_type == other.entity_type


@dataclass
class CausalRelation:
    """Extracted causal relationship"""
    source: str  # Entity or concept name
    target: str  # Entity or concept name
    relation_type: str  # causes, increases, decreases, modulates, etc.

    # Evidence
    evidence_text: str
    paper_id: str
    confidence: float  # 0.0 to 1.0

    # Context
    domains: List[KnowledgeDomain] = field(default_factory=list)

    # Supporting evidence
    study_design: Optional[str] = None
    sample_size: Optional[int] = None
    effect_size: Optional[float] = None
    p_value: Optional[float] = None

    def __hash__(self):
        return hash((self.source, self.target, self.relation_type, self.paper_id))


@dataclass
class BehavioralIndicator:
    """Behavioral indicator or outcome extracted from papers"""
    indicator: str
    indicator_type: str  # risk_factor, protective_factor, intervention, outcome
    domain: KnowledgeDomain
    evidence_text: str
    paper_id: str
    confidence: float


@dataclass
class ExtractionResult:
    """Results from processing a paper"""
    paper_id: str

    # Extracted information
    entities: List[Entity] = field(default_factory=list)
    causal_relations: List[CausalRelation] = field(default_factory=list)
    behavioral_indicators: List[BehavioralIndicator] = field(default_factory=list)

    # Classification
    domains: List[KnowledgeDomain] = field(default_factory=list)

    # Metadata
    extraction_time: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None

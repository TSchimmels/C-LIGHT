"""
C-LIGHT: Cognitive, Life-science, Intelligence Gathering & Hypothesis Testing

An open-source RAG system for behavioral and cognitive science research.

Basic Usage:
    from c_light_opensource import CLightSystem

    # Initialize system
    clight = CLightSystem()

    # Harvest papers
    clight.harvest(['cs.AI', 'q-bio.NC'], max_papers=100)

    # Query knowledge base
    result = clight.query("How does dopamine affect motivation?")
    print(result.answer)

For CANDLE Integration:
    from c_light_opensource.adapters import create_candle_adapter

    adapter = create_candle_adapter(
        storage_path="/candle/data",
        candle_config=your_config
    )

    result = adapter.query("Your question")
"""

__version__ = "0.1.0"

from .core.base_types import (
    Paper,
    KnowledgeDomain,
    CausalRelation,
    Entity,
    ProcessingStatus
)

from .core.knowledge_graph import (
    CLightKnowledgeGraph,
    Node,
    Edge,
    NodeType,
    EdgeType
)

from .core.rag_system import (
    CLightRAG,
    QueryResult,
    Evidence
)

from .harvesting import (
    ArxivHarvester,
    ArxivCategory,
    DOIPaperDatabase
)

from .extractors import (
    CausalRelationExtractor,
    EntityExtractor,
    TemporalEntityExtractor
)

from .adapters import (
    CANDLEAdapter,
    create_candle_adapter
)


class CLightSystem:
    """
    Main C-LIGHT system class - simplified API for common operations
    """

    def __init__(
        self,
        storage_path: str = "/mnt/hdd/c-light",
        db_path: str = "/mnt/nvme/c-light/db",
        graph_path: str = "/mnt/nvme/c-light/graph"
    ):
        self.harvester = ArxivHarvester(storage_path + "/papers", db_path)
        self.kg = CLightKnowledgeGraph(graph_path)
        self.causal_extractor = CausalRelationExtractor()
        self.entity_extractor = EntityExtractor()
        self.rag = CLightRAG(self.kg, self.kg.paper_index)

    def harvest(
        self,
        categories: list,
        max_papers: int = 1000,
        keywords: list = None
    ):
        """Harvest papers from ArXiv"""
        papers = self.harvester.harvest(
            categories=categories,
            max_papers=max_papers,
            keywords=keywords
        )

        # Process papers
        for paper in papers:
            self._process_paper(paper)

        return papers

    def _process_paper(self, paper: Paper):
        """Process a single paper"""
        # Add to graph
        self.kg.add_paper(paper)

        # Extract knowledge
        relations = self.causal_extractor.extract(paper)
        for relation in relations:
            self.kg.add_causal_relation(relation)

        entities = self.entity_extractor.extract(paper)
        for entity in entities:
            self.kg.add_entity(entity)

    def query(self, question: str, domains: list = None):
        """Query the knowledge base"""
        return self.rag.query(question, domains=domains)

    def find_path(self, source: str, target: str):
        """Find causal path between concepts"""
        return self.kg.find_causal_path(source, target)

    def get_influences(self, concept: str):
        """Get what influences a concept"""
        return self.kg.query_influences(concept)

    def save(self):
        """Save state to disk"""
        self.kg.save_to_disk()

    def load(self):
        """Load state from disk"""
        self.kg.load_from_disk()


__all__ = [
    # Version
    '__version__',

    # Main system
    'CLightSystem',

    # Base types
    'Paper',
    'KnowledgeDomain',
    'CausalRelation',
    'Entity',
    'ProcessingStatus',

    # Knowledge graph
    'CLightKnowledgeGraph',
    'Node',
    'Edge',
    'NodeType',
    'EdgeType',

    # RAG
    'CLightRAG',
    'QueryResult',
    'Evidence',

    # Harvesting
    'ArxivHarvester',
    'ArxivCategory',
    'DOIPaperDatabase',

    # Extractors
    'CausalRelationExtractor',
    'EntityExtractor',
    'TemporalEntityExtractor',

    # Adapters
    'CANDLEAdapter',
    'create_candle_adapter',
]

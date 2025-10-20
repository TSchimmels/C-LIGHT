"""
CANDLE Integration Adapter for C-LIGHT
Provides clean integration points for the proprietary CANDLE system
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from ..core.base_types import Paper, CausalRelation, Entity, KnowledgeDomain
from ..core.knowledge_graph import CLightKnowledgeGraph
from ..core.rag_system import CLightRAG, QueryResult
from ..harvesting.arxiv_harvester import ArxivHarvester
from ..extractors.causal_extractor import CausalRelationExtractor
from ..extractors.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


class CANDLEAdapter:
    """
    Adapter for integrating C-LIGHT with CANDLE

    This adapter provides a clean interface between the open-source C-LIGHT
    and your proprietary CANDLE system. It allows CANDLE to:

    1. Access C-LIGHT's knowledge graph
    2. Query the RAG system
    3. Add custom papers and knowledge
    4. Override storage paths
    5. Hook into the processing pipeline

    Usage in CANDLE:
        from c_light_opensource.adapters import CANDLEAdapter

        adapter = CANDLEAdapter(
            storage_path="/candle/storage/c-light",
            db_path="/candle/db/c-light"
        )

        # Use C-LIGHT's capabilities
        result = adapter.query("How does dopamine affect motivation?")

        # Add CANDLE-specific papers
        adapter.add_paper(candle_paper)

        # Access knowledge graph for CANDLE's reasoning engine
        kg = adapter.get_knowledge_graph()
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        db_path: Optional[str] = None,
        graph_path: Optional[str] = None,
        candle_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize adapter with custom paths for CANDLE integration

        Args:
            storage_path: Where to store downloaded papers (override default)
            db_path: Where to store DOI database (override default)
            graph_path: Where to store knowledge graph (override default)
            candle_config: CANDLE-specific configuration
        """
        self.storage_path = storage_path or "/mnt/hdd/c-light/raw_papers"
        self.db_path = db_path or "/mnt/nvme/c-light/doi_database"
        self.graph_path = graph_path or "/mnt/nvme/c-light/knowledge_graph"
        self.candle_config = candle_config or {}

        # Initialize C-LIGHT components
        self.harvester = ArxivHarvester(
            storage_path=self.storage_path,
            db_path=self.db_path
        )

        self.knowledge_graph = CLightKnowledgeGraph(graph_path=self.graph_path)

        self.causal_extractor = CausalRelationExtractor()
        self.entity_extractor = EntityExtractor()

        self.rag = CLightRAG(
            knowledge_graph=self.knowledge_graph,
            paper_index=self.knowledge_graph.paper_index
        )

        logger.info(f"CANDLE adapter initialized with paths: "
                   f"storage={self.storage_path}, db={self.db_path}, "
                   f"graph={self.graph_path}")

    # =========================================================================
    # Public API for CANDLE
    # =========================================================================

    def query(
        self,
        question: str,
        domains: Optional[List[KnowledgeDomain]] = None,
        max_results: int = 10
    ) -> QueryResult:
        """
        Query C-LIGHT's knowledge base

        This is the main entry point for CANDLE to ask questions and get
        answers backed by scientific evidence.

        Args:
            question: Natural language question
            domains: Optional domain filter
            max_results: Maximum evidence pieces to return

        Returns:
            QueryResult with answer, evidence, and confidence
        """
        return self.rag.query(
            question=question,
            max_results=max_results,
            domains=domains
        )

    def add_paper(self, paper: Paper) -> bool:
        """
        Add a paper to C-LIGHT's knowledge base

        CANDLE can use this to add papers from its own sources
        (e.g., proprietary databases, classified research)

        Args:
            paper: Paper object to add

        Returns:
            True if successfully added
        """
        try:
            # Add to knowledge graph
            self.knowledge_graph.add_paper(paper)

            # Extract and add knowledge
            relations = self.causal_extractor.extract(paper)
            for relation in relations:
                self.knowledge_graph.add_causal_relation(relation)

            entities = self.entity_extractor.extract(paper)
            for entity in entities:
                self.knowledge_graph.add_entity(entity)

            # Update paper index
            self.knowledge_graph.paper_index[paper.paper_id] = paper

            logger.info(f"Added paper {paper.paper_id} to C-LIGHT via CANDLE adapter")
            return True

        except Exception as e:
            logger.error(f"Error adding paper {paper.paper_id}: {e}")
            return False

    def add_causal_relation(self, relation: CausalRelation) -> bool:
        """
        Add a causal relation to the knowledge graph

        CANDLE can use this to add relationships discovered through
        its proprietary reasoning engines

        Args:
            relation: CausalRelation object

        Returns:
            True if successfully added
        """
        try:
            self.knowledge_graph.add_causal_relation(relation)
            return True
        except Exception as e:
            logger.error(f"Error adding causal relation: {e}")
            return False

    def get_knowledge_graph(self) -> CLightKnowledgeGraph:
        """
        Get direct access to the knowledge graph

        CANDLE can use this for advanced graph operations,
        custom algorithms, or integration with proprietary reasoning

        Returns:
            CLightKnowledgeGraph instance
        """
        return self.knowledge_graph

    def find_causal_path(
        self,
        source: str,
        target: str,
        max_hops: int = 5
    ) -> List[List[str]]:
        """
        Find causal pathways between concepts

        Args:
            source: Source concept
            target: Target concept
            max_hops: Maximum path length

        Returns:
            List of paths (each path is a list of concept names)
        """
        return self.knowledge_graph.find_causal_path(source, target, max_hops)

    def query_influences(
        self,
        concept: str,
        domains: Optional[List[KnowledgeDomain]] = None
    ) -> Dict[str, Any]:
        """
        Find what influences a concept

        Args:
            concept: Concept to query
            domains: Optional domain filter

        Returns:
            Dictionary with influences and evidence
        """
        return self.knowledge_graph.query_influences(concept, domains=domains)

    def cross_domain_effects(
        self,
        source_domains: List[KnowledgeDomain],
        target_domains: List[KnowledgeDomain]
    ) -> List[Dict[str, Any]]:
        """
        Find causal relationships crossing domain boundaries

        This is particularly useful for discovering upstream factors
        (e.g., molecular → behavioral, social → neural)

        Args:
            source_domains: Source domains
            target_domains: Target domains

        Returns:
            List of cross-domain relationships
        """
        return self.knowledge_graph.cross_domain_effects(
            source_domains,
            target_domains
        )

    def harvest_papers(
        self,
        categories: List[str],
        max_papers: int = 1000,
        days_back: int = 7,
        keywords: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Harvest papers from ArXiv

        CANDLE can trigger harvesting on-demand

        Args:
            categories: ArXiv categories
            max_papers: Max papers to download
            days_back: How far back to search
            keywords: Optional keyword filter

        Returns:
            List of harvested papers
        """
        return self.harvester.harvest(
            categories=categories,
            max_papers=max_papers,
            days_back=days_back,
            keywords=keywords
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about C-LIGHT's knowledge base

        Returns:
            Dictionary with stats
        """
        return {
            'knowledge_graph': self.knowledge_graph.get_stats(),
            'harvester': self.harvester.get_stats()
        }

    def save_state(self):
        """
        Save C-LIGHT's state to disk

        CANDLE should call this periodically to persist knowledge
        """
        self.knowledge_graph.save_to_disk()
        logger.info("C-LIGHT state saved to disk")

    def load_state(self):
        """
        Load C-LIGHT's state from disk

        CANDLE should call this on startup
        """
        self.knowledge_graph.load_from_disk()
        logger.info("C-LIGHT state loaded from disk")

    # =========================================================================
    # Advanced Integration Points for CANDLE
    # =========================================================================

    def register_custom_extractor(self, extractor_name: str, extractor_func):
        """
        Register a custom extraction function

        CANDLE can add proprietary extractors for domain-specific knowledge

        Args:
            extractor_name: Name for the extractor
            extractor_func: Function that takes a Paper and returns extractions
        """
        if not hasattr(self, 'custom_extractors'):
            self.custom_extractors = {}

        self.custom_extractors[extractor_name] = extractor_func
        logger.info(f"Registered custom extractor: {extractor_name}")

    def process_with_custom_extractors(self, paper: Paper) -> Dict[str, Any]:
        """
        Process a paper with all registered custom extractors

        Args:
            paper: Paper to process

        Returns:
            Dictionary mapping extractor names to results
        """
        if not hasattr(self, 'custom_extractors'):
            return {}

        results = {}
        for name, func in self.custom_extractors.items():
            try:
                results[name] = func(paper)
            except Exception as e:
                logger.error(f"Error in custom extractor {name}: {e}")
                results[name] = None

        return results

    def export_for_candle(self, format: str = 'json') -> Any:
        """
        Export C-LIGHT knowledge in a format optimized for CANDLE

        Args:
            format: Export format ('json', 'pickle', 'networkx')

        Returns:
            Exported data
        """
        return self.knowledge_graph.export_knowledge_graph(format)

    def sync_from_candle(self, candle_knowledge: Dict[str, Any]):
        """
        Import knowledge from CANDLE's proprietary systems

        This allows bidirectional knowledge flow between C-LIGHT and CANDLE

        Args:
            candle_knowledge: Knowledge from CANDLE in agreed-upon format
        """
        # Implementation depends on CANDLE's data format
        # This is a placeholder that can be customized
        logger.info("Syncing knowledge from CANDLE (implement based on your format)")
        pass


def create_candle_adapter(
    storage_path: Optional[str] = None,
    db_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    candle_config: Optional[Dict[str, Any]] = None
) -> CANDLEAdapter:
    """
    Factory function to create CANDLE adapter

    This is the main entry point from CANDLE code:
        from c_light_opensource.adapters import create_candle_adapter

        adapter = create_candle_adapter(
            storage_path="/candle/data/c-light",
            candle_config=candle_config
        )
    """
    return CANDLEAdapter(
        storage_path=storage_path,
        db_path=db_path,
        graph_path=graph_path,
        candle_config=candle_config
    )

#!/usr/bin/env python3
"""
CANDLE Integration Example
Shows how to integrate C-LIGHT (both System A and B) with CANDLE

This demonstrates:
1. Using the CANDLE adapter
2. Integrating both RAG systems with CANDLE workflow
3. Accessing knowledge graph from CANDLE
4. Custom paper processing from CANDLE
5. Query routing between systems
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from core.base_types import Paper, QueryResult, KnowledgeDomain
from core.knowledge_graph import CLightKnowledgeGraph
from adapters.candle_adapter import CANDLEAdapter
from system_a.llm_rag import MixtralRAG
from system_b.seal_rag import SEALBasedRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CANDLEIntegratedSystem:
    """
    Wrapper that integrates C-LIGHT with CANDLE

    This shows how CANDLE can use both RAG systems while maintaining
    its own proprietary algorithms
    """

    def __init__(
        self,
        storage_path: str,
        db_path: str,
        graph_path: str,
        candle_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize integrated system

        Args:
            storage_path: Where to store papers
            db_path: DOI database path
            graph_path: Knowledge graph path
            candle_config: Optional CANDLE-specific configuration
        """
        logger.info("Initializing CANDLE-integrated C-LIGHT system...")

        # Basic adapter (for harvesting and graph access)
        self.adapter = CANDLEAdapter(
            storage_path=storage_path,
            db_path=db_path,
            graph_path=graph_path,
            candle_config=candle_config or {}
        )

        # Initialize both RAG systems
        self.system_a = MixtralRAG(
            knowledge_graph=self.adapter.knowledge_graph,
            storage_path=str(Path(storage_path) / "system_a")
        )

        self.system_b = SEALBasedRAG(
            knowledge_graph=self.adapter.knowledge_graph,
            storage_path=str(Path(storage_path) / "system_b")
        )

        # Default system preference
        self.default_system = "seal"  # or "mixtral"

        logger.info("✓ Integrated system initialized")

    def query(
        self,
        question: str,
        system: Optional[str] = None,
        domains: Optional[list] = None
    ) -> QueryResult:
        """
        Query with automatic system selection

        Args:
            question: Natural language question
            system: "mixtral", "seal", or None (auto-select)
            domains: Optional domain filter

        Returns:
            QueryResult from selected system
        """
        # Auto-select system if not specified
        if system is None:
            system = self._auto_select_system(question)

        logger.info(f"Routing query to System {system.upper()}")

        if system == "mixtral":
            return self.system_a.query(question, domains)
        elif system == "seal":
            return self.system_b.query(question, domains)
        else:
            raise ValueError(f"Unknown system: {system}")

    def _auto_select_system(self, question: str) -> str:
        """
        Automatically select which system to use

        Strategy:
        - Use SEAL for causal queries (it's built for this)
        - Use Mixtral for general questions
        - Use SEAL if it has high success rate
        """
        # Check if causal query
        causal_keywords = ['cause', 'lead to', 'result in', 'affect', 'influence']
        is_causal = any(kw in question.lower() for kw in causal_keywords)

        if is_causal:
            # SEAL is better for causal reasoning
            return "seal"

        # Check SEAL performance
        seal_stats = self.system_b.get_stats()
        if seal_stats['feedback']['total_feedback'] > 50:
            # SEAL has enough feedback
            if seal_stats['feedback'].get('avg_rating', 0) >= 4.0:
                # SEAL is performing well
                return "seal"

        # Default to Mixtral
        return "mixtral"

    def add_paper_from_candle(
        self,
        paper: Paper,
        pdf_path: Optional[str] = None,
        process_with: str = "both"
    ):
        """
        Add a paper from CANDLE's proprietary sources

        Args:
            paper: Paper object (can come from CANDLE's own harvesting)
            pdf_path: Optional PDF path
            process_with: "system_a", "system_b", or "both"
        """
        logger.info(f"Adding CANDLE paper: {paper.title[:60]}...")

        # Add to knowledge graph (shared)
        self.adapter.add_paper(paper)

        # Process with selected systems
        if process_with in ["system_a", "both"]:
            self.system_a.add_paper(paper, pdf_path)
            logger.info("  ✓ Added to System A")

        if process_with in ["system_b", "both"]:
            self.system_b.add_paper(paper, pdf_path)
            logger.info("  ✓ Added to System B")

    def provide_feedback_from_candle(
        self,
        query_id: str,
        rating: int,
        correct_answer: Optional[str] = None
    ):
        """
        Provide feedback from CANDLE users

        This allows SEAL to learn from CANDLE user interactions

        Args:
            query_id: Query to rate
            rating: 1-5 rating
            correct_answer: Optional correction
        """
        # Only SEAL learns from feedback
        self.system_b.provide_feedback(query_id, rating, correct_answer)
        logger.info(f"✓ SEAL learned from CANDLE user feedback")

    def get_knowledge_graph_for_candle(self) -> CLightKnowledgeGraph:
        """
        Expose knowledge graph to CANDLE

        CANDLE can use this for its own algorithms:
        - Graph traversal
        - Path finding
        - Community detection
        - Custom analytics

        Returns:
            Knowledge graph instance
        """
        return self.adapter.get_knowledge_graph()

    def run_candle_algorithm_on_graph(self, algorithm_name: str):
        """
        Example: Run CANDLE's proprietary algorithm on the knowledge graph

        This shows how CANDLE can use C-LIGHT's graph for its own purposes
        """
        kg = self.get_knowledge_graph_for_candle()

        logger.info(f"Running CANDLE algorithm: {algorithm_name}")
        logger.info(f"Graph size: {len(kg.graph.nodes)} nodes, {len(kg.graph.edges)} edges")

        # CANDLE would implement its proprietary algorithms here
        # For example:
        # - Advanced causal inference
        # - Multi-hop reasoning
        # - Custom graph embeddings
        # - Proprietary scoring functions

        logger.info("(CANDLE algorithm would run here)")

        return {
            'algorithm': algorithm_name,
            'nodes': len(kg.graph.nodes),
            'edges': len(kg.graph.edges)
        }

    def compare_systems_for_candle(self, question: str) -> Dict[str, QueryResult]:
        """
        Compare both systems for CANDLE evaluation

        CANDLE can use this to decide which system to prefer

        Returns:
            Dict with results from both systems
        """
        logger.info("Running comparison for CANDLE...")

        result_a = self.system_a.query(question)
        result_b = self.system_b.query(question)

        return {
            'mixtral': result_a,
            'seal': result_b,
            'recommendation': self._recommend_for_candle(result_a, result_b)
        }

    def _recommend_for_candle(
        self,
        result_a: QueryResult,
        result_b: QueryResult
    ) -> str:
        """Recommend which result CANDLE should use"""
        # Simple heuristic: prefer higher confidence
        if result_a.confidence > result_b.confidence + 0.2:
            return "mixtral"
        elif result_b.confidence > result_a.confidence + 0.1:
            return "seal"
        else:
            return "either"


def demonstrate_candle_integration():
    """Demonstrate CANDLE integration"""

    logger.info("=" * 70)
    logger.info("CANDLE INTEGRATION DEMONSTRATION")
    logger.info("=" * 70)

    # Setup paths
    base_path = Path("/mnt/nvme/candle-integrated")
    base_path.mkdir(parents=True, exist_ok=True)

    storage_path = base_path / "papers"
    db_path = base_path / "doi_database.db"
    graph_path = base_path / "knowledge_graph"

    # Initialize integrated system
    logger.info("\n📦 Initializing CANDLE-integrated system...")

    system = CANDLEIntegratedSystem(
        storage_path=str(storage_path),
        db_path=str(db_path),
        graph_path=str(graph_path),
        candle_config={
            'project': 'CANDLE',
            'classification_level': 'UNCLASSIFIED',
            'custom_setting': 'value'
        }
    )

    # ========================================
    # USE CASE 1: Add paper from CANDLE
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("USE CASE 1: Adding Paper from CANDLE's Sources")
    logger.info("=" * 70)

    # Simulate CANDLE having its own paper sources
    candle_paper = Paper(
        paper_id="candle:classified:001",
        title="Proprietary CANDLE Research on Neural Mechanisms",
        authors=["CANDLE Researcher A"],
        abstract="This paper from CANDLE's internal research discusses...",
        domains=[KnowledgeDomain.NEUROSCIENCE]
    )

    logger.info(f"\n📄 CANDLE paper: {candle_paper.title}")
    logger.info("Adding to both RAG systems...")

    system.add_paper_from_candle(
        paper=candle_paper,
        pdf_path=None,
        process_with="both"
    )

    logger.info("✓ Paper integrated into C-LIGHT systems")

    # ========================================
    # USE CASE 2: Query with auto-routing
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("USE CASE 2: Querying with Automatic System Selection")
    logger.info("=" * 70)

    questions = [
        "What causes neural plasticity?",  # Causal → SEAL
        "Explain the concept of reward learning",  # General → Mixtral
    ]

    for question in questions:
        logger.info(f"\n❓ {question}")

        result = system.query(question)  # Auto-selects system

        logger.info(f"   System used: {result.metadata.get('system', 'unknown')}")
        logger.info(f"   Answer: {result.answer[:100]}...")
        logger.info(f"   Confidence: {result.confidence:.2f}")

    # ========================================
    # USE CASE 3: CANDLE provides feedback
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("USE CASE 3: CANDLE Users Provide Feedback")
    logger.info("=" * 70)

    logger.info("\nSimulating CANDLE user rating a query...")

    result = system.query("How does dopamine affect behavior?", system="seal")
    query_id = result.metadata['query_id']

    logger.info(f"Query ID: {query_id}")
    logger.info(f"Answer: {result.answer[:100]}...")

    # CANDLE user rates the answer
    candle_user_rating = 5
    logger.info(f"\nCANDLE user rating: {candle_user_rating}/5")

    system.provide_feedback_from_candle(
        query_id=query_id,
        rating=candle_user_rating
    )

    logger.info("✓ SEAL learned from CANDLE user feedback")

    # ========================================
    # USE CASE 4: CANDLE uses knowledge graph
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("USE CASE 4: CANDLE Runs Proprietary Algorithm on Graph")
    logger.info("=" * 70)

    logger.info("\nCANDLE accessing knowledge graph...")

    kg = system.get_knowledge_graph_for_candle()

    logger.info(f"Knowledge graph stats:")
    stats = kg.get_stats()
    logger.info(f"  Papers: {stats['paper_count']}")
    logger.info(f"  Relations: {stats['relation_count']}")
    logger.info(f"  Entities: {stats['entity_count']}")

    # CANDLE runs its algorithm
    result = system.run_candle_algorithm_on_graph("candle_advanced_inference")
    logger.info(f"\nCANDLE algorithm result: {result}")

    # ========================================
    # USE CASE 5: Compare systems
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("USE CASE 5: CANDLE Compares Both Systems")
    logger.info("=" * 70)

    question = "Does reward processing influence learning?"
    logger.info(f"\n❓ {question}")

    comparison = system.compare_systems_for_candle(question)

    logger.info("\nSystem A (Mixtral):")
    logger.info(f"  Answer: {comparison['mixtral'].answer[:100]}...")
    logger.info(f"  Confidence: {comparison['mixtral'].confidence:.2f}")

    logger.info("\nSystem B (SEAL):")
    logger.info(f"  Answer: {comparison['seal'].answer[:100]}...")
    logger.info(f"  Confidence: {comparison['seal'].confidence:.2f}")

    logger.info(f"\nRecommendation for CANDLE: Use {comparison['recommendation']}")

    # ========================================
    # SUMMARY
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRATION SUMMARY")
    logger.info("=" * 70)

    logger.info("\n✅ Integration Capabilities:")
    logger.info("1. ✓ CANDLE can add papers from proprietary sources")
    logger.info("2. ✓ Both RAG systems process CANDLE papers")
    logger.info("3. ✓ Auto-routing selects best system for query")
    logger.info("4. ✓ SEAL learns from CANDLE user feedback")
    logger.info("5. ✓ CANDLE can access knowledge graph directly")
    logger.info("6. ✓ CANDLE can compare both systems")

    logger.info("\n🔒 Security:")
    logger.info("- C-LIGHT code is open source")
    logger.info("- CANDLE algorithms remain proprietary")
    logger.info("- Classified papers stay in CANDLE's control")
    logger.info("- Knowledge graph can be filtered for classification")

    logger.info("\n💡 Best Practices:")
    logger.info("- Use System A (Mixtral) for immediate high quality")
    logger.info("- Use System B (SEAL) for causal reasoning")
    logger.info("- Provide feedback to SEAL for continuous improvement")
    logger.info("- Compare systems periodically to track progress")
    logger.info("- Use knowledge graph for CANDLE's own algorithms")

    return system


if __name__ == "__main__":
    integrated_system = demonstrate_candle_integration()

    print("\n" + "=" * 70)
    print("CANDLE Integration Ready!")
    print("=" * 70)
    print("\nExample CANDLE usage:")
    print(">>> # Query with auto-routing")
    print(">>> result = integrated_system.query('Your question')")
    print(">>>")
    print(">>> # Add CANDLE's own papers")
    print(">>> integrated_system.add_paper_from_candle(candle_paper)")
    print(">>>")
    print(">>> # Access graph for CANDLE algorithms")
    print(">>> kg = integrated_system.get_knowledge_graph_for_candle()")
    print(">>> # Run CANDLE's proprietary algorithms on kg")

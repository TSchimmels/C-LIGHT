#!/usr/bin/env python3
"""
Complete C-LIGHT Workflow Example
Demonstrates the full pipeline from harvesting to querying to learning

This script shows:
1. Harvesting papers from ArXiv
2. Processing papers with both System A and System B
3. Querying both systems
4. Comparing results
5. Providing feedback for learning
6. Analyzing improvement over time
"""

import logging
from pathlib import Path

# Core
from core.base_types import Paper, KnowledgeDomain
from core.knowledge_graph import CLightKnowledgeGraph

# Harvesting
from harvesting.arxiv_harvester import ArxivHarvester

# System A (Mixtral)
from system_a.llm_rag import MixtralRAG

# System B (SEAL)
from system_b.seal_rag import SEALBasedRAG

# Evaluation
from evaluation.dual_system_evaluator import DualSystemEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete workflow"""

    # ===========================================
    # STEP 1: Setup Paths
    # ===========================================
    logger.info("=" * 60)
    logger.info("STEP 1: Setting up paths")
    logger.info("=" * 60)

    base_path = Path("/mnt/nvme/c-light")
    base_path.mkdir(parents=True, exist_ok=True)

    storage_path = base_path / "papers"
    db_path = base_path / "doi_database.db"
    graph_path = base_path / "knowledge_graph"
    system_a_path = base_path / "system_a"
    system_b_path = base_path / "system_b"
    eval_path = base_path / "evaluation"

    logger.info(f"Base path: {base_path}")


    # ===========================================
    # STEP 2: Initialize Harvester
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Initializing paper harvester")
    logger.info("=" * 60)

    harvester = ArxivHarvester(
        storage_path=str(storage_path),
        db_path=str(db_path)
    )


    # ===========================================
    # STEP 3: Harvest Papers
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Harvesting papers from ArXiv")
    logger.info("=" * 60)

    # Example: Harvest papers on dopamine and motivation
    queries = [
        "dopamine motivation",
        "reward learning",
        "neural plasticity behavior"
    ]

    logger.info(f"Searching for papers on: {queries}")

    for query in queries:
        logger.info(f"\nSearching: {query}")
        results = harvester.search_and_download(
            query=query,
            max_results=10,
            categories=["q-bio.NC", "cs.AI"]  # Neuroscience, AI
        )
        logger.info(f"  Downloaded {len(results)} papers")


    # ===========================================
    # STEP 4: Initialize Knowledge Graph
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Initializing knowledge graph")
    logger.info("=" * 60)

    kg = CLightKnowledgeGraph(graph_path=str(graph_path))
    logger.info(f"Knowledge graph initialized")


    # ===========================================
    # STEP 5: Initialize Both Systems
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Initializing System A (Mixtral) and System B (SEAL)")
    logger.info("=" * 60)

    # System A: Mixtral LLM-based RAG
    logger.info("Initializing System A (Mixtral)...")
    system_a = MixtralRAG(
        knowledge_graph=kg,
        storage_path=str(system_a_path),
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )

    # System B: SEAL self-learning RAG
    logger.info("Initializing System B (SEAL)...")
    system_b = SEALBasedRAG(
        knowledge_graph=kg,
        storage_path=str(system_b_path)
    )

    logger.info("✓ Both systems ready")


    # ===========================================
    # STEP 6: Process Papers with Both Systems
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Processing papers with both systems")
    logger.info("=" * 60)

    # Get unprocessed papers
    papers = harvester.get_unprocessed_papers(limit=20)
    logger.info(f"Found {len(papers)} unprocessed papers")

    # For SEAL, use active learning to prioritize
    priority_papers = system_b.get_next_papers_to_process(
        available_papers=papers,
        top_k=10
    )

    logger.info(f"SEAL prioritized {len(priority_papers)} papers for processing")

    # Process papers
    for i, paper in enumerate(priority_papers, 1):
        logger.info(f"\n[{i}/{len(priority_papers)}] Processing: {paper.title[:60]}...")

        # Get PDF path
        pdf_path = harvester.get_paper_path(paper.paper_id)

        # Add to System A
        logger.info("  Adding to System A...")
        system_a.add_paper(paper, pdf_path)

        # Add to System B
        logger.info("  Adding to System B...")
        system_b.add_paper(paper, pdf_path)

        logger.info("  ✓ Added to both systems")

    logger.info(f"\n✓ Processed {len(priority_papers)} papers")


    # ===========================================
    # STEP 7: Initialize Evaluator
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Initializing dual system evaluator")
    logger.info("=" * 60)

    evaluator = DualSystemEvaluator(
        system_a=system_a,
        system_b=system_b,
        storage_path=str(eval_path)
    )


    # ===========================================
    # STEP 8: Run Comparison Queries
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Running comparison queries")
    logger.info("=" * 60)

    test_questions = [
        "How does dopamine affect motivation?",
        "What is the relationship between reward and learning?",
        "Does neural plasticity influence behavior?",
        "What causes changes in synaptic strength?",
        "How do neurotransmitters modulate cognition?"
    ]

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n[{i}/{len(test_questions)}] Question: {question}")
        logger.info("-" * 60)

        # Compare both systems
        comparison = evaluator.compare_query(question)

        logger.info("\nSystem A (Mixtral) Answer:")
        logger.info(f"{comparison.system_a_answer[:200]}...")
        logger.info(f"Confidence: {comparison.system_a_confidence:.2f}")
        logger.info(f"Latency: {comparison.system_a_latency:.2f}s")
        logger.info(f"Sources: {comparison.system_a_sources}")

        logger.info("\nSystem B (SEAL) Answer:")
        logger.info(f"{comparison.system_b_answer[:200]}...")
        logger.info(f"Confidence: {comparison.system_b_confidence:.2f}")
        logger.info(f"Latency: {comparison.system_b_latency:.2f}s")
        logger.info(f"Sources: {comparison.system_b_sources}")


        # ===========================================
        # STEP 9: Simulate User Feedback
        # ===========================================
        # In real usage, user would rate the answers
        # Here we simulate ratings

        # For demonstration: System A starts better, System B improves
        if i <= 2:
            # Early queries: System A better
            system_a_rating = 4
            system_b_rating = 3
        else:
            # Later queries: System B catches up
            system_a_rating = 4
            system_b_rating = 4

        logger.info(f"\nSimulated ratings: A={system_a_rating}, B={system_b_rating}")

        evaluator.add_user_ratings(
            query_id=comparison.query_id,
            system_a_rating=system_a_rating,
            system_b_rating=system_b_rating
        )

        logger.info("✓ Feedback recorded and System B learned")


    # ===========================================
    # STEP 10: Analyze Results
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 10: Analyzing results and generating report")
    logger.info("=" * 60)

    # Get summary
    summary = evaluator.get_comparison_summary()

    logger.info("\n📊 Comparison Summary:")
    logger.info(f"Total comparisons: {summary['total_comparisons']}")
    logger.info(f"Rated comparisons: {summary['rated_comparisons']}")
    logger.info(f"\nSystem A:")
    logger.info(f"  Win rate: {summary['system_a']['win_rate']:.1%}")
    logger.info(f"  Avg rating: {summary['system_a']['avg_rating']:.2f}/5")
    logger.info(f"  Avg latency: {summary['system_a']['avg_latency']:.2f}s")
    logger.info(f"\nSystem B:")
    logger.info(f"  Win rate: {summary['system_b']['win_rate']:.1%}")
    logger.info(f"  Avg rating: {summary['system_b']['avg_rating']:.2f}/5")
    logger.info(f"  Avg latency: {summary['system_b']['avg_latency']:.2f}s")

    # Generate full report
    report_path = eval_path / "comparison_report.md"
    report = evaluator.generate_report(output_path=str(report_path))

    logger.info(f"\n✓ Full report saved to: {report_path}")


    # ===========================================
    # STEP 11: Check SEAL Learning
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 11: Checking System B (SEAL) learning progress")
    logger.info("=" * 60)

    seal_stats = system_b.get_stats()

    logger.info("\n📈 SEAL Statistics:")
    logger.info(f"Queries processed: {seal_stats['queries_processed']}")
    logger.info(f"Patterns learned: {seal_stats['patterns']['learned_patterns']}")
    logger.info(f"Total patterns: {seal_stats['patterns']['total_patterns']}")
    logger.info(f"Avg pattern confidence: {seal_stats['patterns']['avg_pattern_confidence']:.2f}")
    logger.info(f"\nEdge optimization:")
    logger.info(f"  Total edges seen: {seal_stats['weights']['total_edges_seen']}")
    logger.info(f"  Edges with adjustments: {seal_stats['weights']['edges_with_adjustments']}")
    logger.info(f"  Reinforcements: {seal_stats['weights']['reinforcements']}")
    logger.info(f"  Penalizations: {seal_stats['weights']['penalizations']}")

    # Check improvement
    improvement = system_b.analyze_improvement()

    if 'feedback_analysis' in improvement:
        logger.info("\n📊 Improvement Analysis:")
        logger.info(f"Total feedback: {improvement['total_feedback']}")
        logger.info(f"Learned patterns: {improvement['learned_patterns']}")


    # ===========================================
    # STEP 12: Save State
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 12: Saving system state")
    logger.info("=" * 60)

    # Save knowledge graph
    kg.save_to_disk()
    logger.info("✓ Knowledge graph saved")

    # Save System B state
    system_b.save_state()
    logger.info("✓ System B state saved")


    # ===========================================
    # SUMMARY
    # ===========================================
    logger.info("\n" + "=" * 60)
    logger.info("WORKFLOW COMPLETE!")
    logger.info("=" * 60)

    logger.info("\nWhat happened:")
    logger.info("1. ✓ Harvested papers from ArXiv")
    logger.info("2. ✓ Processed papers with both systems")
    logger.info("3. ✓ Compared systems on test queries")
    logger.info("4. ✓ System B learned from feedback")
    logger.info("5. ✓ Generated comparison report")
    logger.info("6. ✓ Saved all state to disk")

    logger.info("\nNext steps:")
    logger.info("- Continue querying and providing feedback")
    logger.info("- Monitor System B improvement over time")
    logger.info("- Run more comparisons (target: 100+ rated queries)")
    logger.info("- Check reports regularly for learning progress")

    logger.info(f"\n📁 All data saved to: {base_path}")
    logger.info(f"📊 View report at: {report_path}")

    return evaluator, system_a, system_b


if __name__ == "__main__":
    evaluator, system_a, system_b = main()

    print("\n" + "=" * 60)
    print("Systems are ready for interactive use!")
    print("=" * 60)
    print("\nExample interactive usage:")
    print(">>> result_a = system_a.query('Your question here')")
    print(">>> result_b = system_b.query('Your question here')")
    print(">>> # Rate System B's answer:")
    print(">>> system_b.provide_feedback(result_b.metadata['query_id'], rating=5)")

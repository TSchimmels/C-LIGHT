#!/usr/bin/env python3
"""
SEAL Learning Demonstration
Shows how System B improves over time with feedback

This demonstrates:
1. Initial performance (baseline)
2. Query → Feedback loop
3. Pattern discovery
4. Edge weight optimization
5. Active learning in action
6. Performance improvement over time
"""

import logging
from pathlib import Path
from typing import List
import time

from core.base_types import Paper, KnowledgeDomain
from core.knowledge_graph import CLightKnowledgeGraph
from system_b.seal_rag import SEALBasedRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_papers() -> List[Paper]:
    """Create sample papers for demonstration"""
    papers = [
        Paper(
            paper_id="demo:001",
            title="Dopamine's Role in Reward-Based Learning",
            authors=["Smith, J.", "Doe, A."],
            abstract="Dopamine increases motivation through reward processing. "
                    "We found that dopamine enhances learning when rewards are present. "
                    "The mesolimbic pathway causes behavioral reinforcement.",
            domains=[KnowledgeDomain.NEUROSCIENCE]
        ),
        Paper(
            paper_id="demo:002",
            title="Neural Plasticity and Behavioral Change",
            authors=["Johnson, B."],
            abstract="Neural plasticity leads to behavioral adaptation. "
                    "Synaptic strength modulates through experience. "
                    "Environmental enrichment increases cognitive flexibility.",
            domains=[KnowledgeDomain.NEUROSCIENCE, KnowledgeDomain.PSYCHOLOGY]
        ),
        Paper(
            paper_id="demo:003",
            title="Social Learning and Cultural Transmission",
            authors=["Chen, L.", "Williams, R."],
            abstract="Social learning causes cultural evolution. "
                    "Imitation enhances through observation. "
                    "Group dynamics affect individual behavior patterns.",
            domains=[KnowledgeDomain.SOCIOLOGY]
        ),
        Paper(
            paper_id="demo:004",
            title="Reward Systems and Decision Making",
            authors=["Garcia, M."],
            abstract="Reward anticipation influences decision processes. "
                    "Ventral striatum activation causes choice bias. "
                    "Expected value modulates neural responses.",
            domains=[KnowledgeDomain.NEUROSCIENCE]
        ),
        Paper(
            paper_id="demo:005",
            title="Epigenetic Mechanisms in Learning",
            authors=["Lee, K."],
            abstract="DNA methylation regulates memory formation. "
                    "Histone modifications cause persistent neural changes. "
                    "Epigenetic marks modulate gene expression during learning.",
            domains=[KnowledgeDomain.MOLECULAR_NEUROSCIENCE]
        ),
    ]
    return papers


def demonstrate_learning_loop():
    """Demonstrate the complete learning loop"""

    logger.info("=" * 70)
    logger.info("SEAL SELF-LEARNING DEMONSTRATION")
    logger.info("=" * 70)

    # Setup
    base_path = Path("/tmp/c-light-demo")
    base_path.mkdir(parents=True, exist_ok=True)

    kg_path = base_path / "kg"
    seal_path = base_path / "seal"

    # Initialize
    logger.info("\n📚 Initializing SEAL system...")
    kg = CLightKnowledgeGraph(graph_path=str(kg_path))
    seal = SEALBasedRAG(knowledge_graph=kg, storage_path=str(seal_path))

    # Add sample papers
    logger.info("\n📄 Adding sample papers...")
    papers = create_sample_papers()

    for i, paper in enumerate(papers, 1):
        logger.info(f"  [{i}/{len(papers)}] {paper.title}")
        seal.add_paper(paper)

    logger.info(f"\n✓ Added {len(papers)} papers")

    # Show initial stats
    stats = seal.get_stats()
    logger.info("\n📊 Initial Statistics:")
    logger.info(f"  Papers: {stats['knowledge_graph']['paper_count']}")
    logger.info(f"  Relations: {stats['knowledge_graph']['relation_count']}")
    logger.info(f"  Patterns: {stats['patterns']['total_patterns']}")
    logger.info(f"  Learned patterns: {stats['patterns']['learned_patterns']}")

    # ========================================
    # PHASE 1: Initial Queries (Baseline)
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: BASELINE PERFORMANCE")
    logger.info("=" * 70)

    test_queries = [
        ("Does dopamine increase motivation?", 3),  # Should find path
        ("How does reward affect learning?", 3),    # Should find path
        ("What causes behavioral change?", 2),      # Might struggle
        ("Does plasticity modulate behavior?", 2),  # Might struggle
    ]

    logger.info("\nRunning baseline queries...")

    baseline_results = []
    for question, expected_rating in test_queries:
        logger.info(f"\n❓ {question}")

        result = seal.query(question)

        logger.info(f"   Answer: {result.answer[:100]}...")
        logger.info(f"   Confidence: {result.confidence:.2f}")
        logger.info(f"   Sources: {result.sources_count}")

        baseline_results.append({
            'question': question,
            'confidence': result.confidence,
            'query_id': result.metadata['query_id'],
            'expected_rating': expected_rating
        })

    # Calculate baseline performance
    baseline_avg_conf = sum(r['confidence'] for r in baseline_results) / len(baseline_results)
    logger.info(f"\n📊 Baseline average confidence: {baseline_avg_conf:.2f}")

    # ========================================
    # PHASE 2: Learning from Feedback
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: LEARNING FROM FEEDBACK")
    logger.info("=" * 70)

    logger.info("\nProviding feedback on baseline queries...")

    for result in baseline_results:
        rating = result['expected_rating']

        logger.info(f"\n✍️  Rating query: {result['question'][:50]}...")
        logger.info(f"   Rating: {rating}/5")

        # Provide feedback
        seal.provide_feedback(
            query_id=result['query_id'],
            rating=rating
        )

        # If bad rating, provide correction to help learning
        if rating <= 2:
            correct_answer = "Let me provide the correct relationship..."
            seal.provide_feedback(
                query_id=result['query_id'],
                rating=rating,
                correct_answer=correct_answer
            )
            logger.info("   ✓ Correction provided for pattern learning")

    logger.info("\n✓ Feedback loop complete - System learned from results")

    # Show learning progress
    stats_after = seal.get_stats()
    logger.info("\n📊 Statistics After Feedback:")
    logger.info(f"  Total patterns: {stats_after['patterns']['total_patterns']}")
    logger.info(f"  Learned patterns: {stats_after['patterns']['learned_patterns']}")
    logger.info(f"  Edge adjustments: {stats_after['weights']['edges_with_adjustments']}")
    logger.info(f"  Reinforcements: {stats_after['weights']['reinforcements']}")
    logger.info(f"  Penalizations: {stats_after['weights']['penalizations']}")

    # ========================================
    # PHASE 3: Re-query (Improved Performance)
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: IMPROVED PERFORMANCE")
    logger.info("=" * 70)

    logger.info("\nRe-running same queries to show improvement...")

    improved_results = []
    for baseline_result in baseline_results:
        question = baseline_result['question']
        baseline_conf = baseline_result['confidence']

        logger.info(f"\n❓ {question}")

        result = seal.query(question)

        logger.info(f"   Answer: {result.answer[:100]}...")
        logger.info(f"   Baseline confidence: {baseline_conf:.2f}")
        logger.info(f"   New confidence: {result.confidence:.2f}")

        improvement = result.confidence - baseline_conf
        if improvement > 0:
            logger.info(f"   📈 Improved by: +{improvement:.2f}")
        elif improvement < 0:
            logger.info(f"   📉 Decreased by: {improvement:.2f}")
        else:
            logger.info(f"   ➡️  No change")

        improved_results.append({
            'question': question,
            'baseline_conf': baseline_conf,
            'new_conf': result.confidence,
            'improvement': improvement
        })

    # ========================================
    # PHASE 4: Active Learning
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: ACTIVE LEARNING")
    logger.info("=" * 70)

    logger.info("\nChecking which papers to process next...")

    # Create some new papers
    new_papers = [
        Paper(
            paper_id="demo:101",
            title="Advanced Dopaminergic Mechanisms",
            authors=["New, A."],
            abstract="Dopamine circuits in detail...",
            domains=[KnowledgeDomain.NEUROSCIENCE]
        ),
        Paper(
            paper_id="demo:102",
            title="Cultural Evolution Patterns",
            authors=["New, B."],
            abstract="Cultural transmission dynamics...",
            domains=[KnowledgeDomain.SOCIOLOGY]
        ),
        Paper(
            paper_id="demo:103",
            title="Molecular Memory Formation",
            authors=["New, C."],
            abstract="Cellular mechanisms of memory...",
            domains=[KnowledgeDomain.MOLECULAR_NEUROSCIENCE]
        ),
    ]

    # Get priorities
    priorities = seal.get_next_papers_to_process(new_papers, top_k=5)

    logger.info("\n📋 Priority ranking:")
    for i, paper in enumerate(priorities, 1):
        logger.info(f"  {i}. {paper.title}")
        logger.info(f"     Domain: {paper.domains[0].value}")

    # Get domain needs
    domain_priorities = seal.active_learner.get_domain_priorities()
    logger.info("\n🎯 Domain Needs:")
    for domain, need in sorted(domain_priorities.items(), key=lambda x: -x[1])[:5]:
        indicator = "🔴" if need > 1.2 else "🟡" if need > 0.9 else "🟢"
        logger.info(f"  {indicator} {domain.value}: {need:.2f}")

    underrep = seal.active_learner.get_underrepresented_domains(threshold=1.2)
    if underrep:
        logger.info(f"\n⚠️  Underrepresented domains: {[d.value for d in underrep]}")
        logger.info("   → Prioritize harvesting papers in these areas")

    # ========================================
    # PHASE 5: Summary
    # ========================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Calculate overall improvement
    avg_baseline = sum(r['baseline_conf'] for r in improved_results) / len(improved_results)
    avg_improved = sum(r['new_conf'] for r in improved_results) / len(improved_results)
    total_improvement = avg_improved - avg_baseline

    logger.info("\n📊 Performance Metrics:")
    logger.info(f"  Baseline avg confidence: {avg_baseline:.2f}")
    logger.info(f"  Improved avg confidence: {avg_improved:.2f}")
    logger.info(f"  Total improvement: {total_improvement:+.2f}")

    if total_improvement > 0.1:
        logger.info("\n✅ System showed significant improvement!")
    elif total_improvement > 0:
        logger.info("\n📈 System showed modest improvement")
    else:
        logger.info("\n⚠️  No improvement - may need more feedback or tuning")

    logger.info("\n📚 Learning Summary:")
    logger.info(f"  Queries processed: {stats_after['queries_processed']}")
    logger.info(f"  Feedback received: {stats_after['feedback']['total_feedback']}")
    logger.info(f"  Patterns learned: {stats_after['patterns']['learned_patterns']}")
    logger.info(f"  Edges optimized: {stats_after['weights']['edges_with_adjustments']}")

    # Show top edges
    top_edges = seal.weight_optimizer.get_top_edges(n=5)
    if top_edges:
        logger.info("\n🏆 Top 5 Most Reliable Edges:")
        for (source, target), confidence in top_edges:
            logger.info(f"  {source} → {target}: {confidence:.2f}")

    # Show learned patterns
    learned_patterns = seal.pattern_learner.get_learned_patterns()
    if learned_patterns:
        logger.info(f"\n🎓 Learned {len(learned_patterns)} New Patterns:")
        for pattern in learned_patterns[:3]:
            logger.info(f"  Pattern: {pattern.pattern_id}")
            logger.info(f"    Confidence: {pattern.confidence:.2f}")
            logger.info(f"    Success rate: {pattern.get_performance():.2f}")

    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 70)

    logger.info("\n🎯 Key Takeaways:")
    logger.info("1. System learns from every query + feedback")
    logger.info("2. Edge weights adjust based on success/failure")
    logger.info("3. New patterns discovered from corrections")
    logger.info("4. Active learning prioritizes useful papers")
    logger.info("5. Performance improves over time with use")

    logger.info("\n💡 In production:")
    logger.info("- Continue querying and providing feedback")
    logger.info("- System will improve over weeks/months")
    logger.info("- Monitor improvement with evaluation tools")
    logger.info("- Adjust learning parameters if needed")

    return seal


if __name__ == "__main__":
    seal_system = demonstrate_learning_loop()

    print("\n" + "=" * 70)
    print("Try it yourself!")
    print("=" * 70)
    print("\nExample usage:")
    print(">>> result = seal_system.query('Your question here')")
    print(">>> seal_system.provide_feedback(result.metadata['query_id'], rating=5)")
    print(">>> stats = seal_system.get_stats()")

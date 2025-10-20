"""
Active Learning for SEAL System
Intelligently selects which papers to process next for maximum learning benefit
"""

import logging
from typing import List, Dict, Set, Optional
from collections import defaultdict
import numpy as np

from ..core.base_types import Paper, KnowledgeDomain

logger = logging.getLogger(__name__)


class ActivePaperSelector:
    """
    Active learning: selects papers that will most improve the system

    Key ideas:
    1. **Uncertainty Sampling**: Prioritize domains where we're weak
    2. **Diversity**: Get papers covering different topics
    3. **Query-Driven**: Process papers relevant to failed queries
    4. **Coverage**: Fill gaps in knowledge graph

    This is much smarter than random paper selection!
    """

    def __init__(self):
        # Domain needs (higher = more needed)
        self.domain_needs: Dict[KnowledgeDomain, float] = defaultdict(lambda: 1.0)

        # Concept coverage (how many papers mention each concept)
        self.concept_coverage: Dict[str, int] = defaultdict(int)

        # Failed query topics (we need more papers on these)
        self.failed_query_topics: Dict[str, int] = defaultdict(int)

        # Recently processed papers (avoid duplicates)
        self.recently_processed: Set[str] = set()

        logger.info("Active paper selector initialized")

    def prioritize_papers(
        self,
        available_papers: List[Paper],
        top_k: int = 100
    ) -> List[Paper]:
        """
        Select top K papers to process next

        Args:
            available_papers: All papers available for processing
            top_k: How many to select

        Returns:
            List of papers to process, ordered by priority
        """
        if not available_papers:
            return []

        logger.info(f"Prioritizing {len(available_papers)} papers...")

        # Score each paper
        scored_papers = []

        for paper in available_papers:
            # Skip recently processed
            if paper.paper_id in self.recently_processed:
                continue

            score = self._calculate_paper_score(paper)
            scored_papers.append((score, paper))

        # Sort by score (highest first)
        scored_papers.sort(reverse=True, key=lambda x: x[0])

        # Select top K
        selected = [paper for score, paper in scored_papers[:top_k]]

        logger.info(f"Selected {len(selected)} papers to process")

        return selected

    def _calculate_paper_score(self, paper: Paper) -> float:
        """
        Calculate priority score for a paper

        Higher score = should process sooner

        Combines multiple factors:
        - Domain need (domains we're weak in)
        - Novelty (concepts we haven't seen much)
        - Query relevance (related to failed queries)
        - Diversity (different from recent papers)
        """
        score = 0.0

        # 1. Domain need (40% of score)
        domain_score = 0.0
        if paper.domains:
            domain_scores = [self.domain_needs[domain] for domain in paper.domains]
            domain_score = np.mean(domain_scores)
        score += 0.4 * domain_score

        # 2. Novelty (30% of score)
        novelty_score = self._calculate_novelty(paper)
        score += 0.3 * novelty_score

        # 3. Query relevance (20% of score)
        query_relevance = self._calculate_query_relevance(paper)
        score += 0.2 * query_relevance

        # 4. Citation count (10% of score) - prioritize influential papers
        citation_score = min(paper.citation_count / 100.0, 1.0) if paper.citation_count else 0.0
        score += 0.1 * citation_score

        return score

    def _calculate_novelty(self, paper: Paper) -> float:
        """
        Calculate how novel this paper is

        Novel = contains concepts we haven't seen much

        Returns:
            Novelty score (0-1, higher = more novel)
        """
        if not paper.abstract:
            return 0.5  # Unknown, neutral

        # Extract concepts from abstract
        concepts = self._extract_concepts(paper.abstract)

        if not concepts:
            return 0.5

        # Calculate average coverage of these concepts
        coverage_scores = []
        for concept in concepts:
            coverage = self.concept_coverage.get(concept, 0)

            # Novel if coverage < 5 papers
            if coverage == 0:
                coverage_scores.append(1.0)  # Completely novel
            elif coverage < 5:
                coverage_scores.append(0.8)  # Quite novel
            elif coverage < 20:
                coverage_scores.append(0.5)  # Somewhat novel
            else:
                coverage_scores.append(0.2)  # Well-covered

        return np.mean(coverage_scores)

    def _calculate_query_relevance(self, paper: Paper) -> float:
        """
        Calculate how relevant this paper is to failed queries

        Returns:
            Relevance score (0-1)
        """
        if not paper.abstract or not self.failed_query_topics:
            return 0.5

        abstract_lower = paper.abstract.lower()

        # Count how many failed query topics appear in this paper
        relevant_count = 0
        for topic, failure_count in self.failed_query_topics.items():
            if topic.lower() in abstract_lower:
                # Weight by how many times this topic failed
                relevant_count += min(failure_count, 5)

        # Normalize
        max_possible = sum(min(count, 5) for count in self.failed_query_topics.values())

        if max_possible == 0:
            return 0.5

        return min(relevant_count / max_possible, 1.0)

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple approach: extract significant words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }

        words = text.lower().split()
        concepts = [
            w.strip('.,;:!?')
            for w in words
            if w not in stop_words and len(w) > 4
        ]

        return concepts[:20]  # Top 20 concepts

    def update_after_processing(self, paper: Paper):
        """
        Update statistics after processing a paper

        Args:
            paper: Paper that was just processed
        """
        # Mark as processed
        self.recently_processed.add(paper.paper_id)

        # Update concept coverage
        if paper.abstract:
            concepts = self._extract_concepts(paper.abstract)
            for concept in concepts:
                self.concept_coverage[concept] += 1

        # Decrease need for these domains (we just got a paper on them)
        for domain in paper.domains:
            self.domain_needs[domain] *= 0.95  # 5% decrease

        logger.debug(f"Updated stats after processing {paper.paper_id}")

    def update_from_query_failure(
        self,
        question: str,
        missing_domains: List[KnowledgeDomain],
        failed_concepts: List[str]
    ):
        """
        Update priorities based on a failed query

        When a query fails, we need more papers on those topics

        Args:
            question: The query that failed
            missing_domains: Domains that need more coverage
            failed_concepts: Concepts we couldn't answer about
        """
        # Increase need for missing domains
        for domain in missing_domains:
            self.domain_needs[domain] *= 1.2  # 20% increase

        # Track failed concepts
        for concept in failed_concepts:
            self.failed_query_topics[concept] += 1

        logger.info(f"Updated priorities after failed query: {missing_domains}")

    def update_from_query_success(
        self,
        domains: List[KnowledgeDomain],
        concepts: List[str]
    ):
        """
        Update priorities based on successful query

        Success means we have good coverage in these areas

        Args:
            domains: Domains used in successful query
            concepts: Concepts used successfully
        """
        # Slightly decrease need (we're doing well here)
        for domain in domains:
            self.domain_needs[domain] *= 0.98  # 2% decrease

        logger.debug(f"Updated priorities after successful query")

    def get_domain_priorities(self) -> Dict[KnowledgeDomain, float]:
        """
        Get current domain priorities

        Returns:
            Dict mapping domains to priority scores
        """
        return dict(self.domain_needs)

    def get_underrepresented_domains(self, threshold: float = 1.5) -> List[KnowledgeDomain]:
        """
        Get domains that need more papers

        Args:
            threshold: Domains with need > threshold are underrepresented

        Returns:
            List of underrepresented domains
        """
        return [
            domain for domain, need in self.domain_needs.items()
            if need > threshold
        ]

    def get_novel_concepts(self, max_count: int = 10) -> List[str]:
        """
        Get concepts with low coverage (good targets for new papers)

        Args:
            max_count: Maximum number to return

        Returns:
            List of novel concepts
        """
        # Sort by coverage (ascending)
        sorted_concepts = sorted(
            self.concept_coverage.items(),
            key=lambda x: x[1]
        )

        return [concept for concept, count in sorted_concepts[:max_count] if count < 5]

    def get_stats(self) -> Dict[str, any]:
        """Get active learning statistics"""
        return {
            'papers_processed': len(self.recently_processed),
            'tracked_concepts': len(self.concept_coverage),
            'failed_query_topics': len(self.failed_query_topics),
            'underrepresented_domains': len(self.get_underrepresented_domains()),
            'avg_domain_need': np.mean(list(self.domain_needs.values())) if self.domain_needs else 1.0,
            'max_domain_need': max(self.domain_needs.values()) if self.domain_needs else 1.0
        }

    def recommend_search_keywords(self, n: int = 10) -> List[str]:
        """
        Recommend keywords for paper search based on current needs

        Returns keywords for topics we need more papers on

        Args:
            n: Number of keywords to return

        Returns:
            List of recommended search keywords
        """
        keywords = []

        # Get underrepresented concepts
        novel_concepts = self.get_novel_concepts(n // 2)
        keywords.extend(novel_concepts)

        # Get failed query topics
        failed_topics = sorted(
            self.failed_query_topics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        keywords.extend([topic for topic, count in failed_topics[:n // 2]])

        return keywords[:n]

    def should_process_now(self, paper: Paper) -> bool:
        """
        Quick check: should we process this paper immediately?

        Returns True if paper is high priority

        Args:
            paper: Paper to check

        Returns:
            True if should process immediately
        """
        score = self._calculate_paper_score(paper)

        # High priority if score > 0.7
        return score > 0.7

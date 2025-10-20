"""
SEAL-Style Self-Learning RAG System
Learns from feedback, discovers patterns, improves over time
"""

import logging
import time
import uuid
from typing import List, Dict, Optional, Any
from pathlib import Path

from ..core.base_types import Paper, KnowledgeDomain, QueryResult, Evidence, CausalRelation
from ..core.knowledge_graph import CLightKnowledgeGraph
from ..core.paper_processor import PaperProcessor

from .feedback_system import FeedbackCollector, QueryFeedback
from .pattern_learner import PatternLearner
from .weight_optimizer import EdgeWeightOptimizer
from .active_learning import ActivePaperSelector

logger = logging.getLogger(__name__)


class SEALBasedRAG:
    """
    SEAL-style self-learning RAG system (System B)

    Unlike System A (Mixtral), this system:
    - Learns new causal patterns from feedback
    - Optimizes edge weights based on query success/failure
    - Actively selects which papers to process
    - Improves over time with use

    Key differences from traditional RAG:
    1. Feedback-driven: Gets better with user ratings
    2. Pattern discovery: Learns new ways to extract causality
    3. Active learning: Smart paper selection
    4. Knowledge graph-based: Explicit reasoning, not black-box LLM
    """

    def __init__(
        self,
        knowledge_graph: Optional[CLightKnowledgeGraph] = None,
        storage_path: str = "/mnt/nvme/c-light/seal"
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing SEAL RAG System B")

        # Knowledge graph
        self.kg = knowledge_graph or CLightKnowledgeGraph(
            graph_path=str(self.storage_path / "knowledge_graph")
        )

        # Paper processor
        self.paper_processor = PaperProcessor()

        # Learning components
        self.feedback_collector = FeedbackCollector(
            storage_path=str(self.storage_path / "feedback")
        )

        self.pattern_learner = PatternLearner(
            storage_path=str(self.storage_path / "patterns")
        )

        self.weight_optimizer = EdgeWeightOptimizer(
            storage_path=str(self.storage_path / "weights")
        )

        self.active_learner = ActivePaperSelector()

        # Query stats
        self.query_count = 0
        self.total_latency = 0.0

        logger.info("✓ SEAL RAG System B initialized")

    def add_paper(self, paper: Paper, pdf_path: Optional[str] = None):
        """
        Add a paper to the system

        Extracts knowledge using learned patterns

        Args:
            paper: Paper object
            pdf_path: Optional path to PDF for full text
        """
        logger.info(f"Processing {paper.paper_id} with SEAL...")

        # Process paper into chunks
        pdf_path_obj = Path(pdf_path) if pdf_path else None
        chunks = self.paper_processor.process_paper(paper, pdf_path_obj)

        # Add paper to knowledge graph
        self.kg.add_paper(paper)

        # Extract causal relations using learned patterns
        relations = self.pattern_learner.extract_relations(
            paper,
            use_learned=True  # Use both base and learned patterns
        )

        logger.info(f"  Extracted {len(relations)} causal relations")

        # Add relations to knowledge graph
        for relation in relations:
            self.kg.add_causal_relation(relation)

        # Update active learning stats
        self.active_learner.update_after_processing(paper)

        logger.info(f"✓ Added {paper.paper_id} to SEAL")

    def query(
        self,
        question: str,
        domains: Optional[List[KnowledgeDomain]] = None,
        max_results: int = 10
    ) -> QueryResult:
        """
        Query the knowledge base

        Uses knowledge graph with learned edge weights

        Args:
            question: Natural language question
            domains: Optional domain filter
            max_results: Maximum results to return

        Returns:
            QueryResult with answer and evidence
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())

        logger.info(f"SEAL query [{query_id}]: {question[:60]}...")

        # Parse question to identify concepts
        concepts = self._extract_concepts(question)

        if len(concepts) < 2:
            return QueryResult(
                query=question,
                answer="I need at least two concepts to find relationships.",
                confidence=0.0,
                metadata={'system': 'seal_rag', 'query_id': query_id}
            )

        # Determine if this is a causal query
        if self._is_causal_query(question):
            result = self._answer_causal_query(
                question, concepts, domains, query_id
            )
        else:
            result = self._answer_influence_query(
                question, concepts, domains, query_id
            )

        latency = time.time() - start_time
        self.query_count += 1
        self.total_latency += latency

        result.metadata['latency'] = latency
        result.metadata['system'] = 'seal_rag'
        result.metadata['query_id'] = query_id

        # Record query for feedback
        self.feedback_collector.record_query(
            query_id=query_id,
            question=question,
            answer=result.answer,
            papers_retrieved=[e.paper_id for e in result.evidence],
            causal_path=result.causal_paths[0] if result.causal_paths else [],
            confidence=result.confidence,
            domains=[d.value for d in domains] if domains else []
        )

        logger.info(f"✓ SEAL answered in {latency:.2f}s (confidence: {result.confidence:.2f})")

        return result

    def _answer_causal_query(
        self,
        question: str,
        concepts: List[str],
        domains: Optional[List[KnowledgeDomain]],
        query_id: str
    ) -> QueryResult:
        """Answer a causal question using knowledge graph"""

        source = concepts[0]
        target = concepts[1]

        # Find causal paths with optimized weights
        paths = self._find_weighted_paths(source, target, max_hops=5)

        if not paths:
            return QueryResult(
                query=question,
                answer=f"I could not find a causal relationship between {source} and {target} in my knowledge base.",
                confidence=0.0,
                metadata={'query_id': query_id}
            )

        # Get best path
        best_path = paths[0]
        path_nodes = [self.kg.graph.nodes[node]['name'] for node in best_path['path']]

        # Calculate confidence based on edge reliabilities
        confidence = best_path['confidence']

        # Build answer
        answer = self._build_causal_answer(best_path, path_nodes)

        # Collect evidence
        evidence = self._collect_evidence(best_path['path'])

        return QueryResult(
            query=question,
            answer=answer,
            evidence=evidence,
            causal_paths=[path_nodes],
            confidence=confidence,
            sources_count=len(evidence),
            metadata={
                'query_id': query_id,
                'total_paths_found': len(paths)
            }
        )

    def _answer_influence_query(
        self,
        question: str,
        concepts: List[str],
        domains: Optional[List[KnowledgeDomain]],
        query_id: str
    ) -> QueryResult:
        """Answer a query about influences"""

        concept = concepts[0]

        influences_data = self.kg.query_influences(concept, max_depth=2, domains=domains)

        if 'error' in influences_data:
            return QueryResult(
                query=question,
                answer=f"I don't have information about {concept}.",
                confidence=0.0,
                metadata={'query_id': query_id}
            )

        influences = influences_data['influences']

        if not influences:
            return QueryResult(
                query=question,
                answer=f"I found {concept} in my knowledge base but no influences.",
                confidence=0.3,
                metadata={'query_id': query_id}
            )

        # Build answer
        top_influences = influences[:5]
        answer_parts = [f"Factors that influence {concept}:"]

        total_confidence = 0.0
        for inf in top_influences:
            # Get learned confidence for this edge
            edge_confidence = self.weight_optimizer.get_edge_confidence(
                inf['source'],
                concept
            )

            answer_parts.append(
                f"- {inf['source']} {inf['relation_type']} {concept} "
                f"(confidence: {edge_confidence:.2f})"
            )
            total_confidence += edge_confidence

        answer = "\n".join(answer_parts)

        # Average confidence
        avg_confidence = total_confidence / len(top_influences)

        # Collect evidence
        evidence = []
        for inf in top_influences:
            if inf['evidence'] in self.kg.paper_index:
                paper = self.kg.paper_index[inf['evidence']]
                evidence.append(Evidence(
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    text=paper.abstract[:200] if paper.abstract else "",
                    confidence=edge_confidence,
                    relevance_score=inf['weight']
                ))

        return QueryResult(
            query=question,
            answer=answer,
            evidence=evidence,
            confidence=avg_confidence,
            sources_count=len(influences),
            metadata={'query_id': query_id, 'total_influences': len(influences)}
        )

    def _find_weighted_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find causal paths using learned edge weights

        Returns paths sorted by learned reliability
        """
        import networkx as nx

        source_id = f"entity:concept:{source}"
        target_id = f"entity:concept:{target}"

        if not self.kg.graph.has_node(source_id) or not self.kg.graph.has_node(target_id):
            return []

        # Find all simple paths
        try:
            all_paths = list(nx.all_simple_paths(
                self.kg.graph,
                source_id,
                target_id,
                cutoff=max_hops
            ))
        except nx.NetworkXNoPath:
            return []

        # Score each path using learned weights
        scored_paths = []

        for path in all_paths:
            total_weight = 0.0
            min_confidence = 1.0
            edge_confidences = []

            for i in range(len(path) - 1):
                # Get base weight from graph
                edges = self.kg.graph.get_edge_data(path[i], path[i+1])

                if edges:
                    # Get max base weight among multiple edges
                    base_weight = max(e.get('weight', 0) for e in edges.values())

                    # Get learned adjustment
                    adjusted_weight = self.weight_optimizer.get_edge_weight(
                        path[i],
                        path[i+1],
                        base_weight
                    )

                    # Get learned confidence
                    edge_conf = self.weight_optimizer.get_edge_confidence(
                        path[i],
                        path[i+1]
                    )

                    total_weight += adjusted_weight
                    edge_confidences.append(edge_conf)
                    min_confidence = min(min_confidence, edge_conf)

            # Path confidence is minimum edge confidence
            # (weakest link determines strength)
            path_confidence = min_confidence if edge_confidences else 0.5

            scored_paths.append({
                'path': path,
                'weight': total_weight,
                'confidence': path_confidence,
                'edge_confidences': edge_confidences
            })

        # Sort by weight (descending)
        scored_paths.sort(key=lambda x: x['weight'], reverse=True)

        return scored_paths

    def _build_causal_answer(
        self,
        path_data: Dict[str, Any],
        path_nodes: List[str]
    ) -> str:
        """Build natural language answer from causal path"""

        path_str = " → ".join(path_nodes)
        confidence = path_data['confidence']

        # Build answer
        answer = f"Causal pathway: {path_str}\n\n"

        if confidence > 0.7:
            answer += "This pathway is well-supported based on past queries and evidence."
        elif confidence > 0.5:
            answer += "This pathway has moderate support in my knowledge base."
        else:
            answer += "This pathway is tentative and may need more evidence."

        # Add edge details
        if 'edge_confidences' in path_data:
            answer += "\n\nEdge reliabilities:"
            for i, conf in enumerate(path_data['edge_confidences']):
                answer += f"\n  {path_nodes[i]} → {path_nodes[i+1]}: {conf:.2f}"

        return answer

    def _collect_evidence(self, path: List[str]) -> List[Evidence]:
        """Collect evidence papers for a path"""
        evidence = []

        for i in range(len(path) - 1):
            edges = self.kg.graph.get_edge_data(path[i], path[i+1])

            if edges:
                for edge_data in edges.values():
                    paper_id = edge_data.get('paper_id')
                    if paper_id and paper_id in self.kg.paper_index:
                        paper = self.kg.paper_index[paper_id]

                        # Get learned confidence for this edge
                        edge_conf = self.weight_optimizer.get_edge_confidence(
                            path[i],
                            path[i+1]
                        )

                        evidence.append(Evidence(
                            paper_id=paper.paper_id,
                            paper_title=paper.title,
                            text=edge_data.get('evidence_text', ''),
                            confidence=edge_conf,  # Use learned confidence!
                            relevance_score=edge_data.get('weight', 0.0)
                        ))

        return evidence

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from query"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where', 'which', 'who'
        }

        words = text.lower().split()
        concepts = [w.strip('?.,;:!') for w in words if w not in stop_words and len(w) > 3]

        return concepts

    def _is_causal_query(self, question: str) -> bool:
        """Check if query is asking about causation"""
        causal_keywords = ['cause', 'lead to', 'result in', 'affect', 'influence', 'impact', 'effect']
        return any(kw in question.lower() for kw in causal_keywords)

    def provide_feedback(
        self,
        query_id: str,
        rating: int,
        correct_answer: Optional[str] = None
    ):
        """
        Provide feedback for a query - THIS IS WHERE LEARNING HAPPENS!

        Args:
            query_id: Query ID to rate
            rating: 1-5 rating (1=terrible, 5=excellent)
            correct_answer: Optional correct answer if system was wrong
        """
        # Record feedback
        success = self.feedback_collector.add_feedback(
            query_id=query_id,
            rating=rating,
            correct_answer=correct_answer
        )

        if not success:
            logger.warning(f"Could not record feedback for {query_id}")
            return

        # Get query details
        feedback = self.feedback_collector.get_feedback(query_id)

        if not feedback:
            return

        logger.info(f"Learning from feedback: query_id={query_id}, rating={rating}")

        # Update edge weights based on rating
        if feedback.causal_path:
            if rating >= 4:  # Success
                self.weight_optimizer.reinforce_path(
                    feedback.causal_path,
                    rating=rating
                )
            else:  # Failure
                self.weight_optimizer.penalize_path(
                    feedback.causal_path,
                    rating=rating
                )

        # Learn new patterns if correct answer provided
        if correct_answer:
            self.pattern_learner.learn_from_feedback(
                question=feedback.question,
                system_answer=feedback.answer,
                correct_answer=correct_answer,
                papers=[],  # Could fetch papers here
                rating=rating
            )

        # Update active learning priorities
        if rating <= 2:  # Failed query
            concepts = self._extract_concepts(feedback.question)
            self.active_learner.update_from_query_failure(
                question=feedback.question,
                missing_domains=[KnowledgeDomain(d) for d in feedback.domains] if feedback.domains else [],
                failed_concepts=concepts
            )
        else:  # Successful query
            self.active_learner.update_from_query_success(
                domains=[KnowledgeDomain(d) for d in feedback.domains] if feedback.domains else [],
                concepts=self._extract_concepts(feedback.question)
            )

        logger.info("✓ Learned from feedback")

    def get_next_papers_to_process(
        self,
        available_papers: List[Paper],
        top_k: int = 100
    ) -> List[Paper]:
        """
        Use active learning to select which papers to process next

        This is smarter than random selection!

        Args:
            available_papers: All papers available
            top_k: How many to select

        Returns:
            Prioritized list of papers
        """
        return self.active_learner.prioritize_papers(available_papers, top_k)

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'system': 'seal_rag',
            'knowledge_graph': self.kg.get_stats(),
            'feedback': self.feedback_collector.get_stats(),
            'patterns': self.pattern_learner.get_stats(),
            'weights': self.weight_optimizer.get_stats(),
            'active_learning': self.active_learner.get_stats(),
            'queries_processed': self.query_count,
            'avg_latency': self.total_latency / max(self.query_count, 1),
        }

    def analyze_improvement(self) -> Dict[str, Any]:
        """
        Analyze how much the system has improved

        Returns metrics showing learning progress
        """
        feedback_analysis = self.feedback_collector.analyze_patterns()
        weight_improvement = self.weight_optimizer.analyze_improvement()

        return {
            'feedback_analysis': feedback_analysis,
            'weight_improvement': weight_improvement,
            'learned_patterns': len(self.pattern_learner.get_learned_patterns()),
            'total_feedback': len(self.feedback_collector.feedback_history)
        }

    def save_state(self):
        """Save all learned knowledge to disk"""
        self.kg.save_to_disk()
        logger.info("✓ SEAL state saved")

    def load_state(self):
        """Load learned knowledge from disk"""
        self.kg.load_from_disk()
        logger.info("✓ SEAL state loaded")

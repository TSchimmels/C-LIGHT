"""
RAG (Retrieval-Augmented Generation) System for C-LIGHT
Standalone implementation for querying the knowledge base
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

from .base_types import Paper, KnowledgeDomain
from .knowledge_graph import CLightKnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """Evidence supporting an answer"""
    paper_id: str
    paper_title: str
    text: str
    confidence: float
    relevance_score: float = 0.0


@dataclass
class QueryResult:
    """Result from a RAG query"""
    query: str
    answer: str
    evidence: List[Evidence] = field(default_factory=list)
    causal_paths: List[List[str]] = field(default_factory=list)
    confidence: float = 0.0
    sources_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CLightRAG:
    """
    Simple RAG system for C-LIGHT
    Uses knowledge graph for retrieval
    Can be extended with vector search and LLM generation
    """

    def __init__(
        self,
        knowledge_graph: CLightKnowledgeGraph,
        paper_index: Optional[Dict[str, Paper]] = None
    ):
        self.kg = knowledge_graph
        self.paper_index = paper_index or {}

    def query(
        self,
        question: str,
        max_results: int = 10,
        min_confidence: float = 0.5,
        domains: Optional[List[KnowledgeDomain]] = None
    ) -> QueryResult:
        """
        Query the knowledge base

        Args:
            question: Natural language question
            max_results: Maximum number of evidence pieces
            min_confidence: Minimum confidence threshold
            domains: Optional domain filter

        Returns:
            QueryResult with answer and evidence
        """
        # Parse question to identify concepts
        # For now, simple keyword extraction
        # TODO: Use NLP for better query understanding
        concepts = self._extract_concepts(question)

        if not concepts:
            return QueryResult(
                query=question,
                answer="Could not identify concepts in query",
                confidence=0.0
            )

        # Check if asking about causal relationships
        if self._is_causal_query(question):
            return self._answer_causal_query(question, concepts, domains)

        # Check if asking about influences
        if self._is_influence_query(question):
            return self._answer_influence_query(question, concepts, domains)

        # Default: return general information
        return self._answer_general_query(question, concepts, domains, max_results)

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from query text
        Simple implementation - can be improved with NLP
        """
        # Remove common words
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
        causal_keywords = ['cause', 'lead to', 'result in', 'affect', 'influence', 'impact']
        return any(kw in question.lower() for kw in causal_keywords)

    def _is_influence_query(self, question: str) -> bool:
        """Check if query is asking about influences"""
        influence_keywords = ['influence', 'affect', 'impact', 'effect on', 'what affects']
        return any(kw in question.lower() for kw in influence_keywords)

    def _answer_causal_query(
        self,
        question: str,
        concepts: List[str],
        domains: Optional[List[KnowledgeDomain]]
    ) -> QueryResult:
        """Answer a causal question"""
        if len(concepts) < 2:
            return QueryResult(
                query=question,
                answer="Need at least two concepts to find causal relationship",
                confidence=0.0
            )

        # Try to find path between first two concepts
        source = concepts[0]
        target = concepts[1]

        paths = self.kg.find_causal_path(source, target, max_hops=5)

        if not paths:
            return QueryResult(
                query=question,
                answer=f"No causal path found between {source} and {target}",
                confidence=0.0
            )

        # Build answer from strongest path
        strongest_path = paths[0]
        path_names = [self.kg.graph.nodes[node]['name'] for node in strongest_path]

        answer = f"Causal pathway: {' → '.join(path_names)}"

        # Collect evidence
        evidence = self._collect_evidence_for_path(strongest_path)

        return QueryResult(
            query=question,
            answer=answer,
            evidence=evidence,
            causal_paths=[path_names],
            confidence=0.8 if paths else 0.0,
            sources_count=len(evidence),
            metadata={'paths_found': len(paths)}
        )

    def _answer_influence_query(
        self,
        question: str,
        concepts: List[str],
        domains: Optional[List[KnowledgeDomain]]
    ) -> QueryResult:
        """Answer a question about influences"""
        if not concepts:
            return QueryResult(
                query=question,
                answer="Could not identify concept in query",
                confidence=0.0
            )

        concept = concepts[0]
        influences_data = self.kg.query_influences(concept, max_depth=2, domains=domains)

        if 'error' in influences_data:
            return QueryResult(
                query=question,
                answer=influences_data['error'],
                confidence=0.0
            )

        influences = influences_data['influences']

        if not influences:
            return QueryResult(
                query=question,
                answer=f"No influences found for {concept}",
                confidence=0.0
            )

        # Build answer
        top_influences = influences[:5]
        answer_parts = [f"Factors that influence {concept}:"]

        for inf in top_influences:
            answer_parts.append(
                f"- {inf['source']} ({inf['relation_type']}, "
                f"confidence: {inf['confidence']:.2f})"
            )

        answer = "\n".join(answer_parts)

        # Collect evidence
        evidence = []
        for inf in top_influences:
            if inf['evidence'] in self.paper_index:
                paper = self.paper_index[inf['evidence']]
                evidence.append(Evidence(
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    text=paper.abstract[:200] + "..." if paper.abstract else "",
                    confidence=inf['confidence'],
                    relevance_score=inf['weight']
                ))

        return QueryResult(
            query=question,
            answer=answer,
            evidence=evidence,
            confidence=0.7,
            sources_count=len(influences),
            metadata={'total_influences': len(influences)}
        )

    def _answer_general_query(
        self,
        question: str,
        concepts: List[str],
        domains: Optional[List[KnowledgeDomain]],
        max_results: int
    ) -> QueryResult:
        """Answer a general information query"""
        # For now, just return concept information
        # TODO: Add semantic search with embeddings

        answer = f"Information about: {', '.join(concepts)}\n"
        answer += "(This is a basic implementation. Add semantic search for better results)"

        return QueryResult(
            query=question,
            answer=answer,
            confidence=0.3,
            metadata={'concepts': concepts}
        )

    def _collect_evidence_for_path(self, path: List[str]) -> List[Evidence]:
        """Collect evidence papers for a causal path"""
        evidence = []

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            # Get edges between these nodes
            edges = self.kg.graph.get_edge_data(source, target)

            if edges:
                for edge_data in edges.values():
                    paper_id = edge_data.get('paper_id')
                    if paper_id and paper_id in self.paper_index:
                        paper = self.paper_index[paper_id]
                        evidence.append(Evidence(
                            paper_id=paper.paper_id,
                            paper_title=paper.title,
                            text=edge_data.get('evidence_text', ''),
                            confidence=edge_data.get('confidence', 0.0),
                            relevance_score=edge_data.get('weight', 0.0)
                        ))

        return evidence

    def get_summary(self, concept: str) -> Dict[str, Any]:
        """Get a summary of information about a concept"""
        influences = self.kg.query_influences(concept)

        # Count papers mentioning this concept
        concept_id = f"entity:concept:{concept}"
        papers = []
        if self.kg.graph.has_node(concept_id):
            papers = self.kg.graph.nodes[concept_id].get('papers', [])

        return {
            'concept': concept,
            'papers_count': len(papers),
            'influences_count': influences.get('count', 0),
            'top_influences': influences.get('influences', [])[:5]
        }

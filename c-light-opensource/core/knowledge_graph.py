"""
Knowledge Graph implementation for C-LIGHT
Uses NetworkX for graph operations - lightweight and standalone
"""

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import networkx as nx

from .base_types import Paper, CausalRelation, Entity, KnowledgeDomain


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    CONCEPT = "concept"
    ENTITY = "entity"
    PAPER = "paper"
    INTERVENTION = "intervention"
    OUTCOME = "outcome"


class EdgeType(Enum):
    """Types of edges in the knowledge graph"""
    CAUSES = "causes"
    CORRELATES = "correlates"
    MODULATES = "modulates"
    PREDICTS = "predicts"
    CONTRADICTS = "contradicts"
    CITES = "cites"


@dataclass
class Node:
    """Node in the knowledge graph"""
    node_id: str
    node_type: NodeType
    name: str
    domains: List[KnowledgeDomain] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    papers: Set[str] = field(default_factory=set)


@dataclass
class Edge:
    """Edge in the knowledge graph"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float
    confidence: float
    evidence: List[str] = field(default_factory=list)  # Paper IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


class CLightKnowledgeGraph:
    """
    Knowledge graph for causal relationships in cognitive science
    Completely standalone implementation using NetworkX
    """

    def __init__(self, graph_path: Optional[str] = None):
        self.graph = nx.MultiDiGraph()
        self.graph_path = Path(graph_path) if graph_path else None

        # Indices for fast lookup
        self.paper_index: Dict[str, Paper] = {}
        self.entity_index: Dict[str, Entity] = {}
        self.domain_index: Dict[KnowledgeDomain, Set[str]] = defaultdict(set)

        if self.graph_path and self.graph_path.exists():
            self.load_from_disk()

    def add_paper(self, paper: Paper):
        """Add a paper to the knowledge graph"""
        node_id = f"paper:{paper.paper_id}"

        self.graph.add_node(
            node_id,
            node_type=NodeType.PAPER.value,
            name=paper.title,
            paper_id=paper.paper_id,
            authors=paper.authors,
            published=paper.published_date.isoformat() if paper.published_date else None,
            domains=[d.value for d in paper.domains],
            citation_count=paper.citation_count
        )

        self.paper_index[paper.paper_id] = paper

        # Index by domain
        for domain in paper.domains:
            self.domain_index[domain].add(node_id)

    def add_entity(self, entity: Entity):
        """Add an entity to the knowledge graph"""
        node_id = f"entity:{entity.entity_type}:{entity.name}"

        if self.graph.has_node(node_id):
            # Update existing node
            node = self.graph.nodes[node_id]
            node['mentions'] = node.get('mentions', 0) + entity.mentions
            node['papers'] = node.get('papers', set()) | entity.papers
        else:
            # Create new node
            self.graph.add_node(
                node_id,
                node_type=NodeType.ENTITY.value,
                name=entity.name,
                entity_type=entity.entity_type,
                mentions=entity.mentions,
                papers=list(entity.papers),
                aliases=list(entity.aliases),
                metadata=entity.metadata
            )

        self.entity_index[node_id] = entity

    def add_causal_relation(self, relation: CausalRelation):
        """Add a causal relationship to the knowledge graph"""
        source_id = f"entity:concept:{relation.source}"
        target_id = f"entity:concept:{relation.target}"

        # Ensure nodes exist
        for node_id, name in [(source_id, relation.source), (target_id, relation.target)]:
            if not self.graph.has_node(node_id):
                self.graph.add_node(
                    node_id,
                    node_type=NodeType.CONCEPT.value,
                    name=name,
                    papers=[]
                )

        # Calculate edge weight
        weight = self._calculate_edge_weight(relation)

        # Add edge
        self.graph.add_edge(
            source_id,
            target_id,
            edge_type=EdgeType.CAUSES.value,
            relation_type=relation.relation_type,
            weight=weight,
            confidence=relation.confidence,
            paper_id=relation.paper_id,
            evidence_text=relation.evidence_text,
            effect_size=relation.effect_size,
            p_value=relation.p_value,
            domains=[d.value for d in relation.domains]
        )

    def _calculate_edge_weight(self, relation: CausalRelation) -> float:
        """
        Calculate edge weight based on evidence strength
        Higher weight = stronger evidence
        """
        weight = relation.confidence

        # Boost for better study designs
        study_design_weights = {
            'meta-analysis': 2.0,
            'systematic_review': 1.8,
            'rct': 1.5,
            'cohort': 1.2,
            'case-control': 1.0,
            'observational': 0.8
        }
        if relation.study_design:
            weight *= study_design_weights.get(relation.study_design.lower(), 1.0)

        # Boost for larger sample sizes
        if relation.sample_size:
            if relation.sample_size > 10000:
                weight *= 1.5
            elif relation.sample_size > 1000:
                weight *= 1.3
            elif relation.sample_size > 100:
                weight *= 1.1

        # Adjust for effect size
        if relation.effect_size:
            if abs(relation.effect_size) > 0.8:
                weight *= 1.4
            elif abs(relation.effect_size) > 0.5:
                weight *= 1.2
            elif abs(relation.effect_size) > 0.2:
                weight *= 1.1

        return min(weight, 10.0)  # Cap at 10

    def find_causal_path(
        self,
        source: str,
        target: str,
        max_hops: int = 5
    ) -> List[List[str]]:
        """Find causal paths between two concepts"""
        source_id = f"entity:concept:{source}"
        target_id = f"entity:concept:{target}"

        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return []

        try:
            # Find all simple paths up to max_hops
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_hops
            ))

            # Sort by total edge weight
            paths_with_weights = []
            for path in paths:
                total_weight = 0
                for i in range(len(path) - 1):
                    edges = self.graph.get_edge_data(path[i], path[i+1])
                    if edges:
                        # Get max weight among multiple edges
                        max_weight = max(e.get('weight', 0) for e in edges.values())
                        total_weight += max_weight
                paths_with_weights.append((path, total_weight))

            # Sort by weight (descending)
            paths_with_weights.sort(key=lambda x: x[1], reverse=True)

            return [path for path, _ in paths_with_weights]

        except nx.NetworkXNoPath:
            return []

    def query_influences(
        self,
        concept: str,
        max_depth: int = 2,
        domains: Optional[List[KnowledgeDomain]] = None
    ) -> Dict[str, Any]:
        """Find what influences a concept (incoming edges)"""
        concept_id = f"entity:concept:{concept}"

        if not self.graph.has_node(concept_id):
            return {"error": "Concept not found"}

        influences = []

        # Get all predecessors (things that influence this concept)
        for predecessor in self.graph.predecessors(concept_id):
            edges = self.graph.get_edge_data(predecessor, concept_id)

            for edge_data in edges.values():
                # Filter by domain if specified
                if domains:
                    edge_domains = [KnowledgeDomain(d) for d in edge_data.get('domains', [])]
                    if not any(d in domains for d in edge_domains):
                        continue

                influences.append({
                    'source': self.graph.nodes[predecessor]['name'],
                    'relation_type': edge_data.get('relation_type'),
                    'weight': edge_data.get('weight'),
                    'confidence': edge_data.get('confidence'),
                    'evidence': edge_data.get('paper_id'),
                    'domains': edge_data.get('domains', [])
                })

        # Sort by weight
        influences.sort(key=lambda x: x['weight'], reverse=True)

        return {
            'concept': concept,
            'influences': influences,
            'count': len(influences)
        }

    def cross_domain_effects(
        self,
        source_domains: List[KnowledgeDomain],
        target_domains: List[KnowledgeDomain],
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find causal relationships crossing domain boundaries"""
        cross_domain_edges = []

        for edge in self.graph.edges(data=True):
            source, target, data = edge

            # Get domains for source and target nodes
            source_node_domains = set(self.graph.nodes[source].get('domains', []))
            target_node_domains = set(self.graph.nodes[target].get('domains', []))

            # Check if source is in source_domains and target is in target_domains
            source_match = any(d.value in source_node_domains for d in source_domains)
            target_match = any(d.value in target_node_domains for d in target_domains)

            if source_match and target_match and data.get('confidence', 0) >= min_confidence:
                cross_domain_edges.append({
                    'source': self.graph.nodes[source]['name'],
                    'target': self.graph.nodes[target]['name'],
                    'relation_type': data.get('relation_type'),
                    'weight': data.get('weight'),
                    'confidence': data.get('confidence'),
                    'evidence': data.get('paper_id'),
                    'source_domains': list(source_node_domains),
                    'target_domains': list(target_node_domains)
                })

        # Sort by weight
        cross_domain_edges.sort(key=lambda x: x['weight'], reverse=True)

        return cross_domain_edges

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'papers': len(self.paper_index),
            'entities': len(self.entity_index),
            'domains': {domain.value: len(nodes) for domain, nodes in self.domain_index.items()},
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1)
        }

    def save_to_disk(self):
        """Save knowledge graph to disk"""
        if not self.graph_path:
            raise ValueError("No graph path specified")

        self.graph_path.mkdir(parents=True, exist_ok=True)

        # Save graph
        nx.write_gpickle(self.graph, self.graph_path / "graph.gpickle")

        # Save indices
        with open(self.graph_path / "paper_index.pkl", 'wb') as f:
            pickle.dump(self.paper_index, f)

        with open(self.graph_path / "entity_index.pkl", 'wb') as f:
            pickle.dump(self.entity_index, f)

        # Save stats
        with open(self.graph_path / "stats.json", 'w') as f:
            json.dump(self.get_stats(), f, indent=2)

    def load_from_disk(self):
        """Load knowledge graph from disk"""
        if not self.graph_path or not self.graph_path.exists():
            raise ValueError("Graph path does not exist")

        # Load graph
        self.graph = nx.read_gpickle(self.graph_path / "graph.gpickle")

        # Load indices
        with open(self.graph_path / "paper_index.pkl", 'rb') as f:
            self.paper_index = pickle.load(f)

        with open(self.graph_path / "entity_index.pkl", 'rb') as f:
            self.entity_index = pickle.load(f)

        # Rebuild domain index
        self.domain_index = defaultdict(set)
        for node_id, data in self.graph.nodes(data=True):
            for domain_str in data.get('domains', []):
                try:
                    domain = KnowledgeDomain(domain_str)
                    self.domain_index[domain].add(node_id)
                except ValueError:
                    pass

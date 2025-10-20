"""
C-LIGHT Core Module
Standalone cognitive science RAG system - no external dependencies
"""

from .base_types import (
    Paper,
    KnowledgeDomain,
    CausalRelation,
    Entity,
    ProcessingStatus
)

from .knowledge_graph import (
    CLightKnowledgeGraph,
    Node,
    Edge,
    NodeType,
    EdgeType
)

from .rag_system import (
    CLightRAG,
    QueryResult,
    Evidence
)

__all__ = [
    'Paper',
    'KnowledgeDomain',
    'CausalRelation',
    'Entity',
    'ProcessingStatus',
    'CLightKnowledgeGraph',
    'Node',
    'Edge',
    'NodeType',
    'EdgeType',
    'CLightRAG',
    'QueryResult',
    'Evidence'
]

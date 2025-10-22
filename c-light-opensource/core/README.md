# Core Module

## Overview
The core module contains fundamental components of the C-LIGHT system, including data structures, knowledge graph management, paper processing pipeline, and the RAG (Retrieval-Augmented Generation) system.

## Components

### 1. `base_types.py`
- **Purpose**: Define fundamental data structures and enums
- **Key Classes**:
  - `Paper`: Base class for scientific papers
  - `ProcessingStatus`: Enum for tracking paper processing states
  - `KnowledgeDomain`: Classification of research domains
  - `CausalRelation`: Structure for causal relationships
  - `Entity`: Named entity representation

### 2. `knowledge_graph.py`
- **Purpose**: Build and manage the scientific knowledge graph
- **Features**:
  - Graph construction from extracted entities and relationships
  - Path finding for causal chains
  - Subgraph extraction for specific domains
  - Graph serialization and persistence
  - NetworkX-based implementation

### 3. `paper_processor.py`
- **Purpose**: Main pipeline for processing scientific papers
- **Processing Steps**:
  1. Paper ingestion and validation
  2. Text extraction and preprocessing
  3. Entity recognition
  4. Causal relationship extraction
  5. Knowledge graph integration
  6. Embedding generation
  7. Index updating

### 4. `rag_system.py`
- **Purpose**: Retrieval-Augmented Generation for question answering
- **Architecture**:
  - **NOT DPR-based**: Uses BGE embeddings with cosine similarity
  - Dense vector retrieval with semantic search
  - Evidence aggregation from multiple sources
  - Confidence scoring
- **Key Classes**:
  - `CLightRAG`: Main RAG interface
  - `QueryResult`: Structured query responses
  - `Evidence`: Supporting evidence for answers

## Data Flow

```
Paper Input
    ↓
Paper Processor
    ↓
    ├─→ Entity Extraction
    ├─→ Causal Extraction
    └─→ Embedding Generation
         ↓
    Knowledge Graph
         ↓
    RAG System → Query Results
```

## Usage Examples

### Processing a Paper
```python
from core import PaperProcessor, Paper

processor = PaperProcessor()
paper = Paper(
    doi="10.1234/example",
    title="Example Paper",
    abstract="...",
    authors=["Author1", "Author2"]
)

processed = processor.process(paper)
```

### Querying the Knowledge Base
```python
from core import CLightRAG, CLightKnowledgeGraph

kg = CLightKnowledgeGraph()
rag = CLightRAG(knowledge_graph=kg)

result = rag.query(
    "What causes cognitive decline in aging?",
    max_results=10,
    min_confidence=0.7
)

for evidence in result.evidence:
    print(f"Source: {evidence.paper_title}")
    print(f"Evidence: {evidence.text}")
    print(f"Confidence: {evidence.confidence}")
```

### Building Knowledge Graph
```python
from core import CLightKnowledgeGraph

kg = CLightKnowledgeGraph()

# Add entities
kg.add_entity("dopamine", type="neurotransmitter")
kg.add_entity("reward_learning", type="cognitive_process")

# Add causal relationship
kg.add_causal_relation(
    cause="dopamine",
    effect="reward_learning",
    confidence=0.85,
    paper_id="paper_123"
)

# Find causal paths
paths = kg.find_causal_paths("stress", "memory_impairment")
```

## Key Features

1. **Modular Design**: Each component can be used independently
2. **Scalable Processing**: Batch processing capabilities
3. **Domain-Aware**: Specialized handling for different research domains
4. **Evidence-Based**: All knowledge linked to source papers
5. **Confidence Scoring**: Probabilistic confidence for all extractions

## Performance Characteristics

- Paper processing: ~2-5 seconds per paper
- Graph queries: < 100ms for most operations
- RAG queries: 200-500ms depending on retrieval size
- Supports graphs with millions of nodes and edges

## Dependencies
- NetworkX: Graph operations
- NumPy: Numerical computations
- Sentence Transformers: Embedding generation
- SQLite: Metadata storage
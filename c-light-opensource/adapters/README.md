# Adapters Module

## Overview
The adapters module provides integration bridges between C-LIGHT and external systems, particularly the CANDLE framework for cancer research AI/ML applications.

## Components

### `candle_adapter.py`
- **Purpose**: Seamless integration with CANDLE (CANcer Distributed Learning Environment)
- **Key Features**:
  - Bidirectional data exchange between C-LIGHT and CANDLE
  - Model compatibility layer
  - Knowledge graph synchronization
  - Paper processing pipeline integration

## Architecture

The adapter pattern allows C-LIGHT to:
1. Import cancer research papers from CANDLE's data sources
2. Export processed knowledge to CANDLE's knowledge bases
3. Share embedding models and vector stores
4. Synchronize research findings across both systems

## Usage Example

```python
from adapters import CandelAdapter

# Initialize adapter
adapter = CandelAdapter(
    candle_path="/mnt/c/CANDLE/CANDLE_CORE",
    c_light_config={...}
)

# Import papers from CANDLE
papers = adapter.import_papers(domain="cancer_research")

# Export knowledge graph to CANDLE
adapter.export_knowledge_graph(graph, format="neo4j")
```

## Integration Points

1. **Data Import**:
   - Papers and publications
   - Research datasets
   - Clinical trial data

2. **Knowledge Export**:
   - Causal relationships
   - Entity mappings
   - Research insights

3. **Model Sharing**:
   - Embedding models
   - Classification models
   - NER models

## Benefits

- **Cross-Domain Learning**: Leverage cancer research insights for broader cognitive science applications
- **Resource Sharing**: Reuse expensive computations (embeddings, parsing)
- **Unified Pipeline**: Single processing pipeline for multiple research domains
- **Scalability**: Distributed processing capabilities from CANDLE

## Configuration

The adapter can be configured via environment variables or config files:

```yaml
candle_adapter:
  candle_root: "/mnt/c/CANDLE/CANDLE_CORE"
  sync_interval: 3600  # seconds
  batch_size: 100
  enable_bidirectional: true
```

## Dependencies
- CANDLE framework
- PyTorch (for model compatibility)
- NetworkX (for graph operations)
- SQLAlchemy (for database abstraction)
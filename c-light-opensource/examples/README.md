# Examples Module

## Overview
The examples module provides practical demonstrations and reference implementations for using the C-LIGHT system, including integration with CANDLE, complete workflow examples, and the SEAL learning system demo.

## Example Scripts

### 1. `complete_workflow.py`
- **Purpose**: End-to-end demonstration of the C-LIGHT pipeline
- **Demonstrates**:
  - Paper harvesting from arXiv
  - Entity and causal relationship extraction
  - Knowledge graph construction
  - Dual-system RAG queries
  - Performance comparison

**Key Features:**
```python
# Complete pipeline example
workflow = CLightWorkflow()

# Step 1: Harvest papers
papers = workflow.harvest_papers(
    categories=["cs.AI", "q-bio.NC"],
    days_back=30
)

# Step 2: Process and extract knowledge
knowledge = workflow.process_papers(papers)

# Step 3: Build knowledge graph
graph = workflow.build_knowledge_graph(knowledge)

# Step 4: Query both systems
result_a = workflow.query_system_a("What causes synaptic plasticity?")
result_b = workflow.query_system_b("What causes synaptic plasticity?")

# Step 5: Compare results
comparison = workflow.compare_systems(result_a, result_b)
```

### 2. `seal_learning_demo.py`
- **Purpose**: Showcase the SEAL (Self-Evolving Associative Learning) system
- **Demonstrates**:
  - Pattern learning from user feedback
  - Weight optimization over time
  - Active learning strategies
  - Performance improvements through usage

**Key Concepts:**
```python
# SEAL system demonstration
seal = SEALSystem()

# Initial query (cold start)
result1 = seal.query("neural mechanisms of memory")
print(f"Initial confidence: {result1.confidence}")

# Provide feedback
seal.provide_feedback(result1, relevance_score=0.8)

# Query again (system has learned)
result2 = seal.query("neural mechanisms of memory")
print(f"After learning: {result2.confidence}")

# Show weight evolution
seal.visualize_weight_evolution()
```

### 3. `candle_integration_example.py`
- **Purpose**: Integration with CANDLE framework for cancer research
- **Demonstrates**:
  - Cross-domain knowledge transfer
  - Shared embedding spaces
  - Unified processing pipeline
  - Cancer-specific pattern recognition

**Integration Flow:**
```python
# CANDLE integration
from adapters import CandelAdapter

adapter = CandelAdapter()

# Import cancer research papers
cancer_papers = adapter.import_from_candle(
    domain="oncology",
    subdomains=["immunotherapy", "drug_discovery"]
)

# Process with C-LIGHT pipeline
processed = c_light.process(cancer_papers)

# Export enhanced knowledge back to CANDLE
adapter.export_to_candle(
    knowledge_graph=processed.graph,
    embeddings=processed.embeddings
)

# Cross-domain query
result = adapter.cross_domain_query(
    "How do cognitive processes relate to cancer treatment outcomes?"
)
```

## Running the Examples

### Prerequisites
```bash
# Install required dependencies
pip install -r requirements_full.txt

# Set up storage paths
export CLIGHT_STORAGE="/mnt/nvme/c-light"
export CLIGHT_HDD="/mnt/hdd/c-light"
```

### Basic Usage
```bash
# Run complete workflow
python examples/complete_workflow.py

# Run SEAL demo with visualization
python examples/seal_learning_demo.py --visualize

# Run CANDLE integration
python examples/candle_integration_example.py --candle-path /path/to/candle
```

## Example Outputs

### Knowledge Graph Visualization
The examples can generate various visualizations:
- Causal relationship networks
- Entity co-occurrence graphs
- Domain interaction maps
- Learning curve plots

### Performance Metrics
Example output from system comparison:
```
System A (LLM-RAG):
  Response Time: 450ms
  Precision: 0.82
  Recall: 0.75
  F1 Score: 0.78

System B (SEAL):
  Response Time: 120ms
  Precision: 0.79
  Recall: 0.83
  F1 Score: 0.81
  Learning Rate: +2.3%/epoch
```

## Customization

### Adding Custom Domains
```python
# Define custom domain patterns
custom_domain = {
    "name": "neuroscience",
    "categories": ["q-bio.NC", "physics.bio-ph"],
    "keywords": ["neural", "brain", "cognition"],
    "extractors": [CustomNeuralExtractor()]
}

workflow.add_domain(custom_domain)
```

### Custom Extractors
```python
class CustomExtractor(BaseExtractor):
    def extract(self, text):
        # Custom extraction logic
        entities = self.extract_entities(text)
        relations = self.extract_relations(text)
        return {"entities": entities, "relations": relations}

workflow.register_extractor(CustomExtractor())
```

## Best Practices

1. **Start Small**: Begin with the basic workflow example
2. **Monitor Performance**: Use built-in metrics collection
3. **Provide Feedback**: Help SEAL system learn through feedback
4. **Domain Focus**: Start with a specific domain before expanding
5. **Resource Management**: Monitor storage and memory usage

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size in configuration
   - Enable disk-based caching

2. **Slow Processing**:
   - Check GPU availability for embeddings
   - Use async processing for I/O operations

3. **Poor Results**:
   - Verify paper quality and relevance
   - Adjust confidence thresholds
   - Provide more feedback to SEAL

## Advanced Examples

For more advanced use cases, see:
- Multi-domain knowledge fusion
- Temporal analysis of research trends
- Contradiction detection in literature
- Meta-analysis automation

## Dependencies
- All core C-LIGHT modules
- Matplotlib/Seaborn for visualization
- Jupyter (optional) for interactive demos
- CANDLE framework (for integration example)
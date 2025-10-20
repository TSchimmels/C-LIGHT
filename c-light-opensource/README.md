# C-LIGHT: Cognitive Literature Intelligence & Graph-based Hypothesis Testing

Open-source RAG system for cognitive and behavioral science research, featuring dual architectures for performance comparison.

## Overview

C-LIGHT is a specialized RAG (Retrieval-Augmented Generation) system designed for cognitive science research. It features **two parallel systems** that can be compared:

- **System A**: Traditional LLM-based RAG using Mixtral-8x7B
- **System B**: SEAL (Self-Enhanced Active Learning) self-learning RAG

### Key Features

✅ **Full PDF Processing** - Extracts complete paper content, not just abstracts
✅ **Explicit Citations** - Every answer cites specific papers, sections, and page numbers
✅ **Knowledge Graph** - Explicit causal relationship tracking using NetworkX
✅ **Self-Learning** - System B improves continuously from user feedback
✅ **Active Learning** - Intelligently selects which papers to process
✅ **CANDLE Integration** - Clean adapter for proprietary CANDLE system
✅ **Dual System Comparison** - Built-in evaluation framework

## Architecture

### System A: Mixtral LLM-Based RAG

Traditional RAG approach using state-of-the-art open-source LLM:

```
Papers → PDF Processing → Chunking → Embeddings (BGE-large) →
FAISS Vector Store → Mixtral-8x7B-Instruct → Answer with Citations
```

**Strengths**:
- Excellent from day 1 (no training needed)
- Natural language generation
- Handles diverse question types
- Fast to deploy

**Components**:
- `system_a/llm_rag.py` - Main RAG implementation
- `system_a/embedding.py` - BGE-large-en-v1.5 embeddings
- `system_a/vector_store.py` - FAISS storage

### System B: SEAL Self-Learning RAG

Novel approach that learns from feedback:

```
Papers → Pattern Extraction → Knowledge Graph → Query Engine →
Feedback Loop → Pattern Learning + Edge Optimization + Active Learning
```

**Strengths**:
- Improves over time with use
- Explicit causal reasoning
- Transparent decision paths
- Smart paper selection
- Domain-focused learning

**Components**:
- `system_b/seal_rag.py` - Main SEAL system
- `system_b/feedback_system.py` - Tracks query success/failure
- `system_b/pattern_learner.py` - Discovers new causal patterns
- `system_b/weight_optimizer.py` - Reinforcement learning for edges
- `system_b/active_learning.py` - Smart paper prioritization

See **[SEAL_SYSTEM_GUIDE.md](SEAL_SYSTEM_GUIDE.md)** for complete documentation on System B.

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended: 24GB+ VRAM for Mixtral)
- 64GB+ RAM recommended

### Install Dependencies

```bash
pip install -r requirements_full.txt
```

Key dependencies:
- `torch` - PyTorch for LLM
- `transformers` - Mixtral-8x7B
- `sentence-transformers` - BGE embeddings
- `faiss-gpu` - Vector search
- `networkx` - Knowledge graph
- `arxiv` - Paper harvesting
- `PyPDF2` - PDF text extraction

## Quick Start

### 1. Complete Workflow

Run the full pipeline (harvesting → processing → querying → learning):

```bash
python examples/complete_workflow.py
```

This demonstrates:
- Harvesting papers from ArXiv
- Processing with both systems
- Comparing results
- Providing feedback
- Analyzing improvement

### 2. SEAL Learning Demo

See System B learn and improve:

```bash
python examples/seal_learning_demo.py
```

Shows:
- Baseline performance
- Learning from feedback
- Pattern discovery
- Edge weight optimization
- Performance improvement over time

### 3. CANDLE Integration

See how to integrate with CANDLE:

```bash
python examples/candle_integration_example.py
```

## Usage

### Basic Query Example

```python
from system_a.llm_rag import MixtralRAG
from system_b.seal_rag import SEALBasedRAG
from core.knowledge_graph import CLightKnowledgeGraph

# Initialize knowledge graph (shared by both systems)
kg = CLightKnowledgeGraph(graph_path="/path/to/graph")

# Initialize System A (Mixtral)
system_a = MixtralRAG(
    knowledge_graph=kg,
    storage_path="/path/to/system_a"
)

# Initialize System B (SEAL)
system_b = SEALBasedRAG(
    knowledge_graph=kg,
    storage_path="/path/to/system_b"
)

# Add papers
from core.base_types import Paper

paper = Paper(
    paper_id="arxiv:2401.12345",
    title="Dopamine and Motivation",
    authors=["Smith, J."],
    abstract="We studied dopamine's role...",
)

system_a.add_paper(paper, pdf_path="/path/to/paper.pdf")
system_b.add_paper(paper, pdf_path="/path/to/paper.pdf")

# Query both systems
question = "How does dopamine affect motivation?"

result_a = system_a.query(question)
result_b = system_b.query(question)

print("System A answer:", result_a.answer)
print("System B answer:", result_b.answer)

# Provide feedback to System B (for learning)
system_b.provide_feedback(
    query_id=result_b.metadata['query_id'],
    rating=5,  # 1-5 scale
    correct_answer=None  # Optional correction
)
```

### System Comparison

```python
from evaluation.dual_system_evaluator import DualSystemEvaluator

# Initialize evaluator
evaluator = DualSystemEvaluator(
    system_a=system_a,
    system_b=system_b,
    storage_path="/path/to/evaluation"
)

# Compare on a query
comparison = evaluator.compare_query(
    question="Does dopamine increase motivation?"
)

print(f"System A: {comparison.system_a_answer}")
print(f"System B: {comparison.system_b_answer}")

# Add user ratings
evaluator.add_user_ratings(
    query_id=comparison.query_id,
    system_a_rating=4,
    system_b_rating=5
)

# Generate comparison report
report = evaluator.generate_report(
    output_path="/path/to/report.md"
)
```

### CANDLE Integration

```python
from adapters.candle_adapter import CANDLEAdapter

# Initialize adapter
adapter = CANDLEAdapter(
    storage_path="/path/to/papers",
    db_path="/path/to/doi_database.db",
    graph_path="/path/to/knowledge_graph",
    candle_config={'project': 'CANDLE'}
)

# Query
result = adapter.query("Your question")

# Add CANDLE's own papers
candle_paper = Paper(...)
adapter.add_paper(candle_paper)

# Access knowledge graph for CANDLE algorithms
kg = adapter.get_knowledge_graph()
# Run proprietary CANDLE algorithms on kg
```

See `examples/candle_integration_example.py` for complete integration patterns.

## Project Structure

```
c-light-opensource/
├── core/                          # Core data structures and logic
│   ├── base_types.py              # Paper, CausalRelation, QueryResult
│   ├── knowledge_graph.py         # NetworkX-based graph
│   ├── paper_processor.py         # PDF extraction and chunking
│   └── rag_system.py              # Base RAG interface
│
├── harvesting/                    # Paper acquisition
│   ├── arxiv_harvester.py         # ArXiv API integration
│   ├── doi_database.py            # Deduplication with RocksDB
│   ├── doi_monitor.py             # Citation tracking
│   └── paper_recall.py            # Reference discovery
│
├── extractors/                    # Information extraction
│   ├── causal_extractor.py        # Pattern-based causal extraction
│   └── entity_extractor.py        # Named entity recognition
│
├── system_a/                      # Mixtral LLM-based RAG
│   ├── llm_rag.py                 # Main System A implementation
│   ├── embedding.py               # BGE embeddings
│   └── vector_store.py            # FAISS storage
│
├── system_b/                      # SEAL self-learning RAG
│   ├── seal_rag.py                # Main SEAL system
│   ├── feedback_system.py         # Query rating tracking
│   ├── pattern_learner.py         # Discovers causal patterns
│   ├── weight_optimizer.py        # Edge weight RL
│   └── active_learning.py         # Smart paper selection
│
├── evaluation/                    # System comparison
│   └── dual_system_evaluator.py  # A/B testing framework
│
├── adapters/                      # External integrations
│   └── candle_adapter.py          # CANDLE integration
│
├── examples/                      # Example scripts
│   ├── complete_workflow.py       # Full pipeline demo
│   ├── seal_learning_demo.py      # SEAL learning demo
│   └── candle_integration_example.py
│
└── docs/                          # Documentation
    ├── SEAL_SYSTEM_GUIDE.md       # Complete SEAL guide
    ├── ARCHITECTURE_DUAL_SYSTEM.md
    └── PAPER_CITATION_SYSTEM.md
```

## Documentation

- **[SEAL_SYSTEM_GUIDE.md](SEAL_SYSTEM_GUIDE.md)** - Complete guide to System B
  - How SEAL works
  - All parameters and tuning
  - Learning loop explained
  - Monitoring and analysis

- **[ARCHITECTURE_DUAL_SYSTEM.md](ARCHITECTURE_DUAL_SYSTEM.md)** - Dual system design
  - System A vs System B comparison
  - Expected performance trajectories
  - Evaluation methodology

- **[PAPER_CITATION_SYSTEM.md](PAPER_CITATION_SYSTEM.md)** - Citation tracking
  - PDF processing pipeline
  - Section detection
  - Citation format

## Comparison: System A vs System B

| Aspect | System A (Mixtral) | System B (SEAL) |
|--------|-------------------|-----------------|
| **Day 1 Performance** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Month 3 Performance** | ⭐⭐⭐⭐⭐ Same | ⭐⭐⭐⭐⭐ Excellent (learned) |
| **Learning** | ❌ Static | ✅ Continuous |
| **Causality** | ⚠️ Implicit (LLM infers) | ✅ Explicit (graph-based) |
| **Interpretability** | ❌ Black box | ✅ Transparent paths |
| **Feedback Required** | ❌ No | ✅ Yes (for learning) |
| **Paper Selection** | 🔀 Random/citation-based | 🎯 Active learning |
| **GPU Requirements** | 💰 High (24GB+ for Mixtral) | 💵 Low (CPU-only capable) |

**Hypothesis**: System B will surpass System A after 2-3 months of active use with feedback.

**Use `evaluation/dual_system_evaluator.py` to test this hypothesis!**

## Key Concepts

### Upstream Causal Chain

C-LIGHT focuses on causal relationships across levels:

```
Molecular → Neural → Cognitive → Behavioral → Social → Cultural
```

Example: "How does serotonin transporter gene variation affect political ideology?"
```
Gene (5-HTTLPR) → Serotonin → Anxiety → Threat sensitivity →
Conservative ideology
```

### Full Paper Processing

Unlike many RAG systems that only use abstracts, C-LIGHT processes complete PDFs:

1. Extract full text with PyPDF2
2. Detect sections (Introduction, Methods, Results, Discussion)
3. Chunk into ~1000 word segments with overlap
4. Track source: section, page numbers, paper metadata
5. Every answer cites: "Smith et al., 'Title', Results section, pp. 5-7"

See `core/paper_processor.py` for implementation.

### SEAL Learning Loop

System B's learning process:

```
1. User queries → System finds path in knowledge graph
2. System answers using current edge weights
3. User rates answer (1-5)
4. Rating ≥ 4: Reinforce edges in path (good!)
5. Rating ≤ 2: Penalize edges in path (bad!)
6. User provides correction: Discover new patterns
7. Update domain priorities for paper selection
8. Future queries benefit from learned weights
```

See **SEAL_SYSTEM_GUIDE.md** for complete details.

## CANDLE Integration

C-LIGHT is designed to integrate cleanly with the proprietary CANDLE system:

**What's Open Source (C-LIGHT)**:
- Paper harvesting and processing
- Knowledge graph construction
- Both RAG systems (Mixtral and SEAL)
- Evaluation framework

**What Stays Proprietary (CANDLE)**:
- Advanced algorithms
- Classified data sources
- Specialized workflows
- Custom analytics

**Integration Points**:
- `adapters/candle_adapter.py` - Clean adapter interface
- CANDLE can add its own papers to C-LIGHT
- CANDLE can query both RAG systems
- CANDLE can access knowledge graph for its algorithms
- No CANDLE dependencies in C-LIGHT core

See `examples/candle_integration_example.py` for detailed integration patterns.

## Evaluation

### Running Comparisons

```python
from evaluation.dual_system_evaluator import DualSystemEvaluator

evaluator = DualSystemEvaluator(system_a, system_b)

# Run 100 queries with ratings
for question in test_questions:
    comparison = evaluator.compare_query(question)
    # User rates both answers
    evaluator.add_user_ratings(
        query_id=comparison.query_id,
        system_a_rating=rating_a,
        system_b_rating=rating_b
    )

# Analyze improvement over time
improvement = evaluator.analyze_improvement_over_time(window_size=50)

print(f"System A change: {improvement['system_a']['change']:+.2f}")
print(f"System B change: {improvement['system_b']['change']:+.2f}")
print(f"Hypothesis confirmed: {improvement['hypothesis_confirmed']}")

# Generate report
report = evaluator.generate_report(output_path="comparison_report.md")
```

### Metrics

- **Win Rate**: How often each system gets higher rating
- **Average Rating**: Mean user rating (1-5)
- **Latency**: Response time
- **Confidence Calibration**: Does confidence match accuracy?
- **Improvement Over Time**: Is System B learning?

## Configuration

### System A Parameters

```python
system_a = MixtralRAG(
    knowledge_graph=kg,
    storage_path="/path",
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    device="cuda",  # or "cpu"
    max_tokens=2000,
    temperature=0.3  # Lower = more deterministic
)
```

### System B Parameters

```python
system_b = SEALBasedRAG(
    knowledge_graph=kg,
    storage_path="/path"
)

# Tune learning parameters
system_b.weight_optimizer = EdgeWeightOptimizer(
    learning_rate=0.01,    # How fast to learn (0.005-0.05)
    decay_rate=0.999       # How long to remember (0.99-0.9999)
)

# Tune active learning priorities
system_b.active_learner._calculate_paper_score = lambda paper: (
    0.4 * domain_score +       # Adjust these weights
    0.3 * novelty_score +
    0.2 * query_relevance +
    0.1 * citation_score
)
```

See **SEAL_SYSTEM_GUIDE.md** section "Parameters & Tuning" for complete tuning guide.

## Performance

### Expected Timeline

**Week 1**:
- System A: 70-80% success rate (excellent from start)
- System B: 50-60% success rate (baseline)

**Month 1**:
- System A: 70-80% (stable)
- System B: 65-70% (learning)

**Month 3**:
- System A: 70-80% (plateaued)
- System B: 75-85% (surpasses System A)

### Resource Requirements

**System A (Mixtral)**:
- GPU: 24GB+ VRAM (48GB recommended for Mixtral-8x7B)
- RAM: 32GB+
- Disk: ~100GB for model + papers

**System B (SEAL)**:
- GPU: Optional (CPU-capable)
- RAM: 16GB+
- Disk: ~10GB for papers + graph

## Contributing

This is the open-source component of CANDLE. Contributions welcome!

Areas for contribution:
- New extraction patterns
- Additional embedding models
- Improved PDF processing
- Enhanced active learning strategies
- Additional evaluation metrics
- Domain-specific adaptations

## License

[Specify License - MIT, Apache 2.0, etc.]

## Citation

If you use C-LIGHT in your research:

```bibtex
@software{clight2025,
  title={C-LIGHT: Cognitive Literature Intelligence \& Graph-based Hypothesis Testing},
  author={[Your Team]},
  year={2025},
  url={https://github.com/[your-org]/c-light-opensource}
}
```

## Contact

- GitHub Issues: [your-repo-url]/issues
- Documentation: See `docs/` directory
- SEAL Guide: `SEAL_SYSTEM_GUIDE.md`

---

**Quick Links**:
- 📚 [SEAL System Guide](SEAL_SYSTEM_GUIDE.md) - Complete System B documentation
- 🏗️ [Architecture](ARCHITECTURE_DUAL_SYSTEM.md) - Dual system design
- 📖 [Paper Citations](PAPER_CITATION_SYSTEM.md) - Citation tracking
- 💻 [Examples](examples/) - Usage examples
- 🔧 [CANDLE Adapter](adapters/candle_adapter.py) - Integration code

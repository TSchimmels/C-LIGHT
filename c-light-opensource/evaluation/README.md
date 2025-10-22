# Evaluation Module

## Overview
The evaluation module provides comprehensive testing and benchmarking capabilities for the C-LIGHT dual-system architecture, comparing System A (traditional LLM-based RAG) against System B (SEAL - Self-Evolving Associative Learning).

## Components

### `dual_system_evaluator.py`
- **Purpose**: Systematic evaluation and comparison of both RAG systems
- **Key Metrics**:
  - Answer accuracy and relevance
  - Retrieval precision and recall
  - Response latency
  - Resource utilization
  - Learning rate (System B specific)
  - Pattern recognition accuracy

## Evaluation Framework

### 1. Benchmark Datasets
- **CogSci-QA**: Cognitive science question-answering pairs
- **Causal-Bench**: Causal reasoning evaluation set
- **BioMed-Retrieval**: Biomedical literature retrieval tasks
- **Custom domain-specific test sets**

### 2. Evaluation Metrics

#### Retrieval Quality
- **Precision@K**: Relevant documents in top K results
- **Recall@K**: Coverage of relevant documents
- **MRR (Mean Reciprocal Rank)**: Ranking quality
- **NDCG (Normalized Discounted Cumulative Gain)**: Graded relevance

#### Answer Quality
- **BLEU Score**: N-gram overlap with reference answers
- **ROUGE Score**: Recall-oriented evaluation
- **BERTScore**: Semantic similarity using BERT embeddings
- **Human evaluation scores** (when available)

#### System Performance
- **Latency**: Query response time
- **Throughput**: Queries per second
- **Memory usage**: RAM and VRAM consumption
- **Cache efficiency**: Hit rates and eviction metrics

### 3. Comparative Analysis

```python
System A (LLM-RAG)          vs         System B (SEAL)
├── Static embeddings                  ├── Dynamic weight learning
├── Fixed retrieval                    ├── Adaptive pattern matching
├── LLM generation                     ├── Associative recall
└── Higher latency                     └── Lower latency
```

## Usage Example

```python
from evaluation import DualSystemEvaluator

# Initialize evaluator
evaluator = DualSystemEvaluator(
    system_a_config={...},
    system_b_config={...},
    benchmark="CogSci-QA"
)

# Run comprehensive evaluation
results = evaluator.evaluate_all(
    test_queries=queries,
    metrics=["precision", "recall", "latency", "bertscore"]
)

# Generate comparison report
report = evaluator.generate_report(results)
print(report.summary())

# Output:
# System A Performance:
#   - Precision@10: 0.82
#   - Recall@10: 0.75
#   - Avg Latency: 450ms
#   - BERTScore: 0.88
#
# System B Performance:
#   - Precision@10: 0.79
#   - Recall@10: 0.83
#   - Avg Latency: 120ms
#   - BERTScore: 0.85
#   - Learning Rate: +2.3% per epoch
```

## Evaluation Scenarios

### 1. Cold Start Evaluation
Tests system performance with no prior learning/caching:
- Initial query performance
- Knowledge base bootstrapping
- First-time retrieval accuracy

### 2. Warm System Evaluation
Tests after system has processed substantial data:
- Cache-hit performance
- Learned pattern effectiveness
- Long-term stability

### 3. Adversarial Testing
Tests robustness against challenging inputs:
- Out-of-distribution queries
- Contradictory information handling
- Noise resistance

### 4. Ablation Studies
Tests contribution of individual components:
- Embedding model variations
- Retrieval strategy modifications
- Knowledge graph impact

## Continuous Monitoring

The evaluator supports real-time monitoring in production:

```python
# Enable continuous evaluation
evaluator.start_monitoring(
    sample_rate=0.1,  # Evaluate 10% of queries
    log_path="/var/log/c-light/eval",
    alert_thresholds={
        "precision": 0.7,  # Alert if precision drops below 70%
        "latency": 1000    # Alert if latency exceeds 1s
    }
)
```

## Visualization

Generate evaluation visualizations:

```python
# Performance comparison plots
evaluator.plot_metrics_comparison()

# Learning curves (System B)
evaluator.plot_learning_curves()

# Latency distribution
evaluator.plot_latency_histogram()

# Retrieval quality heatmap
evaluator.plot_retrieval_heatmap()
```

## Best Practices

1. **Stratified Sampling**: Ensure test sets represent all domains
2. **Cross-Validation**: Use k-fold validation for robust results
3. **Temporal Splits**: Test on papers from different time periods
4. **A/B Testing**: Run parallel evaluations in production
5. **Human-in-the-Loop**: Incorporate expert feedback

## Configuration

```yaml
evaluation:
  benchmarks:
    - CogSci-QA
    - Causal-Bench
  metrics:
    retrieval: [precision, recall, mrr, ndcg]
    generation: [bleu, rouge, bertscore]
    system: [latency, throughput, memory]
  sample_size: 1000
  confidence_level: 0.95
  parallel_workers: 4
```

## Dependencies
- scikit-learn: Metric calculations
- matplotlib/seaborn: Visualization
- pandas: Data analysis
- transformers: BERTScore calculation
- numpy: Statistical computations
# SEAL System: Complete Technical Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Learning Loop](#learning-loop)
5. [Parameters & Tuning](#parameters--tuning)
6. [Usage Guide](#usage-guide)
7. [Monitoring & Analysis](#monitoring--analysis)
8. [Comparison with System A](#comparison-with-system-a)
9. [Advanced Topics](#advanced-topics)

---

## Overview

### What is SEAL?

SEAL (Self-Enhanced Active Learning) is a RAG system that **learns from feedback** instead of relying solely on a large language model. Unlike traditional RAG systems that use static retrieval + fixed LLM, SEAL:

- **Learns new patterns** from user corrections
- **Optimizes edge weights** based on query success/failure
- **Actively selects** which papers to process for maximum learning benefit
- **Improves over time** without retraining base models

### Key Philosophy

> "A system that learns from mistakes is more valuable than one that's merely good from the start."

SEAL embodies this by treating every user interaction as a learning opportunity.

### Why SEAL vs. Traditional RAG?

| Aspect | Traditional RAG (System A) | SEAL (System B) |
|--------|---------------------------|-----------------|
| **Learning** | No learning after deployment | Continuous learning from feedback |
| **Causality** | LLM infers implicitly | Explicit knowledge graph |
| **Interpretability** | Black box | Transparent reasoning paths |
| **Adaptation** | Requires retraining | Self-adjusts in production |
| **Paper Selection** | Random or citation-based | Active learning based on needs |
| **Edge Reliability** | Static | Reinforcement learning |

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      SEAL RAG SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Papers     │───▶│   Pattern    │───▶│  Knowledge   │ │
│  │              │    │   Learner    │    │    Graph     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                             │                    │          │
│                             ▼                    ▼          │
│                      Causal Relations      Weighted Edges   │
│                             │                    │          │
│                             └────────┬───────────┘          │
│                                      │                      │
│  ┌──────────────────────────────────▼───────────────────┐  │
│  │               QUERY ENGINE                           │  │
│  │  - Parse question                                    │  │
│  │  - Find paths using learned weights                 │  │
│  │  - Calculate confidence from edge reliability       │  │
│  │  - Build answer with citations                      │  │
│  └──────────────────────────────────┬───────────────────┘  │
│                                      │                      │
│                                      ▼                      │
│                                   Answer                    │
│                                      │                      │
│                                      ▼                      │
│  ┌──────────────────────────────────┴───────────────────┐  │
│  │               LEARNING LOOP                          │  │
│  │                                                       │  │
│  │  1. User rates answer (1-5)                         │  │
│  │  2. Update edge weights (reinforce/penalize)        │  │
│  │  3. Learn new patterns (if correction provided)     │  │
│  │  4. Update domain priorities                        │  │
│  │                                                       │  │
│  └──────────────────────────────────┬───────────────────┘  │
│                                      │                      │
│                                      ▼                      │
│  ┌──────────────────────────────────▼───────────────────┐  │
│  │            ACTIVE LEARNING                           │  │
│  │  - Identify weak domains                            │  │
│  │  - Prioritize novel papers                          │  │
│  │  - Target failed query topics                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
system_b/
├── seal_rag.py              # Main integration - USE THIS
├── feedback_system.py       # Tracks query ratings
├── pattern_learner.py       # Discovers new causal patterns
├── weight_optimizer.py      # RL for edge weights
└── active_learning.py       # Smart paper selection
```

---

## Core Components

### 1. Feedback Collection (`feedback_system.py`)

**Purpose**: Track which queries succeed and which fail.

**Key Classes**:

```python
@dataclass
class QueryFeedback:
    query_id: str
    timestamp: datetime
    question: str
    answer: str
    rating: int  # 1-5 (1=terrible, 5=excellent)
    correct_answer: Optional[str]  # User can provide correction
    papers_retrieved: List[str]
    causal_path: List[str]
    confidence: float
    domains: List[str]
```

**What it does**:
1. Records every query with its answer
2. Waits for user rating
3. Stores rating + optional correction
4. Provides analytics on success/failure patterns

**Storage**: JSON files in `/mnt/nvme/c-light/seal/feedback/{query_id}.json`

**Key Methods**:
- `record_query()` - Save query for feedback
- `add_feedback()` - User provides rating
- `get_failed_queries()` - Queries with rating ≤ 2
- `get_successful_queries()` - Queries with rating ≥ 4
- `analyze_patterns()` - Find weak domains, calibration issues

---

### 2. Pattern Learning (`pattern_learner.py`)

**Purpose**: Discover new causal extraction patterns from user feedback.

**How it works**:

1. **Base Patterns**: Starts with 9 hand-coded patterns:
   - `causes`, `leads to`, `results in` (strong causal)
   - `increases`, `decreases`, `enhances` (moderate)
   - `affects`, `modulates` (weak)

2. **Pattern Discovery**:
   - When user provides correct answer
   - Extract concepts from correction
   - Find those concepts co-occurring in papers
   - Extract connecting words between concepts
   - Promote to pattern if seen 3+ times

3. **Pattern Evolution**:
   - Each pattern tracks successes/failures
   - Confidence = `(successes + 1) / (total + 2)` (Bayesian smoothing)
   - Bad patterns naturally decay over time

**Example Learning**:
```
User question: "Does dopamine affect motivation?"
System answer: "No clear relationship found"
User rating: 1 (terrible)
User correction: "Dopamine enhances motivation through reward circuits"

→ System finds: "dopamine" and "motivation" in Paper X
→ Extract connector: "enhances...through"
→ After 3 occurrences, creates pattern:
   learned_enhances_through:
     regex: r'(\w+(?:\s+\w+){0,3})\s+enhances\s+(\w+(?:\s+\w+){0,3})\s+through'
     relation_type: 'increases'
     confidence: 0.5
```

**Key Parameters**:

```python
class PatternLearner:
    def __init__(self, storage_path: str = "/mnt/nvme/c-light/seal/patterns"):
        # Where learned patterns are saved

    def _promote_candidate_pattern(self, connector: str):
        # Requires 3 occurrences before promotion
        if self.candidate_patterns[connector] >= 3:
            # Create new pattern
```

**Tuning**:
- **Promotion threshold**: Change `>= 3` to require more/fewer examples
- **Initial confidence**: New patterns start at 0.5 (neutral)
- **Base pattern confidence**: Hand-coded patterns have fixed confidence (0.5-0.9)

**Storage**: `/mnt/nvme/c-light/seal/patterns/learned_patterns.json`

---

### 3. Edge Weight Optimization (`weight_optimizer.py`)

**Purpose**: Learn which knowledge graph edges are reliable through reinforcement learning.

**How it works**:

1. **Initial State**:
   - All edges have base weights from extraction (0-10)
   - No learned adjustments

2. **Query Uses Edges**:
   - System finds path: A → B → C
   - Uses current weights to rank paths

3. **User Rates Answer**:
   - Rating ≥ 4: **Reinforce** all edges in path
   - Rating ≤ 2: **Penalize** all edges in path

4. **Weight Update**:
   ```python
   if success:
       adjustment[edge] += learning_rate * (rating / 5.0)
   else:
       adjustment[edge] -= learning_rate * ((5 - rating) / 4.0)
   ```

5. **Future Queries**:
   - Good edges get higher weights → more likely to be used
   - Bad edges get lower weights → less likely to be used

**Example**:
```
Initial:
  dopamine → motivation: base_weight=1.0, adjustment=0.0

Query 1: "Does dopamine affect motivation?"
Answer uses: dopamine → motivation
User rating: 5 (excellent)
→ adjustment += 0.01 * (5/5) = 0.01
→ new effective weight = 1.01

Query 2: Same question
Answer uses same edge
User rating: 4 (good)
→ adjustment += 0.01 * (4/5) = 0.008
→ new effective weight = 1.018

After 50 successful uses:
→ adjustment ≈ 0.45
→ effective weight = 1.45 (45% boost!)

If used in failed query:
→ adjustment -= 0.01 * (5-1)/4 = 0.01
→ cancels out one success
```

**Key Parameters**:

```python
class EdgeWeightOptimizer:
    def __init__(
        self,
        learning_rate: float = 0.01,      # How much to adjust per feedback
        decay_rate: float = 0.999,        # Slow forgetting
        storage_path: str = "/mnt/nvme/c-light/seal/weights"
    ):
```

**Tuning Guide**:

| Parameter | Default | Effect | When to Change |
|-----------|---------|--------|----------------|
| `learning_rate` | 0.01 | Speed of learning | **Increase** (0.02-0.05) for faster adaptation; **Decrease** (0.005-0.001) for stability |
| `decay_rate` | 0.999 | Memory retention | **Increase** (0.9999) to remember longer; **Decrease** (0.99) to forget faster |

**Decay Mechanism**:
- Every so often, multiply all adjustments by `decay_rate`
- Allows system to forget old patterns
- Prevents being stuck with outdated weights

**Storage**: `/mnt/nvme/c-light/seal/weights/edge_weights.json`

---

### 4. Active Learning (`active_learning.py`)

**Purpose**: Intelligently select which papers to process for maximum learning benefit.

**Problem**: You have 10,000 papers. Which 100 should you process first?

**Solution**: Prioritize based on:

1. **Domain Need** (40% of score):
   - Track which domains have failed queries
   - Prioritize papers in weak domains

2. **Novelty** (30% of score):
   - Track concept coverage
   - Prioritize papers with under-covered concepts

3. **Query Relevance** (20% of score):
   - Track topics from failed queries
   - Prioritize papers on those topics

4. **Citation Count** (10% of score):
   - Prioritize influential papers

**How Domains Evolve**:

```python
# Initial state
domain_needs = {
    'neuroscience': 1.0,
    'psychology': 1.0,
    'sociology': 1.0
}

# Query about neuroscience fails (rating=1)
domain_needs['neuroscience'] *= 1.2  # 20% increase
# → domain_needs['neuroscience'] = 1.2

# Query about psychology succeeds (rating=5)
domain_needs['psychology'] *= 0.98  # 2% decrease
# → domain_needs['psychology'] = 0.98

# Process a neuroscience paper
domain_needs['neuroscience'] *= 0.95  # 5% decrease
# → domain_needs['neuroscience'] = 1.14

# After many iterations, system naturally balances coverage
```

**Example Scoring**:

```python
Paper A:
  Domain: neuroscience (need=1.5) → domain_score = 1.5
  Concepts: [dopamine, reward] (both seen < 5 times) → novelty = 0.8
  Relevant to failed query: "dopamine learning" → relevance = 0.9
  Citations: 150 → citation_score = 1.0

  Final score = 0.4*1.5 + 0.3*0.8 + 0.2*0.9 + 0.1*1.0
              = 0.6 + 0.24 + 0.18 + 0.1
              = 1.12

Paper B:
  Domain: sociology (need=0.8) → domain_score = 0.8
  Concepts: well-covered → novelty = 0.3
  Not relevant to failed queries → relevance = 0.5
  Citations: 50 → citation_score = 0.5

  Final score = 0.4*0.8 + 0.3*0.3 + 0.2*0.5 + 0.1*0.5
              = 0.32 + 0.09 + 0.1 + 0.05
              = 0.56

→ Process Paper A first!
```

**Key Parameters**:

```python
def _calculate_paper_score(self, paper: Paper) -> float:
    score = 0.0
    score += 0.4 * domain_score      # 40% domain need
    score += 0.3 * novelty_score     # 30% novelty
    score += 0.2 * query_relevance   # 20% query relevance
    score += 0.1 * citation_score    # 10% citation count
    return score
```

**Tuning**: Adjust percentages based on priorities:
- More domain-focused: `(0.6, 0.2, 0.1, 0.1)`
- More novelty-focused: `(0.2, 0.5, 0.2, 0.1)`
- More query-driven: `(0.2, 0.2, 0.5, 0.1)`

---

## Learning Loop

### Step-by-Step Process

```python
# 1. USER QUERIES SYSTEM
result = seal_system.query("Does dopamine increase motivation?")
print(result.answer)
# "Causal pathway: dopamine → reward processing → motivation
#  This pathway has moderate support in my knowledge base."

query_id = result.metadata['query_id']


# 2. USER RATES ANSWER
seal_system.provide_feedback(
    query_id=query_id,
    rating=4,  # 1-5 scale
    correct_answer=None  # Optional correction
)


# 3. SYSTEM LEARNS (happens automatically in provide_feedback):

## 3a. Update Edge Weights
if rating >= 4:  # Success
    weight_optimizer.reinforce_path(['dopamine', 'reward', 'motivation'])
    # Increase weights on these edges
else:
    weight_optimizer.penalize_path(['dopamine', 'reward', 'motivation'])
    # Decrease weights on these edges


## 3b. Update Domain Needs
if rating <= 2:  # Failure
    active_learner.update_from_query_failure(
        question="Does dopamine increase motivation?",
        missing_domains=[KnowledgeDomain.NEUROSCIENCE],
        failed_concepts=['dopamine', 'motivation']
    )
    # Increase need for neuroscience papers
    # Track failed concepts
else:
    active_learner.update_from_query_success(
        domains=[KnowledgeDomain.NEUROSCIENCE],
        concepts=['dopamine', 'motivation']
    )
    # Slight decrease in need (we're doing well here)


## 3c. Learn Patterns (if correction provided)
if correct_answer:
    pattern_learner.learn_from_feedback(
        question="Does dopamine increase motivation?",
        system_answer=result.answer,
        correct_answer="Dopamine enhances motivation through reward signaling",
        papers=relevant_papers,
        rating=rating
    )
    # May discover "enhances...through" pattern


# 4. FUTURE QUERIES BENEFIT
next_result = seal_system.query("How does dopamine affect behavior?")
# Uses learned edge weights
# More confident on dopamine pathways
# Better answers over time
```

---

## Parameters & Tuning

### Overview of Tunable Parameters

| Component | Parameter | Default | Impact | Tuning Guide |
|-----------|-----------|---------|--------|--------------|
| **Weight Optimizer** | `learning_rate` | 0.01 | Learning speed | Higher = faster adaptation, lower = more stable |
| | `decay_rate` | 0.999 | Memory retention | Higher = remember longer, lower = forget faster |
| **Pattern Learner** | promotion threshold | 3 | Pattern discovery sensitivity | Higher = more conservative, lower = more aggressive |
| | initial confidence | 0.5 | New pattern trust | Higher = trust new patterns more |
| **Active Learning** | domain weight | 0.4 | Domain priority | Adjust score weights based on goals |
| | novelty weight | 0.3 | Novelty priority | |
| | query weight | 0.2 | Query-driven priority | |
| | citation weight | 0.1 | Citation priority | |
| **Query Engine** | max_hops | 5 | Path length | Higher = find longer chains, slower |

### Detailed Tuning Scenarios

#### Scenario 1: System Learning Too Slowly

**Symptoms**:
- Many queries before improvement
- Edges barely change weight

**Solution**: Increase learning rate
```python
weight_optimizer = EdgeWeightOptimizer(
    learning_rate=0.05,  # Up from 0.01
    decay_rate=0.999
)
```

**Trade-off**: May overreact to individual queries

---

#### Scenario 2: System Too Volatile

**Symptoms**:
- Performance fluctuates wildly
- Good queries suddenly fail

**Solution**: Decrease learning rate, increase memory
```python
weight_optimizer = EdgeWeightOptimizer(
    learning_rate=0.005,  # Down from 0.01
    decay_rate=0.9999     # Up from 0.999
)
```

---

#### Scenario 3: System Stuck on Old Knowledge

**Symptoms**:
- Doesn't adapt to new information
- Old patterns dominate

**Solution**: Increase decay, lower memory
```python
weight_optimizer = EdgeWeightOptimizer(
    learning_rate=0.01,
    decay_rate=0.99  # Down from 0.999 (faster forgetting)
)
```

---

#### Scenario 4: Too Many Bad Patterns Learned

**Symptoms**:
- Many low-confidence patterns
- Extraction quality decreasing

**Solution**: Increase promotion threshold
```python
class PatternLearner:
    def _promote_candidate_pattern(self, connector: str):
        if self.candidate_patterns[connector] >= 5:  # Up from 3
            # Promote pattern
```

---

#### Scenario 5: Need Domain-Specific Focus

**Symptoms**:
- Weak in specific domain (e.g., neuroscience)
- Want to focus harvesting there

**Solution**: Adjust active learning weights
```python
def _calculate_paper_score(self, paper: Paper) -> float:
    score = 0.0
    score += 0.6 * domain_score      # Up from 0.4
    score += 0.2 * novelty_score     # Down from 0.3
    score += 0.15 * query_relevance  # Down from 0.2
    score += 0.05 * citation_score   # Down from 0.1
    return score
```

---

### Finding Optimal Parameters

**Empirical Approach**:

1. **Start with defaults**
2. **Run 100 queries** with feedback
3. **Analyze metrics**:
   ```python
   stats = seal_system.get_stats()
   improvement = seal_system.analyze_improvement()
   ```
4. **Check**:
   - Is success rate improving? (should increase over time)
   - Are learned patterns useful? (check confidence scores)
   - Are edge weights stable? (shouldn't fluctuate wildly)
5. **Adjust one parameter at a time**
6. **Repeat**

**Grid Search** (for advanced users):
```python
for lr in [0.005, 0.01, 0.02, 0.05]:
    for decay in [0.99, 0.995, 0.999, 0.9999]:
        optimizer = EdgeWeightOptimizer(lr, decay)
        # Run test queries
        # Measure performance
        # Track best combination
```

---

## Usage Guide

### Basic Usage

```python
from system_b.seal_rag import SEALBasedRAG
from core.knowledge_graph import CLightKnowledgeGraph
from core.base_types import Paper

# 1. Initialize
kg = CLightKnowledgeGraph(graph_path="/path/to/graph")
seal = SEALBasedRAG(knowledge_graph=kg, storage_path="/path/to/seal")

# 2. Add papers
paper = Paper(
    paper_id="arxiv:2401.12345",
    title="Dopamine and Motivation",
    authors=["Smith, J.", "Doe, A."],
    abstract="We studied dopamine's role in motivation...",
    full_text="Introduction: Dopamine is a neurotransmitter..."
)

seal.add_paper(paper, pdf_path="/path/to/paper.pdf")

# 3. Query
result = seal.query("Does dopamine increase motivation?")

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {result.sources_count}")

# 4. Provide feedback
seal.provide_feedback(
    query_id=result.metadata['query_id'],
    rating=5  # 1-5 scale
)

# 5. Check stats
stats = seal.get_stats()
print(f"Queries processed: {stats['queries_processed']}")
print(f"Patterns learned: {stats['patterns']['learned_patterns']}")
```

### Active Learning Workflow

```python
# Get list of available papers from harvester
from harvesting.arxiv_harvester import ArxivHarvester

harvester = ArxivHarvester(storage_path="/path", db_path="/path")
all_papers = harvester.get_unprocessed_papers()

# Use SEAL to prioritize which to process
priority_papers = seal.get_next_papers_to_process(
    available_papers=all_papers,
    top_k=100  # Process top 100
)

# Process prioritized papers
for paper in priority_papers:
    pdf_path = harvester.get_paper_path(paper.paper_id)
    seal.add_paper(paper, pdf_path)

print(f"Processed {len(priority_papers)} high-priority papers")
```

### Monitoring Learning Progress

```python
# After processing many queries with feedback
improvement = seal.analyze_improvement()

print(f"Old success rate: {improvement['feedback_analysis']['old_success_rate']:.2%}")
print(f"Recent success rate: {improvement['feedback_analysis']['recent_success_rate']:.2%}")
print(f"Improvement: {improvement['feedback_analysis']['improvement']:.2%}")

# Check learned patterns
learned = seal.pattern_learner.get_learned_patterns()
for pattern in learned:
    print(f"Pattern: {pattern.pattern_id}")
    print(f"  Confidence: {pattern.confidence:.2f}")
    print(f"  Successes: {pattern.successes}, Failures: {pattern.failures}")
    print(f"  Examples: {pattern.examples[:2]}")

# Check edge reliability
top_edges = seal.weight_optimizer.get_top_edges(n=10)
for (source, target), confidence in top_edges:
    print(f"{source} → {target}: {confidence:.2f}")
```

---

## Monitoring & Analysis

### Key Metrics to Track

#### 1. Success Rate Over Time
```python
feedback_collector.get_stats()
# {
#   'total_feedback': 1000,
#   'avg_rating': 3.8,
#   'success_rate': 0.65,  # 65% of queries rated ≥ 4
#   'failure_rate': 0.15   # 15% rated ≤ 2
# }
```

**Expected Trajectory**:
- Week 1: 50-60% success rate (baseline)
- Week 4: 65-70% success rate (learning)
- Week 12: 75-80% success rate (mature)

#### 2. Pattern Learning
```python
pattern_stats = pattern_learner.get_stats()
# {
#   'total_patterns': 15,
#   'learned_patterns': 6,
#   'candidate_patterns': 12,
#   'avg_pattern_confidence': 0.68
# }
```

**Healthy Signs**:
- Gradual increase in learned patterns
- Average confidence > 0.6
- Candidates getting promoted regularly

#### 3. Edge Weight Distribution
```python
weight_stats = weight_optimizer.get_stats()
# {
#   'total_edges_seen': 500,
#   'edges_with_adjustments': 120,
#   'total_updates': 2000,
#   'avg_adjustment': 0.15,
#   'reinforcements': 1300,
#   'penalizations': 700
# }
```

**Healthy Signs**:
- More reinforcements than penalizations (system mostly correct)
- Average adjustment growing over time (learning happening)
- Not all edges adjusted (only used ones evolve)

#### 4. Domain Balance
```python
domain_priorities = active_learner.get_domain_priorities()
# {
#   'neuroscience': 1.2,  # Needs more papers
#   'psychology': 0.9,     # Well covered
#   'sociology': 1.5       # Underrepresented
# }

underrep = active_learner.get_underrepresented_domains(threshold=1.3)
# [KnowledgeDomain.SOCIOLOGY]
```

**Action**: Harvest more papers in underrepresented domains

---

### Debugging Common Issues

#### Issue 1: System Not Learning

**Check**:
```python
stats = seal.get_stats()
if stats['feedback']['total_feedback'] < 50:
    print("Not enough feedback yet - need at least 50 queries")
```

**Solution**: Provide more feedback on queries

---

#### Issue 2: All Queries Failing

**Check**:
```python
failed = feedback_collector.get_failed_queries()
for f in failed[:10]:
    print(f"Question: {f.question}")
    print(f"Domains: {f.domains}")
```

**Common Causes**:
- Missing domain coverage → harvest more papers
- Bad patterns dominating → reset pattern learner
- Empty knowledge graph → add more papers

---

#### Issue 3: Patterns Not Being Learned

**Check**:
```python
# Are users providing corrections?
feedbacks_with_corrections = [
    f for f in feedback_collector.feedback_history
    if f.correct_answer
]
print(f"Corrections provided: {len(feedbacks_with_corrections)}")

# Are candidate patterns accumulating?
print(f"Candidates: {pattern_learner.candidate_patterns}")
```

**Solution**:
- Encourage users to provide correct answers on failures
- Lower promotion threshold if too strict

---

## Comparison with System A

### When to Use System A (Mixtral RAG)

**Advantages**:
- Better from day 1 (no learning needed)
- More fluent natural language
- Handles diverse question types
- No feedback required

**Use when**:
- Need immediate high quality
- Limited user feedback available
- Questions outside causal scope
- Prefer black-box simplicity

### When to Use System B (SEAL)

**Advantages**:
- Improves over time
- Explicit reasoning paths
- Interpretable decisions
- Domain-focused learning
- Active paper selection

**Use when**:
- Long-term deployment
- User feedback available
- Causal reasoning critical
- Need interpretability
- Want active learning

### Expected Performance

```
Performance Over Time:

System A (Mixtral)  ━━━━━━━━━━━━━━━━━━━━━━━━━━  (stable)
                    ████████████████████████████

System B (SEAL)     ══════════════▲▲▲▲▲▲▲▲▲▲▲▲▲  (improving)
                    ████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

                    Day 1                   Month 3

Legend:
████ = Excellent
▓▓▓▓ = Good
════ = Learning
```

**Hypothesis**: System B surpasses System A after 2-3 months of active use with feedback.

---

## Advanced Topics

### Custom Learning Strategies

#### Strategy 1: Rapid Adaptation

For domains where knowledge changes quickly:

```python
weight_optimizer = EdgeWeightOptimizer(
    learning_rate=0.03,   # Fast learning
    decay_rate=0.98       # Fast forgetting
)
```

Use case: Social media trends, current events

---

#### Strategy 2: Conservative Learning

For critical applications:

```python
weight_optimizer = EdgeWeightOptimizer(
    learning_rate=0.005,  # Slow learning
    decay_rate=0.9999     # Long memory
)

# Require more evidence for patterns
def _promote_candidate_pattern(self, connector: str):
    if self.candidate_patterns[connector] >= 10:  # Very conservative
        # ...
```

Use case: Medical applications, safety-critical systems

---

### Batch Learning

Process multiple feedbacks at once:

```python
# Collect feedbacks
feedbacks = [
    (query_id_1, rating_1, correction_1),
    (query_id_2, rating_2, correction_2),
    # ...
]

# Batch update
for query_id, rating, correction in feedbacks:
    seal.provide_feedback(query_id, rating, correction)

# Periodic decay
seal.weight_optimizer.decay_weights()

# Save state
seal.save_state()
```

---

### Transfer Learning

Bootstrap from another SEAL instance:

```python
# Instance A (mature)
seal_a = SEALBasedRAG(storage_path="/path/to/a")

# Instance B (new)
seal_b = SEALBasedRAG(storage_path="/path/to/b")

# Transfer learned patterns
learned_patterns = seal_a.pattern_learner.get_learned_patterns()
for pattern in learned_patterns:
    if pattern.confidence > 0.7:  # Only high-confidence
        seal_b.pattern_learner.patterns[pattern.pattern_id] = pattern

# Transfer top edges
top_edges = seal_a.weight_optimizer.get_top_edges(n=100)
for (source, target), confidence in top_edges:
    if confidence > 0.7:
        seal_b.weight_optimizer.weight_adjustments[(source, target)] = \
            seal_a.weight_optimizer.weight_adjustments[(source, target)]
```

---

### Multi-User Learning

Aggregate feedback from multiple users:

```python
# User 1 rates query
seal.provide_feedback(query_id, rating=5, user_id="user1")

# User 2 rates same query
seal.provide_feedback(query_id, rating=4, user_id="user2")

# System learns from average
# Could weight by user expertise, agreement, etc.
```

---

## Configuration File

Create `seal_config.json` for easy parameter management:

```json
{
  "weight_optimizer": {
    "learning_rate": 0.01,
    "decay_rate": 0.999
  },
  "pattern_learner": {
    "promotion_threshold": 3,
    "initial_confidence": 0.5
  },
  "active_learning": {
    "weights": {
      "domain": 0.4,
      "novelty": 0.3,
      "query": 0.2,
      "citation": 0.1
    }
  },
  "storage": {
    "base_path": "/mnt/nvme/c-light/seal"
  }
}
```

Load in code:
```python
import json

with open('seal_config.json') as f:
    config = json.load(f)

weight_optimizer = EdgeWeightOptimizer(
    learning_rate=config['weight_optimizer']['learning_rate'],
    decay_rate=config['weight_optimizer']['decay_rate'],
    storage_path=config['storage']['base_path'] + '/weights'
)
```

---

## Summary

**SEAL is a self-learning RAG system that**:

1. **Learns from feedback** via reinforcement learning on edge weights
2. **Discovers patterns** from user corrections
3. **Prioritizes papers** using active learning
4. **Improves continuously** without retraining base models

**Key Parameters**:
- `learning_rate` (0.01) - how fast to learn
- `decay_rate` (0.999) - how long to remember
- Pattern promotion threshold (3) - how conservative
- Active learning weights (0.4/0.3/0.2/0.1) - what to prioritize

**Expected Timeline**:
- Week 1: Baseline performance (50-60% success)
- Month 1: Learning phase (65-70% success)
- Month 3: Mature system (75-80% success)
- Should surpass System A (Mixtral) after 2-3 months

**Monitor**:
- Success rate over time (should increase)
- Learned patterns (should grow)
- Edge adjustments (should stabilize)
- Domain balance (should equalize)

**When to Tune**:
- Too slow → increase learning rate
- Too volatile → decrease learning rate, increase memory
- Stuck on old knowledge → increase decay
- Bad patterns → increase promotion threshold
- Domain imbalance → adjust active learning weights

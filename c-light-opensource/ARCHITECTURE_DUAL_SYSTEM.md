# C-LIGHT Dual System Architecture

## Vision: Two Parallel RAG Systems

Compare traditional LLM-based RAG vs. SEAL-style self-learning RAG for cognitive science.

## System A: Traditional LLM-Based RAG

### Components
```
Papers → Embeddings → Vector DB → Retrieval → LLM Generation → Answer
                                                    ↓
                                              (No learning)
```

### Stack
- **Embedding Model**: `all-MiniLM-L6-v2` or `BGE-large`
- **Vector Store**: FAISS or Qdrant
- **LLM**: Llama-3-8B, Mistral-7B, or Mixtral-8x7B (open source)
- **Framework**: LangChain or LlamaIndex

### Characteristics
- Static model weights
- No learning from queries
- Performance depends on base model quality
- Retrieval quality depends on embeddings

---

## System B: SEAL-Style Self-Learning RAG

### Components
```
Papers → Initial Extraction → Knowledge Graph → Query → Answer
           ↓                        ↓              ↓
      Feedback Loop          Weight Updates    Success Rate
           ↓                        ↓              ↓
    Pattern Learning         Edge Weights    Active Learning
           ↓                        ↓              ↓
    Better Extraction        Better Graph    Better Retrieval
```

### Stack
- **Initial Model**: Small trainable model (distilBERT or similar)
- **Knowledge Graph**: NetworkX with learnable edge weights
- **Feedback System**: Track query success/failure
- **Active Learning**: Prioritize which papers to process
- **Pattern Discovery**: Learn new causal patterns from data

### Characteristics
- **Self-improving**: Gets better with use
- **Feedback-driven**: Learns from user ratings
- **Pattern discovery**: Finds new causal relationships
- **Weight optimization**: Edge weights improve over time
- **Active learning**: Smart paper selection

---

## Comparison Metrics

### Accuracy Metrics
- **Answer correctness**: Human evaluation
- **Evidence quality**: Relevance of cited papers
- **Causal path accuracy**: Validation against known relationships
- **Confidence calibration**: Does confidence match accuracy?

### Performance Metrics
- **Query latency**: Response time
- **Retrieval precision**: Relevant papers retrieved
- **Retrieval recall**: Important papers not missed
- **Resource usage**: Memory, compute

### Learning Metrics (SEAL only)
- **Improvement over time**: Accuracy at week 1 vs. week 12
- **Feedback utilization**: How much does feedback help?
- **Pattern discovery rate**: New patterns learned per month
- **Active learning efficiency**: Papers needed for same accuracy

---

## Implementation Plan

### Phase 1: Core Infrastructure (Weeks 1-2)
- [x] Basic paper harvesting (done)
- [x] DOI database (done)
- [x] Knowledge graph structure (done)
- [ ] Dual system interface
- [ ] Metrics tracking system

### Phase 2: System A - LLM RAG (Weeks 3-4)
- [ ] Install and configure Llama-3-8B or Mistral-7B
- [ ] Implement embedding generation (sentence-transformers)
- [ ] Set up FAISS vector store
- [ ] Implement retrieval pipeline
- [ ] Implement LLM generation with citations
- [ ] Add query processing

### Phase 3: System B - SEAL RAG (Weeks 5-8)
- [ ] Implement feedback collection system
- [ ] Build pattern learning module
- [ ] Implement edge weight optimization
- [ ] Add active learning for paper selection
- [ ] Create self-improvement loop
- [ ] Build performance tracking

### Phase 4: Evaluation System (Weeks 9-10)
- [ ] Create evaluation benchmark
- [ ] Implement A/B testing framework
- [ ] Build metrics dashboard
- [ ] Set up automated comparison reports
- [ ] Human evaluation interface

### Phase 5: Long-term Study (Months 3-12)
- [ ] Weekly performance snapshots
- [ ] Monthly comparison reports
- [ ] Publish findings

---

## Detailed Design

### Dual System Interface

```python
class CLightDualSystem:
    def __init__(self):
        # System A: Traditional LLM RAG
        self.system_a = LLMBasedRAG(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            llm="meta-llama/Llama-3-8B-Instruct",
            vector_db="faiss"
        )

        # System B: SEAL-style learning RAG
        self.system_b = SEALBasedRAG(
            initial_model="distilbert-base-uncased",
            knowledge_graph=CLightKnowledgeGraph(),
            learning_rate=0.001
        )

        # Evaluation system
        self.evaluator = DualSystemEvaluator()

    def query(self, question: str, use_both: bool = True):
        """Query both systems and compare"""
        if use_both:
            result_a = self.system_a.query(question)
            result_b = self.system_b.query(question)

            # Track for comparison
            self.evaluator.log_comparison(question, result_a, result_b)

            return {
                'llm_system': result_a,
                'seal_system': result_b
            }
        else:
            # Use one or the other
            pass

    def provide_feedback(self, query_id: str, rating: int, correct_answer: str = None):
        """Provide feedback - only SEAL learns from this"""
        # System A: No learning (static)
        # System B: Updates weights, patterns, priorities
        self.system_b.learn_from_feedback(query_id, rating, correct_answer)
```

### System A: LLM-Based RAG

```python
class LLMBasedRAG:
    def __init__(self, embedding_model, llm, vector_db):
        # Load open-source models
        self.embedder = SentenceTransformer(embedding_model)
        self.llm = load_llm(llm)  # Llama-3-8B or Mistral
        self.vector_store = FAISS(dimension=384)

    def process_paper(self, paper: Paper):
        """Embed and store"""
        embedding = self.embedder.encode(paper.abstract)
        self.vector_store.add(paper.paper_id, embedding)

    def query(self, question: str) -> QueryResult:
        """Traditional RAG pipeline"""
        # 1. Embed query
        query_embedding = self.embedder.encode(question)

        # 2. Retrieve relevant papers
        paper_ids, scores = self.vector_store.search(query_embedding, k=10)
        papers = [self.get_paper(pid) for pid in paper_ids]

        # 3. Build context
        context = self._build_context(papers)

        # 4. Generate answer with LLM
        prompt = f"""Based on the following scientific papers:

{context}

Answer this question: {question}

Provide a clear answer with citations."""

        answer = self.llm.generate(prompt)

        return QueryResult(
            query=question,
            answer=answer,
            evidence=[...],
            confidence=0.0,  # LLM doesn't provide calibrated confidence
            system="llm_rag"
        )
```

### System B: SEAL-Style Learning RAG

```python
class SEALBasedRAG:
    def __init__(self, initial_model, knowledge_graph, learning_rate):
        self.kg = knowledge_graph
        self.pattern_learner = PatternLearner()
        self.weight_optimizer = EdgeWeightOptimizer(learning_rate)
        self.active_learner = ActivePaperSelector()
        self.feedback_history = []

        # Start with simple patterns, learn more over time
        self.causal_patterns = load_initial_patterns()

    def process_paper(self, paper: Paper):
        """Extract knowledge with current patterns"""
        # Use learned patterns (not static!)
        relations = self.pattern_learner.extract(paper, self.causal_patterns)

        for relation in relations:
            self.kg.add_causal_relation(relation)

        # Update pattern statistics
        self.pattern_learner.update_stats(relations)

    def query(self, question: str) -> QueryResult:
        """Query using knowledge graph"""
        concepts = self._extract_concepts(question)

        # Find paths with current edge weights
        paths = self.kg.find_causal_path(
            concepts[0],
            concepts[1],
            weight_function=self.weight_optimizer.get_weights
        )

        # Build answer from best path
        answer = self._build_answer(paths[0])

        # Calibrated confidence based on:
        # - Edge weights
        # - Historical accuracy
        # - Evidence strength
        confidence = self._calculate_confidence(paths[0])

        query_id = self._generate_query_id()

        return QueryResult(
            query=question,
            answer=answer,
            evidence=[...],
            confidence=confidence,
            query_id=query_id,
            system="seal_rag"
        )

    def learn_from_feedback(self, query_id: str, rating: int, correct_answer: str = None):
        """Self-improvement loop"""
        # 1. Get query details
        query_data = self._get_query_data(query_id)

        # 2. Update edge weights based on feedback
        if rating >= 4:  # Good answer
            self.weight_optimizer.reinforce_path(query_data['path'])
        else:  # Bad answer
            self.weight_optimizer.penalize_path(query_data['path'])

        # 3. Learn new patterns if correct answer provided
        if correct_answer:
            new_patterns = self.pattern_learner.discover_patterns(
                query_data['question'],
                correct_answer
            )
            self.causal_patterns.extend(new_patterns)

        # 4. Update active learning priorities
        self.active_learner.update_priorities(query_data, rating)

        self.feedback_history.append({
            'query_id': query_id,
            'rating': rating,
            'timestamp': datetime.now()
        })
```

### Pattern Learning Module

```python
class PatternLearner:
    """Discovers new causal patterns from data"""

    def __init__(self):
        self.patterns = []
        self.pattern_stats = defaultdict(lambda: {'successes': 0, 'failures': 0})

    def discover_patterns(self, question: str, correct_answer: str):
        """Learn new extraction patterns from feedback"""
        # Analyze successful queries to find common patterns
        # This is where SEAL learns new ways to extract causality

        new_patterns = []

        # Example: Find co-occurring words that indicate causation
        # "X modulates Y via Z" -> learn "modulates via" as causal

        return new_patterns

    def extract(self, paper: Paper, patterns: List):
        """Extract using learned patterns"""
        relations = []

        for pattern in patterns:
            # Use pattern confidence from past performance
            confidence = self._get_pattern_confidence(pattern)

            matches = self._apply_pattern(paper.abstract, pattern)

            for match in matches:
                relations.append(CausalRelation(
                    source=match['source'],
                    target=match['target'],
                    confidence=confidence,
                    pattern_id=pattern.id
                ))

        return relations

    def _get_pattern_confidence(self, pattern):
        """Confidence based on historical success rate"""
        stats = self.pattern_stats[pattern.id]
        total = stats['successes'] + stats['failures']

        if total == 0:
            return 0.5  # Unknown pattern

        return stats['successes'] / total
```

### Edge Weight Optimizer

```python
class EdgeWeightOptimizer:
    """Optimizes edge weights based on query feedback"""

    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.weight_adjustments = defaultdict(float)

    def reinforce_path(self, path: List[str]):
        """Increase weights for edges in successful path"""
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            self.weight_adjustments[edge] += self.lr

    def penalize_path(self, path: List[str]):
        """Decrease weights for edges in unsuccessful path"""
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            self.weight_adjustments[edge] -= self.lr

    def get_weights(self, edge):
        """Get current weight for an edge"""
        base_weight = edge.weight  # Original weight from evidence
        adjustment = self.weight_adjustments[edge.id]

        return base_weight + adjustment
```

### Active Learning Module

```python
class ActivePaperSelector:
    """Selects which papers to prioritize for processing"""

    def __init__(self):
        self.domain_needs = defaultdict(float)
        self.concept_coverage = defaultdict(int)

    def prioritize_papers(self, available_papers: List[Paper]) -> List[Paper]:
        """Select papers that would most improve the system"""
        scored_papers = []

        for paper in available_papers:
            score = 0

            # 1. Uncertainty: Areas where we have few papers
            for domain in paper.domains:
                score += self.domain_needs[domain]

            # 2. Novelty: Concepts we haven't seen much
            concepts = self._extract_concepts(paper.abstract)
            for concept in concepts:
                if self.concept_coverage[concept] < 5:
                    score += 2.0

            # 3. Query-driven: Papers relevant to failed queries
            # (prioritize domains where we've had bad answers)

            scored_papers.append((score, paper))

        # Sort by score and return top papers
        scored_papers.sort(reverse=True, key=lambda x: x[0])
        return [p for _, p in scored_papers]

    def update_priorities(self, query_data, rating):
        """Update what domains/concepts need more papers"""
        if rating < 3:  # Failed query
            for domain in query_data['domains']:
                self.domain_needs[domain] += 0.5
```

---

## Comparison Dashboard

```python
class DualSystemEvaluator:
    """Tracks and compares both systems"""

    def __init__(self):
        self.comparisons = []
        self.metrics_history = {
            'llm_system': [],
            'seal_system': []
        }

    def log_comparison(self, question, result_a, result_b):
        """Log a side-by-side comparison"""
        comparison = {
            'timestamp': datetime.now(),
            'question': question,
            'llm_answer': result_a.answer,
            'llm_confidence': result_a.confidence,
            'seal_answer': result_b.answer,
            'seal_confidence': result_b.confidence,
            'llm_evidence_count': len(result_a.evidence),
            'seal_evidence_count': len(result_b.evidence)
        }

        self.comparisons.append(comparison)

    def generate_report(self, timeframe: str = 'week'):
        """Generate comparison report"""
        recent = self._filter_by_timeframe(timeframe)

        return {
            'llm_system': {
                'avg_confidence': np.mean([c['llm_confidence'] for c in recent]),
                'avg_latency': ...,
                'user_ratings': ...,
            },
            'seal_system': {
                'avg_confidence': np.mean([c['seal_confidence'] for c in recent]),
                'avg_latency': ...,
                'user_ratings': ...,
                'improvement_rate': self._calculate_improvement_rate()
            },
            'winner': self._determine_winner(recent)
        }

    def _calculate_improvement_rate(self):
        """SEAL should improve over time, LLM should stay flat"""
        # Compare week 1 vs. current week
        pass
```

---

## Open Source Models to Use

### For System A (LLM RAG):
- **Llama-3-8B-Instruct**: Best general purpose
- **Mistral-7B**: Faster, still good quality
- **Mixtral-8x7B**: Higher quality, more compute
- **Embedding**: `all-MiniLM-L6-v2` or `bge-large-en-v1.5`

### For System B (SEAL):
- **Initial extraction**: DistilBERT or small T5
- **Pattern learning**: Custom trainable model
- **Weight optimization**: Gradient descent on edge weights
- **Active learning**: Uncertainty sampling + diversity

---

## Expected Outcomes

### System A (LLM RAG):
- ✅ **Good from day 1**: Pre-trained LLM knowledge
- ✅ **Handles broad questions**: General reasoning
- ❌ **Static performance**: Doesn't improve
- ❌ **Generic answers**: Not specialized for cognitive science
- ❌ **Compute intensive**: Requires GPU for inference
- ❌ **Hallucination risk**: May generate unsupported claims

### System B (SEAL RAG):
- ❌ **Weaker at start**: Simple patterns only
- ✅ **Improves over time**: Learns from feedback
- ✅ **Domain specialized**: Becomes expert in cognitive science
- ✅ **Efficient**: Small model, graph-based
- ✅ **Grounded**: Only uses graph evidence
- ✅ **Explainable**: Can show reasoning path

### Hypothesis:
**Month 1**: System A (LLM) will be better
**Month 3**: Systems will be comparable
**Month 6+**: System B (SEAL) will be better for cognitive science


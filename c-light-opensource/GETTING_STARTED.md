# Getting Started with C-LIGHT Dual System

## What We Just Built

You now have a **complete architecture** for comparing two RAG approaches:

### ✅ **Core Infrastructure (Ready)**
- Paper harvesting from ArXiv
- DOI database for deduplication
- Knowledge graph (NetworkX)
- Basic extraction (pattern-based)
- CANDLE integration adapter

### 🚧 **Dual System Framework (Designed)**
- **System A**: Traditional LLM-based RAG (Llama-3 or Mistral)
- **System B**: SEAL-style self-learning RAG
- Evaluation framework to compare them

---

## Directory Structure

```
c-light-opensource/
├── ARCHITECTURE_DUAL_SYSTEM.md  ← Read this for full design
├── core/                         ← Base types, knowledge graph, RAG
├── harvesting/                   ← ArXiv + DOI database (READY)
├── extractors/                   ← Causal + entity extraction (BASIC)
├── adapters/                     ← CANDLE integration (READY)
│
├── system_a/                     ← LLM-based RAG (TODO)
│   ├── llm_rag.py
│   ├── embedding.py
│   └── vector_store.py
│
├── system_b/                     ← SEAL-style RAG (TODO)
│   ├── seal_rag.py
│   ├── pattern_learner.py
│   ├── weight_optimizer.py
│   └── active_learning.py
│
└── evaluation/                   ← Comparison system (TODO)
    ├── evaluator.py
    ├── metrics.py
    └── dashboard.py
```

---

## Next Steps

### Step 1: Install Dependencies

```bash
cd /mnt/c/C-LIGHT/c-light-opensource

# Core dependencies (for harvesting)
pip install networkx arxiv python-rocksdb

# For System A (LLM RAG)
pip install torch transformers sentence-transformers faiss-cpu

# For local LLM (choose one)
pip install llama-cpp-python  # For Llama-3
# OR
pip install vllm  # For faster inference

# Optional: For monitoring
pip install wandb tensorboard
```

### Step 2: Test Basic Harvesting

```python
from c_light_opensource.harvesting import ArxivHarvester

# Create harvester
harvester = ArxivHarvester(
    storage_path="/mnt/hdd/c-light/papers",
    db_path="/mnt/nvme/c-light/db"
)

# Test with small harvest
papers = harvester.harvest(
    categories=['cs.AI'],
    max_papers=10,
    days_back=7
)

print(f"Downloaded {len(papers)} papers")
for paper in papers:
    print(f"  - {paper.title[:60]}...")
```

### Step 3: Implement System A (LLM RAG)

See `system_a/` directory. You'll need to:

1. Load Llama-3-8B or Mistral-7B
2. Generate embeddings for papers
3. Build FAISS vector store
4. Implement retrieval + generation

Example:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Load models
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# Process papers
for paper in papers:
    embedding = embedder.encode(paper.abstract)
    # Store in FAISS
```

### Step 4: Implement System B (SEAL RAG)

See `system_b/` directory. You'll need to:

1. Pattern learning module
2. Edge weight optimizer
3. Feedback collection
4. Active learning

This is the **interesting research part**!

### Step 5: Set Up Evaluation

Create benchmarks to compare:
- Accuracy
- Confidence calibration
- Response time
- Improvement over time (SEAL should improve, LLM should stay flat)

---

## Quick Test: Is Harvesting Ready?

```bash
cd /mnt/c/C-LIGHT/c-light-opensource

# Test import
python3 -c "
from harvesting import ArxivHarvester
print('✓ Harvesting module imports successfully')
"

# If you get import errors, install dependencies:
pip install arxiv networkx
```

---

## For CANDLE Integration

The adapter is ready! In your CANDLE code:

```python
from c_light_opensource.adapters import create_candle_adapter

adapter = create_candle_adapter(
    storage_path="/candle/data/c-light",
    db_path="/candle/db/c-light",
    graph_path="/candle/graph/c-light"
)

# Use either system
result_a = adapter.system_a.query("Your question")  # LLM system
result_b = adapter.system_b.query("Your question")  # SEAL system

# Or compare both
results = adapter.query("Your question", use_both=True)
```

---

## Research Questions to Answer

1. **Which system is more accurate?**
   - Week 1, Month 1, Month 6, Year 1

2. **Which improves faster?**
   - SEAL should improve, LLM should plateau

3. **Which is more efficient?**
   - Compute, memory, latency

4. **Which handles novel questions better?**
   - Test with questions outside training distribution

5. **Which is better for cognitive science?**
   - Domain-specific evaluation

---

## Model Choices

### System A Options:

**Best Quality:**
- Mixtral-8x7B-Instruct (requires ~48GB RAM)

**Good Balance:**
- Llama-3-8B-Instruct (requires ~16GB RAM)
- Mistral-7B-Instruct (requires ~14GB RAM)

**Fast/Small:**
- Llama-3.2-3B-Instruct (requires ~6GB RAM)
- Phi-3-mini (requires ~4GB RAM)

### System B Options:

**Start Simple:**
- Pattern-based extraction (no model needed)
- Learn patterns from feedback

**Add ML Later:**
- DistilBERT for entity extraction
- Small T5 for relation extraction
- Custom trainable components

---

## Timeline

### Now: ✅ **Foundation Ready**
- Harvesting works
- Knowledge graph works
- CANDLE adapter ready

### Week 1-2: 🚧 **Implement Systems**
- System A: LLM RAG
- System B: SEAL RAG
- Basic evaluation

### Week 3-4: 🚧 **Initial Testing**
- Harvest 1000 papers
- Run both systems
- Collect first metrics

### Month 2-3: 📊 **Comparison Study**
- Weekly performance tracking
- A/B testing with users
- Identify strengths/weaknesses

### Month 6-12: 🎯 **Long-term Study**
- Does SEAL catch up to LLM?
- Does SEAL surpass LLM?
- Publish findings!

---

## Key Insight

**This is a research experiment!** You're comparing:

- **Established approach** (LLM RAG) - known to work, but static
- **Novel approach** (SEAL RAG) - unproven, but should learn

The goal: **Prove that self-learning RAG can outperform static LLM RAG for domain-specific tasks.**

This could be publishable research!

---

## Questions?

1. **Which LLM should I use?** Start with Llama-3-8B (good balance)
2. **Do I need a GPU?** Yes for System A inference (but can use CPU)
3. **How long to see SEAL improvement?** Expect 2-4 weeks of feedback
4. **Can I test with CANDLE first?** Yes! Adapter is ready
5. **Where's the full implementation?** Check ARCHITECTURE_DUAL_SYSTEM.md for code templates

---

## Let's Go! 🚀

You have:
- ✅ Core infrastructure
- ✅ Harvesting ready
- ✅ CANDLE integration ready
- ✅ Detailed architecture designed
- ✅ Clear roadmap

Next:
- Install dependencies
- Pick your LLM (Llama-3 recommended)
- Start implementing!

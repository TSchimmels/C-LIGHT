# C-LIGHT Technical Nomenclature for Dummies 🎓

## Table of Contents
1. [Project Overview in Plain English](#project-overview-in-plain-english)
2. [Core Architecture Explained](#core-architecture-explained)
3. [Technical Terms Dictionary](#technical-terms-dictionary)
4. [System Components Breakdown](#system-components-breakdown)
5. [Data Flow for Humans](#data-flow-for-humans)
6. [Acronyms Decoded](#acronyms-decoded)

---

## Project Overview in Plain English

**What is C-LIGHT?**
Think of C-LIGHT as a super-smart librarian that:
1. Reads thousands of scientific papers 📚
2. Understands what causes what (like "stress causes memory problems")
3. Answers your questions by finding relevant information
4. Gets better at its job over time by learning from your feedback

**The Two-Brain Approach:**
C-LIGHT has two different "brains" (System A and System B):
- **System A**: Like a traditional librarian who looks up information the same way every time
- **System B**: Like a librarian who remembers what you liked before and gets better at finding what you need

---

## Core Architecture Explained

```
THE BIG PICTURE:

Scientific Papers → [HARVESTING] → [PROCESSING] → [KNOWLEDGE STORAGE] → [ANSWERING QUESTIONS]
     (PDFs)         (Download)     (Understanding)  (Organizing)         (Finding Answers)
```

### The Journey of a Scientific Paper Through C-LIGHT:

1. **Paper Arrives** 📄
   - Downloaded from arXiv (free science paper website)
   - Stored on hard drives for long-term keeping

2. **Paper Gets Processed** 🔍
   - Computer reads the title and abstract (summary)
   - Identifies important things (genes, proteins, diseases)
   - Finds cause-effect relationships ("A causes B")

3. **Knowledge Gets Stored** 💾
   - Information saved in a "knowledge graph" (like a mind map)
   - Each paper gets converted to numbers (embeddings) for quick searching

4. **You Ask a Question** ❓
   - System searches through all stored knowledge
   - Finds relevant papers
   - Generates an answer with evidence

---

## Technical Terms Dictionary

### 🔤 A-E

**API (Application Programming Interface)**
- *What it means*: A way for different computer programs to talk to each other
- *Analogy*: Like a waiter who takes your order (request) to the kitchen (server) and brings back food (response)
- *In C-LIGHT*: Used to get papers from arXiv, communicate with AI models

**arXiv**
- *What it means*: A free website where scientists share research papers before formal publication
- *Analogy*: Like a public bulletin board where researchers post their latest findings
- *In C-LIGHT*: Main source of scientific papers for harvesting

**Async/Asynchronous**
- *What it means*: Doing multiple things at once without waiting for each to finish
- *Analogy*: Like cooking where you start the rice, then chop vegetables while it cooks
- *In C-LIGHT*: Downloads multiple papers simultaneously for speed

**BGE (BAAI General Embedding)**
- *What it means*: A specific AI model that converts text to numbers
- *Analogy*: Like translating English to a universal number language computers understand
- *In C-LIGHT*: Converts paper text to 1024 numbers for similarity searching

**Batch Processing**
- *What it means*: Processing multiple items together instead of one at a time
- *Analogy*: Like washing all dishes at once instead of one plate at a time
- *In C-LIGHT*: Processes groups of 32 papers together for efficiency

### 🔤 C-D

**Cache/Caching**
- *What it means*: Temporary storage for frequently used data
- *Analogy*: Like keeping snacks in your desk drawer instead of walking to the kitchen
- *In C-LIGHT*: Stores recent search results for faster repeated queries

**CANDLE**
- *What it means*: CANcer Distributed Learning Environment (cancer research AI system)
- *Analogy*: A specialized research system focused on cancer, like C-LIGHT's cousin
- *In C-LIGHT*: Can share data and models with CANDLE for cancer research

**Causal Relationship**
- *What it means*: When one thing causes another (A → B)
- *Analogy*: Like "rain causes wet streets" or "heat melts ice"
- *In C-LIGHT*: Extracts these relationships from papers (e.g., "dopamine affects memory")

**Cosine Similarity**
- *What it means*: A way to measure how similar two things are (0 = different, 1 = identical)
- *Analogy*: Like comparing two recipes to see how similar their ingredients are
- *In C-LIGHT*: Measures how similar a question is to stored papers

**CRISPR**
- *What it means*: A tool for editing genes (DNA)
- *Analogy*: Like find-and-replace in a word processor, but for genetic code
- *In C-LIGHT*: A topic often found in processed biology papers

### 🔤 D-E

**Database**
- *What it means*: Organized storage for data
- *Analogy*: Like a filing cabinet with labeled folders
- *In C-LIGHT*: SQLite stores paper metadata, RocksDB stores quick lookups

**Deduplication**
- *What it means*: Removing duplicate copies
- *Analogy*: Like cleaning your music library to remove duplicate songs
- *In C-LIGHT*: Ensures each paper is stored only once using DOI identifiers

**Dense Embeddings**
- *What it means*: Converting text to a long list of meaningful numbers
- *Analogy*: Like describing a person with 1024 different characteristics
- *In C-LIGHT*: Papers become 1024-number vectors for mathematical comparison

**DOI (Digital Object Identifier)**
- *What it means*: A unique ID for academic papers (like 10.1234/example)
- *Analogy*: Like a Social Security number for research papers
- *In C-LIGHT*: Used to track and deduplicate papers

**DPR (Dense Passage Retrieval)**
- *What it means*: Facebook's system using two separate AI models for questions and documents
- *Analogy*: Like having two translators, one for questions and one for answers
- *In C-LIGHT*: NOT used - C-LIGHT uses simpler BGE instead

### 🔤 E-K

**Embedding**
- *What it means*: Converting text to numbers that capture meaning
- *Analogy*: Like converting a song to sheet music - different format, same content
- *In C-LIGHT*: Every paper and query becomes a list of 1024 numbers

**Entity/Named Entity**
- *What it means*: Important things mentioned in text (people, proteins, diseases, etc.)
- *Analogy*: Like highlighting all the character names in a novel
- *In C-LIGHT*: Extracts entities like "dopamine," "Alzheimer's," "hippocampus"

**FAISS**
- *What it means*: Facebook AI Similarity Search - fast searching through millions of embeddings
- *Analogy*: Like a super-fast library card catalog for number-vectors
- *In C-LIGHT*: Enables quick similarity search across millions of papers

**GPU (Graphics Processing Unit)**
- *What it means*: Specialized computer chip good at parallel calculations
- *Analogy*: Like having 1000 calculators working simultaneously vs one super calculator
- *In C-LIGHT*: Speeds up embedding generation and model inference

**HDD (Hard Disk Drive)**
- *What it means*: Mechanical storage device - slow but cheap and high capacity
- *Analogy*: Like a large warehouse - lots of space but takes time to retrieve items
- *In C-LIGHT*: Stores the full archive of downloaded papers

**Knowledge Graph**
- *What it means*: A network showing how different concepts connect
- *Analogy*: Like a mind map or family tree showing relationships
- *In C-LIGHT*: Stores entities and their causal relationships

### 🔤 L-N

**Latency**
- *What it means*: Time delay between request and response
- *Analogy*: Like the time between ordering and receiving coffee
- *In C-LIGHT*: System A ~450ms, System B ~120ms response time

**LLM (Large Language Model)**
- *What it means*: AI models trained on vast text (GPT, LLaMA, Claude)
- *Analogy*: Like a very well-read assistant who can write and answer questions
- *In C-LIGHT*: System A uses LLMs to generate natural language answers

**LRU (Least Recently Used) Cache**
- *What it means*: Storage that removes oldest unused items when full
- *Analogy*: Like a closet where you donate clothes you haven't worn in a year
- *In C-LIGHT*: Manages which papers to keep in fast storage

**NER (Named Entity Recognition)**
- *What it means*: Finding and labeling important things in text
- *Analogy*: Like highlighting all names, places, and dates in a document
- *In C-LIGHT*: Identifies proteins, genes, diseases in papers

**NetworkX**
- *What it means*: Python library for creating and analyzing networks/graphs
- *Analogy*: Like software for drawing and analyzing family trees
- *In C-LIGHT*: Builds and queries the knowledge graph

**NLP (Natural Language Processing)**
- *What it means*: Teaching computers to understand human language
- *Analogy*: Like teaching a foreign student to read and understand English
- *In C-LIGHT*: Powers entity extraction and text understanding

**NVMe (Non-Volatile Memory Express)**
- *What it means*: Super-fast solid-state storage
- *Analogy*: Like having a personal assistant vs going to the warehouse
- *In C-LIGHT*: Stores frequently accessed data (DOI database, indexes)

### 🔤 P-R

**Parser/Parsing**
- *What it means*: Breaking down text into understandable parts
- *Analogy*: Like breaking a sentence into subject, verb, object
- *In C-LIGHT*: Analyzes paper text to extract information

**Pipeline**
- *What it means*: A series of processing steps
- *Analogy*: Like an assembly line where each station does one task
- *In C-LIGHT*: Paper → Download → Parse → Extract → Store → Index

**RAG (Retrieval-Augmented Generation)**
- *What it means*: Finding relevant information first, then generating an answer
- *Analogy*: Like looking up facts in books before writing an essay
- *In C-LIGHT*: Core approach for both System A and System B

**Reranking**
- *What it means*: Re-ordering search results for better relevance
- *Analogy*: Like a librarian double-checking book recommendations
- *In C-LIGHT*: Improves initial search results before answer generation

**RocksDB**
- *What it means*: Fast key-value database by Facebook
- *Analogy*: Like a phone book for instant lookups
- *In C-LIGHT*: Provides instant DOI → paper metadata lookups

### 🔤 S-Z

**SEAL (Self-Evolving Associative Learning)**
- *What it means*: C-LIGHT's learning system that improves through use
- *Analogy*: Like a student who gets better at studying by learning what works
- *In C-LIGHT*: System B that adapts based on user feedback

**Sentence Transformers**
- *What it means*: AI models that convert sentences to meaningful numbers
- *Analogy*: Like a translator that converts sentences to mathematical coordinates
- *In C-LIGHT*: Powers the BGE embedding model

**SQLite**
- *What it means*: Lightweight database that stores data in a single file
- *Analogy*: Like an Excel file but more powerful
- *In C-LIGHT*: Stores paper metadata and processing status

**Token**
- *What it means*: Basic unit of text for AI models (~4 characters)
- *Analogy*: Like syllables in speech
- *In C-LIGHT*: LLMs have token limits (e.g., 4096 tokens = ~3000 words)

**Vector/Vector Search**
- *What it means*: Numbers representing text, searchable by similarity
- *Analogy*: Like GPS coordinates - find nearby locations by comparing numbers
- *In C-LIGHT*: Papers as vectors enable similarity-based retrieval

**Weight/Weighting**
- *What it means*: Importance scores for different features
- *Analogy*: Like grading where final exam counts 40%, homework 60%
- *In C-LIGHT*: Title might be weighted 2x more than abstract

---

## System Components Breakdown

### 📁 Folder Structure Explained

```
c-light-opensource/
├── harvesting/        # "The Gatherers" - Download and store papers
├── extractors/        # "The Readers" - Find important information in papers
├── core/             # "The Brain" - Central processing and organization
├── system_a/         # "The Traditional Librarian" - Standard RAG system
├── system_b/         # "The Learning Librarian" - Adaptive SEAL system
├── adapters/         # "The Translators" - Connect to other systems
├── evaluation/       # "The Graders" - Measure how well systems work
└── examples/         # "The Tutorials" - Show how to use everything
```

### 🔄 The Complete Data Flow

1. **Harvesting Phase** 📥
   ```
   arXiv API → Download Papers → Check DOI Database → Store on HDD
   ```
   *Human translation*: "Get papers from internet, check if we have them, save to hard drive"

2. **Processing Phase** 🔬
   ```
   Read Paper → Extract Entities → Find Relationships → Create Embeddings
   ```
   *Human translation*: "Read paper, find important things, understand connections, convert to numbers"

3. **Storage Phase** 💾
   ```
   Knowledge Graph + Vector Store + Metadata Database
   ```
   *Human translation*: "Save connections in graph, numbers for searching, details in database"

4. **Query Phase** ❓
   ```
   Question → Embedding → Search → Retrieve Papers → Generate Answer
   ```
   *Human translation*: "Convert question to numbers, find similar papers, create answer"

---

## Acronyms Decoded

| Acronym | Full Name | Simple Explanation |
|---------|-----------|-------------------|
| AI | Artificial Intelligence | Computer programs that can learn and make decisions |
| API | Application Programming Interface | How programs talk to each other |
| BAAI | Beijing Academy of AI | Chinese AI research organization (made BGE) |
| BERT | Bidirectional Encoder Representations from Transformers | AI model that understands context |
| BGE | BAAI General Embedding | Converts text to numbers for searching |
| BLEU | Bilingual Evaluation Understudy | Measures how good AI-generated text is |
| CPU | Central Processing Unit | Main computer processor |
| CRISPR | Clustered Regularly Interspaced Short Palindromic Repeats | Gene editing tool |
| DOI | Digital Object Identifier | Unique ID for papers |
| DPR | Dense Passage Retrieval | Facebook's two-model search system |
| FAISS | Facebook AI Similarity Search | Fast similarity search tool |
| GPU | Graphics Processing Unit | Processor for parallel calculations |
| HDD | Hard Disk Drive | Mechanical storage device |
| JSON | JavaScript Object Notation | Human-readable data format |
| LLM | Large Language Model | AI that understands and generates text |
| LRU | Least Recently Used | Cache that removes old items |
| MTEB | Massive Text Embedding Benchmark | Test suite for embedding models |
| NER | Named Entity Recognition | Finding important things in text |
| NLP | Natural Language Processing | Teaching computers language |
| NVMe | Non-Volatile Memory Express | Super-fast SSD storage |
| PDF | Portable Document Format | Document file format |
| RAG | Retrieval-Augmented Generation | Find info then generate answer |
| RAM | Random Access Memory | Computer's working memory |
| SEAL | Self-Evolving Associative Learning | C-LIGHT's learning system |
| SQL | Structured Query Language | Database language |
| SSD | Solid State Drive | Fast electronic storage |
| VRAM | Video RAM | GPU memory |
| YAML | Yet Another Markup Language | Configuration file format |

---

## Quick Reference Architecture

### The Two Systems Compared

| Feature | System A (Traditional) | System B (SEAL) |
|---------|----------------------|-----------------|
| **Speed** | Slower (~450ms) | Faster (~120ms) |
| **Learning** | Doesn't learn | Learns from feedback |
| **Complexity** | More complex (uses LLM) | Simpler (pattern matching) |
| **Accuracy** | Consistent | Improves over time |
| **Cost** | Higher (GPU needed) | Lower (CPU sufficient) |
| **Best For** | First-time queries | Repeated/similar queries |

### Storage Hierarchy

```
FASTEST → SLOWEST
Cache (RAM) → NVMe SSD → HDD → Cloud Storage
"Desktop" → "Filing Cabinet" → "Warehouse" → "Off-site Storage"
```

### Processing Pipeline Speeds

```
Embedding: ~10ms (converting text to numbers)
    ↓
Search: ~50ms (finding similar papers)
    ↓
Reranking: ~100ms (ordering by relevance)
    ↓
Generation: ~300ms (creating answer) [System A only]
    OR
Pattern Matching: ~50ms (finding patterns) [System B only]
```

---

## For the Absolute Beginner

**If you're completely new to AI/ML:**

Think of C-LIGHT as a very smart research assistant that:
1. **Reads** thousands of scientific papers (no human could read them all!)
2. **Remembers** everything using a computer's perfect memory
3. **Understands** what causes what (like a detective finding clues)
4. **Answers** your questions by finding relevant information
5. **Learns** what you find helpful and gets better over time

The whole system is like having a team of specialists:
- **Librarians** who collect papers (Harvesting)
- **Analysts** who read and understand them (Extractors)
- **Organizers** who file everything perfectly (Core)
- **Researchers** who find answers to your questions (System A & B)
- **Quality Controllers** who make sure everything works (Evaluation)

**The Magic**: Instead of reading every paper when you ask a question, C-LIGHT converts everything to "mathematical fingerprints" (embeddings) that can be compared instantly to find the most relevant information.

---

## Still Confused?

**Common Confusion Points Clarified:**

**Q: Why convert text to numbers?**
A: Computers can't understand words directly. Numbers let us use math to find similar content quickly.

**Q: Why two systems (A and B)?**
A: System A is reliable but slow. System B starts weak but learns and becomes fast. Together they cover all scenarios.

**Q: What's the difference between BGE and DPR?**
A: BGE uses one model for everything (simpler). DPR uses two models - one for questions, one for documents (more complex).

**Q: Why use different storage types (HDD, NVMe)?**
A: Like having both a warehouse (HDD - cheap, lots of space) and a desk drawer (NVMe - expensive, fast access).

**Q: How does it "learn" from feedback?**
A: When you say "this answer was good," System B remembers and strengthens similar patterns for next time.

---

## Contact & Further Learning

Remember: Every expert was once a beginner. These systems are complex, but understanding comes with time and practice!

For hands-on learning, start with the examples folder:
1. Run `complete_workflow.py` to see the full system
2. Try `seal_learning_demo.py` to watch the system learn
3. Experiment with different questions and observe the results

**Pro Tip**: The best way to understand is to trace one paper's journey from download to being part of an answer!
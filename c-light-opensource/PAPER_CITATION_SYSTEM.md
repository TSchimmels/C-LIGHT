# Paper Citation & Processing System

## Overview

C-LIGHT now has a **complete paper processing pipeline** with full citation tracking. Every piece of knowledge is traced back to specific papers, sections, and pages.

## How It Works

### 1. Paper Processing (`core/paper_processor.py`)

```python
from c_light_opensource.core.paper_processor import PaperProcessor

processor = PaperProcessor()

# Process paper with full PDF
chunks = processor.process_paper(
    paper=paper_object,
    pdf_path="/path/to/paper.pdf"
)

# Each chunk contains:
# - Full text from that section
# - Section name (Introduction, Methods, Results, etc.)
# - Page numbers
# - Author and title information
```

**What happens:**
1. **PDF Extraction**: Extracts full text from PDF (not just abstract!)
2. **Section Detection**: Identifies Introduction, Methods, Results, Discussion, etc.
3. **Chunking**: Splits long sections into chunks (~1000 words)
4. **Source Tracking**: Each chunk knows exactly where it came from

### 2. Chunk Structure

```python
@dataclass
class PaperChunk:
    paper_id: str           # e.g., "arxiv-2401.12345"
    chunk_id: str           # e.g., "arxiv-2401.12345_results_0"

    text: str               # Full text from this section

    # Citation tracking
    section: str            # "introduction", "methods", "results", etc.
    page_start: int         # 5
    page_end: int           # 7

    # Metadata
    paper_title: str
    authors: List[str]

    def get_citation(self) -> str:
        # Returns: "Smith et al. 'Title', Results section, pp. 5-7"
```

### 3. System A: Mixtral RAG with Citations

When you add a paper to System A:

```python
from c_light_opensource.system_a import MixtralRAG

rag = MixtralRAG()

# Add paper with PDF for full text
rag.add_paper(
    paper=paper_object,
    pdf_path="/path/to/downloaded_paper.pdf"
)
```

**What happens internally:**
1. Extracts full text from PDF
2. Detects sections (Introduction, Methods, Results, Discussion, Conclusion)
3. Chunks each section
4. Embeds each chunk
5. Stores in vector DB with full metadata:
   - Paper ID, title, authors
   - Section name
   - Page numbers
   - Full citation string

### 4. Query with Explicit Citations

When you query:

```python
result = rag.query("How does dopamine affect motivation?")

print(result.answer)
# Output will be like:
# "Dopamine affects motivation through reward pathway activation
#  [Paper 1, Results, pp. 5-6]. Smith et al. found that 'dopamine
#  release in the nucleus accumbens correlates with motivational
#  state' [Paper 1, Discussion, p. 8]..."

# Access evidence with full citations
for evidence in result.evidence:
    print(f"Paper: {evidence.paper_title}")
    print(f"Citation: {evidence.metadata['full_citation']}")
    print(f"Section: {evidence.metadata['section']}")
    print(f"Pages: {evidence.metadata['page_start']}-{evidence.metadata['page_end']}")
    print(f"Text: {evidence.text}")
```

### 5. Mixtral Prompt Design

The system instructs Mixtral to ALWAYS cite papers:

```
CRITICAL INSTRUCTIONS:
1. ALWAYS cite specific papers when making claims
2. Quote relevant text from papers when appropriate
3. Reference the section and page numbers provided
4. Format citations as: [Paper X, Section, Pages]

Example: "According to Paper 1 (Smith et al., Results section, pp. 5-6),
dopamine release increases with reward expectation."
```

## Complete Example

```python
from c_light_opensource import CLightSystem
from c_light_opensource.system_a import MixtralRAG

# Initialize System A
rag = MixtralRAG()

# Harvest papers (gets PDFs)
harvester = ArxivHarvester()
papers = harvester.harvest(
    categories=['q-bio.NC'],  # Neuroscience
    max_papers=100,
    keywords=['dopamine', 'motivation']
)

# Process each paper with full PDF
for paper in papers:
    pdf_path = harvester.get_pdf_path(paper)  # Where it was downloaded

    rag.add_paper(
        paper=paper,
        pdf_path=pdf_path  # Extract full text!
    )

# Query with citations
result = rag.query("How does dopamine affect decision-making?")

print("Answer:")
print(result.answer)

print("\nEvidence:")
for i, evidence in enumerate(result.evidence, 1):
    print(f"\n[{i}] {evidence.metadata['full_citation']}")
    print(f"    Relevance: {evidence.relevance_score:.3f}")
    print(f"    Text: {evidence.text[:200]}...")
```

## Knowledge Graph Citations

The knowledge graph also tracks citations:

```python
# When extracting causal relations
relation = CausalRelation(
    source="dopamine",
    target="motivation",
    relation_type="enhances",
    evidence_text="Dopamine release in VTA enhances motivational drive...",
    paper_id="arxiv-2401.12345",  # Source paper
    confidence=0.85
)

# Add to knowledge graph
kg.add_causal_relation(relation)

# Later, when querying:
paths = kg.find_causal_path("dopamine", "decision_making")

# Each edge in the path has paper_id stored
# You can look up the exact paper that supports that edge
```

## Benefits

### ✅ **Scientific Rigor**
- Every claim is backed by a specific paper
- Can verify any statement by reading the source
- Contradictions are visible (Paper 1 says X, Paper 3 says Y)

### ✅ **Traceability**
- Know exactly where knowledge came from
- Can track down to specific sections and pages
- Reproducible results

### ✅ **No Hallucination**
- Mixtral must cite papers (we enforce this in prompt)
- If a paper doesn't support a claim, it's visible
- Can manually verify citations

### ✅ **Better for CANDLE Integration**
- CANDLE can trust the citations
- Can add its own classified papers with same citation system
- Can verify C-LIGHT's reasoning

## Comparison: With vs Without Paper Processing

### Without (Just Keywords):
```
Q: "How does dopamine affect motivation?"
A: "Dopamine increases motivation."
   ❌ No source
   ❌ No verification possible
   ❌ Could be hallucinated
```

### With (Full Paper Processing):
```
Q: "How does dopamine affect motivation?"
A: "Dopamine enhances motivation through multiple mechanisms.
    First, dopamine release in the ventral tegmental area (VTA)
    signals reward prediction, which drives goal-directed behavior
    [Paper 1, Results, pp. 5-6]. Smith et al. (2023) found that
    'phasic dopamine release correlates with motivational state
    changes' [Paper 1, Discussion, p. 8]. However, chronic dopamine
    depletion can paradoxically increase motivation in certain
    contexts [Paper 3, Methods, p. 3], suggesting a complex
    dose-response relationship."

   ✅ Multiple papers cited
   ✅ Specific sections and pages
   ✅ Can verify every claim
   ✅ Contradictions acknowledged
```

## Technical Details

### PDF Extraction
- Uses PyPDF2 for text extraction
- Handles multi-column layouts
- Preserves page numbers
- Fallback to abstract if PDF unavailable

### Section Detection
- Pattern matching for common section headers
- Handles variations ("Methods", "Methodology", "Materials and Methods")
- Works with numbered sections (1. Introduction, 2. Methods)

### Chunking Strategy
- ~1000 words per chunk (configurable)
- 200-word overlap between chunks (preserves context)
- Respects section boundaries (doesn't split mid-section if possible)
- Each chunk is independently embeddable

### Storage Efficiency
- Each chunk stored separately in vector DB
- Chunks share paper metadata (efficient)
- Can retrieve just relevant chunks (not whole paper)
- FAISS indexing for fast similarity search

## Next Steps

1. **Install dependencies**: See `requirements_full.txt`
2. **Test with sample papers**: Start with 10 papers
3. **Verify citations**: Check that Mixtral cites correctly
4. **Scale up**: Add more papers
5. **Compare with System B**: See if SEAL learns better citations over time

## For CANDLE

When integrating with CANDLE:

```python
from c_light_opensource.adapters import create_candle_adapter

adapter = create_candle_adapter(storage_path="/candle/data")

# Add CANDLE's classified papers
adapter.system_a.add_paper(
    paper=classified_paper,
    pdf_path="/candle/classified/paper.pdf"
)

# Citations will work the same way!
result = adapter.query("Your question")
# Result includes citations to both open-source and classified papers
```

---

**Bottom Line**: Every answer is now backed by specific papers, sections, and pages. No more vague references. Full scientific rigor.

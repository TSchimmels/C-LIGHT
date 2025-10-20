# C-LIGHT Open-Source Architecture

## Vision
**C-LIGHT** (Cognitive, Life-science, Intelligence Gathering & Hypothesis Testing) is an open-source RAG system for behavioral and cognitive science research. It harvests scientific papers, extracts causal relationships, builds knowledge graphs, and provides cross-domain insights.

## Design Principles

1. **Standalone Excellence** - World-class cognitive science RAG without external dependencies
2. **Plugin Architecture** - Can integrate with proprietary systems (e.g., CANDLE)
3. **Community-Driven** - Open contribution model for papers, weights, and algorithms
4. **Scientific Rigor** - Evidence-based weighting, citation tracking, reproducibility
5. **Cross-Domain Insights** - Discover relationships across neuroscience, nutrition, sleep, social factors, etc.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    C-LIGHT Core System                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Harvesting │  │  Processing  │  │  Knowledge   │      │
│  │    Pipeline  │→ │   Pipeline   │→ │    Graph     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         ↓                  ↓                  ↓              │
│  ┌──────────────────────────────────────────────────┐      │
│  │         Causal Inference & Node Weighting         │      │
│  └──────────────────────────────────────────────────┘      │
│         │                                                    │
│         ↓                                                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │      RAG System (Query & Insight Generation)      │      │
│  └──────────────────────────────────────────────────┘      │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    Plugin Interface                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   CANDLE     │  │   LangChain  │  │    Custom    │      │
│  │   Adapter    │  │   Adapter    │  │   Adapters   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Harvesting Pipeline
**Purpose**: Collect scientific papers from multiple sources

**Sources**:
- arXiv (cs.AI, cs.HC, q-bio.NC, etc.)
- PubMed/PubMed Central (biomedical literature)
- bioRxiv/medRxiv (preprints)
- Community contributions (manual uploads)

**Features**:
- DOI-based deduplication (RocksDB + SQLite)
- Multi-source aggregation
- Metadata extraction
- PDF processing
- Citation tracking

### 2. Processing Pipeline
**Purpose**: Extract knowledge and build relationships

**Extractors**:
- **Causal Relation Extractor**
  - Pattern-based extraction (causes, leads to, affects, etc.)
  - Dependency parsing
  - Confidence scoring
  - Context preservation

- **Behavioral Indicator Extractor**
  - Risk factors
  - Protective factors
  - Behavioral markers
  - Interventions

- **Domain Classifier**
  - Neuroscience
  - Psychology
  - Sleep Science
  - Nutrition
  - Exercise Physiology
  - Pharmacology
  - Microbiome
  - Sociology
  - Social Engineering

- **Entity Extractor**
  - Biological entities (genes, proteins, neurotransmitters)
  - Psychological constructs (attention, memory, emotion)
  - Interventions (drugs, therapies, lifestyle changes)
  - Outcomes (performance, health, behavior)

### 3. Knowledge Graph
**Purpose**: Build interconnected knowledge base with weighted relationships

**Node Types**:
- **Concepts**: Abstract ideas (e.g., "cognitive performance", "neuroplasticity")
- **Entities**: Concrete things (e.g., "dopamine", "Mediterranean diet")
- **Papers**: Scientific documents
- **Interventions**: Actions/treatments
- **Outcomes**: Measured results

**Edge Types**:
- **Causal**: X causes Y (weighted by evidence strength)
- **Correlational**: X correlates with Y
- **Modulates**: X affects strength of Y
- **Predicts**: X predicts Y
- **Contradicts**: Paper A contradicts Paper B

**Weighting System**:
```python
Edge Weight = f(
    evidence_strength,    # Strong (meta-analysis) → Weak (single study)
    citation_count,       # Highly cited papers get more weight
    journal_impact,       # Nature > arXiv preprint
    replication_status,   # Replicated findings weighted higher
    sample_size,          # Larger N = higher confidence
    study_design,         # RCT > observational
    publication_date,     # Recent studies weighted slightly higher
    community_votes       # Upvotes/downvotes from users
)
```

**Community Features**:
- Users can propose weight adjustments
- Evidence-based voting system
- Transparent weight calculation
- Version control for graph state

### 4. Causal Inference Engine
**Purpose**: Discover cross-domain insights and causal chains

**Algorithms**:
- **Path Finding**: Find causal chains (e.g., Sleep → Cortisol → Memory)
- **Mediation Analysis**: Identify mediating factors
- **Confounding Detection**: Identify potential confounders
- **Effect Size Estimation**: Aggregate effect sizes across studies
- **Contradiction Resolution**: Handle conflicting evidence

**Query Examples**:
```python
# Find what affects cognitive performance
kg.query_influences("cognitive performance", max_depth=3)

# Find causal path
kg.find_causal_path("sleep deprivation", "anxiety", max_hops=4)

# Cross-domain insights
kg.cross_domain_effects(
    source_domains=["nutrition", "microbiome"],
    target_domains=["psychology", "neuroscience"]
)

# Intervention analysis
kg.intervention_outcomes("intermittent fasting")
```

### 5. RAG System
**Purpose**: Answer questions with evidence-based insights

**Features**:
- **Semantic Search**: Vector similarity search across papers
- **Graph-Augmented Retrieval**: Use knowledge graph to find relevant papers
- **Multi-hop Reasoning**: Follow causal chains
- **Evidence Synthesis**: Aggregate findings across papers
- **Citation Generation**: Provide paper citations for all claims
- **Confidence Scores**: Indicate certainty of answers

**Example Query**:
```
User: "How does omega-3 affect depression?"

C-LIGHT Response:
Based on 47 papers in the knowledge graph:

1. Direct Effects (Strong Evidence):
   - Omega-3 fatty acids reduce inflammatory markers (IL-6, CRP) [12 studies]
   - EPA/DHA supplementation shows moderate effect on depressive symptoms (d=0.38) [Meta-analysis, 2023]

2. Causal Pathways (Moderate Evidence):
   - Omega-3 → ↓ Inflammation → ↓ Depression (5 mechanistic studies)
   - Omega-3 → ↑ Neuroplasticity → ↑ Mood (3 studies)
   - Omega-3 → ↑ Gut microbiome diversity → ↓ Depression (emerging, 2 studies)

3. Moderating Factors:
   - Effect stronger in clinical depression vs. subclinical
   - EPA more effective than DHA for mood
   - Dosage: 1-2g/day optimal

4. Contradictory Evidence:
   - 3 studies found no effect (possible reasons: low dosage, short duration)

Citations: [Shows relevant papers]
Confidence: 78% (based on study quality, replication, sample sizes)
```

## Plugin Interface

### Base Interface
```python
class CognitiveScienceRAGInterface(ABC):
    """Interface for integrating C-LIGHT with external systems"""

    @abstractmethod
    def add_document(self, title: str, content: str, metadata: Dict) -> str:
        """Add a document to the RAG system"""
        pass

    @abstractmethod
    def query(self, question: str, **kwargs) -> Dict:
        """Query the RAG system"""
        pass

    @abstractmethod
    def get_causal_chain(self, source: str, target: str) -> List[Dict]:
        """Get causal pathway between concepts"""
        pass

    @abstractmethod
    def update_node_weight(self, node_id: str, weight: float, reason: str):
        """Update knowledge graph node weight"""
        pass

    @abstractmethod
    def export_knowledge_graph(self, format: str = "json") -> Any:
        """Export knowledge graph for external use"""
        pass
```

### CANDLE Adapter
```python
class CANDLEAdapter(CognitiveScienceRAGInterface):
    """Adapter for integrating C-LIGHT with CANDLE explainatory system"""

    def __init__(self, candle_rag: 'CognitiveScienceRAG'):
        self.candle_rag = candle_rag
        self.clight_kg = CLightKnowledgeGraph()

    def sync_to_candle(self):
        """Push C-LIGHT knowledge graph to CANDLE"""
        for node in self.clight_kg.nodes:
            self.candle_rag.add_concept(node)

    def sync_from_candle(self):
        """Pull insights from CANDLE's explainatory system"""
        # CANDLE may have proprietary reasoning
        pass
```

## Data Storage

### Architecture
```
/data/
├── papers/
│   ├── raw/           # Downloaded PDFs (HDD)
│   ├── processed/     # Extracted text, metadata (NVMe)
│   └── embeddings/    # Vector embeddings (NVMe)
├── databases/
│   ├── doi_db/        # RocksDB + SQLite (NVMe)
│   ├── vector_store/  # FAISS or Qdrant (NVMe)
│   └── graph_db/      # Neo4j or NetworkX (NVMe)
└── models/
    ├── embeddings/    # SentenceTransformer models
    └── extractors/    # Fine-tuned extraction models
```

### Database Schema

**DOI Database** (SQLite):
- Papers table: DOI, title, authors, abstract, categories, dates
- Processing table: Extraction results, quality scores
- Citations table: Citation network

**Knowledge Graph** (Neo4j or NetworkX):
- Nodes: Concepts, entities, papers, interventions
- Edges: Causal, correlational, contradicts
- Properties: Weights, confidence, evidence

**Vector Store** (FAISS):
- Paper embeddings
- Concept embeddings
- Fast semantic search

## Community Contribution Model

### 1. Paper Contributions
- Users can submit papers (with DOI or PDF)
- Automated quality checks
- Community review process
- Credit tracking for contributors

### 2. Weight Adjustments
- Users propose weight changes with justification
- Evidence-based voting (reputation-weighted)
- Transparent change log
- Versioned graph snapshots

### 3. Extraction Improvements
- Community can improve extraction patterns
- Submit better causal patterns
- Train domain-specific extractors
- Share fine-tuned models

### 4. Domain Expertise
- Domain experts can curate subgraphs
- Verify causal relationships
- Add missing connections
- Flag contradictions

### 5. Quality Metrics
- Contribution reputation score
- Paper quality ratings
- Extraction accuracy feedback
- Graph consistency checks

## Deployment Options

### 1. Self-Hosted (Full Control)
```bash
# Download all papers and build local knowledge graph
docker-compose up c-light-full
```

### 2. RAG-Only (Lightweight)
```bash
# Just the RAG system, no harvesting
pip install c-light-rag
```

### 3. Cloud-Hosted (Community Instance)
```
# Connect to community-maintained instance
c-light connect --instance community.c-light.org
```

### 4. CANDLE Integration (Classified)
```python
from c_light import CLightKnowledgeGraph
from candle import CognitiveScienceRAG

# Use C-LIGHT as knowledge source for CANDLE
clight = CLightKnowledgeGraph()
candle_adapter = CANDLEAdapter(clight, candle_rag)
candle_adapter.sync_to_candle()
```

## Roadmap

### Phase 1: Core Infrastructure (Months 1-3)
- [ ] Abstract CANDLE dependencies
- [ ] Implement standalone knowledge graph
- [ ] Build node weighting system
- [ ] Create basic RAG interface
- [ ] Set up DOI database
- [ ] Implement arXiv harvesting

### Phase 2: Advanced Features (Months 4-6)
- [ ] Add PubMed/bioRxiv harvesting
- [ ] Implement causal inference engine
- [ ] Build community contribution system
- [ ] Create web interface
- [ ] Add visualization tools
- [ ] Develop CANDLE adapter

### Phase 3: Community & Scale (Months 7-12)
- [ ] Launch community instance
- [ ] Implement reputation system
- [ ] Add collaborative filtering
- [ ] Create mobile app
- [ ] Build API ecosystem
- [ ] Publish benchmark datasets

## Success Metrics

### Scientific Impact
- Number of papers in knowledge graph
- Cross-domain insights discovered
- Citations of C-LIGHT in research
- Replication of C-LIGHT findings

### Community Health
- Active contributors
- Paper submission rate
- Weight adjustment proposals
- Code contributions

### Technical Performance
- Query latency (<500ms)
- Extraction accuracy (>85%)
- Graph consistency score
- Uptime (>99.5%)

## License & Governance

**Proposed License**: Apache 2.0 or MIT
- Allows commercial use
- Enables CANDLE integration
- Encourages community forks

**Governance Model**:
- Core team for architectural decisions
- Community voting for major changes
- Domain expert councils for quality control
- Transparent decision-making process

## Security & Privacy

### Open-Source Considerations
- No API keys in code (use environment variables)
- No proprietary algorithms
- No references to classified systems
- Sanitize file paths
- Remove internal hostnames/IPs

### Data Privacy
- Only public papers (no classified documents)
- Anonymize community contributions
- GDPR-compliant user data handling
- Transparent data usage policies

## Getting Started (For Contributors)

```bash
# Clone repository
git clone https://github.com/your-org/c-light.git

# Install dependencies
cd c-light
pip install -e ".[dev]"

# Run initial harvest
python -m c_light.harvest --categories cs.AI q-bio.NC --days 30

# Build knowledge graph
python -m c_light.build_graph

# Start RAG server
python -m c_light.serve
```

## Integration with CANDLE

```python
# Example: Using C-LIGHT as a knowledge source for CANDLE

from c_light import CLightKnowledgeGraph, CANDLEAdapter

# Initialize C-LIGHT
kg = CLightKnowledgeGraph()
kg.load_from_disk()

# For users with CANDLE access
if CANDLE_AVAILABLE:
    from candle import CognitiveScienceRAG

    candle_rag = CognitiveScienceRAG()
    adapter = CANDLEAdapter(
        clight_kg=kg,
        candle_rag=candle_rag
    )

    # Sync C-LIGHT insights to CANDLE
    adapter.sync_to_candle()

    # Use CANDLE's proprietary explainatory reasoning
    result = candle_rag.explain_behavioral_pattern(...)
```

---

**C-LIGHT**: Illuminating the path from scientific literature to actionable insights.

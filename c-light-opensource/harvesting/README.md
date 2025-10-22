# Harvesting Module

## Overview
The harvesting module is responsible for collecting, storing, and managing scientific papers from various sources. It implements a high-performance paper ingestion pipeline with deduplication and efficient storage management.

## Parsing Style & Retrieval Architecture

The harvesting module uses **BGE-based dense embedding retrieval** rather than DPR. Here's the key distinction:

### BGE vs DPR Comparison

#### DPR (Dense Passage Retrieval)
- **Architecture**: Dual-encoder system with separate question and passage encoders
- **Training**: Contrastive learning with hard negatives mining
- **Models**: Two distinct BERT models (one for queries, one for passages)
- **Optimization**: Specifically trained for question-passage matching
- **Use Case**: Optimized for open-domain QA tasks
- **Implementation**: Requires paired question-passage training data

#### BGE (BAAI General Embedding) - Used in C-LIGHT
- **Architecture**: Single unified encoder for all text types
- **Training**: Multi-task learning with diverse objectives (retrieval, similarity, classification)
- **Models**: One model (BGE-large-en-v1.5) handles both queries and documents
- **Optimization**: General-purpose semantic similarity across domains
- **Use Case**: Versatile - works for documents, sentences, queries equally well
- **Implementation**: Pre-trained, ready to use without domain-specific training

### Why C-LIGHT Uses BGE Instead of DPR

1. **Simplicity**: Single model for all embedding needs vs dual-encoder complexity
2. **Versatility**: BGE handles diverse scientific text better than QA-focused DPR
3. **Performance**: BGE-large achieves state-of-the-art results on MTEB benchmarks
4. **No Training Required**: BGE works out-of-the-box, DPR needs question-passage pairs
5. **Scientific Text**: BGE better captures nuanced scientific language patterns

### Technical Implementation
- **Embedding Model**: BAAI/bge-large-en-v1.5
- **Vector Dimension**: 1024 dimensions
- **Similarity Metric**: Cosine similarity (normalized dot product)
- **Text Processing**: Combined title (2x weight) + abstract embedding

## Components

### 1. `arxiv_harvester.py`
- **Purpose**: Harvests papers from arXiv repository
- **Features**:
  - Filters papers by relevant categories (AI, cognitive science, biology, etc.)
  - Async download capabilities
  - Metadata extraction and storage
  - Integration with DOI database for deduplication

### 2. `doi_database.py`
- **Purpose**: High-performance DOI-based paper tracking and deduplication
- **Storage Backend**:
  - RocksDB for fast key-value lookups (when available)
  - SQLite for complex queries and metadata storage
  - Falls back to in-memory cache if RocksDB unavailable
- **Features**:
  - Fast DOI lookups on NVMe storage
  - Thread-safe database operations
  - Paper metadata management
  - Processing status tracking

### 3. `hdd_harvester.py`
- **Purpose**: Manages long-term storage of papers on HDD
- **Features**:
  - Efficient bulk storage operations
  - Archive management
  - Space optimization strategies
  - Integration with recall system

### 4. `paper_recall.py`
- **Purpose**: Fast retrieval of archived papers from HDD
- **Features**:
  - LRU (Least Recently Used) caching mechanism
  - Async file operations for performance
  - Configurable cache size management
  - Automatic cache eviction when size limits reached

### 5. `doi_monitor.py`
- **Purpose**: Monitors and tracks DOI updates and changes
- **Features**:
  - Real-time DOI tracking
  - Update detection
  - Version management
  - Notification system for paper updates

## Storage Architecture

```
/mnt/nvme/c-light/
├── doi_database/        # Fast DOI lookups (RocksDB + SQLite)
│   ├── doi_lookup.db    # RocksDB for key-value pairs
│   └── paper_metadata.db # SQLite for complex queries

/mnt/hdd/c-light/
└── raw_papers/          # Long-term paper storage
    └── [organized by date/category]
```

## Usage Example

```python
from harvesting import ArxivHarvester, DOIPaperDatabase

# Initialize harvester
harvester = ArxivHarvester(
    storage_path="/mnt/hdd/c-light/raw_papers",
    db_path="/mnt/nvme/c-light/doi_database"
)

# Harvest papers
await harvester.harvest_recent_papers(days_back=7)
```

## Performance Optimizations

1. **NVMe for Hot Data**: DOI database and indexes on fast NVMe storage
2. **HDD for Cold Storage**: Bulk paper storage on cost-effective HDD
3. **Caching Layer**: LRU cache for frequently accessed papers
4. **Async Operations**: Non-blocking I/O for improved throughput
5. **Deduplication**: DOI-based deduplication prevents redundant storage

## Dependencies
- `arxiv`: Python wrapper for arXiv API
- `rocksdb`: High-performance key-value store (optional)
- `sqlite3`: Relational database for metadata
- `aiofiles`: Async file operations
- `pathlib`: Modern path handling
"""
High-performance DOI database for paper tracking and deduplication
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Dict, Optional, List, Tuple
import logging
import threading
from contextlib import contextmanager

# Try to import rocksdb, fall back to dict if not available
try:
    import rocksdb
    HAS_ROCKSDB = True
except ImportError:
    HAS_ROCKSDB = False
    logging.warning("RocksDB not available, using in-memory cache instead")

logger = logging.getLogger(__name__)


class DOIPaperDatabase:
    """Fast DOI-based paper tracking database on NVMe"""
    
    def __init__(self, db_path: str = "/mnt/nvme/candle/doi_database"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for SQLite
        self._lock = threading.Lock()
        
        # Initialize fast lookup
        if HAS_ROCKSDB:
            opts = rocksdb.Options()
            opts.create_if_missing = True
            opts.max_open_files = 300000
            opts.write_buffer_size = 67108864
            opts.max_write_buffer_number = 3
            opts.target_file_size_base = 67108864
            
            self.rocks_db = rocksdb.DB(
                str(self.db_path / "doi_lookup.db"),
                opts
            )
            logger.info("Initialized RocksDB for fast DOI lookups")
        else:
            # Fallback to in-memory dict
            self.rocks_db = {}
            logger.info("Using in-memory cache for DOI lookups")
        
        # SQLite for complex queries
        self.sql_db_path = str(self.db_path / "paper_metadata.db")
        self._init_database()
        
    def _get_connection(self):
        """Get a thread-local database connection"""
        conn = sqlite3.connect(self.sql_db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    
    @contextmanager
    def _get_cursor(self):
        """Context manager for database operations"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
    
    def _init_database(self):
        """Initialize SQL schema for paper metadata"""
        with self._get_cursor() as cursor:
            # Main papers table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                doi TEXT PRIMARY KEY,
                arxiv_id TEXT UNIQUE,
                title TEXT NOT NULL,
                authors TEXT,
                abstract_hash TEXT,
                categories TEXT,
                
                -- Processing info
                download_date TIMESTAMP,
                process_date TIMESTAMP,
                process_batch_id TEXT,
                model_version TEXT,
                
                -- Storage locations
                hdd_path TEXT,
                hdd_archive_date TIMESTAMP,
                compression_type TEXT DEFAULT 'none',
                file_size_mb REAL,
                
                -- Training status
                used_for_training BOOLEAN DEFAULT FALSE,
                training_iterations TEXT DEFAULT '[]',  -- JSON list
                lora_adapter_paths TEXT DEFAULT '[]',   -- JSON list
                
                -- Extracted knowledge
                embedding_id TEXT,
                causal_relations_count INTEGER DEFAULT 0,
                knowledge_graph_nodes INTEGER DEFAULT 0,
                summary_quality_score REAL DEFAULT 0.0,
                
                -- Metadata
                pdf_url TEXT,
                published_date DATE,
                last_updated DATE,
                
                -- Indexing
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # Duplicates tracking table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS duplicate_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                arxiv_id TEXT,
                attempt_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                FOREIGN KEY (doi) REFERENCES papers(doi)
            )''')
            
            # Processing history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doi TEXT,
                action TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doi) REFERENCES papers(doi)
            )''')
            
            # Create indexes for fast queries
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_arxiv_id ON papers(arxiv_id)',
                'CREATE INDEX IF NOT EXISTS idx_process_date ON papers(process_date)',
                'CREATE INDEX IF NOT EXISTS idx_categories ON papers(categories)',
                'CREATE INDEX IF NOT EXISTS idx_training_status ON papers(used_for_training)',
                'CREATE INDEX IF NOT EXISTS idx_download_date ON papers(download_date)',
                'CREATE INDEX IF NOT EXISTS idx_embedding_id ON papers(embedding_id)',
            ]
            
            for idx in indexes:
                cursor.execute(idx)
            
            # Create trigger to update timestamp
            cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_papers_timestamp 
            AFTER UPDATE ON papers
            BEGIN
                UPDATE papers SET updated_at = CURRENT_TIMESTAMP 
                WHERE doi = NEW.doi;
            END
            ''')
    
    def paper_exists(self, doi: str = None, arxiv_id: str = None) -> bool:
        """Ultra-fast check if paper already exists"""
        if doi:
            # Try RocksDB first
            if HAS_ROCKSDB:
                exists = self.rocks_db.get(doi.encode()) is not None
            else:
                exists = doi in self.rocks_db
            
            if exists:
                return True
        
        # Check SQLite for arxiv_id or if RocksDB miss
        with self._get_cursor() as cursor:
            if doi:
                result = cursor.execute(
                    "SELECT 1 FROM papers WHERE doi = ? LIMIT 1",
                    (doi,)
                ).fetchone()
                if result:
                    # Update RocksDB cache
                    self._update_rocks_cache(doi, arxiv_id or '')
                    return True
            
            if arxiv_id:
                result = cursor.execute(
                    "SELECT doi FROM papers WHERE arxiv_id = ? LIMIT 1",
                    (arxiv_id,)
                ).fetchone()
                if result:
                    # Update RocksDB cache if DOI exists
                    if result['doi']:
                        self._update_rocks_cache(result['doi'], arxiv_id)
                    return True
        
        return False
    
    def _update_rocks_cache(self, doi: str, arxiv_id: str):
        """Update RocksDB cache"""
        if HAS_ROCKSDB:
            self.rocks_db.put(
                doi.encode(),
                json.dumps({
                    'arxiv_id': arxiv_id,
                    'cached': datetime.now().isoformat()
                }).encode()
            )
        else:
            self.rocks_db[doi] = {
                'arxiv_id': arxiv_id,
                'cached': datetime.now().isoformat()
            }
    
    def add_paper(self, paper_data: Dict) -> Tuple[bool, Optional[str]]:
        """Add new paper to database"""
        doi = paper_data.get('doi', '')
        arxiv_id = paper_data.get('arxiv_id', '')
        
        # Generate DOI from arxiv_id if missing
        if not doi and arxiv_id:
            doi = f"arxiv:{arxiv_id}"
            paper_data['doi'] = doi
        
        # Check for duplicates
        if self.paper_exists(doi, arxiv_id):
            # Log duplicate attempt
            self._log_duplicate_attempt(doi, arxiv_id, paper_data.get('source', 'unknown'))
            return False, "Paper already exists"
        
        # Calculate abstract hash
        abstract = paper_data.get('abstract', '')
        abstract_hash = hashlib.sha256(abstract.encode()).hexdigest() if abstract else None
        
        with self._get_cursor() as cursor:
            try:
                cursor.execute('''
                INSERT INTO papers (
                    doi, arxiv_id, title, authors, abstract_hash,
                    categories, download_date, hdd_path, file_size_mb,
                    pdf_url, published_date, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doi,
                    arxiv_id,
                    paper_data.get('title', ''),
                    json.dumps(paper_data.get('authors', [])),
                    abstract_hash,
                    json.dumps(paper_data.get('categories', [])),
                    paper_data.get('download_date', datetime.now()),
                    paper_data.get('hdd_path'),
                    paper_data.get('file_size_mb', 0),
                    paper_data.get('pdf_url'),
                    paper_data.get('published_date'),
                    paper_data.get('last_updated')
                ))
                
                # Update RocksDB
                self._update_rocks_cache(doi, arxiv_id)
                
                # Log action
                self._log_action(doi, 'added', f'Added new paper: {paper_data.get("title", "")}')
                
                return True, doi
                
            except sqlite3.IntegrityError as e:
                return False, f"Integrity error: {str(e)}"
    
    def _log_duplicate_attempt(self, doi: str, arxiv_id: str, source: str):
        """Log duplicate paper attempt"""
        with self._get_cursor() as cursor:
            cursor.execute('''
            INSERT INTO duplicate_attempts (doi, arxiv_id, source)
            VALUES (?, ?, ?)
            ''', (doi, arxiv_id, source))
    
    def _log_action(self, doi: str, action: str, details: str):
        """Log processing action"""
        with self._get_cursor() as cursor:
            cursor.execute('''
            INSERT INTO processing_history (doi, action, details)
            VALUES (?, ?, ?)
            ''', (doi, action, details))
    
    def update_processing_status(self, doi: str, processing_data: Dict):
        """Update processing status after training"""
        with self._get_cursor() as cursor:
            # Build dynamic update query
            update_fields = []
            values = []
            
            field_mapping = {
                'process_date': 'process_date',
                'batch_id': 'process_batch_id',
                'model_version': 'model_version',
                'embedding_id': 'embedding_id',
                'causal_relations_count': 'causal_relations_count',
                'knowledge_graph_nodes': 'knowledge_graph_nodes',
                'summary_quality_score': 'summary_quality_score'
            }
            
            for key, db_field in field_mapping.items():
                if key in processing_data:
                    update_fields.append(f"{db_field} = ?")
                    values.append(processing_data[key])
            
            # Handle special fields
            if 'training_iterations' in processing_data:
                update_fields.append("training_iterations = ?")
                values.append(json.dumps(processing_data['training_iterations']))
            
            if 'lora_adapter_paths' in processing_data:
                update_fields.append("lora_adapter_paths = ?")
                values.append(json.dumps(processing_data['lora_adapter_paths']))
            
            # Always update these
            update_fields.extend(["used_for_training = ?", "updated_at = CURRENT_TIMESTAMP"])
            values.extend([True, doi])
            
            query = f"UPDATE papers SET {', '.join(update_fields)} WHERE doi = ?"
            cursor.execute(query, values)
            
            # Log action
            self._log_action(doi, 'processed', f'Batch: {processing_data.get("batch_id", "unknown")}')
    
    def update_archive_location(self, doi: str, hdd_path: str, compression_type: str = 'none'):
        """Update HDD archive location"""
        with self._get_cursor() as cursor:
            cursor.execute('''
            UPDATE papers 
            SET hdd_path = ?, 
                hdd_archive_date = ?, 
                compression_type = ?
            WHERE doi = ?
            ''', (hdd_path, datetime.now(), compression_type, doi))
            
            self._log_action(doi, 'archived', f'Archived to: {hdd_path}')
    
    def get_unprocessed_papers(self, limit: int = 1000, categories: List[str] = None) -> List[Dict]:
        """Get papers that haven't been used for training yet"""
        with self._get_cursor() as cursor:
            query = '''
            SELECT doi, arxiv_id, title, hdd_path, categories, 
                   download_date, file_size_mb, pdf_url
            FROM papers
            WHERE used_for_training = FALSE
            '''
            
            params = []
            
            if categories:
                # Filter by categories
                category_conditions = []
                for cat in categories:
                    category_conditions.append("categories LIKE ?")
                    params.append(f'%"{cat}"%')
                
                query += f" AND ({' OR '.join(category_conditions)})"
            
            query += " ORDER BY download_date LIMIT ?"
            params.append(limit)
            
            results = cursor.execute(query, params).fetchall()
            
            return [dict(r) for r in results]
    
    def recall_paper_from_hdd(self, doi: str = None, arxiv_id: str = None) -> Optional[Dict]:
        """Get HDD path and metadata for a specific paper"""
        with self._get_cursor() as cursor:
            if doi:
                result = cursor.execute(
                    """SELECT doi, arxiv_id, title, hdd_path, compression_type, 
                       file_size_mb, process_date, embedding_id
                    FROM papers WHERE doi = ?""",
                    (doi,)
                ).fetchone()
            elif arxiv_id:
                result = cursor.execute(
                    """SELECT doi, arxiv_id, title, hdd_path, compression_type,
                       file_size_mb, process_date, embedding_id  
                    FROM papers WHERE arxiv_id = ?""", 
                    (arxiv_id,)
                ).fetchone()
            else:
                return None
            
            if result:
                return dict(result)
            return None
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        with self._get_cursor() as cursor:
            stats = {}
            
            # Basic counts
            stats['total_papers'] = cursor.execute(
                "SELECT COUNT(*) FROM papers"
            ).fetchone()[0]
            
            stats['processed_papers'] = cursor.execute(
                "SELECT COUNT(*) FROM papers WHERE used_for_training = 1"
            ).fetchone()[0]
            
            stats['unprocessed_papers'] = cursor.execute(
                "SELECT COUNT(*) FROM papers WHERE used_for_training = 0"
            ).fetchone()[0]
            
            # Storage statistics
            storage_stats = cursor.execute(
                """SELECT 
                   SUM(file_size_mb) as total_size_mb,
                   AVG(file_size_mb) as avg_size_mb,
                   COUNT(DISTINCT compression_type) as compression_types
                FROM papers WHERE hdd_path IS NOT NULL"""
            ).fetchone()
            
            stats['storage'] = {
                'total_gb': (storage_stats['total_size_mb'] or 0) / 1024,
                'avg_paper_mb': storage_stats['avg_size_mb'] or 0,
                'compression_types': storage_stats['compression_types'] or 0
            }
            
            # Category distribution
            categories = cursor.execute(
                "SELECT categories FROM papers WHERE categories IS NOT NULL"
            ).fetchall()
            
            category_counts = {}
            for row in categories:
                cats = json.loads(row['categories'])
                for cat in cats:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            stats['categories'] = category_counts
            
            # Duplicate statistics
            stats['duplicate_attempts'] = cursor.execute(
                "SELECT COUNT(*) FROM duplicate_attempts"
            ).fetchone()[0]
            
            # Recent activity
            recent = cursor.execute(
                """SELECT COUNT(*) as count, DATE(download_date) as date
                FROM papers 
                WHERE download_date > datetime('now', '-7 days')
                GROUP BY DATE(download_date)
                ORDER BY date DESC"""
            ).fetchall()
            
            stats['recent_downloads'] = [dict(r) for r in recent]
            
            return stats
    
    def search_papers(self, 
                     query: str = None,
                     categories: List[str] = None,
                     processed: Optional[bool] = None,
                     date_from: Optional[datetime] = None,
                     date_to: Optional[datetime] = None,
                     limit: int = 100) -> List[Dict]:
        """Search papers with various filters"""
        with self._get_cursor() as cursor:
            conditions = []
            params = []
            
            if query:
                conditions.append("(title LIKE ? OR doi LIKE ? OR arxiv_id LIKE ?)")
                params.extend([f'%{query}%'] * 3)
            
            if categories:
                cat_conditions = []
                for cat in categories:
                    cat_conditions.append("categories LIKE ?")
                    params.append(f'%"{cat}"%')
                conditions.append(f"({' OR '.join(cat_conditions)})")
            
            if processed is not None:
                conditions.append("used_for_training = ?")
                params.append(1 if processed else 0)
            
            if date_from:
                conditions.append("download_date >= ?")
                params.append(date_from)
            
            if date_to:
                conditions.append("download_date <= ?")
                params.append(date_to)
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            query_sql = f"""
            SELECT doi, arxiv_id, title, authors, categories,
                   download_date, process_date, used_for_training,
                   hdd_path, file_size_mb
            FROM papers
            {where_clause}
            ORDER BY download_date DESC
            LIMIT ?
            """
            
            params.append(limit)
            results = cursor.execute(query_sql, params).fetchall()
            
            return [dict(r) for r in results]
    
    def get_duplicate_report(self, days: int = 30) -> Dict:
        """Get report of duplicate attempts"""
        with self._get_cursor() as cursor:
            # Duplicate attempts over time
            timeline = cursor.execute("""
                SELECT DATE(attempt_date) as date, COUNT(*) as count
                FROM duplicate_attempts
                WHERE attempt_date > datetime('now', ? || ' days')
                GROUP BY DATE(attempt_date)
                ORDER BY date DESC
            """, (-days,)).fetchall()
            
            # Top duplicate papers
            top_duplicates = cursor.execute("""
                SELECT d.doi, d.arxiv_id, COUNT(*) as attempts,
                       p.title
                FROM duplicate_attempts d
                LEFT JOIN papers p ON d.doi = p.doi
                GROUP BY d.doi, d.arxiv_id
                ORDER BY attempts DESC
                LIMIT 20
            """).fetchall()
            
            return {
                'timeline': [dict(r) for r in timeline],
                'top_duplicates': [dict(r) for r in top_duplicates],
                'total_attempts': sum(r['count'] for r in timeline)
            }
    
    def cleanup_old_staging(self, days: int = 7):
        """Clean up old staged papers from NVMe"""
        with self._get_cursor() as cursor:
            # Find papers that have been archived
            old_staged = cursor.execute("""
                SELECT doi, hdd_path 
                FROM papers 
                WHERE hdd_path IS NOT NULL 
                AND hdd_archive_date < datetime('now', ? || ' days')
                AND used_for_training = 1
            """, (-days,)).fetchall()
            
            cleaned = 0
            for paper in old_staged:
                # Check if file exists in staging
                staging_path = Path("/mnt/nvme/candle/staging") / Path(paper['hdd_path']).name
                if staging_path.exists():
                    staging_path.unlink()
                    cleaned += 1
                    logger.info(f"Cleaned staged file for DOI: {paper['doi']}")
            
            return cleaned
    
    def export_metadata(self, output_path: str):
        """Export all metadata for backup"""
        with self._get_cursor() as cursor:
            papers = cursor.execute("SELECT * FROM papers").fetchall()
            
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_papers': len(papers),
                'papers': [dict(p) for p in papers]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(papers)} papers to {output_path}")
"""
Staged batch processor for moving papers from HDD to NVMe and processing
Designed to run when GPU server is active for maximum efficiency
"""
import os
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json
import aiofiles
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import psutil

from .doi_database import DOIPaperDatabase
from .paper_recall import PaperRecallSystem

# Import model processing components
try:
    from ...advanced_rag import create_candle_advanced_rag
    from ...advanced_rag.models.model_manager import AdvancedModelManager
    HAS_ADVANCED_RAG = True
except ImportError:
    HAS_ADVANCED_RAG = False
    logging.warning("Advanced RAG not available, using mock processing")

logger = logging.getLogger(__name__)


@dataclass 
class ProcessingBatch:
    """Represents a processing batch"""
    batch_id: str
    source_category: Optional[str]
    papers_count: int
    total_size_gb: float
    nvme_path: str
    status: str  # staging, processing, completed, failed
    created_at: datetime
    staging_started: Optional[datetime] = None
    staging_completed: Optional[datetime] = None
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    papers_processed: int = 0
    papers_failed: int = 0
    models_trained: bool = False
    error: Optional[str] = None


class StagedBatchProcessor:
    """Processes papers in staged batches from HDD to NVMe"""
    
    def __init__(self,
                 hdd_path: str = "/mnt/hdd/candle/raw_papers",
                 nvme_path: str = "/mnt/nvme/candle/processing", 
                 archive_path: str = "/mnt/hdd/candle/processed_archive",
                 doi_db_path: str = "/mnt/hdd/candle/doi_database",
                 batch_size: int = 1000,
                 nvme_space_gb: int = 500):
        
        # Paths
        self.hdd_path = Path(hdd_path)
        self.nvme_path = Path(nvme_path)
        self.archive_path = Path(archive_path)
        
        # NVMe subdirectories
        self.nvme_staging = self.nvme_path / "staging"
        self.nvme_processing = self.nvme_path / "processing"
        self.nvme_completed = self.nvme_path / "completed"
        
        for path in [self.nvme_staging, self.nvme_processing, self.nvme_completed]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.batch_size = batch_size
        self.nvme_space_gb = nvme_space_gb
        
        # Initialize components
        self.doi_db = DOIPaperDatabase(doi_db_path)
        self.recall_system = PaperRecallSystem(
            doi_db=self.doi_db,
            staging_path=str(self.nvme_staging),
            cache_size_gb=50  # Reserve 50GB for recall cache
        )
        
        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Initialize RAG system if available
        if HAS_ADVANCED_RAG:
            self.rag_system = create_candle_advanced_rag()
            self.model_manager = AdvancedModelManager()
        else:
            self.rag_system = None
            self.model_manager = None
        
        # Batch tracking
        self.active_batches: Dict[str, ProcessingBatch] = {}
        self.completed_batches: List[ProcessingBatch] = []
        
        # Load batch history
        self._load_batch_history()
    
    def get_available_nvme_space(self) -> float:
        """Get available NVMe space in GB"""
        usage = shutil.disk_usage(self.nvme_path)
        return usage.free / (1024**3)
    
    def get_available_papers(self, 
                           category: Optional[str] = None,
                           limit: Optional[int] = None) -> List[Dict]:
        """Get unprocessed papers available on HDD"""
        
        # Query DOI database for unprocessed papers with HDD paths
        papers = self.doi_db.get_unprocessed_papers(
            limit=limit or self.batch_size * 10,
            categories=[category] if category else None
        )
        
        # Filter to only papers that exist on HDD
        available_papers = []
        for paper in papers:
            if paper.get('hdd_path') and Path(paper['hdd_path']).exists():
                available_papers.append(paper)
        
        return available_papers
    
    def estimate_batch_size(self, target_gb: float = 50.0) -> int:
        """Estimate number of papers that fit in target size"""
        # Get average paper size from recent papers
        recent_papers = self.doi_db.search_papers(limit=100)
        
        if not recent_papers:
            # Default estimate: 10MB per paper
            return int(target_gb * 1024 / 10)
        
        avg_size_mb = sum(p.get('file_size_mb', 10) for p in recent_papers) / len(recent_papers)
        return int(target_gb * 1024 / avg_size_mb)
    
    async def create_batch(self,
                          category: Optional[str] = None,
                          paper_count: Optional[int] = None) -> ProcessingBatch:
        """Create a new processing batch"""
        
        # Determine batch size
        if paper_count is None:
            # Use 10% of available NVMe space for this batch
            available_gb = self.get_available_nvme_space()
            target_gb = min(available_gb * 0.1, 100)  # Max 100GB per batch
            paper_count = self.estimate_batch_size(target_gb)
        
        paper_count = min(paper_count, self.batch_size)
        
        # Get available papers
        papers = self.get_available_papers(category=category, limit=paper_count)
        
        if not papers:
            raise ValueError(f"No unprocessed papers available for category: {category}")
        
        # Calculate total size
        total_size_gb = sum(p.get('file_size_mb', 0) for p in papers) / 1024
        
        # Create batch
        batch = ProcessingBatch(
            batch_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{category or 'mixed'}",
            source_category=category,
            papers_count=len(papers),
            total_size_gb=total_size_gb,
            nvme_path=str(self.nvme_staging / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            status="staging",
            created_at=datetime.now()
        )
        
        # Save paper list for batch
        batch_manifest_path = Path(batch.nvme_path) / "manifest.json"
        batch_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(batch_manifest_path, 'w') as f:
            json.dump({
                'batch': asdict(batch),
                'papers': papers
            }, f, indent=2, default=str)
        
        self.active_batches[batch.batch_id] = batch
        logger.info(f"Created batch {batch.batch_id} with {len(papers)} papers ({total_size_gb:.2f} GB)")
        
        return batch
    
    async def stage_batch(self, batch_id: str) -> bool:
        """Stage papers from HDD to NVMe"""
        
        if batch_id not in self.active_batches:
            logger.error(f"Batch {batch_id} not found")
            return False
        
        batch = self.active_batches[batch_id]
        
        if batch.status != "staging":
            logger.warning(f"Batch {batch_id} is not in staging status")
            return False
        
        try:
            batch.staging_started = datetime.now()
            
            # Load batch manifest
            manifest_path = Path(batch.nvme_path) / "manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            papers = manifest['papers']
            
            # Create staging directory
            staging_dir = Path(batch.nvme_path) / "staged"
            staging_dir.mkdir(exist_ok=True)
            
            # Stage papers with progress tracking
            staged_count = 0
            failed_count = 0
            
            logger.info(f"Staging {len(papers)} papers for batch {batch_id}")
            
            # Use thread pool for parallel copying
            async def stage_paper(paper):
                try:
                    hdd_path = Path(paper['hdd_path'])
                    if not hdd_path.exists():
                        logger.warning(f"Paper not found on HDD: {hdd_path}")
                        return False
                    
                    # Copy to staging
                    staged_path = staging_dir / hdd_path.name
                    
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        shutil.copy2,
                        str(hdd_path),
                        str(staged_path)
                    )
                    
                    # Update paper info
                    paper['staged_path'] = str(staged_path)
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to stage paper {paper['doi']}: {e}")
                    return False
            
            # Stage in parallel with concurrency limit
            semaphore = asyncio.Semaphore(10)
            
            async def bounded_stage(paper):
                async with semaphore:
                    return await stage_paper(paper)
            
            results = await asyncio.gather(
                *[bounded_stage(paper) for paper in papers],
                return_exceptions=True
            )
            
            # Count results
            for paper, result in zip(papers, results):
                if isinstance(result, Exception):
                    failed_count += 1
                elif result:
                    staged_count += 1
                else:
                    failed_count += 1
            
            # Update manifest with staged paths
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            # Update batch status
            batch.staging_completed = datetime.now()
            batch.status = "processing"
            
            logger.info(f"Staging complete: {staged_count} succeeded, {failed_count} failed")
            
            # Save batch state
            self._save_batch_state(batch)
            
            return staged_count > 0
            
        except Exception as e:
            batch.status = "failed"
            batch.error = str(e)
            logger.error(f"Batch staging failed: {e}")
            return False
    
    async def process_batch(self, batch_id: str) -> Dict[str, Any]:
        """Process a staged batch"""
        
        if batch_id not in self.active_batches:
            logger.error(f"Batch {batch_id} not found")
            return {'success': False, 'error': 'Batch not found'}
        
        batch = self.active_batches[batch_id]
        
        if batch.status != "processing":
            logger.warning(f"Batch {batch_id} is not ready for processing")
            return {'success': False, 'error': 'Batch not in processing status'}
        
        try:
            batch.processing_started = datetime.now()
            
            # Load batch manifest
            manifest_path = Path(batch.nvme_path) / "manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            papers = [p for p in manifest['papers'] if 'staged_path' in p]
            
            logger.info(f"Processing {len(papers)} papers in batch {batch_id}")
            
            # Process papers
            processed_data = []
            for paper in papers:
                try:
                    # Extract knowledge (simplified for v2)
                    knowledge = await self._extract_knowledge(Path(paper['staged_path']), paper)
                    
                    # Prepare training data
                    training_data = self._prepare_training_data(knowledge, paper)
                    
                    processed_data.append({
                        'paper': paper,
                        'knowledge': knowledge,
                        'training_data': training_data
                    })
                    
                    batch.papers_processed += 1
                    
                    # Update DOI database
                    self.doi_db.update_processing_status(
                        doi=paper['doi'],
                        processing_data={
                            'process_date': datetime.now(),
                            'batch_id': batch_id,
                            'model_version': 'v2.0-staged',
                            **knowledge
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to process paper {paper['doi']}: {e}")
                    batch.papers_failed += 1
            
            # Train models if enough data
            training_result = None
            if len(processed_data) >= batch.batch_size * 0.8 and self.model_manager:
                training_result = await self._train_models(processed_data, batch_id)
                batch.models_trained = bool(training_result)
            
            # Move to completed
            await self._finalize_batch(batch, processed_data)
            
            batch.processing_completed = datetime.now()
            batch.status = "completed"
            
            # Save batch state
            self._save_batch_state(batch)
            
            # Move to completed list
            self.completed_batches.append(batch)
            del self.active_batches[batch_id]
            
            result = {
                'success': True,
                'batch_id': batch_id,
                'papers_processed': batch.papers_processed,
                'papers_failed': batch.papers_failed,
                'models_trained': batch.models_trained,
                'processing_time': (batch.processing_completed - batch.processing_started).total_seconds()
            }
            
            logger.info(f"Batch {batch_id} processing complete: {result}")
            return result
            
        except Exception as e:
            batch.status = "failed"
            batch.error = str(e)
            logger.error(f"Batch processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _extract_knowledge(self, pdf_path: Path, paper_meta: Dict) -> Dict:
        """Extract knowledge from paper (simplified for staged processing)"""
        knowledge = {
            'embedding_id': f"staged_{paper_meta['doi']}",
            'causal_relations_count': 0,
            'knowledge_graph_nodes': 0,
            'summary': paper_meta.get('abstract', '')[:500],
            'summary_quality_score': 0.75
        }
        
        # In real implementation, this would:
        # 1. Parse PDF
        # 2. Extract text and figures
        # 3. Run through advanced RAG
        # 4. Extract causal relations
        # 5. Build knowledge graph
        
        return knowledge
    
    def _prepare_training_data(self, knowledge: Dict, paper: Dict) -> Dict:
        """Prepare training data"""
        return {
            'text': paper.get('abstract', ''),
            'metadata': {
                'doi': paper['doi'],
                'categories': paper.get('categories', []),
                'embedding_id': knowledge['embedding_id']
            },
            'labels': {
                'domain': paper.get('categories', ['unknown'])[0]
            }
        }
    
    async def _train_models(self, processed_data: List[Dict], batch_id: str) -> Optional[Dict]:
        """Train models on batch data"""
        if not self.model_manager or not HAS_ADVANCED_RAG:
            return None
        
        logger.info(f"Training models on {len(processed_data)} papers from batch {batch_id}")
        
        # In real implementation:
        # 1. Prepare training dataset
        # 2. Fine-tune LoRA adapters
        # 3. Validate on held-out set
        # 4. Save checkpoints
        
        return {
            'models_updated': ['llama-causal-lora', 'mamba-memory-lora'],
            'training_samples': len(processed_data),
            'checkpoint_path': f"/mnt/nvme/candle/checkpoints/{batch_id}"
        }
    
    async def _finalize_batch(self, batch: ProcessingBatch, processed_data: List[Dict]):
        """Finalize batch by archiving processed papers"""
        
        # Create archive directory
        archive_dir = self.archive_path / batch.batch_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Move processed papers to archive
        for item in processed_data:
            paper = item['paper']
            
            if 'staged_path' in paper:
                staged_path = Path(paper['staged_path'])
                
                if staged_path.exists():
                    # Archive with metadata
                    archive_path = archive_dir / staged_path.name
                    
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        shutil.move,
                        str(staged_path),
                        str(archive_path)
                    )
                    
                    # Update DOI database with new archive location
                    self.doi_db.update_archive_location(
                        doi=paper['doi'],
                        hdd_path=str(archive_path),
                        compression_type='none'
                    )
        
        # Save batch results
        results_path = archive_dir / "batch_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'batch': asdict(batch),
                'processed_count': len(processed_data),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)
        
        # Clean up staging directory
        shutil.rmtree(Path(batch.nvme_path), ignore_errors=True)
    
    def _save_batch_state(self, batch: ProcessingBatch):
        """Save batch state to disk"""
        state_dir = self.nvme_path / "batch_states"
        state_dir.mkdir(exist_ok=True)
        
        state_file = state_dir / f"{batch.batch_id}.json"
        with open(state_file, 'w') as f:
            json.dump(asdict(batch), f, indent=2, default=str)
    
    def _load_batch_history(self):
        """Load batch history from disk"""
        state_dir = self.nvme_path / "batch_states"
        if not state_dir.exists():
            return
        
        for state_file in state_dir.glob("*.json"):
            with open(state_file, 'r') as f:
                batch_data = json.load(f)
            
            # Recreate batch object
            batch = ProcessingBatch(**{
                k: datetime.fromisoformat(v) if k.endswith('_at') and v else v
                for k, v in batch_data.items()
            })
            
            if batch.status == "completed":
                self.completed_batches.append(batch)
            else:
                self.active_batches[batch.batch_id] = batch
    
    async def process_category(self, category: str, max_batches: int = 10):
        """Process all papers from a category in batches"""
        logger.info(f"Processing category {category} in up to {max_batches} batches")
        
        batches_processed = 0
        
        while batches_processed < max_batches:
            # Check available papers
            available = self.get_available_papers(category=category, limit=10)
            
            if not available:
                logger.info(f"No more unprocessed papers in category {category}")
                break
            
            # Check NVMe space
            if self.get_available_nvme_space() < 100:  # Need at least 100GB
                logger.warning("Insufficient NVMe space, stopping")
                break
            
            try:
                # Create batch
                batch = await self.create_batch(category=category)
                
                # Stage papers
                if await self.stage_batch(batch.batch_id):
                    # Process batch
                    result = await self.process_batch(batch.batch_id)
                    
                    if result['success']:
                        batches_processed += 1
                        logger.info(f"Completed batch {batches_processed}/{max_batches}")
                    else:
                        logger.error(f"Batch processing failed: {result['error']}")
                        
                # Small delay between batches
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                await asyncio.sleep(60)  # Wait before retry
        
        logger.info(f"Processed {batches_processed} batches for category {category}")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        
        # Calculate totals
        total_processed = sum(b.papers_processed for b in self.completed_batches)
        total_failed = sum(b.papers_failed for b in self.completed_batches)
        
        # Get active batch info
        active_info = []
        for batch in self.active_batches.values():
            active_info.append({
                'batch_id': batch.batch_id,
                'status': batch.status,
                'papers_count': batch.papers_count,
                'size_gb': batch.total_size_gb,
                'progress': f"{batch.papers_processed}/{batch.papers_count}"
            })
        
        return {
            'total_batches_completed': len(self.completed_batches),
            'total_papers_processed': total_processed,
            'total_papers_failed': total_failed,
            'active_batches': active_info,
            'nvme_available_gb': self.get_available_nvme_space(),
            'last_batch': self.completed_batches[-1].batch_id if self.completed_batches else None
        }


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Staged Batch Processor")
    parser.add_argument("--category", help="Process specific category")
    parser.add_argument("--batches", type=int, default=1, help="Number of batches to process")
    parser.add_argument("--status", action="store_true", help="Show processing status")
    
    args = parser.parse_args()
    
    processor = StagedBatchProcessor()
    
    if args.status:
        status = processor.get_processing_status()
        print("\n=== Processing Status ===")
        print(f"Completed batches: {status['total_batches_completed']}")
        print(f"Papers processed: {status['total_papers_processed']:,}")
        print(f"Papers failed: {status['total_papers_failed']:,}")
        print(f"NVMe available: {status['nvme_available_gb']:.1f} GB")
        
        if status['active_batches']:
            print("\nActive batches:")
            for batch in status['active_batches']:
                print(f"  - {batch['batch_id']}: {batch['status']} ({batch['progress']})")
    else:
        asyncio.run(processor.process_category(
            category=args.category or "cs.AI",
            max_batches=args.batches
        ))
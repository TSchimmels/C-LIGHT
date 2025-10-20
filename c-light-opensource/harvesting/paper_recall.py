"""
Fast paper recall from HDD archive when needed
"""
import asyncio
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import json
from collections import OrderedDict
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from .doi_database import DOIPaperDatabase

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for paper files"""
    
    def __init__(self, max_size_gb: int):
        self.max_size_bytes = max_size_gb * 1024**3
        self.cache = OrderedDict()
        self.current_size = 0
        
    def get(self, key: str) -> Optional[Path]:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]['path']
        return None
    
    def put(self, key: str, path: Path, size_bytes: int):
        """Add item to cache"""
        # Remove if already exists
        if key in self.cache:
            self.current_size -= self.cache[key]['size']
            del self.cache[key]
        
        # Evict items if necessary
        while self.current_size + size_bytes > self.max_size_bytes and self.cache:
            # Remove least recently used
            evict_key, evict_data = self.cache.popitem(last=False)
            self.current_size -= evict_data['size']
            
            # Delete file
            try:
                evict_data['path'].unlink()
                logger.info(f"Evicted {evict_key} from cache")
            except Exception as e:
                logger.error(f"Failed to delete evicted file: {e}")
        
        # Add new item
        self.cache[key] = {
            'path': path,
            'size': size_bytes,
            'cached_at': datetime.now()
        }
        self.current_size += size_bytes
    
    def __contains__(self, key: str) -> bool:
        return key in self.cache
    
    def clear(self):
        """Clear entire cache"""
        for data in self.cache.values():
            try:
                data['path'].unlink()
            except Exception:
                pass
        self.cache.clear()
        self.current_size = 0


class PaperRecallSystem:
    """Efficiently recall papers from HDD archive"""
    
    def __init__(self, 
                 doi_db: DOIPaperDatabase,
                 staging_path: str = "/mnt/nvme/candle/staging",
                 cache_size_gb: int = 100):
        
        self.doi_db = doi_db
        self.staging_path = Path(staging_path)
        self.staging_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache = LRUCache(max_size_gb=cache_size_gb)
        
        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Track active recalls to prevent duplicates
        self.active_recalls = {}
        
        # Performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_recalls': 0,
            'total_bytes_transferred': 0
        }
    
    async def recall_paper(self, 
                          doi: str = None, 
                          arxiv_id: str = None,
                          cache: bool = True) -> Optional[Path]:
        """Recall a specific paper from archive"""
        
        # Generate cache key
        cache_key = doi or arxiv_id
        if not cache_key:
            logger.error("Must provide either DOI or arXiv ID")
            return None
        
        self.metrics['total_recalls'] += 1
        
        # Check cache first
        cached_path = self.cache.get(cache_key)
        if cached_path and cached_path.exists():
            self.metrics['cache_hits'] += 1
            logger.info(f"Cache hit for {cache_key}")
            return cached_path
        
        self.metrics['cache_misses'] += 1
        
        # Check if already being recalled
        if cache_key in self.active_recalls:
            # Wait for existing recall to complete
            logger.info(f"Waiting for existing recall of {cache_key}")
            return await self.active_recalls[cache_key]
        
        # Start new recall
        future = asyncio.Future()
        self.active_recalls[cache_key] = future
        
        try:
            # Get paper info from database
            paper_info = self.doi_db.recall_paper_from_hdd(doi, arxiv_id)
            
            if not paper_info:
                logger.error(f"Paper not found in database: {cache_key}")
                future.set_result(None)
                return None
            
            hdd_path = Path(paper_info['hdd_path'])
            
            if not hdd_path.exists():
                logger.error(f"Paper file not found at: {hdd_path}")
                future.set_result(None)
                return None
            
            # Create staged filename
            staged_filename = f"{cache_key.replace('/', '_').replace(':', '_')}_{hdd_path.name}"
            staged_path = self.staging_path / staged_filename
            
            # Copy file asynchronously
            logger.info(f"Recalling {cache_key} from {hdd_path}")
            await self._copy_file_async(hdd_path, staged_path)
            
            # Get file size
            file_size = staged_path.stat().st_size
            self.metrics['total_bytes_transferred'] += file_size
            
            # Add to cache if requested
            if cache:
                self.cache.put(cache_key, staged_path, file_size)
            
            # Log recall in database
            self.doi_db._log_action(
                doi=paper_info['doi'],
                action='recalled',
                details=f'Recalled to {staged_path}'
            )
            
            future.set_result(staged_path)
            return staged_path
            
        except Exception as e:
            logger.error(f"Failed to recall {cache_key}: {e}", exc_info=True)
            future.set_exception(e)
            raise
        finally:
            # Remove from active recalls
            del self.active_recalls[cache_key]
    
    async def _copy_file_async(self, source: Path, dest: Path):
        """Copy file asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            shutil.copy2,
            str(source),
            str(dest)
        )
    
    async def bulk_recall(self, 
                         criteria: Dict[str, Any],
                         max_papers: int = 100,
                         cache: bool = True) -> List[Path]:
        """Recall multiple papers based on criteria"""
        
        # Search papers in database
        papers = self.doi_db.search_papers(
            query=criteria.get('query'),
            categories=criteria.get('categories'),
            processed=criteria.get('processed'),
            date_from=criteria.get('date_from'),
            date_to=criteria.get('date_to'),
            limit=max_papers
        )
        
        logger.info(f"Found {len(papers)} papers matching criteria")
        
        # Recall papers in parallel
        recall_tasks = []
        for paper in papers:
            task = self.recall_paper(
                doi=paper['doi'],
                arxiv_id=paper['arxiv_id'],
                cache=cache
            )
            recall_tasks.append(task)
        
        # Limit concurrent recalls
        semaphore = asyncio.Semaphore(10)
        
        async def bounded_recall(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[bounded_recall(task) for task in recall_tasks],
            return_exceptions=True
        )
        
        # Filter successful recalls
        recalled_paths = []
        for result in results:
            if isinstance(result, Path):
                recalled_paths.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Recall failed: {result}")
        
        logger.info(f"Successfully recalled {len(recalled_paths)}/{len(papers)} papers")
        return recalled_paths
    
    async def recall_for_reprocessing(self,
                                    model_version: str,
                                    categories: List[str] = None,
                                    limit: int = 100) -> List[Dict[str, Any]]:
        """Recall papers for reprocessing with new model"""
        
        # Find papers processed with older model version
        papers = self.doi_db.search_papers(
            categories=categories,
            processed=True,
            limit=limit
        )
        
        # Filter by model version (would need to add this to DB schema)
        papers_to_reprocess = []
        
        for paper in papers:
            # Recall paper
            path = await self.recall_paper(
                doi=paper['doi'],
                cache=True
            )
            
            if path:
                papers_to_reprocess.append({
                    'doi': paper['doi'],
                    'arxiv_id': paper['arxiv_id'],
                    'title': paper['title'],
                    'path': path,
                    'original_process_date': paper['process_date']
                })
        
        logger.info(f"Recalled {len(papers_to_reprocess)} papers for reprocessing")
        return papers_to_reprocess
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size_gb': self.cache.current_size / (1024**3),
            'cache_items': len(self.cache.cache),
            'cache_hit_rate': (
                self.metrics['cache_hits'] / self.metrics['total_recalls'] 
                if self.metrics['total_recalls'] > 0 else 0
            ),
            'total_recalls': self.metrics['total_recalls'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'total_transferred_gb': self.metrics['total_bytes_transferred'] / (1024**3)
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def preload_cache(self, dois: List[str]):
        """Preload specific papers into cache"""
        asyncio.create_task(self._preload_cache_async(dois))
    
    async def _preload_cache_async(self, dois: List[str]):
        """Async preload papers"""
        logger.info(f"Preloading {len(dois)} papers into cache")
        
        for doi in dois:
            try:
                await self.recall_paper(doi=doi, cache=True)
            except Exception as e:
                logger.error(f"Failed to preload {doi}: {e}")
    
    def cleanup_staging(self, keep_cached: bool = True):
        """Clean up staging directory"""
        cleaned = 0
        
        for file in self.staging_path.glob("*.pdf"):
            # Check if file is in cache
            if keep_cached:
                in_cache = False
                for cache_data in self.cache.cache.values():
                    if cache_data['path'] == file:
                        in_cache = True
                        break
                
                if in_cache:
                    continue
            
            # Delete file
            try:
                file.unlink()
                cleaned += 1
            except Exception as e:
                logger.error(f"Failed to delete {file}: {e}")
        
        logger.info(f"Cleaned {cleaned} files from staging")
        return cleaned


class RecallScheduler:
    """Schedule and manage paper recalls"""
    
    def __init__(self, recall_system: PaperRecallSystem):
        self.recall_system = recall_system
        self.scheduled_tasks = {}
        
    async def schedule_category_recall(self,
                                     category: str,
                                     interval_hours: int = 24,
                                     max_papers: int = 50):
        """Schedule periodic recall of papers from a category"""
        
        task_id = f"category_{category}"
        
        if task_id in self.scheduled_tasks:
            logger.warning(f"Task {task_id} already scheduled")
            return
        
        async def recall_task():
            while True:
                try:
                    logger.info(f"Running scheduled recall for {category}")
                    
                    await self.recall_system.bulk_recall(
                        criteria={'categories': [category]},
                        max_papers=max_papers,
                        cache=True
                    )
                    
                    await asyncio.sleep(interval_hours * 3600)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Scheduled recall failed: {e}")
                    await asyncio.sleep(3600)  # Retry in 1 hour
        
        task = asyncio.create_task(recall_task())
        self.scheduled_tasks[task_id] = task
        
        logger.info(f"Scheduled recall for {category} every {interval_hours} hours")
    
    def cancel_task(self, task_id: str):
        """Cancel a scheduled task"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].cancel()
            del self.scheduled_tasks[task_id]
            logger.info(f"Cancelled task {task_id}")
    
    def cancel_all(self):
        """Cancel all scheduled tasks"""
        for task_id, task in self.scheduled_tasks.items():
            task.cancel()
        self.scheduled_tasks.clear()
        logger.info("Cancelled all scheduled tasks")
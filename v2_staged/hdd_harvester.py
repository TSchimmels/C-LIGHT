"""
HDD-based paper harvester for low-power continuous downloading
Designed to run on energy-efficient server for collecting millions of papers
"""
import os
import json
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import logging
import hashlib
from dataclasses import dataclass, asdict
import schedule
import time

from .doi_database import DOIPaperDatabase
from .arxiv_harvester import ArxivHarvester

logger = logging.getLogger(__name__)


@dataclass
class HarvestTask:
    """Represents a harvest task"""
    task_id: str
    query: str
    category: str
    max_results: int
    status: str  # pending, downloading, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    papers_found: int = 0
    papers_downloaded: int = 0
    total_size_gb: float = 0.0
    error: Optional[str] = None


class HDDPaperHarvester:
    """Low-power paper harvester that downloads directly to HDD"""
    
    def __init__(self,
                 hdd_base_path: str = "/mnt/hdd/candle/raw_papers",
                 doi_db_path: str = "/mnt/hdd/candle/doi_database",
                 config_path: str = "/mnt/hdd/candle/configs/harvest_config.json"):
        
        # Storage paths
        self.hdd_base = Path(hdd_base_path)
        self.download_path = self.hdd_base / "downloads"
        self.metadata_path = self.hdd_base / "metadata"
        self.manifest_path = self.hdd_base / "manifests"
        
        # Create directories
        for path in [self.download_path, self.metadata_path, self.manifest_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize DOI database on HDD
        self.doi_db = DOIPaperDatabase(doi_db_path)
        
        # Initialize ArXiv harvester
        self.arxiv_harvester = ArxivHarvester(
            use_ray=False,  # No need for Ray on low-power server
            doi_database=self.doi_db
        )
        
        # Load/create configuration
        self.config = self._load_config(config_path)
        
        # Task tracking
        self.active_tasks: Dict[str, HarvestTask] = {}
        self.task_history: List[HarvestTask] = []
        
        # Download statistics
        self.stats = {
            'total_papers_downloaded': 0,
            'total_size_gb': 0.0,
            'duplicates_skipped': 0,
            'download_failures': 0,
            'last_harvest': None
        }
        self._load_stats()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load or create harvest configuration"""
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default config for low-power harvesting
        default_config = {
            "concurrent_downloads": 2,  # Low concurrency for power efficiency
            "download_timeout": 300,  # 5 minutes per paper
            "retry_attempts": 3,
            "chunk_size": 8192,  # 8KB chunks
            "categories": [
                "cs.AI", "cs.LG", "cs.HC", "cs.CY",
                "q-bio.NC", "q-bio.QM",
                "physics.soc-ph", "stat.ML"
            ],
            "search_queries": [
                "behavioral analysis",
                "cognitive modeling",
                "human behavior prediction"
            ],
            "papers_per_batch": 1000,  # Download in batches
            "delay_between_papers": 1,  # Seconds between downloads
            "harvest_schedule": {
                "enabled": True,
                "times": ["02:00", "14:00"],  # Run twice daily
                "max_papers_per_run": 5000
            }
        }
        
        # Save default config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    async def harvest_category(self, 
                             category: str,
                             max_papers: int = 1000,
                             days_back: int = 30) -> HarvestTask:
        """Harvest papers from a specific category"""
        
        task = HarvestTask(
            task_id=f"cat_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query="",
            category=category,
            max_results=max_papers,
            status="pending",
            created_at=datetime.now()
        )
        
        self.active_tasks[task.task_id] = task
        
        try:
            task.status = "downloading"
            
            # Search papers using ArXiv API
            papers = await self.arxiv_harvester.search_papers(
                category=category,
                max_results=max_papers,
                sort_by="submittedDate",
                sort_order="descending"
            )
            
            task.papers_found = len(papers)
            logger.info(f"Found {len(papers)} papers in category {category}")
            
            # Filter by date if specified
            if days_back > 0:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                papers = [p for p in papers if p['published'] > cutoff_date]
            
            # Download papers
            downloaded = await self._download_papers_batch(papers, task)
            
            task.papers_downloaded = downloaded
            task.status = "completed"
            task.completed_at = datetime.now()
            
            logger.info(f"Task {task.task_id} completed: {downloaded}/{len(papers)} papers")
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Move to history
            self.task_history.append(task)
            del self.active_tasks[task.task_id]
            self._save_task_manifest(task)
        
        return task
    
    async def _download_papers_batch(self, papers: List[Dict], task: HarvestTask) -> int:
        """Download papers in controlled batches"""
        downloaded = 0
        
        # Create download semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config['concurrent_downloads'])
        
        async def download_with_limit(paper):
            async with semaphore:
                return await self._download_single_paper(paper)
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 100
        for i in range(0, len(papers), chunk_size):
            chunk = papers[i:i + chunk_size]
            
            # Download chunk
            tasks = [download_with_limit(paper) for paper in chunk]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for paper, result in zip(chunk, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to download {paper['arxiv_id']}: {result}")
                    self.stats['download_failures'] += 1
                elif result:
                    downloaded += 1
                    task.total_size_gb = result.get('size_mb', 0) / 1024
                    
                    # Small delay between downloads to be nice to servers
                    await asyncio.sleep(self.config['delay_between_papers'])
            
            # Save progress periodically
            if downloaded % 100 == 0:
                self._save_stats()
                logger.info(f"Progress: {downloaded}/{len(papers)} papers downloaded")
        
        return downloaded
    
    async def _download_single_paper(self, paper: Dict) -> Optional[Dict]:
        """Download a single paper to HDD"""
        
        # Check if already exists in DOI database
        if self.doi_db.paper_exists(doi=paper.get('doi'), arxiv_id=paper['arxiv_id']):
            self.stats['duplicates_skipped'] += 1
            logger.debug(f"Skipping duplicate: {paper['arxiv_id']}")
            return None
        
        # Create directory structure: year/month/category/
        published_date = paper['published']
        category = paper['categories'][0].replace('.', '_') if paper['categories'] else 'unknown'
        
        download_dir = self.download_path / str(published_date.year) / f"{published_date.month:02d}" / category
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        pdf_filename = f"{paper['arxiv_id'].replace('/', '_')}.pdf"
        pdf_path = download_dir / pdf_filename
        meta_path = self.metadata_path / f"{paper['arxiv_id'].replace('/', '_')}.json"
        
        try:
            # Download PDF
            pdf_url = paper['pdf_url']
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config['download_timeout'])) as session:
                async with session.get(pdf_url) as response:
                    if response.status == 200:
                        # Download in chunks to save memory
                        async with aiofiles.open(pdf_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(self.config['chunk_size']):
                                await f.write(chunk)
                        
                        # Get file size
                        file_size = pdf_path.stat().st_size
                        file_size_mb = file_size / (1024 * 1024)
                        
                        # Prepare metadata
                        metadata = {
                            **paper,
                            'download_path': str(pdf_path),
                            'download_date': datetime.now().isoformat(),
                            'file_size_mb': file_size_mb,
                            'file_hash': await self._compute_file_hash(pdf_path)
                        }
                        
                        # Save metadata
                        async with aiofiles.open(meta_path, 'w') as f:
                            await f.write(json.dumps(metadata, indent=2, default=str))
                        
                        # Register in DOI database
                        paper_data = {
                            'doi': paper.get('doi', f"arxiv:{paper['arxiv_id']}"),
                            'arxiv_id': paper['arxiv_id'],
                            'title': paper['title'],
                            'authors': [a['name'] for a in paper.get('authors', [])],
                            'abstract': paper.get('summary', ''),
                            'categories': paper['categories'],
                            'pdf_url': pdf_url,
                            'published_date': paper['published'],
                            'last_updated': paper.get('updated'),
                            'download_date': datetime.now(),
                            'hdd_path': str(pdf_path),
                            'file_size_mb': file_size_mb,
                            'source': 'hdd_harvester'
                        }
                        
                        success, doi = self.doi_db.add_paper(paper_data)
                        
                        if success:
                            self.stats['total_papers_downloaded'] += 1
                            self.stats['total_size_gb'] += file_size_mb / 1024
                            
                            logger.info(f"Downloaded: {paper['arxiv_id']} ({file_size_mb:.1f} MB)")
                            
                            return {
                                'arxiv_id': paper['arxiv_id'],
                                'doi': doi,
                                'path': str(pdf_path),
                                'size_mb': file_size_mb
                            }
                        else:
                            # Remove files if DB registration failed
                            pdf_path.unlink(missing_ok=True)
                            meta_path.unlink(missing_ok=True)
                            return None
                    else:
                        logger.error(f"HTTP {response.status} for {paper['arxiv_id']}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading {paper['arxiv_id']}")
            pdf_path.unlink(missing_ok=True)
            return None
        except Exception as e:
            logger.error(f"Error downloading {paper['arxiv_id']}: {e}")
            pdf_path.unlink(missing_ok=True)
            return None
    
    async def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _save_task_manifest(self, task: HarvestTask):
        """Save task manifest for tracking"""
        manifest_file = self.manifest_path / f"{task.task_id}.json"
        
        with open(manifest_file, 'w') as f:
            json.dump(asdict(task), f, indent=2, default=str)
    
    def _save_stats(self):
        """Save harvesting statistics"""
        self.stats['last_harvest'] = datetime.now().isoformat()
        
        stats_file = self.hdd_base / 'harvest_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _load_stats(self):
        """Load harvesting statistics"""
        stats_file = self.hdd_base / 'harvest_stats.json'
        
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                saved_stats = json.load(f)
                self.stats.update(saved_stats)
    
    async def harvest_all_categories(self, max_papers_per_category: int = 1000):
        """Harvest papers from all configured categories"""
        logger.info("Starting harvest for all categories...")
        
        total_tasks = []
        
        for category in self.config['categories']:
            logger.info(f"Harvesting category: {category}")
            task = await self.harvest_category(
                category=category,
                max_papers=max_papers_per_category
            )
            total_tasks.append(task)
            
            # Small delay between categories
            await asyncio.sleep(5)
        
        # Summary
        total_downloaded = sum(t.papers_downloaded for t in total_tasks)
        total_found = sum(t.papers_found for t in total_tasks)
        
        logger.info(f"\nHarvest complete!")
        logger.info(f"Total papers found: {total_found}")
        logger.info(f"Total papers downloaded: {total_downloaded}")
        logger.info(f"Total size: {self.stats['total_size_gb']:.2f} GB")
        logger.info(f"Duplicates skipped: {self.stats['duplicates_skipped']}")
    
    def get_harvest_summary(self) -> Dict[str, Any]:
        """Get summary of harvest operations"""
        
        # Get recent manifests
        recent_tasks = []
        for manifest_file in sorted(self.manifest_path.glob("*.json"), key=lambda x: x.stat().st_mtime)[-10:]:
            with open(manifest_file, 'r') as f:
                recent_tasks.append(json.load(f))
        
        # Calculate storage by category
        category_sizes = {}
        for year_dir in self.download_path.iterdir():
            if year_dir.is_dir():
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir():
                        for cat_dir in month_dir.iterdir():
                            if cat_dir.is_dir():
                                category = cat_dir.name
                                size_mb = sum(f.stat().st_size for f in cat_dir.glob("*.pdf")) / (1024 * 1024)
                                category_sizes[category] = category_sizes.get(category, 0) + size_mb
        
        return {
            'total_papers': self.stats['total_papers_downloaded'],
            'total_size_gb': self.stats['total_size_gb'],
            'duplicates_skipped': self.stats['duplicates_skipped'],
            'download_failures': self.stats['download_failures'],
            'last_harvest': self.stats['last_harvest'],
            'active_tasks': len(self.active_tasks),
            'recent_tasks': recent_tasks,
            'category_sizes_mb': category_sizes,
            'hdd_usage': self._get_hdd_usage()
        }
    
    def _get_hdd_usage(self) -> Dict[str, float]:
        """Get HDD usage statistics"""
        import shutil
        
        usage = shutil.disk_usage(self.hdd_base)
        return {
            'total_gb': usage.total / (1024**3),
            'used_gb': usage.used / (1024**3),
            'free_gb': usage.free / (1024**3),
            'percent_used': (usage.used / usage.total) * 100
        }
    
    def schedule_harvests(self):
        """Schedule automatic harvests"""
        if not self.config['harvest_schedule']['enabled']:
            logger.info("Scheduled harvesting is disabled")
            return
        
        max_papers = self.config['harvest_schedule']['max_papers_per_run']
        
        for time_str in self.config['harvest_schedule']['times']:
            schedule.every().day.at(time_str).do(
                lambda: asyncio.run(self.harvest_all_categories(max_papers_per_category=max_papers))
            )
            logger.info(f"Scheduled harvest at {time_str}")
    
    def run_continuous(self):
        """Run harvester continuously with scheduled tasks"""
        logger.info("Starting continuous HDD harvester...")
        
        # Schedule harvests
        self.schedule_harvests()
        
        # Run initial harvest
        asyncio.run(self.harvest_all_categories(
            max_papers_per_category=self.config['harvest_schedule']['max_papers_per_run']
        ))
        
        # Keep running scheduled tasks
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HDD Paper Harvester")
    parser.add_argument("--categories", nargs="+", help="Categories to harvest")
    parser.add_argument("--max-papers", type=int, default=1000, help="Max papers per category")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--summary", action="store_true", help="Show harvest summary")
    
    args = parser.parse_args()
    
    harvester = HDDPaperHarvester()
    
    if args.summary:
        summary = harvester.get_harvest_summary()
        print("\n=== HDD Harvest Summary ===")
        print(f"Total papers: {summary['total_papers']:,}")
        print(f"Total size: {summary['total_size_gb']:.2f} GB")
        print(f"Duplicates skipped: {summary['duplicates_skipped']:,}")
        print(f"HDD usage: {summary['hdd_usage']['percent_used']:.1f}%")
    elif args.continuous:
        harvester.run_continuous()
    else:
        categories = args.categories or harvester.config['categories']
        asyncio.run(harvester.harvest_all_categories(
            max_papers_per_category=args.max_papers
        ))
#!/usr/bin/env python3
"""
Orchestrator for staged paper processing workflow
Manages the complete pipeline from harvesting to training
"""
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
from typing import Dict, List, Optional

from hdd_harvester import HDDPaperHarvester
from staged_batch_processor import StagedBatchProcessor
from doi_database import DOIPaperDatabase
from doi_monitor import DOISystemMonitor, print_stats

logger = logging.getLogger(__name__)


class PaperProcessingOrchestrator:
    """Orchestrates the complete paper processing pipeline"""
    
    def __init__(self, config_path: str = "/mnt/hdd/candle/configs/orchestrator_config.json"):
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.doi_db = DOIPaperDatabase(self.config['doi_db_path'])
        
        self.harvester = HDDPaperHarvester(
            hdd_base_path=self.config['hdd_harvest_path'],
            doi_db_path=self.config['doi_db_path']
        )
        
        self.processor = StagedBatchProcessor(
            hdd_path=self.config['hdd_harvest_path'],
            nvme_path=self.config['nvme_process_path'],
            archive_path=self.config['hdd_archive_path'],
            doi_db_path=self.config['doi_db_path'],
            batch_size=self.config['batch_size']
        )
        
        self.monitor = DOISystemMonitor(
            doi_db=self.doi_db,
            batch_processor=self.processor,
            recall_system=self.processor.recall_system
        )
        
        # State tracking
        self.state = {
            'harvest_runs': 0,
            'processing_runs': 0,
            'last_harvest': None,
            'last_processing': None,
            'total_papers_harvested': 0,
            'total_papers_processed': 0
        }
        self._load_state()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load or create orchestrator configuration"""
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        default_config = {
            "doi_db_path": "/mnt/hdd/candle/doi_database",
            "hdd_harvest_path": "/mnt/hdd/candle/raw_papers",
            "hdd_archive_path": "/mnt/hdd/candle/processed_archive",
            "nvme_process_path": "/mnt/nvme/candle/processing",
            "batch_size": 1000,
            "categories": ["cs.AI", "cs.LG", "cs.HC", "stat.ML"],
            "harvest_schedule": {
                "max_papers_per_category": 1000,
                "days_back": 30
            },
            "processing_schedule": {
                "min_papers_for_batch": 500,
                "max_batches_per_run": 10,
                "min_nvme_space_gb": 100
            },
            "monitoring": {
                "enable_web_dashboard": True,
                "dashboard_port": 5000,
                "export_reports": True,
                "report_path": "/mnt/hdd/candle/reports"
            }
        }
        
        # Save default config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    async def run_harvest_phase(self):
        """Run the harvesting phase"""
        logger.info("=" * 60)
        logger.info("Starting Harvest Phase")
        logger.info("=" * 60)
        
        self.state['harvest_runs'] += 1
        self.state['last_harvest'] = datetime.now()
        
        # Get current stats
        initial_stats = self.harvester.get_harvest_summary()
        initial_count = initial_stats['total_papers']
        
        # Run harvest for each category
        for category in self.config['categories']:
            logger.info(f"\nHarvesting category: {category}")
            
            try:
                task = await self.harvester.harvest_category(
                    category=category,
                    max_papers=self.config['harvest_schedule']['max_papers_per_category'],
                    days_back=self.config['harvest_schedule']['days_back']
                )
                
                logger.info(f"Category {category}: {task.papers_downloaded}/{task.papers_found} papers")
                
            except Exception as e:
                logger.error(f"Failed to harvest {category}: {e}")
            
            # Small delay between categories
            await asyncio.sleep(5)
        
        # Get final stats
        final_stats = self.harvester.get_harvest_summary()
        papers_harvested = final_stats['total_papers'] - initial_count
        
        self.state['total_papers_harvested'] += papers_harvested
        
        logger.info(f"\nHarvest phase complete!")
        logger.info(f"Papers harvested: {papers_harvested:,}")
        logger.info(f"Total size: {final_stats['total_size_gb']:.2f} GB")
        logger.info(f"HDD usage: {final_stats['hdd_usage']['percent_used']:.1f}%")
        
        self._save_state()
        
        return papers_harvested
    
    async def run_processing_phase(self):
        """Run the processing phase"""
        logger.info("=" * 60)
        logger.info("Starting Processing Phase")
        logger.info("=" * 60)
        
        self.state['processing_runs'] += 1
        self.state['last_processing'] = datetime.now()
        
        # Check NVMe space
        nvme_space = self.processor.get_available_nvme_space()
        if nvme_space < self.config['processing_schedule']['min_nvme_space_gb']:
            logger.warning(f"Insufficient NVMe space: {nvme_space:.1f} GB")
            return 0
        
        # Get current stats
        initial_status = self.processor.get_processing_status()
        initial_processed = initial_status['total_papers_processed']
        
        # Process batches for each category
        total_batches = 0
        max_batches = self.config['processing_schedule']['max_batches_per_run']
        
        for category in self.config['categories']:
            # Check available papers
            available = self.processor.get_available_papers(category=category)
            
            if len(available) < self.config['processing_schedule']['min_papers_for_batch']:
                logger.info(f"Not enough papers for {category}: {len(available)}")
                continue
            
            logger.info(f"\nProcessing category: {category}")
            logger.info(f"Available papers: {len(available)}")
            
            # Calculate batches for this category
            category_batches = min(
                len(available) // self.config['batch_size'],
                max_batches - total_batches
            )
            
            if category_batches > 0:
                await self.processor.process_category(
                    category=category,
                    max_batches=category_batches
                )
                
                total_batches += category_batches
            
            if total_batches >= max_batches:
                break
        
        # Get final stats
        final_status = self.processor.get_processing_status()
        papers_processed = final_status['total_papers_processed'] - initial_processed
        
        self.state['total_papers_processed'] += papers_processed
        
        logger.info(f"\nProcessing phase complete!")
        logger.info(f"Batches processed: {total_batches}")
        logger.info(f"Papers processed: {papers_processed:,}")
        logger.info(f"Failed: {final_status['total_papers_failed']}")
        
        self._save_state()
        
        return papers_processed
    
    def generate_report(self):
        """Generate comprehensive system report"""
        
        # Gather all statistics
        harvest_stats = self.harvester.get_harvest_summary()
        process_status = self.processor.get_processing_status()
        db_stats = self.doi_db.get_statistics()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'orchestrator_state': self.state,
            'harvest_summary': harvest_stats,
            'processing_status': process_status,
            'database_stats': db_stats,
            'system_health': self._calculate_system_health()
        }
        
        # Save report if configured
        if self.config['monitoring']['export_reports']:
            report_path = Path(self.config['monitoring']['report_path'])
            report_path.mkdir(parents=True, exist_ok=True)
            
            report_file = report_path / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Report saved to: {report_file}")
        
        return report
    
    def _calculate_system_health(self) -> Dict:
        """Calculate overall system health"""
        
        issues = []
        score = 100
        
        # Check HDD space
        harvest_stats = self.harvester.get_harvest_summary()
        hdd_usage = harvest_stats['hdd_usage']['percent_used']
        
        if hdd_usage > 90:
            issues.append("HDD space critical")
            score -= 30
        elif hdd_usage > 80:
            issues.append("HDD space high") 
            score -= 10
        
        # Check processing backlog
        available = len(self.processor.get_available_papers())
        if available > 10000:
            issues.append(f"Large backlog: {available} papers")
            score -= 20
        
        # Check processing failures
        process_status = self.processor.get_processing_status()
        if process_status['total_papers_failed'] > process_status['total_papers_processed'] * 0.1:
            issues.append("High failure rate")
            score -= 20
        
        return {
            'score': max(0, score),
            'status': 'healthy' if score >= 70 else 'warning' if score >= 40 else 'critical',
            'issues': issues
        }
    
    def _save_state(self):
        """Save orchestrator state"""
        state_file = Path(self.config['doi_db_path']).parent / "orchestrator_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def _load_state(self):
        """Load orchestrator state"""
        state_file = Path(self.config['doi_db_path']).parent / "orchestrator_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                saved_state = json.load(f)
                self.state.update(saved_state)
    
    async def run_full_pipeline(self):
        """Run complete harvest and process pipeline"""
        logger.info("Starting full pipeline run...")
        
        # Phase 1: Harvest
        papers_harvested = await self.run_harvest_phase()
        
        # Phase 2: Process (if enough papers)
        papers_processed = 0
        if papers_harvested >= self.config['processing_schedule']['min_papers_for_batch']:
            # Wait a bit for system to settle
            await asyncio.sleep(60)
            
            papers_processed = await self.run_processing_phase()
        else:
            logger.info(f"Not enough new papers for processing: {papers_harvested}")
        
        # Generate report
        report = self.generate_report()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Run Complete")
        logger.info("=" * 60)
        logger.info(f"Papers harvested: {papers_harvested:,}")
        logger.info(f"Papers processed: {papers_processed:,}")
        logger.info(f"System health: {report['system_health']['status']} ({report['system_health']['score']}%)")
        
        if report['system_health']['issues']:
            logger.warning("Issues detected:")
            for issue in report['system_health']['issues']:
                logger.warning(f"  - {issue}")
    
    def start_monitoring_dashboard(self):
        """Start the web monitoring dashboard"""
        if self.config['monitoring']['enable_web_dashboard']:
            from doi_monitor import create_web_dashboard
            
            app = create_web_dashboard(self.monitor, port=self.config['monitoring']['dashboard_port'])
            logger.info(f"Starting dashboard on port {self.config['monitoring']['dashboard_port']}")
            app.run(host='0.0.0.0', port=self.config['monitoring']['dashboard_port'])


# CLI interface
async def main():
    parser = argparse.ArgumentParser(description="Paper Processing Orchestrator")
    parser.add_argument("--harvest", action="store_true", help="Run harvest phase only")
    parser.add_argument("--process", action="store_true", help="Run processing phase only")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--dashboard", action="store_true", help="Start monitoring dashboard")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--status", action="store_true", help="Show current status")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = PaperProcessingOrchestrator()
    
    if args.harvest:
        await orchestrator.run_harvest_phase()
    elif args.process:
        await orchestrator.run_processing_phase()
    elif args.full:
        await orchestrator.run_full_pipeline()
    elif args.dashboard:
        orchestrator.start_monitoring_dashboard()
    elif args.report:
        report = orchestrator.generate_report()
        print(json.dumps(report, indent=2, default=str))
    elif args.status:
        print_stats(orchestrator.monitor)
    else:
        # Default: show status
        print_stats(orchestrator.monitor)


if __name__ == "__main__":
    asyncio.run(main())
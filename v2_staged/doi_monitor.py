"""
Monitor paper processing and storage with web dashboard
"""
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from collections import defaultdict
import asyncio

from .doi_database import DOIPaperDatabase
from .batch_processor import BatchPaperProcessor
from .paper_recall import PaperRecallSystem

# Try to import Flask for web dashboard
try:
    from flask import Flask, render_template, jsonify, request
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    logging.warning("Flask not available, web dashboard disabled")

logger = logging.getLogger(__name__)


class DOISystemMonitor:
    """Monitor paper processing and storage"""
    
    def __init__(self, 
                 doi_db: DOIPaperDatabase,
                 batch_processor: Optional[BatchPaperProcessor] = None,
                 recall_system: Optional[PaperRecallSystem] = None):
        
        self.doi_db = doi_db
        self.batch_processor = batch_processor
        self.recall_system = recall_system
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.max_history_items = 1000
        
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        
        # Get database statistics
        db_stats = self.doi_db.get_statistics()
        
        # Storage analysis
        nvme_usage = shutil.disk_usage("/mnt/nvme") if Path("/mnt/nvme").exists() else None
        hdd_usage = shutil.disk_usage("/mnt/hdd") if Path("/mnt/hdd").exists() else None
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'papers': {
                'total': db_stats['total_papers'],
                'processed': db_stats['processed_papers'],
                'pending': db_stats['unprocessed_papers'],
                'duplicates_prevented': self._get_duplicate_count(),
                'categories': db_stats['categories']
            },
            'storage': {
                'nvme': self._format_storage_stats(nvme_usage) if nvme_usage else None,
                'hdd': {
                    **self._format_storage_stats(hdd_usage) if hdd_usage else {},
                    'papers_gb': db_stats['storage']['total_gb']
                }
            },
            'processing': {
                'current_batch': None,
                'processing_speed': self._get_processing_speed(),
                'avg_paper_time': self._get_avg_processing_time()
            },
            'cache': None
        }
        
        # Add batch processor stats
        if self.batch_processor:
            batch_summary = self.batch_processor.get_batch_summary()
            stats['processing']['current_batch'] = batch_summary
        
        # Add recall system stats
        if self.recall_system:
            stats['cache'] = self.recall_system.get_cache_stats()
        
        # Recent activity
        stats['recent_activity'] = db_stats.get('recent_downloads', [])
        
        # System health
        stats['health'] = self._calculate_health_score(stats)
        
        return stats
    
    def _format_storage_stats(self, usage) -> Dict[str, Any]:
        """Format storage statistics"""
        return {
            'total_gb': usage.total / (1024**3),
            'used_gb': usage.used / (1024**3),
            'free_gb': usage.free / (1024**3),
            'percent_used': (usage.used / usage.total * 100)
        }
    
    def _get_duplicate_count(self) -> int:
        """Get total duplicate attempts"""
        report = self.doi_db.get_duplicate_report(days=30)
        return report['total_attempts']
    
    def _get_processing_speed(self) -> float:
        """Calculate average processing speed (papers/hour)"""
        # Get recent processing history
        with self.doi_db._get_cursor() as cursor:
            result = cursor.execute("""
                SELECT COUNT(*) as count
                FROM papers
                WHERE process_date > datetime('now', '-24 hours')
            """).fetchone()
            
            papers_24h = result['count']
            return papers_24h / 24.0 if papers_24h > 0 else 0
    
    def _get_avg_processing_time(self) -> float:
        """Calculate average time to process a paper"""
        # This would need more detailed timing data
        # For now, return estimate based on batch size
        return 30.0  # 30 seconds per paper estimate
    
    def _calculate_health_score(self, stats: Dict) -> Dict[str, Any]:
        """Calculate system health score"""
        issues = []
        score = 100
        
        # Check storage
        if stats['storage']['nvme']:
            nvme_percent = stats['storage']['nvme']['percent_used']
            if nvme_percent > 90:
                issues.append("NVMe storage critical")
                score -= 30
            elif nvme_percent > 80:
                issues.append("NVMe storage high")
                score -= 10
        
        # Check processing backlog
        pending = stats['papers']['pending']
        if pending > 10000:
            issues.append(f"Large backlog: {pending} papers")
            score -= 20
        elif pending > 5000:
            issues.append(f"Moderate backlog: {pending} papers")
            score -= 10
        
        # Check cache hit rate
        if stats['cache'] and stats['cache']['total_recalls'] > 100:
            hit_rate = stats['cache']['cache_hit_rate']
            if hit_rate < 0.3:
                issues.append(f"Low cache hit rate: {hit_rate:.1%}")
                score -= 10
        
        return {
            'score': max(0, score),
            'status': 'healthy' if score >= 70 else 'warning' if score >= 40 else 'critical',
            'issues': issues
        }
    
    def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        
        # Processing timeline
        with self.doi_db._get_cursor() as cursor:
            timeline = cursor.execute("""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', process_date) as hour,
                    COUNT(*) as papers_processed,
                    AVG(summary_quality_score) as avg_quality
                FROM papers
                WHERE process_date > datetime('now', ? || ' hours')
                GROUP BY hour
                ORDER BY hour
            """, (-hours,)).fetchall()
            
        # Category breakdown
        category_stats = cursor.execute("""
            SELECT 
                categories,
                COUNT(*) as total,
                SUM(CASE WHEN used_for_training = 1 THEN 1 ELSE 0 END) as processed
            FROM papers
            WHERE download_date > datetime('now', '-30 days')
            GROUP BY categories
        """).fetchall()
        
        return {
            'timeline': [dict(t) for t in timeline],
            'categories': self._parse_category_stats(category_stats),
            'efficiency': {
                'deduplication_rate': self._calculate_dedup_rate(),
                'processing_success_rate': self._calculate_success_rate(),
                'storage_efficiency': self._calculate_storage_efficiency()
            }
        }
    
    def _parse_category_stats(self, raw_stats) -> List[Dict]:
        """Parse category statistics"""
        category_totals = defaultdict(lambda: {'total': 0, 'processed': 0})
        
        for row in raw_stats:
            if row['categories']:
                categories = json.loads(row['categories'])
                for cat in categories:
                    category_totals[cat]['total'] += row['total']
                    category_totals[cat]['processed'] += row['processed']
        
        # Convert to list and calculate percentages
        result = []
        for cat, stats in category_totals.items():
            result.append({
                'category': cat,
                'total': stats['total'],
                'processed': stats['processed'],
                'percentage': (stats['processed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            })
        
        return sorted(result, key=lambda x: x['total'], reverse=True)
    
    def _calculate_dedup_rate(self) -> float:
        """Calculate deduplication effectiveness"""
        stats = self.doi_db.get_statistics()
        duplicates = self._get_duplicate_count()
        total_attempts = stats['total_papers'] + duplicates
        
        return (duplicates / total_attempts * 100) if total_attempts > 0 else 0
    
    def _calculate_success_rate(self) -> float:
        """Calculate processing success rate"""
        # Would need error tracking for accurate calculation
        return 95.0  # Placeholder
    
    def _calculate_storage_efficiency(self) -> float:
        """Calculate storage efficiency (compression ratio)"""
        # Would need to track original vs compressed sizes
        return 1.0  # No compression currently
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get system alerts"""
        alerts = []
        stats = self.get_dashboard_stats()
        
        # Storage alerts
        if stats['storage']['nvme'] and stats['storage']['nvme']['percent_used'] > 85:
            alerts.append({
                'level': 'warning' if stats['storage']['nvme']['percent_used'] < 95 else 'critical',
                'message': f"NVMe storage at {stats['storage']['nvme']['percent_used']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # Processing alerts
        if stats['papers']['pending'] > 5000:
            alerts.append({
                'level': 'warning',
                'message': f"{stats['papers']['pending']} papers pending processing",
                'timestamp': datetime.now().isoformat()
            })
        
        # Performance alerts
        speed = stats['processing']['processing_speed']
        if speed < 10:  # Less than 10 papers/hour
            alerts.append({
                'level': 'warning',
                'message': f"Low processing speed: {speed:.1f} papers/hour",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def export_report(self, output_path: str, days: int = 7):
        """Export detailed system report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_days': days,
            'system_stats': self.get_dashboard_stats(),
            'performance_metrics': self.get_performance_metrics(hours=days*24),
            'duplicate_report': self.doi_db.get_duplicate_report(days),
            'alerts': self.get_alerts()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Exported report to {output_path}")


def create_web_dashboard(monitor: DOISystemMonitor, port: int = 5000):
    """Create Flask web dashboard"""
    if not HAS_FLASK:
        logger.error("Flask not available, cannot create web dashboard")
        return None
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def index():
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>CANDLE DOI System Monitor</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; }
                .metric { font-size: 2em; font-weight: bold; color: #333; }
                .label { color: #666; font-size: 0.9em; }
                .alert { padding: 10px; margin: 10px 0; border-radius: 4px; }
                .alert-warning { background: #fff3cd; border: 1px solid #ffeeba; }
                .alert-critical { background: #f8d7da; border: 1px solid #f5c6cb; }
                .health-good { color: #28a745; }
                .health-warning { color: #ffc107; }
                .health-critical { color: #dc3545; }
            </style>
        </head>
        <body>
            <h1>CANDLE DOI System Monitor</h1>
            <div id="alerts"></div>
            <div class="dashboard" id="dashboard"></div>
            <div style="margin-top: 40px;">
                <canvas id="processingChart" width="400" height="200"></canvas>
            </div>
            
            <script>
            async function updateDashboard() {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                // Update metrics
                const dashboard = document.getElementById('dashboard');
                dashboard.innerHTML = `
                    <div class="card">
                        <div class="label">Total Papers</div>
                        <div class="metric">${stats.papers.total.toLocaleString()}</div>
                        <div class="label">Processed: ${stats.papers.processed.toLocaleString()}</div>
                        <div class="label">Pending: ${stats.papers.pending.toLocaleString()}</div>
                    </div>
                    <div class="card">
                        <div class="label">NVMe Storage</div>
                        <div class="metric">${stats.storage.nvme ? stats.storage.nvme.percent_used.toFixed(1) + '%' : 'N/A'}</div>
                        <div class="label">Free: ${stats.storage.nvme ? stats.storage.nvme.free_gb.toFixed(1) + ' GB' : 'N/A'}</div>
                    </div>
                    <div class="card">
                        <div class="label">HDD Archive</div>
                        <div class="metric">${stats.storage.hdd ? stats.storage.hdd.papers_gb.toFixed(1) + ' GB' : 'N/A'}</div>
                        <div class="label">Papers stored</div>
                    </div>
                    <div class="card">
                        <div class="label">Processing Speed</div>
                        <div class="metric">${stats.processing.processing_speed.toFixed(1)}</div>
                        <div class="label">papers/hour</div>
                    </div>
                    <div class="card">
                        <div class="label">Cache Hit Rate</div>
                        <div class="metric">${stats.cache ? (stats.cache.cache_hit_rate * 100).toFixed(1) + '%' : 'N/A'}</div>
                        <div class="label">${stats.cache ? stats.cache.total_recalls + ' total recalls' : ''}</div>
                    </div>
                    <div class="card">
                        <div class="label">System Health</div>
                        <div class="metric health-${stats.health.status}">${stats.health.score}%</div>
                        <div class="label">${stats.health.status}</div>
                    </div>
                `;
                
                // Update alerts
                const alertsResponse = await fetch('/api/alerts');
                const alerts = await alertsResponse.json();
                const alertsDiv = document.getElementById('alerts');
                alertsDiv.innerHTML = alerts.map(alert => 
                    `<div class="alert alert-${alert.level}">${alert.message}</div>`
                ).join('');
                
                // Update chart
                const metricsResponse = await fetch('/api/metrics?hours=24');
                const metrics = await metricsResponse.json();
                updateChart(metrics.timeline);
            }
            
            let chart = null;
            function updateChart(timeline) {
                const ctx = document.getElementById('processingChart').getContext('2d');
                
                if (chart) {
                    chart.destroy();
                }
                
                chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: timeline.map(t => new Date(t.hour).toLocaleTimeString()),
                        datasets: [{
                            label: 'Papers Processed',
                            data: timeline.map(t => t.papers_processed),
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Processing Activity (24 hours)'
                            }
                        }
                    }
                });
            }
            
            // Update every 10 seconds
            updateDashboard();
            setInterval(updateDashboard, 10000);
            </script>
        </body>
        </html>
        '''
    
    @app.route('/api/stats')
    def api_stats():
        return jsonify(monitor.get_dashboard_stats())
    
    @app.route('/api/metrics')
    def api_metrics():
        hours = request.args.get('hours', 24, type=int)
        return jsonify(monitor.get_performance_metrics(hours))
    
    @app.route('/api/alerts')
    def api_alerts():
        return jsonify(monitor.get_alerts())
    
    @app.route('/api/papers/search')
    def api_search():
        criteria = {
            'query': request.args.get('q'),
            'categories': request.args.getlist('category'),
            'processed': request.args.get('processed', type=bool),
            'limit': request.args.get('limit', 100, type=int)
        }
        papers = monitor.doi_db.search_papers(**criteria)
        return jsonify(papers)
    
    return app


# CLI monitoring functions
def print_stats(monitor: DOISystemMonitor):
    """Print statistics to console"""
    stats = monitor.get_dashboard_stats()
    
    print("\n=== CANDLE DOI System Status ===")
    print(f"Time: {stats['timestamp']}")
    print("\nPapers:")
    print(f"  Total: {stats['papers']['total']:,}")
    print(f"  Processed: {stats['papers']['processed']:,}")
    print(f"  Pending: {stats['papers']['pending']:,}")
    print(f"  Duplicates Prevented: {stats['papers']['duplicates_prevented']:,}")
    
    if stats['storage']['nvme']:
        print("\nNVMe Storage:")
        print(f"  Used: {stats['storage']['nvme']['used_gb']:.1f} GB ({stats['storage']['nvme']['percent_used']:.1f}%)")
        print(f"  Free: {stats['storage']['nvme']['free_gb']:.1f} GB")
    
    if stats['storage']['hdd']:
        print("\nHDD Archive:")
        print(f"  Papers: {stats['storage']['hdd']['papers_gb']:.1f} GB")
        
    print(f"\nProcessing Speed: {stats['processing']['processing_speed']:.1f} papers/hour")
    
    if stats['cache']:
        print(f"\nCache Performance:")
        print(f"  Hit Rate: {stats['cache']['cache_hit_rate']:.1%}")
        print(f"  Size: {stats['cache']['cache_size_gb']:.1f} GB")
    
    print(f"\nSystem Health: {stats['health']['status']} ({stats['health']['score']}%)")
    if stats['health']['issues']:
        print("Issues:")
        for issue in stats['health']['issues']:
            print(f"  - {issue}")


async def monitor_loop(monitor: DOISystemMonitor, interval: int = 60):
    """Run monitoring loop"""
    while True:
        print_stats(monitor)
        
        # Check for alerts
        alerts = monitor.get_alerts()
        if alerts:
            print("\n!!! ALERTS !!!")
            for alert in alerts:
                print(f"[{alert['level'].upper()}] {alert['message']}")
        
        await asyncio.sleep(interval)
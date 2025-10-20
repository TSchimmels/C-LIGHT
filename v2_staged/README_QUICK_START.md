# Version 2: Staged Processing Quick Start

This version is optimized for downloading millions of papers efficiently on low-power hardware, then processing them in batches when GPU resources are available.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Low-Power     │     │   GPU Server    │     │   Storage       │
│   Harvester     │     │   (When Ready)  │     │   Archive       │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ ArXiv API       │     │ Staged Batch    │     │ Processed       │
│ ↓               │     │ Processor       │     │ Papers +        │
│ Download to HDD │ --> │ ↓               │ --> │ Embeddings      │
│ ↓               │     │ Move to NVMe    │     │ on HDD          │
│ DOI Database    │     │ ↓               │     │                 │
│                 │     │ Process & Train │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Quick Setup

### 1. Initial Configuration

```bash
# Create directories
sudo mkdir -p /mnt/hdd/candle/{raw_papers,processed_archive,doi_database,configs}
sudo mkdir -p /mnt/nvme/candle/processing

# Set permissions
sudo chown -R $USER:$USER /mnt/hdd/candle /mnt/nvme/candle
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Harvesting (Low-Power Server)

```bash
# Run continuous harvesting (runs 24/7 on low power)
python hdd_harvester.py --continuous

# Or harvest specific categories
python hdd_harvester.py --categories cs.AI cs.LG --max-papers 5000
```

### 4. Check Harvest Status

```bash
# View harvest summary
python hdd_harvester.py --summary

# Example output:
# === HDD Harvest Summary ===
# Total papers: 125,432
# Total size: 1,254.3 GB
# Duplicates skipped: 15,234
# HDD usage: 45.2%
```

### 5. Process Papers (GPU Server)

```bash
# Process a single category
python staged_batch_processor.py --category cs.AI --batches 5

# Check processing status
python staged_batch_processor.py --status
```

### 6. Use Orchestrator (Recommended)

The orchestrator manages the complete pipeline:

```bash
# Run full pipeline (harvest + process)
python orchestrator.py --full

# Run harvest only
python orchestrator.py --harvest

# Run processing only  
python orchestrator.py --process

# Start monitoring dashboard
python orchestrator.py --dashboard
```

## Energy-Efficient Schedule

### Low-Power Server (24/7)
```python
# In harvest_config.json
{
  "harvest_schedule": {
    "enabled": true,
    "times": ["02:00", "14:00"],  # Harvest twice daily
    "max_papers_per_run": 5000
  }
}
```

### GPU Server (Weekly)
```bash
# Cron job to process weekly
0 2 * * 0 /usr/bin/python3 /path/to/orchestrator.py --process
```

## Storage Management

### Monitor Storage
```python
# Check HDD usage
python orchestrator.py --status

# Clean old staging files
from staged_batch_processor import StagedBatchProcessor
processor = StagedBatchProcessor()
processor.recall_system.cleanup_staging()
```

### Archive Strategy
- Raw papers: `/mnt/hdd/candle/raw_papers/YYYY/MM/category/`
- Processed: `/mnt/hdd/candle/processed_archive/batch_id/`
- Metadata: Stored in DOI database

## Common Workflows

### 1. Initial Bulk Download
```bash
# Download last 6 months of papers
python hdd_harvester.py \
  --categories cs.AI cs.LG cs.HC q-bio.NC \
  --max-papers 10000 \
  --days-back 180
```

### 2. Process in Batches
```bash
# Process 50 batches (50,000 papers)
for i in {1..50}; do
  python staged_batch_processor.py --batches 1
  sleep 300  # 5 min cooldown
done
```

### 3. Continuous Operation
```bash
# Terminal 1: Continuous harvesting
python hdd_harvester.py --continuous

# Terminal 2: Weekly processing (via cron)
# 0 2 * * 0 python orchestrator.py --process
```

## Performance Tuning

### Harvest Server
```json
{
  "concurrent_downloads": 2,      # Low for energy efficiency
  "chunk_size": 8192,            # 8KB chunks
  "delay_between_papers": 1,     # Be nice to ArXiv
  "papers_per_batch": 1000      # Manageable chunks
}
```

### GPU Server
```json
{
  "batch_size": 1000,           # Papers per batch
  "nvme_space_gb": 500,         # Reserved NVMe space
  "min_papers_for_batch": 500,  # Minimum to start processing
  "max_batches_per_run": 10     # Limit per session
}
```

## Monitoring

### Web Dashboard
```bash
python orchestrator.py --dashboard
# Open http://localhost:5000
```

### CLI Monitoring
```bash
# Overall status
python orchestrator.py --status

# Detailed report
python orchestrator.py --report > system_report.json
```

## Troubleshooting

### "No unprocessed papers available"
- Check if papers exist: `python hdd_harvester.py --summary`
- Verify DOI database: `sqlite3 /mnt/hdd/candle/doi_database/paper_metadata.db`

### "Insufficient NVMe space"
- Check space: `df -h /mnt/nvme`
- Clean staging: `rm -rf /mnt/nvme/candle/processing/staging/*`

### "High failure rate"
- Check logs: `tail -f candle_processor.log`
- Verify paper integrity: `python verify_papers.py`

## Best Practices

1. **Start Small**: Test with 100 papers before scaling up
2. **Monitor Storage**: Keep HDD < 80% full
3. **Schedule Wisely**: Process during off-peak hours
4. **Backup DOI DB**: Critical for preventing duplicates
5. **Clean Regularly**: Remove old staging files

## Example Production Setup

```bash
# 1. Setup harvest server (Raspberry Pi 4)
ssh pi@harvest-server
python hdd_harvester.py --continuous &

# 2. Setup GPU server (8x V100)
ssh gpu-server
# Add to crontab:
0 2 * * 0 cd /candle && python orchestrator.py --process

# 3. Monitor remotely
ssh -L 5000:localhost:5000 gpu-server
python orchestrator.py --dashboard
```

## Next Steps

1. Review [full documentation](../README_DOI_SYSTEM.md)
2. Configure for your categories of interest
3. Set up automated scheduling
4. Enable monitoring alerts
5. Start harvesting!
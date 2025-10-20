# CANDLE DOI Paper Management System

The CANDLE DOI system provides two implementation strategies for managing academic papers:

## Version 1: Real-time Processing (v1_realtime/)
**Use Case**: Continuous operation with GPU server always running

### Architecture
```
ArXiv API → Download to NVMe → Process Immediately → Train Models → Archive to HDD
```

### Components
- `arxiv_harvester.py`: Downloads papers directly to NVMe
- `batch_processor.py`: Processes papers immediately after download
- `doi_database.py`: Tracks all papers to prevent duplicates
- `paper_recall.py`: Retrieves archived papers when needed
- `doi_monitor.py`: Real-time monitoring dashboard

### Advantages
- Low latency from discovery to processing
- Continuous model updates
- Real-time insights

### Disadvantages  
- High energy consumption (GPU server always on)
- Higher operational costs
- Not suitable for massive paper collections

### Usage
```bash
cd v1_realtime
python -m batch_processor --categories cs.AI cs.LG --continuous
```

## Version 2: Staged Processing (v2_staged/)
**Use Case**: Energy-efficient bulk processing of millions of papers

### Architecture
```
Stage 1 (Low-power server): ArXiv API → Download to HDD → Track in DOI DB
Stage 2 (GPU server): Move to NVMe → Batch Process → Train Models → Archive
```

### Components
- `hdd_harvester.py`: Downloads papers to HDD on low-power server
- `staged_batch_processor.py`: Moves papers to NVMe and processes in batches
- `doi_database.py`: Shared component for tracking papers
- `paper_recall.py`: Shared component for paper retrieval
- `doi_monitor.py`: Shared monitoring dashboard

### Advantages
- Energy efficient (GPU only runs when needed)
- Cost effective for large collections
- Can download millions of papers without GPU
- Flexible processing schedule

### Disadvantages
- Higher latency from discovery to processing
- Requires manual batch triggering
- More complex workflow

### Usage

**Stage 1: Continuous harvesting (low-power server)**
```bash
cd v2_staged
python hdd_harvester.py --continuous
```

**Stage 2: Batch processing (GPU server)**
```bash
cd v2_staged
python staged_batch_processor.py --category cs.AI --batches 10
```

## Choosing Between Versions

### Use Version 1 when:
- You have a dedicated GPU server
- Need real-time processing
- Paper volume is moderate (< 10K/day)
- Low latency is critical

### Use Version 2 when:
- Want to minimize energy costs
- Processing millions of papers
- Can batch process weekly/monthly
- Building large training datasets

## Storage Architecture

Both versions use the same storage strategy:

### NVMe Storage (Fast)
- DOI database for deduplication
- Active processing workspace
- LRU cache for frequently accessed papers
- Model checkpoints

### HDD Storage (Large)
- Raw downloaded papers (v2)
- Processed paper archive
- Historical data
- Backup and recovery

## DOI Database Features

The DOI database is the core component shared by both versions:

- **Deduplication**: Prevents downloading/processing the same paper twice
- **Tracking**: Complete lifecycle tracking from download to archive
- **Search**: Fast queries by DOI, ArXiv ID, category, date range
- **Metadata**: Stores embeddings, causal relations, quality scores
- **Integration**: Works with both processing strategies

## Monitoring

Both versions include comprehensive monitoring:

```bash
# Web dashboard
python -c "from doi_monitor import create_web_dashboard; app = create_web_dashboard(monitor); app.run()"

# CLI status
python doi_monitor.py --status
```

## Migration Between Versions

You can migrate from v1 to v2 or vice versa:

```bash
# Export from v1
cd v1_realtime
python doi_database.py --export /tmp/papers.json

# Import to v2  
cd v2_staged
python doi_database.py --import /tmp/papers.json
```

## Hardware Requirements

### Version 1 (Real-time)
- **GPU Server**: 8x V100 32GB or better
- **NVMe**: 2-4TB for processing
- **HDD**: 32TB+ for archive
- **Network**: 1Gbps+ for continuous downloads

### Version 2 (Staged)
- **Harvest Server**: Low-power CPU (Raspberry Pi 4+ works)
- **HDD**: 32TB+ for raw storage
- **GPU Server**: 8x V100 32GB (only when processing)
- **NVMe**: 2-4TB for batch processing

## Best Practices

1. **Start with v2** for initial bulk collection
2. **Switch to v1** for ongoing real-time updates
3. **Monitor storage** regularly
4. **Schedule processing** during off-peak hours
5. **Backup DOI database** weekly

## Example Workflows

### Research Project (v2)
```bash
# 1. Define categories of interest
echo '["cs.AI", "cs.LG", "stat.ML"]' > categories.json

# 2. Harvest papers for 1 month
python hdd_harvester.py --continuous

# 3. Process in batches weekly
python staged_batch_processor.py --batches 50

# 4. Train models on processed data
python train_models.py --checkpoint latest
```

### Production System (v1)
```bash
# 1. Start continuous processing
python batch_processor.py --continuous &

# 2. Monitor system
python doi_monitor.py --web

# 3. Set up alerts
python doi_monitor.py --alert-email admin@example.com
```

## Performance Metrics

### Version 1
- Download rate: 10-50 papers/minute
- Processing rate: 5-20 papers/minute  
- Model update frequency: Every 1000 papers
- Energy usage: ~2kW continuous

### Version 2
- Download rate: 100-500 papers/minute (no processing)
- Batch processing: 10K papers/hour
- Model update frequency: Per batch
- Energy usage: ~100W (harvest) + 2kW (processing)
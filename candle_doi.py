#!/usr/bin/env python3
"""
CANDLE DOI Paper Management System Entry Point

Usage:
    python candle_doi.py --help
"""
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="CANDLE DOI Paper Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CANDLE provides two paper processing strategies:

Version 1 - Real-time Processing (v1_realtime/):
  For continuous GPU operation with immediate processing
  Usage: cd v1_realtime && python batch_processor.py --continuous

Version 2 - Staged Processing (v2_staged/):  
  For energy-efficient bulk processing
  Stage 1: cd v2_staged && python hdd_harvester.py --continuous
  Stage 2: cd v2_staged && python orchestrator.py --process

See README.md for detailed documentation.
        """
    )
    
    parser.add_argument(
        "--version",
        choices=["1", "2"],
        help="Show quick start for version 1 or 2"
    )
    
    args = parser.parse_args()
    
    if args.version == "1":
        print("\n=== Version 1: Real-time Processing ===\n")
        print("Directory: v1_realtime/")
        print("\nQuick Start:")
        print("  cd v1_realtime")
        print("  python batch_processor.py --categories cs.AI cs.LG --continuous")
        print("\nKey Features:")
        print("  - Downloads directly to NVMe")
        print("  - Processes immediately")
        print("  - Requires GPU server always on")
        print("  - Best for real-time insights")
        
    elif args.version == "2":
        print("\n=== Version 2: Staged Processing ===\n")
        print("Directory: v2_staged/")
        print("\nQuick Start:")
        print("  # Stage 1 - Harvest (low-power server)")
        print("  cd v2_staged")
        print("  python hdd_harvester.py --continuous")
        print("\n  # Stage 2 - Process (GPU server)")
        print("  python orchestrator.py --process")
        print("\nKey Features:")
        print("  - Downloads to HDD on low-power server")
        print("  - Processes in batches when GPU available")
        print("  - Energy efficient (90% savings)")
        print("  - Best for millions of papers")
    
    else:
        print("\nCANDLE DOI Paper Management System")
        print("==================================\n")
        print("Choose your processing strategy:\n")
        print("Version 1 (v1_realtime/):")
        print("  Real-time processing for immediate insights")
        print("  Requires dedicated GPU server\n")
        print("Version 2 (v2_staged/):")
        print("  Energy-efficient staged processing")
        print("  Download on low-power, process when ready\n")
        print("Run with --version 1 or --version 2 for quick start")
        print("\nFull documentation: README.md")


if __name__ == "__main__":
    main()
"""
C-LIGHT Harvesting Module
Paper collection from multiple sources
"""

from .arxiv_harvester import ArxivHarvester, ArxivCategory, create_harvester
from .doi_database import DOIPaperDatabase

__all__ = [
    'ArxivHarvester',
    'ArxivCategory',
    'create_harvester',
    'DOIPaperDatabase'
]

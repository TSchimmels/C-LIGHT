"""
CANDLE Intelligent Agents Module
Automated agents for data harvesting, processing, and system maintenance
"""

from .arxiv_harvester import (
    ArxivHarvester,
    ArxivHarvestingAgent,
    create_arxiv_agent,
    ArxivPaper,
    DuplicateDetector,
    DomainClassifier,
    CausalRelationExtractor,
    BehavioralIndicatorExtractor
)

__all__ = [
    'ArxivHarvester',
    'ArxivHarvestingAgent', 
    'create_arxiv_agent',
    'ArxivPaper',
    'DuplicateDetector',
    'DomainClassifier',
    'CausalRelationExtractor',
    'BehavioralIndicatorExtractor'
]
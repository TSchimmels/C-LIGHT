"""
ArXiv Harvesting for C-LIGHT
Standalone implementation - no CANDLE dependencies
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import arxiv

from ..core.base_types import Paper, ProcessingStatus, KnowledgeDomain
from .doi_database import DOIPaperDatabase

logger = logging.getLogger(__name__)


class ArxivCategory(Enum):
    """Relevant arXiv categories for behavioral and cognitive science"""
    # Computer Science
    CS_AI = "cs.AI"  # Artificial Intelligence
    CS_HC = "cs.HC"  # Human-Computer Interaction
    CS_CY = "cs.CY"  # Computers and Society
    CS_LG = "cs.LG"  # Machine Learning
    CS_NE = "cs.NE"  # Neural and Evolutionary Computing
    CS_CL = "cs.CL"  # Computation and Language
    CS_SI = "cs.SI"  # Social and Information Networks

    # Quantitative Biology
    Q_BIO_NC = "q-bio.NC"  # Neurons and Cognition
    Q_BIO_QM = "q-bio.QM"  # Quantitative Methods
    Q_BIO_BM = "q-bio.BM"  # Biomolecules
    Q_BIO_CB = "q-bio.CB"  # Cell Behavior
    Q_BIO_TO = "q-bio.TO"  # Tissues and Organs
    Q_BIO_MN = "q-bio.MN"  # Molecular Networks
    Q_BIO_SC = "q-bio.SC"  # Subcellular Processes
    Q_BIO_GN = "q-bio.GN"  # Genomics

    # Physics
    PHYSICS_BIO = "physics.bio-ph"  # Biological Physics
    PHYSICS_MED = "physics.med-ph"  # Medical Physics
    PHYSICS_SOC = "physics.soc-ph"  # Physics and Society
    PHYSICS_DATA = "physics.data-an"  # Data Analysis
    QUANT_PH = "quant-ph"  # Quantum Physics

    # Economics
    ECON_GN = "econ.GN"  # General Economics

    # Statistics
    STAT_AP = "stat.AP"  # Applications
    STAT_ML = "stat.ML"  # Machine Learning
    STAT_ME = "stat.ME"  # Methodology


class ArxivHarvester:
    """
    Standalone ArXiv harvester for C-LIGHT
    Downloads papers and stores metadata
    """

    def __init__(
        self,
        storage_path: str = "/mnt/hdd/c-light/raw_papers",
        db_path: str = "/mnt/nvme/c-light/doi_database"
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.doi_db = DOIPaperDatabase(db_path)

        # Stats tracking
        self.stats = {
            'downloaded': 0,
            'duplicates': 0,
            'errors': 0
        }

    def harvest(
        self,
        categories: List[str],
        max_papers: int = 1000,
        days_back: int = 7,
        keywords: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Harvest papers from ArXiv

        Args:
            categories: List of arXiv category strings (e.g., ['cs.AI', 'q-bio.NC'])
            max_papers: Maximum number of papers to download
            days_back: How many days back to search
            keywords: Optional keywords to filter by

        Returns:
            List of downloaded Paper objects
        """
        papers = []

        for category in categories:
            logger.info(f"Harvesting category: {category}")

            # Build query
            query_parts = [f"cat:{category}"]

            if keywords:
                keyword_query = " OR ".join(f'all:"{kw}"' for kw in keywords)
                query_parts.append(f"({keyword_query})")

            query = " AND ".join(query_parts)

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Search ArXiv
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=max_papers,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )

                for result in search.results():
                    # Check if within date range
                    if result.published < start_date:
                        continue

                    # Check for duplicates
                    paper_id = result.entry_id.split('/')[-1]
                    if self.doi_db.paper_exists(paper_id):
                        self.stats['duplicates'] += 1
                        logger.debug(f"Skipping duplicate: {paper_id}")
                        continue

                    # Download paper
                    paper = self._download_paper(result)
                    if paper:
                        papers.append(paper)
                        self.stats['downloaded'] += 1

                        # Add to database
                        self.doi_db.add_paper(
                            doi=paper.paper_id,
                            arxiv_id=paper.arxiv_id,
                            title=paper.title,
                            authors=json.dumps(paper.authors),
                            abstract_hash=hashlib.md5(paper.abstract.encode()).hexdigest(),
                            categories=json.dumps(paper.categories),
                            pdf_url=paper.pdf_url,
                            published_date=paper.published_date.isoformat() if paper.published_date else None,
                            hdd_path=str(self._get_paper_path(paper))
                        )

                    # Rate limiting
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error harvesting category {category}: {e}")
                self.stats['errors'] += 1

        logger.info(f"Harvest complete. Downloaded: {self.stats['downloaded']}, "
                   f"Duplicates: {self.stats['duplicates']}, Errors: {self.stats['errors']}")

        return papers

    def _download_paper(self, result: arxiv.Result) -> Optional[Paper]:
        """Download a single paper from ArXiv"""
        try:
            paper_id = result.entry_id.split('/')[-1]

            # Create Paper object
            paper = Paper(
                paper_id=paper_id,
                arxiv_id=paper_id,
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                categories=[cat for cat in result.categories],
                published_date=result.published,
                pdf_url=result.pdf_url,
                doi=result.doi,
                status=ProcessingStatus.DOWNLOADED,
                source="arxiv"
            )

            # Download PDF
            paper_path = self._get_paper_path(paper)
            paper_path.parent.mkdir(parents=True, exist_ok=True)

            result.download_pdf(filename=str(paper_path))

            logger.info(f"Downloaded: {paper.title[:50]}...")

            return paper

        except Exception as e:
            logger.error(f"Error downloading paper {result.entry_id}: {e}")
            self.stats['errors'] += 1
            return None

    def _get_paper_path(self, paper: Paper) -> Path:
        """Get storage path for a paper"""
        # Organize by year/month/category
        year = paper.published_date.year if paper.published_date else "unknown"
        month = f"{paper.published_date.month:02d}" if paper.published_date else "unknown"
        category = paper.categories[0] if paper.categories else "unknown"

        return self.storage_path / str(year) / str(month) / category / f"{paper.arxiv_id}.pdf"

    def get_stats(self) -> Dict:
        """Get harvesting statistics"""
        return {
            **self.stats,
            'total_papers_in_db': self.doi_db.get_paper_count(),
            'storage_path': str(self.storage_path)
        }


def create_harvester(storage_path: Optional[str] = None, db_path: Optional[str] = None):
    """
    Factory function to create a harvester
    Allows CANDLE to override paths easily
    """
    return ArxivHarvester(
        storage_path=storage_path or "/mnt/hdd/c-light/raw_papers",
        db_path=db_path or "/mnt/nvme/c-light/doi_database"
    )

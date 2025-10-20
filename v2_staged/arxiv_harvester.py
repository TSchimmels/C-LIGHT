"""
ArXiv Harvesting Agent for CANDLE Cognitive Science RAG
Automated paper collection, deduplication, and integration for causal explanatory inference
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import aiohttp
import arxiv
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import schedule
from tqdm import tqdm
import ray

from ..core.cognitive_science_rag import (
    CognitiveScienceKnowledgeGraph, 
    CognitiveScienceRAG,
    KnowledgeDomain
)
from ..storage.storage_manager import StorageManager
from .doi_database import DOIPaperDatabase

logger = logging.getLogger(__name__)


class ArxivCategory(Enum):
    """Relevant arXiv categories for behavioral and cognitive science"""
    # Computer Science
    CS_AI = "cs.AI"  # Artificial Intelligence
    CS_HC = "cs.HC"  # Human-Computer Interaction (persuasive tech, dark patterns)
    CS_CY = "cs.CY"  # Computers and Society (social media, digital manipulation)
    CS_LG = "cs.LG"  # Machine Learning
    CS_NE = "cs.NE"  # Neural and Evolutionary Computing
    CS_CL = "cs.CL"  # Computation and Language
    CS_SI = "cs.SI"  # Social and Information Networks (social network analysis, information spread)
    
    # Quantitative Biology
    Q_BIO_NC = "q-bio.NC"  # Neurons and Cognition (includes organoid electrophysiology, firing thresholds)
    Q_BIO_QM = "q-bio.QM"  # Quantitative Methods
    Q_BIO_BM = "q-bio.BM"  # Biomolecules (ion channels, proteins, drug interactions, molecular mechanisms)
    Q_BIO_CB = "q-bio.CB"  # Cell Behavior (stem cells, organoid culture)
    Q_BIO_TO = "q-bio.TO"  # Tissues and Organs (brain organoids, tissue engineering)
    Q_BIO_MN = "q-bio.MN"  # Molecular Networks (metabolism, energy pathways, signaling cascades)
    Q_BIO_SC = "q-bio.SC"  # Subcellular Processes (cellular differentiation)
    Q_BIO_GN = "q-bio.GN"  # Genomics (epigenetics, gene expression, gene-environment interactions)
    
    # Physics
    PHYSICS_BIO = "physics.bio-ph"  # Biological Physics (EMF effects, biophysics, neuronal excitability)
    PHYSICS_MED = "physics.med-ph"  # Medical Physics
    PHYSICS_SOC = "physics.soc-ph"  # Physics and Society (social dynamics, cultural evolution, political polarization)
    PHYSICS_DATA = "physics.data-an"  # Data Analysis, Statistics and Probability
    QUANT_PH = "quant-ph"  # Quantum Physics (quantum biology, consciousness studies)

    # Economics
    ECON_GN = "econ.GN"  # General Economics (behavioral economics, decision-making, nudging)
    
    # Statistics
    STAT_AP = "stat.AP"  # Applications
    STAT_ML = "stat.ML"  # Machine Learning
    STAT_ME = "stat.ME"  # Methodology
    
    # Mathematics
    MATH_DS = "math.DS"  # Dynamical Systems
    MATH_ST = "math.ST"  # Statistics Theory
    
    # Other
    EESS_SP = "eess.SP"  # Signal Processing
    EESS_AS = "eess.AS"  # Audio and Speech Processing


@dataclass
class ArxivPaper:
    """Structured representation of an arXiv paper"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published: datetime
    updated: datetime
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    pdf_url: str = ""
    
    # Processing metadata
    content_hash: str = ""
    embedding: Optional[np.ndarray] = None
    extracted_concepts: List[str] = field(default_factory=list)
    cognitive_domains: List[KnowledgeDomain] = field(default_factory=list)
    behavioral_indicators: Dict[str, List[str]] = field(default_factory=dict)
    causal_relations: List[Dict[str, Any]] = field(default_factory=list)
    relevance_score: float = 0.0
    processed: bool = False
    processing_timestamp: Optional[datetime] = None


class DuplicateDetector:
    """Advanced duplicate detection using multiple strategies"""
    
    def __init__(self, 
                 storage_path: str = "/mnt/c/CANDLE/CANDLE_CORE/data/arxiv_dedup",
                 similarity_threshold: float = 0.92):
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        
        # Title and abstract encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Persistent storage
        self.arxiv_ids: Set[str] = set()
        self.title_hashes: Set[str] = set()
        self.doi_set: Set[str] = set()
        
        # Semantic index
        self.embedding_dim = 384
        self.semantic_index = None
        self.indexed_papers: List[str] = []
        
        # Load existing data
        self._load_state()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove latex commands
        text = re.sub(r'\\[a-zA-Z]+{[^}]*}', '', text)
        text = re.sub(r'\$[^$]+\$', '', text)
        
        # Normalize whitespace and case
        text = ' '.join(text.lower().split())
        return text
    
    def _compute_hash(self, text: str) -> str:
        """Compute hash of normalized text"""
        normalized = self._normalize_text(text)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def is_duplicate(self, paper: ArxivPaper) -> Tuple[bool, str]:
        """Check if paper is duplicate"""
        # Check arXiv ID
        if paper.arxiv_id in self.arxiv_ids:
            return True, f"Duplicate arXiv ID: {paper.arxiv_id}"
        
        # Check DOI
        if paper.doi and paper.doi in self.doi_set:
            return True, f"Duplicate DOI: {paper.doi}"
        
        # Check title similarity
        title_hash = self._compute_hash(paper.title)
        if title_hash in self.title_hashes:
            return True, "Duplicate title detected"
        
        # Semantic similarity check
        if self.semantic_index and self.semantic_index.ntotal > 0:
            embedding = self.encoder.encode(f"{paper.title} {paper.abstract}")
            distances, indices = self.semantic_index.search(
                np.array([embedding]).astype('float32'), k=3
            )
            
            # Check if too similar to existing papers
            if distances[0][0] > self.similarity_threshold:
                similar_id = self.indexed_papers[indices[0][0]]
                return True, f"High similarity ({distances[0][0]:.3f}) with {similar_id}"
        
        return False, ""
    
    def add_paper(self, paper: ArxivPaper):
        """Add paper to deduplication indices"""
        self.arxiv_ids.add(paper.arxiv_id)
        if paper.doi:
            self.doi_set.add(paper.doi)
        
        self.title_hashes.add(self._compute_hash(paper.title))
        
        # Add to semantic index
        if paper.embedding is None:
            paper.embedding = self.encoder.encode(f"{paper.title} {paper.abstract}")
        
        if self.semantic_index is None:
            self.semantic_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize for cosine similarity
        embedding_norm = paper.embedding / np.linalg.norm(paper.embedding)
        self.semantic_index.add(np.array([embedding_norm]).astype('float32'))
        self.indexed_papers.append(paper.arxiv_id)
        
        # Save state periodically
        if len(self.indexed_papers) % 100 == 0:
            self._save_state()
    
    def _save_state(self):
        """Save deduplication state to disk"""
        state = {
            'arxiv_ids': list(self.arxiv_ids),
            'doi_set': list(self.doi_set),
            'title_hashes': list(self.title_hashes),
            'indexed_papers': self.indexed_papers
        }
        
        with open(self.storage_path / 'dedup_state.json', 'w') as f:
            json.dump(state, f)
        
        if self.semantic_index:
            faiss.write_index(self.semantic_index, str(self.storage_path / 'semantic.index'))
    
    def _load_state(self):
        """Load deduplication state from disk"""
        state_file = self.storage_path / 'dedup_state.json'
        index_file = self.storage_path / 'semantic.index'
        
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.arxiv_ids = set(state['arxiv_ids'])
            self.doi_set = set(state['doi_set'])
            self.title_hashes = set(state['title_hashes'])
            self.indexed_papers = state['indexed_papers']
            
            logger.info(f"Loaded {len(self.arxiv_ids)} existing papers for deduplication")
        
        if index_file.exists():
            self.semantic_index = faiss.read_index(str(index_file))


class CausalRelationExtractor:
    """Extract causal relations from paper abstracts"""
    
    def __init__(self):
        self.causal_patterns = [
            # Direct causation
            (r'(\w+)\s+causes?\s+(\w+)', 'causes'),
            (r'(\w+)\s+leads?\s+to\s+(\w+)', 'leads_to'),
            (r'(\w+)\s+results?\s+in\s+(\w+)', 'results_in'),
            (r'(\w+)\s+induces?\s+(\w+)', 'induces'),
            
            # Effects and influences
            (r'(\w+)\s+affects?\s+(\w+)', 'affects'),
            (r'(\w+)\s+influences?\s+(\w+)', 'influences'),
            (r'(\w+)\s+modulates?\s+(\w+)', 'modulates'),
            (r'(\w+)\s+regulates?\s+(\w+)', 'regulates'),
            
            # Correlations
            (r'(\w+)\s+(?:is\s+)?associated\s+with\s+(\w+)', 'associated_with'),
            (r'(\w+)\s+correlates?\s+with\s+(\w+)', 'correlates_with'),
            
            # Predictions
            (r'(\w+)\s+predicts?\s+(\w+)', 'predicts'),
            (r'(\w+)\s+indicates?\s+(\w+)', 'indicates'),
        ]
    
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """Extract causal relations from text"""
        relations = []
        
        # Normalize text
        text = text.lower()
        sentences = text.split('.')
        
        for sentence in sentences:
            for pattern, relation_type in self.causal_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    source = match.group(1)
                    target = match.group(2)
                    
                    # Filter out common words
                    if len(source) > 2 and len(target) > 2:
                        relations.append({
                            'source': source,
                            'target': target,
                            'relation': relation_type,
                            'context': sentence.strip()
                        })
        
        return relations


class DomainClassifier:
    """Classify papers into CANDLE cognitive science domains"""
    
    def __init__(self):
        # Enhanced domain keywords based on CANDLE's focus areas
        self.domain_keywords = {
            KnowledgeDomain.NEUROSCIENCE: {
                'primary': ['brain', 'neural', 'neuron', 'cortex', 'neurotransmitter'],
                'secondary': ['synapse', 'plasticity', 'hippocampus', 'amygdala', 'prefrontal']
            },
            KnowledgeDomain.PSYCHOLOGY: {
                'primary': ['behavior', 'cognitive', 'emotion', 'personality', 'social'],
                'secondary': ['attention', 'memory', 'perception', 'learning', 'motivation']
            },
            KnowledgeDomain.SLEEP_SCIENCE: {
                'primary': ['sleep', 'circadian', 'melatonin', 'rem', 'insomnia'],
                'secondary': ['chronotype', 'dreams', 'rest', 'fatigue', 'alertness']
            },
            KnowledgeDomain.NUTRITION: {
                'primary': ['diet', 'nutrition', 'nutrient', 'food', 'eating'],
                'secondary': ['vitamin', 'mineral', 'metabolism', 'glucose', 'omega-3']
            },
            KnowledgeDomain.SOCIOLOGY: {
                'primary': ['social', 'society', 'culture', 'group', 'community'],
                'secondary': ['interaction', 'network', 'collective', 'norm', 'identity']
            },
            KnowledgeDomain.PHARMACOLOGY: {
                'primary': ['drug', 'medication', 'pharmaceutical', 'dose', 'treatment'],
                'secondary': ['receptor', 'agonist', 'antagonist', 'therapy', 'side effect']
            },
            KnowledgeDomain.EXERCISE_PHYSIOLOGY: {
                'primary': ['exercise', 'physical activity', 'fitness', 'training', 'sport'],
                'secondary': ['aerobic', 'strength', 'endurance', 'movement', 'performance']
            },
            KnowledgeDomain.MICROBIOME: {
                'primary': ['microbiome', 'bacteria', 'gut-brain', 'microbiota', 'probiotic'],
                'secondary': ['flora', 'microbial', 'prebiotic', 'dysbiosis', 'fermentation']
            }
        }
    
    def classify(self, paper: ArxivPaper) -> List[KnowledgeDomain]:
        """Classify paper into cognitive domains"""
        text = f"{paper.title} {paper.abstract}".lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            
            # Primary keywords worth more
            for keyword in keywords['primary']:
                score += text.count(keyword) * 2
            
            # Secondary keywords
            for keyword in keywords['secondary']:
                score += text.count(keyword)
            
            if score > 0:
                domain_scores[domain] = score
        
        # Return domains with scores above threshold
        threshold = 3
        domains = [domain for domain, score in domain_scores.items() if score >= threshold]
        
        # Default to psychology if no clear match
        if not domains and any(cat.startswith('q-bio') for cat in paper.categories):
            domains = [KnowledgeDomain.PSYCHOLOGY]
        
        return domains


class BehavioralIndicatorExtractor:
    """Extract behavioral indicators from papers"""
    
    def __init__(self):
        self.indicator_patterns = {
            'risk_factors': [
                'risk factor', 'vulnerability', 'predisposition', 'susceptibility',
                'increases risk', 'associated with increased', 'predictor of'
            ],
            'protective_factors': [
                'protective factor', 'resilience', 'buffer', 'mitigates',
                'reduces risk', 'associated with decreased', 'prevents'
            ],
            'behavioral_markers': [
                'behavioral marker', 'indicator', 'sign', 'symptom',
                'characteristic', 'feature', 'pattern'
            ],
            'interventions': [
                'intervention', 'treatment', 'therapy', 'approach',
                'strategy', 'technique', 'method'
            ]
        }
    
    def extract_indicators(self, text: str) -> Dict[str, List[str]]:
        """Extract behavioral indicators from text"""
        indicators = defaultdict(list)
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            for indicator_type, patterns in self.indicator_patterns.items():
                for pattern in patterns:
                    if pattern in sentence_lower:
                        # Extract the relevant part of the sentence
                        indicators[indicator_type].append(sentence.strip())
                        break
        
        return dict(indicators)


@ray.remote
class ArxivProcessor:
    """Ray actor for distributed paper processing"""
    
    def __init__(self):
        self.domain_classifier = DomainClassifier()
        self.causal_extractor = CausalRelationExtractor()
        self.indicator_extractor = BehavioralIndicatorExtractor()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def process_paper(self, paper: ArxivPaper) -> ArxivPaper:
        """Process a single paper"""
        # Extract cognitive domains
        paper.cognitive_domains = self.domain_classifier.classify(paper)
        
        # Extract causal relations
        paper.causal_relations = self.causal_extractor.extract_relations(
            f"{paper.title} {paper.abstract}"
        )
        
        # Extract behavioral indicators
        paper.behavioral_indicators = self.indicator_extractor.extract_indicators(
            paper.abstract
        )
        
        # Generate embedding if not present
        if paper.embedding is None:
            paper.embedding = self.encoder.encode(f"{paper.title} {paper.abstract}")
        
        # Extract key concepts (simple keyword extraction)
        words = re.findall(r'\b[a-z]+\b', paper.abstract.lower())
        word_freq = defaultdict(int)
        
        stopwords = {'the', 'and', 'of', 'in', 'to', 'for', 'with', 'by', 'from'}
        for word in words:
            if len(word) > 4 and word not in stopwords:
                word_freq[word] += 1
        
        paper.extracted_concepts = [
            word for word, _ in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        ]
        
        paper.processed = True
        paper.processing_timestamp = datetime.now()
        
        return paper


class ArxivHarvester:
    """Main arXiv harvesting agent with DOI database integration"""
    
    def __init__(self,
                 config_path: str = "/mnt/c/CANDLE/CANDLE_CORE/configs/arxiv_config.json",
                 use_ray: bool = True,
                 doi_database: Optional[DOIPaperDatabase] = None):
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.doi_db = doi_database or DOIPaperDatabase()
        self.domain_classifier = DomainClassifier()
        
        # Storage paths
        self.data_dir = Path("/mnt/c/CANDLE/CANDLE_CORE/data/arxiv_papers")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Ray for distributed processing
        self.use_ray = use_ray and ray.is_initialized()
        if self.use_ray:
            self.processors = [ArxivProcessor.remote() for _ in range(4)]
        
        # Statistics
        self.stats = {
            'total_fetched': 0,
            'duplicates_found': 0,
            'papers_processed': 0,
            'papers_added_to_rag': 0,
            'papers_registered_in_doi_db': 0,
            'last_harvest': None
        }
        
        self._load_stats()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load or create configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        default_config = {
            "search_queries": [
                "behavioral analysis deep learning",
                "cognitive performance prediction",
                "human behavior modeling",
                "social contagion networks",
                "multimodal behavioral sensing",
                "psychological state estimation",
                "group dynamics modeling",
                "cognitive load assessment",
                "emotion recognition multimodal",
                "behavioral intervention AI"
            ],
            "categories": [
                "cs.AI", "cs.HC", "cs.LG", "cs.CY",
                "q-bio.NC", "q-bio.QM",
                "physics.soc-ph",
                "stat.ML", "stat.AP"
            ],
            "max_results_per_query": 50,
            "days_back": 30,
            "relevance_threshold": 0.6,
            "harvest_schedule": "daily"
        }
        
        # Save default config
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    async def search_papers(self, 
                          query: str = None,
                          category: str = None,
                          max_results: int = None,
                          sort_by: str = "lastUpdatedDate",
                          sort_order: str = "descending") -> List[Dict]:
        """Search arXiv and return raw paper data"""
        papers = []
        
        try:
            # Build query
            if category and not query:
                query = f"cat:{category}"
            elif category and query:
                query = f"{query} AND cat:{category}"
            
            if not max_results:
                max_results = self.config['max_results_per_query']
            
            # Map sort options to arxiv API
            sort_map = {
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
                "relevance": arxiv.SortCriterion.Relevance
            }
            sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.LastUpdatedDate)
            
            sort_order_map = {
                "descending": arxiv.SortOrder.Descending,
                "ascending": arxiv.SortOrder.Ascending
            }
            sort_order_obj = sort_order_map.get(sort_order, arxiv.SortOrder.Descending)
            
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion,
                sort_order=sort_order_obj
            )
            
            cutoff_date = datetime.now() - timedelta(days=self.config['days_back'])
            
            for result in search.results():
                # Skip old papers
                if result.published.replace(tzinfo=None) < cutoff_date:
                    continue
                
                paper_data = {
                    'id': result.entry_id.split('/')[-1],
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'authors': [{'name': author.name} for author in result.authors],
                    'summary': result.summary,
                    'abstract': result.summary,
                    'categories': result.categories,
                    'published': result.published.replace(tzinfo=None),
                    'updated': result.updated.replace(tzinfo=None),
                    'doi': getattr(result, 'doi', None),
                    'journal_ref': getattr(result, 'journal_ref', None),
                    'pdf_url': result.pdf_url
                }
                
                papers.append(paper_data)
                
        except Exception as e:
            logger.error(f"Error searching arXiv for '{query}': {e}")
        
        return papers
    
    async def download_pdf(self, pdf_url: str, output_path: str) -> bool:
        """Download PDF from arXiv"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        with open(output_path, 'wb') as f:
                            f.write(content)
                        return True
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
        return False
    
    async def harvest_papers(self) -> List[ArxivPaper]:
        """Main harvesting method"""
        logger.info("Starting arXiv paper harvest...")
        start_time = time.time()
        
        all_papers = []
        
        # Search by queries
        for query in self.config['search_queries']:
            papers = await self._search_arxiv(query)
            all_papers.extend(papers)
            logger.info(f"Found {len(papers)} papers for query: {query}")
        
        # Search by categories
        for category in self.config['categories']:
            papers = await self._search_category(category)
            all_papers.extend(papers)
            logger.info(f"Found {len(papers)} papers in category: {category}")
        
        self.stats['total_fetched'] = len(all_papers)
        
        # Deduplicate
        unique_papers = self._deduplicate_papers(all_papers)
        logger.info(f"After deduplication: {len(unique_papers)} unique papers")
        
        # Process papers
        if self.use_ray:
            processed_papers = await self._process_papers_distributed(unique_papers)
        else:
            processed_papers = await self._process_papers_sequential(unique_papers)
        
        # Save results
        self._save_papers(processed_papers)
        self._save_stats()
        
        duration = time.time() - start_time
        logger.info(f"Harvest completed in {duration:.2f}s. Processed {len(processed_papers)} papers.")
        
        return processed_papers
    
    async def _search_arxiv(self, query: str) -> List[ArxivPaper]:
        """Search arXiv by query"""
        papers = []
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=self.config['max_results_per_query'],
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            cutoff_date = datetime.now() - timedelta(days=self.config['days_back'])
            
            for result in search.results():
                # Skip old papers
                if result.published.replace(tzinfo=None) < cutoff_date:
                    continue
                
                paper = ArxivPaper(
                    arxiv_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    categories=result.categories,
                    published=result.published.replace(tzinfo=None),
                    updated=result.updated.replace(tzinfo=None),
                    doi=getattr(result, 'doi', None),
                    journal_ref=getattr(result, 'journal_ref', None),
                    pdf_url=result.pdf_url
                )
                
                papers.append(paper)
                
        except Exception as e:
            logger.error(f"Error searching arXiv for '{query}': {e}")
        
        return papers
    
    async def _search_category(self, category: str) -> List[ArxivPaper]:
        """Search arXiv by category"""
        query = f"cat:{category}"
        return await self._search_arxiv(query)
    
    def _deduplicate_papers(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """Remove duplicate papers using DOI database"""
        unique_papers = []
        
        for paper in papers:
            # Check if paper exists in DOI database
            if self.doi_db.paper_exists(
                doi=paper.doi,
                arxiv_id=paper.arxiv_id
            ):
                self.stats['duplicates_found'] += 1
                logger.debug(f"Skipping duplicate: {paper.arxiv_id}")
                continue
            
            # Add to DOI database immediately
            paper_data = {
                'doi': paper.doi or f"arxiv:{paper.arxiv_id}",
                'arxiv_id': paper.arxiv_id,
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'categories': paper.categories,
                'pdf_url': paper.pdf_url,
                'published_date': paper.published,
                'last_updated': paper.updated,
                'download_date': datetime.now(),
                'source': 'arxiv_harvester'
            }
            
            success, doi = self.doi_db.add_paper(paper_data)
            if success:
                paper.doi = doi  # Update paper with assigned DOI
                unique_papers.append(paper)
                self.stats['papers_registered_in_doi_db'] += 1
            else:
                logger.warning(f"Failed to register paper in DOI database: {paper.title}")
        
        return unique_papers
    
    async def _process_papers_distributed(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """Process papers using Ray for distribution"""
        logger.info(f"Processing {len(papers)} papers using Ray...")
        
        # Batch papers for processing
        batch_size = max(1, len(papers) // len(self.processors))
        futures = []
        
        for i, processor in enumerate(self.processors):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < len(self.processors) - 1 else len(papers)
            batch = papers[start_idx:end_idx]
            
            # Process batch
            batch_futures = [processor.process_paper.remote(paper) for paper in batch]
            futures.extend(batch_futures)
        
        # Collect results
        processed_papers = ray.get(futures)
        
        # Filter by relevance
        relevant_papers = [p for p in processed_papers if self._is_relevant(p)]
        self.stats['papers_processed'] = len(relevant_papers)
        
        return relevant_papers
    
    async def _process_papers_sequential(self, papers: List[ArxivPaper]) -> List[ArxivPaper]:
        """Process papers sequentially"""
        logger.info(f"Processing {len(papers)} papers sequentially...")
        
        processor = ArxivProcessor()
        processed_papers = []
        
        for paper in tqdm(papers, desc="Processing papers"):
            try:
                processed = processor.process_paper(paper)
                if self._is_relevant(processed):
                    processed_papers.append(processed)
            except Exception as e:
                logger.error(f"Error processing paper {paper.arxiv_id}: {e}")
        
        self.stats['papers_processed'] = len(processed_papers)
        return processed_papers
    
    def _is_relevant(self, paper: ArxivPaper) -> bool:
        """Check if paper is relevant to CANDLE's focus"""
        # Must have at least one cognitive domain
        if not paper.cognitive_domains:
            return False
        
        # Must have behavioral indicators or causal relations
        if not paper.behavioral_indicators and not paper.causal_relations:
            return False
        
        # Check for behavioral/cognitive keywords
        behavioral_keywords = [
            'behavior', 'cognitive', 'prediction', 'human', 'social',
            'emotion', 'mental', 'psychological', 'group', 'interaction'
        ]
        
        text = f"{paper.title} {paper.abstract}".lower()
        keyword_count = sum(1 for kw in behavioral_keywords if kw in text)
        
        return keyword_count >= 2
    
    def _save_papers(self, papers: List[ArxivPaper]):
        """Save processed papers to disk"""
        output_file = self.data_dir / f"harvest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(papers, f)
        
        logger.info(f"Saved {len(papers)} papers to {output_file}")
        
        # Also save as JSON for inspection
        json_file = output_file.with_suffix('.json')
        papers_dict = []
        
        for paper in papers:
            paper_dict = {
                'arxiv_id': paper.arxiv_id,
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract[:500] + '...',
                'categories': paper.categories,
                'published': paper.published.isoformat(),
                'cognitive_domains': [d.value for d in paper.cognitive_domains],
                'behavioral_indicators': paper.behavioral_indicators,
                'causal_relations': paper.causal_relations[:5],  # First 5
                'concepts': paper.extracted_concepts[:10]
            }
            papers_dict.append(paper_dict)
        
        with open(json_file, 'w') as f:
            json.dump(papers_dict, f, indent=2)
    
    def _load_stats(self):
        """Load harvesting statistics"""
        stats_file = self.data_dir / 'harvest_stats.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                saved_stats = json.load(f)
                self.stats.update(saved_stats)
    
    def _save_stats(self):
        """Save harvesting statistics"""
        self.stats['last_harvest'] = datetime.now().isoformat()
        
        with open(self.data_dir / 'harvest_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    async def integrate_with_rag(self, 
                                cognitive_rag: CognitiveScienceRAG,
                                papers: List[ArxivPaper]) -> int:
        """Integrate harvested papers with CANDLE's RAG system"""
        logger.info(f"Integrating {len(papers)} papers into Cognitive Science RAG...")
        
        added_count = 0
        
        for paper in tqdm(papers, desc="Adding to RAG"):
            try:
                # Add to each relevant domain
                for domain in paper.cognitive_domains:
                    metadata = {
                        'arxiv_id': paper.arxiv_id,
                        'authors': paper.authors,
                        'publication_date': paper.published.isoformat(),
                        'categories': paper.categories,
                        'concepts': paper.extracted_concepts,
                        'behavioral_indicators': paper.behavioral_indicators,
                        'causal_relations': paper.causal_relations,
                        'evidence_level': self._determine_evidence_level(paper)
                    }
                    
                    # Combine abstract with extracted information
                    content = f"{paper.abstract}\n\n"
                    
                    if paper.behavioral_indicators:
                        content += "Behavioral Indicators:\n"
                        for indicator_type, indicators in paper.behavioral_indicators.items():
                            content += f"- {indicator_type}: {'; '.join(indicators[:3])}\n"
                    
                    if paper.causal_relations:
                        content += "\nCausal Relations:\n"
                        for relation in paper.causal_relations[:5]:
                            content += f"- {relation['source']} {relation['relation']} {relation['target']}\n"
                    
                    cognitive_rag.add_scientific_document(
                        domain=domain,
                        title=paper.title,
                        content=content,
                        metadata=metadata
                    )
                    
                added_count += 1
                
            except Exception as e:
                logger.error(f"Error adding paper {paper.arxiv_id} to RAG: {e}")
        
        self.stats['papers_added_to_rag'] += added_count
        logger.info(f"Successfully added {added_count} papers to RAG system")
        
        return added_count
    
    def _determine_evidence_level(self, paper: ArxivPaper) -> str:
        """Determine evidence level based on paper characteristics"""
        # Papers with journal references are likely peer-reviewed
        if paper.journal_ref:
            return 'strong'
        
        # Papers with many authors and citations to behavioral indicators
        if len(paper.authors) >= 3 and len(paper.behavioral_indicators) > 0:
            return 'moderate'
        
        # Papers with causal relations extracted
        if len(paper.causal_relations) >= 3:
            return 'moderate'
        
        return 'preliminary'


class ArxivHarvestingAgent:
    """Autonomous agent for continuous arXiv harvesting with DOI tracking"""
    
    def __init__(self,
                 cognitive_rag: Optional[CognitiveScienceRAG] = None,
                 doi_database: Optional[DOIPaperDatabase] = None,
                 use_ray: bool = True):
        
        # Initialize Ray if requested
        if use_ray and not ray.is_initialized():
            ray.init(num_cpus=4, dashboard_host='0.0.0.0')
            logger.info("Ray initialized for distributed processing")
        
        self.cognitive_rag = cognitive_rag
        self.doi_db = doi_database or DOIPaperDatabase()
        self.harvester = ArxivHarvester(use_ray=use_ray, doi_database=self.doi_db)
        
        # Monitoring
        self.running = False
    
    async def run_harvest_cycle(self):
        """Run a single harvest cycle"""
        logger.info("=" * 60)
        logger.info("Starting ArXiv Harvest Cycle")
        logger.info("=" * 60)
        
        try:
            # Harvest papers
            papers = await self.harvester.harvest_papers()
            
            # Integrate with RAG if available
            if self.cognitive_rag and papers:
                await self.harvester.integrate_with_rag(self.cognitive_rag, papers)
            
            # Log summary
            logger.info("\nHarvest Summary:")
            logger.info(f"- Total papers fetched: {self.harvester.stats['total_fetched']}")
            logger.info(f"- Duplicates found: {self.harvester.stats['duplicates_found']}")
            logger.info(f"- Papers processed: {self.harvester.stats['papers_processed']}")
            logger.info(f"- Papers added to RAG: {self.harvester.stats['papers_added_to_rag']}")
            logger.info(f"- Papers registered in DOI DB: {self.harvester.stats['papers_registered_in_doi_db']}")
            
        except Exception as e:
            logger.error(f"Harvest cycle failed: {e}", exc_info=True)
    
    def schedule_harvesting(self):
        """Schedule periodic harvesting"""
        config = self.harvester.config
        
        if config['harvest_schedule'] == 'daily':
            schedule.every().day.at("03:00").do(
                lambda: asyncio.run(self.run_harvest_cycle())
            )
            logger.info("Scheduled daily harvesting at 3:00 AM")
        
        elif config['harvest_schedule'] == 'weekly':
            schedule.every().monday.at("03:00").do(
                lambda: asyncio.run(self.run_harvest_cycle())
            )
            logger.info("Scheduled weekly harvesting on Mondays at 3:00 AM")
    
    def run(self):
        """Run the agent continuously"""
        logger.info("Starting ArXiv Harvesting Agent...")
        self.running = True
        
        # Run initial harvest
        asyncio.run(self.run_harvest_cycle())
        
        # Schedule future harvests
        self.schedule_harvesting()
        
        # Keep running
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop the agent"""
        self.running = False
        if ray.is_initialized():
            ray.shutdown()


# Integration with CANDLE
def create_arxiv_agent(cognitive_rag: Optional[CognitiveScienceRAG] = None,
                      doi_database: Optional[DOIPaperDatabase] = None) -> ArxivHarvestingAgent:
    """Factory function to create configured ArXiv agent with DOI tracking"""
    return ArxivHarvestingAgent(
        cognitive_rag=cognitive_rag,
        doi_database=doi_database,
        use_ray=True
    )


if __name__ == "__main__":
    # For testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run agent
    agent = create_arxiv_agent()
    
    # Run single harvest cycle for testing
    asyncio.run(agent.run_harvest_cycle())
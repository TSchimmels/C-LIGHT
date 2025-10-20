"""
LLM-Based RAG using Mixtral-8x7B-Instruct
Traditional RAG: Embed → Retrieve → Generate
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ..core.base_types import Paper, KnowledgeDomain, QueryResult, Evidence
from ..core.paper_processor import PaperProcessor, PaperChunk
from .embedding import EmbeddingManager
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM-based RAG"""
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    embedding_model: str = "BAAI/bge-large-en-v1.5"  # Best quality embeddings
    device: str = "cuda"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95

    # Retrieval settings
    top_k_papers: int = 10
    min_relevance_score: float = 0.5

    # Memory optimization (with your GPU, can use full precision)
    load_in_8bit: bool = False  # Set True if you want to save VRAM
    load_in_4bit: bool = False
    use_flash_attention: bool = True


class MixtralRAG:
    """
    High-quality RAG system using Mixtral-8x7B-Instruct

    This is System A - the traditional LLM approach
    Static weights, no learning, but excellent quality from day 1
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

        logger.info(f"Initializing Mixtral RAG System A")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Embedding: {self.config.embedding_model}")
        logger.info(f"  Device: {self.config.device}")

        # Load embedding model
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.embedding_model,
            device=self.config.device
        )

        # Initialize vector store
        self.vector_store = VectorStore(
            dimension=self.embedding_manager.dimension
        )

        # Initialize paper processor
        self.paper_processor = PaperProcessor(
            chunk_size=1000,
            chunk_overlap=200,
            extract_sections=True
        )

        # Storage for chunks (for citation lookup)
        self.chunks_by_id = {}

        # Load Mixtral
        self._load_mixtral()

        # Stats
        self.query_count = 0
        self.total_latency = 0.0

        logger.info("✓ Mixtral RAG System A initialized")

    def _load_mixtral(self):
        """Load Mixtral-8x7B-Instruct"""
        logger.info("Loading Mixtral-8x7B-Instruct (this may take a few minutes)...")

        # Quantization config (optional - you have enough VRAM for full precision)
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Load model
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        self.model.eval()  # Set to evaluation mode

        logger.info("✓ Mixtral-8x7B loaded successfully")

    def add_paper(self, paper: Paper, pdf_path: Optional[str] = None):
        """
        Add a paper to the RAG system
        Processes full PDF, creates chunks, embeds each chunk with source tracking

        Args:
            paper: Paper object
            pdf_path: Optional path to PDF file for full text extraction
        """
        from pathlib import Path

        # Process paper into chunks with section tracking
        pdf_path_obj = Path(pdf_path) if pdf_path else None
        chunks = self.paper_processor.process_paper(paper, pdf_path_obj)

        logger.info(f"Processing {paper.paper_id}: {len(chunks)} chunks")

        # Embed each chunk
        for chunk in chunks:
            # Generate embedding for this chunk
            embedding = self.embedding_manager.embed_text(chunk.text)

            # Store in vector DB with full citation metadata
            self.vector_store.add(
                paper_id=chunk.chunk_id,  # Use chunk_id as unique identifier
                embedding=embedding,
                metadata={
                    'paper_id': paper.paper_id,
                    'chunk_id': chunk.chunk_id,
                    'title': paper.title,
                    'authors': paper.authors,
                    'section': chunk.section,
                    'page_start': chunk.page_start,
                    'page_end': chunk.page_end,
                    'text': chunk.text[:1000],  # Store first 1000 chars for preview
                    'full_citation': chunk.get_citation(),
                    'categories': paper.categories,
                    'domains': [d.value for d in paper.domains],
                    'published': paper.published_date.isoformat() if paper.published_date else None
                }
            )

            # Store chunk for later retrieval
            self.chunks_by_id[chunk.chunk_id] = chunk

        logger.info(f"✓ Added {paper.paper_id} to System A ({len(chunks)} chunks)")

    def add_papers_batch(self, papers: List[Paper], batch_size: int = 32):
        """
        Add multiple papers efficiently
        """
        logger.info(f"Adding {len(papers)} papers to System A...")

        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]

            # Batch embed
            embeddings = self.embedding_manager.embed_papers_batch(batch)

            # Add to vector store
            for paper, embedding in zip(batch, embeddings):
                self.vector_store.add(
                    paper_id=paper.paper_id,
                    embedding=embedding,
                    metadata={
                        'title': paper.title,
                        'abstract': paper.abstract[:500],
                    }
                )

            if (i + batch_size) % 100 == 0:
                logger.info(f"  Processed {min(i + batch_size, len(papers))}/{len(papers)} papers")

        logger.info(f"✓ Added {len(papers)} papers to System A")

    def query(
        self,
        question: str,
        domains: Optional[List[KnowledgeDomain]] = None,
        max_papers: int = None
    ) -> QueryResult:
        """
        Query the RAG system

        Pipeline:
        1. Embed query
        2. Retrieve relevant papers
        3. Build context
        4. Generate answer with Mixtral

        Args:
            question: Natural language question
            domains: Optional domain filter
            max_papers: Max papers to retrieve (default from config)

        Returns:
            QueryResult with answer and evidence
        """
        start_time = time.time()

        max_papers = max_papers or self.config.top_k_papers

        logger.info(f"System A query: {question[:60]}...")

        # 1. Embed query
        query_embedding = self.embedding_manager.embed_text(question)

        # 2. Retrieve relevant papers
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=max_papers,
            min_score=self.config.min_relevance_score
        )

        if not results:
            return QueryResult(
                query=question,
                answer="I could not find any relevant papers to answer this question.",
                confidence=0.0,
                sources_count=0,
                metadata={'system': 'mixtral_rag', 'retrieval_failed': True}
            )

        # 3. Build context from papers
        context = self._build_context(results, question)

        # 4. Generate answer with Mixtral
        answer = self._generate_answer(question, context)

        # 5. Build evidence list
        evidence = []
        for result in results:
            evidence.append(Evidence(
                paper_id=result['paper_id'],
                paper_title=result['metadata'].get('title', 'Unknown'),
                text=result['metadata'].get('abstract', ''),
                confidence=1.0,  # Mixtral doesn't provide per-evidence confidence
                relevance_score=result['score']
            ))

        latency = time.time() - start_time
        self.query_count += 1
        self.total_latency += latency

        logger.info(f"✓ System A answered in {latency:.2f}s")

        return QueryResult(
            query=question,
            answer=answer,
            evidence=evidence,
            confidence=0.0,  # Mixtral doesn't provide calibrated confidence
            sources_count=len(results),
            metadata={
                'system': 'mixtral_rag',
                'model': self.config.model_name,
                'latency': latency,
                'papers_retrieved': len(results)
            }
        )

    def _build_context(self, results: List[Dict], question: str) -> str:
        """
        Build context string from retrieved paper chunks
        Each chunk has full citation information
        """
        context_parts = []

        for i, result in enumerate(results, 1):
            metadata = result['metadata']

            # Get full chunk text if available
            chunk_id = metadata.get('chunk_id')
            chunk = self.chunks_by_id.get(chunk_id)
            chunk_text = chunk.text if chunk else metadata.get('text', 'No text available')

            # Build detailed citation
            citation = metadata.get('full_citation', 'Unknown source')
            section = metadata.get('section', 'unknown section')
            pages = ""
            if metadata.get('page_start'):
                if metadata.get('page_end') and metadata['page_end'] != metadata['page_start']:
                    pages = f" (pp. {metadata['page_start']}-{metadata['page_end']})"
                else:
                    pages = f" (p. {metadata['page_start']})"

            context_parts.append(f"""
[Paper {i}]
Citation: {citation}
Section: {section.title()}{pages}
Text:
{chunk_text}

Relevance: {result['score']:.3f}
""")

        return "\n" + "="*80 + "\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using Mixtral-8x7B with explicit citation requirements
        """
        # Build prompt for Mixtral with strict citation requirements
        prompt = f"""<s>[INST] You are an expert in cognitive and behavioral science. Answer the question based ONLY on the provided scientific papers. You MUST cite papers explicitly.

Scientific Papers:
{context}

Question: {question}

CRITICAL INSTRUCTIONS:
1. ALWAYS cite specific papers when making claims (e.g., "According to Paper 1 (Smith et al., Introduction section)...")
2. Quote relevant text from papers when appropriate
3. Reference the section and page numbers provided
4. If papers contradict each other, present both views with citations
5. If the papers don't fully answer the question, explicitly state this
6. DO NOT make claims without citing a specific paper
7. Format citations as: [Paper X, Section, Pages]

Example good answer:
"Dopamine affects motivation through reward pathway activation [Paper 1, Results, pp. 5-6]. Smith et al. found that 'dopamine release in the nucleus accumbens correlates with motivational state' [Paper 1, Discussion, p. 8]. However, Johnson et al. reported conflicting results in their study [Paper 3, Results, p. 12]..."

Answer with explicit citations: [/INST]"""

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # Leave room for generation
        ).to(self.config.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        answer = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return answer.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'system': 'mixtral_rag',
            'model': self.config.model_name,
            'embedding_model': self.config.embedding_model,
            'papers_indexed': self.vector_store.count(),
            'queries_processed': self.query_count,
            'avg_latency': self.total_latency / max(self.query_count, 1),
            'device': self.config.device
        }

    def save_index(self, path: str):
        """Save vector index to disk"""
        self.vector_store.save(path)
        logger.info(f"✓ Saved System A index to {path}")

    def load_index(self, path: str):
        """Load vector index from disk"""
        self.vector_store.load(path)
        logger.info(f"✓ Loaded System A index from {path}")


# Alias for clarity
LLMBasedRAG = MixtralRAG

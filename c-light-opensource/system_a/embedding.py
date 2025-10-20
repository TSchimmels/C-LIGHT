"""
Embedding manager for System A
Uses BGE-large for highest quality embeddings
"""

import logging
from typing import List
import numpy as np

import torch
from sentence_transformers import SentenceTransformer

from ..core.base_types import Paper

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embeddings for papers and queries
    Uses BGE-large-en-v1.5 - best quality open-source embeddings
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading embedding model: {model_name}")

        # Load sentence transformer
        self.model = SentenceTransformer(model_name, device=device)

        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"✓ Embedding model loaded (dimension: {self.dimension})")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string

        Args:
            text: Text to embed

        Returns:
            Embedding vector (numpy array)
        """
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
                show_progress_bar=False
            )

        return embedding

    def embed_texts_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts efficiently

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding

        Returns:
            Array of embeddings (shape: [len(texts), dimension])
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100
            )

        return embeddings

    def embed_paper(self, paper: Paper) -> np.ndarray:
        """
        Embed a paper's content

        Strategy: Combine title and abstract with proper weighting
        Title gets 2x weight as it's most informative

        Args:
            paper: Paper object

        Returns:
            Embedding vector
        """
        # Combine title (2x) and abstract
        text = f"{paper.title} {paper.title} {paper.abstract}"

        return self.embed_text(text)

    def embed_papers_batch(
        self,
        papers: List[Paper],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed multiple papers efficiently

        Args:
            papers: List of papers to embed
            batch_size: Batch size for encoding

        Returns:
            Array of embeddings
        """
        # Prepare texts
        texts = [
            f"{paper.title} {paper.title} {paper.abstract}"
            for paper in papers
        ]

        return self.embed_texts_batch(texts, batch_size)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        # Since embeddings are normalized, dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))

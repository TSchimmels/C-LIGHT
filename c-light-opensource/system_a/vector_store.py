"""
Vector store for System A
Uses FAISS for efficient similarity search
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

import faiss

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for paper embeddings
    Supports exact and approximate nearest neighbor search
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",  # "flat" for exact, "ivf" for approximate
        metric: str = "cosine"
    ):
        """
        Initialize vector store

        Args:
            dimension: Embedding dimension
            index_type: "flat" (exact) or "ivf" (approximate, faster for large scale)
            metric: Distance metric ("cosine" or "l2")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric

        # Create FAISS index
        if index_type == "flat":
            if metric == "cosine":
                # Inner product (after L2 normalization) = cosine similarity
                self.index = faiss.IndexFlatIP(dimension)
            else:
                self.index = faiss.IndexFlatL2(dimension)

        elif index_type == "ivf":
            # For large scale (>1M vectors)
            quantizer = faiss.IndexFlatIP(dimension) if metric == "cosine" else faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
            self.needs_training = True
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Metadata storage (paper IDs and metadata)
        self.id_to_paper_id: List[str] = []
        self.paper_id_to_idx: Dict[str, int] = {}
        self.metadata: List[Dict[str, Any]] = []

        logger.info(f"✓ Vector store initialized (dimension={dimension}, type={index_type}, metric={metric})")

    def add(
        self,
        paper_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a paper embedding to the index

        Args:
            paper_id: Unique paper identifier
            embedding: Embedding vector
            metadata: Optional metadata dict
        """
        # Check if already exists
        if paper_id in self.paper_id_to_idx:
            logger.debug(f"Paper {paper_id} already in index, skipping")
            return

        # Ensure embedding is 2D for FAISS
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Add to FAISS index
        idx = len(self.id_to_paper_id)
        self.index.add(embedding.astype('float32'))

        # Store metadata
        self.id_to_paper_id.append(paper_id)
        self.paper_id_to_idx[paper_id] = idx
        self.metadata.append(metadata or {})

    def add_batch(
        self,
        paper_ids: List[str],
        embeddings: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add multiple embeddings efficiently

        Args:
            paper_ids: List of paper IDs
            embeddings: Array of embeddings (shape: [N, dimension])
            metadata_list: Optional list of metadata dicts
        """
        if metadata_list is None:
            metadata_list = [{}] * len(paper_ids)

        # Filter out duplicates
        new_indices = []
        for i, paper_id in enumerate(paper_ids):
            if paper_id not in self.paper_id_to_idx:
                new_indices.append(i)

        if not new_indices:
            return

        # Add only new papers
        new_paper_ids = [paper_ids[i] for i in new_indices]
        new_embeddings = embeddings[new_indices]
        new_metadata = [metadata_list[i] for i in new_indices]

        # Add to FAISS
        self.index.add(new_embeddings.astype('float32'))

        # Update metadata
        start_idx = len(self.id_to_paper_id)
        for i, paper_id in enumerate(new_paper_ids):
            self.id_to_paper_id.append(paper_id)
            self.paper_id_to_idx[paper_id] = start_idx + i
            self.metadata.append(new_metadata[i])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar papers

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of dicts with keys: paper_id, score, metadata
        """
        if self.count() == 0:
            return []

        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Check if valid index (FAISS returns -1 for missing)
            if idx < 0 or idx >= len(self.id_to_paper_id):
                continue

            # Filter by minimum score
            if score < min_score:
                continue

            results.append({
                'paper_id': self.id_to_paper_id[idx],
                'score': float(score),
                'metadata': self.metadata[idx]
            })

        return results

    def get_by_paper_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific paper

        Args:
            paper_id: Paper identifier

        Returns:
            Metadata dict or None if not found
        """
        idx = self.paper_id_to_idx.get(paper_id)
        if idx is None:
            return None

        return {
            'paper_id': paper_id,
            'metadata': self.metadata[idx]
        }

    def count(self) -> int:
        """Get number of papers in index"""
        return len(self.id_to_paper_id)

    def save(self, path: str):
        """
        Save index and metadata to disk

        Args:
            path: Directory to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'id_to_paper_id': self.id_to_paper_id,
                'paper_id_to_idx': self.paper_id_to_idx,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'metric': self.metric
            }, f)

        logger.info(f"✓ Saved vector store to {path}")

    def load(self, path: str):
        """
        Load index and metadata from disk

        Args:
            path: Directory to load from
        """
        path = Path(path)

        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))

        # Load metadata
        with open(path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.id_to_paper_id = data['id_to_paper_id']
            self.paper_id_to_idx = data['paper_id_to_idx']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
            self.metric = data['metric']

        logger.info(f"✓ Loaded vector store from {path} ({self.count()} papers)")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'paper_count': self.count(),
            'index_size_mb': self.index.ntotal * self.dimension * 4 / (1024**2)  # Rough estimate
        }

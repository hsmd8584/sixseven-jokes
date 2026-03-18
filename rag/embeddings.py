"""
Joke Embedding Index

Manages the FAISS vector index for joke embeddings:
- Build index from joke dataset
- Lazy-load model (important for Cloud Run cold start optimization)
- Efficient batch encoding and search
- Persistence (save/load index to disk)
"""

import os
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

from config import config


class JokeEmbeddingIndex:
    """
    FAISS-backed vector index for joke content.

    Key design decisions:
    - Lazy model loading: SentenceTransformer is loaded on first use,
      not at import time. This is critical for Cloud Run where cold
      start time directly impacts user-facing latency.
    - Normalized embeddings: We use L2-normalized vectors with inner
      product search, which is equivalent to cosine similarity but
      faster with FAISS IndexFlatIP.
    - Metadata coupling: Each vector position maps 1:1 to a joke in
      the metadata list, enabling efficient retrieval without a
      separate metadata store.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        index_path: Optional[str] = None,
    ):
        self.model_name = model_name or config.embedding.model_name
        self.index_path = index_path or config.embedding.faiss_index_path
        self.dimension = config.embedding.dimension

        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._jokes: List[Dict] = []  # Metadata paired with index positions

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def index(self) -> faiss.Index:
        """Get the FAISS index, building or loading as needed."""
        if self._index is None:
            if os.path.exists(os.path.join(self.index_path, "index.faiss")):
                self.load()
            else:
                raise RuntimeError(
                    "No FAISS index available. Call build_index() first."
                )
        return self._index

    @property
    def size(self) -> int:
        """Number of jokes in the index."""
        return len(self._jokes)

    def build_index(self, jokes: List[Dict], text_key: str = "question") -> None:
        """
        Build a new FAISS index from a list of jokes.

        Args:
            jokes: List of joke dicts.
            text_key: Which field to embed. Default "question" for setup-based
                      matching. Use "question answer" composite for full-joke matching.
        """
        logger.info(f"Building FAISS index from {len(jokes)} jokes")

        # Create composite text for richer embeddings
        texts = [
            f"{j.get('question', '')} {j.get('answer', '')}".strip()
            for j in jokes
        ]

        # Batch encode with normalization
        embeddings = self.model.encode(
            texts,
            batch_size=config.embedding.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Build FAISS index (Inner Product = Cosine Similarity when normalized)
        self.dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self.dimension)
        self._index.add(embeddings.astype(np.float32))
        self._jokes = jokes

        logger.info(
            f"Built index: {self._index.ntotal} vectors, dim={self.dimension}"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Dict, float]]:
        """
        Search the index for jokes similar to the query.

        Args:
            query: Search text (e.g., user's scenario description).
            top_k: Maximum number of results.
            score_threshold: Minimum similarity score to include.

        Returns:
            List of (joke_dict, similarity_score) tuples, sorted by score descending.
        """
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._jokes):
                continue
            if score >= score_threshold:
                results.append((self._jokes[idx], float(score)))

        return results

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[Tuple[Dict, float]]]:
        """Batch search for multiple queries at once."""
        query_embeddings = self.model.encode(
            queries, normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_embeddings, top_k)

        all_results = []
        for q_scores, q_indices in zip(scores, indices):
            results = []
            for score, idx in zip(q_scores, q_indices):
                if 0 <= idx < len(self._jokes):
                    results.append((self._jokes[idx], float(score)))
            all_results.append(results)

        return all_results

    def save(self, path: Optional[str] = None) -> None:
        """Persist FAISS index and metadata to disk."""
        save_path = path or self.index_path
        os.makedirs(save_path, exist_ok=True)

        faiss.write_index(self._index, os.path.join(save_path, "index.faiss"))
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(self._jokes, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved index ({self._index.ntotal} vectors) to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load FAISS index and metadata from disk."""
        load_path = path or self.index_path
        index_file = os.path.join(load_path, "index.faiss")
        meta_file = os.path.join(load_path, "metadata.json")

        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index not found at {index_file}")

        self._index = faiss.read_index(index_file)
        with open(meta_file, "r") as f:
            self._jokes = json.load(f)

        logger.info(
            f"Loaded index: {self._index.ntotal} vectors from {load_path}"
        )

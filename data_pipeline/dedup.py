"""
Joke Deduplicator

Two-stage deduplication pipeline:
1. Exact dedup: Normalized text hashing to remove identical jokes
2. Semantic dedup: Embedding-based similarity to remove paraphrased duplicates

Example from our dataset:
- Raw extracted: 4,268 jokes
- After exact dedup: 3,997 jokes (-6.3%)
- After semantic dedup: 3,303 jokes (-17.3% additional)
"""

import hashlib
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm

from config import config


@dataclass
class DedupStats:
    """Statistics from the deduplication process."""
    total_input: int
    after_exact_dedup: int
    after_semantic_dedup: int
    exact_duplicates_removed: int
    semantic_duplicates_removed: int

    @property
    def total_removed(self) -> int:
        return self.total_input - self.after_semantic_dedup

    @property
    def removal_rate(self) -> float:
        return self.total_removed / self.total_input if self.total_input > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"Dedup Stats:\n"
            f"  Input: {self.total_input}\n"
            f"  After exact dedup: {self.after_exact_dedup} "
            f"(-{self.exact_duplicates_removed}, {self.exact_duplicates_removed/self.total_input*100:.1f}%)\n"
            f"  After semantic dedup: {self.after_semantic_dedup} "
            f"(-{self.semantic_duplicates_removed}, {self.semantic_duplicates_removed/self.after_exact_dedup*100:.1f}%)\n"
            f"  Total removal rate: {self.removal_rate*100:.1f}%"
        )


class JokeDeduplicator:
    """
    Two-stage joke deduplication engine.

    Stage 1 (Exact Dedup):
    - Normalize text: lowercase, strip whitespace, remove punctuation
    - Hash normalized Q+A text
    - O(n) time, catches identical/near-identical jokes

    Stage 2 (Semantic Dedup):
    - Encode jokes using SentenceTransformer
    - Build FAISS index for efficient similarity search
    - Remove jokes with cosine similarity above threshold
    - Catches paraphrased duplicates (e.g., "Why was 6 afraid of 7?"
      vs "Why is six scared of seven?")
    """

    def __init__(
        self,
        model_name: str = None,
        similarity_threshold: float = None,
        batch_size: int = 64,
    ):
        self.model_name = model_name or config.embedding.model_name
        self.similarity_threshold = (
            similarity_threshold or config.pipeline.dedup_similarity_threshold
        )
        self.batch_size = batch_size
        self._model = None  # Lazy loading

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load embedding model to avoid cold start overhead."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def deduplicate(self, jokes: List[Dict]) -> Tuple[List[Dict], DedupStats]:
        """
        Run full two-stage deduplication pipeline.

        Args:
            jokes: List of joke dicts with "question" and "answer" keys.

        Returns:
            Tuple of (deduplicated jokes list, dedup statistics).
        """
        total_input = len(jokes)
        logger.info(f"Starting deduplication of {total_input} jokes")

        # Stage 1: Exact dedup
        exact_deduped = self._exact_dedup(jokes)
        exact_removed = total_input - len(exact_deduped)
        logger.info(f"Exact dedup: {total_input} → {len(exact_deduped)} (-{exact_removed})")

        # Stage 2: Semantic dedup
        semantic_deduped = self._semantic_dedup(exact_deduped)
        semantic_removed = len(exact_deduped) - len(semantic_deduped)
        logger.info(
            f"Semantic dedup: {len(exact_deduped)} → {len(semantic_deduped)} (-{semantic_removed})"
        )

        stats = DedupStats(
            total_input=total_input,
            after_exact_dedup=len(exact_deduped),
            after_semantic_dedup=len(semantic_deduped),
            exact_duplicates_removed=exact_removed,
            semantic_duplicates_removed=semantic_removed,
        )

        return semantic_deduped, stats

    def _exact_dedup(self, jokes: List[Dict]) -> List[Dict]:
        """
        Stage 1: Remove exact duplicates using normalized text hashing.

        Normalization: lowercase -> strip -> remove extra whitespace -> remove punctuation
        Hash: MD5 of normalized (question + answer)
        """
        seen_hashes: Set[str] = set()
        unique_jokes = []

        for joke in jokes:
            normalized = self._normalize_text(
                joke["question"] + " " + joke["answer"]
            )
            text_hash = hashlib.md5(normalized.encode()).hexdigest()

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_jokes.append(joke)

        return unique_jokes

    def _semantic_dedup(self, jokes: List[Dict]) -> List[Dict]:
        """
        Stage 2: Remove semantic duplicates using embedding similarity.

        Algorithm:
        1. Encode all jokes into dense embeddings
        2. Build a FAISS index for efficient nearest-neighbor search
        3. For each joke, find neighbors above similarity threshold
        4. Use greedy selection: keep the first occurrence, mark later
           duplicates for removal
        """
        if len(jokes) <= 1:
            return jokes

        # Encode all jokes
        texts = [f"{j['question']} {j['answer']}" for j in jokes]
        logger.info(f"Encoding {len(texts)} jokes for semantic dedup...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # For cosine similarity via dot product
        )

        # Build FAISS index (inner product = cosine similarity when normalized)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))

        # Find semantic duplicates using greedy selection
        removed_indices: Set[int] = set()

        for i in tqdm(range(len(jokes)), desc="Semantic dedup"):
            if i in removed_indices:
                continue

            # Search for similar jokes
            query = embeddings[i : i + 1].astype(np.float32)
            scores, indices = index.search(query, min(50, len(jokes)))

            for score, idx in zip(scores[0], indices[0]):
                if idx <= i or idx in removed_indices:
                    continue
                if score >= self.similarity_threshold:
                    removed_indices.add(idx)

        # Collect non-removed jokes
        unique_jokes = [j for i, j in enumerate(jokes) if i not in removed_indices]
        return unique_jokes

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for exact comparison."""
        import re

        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text)  # Collapse whitespace
        return text

"""
Semantic Scenario Matcher

Maps free-text user input to predefined scenario categories using
SentenceTransformer embeddings + FAISS nearest-neighbor search.

Problem this solves:
- System has predefined scenarios: "school", "animals", "family", etc.
- Users input free text: "jokes about my classroom", "funny pet stories"
- We need to map "classroom" → "school", "pet stories" → "animals"
- Simple keyword matching fails; semantic matching handles this naturally.

Example:
    matcher = ScenarioMatcher()
    result = matcher.match("jokes for a road trip")
    # Returns: ("everyday", 0.52)  or ("family", 0.48)
"""

from typing import List, Tuple, Optional, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

from config import config


# Scenario descriptions for richer embedding (not just the label)
DEFAULT_SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "animals": "funny jokes about animals, pets, dogs, cats, bears, fish, birds, zoo",
    "school": "jokes about school, classroom, teachers, students, homework, math, science class",
    "food": "jokes about food, cooking, eating, fruit, vegetables, candy, restaurants",
    "science": "science jokes, physics, chemistry, biology, experiments, atoms, space",
    "family": "family jokes, parents, siblings, grandparents, home, road trip, vacation",
    "sports": "sports jokes, soccer, basketball, baseball, football, swimming, running",
    "nature": "nature jokes, weather, ocean, mountains, trees, seasons, rain",
    "holidays": "holiday jokes, Christmas, Halloween, Thanksgiving, Easter, birthday, party",
    "fantasy": "fantasy jokes, dragons, wizards, unicorns, pirates, superheroes, magic",
    "everyday": "everyday life jokes, general humor, silly jokes, random fun, daily situations",
}


class ScenarioMatcher:
    """
    Semantic scenario normalization using embedding similarity.

    How it works:
    1. Pre-encode all scenario descriptions into a FAISS index
    2. When a user query arrives, encode it and find nearest scenario
    3. Return the best matching scenario if similarity exceeds threshold

    This enables the frontend to accept natural language input while
    the backend maintains a consistent tag vocabulary for retrieval.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        scenario_descriptions: Optional[Dict[str, str]] = None,
        match_threshold: Optional[float] = None,
    ):
        self.model_name = model_name or config.embedding.model_name
        self.match_threshold = (
            match_threshold or config.embedding.scenario_match_threshold
        )
        self.scenario_descriptions = (
            scenario_descriptions or DEFAULT_SCENARIO_DESCRIPTIONS
        )

        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._scenario_labels: List[str] = []

        self._build_index()

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _build_index(self) -> None:
        """Pre-encode scenario descriptions into a FAISS index."""
        self._scenario_labels = list(self.scenario_descriptions.keys())
        descriptions = list(self.scenario_descriptions.values())

        embeddings = self.model.encode(
            descriptions, normalize_embeddings=True
        ).astype(np.float32)

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

        logger.info(
            f"ScenarioMatcher initialized with {len(self._scenario_labels)} scenarios"
        )

    def match(self, user_input: str) -> Tuple[str, float]:
        """
        Map user input to the best matching predefined scenario.

        Args:
            user_input: Free-text description (e.g., "jokes for my classroom").

        Returns:
            Tuple of (scenario_label, similarity_score).
            Returns ("everyday", 0.0) if no match exceeds threshold.
        """
        # Check for exact match first (fast path)
        normalized = user_input.lower().strip()
        if normalized in self._scenario_labels:
            return (normalized, 1.0)

        # Semantic matching
        query_emb = self.model.encode(
            [user_input], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self._index.search(query_emb, 3)  # Top 3

        best_score = scores[0][0]
        best_idx = indices[0][0]

        if best_score >= self.match_threshold:
            matched = self._scenario_labels[best_idx]
            logger.info(
                f"Scenario match: '{user_input}' → '{matched}' (score={best_score:.3f})"
            )
            return (matched, float(best_score))
        else:
            logger.info(
                f"No confident match for '{user_input}' (best={best_score:.3f}), defaulting to 'everyday'"
            )
            return ("everyday", 0.0)

    def match_top_k(self, user_input: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k matching scenarios with scores."""
        query_emb = self.model.encode(
            [user_input], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self._index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._scenario_labels):
                results.append((self._scenario_labels[idx], float(score)))

        return results

    @property
    def available_scenarios(self) -> List[str]:
        """List all available scenario labels."""
        return list(self._scenario_labels)

"""
Preference-Aware Joke Retriever

Retrieves jokes from the dataset using a multi-signal approach:
1. Age group filtering
2. Scenario matching (exact + semantic)
3. User preference filtering (like/dislike/viewed history)
4. Randomized scoring to prevent stale results

The core insight: retrieval isn't just about finding relevant content,
it's about finding content that's relevant AND fresh AND aligned with
user taste. This is what separates a production recommender from a
simple database query.
"""

import random
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from loguru import logger

from .embeddings import JokeEmbeddingIndex
from .scenario_matcher import ScenarioMatcher
from config import config


@dataclass
class UserPreferences:
    """
    Encapsulates user preference signals for personalized retrieval.

    These signals come from the frontend via API:
    - liked_joke_ids: Jokes the user explicitly liked
    - disliked_joke_ids: Jokes the user explicitly disliked
    - viewed_joke_ids: Jokes the user has already seen
    - favorite_joke_ids: Jokes the user saved to favorites
    """
    liked_joke_ids: Set[str] = field(default_factory=set)
    disliked_joke_ids: Set[str] = field(default_factory=set)
    viewed_joke_ids: Set[str] = field(default_factory=set)
    favorite_joke_ids: Set[str] = field(default_factory=set)

    @property
    def has_history(self) -> bool:
        return bool(self.liked_joke_ids or self.disliked_joke_ids)


@dataclass
class RetrievalRequest:
    """Parameters for a joke retrieval request."""
    age_range: str = "5-7"
    scenario: str = "everyday"
    num_jokes: int = 5
    user_preferences: Optional[UserPreferences] = None
    is_authenticated: bool = False


@dataclass
class RetrievalResult:
    """Result of a retrieval operation with metadata."""
    jokes: List[Dict[str, Any]]
    matched_scenario: str
    scenario_confidence: float
    source: str  # "database", "semantic_search", or "mixed"
    shortfall: int  # How many more jokes are needed (for generation fallback)


class PreferenceAwareRetriever:
    """
    Multi-stage joke retriever with preference-based personalization.

    Retrieval pipeline:
    ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────┐
    │ Scenario     │ ──▶ │ Age + Theme  │ ──▶ │ Preference      │ ──▶ │ Rank &   │
    │ Normalization│     │ Filtering    │     │ Filtering       │     │ Diversify│
    └─────────────┘     └──────────────┘     └─────────────────┘     └──────────┘

    If the curated pool doesn't have enough jokes after filtering,
    the shortfall count tells the generation layer how many to produce.
    """

    def __init__(
        self,
        joke_index: JokeEmbeddingIndex,
        scenario_matcher: Optional[ScenarioMatcher] = None,
    ):
        self.joke_index = joke_index
        self.scenario_matcher = scenario_matcher or ScenarioMatcher()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Execute the full retrieval pipeline.

        Steps:
        1. Normalize scenario (semantic matching if needed)
        2. Filter by age range and scenario
        3. Apply preference-based filtering
        4. Rank, diversify, and return
        """
        # Step 1: Scenario normalization
        matched_scenario, scenario_confidence = self._normalize_scenario(
            request.scenario
        )
        logger.info(
            f"Scenario '{request.scenario}' → '{matched_scenario}' "
            f"(confidence={scenario_confidence:.3f})"
        )

        # Step 2: Content filtering (age + scenario)
        candidates = self._filter_candidates(
            age_range=request.age_range,
            scenario=matched_scenario,
        )
        logger.info(f"Found {len(candidates)} candidates after age+scenario filter")

        # Step 3: Preference-based filtering
        if request.user_preferences:
            candidates = self._apply_preference_filter(
                candidates, request.user_preferences
            )
            logger.info(
                f"{len(candidates)} candidates after preference filtering"
            )

        # Step 4: Rank and select
        selected = self._rank_and_select(
            candidates, request.num_jokes, request.user_preferences
        )

        shortfall = max(0, request.num_jokes - len(selected))

        if shortfall > 0:
            logger.info(
                f"Shortfall of {shortfall} jokes - generation fallback needed"
            )

        return RetrievalResult(
            jokes=selected,
            matched_scenario=matched_scenario,
            scenario_confidence=scenario_confidence,
            source="database" if len(selected) == request.num_jokes else "mixed",
            shortfall=shortfall,
        )

    def _normalize_scenario(self, scenario: str) -> tuple:
        """
        Map user input to a predefined scenario using semantic matching.

        Returns (matched_scenario, confidence_score).
        """
        return self.scenario_matcher.match(scenario)

    def _filter_candidates(
        self,
        age_range: str,
        scenario: str,
    ) -> List[Dict[str, Any]]:
        """
        Filter jokes by age range and scenario.

        Uses FAISS semantic search rather than exact tag matching,
        which handles cases where tags are incomplete or slightly
        different from the query.
        """
        # Compose a search query that captures both age and scenario
        search_query = f"{scenario} jokes for kids age {age_range}"
        results = self.joke_index.search(
            query=search_query,
            top_k=100,  # Large candidate pool for downstream filtering
            score_threshold=0.2,
        )

        candidates = []
        for joke, score in results:
            # Additional age range validation if metadata available
            joke_ages = joke.get("age_groups", [])
            if joke_ages and age_range not in joke_ages:
                # Soft penalty: don't exclude, but lower priority
                score *= 0.7

            joke_with_score = {**joke, "_retrieval_score": score}
            candidates.append(joke_with_score)

        return candidates

    def _apply_preference_filter(
        self,
        candidates: List[Dict],
        preferences: UserPreferences,
    ) -> List[Dict]:
        """
        Filter and re-score candidates based on user preferences.

        Rules:
        - Remove explicitly disliked jokes
        - Deprioritize already-viewed jokes (but don't hard exclude)
        - Boost jokes similar to liked/favorited ones
        """
        filtered = []

        for joke in candidates:
            joke_id = joke.get("id", joke.get("question", ""))

            # Hard exclude: disliked jokes
            if joke_id in preferences.disliked_joke_ids:
                continue

            score = joke.get("_retrieval_score", 0.5)

            # Soft penalty: already viewed
            if joke_id in preferences.viewed_joke_ids:
                score *= 0.3  # Strong penalty for seen jokes

            # Boost: similar to favorites (if we had their embeddings)
            if joke_id in preferences.favorite_joke_ids:
                score *= 1.5

            joke["_final_score"] = score
            filtered.append(joke)

        return filtered

    def _rank_and_select(
        self,
        candidates: List[Dict],
        num_jokes: int,
        preferences: Optional[UserPreferences] = None,
    ) -> List[Dict]:
        """
        Final ranking with controlled randomization.

        Uses a weighted random selection rather than pure top-k to
        ensure variety across repeated requests. This prevents the
        "always seeing the same top jokes" problem.
        """
        if not candidates:
            return []

        # Sort by score
        candidates.sort(
            key=lambda x: x.get("_final_score", x.get("_retrieval_score", 0)),
            reverse=True,
        )

        # Take top pool (3x requested to allow randomization)
        pool_size = min(len(candidates), num_jokes * 3)
        pool = candidates[:pool_size]

        # Weighted random selection
        if len(pool) <= num_jokes:
            selected = pool
        else:
            scores = [
                max(j.get("_final_score", j.get("_retrieval_score", 0.1)), 0.01)
                for j in pool
            ]
            total = sum(scores)
            weights = [s / total for s in scores]

            indices = []
            remaining_weights = list(weights)
            remaining_indices = list(range(len(pool)))

            for _ in range(num_jokes):
                if not remaining_indices:
                    break
                chosen = random.choices(
                    remaining_indices, weights=remaining_weights, k=1
                )[0]
                indices.append(chosen)

                # Remove chosen to avoid duplicates
                pos = remaining_indices.index(chosen)
                remaining_indices.pop(pos)
                remaining_weights.pop(pos)

            selected = [pool[i] for i in indices]

        # Clean up internal scoring fields before returning
        for joke in selected:
            joke.pop("_retrieval_score", None)
            joke.pop("_final_score", None)

        return selected

"""
Joke RAG Pipeline

End-to-end retrieval-augmented generation pipeline:
  Retrieve → Filter → Generate (if needed) → Return

This is the core serving logic that powers joke requests.
Design principle: retrieval-first, generation-as-fallback.

Benefits of this approach vs. pure generation:
- Lower cost: most requests served from curated content
- Lower latency: FAISS retrieval is sub-millisecond
- Higher quality: curated jokes are pre-vetted
- Consistency: same joke pool across all users
- Expandable: generated jokes are written back to the pool
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from loguru import logger

from .retrieval import PreferenceAwareRetriever, RetrievalRequest, UserPreferences
from .generation import GeminiJokeGenerator
from .embeddings import JokeEmbeddingIndex
from .scenario_matcher import ScenarioMatcher
from config import config


@dataclass
class JokeRequest:
    """Incoming joke request from the API layer."""
    age_range: str = "5-7"
    scenario: str = "everyday"
    num_jokes: int = 5
    user_id: Optional[str] = None
    liked_joke_ids: Optional[List[str]] = None
    disliked_joke_ids: Optional[List[str]] = None
    viewed_joke_ids: Optional[List[str]] = None
    favorite_joke_ids: Optional[List[str]] = None
    is_authenticated: bool = False


@dataclass
class JokeResponse:
    """Response containing jokes and metadata."""
    jokes: List[Dict[str, Any]]
    scenario: str
    scenario_confidence: float
    retrieval_count: int
    generation_count: int
    total: int


class JokeRAGPipeline:
    """
    Production RAG pipeline for joke serving.

    Request flow:
    ┌──────────┐     ┌───────────┐     ┌─────────────┐     ┌──────────────┐
    │ Request  │ ──▶ │ Retrieve  │ ──▶ │ Check count │ ──▶ │ Return jokes │
    │ (API)    │     │ + Filter  │     │ Sufficient? │     │ + metadata   │
    └──────────┘     └───────────┘     └──────┬──────┘     └──────────────┘
                                              │ NO
                                              ▼
                                       ┌──────────────┐     ┌─────────────┐
                                       │ Gemini Gen   │ ──▶ │ Async write │
                                       │ (fallback)   │     │ back to DB  │
                                       └──────────────┘     └─────────────┘
    """

    def __init__(
        self,
        joke_index: Optional[JokeEmbeddingIndex] = None,
        scenario_matcher: Optional[ScenarioMatcher] = None,
        generator: Optional[GeminiJokeGenerator] = None,
    ):
        self.joke_index = joke_index or JokeEmbeddingIndex()
        self.scenario_matcher = scenario_matcher or ScenarioMatcher()
        self.retriever = PreferenceAwareRetriever(
            joke_index=self.joke_index,
            scenario_matcher=self.scenario_matcher,
        )
        self.generator = generator or GeminiJokeGenerator()

    def serve(self, request: JokeRequest) -> JokeResponse:
        """
        Main entry point: serve a joke request.

        1. Build preference signals from user history
        2. Attempt retrieval from curated pool
        3. If shortfall, generate with Gemini (preference-aware)
        4. Merge retrieval + generation results
        5. Schedule async write-back of generated jokes
        """
        # Build user preferences
        preferences = UserPreferences(
            liked_joke_ids=set(request.liked_joke_ids or []),
            disliked_joke_ids=set(request.disliked_joke_ids or []),
            viewed_joke_ids=set(request.viewed_joke_ids or []),
            favorite_joke_ids=set(request.favorite_joke_ids or []),
        )

        # Step 1: Retrieve from curated pool
        retrieval_req = RetrievalRequest(
            age_range=request.age_range,
            scenario=request.scenario,
            num_jokes=request.num_jokes,
            user_preferences=preferences,
            is_authenticated=request.is_authenticated,
        )

        retrieval_result = self.retriever.retrieve(retrieval_req)
        retrieved_jokes = retrieval_result.jokes
        retrieval_count = len(retrieved_jokes)

        # Step 2: Generate fallback if shortfall
        generated_jokes = []
        if retrieval_result.shortfall > 0:
            logger.info(
                f"Generating {retrieval_result.shortfall} jokes via Gemini fallback"
            )

            # Fetch full liked/disliked jokes for preference conditioning
            liked_jokes = self._get_jokes_by_ids(
                request.liked_joke_ids or []
            )
            disliked_jokes = self._get_jokes_by_ids(
                request.disliked_joke_ids or []
            )

            generated_jokes = self.generator.generate(
                num_jokes=retrieval_result.shortfall,
                age_range=request.age_range,
                scenario=retrieval_result.matched_scenario,
                liked_jokes=liked_jokes,
                disliked_jokes=disliked_jokes,
                existing_jokes=retrieved_jokes,
            )

            # Schedule async write-back to expand content pool
            self._schedule_writeback(generated_jokes)

        # Merge results
        all_jokes = retrieved_jokes + generated_jokes

        return JokeResponse(
            jokes=all_jokes,
            scenario=retrieval_result.matched_scenario,
            scenario_confidence=retrieval_result.scenario_confidence,
            retrieval_count=retrieval_count,
            generation_count=len(generated_jokes),
            total=len(all_jokes),
        )

    def _get_jokes_by_ids(self, joke_ids: List[str]) -> List[Dict]:
        """
        Retrieve full joke objects by their IDs.
        Used to populate liked/disliked joke content for generation prompts.
        """
        if not joke_ids:
            return []

        id_set = set(joke_ids)
        return [
            joke for joke in self.joke_index._jokes
            if joke.get("id") in id_set
        ]

    def _schedule_writeback(self, jokes: List[Dict]) -> None:
        """
        Schedule async write-back of generated jokes to the content pool.

        Generated jokes that pass quality checks get added to the database,
        gradually expanding the curated pool and reducing future generation
        dependency. This creates a virtuous cycle:
        more users → more generation → bigger pool → less generation needed.
        """
        if not jokes:
            return

        logger.info(
            f"Scheduling write-back of {len(jokes)} generated jokes to content pool"
        )

        # In production, this would be an async task (e.g., Cloud Tasks, Celery)
        # For now, we add to the in-memory index
        try:
            for joke in jokes:
                self.joke_index._jokes.append(joke)
            logger.info(f"Write-back complete: pool size now {self.joke_index.size}")
        except Exception as e:
            logger.error(f"Write-back failed: {e}")

    def load_from_dataset(self, dataset_path: str) -> None:
        """
        Initialize the pipeline from a joke dataset file.
        Builds the FAISS index from the dataset.
        """
        import json

        with open(dataset_path, "r") as f:
            jokes = json.load(f)

        self.joke_index.build_index(jokes)
        logger.info(f"Pipeline initialized with {len(jokes)} jokes from {dataset_path}")

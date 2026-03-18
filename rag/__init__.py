"""
SixSeven Jokes - RAG Pipeline

Hybrid retrieval-generation system:
- FAISS-based semantic retrieval with scenario matching
- User preference signals (like/dislike/viewed history)
- Gemini fallback generation with preference-aware prompting
- Robust structured output parsing
"""

from .pipeline import JokeRAGPipeline
from .retrieval import PreferenceAwareRetriever
from .generation import GeminiJokeGenerator
from .scenario_matcher import ScenarioMatcher
from .embeddings import JokeEmbeddingIndex

__all__ = [
    "JokeRAGPipeline",
    "PreferenceAwareRetriever",
    "GeminiJokeGenerator",
    "ScenarioMatcher",
    "JokeEmbeddingIndex",
]

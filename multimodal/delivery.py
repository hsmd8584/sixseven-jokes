"""
Multimodal Content Delivery Pipeline

End-to-end pipeline for delivering jokes as multimodal content:
- Text joke → Voice synthesis → Cached audio → User delivery

Supports both synchronous and asynchronous delivery modes:
- Sync: Generate and return audio immediately (higher latency)
- Async: Return text immediately, process audio in background (lower TTFB)
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from loguru import logger

from .voice_synthesis import VoiceSynthesizer
from .audio_cache import AudioCacheManager
from config import config


@dataclass
class MultimodalJoke:
    """A joke with both text and audio representations."""
    question: str
    answer: str
    audio_url: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    content_hash: str = ""
    voice_setup: str = ""
    voice_punchline: str = ""
    audio_cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultimodalDeliveryPipeline:
    """
    Orchestrates multimodal joke delivery.

    Pipeline architecture:
    ┌──────────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────┐
    │ Joke     │ ──▶ │ Check Cache │ ──▶ │ Voice Synth  │ ──▶ │ Deliver  │
    │ (text)   │     │ (2-tier)    │     │ (if miss)    │     │ + cache  │
    └──────────┘     └─────────────┘     └──────────────┘     └──────────┘

    Async mode (recommended for production):
    - Returns text joke immediately (0 latency added)
    - Generates audio in background
    - Notifies frontend when audio is ready via callback/SSE
    """

    def __init__(
        self,
        synthesizer: Optional[VoiceSynthesizer] = None,
        cache_manager: Optional[AudioCacheManager] = None,
    ):
        self.synthesizer = synthesizer or VoiceSynthesizer()
        self.cache_manager = cache_manager or AudioCacheManager()

    def deliver(
        self,
        jokes: List[Dict[str, Any]],
        include_audio: bool = True,
        custom_voice: Optional[str] = None,
    ) -> List[MultimodalJoke]:
        """
        Synchronous multimodal delivery.

        Processes each joke: check cache → generate if needed → return.

        Args:
            jokes: List of joke dicts with "question" and "answer".
            include_audio: Whether to generate audio.
            custom_voice: Optional specific voice for all jokes.

        Returns:
            List of MultimodalJoke objects with text and audio.
        """
        results = []

        for joke in jokes:
            question = joke.get("question", "")
            answer = joke.get("answer", "")

            mm_joke = MultimodalJoke(
                question=question,
                answer=answer,
                voice_setup=custom_voice or self.synthesizer.setup_voice,
                voice_punchline=custom_voice or self.synthesizer.punchline_voice,
                metadata=joke,
            )

            if include_audio:
                audio_bytes, content_hash = self._get_or_generate_audio(
                    question, answer, custom_voice
                )
                mm_joke.audio_bytes = audio_bytes
                mm_joke.content_hash = content_hash

            results.append(mm_joke)

        logger.info(f"Delivered {len(results)} multimodal jokes")
        return results

    async def deliver_async(
        self,
        jokes: List[Dict[str, Any]],
        custom_voice: Optional[str] = None,
    ) -> List[MultimodalJoke]:
        """
        Asynchronous multimodal delivery.

        Processes all jokes concurrently for lower total latency.
        """
        tasks = [
            self._process_joke_async(joke, custom_voice)
            for joke in jokes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures
        successful = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Async delivery failed: {result}")
            else:
                successful.append(result)

        logger.info(
            f"Async delivery: {len(successful)}/{len(jokes)} jokes processed"
        )
        return successful

    async def _process_joke_async(
        self,
        joke: Dict,
        custom_voice: Optional[str],
    ) -> MultimodalJoke:
        """Process a single joke asynchronously."""
        question = joke.get("question", "")
        answer = joke.get("answer", "")

        # Check cache first (sync, fast)
        content_hash = VoiceSynthesizer._content_hash(question, answer)
        cached_audio = self.cache_manager.get(content_hash)

        if cached_audio:
            return MultimodalJoke(
                question=question,
                answer=answer,
                audio_bytes=cached_audio,
                content_hash=content_hash,
                audio_cached=True,
                voice_setup=custom_voice or self.synthesizer.setup_voice,
                voice_punchline=custom_voice or self.synthesizer.punchline_voice,
                metadata=joke,
            )

        # Generate audio asynchronously
        audio_bytes, content_hash = await self.synthesizer.synthesize_joke_async(
            question, answer, custom_voice
        )

        # Cache the result (async)
        await asyncio.to_thread(
            self.cache_manager.put,
            content_hash,
            audio_bytes,
            {"question": question[:100], "answer": answer[:100]},
        )

        return MultimodalJoke(
            question=question,
            answer=answer,
            audio_bytes=audio_bytes,
            content_hash=content_hash,
            audio_cached=False,
            voice_setup=custom_voice or self.synthesizer.setup_voice,
            voice_punchline=custom_voice or self.synthesizer.punchline_voice,
            metadata=joke,
        )

    def _get_or_generate_audio(
        self,
        question: str,
        answer: str,
        custom_voice: Optional[str],
    ) -> Tuple[bytes, str]:
        """
        Get audio from cache or generate new.

        Cache-aside pattern:
        1. Check cache
        2. If miss, generate
        3. Store in cache
        4. Return
        """
        content_hash = VoiceSynthesizer._content_hash(question, answer)

        # Check cache
        cached = self.cache_manager.get(content_hash)
        if cached is not None:
            logger.debug(f"Audio cache hit: {content_hash[:8]}...")
            return cached, content_hash

        # Generate
        audio_bytes, content_hash = self.synthesizer.synthesize_joke(
            question=question,
            answer=answer,
            custom_voice=custom_voice,
        )

        # Cache
        self.cache_manager.put(
            content_hash,
            audio_bytes,
            metadata={
                "question": question[:100],
                "answer": answer[:100],
                "voice": custom_voice or "dual_default",
            },
        )

        return audio_bytes, content_hash

    def get_cache_stats(self) -> Dict:
        """Get audio cache statistics."""
        return self.cache_manager.get_stats()

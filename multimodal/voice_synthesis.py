"""
Voice Synthesis Engine

Integrates with ElevenLabs API for high-quality text-to-speech generation.
Supports dual-voice mode where the joke setup and punchline are read by
different voices for a more engaging listening experience.

Voice configuration:
- Setup voice: Warm, storytelling tone (default: "Rachel")
- Punchline voice: Energetic, comedic tone (default: "Adam")
"""

import hashlib
import asyncio
from typing import Optional, Tuple
from pathlib import Path

from loguru import logger

from config import config


class VoiceSynthesizer:
    """
    ElevenLabs-based voice synthesis for joke audio generation.

    Design decisions:
    - Dual voice: Using two different voices for setup and punchline
      creates a conversational feel, like a comedy duo. This significantly
      improves the listening experience compared to single-voice narration.
    - Content-based hashing: Audio files are named by content hash,
      enabling natural dedup - the same joke text always produces the
      same file path, regardless of when it was generated.
    - Async generation: Audio generation is IO-bound (API calls),
      so async processing allows concurrent generation of multiple
      joke audio files.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        setup_voice: Optional[str] = None,
        punchline_voice: Optional[str] = None,
    ):
        self.api_key = api_key or config.elevenlabs.api_key
        self.setup_voice = setup_voice or config.elevenlabs.default_setup_voice
        self.punchline_voice = punchline_voice or config.elevenlabs.default_punchline_voice
        self.model_id = config.elevenlabs.model_id
        self.output_format = config.elevenlabs.output_format

        self._client = None

    @property
    def client(self):
        """Lazy-initialize ElevenLabs client."""
        if self._client is None:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self.api_key)
        return self._client

    def synthesize_joke(
        self,
        question: str,
        answer: str,
        output_dir: Optional[str] = None,
        custom_voice: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """
        Generate dual-voice audio for a joke.

        The setup (question) is read by one voice, and the punchline (answer)
        by a different voice. The two audio segments are concatenated.

        Args:
            question: The joke setup text.
            answer: The joke punchline text.
            output_dir: Optional directory to save the audio file.
            custom_voice: If provided, use this single voice for both parts.

        Returns:
            Tuple of (audio_bytes, content_hash).
        """
        content_hash = self._content_hash(question, answer)

        if custom_voice:
            # Single voice mode
            full_text = f"{question} ... {answer}"
            audio_bytes = self._generate_speech(full_text, custom_voice)
        else:
            # Dual voice mode: different voices for setup and punchline
            setup_audio = self._generate_speech(question, self.setup_voice)
            # Add a brief pause between setup and punchline
            pause = self._generate_silence(duration_ms=800)
            punchline_audio = self._generate_speech(answer, self.punchline_voice)
            audio_bytes = setup_audio + pause + punchline_audio

        # Optionally save to disk
        if output_dir:
            self._save_audio(audio_bytes, output_dir, content_hash)

        logger.info(
            f"Synthesized audio for joke (hash={content_hash[:8]}...): "
            f"{len(audio_bytes)} bytes"
        )

        return audio_bytes, content_hash

    async def synthesize_joke_async(
        self,
        question: str,
        answer: str,
        custom_voice: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """
        Async version of synthesize_joke.

        Generates setup and punchline audio concurrently for lower latency.
        """
        content_hash = self._content_hash(question, answer)

        if custom_voice:
            audio_bytes = await asyncio.to_thread(
                self._generate_speech, f"{question} ... {answer}", custom_voice
            )
        else:
            # Concurrent generation of both parts
            setup_task = asyncio.to_thread(
                self._generate_speech, question, self.setup_voice
            )
            punchline_task = asyncio.to_thread(
                self._generate_speech, answer, self.punchline_voice
            )

            setup_audio, punchline_audio = await asyncio.gather(
                setup_task, punchline_task
            )

            pause = self._generate_silence(duration_ms=800)
            audio_bytes = setup_audio + pause + punchline_audio

        return audio_bytes, content_hash

    def _generate_speech(self, text: str, voice: str) -> bytes:
        """Generate speech audio using ElevenLabs API."""
        try:
            audio_generator = self.client.text_to_speech.convert(
                voice_id=self._resolve_voice_id(voice),
                text=text,
                model_id=self.model_id,
                output_format=self.output_format,
            )

            # Collect all audio chunks
            audio_bytes = b""
            for chunk in audio_generator:
                audio_bytes += chunk

            return audio_bytes

        except Exception as e:
            logger.error(f"ElevenLabs TTS failed for voice '{voice}': {e}")
            raise

    def _resolve_voice_id(self, voice_name: str) -> str:
        """
        Resolve a voice name to an ElevenLabs voice ID.
        Caches the mapping after first lookup.
        """
        if not hasattr(self, "_voice_cache"):
            self._voice_cache = {}

        if voice_name in self._voice_cache:
            return self._voice_cache[voice_name]

        # If it looks like a voice ID already, return as-is
        if len(voice_name) > 15 and not " " in voice_name:
            return voice_name

        # Look up by name
        try:
            voices = self.client.voices.get_all()
            for v in voices.voices:
                if v.name.lower() == voice_name.lower():
                    self._voice_cache[voice_name] = v.voice_id
                    return v.voice_id
        except Exception as e:
            logger.warning(f"Voice lookup failed: {e}")

        # Return the name as-is (might work as ID)
        return voice_name

    @staticmethod
    def _generate_silence(duration_ms: int = 800) -> bytes:
        """Generate silent audio bytes for pause between setup and punchline."""
        # Simple silence: zero bytes for the given duration
        # For MP3 format, we approximate with empty frames
        # In production, use a proper audio library for accurate silence
        samples = int(44100 * duration_ms / 1000)
        return b"\x00" * (samples * 2)  # 16-bit silence

    @staticmethod
    def _content_hash(question: str, answer: str) -> str:
        """Generate deterministic hash from joke content."""
        content = f"{question.strip()}|{answer.strip()}"
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def _save_audio(audio_bytes: bytes, output_dir: str, filename: str) -> str:
        """Save audio bytes to disk."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        filepath = path / f"{filename}.mp3"
        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        return str(filepath)

    def list_available_voices(self) -> list:
        """List all available ElevenLabs voices."""
        try:
            voices = self.client.voices.get_all()
            return [
                {"name": v.name, "voice_id": v.voice_id, "category": v.category}
                for v in voices.voices
            ]
        except Exception as e:
            logger.error(f"Failed to list voices: {e}")
            return []

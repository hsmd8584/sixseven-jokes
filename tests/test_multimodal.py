"""
Tests for the Multimodal module.

Covers:
- Content hashing for cache keys
- Audio cache operations
- Delivery pipeline logic
"""

import os
import pytest
import tempfile

from multimodal.voice_synthesis import VoiceSynthesizer
from multimodal.audio_cache import AudioCacheManager


# ============================================================
# Voice Synthesis Tests
# ============================================================

class TestVoiceSynthesis:
    """Test voice synthesis utility methods."""

    def test_content_hash_deterministic(self):
        """Same content always produces the same hash."""
        hash1 = VoiceSynthesizer._content_hash("Why?", "Because!")
        hash2 = VoiceSynthesizer._content_hash("Why?", "Because!")
        assert hash1 == hash2

    def test_content_hash_different_for_different_content(self):
        """Different content produces different hashes."""
        hash1 = VoiceSynthesizer._content_hash("Why?", "Because!")
        hash2 = VoiceSynthesizer._content_hash("What?", "Nothing!")
        assert hash1 != hash2

    def test_content_hash_strips_whitespace(self):
        """Hash should normalize whitespace."""
        hash1 = VoiceSynthesizer._content_hash("Why?", "Because!")
        hash2 = VoiceSynthesizer._content_hash("  Why?  ", "  Because!  ")
        assert hash1 == hash2

    def test_generate_silence(self):
        """Test silence generation produces non-empty bytes."""
        silence = VoiceSynthesizer._generate_silence(duration_ms=500)
        assert len(silence) > 0

    def test_save_audio(self):
        """Test saving audio to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_data = b"fake_audio_data"
            path = VoiceSynthesizer._save_audio(audio_data, tmpdir, "test_hash")
            assert os.path.exists(path)
            assert path.endswith(".mp3")


# ============================================================
# Audio Cache Tests
# ============================================================

class TestAudioCache:
    """Test audio caching operations."""

    def test_local_cache_put_and_get(self):
        """Test writing and reading from local cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AudioCacheManager(local_cache_dir=tmpdir, firebase_bucket="")

            audio_data = b"test_audio_bytes"
            content_hash = "abc123"

            cache._save_local(content_hash, audio_data)
            retrieved = cache._get_local(content_hash)

            assert retrieved == audio_data

    def test_cache_miss_returns_none(self):
        """Test that cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AudioCacheManager(local_cache_dir=tmpdir, firebase_bucket="")
            result = cache._get_local("nonexistent_hash")
            assert result is None

    def test_cache_exists(self):
        """Test cache existence check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AudioCacheManager(local_cache_dir=tmpdir, firebase_bucket="")

            cache._save_local("exists_hash", b"data")
            assert cache.exists("exists_hash")

    def test_cache_stats(self):
        """Test cache statistics reporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AudioCacheManager(local_cache_dir=tmpdir, firebase_bucket="")

            cache._save_local("hash1", b"data1")
            cache._save_local("hash2", b"data2")

            stats = cache.get_stats()
            assert stats["local_entries"] == 2
            assert stats["local_size_mb"] >= 0

    def test_cache_invalidation(self):
        """Test removing cached audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = AudioCacheManager(local_cache_dir=tmpdir, firebase_bucket="")

            cache._save_local("to_delete", b"data")
            assert cache._get_local("to_delete") is not None

            cache.invalidate("to_delete")
            assert cache._get_local("to_delete") is None


# ============================================================
# Safety Filter Tests
# ============================================================

class TestSafetyFilter:
    """Test content safety filtering."""

    def test_safe_joke_passes(self):
        """Test that a clean joke passes safety check."""
        from guardrail.safety_filter import SafetyFilter

        filter = SafetyFilter(enable_llm_tier=False)
        result = filter.check_joke(
            "What do you call a bear with no teeth?",
            "A gummy bear!",
        )
        assert result.is_safe is True

    def test_unsafe_content_blocked(self):
        """Test that obviously unsafe content is blocked."""
        from guardrail.safety_filter import SafetyFilter

        filter = SafetyFilter(enable_llm_tier=False)
        result = filter.check_joke(
            "A joke about weapons",
            "Something about a gun",
        )
        assert result.is_safe is False
        assert "violence" in result.flagged_categories

    def test_batch_filtering(self):
        """Test batch safety filtering."""
        from guardrail.safety_filter import SafetyFilter

        filter = SafetyFilter(enable_llm_tier=False)
        jokes = [
            {"question": "Why was 6 afraid of 7?", "answer": "Because 7 ate 9!"},
            {"question": "Bad joke about murder", "answer": "Something violent"},
        ]
        safe_jokes = filter.filter_safe(jokes)
        assert len(safe_jokes) == 1

"""
Audio Cache Manager

Implements a two-tier caching strategy for generated audio:
1. Local disk cache: Immediate access, no network latency
2. Firebase Storage: Persistent, shared across instances

Cache key = SHA-256 hash of joke content, ensuring:
- Same joke always maps to the same cache entry
- No duplicate audio generation for the same content
- Natural invalidation when joke content changes
"""

import os
import hashlib
import json
from typing import Optional, Dict, Tuple
from pathlib import Path
from datetime import datetime

from loguru import logger

from config import config


class AudioCacheManager:
    """
    Two-tier audio caching with local disk and Firebase Storage.

    Cache flow:
    1. Check local disk cache (fastest, <1ms)
    2. Check Firebase Storage (slower, ~100ms)
    3. Generate new audio (slowest, ~2-5s)
    4. Write-through: save to both tiers on generation

    This dramatically reduces cost and latency:
    - Default voices: Most popular jokes get cached quickly
    - Custom voices: Per-user cache in Firebase Storage
    - Cache hit rate typically >60% after initial warmup
    """

    def __init__(
        self,
        local_cache_dir: Optional[str] = None,
        firebase_bucket: Optional[str] = None,
    ):
        self.local_cache_dir = local_cache_dir or config.elevenlabs.cache_dir
        self.firebase_bucket = firebase_bucket or config.firebase.storage_bucket

        # Ensure local cache directory exists
        os.makedirs(self.local_cache_dir, exist_ok=True)

        # Firebase Storage client (lazy init)
        self._storage_client = None

        # In-memory index for fast lookups
        self._cache_index: Dict[str, Dict] = {}
        self._load_local_index()

    def get(self, content_hash: str) -> Optional[bytes]:
        """
        Retrieve cached audio by content hash.

        Checks local cache first, then Firebase Storage.
        Returns None if not cached.
        """
        # Tier 1: Local disk
        local_audio = self._get_local(content_hash)
        if local_audio is not None:
            logger.debug(f"Cache HIT (local): {content_hash[:8]}...")
            return local_audio

        # Tier 2: Firebase Storage
        remote_audio = self._get_firebase(content_hash)
        if remote_audio is not None:
            logger.debug(f"Cache HIT (firebase): {content_hash[:8]}...")
            # Promote to local cache
            self._save_local(content_hash, remote_audio)
            return remote_audio

        logger.debug(f"Cache MISS: {content_hash[:8]}...")
        return None

    def put(
        self,
        content_hash: str,
        audio_bytes: bytes,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Store audio in both cache tiers (write-through).

        Args:
            content_hash: SHA-256 hash of the joke content.
            audio_bytes: The generated audio data.
            metadata: Optional metadata (voice info, joke text, etc.).
        """
        # Write to local cache
        self._save_local(content_hash, audio_bytes, metadata)

        # Write to Firebase Storage (async in production)
        self._save_firebase(content_hash, audio_bytes, metadata)

        logger.debug(
            f"Cached audio: {content_hash[:8]}... ({len(audio_bytes)} bytes)"
        )

    def exists(self, content_hash: str) -> bool:
        """Check if audio exists in any cache tier."""
        return (
            content_hash in self._cache_index
            or self._local_path(content_hash).exists()
        )

    def invalidate(self, content_hash: str) -> None:
        """Remove audio from all cache tiers."""
        # Remove from local
        local_path = self._local_path(content_hash)
        if local_path.exists():
            local_path.unlink()

        # Remove from index
        self._cache_index.pop(content_hash, None)

        # Remove from Firebase Storage
        self._delete_firebase(content_hash)

        logger.info(f"Invalidated cache for {content_hash[:8]}...")

    def get_stats(self) -> Dict:
        """Return cache statistics."""
        local_files = list(Path(self.local_cache_dir).glob("*.mp3"))
        total_size = sum(f.stat().st_size for f in local_files)

        return {
            "local_entries": len(local_files),
            "local_size_mb": round(total_size / (1024 * 1024), 2),
            "index_entries": len(self._cache_index),
        }

    # --- Local cache operations ---

    def _get_local(self, content_hash: str) -> Optional[bytes]:
        """Read audio from local disk cache."""
        path = self._local_path(content_hash)
        if path.exists():
            return path.read_bytes()
        return None

    def _save_local(
        self,
        content_hash: str,
        audio_bytes: bytes,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save audio to local disk cache."""
        path = self._local_path(content_hash)
        path.write_bytes(audio_bytes)

        # Update index
        self._cache_index[content_hash] = {
            "size": len(audio_bytes),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._save_local_index()

    def _local_path(self, content_hash: str) -> Path:
        """Get local file path for a content hash."""
        return Path(self.local_cache_dir) / f"{content_hash}.mp3"

    def _load_local_index(self) -> None:
        """Load the local cache index from disk."""
        index_path = Path(self.local_cache_dir) / "cache_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    self._cache_index = json.load(f)
            except Exception:
                self._cache_index = {}

    def _save_local_index(self) -> None:
        """Persist the cache index to disk."""
        index_path = Path(self.local_cache_dir) / "cache_index.json"
        try:
            with open(index_path, "w") as f:
                json.dump(self._cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    # --- Firebase Storage operations ---

    def _get_firebase(self, content_hash: str) -> Optional[bytes]:
        """Download audio from Firebase Storage."""
        try:
            bucket = self._get_bucket()
            if bucket is None:
                return None

            blob = bucket.blob(f"audio_cache/{content_hash}.mp3")
            if blob.exists():
                return blob.download_as_bytes()
        except Exception as e:
            logger.warning(f"Firebase Storage read failed: {e}")
        return None

    def _save_firebase(
        self,
        content_hash: str,
        audio_bytes: bytes,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Upload audio to Firebase Storage."""
        try:
            bucket = self._get_bucket()
            if bucket is None:
                return

            blob = bucket.blob(f"audio_cache/{content_hash}.mp3")
            blob.upload_from_string(
                audio_bytes,
                content_type="audio/mpeg",
            )

            if metadata:
                blob.metadata = metadata
                blob.patch()
        except Exception as e:
            logger.warning(f"Firebase Storage write failed: {e}")

    def _delete_firebase(self, content_hash: str) -> None:
        """Delete audio from Firebase Storage."""
        try:
            bucket = self._get_bucket()
            if bucket is None:
                return

            blob = bucket.blob(f"audio_cache/{content_hash}.mp3")
            if blob.exists():
                blob.delete()
        except Exception as e:
            logger.warning(f"Firebase Storage delete failed: {e}")

    def _get_bucket(self):
        """Get Firebase Storage bucket (lazy init)."""
        if not self.firebase_bucket:
            return None

        try:
            import firebase_admin
            from firebase_admin import storage

            if not firebase_admin._apps:
                cred_path = config.firebase.credentials_path
                if cred_path and os.path.exists(cred_path):
                    cred = firebase_admin.credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(
                        cred, {"storageBucket": self.firebase_bucket}
                    )
                else:
                    return None

            return storage.bucket()
        except Exception as e:
            logger.warning(f"Firebase bucket init failed: {e}")
            return None

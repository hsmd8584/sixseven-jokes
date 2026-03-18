"""
SixSeven Jokes - Configuration Management

Centralized configuration with environment variable support
and sensible defaults for development and production.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Project root
ROOT_DIR = Path(__file__).parent


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and FAISS index."""
    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    dimension: int = 384  # Matches all-MiniLM-L6-v2 output dim
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", str(ROOT_DIR / "data" / "faiss_index"))
    similarity_threshold: float = 0.85  # For semantic dedup
    scenario_match_threshold: float = 0.4  # For scenario normalization
    batch_size: int = 64


@dataclass
class GeminiConfig:
    """Configuration for Google Gemini LLM."""
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    model_name: str = "gemini-1.5-flash"
    vision_model_name: str = "gemini-1.5-flash"
    temperature: float = 0.8
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40


@dataclass
class FirebaseConfig:
    """Configuration for Firebase services."""
    credentials_path: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "")
    storage_bucket: str = os.getenv("FIREBASE_STORAGE_BUCKET", "")
    jokes_collection: str = "jokes"
    users_collection: str = "users"
    audio_collection: str = "audio_cache"


@dataclass
class ElevenLabsConfig:
    """Configuration for ElevenLabs voice synthesis."""
    api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    default_setup_voice: str = "Rachel"  # Voice for joke setup
    default_punchline_voice: str = "Adam"  # Voice for punchline
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"
    cache_dir: str = str(ROOT_DIR / "audio_cache")


@dataclass
class FineTuningConfig:
    """Configuration for model fine-tuning."""
    base_model: str = os.getenv("BASE_MODEL", "google/gemma-2b")
    hf_token: str = os.getenv("HF_TOKEN", "")
    output_dir: str = os.getenv("FINETUNED_MODEL_PATH", str(ROOT_DIR / "models" / "sixseven-joke-gen"))
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    raw_data_dir: str = str(ROOT_DIR / "data" / "raw")
    processed_data_dir: str = str(ROOT_DIR / "data" / "processed")
    joke_dataset_path: str = os.getenv("JOKE_DATASET_PATH", str(ROOT_DIR / "data" / "jokes.json"))
    dedup_similarity_threshold: float = 0.88
    supported_age_groups: List[str] = field(default_factory=lambda: [
        "3-5", "5-7", "7-9", "9-12", "12-15"
    ])
    supported_themes: List[str] = field(default_factory=lambda: [
        "animals", "school", "food", "science", "family",
        "sports", "nature", "holidays", "fantasy", "everyday"
    ])
    supported_joke_types: List[str] = field(default_factory=lambda: [
        "pun", "knock_knock", "riddle", "one_liner", "story"
    ])


@dataclass
class GuardrailConfig:
    """Configuration for content safety guardrails."""
    enabled: bool = True
    model_name: str = "gemini-1.5-flash"  # Lightweight watchdog model
    max_toxicity_score: float = 0.3
    blocked_categories: List[str] = field(default_factory=lambda: [
        "violence", "sexual", "hate_speech", "self_harm", "dangerous"
    ])


@dataclass
class AppConfig:
    """Root configuration combining all sub-configs."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Singleton config instance
config = AppConfig()

"""
Joke Data Pipeline Orchestrator

End-to-end pipeline that:
1. Discovers and routes source files to appropriate extractors
2. Runs LLM-based tagging on extracted jokes
3. Performs two-stage deduplication
4. Outputs a clean, tagged joke dataset

Usage:
    pipeline = JokeDataPipeline(gemini_api_key="...")
    dataset, stats = pipeline.run("data/raw/")
    pipeline.save_dataset(dataset, "data/jokes.json")
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

from loguru import logger

from .extractors.base import RawJoke
from .extractors.pdf_extractor import PDFJokeExtractor
from .extractors.image_extractor import ImageJokeExtractor
from .extractors.text_extractor import TextJokeExtractor
from .tagger import LLMJokeTagger, TaggedJoke
from .dedup import JokeDeduplicator, DedupStats
from config import config


class JokeDataPipeline:
    """
    Orchestrates the full joke data pipeline.

    Architecture:
    ┌──────────────┐     ┌───────────┐     ┌────────────┐     ┌──────────┐
    │ Source Files  │ ──▶ │ Extractors│ ──▶ │ LLM Tagger │ ──▶ │  Dedup   │ ──▶ Dataset
    │ PDF/Img/Text │     │ (per type)│     │ (batched)  │     │ (2-stage)│
    └──────────────┘     └───────────┘     └────────────┘     └──────────┘

    Design decisions:
    - Factory pattern for extractor selection based on file type
    - Batched LLM calls for cost/latency optimization
    - Two-stage dedup (exact + semantic) for maximum quality
    """

    # File extension to extractor mapping
    EXTRACTOR_MAP = {
        ".pdf": "pdf",
        ".txt": "text",
        ".md": "text",
        ".csv": "text",
        ".jsonl": "text",
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".webp": "image",
        ".gif": "image",
        ".bmp": "image",
    }

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        dedup_threshold: Optional[float] = None,
    ):
        api_key = gemini_api_key or config.gemini.api_key

        # Initialize extractors (Factory Pattern)
        self.extractors = {
            "pdf": PDFJokeExtractor(gemini_api_key=api_key),
            "image": ImageJokeExtractor(gemini_api_key=api_key),
            "text": TextJokeExtractor(gemini_api_key=api_key),
        }

        # Initialize tagger and deduplicator
        self.tagger = LLMJokeTagger(gemini_api_key=api_key)
        self.deduplicator = JokeDeduplicator(
            model_name=embedding_model,
            similarity_threshold=dedup_threshold,
        )

    def run(
        self,
        source_dir: str,
        tag_jokes: bool = True,
        deduplicate: bool = True,
        filter_unsafe: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute the full pipeline.

        Args:
            source_dir: Directory containing source files (PDFs, images, text).
            tag_jokes: Whether to run LLM tagging.
            deduplicate: Whether to run deduplication.
            filter_unsafe: Whether to filter non-family-friendly content.

        Returns:
            Tuple of (dataset as list of dicts, pipeline statistics).
        """
        start_time = datetime.now()
        stats = {"start_time": start_time.isoformat()}

        # Step 1: Discover and extract
        logger.info(f"Step 1/3: Extracting jokes from {source_dir}")
        raw_jokes = self._extract_from_directory(source_dir)
        stats["raw_extracted"] = len(raw_jokes)
        logger.info(f"Extracted {len(raw_jokes)} raw jokes")

        # Convert to dicts for downstream processing
        joke_dicts = [j.to_dict() for j in raw_jokes]

        # Step 2: Tag with LLM
        if tag_jokes:
            logger.info("Step 2/3: Tagging jokes with LLM")
            tagged_jokes = self.tagger.tag_jokes(joke_dicts, filter_unsafe=filter_unsafe)
            joke_dicts = [j.to_dict() for j in tagged_jokes]
            stats["after_tagging"] = len(joke_dicts)
        else:
            logger.info("Step 2/3: Skipping LLM tagging")
            stats["after_tagging"] = len(joke_dicts)

        # Step 3: Deduplicate
        if deduplicate:
            logger.info("Step 3/3: Deduplicating jokes")
            joke_dicts, dedup_stats = self.deduplicator.deduplicate(joke_dicts)
            stats["dedup"] = {
                "total_input": dedup_stats.total_input,
                "after_exact": dedup_stats.after_exact_dedup,
                "after_semantic": dedup_stats.after_semantic_dedup,
                "removal_rate": f"{dedup_stats.removal_rate*100:.1f}%",
            }
            logger.info(f"\n{dedup_stats}")
        else:
            logger.info("Step 3/3: Skipping deduplication")

        stats["final_count"] = len(joke_dicts)
        stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Pipeline complete: {stats['raw_extracted']} → {stats['final_count']} jokes "
            f"in {stats['duration_seconds']:.1f}s"
        )

        return joke_dicts, stats

    def _extract_from_directory(self, source_dir: str) -> List[RawJoke]:
        """
        Walk directory, route each file to the appropriate extractor.
        Uses factory pattern to select extractor based on file extension.
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        all_jokes = []
        file_count = 0

        for file_path in sorted(source_path.rglob("*")):
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            extractor_type = self.EXTRACTOR_MAP.get(ext)

            if extractor_type is None:
                logger.debug(f"Skipping unsupported file: {file_path}")
                continue

            extractor = self.extractors[extractor_type]
            file_count += 1

            try:
                jokes = extractor.extract(str(file_path))
                all_jokes.extend(jokes)
                logger.info(f"  [{extractor_type.upper()}] {file_path.name}: {len(jokes)} jokes")
            except Exception as e:
                logger.error(f"  Failed to process {file_path.name}: {e}")

        logger.info(f"Processed {file_count} files, extracted {len(all_jokes)} total jokes")
        return all_jokes

    @staticmethod
    def save_dataset(dataset: List[Dict], output_path: str) -> None:
        """Save the processed dataset to JSON."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(dataset)} jokes to {output_path}")

    @staticmethod
    def load_dataset(path: str) -> List[Dict]:
        """Load a previously saved dataset from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

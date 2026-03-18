"""
Tests for the Data Pipeline module.

Covers:
- Text extractor regex patterns
- Deduplication logic (exact + semantic)
- Tagger validation
- Pipeline orchestration
"""

import pytest
from unittest.mock import patch, MagicMock

from data_pipeline.extractors.base import RawJoke
from data_pipeline.extractors.text_extractor import TextJokeExtractor
from data_pipeline.dedup import JokeDeduplicator
from data_pipeline.tagger import LLMJokeTagger, TaggedJoke


# ============================================================
# Text Extractor Tests
# ============================================================

class TestTextExtractor:
    """Test regex-based and multi-strategy text extraction."""

    def setup_method(self):
        self.extractor = TextJokeExtractor(gemini_api_key=None)

    def test_qa_format_extraction(self):
        """Test extraction from Q: / A: format."""
        content = """
Q: What do you call a bear with no teeth?
A: A gummy bear!

Q: Why did the banana go to the doctor?
A: Because it wasn't peeling well!
"""
        jokes = self.extractor._extract_with_regex(content, "test.txt")
        assert len(jokes) == 2
        assert "bear" in jokes[0].question.lower()
        assert "gummy" in jokes[0].answer.lower()

    def test_question_answer_format(self):
        """Test extraction from Question: / Answer: format."""
        content = """
Question: Why don't scientists trust atoms?
Answer: Because they make up everything!
"""
        jokes = self.extractor._extract_with_regex(content, "test.txt")
        assert len(jokes) >= 1
        assert "atoms" in jokes[0].question.lower()

    def test_natural_question_format(self):
        """Test extraction from natural question patterns."""
        content = """
Why did the math book look so sad?
Because it had too many problems!

What do you call a sleeping dinosaur?
A dino-snore!
"""
        jokes = self.extractor._extract_with_regex(content, "test.txt")
        assert len(jokes) >= 1

    def test_empty_content(self):
        """Test graceful handling of empty content."""
        jokes = self.extractor._extract_with_regex("", "test.txt")
        assert len(jokes) == 0

    def test_no_jokes_in_content(self):
        """Test content without jokes."""
        content = "This is just regular text with no jokes."
        jokes = self.extractor._extract_with_regex(content, "test.txt")
        assert len(jokes) == 0

    def test_dedup_in_regex(self):
        """Test that duplicate questions are deduplicated."""
        content = """
Q: What do you call a bear with no teeth?
A: A gummy bear!

Q: What do you call a bear with no teeth?
A: A gummy bear!
"""
        jokes = self.extractor._extract_with_regex(content, "test.txt")
        assert len(jokes) == 1  # Should deduplicate


# ============================================================
# Deduplication Tests
# ============================================================

class TestDeduplicator:
    """Test exact and semantic deduplication."""

    def test_exact_dedup_identical(self):
        """Test removal of exactly identical jokes."""
        dedup = JokeDeduplicator()
        jokes = [
            {"question": "Why was 6 afraid of 7?", "answer": "Because 7 ate 9!"},
            {"question": "Why was 6 afraid of 7?", "answer": "Because 7 ate 9!"},
            {"question": "What do you call a bear?", "answer": "A gummy bear!"},
        ]
        result = dedup._exact_dedup(jokes)
        assert len(result) == 2

    def test_exact_dedup_case_insensitive(self):
        """Test that exact dedup is case-insensitive."""
        dedup = JokeDeduplicator()
        jokes = [
            {"question": "Why was 6 afraid of 7?", "answer": "Because 7 ate 9!"},
            {"question": "WHY WAS 6 AFRAID OF 7?", "answer": "BECAUSE 7 ATE 9!"},
        ]
        result = dedup._exact_dedup(jokes)
        assert len(result) == 1

    def test_exact_dedup_whitespace(self):
        """Test that exact dedup normalizes whitespace."""
        dedup = JokeDeduplicator()
        jokes = [
            {"question": "Why  was  6  afraid?", "answer": "Because 7 ate 9!"},
            {"question": "Why was 6 afraid?", "answer": "Because 7 ate 9!"},
        ]
        result = dedup._exact_dedup(jokes)
        assert len(result) == 1

    def test_normalize_text(self):
        """Test text normalization."""
        result = JokeDeduplicator._normalize_text("  Hello, World!  ")
        assert result == "hello world"


# ============================================================
# Tagger Tests
# ============================================================

class TestTagger:
    """Test tag validation and default handling."""

    def test_validate_tags_valid(self):
        """Test validation with all valid tags."""
        tagger = LLMJokeTagger.__new__(LLMJokeTagger)
        tagger.age_groups = ["3-5", "5-7", "7-9"]
        result = tagger._validate_tags(["3-5", "5-7"], ["3-5", "5-7", "7-9"])
        assert result == ["3-5", "5-7"]

    def test_validate_tags_invalid(self):
        """Test validation with invalid tags defaults to first valid."""
        tagger = LLMJokeTagger.__new__(LLMJokeTagger)
        result = tagger._validate_tags(["invalid"], ["3-5", "5-7"])
        assert result == ["3-5"]

    def test_validate_single_valid(self):
        """Test single value validation."""
        tagger = LLMJokeTagger.__new__(LLMJokeTagger)
        result = tagger._validate_single("pun", ["pun", "riddle", "knock_knock"])
        assert result == "pun"

    def test_validate_single_invalid(self):
        """Test single value validation with invalid input."""
        tagger = LLMJokeTagger.__new__(LLMJokeTagger)
        result = tagger._validate_single("invalid", ["pun", "riddle"])
        assert result == "pun"

    def test_default_tags(self):
        """Test applying default tags as fallback."""
        tagger = LLMJokeTagger.__new__(LLMJokeTagger)
        batch = [
            {"question": "Test Q", "answer": "Test A"},
        ]
        result = tagger._apply_default_tags(batch)
        assert len(result) == 1
        assert result[0].family_friendly is True
        assert result[0].joke_type == "pun"


# ============================================================
# RawJoke Model Tests
# ============================================================

class TestRawJoke:
    """Test RawJoke dataclass."""

    def test_to_dict(self):
        joke = RawJoke(
            question="Why?",
            answer="Because!",
            source_file="test.txt",
            source_type="text",
        )
        d = joke.to_dict()
        assert d["question"] == "Why?"
        assert d["source_type"] == "text"

    def test_default_confidence(self):
        joke = RawJoke(
            question="Q", answer="A",
            source_file="f", source_type="text"
        )
        assert joke.confidence == 1.0

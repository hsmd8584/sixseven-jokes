"""
Tests for the RAG Pipeline module.

Covers:
- Structured output parsing (most critical for production reliability)
- Scenario matching
- Retrieval preference filtering
- Generation dedup
"""

import pytest
import json

from rag.structured_output import StructuredOutputParser


# ============================================================
# Structured Output Parser Tests (Critical for Production)
# ============================================================

class TestStructuredOutputParser:
    """
    Extensive tests for JSON parsing from LLM output.
    This is one of the most failure-prone areas in production LLM systems.
    """

    def test_parse_clean_json(self):
        """Test parsing perfectly formatted JSON."""
        text = '[{"question": "Why?", "answer": "Because!"}]'
        result = StructuredOutputParser.parse_json_array(text)
        assert len(result) == 1
        assert result[0]["question"] == "Why?"

    def test_parse_markdown_wrapped(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        text = '```json\n[{"question": "Why?", "answer": "Because!"}]\n```'
        result = StructuredOutputParser.parse_json_array(text)
        assert len(result) == 1

    def test_parse_markdown_no_language(self):
        """Test markdown code block without language specifier."""
        text = '```\n[{"question": "Why?", "answer": "Because!"}]\n```'
        result = StructuredOutputParser.parse_json_array(text)
        assert len(result) == 1

    def test_parse_with_preamble(self):
        """Test JSON with text before the array."""
        text = 'Here are the jokes:\n[{"question": "Why?", "answer": "Because!"}]'
        result = StructuredOutputParser.parse_json_array(text)
        assert len(result) == 1

    def test_parse_trailing_comma(self):
        """Test JSON with trailing comma (common LLM error)."""
        text = '[{"question": "Why?", "answer": "Because!"},]'
        result = StructuredOutputParser.parse_json_array(text)
        assert len(result) == 1

    def test_parse_single_object(self):
        """Test that a single object (not array) is wrapped."""
        text = '{"question": "Why?", "answer": "Because!"}'
        result = StructuredOutputParser.parse_json_array(text)
        assert len(result) == 1

    def test_parse_empty_input(self):
        """Test empty input returns empty list."""
        assert StructuredOutputParser.parse_json_array("") == []
        assert StructuredOutputParser.parse_json_array("  ") == []
        assert StructuredOutputParser.parse_json_array(None) == []

    def test_parse_multiple_objects(self):
        """Test parsing multiple joke objects."""
        text = """[
            {"question": "Why was 6 afraid of 7?", "answer": "Because 7 ate 9!"},
            {"question": "What do you call a bear?", "answer": "A gummy bear!"}
        ]"""
        result = StructuredOutputParser.parse_json_array(text)
        assert len(result) == 2

    def test_parse_with_extra_fields(self):
        """Test that extra fields are preserved."""
        text = '[{"question": "Why?", "answer": "Because!", "confidence": 0.9}]'
        result = StructuredOutputParser.parse_json_array(text)
        assert result[0]["confidence"] == 0.9

    def test_strip_markdown_basic(self):
        """Test markdown stripping."""
        text = "```json\n{\"key\": \"value\"}\n```"
        result = StructuredOutputParser._strip_markdown(text)
        assert result == '{"key": "value"}'

    def test_extract_json_array(self):
        """Test JSON array extraction from mixed text."""
        text = 'Some text [1, 2, 3] more text'
        result = StructuredOutputParser._extract_json_array(text)
        assert result == "[1, 2, 3]"

    def test_fix_trailing_commas(self):
        """Test fixing trailing commas."""
        text = '{"a": 1,}'
        result = StructuredOutputParser._fix_common_issues(text)
        assert result == '{"a": 1}'

    def test_regex_fallback(self):
        """Test regex fallback for severely malformed JSON."""
        text = 'broken { "question": "Why?", "answer": "Because!" } garbage'
        result = StructuredOutputParser._regex_fallback(text)
        assert len(result) == 1
        assert result[0]["question"] == "Why?"

    def test_nested_quotes_in_json(self):
        """Test handling of escaped quotes in JSON strings."""
        text = r'[{"question": "What did the \"bear\" say?", "answer": "Nothing!"}]'
        result = StructuredOutputParser.parse_json_array(text)
        assert len(result) == 1


# ============================================================
# Scenario Matcher Tests (using mock to avoid model loading)
# ============================================================

class TestScenarioMatcher:
    """Test scenario matching logic."""

    def test_exact_match_bypass(self):
        """Test that exact matches skip embedding search."""
        from unittest.mock import MagicMock, patch
        import numpy as np

        with patch("rag.scenario_matcher.SentenceTransformer") as MockModel:
            mock_instance = MagicMock()
            mock_instance.encode.return_value = np.random.rand(10, 384).astype(np.float32)
            MockModel.return_value = mock_instance

            from rag.scenario_matcher import ScenarioMatcher
            matcher = ScenarioMatcher()
            result, score = matcher.match("animals")
            assert result == "animals"
            assert score == 1.0


# ============================================================
# Retrieval Preference Tests
# ============================================================

class TestPreferenceFiltering:
    """Test user preference filtering logic."""

    def test_disliked_jokes_excluded(self):
        """Test that disliked jokes are hard-excluded."""
        from rag.retrieval import PreferenceAwareRetriever, UserPreferences

        retriever = PreferenceAwareRetriever.__new__(PreferenceAwareRetriever)

        candidates = [
            {"id": "1", "question": "Q1", "_retrieval_score": 0.9},
            {"id": "2", "question": "Q2", "_retrieval_score": 0.8},
            {"id": "3", "question": "Q3", "_retrieval_score": 0.7},
        ]

        prefs = UserPreferences(disliked_joke_ids={"2"})
        filtered = retriever._apply_preference_filter(candidates, prefs)

        ids = [j["id"] for j in filtered]
        assert "2" not in ids
        assert len(filtered) == 2

    def test_viewed_jokes_deprioritized(self):
        """Test that viewed jokes get score penalty."""
        from rag.retrieval import PreferenceAwareRetriever, UserPreferences

        retriever = PreferenceAwareRetriever.__new__(PreferenceAwareRetriever)

        candidates = [
            {"id": "1", "question": "Q1", "_retrieval_score": 0.9},
            {"id": "2", "question": "Q2", "_retrieval_score": 0.9},
        ]

        prefs = UserPreferences(viewed_joke_ids={"1"})
        filtered = retriever._apply_preference_filter(candidates, prefs)

        # Viewed joke should have lower score
        viewed = next(j for j in filtered if j["id"] == "1")
        unviewed = next(j for j in filtered if j["id"] == "2")
        assert viewed["_final_score"] < unviewed["_final_score"]

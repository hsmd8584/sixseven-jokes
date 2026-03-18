"""
Robust Structured Output Parser

Handles common LLM output issues when requesting JSON:
- Markdown code block wrapping (```json ... ```)
- Trailing commas
- Partial JSON arrays
- Non-JSON preamble/postamble text
- Escaped characters
- Regex fallback for severely malformed output

This is critical for production AI systems where LLM output
format is never guaranteed to be perfectly structured.
"""

import re
import json
from typing import List, Dict, Any, Optional

from loguru import logger


class StructuredOutputParser:
    """
    Production-grade JSON parser for LLM outputs.

    Unlike a simple json.loads() call, this parser handles the reality
    that LLMs frequently produce almost-valid JSON with minor formatting
    issues that would break naive parsing.
    """

    @staticmethod
    def parse_json_array(text: str) -> List[Dict[str, Any]]:
        """
        Parse a JSON array from LLM output with multiple fallback strategies.

        Strategy chain:
        1. Direct JSON parse (ideal case)
        2. Strip markdown code blocks and retry
        3. Extract JSON array substring and retry
        4. Fix common JSON malformations and retry
        5. Regex-based field extraction (last resort)

        Args:
            text: Raw LLM output text.

        Returns:
            Parsed list of dicts. Returns empty list on total failure.
        """
        if not text or not text.strip():
            return []

        # Strategy 1: Direct parse
        try:
            result = json.loads(text.strip())
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
        except json.JSONDecodeError:
            pass

        # Strategy 2: Strip markdown code blocks
        cleaned = StructuredOutputParser._strip_markdown(text)
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
        except json.JSONDecodeError:
            pass

        # Strategy 3: Extract JSON array substring
        extracted = StructuredOutputParser._extract_json_array(text)
        if extracted:
            try:
                result = json.loads(extracted)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        # Strategy 4: Fix common malformations
        fixed = StructuredOutputParser._fix_common_issues(extracted or cleaned)
        try:
            result = json.loads(fixed)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 5: Regex fallback
        logger.warning("All JSON parse strategies failed, attempting regex extraction")
        return StructuredOutputParser._regex_fallback(text)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove markdown code block wrappers."""
        # Match ```json ... ``` or ``` ... ```
        pattern = r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _extract_json_array(text: str) -> Optional[str]:
        """Extract the first JSON array from text, handling nested brackets."""
        # Find the first '[' and its matching ']'
        start = text.find("[")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        # If we didn't find matching ']', return everything from '['
        return text[start:] + "]"

    @staticmethod
    def _fix_common_issues(text: str) -> str:
        """Fix common JSON malformations from LLM output."""
        # Remove trailing commas before ] or }
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # Fix single quotes to double quotes (but not in contractions)
        # Only do this if there are no double quotes (avoiding mixed quote issues)
        if '"' not in text and "'" in text:
            text = text.replace("'", '"')

        # Remove control characters
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

        # Fix missing commas between objects
        text = re.sub(r"}\s*{", "},{", text)

        return text

    @staticmethod
    def _regex_fallback(text: str) -> List[Dict[str, Any]]:
        """
        Last-resort regex extraction for severely malformed JSON.
        Attempts to extract question/answer pairs using pattern matching.
        """
        results = []

        # Pattern: "question": "...", "answer": "..."
        pattern = r'"question"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer"\s*:\s*"((?:[^"\\]|\\.)*)"'
        matches = re.findall(pattern, text, re.DOTALL)

        for question, answer in matches:
            # Unescape JSON strings
            question = question.replace('\\"', '"').replace("\\n", "\n")
            answer = answer.replace('\\"', '"').replace("\\n", "\n")
            results.append({"question": question, "answer": answer})

        if results:
            logger.info(f"Regex fallback extracted {len(results)} items")

        return results

    @staticmethod
    def parse_single_object(text: str) -> Dict[str, Any]:
        """Parse a single JSON object from LLM output."""
        array = StructuredOutputParser.parse_json_array(f"[{text}]")
        return array[0] if array else {}

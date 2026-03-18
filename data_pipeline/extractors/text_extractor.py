"""
Text Joke Extractor

Parses plain text and markdown files containing jokes in various formats:
- Q: ... / A: ... format
- Setup: ... / Punchline: ... format
- Numbered list format
- Freeform text (falls back to LLM extraction)
"""

import re
from typing import List, Optional, Tuple
from pathlib import Path

import google.generativeai as genai
from loguru import logger

from .base import BaseExtractor, RawJoke


# Regex patterns for common Q&A joke formats
QA_PATTERNS = [
    # Q: ... A: ... format
    re.compile(
        r"Q:\s*(.+?)\s*\n\s*A:\s*(.+?)(?=\n\s*Q:|\n\n|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    # Question: ... Answer: ... format
    re.compile(
        r"Question:\s*(.+?)\s*\n\s*Answer:\s*(.+?)(?=\nQuestion:|\n\n|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    # Setup: ... Punchline: ... format
    re.compile(
        r"Setup:\s*(.+?)\s*\n\s*Punchline:\s*(.+?)(?=\nSetup:|\n\n|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    # "Why did...?" / "Because..." pattern
    re.compile(
        r"((?:Why|What|How|When|Where|Who|Which|Knock)[^\n?]+\?)\s*\n\s*([^\n]+?)(?=\n\n|\n(?:Why|What|How)|$)",
        re.DOTALL | re.IGNORECASE,
    ),
]

LLM_FALLBACK_PROMPT = """Extract all Q&A jokes from the following text.
Each joke should have a clear setup (question) and punchline (answer).

Text:
---
{text}
---

Return a JSON array of objects with keys: "question", "answer", "confidence"
Return ONLY the JSON array, no markdown formatting.
If no jokes found, return: []"""


class TextJokeExtractor(BaseExtractor):
    """
    Extract jokes from text files using a multi-strategy approach:

    Strategy 1: Regex pattern matching for well-formatted Q&A jokes
    Strategy 2: LLM-based extraction as fallback for freeform text

    This two-stage approach maximizes extraction recall while keeping
    cost low (LLM is only called when regex matching yields few results).
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        llm_fallback_threshold: int = 3,
    ):
        super().__init__(source_type="text")
        self.llm_fallback_threshold = llm_fallback_threshold

        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None

    @property
    def supported_extensions(self) -> set:
        return {".txt", ".md", ".csv", ".jsonl"}

    def _read_source(self, path: Path) -> str:
        """Read text file with encoding detection fallback."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode {path} with any supported encoding")

    def _parse_jokes(self, content: str, source_file: str) -> List[RawJoke]:
        """
        Multi-strategy joke extraction:
        1. Try all regex patterns
        2. If yield is below threshold and LLM is available, use LLM fallback
        """
        # Strategy 1: Regex pattern matching
        regex_jokes = self._extract_with_regex(content, source_file)
        logger.info(f"Regex extraction found {len(regex_jokes)} jokes from {source_file}")

        # Strategy 2: LLM fallback if regex yield is low
        if len(regex_jokes) < self.llm_fallback_threshold and self.model:
            logger.info(f"Low regex yield, falling back to LLM extraction")
            llm_jokes = self._extract_with_llm(content, source_file)

            # Merge results, preferring LLM extractions for better quality
            return self._merge_extractions(regex_jokes, llm_jokes)

        return regex_jokes

    def _extract_with_regex(self, content: str, source_file: str) -> List[RawJoke]:
        """Apply all regex patterns and collect unique matches."""
        seen_questions = set()
        jokes = []

        for pattern in QA_PATTERNS:
            matches = pattern.findall(content)
            for match in matches:
                question = match[0].strip()
                answer = match[1].strip()
                q_normalized = question.lower()

                if q_normalized not in seen_questions:
                    seen_questions.add(q_normalized)
                    jokes.append(
                        RawJoke(
                            question=question,
                            answer=answer,
                            source_file=source_file,
                            source_type="text",
                            confidence=0.9,
                        )
                    )
        return jokes

    def _extract_with_llm(self, content: str, source_file: str) -> List[RawJoke]:
        """Use Gemini to extract jokes from freeform text."""
        try:
            # Truncate to avoid token limits
            prompt = LLM_FALLBACK_PROMPT.format(text=content[:6000])

            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )

            from rag.structured_output import StructuredOutputParser

            parsed = StructuredOutputParser.parse_json_array(response.text)

            jokes = []
            for item in parsed:
                if "question" in item and "answer" in item:
                    jokes.append(
                        RawJoke(
                            question=item["question"].strip(),
                            answer=item["answer"].strip(),
                            source_file=source_file,
                            source_type="text",
                            confidence=item.get("confidence", 0.75),
                        )
                    )
            return jokes

        except Exception as e:
            logger.error(f"LLM fallback extraction failed: {e}")
            return []

    def _merge_extractions(
        self, regex_jokes: List[RawJoke], llm_jokes: List[RawJoke]
    ) -> List[RawJoke]:
        """
        Merge regex and LLM extraction results.
        Uses normalized question text to avoid duplicates.
        Prefers LLM extraction when there's overlap (typically better quality).
        """
        seen = {j.question.lower().strip() for j in llm_jokes}
        merged = list(llm_jokes)

        for joke in regex_jokes:
            if joke.question.lower().strip() not in seen:
                merged.append(joke)

        logger.info(
            f"Merged {len(regex_jokes)} regex + {len(llm_jokes)} LLM = {len(merged)} unique jokes"
        )
        return merged

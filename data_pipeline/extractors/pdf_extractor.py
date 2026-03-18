"""
PDF Joke Extractor

Extracts Q&A jokes from PDF joke books using PyMuPDF for text extraction
and Google Gemini for intelligent joke parsing. Handles both structured
(clearly formatted Q/A) and unstructured (narrative) PDF content.
"""

import json
from typing import List
from pathlib import Path

import fitz  # PyMuPDF
import google.generativeai as genai
from loguru import logger

from .base import BaseExtractor, RawJoke


# Prompt for LLM-based joke extraction from PDF text
EXTRACTION_PROMPT = """You are a joke extraction assistant. Given the following text from a joke book page, 
extract all Q&A format jokes. Each joke should have a clear "question" (setup) and "answer" (punchline).

Rules:
- Only extract jokes that have a clear setup and punchline
- Skip any non-joke content (table of contents, author notes, etc.)
- If a joke is a knock-knock joke, format the setup as the full knock-knock exchange
- Preserve the original wording as closely as possible
- Rate your confidence (0.0 to 1.0) for each extraction

Return a JSON array of objects with keys: "question", "answer", "confidence"
If no jokes are found, return an empty array: []

Text from page {page_num}:
---
{text}
---

Return ONLY the JSON array, no markdown formatting."""


class PDFJokeExtractor(BaseExtractor):
    """
    Extract jokes from PDF files using a two-stage approach:
    1. PyMuPDF extracts raw text from each page
    2. Gemini LLM parses the text into structured Q&A jokes

    This handles the diversity of PDF joke book formats where simple
    regex-based parsing would fail on varied layouts and formatting.
    """

    def __init__(self, gemini_api_key: str, model_name: str = "gemini-1.5-flash"):
        super().__init__(source_type="pdf")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.pages_per_batch = 3  # Process multiple pages together for context

    @property
    def supported_extensions(self) -> set:
        return {".pdf"}

    def _read_source(self, path: Path) -> str:
        """Read PDF and return page-separated text."""
        doc = fitz.open(str(path))
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append(f"[PAGE {page_num + 1}]\n{text}")
        doc.close()
        return "\n\n---PAGE_BREAK---\n\n".join(pages)

    def _parse_jokes(self, content: str, source_file: str) -> List[RawJoke]:
        """
        Parse PDF content into jokes using Gemini LLM.

        Processes pages in batches to maintain context while staying
        within token limits. Uses structured prompting to get consistent
        JSON output from the LLM.
        """
        if not content.strip():
            logger.warning(f"Empty content from {source_file}")
            return []

        pages = content.split("---PAGE_BREAK---")
        all_jokes = []

        # Process pages in batches for better context
        for i in range(0, len(pages), self.pages_per_batch):
            batch = pages[i : i + self.pages_per_batch]
            batch_text = "\n".join(batch)
            page_start = i + 1

            try:
                jokes = self._extract_from_page_batch(
                    batch_text, page_start, source_file
                )
                all_jokes.extend(jokes)
                logger.info(
                    f"Extracted {len(jokes)} jokes from pages {page_start}-{page_start + len(batch) - 1}"
                )
            except Exception as e:
                logger.error(f"Failed to extract from pages {page_start}+: {e}")
                continue

        return all_jokes

    def _extract_from_page_batch(
        self, text: str, page_num: int, source_file: str
    ) -> List[RawJoke]:
        """Send a page batch to Gemini and parse the structured response."""
        prompt = EXTRACTION_PROMPT.format(page_num=page_num, text=text[:4000])

        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temp for consistent extraction
                max_output_tokens=2048,
            ),
        )

        return self._parse_llm_response(response.text, source_file, page_num)

    def _parse_llm_response(
        self, response_text: str, source_file: str, page_num: int
    ) -> List[RawJoke]:
        """
        Parse LLM JSON response with robust error handling.

        Handles common LLM output issues:
        - Markdown code block wrapping
        - Trailing commas
        - Partial JSON arrays
        """
        from rag.structured_output import StructuredOutputParser

        parsed = StructuredOutputParser.parse_json_array(response_text)

        jokes = []
        for item in parsed:
            if "question" in item and "answer" in item:
                jokes.append(
                    RawJoke(
                        question=item["question"].strip(),
                        answer=item["answer"].strip(),
                        source_file=source_file,
                        source_type="pdf",
                        page_number=page_num,
                        confidence=item.get("confidence", 0.8),
                    )
                )
        return jokes

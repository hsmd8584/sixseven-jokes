"""
Image Joke Extractor

Uses Google Gemini's multimodal (vision) capability to extract Q&A jokes
from images of joke books, memes, or any image-based joke content.
Handles OCR and semantic understanding in a single LLM call.
"""

import json
import base64
from typing import List
from pathlib import Path

import google.generativeai as genai
from PIL import Image
from loguru import logger

from .base import BaseExtractor, RawJoke


VISION_EXTRACTION_PROMPT = """You are a joke extraction assistant with vision capabilities.
Look at this image from a joke book or joke content and extract all Q&A format jokes you can find.

Rules:
- Extract every joke visible in the image
- Each joke must have a "question" (setup) and "answer" (punchline)
- If text is partially obscured, do your best to reconstruct it
- Rate your confidence (0.0 to 1.0) for each extraction
- If the image doesn't contain jokes, return an empty array

Return a JSON array of objects with keys: "question", "answer", "confidence"
Return ONLY the JSON array, no markdown formatting."""


class ImageJokeExtractor(BaseExtractor):
    """
    Extract jokes from images using Gemini Vision.

    Combines OCR + semantic understanding in one step, which is more
    robust than traditional OCR -> text parsing for joke book images
    with varied layouts, fonts, and decorative elements.
    """

    def __init__(self, gemini_api_key: str, model_name: str = "gemini-1.5-flash"):
        super().__init__(source_type="image")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)

    @property
    def supported_extensions(self) -> set:
        return {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def _read_source(self, path: Path) -> str:
        """
        Read image file and return base64-encoded content.
        Also validates image can be opened by PIL.
        """
        # Validate image integrity
        try:
            img = Image.open(str(path))
            img.verify()
        except Exception as e:
            raise ValueError(f"Invalid image file {path}: {e}")

        with open(str(path), "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _parse_jokes(self, content: str, source_file: str) -> List[RawJoke]:
        """
        Send image to Gemini Vision for joke extraction.

        Uses multimodal input: image + text prompt to get structured
        joke data directly from the visual content.
        """
        try:
            # Load image for Gemini
            image = Image.open(source_file)

            response = self.model.generate_content(
                [VISION_EXTRACTION_PROMPT, image],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )

            return self._parse_response(response.text, source_file)

        except Exception as e:
            logger.error(f"Vision extraction failed for {source_file}: {e}")
            return []

    def _parse_response(self, response_text: str, source_file: str) -> List[RawJoke]:
        """Parse Gemini Vision response into RawJoke objects."""
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
                        source_type="image",
                        confidence=item.get("confidence", 0.7),
                    )
                )

        logger.info(f"Extracted {len(jokes)} jokes from image {source_file}")
        return jokes

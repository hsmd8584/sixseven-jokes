"""
Joke extractors for different source formats.
Each extractor implements the BaseExtractor interface.
"""

from .base import BaseExtractor
from .pdf_extractor import PDFJokeExtractor
from .image_extractor import ImageJokeExtractor
from .text_extractor import TextJokeExtractor

__all__ = [
    "BaseExtractor",
    "PDFJokeExtractor",
    "ImageJokeExtractor",
    "TextJokeExtractor",
]

"""
Base Extractor Interface

Defines the abstract contract for all joke extractors.
Each source format (PDF, Image, Text) implements this interface,
enabling a unified pipeline that processes heterogeneous data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class RawJoke:
    """Represents a single extracted joke before tagging and dedup."""
    question: str
    answer: str
    source_file: str
    source_type: str  # "pdf", "image", "text"
    page_number: Optional[int] = None
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BaseExtractor(ABC):
    """
    Abstract base class for joke extractors.

    Design Pattern: Template Method
    - extract() defines the skeleton: validate -> read -> parse -> filter
    - Subclasses implement _read_source() and _parse_jokes()
    """

    def __init__(self, source_type: str):
        self.source_type = source_type

    def extract(self, file_path: str) -> List[RawJoke]:
        """
        Template method: validate input, read source, parse jokes, filter low-quality.

        Args:
            file_path: Path to the source file.

        Returns:
            List of extracted RawJoke objects.
        """
        path = Path(file_path)
        self._validate(path)
        raw_content = self._read_source(path)
        jokes = self._parse_jokes(raw_content, file_path)
        return self._filter_low_quality(jokes)

    def _validate(self, path: Path) -> None:
        """Validate that the file exists and has a supported extension."""
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        if not path.suffix.lower() in self.supported_extensions:
            raise ValueError(
                f"Unsupported file extension '{path.suffix}' for {self.source_type} extractor. "
                f"Supported: {self.supported_extensions}"
            )

    @property
    @abstractmethod
    def supported_extensions(self) -> set:
        """Set of supported file extensions (e.g., {'.pdf', '.txt'})."""
        ...

    @abstractmethod
    def _read_source(self, path: Path) -> str:
        """Read and return raw content from the source file."""
        ...

    @abstractmethod
    def _parse_jokes(self, content: str, source_file: str) -> List[RawJoke]:
        """Parse raw content into a list of RawJoke objects."""
        ...

    def _filter_low_quality(self, jokes: List[RawJoke], min_q_len: int = 10, min_a_len: int = 3) -> List[RawJoke]:
        """
        Filter out low-quality extractions.

        Removes jokes where:
        - Question is too short (likely extraction noise)
        - Answer is too short
        - Confidence is below threshold
        """
        filtered = []
        for joke in jokes:
            if (
                len(joke.question.strip()) >= min_q_len
                and len(joke.answer.strip()) >= min_a_len
                and joke.confidence >= 0.5
            ):
                filtered.append(joke)
        return filtered

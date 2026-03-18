"""
LLM-Based Joke Tagger

Uses Google Gemini to automatically tag jokes with:
- Age groups (e.g., "3-5", "5-7", "7-9", "9-12", "12-15")
- Themes (e.g., "animals", "school", "food", "science")
- Joke types (e.g., "pun", "knock_knock", "riddle")

Processes jokes in batches for efficiency and cost optimization.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import google.generativeai as genai
from loguru import logger
from tqdm import tqdm

from config import config


TAGGING_PROMPT = """You are a joke content tagger for a children's joke app.
For each joke below, assign:

1. "age_groups": List of appropriate age ranges from: {age_groups}
2. "themes": List of relevant themes from: {themes}
3. "joke_type": One of: {joke_types}
4. "difficulty": One of: "easy", "medium", "hard" (how hard is the joke to understand)
5. "family_friendly": true/false (is this appropriate for children?)

Jokes to tag:
{jokes}

Return a JSON array with one object per joke. Each object must have keys:
"index", "age_groups", "themes", "joke_type", "difficulty", "family_friendly"

Return ONLY the JSON array."""


@dataclass
class TaggedJoke:
    """A joke enriched with LLM-generated metadata tags."""
    question: str
    answer: str
    age_groups: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    joke_type: str = "pun"
    difficulty: str = "easy"
    family_friendly: bool = True
    source_file: str = ""
    source_type: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "age_groups": self.age_groups,
            "themes": self.themes,
            "joke_type": self.joke_type,
            "difficulty": self.difficulty,
            "family_friendly": self.family_friendly,
            "source_file": self.source_file,
            "source_type": self.source_type,
        }


class LLMJokeTagger:
    """
    Batch joke tagger using Gemini LLM.

    Design decisions:
    - Batched processing (default 10 jokes/batch) to minimize API calls
    - Constrained tag vocabulary to maintain dataset consistency
    - Fallback defaults when LLM output is ambiguous
    - Filters out non-family-friendly content automatically
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        batch_size: int = 10,
    ):
        api_key = gemini_api_key or config.gemini.api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.batch_size = batch_size

        # Constrained vocabularies from config
        self.age_groups = config.pipeline.supported_age_groups
        self.themes = config.pipeline.supported_themes
        self.joke_types = config.pipeline.supported_joke_types

    def tag_jokes(self, jokes: List[Dict[str, Any]], filter_unsafe: bool = True) -> List[TaggedJoke]:
        """
        Tag a list of jokes with age groups, themes, and types.

        Args:
            jokes: List of dicts with "question" and "answer" keys.
            filter_unsafe: If True, removes jokes flagged as not family-friendly.

        Returns:
            List of TaggedJoke objects with metadata.
        """
        tagged_results = []

        for i in tqdm(range(0, len(jokes), self.batch_size), desc="Tagging jokes"):
            batch = jokes[i : i + self.batch_size]
            try:
                batch_tags = self._tag_batch(batch)
                tagged_results.extend(batch_tags)
            except Exception as e:
                logger.error(f"Tagging failed for batch {i // self.batch_size}: {e}")
                # Apply default tags as fallback
                tagged_results.extend(self._apply_default_tags(batch))

        if filter_unsafe:
            safe_count = len([j for j in tagged_results if j.family_friendly])
            logger.info(
                f"Safety filter: {safe_count}/{len(tagged_results)} jokes are family-friendly"
            )
            tagged_results = [j for j in tagged_results if j.family_friendly]

        logger.info(f"Tagged {len(tagged_results)} jokes total")
        return tagged_results

    def _tag_batch(self, batch: List[Dict[str, Any]]) -> List[TaggedJoke]:
        """Send a batch of jokes to Gemini for tagging."""
        # Format jokes for the prompt
        jokes_text = ""
        for idx, joke in enumerate(batch):
            jokes_text += f"\n[{idx}] Q: {joke['question']}\n    A: {joke['answer']}\n"

        prompt = TAGGING_PROMPT.format(
            age_groups=", ".join(self.age_groups),
            themes=", ".join(self.themes),
            joke_types=", ".join(self.joke_types),
            jokes=jokes_text,
        )

        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )

        return self._parse_tag_response(response.text, batch)

    def _parse_tag_response(
        self, response_text: str, batch: List[Dict[str, Any]]
    ) -> List[TaggedJoke]:
        """Parse LLM tagging response and merge with original joke data."""
        from rag.structured_output import StructuredOutputParser

        tags_list = StructuredOutputParser.parse_json_array(response_text)

        tagged_jokes = []
        for idx, joke in enumerate(batch):
            # Find matching tags by index
            tags = next(
                (t for t in tags_list if t.get("index") == idx),
                {},
            )

            tagged_jokes.append(
                TaggedJoke(
                    question=joke["question"],
                    answer=joke["answer"],
                    age_groups=self._validate_tags(
                        tags.get("age_groups", ["5-7"]), self.age_groups
                    ),
                    themes=self._validate_tags(
                        tags.get("themes", ["everyday"]), self.themes
                    ),
                    joke_type=self._validate_single(
                        tags.get("joke_type", "pun"), self.joke_types
                    ),
                    difficulty=self._validate_single(
                        tags.get("difficulty", "easy"), ["easy", "medium", "hard"]
                    ),
                    family_friendly=tags.get("family_friendly", True),
                    source_file=joke.get("source_file", ""),
                    source_type=joke.get("source_type", ""),
                )
            )

        return tagged_jokes

    def _validate_tags(self, tags: List[str], valid_set: List[str]) -> List[str]:
        """Filter tags to only include values from the valid vocabulary."""
        validated = [t for t in tags if t in valid_set]
        return validated if validated else [valid_set[0]]

    def _validate_single(self, value: str, valid_set: List[str]) -> str:
        """Validate a single tag value against the valid vocabulary."""
        return value if value in valid_set else valid_set[0]

    def _apply_default_tags(self, batch: List[Dict[str, Any]]) -> List[TaggedJoke]:
        """Apply conservative default tags when LLM tagging fails."""
        return [
            TaggedJoke(
                question=joke["question"],
                answer=joke["answer"],
                age_groups=["5-7"],
                themes=["everyday"],
                joke_type="pun",
                difficulty="easy",
                family_friendly=True,
                source_file=joke.get("source_file", ""),
                source_type=joke.get("source_type", ""),
            )
            for joke in batch
        ]

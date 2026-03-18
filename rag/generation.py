"""
Gemini Joke Generator

Preference-aware joke generation using Google Gemini as fallback
when the curated joke pool doesn't have enough matching content.

Key features:
- Preference-aware prompting: injects liked/disliked examples
- Structured output with robust parsing
- Dedup against existing jokes before returning
- Async write-back to expand the content pool
"""

import json
import hashlib
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from loguru import logger

from .structured_output import StructuredOutputParser
from config import config


GENERATION_PROMPT = """You are a professional children's joke writer creating jokes for a kids' humor app.

Generate {num_jokes} original Q&A jokes following these requirements:

**Audience:**
- Age range: {age_range}
- Scenario/Theme: {scenario}
- MUST be 100% family-friendly and child-appropriate

**Style Guidelines:**
- Each joke must have a clear "question" (setup) and "answer" (punchline)
- Use age-appropriate vocabulary and concepts
- Jokes should be genuinely funny, not just silly
- Avoid any content that could be scary, violent, or inappropriate

{preference_section}

**Output Format:**
Return a JSON array of objects, each with:
- "question": the joke setup
- "answer": the punchline
- "theme": one of [{themes}]

Return ONLY the JSON array, no other text."""

PREFERENCE_SECTION_TEMPLATE = """**User Preferences (generate jokes similar to liked, avoid style of disliked):**

Jokes the user LIKED (generate similar style):
{liked_jokes}

Jokes the user DISLIKED (avoid this style):
{disliked_jokes}
"""


class GeminiJokeGenerator:
    """
    LLM-based joke generator with preference conditioning.

    This is the fallback layer in the retrieval-generation pipeline:
    only called when the curated joke pool doesn't have enough
    matching content for a user request.

    Design decisions:
    - Preference-aware prompting: We inject the user's liked/disliked
      jokes directly into the prompt rather than fine-tuning, because:
      (1) it's zero-latency to update, (2) works with any base model,
      (3) preferences change per-session.
    - Structured output with robust parsing: We never assume the LLM
      will return perfect JSON. Multiple fallback parsing strategies
      ensure reliability.
    - Generated jokes are written back to the database asynchronously,
      expanding the content pool over time.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        api_key = api_key or config.gemini.api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name or config.gemini.model_name
        )
        self.generation_config = genai.GenerationConfig(
            temperature=config.gemini.temperature,
            max_output_tokens=config.gemini.max_output_tokens,
            top_p=config.gemini.top_p,
            top_k=config.gemini.top_k,
        )

    def generate(
        self,
        num_jokes: int,
        age_range: str = "5-7",
        scenario: str = "everyday",
        liked_jokes: Optional[List[Dict]] = None,
        disliked_jokes: Optional[List[Dict]] = None,
        existing_jokes: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate new jokes using Gemini with preference conditioning.

        Args:
            num_jokes: Number of jokes to generate.
            age_range: Target age range.
            scenario: Theme/scenario for the jokes.
            liked_jokes: User's liked jokes (for style conditioning).
            disliked_jokes: User's disliked jokes (for negative conditioning).
            existing_jokes: Existing jokes to dedup against.

        Returns:
            List of generated joke dicts with "question", "answer", "theme".
        """
        # Build preference-aware prompt
        prompt = self._build_prompt(
            num_jokes=num_jokes,
            age_range=age_range,
            scenario=scenario,
            liked_jokes=liked_jokes or [],
            disliked_jokes=disliked_jokes or [],
        )

        # Generate with retry logic
        generated = self._generate_with_retry(prompt, max_retries=2)

        # Deduplicate against existing content
        if existing_jokes:
            generated = self._dedup_against_existing(generated, existing_jokes)

        # Add metadata
        for joke in generated:
            joke["age_range"] = age_range
            joke["scenario"] = scenario
            joke["source"] = "gemini_generated"
            joke["id"] = self._generate_id(joke)

        logger.info(f"Generated {len(generated)} new jokes for {scenario}/{age_range}")
        return generated

    def _build_prompt(
        self,
        num_jokes: int,
        age_range: str,
        scenario: str,
        liked_jokes: List[Dict],
        disliked_jokes: List[Dict],
    ) -> str:
        """
        Build a preference-aware generation prompt.

        Injects the user's liked and disliked jokes as in-context examples
        so the model adapts its style to match user preferences without
        requiring any fine-tuning.
        """
        # Build preference section if user has history
        preference_section = ""
        if liked_jokes or disliked_jokes:
            liked_text = "\n".join(
                f'  - Q: {j["question"]} A: {j["answer"]}'
                for j in liked_jokes[:5]  # Limit to top 5 to save tokens
            ) or "  (none)"

            disliked_text = "\n".join(
                f'  - Q: {j["question"]} A: {j["answer"]}'
                for j in disliked_jokes[:3]
            ) or "  (none)"

            preference_section = PREFERENCE_SECTION_TEMPLATE.format(
                liked_jokes=liked_text,
                disliked_jokes=disliked_text,
            )

        return GENERATION_PROMPT.format(
            num_jokes=num_jokes,
            age_range=age_range,
            scenario=scenario,
            preference_section=preference_section,
            themes=", ".join(config.pipeline.supported_themes),
        )

    def _generate_with_retry(
        self, prompt: str, max_retries: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Call Gemini with retry logic and robust output parsing.

        Retries on:
        - Empty response
        - Unparseable output (after all parsing fallbacks)
        - API errors
        """
        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                )

                if not response.text:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    continue

                # Parse with robust structured output handler
                jokes = StructuredOutputParser.parse_json_array(response.text)

                if jokes:
                    # Validate structure
                    valid_jokes = [
                        j for j in jokes
                        if "question" in j and "answer" in j
                    ]
                    if valid_jokes:
                        return valid_jokes

                logger.warning(
                    f"No valid jokes parsed on attempt {attempt + 1}"
                )

            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    raise

        return []

    def _dedup_against_existing(
        self,
        generated: List[Dict],
        existing: List[Dict],
    ) -> List[Dict]:
        """
        Remove generated jokes that are too similar to existing ones.
        Uses normalized text hashing for fast comparison.
        """
        import re

        def normalize(text: str) -> str:
            text = text.lower().strip()
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text)
            return text

        existing_hashes = {
            hashlib.md5(
                normalize(f"{j['question']} {j['answer']}").encode()
            ).hexdigest()
            for j in existing
        }

        unique = []
        for joke in generated:
            h = hashlib.md5(
                normalize(f"{joke['question']} {joke['answer']}").encode()
            ).hexdigest()
            if h not in existing_hashes:
                unique.append(joke)
            else:
                logger.debug(f"Deduped generated joke: {joke['question'][:50]}...")

        return unique

    @staticmethod
    def _generate_id(joke: Dict) -> str:
        """Generate a deterministic ID for a joke based on content."""
        content = f"{joke['question']}|{joke['answer']}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

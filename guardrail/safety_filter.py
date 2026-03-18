"""
Content Safety Filter

Lightweight "watchdog" model for filtering generated joke content
before serving to children. Uses a tiered approach:

Tier 1: Rule-based keyword/pattern matching (fast, ~0ms)
Tier 2: LLM-based content assessment (slower, ~200ms, higher accuracy)

Design principle: This is a post-generation guardrail, not a content
moderation system. It catches obvious safety issues in newly generated
content while keeping latency low for the common case (safe content).

Position in the system:
  Gemini Generation → Safety Filter → User Delivery
                          ↓ (if flagged)
                     Block + Log
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from loguru import logger

from config import config


@dataclass
class SafetyResult:
    """Result of a safety check."""
    is_safe: bool
    confidence: float
    flagged_categories: List[str]
    tier: str  # "rule_based" or "llm_based"
    details: str = ""


# Rule-based patterns for obvious violations
BLOCKED_PATTERNS = [
    # Violence
    (re.compile(r"\b(kill|murder|blood|gore|weapon|gun|stab)\b", re.I), "violence"),
    # Sexual content
    (re.compile(r"\b(sex|nude|naked|porn)\b", re.I), "sexual"),
    # Hate speech
    (re.compile(r"\b(hate|racist|bigot|slur)\b", re.I), "hate_speech"),
    # Self-harm
    (re.compile(r"\b(suicide|self.harm|cut yourself)\b", re.I), "self_harm"),
    # Dangerous activities
    (re.compile(r"\b(drug|cocaine|heroin|meth)\b", re.I), "dangerous"),
]

# Patterns that are suspicious but need context (LLM check)
SUSPICIOUS_PATTERNS = [
    re.compile(r"\b(die|dead|death|hurt)\b", re.I),
    re.compile(r"\b(stupid|dumb|idiot|ugly)\b", re.I),
    re.compile(r"\b(scary|horror|nightmare)\b", re.I),
]

LLM_SAFETY_PROMPT = """You are a content safety reviewer for a children's joke app (ages 3-15).

Evaluate whether this joke is appropriate for children:
Q: {question}
A: {answer}

Check for:
1. Violence or harm references
2. Sexual content or innuendo
3. Discriminatory language
4. Scary or anxiety-inducing content
5. Inappropriate adult humor
6. Bullying or mean-spirited content

Respond with JSON:
{{"is_safe": true/false, "confidence": 0.0-1.0, "flagged_categories": [], "reason": ""}}

Return ONLY the JSON object."""


class SafetyFilter:
    """
    Two-tier content safety filter.

    Processing flow:
    1. Rule-based check (all jokes, ~0ms overhead)
       - If BLOCKED pattern found → reject immediately
       - If SUSPICIOUS pattern found → escalate to Tier 2
       - If clean → approve
    2. LLM-based check (only suspicious content, ~200ms)
       - More nuanced understanding of context
       - Can distinguish "Why did the chicken die?" (benign joke)
         from actually harmful content

    Usage:
        filter = SafetyFilter(gemini_api_key="...")
        result = filter.check_joke("Why was six afraid of seven?", "Because seven ate nine!")
        if result.is_safe:
            serve_to_user(joke)
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        enable_llm_tier: bool = True,
    ):
        self.enable_llm_tier = enable_llm_tier
        self._model = None

        if enable_llm_tier and (gemini_api_key or config.gemini.api_key):
            import google.generativeai as genai
            api_key = gemini_api_key or config.gemini.api_key
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(config.guardrail.model_name)

    def check_joke(self, question: str, answer: str) -> SafetyResult:
        """
        Check a single joke for safety.

        Args:
            question: Joke setup text.
            answer: Joke punchline text.

        Returns:
            SafetyResult with is_safe flag and details.
        """
        full_text = f"{question} {answer}"

        # Tier 1: Rule-based check
        blocked_categories = self._check_blocked_patterns(full_text)
        if blocked_categories:
            logger.warning(
                f"Safety BLOCKED (rule-based): {blocked_categories} - "
                f"Q: {question[:50]}..."
            )
            return SafetyResult(
                is_safe=False,
                confidence=0.95,
                flagged_categories=blocked_categories,
                tier="rule_based",
                details="Matched blocked content patterns",
            )

        # Check if suspicious patterns warrant LLM review
        is_suspicious = self._check_suspicious_patterns(full_text)

        if is_suspicious and self._model and self.enable_llm_tier:
            # Tier 2: LLM-based check
            return self._llm_safety_check(question, answer)

        # Clean
        return SafetyResult(
            is_safe=True,
            confidence=0.9 if not is_suspicious else 0.7,
            flagged_categories=[],
            tier="rule_based",
        )

    def check_batch(self, jokes: List[Dict]) -> List[Tuple[Dict, SafetyResult]]:
        """
        Check a batch of jokes. Returns list of (joke, safety_result) tuples.
        """
        results = []
        for joke in jokes:
            result = self.check_joke(
                joke.get("question", ""),
                joke.get("answer", ""),
            )
            results.append((joke, result))

        safe_count = sum(1 for _, r in results if r.is_safe)
        logger.info(f"Safety batch check: {safe_count}/{len(jokes)} jokes passed")

        return results

    def filter_safe(self, jokes: List[Dict]) -> List[Dict]:
        """
        Filter a list of jokes, returning only safe ones.
        Convenience method for pipeline integration.
        """
        results = self.check_batch(jokes)
        return [joke for joke, result in results if result.is_safe]

    def _check_blocked_patterns(self, text: str) -> List[str]:
        """Check text against blocked pattern rules."""
        flagged = []
        for pattern, category in BLOCKED_PATTERNS:
            if pattern.search(text):
                flagged.append(category)
        return list(set(flagged))

    def _check_suspicious_patterns(self, text: str) -> bool:
        """Check if text contains suspicious patterns needing LLM review."""
        return any(pattern.search(text) for pattern in SUSPICIOUS_PATTERNS)

    def _llm_safety_check(self, question: str, answer: str) -> SafetyResult:
        """Use Gemini to assess content safety with nuanced understanding."""
        try:
            prompt = LLM_SAFETY_PROMPT.format(question=question, answer=answer)

            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 256,
                },
            )

            from rag.structured_output import StructuredOutputParser
            parsed = StructuredOutputParser.parse_single_object(response.text)

            is_safe = parsed.get("is_safe", True)
            confidence = parsed.get("confidence", 0.5)
            categories = parsed.get("flagged_categories", [])
            reason = parsed.get("reason", "")

            if not is_safe:
                logger.warning(
                    f"Safety BLOCKED (LLM): {categories} - {reason} - "
                    f"Q: {question[:50]}..."
                )

            return SafetyResult(
                is_safe=is_safe,
                confidence=confidence,
                flagged_categories=categories,
                tier="llm_based",
                details=reason,
            )

        except Exception as e:
            logger.error(f"LLM safety check failed: {e}")
            # Fail-open: if LLM check fails, default to safe
            # (rule-based already passed at this point)
            return SafetyResult(
                is_safe=True,
                confidence=0.5,
                flagged_categories=[],
                tier="llm_based",
                details=f"LLM check failed, defaulting to safe: {e}",
            )

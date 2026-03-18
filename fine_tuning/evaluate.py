"""
Joke Generation Evaluation

Metrics for evaluating joke generation quality:
1. Format compliance: Does the output have proper Q/A structure?
2. Theme consistency: Does the joke match the requested theme?
3. Age appropriateness: Is the vocabulary suitable for the target age?
4. Diversity: How diverse are generated jokes vs. training data?
5. Preference alignment: Do generated jokes align with liked examples?

Note: Evaluating "humor" is inherently subjective. Our approach starts
with measurable structural/safety metrics and uses LLM-as-judge for
subjective quality when needed.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
from loguru import logger


class JokeEvaluator:
    """
    Multi-dimensional joke quality evaluator.

    Evaluation philosophy:
    - Start with objective, automatable metrics
    - Use LLM-as-judge for subjective quality assessment
    - Track aggregate statistics, not individual joke scores
    - Focus on what's actionable for the system
    """

    def __init__(self, gemini_model=None):
        """
        Args:
            gemini_model: Optional Gemini model for LLM-based evaluation.
                          If None, only structural metrics are computed.
        """
        self.gemini_model = gemini_model

    def evaluate(
        self,
        generated_jokes: List[Dict],
        reference_jokes: Optional[List[Dict]] = None,
        target_theme: Optional[str] = None,
        target_age_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite on generated jokes.

        Returns a dict of metric names to values.
        """
        metrics = {}

        # 1. Format compliance
        metrics["format_compliance"] = self._eval_format_compliance(generated_jokes)

        # 2. Length statistics
        metrics["length_stats"] = self._eval_length_stats(generated_jokes)

        # 3. Diversity metrics
        metrics["diversity"] = self._eval_diversity(generated_jokes)

        # 4. Self-repetition (within batch)
        metrics["self_repetition_rate"] = self._eval_self_repetition(generated_jokes)

        # 5. Training data overlap (if reference provided)
        if reference_jokes:
            metrics["training_overlap"] = self._eval_training_overlap(
                generated_jokes, reference_jokes
            )

        # 6. LLM-based quality (if model available)
        if self.gemini_model:
            metrics["llm_quality"] = self._eval_llm_quality(
                generated_jokes, target_theme, target_age_range
            )

        return metrics

    def _eval_format_compliance(self, jokes: List[Dict]) -> Dict[str, float]:
        """
        Check structural quality:
        - Has both Q and A fields
        - Q ends with '?'
        - Q and A are non-trivially different
        - No empty fields
        """
        if not jokes:
            return {"rate": 0.0, "details": {}}

        checks = {
            "has_question": 0,
            "has_answer": 0,
            "question_ends_with_mark": 0,
            "q_a_different": 0,
            "reasonable_length": 0,
        }

        for joke in jokes:
            q = joke.get("question", "").strip()
            a = joke.get("answer", "").strip()

            if q:
                checks["has_question"] += 1
            if a:
                checks["has_answer"] += 1
            if q.endswith("?"):
                checks["question_ends_with_mark"] += 1
            if q and a and q.lower() != a.lower():
                checks["q_a_different"] += 1
            if 10 <= len(q) <= 200 and 3 <= len(a) <= 200:
                checks["reasonable_length"] += 1

        n = len(jokes)
        rates = {k: v / n for k, v in checks.items()}
        overall = sum(rates.values()) / len(rates)

        return {"overall_rate": round(overall, 3), "details": rates}

    def _eval_length_stats(self, jokes: List[Dict]) -> Dict[str, float]:
        """Compute length statistics for questions and answers."""
        q_lens = [len(j.get("question", "")) for j in jokes]
        a_lens = [len(j.get("answer", "")) for j in jokes]

        return {
            "question_length": {
                "mean": round(np.mean(q_lens), 1) if q_lens else 0,
                "std": round(np.std(q_lens), 1) if q_lens else 0,
                "min": min(q_lens) if q_lens else 0,
                "max": max(q_lens) if q_lens else 0,
            },
            "answer_length": {
                "mean": round(np.mean(a_lens), 1) if a_lens else 0,
                "std": round(np.std(a_lens), 1) if a_lens else 0,
                "min": min(a_lens) if a_lens else 0,
                "max": max(a_lens) if a_lens else 0,
            },
        }

    def _eval_diversity(self, jokes: List[Dict]) -> Dict[str, float]:
        """
        Measure lexical diversity of generated jokes.
        - Unique unigrams ratio
        - Unique bigrams ratio
        - Theme distribution entropy
        """
        all_words = []
        all_bigrams = []

        for joke in jokes:
            text = f"{joke.get('question', '')} {joke.get('answer', '')}"
            words = re.findall(r"\w+", text.lower())
            all_words.extend(words)
            all_bigrams.extend(zip(words, words[1:]))

        unique_unigram_ratio = (
            len(set(all_words)) / len(all_words) if all_words else 0
        )
        unique_bigram_ratio = (
            len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
        )

        # Theme distribution entropy
        themes = [j.get("theme", "unknown") for j in jokes]
        theme_counts = Counter(themes)
        total = sum(theme_counts.values())
        theme_entropy = -sum(
            (c / total) * np.log2(c / total) for c in theme_counts.values() if c > 0
        )

        return {
            "unique_unigram_ratio": round(unique_unigram_ratio, 3),
            "unique_bigram_ratio": round(unique_bigram_ratio, 3),
            "theme_entropy": round(theme_entropy, 3),
            "num_unique_themes": len(theme_counts),
        }

    def _eval_self_repetition(self, jokes: List[Dict]) -> float:
        """
        Check for repeated jokes within the same generation batch.
        Returns the fraction of duplicates.
        """
        if len(jokes) <= 1:
            return 0.0

        seen = set()
        duplicates = 0
        for joke in jokes:
            key = (
                joke.get("question", "").lower().strip(),
                joke.get("answer", "").lower().strip(),
            )
            if key in seen:
                duplicates += 1
            seen.add(key)

        return round(duplicates / len(jokes), 3)

    def _eval_training_overlap(
        self, generated: List[Dict], reference: List[Dict]
    ) -> Dict[str, float]:
        """
        Check how many generated jokes are exact copies from training data.
        High overlap suggests memorization rather than generalization.
        """
        ref_set = set()
        for j in reference:
            key = (
                j.get("question", "").lower().strip(),
                j.get("answer", "").lower().strip(),
            )
            ref_set.add(key)

        exact_matches = 0
        for j in generated:
            key = (
                j.get("question", "").lower().strip(),
                j.get("answer", "").lower().strip(),
            )
            if key in ref_set:
                exact_matches += 1

        return {
            "exact_copy_rate": round(exact_matches / len(generated), 3) if generated else 0.0,
            "exact_copies": exact_matches,
            "total_generated": len(generated),
        }

    def _eval_llm_quality(
        self,
        jokes: List[Dict],
        target_theme: Optional[str],
        target_age_range: Optional[str],
    ) -> Dict[str, float]:
        """
        Use Gemini as a judge to evaluate joke quality.
        Scores: humor (1-5), appropriateness (1-5), theme relevance (1-5).
        """
        if not self.gemini_model or not jokes:
            return {}

        import google.generativeai as genai

        jokes_text = "\n".join(
            f"{i+1}. Q: {j['question']} A: {j['answer']}"
            for i, j in enumerate(jokes[:20])  # Limit to 20 for cost
        )

        prompt = f"""Rate each joke on three dimensions (1-5 scale):
1. humor: How funny is it?
2. appropriateness: Is it suitable for children {target_age_range or '5-12'}?
3. theme_relevance: Does it match the theme "{target_theme or 'general'}"?

Jokes:
{jokes_text}

Return a JSON array with objects having keys: "index", "humor", "appropriateness", "theme_relevance"
Return ONLY the JSON array."""

        try:
            response = self.gemini_model.generate_content(prompt)
            from rag.structured_output import StructuredOutputParser
            scores = StructuredOutputParser.parse_json_array(response.text)

            if scores:
                avg_humor = np.mean([s.get("humor", 3) for s in scores])
                avg_approp = np.mean([s.get("appropriateness", 3) for s in scores])
                avg_theme = np.mean([s.get("theme_relevance", 3) for s in scores])

                return {
                    "avg_humor": round(avg_humor, 2),
                    "avg_appropriateness": round(avg_approp, 2),
                    "avg_theme_relevance": round(avg_theme, 2),
                    "jokes_evaluated": len(scores),
                }
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")

        return {}

    def print_report(self, metrics: Dict[str, Any]) -> None:
        """Pretty-print evaluation results."""
        print("\n" + "=" * 60)
        print("  JOKE GENERATION EVALUATION REPORT")
        print("=" * 60)

        for category, values in metrics.items():
            print(f"\n  {category.upper().replace('_', ' ')}:")
            if isinstance(values, dict):
                for k, v in values.items():
                    if isinstance(v, dict):
                        print(f"    {k}:")
                        for kk, vv in v.items():
                            print(f"      {kk}: {vv}")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"    {values}")

        print("\n" + "=" * 60)

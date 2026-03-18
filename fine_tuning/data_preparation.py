"""
Fine-tuning Data Preparation

Converts the curated joke dataset into instruction-tuning format
suitable for LoRA fine-tuning. Creates train/val/test splits with
stratification by age group and theme.

Dataset format (instruction-tuning):
{
    "instruction": "Generate a funny pun joke about animals for kids age 5-7",
    "input": "",
    "output": "Q: What do you call a bear with no teeth?\nA: A gummy bear!"
}
"""

import json
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict

from loguru import logger

from config import config


# Instruction templates for diversity
INSTRUCTION_TEMPLATES = [
    "Generate a funny {joke_type} joke about {theme} for kids age {age_range}.",
    "Write a {joke_type} joke on the topic of {theme} suitable for {age_range} year olds.",
    "Create a child-friendly {joke_type} about {theme} for the {age_range} age group.",
    "Come up with a {difficulty} {joke_type} joke about {theme} for children aged {age_range}.",
    "Tell me a {joke_type} about {theme} that a {age_range} year old would find funny.",
]

# Preference-conditioned instruction templates
PREFERENCE_TEMPLATES = [
    "Generate a joke about {theme} for kids age {age_range}. "
    "The user likes jokes like: '{liked_example}'. Avoid styles like: '{disliked_example}'.",
    "Write a {theme} joke for {age_range} year olds. "
    "Similar style to: '{liked_example}'. Different from: '{disliked_example}'.",
]


class JokeDatasetBuilder:
    """
    Builds instruction-tuning datasets from curated joke data.

    Creates three types of training examples:
    1. Basic generation: instruction → joke
    2. Preference-conditioned: instruction + liked/disliked → joke
    3. Scenario-specific: detailed scenario description → joke

    This diversity in training data helps the fine-tuned model handle
    various request types at inference time.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def build_dataset(
        self,
        jokes: List[Dict[str, Any]],
        include_preference_examples: bool = True,
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
    ) -> Dict[str, List[Dict]]:
        """
        Build instruction-tuning dataset with train/val/test splits.

        Args:
            jokes: List of tagged joke dicts from the data pipeline.
            include_preference_examples: Whether to generate preference-
                conditioned training examples.
            train_ratio: Fraction of data for training.
            val_ratio: Fraction of data for validation.

        Returns:
            Dict with "train", "val", "test" keys, each containing
            a list of instruction-tuning examples.
        """
        logger.info(f"Building dataset from {len(jokes)} jokes")

        # Generate instruction-tuning examples
        examples = []
        for joke in jokes:
            examples.extend(self._create_basic_examples(joke))

        if include_preference_examples:
            pref_examples = self._create_preference_examples(jokes)
            examples.extend(pref_examples)
            logger.info(f"Added {len(pref_examples)} preference-conditioned examples")

        # Shuffle
        random.shuffle(examples)

        # Split
        splits = self._stratified_split(examples, train_ratio, val_ratio)

        logger.info(
            f"Dataset built: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

        return splits

    def _create_basic_examples(self, joke: Dict) -> List[Dict]:
        """Create basic instruction-tuning examples for a single joke."""
        examples = []

        age_groups = joke.get("age_groups", ["5-7"])
        themes = joke.get("themes", ["everyday"])
        joke_type = joke.get("joke_type", "pun")
        difficulty = joke.get("difficulty", "easy")

        # Format the joke output
        output = f"Q: {joke['question']}\nA: {joke['answer']}"

        # Create examples for each age group × theme combination
        for age_range in age_groups:
            for theme in themes:
                # Pick a random instruction template
                template = random.choice(INSTRUCTION_TEMPLATES)
                instruction = template.format(
                    joke_type=joke_type,
                    theme=theme,
                    age_range=age_range,
                    difficulty=difficulty,
                )

                examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "age_range": age_range,
                    "theme": theme,
                })

        return examples

    def _create_preference_examples(
        self, jokes: List[Dict], num_examples: int = 500
    ) -> List[Dict]:
        """
        Create preference-conditioned examples.

        Simulates the scenario where a user has liked/disliked jokes
        and we want to generate jokes matching their taste.
        """
        if len(jokes) < 10:
            return []

        examples = []

        for _ in range(min(num_examples, len(jokes) * 2)):
            # Randomly select a "target" joke
            target = random.choice(jokes)
            output = f"Q: {target['question']}\nA: {target['answer']}"

            # Select a "liked" example (same theme as target)
            same_theme = [
                j for j in jokes
                if j != target
                and set(j.get("themes", [])) & set(target.get("themes", []))
            ]
            liked = random.choice(same_theme) if same_theme else random.choice(jokes)
            liked_str = f"Q: {liked['question']} A: {liked['answer']}"

            # Select a "disliked" example (different theme)
            diff_theme = [
                j for j in jokes
                if j != target
                and not (set(j.get("themes", [])) & set(target.get("themes", [])))
            ]
            disliked = random.choice(diff_theme) if diff_theme else random.choice(jokes)
            disliked_str = f"Q: {disliked['question']} A: {disliked['answer']}"

            age_range = random.choice(target.get("age_groups", ["5-7"]))
            theme = random.choice(target.get("themes", ["everyday"]))

            template = random.choice(PREFERENCE_TEMPLATES)
            instruction = template.format(
                theme=theme,
                age_range=age_range,
                liked_example=liked_str[:100],
                disliked_example=disliked_str[:100],
            )

            examples.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "age_range": age_range,
                "theme": theme,
            })

        return examples

    def _stratified_split(
        self,
        examples: List[Dict],
        train_ratio: float,
        val_ratio: float,
    ) -> Dict[str, List[Dict]]:
        """
        Split dataset with approximate stratification by theme.
        Ensures each split has a representative distribution of themes.
        """
        # Group by theme
        by_theme = defaultdict(list)
        for ex in examples:
            theme = ex.get("theme", "everyday")
            by_theme[theme].append(ex)

        train, val, test = [], [], []

        for theme, theme_examples in by_theme.items():
            random.shuffle(theme_examples)
            n = len(theme_examples)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train.extend(theme_examples[:n_train])
            val.extend(theme_examples[n_train : n_train + n_val])
            test.extend(theme_examples[n_train + n_val :])

        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        return {"train": train, "val": val, "test": test}

    @staticmethod
    def save_dataset(splits: Dict[str, List[Dict]], output_dir: str) -> None:
        """Save dataset splits to JSONL files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, examples in splits.items():
            filepath = output_path / f"{split_name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")

            logger.info(f"Saved {len(examples)} examples to {filepath}")

    @staticmethod
    def format_for_chat(example: Dict) -> str:
        """Format an example as a chat-style training string."""
        return (
            f"<|user|>\n{example['instruction']}\n"
            f"<|assistant|>\n{example['output']}\n"
        )

"""
SixSeven Jokes - Fine-tuning Module

LoRA/QLoRA fine-tuning pipeline for training a custom joke generation model:
- Data preparation from curated joke dataset
- Parameter-efficient fine-tuning with LoRA
- Evaluation metrics for joke quality
"""

from .data_preparation import JokeDatasetBuilder
from .train import JokeFineTuner
from .evaluate import JokeEvaluator

__all__ = ["JokeDatasetBuilder", "JokeFineTuner", "JokeEvaluator"]

"""
SixSeven Jokes - Content Safety Guardrail

Lightweight post-generation safety filter for child-appropriateness.
Sits between the generation layer and the user, ensuring all content
meets family-friendly standards before delivery.
"""

from .safety_filter import SafetyFilter

__all__ = ["SafetyFilter"]

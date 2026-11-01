"""v1_support: CardioSentry V1 ECG-image generation support package.

Implements PLAN.md v1.1. The notebook (v1_image_generation.ipynb) is a thin
driver over these modules so every step is reviewable, testable, and reusable.
"""
from .config import Config, default_config  # noqa: F401

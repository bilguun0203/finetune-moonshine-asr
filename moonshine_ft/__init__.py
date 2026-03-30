"""
Moonshine Fine-Tuning Package

First implementation of fine-tuning for the Moonshine ASR model using curriculum learning.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .curriculum import CurriculumPhase, CurriculumScheduler
from .data_loader import MoonshineDataLoader

__all__ = [
    "MoonshineDataLoader",
    "CurriculumScheduler",
    "CurriculumPhase",
]

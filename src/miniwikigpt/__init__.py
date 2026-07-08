"""MiniWikiGPT package."""

from .config import GPTConfig, TrainConfig, ProjectConfig
from .model import GPT

__all__ = ["GPTConfig", "TrainConfig", "ProjectConfig", "GPT"]

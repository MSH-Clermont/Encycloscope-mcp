"""Core components for Encycloscope MCP."""

from .config import get_settings, Settings
from .corpus import EncyclopediaCorpus

__all__ = ["get_settings", "Settings", "EncyclopediaCorpus"]

"""信号抽取层：词典基线与 LLM 抽取器。"""

from __future__ import annotations

from finpit.signals.base import Extractor, NewsItem, Signal, anonymize
from finpit.signals.lexicon import LexiconExtractor
from finpit.signals.llm import LLMExtractor

__all__ = [
    "Extractor",
    "LLMExtractor",
    "LexiconExtractor",
    "NewsItem",
    "Signal",
    "anonymize",
]

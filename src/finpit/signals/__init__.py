"""信号抽取层：词典基线、单构念 LLM、多构念 LLM。"""

from __future__ import annotations

from finpit.signals.base import Extractor, NewsItem, Signal, anonymize
from finpit.signals.lexicon import LexiconExtractor
from finpit.signals.llm import LLMExtractor
from finpit.signals.multi import (
    CONSTRUCT_SETS,
    FINANCIAL_CONSTRUCTS,
    ORTHOGONAL_CONSTRUCTS,
    ConstructSpec,
    MultiConstructExtractor,
)

__all__ = [
    "CONSTRUCT_SETS",
    "FINANCIAL_CONSTRUCTS",
    "ORTHOGONAL_CONSTRUCTS",
    "ConstructSpec",
    "Extractor",
    "LLMExtractor",
    "LexiconExtractor",
    "MultiConstructExtractor",
    "NewsItem",
    "Signal",
    "anonymize",
]

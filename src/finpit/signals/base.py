"""信号层的基础类型与协议。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from finpit.pit.store import ensure_utc


@dataclass(frozen=True, slots=True)
class NewsItem:
    """一条待打分的文本。``published_at`` 决定它在哪个时点之后才可见。"""

    asset: str
    text: str
    published_at: datetime
    raw: dict | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "published_at", ensure_utc(self.published_at))


@dataclass(frozen=True, slots=True)
class Signal:
    """一次抽取产出的信号。

    Attributes:
        asset: 标的。
        score: 情绪/方向得分，约定落在 ``[-1, 1]``。
        confidence: 置信度 ``[0, 1]``；无置信度概念的抽取器给 1.0。
        source: 抽取器标识，例如 ``lexicon-v1``、``llm:deepseek-v4-pro``。
        rationale: 可读的判据，便于人工审查与审计。
    """

    asset: str
    score: float
    confidence: float = 1.0
    source: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", max(-1.0, min(1.0, float(self.score))))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))

    @property
    def weighted(self) -> float:
        return self.score * self.confidence


@runtime_checkable
class Extractor(Protocol):
    """把文本批量映射成信号的抽取器。"""

    name: str

    def score_batch(self, items: list[NewsItem]) -> list[Signal]:
        """对一批文本打分，返回顺序与输入一致。"""
        ...


# --------------------------------------------------------------- 匿名化工具
_ASSET_RE = re.compile(r"\b(?:ASSET|STOCK|TICKER)\d+\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?\b")


def anonymize(text: str) -> str:
    """抹掉文本中的标的代码与日期。

    ## 为什么需要它

    如果直接把 "2024-03-05 ASSET017 下调业绩指引" 喂给模型，模型可能不是在做
    情绪判断，而是在**回忆**这个标的在那天的真实走势 —— 它早在预训练语料里
    见过。这是一种隐蔽的训练截止污染。

    抹掉标识符后，模型只能依据文本语义作答，"记忆作弊"的通道被堵上。
    在评测里对照 ``anonymize=True/False`` 的 IC 差异，本身就是一项
    污染诊断指标。
    """
    stripped = _ASSET_RE.sub("该公司", text)
    stripped = _DATE_RE.sub("近期", stripped)
    return stripped


__all__ = ["Extractor", "NewsItem", "Signal", "anonymize"]

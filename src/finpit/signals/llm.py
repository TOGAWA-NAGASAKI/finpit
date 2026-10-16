"""基于大模型的金融文本信号抽取。

## 三个方法论点

1. **匿名化**（``anonymize=True``）：抹去标的与日期，堵住"模型靠记忆作答"
   的通道。对照开关前后的 IC 差异，即是一项训练截止污染诊断。
2. **强制结构化输出**：要求返回 JSON 数组，解析失败即计入 ``parse_failures``，
   而不是悄悄退化成 0 分 —— 静默降级会伪造出"模型很稳"的假象。
3. **批量打分**：一次调用评多条，显著降低额度消耗；配合
   :class:`~finpit.llm.LLMClient` 的磁盘缓存，重跑评测零成本。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from finpit.llm.client import LLMClient, LLMError
from finpit.signals.base import NewsItem, Signal, anonymize as anonymize_text

log = logging.getLogger(__name__)

DEFAULT_SYSTEM = (
    "你是一名严谨的金融文本分析师。你的任务是判断给定财经新闻对该公司"
    "**次日股价**的方向性影响。\n"
    "规则：\n"
    "1. score 为 -1 到 1 之间的浮点数，-1 表示极强利空，+1 表示极强利好，0 表示中性。\n"
    "2. confidence 为 0 到 1 之间的浮点数，表示你对该判断的把握程度。\n"
    "3. 只依据文本字面语义判断。如果文本信息不足以判断方向，score 取 0。\n"
    "4. 严格输出 JSON 数组，不要输出任何解释、前言或 markdown 代码块。"
)

DEFAULT_TEMPLATE = (
    "请对下列 {n} 条财经新闻逐条打分。\n\n"
    "{items}\n\n"
    "输出格式（长度必须为 {n} 的 JSON 数组）：\n"
    '[{{"score": 0.0, "confidence": 0.0}}, ...]'
)

# 抓取 ```json ... ``` 或裸 JSON 数组
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(slots=True)
class LLMExtractor:
    """用 LLM 给新闻文本打情绪/方向分。"""

    client: LLMClient
    anonymize: bool = True
    batch_size: int = 8
    max_tokens: int = 2048
    system: str = DEFAULT_SYSTEM
    template: str = DEFAULT_TEMPLATE
    name: str = ""
    parse_failures: int = field(default=0, init=False)
    items_scored: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            mode = "anon" if self.anonymize else "raw"
            self.name = f"llm:{self.client.settings.model}:{mode}"

    # ------------------------------------------------------------- 公开接口
    def score_batch(self, items: list[NewsItem]) -> list[Signal]:
        """按 ``batch_size`` 分批打分，保持与输入等长且顺序一致。"""
        signals: list[Signal] = []
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            signals.extend(self._score_chunk(chunk))
        self.items_scored += len(signals)
        return signals

    # --------------------------------------------------------------- 内部
    def _score_chunk(self, chunk: list[NewsItem]) -> list[Signal]:
        if not chunk:
            return []

        lines = []
        for i, item in enumerate(chunk, start=1):
            text = anonymize_text(item.text) if self.anonymize else item.text
            lines.append(f"{i}. {text}")
        prompt = self.template.format(n=len(chunk), items="\n".join(lines))

        try:
            completion = self.client.complete(
                prompt,
                system=self.system,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
        except LLMError as exc:
            log.error("LLM 调用失败，本批 %d 条按缺失处理：%s", len(chunk), exc)
            self.parse_failures += len(chunk)
            return [self._fallback(item, f"调用失败: {exc}") for item in chunk]

        parsed = self.parse_scores(completion.text, len(chunk))
        if parsed is None:
            self.parse_failures += len(chunk)
            log.warning("JSON 解析失败，原文前 200 字：%s", completion.text[:200])
            return [self._fallback(item, "解析失败") for item in chunk]

        out: list[Signal] = []
        for item, entry in zip(chunk, parsed, strict=True):
            out.append(
                Signal(
                    asset=item.asset,
                    score=float(entry.get("score", 0.0)),
                    confidence=float(entry.get("confidence", 0.5)),
                    source=self.name,
                    rationale=str(entry.get("reason", ""))[:120],
                )
            )
        return out

    @staticmethod
    def _fallback(item: NewsItem, why: str) -> Signal:
        """解析/调用失败时的占位信号。

        刻意给 ``confidence=0``：它会在加权聚合中失效，但**总数与顺序保持**，
        从而让失败率可被统计，而不是污染成"中性观点"。
        """
        return Signal(
            asset=item.asset,
            score=0.0,
            confidence=0.0,
            source="llm:error",
            rationale=why,
        )

    # ------------------------------------------------------------ JSON 解析
    @staticmethod
    def parse_scores(text: str, expected: int) -> list[dict] | None:
        """从模型输出里稳健地抠出长度为 ``expected`` 的 JSON 数组。

        依次尝试：代码块内容 → 裸数组 → 整体解析。
        长度不匹配或字段非法一律返回 ``None``，由调用方计为解析失败。
        """
        if not text or not text.strip():
            return None

        candidates: list[str] = []
        fence = _FENCE_RE.search(text)
        if fence:
            candidates.append(fence.group(1))
        array = _ARRAY_RE.search(text)
        if array:
            candidates.append(array.group(0))
        candidates.append(text.strip())

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, list) or len(data) != expected:
                continue
            cleaned: list[dict] = []
            ok = True
            for entry in data:
                if not isinstance(entry, dict):
                    ok = False
                    break
                try:
                    score = float(entry.get("score", 0.0))
                except (TypeError, ValueError):
                    ok = False
                    break
                try:
                    conf = float(entry.get("confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                cleaned.append(
                    {"score": score, "confidence": conf, "reason": entry.get("reason", "")}
                )
            if ok:
                return cleaned
        return None

    # ------------------------------------------------------------------ 统计
    def stats(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "items_scored": self.items_scored,
            "parse_failures": self.parse_failures,
            "anonymized": self.anonymize,
            "batch_size": self.batch_size,
        }


__all__ = ["DEFAULT_SYSTEM", "DEFAULT_TEMPLATE", "LLMExtractor"]

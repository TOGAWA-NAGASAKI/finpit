"""多构念打分 —— 一次调用让模型在多个维度上给同一段文本打分。

## 为什么需要它

MTMM 分析要求「多方法 × 多构念」的设计。若每个 (模型, 构念) 组合单独调用，
会有两个问题：调用数翻倍；且不同构念的提示词差异会混入方法因子，让分解失去意义。

**一次调用同时取多个构念**更干净：此时"方法"就是**这个模型对同一套指令的整体反应风格**，
这正是我们要估计的方法方差。

## 构念选择

采用 Loughran-McDonald 金融文本词典的经典分类维度（该词典把金融语气分为
positive/negative、uncertainty、litigious、constraining、forward-looking 等），
此处取其中三个在 10-K 叙述文本上有区分度的：

* ``sentiment`` —— 对公司前景的正面/负面倾向
* ``uncertainty`` —— 表述的不确定性/模糊程度
* ``forward_looking`` —— 讨论未来还是回顾过去

它们在概念上应当**只弱相关**。若实测发现同一模型在这三个维度上的打分高度一致，
那就是方法方差（晕轮效应）压过构念方差的直接证据。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from finpit.llm.client import LLMClient, LLMError

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConstructSpec:
    """一个待打分的构念。"""

    name: str
    description: str
    low_anchor: str
    high_anchor: str

    def prompt_line(self, idx: int) -> str:
        return (
            f"{idx}. {self.name}（{self.description}）："
            f"{self.low_anchor} 记 0 分，{self.high_anchor} 记 1 分，中间取小数"
        )


# Loughran-McDonald 风格的三个维度
FINANCIAL_CONSTRUCTS: list[ConstructSpec] = [
    ConstructSpec(
        name="sentiment",
        description="这段陈述对该公司前景的正面/负面倾向",
        low_anchor="-1 表示强烈负面",
        high_anchor="+1 表示强烈正面",
    ),
    ConstructSpec(
        name="uncertainty",
        description="这段陈述表述的不确定性、模糊或保留程度",
        low_anchor="0 表示非常确定、直接",
        high_anchor="1 表示高度含糊、充满保留",
    ),
    ConstructSpec(
        name="forward_looking",
        description="这段陈述是在讨论未来还是在回顾过去",
        low_anchor="0 表示纯粹回顾历史",
        high_anchor="1 表示纯粹前瞻未来",
    ),
]

_SYSTEM = (
    "你是一名严谨的金融文本分析师。你会收到若干段来自上市公司定期报告"
    "（10-K/10-Q）的原文句子，需要对每句话在指定的若干维度上分别打分。\n"
    "规则：\n"
    "1. 严格只依据文本字面语义判断，不要推测公司实际经营情况。\n"
    "2. 各维度相互独立，不要因为一句话某维度分高就抬高其他维度。\n"
    "3. 严格输出 JSON 数组，长度与句子数一致，不要输出任何解释或 markdown 代码块。"
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(slots=True)
class MultiConstructExtractor:
    """在同一批文本上对多个构念同时打分。"""

    client: LLMClient
    constructs: list[ConstructSpec] = field(
        default_factory=lambda: list(FINANCIAL_CONSTRUCTS)
    )
    batch_size: int = 6
    model: str | None = None
    max_tokens: int = 8192
    anonymize: bool = True
    parse_failures: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        used = self.model or self.client.settings.model
        return f"multi:{used}"

    def _build_prompt(self, texts: list[str]) -> str:
        dims = "\n".join(c.prompt_line(i + 1) for i, c in enumerate(self.constructs))
        keys = ", ".join(f'"{c.name}": 0.0' for c in self.constructs)
        items = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
        return (
            f"评分维度：\n{dims}\n\n"
            f"待评句子（共 {len(texts)} 句）：\n{items}\n\n"
            f"输出格式（长度必须为 {len(texts)} 的 JSON 数组）：\n"
            f"[{{{keys}}}, ...]"
        )

    @staticmethod
    def _parse_array(text: str, expected: int, keys: list[str]) -> list[dict[str, float]] | None:
        if not text or not text.strip():
            return None
        candidates: list[str] = []
        fence = _FENCE_RE.search(text)
        if fence:
            candidates.append(fence.group(1))
        arr = _ARRAY_RE.search(text)
        if arr:
            candidates.append(arr.group(0))
        candidates.append(text.strip())

        for cand in candidates:
            try:
                data = json.loads(cand)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, list) or len(data) != expected:
                continue
            out: list[dict[str, float]] = []
            ok = True
            for entry in data:
                if not isinstance(entry, dict):
                    ok = False
                    break
                row: dict[str, float] = {}
                for k in keys:
                    try:
                        row[k] = float(entry.get(k, 0.0))
                    except (TypeError, ValueError):
                        row[k] = 0.0
                out.append(row)
            if ok:
                return out
        return None

    def score_texts(self, texts: list[str]) -> list[dict[str, float]]:
        """返回与输入等长的列表，每项是 ``{构念名: 分数}``。"""
        keys = [c.name for c in self.constructs]
        results: list[dict[str, float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            try:
                completion = self.client.complete(
                    self._build_prompt(chunk),
                    system=_SYSTEM,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
            except LLMError as exc:
                log.error("多构念打分调用失败：%s", exc)
                self.parse_failures += len(chunk)
                results.extend([{k: 0.0 for k in keys} for _ in chunk])
                continue

            parsed = self._parse_array(completion.text, len(chunk), keys)
            if parsed is None:
                self.parse_failures += len(chunk)
                if completion.truncated:
                    log.warning("本批被 max_tokens 截断，解析失败")
                else:
                    log.warning("JSON 解析失败，前 160 字：%s", completion.text[:160])
                results.extend([{k: 0.0 for k in keys} for _ in chunk])
                continue
            results.extend(parsed)
        return results

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "constructs": [c.name for c in self.constructs],
            "parse_failures": self.parse_failures,
            "batch_size": self.batch_size,
        }


__all__ = [
    "FINANCIAL_CONSTRUCTS",
    "ConstructSpec",
    "MultiConstructExtractor",
]

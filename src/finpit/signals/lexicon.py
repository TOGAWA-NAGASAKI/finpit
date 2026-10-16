"""词典法情绪基线 —— 不依赖任何 LLM 的确定性抽取器。

存在的意义有三：

1. **可跑通的兜底路径**：没有 API key 时整个评测链路仍能端到端跑完；
2. **诚实的对照**：很多论文只报 LLM 相对"随机"的提升，而不报相对
   FinBERT/词典这类强基线的提升。真正的增量必须相对强基线度量；
3. **上限参照**：词典法的 IC 衡量了"文本表层信息"能榨出多少，
   LLM 超出它的部分才是语义理解的贡献。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from finpit.signals.base import NewsItem, Signal

# ---------------------------------------------------------------- 词典
# 中文财经语料的极性词表。权重按强度粗分：2=强，1=弱。
POSITIVE: dict[str, float] = {
    "超预期": 2.0,
    "上调": 2.0,
    "回购": 1.5,
    "增长": 1.5,
    "改善": 1.5,
    "强劲": 1.5,
    "稳健": 1.0,
    "提升": 1.0,
    "投产": 1.0,
    "订单": 0.5,
    "能见度": 0.5,
    "成本控制": 1.0,
    "见效": 1.0,
    "盈利": 1.5,
    "扭亏": 2.0,
    "中标": 1.5,
    "突破": 1.0,
    "龙头": 0.5,
    "放量": 0.5,
    "利好": 2.0,
    # 行情描述词（真实财经词典必然收录；也是结果泄漏能否被词典捕捉的关键）
    "上涨": 1.5,
    "大涨": 2.0,
    "涨停": 2.0,
    "走强": 1.5,
    "反弹": 1.5,
    "新高": 1.5,
}

NEGATIVE: dict[str, float] = {
    "下调": 2.0,
    "走弱": 2.0,
    "高企": 1.5,  # 库存高企
    "离职": 1.5,
    "减持": 1.5,
    "亏损": 2.0,
    "调查": 1.5,
    "合规成本": 1.0,
    "砍单": 2.0,
    "下滑": 2.0,
    "不确定性": 1.0,
    "风险": 0.5,
    "诉讼": 1.5,
    "违约": 2.0,
    "退市": 2.0,
    "商誉减值": 2.0,
    "停产": 2.0,
    "召回": 1.5,
    "处罚": 1.5,
    "利空": 2.0,
    # 行情描述词
    "下跌": 1.5,
    "大跌": 2.0,
    "跌停": 2.0,
    "走软": 1.5,
    "重挫": 2.0,
    "新低": 1.5,
}

# 明确表示"无方向"的措辞：命中则把分数往 0 拉
NEUTRAL_MARKERS: tuple[str, ...] = (
    "无重大变化",
    "例行",
    "换届",
    "完成",
    "披露月度",
)


@dataclass(slots=True)
class LexiconExtractor:
    """加权词典打分器。

    打分是词权之和经 ``tanh`` 压缩到 ``[-1, 1]``，避免长文本因词多而失真。
    """

    name: str = "lexicon-v1"
    positive: dict[str, float] = field(default_factory=lambda: dict(POSITIVE))
    negative: dict[str, float] = field(default_factory=lambda: dict(NEGATIVE))
    neutral_markers: tuple[str, ...] = NEUTRAL_MARKERS
    scale: float = 3.0
    """tanh 的尺度：越小越容易饱和。"""

    def score_text(self, text: str) -> tuple[float, float, str]:
        """返回 ``(分数, 置信度, 判据)``。"""
        pos_hits = [(w, wt) for w, wt in self.positive.items() if w in text]
        neg_hits = [(w, wt) for w, wt in self.negative.items() if w in text]
        neutral_hit = [w for w in self.neutral_markers if w in text]

        raw = sum(wt for _, wt in pos_hits) - sum(wt for _, wt in neg_hits)
        score = math.tanh(raw / self.scale) if raw else 0.0

        # 无极性词但命中中性措辞 -> 明确判为中性
        if not pos_hits and not neg_hits and neutral_hit:
            score = 0.0

        n_hits = len(pos_hits) + len(neg_hits)
        confidence = min(1.0, 0.35 + 0.25 * n_hits) if n_hits else 0.25

        parts = []
        if pos_hits:
            parts.append("正:" + ",".join(w for w, _ in pos_hits))
        if neg_hits:
            parts.append("负:" + ",".join(w for w, _ in neg_hits))
        if neutral_hit:
            parts.append("中性:" + ",".join(neutral_hit))
        rationale = " ".join(parts) or "未命中词典"

        return score, confidence, rationale

    def score_batch(self, items: list[NewsItem]) -> list[Signal]:
        out: list[Signal] = []
        for item in items:
            score, conf, why = self.score_text(item.text)
            out.append(
                Signal(
                    asset=item.asset,
                    score=score,
                    confidence=conf,
                    source=self.name,
                    rationale=why,
                )
            )
        return out


__all__ = ["NEGATIVE", "NEUTRAL_MARKERS", "POSITIVE", "LexiconExtractor"]

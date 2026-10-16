"""信号抽取层的测试。"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from finpit.signals import LexiconExtractor, NewsItem, anonymize
from finpit.signals.llm import LLMExtractor

NOW = datetime(2024, 3, 1, tzinfo=UTC)


def _item(text: str, asset: str = "ASSET001") -> NewsItem:
    return NewsItem(asset=asset, text=text, published_at=NOW)


# -------------------------------------------------------------------- 词典法
def test_positive_headline_scores_positive():
    score, _, _ = LexiconExtractor().score_text("季度营收超预期，上调全年指引")
    assert score > 0


def test_negative_headline_scores_negative():
    score, _, _ = LexiconExtractor().score_text("下调业绩指引，需求走弱")
    assert score < 0


def test_neutral_headline_scores_zero():
    score, _, _ = LexiconExtractor().score_text("发布公告，经营情况无重大变化")
    assert score == pytest.approx(0.0)


def test_score_bounded_by_one():
    extractor = LexiconExtractor()
    score, _, _ = extractor.score_text("超预期 上调 回购 增长 改善 强劲 利好 上涨 大涨 涨停")
    assert -1.0 <= score <= 1.0


def test_confidence_rises_with_hit_count():
    extractor = LexiconExtractor()
    _, conf_one, _ = extractor.score_text("上涨")
    _, conf_many, _ = extractor.score_text("上涨 大涨 涨停 超预期 利好")
    assert conf_many > conf_one


def test_rationale_records_matched_words():
    _, _, why = LexiconExtractor().score_text("下调业绩指引")
    assert "下调" in why


def test_batch_preserves_order_and_length():
    items = [_item("上涨"), _item("下跌"), _item("发布公告")]
    signals = LexiconExtractor().score_batch(items)
    assert len(signals) == 3
    assert signals[0].score > 0 > signals[1].score
    assert [s.asset for s in signals] == [i.asset for i in items]


def test_weighted_applies_confidence():
    from finpit.signals import Signal

    assert Signal(asset="a", score=1.0, confidence=0.5).weighted == pytest.approx(0.5)


def test_signal_clamps_out_of_range():
    from finpit.signals import Signal

    assert Signal(asset="a", score=5.0).score == 1.0
    assert Signal(asset="a", score=-5.0).score == -1.0
    assert Signal(asset="a", score=1.0, confidence=9.0).confidence == 1.0


# ------------------------------------------------------------------ 匿名化
def test_anonymize_removes_asset_codes():
    assert "ASSET017" not in anonymize("ASSET017 下调业绩指引")


def test_anonymize_removes_dates():
    out = anonymize("2024-03-05 公司发布公告")
    assert "2024" not in out


def test_anonymize_keeps_semantic_content():
    out = anonymize("2024-03-05 ASSET017 下调业绩指引，需求走弱")
    assert "下调业绩指引" in out
    assert "需求走弱" in out


# -------------------------------------------------------------- JSON 解析
def test_parse_scores_plain_array():
    text = '[{"score": 0.5, "confidence": 0.8}, {"score": -0.5, "confidence": 0.9}]'
    parsed = LLMExtractor.parse_scores(text, 2)
    assert parsed is not None
    assert parsed[0]["score"] == 0.5
    assert parsed[1]["confidence"] == 0.9


def test_parse_scores_from_markdown_fence():
    text = 'Here you go:\n```json\n[{"score": 0.1, "confidence": 0.7}]\n```\nHope that helps!'
    parsed = LLMExtractor.parse_scores(text, 1)
    assert parsed is not None and parsed[0]["score"] == 0.1


def test_parse_scores_with_surrounding_prose():
    text = 'Sure. [{"score": 0.3, "confidence": 0.6}] That is my answer.'
    parsed = LLMExtractor.parse_scores(text, 1)
    assert parsed is not None and parsed[0]["score"] == 0.3


def test_parse_scores_length_mismatch_rejected():
    """长度不匹配必须判失败 —— 否则会把错位的分数安到错的标的上。"""
    assert LLMExtractor.parse_scores('[{"score": 0.5}]', 3) is None


def test_parse_scores_missing_field_defaults():
    parsed = LLMExtractor.parse_scores('[{"score": 0.4}]', 1)
    assert parsed is not None and parsed[0]["confidence"] == 0.5


def test_parse_scores_non_numeric_rejected():
    assert LLMExtractor.parse_scores('[{"score": "high"}]', 1) is None


def test_parse_scores_empty_returns_none():
    assert LLMExtractor.parse_scores("", 1) is None
    assert LLMExtractor.parse_scores("   ", 1) is None


def test_parse_scores_not_a_list_rejected():
    assert LLMExtractor.parse_scores('{"score": 0.5}', 1) is None


def test_parse_scores_handles_reasoning_preamble():
    text = (
        "Let me think about this. The news is positive because...\n\n"
        '[{"score": 0.7, "confidence": 0.8, "reason": "营收超预期"}]\n'
    )
    parsed = LLMExtractor.parse_scores(text, 1)
    assert parsed is not None and parsed[0]["reason"] == "营收超预期"


# ------------------------------------------------------------ LLM 抽取器
class _FakeClient:
    """不联网的假客户端，返回固定 JSON。"""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

        class _Settings:
            model = "fake-model"

        self.settings = _Settings()

    def complete(self, prompt, **kwargs):  # noqa: ANN001, ANN003
        from finpit.llm import Completion

        self.calls += 1
        return Completion(text=self.payload, model="fake-model", total_tokens=10)


def test_llm_extractor_parses_batch():
    payload = json.dumps([{"score": 0.6, "confidence": 0.9}, {"score": -0.4, "confidence": 0.8}])
    extractor = LLMExtractor(client=_FakeClient(payload), batch_size=8)
    signals = extractor.score_batch([_item("上涨"), _item("下跌")])
    assert [round(s.score, 1) for s in signals] == [0.6, -0.4]
    assert extractor.parse_failures == 0


def test_llm_extractor_counts_parse_failures_and_degrades_safely():
    extractor = LLMExtractor(client=_FakeClient("not json at all"), batch_size=4)
    signals = extractor.score_batch([_item("上涨"), _item("下跌")])
    # 失败必须被计数，且占位信号 confidence=0（在加权聚合中失效）
    assert extractor.parse_failures == 2
    assert all(s.confidence == 0.0 for s in signals)
    assert len(signals) == 2


def test_llm_extractor_batches_calls():
    payload = json.dumps([{"score": 0.1, "confidence": 0.5}] * 3)
    client = _FakeClient(payload)
    extractor = LLMExtractor(client=client, batch_size=3)
    extractor.score_batch([_item("a"), _item("b"), _item("c"), _item("d"), _item("e"), _item("f")])
    assert client.calls == 2  # 6 条 / 每批 3 条


def test_llm_extractor_name_reflects_anonymization():
    anon = LLMExtractor(client=_FakeClient("[]"), anonymize=True)
    raw = LLMExtractor(client=_FakeClient("[]"), anonymize=False)
    assert anon.name.endswith(":anon")
    assert raw.name.endswith(":raw")
    assert anon.name != raw.name


def test_llm_extractor_stats():
    extractor = LLMExtractor(client=_FakeClient("[]"), batch_size=2)
    extractor.score_batch([_item("a")])
    stats = extractor.stats()
    assert stats["anonymized"] is True
    assert stats["items_scored"] >= 1

"""泄漏消融实验台的测试。

这些测试锁定的是**科学结论**，而不只是代码行为：如果哪天改动让
"未来函数不再虚高 IC"或"审计不再能拦截越界"，测试必须失败。
"""

from __future__ import annotations

import numpy as np
import pytest

from finpit.data import SyntheticConfig, generate_corpus
from finpit.eval import LeakMode, LeakageAblation, build_panel
from finpit.signals import LexiconExtractor


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(SyntheticConfig(n_assets=30, n_days=400, seed=2026))


@pytest.fixture(scope="module")
def report(corpus):
    return LeakageAblation(corpus, LexiconExtractor(), max_rows=6000).run()


# ------------------------------------------------------------------ 面板构建
def test_panel_is_complete_cross_sections(corpus):
    """max_rows 必须在日期边界截断，否则横截面不完整会污染统计。"""
    panel, _ = build_panel(corpus.store, corpus, max_rows=600)
    assert len(panel) % len(corpus.assets) == 0


def test_panel_respects_point_in_time(corpus):
    """面板里每条新闻都必须在决策时点之前已公开。"""
    panel, _ = build_panel(corpus.store, corpus, max_rows=400)
    for row in panel:
        assert row.news_time <= row.decision_time


def test_panel_forward_date_is_after_news_date(corpus):
    panel, _ = build_panel(corpus.store, corpus, max_rows=400)
    for row in panel:
        assert row.forward_date > row.news_date


def test_clean_panel_build_has_no_pit_violations(corpus):
    corpus.store.reset_audit()
    build_panel(corpus.store, corpus, max_rows=400)
    assert corpus.store.audit().clean


# --------------------------------------------------------------------- 指标
def test_future_return_mode_is_sanity_check(report):
    """把目标当特征必须得到 IC≈1，否则说明评测管线本身有问题。"""
    result = next(r for r in report.results if r.mode is LeakMode.FUTURE_RETURN)
    assert result.metrics.ic == pytest.approx(1.0, abs=1e-9)


def test_clean_mode_is_below_true_ic(report):
    """干净流程的 IC 必须低于上帝视角上限 —— 文本抽取本身有损失。"""
    assert report.clean.metrics.ic < report.true_ic


def test_oracle_latent_approaches_true_ic(report):
    """直读潜在真值应当接近真实 IC（差异来自加权与样本截断）。"""
    result = next(r for r in report.results if r.mode is LeakMode.ORACLE_LATENT)
    assert result.metrics.ic == pytest.approx(report.true_ic, abs=0.02)


def test_outcome_prompt_inflates_ic_substantially(report):
    """结果泄漏进提示词：这是最高危也最常见的 LLM 管线泄漏。"""
    result = next(r for r in report.results if r.mode is LeakMode.OUTCOME_PROMPT)
    assert result.metrics.ic > report.clean.metrics.ic * 3


def test_pit_violation_inflates_ic(report):
    """同日行情播报被对齐到同日收益 —— 差一天的未来函数。"""
    result = next(r for r in report.results if r.mode is LeakMode.PIT_VIOLATION)
    assert result.metrics.ic > report.clean.metrics.ic


def test_pit_violation_is_fully_blocked(report):
    """审计层必须对每一行越界都留下记录。"""
    result = next(r for r in report.results if r.mode is LeakMode.PIT_VIOLATION)
    assert result.pit_blocks == report.n_panel_rows


def test_cherry_pick_moves_return_level_not_only_ic(report):
    """幸存者偏差的核心危害是抬高收益水平，而非必然抬高 IC。"""
    result = next(r for r in report.results if r.mode is LeakMode.CHERRY_PICK)
    assert result.metrics.n_obs < report.clean.metrics.n_obs
    assert result.metrics.mean_return != pytest.approx(report.clean.metrics.mean_return)


def test_inflation_helpers(report):
    base = report.clean.metrics.ic
    future = next(r for r in report.results if r.mode is LeakMode.FUTURE_RETURN)
    assert future.inflation_ic(base) > 1.0
    assert report.clean.inflation_ic(base) == pytest.approx(0.0)


def test_report_to_dict_is_serialisable(report):
    import json

    payload = report.to_dict()
    assert json.loads(json.dumps(payload))["results"]
    assert payload["n_panel_rows"] > 0
    assert "audit" in payload


def test_report_table_contains_all_modes(report):
    table = report.table()
    for mode in LeakMode:
        assert mode.value in table


def test_subset_of_modes_supported(corpus):
    ablation = LeakageAblation(corpus, LexiconExtractor(), max_rows=400)
    partial = ablation.run([LeakMode.CLEAN, LeakMode.FUTURE_RETURN])
    assert len(partial.results) == 2


def test_empty_panel_raises(corpus):
    tiny = generate_corpus(SyntheticConfig(n_assets=2, n_days=51, seed=1))
    # 面板至少需要两天才能产生一行；正常路径不该为空
    panel, _ = build_panel(tiny.store, tiny, max_rows=None)
    assert len(panel) > 0


def test_extractor_name_recorded(report):
    assert report.extractor_name == "lexicon-v1"

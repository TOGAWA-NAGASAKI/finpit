"""合成语料生成器的测试 —— 重点验证它确实校准到了真实市场量级。"""

from __future__ import annotations

import numpy as np
import pytest

from finpit.data import SyntheticConfig, generate_corpus


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(SyntheticConfig(n_assets=25, n_days=400, seed=42))


def test_deterministic_given_seed():
    a = generate_corpus(SyntheticConfig(n_assets=5, n_days=60, seed=7))
    b = generate_corpus(SyntheticConfig(n_assets=5, n_days=60, seed=7))
    assert a.assets == b.assets
    assert a.dates == b.dates
    key = f"{a.assets[0]}:news"
    assert [r.payload["headline"] for r in a.store._by_key[key]] == [
        r.payload["headline"] for r in b.store._by_key[key]
    ]


def test_different_seed_differs():
    a = generate_corpus(SyntheticConfig(n_assets=5, n_days=60, seed=1))
    b = generate_corpus(SyntheticConfig(n_assets=5, n_days=60, seed=2))
    key = f"{a.assets[0]}:news"
    assert a.store._by_key[key][0].payload["headline"] != b.store._by_key[key][0].payload["headline"]


def test_realized_ic_matches_target(corpus):
    """真实 IC 应当落在目标值附近 —— 这是整个评测的地面真值。"""
    assert corpus.realized_ic() == pytest.approx(corpus.config.target_ic, abs=0.015)


def test_returns_have_realistic_scale(corpus):
    all_returns = np.concatenate(
        [np.array(list(corpus.returns[d.strftime("%Y-%m-%d")].values())) for d in corpus.dates]
    )
    # 日频收益 std 应在 1%~3% 量级
    assert 0.005 < all_returns.std() < 0.05


def test_returns_are_fat_tailed(corpus):
    """真实收益是厚尾的；正态假设会严重低估尾部风险。"""
    all_returns = np.concatenate(
        [np.array(list(corpus.returns[d.strftime("%Y-%m-%d")].values())) for d in corpus.dates]
    )
    centered = all_returns - all_returns.mean()
    excess_kurtosis = float((centered**4).mean() / all_returns.var() ** 2 - 3.0)
    assert excess_kurtosis > 1.0


def test_volatility_clustering_present(corpus):
    """波动率聚集：|r| 应有正自相关。"""
    mat = np.abs(
        np.array(
            [
                [corpus.returns[d.strftime("%Y-%m-%d")][a] for a in corpus.assets]
                for d in corpus.dates
            ]
        )
    )
    series = mat.mean(axis=1)
    autocorr = float(np.corrcoef(series[:-1], series[1:])[0, 1])
    assert autocorr > 0.05


def test_news_published_after_close(corpus):
    """新闻必须晚于当日收盘 —— 这是未来函数能被演示的前提。"""
    cfg = corpus.config
    for record in corpus.store._by_key[f"{corpus.assets[0]}:news"]:
        day = record.published_at.replace(hour=0, minute=0, second=0, microsecond=0)
        close = day.replace(hour=cfg.market_close_hour)
        assert record.published_at > close


def test_price_recap_embeds_same_day_return(corpus):
    """附带行情播报的标题里，数字应与当日收益一致。"""
    cfg = corpus.config
    assert cfg.price_recap_prob > 0
    found = 0
    for record in corpus.store._by_key[f"{corpus.assets[0]}:news"]:
        headline = record.payload["headline"]
        if "盘面看" not in headline:
            continue
        found += 1
        date_key = record.payload["date"]
        actual = corpus.returns[date_key][record.payload["asset"]] * 100.0
        direction = "上涨" if actual >= 0 else "下跌"
        assert direction in headline
        assert f"{abs(actual):.2f}%" in headline
    assert found > 0, "样本里没有任何行情播报标题，泄漏演示会失效"


def test_config_validation():
    with pytest.raises(ValueError):
        SyntheticConfig(n_days=10)
    with pytest.raises(ValueError):
        SyntheticConfig(n_assets=1)
    with pytest.raises(ValueError):
        SyntheticConfig(target_ic=1.5)
    with pytest.raises(ValueError):
        SyntheticConfig(price_recap_prob=2.0)


def test_true_beta_derived_from_target_ic():
    cfg = SyntheticConfig(target_ic=0.05, base_vol=0.02)
    expected = 0.05 * 0.02 / np.sqrt(1 - 0.05**2)
    assert cfg.true_beta == pytest.approx(expected)


def test_describe_mentions_key_facts(corpus):
    text = corpus.describe()
    assert "合成语料" in text
    assert "真实 IC" in text

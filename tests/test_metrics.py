"""指标计算的测试 —— 用可手算的构造样本验证正确性。"""

from __future__ import annotations

import numpy as np
import pytest

from finpit.eval.metrics import (
    evaluate,
    hit_rate,
    information_coefficient,
    quantile_spread,
    rank_ic,
)


def test_perfect_correlation():
    x = [1.0, 2.0, 3.0, 4.0]
    assert information_coefficient(x, x) == pytest.approx(1.0)


def test_perfect_anticorrelation():
    x = [1.0, 2.0, 3.0, 4.0]
    assert information_coefficient(x, [-v for v in x]) == pytest.approx(-1.0)


def test_rank_ic_ignores_monotone_rescaling():
    """秩相关对单调变换不变 —— 这正是它比 Pearson 稳健的原因。"""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 4.0, 9.0, 16.0, 25.0]
    assert rank_ic(x, y) == pytest.approx(1.0)
    assert information_coefficient(x, y) < 1.0


def test_rank_ic_handles_ties():
    x = [1.0, 1.0, 2.0, 3.0]
    y = [1.0, 1.0, 2.0, 3.0]
    assert rank_ic(x, y) == pytest.approx(1.0)


def test_hit_rate_known_value():
    pred = [1.0, -1.0, 1.0, -1.0]
    actual = [0.5, -0.5, -0.5, 0.5]
    assert hit_rate(pred, actual) == pytest.approx(0.5)


def test_hit_rate_ignores_zeros():
    pred = [1.0, 0.0, -1.0]
    actual = [0.5, 0.5, -0.5]
    # 0 分不计入分母
    assert hit_rate(pred, actual) == pytest.approx(1.0)


def test_quantile_spread_perfect_signal():
    pred = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    actual = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    spread = quantile_spread(pred, actual, n_buckets=2)
    assert spread == pytest.approx(1.0)


def test_quantile_spread_zero_signal():
    pred = [1.0, 2.0, 3.0, 4.0]
    actual = [5.0, 5.0, 5.0, 5.0]
    assert quantile_spread(pred, actual, n_buckets=2) == pytest.approx(0.0)


def test_nan_and_inf_filtered():
    pred = [1.0, np.nan, 3.0, np.inf, 5.0]
    actual = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert information_coefficient(pred, actual) == pytest.approx(1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        information_coefficient([1.0, 2.0], [1.0])


def test_constant_input_returns_nan():
    assert np.isnan(information_coefficient([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_too_few_points_returns_nan():
    assert np.isnan(information_coefficient([1.0, 2.0], [1.0, 2.0]))


def test_evaluate_bundles_metrics():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 0.3 * x + rng.standard_normal(500)
    metrics = evaluate(x, y)
    assert metrics.n_obs == 500
    assert metrics.ic > 0.2
    assert metrics.rank_ic > 0.2
    assert 0.0 <= metrics.hit_rate <= 1.0
    assert isinstance(metrics.as_dict()["ic"], float)


def test_mean_return_recorded():
    metrics = evaluate([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert metrics.mean_return == pytest.approx(0.2)

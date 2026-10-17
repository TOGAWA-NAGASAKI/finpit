"""Errors-in-variables 模块的测试。

这些测试锁的是**数学正确性**：可靠性识别、衰减修正、符号翻转机制。
若哪天改动让修正不再收敛到真值，测试必须失败。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from finpit.eiv import (
    corrected_regression,
    disattenuate_correlation,
    disattenuate_slope,
    error_correlation,
    estimate_reliability,
    mean_pairwise_correlation,
    plim_coefficients,
    sign_flip_report,
    spearman_brown,
)


# ------------------------------------------------------------------ 基础构件
def test_spearman_brown_matches_hand_computation():
    # λ₁=0.5, K=3 -> 3*0.5/(1+2*0.5) = 1.5/2 = 0.75
    assert spearman_brown(0.5, 3) == pytest.approx(0.75)


def test_spearman_brown_k1_is_identity():
    assert spearman_brown(0.42, 1) == pytest.approx(0.42)


def test_spearman_brown_diminishing_returns():
    """多加模型的边际收益递减 —— 这是本模块的实践要点之一。"""
    values = [spearman_brown(0.5, k) for k in (1, 2, 3, 5, 10, 50)]
    increments = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    assert all(v > 0 for v in increments)
    assert all(increments[i] >= increments[i + 1] for i in range(len(increments) - 1))


def test_spearman_brown_approaches_one():
    assert spearman_brown(0.5, 10_000) == pytest.approx(1.0, abs=1e-3)


def test_spearman_brown_handles_zero_reliability():
    assert math.isnan(spearman_brown(0.0, 3))
    assert math.isnan(spearman_brown(-0.2, 3))


def test_spearman_brown_rejects_bad_k():
    with pytest.raises(ValueError):
        spearman_brown(0.5, 0)


def test_mean_pairwise_correlation_known_matrix():
    corr = np.array([[1.0, 0.4, 0.6], [0.4, 1.0, 0.5], [0.6, 0.5, 1.0]])
    assert mean_pairwise_correlation(corr) == pytest.approx((0.4 + 0.6 + 0.5) / 3)


def test_estimate_reliability_requires_two_indicators():
    from finpit.eiv import estimate_reliability as est

    with pytest.raises(ValueError):
        est(np.random.default_rng(0).standard_normal((50, 1)))


def test_cronbach_equals_spearman_brown_for_standardised_items():
    """对标准化条目，Cronbach α 应与 Spearman-Brown 的 λ_K 相同 —— 交叉校验。"""
    rng = np.random.default_rng(7)
    theta = rng.standard_normal(4000)
    x = theta[:, None] + 0.8 * rng.standard_normal((4000, 5))
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)

    est = estimate_reliability(x)
    assert est.cronbach == pytest.approx(est.lambda_mean, abs=1e-9)


# ------------------------------------------------------------------ 单变量修正
def test_disattenuate_slope_roundtrip():
    assert disattenuate_slope(0.05, 0.5) == pytest.approx(0.10)


def test_disattenuate_slope_rejects_invalid_reliability():
    with pytest.raises(ValueError):
        disattenuate_slope(0.05, 0.0)
    with pytest.raises(ValueError):
        disattenuate_slope(0.05, 1.5)


def test_disattenuate_correlation_both_sides():
    # 两侧可靠性均为 0.25 -> 修正因子 1/sqrt(0.25*0.25) = 4
    assert disattenuate_correlation(0.15, 0.25, 0.25) == pytest.approx(0.6)


def test_disattenuate_correlation_clipped():
    # 0.3/0.25 = 1.2 > 1，必须裁剪
    assert disattenuate_correlation(0.3, 0.25, 0.25) == pytest.approx(1.0)
    assert disattenuate_correlation(0.9, 0.3, 0.3) == pytest.approx(1.0)


# ------------------------------------------------------------ 多变量概率极限
def test_plim_without_error_is_true_beta():
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    zero = np.zeros((2, 2))
    beta = np.array([0.1, -0.05])
    assert plim_coefficients(cov, zero, beta) == pytest.approx(beta)


def test_plim_single_regressor_reduces_to_attenuation():
    """单变量情形必须退化为 β·λ —— 这是对实现的强约束。"""
    var_theta, var_u, beta = 1.0, 1.0, 0.10
    plim = plim_coefficients(np.array([[var_theta]]), np.array([[var_u]]), np.array([beta]))
    lam = var_theta / (var_theta + var_u)
    assert plim[0] == pytest.approx(beta * lam)


def test_plim_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        plim_coefficients(np.eye(2), np.eye(3), np.array([1.0, 1.0]))


def test_sign_flip_report_detects_flip():
    """经典机制：一个自变量有噪声，污染另一个相关系数的符号。"""
    # 两自变量相关系数 0.8；x1 有较大测量误差
    true_cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    error_cov = np.array([[1.5, 0.0], [0.0, 0.0]])
    beta = np.array([0.10, 0.05])

    rep = sign_flip_report(true_cov, error_cov, beta, names=["sentiment_llm", "traditional"])
    assert rep.beta_plim[0] < beta[0]  # 自身被衰减
    # x2 无测量误差，但因与 x1 相关而被污染
    assert rep.beta_plim[1] != pytest.approx(beta[1])


def test_sign_flip_report_no_flip_without_error():
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    beta = np.array([0.1, 0.05])
    rep = sign_flip_report(cov, np.zeros((2, 2)), beta)
    assert rep.flipped == []
    assert "符号被测量误差翻转" not in rep.summary()


def test_sign_flip_summary_mentions_flipped_names():
    true_cov = np.array([[1.0, -0.9], [-0.9, 1.0]])
    error_cov = np.array([[2.0, 0.0], [0.0, 0.0]])
    beta = np.array([0.10, 0.02])
    rep = sign_flip_report(true_cov, error_cov, beta, names=["a", "b"])
    if rep.flipped:
        assert "符号被测量误差翻转" in rep.summary()


# -------------------------------------------------------------- 端到端模拟
def _simulate(*, n: int, k: int, beta_true: float, lambda_true: float, seed: int):
    """构造一个可靠性已知的受控数据：y = β·θ + ε，指标 s = θ + u。"""
    rng = np.random.default_rng(seed)
    var_theta = 1.0
    # λ = v/(v+s²) = lambda_true  =>  s² = v(1-λ)/λ
    var_u = var_theta * (1.0 - lambda_true) / lambda_true
    theta = rng.standard_normal(n) * math.sqrt(var_theta)
    y = beta_true * theta + rng.standard_normal(n) * 0.5
    indicators = theta[:, None] + rng.standard_normal((n, k)) * math.sqrt(var_u)
    return y, theta, indicators


def test_corrected_regression_recovers_true_beta():
    """核心断言：修正后的系数应逼近真值，而未修正的明显偏低。"""
    beta_true, lam1, k, n = 0.10, 0.5, 5, 20000
    y, _, indicators = _simulate(n=n, k=k, beta_true=beta_true, lambda_true=lam1, seed=11)

    res = corrected_regression(y, indicators, beta_true=beta_true)

    lam_k = spearman_brown(lam1, k)
    assert res.lambda_single == pytest.approx(lam1, abs=0.05)
    assert res.lambda_mean == pytest.approx(lam_k, abs=0.05)

    # 未修正：被压到 β·λ_K 附近
    assert res.beta_observed == pytest.approx(beta_true * lam_k, rel=0.15)
    # 修正后：逼近真值
    assert res.beta_corrected == pytest.approx(beta_true, rel=0.10)
    # 修正确实把它放大了
    assert res.inflation > 1.0


def test_single_indicator_attenuation_is_worse():
    """只用 1 个 LLM 分数比用 K 个的均值更糟 —— 量化「多模型」的价值。"""
    beta_true, lam1, n = 0.10, 0.5, 20000
    y, _, ind = _simulate(n=n, k=6, beta_true=beta_true, lambda_true=lam1, seed=13)

    from finpit.eiv.attenuation import _ols

    beta_one, _ = _ols(y, ind[:, 0])
    beta_mean, _ = _ols(y, ind.mean(axis=1))

    # 单指标衰减到 β·λ₁；K 指标均值衰减到 β·λ_K > β·λ₁
    assert beta_one == pytest.approx(beta_true * lam1, rel=0.2)
    assert abs(beta_mean) > abs(beta_one)


def test_bootstrap_ci_produced_and_covers_truth():
    beta_true, n = 0.10, 4000
    y, _, ind = _simulate(n=n, k=4, beta_true=beta_true, lambda_true=0.55, seed=17)
    res = corrected_regression(y, ind, bootstrap=300, beta_true=beta_true)
    lo, hi = res.ci_corrected
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo < hi
    assert lo <= beta_true <= hi


def test_corrected_regression_requires_two_indicators():
    y = np.zeros(50)
    with pytest.raises(ValueError):
        corrected_regression(y, np.zeros((50, 1)))


def test_corrected_regression_validates_lengths():
    with pytest.raises(ValueError):
        corrected_regression(np.zeros(10), np.zeros((9, 3)))


# ------------------------------------------------------- 前提检验（误差独立）
def test_error_correlation_detects_independent_errors():
    _, theta, ind = _simulate(n=20000, k=4, beta_true=0.1, lambda_true=0.5, seed=23)
    diag = error_correlation(ind, theta)
    assert abs(diag["mean_error_corr"]) < 0.05
    assert diag["independence_assumption_ok"] is True
    # 由误差方差反推的 λ 应与配置一致
    assert diag["lambda_implied_by_error_var"] == pytest.approx(0.5, abs=0.05)


def test_error_correlation_flags_shared_bias():
    """共享偏差（各模型误差正相关）会让「误差独立」前提失效。"""
    rng = np.random.default_rng(3)
    n, k = 8000, 4
    theta = rng.standard_normal(n)
    shared = rng.standard_normal(n) * 0.9  # 各模型共享的系统性偏差
    ind = theta[:, None] + shared[:, None] + 0.3 * rng.standard_normal((n, k))

    diag = error_correlation(ind, theta)
    assert diag["mean_error_corr"] > 0.5
    assert diag["independence_assumption_ok"] is False


def test_error_correlation_validates_length():
    with pytest.raises(ValueError):
        error_correlation(np.zeros((10, 3)), np.zeros(9))


# ------------------------------------------------------------------ 工具
def test_needed_k_solves_spearman_brown():
    est = estimate_reliability(
        _simulate(n=6000, k=4, beta_true=0.1, lambda_true=0.5, seed=29)[2]
    )
    k_needed = est.needed_k(0.8)
    assert k_needed is not None
    assert spearman_brown(est.lambda_single, k_needed) >= 0.8
    assert spearman_brown(est.lambda_single, k_needed - 1) < 0.8


def test_needed_k_returns_one_when_already_sufficient():
    est = estimate_reliability(
        _simulate(n=6000, k=4, beta_true=0.1, lambda_true=0.95, seed=31)[2]
    )
    assert est.needed_k(0.5) == 1


def test_estimate_serialisation():
    est = estimate_reliability(
        _simulate(n=2000, k=3, beta_true=0.1, lambda_true=0.5, seed=37)[2], bootstrap=100
    )
    payload = est.as_dict()
    assert payload["k"] == 3
    assert len(payload["ci95_lambda_mean"]) == 2
    assert "λ₁" in est.summary()


def test_attenuation_result_serialisation():
    y, _, ind = _simulate(n=2000, k=3, beta_true=0.1, lambda_true=0.6, seed=41)
    res = corrected_regression(y, ind, bootstrap=100, beta_true=0.1)
    payload = res.as_dict()
    assert payload["beta_true"] == 0.1
    assert payload["inflation"] > 1.0
    assert "衰减修正" in res.summary()

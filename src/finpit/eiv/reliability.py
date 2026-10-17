"""可靠性估计 —— 无需真值即可识别测量误差的大小。

## 问题

当 LLM 生成的分数 ``x`` 被当作真实构念 ``θ`` 的代理放进回归时：

    x = θ + u,   u ⊥ θ

OLS 斜率会被**衰减**：

    β̂ = β · λ,   λ = Var(θ) / (Var(θ) + σ²_u) ∈ (0, 1]

``λ`` 称为**可靠性**（reliability）。要用它做修正就必须先估计它 ——
而难点在于 ``θ`` 本身不可观测。

## 经典识别策略：平行指标

若手上有 K 个**独立测量同一构念**的指标 ``s_1..s_K``（例如 K 个不同 LLM
对同一文本打的分），且误差相互独立，则任意两个指标的相关系数就等于单指标可靠性：

    corr(s_j, s_k) = λ₁        （j ≠ k）

于是 ``λ̂₁ = 两两相关系数的均值``。这不是新结果 —— 它就是
Spearman-Brown 与 Cronbach's α 的经典内容 —— 但把它用在 LLM 生成的
变量上，能直接回答一个实际问题：**你的 LLM 因子有多少是噪声？**

K 个指标取均值后的可靠性由 Spearman-Brown 给出：

    λ_K = K·λ₁ / (1 + (K - 1)·λ₁)

## 关键前提及其脆弱性

上述识别**依赖"误差相互独立"**。多个 LLM 共享训练语料与架构先验，
其误差很可能正相关；一旦如此，``λ_K`` 会被**高估**，修正量随之不足。
本模块因此额外提供 :func:`error_correlation`，在有真值的受控环境下
检验这一前提。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_matrix(indicators: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(indicators, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"指标矩阵必须是二维 (n_samples, k_indicators)，收到 {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError("至少需要 2 个指标才能识别可靠性")
    return arr


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    """皮尔逊相关矩阵（列之间）。"""
    return np.corrcoef(x, rowvar=False)


def mean_pairwise_correlation(corr: np.ndarray) -> float:
    """相关矩阵的非对角均值 —— 即单指标可靠性 λ₁ 的估计。"""
    k = corr.shape[0]
    if k < 2:
        raise ValueError("至少 2 个指标")
    off_diag = (corr.sum() - np.trace(corr)) / (k * (k - 1))
    return float(off_diag)


def spearman_brown(lambda_single: float, k: int) -> float:
    """K 个平行指标取均值后的可靠性。

    ``λ_K = K·λ₁ / (1 + (K-1)·λ₁)``

    注意边际收益递减：λ₁=0.5 时 K=3 得到 0.75，K=10 才到 0.91。
    这解释了为什么「多加几个模型」不能无限提升信号质量。
    """
    if k < 1:
        raise ValueError("k 至少为 1")
    if not 0.0 < lambda_single <= 1.0:
        # λ ≤ 0 意味着指标间无共同成分，此时谈可靠性没有意义
        return float("nan") if lambda_single <= 0 else 1.0
    return (k * lambda_single) / (1.0 + (k - 1.0) * lambda_single)


def cronbach_alpha(indicators: np.ndarray | list) -> float:
    """Cronbach's α（标准化条目版）。

    对标准化条目，``α`` 与 Spearman-Brown 给出的 ``λ_K`` 数值相同 ——
    二者本是同一件事的两种表述。此处保留独立实现以便交叉验证。
    """
    x = _as_matrix(indicators)
    item_vars = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    k = x.shape[1]
    if total_var <= 0:
        return float("nan")
    return float((k / (k - 1.0)) * (1.0 - item_vars.sum() / total_var))


@dataclass(slots=True)
class ReliabilityEstimate:
    """一组指标的可靠性估计。"""

    k: int
    n: int
    lambda_single: float
    """λ₁：单指标可靠性 = 两两相关系数均值。"""

    lambda_mean: float
    """λ_K：K 个指标取均值后的可靠性（Spearman-Brown）。"""

    cronbach: float
    """Cronbach's α（应与 λ_K 一致，用作交叉校验）。"""

    r_min: float
    r_max: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")

    @property
    def noise_share(self) -> float:
        """单个指标中噪声占比 = 1 - λ₁。"""
        return 1.0 - self.lambda_single

    @property
    def attenuation(self) -> float:
        """用**单个** LLM 分数做回归时，真实斜率的保留比例。"""
        return self.lambda_single

    def needed_k(self, target: float) -> int | None:
        """要达到目标可靠性 ``target``，需要多少个平行指标。

        由 Spearman-Brown 反解：``K = λ(1-λ₁) / (λ₁(1-λ))``。
        若 λ₁ ≥ target 则为 1；若 λ₁ ≤ 0 则无解。
        """
        if self.lambda_single >= target:
            return 1
        if self.lambda_single <= 0:
            return None
        k = (target * (1.0 - self.lambda_single)) / (
            self.lambda_single * (1.0 - target)
        )
        return int(math.ceil(k))

    def summary(self) -> str:
        return (
            f"可靠性估计（n={self.n}, K={self.k}）\n"
            f"  λ₁（单指标）      = {self.lambda_single:.4f}  "
            f"→ 噪声占 {self.noise_share:.1%}\n"
            f"  λ_K（K 指标均值） = {self.lambda_mean:.4f}\n"
            f"  Cronbach α        = {self.cronbach:.4f}  （应与 λ_K 一致）\n"
            f"  两两相关范围      = [{self.r_min:+.3f}, {self.r_max:+.3f}]\n"
            f"  95% 区间（λ_K）   = [{self.ci_low:.4f}, {self.ci_high:.4f}]"
            if math.isfinite(self.ci_low)
            else (
                f"可靠性估计（n={self.n}, K={self.k}）\n"
                f"  λ₁（单指标）      = {self.lambda_single:.4f}  "
                f"→ 噪声占 {self.noise_share:.1%}\n"
                f"  λ_K（K 指标均值） = {self.lambda_mean:.4f}\n"
                f"  Cronbach α        = {self.cronbach:.4f}"
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n": self.n,
            "lambda_single": round(self.lambda_single, 5),
            "lambda_mean": round(self.lambda_mean, 5),
            "cronbach_alpha": round(self.cronbach, 5),
            "noise_share": round(self.noise_share, 5),
            "r_min": round(self.r_min, 5),
            "r_max": round(self.r_max, 5),
            "ci95_lambda_mean": None
            if not math.isfinite(self.ci_low)
            else [round(self.ci_low, 5), round(self.ci_high, 5)],
        }


def estimate_reliability(
    indicators: np.ndarray | list,
    *,
    bootstrap: int = 0,
    seed: int = 20261016,
) -> ReliabilityEstimate:
    """从 K 个平行指标估计可靠性。

    Args:
        indicators: ``(n_samples, k_indicators)``。
        bootstrap: > 0 时对样本重抽样以给出 λ_K 的 95% 区间。
    """
    x = _as_matrix(indicators)
    n, k = x.shape
    corr = correlation_matrix(x)
    lam1 = mean_pairwise_correlation(corr)
    off = corr[np.triu_indices(k, 1)]

    est = ReliabilityEstimate(
        k=k,
        n=n,
        lambda_single=lam1,
        lambda_mean=spearman_brown(lam1, k),
        cronbach=cronbach_alpha(x),
        r_min=float(off.min()),
        r_max=float(off.max()),
    )

    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        samples = np.empty(bootstrap, dtype=float)
        for b in range(bootstrap):
            idx = rng.integers(0, n, n)
            c = correlation_matrix(x[idx])
            samples[b] = spearman_brown(mean_pairwise_correlation(c), k)
        finite = samples[np.isfinite(samples)]
        if len(finite) >= bootstrap // 2:
            est.ci_low = float(np.percentile(finite, 2.5))
            est.ci_high = float(np.percentile(finite, 97.5))

    return est


# ------------------------------------------------------- 前提检验（需真值）
def error_correlation(
    indicators: np.ndarray | list,
    truth: np.ndarray | list,
) -> dict[str, float]:
    """检验「误差相互独立」这一识别前提。

    **只在有真值的受控环境下可用**（合成数据、或已知基准标签）。

    若各指标的误差正相关，则 Spearman-Brown 会**高估** ``λ_K``，
    修正量不足，结论仍偏向原估计 —— 这是多模型集成最常见的失效方式：
    模型越多，共享的系统性偏差被平均得越少。

    Returns:
        ``mean_error_corr`` 为误差相关系数的非对角均值；
        ``lambda_implied`` 为由误差方差反推的 λ₁，可与
        :func:`estimate_reliability` 的估计对照。
    """
    x = _as_matrix(indicators)
    t = np.asarray(truth, dtype=float).ravel()
    if len(t) != x.shape[0]:
        raise ValueError("truth 长度必须等于样本数")

    errors = x - t[:, None]
    err_corr = correlation_matrix(errors)
    mean_ec = mean_pairwise_correlation(err_corr)

    var_theta = float(t.var(ddof=1))
    mean_err_var = float(errors.var(axis=0, ddof=1).mean())
    lam_implied = var_theta / (var_theta + mean_err_var) if var_theta > 0 else float("nan")

    return {
        "mean_error_corr": float(mean_ec),
        "lambda_implied_by_error_var": float(lam_implied),
        "mean_error_var": mean_err_var,
        "var_theta": var_theta,
        "independence_assumption_ok": bool(mean_ec < 0.15),
    }


__all__ = [
    "ReliabilityEstimate",
    "correlation_matrix",
    "cronbach_alpha",
    "error_correlation",
    "estimate_reliability",
    "mean_pairwise_correlation",
    "spearman_brown",
]

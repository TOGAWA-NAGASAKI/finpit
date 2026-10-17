"""衰减偏误及其修正。

## 单变量：系数被系统性地压向 0

真实模型 ``y = α + β·θ + ε``，而回归里用的是有噪声的 ``x = θ + u``：

    β̂ → β · λ,   λ = 可靠性

注意这不是随机误差，而是**系统性偏差**：样本再多也消不掉，
只会让 ``β̂`` 越来越稳定地收敛到 ``β·λ`` 这个错的值。
``λ = 0.5`` 时，一个真实效应为 0.10 的因子会被报成 0.05 —— 看起来"没效果"。

修正很直接：``β̂ / λ̂``。难点全在 ``λ`` 的识别（见 :mod:`.reliability`）。

## 多变量：真正的麻烦在于符号会翻转

当模型里有多个相关的自变量、而**其中一部分**带测量误差时，误差不只压低
它自己的系数，还会**污染其他系数** —— 方向取决于符号组合，完全可以翻转。
这是 `arXiv:2609.31013` 观察到的"下游回归系数符号都会变"的机制。

概率极限由经典公式给出（Levi 1973 / Garber & Klepper 1980）：

    plim(β̂) = (Σ_** + Σ_uu)^{-1} Σ_** β

其中 ``Σ_**`` 是真实自变量的协方差、``Σ_uu`` 是测量误差协方差。
:func:`plim_coefficients` 直接实现它，:func:`sign_flip_report` 用来
逐个系数判断符号是否被翻转。

**实践含义**：若你的回归里既有 LLM 生成的变量、又有传统变量，
那么传统变量的系数也是不可信的 —— 而这一点在论文里几乎从不被讨论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from finpit.eiv.reliability import estimate_reliability, spearman_brown


# --------------------------------------------------------------- 单变量修正
def disattenuate_slope(beta_observed: float, reliability: float) -> float:
    """把被衰减的斜率还原：``β = β̂ / λ``。"""
    if not 0.0 < reliability <= 1.0:
        raise ValueError(f"可靠性必须在 (0, 1] 内，收到 {reliability}")
    return beta_observed / reliability


def disattenuate_correlation(r_observed: float, rel_x: float, rel_y: float) -> float:
    """相关系数的衰减修正：``r = r̂ / sqrt(λ_x · λ_y)``。

    修正结果被裁剪到 ``[-1, 1]``。**若发生裁剪，不要忽略它** ——
    相关系数不可能是 1，出现裁剪说明至少一侧的可靠性被**低估**了
    （或该相关本身来自共同方法偏差而非真实关系）。这是一个有用的诊断信号，
    本函数保留裁剪以保证数值合法，调用方应自行检查是否触顶。
    """
    if not (0.0 < rel_x <= 1.0 and 0.0 < rel_y <= 1.0):
        raise ValueError("可靠性必须在 (0, 1] 内")
    corrected = r_observed / math.sqrt(rel_x * rel_y)
    return float(np.clip(corrected, -1.0, 1.0))


# --------------------------------------------------------------- 多变量偏误
def plim_coefficients(
    true_cov: np.ndarray,
    error_cov: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """测量误差下 OLS 系数的概率极限。

    ``plim(β̂) = (Σ_** + Σ_uu)^{-1} Σ_** β``

    Args:
        true_cov: 真实自变量的协方差矩阵 ``Σ_**``（p×p）。
        error_cov: 测量误差协方差 ``Σ_uu``（p×p，可只在对角部分非零）。
        beta: 真实系数向量。

    Returns:
        概率极限下的系数向量。
    """
    true_cov = np.asarray(true_cov, dtype=float)
    error_cov = np.asarray(error_cov, dtype=float)
    beta = np.asarray(beta, dtype=float).ravel()
    p = true_cov.shape[0]
    if true_cov.shape != (p, p) or error_cov.shape != (p, p):
        raise ValueError("协方差矩阵必须为方阵且维度一致")
    if len(beta) != p:
        raise ValueError("beta 长度必须与协方差维度一致")
    sigma_x = true_cov + error_cov
    try:
        return np.linalg.solve(sigma_x, true_cov @ beta)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - 病态输入
        raise ValueError(f"协方差矩阵奇异，无法求解：{exc}") from exc


@dataclass(slots=True)
class SignFlipReport:
    """逐个系数的符号翻转诊断。"""

    names: list[str]
    beta_true: np.ndarray
    beta_plim: np.ndarray
    flipped: list[str] = field(default_factory=list)
    attenuated: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["系数诊断（真实 → 测量误差下的概率极限）："]
        for i, name in enumerate(self.names):
            b, p = float(self.beta_true[i]), float(self.beta_plim[i])
            flag = "  ← 符号翻转" if name in self.flipped else ""
            ratio = f"{p / b:+.2f}×" if b != 0 else "—"
            lines.append(f"  {name:16} {b:+.4f} → {p:+.4f}   ({ratio}){flag}")
        if self.flipped:
            lines.append(
                "  警告："
                + "、".join(self.flipped)
                + " 的系数符号被测量误差翻转 —— 结论方向会反。"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": self.names,
            "beta_true": [round(float(v), 5) for v in self.beta_true],
            "beta_plim": [round(float(v), 5) for v in self.beta_plim],
            "flipped": self.flipped,
            "attenuation_ratio": {
                n: round(float(self.beta_plim[i] / self.beta_true[i]), 5)
                for i, n in enumerate(self.names)
                if self.beta_true[i] != 0
            },
        }


def sign_flip_report(
    true_cov: np.ndarray,
    error_cov: np.ndarray,
    beta: np.ndarray,
    names: list[str] | None = None,
) -> SignFlipReport:
    """判断哪些系数的符号被测量误差翻转。"""
    beta = np.asarray(beta, dtype=float).ravel()
    p = len(beta)
    names = names or [f"x{i}" for i in range(p)]
    plim = plim_coefficients(true_cov, error_cov, beta)

    flipped = [
        names[i]
        for i in range(p)
        if beta[i] != 0 and np.sign(plim[i]) != np.sign(beta[i]) and abs(plim[i]) > 1e-12
    ]
    attenuated = {
        names[i]: float(plim[i] / beta[i]) for i in range(p) if beta[i] != 0
    }
    return SignFlipReport(
        names=names,
        beta_true=beta,
        beta_plim=plim,
        flipped=flipped,
        attenuated=attenuated,
    )


# ------------------------------------------------------------- 完整修正流程
@dataclass(slots=True)
class AttenuationResult:
    """一次「估计 → 修正」的完整结果。"""

    n: int
    k: int
    beta_observed: float
    beta_corrected: float
    se_observed: float
    se_corrected: float
    lambda_single: float
    lambda_mean: float
    ci_observed: tuple[float, float] = (float("nan"), float("nan"))
    ci_corrected: tuple[float, float] = (float("nan"), float("nan"))
    beta_true: float | None = None

    @property
    def inflation(self) -> float:
        """修正相对原始估计放大了多少倍。"""
        if self.beta_observed == 0:
            return float("nan")
        return self.beta_corrected / self.beta_observed

    @property
    def ci_excludes_zero_observed(self) -> bool:
        lo, hi = self.ci_observed
        return math.isfinite(lo) and (lo > 0 or hi < 0)

    @property
    def ci_excludes_zero_corrected(self) -> bool:
        lo, hi = self.ci_corrected
        return math.isfinite(lo) and (lo > 0 or hi < 0)

    def summary(self) -> str:
        lines = [
            f"衰减修正（n={self.n}, 指标数 K={self.k}）",
            f"  λ₁ = {self.lambda_single:.4f}   λ_K = {self.lambda_mean:.4f}",
            f"  原始 β̂      = {self.beta_observed:+.5f}  (SE {self.se_observed:.5f})",
            f"  修正 β̂/λ    = {self.beta_corrected:+.5f}  (SE {self.se_corrected:.5f})"
            f"   [放大 {self.inflation:.2f}×]",
        ]
        if self.beta_true is not None:
            lines.append(f"  真实 β      = {self.beta_true:+.5f}")
            lines.append(
                f"  原始相对误差 = "
                f"{abs(self.beta_observed - self.beta_true) / abs(self.beta_true):.1%}"
                f"   修正后相对误差 = "
                f"{abs(self.beta_corrected - self.beta_true) / abs(self.beta_true):.1%}"
            )
        if math.isfinite(self.ci_corrected[0]):
            lines.append(
                f"  95% 区间：原始 [{self.ci_observed[0]:+.5f}, {self.ci_observed[1]:+.5f}]"
                f"  修正 [{self.ci_corrected[0]:+.5f}, {self.ci_corrected[1]:+.5f}]"
            )
            if not self.ci_excludes_zero_observed and self.ci_excludes_zero_corrected:
                lines.append(
                    "  注意：修正前区间跨 0（看起来「无效应」），修正后才排除 0 ——"
                    " 这正是 LLM 因子被误判为无效的典型情形。"
                )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "k": self.k,
            "lambda_single": round(self.lambda_single, 5),
            "lambda_mean": round(self.lambda_mean, 5),
            "beta_observed": round(self.beta_observed, 6),
            "beta_corrected": round(self.beta_corrected, 6),
            "beta_true": None if self.beta_true is None else round(self.beta_true, 6),
            "se_observed": round(self.se_observed, 6),
            "se_corrected": round(self.se_corrected, 6),
            "inflation": None if not math.isfinite(self.inflation) else round(self.inflation, 4),
            "ci95_observed": [round(v, 6) for v in self.ci_observed],
            "ci95_corrected": [round(v, 6) for v in self.ci_corrected],
        }


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """一元 OLS，返回 ``(斜率, 斜率标准误)``。"""
    n = len(y)
    xc = x - x.mean()
    sxx = float((xc**2).sum())
    if sxx <= 0:
        return float("nan"), float("nan")
    beta = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - (y.mean() + beta * xc)
    dof = max(n - 2, 1)
    sigma2 = float((resid**2).sum() / dof)
    se = math.sqrt(sigma2 / sxx) if sigma2 > 0 else float("nan")
    return beta, se


def corrected_regression(
    y: np.ndarray | list,
    indicators: np.ndarray | list,
    *,
    bootstrap: int = 0,
    seed: int = 20261016,
    beta_true: float | None = None,
) -> AttenuationResult:
    """对 ``y ~ mean(indicators)`` 做衰减修正。

    修正量取自 K 个平行指标估计出的可靠性。Bootstrap 会**同时**重抽样
    斜率与可靠性，因此得到的区间涵盖了两种不确定性 —— 只重抽样其一会
    严重低估修正后系数的误差。

    Args:
        y: 因变量。
        indicators: ``(n, K)`` 平行指标，回归里用其行均值。
        bootstrap: 重抽样次数，0 表示不算区间。
        beta_true: 若已知真值（受控实验），一并报告相对误差。
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(indicators, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(y):
        raise ValueError("indicators 必须是 (n_samples, k) 且与 y 等长")
    if x.shape[1] < 2:
        raise ValueError("修正至少需要 2 个平行指标来识别可靠性")

    rel = estimate_reliability(x)
    mean_score = x.mean(axis=1)
    beta_hat, se_hat = _ols(y, mean_score)
    beta_corr = disattenuate_slope(beta_hat, rel.lambda_mean)

    result = AttenuationResult(
        n=len(y),
        k=x.shape[1],
        beta_observed=beta_hat,
        beta_corrected=beta_corr,
        se_observed=se_hat,
        se_corrected=float("nan"),
        lambda_single=rel.lambda_single,
        lambda_mean=rel.lambda_mean,
        beta_true=beta_true,
    )

    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        obs = np.empty(bootstrap)
        cor = np.empty(bootstrap)
        for b in range(bootstrap):
            idx = rng.integers(0, len(y), len(y))
            yb, xb = y[idx], x[idx]
            c = np.corrcoef(xb, rowvar=False)
            k = xb.shape[1]
            lam1 = float((c.sum() - np.trace(c)) / (k * (k - 1)))
            lam_k = spearman_brown(lam1, k)
            bb, _ = _ols(yb, xb.mean(axis=1))
            obs[b] = bb
            cor[b] = bb / lam_k if 0 < lam_k <= 1 else np.nan

        for arr, target in ((obs, "obs"), (cor, "cor")):
            finite = arr[np.isfinite(arr)]
            if len(finite) < bootstrap // 2:
                continue
            lo, hi = (float(v) for v in np.percentile(finite, [2.5, 97.5]))
            if target == "obs":
                result.ci_observed = (lo, hi)
            else:
                result.ci_corrected = (lo, hi)
        finite_cor = cor[np.isfinite(cor)]
        if len(finite_cor) > 1:
            result.se_corrected = float(finite_cor.std(ddof=1))

    return result


__all__ = [
    "AttenuationResult",
    "SignFlipReport",
    "corrected_regression",
    "disattenuate_correlation",
    "disattenuate_slope",
    "plim_coefficients",
    "sign_flip_report",
]

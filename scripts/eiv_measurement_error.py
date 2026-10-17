"""LLM 生成变量的测量误差：受控实验（不需要任何 API key）。

本脚本产出四个结果：

1. **单变量衰减**：不同可靠性下，报告系数被压低多少；
2. **多模型的价值与上限**：以文献报告的跨模型秩相关 0.52 为锚，
   看 K 个模型能把可靠性提到多少、需要几个模型、边际收益何时消失；
3. **多变量符号翻转**：一个 LLM 变量带测量误差时，如何污染**另一个**
   传统变量的系数并翻转其符号；
4. **前提失效**：当各模型误差含共享成分时，Spearman-Brown **高估**可靠性，
   修正量不足 —— 且共享偏差**无法通过多加模型消除**。

用法::

    python scripts/eiv_measurement_error.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from finpit.eiv import (  # noqa: E402
    corrected_regression,
    error_correlation,
    estimate_reliability,
    sign_flip_report,
    spearman_brown,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "eiv_measurement_error.json"
RESULTS: dict[str, object] = {}

# 文献锚点：arXiv:2609.31013 报告跨模型秩相关仅 0.52
LITERATURE_LAMBDA = 0.52


def hr(title: str) -> None:
    print("\n" + "=" * 86)
    print(title)
    print("=" * 86)


def simulate(
    *,
    n: int,
    k: int,
    beta_true: float,
    lambda_true: float,
    shared_share: float = 0.0,
    y_noise: float = 0.5,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造受控数据。

    ``shared_share`` 为误差方差中**共享成分**（所有模型共有）的占比。
    0 表示误差独立（Spearman-Brown 的前提成立）。
    """
    rng = np.random.default_rng(seed)
    var_theta = 1.0
    total_err = var_theta * (1.0 - lambda_true) / lambda_true
    var_shared = total_err * shared_share
    var_idio = total_err - var_shared

    theta = rng.standard_normal(n) * math.sqrt(var_theta)
    y = beta_true * theta + rng.standard_normal(n) * y_noise

    # 共享成分必须显式定型为 (n, 1)：
    # 若留成 (n,)，与 theta[:, None] 相加会广播成 (n, n) 并耗尽内存。
    if var_shared > 0:
        shared = (rng.standard_normal(n) * math.sqrt(var_shared))[:, None]
    else:
        shared = 0.0
    ind = (
        theta[:, None]
        + shared
        + rng.standard_normal((n, k)) * math.sqrt(var_idio)
    )
    return y, theta, ind


def true_reliability_of_mean(lambda1: float, k: int, shared_share: float) -> float:
    """在存在共享偏差时，K 指标均值的**真实**可靠性。

    误差方差 = v_s + v_i，其中 v_s 为共享部分。均值误差方差 = v_s + v_i/K。
    K→∞ 时可靠性被卡在 v_θ/(v_θ+v_s) —— **共享偏差无法被平均消除**。
    """
    var_theta = 1.0
    total_err = var_theta * (1.0 - lambda1) / lambda1
    var_shared = total_err * shared_share
    var_idio = total_err - var_shared
    return var_theta / (var_theta + var_shared + var_idio / k)


# ============================================================ 1) 单变量衰减
hr("1) 单变量衰减：LLM 分数当自变量时，系数被压低多少")

beta_true = 0.10
print(f"真实效应 β = {beta_true}；n = 20,000\n")
print(f"{'λ(可靠性)':>10}{'β̂ 观测':>12}{'β̂/λ 修正':>14}{'原始误差':>12}{'修正后误差':>14}")
attenuation_rows = []
for lam in (0.30, 0.40, 0.52, 0.65, 0.80, 0.95):
    y, _, ind = simulate(n=20000, k=1, beta_true=beta_true, lambda_true=lam, seed=101)
    from finpit.eiv.attenuation import _ols

    beta_hat, _ = _ols(y, ind[:, 0])
    beta_fix = beta_hat / lam
    err_raw = abs(beta_hat - beta_true) / beta_true
    err_fix = abs(beta_fix - beta_true) / beta_true
    print(f"{lam:>10.2f}{beta_hat:>12.5f}{beta_fix:>14.5f}{err_raw:>11.1%}{err_fix:>13.1%}")
    attenuation_rows.append(
        {
            "lambda": lam,
            "beta_observed": round(beta_hat, 6),
            "beta_corrected": round(beta_fix, 6),
            "rel_error_observed": round(err_raw, 4),
            "rel_error_corrected": round(err_fix, 4),
        }
    )
RESULTS["attenuation_by_lambda"] = attenuation_rows
print(
    f"\n=> λ = {LITERATURE_LAMBDA}（文献报告的跨模型秩相关）时，"
    f"一个真实效应为 {beta_true} 的因子会被报成 {beta_true * LITERATURE_LAMBDA:.5f}，"
    "**恰好一半**。"
)

# ============================================================ 2) 多模型价值
hr("2) 多模型的价值与天花板（锚定文献 λ₁ = 0.52）")

print(f"{'K':>4}{'λ_K (Spearman-Brown)':>24}{'噪声占比':>12}{'相对 K=1 增益':>16}")
lam_rows = []
base = LITERATURE_LAMBDA
for k in (1, 2, 3, 5, 8, 10, 20, 50):
    lam_k = spearman_brown(base, k)
    lam_rows.append({"k": k, "lambda_k": round(lam_k, 5)})
    gain = lam_k / base
    print(f"{k:>4}{lam_k:>24.4f}{1 - lam_k:>11.1%}{gain:>15.2f}×")
RESULTS["lambda_by_k"] = lam_rows

est = None
y, theta, ind = simulate(n=20000, k=6, beta_true=0.10, lambda_true=base, seed=103)
est = estimate_reliability(ind)
need = est.needed_k(0.90)
print(
    f"\n由上表：\n"
    f"  用 1 个模型      → λ = {base:.2f}，只拿到真实效应的 {base:.0%}\n"
    f"  用 3 个模型      → λ = {spearman_brown(base, 3):.3f}\n"
    f"  达到 λ = 0.90 需要 **约 {need} 个模型**\n"
    f"  从 10 个加到 50 个，λ 仅从 {spearman_brown(base, 10):.3f} 升到 "
    f"{spearman_brown(base, 50):.3f} —— 边际收益早已消失"
)
RESULTS["models_needed_for_090"] = need

# ============================================================ 3) 符号翻转
hr("3) 多变量符号翻转：LLM 变量的测量误差如何污染传统变量的系数")

print("设定：y ~ β₁·LLM情绪 + β₂·传统因子，两者真实相关 ρ = 0.8\n")
for rho, var_u in ((0.8, 1.5), (0.8, 0.6), (-0.7, 1.5)):
    true_cov = np.array([[1.0, rho], [rho, 1.0]])
    err_cov = np.array([[var_u, 0.0], [0.0, 0.0]])
    beta = np.array([0.10, 0.05])
    rep = sign_flip_report(true_cov, err_cov, beta, names=["llm_sentiment", "traditional"])
    print(f"ρ={rho:+.1f}, 误差方差={var_u:.1f}")
    print("  " + rep.summary().replace("\n", "\n  "))
    print()
RESULTS["sign_flip_examples"] = []
for rho, var_u in ((0.8, 1.5), (0.8, 0.6), (-0.7, 1.5)):
    true_cov = np.array([[1.0, rho], [rho, 1.0]])
    err_cov = np.array([[var_u, 0.0], [0.0, 0.0]])
    rep = sign_flip_report(true_cov, err_cov, np.array([0.10, 0.05]),
                           names=["llm_sentiment", "traditional"])
    RESULTS["sign_flip_examples"].append(
        {"rho": rho, "error_var": var_u, **rep.as_dict()}
    )

print(
    "=> 传统变量本身没有任何测量误差，其系数依然被污染。\n"
    "   含义：只要回归里混入了 LLM 生成的变量，**所有**系数的解释都必须打折扣，\n"
    "   而这一点在实证论文里几乎从不被讨论。"
)

# ==================================================== 4) 前提失效：共享偏差
hr("4) 前提失效：模型误差含共享成分时，Spearman-Brown 会高估可靠性")

print("设定：误差 = 共享成分 c（所有模型共有）+ 独立成分 e")
print("      真实 λ₁ = 0.50，共享占比 = 50%\n")
print(
    f"{'K':>4}{'λ̂ (两两相关,会高估)':>22}{'λ_K (Spearman-Brown)':>22}"
    f"{'λ 真实':>12}{'高估幅度':>12}"
)
shared_rows = []
for k in (2, 3, 5, 10, 50, 200):
    y2, th2, ind2 = simulate(n=20000, k=k, beta_true=0.10, lambda_true=0.50,
                             shared_share=0.5, seed=107)
    e = estimate_reliability(ind2)
    lam_true_k = true_reliability_of_mean(0.50, k, 0.5)
    over = e.lambda_mean / lam_true_k - 1.0
    shared_rows.append(
        {
            "k": k,
            "lambda_hat_single": round(e.lambda_single, 4),
            "lambda_hat_mean": round(e.lambda_mean, 4),
            "lambda_true_mean": round(lam_true_k, 4),
            "overstatement": round(over, 4),
        }
    )
    print(
        f"{k:>4}{e.lambda_single:>22.4f}{e.lambda_mean:>22.4f}"
        f"{lam_true_k:>12.4f}{over:>11.1%}"
    )
RESULTS["shared_bias"] = shared_rows

print(
    f"\n  两两相关给出 λ̂₁ ≈ {shared_rows[-1]['lambda_hat_single']:.2f}，"
    f"而真实 λ₁ = 0.50（因为它把共享偏差也算成了「共同信号」）"
)
print(
    f"  K→∞ 时 Spearman-Brown 预测 λ → "
    f"{spearman_brown(shared_rows[-1]['lambda_hat_single'], 10**6):.3f}，"
    f"但真实可靠性被共享偏差卡死在 {true_reliability_of_mean(0.50, 10**9, 0.5):.3f}"
)

# 修正不足的量化
print("\n  后果 —— 修正量不足：")
y3, th3, ind3 = simulate(n=40000, k=10, beta_true=0.10, lambda_true=0.50,
                         shared_share=0.5, seed=109)
res = corrected_regression(y3, ind3, bootstrap=200, beta_true=0.10)
lam_true_k = true_reliability_of_mean(0.50, 10, 0.5)
beta_honest = res.beta_observed / lam_true_k
print(f"    观测 β̂                = {res.beta_observed:+.5f}")
print(f"    用 λ̂ (被高估) 修正     = {res.beta_corrected:+.5f}   ← 本模块默认输出")
print(f"    用真实 λ 修正          = {beta_honest:+.5f}")
print(f"    真值                   = {0.10:+.5f}")
print(
    f"    => 高估 λ 使修正后仍偏低 {(0.10 - res.beta_corrected) / 0.10:.1%}"
    "，看起来像「即便修正也无效」"
)
RESULTS["shared_bias_correction"] = {
    "beta_observed": round(res.beta_observed, 6),
    "beta_corrected_with_estimated_lambda": round(res.beta_corrected, 6),
    "beta_corrected_with_true_lambda": round(beta_honest, 6),
    "beta_true": 0.10,
}

diag = error_correlation(ind3, th3)
print(
    f"\n  诊断：误差平均相关 = {diag['mean_error_corr']:.3f} "
    f"({'> 0.15，独立性前提已失效' if not diag['independence_assumption_ok'] else 'OK'})"
)
print("  => 多模型集成前必须先查误差相关性；共享偏差**无法靠加模型消除**。")
RESULTS["error_correlation_diagnostic"] = {
    k: round(v, 5) if isinstance(v, float) else v for k, v in diag.items()
}

# ---------------------------------------------------------------------- 落盘
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n已写出: {OUT}")

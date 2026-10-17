"""Errors-in-variables：LLM 生成变量的测量误差及其经济含义。

## 为什么这值得单独一层

论文里常见的做法是：让 LLM 给文本打个情绪分，把分数当自变量跑回归，
然后报告系数与显著性。但 LLM 打出的分数**只是真实构念的含噪代理**：

    x = θ + u

这带来两个后果，且都不在标准误里体现：

1. **单变量衰减**：``β̂ → β·λ``，``λ`` 是可靠性。系数被系统性压向 0，
   样本再多也消不掉。
   `arXiv:2609.31013` 报告跨模型秩相关仅 **0.52** —— 意味着 ``λ`` 只有
   0.5 量级，真实效应会被报成一半大小。

2. **多变量符号翻转**：当一个自变量有测量误差、而自变量之间相关时，
   误差会污染**其他**系数，符号都可能翻转。所以"系数符号随模型而变"
   并非玄学，而是可以精确计算的偏误。

## 这一层做什么

* :mod:`.reliability` —— 用 K 个平行指标（K 个 LLM）**无需真值**地识别 ``λ``；
  并提供前提检验（误差是否真的独立）。
* :mod:`.attenuation` —— 修正系数、给出概率极限、诊断符号翻转。

## 与「泄漏」那一层的关系

`finpit.pit` 管的是**时间上不该看到的信息**；本层管的是**看得到但测不准**。
两者是金融 LLM 实证研究里最常被忽略的两个偏误来源，且互相独立：
一个干净的 PIT 流程仍可能因测量误差把真实效应报成零。
"""

from __future__ import annotations

from finpit.eiv.attenuation import (
    AttenuationResult,
    SignFlipReport,
    corrected_regression,
    disattenuate_correlation,
    disattenuate_slope,
    plim_coefficients,
    sign_flip_report,
)
from finpit.eiv.reliability import (
    ReliabilityEstimate,
    cronbach_alpha,
    error_correlation,
    estimate_reliability,
    mean_pairwise_correlation,
    spearman_brown,
)

__all__ = [
    "AttenuationResult",
    "ReliabilityEstimate",
    "SignFlipReport",
    "corrected_regression",
    "cronbach_alpha",
    "disattenuate_correlation",
    "disattenuate_slope",
    "error_correlation",
    "estimate_reliability",
    "mean_pairwise_correlation",
    "plim_coefficients",
    "sign_flip_report",
    "spearman_brown",
]

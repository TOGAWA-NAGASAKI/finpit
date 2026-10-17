"""多质多法（MTMM）分析 —— 无需真值即可分解构念方差与方法方差。

## 为什么需要它

上一节的发现是：5 个 LLM 彼此相关 0.97，却各自只与真实构念相关 0.72。
但那个实验**依赖已知真值**（合成语料的潜在情绪），无法在真实文本上重复。

MTMM 提供了绕开真值的路径。Campbell & Fiske (1959) 的洞见是：
**用多个方法测量多个构念**，仅凭相关矩阵的结构就能把两者分开。

设 ``s[c, k]`` 为「方法 k 对构念 c 的打分」，则相关矩阵里有三类格子：

============  ==========================  ==================================
类型           含义                         估计的是
============  ==========================  ==================================
MTHM          同构念、异方法                  **收敛效度**（构念信号）
HTMM          异构念、同方法                  **方法偏差**（晕轮效应）
HTHM          异构念、异方法                  基线（应接近 0）
============  ==========================  ==================================

## 核心判据

Campbell-Fiske 第三条（最关键）：**MTHM 应当大于 HTMM** —— 即"同一构念用不同方法测"
应当比"同一方法测不同构念"更一致。

若这条**不成立**（HTMM ≥ MTHM），说明方法方差压过构念方差：
那些"构念"根本没有被区分开，分数主要反映的是**这个模型怎么说话**，
而不是**文本讲了什么**。

## 与真值实验的关系

真值实验能给出「可靠性」的绝对值；MTMM 只能给出**相对**分解。
两者互补：前者说"有多少是噪声"，后者说"剩下的里面有多少是构念、多少是方法"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np


def _as_cube(scores: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"打分必须是三维 (n_texts, n_constructs, n_methods)，收到 {arr.shape}"
        )
    n_t, n_c, n_k = arr.shape
    if n_c < 2:
        raise ValueError("MTMM 至少需要 2 个构念（否则无法区分构念与方法）")
    if n_k < 2:
        raise ValueError("MTMM 至少需要 2 个方法（模型）")
    if n_t < 10:
        raise ValueError("样本太少，相关系数不稳定")
    return arr


@dataclass(slots=True)
class PairDiagnostic:
    """一对构念的 MTMM 诊断。

    **为什么需要逐配对看**：把三类配对各自取均值会**掩盖结构**。
    实测真实 10-K 文本上，某对构念的同方法相关高达 0.80，另两对却是 −0.3，
    平均之后变成 +0.05 —— 看上去"没有方法偏差"，实际上有一对严重混杂。

    ``method_excess``（同方法 − 异方法）才是该配对是否存在方法假象的判据：
    若显著为正，说明"这两者相关"部分只是因为**同一个模型**在打分。
    """

    c1: str
    c2: str
    htmm: float
    """同方法下这两个构念的相关（各方法平均）。"""

    hthm: float
    """不同方法下这两个构念的相关。"""

    @property
    def method_excess(self) -> float:
        return self.htmm - self.hthm

    @property
    def confounded(self) -> bool:
        """同方法相关是否显著高于异方法 —— 即存在方法假象。"""
        return self.method_excess > 0.15

    def line(self) -> str:
        tag = "  ← 存在方法假象" if self.confounded else ""
        return (
            f"    {self.c1:16} ↔ {self.c2:16} 同方法 {self.htmm:+.3f}  "
            f"异方法 {self.hthm:+.3f}  差值 {self.method_excess:+.3f}{tag}"
        )


@dataclass(slots=True)
class MTMMResult:
    """MTMM 分析结果。"""

    n_texts: int
    n_constructs: int
    n_methods: int
    construct_names: list[str]
    method_names: list[str]

    convergent: float
    """MTHM 均值：同构念、异方法的相关 —— 收敛效度。"""

    method_inflated: float
    """HTMM 均值：异构念、同方法的相关 —— 方法（晕轮）偏差。"""

    baseline: float
    """HTHM 均值：异构念、异方法的相关。"""

    mthm_values: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    htmm_values: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    hthm_values: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    pairs: list[PairDiagnostic] = field(default_factory=list, repr=False)

    def pair_breakdown(self) -> str:
        """逐配对诊断 —— 均值会掩盖结构，这个才是要看的东西。"""
        if not self.pairs:
            return "    （无逐配对信息）"
        lines = [d.line() for d in self.pairs]
        bad = [d for d in self.pairs if d.confounded]
        if bad:
            lines.append(
                "    => "
                + "、".join(f"{d.c1}↔{d.c2}" for d in bad)
                + " 的相关主要来自**同一模型**而非文本内容"
            )
        else:
            lines.append("    => 无配对存在明显方法假象")
        return "\n".join(lines)

    @property
    def method_excess(self) -> float:
        """方法带来的额外相关 = HTMM − HTHM。

        这是**纯粹由"同一个模型"贡献**的那部分一致性。
        """
        return self.method_inflated - self.baseline

    @property
    def trait_excess(self) -> float:
        """构念带来的额外相关 = MTHM − HTHM。"""
        return self.convergent - self.baseline

    @property
    def trait_share(self) -> float:
        """在"超出基线的相关"里，构念贡献的占比。"""
        total = self.trait_excess + self.method_excess
        if total <= 0:
            return float("nan")
        return self.trait_excess / total

    # ------------------------------------------------- Campbell-Fiske 判据
    @property
    def criterion_convergent(self) -> bool:
        """判据一：MTHM 全为正且足够大（收敛效度存在）。"""
        return bool(self.mthm_values.size and self.mthm_values.min() > 0.2)

    @property
    def criterion_discriminant(self) -> bool:
        """判据二：MTHM 总体上大于 HTHM（构念可区分于噪声）。"""
        return self.convergent > self.baseline

    @property
    def criterion_trait_over_method(self) -> bool:
        """判据三（最关键）：MTHM > HTMM，即构念方差压过方法方差。"""
        return self.convergent > self.method_inflated

    @property
    def verdict(self) -> str:
        if self.criterion_trait_over_method:
            return "构念方差占优 —— 分数主要反映文本内容"
        if self.method_inflated > self.baseline + 0.1:
            return "**方法方差占优** —— 分数主要反映「这个模型怎么说话」，构念未被区分"
        return "构念与方法方差相当，结论需谨慎"

    def summary(self) -> str:
        flag = "通过" if self.criterion_trait_over_method else "**未通过**"
        lines = [
            f"MTMM 分析（{self.n_texts} 条文本 × {self.n_constructs} 构念 × "
            f"{self.n_methods} 模型）",
            f"  MTHM 同构念异方法（收敛效度）= {self.convergent:+.4f}",
            f"  HTMM 异构念同方法（方法偏差）= {self.method_inflated:+.4f}",
            f"  HTHM 异构念异方法（基线）    = {self.baseline:+.4f}",
            "",
            f"  构念增量（MTHM − HTHM）      = {self.trait_excess:+.4f}",
            f"  方法增量（HTMM − HTHM）      = {self.method_excess:+.4f}",
        ]
        if math.isfinite(self.trait_share):
            lines.append(f"  => 超出基线的相关中，构念占 {self.trait_share:.1%}")
        lines += [
            "",
            "  Campbell-Fiske 判据：",
            f"    ① 收敛效度存在       "
            f"{'通过' if self.criterion_convergent else '未通过'}"
            f"  (min MTHM = {self.mthm_values.min():+.3f})",
            f"    ② 构念可区分于噪声   "
            f"{'通过' if self.criterion_discriminant else '未通过'}",
            f"    ③ 构念 > 方法        {flag}",
            f"  结论：{self.verdict}",
            "",
            "  逐配对诊断（均值会掩盖结构，这一节才关键）：",
            self.pair_breakdown(),
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_texts": self.n_texts,
            "n_constructs": self.n_constructs,
            "n_methods": self.n_methods,
            "constructs": self.construct_names,
            "methods": self.method_names,
            "mthm_convergent": round(self.convergent, 5),
            "htmm_method_inflated": round(self.method_inflated, 5),
            "hthm_baseline": round(self.baseline, 5),
            "trait_excess": round(self.trait_excess, 5),
            "method_excess": round(self.method_excess, 5),
            "trait_share": (
                None if not math.isfinite(self.trait_share) else round(self.trait_share, 5)
            ),
            "criteria": {
                "convergent": self.criterion_convergent,
                "discriminant": self.criterion_discriminant,
                "trait_over_method": self.criterion_trait_over_method,
            },
            "verdict": self.verdict,
            "mthm_min": round(float(self.mthm_values.min()), 5) if self.mthm_values.size else None,
            "mthm_max": round(float(self.mthm_values.max()), 5) if self.mthm_values.size else None,
            "htmm_max": round(float(self.htmm_values.max()), 5) if self.htmm_values.size else None,
            "pairs": [
                {
                    "c1": d.c1,
                    "c2": d.c2,
                    "htmm_same_method": round(d.htmm, 5),
                    "hthm_diff_method": round(d.hthm, 5),
                    "method_excess": round(d.method_excess, 5),
                    "confounded": d.confounded,
                }
                for d in self.pairs
            ],
            "confounded_pairs": [f"{d.c1}↔{d.c2}" for d in self.pairs if d.confounded],
        }


def mtmm_analyze(
    scores: np.ndarray | list,
    *,
    construct_names: list[str] | None = None,
    method_names: list[str] | None = None,
) -> MTMMResult:
    """对 ``(n_texts, n_constructs, n_methods)`` 的打分做 MTMM 分析。"""
    cube = _as_cube(scores)
    n_t, n_c, n_k = cube.shape
    construct_names = construct_names or [f"C{i}" for i in range(n_c)]
    method_names = method_names or [f"M{i}" for i in range(n_k)]
    if len(construct_names) != n_c or len(method_names) != n_k:
        raise ValueError("名称数量与维度不符")

    # 展平成 (n_texts, n_c * n_k)，列索引 = c * n_k + k
    flat = cube.reshape(n_t, n_c * n_k)
    corr = np.corrcoef(flat, rowvar=False)

    def col(c: int, k: int) -> int:
        return c * n_k + k

    mthm, htmm, hthm = [], [], []
    # MTHM：同一构念、不同方法 —— 收敛效度
    for c in range(n_c):
        for k1, k2 in combinations(range(n_k), 2):
            mthm.append(corr[col(c, k1), col(c, k2)])
    # HTMM：不同构念、同一方法 —— 方法（晕轮）偏差
    # 注意：每个方法只应计一次。早期实现在方法对上循环，导致每个方法被
    # 重复计入 (n_k - 1) 次 —— 均值虽然不变（各方法等权），但配对计数是错的。
    for c1, c2 in combinations(range(n_c), 2):
        for k in range(n_k):
            htmm.append(corr[col(c1, k), col(c2, k)])
    # HTHM：不同构念、不同方法 —— 基线
    for c1, c2 in combinations(range(n_c), 2):
        for k1, k2 in combinations(range(n_k), 2):
            hthm.append(corr[col(c1, k1), col(c2, k2)])
            hthm.append(corr[col(c1, k2), col(c2, k1)])

    # 逐配对诊断 —— 均值会掩盖"某一对严重混杂"这种结构
    pairs: list[PairDiagnostic] = []
    for c1, c2 in combinations(range(n_c), 2):
        same = [corr[col(c1, k), col(c2, k)] for k in range(n_k)]
        diff = [
            corr[col(c1, k1), col(c2, k2)]
            for k1, k2 in combinations(range(n_k), 2)
        ] + [
            corr[col(c1, k2), col(c2, k1)]
            for k1, k2 in combinations(range(n_k), 2)
        ]
        pairs.append(
            PairDiagnostic(
                c1=construct_names[c1],
                c2=construct_names[c2],
                htmm=float(np.mean(same)),
                hthm=float(np.mean(diff)),
            )
        )

    return MTMMResult(
        n_texts=n_t,
        n_constructs=n_c,
        n_methods=n_k,
        construct_names=list(construct_names),
        method_names=list(method_names),
        convergent=float(np.mean(mthm)),
        method_inflated=float(np.mean(htmm)),
        baseline=float(np.mean(hthm)),
        mthm_values=np.asarray(mthm),
        htmm_values=np.asarray(htmm),
        hthm_values=np.asarray(hthm),
        pairs=pairs,
    )


__all__ = ["MTMMResult", "PairDiagnostic", "mtmm_analyze"]

"""预测性能指标。

金融信号评测里指标的选择本身就是个陷阱：用 MSE 评一个 IC=0.03 的信号，
得到的结论和用 IC 评完全不同。本模块统一采用**排序类**指标，因为它们
对厚尾与量纲不敏感，也更贴近真实组合构建的用法。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_array_pair(pred: np.ndarray | list[float], actual: np.ndarray | list[float]):
    p = np.asarray(pred, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        raise ValueError(f"长度不一致: pred={p.shape} actual={a.shape}")
    mask = np.isfinite(p) & np.isfinite(a)
    return p[mask], a[mask]


def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列），等价于 scipy.stats.rankdata 的 'average'。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # 处理并列：同值取平均秩
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = (i + j + 1) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def information_coefficient(pred, actual) -> float:
    """Pearson IC：预测值与实际收益的线性相关系数。"""
    p, a = _as_array_pair(pred, actual)
    if len(p) < 3 or p.std() == 0 or a.std() == 0:
        return float("nan")
    return float(np.corrcoef(p, a)[0, 1])


def rank_ic(pred, actual) -> float:
    """Spearman 秩相关。对厚尾稳健，是业界报 IC 的默认口径。"""
    p, a = _as_array_pair(pred, actual)
    if len(p) < 3:
        return float("nan")
    rp, ra = _rankdata(p), _rankdata(a)
    if rp.std() == 0 or ra.std() == 0:
        return float("nan")
    return float(np.corrcoef(rp, ra)[0, 1])


def hit_rate(pred, actual) -> float:
    """方向命中率：符号一致的比例。0.5 即无预测力。"""
    p, a = _as_array_pair(pred, actual)
    if len(p) == 0:
        return float("nan")
    nonzero = (p != 0) & (a != 0)
    if nonzero.sum() == 0:
        return float("nan")
    return float((np.sign(p[nonzero]) == np.sign(a[nonzero])).mean())


def quantile_spread(pred, actual, *, n_buckets: int = 5) -> float:
    """最高分位与最低分位的平均实际收益之差（多空组合收益）。

    这是最贴近"能不能赚到钱"的指标：IC 高但分位差为 0 是可能的，
    说明信号只在中间区域有效而两端无用。
    """
    p, a = _as_array_pair(pred, actual)
    if len(p) < n_buckets * 2:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    size = len(p) // n_buckets
    if size == 0:
        return float("nan")
    low = a[order[:size]].mean()
    high = a[order[-size:]].mean()
    return float(high - low)


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    """一组信号的完整评估结果。

    刻意同时记录**相关性指标**（IC/RankIC）与**水平指标**（mean_return），
    因为不同泄漏污染的维度不同：未来函数抬高相关性，幸存者偏差抬高收益水平。
    只看其中一个，就会漏掉另一类问题。
    """

    ic: float
    rank_ic: float
    hit_rate: float
    quantile_spread: float
    mean_return: float
    n_obs: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "ic": round(self.ic, 5),
            "rank_ic": round(self.rank_ic, 5),
            "hit_rate": round(self.hit_rate, 5),
            "quantile_spread": round(self.quantile_spread, 6),
            "mean_return": round(self.mean_return, 6),
            "n_obs": self.n_obs,
        }

    def line(self) -> str:
        return (
            f"IC={self.ic:+.4f}  RankIC={self.rank_ic:+.4f}  "
            f"命中率={self.hit_rate:.4f}  分位差={self.quantile_spread:+.5f}  n={self.n_obs}"
        )


def evaluate(pred, actual, *, n_buckets: int = 5) -> SignalMetrics:
    """一次性算出全部指标。"""
    p, a = _as_array_pair(pred, actual)
    return SignalMetrics(
        ic=information_coefficient(p, a),
        rank_ic=rank_ic(p, a),
        hit_rate=hit_rate(p, a),
        quantile_spread=quantile_spread(p, a, n_buckets=n_buckets),
        mean_return=float(a.mean()) if len(a) else float("nan"),
        n_obs=int(len(a)),
    )


__all__ = [
    "SignalMetrics",
    "evaluate",
    "hit_rate",
    "information_coefficient",
    "quantile_spread",
    "rank_ic",
]

"""切分泄漏审计（split audit）。

## 问题

机器学习里最常见、也最容易被忽视的泄漏不是未来函数，而是**随机切分**。
把时间序列样本随机分成训练/测试集，会让测试样本的"邻居"（前后几天、
同一事件的后续报道）落进训练集，于是模型学到的是**记忆**而非**泛化**。

这在金融文本上尤其致命：同一条新闻会被多家媒体在连续几天内反复改写，
随机切分几乎必然把改写版本分到训练集。

## 本模块做什么

对**同一份特征、同一个模型**，分别用随机切分与时序切分评测，报出虚高倍数：

    虚高倍数 = 随机切分得分 / 时序切分得分

论文 `arXiv:2608.17223` 在金融新闻 NLP 上测得 MCC 虚高 **1.1–6.5×**，
且发现模型容量越大虚高越严重 —— 本模块复现该结论。

## 实现说明

刻意只用 numpy 实现（哈希字符二元组特征 + 逻辑回归），不引入 scikit-learn：
一是保持依赖轻量，二是避免"用封装好的东西复现别人的结论"这种循环论证。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ 特征化
def hash_char_bigrams(texts: Sequence[str], dim: int = 512) -> np.ndarray:
    """把中文/英文文本哈希成固定维度的字符二元组特征。

    中文没有天然空格分词，字符二元组（bigram）是不依赖分词器的稳健选择，
    也是金融短文本上的常用做法。哈希技巧免去维护词表。
    """
    features = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        cleaned = "".join(ch for ch in text if not ch.isspace())
        if len(cleaned) < 2:
            continue
        for i in range(len(cleaned) - 1):
            bigram = cleaned[i : i + 2]
            digest = hashlib.md5(bigram.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            features[row, idx] += sign
    # L2 归一化，避免长文本因 bigram 多而主导
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


# --------------------------------------------------------------------- 模型
@dataclass(slots=True)
class LogisticRegression:
    """极简逻辑回归（全批量梯度下降 + L2）。

    足以演示切分泄漏，且行为完全可复现。
    """

    dim: int
    lr: float = 0.5
    epochs: int = 300
    l2: float = 1e-3
    seed: int = 0
    # 参数在 __post_init__ 里按 dim 初始化，故声明为 init=False 的字段
    # （slots=True 的 dataclass 不允许凭空新增属性）
    w: np.ndarray = field(init=False, repr=False)
    b: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.w = np.zeros(self.dim, dtype=np.float64)
        self.b = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticRegression:
        n = len(y)
        if n == 0:
            return self
        for _ in range(self.epochs):
            logits = x @ self.w + self.b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            error = probs - y
            grad_w = x.T @ error / n + self.l2 * self.w
            grad_b = float(error.mean())
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.array([], dtype=int)
        probs = 1.0 / (1.0 + np.exp(-np.clip(x @ self.w + self.b, -30, 30)))
        return (probs >= 0.5).astype(int)


# ------------------------------------------------------------------- 指标
def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews 相关系数。类别不平衡时比准确率可靠得多。"""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


# ------------------------------------------------------------------- 结果
@dataclass(slots=True)
class SplitAuditResult:
    """一次切分审计的结果。"""

    n_samples: int
    n_features: int
    random_mcc: float
    chronological_mcc: float
    random_acc: float
    chronological_acc: float
    n_folds: int

    @property
    def mcc_inflation(self) -> float:
        """MCC 虚高倍数。1.0 表示随机切分没有带来虚假优势。"""
        if self.chronological_mcc == 0:
            return float("inf") if self.random_mcc > 0 else 1.0
        return self.random_mcc / self.chronological_mcc

    @property
    def acc_inflation(self) -> float:
        if self.chronological_acc == 0:
            return float("inf")
        return self.random_acc / self.chronological_acc

    @property
    def leaking(self) -> bool:
        """是否判定为存在切分泄漏。

        阈值取 1.15：文献 ``arXiv:2608.17223`` 在金融新闻 NLP 上测得随机切分
        相对时序切分的 MCC 虚高区间为 **1.1×–6.5×**，故 1.15 落在该区间的下端，
        是一个保守但不至于漏报的判据。
        """
        return self.mcc_inflation > 1.15

    def summary(self) -> str:
        verdict = "存在切分泄漏" if self.leaking else "切分泄漏不明显"
        return (
            f"切分审计（{self.n_folds} 折，{self.n_samples} 样本 / {self.n_features} 维）：\n"
            f"  随机切分   MCC={self.random_mcc:+.4f}  准确率={self.random_acc:.4f}\n"
            f"  时序切分   MCC={self.chronological_mcc:+.4f}  准确率={self.chronological_acc:.4f}\n"
            f"  虚高倍数   MCC {self.mcc_inflation:.2f}×  准确率 {self.acc_inflation:.2f}×  → {verdict}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_folds": self.n_folds,
            "random_mcc": round(self.random_mcc, 5),
            "chronological_mcc": round(self.chronological_mcc, 5),
            "random_acc": round(self.random_acc, 5),
            "chronological_acc": round(self.chronological_acc, 5),
            "mcc_inflation": round(self.mcc_inflation, 3),
            "acc_inflation": round(self.acc_inflation, 3),
            "leaking": self.leaking,
        }


# -------------------------------------------------------------------- 审计
# ------------------------------------------------------- 受控模拟（验证灵敏度）
_FIRM_WORDS = ("公司", "集团", "控股", "实业", "科技", "能源", "医药", "材料")
_SIGNAL_POS = ("订单饱满", "现金流充裕", "毛利率抬升", "客户拓展顺利")
_SIGNAL_NEG = ("回款承压", "库存积压", "毛利收窄", "客户流失")
_FILLER = ("本期", "经营", "情况", "说明", "披露", "相关", "事项", "进展")


def simulate_persistent_panel(
    *,
    n_firms: int = 40,
    n_periods: int = 100,
    noise: float = 2.5,
    persistence: float = 0.97,
    seed: int = 0,
) -> tuple[list[str], list[int], list[datetime]]:
    """生成一个**标签具有时间持续性**的受控面板，用来验证切分审计的灵敏度。

    ## 为什么需要它

    在次日收益符号这类任务上，信号弱到根本无法分类（见 README 的实测），
    因此"随机切分没有虚高"是正确结论，却无法证明审计器**能**发现泄漏。

    真实金融任务里标签往往是持续状态——信用评级、是否违约、是否处于困境。
    这类标签的相邻观测高度相关。此时随机切分会让训练集里的"邻居"
    （同一主体的前后期）泄漏进测试集，模型学的是记忆而非泛化。

    本函数严格复刻该结构：
      * 每个主体有一条 AR(1) 的潜在状态，切换缓慢（``persistence`` 控制）；
      * 标签 = 状态是否为正；
      * 文本只**含噪**地透露状态（``noise`` 越大越难），并带主体标识词。

    返回 ``(texts, labels, timestamps)``，可直接喂给 :class:`SplitAuditor`。
    """
    rng = np.random.default_rng(seed)
    base = datetime(2023, 1, 2, tzinfo=UTC)

    texts: list[str] = []
    labels: list[int] = []
    stamps: list[datetime] = []

    for firm in range(n_firms):
        firm_word = _FIRM_WORDS[firm % len(_FIRM_WORDS)]
        state = float(rng.standard_normal())
        for t in range(n_periods):
            state = persistence * state + (1.0 - persistence) * rng.standard_normal()
            # 含噪观测：状态越正，越可能命中正面措辞
            observed = state + noise * rng.standard_normal()
            if observed > 0:
                signal = _SIGNAL_POS[rng.integers(len(_SIGNAL_POS))]
            else:
                signal = _SIGNAL_NEG[rng.integers(len(_SIGNAL_NEG))]
            filler = "".join(_FILLER[rng.integers(len(_FILLER))] for _ in range(3))

            texts.append(f"{firm_word}{firm:02d}{signal}，{filler}")
            labels.append(1 if state > 0 else 0)
            stamps.append(base + timedelta(days=t * 7))

    return texts, labels, stamps


class SplitAuditor:
    """对同一特征/模型做随机切分 vs 时序切分的对照评测。"""

    def __init__(
        self,
        *,
        n_folds: int = 5,
        dim: int = 512,
        seed: int = 20261016,
        lr: float = 0.5,
        epochs: int = 300,
    ) -> None:
        self.n_folds = n_folds
        self.dim = dim
        self.seed = seed
        self.lr = lr
        self.epochs = epochs

    def audit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        timestamps: Sequence[datetime],
    ) -> SplitAuditResult:
        """跑一次完整审计。

        Args:
            texts: 样本文本。
            labels: 二分类标签（0/1）。
            timestamps: 每个样本的**可得时刻**，时序切分依据它排序。
        """
        if not (len(texts) == len(labels) == len(timestamps)):
            raise ValueError("texts / labels / timestamps 长度必须一致")
        if len(texts) < self.n_folds * 4:
            raise ValueError("样本太少，无法做有意义的折间比较")

        x = hash_char_bigrams(texts, self.dim)
        y = np.asarray(labels, dtype=int)
        # 用 POSIX 时间戳排序，避免 numpy 对带时区 datetime 的隐式转换警告
        order_time = np.argsort(
            np.array([ts.timestamp() for ts in timestamps], dtype=np.float64),
            kind="mergesort",
        )

        random_mcc, random_acc = self._cross_validate(x, y, self._random_folds(len(y)))
        chrono_mcc, chrono_acc = self._cross_validate(
            x, y, self._chronological_folds(order_time)
        )

        return SplitAuditResult(
            n_samples=len(y),
            n_features=self.dim,
            random_mcc=random_mcc,
            chronological_mcc=chrono_mcc,
            random_acc=random_acc,
            chronological_acc=chrono_acc,
            n_folds=self.n_folds,
        )

    # -------------------------------------------------------------- 折构造
    def _random_folds(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """随机 K 折 —— 时间结构被彻底打乱，泄漏的温床。"""
        rng = np.random.default_rng(self.seed)
        shuffled = rng.permutation(n)
        return [
            (
                np.concatenate([f for j, f in enumerate(np.array_split(shuffled, self.n_folds)) if j != k]),
                np.array_split(shuffled, self.n_folds)[k],
            )
            for k in range(self.n_folds)
        ]

    def _chronological_folds(
        self, order_time: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """滚动时序折：永远用过去训练、未来测试。"""
        chunks = np.array_split(order_time, self.n_folds + 1)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(1, self.n_folds + 1):
            train = np.concatenate(chunks[:k])
            test = chunks[k]
            if len(train) and len(test):
                folds.append((train, test))
        return folds

    # -------------------------------------------------------------- 评测
    def _cross_validate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[float, float]:
        mccs: list[float] = []
        accs: list[float] = []
        for train_idx, test_idx in folds:
            y_train = y[train_idx]
            # 单类训练集无法训练，跳过该折
            if len(np.unique(y_train)) < 2:
                continue
            model = LogisticRegression(dim=self.dim, lr=self.lr, epochs=self.epochs)
            model.fit(x[train_idx], y_train.astype(float))
            pred = model.predict(x[test_idx])
            mccs.append(matthews_corrcoef(y[test_idx], pred))
            accs.append(accuracy(y[test_idx], pred))
        if not mccs:
            return 0.0, float("nan")
        return float(np.mean(mccs)), float(np.mean(accs))


__all__ = [
    "LogisticRegression",
    "SplitAuditResult",
    "SplitAuditor",
    "accuracy",
    "hash_char_bigrams",
    "matthews_corrcoef",
    "simulate_persistent_panel",
]

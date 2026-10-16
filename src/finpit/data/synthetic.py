"""校准过的合成金融语料生成器。

## 为什么用合成数据

要**精确测量**泄漏造成的性能虚高，你必须知道真实信号强度 —— 真实市场数据
给不了这个。合成数据的价值不在于"像真的"，而在于它有一个已知的
``true_beta``，于是"论文报的 IC"与"真实 IC"之间的差就可以被干净地归因。

为了让结论不至于沦为玩具，生成器刻意复刻了若干真实市场的典型事实
（stylized facts）：

* **厚尾**：收益率服从 Student-t，而非正态；
* **波动率聚集**：用类 GARCH(1,1) 过程生成时变波动率；
* **弱信号**：真实预测性很低（日频 IC 在 0.02~0.05 量级）；
* **信息延迟**：新闻的 ``published_at`` 晚于事件时刻，且**收盘后才可得的信息
  不能用于当日决策** —— 这正是未来函数的高发地带。

生成结果全部装进 :class:`~finpit.pit.PITStore`，因此后续任何评测都天然受
时间点门控约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from finpit.pit.store import DatedRecord, PITStore

# 交易日近似：周一至周五
_WEEKDAYS = (0, 1, 2, 3, 4)


@dataclass(slots=True)
class SyntheticConfig:
    """模拟器配置。全部字段参与随机种子，保证可复现。"""

    n_assets: int = 40
    n_days: int = 750
    start: datetime = datetime(2023, 1, 2, 9, 30, tzinfo=UTC)

    # --- 真实数据生成过程 ---
    target_ic: float = 0.035
    """目标信息系数（IC）—— 真实信号强度。

    这是"上帝视角"的答案：合成数据里真实的 corr(信号, 次日收益)。
    日频真实市场的 IC 通常在 0.02~0.05，任何显著高于此的"预测能力"
    都应当被怀疑来自泄漏。真实回归系数 beta 由本字段反解得到。
    """

    sentiment_persistence: float = 0.65
    """情绪自回归系数，制造情绪的持续性（否则信号无法被学习）。"""

    base_vol: float = 0.015
    """基础日波动率。"""

    vol_clustering: float = 0.88
    """GARCH 波动率聚集系数。"""

    shock_df: float = 6.0
    """Student-t 自由度：越小尾部越厚。

    注意 t(4) 的四阶矩发散（峰度无穷），会造出极端离群值并主导回归，
    故默认取 6.0，对应超额峰度约 3，接近真实股指日收益量级。
    """

    # --- 信息可得性 ---
    news_delay_hours: float = 2.0
    """新闻相对事件时刻的发布延迟。"""

    market_close_hour: int = 21
    """UTC 收盘时刻（对应美东 16:00）。决策发生在开盘，故当日收盘价不可用。"""

    price_recap_prob: float = 0.4
    """新闻中附带"当日行情播报"的比例。

    真实财经新闻有相当一部分是在**报道当天涨跌**（"该股今日大涨 5%"）。
    这个特征看似无害，实则是未来函数的高危区：若把这类新闻对齐到
    **同一天**的收益上，就能凭空造出极高的 IC —— 因为它复述的就是答案本身。
    本字段让 :attr:`~finpit.eval.harness.LeakMode.PIT_VIOLATION` 能被真实地演示。
    """

    seed: int = 20261016

    def __post_init__(self) -> None:
        if self.n_days < 50:
            raise ValueError("n_days 至少 50，否则统计量无意义")
        if self.n_assets < 2:
            raise ValueError("n_assets 至少 2")
        if not 0.0 < self.target_ic < 1.0:
            raise ValueError("target_ic 必须在 (0, 1) 开区间内")
        if not 0.0 <= self.price_recap_prob <= 1.0:
            raise ValueError("price_recap_prob 必须在 [0, 1] 内")

    @property
    def true_beta(self) -> float:
        """由目标 IC 反解出的真实回归系数。

        生成过程为 ``r = beta * s + sigma * eps``，其中 ``s``、``eps`` 标准化，
        于是 ``corr(beta*s, r) = beta / sqrt(beta^2 + sigma^2)``。
        令其等于 ``target_ic`` 即可解出 ``beta``。
        """
        rho = self.target_ic
        return rho * self.base_vol / np.sqrt(1.0 - rho**2)


@dataclass(slots=True)
class SyntheticCorpus:
    """生成结果：一个受 PIT 门控的仓库 + 上帝视角的答案表。"""

    store: PITStore
    config: SyntheticConfig
    # 宽表：index=日期, columns=资产, values=次日收益（用于评测对齐）
    returns: dict[str, dict[str, float]]
    sentiment: dict[str, dict[str, float]]
    dates: list[datetime]
    assets: list[str]

    @property
    def true_beta(self) -> float:
        """真实回归系数（由 ``target_ic`` 反解）。"""
        return self.config.true_beta

    def realized_ic(self) -> float:
        """生成数据上 **实测** 的 corr(前一日情绪, 当日收益)。

        这才是任何只用滞后情绪的预测器所能达到的**真实上限**。
        它通常略低于 ``target_ic``，因为 GARCH 让高波动期的相关性被稀释。
        评测时应以本值为地面真值，而不是配置里的目标值。
        """
        xs: list[float] = []
        ys: list[float] = []
        for i in range(1, len(self.dates)):
            prev = self.dates[i - 1].strftime("%Y-%m-%d")
            cur = self.dates[i].strftime("%Y-%m-%d")
            for asset in self.assets:
                xs.append(self.sentiment[prev][asset])
                ys.append(self.returns[cur][asset])
        if len(xs) < 3:
            return float("nan")
        x = np.asarray(xs)
        y = np.asarray(ys)
        if x.std() == 0 or y.std() == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def describe(self) -> str:
        span = self.store.time_span()
        assert span is not None
        return (
            f"合成语料：{len(self.assets)} 资产 × {len(self.dates)} 交易日 "
            f"({span[0]:%Y-%m-%d} .. {span[1]:%Y-%m-%d})，"
            f"{self.store.record_count} 条记录，真实 IC={self.config.target_ic} "
            f"(beta={self.true_beta:.2e})"
        )


def _trading_days(start: datetime, count: int) -> list[datetime]:
    """从 ``start`` 起生成 ``count`` 个交易日（跳过周末）。"""
    days: list[datetime] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() in _WEEKDAYS:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def generate_corpus(config: SyntheticConfig | None = None) -> SyntheticCorpus:
    """生成一份确定性的合成金融语料。

    同样的 ``config``（含 ``seed``）永远产出同样的结果。
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    assets = [f"ASSET{i:03d}" for i in range(cfg.n_assets)]
    dates = _trading_days(cfg.start, cfg.n_days)
    n_assets, n_days = cfg.n_assets, cfg.n_days

    # ------------------------------------------------------------------
    # 1) 潜在情绪：AR(1)，为每个资产独立生成
    # ------------------------------------------------------------------
    phi = cfg.sentiment_persistence
    shocks = rng.standard_normal((n_days, n_assets))
    sentiment = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        sentiment[t] = phi * sentiment[t - 1] + np.sqrt(1.0 - phi**2) * shocks[t]

    # ------------------------------------------------------------------
    # 2) 时变波动率（GARCH(1,1) 风格）+ 厚尾冲击
    # ------------------------------------------------------------------
    omega = cfg.base_vol**2 * (1.0 - cfg.vol_clustering - 0.10)
    alpha, beta_g = 0.10, cfg.vol_clustering
    var = np.full((n_days, n_assets), cfg.base_vol**2)
    eps = rng.standard_t(cfg.shock_df, size=(n_days, n_assets)) / np.sqrt(
        cfg.shock_df / (cfg.shock_df - 2.0)
    )
    for t in range(1, n_days):
        var[t] = omega + alpha * (eps[t - 1] ** 2) * var[t - 1] + beta_g * var[t - 1]
    sigma = np.sqrt(var)

    # ------------------------------------------------------------------
    # 3) 收益：真实信号来自 **前一日** 情绪（决策时点上情绪已可得）
    # ------------------------------------------------------------------
    signal = np.zeros((n_days, n_assets))
    signal[1:] = cfg.true_beta * sentiment[:-1]
    returns = signal + sigma * eps

    # ------------------------------------------------------------------
    # 4) 装入 PITStore：每条记录声明自己的 published_at
    # ------------------------------------------------------------------
    store = PITStore("synthetic")
    returns_by_date: dict[str, dict[str, float]] = {}
    sentiment_by_date: dict[str, dict[str, float]] = {}
    records: list[DatedRecord] = []

    for ti, day in enumerate(dates):
        close_at = day.replace(hour=cfg.market_close_hour, minute=0, second=0, microsecond=0)
        key_date = day.strftime("%Y-%m-%d")
        returns_by_date[key_date] = {}
        sentiment_by_date[key_date] = {}

        for ai, asset in enumerate(assets):
            sent = float(sentiment[ti, ai])
            ret = float(returns[ti, ai])
            sentiment_by_date[key_date][asset] = sent
            returns_by_date[key_date][asset] = ret

            # 情绪/新闻：事件发生在收盘后，延迟 news_delay_hours 才公开
            published = close_at + timedelta(hours=cfg.news_delay_hours)
            records.append(
                DatedRecord(
                    key=f"{asset}:news",
                    published_at=published,
                    payload={
                        "date": key_date,
                        "asset": asset,
                        "headline": _headline(asset, sent, ret, rng, cfg.price_recap_prob),
                        "sentiment_latent": sent,
                    },
                    source="synthetic-news",
                )
            )
            # 收盘价：只有到收盘时刻才可得
            records.append(
                DatedRecord(
                    key=f"{asset}:close",
                    published_at=close_at,
                    payload={"date": key_date, "asset": asset, "return": ret},
                    source="synthetic-market",
                )
            )

    store.add_many(records)
    return SyntheticCorpus(
        store=store,
        config=cfg,
        returns=returns_by_date,
        sentiment=sentiment_by_date,
        dates=dates,
        assets=assets,
    )


# ---------------------------------------------------------------- 标题模板
_POSITIVE = [
    "{a} 季度营收超预期，上调全年指引",
    "{a} 获多家机构上调评级，需求强劲",
    "{a} 宣布回购计划，现金流稳健",
    "{a} 新产线投产，订单能见度提升",
    "{a} 毛利率改善，成本控制见效",
]
_NEGATIVE = [
    "{a} 下调业绩指引，需求走弱",
    "{a} 遭机构下调评级，库存高企",
    "{a} 核心高管离职，战略存不确定性",
    "{a} 面临监管调查，合规成本上升",
    "{a} 主要客户砍单，产能利用率下滑",
]
_NEUTRAL = [
    "{a} 发布公告，经营情况无重大变化",
    "{a} 召开例行股东大会",
    "{a} 完成董事会换届",
    "{a} 披露月度经营数据",
]


def _headline(
    asset: str,
    sentiment: float,
    same_day_return: float,
    rng: np.random.Generator,
    recap_prob: float,
) -> str:
    """把潜在情绪映射成一条带噪声的标题 —— 模拟"文本 ≠ 情绪真值"。

    另有 ``recap_prob`` 的概率附上一句**当日行情播报**。这句话本身是事后陈述，
    却在时间对齐错误时变成"完美预测器"，是本研究要演示的核心陷阱之一。
    """
    noisy = sentiment + 0.6 * rng.standard_normal()
    if noisy > 0.45:
        template = _POSITIVE[rng.integers(len(_POSITIVE))]
    elif noisy < -0.45:
        template = _NEGATIVE[rng.integers(len(_NEGATIVE))]
    else:
        template = _NEUTRAL[rng.integers(len(_NEUTRAL))]

    headline = template.format(a=asset)
    if rng.random() < recap_prob:
        pct = same_day_return * 100.0
        direction = "上涨" if pct >= 0 else "下跌"
        headline = f"{headline}；盘面看，该股当日{direction}{abs(pct):.2f}%"
    return headline


__all__ = [
    "SyntheticConfig",
    "SyntheticCorpus",
    "generate_corpus",
]

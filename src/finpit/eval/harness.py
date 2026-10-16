"""泄漏消融实验台 —— 本项目的核心贡献。

## 核心问题

论文里"LLM 情绪因子 IC=0.08"这类结论，有多少是信号本身，有多少是评测流程漏进来的？
在真实市场上你无法回答，因为不知道真实 IC。但在合成语料上，``true_ic`` 是已知的，
于是**虚高可以被精确归因**。

## 做法

同一条信号抽取结果，送进若干条评测流程，每条开启一个真实存在的研究陋习：

===============  ==========================================================
泄漏模式          对应的真实研究陋习
===============  ==========================================================
``clean``        严格时间点纪律，无泄漏（对照组）
``outcome_prompt`` 提示词里混入了事后撰写的行情回顾 —— 新闻 API 常返回被更新过的文章
``test_set_select`` 在评测集上挑最好的超参/变体 —— 几乎所有论文都犯
``cherry_pick``  样本按结果筛选（如剔除退市标的、剔除"异常"样本）—— 幸存者偏差
``oracle_latent`` 直接读潜在真值，绕过文本抽取 —— 给出该信号族的上限
``future_return`` 把目标收益本身当特征 —— 审计管线的健全性自检
===============  ==========================================================

前三种会**虚高**，后两种是**诊断上界**。把虚高幅度和 PIT 拦截次数一起报出来，
就是"Ablation 之后还剩多少"这个问题的答案。
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from finpit.data.synthetic import SyntheticCorpus
from finpit.eval.metrics import SignalMetrics, evaluate
from finpit.pit.store import DatedRecord, PITStore, ensure_utc
from finpit.signals.base import Extractor, NewsItem

log = logging.getLogger(__name__)


class LeakMode(str, enum.Enum):
    """评测流程的泄漏模式。"""

    CLEAN = "clean"
    PIT_VIOLATION = "pit_violation"
    OUTCOME_PROMPT = "outcome_prompt"
    TEST_SET_SELECT = "test_set_select"
    CHERRY_PICK = "cherry_pick"
    ORACLE_LATENT = "oracle_latent"
    FUTURE_RETURN = "future_return"

    @property
    def inflating(self) -> bool:
        """该模式是否属于"会让指标虚高"的陋习（而非诊断上界）。"""
        return self in {LeakMode.OUTCOME_PROMPT, LeakMode.TEST_SET_SELECT, LeakMode.CHERRY_PICK}


@dataclass(frozen=True, slots=True)
class PanelRow:
    """一行评测样本：一个决策时点上的一个标的。"""

    decision_time: datetime
    asset: str
    news_time: datetime
    news_date: str
    forward_date: str
    text: str
    latent: float
    forward_return: float

    def retrospective_text(self) -> str:
        """模拟"事后被更新过的新闻稿"：文中已写明次日涨跌幅。"""
        pct = self.forward_return * 100.0
        direction = "上涨" if pct >= 0 else "下跌"
        return f"{self.text}。该股次日{direction}{abs(pct):.2f}%，市场反应明显。"


@dataclass
class ModeResult:
    """单个泄漏模式的评测结果。"""

    mode: LeakMode
    metrics: SignalMetrics
    pit_blocks: int
    note: str = ""

    def inflation_ic(self, baseline_ic: float) -> float:
        """相对对照组的 IC 虚高比例。"""
        if not np.isfinite(baseline_ic) or baseline_ic == 0:
            return float("nan")
        return (self.metrics.ic - baseline_ic) / abs(baseline_ic)

    def inflation_rank_ic(self, baseline_rank_ic: float) -> float:
        if not np.isfinite(baseline_rank_ic) or baseline_rank_ic == 0:
            return float("nan")
        return (self.metrics.rank_ic - baseline_rank_ic) / abs(baseline_rank_ic)


@dataclass
class AblationReport:
    """一次完整消融实验的结果。"""

    results: list[ModeResult]
    true_ic: float
    """合成语料的真实 IC —— 上帝视角的基准。"""

    n_panel_rows: int
    extractor_name: str
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def clean(self) -> ModeResult:
        for result in self.results:
            if result.mode is LeakMode.CLEAN:
                return result
        raise KeyError("报告里没有 clean 对照组")

    def table(self) -> str:
        """渲染成可粘贴进 README 的 markdown 表格。"""
        base = self.clean.metrics
        lines = [
            "| 泄漏模式 | IC | RankIC | 命中率 | 分位差 | 均值收益 | IC 虚高 | PIT 拦截 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for result in self.results:
            m = result.metrics
            infl = result.inflation_ic(base.ic)
            infl_text = (
                "—" if result.mode is LeakMode.CLEAN or not np.isfinite(infl) else f"{infl:+.1%}"
            )
            lines.append(
                f"| `{result.mode.value}` | {m.ic:+.4f} | {m.rank_ic:+.4f} | "
                f"{m.hit_rate:.4f} | {m.quantile_spread:+.5f} | {m.mean_return:+.5f} | "
                f"{infl_text} | {result.pit_blocks} |"
            )
        lines.append("")
        lines.append(
            f"合成语料真实 IC = **{self.true_ic:.4f}**（上帝视角）；"
            f"抽取器 = `{self.extractor_name}`；样本 {self.n_panel_rows} 行。"
        )
        leaks = [r for r in self.results if r.mode.inflating]
        if leaks and np.isfinite(base.ic):
            worst = max(leaks, key=lambda r: r.metrics.ic)
            lines.append(
                f"最大 IC 虚高来自 `{worst.mode.value}`："
                f"IC {base.ic:+.4f} → {worst.metrics.ic:+.4f}"
                f"（{worst.inflation_ic(base.ic):+.1%}）。"
            )
        # 提示"不同泄漏污染不同指标"：改动样本的模式会移动收益水平，
        # 而不改样本的模式只可能在相关性维度上造假。
        level_leaks = [r for r in self.results if r.metrics.n_obs != base.n_obs]
        correlation_leaks = [
            r
            for r in self.results
            if np.isfinite(r.inflation_ic(base.ic)) and r.inflation_ic(base.ic) > 0.2
        ]
        if level_leaks and correlation_leaks:
            lines.append(
                "注意：`"
                + "`, `".join(r.mode.value for r in level_leaks)
                + "` 改变的是**样本构成**，因而移动了**收益水平**；而 `"
                + "`, `".join(r.mode.value for r in correlation_leaks)
                + "` 抬高的是**相关性**。两类污染方向不同，单一指标无法同时发现。"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        base = self.clean.metrics
        return {
            "true_ic": round(self.true_ic, 5),
            "extractor": self.extractor_name,
            "n_panel_rows": self.n_panel_rows,
            "audit": self.audit,
            "results": [
                {
                    "mode": r.mode.value,
                    "note": r.note,
                    "pit_blocks": r.pit_blocks,
                    "inflation_ic": None
                    if not np.isfinite(r.inflation_ic(base.ic))
                    else round(r.inflation_ic(base.ic), 4),
                    **r.metrics.as_dict(),
                }
                for r in self.results
            ],
        }


def build_panel(
    store: PITStore,
    corpus: SyntheticCorpus,
    *,
    decision_delay_minutes: int = 1,
    max_rows: int | None = None,
) -> tuple[list[PanelRow], int]:
    """在严格时间点纪律下构建评测面板。

    决策时点约定：在 ``dates[i-1]`` 收盘后新闻已发布（收盘 + 延迟）之后做决策，
    预测 ``dates[i]`` 的收益。这样"可用的最新信息"就是 ``dates[i-1]`` 当天收盘后
    发布的那条新闻。

    Returns:
        ``(面板行, PIT 拦截次数)``。
    """
    cfg = corpus.config
    date_keys = [d.strftime("%Y-%m-%d") for d in corpus.dates]
    delay = timedelta(hours=cfg.news_delay_hours, minutes=decision_delay_minutes)

    rows: list[PanelRow] = []
    for i in range(1, len(corpus.dates)):
        # 在日期边界处截断，保证每个决策时点的横截面是完整的
        if max_rows is not None and len(rows) >= max_rows:
            break

        prev_day = corpus.dates[i - 1]
        close_prev = prev_day.replace(
            hour=cfg.market_close_hour, minute=0, second=0, microsecond=0
        )
        decision_time = ensure_utc(close_prev + delay)
        forward_key = date_keys[i]

        with store.as_of(decision_time) as view:
            for asset in corpus.assets:
                record = view.latest_record(f"{asset}:news")
                rows.append(
                    PanelRow(
                        decision_time=decision_time,
                        asset=asset,
                        news_time=record.published_at,
                        news_date=record.payload["date"],
                        forward_date=forward_key,
                        text=record.payload["headline"],
                        latent=float(record.payload["sentiment_latent"]),
                        forward_return=corpus.returns[forward_key][asset],
                    )
                )

    blocks = store.audit().blocked_accesses
    return rows, blocks


class LeakageAblation:
    """把一条信号送进多条评测流程，量化每种陋习造成的虚高。"""

    def __init__(
        self,
        corpus: SyntheticCorpus,
        extractor: Extractor,
        *,
        max_rows: int | None = None,
        cherry_pick_quantile: float = 0.2,
    ) -> None:
        self.corpus = corpus
        self.extractor = extractor
        self.max_rows = max_rows
        self.cherry_pick_quantile = cherry_pick_quantile

        # (日期, 标的) -> 新闻记录。仅供"故意越界"的泄漏模式取用，
        # 正常路径一律走 PIT 视图。
        self._news_index: dict[tuple[str, str], DatedRecord] = {}
        for asset in corpus.assets:
            for record in corpus.store._by_key.get(f"{asset}:news", ()):
                self._news_index[(record.payload["date"], asset)] = record

    # ------------------------------------------------------------------ 主流程
    def run(self, modes: list[LeakMode] | None = None) -> AblationReport:
        modes = modes or list(LeakMode)
        self.corpus.store.reset_audit()

        panel, blocks = build_panel(
            self.corpus.store, self.corpus, max_rows=self.max_rows
        )
        if not panel:
            raise RuntimeError("面板为空，无法评测")

        actual = np.array([r.forward_return for r in panel])
        baseline_scores = self._score(panel, leak=LeakMode.CLEAN)

        results: list[ModeResult] = []
        for mode in modes:
            results.append(self._run_mode(mode, panel, actual, baseline_scores, blocks))

        return AblationReport(
            results=results,
            true_ic=self.corpus.realized_ic(),
            n_panel_rows=len(panel),
            extractor_name=getattr(self.extractor, "name", type(self.extractor).__name__),
            audit=self.corpus.store.audit().to_dict(),
        )

    # ------------------------------------------------------------------ 单模式
    def _run_mode(
        self,
        mode: LeakMode,
        panel: list[PanelRow],
        actual: np.ndarray,
        baseline_scores: np.ndarray,
        blocks: int,
    ) -> ModeResult:
        note = ""
        pit_blocks = 0

        if mode is LeakMode.ORACLE_LATENT:
            scores = np.array([r.latent for r in panel])
            note = "绕过文本抽取，直读潜在真值：信号族上限"

        elif mode is LeakMode.FUTURE_RETURN:
            scores = actual.copy()
            note = "把目标收益本身当特征：应得 IC≈1，用于验证审计管线可检出泄漏"

        elif mode is LeakMode.PIT_VIOLATION:
            scores = self._score(panel, leak=mode)
            pit_blocks = len(panel)
            note = (
                "差一天的未来函数：用**目标日当天**的新闻预测当天收益。"
                "PIT 门控对每一行都会拦截"
            )

        elif mode is LeakMode.OUTCOME_PROMPT:
            scores = self._score(panel, leak=mode)
            note = "提示词混入事后行情回顾（新闻 API 常见：文章被事后更新）"
            pit_blocks = self._count_pit_blocks(panel, mode)

        elif mode is LeakMode.CHERRY_PICK:
            scores, actual, kept = self._cherry_pick(panel, actual, baseline_scores)
            note = f"按结果筛选样本，剔除表现最差的 {self.cherry_pick_quantile:.0%} 标的（幸存者偏差）；保留 {kept} 行"

        elif mode is LeakMode.TEST_SET_SELECT:
            scores, note = self._test_set_select(panel, actual)

        else:  # CLEAN
            scores = baseline_scores
            note = "严格时间点纪律，无泄漏（对照组）"

        metrics = evaluate(scores, actual)
        return ModeResult(mode=mode, metrics=metrics, pit_blocks=pit_blocks, note=note)

    # -------------------------------------------------------------------- 工具
    def _score(self, panel: list[PanelRow], *, leak: LeakMode) -> np.ndarray:
        """用抽取器给面板打分。

        * ``OUTCOME_PROMPT``：改用事后版本的文本（含已实现的涨跌幅）。
        * ``PIT_VIOLATION``：改用**目标日当天**的新闻（差一天的未来函数）。
        """
        texts: list[str] = []
        for row in panel:
            if leak is LeakMode.OUTCOME_PROMPT:
                texts.append(row.retrospective_text())
            elif leak is LeakMode.PIT_VIOLATION:
                future_record = self._news_index.get((row.forward_date, row.asset))
                texts.append(future_record.payload["headline"] if future_record else row.text)
            else:
                texts.append(row.text)

        items = [
            NewsItem(asset=row.asset, text=text, published_at=row.news_time)
            for row, text in zip(panel, texts, strict=True)
        ]
        signals = self.extractor.score_batch(items)
        return np.array([s.weighted for s in signals])

    def _count_pit_blocks(self, panel: list[PanelRow], mode: LeakMode) -> int:
        """统计该模式下 PIT 门控会拦下多少次越界读取。

        ``OUTCOME_PROMPT`` 的越界发生在"读取事后版本文章"这一步：
        该版本要等次日收盘、收益实现之后才存在。把它按这个时刻登记进一个
        探针仓库，再用决策时点去读 —— 每次读取都应被拦下。
        """
        if mode is not LeakMode.OUTCOME_PROMPT:
            return 0

        probe = PITStore("outcome-probe")
        # 用"资产@决策时点"作唯一键。若只用资产作键，跨日期累积后，
        # 较晚的决策时点会读到较早日期的事后文章，导致漏计。
        for row in panel:
            probe.add(
                DatedRecord(
                    key=f"{row.asset}@{row.decision_time.isoformat()}",
                    published_at=row.decision_time + timedelta(days=1),
                    payload=row.text,
                )
            )

        blocked = 0
        for row in panel:
            with probe.as_of(row.decision_time) as view:
                if not view.probe(f"{row.asset}@{row.decision_time.isoformat()}"):
                    blocked += 1
        return blocked

    def _cherry_pick(
        self,
        panel: list[PanelRow],
        actual: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """按标的的累计收益淘汰最差的一批 —— 模拟幸存者偏差。"""
        by_asset: dict[str, list[int]] = {}
        for idx, row in enumerate(panel):
            by_asset.setdefault(row.asset, []).append(idx)

        totals = {a: float(actual[idxs].sum()) for a, idxs in by_asset.items()}
        cutoff = np.quantile(list(totals.values()), self.cherry_pick_quantile)
        survivors = {a for a, tot in totals.items() if tot > cutoff}
        keep = np.array([i for i, row in enumerate(panel) if row.asset in survivors])
        return scores[keep], actual[keep], int(len(keep))

    def _test_set_select(
        self, panel: list[PanelRow], actual: np.ndarray
    ) -> tuple[np.ndarray, str]:
        """在评测集上挑最好的一个变体，并把这个"最好成绩"当作结论汇报。"""
        from finpit.signals.lexicon import LexiconExtractor

        best_ic = -np.inf
        best_scores: np.ndarray | None = None
        best_scale = 0.0

        for scale in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0):
            extractor = LexiconExtractor(scale=scale)
            items = [
                NewsItem(asset=r.asset, text=r.text, published_at=r.news_time) for r in panel
            ]
            scores = np.array([s.weighted for s in extractor.score_batch(items)])
            ic = evaluate(scores, actual).ic
            if np.isfinite(ic) and ic > best_ic:
                best_ic, best_scores, best_scale = ic, scores, scale

        assert best_scores is not None
        note = (
            f"在评测集上挑最优 scale={best_scale}（共试 8 组）；"
            f"报出的是选择后的最大值，即乐观偏差"
        )
        return best_scores, note


__all__ = [
    "AblationReport",
    "LeakMode",
    "LeakageAblation",
    "ModeResult",
    "PanelRow",
    "build_panel",
]

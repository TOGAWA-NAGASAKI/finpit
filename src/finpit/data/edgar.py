"""SEC EDGAR 真实数据接入 —— 真实世界的披露时刻。

## 为什么需要它

合成数据能精确测量泄漏，但无法替代真实验证。EDGAR 恰好提供了真实世界
最经典的未来函数来源：**三个不同的时间戳**

============  ==========================================================
字段           含义
============  ==========================================================
``reportDate`` 财报所**归属**的会计期间结束日（如 2024-09-28）
``filingDate`` 申报**日期**（只有日粒度）
``acceptanceDateTime`` EDGAR 系统**实际接收**该申报的时刻（精确到秒，带时区）
============  ==========================================================

朴素做法按 ``reportDate`` 对齐基本面数据 —— 但那时财报根本还不存在。
真实可得时刻是 ``acceptanceDateTime``。二者之间隔着数周到数月，
**这段时间就是泄漏**。

本模块把这个间隔量化出来，并可直接转成受 :class:`~finpit.pit.PITStore`
门控的记录，从而让"按期间对齐"这种朴素做法当场抛 ``LookaheadError``。

## 数据来源与礼仪

SEC 要求声明 User-Agent 并限制约 10 请求/秒。本模块默认串行请求 + 磁盘缓存，
重复运行不产生网络流量。
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any

import httpx

from finpit.pit.store import DatedRecord, PITStore

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC 要求 User-Agent 声明应用名**与真实联系方式**，否则返回 403。
# 实测：不含邮箱的 UA（哪怕带项目 URL）一律被拒。
# 因此联系方式从环境变量读取，不硬编码进仓库。
ENV_CONTACT = "FINPIT_SEC_CONTACT"
APP_NAME = "finpit-research/0.1"

# SEC 允许约 10 req/s；留足余量
DEFAULT_INTERVAL_S = 0.25


def default_user_agent() -> str:
    """构造 SEC 要求的 User-Agent。

    Raises:
        RuntimeError: 未设置 ``FINPIT_SEC_CONTACT`` 时。
            SEC 会拒绝不含联系方式的请求，与其拿到一个没头没脑的 403，
            不如在这里明确告诉使用者要配什么。
    """
    contact = os.environ.get(ENV_CONTACT, "").strip()
    if not contact:
        raise RuntimeError(
            f"未设置 {ENV_CONTACT}。SEC 要求请求方声明真实联系方式"
            "（仅带项目 URL 的 User-Agent 会被 403 拒绝）。\n"
            f"请在 .env 中设置，例如：{ENV_CONTACT}=you@example.com"
        )
    return f"{APP_NAME} ({contact})"


@dataclass(frozen=True, slots=True)
class Filing:
    """一条 EDGAR 申报记录。"""

    cik: int
    entity: str
    form: str
    report_date: date | None
    filing_date: date
    acceptance: datetime
    accession: str

    @property
    def period_lag_days(self) -> float | None:
        """会计期间结束 → 实际公开可得，相隔多少天。

        这就是"按 ``reportDate`` 对齐"所引入的泄漏时长。
        """
        if self.report_date is None:
            return None
        period_end = datetime.combine(self.report_date, dtime(0, 0), tzinfo=UTC)
        return (self.acceptance - period_end).total_seconds() / 86400.0

    @property
    def filing_lag_hours(self) -> float:
        """``filingDate`` 当天零点 → 实际可得时刻，相隔多少小时。"""
        day_start = datetime.combine(self.filing_date, dtime(0, 0), tzinfo=UTC)
        return (self.acceptance - day_start).total_seconds() / 3600.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cik": self.cik,
            "entity": self.entity,
            "form": self.form,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "filing_date": self.filing_date.isoformat(),
            "acceptance": self.acceptance.isoformat(),
            "accession": self.accession,
            "period_lag_days": round(self.period_lag_days, 3) if self.period_lag_days else None,
        }


def _parse_acceptance(raw: str) -> datetime:
    """解析 ``2026-10-16T02:30:17.000Z`` 这类时刻。"""
    cleaned = raw.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    return dt.astimezone(UTC)


def _parse_date(raw: str) -> date | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


class EdgarClient:
    """带磁盘缓存的 EDGAR 申报客户端。"""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        user_agent: str | None = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        timeout: float = 45.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) / "edgar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent or default_user_agent()
        self.interval_s = interval_s
        self.timeout = timeout
        self._last_request = 0.0
        self.network_calls = 0
        self.cache_hits = 0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if self._last_request > 0 and elapsed < self.interval_s:
            time.sleep(self.interval_s - elapsed)

    def submissions(self, cik: int, *, refresh: bool = False) -> dict[str, Any]:
        """取某公司的全部申报索引（带缓存）。"""
        path = self.cache_dir / f"CIK{cik:010d}.json"
        if path.is_file() and not refresh:
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))

        self._throttle()
        url = SUBMISSIONS_URL.format(cik=cik)
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
        self._last_request = time.monotonic()
        self.network_calls += 1
        resp.raise_for_status()
        data = resp.json()
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def filings(
        self,
        cik: int,
        *,
        forms: Sequence[str] = ("10-K", "10-Q"),
        limit: int | None = None,
        refresh: bool = False,
    ) -> list[Filing]:
        """取某公司的指定表单申报，按实际公开时刻升序。"""
        data = self.submissions(cik, refresh=refresh)
        recent = data.get("filings", {}).get("recent", {})
        entity = data.get("name", str(cik))
        n = len(recent.get("form", []))
        wanted = set(forms)

        out: list[Filing] = []
        for i in range(n):
            form = recent["form"][i]
            if wanted and form not in wanted:
                continue
            try:
                acceptance = _parse_acceptance(recent["acceptanceDateTime"][i])
            except (KeyError, ValueError):
                continue
            out.append(
                Filing(
                    cik=cik,
                    entity=entity,
                    form=form,
                    report_date=_parse_date(recent.get("reportDate", [""] * n)[i]),
                    filing_date=_parse_date(recent["filingDate"][i]) or acceptance.date(),
                    acceptance=acceptance,
                    accession=recent.get("accessionNumber", [""] * n)[i],
                )
            )
        out.sort(key=lambda f: f.acceptance)
        return out[:limit] if limit else out


# ------------------------------------------------------------------ PIT 接入
def filing_key(filing: Filing) -> str:
    """单条申报的 PIT 主键。

    **必须精确到具体期间。** 曾用 ``f"{cik}:{form}"`` 这种粗粒度键，
    结果同一公司的历年申报被塌缩成一个键：在"本期期间刚结束"的时点去读，
    ``PITView.get`` 会**静默返回上一期的旧记录**而不抛 ``LookaheadError``，
    门控形同虚设（实测拦截率仅 3.5%，而正确设计应为 ~100%）。

    这是 PIT 守卫的一类隐蔽失效：键的粒度决定了门控是否真的有效。
    """
    if filing.report_date is not None:
        period = filing.report_date.isoformat()
    else:
        period = filing.acceptance.date().isoformat()
    return f"{filing.cik}:{filing.form}:{period}"


def to_pit_records(filings: Iterable[Filing]) -> list[DatedRecord]:
    """把申报转成受时间点门控的记录。

    ``published_at`` 取 **acceptanceDateTime** —— 系统实际接收的时刻，
    也就是公众真正能拿到的时刻。用 ``reportDate`` 或 ``filingDate`` 都会泄漏。
    """
    return [
        DatedRecord(
            key=filing_key(f),
            published_at=f.acceptance,
            payload=f.as_dict(),
            source="sec-edgar",
        )
        for f in filings
    ]


def build_pit_store(filings: Iterable[Filing], name: str = "edgar") -> PITStore:
    store = PITStore(name)
    store.add_many(to_pit_records(filings))
    return store


# ------------------------------------------------------------------ 延迟统计
def _weekdays_between(start: datetime, end: datetime) -> int:
    """两个时刻之间的工作日数（周一至周五）。

    这是**近似**：未扣除美国联邦假日。用作量级参照足够，精确回测需接交易日历。
    """
    if end <= start:
        return 0
    days = 0
    cursor = start.date()
    end_date = end.date()
    while cursor < end_date:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            days += 1
    return days


@dataclass(slots=True)
class FilingLagReport:
    """真实披露延迟的分布统计。"""

    form: str
    n: int
    median_days: float
    p10_days: float
    p90_days: float
    min_days: float
    max_days: float
    median_weekdays: int
    median_filing_lag_hours: float

    def summary(self) -> str:
        return (
            f"{self.form:6} n={self.n:4}  "
            f"期间结束→公开可得：中位 {self.median_days:6.1f} 天 "
            f"(≈{self.median_weekdays:3d} 个工作日)  "
            f"P10–P90 {self.p10_days:.0f}–{self.p90_days:.0f} 天  "
            f"[{self.min_days:.0f}, {self.max_days:.0f}]"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "n": self.n,
            "median_days": round(self.median_days, 2),
            "p10_days": round(self.p10_days, 2),
            "p90_days": round(self.p90_days, 2),
            "min_days": round(self.min_days, 2),
            "max_days": round(self.max_days, 2),
            "median_weekdays": self.median_weekdays,
            "median_filing_lag_hours": round(self.median_filing_lag_hours, 2),
        }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def filing_lag_report(filings: Sequence[Filing], form: str) -> FilingLagReport:
    """计算某一表单的真实披露延迟分布。"""
    subset = [f for f in filings if f.form == form and f.period_lag_days is not None]
    if not subset:
        raise ValueError(f"没有 {form} 的可统计样本")

    lags = sorted(f.period_lag_days for f in subset)  # type: ignore[misc]
    filing_lags = sorted(f.filing_lag_hours for f in subset)
    median_days = _percentile(lags, 0.5)
    median_accept = sorted(subset, key=lambda f: f.period_lag_days)[len(subset) // 2]
    median_wd = _weekdays_between(
        datetime.combine(median_accept.report_date, dtime(0, 0), tzinfo=UTC),  # type: ignore[arg-type]
        median_accept.acceptance,
    )
    return FilingLagReport(
        form=form,
        n=len(subset),
        median_days=median_days,
        p10_days=_percentile(lags, 0.10),
        p90_days=_percentile(lags, 0.90),
        min_days=lags[0],
        max_days=lags[-1],
        median_weekdays=median_wd,
        median_filing_lag_hours=_percentile(filing_lags, 0.5),
    )


__all__ = [
    "APP_NAME",
    "DEFAULT_INTERVAL_S",
    "ENV_CONTACT",
    "EdgarClient",
    "Filing",
    "FilingLagReport",
    "TICKER_MAP_URL",
    "build_pit_store",
    "default_user_agent",
    "filing_key",
    "filing_lag_report",
    "to_pit_records",
]

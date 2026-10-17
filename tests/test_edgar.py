"""真实 EDGAR 接入的测试。

不联网：全部用构造的 Filing 对象，锁住时间语义与键粒度这两个关键点。
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from finpit.data.edgar import (
    EdgarClient,
    Filing,
    build_pit_store,
    default_user_agent,
    filing_key,
    filing_lag_report,
)
from finpit.pit import LookaheadError


def _filing(
    *,
    form: str = "10-Q",
    report: date | None = date(2024, 3, 31),
    filed: date = date(2024, 5, 2),
    accepted: datetime = datetime(2024, 5, 2, 20, 15, tzinfo=UTC),
    cik: int = 1,
) -> Filing:
    return Filing(
        cik=cik,
        entity="Test Co",
        form=form,
        report_date=report,
        filing_date=filed,
        acceptance=accepted,
        accession="0000000000-24-000001",
    )


# ------------------------------------------------------------------ 时间语义
def test_period_lag_measures_period_end_to_availability():
    """期间结束 → 实际可得，这是"按 reportDate 对齐"所引入的泄漏时长。"""
    f = _filing(report=date(2024, 3, 31), accepted=datetime(2024, 5, 2, 20, 15, tzinfo=UTC))
    assert f.period_lag_days == pytest.approx(32 + 20.25 / 24, abs=1e-6)


def test_period_lag_none_when_no_report_date():
    assert _filing(report=None).period_lag_days is None


def test_filing_lag_is_intraday():
    """filingDate 只有日粒度，实际可得时刻通常在当天某个钟点。"""
    f = _filing(filed=date(2024, 5, 2), accepted=datetime(2024, 5, 2, 20, 15, tzinfo=UTC))
    assert f.filing_lag_hours == pytest.approx(20.25)


def test_acceptance_is_timezone_aware_utc():
    f = _filing()
    assert f.acceptance.tzinfo is not None
    assert f.acceptance.utcoffset().total_seconds() == 0


# -------------------------------------------------------------------- 键粒度
def test_filing_key_includes_period():
    """键必须精确到期间。

    曾用 ``cik:form`` 这种粗粒度键，导致同一公司的历年申报塌缩成一个键，
    PIT 门控在"本期尚未公开"时会静默返回上一期的旧记录。
    """
    a = _filing(report=date(2024, 3, 31))
    b = _filing(report=date(2024, 6, 30))
    assert filing_key(a) != filing_key(b)
    assert "2024-03-31" in filing_key(a)


def test_filing_key_stable_for_same_period():
    a = _filing(report=date(2024, 3, 31))
    b = _filing(report=date(2024, 3, 31))
    assert filing_key(a) == filing_key(b)


# ------------------------------------------------------------------ PIT 门控
def test_published_at_is_acceptance_not_period_end():
    """published_at 必须取 acceptanceDateTime，取 reportDate 就是泄漏。"""
    f = _filing(report=date(2024, 3, 31), accepted=datetime(2024, 5, 2, 20, 15, tzinfo=UTC))
    store = build_pit_store([f])
    with store.as_of(datetime(2024, 5, 1, tzinfo=UTC)) as view:
        # 期间已结束（4/1 之后可用），但申报还没公开
        with pytest.raises(LookaheadError):
            view.get(filing_key(f))


def test_readable_after_acceptance():
    f = _filing(accepted=datetime(2024, 5, 2, 20, 15, tzinfo=UTC))
    store = build_pit_store([f])
    with store.as_of(datetime(2024, 5, 3, tzinfo=UTC)) as view:
        payload = view.get(filing_key(f))
    assert payload["form"] == "10-Q"


def test_period_based_alignment_always_raises():
    """朴素按期间对齐的每一次读取都应被拦下 —— 拦截率必须是 100%。"""
    filings = [
        _filing(
            cik=i,
            report=date(2024, 3, 31),
            accepted=datetime(2024, 5, 2, 20, 0, tzinfo=UTC),
        )
        for i in range(1, 11)
    ]
    store = build_pit_store(filings)
    period_end = datetime(2024, 3, 31, tzinfo=UTC)

    blocked = 0
    for f in filings:
        with store.as_of(period_end) as view:
            try:
                view.get(filing_key(f))
            except LookaheadError:
                blocked += 1
    assert blocked == len(filings)


def test_coarse_key_would_silently_pass():
    """反证：粗粒度键会让门控失效。

    这条测试记录了那个真实 bug —— 若键不含期间，读"本期"会拿到上期旧记录而不报错。
    """
    from finpit.pit import DatedRecord, PITStore

    coarse = PITStore("coarse")
    coarse.add(
        DatedRecord(key="1:10-Q", published_at=datetime(2023, 5, 1, tzinfo=UTC), payload="旧")
    )
    coarse.add(
        DatedRecord(key="1:10-Q", published_at=datetime(2024, 5, 2, tzinfo=UTC), payload="新")
    )

    with coarse.as_of(datetime(2024, 3, 31, tzinfo=UTC)) as view:
        # 本期尚未公开，但 get 静默返回了上期记录 —— 门控形同虚设
        assert view.get("1:10-Q") == "旧"


# ------------------------------------------------------------------ 延迟统计
def test_filing_lag_report_statistics():
    filings = [
        _filing(
            form="10-Q",
            report=date(2024, 3, 31),
            accepted=datetime(2024, 5, 2, tzinfo=UTC),
            cik=i,
        )
        for i in range(5)
    ]
    rep = filing_lag_report(filings, "10-Q")
    assert rep.n == 5
    assert rep.median_days == pytest.approx(32.0, abs=0.01)
    assert rep.min_days == rep.max_days == pytest.approx(32.0)


def test_filing_lag_report_filters_by_form():
    filings = [_filing(form="10-K"), _filing(form="10-Q"), _filing(form="10-Q")]
    assert filing_lag_report(filings, "10-K").n == 1
    assert filing_lag_report(filings, "10-Q").n == 2


def test_filing_lag_report_raises_on_empty():
    with pytest.raises(ValueError):
        filing_lag_report([_filing(form="10-K")], "10-Q")


def test_filing_lag_report_skips_missing_report_date():
    filings = [_filing(report=date(2024, 3, 31)), _filing(report=None)]
    assert filing_lag_report(filings, "10-Q").n == 1


# ---------------------------------------------------------------- 客户端礼仪
def test_default_user_agent_requires_contact(monkeypatch):
    """SEC 会拒绝不含联系方式 UA —— 与其拿到莫名的 403，不如明确报错。"""
    monkeypatch.delenv("FINPIT_SEC_CONTACT", raising=False)
    with pytest.raises(RuntimeError, match="FINPIT_SEC_CONTACT"):
        default_user_agent()


def test_default_user_agent_includes_contact(monkeypatch):
    monkeypatch.setenv("FINPIT_SEC_CONTACT", "someone@example.com")
    ua = default_user_agent()
    assert "someone@example.com" in ua
    assert "finpit-research" in ua


def test_client_uses_provided_user_agent(tmp_path):
    client = EdgarClient(tmp_path, user_agent="custom-agent/1.0")
    assert client.user_agent == "custom-agent/1.0"

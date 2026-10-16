"""时间点数据层与泄漏审计的测试。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from finpit.pit import DatedRecord, LookaheadError, MissingRecordError, PITStore, ensure_utc

T0 = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


def _store() -> PITStore:
    store = PITStore("t")
    store.add_many(
        [
            DatedRecord("px", T0 - timedelta(days=2), 100.0),
            DatedRecord("px", T0 - timedelta(days=1), 101.0),
            DatedRecord("px", T0 + timedelta(days=2), 105.0),
            DatedRecord("filing", T0 + timedelta(hours=3), {"rev": 1}),
        ]
    )
    return store


# ------------------------------------------------------------------ 基本读取
def test_latest_visible_record():
    with _store().as_of(T0) as view:
        assert view.get("px") == 101.0


def test_history_is_ascending_and_capped():
    with _store().as_of(T0) as view:
        assert [r.payload for r in view.history("px")] == [100.0, 101.0]


def test_keys_only_lists_visible():
    with _store().as_of(T0) as view:
        assert view.keys() == ["px"]


# -------------------------------------------------------------------- 越界
def test_lookahead_raises_and_reports_gap():
    with _store().as_of(T0) as view:
        with pytest.raises(LookaheadError) as exc:
            view.get("filing")
    err = exc.value
    assert err.key == "filing"
    assert err.published_at == T0 + timedelta(hours=3)


def test_future_record_not_reachable_even_though_key_exists():
    """核心保证：key 存在 ≠ 可读。"""
    store = _store()
    with store.as_of(T0) as view:
        assert view.probe("filing") is False
        # 但 px 在 T0 之后确实还有一条记录，它同样必须被挡住
        with store.as_of(T0 - timedelta(days=3)) as early:
            with pytest.raises(LookaheadError):
                early.get("px")


def test_missing_key_distinct_from_future_key():
    """"不存在"和"存在但未公开"必须是两种不同结果，否则无法定位问题。"""
    with _store().as_of(T0) as view:
        with pytest.raises(MissingRecordError):
            view.get("nonexistent")
        assert view.get("nonexistent", default=None) is None
        with pytest.raises(LookaheadError):
            view.get("filing")


def test_probe_does_not_raise_but_audits():
    store = _store()
    with store.as_of(T0) as view:
        assert view.probe("filing") is False
    report = store.audit()
    assert report.blocked_accesses == 1
    assert not report.clean


# -------------------------------------------------------------------- 审计
def test_clean_access_passes_audit():
    store = _store()
    with store.as_of(T0) as view:
        view.get("px")
    assert store.audit().clean


def test_audit_quantifies_how_early():
    store = _store()
    with store.as_of(T0) as view:
        with pytest.raises(LookaheadError):
            view.get("filing")
    report = store.audit()
    assert report.worst_violation is not None
    assert report.worst_violation.hours_early == pytest.approx(3.0)


def test_audit_reset():
    store = _store()
    with store.as_of(T0) as view:
        with pytest.raises(LookaheadError):
            view.get("filing")
    assert store.audit().blocked_accesses == 1
    store.reset_audit()
    assert store.audit().total_accesses == 0


def test_audit_to_dict_shape():
    store = _store()
    with store.as_of(T0) as view:
        with pytest.raises(LookaheadError):
            view.get("filing")
    payload = store.audit().to_dict()
    assert payload["clean"] is False
    assert payload["violations"][0]["key"] == "filing"
    assert payload["violations"][0]["hours_early"] == pytest.approx(3.0)


# ---------------------------------------------------------------- 时间处理
def test_naive_datetime_treated_as_utc():
    assert ensure_utc(datetime(2024, 1, 1)).tzinfo is UTC


def test_non_utc_aware_datetime_converted():
    from datetime import timezone

    tz = timezone(timedelta(hours=8))
    converted = ensure_utc(datetime(2024, 3, 1, 20, 0, tzinfo=tz))
    assert converted == datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


def test_record_normalises_naive_published_at():
    rec = DatedRecord("k", datetime(2024, 1, 1), 1)
    assert rec.published_at.tzinfo is UTC


# ------------------------------------------------------------------ 元信息
def test_store_metadata():
    store = _store()
    assert store.record_count == 4
    assert store.key_count == 2
    span = store.time_span()
    assert span is not None and span[0] < span[1]
    assert len(store) == 4

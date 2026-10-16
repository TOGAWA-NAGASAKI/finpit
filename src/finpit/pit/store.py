"""时间点（point-in-time）数据层 —— 本框架的核心。

## 为什么需要它

绝大多数"LLM 能预测股价"的论文都栽在同一件事上：回测时用了当时还拿不到的信息。
典型形态有三种：

1. **未来函数**：在 2024-03-01 的决策点上，读了 2024-03-05 才发布的财报。
2. **修订污染**：读了被事后修订过的宏观数据（初值 vs 终值）。
3. **训练截止污染**：用 2025 年发布的模型去"预测" 2024 年的事件，而模型早已在
   预训练语料里见过答案。

本模块用**访问门控**把第 1、2 类变成硬错误：所有数据必须声明自己的
``published_at``（真正对公众可见的时刻），而 ``as_of`` 视图只放行
``published_at <= as_of`` 的记录。任何越界访问都会被记录成一条
``AccessEvent``，最终汇总成泄漏审计报告。

## 用法

    store = PITStore("market")
    store.add_many(records)

    with store.as_of(datetime(2024, 3, 1, tzinfo=UTC)) as view:
        view.get("AAPL:close")          # 放行
        view.get("AAPL:10-K")           # 若该 10-K 在 3/5 才发布 -> LookaheadError

    report = store.audit()
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Self

from finpit.pit.errors import LookaheadError, MissingRecordError

# 哨兵：区分"未传 default"与"传了 default=None"
_MISSING: Any = object()


def ensure_utc(ts: datetime) -> datetime:
    """统一成带时区的 UTC。naive 时间一律视为 UTC，避免隐式本地时区陷阱。"""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class DatedRecord:
    """一条带"公开时刻"的数据记录。

    Attributes:
        key: 逻辑主键，例如 ``"AAPL:price:close"``、``"AAPL:filing:10-K:2024Q1"``。
        published_at: 该信息**真正对公众可得**的时刻（不是事件发生时刻，
            也不是数据里的日期字段 —— 这是最容易搞错的地方）。
        payload: 任意载荷。
        source: 来源标识，便于审计追溯。
    """

    key: str
    published_at: datetime
    payload: Any = None
    source: str = ""

    def __post_init__(self) -> None:
        # frozen dataclass 需要 object.__setattr__ 才能规范化字段
        object.__setattr__(self, "published_at", ensure_utc(self.published_at))

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return (
            f"DatedRecord(key={self.key!r}, published_at={self.published_at:%Y-%m-%d %H:%M}, "
            f"source={self.source!r})"
        )


@dataclass(frozen=True, slots=True)
class AccessEvent:
    """一次数据访问的审计记录。"""

    key: str
    as_of: datetime
    allowed: bool
    published_at: datetime | None
    reason: str

    @property
    def hours_early(self) -> float:
        """若被拦截，早用了多少小时。"""
        if self.allowed or self.published_at is None:
            return 0.0
        return (self.published_at - self.as_of).total_seconds() / 3600.0


@dataclass
class AuditReport:
    """泄漏审计结果。"""

    store: str
    total_accesses: int
    blocked_accesses: int
    distinct_blocked_keys: int
    earliest_violation: AccessEvent | None
    worst_violation: AccessEvent | None
    events: list[AccessEvent] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.blocked_accesses == 0

    def summary(self) -> str:
        if self.clean:
            return (
                f"[{self.store}] 审计通过：{self.total_accesses} 次访问，"
                f"0 次越界（无未来函数）"
            )
        assert self.worst_violation is not None
        assert self.earliest_violation is not None
        return (
            f"[{self.store}] 审计失败：{self.total_accesses} 次访问中 "
            f"{self.blocked_accesses} 次越界，涉及 {self.distinct_blocked_keys} 个键。\n"
            f"  最严重：{self.worst_violation.key} 早用了 "
            f"{self.worst_violation.hours_early:.1f} 小时\n"
            f"  最早发生：as_of={self.earliest_violation.as_of:%Y-%m-%d %H:%M} "
            f"读取 {self.earliest_violation.key}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "total_accesses": self.total_accesses,
            "blocked_accesses": self.blocked_accesses,
            "distinct_blocked_keys": self.distinct_blocked_keys,
            "clean": self.clean,
            "violations": [
                {
                    "key": e.key,
                    "as_of": e.as_of.isoformat(),
                    "published_at": e.published_at.isoformat() if e.published_at else None,
                    "hours_early": round(e.hours_early, 3),
                    "reason": e.reason,
                }
                for e in self.events
                if not e.allowed
            ],
        }


class PITView:
    """某个 ``as_of`` 时刻的数据视图。只放行已公开记录。

    通常通过 ``PITStore.as_of(...)`` 获取，可直接当作上下文管理器使用。
    """

    def __init__(self, store: PITStore, as_of: datetime) -> None:
        self._store = store
        self.as_of = ensure_utc(as_of)

    # ------------------------------------------------------------ 生命周期
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    # -------------------------------------------------------------- 读取接口
    def get(self, key: str, *, default: Any = _MISSING) -> Any:
        """取 ``key`` 在 ``as_of`` 时点可得的**最新**记录载荷。

        Raises:
            LookaheadError: 记录存在，但其 ``published_at`` 晚于 ``as_of``。
            MissingRecordError: 数据集中完全没有该 key（且有 default 时返回 default）。
        """
        candidates = self._visible(key)
        if candidates:
            return max(candidates, key=lambda r: r.published_at).payload

        future = self._future(key)
        if future:
            earliest = min(future, key=lambda r: r.published_at)
            self._store._record_access(
                AccessEvent(
                    key=key,
                    as_of=self.as_of,
                    allowed=False,
                    published_at=earliest.published_at,
                    reason="记录存在但尚未公开",
                )
            )
            raise LookaheadError(
                key,
                as_of=self.as_of,
                published_at=earliest.published_at,
                store=self._store.name,
            )

        self._store._record_access(
            AccessEvent(
                key=key,
                as_of=self.as_of,
                allowed=False,
                published_at=None,
                reason="数据集中不存在该键",
            )
        )
        if default is not _MISSING:
            return default
        raise MissingRecordError(f"数据集中不存在 {key!r} [store={self._store.name}]")

    def latest_record(self, key: str) -> DatedRecord:
        """取 ``as_of`` 时点可得的**最新**记录对象本身（含发布时刻）。"""
        candidates = self._visible(key)
        if not candidates:
            self.get(key)  # 触发统一的越界/缺失处理与审计记录
            raise MissingRecordError(key)  # pragma: no cover - get 已抛
        record = max(candidates, key=lambda r: r.published_at)
        self._store._record_access(
            AccessEvent(
                key=key,
                as_of=self.as_of,
                allowed=True,
                published_at=record.published_at,
                reason="ok",
            )
        )
        return record

    def history(self, key: str) -> list[DatedRecord]:
        """``as_of`` 之前该键的全部可见记录，按时间升序。"""
        visible = sorted(self._visible(key), key=lambda r: r.published_at)
        self._store._record_access(
            AccessEvent(
                key=key,
                as_of=self.as_of,
                allowed=True,
                published_at=visible[-1].published_at if visible else None,
                reason=f"history({len(visible)})",
            )
        )
        return visible

    def keys(self) -> list[str]:
        """当前时点可见的所有键。"""
        return sorted(k for k in self._store._by_key if self._visible(k))

    def probe(self, key: str) -> bool:
        """判断 ``key`` 在此时点是否可得，**不抛异常但记录审计**。"""
        visible = bool(self._visible(key))
        future = self._future(key)
        self._store._record_access(
            AccessEvent(
                key=key,
                as_of=self.as_of,
                allowed=visible,
                published_at=min(future, key=lambda r: r.published_at).published_at
                if future
                else None,
                reason="probe",
            )
        )
        return visible

    # ---------------------------------------------------------------- 内部
    def _visible(self, key: str) -> list[DatedRecord]:
        return [r for r in self._store._by_key.get(key, ()) if r.published_at <= self.as_of]

    def _future(self, key: str) -> list[DatedRecord]:
        return [r for r in self._store._by_key.get(key, ()) if r.published_at > self.as_of]


class PITStore:
    """带时间点门控的记录仓库。

    所有写入的记录都带 ``published_at``；所有读取都必须经过
    ``as_of(...)`` 视图，越界访问被拦下并记入审计。
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._by_key: dict[str, list[DatedRecord]] = {}
        self._events: list[AccessEvent] = []

    # ---------------------------------------------------------------- 写入
    def add(self, record: DatedRecord) -> None:
        self._by_key.setdefault(record.key, []).append(record)

    def add_many(self, records: Iterable[DatedRecord]) -> int:
        count = 0
        for record in records:
            self.add(record)
            count += 1
        return count

    def extend(self, key: str, published_at: datetime, payloads: Sequence[Any]) -> None:
        """便捷写入：同一 key 的多个时间点载荷。"""
        for payload in payloads:
            self.add(DatedRecord(key=key, published_at=published_at, payload=payload))

    # ---------------------------------------------------------------- 读取
    def as_of(self, as_of: datetime) -> PITView:
        """取得 ``as_of`` 时刻的数据视图。"""
        return PITView(self, as_of)

    def _record_access(self, event: AccessEvent) -> None:
        self._events.append(event)

    # ---------------------------------------------------------------- 审计
    def audit(self) -> AuditReport:
        """汇总所有访问，产出泄漏审计报告。"""
        blocked = [e for e in self._events if not e.allowed]
        early = [e for e in blocked if e.published_at is not None]
        return AuditReport(
            store=self.name,
            total_accesses=len(self._events),
            blocked_accesses=len(blocked),
            distinct_blocked_keys=len({e.key for e in blocked}),
            earliest_violation=min(blocked, key=lambda e: e.as_of) if blocked else None,
            worst_violation=max(early, key=lambda e: e.hours_early) if early else None,
            events=list(self._events),
        )

    def reset_audit(self) -> None:
        self._events.clear()

    # ------------------------------------------------------------- 元信息
    @property
    def record_count(self) -> int:
        return sum(len(v) for v in self._by_key.values())

    @property
    def key_count(self) -> int:
        return len(self._by_key)

    def time_span(self) -> tuple[datetime, datetime] | None:
        stamps = [r.published_at for v in self._by_key.values() for r in v]
        if not stamps:
            return None
        return min(stamps), max(stamps)

    def __iter__(self) -> Iterator[DatedRecord]:
        for records in self._by_key.values():
            yield from records

    def __len__(self) -> int:
        return self.record_count

    def __repr__(self) -> str:
        span = self.time_span()
        span_text = f", {span[0]:%Y-%m-%d}..{span[1]:%Y-%m-%d}" if span else ""
        return (
            f"PITStore(name={self.name!r}, keys={self.key_count}, "
            f"records={self.record_count}{span_text})"
        )


__all__ = [
    "AccessEvent",
    "AuditReport",
    "DatedRecord",
    "PITStore",
    "PITView",
    "ensure_utc",
]

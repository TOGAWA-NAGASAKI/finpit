"""finpit 的异常类型。"""

from __future__ import annotations

from datetime import datetime


class FinPitError(Exception):
    """本包所有异常的基类。"""


class LookaheadError(FinPitError):
    """试图在 ``as_of`` 时刻读取尚未公开的信息 —— 即未来函数。

    这是本框架最重要的异常：它把"回测里悄悄用了未来数据"从一个
    难以察觉的统计现象，变成一个当场炸出来的硬错误。
    """

    def __init__(
        self,
        key: str,
        *,
        as_of: datetime,
        published_at: datetime,
        store: str = "",
    ) -> None:
        self.key = key
        self.as_of = as_of
        self.published_at = published_at
        self.store = store
        gap = (published_at - as_of).total_seconds()
        super().__init__(
            f"未来函数：在 as_of={as_of:%Y-%m-%d %H:%M} 读取 {key!r}，"
            f"但该记录到 {published_at:%Y-%m-%d %H:%M} 才公开（早用了 {gap / 3600:.1f} 小时）"
            + (f" [store={store}]" if store else "")
        )


class MissingRecordError(FinPitError):
    """请求的记录在数据集中根本不存在（与"存在但尚未公开"不同）。"""


class ContaminationError(FinPitError):
    """检测到评测样本落在模型训练截止日之后/之前的污染区间。"""


__all__ = [
    "ContaminationError",
    "FinPitError",
    "LookaheadError",
    "MissingRecordError",
]

"""数据层：校准合成语料与真实 EDGAR 接入。"""

from __future__ import annotations

from finpit.data.edgar import (
    EdgarClient,
    Filing,
    FilingLagReport,
    build_pit_store,
    default_user_agent,
    filing_lag_report,
)
from finpit.data.synthetic import SyntheticConfig, SyntheticCorpus, generate_corpus

__all__ = [
    "EdgarClient",
    "Filing",
    "FilingLagReport",
    "SyntheticConfig",
    "SyntheticCorpus",
    "build_pit_store",
    "default_user_agent",
    "filing_lag_report",
    "generate_corpus",
]

"""finpit —— 面向 LLM 金融信号的时间点正确（point-in-time）评测框架。

核心命题：金融文本上的大模型信号之所以在论文里普遍"有效"，很大程度来自
时间泄漏（lookahead bias）、训练截止污染（contamination）与幸存者偏差。
本包提供一套强制执行时间点纪律的数据层、泄漏审计工具与可复现评测台，
用来回答一个具体问题：**剥掉泄漏之后，LLM 信号还剩多少增量？**
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

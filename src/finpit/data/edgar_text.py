"""从 SEC EDGAR 申报原文中抽取干净的叙述性句子。

## 为什么需要它

前面的「跨模型一致性 ≠ 构念效度」实验用的是**模板化合成标题**（5 正 5 负句式）。
模板措辞固定，很可能放大跨模型共同成分 —— 各模型都在对同一批关键词做同一套映射。
要判断该现象是真实的还是模板的产物，必须换成**真实文本**。

EDGAR 申报原文是理想来源：真实、多样、且已有 PIT 化的获取链路。

## 难点

10-K/10-Q 的 HTML 里混着三类内容：

1. **XBRL 元数据**（``us-gaap:...``、日期串、CIK）—— 必须剔除；
2. **财务报表表格** —— 数字为主，不是叙述性语言；
3. **叙述性正文**（MD&A、风险因素、业务描述）—— 这才是要的。

判别式很简单但有效：叙述句应当是**以字母为主、有实义、长度适中、
以句号结尾**的文本。XBRL 片段和表格行几乎都过不了这几关。
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import numpy as np

from finpit.data.edgar import (
    EdgarClient,
    Filing,
    default_user_agent,
    parse_acceptance,
)

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

# 剥离 script/style 后再剥标签
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_WS_RE = re.compile(r"[ \t\r\f\v]+")

# 明显的非叙述内容
_XBRL_HINTS = re.compile(
    r"us-gaap:|dei:|srt:|xbrli:|iso4217:|\d{4}-\d{2}-\d{2}|Member\b|"
    r"^\s*\d|CIK\d|0000\d{6}",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def strip_html(raw: str) -> str:
    """把申报 HTML 变成纯文本。"""
    text = _SCRIPT_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _ENTITY_RE.sub(" ", text)
    return _WS_RE.sub(" ", text)


def _alpha_ratio(s: str) -> float:
    letters = sum(ch.isalpha() or ch.isspace() for ch in s)
    return letters / max(len(s), 1)


def _digit_ratio(s: str) -> float:
    return sum(ch.isdigit() for ch in s) / max(len(s), 1)


def is_narrative(
    sentence: str,
    *,
    min_len: int = 90,
    max_len: int = 320,
) -> bool:
    """判断一个句子是否是**叙述性语言**（而非 XBRL 片段或表格行）。"""
    s = sentence.strip()
    if not (min_len <= len(s) <= max_len):
        return False
    if not s.endswith((".", "!", "?")):
        return False
    # XBRL / 日期 / 表格特征
    if _XBRL_HINTS.search(s):
        return False
    # 叙述文本应当以字母和空格为主
    if _alpha_ratio(s) < 0.82:
        return False
    # 数字不能太多（表格行）
    if _digit_ratio(s) > 0.06:
        return False
    # 至少要有一个真正的英文单词长度 >= 4
    if not re.search(r"[A-Za-z]{4,}", s):
        return False
    # 排除全大写（表头）
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        return False
    return True


def extract_sentences(
    raw: str,
    *,
    min_len: int = 90,
    max_len: int = 320,
    skip_leading_frac: float = 0.35,
    limit: int | None = None,
    seed: int = 20261016,
) -> list[str]:
    """从申报原文抽取叙述句。

    Args:
        skip_leading_frac: 跳过开头这一比例的内容。申报的前 30–40% 是封面页、
            目录与财务报表表格，抽出来的全是样板文字与表头；真正的叙述性正文
            （MD&A、风险因素、法律诉讼）在后面。实测 10-Q 约在 45% 处才开始。
        limit: 上限。若合格句子多于上限，会**全文均匀抽样**而非取最前面若干条，
            否则样本会全部集中在某一节，失去多样性。
    """
    text = strip_html(raw) if "<" in raw[:5000] else raw
    if skip_leading_frac > 0:
        text = text[int(len(text) * skip_leading_frac) :]

    out: list[str] = []
    seen: set[str] = set()
    for chunk in _SENTENCE_SPLIT.split(text):
        s = chunk.strip()
        if not is_narrative(s, min_len=min_len, max_len=max_len):
            continue
        key = s[:80].lower()
        if key in seen:  # 申报里有大量重复模板句
            continue
        seen.add(key)
        out.append(s)

    if limit is not None and len(out) > limit:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(out), size=limit, replace=False))
        out = [out[i] for i in idx]
    return out


@dataclass(slots=True)
class FilingText:
    """一份申报的抽取结果。"""

    cik: int
    entity: str
    form: str
    accession: str
    acceptance: str
    sentences: list[str]

    def as_dict(self) -> dict:
        return {
            "cik": self.cik,
            "entity": self.entity,
            "form": self.form,
            "accession": self.accession,
            "acceptance": self.acceptance,
            "n_sentences": len(self.sentences),
        }


class EdgarTextClient:
    """带缓存地抓取申报正文并抽取叙述句。"""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        user_agent: str | None = None,
        interval_s: float = 0.3,
        timeout: float = 90.0,
    ) -> None:
        self.client = EdgarClient(cache_dir, user_agent=user_agent, interval_s=interval_s)
        self.doc_dir = Path(cache_dir) / "edgar_docs"
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.user_agent = user_agent or default_user_agent()
        self.doc_downloads = 0

    def _doc_path(self, filing: Filing) -> Path:
        return self.doc_dir / f"{filing.cik}_{filing.accession.replace('-', '')}.txt"

    def fetch_text(self, filing: Filing, primary_doc: str, *, refresh: bool = False) -> str:
        """取一份申报的纯文本（带缓存）。"""
        path = self._doc_path(filing)
        if path.is_file() and not refresh:
            return path.read_text(encoding="utf-8", errors="ignore")

        acc = filing.accession.replace("-", "")
        url = ARCHIVE_URL.format(cik=filing.cik, accession=acc, doc=primary_doc)
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as cl:
            resp = cl.get(url, headers=headers)
        resp.raise_for_status()
        self.doc_downloads += 1
        text = strip_html(resp.text)
        path.write_text(text, encoding="utf-8")
        return text

    def filing_texts(
        self,
        cik: int,
        *,
        forms: Sequence[str] = ("10-K", "10-Q"),
        max_filings: int = 2,
        per_filing: int = 120,
        skip_leading_frac: float = 0.35,
        refresh: bool = False,
    ) -> list[FilingText]:
        """抓取某公司的若干份申报并抽句。"""
        submissions = self.client.submissions(cik, refresh=refresh)
        rec = submissions.get("filings", {}).get("recent", {})
        entity = submissions.get("name", str(cik))
        n = len(rec.get("form", []))
        wanted = set(forms)

        out: list[FilingText] = []
        for i in range(n):
            if len(out) >= max_filings:
                break
            if rec["form"][i] not in wanted:
                continue
            try:
                filing = Filing(
                    cik=cik,
                    entity=entity,
                    form=rec["form"][i],
                    report_date=None,
                    filing_date=date.fromisoformat(rec["filingDate"][i]),
                    acceptance=parse_acceptance(rec["acceptanceDateTime"][i]),
                    accession=rec["accessionNumber"][i],
                )
                raw = self.fetch_text(filing, rec["primaryDocument"][i], refresh=refresh)
            except Exception as exc:  # noqa: BLE001
                log.warning("CIK %s 第 %d 条申报抓取失败：%s", cik, i, exc)
                continue
            sents = extract_sentences(
                raw, limit=per_filing, skip_leading_frac=skip_leading_frac
            )
            if sents:
                out.append(
                    FilingText(
                        cik=cik,
                        entity=entity,
                        form=filing.form,
                        accession=filing.accession,
                        acceptance=filing.acceptance.isoformat(),
                        sentences=sents,
                    )
                )
        return out


__all__ = [
    "EdgarTextClient",
    "FilingText",
    "extract_sentences",
    "is_narrative",
    "strip_html",
]

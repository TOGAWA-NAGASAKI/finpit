"""真实 EDGAR 数据：量化"按期间对齐"造成的披露延迟泄漏。

不做任何模拟 —— 全部数字来自 SEC EDGAR 实际申报记录。

用法::

    python scripts/edgar_real_data.py            # 采样 60 家公司
    python scripts/edgar_real_data.py --n 120    # 扩大样本
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from finpit.config import export_env_file  # noqa: E402
from finpit.data.edgar import (  # noqa: E402
    TICKER_MAP_URL,
    EdgarClient,
    Filing,
    build_pit_store,
    default_user_agent,
    filing_key,
    filing_lag_report,
)
from finpit.pit import LookaheadError  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"

# 把 .env 注入环境，供 default_user_agent() 读取 FINPIT_SEC_CONTACT
export_env_file(ROOT / ".env")


def load_cik_universe(cache_dir: pathlib.Path) -> list[tuple[int, str]]:
    """取 SEC 官方的 ticker→CIK 映射，避免凭空编造 CIK。"""
    path = cache_dir / "company_tickers.json"
    if not path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(TICKER_MAP_URL, headers={"User-Agent": default_user_agent()})
            resp.raise_for_status()
            path.write_text(resp.text, encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = [(int(v["cik_str"]), str(v["title"])) for v in data.values()]
    # 按 CIK 升序去重，保证可复现
    seen: dict[int, str] = {}
    for cik, title in sorted(pairs):
        seen.setdefault(cik, title)
    return list(seen.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="采样公司数")
    ap.add_argument("--stride", type=int, default=37, help="采样步长（质数，避免聚集）")
    args = ap.parse_args()

    universe = load_cik_universe(CACHE)
    print(f"SEC 官方 ticker 映射：{len(universe)} 家公司")

    # 等距采样，起点固定 -> 完全可复现
    start = 500
    picks = universe[start : start + args.n * args.stride : args.stride][: args.n]
    print(f"等距采样 {len(picks)} 家（步长 {args.stride}，起点 {start}，可复现）\n")

    client = EdgarClient(CACHE)
    all_filings: list[Filing] = []
    failed = 0
    for i, (cik, title) in enumerate(picks, 1):
        try:
            got = client.filings(cik, forms=("10-K", "10-Q"))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [{i:3}/{len(picks)}] CIK {cik:7} {title[:34]:34} 失败: {type(exc).__name__}")
            continue
        all_filings.extend(got)
        if i % 15 == 0 or i == len(picks):
            print(f"  [{i:3}/{len(picks)}] 累计 10-K/10-Q {len(all_filings)} 条")

    print(
        f"\n网络请求 {client.network_calls} 次，缓存命中 {client.cache_hits} 次，"
        f"失败 {failed} 家"
    )
    if not all_filings:
        print("没有取到任何申报", file=sys.stderr)
        return 1

    print(f"\n合计 10-K/10-Q 申报 {len(all_filings)} 条，"
          f"覆盖 {len({f.cik for f in all_filings})} 家公司，"
          f"时间跨度 {min(f.acceptance for f in all_filings):%Y-%m-%d} .. "
          f"{max(f.acceptance for f in all_filings):%Y-%m-%d}")

    reports = []
    print("\n" + "=" * 88)
    print("真实披露延迟：会计期间结束 → 公众实际可得")
    print("=" * 88)
    for form in ("10-K", "10-Q"):
        try:
            rep = filing_lag_report(all_filings, form)
        except ValueError:
            continue
        reports.append(rep)
        print(rep.summary())
    if reports:
        med = max(r.median_days for r in reports)
        print(f"\n=> 按 reportDate 对齐的朴素做法，平均要凭空多用 {med:.0f} 天（≈"
              f"{max(r.median_weekdays for r in reports)} 个工作日）尚未公开的信息。")

    # ---------------------------------------------------------- PIT 拦截演示
    print("\n" + "=" * 88)
    print("同一份数据、两种对齐方式")
    print("=" * 88)
    store = build_pit_store(all_filings)

    accept_sorted = sorted(f.acceptance for f in all_filings)

    # 按公司看：朴素管线会不会为这家公司用到"尚未公开"的那一期申报。
    # 这才是操作上真正要命的度量 —— 基本面策略本来就该用最近一期，
    # 而"最近一期按 reportDate 算"与"最近一期按实际可得算"经常不是同一条。
    by_cik: dict[int, list[Filing]] = {}
    for f in all_filings:
        if f.report_date is not None:
            by_cik.setdefault(f.cik, []).append(f)

    print(f"{'决策时点':<20}{'有申报的公司':>12}{'朴素用错期':>12}{'占比':>9}")
    leak_rates = []
    for frac in (0.25, 0.50, 0.75, 0.95):
        ts = accept_sorted[int(frac * (len(accept_sorted) - 1))]
        n_comp = n_leak = 0
        for _cik, fs in by_cik.items():
            naive = [
                f
                for f in fs
                if datetime.combine(f.report_date, datetime.min.time(), tzinfo=UTC) <= ts  # type: ignore[arg-type]
            ]
            if not naive:
                continue
            n_comp += 1
            latest = max(naive, key=lambda f: f.report_date)  # type: ignore[arg-type,return-value]
            if latest.acceptance > ts:
                # 朴素管线以为这期可用，其实还没公开
                n_leak += 1
        rate = n_leak / n_comp if n_comp else 0.0
        leak_rates.append(
            {
                "ts": ts.isoformat(),
                "companies": n_comp,
                "using_unpublished": n_leak,
                "rate": round(rate, 4),
            }
        )
        print(f"{ts:%Y-%m-%d %H:%M}{n_comp:>12}{n_leak:>12}{rate:>9.1%}")

    if leak_rates:
        mr = sum(r["rate"] for r in leak_rates) / len(leak_rates)
        print(f"\n=> 按 reportDate 对齐的基本面管线，平均有 {mr:.1%} 的公司在用"
              "尚未公开的申报做决策。")

    # 实际触发越界，证明门控有效
    print("\n严格门控下的越界尝试（在「期间结束日」读取该期申报）：")
    leaked = checked = 0
    for f in all_filings:
        if f.report_date is None:
            continue
        period_end = datetime.combine(f.report_date, datetime.min.time(), tzinfo=UTC)
        if period_end >= f.acceptance:
            continue
        checked += 1
        with store.as_of(period_end) as view:
            try:
                view.get(filing_key(f))
            except LookaheadError:
                leaked += 1
    leak_rate = leaked / checked if checked else 0.0
    print(f"  {leaked} / {checked} 次被拦截 = {leak_rate:.1%}")
    print("  100% 才说明门控有效；低于此值意味着键粒度太粗，守卫会静默放行旧记录。")

    # ------------------------------------------------------------------ 落盘
    audit = store.audit()
    out = ROOT / "results" / "edgar_real_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "source": "SEC EDGAR submissions API (data.sec.gov)",
                "companies_sampled": len({f.cik for f in all_filings}),
                "filings_total": len(all_filings),
                "lag_reports": [r.as_dict() for r in reports],
                "period_lag_days_all": sorted(
                    round(f.period_lag_days, 2)
                    for f in all_filings
                    if f.period_lag_days is not None
                ),
                "pit_audit": {
                    "total_accesses": audit.total_accesses,
                    "blocked_accesses": audit.blocked_accesses,
                },
                "naive_vs_pit": {
                    "leak_rates_by_date": leak_rates,
                    "mean_company_leak_rate": round(
                        sum(r["rate"] for r in leak_rates) / len(leak_rates), 4
                    )
                    if leak_rates
                    else None,
                    "interception_rate_at_period_end": {
                        "blocked": leaked,
                        "checked": checked,
                        "rate": round(leak_rate, 4),
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n已写出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

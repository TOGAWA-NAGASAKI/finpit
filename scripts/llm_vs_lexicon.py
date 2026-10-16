"""LLM vs 词典 vs Oracle 的干净路径对比（单遍打分，样本量更足）。

240 行 = 40 资产 × 仅 6 个决策日，横截面日太少导致 IC 估计噪声极大
（oracle_latent 都能跑出 0.089，而真值只有 0.0356）。本脚本用更少资产、
更多交易日换取更好的时间覆盖，且只跑一遍打分以控制调用数。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from finpit.config import Settings
from finpit.data import SyntheticConfig, generate_corpus
from finpit.eval import build_panel
from finpit.eval.metrics import evaluate
from finpit.llm import LLMClient
from finpit.signals import LLMExtractor, LexiconExtractor

ASSETS = 12
ROWS = 1200
BATCH = 12

settings = Settings.load()
cfg = SyntheticConfig(n_assets=ASSETS, n_days=750, seed=20261016)
corpus = generate_corpus(cfg)
panel, _ = build_panel(corpus.store, corpus, max_rows=ROWS)
n_dates = len({r.decision_time for r in panel})
print(f"面板 {len(panel)} 行 = {ASSETS} 资产 × {n_dates} 个决策日")
print(f"全样本真实 IC     = {corpus.realized_ic():.5f}   （750 个交易日）")
print("注意：面板只取了前若干个决策日，是一个**时间切片**，")
print("      其切片内的真实相关性会偏离全样本值 —— 所以下面以 panel_true_ic 为准。")

actual = np.array([r.forward_return for r in panel])
latent = np.array([r.latent for r in panel])
# 面板内的真实 IC：oracle 按定义就等于它
panel_true_ic = evaluate(latent, actual).ic
print(f"面板内真实 IC     = {panel_true_ic:.5f}   （本对比的地面真值）\n")

results: dict[str, object] = {}

# --- Oracle 上界 ---
m = evaluate(latent, actual)
results["oracle_latent"] = m.as_dict()
print(f"{'oracle(潜在真值)':<28} {m.line()}")

# --- 词典基线 ---
lex = LexiconExtractor()
from finpit.signals import NewsItem  # noqa: E402

items = [NewsItem(asset=r.asset, text=r.text, published_at=r.news_time) for r in panel]
lex_scores = np.array([s.weighted for s in lex.score_batch(items)])
m = evaluate(lex_scores, actual)
results["lexicon"] = m.as_dict()
print(f"{'lexicon-v1':<28} {m.line()}")

# --- LLM（匿名化）---
client = LLMClient(settings)
ex = LLMExtractor(
    client=client,
    anonymize=True,
    batch_size=BATCH,
    max_tokens=8192,
    model="deepseek-v4.1-flash",
)
llm_scores = np.array([s.weighted for s in ex.score_batch(items)])
m = evaluate(llm_scores, actual)
results["llm_anon"] = m.as_dict()
print(f"{'llm:deepseek-v4.1-flash':<28} {m.line()}")

print(f"\n解析失败: {ex.parse_failures} / {len(items)}  ({ex.parse_failures / len(items):.1%})")
print("LLM 用量:", client.stats())

# ---------------------------------------------------------------- 显著性
# 面板按决策日聚簇（同一时点的横截面不独立），故用**按日分组**的 bootstrap
# 估计 IC 的标准误，而不是朴素地假设 n=1200 个独立样本。
rng = np.random.default_rng(20261016)
by_date: dict[object, list[int]] = {}
for idx, row in enumerate(panel):
    by_date.setdefault(row.decision_time, []).append(idx)
date_groups = list(by_date.values())
print(f"\n按决策日分组 bootstrap（{len(date_groups)} 组，2000 次重抽样）")

N_BOOT = 2000
boot = {
    "oracle_latent": np.empty(N_BOOT),
    "lexicon": np.empty(N_BOOT),
    "llm_anon": np.empty(N_BOOT),
    "lexicon_minus_llm": np.empty(N_BOOT),
}
for b in range(N_BOOT):
    picks = rng.integers(0, len(date_groups), len(date_groups))
    idx = np.concatenate([date_groups[i] for i in picks])
    ic_o = evaluate(latent[idx], actual[idx]).ic
    ic_l = evaluate(lex_scores[idx], actual[idx]).ic
    ic_m = evaluate(llm_scores[idx], actual[idx]).ic
    boot["oracle_latent"][b] = ic_o
    boot["lexicon"][b] = ic_l
    boot["llm_anon"][b] = ic_m
    boot["lexicon_minus_llm"][b] = ic_l - ic_m

print(f"{'方法':<24} {'IC':>8} {'bootstrap SE':>13} {'95% 区间':>22}")
for name in ("oracle_latent", "lexicon", "llm_anon"):
    arr = boot[name]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    ci = f"[{lo:+.4f}, {hi:+.4f}]"
    print(f"{name:<24} {results[name]['ic']:>+8.4f} {arr.std(ddof=1):>13.4f} {ci:>22}")

diff = boot["lexicon_minus_llm"]
lo, hi = np.percentile(diff, [2.5, 97.5])
p_two = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
print(f"\n词典 − LLM 的 IC 差 = {results['lexicon']['ic'] - results['llm_anon']['ic']:+.4f}")
print(f"  95% 区间 [{lo:+.4f}, {hi:+.4f}]   双侧 p ≈ {p_two:.3f}")
verdict = "差异不显著（区间跨 0）" if lo <= 0 <= hi else "差异显著"
print(f"  => {verdict}")

# --- 元信息 ---
results["_meta"] = {
    "n_rows": len(panel),
    "n_dates": n_dates,
    "n_assets": ASSETS,
    "true_ic_full_sample": round(corpus.realized_ic(), 5),
    "true_ic_panel": round(panel_true_ic, 5),
    "parse_failures": ex.parse_failures,
    "items_scored": len(items),
    "llm_stats": client.stats(),
}
results["_bootstrap"] = {
    "n_boot": N_BOOT,
    "n_date_groups": len(date_groups),
    "se": {k: round(float(v.std(ddof=1)), 5) for k, v in boot.items()},
    "ci95": {
        k: [round(float(x), 5) for x in np.percentile(v, [2.5, 97.5])] for k, v in boot.items()
    },
    "p_two_sided_lexicon_minus_llm": round(float(p_two), 4),
    "verdict": verdict,
}

out = pathlib.Path("results/llm_vs_lexicon.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n已写出: {out}")

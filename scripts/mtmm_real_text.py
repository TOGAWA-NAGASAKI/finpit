"""真实文本上的多质多法（MTMM）实验 —— 检验「跨模型一致性 ≠ 构念效度」。

## 为什么做这个

上一轮在**合成模板文本**上发现：5 个 LLM 彼此相关 0.97，却只与真实构念相关 0.72。
但模板措辞固定，可能**放大**了这个共同成分。要判断它是真实现象还是模板产物，
必须换到真实文本 —— 而真实文本没有真值。

MTMM 绕开了真值：用「多构念 × 多模型」的相对结构把构念方差与方法方差分开。

## 设计

* **文本**：SEC EDGAR 真实 10-K/10-Q 叙述句（MD&A、风险因素、法律诉讼）
* **构念（3 个）**：sentiment / uncertainty / forward_looking
  （Loughran-McDonald 金融语气词典的经典维度，概念上应只弱相关）
* **方法（K 个）**：不同厂商的 LLM

若判据三（MTHM > HTMM）**不通过**，说明同一模型跨构念的高度一致
压过了跨模型的同构念一致 —— 分数主要反映「这个模型怎么说话」。

用法::

    python scripts/mtmm_real_text.py --companies 18 --sentences 120
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from finpit.config import Settings, export_env_file  # noqa: E402
from finpit.data.edgar import TICKER_MAP_URL  # noqa: E402
from finpit.data.edgar_text import EdgarTextClient  # noqa: E402
from finpit.eiv import mtmm_analyze  # noqa: E402
from finpit.llm import LLMClient  # noqa: E402
from finpit.signals import MultiConstructExtractor  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
export_env_file(ROOT / ".env")

DEFAULT_MODELS = [
    "deepseek-v4.1-flash",
    "glm-5.3-flash",
    "kimi-k3",
    "mimo-v2.6-flash",
]


def load_universe() -> list[tuple[int, str]]:
    import httpx

    from finpit.data.edgar import default_user_agent

    path = CACHE / "company_tickers.json"
    if not path.is_file():
        with httpx.Client(timeout=60, follow_redirects=True) as cl:
            r = cl.get(TICKER_MAP_URL, headers={"User-Agent": default_user_agent()})
            r.raise_for_status()
            path.write_text(r.text, encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    seen: dict[int, str] = {}
    for v in sorted(data.values(), key=lambda d: int(d["cik_str"])):
        seen.setdefault(int(v["cik_str"]), str(v["title"]))
    return list(seen.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--companies", type=int, default=18)
    ap.add_argument("--stride", type=int, default=613, help="采样步长（质数）")
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--sentences", type=int, default=120, help="最终用于 MTMM 的句子数")
    ap.add_argument("--per-filing", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()

    # ---------------------------------------------------------- 抓真实文本
    universe = load_universe()
    picks = universe[args.start : args.start + args.companies * args.stride : args.stride][
        : args.companies
    ]
    print(f"从 SEC 官方映射采样 {len(picks)} 家公司：")
    for cik, title in picks[:5]:
        print(f"  CIK {cik:7} {title[:46]}")
    if len(picks) > 5:
        print(f"  … 其余 {len(picks) - 5} 家")

    tc = EdgarTextClient(CACHE)
    pool: list[tuple[str, str]] = []  # (公司, 句子)
    for i, (cik, title) in enumerate(picks, 1):
        try:
            fts = tc.filing_texts(cik, max_filings=1, per_filing=args.per_filing)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:2}/{len(picks)}] CIK {cik} 失败: {type(exc).__name__}")
            continue
        for ft in fts:
            pool.extend((ft.entity, s) for s in ft.sentences)
        print(
            f"  [{i:2}/{len(picks)}] {title[:34]:34} 累计句子 {len(pool)}"
        )

    if len(pool) < args.sentences:
        print(f"句子不足（{len(pool)} < {args.sentences}）", file=sys.stderr)
        return 1

    rng = np.random.default_rng(20261016)
    idx = np.sort(rng.choice(len(pool), size=args.sentences, replace=False))
    sample = [pool[i] for i in idx]
    texts = [s for _, s in sample]
    entities = {e for e, _ in sample}

    print(
        f"\n最终样本：{len(texts)} 句，来自 {len(entities)} 家公司，"
        f"平均句长 {np.mean([len(t) for t in texts]):.0f} 字符"
    )
    print(f"文档下载 {tc.doc_downloads} 次，网络请求 {tc.client.network_calls} 次\n")
    print("样例：")
    for t in texts[:3]:
        print(f"  · {t[:140]}")

    # -------------------------------------------------------- 多构念打分
    settings = Settings.load()
    settings.require_llm()
    client = LLMClient(settings)

    k = len(args.models)
    construct_names = [c.name for c in MultiConstructExtractor(client=client).constructs]
    c = len(construct_names)
    cube = np.zeros((len(texts), c, k))
    per_model: list[dict] = []

    print(f"\n打分：{c} 个构念 × {k} 个模型 = {c * k} 条变量，"
          f"共 {len(texts)} 句\n")
    for j, model in enumerate(args.models):
        ex = MultiConstructExtractor(
            client=client,
            batch_size=args.batch_size,
            model=model,
            max_tokens=8192,
        )
        rows = ex.score_texts(texts)
        for ci, name in enumerate(construct_names):
            cube[:, ci, j] = [r[name] for r in rows]
        means = {n: float(np.mean(cube[:, ci, j])) for ci, n in enumerate(construct_names)}
        per_model.append(
            {
                "model": model,
                "means": {n: round(v, 4) for n, v in means.items()},
                "parse_failures": ex.parse_failures,
            }
        )
        print(
            f"  {model:26} "
            + "  ".join(f"{n}={means[n]:+.3f}" for n in construct_names)
            + f"   解析失败={ex.parse_failures}"
        )

    # -------------------------------------------------------- MTMM 分析
    print()
    result = mtmm_analyze(cube, construct_names=construct_names, method_names=args.models)
    print(result.summary())

    # 补充：各模型跨构念的平均相关（方法方差的直接读数）
    print("\n  逐模型：该模型在三个构念上的打分两两相关（方法方差直接读数）")
    for j, model in enumerate(args.models):
        cc = np.corrcoef(cube[:, :, j], rowvar=False)
        off = cc[np.triu_indices(c, 1)]
        print(f"    {model:26} 平均跨构念相关 = {off.mean():+.4f}   {np.round(off, 3)}")

    print(f"\nLLM 用量: {client.stats()}")

    # ------------------------------------------------------------------ 落盘
    out = ROOT / "results" / "mtmm_real_text.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "data_source": "SEC EDGAR 10-K/10-Q 叙述句（真实文本）",
                "n_sentences": len(texts),
                "n_companies": len(entities),
                "companies": sorted(entities),
                "models": args.models,
                "constructs": construct_names,
                "per_model_means": per_model,
                "mtmm": result.as_dict(),
                "cross_construct_within_model": {
                    model: round(float(
                        np.corrcoef(cube[:, :, j], rowvar=False)[np.triu_indices(c, 1)].mean()
                    ), 5)
                    for j, model in enumerate(args.models)
                },
                "llm_stats": client.stats(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n已写出: {out}")

    # 原始打分一并落盘，便于日后换分析口径而无需重新调用 LLM
    raw_out = ROOT / "results" / "mtmm_raw_scores.json"
    raw_out.write_text(
        json.dumps(
            {
                "constructs": construct_names,
                "models": args.models,
                "texts": texts,
                "entities": [e for e, _ in sample],
                "cube": cube.tolist(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"已写出: {raw_out}（原始打分，可离线重新分析）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

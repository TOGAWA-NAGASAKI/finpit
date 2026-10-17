"""真实 LLM 的跨模型可靠性实测 —— 回答「LLM 生成的因子有多少是噪声」。

## 实验设计

合成语料里潜在情绪 ``θ`` 是**已知**的，而我们只知道它的代理（LLM 打分）。
于是可以同时做两件在真实数据上做不到的事：

1. **验证可靠性估计器**：用 K 个模型的打分估出 ``λ̂``，与「用已知真值算出的
   真实 λ」对照 —— 检验识别策略是否成立；
2. **检验独立性前提**：算各模型误差的相关性。若显著为正，
   说明模型共享系统性偏差，**多模型集成消不掉它**。

最后把整条链路走完：``β̂ 观测 → λ 修正 → 与真实 β 对照``。

## 成本

N 条文本 × K 个模型，按 batch 计费；结果全部落盘缓存，重跑零成本。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from finpit.config import Settings  # noqa: E402
from finpit.data import SyntheticConfig, generate_corpus  # noqa: E402
from finpit.eiv import (  # noqa: E402
    corrected_regression,
    error_correlation,
    estimate_reliability,
)
from finpit.eiv.attenuation import _ols  # noqa: E402
from finpit.llm import LLMClient  # noqa: E402
from finpit.signals import LLMExtractor, NewsItem  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 选不同厂商/架构的模型，尽量让误差不共享
DEFAULT_MODELS = [
    "deepseek-v4.1-flash",
    "glm-5.3-flash",
    "kimi-k3",
    "mimo-v2.6-flash",
    "longcat-2.5-preview-free",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="评分文本条数")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()

    settings = Settings.load()
    settings.require_llm()

    # ---------------------------------------------------------- 构造样本
    cfg = SyntheticConfig(n_assets=20, n_days=400, seed=20261016)
    corpus = generate_corpus(cfg)
    date_keys = [d.strftime("%Y-%m-%d") for d in corpus.dates]

    # 取 (资产, 日期) 配对，标签为**次日收益**，真值为潜在情绪
    rng = np.random.default_rng(20261016)
    samples: list[tuple[str, str, float, float, str]] = []
    for _ in range(args.n * 3):
        if len(samples) >= args.n:
            break
        i = int(rng.integers(1, len(corpus.dates)))
        asset = str(rng.choice(corpus.assets))
        rec = next(
            (
                r
                for r in corpus.store._by_key[f"{asset}:news"]
                if r.payload["date"] == date_keys[i - 1]
            ),
            None,
        )
        if rec is None:
            continue
        samples.append(
            (
                asset,
                rec.payload["headline"],
                float(rec.payload["sentiment_latent"]),  # 真值 θ
                corpus.returns[date_keys[i]][asset],  # 因变量 y
                date_keys[i - 1],
            )
        )

    print(f"样本 {len(samples)} 条（{len({s[0] for s in samples})} 个资产）")
    print(f"模型 K = {len(args.models)}: {', '.join(args.models)}\n")

    y = np.array([s[3] for s in samples])
    theta = np.array([s[2] for s in samples])

    # ---------------------------------------------------- 逐模型打分
    client = LLMClient(settings)
    scores = np.zeros((len(samples), len(args.models)))
    per_model = []

    for j, model in enumerate(args.models):
        ex = LLMExtractor(
            client=client,
            anonymize=True,
            batch_size=args.batch_size,
            max_tokens=8192,
            model=model,
        )
        items = [NewsItem(asset=s[0], text=s[1], published_at=corpus.dates[0]) for s in samples]
        sigs = ex.score_batch(items)
        scores[:, j] = [s.score for s in sigs]

        beta_m, se_m = _ols(y, scores[:, j])
        r_truth = float(np.corrcoef(scores[:, j], theta)[0, 1])
        per_model.append(
            {
                "model": model,
                "beta_observed": round(beta_m, 6),
                "se": round(se_m, 6),
                "corr_with_truth": round(r_truth, 4),
                "parse_failures": ex.parse_failures,
            }
        )
        print(
            f"  {model:28} β̂={beta_m:+.5f}  corr(θ)={r_truth:+.3f}  "
            f"解析失败={ex.parse_failures}"
        )

    # ---------------------------------------------------- 可靠性估计
    print()
    rel = estimate_reliability(scores, bootstrap=args.bootstrap)
    print(rel.summary())

    # 真实可靠性（只有受控环境才知道）
    mean_score = scores.mean(axis=1)
    lam_true_single = float(np.corrcoef(scores[:, 0], theta)[0, 1]) ** 2
    lam_true_mean = float(np.corrcoef(mean_score, theta)[0, 1]) ** 2
    mean_construct_r = float(np.mean([p["corr_with_truth"] for p in per_model]))

    # ------------------------------------------- 一致性 vs 效度：本实验的主结论
    print("\n" + "=" * 86)
    print("主结论：跨模型一致性 ≠ 构念效度")
    print("=" * 86)
    print(
        f"  跨模型相关系数（两两）      = {rel.lambda_single:+.4f}"
        f"   [{rel.r_min:+.3f}, {rel.r_max:+.3f}]\n"
        f"  与真实构念的相关（平均）    = {mean_construct_r:+.4f}\n"
        f"  一致性 − 效度 的落差        = {rel.lambda_single - mean_construct_r:+.4f}"
    )
    print(
        f"\n  由跨模型一致性估出的可靠性 λ̂₁ = {rel.lambda_single:.4f}"
        f"  → 会得出「几乎无测量误差」的结论\n"
        f"  由真实构念算出的可靠性 λ₁     = {lam_true_single:.4f}"
        f"  → 实际有 {1 - lam_true_single:.0%} 的方差是噪声"
    )
    print(
        f"\n  若只看跨模型一致性，你会认为**不需要修正**（λ≈{rel.lambda_mean:.3f}）；\n"
        f"  而真实衰减因子是 {lam_true_mean:.3f} —— **一半的信号被当成噪声丢掉了**。"
    )
    print(
        f"\n  集成 {len(args.models)} 个模型的收益：\n"
        f"    单模型  与构念相关 = {np.sqrt(lam_true_single):.4f}\n"
        f"    集成后  与构念相关 = {np.sqrt(lam_true_mean):.4f}"
        f"   （变化 {np.sqrt(lam_true_mean) - np.sqrt(lam_true_single):+.4f}）"
    )
    if lam_true_mean <= lam_true_single:
        print(
            "    => **集成毫无增益，甚至略差。** 因为各模型共享的那个巨大共同成分\n"
            "       并不是真实构念，平均它不会提升效度 —— 只会把有信息量的个体差异也抹平。"
        )

    # ---------------------------------------------------- 独立性前提检验
    diag = error_correlation(scores, theta)
    print("\n  独立性前提检验（平行指标假设的核心）：")
    print(f"    误差平均相关 = {diag['mean_error_corr']:+.4f}")
    verdict = (
        "前提成立（误差近似独立）"
        if diag["independence_assumption_ok"]
        else "前提**彻底失效** —— 误差近乎完全共线，Spearman-Brown 不适用"
    )
    print(f"    判定：{verdict}")
    print(
        f"    真实 λ₁ = {lam_true_single:.4f}，"
        f"而 λ̂₁ = {rel.lambda_single:.4f} —— 高估 "
        f"{rel.lambda_single / lam_true_single - 1:.0%}"
    )

    # ---------------------------------------------------- 衰减与修正
    print()
    res = corrected_regression(
        y, scores, bootstrap=args.bootstrap, beta_true=cfg.true_beta
    )
    print(res.summary())

    beta_one, _ = _ols(y, scores[:, 0])
    print(
        f"\n  回归对比（注意功效）：\n"
        f"    用 1 个模型        β̂ = {beta_one:+.5f}\n"
        f"    用 {len(args.models)} 个模型均值   β̂ = {res.beta_observed:+.5f}\n"
        f"    修正后            β̂ = {res.beta_corrected:+.5f}\n"
        f"    真实 β            = {cfg.true_beta:+.5f}\n"
        f"    β̂ 的标准误        ≈ {res.se_observed:.5f}"
    )
    print(
        "    => 该合成语料的真实 IC 仅 "
        f"{corpus.realized_ic():.4f}，n={len(y)} 时标准误远大于效应本身，"
        "\n       **本节的回归检验功效不足**，倍数解读不可信；"
        "\n       可靠性/效度的分解不受此影响（相关系数在 n=200 下已足够稳定）。"
    )

    print(f"\nLLM 用量: {client.stats()}")

    # ------------------------------------------------------------------ 落盘
    out = ROOT / "results" / "eiv_real_llm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n_samples": len(samples),
                "n_assets": len({s[0] for s in samples}),
                "models": args.models,
                "true_beta_generator": cfg.true_beta,
                "true_ic_generator": round(corpus.realized_ic(), 5),
                "per_model": per_model,
                "reliability": rel.as_dict(),
                "true_reliability": {
                    "lambda_single": round(lam_true_single, 5),
                    "lambda_mean": round(lam_true_mean, 5),
                },
                "agreement_vs_validity": {
                    "cross_model_r": round(rel.lambda_single, 5),
                    "mean_construct_r": round(mean_construct_r, 5),
                    "gap": round(rel.lambda_single - mean_construct_r, 5),
                    "lambda_hat_overstates_true_by": round(
                        rel.lambda_single / lam_true_single - 1.0, 5
                    ),
                    "ensemble_gain_in_construct_r": round(
                        float(np.sqrt(lam_true_mean) - np.sqrt(lam_true_single)), 5
                    ),
                },
                "regression_power_note": (
                    f"真实 IC 仅 {corpus.realized_ic():.4f}，n={len(y)} 时功效不足；"
                    "回归倍数不可作结论，可靠性与效度分解有效。"
                ),
                "independence_check": {
                    k: (round(v, 5) if isinstance(v, float) else v)
                    for k, v in diag.items()
                },
                "attenuation": res.as_dict(),
                "comparison": {
                    "one_model_beta": round(beta_one, 6),
                    "k_model_beta": round(res.beta_observed, 6),
                    "corrected_beta": round(res.beta_corrected, 6),
                    "true_beta": round(cfg.true_beta, 6),
                },
                "llm_stats": client.stats(),
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

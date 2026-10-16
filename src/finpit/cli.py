"""finpit 命令行入口。

    finpit corpus                      查看合成语料概况
    finpit ablation --extractor lexicon  跑泄漏消融（不需要 API）
    finpit ablation --extractor llm      用大模型跑（需要 .env 里的 key）
    finpit demo                        一键复现全部结果
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from finpit.audit import SplitAuditor, simulate_persistent_panel
from finpit.config import Settings
from finpit.data import SyntheticConfig, generate_corpus
from finpit.eval import LeakMode, LeakageAblation, build_panel
from finpit.signals import LexiconExtractor

# 默认报告落盘位置
DEFAULT_OUT_DIR = Path("results")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------- corpus
def cmd_corpus(args: argparse.Namespace) -> int:
    cfg = SyntheticConfig(n_assets=args.assets, n_days=args.days, seed=args.seed)
    corpus = generate_corpus(cfg)
    print(corpus.describe())
    print(f"  真实回归系数 beta = {cfg.true_beta:.3e}")
    print(f"  实测 IC（上帝视角上限） = {corpus.realized_ic():.5f}")
    print(f"  交易日跨度 = {corpus.dates[0]:%Y-%m-%d} .. {corpus.dates[-1]:%Y-%m-%d}")
    print()
    print("样例新闻（前 3 条）：")
    for asset in corpus.assets[:1]:
        for record in corpus.store._by_key[f"{asset}:news"][:3]:
            print(f"  [{record.published_at:%Y-%m-%d %H:%M}] {record.payload['headline']}")
    return 0


# ------------------------------------------------------------------- ablation
def _build_extractor(args: argparse.Namespace, settings: Settings):
    if args.extractor == "lexicon":
        return LexiconExtractor()
    if args.extractor == "llm":
        from finpit.llm import LLMClient
        from finpit.signals import LLMExtractor

        settings.require_llm()
        client = LLMClient(settings)
        return LLMExtractor(
            client=client,
            anonymize=not args.no_anonymize,
            batch_size=args.batch_size,
        )
    raise ValueError(f"未知抽取器: {args.extractor}")


def cmd_ablation(args: argparse.Namespace) -> int:
    settings = Settings.load()
    if args.model:
        settings.model = args.model

    cfg = SyntheticConfig(n_assets=args.assets, n_days=args.days, seed=args.seed)
    corpus = generate_corpus(cfg)
    print(corpus.describe())
    print()

    extractor = _build_extractor(args, settings)
    modes = [LeakMode(m) for m in args.modes] if args.modes else list(LeakMode)
    ablation = LeakageAblation(corpus, extractor, max_rows=args.rows)
    report = ablation.run(modes)

    print(report.table())
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ablation_{args.extractor}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{stem}.md").write_text(report.table() + "\n", encoding="utf-8")
    print(f"已写出: {out_dir / (stem + '.json')}")
    print(f"已写出: {out_dir / (stem + '.md')}")

    if hasattr(extractor, "stats"):
        print()
        print("抽取器统计:", json.dumps(extractor.stats(), ensure_ascii=False))
    if hasattr(extractor, "client"):
        print("LLM 用量:", json.dumps(extractor.client.stats(), ensure_ascii=False))
    return 0


# ----------------------------------------------------------------- split audit
def cmd_split_audit(args: argparse.Namespace) -> int:
    """随机切分 vs 时序切分的对照 —— 量化最常见的泄漏。"""
    if args.demo:
        print("模式：受控模拟（标签具时间持续性，用来验证审计器灵敏度）")
        texts, labels, stamps = simulate_persistent_panel(seed=args.seed)
    else:
        cfg = SyntheticConfig(n_assets=args.assets, n_days=args.days, seed=args.seed)
        corpus = generate_corpus(cfg)
        panel, _ = build_panel(corpus.store, corpus, max_rows=args.rows)
        if not panel:
            print("面板为空", file=sys.stderr)
            return 1
        texts = [row.text for row in panel]
        labels = [1 if row.forward_return > 0 else 0 for row in panel]
        stamps = [row.decision_time for row in panel]
        print(f"样本 {len(panel)} 行，标签 = 次日收益符号")
        print("提示：该任务信号极弱（LC 0.02~0.04），符号分类接近抛硬币，")
        print("      因此本模式通常**测不出**切分泄漏 —— 这是正确结论。")
        print("      若要看审计器确实能发现泄漏，请加 --demo。")

    print(f"特征 = 文本字符二元组（{args.dim} 维）；正类占比 {sum(labels) / len(labels):.3f}")
    print()

    auditor = SplitAuditor(
        n_folds=args.folds, dim=args.dim, epochs=args.epochs, seed=args.seed
    )
    result = auditor.audit(texts, labels, stamps)
    print(result.summary())
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "split_audit_demo.json" if args.demo else "split_audit.json"
    (out_dir / name).write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写出: {out_dir / name}")
    return 0


# ----------------------------------------------------------------------- demo
def cmd_demo(args: argparse.Namespace) -> int:
    """一键复现：词典基线的完整消融 + 审计。"""
    settings = Settings.load()
    cfg = SyntheticConfig(n_assets=args.assets, n_days=args.days, seed=args.seed)
    corpus = generate_corpus(cfg)
    print("=" * 78)
    print(corpus.describe())
    print(f"上帝视角真实 IC = {corpus.realized_ic():.5f}")
    print("=" * 78)
    print()

    extractor = LexiconExtractor()
    ablation = LeakageAblation(corpus, extractor, max_rows=args.rows)
    report = ablation.run()
    print(report.table())
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "demo.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "demo.md").write_text(report.table() + "\n", encoding="utf-8")
    print(f"结果已写入 {out_dir}/demo.md 与 demo.json")

    if settings.llm_configured:
        print()
        print("检测到已配置 LLM，可运行：finpit ablation --extractor llm --rows 800")
    return 0


# ---------------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finpit",
        description="面向 LLM 金融信号的时间点正确评测与泄漏消融",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="打开调试日志")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--assets", type=int, default=40, help="资产数（默认 40）")
        p.add_argument("--days", type=int, default=750, help="交易日数（默认 750）")
        p.add_argument("--seed", type=int, default=20261016, help="随机种子")
        p.add_argument("--rows", type=int, default=8000, help="评测面板行数上限")
        p.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="结果输出目录")

    p_corpus = sub.add_parser("corpus", help="查看合成语料概况")
    add_common(p_corpus)
    p_corpus.set_defaults(func=cmd_corpus)

    p_abl = sub.add_parser("ablation", help="跑泄漏消融实验")
    add_common(p_abl)
    p_abl.add_argument(
        "--extractor", choices=["lexicon", "llm"], default="lexicon", help="信号抽取器"
    )
    p_abl.add_argument("--model", default=None, help="覆盖 LLM 模型名")
    p_abl.add_argument("--batch-size", type=int, default=8, help="LLM 批量打分条数")
    p_abl.add_argument(
        "--no-anonymize",
        action="store_true",
        help="关闭匿名化（用于对照训练截止污染）",
    )
    p_abl.add_argument(
        "--modes",
        nargs="*",
        default=None,
        help=f"只跑指定模式，可选: {', '.join(m.value for m in LeakMode)}",
    )
    p_abl.set_defaults(func=cmd_ablation)

    p_demo = sub.add_parser("demo", help="一键复现全部结果")
    add_common(p_demo)
    p_demo.set_defaults(func=cmd_demo)

    p_split = sub.add_parser("split-audit", help="随机切分 vs 时序切分的泄漏审计")
    add_common(p_split)
    p_split.add_argument("--folds", type=int, default=5, help="折数（默认 5）")
    p_split.add_argument("--dim", type=int, default=512, help="哈希特征维度")
    p_split.add_argument("--epochs", type=int, default=120, help="逻辑回归迭代轮数")
    p_split.add_argument(
        "--demo",
        action="store_true",
        help="改用受控模拟面板（标签具时间持续性），用于验证审计器灵敏度",
    )
    p_split.set_defaults(func=cmd_split_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

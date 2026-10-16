# 论文调研与想法提取

> 本文件记录 `finpit` 的文献依据。**全部 arXiv ID 均经 arXiv Atom API 逐一独立回验，0 虚构。**
> 回验方式：`https://export.arxiv.org/api/query?id_list=<id>`，核对标题与 published 日期。

## 1. 检索方法

| 项 | 值 |
|---|---|
| 数据源 | arXiv 官方 Atom API（`https://export.arxiv.org/api/query`） |
| 查询组数 | 127 组（38 长文档 + 48 情绪事件 + 41 Agent） |
| 排序 | `sortBy=submittedDate&sortOrder=descending` |
| 时间窗 | 2023-01 – 2026-10（当前日期 2026-10-16） |
| 去重命中 | 3,233 篇（988 + 1,285 + 960） |
| 年度分布 | 2023 显著少 → **2026 年占约 60%** |

**踩坑记录**（供复现）：
- arXiv API 在本机 `http://` 超时，必须用 `https://`；
- 请求间隔 **8 秒触发 HTTP 429 限流**，**18–20 秒稳定**；
- `all:"FLARE"` 被天体物理术语（太阳耀斑）严重污染，金融基准检索不可用。

**证据边界**：所有判断基于**摘要 + 元数据**，未读全文 PDF 与代码。凡涉及论文内部细节一律标注「未核实」，并严格区分「摘要未披露 X」与「作者没做 X」。

---

## 2. 三个子方向的结论

### 2.1 长文档理解与 RAG —— **已饱和，不建议进入**

单文档单表单的 10-K 纯文本 QA（FinanceBench → SEC-QA → SECQUE → Fin-RATE/FinRank/FinAuditing）问题形态已无增量；朴素 RAG 配方（dense+BM25+rerank+固定 chunk）已在 2026 年被当作 baseline 超越；通用多模态金融榜单高度密集。

**值得做但不能是"又一个基准"**：页/段落级**文档内检索失败**诊断（整池仅 [`2602.17981`](https://arxiv.org/abs/2602.17981) 一篇直接研究）；跨表跨文档数值推理与**勾稽对账**（[`2604.03664`](https://arxiv.org/abs/2604.03664) 明确指出既有基准 "largely single-table"）；**XBRL 与 LLM 接地**（`all:XBRL` 全库仅 16 条命中，几乎无人做"自然语言数字 → 标准 XBRL 标签"映射）。

### 2.2 情绪 / 事件抽取 —— **"LLM vs FinBERT"已成熟，但失败模式有空白**

一句话结论：**LLM 在语言层面追平了 FinBERT，在经济层面没有超过，并带来 FinBERT 没有的新失败模式。**

- **增益分三层**：语言/分类层追平甚至超越（[`2410.01987`](https://arxiv.org/abs/2410.01987)、[`2512.00946`](https://arxiv.org/abs/2512.00946)）；**经济/收益层没有增益甚至更差**（[`2609.20198`](https://arxiv.org/abs/2609.20198)、[`2608.04200`](https://arxiv.org/abs/2608.04200)、[`2602.00086`](https://arxiv.org/abs/2602.00086) 三篇独立证实）；**真正增量在结构化互补信息**（[`2607.28496`](https://arxiv.org/abs/2607.28496) 报告 LLM 结构化特征与 FinBERT 情绪 **53.5% 不一致**，融合显著更优）。
- **反向证据必须并列**：FinBERT2 在 5 个分类任务超 LLM **9.7–12.3%**（[`2506.06335`](https://arxiv.org/abs/2506.06335)）；工业级对照中 8 组任务 FinBERT 有 7 组 F1 更高（[`2607.27611`](https://arxiv.org/abs/2607.27611)）。
- **最值得做的是失败模式，不是又一次总体对比。** 其中两条被独立证实、且交集处仍空白：
  - **公司特定偏见**：匿名化后模型表现反而**更好**（[`2309.17322`](https://arxiv.org/abs/2309.17322)、[`2411.00420`](https://arxiv.org/abs/2411.00420)）—— 说明模型测的是**公司先验**而非文本情绪。
  - **依赖非信息线索**：6 个模型中有 5 个的**语调效应大于 EPS 效应**（[`2610.22043`](https://arxiv.org/abs/2610.22043)）。

### 2.3 LLM 交易 Agent —— **回测可信度是核心问题**

四篇独立工作同向证实：**"组合暴露"被误读为"选择能力"**。

| 论文 | 关键数字 |
|---|---|
| [`2610.04040`](https://arxiv.org/abs/2610.04040) | 标准零中心配对检验在**零技能**下假阳性率 **11.6%**（transition-matched 降至 5.3%）；723 个 events 上 gross +15.2 bps 分解为 composition **+25.8** / selection **−10.6** |
| [`2605.28359`](https://arxiv.org/abs/2605.28359) | 标识/日历掩码 + Barra 归因 → 累计收益**主要由被动市场与风格暴露解释** |
| [`2606.29771`](https://arxiv.org/abs/2606.29771) | *"apparent alpha can dissolve once look-ahead bias and trading costs are controlled"* |
| [`2608.28399`](https://arxiv.org/abs/2608.28399) | 暴露匹配度量下 LLM 的 long/flat 择时**持续为负** |

**决定性证据**：[`2608.27734`](https://arxiv.org/abs/2608.27734) 中一个**故意泄漏的 oracle 报 Sharpe=35，完整通过 Deflated Sharpe 与 PBO 检验**；同文在 453 只 PIT 美股 + 39 ETF、含真实成本下，**拒绝每一个 LLM 发现的策略**。

**多智能体结构本身证据薄弱**：[`2609.29701`](https://arxiv.org/abs/2609.29701) 的 210 次受控运行显示**推理质量与 Sharpe 无关**（r=0.07, p=0.29）、与收益无关（r=0.03, p=0.70）；结构化提示把推理分从 0.72 提到 0.84（d≈2.0）但**不转化为收益**；只有保留分歧的 JSD 干预提升 Sharpe +0.14 (p=0.028)。

---

## 3. 提取出的论文级想法

按「可复现性 × 数据可得性 × 想法密度」排序。**★ 表示本项目已实现。**

| # | 想法 | 来源 | 状态 |
|---|---|---|---|
| 1 | **时间点门控数据访问层**：所有读取经 `as_of` 视图，越界抛硬错误并量化早用时长 | [`2608.27734`](https://arxiv.org/abs/2608.27734)、[`2607.04958`](https://arxiv.org/abs/2607.04958)、[`2608.09988`](https://arxiv.org/abs/2608.09988) | ★ 已实现 |
| 2 | **泄漏消融基准**：同一条信号跑多种陋习流程，量化各自虚高 | [`2608.17223`](https://arxiv.org/abs/2608.17223)、[`2605.16895`](https://arxiv.org/abs/2605.16895) | ★ 已实现 |
| 3 | **切分泄漏审计**：同一特征/模型做随机 vs 时序切分对照 | [`2608.17223`](https://arxiv.org/abs/2608.17223) | ★ 已实现 |
| 4 | **匿名化作为污染控制**：抹掉标的与日期，堵住"记忆作弊"通道 | [`2309.17322`](https://arxiv.org/abs/2309.17322)、[`2411.00420`](https://arxiv.org/abs/2411.00420)、[`2605.28359`](https://arxiv.org/abs/2605.28359) | ★ 已实现（`LLMExtractor(anonymize=)`） |
| 5 | **Oracle 上界基准**：直读潜在真值，量化"文本抽取损失"，取代"对比随机猜" | 本项目设计 | ★ 已实现 |
| 6 | **多指标联合诊断**：相关性指标与水平指标分列，因不同泄漏污染不同维度 | [`2610.04040`](https://arxiv.org/abs/2610.04040) | ★ 已实现 |
| 7 | **Availability Contract 校验器**：强制 `input_boundary` / `outcome_start`，`outcome_start < input_boundary` 直接拒绝 | [`2610.21659`](https://arxiv.org/abs/2610.21659) | 路线图 |
| 8 | **LAP 泄漏探针**：只给日期的回忆查询，输出 `LAP∈[0,1]`；回归加 `LAP×forecast` 交互项 | [`2512.23847`](https://arxiv.org/abs/2512.23847) | 路线图 |
| 9 | **Trial Ledger 搜索强度记账**：累计试验数 → 驱动 Deflated Sharpe / PBO / FDR | [`2608.27734`](https://arxiv.org/abs/2608.27734)、[`2610.17130`](https://arxiv.org/abs/2610.17130) | 路线图 |
| 10 | **确定性风险外壳**：约束执行在 LLM 之外（SoK 审计发现 7 个开源实现无一在 LLM 外强制风控） | [`2609.19705`](https://arxiv.org/abs/2609.19705) | 未采纳 |
| 11 | **Transition-matched 反事实审计**：把收益拆成 deployment / composition / selection 三段 + 置换检验 | [`2610.04040`](https://arxiv.org/abs/2610.04040) | 未采纳 |
| 12 | **双效度报告**：构念效度与预测效度强制分栏，禁止用前者论证后者 | [`2609.11144`](https://arxiv.org/abs/2609.11144)、[`2609.20198`](https://arxiv.org/abs/2609.20198) | 部分（指标已分列） |
| 13 | **页级中间检索器**：消灭"文档检索对、页/chunk 错"的主因失败 | [`2602.17981`](https://arxiv.org/abs/2602.17981) | 未采纳 |
| 14 | **结构感知 chunker**：按元素类型切分，表格绝不与表头脚注被切开 | [`2402.05131`](https://arxiv.org/abs/2402.05131) | 未采纳 |
| 15 | **可执行验证层**：论断分型核验，"计算推导型"必须走公式重建 + 算术执行 | [`2608.10213`](https://arxiv.org/abs/2608.10213)、[`2604.23588`](https://arxiv.org/abs/2604.23588) | 路线图 |

---

## 4. 明确空白（值得投入）

1. **LLM 生成变量的测量误差的经济含义** —— 衰减偏误如何修正？能否用多模型集成做潜变量估计？计量 × LLM 交叉口，几乎无人做且不需大算力。[`2609.31013`](https://arxiv.org/abs/2609.31013) 报告跨模型秩相关仅 **0.52**，下游回归系数**符号**都会变 —— 这是最大空白。
2. **统一开源泄漏审计工具链** —— 成果散落在 10+ 篇，**不存在开箱即用的包**。本项目是对这一空白的直接回应。
3. **事件内部时序可用性**的工程化工具链（时序切分无法防止该类错配）。
4. **因果识别** —— 现有工作全是预测性关联，分不清"预测"还是"复述"。
5. **中文金融文本泄漏审计** —— 制度差异使之更有意思，且几乎空白。
6. **泄漏程度 → 可交易性损失的定量映射**。

---

## 5. 2025 → 2026 的六个转向

1. 从"能不能做" → "做了有没有用" → **"凭什么说有用"**。2026 年的核心产出不是新模型，而是"我们之前的数字不可信"的证据。
2. 从经验检测泄漏 → **形式化保证无泄漏**（[`2607.04958`](https://arxiv.org/abs/2607.04958) 证明 look-ahead-freedom ≡ 时间 non-interference，值无关片段上线性时间可判定）。
3. 评测对象从**最终答案**下沉到**证据链与中间推理步骤**。
4. 从单文档 → **多文档、跨期、跨辖区**；"跨文档聚合与对账"取代"单文档定位"。
5. 多智能体叙事从"分工出智慧"退到**"分歧出信息"**。
6. 从"刷准确率" → **成本、时延、可靠性与评测效度工程**。

> 一句话：2023 在证明能不能读财报，2024 在堆架构，2025 发现回测在骗人，2026 开始测量**"骗了多少、怎样才算没被骗"**。当前最稀缺的不是更强的交易 agent，而是**让交易 agent 无法自欺的评测与执行基础设施**。

---

## 6. 完整引用表（63 个 ID，全部经 API 回验）

<details>
<summary>展开</summary>

### 泄漏与效度（核心）
| ID | 标题 | 日期 |
|---|---|---|
| [2608.27734](https://arxiv.org/abs/2608.27734) | What survives honest evaluation? Leakage-safe, search-aware assessment of LLM-discovered strategies | 2026-08-28 |
| [2609.20554](https://arxiv.org/abs/2609.20554) | Does Training on Future Data Pay? Look-Ahead Bias in Forecasting with Pretrained Models | 2026-09-17 |
| [2512.23847](https://arxiv.org/abs/2512.23847) | Detecting Lookahead Bias in LLM Forecasts | 2025-12-30 |
| [2608.17223](https://arxiv.org/abs/2608.17223) | Temporal Leakage in Financial News NLP: A Multi-Architecture Audit | 2026-08-18 |
| [2601.13770](https://arxiv.org/abs/2601.13770) | Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs | 2026-01-20 |
| [2610.21659](https://arxiv.org/abs/2610.21659) | When Does "Post-Call" Begin? An Availability Contract for Earnings-Call NLP Benchmarks | 2026-10-15 |
| [2607.18867](https://arxiv.org/abs/2607.18867) | HindsightBench: A Black-Box Behavioral Audit Protocol for Parametric Hindsight | 2026-07-21 |
| [2607.04958](https://arxiv.org/abs/2607.04958) | Look-Ahead-Freedom as Temporal Non-Interference: A Verifiable Correctness Notion | 2026-07-06 |
| [2309.17322](https://arxiv.org/abs/2309.17322) | Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment | 2023-09-29 |
| [2502.21206](https://arxiv.org/abs/2502.21206) | Chronologically Consistent Large Language Models | 2025-03-01 |
| [2607.11889](https://arxiv.org/abs/2607.11889) | Scaling Point-in-Time Language Models: Enabling Economic Evaluation | 2026-04-25 |

### 情绪与效度
| ID | 标题 | 日期 |
|---|---|---|
| [2609.20198](https://arxiv.org/abs/2609.20198) | Evaluating Financial Sentiment in the Age of AI | 2026-07-29 |
| [2608.04200](https://arxiv.org/abs/2608.04200) | From Financial Sentiment Classification to Return Predictability: A QLoRA Benchmark | 2026-08-05 |
| [2609.11144](https://arxiv.org/abs/2609.11144) | Human Agreement and Return Association Are Not Interchangeable Criteria | 2026-09-10 |
| [2506.04574](https://arxiv.org/abs/2506.04574) | Reasoning or Overthinking: Evaluating LLMs on Financial Sentiment Analysis | 2025-06-05 |
| [2607.28496](https://arxiv.org/abs/2607.28496) | Beyond Sentiment: Structured Information Extraction from Financial News | 2026-07-31 |
| [2410.01987](https://arxiv.org/abs/2410.01987) | Financial Sentiment Analysis on News and Reports Using LLMs | 2024-10-03 |
| [2512.00946](https://arxiv.org/abs/2512.00946) | Fine-tuning of lightweight LLMs for sentiment classification | 2025-11-30 |
| [2506.06335](https://arxiv.org/abs/2506.06335) | FinBERT2: A Specialized Bidirectional Encoder | 2025-05-31 |
| [2602.00086](https://arxiv.org/abs/2602.00086) | Impact of LLMs News Sentiment Analysis on Stock Price Movement Prediction | 2026-01-23 |
| [2603.11408](https://arxiv.org/abs/2603.11408) | Beyond Polarity: Multi-Dimensional LLM Sentiment Signals | 2026-03-12 |
| [2411.00420](https://arxiv.org/abs/2411.00420) | Evaluating Company-specific Biases in Financial Sentiment Analysis | 2024-11-01 |
| [2609.31013](https://arxiv.org/abs/2609.31013) | Same Text, Different Numbers: The Divergence of LLM-Based Measures | 2026-09-25 |
| [2610.22043](https://arxiv.org/abs/2610.22043) | Momentum, Commentary Tone, and Earnings Surprise in LLM Financial Forecasting | 2026-10-15 |
| [2609.04806](https://arxiv.org/abs/2609.04806) | When Financial Fine-tuning Fails: A Three-Level Detectability Analysis | 2026-09-04 |
| [2607.27611](https://arxiv.org/abs/2607.27611) | AWARE-FX: An Auditable Knowledge-Guided AI System | 2026-07-30 |
| [2607.05937](https://arxiv.org/abs/2607.05937) | Is Domain Adaptation Always Helpful? A Frozen-Backbone Study | 2026-07-07 |
| [2607.24072](https://arxiv.org/abs/2607.24072) | LLM-Based vs. Lexicon-Based Sentiment Signals for Tail-Risk Detection | 2026-07-27 |

### Agent 与回测可信度
| ID | 标题 | 日期 |
|---|---|---|
| [2610.04040](https://arxiv.org/abs/2610.04040) | Agent Policy-Value Audit: Separating Transition Composition from Selection | 2026-10-03 |
| [2605.16895](https://arxiv.org/abs/2605.16895) | The Alpha Illusion: Reported Alpha from LLM Trading Agents | 2026-05-16 |
| [2605.28359](https://arxiv.org/abs/2605.28359) | From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading | 2026-05-27 |
| [2608.09988](https://arxiv.org/abs/2608.09988) | OpenPM: Auditable Point-in-Time Evaluation for LLM Portfolio Management | 2026-08-07 |
| [2610.11732](https://arxiv.org/abs/2610.11732) | MemTrial: Learning When to Trust Memory in LLM Portfolio Agents | 2026-10-08 |
| [2505.07078](https://arxiv.org/abs/2505.07078) | Can LLM-based Financial Investing Strategies Outperform the Market? (FINSABER) | 2025-05-12 |
| [2505.11065](https://arxiv.org/abs/2505.11065) | Time Travel is Cheating: Going Live with DeepFund | 2025-05-16 |
| [2510.02209](https://arxiv.org/abs/2510.02209) | StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets? | 2025-10-03 |
| [2609.29701](https://arxiv.org/abs/2609.29701) | Multi-Agent Debate for Explainable Trading | 2026-08-31 |
| [2603.27539](https://arxiv.org/abs/2603.27539) | Toward Reliable Evaluation of LLM-Based Financial Multi-Agent Systems | 2026-03-29 |
| [2311.13743](https://arxiv.org/abs/2311.13743) | FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory | 2023-11-23 |
| [2402.18485](https://arxiv.org/abs/2402.18485) | A Multimodal Foundation Agent for Financial Trading | 2024-02-29 |
| [2412.20138](https://arxiv.org/abs/2412.20138) | TradingAgents: Multi-Agents LLM Financial Trading Framework | 2024-12-28 |

### 长文档与 RAG
| ID | 标题 | 日期 |
|---|---|---|
| [2606.13100](https://arxiv.org/abs/2606.13100) | LEDGER: A Long-Context Benchmark of Corporate Annual Reports | 2026-06-11 |
| [2608.22842](https://arxiv.org/abs/2608.22842) | FinixDoc: Rethinking Financial Document Parsing Beyond Saturated Benchmarks | 2026-08-24 |
| [2607.19238](https://arxiv.org/abs/2607.19238) | FinanceComplexQA: Benchmarking Agentic Reasoning on Financial Documents | 2026-07-22 |
| [2602.17981](https://arxiv.org/abs/2602.17981) | Decomposing Retrieval Failures in RAG for Long-Document Financial QA | 2026-02-20 |
| [2604.03664](https://arxiv.org/abs/2604.03664) | Document-Level Numerical Reasoning across Single and Multiple Tables | 2026-04-04 |
| [2608.12335](https://arxiv.org/abs/2608.12335) | HC-RAG: Evidence-Centric RAG over Heterogeneous Financial Filings | 2026-06-03 |
| [2510.06426](https://arxiv.org/abs/2510.06426) | FinLFQA: Evaluating Attributed Text Generation of LLMs | 2025-10-08 |
| [2512.24903](https://arxiv.org/abs/2512.24903) | FinMMDocR: Benchmarking Financial Multimodal Reasoning | 2025-12-31 |
| [2506.08726](https://arxiv.org/abs/2506.08726) | Improved LLM Agents for Financial Document Question Answering | 2025-06-10 |
| [2402.05131](https://arxiv.org/abs/2402.05131) | Financial Report Chunking for Effective Retrieval Augmented Generation | 2024-02-06 |
| [2401.15050](https://arxiv.org/abs/2401.15050) | LongFin: A Multimodal Document Understanding Model | 2024-01-27 |
| [2311.11944](https://arxiv.org/abs/2311.11944) | FinanceBench: A New Benchmark for Financial Question Answering | 2023-11-21 |

### 幻觉治理与可执行验证
| ID | 标题 | 日期 |
|---|---|---|
| [2604.23588](https://arxiv.org/abs/2604.23588) | FinGround: Detecting and Grounding Financial Hallucinations | 2026-04-26 |
| [2608.10213](https://arxiv.org/abs/2608.10213) | VeriFin: A Neurosymbolic Framework for Verifying LLM-Generated Financial Claims | 2026-08-11 |
| [2610.16587](https://arxiv.org/abs/2610.16587) | Code-Interactive Visual Reasoning for Financial Candlestick Understanding | 2026-10-12 |
| [2609.38523](https://arxiv.org/abs/2609.38523) | MM-FinEval: A Multi-Task Multimodal Benchmark for Financial Forecasting | 2026-09-30 |

### 基准效度批判
| ID | 标题 | 日期 |
|---|---|---|
| [2607.28840](https://arxiv.org/abs/2607.28840) | Benchmarks Are Not Validation: A System-Level View of Financial LLM Applications | 2026-07-31 |
| [2609.23201](https://arxiv.org/abs/2609.23201) | Do Not Trust the Benchmark: Limitations of General LLM Rankings | 2026-09-20 |
| [2610.00969](https://arxiv.org/abs/2610.00969) | A Citation-Grounded Benchmark for Trustworthy Earnings Call Transcripts | 2026-10-01 |

</details>

---

## 7. 未核实清单

- 各论文的**代码与数据开源状态**：除 FinanceBench（[`2311.11944`](https://arxiv.org/abs/2311.11944)，arXiv comment 给出 HuggingFace 链接）外，多数在摘要/comment 中未说明，**未核实**。
- 各论文**全文的消融表、成本模型、回测区间**：本次仅读摘要，**未核实**。
- **`FLARE` 金融基准的原始论文**：该词被天体物理术语严重污染，48 组查询未直接命中，**未核实**。
- 论文间互相点名关系仅取自被引论文**自身摘要原文**（如 Alpha Illusion 点名 FinCon/FinMem/TradingAgents/FinAgent），非独立核实。
- [`2607.11889`](https://arxiv.org/abs/2607.11889) 的 published 为 2026-04-25 而 ID 前缀为 2607，成因**未核实**；本文件一律采用 API 返回的 published 日期。

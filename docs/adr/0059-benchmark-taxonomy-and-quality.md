# 0059：基准体系的命名分类、分层架构、统计判定与质量口径

- 状态：**提案（proposed）**。本文是对 `crates/benchmarks` 的现状审计、业界对标与目标设计；
  落地按 §6 路线分阶段，未落地的能力在验收前一律记 OPEN，不据本文宣称全绿或数值最优。
- 日期：2026-11-26
- 相关：[`engine-benchmark-roadmap.md`](../engine-benchmark-roadmap.md)（P0–P4 阶段与可证伪条件）、
  [`bench-roofline-analysis.md`](../bench-roofline-analysis.md)（roofline 分子/分母口径）、
  [`lld.md`](../lld.md)（E1 TTFT/TPOT、E3 服务吞吐、E6 适配性、E7 带宽、E10 可预测性）、
  [`PROMPT-S++-AUDIT.md`](../PROMPT-S++-AUDIT.md) §8（compute ≥90% / bandwidth ≥85%）、
  [`crates/tools/src/bench.rs`](../../crates/tools/src/bench.rs)、[`card.rs`](../../crates/tools/src/card.rs)、
  [`regress.rs`](../../crates/tools/src/regress.rs)、[`stats.rs`](../../crates/tools/src/stats.rs)；
  ADR：[`0005 注解生成契约不生成注册表`](0005-annotations-as-contracts.md)、
  [`0006 配置机制与身份分离`](0006-config-mechanism-vs-identity.md)、
  [`0017 kernel AST 与 IR 冻结`](0017-kernel-ast-and-ir-freeze.md)、
  [`0018 dtype 单一事实源`](0018-dtype-single-source.md)、
  [`0022 构建期语料解析`](0022-build-time-corpora.md)

## 1. 本文件回答什么

把 `crates/benchmarks` 当作一套**测量装置**来设计：命名唯一可检索、分层单向无环、判定可统计复现、
证据可回查、新增同类需求只加台账行不改内核。本文不追求「跑分第一」，而是把可复现、可判定、可追溯、
可扩展的工程基线做到行业第一梯队；按 `lld` 规则 3.5，实证结论只在同设备、同负载、同口径内成立，
不引用异源论文倍数当本仓证明。

### 1.1 现状实测基线（`cargo` / `find` 可复算）

- 规模：17 个 `.rs`、3791 行；`src/` 9 文件 1905 行，`tests/` 8 文件 1886 行；最大文件 `src/kernel.rs` 599 行，
  均在 AGENTS.md 文件 ≤899 行门禁内。
- 依赖形态：仅 dev 依赖 `criterion / half / serde / serde_json / rand`；`llm-benchmarks` 不被任何生产 crate 引用，
  是独立测量容器（与现有分层纪律一致）。
- 测量覆盖：原语（elementwise/gemv/gemm/matmul/归一化/激活）、注意力（flash/prefill/decode）、
  内存流三档读写、FMA 链扫 + 整机峰值 + 指令通道探针、模型单步与 batch4、train（schedule/snapshot/bufs）、
  IO（容器/加载冷热/mmap）、tokenizer、wire、协议解析与协商，共 27 个 criterion 分组。
- 判定底座：
  - `tools::bench`：`EntryKind{Op,Layer,Model}`、`Geometry`、`Harness`、`Roofline`（峰值带宽/算力 + AI 打分）。
  - `tools::regress`：`Case{baseline, budget_percent, Direction}`，**单点估计**对预算线判 Pass/Regressed/Improved。
  - `tools::card`：`Protocol/Statistic{Mean,Cv}` + 固定种子 bootstrap 2000 次重采样 95% 区间，
    三态 Pass/Fail/Open（Host/Samples/Degenerate/Straddle），附 p50/p95/p99/CV。
  - `tools::stats`：mean、样本 sd(n−1)、秩分位、固定种子重采样区间。
- 归档：`tests/archive.rs` 写 `target/bench-report/<host>/*.json`；登记台账分布在 `tests/table.rs`
  （`SCENARIOS / MEASUREMENT_GROUPS / BASELINES / LOAD_SCENARIOS / DECODE_SCENARIOS`）与 `tests/host.rs`
  （`HOSTS / CEILINGS`），身份唯一性、未登记测量拦截、档案完整性由 `tests/host.rs` 等自检守住。

### 1.2 已具备的长板（保持，不要重写）

1. **上界不拿标称值冒充**：roofline 分子由 `nn::Bounds` 语义表推出，分母是同机三档实测搬运上界，
   未测得即 `None`/OPEN，不填假数（`bench-roofline-analysis.md` §二）。
2. **三态判定 + Open 细分**：样本不够、闭集外主机、统计量退化、区间跨门槛分别记不同 shortfall，
   Open 不降 Must、不静默通过（`card.rs` `Shortfall`）。
3. **跨机不可比显式跳过**：基线/上界/主机档案按身份闭集登记，未建档主机在 CI 跳过判定而非借用他机数字。
4. **后端与身份表驱动**：后端是台账一列而非常量分支，新增后端 = 加一行，判定与归档不动。
5. **固定种子可复现**：重采样种子写死，同一份样本只出一段区间，复核可逐位对拍。

## 2. 业界对标（调研依据）

| 维度 | 行业第一梯队做法 | 来源 |
|---|---|---|
| 基准分层分类学 | 四级构成式层级：L1 原语算子、L2 融合算子、L3 层/块、L4 端到端模型；高层复用低层模块，既在独立模式又在组合模式评测；任务必须对应真实模型里跑的算子，形状/精度/接口对齐生产实现，零合成任务 | [FastKernels](https://fastkernels.github.io/)、[arXiv 2605.23215](https://arxiv.org/html/2605.23215v1)；[BenchCouncil AIBench](https://www.benchcouncil.org/AIBench/)（micro/component/e2e） |
| 服务负载与指标 | MLPerf 四场景：single-stream、multi-stream、server（Poisson 到达）、offline；LLM 拆 TTFT/TPOT/ITL/E2EL，报 p50/p90/p99 长尾，有 accuracy gate，禁止缓存结果或用额外权重。开环（Poisson/到达率独立于响应）暴露排队与尾延迟，是 goodput 与 SLO 测量前提；闭环（固定并发、完成即补）测吞吐天花板，但 back-pressure 会掩盖变慢。计时从调度到达时刻起算（避免 coordinated omission）；warmup 与首尾 ramp-up/drain 瞬态剔除出稳态窗口；goodput = 满足 SLO 的吞吐（总吞吐 × 达成率）。负载须登记到达分布、ISL/OSL 分布、think time、并发/到达率、前缀共享 | [MLPerf Inference](https://github.com/mlcommons/inference)、[MLPerf Endpoints 术语](https://docs.mlcommons.org/endpoints-submission-docs/reference/glossary/)、[IETF LLM 基准术语草案](https://datatracker.ietf.org/doc/html/draft-gaikwad-llm-benchmarking-terminology)、[GPUSmith 开/闭环与 goodput](https://gpusmith.com/articles/en/llm-load-testing-methodology-open-vs-closed-loop)、[vLLM benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) |
| 统计严谨性 | criterion：bootstrap 置信区间、OLS 斜率/R²、中位数/MAD、IQR 离群比例；iai-callgrind：走 Cachegrind 出确定性指令数/L1L2/LLC 缓存计数，绕开墙钟噪声（Linux/Valgrind 边界内）；hypobench：最小效应规模门、多重比较 Bonferroni 校正、bootstrap change CI、Welch t 检验按迭代均值、A/A 重复消除假阳性 | [criterion.rs book](https://bheisler.github.io/criterion.rs/book/)、[iai-callgrind](https://github.com/iai-callgrind/iai-callgrind)、[hypobench](https://github.com/vorner/hypobench) |
| roofline | 算术强度 I=FLOPs/byte，上界 P=min(π, I·β)，ridge/balance point，分层 L1/L2/DRAM 多坡；FLOPs 与流量优先用 PMU 实测而非估算；缓存计数可作确定性佐证 | 本仓 `bench-roofline-analysis.md`（roofline 上界口径） |
| 环境可重复性 | performance governor、锁频、关 turbo/boost、绑核（taskset）、隔离核、关 SMT/NUMA 干扰、drop caches、冷热分离、A/A 重复、长采样；记录调频/温度/功耗/页缓存状态 | [Google Benchmark: perf vs micro](https://github.com/google/benchmark/blob/main/docs/perf-vs-microbenchmark.md) |
| 趋势与回归归档 | 按 commit + machine 存历史结果；CI compare 两版，因子阈值 + 统计上下界；显著回退自动定位到提交区间；历史以 JSON 入库，滚动窗口 + 中位数 + 容差 | [airspeed velocity](https://asv.readthedocs.io/en/stable/commands.html)、[github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark) |
| 计时洁净与防消除 | 输入与输出双向 `black_box` 阻断常量折叠与死代码消除；结果必须被消费（返回/写入屏障）；fixture 构造、克隆、预热在计时区外；被测方法用委托包装防内联并单独扣 overhead；E/S、打印、文件访问排除出计时 | [rustbites black_box](https://rustbites.com/posts/bite-090/)、[Oracle JVM 基准陷阱](https://www.oracle.com/technetwork/articles/java/architect-benchmarking-2266277.html)、[CodSpeed JMH 指南](https://codspeed.io/docs/guides/how-to-benchmark-java-with-jmh) |
| 数值对拍容差 | 参考用高精度（f64，必要时任意精度）；按 dtype 分级 rtol/atol 或 ULP；整数默认精确相等，量化输出可声明有界容差；mean-RE 与 max-RE 双门，max 可放宽以容纳孤立离群；近零/溢出/常态三区域分别判定 | [JAX 数值精度](https://docs.jax.dev/en/latest/accuracy.html)、[PyTorch testing 容差表](https://pytorch.ac.cn/docs/2.5/testing.html)、[TOSA 精度要求](https://www.mlplatform.org/tosa/tosa_spec_1_0_1.html)、[CANN Bench 分 dtype 门槛](https://arxiv.org/html/2607.20518) |
| 可复现元数据 | 每条产物绑定：commit SHA（base/candidate）、二进制/权重摘要、编译器/链接器版本与 flags、工具链版本、CPU 型号/ISA/核数/亲和/governor/频率/温度/功耗、加速卡型号/架构/驱动/运行时、命令行、schema 版本、环境变量、负载种子与输入摘要、原始产物摘要；摘要须能从原始产物复算，不手写数字；版本化 JSON schema + 稳定 workload 标识供趋势 join | [pyperf 元数据字段](https://pyperf.readthedocs.io/en/stable/api.html)、[AIPerf JSON 导出 schema](https://docs.nvidia.com/aiperf/v0.9.0/reference/json-export-schema)、[SPEC CPU2026 运行规则](https://www.spec.org/cpu2026/docs/runrules.html) |
| 形状网格选型 | 形状来自真实模型配置派生（M×K×N / hidden×head_dim×heads×seqlen）；覆盖整除对齐点与非整除边界（tile 量化余量）；兼顾均分计算单元的高利用点与小维度碎片点；每形状登记算法 FLOPs 与字节口径 | [CUTLASS Profiler](https://docs.nvidia.com/cutlass/4.7.0/media/docs/cpp/profiler.html)、[NVIDIA GEMM 背景（tile 量化）](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)、[ROCm MAF 矩阵尺寸建议](https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html) |

## 3. 差距诊断（现状证据 → 对标 → 缺口）

缺口分四象限：**命名、设计、架构、质量**。等级：P0 架构定型（低风险纯重构）、P1 CPU 闭环、
P2 运行时层、P3 后端/量化/硬件观测、P4 服务化。

### 3.1 命名

| # | 现状证据 | 对标缺口 | 等级 |
|---|---|---|---|
| N1 | `EntryKind` 仅 `Op/Layer/Model`（`bench.rs:13`），无融合算子级、无服务负载级 | 分类学缺 L2 融合、L5 serving 两级；层级不可由名字区分 | P0 |
| N2 | 两套近似 `Roofline`：`tools::bench::Roofline`（端口峰值档案 + AI 打分）与 `tests/analysis::Roofline`（read/write/flops/档位/达成率/瓶颈），名字相近职责不同 | 同一术语挂两套概念，阅读与复用易混；应正名为「峰值档案」与「达成分析件」两个视图 | P0 |
| N3 | 场景名靠 `row!` 宏拼 `op/shape`，落盘路径另在 `analysis::on_disk` 做斜杠→下划线镜像 | 身份编码与磁盘形态存在第二套转换规则，台账与文件名可漂移；身份应直接派生安全路径 | P0 |
| N4 | 指标字段散落在归档 JSON（ns、tokens/s、gb_s、gflops、achieved_percent、cv、p50/p95/p99…），无闭集枚举与方向绑定 | 新指标易起近义词（吞吐/速率/gflops 混用）；指标名、单位、方向应进同一闭集表 | P1 |
| N5 | 分组名含 `weights/load`、`cutile/negotiate` 等跨域斜杠段 | 命名空间未按层级分段；`op/shape/backend` 三段式应统一 | P1 |

### 3.2 设计

| # | 现状证据 | 对标缺口 | 等级 |
|---|---|---|---|
| D1 | 算子回归 `regress::Case.judge` 用单点 observed 对预算线，不挂区间（`regress.rs:129`） | 噪声期易假回退/假通过；判定应基于统计量区间 + 最小效应规模，与 `card` 口径统一 | P1 |
| D2 | 多数算子性能基准未绑数值对拍；仅 `gemv_i4` 有标量 vs 向量误差对照 | 缺 accuracy gate：正确性不过不应出性能数（MLPerf / FastKernels 同口径） | P1 |
| D3 | `model.rs` 只测单步/batch4 的 ns 与 tokens/s；无 TTFT、TPOT、ITL、E2EL、长尾分位、稳态窗口、goodput | 服务侧指标与负载场景缺失，无法支撑 E1/E3 验收 | P4 |
| D4 | 精度覆盖 f32/i2/i4/i8，无 f16/bf16/f8(e4m3/e5m2)/nf4 | 缺分 dtype 的精度-性能曲线与容差对照 | P3 |
| D5 | 热路径分配/复制/缓冲复用未在基准内计数；内存峰值、tokens/J 未测量 | 性能结论缺分配与能效证据，违反 AGENTS.md §6.1 缓冲复用可观测要求 | P2 |
| D6 | 多个迭代闭包内在计时区克隆输入（`kernel.rs:70-73/96-99/173/187/206/221`、`attention.rs:63/166`），fixture 构造与被测成本混在一起 | 输入克隆/构造应在区外完成，迭代内只传视图；否则时延含非被测复制成本 | P1 |
| D7 | 形状偏窄且手填：decode 行宽固定 4096，缺整除对齐点与非整除边界，未从真实模型配置派生形状网格 | 形状应由模型配置 + 边界余数点组成网格，复用注解元数据，不靠逐组手写 | P3 |
| D8 | 各基准闭包各自手写 `black_box` 包装、克隆、对拍、归档样板，骨架重复 | 重复骨架应收敛到公共采集器；用例只声明「算子/形状/后端/容差/指标」，样板由配置表展开 | P0 |
| D9 | 基准元数据（shape/dtype/skeleton）与 `nn` 侧 `#[model]` 注解各写一份 | 应以注解编译期常量为单一事实源派生台账，消除第二份可漂移的形状/精度登记 | P0 |

### 3.3 架构

| # | 现状证据 | 对标缺口 | 等级 |
|---|---|---|---|
| A1 | 判定类型在 `tools`，采集在 `benchmarks/src`，台账/分析/归档在 `benchmarks/tests`；依赖方向基本正确，但峰值档案与达成分析件分散在两处 | 五层边界需显式固化：定义 / 统计 / 台账 / 采集 / 归档，统计与判定层保持零时钟、零 IO、可单测 | P0 |
| A2 | `KernelCase`（形状/dtype/后端）与服务负载（ISL/OSL/批/并发/到达分布/窗口）未分离抽象 | 新增 serving 场景会侵入算子基准；应把负载生成器独立于被测引擎 | P4 |
| A3 | `Harness` 仅 host/cpu/os/arch/toolchain/driver（`bench.rs:36`） | 环境档案缺绑核、governor/boost、SMT/NUMA、页缓存冷热、温度/功耗、采样数/预热；未测得留 `None` | P1 |
| A4 | 归档 JSON 落本机 `target/bench-report/`，无跨 commit/跨运行历史库与趋势对比 | 缺可回查的时间序列与 CI 回归对比层 | P2 |
| A5 | roofline 只分 bandwidth / not-bandwidth，算力 vs 延迟瓶颈不分；无 PMU 佐证 | 缺分层 roofline（L1/L2/DRAM 多坡）与计数器证据；延迟瓶颈需 prof 采样，当前不具备 | P3 |
| A6 | 归档只带 host/cpu/os/arch/toolchain/driver；缺 commit SHA、二进制/权重摘要、编译 flags、亲和/governor/频率/温度、种子与输入摘要、原始产物摘要 | 结果无法从产物独立复算，也无法 join 历史趋势；溯源字段需进版本化 schema | P1 |

### 3.4 质量

| # | 现状证据 | 对标缺口 | 等级 |
|---|---|---|---|
| Q1 | 算子基准从 criterion `estimates.json` 字符串 `find("mean")` 取 `point_estimate`（`analysis.rs`） | 未利用 criterion 自带 slope CI、std、R²、MAD、离群比例；字符串解析脆弱，应直接消费结构化产物 | P1 |
| Q2 | 无 A/A 自检（同一基线跑两次不应显著不同）、无多重比较校正、无最小效应规模门 | 几十条基准同时判定存在族系假阳性；1–2% 预测漂移不应报警 | P1 |
| Q3 | 样本量部分固定 15；预热/迭代自适应结果未登记入档 | 样本下限、预热轮次、迭代次数应进环境档案，可回查 | P1 |
| Q4 | 趋势判定只与单份基线比，无滚动中位数/历史区间对照 | 缺基线漂移检测与多轮独立重复口径（对齐 E3：每配置 ≥3 次独立稳态重复） | P2 |
| Q5 | 归档字段键序固定、缺测写 null，诚实性已立；但未披露 ignored/skipped/外部阻塞的结构化汇总 | 需把跳过原因、工具版本、命令、产物路径结构化进报告，满足 AGENTS.md §7.3 | P2 |

## 4. 目标架构（决策）

### 4.1 五层单向依赖（定义 ← 统计 ← 台账 ← 采集 ← 归档；下层不反向依赖上层）

```
tools::bench   定义层：EntryKind / Geometry / Harness / Metric / 协议闭集 / 峰值档案
   ▲ 零被测依赖、零时钟、零 IO，纯类型与校验
tools::stats · card · regress   统计判定层：区间、三态、方向、效应规模、校正
   ▲ 确定性、可单测；不读时钟、不读文件
benchmarks 台账层：KernelCase 表、Workload 表、形状/精度/后端/负载登记（数据定义，无副作用）
benchmarks 采集层：criterion 闭包、fixture、环境档案采集、负载生成、分配计数
benchmarks 归档层：JSON 记录、历史库、对比、报告渲染（只消费前四层产物）
```

- **复用路径**：分子流量仍由 `nn::Bounds` 语义表推出；峰值档案按指令通道成对登记；判定内核不出现具体业务算子名。
- **扩展点**：新增算子/后端/形状/精度 = 台账加一行；新增服务负载 = 加一个 `Workload` 表项；判定、归档、统计不动。
- **纪律约束**：`src/` 平铺不新增子目录；新增/修改文件 ≤899 行、函数 ≤30 行、圈复杂度 ≤5；
  核心 `unsafe_code = "deny"`；criterion 等仅 dev 依赖，生产图零新增非必要依赖。

### 4.2 命名闭集（D1/N1–N5）

- **层级枚举** `EntryKind` 扩为五级：`Op`（L1 原语）、`Fused`（L2 融合）、`Layer`（L3 层/块）、
  `Model`（L4 端到端）、`Serving`（L5 服务负载）。
- **身份三元组** `(surface, backend, geometry)` 唯一确定一条基准；场景名由三元组派生，
  路径由同一派生函数产出，取消 `on_disk` 第二套镜像规则。
- **指标闭集** `Metric`：`ttft_ns / tpot_ns / itl_ns / e2e_ns / tokens_s / req_s / goodput_tokens_s /
  gb_s / gflops / achieved_percent / cv`；每个指标绑定单位与 `Direction`（Lower/Higher），禁止近义词新增。
- **术语正名**：`bench::Roofline` 更名 `PeakProfile`（端口峰值档案）；`analysis::Roofline` 更名
  `Attainment`（实测达成分析件）；二者通过 AI/上界/达成率组装，不再共用 `Roofline` 一名。

### 4.3 场景与负载分离（A2/D3）

- `KernelCase`：算子、层级、形状、dtype、后端、工作集档位、对拍容差。
- `Workload`：ISL/OSL 长度分布、批大小网格、并发度、到达过程（开环 Poisson / 闭环固定并发）、
  稳态窗口定义、SLO/超时、预热与重复次数。开环暴露排队与尾延迟，用于 SLO/goodput 判定；闭环测吞吐
  天花板。计时从调度到达时刻起算（避免 coordinated omission）；warmup 与首尾 ramp-up/drain 瞬态剔除出
  稳态窗口。负载生成器不依赖被测引擎内部实现，只经公共接口发请求。

### 4.4 统计判定（D1/Q1/Q2）

- 算子回归从单点判定改为**区间判定**：统计量 bootstrap 区间对预算线，整段内侧 Pass、整段外侧 Fail、
  跨门槛 Open（复用 `card` 的三态骨架）。
- 引入**最小效应规模门**（相对基线的不可察觉漂移带）与**多重比较校正**；A/A 自检作为一道前置基准，
  两次同基线对照必须落在无显著差异区间，否则整套结果记 Open。
- 直接消费 criterion 结构化产物（slope CI、std、R²、MAD、离群比例），不再字符串解析 estimates.json。
- Linux 环境下可选挂 iai-callgrind 出指令/缓存确定性计数，作为墙钟噪声外的第二佐证；边界（非 Linux、
  需要真实计时）显式记 Open，不伪装。

### 4.5 环境档案与证据（A3/Q3/Q5）

- `Harness` 扩 `EnvProfile`：绑核/亲和、governor/boost、SMT/NUMA、页缓存冷热、温度/功耗、
  工具链/驱动版本、样本数、预热轮次、迭代次数；任一项采集不到留 `None`，不填标称值。
- **溯源字段（每条产物必填，缺测写 `null`）**：base/candidate commit SHA、二进制与权重摘要、
  编译器/链接器版本与 flags、CPU 型号/ISA/核数/频率、加速卡型号/架构/驱动/运行时、
  命令行与 schema 版本、相关环境变量、负载种子与输入摘要、原始时长产物摘要、稳定 workload 标识。
- 摘要必须能从原始产物复算，不手写数字；Open/Skip/Ignored/阻塞原因结构化汇总，附命令、工具版本与产物路径。

### 4.6 趋势归档（A4/Q4）

- 历史结果按 `(commit, machine, profile)` 入本地/CI 历史库；CI compare 两版，输出统计上下界与因子阈值，
  显著回退定位到提交区间；基线用滚动中位数 + 区间对照，而非单份快照。
- 稳态吞吐按 E3 口径：每配置 ≥3 次独立重复，报均值/中位数与 95% CI，附错误率/超时率/队列时间。

### 4.7 计时洁净与防优化消除（D6）

- 输入与输出双向加屏障：输入 `black_box` 阻断常量折叠，输出被公共采集器消费（返回/写入屏障）阻断死代码消除。
- fixture（权重、输入、输出缓冲）在计时区外构造一次；迭代闭包内只传视图（`&[T]`、`View`），不在闭包内 `clone` 输入。
- 预热迭代不计入统计；被测方法经委托包装防内联，包装 overhead 单独跑一轮扣减。
- 被测闭包内禁止 E/S、打印、文件访问；这一类辅助操作移到区外或用内存替身。
- 屏障施加、缓冲复用、overhead 扣减由公共采集器统一完成，用例不各自手写。

### 4.8 形状网格与元数据单一事实源（D7/D9）

- 算子的 shape/dtype/skeleton 取自 `nn` 侧 `#[model]` 注解展开的编译期规格常量（dtype 同时遵循 0018 单一事实源）；基准台账只登记差异项（后端、容差、负载、基线预算），不复制第二份形状与精度。按 0005，注解只出契约常量、不生成注册表；基准用例注册表独立维护，不由注解扫描产出。
- 形状网格三类点：真实模型配置派生点、整除对齐点（tile 满块）、非整除边界与小维度碎片点（tile 量化余量）；每形状的算法 FLOPs 与字节仍由 `nn::Bounds` 语义表推出。
- 新增形状 = 网格表加一项；采集、判定、归档内核不改。

### 4.9 内存与分配计数（D5）

- 基准 binary 挂 dev-only 全局计数分配器，记录每迭代 `alloc_count`、`alloc_bytes` 与峰值驻留；计数与耗时并列归档。
- 堆测量串行或放独立进程，避免并行测试的分配混入；全局分配器只在基准 binary 声明。
- 「零分配 / 缓冲复用」结论必须有分配计数或容量、指针、分配计数证据支撑；无证据只陈述结构变化，不宣称性能收益。

### 4.10 复用与配置化优先（D8，本轮核心约束）

- **优先级**：配置表 / 枚举 / 注解 ＞ 泛型 / 闭包 / 高阶组件 ＞ 声明式宏 ＞ 复制代码；禁止平行实现、仅转发的包装层和伪泛型。
- **基准用例即配置项**：一条项声明层级、算子引用、形状网格、dtype、后端、参考实现、容差、指标、方向、基线预算；公共采集器按项展开计时闭包、屏障、对拍、归档与 roofline 派生。
- **共用能力各实现一次**：防消除屏障、对拍容差、区间判定、环境档案采集、归档字段、roofline 上界派生，均为单一公共组件，被配置项引用；同一逻辑出现两次即提取，不复制分支骨架。
- **扩展路径**：新增同类基准只加表项或注解；只有证明配置/泛型无法承载时才动公共内核；新增文件必须证明独立职责，碎片文件折回既有模块。
- **复用既有设施**：沿用 `annotation` crate 的 `#[model]` 规格常量与 `prof_ops!` 表驱动模式，不另起第二套注册或配置源；配置语料在构建期展开为静态表（0022），运行期不挂 TOML 解析器。

### 4.11 硬件观测与分层 roofline（A5）

- roofline 按 L1 / L2 / DRAM 分层成多坡曲线，算术强度 I=FLOPs/byte，上界取 `min(峰值算力, I·峰值带宽)`；当前只分 bandwidth / not-bandwidth，算力与延迟瓶颈暂不细分。
- FLOPs、缓存/内存流量优先用 PMU 或缓存计数器实测，不用估算冒充；Linux 下 iai-callgrind 的指令/L1L2/LLC 计数作确定性第二佐证。
- 无 PMU 权限、计数器不可用或延迟瓶颈缺 prof 采样证据时，对应格记 Open，只报 `not-bandwidth`，不猜算力或延迟归属。

## 5. 正确性与性能绑定（D2）

- 每条 `Op/Fused/Layer` 性能基准挂一个数值对拍：参考实现为高精度锚点（f32/f16/bf16 对拍用 f64，
  f64 对拍用任意精度）；正确性未过门，该性能格记 Open，不出性能数。
- **容差按 dtype 分级登记**：浮点用 rtol/atol 或 ULP（scale-aware），整数默认精确相等，量化输出可声明
  有界整数容差；分档参考行业容差表（f16/bf16 宽于 f32，f32 宽于 f64），具体数值在台账行冻结，不写进算子实现。
- **双门判定**：mean-RE 控整体精度，max-RE 控孤立离群；max 门可按约定倍数放宽，避免单点离群拖垮整张量。
  近零（参考值趋零，相对误差失真）、溢出/∞、常态三区域分别选误差口径，不在近零点套用相对误差。
- 性能与正确性用同一 `KernelCase` 几何，避免两套形状漂移；对拍样本、种子与输入摘要随归档留存。

## 6. 落地路线（对齐 roadmap P0–P4，每阶段可证伪验收）

| 阶段 | 改动面 | 验收口径（可回查） |
|---|---|---|
| P0 架构定型 | `EntryKind` 补五级；两套 Roofline 正名；身份/路径单一派生；五层边界与依赖单向化自检；基准用例配置化/注解注册，复用 `#[model]` 元数据；公共采集器收敛 `black_box`/克隆/对拍/归档样板 | 依赖方向门通过；命名门零新违规；台账身份唯一自检通过；无第二套路径转换与第二份形状登记；重复骨架只存在于公共采集器一处 |
| P1 CPU 闭环 | 算子回归改区间判定 + 效应规模 + 校正；A/A 前置基准；消费 criterion 结构化产物；EnvProfile 与溯源字段扩位；正确性对拍绑定；计时洁净（fixture 出区、闭包内不克隆） | 区间判定单测覆盖 Pass/Fail/Open 三态；A/A 基准入档；每条性能格带容差对拍；归档字段完整可解析且可从原始产物复算；闭包内无输入克隆 |
| P2 运行时层 | Layer/Model 基准；KV 长度/批网格；TPOT/ITL/P50/P95/P99；稳态窗口；内存峰值/分配计数；历史趋势库 | 每配置 ≥3 次独立重复入档；分配计数有计数证据；趋势 compare 产出上下界与回退标记 |
| P3 后端/量化/观测 | 多后端行；f16/bf16/f8/nf4 精度曲线；形状网格（对齐点 + 非整除边界）；分层 roofline；PMU/缓存计数佐证；能效 tokens/J | 同口径跨后端对照；精度-性能曲线可复算；形状覆盖对齐与边界点；瓶颈侧有计数器证据，不靠猜测 |
| P4 服务化 | Serving 场景（single-stream / server Poisson / offline）；TTFT/TPOT/goodput；并发-Pareto；同口径对照 llama.cpp/vLLM | E1/E3 口径：分相报数、长尾分位、goodput、SLO 满足率；跨实现对照须同设备同负载，异源数字只作情报 |

### 6.1 P0 落地设计拆解（架构定型，不改测量口径）

**范围（只做结构与命名定型，不碰计时与判定口径）**

1. 分类学：`tools::bench::EntryKind` 由 Op/Layer/Model 扩为 Op/Fused/Layer/Model/Serving 五级闭集；`as_str` 短名随闭集登记。Serving 在本阶段只占类型位，不实现服务化基准（归 P4）。
2. 正名：`tools::bench::Roofline`（硬件峰值上界）→ `PeakProfile`；`tests/analysis.rs::Roofline`（报告侧达成率口径）→ `Attainment`。Rust 符号重命名，归档 JSON 键名（`roofline_ns` / `bottleneck_side` 等）保持不变以兼容历史快照。
3. 身份单一派生：台账 `scenario` 继续由 `row!` 宏按 `group/bench` 编译期拼接；新增台账自检：身份三元组 `(scenario, backend, op)` 唯一、分组/用例可定位；磁盘路径清洗只走 `analysis::on_disk` 一处。
4. 配置化优先：用例即台账表行；不造注解注册表（遵 0005：注解只出契约常量）；dtype 口径归 0018 单一事实源；roofline 分子继续由 `nn::Bounds` 推出，不手写第二份 read/write/flops 公式。
5. 采集族承载点：既有 `dual` / `gemm_pair` / `spread` / `f32_paths` / `i8_paths` 已是对应采集族的公共承载点。P0 不新增 `bench_function` 转发包装——各闭包借用 / 拥有 / 返回 `Result` 形态各异，抽取只会变成一层转发，违反复用铁律；闭包内 clone、输入输出屏障与缓冲复用的统一收敛随 P1 计时洁净一并处理。线程屏障 / `iter_custom` 保持独立族，不强行套同一闭包。

**领域解耦**：不变核心是身份三元组 `(surface, backend, geometry)`、几何校验与峰值上界结构（tools，不依赖 criterion、不读环境）；可变差异是分组名、形状、后端、负载档位、基线预算，全部外置到台账表行，不写进被测内核。

**正交分层（单向）**：数据定义（tools::bench 结构与校验）→ 台账配置（tests/table 表与派生宏）→ 采集装配（src 各分组与既有采集族承载点）→ 记录归档（record/archive）；入口 `main.rs` 只声明 criterion 分组。采集器依赖被测对象，不反向。crate 内模块单向依赖由 `pub(crate)` 可见性约束；工作空间层表与边集归既有 testkit 架构门，P0 不新增平行源码扫描器。

**复用路径**：复用既有 `tools::bench` 结构与 `row!` 宏；采集族只承载装配，不复制判定/统计；线程与 custom 采集保留各自族。新增同类基准只加台账行 + 登记一个分组；新增后端只加一行同形状不同 `backend`，判定与归档不动。

**验证计划**：tools 静态门（CCN/NLOC/文件行数）、name-lint 命名门零新违规、testkit 架构门层表/边集不破坏；新增五级 `as_str` 闭集断言与台账身份唯一自检；正名后既有 card/record/archive/cases 测试全绿；jscpd 重复率不上升。

**P0 明确不做**：闭包内夹具出区等计时洁净（P1）、区间判定与 A/A 前置基准、正确性对拍绑定（P1）、形状网格与精度曲线（P3）、分配计数与分层 roofline（P2/P3）、Serving 场景实现（P4）。

### 6.2 P1 落地设计拆解（CPU 闭环判定与洁净）

**批次划分**（每批可独立回查，不把半成品混进同一门禁）：

1. **判定内核（`tools::regress`）**：观测由单点 `f64` 升级为带置信区间的 `Estimate { point, lower, upper }`；区间对预算线落成三态——整段内侧 Pass、整段外侧 Regressed、跨门槛 Open(Straddle)；效应规模归一为「改进为正」并随结论携带；最小效应规模门（不可察觉漂移带 `epsilon_percent`）内的漂移记持平而非 Improved；旧单点入口保留，内部退化为零宽区间。
2. **结构化产物（`tests/analysis.rs`）**：用仓库唯一 JSON 解析器 `weights::json` 读取 criterion `estimates.json` 的 `mean.point_estimate` 与 `confidence_interval` 两端，替换手写字符串扫描；解析不出即明说，不猜数。
3. **A/A 自检内核（`tools::regress::self_check`）**：两份同实现 `Estimate` 的对称等价性检验，无方向概念。点估计相对差（分母取较大值，避免近零失真）落在不可察觉带内且两区间重叠 → `Agree`；超带 → `Disagree { ratio }`；区间不重叠 → `Open(Straddle)`；非有限或非正 → `Open(Degenerate)`。成对采集探针、台账登记与归档对照随批次五采集器统一接入，本批只落纯判定内核与单测。
4. **本批不做（P1 后续批次）**：`EnvProfile` 与溯源字段扩位；正确性对拍绑定骨架（容差分级、双门）；计时洁净（fixture 出区、闭包内不克隆、屏障与 overhead 扣减统一到采集器）；A/A 成对探针与归档对照。

**领域解耦**：不变核心是方向折算、基线/预算线、效应规模口径；可变差异是观测为点或区间、不可察觉带宽度，外置到 `Case` 配置。判定区间来自 criterion mean CI，`regress` 不做第二套 bootstrap 重采样（重采样区间仍归 `card::Tally`）。

**正交分层（单向）**：`tools::regress` 纯判定——不读文件、不依赖 criterion、不碰环境；`analysis` 负责产物解析与区间装配；`archive` 只消费结论并归档。依赖方向 tools → 无外部测量源，analysis → tools，archive → analysis/tools。

**复用路径**：三态「整段内 Pass / 整段外 Fail / 跨线 Straddle」的方向分支与 `card::verdict` 同源；但 card 判绝对门槛、regress 判相对基线的预算线，两条规则各自一处，不合并成一个泛型门槛组件。`stats::interval` 只服务 card 的样本重采样，regress 不调用。

**扩展方式**：新增未判定理由（如样本不足 Noise）只加 `OpenReason` 表项与短名；新增指标方向不改内核。多重比较校正在 `Baseline` 批量判定层（族系错误率的阈值缩放），单 `Case` 不挂校正状态；本批先留批量层接入位，不内建校正算法。

**验证计划**：tools 集成测试覆盖 Higher/Lower 各 Pass / Regressed / Open、改进 / 持平 / 越不可察觉带、效应符号与量级；analysis 用真实 `estimates.json` 片段断言 point/lower/upper；既有 regress 14 用例、card 测试与台账自检不退化；jscpd 重复率不上升。

**P1 边界**：iai-callgrind / PMU 计数在当前装置不具备条件，记 Open，不伪装；A/A、对拍绑定、溯源扩位、计时洁净未完成前，对应归档格保持 OPEN。

### 6.3 P2 运行时层（测量内核先行，运行时接入位留 OPEN）

**本轮落地的装置无关纯内核**：

1. **历史趋势对照（`tools::trend`）**：`HistorySample` 只持一次实测观测（commit / host / 产物摘要等身份由归档层补充）；`median_baseline` 取历史滚动中位数（复用 `stats::quantile(0.5)`）；`compare` 用基线构造 `regress::Case`，三态判定与效应规模仍走 `regress` 口径，不平行造第二套门槛。方向取 `Lower`（ns 越小越好，回退 = 观测越过基线预算线）；历史为空、基线非法或当前值非有限时返回 `None`，不拿单份历史冒充滚动基线。
2. **分配计数值对象（`tools::alloc::AllocProfile`）**：`count` / `bytes` / `peak_bytes` 三计数口径；只持数值与访问器，不依赖分配器。

**复用路径**：分位 p50 / p95 / p99 已由 `card::Tally` 覆盖（`stats::quantile`），本轮不重复造分位内核；趋势三态复用 `regress`；中位数复用 `stats`。

**OPEN（依赖 engine / 采集器，本轮不切入）**：每配置 ≥3 次独立稳态重复入档；稳态窗口剔除首尾 ramp-up / drain；KV 长度与批网格；逐 token 时序采集；dev-only 全局计数分配器挂基准 binary、串行测量与驻留高水位采样；历史 JSON 库按 `(commit, machine, profile)` join；离散统计量（std_dev / MAD / slope）与 commit SHA 注入通道。

### 6.4 P3 后端 / 量化 / 观测（形状网格纯函数先行）

**本轮落地的纯函数**：`tools::grid::aligned_points`（tile 整除对齐点 `tile·q`）与 `boundary_points`（满块减一碎片边界点 `tile·q − 1`，覆盖 tile 量化余量的最后一块不满路径）。每形状的算法 FLOPs 与字节仍由 `nn::Bounds` 语义表推出，这里不写第二份 read / write / flops 公式；真实模型配置派生点由台账行直接给出，网格函数只生成两类机械点。

**OPEN**：多后端行对照；f16 / bf16 / f8(e4m3/e5m2) / nf4 精度-性能曲线；分层 roofline L1 / L2 / DRAM 三坡（分子 `Bounds`，分母同机三档实测搬运上界，未测得留 `None`）；PMU / 缓存计数佐证；能效 tokens/J。iai-callgrind 与 PMU 仅在 Linux / Valgrind 边界内可用，macOS 记 Open 不伪装。

### 6.5 P4 服务化（场景与指标闭集预留）

**本轮落地的纯内核（`tools::serve`）**：`ServingScene` 对齐 MLPerf 四档（single-stream / multi-stream / server / offline）；`Arrival` 分开环 Poisson 与闭环固定并发；`PhaseLatency` 持 TTFT / TPOT / ITL / E2E；`phase_latency` 由单调不减的 token 产出时刻拆分相（空 / 非有限 / 时序回退返回 `None`，单 token 时 ITL 为空、TPOT 为 0）；`slo_hit_rate` 出 SLO 达成率；`tokens_per_s` 归一吞吐；`goodput_tokens_per_s` = 归一吞吐 × 达成率。

**边界**：本层不读时钟、不起服务、不做调度，逐 token 时刻由采集层喂入；长尾分位复用 `card::Tally`，不另造列统计。

**OPEN**：Poisson 到达与并发 Pareto 负载生成器；ISL / OSL 分布、think time、前缀共享登记；真实 HTTP / gRPC 端点；跨实现对照 llama.cpp / vLLM 须同设备同负载，异源数字只作情报。

## 7. 边界与不做

- 不自行定义张量 IR/AST 或融合编译器；沿用 0017 冻结的 kernel AST 与运行时张量 IR 边界，基准只经公共接口调用被测对象。
- 不用理论峰值、他机数字或参考实现冒充实测；达成率不等于硬件利用率。
- 不跨机比绝对值；未建档主机/无设备行记 Open，不删行、不降阈值。
- 不引入生产非必要依赖；确定性计数工具按平台边界挂接，不可用记 Open。
- 不把 kernel-only 数字当服务吞吐；prefill 与 decode、输入与输出 token 不混加（E3 口径）。

## 8. 未完成项与证据边界

- P0 已落地：五级 `EntryKind`、`PeakProfile` / `Attainment` 正名、台账身份唯一与字段完整自检；§4 其余决策（区间判定、对拍绑定、形状网格、分配计数、分层 roofline、Serving 场景等）未落地，验收前对应格保持 OPEN。
- P0 取舍：未造注解注册表（遵 0005，注解只出契约常量）；未抽 `bench_function` 转发包装（闭包形态各异，抽取即转发层）；台账 dtype/形状尚未回读 `#[model]` 常量，留 P3；闭包内 clone 与输入输出屏障的统一收敛留 P1 计时洁净。
- **P1 本批已落地（判定内核 + 产物解析 + A/A 自检内核）**：`tools::regress` 新增 `Estimate`（点估计 + 置信区间）与 `OpenReason::{Straddle, Degenerate}`；区间对预算线三态——整段内侧 Pass、整段外侧 Regressed、跨门槛 Open；效应规模归一为改进为正并随结论携带；`epsilon_percent` 不可察觉带内漂移记持平不报改进；单点入口保留为零宽区间。A/A 自检 `self_check` / `SelfCheck`：两份同实现估计件的对称等价性（Agree / Disagree{ratio} / Open），区间重叠且相对差在带内才记 Agree。`tests/analysis.rs` 改用 `weights::json::parse_ledger` 解析 criterion `mean.point_estimate` 与 `confidence_interval` 两端，归档判定链路（`judge_one` / `judge_ceiling`）接入区间；新公共类型已登记进 `docs/ARCHITECTURE.md` 所有权表。
- **P1 剩余批次（未落地，对应格保持 OPEN）**：
  - **#35 EnvProfile / 溯源扩位**：现有 `Harness` 六字段（host/cpu/os/arch/toolchain/driver）已随记录进归档，`arch` 取 `std::env::consts::ARCH`。§4.5 溯源闭集已定：绑核 / 亲和、governor / boost、SMT / NUMA、页缓存冷热、温度 / 功耗、核数 / 频率 / ISA、加速卡型号 / 驱动 / 运行时、样本数 / 预热轮次 / 迭代次数、base/candidate commit SHA、二进制与权重摘要、编译器 / 链接器版本与 flags、命令行与 schema 版本、相关环境变量、负载种子与输入摘要、原始时长产物摘要、workload 标识。本轮不新增无归档消费方的 `Option` 字段（与批次一移除 `stderr` 同一纪律）；上述字段在采集器 / 归档链路打通前只留接入位，本机 macOS 采不到的项一律 `None`，不填标称值。estimates.json 离散统计量（`std_dev` / `median_abs_dev` / `slope`）与 commit SHA 注入通道随 P2 趋势库（`(commit, machine, profile)` 历史归档）一起落地；当前无构建期 SHA 注入（无 vergen / build.rs 环境注入），不引非必要依赖。
  - **#36 对拍绑定**：复用既有 `testkit::compare`（`close` / `assert_close` / `assert_vec_near` / `max_rel_err` / `Scale` / 非有限配对）与 `cutile::reference::tolerance` 的 dtype 分级容差表（F16/BF16 1e-3、F32 1e-5、整数与布尔精确），**不新造平行容差组件**。已落地：在既有 `tally` 单次遍历内累加 `mean-RE`（`mean_rel_err` / `mean_rel_err_with`，与 max-RE 同分母策略，长度不符与非有限不一致归无穷，空批为零，全零参考退绝对误差）；本轮补双门读数与组合判定——`RelGate { mean, max }` 由一次 `rel_gate` 遍历同时出 mean-RE / max-RE，`passes(mean_tol, max_tol)` 判两门各自阈值，`assert_rel_gate` 双门断言并把超门读数写进失败消息；max 门阈值由调用方显式放宽，不内置臆造倍数。唯一既有数值对拍点（`gemv_i4` 标量叶 vs 向量通道）已从手写 `slack = 1e-5·max(1,|want|)` 循环迁到公共 `assert_vec_near`（Unit 地板口径），消除第二份容差公式。剩余缺口：台账登记每格 dtype 双门阈值（mean 门 / max 放宽倍数）、标量参考 vs tile 执行路径的预检对照缓冲、未过门记 Open 不出性能数的绑定骨架——当前 `cpu-reference` 装置被测即标量参考，无第二份高精度锚点，标量/tiled 对照与双门阈值随 P3 精度曲线（f16/bf16/f8/nf4 精度-性能网格）一并定型。
  - **#37 计时洁净与 A/A 采集**：已逐闭包核查——fixture（权重 / 输入 / 输出缓冲）在装配段构造一次，迭代闭包内只传 `&[T]` / `View` 复用；输入 `View` 的 `clone` 只是视图描述符（引用 + 小 strides）复制，**不是元素深拷贝**；输出缓冲跨迭代复用，闭包内无 `to_vec` / `collect` 堆分配。防消除返回值由 criterion 0.8.2 `Bencher::iter` 内部 `black_box(routine())` 统一兜底（`bencher.rs` 单次调用），闭包返回 `Result` 不必各自再包一层；既有闭包外层 `black_box` 属冗余屏障，不构成口径缺口。线程 / `iter_custom` 峰值探针保留独立族，不套同一闭包。剩余接入位：委托包装防内联与 overhead 扣减统一到公共采集器（随 P2 采集器收敛）；A/A 成对探针需要第二份独立重复测量与台账登记，当前装置只有一套实现，`self_check` 判定内核已备，探针与归档对照留接入位。
  - 多重比较校正落在批量判定层，单 `Case` 不挂。
- **P2–P4 本轮已落地（装置无关纯内核，测量层预留）**：
  - **#31 P2**：`tools::trend` 历史趋势对照——`HistorySample`（单条观测）、`median_baseline`（滚动中位数，复用 `stats::quantile`）、`compare`（复用 `regress::Case` 三态，不另造门槛）；`tools::alloc::AllocProfile`（分配次数 / 累计字节 / 驻留峰值值对象）。分位 p50/p95/p99 复用 `card::Tally`，未重复造内核。
  - **#32 P3**：`tools::grid` 形状网格纯函数——`aligned_points`（tile 整除对齐点）与 `boundary_points`（满块减一碎片边界点）；每形状 FLOPs / 字节仍由 `nn::Bounds` 推出。
  - **#33 P4**：`tools::serve` 服务指标闭集——`ServingScene` 四档、`Arrival` 开 / 闭环、`PhaseLatency`（TTFT/TPOT/ITL/E2E）、`phase_latency`、`slo_hit_rate`、`tokens_per_s`、`goodput_tokens_per_s`（归一吞吐 × SLO 达成率）。不读时钟、不起服务、不做调度。
  - 新公共类型已登记进 `docs/ARCHITECTURE.md` 所有权表；集成测试挂 `tools --test runtime`。
- **P2–P4 剩余 OPEN（运行时 / 后端 / 负载受装置与原型阶段约束，对应格保持 OPEN）**：≥3 次独立稳态重复与稳态窗口、KV / 批网格、逐 token 时序采集；全局计数分配器挂基准 binary 与串行高水位采样；历史 JSON 库 `(commit, machine, profile)` join、离散统计量与 commit SHA 注入；多后端行、f16/bf16/f8/nf4 精度曲线、分层 roofline 三坡、PMU / 缓存计数、能效 tokens/J；Poisson / Pareto 负载生成器、真实端点、跨实现对照（同设备同负载，异源数字只作情报）。PMU / iai-callgrind 仅 Linux / Valgrind 边界内可用，macOS 记 Open 不伪装。
- 门禁证据：本批 `cargo fmt --all --check` 通过；`llm-tools --test card` 49 passed（含 6 个区间判定用例与 4 个 A/A 自检用例）；`llm-benchmarks --test regress` 16 passed（含 2 个产物解析用例）；`llm-testkit --lib` 127 passed（含 5 条 mean-RE 用例与 5 条双门判定用例；架构 / 所有权门）；jscpd 0 clones / 0.00%；命名门 0 命中；`clippy -p llm-tools -p llm-benchmarks -p llm-testkit` 无 warning。代码核查：benchmarks 全部基准入口在 `main.rs` 挂载，无未挂载死入口；fixtures / io / attention 辅助函数均有调用方；testkit `compare` 公共判据均被引用，`math` 仅为 `compare` 的路径别名重导出、无第二份实现。计时闭包输入为 `View`，clone 是引用 + 小 strides 的描述符复制，非元素深拷贝，输出缓冲跨迭代复用；返回值由 criterion 0.8.2 `Bencher::iter` 内部 `black_box(routine())` 兜底。
- **原登记存量阻塞（后续批次均已收口）**：`crates/diagnostic/src/buildgen.rs` 早先两个 `Result` 函数缺 `# Errors` 小节、`clippy -D warnings` 在依赖链阻断——现已补齐全部 `Result` 函数的 `# Errors` 小节，workspace `clippy --all-targets --all-features -D warnings` 全绿；`crates/prof/src/build.rs` 90 行触生产文件行数下限门——构建入口已迁至包根 `crates/prof/build.rs`，静态门按构建脚本口径归类（不套生产行数下限 / Ccn），`every_source_passes_the_static_gate` 通过。
- 函数长度、圈复杂度、嵌套等门禁数值以最终格式化代码的静态报告为准；本批新增判定函数均 ≤10 行、分支 ≤3。
- PMU 采样、iai-callgrind、多后端与服务化场景在当前 CPU 参考装置上不具备运行条件，列入后续阶段。
- 未生成性能报告或基准产物；复跑命令为 `cargo bench -p llm-benchmarks` 与 `cargo test -p llm-benchmarks --test regress`，
  历史归档路径 `target/bench-report/<host>/`。

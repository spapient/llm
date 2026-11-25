# 0059：基准体系的命名分类、分层架构、统计判定与质量口径

- 状态：**提案（proposed）**。本文是对 `crates/benchmarks` 的现状审计、业界对标与目标设计；
  落地按 §6 路线分阶段，未落地的能力在验收前一律记 OPEN，不据本文宣称全绿或数值最优。
- 日期：2026-11-26
- 相关：[`engine-benchmark-roadmap.md`](../engine-benchmark-roadmap.md)（P0–P4 阶段与可证伪条件）、
  [`bench-roofline-analysis.md`](../bench-roofline-analysis.md)（roofline 分子/分母口径）、
  [`lld.md`](../lld.md)（E1 TTFT/TPOT、E3 服务吞吐、E6 适配性、E7 带宽、E10 可预测性）、
  [`PROMPT-S++-AUDIT.md`](../PROMPT-S++-AUDIT.md) §8（compute ≥90% / bandwidth ≥85%）、
  [`crates/tools/src/bench.rs`](../../crates/tools/src/bench.rs)、[`card.rs`](../../crates/tools/src/card.rs)、
  [`regress.rs`](../../crates/tools/src/regress.rs)、[`stats.rs`](../../crates/tools/src/stats.rs)

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
| 服务负载与指标 | MLPerf 四场景：single-stream、multi-stream、server（Poisson 到达）、offline；LLM 拆 TTFT/TPOT，报 tail p90/p99，有 accuracy gate，禁止缓存结果或用额外权重；vLLM 基准报 TTFT/TPOT/ITL/E2EL、mean/median/p90/p99、并发扫描、稳态平台窗口（剔除 ramp-up/drain）、满足 SLO 的 goodput、ISL/OSL 分布 | [MLPerf Inference](https://github.com/mlcommons/inference)、[vLLM benchmarks](https://docs.vllm.ai/projects/benchmarks/en/latest/)、[benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) |
| 统计严谨性 | criterion：bootstrap 置信区间、OLS 斜率/R²、中位数/MAD、IQR 离群比例；iai-callgrind：走 Cachegrind 出确定性指令数/L1L2/LLC 缓存计数，绕开墙钟噪声（Linux/Valgrind 边界内）；hypobench：最小效应规模门、多重比较 Bonferroni 校正、bootstrap change CI、Welch t 检验按迭代均值、A/A 重复消除假阳性 | [criterion.rs book](https://bheisler.github.io/criterion.rs/book/)、[iai-callgrind](https://github.com/iai-callgrind/iai-callgrind)、[hypobench](https://github.com/vorner/hypobench) |
| roofline | 算术强度 I=FLOPs/byte，上界 P=min(π, I·β)，ridge/balance point，分层 L1/L2/DRAM 多坡；FLOPs 与流量优先用 PMU 实测而非估算；缓存计数可作确定性佐证 | 本仓 `bench-roofline-analysis.md`（roofline 上界口径） |
| 环境可重复性 | performance governor、锁频、关 turbo/boost、绑核（taskset）、隔离核、关 SMT/NUMA 干扰、drop caches、冷热分离、A/A 重复、长采样；记录调频/温度/功耗/页缓存状态 | [Google Benchmark: perf vs micro](https://github.com/google/benchmark/blob/main/docs/perf-vs-microbenchmark.md) |
| 趋势与回归归档 | 按 commit + machine 存历史结果；CI compare 两版，因子阈值 + 统计上下界；显著回退自动定位到提交区间；历史以 JSON 入库，滚动窗口 + 中位数 + 容差 | [airspeed velocity](https://asv.readthedocs.io/en/stable/commands.html)、[github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark) |

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
| D4 | 形状偏窄：decode 行宽固定 4096；精度 f32/i2/i4/i8，无 f16/bf16/f8(e4m3/e5m2)/nf4 | 缺真实模型形状网格与精度-性能曲线 | P3 |
| D5 | 热路径分配/复制/缓冲复用未在基准内计数；内存峰值、tokens/J 未测量 | 性能结论缺分配与能效证据，违反 AGENTS.md §6.1 缓冲复用可观测要求 | P2 |

### 3.3 架构

| # | 现状证据 | 对标缺口 | 等级 |
|---|---|---|---|
| A1 | 判定类型在 `tools`，采集在 `benchmarks/src`，台账/分析/归档在 `benchmarks/tests`；依赖方向基本正确，但峰值档案与达成分析件分散在两处 | 五层边界需显式固化：定义 / 统计 / 台账 / 采集 / 归档，统计与判定层保持零时钟、零 IO、可单测 | P0 |
| A2 | `KernelCase`（形状/dtype/后端）与服务负载（ISL/OSL/批/并发/到达分布/窗口）未分离抽象 | 新增 serving 场景会侵入算子基准；应把负载生成器独立于被测引擎 | P4 |
| A3 | `Harness` 仅 host/cpu/os/arch/toolchain/driver（`bench.rs:36`） | 环境档案缺绑核、governor/boost、SMT/NUMA、页缓存冷热、温度/功耗、采样数/预热；未测得留 `None` | P1 |
| A4 | 归档 JSON 落本机 `target/bench-report/`，无跨 commit/跨运行历史库与趋势对比 | 缺可回查的时间序列与 CI 回归对比层 | P2 |
| A5 | roofline 只分 bandwidth / not-bandwidth，算力 vs 延迟瓶颈不分；无 PMU 佐证 | 缺分层 roofline（L1/L2/DRAM 多坡）与计数器证据；延迟瓶颈需 prof 采样，当前不具备 | P3 |

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
- `Workload`：ISL/OSL 长度分布、批大小网格、并发度、到达过程（closed-loop / Poisson）、
  稳态窗口定义、SLO/超时、预热与重复次数。负载生成器不依赖被测引擎内部实现，只经公共接口发请求。

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
- 每条归档带：命令、工具版本、产物路径、主机档案、跳过/忽略/阻塞原因；Open/Skip/Ignored 结构化汇总。

### 4.6 趋势归档（A4/Q4）

- 历史结果按 `(commit, machine, profile)` 入本地/CI 历史库；CI compare 两版，输出统计上下界与因子阈值，
  显著回退定位到提交区间；基线用滚动中位数 + 区间对照，而非单份快照。
- 稳态吞吐按 E3 口径：每配置 ≥3 次独立重复，报均值/中位数与 95% CI，附错误率/超时率/队列时间。

## 5. 正确性与性能绑定（D2）

- 每条 `Op/Fused/Layer` 性能基准挂一个数值对拍：按 dtype 预注册容差（max rel err / abs err），
  参考实现为 CPU 标量锚点；正确性未过门，该性能格记 Open，不出性能数。
- 性能与正确性用同一 `KernelCase` 几何，避免两套形状漂移；容差口径写入台账行，不写进算子实现。

## 6. 落地路线（对齐 roadmap P0–P4，每阶段可证伪验收）

| 阶段 | 改动面 | 验收口径（可回查） |
|---|---|---|
| P0 架构定型 | `EntryKind` 补五级；两套 Roofline 正名；身份/路径单一派生；五层边界与依赖单向化自检 | 依赖方向门通过；命名门（name-lint）零新违规；台账身份唯一自检通过；无第二套路径转换 |
| P1 CPU 闭环 | 算子回归改区间判定 + 效应规模 + 校正；A/A 前置基准；消费 criterion 结构化产物；EnvProfile 扩位；正确性对拍绑定 | 区间判定单测覆盖 Pass/Fail/Open 三态；A/A 基准入档；每条性能格带容差对拍；产物字段完整可解析 |
| P2 运行时层 | Layer/Model 基准；KV 长度/批网格；TPOT/ITL/P50/P95/P99；稳态窗口；内存峰值/分配计数；历史趋势库 | 每配置 ≥3 次独立重复入档；分配计数有计数证据；趋势 compare 产出上下界与回退标记 |
| P3 后端/量化/观测 | 多后端行；f16/bf16/f8/nf4 精度曲线；分层 roofline；PMU/缓存计数佐证；能效 tokens/J | 同口径跨后端对照；精度-性能曲线可复算；瓶颈侧有计数器证据，不靠猜测 |
| P4 服务化 | Serving 场景（single-stream / server Poisson / offline）；TTFT/TPOT/goodput；并发-Pareto；同口径对照 llama.cpp/vLLM | E1/E3 口径：分相报数、长尾分位、goodput、SLO 满足率；跨实现对照须同设备同负载，异源数字只作情报 |

## 7. 边界与不做

- 不造通用 IR/融合编译器（ADR 0017 边界）；基准只经公共接口调用被测对象。
- 不用理论峰值、他机数字或参考实现冒充实测；达成率不等于硬件利用率。
- 不跨机比绝对值；未建档主机/无设备行记 Open，不删行、不降阈值。
- 不引入生产非必要依赖；确定性计数工具按平台边界挂接，不可用记 Open。
- 不把 kernel-only 数字当服务吞吐；prefill 与 decode、输入与输出 token 不混加（E3 口径）。

## 8. 未完成项与证据边界

- 本文为提案；§4 各决策均未落地，验收前对应格保持 OPEN。
- 函数长度、圈复杂度、嵌套、jscpd 等门禁数值须以最终格式化代码的静态报告为准，本文不预设全绿。
- PMU 采样、iai-callgrind、多后端与服务化场景在当前 CPU 参考装置上不具备运行条件，列入后续阶段。
- 未生成性能报告或基准产物；复跑命令沿用 `cargo bench -p llm-benches` 与 `cargo test -p llm-benches --test regress`，
  历史归档路径 `target/bench-report/<host>/`。

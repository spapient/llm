# 推理引擎对标与超越路线（llama.cpp / burn / candle / vLLM / TensorRT-LLM）

- 版本：v1.2
- 日期：2026-11-15
- 状态：方向已定（**速度优先**；**Metal + CUDA 双后端并行**）；架构决策看板见 `implementation-task-plan.md` §4 D1–D13
- 适用范围：全 workspace（24 个 `crates/*` 成员 + 根包 `llm`）
- 配套文档：[`ARCHITECTURE.md`](ARCHITECTURE.md)（分层与依赖方向）、[`10k-analysis.md`](10k-analysis.md)（规模、脂肪与万行预算）、[`audit.md`](audit.md)（静态审计）
- 取证口径：现状条目均回链到源码 `文件:行`；标「推断」的是由源码事实推出、未在目标硬件实测；未实测的性能数字不预填、不谎报。

---

## 0. 结论与定位

**北极星目标：在明确、可测量的指标上逐对象超越 llama.cpp、burn、candle，并长期对照 vLLM / TensorRT-LLM 的服务化指标。** 超越是终局目标，不是当前能力——设备后端仍是契约级原型（无真实设备队列或 kernel launch）；CPU 侧已有参考内核、可写张量与分配器，经训练缓冲通路跑通链式前向。本路线的职责是：

1. 把终局目标拆成可验证的追平/超越里程碑，每个里程碑都有同硬件、同输入的对照基准；
2. **现在就把通向目标的抽象边界定死**（后端、张量、内存、KV、量化、通信、调度、基准），避免到 P3 接后端时推倒重写；
3. 诚实记录每一步的实测差距，而不是用口号替代测量。

本仓在架构纪律、类型闭集、对拍质量、依赖密度上已是一流；这些是超越 burn/candle 这类纯 Rust 框架的天然长板。与 llama.cpp / vLLM / TRT-LLM 的差距集中在后端实现密度、内核矩阵与服务化成熟度，属于阶段问题，按路线逐阶段补齐。

## 1. 对标阵营与超越指标

不同对手的强项不同，超越指标不能混为一谈。对照必须在**同一硬件、同一模型、同一输入形状与批次**下进行；公开宣传数字不可直接对比。

| 对标对象 | 它的强项 | 本仓要超越的指标 | 追赶里程碑 |
|---|---|---|---|
| **llama.cpp** | CPU/端侧推理、量化矩阵、多指令集手写 kernel、极低内存占用 | 同模型同硬件：TPOT、decode 单 token ms、`tokens/s`（首要）；TTFT、峰值内存、量化精度-性能曲线、端侧能效（次要） | M2 CPU 追平同档 decode 速度；M3 起 decode 吞吐/时延目标超越，内存与代码密度并行保持 |
| **burn** | 纯 Rust 后端泛型 trait 塔、可移植后端覆盖 | 同后端下算子性能、新增一个后端的代码成本、编译期保证强度、对拍完备度、外部依赖数 | M3 同后端打平到领先；架构/密度指标全程保持领先 |
| **candle** | Rust 模型仓库 + CPU/CUDA/Metal 三后端 | Rust 生态内端到端前向速度、后端覆盖广度、模型加载内存 | M3/M4 对照 |
| **vLLM / TRT-LLM** | 量产 GPU 服务、continuous batching、PagedAttention、成熟内核库 | 服务化吞吐、并发度、TTFT / TPOT / P99、长尾时延 | M4 在选定细分场景（端侧、可移植、内存受限）对照；通用 GPU 峰值作为长期北极星 |

### 1.1 指标族（所有对照统一口径）

优先级（已定）：**decode 速度第一**——TPOT、单 token ms、`tokens/s` 是超越判定的首要指标；内存占用、能效次之；架构密度与依赖面作为长期保持项，不为速度牺牲。

- **延迟**：首 token 时延 TTFT、每 token 生成时延 TPOT、decode 单 token ms。
- **吞吐**：单序列 `tokens/s`、批 `tokens/s`、服务化聚合吞吐。
- **内存**：权重驻留、KV 缓存字节/序列、峰值 RSS / 显存。
- **能效**：`tokens/J`（M3 起测量）。
- **正确性**：对拍 `max_rel_err` 在 dtype 容忍阈值内。
- **架构密度**：机制代码行数、外部依赖数、`unsafe` 边界、每后端新增代码量、复用率。

## 2. 口径与边界

### 2.1 超越主张的可证伪条件

任何「领先 / 超越」结论必须对应一条可重复执行的基准：输入固定、硬件与驱动记录、批次与形状记录、回归门禁守护。静态代价估算（einsum 推 flops/流量）**不是实测吞吐**，只作 roofline 上界参考。在自己的可运行后端落地之前，本仓不出任何吞吐对比结论——M1 的第一件事是把对照基线 harness 建起来。

### 2.2 不变的取舍边界

1. **不造通用 IR / 跨算子融合编译器**（ADR 0017；重建触发条件见 `10k-analysis.md §9`：实测算子边界开销可量化且 roofline 未达带宽上限，或片段无法表达所需融合）。这是为代码密度与可解释性主动放弃的广度，不影响后端性能超越。
2. 路线内不含模型动物园、vision、RL 等功能广度目标；本仓定位是机制与内核/运行时，产品面不在本路线。
3. 超出 roofline 包络的特化（wmma/wgmma、手写张量核）按阶段以片段数据接入，不写进框架本体；是否接入由实测 roofline 达成率决定（衔接 `10k-analysis.md §7.2`）。

## 3. 现状基线（取证事实）

| 项 | 现状 | 证据 |
|---|---|---|
| 成员规模 | 24 个 `crates/*` + 根包 `llm`；317 个 `.rs` | `ls crates/*/`；`find crates -name '*.rs'` |
| 测试 | 1,396 passed / 0 failed（1 ignored）；doc-test 全过 | 收口工作树实测（2026-11-17） |
| 注册算子 | 24 个命名算子（gemv/gemm/attention/flash_attention/decode_attention/decode_flash_attention/rope/rmsnorm/softmax/激活/归约/量化反量化等） | `crates/nn/src/registry.rs` 注册表投影 |
| 输出语言闭集 | 9 种 Emit：Cuda、Ptx、Hip、Opencl、Spirv、Metal、Ascendc、C、Wgsl | `crates/prof/src/prof.rs:16-35` |
| 执行后端 | **只有契约**：图/流/节点/启动几何，没有真实流、设备队列或 kernel launch | `crates/engine/src/exec.rs:8` |
| 张量存储 | 只读 `&'a [u8]` 视图 + 可写 `OwnedTensor`（元数据 + 拥有字节向量）；CPU `Buffer` / `Allocator` / `CpuAllocator`；跨设备拷贝是 `Allocator` 显式方法、后端实现未落地 | `crates/tensor/src/lib.rs`、`crates/storage/src/lib.rs` |
| KV 缓存 | 页池已持真实 `Vec<f32>`（row/row_mut/write 经 Slot 寻址）；前缀页共享引用、多序列批量循环未接 | `crates/engine/src/kv.rs:204-312` |
| 服务端点 | 路由/响应/线格式契约，无 socket；HTTP/gRPC 是待实现的 `Endpoint` 后端 | `crates/serve/src/lib.rs` |
| 训练侧 | 循环状态、检查点、激活缓冲簿记；反传公式在 `autograd`，激活显存计划未定 | `crates/train/src/lib.rs:1-15` |
| 分布式 | 切分拓扑与集合簿记，无通信后端 | `crates/shard/src/*` |
| 外部 FFI | 全仓唯一生产 FFI 是 `storage` 的 mmap 内存映射 | `crates/storage/src/lib.rs:252`（内联 `mmap` 模块） |
| unsafe 策略 | `unsafe_code = "deny"`，生产核心无 `unsafe` | `Cargo.toml:56` |
| 生产依赖 | 生产路径仅 `dtype` 可选 feature 引入 `serde`/`bytemuck`，`fuzz` 可选 `libfuzzer-sys`；`criterion`/`clap` 等属 dev/build 依赖 | 各 `Cargo.toml` |
| 基准 | criterion 结构/编解码基准 + 七条真实算子基准（`gemv_f32` / `softmax_f32` / `rope_f32` / `rmsnorm_f32` / `attention_f32` / `gemm_f32` / `silu_f32`）全部入 `tools::regress` 基线判定；macos-aarch64 首跑 mean 已登记（gemv 2186061 ns、softmax 15492 ns、rope 9487 ns、rmsnorm 2374 ns、attention 517442 ns、gemm 16599 ns、silu 3287 ns，跨机绝对值不可比）；另有 `mem_stream` 三档读写搬往上界作 roofline 分母，定位结果见 [`bench-roofline-analysis.md`](bench-roofline-analysis.md) | `crates/benches/Cargo.toml`、`crates/benches/tests/regress.rs` |
| 现有门禁 | fmt、clippy `-D warnings`、nextest、doc-test、cargo-deny、cargo-machete、jscpd（行 1.44% / token 1.73%，阈 0.48）、bench `--no-run`、cutile 覆盖率下限（行 75% / 函数 70%；上次 `--coverage` 采集行 76.84% / 函数 77.98%） | `scripts/ci.sh`、`../.config/.jscpd.json`、`.github/workflows/ci.yml` |

## 4. 能力维度对标矩阵

成熟度口径：**契约** = 类型/接口已定、无实现；**簿记** = 只管理元数据/账；**片段** = 生成代码文本、未接运行时；**实现** = 可在真实缓冲上跑出结果。

| 维度 | vLLM / TRT-LLM | burn / candle（Rust 路线） | llama.cpp | 本仓现状 | 证据 | 归属阶段 |
|---|---|---|---|---|---|---|
| 后端执行 | 真实设备队列、kernel launch、事件同步 | 多后端 trait | 多指令集手写 kernel | 契约 | `engine/src/exec.rs:8` | P1（CPU）/ P3（加速后端） |
| 张量与内存 | 可写显存/内存、布局、分配器、跨设备拷贝 | 张量 + 后端分配器 | 紧凑缓冲、低内存 | 只读 `&[u8]` 视图 | `tensor/src/lib.rs:130` | P1 |
| 算子内核 | 手写高性能矩阵/注意力/量化 | 每后端手写 kernel | 指令集特化 kernel | 24 条声明 + 模板片段 + 标量参考 | `nn/src/registry.rs` | P1 起逐算子落地 |
| 注意力/解码 | PagedAttention、FlashAttention、投机解码 | 基础注意力 | 优化 decode 路径 | 页表簿记 + 两个注意力声明 | `engine/src/kv.rs`、`nn` | P2 |
| 批处理调度 | continuous batching、prefix cache、并发准入 | 基础批次 | 单序列/小批 | 调度契约 | `engine/src/exec.rs` | P2 |
| 量化 | AWQ/GPTQ/FP8/INT4 反量化内核与权重布局 | 部分量化 | 多比特量化矩阵 | 精度事实源 + 权重分组推断，无计算内核 | `dtype`、`weights` | P3 |
| 分布式 | TP/PP + 集合通信（NCCL 等） | 有限 | 有限 | 拓扑簿记，无通信后端 | `shard/src/*` | P3 |
| 性能测量 | 吞吐/时延/带宽 roofline 回归门禁 | criterion/bench 对拍矩阵 | 自研 bench | CPU 参考实现基准 + 本机 roofline 定位（7 条算子 + 3 档搬运上界），无 kernel 吞吐 | `benches/`、`docs/bench-roofline-analysis.md` | P1 搭骨架，P2/P3 接内核 |
| 模型端到端 | 主流权重开箱前向 | 模型仓库 | 主流模型 | 读 safetensors，对拍标量参考，无模型前向 | `weights`、`llm` bin | P4 |
| 服务化 | HTTP/gRPC 服务、并发、流式 | 有限 | 有限 | 端点契约 + 线格式，无 socket | `serve/src/lib.rs` | P4 |
| 架构纪律 | vLLM 服务单体；TRT 绑定厂商 | burn trait 塔；candle 三后端重复 | C 过程式 | 分层所有权、依赖方向门、闭集、单一所有者 | `ARCHITECTURE.md`、`crates/tool/tests/gate.rs` | 已领先，P0 加固 |
| 质量工程 | 测试/CI 成熟 | 后端对拍矩阵 | C 测试 | 980 用例 + 标量对拍 + 全套静态门禁 | `scripts/ci.sh` | 已一流，P0 补属性测试 |

## 5. 面向超越的架构预留（现在定死，防止后期返工）

这是本路线最容易被低估的部分：P0/P1 阶段如果只把「能跑通 CPU」做掉、不留设备抽象，P3 接 GPU/第二后端时张量所有权、算子分发、内存布局都要重写。下列边界在对应阶段动工前必须先以 ADR 定形，即使第一版只有一个实现。

| 预留点 | 现在必须定死的接口形状 | 若现在不做对，后期返工什么 | 落地阶段 |
|---|---|---|---|
| 后端 trait 边界 | `Device` / `Backend` / `Allocator` / `Executor` 抽象；算子经 trait 分发，新增后端不改模型、调度、权重代码；trait 形状须同时容纳 Metal 与 CUDA 两套实现，两套 FFI/unsafe 都收在同一后端边界 crate | 双后端各自分叉出私有设备假设，第二套实现要回改所有算子与模型代码 | P0 定形状，P1 出 CPU 实现，P3 挂 Metal/CUDA |
| 张量所有权与地址空间 | 可写缓冲 + 只读视图双形态；布局/stride/设备标识与具体设备解耦；地址抽象预留（CPU 指针 → 未来设备句柄） | P3 接设备内存时重写张量与所有权模型 | P1 |
| 内存生命周期 | 拥有/借用/跨设备拷贝接口预留；零拷贝视图贯穿读路径 | 缓冲生命周期与设备拷贝散落各处，逐点补 | P1/P2 |
| 算子多后端挂载 | 算子声明与实现解耦；同一算子可注册多后端实现，选择走能力闭集；CPU 算法不硬编码进算子 | 加后端要在每个算子里开条件分支 | P1 |
| KV 数据接口 | 页表簿记与数据持有分离；「页背后的缓冲」接口在簿记阶段就留好 | P2 翻页表结构以挂真实数据 | P1 留接口，P2 填实现 |
| 量化抽象 | 精度闭集、权重容器、反量化内核三者解耦；新量化格式不碰权重读取 | P3 加量化格式要回改 `weights` | P1 解耦，P3 接内核 |
| 通信接口 | 拓扑簿记与集合通信 trait 分离；先定 trait 形状，P3 填后端实现 | P3 接通信要翻 `shard` 拓扑结构 | P2 留接口，P3 填实现 |
| 调度与后端解耦 | continuous batching 策略不依赖具体执行后端；准入决策只看能力闭集与 KV 状态 | 调度器绑死 CPU 假设，换后端要重写 | P2 |
| 产物缓存 | 内容寻址 key/artifact/复用计数；为 AOT/JIT 产物预留 | 特化产物复用路径后期重搭 | P0/P1 |
| 基准 harness | 统一测量接口（形状、硬件、批次、精度、后端标识、roofline 上界）；新后端自动接入对照 | P3 没有可对比的吞吐基线，无法判定超越 | P1 |

## 6. 分阶段路线与超越里程碑

每阶段出口必须「可运行 + 可测量 + 门禁守护」；上一阶段出口未达标，不进入下一阶段。所有阶段遵守 `ARCHITECTURE.md` 依赖方向与 ADR 不变量。里程碑中的「追平/超越」均指同硬件同输入对照，数字在对应阶段实测采集后回填本文档。

### P0 质量与架构加固（低风险，不引入后端）

- **目标**：把已领先的结构质量做到无懈可击；按 §5 定死后端、张量、KV、量化、通信、基准的接口形状，为真实实现铺好骨架。
- **内容**：
  - 关键纯函数补属性测试，边界与退化情形进模糊测试；
  - 固化后端/分配器/缓冲/算子挂载/KV/通信/基准的 trait 形状（可先只有编译期类型，不出实现）；
  - 按 `10k-analysis.md §6` 压平台侧脂肪（解析器收敛、审计/报告器合一、孤儿成员处置）；
  - 文档与契约对齐，消除文档漂移。
- **出口指标**：nextest 全绿且用例数不跌；clippy `-D warnings` 零告警；生产依赖面不扩大；覆盖率不下限；jscpd 不破 0.48；§5 接口形状有 ADR 落档；不新增运行时后端或可写缓冲。
- **暂缓**：算子真实实现、调度器落地、FFI。

### P1 最小可运行闭环（从契约到能跑，建立对照基线）

- **目标**：CPU 上跑通一条真实前向；**建成可复用的对照基准 harness**，第一次采集与 llama.cpp 的同硬件基线（不论差距大小，先有可对比数字）。
- **内容**：
  - 可写张量与 CPU 分配器；视图保留为零拷贝读路径；
  - 落地 ≥3 个真实算子，按 decode 主路径优先：`gemv_f32`（decode 带宽瓶颈）、`softmax_f32`、`rope_f32`，elementwise 次之；每个同时保留标量参考对拍；
  - 一层模型端到端前向，数值与标量参考对拍达标；
  - 基准 harness 统一记录硬件/形状/批次/后端与 roofline 上界；criterion 产出真实时延/吞吐采样入回归基线（现状见 [`bench-roofline-analysis.md`](bench-roofline-analysis.md)）。
- **里程碑 M1**：CPU 前向可重复运行；对拍 `max_rel_err` 在容忍阈值内；跑出与 llama.cpp 同模型同硬件的首批对照数字并回填文档；架构门通过；新增 unsafe 为零。
- **暂缓**：GPU 后端、连续批、量化内核、分布式。

### P2 推理运行时（CPU 追平 llama.cpp 同档）

- **目标**：簿记升级为持有真实数据的推理运行时；批处理与 KV 落地，CPU 对照 llama.cpp 达到同档 decode 速度。
- **内容**：KV 缓冲落地（页表背后是真实张量）；continuous batching 调度器与准入；注意力内核接入；prefix cache；优化排序按 decode 瓶颈算子（注意力、gemv）的 roofline 贡献度。
- **里程碑 M2**：多序列批处理端到端跑通；单序列/批次吞吐与时延对照 llama.cpp 曲线入表，CPU 达到同档；内存占用对比入表；KV 分配/释放有对拍与边界测试；调度不违反页表不变量。
- **暂缓**：加速后端特化、分布式、服务化并发。

### P3 后端、量化与性能（Metal + CUDA 并行，同后端对标 burn/candle）

- **目标**：在隔离边界内同时接入 Metal 与 CUDA 两条加速链路，补量化与分布式原型，进入 roofline 优化；同后端对比 burn/candle 打平到领先，decode 速度为首要判定。
- **前置约束**：新增**独立后端边界 crate**，把 `unsafe` 与 FFI 关在该边界内；Metal（`metal-rs`/系统框架）与 CUDA（driver/runtime）两套 FFI 都隔离在边界 crate 内，生产核心仍保持安全 Rust 与 `unsafe deny` 策略，靠局部 `allow` 收敛到边界模块（需一条新 ADR 明确边界规则）。
- **环境前提**：Metal 链路在本机 macOS（M5 Pro / Metal 4）验证；CUDA 链路在 win 远端（T600 sm_75，CUDA 13.3）验证；设备内核管线未建前两条链路只保留编译与契约对拍，不产出吞吐数字。
- **内容**：Metal 与 CUDA 并行落地，经同一后端 trait 挂载；量化反量化内核；roofline 达成率报告；集合通信原型。
- **里程碑 M3**：Metal 与 CUDA 后端各自在目标硬件跑通 P2 同一前向；同算子 CPU/加速后端对拍一致；同后端关键算子对比 burn/candle 打平到领先；decode 吞吐/时延有对照数字；roofline 达成率可追；内存占用、代码密度、依赖数并行保持领先；边界 crate 之外无新增 `unsafe`。
- **暂缓**：投机解码、大规模分布式、第三加速链路。
- **实测记录回填（2026-11-19/20，本仓 `tools/musa-harness/`）**：
  - **Windows CUDA 链路（性能线 1）**：win 远端（T600、driver 621.14、nvcc）全仓 `cargo check`
    通过、cli lib 43/43 + 集成 14/14 全绿，CUDA 真机对拍（逐点网格 + 分块归约）2/2 与标量参考一致；
    macOS 本机 Metal 对拍 14 条同口径全绿——两条链路各自对同一标量参考一致，M3 的「同算子对拍一致」
    以间接互证形式达成。设备内核管线未建前不出吞吐结论（本仓自研 kernel 的吞吐测量属 MUSA 研究线，见下）。
  - **MUSA 研究线（S4000 / MUSA 5.2.0 / muBLAS 5.2，纯 kernel 计时，2026-11-19）**：muBLAS 基线
    cube 1024³ 6.55、prefill 4096×512×14336 14.40、n1536 17.77、n768 15.64 TFLOPS，精度
    `‖Δ‖∞/‖C‖∞ = 1.56e-6`（纯 FP32 路径）。本仓自研：WMMA(tf32) wave128 正确形态 cube **10.60
    （领先 62%）**、prefill 13.24、n1536 10.18、n768 8.61（精度 2.79e-4 超 1e-2 阈值）；
    FP32 SIMT v4 全精度 6.32/6.74/9.81/9.68；3xTF32 长 K 精度出局。硬件边界：S4000(mp_22) 无
    cp.async（SINCE_MP_43），5.2 工具链 SIMT 调度退化（旧 3.1 的 prefill 10.74 无法复现）。
  - **口径说明**：以上 MUSA 数字是研究线测量（远端 S4000、纯 kernel 事件计时），非本仓框架后端
    产物；回填为对照基线供 M3 后端落地时参照。**主目标（通用 F32 全面超越 muBLAS）当前状态：
    cube 已超越，主场景 55-92%，差距根因已量化**。


### P4 模型与服务化（对照 vLLM / TRT-LLM）

- **目标**：真实权重端到端前向 + 真实服务端点；在选定细分场景对照量产服务引擎。
- **内容**：主流权重格式端到端加载与前向；`Endpoint` 的真实 HTTP/gRPC 实现；并发与流式输出；服务化吞吐/时延基准。
- **里程碑 M4**：给定真实权重跑通生成；端点契约被真实服务器实现替换；服务基准记录并发与 P50/P99；与 vLLM/TRT-LLM 在同硬件同负载下对照 TTFT/TPOT/吞吐，端侧/内存受限细分场景争取领先。

## 7. 性能主张的可证伪口径

- 性能来源只承认三处：数据驱动特化（每形状一份产物）、原生编译器做寄存器/调度、引擎层分页 KV + 连续批 + 量化；框架层保持零动态分发、零 IR 往返、零拷贝视图。
- 对外只报可验证的门：roofline 达成率、对拍 `max_rel_err` 在阈值内、产物复用率、基准回归不退化、同硬件对照差距数字。
- 硬件相关数字必须记录机型、驱动、输入形状与批次；跨机器绝对值不可比，CI 守「不回归」，绝对领先结论只在同环境对照下给出。
- 基准基线采集计划：M1 起每个算子/层/整模型都有对照条目；先记 llama.cpp（CPU），M3 起加 burn/candle（同后端），M4 加 vLLM/TRT-LLM（服务化）。

## 8. 前置约束与决策点

| 决策点 | 选项 | 代价 / 约束 |
|---|---|---|
| unsafe 与 FFI 边界 | P3 前保持核心 `unsafe deny`；后端 unsafe 隔离到边界 crate 并出 ADR | 不隔离则破坏既有内存安全定位 |
| 依赖策略 | 生产路径继续趋零外部依赖；或接受唯一解析/序列化依赖换掉数千行手写解析 | 破零依赖原则需显式拍板（`10k-analysis.md §八`） |
| 后端策略（已定） | **Metal + CUDA 并行**：经同一后端 trait 挂两套实现，FFI/unsafe 同收边界 crate | 双后端要求 trait 先稳定，避免私有假设分叉；Metal 在 macOS 本机验，CUDA 在 win 远端验；设备内核管线未建前只出编译/契约对拍，不出吞吐 |
| crate 合并 | 小 crate 降为模块文件以压仪式成本 | 失去编译期并行与强制边界，靠架构门补偿 |
| 行数预算 | 新功能先给行数预算（棘轮），超预算红灯、功能排队 | 见 `10k-analysis.md §6.1` |
| IR 取舍 | 维持无 IR；触发条件满足再以 ADR 重建 | 通用 IR 会推翻万行预算与「无 IR」论点 |
| 超越优先级（已定） | **速度优先**：decode 时延/吞吐为首要指标，内存与密度为并行保持项 | 决定 P1/P2/P3 算子与优化排序按 decode 瓶颈（gemv/注意力）优先；不为速度牺牲内存安全与正确性对拍 |

## 9. 与既有文档的关系

- 本文定义**对标目标、超越里程碑与架构预留**；规模/脂肪/行数预算见 `10k-analysis.md`，落地模块划分见 `ARCHITECTURE.md`，类型所有权与不变量见 `docs/adr/0001-0024`。
- P0 的脂肪清理承接 `10k-analysis.md §6.2` 手段清单；P1 起的运行时实现受 ADR 0003（kernel 与运行时类型）、0012（量化权重）、0014（真实权重执行）约束。
- §5 的每个预留点方向已定（见 `implementation-task-plan.md` §4 看板 D1–D13）；对应 ADR 正文在动工前补档（尤其 P0 后端 trait 形状、P3 的 unsafe 边界与后端选型）。
- 里程碑实测数字在对应阶段回填本文档 §1.1 指标族与各阶段里程碑条目；未回填前不对外引用为领先结论。

# `llm` / `llm.u` / Burn 三仓源码级重评（2026-11-24 快照）

> 本轮为**重新取证**：不沿用历史评分向量，所有 `confirmed` 主张回链到本轮实际读取的 `path:symbol:line`；读取范围之外的运行激活状态标 `inferred` / `unknown`，不以目录名、注释或旧实现补齐。
>
> 取证纪律见 `doubao-coding-analyze-codebase`：定义存在 ≠ 已装配 ≠ 生产激活；测试装配 ≠ 目标环境；局部机制不外推到跨组件副作用。

## 0. 身份卡与证据矩阵基线

| 仓库 | revision | worktree 状态 | 规模（源码实测，剔除 `target`） | 测试数（`#[test]`/`#[proptest]` 计数） |
| --- | --- | --- | --- | --- |
| llm（新） | `4b369630` 2026-11-24 00:35 +0800 | 脏：`docs/COMPARISON-…md` M；`crates/device/src/tiled/attention.rs`、`transpose.rs` 未跟踪（并行线产物，仅作快照差异） | 23 crate，631 `.rs`，163,319 行 | 2721 |
| llm.u | `cf4e8fe3` 2026-11-11 04:55 +0800 | 脏：`Cargo.lock`/`Makefile`/`README.md`/多 crate `Cargo.toml` 与 `device/src/…` 大量 M/RM | 14 crate，349 `.rs`，104,592 行 | 1127 |
| Burn | `c81a9042` 2026-11-22 13:06 -0500 | 干净（零改动） | 41 crate（`burn-*`），1980 `.rs`，488,917 行 | 6754 |

**证据矩阵（本轮点名阶段 → 定位 → 观测到的分支/被调实现 → 主张范围 → 状态）**

| 阶段 | literal locator（llm） | 观测 | 范围 | 状态 |
| --- | --- | --- | --- | --- |
| 逻辑张量类型 | `crates/tensor/src/tensor.rs:57` `Tensor<B,R,K>`；`:68` `View`；`:71` `ViewMut`；`:78` `DynTensor` | 存储 `B` / 秩 `R` / 元素 `K` 三位泛型；`Rank` 对 `Vec<usize>` 与 `[usize;D]` 双实现（`rank.rs:63,77`） | 类型定义与静态关系 | confirmed |
| 元素能力分组 | `crates/tensor/src/lib.rs:129-149` `Element/NumericElement/OrderedElement/FloatElement/IntElement/BoolElement` | sealed 分组，算子族把 `Elem` bound 收紧到对应分组 | trait 定义 | confirmed |
| 生产执行器装配 | `crates/serve/src/model.rs:27` `CpuExecutor::new(Device::new("cpu","generic"))` | serving 路径选 CPU 执行器 | 该装配点 confirmed；其余设备选择未在本仓生产入口见到 | confirmed |
| Metal 执行器 | `crates/device/src/executor.rs:279,404` `MetalExecutor`/`impl Executor`；构造仅见 `crates/cli/tests/device.rs:17,493`、`cli/tests/run.rs` | 定义存在；构造点只在测试 | 测试装配，非生产路径 | confirmed（测试侧）/ inferred（生产未接） |
| CUDA / 昇腾 | `crates/device/src/cuda.rs`、`metal.rs`、`platform.rs`；`Executor` 实现者枚举仅 `MetalExecutor`/`CpuExecutor`/`LocalExecutor`/`Trace`/`Counting`(test) | 未发现 CUDA/昇腾的 `Executor` impl；源生成 / 见证回读范围 | 仓内无对应执行器实现 | confirmed（搜索范围 `crates/` 全量） |
| 自动微分消费 | `crates/train/src/session.rs:5,138,178,183` 引用 `autograd::{Requirement,Sizes,Tape,Var}`；`backprop.rs`/`learner.rs`/`pool.rs` 同引 | train 生产模块消费 tape/var/schedule | 静态依赖 confirmed；端到端梯度数值正确性本轮未跑 test 以外的运行观测 | confirmed（接线）/ unknown（运行数值） |
| 优化器 | `crates/train/src/optimizer.rs:3,186` `Sgd`/`Adam`/`Muon`；AdamW 解耦衰减 + 偏差修正 + 范数裁剪 | 优化器实现在位 | 定义与结构 confirmed | confirmed |

> 未覆盖范围（本轮未逐项深核）：llm.u 各后端在真机上的运行激活与性能；Burn 后端在本机设备的实测；三仓训练数值精度对拍；`benchmarks` 数值结果。这些项下文只给代码结构层面的定级。

## 1. 仓库定位与架构哲学

### 1.1 llm（新）：七层契约 + 编译期守门的参考/编排栈

- 工作区根 `Cargo.toml` 是纯 workspace，无 package；组合根与两个二进制 `llm`/`cutile` 都在 `crates/cli`。
- 分层自述（各 `lib.rs` 头部）：`tensor` 逻辑张量 → `dtype` 元素/量化格式 → `storage` 双 lane 缓冲视图 → `nn` 算子与层声明 → `algorithm` 模型变体语义 → `engine` serving 状态机 → `serve` 请求/采样/路由；`autograd`/`train` 训练侧；`device` 设备源 lowering；`cutile`/`annotation` 内核 DSL 与契约宏；`prof` 能力档案与片段构建期对齐；`weights` 权重编码；`shard` 拓扑切分。
- 工程口径：`[workspace.lints.rust] unsafe_code = "deny"`；`clippy = all warn`；`dbg_macro`/`todo`/`unimplemented` = deny。运行期外部依赖极薄（`half`/`num-traits`/`bytemuck`，`serde` 仅 `dtype` 可选 derive feature）。

### 1.2 llm.u：配置驱动的旧版执行框架

- `device` config-first：`config/device.toml` 为后端选择 + tile 闭集唯一事实源，越界启动期报错；无 `build.rs`。
- `exec` 承载算子族（algo/attn/comm/decode/dequant/engram/format/mixer/model/nn/plan/prec/quantize/reg/resident/safetensor/sampler/stack/teacher/weights/worker）；`engine` driver/kv/output/queue/recovery/scheduler；`serve` 经 `ExecutorClient` 端口提交/轮询/取消。
- 后端代码按 feature 条件编译，多后端目录在位（见 §4）。

### 1.3 Burn：通用张量框架的经典后端分层

- `burn-backend` 定义 `Backend`/`BackendTypes` 算子族 trait；`burn-cubecl` 是跨设备编译层；`burn-cpu`/`burn-cuda`/`burn-wgpu`/`burn-rocm` 为薄壳后端；`burn-tensor` 张量 API；`burn-nn`/`burn-train`/`burn-autodiff`/`burn-optim`/`burn-linalg`/`burn-einsum`/`burn-fusion`/`burn-ir` 上层；`burn-remote`/`burn-shard`/`burn-communication`/`burn-dataset`/`burn-store` 周边。

## 2. 规模与工程度量（本轮实测）

### 2.1 体量对比

| 指标 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| crate 数 | 23 | 14 | 41 |
| `.rs` 文件 | 631 | 349 | 1980 |
| 源码行 | 163,319 | 104,592 | 488,917 |
| 测试计数 | 2721 | 1127 | 6754 |

### 2.2 llm 生产行 Top crate（行）

| crate | 行 | 职责 |
| --- | --- | --- |
| nn | 23,261 | 算子内核 / 层声明 / 性能下界 |
| weights | 15,329 | 权重格式、PTQ、codec、存档 |
| train | 12,776 | 训练循环状态、loss、优化器、检查点 |
| prof | 12,579 | 能力档案 + 设备片段构建期对齐 |
| device | 12,200 | 设备源 lowering 与厂商后端 |
| algorithm | 10,397 | 注意力/SSM/记忆/路由变体语义 |
| serve | 10,106 | 请求文法、采样、补全端点、路由 |
| shard | 9,535 | 拓扑/流水线/切分/见证 |
| cutile | 7,846 | 内核 DSL：AST/emit/fuse/pass/plan |
| cli | 6,723 | 组合根与二进制 |

### 2.3 llm.u 生产行 Top crate（行）

| crate | 行 |
| --- | --- |
| device | 30,809 |
| exec | 26,660 |
| storage | 10,343 |
| train | 7,821 |
| tool | 5,765 |
| engine | 5,677 |
| config | 4,799 |
| benches | 4,772 |

### 2.4 Burn 关键层行（行）

| crate | 行 | crate | 行 |
| --- | --- | --- | --- |
| burn-train | 33,086 | burn-tensor | 24,949 |
| burn-nn | 29,463 | burn-fusion | 27,313 |
| burn-cubecl | 25,240 | burn-backend | 16,399 |
| burn-autodiff | 12,368 | burn-core | 11,675 |
| burn-optim | 10,777 | burn-ir | 9,040 |
| burn-linalg | 6,010 | burn-einsum | 1,113 |

### 2.5 unsafe 边界

- llm：workspace 默认 `#![deny(unsafe_code)]`；`grep` 到含 `unsafe`/`allow(unsafe_code)` 的文件 10 个，其中 `device/src/executor.rs`（Metal FFI）、`storage/src/mmap.rs` 为 ADR 点名的越界岛，其余为测试或 native 探针。岛清单由 `testkit` 逐个点名守卫（`device/src/lib.rs:31-33`）。
- llm.u：`engine`/`exec`/`graph` `#![forbid(unsafe_code)]`；设备 FFI 在 `device` 内。
- Burn：按后端 crate 各自管理 FFI（cuda/wgpu/metal 路径），本轮未逐文件核 unsafe 计数。

## 3. 张量模型与类型系统（维度 1）

### 3.1 张量抽象对照

| 对照点 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| 张量形态 | `Tensor<B, R, K>`（`tensor.rs:57`） | 存储层扁平 `Tensor`（`storage/src/legacy.rs:18`、`gguf.rs:127`）/ `View`（`store.rs:262`） | `Tensor<const D: usize, K = Float>`（`burn-tensor/.../base.rs:76`） |
| 存储位 `B` | `Store`：`&'a [K]` / `Owned<K>`（`store.rs`） | 缓冲持有在 storage 自身 | 后端句柄经 `BackendTypes` 关联类型 |
| 秩位 `R` | `Rank` trait，`Vec<usize>` 运行期秩与 `[usize;D]` 定长秩**双实现**（`rank.rs:63,77`） | 无独立秩泛型 | `const D: usize` 编译期定秩 |
| 元素位 `K` | sealed 元素分组：`Element/NumericElement/OrderedElement/FloatElement/IntElement/BoolElement`（`lib.rs:129-149`） | 元素类型落在 codec/format | `K = Float/Int/Bool/Quantized` |
| 视图桥 | `View<'a,K>` / `ViewMut<'a,K>` / `DynTensor<K>`（`:68,71,78`） | `store::View` | 后端 tensor 句柄；`BridgeTensor` 进出后端 |
| 形状校验 | 构造期拦下轴数/元素数/可达性/溢出；连续与非连续布局、stride、transpose/permute/reshape 零拷贝元数据重排（`tensor.rs:12-37`） | 存储布局在 `format`/`plane` | 成熟形状算子面（reshape/flatten/squeeze/unsqueeze 等，`base.rs:445,748,790…`） |

**confirmed 结构差异**

- llm 的 `R` 同时接受定长数组与运行期 `Vec<usize>`；Burn 的 `D` 只能是编译期定长数组。llm 文档自述这是刻意放宽：运行期 batch / segment 几何要进同一类型（ADR 0046）。
- llm 的 `B` 是**存储**而不是后端算子集；算子由 `nn` 数据驱动注册表给出，不走泛型派发的百方法后端接口（ADR 0053）。Burn 的 `Backend` 是算子族 trait 组合（见 §4.2）。
- llm 逻辑张量数值算子面薄：`ops.rs` 12 个元素/归约（`map/add/sub/mul/div/add_scalar/mul_scalar/sum/max/min/argmax/all_close`），`tensor.rs` 45 个形状/构造/存取方法；重算子在 `nn` crate。Burn `burn-tensor` 24.9k 行，算子面显著更宽。

### 3.2 算子接口与注册表

- **llm**：编译期 const 注册表 `REGISTRY`（`nn/src/registry.rs:79`），`for_each_operator!` 单清单声明 52 个算子（`:20-71`），名字与 `prof::catalogue` 同序，错位编译期失败；查表二分（`:123`）。算子类型由清单一次展开类型化 API，注册表/类型化入口/样本三者不漂移。
- **llm.u**：`exec/src/reg.rs` `Impl` 七臂枚举 `Gemm/Scalar/Qgemm/Attention/Kv/Map/Reduce`（`:32-41`），`dispatch!` 宏 match 非穷尽静态分派，漏改编译失败；`Factory = fn() -> Impl`（`:68`）。
- **Burn**：算子是 `Backend` trait 上的关联方法族，后端逐个实现；无仓内名字注册表。

### 3.3 形状视图算子面

llm 文档列的形状族（`tensor.rs:20-31`）：`broadcast/narrow/slice/chunks/split/select/unbind/squeeze/unsqueeze/diagonal/flatten/transpose/permute/movedim/slice_fill/slice_assign/flip`；除 `flip` 外零拷贝。改秩六算子需指定目标秩见证（稳定 Rust 算不出 `D-1`），与 Burn `.squeeze::<2>()` 同一手法。本轮未逐算子对拍 llm 与 Burn 的边界行为，算子等价性标 unknown。

## 4. 后端与执行模型

### 4.1 后端清单与实现状态（本轮实测）

| 后端 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| CPU | `CpuExecutor`（`engine/src/cpu.rs:246`）；生产 serving 装配点 `serve/src/model.rs:27` confirmed | `crates/device/src/host/cpu.rs:46` `backend_impl!(Cpu,…)` | `burn-cpu`（薄壳，经 CubeCL/CPU 路径） |
| CUDA | `device/src/cuda.rs` 源生成；**无 `Executor` impl**（全仓实现者枚举已核） | `backend/cuda` 9546 行；feature `cuda → dep:cudarc` | `burn-cuda` 薄壳 + CubeCL CUDA |
| Metal | `device/src/metal.rs` + `executor.rs:404 impl Executor for MetalExecutor`；构造点仅见 `cli/tests/device.rs:17,493`、`cli/tests/run.rs`（macos cfg） | `backend/metal` 4291 行；feature `metal → dep:metal` | CubeCL wgpu/MSL 路径 |
| AMD ROCm | 仓内未发现后端目录 | feature `rocm → dep:libloading`；`host/traits/ssm.rs` rocm 条件段 | `burn-rocm` 薄壳 + CubeCL HIP |
| OpenCL/SYCL/INTEL | `device/src/…` intel fragment（`backend/intel` 在 llm.u 1352 行） | feature `opencl`/`sycl`/`clblast`；`backend/blas` 2984 行 | — |
| MUSA / 昇腾 / ANE | 源生成 / 档案位流，无执行器 impl | `musa` feature；`npu.rs`；`ane → dep:coreml-native` | — |

> 激活边界纪律：上表 llm 一行，"代码目录/源生成存在"不等于该后端在生产入口被选中。本轮 confirmed 的生产设备选择只有 `serve/src/model.rs:27` 的 CPU；Metal 仅测试构造；CUDA/昇腾无 `Executor` 实现。llm.u 后端的真机运行激活本轮未取证（unknown）；Burn 各后端运行状态以其仓库 CI/文档为准，本轮未在本机复跑。

### 4.2 执行缝（trait 面）

- **llm**：`Executor` trait（`engine/src/queue.rs:10`）：`device/submit/sync/register_program/bind_f32/bind_bytes/fetch_f32/fetch_bytes`；除 `submit/device` 外多为默认 no-op 实现。实现者：`MetalExecutor`、`CpuExecutor`、`LocalExecutor`、`Trace<E>` 装饰器、`Counting`（`engine/tests` 测试桩）。`device → engine` 依赖单边（`device/src/lib.rs:7-8,18-20`）。
- **Burn**：`Backend` trait（`burn-backend/src/backend/base.rs:124`）= `BackendTypes + FloatTensorOps + BoolTensorOps + IntTensorOps + ModuleOps + ActivationOps + QTensorOps + TransactionOps + DistributedOps + CollectiveOps + Clone/Default/Sized/Send/Sync/Debug/'static`；`BackendTypes` 关联 `Device/TensorPrimitive/FloatElem/IntElem`（`:29-31`）。`base.rs` 内 trait 方法 55 个。
- **llm.u**：`device/src/host/traits.rs` 后端能力槽；未覆盖槽位默认返回 `Err(Run("backend does not provide …"))`（grad/adam/fetch/half/fp8 等，`:405,426,431,443…`），由覆盖后端实现。

### 4.3 内核 DSL 与代码生成

- **llm**：`cutile`（ast/builtin/cache/cap/emit/fuse/harness/kernel/lru/pass/plan/probe/reference/space/substitute/validate）+ `annotation`（`#[spec]`/`#[model]`/`#[dsl]` 宏，syn 解析、quote 生成，编译期依赖不进运行期图）。`cutile` 自述：P0 模板纵切面 + typed kernel 开始迁移，**不取代旧模板，不宣称通用优化或 GPU 执行**（`cutile/src/lib.rs`）。
- **Burn**：`burn-cubecl` 25.2k 行跨设备张量 DSL/编译层；`burn-ir` 9.0k 行中间表示；`burn-fusion` 27.3k 行算子融合。
- **llm.u**：内核族直接落 `exec/` 与 `device/backend`，无独立 IR crate（`graph` 818 行）。

## 5. 自动微分

| 对照点 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| 图/tape 结构 | `autograd` 3954 行：`Requirement/Role/Var/Node/Tape/Ordering/Schedule/Account/MemoryPolicy/Plan/Budget/Sizes/Footprint/Choice/Config` | 手写反向（`exec` 内 `plan.rs`/`teacher`） | `burn-autodiff` 12.4k：`graph/grads/checkpoint/distributed/ops/runtime/backend` |
| train 接线 | `session.rs:5` 引 `Tape/Var`；`backprop/learner/pool` 同引 confirmed | train 复用 exec 计算面 | autodiff 作后端装饰器挂接 |
| 梯度检查点 | segment 级（`autograd` 结构内） | 未见独立检查点策略 | `autodiff/checkpoint` 策略可配 |
| 状态 | 仓内小图，生产模块接线在位；梯度数值正确性本轮未跑运行对拍 | 手写反向，成熟度本轮未深核 | 生产级梯度跟踪/快照/反向 |

## 6. 模型层（nn）

### 6.1 层/算子清单

- **llm `nn`**（23.3k）：注册表 52 算子（`registry.rs:20-71`）——elementwise（add/sub/mul/div/max/min/scale/residual）、激活（gelu/silu/relu/sigmoid/tanh/exp/log/leaky_relu/hard_sigmoid）、归一化（rmsnorm/batch/group/instance/layer norm）、注意力（attention/flash/decode_attention/decode_flash/latent）、池化（adaptive/avg/max pool1d/2d）、卷积（conv2d）、reduce（max/sum）、embed、rope、scan、softmax、gemm/gemv/matmul（f32 + i2/i4/i8）、dequant（i2/i4/i8）。层声明 `Linear`/`OpLayer`/`ContractSpec`/`Bounds`（`contract.rs:178`、`declare.rs:161,285`）。
- **Burn `burn-nn`**（29.5k）：`modules/` 下 attention/cache/conv/embedding/dropout/linear/interpolate/norm/pool/pos_encoding/rnn/transformer/unfold/rope_encoding/pixel_shuffle/fold/noise/identity/pairwise_distance/cosine_similarity；`loss/` 18 个文件（17 种 loss + `reduction` 归约工具；BCE/cross_entropy/ctc/gaussian_nll/hinge_embedding/huber/kldiv/lp_loss/margin_ranking/mse/multi_margin/poisson/rnnt/smooth_l1/soft_margin/triplet/cosine_embedding）。
- **llm.u `exec`**：`Impl` 七臂（Gemm/Scalar/Qgemm/Attention/Kv/Map/Reduce）。

### 6.2 架构变体覆盖

- **llm `algorithm`**（10.4k）：`Mla/Latent/BlockTable/Mamba/Kda/AttnRes/Moe/Mhc/Hybrid/Tower`；`Wiring/Family` 拓扑组装。注意 `algorithm/src/lib.rs:199` 自述"nn 无专家路由层（Moe 不给 FFN 宽）"——MoE gating 在算法语义层，nn 层无对应专家 FFN 层。
- RNN/GRU/LSTM：llm 与 llm.u 本轮未见到对应层实现；Burn `modules/rnn` 在位。
- Transformer：Burn 有 Encoder/Decoder + cache；llm 注意力层 + KV 在 engine/serve 侧组装。

### 6.3 模块组织方式

- llm：层由声明宏 + 契约 spec 生成，算子注册表单一清单派生类型化 API；无 `Module` derive 宏。
- Burn：`burn-derive` 提供模块派生宏，层为普通 Rust struct + trait 实现。
- llm.u：内核经 `Impl` 臂 + Factory 分派。

## 7. 量化与精度

| 对照点 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| 量化格式 | `dtype`：GGML block 量化 + MX 元素集 + 亚字节；算子侧 i2/i4/i8 dequant 与 gemm/gemv 各精度 | I2–I8，fp8/fp4 槽位（`host/traits` 默认 Err 槽） | Q8/Q4/Q2 + FP8/FP4，`QTensorOps` |
| 权重 PTQ | `weights/src/ptq.rs`/`ptq_adaptive.rs`/`quant.rs`/`svd.rs`/`rans.rs` | 打包 codec 枚举表驱动 | `burn-backend/quantization` |
| 参数级量化映射 | 未见 ModuleMapper/Quantizer 式参数级映射 | 未见 | ModuleMapper/Quantizer 配套 |
| QAT | 三仓本轮均未见到 QAT 训练链路 | 同左 | 同左 |

## 8. 训练栈

| 对照点 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| 训练循环 | `train` 12.8k：session/learner/backprop/loss/optimizer/geometry/kernel/pool/rng；forward/backward 以闭包注入（`learner.rs:164-182`） | `train` 7.8k：loss/优化器/调度/打包/检查点/rollout 桥 | `burn-train` 33.1k：完整训练循环/指标/检查点 |
| 优化器 | `Sgd/Adam`(AdamW 解耦衰减 + 偏差修正 + 全局范数裁剪) `/Muon`（`optimizer.rs:3,186`） | 仅 `Adam`（`train/src/optim.rs:12`） | 12 个具体优化器：sgd/adam/adamw/adan/lamb/lbfgs/lion/momentum/muon/rmsprop/adafactor/adagrad + decay/grad_accum/parallel |
| LR 调度 | warmup + anneal（`session.rs` `Schedule`） | cosine + warmup 类 | `burn-optim/lr_scheduler` 多策略 |
| loss | cross_entropy/argmax_hits/mse/bce_with_logits/huber/hinge + Distill/Grpo（`loss.rs:143,284`） | 复用 exec 计算面 | 17 种 nn loss（18 文件含 reduction）+ train 指标 |
| 指标/评估 | Learner 账表 + evaluate；无可插拔指标/渲染器 | 未见独立指标注册表 | 30+ 指标（burn-train） |
| 数据集 | `Dataset` trait + chunks 游标；无物理数据源 | 未见独立 dataset crate | `burn-dataset` |
| 检查点 | weights/storage 自描述编码 + CRC-32 + 恢复事实 | checkpoint 字节存档 | `burn-store`/checkpoint 完整 |
| RL | log_prob/entropy/advantage/GRPO 槽位（`loss.rs` Grpo/Distill） | rollout 桥 | `burn-rl`：env/policy/buffer |

## 9. 推理服务与运行时

### 9.1 调度 / 连续批 / KV

- **llm `engine`**（5.2k）：`Sequence/Scheduler/Batch/Admission/State`、KV 页账（`kv.rs`）、radix/pager、提交执行（`exec.rs`）、队列（`queue.rs`）；`serve`（10.1k）：`Request/Predict/Sampling/Greedy/Completion/Router/endpoint/http/frame/block/plan/params/admit/assembly` + `speculate.rs` 投机采样 + `kernels.rs` CPU 内核登记。生产装配 `CpuExecutor` confirmed。
- **llm.u `engine`**：driver/kv/output/queue/recovery/scheduler；`serve` `Facade` 持 `Box<dyn ExecutorClient>`（`facade.rs:10-16`），测试用 `Mock` 实现端口（`:57`）。
- **Burn**：库本身不携带 serving 状态机；连续批/分页 KV/前缀共享属应用层，`burn-remote` 是客户端发张量操作到服务端执行的远程后端。

### 9.2 服务能力判定（边界纪律）

- llm：请求文法、采样、补全、路由、投机采样在仓内实现；HTTP 为薄适配器（`serve/src/http.rs`）。本轮未起真实服务进程测端到端，HTTP 运行时接通状态标 inferred。
- llm.u：请求入口与前端组件为参考层，经端口提交/轮询/取消；后端实现可换，本轮未核生产端口接线。
- Burn：示例 server，无内置生产 serving 框架。

## 10. 分布式与并行

| 对照点 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| 拓扑/切分结构 | `shard` 9.5k：`topology/collective/pipeline/tiling/program/ir/remote/ledger/runner/witness/verify/sweep` | `exec/comm.rs` + config 能力协商 | `burn-shard`（张量按设备切分，张量并行）、`burn-communication`（remote 后端网络） |
| 跨进程集合通信 | `shard/collective.rs` 结构在位；真实传输链路本轮未核运行证据 | Comm trait + CPU 空参考实现倾向（本轮未逐行核） | WebSocket/remote backend |
| 数据/张量并行 | 拓扑轴切分数据结构（`topology.rs`/`tiling.rs`） | config 表 | shard rules + DistributedOps/CollectiveOps trait |

> 三仓的集合通信本轮都只核到仓内结构/trait 层；真实多进程传输是否接通，均标 unknown。

## 11. 随机数与确定性

- **llm**：`train/src/rng.rs` 计数器式 PRNG：`block(key, stream, idx) -> [u32;4]`（`:55`），`Rng::new(seed)`/`with_stream`/`next_u32/u64/f32/f64`（`:78-121`）。确定性由计数器流位置决定，不依赖系统随机源。
- **llm.u**：随机数以注入闭包方式进入（本轮未见到独立 PRNG crate）。
- **Burn**：`rand`/seed 体系，按设备种子派生。

## 12. 算子与内核覆盖（按家族实测）

### 12.1 llm 注册算子（52 项，`registry.rs:20-71`）

池化（adaptive_pool/avg_pool2d/max_pool1d/max_pool2d）、卷积（conv2d）、归一化（batch/group/instance/layer/rmsnorm）、注意力（attention/flash/decode_attention/decode_flash/latent）、elementwise（add/div/max/min/mul/sub/residual_add/scale）、激活（exp/gelu/hard_sigmoid/leaky_relu/relu/log/silu/sigmoid/tanh）、归约（reduce_max/reduce_sum）、embed、rope、scan、softmax、矩阵族（gemm_f32/matmul_f32/gemv_f32 + gemm_i2/i4/i8、gemv_i2/i4/i8、matmul_i8）、反量化（dequant_i2/i4/i8）。

### 12.2 llm.u 内核族（`reg.rs:32-41`）

`Gemm/Scalar/Qgemm/Attention/Kv/Map/Reduce` 七臂；`exec` 内另见 decode/engram/mixer/quantize/deformat/format/sampler 等描述符。

### 12.3 Burn 算子面

后端 `FloatTensorOps/BoolTensorOps/IntTensorOps/ModuleOps/ActivationOps/QTensorOps/TransactionOps/DistributedOps/CollectiveOps` 全族；`burn-linalg` 6.0k、`burn-einsum` 1.1k；nn 层与 loss 见 §6.1。

### 12.4 CPU 手写微核

- llm：`kernel_fast`/`kernel_fast_tn`/`kernel_fast_i8`/`kernel_fast_pool`/`kernel_gemv_simd_stable`/`simd_nightly`/`prof/src/native`（NEON/I8/SME 探针在位；AVX 路径本轮未核到）。
- llm.u：`host/simd`/`host/gemm`/`backend/blas`（NEON/AVX/Accelerate 倾向，本轮未在本机复跑）。
- Burn：CPU 后端可用，极致调优依赖 CubeCL 生成。

## 13. 工程质量与可维护性

| 对照点 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| lint 门槛 | unsafe_code deny、clippy all warn、dbg/todo/unimplemented deny；missing_docs/unused_lifetimes warn | forbid(unsafe_code) 分 crate | 成熟 CI/deny 配置 |
| 外部运行期依赖 | 极薄：half/num-traits/bytemuck；serde 仅 dtype 可选 derive | 多后端 feature 拉 cudarc/metal/libloading/coreml-native | wgpu/cudarc(经 cubecl)/serde/half 等 |
| 测试计数 | 2721 | 1127 | 6754 |
| 静态门 | `tools` config 驱动质量门；本轮 gate 测试 `every_source_passes_the_static_gate` 拦截 19 项行数/复杂度赤字（既有基线，见 §15.3） | 未见同构静态门 | CI 门 |
| 文档/契约 | ADR 目录（`docs/adr`）、层职责自述写在各 lib.rs 头部 | config-first 契约 | contributor book/book |

### 13.1 本轮门禁实测（llm 工作区）

- `cargo check --workspace --all-targets`：绿，无 error/warning。
- `cargo clippy --workspace --all-targets`：绿。
- `cargo fmt --check`：绿。
- `cargo test --workspace`：除静态门 1 例外全绿；失败项 `llm-tools --test gate` 的 `every_source_passes_the_static_gate`，列 19 项超标（weights 5、prof einsum、train 6、device 4、fuzz harness Call=1、tensor.rs 730 行）。

## 14. 能力矩阵总表

图例：● 仓内实现且本轮见到接线/可用证据；◐ 端口/骨架在位，部分实现、占位或本轮未核到运行激活；○ 本轮范围内未发现；？ 本轮未取证，不下定级。

| 能力维度 | llm（新） | llm.u | Burn |
| --- | --- | --- | --- |
| CPU 内核（手写 SIMD 微核） | ◐ NEON/I8/SME 微核在位；AVX 路径待核 | ◐ simd/gemm/blas 在位，本机未复跑 | ● CPU 后端可用 |
| CUDA 后端 | ○ 源生成在位，无 Executor impl | ◐ cuda 9.5k + cudarc feature，运行状态 ？ | ● CubeCL CUDA |
| Metal 后端 | ◐ executor impl 在位，仅测试构造 | ◐ metal 4.3k + metal feature，运行状态 ？ | ◐ CubeCL wgpu/MSL |
| AMD ROCm/HIP | ○ 未发现后端目录 | ◐ rocm feature + libloading | ● CubeCL HIP |
| OpenCL/SYCL/INTEL | ○ 仅 fragment | ● intel 1.4k + blas 3.0k + opencl/sycl feature | ○ |
| MUSA / 昇腾 / ANE | ◐ 源/档案位流，无执行器 | ◐ musa/npu/ane feature 门控 | ○ |
| 张量秩 / 后端泛型 | ● `Tensor<B,R,K>`，秩双形态 | ○ 存储层扁平 Tensor/View | ● `Tensor<const D,K>` |
| 运行期形状/精度校验 | ● 构造期校验 + spec/testkit | ● config 三层合并校验 | ● 后端能力协商 |
| 自动微分（图/tape） | ◐ 仓内小图，train 接线在位 | ○ 手写反向 | ● 后端装饰器 autodiff |
| 梯度检查点 | ◐ segment 级 | ？ | ● 策略可配 |
| 优化器丰富度 | ◐ Sgd/Adam/Muon | ◐ 仅 Adam | ● 12 个含 LBFGS/Momentum/Muon |
| LR 调度 | ● warmup+anneal | ● cosine+warmup 类 | ● 多策略 |
| 指标注册表/evaluator | ◐ Learner 账表 + evaluate，无可插拔指标 | ？ | ● 30+ 指标 |
| 数据集 crate | ◐ Dataset trait + chunks，无物理源 | ？ | ● burn-dataset |
| 检查点体系 | ● 自描述编码 + CRC-32 + 恢复事实 | ◐ checkpoint 字节存档 | ● store/checkpoint |
| CNN 卷积层 | ● conv2d | ？ | ● |
| 池化层 | ● | ？ | ● |
| RNN/GRU/LSTM | ○ | ○ | ● |
| Transformer 模块 | ◐ 注意力/算法语义 + engine 组装 | ◐ 注意力 KV | ● Encoder/Decoder + cache |
| SSM/Mamba | ● algorithm/kernel | ● 前反向 | ○ |
| MoE 路由 | ◐ algorithm gating，nn 无专家 FFN 层 | ？ | ○ |
| 量化格式广度 | ● GGML block + MX + 亚字节 i2/i4/i8 | ◐ I2–I8 + fp8/fp4 槽位 | ● Q8/Q4/Q2 + FP8/FP4 |
| 参数级量化映射 | ○ | ○ | ● ModuleMapper/Quantizer |
| QAT | ○ | ○ | ○ |
| 连续批调度 | ● engine/ifb | ● scheduler | ○ 应用层 |
| 分页 KV + 前缀共享 | ● radix/page | ● radix/page | ○ 序列 cache |
| 投机采样 | ● speculate | ● speculative | ○ |
| OpenAI 兼容端点 | ◐ route/protocol + HTTP 薄适配器 | ◐ schema + 纯逻辑路由，HTTP 运行时未核 | ○ 示例 server |
| 权重格式读取 | ● safetensors/gguf/json/torch.bin | ● gguf/safetensors/hf-pickle | ○ pytorch-reader |
| 内置分词器 | ● tokenizer | ○ | ○ |
| 拓扑/切分数据结构 | ● shard | ◐ config | ● shard rules |
| 跨进程集合通信 | ○ 结构在位，传输未核 | ◐ Comm trait，CPU 参考 | ◐ WebSocket/remote |
| 确定性 RNG | ● 计数器 PRNG + 三分布 | ○ 注入闭包 | ● rand/seed |
| 自定义内核 DSL | ● cutile + annotation 宏 | ○ 直写后端 | ● CubeCL + IR + fusion |
| 模块派生宏 | ○ | ○ | ● burn-derive |
| RL 栈 | ◐ log_prob/entropy/advantage/GRPO | ○ rollout 桥 | ● env/policy/buffer |
| 远程训练/推理 | ○ | ○ | ● remote/communication |

## 15. 关键差距与风险（按优先级）

### 15.1 llm 相对 Burn 的主要缺口（本轮证据支持）

1. **后端运行面窄**：生产入口本轮见到的只有 CPU 装配（`serve/src/model.rs:27`）；Metal 仅测试构造，CUDA/ROCm/OpenCL/MUSA/昇腾无 `Executor` 实现。Burn 经 CubeCL 提供跨设备生产后端。
2. **算子/层广度**：Burn 有 RNN、完整 Transformer encoder/decoder、17 种 loss、12 个优化器、30+ 指标、dataset、linalg/einsum/fusion/IR；llm 层与 loss/优化器面更窄，RNN/可插拔指标/物理数据源缺位。
3. **张量算子深度**：llm 逻辑张量数值算子仅 12 个元素/归约，重算子在 nn；Burn tensor 24.9k 行、后端算子族全。
4. **模块派生与生态**：llm 无 Module derive 宏、无远程后端、无独立 dataset crate。

### 15.2 llm 的相对强项（本轮证据支持）

1. **类型系统结构新意**：`Tensor<B,R,K>` 存储/秩/元素三位分离；秩同时支持定长数组与运行期 `Vec<usize>`；元素 sealed 分组收紧算子 bound；View/ViewMut/DynTensor 三桥齐备。
2. **依赖极薄 + 编译期守门**：运行期几乎不拖重型数学/序列化依赖；unsafe 默认 deny，仅点名越界岛；静态门 + 单清单注册表防三处漂移。
3. **serving 状态机自给**：连续批、分页 KV/radix、准入、投机采样、请求文法与路由在仓内成体系；内置分词器与自有权重 codec。
4. **算法变体覆盖**：Mla/Latent/Mamba/Kda/Moe/Mhc/Hybrid/Tower 等注意力/SSM/记忆/路由语义在 algorithm 层成族。

### 15.3 本轮发现的存量债务（llm，非本轮引入）

- 静态门 19 项行数/复杂度超标（weights 5、prof einsum、train 6、device 4、fuzz harness Call=1、tensor.rs 730 行），gate 测试 1 例失败；需拆文件/降复杂度，运行时生产代码不允许挂债。
- `tensor/src/tensor.rs:140` `into_slice_mut` 在输出侧回退后无生产调用方，作为 pub 消费式解包方法保留；去留属 API 口径决策。

### 15.4 llm.u 的定位

旧版配置驱动执行框架：设备后端代码量厚（device 30.8k + exec 26.7k），多后端 feature 门控；train/serve 为参考层，经端口抽象。本轮未核其真机运行激活与数值正确性，成熟度定级留待运行观测。

## 16. 未覆盖与待核清单（诚实分层）

1. 三仓后端在真机（CUDA/Metal/ROCm/WGPU）的运行激活、性能与数值正确性——本轮只读代码结构，未起设备复跑。
2. llm 训练闭环的梯度数值正确性：接线 confirmed，未跑训练对拍。
3. llm serve HTTP 运行时是否端到端接通——本轮仅见 `http.rs` 薄适配器，未起服务进程。
4. llm.u 后端各 feature 的实际编译状态与设备覆盖；其 train/serve 端口是否接生产实现。
5. Burn 本机各后端编译/运行状态；CubeCL fusion/IR 的实际生效路径。
6. 三仓形状算子的边界行为逐一对拍（广播/切片/对角/非连续布局）。
7. 旧报告 `docs/COMPARISON-llm-llmu-burn.md` 的历史评分向量本轮未沿用；本报告定级以 2026-11-24 快照源码为准。

> 证据分级：标注 `confirmed` 的定位均可回链到上文 path:symbol:line；标 `◐`/`？` 的项为结构在位但运行激活未核，不构成"生产可用"主张。

# `llm` / `llm.u` / Burn 三仓源码级重评（2026-11-25 第八轮快照）

> 本轮为**回到现场重新取证**：不沿用历史评分向量，所有 `confirmed` 主张回链到本轮实际读取的 `path:symbol:line`；读取范围之外的运行激活状态标 `inferred` / `unknown`，不以目录名、注释、旧实现或仓外材料补齐。
>
> 取证纪律见 `doubao-coding-analyze-codebase`：定义存在 ≠ 已装配 ≠ 生产激活；测试装配 ≠ 目标环境；局部机制不外推到跨组件副作用；未提交 diff 只作快照差异，不计入已提交基线。

## 0. 身份卡与证据矩阵基线

### 0.1 三仓快照（本轮实测）

| 仓库 | revision | worktree 状态 | 规模（源码实测，剔除 `target` / `.tmp` / `.claude`） | `#[test]`/`#[tokio::test]` 标签 | unsafe 计数 |
| --- | --- | --- | --- | --- | --- |
| llm（新） | HEAD `0178a110` 2026-11-25 18:50:36 +0800（取证区间 HEAD 由 `12410532`→`33bd536`→`3c2dd74a`→`0178a110` 连续推进） | 工作区 mixed：train `checkpoint.rs`/`geometry_shapes.rs`/`lib.rs`/`session.rs`/`tests/session_contract.rs` 与 `cutile/harness.rs` 等处于并行重构中途；`.tmp/` 脚本清理不计入代码基线 | 21 members，磁盘 560 `.rs` / 165,796 行（HEAD tree 口径 554 `.rs`） | 2813 | 94 |
| llm.u | `cf4e8fe3` 2026-11-11 04:55:22 +0800 | 170 项未提交；结论以 HEAD 已提交代码为准，脏改动仅作快照差异 | 349 `.rs` / 104,592 行 | 1127 | 437 |
| Burn | `c81a9042` 2026-11-22 13:06:56 -0500 | 干净（零改动） | 1989 `.rs` / 490,469 行 | 6788 | 246 |

> **快照漂移与编译观测分层（重要）**：llm 工作区存在并行工作流连续自动提交与未提交文件周期收编，取证期间 HEAD 持续推进。下列观测按「当前磁盘快照（最后一次可复核命令）」与「取证中途快照」分开记账；中途断裂不计入当前基线能力裁决。
>
> - **当前磁盘快照观测（confirmed，最后一次可复核命令）**：`cargo check --workspace` `Finished` 无 error；`cargo test --workspace` **3060 passed / 0 failed / 22 ignored**，无 test binary 编译失败；静态门 `every_source_passes_the_static_gate` **13 passed / 0 failed**；`cargo test -p llm-train backprop` **50 passed / 0 failed**（202 filtered）。本轮另取证据：`cargo fmt --all --check` 通过；`cargo clippy --workspace --all-targets --all-features --locked` 干净 `Finished`（无 error/warning）；`cargo deny check` advisories/bans/licenses/sources 四项 ok；`cargo machete` 未发现未用依赖。
> - **取证中途快照观测（已披露，非当前基线）**：① HEAD `12410532` 提交态 `shapes.rs` 两份相同 `check_pool1d` 无 cfg 门控，`llm-train` lib 报 E0428 重复定义；② 另一 dirty 时点调用在、定义缺，报 E0425；③ optimizer.rs 重构中途 clean 后 lib 报 9 errors（E0308 类型不匹配 ×8、E0369 `Result<f32>` 乘 `&f32` ×1），落点 `:310/:314/:371/:376/:380/:476/:479/:483/:646`，根因 `:144` 宏展开 `check_slots` 调用；④ geometry 重组中途 test target 报 E0599 `outputs` 方法缺失 14 errors、`fmt --check` 一度报悬空 `geometry_kernels.rs` path；⑤ 中间快照曾聚合 **2447 passed / 1 failed / 9 ignored**，静态门赤字 `optimizer.rs:check_slots#0` Ccn=7。上述中途状态均被并行流后续收编，当前快照已不复现。
> - llm.u 的 170 项未提交改动未纳入能力裁决；Burn 本机后端运行状态本轮未在本机复跑。

### 0.2 证据矩阵（本轮点名阶段 → 定位 → 观测到的分支/被调实现 → 主张范围 → 状态）

| 阶段 | literal locator（llm，当前快照） | 观测 | 范围 | 状态 |
| --- | --- | --- | --- | --- |
| 逻辑张量类型 | `crates/tensor/src/tensor.rs:11` `Tensor<B,R,K>`；`:22` `View`；`:25` `ViewMut`；`:28` `DynTensor`；`:177` `into_slice_mut` | 存储 `B` / 秩 `R` / 元素 `K` 三位泛型；三视图桥为存储别名；`into_slice_mut` 全仓仅定义处一处、无生产调用方（去留属 API 口径决策） | 类型定义与静态关系 | confirmed |
| 元素能力分组 | `crates/tensor/src/lib.rs:33` 再导出 `Element/BoolElement/FloatElement/IntElement/NumericElement/OrderedElement` | sealed 分组，算子族把 `Elem` bound 收紧到对应分组 | trait 定义 | confirmed |
| 生产执行器装配 | `crates/serve/src/model.rs:19` `CpuModel<E>`、`:24` `CpuModel<CpuExecutor>`、`:27` `CpuExecutor::new(Device::new("cpu","generic"))`、`:453` `impl Predict` | serving 路径选 CPU 执行器；`use engine::exec::{Buf,CpuExecutor,Device,Executor}`（`:7`） | 该装配点 confirmed；其余设备选择未在本仓生产入口见到 | confirmed（CPU 装配）/ inferred（其余设备） |
| Executor 实现者全集 | `prof/src/exec.rs:287` `trait Executor`；`engine/src/queue.rs:10` 同名 `trait Executor`；`prof/src/metal/runtime.rs:284` `MetalExecutor`、`:409` `impl Executor for MetalExecutor`；`engine/src/cpu.rs:92` `CpuExecutor`、`:242` `impl Executor for CpuExecutor`；`engine/src/queue.rs:43` `Trace<E>`、`:107` `LocalExecutor`、`:185` `VacantExecutor` | 生产/参考实现为 CPU/Local/Vacant/Trace 装饰器；Metal 定义在位、new 在 `runtime.rs:293` 内部，仓内未见生产构造点；两处 Executor trait 分属 prof 运行时缝与 engine 队列缝 | 实现者枚举 confirmed；Metal 生产接线未见到 | confirmed（定义/装配）/ inferred（Metal 生产未接） |
| CUDA / 昇腾 / ROCm | `crates/` 全量检索；未发现 CUDA/昇腾/ROCm 的 `Executor` impl | 仓内无对应执行器实现；prof 下 `ascend`/`intel`/`metal`/`musa`/`nvidia` 为源生成/档案 fragment，`opencl.rs` 为受控 FFI 岛 | 仓内无对应执行器实现 | confirmed（搜索范围 `crates/` 全量） |
| 自动微分 | `crates/autograd/src/graph.rs:96` `Var`；`tape.rs:76` `record`、`:84` `record_with`、`:210` `backward`；`crates/train/tests/backprop_tests.rs`（`ABSOLUTE=2e-3`/`RELATIVE=0.05`/`central`/`check_grad`）；`train/src/kernel.rs:44,46` `Reference`；`session.rs:314` `gradient_snapshot` | tape/ordering/schedule 元数据层；`Session<Reference,Sgd>` 在 CPU 参考 Kernel 上建图→plan→backward，解析梯度与中心差分逐参数对拍；本轮 backprop **50 passed / 0 failed**；独立梯度快照按 `Var` 复制、脱离累积窗口与缓冲生命周期。设备反向：仓内未检出设备反向梯度核契约或 Metal/CUDA 反向派发；`docs/evidence/cuda-backward-probes/` 下 `.cu` 为仓外历史材料，未接进仓内通道 | CPU Reference 闭环 + 快照 confirmed；设备反向通路 unknown；仓外探针不计仓内能力 | confirmed（CPU + 快照）/ unknown（设备反向） |
| 优化器 | `optimizer.rs:10` `Group`、`:43` `Slot`、`:65` `OwnedSlot`、`:88` `trait Optimizer`、`:90` `step`、`:224` `check_slots`、`:301` `Sgd`、`:333` `Adam`、`:431` `Muon`、`:589` `Hybrid<M,N>` | `Hybrid` 按矩阵/非矩阵分流到两个子优化器（`split_slots` + `impl Optimizer for Hybrid`）；`MomentState`/`Partitioned`/`DecayClip` 在位；当前快照静态门已转绿 | 定义与结构 confirmed | confirmed |
| train 数据集 | `dataset.rs:49` `from_config`、`:196` `trait Dataset<'a>`、`:218` `Loader<'a,T>`；`dataset.rs:9` 从 `rng` 再导出 `LoaderConfig/Rng/Sampling`；`rng.rs:55` `block`、`:84` `with_stream`、`:96` `next_u32`、`:115` `next_f32` | 顺序/洗牌/加权三族采样（`Sampling` 取代旧 `Sampler`）+ 权重校验；物理分词/打包源仍缺位 | 结构 confirmed；生产数据链路 inferred | confirmed（结构）/ inferred（生产链路） |
| 算子注册表 | `nn/src/registry.rs:17` `for_each_operator!` 宏；清单展开 **72 条**算子；`:99` `REGISTRY`；`:132` 二分查表 | 注册表驱动算子面；编译期名字/顺序双断言 | 注册表与算子名 confirmed；各算子真机内核激活未核 | confirmed（注册表）/ unknown（真机内核） |
| 损失与 RL | `loss.rs:134` `cross_entropy`、`:144` `argmax_hits`、`:171` `mse`、`:179` `bce_with_logits`、`:186` `huber`、`:202` `hinge`、`:243` `log_prob`、`:258` `entropy`、`:274` `advantage`、`:294` `policy_loss`、`:307` `Grpo`、`:399` `grpo_loss`、`:524` `Distill` | 旧 `loss/rl.rs` 子目录已撤，RL 族并入单文件 `loss.rs` | 函数定义 confirmed | confirmed |
| 检查点 / LR 调度 | `checkpoint.rs:18` `Step`、`:39` `Checkpoint`、`:152` `Schedule`（`:161` `new`、`:166` `with`、`:192` `lr`）；`session.rs:300` `accum_snapshot`、`:314` `gradient_snapshot`、`:328` `ties_snapshot`、`:333` `zero_gradients` | 独立 checkpoint 文件承载 step 计数/权重存档/学习率退火；快照按 Var 复制 | 结构 confirmed | confirmed |
| 分布式 | `shard/src/collective.rs:72` `trait Collective<T>`、`:74` `all_reduce`、`:77` `all_gather`、`:86` `broadcast`、`:95` `reduce_scatter`、`:202` `Remote`、`:204` `impl Collective for Remote` | `Remote` 为**显式占位后端**：`all_reduce` 直接 `fail!(Unsupported, "远端集合通信未挂载")`；其余三算子走 trait 默认 no-op；旧 `FileRemote` 文件跨进程后端本轮已撤；`train/src/session.rs:59` `Session<…,C: Collective = shard::Remote>`，`:478` `collective: Option<E>`；24 个 collective 测试针对本地镜像算法 `ReplicaStore` | 端口在位 confirmed；真实跨进程织物未挂载 confirmed；多进程 spawn E2E unknown | confirmed（端口 + 占位）/ unknown（高速织物/E2E） |
| 内核 DSL | `cutile/src/native.rs:17/20/23/27` `pub use` 零成本转发 `prof::native`；`cutile` ast/emit/fuse/pass/plan/harness 等模块；`annotation` 派生宏 | P0 模板纵切面 + typed kernel 起步；不宣称通用优化或 GPU 执行 | 静态结构 confirmed；真机内核激活 unknown | confirmed（结构）/ unknown（真机） |
| unsafe 与内存安全 | workspace lints：`unsafe_code="deny"`、`unsafe_op_in_unsafe_fn="deny"`、`missing_docs`/`unused_lifetimes="deny"`；本轮计数 **94** 处 | 分布 `prof/src/metal/runtime.rs`(37)、`prof/src/opencl.rs`(26，`#![allow(unsafe_code)]` 受控 FFI/RAII 岛)、`prof/src/runtime.rs`(7)、`cli/tests/audit.rs`(7)、`prof/src/sign.rs`(6)、`storage/src/mmap.rs`(5)、`nn/src/kernel/fast.rs`(4)、`nn/src/kernel/fast/pool.rs`(2) | 策略与分布 confirmed；逐块正确性未审计 | confirmed |
| 外部依赖/供应链 | 运行期外部仅 `half`(3)、`num-traits`(2)；`bytemuck`(2)/`serde`(3)/`serde_json`(2)/`toml`(6) optional；`proc-macro2/quote/syn` 编译期宏；`proptest`(11)/`criterion`(1)/`similar-asserts`(1) 测试/bench；`libfuzzer-sys`(1) optional fuzz；无 cudarc/metal-rs/libloading/wgpu/coreml/ndarray/opencl-sys 运行期绑定（Metal/OpenCL 为手写 FFI，未引包装 crate） | 运行期依赖极薄；系统绑定缺位 | 依赖清单 confirmed | confirmed |

> 未覆盖范围（本轮未逐项深核）：三仓真机后端（CUDA/Metal/ROCm/WGPU）运行激活与性能；llm serve HTTP 端到端接通；llm.u 各 feature 实际编译状态与设备覆盖；Burn 本机后端与 CubeCL fusion/IR 生效路径；三仓形状算子边界行为逐一对拍；`benchmarks` 数值结果。下文只给代码结构层面定级。

## 1. 本轮现场结构刷新（相对第七轮）

1. **train 目录继续收口重组**：`dataset/` 目录 → 单文件 `dataset.rs`（300 行）；`loss/rl.rs` 子目录撤掉，RL 族并入 `loss.rs`（703 行）；`geometry.rs`（375）与 `geometry/kernels.rs`（553）、`geometry/shapes.rs`（312）并存，取证中途另有 `geometry_shapes.rs` 平铺文件；`checkpoint.rs` 独立成文件（816 行，Step/Checkpoint/Schedule）。
2. **优化器重写**：`optimizer.rs`（693 行）由旧 Sgd/Adam/Muon 三结构体扩为 `Group`/`Slot`/`OwnedSlot`/`Optimizer` trait/`DecayClip`/`MomentState`/`Partitioned`/`Hybrid<M,N>` 组合器；`check_slots` 承担配置预检与裁剪。中途快照曾有 `check_slots` Ccn=7 赤字，当前快照静态门已转绿。
3. **分布式迁入 `shard`**：train 下旧 `collective.rs`/`remote.rs` 移除，`Collective` 端口在 `shard/src/collective.rs`；`Remote` 由旧文件后端 `FileRemote` 降级为报错占位；`Session` 经泛型 `C: Collective` 持有可选 collective。
4. **算法变体扩充**：`algorithm/src/` 本轮见 `attnres/cell/codec/csa/emit/hybrid/kda/latent/mamba/mhc/mix/mla/moe/patch/policy/pulse/spike/tower`，较上轮新增 `csa/mix/patch/policy/pulse/spike/codec/emit/cell` 等语义文件。
5. **规模**：21 members（上轮 22），磁盘 560 `.rs`/165,796 行/2813 测试标签；注册表 72 条（上轮 71）；unsafe 94（上轮 95）。
6. **后端 FFI 岛**：本轮检出 `prof/src/opencl.rs`（26 unsafe）为 Intel OpenCL 受控 FFI/RAII 句柄岛，`Cargo.toml:34` 注释声明不引入包装依赖；无 OpenCL crate 绑定。

## 2. 规模与工程度量（本轮实测）

### 2.1 体量对比

| 指标 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| crate 数 | 21 | 15 | 42 |
| `.rs` 文件 | 560（磁盘）/ 554（HEAD tree） | 349 | 1989 |
| 源码行 | 165,796 | 104,592 | 490,469 |
| 测试标签 | 2813 | 1127 | 6788 |

### 2.2 llm 生产行 Top crate（行，当前磁盘实测）

| crate | 行 | 职责 |
| --- | --- | --- |
| nn | 24,863 | 算子内核 / 层声明 / 性能下界 |
| weights | 15,875 | 权重格式、PTQ、codec、存档 |
| prof | 15,382 | 能力档案、设备片段、执行缝与 native 微核 |
| train | 13,181 | 训练循环状态、loss、优化器、检查点、数据集、几何 |
| cutile | 12,869 | 内核 DSL：AST/emit/fuse/pass/plan |
| shard | 11,521 | 拓扑/流水线/切分/见证/集合通信端口 |
| algorithm | 10,377 | 注意力/SSM/记忆/路由变体语义 |
| serve | 10,200 | 请求文法、采样、补全端点、路由 |
| cli | 5,996 | 组合根与二进制 |
| testkit | 5,928 | 分层/账本/所有权架构守卫 |
| annotation | 5,801 | 契约宏 |
| engine | 4,958 | serving 状态机、CPU 执行器、队列 |

### 2.3 unsafe 边界（本轮计数）

- **llm**：workspace 默认 `unsafe_code = "deny"`；94 处 unsafe，集中为显式岛（Metal objc FFI、OpenCL 受控 FFI、native 微核、mmap、fast kernel、审计测试）。
- **llm.u**：437 处；`device/engine/exec/tool` 源码层 `#![forbid(unsafe_code)]`，`cli`/`benches` Cargo deny，其余分 crate 未统一；设备 FFI 在 `device` 内。
- **Burn**：246 处；按后端 crate 各自管理 FFI，未检索到 `forbid/deny unsafe_code`。
- 计数口径：源码 `unsafe {`/`unsafe fn`/`unsafe impl`/extern unsafe，剔除所有层级 `target/`、`.tmp/` 与 fuzz 生成 codegen。

## 3. 22 维评分（0–10，本轮证据定档）

### 3.1 标尺

| 分段 | 含义 |
| --- | --- |
| 9–10 | 成熟完备，生产可用，生态/类型支撑完整 |
| 7–8 | 主体完整，少量缺口或局部调优不足 |
| 5–6 | 端口/骨架在位，部分实现或仅单点落地 |
| 3–4 | 最小可用，或只有单点实现/手写集合 |
| 1–2 | 仅声明、配置标签或占位，无真实链路 |
| 0 | 缺失 |

> 分数是当前代码快照的相对成熟度，不是能力上限；三仓定位不同（Burn 通用训练框架 / llm.u LLM 推理多后端执行 / llm 分层重构参考栈），总分须结合权重视角解读。后端真机运行激活与设备反向通路本轮未核，相关维度按代码结构定档。#21 为 unsafe 边界（越界岛越少、lint 越严越高分）；#22 为依赖精简（薄=高分）。

### 3.2 维度速览表

| # | 维度 | llm | llm.u | Burn | 本轮定档依据 |
| -- | --- | --- | --- | --- | --- |
| 1 | 张量/类型系统 | 10 | 2 | 10 | llm `Tensor<B,R,K>`（tensor.rs:11）双秩 + sealed 分组 + View/ViewMut/DynTensor 三桥；llm.u 扁平 Tensor/View；Burn `Tensor<const D,K>` |
| 2 | 后端广度 | 5 | 6 | 8 | llm prof 六厂商 fragment（ascend/intel/metal/musa/nvidia）+ opencl.rs FFI 岛，Executor 实现者 Metal/Cpu/Local/Vacant/Trace；llm.u feature 门控 cuda/metal/intel/blas/rocm/musa/opencl/ane；Burn CubeCL 多 runtime + Flex/Remote |
| 3 | 后端执行成熟度/性能 | 3 | 6 | 8 | llm 生产装配仅 CPU（model.rs:27），Metal impl 在位但仓内未见生产构造点，CUDA/昇腾/ROCm 无 Executor impl，真机激活未核；llm.u 多后端 GEMM/内核族厚，真机激活本轮未复跑；Burn 跨后端可跑完整算子集 |
| 4 | 算子/内核覆盖 | 5 | 4 | 9 | llm 注册表 **72 条**算子（elementwise/unary/激活/norm/attention/pool/conv/reduce/embed/rope/gemm·gemv·matmul f32+i2/i4/i8/dequant）；真机内核激活未核；llm.u 七臂内核族；Burn 张量 API 全套 + linalg/einsum/fusion |
| 5 | nn 层完整度 | 5 | 3 | 9 | llm conv/pool/norm/attn/embed/rope 齐，无 RNN；llm.u 无 conv/pool/RNN；Burn 8 大类 + 49 Config 含 RNN/Transformer |
| 6 | 自动微分 | 7 | 2 | 10 | llm 独立 tape/schedule/account/plan/replay 元数据层；本轮 backprop **50 例全绿**：Reference CPU 后端解析梯度与中心差分逐参数对拍（ABSOLUTE 2e-3 + RELATIVE 0.05），含 conv2d/GRPO/distillation KL/端到端 SGD；按 Var 自持独立快照在位；缺口：无后端装饰器式任意可微闭包、设备反向仓内通路未检出（`.cu` 探针为仓外历史材料）、分布式仅进程内 + 占位；llm.u 逐层手写 bwd；Burn 后端装饰器任意可微 + checkpoint |
| 7 | 优化器 | 4 | 3 | 9 | llm Sgd/Adam/Muon + `Hybrid<M,N>` 矩阵/非矩阵分流组合器；无 LBFGS/Adafactor/AdamW 独立族；llm.u 仅 Adam；Burn 12 个 impl + LBFGS/Momentum/累积/裁剪/衰减 |
| 8 | 调度/训练循环 | 5 | 4 | 9 | llm Session/Learner/evaluate/账表/优化步/Checkpoint/Schedule 在位，当前快照 train lib 编译绿、backprop 端到端 SGD 例在 50 例内实测通过；真机/大规模训练收敛与调度策略仍未核；llm.u Session/Trainer + Cosine；Burn Learner/early stopping/sharder/logger |
| 9 | 损失函数 | 5 | 4 | 9 | llm CE/argmax/mse/bce/huber/hinge + Distill/Grpo/RL 族（log_prob/entropy/advantage/policy_loss）并入 loss.rs；llm.u 7 fn；Burn 17 种 loss（含 CTC/RNNT/三元组/NLL） |
| 10 | 指标/评估器 | 3 | 1 | 10 | llm 仅 loss/命中/样本数账表；llm.u 仅 perplexity；Burn 30+ 指标 + evaluator/renderer |
| 11 | 数据集管线 | 3 | 1 | 9 | llm `Dataset` trait + `Loader` + `Sampling` 三族采样 + 权重校验；无物理分词/打包源；llm.u pack/shard 函数；Burn audio/nlp/vision/source/transform |
| 12 | 检查点/序列化 | 6 | 4 | 9 | llm weights/storage 自描述 + CRC-32 + Checkpoint/Schedule 恢复事实；llm.u 状态字节存档；Burn store/checkpoint/burnpack/pytorch |
| 13 | 推理服务 | 6 | 6 | 3 | llm ifb/radix/page/speculate + Httpd 薄适配器（http.rs:26/63）；llm.u 调度生命周期完整，HTTP 运行时未核；Burn 仅示例 server/remote |
| 14 | 分布式/并行 | 2 | 2 | 6 | llm 拓扑/切分/Reduce 齐；`Collective` 端口仅 `Local` 进程内镜像 + `Remote` 报错占位（旧 FileRemote 文件后端本轮已撤），all_reduce 直接 `fail!(Unsupported)`；无高速织物与多进程 spawn E2E；llm.u Comm trait + CPU 空参考；Burn WebSocket/remote + shard 规则；三家均无 NCCL 级栈 |
| 15 | 量化/精度 | 6 | 5 | 7 | llm GGML+MX+FP4+亚字节 i2/i4/i8 最广，落点权重档案侧；llm.u I2–I8+FP8/FP4+AWQ；Burn 参数级 ModuleMapper/Quantizer/Calibration；三家均无 QAT |
| 16 | 随机数/确定性 | 7 | 1 | 8 | llm 计数器 PRNG + 三分布可复现；llm.u 采样注入闭包；Burn rand/seed/Distribution |
| 17 | 内核 DSL 扩展 | 5 | 2 | 9 | llm cutile 协商/拼接/emit/指纹 + annotation 宏，P0 模板 + typed kernel 起步；native.rs 零成本转发面；llm.u 内核直写后端；Burn CubeCL IR 成熟多 runtime lowering + 融合 |
| 18 | 架构/可维护性 | 8 | 3 | 8 | llm 七层契约 + testkit 强制守分层 + ADR；train 目录本轮仍在拆分重组（geometry 多形态并存）；llm.u device/exec 耦合、unsafe 分散；Burn crate 分层清晰 + derive 宏 |
| 19 | 测试/质量门禁 | 8 | 4 | 8 | llm 2813 测试标签；当前快照实测：`cargo test --workspace` **3060 passed / 0 failed / 22 ignored**，backprop 50 绿，静态门 13/0，`fmt --check` 通过，`clippy --all-targets --all-features` 干净，`cargo deny` 四项 ok，`machete` 无未用依赖；自研门维度三仓最细——tools gate 度量 ccn/nloc/lines/call/dup/docs/词法/命名 + 棘轮 + 扫描下限防假绿，testkit 守架构边闭集/注册表镜像/入口类型/错误边界/文档引用，ci.sh 覆盖 fmt/clippy/nextest/doc/deny/machete/gate/testkit/expand/bench-regress，quality.yml 加宏展开/拼写/actionlint/Lizard/rust-code-analysis 独立复杂度报告；缺口：doc-test/expand/bench-regress 本轮未单独取证据，coverage/fuzz/miri 为可选专项非默认，测试体量小于 Burn；llm.u 1127 测试 + fuzz nightly，无同构自研静态门；Burn 6788 测试 + backend-tests 套件 + 9 个 CI workflow（GPU/valgrind/vulnerabilities），但无自研架构/静态/命名门，未设 unsafe deny |
| 20 | 生态/示例/文档 | 4 | 4 | 8 | llm 内部架构文档/ADR 完备，无对外模型与示例；llm.u README/perf 实测台账详尽；Burn book + 示例生态 |
| 21 | unsafe/内存安全边界 | 8 | 3 | 6 | llm `unsafe_code`/`unsafe_op_in_unsafe_fn` 均 deny，**94** 处显式岛；llm.u **437** 处分散，lint 仅 cli deny；Burn **246** 处、未设 deny/forbid |
| 22 | 依赖精简/供应链 | 9 | 4 | 5 | 薄=高分。llm 运行期依赖极薄、无厂商系统绑定；llm.u 携带 cudarc/metal/libloading/coreml/clblast/opencl/musa 厚系统栈；Burn 跨后端谱系最宽（serde 30/cubecl 21/wgpu 20/metal 15/num-traits 13/wgpu-msl 11/half 8/bytemuck 6/serde_json 5/ndarray 2，cudarc 经 cubecl） |

### 3.3 总分

**A. 等权视角（22 维各 1/22，通用框架口径）**

| | llm | llm.u | Burn |
| --- | --- | --- | --- |
| 合计 | 124 | 74 | 171 |
| 平均分 | **5.64** | **3.36** | **7.77** |

**B. LLM 推理/性能视角（侧重后端执行、推理服务、量化、内核 DSL；训练指标/数据集降权，权重合计 100）**

| 权重 | 维度 | llm | llm.u | Burn |
| -- | --- | --- | --- | --- |
| 15 | 推理服务 | 6 | 6 | 3 |
| 13 | 后端执行成熟度/性能 | 3 | 6 | 8 |
| 9 | 量化/精度 | 6 | 5 | 7 |
| 7 | 后端广度 | 5 | 6 | 8 |
| 7 | 算子/内核覆盖 | 5 | 4 | 9 |
| 5 | 内核 DSL 扩展 | 5 | 2 | 9 |
| 5 | 架构/可维护性 | 8 | 3 | 8 |
| 4 | nn 层完整度 | 5 | 3 | 9 |
| 4 | unsafe/内存安全边界 | 8 | 3 | 6 |
| 4 | 依赖精简/供应链 | 9 | 4 | 5 |
| 3 | 张量/类型系统 | 10 | 2 | 10 |
| 3 | 检查点/序列化 | 6 | 4 | 9 |
| 3 | 分布式/并行 | 2 | 2 | 6 |
| 3 | 随机数/确定性 | 7 | 1 | 8 |
| 3 | 测试/质量门禁 | 8 | 4 | 8 |
| 2 | 自动微分 | 7 | 2 | 10 |
| 2 | 优化器 | 4 | 3 | 9 |
| 2 | 调度/训练循环 | 5 | 4 | 9 |
| 2 | 损失函数 | 5 | 4 | 9 |
| 2 | 生态/示例/文档 | 4 | 4 | 8 |
| 1 | 指标/评估器 | 3 | 1 | 10 |
| 1 | 数据集管线 | 3 | 1 | 9 |
| | **加权总分** | **5.61** | **4.23** | **7.11** |

> 权重向量 w（dim1..22）：`{3,7,13,7,4,2,2,2,2,1,1,3,15,3,9,3,5,5,3,2,4,4}`，合计 100。合计与均值经 calculator 闭合：llm 124/22=5.64、561/100=5.61；llm.u 74/22=3.36、423/100=4.23；Burn 171/22=7.77、711/100=7.11。

## 4. 本轮校准说明（升降档逐项）

- **#8 调度/训练循环 维持 5（llm）**：Session/Learner/Checkpoint/Schedule 与端到端 SGD 例在 backprop 50 例内实测通过；当前快照 train lib 编译绿、全量 test 绿。取证中途观察到的 optimizer/geometry 断裂均属并行重构快照，已被后续收编，不计入当前基线；真机/大规模收敛与调度策略仍未核，故定 5 而非更高。
- **#14 分布式/并行 3→2（llm）**：旧 `FileRemote` 文件跨进程后端本轮已撤，`Remote` 降级为显式报错占位（all_reduce `fail!(Unsupported)`）；剩余仅进程内 `Local` 镜像算法 + 端口定义，无任何跨进程传输实现，故由 3 下调到 2。
- **#19 测试/质量门禁 上调到 8（llm）**：首轮定档误把取证中途一次性 gate 红灯（`check_slots` Ccn=7）与并行重构编译波动计入了设施成熟度。回到当前快照复核：全量 test **3060 passed / 0 failed / 22 ignored**、静态门 13/0、backprop 50 绿，且本轮补取 `fmt --check`、`clippy --all-targets --all-features`（干净）、`cargo deny`（advisories/bans/licenses/sources 四项 ok）、`machete`（无未用依赖）四道门证据。门禁设施自研深度为三仓最细：tools gate 覆盖 ccn/nloc/文件行数/调用形态/重复/文档/词法/命名 + 棘轮 + 扫描下限防假绿；testkit 守架构边闭集、注册表与注解镜像、入口类型、错误边界、文档引用；ci.sh 串联 fmt/clippy/nextest/doc/deny/machete/gate/testkit/expand/bench-regress；quality.yml 加宏展开、拼写、actionlint、Lizard/rust-code-analysis 独立复杂度报告。未到 9–10 的原因：doc-test/expand/bench-regress 本轮未单独取证据，coverage/fuzz/miri 为可选专项非默认，测试体量（2813 标签）小于 Burn（6788）。
- **#7 优化器 维持 4（llm）**：新增 `Hybrid<M,N>` 矩阵/非矩阵分流组合器，但未新增独立优化算法族（无 LBFGS/Adafactor/AdamW 独立实现），组合器只是已有两优化器的分流编排，不构成算法广度跃升。
- **#4 算子覆盖 维持 5（llm）**：注册表本轮 72 条（+1），族齐备；各算子真机内核激活仍未核、无 RNN 类算子。
- **#6 自动微分 维持 7（llm）**：CPU Reference 闭环 50 例对拍绿 + 独立快照在位；设备反向通路仍 unknown，仓外 `.cu` 材料不计仓内能力。
- **#21 unsafe 维持 8（llm）**：deny 策略不变，计数 94（−1）；新检出 `opencl.rs` 26 处受控 FFI 岛，属显式标注越界岛，不改变整体边界评分。
- **#22 依赖 维持 9（llm）**：运行期依赖谱未变，仍无厂商系统绑定。
- **llm.u / Burn 档位维持**：两仓 HEAD 与代码结构本轮未漂移；unsafe 计数微变（llm.u 437、Burn 246）不改变档位定档。Burn 等权/加权总分以本轮现场重算为准（7.77 / 7.11）。

## 5. 未覆盖与待核清单（诚实分层）

1. 三仓后端在真机（CUDA/Metal/ROCm/WGPU）的运行激活、性能与数值正确性——本轮只读代码结构，未起设备复跑。
2. llm 设备反向梯度通路：本轮仓内仍未检出设备反向核契约或 Metal/CUDA 反向派发；真机后端梯度激活、大规模训练收敛与调度策略未核；`docs/evidence/cuda-backward-probes/` 下 `.cu` 为仓外历史对拍材料，未接进仓内通道。
3. llm serve HTTP 运行时是否端到端接通——仅见 `http.rs` `Httpd` 薄适配器，未起服务进程。
4. llm.u 后端各 feature 的实际编译状态与设备覆盖；其 train/serve 端口是否接生产实现。
5. Burn 本机各后端编译/运行状态；CubeCL fusion/IR 的实际生效路径。
6. 三仓形状算子边界行为逐一对拍（广播/切片/对角/非连续布局）。
7. llm 本轮已实测门：`check --workspace`、`fmt --check`、`clippy --all-targets --all-features`、`test --workspace`（3060/0/22）、tools gate（13/0）、`cargo deny` 四项 ok、`machete` 无未用依赖、backprop 50 绿。尚未单独取证据：`cargo test --doc`、`cargo expand` 宏展开校验、`benchmarks` regress 回归；`--coverage`/`--fuzz`/`--miri` 为可选专项，本轮未跑。
8. 独立线 #44（静态门存量赤字拆分治理）：当前快照静态门 13/0 绿，中途快照观测到的 checkpoint `restore_plan`/`accum_plan`、optimizer `step`/`check_slots` 圈复杂度赤字均已闭合；后续以 `scripts/ci.sh` 完整矩阵复跑为准。

> 证据分级：标注 `confirmed` 的定位均可回链到上文 `path:symbol:line`；标 `◐`/`？` 的项为结构在位但运行激活未核，不构成"生产可用"主张。历史报告 `docs/COMPARISON-2026-11-24-reeval.md` 与 `docs/COMPARISON-llm-llmu-burn.md` 为历史参照，本轮评分向量未沿用。

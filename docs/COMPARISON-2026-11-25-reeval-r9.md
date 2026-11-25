# `llm` / `llm.u` / Burn 三仓源码级重评（2026-11-25 第九轮干净基线）

> 本轮回到现场重新取证：不沿用历史评分向量。`confirmed` 主张回链到本轮实际读取的 `path:symbol:line`；读取范围之外的运行激活状态标 `inferred` / `unknown`，不以目录名、注释、旧实现或仓外材料补齐。取证纪律见 `doubao-coding-analyze-codebase`：定义存在 ≠ 已装配 ≠ 生产激活；测试装配 ≠ 目标环境；局部机制不外推到跨组件副作用；未提交 diff 只作快照差异，不计入已提交基线。

## 0. 身份卡与证据矩阵基线

### 0.1 三仓快照（本轮实测，剔除 `target/`；llm 另剔 `.tmp/`、`.claude/worktrees/`）

| 仓库 | revision | worktree 状态 | 规模（源码实测） | `#[test]`/`#[tokio::test]` 标签 | unsafe 计数 |
| --- | --- | --- | --- | --- | --- |
| llm（新） | HEAD `b410d1e0` 2026-11-25 22:50:57 +0800 | **干净（dirty 0）** | 21 members，563 `.rs` / 165,134 行 | 2794 | 93 |
| llm.u | `cf4e8fe3` 2026-11-11 04:55:22 +0800 | 170 项未提交；结论以 HEAD 已提交代码为准，脏改动仅作快照差异 | 349 `.rs` / 104,592 行（15 crate） | 1127 | 437 |
| Burn | `c81a9042` 2026-11-22 13:06:56 -0500 | 干净（零改动） | 1989 `.rs` / 490,469 行（42 库 crate；examples/xtask 不计） | 6788 | 246 |

### 0.2 llm 本轮门禁实测（干净 HEAD 基线）

| 门 | 命令 | 结果 |
| --- | --- | --- |
| 编译 | `cargo check --workspace` | `Finished`，无 error |
| 全量测试 | `cargo test --workspace` | **3069 passed / 0 failed / 22 ignored**，无 test binary 编译失败 |
| 反向传播 | `cargo test -p llm-train backprop` | **50 passed / 0 failed**（203 filtered） |
| 静态门 | `cargo test -p llm-tools --test gate` | **13 passed / 0 failed** |
| 架构/契约测试 | `cargo test -p llm-testkit --lib` | **117 passed / 0 failed** |
| 格式 | `cargo fmt --all --check` | exit 0 |
| Clippy | `cargo clippy --workspace --all-targets --all-features --locked` | 干净 `Finished`，无 error/warning |
| 供应链 | `cargo deny check` | advisories / bans / licenses / sources 四项 ok |
| 未用依赖 | `cargo machete` | 未发现未用依赖 |

> 本轮未单独取证据：`cargo test --doc`、`cargo expand` 宏展开校验、`benchmarks` regress 回归；`--coverage` / `--fuzz` / `--miri` 为可选专项，本轮未跑。llm.u 170 项未提交改动不纳入能力裁决；Burn 本机后端运行状态本轮未在本机复跑。

### 0.3 证据矩阵（llm，本轮现场定位 → 观测 → 范围 → 状态）

| 阶段 | literal locator | 观测 | 范围 | 状态 |
| --- | --- | --- | --- | --- |
| 逻辑张量类型 | `crates/tensor/src/tensor.rs:11` `Tensor<B,R,K>`；`:22` `View`；`:25` `ViewMut`；`:28` `DynTensor`；`:177` `into_slice_mut` | 存储 `B` / 秩 `R` / 元素 `K` 三位泛型；三视图桥为存储别名；`into_slice_mut` 全仓仅定义处一处、无生产调用方（去留属 API 口径决策） | 类型定义与静态关系 | confirmed |
| 生产执行器装配 | `crates/serve/src/model.rs:19` `CpuModel<E>`；`:24` `impl CpuModel<CpuExecutor>`；`:27` `CpuExecutor::new(Device::new("cpu","generic"))` | serving 路径选 CPU 执行器；其余设备选择未在本仓生产入口见到 | 该装配点 confirmed；其余设备 inferred | confirmed（CPU 装配）/ inferred（其余设备） |
| Executor 实现者全集 | `prof/src/exec.rs:220` `trait Executor`；`engine/src/queue.rs:10` 同名 `trait Executor`；`prof/src/metal/runtime.rs:284` `MetalExecutor`、`:409` `impl Executor for MetalExecutor`；`engine/src/cpu.rs:91` `CpuExecutor`、`:237` `impl Executor for CpuExecutor`；`queue.rs:43` `Trace<E>`、`:107` `LocalExecutor`、`:185` `VacantExecutor` | 生产/参考实现为 CPU/Local/Vacant/Trace 装饰器；Metal 定义在位，仓内未见生产构造点；两处 Executor trait 分属 prof 运行时缝与 engine 队列缝 | 实现者枚举 confirmed；Metal 生产接线未见到 | confirmed（定义/装配）/ inferred（Metal 生产未接） |
| CUDA / 昇腾 / ROCm | `crates/` 全量检索；未发现对应 `Executor` impl | 仓内无对应执行器实现；prof 下 `ascend`/`intel`/`metal`/`musa`/`nvidia` 为源生成/档案 fragment，`opencl.rs` 为受控 FFI 岛 | 仓内无对应执行器实现 | confirmed（搜索范围 `crates/` 全量） |
| 自动微分 | `autograd/src/graph.rs:96` `Var`；`tape.rs:76` `record`、`:84` `record_with`、`:205` `backward_order`、`:210` `backward`；`session.rs:332` `gradient_snapshot`；`train/tests/backprop_tests.rs` | tape/ordering/schedule 元数据层；`Session<Reference,Sgd>` 在 CPU 参考 Kernel 上建图→plan→backward，解析梯度与中心差分逐参数对拍；本轮 backprop **50/0**；独立梯度快照按 `Var` 复制。设备反向：仓内未检出设备反向梯度核契约或 Metal/CUDA 反向派发；`docs/evidence/cuda-backward-probes/` 下 `.cu` 为仓外历史材料，未接进仓内通道 | CPU Reference 闭环 + 快照 confirmed；设备反向 unknown；仓外探针不计仓内能力 | confirmed（CPU + 快照）/ unknown（设备反向） |
| 优化器 | `train/src/optimizer.rs:12` `Group`；`:83` `trait Optimizer`；`:85` `step`；`:178` 宏展开 `step`；`:297` `Sgd`；`:327` `Adam`；`:403` `Muon`；`:538` `Hybrid<M,N>`；`:587` `Hybrid::step` | Sgd / Adam（注释 AdamW：偏差修正双矩与解耦衰减）/ Muon（Newton–Schulz 正交化）+ `Hybrid` 矩阵/非矩阵分流组合器；`MomentState`/`Partitioned`/`DecayClip` 在位；无 LBFGS/Adafactor/RMSprop/Adagrad 独立族 | 定义与结构 confirmed | confirmed |
| 训练数据集 | `dataset.rs:49` `from_config`；`:196` `trait Dataset<'a>`；`:218` `Loader<'a,T>`；`rng.rs:55` `block`、`:84` `with_stream`、`:96` `next_u32`、`:115` `next_f32` | 顺序/洗牌/加权三族采样 + 权重校验；物理分词/打包源仍缺位 | 结构 confirmed；生产数据链路 inferred | confirmed（结构）/ inferred（生产链路） |
| 算子注册表 | `nn/src/registry.rs:17` `for_each_operator!`；清单展开 **72 条**；`:99` `REGISTRY`；二分查表 | 注册表驱动算子面；编译期名字/顺序双断言 | 注册表与算子名 confirmed；各算子真机内核激活未核 | confirmed（注册表）/ unknown（真机内核） |
| 损失与 RL | `loss.rs:134` `cross_entropy`、`:144` `argmax_hits`、`:171` `mse`、`:179` `bce_with_logits`、`:186` `huber`、`:202` `hinge`、`:243` `log_prob`、`:258` `entropy`、`:274` `advantage`、`:294` `policy_loss`、`:307` `Grpo`、`:399` `grpo_loss`、`:524` `Distill` | RL 族并入单文件 `loss.rs` | 函数定义 confirmed | confirmed |
| 检查点 / LR 调度 | `checkpoint.rs:18` `Step`、`:39` `Checkpoint`、`:152` `Schedule`；`session.rs:318` `accum_snapshot`、`:332` `gradient_snapshot`、`:346` `ties_snapshot`、`:351` `zero_gradients` | 独立 checkpoint 文件承载 step 计数/权重存档/学习率退火；快照按 Var 复制 | 结构 confirmed | confirmed |
| 循环层 | `nn/src/recurrent.rs:38` `Cell`（`:43` `gates`、`:60` `name`、`:72` `activations`）；`:97` `RecurrentTensor`；`:134` `Recurrent`；`:145`/`:359` impl；外置 `nn/tests/recurrent_tests.rs` | RNN/GRU/LSTM 三单元共用一份声明；权重名对齐 PyTorch `lstm.weight_ih_l0`；测试围绕参数字节/偏置/方向张量/形状堆叠/双向下界。**无前向递归内核（无 forward/step），未进 `registry.rs`** | 声明 + 权重契约 + 几何下界 confirmed；前向内核缺口 confirmed | confirmed（声明骨架）/ unknown（前向内核） |
| 推理服务 | `serve/src/generate.rs:9` `trait Predict`（`logits`）；`serve/src/model.rs:86` `forward`、`:123` `predict_input_batch`、`:294` `forward_block_batch`、`:312` `forward_at`、`:327` `forward_batch`；`http.rs:26` `Httpd`、`:64`/`:73` `TcpListener::bind`、`:86` `serve_once`、`:96` `serve`；`lib.rs:66` 再导出 `Httpd` | admit/assembly/block/plan/speculate/completion/endpoint 调度族在位；`Httpd` 真实 listener 绑定与 serve 方法在位。**仓内未检出 `impl Predict for CpuModel`；cli 生产入口未见起服务进程** | 调度与适配器结构 confirmed；trait 接线与服务进程激活 unknown | confirmed（结构）/ unknown（契约接线/生产进程） |
| 分布式 | `shard/src/collective.rs:72` `trait Collective<T>`；`:74` `all_reduce`、`:77` `all_gather`、`:86` `broadcast`、`:95` `reduce_scatter`；`:202` `Remote`、`:204` `impl Collective for Remote`（`:205` 报错占位）；`:327` `Local`；`:426` `FileRemote`、`:437` `ReplicaStore<f32>`；`:460`–`:472` 泛型 `impl<T,S> Collective<T> for S where S: ReplicaStore`；`device.rs:379` `FileRemote`、`:387` `open`；`tests/collective_tests.rs` 24 例 | `Local` 进程内镜像；`FileRemote` 文件跨秩后端本轮回归，双句柄同进程写同一临时文件对拍（reduce / all-gather / broadcast / 越界拒绝三例），四算子共享同一组泛型算法；`Remote` 为显式 `fail!(Unsupported)` 占位。无 NCCL/RDMA/socket 高速织物；未做真 OS 多进程 spawn E2E | 端口 + Local + FileRemote 实现 confirmed；高速织物 / 多进程 E2E unknown | confirmed（端口 + 两实现）/ unknown（织物 / E2E） |
| 内核 DSL | `cutile/src/` 20 模块：`ast/parser/builtin/cache/cap/dirs/emit/expr/fuse/harness/kernel/lru/pass/plan/printer/probe/reference/space/substitute/validate`；`cutile/src/lib.rs:75` 再导出 `dsl`、`:79` `atom`、`:80` `prof`、`:85` `decl`、`:86` `layout`、`:87` `model/sch/sig` | P0 模板纵切面 + typed kernel 起步；旧 `native.rs` 零成本转发面本轮已撤；DSL 未打通生产内核编译 | 静态结构 confirmed；真机内核激活 unknown | confirmed（结构）/ unknown（真机） |
| 量化 / 精度 | `weights/src/quant.rs:347` `I2`、`:354` `I3`、`:361` `I4`、`:368` `I8`、`:625` `LADDER`；`DType::U4/U8/I8`；容器 `gguf/safetensors/torch/json/archive/ledger/seg` | 亚整数四档位 + 权重容器格式；落点权重档案侧；**本轮 `weights/` `nn/` 仓内未命中 FP4/FP8/MX（mxfloat）符号**；无 QAT | 亚整数档位 + 容器 confirmed；浮点低精度未核到 | confirmed（整数档位/容器）/ unknown（FP4/FP8/MX） |
| unsafe 与内存安全 | workspace lints：`unsafe_code="deny"`、`unsafe_op_in_unsafe_fn="deny"`、`missing_docs`/`unused_lifetimes="deny"`；本轮计数 **93** | 分布 `prof/src/metal/runtime.rs`(37)、`prof/src/opencl.rs`(26，`#![allow(unsafe_code)]` 受控 FFI/RAII 岛)、`cli/tests/audit.rs`(7)、`prof/src/sign.rs`(6)、`prof/src/runtime.rs`(6)、`storage/src/mmap.rs`(5)、`nn/src/fast.rs`(4)、`nn/src/pool.rs`(2) | 策略与分布 confirmed；逐块正确性未审计 | confirmed |
| 外部依赖 / 供应链 | 运行期外部 `half`(3)、`num-traits`(2)；`bytemuck`(2)/`serde`(3)/`serde_json`(2)/`toml`(7) optional；`proc-macro2/quote/syn` 编译期宏；`proptest/criterion/similar-asserts` 测试/bench；`libfuzzer-sys` optional fuzz | 无 cudarc/metal-rs/libloading/wgpu/coreml/ndarray/opencl-sys 运行期绑定；Metal/OpenCL 为手写 FFI，未引包装 crate | 依赖清单 confirmed | confirmed |

> 未覆盖范围（本轮未逐项深核）：三仓真机后端（CUDA/Metal/ROCm/WGPU）运行激活与性能；llm serve HTTP 端到端接通；llm.u 各 feature 实际编译状态与设备覆盖；Burn 本机后端与 CubeCL fusion/IR 生效路径；三仓形状算子边界行为逐一对拍；`benchmarks` 数值结果。下文只给代码结构层面定级。

## 1. 本轮现场结构刷新（相对第八轮干净基线）

1. **基线由并行重构中途快照转为干净 HEAD**：`b410d1e0` dirty 0；全量测试 **3069/0/22**（上轮快照 3060/0/22），testkit 117/0 单独复核；规模 563 `.rs` / 165,134 行 / 2794 标签 / unsafe 93。

2. **分布式文件后端回归**：第八轮中途快照观察到 `FileRemote` 被撤、`Remote` 占位；本轮现场 `FileRemote` 文件跨秩后端在位（`collective.rs:426/:437` + `device.rs:379/:387`），`ReplicaStore` 四算子 site 常量齐备，Local/FileRemote 共享泛型归约/gather/broadcast/scatter 算法，测试含双句柄跨秩 reduce、all-gather/broadcast 与越界拒绝。

3. **循环层补齐声明骨架**：`recurrent.rs`（375 行）承载 RNN/GRU/LSTM 三单元的门数/激活次数/权重名映射/几何下界；前向递归内核未实现、未进注册表，测试只覆盖参数与形状契约。

4. **serving 契约接线复核**：`Predict` trait 在 `generate.rs:9` 定义（`logits`），`CpuModel` 侧为自有 `forward`/`predict_input_batch`/`forward_block_batch` 方法族；本轮未检出 `impl Predict for CpuModel`。旧报告 `impl Predict` 锚点在当前基线已不存在。

5. **cutile 重组**：旧 `native.rs` 零成本转发面移除，DSL 拆为 20 个职责模块（AST/parser/builtin/cache/cap/emit/expr/fuse/harness/kernel/lru/pass/plan/printer/probe/reference/space/substitute/validate）；仍未进入生产内核编译路径。

6. **量化口径收紧**：本轮现场只确认亚整数 `I2/I3/I4/I8` 档位与多容器格式；FP4/FP8/MX 符号在 `weights/` `nn/` 仓内未命中，不计入仓内量化能力。

## 2. 规模与工程度量（本轮实测）

### 2.1 体量对比

| 指标 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| crate 数 | 21 | 15 | 42 |
| `.rs` 文件 | 563 | 349 | 1989 |
| 源码行 | 165,134 | 104,592 | 490,469 |
| 测试标签 | 2794 | 1127 | 6788 |

### 2.2 llm 生产行 Top crate（行，当前磁盘实测）

| crate | 行 | 职责 |
| --- | --- | --- |
| nn | 24,253 | 算子内核 / 层声明 / 性能下界 |
| weights | 15,911 | 权重格式、PTQ、codec、存档 |
| prof | 15,204 | 能力档案、设备片段、执行缝与 native 微核 |
| train | 13,195 | 训练循环状态、loss、优化器、检查点、数据集、几何 |
| cutile | 12,852 | 内核 DSL：AST/emit/fuse/pass/plan |
| shard | 11,521 | 拓扑 / 流水线 / 切分 / 见证 / 集合通信端口 |
| algorithm | 10,372 | 注意力 / SSM / 记忆 / 路由变体语义 |
| serve | 10,216 | 请求文法、采样、补全端点、路由 |
| cli | 5,999 | 组合根与二进制 |
| testkit | 5,955 | 分层 / 账本 / 所有权架构守卫 |
| annotation | 5,801 | 契约宏 |
| engine | 4,759 | serving 状态机、CPU 执行器、队列 |

### 2.3 unsafe 边界（本轮计数）

* **llm**：workspace 默认 `unsafe_code = "deny"`；93 处 unsafe，集中为显式岛（Metal objc FFI、OpenCL 受控 FFI、native 微核、mmap、fast kernel、审计测试）。
* **llm.u**：437 处；`device/engine/exec/tool` 源码层 `#![forbid(unsafe_code)]`，`cli`/`benches` Cargo deny，其余分 crate 未统一；本轮 0 个 Cargo.toml 设 forbid/deny `unsafe_code`。
* **Burn**：246 处；按后端 crate 各自管理 FFI，未检索到 `forbid/deny unsafe_code`。
* 计数口径：源码 `unsafe {`/`unsafe fn`/`unsafe impl`/extern unsafe，剔除所有层级 `target/`、`.tmp/` 与 fuzz 生成 codegen。

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

> 分数是当前代码快照的相对成熟度，不是能力上限；三仓定位不同（Burn 通用训练框架 / llm.u LLM 推理多后端执行 / llm 分层重构参考栈），总分须结合权重视角解读。后端真机运行激活与设备反向通路本轮未核，相关维度按代码结构定档。#21 为 unsafe 边界（越界岛越少、lint 越严越高分）；#22 为依赖精简（薄 = 高分）。

### 3.2 维度速览表

| # | 维度 | llm | llm.u | Burn | 本轮定档依据 |
| --- | --- | --- | --- | --- | --- |
| 1 | 张量 / 类型系统 | 10 | 2 | 10 | llm `Tensor<B,R,K>`（tensor.rs:11）双秩 + sealed 分组 + View/ViewMut/DynTensor 三桥；llm.u 扁平 Tensor/View；Burn `Tensor<const D,K>` |
| 2 | 后端广度 | 5 | 6 | 8 | llm prof 六厂商 fragment（ascend/intel/metal/musa/nvidia）+ opencl.rs FFI 岛，Executor 实现 Metal/Cpu/Local/Vacant/Trace；llm.u feature 门控 cuda/metal/intel/blas/rocm/musa/opencl/ane；Burn CubeCL 多 runtime + Flex/Remote |
| 3 | 后端执行成熟度 / 性能 | 3 | 6 | 8 | llm 生产装配仅 CPU（model.rs:27），Metal impl 在位但仓内未见生产构造点，CUDA/昇腾/ROCm 无 Executor impl，真机激活未核；llm.u 多后端 GEMM/内核族厚，真机激活本轮未复跑；Burn 跨后端可跑完整算子集 |
| 4 | 算子 / 内核覆盖 | 5 | 4 | 9 | llm 注册表 **72 条**算子；真机内核激活未核；llm.u 七臂内核族；Burn 张量 API 全套 + linalg/einsum/fusion |
| 5 | nn 层完整度 | 5 | 3 | 9 | llm conv/pool/norm/attn/embed/rope/deconv + recurrent 声明骨架（RNN/GRU/LSTM 权重契约与几何下界在位，前向内核缺、未注册）；llm.u 无 conv/pool/RNN；Burn 8 大类 + 49 Config 含 RNN/Transformer |
| 6 | 自动微分 | 7 | 2 | 10 | llm 独立 tape/schedule/account/plan/replay 元数据层；本轮 backprop **50 例全绿**：Reference CPU 后端解析梯度与中心差分逐参数对拍，按 Var 自持独立快照在位；缺口：无后端装饰器式任意可微闭包、设备反向仓内通路未检出（`.cu` 探针为仓外历史材料）；llm.u 逐层手写 bwd；Burn 后端装饰器任意可微 + checkpoint |
| 7 | 优化器 | 4 | 3 | 9 | llm Sgd/Adam(AdamW)/Muon + `Hybrid<M,N>` 矩阵/非矩阵分流组合器；无 LBFGS/Adafactor/RMSprop/Adagrad 独立族；llm.u 仅 Adam；Burn 12 个 impl + LBFGS/Momentum/累积/裁剪/衰减 |
| 8 | 调度 / 训练循环 | 5 | 4 | 9 | llm Session/Learner/evaluate/账表/优化步/Checkpoint/Schedule 在位，全量 test 绿、backprop 端到端 SGD 例在 50 例内实测通过；真机/大规模训练收敛与调度策略仍未核；llm.u Session/Trainer + Cosine；Burn Learner/early stopping/sharder/logger |
| 9 | 损失函数 | 5 | 4 | 9 | llm CE/argmax/mse/bce/huber/hinge + Distill/Grpo/RL 族（log_prob/entropy/advantage/policy_loss）并入 loss.rs；llm.u 7 fn；Burn 17 种 loss（含 CTC/RNNT/三元组/NLL） |
| 10 | 指标 / 评估器 | 3 | 1 | 10 | llm 仅 loss/命中/样本数账表；llm.u 仅 perplexity；Burn 30+ 指标 + evaluator/renderer |
| 11 | 数据集管线 | 3 | 1 | 9 | llm `Dataset` trait + `Loader` + `Sampling` 三族采样 + 权重校验；无物理分词/打包源；llm.u pack/shard 函数；Burn audio/nlp/vision/source/transform |
| 12 | 检查点 / 序列化 | 6 | 4 | 9 | llm weights/storage 自描述 + CRC-32 + Checkpoint/Schedule 恢复事实；llm.u 状态字节存档；Burn store/checkpoint/burnpack/pytorch |
| 13 | 推理服务 | 6 | 6 | 3 | llm ifb 类 admit/block/assembly/plan + speculate + Httpd（真实 bind/serve/serve_once）；`Predict` trait 定义在位、CpuModel 未 impl，服务进程未起；llm.u 调度生命周期完整，HTTP 运行时未核；Burn 仅示例 server/remote |
| 14 | 分布式 / 并行 | 3 | 2 | 6 | llm 拓扑/切分/Reduce 齐；`Collective` 端口含 `Local` 进程内镜像 + `FileRemote` 文件双句柄跨秩对拍（24 测试）+ `Remote` 报错占位；无高速织物与多进程 spawn E2E；llm.u Comm trait + CPU 空参考；Burn WebSocket/remote + shard 规则；三家均无 NCCL 级栈 |
| 15 | 量化 / 精度 | 6 | 5 | 7 | llm 亚整数 I2/I3/I4/I8 四档位 + GGUF/SAFE/torch 容器，落点权重档案侧；本轮仓内未核到 FP4/FP8/MX；llm.u I2–I8+FP8/FP4+AWQ；Burn 参数级 ModuleMapper/Quantizer/Calibration；三家均无 QAT |
| 16 | 随机数 / 确定性 | 7 | 1 | 8 | llm 计数器 PRNG + 三分布可复现；llm.u 采样注入闭包；Burn rand/seed/Distribution |
| 17 | 内核 DSL 扩展 | 5 | 2 | 9 | llm cutile 20 模块 AST/parser/emit/fuse/pass/plan/printer/validate + annotation 宏，P0 模板 + typed kernel 起步；未接生产编译；llm.u 内核直写后端；Burn CubeCL IR 成熟多 runtime lowering + 融合 |
| 18 | 架构 / 可维护性 | 8 | 3 | 8 | llm 七层契约 + testkit 强制守分层 + ADR；train 目录本轮继续收口；llm.u device/exec 耦合、unsafe 分散；Burn crate 分层清晰 + derive 宏 |
| 19 | 测试 / 质量门禁 | 8 | 4 | 8 | llm 2794 标签；本轮实测 **3069/0/22**，backprop 50 绿，静态门 13/0，testkit 117/0，fmt 通过，clippy 干净，deny 四项 ok，machete 无未用依赖；自研门维度三仓最细（ccn/nloc/lines/call/dup/docs/词法/命名 + 棘轮 + 扫描下限防假绿；testkit 守架构边闭集/注册表镜像/入口类型/错误边界/文档引用）；缺口：doc-test/expand/bench-regress 未单独取证，coverage/fuzz/miri 可选非默认，体量小于 Burn；llm.u 1127 测试 + fuzz nightly，无同构自研静态门；Burn 6788 测试 + backend-tests 套件 + 9 个 CI workflow（GPU/valgrind/vulnerabilities），但无自研架构/静态/命名门，未设 unsafe deny |
| 20 | 生态 / 示例 / 文档 | 4 | 4 | 8 | llm 内部架构文档/ADR 完备，无对外模型与示例；llm.u README/perf 实测台账详尽；Burn book + 示例生态 |
| 21 | unsafe / 内存安全边界 | 8 | 3 | 6 | llm `unsafe_code`/`unsafe_op_in_unsafe_fn` 均 deny，**93** 处显式岛；llm.u **437** 处分散，lint 仅 cli deny；Burn **246** 处、未设 deny/forbid |
| 22 | 依赖精简 / 供应链 | 9 | 4 | 5 | 薄 = 高分。llm 运行期依赖极薄、无厂商系统绑定；llm.u 携带 cudarc/metal/libloading/coreml/clblast/opencl/musa 厚系统栈；Burn 跨后端谱系最宽（serde 30/cubecl 21/wgpu 20/metal 15/num-traits 13/wgpu-msl 11/half 8/bytemuck 6/serde_json 5/ndarray 2，cudarc 经 cubecl） |

### 3.3 总分

**A. 等权视角（22 维各 1/22，通用框架口径）**

| | llm | llm.u | Burn |
| --- | --- | --- | --- |
| 合计 | 125 | 74 | 171 |
| 平均分 | **5.68** | **3.36** | **7.77** |

**B. LLM 推理 / 性能视角（侧重后端执行、推理服务、量化、内核 DSL；训练指标/数据集降权，权重合计 100）**

| 权重 | 维度 | llm | llm.u | Burn |
| --- | --- | --- | --- | --- |
| 15 | 推理服务 | 6 | 6 | 3 |
| 13 | 后端执行成熟度 / 性能 | 3 | 6 | 8 |
| 9 | 量化 / 精度 | 6 | 5 | 7 |
| 7 | 后端广度 | 5 | 6 | 8 |
| 7 | 算子 / 内核覆盖 | 5 | 4 | 9 |
| 5 | 内核 DSL 扩展 | 5 | 2 | 9 |
| 5 | 架构 / 可维护性 | 8 | 3 | 8 |
| 4 | nn 层完整度 | 5 | 3 | 9 |
| 4 | unsafe / 内存安全边界 | 8 | 3 | 6 |
| 4 | 依赖精简 / 供应链 | 9 | 4 | 5 |
| 3 | 张量 / 类型系统 | 10 | 2 | 10 |
| 3 | 检查点 / 序列化 | 6 | 4 | 9 |
| 3 | 分布式 / 并行 | 3 | 2 | 6 |
| 3 | 随机数 / 确定性 | 7 | 1 | 8 |
| 3 | 测试 / 质量门禁 | 8 | 4 | 8 |
| 2 | 自动微分 | 7 | 2 | 10 |
| 2 | 优化器 | 4 | 3 | 9 |
| 2 | 调度 / 训练循环 | 5 | 4 | 9 |
| 2 | 损失函数 | 5 | 4 | 9 |
| 2 | 生态 / 示例 / 文档 | 4 | 4 | 8 |
| 1 | 指标 / 评估器 | 3 | 1 | 10 |
| 1 | 数据集管线 | 3 | 1 | 9 |
| | **加权总分** | **5.64** | **4.23** | **7.11** |

> 权重向量 w（dim1..22）：`{3,7,13,7,4,2,2,2,2,1,1,3,15,3,9,3,5,5,3,2,4,4}`，合计 100。合计与均值经 calculator 闭合：llm 125/22=5.68、564/100=5.64；llm.u 74/22=3.36、423/100=4.23；Burn 171/22=7.77、711/100=7.11。

## 4. 本轮校准说明（升降档逐项）

* **#14 分布式 / 并行 2→3（llm）**：本轮干净基线现场复核，`FileRemote` 文件跨秩后端回归（`collective.rs:426/:437` + `device.rs:379/:387`），`ReplicaStore` 四算子 site 常量齐备，Local/FileRemote 共享泛型归约/gather/broadcast/scatter 算法；测试含双句柄同进程跨秩 reduce、all-gather/broadcast 与 world/rank/宽度越界拒绝。仍无 NCCL/RDMA/socket 高速织物，未做真 OS 多进程 spawn E2E，故定 3 而非更高。

* **#5 nn 层完整度 维持 5（llm）**：本轮补齐循环层声明骨架（`recurrent.rs:375`，RNN/GRU/LSTM 门数、激活次数、PyTorch 权重名映射、几何下界），但前向递归内核未实现、未进注册表，测试只覆盖参数与形状契约；广度声明补齐不改内核深度，档位维持 5。

* **#13 推理服务 维持 6（llm）**：`Httpd` 具备真实 listener 绑定与 `serve`/`serve_once` 方法；`Predict` trait（`generate.rs:9`）定义在位，但仓内未检出 `impl Predict for CpuModel`，cli 生产入口未见起服务进程；契约接线与运行激活为缺口，维持 6。

* **#15 量化 / 精度 维持 6（llm）**：现场只核到亚整数 `I2/I3/I4/I8` 四档位与多权重容器；FP4/FP8/MX 符号在 `weights/` `nn/` 仓内未命中，不计入已确认能力；落点仍在权重档案侧、无 QAT，维持 6。

* **#17 内核 DSL 维持 5（llm）**：cutile 重组为 20 模块，旧 `native.rs` 转发面移除；DSL 仍未打通生产内核编译，维持 5。

* **#6 自动微分 维持 7（llm）**：CPU Reference 闭环 50 例对拍绿 + 独立快照在位；设备反向通路仍 unknown，仓外 `.cu` 材料不计仓内能力。

* **#7 优化器 维持 4（llm）**：`Hybrid<M,N>` 分流组合器在位，但无独立优化算法族（无 LBFGS/Adafactor/RMSprop/Adagrad），组合器只是已有两优化器的编排，不构成算法广度跃升。

* **#19 测试 / 质量门禁 维持 8（llm）**：本轮干净基线实测 3069/0/22、gate 13/0、testkit 117/0，fmt/clippy/deny/machete 全绿；自研门深度三仓最细。未到 9–10：doc-test/expand/bench-regress 未单独取证，coverage/fuzz/miri 可选非默认，测试体量小于 Burn。

* **#21 unsafe 维持 8（llm）**：deny 策略不变，计数 93；Metal/OpenCL 属显式标注越界岛，不改变整体边界评分。

* **#22 依赖 维持 9（llm）**：运行期依赖谱未变，仍无厂商系统绑定。

* **llm.u / Burn 档位维持**：两仓 HEAD 本轮未漂移（`cf4e8fe3` / `c81a9042`）；本轮重取规模、依赖谱系、unsafe 计数与 workflow 数，与既有档位一致。Burn 等权/加权总分以现场重算为准（7.77 / 7.11）。

## 5. 未覆盖与待核清单（诚实分层）

1. 三仓后端在真机（CUDA/Metal/ROCm/WGPU）的运行激活、性能与数值正确性 —— 本轮只读代码结构，未起设备复跑。

2. llm 设备反向梯度通路：本轮仓内仍未检出设备反向核契约或 Metal/CUDA 反向派发；真机后端梯度激活、大规模训练收敛与调度策略未核；`docs/evidence/cuda-backward-probes/` 下 `.cu` 为仓外历史对拍材料，未接进仓内通道。

3. llm serve HTTP 运行时是否端到端接通 —— 仅见 `http.rs` `Httpd` listener/serve 方法，未起服务进程；`CpuModel` 未实现 `Predict` trait。

4. llm.u 后端各 feature 的实际编译状态与设备覆盖；其 train/serve 端口是否接生产实现。

5. Burn 本机各后端编译/运行状态；CubeCL fusion/IR 的实际生效路径。

6. 三仓形状算子边界行为逐一对拍（广播/切片/对角/非连续布局）。

7. llm 本轮已实测门：`check --workspace`、`fmt --check`、`clippy --all-targets --all-features`、`test --workspace`（3069/0/22）、tools gate（13/0）、testkit（117/0）、`cargo deny` 四项 ok、`machete` 无未用依赖、backprop 50 绿。尚未单独取证据：`cargo test --doc`、`cargo expand` 宏展开校验、`benchmarks` regress 回归；`--coverage`/`--fuzz`/`--miri` 为可选专项，本轮未跑。

8. 独立线 #44（静态门存量赤字拆分治理）：当前基线静态门 13/0 绿；后续以 `scripts/ci.sh` 完整矩阵复跑为准。

> 证据分级：标注 `confirmed` 的定位均可回链到上文 `path:symbol:line`；标 `inferred`/`unknown` 的项为结构在位但运行激活未核，不构成"生产可用"主张。历史报告 `COMPARISON-2026-11-25-reeval.md`（第八轮）、`COMPARISON-2026-11-24-reeval.md`、`COMPARISON-llm-llmu-burn.md` 为历史参照，本轮评分向量未沿用。

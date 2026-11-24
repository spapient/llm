# `llm` / `llm.u` / Burn 三仓源码级重评（2026-11-24 快照）

> 本轮为**重新取证**：不沿用历史评分向量，所有 `confirmed` 主张回链到本轮实际读取的 `path:symbol:line`；读取范围之外的运行激活状态标 `inferred` / `unknown`，不以目录名、注释或旧实现补齐。
>
> 取证纪律见 `doubao-coding-analyze-codebase`：定义存在 ≠ 已装配 ≠ 生产激活；测试装配 ≠ 目标环境；局部机制不外推到跨组件副作用。

## 0. 身份卡与证据矩阵基线

| 仓库 | revision | worktree 状态 | 规模（源码实测，剔除 `target`） | 测试数（`#[test]`/`#[tokio::test]` 计数） |
| --- | --- | --- | --- | --- |
| llm（新） | `64233760` 2026-11-24 15:00 +0800（取证区间 HEAD 被并行工作流连续推进，cf81efb8→64233760） | 工作区 5 项 `M`：代码侧仅 `autograd/src/schedule.rs`、`autograd/src/tape.rs` 两文件小改（未提交，不计基线）；其余为文档（含本报告、ADR、review 笔记）。上一轮 `cos/sin/sqrt` unary 增量已收进 HEAD 基线 | 净口径（剔 `target` 与 `.claude/worktrees` 副本）22 crate，690 `.rs`，167,119 行 | 2787 |
| llm.u | `cf4e8fe3` 2026-11-11 04:55 +0800 | 170 项未提交；结论以 HEAD 已提交代码为准，脏改动仅作快照差异 | 15 crate，344 `.rs`，104,592 行（剔所有层级 `target/` 与 fuzz 生成 codegen） | 1127 |
| Burn | `c81a9042` 2026-11-22 13:06 -0500 | 干净（零改动） | 42 crate dirs，1989 `.rs`，490,469 行 | 6788 |

> 快照漂移说明：llm 工作区存在外部自动提交/并行工作流，取证期间 HEAD 持续推进、未提交文件被周期收编；定位锚点（`path:symbol:line`）均在复核时回链命中。规模统计统一剔除 `target`；llm 另需排除 `.claude/worktrees/` 并行副本目录。llm.u 的 170 项未提交改动未纳入能力裁决。编译观测曾受增量缓存陈旧误报影响，复核前 `cargo clean -p <crate>`；结论分层见 §13.1、§16。

**证据矩阵（本轮点名阶段 → 定位 → 观测到的分支/被调实现 → 主张范围 → 状态）**

| 阶段 | literal locator（llm） | 观测 | 范围 | 状态 |
| --- | --- | --- | --- | --- |
| 逻辑张量类型 | `crates/tensor/src/tensor.rs:38` `Tensor<B,R,K>`；`:49` `View`；`:52` `ViewMut`；`:59` `DynTensor`；`lib.rs:96` 再导出 | 存储 `B` / 秩 `R` / 元素 `K` 三位泛型；`View`/`ViewMut`/`DynTensor` 均为 `Tensor` 的存储别名；元素 sealed 分组收紧算子 bound | 类型定义与静态关系 | confirmed |
| 元素能力分组 | `crates/tensor/src/lib.rs:129-149` `Element/NumericElement/OrderedElement/FloatElement/IntElement/BoolElement` | sealed 分组，算子族把 `Elem` bound 收紧到对应分组 | trait 定义 | confirmed |
| 生产执行器装配 | `crates/serve/src/model.rs:19` `CpuModel<E>`、`:24` `CpuModel<CpuExecutor>`、`:27` `CpuExecutor::new(Device::new("cpu","generic"))`、`:453` `impl Predict` | serving 路径选 CPU 执行器 | 该装配点 confirmed；其余设备选择未在本仓生产入口见到 | confirmed |
| Executor 实现者全集 | `device/src/executor.rs:404` `MetalExecutor`；`engine/src/cpu.rs:242` `CpuExecutor`；`engine/src/queue.rs:73` `Trace<E>`、`:212` `LocalExecutor`、`:276` `VacantExecutor`；`engine/tests/exec.rs:11` `Counting`(test) | 生产/参考实现为 CPU/Local/Vacant/Trace 装饰器；Metal 定义在位、构造仅见测试；Counting 为测试桩 | 实现者枚举 confirmed；Metal 生产接线未见到 | confirmed（定义/装配）/ inferred（Metal 生产未接） |
| CUDA / 昇腾 | `crates/device/src/` 全量检索；未发现 CUDA/昇腾的 `Executor` impl | 仓内无对应执行器实现；源生成 / 见证回读范围 | 仓内无对应执行器实现 | confirmed（搜索范围 `crates/` 全量） |
| 自动微分（含梯度对拍运行观测） | `crates/autograd/src/{tape,plan,replay,schedule,graph,facts}.rs`；`crates/train/tests/backprop_tests.rs:1,25,82,88,97,131`；`train/src/kernel.rs:45,47` `Reference` | 仓内 tape/plan/replay/schedule 小图；`Session<Reference,Sgd>` 在 CPU 参考 Kernel 上建图→`plan`→`backward`，取解析梯度与 `central` 中心差分数值梯度逐元素 `check_grad` 容差断言；本轮 `cargo test -p llm-train backprop` **50 passed / 0 failed**，含 `conv2d_gradients_match_central_differences_on_any_geometry`、`an_arbitrary_geometry_trains_end_to_end_under_sgd` | 接线 + CPU 参考后端梯度数值对拍 confirmed；真机后端梯度、大规模训练数值仍未取证 | confirmed（CPU Reference 对拍）/ unknown（真机） |
| 优化器 | `crates/train/src/optimizer.rs:186 Adam`、`:334 Sgd`、`:393 pub use muon::Muon`；AdamW 解耦衰减 + 偏差修正 + 范数裁剪 | 优化器实现在位 | 定义与结构 confirmed | confirmed |
| train 数据集/采样器 | `dataset/mod.rs:91` `Dataset`、`:117` `LoaderConfig`、`:143` `Loader`；`dataset/sampler.rs:15` `Sampler`、`:38` `from_config`；`rng.rs:55` `block`、`:84` `with_stream`、`:96` `next_u32`、`:115` `next_f32` | 三族采样 + 权重校验在位；基线编译绿；物理分词/打包源仍缺位 | confirmed（基线编译）/ inferred（生产数据链路） |
| 算子注册表 | `nn/src/registry.rs:17` `for_each_operator!` 宏；清单展开 **70 条**算子（elementwise/unary/激活/norm/attention/pool/conv/reduce/embed/rope/gemm·gemv·matmul f32+i2/i4/i8/dequant），编译期双断言名字与目录顺序一致 | 注册表驱动算子面；unary 族（abs/ceil/clamp/exp/exp2/floor/log/log1p/neg/recip/round/rsqrt/sign/square/sin/cos/sqrt/trunc/remainder）本轮已进 HEAD 基线 | 注册表与算子名 confirmed；各算子真机内核激活未核 | confirmed（注册表）/ unknown（真机内核） |
| cutile 内核 DSL | `crates/cutile/src/kernel.rs:141,554-557,572`；`pass.rs` 显式 match 子节点 | 上一轮子节点抽象中间态已收进 HEAD；`cargo clean -p cutile` 后 `Finished`，`write_call` 以切片作函数实参传参合法 | cutile 基线编译 confirmed；内核覆盖广度仍待扩 | confirmed（基线编译） |
| device 后端 | `backends/mod.rs:8`；`backends/metal.rs:14`；`prof/src/annotate.rs:46`；`dialect.rs:184,202,308`；`metal/program.rs:9`、`metal/chunk.rs:8` | `mod harness;` 挂载；`closed_enum!` 生成 `pub enum Emit` 跨 crate 可达；`chunk_reduce`/`entry` 为 trait 默认方法；使用方均导入 `Dialect`；`cargo check -p llm-device` `Finished` | device 基线编译 confirmed；真机后端激活/性能未核 | confirmed（基线编译）/ unknown（真机） |
| unsafe 与内存安全边界 | `Cargo.toml` workspace.lints.rust：`unsafe_code = "deny"`、`unsafe_op_in_unsafe_fn = "deny"`、`missing_docs`/`unused_lifetimes = "deny"`；本轮精确计数 101 处（unsafe 块 67 + unsafe fn 23 + unsafe impl 5 + extern unsafe 6），集中 `device/executor.rs`(37)、`prof/native/{sme,gemm8x8,i8}.rs`(14/14/12)、`cli/tests/audit.rs`(7)、`storage/mmap.rs`(5)、`nn/kernel_fast_i8.rs`(4)、`nn/kernel_fast_pool.rs`(2) | deny 默认禁止 unsafe，越界岛显式标注、编译期可见；unsafe 集中在设备执行器/微核/mmap/fast kernel | 策略与分布 confirmed；逐块正确性未审计 | confirmed |
| 外部依赖/供应链 | 运行期外部依赖仅 `half`、`num-traits`；`bytemuck`/`serde`(derive)/`toml` 为 optional；`proc-macro2/quote/syn` 编译期宏、`proptest/criterion/similar-asserts` 测试/bench、`libfuzzer-sys` optional fuzz；无 cudarc/metal-rs/libloading/wgpu/coreml/opencl 运行期绑定 | 运行期依赖极薄；系统绑定缺位 | 依赖清单 confirmed | confirmed |

> 未覆盖范围（本轮未逐项深核）：llm.u 各后端在真机上的运行激活与性能；Burn 后端在本机设备的实测；三仓训练数值精度对拍；`benchmarks` 数值结果。这些项下文只给代码结构层面的定级。

## 1. 仓库定位与架构哲学

### 1.1 llm（新）：七层契约 + 编译期守门的参考/编排栈

- 工作区根 `Cargo.toml` 是纯 workspace，无 package；组合根与两个二进制 `llm`/`cutile` 都在 `crates/cli`。
- 分层自述（各 `lib.rs` 头部）：`tensor` 逻辑张量 → `dtype` 元素/量化格式 → `storage` 双 lane 缓冲视图 → `nn` 算子与层声明 → `algorithm` 模型变体语义 → `engine` serving 状态机 → `serve` 请求/采样/路由；`autograd`/`train` 训练侧；`device` 设备源 lowering；`cutile`/`annotation` 内核 DSL 与契约宏；`prof` 能力档案与片段构建期对齐；`weights` 权重编码；`shard` 拓扑切分。
- 工程口径：`[workspace.lints.rust] unsafe_code = "deny"`、`unsafe_op_in_unsafe_fn = "deny"`、`missing_docs`/`unused_lifetimes = "deny"`；`clippy all`/`cargo`/`dbg_macro`/`todo`/`unimplemented`/`needless_range_loop` = deny。运行期外部依赖极薄（`half`/`num-traits`；`bytemuck`/`serde`/`toml` 为 optional）。

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
| crate 数 | 22 | 15 | 42 |
| `.rs` 文件 | 690 | 344 | 1989 |
| 源码行 | 167,119 | 104,592 | 490,469 |
| 测试计数 | 2787 | 1127 | 6788 |

### 2.2 llm 生产行 Top crate（行，HEAD 基线实测）

| crate | 行 | 职责 |
| --- | --- | --- |
| nn | 23,982 | 算子内核 / 层声明 / 性能下界 |
| weights | 15,362 | 权重格式、PTQ、codec、存档 |
| train | 13,461 | 训练循环状态、loss、优化器、检查点、数据集 |
| prof | 12,705 | 能力档案 + 设备片段构建期对齐 |
| device | 11,633 | 设备源 lowering 与厂商后端 |
| shard | 10,700 | 拓扑/流水线/切分/见证/集合通信参考 |
| algorithm | 10,397 | 注意力/SSM/记忆/路由变体语义 |
| serve | 10,098 | 请求文法、采样、补全端点、路由 |
| cutile | 8,617 | 内核 DSL：AST/emit/fuse/pass/plan |
| cli | 6,717 | 组合根与二进制 |

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
| cli | 2,926 |
| kernel | 1,451 |
| proto | 1,422 |
| serve | 1,088 |

### 2.4 Burn 关键层行（行）

主框架层：

| crate | 行 | crate | 行 |
| --- | --- | --- | --- |
| burn-train | 34,366 | burn-tensor | 25,940 |
| burn-nn | 29,613 | burn-fusion | 27,313 |
| burn-cubecl | 25,240 | burn-backend | 16,399 |
| burn-autodiff | 12,368 | burn-core | 14,786 |
| burn-optim | 10,941 | burn-ir | 9,040 |
| burn-linalg | 10,063 | burn-einsum | 1,113 |

周边/测试层：`burn-backend-tests` 69,742、`burn-flex` 40,234、`burn-remote` 17,276、`burn-store` 17,198、`burn-cubecl-fusion` 14,771。

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
- llm 逻辑张量数值算子面薄：`ops.rs` 12 个元素/归约（`map/add/sub/mul/div/add_scalar/mul_scalar/sum/max/min/argmax/all_close`），tensor 层 45 个形状/构造/存取方法；重算子在 `nn` crate。Burn `burn-tensor` 25.9k 行，算子面显著更宽。

### 3.2 算子接口与注册表

- **llm**：编译期 const 注册表 `REGISTRY`（`nn/src/registry.rs:97`），`for_each_operator!` 单清单声明 70 个算子（`:17-91`），名字与 `prof::catalogue` 同序，错位编译期失败（断言 `:109,:113`）；查表二分（`:131`）。算子类型由清单一次展开类型化 API，注册表/类型化入口/样本三者不漂移。
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

> 激活边界纪律：上表 llm 一行，"代码目录/源生成存在"不等于该后端在生产入口被选中。本轮 confirmed 的生产设备选择只有 `serve/src/model.rs:27` 的 CPU；Metal 仅测试构造；CUDA/昇腾无 `Executor` 实现。`llm-device` 基线编译本轮已通过（§13.1），但真机后端运行激活/性能未取得干净观测。llm.u 后端的真机运行激活本轮未取证（unknown）；Burn 各后端运行状态以其仓库 CI/文档为准，本轮未在本机复跑。

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
| 图/tape 结构 | `autograd` 4,044 行（src 2,119 / tests 1,925）：`Requirement/Role/Var/Node/Tape/Ordering/Schedule/Summary/Account/MemoryPolicy/Plan/Budget/Sizes/Footprint/Choice/Config/Replay/Store`（ADR 0055） | 手写反向（`exec` 内 `plan.rs`/`teacher`） | `burn-autodiff` 12.4k：`graph/grads/checkpoint/distributed/ops/runtime/backend` |
| train 接线 | `session.rs:5` 引 `Tape/Var`；`backprop/learner/pool` 同引 confirmed | train 复用 exec 计算面 | autodiff 作后端装饰器挂接 |
| 梯度检查点 | segment 级（`autograd` 结构内，`MemoryPolicy` 自动/全留/分段三档） | 未见独立检查点策略 | `autodiff/checkpoint` 策略可配 |
| 状态 | 元数据层闭环：`gradcheck` 中心差分对拍 200 张随机图（扇出累加、detach 分支、分层与顺序逐位一致、重复反向逐位一致、图外参数梯度恰为 0）；train 闭环 `Reference` CPU 后端 50 例解析梯度 vs 中心差分对拍实测通过（含 conv2d/GRPO/KL/端到端 SGD）；真机后端梯度仍未观测 | 手写反向，成熟度本轮未深核 | 生产级梯度跟踪/快照/反向 |

## 6. 模型层（nn）

### 6.1 层/算子清单

- **llm `nn`**（24.0k）：注册表 70 算子（`registry.rs:17-91`）——elementwise/unary（add/sub/mul/div/max/min/scale/residual/abs/ceil/clamp/exp/exp2/floor/log/log1p/neg/recip/round/rsqrt/sign/square/sin/cos/sqrt/trunc/remainder/exp2）、激活（gelu/silu/relu/sigmoid/tanh/leaky_relu/hard_sigmoid）、归一化（rmsnorm/batch/group/instance/layer norm）、注意力（attention/flash/decode_attention/decode_flash/latent）、池化（adaptive/avg/max pool1d/2d）、卷积（conv2d）、reduce（max/sum/prod）、embed、rope、scan、softmax、gemm/gemv/matmul（f32 + i2/i4/i8）、dequant（i2/i4/i8）。层声明 `Linear`/`OpLayer`/`ContractSpec`/`Bounds`（`contract.rs:178`、`declare.rs:161,285`）。
- **Burn `burn-nn`**（29.6k）：`modules/` 下 attention/cache/conv/embedding/dropout/linear/interpolate/norm/pool/pos_encoding/rnn/transformer/unfold/rope_encoding/pixel_shuffle/fold/noise/identity/pairwise_distance/cosine_similarity；`loss/` 19 个文件（17 种 loss + `mod`/`reduction`；BCE/cross_entropy/ctc/gaussian_nll/hinge_embedding/huber/kldiv/lp_loss/margin_ranking/mse/multi_margin/poisson/rnnt/smooth_l1/soft_margin/triplet/cosine_embedding）。
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
| 训练循环 | `train` 13.3k：session/learner/backprop/loss/optimizer/geometry/kernel/pool/rng；forward/backward 以闭包注入（`learner.rs:164-182`）；基线编译本轮绿，backprop 套件含端到端 SGD 训练例实测通过；真机大规模收敛未核（见 §13.1） | `train` 7.8k：loss/优化器/调度/打包/检查点/rollout 桥 | `burn-train` 34.4k：完整训练循环/指标/检查点 |
| 优化器 | `Sgd/Adam`(AdamW 解耦衰减 + 偏差修正 + 全局范数裁剪) `/Muon`（`optimizer.rs:186 Adam`、`:334 Sgd`、`:393 pub use muon::Muon`） | 仅 `Adam`（`train/src/optim.rs:12`） | 12 个具体优化器：sgd/adam/adamw/adan/lamb/lbfgs/lion/momentum/muon/rmsprop/adafactor/adagrad + decay/grad_accum/parallel |
| LR 调度 | warmup + anneal（`session.rs` `Schedule`） | cosine + warmup 类 | `burn-optim/lr_scheduler` 多策略 |
| loss | cross_entropy/argmax_hits/mse/bce_with_logits/huber/hinge（`loss.rs:148` 等）+ Distill（`loss.rs:361`）/Grpo 枚举变体，策略族函数体在 `loss_rl.rs:109,201`（`loss.rs:10` `mod rl`、`:12` 再导出） | 复用 exec 计算面 | 17 种 nn loss（19 文件含 mod/reduction）+ train 指标 |
| 指标/评估 | Learner 账表 + evaluate；无可插拔指标/渲染器 | 未见独立指标注册表 | 30+ 指标（burn-train） |
| 数据集 | `Dataset` trait + `Loader`/`Chunks` + 顺序/洗牌/加权三族采样 + 权重校验，基线编译绿；无物理分词/打包源 | 未见独立 dataset crate | `burn-dataset` |
| 检查点 | weights/storage 自描述编码 + CRC-32 + 恢复事实 | checkpoint 字节存档 | `burn-store`/checkpoint 完整 |
| RL | log_prob/entropy/advantage/GRPO 槽位（`loss_rl.rs:109` Grpo、`:201` grpo_loss；`loss.rs:361` Distill） | rollout 桥 | `burn-rl`：env/policy/buffer |

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

### 12.1 llm 注册算子（70 项，`registry.rs:17-91`）

池化（adaptive_pool/avg_pool2d/max_pool1d/max_pool2d）、卷积（conv2d）、插值（interpolate）、归一化（batch/group/instance/layer/rmsnorm）、注意力（attention/flash/decode_attention/decode_flash/latent）、elementwise（add/div/max/min/mul/sub/residual_add/scale/neg/recip/remainder/clip/clamp）、unary（abs/ceil/floor/round/trunc/sign/exp/exp2/log/log1p/rsqrt/square/sin/cos/sqrt）、激活（gelu/hard_sigmoid/leaky_relu/relu/silu/sigmoid/tanh）、归约（reduce_max/reduce_sum/reduce_prod）、embed、rope、scan、softmax、矩阵族（gemm_f32/matmul_f32/gemv_f32 + gemm_i2/i4/i8、gemv_i2/i4/i8、matmul_i8）、反量化（dequant_i2/i4/i8）。

### 12.2 llm.u 内核族（`reg.rs:32-41`）

`Gemm/Scalar/Qgemm/Attention/Kv/Map/Reduce` 七臂；`exec` 内另见 decode/engram/mixer/quantize/deformat/format/sampler 等描述符。

### 12.3 Burn 算子面

后端 `FloatTensorOps/BoolTensorOps/IntTensorOps/ModuleOps/ActivationOps/QTensorOps/TransactionOps/DistributedOps/CollectiveOps` 全族；`burn-linalg` 10.1k、`burn-einsum` 1.1k；nn 层与 loss 见 §6.1。

### 12.4 CPU 手写微核

- llm：微核体、`target_feature` 门、特性探测与协商表在 `prof::native`（gemm8x8 NEON/AVX2/scalar、i8 aarch64 dotprod、sme Apple SME）；`cutile/src/native.rs` 为 nn→cutile→prof::native 的零成本 `pub use` 转发面；nn 侧 `kernel_fast`/`kernel_fast_tn`/`kernel_fast_i8`/`kernel_fast_pool`/`kernel_gemv_simd_stable`/`simd_nightly` 经此引用（AVX 路径本轮未核到）。
- llm.u：`host/simd`/`host/gemm`/`backend/blas`（NEON/AVX/Accelerate 倾向，本轮未在本机复跑）。
- Burn：CPU 后端可用，极致调优依赖 CubeCL 生成。

## 13. 工程质量与可维护性

| 对照点 | llm | llm.u | Burn |
| --- | --- | --- | --- |
| lint 门槛 | unsafe_code/unsafe_op_in_unsafe_fn/missing_docs/unused_lifetimes deny，clippy all/cargo/needless_range_loop deny；dbg/todo/unimplemented deny | 仅 cli crate `unsafe_code = "deny"`，device/exec 等其余未统一 | 成熟 CI/deny 配置 |
| 外部运行期依赖 | 极薄：half/num-traits；bytemuck/serde/toml optional | 多后端 feature 拉 cudarc/metal/libloading/coreml-native/clblast | wgpu/metal/serde/half/bytemuck/ndarray 等（cudarc 经 cubecl） |
| 测试计数 | 2787 | 1127 | 6788 |
| 静态门 | `tools` config 驱动质量门；本轮 gate 测试 `every_source_passes_the_static_gate` 拦截 3 项 train 文件 Lines 存量赤字（既有基线 #44，见 §13.1） | 未见同构静态门 | CI 门 |
| 文档/契约 | ADR 目录（`docs/adr`）、层职责自述写在各 lib.rs 头部 | config-first 契约 | contributor book/book |

### 13.1 本轮门禁实测（llm，快照 `64233760`，toolchain 1.100.0）

> 取证期间 llm 存在并行工作流连续提交与未提交文件周期收编，HEAD 在区间内由 `cf81efb8` 推进到 `64233760`；下列结论按「基线编译/测试观测」与「工作区未提交快照差异」分开记账，未提交 diff 不计入已提交基线。

**编译观测（本轮实测，confirmed）**

- `cargo check --workspace` **`Finished` 无 error**（1.100.0，edition 2024）。workspace lints：`unsafe_code = "deny"`、`unsafe_op_in_unsafe_fn = "deny"`、`missing_docs`/`unused_lifetimes = "deny"`，clippy `all`/`cargo`/`dbg_macro`/`needless_range_loop`/`todo`/`unimplemented` deny。

**测试观测（本轮实测，confirmed）**

- `cargo test -p llm-train backprop`：**50 passed / 0 failed**（0.01s），CPU `Reference` 后端解析梯度与中心差分数值梯度逐参数对拍，含端到端 SGD 训练例、conv2d/GRPO/distillation KL 图对拍。
- `cargo test --workspace`：**2369 passed / 1 failed**（33 ignored）。唯一红灯为 `llm-tools --test gate` 的静态门 `every_source_passes_the_static_gate`（`crates/tools/tests/gate.rs:108`）；其余单测/集成测全绿。
- 静态门 3 项存量 Lines 赤字（#44，源码行数超阈，非编译错误；上一轮 8 项，其余 5 项本轮已治理闭合）：`train/session.rs` Lines=1211、`train/backprop.rs` 915、`train/geometry.rs` 1403。

**工作区未提交快照差异（不计基线）**

- 代码侧 2 项 `M`：`autograd/src/schedule.rs`、`autograd/src/tape.rs`（小改，进行中）；其余 3 项为文档（本报告、ADR 0055、review 笔记）。上一轮 `cos/sin/sqrt` 等 unary 算子增量已收进 HEAD 基线（注册表 70 条）。

**本轮仍未取得干净观测（unknown）**

- 三仓真机后端（CUDA/Metal/ROCm/WGPU）运行激活、性能与数值正确性；llm serve HTTP 端到端接通（仅见 `http.rs` 薄适配器，未起进程）；llm.u 各 feature 实际编译状态与设备覆盖；burn 本机后端状态；跨进程集合通信真实传输链路；形状算子边界行为逐一对拍。`cargo clippy --workspace --all-targets`、`cargo fmt --check` 本轮未在收尾门禁外单独复跑。

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
| 自动微分（图/tape） | ◐ 仓内小图，train 接线在位，CPU Reference 后端梯度对拍 50 例实测通过；真机后端梯度未核 | ○ 手写反向 | ● 后端装饰器 autodiff |
| 梯度检查点 | ◐ segment 级 | ？ | ● 策略可配 |
| 优化器丰富度 | ◐ Sgd/Adam/Muon | ◐ 仅 Adam | ● 12 个含 LBFGS/Momentum/Muon |
| LR 调度 | ● warmup+anneal | ● cosine+warmup 类 | ● 多策略 |
| 指标注册表/evaluator | ◐ Learner 账表 + evaluate，无可插拔指标 | ？ | ● 30+ 指标 |
| 数据集 crate | ◐ Dataset trait + Loader/Chunks + 三族采样在位，基线编译绿；无物理分词/打包源 | ？ | ● burn-dataset |
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
3. **张量算子深度**：llm 逻辑张量数值算子仅 12 个元素/归约，重算子在 nn；Burn tensor 25.9k 行、后端算子族全。
4. **模块派生与生态**：llm 无 Module derive 宏、无远程后端、无独立 dataset crate。

### 15.2 llm 的相对强项（本轮证据支持）

1. **类型系统结构新意**：`Tensor<B,R,K>` 存储/秩/元素三位分离；秩同时支持定长数组与运行期 `Vec<usize>`；元素 sealed 分组收紧算子 bound；View/ViewMut/DynTensor 三桥齐备。
2. **依赖极薄 + 编译期守门**：运行期几乎不拖重型数学/序列化依赖；unsafe 默认 deny，仅点名越界岛；静态门 + 单清单注册表防三处漂移。
3. **serving 状态机自给**：连续批、分页 KV/radix、准入、投机采样、请求文法与路由在仓内成体系；内置分词器与自有权重 codec。
4. **算法变体覆盖**：Mla/Latent/Mamba/Kda/Moe/Mhc/Hybrid/Tower 等注意力/SSM/记忆/路由语义在 algorithm 层成族。

### 15.3 本轮发现的存量债务（llm）

- **未提交增量（快照差异，不计基线）**：代码侧 2 项 `M`——`autograd/src/schedule.rs`、`autograd/src/tape.rs` 小改；其余为文档。上一轮 unary 算子增量已收进 HEAD 基线（注册表 70 条）。
- **基线编译/测试观测已取得（本轮实测）**：`cargo check --workspace` `Finished` 无 error；`cargo test --workspace` 2369 passed / 1 failed（33 ignored），唯一红灯为 tools 静态门（非编译错误）；train `Reference` 后端梯度对拍 50 例全绿。`cargo clippy --workspace --all-targets`、`cargo fmt --check` 未在收尾门禁外单独复跑。
- 静态门 3 项 Lines 存量赤字（#44：`train/session.rs` 1211、`train/backprop.rs` 915、`train/geometry.rs` 1403）需拆文件/降复杂度。
- `tensor/src/tensor.rs:121` `into_slice_mut` 在输出侧回退后无生产调用方（全仓 grep 仅定义处一处），作为 pub 消费式解包方法保留；去留属 API 口径决策。

### 15.4 llm.u 的定位

旧版配置驱动执行框架：设备后端代码量厚（device 30.8k + exec 26.7k），多后端 feature 门控；train/serve 为参考层，经端口抽象。本轮未核其真机运行激活与数值正确性，成熟度定级留待运行观测。

## 16. 未覆盖与待核清单（诚实分层）

1. 三仓后端在真机（CUDA/Metal/ROCm/WGPU）的运行激活、性能与数值正确性——本轮只读代码结构，未起设备复跑。
2. llm 训练闭环：CPU `Reference` 后端解析梯度 vs 中心差分对拍 50 例已实测通过、含端到端 SGD 训练例；真机后端梯度激活、大规模训练收敛与调度策略仍未核。
3. llm serve HTTP 运行时是否端到端接通——本轮仅见 `http.rs` 薄适配器，未起服务进程。
4. llm.u 后端各 feature 的实际编译状态与设备覆盖；其 train/serve 端口是否接生产实现。
5. Burn 本机各后端编译/运行状态；CubeCL fusion/IR 的实际生效路径。
6. 三仓形状算子的边界行为逐一对拍（广播/切片/对角/非连续布局）。
7. 旧报告 `docs/COMPARISON-llm-llmu-burn.md` 的历史评分向量本轮未沿用；本报告定级以 2026-11-24 快照源码为准。

> 证据分级：标注 `confirmed` 的定位均可回链到上文 path:symbol:line；标 `◐`/`？` 的项为结构在位但运行激活未核，不构成"生产可用"主张。

## 17. 22 维评分（0–10，本轮证据定档）

### 17.1 标尺

| 分段 | 含义 |
| --- | --- |
| 9–10 | 成熟完备，生产可用，生态/类型支撑完整 |
| 7–8 | 主体完整，少量缺口或局部调优不足 |
| 5–6 | 端口/骨架在位，部分实现或仅单点落地 |
| 3–4 | 最小可用，或只有单点实现/手写集合 |
| 1–2 | 仅声明、配置标签或占位，无真实链路 |
| 0 | 缺失 |

> 分数是当前代码快照的相对成熟度，不是能力上限；三仓定位不同（Burn 通用训练框架 / llm.u LLM 推理多后端执行 / llm 面向 Burn 能力端口的分层重构），总分须结合权重视角解读。本轮整 workspace `cargo check` 绿、`cargo test --workspace` 2369 passed / 1 failed，CPU `Reference` 后端解析梯度 vs 中心差分对拍 50 例实测通过；后端真机（Metal/CUDA/昇腾）运行激活本轮未核，相关维度按代码结构定档。新增两维：#21 unsafe/内存安全边界、#22 依赖精简/供应链厚度（该维为「薄=高分」方向，依赖越少、系统绑定越弱得分越高）。工作区未提交 autograd 两文件小改只作快照差异，不计入已提交基线。

### 17.2 维度速览表

| # | 维度 | llm | llm.u | Burn | 本轮定档依据 |
| -- | --- | --- | --- | --- | --- |
| 1 | 张量/类型系统 | 10 | 2 | 10 | llm `Tensor<B,R,K>` 双秩 + sealed 分组 + 三视图桥；llm.u 扁平 Tensor/View；Burn `Tensor<const D,K>` |
| 2 | 后端广度 | 5 | 6 | 8 | llm 六厂商 Fragment/源生成，Executor 实现者为 Metal/Cpu/Local/Vacant/Trace 装饰器（Counting 测试桩），基线编译本轮已绿；llm.u feature 门控 cuda/metal/intel/blas/rocm/musa/opencl/ane；Burn CubeCL 多 runtime + Flex/Remote |
| 3 | 后端执行成熟度/性能 | 3 | 6 | 8 | llm 生产装配仅 CPU（`serve/src/model.rs:27`），Metal 仅测试构造，CUDA/昇腾无 Executor impl，真机激活未核；llm.u 多后端 GEMM/内核族厚，真机激活本轮未复跑；Burn 跨后端可跑完整算子集 |
| 4 | 算子/内核覆盖 | 5 | 4 | 9 | llm 注册表 **70 条**算子（elementwise/unary/激活/norm/attention/pool/conv/reduce/embed/rope/gemm·gemv·matmul f32+i2/i4/i8/dequant）+ 四层契约，unary 族本轮已进 HEAD 基线；真机内核激活未核；llm.u 七臂内核族；Burn 张量 API 全套 + linalg/einsum/fusion |
| 5 | nn 层完整度 | 5 | 3 | 9 | llm conv/pool/norm/attn/embed/rope 齐，无 RNN；llm.u 无 conv/pool/RNN；Burn 8 大类 + 49 Config 含 RNN/Transformer |
| 6 | 自动微分 | 7 | 2 | 10 | llm 独立 tape/schedule/account/plan/replay 元数据层；本轮 `cargo test -p llm-train backprop` 50 例全绿：`Reference` CPU 后端解析梯度与中心差分逐参数对拍（ABSOLUTE 2e-3 + RELATIVE 0.05），含 conv2d/GRPO/distillation KL/端到端 SGD；缺口：无后端装饰器式任意可微闭包（算子按名注册、显式 `record`）、分布式无真实传输后端/NCCL、GPU 后端梯度激活未观测；梯度侧仅有逐 Var 取读（`gradient`/`take_gradient`）与按槽 `GradAccum` 窗口快照（`accum_snapshot`），无脱离优化器窗口的 Var→向量独立快照导出；llm.u 逐层手写 bwd；Burn 后端装饰器任意可微 + checkpoint |
| 7 | 优化器 | 4 | 3 | 9 | llm Sgd/Adam/Muon；llm.u 仅 Adam；Burn 12 个 impl + LBFGS/Momentum/累积/裁剪/衰减 |
| 8 | 调度/训练循环 | 5 | 4 | 9 | llm Session/Learner/evaluate/账表/优化步在位，本轮基线编译绿，backprop 套件含 `an_arbitrary_geometry_trains_end_to_end_under_sgd` 端到端 SGD 训练例实测通过；真机/大规模训练收敛与调度仍未核；llm.u Session/Trainer + Cosine；Burn Learner/early stopping/sharder/logger |
| 9 | 损失函数 | 5 | 4 | 9 | llm CE/argmax/mse/bce/huber/hinge + Distill/Grpo（策略族物理拆至 `loss_rl.rs`，`loss.rs:10,12` 挂载再导出）；llm.u 7 fn；Burn 17 种 loss（含 CTC/RNNT/三元组/NLL） |
| 10 | 指标/评估器 | 3 | 1 | 10 | llm 仅 loss/命中/样本数账表；llm.u 仅 perplexity；Burn 30+ 指标 + evaluator/renderer |
| 11 | 数据集管线 | 3 | 1 | 9 | llm `Dataset` trait + `Loader`/`Chunks` + 顺序/洗牌/加权三族采样 + 权重校验 + `dataset_tests.rs` 181 行，基线编译绿；无物理分词/打包源；llm.u pack/shard 函数；Burn audio/nlp/vision/source/transform |
| 12 | 检查点/序列化 | 6 | 4 | 9 | llm weights/storage 自描述 + CRC-32 + 恢复事实；llm.u 状态字节存档；Burn store/checkpoint/burnpack/pytorch |
| 13 | 推理服务 | 6 | 6 | 3 | llm ifb/radix/page/speculate + HTTP 薄适配器；llm.u 调度生命周期完整，HTTP 运行时未核；Burn 仅示例 server/remote |
| 14 | 分布式/并行 | 3 | 2 | 6 | llm 拓扑/切分/Reduce 齐、Remote 占位；llm.u Comm trait + CPU 空参考实现；Burn WebSocket/remote + shard 规则；三家均无 NCCL 级栈 |
| 15 | 量化/精度 | 6 | 5 | 7 | llm GGML+MX+FP4+亚字节格式最广，落点权重档案侧；llm.u I2–I8+FP8/FP4+AWQ；Burn 参数级 ModuleMapper/Quantizer/Calibration；三家均无 QAT |
| 16 | 随机数/确定性 | 7 | 1 | 8 | llm 计数器 PRNG + 三分布可复现；llm.u 采样注入闭包；Burn rand/seed/Distribution |
| 17 | 内核 DSL 扩展 | 5 | 2 | 9 | llm cutile 协商/拼接/emit/指纹，自述仅 P0 模板纵切面 + typed kernel 起步；`cutile/src/native.rs` 为 nn→cutile→prof::native 零成本转发面；本轮 clean 后 cutile 基线编译绿；llm.u 内核直写后端；Burn CubeCL IR 成熟多 runtime lowering + 融合 |
| 18 | 架构/可维护性 | 8 | 3 | 8 | llm 七层契约 + testkit 强制守分层 + ADR；llm.u device/exec 耦合、unsafe 分散；Burn crate 分层清晰 + derive 宏 |
| 19 | 测试/质量门禁 | 5 | 4 | 8 | llm 2787 测试计数；本轮 `cargo check --workspace` 绿、`cargo test --workspace` 2369 passed / 1 failed，唯一红灯为 tools 静态门 `every_source_passes_the_static_gate`（`gate.rs:108`，3 项 train 文件 Lines 存量赤字，非编译错误）；llm.u 1127 测试 + fuzz nightly；Burn 6788 测试 + backend-tests 套件 + CI |
| 20 | 生态/示例/文档 | 4 | 4 | 8 | llm 内部架构文档/ADR 完备，无对外模型与示例；llm.u README/perf 实测台账详尽；Burn book + 示例生态 |
| 21 | unsafe/内存安全边界 | 8 | 3 | 6 | llm workspace lint `unsafe_code = "deny"`、`unsafe_op_in_unsafe_fn = "deny"`；本轮精确计数 101 处（unsafe 块 67 + unsafe fn 23 + unsafe impl 5 + extern unsafe 6），集中 `device/executor.rs`(37)、`prof/native/{sme,gemm8x8,i8}.rs`、`storage/mmap.rs`、`nn/kernel_fast_i8.rs`/`pool.rs` 显式岛；llm.u 441 处（块 322 + fn 104 + impl 15），散见 device/ffi·opencl·simd·blas、exec/tests、cuda 后端，lint 仅 cli deny、其余分 crate 未统一；Burn 246 处（块 208 + fn 27 + impl 11），未检索到 `forbid/deny unsafe_code`，散见 graph/cast/quantization/dispatch |
| 22 | 依赖精简/供应链 | 9 | 4 | 5 | 方向：运行期依赖越薄、系统绑定越少越高分。llm 运行期外部依赖仅 half、num-traits，bytemuck/serde(derive)/toml 为 optional，无 cudarc/metal-rs/libloading/wgpu/coreml/opencl 运行期绑定；llm.u 厚系统栈：cudarc 0.19(cuda-13000/cublas/cublaslt/f16)、metal 0.33、libloading 0.8、coreml-native 0.2、clblast/opencl、musa 门控；Burn 谱系最宽：serde 29/num-traits 13/wgpu 20/wgpu-msl 11/metal 15/half 7/bytemuck 6/serde_json 5/ndarray 2（cudarc 经 cubecl） |

### 17.3 总分

**A. 等权视角（22 维各 1/22，通用框架口径）**

| | llm | llm.u | Burn |
| --- | --- | --- | --- |
| 平均分 | **5.55** | **3.36** | **8.05** |

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
| 3 | 分布式/并行 | 3 | 2 | 6 |
| 3 | 随机数/确定性 | 7 | 1 | 8 |
| 3 | 测试/质量门禁 | 5 | 4 | 8 |
| 2 | 自动微分 | 7 | 2 | 10 |
| 2 | 优化器 | 4 | 3 | 9 |
| 2 | 调度/训练循环 | 5 | 4 | 9 |
| 2 | 损失函数 | 5 | 4 | 9 |
| 2 | 生态/示例/文档 | 4 | 4 | 8 |
| 1 | 指标/评估器 | 3 | 1 | 10 |
| 1 | 数据集管线 | 3 | 1 | 9 |
| | **加权总分** | **5.55** | **4.23** | **7.28** |

### 17.4 相对上一版向量的校准说明

- **#8 调度/训练循环 3 → 5（llm）**：上一版因 train lib 被 cutile 未提交中间态阻塞、编译与运行观测缺失定 3；本轮 `cargo check --workspace` 绿，`cargo test -p llm-train backprop` 50 例全绿，含 `an_arbitrary_geometry_trains_end_to_end_under_sgd` 端到端 SGD 训练例实测通过，Session/Learner/evaluate/账表/优化步链路达最小可用闭环；真机大规模训练收敛、调度策略仍未核，故定 5 而非更高。
- **#4 算子/内核覆盖 4 → 5（llm）**：上一版注册表 52 条定 4；本轮上一轮 unary 增量收进 HEAD 基线，注册表展开 **70 条**，elementwise/unary/激活/norm/attention/pool/conv/reduce/embed/rope/gemm·gemv·matmul 与 i2/i4/i8/dequant 族齐备；各算子真机内核激活仍未核、无 RNN 类算子，故定 5（骨架在位、部分实现）而非 6–7。
- **#19 测试/质量门禁 维持 5（llm）**：本轮整 workspace `cargo check` 绿、`cargo test --workspace` 2369 passed / 1 failed，唯一红灯为 tools 静态门 3 项 train 文件 Lines 存量赤字（#44，非编译错误）；静态门未清、clippy/fmt 未收尾复跑，维持 5。
- **#6 自动微分 维持 7（llm）**：本轮独立读测试源码复核，`Reference` CPU 内核后端解析梯度与中心差分逐参数对拍，50 例实测通过（上一版口径 43 例）；仍缺后端装饰器式任意可微闭包、真实分布式传输/NCCL、GPU 后端梯度激活，以及脱离优化器窗口的 Var→向量独立梯度快照（现有逐 Var 取读与按槽 `GradAccum` 窗口快照），维持 7。
- **#11 数据集管线 维持 3（llm）**：`Dataset` trait / `Loader`/`Chunks` / 三族采样 / 权重校验 / 181 行测试在位且基线编译绿；仍无物理分词/打包源，维持 3。
- **#17 内核 DSL 维持 5（llm）**：上一版因 cutile 未提交抽象重构中间态存疑；本轮 `cargo clean -p cutile` 后 `Finished`，中间态已收进 HEAD；但 DSL 仍为 P0 模板纵切面 + typed kernel 起步，内核覆盖待扩，维持 5。
- **#21 unsafe/内存安全边界（llm 8 / llm.u 3 / Burn 6）**：llm `unsafe_code`/`unsafe_op_in_unsafe_fn` 均 deny，101 处 unsafe（块 67+fn 23+impl 5+extern 6）集中为显式岛；llm.u 441 处（块 322+fn 104+impl 15）分散且 lint 仅 cli deny；Burn 246 处（块 208+fn 27+impl 11）、未设 deny/forbid。计数口径为源码 `unsafe {`/`unsafe fn`/`unsafe impl`/extern unsafe，已排除所有层级 `target/` 与 fuzz 生成 codegen。
- **#22 依赖精简/供应链（llm 9 / llm.u 4 / Burn 5）**：该维「薄=高分」。llm 运行期依赖极薄、无厂商系统绑定；llm.u 携带 cudarc/metal/libloading/coreml/clblast 厚系统栈；Burn 跨后端谱系最宽（serde/wgpu/metal/half/bytemuck/ndarray 等）。
- **计数校正**：llm 2787 测试、690 `.rs`、167,119 行；llm.u 1127、344 `.rs`、104,592 行；Burn 6788 测试、1989 `.rs`、490,469 行；档位不变。
- 后端真机性能与精度、HTTP 端到端接通仍属未取证边界；CPU 参考后端训练数值正确性本轮已由梯度对拍观测。静态门存量赤字（#44）拆分治理完成后，#19 可重新定档。

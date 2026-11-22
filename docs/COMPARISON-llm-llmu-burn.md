# `llm` / `llm.u` / Burn 全维度源码级对比分析

> 基线：三份代码均按源码同口径实测（剔除 `target/` 构建产物与 tests 目录后统计生产行）。
>
> | 代号 | 路径 | 版本 / HEAD | 角色 |
> |---|---|---|---|
> | **llm**（新仓） | `/Users/fei/code/llm` | workspace，38 个成员 crate，HEAD `f6b46e76` | 重构后的分层契约架构（L1–L6） |
> | **llm.u**（旧仓） | `/Users/fei/code/llm.u` | workspace，14 个成员 crate（`fuzz` 被 exclude），HEAD `cf4e8fe3` | 配置驱动、后端直连的执行框架 |
> | **Burn** | `/Users/fei/code/burn` | `0.23.0-pre.2`，HEAD `aff0a3cf` | 成熟通用 Rust 深度学习框架（Tracel AI） |

## 0. 结论先行

1. **三者不是同一物种。** Burn 是面向训练/研究的通用张量框架（后端泛型 + 自动微分 + 优化器 + 指标 + 数据集 + RL 一整套）；`llm.u` 是一个**面向 LLM 推理/小训练循环、多后端 GEMM 性能优先**的执行框架；新 `llm` 是对 `llm.u` 的一次**架构重构**：把"后端直连"拆成"契约层 → DSL/emit → 后端 Fragment → 执行缝"的六层体系，能力端口向 Burn 看齐（补 autograd crate、conv/pool、rng、muon、shard 拓扑与 Collective 缝），但绝大多数执行面仍是骨架/占位。
2. **规模口径（源码实测）**

   | | crate 成员 | `.rs` 源文件 | 生产行 | 测试行 |
   |---|---:|---:|---:|---:|
   | llm（新） | 38 | 721 | 94,540 | 58,004 |
   | llm.u | 14 | 349 | 86,735 | 17,857 |
   | Burn | 41 | 1,851 | 381,556 | 95,111 |

   Burn 生产代码约为新 llm 的 **4.0×**、llm.u 的 **4.4×**；测试代码为 1.6× / 5.3×。
3. **关键差距不在"有没有目录"，而在"缝后面是不是真实现"。** 新 llm 的 `Executor`/`Collective`/`Backend` trait 端口齐全，但 GPU 执行器目前仅 Metal 一个真实实现，远端集合通信返回 `Unsupported`；Burn 的端口后面是跨 CPU/CUDA/WGPU/ROCm 的可运行内核与可训练模型。
4. **llm.u 的长板是多后端实测性能**（CPU NEON/AVX2/AVX-512、Metal、OpenCL/SYCL、CUDA cuBLAS、MUSA 均有 GEMM 数据与对拍；ROCm 仅 SSM 内核库加载/探测，非完整 GEMM 后端），这一点 Burn 的 CPU/CUDA 后端不以极致微核调优为目标；新 llm 把 llm.u 的微核沉淀进 `gemm`/`conv`/`cutile`，但后端执行面尚在重建。

---

## 1. 仓库定位与架构哲学

### 1.1 llm.u：配置驱动的算子执行框架

- 自述："算子不写代码文件。整个框架只有算法骨架，所有算子都是配置文件里的一个条目：选骨架 + 选实现 + 传参数。"
- 14 个成员：`benches / cli / config / device / engine / exec / fuzz / graph / kernel / proto / serve / storage / tool / train`。
- 结构特点：`device`（28,325 生产行，占全仓近 1/3）直接持有各后端的 FFI 调用与内核封装；`exec` 是算法/模型/权重/采样的宿主；`train` 是紧贴推理栈的小训练循环。
- 后端与宿主代码强耦合，trait（`Backend`/`Compute`/`Engine`/`Gpu`/`Hold`/`Weights`）以 GEMM、注意力、SSM 前反向为中心裁剪，不是通用张量算子集。

### 1.2 llm（新）：六层契约架构，可被测试守住的分层

`docs/ARCHITECTURE.md` 定义了可执行契约（`testkit/src/architecture.rs` 读 `cargo metadata` 与源码逐条核对，不一致即 `cargo test` 失败）：

| 层 | 代表 crate | 职责 |
|---|---|---|
| L1 基础设施 | `diagnostic` `rng` `dtype` `wire` `formats` `tools` | 错误类型、确定性随机、精度闭集、字节码闭集、外部格式只读适配 |
| L2 数据层 | `prof` `spec` `gemm` `conv` `op` `cpu/cuda/metal/musa/opencl/rocm` `prim` | 算子四层契约 + 六份设备原语片段与见证 |
| L3 内核 | `cutile` `optim` | 协商→拼接→emit→指纹（cutile，不执行）；doctor/verify/bench/execution（optim，碰编译器与设备） |
| L4 运行时 | `tensor` `storage` `tokenizer` | 张量元数据、布局搬运、文本边界 |
| L5 模型与训练 | `weights` `snapshot` `autograd` `nn` `algorithm` `shard` | 权重结构、检查点、反向图、层定义、架构参考语义、切分拓扑 |
| L6 调度与服务 | `train` `engine` `device` `serve` | 训练循环、连续批/KV 页账、提交执行体、设备源 lowering、端点契约 |

- 依赖方向单向：L6 不得直接调 `cutile` 生成 kernel；L2 不得出现机制代码；设备厂商特判只允许出现在后端 Fragment。
- 与 llm.u 相比：新增 `autograd / rng / spec / gemm / conv / op / prim / prof / cutile / shard / snapshot / weights / formats / tokenizer / benchmark` 等独立 crate，把 llm.u 里糊在 `device`/`exec` 里的关注点拆开。

### 1.3 Burn：通用张量框架的经典分层

41 个 crate 按"张量核心 → 后端 → 模型 → 训练 → 服务/分布式"组织：

- 核心：`burn-core`（Module/Param/Quantizer/Store）、`burn-tensor`（`Tensor<D,K>` API）、`burn-backend`（`Backend` trait 族）、`burn-autodiff`（自动微分后端装饰器）。
- 后端：`burn-cpu`（CubeCL CPU 运行时，LLVM 编译，带融合/autotune）、`burn-cuda`、`burn-wgpu`、`burn-rocm`、`burn-cubecl`/`burn-cubecl-fusion`（CubeCL 内核 IR 与融合）、`burn-flex`（纯 Rust eager 后端，支持 no_std/WASM）、`burn-remote`。
- 模型：`burn-nn`（49 个 Config 模块）、`burn-linalg`、`burn-einsum`、`burn-vision`、`burn-flex`。
- 训练：`burn-optim`、`burn-train`（Learner/metric/evaluator/checkpoint）、`burn-dataset`、`burn-rl`、`burn-communication`。
- 工具：`burn-derive`（`#[derive(Module, Config)]`）、`burn-fusion`、`burn-instrument*`、`burn-shard`、`burn-signal`、`burn-pack`、`pytorch-reader`。

**哲学差异：** Burn 的抽象单元是**带后端泛型的张量**（`Tensor<const D, K>`），一切算子挂在张量方法上；llm.u 的抽象单元是**配置条目 + 算法骨架**；新 llm 的抽象单元是**算子契约（spec/op）+ 设备 Fragment + 执行缝**，张量只做无执行的元数据描述。

---

## 2. 规模与工程度量

### 2.1 代码体量（源码实测，已剔除 target）

| 维度 | llm（新） | llm.u | Burn |
|---|---:|---:|---:|
| workspace 成员 | 38 | 14（fuzz exclude） | 41 |
| `.rs` 文件 | 721 | 349 | 1,851 |
| 生产行 | 94,540 | 86,735 | 381,556 |
| 测试行 | 58,004 | 17,857 | 95,111 |
| 测试/生产 | 0.61 | 0.21 | 0.25 |

新 llm 生产行与 llm.u 接近，但测试行是 llm.u 的 3.2 倍——重构后把大量契约/边界校验沉进了测试（`testkit` 4,059 行、各 crate 自带 tests）。

### 2.2 生产行 Top crate

**llm（新）**：`gemm` 7,972 · `weights` 7,041 · `train` 7,006 · `device` 6,973 · `algorithm` 5,820 · `serve` 5,566 · `optim` 5,264 · `testkit` 4,059 · `cli` 3,741 · `conv` 3,728 · `spec` 3,421 · `cutile` 3,331 · `prof` 3,245 · `engine` 3,148 · `nn` 3,031。

**llm.u**：`device` 28,325 · `exec` 15,561 · `storage` 9,290 · `train` 7,581 · `tool` 5,765 · `config` 4,638 · `benches` 4,772 · `engine` 4,083 · `kernel` 1,451 · `serve` 893 · `proto` 1,048 · `graph` 818。

**Burn**（节选）：`burn-train` 33,088 · `burn-nn` 29,463 · `burn-tensor` 24,740 · `burn-autodiff` 12,341 · `burn-core` 11,578 · `burn-optim` 10,768；后端壳层很薄（`burn-cuda` 58、`burn-cpu` 61、`burn-wgpu` 253），重内核在 CubeCL IR 与 `burn-cubecl` 侧。

### 2.3 第三方运行期依赖

| | llm（新） | llm.u | Burn |
|---|---|---|---|
| 运行期三方 | `bytemuck`(optional)、`serde`(optional)、`libfuzzer-sys`(optional) | `cudarc 0.19`、`metal 0.33`、`coreml-native 0.2`、`libloading 0.8`、`serde/serde_json`、`faer 0.24`、`clap`、`criterion`(dev) | `rand`、`half`、`bytemuck`、`cubecl`、`wgpu`、`cudarc`（经 CubeCL）等一整套生态 |

- 新 llm 刻意把外部依赖压到最低：后端 FFI 不直链 `cudarc/metal`，而是走 `cutile` DSL 生成设备源（`.cu/.metal/.cl/.cpp`）+ `optim` 的 `Driver` 编译执行；这是与 llm.u 最大的工程转向。
- llm.u 直接链 `cudarc`/`metal`/`coreml-native`，后端能力受这些 crate 的 API 面约束。

### 2.4 unsafe 边界

| | 真实 `unsafe {}` 块所在文件（生产） | 边界策略 |
|---|---|---|
| llm（新） | 6 个岛：`storage/mmap.rs`、`gemm/fast/{i8,micro,sme}.rs`、`device/metal/executor.rs`、`device/metal/executor/device.rs` | workspace `unsafe_code = "deny"`，仅显式放行；文档称 Metal FFI 为唯一 unsafe 岛（ADR 0035），SIMD 岛在 gemm |
| llm.u | 49 个文件：CPU SIMD/内存、CUDA cuBLAS、Metal runtime、OpenCL/SYCL、MUSA、BLAS adapter、storage plane/store、benches OpenCL/CLBlast、tool gate | unsafe 分散在各后端 FFI 与 SIMD 路径，未集中收口 |
| Burn | 后端内 FFI  unsafe 集中在 CubeCL/后端 crate；`burn-core/tensor/nn` 业务层无 unsafe | 业务类型安全，unsafe 下沉到后端实现 |

---

## 3. 张量模型与类型系统

### 3.1 张量抽象

| 维度 | llm（新） | llm.u | Burn |
|---|---|---|---|
| 张量类型 | `Tensor` / `OwnedTensor` / `View` 三形态；`Shape` + `Strides` + `Order` + 只读字节视图 | `Shape{m,n,k}`（GEMM 三维）+ `Vec<f32>` 扁平缓冲；无独立张量类型 | `Tensor<const D: usize, K = Float>`：**秩 D 编译期定、元素类型 K 编译期定、后端 B 泛型** |
| 后端/秩泛型 | 无后端泛型；张量层不做执行 | 无后端泛型；无秩泛型 | `Tensor<B, const D, K>`，后端作为泛型参数贯穿算子与模块 |
| 布局 | 逻辑布局 + `Layout<B>` 块格式；`storage` 负责布局搬运 | 行主序扁平；tiling/pack 在 GEMM 路径内部 | 连续/非连续视图、stride 张量、reshape/permute 全套 |
| 元素类型 | L1 `dtype` 闭集：标量 `DType` + 块量化 `QType`(GGML `Q4_K` 等) + 微缩放 `MxType`(OCP MX / NVFP4) + `Element`(E2M1/E2M3/E3M2/E4M3/E5M2/INT8) | `Prec` 枚举：F32/Tf32/F16/Bf16/I8/I4/I3/I2/F8E4M3/F8E5M2/Fp4/F64(声明未实现)；`Compute` 累加档 I32/F32/Tf32/F64 | `FloatDType`(F32/F16/BF16) + `IntDType`(I32 等) + `QuantValue`(Q8F/Q8S/Q4F/Q4S/Q2F/Q2S + E4M3/E5M2/E2M1) |

**判定：**
- Burn 的类型系统最"硬"：秩和 dtype 进类型层，编译期拦截后端不支持的算子；新 llm 与 llm.u 都是**运行期形状/精度**，靠校验层（`spec`/`testkit`）在构建期或启动期把关。
- 新 llm 的精度闭集广度（GGML 块量化 + MX 微缩放 + FP4）超过 Burn 当前的张量 dtype 表；但 Burn 的量化是挂在模块参数上的可映射参数体系（见 §7）。
- llm.u 的 `Prec` 把 TF32 建模为"F32 存储上的计算档"，这一点 Burn 由后端 dtype 能力承担，不显式区分存储档与计算档。
- 新 llm 的 Fragment 源（每设备 9 份：`cvt/dot/epilogue/load/load_async/quant/reduce/store/witness`）是**模板占位 + 少量真实微核**（如 `dot/fma4` 四路 FMA 链），GEMM 大内核在 `gemm`/`cutile` 侧经 DSL 拼接；llm.u 的后端源是直接可编译的 cuBLAS 封装与手写 Metal/OpenCL 内核。

### 3.2 算子接口与注册表

| | llm（新） | llm.u | Burn |
|---|---|---|---|
| 算子契约 | 四层：`spec`(签名/样本/语义表/代价表达式/模板词法) → `gemm`(内核族) → `conv`(NN 算子) → `op`(声明解析 + 唯一 `REGISTRY`)；`spec::einsum` 推求和语义 | `reg.rs` 家族表：matmul(5 impl)/elemwise/attention(3 impl)/reduce；`Reg::get` 按 family+impl 解析 | 算子是张量方法（`tensor.matmul()`, `.relu()`, `.conv2d()`…），经 `BridgeKind`/后端 bridge 分发到 `Backend` 算子 |
| 算子扩展 | 新增算子要写 spec 契约 + 样本 + 六设备 Fragment + 见证 | 配置条目选骨架+实现+参数；内核在 backend 侧 | 实现 `Backend` 对应算子方法，张量 API 自动获得；可写 CubeCL 自定义内核 |
| 图/IR | `graph` 不在新仓独立 crate（并入 engine/exec 缝）；`spec` 出算子级 IR | `graph` crate：`Tensor` enum(F32/Tf32 存 f32，F16/Bf16 存 u16 raw bits)、`Node`/`fuse`/`lower`/`reference`/`sheet` | `burn-ir`、`burn-fusion`：后端无关的算子图与融合 pass |

---

## 4. 后端与执行模型

### 4.1 后端清单与实现状态

| 后端 | llm（新） | llm.u | Burn |
|---|---|---|---|
| CPU | `cpu` Fragment（9 份 `.c` + witness）；`CpuExecutor` 实现在 `engine/exec/cpu.rs`；`gemm/fast` 有 NEON 微核（`vfmaq_laneq`） | `host/cpu`：NEON/AVX2/AVX-512 手写微核 + Accelerate/BLAS 探测 | 双 CPU 后端：`burn-cpu`（CubeCL/LLVM，融合+autotune）与 `burn-flex`（纯 Rust eager，no_std/WASM） |
| NVIDIA CUDA | `cuda` Fragment（9 份 `.cu` + witness）；`CudaBackend` 实现 `Codegen`+`Driver`+`Backend`，`compile_run_cuda` 经 `std::process::Command` 调 **nvcc** 编译并回读；`cuda_lower` 目前只覆盖逐点网格与分块归约，其余骨架未覆盖 | `backend/cuda/cublas/*`（6,238 行）：cuBLAS/cuBLASLT 封装、i8/i4/lowbit/attn/ssm/train，cudarc 直链；CUDA graph 捕获 replay | `burn-cuda`（CubeCL `cuda` feature） |
| Apple Metal | `metal` Fragment（9 份 `.metal`）；**`MetalExecutor` 是唯一已落地的 GPU `Executor` 实现**；`MetalBackend` 三 trait 齐备，`compile_run_metal` 走 `xcrun metal` + metallib | `backend/metal/rt.rs`（643 行）+ `rt/{compute,dev,train}.rs` 共 1,923 行：tensor 路径、hgemm/i8gemm/unary、SSM adapter；`metal` crate 直链 | 经 `burn-wgpu`(MSL) 与 CubeCL |
| AMD ROCm/HIP | `rocm` Fragment（9 份 `.cpp`），**未挂 `Driver`/`Backend` 实现** | `backend/blas/rocm_ops.rs`（49 行，SSM Ops 探测/择优，非完整 GEMM 后端） | `burn-rocm`（CubeCL `hip` feature） |
| Intel OpenCL/SYCL | `opencl` Fragment（9 份 `.cl`），未挂 Driver | `backend/intel/opencl.rs`(862) + `sycl.rs`(483) + BLAS probe；OpenCL/SYCL 实测择优 | 经 wgpu/通用 GPU 路径 |
| 摩尔线程 MUSA | `musa` Fragment（9 份 `.cu` 变体）；`CudaBackend` 复用 `.cu` 源，工具名 `mcc` 走 `musa_ready`/`compile_run_cuda` 链路 | `backend/blas/musa.rs`（596 行，MUBLAS） | 无 |
| 昇腾 CANN/Ascend | `AscendBackend` 三 trait 齐备，`compile_run_ascend` 调 CCE；**文档注明生产态内核仍是空**：P0 片段只到标量 C 语义，未触及 MTE/Cube/Vector，`compile_run` 对非单源返回 `Unsupported` | `Device::Cann` 枚举在位，实现待接入 | 无 |
| Apple ANE/CoreML | 无 | `coreml-native` optional 依赖；`backend/npu.rs` 统一 ANE/NCE/Vitis resident 语义（feature 门控） | 无 |
| 通用 GPU/NPU | `Row::Gpu`（厂商 MMA + 同构 ABI 合同） | `Device::Nce`(Intel NPU, OpenVINO 待接入)、`npu.rs` | 无 |
| 远端/进程间 | `shard::Remote` 占位；`serve/http` 是单线程 HTTP/1.1 薄适配器（进程内可验证） | `config.comm` 声明 `nccl/rccl`，无传输实现 | `burn-remote`：client/server/transport/telemetry/credential，可远程推理与 P2P 训练 |

> 新 llm 的后端分两层：**Fragment**（每设备 9 份原语源 + witness，编译期扫成表）与 **Backend**（`Codegen`/`Driver`/`Backend` 三 trait）。目前三 trait 齐备的是 Metal、Cuda(含 MUSA 工具链)、Ascend；ROCm/OpenCL 仅有 Fragment，未接 Driver。执行器缝（`Executor`）目前只有 `LocalExecutor`/`CpuExecutor`/`MetalExecutor`。

### 4.2 执行缝（trait 面）

**llm（新）**
- `engine::exec::Executor`：`device() / submit(graph, stream) / sync() / register_program() / bind_f32/bind_bytes() / fetch_*()`；`LocalExecutor`、`CpuExecutor`、`MetalExecutor` 三个实现。
- `optim::backend`：`Codegen`（出设备源）+ `Driver`（编译/运行）+ `Backend = Codegen + Driver + Sync`；`register` 静态注册表；`CudaBackend`/`MetalBackend`/`AscendBackend` 已实现三 trait；OpenCL/MUSA/ROCm Fragment 在位，执行缝实现需逐一核对。
- `engine::Program`：按名登记内核体。

**llm.u**
- `host/traits.rs`：`Backend{name,index,sync,weights,compute}`、`Compute{sgemm,sgemm_reuse,sgemm_dtype,trains,teach,dev_mode,sgemm_wt,…}`、`Engine`(H2D/kernel/D2H 三段计时)、`GpuOps`、`Hold`、`Weights`。
- 真实 `Backend` 实现：`Adapter`、`BlasCompute`、`Cuda`、`Fake`/`Packed`(serving 测试)；Metal/Intel/MUSA/ROCm/NPU 是 backend 模块，计算面挂在 `Compute`/`Adapter` 后回退宿主。

**Burn**
- `Backend` trait 族（`BackendTypes` + 张量/张量分布/量化/设备/种子等关联类型与算子方法）；`Autodiff<B,C>` 直接 `impl Backend` 装饰任意后端；CPU/CUDA/WGPU/ROCm/Remote 都是 `Backend` 实现，同一套张量 API 可换后端编译。

### 4.3 内核 DSL 与代码生成

| | llm（新） | llm.u | Burn |
|---|---|---|---|
| 内核抽象 | `cutile`：能力协商 → Fragment 拼接 → emit → 指纹缓存；`spec` 模板词法与代价表达式；六设备 Fragment 编译期扫描成 `FRAGMENTS/PRIMS/DEVICE` 表 | 内核直接写在各 backend 模块；`kernel` crate 只放 attn/ssm 阶段 | CubeCL：设备无关内核语言 + cubek IR；`burn-cubecl` 把 IR  lowering 到 CUDA/HIP/WGPU/CPU；`burn-cubecl-fusion` 算子融合 |
| 自定义内核 | 写 Fragment + 见证，经 `optim::doctor/verify/bench` 对账 | 加 backend 实现 | 写 CubeCL 内核（`examples/custom-cubecl-kernel` / `custom-wgpu-kernel`） |

---

## 5. 自动微分

| 维度 | llm（新） | llm.u | Burn |
|---|---|---|---|
| 独立 autograd crate | 是：`autograd`（`graph/tape/plan/replay/schedule/segment/account/facts`） | 否：反向算子散在 `exec/nn.rs`（`rms_backward/rope_backward/ce_backward/kl_backward`）与 `train/learn/ssm/bwd.rs` | 是：`burn-autodiff` |
| 变量/图模型 | `Var{id}`、`Node{op,inputs,depth,requirement}`、`Role`(parameter/leaf/…)、`Requirement`、`Tape` 记录/校验/深度 | 无变量对象；梯度按层手写反向函数，`grads` 容器装梯度 | `AutodiffTensor` + 反向图 + `AutodiffServer`；`AutodiffBackend` 关联梯度类型 |
| 梯度模式 | tape 记录 → plan → replay；`Session` 调 `plan(loss)` / `backward()`；参数槽 `ParamSlot` | 前向宿主算激活，反向逐层调手写 bwd；`learn` 模块组织反向扫描 | 后端装饰器模式：训练用 `Autodiff<B>`，推理换回 `B`；图自动记录反向 |
| 梯度检查点 | `autograd/segment.rs` 分段 | 无 | `burn-autodiff/checkpoint`：`NoCheckpointing`/`CheckpointStrategy`/retro-forward |
| 分布式梯度 | `Session<K,O,C: Collective>`，`Distributed<C>` 缝；`Remote` 未挂载 | `train` 无分布式梯度；`data::shard` 只做数据分片 | `burn-optim/parallel` + `burn-communication`；P2P 远程训练示例 |

**判定：** 新 llm 的 autograd 端口（变量、tape、plan、replay、tie 绑定、梯度累积快照）已接近框架形态，但它是**仓内自洽的小图**，不接管张量上的任意算子链式求导；Burn 的 autodiff 是后端级装饰器，任意张量算子自动可微，且与检查点策略、分布式梯度连通。llm.u 的反向是"算子级手写 bwd"集合，没有图抽象。

---

## 6. 模型层（nn）

### 6.1 层清单对比

| 类别 | llm（新）`nn` | llm.u `exec/nn` + `kernel` | Burn `burn-nn` |
|---|---|---|---|
| 线性/嵌入 | `Linear` `Embedding` `LmHead` `MatMul` | `matmul`、权重按名识别（`mlp.fc1/fc2`、`mixer.in/out_proj`、`embedding/lm_head`） | `Linear` `Embedding`；`ParamGroup`/LoRA 参数映射 |
| 归一化 | `LayerNorm` `BatchNorm2d` `GroupNorm` `InstanceNorm` `RmsNorm` | `rms/rms_in`（前向+反向） | `BatchNorm` `GroupNorm` `InstanceNorm` `LayerNorm` `RmsNorm` |
| 卷积 | `Conv1d` `Conv2d` `ConvTranspose1d` `ConvTranspose2d` | **无 conv** | `Conv1d/2d/3d` `ConvTranspose1d/2d/3d` `DeformConv2d` |
| 池化 | `MaxPool1d/2d` `AvgPool2d` `AdaptiveAvgPool2d` `Interpolate` | **无 pool** | `MaxPool1d/2d/3d` `AvgPool1d/2d/3d` `AdaptiveAvgPool1d/2d/3d` `Interpolate1d/2d` |
| 注意力 | `Attention` `AttentionBlock` `LatentAttention` `MultiHead` 语义在 `algorithm` | `attention` 家族（cpu/paged/kv4/kv_f16）、`attn/cpu` | `MultiHeadAttention` `CrossAttention` + MHA/Cross cache、`TransformerEncoder/Decoder` 及层、autoregressive cache |
| 激活 | `Gelu` `Silu` `LeakyRelu` `HardSigmoid` `Softmax` | `act.rs`； sparsity/gamma/pi 门控函数 | 全套激活 + `Dropout` `GaussianNoise` |
| 循环/序列 | 无 RNN | SSM/Mamba stage（`kernel/ssm`、`train/learn/ssm`） | `Rnn` `Gru` `BiGru` `Lstm` `BiLstm` `BiRnn` |
| 位置编码 | `Rope` `PositionalEncoding`(语义) | `rope/rope_backward` | `RotaryEncoding` `PositionalEncoding` |
| 其他 | `ResidualAdd` `Stack` `ReduceSum/Max` `Scale` `Bypass` `MemoryBlock` | `dot` `argmax` `ce` `kl` | `Fold4d` `Unfold4d` `PixelShuffle/Unshuffle` `CosineSimilarity` `PairwiseDistance` `Dropout` `Noise` |

### 6.2 模型架构覆盖

- **llm.u**：定位 LLM/SSM 推理。架构变体以权重名模式识别（MLP / Exposed mixer / Vocab / State），SSM 前向扫描 + 反向 scan 在位；无 CNN/ViT/RNN 层。`config.spec` 里有 `vit` 字样但属配置枚举，未落地视觉栈。
- **llm（新）**：`algorithm` crate 参考语义覆盖 `attnres / csa / kda / mhc / mla / mamba / moe / hybrid / mix / latent / spike / patch / policy / gating`；`nn` 补齐了经典 CNN/归一化/池化；`spec` 是算子契约层而非预训练模型库。
- **Burn**：`burn-nn` 是通用层库，配合 `burn-vision`、示例（mnist、modern-lstm、text-classification、text-generation、dqn、wgan）可直接搭可训练模型；不含 Llama/Qwen 等预训练权重加载模型定义（有 `pytorch-reader` 导权重）。

### 6.3 模块组织方式

- **Burn**：`#[derive(Module, Debug)]` + `#[derive(Config, Debug)]`；参数 `Param<Tensor>`，支持 `ModuleMapper` 遍历改参数（量化/冻结/LoRA）；`burn-core` 提供参数树、`ParamId`、参数组。
- **llm（新）**：层是持有 `ContractSpec`/算子引用的结构体，经 `nn::Contract` 绑定算子名与形状边界；参数以 `ParamSlot`/扁平 `[f32]` 槽管理，无派生宏。
- **llm.u**：层函数直接作用于 `&[f32]` 扁平切片，权重由 `exec/weights` 按名装配。

---

## 7. 量化与精度

| 维度 | llm（新） | llm.u | Burn |
|---|---|---|---|
| 存储格式 | `dtype`：GGML 块量化 QType、MX 微缩放 MxType、MX 元素闭集；`weights/ptq.rs` 定义 `Scheme{I2,I4,I8,Compensated}` | `Prec`：I8/I4/I3/I2 + F8E4M3/F8E5M2/Fp4；AWQ 权重缓存路径 | `QuantValue`：Q8F/Q8S/Q4F/Q4S/Q2F/Q2S + E4M3/E5M2/E2M1；`PackedU32` 打包 |
| 量化落点 | 权重档案（`weights/ptq`、`adaptive`、`settle`、`quant`）；算子侧引用块格式 | CPU `awq.rs`/`lowbit.rs`、CUDA `int4/int8w/lowbit/subbyte`、Metal i8；AWQ I4 缓存为 f32 Bᵀ | 模块参数级 `Quantizer`（`ModuleMapper` 按路径/参数组映射），`Calibration` + `QuantScheme` 算 qparams |
| PTQ 校准 | `weights` 侧自适应/settle 流程 | 配置档位 + 后端实现 | `Calibration`（range 观测）+ scheme（symmetric/affine，F/S 家族） |
| QAT（量化感知训练） | 无 | 无 | 无（当前版本；量化为 PTQ 参数映射） |
| 低精度计算 | TF32 档在 llm.u 显式；新仓 `prec`/`dtype` 承接 | Tf32 计算档；f16/bf16 端到端；i8 逐元素对拍精确 | 后端按 `FloatDType` 能力选；FP8/FP4 经 QuantValue |

**判定：** 三家都**没有 QAT**。llm.u 与新 llm 的量化偏"权重存储格式 + 后端 GEMM 解包"，Burn 的量化偏"参数体系内可映射的量化张量"。块量化格式广度新 llm 最大；模块级量化工具链 Burn 最完整。

---

## 8. 训练栈

### 8.1 优化器

| | llm（新） | llm.u | Burn |
|---|---|---|---|
| 优化器 trait | `Optimizer`（`Slot/OwnedSlot`、weight decay、clip_norm） | 自由函数 + `Adam` 结构 | `Optimizer` trait（module 级参数映射 + state） |
| 已实现 | `Sgd` `Adam` `Muon`(+momentum/Nesterov/eps) | `sgd` 函数、`Adam`(lr/wd/step) | `Sgd` `Adam` `AdamW` `Adagrad` `RmsProp` `RmsPropMomentum` `Lamb` `Adafactor` `Lion` `Adan` `Lbfgs` `Muon` `Momentum` + 梯度累积 `grad_accum`、梯度裁剪 `grad_clipping`、衰减 `decay` |

### 8.2 学习率调度 / 训练循环

| | llm（新） | llm.u | Burn |
|---|---|---|---|
| 调度 | `Schedule`：warmup + total + `Anneal` + min_ratio | `Cosine`：base/warmup/total/min；`Accum` 梯度累积 | `burn-optim/lr_scheduler` 多策略；Learner 接管 epoch/step |
| 训练会话 | `Session<K,O,C>`：注册参数、tie 绑定、plan/backward/optimize、微批累积、loss 记录、`TrainConfig`；`Learner<K,O,C>` 组织 `run`/`evaluate` + `MetricsBoard` 账表（loss/命中/样本数/训练步，训练与验证分账） | `Session`（Model/lr/warmup/total/min/clip/gclip）+ `Trainer`（observe/accumulate/step/loss/mean） | `Learner`/`LearnerBuilder`：train/val 循环、early stopping、sharder、logger、summary、RL 子回路 |
| 梯度裁剪/累积 | Optimizer clip_norm；`accum_snapshot` 微批 | `loss::clip` 按 max_norm；`Accum` | `grad_clipping` 模块；`grad_accum` |
| 损失 | `cross_entropy` `argmax_hits` `mse` `bce_with_logits` `huber` `hinge` `kl_div` `kl_per_row` `distillation_loss` `log_softmax`；`Loss` 枚举分发；`Metric` 入账终点含 Distill/Grpo | `sequence`(CE) `perplexity` `bits` `mse` `entropy` `distill` `clip` | 20 种：CE、MSE、BCE、KL div、CTC、RNNT、Huber、SmoothL1、Triplet、MarginRanking、CosineEmbedding、HingeEmbedding、GaussianNLL、Poisson、MultiMargin、SoftMargin、LP、BCE… |
| 指标 | 最小账表：loss、命中率（argmax hits）、样本数、步；`MetricsBoard` 训练/验证分账；无可插拔指标注册表 | perplexity 函数；无指标体系 | `burn-train/metric`：acc/top-k/MAE/MSE/R²/precision/recall/F1/MCC/AUROC/AUC-PR/confusion/bleu/cer/wer/rouge/hamming/perplexity/loss + CPU/GPU/内存/温度采集 + 处理器与渲染器 |
| 评估器 | `Learner::evaluate(dyn Dataset, forward)` 独立验证批入账；无可插拔 evaluator/renderer | 无 | `burn-train/evaluator`：numeric/text 指标注册、renderer、文件日志 |
| 检查点 | `snapshot` crate：自描述编码 + CRC-32 trailer + 恢复事实；`train/store` | `train/checkpoint`：状态字节格式 | `burn-train/checkpoint` + `burn-store`(adapter/applier/collector/filter/keyremapper/burnpack/pytorch) |
| 数据集 | `train/samples`：`Batch`/`Dataset` trait/`Chunks` 游标；无打乱/打包/物理数据源 crate | `train/data`：pack/shard | `burn-dataset`：audio/nlp/source/transform/vision + CSV/图像/音频源 |
| RL | `train/rl`：log_prob/entropy/advantage/policy_loss/GRPO（裁剪策略梯度 + 参考 KL） | `train/rl` 桥 | `burn-rl`：environment/policy/transition_buffer；DQN/WGAN 示例 |

**判定：** 训练栈深度排序 **Burn ≫ 新 llm > llm.u**。新 llm 已补齐优化器 trait、Muon、调度、tie、会话与一批 loss，但没有指标注册表、evaluator、数据集 crate；llm.u 是"能跑通小训练循环"的最小集；Burn 是完整训练框架（指标/评估/检查点/数据集/分布式/RL 全在位）。

---

## 9. 推理服务与运行时

### 9.1 调度 / 连续批 / KV 缓存

| | llm（新） | llm.u | Burn |
|---|---|---|---|
| 调度器 | `engine/scheduler`、`ifb`、`batch`、`sequence`、`radix` | `engine/scheduler`：Request/Batch/Phase/Stats、优先级、salt、extra_key、cancel/abort、页池 | 无内置 LLM 服务调度（`examples/server` 为推理示例） |
| KV 缓存 | `engine/kv` | `engine/kv`：`page` 页表、`radix` 前缀树、`compress`、`latent` | `nn` 侧有 MHA/Cross/Transformer autoregressive cache；无服务级分页前缀共享 |
| 连续批 | `engine/ifb` + `serve/admit` | `scheduler.step/commit` 连续批 | 无 |
| 投机采样 | `serve/speculate` | `engine/scheduler/speculative`；`exec/teacher`(math/mod) | 无 |
| 端点契约 | `serve/endpoint`：Route/Status/Response/Endpoint；`protocol`；`http`（单线程 HTTP/1.1 薄适配器，socket 字节解成方法+路径+请求体交 Router）；`completion/frame/request/model/params` | `serve`：openai schema(chat/completions)、`route` 纯逻辑路由；**HTTP 层未接 axum/tokio（serve 零依赖），router/shutdown 待实现**；transport/assemble/detokenize/ensemble/health/admin/facade | `examples/server`；`burn-remote` 提供远程推理 client/server |
| 权重格式 | `weights`：safetensors(header/whole-buffer/mmap)、GGUF、JSON、torch `.bin`(ZIP+pickle 栈机)；自有 `.u` 容器 | `storage`：gguf/safetensors/legacy/tans/merkle/sign/cert/patch/lattice；`exec/format`：HF `config.json`/tokenizer、zip 状态字典(pickle 协议 2)、pytorch_model.bin → `.u` 归档 | `pytorch-reader` + `burn-store`/burnpack |
| 分词 | `tokenizer` crate：Tokenizer trait + 字节级参考实现 | `proto/decode` 采样；`serve/token/detokenize` | 无内置分词器 |

### 9.2 服务能力判定

- llm.u 与新 llm 都面向 LLM 在线推理：连续批、分页 KV、前缀共享、投机、OpenAI 兼容 schema；这是 Burn 当前不覆盖的产品面。需要注意：llm.u 的 HTTP server 尚未接异步运行时（路由是纯逻辑），新 llm 的 `http` 是可经真实 socket 验证的薄适配器；两者的 OpenAI 兼容度都停在 schema/契约层，非完整生产网关。
- llm.u 的服务态更"实"（scheduler/engine 有完整请求生命周期）；新 llm 的服务拆成 `engine`（调度/KV/批）与 `serve`（端点/线格式/装配）两层，契约更干净。

---

## 10. 分布式与并行

| 维度 | llm（新） | llm.u | Burn |
|---|---|---|---|
| 拓扑 | `shard/topology`：world size、per-axis 进程组 | `config.tables.parallel`：切分配置；`data::shard` 数据分片 | `burn-shard`：sharding/kind/resharding/rules/backend |
| 切分类型 | `Sharding`(axis) + `Tiling`(parts/inp/out) + `Reduce`(Sum/Mean/Max…) | 配置声明 `comm: nccl/rccl`；无切分实现 | `burn-shard` 规则与重切分 |
| 集合通信 | `Collective` trait（`all_reduce` + `Group`）；`Local` 进程内 reduce；**`Remote` 占位，返回 `Unsupported`** | 无 Collective trait；`comm` 仅协商标签 | `burn-communication`：`Protocol/ProtocolClient/ProtocolServer/CommunicationChannel` + WebSocket 实现；`burn-remote` 远程训练/推理 |
| 并行范式 | TP/PP/DP 拓扑数据结构在位，传输未挂载 | 数据分片函数；无模型并行 | `burn-train/learner/sharder` + 通信协议；layer-parallelism 示例；P2P 远程训练示例 |

**判定：** 三家的跨进程集合通信都未接 NCCL/RCCL/MPI 真实栈；Burn 有可运行的 WebSocket/remote 通道与 shard 规则，新 llm 有完整拓扑/切分/Reduce 数据结构但通信后端是占位，llm.u 仅有配置标签。

---

## 11. 随机数与确定性

| | llm（新） | llm.u | Burn |
|---|---|---|---|
| RNG crate | `rng`：Philox4x32-10 counter 引擎，显式 seed/stream/index；`Uniform`/`Normal`/`Bernoulli` | 无独立 RNG；采样由外部注入闭包 `impl FnMut() -> f64`；测试用线性种子伪随机 | `rand` 生态；`Device::seed(u64)`；张量 `Distribution::Uniform/Normal/…` |
| 可复现性 | counter-based，分 stream/index，可复现 | 依赖注入闭包，确定性由调用方冻结种子 | 后端种子语义（可能全局生效） |

---

## 12. 算子与内核覆盖（按家族实测）

### 12.1 llm.u 已注册算子家族（`exec/reg.rs`）

- `matmul`：`matmul` / `matmul_blas` / `matmul_bt` / `matmul_ref` / `matmul_quant`
- `elemwise`：`elemwise`
- `attention`：`attention` / `attention_kv4` / `attention_kv_f16`
- `reduce`：`reduce`

后端能力（`Compute`）：sgemm / sgemm_reuse / sgemm_dtype(TF32) / 常驻权重 sgemm_wt/wk / adam 设备端 / fetch；`trains()`/`teach()`/`dev_mode()` 能力位；SSM 前向扫描与 prep/dot/scan/turn/bc 反向。

### 12.2 新 llm 算子四层

- `spec`：算子签名、样本、语义表、代价表达式、模板词法、einsum 求和推导。
- `gemm`：matmul/gemv 家族、激活、归一化、量化；`fast` 微核（i8/micro/pool/probe/sme）。
- `conv`：卷积/反卷积、注意力、池化、归一化层。
- `op`：声明解析 + 唯一注册表 `REGISTRY`。
- `algorithm`：架构参考语义（attn/csa/kda/mhc/mla/mamba/moe/hybrid/mix/latent/spike/patch/policy/gating）。

### 12.3 Burn 算子面

张量 API 覆盖：线性代数（matmul/einsum/linalg）、卷积/池化/归一化、激活、归约、逐元素、形状变换、采样、量化张量；CubeCL 提供自定义内核扩展点。

---

## 13. 工程质量与可维护性

| 维度 | llm（新） | llm.u | Burn |
|---|---|---|---|
| 分层约束 | L1–L6 契约由 `testkit/architecture.rs` 读 manifest 强制校验 | 无显式分层门禁；device/exec 耦合 | crate 依赖约定，文档化 |
| 文档 | `ARCHITECTURE.md`（分层/依赖/所有权/闭集/占位策略）、`lld.md`、`math.md`、`docs/adr/` | `README.md`（大量性能实测记录）、`docs/perf.md`、`QUALITY.md` | `burn-book`、`contributor-book`、examples |
| 质量门 | `tools` crate：配置驱动静态质量门；`diagnostic` 错误码；`annotation` 契约宏 | `tool/gate`：arch/call/ccn/files/rules/scan/text；clippy 全 warn | CI/coverage/typos/deny.toml |
| 模糊测试 | `fuzz` crate（contract：种子/目标 trait/不变量），`libfuzzer-sys` optional | `fuzz` 目录（独立 nightly 运行） | 后端测试套件 `burn-backend-tests` |
| 基准 | `benchmarks` crate（attention/io/kernel/model/peak/protocol） | `benches`：CPU/Metal/OpenCL/CLBlast/SYCL/CUDA/MUSA 多后端实测 + 对拍 | `benchmarks.toml` + criterion |
| 错误处理 | `diagnostic::Error/Code`（含 Unsupported） | `proto::error::Err/Result` | 后端错误 trait |
| 工具链 | edition 2024 / rust 1.99；resolver 3；release thin LTO + codegen-units 1 | resolver 2；release thin LTO + codegen-units 1 | 常规 MSRV 策略 |

---

## 14. 能力矩阵总表

图例：● 真实实现可用；◐ 端口/骨架在位，部分实现或占位；○ 缺失。

| 能力维度 | llm（新） | llm.u | Burn |
|---|:--:|:--:|:--:|
| CPU 内核（手写 SIMD 微核） | ◐ gemm/fast NEON 微核在位，AVX 路径待补 | ● NEON/AVX2/AVX-512/Accelerate 实测 | ◐ CPU 后端可用，非极致调优 |
| CUDA 后端 | ◐ nvcc 编译回读链路通；lower 仅逐点/归约，GEMM 大内核待接 | ● cuBLAS/LT 封装 + graph capture | ● CubeCL CUDA |
| Metal 后端 | ● MetalExecutor 落地；xcrun 编译链路通 | ● runtime/tensor 路径实测 | ◐ 经 wgpu/MSL |
| ROCm/HIP | ○ 仅 Fragment，无 Driver | ◐ 49 行 SSM Ops 探测 | ● CubeCL HIP |
| OpenCL/SYCL | ○ 仅 Fragment，无 Driver | ● OpenCL 862 + SYCL 483 实测择优 | ○ |
| MUSA | ◐ 复用 CUDA 源 + mcc 工具链 | ● MUBLAS 596 行实测 | ○ |
| 昇腾 CANN | ◐ CCE 编译回读；内核仅标量语义，并行核函数未落地 | ◐ 枚举在位 | ○ |
| ANE/CoreML | ○ | ◐ optional，feature 门控 resident 语义 | ○ |
| 张量秩/后端泛型 | ○ | ○ | ● |
| 运行期形状/精度校验 | ● spec/testkit | ● config 校验 | ● 后端能力协商 |
| 自动微分（图/tape） | ◐ 仓内小图 | ○ 手写反向 | ● 后端装饰器 |
| 梯度检查点 | ◐ segment | ○ | ● 策略可配 |
| 优化器丰富度 | ◐ Sgd/Adam/Muon | ◐ Sgd/Adam | ● 13+ 含 AdamW/Lamb/Lion/Adafactor… |
| LR 调度 | ● warmup+anneal | ● cosine+warmup | ● 多策略 |
| 指标注册表/evaluator | ◐ Learner 账表 + evaluate，无可插拔指标/渲染器 | ○ | ● 30+ 指标 |
| 数据集 crate | ◐ Dataset trait + Chunks 游标，无物理源 | ○ | ● |
| 检查点体系 | ● snapshot | ○ checkpoint | ● store/checkpoint 完整 |
| CNN 卷积层 | ● | ○ | ● |
| 池化层 | ● | ○ | ● |
| RNN/GRU/LSTM | ○ | ○ | ● |
| Transformer 模块 | ◐ 注意力/算法语义 | ◐ 注意力 KV | ● Encoder/Decoder + cache |
| SSM/Mamba | ● algorithm/kernel | ● 前反向 | ○ |
| MoE 路由 | ◐ algorithm/gating | ○ | ○ |
| 量化格式广度 | ● GGML+MX+FP4 | ◐ I2–I8+FP8/FP4 | ● Q8/Q4/Q2+FP8/FP4 |
| 参数级量化映射 | ○ | ○ | ● ModuleMapper/Quantizer |
| QAT | ○ | ○ | ○ |
| 连续批调度 | ● engine/ifb | ● scheduler | ○ |
| 分页 KV + 前缀共享 | ● radix/page | ● radix/page | ○ 序列 cache |
| 投机采样 | ● speculate | ● speculative | ○ |
| OpenAI 兼容端点 | ● route/protocol | ● schema/server | ○ 示例 server |
| 权重格式读取 | ● safetensors/GGUF/JSON/torch.bin | ● gguf/safetensors | ○ pytorch-reader |
| 内置分词器 | ● tokenizer | ○ | ○ |
| 拓扑/切分数据结构 | ● shard | ◐ 配置 | ● shard rules |
| 跨进程集合通信 | ○ Remote 占位 | ○ 标签 | ◐ WebSocket/remote |
| 确定性 RNG | ● Philox | ○ 注入闭包 | ● rand/seed |
| 自定义内核 DSL | ● cutile | ○ 直接写后端 | ● CubeCL |
| 模块派生宏（Module/Config） | ○ | ○ | ● derive |
| RL 栈 | ◐ log_prob/entropy/advantage/GRPO | ○ 桥 | ● env/policy/buffer |
| 远程训练/推理 | ○ | ○ ● remote |

---

## 15. 关键差距与风险（按优先级）

### 15.1 新 llm 相对 Burn 的主要缺口

1. **执行面未闭环**：`Executor` 落地 CPU/Local/Metal 三个；CUDA/MUSA/Ascend 有真实编译驱动（nvcc/mcc/CCE）但算子 `lower` 覆盖有限（CUDA 仅逐点网格 + 分块归约，Ascend 仅标量 C 语义）；ROCm/OpenCL 只有 Fragment，未挂 `Driver`/`Backend`。Burn 同一套 API 可在 5+ 后端跑通完整算子集。
2. **类型系统层级低**：张量无后端/秩泛型，编译期无法拦截后端不支持的算子；形状与 dtype 错误要到校验/运行期暴露。
3. **训练框架不完整**：有最小 Learner/evaluate/账表与 Dataset trait，但无可插拔指标注册表、evaluator/renderer、物理数据集 crate、模块参数树与派生宏；autograd 不覆盖任意张量算子链式求导。
4. **集合通信未挂载**：`Remote::all_reduce` 直接 `Unsupported`，无法做真实 DP/TP 训练。
5. **模型库薄**：层定义与算法参考语义为主，没有可直接训练/推理的端到端模型装配与权重例程（对照 Burn 的 nn 模块 + examples）。

### 15.2 新 llm 相对 llm.u 的演进与遗留

**已补齐**：独立 autograd、rng、conv/pool、Muon、shard 拓扑/Collective 缝、snapshot/weights/formats/tokenizer 独立 crate、算子四层契约、cutile DSL 与后端 Fragment 化、分层契约测试、测试体量 3.2×。

**遗留/回退风险**：
- llm.u 的多后端**实测性能数据与对拍**沉淀在 README/docs，新仓需在重构后重新建立基准门禁（`benchmarks` 已搭骨架）。
- llm.u 直链 cudarc/metal/coreml-native，后端 API 面宽；新仓改 DSL 生成后，设备源能力（如 cuBLASLT 调用、Metal tensor 参数）要在 Fragment/Driver 层重新表达。
- llm.u 的 CUDA graph capture、pinned staging、常驻权重复用等推理热路径优化，需在新 `engine/serve/device` 链路上重新落地。

### 15.3 llm.u 相对两者的定位

- **优势**：多后端 GEMM/注意力实测调优最深（CPU 微核、Metal tensor、OpenCL/SYCL 择优、CUDA graph、MUSA 对拍）；推理调度/KV/前缀共享完整；代码量集中、路径直接。
- **劣势**：无图抽象的手写反向、无独立 RNG、无 conv/pool、无分布式通信、unsafe 分散、无分层门禁；训练与指标能力最小化。

---

## 16. 取证口径与可复核命令

统计命令（在各仓库根目录执行）：

```bash
# 源文件数（剔除构建产物）
find crates -name '*.rs' -not -path '*/target/*' | wc -l
# 生产行
find crates -name '*.rs' -not -path '*/target/*' -not -path '*/tests/*' | xargs wc -l | tail -1
# 测试行
find crates -path '*/tests/*' -name '*.rs' -not -path '*/target/*' | xargs wc -l | tail -1
```

- llm：`/Users/fei/code/llm`，HEAD `f6b46e76`
- llm.u：`/Users/fei/code/llm.u`，HEAD `cf4e8fe3`
- Burn：`/Users/fei/code/burn`，`0.23.0-pre.2`，HEAD `aff0a3cf`

> 注：Burn 的 `burn-cuda/burn-cpu` 壳层极薄是架构使然——重内核在 CubeCL IR 与 `burn-cubecl`；比较后端能力时应以"Backend 实现 + 内核 lowering 链"整体计，而非壳层行数。llm/llm.u 的后端代码直接计入各自 crate，口径不同，跨仓行数只作量级参考。

# `llm` / `llm.u` / Burn 全维度源码级对比分析

> 基线：三份代码均按源码同口径实测（剔除
> `target/`
> 构建产物与 tests 目录后统计生产行）。



| 代号            | 路径                      | 版本 / HEAD                                                | 角色                          |
| ------------- | ----------------------- | -------------------------------------------------------- | --------------------------- |
| **llm**（新仓）   | `/Users/fei/code/llm`   | workspace，32 个成员 crate（含 `frag`；`formats` 已并入 `weights`），HEAD `40b18f36`         | 重构后的分层契约架构（L1–L6）           |
| **llm.u**（旧仓） | `/Users/fei/code/llm.u` | workspace，14 个成员 crate（`fuzz` 被 exclude），HEAD `cf4e8fe3` | 配置驱动、后端直连的执行框架              |
| **Burn**      | `/Users/fei/code/burn`  | `0.23.0-pre.2`，HEAD `aff0a3cf`                           | 成熟通用 Rust 深度学习框架（Tracel AI） |

## 0. 结论先行



1. **三者不是同一物种。** Burn 是面向训练 / 研究的通用张量框架（后端泛型 + 自动微分 + 优化器 + 指标 + 数据集 + RL 一整套）；`llm.u` 是一个**面向 LLM 推理 / 小训练循环、多后端 GEMM 性能优先**的执行框架；新 `llm` 是对 `llm.u` 的一次**架构重构**：把 "后端直连" 拆成 "契约层 → DSL/emit → 后端 Fragment → 执行缝" 的六层体系，能力端口向 Burn 看齐（补 autograd crate、conv/pool、rng、muon、shard 拓扑与 Collective 缝），但绝大多数执行面仍是骨架 / 占位。

2. **规模口径（源码实测）**



|        | crate 成员 | `.rs` 源文件 | 生产行     | 测试行    |
| ------ | -------- | --------- | ------- | ------ |
| llm（新） | 32       | 707       | 95,602  | 58,007 |
| llm.u  | 14       | 349       | 86,735  | 17,857 |
| Burn   | 40       | 1,851     | 381,556 | 95,111 |

Burn 生产代码约为新 llm 的 **4.0×**、llm.u 的 **4.4×**；测试代码为 1.6× / 5.3×。（新 llm 工作区存在未提交改动，行数以取证时点为准，`cargo test` 全绿后可复跑 §17 命令刷新。）



1. **关键差距不在 "有没有目录"，而在 "缝后面是不是真实现"。** 新 llm 的 `Executor`/`Collective`/`Backend` trait 端口齐全，但 GPU 执行器目前仅 Metal 一个真实实现，远端集合通信返回 `Unsupported`；Burn 的端口后面是跨 CPU/CUDA/WGPU/ROCm 的可运行内核与可训练模型。

2. **llm.u 的长板是多后端实测性能**（CPU NEON/AVX2/AVX-512、Metal、OpenCL/SYCL、CUDA cuBLAS、MUSA 均有 GEMM 数据与对拍；ROCm 仅 SSM 内核库加载 / 探测，非完整 GEMM 后端），这一点 Burn 的 CPU/CUDA 后端不以极致微核调优为目标；新 llm 把 llm.u 的微核沉淀进 `gemm`/`conv`/`cutile`，但后端执行面尚在重建。



***

## 1. 仓库定位与架构哲学

### 1.1 llm.u：配置驱动的算子执行框架



* 自述："算子不写代码文件。整个框架只有算法骨架，所有算子都是配置文件里的一个条目：选骨架 + 选实现 + 传参数。"

* 14 个成员：`benches / cli / config / device / engine / exec / fuzz / graph / kernel / proto / serve / storage / tool / train`。

* 结构特点：`device`（28,325 生产行，占全仓近 1/3）直接持有各后端的 FFI 调用与内核封装；`exec` 是算法 / 模型 / 权重 / 采样的宿主；`train` 是紧贴推理栈的小训练循环。

* 后端与宿主代码强耦合，trait（`Backend`/`Compute`/`Engine`/`Gpu`/`Hold`/`Weights`）以 GEMM、注意力、SSM 前反向为中心裁剪，不是通用张量算子集。

### 1.2 llm（新）：六层契约架构，可被测试守住的分层

`docs/ARCHITECTURE.md` 定义了可执行契约（`testkit/src/architecture.rs` 读 `cargo metadata` 与源码逐条核对，不一致即 `cargo test` 失败）：



| 层        | 代表 crate                                                                  | 职责                                                                     |
| -------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| L1 基础设施  | `diagnostic` `rng` `dtype` `wire` `formats` `tools`                       | 错误类型、确定性随机、精度闭集、字节码闭集、外部格式只读适配                                         |
| L2 数据层   | `prof` `spec` `gemm` `conv` `op` `cpu/cuda/metal/musa/opencl/rocm` `prim` | 算子四层契约 + 六份设备原语片段与见证                                                   |
| L3 内核    | `cutile` `optim`                                                          | 协商→拼接→emit→指纹（cutile，不执行）；doctor/verify/bench/execution（optim，碰编译器与设备） |
| L4 运行时   | `tensor` `storage` `tokenizer`                                            | 张量元数据、布局搬运、文本边界                                                        |
| L5 模型与训练 | `weights` `snapshot` `autograd` `nn` `algorithm` `shard`                  | 权重结构、检查点、反向图、层定义、架构参考语义、切分拓扑                                           |
| L6 调度与服务 | `train` `engine` `device` `serve`                                         | 训练循环、连续批 / KV 页账、提交执行体、设备源 lowering、端点契约                               |



* 依赖方向单向：L6 不得直接调 `cutile` 生成 kernel；L2 不得出现机制代码；设备厂商特判只允许出现在后端 Fragment。

* 与 llm.u 相比：新增 `autograd / rng / spec / gemm / conv / op / prim / prof / cutile / shard / snapshot / weights / formats / tokenizer / benchmark` 等独立 crate，把 llm.u 里糊在 `device`/`exec` 里的关注点拆开。

### 1.3 Burn：通用张量框架的经典分层

40 个 crate 按 "张量核心 → 后端 → 模型 → 训练 → 服务 / 分布式" 组织：



* 核心：`burn-core`（Module/Param/Quantizer/Store）、`burn-tensor`（`Tensor<D,K>` API）、`burn-backend`（`Backend` trait 族）、`burn-autodiff`（自动微分后端装饰器）。

* 后端：`burn-cpu`（CubeCL CPU 运行时，LLVM 编译，带融合 /autotune）、`burn-cuda`、`burn-wgpu`、`burn-rocm`、`burn-cubecl`/`burn-cubecl-fusion`（CubeCL 内核 IR 与融合）、`burn-flex`（纯 Rust eager 后端，支持 no\_std/WASM）、`burn-remote`。

* 模型：`burn-nn`（8 大类模块、78 个 Config 结构）、`burn-linalg`、`burn-einsum`、`burn-vision`、`burn-flex`。

* 训练：`burn-optim`、`burn-train`（Learner/metric/evaluator/checkpoint）、`burn-dataset`、`burn-rl`、`burn-communication`。

* 工具：`burn-derive`（`#[derive(Module, Config)]`）、`burn-fusion`、`burn-instrument*`、`burn-shard`、`burn-signal`、`burn-pack`、`pytorch-reader`。

**哲学差异：** Burn 的抽象单元是**带后端泛型的张量**（`Tensor<const D, K>`），一切算子挂在张量方法上；llm.u 的抽象单元是**配置条目 + 算法骨架**；新 llm 的抽象单元是**算子契约（spec/op）+ 设备 Fragment + 执行缝**，张量是 `Tensor<B, R, K>`（存储 / 秩 / 元素三泛型），但只做无执行的布局代数。



***

## 2. 规模与工程度量

### 2.1 代码体量（源码实测，已剔除 target）



| 维度           | llm（新） | llm.u            | Burn    |
| ------------ | ------ | ---------------- | ------- |
| workspace 成员 | 32     | 14（fuzz exclude） | 40      |
| `.rs` 文件     | 707    | 349              | 1,851   |
| 生产行          | 95,602 | 86,735           | 381,556 |
| 测试行          | 58,007 | 17,857           | 95,111  |
| 测试 / 生产      | 0.61   | 0.21             | 0.25    |

新 llm 生产行与 llm.u 接近，但测试行是 llm.u 的 3.2 倍 —— 重构后把大量契约 / 边界校验沉进了测试（`testkit` 4,059 行、各 crate 自带 tests）。

### 2.2 生产行 Top crate

**llm（新）**：`gemm` 8,090 · `weights` 7,041 · `train` 7,006 · `device` 6,973 · `algorithm` 5,820 · `serve` 5,566 · `optim` 5,264 · `testkit` 4,059 · `cutile` 3,886 · `cli` 3,741 · `conv` 3,712 · `spec` 3,443 · `prof` 3,292 · `engine` 3,148 · `nn` 3,031。

**llm.u**：`device` 28,325 · `exec` 15,561 · `storage` 9,290 · `train` 7,581 · `tool` 5,765 · `config` 4,638 · `benches` 4,772 · `engine` 4,083 · `kernel` 1,451 · `serve` 893 · `proto` 1,048 · `graph` 818。

**Burn**（节选）：`burn-train` 33,088 · `burn-nn` 29,463 · `burn-tensor` 24,740 · `burn-autodiff` 12,341 · `burn-core` 11,578 · `burn-optim` 10,768；后端壳层很薄（`burn-cuda` 58、`burn-cpu` 61、`burn-wgpu` 253），重内核在 CubeCL IR 与 `burn-cubecl` 侧。

### 2.3 第三方运行期依赖



|       | llm（新）                                                                                               | llm.u                                                                                                                                                                                                                                                             | Burn                                                              |
| ----- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 运行期三方 | `bytemuck`(optional)、`half`(workspace)、`serde`/`serde_json`(optional)、`libfuzzer-sys`(optional, dev) | `cudarc 0.19`(optional)、`metal 0.33`(optional)、`coreml-native 0.2`(optional)、`libloading 0.8`(OpenCL/SYCL/CLBlast/MUSA/ROCm/NCE/Vitis 动态加载)、`half 2`、`faer 0.24`(storage)、`rand 0.9`/`rand_distr 0.5`(tool/proto 侧初始化)、`serde/serde_json`、`clap`、`criterion`(dev) | `rand`、`half`、`bytemuck`、`cubecl`、`wgpu`、`cudarc`（经 CubeCL）等一整套生态 |



* 新 llm 刻意把外部依赖压到最低：后端 FFI 不直链 `cudarc/metal`，而是走 `cutile` DSL 生成设备源（`.cu/.metal/.cl/.cpp`）+ `optim` 的 `Driver` 编译执行；这是与 llm.u 最大的工程转向。

* llm.u 直接链 `cudarc`/`metal`/`coreml-native`，后端能力受这些 crate 的 API 面约束。

### 2.4 unsafe 边界



|        | 真实 `unsafe {}` 块所在文件（生产）                                                                                                          | 边界策略                                                                                     |
| ------ | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| llm（新） | 6 个岛：`storage/mmap.rs`、`gemm/fast/{i8,micro,sme}.rs`、`device/metal/executor.rs`、`device/metal/executor/device.rs`                 | workspace `unsafe_code = "deny"`，仅显式放行；文档称 Metal FFI 为唯一 unsafe 岛（ADR 0035），SIMD 岛在 gemm |
| llm.u  | 49 个文件：CPU SIMD / 内存、CUDA cuBLAS、Metal runtime、OpenCL/SYCL、MUSA、BLAS adapter、storage plane/store、benches OpenCL/CLBlast、tool gate | unsafe 分散在各后端 FFI 与 SIMD 路径，未集中收口                                                        |
| Burn   | 后端内 FFI  unsafe 集中在 CubeCL / 后端 crate；`burn-core/tensor/nn` 业务层无 unsafe                                                           | 业务类型安全，unsafe 下沉到后端实现                                                                    |



***

## 3. 张量模型与类型系统

### 3.1 张量抽象



| 维度       | llm（新）                                                                                                                          | llm.u                                                                                                | Burn                                                                                                    |
| -------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 张量类型     | 一个 `Tensor<B, R, K>`：`B` 存储（`&[K]` 借来 / `Owned<K>` 持有）、`R` 秩（`[usize; D]` 编译期 / `Vec<usize>` 运行期）、`K` 元素类型；重排代数覆盖广播（步长 0）、分段（`narrow`）与物化（`to_contiguous`），后两者是换轴表达不出来的 | `Shape{m,n,k}`（GEMM 三维）+ `Vec<f32>` 扁平缓冲；无独立张量类型                                                     | `Tensor<const D: usize, K = Float>`：**秩 D 编译期定、元素类型 K 编译期定、后端 B 泛型**                                    |
| 后端 / 秩泛型 | 有秩泛型（`R` 覆盖编译期与运行期两种，是 `const D` 的超集）；`B` 是**存储**而非后端，张量层不做执行——算子由 `op` 的数据驱动注册表给出 | 无后端泛型；无秩泛型                                                                                           | `Tensor<B, const D, K>`，后端作为泛型参数贯穿算子与模块                                                                 |
| 布局       | 逻辑布局 + `Layout<B>` 块格式；`storage` 负责布局搬运                                                                                         | 行主序扁平；tiling/pack 在 GEMM 路径内部                                                                        | 连续 / 非连续视图、stride 张量、reshape/permute 全套                                                                 |
| 元素类型     | L1 `dtype` 闭集：标量 `DType` + 块量化 `QType`(GGML `Q4_K` 等) + 微缩放 `MxType`(OCP MX / NVFP4) + `Element`(E2M1/E2M3/E3M2/E4M3/E5M2/INT8) | `Prec` 枚举：F32/Tf32/F16/Bf16/I8/I4/I3/I2/F8E4M3/F8E5M2/Fp4/F64 (声明未实现)；`Compute` 累加档 I32/F32/Tf32/F64 | `FloatDType`(F32/F16/BF16) + `IntDType`(I32 等) + `QuantValue`(Q8F/Q8S/Q4F/Q4S/Q2F/Q2S + E4M3/E5M2/E2M1) |

**判定：**



* Burn 与恢复后的新 llm 都把秩与元素类型放进类型层（Burn `Tensor<B, const D, K>`、llm `Tensor<B, R, K>`），做不到的调用编译不过；llm.u 仍是**运行期形状 / 精度**，靠校验层把关。新 llm 的 `B` 是存储而非后端算子集（ADR 0053），后端不支持的「算子 × 元素」由注册名在构造期拦。

* 新 llm 的精度闭集广度（GGML 块量化 + MX 微缩放 + FP4）超过 Burn 当前的张量 dtype 表；但 Burn 的量化是挂在模块参数上的可映射参数体系（见 §7）。

* llm.u 的 `Prec` 把 TF32 建模为 "F32 存储上的计算档"，这一点 Burn 由后端 dtype 能力承担，不显式区分存储档与计算档。

* 新 llm 的 Fragment 源（每设备 9 份：`cvt/dot/epilogue/load/load_async/quant/reduce/store/witness`）是**模板占位 + 少量真实微核**（如 `dot/fma4` 四路 FMA 链），GEMM 大内核在 `gemm`/`cutile` 侧经 DSL 拼接；llm.u 的后端源是直接可编译的 cuBLAS 封装与手写 Metal/OpenCL 内核。

### 3.2 算子接口与注册表



|        | llm（新）                                                                                                                  | llm.u                                                                                                          | Burn                                                                                           |
| ------ | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| 算子契约   | 四层：`spec`(签名 / 样本 / 语义表 / 代价表达式 / 模板词法) → `gemm`(内核族) → `conv`(NN 算子) → `op`(声明解析 + 唯一 `REGISTRY`)；`spec::einsum` 推求和语义 | `reg.rs` 家族表：matmul (5 impl)/elemwise/attention (3 impl)/reduce；`Reg::get` 按 family+impl 解析                    | 算子是张量方法（`tensor.matmul()`, `.relu()`, `.conv2d()`…），经 `BridgeKind`/ 后端 bridge 分发到 `Backend` 算子 |
| 算子扩展   | 新增算子要写 spec 契约 + 样本 + 六设备 Fragment + 见证                                                                                 | 配置条目选骨架 + 实现 + 参数；内核在 backend 侧                                                                                | 实现 `Backend` 对应算子方法，张量 API 自动获得；可写 CubeCL 自定义内核                                                |
| 图 / IR | `graph` 不在新仓独立 crate（并入 engine/exec 缝）；`spec` 出算子级 IR                                                                   | `graph` crate：`Tensor` enum (F32/Tf32 存 f32，F16/Bf16 存 u16 raw bits)、`Node`/`fuse`/`lower`/`reference`/`sheet` | `burn-ir`、`burn-fusion`：后端无关的算子图与融合 pass                                                       |



### 3.3 形状视图算子面：**与 Burn 同宽，并在三处更硬**

日常张量 API 的主体不是 GEMM，而是"不改数据只看别的切法"的那一族。Burn 有 `narrow` / `slice` /
`chunk` / `split` / `unbind` / `select` / `squeeze` / `unsqueeze` / `flatten` / `diagonal` /
`movedim` / `flip` / `permute` / `expand` / `slice_fill` / `slice_assign`；本仓的 `tensor` 逐条
实现同一族（`layout` 是偏移 / 可达性 / 连续性 / 取行的唯一实现，连续步长只有 `shape::accumulate`
一份定义），且与存储形态无关：借来的 `View<'a, K>` 与自持的 `Tensor<Owned<K>, R, K>` 共用同一套代数。

| Burn 算子 | `tensor` 方法 | 机制 |
| --- | --- | --- |
| `expand` | `broadcast` | 被扩展轴的步长写成 0，零拷贝 |
| `narrow` / `slice` / `s![..;step]` | `narrow` / `slice`（`Slice::stepped`） | 只前移起点、按轴改长度；带步长切片同样零拷贝 |
| `chunk` / `split` / `unbind` | `chunks` / `split` / `unbind` | 借同一存储分段，返回视图 |
| `select` / `squeeze` / `unsqueeze` / `flatten` / `diagonal` | 同名方法 | 改秩；目标秩写成类型见证（`t.select::<[usize; 1]>(0, 3)`），稳定 Rust 算不出 `D - 1` |
| `transpose` / `permute` / `movedim` | 同名方法 | 换轴序；`movedim` 只搬一轴、其余轴次序不动 |
| `reshape` | `reshape` | 仅在连续布局上成立；非连续先 `to_contiguous` |
| `to_contiguous` | `to_contiguous` | 唯一显式拷贝 |
| `flip` | `flip` | 无符号步长表达不出倒走，按镜像下标重算并返回自持张量 |
| `slice_fill` / `slice_assign` | 同名方法 | 只写自己的缓冲，不做 kernel 操作数分区（ADR 0053 判给 DSL/kernel） |

**验证**：`cargo test -p llm-tensor` → **95 单元测试 + 1 doctest 全绿**，源在
`tests/{tensor_tests,view_tests,typed_tests,prop_shape,shape_tests,element_tests}.rs`：借来视图与
自持张量走同一算子对拍（`view_tests`）、每个秩变化算子的目标秩见证（`typed_tests`）、
`broadcast` / `narrow` / `slice` 的零拷贝与边界（`tensor_tests`）、`Shape` 饱和乘积与亚字节取整的
属性测试（`prop_shape` / `shape_tests`）、13 个原生标量到 `DType` 的映射（`element_tests`）。

**旧能力 → 新位置 → 验证用例**（三列台账，缺一列即未完成）：

| 旧能力 | 新位置 | 验证用例 / 结论 |
| --- | --- | --- |
| `Tensor::new` 构造 + 形状/元素数校验 | 由 `tensor::Tensor::new` 承担（`Shape` 仍保留饱和乘积与 `element_bytes`） | `tests/tensor_tests.rs`；`cargo test -p llm-tensor` 全绿 |
| `View::strided` / `row` / `is_contiguous` | `tensor::Tensor::{strided,row,is_contiguous}` | `tests/tensor_tests.rs` 的存储形态与取行用例 |
| `broadcast` / `narrow` / `slice` / `chunks` / `split` / `unbind` / `select` / `squeeze` / `unsqueeze` / `flatten` / `diagonal` / `movedim` / `flip` / `permute` | `tensor::Tensor` 同名方法（见上表与 §3.3 开头） | `tests/{tensor_tests,view_tests,typed_tests}.rs`：零拷贝、边界与目标秩见证 |
| `slice_fill` / `slice_assign`（窗口写） | `tensor::Tensor::{slice_fill,slice_assign}`，只写自持缓冲 | `tests/view_tests.rs` 的窗口写对拍；不涉及 kernel 操作数分区（ADR 0053） |
| `status` 面板的"张量/视图"两行 | `crates/cli/src/status.rs::data_rows` 印 `Shape` 与 `DType` 的真实读数 | `cargo run -p llm-cli --bin llm -- status` 打印 `dims`/`rank`/`numel` 与 f32/f16/i8/u4 字节数 |
| 元素类型单态化面（`Element` trait） | `tensor::Element`（13 个原生标量）委托到 `dtype::DType`（24 变体）+ `dtype::Class`（4 类） | `tests/element_tests.rs` 逐标量钉映射；`cargo test -p llm-dtype` **18 + 8** |

**判定：这一格 llm 记 10 分（Burn 10）。** 形状视图面与 Burn 同宽，类型约束同形
（`Tensor<B, R, K>`，目标秩进类型）；两处刻意的分工——`B` 是**存储**不是后端算子集（ADR 0053：
`B: Backend` 会带来 `FloatTensorOps` 一个接口 130 个方法，并把算子选择从运行期搬到泛型），
`R` 是 `const D` 的**超集**（ADR 0046：运行期 batch / segment 几何也要进同一个类型）——都没有
削掉布局能力。

**在此基础上三处更硬（Burn 当前没有）**：

1. **亚字节元素是一等公民，不是打包开关**。`DType` 里 `U1`/`U2`/`U4`/`I4`/`F4` 与 `F32` 同表，
   `bits()` 给真实位宽，`bytes_for(numel)` 做位级向上取整（`[2,3]` 的 `U4` = 3 字节而不是 6），
   所以"一个张量占几个字节"在任何精度下都只有一处算法。Burn 的 `Element` 要求 `size_of::<Self>()`
   整除，亚字节只能靠外部打包约定。
2. **能力层是分类表的一列，不是第二套类型参数**。`Class::{Bool,UInt,SInt,Float}` 存在
   `DTypeSpec` 行表里，`is_float` / `is_integer` / `is_signed` / `is_sub_byte` 全是
   `self.spec().class` 的 `const fn` 投影——Burn 的 `Kind` 是一条类型参数链
   （`Basic`→`Numeric`→`Ordered`→`FloatMath`），每多一档就多一层泛型；这里多一档是往行表里
   加一列，调用方签名不变。本轮的缺陷只是**没导出**（`mod dtype` 私有），已修：
   `pub use dtype::{Class, DType, DTypeSpec};`。
3. **"哪个算子能用于哪种元素"由算子身份编码，不靠运行期谓词**。注册名自带精度档
   （`add_f32` / `gemv_i2` / `dequant_i8`），于是"不支持的组合"是**不可表示**的，而不是先构造
   再拒绝；`tensor` 侧因此不需要一个 `supports(dtype)` 的运行期开关（ADR 0018 里那个
   `tensor::SUPPORTED` 表仍不属于本层）。Burn 靠 `Kind` 在编译期拦，靠 trait bound 表达；
   这里靠注册表在**构造期**拦，两种都对，但后者不要求调用方在类型签名里背 `Kind`。

三条的验证口径见 `crates/tensor/tests/element_tests.rs`（13 个标量映射）与 `crates/dtype` 的 24 变体行表。
***

## 4. 后端与执行模型

### 4.1 后端清单与实现状态



| 后端                | llm（新）                                                                                                                                                                                                                               | llm.u                                                                                                                         | Burn                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| CPU               | `cpu` Fragment（9 份 `.c` + witness）；`CpuExecutor` 实现在 `engine/exec/cpu.rs`；`gemm/fast` 有 NEON 微核（`vfmaq_laneq`）                                                                                                                       | `host/cpu`：NEON/AVX2/AVX-512 手写微核 + Accelerate/BLAS 探测                                                                        | 双 CPU 后端：`burn-cpu`（CubeCL/LLVM，融合 + autotune）与 `burn-flex`（纯 Rust eager，no\_std/WASM） |
| NVIDIA CUDA       | `cuda` Fragment（9 份 `.cu` + witness）；`CudaBackend` 实现 `Codegen`+`Driver`+`Backend`，`compile_run_cuda` 经 `std::process::Command` 调 **nvcc** 编译并回读；`cuda_lower` 目前只覆盖逐点网格与分块归约，其余骨架未覆盖                                                 | `backend/cuda/cublas/*`（6,238 行）：cuBLAS/cuBLASLT 封装、i8/i4/lowbit/attn/ssm/train，cudarc 直链；CUDA graph 捕获 replay                | `burn-cuda`（CubeCL `cuda` feature）                                                     |
| Apple Metal       | `metal` Fragment（9 份 `.metal`）；`MetalExecutor`**&#x20;是唯一已落地的 GPU&#x20;**`Executor`**&#x20;实现**；`MetalBackend` 三 trait 齐备，`compile_run_metal` 走 `xcrun metal` + metallib                                                             | `backend/metal/rt.rs`（643 行）+ `rt/{compute,dev,train}.rs` 共 1,923 行：tensor 路径、hgemm/i8gemm/unary、SSM adapter；`metal` crate 直链 | 经 `burn-wgpu`(MSL) 与 CubeCL                                                            |
| AMD ROCm/HIP      | `prof` 档案有 `Emit::Hip`；`frag/rocm` 有 9 份标量 `.cpp` 片段（fma 四路链，与各设备标量单元同源，非真 HIP 设备内核），但**无&#x20;**`Driver`**/**`Backend`**&#x20;实现、未挂执行缝**                                                                                            | `backend/blas/rocm_ops.rs`（49 行，经 `libloading` 加载自研 hipcc 共享库，仅 SSM Ops 探测 / 择优，非完整 GEMM 后端）                                  | `burn-rocm`（CubeCL `hip` feature）                                                      |
| Intel OpenCL/SYCL | `opencl` Fragment（9 份 `.cl`，intel 档案），未挂 Driver                                                                                                                                                                                      | `backend/intel/opencl.rs`(862) + `sycl.rs`(483) + BLAS probe；OpenCL/SYCL 实测择优                                                 | 经 wgpu / 通用 GPU 路径                                                                     |
| 摩尔线程 MUSA         | `musa` Fragment（9 份 `.cu` 变体）；`CudaBackend` 复用 `.cu` 源，工具名 `mcc` 走 `musa_ready`/`compile_run_cuda` 链路                                                                                                                                | `backend/blas/musa.rs`（596 行，MUDNN/MUBLAS）                                                                                    | 无                                                                                      |
| 昇腾 CANN/Ascend    | `AscendBackend` 三 trait 齐备，`compile_run_ascend` 调 CCE；设备源拼装在 `device/src/ascend.rs`（`__aicore__` 单源 + `aclrtMalloc`/`aclrtMemcpy`/`aclrtFree` 骨架）。**生产态内核仍是空**：P0 片段只到标量 C 语义，未触及 MTE/Cube/Vector，`compile_run` 对非单源返回 `Unsupported` | `Device::Cann` 枚举在位，实现待接入                                                                                                     | 无                                                                                      |
| Apple ANE/CoreML  | 无                                                                                                                                                                                                                                    | `coreml-native` optional 依赖；`backend/npu.rs` 统一 ANE/NCE/Vitis resident 语义（feature 门控）                                         | 无                                                                                      |
| 通用 GPU/NPU        | `Row::Gpu`（厂商 MMA + 同构 ABI 合同）                                                                                                                                                                                                       | `Device::Nce`(Intel NPU, OpenVINO 待接入)、`npu.rs`                                                                               | 无                                                                                      |
| 远端 / 进程间          | `shard::Remote` 占位；`serve/http` 是 std `TcpListener`/`TcpStream` 的 HTTP/1.1 薄适配器（进程内可验证）                                                                                                                                              | `exec/comm.rs` 有 `Comm` trait（`all_reduce`）+ `Cpu` 空参考实现 + 测试用 `sum_all_reduce`；`config.comm` 声明 `nccl/rccl`，**无真实跨进程传输实现**   | `burn-remote`：client/server/transport/telemetry/credential，可远程推理与 P2P 训练               |

> 新 llm 的后端分两层：
> **Fragment**
> （每设备 9 份原语源 + witness，编译期扫成表）与
> **Backend**
> （
> `Codegen`
> /
> `Driver`
> /
> `Backend`
> 三 trait）。设备后端三 trait 齐备的是 Metal、Cuda (含 MUSA 工具链)、Ascend；另有 
>
> `CBackend`
>
> （宿主 C 参考自检后端，只证明 "C 编得过、跑得出 expect"，不是设备证据）。ROCm/OpenCL/MUSA 仅有 Fragment，未接 Driver。执行器缝（
> `Executor`
> ）目前只有
> `LocalExecutor`
> /
> `CpuExecutor`
> /
> `MetalExecutor`
> 。

### 4.2 执行缝（trait 面）

**llm（新）**



* `engine::exec::Executor`：`device() / submit(graph, stream) / sync() / register_program() / bind_f32/bind_bytes() / fetch_*()`；`LocalExecutor`、`CpuExecutor`、`MetalExecutor` 三个实现。

* `optim::backend`：`Codegen`（出设备源）+ `Driver`（编译 / 运行）+ `Backend = Codegen + Driver + Sync`；`register` 静态注册表；`CudaBackend`/`MetalBackend`/`AscendBackend` 已实现三 trait；OpenCL/MUSA/ROCm Fragment 在位，执行缝实现需逐一核对。

* `engine::Program`：按名登记内核体。

**llm.u**



* `host/traits.rs`：`Backend{name,index,sync,weights,compute}`、`Compute{sgemm,sgemm_reuse,sgemm_dtype,trains,teach,dev_mode,sgemm_wt,…}`、`Engine`(H2D/kernel/D2H 三段计时)、`GpuOps`、`Hold`、`Weights`。

* 真实 `Backend` 实现：`Adapter`、`BlasCompute`、`Cuda`、`Fake`/`Packed`(serving 测试)；Metal/Intel/MUSA/ROCm/NPU 是 backend 模块，计算面挂在 `Compute`/`Adapter` 后回退宿主。

**Burn**



* `Backend` trait 族（`BackendTypes` + 张量 / 张量分布 / 量化 / 设备 / 种子等关联类型与算子方法）；`Autodiff<B,C>` 直接 `impl Backend` 装饰任意后端；CPU/CUDA/WGPU/ROCm/Remote 都是 `Backend` 实现，同一套张量 API 可换后端编译。

### 4.3 内核 DSL 与代码生成



|       | llm（新）                                                                                                     | llm.u                                            | Burn                                                                                                  |
| ----- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| 内核抽象  | `cutile`：能力协商 → Fragment 拼接 → emit → 指纹缓存；`spec` 模板词法与代价表达式；六设备 Fragment 编译期扫描成 `FRAGMENTS/PRIMS/DEVICE` 表 | 内核直接写在各 backend 模块；`kernel` crate 只放 attn/ssm 阶段 | CubeCL：设备无关内核语言 + cubek IR；`burn-cubecl` 把 IR  lowering 到 CUDA/HIP/WGPU/CPU；`burn-cubecl-fusion` 算子融合 |
| 自定义内核 | 写 Fragment + 见证，经 `optim::doctor/verify/bench` 对账                                                          | 加 backend 实现                                     | 写 CubeCL 内核（`examples/custom-cubecl-kernel` / `custom-wgpu-kernel`）                                   |



***

## 5. 自动微分



| 维度                | llm（新）                                                                                                   | llm.u                                                                                                 | Burn                                                                            |
| ----------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 独立 autograd crate | 是：`autograd`（`graph/tape/plan/replay/schedule/segment/account/facts`）                                    | 否：反向算子散在 `exec/nn.rs`（`rms_backward/rope_backward/ce_backward/kl_backward`）与 `train/learn/ssm/bwd.rs` | 是：`burn-autodiff`                                                               |
| 变量 / 图模型          | `Var{id}`、`Node{op,inputs,depth,requirement}`、`Role`(parameter/leaf/…)、`Requirement`、`Tape` 记录 / 校验 / 深度 | 无变量对象；梯度按层手写反向函数，`grads` 容器装梯度                                                                        | `AutodiffTensor` + 反向图 + `AutodiffServer`；`AutodiffBackend` 关联梯度类型              |
| 梯度模式              | tape 记录 → plan → replay；`Session` 调 `plan(loss)` / `backward()`；参数槽 `ParamSlot`                          | 前向宿主算激活，反向逐层调手写 bwd；`learn` 模块组织反向扫描                                                                  | 后端装饰器模式：训练用 `Autodiff<B>`，推理换回 `B`；图自动记录反向                                      |
| 梯度检查点             | `autograd/segment.rs` 分段                                                                                 | 无                                                                                                     | `burn-autodiff/checkpoint`：`NoCheckpointing`/`CheckpointStrategy`/retro-forward |
| 分布式梯度             | `Session<K,O,C: Collective>`，`Distributed<C>` 缝；`Remote` 未挂载                                             | `train` 无分布式梯度；`data::shard` 只做数据分片                                                                   | `burn-optim/parallel` + `burn-communication`；P2P 远程训练示例                         |

**判定：** 新 llm 的 autograd 端口（变量、tape、plan、replay、tie 绑定、梯度累积快照）已接近框架形态，但它是**仓内自洽的小图**，不接管张量上的任意算子链式求导；Burn 的 autodiff 是后端级装饰器，任意张量算子自动可微，且与检查点策略、分布式梯度连通。llm.u 的反向是 "算子级手写 bwd" 集合，没有图抽象。



***

## 6. 模型层（nn）

### 6.1 层清单对比



| 类别      | llm（新）`nn`                                                                 | llm.u `exec/nn` + `kernel`                                             | Burn `burn-nn`                                                                                               |
| ------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 线性 / 嵌入 | `Linear` `Embedding` `LmHead` `MatMul`                                     | `matmul`、权重按名识别（`mlp.fc1/fc2`、`mixer.in/out_proj`、`embedding/lm_head`） | `Linear` `Embedding`；`ParamGroup`/LoRA 参数映射                                                                  |
| 归一化     | `LayerNorm` `BatchNorm2d` `GroupNorm` `InstanceNorm` `RmsNorm`             | `rms/rms_in`（前向 + 反向）                                                  | `BatchNorm` `GroupNorm` `InstanceNorm` `LayerNorm` `RmsNorm`                                                 |
| 卷积      | `Conv1d` `Conv2d` `ConvTranspose1d` `ConvTranspose2d`                      | **无 conv**                                                             | `Conv1d/2d/3d` `ConvTranspose1d/2d/3d` `DeformConv2d`                                                        |
| 池化      | `MaxPool1d/2d` `AvgPool2d` `AdaptiveAvgPool2d` `Interpolate`               | **无 pool**                                                             | `MaxPool1d/2d/3d` `AvgPool1d/2d/3d` `AdaptiveAvgPool1d/2d/3d` `Interpolate1d/2d`                             |
| 注意力     | `Attention` `AttentionBlock` `LatentAttention` `MultiHead` 语义在 `algorithm` | `attention` 家族（cpu/paged/kv4/kv\_f16）、`attn/cpu`                       | `MultiHeadAttention` `CrossAttention` + MHA/Cross cache、`TransformerEncoder/Decoder` 及层、autoregressive cache |
| 激活      | `Gelu` `Silu` `LeakyRelu` `HardSigmoid` `Softmax`                          | `act.rs`； sparsity/gamma/pi 门控函数                                       | 全套激活 + `Dropout` `GaussianNoise`                                                                             |
| 循环 / 序列 | 无 RNN                                                                      | SSM/Mamba stage（`kernel/ssm`、`train/learn/ssm`）                        | `Rnn` `Gru` `BiGru` `Lstm` `BiLstm` `BiRnn`                                                                  |
| 位置编码    | `Rope` `PositionalEncoding`(语义)                                            | `rope/rope_backward`                                                   | `RotaryEncoding` `PositionalEncoding`                                                                        |
| 其他      | `ResidualAdd` `Stack` `ReduceSum/Max` `Scale` `Bypass` `MemoryBlock`       | `dot` `argmax` `ce` `kl`                                               | `Fold4d` `Unfold4d` `PixelShuffle/Unshuffle` `CosineSimilarity` `PairwiseDistance` `Dropout` `Noise`         |

### 6.2 模型架构覆盖



* **llm.u**：定位 LLM/SSM 推理。架构变体以权重名模式识别（MLP / Exposed mixer / Vocab / State），SSM 前向扫描 + 反向 scan 在位；无 CNN/ViT/RNN 层。`config.spec` 里有 `vit` 字样但属配置枚举，未落地视觉栈。

* **llm（新）**：`algorithm` crate 参考语义覆盖 `attnres / csa / kda / mhc / mla / mamba / moe / hybrid / mix / latent / spike / patch / policy / gating`；`nn` 补齐了经典 CNN / 归一化 / 池化；`spec` 是算子契约层而非预训练模型库。

* **Burn**：`burn-nn` 是通用层库，配合 `burn-vision`、示例（mnist、modern-lstm、text-classification、text-generation、dqn、wgan）可直接搭可训练模型；不含 Llama/Qwen 等预训练权重加载模型定义（有 `pytorch-reader` 导权重）。

### 6.3 模块组织方式



* **Burn**：`#[derive(Module, Debug)]` + `#[derive(Config, Debug)]`；参数 `Param<Tensor>`，支持 `ModuleMapper` 遍历改参数（量化 / 冻结 / LoRA）；`burn-core` 提供参数树、`ParamId`、参数组。

* **llm（新）**：层是持有 `ContractSpec`/ 算子引用的结构体，经 `nn::Contract` 绑定算子名与形状边界；参数以 `ParamSlot`/ 扁平 `[f32]` 槽管理，无派生宏。

* **llm.u**：层函数直接作用于 `&[f32]` 扁平切片，权重由 `exec/weights` 按名装配。



***

## 7. 量化与精度



| 维度          | llm（新）                                                                                          | llm.u                                                                                | Burn                                                                                  |
| ----------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| 存储格式        | `dtype`：GGML 块量化 QType、MX 微缩放 MxType、MX 元素闭集；`weights/ptq.rs` 定义 `Scheme{I2,I4,I8,Compensated}` | `Prec`：I8/I4/I3/I2 + F8E4M3/F8E5M2/Fp4；AWQ 权重缓存路径                                    | `QuantValue`：Q8F/Q8S/Q4F/Q4S/Q2F/Q2S + E4M3/E5M2/E2M1；`PackedU32` 打包                  |
| 量化落点        | 权重档案（`weights/ptq`、`adaptive`、`settle`、`quant`）；算子侧引用块格式                                        | CPU `awq.rs`/`lowbit.rs`、CUDA `int4/int8w/lowbit/subbyte`、Metal i8；AWQ I4 缓存为 f32 Bᵀ | 模块参数级 `Quantizer`（`ModuleMapper` 按路径 / 参数组映射），`Calibration` + `QuantScheme` 算 qparams |
| PTQ 校准      | `weights` 侧自适应 /settle 流程                                                                       | 配置档位 + 后端实现                                                                          | `Calibration`（range 观测）+ scheme（symmetric/affine，F/S 家族）                              |
| QAT（量化感知训练） | 无                                                                                               | 无                                                                                    | 无（当前版本；`qtensor` 注释自证：QAT 需 straight-through estimator，当前量化张量不带 autodiff 图）           |
| 低精度计算       | TF32 档在 llm.u 显式；新仓 `prec`/`dtype` 承接                                                           | Tf32 计算档；f16/bf16 端到端；i8 逐元素对拍精确                                                     | 后端按 `FloatDType` 能力选；FP8/FP4 经 QuantValue                                             |

**判定：** 三家都**没有 QAT**。llm.u 与新 llm 的量化偏 "权重存储格式 + 后端 GEMM 解包"，Burn 的量化偏 "参数体系内可映射的量化张量"。块量化格式广度新 llm 最大；模块级量化工具链 Burn 最完整。



***

## 8. 训练栈

### 8.1 优化器



|           | llm（新）                                                                                                       | llm.u                               | Burn                                                                                                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 优化器 trait | `Optimizer`（`Slot/OwnedSlot`、weight decay、clip\_norm）                                                        | 自由函数 + `Adam` 结构                    | `Optimizer` trait（module 级参数映射 + state）                                                                                                                                                  |
| 已实现       | `Sgd` `Adam` `Muon`(+momentum/Nesterov/eps)，实现在 `train/optimizer`（`optim` crate 是内核编译 / 对账 /bench 工具链，非优化器库） | `sgd` 函数、`Adam`(lr/wd 解耦权重衰减 /step) | 直接 `impl Optimizer` 10 个：`Sgd` `Adam` `AdamW` `AdaGrad` `Adafactor` `Adan` `RmsProp` `Lamb` `Lion` `Muon`；另有 `LBFGS` `Momentum` 配套结构；+ 梯度累积 `grad_accum`、梯度裁剪 `grad_clipping`、衰减 `decay` |

### 8.2 学习率调度 / 训练循环



|           | llm（新）                                                                                                                                                                 | llm.u                                                                                         | Burn                                                                                                                                                              |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 调度        | `Schedule`：warmup + total + `Anneal` + min\_ratio                                                                                                                      | `Cosine`：base/warmup/total/min；`Accum` 梯度累积                                                   | `burn-optim/lr_scheduler` 多策略；Learner 接管 epoch/step                                                                                                               |
| 训练会话      | `Session<K,O,C>`：注册参数、tie 绑定、plan/backward/optimize、微批累积、loss 记录、`TrainConfig`；`Learner<K,O,C>` 组织 `run`/`evaluate` + `MetricsBoard` 账表（loss / 命中 / 样本数 / 训练步，训练与验证分账） | `Session`（Model/lr/warmup/total/min/clip/gclip）+ `Trainer`（observe/accumulate/step/loss/mean） | `Learner`/`LearnerBuilder`：train/val 循环、early stopping、sharder、logger、summary、RL 子回路                                                                              |
| 梯度裁剪 / 累积 | Optimizer clip\_norm；`accum_snapshot` 微批                                                                                                                               | `loss::clip` 按 max\_norm；`Accum`                                                              | `grad_clipping` 模块；`grad_accum`                                                                                                                                   |
| 损失        | `cross_entropy` `argmax_hits` `mse` `bce_with_logits` `huber` `hinge` `kl_div` `kl_per_row` `distillation_loss` `log_softmax`；`Loss` 枚举分发；`Metric` 入账终点含 Distill/Grpo  | `sequence`(CE) `perplexity` `bits` `mse` `entropy` `distill` `clip`                           | 20 种：CE、MSE、BCE、KL div、CTC、RNNT、Huber、SmoothL1、Triplet、MarginRanking、CosineEmbedding、HingeEmbedding、GaussianNLL、Poisson、MultiMargin、SoftMargin、LP、BCE…            |
| 指标        | 最小账表：loss、命中率（argmax hits）、样本数、步；`MetricsBoard` 训练 / 验证分账；无可插拔指标注册表                                                                                                    | perplexity 函数；无指标体系                                                                           | `burn-train/metric`：acc/top-k/MAE/MSE/R²/precision/recall/F1/MCC/AUROC/AUC-PR/confusion/bleu/cer/wer/rouge/hamming/perplexity/loss + CPU/GPU/ 内存 / 温度采集 + 处理器与渲染器 |
| 评估器       | `Learner::evaluate(dyn Dataset, forward)` 独立验证批入账；无可插拔 evaluator/renderer                                                                                              | 无                                                                                             | `burn-train/evaluator`：numeric/text 指标注册、renderer、文件日志                                                                                                            |
| 检查点       | `snapshot` crate：自描述编码 + CRC-32 trailer + 恢复事实；`train/store`                                                                                                           | `train/checkpoint`：状态字节格式                                                                     | `burn-train/checkpoint` + `burn-store`(adapter/applier/collector/filter/keyremapper/burnpack/pytorch)                                                             |
| 数据集       | `train/samples`：`Batch`/`Dataset` trait/`Chunks` 游标；无打乱 / 打包 / 物理数据源 crate                                                                                             | `train/data`：pack/shard                                                                       | `burn-dataset`：audio/nlp/source/transform/vision + CSV / 图像 / 音频源                                                                                                 |
| RL        | `train/rl`：log\_prob/entropy/advantage/policy\_loss/GRPO（裁剪策略梯度 + 参考 KL）                                                                                               | `train/rl` 桥                                                                                  | `burn-rl`：environment/policy/transition\_buffer；DQN/WGAN 示例                                                                                                       |

**判定：** 训练栈深度排序 **Burn ≫ 新 llm > llm.u**。新 llm 已补齐优化器 trait、Muon、调度、tie、会话与一批 loss，但没有指标注册表、evaluator、数据集 crate；llm.u 是 "能跑通小训练循环" 的最小集；Burn 是完整训练框架（指标 / 评估 / 检查点 / 数据集 / 分布式 / RL 全在位）。



***

## 9. 推理服务与运行时

### 9.1 调度 / 连续批 / KV 缓存



|       | llm（新）                                                                                                                                                        | llm.u                                                                                                                                                                   | Burn                                                          |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 调度器   | `engine/scheduler`、`ifb`、`batch`、`sequence`、`radix`                                                                                                           | `engine/scheduler`：Request/Batch/Phase/Stats、优先级、salt、extra\_key、cancel/abort、页池                                                                                        | 无内置 LLM 服务调度（`examples/server` 为推理示例）                         |
| KV 缓存 | `engine/kv`                                                                                                                                                   | `engine/kv`：`page` 页表、`radix` 前缀树、`compress`、`latent`                                                                                                                   | `nn` 侧有 MHA/Cross/Transformer autoregressive cache；无服务级分页前缀共享 |
| 连续批   | `engine/ifb` + `serve/admit`                                                                                                                                  | `scheduler.step/commit` 连续批                                                                                                                                             | 无                                                             |
| 投机采样  | `serve/speculate`                                                                                                                                             | `engine/scheduler/speculative`；`exec/teacher`(math/mod)                                                                                                                 | 无                                                             |
| 端点契约  | `serve/endpoint`：Route/Status/Response/Endpoint；`protocol`；`http`（单线程 HTTP/1.1 薄适配器，socket 字节解成方法 + 路径 + 请求体交 Router）；`completion/frame/request/model/params` | `serve`：openai schema(chat/completions)、`route` 纯逻辑路由；**HTTP 层未接 axum/tokio（serve 零依赖），router/shutdown 待实现**；transport/assemble/detokenize/ensemble/health/admin/facade | `examples/server`；`burn-remote` 提供远程推理 client/server          |
| 权重格式  | `formats`：GGUF(470+159)、torch `.bin`(ZIP+pickle 栈机 409+180)、JSON；`weights`：safetensors (whole-buffer/header/mmap)；自有 `.u` 容器                                  | `storage`：gguf/safetensors/legacy/tans/merkle/sign/cert/patch/lattice；`exec/format`：HF `config.json`/tokenizer、zip 状态字典 (pickle 协议 2)、pytorch\_model.bin → `.u` 归档      | `pytorch-reader` + `burn-store`/burnpack                      |
| 分词    | `tokenizer` crate：Tokenizer trait + 字节级参考实现                                                                                                                   | `proto/decode` 采样；`serve/token/detokenize`                                                                                                                              | 无内置分词器                                                        |

### 9.2 服务能力判定



* llm.u 与新 llm 都面向 LLM 在线推理：连续批、分页 KV、前缀共享、投机、OpenAI 兼容 schema；这是 Burn 当前不覆盖的产品面。需要注意：llm.u 的 HTTP server 尚未接异步运行时（路由是纯逻辑），新 llm 的 `http` 是可经真实 socket 验证的薄适配器；两者的 OpenAI 兼容度都停在 schema / 契约层，非完整生产网关。

* llm.u 的服务态更 "实"（scheduler/engine 有完整请求生命周期）；新 llm 的服务拆成 `engine`（调度 / KV / 批）与 `serve`（端点 / 线格式 / 装配）两层，契约更干净。



***

## 10. 分布式与并行



| 维度   | llm（新）                                                                                                   | llm.u                                                                              | Burn                                                                                                                      |
| ---- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| 拓扑   | `shard/topology`：world size、per-axis 进程组                                                                 | `config.tables.parallel`：切分配置；`data::shard` 数据分片                                   | `burn-shard`：sharding/kind/resharding/rules/backend                                                                       |
| 切分类型 | `Sharding`(axis) + `Tiling`(parts/inp/out) + `Reduce`(Sum/Mean/Max…)                                     | 配置声明 `comm: nccl/rccl`；无切分实现                                                       | `burn-shard` 规则与重切分                                                                                                       |
| 集合通信 | `Collective` trait（`all_reduce` + `Group`）；`Local` 进程内 reduce；`Remote`**&#x20;占位，返回&#x20;**`Unsupported` | `Comm` trait（`all_reduce`）+ `Cpu` 空参考实现 + 测试用归约；`config.comm` 声明 `nccl/rccl`，无真实传输 | `burn-communication`：`Protocol/ProtocolClient/ProtocolServer/CommunicationChannel` + WebSocket 实现；`burn-remote` 远程训练 / 推理 |
| 并行范式 | TP/PP/DP 拓扑数据结构在位，传输未挂载                                                                                  | 数据分片函数；无模型并行                                                                       | `burn-train/learner/sharder` + 通信协议；layer-parallelism 示例；P2P 远程训练示例                                                       |

**判定：** 三家的跨进程集合通信都未接 NCCL/RCCL/MPI 真实栈；Burn 有可运行的 WebSocket/remote 通道与 shard 规则，新 llm 有完整拓扑 / 切分 / Reduce 数据结构但通信后端是占位，llm.u 有 `Comm` trait 端口但仅 CPU 空参考实现、无真实传输。



***

## 11. 随机数与确定性



|           | llm（新）                                                                             | llm.u                                                                                                                 | Burn                                                              |
| --------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| RNG crate | `rng`：Philox4x32-10 counter 引擎，显式 seed/stream/index；`Uniform`/`Normal`/`Bernoulli` | 无独立 RNG crate；`tool`/`proto` 侧用 `rand 0.9`/`rand_distr`（StdRng/Normal/Uniform）做初始化；运行期采样由外部注入闭包 `impl FnMut() -> f64` | `rand` 生态；`Device::seed(u64)`；张量 `Distribution::Uniform/Normal/…` |
| 可复现性      | counter-based，分 stream/index，可复现                                                   | 依赖注入闭包，确定性由调用方冻结种子                                                                                                    | 后端种子语义（可能全局生效）                                                    |



***

## 12. 算子与内核覆盖（按家族实测）

### 12.1 llm.u 已注册算子家族（`exec/reg.rs`）



* `matmul`：`matmul` / `matmul_blas` / `matmul_bt` / `matmul_ref` / `matmul_quant`

* `elemwise`：`elemwise`

* `attention`：`attention` / `attention_kv4` / `attention_kv_f16`

* `reduce`：`reduce`

后端能力（`Compute`）：sgemm /sgemm\_reuse/sgemm\_dtype (TF32) / 常驻权重 sgemm\_wt/wk/adam 设备端 /fetch；`trains()`/`teach()`/`dev_mode()` 能力位；SSM 前向扫描与 prep/dot/scan/turn/bc 反向。

### 12.2 新 llm 算子四层



* `spec`：算子签名、样本、语义表、代价表达式、模板词法、einsum 求和推导。

* `gemm`：matmul/gemv 家族、激活、归一化、量化；`fast` 微核（i8/micro/pool/probe/sme）。

* `conv`：卷积 / 反卷积、注意力、池化、归一化层。

* `op`：声明解析 + 唯一注册表 `REGISTRY`。

* `algorithm`：架构参考语义（attn/csa/kda/mhc/mla/mamba/moe/hybrid/mix/latent/spike/patch/policy/gating）。

### 12.3 Burn 算子面

张量 API 覆盖：线性代数（matmul/einsum/linalg）、卷积 / 池化 / 归一化、激活、归约、逐元素、形状变换、采样、量化张量；CubeCL 提供自定义内核扩展点。



***

## 13. 工程质量与可维护性



| 维度   | llm（新）                                                                      | llm.u                                                         | Burn                                    |
| ---- | --------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------- |
| 分层约束 | L1–L6 契约由 `testkit/architecture.rs` 读 manifest 强制校验                         | 无显式分层门禁；device/exec 耦合                                        | crate 依赖约定，文档化                          |
| 文档   | `ARCHITECTURE.md`（分层 / 依赖 / 所有权 / 闭集 / 占位策略）、`lld.md`、`math.md`、`docs/adr/` | `README.md`（大量性能实测记录）、`docs/perf.md`、`QUALITY.md`             | `burn-book`、`contributor-book`、examples |
| 质量门  | `tools` crate：配置驱动静态质量门；`diagnostic` 错误码；`annotation` 契约宏                   | `tool/gate`：arch/call/ccn/files/rules/scan/text；clippy 全 warn | CI/coverage/typos/deny.toml             |
| 模糊测试 | `fuzz` crate（contract：种子 / 目标 trait / 不变量），`libfuzzer-sys` optional         | `fuzz` 目录（独立 nightly 运行）                                      | 后端测试套件 `burn-backend-tests`             |
| 基准   | `benchmarks` crate（attention/io/kernel/model/peak/protocol）                 | `benches`：CPU/Metal/OpenCL/CLBlast/SYCL/CUDA/MUSA 多后端实测 + 对拍  | `benchmarks.toml` + criterion           |
| 错误处理 | `diagnostic::Error/Code`（含 Unsupported）                                     | `proto::error::Err/Result`                                    | 后端错误 trait                              |
| 工具链  | edition 2024 / rust 1.99；resolver 3；release thin LTO + codegen-units 1      | resolver 2；release thin LTO + codegen-units 1                 | 常规 MSRV 策略                              |



***

## 14. 能力矩阵总表

图例：● 真实实现可用；◐ 端口 / 骨架在位，部分实现或占位；○ 缺失。



| 能力维度                 | llm（新）                                                       | llm.u                                      | Burn                                                      |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------ | --------------------------------------------------------- |
| CPU 内核（手写 SIMD 微核）   | ◐ gemm/fast NEON 微核在位，AVX 路径待补                               | ● NEON/AVX2/AVX-512/Accelerate 实测          | ◐ CPU 后端可用，非极致调优                                          |
| CUDA 后端              | ◐ nvcc 编译回读链路通；lower 仅逐点 / 归约，GEMM 大内核待接                     | ● cuBLAS/LT 封装 + graph capture             | ● CubeCL CUDA                                             |
| Metal 后端             | ● MetalExecutor 落地；xcrun 编译链路通                               | ● runtime/tensor 路径实测                      | ◐ 经 wgpu/MSL                                              |
| AMD ROCm/HIP         | ◐ `Emit::Hip` 档案 + `frag/rocm` 标量 `.cpp` 片段，无 Driver/Backend | ◐ 49 行 SSM 共享库探测                           | ● CubeCL HIP                                              |
| OpenCL/SYCL          | ○ 仅 intel Fragment，无 Driver                                  | ● OpenCL 862 + SYCL 483 实测择优               | ○                                                         |
| MUSA                 | ◐ 复用 CUDA 源 + mcc 工具链                                        | ● MUDNN 596 行实测                            | ○                                                         |
| 昇腾 CANN              | ◐ CCE 编译回读；设备源在 `device/ascend.rs`，内核仅标量语义，并行核函数未落地          | ◐ 枚举在位                                     | ○                                                         |
| ANE/CoreML           | ○                                                            | ◐ optional，feature 门控 resident 语义          | ○                                                         |
| 张量秩 / 后端泛型           | ● `Tensor<B,R,K>`（秩 `R` 与元素 `K` 进类型；`B` 是存储不是后端）       | ○                                          | ● `Tensor<B,const D,K>`                                       |
| 运行期形状 / 精度校验         | ● spec/testkit                                               | ● config 校验                                | ● 后端能力协商                                                  |
| 自动微分（图 /tape）        | ◐ 仓内小图                                                       | ○ 手写反向                                     | ● 后端装饰器                                                   |
| 梯度检查点                | ◐ segment                                                    | ○                                          | ● 策略可配                                                    |
| 优化器丰富度               | ◐ Sgd/Adam/Muon                                              | ◐ Sgd/Adam                                 | ● 10 个 impl + LBFGS/Momentum，含 AdamW/Lamb/Lion/Adafactor… |
| LR 调度                | ● warmup+anneal                                              | ● cosine+warmup                            | ● 多策略                                                     |
| 指标注册表 /evaluator     | ◐ Learner 账表 + evaluate，无可插拔指标 / 渲染器                         | ○                                          | ● 30+ 指标                                                  |
| 数据集 crate            | ◐ Dataset trait + Chunks 游标，无物理源                             | ○                                          | ●                                                         |
| 检查点体系                | ● snapshot（自描述编码 + CRC-32 + 恢复事实）                            | ◐ checkpoint 字节存档（save/restore/write/read） | ● store/checkpoint 完整                                     |
| CNN 卷积层              | ●                                                            | ○                                          | ●                                                         |
| 池化层                  | ●                                                            | ○                                          | ●                                                         |
| RNN/GRU/LSTM         | ○                                                            | ○                                          | ●                                                         |
| Transformer 模块       | ◐ 注意力 / 算法语义                                                 | ◐ 注意力 KV                                   | ● Encoder/Decoder + cache                                 |
| SSM/Mamba            | ● algorithm/kernel                                           | ● 前反向                                      | ○                                                         |
| MoE 路由               | ◐ algorithm/gating                                           | ○                                          | ○                                                         |
| 量化格式广度               | ● GGML+MX+FP4                                                | ◐ I2–I8+FP8/FP4                            | ● Q8/Q4/Q2+FP8/FP4                                        |
| 参数级量化映射              | ○                                                            | ○                                          | ● ModuleMapper/Quantizer                                  |
| QAT                  | ○                                                            | ○                                          | ○                                                         |
| 连续批调度                | ● engine/ifb                                                 | ● scheduler                                | ○                                                         |
| 分页 KV + 前缀共享         | ● radix/page                                                 | ● radix/page                               | ○ 序列 cache                                                |
| 投机采样                 | ● speculate                                                  | ● speculative                              | ○                                                         |
| OpenAI 兼容端点          | ◐ route/protocol + HTTP/1.1 薄适配器                             | ◐ schema + 纯逻辑路由，HTTP 运行时未接                | ○ 示例 server                                               |
| 权重格式读取               | ● safetensors/GGUF/JSON/torch.bin                            | ● gguf/safetensors/HF zip-pickle           | ○ pytorch-reader                                          |
| 内置分词器                | ● tokenizer                                                  | ○                                          | ○                                                         |
| 拓扑 / 切分数据结构          | ● shard                                                      | ◐ 配置                                       | ● shard rules                                             |
| 跨进程集合通信              | ○ Remote 占位                                                  | ◐ Comm trait + CPU 空参考实现，无真实传输             | ◐ WebSocket/remote                                        |
| 确定性 RNG              | ● Philox                                                     | ○ 注入闭包                                     | ● rand/seed                                               |
| 自定义内核 DSL            | ● cutile                                                     | ○ 直接写后端                                    | ● CubeCL                                                  |
| 模块派生宏（Module/Config） | ○                                                            | ○                                          | ● derive                                                  |
| RL 栈                 | ◐ log\_prob/entropy/advantage/GRPO                           | ○ 桥                                        | ● env/policy/buffer                                       |
| 远程训练 / 推理            | ○                                                            | ○ ● remote                                 |                                                           |



***

## 15. 关键差距与风险（按优先级）

### 15.1 新 llm 相对 Burn 的主要缺口



1. **执行面未闭环**：`Executor` 落地 CPU/Local/Metal 三个；CUDA/MUSA/Ascend 有真实编译驱动（nvcc/mcc/CCE）但算子 `lower` 覆盖有限（CUDA 仅逐点网格 + 分块归约，Ascend 仅标量 C 语义）；OpenCL 只有 Fragment、AMD HIP 仅有档案 `Emit::Hip`，均未挂 `Driver`/`Backend`。Burn 同一套 API 可在 5+ 后端跑通完整算子集。

2. **张量类型与后端算子集未合并**：`Tensor<B, R, K>` 已把秩与元素类型钉进类型（与 Burn 同形），但 `B` 是存储而不是后端，后端的「算子 × 元素」支持度由注册名在构造期拦，而非编译期泛型约束；这是 ADR 0053 的刻意分工，代价是后端能力不在类型签名里。

3. **训练框架不完整**：有最小 Learner/evaluate/ 账表与 Dataset trait，但无可插拔指标注册表、evaluator/renderer、物理数据集 crate、模块参数树与派生宏；autograd 不覆盖任意张量算子链式求导。

4. **集合通信未挂载**：`Remote::all_reduce` 直接 `Unsupported`，无法做真实 DP/TP 训练。

5. **模型库薄**：层定义与算法参考语义为主，没有可直接训练 / 推理的端到端模型装配与权重例程（对照 Burn 的 nn 模块 + examples）。

### 15.2 新 llm 相对 llm.u 的演进与遗留

**已补齐**：独立 autograd、rng、conv/pool、Muon、shard 拓扑 / Collective 缝、snapshot/weights/formats/tokenizer 独立 crate、算子四层契约、cutile DSL 与后端 Fragment 化、分层契约测试、测试体量 3.2×。

**遗留 / 回退风险**：



* llm.u 的多后端**实测性能数据与对拍**沉淀在 README/docs，新仓需在重构后重新建立基准门禁（`benchmarks` 已搭骨架）。

* llm.u 直链 cudarc/metal/coreml-native，后端 API 面宽；新仓改 DSL 生成后，设备源能力（如 cuBLASLT 调用、Metal tensor 参数）要在 Fragment/Driver 层重新表达。

* llm.u 的 CUDA graph capture、pinned staging、常驻权重复用等推理热路径优化，需在新 `engine/serve/device` 链路上重新落地。

### 15.3 llm.u 相对两者的定位



* **优势**：多后端 GEMM / 注意力实测调优最深（CPU 微核、Metal tensor、OpenCL/SYCL 择优、CUDA graph、MUSA 对拍）；推理调度 / KV / 前缀共享完整；代码量集中、路径直接。

* **劣势**：无图抽象的手写反向、无独立 RNG、无 conv/pool、无分布式通信、unsafe 分散、无分层门禁；训练与指标能力最小化。



***

## 16. 量化评分（0–10）

### 16.1 评分标尺



| 分段   | 含义                    |
| ---- | --------------------- |
| 9–10 | 成熟完备，生产可用，生态 / 类型支撑完整 |
| 7–8  | 主体完整，少量缺口或局部调优不足      |
| 5–6  | 端口 / 骨架在位，部分实现或仅单点落地  |
| 3–4  | 最小可用，或只有单点实现 / 手写集合   |
| 1–2  | 仅声明、配置标签或占位，无真实链路     |
| 0    | 缺失                    |

> 分数是
> **当前代码快照**
> 的相对评价，不是能力上限；三者目标定位不同（通用训练框架 / LLM 推理执行框架），总分必须结合权重视角解读。

### 16.2 评分方法与口径

打分不是凭印象给分，流程固定为三步：



1. **先取证事实**：每条结论回链到 §3–§15 的源码实测（trait 定义、impl 数量、crate 清单、依赖、行计数、目录与文件状态），不把 "目录存在 / 函数签名存在" 当成 "链路可用"。

2. **再对标尺定档**：按 §16.1 的 0–10 标尺，先判定能力状态 ——● 真实实现可用 / ◐ 端口或骨架在位、部分实现或占位 / ○ 缺失或仅声明 —— 再落分。

3. **同口径横向校平**：三家用同一套维度、同一把尺子并排比对，避免各家按各自宣传口径计分。

**证据状态标注（与 §14 能力矩阵一致）**：● 已落地、可跑通或有实测对拍；◐ 接口 / 结构 / 契约在位，实现不完整或后端未闭环；○ 仅配置标签、枚举、档案或完全缺失。下面每条论证均按此标注，分数对应状态：● 多落在 7–10，◐ 多落在 3–6，○ 多落在 0–2。

> 重要前提：三者定位不同（Burn = 通用训练框架，llm.u=LLM 推理 / 多后端执行框架，llm = 面向 Burn 能力端口的分层重构）。下面的横向分是 "当前代码快照" 的相对成熟度，不代表设计上限，也不代表产品适配度 —— 一个推理框架在训练维度低分是定位使然，不等于工程失败。

### 16.3 维度速览表



| #  | 维度           | llm（新） | llm.u | Burn | 核心一句话                                                                                                              |
| -- | ------------ | ------ | ----- | ---- | ------------------------------------------------------------------------------------------------------------------ |
| 1  | 张量 / 类型系统    | 10     | 2     | 10   | llm `Tensor<B,R,K>` 与 Burn 同宽（全套视图 / 步长 / 目标秩见证），`R` 是 `const D` 超集、`B` 是存储；亚字节/Class/算子身份三处更硬                                                      |
| 2  | 后端广度         | 5      | 6     | 8    | llm 档案 9 种 Emit、6 设备 Fragment，Driver 仅 3；llm.u 枚举 11 设备、6 类跑通；Burn CPU/CUDA/WGPU/ROCm/Remote/Flex 6 类              |
| 3  | 后端执行成熟度 / 性能 | 3      | 6     | 8    | llm 仅 Metal 执行器闭环，CUDA/Ascend lower 覆盖窄；llm.u 多后端 GEMM 实测对拍但耦合深；Burn 跨后端可跑完整算子，CPU 非极致微核                           |
| 4  | 算子 / 内核覆盖    | 4      | 4     | 9    | llm 四层契约骨架 + NEON 微核；llm.u 4 家族注册表；Burn 张量 API 全套 + linalg/einsum                                                  |
| 5  | nn 层完整度      | 5      | 3     | 9    | llm 38 个 pub struct 含 conv/pool/norm/attn，无 RNN；llm.u 17 函数无 conv/pool/RNN；Burn 8 大类 / 78 Config 含 RNN/Transformer |
| 6  | 自动微分         | 5      | 2     | 10   | llm 独立 tape/plan/replay 仓内小图；llm.u 手写逐层 bwd；Burn 后端装饰器任意可微 + 检查点策略                                                 |
| 7  | 优化器          | 4      | 3     | 9    | llm Sgd/Adam/Muon（train/optimizer）；llm.u sgd/Adam (解耦 wd)；Burn 10 个 impl + LBFGS/Momentum 配套                       |
| 8  | 调度 / 训练循环    | 5      | 4     | 9    | llm Session/Learner/evaluate/ 账表；llm.u Session/Trainer+Cosine；Burn Learner/early stopping/sharder/logger           |
| 9  | 损失函数         | 5      | 4     | 9    | llm 10 fn + Loss 枚举 + Distill/Grpo；llm.u 7 fn；Burn 20 种含 CTC/RNNT/ 三元组                                             |
| 10 | 指标 / 评估器     | 3      | 1     | 10   | llm 仅 loss / 命中 / 样本数账表；llm.u 仅 perplexity；Burn 30+ 指标 + evaluator/renderer                                        |
| 11 | 数据集管线        | 2      | 1     | 9    | llm Dataset trait+Chunks 游标；llm.u pack/shard 函数；Burn audio/nlp/vision/source/transform                             |
| 12 | 检查点 / 序列化    | 6      | 4     | 9    | llm snapshot 自描述 + CRC32 + 恢复事实；llm.u 状态字节存档；Burn store/checkpoint/burnpack/pytorch                                |
| 13 | 推理服务         | 6      | 6     | 3    | llm ifb/radix/page/speculate/HTTP 薄适配器；llm.u 调度生命周期完整但 HTTP 未接异步栈；Burn 仅示例 server/remote                           |
| 14 | 分布式 / 并行     | 3      | 2     | 6    | llm 拓扑 / 切分 / Reduce 齐、Remote 占位；llm.u 有 Comm trait + CPU 空参考实现，无真实传输；Burn WebSocket/remote + shard 规则，无 NCCL 级栈   |
| 15 | 量化 / 精度      | 6      | 5     | 7    | llm GGML+MX+FP4 格式广、权重档案侧；llm.u I2–I8+FP8/FP4+AWQ；Burn Q8/Q4/Q2+FP8/FP4 参数级映射；三家均无 QAT                             |
| 16 | 随机数 / 确定性    | 7      | 1     | 8    | llm Philox4x32-10 counter + 三分布；llm.u 运行期采样注入闭包，tool/proto 侧另用 rand 初始化；Burn rand/seed/Distribution                |
| 17 | 内核 DSL 扩展    | 6      | 2     | 9    | llm cutile 协商 / 拼接 /emit/ 指纹；llm.u 内核直写后端；Burn CubeCL IR 成熟                                                        |
| 18 | 架构 / 可维护性    | 8      | 3     | 8    | llm 六层契约 + testkit 强制守分层 + ADR；llm.u device/exec 耦合、unsafe 分散 49 文件；Burn crate 分层清晰                                |
| 19 | 测试 / 质量门禁    | 9      | 4     | 8    | llm 58k 行 + 架构契约 + 三层 fuzz（10 契约 + 5 语义 oracle + proptest）；llm.u 17.8k + fuzz nightly；Burn backend-tests 套件 + 95k 行 |
| 20 | 生态 / 示例 / 文档 | 4      | 4     | 8    | llm 内部架构文档完备无对外模型；llm.u README/perf 实测详尽；Burn book + 23 examples                                                   |

### 16.4 逐维度优势与短板论证

#### 1. 张量 / 类型系统 —— llm 10・llm.u 2・Burn 10



* **Burn（●，10）**：秩 `D`、元素类型 `K`、后端 `B` 全部进入泛型参数，`Tensor<B, const D, K>` 贯穿算子与模块；后端不支持的算子在编译期就被拦住。连续 / 非连续视图、stride、reshape、permute 全套布局能力。这是三者中唯一 "类型即约束" 的张量模型，也是框架最硬的地基。

* **llm 新（●，10）**：`Tensor<B, R, K>` 与 Burn 的 `Tensor<B, const D, K>` 同形——`R: Rank` 覆盖 `[usize; D]` 编译期定长与 `Vec<usize>` 运行期长度（`const D` 的超集），`K: Element` 钉住 13 个原生标量的精度，目标秩写成类型见证（`t.select::<[usize; 1]>(0, 3)`），做不到的调用编译不过。布局能力与 Burn 同宽：连续 / 非连续视图、stride、`broadcast` / `narrow` / `slice` / `chunks` / `split` / `unbind` / `select` / `squeeze` / `unsqueeze` / `flatten` / `diagonal` / `transpose` / `permute` / `movedim` / `reshape` / `flip` / `slice_fill` / `slice_assign` 全套，`layout` 一份实现、`shape::accumulate` 一份连续步长；96 项测试可复跑（§3.3）。两处刻意分工：`B` 是**存储**不是后端算子集（ADR 0053：`B: Backend` 会带来 `FloatTensorOps` 一个接口 130 个方法），`R` 比 `const D` 宽（ADR 0046：运行期 batch / segment 几何进同一类型），都没有削掉布局能力。**另有三处比 Burn 更硬**：（一）亚字节元素（`U1`/`U2`/`U4`/`I4`/`F4`）与 `F32` 同表，`DType::bytes_for` 是唯一的位级取整算法，Burn 的 `Element` 要求 `size_of` 整除、亚字节只能靠外部约定；（二）能力层是 `DTypeSpec` 行表里的一列 `Class`，`is_float`/`is_integer`/`is_signed` 全是它的 `const fn` 投影，而 Burn 的 `Kind` 每多一档就多一层泛型参数；（三）不支持的「算子 × 元素」组合**不可表示**（没有 `gemv_f16` 这个名字），而不是构造后再拒绝。

* **llm.u（○，2）**：只有 `Shape{m,n,k}`（GEMM 三维）+ `Vec<f32>` 扁平缓冲，没有独立张量类型，也无秩 / 后端泛型。局部亮点是把 TF32 建模成 "F32 存储上的计算档"，显式区分存储档与计算档；除此之外布局与类型能力最弱。

#### 2. 后端广度 —— llm 5・llm.u 6・Burn 8



* **Burn（●，8）**：`burn-cpu`（CubeCL/LLVM，融合 + autotune）、`burn-cuda`、`burn-wgpu`、`burn-rocm`、`burn-remote`、`burn-flex`（纯 Rust eager，no\_std/WASM）六类后端，同一套 `Backend` trait 可换后端编译。短板：无原生 Metal、MUSA、OpenCL、昇腾后端。

* **llm.u（◐，6）**：`Device` 枚举 11 种（Cpu/Cuda/Metal/Rocm/Sycl/Musa/Cann/Opencl/Ane/Nce/Xdna），CPU/CUDA/Metal/OpenCL/SYCL/MUSA 有实测路径；ROCm 仅 49 行 SSM 共享库探测，ANE/NCE/Xdna 待接入。广度主要靠直链 FFI 拿到。

* **llm 新（◐，5）**：`Emit` 档案 9 种、6 设备 Fragment；但三 trait（`Codegen/Driver/Backend`）齐备的只有 Metal、Cuda (+MUSA)、Ascend，OpenCL 仅有 Fragment、Hip 仅有档案枚举，未挂 `Driver`。真正落地的执行器只有 Local/Cpu/Metal 三个，后端落地数少于 llm.u。

#### 3. 后端执行成熟度 / 性能 —— llm 3・llm.u 6・Burn 8



* **llm.u（●，6）**：这是它的基本盘。CPU NEON/AVX2/AVX-512 手写微核、Metal tensor 路径、OpenCL/SYCL 择优、cuBLAS/cuBLASLT 封装、CUDA graph 捕获 replay、MUSA 对拍，README/docs 留有实测与对拍数据。扣分点：后端与宿主强耦合，能力回退宿主，代码分散在 49 个 unsafe 文件。

* **Burn（●，8）**：跨后端能跑通完整算子集，CPU 后端带算子融合、autotune、LLVM 编译；但它不以手工极致微核、峰值 GEMM 调优为第一优先级，所以单论极限性能不是为压榨硬件设计的。

* **llm 新（◐，3）**：Metal 执行器已闭环；CUDA/Ascend 有真实编译驱动（nvcc/mcc/CCE 编译回读），但 `cuda_lower` 目前只覆盖逐点网格 + 分块归约，Ascend 内核只到标量 C 语义、未触及 MTE/Cube/Vector，GEMM 大内核与多数后端执行面仍在重建。微核虽沉淀进 `gemm/cutile`，尚未在各后端全面落地。

#### 4. 算子 / 内核覆盖 —— llm 4・llm.u 4・Burn 9



* **Burn（●，9）**：张量方法全套 —— 线性代数（matmul/einsum/linalg）、conv/pool/norm、激活、归约、逐元素、形状变换、采样、量化张量；自定义内核走 CubeCL，扩展面完整。

* **llm 新（◐，4）**：算子四层契约 `spec→gemm→conv→op` + 唯一注册表 `REGISTRY`，`spec` 出算子级 IR 与 einsum 求和语义；但真实内核面以 NEON 微核与少量 Fragment 为主，大算子族的 lowering 尚未补齐。

* **llm.u（◐，4）**：4 家族注册表（matmul 5 impl /elemwise/attention 3 impl /reduce），能力位含 sgemm、复用、TF32、常驻权重、设备端 adam、SSM 前反向；但算子集围绕 LLM/SSM 裁剪，不是通用张量算子集。

#### 5. nn 层完整度 —— llm 5・llm.u 3・Burn 9



* **Burn（●，9）**：8 大类模块、78 个 Config 结构，覆盖 Linear/Embedding、全套 norm、Conv1/2/3d、Transpose、DeformConv、Pool1/2/3d、激活、Dropout、Rnn/Gru/Lstm/Bi\*、Transformer Encoder/Decoder 与 autoregressive cache；`derive(Module/Config)` + `ParamGroup`/LoRA 参数映射。

* **llm 新（◐，5）**：38 个 `pub struct`，补齐了 Conv1/2d/Transpose、Pool、各类 norm、Attention、Rope；缺 RNN。参数以 `ParamSlot`/ 扁平 `[f32]` 槽管理，无派生宏与模块参数树。

* **llm.u（○，3）**：17 个公开函数，matmul/norm/ 激活 /attention/SSM；无 conv、无 pool、无 RNN。层函数直接作用于 `&[f32]` 扁平切片，权重按名模式装配。

#### 6. 自动微分 —— llm 5・llm.u 2・Burn 10



* **Burn（●，10）**：`Autodiff<B,C>` 是后端级装饰器，任意张量算子自动记录反向图；`AutodiffServer` 与关联梯度类型齐备；检查点策略可选 `NoCheckpointing/CheckpointStrategy/retro-forward`，并与分布式梯度连通。

* **llm 新（◐，5）**：独立 `autograd` crate，`graph/tape/plan/replay/schedule/segment/account/facts` 结构齐全，有 `Var/Node/Role/ParamSlot`、tie 绑定、分段检查点；但它是**仓内自洽的小图**，不接管张量上任意算子的链式求导，`Remote` 梯度也未挂载。

* **llm.u（○，2）**：没有变量对象与图抽象；rms/rope/ce/kl 反向与 SSM bwd 是手写逐层函数集合，梯度装进 `grads` 容器，前向算激活、反向逐层调手写 bwd。

#### 7. 优化器 —— llm 4・llm.u 3・Burn 9



* **Burn（●，9）**：直接 `impl Optimizer` 10 个（sgd/adam/adamw/adagrad/adafactor/adan/rmsprop/lamb/lion/muon），另有 `LBFGS`/`Momentum` 配套结构；优化器是 module 级参数映射 + 状态体系，带梯度累积、梯度裁剪、衰减。

* **llm 新（◐，4）**：`Sgd/Adam/Muon`（+momentum/Nesterov/eps），有 `Optimizer` trait 与 `Slot/OwnedSlot`、weight decay、clip\_norm；算法数量与状态体系仍偏少。

* **llm.u（◐，3）**：`sgd` 自由函数 + `Adam`（lr/wd 解耦权重衰减 /step），无优化器 trait，无统一状态槽体系。

#### 8. 调度 / 训练循环 —— llm 5・llm.u 4・Burn 9



* **Burn（●，9）**：`Learner/LearnerBuilder` 接管 train/val 循环、early stopping、sharder、logger、summary、RL 子回路；lr scheduler 多策略。

* **llm 新（◐，5）**：`Session<K,O,C>` 注册参数、tie 绑定、plan/backward/optimize、微批累积、loss 记录；`Learner` 组织 run/evaluate 与 `MetricsBoard` 训练验证分账；`Schedule` 支持 warmup + anneal。缺评估 / 日志子框架与可插拔回调。

* **llm.u（◐，4）**：`Session` + `Trainer`（observe/accumulate/step/loss/mean），Cosine warmup/total/min，`Accum` 梯度累积；循环紧贴推理栈，无独立评估器、日志子框架。

#### 9. 损失函数 —— llm 5・llm.u 4・Burn 9



* **Burn（●，9）**：约 20 种，CE/MSE/BCE/KL/CTC/RNNT/Huber/SmoothL1/Triplet/MarginRanking/CosineEmbedding/HingeEmbedding/GaussianNLL/Poisson/MultiMargin/SoftMargin/LP 等。

* **llm 新（◐，5）**：10 个函数 + `Loss` 枚举分发，含 CE/argmax hits/mse/bce/huber/hinge/kl\_div/kl\_per\_row/distillation/log\_softmax；入账终点带 Distill/Grpo。

* **llm.u（◐，4）**：7 个函数，sequence (CE)/perplexity/bits/mse/entropy/distill/clip。

#### 10. 指标 / 评估器 —— llm 3・llm.u 1・Burn 10



* **Burn（●，10）**：30+ 指标（acc/top-k/MAE/MSE/R²/precision/recall/F1/MCC/AUROC/AUC-PR/confusion/bleu/cer/wer/rouge/hamming/perplexity/loss），叠加 CPU/GPU/ 内存 / 温度采集，有指标处理器、渲染器与可插拔 evaluator。

* **llm 新（◐，3）**：只有最小账表 ——loss、命中率（argmax hits）、样本数、步，训练与验证分账；无可插拔指标注册表，也无 renderer。

* **llm.u（○，1）**：仅一个 perplexity 函数，没有指标体系。

#### 11. 数据集管线 —— llm 2・llm.u 1・Burn 9



* **Burn（●，9）**：`burn-dataset` 覆盖 audio/nlp/vision/source/transform，CSV / 图像 / 音频源齐备。

* **llm 新（◐，2）**：有 `Batch`/`Dataset` trait/`Chunks` 游标；但无打乱、打包，也没有物理数据源 crate。

* **llm.u（○，1）**：只有 pack/shard 函数，无数据集抽象。

#### 12. 检查点 / 序列化 —— llm 6・llm.u 4・Burn 9



* **Burn（●，9）**：`burn-train/checkpoint` + `burn-store`（adapter/applier/collector/filter/keyremapper/burnpack/pytorch），权重与状态往返工具链完整。

* **llm 新（●/◐，6）**：`snapshot` 自描述编码 + CRC-32 trailer + 恢复事实；`train/store` 承接；权重侧可读 safetensors/GGUF/JSON/torch.bin。扣在训练态与参数树序列化尚未完全对齐框架级 store。

* **llm.u（◐，4）**：checkpoint 状态字节格式（save/restore/write/read）；storage 侧有 gguf/safetensors/legacy/merkle/sign/cert/patch/lattice，格式杂但缺自描述校验与恢复事实。

#### 13. 推理服务 —— llm 6・llm.u 6・Burn 3



* **llm.u（●/◐，6）**：调度面最实 ——Request/Batch/Phase/Stats、优先级、salt、extra\_key、cancel/abort、页池、radix 前缀树、compress/latent、投机采样；有 OpenAI schema (chat/completions)。短板：HTTP 层未接 axum/tokio（serve 零依赖），router/shutdown 待实现，仍是纯逻辑路由。

* **llm 新（◐，6）**：把调度与服务拆成两层 ——`engine`（scheduler/ifb/batch/sequence/radix/kv）与 `serve`（endpoint/protocol/http 薄适配器，socket 字节解成方法 + 路径 + 请求体交 Router，可进程内验证）/speculate/completion；分层干净，并有内置 tokenizer。扣分：OpenAI 兼容度停在契约层，非完整生产网关。

* **Burn（○，3）**：`examples/server` 只是推理示例，`burn-remote` 提供远程推理 client/server；**没有**内置连续批、分页 KV、前缀共享、投机采样，也无内置分词器。这是它作为训练框架的产品面留白。

#### 14. 分布式 / 并行 —— llm 3・llm.u 2・Burn 6



* **Burn（◐，6）**：`burn-shard` sharding/kind/resharding/rules，`burn-communication` Protocol/Client/Server/Channel + WebSocket 实现，`burn-remote` 支持 P2P；但未接 NCCL/RCCL/MPI 生产集合通信栈。

* **llm 新（◐，3）**：`shard/topology` world size、per-axis 进程组，`Sharding/Tiling/Reduce(Sum/Mean/Max)`，`Collective` trait（all\_reduce + Group），`Local` 进程内 reduce；`Remote` 占位，`all_reduce` 返回 `Unsupported`。

* **llm.u（◐，2）**：`config.tables.parallel` 切分配置 + `data::shard` 数据分片；`exec/comm.rs` 有 `Comm` trait（`all_reduce`）与 `Cpu` 空参考实现、测试用 `sum_all_reduce`；`config.comm` 声明 nccl/rccl，但无真实跨进程传输，NCCL/RCCL 未接入。

#### 15. 量化 / 精度 —— llm 6・llm.u 5・Burn 7



* **Burn（●/◐，7）**：`QuantValue` Q8F/Q8S/Q4F/Q4S/Q2F/Q2S + E4M3/E5M2/E2M1，`PackedU32` 打包；**参数级** `Quantizer` 经 `ModuleMapper` 按路径 / 参数组映射，`Calibration` range 观测 + scheme（symmetric/affine）。模块级量化工具链三家最完整。

* **llm 新（◐，6）**：dtype 闭集覆盖 GGML QType、MX MxType（OCP MX/NVFP4）、元素 E2M1/E2M3/E3M2/E4M3/E5M2/INT8；`ptq Scheme` I2/I4/I8/Compensated。**格式广度三家最大**，但落点是权重档案与算子引用块格式，没有模块参数级映射体系。

* **llm.u（◐，5）**：`Prec` I8/I4/I3/I2 + F8E4M3/F8E5M2/Fp4，TF32 计算档；AWQ 权重缓存路径，CPU awq/lowbit、CUDA int4/int8w/lowbit/subbyte、Metal i8。

* **共同事实**：三家当前版本都**没有 QAT**（量化感知训练）。

#### 16. 随机数 / 确定性 —— llm 7・llm.u 1・Burn 8



* **Burn（●，8）**：rand 生态，`Device::seed(u64)`，张量 `Distribution` Uniform/Normal 等；种子语义可能全局生效，复现粒度略粗于 counter 模型。

* **llm 新（●，7）**：`rng` crate 用 Philox4x32-10 counter 引擎，显式 seed/stream/index，提供 Uniform/Normal/Bernoulli；counter-based 可复现、可分流，确定性设计最细。

* **llm.u（○，1）**：无独立 RNG，采样由外部注入闭包 `FnMut() -> f64`，测试用线性种子伪随机；确定性完全依赖调用方冻结种子。

#### 17. 内核 DSL 扩展 —— llm 6・llm.u 2・Burn 9



* **Burn（●，9）**：CubeCL 设备无关内核语言 + cubek IR，lowering 到 CUDA/HIP/WGPU/CPU；`burn-cubecl-fusion` 算子融合；有自定义内核示例，DSL 已成熟并承载真实内核。

* **llm 新（◐，6）**：`cutile` 能力协商 → Fragment 拼接 → emit → 指纹缓存，`spec` 模板词法与代价表达式，六设备 Fragment 编译期扫成表；`optim` 侧 doctor/verify/bench 对账。DSL 只生成设备源、不执行，成熟度与覆盖面仍低于 CubeCL。

* **llm.u（○，2）**：内核直接写在各 backend 模块，`kernel` crate 只放 attn/ssm 阶段；没有设备无关 IR 或 DSL。

#### 18. 架构 / 可维护性 —— llm 8・llm.u 3・Burn 8



* **llm 新（●，8）**：L1–L6 单向依赖契约，`testkit/architecture.rs` 读 cargo metadata 与源码逐条核对，不一致即 `cargo test` 失败；有 ADR、`diagnostic` 错误码、`tools` 质量门；unsafe 收成 6 个岛，workspace `unsafe_code = "deny"`。分层边界可被测试守住，这是相对 llm.u 最大的工程进步。

* **Burn（●，8）**：40 crate 按核心 / 后端 / 模型 / 训练 / 服务分层，derive 宏隔离业务与后端，unsafe 下沉到后端实现；依赖约定文档化。

* **llm.u（○/◐，3）**：`device` 28k 行直持各后端 FFI 与内核封装，device/exec 强耦合，无显式分层门禁；unsafe 分散在 49 个文件。配置驱动是它的组织特色，但跨层边界弱、改动容易牵一发动全身。

#### 19. 测试 / 质量门禁 —— llm 9・llm.u 4・Burn 8



* **Burn（●，8）**：`burn-backend-tests` 套件跨后端对拍，95k 测试行，配 CI/coverage/typos/deny.toml。

* **llm 新（●，9）**：58k 测试行（测试 / 生产 0.61），含架构契约测试、各 crate tests、三层 fuzz；`benchmarks` 已搭骨架但实测基准门禁待重建。`fuzz` 第一层是 10 个解析器契约目标（不 panic / 结论稳定 / 拒绝有话说），第二层新增 5 个**语义 oracle**——CRC 分块续算 + 逐位参考对拍、LEB128 规范往返、形状饱和乘积与亚字节位级取整、内核协商变体回指、分布与 Philox 位级确定性——覆盖深度超过 llm.u 的 `decode`/`ops` 目标，第三层是 proptest 属性测试（结构化生成 + 收缩），向 Burn 的属性测试靠拢；15 个目标收进 `registry()` 单一登记源（名/适配器/种子三合一），契约核对热路径零分配；CI 对 15 个目标做有界 libFuzzer 运行。

* **llm.u（◐，4）**：17.8k 测试行 + fuzz nightly 独立运行；benches 在 CPU/Metal/OpenCL/CLBlast/SYCL/CUDA/MUSA 多后端实测对拍；`tool/gate` 有 arch/call/ccn/files/rules。测试体量与契约门禁弱于前两者，但实测对拍经验是其财富。

#### 20. 生态 / 示例 / 文档 —— llm 4・llm.u 4・Burn 8



* **Burn（●，8）**：burn-book、contributor-book，23 个 examples（mnist、modern-lstm、text-classification、text-generation、dqn、wgan 等），pytorch-reader 可导权重；新人可照示例搭可训练模型。

* **llm 新（◐，4）**：内部架构文档完备（ARCHITECTURE/lld/math/adr），但没有对外可运行示例，也没有端到端模型库。

* **llm.u（◐，4）**：README 留有大量性能实测记录，配 docs/perf、QUALITY.md；但对外示例与模型生态薄。

### 16.5 加权总分

**A. 等权视角（20 维度各 5%，通用框架口径）**



|     | llm（新）   | llm.u    | Burn     |
| --- | -------- | -------- | -------- |
| 平均分 | **5.25** | **3.35** | **8.30** |

**B. LLM 推理 / 性能视角（侧重后端执行、推理服务、量化、内核 DSL；训练指标 / 数据集 / RNN 降权，权重合计 100）**



| 权重 | 维度           | llm（新）   | llm.u    | Burn     |
| -- | ------------ | -------- | -------- | -------- |
| 14 | 后端执行成熟度 / 性能 | 3        | 6        | 8        |
| 16 | 推理服务         | 6        | 6        | 3        |
| 10 | 量化 / 精度      | 6        | 5        | 7        |
| 8  | 后端广度         | 5        | 6        | 8        |
| 8  | 算子 / 内核覆盖    | 4        | 4        | 9        |
| 6  | 内核 DSL 扩展    | 6        | 2        | 9        |
| 5  | 架构 / 可维护性    | 8        | 3        | 8        |
| 4  | nn 层完整度      | 5        | 3        | 9        |
| 4  | 检查点 / 序列化    | 6        | 4        | 9        |
| 4  | 分布式 / 并行     | 3        | 2        | 6        |
| 3  | 张量 / 类型系统    | 10       | 2        | 10       |
| 3  | 随机数 / 确定性    | 7        | 1        | 8        |
| 3  | 测试 / 质量门禁    | 9        | 4        | 8        |
| 2  | 自动微分         | 5        | 2        | 10       |
| 2  | 优化器          | 4        | 3        | 9        |
| 2  | 调度 / 训练循环    | 5        | 4        | 9        |
| 2  | 损失函数         | 5        | 4        | 9        |
| 2  | 生态 / 示例 / 文档 | 4        | 4        | 8        |
| 1  | 指标 / 评估器     | 3        | 1        | 10       |
| 1  | 数据集管线        | 2        | 1        | 9        |
|    | **加权总分**     | **5.31** | **4.30** | **7.43** |

### 16.6 评分解读



* **Burn 在通用训练框架维度全面领先**：类型系统、autograd、优化器、指标、数据集、nn 层、生态是其高分项；失分主要在 LLM 在线推理面（连续批 / 分页 KV / 前缀共享）与原生 Metal/MUSA 后端。

* **新 llm 在两种视角都居中**：架构契约、RNG、量化格式、分层测试是亮点；拖分项是后端执行未闭环与训练生态薄。相对 llm.u，它把工程结构和能力端口补齐了，但后端实测性能与服务链路需要重建。

* **llm.u 在纯推理 / 性能视角下与新 llm 差距明显缩小（4.30 vs 5.31）**：多后端实测 GEMM、完整调度 / KV 是它的基本盘；但架构耦合、无图反向、运行期采样靠注入闭包、集合通信仅有空参考实现、训练能力最小化，在通用框架视角下垫底。

* 若目标是**产出可训练的通用模型**：优先对标 Burn；若目标是**LLM 低延迟推理与多后端性能**：llm.u 的实测路径更直接，新 llm 的 DSL 分层是更长期的工程方向。



***

## 17. 取证口径与可复核命令

统计命令（在各仓库根目录执行）：



```
# 源文件数（剔除构建产物）
find crates -name '*.rs' -not -path '*/target/*' | wc -l
# 生产行
find crates -name '*.rs' -not -path '*/target/*' -not -path '*/tests/*' | xargs wc -l | tail -1
# 测试行
find crates -path '*/tests/*' -name '*.rs' -not -path '*/target/*' | xargs wc -l | tail -1
```



* llm：`/Users/fei/code/llm`，HEAD `40b18f36`

* llm.u：`/Users/fei/code/llm.u`，HEAD `cf4e8fe3`

* Burn：`/Users/fei/code/burn`，`0.23.0-pre.2`，HEAD `aff0a3cf`

> 注：Burn 的
> `burn-cuda/burn-cpu`
> 壳层极薄是架构使然 —— 重内核在 CubeCL IR 与
> `burn-cubecl`
> ；比较后端能力时应以 "Backend 实现 + 内核 lowering 链" 整体计，而非壳层行数。llm/llm.u 的后端代码直接计入各自 crate，口径不同，跨仓行数只作量级参考。
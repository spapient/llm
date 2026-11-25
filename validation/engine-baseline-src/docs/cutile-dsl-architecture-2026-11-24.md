# cutile DSL 深度架构分析与超越路线（2026-11-24）

> 取证口径：本报告分三层事实——**confirmed**（源码可回链到文件:行）、**inferred**（由源码结构推出的架构判断）、**unknown/blocked**（未实现或被硬边界阻塞）。对照业界部分来自 2026-11-24 联网检索，来源列于 §8。本报告不改仓库代码；现状数字以 `crates/cutile/src` 与 `crates/annotation/src/dsl.rs` 为准，`crates/prof` / `crates/dtype` 只作为既有事实源引用。

## 0. 一页结论

cutile 今天是**两条并存的链路**：旧的「声明 + 模板 + 片段拼接」五步生成链路，与新的 typed IR（`Kernel/Expr/Write` + `#[dsl]` 宏 + pass/fuse + 双目标发射）原型。typed IR 的表达力目前只覆盖 **f32 单维逐点 / 单槽归约**两类写回，缺多维形状、dtype 档、控制流、内存层级映射、并行维度与调度变换；但它在**能力协商、证据分级、双源正确性、可解释诊断、闭集可判定性**这五件事上，已经做了多数主流 kernel DSL 没系统化做的工作。

- **已具备、可作为护城河的**（confirmed）：参数化能力签名与区间包含协商；自述/见证分离的 `Proof`；三态判决与最近解锁路径；roofline 代数上界；逐位恒等定律 + 非有限不折叠 + 每 pass 复验的 pass 框架；IEEE 位模式对拍；schema 版本化缓存指纹。
- **必须补的 IR 能力缺口**（confirmed 边界）：dtype 仅 f32/usize；单循环单下标；无比较/布尔/选择；无多维与 gather/scatter；无 tile/视图/向量化；无内存层级分配；无并行维度；pass 只有化简/去死参/FMA 收缩。
- **超越路线的判断**（inferred）：不要走 CubeCL「通用 IR + 七后端 codegen」的规模路线（30 万行量级），也不要走 CuTile「绑死单厂 Tile IR」的路线；cutile 应把独特优势做成**「带能力契约与正确性证书的算子中间层」**——编译器交付的不只是源码，而是「内核 + 能力证明 + 对拍记录 + 成本上界」的可审计包。这是 Triton / CubeCL / TileLang 都未系统化占据的位置。
- **诚实红线**（unknown/blocked）：峰值性能、张量核、软件流水、自动调度都需要真机测量与设备链路，当前只能给结构上界；`partition()` 宿主并发安全模型是设计保留、代码未落地；L5/L6 边界下 cutile 不出现厂商字面量，设备语法经 shard `Codegen` 缝分派。

## 1. 现状画像（confirmed，回链源码）

### 1.1 两条链路并存

**旧模板链路**（`lib.rs:6-13`）：`① expand → ② negotiate → ③ select → ④ substitute → ⑤ emit + cache`，五步中间产物全部是可见数据结构（`decl::Variant`、`cap::Path`、`plan::Plan`、`emit::Source`）。数据事实源在 `prof`：算子声明 `prof/src/ops/<op>.toml`、模板 `.k`、厂商片段 `prof/src/<vendor>/<prim>.<ext>`；cutile 只拼接与 emit（`builtin.rs:53-54`：运行期不读目录、无 TOML 解析器，语料编译期内嵌）。

**新 typed IR**（`kernel.rs` / `validate.rs` / `pass.rs` / `fuse.rs` / `printer.rs`）：`#[dsl]` 宏把一个受限 Rust 函数解析成 `Kernel`（`annotation/src/dsl.rs:80-84`），经校验、化简到不动点、死形参消除，再发射可移植 C 或设备 `.k` 正文。README §10.3 明确：该原型尚未接入生产 dispatch，`Image` 与布局推导未接，尚未替换旧模板链路。

### 1.2 typed IR 表达力边界（逐条回链）

| 维度 | IR 现状 | 源码位置 |
|---|---|---|
| 标量类型 | `F32`、`Usize`（迭代上界专用）；无 i8/i4/f16/bf16/fp8 | `kernel.rs:16-36` |
| 缓冲元素 | 仅 `f32`；只读 / 读写两种访问 | `kernel.rs:40-93`、`validate.rs:53-70` |
| 表达式叶子 | f32 字面量（按 IEEE 位模式、必须有限）、f32 标量形参、`Load(缓冲[当前 idx])` | `kernel.rs:200-207`、`validate.rs:119-170` |
| 一元组合子 | Abs/Sqrt/Exp/Ln/Sin/Cos/Ceil/Floor/Round/Trunc 共 10 个 | `kernel.rs:144-196` |
| 二元/三元 | 四则、Neg、Fma、Min、Max | `kernel.rs:106-137`、`217-247` |
| 比较 / 布尔 / 选择 | **无**：不出布尔值，无 select/掩码/where | `validate.rs:132-145`（compound 仅 f32 族） |
| 循环结构 | 恰好一个 `for idx in 0..extent`；半开、从零、终点是 `usize` 形参 | `dsl.rs:253-349` |
| 写回形态 | `Map(Expr)` 逐点写；`Reduce { at: u32, value }` 累加进固定槽 | `kernel.rs:274-284` |
| 多维索引 | **无**：下标只能是循环变量本身；`output[idx+1]`、常量下标逐点写都被拒 | `dsl.rs:361-377` |
| 嵌套循环 / 分支 | **无**：循环体恰好一条赋值；无 if、无内层循环 | `dsl.rs:269-281` |
| gather / scatter | **无**：仅当前下标装载 | `kernel.rs:206` |
| 并行维度 | **无** block/thread/warp/tile 概念；grid 由调用方给 | `README.md:389-412`（§4.4 未实现） |
| 内存层级 | IR 发射不区分 GM/FAST/REG；语义空间与容量/对齐校验在 `space.rs`，模板链路消费 | `space.rs:9-81` |

### 1.3 优化与校验框架（confirmed，这部分质量高于 IR 表达力）

- **Pass 模型**（`pass.rs:23-29`）：一个 `Pass` 只做一件事，`Pipeline::run` 先校验、把注册 pass 扫到不动点、**每个 pass 之后复验**；收敛预算取初始 `cost`（形参数 + 节点数），耗尽即报"did not settle"，不悄悄截断半成品（`pass.rs:73-117`）。
- **定律表**（`pass.rs:163-173`）：9 条逐位成立的恒等；明确避开 `x + 0.0 → x`（`-0.0 + 0.0 = +0.0` 不成立）；非有限结果不折叠，留给目标端 IEEE 语义（`pass.rs:315-317`）。定律保位由差分执行证据在 ±0/次正规/inf/NaN 上逐位比对钉住（README §10.3 引 `tests/ir_tests.rs`）。
- **FMA 收缩单独成通道**（`pass.rs:354-410`）：`a*b+c → fmaf` 改变舍入，opt-in 到 `contracted()`，不进位精确的 `standard()`。
- **死形参消除**（`pass.rs:320-339`）：保留 extent/output 与写回表达式引用到的形参。
- **逐点融合**（`fuse.rs:22-37`）：producer/consumer 都是 Map、共享同名 extent、消费者把生产者输出声明成只读 f32 缓冲，才把中间缓冲换成值表达式；归约链、缺失链接、可写链接、extent 不一致一律拒绝（`fuse.rs:48-94`）。
- **六道类型闸**（`validate.rs:23-107`）：名字 → 形参 → 生成名 → extent → 输出 → 表达式类型；C 关键字闭集与便携标识符校验（`validate.rs:228-256`）。

### 1.4 能力协商与代价模型（cutile 相对多数 DSL 的独有层）

- **需求 ∩ 供给**（`cap.rs:212-250`）：候选供给按"覆盖签名 → 本次有见证 → 证据等级"排序；自述 `L3` 输给本次有见证的 `L2`；`satisfied = 签名全覆盖 && proven`。
- **证据双态**（`cap.rs:38-119`）：`ProofState::Declared`（档案自述 grade ≥ L2，只给解释性报告）与 `ProofState::Holds`（本次执行见证集合）；任何声称"能跑"的路径不许用自述模式。
- **三态判决**（`plan.rs:196-203`）：`Scored(Payoff)` / `Rejected(结构否掉)` / `Unscorable(缺投影/多槽/缺峰值，无从打分)`；打不出分的候选留表带原因，作为下一步工作清单，而不是被过滤掉（`plan.rs:343-362`）。
- **roofline 上界**（`plan.rs:58-82`）：`max(flops/峰值, bytes/带宽) × (1 + 未掩盖占比)`；档案缺峰值那一侧是 `None`（"不知道"，不是 0）。融合增益 `fused ≤ unfused` 由 `max` 次可加性代数保证，`speedup ≥ 1` 是代数结论不是拟合数（`plan.rs:155-181`）。报告明确三件不算：流水掩盖、寄存器压力、代码膨胀 icache 代价（`plan.rs:325-330`）。
- **组合爆炸闸**（`plan.rs:127-153`）：窗口半径 `√budget`，邻域判定量级 `O(N·(2r+1))` 而非 `O(N²)`。
- **对拍口径**（`reference.rs:10-37`）：相对误差在 f64 里算；非有限值先定性（同 NaN、同号 inf 算对上，其余判不一致），避免 NaN 被 `max` 当缺席漏报；阈值按 dtype 尾数宽度分档。

## 2. 业界坐标系（2026-11 联网检索）

### 2.1 路线分四类

**A. 低阶 tile DSL（程序员写 tile，编译器做低层优化）**

- **Triton**：tile 变量为一等值；Triton-IR 做 tile 级程序分析与变换；机无关 pass + 机相关 pass 分层；autotuner 首次运行搜配置；IR 级 tile 粒度融合、软件流水、warp 特化在开发中（FBTriton 分支）。
- **TileLang**：显式声明缓冲在内存层级中的位置；Layout Inference 自动并行化；自动流水推导、动态形状 tail 切分；仍保留 `annotate_layout`/`use_swizzle` 逃生舱（说明布局推导覆盖不全）。
- **CuTe DSL（CUTLASS 4）**：布局代数 `Layout/Swizzle/ComposedLayout` + 原子 `TiledCopy/TiledMMA`，编译期组合、跨代指令；Python 前端近 C++ 模板元编程性能。
- **Pallas（JAX）/ TileIR（TVM）**：算子按 tile 写；TVM TensorIR 把算法定义与调度原语分离——`split/reorder/cache_read/cache_write/compute_at/bind` 等调度原语决定 tiling、重排、缓存放置、线程绑定。

**B. Rust 原生 kernel 语言扩展**

- **CubeCL**：`#[cube]` 普通 Rust 函数 → 展开成 Rust → pliron IR → LLVM/MSL/SPIR-V/WGSL/PTX/AMDGPU/CPU SIMD；comptime 编译期特化、自动向量化、shape 特化、autotune 落盘、张量核路径按硬件自动选择；CPU 运行时可 `cargo test` 本机跑核。规模约 30 万行量级，靠通用 IR 换七后端覆盖面。
- **CuTile Rust（NVIDIA `cutile-rs`，注意与本仓 `cutile` 同名无关）**：tile 一等值 `Tensor<T,[BM,BN]>`；宿主 `partition()` 切不相交分区，借用检查器保证无数据竞争；stable Rust 1.89+、JIT 走 Tile IR；绑 NVIDIA 生态，已用于 Grout / mistral.rs。
- **cuda-oxide（NVIDIA 另一轨）**：SIMT 风格 Rust → pliron IR + LLVM → PTX；DisjointSlice + launch 契约防别名；nightly + LLVM，早期 alpha。

**C. 编译器基础设施**

- **MLIR**：Linalg/affine/vector/GPU 多 dialect；多维向量在 lowering 中逐级拆分到目标宽度；transform dialect 复现 Halide 调度。
- **Halide**：算法与调度严格分离，调度是独立于算法的一串变换。

**D. 自动调度 / 搜索**

- **Nautilus 等**：在 schedule 变换空间搜索；高层张量编译器把优化义务交给编译器，但无约束穷举不可行，至今难稳胜手写核。tile 语言则把高层优化仍留给程序员，编译器只做低层 SIMD。

### 2.2 业界已普遍具备、cutile 尚缺的能力

1. 形状符号化与多维下标（rank、stride、多维 tile 视图）；
2. dtype 档（f16/bf16/i4/i8/fp8）与量化块类型；
3. 控制流：比较、布尔、select/掩码、条件、嵌套循环；
4. 内存层级显式放置（shared/register/global）与缓存放置调度；
5. 并行维度声明与线程/块/warp 映射；
6. 循环变换集合（tiling、interchange、unroll、向量化、软件流水）；
7. 跨算子图融合 / megakernel；
8. autotune 实测收敛与配置落盘；
9. 借用作别名/独占的编译期并发安全证明；
10. 多后端 codegen 成熟度与真机性能台账。

## 3. 差距矩阵（现状 vs 标杆，按 IR 层维度）

| 维度 | cutile 现状（confirmed） | 标杆做法 | 差距性质 |
|---|---|---|---|
| dtype 泛型 | IR 仅 f32/usize；仓内 `llm-dtype` 已备 I1–I64/FP4–FP64/Bool/QType/MxType 闭集，但未接进 IR | CubeCL `Float` 泛型特化；TileLang 多档张量 | **接缝缺口**：基座已有，IR 类型闸与发射待扩 |
| 多维形状 | 单 extent、单下标 | Triton/TVM 多维 tile 与 stride | **结构缺口**：Kernel 需要 shape/stride 抽象 |
| 控制流 | 纯算术表达式，无比较/布尔/选择 | 所有 DSL 都有 select/分支 | **表达力缺口**：闭集需扩 Bool/Select 变体 |
| 归约模式 | 仅固定槽 `+=` | sum/max/min/prod/mean/argmax 族 | **算子族缺口**：归约算子与槽位泛化 |
| tile / 视图 | 无；模板链路有 4 布局构造子闭集 | CuTe 布局代数；CuTile 分区 | **两层未接通**：布局契约在 prof，IR 不消费 |
| 内存层级 | `space.rs` 有 Fast/Reg/Gm 语义与容量/对齐校验，IR 发射不映射 | TileLang 显式层级放置 | **注解进 IR 缺口** |
| 并行映射 | 无；§4.4 partition 未实现 | TVM `bind`；CuTile 分区比推 grid | **宿主模型未落地** |
| 循环变换 | 无 tiling/interchange/unroll/pipeline | TensorIR 调度原语；MLIR vector lowering | **调度层缺口**（应走闭集调度枚举，非自由 pass） |
| 融合 | 仅两核逐点表达式替换 | Triton tile 级融合；megakernel | **图级融合缺口** |
| 优化 pass | 化简/折叠/去死参/FMA 收缩 | CubeCL/MLIR 一整套分析变换 | **规模差距**，但 cutile 公理是闭集窄 pass |
| 自动调优 | roofline 估值，无实测搜索 | Triton/CubeCL autotune 落盘 | **运行期缺口**，受真机链路阻塞 |
| 并发安全 | 文档设计 `&mut` 分片，代码未落地 | CuTile 借用检查证明 | **未实现** |
| 能力协商/证据 | 需求∩供给 + 见证分离 + 三态 + 解锁路径 | 多数 DSL 假设后端可用，无此层 | **cutile 领先项** |
| 正确性证据 | 双源对拍 + 逐位定律 + 非有限定性 | 业界靠测试与数值容差 | **cutile 领先项** |
| 跨后端 | 设备无关模板 + 六档案；IR 双目标 C/.k | CubeCL 七后端；Triton 多厂 | **成熟度差距** |

## 4. 定位判断：cutile 该走哪条路（inferred）

### 4.1 三条被明确排除的路（README §0/§1.3 已认输，这里复核其成立性）

- **不学 CubeCL 的通用 IR + 多后端 codegen**：那套路线把成本转移到 pass 与每后端两条 codegen 路径上，30 万行量级；cutile 公理 1（闭集优先于表达力）就是为避开这个"范围税"。
- **不学 CuTile 绑死单厂 Tile IR**：cutile 的设备无关是硬约束（README:21），同一声明要能在 8 类设备产出代码；所有权模型本身不依赖 CUDA，值得保留，绑死 IR 不值得。
- **不学 TileLang 把布局推导藏进 pass**：推导覆盖不全就要留逃生舱，等于把不可判定性泄漏给用户；cutile 公理 3（布局是契约不是推导结果）与之相反。

### 4.2 cutile 真正的差异化位置：可审计的算子中间层

主流 DSL 优化的目标是"生成快的内核源码"。cutile 已有而业界未系统化的，是把**能力、证据、代价、正确性**做成一等产物。据此推断，cutile 应占据的位置是：

> **输入**：算子声明（形状/布局/需求）+ 设备档案（供给/峰值/层级）+ 标量参考实现；
> **输出**：一个可审计包 = 内核源码 + 能力路径（命中变体/降级档/见证状态）+ 对拍记录（相对误差/非有限定性/dtype 阈值）+ 成本上界（roofline 定界/融合增益）+ 缓存指纹与 schema 版本。

这个位置的护城河来自五条已有 confirmed 机制：协商可解释、证据分级、双源正确性、代数上界、闭集可判定。它们互相强化：**能力声明必须有见证，变换必须保位，代价必须是可证上界，越界必须显式失败**。

### 4.3 "超越 20 年"的诚实拆解

- **能领先的方向**（不依赖真机也能推进）：能力契约与证据包的形式化；变换保位的机器可证契约；闭集内新算子/新档的穷举式扩展；跨后端一致性对拍；可解释诊断。这些是工程与方法学领先，不是堆代码。
- **不能用口号领先的方向**：峰值吞吐、张量核利用率、软件流水重叠、自动调度搜索——这些必须真机矩阵（Metal/CUDA/CPU-AVX2/NEON）测量闭环，当前只能给上界（unknown，受设备链路与本机环境阻塞）。
- **20 年视角的内核判断**：DSL 的长期价值不在"又一种 shader 语法"，而在**让算子在任何后端上都可被审计、被复现、被证伪**。芯片换代时，源码可重生成，但能力台账、对拍证书、成本模型是可迁移资产。cutile 的证据分级台账（L0–L3）与双源对拍正是这个方向的雏形。

## 5. 演进路线（分阶段，尊重硬边界）

> 硬边界（既有约束，不突破）：`#![deny(unsafe_code)]`；不接 NCCL/GPU/网络/FFI；cutile 不出现厂商字面量，vendor 语法经 `shard::backend::Codegen` 缝分派；配置是输入不是职责（内核不读 `CUTILE_*`）；闭集外显式失败；门禁阈值不放宽。每阶段落地前先证无旧消费者，不留半截断裂态。

### 阶段 0 — 稳固现有原型（先收口，再扩展）

1. 接 `Image`（IR 与模板链路的图像/视图接缝），`.k` 正文经 `shard::ir` 桥到设备 dispatch 的对拍闭环（README §10.3 未接项）。
2. 把 `#[dsl]` 已拒绝的形状诊断补齐 span 与最近可行解（复用 `cap::NoPath/Unlock` 口径）。
3. 旧模板链路与 typed IR 的双轨边界写清：哪些算子族已迁 IR、哪些仍走模板，避免第二事实源。

### 阶段 1 — IR 类型与表达式扩展（接缝最小、收益最大）

1. **dtype 泛型**：`ScalarType`/`ValueType` 接 `llm-dtype` 闭集（先 f16/bf16，再 i8/i4/fp8）；缓冲元素类型化，发射按 dtype 选 C/片段拼写；对拍阈值表已在 `reference.rs` 分档。
2. **比较与选择**：表达式加 `Cmp`（EQ/LT/LE/…）出 `Bool`、`Select(cond, a, b)`；类型闸允许布尔条件表达式；发射映射到目标三元/选择指令。
3. **归约族泛化**：`Reduce` 的累加算子从 `+=` 扩到 max/min/prod/mean/argmax；槽位与初始值成显式参数。
4. **常量与标量闭集**：形状常量、广播语义进表达式层。

### 阶段 2 — 结构与形状扩展

1. **多维形状**：`Shape { rank, dims, strides }` 一等值；下标从单 `idx` 扩到多维索引表达式；视图（切片/重排）不拷贝、只改 stride。
2. **gather/scatter**：进闭集显式登记（索引缓冲 + 长度校验），不允许任意指针算术。
3. **内存层级注解**：把 `space.rs` 的 Fast/Reg/Gm 接进 IR 参数与发射；容量/对齐校验在构造期拒绝，保留现有 `check_capacity/check_alignment`。
4. **tile 视图**：先做静态形状 tile（`[BM,BN]` 固定块），对齐 CuTile 的分区粒度，但不绑厂商指令；动态 tile 留阶段 3。

### 阶段 3 — 调度层：算法/调度分离（闭集枚举，不做自由搜索）

1. 学 TensorIR/Halide 把**算法 IR** 与**调度描述**分开：算法只写逐元素/归约语义；调度是一组有限原语闭集——`split`（strip-mining）、`reorder`、`cache_read/write`、`compute_at`、`bind`（到 block/thread 槽位）、`unroll`。
2. 并行维度由调度 `bind` 声明，不由算法表达式写线程号；映射合法性（整除、槽位不重叠）在构造期校验。
3. 布局代数保持 4 构造子闭集（row/col/swz/pad），仅在闭集内允许组合；非常规布局仍显式拒绝（公理 3）。
4. 软件流水以**注解/调度原语**表达（学 FA4 的三条工程结论：非 MMA 工作被掩盖、搬运/MMA/归约分 warp、软缩放延后），不手写某代 GPU kernel。

### 阶段 4 — 优化 pass 与融合

1. 在现有 `Pass` trait/fixpoint/复验框架上扩：向量化宽度推导（按档案 SIMD 位宽）、形状/布局特化、公共子表达式、tile 级融合。
2. 跨算子图融合：以 `Kernel::fuse` 现有的只读链接校验为种子，扩到算子链；融合仍走三态判决（可拼/结构否掉/无从打分）。
3. pass 仍守"要么严格收缩、要么幂等"契约；改变舍入的优化（FMA/重结合）继续 opt-in 单独通道。

### 阶段 5 — 宿主并发安全与自动调优

1. 落地 §4.4 `partition()`：分区不相交、`&` 只读 / `&mut` 独占在构造期或借用检查期保证；分区比推 grid，不整除构造期报错；同一套模式服务 8 类设备。
2. autotune：以 roofline 上界做候选排序，首次真机实测收敛，配置按指纹落盘（复用现有 cache 指纹 schema）；未测设备回退上界并标 `Unscorable`，不猜实测性能。

### 阶段 6 — 后端收敛与旧链路退役

IR 稳定且每族对拍闭环后，逐族删除对应 `.k` 模板与重复 lowering；多 target lowering 走既有 `Target` 枚举模式扩展（C/K 之外加 vendor 目标）；旧模板链路只保留未迁算子族。

## 6. 风险与反证（避免把推断当事实）

- **闭集天花板**：算子/模式增长到闭集装不下时，要么扩闭集（带 schema 递增与缓存失效），要么破公理 1。需要一个可度量的"闭集覆盖率"台账：声明算子面 vs IR 可表达族。
- **双轨维护成本**：模板链路与 typed IR 并存期，同一算子可能有两份 lowering；阶段 0 的边界登记是前置条件。
- **性能主张边界**：当前所有性能相关结论是结构上界；在真机矩阵接入前，报告不得出现实测加速比承诺。
- **L5/L6 张力**：设备 vendor 语法、FFI 句柄不得进 cutile；调度原语落到设备指令的部分归 shard/engine，cutile 只出与厂商无关的中间描述。
- **规模纪律**：演进各阶段受仓库密度预算与文件下限约束；新能力优先加在既有模块的闭集表里，不靠堆新文件；结构性拆 crate 需 ADR。
- **同名混淆**：NVIDIA `cutile-rs`（CuTile Rust）与本仓 `cutile` 无代码关系，文档与对外沟通须显式区分。

## 7. 未知与最小验证动作

| 未知项 | 最小验证动作 |
|---|---|
| IR 扩 dtype 后各发射目标的拼写完备性 | 先 f16 一族：加类型闸 → C/.k 发射 → reference 阈值对拍 → 差分执行 |
| 多维形状 stride 与视图不拷贝的合法性 | 先静态 rank-2 行主序切片，构造期校验 stride 整除与重叠 |
| 调度原语闭集能否覆盖 LLM 算子面 | 把现有 22 条声明算子的手工调度逐一映射到候选原语，映射不上的列入闭集增补 |
| 宿主 partition 并发安全在非 CUDA 后端的可用性 | CPU/Metal 两后端各写一个分区独占对拍测试，验证不依赖厂商线程模型 |
| autotune 实测与 roofline 上界的偏差区间 | 同一算子在有真机的后端跑 N 组 tile 配置，记录上界/实测比，标定模型可信度 |

## 8. 参考来源（联网检索，2026-11-24）

- CubeCL：多平台 Rust 计算语言扩展，`#[cube]` → IR → 多后端 codegen，comptime/自动向量化/autotune。https://crates.io/crates/cubecl/0.9.0 ；https://crates.io/crates/cubecl-std/0.12.0-pre.1 ；https://github.com/tracel-ai/cubecl
- NVIDIA CUDA Rust 双轨：cuda-oxide（SIMT/PLIRON/LLVM→PTX）与 cutile-rs（Tile 模型/stable Rust/所有权分区）。https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels
- CuTile Rust 并发安全：tile 模型 + 别名 XOR 可变性 + 运行期检查落点。https://arxiv.org/html/2606.15991
- Triton：tile 程序 IR、机无关/机相关 pass、autotuner；warp 特化/软件流水/tile 融合在研。https://triton-lang.org/main/programming-guide/chapter-2/related-work.html ；https://facebookexperimental.github.io/triton/website/triton.html
- TileLang：tile 级语言、布局显式放置、Layout Inference、自动流水与 tail 切分。https://arxiv.org/html/2504.17577v2
- TVM TensorIR：算法/调度分离，split/reorder/cache/compute_at/bind 原语。https://dl.acm.org/doi/full/10.1145/3695053.3731096
- CuTe DSL：布局代数与 TiledCopy/TiledMMA 原子，CUTLASS 4 Python 前端。https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/
- MLIR：多维向量逐级拆分到目标宽度，transform dialect 复现 Halide 调度。https://mlir.llvm.org/docs/Tutorials/transform/ChH/
- 自动调度现状：高层张量编译器 schedule 空间穷举不可行；tile 语言保留高层优化给程序员。https://arxiv.org/html/2604.14825

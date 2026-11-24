# 0035：unsafe/FFI 安全边界规则（P3 动工前置裁决，缺口 G13）

- 状态：Accepted（2026-11-18 评审通过，G1/G2/G3 三处订正已并入；作为 M3-1 前置约束，边界 crate 实现随 M3 填入）
- 日期：2026-11-17
- 相关：[0025](0025-runtime-backend-abstraction.md)（后端三件套、trait 归属与依赖方向、§9 unsafe/FFI 预告）、ADR 0028（运行时后端落点与触发条件）、ADR 0001（依赖方向）

## 背景

workspace lints 在根 [`../../Cargo.toml`](../../Cargo.toml) 全局 `unsafe_code = "deny"`；全仓生产核心不出现裸指针与 FFI。当时的唯一生产 unsafe/FFI 岛是 storage 的只读内存映射（存量白名单见 §6） [`../../crates/storage/src/lib.rs`](../../crates/storage/src/lib.rs) 的内联 `mmap` 模块：模块级 `#![allow(unsafe_code)]`，对外只暴露拥有型、只读的 `&[u8]` 视图，不变量（只读、私有映射、`Drop` 归还、线程归属）写在模块头注释里。

P3 要在边界 crate 挂 Metal/CUDA 后端（ADR 0025 §1、§9；ADR 0028 §3 触发条件）：设备句柄、命令队列、缓冲映射、kernel launch 与事件同步必然触及裸指针、设备句柄与平台 FFI。ADR 0025 §9 只预告「unsafe/FFI 只允许出现在后端边界 crate」，没有给出白名单式的边界规则：哪些函数/句柄可以越过安全边界、`allow` 收敛在哪一层、不变量由谁写、校验放哪一侧。规则不定，后端代码里就会到处散落 `#[allow(unsafe_code)]`，审计面无法枚举，也与既有 mmap 岛的收敛形态不一致。

## 决策

### 1. 越界对象白名单：只收在边界模块

可越过 Rust 安全边界的对象收敛到边界 crate 的**边界模块**内，白名单穷尽为四类：

- **裸指针解引用**：从 FFI 取得或落到 FFI 所需的指针，只在边界模块内创建、解引用与切片重建；
- **设备缓冲句柄**：Metal/CUDA 设备侧缓冲、命令队列、事件等跨调用存活的设备私有句柄，藏在边界 crate 内部类型里；
- **FFI 调用**：平台库（Metal/CUDA 驱动或绑定层）的 `extern` 声明与调用，只在边界模块；
- **mmap 映射**：沿用现有 storage mmap 岛形态，新增映射类越界同样以独立模块隔离。
- **通信/进程级库句柄**：NCCL communicator、MPI comm、Metal 对等组等跨调用存活的进程级库句柄，与设备缓冲句柄同等对待——藏在边界 crate 内部类型里，经 `Collective` trait 向上只暴露不透明标识；M3-4 动工前若出现新句柄形态，回本 ADR 扩白名单。

边界 crate 实现运行时一侧已定形的四个 trait：`engine::Executor`、`engine::Program`、`storage::Allocator`、`shard::Collective`；`engine::Device` 是数据 struct（厂商标签），不是 trait。这些 trait 只持**不透明标识**：trait 方法签名不出现裸指针与设备句柄类型；缓冲句柄以边界内不透明类型向上传递，模型、调度、算子代码拿不到指针与厂商句柄（ADR 0025 §2、§7）。

### 2. `#[allow(unsafe_code)]` 的收敛点

- 允许点只出现在边界 crate 的边界模块或边界函数；模块头注释必须写明：越界理由、适用的安全不变量（指针生命周期、对齐、有效性、线程归属）、对外暴露的安全类型是什么。
- 其余核心继续保持 workspace `unsafe_code = "deny"`；不在核心成员（`engine` / `tensor` / `storage` 除既有 mmap 岛外 / `op` / `nn` / `shard` / `serve` 等）新增 `allow(unsafe_code)`。
- 既有 storage mmap 岛是本规则的既有实例：模块级 allow + 头注释不变量 + 对外只读视图；新边界模块照同一形态写。存量越界点的全表见 §6。

### 3. 依赖方向

边界 crate 只能指向运行时 trait；trait 定义归运行时一侧（ADR 0025 §1），实现归边界 crate，不允许反向引用：边界 crate 不定义新的运行时 trait，运行时一侧不持有边界类型。依赖方向对 `engine`/`storage`/`shard` 三个运行时 crate 同时成立：边界 crate 只指向三者的 trait，三者均不得反向持有边界类型。方向冲突时先扩 trait 定义，不在上层加厂商特判（ADR 0025 §4、§7）。

### 4. 不安全块最小化

- `unsafe` 块内只做越过安全边界的操作（解引用、FFI 调用、`from_raw_parts` 类重建）；长度、对齐、非空、生命周期、句柄有效性等安全校验放在块外安全代码，校验通过后再进入块内越界操作。
- 校验失败走既有 [`diagnostic`](../../crates/diagnostic/src/code.rs) 错误类别闭集（`Unsupported` / `Invalid` / `Missing` / `Io` / `Internal`），不绕过既有错误类别自造 panic 或裸 `unwrap`；越界结果对外仍以 `Result` 暴露。
- `unsafe impl`（如跨线程归属）只在不变量于头注释写明且被类型状态保证时书写，同样收在边界模块内。

### 5. 与分层、数据/机制边界的关系

本 ADR 不改变 ADR 0025 的分层归属与 ADR 0028 的数据/机制边界：L2 片段与档案、L3 机制编排、运行时后端落点的归属裁决照旧；本 ADR 只补充「哪些操作被许可越过安全边界、许可收敛在哪一层」的规则，不新增成员、不改 trait 形状。

### 6. 现状越界点（可枚举白名单）

§2 管的是「新增」；存量越界点如果不列举，这条规则就只剩口头约束。`crates/*/src` 里 `allow(unsafe_code)`
的现状穷尽为下表，由 [`../../crates/testkit/src/boundary.rs`](../../crates/testkit/src/boundary.rs)
逐文件比对：多一行（新岛）或少一行（基线过期）都失败。`crates/*/tests` 不在口径内——测试夹具不是越界岛。

| 位置 | 越界对象（§1 类别） | 不变量与对外视图 |
|---|---|---|
| `crates/storage/src/mmap.rs`（模块级，1 处） | mmap 映射 | 模块级 allow；只读、私有映射、`Drop` 归还、线程归属；对外只暴露拥有型 `&[u8]` 视图 |
| `crates/engine/src/metal.rs`（模块级，1 处） | FFI 调用 + 设备缓冲句柄 | 手写 `objc_msgSend` 与 Metal 运行时符号；+1 保留的对象在 `Drop` 恰好 `-release` 一次，句柄私有、不跨线程，自动释放对象由 autorelease 池收口；对外是 `Executor` / `Allocator` 的安全实现 |
| `crates/nn/src/kernel/fast/i8.rs`（函数级，4 处） | 裸指针切片重建（打包/计算包装） | i8 sdot 瓦片体 `sdot_tile` / `dot_one` / `harvest` 与端口探针已迁入 `prof::native::i8`；nn 侧只剩面板打包与行段计算的裸指针重建包装，长度守卫由调用方打包布局保证 |
| `crates/nn/src/kernel/fast/pool.rs`（函数级，3 处） | 跨线程发布裸指针 | 常驻 GEMM 线程池：任务只发布裸指针 + 一次闸门；裸指针只活到 `done` 齐、主线程返回前 |
| `crates/prof/src/native/gemm8x8.rs`（函数级，11 处） | 裸指针切片重建 + NEON/AVX2 intrinsics | f32 8×8 寄存器块微内核；`fma_step` 生产与探针共用；nn `kernel_fast_micro.rs` 的瓦片体迁出后 nn 岛已清空，本岛是唯一宿主 |
| `crates/prof/src/native/i8.rs`（模块级，1 处） | 裸指针 + aarch64 dotprod `sdot` intrinsics | i8 4×4 瓦片体 `sdot_tile` 由生产微内核与取数受限探针共用同一份；模块级一处 `#![allow(unsafe_code)]` 放行整岛，头注释载探测与读边界不变量；安全入口先做运行时探测与形状守卫再进 `unsafe` 工人 |
| `crates/prof/src/native/sme.rs`（函数级，8 处） | 内联汇编 + SME intrinsics | aarch64 SME 16×64 ZA 外积；nightly + `target_arch = "aarch64"` 门控；nn `kernel_fast_sme.rs` 迁出后已删，本岛是唯一宿主 |

`gemm` 那几行是**存量冻结**：只许减少，不许增加。要新增一行，先改本表，并说明这个越界点为什么
不能落在边界 crate。拆文件不改变冻结口径——`fast` 平铺拆成 `kernel_fast_{i8,micro,pool,sme}.rs`
后按文件重新列举，
总数由 17 涨到 21 是拆分时每个微内核各自重建切片的重复，**属于需要往下削的欠账而不是新增许可**；当前各行相加 **29 处**（nn `i8` 4 + `pool` 3 = 7，prof native `gemm8x8` 11 + `i8` 1 + `sme` 8 = 20，device 1，storage 1）。f32 `micro`、SME `sme`、i8 `sdot` 三条瓦片通道已从 nn 迁入 `prof::native` 隔离子岛，nn 侧对应岛清空或删除，prof 三岛是唯一宿主——原「双份过渡态」至此收口，prof 计数是迁出后的存量而非新增越界面。此前文首写「总计 26」与各行相加 27 已不符，本轮一并改正为按表求和。另删除一处**空头许可**：`crates/nn/src/kernel/fast/probe.rs` 曾有模块级 `#![allow(unsafe_code)]` 而文件内`unsafe` 用量为 0（`git grep unsafe` 只命中那一行本身），已删——白名单里挂一个不越界的文件，等于把「可枚举」这条判据本身弄假。

`sme.rs` 由 4 涨到 8 是**唯一一次有意的增量**：四个探针入口（`port_run`/`port_mix` 各一个安全包装 + 一个 `#[target_feature]` 工人）不是新的越界面，而是把已在岛内的 `fmopa` 与 `ld1w` 路径暴露成可计时的循环，用来给出 GF/s 的分母与**取数代价**（否则 K7 只能拿 NEON 峰值当分母，那是口径混用）。它们受 `bench` 特性门控，生产公开面不含该项，不变量与 `sme_tile` 一致：包装只做 `available()` 判定与指针透传，工人只为 `smstart`/`smstop` 配对与读边界负责，`port_run` 完全不接触内存，`port_mix` 只读调用方给的 `&[f32; 16]` 与 `&[f32; 64]`（各恰好一次读满，无越界）。

`i8.rs` 由 8 涨到 12、`micro.rs` 由 6 涨到 11 是同一形态的**有意增量**：把生产微内核的瓦片体抽成`sdot_tile`（i8）与 `fma_step`（f32）后，探针与被测内核共用一份实现，代价是每个新函数各自一个 `#[allow]`，收益是**消掉一份重复的瓦片体**——若探针自写一份，分母就与分子不同源，K7/K8 变成两个内核相除。其中 `i8.rs` 的 4 处是上一轮合并 `sdot_tile` 时新增、本轮才补登记（欠账，已补）。新增的 f32 探针只在 `bench` 特性下编译，生产公开面不含该项；不变量与 `micro_packed` 一致：包装只做长度守卫与指针透传，工人只为 `fma_step` 的读边界（`NR*k` 内的 `NR` 行）负责。

三个逻辑岛：`storage` 的 mmap、`device` 的 Metal FFI、CPU 微内核——瓦片 intrinsic 段已收进
`prof::native` 三岛（`gemm8x8` / `i8` / `sme`），`nn` 这一组只剩 `i8` 的打包/计算裸指针包装
与 `pool` 线程池（裸指针只活到 `done` 齐、主线程返回前）；线程池与并发机制仍留 nn，属调度骨架
而非设备原语。

## 备选方案

- **核心整体放开 `unsafe_code`**：否决。攻击面与审计面随每个后端线性扩散，无法枚举审查；与 workspace deny 基线背离，既有 mmap 岛的隔离形态也失去意义。
- **每个后端文件自行 `#[allow(unsafe_code)]`**：否决。allow 点发散到文件粒度，审计时要逐文件确认不变量；白名单无法收敛，新增后端时容易在非边界模块长出越界代码。
- **把 FFI 绑定层直接暴露给 engine trait**：否决。trait 签名泄漏指针/句柄会把越界面带进运行时核心，违反 §1 不透明标识规则与 §3 依赖方向。

## 后果

### 正面

- 越界点可枚举：`allow(unsafe_code)` 与 `unsafe` 块的位置收敛到边界模块，审计时只检查边界白名单与头注释不变量；核心新增 `unsafe` 由 deny 门拦截。
- 边界类型对外是安全封装：上层代码只经 trait 与不透明句柄工作，后端替换不影响模型与调度。

### 负面（代价）

- 边界模块必须承担安全不变量文档：指针生命周期、对齐、有效性、线程归属写不全，模块头注释即视为不落地；评审时逐条核对。
- 边界 crate 需要在安全包装层投入样板代码（校验 + 封装 + `Result` 翻译），薄 FFI 直调路径被禁止。

### 取舍

可审计性优先于薄封装的开发速度；本规则只覆盖越界许可，不规定后端内部组织。

### 重议条件

- 新增第二类设备后端（除 Metal/CUDA 外，如 NPU/DNN 栈）时，复核白名单四类是否够覆盖；出现新越界对象（新映射语义、新线程模型等）须先回本 ADR 扩白名单，再动边界代码。
- 既有 storage mmap 岛若需要扩展可写映射或新映射语义，按本规则复核其不变量与对外视图。

## 不做

- 本轮不实现 Metal/CUDA 后端，不引入 GPU 依赖、FFI crate 或边界 crate 骨架（P3）。
- 不改 ADR 0025 的 trait 形状与 ADR 0028 的落点裁决；trait 暴露不足时先扩运行时 trait 定义。
- 不定义后端内部模块组织（系统绑定 vs 生成绑定的具体绑定层形态除外，见下）。
- **FFI 绑定默认策略（M3-1 前置）**：在独立 ADR 批准引入 FFI 绑定依赖（如 metal-rs / cuda-sys）之前，边界 crate 一律**手写 `extern "C"` 声明**，不引入任何 FFI crate；Metal 走 objc/Metal 运行时符号，CUDA 走 Driver API（`cuInit`/`cuMemAlloc`/`cuLaunchKernel` 等）手写绑定。若某后端证明手写绑定成本不可接受，需另立 ADR 申请 D5 例外，通过前维持零依赖。

## 修订

- 2026-11-17：初稿，回应 G13；白名单四类、allow 收敛点、依赖方向、unsafe 块最小化规则落档，实现随 P3 边界 crate。
- 2026-11-18：评审通过（G1/G2/G3）并置 Accepted：① trait 清单订正为四 trait（`Executor`/`Program`/`Allocator`/`Collective`）且 `Device` 为 struct；② 白名单增第 5 类「通信/进程级库句柄」；③ 补 FFI 绑定默认策略（手写 `extern "C"`、零依赖）。
- 2026-11-21：补 §6「现状越界点」：四个文件（`storage` mmap、`device::metal::executor`、`op::fast` 两处）穷尽为可执行白名单，门禁落在 `testkit/tests/architecture.rs`；`op` 的 14 处明确为 P1 待搬出的存量冻结。
- 2026-11-22：`op::fast` 增 x86_64 AVX2/FMA 微内核（与 NEON 同一组不变量，仍属同一逻辑岛），`fast.rs` 11 → 14 处，`op` 合计 17 处；白名单与门禁同步更新，未放宽任何阈值。
- 2026-11-23：`nn::kernel` 由 `kernel/` 目录合并回单文件 `kernel.rs`（`fast` 四段 `i8`/`micro`/`pool`/`sme` 的 34 处越界点随合并回 `kernel.rs`），§6 表格与 `testkit/src/boundary.rs` 白名单同步改回单行；总处数仍为 36，存量冻结口径不变。
- 2026-11-23：`nn::kernel` 由单文件 `kernel.rs` 平铺为「门面 + 每族一文件」（`kernel_fast.rs` 等 15 个 `kernel_*.rs`、`conv_*.rs` 8 个），`unsafe` 岛随 `fast` 段落到 `kernel_fast.rs`；§6 表格与白名单同步到新路径，总处数仍为 36，存量冻结口径不变。
- 2026-11-24：`fast` 各段由 `kernel_fast_*.rs` 平铺文件收拢进 `kernel/fast/` 目录（`i8.rs`、`pool.rs` 等），越界点随文件迁移、处数不变（i8 4 + pool 3）；§6 表格与白名单同步到目录路径，存量冻结口径不变。

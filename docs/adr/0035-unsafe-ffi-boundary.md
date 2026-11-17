# 0035：unsafe/FFI 安全边界规则（P3 动工前置裁决，缺口 G13）

- 状态：接受（P3 前置裁决；边界 crate 实现随 P3 填入，本轮不改任何 `src` 代码）
- 日期：2026-11-17
- 相关：[0025](0025-runtime-backend-abstraction.md)（后端三件套、trait 归属与依赖方向、§9 unsafe/FFI 预告）、[0028](0028-device-data-vs-mechanism-boundary.md)（运行时后端落点与触发条件）、[0001](0001-layered-dependency-order.md)（依赖方向）；设计取证见 [`../backend-runtime-design.md`](../backend-runtime-design.md) §3.3 G13、§5、§7 P3 前置步骤

## 背景

workspace lints 在根 [`../../Cargo.toml`](../../Cargo.toml) 全局 `unsafe_code = "deny"`；全仓生产核心不出现裸指针与 FFI。唯一生产 unsafe/FFI 岛是 storage 的只读内存映射 [`../../crates/storage/src/mmap.rs`](../../crates/storage/src/mmap.rs)：模块级 `#![allow(unsafe_code)]`，对外只暴露拥有型、只读的 `&[u8]` 视图，不变量（只读、私有映射、`Drop` 归还、线程归属）写在模块头注释里。

P3 要在边界 crate 挂 Metal/CUDA 后端（ADR 0025 §1、§9；ADR 0028 §3 触发条件）：设备句柄、命令队列、缓冲映射、kernel launch 与事件同步必然触及裸指针、设备句柄与平台 FFI。ADR 0025 §9 只预告「unsafe/FFI 只允许出现在后端边界 crate」，没有给出白名单式的边界规则：哪些函数/句柄可以越过安全边界、`allow` 收敛在哪一层、不变量由谁写、校验放哪一侧。规则不定，后端代码里就会到处散落 `#[allow(unsafe_code)]`，审计面无法枚举，也与既有 mmap 岛的收敛形态不一致。

## 决策

### 1. 越界对象白名单：只收在边界模块

可越过 Rust 安全边界的对象收敛到边界 crate 的**边界模块**内，白名单穷尽为四类：

- **裸指针解引用**：从 FFI 取得或落到 FFI 所需的指针，只在边界模块内创建、解引用与切片重建；
- **设备缓冲句柄**：Metal/CUDA 设备侧缓冲、命令队列、事件等跨调用存活的设备私有句柄，藏在边界 crate 内部类型里；
- **FFI 调用**：平台库（Metal/CUDA 驱动或绑定层）的 `extern` 声明与调用，只在边界模块；
- **mmap 映射**：沿用现有 storage mmap 岛形态，新增映射类越界同样以独立模块隔离。

核心运行时 trait（`Device` / `Allocator` / `Executor`，ADR 0025 §1）只持**不透明标识**：trait 方法签名不出现裸指针与设备句柄类型；缓冲句柄以边界内不透明类型向上传递，模型、调度、算子代码拿不到指针与厂商句柄（ADR 0025 §2、§7）。

### 2. `#[allow(unsafe_code)]` 的收敛点

- 允许点只出现在边界 crate 的边界模块或边界函数；模块头注释必须写明：越界理由、适用的安全不变量（指针生命周期、对齐、有效性、线程归属）、对外暴露的安全类型是什么。
- 其余核心继续保持 workspace `unsafe_code = "deny"`；不在核心成员（`engine` / `tensor` / `storage` 除既有 mmap 岛外 / `op` / `nn` / `shard` / `serve` 等）新增 `allow(unsafe_code)`。
- 既有 storage mmap 岛是本规则的既有实例：模块级 allow + 头注释不变量 + 对外只读视图；新边界模块照同一形态写。

### 3. 依赖方向

边界 crate 只能指向运行时 trait；trait 定义归运行时一侧（ADR 0025 §1），实现归边界 crate，不允许反向引用：边界 crate 不定义新的运行时 trait，运行时一侧不持有边界类型。方向冲突时先扩 trait 定义，不在上层加厂商特判（ADR 0025 §4、§7）。

### 4. 不安全块最小化

- `unsafe` 块内只做越过安全边界的操作（解引用、FFI 调用、`from_raw_parts` 类重建）；长度、对齐、非空、生命周期、句柄有效性等安全校验放在块外安全代码，校验通过后再进入块内越界操作。
- 校验失败走既有 [`diagnostic`](../../crates/diagnostic/src/code.rs) 错误类别闭集（`Unsupported` / `Invalid` / `Missing` / `Io` / `Internal`），不绕过既有错误类别自造 panic 或裸 `unwrap`；越界结果对外仍以 `Result` 暴露。
- `unsafe impl`（如跨线程归属）只在不变量于头注释写明且被类型状态保证时书写，同样收在边界模块内。

### 5. 与分层、数据/机制边界的关系

本 ADR 不改变 ADR 0025 的分层归属与 ADR 0028 的数据/机制边界：L2 片段与档案、L3 机制编排、运行时后端落点的归属裁决照旧；本 ADR 只补充「哪些操作被许可越过安全边界、许可收敛在哪一层」的规则，不新增成员、不改 trait 形状。

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
- 不定义后端内部模块组织与 FFI 绑定选型（系统绑定 vs 生成绑定）。

## 修订

- 2026-11-17：初稿，回应 G13；白名单四类、allow 收敛点、依赖方向、unsafe 块最小化规则落档，实现随 P3 边界 crate。

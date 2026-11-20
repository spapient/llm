# 0025：运行时后端抽象与多后端预留边界

- 状态：接受（P1 地基 trait 形状与 CPU 安全实现已落地；真实 CPU 内核、P3 Metal/CUDA 边界实现随后续阶段填入）
- 日期：2026-11-15
- 相关：ADR 0003（kernel 侧与运行时类型分离）、ADR 0012（量化权重同源）、[0017](0017-kernel-ast-and-ir-freeze.md)（无 IR 冻结）

## 背景

路线已定两个方向：**速度优先**；**Metal 与 CUDA 双后端并行**。当前运行时是契约级原型，相关现状（回链源码）：

- 后端只有 [`Executor`] trait 雏形：`submit(graph, stream)`，没有设备、缓冲分配或事件同步（`crates/engine/src/exec.rs:136-143`）。
- 通信只有 [`Collective`] trait：`all_reduce`，无通信后端实现（`crates/shard/src/collective.rs:11-14`）。
- 张量是只读 `&'a [u8]` 视图，无可写缓冲、无分配器、无跨设备拷贝（`crates/tensor/src/lib.rs:126-131`）。
- KV 只做页表/存活位图簿记，页内不持有数据（`crates/engine/src/kv.rs:3,7`）。
- 算子注册是编译期闭集 [`REGISTRY`]，只持声明/模板/标量参考，没有「算子 × 后端」的实现挂载点（`crates/op/src/lib.rs:122`）。
- 全仓生产核心 `unsafe_code = "deny"`，唯一生产 FFI 是 `storage` 的 mmap（`Cargo.toml:56`）。

双后端并行最大的返工风险是：抽象层不稳，Metal 与 CUDA 各自分叉出私有设备假设，第二套实现要回改所有算子、模型与调度。本 ADR 在 P0 把通向多后端的接口形状定死；第一版可以只有 CPU 实现，但 trait 形状必须能同时容纳 Metal 与 CUDA。

## 决策

### 1. 后端抽象三件套

运行时侧定义三个 trait，边界 crate 提供实现：

- `Device`：设备标识与能力（引用 `prof` 的设备档案闭集），不持有设备私有句柄的具体类型；
- `Allocator`：缓冲分配、释放、跨设备拷贝；缓冲句柄是后端边界内的不透明类型；
- `Executor`：图提交到流（扩展现有 trait，补流同步与设备关联的形状）。

trait 定义归运行时一侧；具体 crate 归属在 P1 落地时按 [`../ARCHITECTURE.md`](../ARCHITECTURE.md) 层表定。后端实现放在**独立后端边界 crate**，依赖方向只能从边界 crate 指向运行时 trait，禁止反向（ADR 0001）。

**P1 归属裁决（2026-11-15 落地）**：

- `Buffer`（拥有的扁平字节缓冲）、`Allocator`（`alloc` / `free` / 显式 `copy_into`）与 `CpuAllocator`（安全 `Vec<u8>`，无 `unsafe`）归 `storage`（L4 内存/布局层）；
- `OwnedTensor`（元数据 + 持有字节向量）归 `tensor`（L4 元数据层）。`tensor` 不反向依赖 `storage`：分配器产出字节后，由编排层把字节向量交给 `OwnedTensor::from_bytes`，两层只经字节握手；
- `Device`（厂商 / 架构轻量标签）与扩展后的 `Executor` 归 `engine`（L6）。`Device` 是 `String` 标签，不直接引用 `prof` 档案类型（避免给 `engine` 新增 `prof` 依赖边）；标签与档案闭集的对应在编排层 / 后端边界完成。
- `Executor` 形状：`device(&self) -> &Device`、`submit(&mut self, &Graph, Stream)`、`sync(&mut self, Stream)`；`sync` 提供空默认实现，CPU 可立即返回，设备后端在此做事件同步。

### 2. 张量双形态

- 保留只读 `Tensor<'a>` 视图作为零拷贝读路径，语义不变；
- 新增拥有缓冲的可写张量形态，`dtype` / `shape` / `strides` 与设备无关；设备地址（CPU 指针或设备句柄）藏在 `Allocator` 返回的不透明句柄里，不泄漏到模型与调度层。

### 3. 内存生命周期

缓冲状态分拥有 / 借用 / 静态借用；分配与释放只经过 `Allocator`。跨设备拷贝是 `Allocator` 的显式方法，框架层不发生隐式拷贝。零拷贝视图贯穿读路径（衔接 ADR 0003：kernel 侧操作数视图与运行时张量不混用）。

### 4. 算子多后端挂载

[`REGISTRY`] 继续只持声明、模板、标量参考；后端内核实现按 `(算子名, 后端)` 键挂载到后端侧。算子选择只走能力闭集（设备档案见证过的能力），模型与调度代码里不出现 `if Metal {} else if CUDA {}` 的厂商特判。

### 5. KV 数据接口

`Page` / `Table` / `Cache` 簿记结构不变；预留「页 → 缓冲」的绑定接口（不透明页句柄），P2 填真实缓冲实现。簿记不关心缓冲在 CPU 还是设备，注意力算子从接口取页，不直接碰设备内存。

### 6. 通信边界

[`Collective`] trait 保留为通信后端边界；`Group` / `Topology` 是纯拓扑簿记，与实现分离。P3 再填 Metal / CUDA（或 NCCL）通信实现，拓扑层不改。

### 7. 调度与后端解耦

调度器与准入只依赖 `Executor` / `Allocator` trait 与 KV 簿记，不引用具体后端类型；continuous batching 策略可替换，换后端不改调度。

### 8. 产物缓存

沿用 `cutile` 现有内容寻址缓存（`CacheKey` / `ArtifactKey` / 复用计数，`crates/cutile/src/cache/`）；后端特化产物继续挂在这套 key/artifact 上，不为每种后端另立状态台账。

### 9. unsafe 与 FFI 边界

Metal / CUDA 的 FFI 与 `unsafe` 只允许出现在后端边界 crate；生产核心继续保持 `unsafe deny`，靠边界模块局部 `allow` 收敛。边界规则（哪些函数、哪些句柄可以越过安全边界）由 P3 动工前的单独 ADR 细化。

## 后果

- 新增一个后端 = 写一个边界 crate + 挂载内核，不改 `engine` / `tensor` / `shard` / `nn`。
- P1 先出 CPU 实现填充这些 trait；P3 Metal 与 CUDA 经同一套 trait 并行挂载。
- trait 形状在 P1 暴露不足时，先扩 trait 定义，不在上层加后端特判分支。
- 图仍是提交单元，不做算子融合或重排（ADR 0017 无 IR 冻结）。

## 不做

- 本 ADR 不实现任何后端，不引入 GPU 依赖，不改张量只读语义（可写形态随 P1 落地）。
- 量化反量化内核的挂载衔接 ADR 0012，不在本 ADR 展开。
- 不定义第二套设备能力闭集；设备能力继续以 `prof` 档案与 L0–L3 证据分级为准。

## 修订

- 2026-11-15：初稿，P0 落档。P1 CPU 实现落地后回填 trait 最终形状；P3 后端边界规则另出 ADR。
- 2026-11-15：P1 地基落地。`storage::Buffer` / `Allocator` / `CpuAllocator`、`tensor::OwnedTensor`、`engine::Device` 与扩展 `Executor`（`device` / `submit` / `sync`）已实现并通过单测；归属与依赖方向按层表裁决见 §1。可写张量形态落地（§2），设备句柄仍不泄漏到模型与调度层；真实 CPU 算子内核、KV 页缓冲绑定与跨设备拷贝实现随 P1 后续子步填入。

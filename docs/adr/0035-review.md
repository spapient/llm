# 0035 评审：unsafe/FFI 边界规则（M3-1 前置）

- 日期：2026-11-18
- 状态：评审结论（**有条件通过**；3 处订正后可由你拍板 Accepted，本评审不自行翻转 ADR 状态）
- 依据：`0035-unsafe-ffi-boundary.md`（Proposed）、`0025-runtime-backend-abstraction.md`（Accepted）、`implementation-task-plan.md` §4 D5、`backlog-status-2026-11-17.md` §7 M3 前置、当前工作树 trait 实测

## 0. 结论

ADR-0035 的四条核心规则——越界对象白名单、`allow` 收敛点、依赖方向、unsafe 块最小化——**方向正确且与既有 mmap 岛形态一致**，可直接作为 M3-1 的前置。但有 **3 处缺口**在 `Accepted` 前必须订正，否则 M3-1/M3-4 一动手就会撞上规则没覆盖的对象：

| # | 缺口 | 一句话 | 不订正的后果 |
|---|---|---|---|
| G1 | trait 清单陈旧 | §1 只写 `Device/Allocator/Executor`，漏 `Program`/`Collective`，且 `Device` 是 struct 不是 trait | 边界 crate 该指向谁/该实现谁说不清，依赖方向规则没覆盖 `shard::Collective` |
| G2 | 白名单漏通信句柄 | 四类白名单只覆盖设备 FFI，未覆盖 M3-4 的 NCCL/MPI/Metal 对等通信句柄 | 通信后端一落地就长出白名单外句柄，规则失守 |
| G3 | FFI 绑定选型被搁置但 D5 卡死 | §不做把「手写绑定 vs 生成绑定」推后，但零生产依赖已裁定必须手写或走 ADR 放宽 | 写第一行 FFI 前仍无据可依，可能直接破坏零依赖基线 |

## 1. 通过项（已验证，无需改）

- **白名单收敛形态**：与 `storage/src/mmap.rs` 既有岛一致——模块级 `#![allow(unsafe_code)]` + 头注释不变量 + 对外只读安全视图（`mmap.rs:1-13`）。新边界模块照此形态写，正确。
- **unsafe 块最小化**：校验在块外、越界在块内、错误走 `diagnostic::Code` 闭集、对外 `Result`（§4）。与工作树现状一致：全仓生产成员 `deny(unsafe_code)`，唯一 `allow` 在 `mmap.rs`。
- **依赖方向**：边界 crate → 运行时 trait，禁止反向（§3），与 ADR-0001 分层一致，且当前无 backend 边界 crate，方向不会被既有成员污染。
- **不改变 ADR-0025/0028 分层与落点**（§5）：正确，评审未发现冲突。

## 2. 订正项

### G1 — trait 清单陈旧，须枚举四个 trait 且修正 `Device` 定性

**证据**（工作树实测）：

- `engine/src/exec.rs:334` `pub trait Executor`（`device/submit/sync`）
- `engine/src/exec.rs:573` `pub trait Program`（算子内核 trait）
- `engine/src/exec.rs:305` `pub struct Device`（**struct，不是 trait**）
- `storage/src/lib.rs:171` `pub trait Allocator`（`alloc/free/copy_into`）
- `shard/src/collective.rs:11` `pub trait Collective`（`all_reduce`）

ADR-0035 §1 写「核心运行时 trait（`Device` / `Allocator` / `Executor`，ADR 0025 §1）」——漏了 `Program` 与 `Collective`，且把 `Device` 当 trait。边界 crate 实际要实现的 trait 是 `Executor`/`Allocator`/`Program`/`Collective`，分布在同一运行时一侧的 **三个 crate**（`engine` / `storage` / `shard`），不是"单一运行时一侧"。

**建议修订**（并入 §1 与 §3）：

> 边界 crate 实现运行时一侧已定形的四个 trait：`engine::Executor`、`engine::Program`、`storage::Allocator`、`shard::Collective`；`engine::Device` 是数据 struct（厂商标签），不是 trait。§3 依赖方向对上述三个运行时 crate 同时成立：边界 crate 只指向 `engine`/`storage`/`shard` 的 trait，三者均不得反向持有边界类型。

### G2 — 白名单漏通信句柄

**证据**：§1 白名单第二类「设备缓冲句柄：Metal/CUDA 设备侧缓冲、命令队列、事件」，只覆盖设备侧；M3-4 是 `Collective` 通信后端（NCCL communicator / MPI comm / Metal 对等组，见 `backlog-status` §7 M3-4），其句柄是**进程级库句柄**，不是设备缓冲/队列/事件，无法归入现有四类。

**建议修订**（§1 白名单增第 5 类）：

> **通信/进程级库句柄**：NCCL communicator、MPI comm、Metal 对等组等跨调用存活的进程级库句柄，与设备缓冲句柄同等对待——藏在边界 crate 内部类型里，经 `Collective` trait 向上只暴露不透明标识；M3-4 动工前若出现新句柄形态，回本 ADR 扩白名单。

### G3 — FFI 绑定选型被搁置，但零依赖已裁定，须补一条「FFI 前置默认」

**证据**：

- §不做：「不定义后端内部模块组织与 FFI 绑定选型（系统绑定 vs 生成绑定）」。
- `backlog-status` §7 M3 前置 #2（承 D5 零依赖原则）：「不引 cuda-sys / metal-rs 之类 FFI crate，要么手写 FFI 绑定，要么走 ADR 评审放宽策略」。

二者矛盾：前置说"必须二选一"，ADR 却"暂不定义"。写第一行 `extern` 前无据可依。

**建议修订**（§不做 改为 §决策新增一条，或降级为"待决-有门"）：

> **FFI 绑定默认策略（M3-1 前置）**：在独立 ADR 批准引入 FFI 绑定依赖（如 metal-rs / cuda-sys）之前，边界 crate 一律**手写 `extern "C"` 声明**，不引入任何 FFI crate；Metal 走 objc/Metal 运行时符号，CUDA 走 Driver API（`cuInit`/`cuMemAlloc`/`cuLaunchKernel` 等）手写绑定。若某后端证明手写绑定成本不可接受，需另立 ADR 申请 D5 例外，通过前维持零依赖。

## 3. 状态建议

- 上表 G1/G2/G3 三项并入 ADR-0035 后，规则已覆盖 M3-1（设备内核 + 执行）与 M3-4（通信）两类的全部越界对象，可作为 M3-1 的前置约束。
- **是否把 0035 置 `Accepted` 由你拍板**；本评审不自行翻转状态，也不据此动任何 `src` 代码（与 ADR-0035 §不做、plan §7 前置一致）。

## 4. 最小后续动作

1. 你确认 G1–G3 修订文案 → 回填 `0035-unsafe-ffi-boundary.md` 并把状态改为 `Accepted`。
2. `Accepted` 之后，M3-1 才具备动工前置；动工第一刀应先落"边界 crate 骨架 + 依赖方向门禁 + 手写 FFI 首条符号"，而非直接铺 Metal/CUDA 内核。

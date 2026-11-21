# 0052：六平台执行合同——六行闭集 + 三态裁决，把"哪台机器能跑"变成可复核的结构

- 状态：已采纳（本轮落地 `device::platform`）
- 日期：2026-11-20
- 相关：[0025](0025-runtime-backend-abstraction.md)（运行时后端三件套）、[0035](0035-unsafe-ffi-boundary.md)（设备后端的越界边界）、[0051](0051-execution-backend-base.md)（执行基座）、ADR 0002（单一主人）

## 背景

`lld.md` 的六平台执行合同（Metal / CUDA / ROCm-HIP / Vulkan / SYCL / CPU）此前只有两种落法，都不成为合同：

| 现状 | 它回答的问题 | 缺的是什么 |
|---|---|---|
| `cutile::profile::Profile` 设备档案闭集 | "这份档案声明了什么能力" | 声明不等于可执行；档案与这台机器能不能跑是两件事 |
| `device::backends::{probe_metal, probe_cuda}` | "这台机器上探测到了什么" | 探测是 `bool`，说不清"设备在但编译器不在" |
| `cutile::doctor` | "这份产物这条流水线过没过" | 逐档案、逐阶段，不是"平台 × 角色"的裁决 |

于是"六平台支持到什么程度"只能靠翻代码或读发布说明。目标形态要的是**一张六行的表**：每行一个平台，每格一个三态结论（覆盖 / 未覆盖 + 理由），且结论只能由实测证据给出，不能由"我打算支持"给出。

## 决策

### 1. 六行是闭集，GPU 行吸收 Vulkan / SYCL

`Row` 的六个变体即六平台，顺序就是合同的登记顺序（`Row::ALL`）。Vulkan 与 SYCL 不是一个独立的执行族，而是**同一个 GPU 行的两条提交路径**：它们和 Metal / CUDA / HIP 一样，平台侧给厂商 MMA、ABI 同构，差异在编译器与驱动。因此：

- `Row::Gpu` 的 `primitive()` 是"厂商 MMA + 同构 ABI"，`Arch::Gpu` 的标签同时接受 `vulkan` 与 `sycl`；
- 新增一个 GPU 族后端**不改行数**，只改该行 `toolchain` 的证据；
- 反过来，"要不要为 SYCL 单开一行"这个问题被结构性地关掉了——行长由执行族决定，不由实现个数决定。

### 2. 设备行要两件证据，CPU 行只要一件

```rust
fn present(row, host) -> bool   // 设备在不在（GPU 行要求设备 + 编译器）
fn identity(row, host) -> Option<String>  // 身份段：编译器 id() 或架构标签
```

`Covered` 必须同时满足两件，缺哪件就落成哪一种 `Why`：

| 情形 | 裁决 |
|---|---|
| 设备与编译器都没有 | `Open { NoDevice }` |
| 设备在、编译器不在 | `Open { NoToolchain }` |
| 设备不在、编译器在 | `Open { NoDevice }` |

`Why` 是三值闭集而不是自由文本：`NoDevice` 与 `NoToolchain` 的处置完全不同（前者是换机器，后者是装工具链），混成一个"未覆盖"就丢掉了唯一的行动信息。

### 3. GPU 行由一等原语自证，不重复写原语

GPU 行的一等原语就是"厂商 MMA + 同构 ABI"——设备证据成立时它自己就是证据，不再单独要求编译器版本进身份串。于是 `identity` 在 GPU 行可以返回空身份段，由 `compose` 决定"不把原语写第二遍"。这条不是排版洁癖：证据串是可复核的对外口径，同一个事实在一格里出现两次，读的人无法判断是两件证据还是一件。

### 4. `audit` 永不失败，`require` 才报告

```rust
pub fn audit(host: &impl Host) -> Vec<Machine>   // 六行全出，不因为"这台机器没有"而少一行
pub fn require(row, &impl Host) -> Result<&'static str>  // 缺证据 → E-PLAT
```

- `audit` 的契约是"每次都六行"。少一行会让"未覆盖"与"根本没问"变成同一种输出，而那正是三态要区分的东西。
- `require` 返回该行的一等原语标签（成功时），失败时给出 `Code::Invalid` 与"哪一行、缺什么"。它**只判平台是否有资格跑**，不判某个内核在某台设备上是否跑出正确结果——后者是 `cutile::verify` 与 `tools::card` 的事。

### 5. `Host` 是注入缝，不是探测实现

`Host` 只有两个方法（`arch: Arch` + `toolchain(row)`），`Machine` 是**这台机器**的默认实现：架构由 `std::env::consts` 给出，工具链复用 `backends` 里那份进程级缓存探测，不新开第二次 `--version` 调用。这样合同在测试里可以用一台假机器逐行摆出九种组合（有设备 / 有编译器 / 都没有），而不必真的换机器。

## 备选方案

| 方案 | 为什么不做 |
|---|---|
| 给每个厂商单开一行（含 Vulkan / SYCL） | 行长随实现个数增长，而 `lld` 的合同是六行；一个执行族多一条路径不是新平台 |
| 用 `bool`/`Option<&str>` 表示未覆盖 | 丢掉了 `NoDevice` 与 `NoToolchain` 的区分，而这恰好是唯一能指导动作的信息 |
| 让 `audit` 在缺证据时返回 `Result` | 那会把"这台机器没有这台设备"变成错误；没有 CUDA 的机器不是错误机器 |
| 在 `platform` 里重新探测工具链 | 同一进程两次 `--version`，两份缓存可以互相矛盾；`backends` 已有进程级快照 |
| 把六平台结论直接写进 `docs/` 表格 | 表格会与代码漂移；结论必须由 `audit` 从实测证据现算，文档只写口径 |

## 后果

- "六平台支持到什么程度"变成一次 `audit` 调用：每行一个三态 + 一条证据串，可打印、可断言、可进报告。
- 新增 GPU 族后端不改行数、不改裁决逻辑，只改该行 `toolchain` 的证据来源。
- `require` 让"在不支持的平台上跑"提前失败在入口，而不是在生成源或加载驱动时以厂商错误码的形式出现。
- 代价：`Row` 把"执行族"写死了六条。若将来出现既非 CPU 也非厂商 MMA 的执行族（例如纯数据流加速器），需要改这张闭集——这是有意的：闭集变化应当是一次 ADR，而不是一次 `push`。

## 不做

- 不新增设备后端代码：Metal 与 CUDA 的执行路径与其越界边界仍按 [ADR 0035](0035-unsafe-ffi-boundary.md) 处理，本 ADR 只裁决"能不能跑"。
- 不裁决性能与正确性：`require` 通过只说明平台有资格，E1–E15 的实测仍归协议卡。
- 不把 `Row` 并进 `cutile::profile::Device`：档案是**声明**，`Row` 是**执行族**，两者的主人与生命周期都不同（ADR 0002）。

## 修订

- 2026-11-20：初稿即采纳。落地形态为 `device::platform`（`Row` / `Arch` / `Abi` / `Host` / `Machine` / `Standing` / `Why` / `audit` / `require`），11 条用例覆盖六行顺序、GPU 行吸收 vulkan/sycl、设备行两件证据、CPU 行单件证据、GPU 行自证不重复、假机器九种组合、`require` 报 E-PLAT 而非静默通过。

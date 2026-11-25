# 0028：设备数据与机制的层级边界

- 状态：接受（P0 分析裁决；真实后端实现随 P1 CPU、P3 Metal/CUDA 填入）
- 日期：2026-11-16
- 相关：[0025](0025-runtime-backend-abstraction.md)（后端三件套、算子多后端挂载、`unsafe`/FFI 边界）、[0027](0027-kernel-registry-and-fragment-contract.md)（注册表键与挂载点、实现挂后端侧）、[0001](0001-layered-dependency-order.md)（依赖方向）；取证见 [`../cutile-analysis.md`](../cutile-analysis.md)；路线见 [`../engine-benchmark-roadmap.md`](../engine-benchmark-roadmap.md) §5、§6

## 背景

[`../cutile-analysis.md`](../cutile-analysis.md) 对 `crates/cutile` 的 25 个模块（7077 行）逐块取证后，把模块判成四类：唯一机制 12、命令策略 5、设备编排 5、纯数据 3。三个可复核的事实构成本 ADR 的依据：

1. **`cutile` 的四条生产依赖全部向下**（`crates/cutile/Cargo.toml:15-18` → `diagnostic` / `prim` / `op` / `prof`），零条指向 `engine` / `tensor` / `storage` / `serve` 等运行时成员，零条 FFI crate；crate 整体 `#![deny(unsafe_code)]`（`src/lib.rs:72`）。
2. **设备差异以"表行 + 档案字段"存在，不以代码分支存在**：`device.rs:44-63` 的 `SPECS` 三行分别对应 C / Metal / CUDA；九个与设备无关的机制文件里厂商名零命中（分析文档 §2.4）。
3. **CPU 的性能全部在 L2 数据里**：`crates/prof/devices/cpu.toml:42-194` 的 153 行前导承载六个 `static inline` helper；`cutile` 源码对 `_cutile_*` / NEON 内建零命中（分析文档 §3.3）。反过来，片段层也允许承载算术（`prim/cpu/dot.c` 27 行有效代码、`quant.c` 6 行），"片段是壳"不是普适判决。

此前一轮曾把 Metal 后端整体塞进 `crates/cutile`；本 ADR 把"数据在 L2、机制在 L3、运行时后端不在 L3"写成可判定的规则，避免下一条加速链路再次越界。

## 决策

### 1. 设备数据归 L2：`prim` / `prof` 是新增加速链路的主战场

下列对象是**数据**，事实源在 L2，新增一条加速链路的改动面落在这两个成员：

- 片段正文：`crates/prim/<vendor>/<prim>.<ext>`（含允许出现的片段内算术）；
- 见证源码：`crates/prim/<vendor>/witness.*`；
- 档案声明：`crates/prof/devices/<device>.toml` 的 `provides`（`prim` / `impl` / `dtype` / `space` / `lanes` / `grade` / `witness`）、`[compile].flags` 与 `preamble`。

判据：改一份档案或片段而 `cutile` 一行不改，编译结果的对错由这份数据决定——这一格就归 L2。片段正文与前导都可以放算术，分工以"实现挂在片段还是前导"为准，不按语言归类（分析文档 §3.2、§5.2）。

### 2. 机制留在 L3 `cutile`：允许"加一行"，不允许新增厂商模块

下列对象是**机制**，留在 `cutile`：装载（`builtin`）、协商（`cap`）、替换（`substitute` / `ast`）、成文（`emit`）、指纹与产物缓存（`cache`）、选 tile（`plan`）、按 `Emit` 走一条 `Spec` 的编排（`device` / `harness` / `toolchain` / `gate` / `runner`）。

- **允许**：在 `device.rs` 的 `SPECS` 表里加一行（新 `Emit` 对应一段既有形态的私有编排）；在既有编排段内补一个私有函数。
- **禁止**：在 `crates/cutile/src/` 新增 `metal.rs` / `cuda.rs` / `ascend.rs` 之类按厂商分岔的模块；在热路径写 `if Metal {} else if CUDA {}` 的厂商特判（沿用 ADR 0025 §4）。
- 既有例外形态：`device.rs` 的三段私有编排、`harness.rs` 的两方言、`toolchain.rs` 的三条探测函数，是已批准的"表行 + 私有编排"；新增设备只能照这个形态加表行，不能长出同构的第四套厂商模块。

### 3. 真设备运行时不进 `cutile`，本轮也不新建 crate

下列对象是**运行时后端**，不进 `cutile`：跨调用存活的设备缓冲、命令队列、kernel launch、设备事件同步、Rust 侧 FFI / `unsafe`。

- `cutile` 现有 Metal / CUDA 段只服务**见证集合**：把 `witness.*` 源码经子进程编译、执行、按 stdout/stderr 文本协议回读（分析文档 §5.1）；生成驱动里出现的 `id<MTLDevice>` 是编译出来的 Objective-C 驱动源码（`device.rs:646`），不是 `cutile` Rust 侧持有句柄。这条见证链不算运行时后端。
- 运行时后端的落点延后到触发条件满足时再裁决，候选：`storage` 的 feature 岛（`Buffer` / `Allocator`，ADR 0025 §1）、`engine` 的 `Executor` 实现（ADR 0025 §1）、P3 的独立后端边界 crate。
- **触发条件**（三者任一出现，即说明机制越出了"子进程 + 文本协议"形态，必须搬出 `cutile`）：需要持有跨调用存活的设备句柄或缓冲；需要命令队列 / 事件同步；需要 FFI 或 `unsafe`。

### 4. 对 roadmap §5/P3"独立后端边界 crate"的偏离记录

[`../engine-benchmark-roadmap.md`](../engine-benchmark-roadmap.md) §5「后端 trait 边界」预留点与 §6 P3 前置约束规定：P3 接入 Metal / CUDA 时新增**独立后端边界 crate**，把 `unsafe` 与两套 FFI 关在边界内。

- **本轮偏离**：不新建该成员，也不新建任何 crate。
- **理由**：P0/P1 阶段没有真实后端实现可挂，`cutile` 内现有见证链是子进程形态，不持有运行时设备状态；此刻立边界 crate 只会多一层仪式（与 ADR 0027「注册表放进新独立 crate」备选方案的拒绝理由同构）。
- **复核判据**：ADR 0025 §1 归属裁决（`Buffer` / `Allocator` 归 `storage`，`Device` / `Executor` 归 `engine`）、ADR 0001 依赖方向、`docs/ARCHITECTURE.md` 层表。P3 动工前必须按 roadmap P3 前置约束补一条边界规则 ADR（ADR 0025 §9 已预告），届时复核本裁决是否仍成立；触发条件（决策三）出现而边界 crate 尚未落地时，本条偏离自动失效。

## 正反例对照

| # | 改动 | 判定 | 依据 |
|---|---|---|---|
| 正例 1 | 改 `../../crates/prof/src/metal/metal.toml` 的 `preamble` / `flags` / `provides` | 合法：数据缺口 B，落在 L2 档案层 | 决策一；分析文档 §5.2 落点一 |
| 正例 2 | 新增或改写 `crates/prim/metal/reduce.metal` 的片段正文 | 合法：数据缺口 A，落在 L2 片段层 | 决策一；分析文档 §5.2 落点二 |
| 正例 3 | 在 `device.rs:44-63` 的 `SPECS` 加一行，复用既有编译 / 驱动 / 回读三段编排 | 合法：机制扩展，形态仍是表行 + 私有段 | 决策二；分析文档 §5.3 |
| 反例 1 | 在 `crates/cutile/src/` 新增 `metal.rs`（或 `cuda.rs` / `ascend.rs`）厂商模块 | 越界：设备编排只能以 `SPECS` 表行 + 私有段存在 | 决策二 |
| 反例 2 | 在 `cutile` 内持有设备缓冲 / 命令队列 / 持久设备句柄，或引入 Rust 侧 FFI / `unsafe` | 越界：运行时后端归 `storage` / `engine` / 边界 crate | 决策三；ADR 0025 §9 |

两条正例与两条反例均可唯一判定：改动对象在 `prim` / `prof` 即合法；改动落在 `cutile` 的新增厂商模块或运行时句柄上即越界。

## 后果

### 正面

- 下一条加速链路（Metal 先行）的工作量被明确拆成 L2 数据改动（片段正文、前导 helper、档案等级）与 L3 机制改动（表行 / 编排段）两段，可分别验收；数据侧有 `wc` / `grep` / 签名对齐的零风险判据。
- 运行时后端的候选落点与触发条件写在明面上，P3 接 Metal / CUDA 时不会在 `cutile` 里长出私有设备假设。
- `cutile` 的 `#![deny(unsafe_code)]` 与"子进程 + 文本协议"形态保持不变。

### 负面

- 本 ADR 只定边界，不产出可运行的 Metal / CUDA 真内核；算子内核执行通道（`Lane::Metal`、Metal harness 方言）按分析文档 §5.5 排到 P2。
- 片段与前导的算术分工仍需逐设备判断，规则不给"算术必须放哪一层"的第二套闭集。

### 中性

- `arch` / `Emit` 标签是登记与分发键，不产生热路径厂商特判（ADR 0027 §1）。
- 见证执行链既有的设备段保留，不作为运行时后端计入本 ADR 的边界统计。

## 不做

- 本轮不实现 Metal / CUDA 后端，不引入 GPU 依赖、线程池或 `unsafe`（P1/P3）。
- 不补 L0 五条原语（`atomic` / `mma` / `scan` / `exp2` / `rcp`）与零槽位原语（`shuffle` / `broadcast`）的片段文本；它们随首条对应真实内核一起补（ADR 0027 §不做）。
- 不新建独立后端边界 crate；其落点与边界规则随 P3 动工前的 ADR 裁决。

## 备选方案

- **把运行时后端留在 `cutile`，按设备各开一个模块**：拒绝。`cutile` 出边全部向下且 `deny(unsafe_code)`，持有设备句柄会反向依赖运行时层并破坏依赖方向（ADR 0001）。
- **本轮就新建独立后端边界 crate**：拒绝。没有后端实现可挂，新 crate 只是空仪式；延后到 P3 真实双后端并行时按 roadmap §5 落地。
- **片段 / 前导不区分算术归属，统一压进片段**：拒绝。CPU 样板是"片段只做一次 `$func` 调用、实现在前导"（`prim/cpu/reduce.c:4` ↔ `cpu.toml:53`）；统一位置会丢掉这条外包路径的契约表达。

## 修订

- 2026-11-16：初稿，定数据 / 机制 / 运行时三层归属、正反例判据、对 roadmap §5/P3 边界 crate 的偏离记录与复核条件；实现随 P1/P3。

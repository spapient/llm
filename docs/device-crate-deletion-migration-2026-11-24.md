# device crate 删除与设备内核下沉迁移方案（2026-11-24）

- 状态：**#29（Metal FFI Executor → engine）已落地，check/clippy/fmt/testkit 全绿**；#30/#31/#32 施工中（任务 #30/#31/#32）
- 触发：`crates/device` 不应再持有设备内核正文；整机删除，职责按层归位到对应设备专属代码
- 范围：`crates/device` 全部 `src`（7218 行）+ `tests`（3712 行）

## 1. 现状勘察（confirmed）

### 1.1 外部依赖面

产品路径上**只有 `cli`** 直接依赖 `llm-device`：

| 引用点 | 符号 | 归属批次 |
|---|---|---|
| `cli/src/lib.rs:59` | `device::backends::install()` | #30 |
| `cli/src/cutile.rs:34` | `device::backends::install()` | #30 |
| `cli/src/platform.rs:3` | `device::platform::{Machine, Row, Standing, audit}` | #30 |
| `cli/tests/run.rs:8` | `::device::backends::install` | #30 |
| `cli/tests/run.rs:94,106,114,128,136,187` | ~~`::device::executor::MetalExecutor`~~ → `::engine::metal::MetalExecutor` | #29 ✅ |
| `cli/tests/device.rs:17,31,45` | ~~`::device::executor::MetalExecutor`~~ → `::engine::metal::MetalExecutor` | #29 ✅ |

甄别排除（**不是** `llm-device` 依赖，是各自 crate 的内部同名模块，搬迁时不得改动）：

- `prof::device`（`crates/prof/src/device.rs`，档案/片段词汇）
- `shard::device`（`crates/shard/src/device.rs`，注册表 + host C 参考后端）
- `testkit::device`（`crates/testkit/src/device.rs`，对拍夹具）

### 1.2 内部模块分类（src 7218 行）

| 类 | 文件（行数） | 职责 | 目标 |
|---|---|---|---|
| C Metal FFI Executor | `executor.rs`(568) | 手写 objc/Metal FFI，实现 `engine::Executor` 缝；唯一生产 unsafe 岛 | **engine** |
| B 六平台合同 | `platform.rs`(580) | ADR 0052：Row/Standing/Why/Abi/Arch/Host/Machine/audit/require/toolchain 探测 | **shard** |
| E 方言 lowering 机制 | `dialect.rs`(360)、`dialect/body.rs`(115)、`dialect/geometry.rs`(133)、`dialect/text.rs`(100) | `Dialect` trait + 后端无关 lowering 原语、几何查表、文本装配、body 改写 | **shard** |
| A 后端装配/注册 | `backends/mod.rs`(50)、`backends/cuda.rs`(360)、`backends/metal.rs`(326)、`backends/ascend.rs`(205)、`backends/parallel.rs`(210)、`backends/common.rs`(87)、`backends/harness.rs`(199)、`metal/host.rs`(327)、`metal/program.rs`(334)、`metal/runtime.rs`(116)、`metal/stage.rs`(96)、`metal/witness.rs`(67)、`metal/device.rs`(73)、`cuda/shared.rs`(316)、`cuda/segments.rs`(145) | `Codegen`/`Driver`/`Backend` 实现、进程级注册、host 驱动装配、缓冲/回读/计时/witness、多段装配 | **shard** |
| D 并行内核正文 | `cuda/gemm.rs`(224, WMMA)、`cuda/grid.rs`(297, gemv_i4 warp 核)、`cuda/chunk.rs`(237)、`cuda/scan.rs`(131)、`metal/chunk.rs`(129)、`tiled/scalar.rs`(383)、`tiled/attention.rs`(336)、`tiled/tensor.rs`(271)、`tiled/geometry.rs`(77)、`tiled/transpose.rs`(49) | 设备端算法：warp 归约、shared 组织、mma/simdgroup 片段、在线 softmax、int4 解包、分块映射 | **prof/<vendor>/** |
| F 反向核契约 | `backward.rs`(202) | 逐点一元导数表达式 + 反向设备核源 | **nn/train**（批次 #32 定） |
| 壳 | `lib.rs`(47)、`cuda/mod.rs`(24)、`metal/mod.rs`(22)、`tiled.rs`(22) | 模块声明与重导出 | 删除 |

### 1.3 层级方向约束（ARCHITECTURE.md 一、分层）

- L5 `shard` 已含「原 optim 编译执行面」（backend/device/toolchain/bench/verify/execution），承接 A/E/B 方向合法（同层/向下）。
- `engine` 在 L6，`Executor` 缝定义在 `engine::queue`/`engine::exec`；FFI 实现回迁 engine 是同层归位，消除 `device → engine` 边。executor **不能**进 L5 shard（L5→L6 向上，违反依赖方向）。
- L2 `prof` 明文「契约/档案子域不得出现机制代码（拼接、协商、emit）、设备知识与厂商内核正文」。D 类下沉时只能搬**设备源片段/模板**；形状校验、签名/样本取值、发射几何、emit 装配必须留在 shard（A/E）。

## 2. 归属映射细则

### 2.1 C → engine（批次 #29）

- 落点：`crates/engine/src/metal.rs`，`#[cfg(target_os = "macos")]` 挂模块。
- use 映射：
  - `engine::exec::{Device, Graph, Node, Stream}` → `crate::exec::{...}`
  - `engine::exec::{Executor, Queued, check_operands, enqueue}` → `crate::queue::{...}`（`exec.rs` 已 re-export，可统一走 `crate::exec`）
- 岛：`#![allow(unsafe_code)]` 随迁；对外类型 `MetalDevice / MetalExecutor / MetalBuf / MetalPipeline / MetalCommandQueue` 由 engine 重导出。
- 连带测试：`tests/metal_executor_harness.rs`(579)、`tests/metal/mod.rs`(30) 及其下 `attention/backward/contract/correctness/gemm/tensor` 与后端生成链耦合，**不在 #29 迁**，随 #30/#31 对应链回迁；#29 先回迁生产代码并摘掉 executor.rs 末尾的 test mod 挂载。
- **#29 落地记录（已闭环）**：生产代码落 `crates/engine/src/metal.rs`，engine lib 以 `#[cfg(target_os="macos")] pub mod metal;` 挂出；device `platform.rs:118` 的 Metal 探测改指 `engine::metal::MetalDevice`，`device/src/executor.rs` 已删、lib 模块声明已撤；cli 测试与 `device/tests/metal_suite.rs` 的 executor 类型路径改走 `engine::metal`；`testkit/src/boundary.rs` 的 `UNSAFE_ISLANDS` 岛路径改 `crates/engine/src/metal.rs`（计数 1）；ADR 0035 §6 岛表、`ARCHITECTURE.md` 所有权表 L139（归属 device→engine）与 L20/L64 死链同步；engine lib 头部注释口径更新。harness / `metal/*` 真机夹具仍挂 `device/tests/`，待 #30/#31 随链回迁。

### 2.2 A/E/B → shard（批次 #30）

- 落点：复用 shard 现有 `backend.rs`（Backend/Codegen/Driver/Source/Launch/Lower/TensorGeometry）、`device.rs`（注册表 + CBackend）、`toolchain`、`bench`、`verify`、`execution`；vendor 实现按 `shard/src/<vendor>/` 收子模块。
- use 映射：`llm_optim::*`（即 `llm-shard` 自身）→ `crate::*`；device 内部 `super::` 路径随模块重排。
- `platform.rs` 合同与 toolchain 探测随执行面进 shard；cli `platform.rs` 改引用 shard 路径。
- `install()` 注册静态后端的入口随迁 shard，cli 改调 shard 入口。

### 2.3 D → prof/<vendor>/（批次 #31）

- 语言归位：CUDA/MUSA → `prof/src/nvidia|musa/*.cu`；Metal → `prof/src/metal/*.metal`；Ascend → `prof/src/ascend/*.cpp`。
- 切分原则（每函数）：纯设备源片段/模板进 prof vendor 目录；取签名/样本/形状校验、整除守卫判定、发射几何、`format!` 装配留 shard（#30）。
- 通道待核：prof `build.rs`/`buildops.rs`/`buildtables.rs` 现按 `read_dir` 收集 `$0` 占位符片段并编译期展开成 FRAGMENTS/PRIMS/EPILOGUE/DEVICE 表；device 并行内核是 Rust 函数做形状特化的自包含源。落地前先定装载方式（静态源文件嵌入，或在 vendor 目录内建 Rust kernel 模块），并复核不违反 L2 禁项。

### 2.4 F → nn/train（批次 #32）

- `backward.rs` 逐点反向核契约：标量反向公式权威在 `train`，设备同式执行体随归属层定；先挂 nn 反向通道，与 `autograd` 图编排解耦。

## 3. 同步清单（批次 #32 收尾前逐项落地）

- `cli/Cargo.toml:46` 删除 `device` 依赖；cli src/tests 引用路径全部改新归属。
- `.config/lints/vocabulary.toml`：清 device 密度基线 `min_lines_baseline` 条目。
- `crates/testkit/src/boundary.rs`：
  - `VENDOR_OWNERS`（现 `["device"]`）按新归属调整；
  - `UNSAFE_ISLANDS`：`crates/device/src/executor.rs` 路径改为 engine 新岛路径，计数 1 不变；
  - `the_device_backends_are_installed_on_product_path` 断言的 `backends::install()` 路径随 #30 改。
- `docs/ARCHITECTURE.md`：L6 层表删 device 行、边表 device 行（`device → cutile,diagnostic,engine,prof,shard`）、所有权表 139/140/141 行；L5 shard 职责补执行面承接内容。
- ADR：0035 §6 岛表 executor 行路径；0052 平台合同归属；0054/0056/0053 内 device 提及；`engine/src/lib.rs:6,7,10` 注释；CHANGELOG。
- CI：members 是 glob `crates/*`，删目录即移出；复核 coverage/nextest/fuzz 名单无 device 残留。

## 4. 风险与未知

- **并发在途股**：shard（collective/tiling/pipeline/build.rs）、cli tests（device.rs/run.rs）、prof（build.rs/buildimage.rs/native）、nn（registry/kernel*）均为并发改动股；施工前 `git status --short` 划边界，改动集中完成后再一次性跑门禁。
- **真机验证受限**：本机 macOS，仅 Metal 链路可编译运行；CUDA/Ascend/MUSA 无真机，回迁后只能保证编译与静态对拍，内核数值正确性留真机回归。
- **测试耦合**：device tests 3712 行与后端生成链深度交织，必须随对应生产链同批回迁，不能先删生产代码留测试。
- **prof L2 张力**：并行内核含线程映射/发射几何，与 L2「不得持机制代码」禁项相邻；#31 切分时严格分离正文与装配，必要时回写 ADR/架构口径。
- **基线偶发断裂**：并发周期打包中间态曾致 prof buildops moved value、nn unary 宏分组、registry/catalogue 顺序不一致；开工前先 `cargo check --workspace` 确认基线绿，陈旧产物先 `cargo clean -p <crate>`。

## 5. 验证口径（每批次收尾）

`cargo check --workspace --all-targets --all-features`、`cargo clippy --workspace --all-targets --all-features -- -D warnings`、`cargo test --workspace --exclude llm-fuzz`、`cargo +nightly fmt --all --check`；device 未删完前保持每批编译绿，不留下半截断裂态。

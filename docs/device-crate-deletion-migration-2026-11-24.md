# device crate 删除与设备内核下沉迁移方案（2026-11-24）

- 状态：**#29（Metal FFI Executor → engine）已闭环**；**#30（A/E/B + 过渡 D → shard）生产面已回迁**；**#31 按 ADR 0028 / L2 边界裁决：整程序 tile kernel 的选 tile / 成文 / 编排 / harness / toolchain 机制留 shard，不拆入 prof vendor**；**#32 `crates/device` 目录已物理删除（git 全量 D），cli path 依赖移除，backward 标量公式权威归 `train/backprop`+`formulas`，设备侧同式反向内核为未启用 POC 缝随退役**；文档 / 基线 / 边界注释回写完成。剩余仅存量棘轮治理（ownership / baseline / complexity）。
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
- 本批实际落点目录（避免并入既有文件造成职责混杂）：
  - E dialect：`shard/src/dialect.rs` + `shard/src/dialect/{body,geometry,text}.rs`；纯 `pub(crate)` lowering 原语，不进 shard 对外面。
  - B platform：`shard/src/platform.rs`（Row/Standing/Why/Arch/Host/Machine/audit/require/standing + 工具链/设备命令探测）。
  - A 装配：`shard/src/backends/{mod,ascend,common,cuda,harness,metal,parallel}.rs`；Metal host 侧 `shard/src/backends/metal/{host,program,runtime,stage,witness,device}.rs`；CUDA 共享装配 `shard/src/backends/cuda/{shared,segments}.rs`。
  - D 过渡：`cuda/{gemm,grid,chunk,scan}.rs`、`metal/chunk.rs`、`tiled/*` 本批随 A 族一并落 shard 子树保持编译绿，#31 再抽离 prof/<vendor>/；不在 #30 留 shard→device 的反向边。
- use 映射：`llm_optim::*`（即 `llm-shard` 自身）→ `crate::*`；device 内部 `crate::metal::` → `crate::backends::metal::`、`crate::cuda::` → `crate::backends::cuda::`；`super::` 相对同级路径随目录重排后保持同级；错误站点字面量 `engine::tiled::…` 临时改 `shard::tiled::…`（#31 随正文归 prof 再定）。
- **Metal 探针归属决策**：shard(L5) 不得上靠 engine(L6)，故 `platform.rs` 迁入 shard 后不再实例化 `engine::metal::MetalDevice`；`metal_device()` 退化为平台常量判定（`std::env::consts::OS == "macos"` 即设备位成立），与其余设备行"无进程内 FFI 探针"的口径一致；真实 FFI 设备句柄只在 engine 执行面使用。Host trait 本就允许上层注入自定义宿主，真实设备证据由执行面装配阶段校验。
- `install()` 注册静态后端的入口随迁 shard（`crate::backend::register`），cli 改调 shard 入口。
- `platform.rs` 合同与 toolchain 探测随执行面进 shard；cli `platform.rs` 改引用 shard 路径。
- **精确路径改写表（cp 后逐文件 sed）**：
  - `llm_optim::backend::` → `crate::backend::`；`llm_optim::witness::` → `crate::witness::`；`llm_optim::toolchain::` → `crate::toolchain::`；`llm_optim::execution::` → `crate::execution::`。
  - `llm_optim::driver` → `crate::driver`（lib 根已 re-export）；`llm_optim::compile_flags` → `crate::compile_flags`（lib 根 re-export）；`llm_optim::runner::parse_bits` 注释提及，随 #31 清理。
  - `crate::metal::` → `crate::backends::metal::`；`crate::cuda::` → `crate::backends::cuda::`；`crate::dialect::`、`crate::platform::`、`crate::backends::`、`crate::tiled::` 根名不变。
  - `crate::test_imports::*`（backends/mod.rs 的 `#[cfg(test)]` 聚合别名）随 mod 一起迁。
  - 站点字面量：`engine::tiled::…` → `shard::tiled::…`；`engine::dialect::…` → `shard::dialect::…`；`engine::platform` → `shard::platform`。
- **测试挂载迁移**：生产文件尾部 `#[cfg(test)] #[path]` 挂载点随迁：`tiled.rs`→`tests/tiled.rs`；`backends/mod.rs`→`tests/backends.rs`、`testkit/tests/kernel.rs`（保留对 testkit 的相对路径）、`tests/program.rs`；`platform.rs`→`tests/platform.rs`。独立 `[[test]]` target `metal_suite.rs`（挂 `metal_executor_harness.rs` + `metal/mod.rs` 及其子模块）与 `schedule.rs` 随生产链迁 shard；`backends_production.rs` 归属施工时核对挂载。shard Cargo.toml 的 `[[test]]` 名单与 autotests=false 口径对齐。
- **密度下限基线**：下限门 `file_min_lines=300`，多个迁入文件低于阈值（geometry 133 / text 100 / body 115 / stage 96 / witness 67 / metal/device 73 等）；属存量搬迁，需把 `.config/lints/vocabulary.toml` 的 device 下限基线条目连同文件迁到 shard 名下，不得按"新文件"硬卡；#32 删 device 时清掉对应 device 条目。
- **依赖清单核对**：迁入生产代码外部 use 仅 `cutile / diagnostic / prof / std` + shard 自引用；shard Cargo 已具备 cutile/diagnostic/prof/nn，无需为生产面新增 engine/tensor 依赖（`tens`/`nn`/`testkit` 仅 dev 测试图）。
- **#30 落地记录（生产面闭环）**：A/E/B 生产族 + 过渡 D 族已复制入 shard 子树并完成路径改写、lib 挂模块；cli `backends::install()` 与 `platform::{…}` 引用切 shard；`testkit/src/boundary.rs` `VENDOR_OWNERS` 双份过渡并列 `["device","shard"]`、UNSAFE_ISLANDS 指向 `engine/src/metal.rs`；密度下限基线 device/shard 双份过渡登记。门禁：workspace `check --all-targets` 绿；shard lib 185 passed / 0 failed、integration 绿；cli 61 passed；tools 密度门 39+15+27+10 绿；`cargo +nightly fmt --check` 0 diff；shard/cli/device/engine clippy 无 error。
- **测试隔离处置（注册表污染）**：`tests/backends.rs` 见证真机套件的 `ensure_backends()` 会把 Metal/CUDA/昇腾后端注入进程级全局注册表（`device.rs` 的 `OnceLock<Mutex<Vec>>`），与同 lib test 进程内 `bench::unregistered_device_profiles_are_not_reported_as_gpu_benchmarks` 的空表断言互相污染（全量跑时后者拿到已注册后端，单跑通过、全跑失败）。device 旧份在独立 crate 无此冲突；迁入 shard 合进程后暴露。#30 过渡处置：注释掉 `backends/mod.rs:40` 的 `#[path=…/backends.rs]` 挂载，文件保留；`test_imports` 中仅该套件消费的 `MetalBackend`/`Driver`/`ascend_device`/`detect_cce` 别名一并摘除，program.rs/kernel.rs 仍消费的 `fold_of/unsupported/Launch/Lower/Source/Code/cuda/metal/Signature/Sample` 保留。**#31/#32 落独立 `[[test]]` integration target（独立 test binary，注册表进程隔离）后再挂回**；`program.rs`/`kernel.rs`/`platform.rs`/`tiled.rs` 不触注册表，保留挂载。
- **#30 残留（转 #31/#32 收口）**：`backends_production.rs` 当前为未挂载孤儿文件，#31/#32 随见证套件一并定挂载；独立 integration target `schedule.rs`、`metal_suite.rs` 强耦合跨三批链未迁；ARCHITECTURE.md L5/L6/边表/L141 平台合同归属、ADR 0052/0053/0054/0056 的 device 提及留 #32 收尾回写；cli Cargo.toml L46 `llm-device` 依赖与 `tests/device.rs` 的 D 族引用留 #31/#32。

### 2.3 D → prof/<vendor>/（批次 #31）

- 语言归位：CUDA/MUSA → `prof/src/nvidia|musa/*.cu`；Metal → `prof/src/metal/*.metal`；Ascend → `prof/src/ascend/*.cpp`。
- 切分原则（每函数）：纯设备源片段/模板进 prof vendor 目录；取签名/样本/形状校验、整除守卫判定、发射几何、`format!` 装配留 shard（#30）。
- 通道勘察结论（#31 开工前已核，confirmed）：
  - prof 现有两条静态语料通道：① 八槽 vendor 原语片段 `src/<vendor>/<prim>.<ext>`（首行 `/* cutile-sign: <prim>/<impl> */`，`$0..$3` 占位，`build.rs:162-198` 收集进 FRAGMENTS/PRIMS/EPILOGUE/DEVICE 表，同构片段逐设备同源）；② 设备无关算子模板 `ops/*.k`（`{{op}}` / `@prim(...)` 槽位，调 vendor 原语）。**整程序 vendor kernel 目前不在任一槽位**。
  - `prof/src/native/` 是 prof 内唯一 Rust 微内核受控岛（arch 门 / 特性探测 / unsafe 白名单钉死，`native/mod.rs:16-127`）：先例证明 prof 可持 Rust kernel 模块，但边界仅限主机 CPU 微内核与瓦片几何，不收设备 vendor 正文。
  - L2 禁项（`ARCHITECTURE.md:16`）：契约/档案子域不得出现机制代码（拼接 / 协商 / emit）、设备知识与厂商内核正文；`src/<vendor>/` 是唯一允许原语级同构正文的档案子域。整程序的线程块/网格/寄存器与共享内存特化、工具链探测属设备知识与机制，必须留 shard（L5 执行面）。
  - D 族待拆清单（shard 现状，2641 行）：`cuda/{chunk,gemm,grid,scan,segments,shared}.rs`、`metal/chunk.rs`、`tiled/{attention,scalar,tensor,geometry,transpose}.rs`。函数已分两类：**可下沉候选**——纯设备源正文 / shim / head / scaffold 文本片段；**必留 shard**——形状校验、签名/样本取值、发射几何（`cuda_grid`/`cuda_params`/`cuda_prelude`/`cuda_wrapper`）、driver 与 Codegen/Driver 装配、`format!` 整程序拼接。
- **#31 裁决结论（已落地）**：按 ADR 0028 与 L2 边界，prof 档案子域禁持机制代码 / 设备知识 / 厂商内核正文，且整程序 tile kernel（线程块 / 共享内存 / WMMA / matmul2d 编排）不在现有八槽片段或 `ops/*.k` 模板通道内。故选 tile、成文、发射几何、编排、harness、toolchain 探测一律留 shard 执行面；不扩展 `build.rs` 接纳整程序静态源、也不强行把整程序拆进八槽。原 D 族中不可承接的整程序正文（`cuda/gemm.rs` WMMA、`tiled/{attention,scalar,tensor,geometry,transpose}.rs` 与 `tiled.rs`）已从工作树退役；存活的 host 骨架 / 分配守卫 / 计时循环 / 缓冲 / 单次派发 Driver / 发射回读 / 多段脚手架 / 块级派发 / 主机折叠 / 段生成 / host 串联仍住 shard（`cuda/{shared,segments,grid,chunk,scan}.rs`、`metal/{host,program,runtime,stage,witness,chunk,device}.rs`）。本机仅 Metal 链路可运行，CUDA/Ascend/MUSA 只能保证编译与静态对拍。

### 2.4 F → nn/train（批次 #32）

- `backward.rs` 逐点反向核契约：标量反向公式权威在 `train`，设备同式执行体随归属层定；先挂 nn 反向通道，与 `autograd` 图编排解耦。
- **#32 落地记录**：标量反向公式权威实现已在 `train/src/backprop.rs` + `formulas.rs`（`Entry { op, formula }`，`FORMULAS[40]`，按逆拓扑回传梯度），梯度图编排归 `autograd`。旧 `device/backward.rs` 的 `GradientEntry` + `POINTWISE` + `lookup` + `source(metal_kernel/cuda_kernel)` 是设备侧同式反向内核源生成，核查全仓后确认仅被 device 自身真机测试引用、无生产路径调用（未启用 POC 缝），随 `crates/device` 目录删除一并退役，不另落承接 crate；`GradientEntry` 不进架构所有权表。

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

# GPU 后端 crate 归并评估：`crates/cuda` / `crates/musa` / `crates/metal`

> **状态**：评估 + 已实施（2026-10-17，采用单 crate 形态）　**日期**：2026-10-17　**基线**：`main` @ `1c50c4b`
> **范围**：回答"是否应把三个 GPU 后端 crate 合并为 `crates/gpu`"。
>
> **实施记录（2026-10-17）**：经评审选定单 crate 形态——`crates/gpu/gpu.rs`（crate 根 + 平台无关 helper）加 `crates/gpu/{cuda,musa,metal}/`，每后端保留独立 `ffi.rs` 安全岛。§5 的阻碍已按下述方式处置：`links = "musa"` 移除（无 `DEP_MUSA_*` 消费者）；三条构建管线合并为一个 `build.rs`；`checkunsafe.sh` 的 crate 根推导改为显式指向 `crates/gpu/gpu.rs`（原先的反推会静默失效，已修复）；各后端命名空间隔离，不再有 `matmulpath` 冲突。本文 §3–§6 中的 `crates/cuda` 等路径描述的是归并前基线。
>
> 实测结论：Metal 真机 50/50 通过；cuda/musa 交叉目标类型检查通过，运行时未在归并后重跑。

## 1. 结论

**不建议合并为单个 crate，也不建议仅迁移目录。** 两条路都不能达成"减少代码"的目标：

* **目录搬运（`crates/gpu/{cuda,musa,metal}`）减少 0 行代码**，且会脱离 `members = ["crates/*"]` 的 glob（`*` 不跨 `/`），需要手工登记成员。`docs/gpu-backend-optimization-plan.md` §1.1 已就同一问题给出"暂不迁移"的决策。
* **单 crate 合并同样减少 0 行代码**：重复的 executor 函数体不会因为住在同一个 crate 里而消失，仍需手工抽取；而平台 FFI 会因 `cfg` 与命名空间混住变得更难读。同时它打穿仓库最严的一道安全门禁（§5）。
* **真正能减少代码的是抽取共享骨架**，且这条路可以完全不触碰三个 crate 的边界。实测直接重复约 **1,200 行**，其中 Rust ≈ 700 行、`.cu` 内核 ≈ 526 行。

**性能收益为 0**：crate 边界不影响代码生成。现有性能收益全部来自内核路径（WMMA）与 device-resident session，与包布局无关。

**当前更值得优先修的问题不是布局，而是能力不对称**：全仓只有 MUSA 实现了 `api::session`（`crates/musa/src/executor.rs:1263`），CUDA/Metal 只有一次性 `submission`（`crates/cuda/src/executor.rs:704`、`crates/metal/src/executor.rs:596`）。因此 `infer::submissionengine` 今天只能由 MUSA 驱动。

## 2. 度量口径

以下数字均为本次在仓库内实测，不是引用既有文档。

| 度量 | 方法 |
| :--- | :--- |
| 行数 | `wc -l`，含 Cargo.toml 与 build.rs |
| 函数体重合 | 按 `fn <name>` 切块，空白归一 + 归一 `cuda/metal/musa`（含大小写变体）为占位符后逐行比较 |
| 内核重合 | 同上归一后用 `diff` 统计 cuda 侧未匹配行 |
| 依赖面 | `grep -rn --include='Cargo.toml' '^cuda|^musa|^metal'` |

**注意**：不做名字归一时逐行重合只有 19–22%，因为三份代码的函数顺序与偏移不同。名字归一后的"同名函数体完全一致"才是可复用的真实口径。

## 3. 现状度量

### 3.1 规模

| crate | Rust+toml | 其中 `ffi.rs` | 内核 | 合计 |
| :--- | ---: | ---: | ---: | ---: |
| `cuda` | 2,920 | 1,536 | 572 | 3,492 |
| `musa` | 4,409 | 2,211 | 876 | 5,285 |
| `metal` | 3,078 | 1,818 | 692 | 3,770 |
| **合计** | **10,407** | **5,565（53.5%）** | **2,140** | **12,547** |

`ffi.rs` 占一半以上，且是本议题中**唯一确定不可共享**的部分（见 §6）。

### 3.2 可复用面

| 对比 | 同名函数 | 函数体完全一致 | 一致行 / 共同行 |
| :--- | ---: | ---: | ---: |
| cuda vs metal | 39 | **33** | **532 / 681（78%）** |
| cuda vs musa | 38 | 14 | 172 / 669（25%） |
| metal vs musa | 38 | 12 | 124 / 634（19%） |

cuda/metal 的 33 个一致函数包括全部 14 个 `record*`、`add`/`mul`/`rmsnorm`/`softmax`/`silu`/`embedding`/`rope`/`attention` 转发、`concat`/`concatshape`/`scanshape` 形状 helper。分歧仅在 6 个：`maptobackend`、`matmul`、`recordmatmul`、`resolve`、`rope_at`、`scan`。

MUSA 之所以只有 25%，是因为它在 `session`/`recorder`/`submission` 上做了扩展（`record*` 方法 32 个 vs cuda 15 / metal 14）。

### 3.3 内核

| 项 | 实测 |
| :--- | :--- |
| cuda 与 musa 的 11 个 `.cu` | cuda 侧 572 行中 **526 行逐行相同（92%）** |
| 完全一致的内核 | `attention`、`attention_cached`、`elementwise`、`rmsnorm`、`rope`、`rope_at` |
| 近一致 | `embedding`（4 行差）、`scan`（1 行差）、`scan_parallel`（2 行差） |
| 真分歧 | `matmul`（cuda 17 行走 cuBLAS 调用，musa 325 行为 WMMA+FMA）、`softmax`（33 行差） |
| `.metal` | 692 行，MSL，**不可跨平台复用** |

### 3.4 测试与基准

| 文件 | 行数 |
| :--- | ---: |
| `tests/it/cudacontract.rs` | 1,148 |
| `tests/it/metalcontract.rs` | 1,079 |
| `tests/it/musacontract.rs` | 1,118 |
| `benches/cuda.rs` / `metal.rs` / `musa.rs` | 236 / 520 / 504 |

`cudacontract` 与 `metalcontract` 对齐后 787/1,148 行（68%）一致，是另一处可抽取的重复。

### 3.5 依赖面

生产 crate **零依赖**：全仓只有根 package 的 `[dev-dependencies]`（`Cargo.toml:30-32`）引用三者，供 `tests/` 与 `benches/` 使用。`infer`、`models`、`cpu`、`cli`、`serve` 均不依赖。这意味着**合并不会破坏任何生产依赖边**——这是合并方案唯一的真实优势，但收益也仅限于此。

## 4. 三条路线对比

| 维度 | A. 目录搬运 | B. 单 crate 合并 | C. 抽取 `gpu-core` |
| :--- | :--- | :--- | :--- |
| 减少代码行 | 0 | 0 | 估计 800–1,400 |
| 提升复用 | 无 | 无（仍需手工抽取） | 是（单一实现源） |
| 性能 | 0 | 0 | 0 |
| 安全门禁 | 需改 glob | **打穿**（§5） | 零改动 |
| 构建管线 | 无变化 | 三条挤进一个 build.rs | 无变化 |
| 文档改动 | workspace 成员 | AGENTS.md §6 + design §21.8 | 一行依赖图 |
| 风险 | 低但无收益 | 中高 | 中 |

## 5. 单 crate 合并的硬阻碍

1. **`links = "musa"`**（`crates/musa/Cargo.toml:9`）。一个 package 只能声明一个 `links`，合并后该键必须挂到 `gpu` 包，语义失真。
2. **三条互斥的构建管线**：`crates/musa/build.rs`（138 行，`mcc` 预编译 fatbin + 发射 `musa_toolkit` cfg）、`crates/metal/build.rs`（14 行，链接 CoreGraphics）、cuda 的运行时 NVRTC。三者的 target cfg 不同，合成一个 build.rs 后每个分支都要带完整条件守卫。
3. **违反 unsafe 豁免条款**。`AGENTS.md` §6 规定豁免 crate 内**恰好一个文件**声明 `#![allow(unsafe_code)]`，登记数 metal=3 / cuda=2 / musa=2。合并后要么出现三个 FFI 文件（违反"恰好一个"），要么扁平化为单文件 7 个站点（须重写宪法条款与 `docs/design.md` §21.8 全文）。
4. **`checkunsafe.sh` 会静默失效**——这是最危险的一条。脚本在 `scripts/checkunsafe.sh:80` 用 `lib="${f%/*}/lib.rs"` 从 FFI 文件路径反推 crate 根。若结构为 `crates/gpu/src/cuda/ffi.rs`，它推导出 `crates/gpu/src/cuda/lib.rs`，该文件不存在，于是 `[ -f "$lib" ]` 为假，整个 `#![deny(unsafe_code)]` 校验块被跳过——**门禁空转而不报错**。同时 `WHITELIST`（`scripts/checkunsafe.sh:25-28`）硬编码旧路径，若未同步，断言 1 直接失败；若只改路径未改计数，扁平化后单文件 7 个站点对 3 条登记项会触发 L124 的精确匹配失败。
5. **命名冲突**：`cuda` 与 `musa` 都在 crate 根导出 `matmulpath`，平铺到同一 crate 根会编译失败；`scripts/checkthresholds.sh` 的根导出宽度门禁（A4BOUND=20）也会被同时打乱。
6. **需要改动的引用面**：`tests/it/main.rs:28,40,43,44`、`tests/it/cudacontract.rs:37`、`metalcontract.rs:24`、`musacontract.rs:47,168,174`（跨后端引用 `cuda::CUDAMAXRANK`）、`musasubmission.rs:22`、`benches/{cuda,metal,musa}.rs`、`.github/workflows/ci.yml:95`（`-p cuda`）。

## 6. 推荐路线：抽取 `gpu-core`

新增 `crates/gpu-core`，`#![forbid(unsafe_code)]`，依赖 `api` / `tensor` / `lib`；依赖方向 `cuda|musa|metal → gpu-core → api`，无环。三个平台 crate 保留，`ffi.rs` 一行不动。

可下沉项，按"收益/成本"排序：

| # | 内容 | 依据 | 估计净减 |
| :-- | :--- | :--- | ---: |
| 1 | `compute` 的 33 个同名同体函数（算子上报、`record*`、形状 helper） | §3.2，532 行 cuda/metal 完全一致 | ~400（扣抽象成本） |
| 2 | `backend` 身份/能力构造 | `backend.rs` 64/64/65 行，仅常量与名字不同 | ~100 |
| 3 | `error` 枚举与 `maptobackend` | `error.rs` 36/63 行一致 | ~60 |
| 4 | `source.rs` 内核符号表契约 | 68/126 行一致 | ~60 |
| 5 | `.cu` 内核单一来源 | §3.3，526/572 行一致 | ~450 |
| 6 | 契约测试表驱动化 | §3.4，68% 一致 | ~300 |

**净减估计 800–1,400 行（总量 6–11%），不是 20%+。** 原因是：cuda/metal 的重复函数体只有 532 行量级，MUSA 因扩展了 session 而与二者分歧较大，无法整体并入同一份实现；FFI 的 5,565 行不可动。任何声称"合并能减 25%"的估算都忽略了抽象层的成本与 MUSA 的分歧度。

第 5 项需要先决策 `matmul.cu` 的归属：cuda 侧 17 行是 cuBLAS 调用壳，musa 侧 325 行是 WMMA+FMA 实现。若按 `docs/gpu-backend-unification.md` 的 P2 阶段把 CUDA 切到 WMMA，两份 `matmul.cu` 才会真正收敛；在此之前只能共享其余 10 个内核。因此**第 5 项应排在 P2 之后**。

## 7. 与既有决策文档的关系

| 文档 | 已有结论 | 本文关系 |
| :--- | :--- | :--- |
| `docs/gpu-backend-optimization-plan.md` §1.1 | "暂时保留两个并列 crate，不迁移到 `crates/gpu/...`；父目录名称不自动提供抽象、调度或性能收益" | 本文以实测数据支持该结论，并把"未来出现公共 GPU 运行时职责时再评估"具体化为 §6 |
| `docs/gpu-backend-unification.md` | 架构统一（构建期预编译、WMMA、unsafe 收敛），明确**不是 crate 合并** | §6 第 5 项依赖其 P2；两文档不冲突 |
| `AGENTS.md` §6 | unsafe 豁免：每 crate 恰好一个 FFI 文件 + 精确站点数 | §5 第 3/4 条据此判定 B 方案打穿门禁 |

## 8. 风险与回滚

抽取路线的主要风险是**抽象泄漏**：若 `gpu-core` 的 trait 为了同时容纳三端而长出大量关联类型或可选钩子，会把"减少代码"变成"增加间接层"。判据：`gpu-core` 若不引入 trait 泛型、仅以泛型参数 + 共享自由函数承载能表达的部分，则实施；若必须引入动态分发或为 MUSA 的 session 开特例分支，则该项放弃，只保留能纯净共享的部分。

回滚代价低：`gpu-core` 是无 unsafe 的纯共享 crate，删除它并把函数体拷回各平台 crate 即可恢复现状，不涉及构建管线、门禁或 tag 变更。

## 9. 未验证项

* `.cu` 单一来源的可行性只做了文本比对，**未验证** CUDA 的 NVRTC 与 MUSA 的 `mcc` 能否消费同一份源码（宏、`#pragma`、内建函数差异）。这是抽取第 5 项前的必做尖刺。
* `gpu-core` 的抽象形状未设计，净减行数是按直接重复量扣除估计成本得出的**估算值**，不是测量值。
* 合并方案的性能影响未测量；基于 crate 边界不影响代码生成的常识判定为 0，但未用基准验证。

## 10. 建议排期

按价值而非按"看起来整齐"排序：

1. **CUDA/Metal 补齐 `api::session`**——消除能力不对称，使 `submissionengine` 能驱动三端。这是当前最大的架构缺口，与 crate 布局无关。
2. **抽取 `gpu-core` 第 1–4 项**（Rust 骨架，无平台依赖，可独立验证）。
3. **契约测试表驱动化**（第 6 项），降低新增后端的成本。
4. **`.cu` 共享**（第 5 项）——排在 `docs/gpu-backend-unification.md` 的 P2（CUDA 切 WMMA）之后。
5. **不做**单 crate 合并与目录搬运。

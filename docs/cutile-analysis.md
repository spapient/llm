# `cutile`（L3 机制层）模块清单与职责判决

- 版本：v1.0（Step 1–5 完成）
- 日期：2026-11-16
- 范围：`crates/cutile` 的全部 25 个 `src/*.rs` 与 `Cargo.toml` 的出边。`prim` / `prof` / `op` 的**内容**分析属 §3、§4，本轮不展开
- 取证口径：**静态只读**。本轮没有执行 `cargo build` / `cargo test` / `cargo bench`，本文不含任何一个性能数字。所有行数、条数、行号都由 §0.2 的命令得出，读者可用同一命令复核；每个断言都带 `文件:行`
- 立场：本文只回答"现在的代码是什么"，不提出实现方案。边界裁决（允许改哪、什么算越界）落在 [ADR 0028](adr/0028-device-data-vs-mechanism-boundary.md)
- 索引位置：`crates/cutile/README.md` §3.1、`docs/ARCHITECTURE.md` 序言 prose、根 `README.md` `## 文档` 表格各有一条指向本文的相对链接；ADR 0028 与本文互相引用。`docs/ARCHITECTURE.md` 的分层表行（`:15`）与所有权表行（`:124`）被 `tests/architecture.rs` 解析，属禁改区，链接只加在正文。

---

## 0. 取证口径与复核命令

### 0.1 口径声明

1. 只读静态阅读 `src/**`、`Cargo.toml`、`docs/**`；没有运行任何会产出测量值的东西。
2. "行数"= `wc -l` 的物理行数，**包含**每个文件末尾的白盒单测挂载点（`#[cfg(test)]` + `#[path = "../tests/…"]` + `mod tests;` 三行）；22 个文件各有这三行，共 66 行。测试正文住在 `crates/cutile/tests/*.rs`，因 `crates/cutile/Cargo.toml:12` 的 `autotests = false` 不作为独立集成测试目标，**不**计入 7077（`Cargo.toml:11-12` 的注释即此意）。
3. "公开 API"只列 `pub` 项；一个模块若在 `lib.rs` 里以 `mod`（而非 `pub mod`）声明，则即使内部有 `pub fn`，对不同 crate 也不可达，本文记作"私有接缝"。

### 0.2 复核命令

| 用途 | 命令（工作目录 `crates/cutile`） |
|---|---|
| 文件数 | `ls src/*.rs \| wc -l` → `25` |
| 行数与总计 | `wc -l src/*.rs \| sort -k2` → 表尾 `7077 total` |
| 模块可见性 | `sed -n '84,108p' src/lib.rs` → 19 个 `pub mod`（`:84-102`）+ 5 个 `mod`（`:104-108`） |
| 公开项与行号 | `grep -n '^pub \(fn\|struct\|enum\|trait\|const\|type\|mod\|use\) ' src/<file>.rs` |
| 私有接缝的 `pub(crate)` 项数 | `grep -c 'pub(crate) \(fn\|struct\|enum\|const\|type\|static\) ' src/<file>.rs` |
| 单测挂载点 | `grep -n '#\[path = "\.\./tests/' src/*.rs` → 22 处 |
| 出边 | `sed -n '14,25p' Cargo.toml` → 4 条依赖（`:15-18`）+ 1 条 feature（`:22`）+ 1 条 dev-dependency（`:25`） |
| `unsafe` 计数 | `grep -rn 'unsafe' src/` → 仅 `lib.rs:66`（说明文字）与 `lib.rs:72`（`#![deny(unsafe_code)]`），代码里零处 |

### 0.3 一处口径更正：是 25 个模块，不是 24 个

计划的 Step 1 写"24 个 `src/*.rs`"，磁盘上是 **25 个**，共 **7077 行**（`ls src/*.rs | wc -l`、`wc -l src/*.rs | tail -1`）。少算的那一个是 **`lib.rs`**：

- 计划枚举的公开模块共 15 个：`ast` / `substitute` / `emit` / `build` / `cache` / `cap` / `plan` / `builtin` / `probe` / `space` / `reference` / `toolchain` / `witness` / `dirs` / `ledger`；
- 加上计划写作"engine 相关"的那 4 个命令/执行模块（`bench` / `doctor` / `verify` / `execution`）= 19 个 `pub mod`，与 `lib.rs:84-102` 逐行相符；
- 19 + 5 个私有模块（`lib.rs:104-108`）= **24**；
- `lib.rs` 自身也是 `src/*.rs`（114 行，导出 `SCHEMA_VER` 与 4 组类型转发），计划的两份清单都没有给它位置 → **25**。

计划的另一个数字"私有接缝 `device`/`harness`/`pipeline`/`runner`/`gate` 共 1655 行、约 24%"经复核**成立**：711 + 360 + 178 + 187 + 219 = **1655**，1655 / 7077 = **23.4%**（见 §1.2）。本文保留这两个数字原样。

---

## 1. 模块清单

### 1.1 25 个模块逐条（顺序 = `wc -l src/*.rs` 的字母序）

| 模块 | 行数 | 公开 API（`文件:行`） | 私有接缝 | 一句话职责 |
|---|---|---|---|---|
| `ast` | 219 | `Node`:15 `PrimCall`:26 `CallArg`:37 `build`:76 `render`:144 `encode`:160 | 否 | 单 kernel 级结构化 AST：词法片段先提升成节点序列，再由 `render` 打印回文本；`encode` 给确定性指纹，不依赖 `Debug`（`src/ast.rs:1-6`） |
| `bench` | 346 | `L3_THRESHOLD`:21 `Bench`:25 `Performance`:111 `Throughput`:140 `bench`:162 `throughput`:325 | 否 | 量一次并折成"达成率"；计时之前必须已有见证，无证据的能力不进入计时（`src/bench.rs:1-5`） |
| `build` | 559 | `source`:37 `execute`:157 `run_program`:183 `dir`:393 `Usage`:415 `contract`:430 | 否 | ⑤ 的编排：指纹 → 查缓存 → 取锁 → 拼接 → emit → 落盘（`src/build.rs:1`） |
| `builtin` | 252 | `Catalogue`:24 `Catalogue::try_builtin`:31 `Catalogue::validate`:137 | 否 | 把 `op` / `prof` 的内嵌副本解析成 `OpDecl` 与 `Profile`，并把交叉约束（重复档案、同名实现、见证文件存在）合成一次校验（`src/builtin.rs:1-6`） |
| `cache` | 608 | `CacheKey`:21 `Changed`:115 `ArtifactKey`:136 `STALE_LOCK`:195 `CACHE_LIMIT`:198 `State`:202 `Entry`:212 `Cache`:273 `cache_gc`:444 `Lock`:533 | 否 | 指纹、编译产物、锁、原子落盘、容量回收（`src/cache.rs:1`） |
| `cap` | 602 | `Path`:13 `Proof`:43 `NeedReport`:118 `NoPath`:179 `FragmentGap`:270 `fragment_gaps`:291 `negotiate`:367 `negotiate_with`:440 `inspect`:456 | 否 | ② 协商：需求 ∩ 供给、命中第一个可行变体即停、失败诊断必须给出"缺什么、供给方是什么、最近的可行解与代价"（`src/cap.rs:1-2`） |
| `device` | 711 | —（模块私有；`pub(crate)` 17 项，如 `Spec`:32 `spec`:66 `specs`:71 `witness_files`:95 `push_dispatch`:212 `HOST_FLAGS`:240；`pub fn compile_run_cuda`:444 经 `src/witness.rs:20` 再导出；`SPECS`:44 与三段编排函数均为文件私有） | **是**（`lib.rs:104`） | 一台统一的见证执行器：按 `Emit` 在描述符表里分发到 C / Metal / CUDA 三段编排，新增设备只加一行（`src/device.rs:1-9`、`:44-63`） |
| `dirs` | 58 | `Dirs`:21 | 否 | 只声明一次调用要用的三个目录（缓存根 / 工作目录 / 可选数据根），不解析默认值与环境变量（`src/dirs.rs:19-25`） |
| `doctor` | 331 | `Doctor`:25 `Storage`:52 `Declared`:65 `OpInfo`:74 `DeviceInfo`:91 `LayoutRow`:115 `Built`:124 `doctor`:139 `CacheWhy`:272 `CacheState`:303 `why`:319 | 否 | 把整条流水线每一步都摊开，包括协商为什么失败（`src/doctor.rs:1`） |
| `emit` | 131 | `Source`:12 `FRAGMENT_LIMIT`:34 `FRAGMENT_GOAL`:37 `Fragment`:45 `fnv1a`:80 `assemble`:90 | 否 | ⑤ 成文：把拼接结果包成最终源文件（划界注释 + 片段长度上限）（`src/emit.rs:1`、`:34-37`） |
| `execution` | 173 | `Execution`:21 `execute`:57 | 否 | 用调用方给的真实样本走完整条流水线并执行一次；与 `verify` 的唯一分工是样本从哪来（`src/execution.rs:1-5`） |
| `gate` | 219 | —（`pub(crate)` 0 项；`LanguageGate`:28 与 `language_gate`:100 经 `src/witness.rs:21` 再导出） | **是**（`lib.rs:105`） | 语法门：设备不在本机时，用该门语言自己的解析器把见证过一遍——仍不是设备证据（`src/gate.rs:1`） |
| `harness` | 360 | —（模块私有；`harness`:14 与 `cuda_harness`:239 是本模块的 `pub fn`，因模块私有而对其他 crate 不可达；另有 `pub(crate)` 辅助 3 个：`output_of`:41 `c_type`:212 `c_literal`:220） | **是**（`lib.rs:106`） | 把一份 `.k` 内核与样本拼成可计时的 `main()`；`cuda_harness` 是同口径的 CUDA 语义对拍**方言**（`src/harness.rs:1-5`） |
| `ledger` | 235 | `Status`:16 `LedgerCell`:45 `BackendRow`:60 `ledger`:79 | 否 | 跨档案的后端能力台账：复用已有证据折成闭集状态，自述不能冒充设备证据（`src/ledger.rs:1-5`） |
| `lib` | 114 | `#![deny(unsafe_code)]`:72、类型转发 `:76-77` `:81-82`、19 个 `pub mod`:84-102、5 个私有 `mod`:104-108、`SCHEMA_VER`:114 | 否（它声明其余 24 个） | crate 根：模块地图、L2 词表的唯一转发路径（`::prof` / `::op` 前缀避免路径歧义）、schema 版本（`src/lib.rs:72-114`） |
| `pipeline` | 178 | —（模块私有；`pub(crate)` 7 项：`Run`:26 `Prepared`:76 `shape`:101 `parse_shape`:111 `model_for`:147 `verify_shapes`:162 `rejected_shapes`:169；`Prepared::harness`:89 是结构体方法） | **是**（`lib.rs:107`） | 四条命令（`doctor` / `verify` / `bench` / `execution`）共用的**唯一**序列：找算子 → 取模型 → 建立证据 → 协商 → 选 tile → emit（`src/pipeline.rs:1-4`；`ledger` 不走这条序列，见 `src/ledger.rs:8-12`） |
| `plan` | 128 | `Plan`:18 `predict`:64 `select`:90 | 否 | ③ 选 tile：取轴展开出的**第一个** tile，再用算子给出的代价做 roofline 估值（`src/plan.rs:86-90`） |
| `probe` | 90 | `detect_cpu`:14 `detect`:40 `select`:48 | 否 | 设备探测：运行期问 CPU（`is_x86_feature_detected!`），指纹描述这台机器而不是这个产物（`src/probe.rs:8-13`） |
| `reference` | 121 | `max_rel_err`:12 `tolerance`:83 `within`:115 | 否 | 对拍口径：相对误差 + 每种 `DType` 的阈值（阈值表是契约，`tests/contracts.rs` 与 `annotation` 的 `DTYPES` 对拍）（`src/reference.rs:75-121`） |
| `runner` | 187 | —（`pub(crate)` 0 项；`compile`:74 `run`:145 `parse_bits`:166 被 `build` / `device` / `harness` 使用） | **是**（`lib.rs:108`） | 编译、执行、回读三个原语：源文本 → 可执行文件 → stdout 位模式与 stderr 计时样本（`src/runner.rs:1`、`:74-108`） |
| `space` | 90 | `SpaceUse`:9 `physical`:24 `capacity`:31 `check_capacity`:44 `check_alignment`:66 | 否 | 三级存储：语义名映射、容量校验、对齐校验（`src/space.rs:1`、`:44-86`） |
| `substitute` | 145 | `Arg`/`Segment`/`lex` 转发`:10` `FragmentSet`:13 `render`:37 | 否 | ④ 拼接：`render` 就是 `ast::build` + `ast::render` 的薄包装，全仓没有第二套字符串实现（`src/substitute.rs:35-45`、`:50`） |
| `toolchain` | 254 | `Toolchain`:16 `parse_version`:61 `detect`:85 `detect_nvcc`:94 `detect_metal`:101 `clang`:108 `probe`:113 | 否 | 工具链探测与身份：这台机器上哪台编译器 / 驱动可用、自报什么版本（`src/toolchain.rs:1`） |
| `verify` | 548 | `Verified`:28 `VerifyEntry`:65 `Negotiation`:89 `VerifyReport`:134 `negotiations`:164 `verify`:218 `verify_one`:317 | 否 | 逐设备把三件事钉在一起：见证跑通、协商选中首档、内核与标量参考对拍（`src/verify.rs:1`） |
| `witness` | 418 | `Emit`/`compile_run_cuda`/`LanguageGate` 转发`:8` `:20-21`、`Selected`:24 `MAX_OUT`:27 `establish`:30 `Scope`:92 `WitnessRun`:101 `run_all`:309 `device_available`:335 `self_check`:347 `Basis`:367 | 否 | 见证执行器：把见证源码编成可执行文件、逐条调用 `int func(float *out)` 并与 `witness.expect` 对拍（`src/witness.rs:1-6`、`:28-54`） |

### 1.2 面与行数总账

| 面 | 模块数 | 行数 | 占比 | 判据 |
|---|---|---|---|---|
| 公开面 | 20 | 5422 | 76.6% | `lib.rs:84-102` 的 19 个 `pub mod` + `lib` 自身 |
| 私有接缝 | 5 | **1655** | **23.4%** | `lib.rs:104-108` 的 `mod device` / `mod gate` / `mod harness` / `mod pipeline` / `mod runner`（711 + 219 + 360 + 178 + 187） |
| 合计 | 25 | **7077** | 100% | `wc -l src/*.rs \| tail -1` |

一句话读法：**本 crate 的 76.6% 是对外可调用的机制与命令，23.4% 是不导出的接缝**——接缝里就包含全部"按语言分岔"的编排（`device` 三节 + `harness` 两方言，1071 行，占 15.1%）。这与 `docs/ARCHITECTURE.md:15`（L3 行）写的「机制（协商 → 拼接 → emit → 编译执行）与其上的命令策略」、「只消费 L2 的数据」是同一件事的两种表述。

### 1.3 `Cargo.toml` 的出边 = 层级归属证据

| 出边 | 行 | 指向 | 归属含义 |
|---|---|---|---|
| `diagnostic` | `Cargo.toml:15` | L1 诊断 | 错误类型、`Code` 闭集来自下层；本层不自建错误词汇表 |
| `prim` | `Cargo.toml:16` | L2 片段层 | 片段正文与见证源码是**数据**，本层只装配（`substitute` 只转发 `FragmentSet`，`src/substitute.rs:13`） |
| `op` | `Cargo.toml:17` | L2 声明层 | `OpDecl` / `Model` / `Signature` 是**数据**；机制层不认识任何算子（`src/plan.rs:88-89`、`src/harness.rs:4-5`） |
| `prof` | `Cargo.toml:18` | L2 档案层 | `Profile` / `Emit` / `atom` 闭集是**数据**；协商只读档案（`src/cap.rs:1-2`） |
| feature `device-image` | `Cargo.toml:22` | `prim/device-image` | 只是把片段层的开关透传下去，本层不新增分支（注释见 `Cargo.toml:21`） |
| dev-dep `testkit` | `Cargo.toml:25` | 测试脚手架 | 只进测试目标 |

**四条出边全部向下（L1、L2），零条指向 `engine` / `tensor` / `storage` / `serve` 等运行时成员，零条引入 FFI crate。** 复核：`awk '/^[a-z].* = \{ path/' crates/cutile/Cargo.toml` 出 5 行——上面 4 条依赖 + 第 5 行 `testkit`（`Cargo.toml:25`，只在 `[dev-dependencies]` 段、只进测试目标）。这就是"L3 是机制层、不是运行时后端"在依赖图上的证据形态。

---

## 2. 职责判决（四类）

### 2.1 四类的定义（先定判据，再判）

| 判决 | 判据（满足其一即可入类） |
|---|---|
| **唯一机制** | 该模块承载的变换在仓内**只有这一份**实现，删掉就没有别的路径能产出同样的结果；且它不含"某个厂商/某门语言"的知识 |
| **纯数据** | 模块内容本质是**表 / 常量 / 契约**，其事实源已经或应该在 L2（`op` 的词汇表、`prof` 的档案 schema、`prim` 的片段正文）；留在 L3 只是落点历史，不是机制 |
| **命令策略** | 模块决定"跑哪些形状、跑几次、对拍什么、报告哪些字段"，即 CLI 面向的策略；换成另一种报告样式不需要动机制 |
| **设备编排** | 模块把"某门语言的编译 → 执行 → 回读"串起来，或回答"这台机器上哪门语言可用"；它的差异表现为**按 `Emit` 表行**，不是算法 |

### 2.2 判决表（25 个模块，无一遗漏）

| 模块 | 判决 | 判决理由（可复核） | 数据 / 机制边界 |
|---|---|---|---|
| `cap` | **唯一机制** | 协商只有一份：`negotiate`（档案自述，`src/cap.rs:367`）与 `negotiate_with`（要求供给出现在本次 `Proof`，`:440`）共用同一条链判定；`Proof`（`:43`）把"档案说"与"这次跑到"分开，`NoPath`（`:179`）与 `FragmentGap`（`:270`）承担可解释性 | 拥有：链判定、降级顺序、诊断文本。只消费：`Profile` 的 `provides` / `variants`（L2 数据）；不含任何设备名字面量 |
| `substitute` | **唯一机制** | 拼接只有一份：`render`（`src/substitute.rs:37-45`）就是 `ast::build` + `ast::render`，注释明写"两条路径不会各写一遍规则"（`:35-36`、`:49-50`） | 拥有：替换与缩进对齐规则。只消费：`op::tpl` 的词法（`:10`）与 `prim::FragmentSet`（`:13`） |
| `ast` | **唯一机制** | 结构化 AST 是模板渲染的可见中间产物；`encode`（`src/ast.rs:160`）给确定性编码作指纹输入，不依赖 `Debug`（`:1-6`） | 拥有：节点序列与编码。只消费：`OpDecl.template`、`TileCombo`、`FragmentSet`（`build` 的入参，`:76-80`）；不解析 C 语句本身（`:1-6`） |
| `emit` | **唯一机制** | 成文只有一份：`assemble`（`src/emit.rs:90`）负责划界注释；`FRAGMENT_LIMIT` / `FRAGMENT_GOAL`（`:34-37`）是唯一一处对片段体量的上限 | 拥有：最终源文件的包装形状。只消费：拼接后的文本；不含语言语法知识 |
| `cache` | **唯一机制** | 指纹 → 产物 → 锁 → 原子落盘 → 回收是一条链：`CacheKey`（`src/cache.rs:21`）、`ArtifactKey`（`:136`）、`Lock`（`:533`）、`cache_gc`（`:444`）；`STALE_LOCK` / `CACHE_LIMIT`（`:195`、`:198`）是唯一常量源 | 拥有：键的构造与失效（`Changed`，`:115`）。只消费：`SCHEMA_VER`（`lib.rs:114`）作为闭集变更信号 |
| `build` | **唯一机制** | ⑤ 的编排只在这里写一次（`src/build.rs:1`）：`source`（`:37`）→ `execute`（`:157`）→ `run_program`（`:183`）；`contract`（`:430`）是编译产物的对外承诺 | 拥有：五步的接线与落盘。只消费：`emit::Source`、`cache`、`runner`；**不含语言分支**（语言在 `device` / `harness`） |
| `plan` | **唯一机制** | 选 tile 是「取轴展开出的第一个 tile」这一条规则（`src/plan.rs:86` 的 `select` 文档、`:95` 的 `first_tile` 调用，`:103-104` 是其定义：「TOML 顺序即优先级，不在这里打分」），不是打分搜索；`predict`（`:64`）只给估值 | 拥有：选择顺序与估值口径。只消费：`op::model::Model::cost` 给的 `flops` / `bytes`（`:88-89` 明写"流量公式属于算子语义，机制层不认识任何算子"） |
| `builtin` | **唯一机制** | 把 `op::builtin()`（`src/builtin.rs:33`）与 `prof::names()` / `prof::embedded`（`:37-40`）装成 `Catalogue`（`:24`），并做一次跨层校验 `validate`（`:137`）；`CATALOGUE` 这个 `OnceLock` 静态项（`:20`）保证解析只发生一次，重复调用只 clone 缓存结果 | 拥有：装载顺序与交叉约束。数据本体在 `op` / `prof`，本模块不复制任何一条档案 |
| `witness` | **唯一机制** | 证据的唯一产生点：`establish`（`src/witness.rs:30-54`）把设备执行、host 自检、语言门合成 `Evidence`，并给出 `Basis`（`:367`）——`grade` 是档案自述，`Basis` 是这次执行到的结论 | 拥有：证据合成与 `Scope` 划分（`:92`）。只消费：`witness = { file, func, expect }`（L2）与 `prim` 的见证源码 |
| `pipeline` | **唯一机制** | **四条命令**（`doctor` / `verify` / `bench` / `execution`）共用一条序列；第五个命令 `ledger` 不走它，只聚合 `witness` 的 `Evidence` / `Proof`（`src/ledger.rs:3-5`、`:8-12`）。理由写在 `src/pipeline.rs:3-4`："四条命令各写一遍曾让同一台设备在两个命令里得出不同结论"；`Run`（`:26`）/ `Prepared`（`:76`）是这条序列的两个产物类型 | 拥有：序列本身。只消费：`op::decl::OpDecl`、`op::model`；不含任何命令专属分支 |
| `runner` | **唯一机制** | 编译 / 执行 / 回读各一份：`compile`（`src/runner.rs:74`）里的源文件名甚至固定为 `kernel.c`（`:82-83`），`run`（`:145`）与 `parse_bits`（`:166`）不认设备 | 拥有：进程原语与 stdout/stderr 解析。只消费：工具名与 flags（来自 `toolchain` + 档案 `[compile]`）；**零设备分支** |
| `lib` | **唯一机制** | 唯一转发路径：`pub use ::prof::atom`（`src/lib.rs:76-77`）、`pub use ::op::decl`（`:81-82`），注释说明"用 `::prof` 前缀强制指向 crate（本 crate 也导出同名模块，避免路径歧义）"（`:75`）与"机制里没有第二份闭集"（`:80`）；`SCHEMA_VER`（`:114`）是缓存与 plan 失效的唯一开关 | 拥有：模块地图、版本号、`deny(unsafe_code)`（`:72`）。只消费：两个 L2 crate 的公开词汇表 |
| `reference` | **纯数据** | `tolerance`（`src/reference.rs:83-112`）是一张以 `DType` 为键的阈值表；文件自述"「非零阈值的类型集合」是**契约**：它必须与注解层可声明的精度闭集（`annotation` 的 `DTYPES`）逐项相等，`tests/contracts.rs` 会对拍"（`:80-82`） | 事实源应在 `op` / `annotation` 的 `DTYPES` 旁（阈值就是类型的属性）；本模块只该剩 `max_rel_err`（`:12`）这一条算法 |
| `space` | **纯数据** | 每条规则都由档案字段决定：`check_capacity`（`src/space.rs:44`）读 `capacity()`（`:31`）→ `Profile.mem.*`，`check_alignment`（`:66`）读 `profile.mem.fast_bank`；没有一条跨设备机制 | 事实源应在 `prof` 的档案 schema 校验（"这份声明与这台设备的组合不存在"，`:41-43`），本模块只是把字段写成代码 |
| `dirs` | **纯数据** | 只有三个字段（`src/dirs.rs:22-24`）；默认值、环境变量与 `--set` 解析住在调用方 `cli::settings`（`src/lib.rs:68-70` 明写「配置是输入，不是职责」，并交代「内核里没有一处读 `CUTILE_*`」） | 纯配置载体：没有算法、没有设备知识，换嵌入方只需换实参 |
| `doctor` | **命令策略** | 决定"摊开哪些步骤、报告哪些字段"：`Doctor` / `Storage` / `Declared` / `OpInfo` / `DeviceInfo` / `LayoutRow` / `Built`（`src/doctor.rs:25-133`）是报告结构；`why`（`:319`）回答缓存为什么失效 | 机制在 `cap` / `cache` / `pipeline`；本模块只选择呈现面 |
| `verify` | **命令策略** | 决定"跑哪些形状、钉哪三件事"：`verify`（`src/verify.rs:218`）/ `verify_one`（`:317`）/ `negotiations`（`:164`）与 `VerifyEntry` / `Negotiation`（`:65`、`:89`）都是判定口径 | 执行走 `pipeline` + `witness` + `reference`；本模块不自己编译 |
| `bench` | **命令策略** | 决定"计时口径与达成率阈值"：`L3_THRESHOLD`（`src/bench.rs:21`）按 `p50` 而非 `min` 判定（`:1-5`），`throughput`（`:325`）只做折算 | 计时原语在 `runner::Timing`；本模块只规定"哪些数进报告、怎么折" |
| `execution` | **命令策略** | 与 `verify` 的差别只在**样本从哪来**：调用方给的真实数据 vs 算子声明的确定性样本（`src/execution.rs:3-5`） | 共用 `pipeline`（`:12` 的 `use crate::pipeline::{Prepared, Run};`），本模块不含第二条执行路径 |
| `ledger` | **命令策略** | 明确"不产生新证据"：复用 `witness::establish` 的 `Evidence` / `Proof` 做聚合（`src/ledger.rs:3-5`），把结论折成 `Status` 闭集（`:16`） | 拥有：聚合口径与闭集词汇。只消费：已有证据 |
| `device` | **设备编排** | 四段私有编排 + 一张表：`SPECS`（`src/device.rs:44-63`）三行分别是 C / Metal / CUDA；`共享装载、命名、起进程与回读` 段 `:75-77`、C 节 `:234-237`、CUDA 节 `:331-336`、Metal 节 `:488-494`；"新增设备只在 `SPECS` 加一行，不改任何 `match`"（`:5`） | 拥有：按 `Emit` 找表行、拼程序、跑、回读。只消费：`prim` 的见证源码 + 档案的 `emit` / `tool` / `flags` |
| `harness` | **设备编排** | 两方言：`harness`（C，`src/harness.rs:14`）与 `cuda_harness`（CUDA，`:239`）；样本与实参顺序来自 `op::model` / `op::sig`，"本模块只按名字与类型把缓冲渲染成 C 数组，不认识任何算子"（`:1-5`） | 拥有：可计时 `main()` 的生成。只消费：`Signature` / `Sample`（L2） |
| `toolchain` | **设备编排** | 回答"这门语言在本机怎么编译"：`detect`（`src/toolchain.rs:85`）按扩展/平台选，`detect_nvcc`（`:94`）、`detect_metal`（`:101`）、`clang`（`:108`）各一条 | 拥有：探测与版本解析（`:61`）。只消费：档案里的 `[compile].tool`；不判断设备能力 |
| `gate` | **设备编排** | 语法门按语言选前端（`src/gate.rs:28`、`:100`），结论只有三种，且"仍不是设备证据"（`src/witness.rs:1-6`） | 拥有：语言 → 解析器的映射。只消费：见证源码 |
| `probe` | **设备编排** | 向机器提问："指令集用 `is_x86_feature_detected!` / `is_aarch64_feature_detected!` 在**运行期**问 CPU，不用 `cfg!(target_feature)`"，理由是"指纹要描述这台机器，不是这个产物"（`src/probe.rs:8-13`）；`select`（`:48`）按命中规则条数挑档案 | 拥有：探测与指纹匹配规则。只消费：`prof::Detected` / `Profile` 的匹配表达式（L2） |

### 2.3 四类合计（自校验：必须等于 7077）

| 判决 | 模块数 | 模块 | 行数合计 | 占比 |
|---|---|---|---|---|
| 唯一机制 | 12 | `lib` `ast` `substitute` `emit` `cache` `cap` `build` `plan` `builtin` `witness` `pipeline` `runner` | **3541** | 50.0% |
| 纯数据 | 3 | `reference` `space` `dirs` | **269** | 3.8% |
| 命令策略 | 5 | `doctor` `verify` `bench` `execution` `ledger` | **1633** | 23.1% |
| 设备编排 | 5 | `device` `harness` `toolchain` `gate` `probe` | **1634** | 23.1% |
| 合计 | **25** | — | **7077** | 100% |

复核方式：`wc -l src/*.rs` 的每个数字在 §1.1 已列出，按本表的四行相加即 3541 + 269 + 1633 + 1634 = **7077**，与 `wc -l src/*.rs | tail -1` 的 `7077 total` 相等；任何一格对不上就说明有模块被漏判。占比 = 该行行数 / 7077。

### 2.4 从判决表能直接读出的三条结论

1. **机制的一半（3541 行）与设备无关**：`cap` / `substitute` / `ast` / `emit` / `cache` / `build` / `plan` / `pipeline` / `runner` 九个文件里 `nvidia|metal|cuda|ascend|intel|nvcc` **零命中**（`awk` 逐文件计数，见 §0.2）；`runner` 连源文件名都固定为 `kernel.c`（`src/runner.rs:82`）。`lib.rs` 的大小写敏感命中 4 次，全在 `//!` 文档里：`:29` 的表格行（`cuda`）与 `:54-55` 的用法示例（`nvidia`），不是代码分支；`:53` 的 `NVIDIA` 大写行不计入这 4 次。设备差异不以"代码分支"存在，只以"表行 + 档案字段"存在（`src/device.rs:44-63`）。
2. **真正该被判定为"越界"的位置是明确的**：本层允许出现某门语言字样的地方只有三处——`device` 里自 `:234` 起的 C / CUDA / Metal 三段编排（该文件共 711 行）、`harness` 的两方言（360 行）、`toolchain` 的三条探测函数（`toolchain.rs:94-110`）。它们的存在形式统一为"表行 + 一段私有编排"，不是"新增一个持有设备句柄的模块"。
3. **"纯数据"只剩 269 行（3.8%）**：`reference` 的阈值表、`space` 的档案字段校验、`dirs` 的三目录声明——三者的事实源都在 L2 或调用方。这正是 [ADR 0028](adr/0028-device-data-vs-mechanism-boundary.md) 判定"数据在 L2、机制在 L3"时要引用的存量证据。

---

## 3. CPU 的完整实现链路

一句话结论：**CPU 的后端不是 Rust 运行时，而是"档案里的一段 C 前导 + 本机 `cc` 子进程 + stdout/stderr 文本协议"**。`cutile` 在整条链路上只做三件与数值无关的事：把模板槽位换成数据、把片段与前导拼成一份 `kernel.c`、把它的输出读回来与标量参考对拍。数值全在 C 里，且 `cutile` 的源码里连一个 NEON 内建或前导宏名都不出现（3.6 的复核命令给出 0 命中）。

### 3.1 模板侧：两份 `.k`，一套槽位

`crates/op/ops/` 共 22 份 `.k` 模板（`ls crates/op/ops/*.k | wc -l`）。两个 GEMV 家族的槽位逐条点数：

| 槽位 | `gemv_f32.k` | `gemv_i4.k` | 说明 |
|---|---|---|---|
| `@load` | `gemv_f32.k:17` | `gemv_i4.k:14` | 把权重行起点绑到 `row`（`const float *row = 0;` 在 `gemv_f32.k:16`） |
| `@load_async` | `gemv_f32.k:24`（`xs`）、`:25`（`wp`） | `gemv_i4.k:23`（只有 `xs`） | f32 版多一次"把权重行搬进寄存器" |
| `@commit_wait` | `gemv_f32.k:26` | `gemv_i4.k:24` | 提交并等待 |
| `@quant` | **不出现** | `gemv_i4.k:25` | int4 解包 + 标度，只有 i4 家族有 |
| `@dot` | `gemv_f32.k:27` | `gemv_i4.k:26` | 乘累加 |
| `@bar` | `gemv_f32.k:28` | `gemv_i4.k:27` | 同步 |
| `@cvt` | `gemv_f32.k:33` | `gemv_i4.k:32` | 类型转换 |
| `@reduce` | `gemv_f32.k:34` | `gemv_i4.k:33` | 部分和 → 总量 |
| `@store` | `gemv_f32.k:35` | `gemv_i4.k:34` | 回写 |

计数复核：`grep -o '@[a-z_]*(' crates/op/ops/gemv_f32.k | sort | uniq -c` 得到 `bar 1 / commit_wait 1 / cvt 1 / dot 1 / load 1 / load_async 2 / reduce 1 / store 1`（`@prim(` 的那 1 次命中在 `gemv_f32.k:2` 的注释里，不是调用点）；`gemv_i4.k` 同法得到 `load_async 1 / quant 1`，其余各 1。

"逻辑只有一份"的含义就是上表两侧的**形状完全相同**——同一份分块、同一套累加顺序，差异只在有无 `@quant` 与 `@load_async` 的条数：

> 「逻辑只有一份；设备差异全部在 `prim/<vendor>/<prim>.<ext>` 里。」——`crates/op/ops/gemv_f32.k:3`、`crates/op/ops/gemv_i4.k:3`

> 「与 gemv_i4 的模板逐行对照：少了 `@quant` 与标度指针，多了把权重行本身搬进寄存器的那次 `@load_async`。累加仍是"每块一个部分和，再加进总量"——与标量参考的分块顺序一致，因此对拍比的是同一份舍入顺序，而不是运气。」——`crates/op/ops/gemv_f32.k:6-8`

模板里的 `{{…}}` 只有轴与布局两类，不含任何设备字样：`grep -o '{{[A-Za-z]*}}' crates/op/ops/gemv_f32.k | sort | uniq -c` → `{{op}} 2 / {{w}} 1 / {{x}} 1`；i4 版多一对 `{{GMAX}} 2`（来自 `gemv_i4.toml` 的 `[axes]`）。

### 3.2 片段侧：十个文件里七个是壳

`crates/prim/cpu/` 共 10 个文件、228 行（`wc -l crates/prim/cpu/*` → 末行 `228 total`），其中 9 个原语片段 62 行、`witness.c` 166 行。

| 片段 | 行数 | 正文形态 | 判决 |
|---|---|---|---|
| `bar.c` | 3 | 空操作（无语句） | 壳（CPU 无线程块同步） |
| `commit_wait.c` | 3 | 空操作 | 壳 |
| `cvt.c` | 3 | `($0) = (float)($1);` | 壳（一次 C 转型，无算术） |
| `load.c` | 3 | 赋值 | 壳 |
| `load_async.c` | 3 | 赋值 | 壳（无异步语义可表达） |
| `store.c` | 3 | 赋值 | 壳 |
| `reduce.c` | 4 | 单行调用 `_cutile_sum_f32` / `_cutile_max_f32` | 壳**但是零**：算术在前导 |
| `quant.c` | 8 | `(` 基址、`_b` 字节、`(float)((int)(_b & 0x0Fu) - 8) * _s` 解包两组 | **有算术**（解包 + 缩放） |
| `dot.c` | 32 | `#if defined(__aarch64__)` 的 `vmlaq_f32` 主循环 + 标量回退 | **有算术**（向量 FMA） |

`reduce.c` 全文只有 4 行，这正是"壳"的最好证据：同步、转换、装载、回写这 7 个片段合计 22 行，一个 `_cutile_*` 实现都不含，只把调用点写出来；真正的实现被 `reduce.c` 这类壳"点名"到前导里。

**对计划的更正一**：计划的"片段只是壳"只对 7 个成立。`dot.c`（32 行）与 `quant.c`（8 行）是 CPU 片段层里两段真算术——这也是"新增一条加速链路的主战场在 `prim`"的第一个反例：片段层**允许**承载算术，只是 CPU 把重活（归约、exp）放进了前导。

### 3.3 真正的实现：`[compile].preamble` 的六个 `static inline`

档案的 `[compile]` 段在 `crates/prof/devices/cpu.toml:40`，`flags = ["-O3", "-std=c99"]` 在 `:41`，`preamble = """` 在 `:42`，前导正文到 `:194` 结束（153 行）。`#include <arm_neon.h>` 被 `#if defined(__aarch64__)` 包住（`:45-47`），因此同一份前导在非 aarch64 上退化为标量。

`grep -n '^static inline' crates/prof/devices/cpu.toml` 恰好 6 条，这就是 CPU 全部的性能实现：

| 前导 helper | 锚点 | 实现要点 |
|---|---|---|
| `_cutile_sum_f32` | `cpu.toml:53` | 四个累加器 `s0..s3`（`:57-60`）、16 宽主循环（`:62-67`）、四路合并（`:68`）、4 宽尾循环与水平归约（`:69-76`）、标量回退（`:81-83`） |
| `_cutile_max_f32` | `cpu.toml:85` | 同结构，`-3.402823466e38f` 起手（`:89-92`），收尾 `vmaxvq`（`:104`） |
| `_cutile_exp_f32` | `cpu.toml:121` | 范围守卫（`:122`）、就近取整（`:125`）、下溢/上溢短路（`:126-127`）、Horner 五项（`:130`）、`2^k` 位模式合成（`:132`） |
| `_cutile_exp4_f32` | `cpu.toml:136` | 上面四条同款 NEON 版：`vrndaq_f32`（`:137`）、`vfmaq` 链（`:143-147`）、指数位移（`:148`）、上下溢掩码（`:149-153`） |
| `_cutile_exp_sum_f32` | `cpu.toml:157` | softmax 分母，8 宽双累加（`:162-165`） |
| `_cutile_exp_norm_f32` | `cpu.toml:179` | softmax 写回，`vdivq_f32`（`:186`） |

软件 exp 的精度是**写进档案契约**的，不是事后测出来的：

> 「规约 exp(t)=2^k·exp(r)，|r|<=ln2/2，5 次 Taylor Horner，2^k 走位模式；aarch64+连续走 NEON 4-wide，余回退标量。近似相对误差 ≤2e-6（阈值 1e-5），下溢置 0、上溢置 inf。exp 不进原语闭集，与 sum/max 同挂 preamble；闭集 exp2 片段按 ADR 0027 仍推迟。」——`crates/prof/devices/cpu.toml:114-116`

而 `_cutile_sum_f32` 的语义与"分块顺序"绑定在前导注释里：

> 「aarch64 且 stride==1 走 NEON 连续归约：四累加器打破单条加法/取大依赖链；余下元素按 4-wide 收口、标量补齐。stride>1 或其余架构回退顺序标量折叠。向量分路累加顺序与标量参考不同，数值以内核对拍为准（f32 阈值 1e-5）；max 对有限输入取大，NaN 边界由算子层守。」——`crates/prof/devices/cpu.toml:50-52`

**结论（可复核）**：`cutile` 把这段前导**原样拼进生成源**，它不知道里面有什么。

```text
emit.rs:112-117   if !profile.preamble.is_empty() { text.push_str(&profile.preamble); … }
emit.rs:119       text.push_str(body.trim_start());      // 模板 + 片段渲染出来的正文
emit.rs:123-124   Source { lang: profile.meta.emit, … }
```

`grep -rn '_cutile_\|vmlaq\|vaddq' crates/cutile/src/ | wc -l` → **0**。也就是说：CPU 的所有浮点算术都在这 153 行 C 里，Rust 侧零算术——Rust 侧只有字符串拼装与阈值（后者在 `reference.rs`，见 §1.1）。

### 3.4 证据侧：`witness.c` 与 14 条 `provides`

**对计划的更正二**：计划写"档案里 8 条 `provides`"，实测是 **14 条**（`grep -c '^\[\[provides\]\]' crates/prof/devices/cpu.toml` → 14），且每条都自带 `witness = { file, func, expect }`（`grep -c '^witness'` → 14）与 `grade`（`grep -c '^grade'` → 14）。第一条在 `cpu.toml:196`，最后一条在 `cpu.toml:324`。

`crates/prim/cpu/witness.c` 166 行，开头两段是 ABI 契约：

> 「统一 ABI：`int witness_<name>(float *out)` 返回写入的元素个数（≤ 8），值按 f32 读回后与档案里的 `witness.expect` 对拍」——`crates/prim/cpu/witness.c:3-4`

> 「两个范围互不重叠、合起来覆盖全部声明：`grade ≥ L2` 的见证由**设备执行**（emit = c 时设备就是本机）跑，结论是 L2 证据；其余（`atomic` 与末尾四条 L0）由**本机自检**跑，结论只说明"源码可编译、期望值算得对"，不改写任何等级。」——`crates/prim/cpu/witness.c:8-10`

14 条 `provides` 逐条（判据 = §2 里 `witness`/`verify` 的"声明 vs 执行"分离）：

| # | `prim` / `impl` | 档案行 | grade | 见证函数（`witness.c`） | `expect` |
|---|---|---|---|---|---|
| 1 | `quant` / `scalar_i4` | `cpu.toml:196` | L2 | `witness_quant`（`witness.c:26`） | `[-4.0, 3.5]`（`cpu.toml:203`） |
| 2 | `dot` / `scalar_f32` | `cpu.toml:205` | L2 | `witness_dot`（`witness.c:14`） | `[70.0]`（`cpu.toml:212`） |
| 3 | `load` / `memcpy` | `cpu.toml:214` | L2 | `witness_load`（`witness.c:37`） | `[1.0]`（`cpu.toml:224`） |
| 4 | `load_async` / `memcpy_prefetch` | `cpu.toml:226` | L2 | `witness_load_async`（`witness.c:45`） | `[1.0]`（`cpu.toml:237`） |
| 5 | `commit_wait` / `noop` | `cpu.toml:239` | L2 | `witness_commit_wait`（`witness.c:55`） | `[0.0]`（`cpu.toml:243`） |
| 6 | `store` / `memcpy_out` | `cpu.toml:245` | L2 | `witness_store`（`witness.c:70`） | `[3.25]`（`cpu.toml:253`） |
| 7 | `bar` / `noop` | `cpu.toml:255` | L2 | `witness_bar`（`witness.c:62`） | `[0.0]`（`cpu.toml:259`） |
| 8 | `cvt` / `c_cast` | `cpu.toml:261` | L2 | `witness_cvt`（`witness.c:79`） | `[3.0]`（`cpu.toml:267`） |
| 9 | `reduce` / `scalar_loop` | `cpu.toml:269` | L2 | `witness_reduce`（`witness.c:86`） | `[7.5]`（`cpu.toml:276`） |
| 10 | `atomic` / `rmw_scalar` | `cpu.toml:280` | **L0** | `witness_atomic`（`witness.c:97`） | `[5.0]`（`cpu.toml:287`） |
| 11 | `mma` / `scalar_tile` | `cpu.toml:293` | **L0** | `witness_mma`（`witness.c:112`） | `[1.0, 2.0, 3.0, 4.0]`（`cpu.toml:303`） |
| 12 | `scan` / `prefix_loop` | `cpu.toml:306` | **L0** | `witness_scan`（`witness.c:128`） | `[1.0, 3.0, 6.0, 10.0]`（`cpu.toml:314`） |
| 13 | `exp2` / `pow2_loop` | `cpu.toml:316` | **L0** | `witness_exp2`（`witness.c:139`） | `[8.0, 0.5]`（`cpu.toml:322`） |
| 14 | `rcp` / `recip_newton` | `cpu.toml:324` | **L0** | `witness_rcp`（`witness.c:157`） | `[0.25, 0.5]`（`cpu.toml:331`） |

grade 分布：**9 条 L2 + 5 条 L0**。L0 的那五条（`atomic` / `mma` / `scan` / `exp2` / `rcp`）不是"没跑"，而是档案自述它们**没有片段文本**，因此声明侧就不成立：

> 「单线程 CPU 只有读改写，没有多线程原子语义；也没有片段文本，所以是 L0——协商会把它当成"不具备"，这正是这条声明要演示的东西。」——`crates/prof/devices/cpu.toml:278-279`

> 「这四条没有片段文本，等级因此是 L0——矩阵补的是覆盖，不是等级。」——`crates/prof/devices/cpu.toml:292`

这 14 条见证由 `witness::establish` 一次跑完，产出**三条命令共用**的那份证据：

```text
witness.rs:30    pub fn establish(profile: &Profile, dirs: &Dirs) -> Evidence
witness.rs:408   pub fn proof(&self, profile: &Profile) -> Proof     // Executed → 实跑；Declared → 档案自述
```

见证的运行参数也来自档案，而不是 `cutile` 的内置常量：`Scope::Device` 时用 `profile.preamble` + `profile.flags`（`device.rs:245`），编译走 `build::run_program`（`device.rs:285`）。

### 3.5 机制侧：从装载到数字的七步

| 步 | 入口（`文件:行`） | 这一步决定什么 | 与 CPU 的关系 |
|---|---|---|---|
| ① 装载 | `builtin.rs:31` `try_builtin` → `:60` `load` | 三层数据（声明 / 档案 / 片段）的闭集与交叉校验；`OnceLock` 静态项 `:20` 保证只解析一次 | 挑出 `cpu` 档案与 `prim/cpu/*.c` |
| ② 证据 | `witness.rs:30` `establish` → `:408` `proof` | 把档案的 `provides` 变成这次执行得到的 `Proof` | 跑 14 条见证，9 条升级为 `Proof` |
| ③ 协商 | `cap.rs:440` `negotiate_with` | 供给必须出现在 `Proof` 里，否则降级并给缺项诊断 | `emit = "c"` 的链只有一条，协商常常是恒等 |
| ④ 选 tile | `plan.rs:90` `select`、`:104` `first_tile` | 按 roofline 估一个 tile 组合；不认识任何算子 | 与设备无关，只吃 `cost` |
| ⑤ 替换 | `substitute.rs:37` `render` → `ast.rs:76` `build` → `:126` `prim_node` | 槽位取值 + 片段拼接（缩进对齐、去重复 `;`） | 把 `@reduce(...)` 换成 `reduce.c` 那 4 行 |
| ⑥ 成文 | `emit.rs:90` `assemble` | 前导 + 正文 → `Source { lang: emit }`（`emit.rs:12`、`:123`） | `emit = "c"` |
| ⑦ 构建/缓存/执行 | `build.rs:157` `execute` → `:206` `cached_run` → `:259` `compile_fresh` → `runner.rs:87` | 指纹命中则直接跑旧产物，否则调 `cc` | 唯一与编译器打交道的地方 |

`runner` 的两个固定事实（"后端"就藏在这两行里）：

```text
runner.rs:82   let src = dir.join("kernel.c");      // 生成源的文件名固定
runner.rs:87   let compile = Command::new(tool)     // tool = toolchain::detect() 挑出的 cc/clang/gcc
               runner.rs:88-91  .args(flags).arg("-o").arg(&exe).arg(&src)
```

`tool` 的来源是探针，不是配置：`toolchain.rs:85-89` 依次试 `cc` → `clang` → `gcc`，第一个能自报版本的胜出；`Toolchain { command, version, banner }` 在 `toolchain.rs:16-23`。

回读是文本协议，只有两种行：

```text
runner.rs:166   pub fn parse_bits(stdout: &str) -> Vec<f32>       // 逐个 u32 位模式 → 比较
runner.rs:174   fn parse_samples(stderr: &str) -> Vec<f64>       // 解析下面那行打的 "ns %.3f" 计时样本
harness.rs:198  printf("%u\n", cutile_u.u);                      // 位模式，避免文本精度损失
harness.rs:193  fprintf(stderr, "ns %.3f\n", (cutile_t1 - cutile_t0) / cutile_batch);
```

- **`verify` 的对拍**：`verify.rs:250` `establish` → `:252` `proof` → `:255/:317` `verify_one` → `:379` `measure` 逐个形状跑（形状矩阵来自 `pipeline.rs:162` `verify_shapes`，含被显式拒绝的形状 `:169`），每格 `cell` 里"样本、计划、参考与判据两条通道完全相同，只有“怎么编译执行”按通道分"（`verify.rs:386`）。`Emit::C` 走 `Lane::Host` + `toolchain::detect()`（`verify.rs:356`；`Lane` 定义在 `:345-350`）：CPU 的"设备通道"就是本机宿主通道。
- **`bench` 的计时**：`bench.rs:162` `bench` → `:226` `run_sample` → `:239` `build::execute` → `:240` `timings`；三个数是 `Timing::min_ns`（`runner.rs:31`）、`p50_ns`（`:37`）、`p95_ns`（`:53`），缺失时退回 `min`（`bench.rs:268-276`）。CPU 没有设备可用性问题，但要先过 `Basis::Executed` 这道门（`bench.rs:200`），否则"没有可测的对象"。

### 3.6 结论：CPU 的"后端"是四样东西，没有一件是 Rust 运行时

| 组成 | 证据 |
|---|---|
| `emit = "c"` | `cpu.toml` 的 `[meta]`；由 `emit.rs:123-124` 变成 `Source.lang` |
| `tool = "cc"` | `toolchain.rs:85-89` 的探针序列；由 `runner.rs:87` 落成一次子进程调用 |
| 前导里的宏与 `static inline` | `cpu.toml:42-194` 的 153 行，`emit.rs:112-117` 原样拼接 |
| 子进程 + stdout/stderr 文本协议 | `runner.rs:82/87/166/174`，`harness.rs:193/198` |

因此 `#![deny(unsafe_code)]`（`crates/cutile/src/lib.rs:72`）与"C 通道"不是矛盾，而是同一件事的两面：

> 「`unsafe` 在本 crate 被整体禁止；P0 的 C 通道通过子进程调用 `cc`，不需要 FFI。」——`crates/cutile/src/lib.rs:66`

`grep -rn 'unsafe' crates/cutile/src/` 只有两处命中：`lib.rs:66`（上面这句说明文字）与 `lib.rs:72`（这条 `deny`）。全仓 `unsafe` 计数保持为零。

这条链路正好印证 §2 的三条判决：

1. **"设备运行时不在 `cutile`"是事实，不是口径**：CPU 的运行时是"宿主进程 + `cc`"，`cutile` 里没有一段代码持有设备句柄。若真要接 CUDA/Metal 的缓冲与命令队列，落点只能是 `gpu`（ADR 0025 的后端三件套）或 `engine` 的执行器，而不是 `cutile`。
2. **"数据在 L2"有 153 行硬证据**：CPU 的性能全在 `cpu.toml:42-194`，把它换成一份写坏的前导，`cutile` 一行不改而结果全错——数据与机制的分界在此可判定。
3. **"`dot.c`/`quant.c` 是反例"**：片段层可以承载算术。所以 §2 里"纯数据"的判决只针对 `reference`/`space`/`dirs` 三个模块，不针对 `prim` 的片段正文。

### 3.7 如何自行复核 §3

```bash
# 模板槽位（期望：f32 8 种、i4 9 种，i4 多一个 quant）
for f in gemv_f32 gemv_i4; do echo "== $f"; grep -o '@[a-z_]*(' crates/op/ops/$f.k | sort | uniq -c; done
# 片段行数（期望：9 个片段 62 行 + witness.c 166 = 228）
wc -l crates/prim/cpu/*.c
# 前导的六个 helper（期望：6 条 static inline）
grep -n '^static inline' crates/prof/devices/cpu.toml
# provides / witness / grade（期望：三条都是 14）
grep -c '^\[\[provides\]\]' crates/prof/devices/cpu.toml; grep -c '^witness' crates/prof/devices/cpu.toml; grep -c '^grade' crates/prof/devices/cpu.toml
# grade 分布（期望输出：5 L0 / 9 L2 —— `sort` 按字典序把 L0 排在 L2 之前）
grep -o 'grade *= *"L[0-9]"' crates/prof/devices/cpu.toml | grep -o 'L[0-9]' | sort | uniq -c
#   5 L0
#   9 L2
# Rust 侧零算术（期望：0）
grep -rn '_cutile_\|vmlaq\|vaddq' crates/cutile/src/ | wc -l
# 全仓 unsafe（期望：只有 lib.rs:66 的说明与 :72 的 deny）
grep -rn 'unsafe' crates/cutile/src/
# 生成源的文件名与编译器调用点
sed -n '82,91p' crates/cutile/src/runner.rs
```

## 4. 五设备 × 十六原语的矩阵

### 4.1 口径更正：四处与计划的出入

| # | 计划口径 | 磁盘事实 | 证据 |
|---|---|---|---|
| 1 | "10 原语" | **16**：`Prim` 是闭集，一个不多一个不少 | `crates/prof/src/atom.rs:165`「原语闭集：模板唯一能调用的 16 个指令级操作」；16 个变体在 `:167-182` |
| 2 | "能否跑真 kernel（`cutile_*_kernel`）" | 仓库**没有** `cutile_*_kernel` 这个符号族；真设备符号只有 `cutile_witness`（见证包装）与 `cutile_launch`（bench 启动垫片） | `crates/cutile/src/device.rs:399`（CUDA `__global__`）、`:546`（Metal `kernel`）、`:658`（`newFunctionWithName`）；`crates/cutile/src/harness.rs:295`（`cutile_launch`） |
| 3 | "五设备 × 十原语" | 片段只有 **9 段/设备、共 45 段**；另 7 个原语（`atomic` `mma` `scan` `exp2` `rcp` `shuffle` `broadcast`）**在任何设备上都没有片段文件** | `for d in cpu nvidia intel metal ascend; do ls crates/prim/$d \| grep -vc witness; done` → 五个 9；`docs/prim-layer-review.md:47` 的"45 段片段"同口径 |
| 4 | "`[compile].preamble`" | `flags`/`preamble` 是 `Profile` 的**顶层字段**，`[compile]` 只是 TOML 分组；缺段即空 | `crates/prof/src/prof.rs:260-264`（`peak` `:260`、`flags` `:262`、`preamble` `:264`）；`crates/prof/src/prof/parse.rs:211` `parse_compile`、`:219` `unwrap_or("")` |
| 5 | 本会话早前记录的"intel 11 L1 + 6 L0" | **10 L1 + 7 L0**：`intel/atomic` 是 L0（nvidia/metal 的 `atomic` 是 L1），差的正是这一条 | `crates/prof/devices/intel.toml:141` vs `crates/prof/devices/nvidia.toml:139`、`crates/prof/devices/metal.toml:151` |

矩阵的横轴因此取闭集的全部 16 个原语（9 个有片段 + 7 个无片段），逐格给出五项状态，见 §4.5。

### 4.2 片段层：45 段正文 + 5 份见证 = 1205 行

每格"总行数/有效代码行"（有效代码 = 去注释、去空行；用 `python3` 一行可复算，见 §4.7）。

| 片段 | cpu | nvidia | intel | metal | ascend |
|---|---|---|---|---|---|
| `bar` | 3/1 | 3/1 | 4/1 | 3/1 | 3/1 |
| `commit_wait` | 3/1 | 3/1 | 3/1 | 3/1 | 3/1 |
| `cvt` | 3/1 | 4/1 | 4/1 | 4/1 | 4/1 |
| `dot` | **32/27** | 5/1 | 5/1 | 6/1 | 6/1 |
| `load` | 3/1 | 6/1 | 6/1 | 6/1 | 7/1 |
| `load_async` | 3/1 | 6/1 | 6/1 | 7/1 | 6/1 |
| `quant` | 8/6 | 10/6 | 10/6 | 11/6 | 10/6 |
| `reduce` | 4/1 | 3/1 | 3/1 | 4/1 | 3/1 |
| `store` | 3/1 | 3/1 | 3/1 | 3/1 | 3/1 |
| 片段小计 | 62/40 | 43/14 | 44/14 | 47/14 | 45/14 |
| `witness.*` | 166 | 192 | 191 | 194 | 221 |
| 合计 | **228** | **235** | **235** | **241** | **266** |

一句话：**四家非 CPU 设备每家的 9 段片段合起来都只有 14 行有效代码，其中 6 行是同一份 i4 解包**。设备差异不在这张表里。

**壳的判定（三档）**：

1. **一级壳（1 行有效代码：一句赋值或空操作）**：`bar` / `commit_wait` / `cvt` / `load` / `load_async` / `store` 在五台设备上全部如此，`reduce` 在四家非 CPU 上如此。原文：
   > 「reduce: `$op` along `$axis` —— 单线程下只有一个部分和，`simd_bfly`（`simd_shuffle_xor`）的宽度契约在这里退化。」+ 正文 `($0) = ($1);` —— `crates/prim/metal/reduce.metal:2-4`（`crates/prim/nvidia/reduce.cu:2-3`、`crates/prim/intel/reduce.cl:2-3`、`crates/prim/ascend/reduce.cpp:2-3` 同构）
   
   注意这条 `cutile-sign` 写的是 `simd_bfly`，正文却是纯拷贝——**签名是意向，正文是现状**，两者之差就是 §5 的缺口。
2. **二级（6 行有效代码，五份同一份 nibble 解包）**：`quant` —— `crates/prim/cpu/quant.c:3-8` 与四家外壳逐字同构（连 `0x0Fu` / `−8` 零点都一样，只有注释里的语言名不同）。
3. **唯一带 SIMD 的片段**：`cpu/dot.c` 的 27 行有效代码（总 32 行）—— `#if defined(__aarch64__)`（`:6`）后 4 个 NEON 累加器、16-wide 步进（`:7-29`），标量循环退在 `#else`（`:30-31`）。其余四家的 `dot` 是同一句顺序循环：
   > `{ for (int _k = 0; _k < ($3); ++_k) { ($0) += ($1)[_k] * ($2)[_k]; } }` —— `crates/prim/metal/dot.metal:6`（`crates/prim/nvidia/dot.cu:5`、`crates/prim/intel/dot.cl:5`、`crates/prim/ascend/dot.cpp:6` 逐字同句）

**片段覆盖到的签名 vs 档案声明的签名**（差值 = "有声明、无正文"）：

| 设备 | 片段签名 | 声明条数 | 无片段的声明（`档案:行` 取 `prim` 行） |
|---|---|---|---|
| cpu | 9 条，全部命中 `provides` | 14 | **5**：`atomic/rmw_scalar`（`cpu.toml:281`）、`mma/scalar_tile`（`:294`）、`scan/prefix_loop`（`:307`）、`exp2/pow2_loop`（`:317`）、`rcp/recip_newton`（`:325`）；`shuffle`/`broadcast` **连声明都没有** |
| nvidia | 9 条（`bar/bar.sync` … `reduce/shfl.bfly`） | 17 | **8**：`dot/dp4a`（`:45`）、`atomic/red.global`（`:134`）、`mma/mma.sync`（`:148`）、`shuffle/shfl.idx`（`:161`）、`broadcast`（`:171`）、`scan`（`:181`）、`exp2`（`:191`）、`rcp`（`:200`） |
| intel | 9 条（`bar/barrier` … `reduce/sub_group_reduce`） | 17 | **8**：`dot/dp4a`（`:44`）、`atomic/atomic_add`（`:136`）、`mma/dpas`（`:149`）、`shuffle`（`:162`）、`broadcast`（`:172`）、`scan`（`:181`）、`exp2`（`:191`）、`rcp`（`:200`） |
| metal | 9 条（`bar/simdgroup_barrier` … `reduce/simd_bfly`） | 17 | **8**：`dot/simd_dp4a`（`:67`）、`atomic/atomic_fetch`（`:146`）、`mma/simdgroup_matrix`（`:159`）、`shuffle/simd_shuffle`（`:172`）、`broadcast/simd_broadcast`（`:182`）、`scan/simd_prefix_sum`（`:192`）、`exp2/exp2_fast`（`:202`）、`rcp/rcp_fast`（`:211`） |
| ascend | 9 条，**全是 `scalar_*` 回落** | **20** | **11**：`quant/vcvt_pack`（`:42`）、`dot/mmad`（`:52`）、`load/copy_gm_to_l1`（`:62`）、`load_async/copy_gm_to_l1_async`（`:74`）、`atomic/set_atomic_add`（`:131`）、`mma/mmad_tile`（`:195`）、`shuffle/vgather`（`:208`）、`broadcast/vbroadcast`（`:219`）、`scan/vcumsum`（`:229`）、`exp2/vexp`（`:240`）、`rcp/vrec`（`:250`） |

ascend 的反差最大：声明最多（20 条），片段却只覆盖它自己标注为回落的四条 `scalar_*`（`quant/scalar_dequant` `:147`、`dot/scalar_fma` `:157`、`load/scalar_copy` `:167`、`load_async/scalar_stage` `:179`），真正标榜的 Cube/向量搬运（`mmad`、`vcvt_pack`、`copy_gm_to_l1`）一条正文都没有。

### 4.3 档案层：五份档案逐项

| 档案 | `emit` | `tool` | `[compile].flags` | 前导 | 前导里真实现的原语 | `provides` | grade 分布 | 带见证 | `[compile]` 锚点 |
|---|---|---|---|---|---|---|---|---|---|
| `cpu.toml` | `c`（`:12`） | `cc`（`:13`） | `["-O3","-std=c99"]`（`:41`） | **153 行**（`:42-194`） | **`sum` / `max` / `exp` / `exp4` / `exp_sum` / `exp_norm`**（6 个 `static inline`；`_cutile_sum_f32` `:53` … `_cutile_exp_norm_f32` `:179`） | 14 | **9 × L2 + 5 × L0** | 14/14 | `:40` |
| `nvidia.toml` | `cuda`（`:11`） | `nvcc`（`:12`） | `["-arch=sm_75","-O3"]`（`:41`） | 1 行（`:42`） | 无（只有 `#include <cuda_fp16.h>`） | 17 | **11 × L1 + 6 × L0** | 17/17 | `:40` |
| `intel.toml` | `opencl`（`:10`） | `clang`（`:11`） | `["-cl-std=CL2.0"]`（`:40`） | 空（`:41`） | 无 | 17 | **10 × L1 + 7 × L0** | 17/17 | `:39` |
| `metal.toml` | `metal`（`:20`） | `xcrun metal`（`:21`） | `["-std=metal3.1"]`（`:53`） | **1 行 TOML**（`:54`，值内含 `\n`，展开后 2 行源文本） | 无（`#include <metal_stdlib>` + `using namespace metal;`） | 17 | **11 × L1 + 6 × L0** | 17/17 | `:52` |
| `ascend.toml` | `ascendc`（`:9`） | `ccec`（`:10`） | **无 `[compile]` 段** → 空 | 空 | 无 | **20** | **20 × L0** | 20/20 | 无（`parse.rs:219` 缺项给空串） |

读法上只有三点值得记：

1. **前导只在一个设备上承载原语实现**：`cpu.toml:42-194` 的 153 行（§3.3 已逐函数取证）。其余四家的前导各只有一行 `include`（`metal` 的值内 `\n` 展开成两行源文本），没有一行算术；`intel`（`:41`）与 `ascend`（无 `[compile]`）干脆是空串。
2. **`grade` 的分布就是"协商能不能过"的预判**：只有 `cpu` 有 L2；四家非 CPU 的最高等级是 L1，`ascend` 20 条全 L0。`intel` 比 `nvidia`/`metal` 少一条 L1——差在 `intel/atomic` 是 L0（`intel.toml:141`）。等级含义见 `crates/prof/src/atom.rs:529-531`（`is_verified` 仅在 L2/L3 为真）。
3. **见证是全覆盖的**：85 条 `provides` 全部带 `witness = { file, func, expect }`（`grep -c '^\[\[provides\]\]'` 与 `grep -c '^witness'` 在五份档案里逐一相等）。所以"能不能跑"的差异**不在证据层**，而在 §4.4 的三关。

### 4.4 "能否跑真 kernel"的判据（三关，短路）

一个（设备，原语）格记为"能跑真 kernel"，当且仅当下列三关**依次**通过：

| 关 | 判据 | 失败即 | 证据 |
|---|---|---|---|
| ① 通道存在 | `Lane::ready(profile)` 为一个 `emit` 返回执行通道（`Host` 或 `Cuda`），而非错误 | `✗{L}` | `crates/cutile/src/verify.rs:354-363`（`Lane` 枚举 `:345-350`；`Emit::C => Host` `:356`；`Emit::Cuda` `:357-359`；`_ =>` 报错 `:361`） |
| ② 协商可过 | 该原语在档案里有 `grade.is_verified()`（L2/L3）的声明，且本次执行有见证（`Proof`） | `✗{G}` | `crates/cutile/src/verify.rs:317` `verify_one` → `:326` `.negotiate()`；`crates/prof/src/atom.rs:529-531`；拒绝语是格式串「{grade} < L2（未验证视为不具备）」（`crates/cutile/src/cap.rs:477`），L1 声明渲染成 `L1 < L2（未验证视为不具备）` |
| ③ 片段与槽位齐备 | 模板里存在调用该原语的槽位，且该设备的片段正文能拼进去（否则 `missing_fragment`） | `✗{F}` | `crates/cutile/src/substitute.rs:24`「片段层缺少 `{}/{}.{}`：该设备尚未实现 `{prim}`」；槽位全集见 §4.5 表注 |

第③关的槽位证据是一条命令：22 个模板里出现的槽位只有 10 种——`@load` 27、`@store` 24、`@prim` 22、`@reduce` 9、`@dot` 8、`@load_async` 5、`@cvt` 5、`@commit_wait` 3、`@bar` 3、`@quant` 2；`@mma` / `@shuffle` / `@broadcast` / `@scan` / `@exp2` / `@rcp` / `@atomic` **零次**。因此那 7 个原语在任何设备上都过不了第③关——它们缺的不是某台设备的实现，而是**模板层的调用方**。

**这一列是静态判读**：所有 `✓`/`✗` 都来自源码分支与档案数据，本机没有跑 `verify`/`bench`（无 GPU，且本节不产出性能结论）。

### 4.5 矩阵：16 行 × 5 设备

单元格 = `片段总行数/有效代码行 │ 见证 │ 前导实现 │ grade │ 真 kernel`。

图例：`✗{L}` 第一关失败（无该 `emit` 的执行通道）；`✗{G}` 第二关失败（无 ≥L2 声明，协商必然拒绝）；`✗{F}` 第三关失败（无片段 / 模板零槽位）；`∅` 表示档案里没有该原语的声明。

| 原语 | cpu | nvidia | intel | metal | ascend |
|---|---|---|---|---|---|
| `bar` | 3/1 │ ✓ │ — │ L2 │ **✓** | 3/1 │ ✓ │ — │ L1 │ ✗{G} | 4/1 │ ✓ │ — │ L1 │ ✗{L,G} | 3/1 │ ✓ │ — │ L1 │ ✗{L,G} | 3/1 │ ✓ │ — │ L0 │ ✗{L,G} |
| `commit_wait` | 3/1 │ ✓ │ — │ L2 │ **✓** | 3/1 │ ✓ │ — │ L1 │ ✗{G} | 3/1 │ ✓ │ — │ L1 │ ✗{L,G} | 3/1 │ ✓ │ — │ L1 │ ✗{L,G} | 3/1 │ ✓ │ — │ L0 │ ✗{L,G} |
| `cvt` | 3/1 │ ✓ │ — │ L2 │ **✓** | 4/1 │ ✓ │ — │ L1 │ ✗{G} | 4/1 │ ✓ │ — │ L1 │ ✗{L,G} | 4/1 │ ✓ │ — │ L1 │ ✗{L,G} | 4/1 │ ✓ │ — │ L0 │ ✗{L,G} |
| `dot` | **32/27** │ ✓ │ — │ L2 │ **✓** | 5/1 │ ✓ │ — │ L1×2 │ ✗{G} | 5/1 │ ✓ │ — │ L1×2 │ ✗{L,G} | 6/1 │ ✓ │ — │ L1×2 │ ✗{L,G} | 6/1 │ ✓ │ — │ L0×2 │ ✗{L,G} |
| `load` | 3/1 │ ✓ │ — │ L2 │ **✓** | 6/1 │ ✓ │ — │ L1 │ ✗{G} | 6/1 │ ✓ │ — │ L1 │ ✗{L,G} | 6/1 │ ✓ │ — │ L1 │ ✗{L,G} | 7/1 │ ✓ │ — │ L0×2 │ ✗{L,G} |
| `load_async` | 3/1 │ ✓ │ — │ L2 │ **✓** | 6/1 │ ✓ │ — │ L1 │ ✗{G} | 6/1 │ ✓ │ — │ L1 │ ✗{L,G} | 7/1 │ ✓ │ — │ L1 │ ✗{L,G} | 6/1 │ ✓ │ — │ L0×2 │ ✗{L,G} |
| `quant` | 8/6 │ ✓ │ — │ L2 │ **✓** | 10/6 │ ✓ │ — │ L1 │ ✗{G} | 10/6 │ ✓ │ — │ L1 │ ✗{L,G} | 11/6 │ ✓ │ — │ L1 │ ✗{L,G} | 10/6 │ ✓ │ — │ L0×2 │ ✗{L,G} |
| `reduce` | 4/1 │ ✓ │ **✓** │ L2 │ **✓** | 3/1 │ ✓ │ — │ L1 │ ✗{G} | 3/1 │ ✓ │ — │ L1 │ ✗{L,G} | 4/1 │ ✓ │ — │ L1 │ ✗{L,G} | 3/1 │ ✓ │ — │ L0 │ ✗{L,G} |
| `store` | 3/1 │ ✓ │ — │ L2 │ **✓** | 3/1 │ ✓ │ — │ L1 │ ✗{G} | 3/1 │ ✓ │ — │ L1 │ ✗{L,G} | 3/1 │ ✓ │ — │ L1 │ ✗{L,G} | 3/1 │ ✓ │ — │ L0 │ ✗{L,G} |
| `atomic` | — │ ✓ │ — │ L0 │ ✗{F} | — │ ✓ │ — │ L1 │ ✗{G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L1 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} |
| `mma` | — │ ✓ │ — │ L0 │ ✗{F} | — │ ✓ │ — │ L0 │ ✗{G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} |
| `scan` | — │ ✓ │ — │ L0 │ ✗{F} | — │ ✓ │ — │ L0 │ ✗{G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} |
| `exp2` | — │ ✓ │ — │ L0 │ ✗{F} | — │ ✓ │ — │ L0 │ ✗{G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} |
| `rcp` | — │ ✓ │ — │ L0 │ ✗{F} | — │ ✓ │ — │ L0 │ ✗{G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} |
| `shuffle` | — │ ✗ │ — │ ∅ │ ✗{F} | — │ ✓ │ — │ L0 │ ✗{G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} |
| `broadcast` | — │ ✗ │ — │ ∅ │ ✗{F} | — │ ✓ │ — │ L0 │ ✗{G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} | — │ ✓ │ — │ L0 │ ✗{L,G,F} |

表注：

- **"见证"列的 `✓`** 指该格有 `witness = { file, func, expect }` 且对应见证函数存在于该设备的 `witness.*`（85/85，见 §4.3）。`cpu` 的 `shuffle` / `broadcast` 是 `✗`，因为档案里根本没有这两条声明（`grep -c 'prim *= *"shuffle"' crates/prof/devices/cpu.toml` → 0）。
- **"前导实现"列的 `✓`** 只出现在 `cpu/reduce`：片段 `cpu/reduce.c:4` 只有一次 `$func(...)` 调用，真正的折叠在 `cpu.toml:53`（`_cutile_sum_f32`）与 `:85`（`_cutile_max_f32`）。其余四家的 `reduce` 片段是纯拷贝，前导里也没有对应 helper —— 它们是**两头都空**，不是"在片段里实现"。
- **`L1×2` / `L0×2`** 表示该原语在该档案里有两条声明（如 `dot/dp4a` + `dot/fma`；`ascend` 的 `quant` 是 `vcvt_pack` + `scalar_dequant`，两条都是 L0）。
- 所有格都是读出来的，没有推断格；唯一的"推断"是把第②关的"协商必然拒绝"由 `atom.rs:529-531` 的语义推出，不是运行观测——本节不跑 `verify`。

### 4.6 读矩阵得到的四条结论

1. **"能否跑真 kernel"这一列不是性能结论，是通路结论**：80 格里只有 `cpu` 的 9 格是 `✓`，其余 71 格全部在①②③关短路。把 `✗` 读成"没测"会误解——`nvidia` 的 `✗{G}` 是"机制在（`Lane::Cuda` `verify.rs:357`）、声明不到位（全 L1）"，而 `metal`/`intel`/`ascend` 的 `✗{L}` 是"连分支都没有"（`verify.rs:361`）。
2. **四台加速设备的片段语义实质同构**：14 行有效代码里 6 行是同一份 i4 解包、其余 8 行是同一句 `($0)=($1)` 与同一句顺序 `dot`。也就是说**新增一条加速链路的实质工作量不在片段**——片段是壳，真正要做的是前导里的原语实现（§5.2）与 `provides` 的等级/签名（§4.3）。
3. **`cpu` 是唯一"前导承载原语实现"的设备**，`reduce` 走 `$func` 槽位把实现外包给前导；这是全仓唯一一处"片段只做调用、实现在数据里"的样板，也是 §5 给 Metal 排补法的参照。
4. **`ascend` 是"声明最富、正文最穷"的极端**：20 条声明（五份里最多）全 L0，片段只覆盖四条自己标注为 `scalar_*` 的回落。它证明了一件事——**声明条数与能力无关**，`grade` 才是可判定的量。

### 4.7 如何自行复核 §4

```bash
# 片段行数（期望：五列与 §4.2 表一致；最后一列是见证文件）
for d in cpu nvidia intel metal ascend; do printf '%-8s ' $d; wc -l crates/prim/$d/* | tail -1; done
# 有效代码行（去掉注释与空行；期望：非 cpu 的 dot 全为 1，cpu 的 dot 为 27、reduce 为 1）
python3 - <<'PY'
import glob,re
for f in sorted(glob.glob('crates/prim/*/*')):
    if 'witness' in f: continue
    txt=open(f).read()
    txt=re.sub(r'/\*.*?\*/','',txt,flags=re.S)
    print(f, sum(1 for l in txt.splitlines() if l.strip()))
PY
# 片段签名（期望：五设备各 9 条，且 cpu 的是 scalar_* / metal 的是 simd_*）
for d in cpu nvidia intel metal ascend; do echo "== $d"; grep -h 'cutile-sign' crates/prim/$d/* | sed 's#/\* cutile-sign: ##; s# \*/##'; done
# 声明条数与等级分布（期望：14 / 17 / 17 / 17 / 20；等级如 §4.3）
for f in cpu nvidia intel metal ascend; do printf '%-8s ' $f; grep -c '^\[\[provides\]\]' crates/prof/devices/$f.toml | tr '\n' ' '; grep -o 'grade *= *"L[0-9]"' crates/prof/devices/$f.toml | grep -o 'L[0-9]' | sort | uniq -c | tr '\n' ' '; echo; done
# 前导规模（期望：cpu 6 个 static inline，其余四家 0；ascend 无 [compile]）
for f in cpu nvidia intel metal ascend; do printf '%-8s ' $f; grep -c 'static inline' crates/prof/devices/$f.toml; done
# 模板槽位全集（期望：只有 10 种，mma/shuffle/broadcast/scan/exp2/rcp/atomic 为 0）
grep -ho '@[a-z_]*' crates/op/ops/*.k | sort | uniq -c | sort -rn
# 三关的源码位置
sed -n '344,364p' crates/cutile/src/verify.rs
sed -n '165,182p;526,534p' crates/prof/src/atom.rs
```

## 5. 缺口结论与后续排序

### 5.1 Metal 的缺口要分三类，不是两类

| 类 | 内容 | 现状 | 判据 |
|---|---|---|---|
| **数据缺口 A：片段正文** | `crates/prim/metal/*.metal` 的 9 段 | 47/14 行；`reduce.metal:4` 是 `($0) = ($1)`、`dot.metal:6` 是顺序循环（§4.2） | 片段正文里出现 `simd_*` 且签名与 `provides` 的 L1 条目同名 |
| **数据缺口 B：前导原语实现** | `metal.toml:54` 的 `[compile].preamble` | **1 行 TOML、2 行源文本**（`#include <metal_stdlib>` + `using namespace metal;`），零个 helper | `grep -c 'static inline' metal.toml` 从 0 变正 |
| **机制缺口：算子内核执行通道** | `verify.rs` 的 `Lane`；`harness.rs` 的方言 | `Lane` 只有 `Host`/`Cuda`（`:345-350`），`Emit::Metal` 落到 `_ =>` 报错（`:361`）；`harness.rs` 无 Metal 方言 | 一条 `Lane::Metal` + `metal_harness`，且 `verify --device m5` 不再返回 `unsupported_emit` |
| **已成立的机制：见证执行** | `device.rs` 的 `metal_*` 一族（`grep -c 'fn metal_'` = 10）；设备句柄在 `crates/cutile/src/device.rs:646`（`id<MTLDevice> device = MTLCreateSystemDefaultDevice();`） | 编译（`metal_program` `:504`）、发射（`metal_run` `:524` + `cutile_witness` 包装 `:546`）、回读（`metal_readback` `:618`）**都在** | `crates/prof/devices/metal.toml:1`「见证已经在本机 GPU 上执行过」+ `:5-6`「`m5` 的 17 条见证走 `xcrun metal` → `metallib` → GPU，拿到的是 `Scope::Device` 的执行结论」 |

所以"Metal 只有见证"**不等于**"Metal 没有机制"。已经成立的是"把 `witness.metal` 的某个函数编译、发射、读回"这条链；缺的是把同一条链指向**算子内核**（而不是见证集合），以及一条能计时的 bench 方言。

### 5.2 缺口 A/B：缺什么、补在哪个文件

**落点一：`crates/prof/devices/metal.toml` 的 `[compile].preamble`（当前 1 行 TOML、2 行源文本，`:54`）**——这是"数据缺口 B"的主战场。要补的是**原语级实现**，与 CPU 的六个 helper 同构（`cpu.toml:53`/`:85`/`:121`/`:136`/`:157`/`:179`）：

| 要补的 helper | 对应原语 / 片段 | 为什么必须是前导而不是片段 | 目标判据 |
|---|---|---|---|
| `simd_bfly` 归约（`simd_shuffle_xor` 宽度契约 + 尾段） | `reduce`（`metal.toml:137`，L1） | 片段只有 `($0) = ($1)`（`reduce.metal:4`），蝶形的宽度与 lane 掩码是"契约"，CPU 也是这样把实现放进前导并让片段只做一次调用（`cpu/reduce.c:4`） | `grep -c 'simd_shuffle_xor' metal.toml` ≥ 1 |
| `simd_exp` / `simd_exp4`（`metal::exp` 或多项式） | `exp2`（`:202`）、模板里的 `@reduce` 前处理 | CPU 的 `_cutile_exp_f32`（`cpu.toml:121`）在片段层没有对应物；Metal 侧同理 | 前导出现 `exp` 且 `exp2` 声明可升级（见 P1-2） |
| `simd_recip` | `rcp`（`:211`） | 与 `exp` 同源，`rcp_fast` 需要一个可拼的入口 | `grep -c 'rsqrt\|recip' metal.toml` ≥ 1 |
| `simdgroup_matrix` 包装（mma） | `mma`（`:159`） | 片段层零槽位（`@mma` 在 22 个模板里 0 次），当前任何落点都无法被调用 | 与 P2 绑定，本项**只需 helper 就位**，不承诺可用 |
| `device_load` / `device_store` 的向量化版本 | `load`（`:85`）/`store`（`:114`） | 片段是 1 行赋值，向量宽度只能在前导决定 | `grep -c 'simd' metal.toml` ≥ 1 |

**落点二：`crates/prim/metal/*.metal` 的正文**——"数据缺口 A"：

- `reduce.metal`（4 行，正文 `($0) = ($1)`，`:4`）：`cutile-sign` 已写 `reduce/simd_bfly`，正文要与之对齐。
- `dot.metal`（6 行，`:6` 顺序循环）：档案里 `dot` 有两条声明——`simd_fma`（`:57`，片段同名）与 `simd_dp4a`（`:67`，**无片段**）。所以"dot 的缺口"是**两条签名只覆盖了一条**。
- `quant.metal`（11/6 行，五份里唯一带算术的片段）：与 `dot/simd_dp4a` 配对才能凑出 i4 GEMV 的真实通路。
- 另外 5 个零片段原语（`atomic` `mma` `scan` `exp2` `rcp`）与 2 个零槽位原语（`shuffle` `broadcast`）：**本轮不必补**，理由与 `cpu` 同（§4.3 的 L0 五条）——真正卡住它们的是模板层没有 `@mma`/`@shuffle`/… 槽位，补片段不会有任何调用方。

### 5.3 机制缺口：两处，都有确切落点

1. **`Lane` 少一条分支**：`crates/cutile/src/verify.rs:345-350` 的 `Lane` 只有 `Host`（`:347`）与 `Cuda`（`:349`）；`Lane::ready` 的 `_ =>`（`:361`）对 `Emit::Metal` 直接返回错误，文案是 `verify.rs:522-523`：
   > 「设备在本机（`emit = {}`），内核执行路径尚未接通（CUDA 已接，其余仍只有见证）」（`:522`）+「见证已经在这台设备上跑过，内核文本还没有对应的编译/回读路径」（`:523`，Rust 字符串以 `\` 续行）
   
   要接通，`device.rs` 的三段可**直接复用**：`metal_program`（`:504`，编译 MSL）、`metal_driver`（`:627`，生成驱动可执行文件）、`metal_readback`（`:618`，读回）。它们现在只服务"见证集合"，参数写死在 `MSL_PARAMS`（`:497`）与 `cutile_witness`（`:546`）的包装里，因此不是"另写一遍"，而是"把同一段编排的参数换成算子内核"。
2. **`harness.rs` 少一个方言**：`crates/cutile/src/harness.rs` 只有 `harness`（C，`:14`）与 `cuda_harness`（`:239`）两个方言，`grep -c 'fn metal' crates/cutile/src/harness.rs` → **0**。bench 要靠方言把 kernel 与样本拼成可计时入口（CUDA 侧是 `cutile_launch`，`:295`；行协议 `runner.rs:174`），没有方言 ⇒ `bench --device m5` 无路径。

### 5.4 与 `docs/prim-layer-review.md` 的对账

**一致、可直接引用**："45 段片段"与"5 个 vendor（`cpu`/`nvidia`/`intel`/`ascend`/`metal`）× 9 段片段 + 1 份 `witness.<ext>`"（`:47`）、量化片段五家同构（`:150`）、L0 五条不补片段（`:283`）。

**需以代码更正的两处**（本节的表格以代码为准）：

1. `docs/prim-layer-review.md:50` 说「全部标量语句块：`dot` 是顺序标量循环 `$0 += $1[k]*$2[k]`」——对 `crates/prim/cpu/dot.c` 不成立：`:6-29` 是 4 累加器 NEON 16-wide，顺序循环只在 `:30-31` 的 `#else` 分支。
2. 同文件 `:150` 说「`dot` 在 cpu/nvidia/intel/ascend/metal 五段正文逐字同构」——实际是**四段逐字同构 + cpu 分架构**：`wc -l crates/prim/cpu/dot.c` = 32，其余四家 5–6 行。

**不矛盾但需补一句口径**：同文件 `:54` 的「只有 Metal 有构建配方」指 `device-image` 的**镜像构建**（`crates/prim/build.rs`），与"算子内核能否执行"是两件事——Metal 镜像能建，不代表 `metal.toml` 的前导够用，更不代表 `Lane` 接通。把这两条混起来，就会得出"Metal 机制齐全，只差数据"或反过来"Metal 机制缺失"的错误结论。

### 5.5 后续排序（每条带判据）

前置理由：**数据先于机制**。`Lane::Metal` 一旦接上，`verify` 立刻会去读 `preamble` 与片段；若前导仍只有那一行 `include`，接通的结果只是"能跑但算错"或"跑出无加速的拷贝"，反而制造假象。而"数据"这一侧有现成的验收手段（`wc`/`grep`/签名对齐），可以独立判定、零风险。

| 序 | 动作 | 落点文件 | 判据（可观测） |
|---|---|---|---|
| **P0-1** | 补 `reduce`/`dot`/`quant` 三段正文，使 `cutile-sign` 与 `provides` 的 L1 条目同名 | `crates/prim/metal/{reduce,dot,quant}.metal` | `grep -h 'cutile-sign' crates/prim/metal/*` 的 9 条与 `metal.toml` 的 11 条 L1 交叉对齐；`reduce.metal` 正文 > 4 行且含 `simd_shuffle_xor` |
| **P0-2** | 补前导的原语级 helper（`simd_bfly` / `simd_exp` / `simd_recip` / mma 包装 / 向量搬运） | `crates/prof/devices/metal.toml` `[compile].preamble`（`:54`） | `grep -c 'static inline' metal.toml` ≥ 5；`grep -c 'simd_' metal.toml` ≥ 5。**不要求**该轮升 grade |
| **P1-1** | 片段改走 `$func` 槽位（真实现放前导，正文只做一次调用），与 `cpu/reduce.c:4` 同构 | 同上两处 + `crates/prim/metal/reduce.metal` | `reduce.metal` 正文能通过 `substitute` 的槽位解析（`crates/cutile/src/substitute.rs:37` `render`）且 `grep -c '\$func' crates/prim/metal/reduce.metal` = 1 |
| **P1-2** | 把 `reduce`/`dot`/`quant` 至少一条声明升到 L2 | `crates/prof/devices/metal.toml`（`:137`/`:57`/`:76`） | `grep -o 'grade *= *"L[0-9]"' crates/prof/devices/metal.toml \| grep -c L2` ≥ 1；且该条 `witness` 的 `expect` 与片段的真实输出一致（`metal.toml:143`） |
| **P2-1** | 接通执行通道：`Lane::Metal` + Metal 方言 | `crates/cutile/src/verify.rs`（`:345-364`）、`crates/cutile/src/harness.rs`、复用 `device.rs:504/618/627` | `verify --device m5` 从 `unsupported_emit`（`:514`）变为逐形状对拍；失败判据反向成立：`verify.rs:504-509` 的 `max_rel_err ≤ tolerance` 全部通过 |
| **P2-2** | 最后才谈性能 | `crates/cutile/src/bench.rs`、`runner.rs:31/37/53` | `bench --device m5` 输出 `min_ns` / `p50_ns` / `p95_ns`，并与 `metal.toml:45-50` 的 `[peak]`（`8.192e12` flop/s、`307.2e9` B/s）对照 roofline |

一句话：**P0 是纯数据改动（不碰 `crates/cutile` 一行），P1 才动等级，P2 才动机制。** 这与 §2 的判决一致：设备差异本应在 `prim`/`prof`，`cutile` 只提供"按 `Emit` 走一条 `Spec`"的编排（`device.rs:44-63` 的 `SPECS` 三行）。

### 5.6 如何自行复核 §5

```bash
# Metal 的前导只有 1 行 TOML（值内 \n）、零 helper（期望：1 行 / 0）
sed -n '52,55p' crates/prof/devices/metal.toml
grep -c 'static inline' crates/prof/devices/metal.toml
# Metal 片段是壳（期望：reduce 正文 1 行赋值、dot 正文 1 行顺序循环）
sed -n '4p' crates/prim/metal/reduce.metal
sed -n '6p' crates/prim/metal/dot.metal
# 机制只有两条通道、没有 Metal 方言（期望：Lane 两变体；metal 命中 0）
sed -n '344,364p' crates/cutile/src/verify.rs
grep -c 'fn metal' crates/cutile/src/harness.rs
# 见证执行机制确实在（期望：十个 metal_* 函数）
grep -n 'fn metal_' crates/cutile/src/device.rs
# 可复用的三段（编译/驱动/回读）
sed -n '504p;618p;627p' crates/cutile/src/device.rs
```

## 6. 索引与边界裁决落点

- 本文接入三处既有索引：`crates/cutile/README.md` §3.1 模块地图附近、`docs/ARCHITECTURE.md` 序言正文、根 `README.md` `## 文档` 表格。
- 边界裁决（允许改哪、什么算越界、对 roadmap §5/P3 边界 crate 的偏离记录）见 [ADR 0028](adr/0028-device-data-vs-mechanism-boundary.md)。
- 本轮为分析 + ADR，`crates/**` 零代码改动；变更记录见 `CHANGELOG.md` Unreleased 文档条目。

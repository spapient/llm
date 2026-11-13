# 全仓架构与质量审计报告

- 审计对象：`/Users/fei/code/llm`（Rust workspace，`crates/*` 26 个成员 + 根包 `llm`）。首轮审计时为 32 个成员，七条合并已在第十一节执行
- 审计方式：只读静态审计 + 在当前工作树上真实执行测试与门禁；未修改任何代码
- 审计时点：2026-11-13 07:30 前后的工作树（含大量未提交改动）
- 口径声明：
  - "已确认" = 本次直接读源码/执行命令得到，可回链到文件行；
  - "推断" = 由已确认事实按语言/平台规则推出，未在目标环境实测；
  - "未知" = 本次没有证据，明确标出而不猜测。

---

## 零、读者须知：版本标记与阅读顺序

本报告有两层时间：**第一至十节是首轮只读审计的现场记录**（成员 33 = 32 个 `crates/*` + 根包），**第十一节是随后按指令执行的七条吸收链的记录**（成员 26 = 25 + 根包）。为避免"同一句话两个数"，按四条规则读：

1. **规模数字以 §2.1 与 §11.7 为准**：两处都按当前工作树重测（生产 38 530 行 / 170 个源文件 / 1 037 用例 / 58 个测试目标），首轮值只作对照保留。
2. **成员表以 §4.1 与 §6.0 为准**：两张表都已按合并后的 26 个 crate 重列；被吸收的七个成员在 §6.0 末尾单列"已消失成员"，追溯不丢。
3. **判断文字保留首轮措辞**：凡被 §11 改判的条目都带「§11 改判」或「已并入」标记；没有标记的，就是仍然成立的结论。
4. **路径换算**：第一至十节的 `文件:行` 回链保留首轮路径（便于与当时的审计记录逐条对照）。读到下列旧路径时按下表换算：

| 旧路径（首轮） | 当前路径 | 证据 |
|---|---|---|
| `crates/audit/src/*` | `crates/cutile/src/report/*` | `crate::witness` 等 20 处路径已改写 |
| `crates/checkpoint/src/lib.rs` | `crates/train/src/snapshot/mod.rs` | 1 609 行 + 2 行文件头 = 1 611，其余原样 |
| `crates/regress/src/lib.rs` | `crates/tool/src/regress.rs` | `Judgement` 与 `Gate` 现在同 crate |
| `crates/exec/src/lib.rs` | `crates/engine/src/exec.rs` | `Stream` 的唯一生产者是 `engine` |
| `crates/kv/src/lib.rs` | `crates/engine/src/kv.rs` | 页账与页数公式现在同一文件树 |
| `crates/einsum/src/*` | `crates/op/src/einsum/*` | 5 条债务键已随路径改写 |
| `crates/proto/src/*` | `crates/serve/src/proto/*` | 4 个文件 1 126 行随迁，`pub` 路径由 `proto::` 变为 `serve::proto::` |

两条例外：`crates/audit/src/witness/` 变成 `crates/cutile/src/report/witness/`；`crates/checkpoint/src/lib.rs` 里的 `checkpoint::` 错误位标签改成了 `train::snapshot::` 前缀（`CONTEXT` 常量）。

---

## 一、总体判断与评级

**一句话定位**：这是一套"架构契约与依赖治理已经达到工业级、但运行时可交付面仍处在 P0 垂直切片"的框架——它的工程纪律（可执行架构门、单一事实源、零运行时依赖、单一 unsafe 岛）明显高于其功能完成度（仅一条 C 后端、无传输层、无集合通信后端、无 BPE）。

| 维度 | 评级 | 分数 | 核心理由（可回链） |
|---|---|---:|---|
| 架构契约与依赖治理 | A | 95 | 分层规则由 `tests/architecture.rs` 对 `cargo metadata` 强制执行，6 项断言全绿；所有权表逐类型落到 crate |
| 依赖最小化 | A+ | 99 | 运行时第三方依赖 **0**；`Cargo.lock` 99 个包中 26 个是本仓成员（25 crate + 根包），其余 73 个全部来自 `criterion` 的 dev 子树 |
| unsafe 治理 | A | 96 | 全仓 unsafe 关键字只出现在 `crates/storage/src/mmap.rs`（1 个 `unsafe extern`、2 个 `unsafe impl`、3 个 `unsafe` 块），模块级 `#![allow(unsafe_code)]` + feature 门 + 架构说明 |
| 正确性验证 | B+ | 85 | 58 个测试目标 / 1 037 个用例全绿；有 gradcheck、真实权重对拍、双源语义；但负例集中在少数条目，部分格式测试可空转 |
| 复杂度与可维护性 | C+ | 65 | 自建门禁登记 **282 条存量超标**；最高 `ccn=77`；8 个文件超 900 行；`impl` 与分支集中在解析器 |
| 文档与代码一致性 | B | 78 | `ARCHITECTURE.md` 是"可执行文档"；README 的成员表与依赖图已随 §11 同步为 26 成员（原写已删除的 `pack`/`comm` 与 33 成员）；仍存：`ARCHITECTURE.md:81` 写"现有两条算子"，实际 22 条注册 |
| 运行时可交付面 | C | 60 | `engine::exec`/`shard`/`serve`/`tokenizer` 只有契约无实现；`engine` 只做页账不做批次张量；仅 C/cc 一个可执行后端 |
| **综合** | **A-** | **82** | 治理能力已经是"顶级开源框架的方法论水平"，交付面是"严谨的 P0 切片" |

关键事实：本报告发现的 P0 级问题**不是架构问题，而是"不可信输入"路径上的 4 处 panic / 大分配**——这恰好说明架构层守住了，鲁棒性层还没守。

---

## 二、关键事实基线表

### 2.1 规模与结构

| 指标 | 数值 | 证据 / 口径 | 等级 |
|---|---:|---|---|
| workspace 成员 | 26（25 `crates/*` + 根包 `llm`） | 根 `Cargo.toml:48` `members = ["crates/*"]`；`crates/*/Cargo.toml` = 25。首轮为 33（32 + 根包），差 7 见 §11 | 已确认 |
| 生产源文件 | 170 | 166 `crates/*/src/*.rs` + 2 根 `src/*.rs` + 1 `crates/prim/build.rs` + 1 `crates/weights/examples/dump_bin.rs`；**另有** `crates/benches/criterion/span_parse.rs` 30 行（criterion 基准，不计入生产） | 已确认 |
| 外部集成测试文件 | 8 | `tests/*.rs` 2 + 各 crate `tests/*.rs` 6 | 已确认 |
| 生产代码行（含内联测试） | 55 535 | `crates/*/src` 53 329 + `prim/build.rs` 622 + 根入口 1 544 + `dump_bin.rs` 40；基准 harness 30 行另计 | 已确认 |
| 生产代码行（不含 `#[cfg(test)]`） | 38 530 | 逐文件在首个 `#[cfg(test)]` 处截断求和（`crates/*/src` 36 673 + 根入口 1 195 + `build.rs` 622 + 40）。首轮同法测得 38 481；上一版曾误报 39 721，成因见 §11.0 | 已确认 |
| 测试代码行 | 19 163 | 内联 16 656 + 外部 `tests/` 2 507（8 个文件）。上一版写 17 005 与 `53 325 − 36 669 = 16 656` 自相矛盾，已按后者改正 | 已确认 |
| 测试/生产比 | 0.50 | 19 163 / 38 530 | 已确认 |
| `trait` 定义 | 31 | `trait [A-Z]` 全仓计数；其中 `pub trait` 27 | 已确认 |
| `impl` 块 | 363（其中 `impl X for Y` 140） | `^\s*impl[<( ]` 计数 | 已确认 |
| `const` 泛型参数 | 28 | `<const ` 计数（`Set<E, WORDS>`、`Quantized<GRAIN>`、`Dense<BLOCK>`、`SymI8<GRAIN>`、`key_gen_internal<K,L>` 等） | 已确认 |
| 泛型形参出现（粗计） | 897 | `<A>`/`<A, B>` 形态正则；含大量 `Vec<T>` 噪声，**仅作量级参考** | 推断 |
| `macro_rules!` 定义 | 2 | `prof/src/atom.rs:598` `constraint_builders!`、`atom.rs:777` `inherent_names!` | 已确认 |
| `#[test]` 属性 | 975 | 全仓计数 | 已确认 |
| 实际执行用例 | 1 037 通过 / 0 失败 / 1 忽略 | `cargo test --workspace --all-features`，**58** 个测试目标，退出码 0（首轮 72 个目标，差 14 见 §11.7） | 已确认 |
| unsafe 关键字（代码） | 6 | 全部在 `crates/storage/src/mmap.rs:25,48,49,74,108,136` | 已确认 |
| 第三方运行时依赖 | 0 | 各 `Cargo.toml` 的 `[dependencies]` 只有 `path` 成员 | 已确认 |
| 第三方 dev 依赖 | 1（`criterion 0.8.2`） | 根 `Cargo.toml:83`；`crates/benches/Cargo.toml` 的 `[dev-dependencies]` | 已确认 |
| `Cargo.lock` 包数 | 100 = 27 成员 + 73 criterion 子树 | `grep -c '^name = ' Cargo.lock` = 100 | 已确认 |
| 数据文件（`.k/.toml/.cu/.metal/.c/.cl/.h`） | 89 | 排除 `Cargo.toml`；其中 `ops/` 22 对 `.toml`+`.k` | 已确认 |
| 已注册算子 | 22 | `crates/op/src/lib.rs:108` `REGISTRY: [Registration; 22]` | 已确认 |
| ADR 数量 | 19 | `docs/adr/0001..0019`（0019 记录七条吸收链） | 已确认 |
| 文档总量 | `ARCHITECTURE.md` 236 行、`lld.md` 2 410 行、`lints/vocabulary.toml` 1 229 行 | `wc -l` | 已确认 |

### 2.2 自建质量门基线（`lints/vocabulary.toml` `[gate]`）

| 指标 | 数值 | 证据 | 等级 |
|---|---:|---|---|
| `ccn_max` | 5 | `lints/vocabulary.toml:71` | 已确认 |
| `nloc_max` | 60 | `:72` | 已确认 |
| `simd_nloc` | 96 | `:73` | 已确认 |
| `file_max_lines` | 899 | `:74` | 已确认 |
| 存量债条目 | **282** | `[[gate.debt]]` 计数 | 已确认 |
| 其中 ccn 超标 | 251 | 同上 | 已确认 |
| 其中 nloc 超标 | 28 | 同上 | 已确认 |
| 其中禁调用形态 | 24 | 同上 | 已确认 |
| 其中文件行数 | 8 | 同上 | 已确认 |
| 按 crate 分布 | weights 84、cutile 48、op 35、tool 21、prof 21、config 16、annotation 12、train 11、autograd 7、serve 2、其余各 ≤3（合并后口径；首轮为 weights 84、op 30、cutile 28、prof 21、audit 20、tool 20、config 16、annotation 12、checkpoint 8、autograd 7、einsum 5） | 同上 | 已确认 |
| 最差 `ccn` | 77（`weights/src/bin.rs:execute`） | `:922` | 已确认 |
| 最长函数 | 233 行（`annotation/src/codegen.rs:render_model`） | 登记值 | 已确认 |
| 最大文件 | `prof/src/prof.rs` 1 712 行；次大 `train/src/snapshot/mod.rs` 1 611 行 | `wc -l` | 已确认 |
| 门禁是否当前为绿 | 是 | `cargo test -p llm-tool --test gate` 1/1 通过 | 已确认 |
| 该门禁是否在 HEAD 中存在 | **否** | `git show HEAD:lints/vocabulary.toml` 中 `[gate]` 出现 0 次；`crates/tool/src/{ccn,call,dup,gate,lex,strip}.rs` 与 `crates/tool/tests/` 均为未跟踪文件 | 已确认 |

口径说明（避免把算术读错）：282 是 `[[gate.debt]]` **条目数**，不是度量数。一条条目可以同时登记 `ccn` 与 `nloc`，所以四个子计数之和（251 + 28 + 24 + 8 = 311）**大于**条目数；311 − 282 = 29 条条目带两个度量，条目身份（`路径:函数#序号`）仍然是 282 个。

结论（事实）：这 282 条**不是历史积累的债务，而是刚安装的严格门禁对既有代码的首次快照**。这是"把标准变成数据"的正确动作，但同时也把"当前实现离自定标准有多远"首次量化暴露出来。

### 2.3 工作树状态（区分在途改动）

| 项 | 数值 | 证据 | 等级 |
|---|---|---|---|
| 跟踪文件改动 | 93 个文件，+3 545 / −10 272（含 29 个删除条目） | `git diff --shortstat` | 已确认 |
| 删除的成员 | `crates/{pack,comm}`（首轮前）+ `crates/{audit,checkpoint,regress,exec,kv,einsum}`（§11 吸收，代码已迁入新家） | `git status` 中 29 条 `D` | 已确认 |
| 未跟踪的新模块 | `cutile/src/report/`、`train/src/snapshot/`、`op/src/einsum/`、`cutile/src/ast.rs`、`crates/dtype/`、`op/ops/*`、`tool/src/*` | `git status` 中 127 条 `??` | 已确认 |
| `weights` crate 未跟踪比例 | 33 个 src 文件中 29 个未跟踪 | `git status crates/weights` | 已确认 |

**判定**：本次审计描述的是"工作树现状"；凡涉及 `weights` 容器、`op` 22 算子、门禁本身、`report` 拆分的结论，都属于**在途未提交改动**，不应记作"已发布版本的质量"，但必须在合并前解决（见问题清单）。§11 的六次搬家只是移动这些未跟踪文件，没有改变它们的提交状态。

---

## 三、世界级优秀实践（应保持不动）

以下 8 项已达到国际顶尖框架"方法论"层面的水准，任何重构都必须保留其语义；削弱它们换取的整齐度是负收益。

1. **架构契约可执行，而不是评审清单。**
   `docs/ARCHITECTURE.md` 的层表、边集、所有权表被 `tests/architecture.rs` 逐条对照 `cargo metadata` 与源码：成员集合（`:29`）、边集合与总数（`:57`）、方向（`:80`）、每个成员至少一条边（`:106`）、每个"主人"类型在声称的 crate 里确实是 `pub`（`:125`）、以及禁止 `Result<_, String>` 回潮（`:151`）。六项在当前树全绿。
   这是本仓最值得保留的资产：**"文档说错了会红"**，而不是"评审时有人记得"。

2. **单一事实源被贯彻到编译期。**
   `dtype::DType` 22 个变体收口了原本三套枚举（`crates/dtype/src/lib.rs` crate 文档、`dtype/src/lib.rs:107,120,225`）；`prof::atom::Closed` + `Set<E, WORDS>` 把"全量/跨层名/位集下标"收成一份声明，容量由 `const FITS: () = assert!(..)` 在单态化时证明（`crates/prof/src/atom.rs:44-151`，`FITS` 在 `:56-59`）。

3. **"设备差异只有数据"。**
   新增设备 = `prof/devices/<vendor>.toml` + `prim/<vendor>/*` 片段 + 一个 `#[frontend]` 类型；`prim/build.rs` 在编译期扫描目录生成 `BUILTIN`/`WITNESS`/`DEVICES` 三张表（`build.rs:58,280,308,336`），漏文件、签名不符、缺 witness 都是构建失败（`:112-187`）。这比"每个后端一份手写 kernel"的路线高一个数量级。

4. **无隐藏优化阶段 + 中间产物全可见。**
   `cutile` 的五段（expand → negotiate → select → substitute → emit）之间传递的都是公开数据结构：`TileCombo`（`op/src/decl.rs:125`）、`Path`（`cutile/src/cap.rs:15-24`）、`Plan`（`cutile/src/plan.rs:18-33`）、`ast::Node`（`cutile/src/ast.rs:30-59`；生产路径在 `build.rs:83` 用它替换后再 `render`）、`Source`（`cutile/src/emit.rs:12-17`）。可打印、可哈希、可对拍。
   这一条有一个已知缺口（P1-4）：`substitute::render`（`substitute.rs:33`）是同一输入的**第二个渲染器**，只由对拍测试 `ast.rs:313` 保证与 `ast` 路径等价；在它删除之前，"中间产物全可见"只对 `ast` 那条路径成立。

5. **"代价只有一个来源"。**
   `einsum` 从一条求和表达式推出 flops/bytes；`op` 只声明表达式与位宽，`cutile` 的 roofline 只消费数对。`op/src/gemv.rs:301` 直接调 `F::formula()?.cost(..)`，没有第二份公式。

6. **unsafe 被压缩成一个有论证的安全岛。**
   `crates/storage/src/mmap.rs` 是全仓唯一 unsafe：`PROT_READ | MAP_PRIVATE` 建立映射（`:25-53`），`Send`/`Sync` 手写断言并给出理由（`:47-49`），`Drop` 里 `munmap`（`:136`），模块级放开只此一处（`:8-9`），且默认构建不含该模块（`storage/src/lib.rs:26-29`）。

7. **双源正确性 + 真实数据对拍。**
   `op` 每算子同时有模板实现与标量参考；`train::store` 的 `KERNELS` 表长度与顺序在编译期对齐 `op::REGISTRY`（`train/src/store.rs:498`，`const _: () = same_order(&KERNELS, &op::REGISTRY);` 在 `:589`）；`autograd/tests/gradcheck.rs` 用中心差分对 200 个随机图（`:214-220`）以 `2e-3 + 0.05·max(|fd|,1)` 容差检查。
   还有若干"高质量测试"值得点名：`safetensors.rs:642` 用 1 GiB 稀疏文件证明只读了头部、`safetensors.rs:704` 断言两种后端给出相等的 `Header`、`merkle.rs:129` 对 7 叶树所有下标证明并验证、`round_trip.rs:238` 关掉 CRC 再篡改字节以证明"签名层而非 CRC 层"在兜底。

8. **错误契约统一且有类别映射。**
   全仓一个 `Error`/`Code`/`Result`（`diagnostic`），类别由产生错误的层给，`Code::exit_code` 是唯一映射；`tests/architecture.rs:151` 禁止退回字符串错误。

---

## 四、逐分层、逐模块审计

### 4.1 分层总览（合并后，与 `docs/ARCHITECTURE.md` 第一节一一对应）

生产 LOC 为 `wc -l` 实测（含文件内联测试）；括号内是**被吸收子模块**的行数，已计入宿主。表中各 crate 之和 = `crates/*/src` 53 329（`prim` 的 788 已含 `build.rs` 622）；加上根入口 1 544 与 `weights/examples/dump_bin.rs` 40，全仓生产合计 **55 535**（与 §2.1 同口径；基准 harness 30 行不计）。

| 层 | 成员 | 生产 LOC（含内联测试） | 健康度 |
|---|---|---:|---|
| L1 基座 | `diagnostic` `dtype` `testkit` `llm-config` `tool` | 260 / 313 / 155 / 2 497 / 1 957（含 `regress` 337） | 高 |
| L2 数据层 | `prof` `op` `prim` | 3 106 / 8 278（含 `einsum` 1 378）/ 788 | 中高（`op` 膨胀） |
| L3 内核 | `cutile` | 9 469（含 `report` 3 947） | 中（双渲染路径、报告与机制同 crate） |
| L4 运行时 | `mem` `tensor` `storage` `tokenizer` | 205 / 585 / 400 / 125 | 高（`storage` 未被使用、`tokenizer` 仅契约） |
| L5 模型与训练 | `weights` `autograd` `data` `nn` `shard` | 10 183 / 3 134 / 233 / 1 534 / 524 | `weights` 低（职责过载），其余中高 |
| L6 调度与服务 | `train` `engine` `serve` `cli` | 3 459（含 `snapshot` 1 611）/ 1 303（含 `exec` 188、`kv` 334）/ 1 461（含 `proto` 1 126）/ 648 | 中（`serve`/`exec` 仅契约；`serve` 现已含线格式） |
| L7 入口 | 根包 `llm`（`src/main.rs` 684、`src/bin/cutile.rs` 860） | 1 544 | 中（含领域逻辑） |
| 工具成员 | `annotation` `fuzz` `benches` | 2 314 / 398 / 0 | `annotation` 中（分层豁免 + 认识算子名） |

### 4.2 逐成员职责与边界评审（26 行 = 25 crate + 根包）

被吸收的七个成员（`audit`/`checkpoint`/`regress`/`exec`/`kv`/`einsum`/`proto`）不再单独成行：它们的能力与判定已并入下表的宿主行，落点与追溯见 §6.0 末尾与 §11.1。

| 成员 | 拥有（已确认） | 不拥有 | 边界判定 | 关键证据 |
|---|---|---|---|---|
| `diagnostic` | `Error`/`Code`/`Result` | 领域错误字面量 | **干净**，零依赖 | `diagnostic/src/lib.rs:3-4` |
| `dtype` | 22 变体 `DType` + code | 层内能力子集 | **干净**，`code` 与旧 Wire 逐位兼容 | `dtype/src/lib.rs:107,120` |
| `testkit` | `approx_eq`/`device_enabled`/`scratch_dir`/位级 CRC | 任何生产逻辑 | **干净**，仅 dev 边 | `testkit/src/lib.rs:3-5`、`:91-104` |
| `llm-config` | `Settings`/`Key`/`Roots`/`App`/TOML | 领域语义 | **基本干净**；`Key` 手写 5 张表（P2-3） | `config/src/key.rs:20-79` |
| `tool` | lex/ccn/dup/call/docs/strip/gate/Report + `regress` | 运行时类型、退出码 | **干净**；扫描面有盲区（P1-7）；`Judgement` 与 `Gate` 现在同域 | `tool/src/lib.rs:1-11`、`tool/src/regress.rs:1-20` |
| `prof` | 档案 schema、`Prim` 闭集、`Set` 位集 | 机制代码、设备特判 | **干净**；`validate` 是语义检查器而非 schema（合理） | `prof/src/atom.rs:44-151`、`prof/src/prof.rs:605-833` |
| `op` | 22 条声明、模板、`REGISTRY`、每算子语义、`einsum` 子模块 | 机制（emit/编译） | **有泄漏**：语义表生成依赖宏内的算子名分支（P1-5） | `op/src/lib.rs:108`、`annotation/src/codegen.rs:318-344`、`op/src/einsum/formula.rs` |
| `prim` | 片段文本、生成的三张表 | 策略、能力区间 | **干净**；`build.rs:103` 目录项静默丢弃（P2-4） | `prim/src/tables.rs:7`、`prim/build.rs:58` |
| `annotation` | 3 个属性宏 | 领域类型 | **有泄漏**：宏内按算子名/族名分支（P1-5）；两套 skeleton 词汇表（P1-6） | `annotation/src/codegen.rs:211-345`、`syntax.rs:385-405` |
| `cutile` | 协商/选择/替换/emit/缓存/编译 + `report` 子模块 | 算子名 | **规则成立但有代价**：把 `op` 的类型原样 re-export 成自己的 API（P2-5）；两条渲染路径并存（P1-4） | `cutile/src/lib.rs:50-51`、`build.rs:82-83`、`substitute.rs:33`、`cutile/src/report/mod.rs` |
| `mem` | `Arena`/`Region`/`align_up` | 地址空间、布局 | **干净但极简**（只顺序分配、不释放，自述 P1） | `mem/src/lib.rs:110-130` |
| `tensor` | 字节 `Tensor` / 元素 `View<E>`、`Shape`、`Strides`、`SUPPORTED` | 分配、搬运 | **干净**；零拷贝是类型保证而非承诺 | `tensor/src/lib.rs:123-128,237-264,470-480` |
| `storage` | `Layout`/`Order`/`remap` + `mmap` | 分配、元数据 | **干净但未被使用**：生产入边 0（见 §6.2.1）；unsafe 岛合法；Windows 可移植性缺口（P1-2） | `storage/src/mmap.rs:1-53` |
| `tokenizer` | `Token`/`Tokenizer`/`Bytes` | BPE/SentencePiece | **纯契约 + 一个字节级实现** | `tokenizer/src/lib.rs:41-89` |
| `weights` | 外部格式只读适配 + 自有 LLPK 容器 + JSON/ZIP/rANS/DSA/SHA-256/Keccak/NTT/Golay/SVD | 张量、布局、GPU 搬运 | **职责过载**：一个 crate 承担 5 个领域（P1-1）；crate 头文档已滞后（P1-8） | `weights/src/` 33 文件；`weights/src/lib.rs:1-6` |
| `autograd` | tape/schedule/account/plan/replay | 数值与激活缓冲 | **深且真实**：可达剪枝、扇出账本、预算解算 | `autograd/src/{tape,schedule,account,plan,replay}.rs` |
| `data` | `Batch`/`Split`/`Chunks` | 分词、打乱 | **干净**，规模小 | `data/src/lib.rs` |
| `nn` | `Layer`/`LayerSpec`/`Linear`/`WeightFormat`/`Bounds`/`Stack` | 算子声明 | **干净**：格式→算子名唯一映射；推不出下界时点名理由 | `nn/src/bounds.rs:174-181`、`stack.rs:279+` |
| `shard` | 通信契约 + 三维拓扑 | 后端实现 | **契约 + 算术**；`Collective` 无实现 | `shard/src/collective.rs:11-14`、`topology.rs:118-152` |
| `train` | 训练状态 + 激活缓冲 + `Kernel` 表 + `snapshot` 子模块 | 反传公式、显存计划 | **干净**；`store.rs` 1 530 行混合 9 项职责（P2-6）；`snapshot` 1 611 行待拆（§11.4） | `train/src/store.rs:1-617,619-1065`、`train/src/snapshot/mod.rs:93-99` |
| `engine` | `Sequence`/`Scheduler` + `exec`/`kv` 子模块 | 批次张量、真正提交 | **干净**；连续批只做到"页账"，`exec` 仍是纯契约 | `engine/src/lib.rs:233-374`、`engine/src/exec.rs:31-145`、`engine/src/kv.rs:228-288` |
| `serve` | `Route`/`Endpoint`/`Response` + `proto` 子模块（`Envelope`/`Frame`/`Decoder`/`Kind`/`Version`） | 调度、载荷序列化 | **纯契约，明确无 socket**；线格式已是同 crate 子模块（半帧/多帧、毒化后须 `reset`） | `serve/src/lib.rs:3-9`、`:167-176`、`serve/src/proto/decoder.rs:163-288` |
| `cli` | `Command`/`Invocation`/`USAGE`/`CutileCommand` | 业务、排版、配置 | **干净但有错位**：`--set` 解析在 `src/bin/cutile.rs:57-78`（语法住在二进制里） | `cli/src/lib.rs:33-95` |
| `fuzz` | `Target`/`Seed`/`Behaviour`/`sweep` | 具体解析器 | **干净**：把 panic 转成 finding | `fuzz/src/lib.rs:126-130,223-225` |
| `benches` | criterion harness | 任何生产代码 | **干净**（工具成员，0 生产 LOC） | `crates/benches/` |
| 根 `llm` | 组装 | 新类型 | **越界**：`src/main.rs:17` 定义 `#[nn::layer]` 类型，与 `src/main.rs:1` "不定义任何新类型" 自相矛盾 | `src/main.rs:1,17,85-235,248-375` |

### 4.3 边界结论（三条判定）

1. **"依赖只指向更低层"这条规则是成立的，而且不是靠人记得**：`tests/architecture.rs` 六项断言全绿，`cargo metadata` 与文档边集逐条相等（含边数与 `+dev` 条数的计数断言；首轮 106/15，六条合并后 83/13，见 §11.7）。
   反例检查：唯一的"例外"是工具成员 `annotation`/`fuzz`/`benches` 的边**不参与方向判定**（`tests/architecture.rs:84-87`）。而 `op` 与 `nn` 对 `annotation` 是**硬依赖（非 dev）**——也就是说，两个分层成员的真实编译依赖被排除在方向规则之外。这不是错误，但让"唯一一条规则"这句话出现了一个需要读者自行理解的豁口（P3-1）。

2. **"数据层不知道机制"成立**：`op`（含 `einsum` 子模块）/`prof`/`prim` 的依赖里没有 `cutile`；`cutile` 生产代码里也没有任何算子名（14 处 `gemv_i4`/`gemm` 字符串全在 `#[cfg(test)]` 内）。

3. **"机制不知道领域"在第三处被破坏**：`annotation/src/codegen.rs` 按算子名分支（`flash_attention_f32:318`、`gemm_f32:324`、`gemm_i8:334`、`gemm_i4:344`），并按族名 11 处分支（`:211-310`）。于是"新增算子不必改机制"只对 `cutile` 成立，对**宏**不成立。

---

## 五、问题清单（P0 / P1 / P2 / P3）

> 判定基准：P0 = 不可信输入可致进程崩溃或超大分配；P1 = 真实边界破坏/可移植性/可维护性显著受损；P2 = 局部质量缺陷或门禁盲区；P3 = 命名/文档/风格级别的长期摩擦。

### P0（必须先修，改动都很小）

**P0-1 `rANS` 解码按文件字段分配内存 —— 4 字节可请求 4 GiB**
- 位置：`crates/weights/src/rans.rs:300`（`let src_len = u32::from_le_bytes(..) as usize;`）→ `:322`（`let mut dst = vec![0u8; src_len];`）
- 事实：`src_len` 来自被解码字节流，分配前没有任何与 `cur` 剩余长度或 lane 长度之和的交叉校验。
- 对照：同一 crate 的 `safetensors.rs:243-252` 在 `vec![0u8; declared]` 之前明确拒绝 `header_len > file_len`，并把理由写在注释里——说明"该有的防线"在本仓已有先例，只是 rANS 这条路没做。
- 影响：一个 4 字节字段即可让验证流程申请 4 GiB；在服务进程里等价于可被单个权重文件触发的内存 DoS。
- 等级：**P0**（安全/可用性）。

**P0-2 pickle 解析在奇数长度 `SETITEMS` 上越界 panic**
- 位置：`crates/weights/src/bin.rs:503-513`：`for chunk in pairs.chunks(2) { entries.push((chunk[0].clone(), chunk[1].clone())) }`
- 事实：`pop_until_marker`（`:362`）返回的计数完全由文件里的字节决定；`chunks(2)` 对奇数长度会产出长度 1 的末块，`chunk[1]` 直接索引越界。
- 影响：一个畸形 `.bin` 让宿主 panic，而该模块文档（`:8-9`）承诺"显式报错，不 panic"。
- 等级：**P0**。

**P0-3 `checkpoint` 编码路径上的 `expect` 可由权重文件触发**
- 位置：首轮 `crates/checkpoint/src/lib.rs:649`，现为 `crates/train/src/snapshot/mod.rs:651`（合并时文件头补了 2 行，行号 +2）：`sink.push(&[tag(entry.dtype()).expect("快照条目类型在标签闭集内")]);`
- 事实：`tag`（现 `:895-904`）只接受 `F32/F16/BF16/I8/U4`，其余返回 `Err`。而 `entry` 来自调用方给的 `weights::Header`，其 dtype 由文件内容决定。
- 影响：用一份含 `I64`/`F64`/量化类型的权重做快照 → panic，而不是"该格式不支持"的错误。这是一条**正常输入就能走到**的 panic。
- 等级：**P0**（改一行：`tag(..)?`）。

**P0-4 DSA 公钥/签名解码对长度零校验，截断即 panic**
- 位置：`crates/weights/src/dsa.rs:353-367`（`pk_decode`：`&pk[..32]`、`&pk[32 + i*step .. 32 + (i+1)*step]`）、`:376-391`（`sig_decode`：`&sig[c_len + i*z_step ..]`、`&sig[z_end..]`）、`:675` 附近 `hint_unpack_dyn` 的 `len - k`
- 事实：`Envelope::decode`（`envelope.rs:99-104`）从文件读 `pk_len`/`sig_len` 后只做 `cur.take(..)`（受输入长度限制，不会超量分配），但**不与参数集要求的最小长度比较**；随后 `Envelope::check`（`:111-119`）把它交给 `verify_internal` → 上述切片。
- 影响：1 字节的侧车即可 panic。fixture 的"fail-closed"只对**伪造**成立，对**截断**不成立。
- 等级：**P0**。

### P1（合并前应处理）

**P1-1 `weights` 单 crate 承担 5 个互不相干的领域**
- 事实：33 个 src 文件中，容器/线格式（`writer/reader/seg/codec/crc/leb/sink/merkle`）、外部格式（`safetensors/gguf/json/zip/bin`）、压缩（`rans`）、哈希（`sha256/keccak`）、数论/后量子（`dsa/ntt/golay/svd/lattice/ledger`）混在一起；`dsa.rs` 923 行是一份完整的 ML-DSA 实现。
- 影响：任何一处改动都触碰最大 crate（10 183 行、84 条债基线）；crate 头文档（`lib.rs:3-6`）已无法描述它（见 P1-8）；审计成本与合并冲突面同步放大。

**P1-2 `mmap` feature 在非 Unix 目标上无法编译**
- 事实：`crates/weights/Cargo.toml:18-20` 以**dev-dependency**（`features = ["mmap"]`）启用 `storage`；`crates/storage/src/mmap.rs:14` 无条件 `use std::os::unix::io::AsRawFd;`，整个文件没有任何 `cfg(unix)`（只有 `:144` 的 `target_pointer_width` 与 `:150` 的 `cfg(test)`）。
- 影响（推断）：CI 的 `windows-latest` 上 `cargo test --workspace --all-features`（`.github/workflows/ci.yml:46,53`）会因找不到 `std::os::unix` 而编译失败。**等级：P1（可移植性）**。本次无法在 Windows 目标上实测，标为推断。
- 附带事实（本次新发现，影响 §6.2）：`storage` 的**生产入边为 0**——`Layout`/`Order`/`remap` 没有任何 crate 调用，`Mmap` 唯一使用点是 `crates/weights/tests/round_trip.rs:175`。

**P1-3 13 处错误上下文仍指向已删除的 `pack` crate**
- 事实：`with("pack::quant")`（`weights/src/affine.rs:109,140`）、`pack::rans`（`rans.rs:377`）、`pack::leb`（`leb.rs:107,111`）、`pack::codec`（`codec.rs:29`）、`pack::seg`（`seg.rs:207`）、`pack::Writer`（`writer.rs:75,195,208,256`）、`pack::Archive`（`reader.rs:110,281`）——**13 个代码位点**；另有 `storage/src/mmap.rs:7` 的文档注释仍写"交给 `pack::Archive::parse`"（第 14 处，属文档）。`dsa.rs:792-793` 的 `packed` 是变量名，不是本问题。
- 影响：用户看到的诊断锚点指向一个不存在的模块，排障时会去搜索一个已删除的 crate。**这是"文档/诊断漂移"而非风格问题**。

**P1-4 `cutile` 同时存在两条模板渲染路径**
- 事实：生产走 `ast::build` + `ast::render`（`build.rs:82-83`）；`substitute::render`（`substitute.rs:33`）仍保留，调用点只有 `builtin.rs:342` 处的一次对拍检查和测试。两条路径的等价性由**一个测试**保证（`ast.rs:313`），代码注释自述"字符串路径保留用于回滚与对拍"（`build.rs:80-81`）。
- 影响：ADR 0017 的迁移处于半完成态；在移除前，任何替换规则的改动都要改两处并靠测试兜底。

**P1-5 属性宏认识算子名与算子族名（分层泄漏）**
- 事实：`annotation/src/codegen.rs:318` 判等 `"flash_attention_f32"`，`:324` 判等 `"gemm_f32"`，`:333/:345` 判 `gemm_i8`/`gemm_i4`；`:211-310` 按族名 11 处分支。`op/src/dequant_i8.rs:83` 与 `dequant_i4.rs:91` 用**不同的 `skeleton` 名**仅仅为了命中不同的 `else if` 分支。
- 影响：新增一个"需要自己的 cost/sample/reference 模块"的算子，必须改宏；`op` 的"加算子不改机制"承诺在宏这一层失效。

**P1-6 存在两套 `skeleton` 词汇表，且多数取值无人构造**
- 事实：`op::decl::Skeleton` 是 5 个变体（`decl.rs:30-36`：Elementwise/Reduce/Gemm/Attention/Scan）；`annotation::syntax::SKELETONS` 是 11 个名字（`syntax.rs:385-405`），两者不是投影关系。`Skeleton::Attention` 与 `Skeleton::Scan` 在全仓**没有任何构造点**（唯一构造 `Skeleton::Gemm` 在 `cap.rs:561` 的测试里）；`op.skeleton` 只被 `emit.rs:53` 的注释头与 `cutile/src/report/doctor.rs:161` 的报告字段消费。
- 影响：一个"闭集"事实上是装饰性的，违反本仓自己"闭集必须有唯一声明处"的契约精神。

**P1-7 静态门禁的扫描面小于"全仓"**
- 事实（已逐条验证）：
  - 只扫 `root/crates`（`tool/tests/gate.rs:23`），且 `is_production` 要求路径含 `src` 目录、文件名不是 `tests.rs`（`:149-152`）；因此 **`crates/prim/build.rs`（622 行）、`crates/weights/examples/dump_bin.rs`、`crates/*/tests/`、根 `src/` 全部不扫**。
  - 文档只扫 `docs/` 的**平铺** `.md`（`:154-166`），`docs/adr/**`、`README.md`、`CHANGELOG.md`、各 crate 的 README 都不检查。
  - `ccn::collect` 跳过整个 `macro_rules!` 体（`tool/src/ccn.rs:82-88`），宏内的函数与条件对 ccn/nloc 全部不可见（本仓 `constraint_builders!`、`inherent_names!` 正在此人眼下）。
  - `strip_tests` 只认外层属性（`tool/src/strip.rs:71-80`），`#![cfg(test)]` 与 `#[cfg(any(test, feature = "x"))]` 会被当成生产代码（`:118-124`）。
  - `call::scan` 只匹配 `.unwrap(`/`.expect(`/`panic!`（`tool/src/call.rs:29-69`），UFCS 形式与 `assert!`/`unreachable!` 不查。
- 影响：门禁为真但**覆盖面被高估**；"零 unwrap"之类的结论不能直接从门禁绿灯推出。

**P1-8 文档与代码的双向漂移（首轮 3 处，现剩 2 处）**
- ~~`README.md:5` 写"33 个成员"、`:51` 的 L4 仍含 `pack`、`:52` 与 `:73` 仍含 `comm`~~：**已修**。现 `README.md:5` 写 26 个成员，成员表与依赖图已按 §11 同步（本轮复查）。
- 仍在：`docs/ARCHITECTURE.md:81` 写"现有两条算子 `gemv_i4` 与 `gemv_f32`"；实际 `op/src/lib.rs:108` 是 22 条注册、`op/ops/` 有 22 对文件。
- 仍在：`crates/weights/src/lib.rs:1-6` 的"拥有/依赖"段只列 `Header`/`Entry`/`Reader`/`Loader`/`safetensors`/`json`，没有 LLPK 容器（`writer`/`reader`/`seg`/`codec`/`merkle`），也没提 `mem`/`dtype` 依赖，与 `Cargo.toml` 和 `ARCHITECTURE.md:100-101` 不一致。
- 影响：本项目最有价值的资产是"文档可执行"；恰恰是**没有门禁覆盖**的那几份文档先漂了。这说明 `docs/` 链接检查（只查链接是否存在）不足以防内容漂移。

**P1-9 架构测试的扫描集曾是手写目录，删成员会静默变绿（已修）**
- 事实：首轮 `tests/architecture.rs` 的错误边界扫描集是五个硬编码目录；`rust_files` 对不存在的目录返回空集合，于是"成员少了、扫描少了"不会红。§11 执行时已改为从层表派生，并加了"层表里每个成员必须扫出至少一个文件"的反向断言（`tests/architecture.rs:149-159`、成员集合断言 `:29`）。
- 为什么记在 P1 而不是 P3：它属于"门禁本身可被绕过"，与 P1-7 同类；修复前的绿灯不能证明"全仓都被扫过"。

### P2（应纳入下一批清理）

**P2-1 `axes.rs` 在公开 API 里 `assert!`**
- `weights/src/axes.rs:89`：`assert!(verified(&best, w), "raw 基线必须逐位还原")` 位于 `pub fn search`（`:74`）内，而该函数已返回 `Result`。崩溃点应当是 `Err`。

**P2-2 未知 mode 码被静默降级为 `Infer`**
- `weights/src/mode.rs:53-55`：`Mode::from_code(code).unwrap_or(Mode::Infer)`。这是该 crate 唯一"闭集码不被拒绝"的地方。

**P2-3 `config::Key` 手写 5 张平行表**
- `config/src/key.rs:20-31` 的 `ALL: [Self; 3]` 派生 `file_key`（`:34-40`）、`env_suffixes`（`:46-52`）、`kind`（`:66-70`）、`doc`（`:73-79`）：加一个键要改 5 处，文件自己也承认（`:3-4`）。它够不到 `Closed`（方向为 `prof → llm-config`），属于"允许本地保留但必须登记"的情形；建议按 `(key, file_key, env_suffix, kind, doc)` 一张表投影成 4 张视图。

**P2-4 `prim/build.rs` 与 `prof` 装载时的目录项静默丢弃**
- `prim/build.rs:103` 的 `filter_map(|e| e.ok())`：权限/竞态导致读不到的目录项**无声消失**，构建仍绿，只是内嵌表变小。
- `prof/src/lib.rs:110-111` 同样的 `filter_map(..ok())`：一只读不到的设备档案等于"这台设备不存在"。

**P2-5 `cutile` 把 `op` 的类型 re-export 成自己的 API**
- `cutile/src/lib.rs:50-51` 直接 `pub use op::decl`/`op::layout`/`op::sch`。机制 crate 的公开面因此等于声明 crate 的公开面，任何 `op` 的类型改名都会变成 `cutile` 的破坏性变更。

**P2-6 三个超大单文件混合多项职责**
- `prof/src/prof.rs`（1 712 行）：schema 解析（`parse_provide:1055`）+ 语义校验（`validate:605`）+ 设备表装配。
- `train/src/snapshot/mod.rs`（1 611 行，首轮为 `crates/checkpoint/src/lib.rs`）：格式常量、编码器 `Sink`（`:677-703`）、解码 `:600+`、`verify`/`check_weights`（`:93-99`）。
- `train/src/store.rs`（1 530 行）：分配器（`:1-99`）、形态/`Lane`（`:101-218`）、缓冲视图（`:220-330`）、`Inputs`（`:333-434`）、`Kernel`/`Reference`（`:436-479`）、kernel 表（`:481-617`）、每算子适配（`:619-774`）、空闲表池（`:776-855`）、`Buffers`（`:857-1065`）、测试（`:1067-1531`）——9 段可独立命名。

**P2-7 重复实现的共享规则（"同一规则三份"）**
- 对齐：`mem::align_up`（`mem/src/lib.rs:31`）、`gguf::align_up`（`weights/src/gguf.rs:420`）、`writer.rs:148` 的 `next_multiple_of`。
- LEB128：`weights/src/leb.rs:11,55` 与 `train/src/snapshot/mod.rs:756,812`（连"非规范形式必须拒绝"的论证都各写一遍）。
- 步幅：`tensor::Strides::contiguous`（`tensor/src/lib.rs:99`）与 `storage::Layout::strides`（`storage/src/lib.rs:67`）。
- 元素数：`tensor::Shape::numel`（`tensor/src/lib.rs:74`）与 `weights::seg::numel`（`seg.rs:196`），根因是 `seg` 用 `Vec<u64>` 而 `Shape` 用 `usize`。

**P2-8 结构性死代码**
- `weights/src/seg.rs:65-68`：`for axis in &self.shape { let _ = axis; }` —— 注释说明"轴长允许 0"，但循环体是空的，校验从未写。
- `tool/src/ccn.rs:138-142`：`let nested = collect(items, &mut nested)` 结果从不读取，却计入自身 `ccn=7`。

**P2-9 可空转的测试（"绿但没测"）**
- `weights/src/bin.rs:582-590`、`zip.rs:198-201`：`/tmp/pt/model.bin` 不存在时直接 `return`；而这是 ZIP+pickle 路径**唯一**的练习。`bin.rs:619` 的断言只检查 `bytes.len() == 16`（`:618` 注释却声称"与原始字节一致"）。
- `cutile/src/report/witness/mod.rs:108-110`：无后端时直接 `return`，与 `testkit::device_enabled` 的显式门控相比是两种风格。
- `tool/tests/ccn.rs:19-22`：旧二进制不存在时打印 `skip` 并 `return`。
- `reader.rs`、`seg.rs`、`codec.rs` 三个解析侧文件 **0 个本地测试**（截断/位翻转的覆盖全在 `tests/round_trip.rs`）。

**P2-10 `cutile::report` 的三处结论强度被高估**
- `common.rs:64-69`：`group <= 1` 时返回空的负例表，而 `verify.rs:345-354` 的"负例也是判据"正是遍历这张表——这类设备实际测了 0 个负例。
- `bench.rs:196-197`：缺 p50 时用最小值顶替且不留标记。
- `witness/evidence.rs:97-102`：设备不在本机时 `Err` 被丢弃；注释解释了理由（存在性结论 ≠ 证据不成立），但"runner 自身坏了"与"没有这台设备"从此不可区分。

**P2-11 三套 dtype 词汇并存且互不校验**
- `dtype::DType`（主人）、`weights::bin::Kind`（9 变体，`:21,45-52`，有意保留）、`weights::gguf::Kind`（15 变体 + `SPAN`/`ELEM`，`:26,42-44`）。三者的一致性没有表驱动判据；`bin.rs:61` 的 `I8|U8 => DType::I8` 只靠注释解释与 `safetensors.rs:56-62` 的差别。

### P3（长期摩擦）

- **P3-1** `annotation` 同时被当作"工具成员"（不参与方向判定）和"`op`/`nn` 的硬依赖"（参与编译），`ARCHITECTURE.md:21-23` 与 `tests/architecture.rs:84-87` 需要一段更准确的措辞。
- **P3-2** 同名字型跨域：`train::Schedule`（学习率）vs `autograd::Schedule`（反传顺序）；`autograd::Plan` vs `cutile::Plan`（已在 `autograd/src/lib.rs:22-24` 用散文区分，靠注释而非类型名）。
- **P3-3** 根入口违约：`src/main.rs:17` 定义 `#[nn::layer]` 类型，而 `:1` 声明"不定义任何新类型"；`src/main.rs:382-423` 的 `decode_f32`/`parse_input` 属于可复用逻辑却只存在于二进制。
- **P3-4** `src/bin/cutile.rs` 860 行中约 760 行是排版（`:129-682`）；排版属于 `tool::Report` 的职责域，二进制不该自己长一套。
- **P3-5** "只准降"的棘轮其实只做到"不许涨"：`Gate::check`（`tool/src/gate.rs:124-136`）只在 `value > allowed` 与 `value <= limit` 两处失败，**改善不受任何约束**；`Gate::stale`（`:140-147`）以身份（而非度量）为键，一个同时登记 ccn 与 nloc 的函数只要 nloc 还超标，其已改善的 ccn 条目永远不会被要求删除。文档"存量项只准降"（`gate.rs:4`）与实现不符。
- **P3-6** `docs/lld.md` 2 410 行、`docs/10k-analysis.md` 的规模数字（生产 25 347 行）已明显落后于当前树（38 530 行），且该文档未纳入任何门禁。

---

## 六、模块合并与拆分建议

### 6.0 全量裁定表（合并后 26 行；本表是"每个成员怎么处置"的唯一入口）

本节取代首轮的"32 成员覆盖表"。成员已从 33 降到 26，做法是七条**单消费者吸收**（ADR 0019）；落点与执行记录见 §11.1，本表只回答"现在每一个成员去哪"。

判定用四级：

- **保持**：边界正确，crate 形状不动（最多做子模块化）。
- **内部整合**：crate 不动，把文件/类型搬进子模块、表驱动化，或删掉第二条实现路径。
- **合并**：消掉 crate，或把它的一部分并入相邻 crate（本轮已执行七条）。
- **拆分**：新建 crate，或把一个文件按职责切成多个。

"生产入边"= 其他成员（**不含根包**、**不含 dev 边**）的 `[dependencies]` 指向它的条数，`cargo metadata` 口径；它与 `ARCHITECTURE.md:38` 的 81 条边（含根包与 13 条 `+dev`）不同，后者是契约全集。"现 LOC"为 `wc -l crates/*/src` 实测（含文件内联测试）。

| 成员 | 生产入边 | 出边 | 现 LOC | 最终处置 | 依据 / 附带动作 |
|---|---:|---:|---:|---|---|
| `diagnostic` | 20 | 0 | 260 | 保持 | 全仓唯一错误契约，`tests/architecture.rs:151` 禁止回退字符串错误 |
| `dtype` | 3 | 0 | 313 | 保持 | L1 精度闭集；并入 `tensor` 会让 `prof` 反向依赖 L4 |
| `testkit` | 0（dev 5） | 0 | 155 | 保持 | dev-only 夹具（`testkit/src/lib.rs:3-5`），不进任何二进制 |
| `llm-config` | 4 | 0 | 2 497 | 内部整合 | `Key` 派生 5 张平行表（`config/src/key.rs:20-79`）→ 单表 + 4 投影（§6.4 W6） |
| `tool` | 0 | 2 | 1 957 | 内部整合 | `dup.rs:11-14` 只数文件行数、不做重复检测；已吸收 `regress` |
| `prof` | 3 | 3 | 3 106 | 拆分文件 | `prof.rs` 1 712 / `atom.rs` 1 190，见 §6.3 S3 |
| `op` | 3 | 4 | 8 278 | 内部整合 + 拆 `decl.rs` | 22 条 `Registration` 可宏生成（§6.6）；`decl.rs` 673 行混三职责；已吸收 `einsum` |
| `prim` | 1 | 2 | 788 | 保持 | 三张内嵌表由 `build.rs:58` 扫描生成，无策略 |
| `annotation` | 3 | 0 | 2 314 | **不可合并** | `proc-macro = true`（`crates/annotation/Cargo.toml:12-13`）；删族名分支（§6.4 W7） |
| `cutile` | 0 | 6 | 9 469 | 内部整合（删双渲染）+ 拆 3 文件 | P1-4；`cache.rs` 1 246 / `cap.rs` 916 / `backend.rs` 859；已吸收 `report` |
| `mem` | 2 | 1 | 205 | 保持 | `tensor` 与 `weights` 都直用它；不释放是有意最小集（`mem/src/lib.rs:85-86`） |
| `tensor` | 7 | 3 | 585 | 保持（并接收布局） | 步幅的唯一主人（`tensor/src/lib.rs:99`），见 §6.3 S2 |
| `storage` | **0** | 2 | 400 | 拆分/合并 | 零生产入边；`Layout` 与 `Strides` 重复，`mmap` 只被测试用 |
| `tokenizer` | 1 | 1 | 125 | 保持 | 文本↔token 唯一边界 + 一个字节级参考实现，不装 BPE 是有意 |
| `weights` | 2 | 4 | 10 183 | 目录重组 → 择机拆 3 crate | P1-1；先 §6.4 W1 再 §6.3 S1 |
| `autograd` | 1 | 2 | 3 134 | 拆分 `plan.rs`/`replay.rs` | 866/683 行，两个算法各占一个文件 |
| `data` | 1 | 2 | 233 | 保持 | 样本切分属 L5 语义，`train` 属 L6 |
| `nn` | 1 | 5 | 1 534 | 内部整合 | `stack.rs` 580 行可切"层链/总下界/未计价"三块 |
| `shard` | 1 | 1 | 524 | 保持 | 通信契约 + 拓扑算术；后端未实现是有意占位 |
| `train` | 0 | 8 | 3 459 | 拆分 `store.rs` + `snapshot` | 1 530 行里 9 段职责（P2-6）；`snapshot` 1 611 行待拆（§11.4） |
| `engine` | 1 | 2 | 1 303 | 拆分 `sequence`/`scheduler` | 已吸收 `exec`/`kv`；`exec` 的 SPI 去留待决（§11.4） |
| `serve` | 0 | 3（+dev 1） | 1 461 | 保持（接收线格式） | 明确无 socket（`serve/src/lib.rs:7-8`）；已吸收 `proto`（`serve/src/proto/`，1 126 行）；出边 `engine`/`tokenizer`/`diagnostic` |
| `cli` | 0 | 1 | 648 | 增边（语法回填） | `--set` 解析在 `src/bin/cutile.rs:57-78`，属 `cli` 的语法域 |
| `fuzz` | 0 | 1 | 398 | 保持（工具成员） | 契约是"任何输入"，与 `testkit` 的"夹具"不同 |
| `benches` | 0 | 0（dev 3） | 0 | 保持（工具成员） | 只有一个 criterion harness |
| 根包 `llm` | — | 15 | 1 544 | 拆分（业务逻辑出仓） | `src/main.rs:17` 定义 `#[nn::layer]` 类型，违反 L7"不定义新类型" |

**已消失的七个成员（追溯，不再计入上表）**

| 首轮成员 | 首轮 LOC | 现在的位置 | 首轮结论 | 今天的读法 |
|---|---:|---|---|---|
| `checkpoint` | 1 609 | `train/src/snapshot/mod.rs` | 拆分 4 文件 | 已并入；拆文件转为 §11.4 待办 |
| `audit` | 3 947 | `cutile/src/report/`（13 文件） | 拆分文件 | 已并入；`report/mod.rs` 现 700 行，未过 899 阈值 |
| `regress` | 337 | `tool/src/regress.rs` | 合并候选 | 已并入；`Judgement` 保留原名 |
| `exec` | 188 | `engine/src/exec.rs` | 暂不合并 | 已并入；SPI 去留仍未决 |
| `kv` | 334 | `engine/src/kv.rs` | 保持 | 已并入；页数公式 4 → 1 |
| `einsum` | 1 378 | `op/src/einsum/`（4 文件） | 保持 | 已并入；`Bits` 改用 `Closed` |
| `proto` | 1 126 | `serve/src/proto/`（4 文件） | 保持 | 已并入；`pub` 路径变 `serve::proto::`，编译期常量与测试原样 |

统计口径（合并后）：25 个 crate 里**0 条生产入边** 9 个（`cutile`/`train`/`serve`/`cli`/`tool`/`storage`/`testkit`/`fuzz`/`benches`）。其中 5 个只被最终二进制消费（正常），`testkit` 是 dev-only，`fuzz`/`benches` 是工具成员，**只有 `storage` 同时"零生产入边 + 零消费者"**——这是 §6.2.1 第一条要处理的成员。要强调的是：**"只有 1 个消费者"本身不是合并理由**；六条吸收成立的理由是 ADR 0019 的"单消费者吸收"，而不是入边数本身。

### 6.1 硬约束与有意设计：不能合并的边界

| 对象 | 为什么不合并 | 证据 |
|---|---|---|
| `op` ↔ `cutile` | 数据 vs 机制；合并会把"加算子"变成"改机制" | `op` 无 `cutile` 依赖；`cutile` 生产代码无算子名 |
| `prof` ↔ `prim` | 档案 schema/能力闭集 vs 片段文本；`prim` 的内容由 `build.rs` 扫描生成 | `prof/src/prof.rs` vs `prim/build.rs:58` |
| `dtype` ↔ `tensor` | `dtype` 是 L1，被 `prof`/`weights`/`train` 共用；并入 `tensor` 会让 `prof` 反向依赖 L4 | `Cargo.toml` 边 |
| `diagnostic` ↔ 任意层 | 唯一错误类型；合并即制造第二事实源 | `tests/architecture.rs:151` |
| `autograd` ↔ `train` | `autograd` 只读 tape/plan，不碰数值；`train` 才是缓冲与数值 | `autograd/src/replay.rs:38-50` |
| `audit` ↔ `cutile` | 机制 vs 命令策略；合并后 `cutile` 会被迫拥有"形状矩阵/重复次数/达成率口径" | 现 `cutile/src/report/mod.rs:1-9`（首轮 `audit/src/lib.rs:7-9`）；**§11.2 改判为合并**（未做的附带动作见 §11.4） |
| `einsum` ↔ `op` | "推导代价"与"声明算子"是两种语义；并入后 `op` 达 8.3k 行，且 `formula.rs` 的自有词表（`Expr`/`Extents`）会被算子目录淹没 | 现 `op/src/einsum/formula.rs` 694 行（首轮 `einsum/src/formula.rs`）；`op` 是唯一入边；**§11.2 改判为合并**（未做的附带动作见 §11.4） |
| `annotation` ↔ 任意 crate | `proc-macro = true` 必须单独成 crate；且"零依赖"本身是契约（`ARCHITECTURE.md:197`） | `crates/annotation/Cargo.toml:12-13` |
| `kv` ↔ `engine` | 页账（`Table::grow` 不悄悄扩界）vs 序列调度；`engine` 只借页号，不碰页内数据 | 现 `engine/src/kv.rs:277-288`、`engine/src/lib.rs:233-262`；**§11.2 改判为合并**（前提：页数公式收口为 1 处，已兑现） |
| `proto` ↔ `serve` | 线格式 vs 端点契约；首轮判"不合并"（帧格式有 ADR 0016） | 现 `serve/src/proto/decoder.rs:163-288`；**§11.2 改判为合并**（第七次吸收，已执行：唯一消费者 + 同层同域） |
| `checkpoint` ↔ `train` | 字节格式 + 恢复事实 vs 训练循环状态；合并会让格式随训练代码演进 | 现 `train/src/snapshot/mod.rs:93-99`；**§11.2 改判为合并**（前提：补字节黄金向量 + 拆 1 611 行文件，均未做，见 §11.4） |
| `shard` ↔ `train` | 通信契约要被**未来的多后端实现**消费，`train` 只是第一个用户 | `shard/src/collective.rs:11-14` |
| `data` ↔ `train` | 样本切分是 L5 语义，训练循环是 L6；合并让 L6 拥有样本 | `Cargo.toml` 边（`train → data`） |
| `mem` ↔ `tensor` | `weights` 直接消费 `mem` 的 arena；并入 `tensor` 会给 `weights` 加一层间接 | `crates/weights/Cargo.toml:13` |
| `weights` ↔ `train::snapshot` | `snapshot` 只借条目表、不复制（`ARCHITECTURE.md:101`）；反向合并会把快照格式塞进权重 crate | `train` 依赖 `weights` 单向 |
| `tensor` ↔ `mem` 以外的 L4 | `tensor` 是 9 个 crate 的公共底座，任何合并都会让其中若干层被迫上移 | `tensor` 生产入边 9 |

### 6.2 合并决策（仅对"零入边/近似零入边"的成员逐个定案）

#### 6.2.1 建议做（按优先级）

| 对象 | 现状（可回链） | 结论 | 收益 | 风险 | 回滚 |
|---|---|---|---|---|---|
| `storage` 的 `Layout`/`Order`/`remap` → `tensor` | 400 行中约 250 行是布局代码，**零生产调用点**；`Layout::strides`（`storage/src/lib.rs:67`）与 `Strides::contiguous`（`tensor/src/lib.rs:99`）是同一规则两份实现 | **做（第一步）** | 删掉 P2-7 的步幅双份，`tensor` 成为步幅唯一主人 | 低：`Order`/`remap` 各有一个 doctest，需随迁 | 低（`pub use` 保留路径） |
| `storage` 的 `Mmap` | 唯一使用点是 `crates/weights/tests/round_trip.rs:175`；生产代码零调用 | **二选一**：(a) 让 `safetensors::File` 真的走 mmap 后端（补生产入边）；(b) 按 ADR 0004 移出 `members`，`mmap` 降为测试夹具 | (a) 兑现 `ARCHITECTURE.md:101` 的"内存/映射"承诺；(b) 少一个 crate | (a) 需处理 `Send`/`Sync` 与生命周期（`mmap.rs:47-49`）；(b) "唯一 unsafe 岛"会从架构图里消失，需同步文档 | 低 |
| `regress` → `tool`（或补生产消费者） | 337 行，0 生产入边，唯一消费者是 `benches` 的 dev 边 | **已并入 `tool`**（§11.1） | 判定与报告同域（`tool/src/lib.rs:1-11`），少一个 crate | `Judgement` 与 `cutile::report::Verdict` 语义不同（`ARCHITECTURE.md:137-141`），已保留原名 | 低（一次 `pub use` + 3 处 import） |

`storage` 这一条是本次新增结论中最强的：它同时满足"零生产入边"与"与其他 crate 重复实现同一规则"两个合并条件，而且删除它不会让任何生产代码编译失败——按 `ARCHITECTURE.md:202-203` 自己的标准，它今天更接近"目录"而不是"模块"。

#### 6.2.2 提出过但驳回的合并（附重议条件）

| 曾考虑的合并 | 驳回理由 | 什么条件下重议 |
|---|---|---|
| `einsum` → `op` | 代价唯一来源需要物理隔离；合并后 `op` 变 8.2k 行，且两套词表（`Expr`/`Cost` vs `OpDecl`/`Skeleton`）互相遮挡 | 若 `op` 先拆成 `op-decl` + `op-model`，再考虑把 `einsum` 归入后者 |
| `exec` → `engine` | 188 行的占位是 ADR 0004 要求的"接口形状"（现 `engine/src/exec.rs:31-145`）；**§11.2 改判为合并**，未使用 SPI 同批决断 | 第一个真后端落地时——此时 `Executor` 的实现者若在 L4，必须把契约拆回 L4 |
| `data`/`shard`/`checkpoint` → `train` | 三者各自拥有字节格式或拓扑算术，合并会让 L6 拥有 L5 语义；**§11.2 只对 `checkpoint` 改判（补黄金向量），`data`/`shard` 维持驳回** | 交付面收缩到"单机单卡"时（即放弃 `shard` 的契约） |
| `proto` → `serve` | 帧格式有独立 ADR，且 `Decoder` 的半帧/毒化语义不属端点策略；**§11.2 改判为合并并已执行**（ADR 0019 第七次） | 若未来出现第二个传输消费者（第二个 crate 要读同一帧格式）时拆回 L5 |
| `kv` → `engine` | 页账有独立的双释放拒绝与 `in_use` 计数（现 `engine/src/kv.rs:228-245`）；**§11.2 改判为合并**（前提：页数公式收口，已兑现） | 出现第二种调度策略、各自持有页账时 |
| `cli` → 根包 | `cli` 是纯语法（648 行、零领域知识），根包是组装；合并会把语法带进 L7 | 不建议 |
| `testkit` → `fuzz` | 夹具（确定性、可重现）与模糊测试（"任何输入"）是两套契约 | 不建议 |
| `audit` → `cutile`（或反向） | 机制 vs 命令策略；`audit` 拥有达成率口径与见证执行；**§11.2 改判为合并**（新增证据：工具链所有权分裂 + 两份"编译并运行"实现） | `audit` 变成纯排版时——但 §11.3 用"`report/` 子模块 + 机制模块零算子名"测试替代了这一物理边界 |

#### 6.2.3 不改 crate 形状、只改依赖边（真删除）

| 动作 | 现状 | 收益 |
|---|---|---|
| `train::snapshot` 改用 `weights::leb` | 自带 `varint`（`train/src/snapshot/mod.rs:754,810`），连"非规范形式必须拒绝"的论证都重写了一遍 | 删一份 LEB 实现（已是 `weights::crc` 的消费者，口径一致） |
| `weights::gguf` 改用 `mem::align_up` | 本地重写 `align_up`（`weights/src/gguf.rs:420`） | 对齐规则变一份（P2-7） |
| `cutile` 停止 `pub use op::decl/layout/sch` | `cutile/src/lib.rs:50-51` 把声明层的公开面变成自己的公开面（P2-5） | `op` 改名不再是对 `cutile` 的破坏性变更 |
| `op`/`nn` 对 `annotation` 的硬依赖写进边表 | `ARCHITECTURE.md:21-23` 把它归为"工具成员、边不参与方向判定"，但 `op`/`nn` 的**非 dev** 依赖确实参与编译 | 边表与实际一致（P3-1） |
| `cli` 收回 `--set` 语法 | `src/bin/cutile.rs:57-78` 在解析前自己处理 `--set key=value` | 语法集中到 `cli`，二进制只剩装配 |

### 6.3 crate 级拆分

**S1 `weights` → 3 个新 crate（最高收益，但分两步）**

| 新 crate | 收纳 | 行数（约） | 说明 |
|---|---|---:|---|
| `llm-container` | `writer` `reader` `seg` `codec` `crc` `leb` `sink` `merkle` `safetensors` `gguf` `json` `zip` `bin` | ~4 500 | "读/写别人的文件 + 自己的线格式"，与 `tensor`/`mem`/`dtype` 同级 |
| `llm-codec` | `rans` `lattice` `quant` `affine` `axes` `scale` `patch` `mode` `envelope` `ir` | ~2 300 | 纯字节变换与压缩，不认识张量 |
| `llm-crypto` | `sha256` `keccak` `dsa` `ntt` `golay` `svd` `ledger` | ~2 700 | 密码学/数论；当前放在 `weights` 里既无必要也放大审计面 |
| 保留 `llm-weights` | `lib`（`Header`/`Entry`/`Reader`/`Loader`） | ~400 | 只留"权重是什么"的契约 |

- 风险：**中**（`weights` 有 84 条债基线与 29 个未跟踪文件，拆分与在途改动叠加）。
- 回滚：低（纯移动 + re-export 保持 `weights::*` 路径不变）。
- 前提：先把 `pack::*` 诊断标签修正为 `weights::llpk::*`（否则一次移动会把 13 处错标签变成 13 处更难找的错标签）。
- **分两步（关键）**：第一步只做目录与 re-export（§6.4 W1），不新建 crate；第二步才真拆。理由是 84 条债基线的身份键含路径（`crates/weights/src/...`），一次移动会让 84 条基线同时失效，而它们刚刚才建立。

**S2 `storage` → 布局并入 `tensor`，`mmap` 留原地**

| 内容 | 行数（约） | 去向 | 理由 |
|---|---:|---|---|
| `Layout`/`Order`/`remap` | ~250 | `tensor` 的 `layout` 子模块 | 与 `Strides::contiguous` 同一域；`tensor/src/lib.rs:99` 已是步幅主人 |
| `Mmap` | ~150 | 留在 `storage`（或按 §6.2.1(b) 降为测试夹具） | 全仓唯一 unsafe，需独立 feature 门（`storage/src/lib.rs:26-29`） |
| 合并后体量 | `tensor` 585 → ~835；`storage` 400 → ~150 | — | 少一个"零入边"crate，少一份步幅实现 |

**S3 文件级拆分（保持公开 API 不变，仅改模块入口与 re-export）**

| 文件 | 建议子模块 | 关键分界（可回链） |
|---|---|---|
| `prof/src/prof.rs` 1 712 | `schema.rs`（`parse*`）、`validate.rs`、`devices.rs` | `:605`（语义不变量）与 `:1055`（schema 读取）是两条不同职责 |
| `prof/src/atom.rs` 1 190 | `closed.rs`、`set.rs`、`signature.rs`、`vocab.rs` | `Set` 在 `:44-151`，两个宏在 `:598` 与 `:777` |
| `train/src/snapshot/mod.rs` 1 611 | `format.rs`、`encode.rs`、`decode.rs`、`snapshot.rs` | `:93-99` 三入口；`Sink` 在 `:677-703`；`tag` 在 `:893-902` |
| `train/src/store.rs` 1 530 | `alloc.rs`、`lane.rs`、`view.rs`、`kernel.rs`（含 `KERNELS`）、`pool.rs`、`buffers.rs` | `:101-218`/`:436-617`/`:776-855`/`:857-1065` 四段 |
| `cutile/src/cache.rs` 1 246 | `key.rs`、`entry.rs`、`gc.rs` | `CacheKey` `:56-78`；`put_artifact` `:467`；三个类型族已分别定义 |
| `cutile/src/cap.rs` 916 | `need.rs`、`path.rs`、`negotiate.rs` | `Path` `:15-24`、`NeedReport` `:110`、`negotiate` `:435`/`:443` |
| `cutile/src/backend.rs` 859 | `detect.rs`、`harness.rs` | `cuda_harness` `:222` 的分支是 6 个校验 + 3 行 `ctype` 表（`:227-305`） |
| `cutile/src/build.rs` 778 | `source.rs`、`key.rs`、`artifact.rs` | `source` `:40`；AST 渲染 `:82-83` |
| `annotation/src/syntax.rs` 1 144 | `model.rs`、`frontend.rs`、`layer.rs` + 共用 `cursor.rs` | 三个 `parse_*` 入口即边界（`:149`/`:325`/`:447`），`Cursor` `:733-787` |
| `autograd/src/plan.rs` 866 | `footprint.rs`、`budget.rs`、`choice.rs` | 存活性 `:358-373`、释放点 `:247-258`、段数解算 `:404-439` |
| `autograd/src/replay.rs` 683 | `table.rs`、`store.rs`、`drive.rs` | 两张区间表 `:57-64`、重算次数对拍 `:89-100`、`Store` `:38-50` |
| `cutile/src/report/mod.rs` 700 | 顶层类型下沉到 `report/types.rs`，`mod.rs` 只留 `pub use` + 文档 | 顶层定义仅 3 处（`:15-26`、`:39-43`、`:80-84`） |
| `cutile/src/report/witness/device.rs` 622 | `probe.rs`、`compile.rs`、`run.rs` | `std::env::consts::OS` 三处分发（`:305,330,469`）是主机探测的天然边界 |
| `cutile/src/report/verify.rs` 618 | 保留 `verify` 入口，拆 `accept.rs`（负例）、`cuda.rs` | `verify` `:211-216`、`verify_one` `:310-315`、`verify_one_cuda` `:431-509` |
| `weights/src/dsa.rs` 923 | 随 S1 移入 `llm-crypto`；内部切 `params`/`encode`/`decode`/`sign` | 参数 `:255-289`、`pk_decode` `:353`、`sig_decode` `:376`、`keygen` `:394`、`sign` `:435`、`verify` `:459` |
| `op/src/decl.rs` 673 | `parse.rs`、`expand.rs`、`skeleton.rs` | `parse` `:177`、`expand` `:476`、`Skeleton` `:30-36`、`TileCombo` `:125` |
| `op/src/lib.rs` 578 | `registry.rs`（`REGISTRY` 在 `:108`）+ 文档留在 `lib.rs` | 22 条注册与 22 对文件已一一对应 |
| `engine/src/lib.rs` 782 | `sequence.rs`、`scheduler.rs`（`exec.rs`/`kv.rs` 已独立） | `Sequence` `:142-145` vs `Scheduler` `:233-374` |
| `nn/src/stack.rs` 580 | `layers.rs`、`floor.rs`、`unpriced.rs` | `from_header` `:279+` 与两个清单（未覆盖/无法计价）本就分开 |
| `op/src/einsum/formula.rs` 694 | `axes.rs`、`cost.rs` | 现有 3 条债项（`formula.rs:axes[1]`、`cost`、`new[1]`）已指向三个职责 |
| `src/main.rs` 684 | 业务出仓：`LmHead` → `nn`，`decode_f32`/`parse_input` → `weights`/`nn` | `:17-18`（新类型）、`:382-396`、`:401-423`；`main`/`status`/`run` 留下 |
| `src/bin/cutile.rs` 860 | 排版 → `cli` 新 `render` 模块；二进制只留 `main`/`run`/`take_overrides` | `:129-682` 全是渲染，`audit` 已返回结构化结论 |

**S4 不动的地方**：`diagnostic`、`dtype`、`tensor/src/lib.rs`（585 行的字节/元素双视图已是单一职责）、`config/src/toml.rs`（583 行，全仓唯一 TOML 解析器）、`serve/src/proto/{mod,frame,decoder}.rs`、`shard/*`、`serve`、`cli`、`tool` 的分析器、`testkit`、`fuzz`、`tests/architecture.rs`、`tests/contracts.rs`。这些要么是零成本抽象 + 单一事实源样板，要么已按职责分完，改动只增回归面。

### 6.4 内部模块重组（不新建 crate 的最小动作）

与 §6.3 的文件级拆分不同，这一组**不改 crate 数量也不改公开路径**，只把文件放回它该在的目录，是拆分的前置动作：

- **W1 `weights` 分四个域**：`llpk/`（`writer` `reader` `seg` `codec` `crc` `leb` `sink` `merkle` `envelope`）、`format/`（`safetensors` `gguf` `json` `zip` `torch`）、`crypto/`（`sha256` `keccak` `dsa` `ntt` `golay` `svd` `ledger`）、`quant/`（`rans` `lattice` `quant` `affine` `scale` `patch` `mode` `axes` `ir`）。收益：`lib.rs` 的 33 个 `pub mod` 收成 4 个域；风险低（纯移动 + `pub use`）。
- **W2 `weights` 四个文件名与内容不符，应改名**：`bin.rs` → `format/torch.rs`（内容是 pickle 栈式 VM，`bin.rs:241`）；`affine.rs` → `quant/desc.rs`（只有 `Quant` 描述符与校验，`affine.rs:14-24`，无仿射反量化算术）；`axes.rs` → `quant/artifact.rs`（自造 `AX01` 容器，`axes.rs:14`）；`ir.rs` → `quant/fused.rs`（扁平融合节点表，`ir.rs:27`）。
- **W3 `op` 分三组**：22 个 `*_f32.rs`/`*_i4.rs`/`*_i8.rs` → `src/ops/`；`gemv.rs`/`gemm.rs`/`elementwise.rs`/`unary.rs`/`reduce.rs` → `src/skeleton/`。`REGISTRY` 留在 `lib.rs`（它是公开面），已有 `builtin_matches_disk` 平价测试兜底。
- **W4 `cutile` 删第二条渲染器**：`substitute::render`（137 行）唯一生产调用点是 `builtin.rs:342` 的对拍检查；删后 `ast` 成为唯一路径（P1-4），同时删掉为该对拍而存在的测试（`ast.rs:313`）。
- **W5 `engine` 拆两文件**：`sequence.rs`（状态机）+ `scheduler.rs`（页预算）；两者已互不引用。
- **W6 `config::Key` 一张表**：`(key, file_key, env_suffix, kind, doc)` 一行一个键，四个方法各自投影一列（P2-3）。
- **W7 `annotation` 删族名分支**：`codegen.rs:211-374` 的 11 处同形分支 → 一张 `(族名 → 模块路径)` 表（P1-5）。
- **W8 根包归位**：`src/main.rs:17-18` 的 `LmHead` 注解类型移入 `nn`；`decode_f32`/`parse_input`（`:382-423`）移入 `weights`/`nn`；`src/bin/cutile.rs` 的排版移入 `cli`。

### 6.5 决策汇总（三档，附不做的事）

| 档位 | 项 | 为何这一档 |
|---|---|---|
| **立刻**（零风险 / 纯删除） | 修 `pack::*` 13 处标签；`storage` 布局并入 `tensor`；`engine`/`report` 拆文件；`config::Key` 单表；四份缓存合一；删 `seg.rs:65-68` 与 `ccn.rs:138-142` | 不改语义、不改公开类型名，只移动与删死代码 |
| **有条件**（需前置） | `weights` 真拆 crate（等 29 个未跟踪文件提交）；`storage::Mmap` 接生产（等 `safetensors::File` 改后端）；`einsum`/`regress`/`proto` 等七条吸收**已执行**（ADR 0019，§11） | 前置项不满足时做，会与在途改动叠加放大冲突面 |
| **不做**（已给理由） | 合并 `diagnostic`/`dtype`/`annotation`/`shard`/`data`/`mem`/`tokenizer`；拆 `tensor`/`toml.rs` | 见 §6.1 与 §6.2.2；`kv`/`exec`/`einsum`/`audit`/`regress`/`checkpoint` 已按 §11 改判为合并，不再属本档 |

一条方法上的判定基准（供下一轮争论时直接用）：**合并的唯一正当理由是"同一知识存在两份实现或两份消费者争用"，不是"行数少"**。按这个基准，首轮只有 `storage`（重复步幅 + 零入边）与 `regress`（零入边 + 与 `tool` 同域）达标；`einsum`/`exec`/`data` 等"只有 1 个消费者"的成员按**这一条**不达标。§11 追加了 ADR 0019 的第二条理由（单消费者吸收），据此 `einsum`/`exec`/`kv`/`audit`/`checkpoint` 也被判为合并——两条理由的分工见 §10.5 末尾。

### 6.6 文件/函数级别的技术债核减（能真正删除基线的）

| 债务 | 现状 | 可消除方式 | 预计收益 |
|---|---|---|---|
| 4 份表达式缓存 | `gemv.rs:250`、`elementwise.rs:41`、`unary.rs:32`、`reduce.rs:31`，同一个 `OnceLock<Result<Formula, Error>>` | 抽一个 `FormulaCache<T: Expr>`，4 个类型变 1 个 + 4 个类型别名 | 代码 −60 行，债条目 −3 |
| `gemv` 输入校验 3 份 | `gemv_i4.rs:~100-135` 与 `gemv_i8.rs:93-123` 共享 20 行（连错误文案都相同） | 把校验提到 `Format` 的默认方法，格式只覆盖差异（偶组检查） | 代码 −25 行，语义单源 |
| 22 条 `Registration` | `op/src/lib.rs:108` 手写 22 行 `include_str!` 三元组 | 一个 `register_all!` 宏按 `ops/` 目录生成（已有 `builtin_matches_disk` 平价测试兜底） | 代码 −100 行，加算子只改文件 |
| 22 个 `forward` shim | `gemv_i4.rs:171`、`gemv_i8.rs:130`、`gemv_f32.rs:118` 等只做一次转发 | 宏 `#[model]` 顺带生成 `forward` 委托 | 代码 −40 行 |
| `codegen.rs:211-374` 11 处同形分支 | 每处 8 行，仅模块路径不同 | 一张 `(族名 → 模块路径)` 表 + 一个循环 | 代码 −80 行，`render_model` 的 nloc 233 → 显著下降 |
| `seg`/`Desc`/`Stored` 三层同构 | `seg.rs:16,119`、`reader.rs:28`，靠 `Desc::into_parts`（`:182`）桥接 | 保留 `Segment` 一个记录 + 一个借用视图 | 代码 −60 行，去掉一类转换 bug |
| 对齐/LEB/步幅/元素数 四处重复 | P2-7 | 分别收口到 `mem`/`weights::leb`/`tensor` | 真重复 −4 处 |
| `cutile` 双渲染 | `substitute.rs:33` 与 `ast.rs:88` 两条路径 | 删字符串路径（§6.4 W4） | 代码 −137 行 + 一个对拍测试 |
| `config::Key` 5 张表 | `key.rs:34-79` | 单表 + 四个投影（§6.4 W6） | 代码 −20 行，加键从 5 处变 1 处 |
| `storage::Layout::strides` | `storage/src/lib.rs:67` 与 `tensor/src/lib.rs:99` | 并入 `tensor`（§6.3 S2） | 代码 −60 行，少一份步幅实现 |

---

## 七、重复与技术债治理

### 7.1 真重复 vs 结构性相似（区分判据）

方法可复现：取 168 个 `crates/*/src/**/*.rs` 与根 `src/**/*.rs` 文件，去掉注释、空行与 `#[cfg(test)]` 之后的部分，逐行归一化空白，做 8 行滑窗，按窗口哈希跨文件比对；命中 **65 个跨文件重复窗口**（约 520 行量级）。窗口数 ≠ 行数，也 ≠ 块数：相邻窗口合并后，真正值得看的只有 9 对各 1–3 个块。这个密度对一个 3.8 万行生产代码的仓库来说很低——规模来自"广度"而不是"复制"。下表逐对判定。

| 对象 | 判定 | 理由（可回链） |
|---|---|---|
| `rmsnorm_f32` ↔ `softmax_f32`（11 窗口）、`rope_f32` ↔ `softmax_f32`（7） | **结构性相似，保留** | 都是"逐行/逐 token 标量参考"的骨架：`M`/`K` 常量 + 一个双重循环 + 一个测试辅助。抽出来的抽象会比重复更难读。 |
| `gemv_i4` ↔ `gemv_i8`（10 窗口） | **一半真重复** | `cost`/`sample`/`reference` 已由 `gemv.rs` 骨架统一；但 `forward` 与输入校验各写一份（P2 表第 2 行），**校验部分应抽**。 |
| `gemm_f32` ↔ `gemm_i8`（5）、`gemm_i4` ↔ `gemm_i8`（4） | **相似，保留** | `gemm.rs:4-6` 明确写了"每种格式保留自己的 cost/sample/reference"，理由成立。 |
| `weights/src/mode.rs:78-84` ↔ `weights/src/patch.rs:31-37` | **真重复（窗口漏检）** | 同一个 5 实参 `Segment::new(name, dtype, shape.to_vec(), quant.cloned(), data)` 构造调用写了两遍，只差外层 `out.push(..?)`。块长 7 行 < 8 行窗口，所以上面的检测器**按设计抓不到**；修法是一行 helper `Segment::from_stored(seg, data)`。 |
| `dequant_i8` ↔ `rmsnorm/softmax/embed`（各 3-4 窗口） | **结构性相似** | 共性只有"参数校验 + 错误文案"。 |
| `serve/src/proto/crc.rs` ↔ `weights/src/crc.rs`（5 窗口） | **真重复（但需先确认）** | 两份 CRC-32 表。`testkit` 已有一份"位级参考"作为对拍（`testkit/src/lib.rs:91-104`），三份里至少有一份是多余的。 |
| `train/src/snapshot/mod.rs` ↔ `weights/src/sink.rs`（7 窗口） | **真重复** | LEB 写入与游标读取的同一套逻辑（见 P2-7）。`snapshot` 已经在用 `weights::crc`（`train/src/snapshot/mod.rs:473`），却没有用 `weights::leb`。 |
| `op/src/lib.rs` ↔ `prof/src/lib.rs`（6 窗口） | **结构性相似** | 两处都在做"闭集投影/内嵌表"的展开，形状相同、数据不同。 |
| `cutile/src/report/witness/contract.rs` ↔ `train/src/store.rs`（6 窗口） | **需人工判定** | 两处都出现"输入形态 → 类型标签"的映射。若确认为同一规则，应抽到 `dtype`/`tensor`；若是两份不同的契约（一份是设备侧 ABI，一份是训练侧 lane），则应各自保留并在文档里点名"为什么像"。 |
| `json.rs` ↔ `config/src/toml.rs` | **不是重复** | 两种格式、两种文法；相似的是"拒绝重复键/前导零"这类策略，属于正确决策的一致性，而非复制。 |
| `annotation` 与 `op` 的 skeleton 名 | **不是重复，是冲突**（P1-6） | 两套词汇表不是投影关系，属于"同一概念两个事实源"。 |

### 7.2 债务核减路径（先删基线，再谈重构）

1. **先让基线可信**：修正 13 处 `pack::*` 标签（+1 处文档）与 `ARCHITECTURE.md:81` 的算子数、`weights` crate 头（P1-3、P1-8）。`README` 的成员表已在 §11 同步。这些是"零风险、纯删除/更正"的条目。
2. **再删结构性死代码**：`seg.rs:65-68` 空循环、`ccn.rs:138-142` 死变量。删掉后对应 nloc/ccn 基线要同步降（否则 `stale` 会报"已达标，删除基线条目"，正好逼着删条目）。
3. **再做"一个类型替代四个"**：表达式缓存（§6.6 第 1 行）、`seg/Desc/Stored`。这类改动**代码下降且质量上升**。
4. **最后改解析器**：`bin.rs:execute`（ccn 77）与 `zip.rs:parse`（ccn 36）属于外部格式解析，建议**不是**追求 `ccn<=5`，而是：
   - `bin.rs`：把固定宽度参数族（`X/M/J/G/r/j/K/q/h`）收成 3 张表，其余栈操作保留分支；目标是 77 → 25 左右，并把"这是 pickle 解释器"的事实写进文件名。
   - `zip.rs`：只把 ZIP64 哨兵判定收成一次查表，`zip64_extra` 的 `size` 校验是必须补的正确性问题（已发现 subfield 声明 8 字节却按 16 读取）。
5. **对 `ccn_max = 5` 的判断（重要）**：对纯函数这是极高标准，对**文法解析器**它不可达——`parse` 的 ccn 本质是"关键字/操作码个数"。建议把阈值按类分档（`ccn_max` 纯逻辑 5、`parser` 15、`dispatcher` 25），并把这条分档写进 `[gate]`；否则 251 条 ccn 基线会长期存在，棘轮变成噪声，反而削弱门禁的威慑力。**这属于"降低形式指标、提升真实约束质量"的双赢项。**

### 7.3 与在途改动的关系（git 取证）

- 282 条债基线**全部来自未提交的工作树**（`git show HEAD:lints/vocabulary.toml` 无 `[gate]`）。
- 93 个跟踪文件 +3 545/−10 272（`git diff --shortstat`），说明本轮改动是"净删除"——方向正确。上一版两处把 `+`/`−` 写反了，以本节为准。
- 但 `weights` 有 29/33 个 src 文件未跟踪，**其债基线对应的代码尚未进入版本库**；一旦提交，基线才有意义。建议把"提交"作为这批基线的验收条件，而不是继续叠加。

---

## 八、对标差距与改进路线

### 8.1 与 Burn（及同类框架）的逐维对比

数据来源：我方数字为本次实测；Burn 的规模数字取自本仓自己的 `docs/10k-analysis.md`（该文注明用 tokei 同口径实测）。

| 维度 | 我方（已确认） | Burn（公开事实/该文实测） | 判定 |
|---|---|---|---|
| 代码规模 | 38 530 生产行（不含内联测试） | 369 340 Rust 行（约 9.6×） | 我方是"机制最小集"，不是"产品全量"；Burn 的规模数字取自 `10k-analysis.md`，该文自己的明细已过时（P3-6），但 36.9 万这一量级结论不受影响 |
| 依赖 | 0 运行时 / 1 dev（criterion） | 1 141 个 lock 条目 | **我方显著领先** |
| Backend 抽象 | 数据驱动：`prof/devices/*.toml` + `prim/<vendor>` + `#[frontend]`，新增设备不改机制 | 泛型 `Backend` trait 塔 + CubeCL 运行时 | 路线不同：我方"配置+文本"，Burn"泛型+IR"；我方移植成本低，编译期类型安全弱 |
| 运行时分派 | 无分派（编译期表 + `DEVICES` 常量表） | 有 `Dispatch`/`Runtime` 层 | 我方无动态派发开销，但也无 JIT/动态形状 |
| 融合/装饰器 | **无**（ADR 0017 显式冻结 L2） | `burn-fusion` 做 kernel 融合 | 我方在 elementwise 链上有真实差距（每算子一次 kernel） |
| no_std | 未声明（依赖 `std`：`OnceLock`、`fs`、`process`） | 支持 no_std 子集 | 差距明确；但本仓定位是 host 侧编译，优先级可低 |
| 多后端可切换 | 3 个 vendor 的片段（cpu/intel/metal），**仅 1 条可执行路径（C/cc）** | 多后端均可用 | 真实差距在"可执行后端数量" |
| unsafe | 6 处关键字、1 个文件、feature 门 | 大量后端内部 unsafe（CubeCL/CUDA） | **我方显著领先** |
| 测试策略 | 1 037 用例、gradcheck、真实权重对拍、双源语义 | burn-backend-tests 约 69k 行对拍矩阵 | 我方"点位精、面窄"；缺后端矩阵与模糊测试常驻 |
| 架构门禁 | 可执行（成员/边/方向/所有权/错误边界） | 无同类等价物（公开资料未见） | **我方领先** |
| 性能证据 | **未知**：本次未跑真机 bench；仓库内只有 `span_parse` 一个 criterion 基准 | 有成熟 benchmark 套件 | 差距是"没有证据"，不是"性能差" |

### 8.2 剩余工作的唯一执行序

§9.2 与 §10.6 都是本节的切面：§9.2 挑"最小样板"，§10.6 展开"生成化/删除"这一维；**三者冲突时以本节为准**。

**第 0 梯队（已完成，不是待办）**：七条吸收链已按 B0–B5 执行（§11.5），收益只记成员与边（§11.6），**不冲抵**下面的行数减量；`kv` 的页数公式 4 → 1 是其中唯一的质量收益。

**第一梯队（1 人日以内，纯收益）**

1. 修 P0-1/2/3/4：`rans` 长度交叉校验、`bin` 奇偶校验、`checkpoint` 的 `tag(..)?`、`dsa` 的 `Result` 化。**代码净增 < 60 行，消灭 4 条崩溃路径。**
2. `mmap` 加 `#[cfg(unix)]`（或把 `weights` 的 `mmap` 依赖改为 `cfg(unix)` 条件），恢复 Windows CI。
3. 修正 13 处 `pack::*` + `mmap.rs:7` 的过时引用；同步 `README`（成员数、L4/L5 表、依赖图）与 `ARCHITECTURE.md:82` 的算子数。
4. 删 `seg.rs:65-68` 空循环与 `ccn.rs:138-142` 死变量。

**第二梯队（1-3 人日，"代码下降且质量上升"的双赢项）**

5. 4 个表达式缓存 → 1 个泛型 `FormulaCache<T>`。
6. `gemv` 输入校验上提为 `Format` 默认方法。
7. `op` 的 22 条 `Registration` + 22 个 `forward` shim 由宏生成。
8. `codegen.rs` 的 11 处同形分支 → 一张族名表。
9. 门禁按类分档 `ccn`（纯逻辑/解析器/分派器），并把"只准降"改成真正的棘轮（记录上一轮实测值，`value >= previous` 即失败）。

**第三梯队（1-2 周，结构性）**

10. `weights` 拆成 4 个 crate（S1），先移动零行为变化的 `crypto` 与 `codec` 两组。
11. 三个超大文件拆模块（S2 前三行）。
12. `reader.rs`/`seg.rs`/`codec.rs` 补负例测试（截断、单比特翻转、超长 rank、零维、`u64` 溢出边界）。

**第四梯队（明确"先不做"）**

13. 跨算子融合 / 运行时 IR：ADR 0017 已冻结 L2，且解冻条件（L2 证据 + 70% roofline）当前**没有数据**。**不建议现在做**——它会把 128 行/算子的机制复杂度变成编译器工程量。
14. `bin.rs:execute` 一次性降到 `ccn<=5`：**不建议**。pickle 是一个 35 条操作码的字节码分派器，强行表驱动会让"哪个操作码动栈"变得不可读。目标是 25 左右 + 补 panic 修复。
15. 把 `serve` 变成真 HTTP、`shard` 接真 collective、`tokenizer` 上 BPE：这三项是**产品化**而非**架构**，应在 P0 鲁棒性修完之后再排。

### 8.3 边际收益判断（明确建议停止的项）

- **继续压 `ccn` 到 5 的全局目标**：对 251 条基线里的解析器/分派器，收益递减且会诱导"把 `match` 拆成 5 个小函数"这种形式主义，**建议停止**，改为分档阈值。
- **继续给 `docs/lld.md`（2 410 行）追加内容**：它已经无法被校验，**建议冻结**，把新增设计写进 ADR（可评审的最小单元）。
- **把 `weights` 的 DSA/NTT 继续留在权重容器里做"完整性证明"**：密码学与容器是两个变更速率，**建议先物理分离**（即使暂时仍由 `weights` re-export）。

---

## 九、结论与下一步

### 9.1 结论

- **架构层是本仓最强的资产**：唯一一条依赖规则被 6 项可执行断言守住，所有权表做到"一个类型一个主人"，运行时第三方依赖为 0，unsafe 收敛到单文件单 feature。这些指标放在国际开源坐标系里属于上游。
- **质量层是"刚上秤"的状态**：自建门禁首次量化出 282 条超标（ccn 251 条），说明当前实现离自定标准仍有可观的系统性距离；且门禁本身有 5 处覆盖盲区（P1-7）与 1 处语义与文档不符（P3-5）。
- **鲁棒层出现了与架构水平不匹配的塌陷**：4 条 P0 全部在"不可信输入"路径上（rANS 分配、pickle 越界、checkpoint `expect`、DSA 切片），且都能用几行代码修好。这说明**"错误契约统一"没有自动变成"输入校验完整"**——类型系统保住了"错误怎么表达"，没保住"错误在哪里产生"。
- **交付面仍是 P0 切片**：22 个算子、1 条可执行后端、无传输/无集合通信/无 BPE/无融合；这不是缺陷，而是路线选择，但对外表述（"LLM runtime"）应与实际能力对齐。

### 9.2 推荐的最小样板切入点（如果要继续编码）

完整顺序见 §8.2（本节只是它的"最小样板"切面）。若只挑两件开始，就是 §8.2 第一梯队第 1 条与第二梯队第 5 条：

1. `weights` 四连修（P0-1..4）：既是安全收益，又能顺带把 `dsa`/`rans` 的输入校验写成"前缀长度契约"的样板——这正是其他格式解析器（`zip64_extra`、`gguf` 的 `start+len`）需要复制的模式。
2. `op` 缓存收口（4 → 1）：证明"泛型 + 泛型常量"能同时降低代码量与重复，而不引入运行期分派。
3. 这两步做完，`weights` 的债基线应从 84 条开始真实下降，门禁的棘轮才第一次"动起来"（注意它只拦"涨"，见 P3-5）。

### 9.3 明确不建议触碰

- `tests/architecture.rs` / `tests/contracts.rs` / `docs/ARCHITECTURE.md` 的三表结构：它们是本仓的核心治理资产，任何"简化"都会把可执行契约降级为散文。
- `diagnostic` / `dtype` / `op::einsum`（已并入 `op`）/ `prof::atom::Set`：单一事实源的样板。
- `op` 的 `Format` 骨架与"每格式一份 `forward`"：monomorphization 的选择是有代价但有理由的（`gemv.rs:1-26` 的取舍表），改成运行期 `match` 是性能倒退。
- `storage::mmap` 的 unsafe 岛：已经是教科书级的最小化，除非要做 Windows 支持（那时是 `cfg` 问题，不是 unsafe 设计问题）。

### 9.4 本次审计未覆盖 / 未知项（诚实边界）

| 项 | 状态 | 原因 |
|---|---|---|
| 真机性能、roofline 达成率 | **未知** | 本次未在目标设备上跑 `audit bench`；仓库内无实测数据文件 |
| Windows/macOS 以外的可移植性 | **未知**（Windows 判定为推断） | 无交叉编译目标 |
| 覆盖率百分比 | **未知** | CI 生成 lcov 但无阈值、无上传门禁；本次未跑 `cargo llvm-cov` |
| `docs/lld.md` 与代码一致性 | **未知** | 2 410 行未逐条核对 |
| `cargo deny` / `cargo machete` / `jscpd` 的当前结果 | **未知** | 需要联网安装 Node 工具链，本次只读了配置（`.jscpd.json` threshold 1.5%） |
| 每个函数的实际运行时调用频次 | **未知** | 无 profiler 数据；本报告只给静态 fan-in（`Error/Code/Result` 各约 780-807 次引用，`DType` 339 次） |

---

## 十、注解化 · 配置化 · 泛型化：剩余减量与二次合并评估

本节只回答两件事：**还有多少代码可以被"生成"掉**，以及**还有哪些合并能过证据关**。行数为 `wc -l` 实测；"净减"一律标为**推断**（按现状结构估算，不是已实现的结果）；每条判定都先给判据。

### 10.1 先定义"该被生成"，再谈重复（四条可判据）

| 判据 | 含义 | 本项目已满足的样例（证明判据可判） |
|---|---|---|
| A 同一知识两份实现且必须同步 | 改一处不改另一处就是 bug | `op::REGISTRY`（`op/src/lib.rs:108`）与磁盘上的 22 对 `.toml`/`.k`，靠 `builtin_matches_disk` 平价测试绑定 |
| B 新增一项要改 N 处 | N ≥ 3 即说明缺一层声明 | `config::Key` 加一个键要改 5 处（`config/src/key.rs:3-4` 自述） |
| C 已有 const 断言/平价测试证明二者必须一致 | 一致性是硬约束而非风格 | `const _: () = same_order(&KERNELS, &op::REGISTRY);`（`train/src/store.rs:589`） |
| D 字段是闭集且只在编译期需要 | 运行期不需要分支 | `Set<E, WORDS>` 的 `FITS` 容量证明（`prof/src/atom.rs:56-59`） |

判据之外的重复**不算债**。据此，`gemv` 每格式一份 `forward`、`crc` 的两份实现（`weights/src/crc.rs:44` 与 `testkit/src/lib.rs:91`）都不进本节的减量清单。

### 10.2 已经做对的部分（禁止为了"更统一"而回退）

| 机制 | 现状 | 为什么它已经是"注解的替代品" |
|---|---|---|
| `Closed` + `Set<E, WORDS>` | 6 个闭集共用一份契约（`prof/src/atom.rs:20-34`），位集唯一（`:44-151`） | 容量越界是编译错误，比任何宏生成的查表都强 |
| `dtype` 单一事实源 | `prof` 直接 `pub use dtype::DType` 并 `impl Closed`（`prof/src/atom.rs:313-328`） | ADR 0018 的落地样本：`F4` 只要加在 L1，所有层免费获得 |
| `KERNELS` ↔ `REGISTRY` 顺序断言 | `train/src/store.rs:498` 与 `:589-617`（const 版 `str` 比较） | 运行期零成本，且编译期就挡住漏注册 |
| `#[model]` 自检模块 | 每个算子生成 `__model_check`（`annotation/src/codegen.rs:382-441`） | 注解声明的 dtype/sample 与 `.k` 签名对不上就编不过 |
| `Sources::set` 的层级仲裁 | `config/src/source.rs:114-123` | "值 + 来源"是类型，不是散落的 `env::var` |

结论：**本项目的问题不是"注解化不足"，而是"注解只铺到了 op 层，没铺到它下游的三张平行表"**（见 §10.3 第 1-4 行）。

### 10.3 还能被表/泛型/宏吞掉的样板（按收益排序）

| # | 机制 | 现状与证据 | 生成/收敛方式 | 净减（推断） | 同步风险 |
|---|---|---|---|---:|---|
| 1 | 22 条 `Registration` | `op/src/lib.rs:108` 手写 22 行 `include_str!` 三元组 | `register_all!` 宏按 `ops/` 目录生成，`builtin_matches_disk` 兜底 | −100 | 低（平价测试已在） |
| 2 | **算子目录的第三份投影** | `op::REGISTRY`（`op/src/lib.rs:108`）、`train::KERNELS`（`store.rs:498`，已被 `:589` 的 const 断言绑定）、`nn`（`format.rs:88` + `nn/src/lib.rs:192,199` 硬编码两个算子名，**未与注册表绑定**） | 把 `format = "int4"` 写进 `ops/*.toml`，由注解生成 `WeightFormat::operator` 与两个常量 | −20 | **中**：`format.rs:168,195` 的断言要改指向 |
| 3 | 11 处同形族分支 | `annotation/src/codegen.rs:211-374`，每处 8 行仅模块路径不同 | 一张 `(族名 → 模块路径)` 表 + 一个循环 | −80 | 低（`render_model` 已有 `__model_check` 兜底） |
| 4 | 逐算子转接段 | `train/src/store.rs:619-774`，每算子解包后转 `op::*::forward` | 由 `ops/*.toml` 的输入签名生成，与 `KERNELS` 同源 | −100 | 中（涉及四个 `Inputs` 形状） |
| 5 | 4 份表达式缓存 | `gemv.rs:250`、`elementwise.rs:41`、`unary.rs:32`、`reduce.rs:31` 同一个 `OnceLock<Result<Formula, Error>>` | 一个 `FormulaCache<T: Expr>` + 4 个类型别名 | −60 | 低 |
| 6 | 3 族格量化器 | `lattice.rs:153/387/431` 同一骨架套三种块宽（24/8/3×E8） | 泛型 `Quantize<const DIM: usize>`，差异只剩 `pack`/`fit` | −80 | 中（数值须与现测试逐位一致） |
| 7 | `config::Key` 5 张表 | `config/src/key.rs:34-79`（`file_key`/`env_suffixes`/`kind`/`doc`） | 一行一个键，四个方法各投影一列 | −20 | 低 |
| 8 | `prof` 标量字段手写 | `prof/src/prof.rs:1104-1128` 逐字段 `dt`/`space`/`grade` 读取 | 字段表（同一函数 `:1091-1103` 的 `[(name, fn)]` 已是先例） | −40 | 低（`validate` 的跨记录不变量不在此列） |
| 9 | `ctype` 拼写 + 6 个校验 | `cutile/src/backend.rs:258-262` 的 3 臂 `match` 与 `:227-305` 的校验序言 | 3 行表 + 一个校验函数 | −40 | 低 |
| 10 | pickle 固定宽度族 | `weights/src/bin.rs:256-338`（`X/M/J/G/r/j/K/q/h` 十种编码） | 3 张 `(opcode, width, ctor)` 表 + 1 张闭包表；栈操作保留分支 | −60 | 中（须逐 opcode 对拍） |
| 11 | ZIP 字段与 64 位哨兵 | `weights/src/zip.rs:51-97` 偏移已是表；`:67-72` 是唯一实分支 | 哨兵收成一次 `Option` 查表；并补 `zip64_extra` 的 `size` 校验（正确性问题） | −10 | 低 |

合计**约 −600 行（推断）**。这十一条全部满足判据 A 或 B：不是"看起来像"，而是"两处必须同步"。

### 10.4 零引用簇：比合并更彻底的减量（本轮最强发现）

`weights/src` 下有 10 个模块，除 `lib.rs` 的 `pub mod` 一行外，**全仓无任何引用**（含 `tests/`、`src/`、其它 crate）；它们也全部出现在 `git status` 的 `??` 里——是**未提交的新增文件**。

| 模块 | 行数 | 仓内引用点 | 判定 |
|---|---:|---|---|
| `gguf.rs` | 753 | 只有 `lib.rs:70` 的 `pub mod gguf;` | 未接线 |
| `axes.rs` | 545 | 只有 `lib.rs:38`；自造 `AX01` 容器（`axes.rs:14`） | 未接线 |
| `lattice.rs` | 646 | 只被 `scale.rs:168` 引用（而 `scale` 自身未接线） | 链条死代码 |
| `ledger.rs` | 314 | 只被 `axes.rs:372` 引用 | 链条死代码 |
| `svd.rs` | 244 | 只被 `ledger.rs` 引用 | 链条死代码 |
| `ir.rs` | 321 | 只有 `lib.rs:79` 的 `pub use ir::{Graph, Node, Op};` | 再导出无消费者 |
| `scale.rs` | 196 | 只有 `lib.rs:50` | 未接线 |
| `golay.rs` | 179 | 只有 `lib.rs:42` | 未接线 |
| `mode.rs` | 128 | 只有 `lib.rs:64` | 未接线 |
| `patch.rs` | 108 | 只有 `tests/round_trip.rs:3,224-232`（测试是唯一消费者） | 仅测试可达 |
| **合计** | **3 434** | — | 占 `weights` 10 183 行的 **33.7%**（含文件内联测试） |

三条事实决定了处理方式，缺一不可：

1. **文档已经自相矛盾。** `lib.rs:68` 写"safetensors/gguf/bin 已接入"，但 `ARCHITECTURE.md:101` 的 `weights` 行只列 `Header/Entry/Reader/Loader/Archive/File/Quantized` 与 `Writer/Stored/Segment/Quant/Codec/Proof`，`gguf`/`axes`/`golay` 一个都没有。两处必有一处是错的（**已确认**）。
2. **ADR 只授权了"薄防腐层"，没授权"无消费者的防腐层"。** ADR 0018 明确把"GGUF / torch bin / C ABI 的 CType"定为外部码镜像并"保留为薄防腐层"；但防腐层的价值来自"保护某个调用方"，现在没有任何调用方。
3. **删/留必须先决定，再提交。** 一旦提交，这 3 434 行就进了版本库并进入后续审计面；不提交则零成本。删除前还要处理 `docs/lld.md`、`docs/weight-container-optimal-plan.md`、`docs/adr/0015`、`docs/adr/0017` 里的正文引用——**文件链接门禁不检查正文**，所以这一步不会有测试提醒。

三选一（按推荐序）：

| 方案 | 动作 | 代价 | 适用前提 |
|---|---|---|---|
| (a) 不提交 | 保留在工作树，按 ADR 0004 的占位口径记为"未落地" | 0 | 没有近期接线计划时**首选** |
| (b) 提交但隔离 | 用 `format/`、`quant/` 子模块 + `#[cfg(feature)]` 收拢，`ARCHITECTURE.md` 同步标"未接线" | 一份 feature 表 | 需要在同一分支上继续开发这些模块时 |
| (c) 接线后保留 | `safetensors::File` 增加 `gguf` 后端，`axes` 成为 `Writer` 的可选压缩策略 | 真实接线代码 + 对拍 | 已排期要做 llama.cpp 权重互通时 |

### 10.5 二次合并评估：新候选逐个定案

上一轮只审了"零/近似零入边"的成员。本轮补审**依赖签名相同**的簇（依赖签名相同说明它们本来就同层，是最容易被误合并的一类）：

| 候选 | 依赖签名 | 判定 | 依据 |
|---|---|---|---|
| `exec`(188) + `shard`(524) | 都只有 `diagnostic` | **不合并** | 无同一知识的两份实现：`exec` 是提交图与线程几何（`exec/src/lib.rs:31-145`），`shard` 是拓扑秩布局（`shard/src/topology.rs:118-152`）；仅共享"后端未实现"这一状态，应在 `ARCHITECTURE.md` 标为同一档占位，而不是合并 |
| `tokenizer`(125) + `data`(233) | `tokenizer`:`diagnostic`；`data`:`diagnostic`+`tensor` | **不合并** | `data/src/lib.rs:3-5` 已声明"不拥有分词（`tokenizer`）"；两者是"trait 契约"与"形状即校验"两个不同机制 |
| `data` + `kv` + `storage` | 都是 `diagnostic`+`tensor` | **不合并** | 批校验（`data/src/lib.rs:36-55`）、页账（`kv/src/lib.rs:228-288`）、布局（`storage/src/lib.rs:67`）三者互不引用，且 `storage` 的处理是并入 `tensor`（§6.3 S2），不是并进这里 |
| `serve`(1 461) + `engine`(1 303) | `serve`:`engine`+`tokenizer`(+dev `testkit`) | **不合并**，但删一份规则 | `engine/src/lib.rs:285-287` 的 `fits` 与 `serve/src/lib.rs:168-169` 的 `admit` 把"这条序列还能不能开工"写了两次；让 `admit` 直接调用 `engine` 的判断即可，crate 边界是对的 |
| `safetensors`(743) + `bin`(626) 与 `gguf`(753) | 同一 crate 内 | **合并为 `format/` 域 + 一张 dtype 映射表** | 三套 dtype 词表并存：`dtype::DType` 22 个、`bin::Kind` 9 个（`bin.rs:21,45-52`）、`gguf::Kind` 15 个（`gguf.rs:26,42-44`）。ADR 0018 允许镜像，但要求可检验；现在**没有任何断言**保证三者一致（`bin.rs:61` 把 I8 与 U8 都映射成 `DType::I8`，只靠注释解释） |
| `weights::ir`(321) → `exec` | — | **不合并，先降级** | 仓内出现三个同名 `Graph`/`Node`：`weights/src/lib.rs:79`、`exec/src/lib.rs:31-136`、`op::Op`。`ir` 的真实内容是"融合节点表 + 段编码"（`ir.rs:217` 的 `to_segment`），应改为 `exec::Graph` 的一个编码器，而不是自带一套图类型 |
| `regress`(337) → `tool` 或 `audit` | `regress` 只有 `diagnostic` | **仍 → `tool`**（维持 §6.2.1） | 新证据：`audit` 的依赖有 7 个（含 `proc-macro` 的 `annotation`），并入会让 `benches` 的编译链变重；`tool` 只依赖 `config`+`diagnostic`。语义上 `Judgement` 与 `audit::Verdict` 是刻意分开的两把尺子（`regress/src/lib.rs:151-153`），合并必须保留原名 |
| `train::snapshot::varint` → `weights::leb` | — | **真删除**（维持 §6.2.3） | `train/src/snapshot/mod.rs:756,812` 与 `weights/src/leb.rs:11,55` 是同一 LEB128 + 同一"拒绝非规范形式"规则的两份实现 |
| `storage` → `tensor` | — | **真合并**（维持 §6.3 S2） | 唯一既有"零生产入边"又有"同一规则两份实现"（步幅）的成员 |

**结论（可直接引用）**：按"同一知识两份实现"这一条判据，crate 级合并的边际本轮基本耗尽——新增证据只支持 1 个新合并（`safetensors`+`bin` 的 dtype 映射收口）与 0 个新 crate 合并；其余 8 个候选都因"没有同一知识的两份实现"被驳回，且每个都写了驳回理由。

**该结论已被 §11 部分取代**：本节只认"真重复"一种合并理由，§11 追加了第二种——**单消费者吸收（包装合并）**，据此把 `checkpoint`/`audit`/`regress`/`exec`/`kv`/`einsum`/`proto` 七条链判为合并（成员 33 → 26，**已执行**，见 §11）。分工是：§10.5 回答"合并能否顺手消掉重复"，§11 回答"合并能否减少成员与边"。剩下的**行数**减量仍必须来自 §10.3 的生成化与 §10.4 的删除。

### 10.6 净收益与执行顺序

本节只排 §10.3/§10.4 的**减量**项；总序与优先级以 §8.2 为准，本表是它在"生成化/删除"这一维上的展开。

| 阶段 | 动作 | 行数变化（推断） | 风险 | 前置 |
|---|---|---:|---|---|
| P0 | §10.4 决定 3 434 行的去留（推荐 (a) 不提交） | −3 434 或 0 | 低 | 无（未跟踪，不产生 diff） |
| P1 | §10.3 的 1/3/5/7（宏、族表、泛型缓存、单表 Key） | −260 | 低 | 无 |
| P2 | §10.3 的 2/4（op 声明生成 `nn` 与 `train` 的平行表） | −120 | 中 | 需先给 `ops/*.toml` 加 `format` 字段 |
| P3 | §10.3 的 6/8/9/10/11（格量化泛型、`prof` 字段表、`ctype` 表、pickle 表、ZIP 哨兵） | −230 | 中 | P3 内每条都要逐位对拍 |
| 合计 | — | **≈ −4 000 行**（3 434 + 610），约占 38 530 生产行的 **10.4%**；crate 数 −1（`storage`，未做）；成员另 −7（`exec`/`kv`/`einsum`/`audit`/`regress`/`checkpoint`/`proto`，**已执行**，见 §11；不改变本表行数） | — | — |

两点必须同时做，否则减量会变成"删代码但丢证据"：

1. **债务基线同步降**：`Gate::check`（`tool/src/gate.rs:130-135`）对"已达标"的条目要求**删除基线条目**，因此 §10.3 每落地一条，`lints/vocabulary.toml` 的对应行必须同批删除，否则门禁会反过来报错。
2. **先提交再生成**：§10.3 的生成化会重写 `op`/`train`/`nn` 三处，与 `weights` 那 29 个未跟踪文件互不相干，但**不要在同一提交里做**——一次提交只包含"生成"或"删除"中的一种，回滚才有意义。

### 10.7 明确不做的事（防止为对称牺牲优势）

- **不把 `Set<E, WORDS>` 换成 `Vec`/宏**：`FITS` 的编译期容量证明（`prof/src/atom.rs:56-59`）是真实优势，换成运行期容器是净损失。
- **不给 `weights` 的容器解析上 proc-macro**：字段由外部格式规定（ADR 0018），宏只会把"字段表"藏进宏体，`parse_header` 的可读性反而下降。
- **不为 ccn 冲 5 而拆 `bin::execute` / `zip::parse`**：复杂度来自问题域（pickle 35 条 opcode、ZIP 中央目录），目标应是 `77 → 25` 量级并保留一张可核对的表，不是形式上的低分。
- **不合并 `shard`/`tokenizer`/`data`**：四条判据（§10.1）一条都不满足；"crate 少"不是架构指标。
- **`exec`/`kv` 已按 §11 改判为合并**（ADR 0019 的第二条理由：单消费者吸收）。这不是标准松了，而是判据不同：它们满足"只有一个生产消费者 + 被吸收方不含第二份知识 + 吸收后方向合法"，但不满足 §10.1 的"真重复"四条。上一版本节与 §11.2 直接冲突，此处已改。
- **不删 `patch.rs`**：它有真实消费者（`tests/round_trip.rs:224-232` 的增量补丁往返），只是消费者是测试；应补一条生产路径或降级为 `testutil` 的邻居。

---

## 十一、七条吸收链：已落地（`checkpoint→train`、`audit→cutile`、`regress→tool`、`exec→engine`、`kv→engine`、`einsum→op`、`proto→serve`）

### 11.0 一句话结论：本节已不是方案，是执行记录

七条链**已全部执行完毕**（前六条按给定顺序，第七条 `proto→serve` 随后执行），不是"建议"也不是"计划"：

| 项 | 首轮基线（§2） | 现在（本轮实测） | 验证方式 |
|---|---:|---:|---|
| workspace 成员 | 33（32 + 根包） | **26（25 + 根包）** | `ls crates` 25 个 `Cargo.toml`；`tests/architecture.rs` 层表集合 = 成员集合 |
| 成员间依赖边 | 106（含 15 `+dev`） | **81（含 13 `+dev`）** | `docs/ARCHITECTURE.md:38` 与 `cargo metadata` 逐条相等（断言见 `tests/architecture.rs`） |
| 根包依赖 | 18 | **15** | 根 `Cargo.toml` `[dependencies]` 去掉 `audit`/`exec`/`kv` |
| 生产源文件（`crates/*/src`） | 166 | **166** | 文件只是换位置，数量不变 |
| 生产代码行（`crates/*/src`，含内联测试） | 53 250（首轮口径） | **53 329**（实测） | 前六条净 +75（注释与一个 12 行 helper），第七条净 +4（`pub mod proto` 与 `use`），合计 +79，**没有删除任何生产逻辑**。上一版写 54 869 是**漏计** `prim/build.rs` 622 与 `dump_bin.rs` 40（55 535 − 662 = 54 873）；而 39 721 是把根入口 1 544 重复计入（38 530 + 1 195）。两个数各自有各自的错因，现都按逐文件截断重测 |
| 债务键 | 282 | **282** | 34 条改路径（路径即身份）：`report` 20 + `snapshot` 8 + `einsum` 5 + `regress` 1；1 条 `lines` 按实测校正 |
| 测试目标 | 72 | **58** | 少掉 7 个 crate 的 lib + doc-test 目标 |
| 执行用例 | 1 037 通过 / 0 失败 / 1 忽略 | **1 037 通过 / 0 失败 / 1 忽略** | `cargo test --workspace --all-features`，退出码 0 |

最后一行是本轮最重要的一条事实：**七条合并移动了 ~13 000 行代码，一条测试都没丢，也没新增**。这意味着"合并"在这里确实是**纯包装**——它换来的是更少的成员与更少的边，不是更少的代码。谁把"合并六个 crate"当成"减量"来理解，都会在第 11.6 节碰壁。

### 11.1 落点表：每条链最终落在哪

| 链 | 消失的目录 | 新位置 | 该 crate 里"拥有"声明的第一句话 |
|---|---|---|---|
| `checkpoint → train` | `crates/checkpoint/` | `crates/train/src/snapshot/mod.rs` | "检查点文件：`Snapshot` 把一次训练恢复所需的事实与权重条目表编成字节" |
| `audit → cutile` | `crates/audit/` | `crates/cutile/src/report/{mod,doctor,verify,bench,cache,execution,common}.rs` + `report/witness/` | "`report`：把一次测量变成可复核的报告……本子模块不是机制" |
| `regress → tool` | `crates/regress/` | `crates/tool/src/regress.rs` | "回退判定：把'实测值 vs 基线'变成一条可执行的结论" |
| `exec → engine` | `crates/exec/` | `crates/engine/src/exec.rs` | "提交与执行：`Stream`、`Graph`、`Node`、`Launch` 与 `Executor` 契约" |
| `kv → engine` | `crates/kv/` | `crates/engine/src/kv.rs` | "分页 KV 缓存的簿记：`Page`、`Table`、`Cache`" |
| `einsum → op` | `crates/einsum/` | `crates/op/src/einsum/{mod,expr,extents,formula}.rs` | "同一份求和形式只被解析、校验一次……" |
| `proto → serve` | `crates/proto/` | `crates/serve/src/proto/{mod,frame,decoder,crc}.rs` | 线协议帧格式（`Envelope`/`Frame`/`Decoder`）现在是端点契约的 `proto` 子模块 |

落点选的都是**存活方的子模块**，而不是把两个 crate 的 `lib.rs` 拼在一起：`engine` 的 `Sequence`/`Scheduler` 与 `exec`/`kv` 分成三个文件（`lib.rs` 769 → 三个文件），`cutile` 的机制与报告分成两片（`report/` 13 个文件），`op` 的算子目录与求和形式分成两个目录。这是"合并不等于糊在一起"的具体做法。

### 11.2 逐条裁决与执行结果

| # | 链 | 合并本身消掉的重复 | 附带修掉的耦合 | 裁决 |
|---|---|---|---|---|
| 1 | `checkpoint → train` | 无（编码仍是独立实现） | 7 条函数级键 + 1 条文件级键改路径；文件级 `lines = 1674` 校正为实测 1 611 | **已合并** |
| 2 | `audit → cutile` | `crate::witness`/`crate::common`/`crate::doctor` 等 **20 处跨模块硬编码路径**在合并瞬间暴露，改为 `crate::report::…` | 20 条债务键改路径；`audit` 的自述从"独立 crate"改为"报告子模块" | **已合并** |
| 3 | `regress → tool` | 无 | 1 条债务键；文档里"判定"这一族（`Gate` 与 `Baseline`）现在同 crate | **已合并** |
| 4 | `exec → engine` | 无（SPI 未删，见 §11.4） | 边 −4；`engine` 不再需要外部 `exec` 依赖 | **已合并** |
| 5 | `kv → engine` | **页数公式 4 → 1**：新增 `pages_for(tokens, page_size)`，`pages_required`/`pages_needed`/`advance`/`needs_page` 全部改调它 | 边 −5；"谁在借页"的四个不变量现在同一文件内 | **已合并** |
| 6 | `einsum → op` | 无 | 5 条债务键；`op` 内部的 `String` 报错被抓出并修掉（见 §11.3） | **已合并** |
| 7 | `proto → serve` | 无（帧格式仍是独立模块，未删一行） | 2 条债务键改路径（`serve/src/proto/frame.rs`）；`testkit` 从 `proto` 的 dev 边平移到 `serve`；`pub` 路径变 `serve::proto::` | **已合并** |

"合并本身消掉的重复"这一列只有两条非空（第 2、5 条），且第 5 条是在合并**之后**才动手收口的——这说明一件事：**包装合并不自动带来质量收益，收益要靠合并后同一 crate 里"顺手的那一改"**。第 1/3/4/6 条如果就此收工，成员少了四个，重复一条没少。

### 11.3 执行中被测试抓到的四个真问题

这四条都不是"合并前的预判"，是做完之后测试直接红的现场记录。它们的共同点：**在旧成员形状下不可能被发现**。

1. **`crates/cutile/src/report/` 的 20 处 `crate::witness` / `crate::common` / `crate::doctor` 路径**。这些是 crate 内部的顶层模块路径，一旦文件从 `crates/audit/src/doctor.rs` 变成 `crates/cutile/src/report/doctor.rs`，`crate::` 就指向了 `cutile` 而不是报告层。第一次 `cargo check` 直接报 20 处 `unresolved module`。**这就是"物理边界"的真实内容**：`audit` 的独立性原本只由目录名维持，代码里没有任何东西阻止它引用机制层。
2. **`tests/architecture.rs` 的错误边界扫描集是手写的五个目录**。删掉 `crates/audit/src` 之后，`rust_files` 对不存在的目录返回空集合，那条断言会**照样绿**——成员少扫一个，测试不会告诉你。已改为**从层表派生**，并加了一条方向的断言：层表里每个成员必须扫出至少一个文件，否则失败。
3. **派生之后立刻抓到真实违规**：`crates/op/src/einsum/expr.rs` 的 `check_label` 返回 `Result<(), String>`。它在 `crates/einsum/` 时期**不在任何扫描范围内**（扫描集只有 `op`/`prof`/`prim`/`cutile`/`audit`），并入 `op` 后进入扫描域，当场变红。修法是拆开职责：`is_label(char) -> bool` 只回答闭集，错误文本由调用点按自己的上下文给（`Expr` 说"第几项"，`Extents` 说"哪个轴"）。
4. **`tests/contracts.rs` 的闭集检查是文本扫描**，我在 `op/src/einsum/mod.rs` 的说明注释里写下了它要搜的关键字面，测试立刻红。结论不是"改测试"，而是记一条 P3：**这套检查对注释敏感，白话解释会被当成代码**；修法是改注释措辞。

另外两条与合并直接相关的记账：

- 债务基线 `crates/checkpoint/src/lib.rs` 的文件级 `lines = 1674` 比实测 1 611 **多 63 行**。`Gate::check`（`gate.rs:124-135`）只拦"涨"、不拦"降"，所以文件缩了 63 行也没人被迫改基线。合并必须改键，顺手把值校正成实测——这是唯一能让这条陈旧基线消失的时机。
- `crates/tool/src/regress.rs` 的 `judge` 落进 `tool` 后触发一条新键（`ccn = 6`），说明**改路径不是纯改名**：函数名与编号一起决定身份，`regress` 时期的键名 `lib.rs:judge#0` 在 `regress.rs` 里仍然成立，但 crate 前缀换了，必须同步。

### 11.4 没做到的附带动作（诚实清单）

首轮我在 §11.3（旧编号）写过"没有这些就别合"。实际执行时**只有第 5 条兑现了**，其余按"可验证性"排序后留下一张待办表：

| 待办 | 为什么本轮没做 | 判据（做完就能验） |
|---|---|---|
| `exec` 的 `Launch`/`Node`/`Graph`/`Executor` 删除或标"占位" | 删它要先改 ADR 0004（"提交契约在 L4"），改 ADR 不是包装合并的一部分 | `engine` 生产代码里 `Graph`/`Launch` 引用数 = 0（现状已是 0） |
| `snapshot` 1 611 行拆 `format`/`decode` | 拆文件要同时改 7 条函数级键 + 1 条文件级键，与合并同批做会让 revert 不再原子 | 两个文件都 < 899 行；`cargo test -p llm-train` 不变 |
| `Facts` 吸收 `tape_nodes`/`weight_entries`，`resume` 校验 6 个事实 | **这是行为变更**（`resume` 会开始拒绝更多快照），必须独立提交 + 对拍，混进合并等于把一个语义改动藏进搬家 | `train::Checkpoint` 与 `snapshot::Facts` 字段数相等 |
| 补字节级**黄金向量**测试 | 现有断言只到 `bytes.len()`；写黄金向量要先把格式冻住，而格式正因"拆文件"待改 | 固定十六进制串成立，改格式必须同时改测试 |
| `weights::leb` 改 `pub` 并删 `snapshot` 的 `varint` | 两份 LEB128 的规范形式拒绝规则要逐条对齐（`weights/src/leb.rs:11,55` vs `snapshot/mod.rs` 的 `varint`/`Cursor::varint`），属于"零漂移"对拍，不该在搬家提交里做 | `snapshot` 里 `fn varint` 消失，字节输出不变 |

这张表是本节最有用的部分：它把"合并完成"和"质量完成"分开记账。把六条链读成"架构问题已解决"是错读——**它解决的是成员数与边数，不是重复与复杂度**。

### 11.5 执行序与每批的验证

| 批 | 内容 | 实际验证 | 结果 |
|---|---|---|---|
| **B0 门禁硬化** | `tests/architecture.rs` 错误边界扫描集改为从层表派生；`ARCHITECTURE.md` 三张表按新形状改写；新增 `docs/adr/0019-packaging-merge.md` | `cargo test --test architecture` | 6/6 绿；派生后立刻抓到第 11.3 节第 3 条的 `String` 报错 |
| **B1 零成本两条** | `regress → tool`、`exec → engine` + `kv → engine`（两条并在一次里做，避免两次改 `engine/src/lib.rs`） | `cargo test -p llm-engine -p llm-serve -p llm-tool` | 26 + 9 + 9 全绿；`engine` 三个文件 1 303 行 |
| **B2 `checkpoint → train`** | 目录搬家 + 8 条键改写 + `lines` 校正 | `cargo test -p llm-train` + `-p llm-tool --test gate` | 全绿；文件级键指向 `train/src/snapshot/mod.rs` |
| **B3 `einsum → op`** | 目录搬家 + `Bits` 改用 `prof::atom::Closed` + 5 条键改写 | `cargo test -p llm-op` | 159 通过；`String` 报错当场暴露并修掉 |
| **B4 `audit → cutile`** | 13 个文件搬进 `report/`、20 处 `crate::` 路径改写、20 条键改写 | `cargo test -p llm-cutile` + `cargo test --test contracts` | 全绿；`cutile` 9 469 行 / 26 文件 |
| **B5 `proto → serve`** | 4 个文件搬进 `serve/src/proto/`、`crate::` 改成 `crate::proto::`、`pub use self::…`、层表/边表/所有权表/README/债务键同步 | `cargo test --test architecture --test contracts` + `cargo test -p llm-tool --test gate` + 全仓测试 | 6/6 + 6/6 + 1/1 全绿；**1 037 通过 / 0 失败 / 1 忽略**，与合并前逐项相同 |

排序依据是**每次提交的爆炸半径**，不是重要性：B1 不碰债务键、不碰 ADR；B2 碰 1 条文件级键；B3 碰闭集词汇表；B4 同时改最大 crate 与 20 条键。B4 放最后，前面任一步失败都不会让最大的 crate 停在半合并状态。

### 11.6 收益核算：合并省的是"成员与边"，不是行数

| 项 | 变化 | 能不能算"代码下降且质量上升" |
|---|---|---|
| 成员 | 33 → 26（crate 32 → 25） | ✅ 审计面、`ARCHITECTURE.md`、§6.0 表同步缩短 |
| 依赖边 | 106 → 81 | ✅ 每条边都要在文档与 `cargo metadata` 两侧维护 |
| 根包依赖 | 18 → 15 | ✅ 二进制少 3 个 use 前缀（`proto` 从来不是根包的直接依赖） |
| 重复实现 | 页数公式 4 → 1 | ✅ 七条里唯一的**质量**收益（`proto→serve` 一条重复都没少） |
| 债务键 | 282 → 282 | ➖ 只是换路径，条目一个没少 |
| 生产代码行 | +79（`crates/*/src` 53 250 → 53 329：前六条 +75，第七条 4 行是模块声明与 `use`） | ❌ 合并**不减行**；减量仍在 §10.3（−610，推断）与 §10.4（−3 434 待决）。注：上一版把合计写成 54 869，是漏计 `prim/build.rs` 622 与 `dump_bin.rs` 40，详见 §11.0 |
| 测试 | 1 037 → 1 037 | ➖ 一条没丢，也一条没加 |

一句话：**七条链让项目少了 7 个 crate、25 条边、3 个根包依赖，代价是 79 行注释与 4 行模块声明**（53 250 → 53 329，生产逻辑一行未改）。这是一笔干净的架构交易，但它不解决"38k 行里有多少该被生成或删除"这个真问题。

### 11.7 合并后实测基线（与 §2 同一口径，可直接对比）

| 指标 | 合并前 | 合并后 | 证据 |
|---|---:|---:|---|
| `crates` 成员 | 32 | 25 | `ls crates` / `cargo metadata` |
| `crates/*/src/*.rs` 文件数 | 166 | 166 | `glob` 计数 |
| 其中 > 899 行 | 8 | **8** | `checkpoint/src/lib.rs` 1 611 → `train/src/snapshot/mod.rs` 1 611：文件没变短，只是换了 crate 前缀 |
| `crates/*/src` 行数 | 53 250 | 53 329 | `wc -l` 求和（+79 = 前六条的注释与 `pages_for` + 第七条的 4 行模块声明与 `use`） |
| `cargo test` 目标 | 72 | 58 | 7 个 crate 的 lib + doc-test 目标消失 |
| `cargo test` 用例 | 1 037 | 1 037 | 同上，退出码 0 |
| 债务键 | 282 | 282 | `lints/vocabulary.toml` `[[gate.debt]]` 计数 |

一条容易忽略的副作用：`> 899 行` 的文件数不变（8 个），但**分布变了**——`checkpoint/src/lib.rs` 的 1 611 行原样变成 `train/src/snapshot/mod.rs`，于是 `train` 一个 crate 里有 2 个超阈值文件（`store.rs` 1 530、`snapshot/mod.rs` 1 611）。合并不会让大文件变小，只会让"大文件"集中到更少的人手里。这正是 §11.4 把"拆 `snapshot`"列进待办的原因。

### 11.7.1 本批的实测证据（不是计划）

| 检查 | 命令 | 结果 |
|---|---|---|
| 架构契约（成员集合、边方向、所有权、错误边界） | `cargo test --test architecture --test contracts` | 6/6 + 6/6 通过 |
| 文档链接 + 棘轮门禁 | `cargo test -p llm-tool` | 27 通过（`regress` 的 6 个用例已计入） |
| 全仓测试 | `cargo test --workspace --all-features` | **1 037 通过 / 0 失败 / 1 忽略**，与合并前逐项相同 |
| 随迁测试没有变成死代码 | `cargo test -p cutile` 中的 `report::*` | 47 个用例确实执行（`report` 已在 `cutile/src/lib.rs` 声明为 `pub mod`） |
| 合并后二进制仍工作 | `cargo run --bin cutile -- doctor` | 退出码 0，协商/plan/布局各段正常输出 |
| 严格文档构建（CI `doc` job 的口径） | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` | 退出码 0（修复前 27 处断链，见下） |

第三条是关键：**用例数不变**，说明七次吸收都是搬家而非重写；如果某个子模块忘了在 `lib.rs` 里声明，用例数会掉而编译仍绿——这是合并类改动最需要在 CI 里盯住的假绿形态（`cargo test` 只数跑起来的用例，不数文件里的 `#[test]`）。

最后一条是本轮顺带修掉的既有缺陷，不属于合并本身，但会让 `doc` job 必红：全仓 **27 处** rustdoc 断链，两类成因——

| 成因 | 位置 | 修法 |
|---|---|---|
| 公开文档链到私有项（`pub(crate) mod` 或私有 `const`/`fn`） | `cutile/src/report/mod.rs:15`（3 处）、`report/witness/mod.rs:37-38`（5 处）、`report/witness/evidence.rs:51`、`cutile/src/ast.rs:17,160`、`weights/src/bin.rs:4`、`weights/src/axes.rs:6`、`weights/src/ir.rs:159,213`、`tool/src/call.rs:3`、`tool/src/strip.rs:3` | 改成代码块（这些名字是内部实现，不该出现在公开 API 的链接图里） |
| 正文里的方括号被当成链接 | `weights/src/ntt.rs:3` 的 `Z_q[x]`、`weights/src/keccak.rs:1` 的 `Keccak-f[1600]`；`op/src/*.rs` 的 `[`elementwise`]`/`[`unary`]`/`[`reduce`]`/`[`einsum`]` 缺少 `crate::` 前缀（10 处） | 前者加反引号，后者补 `crate::` 前缀 |

这条值得单独记的原因是：`tests/architecture.rs` 与 `tests/contracts.rs` 都只扫源码文本，`cargo test` 也不会碰文档，于是**"新增一个私有子模块 + 在公开文档里链它"可以一路绿到 `doc` job 才失败**（而 `doc` job 在 CI 里排在 `test` 之后）。

### 11.8 重议条件（写死，避免合并变成永久决定）

| 链 | 什么时候应重新拆开 |
|---|---|
| `checkpoint → train` | 检查点需要跨语言/跨工具读写（外部训练器产出 `LLMC`）时：格式需要独立版本节奏 |
| `audit → cutile` | `audit` 需要独立发版或独立 CI 目标（例如只跑 verify 的 GPU 机器）；或 `report` 开始出现算子名 |
| `regress → tool` | 出现第二个非 CI 消费者（例如在线回归服务） |
| `exec → engine` | 第一个真后端落地：若 `Executor` 的实现者必须在 L4，契约必须拆回 L4，否则只能依赖 L6 |
| `kv → engine` | 出现第二种调度策略、各自持有页账（prefix cache 与 continuous batching 分离）时，页账回到 L5 |
| `einsum → op` | `op` 因算子数量需要再拆（`op-decl` / `op-model`）时随 `op-model` 走；或出现绕过 `op` 直接用 `Formula` 的消费者 |
| `proto → serve` | 出现第二个传输消费者（另一个 crate 要读同一帧格式）时：帧格式必须回到 L5，否则第二个消费者要反向依赖 `serve` |

### 11.9 与 §6/§10 旧判定的关系

§4 与 §6.0 已按当前成员重写；§5 与 §10 的判断文字保留首轮措辞，读法如下：

| 旧结论位置 | 旧判定 | 现在的读法 |
|---|---|---|
| §6.0 全量裁定表（26 行） | — | **已重写**：只列当前 25 个 crate + 根包；七个已消失成员在表下单列"追溯"，不再混在"保持/合并候选"里 |
| §6.1 `audit↔cutile`、`einsum↔op`、`kv↔engine`、`checkpoint↔train` | 不合并 | 已合并；边界理由仍然成立，只是被"单消费者吸收"这一新类别覆盖 |
| §6.2.2 `exec → engine` | 驳回 | 已合并；`Launch`/`Graph` 仍无人使用（§11.4 第 1 行） |
| §6.5「不做」档 | 含 `kv`/`exec` | 两者已移出 |
| §10.5 结论 | "0 个新 crate 合并" | 数字改为 6，理由是**产品裁决**而非新证据 |
| §10.6 合计 | 成员另 −6（待执行） | 已执行 |

### 11.10 文档本身的自检（本次修订改了什么）

这一小节回答"文档为什么还没优化到位"，并留下可复核的改法。上一版的问题是**叠加式改判**：第十一节推翻了前文的判定，但前文的数字、路径与执行序没有回填，读者必须自己拼三处才能回答"这个成员最后怎么处置"。

| 上一版的问题 | 本次改法 | 证据 |
|---|---|---|
| 规模数字互相矛盾（57 075 / 39 721 / 54 869） | 全部按当前树重测并给出组成式：55 535 = 53 329 + 622 + 1 544 + 40；38 530 = 36 673 + 622 + 1 195 + 40；上一版 54 869 漏计 build.rs 与示例、39 721 重复计入根入口，错因写在 §11.0 与 §11.6 | `wc -l`、逐文件截断求和 |
| §4 逐 crate 表仍是 32 成员（含 6 个已消失 crate） | §4.1/§4.2 按 26 个 crate 重写，被吸收能力并入宿主行 | `cargo metadata`、`crates/*/src` 计数 |
| §6.0 全量覆盖表 32 行、6 行标注"已并入" | 换成 26 行**全量裁定表** + 7 行"已消失成员"追溯，成为成员处置的唯一入口 | `cargo metadata` 入边/出边、`wc -l` |
| §10.7 写"不合并 exec/kv"，与 §11.2 正面冲突 | 拆成两行：`shard`/`tokenizer`/`data` 维持驳回；`exec`/`kv` 按 ADR 0019 已合并，并说明是**判据不同**而非标准放松 | `docs/adr/0019-packaging-merge.md` |
| 三套执行序（§8.2 / §9.2 / §10.6）互不引用 | §8.2 声明为唯一执行序，§9.2 与 §10.6 改为它的切面；§11.5 明确是**已执行的历史批次** | 本节 B0–B5 |
| P0-3 的两处行号（原写 `snapshot:649`）已漂移 | 改为 `:651`，并说明合并时文件头 +2 行；`tag` 改为 `:895-904` | `train/src/snapshot/mod.rs:651,895` |
| `P1-8` 仍写 README 漂移 | 标注"已修"（`README.md:5` 现为 26 成员），只剩 `ARCHITECTURE.md:81` 与 `weights` crate 头 | `README.md:5` |
| 门禁本身的"删成员仍变绿"只记在 §11 | 提升为 P1-9，并给修复后的断言位置 | `tests/architecture.rs:149-159` |
| 重复检测的 65 个窗口没有方法说明 | 写明口径（8 行滑窗、去注释与测试、168 文件），并补一对**窗口漏检**的真重复 | `weights/src/mode.rs:78-84` ↔ `patch.rs:31-37` |
| 311 > 282 像算术错误 | 加口径说明：29 条条目同时登记 ccn 与 nloc | `[[gate.debt]]` 计数 |
| §3 第 4 条（"中间产物全可见"）缺第 4 段产物 | 补 `ast::Node` 与"第二渲染器"缺口 | `cutile/src/ast.rs:30-59`、`substitute.rs:33` |
| 测试代码行 19 512 与"53 325 − 36 669 = 16 656"自相矛盾 | 全部按截断求和重算：内联 16 656 + 外部 2 507 = 19 163，比值 0.50；并注明上一版 17 005 的来源错误 | 逐文件截断求和 |
| 第七次吸收（`proto → serve`）后未回填前文 | 成员 27 → 26、边 83 → 81、测试目标 60 → 58、`serve` 331 → 1 461 行、`crates/*/src` 53 325 → 53 329，逐处回填并补 §11.1/§11.2/§11.5/§11.6/§11.8 七处 | `cargo metadata`、`wc -l`、`cargo test` 目标计数 |
| `cargo doc -D warnings` 在全仓 27 处断链（CI `doc` job 必红） | 私有项改代码块、缺 `crate::` 前缀补齐、正文方括号加反引号；成因表写进 §11.7.1 | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` 退出码 0 |
| 工作树数字 `+3 433/−5 714` 在 §2.3 与 §7.3 符号相反 | 统一为 `+3 545/−10 272`（93 个文件），并注明上一版写反 | `git diff --shortstat` |

仍未做（诚实边界）：真机性能、覆盖率、`cargo deny`/`jscpd` 实测、`docs/lld.md` 一致性——与 §9.4 相同，不因为这次改写而变成"已知"。

---

*本报告首轮为只读审计；第十一节记录的是在其后按指令执行的六条合并，改动集合见 §11.1 与 §11.5，验证见 §11.7。所有结论均可回链到上文给出的 `文件:行`。*

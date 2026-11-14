# 全仓架构与质量审计报告

- 审计对象：`/Users/fei/code/llm`（Rust workspace，`crates/*` 26 个成员 + 根包 `llm`）。首轮审计时为 32 个成员，七条合并已在第十一节执行
- 审计方式：只读静态审计 + 在当前工作树上真实执行测试与门禁；未修改任何代码
- 审计时点：2026-11-13 07:30 前后的工作树（含大量未提交改动）；**复核时点 2026-11-14 06:47，HEAD `4e363270`，工作树干净**（`git status --short` 只剩 `?? .junie/`），复核结论见第十二节
- 口径声明：
  - "已确认" = 本次直接读源码/执行命令得到，可回链到文件行；
  - "推断" = 由已确认事实按语言/平台规则推出，未在目标环境实测；
  - "未知" = 本次没有证据，明确标出而不猜测。

---

## 零、读者须知：版本标记与阅读顺序

本报告有两层时间：**第一至十节是首轮只读审计的现场记录**（成员 33 = 32 个 `crates/*` + 根包），**第十一节是随后按指令执行的七条吸收链的记录**（成员 26 = 25 + 根包）。为避免"同一句话两个数"，按四条规则读：

1. **规模数字以 §2.1 为基线、以 §12.2 为现行值**：§2.1 记录 2026-11-13 工作树（生产 38 530 行 / 170 个源文件 / 1 037 用例 / 58 个测试目标），HEAD `4e363270` 复核为 40 478 行 / 175 个源文件 / 1 083 用例 / 62 个测试目标；§11.6/§11.7 只是合并当时的快照，已被 §12.2 取代。§12.2 之后还有三轮代码改动（§15 五处崩溃路径、§16 平台门禁与文档漂移、§17 四条残余长度缺陷），**最新一次实测见 §18.6**：60 个测试目标 / 1 066 通过 / 173 个 `crates/*/src` 文件 / 55 137 行 / `[[gate.debt]]` 251 条 / 成员 25 / 边 75（`+dev` 11）。
2. **成员表以 §4.1 与 §6.0 为准**：两张表都已按合并后的 26 个 crate 重列；被吸收的七个成员在 §6.0 末尾单列"已消失成员"，追溯不丢。
3. **判断文字保留首轮措辞**：凡被 §11 改判的条目都带「§11 改判」或「已并入」标记；没有标记的，就是仍然成立的结论。
4. **配置层与语料的结论以 §13/§14 为最新，A1 的落地见 §13.11，A2 的落地见 §13.12，解析器退场的落地见 §14**：`crates/config` 的处置在 §4.2 / §6.0 / §6.4 W6 / §10.3 里是旧判定（内部整合）。§13 先把它拆成"设置层并入 `cutile`"与"语法层怎么办"两条独立裁决；A1 落地了前者，A2 否决了 §13.6 的"改名 `llm-toml`"写法，改为把语法层并进 `diagnostic::toml` 并删除成员——**`crates/config` 今天已不存在**，`llm-config` 也不再是成员。
5. **路径换算**：第一至十节的 `文件:行` 回链保留首轮路径（便于与当时的审计记录逐条对照）。读到下列旧路径时按下表换算：

| 旧路径（首轮） | 当前路径 | 证据 |
| --- | --- | --- |
| `crates/audit/src/*` | `crates/cutile/src/report/*` → `crates/cutile/src/*.rs` | 合并时先落 `report/`；`87ad7486` 又删掉 `report/` 与 `witness/` 两层目录，现为平铺文件（§12.3） |
| `crates/checkpoint/src/lib.rs` | `crates/train/src/snapshot/mod.rs` | 1 609 行 + 2 行文件头 = 1 611，其余原样 |
| `crates/regress/src/lib.rs` | `crates/tool/src/regress.rs` | `Judgement` 与 `Gate` 现在同 crate |
| `crates/exec/src/lib.rs` | `crates/engine/src/exec.rs` | `Stream` 的唯一生产者是 `engine` |
| `crates/kv/src/lib.rs` | `crates/engine/src/kv.rs` | 页账与页数公式现在同一文件树 |
| `crates/einsum/src/*` | `crates/op/src/einsum/*` | 5 条债务键已随路径改写 |
| `crates/proto/src/*` | `crates/serve/src/proto/*` | 4 个文件 1 126 行随迁，`pub` 路径由 `proto::` 变为 `serve::proto::` |

两条例外：`crates/audit/src/witness/` 先变成 `crates/cutile/src/report/witness/`，随后在 `87ad7486` 中拆平为 `cutile/src/{witness,device,probe,reference}.rs`——**`report/` 与 `witness/` 两层目录在 HEAD 已不存在**，本报告里凡以 `cutile/src/report/…` 开头的引用只保留"文件级"可回链（行号需重锚，§12.5）；`crates/checkpoint/src/lib.rs` 里的 `checkpoint::` 错误位标签改成了 `train::snapshot::` 前缀（`CONTEXT` 常量）。

---

## 一、总体判断与评级

**一句话定位**：这是一套"架构契约与依赖治理已经达到工业级、但运行时可交付面仍处在 P0 垂直切片"的框架——它的工程纪律（可执行架构门、单一事实源、零运行时依赖、单一 unsafe 岛）明显高于其功能完成度（仅一条 C 后端、无传输层、无集合通信后端、无 BPE）。

| 维度 | 评级 | 分数 | 核心理由（可回链） |
| --- | --- | ---: | --- |
| 架构契约与依赖治理 | A | 95 | 分层规则由 `tests/architecture.rs` 对 `cargo metadata` 强制执行，6 项断言全绿；所有权表逐类型落到 crate |
| 依赖最小化 | A+ | 99 | **默认构建第三方运行时依赖 0**；`--all-features` 下 `dtype` 打开 `serde`/`bytemuck` 两个可选依赖（`crates/dtype/Cargo.toml:12-21`）。`Cargo.lock` 101 个包中 26 个是本仓成员，其余 75 个只由两条链拉入：`benches`→`criterion`（dev）与 `dtype`→`serde`/`bytemuck`/`serde_json`（§12.2） |
| unsafe 治理 | A | 96 | 全仓 unsafe 关键字只出现在 `crates/storage/src/mmap.rs`（1 个 `unsafe extern`、2 个 `unsafe impl`、3 个 `unsafe` 块），模块级 `#![allow(unsafe_code)]` + feature 门 + 架构说明 |
| 正确性验证 | B+ | 85 | 62 个测试目标 / 1 083 个用例全绿；有 gradcheck、真实权重对拍、双源语义；但负例集中在少数条目，部分格式测试可空转 |
| 复杂度与可维护性 | C+ | 65 | 自建门禁登记 **274 条存量超标**；最高 `ccn=77`；8 个文件超 900 行；`impl` 与分支集中在解析器 |
| 文档与代码一致性 | B | 78 | `ARCHITECTURE.md` 是"可执行文档"；README 的成员表与依赖图已随 §11 同步为 26 成员（原写已删除的 `pack`/`comm` 与 33 成员）；仍存：`ARCHITECTURE.md:81` 写"现有两条算子"，实际 22 条注册 |
| 运行时可交付面 | C | 60 | `engine::exec`/`shard`/`serve`/`tokenizer` 只有契约无实现；`engine` 只做页账不做批次张量；仅 C/cc 一个可执行后端 |
| **综合** | **A-** | **82** | 治理能力已经是"顶级开源框架的方法论水平"，交付面是"严谨的 P0 切片" |

关键事实：本报告发现的 P0 级问题**不是架构问题，而是"不可信输入"路径上的 4 处 panic / 大分配**——这恰好说明架构层守住了，鲁棒性层还没守。

---

## 二、关键事实基线表

### 2.1 规模与结构

| 指标 | 数值 | 证据 / 口径 | 等级 |
| --- | ---: | --- | --- |
| workspace 成员 | 26（25 `crates/*` + 根包 `llm`） | 根 `Cargo.toml:48` `members = ["crates/*"]`；`crates/*/Cargo.toml` = 25。首轮为 33（32 + 根包），差 7 见 §11 | 已确认 |
| 生产源文件 | 175 | 171 `crates/*/src/*.rs` + 2 根 `src/*.rs` + 1 `crates/prim/build.rs` + 1 `crates/weights/examples/dump_bin.rs`（首轮 170；+5 全在 `dtype` 拆成 6 文件与 `cutile` 报告层拆平，§12.3）；**另有** `crates/benches/criterion/span_parse.rs` 30 行（criterion 基准，不计入生产） | 已确认 |
| 外部集成测试文件 | 12 | `tests/*.rs` 2 + 各 crate `tests/*.rs` 10（`autograd` 1、`benches` 1、`dtype` 4、`fuzz` 1、`tool` 2、`weights` 1） | 已确认 |
| 生产代码行（含内联测试） | 57 925 | `crates/*/src` 55 718 + `prim/build.rs` 622 + 根入口 1 545 + `dump_bin.rs` 40；基准 harness 30 行另计（首轮 55 535，+2 390 来自 `dtype`/`data`/`cutile` 的四个新提交，§12.3） | 已确认 |
| 生产代码行（不含 `#[cfg(test)]`） | 40 478 | 逐文件在首个 `#[cfg(test)]` 处截断求和（`crates/*/src` 38 620 + 根入口 1 196 + `build.rs` 622 + 40）。首轮同法测得 38 481；上一版曾误报 39 721，成因见 §11.0；本次复核 +1 948，见 §12.2 | 已确认 |
| 测试代码行 | 20 272 | 内联 17 098 + 外部 `tests/` 3 174（12 个文件）。上一版写 17 005 与 `53 325 − 36 669 = 16 656` 自相矛盾，已按后者改正 | 已确认 |
| 测试/生产比 | 0.50 | 20 272 / 40 478 = 0.501 | 已确认 |
| `trait` 定义 | 32 | `trait [A-Z]` 全仓计数；其中 `pub trait` 30 | 已确认 |
| `impl` 块 | 390（其中 `impl X for Y` 160） | `^\s*impl[<( ]` 计数 | 已确认 |
| `const` 泛型参数 | 28 | `<const ` 计数（`Set<E, WORDS>`、`Quantized<GRAIN>`、`Dense<BLOCK>`、`SymI8<GRAIN>`、`key_gen_internal<K,L>` 等） | 已确认 |
| 泛型形参出现（粗计） | 897 | `<A>`/`<A, B>` 形态正则；含大量 `Vec<T>` 噪声，**仅作量级参考** | 推断 |
| `macro_rules!` 定义 | 5 | `prof/src/atom.rs:602` `constraint_builders!`、`:781` `inherent_names!`；`dtype/src/format.rs:94` `impl_from_str!`、`:116` `impl_serde!`、`dtype/src/blocked.rs:12` `blocked_accessors!`（首轮漏计后三个） | 已确认 |
| `#[test]` 属性 | 1 015 | 全仓计数 | 已确认 |
| 实际执行用例 | 1 083 通过 / 0 失败 / 1 忽略 | `cargo test --workspace --all-features`，**62** 个测试目标，退出码 0（首轮 72 → 合并后 58 → 复核 62，差 4 见 §12.2） | 已确认 |
| unsafe 关键字（代码） | 6 | 全部在 `crates/storage/src/mmap.rs:25,48,49,74,108,136` | 已确认 |
| 第三方运行时依赖 | 0（默认）/ 2（`--all-features`） | 默认构建下各 `Cargo.toml` 的 `[dependencies]` 只有 `path` 成员；`--all-features` 打开 `dtype` 的 `serde`/`bytemuck` 两个可选依赖（`crates/dtype/Cargo.toml:12-21`） | 已确认 |
| 第三方 dev 依赖 | 2（`criterion 0.8.2`、`serde_json 1.0.154`） | 根 `Cargo.toml:83`；`crates/benches/Cargo.toml` 的 `[dev-dependencies]`；`crates/dtype/Cargo.toml:23-25` | 已确认 |
| `Cargo.lock` 包数 | 101 = 26 成员 + 75 第三方 | `grep -c '^name = ' Cargo.lock` = 101；75 个第三方只由 `benches`→`criterion` 与 `dtype`→`serde`/`bytemuck`/`serde_json` 两条链拉入（`cargo metadata` 逐成员核对） | 已确认 |
| 数据文件（`.k/.toml/.cu/.metal/.c/.cl/.h`） | 89 | 排除 `Cargo.toml`；其中 `ops/` 22 对 `.toml`+`.k` | 已确认 |
| 已注册算子 | 22 | `crates/op/src/lib.rs:122`（`registry!` 宏展开成 `REGISTRY: [Registration; 22]`，§20.2） | 已确认 |
| ADR 数量 | 19 | `docs/adr/0001..0019`（0019 记录七条吸收链） | 已确认 |
| 文档总量 | `ARCHITECTURE.md` 249 行、`lld.md` 2 410 行、`lints/vocabulary.toml` 1 281 行 | `wc -l` | 已确认 |

**时点说明（避免把本表当现行值）**：本表是 HEAD `4e363270`（§12）的快照。此后三轮改动让其中四行发生变化——
`生产源文件` 175 → **172**、`生产代码行（含内联测试）` 57 925 → **55 221**（`crates/*/src` 口径）、
`实际执行用例` 1 083 / **62** 目标 → **1 066 / 60**、`存量债条目` 274 → **248**。
差额全部来自 §13（`crates/config` 删除）、§14（运行期 TOML 解析器退场）、§15–§17（崩溃路径与长度校验）。
**引用规模数字时请用 §17.6 的实测值**；本表保留原值是为了与首轮审计逐条对照，不是现状。

### 2.2 自建质量门基线（`lints/vocabulary.toml` `[gate]`）

| 指标 | 数值 | 证据 | 等级 |
| --- | ---: | --- | --- |
| `ccn_max` | 5 | `lints/vocabulary.toml:71` | 已确认 |
| `nloc_max` | 60 | `:72` | 已确认 |
| `file_max_lines` | 899 | `:74` | 已确认 |
| （已删除）`simd_nloc` | ~~96~~ | 原本是「SIMD 微核长度上限」，但 Rust 侧没有任何 SIMD 微核，字段零消费者；**§22.2 已删除**该档位 | 已确认 |
| 存量债条目 | **274** | `[[gate.debt]]` 条目数；`[[gate.debt]]` 字样在文件里出现 275 次，其中 `lints/vocabulary.toml:71` 是说明注释，不是条目 | 已确认 |
| 其中 ccn 超标 | 244 | 同上 | 已确认 |
| 其中 nloc 超标 | 28 | 同上 | 已确认 |
| 其中禁调用形态 | 23 | 同上 | 已确认 |
| 其中文件行数 | 8 | 同上 | 已确认 |
| 按 crate 分布 | weights 84、cutile 45、op 35、tool 21、prof 21、config 16、annotation 12、train 11、autograd 7、cli 3、serve 3、nn 3、prim 3、tensor 3、engine 2、shard 2、diagnostic 1、fuzz 1、testkit 1（合计 274；首轮为 weights 84、op 30、cutile 28、prof 21、audit 20、tool 20、config 16、annotation 12、checkpoint 8、autograd 7、einsum 5） | 同上 | 已确认 |
| 最差 `ccn` | 77（`weights/src/bin.rs:execute`） | `:922` | 已确认 |
| 最长函数 | 202 行（`prof/src/prof.rs:validate`，§19.7 实测） | 登记值 + 实测 | 已确认 |
| 最大文件 | `prof/src/prof.rs` 1 712 行；次大 `train/src/snapshot/mod.rs` 1 611 行 | `wc -l` | 已确认 |
| 门禁是否当前为绿 | 是 | `cargo test -p llm-tool --test gate` 1/1 通过 | 已确认 |
| 该门禁是否在 HEAD 中存在 | **是** | `git show HEAD:lints/vocabulary.toml` 含 274 条 `[[gate.debt]]`；`crates/tool/src/*.rs` 与 `crates/tool/tests/` 均已跟踪（首轮回答为"否"，复核时点已失效） | 已确认 |

口径说明（避免把算术读错）：274 是 `[[gate.debt]]` **条目数**，不是度量数。一条条目可以同时登记 `ccn` 与 `nloc`，所以四个子计数之和（244 + 28 + 23 + 8 = 303）**大于**条目数；303 − 274 = 29 条条目带两个度量，条目身份（`路径:函数#序号`）仍然是 274 个。

结论（事实）：这 274 条**不是历史积累的债务，而是刚安装的严格门禁对既有代码的首次快照**。这是"把标准变成数据"的正确动作，但同时也把"当前实现离自定标准有多远"首次量化暴露出来。

### 2.3 工作树状态（区分在途改动）

| 项 | 数值 | 证据 | 等级 |
| --- | --- | --- | --- |
| 跟踪文件改动 | 0（工作树干净） | `git status --short` 只有 `?? .junie/`（复核时点） | 已确认 |
| 删除的成员 | `crates/{pack,comm}`（首轮前）+ `crates/{audit,checkpoint,regress,exec,kv,einsum,proto}`（§11 吸收，已提交） | `git log --oneline` 与 `crates/` 目录清单 | 已确认 |
| 未跟踪的新模块 | 无：首轮列出的 `train/src/snapshot/`、`op/src/einsum/`、`crates/dtype/`、`op/ops/*`、`tool/src/*` 已全部提交；`cutile/src/report/` 又在 `87ad7486` 中被拆平 | `git status --short` = 1 条 `??`；`git ls-tree -r HEAD` 含上述路径 | 已确认 |
| `weights` crate 跟踪比例 | 33 个 src 文件全部已跟踪 | `git ls-files crates/weights/src` = 33 | 已确认 |

**判定**：首轮审计描述的是"工作树现状"，当时 `weights` 容器、`op` 22 算子、门禁本身与 `report` 拆分都是**在途未提交改动**；到复核时点，这些已全部进入版本库（`de411432`…`ee0cedb9` 一串提交），所以"尚未发布"这一限定**已失效**——它们现在就是发布内容。七次搬家只移动文件，没有改变代码行为。

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
   `op` 每算子同时有模板实现与标量参考；`train::store` 的 `KERNELS` 表长度与顺序在编译期对齐 `op::REGISTRY`（`train/src/store.rs:512`，`const _: () = same_order(&KERNELS, &op::REGISTRY);` 在 `:537`）；`autograd/tests/gradcheck.rs` 用中心差分对 200 个随机图（`:214-220`）以 `2e-3 + 0.05·max(|fd|,1)` 容差检查。
   还有若干"高质量测试"值得点名：`safetensors.rs:642` 用 1 GiB 稀疏文件证明只读了头部、`safetensors.rs:704` 断言两种后端给出相等的 `Header`、`merkle.rs:129` 对 7 叶树所有下标证明并验证、`round_trip.rs:238` 关掉 CRC 再篡改字节以证明"签名层而非 CRC 层"在兜底。

8. **错误契约统一且有类别映射。**
   全仓一个 `Error`/`Code`/`Result`（`diagnostic`），类别由产生错误的层给，`Code::exit_code` 是唯一映射；`tests/architecture.rs:151` 禁止退回字符串错误。

---

## 四、逐分层、逐模块审计

### 4.1 分层总览（合并后，与 `docs/ARCHITECTURE.md` 第一节一一对应）

生产 LOC 为 `wc -l` 实测（含文件内联测试）；括号内是**被吸收子模块**的行数，已计入宿主。表中各 crate 之和 = `crates/*/src` **54 959**（`prim` 的 788 已含 `build.rs` 622；A1/A2 之后的重测值，见 §13.12）；加上根入口 1 544 与 `weights/examples/dump_bin.rs` 40，全仓生产合计 **56 543**（基准 harness 30 行不计）。

| 层 | 成员 | 生产 LOC（含内联测试） | 健康度 |
| --- | --- | ---: | --- |
| L1 基座 | `diagnostic` `dtype` `testkit` `tool` | 856（含 `toml` 585）/ 2 136 / 167 / 1 955（含 `regress` 341） | 高（`dtype` 已升为精度事实源，6 文件；`diagnostic` 现在同时拥有错误类型与全仓唯一 TOML 解析器） |
| L2 数据层 | `prof` `op` `prim` | 3 116 / 8 278（含 `einsum` 1 378）/ 788 | 中高（`op` 膨胀） |
| L3 内核 | `cutile` | 9 365（22 文件；报告层已拆平，`report/` 目录不存在） | 中（双渲染路径、报告与机制同 crate） |
| L4 运行时 | `mem` `tensor` `storage` `tokenizer` | 205 / 590 / 414 / 125 | 高（`storage` 未被使用、`tokenizer` 仅契约） |
| L5 模型与训练 | `weights` `autograd` `data` `nn` `shard` | 10 216 / 3 134 / 630 / 1 543 / 526 | `weights` 低（职责过载），其余中高 |
| L6 调度与服务 | `train` `engine` `serve` `cli` | 3 462（含 `snapshot` 1 611）/ 1 333（含 `exec` 188、`kv` 334）/ 1 478（含 `proto` 1 126）/ 648 | 中（`serve`/`exec` 仅契约；`serve` 现已含线格式） |
| L7 入口 | 根包 `llm`（`src/main.rs` 682、`src/bin/cutile.rs` 863） | 1 545 | 中（含领域逻辑） |
| 工具成员 | `annotation` `fuzz` `benches` | 2 444 / 398 / 0 | `annotation` 中（分层豁免 + 认识算子名） |

### 4.2 逐成员职责与边界评审（25 行 = 24 crate + 根包）

被吸收的七个成员（`audit`/`checkpoint`/`regress`/`exec`/`kv`/`einsum`/`proto`）不再单独成行：它们的能力与判定已并入下表的宿主行，落点与追溯见 §6.0 末尾与 §11.1。第八个被删除的成员 `llm-config` 同样不再成行：设置层并入 `cutile::settings`（§13.11），语法层并入 `diagnostic::toml`（§13.12）。

| 成员 | 拥有（已确认） | 不拥有 | 边界判定 | 关键证据 |
| --- | --- | --- | --- | --- |
| `diagnostic` | `Error`/`Code`/`Result` + `toml::Table`/`Value`/`ParseError` | 领域错误字面量、任何领域语义 | **干净**，零依赖、出边为零；两个原语互不引用（ADR 0021） | `diagnostic/src/lib.rs:1-18`、`diagnostic/src/toml.rs:1-13` |
| `dtype` | 标量 24 变体 `DType` + `QType` 25 + `MxType` 7 + MX 元素子集，6 个文件各一张身份表 | 反量化算法（留在 `weights`） | **已升为 L1 精度事实源**：`prof`/`tensor`/`weights`/`annotation` 四条入边；`code` 前 21 个与旧 Wire 逐位兼容；可选 `serde`/`bytemuck` 是全仓唯一的第三方运行时依赖 | `crates/dtype/{dtype,qtype,mxtype,blocked,format}.rs`、`crates/dtype/Cargo.toml:12-21` |
| `testkit` | `approx_eq`/`device_enabled`/`scratch_dir`/位级 CRC | 任何生产逻辑 | **干净**，仅 dev 边 | `testkit/src/lib.rs:3-5`、`:91-104` |
| ~~`llm-config`~~（已删除） | ~~`Settings`/`Key`/`Roots`/`App`/TOML~~ | — | **§13.12 改判**：设置层只有一个生产消费者 → 并入 `cutile`（§13.11，`Key` 改 `impl prof::atom::Closed`，P2-3 随之消失）；语法层有六个跨层消费者、而它们本来就都依赖 `diagnostic` → 并入 `diagnostic::toml`。改名 `llm-toml` 的方案被否决：它不改成员数、不改边数 | §13.11、§13.12、ADR 0020/0021 |
| `tool` | lex/ccn/dup/call/docs/strip/gate/Report + `regress` | 运行时类型、退出码 | **干净**；扫描面有盲区（P1-7）；`Judgement` 与 `Gate` 现在同域 | `tool/src/lib.rs:1-11`、`tool/src/regress.rs:1-20` |
| `prof` | 档案 schema、`Prim` 闭集、`Set` 位集 | 机制代码、设备特判 | **干净**；`validate` 是语义检查器而非 schema（合理） | `prof/src/atom.rs:44-151`、`prof/src/prof.rs:605-833` |
| `op` | 22 条声明、模板、`REGISTRY`、每算子语义、`einsum` 子模块 | 机制（emit/编译） | **有泄漏（已收窄，§19.6）**：族名分支已成表，宏内只剩 1 行按注册名改写 `reference` 的数据 | `op/src/lib.rs:122`、`annotation/src/codegen.rs:288-296`、`op/src/einsum/formula.rs` |
| `prim` | 片段文本、生成的三张表 | 策略、能力区间 | **干净**；`build.rs:103` 目录项静默丢弃（P2-4） | `prim/src/tables.rs:7`、`prim/build.rs:58` |
| `annotation` | 3 个属性宏 | 领域类型 | **有泄漏（已收窄，§19.6）**：12 段族分支已成 12 行骨架表，只剩 1 行按注册名改写；两套 skeleton 词汇表（P1-6） | `annotation/src/codegen.rs:262-296`、`syntax.rs:384-405` |
| `cutile` | 协商/选择/替换/emit/缓存/编译 + 报告层（已拆平为同层文件） | 算子名 | **规则成立但有代价**：把 `op` 的类型原样 re-export 成自己的 API（P2-5）；两条渲染路径并存（P1-4） | `cutile/src/lib.rs:50-51`、`build.rs:82-83`、`substitute.rs:33`、`cutile/src/{doctor,verify,bench,witness}.rs` |
| `mem` | `Arena`/`Region`/`align_up` | 地址空间、布局 | **干净但极简**（只顺序分配、不释放，自述 P1） | `mem/src/lib.rs:110-130` |
| `tensor` | 字节 `Tensor` / 元素 `View<E>`、`Shape`、`Strides`、`SUPPORTED` | 分配、搬运 | **干净**；零拷贝是类型保证而非承诺 | `tensor/src/lib.rs:123-128,237-264,470-480` |
| `storage` | `Layout`/`Order`/`remap` + `mmap` | 分配、元数据 | **干净但未被使用**：生产入边 0（见 §6.2.1）；unsafe 岛合法，且随 `mmap` feature 限在 Unix（§16.2） | `storage/src/mmap.rs:1-53` |
| `tokenizer` | `Token`/`Tokenizer`/`Bytes` | BPE/SentencePiece | **纯契约 + 一个字节级实现** | `tokenizer/src/lib.rs:41-89` |
| `weights` | 外部格式只读适配 + 自有 LLPK 容器 + JSON/ZIP/rANS/DSA/SHA-256/Keccak/NTT/Golay/SVD | 张量、布局、GPU 搬运 | **职责过载**：一个 crate 承担 5 个领域（P1-1）；crate 头文档已滞后（P1-8） | `weights/src/` 33 文件；`weights/src/lib.rs:1-6` |
| `autograd` | tape/schedule/account/plan/replay | 数值与激活缓冲 | **深且真实**：可达剪枝、扇出账本、预算解算 | `autograd/src/{tape,schedule,account,plan,replay}.rs` |
| `data` | `Batch`/`Split`/`Dataset`/`Chunks` | 分词、打乱 | **干净**，规模小；`Batch`/`Chunks` 借用调用方缓冲，批几何在构造期校验 | `data/src/{batch,split,dataset}.rs` |
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

1. **"依赖只指向更低层"这条规则是成立的，而且不是靠人记得**：`tests/architecture.rs` 六项断言全绿，`cargo metadata` 与文档边集逐条相等（含边数与 `+dev` 条数的计数断言；首轮 106/15，七条合并后 82/13，见 §12.2；`llm-config` 删除后是 **76/12**，见 §13.12）。
   反例检查：唯一的"例外"是工具成员 `annotation`/`fuzz`/`benches` 的边**不参与方向判定**（`tests/architecture.rs:84-87`）。而 `op` 与 `nn` 对 `annotation` 是**硬依赖（非 dev）**——也就是说，两个分层成员的真实编译依赖被排除在方向规则之外。这不是错误，但让"唯一一条规则"这句话出现了一个需要读者自行理解的豁口（P3-1）。

2. **"数据层不知道机制"成立**：`op`（含 `einsum` 子模块）/`prof`/`prim` 的依赖里没有 `cutile`；`cutile` 生产代码里也没有任何算子名（14 处 `gemv_i4`/`gemm` 字符串全在 `#[cfg(test)]` 内）。

3. **"机制不知道领域"在第三处被破坏**：`annotation/src/codegen.rs` 按算子名分支（`flash_attention_f32:318`、`gemm_f32:324`、`gemm_i8:334`、`gemm_i4:344`），并按族名 11 处分支（`:211-310`）。于是"新增算子不必改机制"只对 `cutile` 成立，对**宏**不成立。（§19 已把 11 处族名分支改成 12 行骨架表；剩下按算子名的只有 `flash_attention_f32` 一行，且是数据而不是分支。）

---

## 五、问题清单（P0 / P1 / P2 / P3）

> 判定基准：P0 = 不可信输入可致进程崩溃或超大分配；P1 = 真实边界破坏/可移植性/可维护性显著受损；P2 = 局部质量缺陷或门禁盲区；P3 = 命名/文档/风格级别的长期摩擦。

### P0（必须先修，改动都很小）

> 复核（2026-11-14）：四条**已全部修复**，落地记录与反向验证证据见 §15；§12.6 的"仍在"列是修复前的快照，已被 §15 取代。

**P0-1 `rANS` 解码按文件字段分配内存 —— 4 字节可请求 4 GiB**
- 位置：`crates/weights/src/rans.rs:300`（`let src_len = u32::from_le_bytes(..) as usize;`）→ `:325`（`let mut dst = vec![0u8; src_len];`）
- 事实：`src_len` 来自被解码字节流，分配前没有任何与 `cur` 剩余长度或 lane 长度之和的交叉校验。
- 对照：同一 crate 的 `safetensors.rs:243-252` 在 `vec![0u8; declared]` 之前明确拒绝 `header_len > file_len`，并把理由写在注释里——说明"该有的防线"在本仓已有先例，只是 rANS 这条路没做。
- 影响：一个 4 字节字段即可让验证流程申请 4 GiB；在服务进程里等价于可被单个权重文件触发的内存 DoS。
- 等级：**P0**（安全/可用性）。
- **已修（§15.2）**：`decode(stored, cap)` 先把帧头长度与调用方给的上限比较（`rans.rs:380-384`），上限来自段描述符的类型与形状（`reader.rs:84-86`）或元素数推出的 `payload_cap`（`axes.rs:180`），不由帧自报。

**P0-2 pickle 解析在奇数长度 `SETITEMS` 上越界 panic**
- 位置：`crates/weights/src/bin.rs:445-458`（`set_items`）中的 `for chunk in pairs.chunks(2) { entries.push((chunk[0].clone(), chunk[1].clone())) }`（`:451`）
- 事实：`pop_until_marker`（`:305`）返回的计数完全由文件里的字节决定；`chunks(2)` 对奇数长度会产出长度 1 的末块，`chunk[1]` 直接索引越界。
- 影响：一个畸形 `.bin` 让宿主 panic，而该模块文档（`:8-9`）承诺"显式报错，不 panic"。
- 等级：**P0**。
- **已修（§15.2）**：`pop_pairs`（`bin.rs:448`）在 `chunks(2)` 之前判奇偶（`:450-452`），奇数返回 `invalid`。

**P0-3 `checkpoint` 编码路径上的 `expect` 可由权重文件触发**
- 位置：首轮 `crates/checkpoint/src/lib.rs:649`，现为 `crates/train/src/snapshot/mod.rs:651`（合并时文件头补了 2 行，行号 +2）：`sink.push(&[tag(entry.dtype()).expect("快照条目类型在标签闭集内")]);`
- 事实：`tag`（现 `:895-904`）只接受 `F32/F16/BF16/I8/U4`，其余返回 `Err`。而 `entry` 来自调用方给的 `weights::Header`，其 dtype 由文件内容决定。
- 影响：用一份含 `I64`/`F64`/量化类型的权重做快照 → panic，而不是"该格式不支持"的错误。这是一条**正常输入就能走到**的 panic。
- 等级：**P0**（改一行：`tag(..)?`）。
- **已修（§15.2）**：`tags()`（`train/src/snapshot/mod.rs:346`）一次把整份标签表判完，`encoded_len`/`encode`/`encode_into` 都走 `tags()?`；`tag`（`:924`）仍返回 `Result`。

**P0-4 DSA 公钥/签名解码对长度零校验，截断即 panic**
- 位置：`crates/weights/src/dsa.rs:353-367`（`pk_decode`：`&pk[..32]`、`&pk[32 + i*step .. 32 + (i+1)*step]`）、`:376-391`（`sig_decode`：`&sig[c_len + i*z_step ..]`、`&sig[z_end..]`）、`:675` 附近 `hint_unpack_dyn` 的 `len - k`
- 事实：`Envelope::decode`（`envelope.rs:100-103`）从文件读 `pk_len`/`sig_len` 后只做 `cur.take(..)`（受输入长度限制，不会超量分配），但**不与参数集要求的最小长度比较**；随后 `Envelope::check`（`:111-119`）把它交给 `verify_internal` → 上述切片。
- 影响：1 字节的侧车即可 panic。fixture 的"fail-closed"只对**伪造**成立，对**截断**不成立。
- 等级：**P0**。
- **已修（§15.2）**：`pk_decode`（`dsa.rs:378`）要求长度等于 `pk_len()`、`sig_decode`（`:415`）要求不小于 `sig_len_min()`（`:359` 给出下界）、`open`（`:395`）先校验公钥；顺带发现并修掉同一路径上第五处（§15.2 的 P0-4b）。

### P1（合并前应处理）

**P1-1 `weights` 单 crate 承担 5 个互不相干的领域**
- 事实：33 个 src 文件中，容器/线格式（`writer/reader/seg/codec/crc/leb/sink/merkle`）、外部格式（`safetensors/gguf/json/zip/bin`）、压缩（`rans`）、哈希（`sha256/keccak`）、数论/后量子（`dsa/ntt/golay/svd/lattice/ledger`）混在一起；`dsa.rs` 923 行是一份完整的 ML-DSA 实现。
- 影响：任何一处改动都触碰最大 crate（10 183 行、84 条债基线）；crate 头文档（`lib.rs:3-6`）已无法描述它（见 P1-8）；审计成本与合并冲突面同步放大。

**P1-2 `mmap` feature 在非 Unix 目标上无法编译（已修，§16.2）**
- 事实：`crates/weights/Cargo.toml:18-20` 以**dev-dependency**（`features = ["mmap"]`）启用 `storage`；`crates/storage/src/mmap.rs:14` 无条件 `use std::os::unix::io::AsRawFd;`，整个文件没有任何 `cfg(unix)`（只有 `:144` 的 `target_pointer_width` 与 `:150` 的 `cfg(test)`）。
- 影响（**已确认，不再是推断**）：`cargo check -p llm-storage --features mmap --target x86_64-pc-windows-msvc` 在本机交叉目标上直接报两错——`mmap.rs:14` 的 `unresolved import std::os::unix` 与 `:71` 的 `no method named as_raw_fd`。修复与全仓交叉验证见 §16.2。**等级：P1（可移植性）**。
- 附带事实（本次新发现，影响 §6.2）：`storage` 的**生产入边为 0**——`Layout`/`Order`/`remap` 没有任何 crate 调用，`Mmap` 唯一使用点是 `crates/weights/tests/round_trip.rs:175`。

**P1-3 13 处错误上下文仍指向已删除的 `pack` crate**
- 事实：`with("pack::quant")`（`weights/src/affine.rs:109,140`）、`pack::rans`（`rans.rs:377`）、`pack::leb`（`leb.rs:107,111`）、`pack::codec`（`codec.rs:29`）、`pack::seg`（`seg.rs:207`）、`pack::Writer`（`writer.rs:75,195,208,256`）、`pack::Archive`（`reader.rs:110,281`）——**13 个代码位点**；另有 `storage/src/mmap.rs:7` 的文档注释仍写"交给 `pack::Archive::parse`"（第 14 处，属文档）。`dsa.rs:792-793` 的 `packed` 是变量名，不是本问题。
- 影响：用户看到的诊断锚点指向一个不存在的模块，排障时会去搜索一个已删除的 crate。**这是"文档/诊断漂移"而非风格问题**。

**P1-4 `cutile` 同时存在两条模板渲染路径**
- 事实：生产走 `ast::build` + `ast::render`（`build.rs:82-83`）；`substitute::render`（`substitute.rs:33`）仍保留，调用点只有 `builtin.rs:342` 处的一次对拍检查和测试。两条路径的等价性由**一个测试**保证（`ast.rs:313`），代码注释自述"字符串路径保留用于回滚与对拍"（`build.rs:80-81`）。
- 影响：ADR 0017 的迁移处于半完成态；在移除前，任何替换规则的改动都要改两处并靠测试兜底。

**P1-5 属性宏认识算子名与算子族名（分层泄漏）**
- 事实（§19 前）：`annotation/src/codegen.rs:318` 判等 `"flash_attention_f32"`，`:324` 判等 `"gemm_f32"`，`:333/:345` 判 `gemm_i8`/`gemm_i4`；`:211-310` 按族名 11 处分支。**已收窄（§19）**：11 处族分支 → `SKELETON_TABLE` 12 行，`gemm_*` 的三个判等消失（模块名就是注册名），只剩 `REFERENCE_OVERRIDES` 一行（`codegen.rs:288-296`）。`dequant_i8`/`dequant_i4` 仍是两行——理由从"命中不同的 `else if`"变成"粒度 1 vs 2"，即真正的差异。
- 影响：新增一个"需要自己的 cost/sample/reference 模块"的算子，必须改宏；`op` 的"加算子不改机制"承诺在宏这一层失效。

**P1-6 存在两套 `skeleton` 词汇表，且多数取值无人构造**
- 事实：`op::decl::Skeleton` 是 5 个变体（`decl.rs:30-36`：Elementwise/Reduce/Gemm/Attention/Scan）；`annotation::syntax::SKELETONS` 是 11 个名字（`syntax.rs:385-405`），两者不是投影关系。`Skeleton::Attention` 与 `Skeleton::Scan` 在全仓**没有任何构造点**（唯一构造 `Skeleton::Gemm` 在 `cap.rs:561` 的测试里）；`op.skeleton` 只被 `emit.rs:53` 的注释头与 `cutile/src/doctor.rs:161` 的报告字段消费（`report/` 已拆平，§12.3）。
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
- **已修（§16.3）**：`docs/ARCHITECTURE.md:99` 原写"现有两条算子"，按 `op/src/lib.rs:111` 的 `[Registration; 22]` 改为 22 条；同时发现 `crates/op/README.md:108` 整节标题就是"现有的两条算子"、表里只有 2 行，已改为按族的 7 行表，并写明"清单不在这里维护，`REGISTRY` 是唯一事实源"。
- **已修（§16.3）**：`crates/weights/src/lib.rs:4-10` 的"拥有/依赖"段按 `crates/weights/Cargo.toml`（`dtype`/`diagnostic`/`mem`/`tensor`，`+dev` `storage`/`testkit`）与 `pub mod`/`mod` 声明实况重写，补上 LLPK 容器、格式适配器与证明栈。
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
- ~~`weights/src/seg.rs:65-68`：`for axis in &self.shape { let _ = axis; }`~~ **已删（§16.4）**：注释说"轴长允许 0"，但循环体是空的、校验从未写；删除时把这条决策移进 `validate` 的文档注释，理由不跟着代码一起消失。
- ~~`tool/src/ccn.rs:138-142`：`let nested = collect(items, &mut nested)` 结果从不读取~~ **复核时已不在**：现为 `Token::Group { .. } => {}` 并附"签名组内递归不会产生定义"的注释（§16.4）。

**P2-9 可空转的测试（"绿但没测"）**
- `weights/src/bin.rs:582-590`、`zip.rs:198-201`：`/tmp/pt/model.bin` 不存在时直接 `return`；而这是 ZIP+pickle 路径**唯一**的练习。`bin.rs:619` 的断言只检查 `bytes.len() == 16`（`:618` 注释却声称"与原始字节一致"）。
- `cutile/src/witness.rs`（首轮 `report/witness/mod.rs:108-110`，路径已随 `87ad7486` 变化）：无后端时直接 `return`，与 `testkit::device_enabled` 的显式门控相比是两种风格。
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
- **P3-5**（**§22 已修**：棘轮改成"登记值必须等于实测"、`stale` 按（身份, 度量）结清、限值收进 `Gate::limit`；下面这段是修前的事实记录）"只准降"的棘轮其实只做到"不许涨"：`Gate::check`（`tool/src/gate.rs:124-136`）只在 `value > allowed` 与 `value <= limit` 两处失败，**改善不受任何约束**；`Gate::stale`（`:140-147`）以身份（而非度量）为键，一个同时登记 ccn 与 nloc 的函数只要 nloc 还超标，其已改善的 ccn 条目永远不会被要求删除。文档"存量项只准降"（`gate.rs:4`）与实现不符。
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

"生产入边"= 其他成员（**不含根包**、**不含 dev 边**）的 `[dependencies]` 指向它的条数，`cargo metadata` 口径；它与 `ARCHITECTURE.md:38` 的 82 条边（含根包与 13 条 `+dev`）不同，后者是契约全集。"现 LOC"为 `wc -l crates/*/src` 实测（含文件内联测试）。表中三列数值已按 HEAD `4e363270` 重新测量（§12.2），与合并当时的差异集中在 `dtype`（313 → 2 136）、`data`（233 → 630）与 `cutile`（9 469 → 9 365）。

| 成员 | 生产入边 | 出边 | 现 LOC | 最终处置 | 依据 / 附带动作 |
| --- | ---: | ---: | ---: | --- | --- |
| `diagnostic` | 19 | 0 | 260 | 保持 | 全仓唯一错误契约，`tests/architecture.rs:151` 禁止回退字符串错误 |
| `dtype` | 4 | 0 | 2 136 | 保持 | L1 精度闭集；并入 `tensor` 会让 `prof` 反向依赖 L4 |
| `testkit` | 0（+dev 5 个消费方） | 0 | 167 | 保持 | dev-only 夹具（`testkit/src/lib.rs:3-5`），不进任何二进制 |
| ~~`llm-config`~~ | 4 | 0 | 605→0 | **已删除（§13.12 改判）** | 设置层只有 `cutile` 一个消费者 → 并入 `crates/cutile/src/settings/`（§13.11）；语法层六个消费者本来就都依赖 `diagnostic` → 并入 `diagnostic::toml`，成员随之删除（§13.12、ADR 0021）。否决了 §13.6 的"改名 `llm-toml`"：改名不改成员数、不改边数 |
| `tool` | 0 | 2 | 1 955 | 内部整合 | `dup.rs:11-14` 只数文件行数、不做重复检测；已吸收 `regress` |
| `prof` | 4 | 3 | 3 116 | 拆分文件 | `prof.rs` 1 712 / `atom.rs` 1 190，见 §6.3 S3 |
| `op` | 3 | 4 | 8 230 | 内部整合 + 拆 `decl.rs` | ~~22 条 `Registration` 可宏生成~~ **已做（§20.2）**：`registry!` 一行一对；`decl.rs` 673 行混三职责；已吸收 `einsum` |
| `prim` | 1 | 3 | 788 | 保持 | 三张内嵌表由 `build.rs:58` 扫描生成，无策略 |
| `annotation` | 3 | 1 | 2 444 | **不可合并** | `proc-macro = true`（`crates/annotation/Cargo.toml:12-13`）；删族名分支（§6.4 W7） |
| `cutile` | 0 | 6 | 9 365 | 内部整合（删双渲染）+ 拆 3 文件 | P1-4；`cache.rs` 1 246 / `cap.rs` 916 / `backend.rs` 859；已吸收 `report` |
| `mem` | 2 | 1 | 205 | 保持 | `tensor` 与 `weights` 都直用它；不释放是有意最小集（`mem/src/lib.rs:85-86`） |
| `tensor` | 7 | 3 | 590 | 保持（并接收布局） | 步幅的唯一主人（`tensor/src/lib.rs:99`），见 §6.3 S2 |
| `storage` | 0 | 2 | 414 | 拆分/合并 | 零生产入边；`Layout` 与 `Strides` 重复，`mmap` 只被测试用 |
| `tokenizer` | 1 | 1 | 125 | 保持 | 文本↔token 唯一边界 + 一个字节级参考实现，不装 BPE 是有意 |
| `weights` | 2 | 4 | 10 216 | 目录重组 → 择机拆 3 crate | P1-1；先 §6.4 W1 再 §6.3 S1 |
| `autograd` | 1 | 2 | 3 134 | 拆分 `plan.rs`/`replay.rs` | 866/683 行，两个算法各占一个文件 |
| `data` | 1 | 2 | 630 | 保持 | 样本切分属 L5 语义，`train` 属 L6 |
| `nn` | 1 | 5 | 1 543 | 内部整合 | `stack.rs` 580 行可切"层链/总下界/未计价"三块 |
| `shard` | 1 | 1 | 526 | 保持 | 通信契约 + 拓扑算术；后端未实现是有意占位 |
| `train` | 0 | 8 | 3 462 | 拆分 `store.rs` + `snapshot` | 1 530 行里 9 段职责（P2-6）；`snapshot` 1 611 行待拆（§11.4） |
| `engine` | 1 | 2 | 1 333 | 拆分 `sequence`/`scheduler` | 已吸收 `exec`/`kv`；`exec` 的 SPI 去留待决（§11.4） |
| `serve` | 0 | 3（+dev 1） | 1 478 | 保持（接收线格式） | 明确无 socket（`serve/src/lib.rs:7-8`）；已吸收 `proto`（`serve/src/proto/`，1 126 行）；出边 `engine`/`tokenizer`/`diagnostic` |
| `cli` | 0 | 1 | 648 | 增边（语法回填） | `--set` 解析在 `src/bin/cutile.rs:57-78`，属 `cli` 的语法域 |
| `fuzz` | 0 | 1（+dev 4） | 398 | 保持（工具成员） | 契约是"任何输入"，与 `testkit` 的"夹具"不同 |
| `benches` | 0 | 0（+dev 3） | 0 | 保持（工具成员） | 只有一个 criterion harness |
| 根包 `llm` | — | 15 | 1 545 | 拆分（业务逻辑出仓） | `src/main.rs:17` 定义 `#[nn::layer]` 类型，违反 L7"不定义新类型" |

**已消失的七个成员（追溯，不再计入上表）**

| 首轮成员 | 首轮 LOC | 现在的位置 | 首轮结论 | 今天的读法 |
| --- | ---: | --- | --- | --- |
| `checkpoint` | 1 609 | `train/src/snapshot/mod.rs` | 拆分 4 文件 | 已并入；拆文件转为 §11.4 待办 |
| `audit` | 3 947 | `cutile/src/*.rs`（13 文件；`report/` 已在 `87ad7486` 拆平） | 拆分文件 | 已并入；拆文件的原始方案（`report/mod.rs` 700 行）**作废**，现按平铺文件拆分 |
| `regress` | 337 | `tool/src/regress.rs` | 合并候选 | 已并入；`Judgement` 保留原名 |
| `exec` | 188 | `engine/src/exec.rs` | 暂不合并 | 已并入；SPI 去留仍未决 |
| `kv` | 334 | `engine/src/kv.rs` | 保持 | 已并入；页数公式 4 → 1 |
| `einsum` | 1 378 | `op/src/einsum/`（4 文件） | 保持 | 已并入；`Bits` 改用 `Closed` |
| `proto` | 1 126 | `serve/src/proto/`（4 文件） | 保持 | 已并入；`pub` 路径变 `serve::proto::`，编译期常量与测试原样 |

统计口径（合并后）：25 个 crate 里**0 条生产入边** 9 个（`cutile`/`train`/`serve`/`cli`/`tool`/`storage`/`testkit`/`fuzz`/`benches`）。其中 5 个只被最终二进制消费（正常），`testkit` 是 dev-only，`fuzz`/`benches` 是工具成员，**只有 `storage` 同时"零生产入边 + 零消费者"**——这是 §6.2.1 第一条要处理的成员。要强调的是：**"只有 1 个消费者"本身不是合并理由**；六条吸收成立的理由是 ADR 0019 的"单消费者吸收"，而不是入边数本身。

### 6.1 硬约束与有意设计：不能合并的边界

| 对象 | 为什么不合并 | 证据 |
| --- | --- | --- |
| `op` ↔ `cutile` | 数据 vs 机制；合并会把"加算子"变成"改机制" | `op` 无 `cutile` 依赖；`cutile` 生产代码无算子名 |
| `prof` ↔ `prim` | 档案 schema/能力闭集 vs 片段文本；`prim` 的内容由 `build.rs` 扫描生成 | `prof/src/prof.rs` vs `prim/build.rs:58` |
| `dtype` ↔ `tensor` | `dtype` 是 L1，被 `prof`/`weights`/`train` 共用；并入 `tensor` 会让 `prof` 反向依赖 L4 | `Cargo.toml` 边 |
| `diagnostic` ↔ 任意层 | 唯一错误类型；合并即制造第二事实源 | `tests/architecture.rs:151` |
| `autograd` ↔ `train` | `autograd` 只读 tape/plan，不碰数值；`train` 才是缓冲与数值 | `autograd/src/replay.rs:38-50` |
| `audit` ↔ `cutile` | 机制 vs 命令策略；合并后 `cutile` 会被迫拥有"形状矩阵/重复次数/达成率口径" | 现 `cutile/src/{doctor,verify,bench}.rs`（`report/` 已拆平；首轮 `audit/src/lib.rs:7-9`）；**§11.2 改判为合并**（未做的附带动作见 §11.4） |
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
| --- | --- | --- | --- | --- | --- |
| `storage` 的 `Layout`/`Order`/`remap` → `tensor` | 400 行中约 250 行是布局代码，**零生产调用点**；`Layout::strides`（`storage/src/lib.rs:67`）与 `Strides::contiguous`（`tensor/src/lib.rs:99`）是同一规则两份实现 | **做（第一步）** | 删掉 P2-7 的步幅双份，`tensor` 成为步幅唯一主人 | 低：`Order`/`remap` 各有一个 doctest，需随迁 | 低（`pub use` 保留路径） |
| `storage` 的 `Mmap` | 唯一使用点是 `crates/weights/tests/round_trip.rs:175`；生产代码零调用 | **二选一**：(a) 让 `safetensors::File` 真的走 mmap 后端（补生产入边）；(b) 按 ADR 0004 移出 `members`，`mmap` 降为测试夹具 | (a) 兑现 `ARCHITECTURE.md:101` 的"内存/映射"承诺；(b) 少一个 crate | (a) 需处理 `Send`/`Sync` 与生命周期（`mmap.rs:47-49`）；(b) "唯一 unsafe 岛"会从架构图里消失，需同步文档 | 低 |
| `regress` → `tool`（或补生产消费者） | 337 行，0 生产入边，唯一消费者是 `benches` 的 dev 边 | **已并入 `tool`**（§11.1） | 判定与报告同域（`tool/src/lib.rs:1-11`），少一个 crate | `Judgement` 与 `cutile::report::Verdict` 语义不同（`ARCHITECTURE.md:137-141`），已保留原名 | 低（一次 `pub use` + 3 处 import） |

`storage` 这一条是本次新增结论中最强的：它同时满足"零生产入边"与"与其他 crate 重复实现同一规则"两个合并条件，而且删除它不会让任何生产代码编译失败——按 `ARCHITECTURE.md:202-203` 自己的标准，它今天更接近"目录"而不是"模块"。

#### 6.2.2 提出过但驳回的合并（附重议条件）

| 曾考虑的合并 | 驳回理由 | 什么条件下重议 |
| --- | --- | --- |
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
| --- | --- | --- |
| `train::snapshot` 改用 `weights::leb` | 自带 `varint`（`train/src/snapshot/mod.rs:754,810`），连"非规范形式必须拒绝"的论证都重写了一遍 | 删一份 LEB 实现（已是 `weights::crc` 的消费者，口径一致） |
| `weights::gguf` 改用 `mem::align_up` | 本地重写 `align_up`（`weights/src/gguf.rs:420`） | 对齐规则变一份（P2-7） |
| `cutile` 停止 `pub use op::decl/layout/sch` | `cutile/src/lib.rs:50-51` 把声明层的公开面变成自己的公开面（P2-5） | `op` 改名不再是对 `cutile` 的破坏性变更 |
| `op`/`nn` 对 `annotation` 的硬依赖写进边表 | `ARCHITECTURE.md:21-23` 把它归为"工具成员、边不参与方向判定"，但 `op`/`nn` 的**非 dev** 依赖确实参与编译 | 边表与实际一致（P3-1） |
| `cli` 收回 `--set` 语法 | `src/bin/cutile.rs:57-78` 在解析前自己处理 `--set key=value` | 语法集中到 `cli`，二进制只剩装配 |

### 6.3 crate 级拆分

**S1 `weights` → 3 个新 crate（最高收益，但分两步）**

| 新 crate | 收纳 | 行数（约） | 说明 |
| --- | --- | ---: | --- |
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
| --- | ---: | --- | --- |
| `Layout`/`Order`/`remap` | ~250 | `tensor` 的 `layout` 子模块 | 与 `Strides::contiguous` 同一域；`tensor/src/lib.rs:99` 已是步幅主人 |
| `Mmap` | ~150 | 留在 `storage`（或按 §6.2.1(b) 降为测试夹具） | 全仓唯一 unsafe，需独立 feature 门（`storage/src/lib.rs:26-29`） |
| 合并后体量 | `tensor` 585 → ~835；`storage` 400 → ~150 | — | 少一个"零入边"crate，少一份步幅实现 |

**S3 文件级拆分（保持公开 API 不变，仅改模块入口与 re-export）**

| 文件 | 建议子模块 | 关键分界（可回链） |
| --- | --- | --- |
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
| ~~`cutile/src/report/mod.rs` 700~~ | **作废**：`87ad7486` 删掉 `report/` 层，该文件已不存在；报告层现在平铺为 `cutile/src/{doctor,verify,bench,witness}.rs` | 现状：`doctor.rs` 538、`verify.rs` 542、`bench.rs` 310、`witness.rs` 859 行 |
| ~~`cutile/src/report/witness/device.rs` 622~~ | **已被 `4e363270` 部分兑现**：设备探测拆到 `cutile/src/probe.rs`，其余留在 `cutile/src/device.rs` | `std::env::consts::OS` 的分发仍在 `device.rs`（行号已变，需重锚） |
| `cutile/src/verify.rs` 542 | 保留 `verify` 入口，拆 `accept.rs`（负例）、`cuda.rs` | 仍成立；路径由 `report/verify.rs` 变 `verify.rs`，行号需重锚 |
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
- **W6 `config::Key` 一张表**：`(key, file_key, env_suffix, kind, doc)` 一行一个键，四个方法各自投影一列（P2-3）。**§13 改判**：键表并入 `cutile` 后是 `prof` 下游，直接 `impl prof::atom::Closed`，`ALL`/`as_str`/`parse` 由 `inherent_names!` 转发，本地表连同 `[closed]` 豁免一起删除。
- **W7 `annotation` 删族名分支（已做，§19）**：`codegen.rs:211-374` 的 11 处同形分支 → `SKELETON_TABLE` 12 行 + 6 个小函数（P1-5）。
- **W8 根包归位**：`src/main.rs:17-18` 的 `LmHead` 注解类型移入 `nn`；`decode_f32`/`parse_input`（`:382-423`）移入 `weights`/`nn`；`src/bin/cutile.rs` 的排版移入 `cli`。

### 6.5 决策汇总（三档，附不做的事）

| 档位 | 项 | 为何这一档 |
| --- | --- | --- |
| **立刻**（零风险 / 纯删除） | 修 `pack::*` 13 处标签；`storage` 布局并入 `tensor`；`engine`/`report` 拆文件；`config::Key` 单表；四份缓存合一；删 `seg.rs:65-68` 与 `ccn.rs:138-142` | 不改语义、不改公开类型名，只移动与删死代码 |
| **有条件**（需前置） | `weights` 真拆 crate（等 29 个未跟踪文件提交）；`storage::Mmap` 接生产（等 `safetensors::File` 改后端）；`einsum`/`regress`/`proto` 等七条吸收**已执行**（ADR 0019，§11） | 前置项不满足时做，会与在途改动叠加放大冲突面 |
| **不做**（已给理由） | 合并 `diagnostic`/`dtype`/`annotation`/`shard`/`data`/`mem`/`tokenizer`；拆 `tensor`/`toml.rs` | 见 §6.1 与 §6.2.2；`kv`/`exec`/`einsum`/`audit`/`regress`/`checkpoint` 已按 §11 改判为合并，不再属本档 |

一条方法上的判定基准（供下一轮争论时直接用）：**合并的唯一正当理由是"同一知识存在两份实现或两份消费者争用"，不是"行数少"**。按这个基准，首轮只有 `storage`（重复步幅 + 零入边）与 `regress`（零入边 + 与 `tool` 同域）达标；`einsum`/`exec`/`data` 等"只有 1 个消费者"的成员按**这一条**不达标。§11 追加了 ADR 0019 的第二条理由（单消费者吸收），据此 `einsum`/`exec`/`kv`/`audit`/`checkpoint` 也被判为合并——两条理由的分工见 §10.5 末尾。

### 6.6 文件/函数级别的技术债核减（能真正删除基线的）

| 债务 | 现状 | 可消除方式 | 预计收益 |
| --- | --- | --- | --- |
| 4 份表达式缓存 | `gemv.rs:250`、`elementwise.rs:41`、`unary.rs:32`、`reduce.rs:31`，同一个 `OnceLock<Result<Formula, Error>>` | **已兑现（§18.2）**：一个 `einsum::FormulaCache`，`site` 由骨架给出，`static` 仍按类型分开 | 实测 −78 行（139 → 61），与预估的 −60 同量级 |
| `gemv` 输入校验 3 份 | `gemv_i4.rs:~100-135` 与 `gemv_i8.rs:93-123` 共享 20 行（连错误文案都相同） | **已兑现（§18.3）**：收成 `non_empty` / `scaled_groups` / `required_weights` 三个 helper；`gemv_i8::layout` 整段删除 | 代码 −≈50 行，债条目 −1，语义单源 |
| 22 条 `Registration`（**已兑现，§20.2**） | `op/src/lib.rs:122` 一行一对：`registry! { "名字" => 语义模块 }` | 表内由宏展开成 `Registration { name, template: include_str!(..), model: <模块>::MODEL }` | 实测 `op/src/lib.rs` 479 → 431（−48），加算子仍改一行 |
| 22 个 `forward` shim（**判定撤回，§20.5**） | 实测 24 个 `pub fn forward`，其中真正单行委托的只有 **9 个**（`add_f32.rs:43`、`gelu_f32.rs:57`、`gemv_f32.rs:111`、`mul_f32.rs:42`、`reduce_max_f32.rs:41`、`reduce_sum_f32.rs:41`、`residual_add_f32.rs:42`、`scale_f32.rs:42`、`silu_f32.rs:38`） | 不宏化：省不足 15 行，却把 9 个公开签名藏进宏 | 0（原估 −40 不成立） |
| `codegen.rs:211-374` 11 处同形分支（**已做，§19**） | 每处 8 行，仅模块路径不同 | `SKELETON_TABLE` 12 行 + 6 个函数 | 实测 ccn 15 → 2、nloc 233 → 19；行数 **+21 → 最终 +33**（原估 −80 不成立，§19.5 说明原因） |
| `seg`/`Desc`/`Stored` 三层同构 | `seg.rs:16,119`、`reader.rs:28`，靠 `Desc::into_parts`（`:182`）桥接 | 保留 `Segment` 一个记录 + 一个借用视图 | 代码 −60 行，去掉一类转换 bug |
| 对齐/LEB/步幅/元素数 四处重复 | P2-7 | 分别收口到 `mem`/`weights::leb`/`tensor` | 真重复 −4 处 |
| `cutile` 双渲染 | `substitute.rs:33` 与 `ast.rs:88` 两条路径 | 删字符串路径（§6.4 W4） | 代码 −137 行 + 一个对拍测试 |
| `config::Key` 5 张表 | `key.rs:34-79` | **§13 改判**：`impl Closed`（§13.3-3），收益并入设置层吸收 | 代码 −≈1 250 行（与 §13 合并计），加键从 5 处变 1 处 |
| `storage::Layout::strides` | `storage/src/lib.rs:67` 与 `tensor/src/lib.rs:99` | 并入 `tensor`（§6.3 S2） | 代码 −60 行，少一份步幅实现 |

---

## 七、重复与技术债治理

### 7.1 真重复 vs 结构性相似（区分判据）

方法可复现：取 168 个 `crates/*/src/**/*.rs` 与根 `src/**/*.rs` 文件，去掉注释、空行与 `#[cfg(test)]` 之后的部分，逐行归一化空白，做 8 行滑窗，按窗口哈希跨文件比对；命中 **65 个跨文件重复窗口**（约 520 行量级）。窗口数 ≠ 行数，也 ≠ 块数：相邻窗口合并后，真正值得看的只有 9 对各 1–3 个块。这个密度对一个 3.8 万行生产代码的仓库来说很低——规模来自"广度"而不是"复制"。下表逐对判定。

| 对象 | 判定 | 理由（可回链） |
| --- | --- | --- |
| `rmsnorm_f32` ↔ `softmax_f32`（11 窗口）、`rope_f32` ↔ `softmax_f32`（7） | **结构性相似，保留** | 都是"逐行/逐 token 标量参考"的骨架：`M`/`K` 常量 + 一个双重循环 + 一个测试辅助。抽出来的抽象会比重复更难读。 |
| `gemv_i4` ↔ `gemv_i8`（10 窗口） | **真重复部分已清（§18.3）** | `cost`/`sample`/`reference` 早已由 `gemv.rs` 骨架统一，输入校验本轮上提为三个 helper；剩下的只有内层点积循环——int4 走 nibble 解包、int8 走 `i8` 解释，是**格式差异**而不是复制。 |
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
5. **对 `ccn_max = 5` 的判断（重要，至今未做）**：对纯函数这是极高标准，对**文法解析器**它不可达——`parse` 的 ccn 本质是"关键字/操作码个数"。建议把阈值按类分档（`ccn_max` 纯逻辑 5、`parser` 15、`dispatcher` 25），并把这条分档写进 `[gate]`；否则 251 条 ccn 基线会长期存在，棘轮变成噪声，反而削弱门禁的威慑力。**这属于"降低形式指标、提升真实约束质量"的双赢项。**

### 7.3 与在途改动的关系（git 取证）

- ~~282 条债基线全部来自未提交的工作树~~：**复核时点已失效**——HEAD 的 `lints/vocabulary.toml` 含 274 条债务，`crates/tool/*` 已全部入库（§12.3）。
- 93 个跟踪文件 +3 545/−10 272（`git diff --shortstat`），说明本轮改动是"净删除"——方向正确。上一版两处把 `+`/`−` 写反了，以本节为准。
- ~~但 `weights` 有 29/33 个 src 文件未跟踪~~：复核时 `git ls-files crates/weights/src` = 33，全部已跟踪；"把提交作为基线验收条件"这条建议**已被执行**。

---

## 八、对标差距与改进路线

### 8.1 与 Burn（及同类框架）的逐维对比

数据来源：我方数字为本次实测；Burn 的规模数字取自本仓自己的 `docs/10k-analysis.md`（该文注明用 tokei 同口径实测）。

| 维度 | 我方（已确认） | Burn（公开事实/该文实测） | 判定 |
| --- | --- | --- | --- |
| 代码规模 | 40 478 生产行（不含内联测试） | 369 340 Rust 行（约 9.6×） | 我方是"机制最小集"，不是"产品全量"；Burn 的规模数字取自 `10k-analysis.md`，该文自己的明细已过时（P3-6），但 36.9 万这一量级结论不受影响 |
| 依赖 | 0 运行时 / 1 dev（criterion） | 1 141 个 lock 条目 | **我方显著领先** |
| Backend 抽象 | 数据驱动：`prof/devices/*.toml` + `prim/<vendor>` + `#[frontend]`，新增设备不改机制 | 泛型 `Backend` trait 塔 + CubeCL 运行时 | 路线不同：我方"配置+文本"，Burn"泛型+IR"；我方移植成本低，编译期类型安全弱 |
| 运行时分派 | 无分派（编译期表 + `DEVICES` 常量表） | 有 `Dispatch`/`Runtime` 层 | 我方无动态派发开销，但也无 JIT/动态形状 |
| 融合/装饰器 | **无**（ADR 0017 显式冻结 L2） | `burn-fusion` 做 kernel 融合 | 我方在 elementwise 链上有真实差距（每算子一次 kernel） |
| no_std | 未声明（依赖 `std`：`OnceLock`、`fs`、`process`） | 支持 no_std 子集 | 差距明确；但本仓定位是 host 侧编译，优先级可低 |
| 多后端可切换 | 3 个 vendor 的片段（cpu/intel/metal），**仅 1 条可执行路径（C/cc）** | 多后端均可用 | 真实差距在"可执行后端数量" |
| unsafe | 6 处关键字、1 个文件、feature 门 | 大量后端内部 unsafe（CubeCL/CUDA） | **我方显著领先** |
| 测试策略 | 1 083 用例、gradcheck、真实权重对拍、双源语义 | burn-backend-tests 约 69k 行对拍矩阵 | 我方"点位精、面窄"；缺后端矩阵与模糊测试常驻 |
| 架构门禁 | 可执行（成员/边/方向/所有权/错误边界） | 无同类等价物（公开资料未见） | **我方领先** |
| 性能证据 | **未知**：本次未跑真机 bench；仓库内只有 `span_parse` 一个 criterion 基准 | 有成熟 benchmark 套件 | 差距是"没有证据"，不是"性能差" |

### 8.2 剩余工作的唯一执行序

§9.2 与 §10.6 都是本节的切面：§9.2 挑"最小样板"，§10.6 展开"生成化/删除"这一维；**三者冲突时以本节为准**。

**第 0 梯队（已完成，不是待办）**：七条吸收链已按 B0–B5 执行（§11.5），收益只记成员与边（§11.6），**不冲抵**下面的行数减量；`kv` 的页数公式 4 → 1 是其中唯一的质量收益。

**第一梯队（1 人日以内，纯收益）**

1. ~~修 P0-1/2/3/4：`rans` 长度交叉校验、`bin` 奇偶校验、`checkpoint` 的 `tag(..)?`、`dsa` 的 `Result` 化~~ **已兑现（§15）**：实测净增 198 行（含五条回归测试），消灭 4 条崩溃路径 + 1 条同族越界。
2. ~~`mmap` 加 `#[cfg(unix)]`~~ **已兑现（§16.2）**：模块与再导出改为 `cfg(all(feature = "mmap", unix))`，消费它的 `round_trip` 用例加 `#[cfg(unix)]`；`--workspace --exclude llm-benches --all-targets --all-features --target x86_64-pc-windows-msvc` 退出码 0。
3. ~~修正 13 处 `pack::*` + `mmap.rs:7` 的过时引用；同步 `README` 与 `ARCHITECTURE.md` 的算子数~~ **已结清**：`pack::` 在更早的提交里已全部消失（现只剩 `dsa.rs:853-854` 的 `packed` 变量名误命中），`README` 成员表随 §11 同步；算子数在 §16.3 修正（22 条注册，含 `op/README.md` 那节整段重写）。
4. ~~删 `seg.rs:65-68` 空循环与 `ccn.rs:138-142` 死变量~~ **已兑现（§16.4）**：前者本轮删除，后者复核时已不在。

**第二梯队（1-3 人日，"代码下降且质量上升"的双赢项）**

5. ~~4 个表达式缓存 → 1 个泛型 `FormulaCache`~~ **已兑现（§18.2）**：收在 `einsum`，`static` 仍按类型分开，四个类型名删除；`[[ownership.unlisted]]` 三个旧名字一并清掉。
6. ~~`gemv` 输入校验上提为骨架方法~~ **已兑现（§18.3）**：三个 helper + `gemv_i8::layout` 删除，错误文案逐字保留。
7. ~~`op` 的 22 条 `Registration` + 22 个 `forward` shim 由宏生成~~ **已兑现一半（§20）**：两张 22 条的注册/内核表都成了一张一行一对的表（`op::REGISTRY`、`train::KERNELS`）；`forward` shim 那一半经实测撤回（只有 9 个单行委托，§20.5）。
8. ~~`codegen.rs` 的 11 处同形分支 → 一张族名表~~（**已做，§19**：12 行 `SKELETON_TABLE` + 双向断言）。
9. ~~把"只准降"改成真正的棘轮~~（**已做，§22**：登记值必须等于实测值，31 个陈旧登记值当轮下调）；"按类分档 `ccn`"这半边**明确不做**，理由见 §22.5。

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

- **架构层是本仓最强的资产**：唯一一条依赖规则被 6 项可执行断言守住，所有权表做到"一个类型一个主人"，默认构建的运行时第三方依赖为 0（`--all-features` 下 `dtype` 有 2 个可选依赖，§12.4），unsafe 收敛到单文件单 feature。这些指标放在国际开源坐标系里属于上游。
- **质量层是"刚上秤"的状态**：自建门禁首次量化出 282 条超标（ccn 251 条；复核为 274 / 244，§12.2），说明当前实现离自定标准仍有可观的系统性距离；且门禁本身有 5 处覆盖盲区（P1-7）与 1 处语义与文档不符（P3-5）。
- **鲁棒层出现了与架构水平不匹配的塌陷**：4 条 P0 全部在"不可信输入"路径上（rANS 分配、pickle 越界、checkpoint `expect`、DSA 切片），且都能用几行代码修好。这说明**"错误契约统一"没有自动变成"输入校验完整"**——类型系统保住了"错误怎么表达"，没保住"错误在哪里产生"。
- **交付面仍是 P0 切片**：22 个算子、1 条可执行后端、无传输/无集合通信/无 BPE/无融合；这不是缺陷，而是路线选择，但对外表述（"LLM runtime"）应与实际能力对齐。

### 9.2 推荐的最小样板切入点（如果要继续编码）

完整顺序见 §8.2（本节只是它的"最小样板"切面）。若只挑两件开始，就是 §8.2 第一梯队第 1 条与第二梯队第 5 条：

1. ~~`weights` 四连修（P0-1..4）~~ **已兑现（§15）**：`dsa`/`rans` 的输入校验已写成"先校验长度、再切片"的样板；样板尚未复制到 `zip64_extra` 与 `gguf` 的 `start+len`（§15.4 列为残余）。
2. `op` 缓存收口（4 → 1）：证明"泛型 + 泛型常量"能同时降低代码量与重复，而不引入运行期分派。
3. 这两步做完，`weights` 的债基线应从 84 条开始真实下降，门禁的棘轮才第一次"动起来"（注意它只拦"涨"，见 P3-5）。

### 9.3 明确不建议触碰

- `tests/architecture.rs` / `tests/contracts.rs` / `docs/ARCHITECTURE.md` 的三表结构：它们是本仓的核心治理资产，任何"简化"都会把可执行契约降级为散文。
- `diagnostic` / `dtype` / `op::einsum`（已并入 `op`）/ `prof::atom::Set`：单一事实源的样板。
- `op` 的 `Format` 骨架与"每格式一份 `forward`"：monomorphization 的选择是有代价但有理由的（`gemv.rs:1-26` 的取舍表），改成运行期 `match` 是性能倒退。
- `storage::mmap` 的 unsafe 岛：已经是教科书级的最小化，除非要做 Windows 支持（那时是 `cfg` 问题，不是 unsafe 设计问题）。

### 9.4 本次审计未覆盖 / 未知项（诚实边界）

| 项 | 状态 | 原因 |
| --- | --- | --- |
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
| --- | --- | --- |
| A 同一知识两份实现且必须同步 | 改一处不改另一处就是 bug | `op::REGISTRY`（`op/src/lib.rs:122`）与磁盘上的 22 对 `.toml`/`.k`，靠 `builtin_matches_disk` 平价测试绑定 |
| B 新增一项要改 N 处 | N ≥ 3 即说明缺一层声明 | `config::Key` 加一个键要改 5 处（`config/src/key.rs:3-4` 自述） |
| C 已有 const 断言/平价测试证明二者必须一致 | 一致性是硬约束而非风格 | `const _: () = same_order(&KERNELS, &op::REGISTRY);`（`train/src/store.rs:537`） |
| D 字段是闭集且只在编译期需要 | 运行期不需要分支 | `Set<E, WORDS>` 的 `FITS` 容量证明（`prof/src/atom.rs:56-59`） |

判据之外的重复**不算债**。据此，`gemv` 每格式一份 `forward`、`crc` 的两份实现（`weights/src/crc.rs:44` 与 `testkit/src/lib.rs:91`）都不进本节的减量清单。

### 10.2 已经做对的部分（禁止为了"更统一"而回退）

| 机制 | 现状 | 为什么它已经是"注解的替代品" |
| --- | --- | --- |
| `Closed` + `Set<E, WORDS>` | 6 个闭集共用一份契约（`prof/src/atom.rs:20-34`），位集唯一（`:44-151`） | 容量越界是编译错误，比任何宏生成的查表都强 |
| `dtype` 单一事实源 | `prof` 直接 `pub use dtype::DType` 并 `impl Closed`（`prof/src/atom.rs:313-328`） | ADR 0018 的落地样本：`F4` 只要加在 L1，所有层免费获得 |
| `KERNELS` ↔ `REGISTRY` 顺序断言 | `train/src/store.rs:512` 与 `:537-565`（const 版 `str` 比较） | 运行期零成本，且编译期就挡住漏注册 |
| `#[model]` 自检模块 | 每个算子生成 `__model_check`（`annotation/src/codegen.rs:382-441`） | 注解声明的 dtype/sample 与 `.k` 签名对不上就编不过 |
| `Sources::set` 的层级仲裁 | `config/src/source.rs:114-123` | "值 + 来源"是类型，不是散落的 `env::var` |

结论：**本项目的问题不是"注解化不足"，而是"注解只铺到了 op 层，没铺到它下游的三张平行表"**（见 §10.3 第 1-4 行）。

### 10.3 还能被表/泛型/宏吞掉的样板（按收益排序）

| # | 机制 | 现状与证据 | 生成/收敛方式 | 净减（推断） | 同步风险 |
| --- | --- | --- | --- | ---: | --- |
| 1 | 22 条 `Registration`（**已做，§20.2**） | `op/src/lib.rs:122` 一行一对 | `registry!` 宏展开成 5 个字段的结构 | **实测 −48**（原估 −100：宏本身 25 行 + 新增 14 行用例） | 低（长度由表数出，不是手写常数） |
| 2 | **算子目录的第三份投影** | `op::REGISTRY`（`op/src/lib.rs:122`）、`train::KERNELS`（`store.rs:512`，已被 `:537` 的 const 断言绑定；名字由 `stringify!` 派生，§20.3）、`nn`（`format.rs:88` + `nn/src/lib.rs:192,199` 硬编码两个算子名，**未与注册表绑定**） | 把 `format = "int4"` 写进 `ops/*.toml`，由注解生成 `WeightFormat::operator` 与两个常量 | −20 | **中**：`format.rs:168,195` 的断言要改指向 |
| 3 | 11 处同形族分支（**已做，§19**） | `annotation/src/codegen.rs:211-374`，每处 8 行仅模块路径不同 | `SKELETON_TABLE` 12 行 + 6 个函数 + 双向断言 | **+33**（预估的 −80 不成立） | 低（`__model_check` + 新增锁表测试） |
| 4 | 逐算子转接段 | `train/src/store.rs:619-774`，每算子解包后转 `op::*::forward` | 由 `ops/*.toml` 的输入签名生成，与 `KERNELS` 同源 | −100 | 中（涉及四个 `Inputs` 形状） |
| 5 | 4 份表达式缓存 | `gemv.rs:250`、`elementwise.rs:41`、`unary.rs:32`、`reduce.rs:31` 同一个 `OnceLock<Result<Formula, Error>>` | **已兑现（§18.2）**：`einsum::FormulaCache` + 每类型 `cache()` | −78（实测） | 低（已落） |
| 6 | 3 族格量化器 | `lattice.rs:153/387/431` 同一骨架套三种块宽（24/8/3×E8） | 泛型 `Quantize<const DIM: usize>`，差异只剩 `pack`/`fit` | −80 | 中（数值须与现测试逐位一致） |
| 7 | `config::Key` 5 张表 | `config/src/key.rs:34-79`（`file_key`/`env_suffixes`/`kind`/`doc`） | 一行一个键，四个方法各投影一列；**§13 改判**为 `impl Closed` | −20（并入 §13 的 −≈1 250） | 低 |
| 8 | `prof` 标量字段手写 | `prof/src/prof.rs:1104-1128` 逐字段 `dt`/`space`/`grade` 读取 | 字段表（同一函数 `:1091-1103` 的 `[(name, fn)]` 已是先例） | −40 | 低（`validate` 的跨记录不变量不在此列） |
| 9 | `ctype` 拼写 + 6 个校验 | `cutile/src/backend.rs:258-262` 的 3 臂 `match` 与 `:227-305` 的校验序言 | 3 行表 + 一个校验函数 | −40 | 低 |
| 10 | pickle 固定宽度族 | `weights/src/bin.rs:256-338`（`X/M/J/G/r/j/K/q/h` 十种编码） | 3 张 `(opcode, width, ctor)` 表 + 1 张闭包表；栈操作保留分支 | −60 | 中（须逐 opcode 对拍） |
| 11 | ZIP 字段与 64 位哨兵 | `weights/src/zip.rs:51-97` 偏移已是表；`:67-72` 是唯一实分支 | 哨兵收成一次 `Option` 查表；并补 `zip64_extra` 的 `size` 校验（正确性问题） | −10 | 低 |

合计**约 −600 行（推断）**。这十一条全部满足判据 A 或 B：不是"看起来像"，而是"两处必须同步"。

### 10.4 零引用簇：比合并更彻底的减量（本轮最强发现）

`weights/src` 下有 10 个模块，除 `lib.rs` 的 `pub mod` 一行外，**全仓无任何引用**（含 `tests/`、`src/`、其它 crate）；它们也全部出现在 `git status` 的 `??` 里——是**未提交的新增文件**。

| 模块 | 行数 | 仓内引用点 | 判定 |
| --- | ---: | --- | --- |
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
| --- | --- | --- | --- |
| (a) 不提交 | 保留在工作树，按 ADR 0004 的占位口径记为"未落地" | 0 | 没有近期接线计划时**首选** |
| (b) 提交但隔离 | 用 `format/`、`quant/` 子模块 + `#[cfg(feature)]` 收拢，`ARCHITECTURE.md` 同步标"未接线" | 一份 feature 表 | 需要在同一分支上继续开发这些模块时 |
| (c) 接线后保留 | `safetensors::File` 增加 `gguf` 后端，`axes` 成为 `Writer` 的可选压缩策略 | 真实接线代码 + 对拍 | 已排期要做 llama.cpp 权重互通时 |

### 10.5 二次合并评估：新候选逐个定案

上一轮只审了"零/近似零入边"的成员。本轮补审**依赖签名相同**的簇（依赖签名相同说明它们本来就同层，是最容易被误合并的一类）：

| 候选 | 依赖签名 | 判定 | 依据 |
| --- | --- | --- | --- |
| `exec`(188) + `shard`(524) | 都只有 `diagnostic` | **不合并** | 无同一知识的两份实现：`exec` 是提交图与线程几何（`exec/src/lib.rs:31-145`），`shard` 是拓扑秩布局（`shard/src/topology.rs:118-152`）；仅共享"后端未实现"这一状态，应在 `ARCHITECTURE.md` 标为同一档占位，而不是合并 |
| `tokenizer`(125) + `data`(233) | `tokenizer`:`diagnostic`；`data`:`diagnostic`+`tensor` | **不合并** | `data/src/lib.rs:3-5` 已声明"不拥有分词（`tokenizer`）"；两者是"trait 契约"与"形状即校验"两个不同机制 |
| `data` + `kv` + `storage` | 都是 `diagnostic`+`tensor` | **不合并** | 批校验（`data/src/batch.rs:16-37`）、页账（`kv/src/lib.rs:228-288`）、布局（`storage/src/lib.rs:67`）三者互不引用，且 `storage` 的处理是并入 `tensor`（§6.3 S2），不是并进这里 |
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
| --- | --- | ---: | --- | --- |
| P0 | §10.4 决定 3 434 行的去留（推荐 (a) 不提交） | −3 434 或 0 | 低 | 无（未跟踪，不产生 diff） |
| P1 | §10.3 的 1/3/5/7（宏、族表、泛型缓存、单表 Key） | −260 | 低 | 无 |
| P2 | §10.3 的 2/4（op 声明生成 `nn` 与 `train` 的平行表） | −120 | 中 | 需先给 `ops/*.toml` 加 `format` 字段 |
| P3 | §10.3 的 6/8/9/10/11（格量化泛型、`prof` 字段表、`ctype` 表、pickle 表、ZIP 哨兵） | −230 | 中 | P3 内每条都要逐位对拍 |
| 合计 | — | **≈ −4 000 行**（3 434 + 610），约占 40 478 生产行的 **9.9%**；crate 数 −1（`storage`，未做）；成员另 −7（`exec`/`kv`/`einsum`/`audit`/`regress`/`checkpoint`/`proto`，**已执行**，见 §11；不改变本表行数） | — | — |

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
| --- | ---: | ---: | --- |
| workspace 成员 | 33（32 + 根包） | **26（25 + 根包）** | `ls crates` 25 个 `Cargo.toml`；`tests/architecture.rs` 层表集合 = 成员集合 |
| 成员间依赖边 | 106（含 15 `+dev`） | **81（含 13 `+dev`）** | `docs/ARCHITECTURE.md:38` 与 `cargo metadata` 逐条相等（断言见 `tests/architecture.rs`） |
| 根包依赖 | 18 | **15** | 根 `Cargo.toml` `[dependencies]` 去掉 `audit`/`exec`/`kv` |
| 生产源文件（`crates/*/src`） | 166 | **166** | 文件只是换位置，数量不变 |
| 生产代码行（`crates/*/src`，含内联测试） | 53 250（首轮口径） | **53 329**（实测） | 前六条净 +75（注释与一个 12 行 helper），第七条净 +4（`pub mod proto` 与 `use`），合计 +79，**没有删除任何生产逻辑**。上一版写 54 869 是**漏计** `prim/build.rs` 622 与 `dump_bin.rs` 40（55 535 − 662 = 54 873）；而 39 721 是把根入口 1 544 重复计入（38 530 + 1 195）。两个数各自有各自的错因，现都按逐文件截断重测 |
| 债务键 | 282 | **282** | 34 条改路径（路径即身份）：`report` 20 + `snapshot` 8 + `einsum` 5 + `regress` 1；1 条 `lines` 按实测校正 |
| 测试目标 | 72 | **58** | 少掉 7 个 crate 的 lib + doc-test 目标 |
| 执行用例 | 1 037 通过 / 0 失败 / 1 忽略 | **1 037 通过 / 0 失败 / 1 忽略** | `cargo test --workspace --all-features`，退出码 0 |

> 本节整张表是 **2026-11-13 21:30 的合并快照**。HEAD `4e363270` 复核为：边 82（含 13 `+dev`）、债务 274 条、测试目标 62、用例 1 083、`crates/*/src` 55 718 行、生产源文件 171（§12.2）。§11.6/§11.7 的表格同样是快照，比较时以 §12.2 为现行值。

最后一行是本轮最重要的一条事实：**七条合并移动了 ~13 000 行代码，一条测试都没丢，也没新增**。这意味着"合并"在这里确实是**纯包装**——它换来的是更少的成员与更少的边，不是更少的代码。谁把"合并六个 crate"当成"减量"来理解，都会在第 11.6 节碰壁。

### 11.1 落点表：每条链最终落在哪

| 链 | 消失的目录 | 新位置 | 该 crate 里"拥有"声明的第一句话 |
| --- | --- | --- | --- |
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
| --- | --- | --- | --- | --- |
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
| --- | --- | --- |
| `exec` 的 `Launch`/`Node`/`Graph`/`Executor` 删除或标"占位" | 删它要先改 ADR 0004（"提交契约在 L4"），改 ADR 不是包装合并的一部分 | `engine` 生产代码里 `Graph`/`Launch` 引用数 = 0（现状已是 0） |
| `snapshot` 1 611 行拆 `format`/`decode` | 拆文件要同时改 7 条函数级键 + 1 条文件级键，与合并同批做会让 revert 不再原子 | 两个文件都 < 899 行；`cargo test -p llm-train` 不变 |
| `Facts` 吸收 `tape_nodes`/`weight_entries`，`resume` 校验 6 个事实 | **这是行为变更**（`resume` 会开始拒绝更多快照），必须独立提交 + 对拍，混进合并等于把一个语义改动藏进搬家 | `train::Checkpoint` 与 `snapshot::Facts` 字段数相等 |
| 补字节级**黄金向量**测试 | 现有断言只到 `bytes.len()`；写黄金向量要先把格式冻住，而格式正因"拆文件"待改 | 固定十六进制串成立，改格式必须同时改测试 |
| `weights::leb` 改 `pub` 并删 `snapshot` 的 `varint` | 两份 LEB128 的规范形式拒绝规则要逐条对齐（`weights/src/leb.rs:11,55` vs `snapshot/mod.rs` 的 `varint`/`Cursor::varint`），属于"零漂移"对拍，不该在搬家提交里做 | `snapshot` 里 `fn varint` 消失，字节输出不变 |

这张表是本节最有用的部分：它把"合并完成"和"质量完成"分开记账。把六条链读成"架构问题已解决"是错读——**它解决的是成员数与边数，不是重复与复杂度**。

### 11.5 执行序与每批的验证

| 批 | 内容 | 实际验证 | 结果 |
| --- | --- | --- | --- |
| **B0 门禁硬化** | `tests/architecture.rs` 错误边界扫描集改为从层表派生；`ARCHITECTURE.md` 三张表按新形状改写；新增 `docs/adr/0019-packaging-merge.md` | `cargo test --test architecture` | 6/6 绿；派生后立刻抓到第 11.3 节第 3 条的 `String` 报错 |
| **B1 零成本两条** | `regress → tool`、`exec → engine` + `kv → engine`（两条并在一次里做，避免两次改 `engine/src/lib.rs`） | `cargo test -p llm-engine -p llm-serve -p llm-tool` | 26 + 9 + 9 全绿；`engine` 三个文件 1 303 行 |
| **B2 `checkpoint → train`** | 目录搬家 + 8 条键改写 + `lines` 校正 | `cargo test -p llm-train` + `-p llm-tool --test gate` | 全绿；文件级键指向 `train/src/snapshot/mod.rs` |
| **B3 `einsum → op`** | 目录搬家 + `Bits` 改用 `prof::atom::Closed` + 5 条键改写 | `cargo test -p llm-op` | 159 通过；`String` 报错当场暴露并修掉 |
| **B4 `audit → cutile`** | 13 个文件搬进 `report/`、20 处 `crate::` 路径改写、20 条键改写 | `cargo test -p llm-cutile` + `cargo test --test contracts` | 全绿；`cutile` 9 469 行 / 26 文件 |
| **B5 `proto → serve`** | 4 个文件搬进 `serve/src/proto/`、`crate::` 改成 `crate::proto::`、`pub use self::…`、层表/边表/所有权表/README/债务键同步 | `cargo test --test architecture --test contracts` + `cargo test -p llm-tool --test gate` + 全仓测试 | 6/6 + 6/6 + 1/1 全绿；**1 037 通过 / 0 失败 / 1 忽略**，与合并前逐项相同 |

排序依据是**每次提交的爆炸半径**，不是重要性：B1 不碰债务键、不碰 ADR；B2 碰 1 条文件级键；B3 碰闭集词汇表；B4 同时改最大 crate 与 20 条键。B4 放最后，前面任一步失败都不会让最大的 crate 停在半合并状态。

### 11.6 收益核算：合并省的是"成员与边"，不是行数

| 项 | 变化 | 能不能算"代码下降且质量上升" |
| --- | --- | --- |
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
| --- | ---: | ---: | --- |
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
| --- | --- | --- |
| 架构契约（成员集合、边方向、所有权、错误边界） | `cargo test --test architecture --test contracts` | 6/6 + 6/6 通过 |
| 文档链接 + 棘轮门禁 | `cargo test -p llm-tool` | 27 通过（`regress` 的 6 个用例已计入） |
| 全仓测试 | `cargo test --workspace --all-features` | **1 037 通过 / 0 失败 / 1 忽略**，与合并前逐项相同 |
| 随迁测试没有变成死代码 | `cargo test -p cutile` 中的 `report::*` | 47 个用例确实执行（`report` 已在 `cutile/src/lib.rs` 声明为 `pub mod`） |
| 合并后二进制仍工作 | `cargo run --bin cutile -- doctor` | 退出码 0，协商/plan/布局各段正常输出 |
| 严格文档构建（CI `doc` job 的口径） | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` | 退出码 0（修复前 27 处断链，见下） |

第三条是关键：**用例数不变**，说明七次吸收都是搬家而非重写；如果某个子模块忘了在 `lib.rs` 里声明，用例数会掉而编译仍绿——这是合并类改动最需要在 CI 里盯住的假绿形态（`cargo test` 只数跑起来的用例，不数文件里的 `#[test]`）。

最后一条是本轮顺带修掉的既有缺陷，不属于合并本身，但会让 `doc` job 必红：全仓 **27 处** rustdoc 断链，两类成因——

| 成因 | 位置 | 修法 |
| --- | --- | --- |
| 公开文档链到私有项（`pub(crate) mod` 或私有 `const`/`fn`；本列是修复当时的路径，`report/` 之后已拆平） | `cutile/src/report/mod.rs:15`（3 处）、`report/witness/mod.rs:37-38`（5 处）、`report/witness/evidence.rs:51`、`cutile/src/ast.rs:17,160`、`weights/src/bin.rs:4`、`weights/src/axes.rs:6`、`weights/src/ir.rs:159,213`、`tool/src/call.rs:3`、`tool/src/strip.rs:3` | 改成代码块（这些名字是内部实现，不该出现在公开 API 的链接图里） |
| 正文里的方括号被当成链接 | `weights/src/ntt.rs:3` 的 `Z_q[x]`、`weights/src/keccak.rs:1` 的 `Keccak-f[1600]`；`op/src/*.rs` 的 `[`elementwise`]`/`[`unary`]`/`[`reduce`]`/`[`einsum`]` 缺少 `crate::` 前缀（10 处） | 前者加反引号，后者补 `crate::` 前缀 |

这条值得单独记的原因是：`tests/architecture.rs` 与 `tests/contracts.rs` 都只扫源码文本，`cargo test` 也不会碰文档，于是**"新增一个私有子模块 + 在公开文档里链它"可以一路绿到 `doc` job 才失败**（而 `doc` job 在 CI 里排在 `test` 之后）。

### 11.8 重议条件（写死，避免合并变成永久决定）

| 链 | 什么时候应重新拆开 |
| --- | --- |
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
| --- | --- | --- |
| §6.0 全量裁定表（26 行） | — | **已重写**：只列当前 25 个 crate + 根包；七个已消失成员在表下单列"追溯"，不再混在"保持/合并候选"里 |
| §6.1 `audit↔cutile`、`einsum↔op`、`kv↔engine`、`checkpoint↔train` | 不合并 | 已合并；边界理由仍然成立，只是被"单消费者吸收"这一新类别覆盖 |
| §6.2.2 `exec → engine` | 驳回 | 已合并；`Launch`/`Graph` 仍无人使用（§11.4 第 1 行） |
| §6.5「不做」档 | 含 `kv`/`exec` | 两者已移出 |
| §10.5 结论 | "0 个新 crate 合并" | 数字改为 6，理由是**产品裁决**而非新证据 |
| §10.6 合计 | 成员另 −6（待执行） | 已执行 |

### 11.10 文档本身的自检（本次修订改了什么）

这一小节回答"文档为什么还没优化到位"，并留下可复核的改法。上一版的问题是**叠加式改判**：第十一节推翻了前文的判定，但前文的数字、路径与执行序没有回填，读者必须自己拼三处才能回答"这个成员最后怎么处置"。

| 上一版的问题 | 本次改法 | 证据 |
| --- | --- | --- |
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

---

## 十二、复核记录：这份报告现在准不准（HEAD `4e363270`，2026-11-14 06:47）

### 12.1 复核做了什么

只读复核，不改任何源码。执行的命令与结果：

| 命令 | 结果 | 用来核对 |
| --- | --- | --- |
| `cargo test --workspace --all-features` | **62 个测试目标 / 1 083 通过 / 0 失败 / 1 忽略**，退出码 0 | §1 的"用例数"、§2.1 的"测试目标"、§11.6 的"用例数不变" |
| `cargo test --test architecture --test contracts` | 8/8 + 6/6 通过 | §3.1 与 §4.3 的"门禁全绿" |
| `cargo metadata --format-version 1 --no-deps` | 26 个成员；69 条普通依赖 + 1 条 build 依赖 + 13 条 dev 依赖；去重后 **82 = 69 + 13** | §4.3、§6.0 的入边/出边、§12.3 的 dtype 新边 |
| 逐文件截断求和（首个 `#[cfg(test)]` 处） | `crates/*/src` 生产 38 620、总 55 718；根入口 1 196 / 1 545 | §2.1 的四行规模数字 |
| `[[gate.debt]]` 解析 | **274 条**（字样出现 275 次，`lints/vocabulary.toml:71` 是注释）；ccn 244 / nloc 28 / call 23 / lines 8 | §2.2、§1 的"存量超标" |
| `git status --short` / `git log --oneline` | 工作树干净（只剩 `?? .junie/`）；首轮之后有 8 个提交（`de411432`…`4e363270`） | §2.3 的"未跟踪比例" |

### 12.2 差异表（报告值 → HEAD 实测）

| 指标 | 报告原值（2026-11-13） | HEAD 实测 | 成因 |
| --- | ---: | ---: | --- |
| workspace 成员 | 26 | 26 | 无变化（七条吸收之后没有新成员） |
| 生产源文件 | 170 | 175 | `dtype` 拆成 6 文件、`cutile` 报告层拆平（§12.3） |
| 外部集成测试文件 | 8 | 12 | `dtype` 新增 4 个契约测试文件 |
| `crates/*/src`（含内联测试） | 53 329 | 55 718 | 同上（+2 389） |
| 全仓生产行（含内联测试） | 55 535 | 57 925 | 同上 |
| 全仓生产行（不含 `#[cfg(test)]`） | 38 530 | 40 478 | 同上（+1 948） |
| 测试代码行 | 19 163 | 20 272 | 内联 17 098 + 外部 3 174（12 文件） |
| 测试目标 / 用例 | 58 / 1 037 | **62 / 1 083** | `dtype` 4 个新测试目标 + 46 个新用例 |
| 债务条目 / 度量 | 282 / 311 | **274 / 303** | 条目实为 274（旧口径把 `vocabulary.toml:71` 的说明文字当成条目）；ccn 244、call 23 |
| 依赖边 | 81 | **82** | `dtype` 新增 4 条入边（`prof`/`tensor`/`weights`/`annotation`），吸收链未再变 |
| `Cargo.lock` 包数 | 100（27 成员 + 73） | **101（26 成员 + 75 第三方）** | `dtype` 的可选 `serde`/`bytemuck` 与 dev `serde_json` |
| 第三方运行时依赖 | 0 | **默认 0；`--all-features` 2** | `crates/dtype/Cargo.toml:12-21` |
| `trait` / `impl` | 31 / 363 | 32 / 390 | `dtype` 的 `Format`/`BlockFormat` 族与 `serde` 适配 |
| `macro_rules!` | 2 | 5 | 首轮漏计 `dtype` 的三个（`format.rs:94,116`、`blocked.rs:12`） |
| `#[test]` 属性 | 975 | 1 015 | 同上 |
| 文档行数（`ARCHITECTURE.md` / `vocabulary.toml`） | 236 / 1 229 | 249 / 1 281 | 四次提交的文档同步 |
| 最差 `ccn` / 最长函数 / 最大文件 / `> 899` 行文件数 | 77 / 233 / `prof.rs` 1 712 / 8 | **不变** | 四处热点没有被新提交改动 |

### 12.3 结构变化：首轮之后的 8 个提交改了什么，影响前文哪些结论

1. **`dtype` 从"一个 313 行的标量闭集"变成 L1 精度事实源**（`de411432`、`05e2b824`、`06bb7c13`，另有 `a8d592b4` 的边界诊断补漏）：6 个文件 2 136 行，`DType` 24 变体 / `QType` 25 / `MxType` 7 / MX 元素子集；新增 `prof → dtype`、`tensor → dtype`、`weights → dtype`、`annotation → dtype` 四条边。影响：§3.2 的"22 个变体"已过时（§4.2 已改）；§6.1 的"`dtype` ↔ `tensor` 不合并"理由仍然成立（`prof` 会反向依赖 L4），但入边由 3 变 4。
2. **`data` 重写**（`c8fd45da`、`ee0cedb9`）：233 → 630 行、4 文件，`Batch`/`Split`/`Dataset`/`Chunks` 重新对齐几何与游标语义。影响：§4.2 的"规模小"改为"中"；§6.0 的 LOC 已更新，处置（保持）不变。
3. **`cutile` 删掉 `report/` 与 `witness/` 两层目录**（`87ad7486`），随后 `4e363270` 收口设备探测、Metal 驱动、MSVC 查找与模板渲染：原 `audit` 的 13 个文件现在平铺在 `cutile/src/*.rs`，`device.rs`/`gate.rs`/`pipeline.rs` 变私有。影响：本报告里 **12 处** `cutile/src/report/…` 形式的引用（§5 的 P2-9/P2-10、§6.1、§6.3 的文件拆分表、§11.7.1 的断链清单）**只剩文件级可回链**；§6.3 的"把 `report/mod.rs` 700 行拆成 `types.rs`"这条方案**作废**——该模块已不存在。
4. **门禁与工具进入版本库**：§2.2 末行"该门禁是否在 HEAD 中存在 = 否"已改为"是"（HEAD 的 `lints/vocabulary.toml` 有 274 条债务）；§2.3 的"127 条 `??`"与"`weights` 29/33 未跟踪"已改为"工作树干净"。这意味着**"未发布/在途"这一限定语在全文失效**：这些代码现在就是发布内容。

### 12.4 复核后的评级

综合评级**维持 A-（82/100）**，三条分量各自的口径变化如下：

- 依赖最小化 **A+ 99 维持**，但口径必须收窄为"**默认构建 0；`--all-features` 下 `dtype` 打开 `serde`/`bytemuck` 两个可选依赖**"。这是本轮唯一一条"报告写了绝对数、事实却依赖 feature"的结论，已改。
- 复杂度 **C+ 65 维持**：274 条债务、最差 `ccn=77`（`weights/src/bin.rs:execute`）、最长函数 233 行（`annotation/src/codegen.rs:render_model`；§19.7 后该位置已是 202 行的 `prof/src/prof.rs:validate`）、8 个文件超 899 行（`prof.rs` 1 712、`snapshot/mod.rs` 1 611、`store.rs` 1 530、`cache.rs` 1 246、`syntax.rs` 1 229、`atom.rs` 1 200、`dsa.rs` 923、`cap.rs` 913）。
- 正确性 **B+ 85 维持**：1 083 用例全绿，负例结构未变。

### 12.5 本轮没有复核的部分（诚实边界）

- **行号级引用只抽样重锚**：`weights`/`op`/`prof`/`config`/`tool` 未在四个新提交中改动，其 `文件:行` 可继续引用；`cutile`/`data`/`dtype`/`train::snapshot`/`serve::proto` 的行号没有逐条重锚，请按"文件级成立、行号可能漂移"读。
- **对标数字未重测**：§8 里 Burn 的 369 340 行仍取自 `docs/10k-analysis.md`（该文自己已过时，见 P3-6）。
- **性能/覆盖率/依赖审计仍未做**：真机 roofline、`cargo-deny`、`jscpd` 与首轮一样没有实测数据。
- **新发现的一处文档不一致**：`docs/ARCHITECTURE.md:38` 写"共 82 条边"，同一文件 `:69` 仍写"边从 106 条降到 81 条"。`tests/architecture.rs` 只断言边集与总数相等，不检查叙述句，所以两处都能绿——这是"可执行文档"目前唯一漏掉的一类漂移：**数字在正文里另写一遍**。

### 12.6 四条 P0 的现场复核（HEAD `4e363270`）

| 条目 | HEAD `4e363270` 现场 | 结论（2026-11-14 二次复核后） |
| --- | --- | --- |
| P0-1 | `rans.rs:300` 读 `src_len`，`:325` 才 `vec![0u8; src_len]`；中间只有 lane 的 `split_at_checked`（`:313-319`），没有"lane 长度之和 ≥ `src_len`"的交叉校验 | 修复前**仍在**；现已修（§15.2） |
| P0-2 | `bin.rs:445-458` 的 `chunks(2)` 仍取 `chunk[1]`；`pop_until_marker`（`:305`）返回的计数完全由文件字节决定，奇数长度即越界 | 修复前**仍在**；现已修（§15.2） |
| P0-3 | `train/src/snapshot/mod.rs:651` 的 `expect("快照条目类型在标签闭集内")`，`tag` 在 `:895` 仍返回 `Result` | 修复前**仍在**；现已修（§15.2） |
| P0-4 | `dsa.rs:353-367`（`pk_decode`）与 `:376-391`（`sig_decode`）三处切片无长度校验；`Envelope::decode`（`:100-103`）只按文件字段 `take` | 修复前**仍在**；现已修（§15.2，含新发现的 P0-4b） |

也就是说：报告里唯一"必须立刻修"的那一组结论**没有被时间冲掉**，四条都能在 5 分钟内改完（`?`、奇偶校验、长度前置校验、`src_len` 上界）——**本轮已经全部按这个规模改完**，实测净增 198 行、用例 1 052 → 1 057，见 §15。

---

## 十三、`crates/config` 的移除分析：2 507 行服务 3 个键，其中 583 行根本不是配置

### 13.0 结论先行

> **更新（2026-11-14）**：A1 已执行，落地记录与实测数据见 §13.11；A2 也已执行，但**没有按本节建议改名 `llm-toml`**——用户裁决"改名不是解耦"，改成了并进 `diagnostic::toml` 并删除成员，落地记录见 §13.12、决策见 ADR 0021。本节 A2 的路线 B/C/D 只是当时的备选记录，不再作为建议。

事实：`crates/config` 是 9 个源文件 2 507 行（另 `Cargo.toml` 15 行），对外登记 **3 个键**（`config/src/key.rs:29-31`），即 **836 行/键**。

结论：这个 crate 是**两件互不相干的资产被塞进一个包**，必须分开裁决，否则"删不删"永远吵不完：

| 资产 | 行数 | 谁在用 | 能否消失 |
| --- | ---: | --- | --- |
| **设置层**：分层优先级、来源、键闭集、身份、路径、错误 | 1 577（63%） | 只有 `cutile`（67 处）与根包（13 处，均含内联测试），根包本身经 `cutile` 取值 | **能**：并入 `cutile`，`config` 作为"配置"这一概念消失 |
| **TOML 语法层**：文本 ↔ `Table`/`Value` | 583（23%） | `op`、`prof`、`tool`、根测试、`fuzz`(dev)，跨 L1/L2/L3/L7 四层 | **不能**：只能换宿主或改名 |
| **值形状**：`Kind` 五种形状 + `Value` | 247（10%） | 只有设置层自己；且 5 种形状里**只有 `Path` 可达** | **能**：随设置层一起删 |
| 入口与文档 | 100（4%） | — | 随包消失 |

因此"移除 `crates/config`"的可执行答案是**两个独立提交**：
**A1** 设置层（1 577 + 247 行）并入 `cutile`；**A2** 剩下的 583 行改名为 `llm-toml`。
结果是：成员 26 → 26、边 82 → 82、源码行数 **−≈1 250（推断，§13.5 给出组成式）**、债务条目 −4 条起、`[closed]` 豁免 −1 条、`tests/architecture.rs` 的错误类型例外 −1 条。

如果坚持"成员数必须 −1"，唯一零新增边的落点是并入 `diagnostic`（§13.4 路线 B）；
把 583 行塞进 `tool`（路线 C）不推荐，理由见 §13.4 与 §13.8。

### 13.1 事实基线：它由哪几块组成，各自有多少消费者

| 部分 | 文件（行数） | 债务条目 | 生产消费者（可回链） | 处置 |
| --- | --- | ---: | --- | --- |
| TOML 语法 | `toml.rs` 583 | 8 | `op/src/decl.rs:12`（+`op/src/lib.rs:350` 的 `load`）、`prof/src/lib.rs:35`（+`prof.rs` 内解析）、`tool/src/gate.rs:8`、根 `tests/architecture.rs:282`、`tests/contracts.rs:94`、`fuzz/tests/parsers.rs:42` | A2：改名 `llm-toml` |
| 值形状 | `value.rs` 247 | 3 | 只有 `config/src/source.rs`（`key.kind().from_toml(..)`，`source.rs:190`） | 随 A1 删除 |
| 分层与来源 | `source.rs` 529 | 4 | 只有 `Settings` 内部 | A1：收敛为 3 个入口 |
| 设置对象 | `settings.rs` 345 | 0 | 设置层方法共 18 处调用（含内联测试）+ 根包 2 处 `load` | A1：搬入 `cutile` |
| 键闭集 | `key.rs` 194 | 0 | 只有 `Settings` 内部（`entry`/`describe` 用） | A1：改 `impl Closed` |
| 身份 | `app.rs` 150 | 0 | **只有 1 个实现**（`app.rs:74-81`） | A1：降为常量 |
| 路径 | `paths.rs` 207 | 0 | `cutile` 6 处、根包 1 处 | A1：原样搬 |
| 错误 | `error.rs` 152 | 1 | 只有 `Settings`/`source` 返回它 | A1：改用 `diagnostic::Error` |

规模旁证：2 507 行里 **56 个 `#[test]`**（`app` 5、`error` 3、`key` 6、`paths` 4、`settings` 8、`source` 11、`toml` 13、`value` 6），占全仓 1 015 个 `#[test]` 的 5.5%。

### 13.2 逐 API 取证：谁真的在用

| API | 生产使用 | 证据 | 处置 |
| --- | --- | --- | --- |
| `Settings::load` / `defaults` | `cutile` 与根包 | `src/main.rs:247`、`src/main.rs:588`（内联测试）、`cutile/src/{bench,build,device,doctor,execution,gate,gc,pipeline,verify,witness,builtin,cache}.rs` | 保留（搬进 `cutile`） |
| `cache_dir` / `work_dir` / `data_root` | 8 / 6 / 3 处调用（含内联测试） | `cutile/src/cache.rs:388`、`build.rs:187,304,310`、`device.rs:212,534`、`witness.rs:339,350`、`builtin.rs:75`、`gc.rs:54`、`src/bin/cutile.rs:105,570,799` | 保留 |
| `apply_override`（`--set`） | 根包 1 处生产 + 3 处测试 | `src/bin/cutile.rs:72`、`cutile/src/{gc.rs:31,build.rs:486,doctor.rs:333}` | 保留 |
| `describe`（`[配置]` 段） | 1 处 | `src/bin/cutile.rs:195` | 保留（收敛为 ≈60 行） |
| `origin` / `entry` / `sources` / `roots` / `from_sources` | **0 处**生产 | 只在 `settings.rs:199-337` 的自测里出现 | 可删 |
| `Key` / `Layer` / `Origin` / `Sources` / `Kind` / `Value` / `App` | **0 处**外部引用 | 全仓按 `config::`/`llm_config::` 前缀 grep 逐符号计数 | 随 A1 删除或降为私有 |
| `cutile::toml` / `prof::toml` 再导出 | **0 处**引用 | `cutile/src/lib.rs:77`、`prof/src/lib.rs:35` | 直接删（已是死公开面） |

### 13.3 为什么"删掉设置层"不是口味问题（5 条可回链证据）

1. **泛型化没有第二个实例**：`App` 只有一个实现 `Cutile`（`app.rs:74-81`），文档里的"`llm`/`train`/`serve` 各用自己身份"是假设（`app.rs:10-12`），而根二进制实际写的是 `Settings::<config::Cutile>`（`src/main.rs:247`）——它借用的是 `cutile` 的身份，不是自己的。
2. **五种值形状只有一种可达**：`Key::kind()` 三个键全部返回 `Kind::Path`（`key.rs:66-70`），`value.rs:12-28` 的 `Text`/`Int`/`Bool`/`Choice` 及其解析分支（`value.rs:41-67`）没有任何生产入口。
3. **键闭集被迫手写并申请豁免**：`key.rs:31` 的 `ALL: [Self; 3]` 需要 4 张平行 `match` 同步（`key.rs:34-79`），`lints/vocabulary.toml:22-23` 为此给 `config` 开了 `[closed.exempt]`，理由写着"L1：设置键是配置层自己的词汇表"。并入 `cutile` 后它变成 `prof` 的下游，可 `impl prof::atom::Closed` + `inherent_names!`：**豁免删除、P2-3 消失**。
4. **全仓最后一个非 `diagnostic` 错误类型**：`crates/diagnostic/src/lib.rs:6-7` 自述"旧形态只剩 `llm-config` 自带的 `Error`"；`tests/architecture.rs:326-339` 专门为它写了一个例外分支。并入后这个例外可以删。
5. **单消费者 + ADR 0019 三条判据全中**：唯一生产消费者是 `cutile`（根包经 `cutile` 取值）；设置层不含"第二方无法复制的知识"（无第二份实现、无独立版本节奏）；落点是存活方的子模块，方向合法（`llm → cutile` 已存在）。

### 13.4 四条移除路线与方向合法性证明

前置规则只有一条且可执行：`tests/architecture.rs:80-103`——依赖只能指向更低的层，或同层更靠左的成员。语法层的消费者分布在 L1（`tool`）、L2（`op`/`prof`）、L3（`cutile`）、L7（根包），所以宿主必须 **≤ L1 且不右于 `tool`**。L1 的成员是 `diagnostic`、`dtype`、`testkit`、`llm-config`、`tool`（`docs/ARCHITECTURE.md:13`）：`dtype` 是精度闭集、`testkit` 是 dev-only 夹具，语义都不符，于是**真实选择只有三个**：`diagnostic`、`tool`、独立 crate。

| 路线 | 语法层落点 | 设置层 | 成员 | 边 | 源码行数（推断） | 主要代价 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **A（推荐）** | 独立 `llm-toml`（改名，583 行） | 并入 `cutile` | 26 | 82 | −≈1 250 | 成员数与边数都没有收益 |
| B | `diagnostic::toml` | 并入 `cutile` | 25 | 76 | −≈1 250 | 错误契约 crate 同时拥有数据语法（两个变更原因） |
| C | `tool::toml` | 并入 `cutile` | 25 | 79 | −≈1 250 | `tool` 成为 `op`/`prof`/`cutile` 的生产依赖，静态门装进运行路径 |
| D | 无（消灭运行期 TOML） | 并入 `cutile` | 25 | ≤76 | −≈2 400 | 破坏 `cutile.toml` 与"全仓唯一 TOML"承诺，≥10 个提交 |

边的算账：`config` 当前有 6 条入边——`cutile→`、`op→`、`prof→`、`tool→`、`llm→`、`fuzz +dev→`（`docs/ARCHITECTURE.md:41-63`）。
路线 B 之所以"零新增边"，是因为这 6 个成员**全都已经依赖 `diagnostic`**（`docs/ARCHITECTURE.md:41-65` 与 `crates/diagnostic/src/lib.rs:5` 的"除 `testkit` 外全部成员"）；
路线 C 需要新增 `op→tool`、`prof→tool`、`cutile→tool` 三条（`llm→tool` 已存在）。

### 13.5 推荐执行序：A1 设置层归位（单独一次提交）

1. 新建 `crates/cutile/src/settings/`（4 文件，估 ≈420 行，**推断**）：
   - `paths.rs`：`Roots` 原样搬（平台目录与 `HOME`/`XDG_*`/`APPDATA` 解析，`paths.rs:82-141`）；
   - `source.rs`：只留 `file`/`env`/`cli_override` 三个入口 + `schema`/未知键校验（`source.rs:170,213,245,279,291`），把 `Layer`/`Origin`/`Sources` 的通用机制收敛成"值 + 三值来源标签"；
   - `settings.rs`：`Settings` 去掉类型参数，字段就是 3 个 `Option<PathBuf>` + 3 个来源；`load`/`defaults`/`cache_dir`/`work_dir`/`data_root`/`apply_override`/`describe` 签名不变；
   - `key.rs`：3 个键改 `impl prof::atom::Closed`（`ALL` 切片）+ `inherent_names!`，`file_key`/`env_suffixes`/`doc` 各一列。
2. 删除：`app.rs`（150 行，`DIR`/`FILE`/`ENV_PREFIX` 变常量）、`value.rs`（247 行，3 个 `PathBuf` 槽取代 `Kind`/`Value`）、`error.rs`（152 行，改用 `diagnostic::Error` 的 `Invalid`/`Missing`/`Io`）。
3. 调用点改写：`crates/cutile/src` 67 处、根包 13 处 `Settings`/`Roots`，`Settings::<Cutile>` → `Settings`。
4. 删两处死再导出（`cutile/src/lib.rs:77`、`prof/src/lib.rs:35`）。
5. **必须逐字保住**的 8 条用户可见承诺（`README.md:136-152`）：四层优先级、`CUTILE_CACHE_DIR` 与兼容名 `CUTILE_CACHE`、未知键报错（不是忽略）、`schema = 1` 校验、配置文件里的相对路径按**文件所在目录**解析、环境变量与 `--set` 的相对路径按**启动目录**解析、`data.root` 一次性替换 `ops/`+`devices/`+`prim/` 三层、`cutile doctor` 的 `[配置]` 段逐键打印"值 ← 来源"。
6. 56 个内联测试的去向：`app` 5、`value` 6、`error` 3 全部随代码删除；`key` 6 → 保留 ≈3（闭集往返）；`settings` 8 → 保留 ≈6；`source` 11 → 保留 ≈8；`paths` 4 保留。**用例总数会下降**，必须在提交信息里写明"删掉的是被删类型自己的测试"，这不是放宽判据。

### 13.6 A2 语法层改名（第二次提交）

1. `crates/config/src/toml.rs` → `crates/toml/src/lib.rs`，包名 `llm-config` → `llm-toml`；`crates/config` 目录删除。
2. 4 类消费者改 `use`：`op/src/decl.rs:12`、`prof/src/lib.rs:35` + `prof.rs`、`tool/src/gate.rs:8`、`fuzz/tests/parsers.rs:42`、根 `tests/architecture.rs:282` 与 `tests/contracts.rs:94`。
3. 债务键：`crates/config/src/toml.rs:*` 8 条改路径到 `crates/toml/src/lib.rs:*`；`value.rs` 3 条与 `error.rs` 1 条**必须删除**（对应函数不再存在，`Gate::stale` 会因过期条目失败）；`source.rs` 4 条随 A1 重测改写。
4. `lints/vocabulary.toml`：删 `[closed.exempt]` 的 `config` 条目（`:22-23`）；`[[ownership.unlisted]]` 的 `config` 行（`:80-81`，7 个名字）拆成"`toml` 的 `Table`/`Value`/`ParseError`"与"随 A1 消失的 4 个设置类型"。

### 13.7 文档与门禁连带改动（逐处）

| 位置 | 现在 | 改成 |
| --- | --- | --- |
| `docs/ARCHITECTURE.md:13` | L1 含 `llm-config` | L1 含 `llm-toml`（或按 B 并入 `diagnostic`） |
| `docs/ARCHITECTURE.md:38` | 共 82 条边 | 路线 A 不变；B 改 76；C 改 79 |
| `docs/ARCHITECTURE.md:41-42,45,59,61,63,65` | 6 条含 `llm-config` 的边与叶行 | 按路线替换名字或删除边 |
| `docs/ARCHITECTURE.md:84` | `Settings`/`Key`/`Roots`/`App` 属 `llm-config` | 改属 `cutile`（`toml` 的 `Table`/`Value`/`ParseError` 改属 `llm-toml`） |
| `docs/ARCHITECTURE.md:163,209` | 把 `config` 列在"不在契约可达范围" | 同步删名 |
| `README.md:48,76,99,131-134,152` | "值与目录的唯一来源是 `llm-config`" | 改为 `cutile`（缓存/工作目录本来就是 `cutile` 的）；`Settings::<你的应用>` 的 ADR 0006 段落整段改写 |
| `docs/adr/0006-config-mechanism-vs-identity.md` | 机制与身份分离 | 被新 ADR 取代（身份只剩一个实例，分离不再成立） |
| `tests/architecture.rs:326-339` | 为 `config::Error` 开例外 | 删该分支 |
| `tests/architecture.rs:282,390`、`tests/contracts.rs:94` | `config::toml` 是唯一解析器 | 改为新 crate 名 |

### 13.8 验收判据（零漂移）

- `cargo test --workspace --all-features`：0 失败；用例数下界 = 1 083 − 被删类型的测试数（提交信息列明）。
- `cargo test --test architecture --test contracts`、`cargo test -p llm-tool --test gate`、`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`、`cargo clippy --workspace --all-targets --all-features -- -D warnings`。
- **行为对拍**（同一二进制、HEAD 与新树各跑一次，逐字节比对 stdout）：无配置文件 / 有文件 / 仅环境变量 / `--set` / 三种相对路径基准 / 未知键与 `schema` 不匹配两类错误文本；`cutile doctor` 的 `[配置]` 段必须逐字相同。
- 结构断言：`crates/config` 目录消失、`grep -rn "llm-config" crates src tests` 零命中、成员与边数等于所选路线的目标值。

### 13.9 明确不建议做（附理由）

1. **不要为了成员 −1 把 583 行塞进 `diagnostic`**（路线 B）：`diagnostic` 今天是"全仓唯一的错误类型"（`crates/diagnostic/src/lib.rs:1-4`），加进数据语法后它有两个变更原因；换来的只是成员数 −1 与边数 −6，而 §13.0 的收益（−≈1 250 行、删 7 个零消费类型、删两条门禁豁免）在 A 路线里同样拿得到。
2. **不要塞进 `tool`**（路线 C）：`tool` 是仓库自己的静态门（`crates/tool/src/lib.rs:1-11`），让 `op`/`prof`/`cutile` 依赖它，等于把词法/复杂度/文档检查装进推理路径。
3. **不要重写 `toml.rs` 来"减行"**：它那 8 条债务（ccn 6–22）是 TOML 子集文法本身的复杂度，重写只会换一组基线（§10.1 的判据），不会消失。
4. **不要现在做路线 D**：用户配置格式是 `README.md:136-152` 的对外承诺，破坏它需要独立 ADR 与弃用周期；在 A/B/C 之前做 D 属于把两件风险叠在一起。
5. **不要保留 `cutile::toml` / `prof::toml`**：两处再导出今天已是零引用的公开面，留着只会让"谁拥有 TOML"继续含混。

### 13.10 与前文结论的关系

- **改判 4 处**：§4.2 的 `llm-config` 行（"基本干净"→ 拆分）、§6.0 的处置列（"内部整合"→ 拆分并入）、§6.4 的 W6、§10.3 的第 7 项（均已就地加注「§13 改判」）。
- **不变 2 处**：§6.4 的 S4"`config/src/toml.rs` 583 行是全仓唯一 TOML 解析器，不动"在 A/B/C 下仍成立（只有 D 才动它）；§7.1 的"`json.rs` ↔ `toml.rs` 不是重复"仍成立——A2 之后它们是两个 crate 的两种语法。
- **P2-3 的处置升级**：从"允许本地保留但必须登记"变成"随吸收消失"。
- **本节未做**：任何代码改动。A1/A2 涉及 80 处调用点与 8 条用户可见承诺的对拍，应各自独立提交、单独 revert，不应与文档或门禁改动混在一起。

### 13.11 A1 落地记录（2026-11-14）

**§21 改判**：本节的**落点**（`crates/cutile/src/settings/`）已被 ADR 0023 取代——删掉配置文件层之后
剩下的三层策略整块搬到 `crates/cli/src/settings.rs`，`cutile` 只留 `cutile::dirs::Dirs`；下面关于
"A1 做了什么"的记录作为历史保留，其中"类型参数删除 / 值形状收敛为路径 / 错误类型统一 / 8 条用户可见
承诺不变"四条仍然成立。

**做了什么**：设置层搬进 `crates/cutile/src/settings/`（`mod.rs` 310 / `source.rs` 490 / `paths.rs` 208 / `key.rs` 137，共 1 145 行，含 23 个测试），`crates/config` 只剩 `src/lib.rs` 22 行 + `src/toml.rs` 583 行 = 605 行。

**删掉的四套抽象**（不是重命名，是入口归零）：

- `App` + `Cutile`：身份只剩一个实现，`Settings` 不再有类型参数；`DIR` / `FILE` / `ENV_PREFIX` 成为 `settings` 的三个私有常量。
- `Kind`：三个键全部是 `Kind::Path`，`Text` / `Int` / `Bool` / `Choice` 与它们的分支没有任何生产入口。
- `Value`：`PathBuf` 就是全部；`set` 的槽位类型从 `(Value, Origin)` 变成 `(PathBuf, Origin)`。
- `config::Error` + `BadKey` / `BadValue`：改用 `diagnostic::Error`，`from_io` 决定 `missing` / `io`，其余是 `invalid`；`tests/architecture.rs` 的错误边界例外分支随之删除。

**逐项实测（`wc -l` / `cargo test`，2026-11-14 工作树）**——本节数字取代 §2.1 与 §12.2 里对应的规模项，那两处是 A1 之前的快照：

| 指标 | A1 前 | A1 后 | 变化 |
| --- | ---: | ---: | --- |
| `crates/config/src` | 2 507 | 605 | −1 902 |
| `crates/cutile/src` | 9 367 | 10 509 | +1 142 |
| `crates/*/src` 合计 | 55 720 | 54 960 | **−760** |
| `crates/*/src` 文件数 | 171 | 168 | −3（删 7、增 4） |
| 用例（`--all-features`） | 1 083 | 1 063 | −20（全部是被删类型自己的测试） |
| 测试目标 | 62 | 62 | 0 |
| `[gate.debt]` 条目 | 274 | 271 | −3 |
| `[closed]` 豁免 | 4 | 3 | −1（`config` 不再手写闭集表） |
| 成员 / 依赖边 | 26 / 82 | 26 / 82 | 0（`cutile → llm-config` 仍需要，TOML 还在它那儿） |

用例减少的 20 条逐文件可核：`app` 5 + `value` 6 + `error` 3 随类型删除；`key` 6→5、`settings` 8→6、`source` 11→8（多应用身份、五种值形状、`Value::render` 那些分支不存在了）。

**8 条用户可见承诺的对拍**：四层优先级、`CUTILE_CACHE` 兼容名、未知键报错、`schema = 1` 校验、文件内相对路径按文件目录解析、环境变量与 `--set` 的相对路径按启动目录解析、`data.root` 三层覆盖、`[配置]` 段逐键"值 ← 来源"。`cargo run --bin cutile -- doctor --device cpu` 的配置段现在仍是：

```text
[配置] 默认 < 用户级文件 < 环境变量 < 命令行
  cache.dir  = /Users/fei/Library/Caches/cutile/cache ← 内置默认值
  work.dir   = /Users/fei/Library/Caches/cutile/work ← 内置默认值
  data.root  = -                            ← 未设置
```

格式串（`  {:<10} = {:<28} ← {}`）、`Origin` 的 `Display` 与默认值推导都逐字未变。

**两处刻意的行为变化**（都在错误路径，正文保留）：

1. 错误文本多了 `diagnostic::Error` 的类别前缀与 ` → ` 位置帧，消息正文逐字保留。实测（`./target/debug/cutile doctor --device cpu`）：

   ```text
   invalid: `/tmp/cfake/Library/Application Support/cutile/cutile.toml` 出现未登记的键 `cache.direkt`；登记的是：`cache.dir`, `work.dir`, `data.root` → settings::file → 配置
   invalid: `--set` 里的键 `nope` 未登记（登记的是：cache.dir、work.dir、data.root） → settings::cli_override → --set
   ```

2. 配置文件读不出来以前一律被折成 `invalid`(2)，现在 `missing`→4、`io`→5。原先 `src/main.rs` 与 `src/bin/cutile.rs` 各自把配置错误包成 `Error::new(Code::Invalid, format!("配置错误：{error}"))`，既丢掉类别又叠加一次前缀；两处 `map_err` 现在只补一帧上下文（`配置` / `--set`）。实测 `chmod 000 cutile.toml` 后是 `io: 读取 … 失败：Permission denied (os error 13) → settings::file → 配置`，退出码 **5**；未知键与坏 `schema` 仍是 **2**。

**门禁同步**：`[gate.debt]` 删 8 条指向已删文件的条目（`error.rs:fmt`、`source.rs` ×4、`value.rs` ×3），新增 5 条 `crates/cutile/src/settings/source.rs:{file,cli_override,collect_leaves,scalar,type_name}`，ccn 与合并前逐项相同（8 / 6 / 6 / 6 / 6）——搬家不降复杂度，降的是类型与入口数。`[[ownership.unlisted]]` 的 `config` 只剩 `ParseError` / `Value`；`[[closed.exempt]]` 删掉 `config`。

**验证**：`cargo test --workspace --all-features` 62 目标 / **1 063 通过 / 0 失败 / 1 忽略**；`--test architecture` 8/8、`--test contracts` 6/6、`-p llm-tool --test gate` 1/1；`cargo clippy --workspace --all-targets --all-features -- -D warnings` 与 `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` 均干净；`cargo run --bin cutile -- doctor --device cpu` 退出码 0。

**未做**：

- A2 当时的写法是「`crates/config` → `crates/toml`，包名 `llm-toml`，8 条债务键改路径、6 处 `use` 改写」。**这个写法已被否决**：改名不改成员数、不改边数、不改"谁真的需要它"，实测见 §13.12。
- 根包 `Cargo.toml` 的 `config` 依赖当时估计只被根测试用到；实际根测试并不 `use llm_config`（它们用 `std::fs` 读文档与源码），所以连 `+dev` 都不必留——A2 直接删掉了这一条（见 §13.12）。

### 13.12 A2 落地记录：解耦删除（不是改名）

**裁决**：`llm-config` 剩下的 585 行是受限 TOML 解析器。§13.6 给的 A2 是"改名 `llm-toml`"，但改名与解耦是两件事——成员数不变、依赖边一条不少、六个 `use` 只换字符串。用户裁决按解耦执行：**语法层并进它已经必须依赖的 L1 成员，成员随之删除**。

**依赖方向的算账**（这是"零新增边"的根据）：

| 消费者 | 原来指向语法层的边 | 它是否已经依赖 `diagnostic` |
| --- | --- | --- |
| `op`（`ops/*.toml`） | `op → llm-config` | 是（P0-7 起，错误类型） |
| `prof`（`devices/*.toml`） | `prof → llm-config` | 是（同上） |
| `tool`（`lints/vocabulary.toml` 阈值） | `tool → llm-config` | 是 |
| `cutile`（经设置层） | `cutile → llm-config` | 是 |
| 根包 `llm`（契约测试的驱动方） | `llm → llm-config` | 是 |
| `fuzz`（`tests/parsers.rs` 的驱动对象） | `fuzz → +dev: llm-config` | 是 |
| 合计 | 6 条（5 条普通 + 1 条 `+dev`） | 6/6 已经依赖 |

于是这条边是纯冗余：删掉它、把同样六个 `use` 指向 `diagnostic::toml`，**成员 26 → 25、依赖边 82 → 76、新增零条**。

**做了什么**（逐项可复核）：

- `crates/config/src/toml.rs`（585 行）→ `crates/diagnostic/src/toml.rs`，`diagnostic/src/lib.rs` 加一行 `pub mod toml;`；`crates/config/`（目录、`Cargo.toml`、`README.md`）删除。
- 六个 `use` 改写：`crates/{op,prof,tool}/src/*.rs`、`crates/cutile/src/settings/source.rs`、`src/*.rs` 与根 `tests/*.rs`、`crates/fuzz/tests/parsers.rs`；`Cargo.toml` 的依赖项同步（`llm-config` 从 5 个 `[dependencies]`（根包、`cutile`、`op`、`prof`、`tool`）与 1 个 `[dev-dependencies]`（`fuzz`）里移除）。
- `prof` 的 `pub use llm_config::toml` 再导出删除。**它一直在被借用**：`prof/src/prof.rs` 有 17 处 `crate::toml`，说明"`op` / `tool` / 根测试通过 `prof` 拿解析器"不是零成本转发，而是真的多一层；现在这 17 处改为 `diagnostic::toml`。
- 不在 `diagnostic::Error` 上再包一层：`ParseError`（行号 + 说明）保留，类别由调用方按语义给（`invalid` / `missing`），语法层不该知道一次语法失败算哪一类。

**逐项实测**（`wc -l` / `cargo test`，2026-11-14 工作树；本节数字取代 §13.11 表里与 `config`/`diagnostic` 有关的行）：

| 指标 | A1 后 | A2 后 | 变化 |
| --- | ---: | ---: | --- |
| `crates/config/src` | 605 | 0 | −605（成员删除） |
| `crates/diagnostic/src` | 260 | 856 | +596（`toml.rs` 585 + `lib.rs` 文档 11） |
| `crates/*/src` 合计 | 54 960 | **54 959** | −1（见下方说明） |
| `crates/*/src` 文件数 | 168 | **167** | −1（`config` 的两个文件换成 `diagnostic` 的一个） |
| `crates/` 下的成员目录 | 25 | **24** | −1 |
| 成员（含根包） | 26 | **25** | −1 |
| 依赖边（含 `+dev` 13） | 82 | **76** | −6，新增 0 |
| 用例（`--all-features`） | 1 063 通过 / 62 目标 | **1 062 通过 / 60 目标** | −1 条，−2 目标 |
| `[gate.debt]` 条目 | 271 | **271** | 0（8 条键改路径，数值不变） |
| `[[closed.exempt]]` / `[[ownership.unlisted]]` | 3 / 15 | **3 / 15** | 0（`config` 条目改为 `diagnostic`） |

两处需要解释的差：

1. **`crates/*/src` 只减 1 行**：这条 A2 买的是成员数与边数，不是行数。−605（`config` 消失）+596（`toml.rs` 585 行原样搬家 + `lib.rs` 的 11 行文档）= −9，再算上"同一文件在两个成员里的路径长度差"与文档同步的零散改动，净 −1。这与 ADR 0019 的结论一致：包装/搬家合并**不自动**带来减行。
2. **用例少 1 条、目标少 2 个**：少的是 `llm-config` 自己的 doctest 目标（它引用 `config::toml::parse`，在 A1 里已改成 `llm_config::toml::parse`，随成员一起消失）。`toml.rs` 内 **13** 个测试随文件迁入 `diagnostic` 并继续执行（`cargo test -p llm-diagnostic` 可见），`prof` / `op` / `tool` / `fuzz` 的解析用例一条没动。

**门禁同步**（三处，都是"路径变了、判据没松"）：

- `lints/vocabulary.toml`：8 条 `[gate.debt]` 键从 `crates/config/src/toml.rs:*` 改为 `crates/diagnostic/src/toml.rs:*`，`ccn` / `nloc` 逐条不变（`parse` 19 / `parse_value` 22 / `split_top_level` 12 / `descend` 8 / `push_table_array` 8 / `strip_comment` 8 / `unescape` 8 / `parse_number` 6）；`[[ownership.unlisted]]` 的 `config` 条目改为 `diagnostic`。
- `tests/architecture.rs`：删掉 `member == "config"` 这个例外分支（成员不存在了），改为**文件级**豁免 `crates/diagnostic/src/toml.rs`（常量 `MESSAGE_FRAGMENTS`）。理由写进测试文档：`parse_value` / `parse_number` / `Table::insert` 的 `String` 是"为什么不合法"的消息片段，对外一直是 `ParseError`，从来不是任何一层的失败契约。**这是把旧例外收窄，不是取消**：`diagnostic` 的其余文件仍被扫描。
- `docs/ARCHITECTURE.md`：层表的 L1 行少一个成员；边清单删 6 条、表头 `82 → 76`（`+dev` 13 → 12）；吸收链新增第八条（拆分后删除）；所有权表 `Error / Code / Result` 一行补 `toml` 的归属说明，`ParseError` / `Value` 仍登记在 `[[ownership.unlisted]]`（`Table` 与 `engine` 的 KV 页表同名，按名字判定）。

**8 条用户可见承诺重新对拍**（与 §13.11 同一份清单，逐条复跑）：四层优先级、`CUTILE_CACHE` 兼容名、未知键报错、`schema = 1` 校验、文件内相对路径按文件目录解析、环境变量与 `--set` 按启动目录解析、`data.root` 三层覆盖、`[配置]` 段逐键"值 ← 来源"。`cargo run --bin cutile -- doctor --device cpu` 的输出与错误文本逐字不变——A2 只搬了"谁解析 TOML"，没有碰"值从哪来"。

**验证**：`cargo test --workspace --all-features` 60 目标 / **1 062 通过 / 0 失败 / 1 忽略**；`cargo test --test architecture` 8/8、`--test contracts` 6/6、`-p llm-tool --test gate` 1/1；`cargo clippy --workspace --all-targets --all-features -- -D warnings`、`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`、`cargo fmt --all -- --check` 均干净。

**这次没有做的事**（都是独立裁决，不随 A2 顺带）：

- 不把 `ParseError` 并进 `diagnostic::Error`：那会让语法层替调用方决定类别，正是 §13.4 反对的"错误契约兼数据语法"。
- 不动 `diagnostic` 的 crate 名字：它现在拥有两个原语（错误类型 + TOML），改名属另一件事，且会牵动 20 个成员的 `use`。
- 不处理 §10.4 的 3 434 行零引用模块与 P0 的四处不可信输入（`rans.rs` / `bin.rs` / `snapshot` / `envelope.rs`），它们与本次删除无关。

## 十四、`diagnostic` 深度审计与「运行期零 TOML 解析」落地（ADR 0022）

> 本节回答两件事：① `crates/diagnostic` 这个 crate 今天该不该存在、该拥有什么；② 手写 TOML 解析器
> （`crates/diagnostic/src/toml.rs`，585 行）为什么可以删、删完哪些行为真的变了。所有数字为
> `wc -l` / `cargo test` / `cargo tree` 实测（2026-11-14 工作树），行号与文件名按落地后的现状给出。

### 14.1 结论

`diagnostic` 应当存在，而且应当**只**拥有两样东西：全仓唯一的错误类型（`Error` / `Code` / `Result`）
与全仓唯一的声明数据树（`data::Table` / `data::Value`）。前者决定退出码契约，后者是构建期语料的运行期形状。
它不该拥有的第三样东西是**解析器**——这正是本轮删掉的部分。

### 14.2 为什么删得掉：判据与反例检查

判据只有一条：**同一份数据在运行期是否还需要"从文本变回结构"。**

- 事实 A（已确认）：三份语料都是编译期资产。`crates/op/ops/*.toml`、`crates/prof/devices/*.toml`、
  `lints/vocabulary.toml` 随二进制发布，运行期读到的是与构建期同一份文件；`ops/`、`devices/` 的
  运行期装载入口（`op::load`、`prof::load`）唯一的生产使用者是 `cutile::builtin` 的 `from_dir`/
  `from_repo` 与一份测试。
- 事实 B（已确认）：用户真正可写的输入只有运行时配置，键闭集是三个（`cache.dir` / `work.dir` /
  `data.root`），值形状只有路径；同一组值可由 `CUTILE_*` 与 `--set` 完整表达。
- 事实 C（已确认）：六个消费者的调用形状都是"拿一份**已解析**的东西"——`OpDecl::parse` 吃声明、
  `Profile::parse` 吃档案、`Gate` 吃阈值。没有一处需要"解析任意用户文本"。

反例检查（哪种情形会推翻结论）：① 语料必须由用户替换（`data.root` 覆盖 `devices/`）→ 那需要一个**显式**
装载入口，不是常驻解析器；② 配置值形状扩展到数组/表（`[[...]]`）→ 三个键之外才会需要文法，届时再加；
③ 出现第二种语料格式 → 多一个 `build.rs`，数据树不变。三条都写进了 ADR 0022 的"重议条件"。

### 14.3 落地后的形状

| crate | 输入（构建期） | 生成物 | 运行期暴露 | 失败行为 |
| --- | --- | --- | --- | --- |
| `op` | `crates/op/ops/*.toml` | `$OUT_DIR/declared.rs` | `op::declared(&str) -> Option<&'static Table>`、`op::names()` | 日期值/非表顶层 → 构建失败 |
| `prof` | `crates/prof/devices/*.toml` | `$OUT_DIR/embedded.rs` | `prof::embedded(&str) -> Option<&'static Table>`、`prof::names()` | 同上 |
| `tool` | `lints/vocabulary.toml` | `$OUT_DIR/vocabulary.rs` | `tool::vocabulary() -> &'static Table` | 同上 |

`toml` 只出现在三个 `[build-dependencies]` 里；`cargo tree -e normal --workspace` 中 **没有** `toml` /
`serde`（运行时第三方依赖 0 未被破坏），`Cargo.lock` 因此是 109 个包（多数为构建期）。

### 14.4 逐消费者对照

| 消费者 | 原来 | 现在 | 行为是否变化 |
| --- | --- | --- | --- |
| `op` | `op::load(dir)` 读 `ops/` + `OpDecl::parse(text, tmpl)` | `op::builtin()` + `OpDecl::parse(&Table, tmpl)` | 不变（同一份 `ops/`，同一套闭集校验） |
| `prof` | `prof::load(dir)` / `Archive::text()` / `Profile::parse(text)` | `prof::embedded(name)` + `Profile::parse(&Table)` | 不变；`ProfileError::Syntax` 消失（没有运行期语法错误） |
| `tool` | `include_str!` + 运行期 `parse` | `tool::vocabulary()` | 不变（同一份阈值与债务表） |
| 根测试（`architecture` / `contracts`） | 各自 `include_str!` + `parse` | 读 `tool::vocabulary()` | 不变；文件只解析一次且发生在构建期 |
| `cutile` | `Catalogue::{from_dir,from_repo,from_layers}` + 档案走 `data.root/devices` | `Catalogue::builtin()` | **变**：`data.root` 不再能替换档案层（只作见证根的定位） |
| `cutile::settings` | 四层：默认 < 文件 < 环境变量 < `--set` | 三层：默认 < 环境变量 < `--set`（**§21 改判**：这三层又整块上移到 `cli::settings`，`cutile` 只留 `dirs::Dirs`） | **变**：见 14.5 |
| `fuzz` | 5 个目标（含 TOML 与档案） | 3 个目标（`cli`、`weights` JSON、safetensors） | **变**：TOML 无入口可灌 |

### 14.5 真正消失的行为（逐条可复核）

| 行为 | 原位置 | 影响面 | 替代 |
| --- | --- | --- | --- |
| 读运行时配置文件 | `cutile/src/settings/mod.rs` 的 `FILE` 常量与 `source::file()` | 用户若写了 `cutile.toml`，现在**不再被读** | `CUTILE_*` 或 `--set` |
| `schema = 1` 校验 | `settings::SCHEMA_VER` + `source::check_schema` | 没有文件就没有该字段 | 无（见 ADR 0022 重议条件 ②） |
| 文件内相对路径基准 | `source::file()` 的 rebase | 只剩 `Roots::base()`（启动目录）一个基准 | `--set` 直接给绝对路径 |
| 未知键的文件侧校验 | `source::check_unknown_keys` | 只有文件能表达"未登记的键" | `--set` 仍拒绝未登记键（`invalid` → 2） |
| 设置层的 `missing` / `io` 分支 | `settings::load_with` | 设置层不再读盘，只可能 `invalid` | 其它路径的 4 / 5 不受影响 |
| `op::load` / `op::dir` / `BUILTIN` | `crates/op/src/lib.rs` | 运行期无法从任意目录装载算子 | 构建期生成表 |
| `prof::load` / `prof::dir` / `Archive` / `builtin()` | `crates/prof/src/lib.rs` | 同上（档案） | `prof::embedded` |
| `ProfileError::Syntax` | `crates/prof/src/prof.rs` | 失败类别少一种 | 结构与语义错误仍在（`invalid` / `missing`） |
| `fuzz` 的 `Toml` / `Profile` 目标 | `crates/fuzz/tests/parsers.rs` | TOML 不再进模糊测试 | 构建期解析由 `toml` crate 负责 |
| `fuzz → +dev: prof` 依赖边 | `crates/fuzz/Cargo.toml` | 死边 | 删除（边 76 → 75） |

> **§21 改判（位置列）**：上两表里所有 `cutile/src/settings/**` 与 `settings::*` 的路径已被
> ADR 0023 取代——三层策略整块搬到 `crates/cli/src/settings.rs`，`cutile` 只剩
> `crates/cutile/src/dirs.rs` 的 `Dirs`（三个路径的纯数据）。**"哪些行为消失"这一列不变**：
> 配置文件、`schema`、文件相对基准、文件侧未知键校验、设置层的 `missing`/`io` 仍然全部消失，
> 只是"消失在哪一层"从 `cutile` 变成了 `cli`。

**没有变**的部分（同一条命令逐字对拍）：三个配置键及其语义、`CUTILE_CACHE` 兼容名、
`CUTILE_CACHE_DIR > CUTILE_CACHE` 的先后、空值拒绝、`invalid` → 2 的退出码、`[配置]` 段的
"值 ← 来源"格式、`cutile doctor --device cpu` 的协商/计划/emit/cache key 输出。

### 14.6 实测对照

| 指标 | A2 之后（§13.12） | 本轮之后 | 变化 |
| --- | ---: | ---: | --- |
| 成员（含根包） | 25 | **25** | 0（解析器原本就在 `diagnostic`） |
| 依赖边（含 `+dev`） | 76（12） | **75（11）** | −1（`fuzz → +dev: prof`） |
| `crates/*/src` 行数 | 54 959 | **54 519** | −440（`toml.rs` 585 行 − 生成/数据树新增） |
| `crates/*/src` 文件数 | 167 | **168** | +1（`data.rs` 进来、`toml.rs` 出去、`prof/tests.rs` 拆出） |
| `diagnostic` 行数 | 856 | **712** | −144（`lib.rs` 文档重写 + 585 行解析器离开、`data.rs` 留下） |
| `[[gate.debt]]` 条目 | 271 | **255** | −16（8 解析器 + 4 设置层 + 2 `prof` + 1 `tool` + 1 `op`） |
| 用例（`--all-features`） | 1 062 通过 / 62 目标 | **1 052 通过 / 60 目标** | 净 −10；删掉的都是只覆盖已消失入口的用例 |
| 运行时第三方依赖 | 0 | **0** | 未变（`toml` 只在构建期） |

三处需要说明的差：① 目标数 62 → 60 是 `fuzz` 的两条目标与 `llm-config` 时代的 doctest 目标叠加后的净效应
（目标数与用例数不同步，逐项见各 crate 的测试输出）；② `crates/*/src` 的 −440 不是"删了 585 行"这么简单——
解析器 585 行离开，同时 `data.rs` 的数据树、构造与路径改写约 450 行进来，`op`/`prof`/`tool` 三个 `build.rs`
376 行不计入 `src`；③ 债务 −16 条全部是**删除**，没有任何限额被抬高，`Gate::stale` 因此保持绿。

### 14.7 这一轮没做的事

- `weights` 的 10 个零引用模块（3 434 行）、`storage` 布局并入 `tensor`、`exec` 的死 SPI、
  `snapshot` 1 611 行拆文件、`Facts` 补两个事实、字节级黄金向量——与"删除解析器"无关，仍在 §11.4 排队。
- 四处 P0（`weights/src/rans.rs` 的按字段分配、`bin.rs` 的奇数 `SETTINGS`、`snapshot` 的 `expect`、
  `envelope.rs` 的零长度校验）在"删除解析器"那一轮**一条未修**——它们与语料解析无关；四条随后在
  **下一轮**全部修完，落地记录见 §15。

### 14.8 复核命令

```bash
cargo test --workspace --all-features            # 60 目标 / 1 052 通过 / 0 失败 / 1 忽略
cargo test -p llm-tool --test gate               # 静态门：ccn ≤ 5、nloc ≤ 60、零 panic、棘轮
cargo test --test architecture --test contracts  # 8/8、6/6（边集 75/11、所有权表、错误边界）
cargo tree -e normal --workspace | grep -E 'toml|serde'   # 空：运行期没有第三方解析器
cargo run --bin cutile -- doctor --device cpu    # 退出码 0，`[配置]` 只剩三层
```

## 十五、P0 五连修落地记录（本轮）

> 本节是 §五 P0 组的落地记录，也是 §12.6「仍在」那一列的取代者。行号一律按落地后的现状给出；
> 反向验证指「把修复删掉，测试必须报红」——这是本轮唯一用来证明"测试真的踩在崩溃点上"的手段。

### 15.1 结论

§五 列的四条 P0 **全部修复**；复核第四条时在它的同一路径上发现**第五处**未被登记的同族缺陷
（`dsa::hint_unpack_dyn` 的提示位计数可与段长自相矛盾，原写法 `y[index..total]` 成为 `start > end`
切片）。五处各有回归测试，其中四处做了反向验证并留证。

修复的原则只有一条：**长度必须由调用方的契约给出，不能由被解析的字节自报**。`rans` 的上限来自段描述符
（`dtype × numel`），`dsa` 的长度来自参数集（`pk_len()` / `sig_len_min()`），`snapshot` 的标签来自闭集
`tag()`，`bin` 的奇偶来自 pickle 文法本身——没有一处是"再读一个字段然后相信它"。

### 15.2 逐条

| 条目 | 修法（落地后位置） | 回归测试 | 反向验证 |
| --- | --- | --- | --- |
| P0-1 `rANS` 按字段分配 | `decode(stored, cap)` 先调 `declared_len`（`weights/src/rans.rs:380`）：帧头长度 > `cap` 即 `invalid`（`:382-384`）；`cap` 由调用方给——段路径用 `seg.logical_len()`（`reader.rs:84-86`，`dtype × numel`），轴路径用 `payload_cap`（`axes.rs:173`，每元素至多 8 字节 + 16 字节头） | `rans.rs:470 a_lying_length_is_rejected_before_allocating` | 删掉长度比较后该测试失败 |
| P0-2 奇数 `SETITEMS` 越界 | `pop_pairs`（`weights/src/bin.rs:448`）先取计数再判奇偶（`:450-452`），奇数返回 `invalid`；`chunks(2)` 只在偶数下执行 | `bin.rs:582 an_odd_setitems_is_an_error_not_a_panic` | 断言要求"报错而非 panic"，修复前 `chunk[1]` 越界 |
| P0-3 `snapshot` 编码路径 `expect` | 标签表一次判完：`tags()`（`train/src/snapshot/mod.rs:346`）对全部条目调用 `tag`；`encoded_len`/`encode`/`encode_into` 都经 `tags()?`，写字节阶段不再可能失败；`tag`（`:924`）仍是 `Result` | `train/src/snapshot/tests.rs:22 an_unsupported_dtype_is_an_error_not_a_panic` | 该测试构造 `I64` 权重头，修复前 panic 于 `expect`，现在断言 `Code::Unsupported` |
| P0-4 `dsa` 解码零长度校验 | `pk_decode`（`weights/src/dsa.rs:378`）要求 `pk.len() == pk_len()`；`sig_decode`（`:415`）要求 `sig.len() >= sig_len_min()`（`:359` 给出"`c̃` + `l` 个 `z` + `k` 个计数"的下界）；`open`（`:395`）先校验公钥长度再解码 | `weights/src/envelope.rs:161 a_truncated_signature_is_rejected_not_sliced` | 1 字节侧车：修复前 panic（`sig[..c_len]`），现在 `Err` |
| P0-4b 提示位计数与段长矛盾（本轮新发现） | `hint_unpack_dyn`（`weights/src/dsa.rs:715`）末尾改为 `y.get(index..total).is_none_or(..)`：越界给 `None`、尾部含非零字节给 `None`，同一表达式覆盖两种坏输入 | `dsa.rs:766 a_hint_count_beyond_the_segment_is_rejected` | 换回原写法 `y[index..total]`，测试报 `slice index starts at 4 but ends at 0` |

P0-4b 的完整写法是 `y.get(index..total).is_none_or(|rest| rest.iter().any(|&b| b != 0))`——`get` 负责
越界（`None`），闭包负责"尾部必须全 0"，两者都是 `None` 时走同一个 `return None`，因此只占一个分支。

成因值得单记一句：`hint_unpack_dyn` 的提示段长度是 `total + k`，`total` 由 `y.len() - k` 反推，
而 `k` 个 stop 计数可以一路累加到超过 `total`。**"段长由减出来"这件事本身就是风险源**——同一形态在
`gguf.rs:199`（`bytes.len() - pos`）与 `gguf.rs:358`（`start + len`）仍有，见 §15.4。

### 15.3 证据等级

- **已确认（实测）**：五条测试在修复后通过（`cargo test -p llm-weights` 170 + 10 + 5 全绿，
  `cargo test -p llm-train` 全绿）；P0-4b 的反向验证留下 `slice index starts at 4 but ends at 0` 的
  panic 原文（`dsa.rs`），证明该测试踩在坏切片上，而不是只断言返回值形状。
- **已确认（门禁）**：`hint_unpack_dyn` 的修法两次把 ccn 从 8 推到 9（先加 `if`、再改用 `?`，两者各算一个
  分支），两次都被 `cargo test -p llm-tool --test gate` 拦回；最终用 `get(..).is_none_or(..)` 落回 8，
  **`[[gate.debt]]` 未抬高**。这是"债务只准删、不准涨"第一次在真实改动上生效。
- **推断（未实测）**：非 macOS 目标没有编译过；五处修复都不涉及数值路径，未做真机对拍。
- **未知**：`Envelope::decode`（`envelope.rs:86-107`）是否还需在解码期就拒绝与参数集不符的 `set`
  （现只校验 `44/65/87` 三个字面量）；这属于"提前一层报错"的取舍，未改。

### 15.4 残余风险（不是"路径已安全"）

- **P0-1 的上限仍来自文件里的类型与形状**：`seg.logical_len()` = `dtype.bytes_for(numel(shape))`，
  而 `numel` 用 saturating（`seg.rs:197`）。门槛从"4 字节任意值"升到"必须自洽的极端形状"，但没有全局封顶。
  真正的上界应是策略（调用方预算或"物理长度 × 压缩比上限"），属设计决策，本轮未做。
- **同族"减出来 / 加出来"仍未复核**（**§17 已结清**）：`gguf.rs:199`（`len - pos`）、
  `gguf.rs:358`（`start + len`）、`zip.rs:128-142`（`zip64_extra` 校验了 `at + 4 <= stop` 却按 `size`
  之外的下标读）、`axes.rs:89`（公开 `search` 上的 `assert!`）。四条在本轮（§15）范围外，翌轮按同一判据
  全部改成"越界即错误"，另加第五条同类违例 `zip::slice` 的 `at + len`；行号已随修复漂移，见 §17.2。
- **`Envelope::check` 的 fail-closed 承诺现在对"截断"成立**；对"长度合法但内容伪造"仍靠签名本身——
  这是密码学保证，不是长度校验能替代的，记为事实而非缺陷。
- **五处修复都在"解码/验证"侧**：编码侧（`writer`、`axes::pack`）仍假设调用方给的形状自洽，
  它们不读文件，故未新增校验。

### 15.5 实测对照

| 指标 | 上一轮（§14.6 快照） | 本轮 | 说明 |
| --- | ---: | ---: | --- |
| `crates/*/src` 行数 | 54 519 | **54 717** | +198：五条回归测试与五处校验；没有任何限额被抬高 |
| `crates/*/src` 文件数 | 168 | **169** | 本轮只改不增删文件；与 §14.6 的 168 差 1，口径差异未核，按 `find` 现值记 |
| `[[gate.debt]]` 条目 | 255 | **254** | −1（`hint_unpack_dyn` 的修法让一处登记项落到限额内并删除） |
| 工作区成员 | 25 | **25** | 不变 |
| 依赖边 | 75（`+dev` 11） | **75（11）** | 不变（本轮不涉及依赖） |
| 用例 | 1 052 通过 / 60 目标 | **1 057 通过 / 60 目标** | +5：五条回归测试 |
| `cutile doctor --device cpu` | 退出码 0 | **退出码 0** | 配置段与协商输出未受影响 |

### 15.6 本轮没碰的东西

- §11.4 队列原样保留：`weights` 的零引用模块、`storage` 布局并入 `tensor`、`exec` 的死 SPI、
  `snapshot` 拆文件、`Facts` 补两个事实、字节级黄金向量。
- §15.4 的四条同族缺陷（`gguf` 两处、`zip64_extra`、`axes::search` 的 `assert!`）。
- 未新增 ADR：五处修复是既有契约（"长度来自契约"）的补齐，不改变任何架构决策。

### 15.7 复核命令

```bash
cargo test --workspace --all-features             # 60 目标 / 1 057 通过 / 0 失败 / 1 忽略
cargo test -p llm-weights dsa                     # 15 通过（含 §15.2 的 P0-4b）
cargo test -p llm-weights a_hint_count_beyond     # 单条：提示位计数越界必须判无效
cargo test -p llm-tool --test gate                # 静态门：ccn≤5、nloc≤60、零 panic、棘轮
cargo run --bin cutile -- doctor --device cpu     # 退出码 0
```

## 十六、第一梯队收尾：平台门禁、文档漂移与死代码（本轮）

> §8.2「第一梯队」的四项在本轮全部结清。其中两项**不是本轮做的**：`pack::*` 引用与
> `ccn.rs` 的死递归在更早的提交里已经消失，本轮只做复核并留下命令。真正落地的是三件：
> `mmap` 的 Unix 门禁（P1-2）、`seg.rs` 的空循环（P2-8）、两处文档漂移（P1-8）。

### 16.1 结论

第一条值得强调的是：**P1-2 从「推断」变成了「已确认并已修」**。首轮写它时无法在 Windows
目标上实测（只装了 macOS 的 std），本轮用 `rustup` 的 Windows std 做了 `cargo check`，
两处错误的原文都拿到了；修完之后同一个命令全仓退出码 0——这也是全仓第一次在 Windows 目标上
整体通过类型检查。

第二条是**收益的类型**：本轮净增 10 行，用例数、债务条数、成员数、依赖边一条未变。
它买的是"Windows CI 不再必红""两处不再指向已删除模块或已过期事实的文档"，不是行数。

### 16.2 P1-2 落地：`mmap` 只在 Unix 编译

**复现（已确认）**：`cargo check -p llm-storage --features mmap --target x86_64-pc-windows-msvc`
报两错——`mmap.rs:14` `unresolved import std::os::unix`、`mmap.rs:71` `no method named as_raw_fd`。
与首轮推断完全一致，只是这次有原文。

**修法**（三处，均只加条件、不改逻辑）：

| 位置 | 改动 | 为什么不是别的写法 |
| --- | --- | --- |
| `crates/storage/src/lib.rs:29-32` | `#[cfg(feature = "mmap")]` → `#[cfg(all(feature = "mmap", unix))]`，模块与再导出同改 | 在 `mmap.rs` 内部逐行加 `cfg` 只会把同一个条件抄 4 遍；条件属于模块边界 |
| `crates/storage/Cargo.toml:14-16`、`mmap.rs:3-6`、`lib.rs:10-12` | 各补一句"只在 Unix 生效" | feature 名不变（`--all-features` 仍能开它），但读者必须知道它不在所有平台有内容 |
| `crates/weights/tests/round_trip.rs:160-164` | 唯一消费者加 `#[cfg(unix)]` | **没有**改成"非 Unix 上直接 `return`"——那会变成 §P2-9 那种可空转的绿；现在它在非 Unix 上根本不存在 |

**拒绝的替代方案**：在非 Unix 上把整文件读进 `Vec<u8>` 冒充映射。这会让"打开 10 GB 权重不占
10 GB 内存"这个前提在 Windows 上静默失真，而调用方看不出区别——属于把可移植性换成了正确性。

**验证**（表 16-1）：修复前后的同一条命令。

| 检查 | 修前 | 修后 |
| --- | --- | --- |
| `cargo check -p llm-storage --features mmap --target x86_64-pc-windows-msvc` | 2 error | 退出码 0 |
| `cargo check -p llm-weights --all-targets --target x86_64-pc-windows-msvc` | 未测 | 退出码 0 |
| `cargo check --workspace --exclude llm-benches --all-targets --all-features --target x86_64-pc-windows-msvc` | 未测 | **退出码 0** |

表 16-1 最后一行是本节最有价值的证据：它说明 `mmap` 是**唯一**的平台门禁缺口，没有第二处。
（首轮 P1-2 的措辞是"推断"，现在可以把它降级为"已确认的单一缺口，且已关闭"。）

**顺带修掉一处本轮自己引入的断链**：crate 头最初把 `Mmap` 写成了文档链接，`cargo doc --workspace
--no-deps`（CI 口径、默认 feature）立刻报 `unresolved link to Mmap` ——`mmap` 关着时这个类型不存在，
链接自然悬空。改成代码块即可，规则与 §11.7.1 那次"公开文档不要链私有项"是同一条：**给条件编译的东西
加文档链接，在默认 feature 下就是断链**。两个口径（默认 feature 与 `--all-features`）现在都是 0 告警。

**边界（未实测，必须写清）**：

- `llm-benches` 被排除。它的 dev 依赖 `criterion → alloca` 的构建脚本需要目标平台的 C 编译器，
  本机没有 MSVC 交叉工具链（`build_script_build` 退出码 1）。真实 `windows-latest` runner 上有
  MSVC，故 CI 不受影响；但**本机无法验证它**，记录为未知。
- 只做了**类型检查**，没有在 Windows 上**运行**测试。`--all-targets` 覆盖了测试代码的编译，
  不覆盖它们的执行（例如 `round_trip` 里依赖临时目录与文件权限的那些断言）。

### 16.3 P1-8 落地：两处文档漂移，其中一处是整节

| 文件:行 | 原文 | 实况 | 改法 |
| --- | --- | --- | --- |
| `docs/ARCHITECTURE.md:99` | "现有两条算子 `gemv_i4` 与 `gemv_f32`" | `op/src/lib.rs:111` 是 `[Registration; 22]` | 改为 22 条注册，并保留"GEMV 三条共用一份泛型骨架"这个仍然为真的部分 |
| `crates/op/README.md:108` | 整节标题"现有的两条算子"+ 2 行表 | 同上，差 11 倍 | 改为按族（GEMV / GEMM / 逐元素 / 归约 / 形态 / 注意力 / 解量化）的 7 行表，节首写明"清单不在这里维护，`REGISTRY` 是唯一事实源" |
| `crates/op/README.md:147-151` | 依赖段写 `llm → cutile → op → einsum`、`op` 依赖 `einsum` 与 `diagnostic` 的 `toml` 子模块 | `crates/op/Cargo.toml` 的依赖是 `annotation`/`diagnostic`/`prof`；`einsum` 已并入 `op`（§11），`toml` 子模块随 ADR 0022 删除 | 按 `Cargo.toml` 实况重写，并注明 `op::einsum` 的来源 |
| `crates/weights/src/lib.rs:4-10` | 拥有段只列契约层；依赖段只有 `tensor`/`diagnostic` | `Cargo.toml` 是 `dtype`/`diagnostic`/`mem`/`tensor`（`+dev` `storage`/`testkit`）；`lib.rs` 另有 33 个 `mod`/`pub mod` | 重写为"契约 + LLPK 容器 + 格式适配器与证明栈"三块，入边写 `train`/`nn`/`llm`（`+dev` `fuzz`） |

**为什么 README 那节要改成"按族"而不是补成 22 行清单**：补成 22 行会让 README 成为与
`REGISTRY` 并列的第二份算子清单，每加一个算子要改两处，而两处能否一致**没有任何门禁**
（`tool/tests/gate.rs:154-166` 的文档检查只扫 `docs/` 平铺的 `.md`，`crates/*/README.md`
与 `docs/adr/**` 都不在扫描面，见 P1-7）。按族写的表是稳定的：新增一个 GEMV 格式不会让表失效。
这与 §10.3 判 `nn` 是"算子目录的第三份投影"是同一类判断——**清单类事实只该有一处，其余引用它**。

### 16.4 死代码：一处本轮删，两处早已不在

| 项 | 位置 | 结论 |
| --- | --- | --- |
| `validate` 里的空循环 | `crates/weights/src/seg.rs:65-68`（修前） | **本轮删除**；"轴长允许 0、`u64` 非负、越界形状由 `numel` 饱和乘与字节数交叉校验兜住"移进 `validate` 的文档注释 |
| `ccn::definition` 的死递归 | `crates/tool/src/ccn.rs:138-142`（首轮） | **复核时已不在**：现为 `Token::Group { .. } => {}` 并附"签名组内递归不会产生定义"的注释；本条从 P2-8 划掉 |
| 13 处 `pack::*` 诊断锚点 | `affine.rs`/`rans.rs`/`leb.rs`/`codec.rs`/`seg.rs`/`writer.rs`/`reader.rs`（首轮） | **已不在**：`grep -rn "pack::" crates/*/src src` 只剩 `dsa.rs:853-854` 的 `hint_unpack::<6>(.., &packed)`，是变量名误命中 |
| `mmap.rs:7` 的 `pack::Archive` 文档引用 | `crates/storage/src/mmap.rs:10`（现） | 现写 `weights::Archive::parse`，已正确 |

第三行值得单记：**这条 P1 是被"更早的某次提交"顺手修掉的**，而报告没有回头改状态。
这不是报告的错读，而是"在途改动"与"审计快照"的固有错位——本节的复核命令就是为下次复核准备的。

### 16.5 实测对照

分两列写，因为这轮复测时撞上一次并发改动（§16.6）：**「本轮自身改动」与「复核时全仓」不是同一棵树**。

| 指标 | 上一轮（§15.5） | 本轮自身改动 | 复核时全仓（09:25） |
| --- | ---: | ---: | ---: |
| `crates/*/src` 行数 | 54 717 | **54 727**（+10） | 54 980（+253，差额来自 §16.6） |
| `crates/*/src` 文件数 | 169 | **169** | 172（+3：`code.rs`/`error.rs`/`buildgen.rs`） |
| `[[gate.debt]]` 条目 | 254 | **254**（不变） | 253（−1，由并发改动） |
| 工作区成员 | 25 | **25** | 25 |
| 依赖边 | 75（`+dev` 11） | **75（11）** | 75（11） |
| 用例 | 1 057 通过 / 60 目标 | **1 057 通过 / 60 目标**（这条 `round_trip` 在 macOS 上仍编译、仍执行） | 1 062 通过 / 60 目标（+5 来自 `buildgen` 自带测试） |
| Windows 目标类型检查 | 未测（P1-2 为推断） | **退出码 0** | 退出码 0 |
| `cutile doctor --device cpu` | 退出码 0 | **退出码 0** | 退出码 0 |

「本轮自身改动」的 +10 行由四个文件构成：`storage/src/lib.rs` +4（平台说明）、`storage/src/mmap.rs` +3（模块说明）、
`weights/src/lib.rs` +5（拥有/依赖段重写）、`weights/src/seg.rs` −1（删空循环、加三行文档注释），
合计 +11；与上轮记录的总数差 1 行属既有口径差，以 `wc -l` 实测为准。

### 16.6 复核期间的一次并发改动（不属于本轮，但影响数字）

- 现象：09:05 我第一次跑全仓时是 60 目标 / 1 057 通过；09:09–09:21 之间，`crates/diagnostic/src/`
  多出 `code.rs`、`error.rs`、`buildgen.rs`（`lib.rs` 改写为模块目录式入口），`crates/{op,prof,tool}/build.rs`
  改为共用 `diagnostic::buildgen::emit`。**这些文件都不在本轮的编辑范围内**，本轮只改
  `crates/storage/**`、`crates/weights/src/{lib,seg}.rs`、`crates/weights/tests/round_trip.rs`、
  `crates/op/README.md`、`docs/ARCHITECTURE.md`、`docs/audit.md`、`CHANGELOG.md`。
- 中途红灯的现场（原样保留，作为"审计数字必须带时点"的例证）：
  - 09:13 `cargo test -p llm-tool --test gate` → `crates/tool/build.rs:27` 的 `E0308`
    （`buildgen::emit` 返回 `Result`，调用点未解包）；同一次报告里还有
    `crates/diagnostic/src/error.rs:fmt#0 新增 Ccn=6`、`buildgen.rs:value_expr#0 新增 Ccn=9`、
    `buildgen.rs 新增 Call=6`、`lib.rs:fmt#1 已全部达标，删除基线条目` —— 四条全在 `diagnostic`/`tool`。
  - 09:15 `cargo check --workspace --all-features` → 退出码 101，`crates/op/build.rs:8`
    `unresolved import diagnostic`（`[build-dependencies]` 尚未补）。
  - 09:18、09:22 两次复测分别回到退出码 0。
- 09:25 收敛后的实测：`cargo check --workspace --all-features` 0、`--test gate` 1/1、
  全仓 **1 062 通过 / 0 失败 / 1 忽略**、`--workspace --exclude llm-benches --all-targets --all-features
  --target x86_64-pc-windows-msvc` 0、`clippy -D warnings` 与 `cargo doc -D warnings` 均 0 告警、
  `cutile doctor --device cpu` 退出码 0。
- 结论（方法论，值得写进报告而不只是记在心里）：**§15.5 与本节的"全仓"两列不是同一棵树**。
  任何"全仓 N 通过"的结论都必须带测量时点与当时的 `git status`；否则一次在途改动就能让
  上个数字看起来像错的。本报告从 §12 起每节都注记时点，§16.6 是把"观察到的并发改动"
  写进正文的第一次——这是本报告自身最该被复制的一条实践。

### 16.7 本轮没碰的东西

- §8.2 第二梯队（5、6 已兑现见 §18；8 见 §19；7 见 §20；9 见 §22）、
  第三梯队（10–12）、第四梯队（13）原样保留。
- §15.4 的四条同族缺陷（`gguf.rs:199,358`、`zip.rs:128-142`、`axes.rs:89` 的 `assert!`）在本轮范围外，
  **下一轮已结清**（§17，另补第五条 `zip::slice`）。
- `llm-benches` 的 Windows 交叉检查（§16.2 边界）。
- `docs/lld.md` 与 `docs/10k-analysis.md` 的规模数字（P3-6）仍未核。
- `crates/diagnostic/**` 与 `crates/{op,prof,tool}/build.rs`：复核期间正在被另一处改动改写（§16.6），
  本轮一行未动；它们的门禁键与测试数由那次改动自己负责。
- 未新增 ADR：本轮三件都是既有契约内的修正（"平台条件属于模块边界""清单只有一处事实源"），
  不改变任何架构决策。

### 16.8 复核命令

```bash
cargo check -p llm-storage --features mmap --target x86_64-pc-windows-msvc   # 修复前 2 错，现绿
cargo check --workspace --exclude llm-benches --all-targets --all-features \
    --target x86_64-pc-windows-msvc                                          # 退出码 0（唯一平台缺口已关闭）
cargo test -p llm-storage --features mmap                                    # 8 + 1 通过
cargo test --workspace --all-features                                        # 60 目标 / 1 057 通过 / 0 失败 / 1 忽略
cargo test -p llm-tool --test gate                                           # 静态门：ccn≤5、nloc≤60、零 panic、棘轮
grep -rn "pack::" crates/*/src src                                           # 只剩变量名误命中
```

---

## 十七、§15.4 四条残余的同族收口（本轮）

> 上一轮（§15）修完四条 P0 时明确留下"同族未复核"的四条。本轮按同一条判据处理：
> **长度只要来自文件，就不许参与未检查的加/减**。过程中又发现第五条同一判据的违例
> （`zip::slice` 的 `at + len`），它不在 §15.4 的清单里，但一个 ZIP64 偏移就能踩到。

### 17.1 结论

四条残余与一条新发现全部改成"越界即错误"，并且**没有抬高任何一条复杂度基线**：`zip64_extra`
抽出字段读取后 ccn 反而降到限额内，棘轮因此要求删除它的基线条目（253 → 252）。三条回归测试在
修复前是红的，报的是 `attempt to add with overflow` 原文而不是断言信息，说明它们踩的是真实的
算术溢出路径；第四条踩的是"声明 8 字节的字段被当成 16 字节读"，旧代码返回 `Ok(错值)`。

### 17.2 五处修法

| 位置 | 修前 | 修后 | 为什么不是别的写法 |
| --- | --- | --- | --- |
| `zip.rs:179-184` `slice` | `bytes.get(at..at + len)`：`at` 是 ZIP64 的 `u64` 原样转成 `usize`，`at + len` 在 debug 下 panic，release 下回绕成 `get(start..start-4)` 再落成 `None` | `at.checked_add(len)` → `Err("ZIP 字段偏移溢出")` | 用饱和加法虽然同样不会 panic，但错误会退化成"越界"，丢掉"偏移本身溢出"这个事实 |
| `zip.rs:127-156` `zip64_extra` / `zip64_field` | `stop = extra_start + extra_len` 与 `at = body + size` 未检查；`off = body + field*8` 也不看子字段声明的 `size`，于是声明 8 字节的字段能读到下一段数据 | 推进用 `saturating_add`；取值交给新函数 `zip64_field`，它要求 `(field+1)*8 <= size`，否则 `Err("ZIP64 extra 声明长度不足，字段被截断")` | 在循环里加 `if` 会把 `zip64_extra` 的 ccn 从 6 推上去、被门禁拦回；抽出后它降到 ≤5，基线条目被棘轮删掉 |
| `gguf.rs:377-388` `raw` | `data + offset` 用了 `checked_add`，但随后的 `start + len` 没有——`offset` 是文件里的 `u64`，取到 `usize::MAX - data` 就在 debug 下 panic | `start.saturating_add(len)` | 加 `?` 会把 `raw` 的 ccn 从 6 推到 7（基线是 6）；饱和后区间必然 `get` 失败，错误语义不变 |
| `gguf.rs:225-231` `Reader::array_count` | `self.bytes.len() - self.pos`，靠 `take` 维持 `pos <= len` 的不变式 | `saturating_sub` | 不变式确实成立（`take` 只在 `get(pos..stop)` 成功后推进），改法的价值是把"依赖别处的正确性"变成"本处自证"，行为零差异 |
| `axes.rs:73-98` `search` | 公开函数末尾 `assert!(verified(&best, w), "raw 基线必须逐位还原")`——库内 panic，而签名本来就返回 `Result` | `verified(..).then_some(best).ok_or_else(baseline_broken)` | `if !verified(..) { return Err(..) }` 会 +1 ccn（基线 8）；闭包形式也会 +1，见 §17.5 |

### 17.3 先红后绿：修复前的原始输出

四条用例在**只加测试、未改代码**的树上跑出来的原文（`cargo test -p llm-weights --lib`）：

```text
---- gguf::tests::raw_rejects_an_offset_whose_length_overflows
thread '...' panicked at crates/weights/src/gguf.rs:386:25: attempt to add with overflow

---- zip::tests::slices_with_overflowing_offsets_are_errors_not_panics
thread '...' panicked at crates/weights/src/zip.rs:167:19: attempt to add with overflow

---- zip::tests::zip64_extra_with_absurd_lengths_is_an_error
thread '...' panicked at crates/weights/src/zip.rs:130:16: attempt to add with overflow

---- zip::tests::zip64_fields_must_fit_the_declared_size
thread '...' panicked at crates/weights/src/zip.rs:228:9: 字段 1 超出声明的 8 字节，不得把邻居数据当成字段值
```

前三条是**算术溢出 panic**（不是断言失败），证明测试走的是真实的溢出路径；第四条是断言失败，
说明旧代码确实把邻居数据当成了字段值并返回 `Ok(..)`。修复后同一条命令：`174 passed; 0 failed`
（原 170 + 4）。

第四条用例在修复后改名为 `zip64_extra_handles_absurd_lengths_without_overflow`：它同时断言"巨大 `extra_len` 不 panic"与"越界即报错"，原名只说了后半句。

上表里的行号是**修复前**的位置（终端输出原样保留）。修复后的锚点：`zip::slice` 在 `zip.rs:179-184`、
`zip64_extra` 在 `zip.rs:131-144`、新增的 `zip64_field` 在 `zip.rs:146-156`、`gguf::raw` 在
`gguf.rs:378-388`、`Reader::array_count` 在 `gguf.rs:225-231`、`axes::search` 在 `axes.rs:73-98`。

### 17.4 有意不改的同类位置（附判据，避免"看见加号就改"）

| 位置 | 为什么安全 | 判据类型 |
| --- | --- | --- |
| `zip.rs:120-124` `data()` 的 `entry.start + entry.len` | `Entry` 字段私有、只由 `parse` 构造，而 `parse` 在 `:88-91` 已用 `checked_add` 保证 `start + len <= bytes.len()` | 不变量由**构造函数**建立，不靠调用方 |
| `reader.rs:200-246` 的 `self.bytes[...]` 直接索引 | `seg.start` / `seg.len` 在 `:153-158` 已按段表边界校验 | 同上；改成 `get` 只是把"已校验"重复成两处事实 |
| `bin.rs:160-164`、`axes.rs:268-272,308-312`、`lattice.rs:403-407,456-460`、`ntt.rs:56-60,78-82`、`rans.rs:351-357`、`sha256.rs:191-195`、`json.rs:192-196,314-318` | 偏移是循环内自增或已知常量宽度，且推进只发生在 `get(..)` 成功之后 | 上限由"成功的切片"本身给出，不可能越过 `usize::MAX` |
| `dsa.rs:90-93` `Xof::take` 的 `&self.bytes[..]` | 缓冲区按 `bound` 一次性生成，`at` 只在成功 `take` 后推进 | 内部类型、不接收外部长度；但它仍是**库内 panic 面**，见 §17.7 |
| `rans.rs` 的分配上限 | §15 已把"4 字节任意值"改成"必须自洽的极端形状"，但仍无全局封顶 | 这是**策略**问题（调用方预算或压缩比上限），不是长度校验 |

### 17.5 一条被门禁逼出来的写法

第一次把它们改完时，`axes::search` 用的是 `ok_or_else(|| bad("…"))`，门禁直接拦回：

```text
静态质量门失败：
  crates/weights/src/axes.rs:search#0: Ccn 9 恶化 > 登记 8
  crates/weights/src/zip.rs:zip64_extra#0: 已全部达标，删除基线条目
```

两行都是有用的信息。第一行说明**闭包里的 `||` 会被当成一个布尔分支**计入 ccn（与 `&&`/`||`
一视同仁），于是"把 panic 换成错误"这种纯正确的改动也会涨复杂度——最终改成传函数项
`baseline_broken`，`search` 回到 8。第二行是棘轮第一次在我没有主动删条目的情况下要求删除条目：
`zip64_extra` 抽出字段读取后 ccn 降到限额内，保留条目本身就是"过期基线"。

这条口径值得记住：**`?` 与 `ok_or_else(|| ..)` 都是有代价的**。在已经贴着基线的函数里，
"更 Rust 的写法"可能比"更笨的写法"更贵。

### 17.6 实测对照

| 指标 | 上一轮（§16.5） | 本轮 | 说明 |
| --- | ---: | ---: | --- |
| `crates/*/src` 行数 | 54 980 | **55 221** | `wc -l` 实测（09:31）；≈+67 可归因于本轮（3 条回归测试 + `zip64_field` + 4 处校验），其余差额与工作树里 69 个未提交路径的在途改动混在一起，未逐项归属 |
| `crates/*/src` 文件数 | 172 | **172** | 不变（只改不增删） |
| `[[gate.debt]]` 条目 | 253 | **252** | −1：`zip64_extra` 降到限额内，条目被棘轮要求删除 |
| 工作区成员 | 25 | **25** | 不变 |
| 依赖边 | 75（`+dev` 11） | **75（11）** | 不变（本轮不动依赖） |
| 用例 | 1 062 通过 / 60 目标 | **1 066 通过 / 60 目标** | +4 条回归测试 |
| `clippy -D warnings` / `cargo doc -D warnings` | 0 / 0 | **0 / 0** | 两个口径都干净 |
| `cutile doctor --device cpu` | 退出码 0 | **退出码 0** | 配置段与协商输出未受影响 |

### 17.7 本轮没碰 / 仍在队列

- `rans` 的全局上限（§17.4 最后一行）：需要"调用方预算或压缩比上限"这类策略决策，不是长度校验。
- `dsa::Xof::take` 的库内索引：它是内部类型、偏移不可由外部给定；改成 `Result` 会牵动
  `rej_ntt_poly` 等 6 个调用点的签名，收益不足，记为已知 panic 面而不是现在改。
- **ZIP64 的字段顺序假设**：真实规格允许"只写需要的那几个字段"，而 `zip64_extra` 按固定序号
  `0..3` 取值。现在 `zip64_field` 会**拒绝**声明长度不足的读取（不再误读邻居），但"只写了 `csize`
  这一个字段"的归档仍会被判成"字段 0"——这是既有的解读歧义，本轮只把它从"静默错值"降级为
  "要么正确、要么报错"，没有引入按 sentinel 重排字段的新逻辑（当前没有真实样本需要）。
- `zip.rs` 的两条既有用例仍可空转（`/tmp/pt/model.bin` 不存在时直接 `return`），属 §P2-9 的老问题。
- §11.4 队列（`weights` 零引用簇、`storage` 布局并入 `tensor`、`exec` 死 SPI、`snapshot` 拆文件、
  `Facts` 补两个事实）与 §8.2 第二梯队（5–9）原样保留。
- 未新增 ADR：本轮是既有契约（"长度来自契约"）的补齐，不改变任何架构决策。

### 17.8 复核命令

```bash
cargo test -p llm-weights --lib          # 174 通过（含 17.3 的四条）
cargo test -p llm-tool --test gate       # 静态门：ccn≤5、nloc≤60、零 panic、棘轮（现 252 条基线）
cargo test --workspace --all-features    # 60 目标 / 1 066 通过 / 0 失败 / 1 忽略
cargo run --bin cutile -- doctor --device cpu   # 退出码 0
```

---

## 十八、§8.2 第二梯队第 5、6 项落地（本轮）

> 上一轮（§17）结清 §15.4 的四条同族长度缺陷后，按 §8.2 的排序继续做第二梯队。本轮做第 5、6 两项：
> 表达式缓存收敛、`gemv` 输入校验上提。两项都在 `op` 内，都是判据 B（"新增一项要改 N 处"）。

### 18.1 结论

四份逐字同构的表达式缓存收成 1 个 `FormulaCache`，三条格式与参考实现里重复的输入校验收成 3 个骨架级
helper。净减 **84 行**（`crates/*/src` 55 221 → 55 137）；用例总数不变（**60 目标 / 1 066 通过**，与
改动前逐项相同）；`[[gate.debt]]` 252 → **251**（`gemv_i8::layout` 整段消失）；**没有新增、也没有抬高
任何一条基线**，`gemv_i8::forward` 反而从 ccn 6 压回 5（§18.5）。

### 18.2 缓存收敛：4 类 → 1 类

 原类型 | 原位置 | 行数（含 `Default`） | 现在 |
 --- | --- | ---: | --- |
 `gemv::Cache` | `op/src/gemv.rs:250` | 32 | 删；三个格式改 `Format::cache()` |
 `elementwise::BinCache` | `op/src/elementwise.rs:41` | 35 | 删；两个实现改 `Binary::cache()` |
 `unary::UnaryCache` | `op/src/unary.rs:32` | 36 | 删；三个实现改 `Unary::cache()` |
 `reduce::ReduceCache` | `op/src/reduce.rs:31` | 36 | 删；两个实现改 `Reduce::cache()` |
 `einsum::FormulaCache` | 新增 `op/src/einsum/cache.rs` | 61（含模块文档 9 行） | 唯一的解析 + 缓存逻辑 |

四者的 `get` 逐字相同（取 `OPERANDS` → `Operand::new` → `Expr::parse` → `Formula::new` → `with(site)`），
差别只有三件：表达式文本、操作数表、错误帧。收口后这三件都成了**参数**，`site` 由骨架给出——同一条骨架
的所有精度共享同一个位置，重复写在每个实现里只会多出几份可以写歪的事实。

**没有做"泛型 `static`"**：`FormulaCache::get` 是普通方法，`static` 留在各实现的 `cache()` 里。这不是
风格偏好——写在泛型作用域里的 `static` 在 Rust 里只有一份，`Dense<B>` 与 `Quantized<G>` 会共用同一个
单元格，先跑的那一份会让另一份拿到错误的轴表（`gemv.rs` 原注释记录过这次实测）。资产是"一份逻辑"，
不是"一份单元格"。

### 18.3 校验上提：3 条格式 + 参考实现 → 3 个 helper

骨架的 `reference()` 一直声称"输出非空、输入够长、权重够长由骨架负责"（`gemv.rs:355-358`），但三条
`forward` 各抄了一遍，`gemv_i8` 还为此专门写了一个 `layout()`（32 行、ccn 6）。现在：

 helper | 覆盖的判据 | 调用方 |
 --- | --- | --- |
 `gemv::non_empty(n)` | 输出维度非 0 | 三条 `forward` + `reference` |
 `gemv::scaled_groups(m, n, scales)` | 标度按行均分、分组能整除输入宽度 | `gemv_i4::forward`、`gemv_i8::forward` |
 `gemv::required_weights(weights, n, row)` | 权重 ≥ `n × row`，返回这个下界 | 三条 `forward` + `reference` |

错误文案逐字保留（`输出维度不能为 0`、`scale 长度 … 不是 … 的整数倍`、`scale 长度 … 推不出能整除 … 的
分组数`、`权重长度 … < …`），只是从 6 处变 3 处。**上界用饱和乘**：`n · row` 溢出时判为"长度不足"，而
不是回绕成一个轻松满足的小数——这是本轮顺带收紧的一点（原实现是裸乘）。

留在各自 `forward` 里的差异只有两条，它们属于格式而不属于骨架：int4 的"分组必须是偶数"
（`gemv_i4.rs`），稠密格式的"不该收到标度"（`gemv_f32.rs`）。

### 18.4 顺带修的层次错位：`OperandSpec` 回到表达式层

`OperandSpec`（表达式里一个操作数的名字/位宽/方向）原来定义在 `gemv.rs`，而 `elementwise`、`unary`、
`reduce` 都写着 `use crate::gemv::OperandSpec` ——三条兄弟骨架反向依赖其中一条。它描述的是**表达式**，
所以随本轮搬进 `op/src/einsum/formula.rs`，由 `einsum` 导出；`gemv` 不再拥有它。同一轮里
`einsum` 也拿到了 `FormulaCache`，"表达式 + 它的缓存"因此只有一个主人。

连带的门禁改动（都是"跟着事实走"，不是放宽）：
- `docs/ARCHITECTURE.md:100` 的所有权表条目 `Cache` → `FormulaCache`，并写清"实现一份、单元格每格式一份"。
- `lints/vocabulary.toml` 的 `[[ownership.unlisted]]`（`crate = "op"`）删掉 `BinCache`/`ReduceCache`/
  `UnaryCache` 三个已不存在的名字；`FormulaCache` 进表，不进基线。

### 18.5 一条被门禁逼出来的写法（与 §17.5 同源）

`gemv_i8::forward` 把校验上提后变成 3 个 `for` + 2 个 `?`，ccn = 6，
门禁直接判"新增 Ccn=6"：

```text
静态质量门失败：
  crates/op/src/gemv_i8.rs:forward#0: 新增 Ccn=6
  crates/op/src/gemv_i8.rs:layout#0: 已全部达标，删除基线条目
```

两条都是有用信息：后者就是本轮该删的条目（`layout` 已不存在）；前者说明"少抄一份校验"会以 ccn 的形式
付账。最终把内层点积改成一次 `map().sum()`（`f32` 的 `sum` 与原循环同为从左到右折叠，逐位相同），
`forward` 回到 5。**这是本轮唯一一处为了门禁而改变写法的地方**，且改完更好读：点积不再借外层累加器。

### 18.6 实测对照

 指标 | 上一轮（§17.6） | 本轮 | 说明 |
 --- | ---: | ---: | --- |
 `crates/*/src` 行数 | 55 221 | **55 137** | `wc -l` 实测；−84 = 缓存 139 → 61、`layout` −32、校验 −≈50、11 个算子文件的 import 与 `cache()` 改写 +≈60（推断，逐文件净额未逐项归属） |
 `crates/*/src` 文件数 | 172 | **173** | +1 为 `einsum/cache.rs` |
 `[[gate.debt]]` 条目 | 252 | **251** | −1：`gemv_i8::layout` 随函数消失 |
 工作区成员 / 依赖边 | 25 / 75（`+dev` 11） | **25 / 75（11）** | 不变（本轮不动依赖） |
 用例 | 1 066 通过 / 60 目标 | **1 066 通过 / 60 目标** | 逐项目标相同，没有测试被删或降级 |
 公开类型 | 4 个缓存类型 | **1 个 `FormulaCache`** | 三个旧类型名从所有权基线里删除 |
 `clippy -D warnings` / `cargo doc -D warnings` | 0 / 0 | **0 / 0** | 两个口径都干净 |
 `cutile doctor --device cpu` | 退出码 0 | **退出码 0** | 配置段与协商输出未受影响 |

### 18.7 本轮没碰 / 仍在队列

- ~~**§8.2 第 9 项**：门禁的 ccn 分档与真棘轮~~（**§22 已落地**：真棘轮已生效、分档明确不做）。（第 8 项 `codegen.rs` 的族分支已在 §19 落地；
  第 7 项的注册表/内核表两半已在 §20 落地，`forward` shim 那一半经实测撤回。）
- **`gemv_i4::forward` 仍是 ccn 7、`gemv_f32::forward` 6、`gemv::reference` 未降到 5**：都低于各自
  登记的基线（10 / 7 / 9），所以既不该删条目，也没必要为凑 5 再拆一层。
- `Code::Invalid` 的四个错误分支现在各自只在 helper 里出现一次，但**没有合并成"一个错误码"**——它们是
  四条不同的判据，合并会丢掉"是哪一个不满足"。
- §11.4 队列（`weights` 零引用簇、`storage` 布局并入 `tensor`、`exec` 死 SPI、`snapshot` 拆文件、
  `Facts` 补两个事实）与 §15.4 的 `rans` 全局上限原样保留。
- 未新增 ADR：本轮是收敛重复实现，不改变任何架构决策；`FormulaCache` 的归属由 §18.4 的层次理由给出。

### 18.8 复核命令

```bash
cargo test -p llm-op --all-features          # 157 通过 + 2 doctest
cargo test -p llm-tool --test gate           # 静态门：ccn≤5、nloc≤60、零 panic、棘轮（现 251 条基线）
cargo test --test architecture --test contracts   # 8 + 6 通过（所有权表与边集）
cargo test --workspace --all-features        # 60 目标 / 1 066 通过 / 0 失败 / 1 忽略
cargo run --bin cutile -- doctor --device cpu     # 退出码 0
```

---

## 十九、§8.2 第二梯队第 8 项落地：`#[model]` 的十二段同形分支改成一张骨架表（本轮）

> 上一轮（§18）结清第二梯队第 5、6 项。本轮做第 8 项：`annotation/src/codegen.rs` 里 12 个骨架各写
> 8 行逐字相同字段的那串 `else if`。判据仍是 B（"新增一项要改 N 处"），但结论与 §8.2 的预估相反：
> **复杂度大降、行数微增**——这不是没做到，是预估的前提错了（§19.5）。

### 19.1 结论

`render_model` 从 **`ccn 15 / nloc 233`** 降到 **`ccn 2 / nloc 19`**（实测，与 `lints/vocabulary.toml`
原登记值 `ccn = 15, nloc = 233` 对应）。十二段同形分支（原 `codegen.rs:212-363`，152 行）变成一张
12 行的 `SKELETON_TABLE` 加 6 个各 ≤ 54 行的函数；`crates/annotation` 用例 53 → **54**（新增 1 条锁表
测试，另在既有测试里加了两条反向断言），其余测试目标逐项不变。**行数没有下降**：`codegen.rs`
761 → 794、`crates/*/src` 55 137 → 55 187。`[[gate.debt]]` 251 → **250**：`render_model#0` 已不在表里，
本轮实测其两条度量都已达标，删除成立。

### 19.2 表里放什么：一列一个差异

`Skeleton` 的五个字段（`codegen.rs:232-243`）各自替换掉 if-chain 里的一类差异：

 字段 | 取值 | 它替换掉的差异
 --- | --- | ---
 `name` | 12 个闭集成员之一 | `== "elementwise"` … 共 12 个判等
 `module` | `Some("rmsnorm_f32")` / `None`（用注册名） | 每个分支里写死的 `::llm_op::{模块}::`
 `group` | `"1u32"` / `"2u32"` / `"<{ty} as …::Format>::GRAIN as u32"` | `group: 1u32` 与 gemv 的 `GRAIN`
 `generic` | 是 / 否 | 入口写 `cost::<T>` 还是 `cost`
 `grain` | 是 / 否 | gemv 分支末尾那两条编译期断言

十二条里只有 `gemv` 一行 `grain = true`、四条骨架 `generic = true`、一行 `group` 是 2。这就是原来十二段
拷贝想表达的全部信息量：**其余部分是同一份字段列表**，抄 12 遍换不来任何区分度。

### 19.3 三个例外为什么留在数据里，而不是回到分支

1. **`gemv` 的粒度与两条断言**：粒度是 `Format::GRAIN`，注解里不该再写一个数（`op` 已经把它做成关联
   常量）。表里因此多一列 `grain`，断言本体留在 `render_grain_asserts`（`codegen.rs:354-369`），规则
   仍来自 `Format::CHECK`——宏不另抄一份"粒度必须大于 0"。
2. **`gemm` 没有固定模块名**：三种格式的模块名就是注册名，所以 `module = None` 表示"用注册名"。这个名字
   合不合法，文法层已经用 `GEMM_SKELETON_NAMES`（`syntax.rs:409-411`）拒过一遍，生成器不必再判。
3. **`flash_attention_f32` 走另一条参考实现**：两条参考实现同在 `op::attention`，骨架、轴表、代价全相同，
   只有算法不同（flash 版分块、朴素版一遍算完）。这属于"这对权重用哪种算法"，不属于"这份语义是哪副
   骨架"，所以它不进骨架表，而进 `REFERENCE_OVERRIDES`（`codegen.rs:288-296`）——一张按**注册名**
   改写的表，也是 P1-5「宏知道算子名」的全部剩余（§19.6）。

### 19.4 生成文本没变的证据

生成物是字符串，等价性只能靠"哪 12 份文本被产出过"来证明。三层判据都在 `cargo test` 里：

1. `every_skeleton_has_a_branch`（`codegen.rs:688-723`）遍历 `syntax::SKELETONS` 逐项渲染，断言不出现
   `compile_error!` 且必绑一份轴表。本轮**加了两条反向断言**：表里每一行都必须属于闭集、表长必须等于
   闭集长度。"少一行"会让某个骨架渲染成 `compile_error!`，"多一行"是死数据，两个方向现在都有判据。
2. `the_table_keeps_the_rows_that_are_easy_to_get_wrong`（新增）：把三处"改一个字符就静默换掉生成代码"
   的事实各钉一条——具体骨架不得带 `::<T>`、`dequant_i4` 的 `group` 必须是 `2u32`、
   `flash_attention_f32` 的 `reference` 必须是 `flash_reference` 而 `attention_f32` 必须是 `reference`。
3. 真正消费生成物的是 22 个算子模块里的 `#[model]`：`cargo test -p llm-op` 157 条通过，每条 `MODEL`
   都编译并跑过它自己的 `__model_check`（由 `codegen.rs:371-434` 生成）。

### 19.5 "预估 −80 行"为什么不成立

§8.2 第 8 项与 §10.4 都写着"一张 `(族名 → 模块路径)` 表 + 一个循环 → −80 行"。这个估算的隐含前提是
**除模块名外没有别的差异**；实际有三个（§19.3），于是两列的表变成五列，还要留两个小函数（粒度断言、
名字改写）。再加上本仓的注释密度（每个函数、每个字段都有一段"为什么"），净结果是 **+33 行**
（`codegen.rs` 761 → 794，实测）。

不逐段列加减账：原来 152 行的 if-chain 是 `render_model` 的一部分，60 行的 `__model_check` 也在里面，
拆开之后"删了多少、加了多少"取决于怎么切分，写成一个精确到行的表格只会让读者以为它可复核。可复核
的是两件事：`wc -l` 的净差是 +33，`render_model` 的 nloc 从 233 掉到 19。

也没有为了凑数把五列压成一行 `row("gemv", Some("gemv"), "<{ty} …>::GRAIN as u32", true, true)` 的密集
写法：那样能省 20 行，但读表的人要在脑子里解位置参数。**本轮买的是"加骨架只加一行数据"与"单函数
复杂度 15 → 2"，不是行数**——把它记成减行才是真的失真。

### 19.6 P1-5 的现状：宏里还剩一行算子名

 检查项 | 本轮前 | 本轮后
 --- | --- | ---
 按**族名**分支 | 11 处 `if / else if`（原 `:211-310`） | **0**：`skeleton_of`（`codegen.rs:298-301`）查表
 按**算子名**判等 | 4 处（`flash_attention_f32`、`gemm_f32`、`gemm_i8`、`gemm_i4`） | **1 行数据**（`REFERENCE_OVERRIDES`）；三个 `gemm_*` 的判等消失，因为模块名就是注册名
 闭集 ↔ 生成器一致性 | 靠 12 段分支的存在来保证 | 双向断言（§19.4 第 1 条）

剩下的"宏知道领域"是 `skeleton` 这个名字本身（`syntax::SKELETONS` 的 12 项）。这是设计而不是缺陷：
注解要表达"这份语义是哪副骨架"，而骨架就是 `op` 的闭集。P1-6 指的两套词汇表（`op::decl::Skeleton`
5 项 vs 注解的 12 项）仍然并存，本轮没有合并——合并的前提是先裁决 `dequant*`、`softmax` 这些"一族一
骨架"的名字要不要进 `op`，那是领域问题，不是生成器问题。

### 19.7 事实基线回填

- **最长函数 233 → 202 行**：`annotation/src/codegen.rs:render_model` 不再是全仓最长，现为
  `prof/src/prof.rs:validate`（202 行，登记 `ccn 38 / nloc 202`）。§2.1 的最长函数行已改写，§12.4 的
  原始快照保留并加了指向。
- **`crates/annotation` 债务**：`render_model#0` 不在 `[[gate.debt]]` 里（工作树中即缺失），实测
  `ccn 2 / nloc 19`，删除成立。`codegen.rs:render#0`（`nloc 66`）与文件级 `call = 1` 保留——`render`
  仍是 66 行，`dtype_variant`（`codegen.rs:167-171`）仍有一处 `panic!`，它只在文法层闭集失守时可达。
- **文件行数棘轮当场生效**：修 `syntax.rs:452` 的文档漂移（写成"缺省为 `gemm`"，实际是 `gemv`）时多写
  了一行，门禁立刻拦回：

  ```text
  静态质量门失败：
    crates/annotation/src/syntax.rs: Lines 1229 恶化 > 登记 1228
  ```

  把那段注释压回两行后才通过。`lines` 基线只准降这一条是真的在跑（与 §17.5 同源）。
- 同处修正 `syntax.rs:384-387` 的措辞："每个成员都有自己的**分支**" → "自己的一**行**"，因为分支已经
  不存在；措辞与代码对齐比留一句过时的描述更重要。

### 19.8 本轮没碰 / 在途差额

- **§8.2 第 7 项已在 §20 落地**（`op::REGISTRY` 与 `train::KERNELS` 两张表都成一行一对；`forward` shim
  那一半撤回，理由见 §20.5）；**第 9 项**（门禁 ccn 分档、真棘轮）**已在 §22 落地**（分档那半边明确不做）。
- **P1-6 两套 `skeleton` 词汇表**未合并（§19.6）：本轮只让生成器的表与注解闭集双向对齐，`op::decl::
  Skeleton` 仍无任何核对方。
- **用例基线与 §18 的差**：本轮实测 **59 目标 / 1 064 通过 / 1 忽略**，§18 记录的是 60 / 1 066。差
  **−1 个目标**可由工作树里未提交的 `crates/benches` 重构解释：`HEAD` 的
  `crates/benches/tests/elementwise_regress.rs`（4 条 `#[test]`）已被删除，改为新加的
  `criterion/elementwise.rs` bench 目标（`crates/benches/Cargo.toml` 同时去掉了 `tool` dev 依赖）。这
  不是本轮改动，也不该由本轮回退：那 4 条里 3 条是 `tool::regress` 的纯判定测试（`crates/tool/src/
  regress.rs` 现有 6 条同类），1 条是同机计时的"测量 → 基线 → 判定链路接通"断言。**后者现在只存在于
  `cargo bench`**，`cargo test` 不再验证这条链路——记为 **P3 待决**：要么在 `tool::regress` 里用注入值
  补一条等价断言，要么明确接受"只在 bench 里验证"。条数差的剩余 1 条无法归因到本轮改动（§18 的
  1 066 是当时工作树的实测值）。
- §11.4 队列（`weights` 零引用簇、`storage` 布局并入 `tensor`、`exec` 死 SPI、`snapshot` 拆文件、
  `Facts` 补两个事实）与 §15.4 的 `rans` 全局上限原样保留。
- 未新增 ADR：这是把同形分支换成表，不改变任何架构决策。

### 19.9 实测对照与复核命令

 指标 | §18 时点 | 本轮 | 说明
 --- | ---: | ---: | ---
 `render_model` ccn / nloc | 15 / 233 | **2 / 19** | 实测（`llm_tool::analyze`）
 全仓最长函数 | 233（`render_model`） | **202**（`prof/src/prof.rs:validate`） | 与登记值一致
 `crates/*/src` 行数 | 55 137 | **55 187** | +50：`codegen.rs` +33，其余 +17 属工作树在途改动，未逐项归属
 `crates/*/src` 文件数 | 173 | **173** | 不变
 `[[gate.debt]]` 条目 | 251 | **250** | `render_model#0` 不在表中，实测已达标
 用例 | 60 目标 / 1 066 | **59 目标 / 1 064 / 1 忽略** | −1 目标见 §19.8（`crates/benches` 在途重构）；本轮 +1 条
 `crates/annotation` 用例 | 53 | **54** | 新增锁表测试 + 两条反向断言
 `clippy -D warnings` / `cargo doc -D warnings` | 0 / 0 | **0 / 0** | 两个口径都干净
 `cutile doctor --device cpu` | 退出码 0 | **退出码 0** | 生成物未变的旁证

```bash
cargo test -p llm-annotation                     # 54 通过（含 §19.4 的三层判据）
cargo test -p llm-op                             # 157 通过：22 个 MODEL 的 __model_check
cargo test -p llm-tool --test gate               # ccn≤5、nloc≤60、零 panic、棘轮（现 250 条基线）
cargo test --test architecture --test contracts  # 8 + 6 通过
cargo test --workspace --all-features            # 59 目标 / 1 064 通过 / 0 失败 / 1 忽略
cargo run --bin cutile -- doctor --device cpu    # 退出码 0
```

---

## 二十、§8.2 第二梯队第 7 项落地：两张 22 条的注册表各收成"一行一对"（本轮）

> §19 结清第 8 项。本轮做第 7 项，同时把一个被反复引用的估算纠正掉：这一项的价值不在
> 减行数，而在**把"同一对事实写两遍"变成写一遍**，而 `forward` shim 那一半经实测不成立（§20.5）。

### 20.1 结论

两处逐字重复的结构体字面量都改成了一张**一行一对**的表：

 文件 | 改前 | 改后 | 行数
 --- | --- | --- | ---: 
 `crates/op/src/lib.rs` | 22 条 `Registration { name, template: include_str!(..), model: .. }`（110 行） | `registry! { "名字" => 语义模块 }` 22 行 + 15 行宏 | 479 → **431**（−48）
 `crates/train/src/store.rs` | 22 条 `Entry { op: "名字", run: 同名函数 }`（88 行） | `kernel_table![add_f32, ..]` 22 行 + 10 行宏 | 1531 → **1479**（−52）

`crates/*/src` 55 138 → **55 086** 行、文件数 173 不变；`[[gate.debt]]` 仍 250 条（本项没有消掉任何
登记项，只把 `store.rs` 的 `lines` 基线按实测 **1529 → 1479** 下调，规则允许的方向）；用例
59 目标 / **1 074 通过 / 0 失败 / 1 忽略**——其中本轮新增 1 条（§20.2），另 9 条来自工作树里正在
进行的 `cutile` 改动（`cutile` 库测试 157 → 166），本项不认领这部分。

### 20.2 `op::REGISTRY`：一列变两列

`Registration` 有三个字段，其中 `name` 同时是 `ops/<name>.toml` 与 `ops/<name>.k` 的文件名，
`template` 由 `include_str!(concat!("../ops/", $name, ".k"))` 拼出、`model` 是 `$module::MODEL`。
于是 22 条注册里**唯一会变的只有两列**（名字、模块），剩下三个字段是宏的机器活：

```rust
macro_rules! registry {
    ($($name:literal => $module:ident),* $(,)?) => {
        pub const REGISTRY: [Registration; <[()]>::len(&[$(registry!(@one $name)),*])] = [
            $(Registration {
                name: $name,
                template: include_str!(concat!("../ops/", $name, ".k")),
                model: $module::MODEL,
            }),*
        ];
    };
    (@one $name:literal) => { () };
}
```

长度用 `<[()]>::len(&[..])` 从表本身数出来，不另写一个 `22`：**数组长度是第二个会漂移的数字**，
原实现里它恰好没漂移过，但没有任何东西拦着它漂。

为什么不用 `build.rs` 扫 `ops/` 目录自动生成（§10.3 表格里原先写的做法）：目录扫描会把
"文件名即算子名"变成隐式约定（`ops/` 里多一个草稿 `.toml` 就多一条注册、少一个 `.k` 就是构建期
`panic!`），还要靠 `rerun-if-changed` 维护增量正确性；而**结构重复已经被宏消掉了**——目录扫描能再
省的只是那 22 行名字，代价是把一条编译期可见的表变成构建脚本的副作用。`builtin_matches_disk`
（`op/src/lib.rs` 的测试）继续把表与磁盘上的 22 对文件对齐，这是原有的平价判据，本轮未动。

新增 1 条用例 `the_registry_covers_the_generated_table_exactly`：把 `names()`（从 `REGISTRY` 投影）
与 `builtin()`（从表生成的声明）逐条比对，并把"生成表里有、注册表里没有"与反向各判一次。它挡的是
宏化之后新出现的一类事故：**名字拼错但文件存在**（`"gemv_i4 "` 之类的尾随空格会拼出一个不存在的
文件名，`include_str!` 直接编译失败；但 `"gemv_i4"` 指向 `gemv_i8.k` 这种事只有内容比对能抓）。

### 20.3 `train::KERNELS`：删掉整列字符串

`train` 的标量内核表每行原本要把同一个名字写两遍：

```rust
Entry { op: "gemv_i4", run: gemv_i4 },   // 22 行 × 4 行 = 88 行
```

实测 22 个适配函数**逐条与算子同名**（`attention_f32` 的适配函数也叫 `attention_f32`，
`flash_attention_f32` 同理），所以名字可以由标识符派生：

```rust
macro_rules! kernel_table {
    ($($run:ident),* $(,)?) => { [$(Entry { op: stringify!($run), run: $run }),*] };
}
```

**这不是把判据省掉了。** 名字错了或顺序挪了，`const _: () = same_order(&KERNELS, &op::REGISTRY);`
（`store.rs:537`）立刻失败。本轮做了反向验证——把 `add_f32` 与 `attention_f32` 两行对调后
`cargo check -p llm-train` 报：

```text
error[E0080]: evaluation panicked: train::Reference 的内核表与 op::REGISTRY 名字或顺序漂移
```

恢复后重新变绿。"名字与函数名同名"这条前提现在由那条 `const` 断言守着，而不是靠人记得抄对。

### 20.4 判据没被削弱：`contracts` 门禁重新锚定而不是删掉

`tests/contracts.rs:474` 的 `the_registry_and_the_model_annotations_are_mirrors` 原本**文本扫描**
`crates/op/src/lib.rs` 里 `name:` 与 `model:` 两个字段——正是本轮删掉的那两列，所以它在改造后立刻
变红（`注册表必须闭合`）。处理方式是**改扫描形状**，不是删断言：

- 现在解析 `registry! { "名字" => 模块, .. }` 的每一行，`models` 由 `format!("{module}::MODEL")` 得出；
- 它仍然检查三件事：① 每个 `#[annotation::model(name = "X")]` 都能在注册表里找到 X；② 每条注册的
  `model` 都是 `<模块>::MODEL`、模块文件存在、且那个文件里确实有注解（"语义表必须有唯一生产者"）；
  ③ 反向——注册表里每个名字都要有同名注解。
- 解析不出任何行时，原有的 `names.len() >= 2` 断言会报"扫描逻辑坏了"，不会静默变绿。

即：这条门禁的**语义**（注解 ↔ 注册表互为镜像）没变，变的只是它读的那份文本形状。

### 20.5 撤回："22 个 `forward` shim"不成立

§10.3 与 §8.2 写的是"22 个 `forward` shim，宏 `#[model]` 顺带生成 `forward` 委托，−40 行"。实测
`crates/op/src` 里共 **24 个 `pub fn forward`**，逐个量函数体行数：

 形态 | 数量 | 例子 | 能否宏化
 --- | ---: | --- | ---
 3 行、单行委托 | **9** | `add_f32.rs:43`、`gelu_f32.rs:57`、`gemv_f32.rs:111`、`mul_f32.rs:42`、`reduce_max_f32.rs:41`、`reduce_sum_f32.rs:41`、`residual_add_f32.rs:42`、`scale_f32.rs:42`、`silu_f32.rs:38` | 能，但只省 9×2 = 18 行
 有实体（校验 + 循环/调骨架） | 15 | `softmax_f32.rs:74`（22 行）、`rmsnorm_f32.rs:83`（23）、`attention.rs:195`（23）、`gemv_i4.rs:150`（8，含 `GRAIN` 断言）… | 不能：各有各的算术

这 9 个"shim"的**内容就是公开签名**（`fn forward(a: &[f32], b: &[f32], out: &mut f32) -> Result<()>`）：
它们存在的意义是让 `train`/测试按算子名调用到一个具名入口。把它们塞进宏只能省 18 行、还要 15 行
宏，而 `rustdoc` 与 IDE 跳转看到的会是一个生成的函数——按本仓自己的判据（"禁止为凑指标引入更难读
的抽象"）这是负收益。**因此这一半明确不做**，理由与计数写在这里，避免下一轮再被当成待办。

`gemv_i4`/`gemv_i8` 的 8 行同理不算 shim：它们各自带一条 `Format::GRAIN` 相关的编译期断言。

### 20.6 与 §19 的关系：两轮合起来才把"加算子改几处"降下来

 动作 | 改前 | 改后
 --- | --- | ---
 加一个算子：注册表 | 改 5 行结构体字面量 | **加 1 行 `"名字" => 模块`**
 加一个算子：内核表 | 改 4 行结构体字面量 | **加 1 个标识符**
 加一个骨架：生成器 | 改一段 8 行 `else if`（§19 前） | **加 1 行 `SKELETON_TABLE`**（§19）
 加一个算子：`ops/` 与模块 | 两个文件 + 一个模块 | 不变（这是内容，不是结构）

§19 与 §20 各自只省了几十行，但它们改的是同一件事：**把"新增一项要改 N 处同一形状的代码"变成
"加一行数据"**。这也是为什么两轮的净减（+33 与 −100）都远小于 §8.2/§10.3 的估算：估算是按
"删掉重复行"算的，实际收益在"下一项的成本"。

### 20.7 本轮没碰

- ~~**§8.2 第 9 项**（门禁 ccn 分档、真棘轮）未动~~（**§22 已落地**，`Gate::check` 的"只准降"现在是
  "登记值必须等于实测"；本轮只把 `store.rs` 的 `lines` 基线按实测下调了一次）。
- **P1-6 两套 `skeleton` 词汇表**（`op::decl::Skeleton` 5 项 vs 注解 12 项）未合并，与 §19.6 同因。
- **`train::store.rs` 的 22 个逐算子转接段**（`:567-722`，§10.3 第 4 项）未动：它们的差异在
  `Inputs` 的四元组形状与调用实参顺序，是"内容"不是"结构"，宏化会把类型签名埋进宏参数。
- §11.4 队列、§15.4 的 `rans` 全局上限、§17 已结清项之外的越界风险，全部原样。
- **`crates/cutile/src/execution.rs` 不是我改的文件**，但它的在途版本让全仓测试红了，我修掉了那一条：
  新增的 `check_parity_reports_the_error_within_tolerance` 用 `[1.05]` 对 `[1.0]` 断言误差
  `≈ 0.05`（容差 `1e-9`），而 `1.05f32` 的最近可表示值是 `1.0499999523…`，差值 `0.0499999523`
  与十进制 `0.05` 天然差 `4.8e-8`——**是样本选错，不是实现对拍漂移**。改法是把样本换成 1/16 网格
  上的 `1.0625`（f32 里精确），断言从"1e-9 近似"升级为 `assert_eq!(err, 0.0625)`，即**加强而不是
  放宽**。这条只动了一个测试，没碰 `check_parity` 的实现。
- 未新增 ADR：两张表换成声明宏不改变任何架构决策（`op` 仍拥有注册表，`train` 仍只消费它）。

### 20.8 实测对照与复核命令

 指标 | 上轮（§19） | 本轮
 --- | ---: | ---:
 `crates/op/src/lib.rs` 行数 | 479 | **431**
 `crates/train/src/store.rs` 行数 | 1531 | **1479**
 `crates/*/src` 行数 | 55 138 | **55 086**
 `crates/*/src` 文件数 | 173 | 173
 `[[gate.debt]]` 条目 | 250 | **250**（`store.rs` 的 `lines` 1529 → 1479）
 用例 | 59 目标 / 1 064 | **59 目标 / 1 074**（+1 本轮：`the_registry_covers_the_generated_table_exactly`；+9 属在途 `cutile` 改动）
 `contracts` / `architecture` | 6 / 8 | **6 / 8**（`contracts` 的扫描形状已改，断言未减）
 `clippy -D warnings` / `cargo doc -D warnings` | 0 / 0 | **0 / 0**
 `cutile doctor --device cpu` | 退出码 0 | **退出码 0**

```bash
cargo test -p llm-op                              # 157 通过：22 个 MODEL 的 __model_check + 注册表闭合
cargo test -p llm-train                           # 50 + 3 通过：KERNELS 与 REGISTRY 的 const 断言
cargo test -p llm-tool --test gate                # ccn≤5、nloc≤60、文件行数、零 panic
cargo test --test architecture --test contracts   # 8 + 6 通过（contracts 读新的表形状）
cargo test --workspace --all-features             # 59 目标 / 1 074 通过 / 0 失败 / 1 忽略
cargo run --bin cutile -- doctor --device cpu     # 退出码 0
```

反向验证（一次性，已恢复）：

```bash
# 把 kernel_table! 里 add_f32 与 attention_f32 两行对调 → 编译期失败
cargo check -p llm-train
# error[E0080]: evaluation panicked: train::Reference 的内核表与 op::REGISTRY 名字或顺序漂移
```

---

## 二十一、配置策略上移到命令行层：`cutile` 只收三个目录（本轮，可维护性 + 高内聚低耦合）

> 用户判据："`crates/cutile/src/settings` 这个目录不要"、"settings 本来就不是这个模块的"、"高内聚低耦合"、
> "代码优化精简"。本轮把这三句话落成一条可复核的边界：**内核只声明输入形状，策略留在能看到 argv 的那一层**。

### 21.1 结论

`crates/cutile/src/settings/`（4 文件 924 行）删除，替换为两处：

 落点 | 行数 | 拥有什么 | 不拥有什么
 --- | ---: | --- | ---
 `crates/cutile/src/dirs.rs` | 64 | `Dirs { cache, work, data }` 三个 `PathBuf` + 三个访问器 + `Dirs::under(root)` | 默认目录、环境变量、`--set`、来源追踪、平台分支（**一条都没有**）
 `crates/cli/src/settings.rs` | 660（含 13 条测试） | `Roots`（平台默认目录 + 相对基准）、`Settings`（三层优先级、来源、`describe`）、`apply_arguments`（从两个二进制收上来的 `--set` argv 扫描） | 任何业务计算

12 个 `cutile` 模块（`cache`/`builtin`/`build`/`cap`/`device`/`doctor`/`gc`/`pipeline`/`probe`/`verify`/`witness`/
`bench`…）与 `src/main.rs`、`src/bin/cutile.rs` 的参数由 `&Settings` 改成 `&Dirs`。

### 21.2 为什么这不是"换个位置放"

判据是"这条知识属于谁"，而不是"哪一层少几行"：

 知识 | 属于谁 | 证据
 --- | --- | ---
 我要一个缓存目录、一个工作目录、一个可选数据根 | 内核（`cutile`） | 它唯一需要的事实，且三个都是路径
 默认目录是 `~/Library/Caches/cutile/cache` 还是 `%LOCALAPPDATA%` | **发布后的二进制** | 这是部署策略；同一个 crate 被测试/服务嵌入时默认值应当不同
 相对路径按哪个目录解析 | **命令行** | 与 `cd` 后执行 shell 的直觉一致，是调用约定
 `--set key=value` 怎么从 argv 里摘出来 | **命令行文法** | 解析 argv 的一层才看得见"哪些参数是前缀"
 每个值的来源要在 `doctor` 里打出来 | **命令行** | 它同时看到 env 与 argv，只有它能把两者排序

改前 `cutile` 同时持有上表五条，后果是可验证的：`Roots::{discover,base}`、三处 `cfg` 平台分支、
`FILE`/`SCHEMA_VER` 常量、键闭集与 `describe` 全在 `crates/cutile/src/settings/**` 里；内核因此**间接**
依赖"进程启动目录"与"用户环境"这两个它无法控制的输入，测试要造 `Settings` 才能拿到一个临时目录。

### 21.3 逐文件改动

 文件 | 改动
 --- | ---
 `crates/cutile/src/settings/{mod,key,paths,source}.rs` | **删除**（4 文件 924 行；其中 `source.rs` 的 `cli_override` 是债务表里唯一指向这里的条目）
 `crates/cutile/src/dirs.rs` | **新增** 64 行：`Dirs::{new,under,cache,work,data}`
 `crates/cutile/src/lib.rs` | `pub mod settings;` → `pub mod dirs;`；crate 文档的"拥有/不拥有"段改写
 `crates/cli/src/settings.rs` | **新增** 660 行：`Roots`、`Settings`、`Key`（`keys!` 宏一行一个）、`Origin`、`apply_arguments`、13 条测试
 `crates/cli/src/lib.rs` | 头文档补"配置为什么在这一层"（ADR 0023），`settings` 进入公开面
 `src/main.rs`、`src/bin/cutile.rs` | `Settings::load()` + `apply_arguments` + 3 行 `kernel_dirs(&Settings) -> Dirs`；`take_overrides` 删除
 `docs/adr/0023-settings-to-cli.md` | 新增（判据表、后果、4 条否决方案、3 条重议条件）
 `tests/architecture.rs` | 所有权表把一行拆成两行：`Dirs` → `cutile`、`Settings`/`Roots` → `cli`（双向校验通过）

### 21.4 用户可见行为对拍（同一二进制）

 场景 | 输出 | 判据
 --- | --- | ---
 默认 | `cache.dir = ~/Library/Caches/cutile/cache ← 内置默认值`（`work.dir` 同） | 与改前逐字相同
 只听环境变量 | `CUTILE_WORK_DIR=/tmp/envwork` → `work.dir = /tmp/envwork ← CUTILE_WORK_DIR（env）` | 与改前逐字相同
 环境 + 命令行 | `-- --set cache.dir=/tmp/clicache` → `cache.dir = /tmp/clicache ← --set cache.dir=/tmp/clicache（cli）` | 命令行仍最高
 标题行 | `[配置] 默认 < 环境变量 < 命令行` | 仍只有三层，且不出现"文件"
 兼容名 | `CUTILE_CACHE` 仍被承认，且 `CUTILE_CACHE_DIR` 优先 | 与改前相同
 退出码 | 未登记键 / 空值 / 缺 `=` → `invalid` → 2 | 与改前相同

### 21.5 用例：19 → 13，逐条给理由

删除的 6 条（都只覆盖随第四层消失的行为）：文件相对基准、文件侧未知键校验、`schema` 校验、
值形状两条（TOML 才有）、"目录里没有配置文件时用默认值"。保留的 5 条判据等价改写为：优先级
（默认 < env < cli）、正式名优先于兼容名、`CUTILE_CACHE` 兼容、相对路径按启动目录解析、
`describe` 逐行"值 ← 来源"且不含"文件"。**新增 2 条**：`--set` 缺值是 `invalid`（argv 扫描的边界，
原来在二进制里无人测）、`Roots::discover` 不锚定仓库目录（防止有人把默认值写回 `CARGO_MANIFEST_DIR`）。

### 21.6 实测

 指标 | 改前 | 改后
 --- | ---: | ---:
 `crates/cutile/src` | 26 文件 / 10 409 行 | **23 文件 / 9 549 行**
 `crates/cli/src` | 2 文件 / 657 行 | **3 文件 / 1 317 行**
 本次搬迁自身 | — | `cutile/src/settings/**` −924、`cutile/src/dirs.rs` +64、`cli/src/settings.rs` +660 ⇒ **净 −200 行 / −3 文件**（推断：全仓 `crates/*/src` 同期还有在途改动，整体行数不作单轮归因）
 成员 / 依赖边 | 25 / 75（`+dev` 11） | **25 / 75（`+dev` 11）**，一条未增
 `[[gate.debt]]` | 251 | **250**（删 `crates/cutile/src/settings/source.rs:cli_override#0`，未新增未抬高）
 用例 | 59 目标 / 1 074 通过 | **59 目标 / 1 068 通过 / 0 失败 / 1 忽略**（−6 见 21.5）
 `gate` / `architecture` / `contracts` | 1 / 8 / 6 | **1 / 8 / 6** 全绿
 `clippy -D warnings` / `cargo doc -D warnings` | 0 / 0 | **0 / 0**
 `cutile doctor --device cpu` | 退出码 0 | **退出码 0**

### 21.7 与 §13、§14 的关系

§13.11 记录的"设置层并入 `cutile`"与 §14.3/§14.5 记录的"四层 → 三层"在**事实**上仍然成立
（配置文件层确实是 ADR 0022 删的，三层确实是剩下的全部），过期的是**位置**：这三层现在住在
`cli::settings`。ADR 0020 与 ADR 0022 都已在文末补"后续（被取代）"段，指明落点由 ADR 0023 迁移。

### 21.8 本轮没碰

- **键表仍是三个路径键**，没有引入 `Kind`/`Value` 值形状系统：三层里唯一的值类型就是路径，
  加一套形状枚举只会把 ADR 0020 删掉的东西重新写一遍。
- **`--set` 只被两个二进制使用**；`fuzz`/`tool`/`tests` 不读设置，因此 `cli` 的公开面扩大没有副作用
  （`fuzz` 早已以 `+dev` 依赖 `cli` 做文法模糊测试）。
- **`Dirs::under` 只服务测试与嵌入方**：它不读环境变量，因此不是"第二个默认值来源"；默认值仍只有
  `Roots::default_cache_dir` / `default_work_dir` 一处。
- §11.4 队列、§15.4 的 `rans` 全局上限、§20.7 列出的未做项全部原样。

### 21.9 复核命令

```bash
cargo test -p llm-cli                              # 13 条设置判据（优先级 / 兼容名 / 基准 / describe）
cargo test -p cutile --lib                         # 12 个模块改用 Dirs 后行为不变
cargo test -p llm-tool --test gate                 # ccn≤5、nloc≤60、文件行数、零 panic
cargo test --test architecture --test contracts    # 所有权表两行、注解↔注册表镜像
cargo test --workspace --all-features              # 59 目标 / 1 068 通过 / 0 失败 / 1 忽略
cargo run --bin cutile -- --set cache.dir=/tmp/x doctor --device cpu   # 三行来源逐行可见
```

---

## 二十二、质量门成为真正的棘轮：限值单一出口、按度量结清、死档位退场（本轮，架构治理）

> 用户判据："架构设计模式顶尖优秀"。本轮挑的是仓库**唯一能约束全仓结构的那件东西**——`crates/tool`
> 的静态质量门。它的模块文档写着"标准是数据，行为是纯函数""存量项只准降"，而实现里三处违反了自己：
> 限值由调用方决定、棘轮按身份而不是按度量结清、一个声明了却从未被读到的档位。

### 22.1 结论

三处缺陷都不在新代码里，而在这套治理机制本身：

 缺陷 | 旧实现 | 后果（可复核）
 --- | --- | ---
 限值策略漏到测量方 | `tests/gate.rs:48,63,76` 把 `gate.max_ccn` / `gate.max_nloc` / `gate.max_file_lines` 塞进 `Over { limit }`，`Call` 的限值写死成字面量 `0` | 阈值虽然来自 TOML，但"哪个度量用哪个数"由测试文件决定；`Over.limit` 是调用方可任意填的字段，"已达标"的判定因此可被测量方改写
 棘轮按身份结清 | 旧 `Gate::stale(&BTreeSet<String>)` 只看 key 在不在 | 同时登记 `ccn` 与 `nloc` 的函数改好一条后，另一条只要还超标，已达标的这条**永远不会被要求删除**（§五 P3-5 已登记）
 "只准降"实际只是"不许涨" | 旧 `Gate::check` 仅 `value > allowed` 失败 | `ccn` 从 47 降到 6 也通过，登记表于是成了"预算"而不是"快照"
 一条从未生效的档位 | `[gate] simd_nloc = 96` + 字段 + "不得小于 nloc_max"校验 | Rust 侧没有任何 SIMD 微核（向量代码在 `prim` 的 C 片段里，由 `cc` 编译、不经本门禁），该字段**零消费者**

### 22.2 现在的形状（设计模式）

 模式 | 落点 | 为什么这样切
 --- | --- | ---
 策略在数据、机制是纯函数 | 阈值与存量债只在 `[gate]`，`Gate` 是它的投影 | 改标准不改代码（原有优点，保留）
 **限值单一出口** | `Gate::limit(Metric)`（`crates/tool/src/gate.rs`），阈值字段转私有 | 测量方只能报"测到多少"；`Over` 不再携带 `limit`，"哪个度量用哪个数"没有第二种说法
 **闭集 + 拒绝未知键** | `unknown_keys` 在 `thresholds` 之前跑；`[gate]` 只允许三个阈值键与 `debt` | 拼错的键不再静默失效；删掉 `simd_nloc` 之后它不会以"没人读的数字"形式复辟
 **棘轮 = 登记值必须等于实测值** | `Gate::check` 按 `Ordering` 三分支：相等通过、变大判"恶化"、变小判"把基线条目改成实测值" | 改善必须在同一提交里记账；登记表是快照，不是预算
 **按（身份, 度量）结清** | `Gate::stale(&BTreeMap<String, BTreeSet<Metric>>)` + `Debt::registered()` | 一个函数的 `ccn` 达标、`nloc` 仍超标时，只要求删掉 `ccn` 那条
 **零 panic 覆盖到 `unreachable!`** | `call::macro_kind` + `Kind::Unreachable` | `todo!`/`unimplemented!` 有 workspace lint 兜底，`unreachable!` 没有——它是 panic 路径

一处顺带的自证：给 `Debt::registered` 配 `pub const ALL: [Self; 4]` 时，仓库自己的
`tests/contracts.rs::closed_vocabularies_are_declared_once` 当场失败（"这些闭集表该用
`impl prof::atom::Closed`"）。于是改成写在用它的函数里的局部表——`tool` 是零业务的静态分析器，
不该为"全部度量"这个内部枚举拉一条到 `prof` 的依赖边。门禁约束到了门禁自己。

### 22.3 新棘轮立刻抓到的真实欠账

接上"登记值必须等于实测"后，第一次运行就有 **31 条登记值高于实测**、**3 条整条或部分豁免已经过期**。
它们不是本轮改坏的：它们在旧规则下**本来就不用记账**。

 条目 | 登记 → 实测 | 说明
 --- | --- | ---
 `crates/cutile/src/verify.rs:verify_one#0` | `nloc 105 → 77`、`ccn 19 → 13` | 两处早就改善了，登记值一直没动
 `crates/cutile/src/backend.rs:cuda_harness#0` | `nloc 109 → 83`、`ccn 20 → 17` | 同上
 `crates/prof/src/atom.rs`（文件） | `lines 1264 → 1199` | 文件缩了 65 行
 `crates/weights/src/dsa.rs`（文件） | `lines 1020 → 983` | 文件缩了 37 行
 `crates/cutile/src/verify.rs:verify_one_cuda#0` | `nloc 64` **删除**（已 ≤60）、`ccn 13 → 10` | 长度已达标，条目不该再在
 `crates/train/src/snapshot/mod.rs`（文件） | `call 1` **删除** | 唯一那处 `expect` 已消失
 `crates/tool/src/gate.rs:check#0` | `ccn 8` **整条删除** | 本轮把 `check` 重写为三分支后实测 3

合计：`[[gate.debt]]` **249 → 248 条**，其中 31 个登记值下调、3 个登记项删除（整条 1 + 度量 2），**没有一条被抬高**。

### 22.4 测量方因此变简单了

`crates/tool/tests/gate.rs` 从"既测量又判定"退化成只测量：

 改前 | 改后
 --- | ---
 4 处 `Over { …, limit: … }` 字面量 + 3 个 `if value > gate.max_*` 分支 | 5 次 `judge(&gate, key, metric, value, …)`，限值与"是否超限"全问 `gate.limit(metric)`
 `live: BTreeSet<String>` | `live: BTreeMap<String, BTreeSet<Metric>>`
 判定与结清分散在两处 | 判定 = `Gate::check`、结清 = `Gate::stale`，两者都不持有阈值

### 22.5 明确不做：ccn 按类分档

§8.2 第 9 项的另一半是"纯逻辑/解析器/分派器用不同 ccn 阈值"。不做，三条理由：

- **它是更粗的机制**：登记表已经给出**逐函数**预算；分档给的是逐类预算，用粗的替掉细的，只会逼出更多"这一类里的例外"。
- **判据不存在**：`ccn` 高低与"是解析器还是分派器"没有可测量的关系；要分档得先有分布证据（例如解析器类的中位 ccn 是纯逻辑类的 3 倍），仓库没有这份数据，凭空给一个数字就是 §22.1 最后一行那种错。
- **可逆**：真需要时按"（度量, 档位）两级表"重开，而不是先放一个空数字——`simd_nloc` 就是反例。

### 22.6 实测

 指标 | 改前 | 改后
 --- | ---: | ---:
 `crates/tool/src` | 未单独测量（工作树未提交） | **9 文件 / 2 240 行**；其中 `gate.rs` **361 → 475**（增在三个新单元测试），`call.rs` 114
 `[[gate.debt]]` | 249 条 | **248 条**（31 个登记值下调、3 个登记项删除、0 新增、0 抬高）
 `[gate]` 键 | 4（含 `simd_nloc`） | **3 + `debt`**，闭集
 用例 | 59 目标 / 1 068 通过 | **59 目标 / 1 071 通过 / 0 失败 / 1 忽略**（+4 新增、−1 随死档位删除）
 `gate` / `architecture` / `contracts` | 1 / 8 / 6 | **1 / 8 / 6** 全绿
 `clippy -D warnings` / `cargo doc -D warnings` | 0 / 0 | **0 / 0**

### 22.7 复核命令

```bash
cargo test -p llm-tool                      # 36 条：三分支棘轮、按度量结清、未知键、unreachable!
cargo test -p llm-tool --test gate          # 全仓 ccn/nloc/lines/call + 闭集与文档链接
cargo test --test contracts --test architecture
cargo test --workspace --all-features       # 59 目标 / 1 071 通过 / 0 失败
```

### 22.8 本轮没碰

- **`[gate]` 之外的门禁盲区**（P1-7 的其余四处）：`macro_rules!` 体内的条件不计 ccn；`crates/*/build.rs`
  与 `crates/*/tests/*.rs` 不在扫描集；`dup` 只数单文件行数；`docs` 只扫 `docs/` 顶层与成员 README。
- **两套防假绿下限**（`crates/tool/tests/gate.rs` 的 `MIN_SOURCES = 100` 与 `tests/contracts.rs` 的 `70`）未统一。
- §11.4 队列（`weights` 零引用模块、`storage` 布局并入 `tensor`、`snapshot` 拆文件等）与 §15.4 的
  `rans` 全局上限原样。

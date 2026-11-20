# llm 全仓代码审计（2026-11-20）

> 审计视角：Rust 语言/编译器、大模型推理、Linux/内核、架构与设计模式、质量与测试。
> 证据口径：每条结论带 `文件:行` 或可复查命令。快照 = `HEAD 73d27b6c` + 未提交工作树。
> 对标标准：[Rust API Guidelines 54 条](https://crates.io/crates/api-guidelines)、[Clippy lint 分组](https://doc.rust-lang.org/clippy/)、Burn / tokio / serde / Candle 的工程范式。

## 0. 方法与边界（先把话说清楚）

- **全量**：25 个 crate、217 个 `src/*.rs`、98,783 原始行全部进了静态指标扫描（行数 / 宏 / unsafe / 转换 / 文档 / trait / 测试 / 依赖 / profile）。
- **逐段精读**：12 个关键文件（`op/unary`、`op/gemv`、`op/attention`、`op/einsum`、`engine/kv`、`engine/codegen`、`engine/metal`、`train/optimizer`、`weights/ptq`、`serve/completion`、`annotation/syntax`、`dtype`）。
- **抽样**：其余 205 文件按六维抽样，每维 ≥15 点。
- **不是"逐字符"**：9.8 万行逐字符人工审计需要 200+ 小时，任何自动化都做不到。本报告给的是「全量指标 + 关键路径精读 + 抽样」，每条可复查，不编造。
- **快照污染**：工作树有 6 个未提交文件（`nn/src/{engram,lib}.rs`、`serve/src/{assembly,plan}.rs`、`weights/src/{lib,ptq}.rs`），当前 `cargo test --workspace` 编译失败（`Scheme.cpb`、`nn::MemoryBlock`）。这是 P0-1，审计数字按当前树取，修复后需复测。

## 1. 总评分

| 维度 | 分数 | 一句话 |
|---|---|---|
| 架构 | 8.0 / 10 | 分层/所有权/闭集三套门禁行业级，缺一个统一抽象基座 |
| 设计模式 | 6.5 / 10 | 局部有顶级范式，全局三种范式并存、宏散装 |
| 规范 | 7.0 / 10 | 文档密度极高、clippy 只开一半，524 处 cast 无检查 |
| 质量 | 7.5 / 10 | unsafe 边界干净、门禁强，当前树不可编译 |
| 性能 | 6.5 / 10 | 热路径写法对，inline/simd/cold/autotune 缺 |
| 测试 | 7.0 / 10 | 288 op + fuzz + coverage 下限，缺属性测试 |

**加权 7.1 / 10：上游工程，离"顶级"差两刀（基座 + 严格化）。**

## 2. 六维详情

### 2.1 架构（8.0）

**好**：唯一依赖规则（依赖只能指向更低层）由 `crates/testkit/tests/architecture.rs` 对照 `cargo metadata` 自动执行；所有权表双向校验（表里能找到 `pub` 定义、源码每个 `pub` 型必须入表）；闭集位集 `Set<E, WORDS>` 编译期拦第 65 个原语；25 crate 边界清楚。

**问题（P0-2 无统一抽象基座）**：全仓 44 个 `pub trait` 散在 12 个 crate——`op` 10、`train` 7、`serve` 3、`cutile` 3、`wire` 2、`weights` 2、`tensor` 2、`engine` 2、`dtype` 2、`autograd` 2、`tokenizer` 1、`storage` 1。没有一个"贯穿全仓"的算子/dtype/后端基座：`op::Unary`、`op::Gemm`、`op::Format` 是局部最优，`train`/`serve`/`engine` 各自另写一套。对比 Burn 的 `Backend` trait 基座 + `Autodiff`/`Fusion` 装饰器，这一层是缺的。

### 2.2 设计模式（6.5）

**好（这是全仓最漂亮的样板）**：`crates/op/src/unary.rs:12-42` 的 `trait Unary`——把变化点（`EXPR`/`OPERANDS`/`cache`/`apply`/`input`）收进 trait，`cost<U>`/`sample<U>`/`reference<U>`/`forward<U>`（:60-108）泛型单态化，`unary_f32!` 宏（:110-120）只做装配。这就是 serde derive + trait 的级别，方向是对的。

**问题**：
- **范式三分**：trait 泛型（`op`）/ 手写单文件（`model`/`decl`/`sig`/`tpl`）/ 纯数据（`prim`/`prof`）。同一概念在不同 crate 用不同形态表达。
- **宏散装**：65 处 `macro_rules!` 散在 31 个文件（`train/kernel.rs` 6、`serve/kernels.rs` 5、`op/elementwise.rs` 4、`op/unary.rs` 3、`op/reduce.rs` 3、`op/gemm.rs` 3……），没有统一装配层。读一个算子要跳三四层宏。
- **宏内文档不可读**：`#[doc = concat!(...)]`（`elementwise.rs:132`、`gemm.rs:565`）让源码里看不到实际文字，只能靠 rustdoc 展开。

### 2.3 规范（7.0）

**好**：`///` 12969 行 vs `pub fn` 1882 个（≈1:7 文档密度），307 处 `# Errors`/`# Panics`/`# Safety`；clippy `all` + `cargo` warn，`dbg_macro`/`todo`/`unimplemented` deny，`missing_docs` warn；`profile.release` 是 `lto = "thin"` + `codegen-units = 1` + `strip = "debuginfo"`，发布配置到位。

**问题**：
- **P1-1 pedantic 未开**：`Cargo.toml` 的 `[workspace.lints.clippy]` 只有 `all` + `cargo`。全仓 **524 处 `as` 转换**没有任何 `cast_possible_truncation` / `cast_precision_loss` 检查——对一个数值/推理框架，这是最容易藏 bug 的盲区（`u64→u32` 截断、`i64→usize` 精度）。
- **P2-3 命名微瑕**：`get_impl`（`prim/src/lib.rs:246`）、`get_suffix`（`weights/src/bin.rs:559`）违反 C-GETTER；`get_mut`/`get_u8` 因返回 `Option` 合规范，不算违规。
- **P1-4 文档漂移**：`README.md:120` 说"unsafe 收敛到单文件"，实际有**两个** unsafe 岛——`storage/src/lib.rs:237`（mmap）和 `engine/src/metal.rs:26`（`objc_msgSend` FFI）。不是散布，但 README 的"唯一"表述要改。

### 2.4 质量（7.5）

**好**：unsafe 只有 2 个模块级 `#![allow(unsafe_code)]` 岛（26 个 `unsafe {}` + 4 个 `unsafe fn`）；生产代码裸 `unwrap/expect` 被门禁 `Call = 0` 守住了（抽查的 unwrap 全在 `#[cfg(test)]`）；jscpd 0.28%；ccn ≤ 5 / nloc ≤ 60 / file ≤ 1200；coverage 有 4 个 crate 的硬下限（cutile 75/70、wire 68/64、dtype 88/88、weights 70/64）。

**问题**：
- **P0-1 当前树不可编译**（见第 0 节）。
- **P1-2 文件粒度失控**：112 个文件 ≥400 行、60 个 ≥600、**22 个 ≥900**（占 11% src），其中 `op/einsum` 1153、`op/attention` 1153、`op/gemv` 1109 是人为合并的结果。1000+ 行单文件让 review 粒度、职责边界、IDE 导航一起变糊。
- **待验**：宏生成代码是否真的被 nloc/ccn 门禁覆盖——门禁按 strip_tests 后的 token 统计，宏展开前的 token 形态可能绕过。

### 2.5 性能（6.5）

**好**：热路径写法规范——`gemv.rs:593` 的 `dot` 用 `w.iter()`、`forward_scaled_rows:427` 用 `out.iter_mut().enumerate()`、`chunks`/`zip` 无下标边界检查、热路径无 `clone`；CHANGELOG 有实测：量化 GEMM i2 9.8× / i4 9.6× / i8 2.8×，E10 CV 1.23%，E12 0.351 ns O(1)。

**问题**：
- **P1-5 编译器提示缺**：`#[inline]` 仅 19 处 / 98k 行（跨 crate 热路径默认不内联），`#[cold]` 0 处（`format!`/`Error::new` 的错误路径污染热路径 icache），`#[repr]` 仅 7 处。
- **无显式 SIMD / `target_feature` / autotune**：靠迭代器自动向量化，对 memory-bound GEMV 够，对 compute-bound GEMM 不够。
- **无 PGO 证据**（`profile` 有 LTO，但没看到 PGO 流程）。

### 2.6 测试（7.0）

**好**：39 个测试文件、op 288 用例、fuzz crate（`corpus`/`src`/`tests`）、CI 走 `cargo llvm-cov nextest` + 4 个 crate 覆盖率下限。

**问题**：
- **P2-1 无属性测试**：没有 `proptest`/`quickcheck`。数值算子最该有性质不变量（交换律、结合律、往返、单调、边界饱和），现在全是手写定点用例。
- **P2-2 coverage 下限只覆盖 4 个 crate**，`op`/`nn`/`train`/`serve`/`engine` 无下限。
- **178 个 `mod tests` 内联在 `src`**：响应"含测试文件"的诉求，但牺牲测试隔离与编译并行。

## 3. 问题清单

| 级别 | ID | 位置 | 危害 | 修复方向 |
|---|---|---|---|---|
| P0 | P0-1 | `nn`/`serve`/`weights` 未提交改动 | 全仓不可编译，门禁全部失效 | 先收敛 `Scheme.cpb`、`nn::MemoryBlock` |
| P0 | P0-2 | 44 trait / 12 crate | 变化点各自为政，文件膨胀的根源 | 补一层 `Semantics`/`Kernel` 基座 |
| P1 | P1-1 | `Cargo.toml` lints | 524 处 cast 无截断/精度检查 | cherry-pick `cast_*`、`missing_errors_doc`、`must_use_candidate` |
| P1 | P1-2 | 22 个 ≥900 行文件 | 可读性/评审粒度下降 | 按职责拆回 300-600 |
| P1 | P1-3 | 65 处 `macro_rules!` / 31 文件 | 装配样板分散，新人成本高 | 收成 1-2 个 derive/注解宏 + trait |
| P1 | P1-4 | `README.md:120` | 文档与实现不符 | 改为"两个 unsafe 岛"并登记 |
| P1 | P1-5 | 热路径 | 跨 crate 不内联、错误路径占 icache | `#[inline]`/`#[cold]`/SIMD |
| P2 | P2-1 | 测试体系 | 数值性质无覆盖 | 引入 `proptest` |
| P2 | P2-2 | CI coverage | 核心 crate 无下限 | 扩到 op/nn/train/serve/engine |
| P2 | P2-3 | `prim`/`weights` | C-GETTER 微瑕 | 改 `get_impl`→`implementation`、`get_suffix`→`entry_by_suffix` |
| P2 | P2-4 | 宏内 `#[doc]` | 源码不可读 | 评估保留或改手写 doc |

## 4. 对标业界标准

- **Rust API Guidelines 54 条**（[crates.io/api-guidelines](https://crates.io/crates/api-guidelines)）：C-CASE ✅、C-CONV ✅（38 个 `as_`/`to_`/`into_`）、C-GETTER 2 处微瑕、C-GOOD-ERR ✅（`diagnostic::Error` + `Display`）、C-FAILURE ✅（307 段 `# Errors/# Panics/# Safety`）、C-COMMON-TRAITS 基本 ✅、C-QUESTION-MARK（doc 里无 `unwrap`）。
- **Clippy 分组**（[doc.rust-lang.org/clippy](https://doc.rust-lang.org/clippy/)）：`correctness`/`suspicious`/`style`/`perf`/`complexity` 都在 `all` 内 ✅；**`pedantic` ❌ 未开**；`restriction` 的关键项（`unwrap_used`/`expect_used`/`panic`）用自定义门禁（Call=0）替代 ✅，这是比 lint 更强的做法。
- **顶级项目范式**：tokio/serde 的「trait 抽象 + derive 装配」= llm 的 `unary.rs` 路线 ✅；Burn 的「Backend trait 基座 + 装饰器」= llm 缺的那层 ❌；Candle 的 device 抽象同理。**llm 缺的是"基座"，不是"细节"。**

## 5. 与 Burn 的差距映射（只列工程规范/性能维度）

| 维度 | llm | Burn | 裁决 |
|---|---|---|---|
| 架构治理（可执行门禁） | ✅ 有 | 无同类 | llm 胜 |
| 统一抽象基座 | ❌ 缺 | ✅ `Backend` trait | Burn 胜 |
| 宏/装配层 | 65 处散 31 文件 | 集中 derive | Burn 胜 |
| 属性测试 | ❌ 无 | ✅ `proptest` | Burn 胜 |
| 性能注解 | 19 `#[inline]`、0 `#[cold]` | 更多 `#[inline]`/`target_feature`/tune | Burn 胜 |
| 文档密度 | ✅ ≈1:7、307 段 | 成书 + API doc | 平 |
| unsafe 边界 | ✅ 2 岛模块级 | 边界 crate 含 FFI | llm 略胜 |
| clippy 严格度 | 只开 `all` | 也未全开 pedantic | 平 |

**结论：llm 在"治理"上已超 Burn，在"基座 + 数值测试 + 性能标注"上落后 Burn，不在"分层"落后。**

## 6. 达到"顶级"的刀（按收益排序）

1. **补抽象基座（P0-2）**：把 `op` 的 `Unary`/`Gemm`/`Format` 上升为一层 `Semantics`/`Kernel`，`train`/`serve`/`engine` 的 trait 收进同一族。这一刀同时解决"文件膨胀"和"范式三分"。
2. **收宏（P1-3）**：65 处 `macro_rules!` → 1-2 个 derive/注解宏 + trait。减样板、统一命名、提升可读性，一次三得。
3. **严格化（P1-1 + P2-1）**：开 `cast_*` + `missing_errors_doc` + `must_use_candidate`；数值算子引入 `proptest` 性质测试（交换律/结合律/往返/单调/饱和）。
4. **拆文件（P1-2）**：22 个 ≥900 行按职责下沉到 300-600；`op` 三个人为合并的文件重组。
5. **热路径注解（P1-5）**：`#[inline]`/`#[cold]`/显式 SIMD，先做 GEMV/attention/页表三条路径。

**保留不动的优势**：三套门禁、unsafe 2 岛、文档密度、coverage 下限、LTO profile。这些是已经兑现的，不要为了"精简"拆掉。

## 7. 附录：指标快照

| 指标 | 值 |
|---|---|
| crate / src 文件 / 原始行 | 25 / 217 / 98,783 |
| 文件粒度 ≥400 / ≥600 / ≥900 / ≥1200 | 112 / 60 / 22 / 0（ptq 曾 1210） |
| `macro_rules!` 处 / 文件 | 65 / 31 |
| `pub trait` / `impl<` | 44 / 93 |
| `pub fn` 数 / `///` 行 | 1,882 / 12,969 |
| `# Errors·Panics·Safety` 段 | 307 |
| `as` 转换 | 524 |
| `#[inline]` / `#[must_use]` / `#[repr]` / `#[cold]` | 19 / 258 / 7 / 0 |
| `unsafe {}` / `unsafe fn` / 岛 | 26 / 4 / 2（storage、engine/metal） |
| 内联 `mod tests` | 178 |
| 依赖（Cargo.lock） | 121 |
| jscpd | 0.28% |
| op 测试 | 288 + 2 doctest |

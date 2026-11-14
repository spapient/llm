# `llm-annotation` S++ 升级证据报告

- 对象：`crates/annotation`（包名 `llm-annotation`，`[lib] proc-macro = true`），三道属性宏 `#[layer]` / `#[frontend]` / `#[model]`。
- 时点：2026-11-14，当前工作树；命令在本机 macOS（aarch64）实跑，数值可复现。
- 口径：**已确认** = 本次直接执行命令或读源码得到；**边界** = 工具/语言限制导致无法在本 crate 内实测，如实标注，不拿推断冒充测量。
- 范围纪律：工作区其他 crate 存在用户未提交改动，本轮**未触碰、未回滚**；报告中与 annotation 无关的失败单独标出。

## 一、总览：六维对标与评级

| 维度 | 对标国际标准工具 / 惯例 | 基线 | 本轮实测 | 结论 |
| --- | --- | --- | --- | --- |
| 类聚与模块职责 | 分层架构校验（`tests/architecture.rs`）、所有权表、依赖树 `cargo tree` | 文法与适配耦合在同一文件思路里，依赖边仅靠文档约束 | 文法/渲染/适配三层分离；依赖树仅 `llm-dtype`；架构门 8/8、契约门 6/6 | S++ |
| 证据 / 判据纪律 | 闭集单源、负例断言、消息可行动性（对标 clippy/diagnostic 文案） | 错误分支覆盖偏薄，部分判据只在注释里 | 每条拒绝路径有位置 + 可行动消息断言；闭集由契约测试核对 | S++ |
| 静态强制门禁 | clippy `-D warnings`、rustfmt、cargo-deny、cargo-machete、复杂度/裸调用门 | lint 仅 `deny(unsafe_code)`+`missing_docs`；无容器/正则/依赖面本地门 | `forbid(unsafe_code)` + 三条 `deny`；新增 `tests/audit.rs` 静态门；工具门零豁免 | S++ |
| 测试质量 | nextest、llvm-cov、确定性回归（对标 proptest/quickcheck 思路） | 28 用例；syntax 覆盖 89.45% | 39 用例全绿；syntax 97.16%、codegen 97.01%、fixture 97.33%；128 用例确定性 sweep | S++ |
| 性能工程 | 复杂度门（lizard 口径 ccn/nloc）、依赖最小化、无重结构静态审计 | 未审计容器/正则；无本地性能门 | 单趟线性扫描；零关联容器/正则/惰性初始化；ccn≤5、nloc≤54；运行期依赖仅 dtype | S++（静态门禁口径；墙钟基准受 proc-macro 边界限制，见 §7.3） |
| 广度与生产成熟度 | README/元数据、跨平台纯净度、下游编译兼容 | 无 README；依赖面未在本地固化 | 补 README；零 path/fs/env/process/thread；下游 `nn/op/cutile` 编译通过 | S++ |

## 二、规模与前后对照数据

### 2.1 源码规模（tokei）

| 指标 | 基线 | 现值 | 说明 |
| --- | ---: | ---: | --- |
| Rust 文件 | 5 | 6 | `src` 5 个 + 新增 `tests/audit.rs` |
| 代码行（`src`） | 1 825 | 1 910 | 含证据测试模块 `syntax/tests.rs`；净增主要是负例与 sweep |
| 注释行（`src`） | 327 | 343 | |
| 空行（`src`） | 224 | 240 | |

生产源文件行数（`wc -l`，门禁上限 899）：

| 文件 | 行数 | 角色 |
| --- | ---: | --- |
| `src/codegen.rs` | 843 | 三类渲染 + 骨架表（基线最大文件 869，降 26 行） |
| `src/syntax.rs` | 817 | 文法 / 闭集表 / 游标 |
| `src/lib.rs` | 383 | 入口宏 + `Streams` 依赖倒置 + 编排 + 适配 |
| `src/fixture.rs` | 112 | `#[cfg(test)]` 词法夹具 |
| `src/syntax/tests.rs` | 344 | 文法负例 + 确定性 sweep |
| `tests/audit.rs` | 90 | 性能 / 供应链静态门 |

### 2.2 测试与覆盖（nextest / llvm-cov）

| 指标 | 基线 | 现值 |
| --- | ---: | ---: |
| 用例通过 | 28 / 28 | 39 / 39（37 单测 + 2 门禁集成测试） |
| `syntax.rs` 行覆盖 | 89.45% | **97.16%** |
| `codegen.rs` 行覆盖 | 96.58% | **97.01%** |
| `fixture.rs` 行覆盖 | 97.40% | 97.33% |
| `lib.rs` 行覆盖 | 59.76% | 56.71%（proc 适配边界，见 §6.3） |
| 包内总体行覆盖 | — | 91.57%（1097/1198） |

### 2.3 静态门禁实测

| 门禁 | 工具 | 基线 | 现值 |
| --- | --- | --- | --- |
| 格式 | `rustfmt --check` | 干净 | 干净 |
| 静态分析 | `clippy --all-targets -D warnings` | 干净 | 干净 |
| 供应链 | `cargo deny check` | ok | advisories/bans/licenses/sources 全 ok |
| 未用依赖 | `cargo machete` | 干净 | 干净 |
| 重复代码（annotation 源码） | jscpd | 无克隆 | **0 精确克隆，0.00% 重复行** |
| 全仓重复率 | jscpd | 1.43% 行 / 1.65% token | 1.40% 行 / 1.63% token（阈值 1.5% 行） |
| 圈复杂度 | lizard 口径 `ccn`，上限 5 | 峰值 5 | 峰值 5 |
| 函数长度 | `nloc`，上限 60 | 峰值 54（`render_model_check`） | 峰值 54（同函数） |
| 文件行数 | 上限 899 | 最大 869 | 最大 843 |
| 裸 panic 形态 | `unwrap/expect/panic!/unreachable` = 0 | 0 | 0 |
| 运行期依赖 | `cargo tree` | 仅 `dtype` | 仅 `llm-dtype` |

## 三、维度一：类聚与模块职责

### 3.1 三层职责（单一事实源，互不越界）

| 层 | 文件 | 职责 | 不允许出现 |
| --- | --- | --- | --- |
| 文法 | `src/syntax.rs` | token 模型、键表驱动解析、全部错误路径、闭集表 | 触碰 `proc_macro`、字符串渲染 |
| 渲染 | `src/codegen.rs` | 声明 → `impl` / 常量 / 校验测试的文本拼接 | 解析 token、持有运行期状态 |
| 适配 | `src/lib.rs` | `TokenStream` ↔ token 模型、错误映射回 span；`Streams` 依赖倒置 | 写文法规则、持有领域类型 |
| 夹具 | `src/fixture.rs`（`cfg(test)`） | 注解文本 → token 模型的最小词法器 | 进入生产构建 |

### 3.2 证据

- **依赖边**：`cargo tree -p llm-annotation` 唯一出边为 `llm-dtype`；不引入 `syn` / `quote` / `proc-macro2`，也不拉 `diagnostic` / `prof` 依赖边。
- **架构门**：`cargo test --test architecture` 8/8 通过（层表、边、方向、所有权双向核对）。
- **契约门**：`cargo test --test contracts` 6/6 通过（词汇表等式、注册表镜像、前后端标注齐全）。
- **代码收口**：删除只打印不断言的调试模块 `dump_golden`（约 57 行死测试代码）；删除冗余的 `expand` 泛型包装，入口直接调 `expand_with`。

## 四、维度二：证据 / 判据纪律

### 4.1 闭集只有一份事实源

| 闭集 | 数量 | 位置 | 对拍对象 |
| --- | ---: | --- | --- |
| `LANGUAGES` | 9 | `syntax.rs` | `prof::Emit::ALL`，契约测试核对 |
| `DTYPES` | 3 | `syntax.rs` | 有阈值的 `DType` 集合 |
| `SKELETONS` | 12 | `syntax.rs` | `codegen.rs` 骨架分支表，逐行对应 |
| `GEMM_SKELETON_NAMES` | 3 | `syntax.rs` | GEMV 三个格式骨架 |
| `KEYWORDS` / `KEYS` | 10 / 4 | `syntax.rs` | 文法保留词与允许键 |

### 4.2 每条拒绝都可执行、可定位

本轮在 `syntax/tests.rs` 用手工构造 token（`ident` / `literal` / `punct` 工厂）补齐此前未测的错误分支：

- 键位非标识符；字符串 / 整数拒绝非字面量与超出 `usize` 的值；
- type 别名缺名 / 非 `=` / 非 `<`；struct / enum 后缺名；
- `Value` 访问器在非自身变体上回落空值；名字拒绝空白与控制字符；
- 四种括号各自的 `describe` 文案。

`codegen.rs` 补齐：`capitalised` 空串、未映射 dtype（用真未知名 `"f96"`；`"f64"` 在 `DType` 内不算未映射）、未列骨架 `"ghost"` 的 `compile_error!` 回落。

### 4.3 判据不是注释

- 解析失败 / 生成失败都保留被标注项，避免次生错误埋掉真因（用例 `a_rejected_annotation_keeps_the_annotated_item`）。
- 拒绝消息非空且包含出错的键 / 词汇名（用例断言 `error.message.contains(..)`）。
- 骨架表一行一条；漏一行先在本 crate 测试里红掉，而不是到下游编译期才暴露。

## 五、维度三：静态强制门禁

### 5.1 编译器与 linter 层

- `#![forbid(unsafe_code)]`（原 `deny` 提到 `forbid`，连 unsafe 块都不允许）。
- `#![deny(missing_docs)]`、`#![deny(missing_debug_implementations)]`。
- `cargo clippy --all-targets -- -D warnings` 干净；`cargo fmt --check` 干净。

### 5.2 仓库工具门（`crates/tool`，`cargo test` 不可绕过）

- 阈值：`ccn ≤ 5`、`nloc ≤ 60`、文件 ≤ 899 行、生产源码裸调用形态 = 0。
- 棘轮：新增身份或恶化即失败；存量基线已清零（`gate.debt` 为空表）。
- annotation 在同一次扫描中**未被列入任何超标项**。

### 5.3 本轮新增的本地静态门 `tests/audit.rs`

两条断言把架构约束固化成可执行测试：

1. **生产源码无重型结构**：禁止 `HashMap/HashSet/BTreeMap/BTreeSet/regex/Regex/lazy_static/once_cell/DashMap/phf`（游标单趟扫描，查名字只在闭集数组上线性包含）。
2. **运行期依赖面只许 `dtype`**：解析 `Cargo.toml` 的 `[dependencies]`，出现其他依赖即失败。

### 5.4 供应链

- `cargo deny check`：advisories / bans / licenses / sources 全部 ok。
- `cargo machete`：无未用依赖。
- `cargo tree`：annotation → `llm-dtype` 一条边。

## 六、维度四：测试质量

### 6.1 结果

`cargo nextest run -p llm-annotation --all-features`：**39 passed, 0 failed, 0 skipped**（37 单测 + 2 条 `audit` 集成测试）。

### 6.2 确定性 sweep（自包含模糊回归）

注解层不允许拉 `llm-fuzz` 的 diagnostic 契约，`proptest` 也未进 lock；因此在 crate 内自包含实现与 fuzz 同一组判据：

- 随机源：xorshift `Rng`，固定种子 `0x9e37_79b9_7f4a_7c15`，无随机设备、无全局状态；同一 `(seed, case)` 在任何机器得到同一串文本。
- 规模：**128 个确定性用例 × 三道文法**（`parse` / `parse_frontend` / `parse_model`）。
- 三条契约（`sweep_one`）：
  1. `catch_unwind` 下不 panic；
  2. 同一输入两次解析结论 `PartialEq` 相等；
  3. 被拒绝时消息 `trim` 后非空，不静默拒绝。

### 6.3 覆盖未覆盖项的诚实归因

| 未覆盖位置 | 性质 | 处置 |
| --- | --- | --- |
| `codegen.rs:264–278`、`syntax.rs:147–159` | `const fn` 函数体，编译期求值，llvm-cov 运行期不计 | 工件，非业务缺口；不强行补运行期调用 |
| `fixture.rs:28,76` | 防御性不可达 `pop` 回落分支 | 夹具保护分支 |
| `lib.rs` 入口宏、`TokenStream` 的 `Streams` 实现、`convert_one` / `bracket` | 只能在编译器展开期运行的 proc 边界；纯编排由 `Text` 字符串替身覆盖 | 由下游 `nn/op/cutile` 编译时间接覆盖 |

### 6.4 诊断完整性

错误消息在 `TokenStream` ↔ 字符串替身之间往返后保持不变（用例 `the_error_message_survives_the_round_trip`）；红线位置指向出错 token，而非笼统指在被标注类型上。

## 七、维度五：性能工程

### 7.1 复杂度与数据结构证据

- 文法是游标**单趟线性扫描**：不回溯、不建关联容器、不上正则。
- 列表类提取（轴、算子名）`Vec::with_capacity` 预分配；闭集匹配走数组线性包含。
- `tests/audit.rs` 静态守住「无容器 / 无正则 / 无惰性初始化」。
- 峰值复杂度：`ccn = 5`（贴阈值，位于测试夹具 `item`）、`nloc = 54`。

### 7.2 零依赖与编译期边界

- 运行期依赖仅 `dtype`；注解在编译期运行，输入只是一小段属性。
- 生产源码无 `std::path` / `std::fs` / `env::` / `process::` / `thread::`，无平台泄漏。

### 7.3 为什么没有 criterion 墙钟基准（边界，不是遗漏）

已实测走不通的三条路，本轮均未重复尝试：

1. **过程宏 crate 无法导出非宏条目**：加 `#[doc(hidden)] pub mod` 触发 E0433（proc-macro crate 当前只能导出 proc-macro 项），外部基准目标够不到内部纯函数。
2. **同包 bench 目标实测 `cfg(test) = true`**：探针二进制输出 `test=true bench=false`；`#[path]` 挂接 `src` 模块会激活外部 `mod tests;`，报 E0583 文件未找到。
3. **外部普通 crate 够不到 `pub(crate)` 内部函数**。

拆出独立普通 rlib 核心 crate 的方案被否决：`tests/architecture.rs` 用 cargo metadata 双向核对层表 / 依赖边 / 所有权表，新增成员必须同步文档行、边、所有权与 ADR，属于跨层架构决策，超出本轮 annotation 范围，需另行拍板。

> 可替代证据：复杂度门（ccn/nloc）、无重结构静态审计、依赖面门禁、下游编译时间接覆盖。墙钟基准在核心 rlib 拆分后即可挂到 `benches` 集中测量容器。

## 八、维度六：广度与生产成熟度

### 8.1 文档与元数据

- 新增 `README.md`：三道注解表、三层结构、判据证据、性能边界、发布状态；相对链接经文档门禁核对有效。
- `description` 说明用途；`publish = false`（尚未作出 MSRV / semver 公开承诺，如实标注）。
- 继承 workspace：edition 2024、MSRV 1.100、license、repository。

### 8.2 跨平台与兼容性

- 解析只依赖 `char` / `str` 操作；无路径、文件、环境变量、进程、线程、平台特判。
- unsafe 由 `forbid(unsafe_code)` 静态禁止；无裸指针、无 FFI。
- 生成代码形状（`MODEL` 常量、`CHECK`、`#[cfg(test)]` 校验模块、`__layer_` 前缀标识符）保持不变；下游编译兼容：

```
cargo check -p llm-nn -p llm-op -p cutile   # Finished，无 annotation 相关错误
```

### 8.3 确定性

同一注解两次展开得到相同文本；不维护运行期注册表（注册表仍是 `op::REGISTRY` 唯一事实源）；无隐藏全局状态。

## 九、工作区状态与隔离边界

- 静态门 `cargo test -p llm-tool --test gate` **1/1 通过**，annotation 未被列入任何超标项；扫描中曾出现的 `crates/cutile/src/cache.rs` 行数波动来自其他 crate 的未提交改动，与 annotation 无关，本轮未改、未回滚该文件（复核时已回到阈值内）。
- 工作区其余 dirty crate（`engine/op/prof/prim/weights/cli/deny.toml` 等）保持原状，本轮改动仅限 `crates/annotation/` 与本报告。

## 十、复现命令

```bash
# 格式 / 静态分析
cargo fmt -p llm-annotation -- --check
cargo clippy -p llm-annotation --all-targets -- -D warnings

# 测试（nextest 与内置 harness 两口径）
cargo nextest run -p llm-annotation --all-features
cargo test -p llm-annotation --all-features

# 覆盖
cargo llvm-cov -p llm-annotation --all-features --lcov --output-path /tmp/ann_final.lcov

# 供应链 / 未用依赖 / 重复率
cargo deny check
cargo machete
npx jscpd --format rust --silent crates/annotation/src

# 仓库门禁
cargo test -p llm-tool --test gate
cargo test --test contracts
cargo test --test architecture

# 下游兼容
cargo check -p llm-nn -p llm-op -p cutile

# 依赖面
cargo tree -p llm-annotation
```

## 十一、遗留与建议（需后续决策，不在本轮实施）

1. **核心 rlib 拆分**：把纯文法 / 渲染从 proc-macro 适配层移出为普通 crate，使 criterion 墙钟基准可落地；需同步架构文档、依赖边、所有权表与 ADR。
2. **miri**：本机 miri 组件未安装；生产源码已由 `forbid(unsafe_code)` 禁掉 unsafe。核心拆分后可对纯函数层跑 miri。
3. **fuzz 接边**：`crates/fuzz` 目前不依赖 annotation；本轮 sweep 是 crate 内确定性等价物，proptest 进入 lock 后可升级为属性化策略。
4. **`ccn=5` 余量**：峰值贴阈值 5，位于测试夹具 `item`；生产路径函数均在阈值 5 以内，可在后续重构中再降一档留余量。

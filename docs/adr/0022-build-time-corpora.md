# 0022：语料改在构建期解析，运行期删除 TOML 解析器

- 状态：已采纳
- 日期：2026-11-14
- 取代：ADR 0021 的"语法层留在 `diagnostic::toml`"这一半（成员归属不变，解析器本身删除）

## 背景

ADR 0021 把 585 行受限 TOML 解析器并进了 `diagnostic`，理由是那条依赖边本来就冗余：六个消费者
（`op` 的 `ops/*.toml`、`prof` 的 `devices/*.toml`、`tool` 的 `lints/vocabulary.toml`、`cutile` 的
设置层、根包的契约测试、`fuzz` 的解析器遍历）全都已经依赖 `diagnostic`。

但那次裁决只解决了"解析器住在谁家"，没有回答"运行期为什么要解析"。事实是：

- 三份**仓库语料**（`ops/`、`devices/`、`lints/vocabulary.toml`）是编译期资产，随二进制发布，
  运行期读它们没有任何收益——却要为此在产物里带一个 585 行的解析器；
- 唯一真正属于**用户**的输入是运行时配置文件，而它承载的只有三个键（`cache.dir` / `work.dir` /
  `data.root`），值形状只有"路径"一种；同一组值已经可以由环境变量与 `--set` 完整表达。

因此真正的选择不是"解析器住哪"，而是"运行期还需不需要解析器"。

## 决策

**运行期不再有任何 TOML 解析。** 三份语料在构建期解析，用户配置只剩三层。

### 1. 数据树归 `diagnostic`（`data` 子模块）

`Table` / `Value` 与它们的读取器搬到 `crates/diagnostic/src/data.rs`，并新增两组入口：

- 构造（给生成的代码用）：`Table::from_pairs`、`Value::{text, int, real, flag, array, table}`；
- 改写（给测试用）：`Table::{set, push, remove, at, at_mut, get_mut, table_mut, array_mut}`，
  路径语法 `variants[0].order`。

"把内置声明改成非法的一份"因此从字符串拼接变成表改写：类型仍然由编译器检查，测试不再依赖
TOML 的书写形式（键顺序、引号、缩进）。

### 2. 三张生成表，三个 `build.rs`

| crate | 输入 | 生成物 | 暴露 |
|---|---|---|---|
| `op` | `crates/op/ops/*.toml` | `$OUT_DIR/declared.rs` | `op::declared(&str) -> Option<&'static Table>`、`op::names()` |
| `prof` | `crates/prof/devices/*.toml` | `$OUT_DIR/embedded.rs` | `prof::embedded(&str) -> Option<&'static Table>`、`prof::names()` |
| `tool` | `lints/vocabulary.toml` | `$OUT_DIR/vocabulary.rs` | `tool::vocabulary() -> &'static Table` |

三个脚本都：`cargo:rerun-if-changed` 指向源文件与目录、`OnceLock` 缓存一张表、遇到 `toml::Datetime`
或非表顶层直接让构建失败（不让它悄悄变成运行期错误）。`toml` 只作 `[build-dependencies]`。

### 3. 入口形状跟着变

- `OpDecl::parse(table: &Table, template: &str)`；`Op` 持有 `&'static Table`，`op::load` / `op::dir` /
  `BUILTIN`（声明文本三元组）删除——运行期没有"按目录装载算子"这件事。
- `Profile::parse(table: &Table)`；`prof::load` / `prof::dir` / `Archive` / `builtin()` 删除，
  `ProfileError::Syntax` 随之删除（没有运行期语法错误这一类）。
- `tool::Gate::from_table(&Table)`；`Gate::load()` 读生成表。
- `cutile::builtin` 的 `from_dir` / `from_repo` / `from_layers` / `disk_profiles` 删除：
  声明层与档案层都只有内嵌一份。

### 4. 用户配置只剩三层，没有配置文件

`cutile::settings` 删掉文件层、`Layer::File`、`SCHEMA_VER` / `schema` 校验、文件相对路径基准、
未知键的文件侧校验、以及只为文件层存在的 `missing` / `io` 分支。保留：内置默认 < `CUTILE_*`
环境变量 < `--set`，三个键与 `CUTILE_CACHE` 兼容名，空值拒绝，`[配置]` 段逐键"值 ← 来源"。

### 5. `fuzz` 不再覆盖 TOML

`tests/parsers.rs` 的 `Toml` / `Profile` 两个目标删除（没有运行期入口可灌），只剩 `cli` 与
`weights`（JSON、safetensors）。这使 `fuzz → +dev: prof` 成为死边，一并删除。

## 后果（实测，2026-11-14 工作树）

- **删除**：`crates/diagnostic/src/toml.rs`（585 行）、`settings/source.rs` 的读取路径、
  三个 crate 的运行期装载函数。`crates/*/src` 54 959 → **54 519** 行（−440），文件 168 个。
- **成员与边**：成员 **25** 不变（解析器原本就在 `diagnostic` 里）；边 **76 → 75**，`+dev` **12 → 11**
  （只少了 `fuzz → +dev: prof`）。
- **债务**：`[[gate.debt]]` **271 → 255**（删掉 8 条 `toml.rs` 的函数键、4 条设置层文件读取、
  2 条 `prof::load` / `prof.rs::fmt`、1 条 `tool::gate::load`、1 条 `op::load`），没有任何限额被抬高。
- **运行期依赖**：`Cargo.lock` 109 个包（`toml` 家族只作构建期依赖）；`cargo tree -e normal`
  里 **没有** `toml` / `serde` —— 运行时第三方依赖仍为 0。
- **用例**：62 目标 / 1 062 通过（ADR 0021 之后）→ **60 目标 / 1 052 通过 / 0 失败 / 1 忽略**。
- **`cutile::settings`**：23 → 19 条测试，删掉的 4 条只覆盖已消失的文件层（文件相对基准、
  未知键、`schema`、值形状），新增的用例补上"默认 < 环境变量 < `--set`"与"输出里不出现文件"。

## 重议条件

- **需要外部可编辑的语料**（用户想把 `devices/` 换成自己的一份、或用 `data.root` 覆盖档案层）：
  那时应该给"装载"设计一个显式入口（例如 `cutile --devices <dir>` 走一次性解析），而不是把
  解析器重新放回常驻路径。今天的答案是不提供。
- **出现第二种语料格式**（YAML/JSON 的声明层）：三种格式各有自己的 `build.rs`，数据树仍在
  `diagnostic::data`——那是"多一个生成器"，不是"多一个运行期解析器"。
- **构建期成本**：三个脚本各解析十几份小文件，增量构建只在源文件改动时触发；如果某天语料量级
  让构建时间明显变长，应缓存生成物（`OUT_DIR` 已经天然缓存），而不是退回运行期解析。

**不重议"运行期是否需要 TOML"**：三份语料是编译期资产，用户输入只有三个路径键——这两条今天
都成立，且都可以用环境变量/`--set` 表达。

## 后续（2026-11-14）：剩下的三层也搬走了

本条删掉第四层之后，`cutile::settings` 只剩三层。这三层随后被
[ADR 0023](0023-settings-to-cli.md) 整块上移到 `cli::settings`，`cutile` 只留
`cutile::dirs::Dirs`（三个路径的纯数据）。因此上面「`cutile::settings` 删掉文件层、`Layer::File`、
`SCHEMA_VER`、文件相对路径基准」这段描述的**位置**已过期，事实（那些东西全部消失、`missing`/`io`
不再由设置层产生、退出码契约不变）仍然成立。

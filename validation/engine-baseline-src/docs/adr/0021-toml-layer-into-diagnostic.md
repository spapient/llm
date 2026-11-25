# 0021：TOML 语法层并进 `diagnostic`（`llm-config` 作为成员删除）

- 状态：已采纳
- 日期：2026-11-14

## 背景

ADR 0020 只裁决了 `llm-config` 的一半（设置层 1 824 行）。剩下的一半是 583 行受限 TOML 解析器，
它有六个跨层消费者：`op`（`ops/*.toml`）、`prof`（`devices/*.toml`）、`tool`
（`lints/vocabulary.toml` 的阈值）、`cutile`（经设置层）、根包与 `fuzz`（`tests/parsers.rs`
的驱动对象）。因此它不能像设置层那样被单个消费者吸收。

审计当时的建议（`docs/audit.md` §13.6 的 A2）是**改名**：`crates/config` → `crates/toml`、
包名 `llm-config` → `llm-toml`，理由是"它是语法层，不是配置层"。但改名与解耦是两件事：

- 改名后成员数不变（25 个 crate + 根包），依赖边一条不少；
- 指向它的六个 `use` 只换字符串，仍然"每个成员各自知道有个 TOML 包"；
- "一个概念一个位置"没有改善——只是把一个名字换成了更准确的名字。

而那条边本来就是**冗余**的：这六个成员全部已经依赖 `diagnostic`（错误类型是全仓共用的），
语法层却要求它们再依赖一个只有 583 行的包。

## 决策

把语法层直接放进它已经必须依赖的那个 L1 成员，然后删掉 `llm-config`：

- `crates/config/src/toml.rs` → `crates/diagnostic/src/toml.rs`，`lib.rs` 加一行 `pub mod toml;`；
- `crates/config/`（目录、`Cargo.toml`、`README.md`）删除；
- 六个消费者（`fuzz` 只在 `tests/parsers.rs` 里用，是 `+dev` 边）的依赖项与 `use` 改写为 `diagnostic::toml::*`；
- `prof` 的再导出（`pub use llm_config::toml`）删除。它原本是给 `op` / `tool` / 根测试用的转发
  路径：`prof/src/prof.rs` 自己有 17 处 `crate::toml`，说明"借道 `prof`"这件事一直在发生。

不在 `diagnostic::Error` 上再包一层：`ParseError` 只带行号与说明，类别（`invalid` / `missing`）
由调用方按语义决定——语法层不该知道一次语法失败算哪一类，正如它不该知道 `ops/` 属于 `op`。

两个原语共处一个 crate 的理由只有一条，而且是依赖方向而非语义：它们都必须比任何数据层更低，
彼此又不引用一行（`toml` 不需要 `Error`，`Error` 不认识表）。

## 后果

- **成员 26 → 25**（crate 24）；**依赖边 82 → 76**（含 `+dev` 12）：消失的六条是 `op` / `prof` /
  `tool` / `cutile` / `llm` / `fuzz` 各自指向语法层的那一条，**新增零条**。
- **行数几乎不变**：`crates/diagnostic/src` 260 → 856（+596：`toml.rs` 585 行 + `lib.rs` 11 行文档），
  `crates/config/src` 605 → 0（−605），`crates/*/src` 净 **−1 行**（`config` 的 22 行入口文档消失，
  `toml.rs` 原样搬家；文件数 168 → 167）。
  这条决策买的是成员数与边数，不是行数——与 ADR 0019 的"包装合并不自动带来减行"同一条结论。
- **用例 62 目标 / 1 063 通过 → 60 目标 / 1 062 通过**：少的 1 条是 `llm-config` 的 doctest
  目标本身；`toml.rs` 内 13 个测试随文件迁入 `diagnostic` 并继续执行。
- **门禁**：`[gate.debt]` 8 条键改路径（ccn / nloc 数值逐条不变，搬家不降复杂度）；
  `[[ownership.unlisted]]` 的 `config` 条目改为 `diagnostic`（`ParseError` / `Value`）。
- **错误边界测试的例外被收窄而不是取消**：`tests/architecture.rs` 删掉 `member == "config"`
  分支（成员不存在了），改成文件级豁免 `crates/diagnostic/src/toml.rs`——`parse_value` /
  `parse_number` / `Table::insert` 的 `String` 是消息片段，对外一直是 `ParseError`，
  从来不是任何一层的失败契约。
- **`diagnostic` 的定位变了**：从"唯一的错误类型"变成"两个 L1 原语（错误类型 + 唯一的 TOML
  解析器）"，crate 文档已按这个口径重写。

## 重议条件

- 出现第二种文本格式（JSON / YAML 要共用同一批 `Table` 访问器），语法层就该有自己的成员，
  而不是继续与错误类型同居；
- 某个成员只想用错误类型、却因为这条依赖被迫拉进解析器（依赖粒度不足）——今天没有这种成员，
  解析器零外部依赖，编译代价只有 583 行。

**不重议"是否删掉 `llm-config`"**：改名不产生解耦，把六个 `use` 指向一个为它们共同的 L1
成员，是同一件事的更小形态。

## 后续（2026-11-14）

本条把语法层并进 `diagnostic`，但保留了解析器本身。随后 [ADR 0022](0022-build-time-corpora.md) 判定
"运行期根本不需要解析器"：`ops/`、`devices/`、`lints/vocabulary.toml` 改由各 crate 的 `build.rs`
在构建期解析成 `diagnostic::data` 的表，`crates/diagnostic/src/toml.rs` 与 `ParseError` 一并删除。
因此本条的结论中只有"语法层不该是独立成员"仍成立；"语法层留在 `diagnostic::toml`"已被取代，
`Table` / `Value` 现在住 `diagnostic::data`。

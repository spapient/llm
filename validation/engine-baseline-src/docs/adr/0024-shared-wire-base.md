# 0024：抽离共享底层 `wire`（字节码闭集：CRC / LEB128 / Sink）

- 状态：已落地（S1），门禁全绿
- 日期：2026-11-14

## 背景

`weights` 内有三个**与权重领域无关**的字节码基础件，目前 `pub(crate)` 藏在 L5：

- `weights::crc`：CRC-32（IEEE 802.3，八路查表），`crc32` / `update`；
- `weights::leb`：LEB128 变长整数（`write` 泛型写 + 只读游标 `Cursor`，非规范编码拒绝）；
- `weights::sink`：统一字节输出 `Sink` 与 `Count` / `Vec<u8>` / `SliceSink` / `FileSink` / `Measured` / `Sealed`（计数、固定缓冲、流式文件、位置跟踪、CRC 密封）。

`train::snapshot` 为了不向上借用这套 `pub(crate)` 件，**第二份实现**了同一套东西：

- 自有 `Sink`（`Count` / `VecSink` / `SliceSink` / `FileSink` / `Sealed`）；
- 自有 `Cursor`（`take` / `u8` / `varint` / `count` / `remaining`）；
- 自有 `varint` 编码与"非规范编码拒绝"判定。

这是 `docs/audit.md` §P2-7 登记在案的重复：LEB128 在 `weights/src/leb.rs:11,55` 与
`train/src/snapshot/mod.rs:687,733` 各写一遍，连论证都各写一遍。两类字节码规则各自只有一处
权威实现，才不会在一个修了非规范拒绝、另一个漏掉时悄悄漂移。

### 反例（为什么必须抽走共享，而不是继续各写一份）

- 动作成功但调用方不知道：快照收窄了某个 LEB 宽度判据，weights 容器没同步，旧快照仍能编出
  读端拒不掉的字节——两端各自有测试、各自绿，没有门禁会变红。
- 跨版本共存：容器与快照格式演进时，第二份游标把同一规则拆成两处维护。

## 决策

新建 L1 基础设施成员 `wire`（包名 `llm-wire`），只承载**纯字节码闭集**，零领域语义：

| 落点 | 来源 | 内容 |
|---|---|---|
| `wire::crc` | `weights/src/crc.rs` | CRC-32 八路查表、`crc32` / `update`，对拍测试随迁 |
| `wire::leb` | `weights/src/leb.rs` | `write`（`Uint` 泛型族）/ `Cursor` 与规范性拒绝 |
| `wire::sink` | `weights/src/sink.rs` | `Sink` 与五种接收器、`Sealed`、`Measured` |

依赖方向：`wire → diagnostic`（L1 内，`diagnostic` 在左）；`weights → wire`（L5→L1 向下）；
`train → wire`（L6→L1 向下）。方向断言不需要放宽。

### 归属与兼容

- `weights` 删除三份实现体，保留薄转发模块以维持既有路径：`weights::crc::{crc32,update}`
  （`train::snapshot` 现依赖 `weights::crc`）等内部 `crate::sink` / `crate::leb` 引用改指 `wire::`。
- `train::snapshot` 删除自有 `Sink` / `Cursor` / `varint` 副本，改用 `wire`；其字段级上下文
  （"第 N 个条目"）仍由 snapshot 自己补在 `wire::Cursor` 的错误上。
- `serve::proto::crc` **保留第二份**：该模块文档明确记录了取舍——`serve` 只依赖 `diagnostic`，
  为二十行算术拖入整条权重依赖链比重复一张 256 项表更贵。这是有据的例外，不并入 `wire`。

### 不做的事

- 不把 rANS、Merkle、SHA-256、压缩或容器布局搬进 `wire`：它们是 `weights` 的容器领域知识，
  不是通用字节码。
- 不改任何字节格式、魔数、版本号或校验语义；这次是搬家 + 去重，不是重写。

## 候选与取舍

- **候选 A：新建 `wire` crate（采纳）**。职责单一、依赖方向向下、第二个字节码消费者出现时
  不必向上借权重容器；代价是工作区成员 +1、运行时边 +3（`wire→diagnostic`、`weights→wire`、
  `train→wire`），开发边一进一出（`wire→testkit` 进、`weights→testkit` 出，dev 仍 11），
  需同步 `docs/ARCHITECTURE.md` 层表/边集/计数与所有权基线。
- **候选 B：把三件塞进现有 L1 `diagnostic`**。不新增成员、不增边；但 `diagnostic` 的职责是
  错误类型与声明数据树（ADR 0021/0022），塞通用编解码会污染其单一职责，与本次"类聚"目标冲突。
- **候选 C：把 weights 的三件升为 `pub`，train 经既有 `weights` 边复用**。不动成员/边；但通用
  字节码留在 L5 权重层，L6 为取一个 CRC 向上借整条权重依赖，分层语义不对。

## 门禁与可验证性

落地后必须仍由既有工具测得全绿：

- `cargo test -p llm-wire -p llm-weights -p llm-train -p llm-storage` 全绿（现有 239 用例不丢）；
- `cargo clippy --all-features -- -D warnings` 零告警；
- `tests/architecture.rs`：成员集、边集（75 → 78，`+dev` 仍 11）、方向、所有权双向校验一致；
- `tool` 静态门：`ccn ≤ 5`、`nloc ≤ 60`、单文件 ≤ 899、生产代码零 `unwrap/expect/panic`、`debt` 空表；
- `cargo deny check` 干净；`cargo llvm-cov` 覆盖率不低于基线（行 87.15%）。

## 回滚

`wire` 是纯搬运：任何切片出问题，回退 `weights` / `train` 的 use 路径并删除 `wire` 目录即可，
字节格式不变；快照与容器之间没有需要迁移的持久状态。重议条件：`wire` 开始出现设备厂商或张量
布局知识时，说明它越界，应拆回对应领域层。

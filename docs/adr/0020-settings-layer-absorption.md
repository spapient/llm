# 0020：设置层的吸收（`llm-config` 的一半并入 `cutile`）

- 状态：已采纳
- 日期：2026-11-14

## 背景

ADR 0006 把"配置的机制"与"配置的身份"分开，`Settings<A: App>` 的类型参数就是那条分界的执行
方式。它当时解决的是真实问题：`FILE_NAME`、`CUTILE_` 前缀与用户级目录名被写死在一个常量里，
于是"`llm` 想读自己的配置"只有两条路——寄生在 `CUTILE_*` 下，或者复制一遍分层逻辑。

但那条设计有三个前提，事后一条也没有成立（逐条取证见 `docs/audit.md` §13.1–§13.3）：

- **身份只有一个实现**：`App` 的生产实现只有 `Cutile` 一个；根二进制想"用自己的身份"，
  写的仍是 `Settings::<config::Cutile>`——它借的是 `cutile` 的名字。`App::KEYS` 让"别的应用
  不继承 cutile 的旋钮"成为可执行事实，可第二个应用从未出现，这条事实只被测试用着。
- **值形状只有一种可达**：`Kind` 登记了五种形状（路径 / 文本 / 整数区间 / 布尔 / 枚举），
  而三个键的 `kind()` 全部返回 `Kind::Path`；`Text` / `Int` / `Bool` / `Choice` 与它们的分支
  没有任何生产入口。
- **消费者只有一个**：设置层 1 577 行的唯一生产消费者是 `cutile`（67 处）与根包（13 处，
  且根包经 `cutile` 取值）。同一份 crate 里的 TOML 语法层 583 行却有六个跨层消费者
  （`op` / `prof` / `tool` / 根测试 / `fuzz`）——两件互不相干的资产被塞在一个包里，
  于是"删不删 `llm-config`"这个问题永远吵不出结论。

代价是可见的：为三个键手写的 `ALL: [Self; 3]` 必须靠 `lints/vocabulary.toml` 的
`[closed]` 豁免才过门禁；`config::Error` 是全仓唯一还活着的非 `diagnostic` 错误类型，
`tests/architecture.rs` 专门为它开了一个例外分支。

## 决策

把配置层拆成两件独立裁决的资产，本条只处理第一件：

1. **设置层**（分层优先级、来源、键闭集、平台目录、设置对象）并入 `cutile`，落点是
   `crates/cutile/src/settings/` 的子模块，不是两个 `lib.rs` 拼在一起：
   `mod.rs`（`Settings`）、`key.rs`（`Key`）、`paths.rs`（`Roots`）、`source.rs`（分层与错误）。
2. **TOML 语法层**留在原成员里（本条不改名；改名 `llm-toml` 是 `docs/audit.md` §13.6 的 A2）。

同时删掉四个已无入口的抽象：`App` 类型参数（身份只剩一个）、`Kind`（三个键全是路径）、
`Value`（`PathBuf` 就是全部）、`config::Error`（改用 `diagnostic::Error`，类别由产生错误的
那一层给；`missing` / `io` / `invalid` 分别落到退出码 4 / 5 / 2）。

## 后果

- **成员数与边数不变**（26 / 82）：`cutile → llm-config` 这条边还要留着，因为 `cutile` 仍然
  读 TOML。合并换的不是成员数，而是"一个概念一个位置"。
- **行数净减**：`crates/config/src` 1 824 行设置层代码与 78 行入口文档删除，`crates/cutile/src/settings/`
  新增 1 145 行（含 23 个测试），净 −≈750 行。少掉的是 `Kind`/`Value`/`App`/`Error` 四套抽象。
- **`Key` 不再是手写闭集**：`impl prof::atom::Closed` + `prof::inherent_names!`，`ALL` 写成切片，
  `as_str` 是配置文件里的点号名（`cache.dir`）。`lints/vocabulary.toml` 的 `[closed]` 白名单
  因此少一条豁免，`[gate.debt]` 少三条（`value.rs` 三条 + `error.rs` 一条删除，`source.rs`
  四条换成新路径下的五条，净 −3）。
- **用例数从 1 083 降到 1 063**：减少的 20 条全部是被删类型自己的测试（`app` 5、`value` 6、
  `error` 3、`key` −1、`settings` −2、`source` −3）。这不是放宽判据——删掉的分支不再存在。
- **用户可见行为一条未变**（8 条承诺的对拍见 `docs/audit.md` §13.5）：四层优先级、
  `CUTILE_CACHE` 兼容名、未知键报错、`schema = 1` 校验、文件内相对路径按文件目录解析、
  环境变量与 `--set` 的相对路径按启动目录解析、`data.root` 三层覆盖、`cutile doctor` 的
  `[配置]` 段逐键打印"值 ← 来源"。错误文本多了 `diagnostic::Error` 的类别前缀与 `→` 位置帧，
  消息正文逐字保留。
- **退出码更准**：配置文件读不出来以前一律折成 `invalid`(2)，现在 `missing`→4、`io`→5。

## 重议条件

出现下列任一条，就把设置层重新拆出去（不需要羞愧，"单消费者"是当时的判断，不是永久事实）：

- 第二个应用真的要读自己的配置，且它**不在** `cutile` 的下游（那时身份参数才重新有意义）；
- 设置层开始持有 `cutile` 之外的语义（例如训练超参、服务并发），那时它本该是训练层或服务层的类型；
- `cutile::settings` 的用户超过两个成员，说明它又被当成"全仓配置层"在用了。

**没有触发条件就不拆**：吸收是产品裁决，不是不可逆的架构结论。

## 后续（2026-11-14）

本条刻意把第二件事（TOML 语法层）留给后续裁决，并在上面写明"改名 `llm-toml` 是 A2"。实际执行时
改名被否决：[ADR 0021](0021-toml-layer-into-diagnostic.md) 判定改名不产生解耦（成员数、边数、
`use` 数量都不变），于是语法层直接并进同层零依赖的 `diagnostic::toml`，`llm-config` 作为成员删除。
因此上面两处描述已被取代：

- **决策 2**「TOML 语法层留在原成员里」→ 语法层进 `diagnostic::toml`，成员 26 → 25；
- **后果 1**「成员数与边数不变（26 / 82）」→ 最终是 **25 / 76**；`cutile → llm-config` 这条边
  变成 `cutile → diagnostic`，而这条边本来就存在，所以它是"net −1 成员、−6 边、+0 边"。

设置层这一半的结论（类型参数删除、值形状收敛、错误类型统一、8 条用户可见承诺不变）不受影响。

## 后续（2026-11-14）：落点再次改变

[ADR 0022](0022-build-time-corpora.md) 删掉第四层（用户级配置文件）之后，`crates/cutile/src/settings/`
只剩「默认值 + `CUTILE_*` + `--set`」，而这三件事没有一件是内核机制该决定的；
[ADR 0023](0023-settings-to-cli.md) 因此把整块策略上移到 `cli::settings`，`cutile` 只留
`cutile::dirs::Dirs` 这份纯数据。因此上面"决策"里的**落点**（`crates/cutile/src/settings/`）已被取代，
`crates/cutile/README.md` 的"配置边界"一节同步改写；本条其余结论（`App` 类型参数删除、值形状收敛为
路径、错误类型统一、8 条用户可见承诺不变、键闭集集中一处）全部仍然成立，其中"键闭集"现在的实现是
`cli::settings` 里那张一行一个的声明宏表。

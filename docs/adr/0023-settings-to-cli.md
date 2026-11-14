# ADR 0023：配置策略上移到命令行层，内核只收三个目录

- **状态**：接受（2026-11-14）
- **取代**：[ADR 0020](0020-settings-layer-absorption.md) 的落点（`crates/cutile/src/settings/`）与
  [ADR 0022](0022-build-time-corpora.md) 中「`cutile::settings` 只剩三层」的定位；两份 ADR 的其余结论不变
- **相关**：[ADR 0006](0006-config-mechanism-vs-identity.md)（四层优先级的出处）、[ADR 0019](0019-packaging-merge.md)（合并判据）

## 背景

ADR 0020 把 `llm-config` 的设置层并进 `cutile`，理由是"只有一个生产消费者"。ADR 0022 删掉第四层
（用户级配置文件）之后，`crates/cutile/src/settings/` 里只剩三件事：

1. 三个键的**默认值**（`~/Library/Caches/cutile/cache` 之类，面向发布后的二进制）；
2. **环境变量层**（`CUTILE_*`，含 `CUTILE_CACHE` 兼容名）与**命令行层**（`--set key=value`）的优先级；
3. 相对路径的**解析基准**与每个值的**来源追踪**（`值 ← 来源`）。

这三件事与"编译、协商、选择、替换、发射、缓存"没有一处共用一个类型，而它们共处一个 crate 只带来
一个后果：`cutile` 的 12 个模块与两个二进制把 `&Settings` 当参数一路透传，内核因此间接持有
"默认目录在哪""相对路径按谁解析""`--set` 怎么拆"这三条**属于调用方**的知识。删掉配置文件层之后，
这套分层机制（`Layer` / `Origin` / `Entry` / 槽位数组）服务的对象只剩两个注入层，成本已高于收益。

## 决策

**一、`cutile` 只声明输入形状。** 新增 `crates/cutile/src/dirs.rs`（64 行）：`Dirs { cache, work, data }`
三个路径的纯数据 + 三个访问器 + 一个 `Dirs::under(root)` 便利构造。它不读环境变量、不拼平台路径、
不认识命令行，依赖只有 `std::path`。12 个模块与 `Cache`/`Catalogue`/`build`/`doctor`/`gc` 等入口的
参数由 `&Settings` 改为 `&Dirs`，方法名随语义改短（`cache_dir()` → `cache()`）。

**二、策略整块搬到命令行层。** 新增 `crates/cli/src/settings.rs`（660 行，含 13 个测试）：默认目录
（`Roots::{discover,new,with_base}`）、`CUTILE_*` 注入、`--set` 写入、优先级、来源追踪与 `describe`
全部在这里；两个二进制里"扫一遍 argv 摘掉 `--set`"的那段（原 `take_overrides`）也变成
`Settings::apply_arguments`，因为它是**命令行文法**，不该散在驱动方。

**三、键表用声明宏一行写完。** 三个键（`cache.dir` / `work.dir` / `data.root`）的点号名、环境变量后缀、
说明、槽位下标由同一行派生：

```rust
keys! {
    CacheDir => "cache.dir", ["CACHE_DIR", "CACHE"], "编译产物缓存的根目录";
    WorkDir  => "work.dir",  ["WORK_DIR"],          "编译与执行时的临时工作目录";
    DataRoot => "data.root", ["DATA_DIR"],          "声明层 / 档案层 / 片段层的替代根；…";
}
```

`describe` 的键序、"键未登记"报错里列出的清单、`--set` 的查表都来自这张表，因此加一个键只加一行，
且不会出现"清单漏了某个键"。键表**不再**借 `prof::atom::Closed`：`cli` 本来不依赖 `prof`，而三个键
不值得为它拉进一整套闭集与 `Set<E, WORDS>` 位集机制（ADR 0018 的判据是"闭集共享一份实现"，不是
"任何枚举都必须走那套机制"）。

**四、两个二进制各写一个 3 行投影。** `kernel_dirs(&Settings) -> Dirs` 在 `src/main.rs` 与
`src/bin/cutile.rs` 各有一份。这是刻意的重复：让 `cli` 依赖 `cutile` 只是为了省这 3 行，会把
"命令行层"钉死在内核上（`cli → cutile → op` 的传递依赖），并把 `Dirs` 变成 `cli` 的公开面。

## 判据

| 判据 | 改前 | 改后 |
|---|---|---|
| 内核是否知道"默认目录在哪" | 是（`settings/paths.rs` 三个 `cfg` 分支） | 否（只收 `Dirs`） |
| 测试/服务能否接管三个目录 | 能，但要构造 `Settings`（分层机制一起搬） | 能，`Dirs::new(..)` 一行 |
| 新增一个配置键要改几处 | 键闭集 + 4 张 `match` + 消费方取值 | 键表一行 + 消费方取值 |
| `cli` / `cutile` 的依赖边 | `cli → diagnostic`、`cutile → {op, prof, prim, weights, diagnostic, mem, storage}` | **逐条不变** |
| 成员数 | 25 | **25** |

## 后果

- **成员数与依赖边一条未增**：`cli` 仍只依赖 `diagnostic`，`cutile` 的出边与入边原样。这次买的是
  "内核与部署策略解耦"，不是减行数（见下）。
- **行数**：`cutile/src/settings/**` 4 文件 924 行删除，新增 `cutile/src/dirs.rs` 64 行与
  `cli/src/settings.rs` 660 行，本次搬迁自身净 **−200 行 / −2 文件**。
- **用例**：`cutile` 的 settings 模块 19 条（ADR 0022 之后）→ `cli::settings` **13 条**。少掉的 6 条
  都只覆盖随第四层一起消失的行为（文件相对基准、未知键的文件校验、`schema`、值形状报错两条），
  其余 5 条改写为同一判据的等价断言：优先级（默认 < env < cli）、正式名优先于兼容名、
  `CUTILE_CACHE` 仍被承认、相对路径按启动目录解析、`describe` 逐行"值 ← 来源"且不出现"文件"。
  新增 2 条：`--set` 缺值时是 `invalid`（argv 扫描的边界）、`Roots::discover` 不锚定仓库目录。
- **用户可见行为逐字不变**：`[配置] 默认 < 环境变量 < 命令行` 三行排版、`← 内置默认值` /
  `← CUTILE_WORK_DIR（env）` / `← --set cache.dir=…（cli）`、三个键的语义、`CUTILE_CACHE` 兼容名、
  空值拒绝、未登记键拒绝、退出码（`invalid` → 2）。`missing` / `io` 自 ADR 0022 起就不再由设置层
  产生，与本条无关。
- **`cutile` 的公开面少了一整块**：`Settings` / `Roots` / `Key` / `Layer` / `Origin` 不再由 `cutile`
  导出；`Settings` / `Roots` 改由 `cli` 导出。`tests/architecture.rs` 的所有权表随之拆成两行。
- **门禁**：删除 `[[gate.debt]]` 中 `crates/cutile/src/settings/source.rs:cli_override#0`（函数随目录
  消失）；新增代码全部在 `ccn ≤ 5` / `nloc ≤ 60` / 文件行数基线内，未新增、未抬高任何条目。

## 否决的替代方案

- **把 settings 提成新的 L1 成员**：+1 成员、+1 条边，只为 660 行开一个 crate；它今天只有一个宿主
  形态（两个二进制同源），等出现第二个宿主再拆不迟。
- **让 `cutile` 自己读 `CUTILE_*`**：内核读环境变量意味着嵌入方无法接管（测试要 `set_var`、
  服务要全局进程状态），这正是本条要消除的东西。
- **让 `cli` 依赖 `cutile` 以复用 `Dirs`**：省 3 行，换来"命令行层绑定内核"，且 `Dirs` 会成为 `cli`
  公开 API 的一部分——ADR 0019 的"单消费者吸收"不适用于"消费者是更上层"的情形。
- **保留分层机制只把 `Settings` 改小**：`Layer` / `Origin` / 槽位数组服务的对象只剩两层，
  而"来源"这件事必须留在能同时看到 env 与 argv 的那一层，所以机制跟着搬家，不是缩水。

## 重议条件

- **出现第二个宿主**（服务、测试框架、构建工具都要按同一组 env / `--set` 解析）：那时 `cli::settings`
  应下沉为独立成员，`Dirs` 仍留在 `cutile`。
- **键数超过约 10 条，或出现非路径值**：`keys!` 的一行四列会开始拥挤，需要按域拆表并给取值加类型
  （今天三个键都是路径，因此没有 `Kind` / `Value` 这套值形状系统）。
- **需要按用户/项目隔离多套目录**（多租户缓存）：那是"目录发现策略"的新维度，应新增一个显式的
  `Roots` 构造，而不是往键表里塞更多键。

**不重议"内核是否读环境变量"**：不读是这条 ADR 的全部意义。

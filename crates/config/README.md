# llm-config

配置基座：本 workspace 里**唯一**的"值从哪来"和"目录在哪"的来源。`cutile` 与 `llm` 都只单向依赖它，
它自己只依赖 `std`，不反向依赖任何成员。

## 为什么单独一层

配置的三种坏味道都来自"谁都能读一点"：同一环境变量在多处各读一遍、相对路径按当前工作目录解析、
新增配置项时到处加字符串。本 crate 用三条规则把它们堵住：

1. **键闭集**：配置项只能登记在 `Key::ALL`，读取只能通过 `Key` 常量；配置文件里出现未登记的键直接报错，
   而不是被静默忽略。新增一项必须先改这里，重复与交叉引用在编译期就暴露。
2. **唯一来源**：`Settings::<A>::load()` 每个进程只调用一次，之后按值传递；消费方不再自己读 `std::env`。
3. **绝对路径**：默认目录由平台用户目录推出，与仓库布局、进程启动目录都无关。

没有消费方的键不登记：一个转了不生效的旋钮，会让 `doctor` 里的"值 ← 来源"变成假承诺。

## 机制与身份

```rust
pub trait App {                     // 身份：四个常量
    const DIR: &'static str;        //   用户级目录名
    const FILE: &'static str;       //   配置文件名
    const ENV_PREFIX: &'static str; //   环境变量前缀
    const KEYS: &'static [Key];     //   本应用认哪些键
}

pub struct Settings<A: App = Cutile> // 机制：分层、来源、键闭集、路径
```

`Cutile` 是默认身份（`cutile.toml` / `CUTILE_*` / `<cache home>/cutile`，键集是全部已登记的键）。
别的层要用自己的配置时写 `Settings::<自己的 App>::load()`——四个常量 + 一个标记类型，
不复制分层逻辑，也不会出现"`llm` 的设置叫 `cutile`"。

**键集也是身份的一部分**：文件层、环境变量层与 `--set` 都只认 `App::KEYS` 里登记过的键。
一个只登记 `data.root` 的应用写 `[cache] dir = ..` 会直接得到 `Error::UnknownKey`（错误信息里
列出它真正可用的键），`CUTILE_CACHE_DIR` 对它也视而不见；`Settings::describe()` 只打印本应用的键。
这是"另一个应用不继承 cutile 的旋钮"从约定变成可执行事实的地方。

注意：关联函数路径不套用类型参数默认值，所以调用点要显式写 `Settings::<Cutile>::load()`
而不是 `Settings::load()`（见 `docs/adr/0006-config-mechanism-vs-identity.md`）。

## 分层与优先级

```text
内置默认  <  用户级配置文件  <  环境变量 <PREFIX>_*  <  命令行 --set key=value
```

每个值都带 `Origin`（层 + 具体来源），`Settings::describe()` 逐键打印"值 ← 来源"，
因此"为什么这次用的是这个缓存目录"永远可以当场回答。

## 目录（面向发布）

默认值不依赖 git checkout：装好的二进制在任意目录运行，命中的都是同一份用户级目录。

| 平台 | 配置文件 | 缓存根 |
|---|---|---|
| macOS | `~/Library/Application Support/cutile/cutile.toml` | `~/Library/Caches/cutile` |
| Linux/BSD | `${XDG_CONFIG_HOME:-~/.config}/cutile/cutile.toml` | `${XDG_CACHE_HOME:-~/.cache}/cutile` |
| Windows | `%APPDATA%\cutile\cutile.toml` | `%LOCALAPPDATA%\cutile` |

仓库里不放配置文件；需要偏离默认时写用户级文件，或用环境变量 / `--set`。

## 键闭集

| 键 | 环境变量 | 默认值 |
|---|---|---|
| `cache.dir` | `<PREFIX>_CACHE_DIR`（兼容旧名 `<PREFIX>_CACHE`） | `<缓存根>/cache` |
| `work.dir` | `<PREFIX>_WORK_DIR` | `<缓存根>/work` |
| `data.root` | `<PREFIX>_DATA_DIR` | 未设置：使用 `op`、`prof` 与 `prim` 编译期内嵌数据 |

`data.root` 指向一个含 `ops/`（声明层）、`devices/`（档案层）、`prim/`（片段层）的数据目录，
让三层数据都能在不改代码的情况下替换；仓库里三层分别是 `crates/op`、`crates/prof`、`crates/prim`。

## 什么不该放进来

- 构建与工具链配置（`Cargo.toml`、`clippy.toml`、`rustfmt.toml`、`rust-toolchain.toml`、CI）：
  这些由工具按路径向上查找，搬进来等于静默失效。
- 领域数据本身（`ops/`、`devices/`、`prim/` 的内容）：它们住在 `crates/op`、`crates/prof` 与 `crates/prim`，
  与各自解释它的 schema 相邻；本 crate 只提供**解析器**与**目录位置**，不解释数据语义。
- "以后可能会用到"的旋钮：没有消费方就不登记。

## 例子

```rust
use llm_config::{Key, Roots, Settings};

let roots = Roots::new("/home/u");
let mut settings = Settings::defaults(&roots);
assert_eq!(settings.cache_dir(), std::path::PathBuf::from("/home/u/cutile/cache"));

settings.apply_override("cache.dir=build/cache").expect("合法覆盖");
assert_eq!(settings.cache_dir(), std::path::PathBuf::from("/home/u/build/cache"));
assert_eq!(
    settings.origin(Key::CacheDir).layer(),
    llm_config::source::Layer::Cli
);
```

## 依赖方向

```text
llm ──▶ cutile ──┬─▶ op
                 ├─▶ prof
                 └─▶ llm-config
```

`llm-config` 不依赖任何成员；`cutile` 通过它拿缓存目录、工作目录与可选的数据根。

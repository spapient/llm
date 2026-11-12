# llm

用 Rust 编写的配置驱动算子训练推理框架：kernel 由声明与档案驱动（`cutile`），上层按层组装成推理与训练链路（`llm`）。

本仓库是一个 Cargo workspace，共 32 个成员（`crates/*` 下的 31 个 crate 与根包 `llm`（入口 bin））。
元数据、依赖版本、lint 策略统一在根 `Cargo.toml` 声明；**模块的层级与依赖方向**写在
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，新增成员时按那里的三步登记。

## 快速开始

需要 Rust 1.100 或更高版本；`rust-toolchain.toml` 已把工具链固定为 `nightly-2026-09-01`（rustc 1.100.0-nightly），
只要本机装了 `rustup`，进入本目录即自动切换。1.100 正式版发布后，把该文件的 `channel` 改成 `"1.100"` 即可切回正式版。

```bash
cargo run -p llm                              # 组装点：默认命令打印各层摘要
cargo run -p llm -- help                      # 命令行用法（解析在 `cli`，执行在 `llm`）
cargo build --workspace                       # 编译整个 workspace
cargo test --workspace                        # 跑全部成员的测试与 doctest
```

## 常用命令

| 命令 | 作用 |
|---|---|
| `cargo fmt --all` | 按 `rustfmt.toml` 格式化所有成员 |
| `cargo clippy --workspace --all-targets --all-features` | 静态检查（lint 策略见根 `Cargo.toml` 的 `[workspace.lints]`） |
| `cargo test --workspace` | 运行测试 |
| `cargo doc --workspace --no-deps --open` | 生成并打开文档 |

注解声明的层（形状与算子写在属性里）与运行期构造的层（宽度按配置计算）走同一条校验路径：

```rust
#[nn::layer(name = "lm_head", out = [4096], ops = ["gemv_i4"])]
struct LmHead;   // 生成 impl nn::Layer + 一条“声明仍与 op 数据一致”的测试
```

注意：因为根目录本身就是一个包，不带 `-p` / `--workspace` 的 `cargo build`、`cargo test`、`cargo clippy`、`cargo doc`
只作用于 `llm`，不会覆盖 `crates/` 下的成员；只有 `cargo fmt --all` 天生遍历全部成员。

## 分层

完整表格、所有权表与边清单见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：

| 层 | 成员 | 职责 |
|---|---|---|
| L1 基础设施 | `diagnostic` `testkit` `llm-config` `tool` `benches` | 统一错误类型、测试夹具、配置分层与路径、人可读输出、基准判定 |
| L2 数据层 | `einsum` `prof` `op` `prim` | 求和表达式的语义代价、设备档案与能力闭集、算子声明与模板、原语片段与见证（纯数据） |
| L3 内核 | `cutile` `audit` | `cutile`：协商 → 拼接 → emit → 编译执行，见 [DESIGN](crates/cutile/DESIGN.md)；`audit`：在其上做 doctor/verify/bench 的结构化报告与见证执行 |
| L4 运行时 | `mem` `tensor` `storage` `exec` `tokenizer` | 分配、张量元数据、布局搬运、提交执行、文本边界 |
| L5 模型与训练 | `weights` `kv` `autograd` `comm` `proto` `data` `nn` `shard` | 权重结构、KV 页、反向图、集合通信、线协议、样本、层定义、切分拓扑 |
| L6 调度与服务 | `checkpoint` `train` `engine` `serve` `cli` | 检查点格式、训练循环状态、连续批与序列、端点契约、命令行解析 |
| L7 入口 | `llm`（bin） | 组装：把上面各层连起来，不定义新类型 |

唯一一条依赖规则：**依赖只能指向更低的层（同层只能依赖更靠左的成员）**。因此向上依赖不可能通过、
环不可能出现、数据层永远不会知道机制（`op` / `prof` / `prim` 不依赖 `cutile`）。
这条规则由 `tests/architecture.rs` 在 `cargo test` 里对照 `cargo metadata` 自动核对。

`annotation`（编译期注解）与 `fuzz`（模糊测试契约）是**工具成员**：它们不计入运行时分层，
也不参与“至少一条依赖边”的统计。

```text
llm ──▶ engine ──▶ kv ──▶ tensor ──▶ mem ──▶ diagnostic
  │        └─────▶ exec                     ▲
  ├──▶ serve ──▶ proto / tokenizer ─────────┘
  ├──▶ cli ──▶ diagnostic
  ├──▶ nn ──▶ op / annotation
  ├──▶ train ──▶ nn ──▶ op
  │             ├▶ checkpoint ──▶ weights
  │             ├▶ autograd ─┐
  │             ├▶ data ─────┤
  │             └▶ shard ──▶ comm
  ├──▶ weights ──▶ storage ──┘
  ├──▶ op / prof / tool
  │        └─────▶ einsum ──▶ diagnostic
  ├──▶ audit ──▶ cutile / op / prof / prim
  └──▶ cutile ──┬─▶ op / prof / prim
                └─▶ llm-config
```

## 目录结构

```
.
├── Cargo.toml            # workspace + 根包 llm（入口）、[workspace.package]/[lints]/[profile.*]
├── src/main.rs           # 组装点：逐层构造最小实例
├── src/bin/cutile.rs     # 内核层审计的驱动方：解析（llm-cli）→ audit → 排版 → 退出码
├── docs/
│   ├── ARCHITECTURE.md   # 分层、依赖规则、所有权表、新增成员三步
│   └── adr/              # 0001 分层序 / 0002 单一主人 / 0003 kernel 与运行时类型
│                         # 0004 占位策略 / 0005 注解即契约 / 0006 配置机制与身份分离
│                         # 0007 代价来自一条求和表达式
└── crates/
    ├── annotation/       # #[layer] 属性宏：注解 → 校验过的 Layer 实现（零外部依赖）
    ├── audit/            # 内核审计：doctor / verify / bench 的结构化报告与见证执行
    ├── checkpoint/       # 检查点文件：恢复事实 + 权重条目表的编码、版本与偏移校验
    ├── config/           # llm-config：配置分层、键闭集、路径解析、App 身份、TOML 解析器
    ├── cutile/           # 机制层（lib）：协商 → 拼接 → emit → 编译执行（bin 在根包 `src/bin/cutile.rs`）
    ├── einsum/           # 语义层：一条求和表达式 → flops 与最少流量（op 的代价来源）
    ├── op/               # 声明层：ops/<name>.{toml,k} + 算子语义（表达式/样本/标量参考）
    ├── prof/             # 档案层：devices/<vendor>.toml + 能力词汇表
    ├── prim/             # 片段层：<vendor>/<prim>.<ext> + witness
    └── …                 # 其余 23 个成员见上表，每个 crate 的 //! 文档写明
                          # “拥有什么 / 不拥有什么 / 依赖谁 / 被谁依赖”
```

`cutile` 的 P0 纵切面已经落地（声明 → 协商 → tile 取值 → 片段拼接 → emit → 编译执行 → 对拍）：

```bash
cargo run --bin cutile -- doctor --op gemv_i4 --device auto            # 协商链 + 布局声明 + 缓存指纹
cargo run --bin cutile -- doctor --op gemv_i4 --device cpu --why 83a3ecdb31cd1225   # 该键为什么没命中
cargo run --bin cutile -- verify --op gemv_i4 --devices cpu            # 编译执行并与标量参考对拍（L2）
cargo run --bin cutile -- bench  --op gemv_i4 --repeat 20              # 形状缺省取算子的生产形状
cargo run --bin cutile -- cache gc --limit 1073741824                  # 按 LRU 回收到上限以内（缺省 2 GiB）
```

| 设备 | 现状 | 说明 |
|---|---|---|
| `cpu` | **L2** | 生成的 C 由 `cc` 编译执行，与 `crates/op` 的标量参考对拍 `max_rel_err = 0` |
| `nvidia` / `intel` | L1 | 仅有档案与片段声明；`verify` 会**失败**（缺设备即失败），协商拒绝未验证能力 |

## 配置

运行时的值与目录只有一处来源——`llm-config`，默认值面向**发布后的二进制**：不依赖仓库布局，也不依赖启动目录。
配置的**机制**（分层、来源、键闭集）与**身份**（目录名、文件名、环境前缀）是分开的：
`Settings<A: App>` 的类型参数只决定名字，`Cutile` 是默认身份。其它层要读自己的配置时写
`Settings::<你的应用>::load()`，不需要复制一层分层逻辑（见 [ADR 0006](docs/adr/0006-config-mechanism-vs-identity.md)）。

```text
内置默认 < 用户级配置文件 < CUTILE_* 环境变量 < --set key=value
```

| 平台 | 配置文件 | 缓存根 |
|---|---|---|
| macOS | `~/Library/Application Support/cutile/cutile.toml` | `~/Library/Caches/cutile` |
| Linux/BSD | `${XDG_CONFIG_HOME:-~/.config}/cutile/cutile.toml` | `${XDG_CACHE_HOME:-~/.cache}/cutile` |
| Windows | `%APPDATA%\cutile\cutile.toml` | `%LOCALAPPDATA%\cutile` |

| 键 | 环境变量 | 默认值 |
|---|---|---|
| `cache.dir` | `CUTILE_CACHE_DIR`（兼容 `CUTILE_CACHE`） | `<缓存根>/cache` |
| `work.dir` | `CUTILE_WORK_DIR` | `<缓存根>/work` |
| `data.root` | `CUTILE_DATA_DIR` | 未设置（用编译期内嵌的 `crates/op/ops/`、`crates/prof/devices/` 与 `crates/prim/<vendor>/`） |

仓库里**不放**配置文件：默认值的唯一来源是 `crates/config/src/paths.rs`；出现未登记的键会直接报错而不是被忽略。

```bash
cargo run --bin cutile -- doctor --op gemv_i4 --device cpu    # [配置] 段逐键标注来源
CUTILE_CACHE_DIR="$PWD/.cache" cargo run --bin cutile -- verify --devices cpu
cargo run --bin cutile -- --set work.dir=/tmp/work verify --devices cpu
```

## 新增成员

```bash
cargo new --lib crates/<name>       # 或 cargo new --bin
```

然后按 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 第五节做三件事：

1. `Cargo.toml`：`version/edition/rust-version/license/repository/readme` 全部 `.workspace = true`，
   写 `description`、`publish = false`、`[lints] workspace = true`；
2. `src/lib.rs` 第一行写 `//!` 文档：拥有什么、不拥有什么、依赖谁、被谁依赖；
3. 在 `docs/ARCHITECTURE.md` 第一节的层表里加上它，并让它至少有一条依赖边——删掉它会有人编译
   失败，才叫模块。

三步做完之后，新成员才真正进入架构。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 分层、依赖规则、所有权表、如何被守住 |
| [`docs/adr/`](docs/adr/) | 架构决策记录（分层序、单一主人、kernel/运行时类型、占位策略、注解即契约、配置机制与身份分离） |
| [`crates/cutile/DESIGN.md`](crates/cutile/DESIGN.md) | kernel 层机制：三层数据、闭集、无 IR lowering、证据分级 |
| [`CHANGELOG.md`](CHANGELOG.md) | 变更记录 |

## 许可证

以 MIT 或 Apache-2.0 双许可分发，任选其一。详见 [LICENSE-MIT](LICENSE-MIT) 与 [LICENSE-APACHE](LICENSE-APACHE)。

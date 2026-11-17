# llm

用 Rust 编写的配置驱动算子训练推理框架：kernel 由声明与档案驱动（`cutile`），上层按层组装成推理与训练链路，入口组合根是 `llm-cli`（bin：`llm` / `cutile`）。

本仓库是一个纯 Cargo workspace，共 24 个 crate，全部位于 `crates/*`；入口二进制归 `llm-cli` 包，根 `Cargo.toml` 只声明 workspace 元数据、依赖版本与 lint 策略。**模块的层级与依赖方向**写在
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，新增成员时按那里的三步登记。

## 快速开始

需要 Rust 1.100 或更高版本；`rust-toolchain.toml` 已把工具链固定为 `nightly-2026-09-01`（rustc 1.100.0-nightly），
只要本机装了 `rustup`，进入本目录即自动切换。1.100 正式版发布后，把该文件的 `channel` 改成 `"1.100"` 即可切回正式版。

```bash
cargo run -p llm-cli --bin llm                              # 组合根：默认命令打印各层摘要
cargo run -p llm-cli --bin llm -- help                      # 命令行用法（文法解析与编排都在 `llm-cli`）
cargo run -p llm-cli --bin llm -- status --weights model.safetensors   # 读真实 safetensors：条目 → 层栈 → 模型下界
cargo run -p llm-cli --bin llm -- run --weights model.safetensors      # 用真实权重算一次：每层一条内核、与标量参考对拍
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
#[nn::spec(name = "lm_head", out = [4096], ops = ["gemv_i4"])]
struct LmHead;   // 生成 impl nn::Contract + 一条“声明仍与 op 数据一致”的测试
```

注意：根目录是纯 workspace（没有自己的包），不带 `-p` 的 `cargo build`、`cargo test`、`cargo clippy`、`cargo doc`
默认作用于整个 workspace；入口二进制需要用 `-p llm-cli --bin llm|cutile` 指定。

## 分层

完整表格、所有权表与边清单见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：

| 层 | 成员 | 职责 |
|---|---|---|
| L1 基础设施 | `diagnostic` `dtype` `tool` | 统一错误类型与声明数据树（`data`：构建期语料在运行期的形状，没有解析器）、精度事实源（标量/块量化/微缩放三轴）、静态门与回退判定 |
| L2 数据层 | `prof` `op` `prim` | 设备档案与能力闭集、算子声明与模板（22 条，`op::REGISTRY` 是唯一清单）及其求和代价（`op/src/einsum/`）、原语片段与见证（纯数据） |
| L3 内核 | `cutile` | 协商 → 拼接 → emit → 编译执行与见证执行；`doctor` / `verify` / `bench` / `execution` 四条命令共用一条流水线（`pipeline`），报告本身只是数据、排版在驱动方；设计契约见 [cutile README](crates/cutile/README.md) |
| L4 运行时 | `tensor` `storage` `tokenizer` | 张量元数据、布局搬运、文本边界 |
| L5 模型与训练 | `weights` `autograd` `data` `nn` `shard` | 权重结构、safetensors 读取与量化分组推断、反向图、样本、层定义、切分拓扑 |
| L6 调度与服务 | `train` `engine` `serve` | 训练循环状态与检查点格式（`train/src/snapshot/`）、连续批与序列及页账与提交契约（`engine/src/{kv,exec}.rs`）、端点契约与其线格式（`serve/src/proto/`） |
| L7 入口 | `cli`（bin：`llm` / `cutile`） | 组合根：命令行文法与配置解析，把上面各层连起来装配、取值解析与人可读排版；编排入口不定义新类型；`status` 算下界，`run` 用真实权重执行 |

唯一一条依赖规则：**依赖只能指向更低的层（同层只能依赖更靠左的成员）**。因此向上依赖不可能通过、
环不可能出现、数据层永远不会知道机制（`op` / `prof` / `prim` 不依赖 `cutile`）。
这条规则由 `crates/testkit/tests/architecture.rs` 在 `cargo test` 里对照 `cargo metadata` 自动核对。

`annotation`（编译期注解）、`fuzz`（模糊测试契约）、`benches`（集中测量容器）与 `testkit`（测试夹具与工作区契约门禁）是**工具成员**：它们不计入运行时分层，
也不参与“至少一条依赖边”的统计。

```text
cli ──▶ engine ──▶ tensor ──▶ diagnostic
  │        └─────▶ diagnostic               ▲
  ├──▶ serve ──▶ tokenizer ────────────────┘
  ├──▶ nn ──▶ weights / op / annotation
  ├──▶ train ──▶ nn ──▶ op
  │             ├▶ autograd ─┤
  │             ├▶ data ─────┤
  │             └▶ shard ────┘
  ├──▶ weights ──▶ storage ──┘
  ├──▶ op / prof / tool
  └──▶ cutile ──┬─▶ op / prof / prim
                └─▶ diagnostic（错误类型 + toml）
```

## 目录结构

```
.
├── Cargo.toml            # 纯 workspace 根：[workspace]/[workspace.package]/[lints]/[profile.*]，无 package/bin
├── docs/
│   ├── ARCHITECTURE.md   # 分层、依赖规则、所有权表、新增成员三步
│   └── adr/              # 0001 分层序 / 0002 单一主人 / 0003 kernel 与运行时类型
│                         # 0004 占位策略 / 0005 注解即契约 / 0006 配置机制与身份分离（身份段已被 0020 取代）
│                         # 0020 设置层吸收进 cutile / 0021 TOML 语法层并进 diagnostic / 0022 语料改构建期解析
│                         # 0007 代价来自一条求和表达式 / 0008 层下界来自算子语义
│                         # 0013 加第二条算子只加数据（稠密 f32 投影）
│                         # 0009 真实权重 → 层栈 → 模型下界
│                         # 0010 借出 vs 载入（权重文件的两条读取契约）
│                         # 0011 下界按方向分账、按文件说格式，并允许“无法计价”
│                         # 0012 量化权重的分组与宽度必须同源
│                         # 0014 真实权重执行：同一件内核，换一份样本
└── crates/
    ├── annotation/       # #[spec] 属性宏：注解 → 校验过的 Contract 实现（零外部依赖）
    ├── cli/              # 组合根（lib + 两个 bin）：文法解析、`cmd/` 编排入口；`llm` 入口在 `src/main.rs`，`cutile` bin 在 `src/cutile_bin.rs`
    ├── cutile/           # 机制层（lib）：协商 → 拼接 → emit → 编译执行；命令侧报告在 `cli` 包
    ├── op/               # 声明层：ops/<name>.{toml,k} + 算子语义；einsum/ 是求和代价的来源
    ├── prof/             # 档案层：devices/<vendor>.toml + 能力词汇表
    ├── prim/             # 片段层：<vendor>/<prim>.<ext> + witness
    └── …                 # 其余 18 个成员见上表，每个 crate 的 //! 文档写明
                          # “拥有什么 / 不拥有什么 / 依赖谁 / 被谁依赖”
```

`cutile` 的 P0 纵切面已经落地（声明 → 协商 → tile 取值 → 片段拼接 → emit → 编译执行 → 对拍）：

```bash
cargo run -p llm-cli --bin cutile -- doctor --op gemv_i4 --device auto            # 协商链 + 布局声明 + 缓存指纹
cargo run -p llm-cli --bin cutile -- doctor --op gemv_i4 --device cpu --why 83a3ecdb31cd1225   # 该键为什么没命中
cargo run -p llm-cli --bin cutile -- verify --op gemv_i4 --devices cpu            # 编译执行并与标量参考对拍（L2）
cargo run -p llm-cli --bin cutile -- verify --op gemv_f32 --devices cpu           # 稠密算子同一套流水线（L2）
cargo run -p llm-cli --bin cutile -- bench  --op gemv_i4 --repeat 20              # 形状缺省取算子的生产形状
cargo run -p llm-cli --bin cutile -- cache gc --limit 1073741824                  # 按 LRU 回收到上限以内（缺省 2 GiB）
```

| 设备 | 现状 | 说明 |
|---|---|---|
| `cpu` | **L2** | 生成的 C 由 `cc` 编译执行，与 `crates/op` 的标量参考对拍 `max_rel_err = 0` |
| `nvidia` / `intel` / `metal` | L1 | 仅有档案、片段与见证；`verify` 会**失败**（缺设备即失败），协商拒绝未验证能力 |

`metal` 的设备侧代码还能在构建期编成 `metallib` 并嵌进二进制：`cargo build -p llm-cli --features device-image`
（默认是源码调试，只带片段与见证）。哪台设备带了什么由 `prim::devices()` 回答，`cargo run -p llm-cli --bin llm`
的 `片段层 prim` 一行会逐台列出 `=源码` / `=metallib N B/M 个符号`；`CUTILE_REQUIRE_IMAGE=metal`
把"缺镜像"升级成构建失败。细节见 [`crates/prim/README.md`](crates/prim/README.md)。

## 配置

运行时的值与目录只有一处来源——`cli::settings`（`crates/cli/src/settings.rs`），默认值面向**发布后的二进制**：
不依赖仓库布局，也不依赖启动目录。它以前是独立成员 `llm-config` 的一半（
[ADR 0020](docs/adr/0020-settings-layer-absorption.md)），后来并入 `cutile`；第四层（用户级配置文件）
在 [ADR 0022](docs/adr/0022-build-time-corpora.md) 随运行期 TOML 解析器一起删除，剩下的策略再上移到
命令行层（[ADR 0023](docs/adr/0023-settings-to-cli.md)）：默认目录是部署策略、`--set` 是命令行文法，
都不是内核的职责。内核只收 `cutile::dirs::Dirs` 这份纯数据。**没有配置文件**：只有下面三层。

```text
内置默认 < CUTILE_* 环境变量 < --set key=value
```

| 平台 | 缓存根 |
|---|---|
| macOS | `~/Library/Caches/cutile` |
| Linux/BSD | `${XDG_CACHE_HOME:-~/.cache}/cutile` |
| Windows | `%LOCALAPPDATA%\cutile` |

| 键 | 环境变量 | 默认值 |
|---|---|---|
| `cache.dir` | `CUTILE_CACHE_DIR`（兼容 `CUTILE_CACHE`） | `<缓存根>/cache` |
| `work.dir` | `CUTILE_WORK_DIR` | `<缓存根>/work` |
| `data.root` | `CUTILE_DATA_DIR` | 未设置（用编译期内嵌的 `crates/op/ops/`、`crates/prof/devices/` 与 `crates/prim/<vendor>/`） |

**没有配置文件**：默认值的唯一来源是 `crates/cli/src/settings.rs`，要偏离默认只能用上面的环境变量或
`--set`；`--set` 里出现未登记的键会直接报错而不是被忽略。

```bash
cargo run -p llm-cli --bin cutile -- doctor --op gemv_i4 --device cpu    # [配置] 段逐键标注来源
CUTILE_CACHE_DIR="$PWD/.cache" cargo run -p llm-cli --bin cutile -- verify --devices cpu
cargo run -p llm-cli --bin cutile -- --set work.dir=/tmp/work verify --devices cpu
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
| [`docs/adr/`](docs/adr/) | 架构决策记录（分层序、单一主人、kernel/运行时类型、占位策略、注解即契约、配置机制与身份分离、层下界、真实权重、借出与载入、量化分组、真实权重执行、设置层吸收、TOML 语法层归属、语料改构建期解析） |
| [`crates/cutile/README.md`](crates/cutile/README.md) | kernel 层机制：三层数据、闭集、无 IR lowering、证据分级 |
| [`docs/cutile-analysis.md`](docs/cutile-analysis.md) | `cutile` 25 模块清单与职责判决、CPU 链路取证、五设备 × 十六原语矩阵与 Metal 缺口 |
| [`docs/backend-runtime-design.md`](docs/backend-runtime-design.md) | 推理后端整仓分析：现状可复用面、P1→P2→P3 缺口、serve 复用落点与边表、trait 形状裁决与分阶段出口门 |
| [`CHANGELOG.md`](CHANGELOG.md) | 变更记录 |

## 许可证

以 MIT 或 Apache-2.0 双许可分发，任选其一。详见 [LICENSE-MIT](LICENSE-MIT) 与 [LICENSE-APACHE](LICENSE-APACHE)。

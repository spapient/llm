# llm

用 Rust 编写的配置驱动算子训练推理框架：kernel 由声明与档案驱动（`cutile`），上层按层组装成推理与
训练链路，入口组合根是 `llm-cli`（bin：`llm` / `cutile`）。

仓库是一个纯 Cargo workspace，成员全部位于 `crates/*`，入口二进制归 `llm-cli` 包；根 `Cargo.toml`
只声明 workspace 元数据、依赖版本与 lint 策略。**成员清单、层级与依赖方向**以
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 为唯一来源——那份表由 `cargo test` 对照
`cargo metadata` 与源码自动核对，本文不复述。

## 快速开始

需要 Rust 1.99 或更高版本。`rust-toolchain.toml` 钉的是**已发布**的 stable 通道，只要本机装了
`rustup`，进入本目录即自动切换，`cargo build` 直接能过——不再需要 nightly。手写 SIMD 车道与
Apple SME 外积这两条通道仍只在 nightly 上编译（`#[cfg(nightly)]`，见 `crates/gemm/build.rs`），
stable 上自动走等价退路，语义一致、只是没有那两条指令路径。

```bash
cargo run -p llm-cli --bin llm                              # 组合根：默认命令打印各层摘要
cargo run -p llm-cli --bin llm -- help                      # 命令行用法（文法解析与编排都在 `llm-cli`）
cargo run -p llm-cli --bin llm -- status --weights model.safetensors   # 读真实 safetensors：条目 → 层栈 → 模型下界
cargo run -p llm-cli --bin llm -- run --weights model.safetensors      # 用真实权重算一次：每层一条内核、与标量参考对拍
cargo build --workspace                       # 编译整个 workspace
cargo test --workspace                        # 跑全部成员的测试与 doctest
```

根目录没有自己的包，不带 `-p` 的 `cargo build` / `test` / `clippy` / `doc` 默认作用于整个
workspace；入口二进制要用 `-p llm-cli --bin llm|cutile` 指定。

| 命令 | 作用 |
|---|---|
| `cargo +nightly fmt --all` | 按 `rustfmt.toml` 格式化所有成员（**必须显式 `+nightly`**：排版基线是 nightly 的 rustfmt，stable 的会把部分文件展开并顶破 `FILE_MAX`/`Ccn` 门） |
| `cargo clippy --workspace --all-targets --all-features` | 静态检查（lint 策略见根 `Cargo.toml` 的 `[workspace.lints]`） |
| `cargo test --workspace` | 运行测试 |
| `cargo doc --workspace --no-deps --open` | 生成并打开文档 |

## 分层与依赖

七层从下到上：L1 基础设施（`diagnostic` `rng` `dtype` `wire` `tool`）→ L2 数据（`prof` `op` `prim`）
→ L3 内核（`cutile`）→ L4 运行时（`tensor` `storage` `tokenizer`）→ L5 模型与训练（`weights`
`autograd` `nn` `shard`）→ L6 调度与服务（`train` `engine` `serve`）→ L7 入口（`cli`）。
`annotation` / `fuzz` / `benchmarks` / `testkit` 是工具成员，不计入运行时分层。

唯一一条依赖规则：**依赖只能指向更低的层（同层只能依赖更靠左的成员）**。因此向上依赖不可能通过、
环不可能出现、数据层永远不会知道机制（`op` / `prof` / `prim` 不依赖 `cutile`）。各层职责、边清单、
所有权表与新增成员三步都在 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## kernel 纵切面

`cutile` 的 P0 纵切面已落地（声明 → 协商 → tile 取值 → 片段拼接 → emit → 编译执行 → 对拍）：

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
| `cpu` | **L2** | 生成的 C 由 `cc` 编译执行，与 `crates/op` 的标量参考对拍 `max_rel_err = 0`（通道：语义基准——串行是这一路的定义，并行形态由设备通道证） |
| `metal` | **L2** | 真机实测：36 个算子**全部**按声明发射并行网格（`[grid]` / `[chunk]`），无一回落单线程；`gemv_i4` 4096² 的 `bench --repeat 50` 连跑三次为 p50 0.049–0.059 ms、177–213 GB/s、带宽达成率 57.6%–69.3%（峰值 307.2 GB/s）——**仍低于 70% 门槛，按 §16 承认需要特化代码**（单次采样曾到 0.044 ms / 77.6%，波动说明这是偏乐观样本，不采纳） |
| `musa` | **L2**（上一轮实测） | MTT S4000（mp_22）曾 36/36 经 `mcc` 编成执行、`max_rel_err ≤ 3.88e-7`；本轮那台机器上**没有该设备**，`verify` 按「缺设备即失败」如实缺席，不记为通过 |
| `nvidia` / `intel` | L1 | 仅有档案、片段与见证；`verify` 会**失败**（缺设备即失败），协商拒绝未验证能力 |

`metal` 的设备侧代码还能在构建期编成 `metallib` 并嵌进二进制：`cargo build -p llm-cli --features device-image`
（默认是源码调试，只带片段与见证）。哪台设备带了什么由 `prim::devices()` 回答，
`cargo run -p llm-cli --bin llm` 的 `片段层 prim` 一行会逐台列出 `=源码` / `=metallib N B/M 个符号`；
`CUTILE_REQUIRE_IMAGE=metal` 把"缺镜像"升级成构建失败。细节见 [`crates/prim/README.md`](crates/prim/README.md)。

## 配置

运行时的值与目录只有一处来源——`cli::settings`（`crates/cli/src/settings.rs`），默认值面向**发布后的
二进制**：不依赖仓库布局，也不依赖启动目录。策略归命令行层而不归内核：语料在构建期解析完
（[ADR 0022](docs/adr/0022-build-time-corpora.md)），分层、来源与键表上移到 `cli`
（[ADR 0023](docs/adr/0023-settings-to-cli.md)），配置文件只是其中的部署形态
（[ADR 0048](docs/adr/0048-runtime-config-file.md)）。默认目录是部署策略、
`--set` 是命令行文法、配置文件是部署形态，都不是内核的职责——内核只收 `cutile::dirs::Dirs` 这份纯数据。

```text
内置默认 < 配置文件 cutile.toml < CUTILE_* 环境变量 < --set key=value
```

配置文件发现顺序：`--config <path>` 或 `CUTILE_CONFIG` → 当前目录 `./cutile.toml` → 平台用户目录
（macOS/Linux `~/.config/cutile/config.toml`、Windows `%APPDATA%\cutile\config.toml`）；只读第一份命中。

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
| `cache.limit_bytes` | — | 2 GiB |

默认值的唯一来源是 `crates/cli/src/settings.rs`；要偏离默认用配置文件、环境变量或 `--set`，
配置文件与 `--set` 里出现未登记的键会直接报错而不是被忽略。

```bash
cargo run -p llm-cli --bin cutile -- doctor --op gemv_i4 --device cpu    # [配置] 段逐键标注来源
CUTILE_CACHE_DIR="$PWD/.cache" cargo run -p llm-cli --bin cutile -- verify --devices cpu
cargo run -p llm-cli --bin cutile -- --set work.dir=/tmp/work verify --devices cpu
cargo run -p llm-cli --bin cutile -- --config ./cutile.toml verify --devices cpu
```

## 目录结构

```
.
├── Cargo.toml            # 纯 workspace 根：[workspace]/[workspace.package]/[lints]/[profile.*]，无 package/bin
├── docs/                 # 见 docs/README.md（架构契约 / 规格 / ADR）
└── crates/
    ├── annotation/       # #[spec] 属性宏：注解 → 校验过的 Contract 实现（零外部依赖）
    ├── benchmarks/      # 集中 criterion 微基准与回退判定，不进主依赖图
    ├── cli/              # 组合根（lib + 两个 bin）：文法解析、`cmd/` 编排入口
    ├── cutile/           # 机制层：协商 → 拼接 → emit → 编译执行；命令侧报告在 `cli` 包
    ├── op/               # 声明层：ops/<name>.{toml,k} + 算子语义；einsum/ 是求和代价的来源
    ├── prof/             # 档案层：devices/<vendor>.toml + 能力词汇表
    ├── prim/             # 片段层：<vendor>/<prim>.<ext> + witness
    └── …                 # 其余成员见 ARCHITECTURE 层表，每个 crate 的 //! 文档写明
                          # “拥有什么 / 不拥有什么 / 依赖谁 / 被谁依赖”
```

## 新增成员

```bash
cargo new --lib crates/<name>       # 或 cargo new --bin
```

然后按 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 第五节做三件事：`Cargo.toml` 全用
`.workspace = true` 并写 `description` / `publish = false` / `[lints] workspace = true`；
`src/lib.rs` 第一行写 `//!` 文档（拥有什么、不拥有什么、依赖谁、被谁依赖）；在层表里登记并让它
至少有一条依赖边——删掉它会有人编译失败，才叫模块。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/README.md`](docs/README.md) | 文档地图与写作规则（一份事实只写一次） |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 分层、依赖规则、所有权表、如何被守住 |
| [`docs/lld.md`](docs/lld.md) | 数学建模与算子判据（`E*` 出口、规则 `0.x`） |
| [`docs/adr/`](docs/adr/) | 架构决策记录，索引见 `docs/README.md` |
| [`crates/cutile/README.md`](crates/cutile/README.md) | kernel 层机制：三层数据、闭集、无 IR lowering、证据分级 |
| [`CHANGELOG.md`](CHANGELOG.md) | 变更记录 |

## 许可证

以 MIT 或 Apache-2.0 双许可分发，任选其一。详见 [LICENSE-MIT](LICENSE-MIT) 与 [LICENSE-APACHE](LICENSE-APACHE)。

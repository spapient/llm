# llm

用 Rust 编写的配置驱动算子训练推理框架：kernel 由声明与档案驱动（`cutile`），上层按层组装成推理与
训练链路，入口组合根是 `llm-cli`（bin：`llm` / `cutile`）。

仓库是一个纯 Cargo workspace，成员全部位于 `crates/*`，入口二进制归 `llm-cli` 包；根 `Cargo.toml`
只声明 workspace 元数据、依赖版本与 lint 策略。**成员清单、层级与依赖方向**以
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 为唯一来源——那份表由 `cargo test` 对照
`cargo metadata` 与源码自动核对，本文不复述。

## 快速开始

需要 Rust 1.100.0 或更高版本。`rust-toolchain.toml` 钉的是固定的 stable 版本 1.100.0，只要本机装了
`rustup`，进入本目录即自动切换，`cargo build` 直接能过——不再需要 nightly。手写 SIMD 车道与
Apple SME 外积这两条通道仍只在 nightly 上编译（`#[cfg(nightly)]`，见 `crates/nn/build.rs`），
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

七层从下到上：L1 基础设施（`diagnostic` `dtype` `wire` `tools`）→ L2 数据（`prof`）
→ L3 内核（`cutile`）→ L4 运行时（`tensor` `storage` `tokenizer`）→ L5 模型与训练（`weights`
`autograd` `nn` `algorithm` `shard`）→ L6 调度与服务（`train` `engine` `device` `serve`）→ L7 入口（`cli`）。
`annotation` / `fuzz` / `benchmarks` / `testkit` 是工具成员，不计入运行时分层。

唯一一条依赖规则：**依赖只能指向更低的层（同层只能依赖更靠左的成员）**。因此向上依赖不可能通过、
环不可能出现、数据层永远不会知道机制（`prof` 不依赖 `cutile`）。各层职责、边清单、
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
| `cpu` | **L2** | 生成的 C 由 `cc` 编译执行，与 `crates/nn` 的标量参考对拍 `max_rel_err = 0`（通道：语义基准——串行是这一路的定义，并行形态由设备通道证） |
| `metal` | **L2** | 真机实测：36 个算子**全部**按声明发射并行网格（`[grid]` / `[chunk]`），无一回落单线程；`gemv_i4` 4096² 的 `bench --repeat 50` 连跑三次为 p50 0.049–0.059 ms、177–213 GB/s、带宽达成率 57.6%–69.3%（峰值 307.2 GB/s）——**仍低于 70% 门槛，按 §16 承认需要特化代码**（单次采样曾到 0.044 ms / 77.6%，波动说明这是偏乐观样本，不采纳） |
| `musa` | **L2**（上一轮实测） | MTT S4000（mp_22）曾 36/36 经 `mcc` 编成执行、`max_rel_err ≤ 3.88e-7`；本轮那台机器上**没有该设备**，`verify` 按「缺设备即失败」如实缺席，不记为通过 |
| `nvidia` | L1 | 有档案、片段与见证；生产态 `verify` 仍待统一运行时装配 |
| `intel` | **L2（UHD 750 见证层）** | Windows + OpenCL 真机 17/17 条供给执行对拍、68/68 份翻译单元编译通过；含真实 DP4A、全局原子与子组操作。生产态 `verify` 仍待统一运行时装配，不以见证通过冒充全模型通过 |

`metal` 的设备侧代码还能在构建期编成 `metallib` 并嵌进二进制：`cargo build -p llm-cli --features device-image`
（默认是源码调试，只带片段与见证）。哪台设备带了什么由 `prof::devices()` 回答，
`cargo run -p llm-cli --bin llm` 的 `片段层 prim` 一行会逐台列出 `=源码` / `=metallib N B/M 个符号`；
`CUTILE_REQUIRE_IMAGE=metal` 把"缺镜像"升级成构建失败。细节见 [`crates/prof/README.md`](crates/prof/README.md)。

## 形状与视图代数（`tensor`）

`tensor` 只有一个类型族：`Tensor<B, R, K>`，三个泛型位各管一件事——`B: Store` 是元素存在哪
（借来的 `&[K]` / 持有的 `Owned<K>`），`R: Rank` 是秩从哪来（`[usize; D]` 编译期定长 /
`Vec<usize>` 运行期长度），`K: Element` 是元素是什么（`bool` / `u8`–`u64` / `i8`–`i64` /
`f32` / `f64` / `f16` / `bf16`，精度是类型常量而不是字段）。这与 Burn 的 `Tensor<B, const D, K>`
同形：形状、元素数、可达性与溢出在构造期一次拦下，做不到的调用在编译期就过不去
（`t.select::<[usize; 1]>(0, 3)` 写出目标秩，换成别的秩编译不过）。

布局是 `(起点偏移, 各轴长度, 各轴步长)` 三元组，整套形状算子只做 O(轴数) 的整数运算、与存储
形态无关：`broadcast` 用步长 0 表达广播，`narrow` / `slice` 前移起点，`chunks` / `split` /
`unbind` 借同一存储分段，`select` / `squeeze` / `unsqueeze` / `flatten` / `diagonal` 改秩，
`transpose` / `permute` / `movedim` 换轴序，`slice_fill` / `slice_assign` 就地写自己的缓冲；
带步长切片（`Slice::stepped`，对应 Burn 的 `s![..;2]`）同样零拷贝。只有 `flip` 按镜像下标重算
并返回自持张量——无符号步长表达不出倒走。偏移、可达性、连续性与取行只有 `layout` 一份实现，
连续步长只有 `shape::accumulate` 一份定义，两种秩、两种存储因此不可能漂移。

它拥有三样全仓唯一事实源，另有两处比 Burn 更硬：

- **`Shape`** —— 各轴长度；`numel()` 是饱和乘积（长度可能来自不受信任的权重文件头，溢出饱和
  而非回绕），`element_bytes(dtype)` 按位向上取整，`u4` 的 `[2, 3]` 只占 3 字节、`f32` 占 24 字节。
- **`AxisLen`** —— 轴长的 `usize` / `u64` 双口径，形状构造不为两种宽度各写一遍。
- **`DType` 的再导出** —— 元素类型闭集的唯一源在 L1 `dtype`（24 个变体、`Class` 数值类别、位宽），
  本层只转出名字，不复制任何分类。
- **亚字节精度是一等公民** —— `U1`/`U2`/`U4`/`I4`/`F4` 与 `F32` 同表，字节数由
  `DType::bytes_for` 位级进位；Burn 的 `Element` 要求 `size_of` 整除，亚字节只能靠外部打包约定。
- **能力层是分类表的一列** —— `Class` 住在 `DTypeSpec` 行表里，`is_float`/`is_integer` 是它的
  `const fn` 投影；Burn 的 `Kind` 每多一档就多一层泛型参数。

`B` 是**存储**而不是后端算子集，`R` 是 `const D` 的**超集**，这两处是刻意的分工（ADR 0053 /
ADR 0046）；其余布局与类型能力与 Burn 对齐，并在亚字节与分类表上更宽。逐维打分与可复核证据见
[`docs/COMPARISON-llm-llmu-burn.md`](docs/COMPARISON-llm-llmu-burn.md) §3.3 与 §16；源码见
[`crates/tensor/src/lib.rs`](crates/tensor/src/lib.rs) 的模块文档。

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
| `data.root` | `CUTILE_DATA_DIR` | 未设置（用编译期内嵌的 `crates/prof/src/ops`、`crates/prof/devices/` 与 `crates/prof/src/<vendor>/`） |
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

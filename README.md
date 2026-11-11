# llm

用 Rust 编写的大语言模型运行时。

本仓库是一个 Cargo workspace：根包 `llm`（可执行程序）与 `crates/cutile`（库）共同构成一个构建单元，元数据、依赖版本与 lint 策略统一在根 `Cargo.toml` 中声明。

## 快速开始

需要 Rust 1.100 或更高版本；`rust-toolchain.toml` 已把工具链固定为 `nightly-2026-09-01`（rustc 1.100.0-nightly），只要本机装了 `rustup`，进入本目录即自动切换。1.100 正式版发布后，把该文件的 `channel` 改成 `"1.100"` 即可切回正式版，其它配置无需改动。

```bash
cargo run -p llm         # 编译并运行根包
cargo build --workspace  # 编译整个 workspace
cargo test --workspace   # 跑全部成员的测试与 doctest
```

## 常用命令

| 命令 | 作用 |
|---|---|
| `cargo fmt --all` | 按 `rustfmt.toml` 格式化所有成员 |
| `cargo clippy --workspace --all-targets --all-features` | 静态检查（lint 策略见根 `Cargo.toml` 的 `[workspace.lints]`） |
| `cargo test --workspace` | 运行测试 |
| `cargo doc --workspace --no-deps --open` | 生成并打开文档 |

注意：因为根目录本身就是一个包，不带 `-p` / `--workspace` 的 `cargo build`、`cargo test`、`cargo clippy`、`cargo doc` 只作用于 `llm`，不会覆盖 `crates/` 下的成员；只有 `cargo fmt --all` 天生遍历全部成员。

CI（`.github/workflows/ci.yml`）会把格式差异、clippy 告警和 rustdoc 告警直接判定为失败，本地推之前先跑一遍即可。

## 目录结构

```
.
├── Cargo.toml            # workspace 声明 + 根包 llm 的元数据，以及共享的
│                         # [workspace.package] / [workspace.lints] / [profile.*]
├── Cargo.lock            # 整个 workspace 共用一份锁文件
├── src/main.rs           # 根包 llm 的程序入口（bin）
├── crates/
│   ├── config/           # 库成员 config（lib）：配置层占位
│   └── cutile/           # 库成员 cutile（lib）+ bin：设备无关 kernel DSL
│       ├── DESIGN.md     # 设计文档（声明层 / 档案层 / 片段层、闭集与预算）
│       ├── ops/          # 声明层：gemv_i4.toml（骨架/轴/需求/布局/降级链）+ gemv_i4.k（模板）
│       ├── devices/      # 档案层：cpu.toml（L2）/ nvidia.toml、intel.toml（L1）
│       ├── prim/cpu/     # 片段层：9 个原语片段 + witness.c
│       ├── Cargo.toml    # 元数据与 lints 全部继承自 workspace
│       └── src/          # atom / prof / cap / layout / space / sch / lower / cache / plan / host
│                         # / reference / backend / probe / cli / builtin / toml + main.rs
├── rust-toolchain.toml
├── rustfmt.toml
├── clippy.toml           # clippy 策略（含 MSRV）
└── .github/workflows/ci.yml
```

## 成员与依赖方向

| 成员 | 类型 | 职责 |
|---|---|---|
| `llm`（根包） | bin | 运行时程序入口 |
| `cutile` | lib | 设备无关 kernel DSL（算子声明 + 设备档案 + 原语片段），设计见 [crates/cutile/DESIGN.md](crates/cutile/DESIGN.md) |

`cutile` 的 P0 纵切面已经落地（声明 → 协商 → tile 选择 → 片段拼接 → emit → 编译执行 → 对拍）：

```bash
cargo run -p cutile -- doctor --op gemv_i4 --device auto            # 协商链 + 布局契约 + 缓存指纹
cargo run -p cutile -- verify --op gemv_i4 --devices cpu            # 编译执行并与标量参考对拍（L2）
cargo run -p cutile -- bench  --op gemv_i4 --shape 4096x4096 --repeat 20
```

| 设备 | 现状 | 说明 |
|---|---|---|
| `cpu` | **L2** | 生成的 C 由 `cc` 编译执行，与 `reference.rs` 对拍 `max_rel_err = 0` |
| `nvidia` / `intel` | L1 | 仅有档案与片段声明；`verify` 会**失败**（缺设备即失败），协商拒绝未验证能力 |

`verify` 在没有任何 L2 供给的设备上返回非零码，这是刻意的（DESIGN §11.2 对"静默跳过"的修正）。

两者目前在代码层面**没有依赖关系**，属于平级成员。如果之后 `llm` 需要调用 `cutile`，在根 `Cargo.toml` 的 `[dependencies]` 中加一行即可：

```toml
[dependencies]
cutile = { path = "crates/cutile" }
```

## 新增成员

```bash
cargo new --lib crates/<name>
```

`crates/*` 已被 `members` 通配覆盖，新包只要把 `version`、`edition`、`rust-version`、`license` 写成 `.workspace = true` 并加上 `[lints] workspace = true`，就会自动继承本仓库的元数据与 lint 策略。

## 许可证

以 MIT 或 Apache-2.0 双许可分发，任选其一。详见 [LICENSE-MIT](LICENSE-MIT) 与 [LICENSE-APACHE](LICENSE-APACHE)。

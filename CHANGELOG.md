# Changelog

本文件的格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- `cutile` P0 纵切面实现（约 4,900 行 Rust + 数据）：`atom`（16 原语闭集 / 参数化能力签名 / 逐字段差异）、`prof`（档案 schema + 指纹规则 + L2 见证校验）、`cap`（协商 + 降级链 + 可解释诊断）、`layout`（4 构造子 / 正规形 / repack 计费）、`space`、`sch`、`lower`（词法 / 轴展开 / 片段拼接 / emit）、`cache`（FNV 指纹 + 锁 + 原子落盘）、`plan`（roofline）、`host`（分区即网格）、`reference`（标量参考 + 对拍口径）、`backend`（`cc` 编译执行）、`probe`、`builtin`、`toml`（手写受限解析）。
- 声明层与档案层数据：`ops/gemv_i4.toml` + `ops/gemv_i4.k`、`devices/{cpu,nvidia,intel}.toml`、`prim/cpu/`（9 段片段 + 见证）。
- CLI：`cutile doctor` / `verify` / `bench`（`src/main.rs`）；`verify` 在缺设备时返回非零码。
- `crates/config` 占位成员补齐元数据与 crate 文档，使 workspace 门禁可整体通过。
- 初始工程：`src/main.rs` 入口，`Cargo.toml` 元数据（双许可、`publish = false`）。
- 工程门禁配置：`rust-toolchain.toml`、`rustfmt.toml`、`clippy.toml`、`[lints]` 段、`.editorconfig`、`.gitattributes`。
- GitHub Actions CI：fmt、clippy、test（三平台 + MSRV）、doc 五个 job。

### Changed

- 设计文档同步实现细节：模块表补 `reference.rs` / `toml.rs` / `main.rs`，并明确 `axes`（tile 候选）与 `[[variants]]`（降级链）是两个维度。
- 设计文档重写为 v1.0《cutile DSL —— 设备无关的替换式 kernel DSL》：改为以设备无关为核心的三层声明（算子声明 / 设备档案 / 原语片段）+ 三组闭集（16 原语、4 布局、4 调度）+ 无 IR 的 lowering，并把预算拆成 P0 2,760 行 / P3 9,780 行两档。
- `README.md` 的成员表与目录树同步：`cutile` 职责改为“设备无关 kernel DSL”，并链接 `crates/cutile/DESIGN.md`。

- MSRV 提升到 1.100：`Cargo.toml` 的 `[workspace.package] rust-version`、`clippy.toml` 的 `msrv`、CI 的 `msrv` job 与 `README` 统一为 1.100。
- `rust-toolchain.toml` 的工具链改为 `nightly-2026-09-01`（rustc 1.100.0-nightly）：1.100 正式版发布前只能用该 nightly 构建，发布后把 `channel` 改为 `"1.100"` 即可。
- CI 的 fmt / clippy / test / doc 四个 job 改用 `nightly-2026-09-01` 工具链；`msrv` job 同步安装该 nightly 并验证 1.100 系列可构建。
- `repository` 统一声明为 `https://github.com/spapient/llm`：写在 `[workspace.package]` 中，并由 `llm` 与 `cutile` 各自以 `repository.workspace = true` 继承（workspace 级字段不会自动下发，成员不写就等于没有）。

- 仓库升级为 Cargo workspace：根 `Cargo.toml` 声明 `[workspace] members = ["crates/*"]`、`resolver = "3"`，元数据与 lint 策略上提到 `[workspace.package]` / `[workspace.lints]`，两个成员改为 `.workspace = true` 继承。
- CI 的 clippy / test / msrv / doc 命令统一改为 `--workspace`，此前根包存在时这些命令只覆盖 `llm`，漏掉 `crates/` 下的成员。
- `.gitignore` 的 `/target/` 改为 `target/`，避免子目录构建产物进入版本库。

### Removed

- `crates/cutile/Cargo.lock` 与 `crates/cutile/target/`：workspace 只保留根锁文件与根构建目录。

## [0.1.0] - 2026-11-11

### Added

- 工程创建。

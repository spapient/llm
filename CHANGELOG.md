# Changelog

本文件的格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 初始工程：`src/main.rs` 入口，`Cargo.toml` 元数据（双许可、`publish = false`）。
- 工程门禁配置：`rust-toolchain.toml`、`rustfmt.toml`、`clippy.toml`、`[lints]` 段、`.editorconfig`、`.gitattributes`。
- GitHub Actions CI：fmt、clippy、test（三平台 + MSRV）、doc 五个 job。

### Changed

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

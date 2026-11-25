#!/usr/bin/env bash
# 变异测试手动入口（cargo-mutants）：注入变异体，观察现有测试能否抓住。
#
# 定位：可选 / 手动，**不进** ci.sh 阻断序列，也不挂 CI 阻断 job——变异轮次多、编译执行成本
# 高，业界通行按需跑而非 PR 必跑（决策见 docs/adr/0057 §8）。配置键见 `.cargo/mutants.toml`。
#
# 用法：
#   scripts/mutants.sh                  # 全 workspace 生产 src 的变异体（慢）
#   scripts/mutants.sh -f src/path.rs   # 只看某个文件（cargo-mutants 原生参数透传）
#   scripts/mutants.sh --package tensor # 只看某个 crate
#   scripts/mutants.sh --list           # 只列出将生成的变异体，不跑测试
#
# 首次需安装：cargo install --locked cargo-mutants
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! cargo mutants --version >/dev/null 2>&1; then
  echo "未安装 cargo-mutants。请先运行：cargo install --locked cargo-mutants" >&2
  exit 2
fi

# 参数原样透传给 cargo-mutants；配置默认从 .cargo/mutants.toml 读取。
exec cargo mutants "$@"

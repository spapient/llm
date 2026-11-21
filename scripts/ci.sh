#!/usr/bin/env bash
# 本地复刻 `.github/workflows/ci.yml` 的门禁序列：任一阶段非零即停，输出阶段名便于定位。
# 用法：
#   scripts/ci.sh              # 全部门禁（格式 / clippy / 测试 / doc-test / 依赖卫生 / 架构 / 静态 / 命名 / 密度 / 基准编译）
#   scripts/ci.sh --coverage  # 额外跑 llvm-cov nextest 覆盖率采集（慢，默认不开）
#
# fuzz（libFuzzer + ASan）与跨平台矩阵只在 CI 跑；本地不复制其 release 构建与 30s 引导。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

step() {
    echo
    echo "==> $*"
}

COVERAGE=0
for arg in "$@"; do
    case "$arg" in
        --coverage) COVERAGE=1 ;;
        *) echo "未知参数: $arg" >&2; exit 2 ;;
    esac
done

step "rustfmt（格式检查，不改动文件）"
cargo fmt --all --check

step "clippy（-D warnings，correctness / complexity / perf 等组）"
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

step "测试（nextest 并行隔离运行；libfuzzer harness 已在 .config/nextest.toml 排除）"
cargo nextest run --workspace --all-features --locked

step "doc-test（文档块示例必须编得过、断言成立；nextest 不跑 doc-test）"
cargo test --doc --workspace --all-features --locked

step "依赖安全 / 许可证 / 来源 / 重复版本（cargo-deny）"
cargo deny check

step "未使用的依赖（cargo-machete）"
cargo machete

step "代码重复率（jscpd 硬门，阈值 0 = 零容忍，与输出目录均由 .config/.jscpd.json 自动发现）"
npx --yes jscpd

step "静态门（源文件圈复杂度 / 行数 / 命名 / unsafe 计数 / Call 门等硬阈值，ratchet：新增必进表、恶化必失败）"
cargo test -p llm-tools --test gate --locked

step "架构门（层表 / 边集 / 所有权 / 入口不定义类型 / 错误边界不回落字符串，与 ARCHITECTURE.md 单一来源比对）"
cargo test -p llm-testkit --test architecture --locked

step "命名门（下划线文件名 / 后缀残留 / 泛名函数 / Op 后缀，零命中，已清零保持阻断）"
bash scripts/name-lint.sh

step "密度预算（crates/*/src ≤90,000 行；R1–R2 报告不阻断，R3 起转阻断——阈值只收紧）"
bash scripts/loc-budget.sh || echo "⚠ 密度超预算（N6 维度，R3 起转阻断门）"

step "基准可编译校验（实测与基线判定在固定硬件上跑，跨机器绝对值不可比）"
cargo bench --workspace --all-features --locked --no-run

if [ "$COVERAGE" -eq 1 ]; then
    step "覆盖率采集（llvm-cov nextest）"
    cargo llvm-cov nextest --workspace --all-features --locked --lcov --output-path lcov.info
fi

echo
echo "==> 全部门禁通过"

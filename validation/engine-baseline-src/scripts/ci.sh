#!/usr/bin/env bash
# 本地复刻 `.github/workflows/` 的门禁序列：任一阶段非零即停，输出阶段名便于定位。
# 用法：
#   scripts/ci.sh              # 基础门禁与独立复杂度证据
#   scripts/ci.sh --coverage  # 额外跑 llvm-cov nextest 覆盖率采集（慢，默认不开）
#   scripts/ci.sh --perf      # 额外跑固定硬件上的基准实测与回归预算
#   scripts/ci.sh --fuzz      # 额外跑有界 libFuzzer + ASan（FUZZ_SECONDS 默认 5）
#   scripts/ci.sh --miri      # 额外跑 nightly Miri 纯计算核心
#
# 三平台、Metal 与完整夜间通道仍由 CI 覆盖。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

step() {
    echo
    echo "==> $*"
}

COVERAGE=0
PERF=0
FUZZ=0
MIRI=0
for arg in "$@"; do
    case "$arg" in
        --coverage) COVERAGE=1 ;;
        --perf) PERF=1 ;;
        --fuzz) FUZZ=1 ;;
        --miri) MIRI=1 ;;
        *) echo "未知参数: $arg" >&2; exit 2 ;;
    esac
done

step "rustfmt（格式检查，不改动文件）"
cargo fmt --all --check

step "clippy（-D warnings，correctness / complexity / perf 等组）"
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

step "Ruff（Python 缺陷、现代化、导入与格式；版本由 pyproject.toml 锁定）"
uv tool run ruff --version
uv tool run ruff check validation
uv tool run ruff format --check validation

step "cloc（跨语言行数与逐文件报告；业务源码与验证脚本全量统计）"
cloc --version
mkdir -p target/cloc
cloc --config .config/cloc/options.txt crates validation scripts

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

step "独立复杂度证据（Lizard CSV + rust-code-analysis JSON；硬阈值仍由上面的静态门判定）"
bash scripts/complexity.sh

step "架构门（层表 / 边集 / 所有权 / 词汇表镜像 / 越界岛 / 对拍口径，与 ARCHITECTURE.md 单一来源比对）"
cargo test -p llm-testkit --lib --locked

step "命名门（下划线文件名 / 后缀残留 / 泛名函数 / Op 后缀，零命中，已清零保持阻断）"
bash scripts/name-lint.sh

step "宏展开烟测（生成代码可读性与覆盖；不把展开文本当作独立编译证明）"
cargo expand --version
mkdir -p target/expanded
cargo expand -p llm-annotation --lib --all-features --locked > target/expanded/annotation.rs
cargo expand -p llm-train --lib --tests --all-features --locked > target/expanded/train.rs
test -s target/expanded/annotation.rs
test -s target/expanded/train.rs

step "拼写门（typos；领域词与精确例外由 _typos.toml 单一持有）"
typos --version
typos --sort

step "工作法门（actionlint；可用 ACTIONLINT 指定固定发布二进制）"
"${ACTIONLINT:-actionlint}" -version
"${ACTIONLINT:-actionlint}"

step "密度预算（crates/*/src ≤60,000 行；超线即失败，不以报告代替达标）"
bash scripts/loc-budget.sh

step "基准可编译校验（实测与基线判定在固定硬件上跑，跨机器绝对值不可比）"
cargo bench --workspace --all-features --locked --no-run

step "基准判定（S++ roofline 达成率 ≥90% 算力 / ≥85% 带宽 + 回归预算；无测量产物的场景逐条跳过）"
cargo test -p llm-benchmarks --all-features --locked --test regress -- --nocapture

if [ "$PERF" -eq 1 ]; then
    step "基准实测（criterion 全量采样：25 组，慢，只在固定硬件上开）"
    cargo bench -p llm-benchmarks --all-features --locked
    step "基准判定（用刚测出的 criterion 产物重判）"
    cargo test -p llm-benchmarks --all-features --locked --test regress -- --nocapture
fi

if [ "$FUZZ" -eq 1 ]; then
    step "有界 fuzz（libFuzzer + ASan；FUZZ_TARGETS 可收窄目标）"
    FUZZ_SECONDS="${FUZZ_SECONDS:-5}" bash scripts/fuzz.sh
fi

if [ "$MIRI" -eq 1 ]; then
    step "Miri 纯计算核心（MIRI_MODULES 可扩展模块）"
    bash scripts/miri.sh
fi

if [ "$COVERAGE" -eq 1 ]; then
    step "覆盖率采集（llvm-cov nextest）"
    cargo llvm-cov nextest --workspace --all-features --locked --lcov --output-path lcov.info
fi

echo
echo "==> 全部门禁通过"

#!/usr/bin/env bash
# 独立复杂度证据：Lizard CSV + rust-code-analysis JSON；硬阈值仍由 llm-tools 静态门判定。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

output="${QUALITY_OUTPUT:-target/complexity}"
ccn="${LIZARD_CCN:-5}"
length="${LIZARD_LENGTH:-30}"
lizard_version="${LIZARD_VERSION:-1.24.1}"
rm -rf "$output/rca"
mkdir -p "$output/rca"

{
    uv tool run --from "lizard==$lizard_version" lizard --version
    rust-code-analysis-cli --version
} | tee "$output/versions.txt"

# cognitive 扩展补充认知复杂度；嵌套由项目静态门检查，-i -1 不替代硬门。
uv tool run --from "lizard==$lizard_version" lizard \
    -l rust -C "$ccn" -L "$length" -Ecognitive \
    --csv -i -1 -o "$output/lizard.csv" crates
rust-code-analysis-cli -m -w -O json --pr -p crates -o "$output/rca"
test -s "$output/lizard.csv"

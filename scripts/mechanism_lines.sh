#!/usr/bin/env bash
# 机制代码口径分账（M1-C1，design §5.3）：按 L1–L7 统计生产机制 Rust 行数。
#
# 口径与排除规则（与含测试粗口径 71,482 严格分开，不得混用）：
#   1. 只统计 `crates/*/src/**/*.rs` 与根包 `src/**/*.rs`；`target/`、`vendor/` 不进树。
#   2. 排除成员级测试目录：任意路径段为 `tests/` 的文件（集成测试，根包 `tests/` 同属此类）。
#   3. 构建脚本 `build.rs` 位于 crate 根、不在 `src/` 内，扫描目录天然排除（编译期产物，
#      不进运行时机制分母）。
#   4. 工具成员整体不计机制口径：`annotation`（编译期注解）、`fuzz`（模糊测试契约）、
#      `benches`（集中测量容器）——见 `docs/ARCHITECTURE.md` 分层表注。
#   5. 内联 `#[cfg(test)] mod tests` 体内行不在本脚本目录级扣除；严格口径由
#      `tool::strip_tests`（基于词法树剔除测试条目、保留行号）在审计/门禁侧再扣一次。
#
# 用法：scripts/mechanism_lines.sh        # 打印分层表与合计（目录级口径）
#       scripts/mechanism_lines.sh --csv # 机器可读 CSV，便于回填基线表
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 工具成员：整体排除机制分母。
TOOL_MEMBERS="annotation fuzz benches"

LAYERS=(L1 L2 L3 L4 L5 L6 L7)
LAYER_LINES=(0 0 0 0 0 0 0)
LAYER_FILES=(0 0 0 0 0 0 0)

# 分层归属（docs/ARCHITECTURE.md §一、design §3.1）。
layer_index() {
    case "$1" in
        diagnostic|testkit|dtype|wire|tool) echo 0 ;;
        prof|op|prim) echo 1 ;;
        cutile) echo 2 ;;
        tensor|storage|tokenizer) echo 3 ;;
        weights|autograd|data|nn|shard) echo 4 ;;
        train|engine|serve|backend|cli) echo 5 ;;
        *) echo -1 ;;
    esac
}

is_tool_member() {
    local member="$1"
    for tool in $TOOL_MEMBERS; do
        [ "$member" = "$tool" ] && return 0
    done
    return 1
}

count_file() {
    wc -l < "$1" | tr -d ' '
}

accumulate() {
    local index="$1" file="$2"
    local lines
    lines="$(count_file "$file")"
    LAYER_LINES[$index]=$((LAYER_LINES[$index] + lines))
    LAYER_FILES[$index]=$((LAYER_FILES[$index] + 1))
}

# 成员 crate：扫描 src 下的 .rs，跳过 tests/ 目录。
#  crate 根的 `build.rs` 不在 `src/` 内，天然不被扫描；`src/build.rs` 若是模块则照常计入。
for dir in crates/*/; do
    member="$(basename "$dir")"
    if is_tool_member "$member"; then
        continue
    fi
    index="$(layer_index "$member")"
    if [ "$index" -lt 0 ]; then
        echo "未分层成员：$member" >&2
        exit 2
    fi
    while IFS= read -r -d '' file; do
        case "$file" in
            */tests/*) continue ;;
        esac
        accumulate "$index" "$file"
    done < <(find "$dir/src" -name '*.rs' -print0 2>/dev/null)
done

# 根包 src/：L7 入口组装（不含根 tests/）。
while IFS= read -r -d '' file; do
    case "$file" in
        */tests/*) continue ;;
    esac
    accumulate 6 "$file"
done < <(find src -name '*.rs' -print0 2>/dev/null)

TOTAL_LINES=0
TOTAL_FILES=0
for i in "${!LAYERS[@]}"; do
    TOTAL_LINES=$((TOTAL_LINES + LAYER_LINES[i]))
    TOTAL_FILES=$((TOTAL_FILES + LAYER_FILES[i]))
done

if [ "${1:-}" = "--csv" ]; then
    echo "layer,files,lines"
    for i in "${!LAYERS[@]}"; do
        echo "${LAYERS[i]},${LAYER_FILES[i]},${LAYER_LINES[i]}"
    done
    echo "total,${TOTAL_FILES},${TOTAL_LINES}"
else
    printf "%-6s %8s %8s\n" "层" "文件" "机制行"
    for i in "${!LAYERS[@]}"; do
        printf "%-6s %8d %8d\n" "${LAYERS[i]}" "${LAYER_FILES[i]}" "${LAYER_LINES[i]}"
    done
    printf "%-6s %8d %8d\n" "合计" "$TOTAL_FILES" "$TOTAL_LINES"
    echo
    echo "注：目录级口径，未扣内联 #[cfg(test)] 体；严格口径经 tool::strip_tests 再扣。"
    echo "注：工具成员 annotation / fuzz / benches 与根 tests/、build.rs、target/ 已排除。"
fi

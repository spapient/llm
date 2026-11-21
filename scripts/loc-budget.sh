#!/usr/bin/env bash
# 密度门（S++ N2 / N3 / N6）：src 总行数、单文件行数、单 crate 行数三项预算。
# 用法：bash scripts/loc-budget.sh；超线即红（exit 1）。
set -eu
cd "$(dirname "$0")/.."

SRC_MAX=90000      # N6：crates/*/src 总行数
FILE_MAX=400       # N3：生产文件 S++ 线
FILE_VETO=600      # N3：>600 一票否决
CRATE_MAX=8000     # N2：单 crate

src_lines=$(find crates/*/src -name '*.rs' -print0 | xargs -0 cat | wc -l | tr -d ' ')
src_files=$(find crates/*/src -name '*.rs' | wc -l | tr -d ' ')
over_file=$(find crates/*/src -name '*.rs' -print0 | xargs -0 wc -l | awk -v m="$FILE_MAX" '$1>m && $2!="total"' | sort -rn)
over_veto=$(printf '%s\n' "$over_file" | awk -v m="$FILE_VETO" '$1>m' | sort -rn)
over_crate=$(for d in crates/*/src; do n=$(find "$d" -name '*.rs' -print0 | xargs -0 cat | wc -l | tr -d ' '); printf '%s %s\n' "$n" "$d"; done | awk -v m="$CRATE_MAX" '$1>m' | sort -rn)

printf 'src 总行数: %s / %s\n' "$src_lines" "$SRC_MAX"
printf 'src 文件数: %s\n' "$src_files"
printf '> %s 行文件: %s 个\n' "$FILE_MAX" "$(printf '%s\n' "$over_file" | grep -c . || true)"
printf '> %s 行（一票否决）: %s 个\n' "$FILE_VETO" "$(printf '%s\n' "$over_veto" | grep -c . || true)"
printf '> %s 行 crate: %s 个\n\n' "$CRATE_MAX" "$(printf '%s\n' "$over_crate" | grep -c . || true)"

[ "$src_lines" -le "$SRC_MAX" ] || { printf '✗ 总量超预算 %s 行\n' "$((src_lines - SRC_MAX))"; exit 1; }
[ -z "$(printf '%s\n' "$over_veto" | grep . || true)" ] || { printf '✗ 存在 >%s 行生产文件\n%s\n' "$FILE_VETO" "$over_veto"; exit 1; }
[ -z "$(printf '%s\n' "$over_crate" | grep . || true)" ] || { printf '✗ 存在 >%s 行 crate\n%s\n' "$CRATE_MAX" "$over_crate"; exit 1; }
printf '密度门：通过\n'

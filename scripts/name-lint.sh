#!/usr/bin/env bash
# 命名门（S++ N4 / N5）：文件名必须单词、符号不得带前后缀、不得有泛名函数。
# 判据全为 grep；零命中才绿。用法：bash scripts/name-lint.sh
set -eu
cd "$(dirname "$0")/.."

fail=0
report() {
  printf '%s\n' "── $1 ──"
  shift
  out="$("$@" || true)"
  if [ -n "$out" ]; then
    printf '%s\n' "$out"
    fail=1
  else
    printf '(none)\n'
  fi
}

report "src 文件名含下划线（N4：限定词应下沉到目录）" \
  sh -c "find crates/*/src -name '*_*.rs' | sort"

report "定义名以 _v2/_old/_bak/_tmp/_util/_helper/_manager/_handler 结尾（N5 残留标记）" \
  sh -c "grep -rEn '(fn|struct|enum|trait|const|static|mod) [A-Za-z][A-Za-z0-9]*_(v2|old|bak|tmp|util|helper|manager|handler)\b' crates/*/src || true"

report "泛名函数 do_/process_/handle_/manage_（N5）" \
  sh -c "grep -rEn 'fn (do|process|handle|manage)_' crates/*/src || true"

report "类型名带 Op/Impl/Helper/Manager/Handler 后缀（N5）" \
  sh -c "grep -rEn '(struct|enum|trait) [A-Z][A-Za-z]+(Op|Impl|Helper|Manager|Handler)\b' crates/*/src || true"

if [ "$fail" -ne 0 ]; then
  printf '\n命名门：未通过（见上）\n'
  exit 1
fi
printf '\n命名门：通过（0 命中）\n'

#!/usr/bin/env bash
# 命名门（S++ N4 / N5）：文件名必须单词、符号不得带前后缀、不得有泛名函数。
# 判据全为 grep；零命中才绿。用法：bash scripts/name-lint.sh
#
# 这里只管**文件名与符号形状**。「一个概念两个词」（`buffer`/`index` 与 `buf`/`idx`
# 并存）是**标识符级**判断，grep 做不了——设备源字符串里的 Metal 选择器
# `[[buffer(0)]]` 与外来 API（`MTLBuffer`/`atIndex`）必须豁免，而裸 grep 无法区分。
# 那条判据在 `crates/tools/src/words.rs`，由 `cargo test -p llm-tools --test gate` 守。
#
# 命名分层（决策见 docs/adr/0057 §1）：
# - 符号命名（类型 UpperCamelCase、函数/变量/模块 snake_case、常量 SCREAMING_SNAKE）src 与
#   tests 同一套官方规则；本脚本的符号后缀/泛名/类型后缀判据扫 src，tests 符号由 clippy 兜底。
# - src 文件名零下划线零容忍：限定词一律下沉到目录，不设类别豁免、不留存量基线。
# - tests/ 文件名允许下划线（`<topic>_tests.rs` 主题聚合惯例），豁免文件名下沉规则；豁免的
#   只是文件名，测试内符号仍守官方规则。故下面的文件名判据只扫 crates/*/src，不扫 tests/。
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

report "src 文件名含下划线（N4：限定词应下沉到目录；零容忍，无基线豁免）" \
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

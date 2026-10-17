#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Mechanical naming gate. The authoritative rules live in docs/engineering.md; this
# script only enforces the mechanical subset defined there (rule.md NM1/NM3/NM5).
#
# Scope: crates/*/src/**.rs, excluding #[cfg(test)] regions (NM4 permits terse
# local names and test helpers are not part of any contract).

# Leading tokens that duplicate a canonical verb (docs/engineering.md).
FORBIDDEN_VERBS=(
  build create make construct instantiate assemble
  fetch retrieve obtain acquire query
  assign put
  dispose free release dealloc
)

# macOS 沙箱（如 Claude Code）会拦截默认 TMPDIR（/var/folders/...）的临时文件；
# 改为在仓库内创建 scratch 文件，保证各环境可写。
tmp="$(mktemp "$ROOT_DIR/.checknaming.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

# Extract, per production line, either a function, a const, or a macro name.
# `#[cfg(test)]` starts a test region that extends to the end of the file.
# Raw string literals (r#"..."#) hold embedded foreign source — MSL, CUDA C —
# which is not Rust and MUST NOT be scanned for Rust identifiers; `raw` carries
# that state across lines.
while IFS= read -r source; do
  awk '
    FNR == 1 { intest = 0; raw = 0 }
    raw { if ($0 ~ /"#/) raw = 0; next }
    /#\[cfg\(test\)\]/ { intest = 1 }
    intest { next }
    /^[[:space:]]*(\/\/|\*|\/\*)/ { next }
    {
      gsub(/r#"[^"]*"#/, "")
      # Raw pointer types (`*const T`, `*mut T`) are not const declarations;
      # strip them so an FFI signature is not reported as a lowercase const.
      gsub(/\*[ \t]*const[ \t]+[A-Za-z_][A-Za-z0-9_:]*/, "")
      if ($0 ~ /r#"/) { sub(/r#".*$/, ""); raw = 1 }
      if (match($0, /fn[ \t]+[a-z_][a-z0-9_]*/)) {
        s = substr($0, RSTART, RLENGTH); sub(/fn[ \t]+/, "", s)
        print "fn", FILENAME, FNR, s
      }
      if (match($0, /const[ \t]+[A-Za-z_][A-Za-z0-9_]*/)) {
        s = substr($0, RSTART, RLENGTH); sub(/const[ \t]+/, "", s)
        if (s != "fn") { print "const", FILENAME, FNR, s }
      }
      if (match($0, /macro_rules![ \t]+[A-Za-z_][A-Za-z0-9_]*/)) {
        s = substr($0, RSTART, RLENGTH); sub(/macro_rules![ \t]+/, "", s)
        print "macro", FILENAME, FNR, s
      }
    }' "$source"
done < <(find crates -path '*/src/*.rs' -not -path '*/target/*') >"$tmp"

log "Checking naming conformance (docs/engineering.md)"

violations=0

report() {
  printf '  %s\n' "$1" >&2
  violations=$((violations + 1))
}

while read -r kind file line name; do
  case "$kind" in
    fn)
      for bad in "${FORBIDDEN_VERBS[@]}"; do
        case "$name" in
          "$bad"*)
            report "$file:$line: forbidden synonym '$bad' in fn '$name' (see docs/engineering.md)"
            break
            ;;
        esac
      done
      ;;
    const | macro)
      case "$name" in
        [A-Z][A-Z0-9_]*) ;;
        *) report "$file:$line: $kind '$name' must be uppercase compact (NM5)" ;;
      esac
      ;;
  esac
done <"$tmp"

# A naming-lint exemption MUST cite the rule.md clause it relies on
# (docs/engineering.md).
while IFS= read -r hit; do
  [ -n "$hit" ] || continue
  case "$hit" in
    *NM*) ;;
    *)
      file="${hit%%:*}"
      rest="${hit#*:}"
      line="${rest%%:*}"
      report "$file:$line: naming #[allow] must cite its NM clause (docs/engineering.md)"
      ;;
  esac
done < <(grep -rnE '#\[allow\([^]]*(non_snake_case|non_camel_case_types|non_upper_case_globals)' crates/*/src 2>/dev/null || true)

if [ "$violations" -ne 0 ]; then
  fail "$violations naming conformance violation(s); see docs/engineering.md"
fi

log "Naming conformance passed."

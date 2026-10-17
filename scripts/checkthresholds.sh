#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Mechanical quality-threshold gate for rule.md DS3 and A4.
#
# DS3 (complexity pre-control): one function must be reviewable in one pass.
#   The rule declares cyclomatic complexity <= 10 and effective lines <= 80.
#   Clippy's `cognitive_complexity` and `too_many_lines` are the Rust proxy
#   (rule.md A8); the thresholds live in clippy.toml. The check runs over the
#   library and binary targets only (`--lib --bins`), so `#[cfg(test)]` code and
#   the repository-root `tests/` are excluded, matching the rule's definition of
#   effective lines as excluding test code.
#
# A4 (minimal interface): a module's exported interface width is <= 20. The
#   width is the count of public symbols at the crate root: every name in a
#   `pub use` re-export plus every plain `pub` item declaration. `pub(crate)` and
#   other restricted visibilities are not exported and do not count.

A4BOUND=20

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

log "Checking DS3 complexity and length budgets (production code only)"
cargo clippy --workspace --lib --bins --locked -- \
  -D clippy::cognitive_complexity \
  -D clippy::too_many_lines

log "Checking A4 exported interface width (<= ${A4BOUND} per crate root)"

violations=0
failures=""
for lib in crates/*/src/lib.rs crates/gpu/gpu.rs; do
  [ -f "$lib" ] || continue
  case "$lib" in
    crates/gpu/gpu.rs) crate="gpu" ;;
    *) crate="$(basename "$(dirname "$(dirname "$lib")")")" ;;
  esac
  width="$(
    awk '
      function countnames(s,   arr, c, i, k) {
        gsub(/[{};]/, "", s)
        gsub(/[ \t\n]/, "", s)
        c = split(s, arr, ",")
        k = 0
        for (i = 1; i <= c; i++) if (arr[i] != "") k++
        return k
      }
      BEGIN { total = 0; inuse = 0; buf = "" }
      {
        line = $0
        if (line ~ /^[[:space:]]*(\/\/|\*|\/\*)/) next
        if (inuse) {
          buf = buf " " line
          if (line ~ /\}/) { inuse = 0; total += countnames(buf); buf = "" }
          next
        }
        if (line ~ /^[[:space:]]*pub[[:space:]]+use[[:space:]].*\{/) {
          buf = line
          if (line ~ /\}/) { total += countnames(buf); buf = "" }
          else inuse = 1
          next
        }
        if (line ~ /^[[:space:]]*pub[[:space:]]+use[[:space:]]/) { total += 1; next }
        if (line ~ /^[[:space:]]*pub[[:space:]]+(fn|struct|enum|trait|const|type|mod|union)[[:space:]]/) {
          total += 1
          next
        }
      }
      END { print total }
    ' "$lib"
  )"
  if [ "$width" -gt "$A4BOUND" ]; then
    failures="${failures}  ${crate} (${lib}): exported interface width ${width} exceeds ${A4BOUND}\n"
    violations=$((violations + 1))
  fi
done

if [ "$violations" -ne 0 ]; then
  printf '%b' "$failures" >&2
  fail "$violations A4 interface-width violation(s); reduce the exported surface or record an M4 declaration"
fi

log "Quality-threshold checks passed."

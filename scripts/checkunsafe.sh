#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Mechanical gate for the restricted unsafe-code exemption.
# Authority: docs/engineering.md (rule 14) and docs/design.md §21.8.
#
# Default policy: unsafe is forbidden workspace-wide. The ONLY tolerated sites
# live in the FFI island of the native GPU backend crates. Four assertions:
#   1. `allow(unsafe_code)` appears ONLY in the whitelisted FFI files.
#   2. Every non-whitelisted crate root still carries `#![forbid(unsafe_code)]`.
#   3. Every whitelisted crate root carries `#![deny(unsafe_code)]`, and its
#      FFI file declares `#![allow(unsafe_code)]`.
#   4. Each `unsafe` construct is directly preceded by a `// SAFETY:` comment,
#      and the per-file site count EQUALS its registered count. The bound is an
#      exact match, not a ceiling: the registry table in docs/design.md §21.8
#      and the code MUST NOT drift in either direction, so a site cannot be
#      added without amending the table, nor silently removed.
# Adding or removing a site REQUIRES updating docs/design.md §21.8 and the
# count below in the same change.

# file:registered-count (exact match, not a ceiling)
WHITELIST=(
  "crates/metal/src/ffi.rs:4"
  "crates/cuda/src/ffi.rs:3"
  "crates/musa/src/ffi.rs:2"
)

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

is_whitelisted() {
  local f="$1" entry
  for entry in "${WHITELIST[@]}"; do
    [ "$f" = "${entry%%:*}" ] && return 0
  done
  return 1
}

# --- 1. allow(unsafe_code) only in whitelisted files -------------------------
while IFS= read -r file; do
  [ -n "$file" ] || continue
  is_whitelisted "$file" || fail "$file: allow(unsafe_code) is permitted only in the whitelisted FFI files"
done < <(rg -n '^\s*#!?\[allow\(unsafe_code\)\]' crates/*/src 2>/dev/null | cut -d: -f1 | sort -u || true)

# --- 2. non-exempt crate roots keep #![forbid(unsafe_code)] ------------------
missing=""
for lib in crates/*/src/lib.rs; do
  [ -f "$lib" ] || continue
  is_whitelisted "${lib%/*}/ffi.rs" && continue
  grep -q '#!\[forbid(unsafe_code)\]' "$lib" || missing="${missing}  ${lib}"$'\n'
done
if [ -n "$missing" ]; then
  fail "these crate roots lost #![forbid(unsafe_code)] (docs/design.md §21.8):"$'\n'"${missing}"
fi

# --- 3 & 4. whitelisted FFI islands -----------------------------------------
for entry in "${WHITELIST[@]}"; do
  f="${entry%%:*}"
  registered="${entry##*:}"
  [ -f "$f" ] || continue

  lib="${f%/*}/lib.rs"
  if [ -f "$lib" ]; then
    grep -q '#!\[forbid(unsafe_code)\]' "$lib" \
      && fail "$f: exempt crate must use deny, not forbid, at the crate root"
    grep -q '#!\[deny(unsafe_code)\]' "$lib" \
      || fail "$lib: exempt crate root must declare #![deny(unsafe_code)]"
  fi
  grep -q '#!\[allow(unsafe_code)\]' "$f" \
    || fail "$f: FFI island must declare #![allow(unsafe_code)]"

  # Line numbers of `unsafe` constructs on code lines (comments excluded, and
  # `unsafe_code` never matches because `_` is a word character). Each site is
  # accepted only when a contiguous `//` comment block directly above it
  # contains `SAFETY:`.
  sites=0
  bad=""
  while read -r status lineno; do
    [ -n "$status" ] || continue
    sites=$((sites + 1))
    [ "$status" = "BAD" ] && bad="${bad}  ${f}:${lineno}"$'\n'
  done < <(awk '
    { lines[NR] = $0 }
    END {
      for (i = 1; i <= NR; i++) {
        t = lines[i]
        sub(/^[ \t]+/, "", t)
        if (t ~ /^\/\//) continue
        if (t ~ /^\*/) continue
        if (lines[i] ~ /unsafe_code/) continue
        if (lines[i] !~ /(^|[^_A-Za-z0-9])unsafe([^_A-Za-z0-9]|$)/) continue
        ok = 0
        j = i - 1
        while (j >= 1) {
          if (lines[j] ~ /SAFETY:/) { ok = 1; break }
          if (lines[j] ~ /\/\//) { j--; continue }
          break
        }
        print (ok ? "OK " : "BAD ") i
      }
    }' "$f")

  if [ -n "$bad" ]; then
    fail "each unsafe site MUST be directly preceded by a // SAFETY: comment (docs/design.md §21.8):"$'\n'"${bad}"
  fi
  if [ "$sites" -ne "$registered" ]; then
    fail "$f: ${sites} unsafe site(s) but docs/design.md §21.8 registers ${registered}; the registry table and the code MUST match exactly (update the table AND the count in this script together)"
  fi
  printf '  %s: %s/%s unsafe site(s) (exact), each with a SAFETY comment.\n' "$f" "$sites" "$registered"
done

printf '  unsafe gate passed: no unregistered unsafe outside the FFI islands.\n'

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Mechanical gate for the black-box testing paradigm
# (docs/decision-records/blackboxtesting.md).
#
# Rule: production source under `crates/*/src` MUST NOT contain any
# `#[cfg(test)]` (or `#[test]`) inline test code. All unit, integration,
# regression, and benchmark tests live at the repository root and exercise the
# crate through its public `pub` interface only.
#
# Staged migration: the workspace is mid-migration, so the gate runs as a
# ratchet against `scripts/srctestbaseline.txt`.
#   * A file that contains inline tests and is NOT in the baseline is a new
#     violation and fails the check.
#   * A file that contains inline tests and IS in the baseline is tolerated
#     while its crate is pending migration.
#   * A baseline entry with no remaining inline tests is a stale entry and
#     fails the check, so the ratchet only ever tightens.
# When the baseline is empty, the gate is a hard prohibition.

BASELINE="scripts/srctestbaseline.txt"

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

[ -f "$BASELINE" ] || fail "missing baseline file $BASELINE"

current="$(rg -l '#\[cfg\(test\)\]|#\[test\]' crates/*/src 2>/dev/null | sort || true)"

violations=""
while IFS= read -r file; do
  [ -n "$file" ] || continue
  if ! grep -qxF "$file" "$BASELINE"; then
    violations="${violations}  new inline test module: ${file}\n"
  fi
done <<<"$current"

stale=""
while IFS= read -r entry; do
  [ -n "$entry" ] || continue
  if ! grep -qxF "$entry" <<<"$current"; then
    stale="${stale}  migrated but still listed: ${entry}\n"
  fi
done <"$BASELINE"

if [ -n "$violations" ] || [ -n "$stale" ]; then
  [ -n "$violations" ] && printf 'New inline test code is forbidden (move it to the repository-root tests/):\n%b\n' "$violations" >&2
  [ -n "$stale" ] && printf 'These files were migrated; remove them from %s:\n%b\n' "$BASELINE" "$stale" >&2
  fail "inline-test gate failed"
fi

printf '  inline-test ratchet: %s file(s) still pending migration.\n' "$(grep -c . "$BASELINE" || true)"

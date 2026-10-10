#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() {
printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
printf '\nERROR: %s\n' "$*" >&2
exit 1
}

command -v cargo >/dev/null 2>&1 || fail "Cargo is not installed or is not on PATH."
command -v cargo-tarpaulin >/dev/null 2>&1 || fail "cargo-tarpaulin is required. Install it with: cargo install cargo-tarpaulin"

[[ -f Cargo.toml ]] || fail "Cargo.toml was not found at the repository root."
[[ -f Cargo.lock ]] || fail "Cargo.lock is missing. Resolve and commit the reviewed lockfile first."

log "Running workspace coverage"
cargo tarpaulin \
--workspace \
--locked \
--out Html \
--out Xml \
--output-dir target/coverage \
--timeout 120

log "Coverage reports generated under target/coverage/"
log "Review uncovered branches and critical paths; coverage percentage alone does not prove correctness."

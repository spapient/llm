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

[[ -f Cargo.toml ]] || fail "Cargo.toml was not found at the repository root."
[[ -f Cargo.lock ]] || fail "Cargo.lock is missing. Resolve dependencies, review the lockfile, and commit it first."

log "Running workspace unit, integration, and documentation tests"
cargo test --workspace --locked

log "Running tests for all targets, including test targets"
cargo test --workspace --all-targets --locked

log "Test suite completed successfully."

#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() {
printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
printf 'ERROR: %s\n' "$*" >&2
exit 1
}

command -v cargo >/dev/null 2>&1 || fail "Cargo is not installed or is not on PATH."

[[ -f Cargo.toml ]] || fail "Cargo.toml was not found at the repository root."
[[ -f Cargo.lock ]] || fail "Cargo.lock is missing. Resolve and review dependencies first."

if ! cargo bench --workspace --no-run --locked; then
fail "Benchmark targets failed to compile."
fi

log "Running workspace benchmarks"
cargo bench --workspace --locked

log "Benchmark run completed. Review the generated results before making performance claims."

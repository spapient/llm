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
command -v rustfmt >/dev/null 2>&1 || fail "rustfmt is not installed."
cargo clippy --version >/dev/null 2>&1 || fail "Clippy is not installed."

if [[ ! -f Cargo.toml ]]; then
fail "Cargo.toml was not found at the repository root."
fi

if [[ ! -f Cargo.lock ]]; then
log "Cargo.lock is missing; locked checks may fail until dependencies are resolved and the lockfile is committed."
fi

log "Checking formatting"
cargo fmt --all -- --check

log "Checking workspace and all targets"
cargo check --workspace --all-targets --locked

log "Running Clippy"
cargo clippy --workspace --all-targets --locked -- -D warnings

log "Checking naming conformance"
bash scripts/checknaming.sh

log "Checking inline-test gate"
bash scripts/checksrctests.sh

log "Checking unsafe exemption gate"
bash scripts/checkunsafe.sh

log "Checking quality thresholds"
bash scripts/checkthresholds.sh

log "Checking documentation"
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked

log "All static quality checks passed."

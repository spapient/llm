//! Shared helpers for the repository-root integration suites.
//!
//! A helper used by a single suite stays in that suite; the helpers here are
//! duplicated by two or more suites, so they live in one place. The tiny-model
//! engine fixtures stay in [`crate::common`], and the independent cross-crate
//! reference implementations stay in `testkit`.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ::tensor::{shape, tensor};
use lib::{limits, tokenid};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Builds an `f32` tensor from dimensions and data.
pub fn build<D: AsRef<[usize]>>(dims: D, data: Vec<f32>) -> tensor<f32> {
    tensor::new(shape::new(dims.as_ref().to_vec()).unwrap(), data).unwrap()
}

/// Builds a shape from dimensions.
pub fn shapeof(dims: &[usize]) -> shape {
    shape::new(dims.to_vec()).unwrap()
}

/// Wraps raw ids as the `lib::tokenid` newtype.
pub fn tokens(values: &[u32]) -> Vec<tokenid> {
    values.iter().copied().map(tokenid::new).collect()
}

/// Returns whether `a` and `b` are within `tol`.
pub fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Returns whether `got` is within `1e-5` relative tolerance of `want`.
pub fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= 1.0e-5 * (1.0 + want.abs())
}

/// Returns tensor limits that admit every test fixture.
pub fn permissive() -> limits {
    limits::new(usize::MAX, usize::MAX, usize::MAX, usize::MAX)
}

/// Returns wide tensor limits for multi-head work.
pub fn wide() -> limits {
    limits::new(1 << 24, 1 << 28, 1 << 20, 1 << 16)
}

/// Frames a safetensors header and data section with the length prefix.
pub fn framed(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

/// Returns a fresh temporary directory for `tag`.
pub fn scratch(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("llm-it-{}-{tag}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Returns a unique temporary file path for `tag`.
pub fn temppath(tag: &str) -> PathBuf {
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("llm-it-{}-{tag}-{unique}", std::process::id()));
    path
}

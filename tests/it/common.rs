//! Shared fixtures for the repository-root integration tests.
//!
//! The `infer` generation/isolation suites and the `serve` service/socket
//! suites both need the same tiny one-layer Llama configuration, deterministic
//! weights, and a bounded CPU provider, so those fixtures live here instead of
//! being copied per suite. Keeping them shared also keeps the fixtures
//! independent of the engine under test: `argmaxlast` re-derives the greedy
//! next token from the public model API rather than calling the generation
//! engine.
//!
//! Every helper is `pub` but the module is private to each test binary, so
//! `dead_code` is expected whenever a suite uses only part of the surface.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::io::Cursor;

use context::executionconfig;
use cpu::cpuexecutor;
use lib::{limits, seed};
use models::{causallm, llamaconfig, llamamodel, llamaweights};
use serve::{httprequest, serviceconfig, transportlimits};

/// Vocabulary size.
pub const VOCAB: usize = 5;
/// Model dimension.
pub const DIM: usize = 4;
/// Query head count.
pub const NHEADS: usize = 1;
/// Key/value head count.
pub const KVHEADS: usize = 1;
/// Feed-forward hidden dimension.
pub const HIDDEN: usize = 8;
/// Maximum sequence length.
pub const MAXSEQ: usize = 8;
/// The `serve` supported form media type.
pub const URLENCODED: &str = "application/x-www-form-urlencoded";

/// Builds the tiny configuration used by every test.
pub fn config() -> llamaconfig {
    llamaconfig::new(
        VOCAB, DIM, 1, NHEADS, KVHEADS, HIDDEN, 1.0e-5, 10_000.0, MAXSEQ,
    )
    .unwrap()
}

/// Builds a tiny model from deterministic random weights.
pub fn model(seedvalue: u64) -> llamamodel {
    let c = config();
    let w = llamaweights::random(&c, seed::new(seedvalue)).unwrap();
    llamamodel::new(c, w).unwrap()
}

/// Builds a CPU compute provider bounded for the tiny workload.
pub fn provider() -> cpuexecutor {
    cpuexecutor::new(limits::new(1 << 20, 1 << 24, 1 << 12, 1 << 8))
}

/// Returns the greedy next token of `tokens`, ties broken by lowest index.
///
/// This is an independent re-derivation using the public model API, not the
/// generation engine under test.
pub fn argmaxlast(model: &llamamodel, provider: &cpuexecutor, tokens: &[usize]) -> usize {
    let logits = model.forward(provider, tokens).unwrap();
    let vocab = model.getconfig().getvocab();
    let data = logits.getdata();
    let last = &data[data.len() - vocab..];
    let mut best = 0usize;
    let mut bestvalue = f32::NEG_INFINITY;
    for (index, &value) in last.iter().enumerate() {
        if value > bestvalue {
            bestvalue = value;
            best = index;
        }
    }
    best
}

/// Builds explicit transport bounds for the tests.
pub fn transportlimits() -> transportlimits {
    transportlimits::new(64 * 1024, 4 * 1024, 32, 8 * 1024, 32, 4 * 1024, 64, 32)
}

/// Builds a service configuration with no deadline.
pub fn serviceconfig(maxconcurrent: usize) -> serviceconfig {
    serviceconfigwith(maxconcurrent, None)
}

/// Builds a service configuration with an explicit deadline.
pub fn serviceconfigwith(
    maxconcurrent: usize,
    deadline: Option<std::time::Duration>,
) -> serviceconfig {
    serviceconfig::new(
        transportlimits(),
        executionconfig::new(1 << 20, 1, 128).unwrap(),
        maxconcurrent,
        deadline,
    )
    .unwrap()
}

/// Parses raw request bytes with the test transport bounds.
pub fn parse(bytes: &[u8]) -> httprequest {
    httprequest::parse(&mut Cursor::new(bytes), &transportlimits()).unwrap()
}

/// Builds an `application/x-www-form-urlencoded` generation request.
pub fn generaterequest(prompt: &[usize], maxnewtokens: usize) -> httprequest {
    let list = prompt
        .iter()
        .map(|token| token.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let body = format!("prompt={list}&maxnewtokens={maxnewtokens}");
    parse(&post(&body, URLENCODED))
}

/// Builds raw bytes for `POST /generate` with the given body and media type.
pub fn post(body: &str, contenttype: &str) -> Vec<u8> {
    let raw = format!(
        "POST /generate HTTP/1.1\r\nHost: test\r\nContent-Type: {contenttype}\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    raw.into_bytes()
}

/// Returns the `tokens=` value of a response body as a token list.
pub fn tokensof(response: &serve::httpresponse) -> Vec<usize> {
    let text = response.getbodytext().unwrap_or("");
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("tokens=") {
            if value.is_empty() {
                return Vec::new();
            }
            return value
                .split(',')
                .map(|item| item.parse::<usize>().unwrap())
                .collect();
        }
    }
    Vec::new()
}

/// Returns the `reason=` value of a response body.
pub fn reasonof(response: &serve::httpresponse) -> String {
    let text = response.getbodytext().unwrap_or("");
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("reason=") {
            return value.to_string();
        }
    }
    String::new()
}

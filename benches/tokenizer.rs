//! Criterion benchmarks for the `tokenizer` crate.
//!
//! Establishes the Phase 0 baseline required by `docs/engineering.md` §4.5
//! and `docs/tokenizer-optimization-plan.md` §6 before any optimization. No
//! improvement is claimed here. Fixtures are synthetic and deterministic: no
//! external PRNG and no real `tokenizer.json` (none is vendored in-tree).
//!
//! * `bytepair/apply_chain` isolates the byte-pair merge loop at the quadratic
//!   worst case — a chain of `len - 1` merges collapses `len` distinct symbols
//!   into one, so each round scans the whole list. This is the cost that
//!   plan step T6 rewrites.
//! * `encode/bytefallback` and `decode/bytefallback` exercise the end-to-end
//!   byte-fallback path (`<0xNN>` tokens) with no merges, isolating per-symbol
//!   resolution and allocation rather than the merge loop.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::tokenizer::{bytepair, tokenizer};
use criterion::{Criterion, criterion_group, criterion_main};

/// Text pattern for the byte-fallback workloads: ASCII plus a CJK run so the
/// byte fallback emits multi-byte UTF-8 sequences.
const PATTERN: &str = "The quick brown fox jumps over 13 lazy dogs. 中文测试字符串。\n";

/// Builds a chain-merge table over `len` distinct CJK symbols and the symbol
/// list it collapses. Merge `i` joins the accumulated prefix with symbol `i+1`,
/// so `apply` performs `len - 1` merges with one merge per round.
fn chainfixture(len: usize) -> (bytepair, Vec<String>) {
    let chars: Vec<char> = (0..len)
        .map(|index| char::from_u32(0x4E00 + index as u32).unwrap())
        .collect();
    let symbols: Vec<String> = chars
        .iter()
        .map(|character| character.to_string())
        .collect();
    let mut merges: Vec<String> = Vec::with_capacity(len - 1);
    let mut prefix = String::new();
    for index in 0..len - 1 {
        prefix.push(chars[index]);
        merges.push(format!("{prefix} {}", chars[index + 1]));
    }
    let bpe = bytepair::frommerges(&merges).unwrap();
    (bpe, symbols)
}

/// Builds a byte-fallback tokenizer whose vocabulary holds the 256 `<0xNN>`
/// byte tokens plus `<unk>`, with no merge rules.
fn bytefallbacktok() -> tokenizer {
    let mut vocab = String::new();
    for byte in 0u32..256 {
        if byte != 0 {
            vocab.push(',');
        }
        vocab.push_str(&format!("\"<0x{byte:02X}>\":{byte}"));
    }
    let json = format!(
        "{{\"model\":{{\"type\":\"BPE\",\"vocab\":{{{vocab},\"<unk>\":256}},\
         \"merges\":[],\"byte_fallback\":true,\"unk_token\":\"<unk>\"}}}}"
    );
    tokenizer::fromjson(&json).unwrap()
}

/// Repeats `PATTERN` until `len` characters are produced.
fn repeatpattern(len: usize) -> String {
    let mut out = String::with_capacity(len * 3);
    let mut chars = PATTERN.chars().cycle();
    for _ in 0..len {
        out.push(chars.next().unwrap());
    }
    out
}

fn applychain(c: &mut Criterion) {
    for len in [64usize, 512, 4096] {
        let (bpe, symbols) = chainfixture(len);
        c.bench_function(&format!("tokenizer/bytepair/apply_chain/{len}"), |b| {
            b.iter(|| black_box(bpe.apply(&symbols)));
        });
    }
}

fn encodebytefallback(c: &mut Criterion) {
    let tok = bytefallbacktok();
    for len in [512usize, 4096] {
        let input = repeatpattern(len);
        c.bench_function(&format!("tokenizer/encode/bytefallback/{len}"), |b| {
            b.iter(|| black_box(tok.encode(&input).unwrap()));
        });
    }
}

fn decodebytefallback(c: &mut Criterion) {
    let tok = bytefallbacktok();
    let input = repeatpattern(4096);
    let ids = tok.encode(&input).unwrap();
    c.bench_function("tokenizer/decode/bytefallback/4096", |b| {
        b.iter(|| black_box(tok.decode(&ids).unwrap()));
    });
}

criterion_group!(benches, applychain, encodebytefallback, decodebytefallback);
criterion_main!(benches);

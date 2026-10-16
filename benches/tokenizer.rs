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

use ::tokenizer::{bytepair, bytetounicode, mergescratch, tiktokenizer, tokenizer, vocabulary};
use criterion::{Criterion, criterion_group, criterion_main};
use lib::tokenid;

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

/// Assigns `text` an id, appending a fresh `(text, id)` entry the first time it
/// is seen and returning the stable id on every call.
fn internid(
    text: &str,
    assigned: &mut std::collections::HashMap<String, u32>,
    entries: &mut Vec<(String, u32)>,
    next: &mut u32,
) -> u32 {
    if let Some(&id) = assigned.get(text) {
        return id;
    }
    let id = *next;
    *next += 1;
    assigned.insert(text.to_string(), id);
    entries.push((text.to_string(), id));
    id
}

/// Builds the chain fixture for the integer-keyed path: the same collapse as
/// [`chainfixture`] plus a vocabulary covering every merge side and result, and
/// the initial symbol ids. Mirrors the string path so the two benchmarks are
/// directly comparable.
fn chainfixtureids(len: usize) -> (bytepair, Vec<tokenid>) {
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
    let mut assigned = std::collections::HashMap::<String, u32>::new();
    let mut entries = Vec::<(String, u32)>::new();
    let mut next = 0u32;
    for symbol in &symbols {
        internid(symbol, &mut assigned, &mut entries, &mut next);
    }
    for merge in &merges {
        let (left, right) = merge.split_once(' ').unwrap();
        let merged = format!("{left}{right}");
        internid(left, &mut assigned, &mut entries, &mut next);
        internid(right, &mut assigned, &mut entries, &mut next);
        internid(&merged, &mut assigned, &mut entries, &mut next);
    }
    let vocab = vocabulary::fromentries(entries).unwrap();
    let bpe = bytepair::frommergesandvocab(&merges, &vocab).unwrap();
    let ids: Vec<tokenid> = symbols.iter().map(|s| vocab.getid(s).unwrap()).collect();
    (bpe, ids)
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

/// Builds a GPT-2 ByteLevel tokenizer whose vocabulary is exactly the 256
/// byte-to-unicode characters, with no merge rules.
fn byteleveltok() -> tokenizer {
    let table = bytetounicode();
    let mut vocab = serde_json::Map::new();
    for (byte, character) in table.iter().enumerate() {
        vocab.insert(character.to_string(), serde_json::json!(byte));
    }
    let doc = serde_json::json!({
        "model": {"type": "BPE", "vocab": vocab, "merges": [], "byte_fallback": false},
        "pre_tokenizer": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
    });
    tokenizer::fromjson(&doc.to_string()).unwrap()
}

/// Repeats `phrase` until `len` characters are produced, so its pre-tokens
/// repeat and exercise the byte-level per-call cache.
fn repeatphrase(phrase: &str, len: usize) -> String {
    let mut out = String::with_capacity(len);
    let mut chars = phrase.chars().cycle();
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

fn applyidschain(c: &mut Criterion) {
    for len in [64usize, 512, 4096] {
        let (bpe, ids) = chainfixtureids(len);
        c.bench_function(&format!("tokenizer/bytepair/applyids_chain/{len}"), |b| {
            b.iter(|| black_box(bpe.applyids(&ids)));
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

fn encodebytelevel(c: &mut Criterion) {
    let tok = byteleveltok();
    for len in [512usize, 4096] {
        let input = repeatphrase("the quick brown fox jumps over ", len);
        c.bench_function(&format!("tokenizer/encode/bytelevel/{len}"), |b| {
            b.iter(|| black_box(tok.encode(&input).unwrap()));
        });
    }
}

fn encodebytelevelbatch(c: &mut Criterion) {
    let tok = byteleveltok();
    let input = repeatphrase("the quick brown fox jumps over ", 512);
    let texts = vec![input.as_str(); 8];
    let mut scratch = mergescratch::default();
    let mut out = Vec::new();
    c.bench_function("tokenizer/encode_batch/bytelevel/8x512", |b| {
        b.iter(|| {
            out.clear();
            tok.encode_batch(&texts, &mut scratch, &mut out).unwrap();
            black_box(&out);
        });
    });
}

fn encodetiktoken(c: &mut Criterion) {
    let content = std::fs::read_to_string("tests/fixtures/o200k/o200k_base.tiktoken").unwrap();
    let specials = [
        ("<|endoftext|>".to_string(), 199_999u32),
        ("<|endofprompt|>".to_string(), 200_018u32),
    ];
    let tok = tiktokenizer::fromfile(&content, &specials).unwrap();
    for len in [512usize, 4096] {
        let input = repeatpattern(len);
        c.bench_function(&format!("tokenizer/encode/tiktoken/{len}"), |b| {
            b.iter(|| black_box(tok.encode(&input).unwrap()));
        });
    }
}

fn encodetiktokeninto(c: &mut Criterion) {
    let content = std::fs::read_to_string("tests/fixtures/o200k/o200k_base.tiktoken").unwrap();
    let specials = [
        ("<|endoftext|>".to_string(), 199_999u32),
        ("<|endofprompt|>".to_string(), 200_018u32),
    ];
    let tok = tiktokenizer::fromfile(&content, &specials).unwrap();
    for len in [512usize, 4096] {
        let input = repeatpattern(len);
        let mut scratch = mergescratch::default();
        let mut out = Vec::new();
        c.bench_function(&format!("tokenizer/encode_into/tiktoken/{len}"), |b| {
            b.iter(|| {
                out.clear();
                tok.encode_into(&input, &mut scratch, &mut out).unwrap();
                black_box(&out);
            });
        });
    }
}

criterion_group!(
    benches,
    applychain,
    applyidschain,
    encodebytefallback,
    decodebytefallback,
    encodebytelevel,
    encodebytelevelbatch,
    encodetiktoken,
    encodetiktokeninto
);
criterion_main!(benches);

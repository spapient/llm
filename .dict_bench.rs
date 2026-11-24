use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

fn payload_binary(weights: &[u32], values: &[u32], width: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(weights.len() * width);
    for value in weights {
        let id = values.binary_search(value).unwrap() as u32;
        out.extend_from_slice(&id.to_le_bytes()[..width]);
    }
    out
}

fn payload_hash(weights: &[u32], values: &[u32], width: usize) -> Vec<u8> {
    let indices: HashMap<u32, u32> = values
        .iter()
        .enumerate()
        .map(|(id, &bits)| (bits, id as u32))
        .collect();
    let mut out = Vec::with_capacity(weights.len() * width);
    for value in weights {
        let id = indices.get(value).copied().unwrap();
        out.extend_from_slice(&id.to_le_bytes()[..width]);
    }
    out
}

fn bench(mut run: impl FnMut() -> Vec<u8>) -> f64 {
    let mut samples = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        black_box(run());
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let unique = 4096usize;
    let mut values: Vec<u32> = (0..unique as u32)
        .map(|value| value.wrapping_mul(2_654_435_761))
        .collect();
    values.sort_unstable();
    let weights: Vec<u32> = (0..1_048_576)
        .map(|index| values[(index * 37) % unique])
        .collect();
    let expected = payload_binary(&weights, &values, 2);
    assert_eq!(payload_hash(&weights, &values, 2), expected);
    let binary = bench(|| payload_binary(black_box(&weights), black_box(&values), 2));
    let hash = bench(|| payload_hash(black_box(&weights), black_box(&values), 2));
    println!("binary={binary:.6}s hash={hash:.6}s speedup={:.2}x", binary / hash);
}

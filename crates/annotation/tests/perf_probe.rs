//! 临时性能探针：量 parse+render 的展开期成本；量完即删。

use super::*;
use crate::fixture::lex;
use std::hint::black_box;
use std::time::Instant;

fn bench(label: &str, code: &str, iterations: u32, step: impl Fn(&[syntax::Token])) {
    let tokens = lex(code);
    for _ in 0..1_000 {
        step(black_box(&tokens));
    }
    let start = Instant::now();
    for _ in 0..iterations {
        step(black_box(&tokens));
    }
    let elapsed = start.elapsed().as_nanos();
    println!(
        "{label}: {} iters, {} ns total, {:.1} ns/iter",
        iterations,
        elapsed,
        elapsed as f64 / iterations as f64
    );
}

#[test]
fn perf_probe() {
    const LAYER: &str = r#"out = [4096], name = "q", ops = ["gemv_i4"], in = 4096"#;
    const RUNTIME: &str = r#"name = $name, out = [$tokens, $dim], ops = [$op], role = $role"#;
    const MODEL: &str = r#"name = "gemv_i4", shape = [4, 4096], dtype = "f32""#;
    bench("layer", LAYER, 200_000, |code| {
        let declaration = syntax::parse(code).expect("合法");
        black_box(render::render(black_box(&declaration), "Head"));
    });
    bench("runtime", RUNTIME, 200_000, |code| {
        let declaration = syntax::parse_runtime(code).expect("合法");
        black_box(render::render_runtime(black_box(&declaration), "Head"));
    });
    bench("model", MODEL, 200_000, |code| {
        let declaration = syntax::parse_model(code).expect("合法");
        black_box(render::render_model(black_box(&declaration), "Q4"));
    });
}

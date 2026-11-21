//! 临时性能探针：量完即删。

use crate::decl::{parse, parse_model, parse_runtime};
use crate::fixture::lex;
use crate::render::{render, render_model, render_runtime};
use std::hint::black_box;
use std::time::Instant;

const ROUNDS: usize = 5;
const ITERATIONS: usize = 200_000;

fn best(mut measure: impl FnMut()) -> f64 {
    let mut samples: Vec<f64> = Vec::new();
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            measure();
        }
        samples.push(start.elapsed().as_nanos() as f64 / ITERATIONS as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[ROUNDS / 2]
}

#[test]
fn probe() {
    let layer_text = r#"name = "q", out = [8], ops = ["gemv_i4"]"#;
    let model_text = r#"name = "lm_head", shape = [4096, 4096], dtype = "f32", skeleton = "gemv""#;
    let runtime_text = r#"name = $name, out = [$tokens, $dim], ops = [$op], role = $role"#;

    let layer = parse(&lex(layer_text)).expect("layer");
    let model = parse_model(&lex(model_text)).expect("model");
    let runtime = parse_runtime(&lex(runtime_text)).expect("runtime");

    let render_layer = best(|| {
        black_box(render(black_box(&layer), black_box("Q")));
    });
    let render_model = best(|| {
        black_box(render_model(black_box(&model), black_box("LmHead")));
    });
    let render_runtime = best(|| {
        black_box(render_runtime(black_box(&runtime), black_box("Embedding")));
    });
    let full = best(|| {
        let parsed = parse(&lex(black_box(layer_text))).expect("layer");
        black_box(render(black_box(&parsed), black_box("Q")));
    });

    println!("render layer   {render_layer:8.1} ns");
    println!("render model   {render_model:8.1} ns");
    println!("render runtime {render_runtime:8.1} ns");
    println!("lex+parse+render {full:8.1} ns");
}

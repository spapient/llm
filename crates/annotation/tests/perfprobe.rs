use crate::decl::{Declaration, ModelDecl, RuntimeDecl};
use crate::render::{holes, render, render_model, render_runtime, unique};
use std::hint::black_box;
use std::time::Instant;

fn layer() -> Declaration {
    Declaration {
        name: "gemm".into(),
        out: vec![512, 768],
        input: Some(768),
        ops: vec!["matmul".into(), "bias".into()],
    }
}

fn runtime() -> RuntimeDecl {
    RuntimeDecl {
        name: "name".into(),
        out: vec!["dim".into(), "dim".into()],
        ops: vec!["matmul".into()],
        role: Some("role".into()),
    }
}

fn report(label: &str, mut run: impl FnMut()) {
    const ITERS: usize = 200_000;
    let mut rounds = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..ITERS {
            run();
        }
        rounds.push(start.elapsed().as_nanos() as f64 / ITERS as f64);
    }
    rounds.sort_by(f64::total_cmp);
    println!("{label:>22}: median {:>7.1} ns", rounds[2]);
}

fn long_layer() -> Declaration {
    Declaration {
        name: "multi_head_attention_projection_layer".into(),
        out: vec![512, 64, 12, 4096],
        input: Some(1024),
        ops: (0..6).map(|index| format!("operator_{index}")).collect(),
    }
}

fn old_unique<'a>(names: impl Iterator<Item = &'a String>) -> Vec<&'a str> {
    let mut seen: Vec<&str> = Vec::new();
    for name in names {
        if !seen.contains(&name.as_str()) {
            seen.push(name);
        }
    }
    seen
}

#[test]
fn unique_ab() {
    let runtime = runtime();
    report("unique new", || {
        black_box(unique(holes(black_box(&runtime))));
    });
    report("unique old", || {
        black_box(old_unique(holes(black_box(&runtime))));
    });
    report("unique new", || {
        black_box(unique(holes(black_box(&runtime))));
    });
    report("unique old", || {
        black_box(old_unique(holes(black_box(&runtime))));
    });
}

#[test]
fn overflow_report() {
    let declaration = long_layer();
    let code = render(&declaration, "crate::layers::attention::MultiHeadAttention");
    println!("render(long)   len {:>5} cap 1536", code.len());
    report("render(long)", || {
        black_box(render(
            black_box(&declaration),
            "crate::layers::attention::MultiHeadAttention",
        ));
    });
}

fn model() -> ModelDecl {
    ModelDecl {
        name: "attention_projection_gemm".into(),
        shape: [4096, 4096],
        dtype: "f32".into(),
        skeleton: "gemm".into(),
        const_name: "MODEL".into(),
    }
}

#[test]
fn model_report() {
    let declaration = model();
    let code = render_model(&declaration, "crate::layers::attention::Projection");
    println!("render_model   len {:>5} cap 3072", code.len());
    report("render_model", || {
        black_box(render_model(
            black_box(&declaration),
            "crate::layers::attention::Projection",
        ));
    });
}

#[test]
fn capacity_report() {
    let declaration = layer();
    let runtime = runtime();
    let static_code = render(&declaration, "Gemm");
    let runtime_code = render_runtime(&runtime, "Gemm");
    println!("render(static)  len {:>5} cap 1536", static_code.len());
    println!("render_runtime  len {:>5} cap 512", runtime_code.len());
}

#[test]
fn perf_report() {
    let declaration = layer();
    let runtime = runtime();
    report("render(static)", || {
        black_box(render(black_box(&declaration), "Gemm"));
    });
    report("render_runtime", || {
        black_box(render_runtime(black_box(&runtime), "Gemm"));
    });
}

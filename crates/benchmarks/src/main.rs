//! 统一基准入口：一个 bench target 声明全部 src 分组。
//!
//! 合并前 23 个 `[[bench]]` 目标各自链接、各起一个进程；合并后只链接一次、起一个进程。
//! 分组名与用例名逐字保留，`target/src/<group>/<benchmark>/new/estimates.json` 的路径
//! 不变，`tests/regress` 照旧按路径读取。选跑某一组仍可用 src 的过滤参数：
//! `cargo bench -p llm-benchmarks -- gemv_f32`。

mod attention;
mod fixtures;
mod gemm;
mod matmul;
mod memory;
mod model;
mod protocol;
mod text;
mod vector;
mod weights;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    vector::elementwise,
    vector::gemv,
    vector::softmax,
    vector::rope,
    vector::rmsnorm,
    vector::silu,
    attention::flash_decode,
    attention::prefill,
    attention::decode,
    gemm::dense,
    gemm::quant,
    matmul::dense,
    matmul::int8,
    matmul::threads,
    memory::read,
    memory::write,
    weights::archive,
    weights::load,
    model::decode,
    model::train,
    text::tokenizer,
    text::wire,
    protocol::span,
    protocol::negotiate,
);
criterion_main!(benches);

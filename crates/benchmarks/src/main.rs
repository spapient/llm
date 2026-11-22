//! 统一基准入口：一个 bench target 声明全部 criterion 分组。

mod attention;
mod fixtures;
mod io;
mod kernel;
mod model;
mod parse;
mod peak;
mod protocol;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    kernel::elementwise,
    kernel::gemv,
    kernel::gemv_quant,
    kernel::softmax,
    kernel::rope,
    kernel::rmsnorm,
    kernel::silu,
    attention::flash_decode,
    attention::prefill,
    attention::decode,
    kernel::gemm_dense,
    kernel::gemm_quant,
    kernel::matmul_dense,
    kernel::matmul_int8,
    kernel::matmul_threads,
    peak::flops_peak,
    kernel::mem_read,
    kernel::mem_write,
    io::weights::archive,
    io::weights::load,
    model::decode,
    model::train,
    io::text::tokenizer,
    io::text::wire,
    protocol::span,
    protocol::negotiate,
    parse::parse,
);
criterion_main!(benches);

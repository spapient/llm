//! 统一基准入口：一个 bench target 声明全部 criterion 分组。
//!
//! 合并前 23 个 `[[bench]]` 目标各自链接、各起一个进程；合并后只链接一次、起一个进程。
//! 分组名与用例名逐字保留，`target/criterion/<group>/<benchmark>/new/estimates.json` 的路径
//! 不变，`tests/regress` 照旧按路径读取。选跑某一组仍可用 criterion 的过滤参数：
//! `cargo bench -p llm-benchmarks -- gemv_f32`。
//!
//! 全部基准平铺在 `src/` 一层：`fixtures` 是夹具与唯一失败出口，`kernel` / `attention` /
//! `io` / `protocol` / `model` 各按被测对象分工，没有子目录。

mod attention;
mod fixtures;
mod io;
mod kernel;
mod model;
mod protocol;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    kernel::elementwise,
    kernel::gemv,
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
);
criterion_main!(benches);

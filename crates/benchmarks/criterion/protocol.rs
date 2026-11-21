//! 协议层基准：`Span::parse` 的四种写法与 `cap::negotiate` 的一次能力协商。
//!
//! 两者都是纯 CPU 路径，不起编译器、不碰设备，任意 CI 都能编译与运行。只出实测耗时；
//! 是否回退由 `tests/regress` 的基线判定负责。

use criterion::Criterion;
use cutile::builtin::Catalogue;
use cutile::cap;
use prof::atom::Span;
use std::hint::black_box;

/// 精确、上界、区间、开区间四种写法各测一次解析。
pub(crate) fn span(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_parse");
    group.bench_function("exact", |b| {
        b.iter(|| Span::parse(black_box("1024")).expect("精确写法"))
    });
    group.bench_function("at_most", |b| {
        b.iter(|| Span::parse(black_box("<=512")).expect("上界写法"))
    });
    group.bench_function("range", |b| {
        b.iter(|| Span::parse(black_box("16..64")).expect("区间写法"))
    });
    group.bench_function("open_range", |b| {
        b.iter(|| Span::parse(black_box("16..")).expect("开区间写法"))
    });
    group.finish();
}

/// 内置目录在 `cpu` 档案上协商 `gemv_i4`：应命中首档。
pub(crate) fn negotiate(c: &mut Criterion) {
    let catalogue = Catalogue::try_builtin().expect("内置目录");
    let op = catalogue.op("gemv_i4").expect("内置算子 gemv_i4");
    let profile = catalogue.profile("cpu").expect("内置 cpu 档案");

    let mut group = c.benchmark_group("cutile/negotiate");
    group.bench_function("gemv_i4_on_cpu", |bench| {
        bench.iter(|| cap::negotiate(black_box(op), black_box(profile)).expect("cpu 档案应命中首档"))
    });
    group.finish();
}

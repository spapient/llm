//! 协议层基准：`Span::parse` 的四种写法与 `cap::negotiate` 的一次能力协商。

use crate::fixtures::{checked_or_exit, found};
use criterion::Criterion;
use cutile::atom::Span;
use cutile::builtin::Catalogue;
use cutile::cap;
use std::hint::black_box;

/// `(用例名, 字面量)`：精确、上界、区间、开区间。
const SPANS: [(&str, &str); 4] = [
    ("exact", "1024"),
    ("at_most", "<=512"),
    ("range", "16..64"),
    ("open_range", "16.."),
];

/// 四种写法各测一次解析。
pub(crate) fn span(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_parse");
    for (name, text) in SPANS {
        checked_or_exit(Span::parse(text), "Span 写法");
        group.bench_function(name, |bench| {
            bench.iter(|| black_box(Span::parse(black_box(text))))
        });
    }
    group.finish();
}

/// 内置目录在 `cpu` 档案上协商 `gemv_i4`：应命中首档。
pub(crate) fn negotiate(c: &mut Criterion) {
    let catalogue = checked_or_exit(Catalogue::try_builtin(), "内置目录");
    let op = found(catalogue.op("gemv_i4"), "内置算子 gemv_i4");
    let profile = found(catalogue.profile("cpu"), "内置 cpu 档案");

    let mut group = c.benchmark_group("cutile/negotiate");
    group.bench_function("gemv_i4_on_cpu", |bench| {
        bench.iter(|| black_box(cap::negotiate(black_box(op), black_box(profile))))
    });
    group.finish();
}

//! 协议层基准：`Span::parse` 的四种写法与 `cap::negotiate` 的一次能力协商。
//!
//! 两者都是纯 CPU 路径，不起编译器、不碰设备，任意 CI 都能编译与运行。只出实测耗时；
//! 是否回退由 `tests/regress` 的基线判定负责。
//!
//! 写法与用例名放在一张表里：四种写法的字面量只出现一次，装配阶段按同一张表校验，
//! 测量阶段只搬运 `Result`。

use crate::fixtures::checked;
use criterion::Criterion;
use cutile::builtin::Catalogue;
use cutile::cap;
use prof::atom::Span;
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
        checked(Span::parse(text), "Span 写法");
        group.bench_function(name, |bench| {
            bench.iter(|| black_box(Span::parse(black_box(text))))
        });
    }
    group.finish();
}

/// 内置目录在 `cpu` 档案上协商 `gemv_i4`：应命中首档。
pub(crate) fn negotiate(c: &mut Criterion) {
    let catalogue = checked(Catalogue::try_builtin(), "内置目录");
    let op = checked(catalogue.op("gemv_i4"), "内置算子 gemv_i4");
    let profile = checked(catalogue.profile("cpu"), "内置 cpu 档案");

    let mut group = c.benchmark_group("cutile/negotiate");
    group.bench_function("gemv_i4_on_cpu", |bench| {
        bench.iter(|| black_box(cap::negotiate(black_box(op), black_box(profile))))
    });
    group.finish();
}

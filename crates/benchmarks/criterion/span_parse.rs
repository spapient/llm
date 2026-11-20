//! `Span::parse` 的微基准：测量四种写法的解析成本。
//!
//! 这里只负责"测量"，产出实测耗时；是否回退由 `tool::regress` 的基线判定负责，
//! 两者分工，bench 本身不写性能目标数字。

use criterion::{Criterion, criterion_group, criterion_main};
use prof::atom::Span;
use std::hint::black_box;

fn bench_parse(c: &mut Criterion) {
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

criterion_group!(benches, bench_parse);
criterion_main!(benches);

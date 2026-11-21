//! `cap::negotiate` 的微基准：内置目录在 `cpu` 档案上的一次能力协商。
//!
//! 与其他 criterion 基准同一分工：这里只产出实测耗时（预热、采样、测量时长由 criterion
//! 自己配置），是否回归由 `tools::regress` 的基线判定负责；本文件不写性能目标数字。
//! 协商是纯 CPU 路径，不起编译器、不碰设备，因此可以在任意 CI 上编译与运行。

use criterion::{Criterion, criterion_group, criterion_main};
use cutile::builtin::Catalogue;
use cutile::cap;
use std::hint::black_box;

fn bench_negotiate(c: &mut Criterion) {
    let catalogue = Catalogue::try_builtin().expect("内置目录");
    let op = catalogue.op("gemv_i4").expect("内置算子 gemv_i4");
    let profile = catalogue.profile("cpu").expect("内置 cpu 档案");

    let mut group = c.benchmark_group("cutile/negotiate");
    group.bench_function("gemv_i4_on_cpu", |bench| {
        bench
            .iter(|| cap::negotiate(black_box(op), black_box(profile)).expect("cpu 档案应命中首档"))
    });
    group.finish();
}

criterion_group!(benches, bench_negotiate);
criterion_main!(benches);

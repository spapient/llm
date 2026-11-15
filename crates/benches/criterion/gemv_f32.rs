//! `gemv_f32::forward` 的微基准：一次 decode 步（1×4096×4096）的参考实现。
//!
//! 为什么是这个算子：它是 decode 阶段每层每 token 都要跑、且是唯一一条"权重全读一遍"的路径。
//! 它同时是**第一个接入 `tool::regress` 基线判定的真实算子**（判据见 `tests/regress.rs`）：
//! 微基准的合成算术（elementwise）只能说明"机器有多快"，只有真实形状才说得清"这一层要多久"。
//!
//! 口径：**ns/iter，越小越好**。不报 FLOPs：`gemv` 的二进制位宽随权重格式变（f32 / i4 / 打包），
//! 同一形状的"理论算力"会因为格式不同而差 4 倍，跨格式不可比；时间是可比的。
//!
//! 预热轮数、测量时长、采样数都不写死——由 criterion 配置决定。本文件只声明"一次迭代做什么"。

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use op::gemv_f32;
use std::hint::black_box;

/// 层形状：`4096 → 4096`（hidden 与中间维度都取这个量级，是 decode 的典型形状）。
const SHAPE: [usize; 2] = [4096, 4096];

fn bench_forward(c: &mut Criterion) {
    let [rows, cols] = SHAPE;
    let weights: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.001).sin()).collect();
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.002).cos()).collect();
    let mut out = vec![0.0f32; rows];

    let mut group = c.benchmark_group("gemv_f32");
    // 字节吞吐（权重全读）也报一份：CI 归档里"时间"与"带宽"两条口径都在，读者不必自己换算。
    group.throughput(Throughput::Bytes((rows * cols * size_of::<f32>()) as u64));
    group.bench_function("decode_1x4096x4096", |bench| {
        bench.iter(|| {
            gemv_f32::forward(black_box(&x), black_box(&weights), black_box(&mut out))
                .expect("形状与缓冲等长")
        })
    });

    group.finish();
}

criterion_group!(benches, bench_forward);
criterion_main!(benches);

//! 单机算子基准：逐元素 / 行内算子、小 K 稠密与量化 GEMM、大 K GEMM 与线程余量、内存搬参照。
//!
//! 四类共用一条契约：几何在装配阶段用同一条内核校验一次（[`checked`]），测量阶段只把 `Result`
//! 交给 `black_box`。标量与分块的对照由 [`dual`] 统一给出——同一份 A/B 缓冲、同一组形状，
//! 只有遍历次序不同，所以两组永远指向同一份算术。
//!
//! 报 ns/iter 与读入字节吞吐；`gemm_threads` 报乘加量吞吐（`GElem/s`），因为那几条路径读入
//! 字节数相同、差别只在算术与并行度。回退判定在 `tests/regress`，本文件不写性能目标数字。

use crate::fixtures::{checked, ok};
use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use op::add_f32::Add;
use op::gemm::{self, Gemm};
use op::quant::I4;
use op::{
    elementwise, fast, gemm_f32, gemm_i2, gemm_i4, gemm_i8, gemv_f32, gemv_i4, matmul, matmul_i8,
    rmsnorm_f32, rope_f32, silu_f32, softmax_f32,
};
use std::fmt;
use std::hint::black_box;
use std::thread;

/// 缓冲尺寸与 `m·n·k` 对不上时的失败文案。
const GEOM: &str = "缓冲尺寸与 m·n·k 一致";

/// decode 典型行宽（hidden 维度）。
const ROW: usize = 4096;

/// elementwise 的向量长度（输入与输出各 256 KiB）。
const VECTOR: usize = 1 << 16;

/// 输出行数 `m`、列数 `n`。
const M: usize = 256;
const N: usize = 256;

/// 大 K 的收缩轴（prefill 主链路代表值）。
const K: usize = 4096;

/// int4 每字节 2 个码（与 [`I4`] 的 `cpb` 同一个值）。
const I4_CPB: usize = 2;

/// 量化权重的分组元素数（生产格式固定 32）。
const GROUP: usize = 32;

/// 参与线程对照的形状 `(m, n, k)`：prefill 主链路与近方阵各一档。
const SHAPES: [(usize, usize, usize); 2] = [(256, 256, 4096), (512, 512, 512)];

/// 一行 decode 的输入与同长输出。
fn row() -> (Vec<f32>, Vec<f32>) {
    let x = (0..ROW).map(|index| index as f32 * 0.25 - 2.0).collect();
    (x, vec![0.0f32; ROW])
}

/// `elementwise::forward::<Add>`：1<<16 个 f32 的逐元素加法。
pub(crate) fn elementwise(c: &mut Criterion) {
    let a: Vec<f32> = (0..VECTOR).map(|i| i as f32 * 0.5 - 1.0).collect();
    let b: Vec<f32> = (0..VECTOR).map(|i| i as f32 * 0.25 + 0.5).collect();
    let mut out = vec![0.0f32; VECTOR];
    checked(elementwise::forward::<Add>(&a, &b, &mut out), "缓冲等长");

    let mut group = c.benchmark_group("elementwise/add_f32");
    group.throughput(Throughput::Elements(VECTOR as u64));
    group.bench_function("forward", |iter| {
        iter.iter(|| {
            black_box(elementwise::forward::<Add>(
                black_box(&a),
                black_box(&b),
                black_box(&mut out),
            ))
        })
    });
    group.finish();
}

/// `gemv_f32::forward`：一次 decode 步（1×4096×4096），权重全读一遍。
pub(crate) fn gemv(c: &mut Criterion) {
    let weights: Vec<f32> = (0..ROW * ROW).map(|i| (i as f32 * 0.001).sin()).collect();
    let x: Vec<f32> = (0..ROW).map(|i| (i as f32 * 0.002).cos()).collect();
    let mut out = vec![0.0f32; ROW];
    checked(gemv_f32::forward(&x, &weights, &mut out), "形状与缓冲等长");

    let mut group = c.benchmark_group("gemv_f32");
    // 时间与带宽两条口径都归档：读者不必自己换算。
    group.throughput(Throughput::Bytes((ROW * ROW * size_of::<f32>()) as u64));
    group.bench_function("decode_1x4096x4096", |iter| {
        iter.iter(|| {
            black_box(gemv_f32::forward(
                black_box(&x),
                black_box(&weights),
                black_box(&mut out),
            ))
        })
    });
    group.finish();
}

/// int4 decode 的两档：同一份打包权重、同一行 `x`，标量叶与向量通道。
///
/// 生产 decode 主链路就是这一档（[`gemv_i4`]），所以形状取 `1×4096×4096`：权重 `n` 行各
/// [`ROW`]/2 个字节（int4 每字节 2 个码），标度 `n × groups` 逐行逐组不同——量化权重的真实读法。
/// 两档指向同一份数据、同一个输出缓冲，差的只有内层算术：标量叶每码查表并串行累加，向量通道
/// 按码值直接换算、4 列共享一次 `x` 载入、权重按列块驻留。开测前先核对两档给出同一个答案，
/// 否则测的是两段不同的算术。
pub(crate) fn gemv_quant(c: &mut Criterion) {
    let row = ROW / I4_CPB;
    let groups = ROW / GROUP;
    let weights: Vec<u8> = (0..ROW * row).map(|i| ((i * 37) % 251) as u8).collect();
    let scales: Vec<f32> = (0..ROW * groups).map(|i| 0.5 + ((i * 7) % 13) as f32 * 0.0625).collect();
    let x: Vec<f32> = (0..ROW).map(|i| (i as f32 * 0.002).cos()).collect();
    let mut out = vec![0.0f32; ROW];

    // 标量叶：与打包格式回退路径调用的是同一个 `dot_row`。
    let scalar = |out: &mut [f32]| {
        for j in 0..ROW {
            out[j] = op::gemv::packed::dot_row(
                &weights[j * row..][..row],
                &x,
                &scales[j * groups..][..groups],
                GROUP / I4_CPB,
                GROUP,
                I4_CPB,
                I4.table,
            );
        }
    };

    scalar(&mut out);
    let expected = out.clone();
    checked(gemv_i4::forward(&x, &weights, &scales, &mut out), GEOM);
    for (j, (&got, &want)) in out.iter().zip(&expected).enumerate() {
        let slack = 1e-5 * want.abs().max(1.0);
        assert!((got - want).abs() <= slack, "第 {j} 列：向量通道 {got} 偏离标量叶 {want}");
    }

    let mut group = c.benchmark_group("gemv_i4");
    group.throughput(Throughput::Bytes((row * ROW + 4 * scales.len()) as u64));
    group.bench_function("scalar_decode_1x4096x4096", |iter| {
        iter.iter(|| black_box(scalar(black_box(&mut out))))
    });
    group.bench_function("vector_decode_1x4096x4096", |iter| {
        iter.iter(|| {
            black_box(gemv_i4::forward(
                black_box(&x),
                black_box(&weights),
                black_box(&scales),
                black_box(&mut out),
            ))
        })
    });
    group.finish();
}

/// `softmax_f32::forward`：decode 一行 4096 个 f32 的三趟数值稳定归一化。
pub(crate) fn softmax(c: &mut Criterion) {
    let (x, mut out) = row();
    checked(softmax_f32::forward(&x, &mut out), "输入与输出等长");

    let mut group = c.benchmark_group("softmax_f32");
    group.throughput(Throughput::Elements(ROW as u64));
    group.bench_function("decode_1x4096", |iter| {
        iter.iter(|| black_box(softmax_f32::forward(black_box(&x), black_box(&mut out))))
    });
    group.finish();
}

/// `rope_f32::forward`：decode 一行 4096 个 f32 的二维位置旋转（位置取 7）。
pub(crate) fn rope(c: &mut Criterion) {
    let (x, mut out) = row();
    checked(rope_f32::forward(7, &x, &mut out), "输入与输出等长");

    let mut group = c.benchmark_group("rope_f32");
    group.throughput(Throughput::Elements(ROW as u64));
    group.bench_function("decode_1x4096", |iter| {
        iter.iter(|| black_box(rope_f32::forward(7, black_box(&x), black_box(&mut out))))
    });
    group.finish();
}

/// `rmsnorm_f32::forward`：decode 一行 4096 个 f32 的均方根归一化。
pub(crate) fn rmsnorm(c: &mut Criterion) {
    let (x, mut out) = row();
    let gamma: Vec<f32> = (0..ROW).map(|i| 0.5 + i as f32 * 0.01).collect();
    checked(rmsnorm_f32::forward(&x, &gamma, &mut out), "三个缓冲等长");

    let mut group = c.benchmark_group("rmsnorm_f32");
    group.throughput(Throughput::Elements(ROW as u64));
    group.bench_function("decode_1x4096", |iter| {
        iter.iter(|| {
            black_box(rmsnorm_f32::forward(
                black_box(&x),
                black_box(&gamma),
                black_box(&mut out),
            ))
        })
    });
    group.finish();
}

/// `silu_f32::forward`：decode 一行 4096 个 f32 的逐元素激活。
pub(crate) fn silu(c: &mut Criterion) {
    let x: Vec<f32> = (0..ROW).map(|i| i as f32 * 0.25 - 4.0).collect();
    let mut out = vec![0.0f32; ROW];
    checked(silu_f32::forward(&x, &mut out), "输入与输出等长");

    let mut group = c.benchmark_group("silu_f32");
    group.throughput(Throughput::Elements(ROW as u64));
    group.bench_function("decode_1x4096", |iter| {
        iter.iter(|| black_box(silu_f32::forward(black_box(&x), black_box(&mut out))))
    });
    group.finish();
}

/// 同一份 A/B 缓冲上的两档内核：标量参考与分块执行路径。
///
/// `names[0]` 配 `scalar`、`names[1]` 配 `tiled`；分组名与用例名是 `tests/regress` 读取的契约，
/// 由调用点逐字给出，不在这里拼。
fn dual<A, C, E: fmt::Display>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    (a, b, out): (&[A], &[A], &mut [C]),
    (m, n, k): (usize, usize, usize),
    names: [&str; 2],
    scalar: fn(&[A], &[A], usize, usize, usize, &mut [C]) -> Result<(), E>,
    tiled: fn(&[A], &[A], usize, usize, usize, &mut [C]) -> Result<(), E>,
) {
    checked(scalar(a, b, m, n, k, &mut *out), GEOM);
    group.bench_function(names[0], |iter| {
        iter.iter(|| {
            black_box(scalar(
                black_box(a),
                black_box(b),
                m,
                n,
                k,
                black_box(&mut *out),
            ))
        })
    });
    group.bench_function(names[1], |iter| {
        iter.iter(|| {
            black_box(tiled(
                black_box(a),
                black_box(b),
                m,
                n,
                k,
                black_box(&mut *out),
            ))
        })
    });
}

/// 同一份 A/B 缓冲上的两档 `gemm` 内核：标量参考与 tile 执行路径。
///
/// `gemm` 的收缩轴固定为 [`gemm::K`]，所以这里只传 `(m, n)`；标量路径 `gemm::matmul`
/// 不返回 `Result`，几何由 tiled 路径的 `check_dims` 一次性校验。
fn gemm_pair<G: Gemm>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    (a, b, out): (&[G::Elem], &[G::Elem], &mut [f32]),
    (m, n): (usize, usize),
    (scalar, tiled): (&str, &str),
) {
    checked(gemm::forward::<G>(a, b, m, n, &mut *out), GEOM);
    group.bench_function(scalar, |iter| {
        iter.iter(|| gemm::matmul::<G>(black_box(a), black_box(b), m, n, black_box(&mut *out)))
    });
    group.bench_function(tiled, |iter| {
        iter.iter(|| {
            black_box(checked(
                gemm::forward::<G>(black_box(a), black_box(b), m, n, black_box(&mut *out)),
                GEOM,
            ))
        })
    });
}

/// f32 小 K：`prefill_256x256_k8` 走执行路径，`scalar_*` 走标量参考。
///
/// `K = 8` 时一行 A 就在寄存器里，分块实测慢 2~3 倍，所以这条格式不分块；本基准把这点留在
/// 台架上可复现。量化格式的分块收益见 [`gemm_quant`]。
pub(crate) fn gemm_dense(c: &mut Criterion) {
    let k = gemm::K;
    let a: Vec<f32> = (0..M * k).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
    let b: Vec<f32> = (0..k * N).map(|i| (i % 7) as f32 * 0.2 - 0.4).collect();
    let mut out = vec![0.0f32; M * N];

    let mut group = c.benchmark_group("gemm_f32");
    group.throughput(Throughput::Bytes(
        ((a.len() + b.len()) * size_of::<f32>()) as u64,
    ));
    gemm_pair::<gemm_f32::GemmF32>(
        &mut group,
        (&a, &b, &mut out),
        (M, N),
        ("scalar_256x256_k8", "prefill_256x256_k8"),
    );
    group.finish();
}

/// 量化三格式：`scalar_*` 逐 `(i, j)` 重解位流，`tiled_*` 只解一次再按列块整数累加。
pub(crate) fn gemm_quant(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_quant");
    quant_pair::<gemm_i2::GemmI2>(&mut group, "i2_256x256_k8");
    quant_pair::<gemm_i4::GemmI4>(&mut group, "i4_256x256_k8");
    quant_pair::<gemm_i8::GemmI8>(&mut group, "i8_256x256_k8");
    group.finish();
}

/// 一种量化格式的两档：同一份 A/B 缓冲，标量参考与 tile 执行路径。
fn quant_pair<G: Gemm>(group: &mut BenchmarkGroup<'_, WallTime>, name: &str) {
    let a = G::sample_a(M);
    let b = G::sample_b(N);
    let mut out = vec![0.0f32; M * N];
    group.throughput(Throughput::Bytes((a.len() + b.len()) as u64));
    let scalar = format!("scalar_{name}");
    let tiled = format!("tiled_{name}");
    gemm_pair::<G>(group, (&a, &b, &mut out), (M, N), (&scalar, &tiled));
}

/// 大 K 稠密 f32：朴素 i→j→p 与分块 rank-1 更新。
pub(crate) fn matmul_dense(c: &mut Criterion) {
    let a: Vec<f32> = (0..M * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let b: Vec<f32> = (0..K * N).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    let mut out = vec![0.0f32; M * N];

    let mut group = c.benchmark_group("matmul_f32_256x256_k4096");
    group.throughput(Throughput::Bytes(
        ((a.len() + b.len()) * size_of::<f32>()) as u64,
    ));
    dual(
        &mut group,
        (&a, &b, &mut out),
        (M, N, K),
        ["scalar", "tiled"],
        matmul::forward_scalar,
        matmul::forward_tiled,
    );
    group.finish();

    projection(c);
}

/// 线性层布局 `[rows, k] × [n, k]`（权重不转置）：逐行 `gemv`、tile 与寄存器分块 + 常驻池对照。
///
/// 这一组量的是"权重读几遍"：逐行路径每行读一遍整块权重，`tiled` 一行权重读一遍服务全部行，
/// `packed` 再叠加 8×8 寄存器分块与多线程。三者算术相同，输出逐位一致。
fn projection(c: &mut Criterion) {
    let rows = M;
    let x: Vec<f32> = (0..rows * K)
        .map(|i| ((i % 5) as f32) * 0.25 - 0.5)
        .collect();
    let w: Vec<f32> = (0..N * K).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    let mut out = vec![0.0f32; rows * N];

    let mut group = c.benchmark_group("matmul_tn_f32_256x256_k4096");
    group.throughput(Throughput::Bytes(
        ((x.len() + w.len()) * size_of::<f32>()) as u64,
    ));
    checked(matmul::forward_tn_tiled(&x, &w, rows, N, K, &mut out), GEOM);
    group.bench_function("row_by_row_gemv", |iter| {
        iter.iter(|| {
            for row in 0..rows {
                black_box(ok(gemv_f32::forward(
                    black_box(&x[row * K..][..K]),
                    black_box(&w),
                    black_box(&mut out[row * N..][..N]),
                )));
            }
        })
    });
    group.bench_function("tiled", |iter| {
        iter.iter(|| {
            black_box(matmul::forward_tn_tiled(
                black_box(&x),
                black_box(&w),
                black_box(rows),
                black_box(N),
                black_box(K),
                black_box(&mut out),
            ))
        })
    });
    group.bench_function("packed", |iter| {
        iter.iter(|| {
            fast::gemm_f32_tn(
                black_box(&x),
                black_box(&w),
                black_box(rows),
                black_box(N),
                black_box(K),
                black_box(&mut out),
            )
        })
    });
    group.finish();
}

/// 大 K int8：标量参考与分块 tile，累加器都是 i32。
pub(crate) fn matmul_int8(c: &mut Criterion) {
    let a: Vec<i8> = (0..M * K).map(|i| ((i % 64) as i8) - 32).collect();
    let b: Vec<i8> = (0..K * N).map(|i| ((i % 64) as i8) - 32).collect();
    let mut out = vec![0i32; M * N];

    let mut group = c.benchmark_group("matmul_i8_256x256_k4096");
    group.throughput(Throughput::Bytes(
        ((a.len() + b.len()) * size_of::<i8>()) as u64,
    ));
    dual(
        &mut group,
        (&a, &b, &mut out),
        (M, N, K),
        ["scalar", "tiled"],
        matmul_i8::forward_scalar,
        matmul_i8::forward_tiled,
    );
    group.finish();
}

/// 单线程分块、单线程快速内核、快速内核线程化（i8 只有分块内核及其线程化版本）。
pub(crate) fn matmul_threads(c: &mut Criterion) {
    let parallel = cores();
    let mut group = c.benchmark_group("gemm_threads");
    group.sample_size(15);
    f32_paths(&mut group);
    i8_paths(&mut group, parallel);
    group.finish();
}

/// 可用核数（取不到时按 1 算，不猜）。
fn cores() -> usize {
    thread::available_parallelism().map_or(1, |count| count.get())
}

/// 把 `m` 行切给 `parallel` 个线程：每线程只碰自己那段输出行。
///
/// 核数由调用点一次问出并传进来：`available_parallelism` 是系统调用，落在计时闭包里会把一次
/// 调度查询的开销摊进被测内核。
fn spread<T: Send>(
    parallel: usize,
    m: usize,
    n: usize,
    out: &mut [T],
    run: impl Fn(usize, usize, &mut [T]) + Sync,
) {
    let chunk = m.div_ceil(parallel.min(m));
    thread::scope(|scope| {
        for (block, part) in out.chunks_mut(chunk * n).enumerate() {
            let run = &run;
            let start = block * chunk;
            scope.spawn(move || run(start, part.len() / n, part));
        }
    });
}

/// f32 三档：单线程分块、单线程快速内核、快速内核的线程化版本。
fn f32_paths(group: &mut BenchmarkGroup<'_, WallTime>) {
    for (m, n, k) in SHAPES {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
        let mut out = vec![0.0f32; m * n];
        let shape = format!("{m}x{n}x{k}");
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));

        group.bench_function(BenchmarkId::new("f32", &shape), |iter| {
            iter.iter(|| {
                matmul::matmul_tiled(black_box(&a), black_box(&b), m, n, k, black_box(&mut out));
            });
        });
        group.bench_function(BenchmarkId::new("f32_fast_single", &shape), |iter| {
            iter.iter(|| {
                fast::gemm_f32_single(black_box(&a), black_box(&b), m, n, k, black_box(&mut out));
            });
        });
        group.bench_function(BenchmarkId::new("f32_fast_threads", &shape), |iter| {
            iter.iter(|| {
                fast::gemm_f32(black_box(&a), black_box(&b), m, n, k, black_box(&mut out));
            });
        });
    }
}

/// i8 两档：单线程分块与线程化分块（快速内核只覆盖 f32）。
fn i8_paths(group: &mut BenchmarkGroup<'_, WallTime>, parallel: usize) {
    for (m, n, k) in SHAPES {
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 13) as i8) - 6).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i % 11) as i8) - 5).collect();
        let mut out = vec![0i32; m * n];
        let shape = format!("{m}x{n}x{k}");
        group.throughput(Throughput::Elements((2 * m * n * k) as u64));
        checked(matmul_i8::forward_tiled(&a, &b, m, n, k, &mut out), GEOM);

        group.bench_function(BenchmarkId::new("i8", &shape), |iter| {
            iter.iter(|| {
                black_box(matmul_i8::forward_tiled(
                    black_box(&a),
                    black_box(&b),
                    m,
                    n,
                    k,
                    black_box(&mut out),
                ))
            });
        });
        group.bench_function(BenchmarkId::new("i8_threads", &shape), |iter| {
            iter.iter(|| {
                spread(parallel, m, n, &mut out, |start, rows, part| {
                    black_box(ok(matmul_i8::forward_tiled(
                        &a[start * k..(start + rows) * k],
                        &b,
                        rows,
                        n,
                        k,
                        part,
                    )));
                });
                black_box(out.len())
            })
        });
    }
}

/// 三档工作集对应三级存储（16 KiB / 1 MiB / 64 MiB），每档读、写分开测。
///
/// 只流式读权重的 GEMV 可以跑过拷贝上限，把两个方向合成一个数，"这层卡在哪一侧"就再也判断
/// 不了（`nn::Bounds` 的分账口径与此相同）。小档在迭代内多跑几趟，让计时噪声不盖过信号。
/// 读用无符号整数异或归约而不是浮点求和：浮点加法默认不可重排，编译器会把归约串行化，测到的
/// 是加法延迟而不是搬运速率；搬运带宽与元素类型无关，整数读同样代表这一档的读上界。
const TIERS: [(&str, &str, usize, usize); 3] = [
    ("l1", "16kib", 2048, 64),
    ("l2", "1mib", 128 * 1024, 8),
    ("dram", "64mib", 8 * 1024 * 1024, 1),
];

/// 每档的填充样式：不同值避免编译器把三档的读写折叠成同一份常量。
const FILLS: [u64; 3] = [
    0x0101_0101_0101_0101,
    0x0202_0202_0202_0202,
    0x0303_0303_0303_0303,
];

/// 读一遍并做异或归约。
fn xor_read(words: &[u64]) -> u64 {
    let mut acc = 0u64;
    for word in words {
        acc ^= *word;
    }
    acc
}

/// 读方向的三档上界。
pub(crate) fn mem_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("mem_stream");
    for (index, (label, size, words, rounds)) in TIERS.into_iter().enumerate() {
        let data = vec![FILLS[index]; words];
        group.throughput(Throughput::Bytes(
            (rounds * words * size_of::<u64>()) as u64,
        ));
        group.bench_function(format!("{label}_read_{size}"), |iter| {
            iter.iter(|| {
                let mut acc = 0u64;
                for _ in 0..rounds {
                    acc ^= xor_read(black_box(&data));
                }
                black_box(acc)
            })
        });
    }
    group.finish();
}

/// 写方向的三档上界：`fill` 直接落到本机的写路径（大块时通常是流式存储指令）。
pub(crate) fn mem_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("mem_stream");
    for (index, (label, size, words, rounds)) in TIERS.into_iter().enumerate() {
        let mut data = vec![0u64; words];
        group.throughput(Throughput::Bytes(
            (rounds * words * size_of::<u64>()) as u64,
        ));
        group.bench_function(format!("{label}_write_{size}"), |iter| {
            iter.iter(|| {
                for _ in 0..rounds {
                    data.fill(black_box(FILLS[index]));
                }
                black_box(&data);
            })
        });
    }
    group.finish();
}

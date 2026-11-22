//! 职责：f32 FMA 端口峰值的单线程与整机测定。

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use criterion::{BenchmarkGroup, Criterion, Throughput, measurement::WallTime};
use op::fast;

/// f32 FMA 端口峰值：寄存器常驻、多条累加链互不依赖，用来把 GF/s 换算成峰值百分比。
///
/// 只有先测出本机峰值，「GEMM 达 90% 端口」才是结论；引用别处的时钟数只是猜。
///
/// **为什么要扫链数而不是只测一个**：一条链串行依赖延迟，链太多又溢出到栈，两者都会把
/// 「端口峰值」量成「延迟」或「访存带宽」。链数扫 4/6/8 三档取平台值：4 与 6 只差 0.2%
/// （平台），8 掉到 83 GFLOP/s（退化）。这不是 best-of-N——best-of-N 是同一口径重复挑最好，
/// 平台判据是「换了口径结果不变」。
pub(crate) fn flops_peak(c: &mut Criterion) {
    let mut group = c.benchmark_group("flops_peak");
    for chains in 0..3 {
        match chains {
            0 => peak_of::<4>(&mut group),
            1 => peak_of::<6>(&mut group),
            _ => peak_of::<8>(&mut group),
        }
    }
    group.finish();
    machine_peak(c);
    sme_peak(c);
}

/// 每链 16 个 f32 即 4 条 128 位寄存器：`C × 4 ≤ 32` 才不溢出，故 `C ≤ 8`。
const LANES: usize = 16;
const ROUNDS: usize = 4096;

/// 探针一轮的浮点次数：4 条 `fmopa` × 16×16 的 f32 瓦片 × 2 flop（乘 + 加）。
const FLOPS_PER_ROUND: usize = 4 * 16 * 16 * 2;
const STEP: [f32; LANES] = [1.000_000_1; LANES];

fn peak_of<const C: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.throughput(Throughput::Elements((2 * LANES * C * ROUNDS) as u64));
    group.bench_function(format!("f32_chain_{C}"), |iter| {
        iter.iter(|| fma_run::<C>(black_box(&mut [[0.0f32; LANES]; C]), black_box(ROUNDS)));
    });
}

/// 整机峰值：K7/K8 的分母。异核机器上「核数 × 单核峰值」是虚构的分母——E 核与 P 核速率不同，
/// 而本机（M5 Pro）6 个 P 核的 6 × 110 已经小于 GEMM 实测的 880 GFLOP/s，说明那个分母连
/// 现象都解释不了。所以直接测整机：线程一次起好、`Barrier` 同步，计时只覆盖屏障之后的
/// 纯计算段（`Instant` 在 `wait` 之后取），线程创建与屏障等待都不进分母。
fn machine_peak(c: &mut Criterion) {
    let count = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let mut group = c.benchmark_group("machine_peak");
    group.throughput(Throughput::Elements(
        (count * 2 * LANES * 4 * ROUNDS) as u64,
    ));
    group.bench_function(format!("f32_{count}"), |iter| {
        iter.iter_custom(|iters| {
            let rounds = iters as usize * ROUNDS;
            let start = Arc::new(Barrier::new(count));
            let mut peak = Duration::ZERO;
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..count)
                    .map(|_| {
                        let start = Arc::clone(&start);
                        scope.spawn(move || {
                            start.wait();
                            let launched = Instant::now();
                            let sum = fma_run::<4>(&mut [[0.0f32; LANES]; 4], rounds);
                            let spent = launched.elapsed();
                            black_box(sum);
                            spent
                        })
                    })
                    .collect();
                for handle in handles {
                    if let Ok(spent) = handle.join() {
                        peak = peak.max(spent);
                    }
                }
            });
            peak
        });
    });
    group.finish();
}

/// `rounds × C × LANES` 次乘加，结果求和返回以避免整段被消除。
fn fma_run<const C: usize>(lanes: &mut [[f32; LANES]; C], rounds: usize) -> f32 {
    for _ in 0..rounds {
        for chain in lanes.iter_mut() {
            for (value, factor) in chain.iter_mut().zip(&STEP) {
                *value = value.mul_add(*factor, *factor);
            }
        }
    }
    lanes.iter().flatten().sum()
}

/// SME 端口峰值：每轮 4 条 `fmopa`、零 load/store，即这块单元每秒能吃多少外积。
///
/// **为什么需要它**：1024³ 走的是 [`sme`] 通道，实测 2300+ GFLOP/s 已经超过 NEON FMA 峰值
/// （1224.8）本身——用 NEON 峰值当分母是口径混用（F2），得 194% 这种无意义的百分比。
/// 分母必须是同一条指令路径的峰值。
///
/// 单线程与整机各测一次：单线程量单元的发射上限，整机量**共享之后**的实际上限。实测
/// 1024³ 从 1 线程到 16 线程只涨 1.55 倍，说明 SME 单元是簇内共享而不是每核一份，
/// 所以只报单线程峰值会把整机 GEMM 的百分比算成假的低值。
fn sme_peak(c: &mut Criterion) {
    ported(c, "", fast::port_run);
    let (a, b) = ([1.0f32; 16], [1.0f32; 64]);
    ported(c, "_mix", |rounds| fast::port_mix(rounds, &a, &b));
}

/// 把端口探针跑成与 [`machine_peak`] 同口径的曲线：`Barrier` 齐步、只计计算段、取各线程最大值
/// （线程创建落在计时区外，否则会把 `spawn` 的微秒算成内核的吞吐）。
///
/// 两条曲线同一份口径：纯 `fmopa` 与「`fmopa` + 每轮 5 条 `ld1w`」。差额就是取数开销——
/// 若差额为 0，说明取数端口完全被 `fmopa` 流水掩住，那么「GEMM 离端口峰值还差多少」就与
/// 瓦片形状无关，不能拿载入条数去解释（A3：先给可核对的数，再谈结论）。
fn ported<F>(c: &mut Criterion, name: &str, probe: F)
where
    F: Fn(usize) -> bool + Sync,
{
    let count = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let probe = &probe;
    for threads in [1, 16, count] {
        let mut group = c.benchmark_group(format!("sme_peak/{threads}{name}"));
        group.throughput(Throughput::Elements(
            (threads * FLOPS_PER_ROUND * ROUNDS) as u64,
        ));
        group.bench_function("port", |iter| {
            iter.iter_custom(|iters| fastest(threads, iters as usize * ROUNDS, probe));
        });
        group.finish();
    }
}

/// 一轮计时：`threads` 个线程在 `Barrier` 上齐步，各自量自己那段计算，取最大值。
///
/// `spawn` 与 `wait` 都在计时区外——否则线程创建的微秒会被算成内核吞吐。`probe` 返回 `false`
/// 表示本机没有这个单元，直接让断言失败，而不是把「没测到」报成一个峰值。
fn fastest(threads: usize, rounds: usize, probe: &(impl Fn(usize) -> bool + Sync)) -> Duration {
    let start = Arc::new(Barrier::new(threads));
    let mut peak = Duration::ZERO;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let start = Arc::clone(&start);
                scope.spawn(move || {
                    start.wait();
                    let launched = Instant::now();
                    assert!(probe(rounds), "本机无 SME");
                    launched.elapsed()
                })
            })
            .collect();
        for handle in handles {
            if let Ok(spent) = handle.join() {
                peak = peak.max(spent);
            }
        }
    });
    peak
}

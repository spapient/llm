//! Criterion microbenchmarks for the native CUDA backend.
//!
//! Linux only: on other targets the `musa` crate is an empty library, so the
//! target compiles to a no-op `main`. The CPU executor measures every workload
//! in the same run, so each CUDA number is reported against a live baseline
//! rather than a stale constant. Both paths are driven through the same
//! `api::compute` contract with identical `bench` operands, so the comparison
//! is reproducible.
//!
//! No improvement is claimed: these numbers are a baseline (E1, RTX 2080 Ti /
//! CUDA 13.0 / `rustc 1.99.0`, 2026-10-12; docs/design.md §21.10.3), not a
//! comparison against an earlier revision. Every call's cost still includes the
//! per-call device context creation and run-time NVRTC compilation this stage
//! performs, which §21.10.3 measures at ≈12.3 ms per NVRTC-served call (about
//! 10.8 ms of it `compile_ptx` plus module load) on that host.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

#[cfg(all(target_os = "linux", musa_toolkit))]
mod linux {
    use std::hint::black_box;

    use ::tensor::{shape, tensor};
    use api::{compute, operand, submission};
    use bench::{
        ELEMENTWISE, EMBEDDIM, GATHER, HEADDIM, HEADS, MATMULDIM, NORMCOLS, NORMROWS, SEQLEN,
        SOFTMAXCOLS, SOFTMAXROWS, attentioninput, elementwiseinput, embeddinginput,
        matmulbatchinput, matmulinput, norminput, ropeinput, scaninput, softmaxinput,
    };
    use context::{executionconfig, executioncontext};
    use cpu::cpuexecutor;
    use criterion::{
        BatchSize, BenchmarkGroup, Criterion, Throughput, criterion_group, measurement::WallTime,
    };
    use gpu::musa::musaexecutor;
    use infer::{
        collectingsink, generationengine, generationparams, generationrequest, samplingparams,
        stopcondition, submissionengine,
    };
    use lib::{limits, requestid, seed};
    use models::{causallm, llamaconfig, llamamodel, llamaweights};

    /// Resource budget that admits every workload without rejecting it.
    const LIMIT: limits = limits::new(1 << 28, 1 << 31, 1 << 20, 1 << 18);

    /// Epsilon used by both sides of the RMS-normalization comparison.
    const EPS: f32 = 1.0e-5;

    /// Rotary base used by both sides of the RoPE comparison.
    const BASE: f32 = 10_000.0;

    /// Vocabulary size of the decode benchmark model.
    const DECODEVOCAB: usize = 128;
    /// Model dimension of the decode benchmark model.
    const DECODEDIM: usize = 64;
    /// Layer count of the decode benchmark model.
    const DECODELAYERS: usize = 2;
    /// Query head count of the decode benchmark model.
    const DECODEHEADS: usize = 2;
    /// Key/value head count of the decode benchmark model.
    const DECODEKVHEADS: usize = 2;
    /// Feed-forward hidden dimension of the decode benchmark model.
    const DECODEHIDDEN: usize = 128;
    /// Maximum sequence length of the decode benchmark model.
    const DECODEMAXSEQ: usize = 16;
    /// Prompt used by the decode benchmark.
    const DECODEPROMPT: [usize; 4] = [0, 1, 2, 3];
    /// Generated tokens per decode benchmark iteration.
    const DECODENEW: usize = 4;

    /// Builds the deterministic decode benchmark model.
    fn decodemodel() -> llamamodel {
        let config = llamaconfig::new(
            DECODEVOCAB,
            DECODEDIM,
            DECODELAYERS,
            DECODEHEADS,
            DECODEKVHEADS,
            DECODEHIDDEN,
            1.0e-5,
            10_000.0,
            DECODEMAXSEQ,
        )
        .unwrap();
        let weights = llamaweights::random(&config, seed::new(42)).unwrap();
        llamamodel::new(config, weights).unwrap()
    }

    /// Measures the `cpu` and `musa` closures of one workload back to back.
    ///
    /// Both are timed in the same run so that a change in the environment moves
    /// both numbers together and the ratio stays meaningful (VF4). Each timed
    /// region returns the operator's output tensor, which `black_box` keeps
    /// observable so the work cannot be optimized away.
    fn pair<F, G>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        name: &str,
        elements: usize,
        cpu: F,
        musa: G,
    ) where
        F: Fn() -> tensor<f32>,
        G: Fn() -> tensor<f32>,
    {
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_function(format!("cpu{name}"), |b| b.iter(|| black_box(cpu())));
        group.bench_function(format!("musa{name}"), |b| b.iter(|| black_box(musa())));
    }

    fn matmul(c: &mut Criterion) {
        let (lhs, rhs) = matmulinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("musa/matmul");
        pair(
            &mut group,
            "256",
            MATMULDIM * MATMULDIM,
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || musaexec.matmul(&lv, &rv).unwrap(),
        );
        group.finish();
    }

    fn matmul_batch(c: &mut Criterion) {
        let (lhs, rhs) = matmulbatchinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("musa/matmulbatch");
        pair(
            &mut group,
            "8x256",
            lhs.getlen(),
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || musaexec.matmul(&lv, &rv).unwrap(),
        );
        group.finish();
    }

    /// A/B of the two matmul paths on the same operands and run: the 256 case
    /// matches the rest of the suite, and the 1024 case keeps the comparison
    /// clear of the per-call launch floor.
    fn matmul_paths(c: &mut Criterion) {
        let fma = musaexecutor::newfma(LIMIT);
        let wmma = musaexecutor::new(LIMIT);
        let (lhs, rhs) = matmulinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let mut group = c.benchmark_group("musa/matmulpath");
        group.throughput(Throughput::Elements((MATMULDIM * MATMULDIM) as u64));
        group.bench_function("fma256", |b| {
            b.iter(|| black_box(fma.matmul(&lv, &rv).unwrap()))
        });
        group.bench_function("wmma256", |b| {
            b.iter(|| black_box(wmma.matmul(&lv, &rv).unwrap()))
        });

        let side = 1024;
        let elements = side * side;
        let lhsbig = tensor::new(
            shape::new(vec![side, side]).unwrap(),
            vec![0.5_f32; elements],
        )
        .unwrap();
        let rhsbig = tensor::new(
            shape::new(vec![side, side]).unwrap(),
            vec![0.25_f32; elements],
        )
        .unwrap();
        let (lvbig, rvbig) = (lhsbig.getview(), rhsbig.getview());
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_function("fma1024", |b| {
            b.iter(|| black_box(fma.matmul(&lvbig, &rvbig).unwrap()))
        });
        group.bench_function("wmma1024", |b| {
            b.iter(|| black_box(wmma.matmul(&lvbig, &rvbig).unwrap()))
        });
        group.finish();
    }

    /// Device-resident matmul chain: eight 1024x1024 contractions per
    /// iteration, with both operands uploaded once and every intermediate
    /// result kept in a device slot. Only the final `apply` synchronises and
    /// downloads, so the measurement is the GPU path plus launch, without the
    /// per-call host round trip the compute API pays.
    fn matmul_chain(c: &mut Criterion) {
        const SIDE: usize = 1024;
        const STEPS: usize = 8;
        let elements = SIDE * SIDE;
        let lhs = tensor::new(
            shape::new(vec![SIDE, SIDE]).unwrap(),
            vec![0.5_f32; elements],
        )
        .unwrap();
        let rhs = tensor::new(
            shape::new(vec![SIDE, SIDE]).unwrap(),
            vec![0.25_f32; elements],
        )
        .unwrap();
        let lv = lhs.getview();
        let rv = rhs.getview();
        let wmma = musaexecutor::new(LIMIT);
        let fma = musaexecutor::newfma(LIMIT);
        let run = |exec: &musaexecutor| {
            let mut sub = exec.newsubmission();
            let mut step = sub
                .recordmatmul(operand::input(lv.clone()), operand::input(rv.clone()))
                .unwrap();
            for _ in 1..STEPS {
                step = sub
                    .recordmatmul(operand::result(step), operand::input(rv.clone()))
                    .unwrap();
            }
            black_box(sub.apply(&[step]).unwrap())
        };
        let mut group = c.benchmark_group("musa/matmulchain");
        group.throughput(Throughput::Elements((elements * STEPS) as u64));
        group.bench_function("wmma1024x8", |b| b.iter(|| run(&wmma)));
        group.bench_function("fma1024x8", |b| b.iter(|| run(&fma)));
        group.finish();
    }

    /// Compares the compute engine with the all-submission engine on a
    /// deterministic tiny decode workload. Both engines prefill and decode the
    /// same prompt; the submission engine records every token into a
    /// device-resident submission and downloads only logits plus the new K/V.
    fn decode(c: &mut Criterion) {
        let exec = musaexecutor::new(LIMIT);
        let model = decodemodel();
        let stop = stopcondition::new(vec![], DECODENEW).unwrap();
        let params = generationparams::formodel(model.getconfig(), stop, samplingparams::greedy());
        let mut group = c.benchmark_group("musa/decode");
        group.throughput(Throughput::Elements(DECODENEW as u64));
        group.bench_function("compute", |b| {
            b.iter_batched(
                || {
                    (
                        generationrequest::new(DECODEPROMPT.to_vec(), params.clone()).unwrap(),
                        executioncontext::new(
                            requestid::new(1),
                            executionconfig::new(1 << 20, 1, 16).unwrap(),
                        ),
                        collectingsink::new(),
                    )
                },
                |(mut request, context, mut sink)| {
                    let engine = generationengine::new(&model, &exec);
                    black_box(
                        engine
                            .generate(&context, &mut request, &mut sink)
                            .unwrap()
                            .getcount(),
                    )
                },
                BatchSize::SmallInput,
            )
        });
        group.bench_function("submission", |b| {
            b.iter_batched(
                || {
                    (
                        generationrequest::new(DECODEPROMPT.to_vec(), params.clone()).unwrap(),
                        executioncontext::new(
                            requestid::new(1),
                            executionconfig::new(1 << 20, 1, 16).unwrap(),
                        ),
                        collectingsink::new(),
                    )
                },
                |(mut request, context, mut sink)| {
                    let engine = submissionengine::new(&model, || exec.newpersistent());
                    black_box(
                        engine
                            .generate(&context, &mut request, &mut sink)
                            .unwrap()
                            .getcount(),
                    )
                },
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }

    fn elementwise(c: &mut Criterion) {
        let x = elementwiseinput().unwrap();
        let y = elementwiseinput().unwrap();
        let v = x.getview();
        let w = y.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("musa/elementwise");
        pair(
            &mut group,
            "silu",
            ELEMENTWISE,
            || cpuexec.silu(&v).unwrap(),
            || musaexec.silu(&v).unwrap(),
        );
        pair(
            &mut group,
            "add",
            ELEMENTWISE,
            || cpuexec.add(&v, &w).unwrap(),
            || musaexec.add(&v, &w).unwrap(),
        );
        pair(
            &mut group,
            "mul",
            ELEMENTWISE,
            || cpuexec.mul(&v, &w).unwrap(),
            || musaexec.mul(&v, &w).unwrap(),
        );
        group.finish();
    }

    fn softmax(c: &mut Criterion) {
        let x = softmaxinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("musa/softmax");
        pair(
            &mut group,
            "rows512cols4096",
            SOFTMAXROWS * SOFTMAXCOLS,
            || cpuexec.softmax(&v).unwrap(),
            || musaexec.softmax(&v).unwrap(),
        );
        group.finish();
    }

    fn rmsnorm(c: &mut Criterion) {
        let (x, w) = norminput().unwrap();
        let (xv, wv) = (x.getview(), w.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("musa/rmsnorm");
        pair(
            &mut group,
            "rows512cols4096",
            NORMROWS * NORMCOLS,
            || cpuexec.rmsnorm(&xv, &wv, EPS).unwrap(),
            || musaexec.rmsnorm(&xv, &wv, EPS).unwrap(),
        );
        group.finish();
    }

    fn rope(c: &mut Criterion) {
        let x = ropeinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("musa/rope");
        pair(
            &mut group,
            "seq128heads8dim64",
            SEQLEN * HEADS * HEADDIM,
            || cpuexec.rope(&v, HEADS, BASE).unwrap(),
            || musaexec.rope(&v, HEADS, BASE).unwrap(),
        );
        group.finish();
    }

    fn embedding(c: &mut Criterion) {
        let (table, indices) = embeddinginput().unwrap();
        let v = table.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("musa/embedding");
        pair(
            &mut group,
            "gather128",
            GATHER * EMBEDDIM,
            || cpuexec.embedding(&v, &indices).unwrap(),
            || musaexec.embedding(&v, &indices).unwrap(),
        );
        group.finish();
    }

    fn attention(c: &mut Criterion) {
        let (q, k, v) = attentioninput().unwrap();
        let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("musa/attention");
        pair(
            &mut group,
            "seq128heads8dim64",
            SEQLEN * HEADS * HEADDIM,
            || {
                cpuexec
                    .attention(&qv, &kv, &vv, HEADS, HEADS, true)
                    .unwrap()
            },
            || {
                musaexec
                    .attention(&qv, &kv, &vv, HEADS, HEADS, true)
                    .unwrap()
            },
        );
        group.finish();
    }

    fn scan(criterion: &mut Criterion) {
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = criterion.benchmark_group("musa/scan");
        for seq in [256usize, 16384] {
            let (input, init, a, b, c, skip) = scaninput(seq).unwrap();
            let (iv, nv, av, bv, cv, sv) = (
                input.getview(),
                init.getview(),
                a.getview(),
                b.getview(),
                c.getview(),
                skip.getview(),
            );
            // seq 256 stays below the single-chunk threshold (serial kernel);
            // seq 16384 crosses it, so the three-pass parallel kernel with the
            // device-resident intermediates is what runs.
            pair(
                &mut group,
                &format!("n{seq}d64state4"),
                input.getlen(),
                || cpuexec.scan(&iv, &nv, &av, &bv, &cv, &sv).unwrap().0,
                || musaexec.scan(&iv, &nv, &av, &bv, &cv, &sv).unwrap().0,
            );
        }
        group.finish();
    }

    /// A 32-step elementwise chain recorded through the accumulative
    /// submission (one synchronise at `apply`) against the same chain run as
    /// 32 separate compute calls (one synchronise each), on both backends.
    fn submissionchain(criterion: &mut Criterion) {
        let steps = 32usize;
        let elements = 64 * 64;
        let (x, y) = (
            tensor::new(shape::new(vec![64, 64]).unwrap(), vec![0.5_f32; elements]).unwrap(),
            tensor::new(shape::new(vec![64, 64]).unwrap(), vec![0.25_f32; elements]).unwrap(),
        );
        let (xv, yv) = (x.getview(), y.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let musaexec = musaexecutor::new(LIMIT);
        let mut group = criterion.benchmark_group("musa/submissionchain");
        group.throughput(Throughput::Elements((elements * steps) as u64));
        group.bench_function("cpucompute32", |b| {
            b.iter(|| {
                let mut acc = x.clone();
                for _ in 0..steps {
                    acc = cpuexec.add(&acc.getview(), &yv).unwrap();
                }
                black_box(acc)
            })
        });
        group.bench_function("musacompute32", |b| {
            b.iter(|| {
                let mut acc = x.clone();
                for _ in 0..steps {
                    acc = musaexec.add(&acc.getview(), &yv).unwrap();
                }
                black_box(acc)
            })
        });
        group.bench_function("musasubmission32", |b| {
            b.iter(|| {
                let mut sub = musaexec.newsubmission();
                let mut current = operand::input(xv.clone());
                let mut last = None;
                for _ in 0..steps {
                    last = Some(sub.recordadd(current, operand::input(yv.clone())).unwrap());
                    current = operand::result(last.unwrap());
                }
                black_box(sub.apply(&[last.unwrap()]).unwrap())
            })
        });
        group.finish();
    }

    criterion_group!(
        benches,
        matmul,
        matmul_batch,
        matmul_paths,
        matmul_chain,
        decode,
        elementwise,
        softmax,
        rmsnorm,
        rope,
        embedding,
        attention,
        scan,
        submissionchain
    );
}

#[cfg(all(target_os = "linux", musa_toolkit))]
criterion::criterion_main!(linux::benches);

#[cfg(not(all(target_os = "linux", musa_toolkit)))]
fn main() {}

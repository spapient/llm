//! Criterion microbenchmarks for the native Metal backend.
//!
//! macOS only: on other targets the `metal` crate is an empty library, so the
//! target compiles to a no-op `main`. The CPU executor measures every workload
//! in the same run, so each Metal number is reported against a live baseline
//! rather than a stale constant. Both paths are driven through the same
//! `api::compute` contract with identical `bench` operands, so the comparison is
//! reproducible.
//!
//! The workload set and the group names mirror `benches/cuda.rs`: the two
//! backends share semantics, not implementation, so their measurements are
//! comparable one workload at a time while each keeps its own kernel and
//! dispatch geometry.
//!
//! No improvement is claimed: every call's cost still includes the per-call
//! device context creation and run-time MSL compilation this stage performs
//! (docs/design.md §21.10 records that as the reason Metal is slower than CPU at
//! the one size measured so far). Establishing the baseline is required by
//! docs/engineering.md; replacing the per-call compilation is a listed review
//! condition (docs/design.md §21.11).
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

#[cfg(target_os = "macos")]
mod macos {
    use std::hint::black_box;

    use ::tensor::{shape, tensor};
    use api::{compute, operand, submission};
    use bench::{
        ELEMENTWISE, EMBEDDIM, GATHER, HEADDIM, HEADS, MATMULDIM, NORMCOLS, NORMROWS, SEQLEN,
        SOFTMAXCOLS, SOFTMAXROWS, attentioninput, elementwiseinput, embeddinginput,
        matmulbatchinput, matmulinput, norminput, ropeinput, scaninput, softmaxinput,
    };
    use cpu::cpuexecutor;
    use criterion::{
        BenchmarkGroup, Criterion, Throughput, criterion_group, measurement::WallTime,
    };
    use infer::kvstate;
    use lib::{limits, seed};
    use metal::metalexecutor;
    use models::{causallm, kvcache, llamaconfig, llamamodel, llamaweights};

    /// Resource budget that admits every workload without rejecting it.
    const LIMIT: limits = limits::new(1 << 28, 1 << 31, 1 << 20, 1 << 18);

    /// Epsilon used by both sides of the RMS-normalization comparison.
    const EPS: f32 = 1.0e-5;

    /// Rotary base used by both sides of the RoPE comparison.
    const BASE: f32 = 10_000.0;

    /// Vocabulary size of the forwarded model.
    const VOCAB: usize = 256;
    /// Model dimension.
    const DIM: usize = 64;
    /// Transformer layer count.
    const LAYERS: usize = 2;
    /// Query head count.
    const NHEADS: usize = 4;
    /// Key/value head count.
    const KVHEADS: usize = 2;
    /// Feed-forward hidden dimension.
    const HIDDEN: usize = 170;
    /// Sequence length of the forwarded model.
    const SEQ: usize = 16;

    /// Measures the `cpu` and `metal` closures of one workload back to back.
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
        metal: G,
    ) where
        F: Fn() -> tensor<f32>,
        G: Fn() -> tensor<f32>,
    {
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_function(format!("cpu{name}"), |b| b.iter(|| black_box(cpu())));
        group.bench_function(format!("metal{name}"), |b| b.iter(|| black_box(metal())));
    }

    fn matmul(c: &mut Criterion) {
        let (lhs, rhs) = matmulinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/matmul");
        pair(
            &mut group,
            "256",
            MATMULDIM * MATMULDIM,
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || metalexec.matmul(&lv, &rv).unwrap(),
        );
        group.finish();
    }

    fn matmul_batch(c: &mut Criterion) {
        let (lhs, rhs) = matmulbatchinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/matmulbatch");
        pair(
            &mut group,
            "8x256",
            lhs.getlen(),
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || metalexec.matmul(&lv, &rv).unwrap(),
        );
        group.finish();
    }

    fn elementwise(c: &mut Criterion) {
        let x = elementwiseinput().unwrap();
        let y = elementwiseinput().unwrap();
        let v = x.getview();
        let w = y.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/elementwise");
        pair(
            &mut group,
            "silu",
            ELEMENTWISE,
            || cpuexec.silu(&v).unwrap(),
            || metalexec.silu(&v).unwrap(),
        );
        pair(
            &mut group,
            "add",
            ELEMENTWISE,
            || cpuexec.add(&v, &w).unwrap(),
            || metalexec.add(&v, &w).unwrap(),
        );
        pair(
            &mut group,
            "mul",
            ELEMENTWISE,
            || cpuexec.mul(&v, &w).unwrap(),
            || metalexec.mul(&v, &w).unwrap(),
        );
        group.finish();
    }

    fn softmax(c: &mut Criterion) {
        let x = softmaxinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/softmax");
        pair(
            &mut group,
            "rows512cols4096",
            SOFTMAXROWS * SOFTMAXCOLS,
            || cpuexec.softmax(&v).unwrap(),
            || metalexec.softmax(&v).unwrap(),
        );
        group.finish();
    }

    fn rmsnorm(c: &mut Criterion) {
        let (x, w) = norminput().unwrap();
        let (xv, wv) = (x.getview(), w.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/rmsnorm");
        pair(
            &mut group,
            "rows512cols4096",
            NORMROWS * NORMCOLS,
            || cpuexec.rmsnorm(&xv, &wv, EPS).unwrap(),
            || metalexec.rmsnorm(&xv, &wv, EPS).unwrap(),
        );
        group.finish();
    }

    fn rope(c: &mut Criterion) {
        let x = ropeinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/rope");
        pair(
            &mut group,
            "seq128heads8dim64",
            SEQLEN * HEADS * HEADDIM,
            || cpuexec.rope(&v, HEADS, BASE).unwrap(),
            || metalexec.rope(&v, HEADS, BASE).unwrap(),
        );
        group.finish();
    }

    fn embedding(c: &mut Criterion) {
        let (table, indices) = embeddinginput().unwrap();
        let v = table.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/embedding");
        pair(
            &mut group,
            "gather128",
            GATHER * EMBEDDIM,
            || cpuexec.embedding(&v, &indices).unwrap(),
            || metalexec.embedding(&v, &indices).unwrap(),
        );
        group.finish();
    }

    fn attention(c: &mut Criterion) {
        let (q, k, v) = attentioninput().unwrap();
        let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/attention");
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
                metalexec
                    .attention(&qv, &kv, &vv, HEADS, HEADS, true)
                    .unwrap()
            },
        );
        group.finish();
    }

    /// End-to-end forward pass of the same tiny Llama `benches/models.rs`
    /// measures, run on both providers in one process.
    ///
    /// This is the workload the backend exists for, and the one where the
    /// per-call cost of a synchronous submission dominates: a forward pass is a
    /// *chain* of operator calls, each of which pays the command-buffer round
    /// trip measured in docs/design.md §21.10.2. The ratio this prints is the
    /// tax that chain pays, and the number any batching execution context must
    /// beat.
    fn model_forward(c: &mut Criterion) {
        let config = llamaconfig::new(
            VOCAB, DIM, LAYERS, NHEADS, KVHEADS, HIDDEN, 1.0e-5, 10_000.0, SEQ,
        )
        .unwrap();
        let weights = llamaweights::random(&config, seed::new(42)).unwrap();
        let model = llamamodel::new(config, weights).unwrap();
        let tokens: Vec<usize> = (0..SEQ).map(|i| i % VOCAB).collect();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);

        let mut group = c.benchmark_group("gpu/modelforward");
        group.throughput(Throughput::Elements(SEQ as u64));
        group.bench_function("cpu", |b| {
            b.iter(|| black_box(model.forward(&cpuexec, black_box(&tokens)).unwrap()));
        });
        group.bench_function("metal", |b| {
            b.iter(|| black_box(model.forward(&metalexec, black_box(&tokens)).unwrap()));
        });
        group.bench_function("metalsubmission", |b| {
            b.iter(|| {
                let mut sub = metalexec.newsubmission().unwrap();
                let logits = model
                    .forwardsubmission(&mut sub, black_box(&tokens))
                    .unwrap();
                black_box(sub.apply(&[logits]).unwrap())
            });
        });
        group.finish();
    }

    /// A 32-step elementwise chain run through `compute` (32 synchronous
    /// submissions) and through `submission` (one submission), isolating the
    /// per-call command-buffer round trip that dominates a GPU forward pass.
    ///
    /// This is a representative chain, not the exact Llama forward: it proves
    /// the one lever `submission` exists to pull — collapsing N round trips into
    /// one — without re-deriving the model's operator sequence here.
    fn submission_chain(c: &mut Criterion) {
        let metalexec = metalexecutor::new(LIMIT);
        let base = tensor::new(shape::new(vec![DIM, DIM]).unwrap(), vec![0.5; DIM * DIM]).unwrap();
        let delta =
            tensor::new(shape::new(vec![DIM, DIM]).unwrap(), vec![0.25; DIM * DIM]).unwrap();

        let mut group = c.benchmark_group("gpu/submissionchain");
        group.throughput(Throughput::Elements((DIM * DIM) as u64));
        group.bench_function("compute32", |bench| {
            bench.iter(|| {
                let mut x = base.clone();
                for _ in 0..32 {
                    x = metalexec.add(&x.getview(), &delta.getview()).unwrap();
                }
                black_box(x)
            });
        });
        group.bench_function("submission32", |bench| {
            bench.iter(|| {
                let mut sub = metalexec.newsubmission().unwrap();
                let mut prev = sub
                    .recordadd(
                        operand::input(base.getview()),
                        operand::input(delta.getview()),
                    )
                    .unwrap();
                for _ in 1..32 {
                    prev = sub
                        .recordadd(operand::result(prev), operand::input(delta.getview()))
                        .unwrap();
                }
                black_box(sub.apply(&[prev]).unwrap())
            });
        });
        group.finish();
    }

    /// KV-cache decode: prefill the prompt token by token, then decode the rest
    /// token by token. The cache-aware path processes only the new token each
    /// step, while the recompute path runs a full forward over the whole context.
    fn decode(c: &mut Criterion) {
        let config = llamaconfig::new(
            VOCAB,
            DIM,
            LAYERS,
            NHEADS,
            KVHEADS,
            HIDDEN,
            1.0e-5,
            10_000.0,
            SEQ * 2,
        )
        .unwrap();
        let weights = llamaweights::random(&config, seed::new(42)).unwrap();
        let model = llamamodel::new(config, weights).unwrap();
        let prompt: Vec<usize> = (0..SEQ).map(|i| i % VOCAB).collect();
        let decode_tokens: Vec<usize> = (SEQ..2 * SEQ).map(|i| i % VOCAB).collect();
        let metalexec = metalexecutor::new(LIMIT);

        let mut group = c.benchmark_group("gpu/decode");
        group.throughput(Throughput::Elements(SEQ as u64));
        // Both paths time only the decode phase (iter_batched setup is untimed):
        // `metalcached` prefills in setup, `metalrecompute` starts from the prompt.
        // Symmetric token counts make the numbers honest; at seq=16 the per-call
        // command-buffer round trip dominates, so no speedup is claimed here.
        group.bench_function("metalcached", |bench| {
            bench.iter_batched(
                || {
                    let mut cache =
                        kvstate::new(config.getnlayers(), config.getkvwidth(), config.getmaxseq())
                            .unwrap();
                    for &token in &prompt {
                        model.forward_cached(&metalexec, &mut cache, token).unwrap();
                    }
                    cache
                },
                |mut cache| {
                    let layers = config.getnlayers();
                    for &token in &decode_tokens {
                        let mut sub = metalexec.newsubmission().unwrap();
                        let handles = model
                            .forwardsubmission_cached(&mut sub, &cache, token)
                            .unwrap();
                        let mut outputs = Vec::with_capacity(1 + 2 * layers);
                        outputs.push(handles.logits);
                        outputs.extend(handles.keys.iter().copied());
                        outputs.extend(handles.values.iter().copied());
                        let got = sub.apply(&outputs).unwrap();
                        for layer in 0..layers {
                            cache
                                .append(layer, &got[1 + layer], &got[1 + layers + layer])
                                .unwrap();
                        }
                        black_box(&got[0]);
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
        group.bench_function("metalrecompute", |bench| {
            bench.iter_batched(
                || prompt.clone(),
                |mut context| {
                    for &token in &decode_tokens {
                        context.push(token);
                        let mut sub = metalexec.newsubmission().unwrap();
                        let logits = model.forwardsubmission(&mut sub, &context).unwrap();
                        black_box(sub.apply(&[logits]).unwrap());
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
        group.finish();
    }

    /// M1 Should: on the same device, a 16K semigroup scan against dense 16K
    /// attention on the same sequence (lld.md §5.6, §10.10 — no cross-model
    /// Table 7 extrapolation). The scan is O(seq·d), dense attention
    /// O(seq²·d), so the Metal ratio is the β-path payoff of the three-stage
    /// parallel scan; the CPU serial number anchors the same workload.
    fn scan(c: &mut Criterion) {
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let seq = 16384;
        let (input, init, a, b, cparam, skip) = scaninput(seq).unwrap();
        let (iv, statev, av, bv, cv, sv) = (
            input.getview(),
            init.getview(),
            a.getview(),
            b.getview(),
            cparam.getview(),
            skip.getview(),
        );
        let mut group = c.benchmark_group("gpu/scan");
        group.throughput(Throughput::Elements((seq * bench::SCANCHANNELS) as u64));
        group.bench_function("cpuserial", |b| {
            b.iter(|| {
                black_box(
                    compute::scan(
                        &cpuexec,
                        black_box(&iv),
                        black_box(&statev),
                        black_box(&av),
                        black_box(&bv),
                        black_box(&cv),
                        black_box(&sv),
                    )
                    .unwrap(),
                )
            })
        });
        group.bench_function("metalparallel", |b| {
            b.iter(|| {
                black_box(
                    compute::scan(
                        &metalexec,
                        black_box(&iv),
                        black_box(&statev),
                        black_box(&av),
                        black_box(&bv),
                        black_box(&cv),
                        black_box(&sv),
                    )
                    .unwrap(),
                )
            })
        });
        group.finish();
    }

    /// The dense 16K attention side of the M1 Should comparison. Metal only:
    /// the same workload on CPU would run for minutes. The `heads8dim64`
    /// suffix pins the workload against the existing `gpu/attention`
    /// `seq128heads8dim64` group.
    fn attention16k(c: &mut Criterion) {
        let metalexec = metalexecutor::new(LIMIT);
        let seq = 16384;
        let width = HEADS * HEADDIM;
        let elements = seq * width;
        let q = tensor::new(shape::new(vec![seq, width]).unwrap(), vec![0.1_f32; elements]).unwrap();
        let k = tensor::new(shape::new(vec![seq, width]).unwrap(), vec![0.2_f32; elements]).unwrap();
        let v = tensor::new(shape::new(vec![seq, width]).unwrap(), vec![0.3_f32; elements]).unwrap();
        let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
        let mut group = c.benchmark_group("gpu/attention16k");
        group.throughput(Throughput::Elements(seq as u64));
        group.bench_function("metalheads8dim64", |b| {
            b.iter(|| {
                black_box(metalexec.attention(&qv, &kv, &vv, HEADS, HEADS, true).unwrap())
            })
        });
        group.finish();
    }

    criterion_group!(
        benches,
        matmul,
        matmul_batch,
        elementwise,
        softmax,
        rmsnorm,
        rope,
        embedding,
        attention,
        scan,
        attention16k,
        model_forward,
        submission_chain,
        decode
    );
}

#[cfg(target_os = "macos")]
criterion::criterion_main!(macos::benches);

#[cfg(not(target_os = "macos"))]
fn main() {}

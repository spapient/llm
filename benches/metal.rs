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

    use ::bench::harness::{report, run};
    use ::tensor::{shape, tensor};
    use api::{compute, operand, submission};
    use bench::{
        HEADDIM, HEADS, attentioninput, elementwiseinput, embeddinginput, matmulbatchinput,
        matmulinput, norminput, ropeinput, scaninput, softmaxinput,
    };
    use cpu::cpuexecutor;
    use gpu::metal::metalexecutor;
    use infer::kvstate;
    use lib::{limits, seed};
    use models::{causallm, devicecache, kvcache, llamaconfig, llamamodel, llamaweights};

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
    fn pair<F, G>(prefix: &str, name: &str, cpu: F, metal: G)
    where
        F: Fn() -> tensor<f32>,
        G: Fn() -> tensor<f32>,
    {
        report(&run(&format!("{prefix}/cpu{name}"), || black_box(cpu())));
        report(&run(
            &format!("{prefix}/metal{name}"),
            || black_box(metal()),
        ));
    }

    pub fn matmul() {
        let (lhs, rhs) = matmulinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        pair(
            "gpu/matmul",
            "256",
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || metalexec.matmul(&lv, &rv).unwrap(),
        );
    }

    pub fn matmul_batch() {
        let (lhs, rhs) = matmulbatchinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        pair(
            "gpu/matmulbatch",
            "8x256",
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || metalexec.matmul(&lv, &rv).unwrap(),
        );
    }

    pub fn elementwise() {
        let x = elementwiseinput().unwrap();
        let y = elementwiseinput().unwrap();
        let v = x.getview();
        let w = y.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        pair(
            "gpu/elementwise",
            "silu",
            || cpuexec.silu(&v).unwrap(),
            || metalexec.silu(&v).unwrap(),
        );
        pair(
            "gpu/elementwise",
            "add",
            || cpuexec.add(&v, &w).unwrap(),
            || metalexec.add(&v, &w).unwrap(),
        );
        pair(
            "gpu/elementwise",
            "mul",
            || cpuexec.mul(&v, &w).unwrap(),
            || metalexec.mul(&v, &w).unwrap(),
        );
    }

    pub fn softmax() {
        let x = softmaxinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        pair(
            "gpu/softmax",
            "rows512cols4096",
            || cpuexec.softmax(&v).unwrap(),
            || metalexec.softmax(&v).unwrap(),
        );
    }

    pub fn rmsnorm() {
        let (x, w) = norminput().unwrap();
        let (xv, wv) = (x.getview(), w.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        pair(
            "gpu/rmsnorm",
            "rows512cols4096",
            || cpuexec.rmsnorm(&xv, &wv, EPS).unwrap(),
            || metalexec.rmsnorm(&xv, &wv, EPS).unwrap(),
        );
    }

    pub fn rope() {
        let x = ropeinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        pair(
            "gpu/rope",
            "seq128heads8dim64",
            || cpuexec.rope(&v, HEADS, BASE).unwrap(),
            || metalexec.rope(&v, HEADS, BASE).unwrap(),
        );
    }

    pub fn embedding() {
        let (table, indices) = embeddinginput().unwrap();
        let v = table.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        pair(
            "gpu/embedding",
            "gather128",
            || cpuexec.embedding(&v, &indices).unwrap(),
            || metalexec.embedding(&v, &indices).unwrap(),
        );
    }

    pub fn attention() {
        let (q, k, v) = attentioninput().unwrap();
        let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        pair(
            "gpu/attention",
            "seq128heads8dim64",
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
    pub fn model_forward() {
        let config = llamaconfig::new(
            VOCAB, DIM, LAYERS, NHEADS, KVHEADS, HIDDEN, 1.0e-5, 10_000.0, SEQ,
        )
        .unwrap();
        let weights = llamaweights::random(&config, seed::new(42)).unwrap();
        let model = llamamodel::new(config, weights).unwrap();
        let tokens: Vec<usize> = (0..SEQ).map(|i| i % VOCAB).collect();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        report(&run("gpu/modelforward/cpu", || {
            black_box(model.forward(&cpuexec, black_box(&tokens)).unwrap())
        }));
        report(&run("gpu/modelforward/metal", || {
            black_box(model.forward(&metalexec, black_box(&tokens)).unwrap())
        }));
        report(&run("gpu/modelforward/metalsubmission", || {
            let mut sub = metalexec.newsubmission().unwrap();
            let logits = model
                .forwardsubmission(&mut sub, black_box(&tokens))
                .unwrap();
            black_box(sub.apply(&[logits]).unwrap())
        }));
    }

    /// A 32-step elementwise chain run through `compute` (32 synchronous
    /// submissions) and through `submission` (one submission), isolating the
    /// per-call command-buffer round trip that dominates a GPU forward pass.
    ///
    /// This is a representative chain, not the exact Llama forward: it proves
    /// the one lever `submission` exists to pull — collapsing N round trips into
    /// one — without re-deriving the model's operator sequence here.
    pub fn submission_chain() {
        let metalexec = metalexecutor::new(LIMIT);
        let base = tensor::new(shape::new(vec![DIM, DIM]).unwrap(), vec![0.5; DIM * DIM]).unwrap();
        let delta =
            tensor::new(shape::new(vec![DIM, DIM]).unwrap(), vec![0.25; DIM * DIM]).unwrap();
        report(&run("gpu/submissionchain/compute32", || {
            let mut x = base.clone();
            for _ in 0..32 {
                x = metalexec.add(&x.getview(), &delta.getview()).unwrap();
            }
            black_box(x)
        }));
        report(&run("gpu/submissionchain/submission32", || {
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
        }));
    }

    /// KV-cache decode at three sequence lengths: prefill the prompt token by
    /// token, then decode the rest token by token. The cache-aware path
    /// processes only the new token each step, while the recompute path runs a
    /// full forward over the whole context.
    ///
    /// Both paths time only the decode phase (setup is untimed): `metalcached`
    /// prefills in setup, `metalrecompute` starts from the prompt. Symmetric
    /// token counts keep the numbers honest. At seq=16 the per-call
    /// command-buffer round trip dominates both paths and no speedup is
    /// claimed; the 128/256 rows are the long-sequence evidence for the
    /// O(seq²) → O(seq) attention saving once a single submission collapses
    /// the round trips — the KV-cache payoff stage (docs/design.md §21.13).
    pub fn decode() {
        for seq in [16usize, 128, 256] {
            decodeat(seq);
        }
    }

    fn decodeat(seq: usize) {
        let config = llamaconfig::new(
            VOCAB,
            DIM,
            LAYERS,
            NHEADS,
            KVHEADS,
            HIDDEN,
            1.0e-5,
            10_000.0,
            seq * 2,
        )
        .unwrap();
        let weights = llamaweights::random(&config, seed::new(42)).unwrap();
        let model = llamamodel::new(config, weights).unwrap();
        let prompt: Vec<usize> = (0..seq).map(|i| i % VOCAB).collect();
        let decode_tokens: Vec<usize> = (seq..2 * seq).map(|i| i % VOCAB).collect();
        let metalexec = metalexecutor::new(LIMIT);
        report(&run(&format!("gpu/decode/metalcached/s{seq}"), || {
            let mut cache =
                kvstate::new(config.getnlayers(), config.getkvwidth(), config.getmaxseq()).unwrap();
            for &token in &prompt {
                model.forward_cached(&metalexec, &mut cache, token).unwrap();
            }
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
        }));
        report(&run(&format!("gpu/decode/devicecached/s{seq}"), || {
            // The same decode workload against a device-resident cache: each
            // step records one `forwardsubmission_cached_device`, appends the
            // new key/value rows with a device-side blit inside `apply`, and
            // downloads only the logits — no KV tensor ever leaves the
            // device. Setup (prefill) and decode both run through the device
            // cache so the token counts match `metalcached` and `recompute`.
            let mut cache =
                devicecache::new(config.getnlayers(), config.getkvwidth(), config.getmaxseq())
                    .unwrap();
            for &token in prompt.iter().chain(decode_tokens.iter()) {
                let mut sub = metalexec.newsubmission().unwrap();
                let handles = model
                    .forwardsubmission_cached_device(&mut sub, &mut cache, token)
                    .unwrap();
                let logits = sub.apply(&[handles.logits]).unwrap();
                black_box(&logits[0]);
            }
        }));
        report(&run(&format!("gpu/decode/metalrecompute/s{seq}"), || {
            let mut context = prompt.clone();
            for &token in &decode_tokens {
                context.push(token);
                let mut sub = metalexec.newsubmission().unwrap();
                let logits = model.forwardsubmission(&mut sub, &context).unwrap();
                black_box(sub.apply(&[logits]).unwrap());
            }
        }));
    }

    /// M1 Should: on the same device, a 16K semigroup scan against dense 16K
    /// attention on the same sequence (lld.md §5.6, §10.10 — no cross-model
    /// Table 7 extrapolation). The scan is O(seq·d), dense attention
    /// O(seq²·d), so the Metal ratio is the β-path payoff of the three-stage
    /// parallel scan; the CPU serial number anchors the same workload.
    pub fn scan() {
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
        report(&run("gpu/scan/cpuserial", || {
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
        }));
        report(&run("gpu/scan/metalparallel", || {
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
        }));
    }

    /// The dense 16K attention side of the M1 Should comparison. Metal only:
    /// the same workload on CPU would run for minutes. The `heads8dim64`
    /// suffix pins the workload against the existing `gpu/attention`
    /// `seq128heads8dim64` group.
    pub fn attention16k() {
        let metalexec = metalexecutor::new(LIMIT);
        let seq = 16384;
        let width = HEADS * HEADDIM;
        let elements = seq * width;
        let q = tensor::new(
            shape::new(vec![seq, width]).unwrap(),
            vec![0.1_f32; elements],
        )
        .unwrap();
        let k = tensor::new(
            shape::new(vec![seq, width]).unwrap(),
            vec![0.2_f32; elements],
        )
        .unwrap();
        let v = tensor::new(
            shape::new(vec![seq, width]).unwrap(),
            vec![0.3_f32; elements],
        )
        .unwrap();
        let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
        report(&run("gpu/attention16k/metalheads8dim64", || {
            black_box(
                metalexec
                    .attention(&qv, &kv, &vv, HEADS, HEADS, true)
                    .unwrap(),
            )
        }));
    }
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "macos")]
fn main() {
    macos::matmul();
    macos::matmul_batch();
    macos::elementwise();
    macos::softmax();
    macos::rmsnorm();
    macos::rope();
    macos::embedding();
    macos::attention();
    macos::scan();
    macos::attention16k();
    macos::model_forward();
    macos::submission_chain();
    macos::decode();
}

#[cfg(not(target_os = "macos"))]
fn main() {}

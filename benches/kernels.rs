//! Criterion microbenchmarks for the raw CPU kernels.
//!
//! These benchmarks call the kernel types directly through the frozen `api`
//! kernel contracts, without the executor's budget checks. The end-to-end
//! executor path is measured separately by the `cpuops` target. Every workload
//! fixes its dimensions and dtype so results are reproducible. These
//! measurements establish the baseline required by `docs/engineering.md`; no
//! improvement is claimed and no allocation is performed inside the timed region
//! beyond the output the kernel contract returns.
//!
//! Shared workload generators live in the `bench` library so that every target
//! measures identical inputs.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::bench::harness::{report, run};
use ::tensor::{shape, tensor};
use api::{binarykernel, gatherkernel, kernel, weightedkernel};
use bench::{HEADS, attentioninput, embeddinginput, matmulinput, norminput, ropeinput};
use cpu::{
    cpuadd, cpuattention, cpuembedding, cpugelutanh, cpumatmul, cpumul, cpurelu, cpurmsnorm,
    cpurope, cpusilu, cpusoftmax,
};

/// Element count for the elementwise activation workloads.
const ELEMENTWISE: usize = 1 << 20;

/// Row count for the softmax workload.
const SOFTMAXROWS: usize = 512;

/// Row width for the softmax workload.
const SOFTMAXCOLS: usize = 4096;

fn elementwise() {
    let t = tensor::new(
        shape::new(vec![ELEMENTWISE]).unwrap(),
        vec![0.5_f32; ELEMENTWISE],
    )
    .unwrap();
    let view = t.getview();

    report(&run("kernel/elementwise/relu", || {
        black_box(cpurelu.apply(black_box(&view)).unwrap())
    }));
    report(&run("kernel/elementwise/silu", || {
        black_box(cpusilu.apply(black_box(&view)).unwrap())
    }));
    report(&run("kernel/elementwise/gelutanh", || {
        black_box(cpugelutanh.apply(black_box(&view)).unwrap())
    }));
}

fn softmax() {
    let elements = SOFTMAXROWS * SOFTMAXCOLS;
    let t = tensor::new(
        shape::new(vec![SOFTMAXROWS, SOFTMAXCOLS]).unwrap(),
        vec![0.25_f32; elements],
    )
    .unwrap();
    let view = t.getview();

    report(&run("kernel/softmax/rows512cols4096", || {
        black_box(cpusoftmax.apply(black_box(&view)).unwrap())
    }));
}

fn binary() {
    let (a, b) = matmulinput().unwrap();
    let (av, bv) = (a.getview(), b.getview());

    report(&run("kernel/binary/matmul256", || {
        black_box(cpumatmul.apply(black_box(&av), black_box(&bv)).unwrap())
    }));
    report(&run("kernel/binary/add", || {
        black_box(cpuadd.apply(black_box(&av), black_box(&bv)).unwrap())
    }));
    report(&run("kernel/binary/mul", || {
        black_box(cpumul.apply(black_box(&av), black_box(&bv)).unwrap())
    }));
}

fn rmsnorm() {
    let (x, w) = norminput().unwrap();
    let (xv, wv) = (x.getview(), w.getview());
    let kernel = cpurmsnorm::new(cpurmsnorm::DEFAULTEPS);

    report(&run("kernel/rmsnorm/rows512cols4096", || {
        black_box(kernel.apply(black_box(&xv), black_box(&wv)).unwrap())
    }));
}

fn rope() {
    let t = ropeinput().unwrap();
    let view = t.getview();
    let kernel = cpurope::new(HEADS, 10000.0);

    report(&run("kernel/rope/seq128heads8dim64", || {
        black_box(kernel.apply(black_box(&view)).unwrap())
    }));
}

fn embedding() {
    let (table, indices) = embeddinginput().unwrap();
    let view = table.getview();

    report(&run("kernel/embedding/gather128", || {
        black_box(
            cpuembedding
                .apply(black_box(&view), black_box(&indices))
                .unwrap(),
        )
    }));
}

fn attention() {
    let (q, k, v) = attentioninput().unwrap();
    let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());

    report(&run("kernel/attention/seq128heads8dim64", || {
        black_box(
            cpuattention
                .apply(
                    black_box(&qv),
                    black_box(&kv),
                    black_box(&vv),
                    HEADS,
                    HEADS,
                    true,
                )
                .unwrap(),
        )
    }));
}

fn main() {
    elementwise();
    softmax();
    binary();
    rmsnorm();
    rope();
    embedding();
    attention();
}

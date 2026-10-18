//! Benchmarks for the CPU kernels.
//!
//! Every workload fixes its dimensions and dtype so results are reproducible.
//! These measurements establish the initial baseline required by
//! `docs/engineering.md`; no improvement is claimed and no allocation is
//! performed inside the timed region beyond the output the kernel contract
//! returns.
//!
//! The workload constructors live in the `bench` library so that this target
//! stays a thin measurement harness.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::bench::harness::{report, run};
use ::tensor::{shape, tensor};
use api::compute;
use bench::{
    ATTNDECODEKV, ATTNPREFILLLONG, ATTNPREFILLSEQ, HEADS, attentioninput, attentionseqinput,
    cpuexec, cpuexecw, elementwiseinput, embeddinginput, matmuldecodeinput, matmulinput,
    matmulprefillinput, norminput, ropeinput, scaninput, softmaxinput,
};
use cpu::{cpuop, cpuscan};

fn elementwise() {
    let exec = cpuexec();
    let t = elementwiseinput().unwrap();
    let view = t.getview();
    for op in [cpuop::relu, cpuop::silu, cpuop::gelutanh] {
        let name = op.getname();
        report(&run(&format!("elementwise/{name}"), || {
            black_box(exec.apply(op, black_box(&view)).unwrap())
        }));
    }
}

fn softmax() {
    let exec = cpuexec();
    let t = softmaxinput().unwrap();
    let view = t.getview();
    report(&run("softmax/rows512cols4096", || {
        black_box(exec.apply(cpuop::softmax, black_box(&view)).unwrap())
    }));
}

fn binary() {
    let exec = cpuexec();
    let (a, b) = matmulinput().unwrap();
    let (av, bv) = (a.getview(), b.getview());
    for op in [cpuop::matmul, cpuop::add, cpuop::mul] {
        let name = op.getname();
        report(&run(&format!("executor/binary/{name}"), || {
            black_box(
                exec.applybinary(op, black_box(&av), black_box(&bv))
                    .unwrap(),
            )
        }));
    }
}

fn weighted() {
    let exec = cpuexec();
    let (x, w) = norminput().unwrap();
    let (xv, wv) = (x.getview(), w.getview());
    report(&run("executor/weighted/rmsnorm", || {
        black_box(
            exec.applyweighted(cpuop::rmsnorm, black_box(&xv), black_box(&wv))
                .unwrap(),
        )
    }));
    for workers in [4usize, 8] {
        let execw = cpuexecw(workers).unwrap();
        report(&run(
            &format!("executor/weighted/rmsnormw{workers}"),
            || {
                black_box(
                    execw
                        .applyweighted(cpuop::rmsnorm, black_box(&xv), black_box(&wv))
                        .unwrap(),
                )
            },
        ));
    }
}

fn gather() {
    let exec = cpuexec();
    let (table, indices) = embeddinginput().unwrap();
    let view = table.getview();
    report(&run("executor/gather/embedding", || {
        black_box(
            exec.applygather(cpuop::embedding, black_box(&view), black_box(&indices))
                .unwrap(),
        )
    }));
}

fn compute_paths() {
    let exec = cpuexec();
    let r = ropeinput().unwrap();
    let rv = r.getview();
    let (q, k, v) = attentioninput().unwrap();
    let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
    report(&run("executor/compute/rope", || {
        black_box(compute::rope(&exec, black_box(&rv), bench::HEADS, 10000.0).unwrap())
    }));
    report(&run("executor/compute/attention", || {
        black_box(
            compute::attention(
                &exec,
                black_box(&qv),
                black_box(&kv),
                black_box(&vv),
                bench::HEADS,
                bench::HEADS,
                true,
            )
            .unwrap(),
        )
    }));
}

fn scan() {
    let exec = cpuexec();
    for seq in [4096usize, 16384, 65536] {
        let (input, init, a, b, cparam, skip) = scaninput(seq).unwrap();
        let (iv, statev, av, bv, cv, sv) = (
            input.getview(),
            init.getview(),
            a.getview(),
            b.getview(),
            cparam.getview(),
            skip.getview(),
        );
        report(&run(&format!("scan/seq{seq}/serial"), || {
            black_box(
                compute::scan(
                    &exec,
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
        report(&run(&format!("scan/seq{seq}/chunked"), || {
            black_box(
                cpuscan
                    .apply_chunked(
                        black_box(&iv),
                        black_box(&statev),
                        black_box(&av),
                        black_box(&bv),
                        black_box(&cv),
                        black_box(&sv),
                        4096,
                    )
                    .unwrap(),
            )
        }));
    }
}

fn matmulscales() {
    let exec = cpuexec();
    let (a, b) = matmulprefillinput().unwrap();
    let (av, bv) = (a.getview(), b.getview());
    report(&run("matmul/scale/prefill2048", || {
        black_box(
            exec.applybinary(cpuop::matmul, black_box(&av), black_box(&bv))
                .unwrap(),
        )
    }));

    let (a, b) = matmuldecodeinput().unwrap();
    let (av, bv) = (a.getview(), b.getview());
    report(&run("matmul/scale/decode1x4096x4096", || {
        black_box(
            exec.applybinary(cpuop::matmul, black_box(&av), black_box(&bv))
                .unwrap(),
        )
    }));

    let (a, b) = matmulprefillinput().unwrap();
    let (av, bv) = (a.getview(), b.getview());
    for workers in [4usize, 8] {
        let execw = cpuexecw(workers).unwrap();
        report(&run(&format!("matmul/scale/prefill2048w{workers}"), || {
            black_box(
                execw
                    .applybinary(cpuop::matmul, black_box(&av), black_box(&bv))
                    .unwrap(),
            )
        }));
    }
}

fn attentionscales() {
    let exec = cpuexec();
    for (name, qseq, kvseq) in [
        ("prefill512", ATTNPREFILLSEQ, ATTNPREFILLSEQ),
        ("prefill2048", ATTNPREFILLLONG, ATTNPREFILLLONG),
        ("decode4096", 1usize, ATTNDECODEKV),
    ] {
        let (q, k, v) = attentionseqinput(qseq, kvseq).unwrap();
        let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
        report(&run(&format!("attention/scale/{name}/causal"), || {
            black_box(
                compute::attention(
                    &exec,
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
        for workers in [4usize, 8] {
            if qseq == 1 {
                break;
            }
            let execw = cpuexecw(workers).unwrap();
            report(&run(
                &format!("attention/scale/{name}/causalw{workers}"),
                || {
                    black_box(
                        compute::attention(
                            &execw,
                            black_box(&qv),
                            black_box(&kv),
                            black_box(&vv),
                            HEADS,
                            HEADS,
                            true,
                        )
                        .unwrap(),
                    )
                },
            ));
        }
    }
}

fn fusedrmsnormmatmul() {
    let exec = cpuexec();
    for (name, seq, n) in [
        ("prefill2048", 2048usize, 4096usize),
        ("decode1", 1usize, 4096usize),
        ("qkv2048", 2048usize, 16384usize),
    ] {
        let k = 4096usize;
        let x = tensor::new(shape::new(vec![seq, k]).unwrap(), vec![0.5_f32; seq * k]).unwrap();
        let w = tensor::new(shape::new(vec![k]).unwrap(), vec![1.0_f32; k]).unwrap();
        let b = tensor::new(shape::new(vec![k, n]).unwrap(), vec![0.25_f32; k * n]).unwrap();
        let (xv, wv, bv) = (x.getview(), w.getview(), b.getview());
        report(&run(&format!("fused/rmsnormmatmul/{name}/fused"), || {
            black_box(
                exec.rmsnormmatmul(black_box(&xv), black_box(&wv), black_box(&bv), 1.0e-5)
                    .unwrap(),
            )
        }));
        report(&run(
            &format!("fused/rmsnormmatmul/{name}/separate"),
            || {
                let normed = exec
                    .rmsnorm(black_box(&xv), black_box(&wv), 1.0e-5)
                    .unwrap();
                black_box(
                    exec.matmul(black_box(&normed.getview()), black_box(&bv))
                        .unwrap(),
                )
            },
        ));
    }
}

fn main() {
    elementwise();
    softmax();
    binary();
    weighted();
    gather();
    compute_paths();
    scan();
    matmulscales();
    attentionscales();
    fusedrmsnormmatmul();
}

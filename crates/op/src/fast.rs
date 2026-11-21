//! 高吞吐 CPU GEMM：多线程 + A 列面板打包 + 8×8 寄存器分块。
//! 语义同 [`crate::matmul::matmul_scalar`]，浮点累加顺序不同，按 IEEE 容差对拍（不做逐位相等）。
//!
//! 关键在「打包」：A 行主序下按列取 8 行会跨 8 条 cache line、每条只用 4 字节，微内核会被访存打穿。
//! 先把每个 8 行块转置成 `[p][8]` 面板，微内核每条 `p` 只做两次**连续**向量加载，再用 `lane` 广播
//! 到 8 个累加器；微内核按目标架构选择：aarch64 走 NEON，x86_64 走 AVX2+FMA（运行时探测，缺
//! 能力回退标量），其余架构走标量。`unsafe` 只在这两段 SIMD 微内核（`#[allow(unsafe_code)]`）。

use std::thread;

use diagnostic::{Code, Error, Result};

mod pack;
mod pool;
mod scratch;
mod sme;
use pack::{pack_b_f32_into, pack_b_f32_par_into, pack_panels};
use pool::{Buffers, Shape, pool, run_pooled};
mod tn;
pub use tn::{MIN_ROWS, gemm_f32_tn};

/// 微内核行数。
const MR: usize = 8;

/// 微内核列数（两个 NEON 向量）。
const NR: usize = 8;

/// 内积分块长度：`K = 4096` 时整段 B 面板有 4MB，按 256 切块让 A 面板与 B 瓦片同时驻留 L1/L2。
const KB: usize = 256;

/// 多线程稠密 f32 GEMM：`out[m,n] = a[m,k] · b[k,n]`。
///
/// # Panics
///
/// 三条缓冲长度不足 `m·k` / `k·n` / `m·n` 时 panic（调用方先校验）。
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    check_dims(a, b, m, n, k, out);
    // SME 通道：16×64 整铺的大 GEMM 交给 ZA 外积；小题或不可整铺时退回下方 NEON 通道。
    if sme::gemm(a, b, m, n, k, out) {
        return;
    }
    let blocks = m.div_ceil(MR);
    let t = threads_override().min(blocks).max(1);
    if t <= 1 {
        gemm_f32_single(a, b, m, n, k, out);
        return;
    }
    // 常驻池：打包与计算共用一次发布（一次 Barrier 隔开两阶段），无重复 spawn。
    let npad = n.div_ceil(NR) * NR;
    scratch::with_panels(npad * k, |buf| {
        if run_pooled(
            pool(),
            Buffers {
                a: a.as_ptr() as *const u8,
                b: b.as_ptr() as *const u8,
                buf: buf.as_mut_ptr() as *mut u8,
                out: out.as_ptr() as *mut u8,
            },
            Shape {
                m,
                n,
                k,
                npad,
                tn: false,
            },
            MR,
            t,
            f32_pack,
            f32_calc,
        ) {
            return;
        }
        scoped_f32(a, b, out, m, n, k, t);
    });
}

/// 显式错误版稠密 f32 GEMM：三条缓冲与 `(m, n, k)` 严格相等才跑，不符返回 [`Code::Invalid`]，
/// 不 panic。算术同 [`gemm_f32`]（NEON 微内核 + 多槽池）。
pub fn gemm_f32_checked(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "gemm_f32 缓冲尺寸不符：a={} b={} out={}，期望 m·k={} k·n={} m·n={}",
                a.len(),
                b.len(),
                out.len(),
                m * k,
                k * n,
                m * n
            ),
        ));
    }
    gemm_f32(a, b, m, n, k, out);
    Ok(())
}

/// 尺寸校验：三条缓冲不足 `m·k` / `k·n` / `m·n` 时 panic。
fn check_dims(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &[f32]) {
    assert!(
        a.len() >= m * k && b.len() >= k * n && out.len() >= m * n,
        "gemm_f32 尺寸不符"
    );
}

/// 兜底（池不可用）：B 打包一次后 shared 给 scope 线程。
fn scoped_f32(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize, t: usize) {
    let npad = n.div_ceil(NR) * NR;
    let blocks = m.div_ceil(MR);
    // 行块按 MR 对齐：否则 18 线程切 256 行会得到 15 行/线程，8 行走微内核、7 行掉回标量尾块。
    let per = blocks.div_ceil(t) * MR;
    scratch::with_panels(npad * k, |bpack| {
        pack_b_f32_par_into(bpack, b, n, k, t);
        let bp: &[f32] = bpack;
        thread::scope(|s| {
            for (ti, block) in out[..m * n].chunks_mut(per * n).enumerate() {
                let r0 = ti * per;
                let rows = block.len() / n;
                let a_t = &a[r0 * k..(r0 + rows) * k];
                s.spawn(move || compute(a_t, bp, b, rows, n, k, block));
            }
        });
    });
}

/// 单线程稠密 f32 GEMM（基准确认内核质量用）：`out[m,n] = a[m,k] · b[k,n]`。
pub fn gemm_f32_single(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    let npad = n.div_ceil(NR) * NR;
    scratch::with_panels(npad * k, |bpack| {
        pack_b_f32_into(bpack, b, n, k);
        compute(a, bpack, b, m, n, k, out);
    });
}

/// 单线程块计算：A 打包成 8 行面板，再 j 外层、面板内层跑微内核；残尾走标量。
fn compute(a: &[f32], bpack: &[f32], b: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
    let full = rows / MR;
    let apack = pack_panels(a, full, k);
    let mut first = true;
    let mut k0 = 0;
    while k0 < k {
        let kb = KB.min(k - k0);
        sweep_columns(
            Sweep {
                a,
                apack: &apack,
                bpack,
                w: b,
                full,
                n,
                k,
                layout: Layout::Nt,
            },
            out,
            k0,
            kb,
            first,
        );
        first = false;
        k0 += kb;
    }
    // 残行只占 ≤7 行，一次把整段 K 走完，不参与分块累加。
    let tail = full * MR;
    if tail < rows {
        edge(
            a,
            b,
            out,
            Span {
                i: tail,
                j: 0,
                mr: rows - tail,
                nr: n,
                n,
                k,
                stride: k,
            },
            0,
            true,
            Layout::Nt,
        );
    }
}

/// 沿列块扫描整行块：整块走微内核，最后一段残列走标量边（`edge` 读未打包的 `a`）。
fn sweep_columns(sweep: Sweep<'_>, out: &mut [f32], k0: usize, kb: usize, first: bool) {
    let Sweep {
        a,
        apack,
        bpack,
        w,
        full,
        n,
        k,
        layout,
    } = sweep;
    let mut j = 0;
    while j < n {
        let nr = NR.min(n - j);
        if nr == NR {
            let start = (j / NR) * k * NR + k0 * NR;
            let b_tile = &bpack[start..start + kb * NR];
            for blk in 0..full {
                micro_packed(
                    &apack[blk * k * MR + k0 * MR..],
                    b_tile,
                    out,
                    Tile {
                        row0: blk * MR,
                        j,
                        n,
                        kb,
                        first,
                    },
                );
            }
        } else {
            edge(
                a,
                w,
                out,
                Span {
                    i: 0,
                    j,
                    mr: full * MR,
                    nr,
                    n,
                    k: kb,
                    stride: k,
                },
                k0,
                first,
                layout,
            );
        }
        j += NR;
    }
}

mod gemm_i8;
mod micro;

use gemm_i8::{f32_calc, f32_pack, threads_override};
pub use gemm_i8::{gemm_i8, gemm_i8_checked, gemm_i8_single};
use micro::{Layout, Span, Sweep, Tile, edge, micro_packed};

#[cfg(test)]
#[path = "../tests/fast_tests.rs"]
mod tests;

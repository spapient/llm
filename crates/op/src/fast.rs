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
use pack::{b_groups, pack_b_f32_into, pack_b_f32_par_into, pack_panels};
use pool::{Buffers, Job, Shape, pool, run_pooled};
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

/// 一次整行块扫描的只读侧：指针、面板与三个尺寸一次带走（实参不散成一长串）。
#[derive(Clone, Copy)]
struct Sweep<'a> {
    a: &'a [f32],
    apack: &'a [f32],
    bpack: &'a [f32],
    w: &'a [f32],
    full: usize,
    n: usize,
    k: usize,
    layout: Layout,
}

/// 一个待算的 8×8 微内核块：左上角 `(row0, j)`、矩阵列宽 `n`、内积分块长 `kb`、是否首块。
///
/// 五数同行进出一个值，理由同 [`Span`]：拆回参数会让三份架构实现各把同一组坐标重拼一遍。
#[derive(Clone, Copy)]
struct Tile {
    row0: usize,
    j: usize,
    n: usize,
    kb: usize,
    first: bool,
}

/// 打包面板上的 NEON 8×8 微内核（A 面板 `[p][8]`、B 面板 `[p][8]`，一次算 8 行 8 列）。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    let Tile {
        row0,
        j,
        n,
        kb,
        first,
    } = tile;
    use std::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let pp = panel.as_ptr();
        let bp = b.as_ptr();
        // 后续内积分块接着上一块的累加器算：首块用 0 起手，否则读回 `out` 的 16 个向量。
        let mut acc = if first {
            [zero; 16]
        } else {
            micro_load(out, row0, n, j)
        };
        for p in 0..kb {
            let av0 = vld1q_f32(pp.add(p * MR));
            let av1 = vld1q_f32(pp.add(p * MR + 4));
            let bv0 = vld1q_f32(bp.add(p * NR));
            let bv1 = vld1q_f32(bp.add(p * NR + 4));
            // lane-indexed FMA 把广播并进乘加：每条 `p` 从 24 条指令降到 20 条（省 8 条 dup）。
            acc[0] = vfmaq_laneq_f32::<0>(acc[0], bv0, av0);
            acc[1] = vfmaq_laneq_f32::<0>(acc[1], bv1, av0);
            acc[2] = vfmaq_laneq_f32::<1>(acc[2], bv0, av0);
            acc[3] = vfmaq_laneq_f32::<1>(acc[3], bv1, av0);
            acc[4] = vfmaq_laneq_f32::<2>(acc[4], bv0, av0);
            acc[5] = vfmaq_laneq_f32::<2>(acc[5], bv1, av0);
            acc[6] = vfmaq_laneq_f32::<3>(acc[6], bv0, av0);
            acc[7] = vfmaq_laneq_f32::<3>(acc[7], bv1, av0);
            acc[8] = vfmaq_laneq_f32::<0>(acc[8], bv0, av1);
            acc[9] = vfmaq_laneq_f32::<0>(acc[9], bv1, av1);
            acc[10] = vfmaq_laneq_f32::<1>(acc[10], bv0, av1);
            acc[11] = vfmaq_laneq_f32::<1>(acc[11], bv1, av1);
            acc[12] = vfmaq_laneq_f32::<2>(acc[12], bv0, av1);
            acc[13] = vfmaq_laneq_f32::<2>(acc[13], bv1, av1);
            acc[14] = vfmaq_laneq_f32::<3>(acc[14], bv0, av1);
            acc[15] = vfmaq_laneq_f32::<3>(acc[15], bv1, av1);
        }
        micro_store(out, row0, n, j, acc);
    }
}

/// 读回上一步的 16 个累加向量（非首块续算用）。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn micro_load(
    out: &[f32],
    row0: usize,
    n: usize,
    j: usize,
) -> [std::arch::aarch64::float32x4_t; 16] {
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = [vdupq_n_f32(0.0); 16];
        let prev = out.as_ptr();
        let r = row0;
        acc[0] = vld1q_f32(prev.add(r * n + j));
        acc[1] = vld1q_f32(prev.add(r * n + j + 4));
        acc[2] = vld1q_f32(prev.add((r + 1) * n + j));
        acc[3] = vld1q_f32(prev.add((r + 1) * n + j + 4));
        acc[4] = vld1q_f32(prev.add((r + 2) * n + j));
        acc[5] = vld1q_f32(prev.add((r + 2) * n + j + 4));
        acc[6] = vld1q_f32(prev.add((r + 3) * n + j));
        acc[7] = vld1q_f32(prev.add((r + 3) * n + j + 4));
        acc[8] = vld1q_f32(prev.add((r + 4) * n + j));
        acc[9] = vld1q_f32(prev.add((r + 4) * n + j + 4));
        acc[10] = vld1q_f32(prev.add((r + 5) * n + j));
        acc[11] = vld1q_f32(prev.add((r + 5) * n + j + 4));
        acc[12] = vld1q_f32(prev.add((r + 6) * n + j));
        acc[13] = vld1q_f32(prev.add((r + 6) * n + j + 4));
        acc[14] = vld1q_f32(prev.add((r + 7) * n + j));
        acc[15] = vld1q_f32(prev.add((r + 7) * n + j + 4));
        acc
    }
}

/// 把 16 个累加向量写回 `out` 的 8×8 块。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn micro_store(
    out: &mut [f32],
    row0: usize,
    n: usize,
    j: usize,
    acc: [std::arch::aarch64::float32x4_t; 16],
) {
    unsafe {
        use std::arch::aarch64::*;
        let op = out.as_mut_ptr();
        let r = row0;
        vst1q_f32(op.add(r * n + j), acc[0]);
        vst1q_f32(op.add(r * n + j + 4), acc[1]);
        vst1q_f32(op.add((r + 1) * n + j), acc[2]);
        vst1q_f32(op.add((r + 1) * n + j + 4), acc[3]);
        vst1q_f32(op.add((r + 2) * n + j), acc[4]);
        vst1q_f32(op.add((r + 2) * n + j + 4), acc[5]);
        vst1q_f32(op.add((r + 3) * n + j), acc[6]);
        vst1q_f32(op.add((r + 3) * n + j + 4), acc[7]);
        vst1q_f32(op.add((r + 4) * n + j), acc[8]);
        vst1q_f32(op.add((r + 4) * n + j + 4), acc[9]);
        vst1q_f32(op.add((r + 5) * n + j), acc[10]);
        vst1q_f32(op.add((r + 5) * n + j + 4), acc[11]);
        vst1q_f32(op.add((r + 6) * n + j), acc[12]);
        vst1q_f32(op.add((r + 6) * n + j + 4), acc[13]);
        vst1q_f32(op.add((r + 7) * n + j), acc[14]);
        vst1q_f32(op.add((r + 7) * n + j + 4), acc[15]);
    }
}

/// 标量微内核：无 SIMD 的架构（以及 x86_64 上未检出 AVX2/FMA 的机器）的兜底。
///
/// 与 SIMD 微内核同一组累加口径：首块从 0 起手，非首块读回 `out` 续算。逐元素点积不做向量化，
/// 只保证正确，不保证吞吐。
#[cfg(not(target_arch = "aarch64"))]
fn micro_packed_scalar(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    let Tile {
        row0,
        j,
        n,
        kb,
        first,
    } = tile;
    for r in 0..MR {
        for c in 0..NR {
            let slot = (row0 + r) * n + j + c;
            let mut acc = if first { 0.0f32 } else { out[slot] };
            for p in 0..kb {
                acc += panel[p * MR + r] * b[p * NR + c];
            }
            out[slot] = acc;
        }
    }
}

/// x86_64 的 8×8 微内核：AVX2 + FMA，8 个 `__m256` 累加器（每行一个、8 列一个向量）。
///
/// `panel` 布局 `[p][8]`（A 面板），`b` 布局 `[p][8]`（B 面板）。CPU 能力在运行时探测一次：
/// 没有 AVX2/FMA 的 x86_64 机器回退 [`micro_packed_scalar`]，不在源码里对硬件能力做隐式假设。
/// x86_64 微内核入口：运行时探测 AVX2+FMA，命中走向量核，否则退回标量核。
#[cfg(target_arch = "x86_64")]
fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        micro_packed_avx2(panel, b, out, tile);
    } else {
        micro_packed_scalar(panel, b, out, tile);
    }
}

/// 8×8 AVX2+FMA 微内核：每行一个 `__m256` 累加器，逐收缩步广播 A 值 × B 向量做 FMA。
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn micro_packed_avx2(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    let Tile {
        row0,
        j,
        n,
        kb,
        first,
    } = tile;
    use std::arch::x86_64::*;
    unsafe {
        let zero = _mm256_setzero_ps();
        let pp = panel.as_ptr();
        let bp = b.as_ptr();
        let mut acc = if first {
            [zero; 8]
        } else {
            micro_load(out, row0, n, j)
        };
        for p in 0..kb {
            let bv = _mm256_loadu_ps(bp.add(p * NR));
            for r in 0..MR {
                acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(*pp.add(p * MR + r)), bv, acc[r]);
            }
        }
        micro_store(out, row0, n, j, acc);
    }
}

/// 读回上一步的 8 个累加向量（x86_64 非首块续算用）。
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn micro_load(out: &[f32], row0: usize, n: usize, j: usize) -> [std::arch::x86_64::__m256; 8] {
    unsafe {
        use std::arch::x86_64::*;
        let prev = out.as_ptr();
        let r = row0;
        [
            _mm256_loadu_ps(prev.add(r * n + j)),
            _mm256_loadu_ps(prev.add((r + 1) * n + j)),
            _mm256_loadu_ps(prev.add((r + 2) * n + j)),
            _mm256_loadu_ps(prev.add((r + 3) * n + j)),
            _mm256_loadu_ps(prev.add((r + 4) * n + j)),
            _mm256_loadu_ps(prev.add((r + 5) * n + j)),
            _mm256_loadu_ps(prev.add((r + 6) * n + j)),
            _mm256_loadu_ps(prev.add((r + 7) * n + j)),
        ]
    }
}

/// 把 8 个累加向量写回 `out` 的 8×8 块（x86_64）。
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
fn micro_store(
    out: &mut [f32],
    row0: usize,
    n: usize,
    j: usize,
    acc: [std::arch::x86_64::__m256; 8],
) {
    unsafe {
        use std::arch::x86_64::*;
        let op = out.as_mut_ptr();
        let r = row0;
        for (i, v) in acc.into_iter().enumerate() {
            _mm256_storeu_ps(op.add((r + i) * n + j), v);
        }
    }
}

/// 无 SIMD 的架构：直接走标量微内核（`micro_packed_scalar` 在这些目标上不会被别处引用）。
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
    micro_packed_scalar(panel, b, out, tile);
}

/// 一个待算的标量尾块：左上角 `(i, j)`、覆盖 `mr × nr`，矩阵列宽 `n`、内积长 `k`。
///
/// 六个数合起来才描述得清「哪一块」，所以它们同行进出一个值——拆回参数会让每处调用把同一组
/// 坐标重拼一遍，也让「块多大」这件事在两处各写一份。
#[derive(Clone, Copy)]
struct Span {
    i: usize,
    j: usize,
    mr: usize,
    nr: usize,
    n: usize,
    k: usize,
    /// `a` 的整行宽：内积分块时 `k` 只是当前块，跨块偏移要按整行宽算。
    stride: usize,
}

/// 权重布局：`Nt` 是收缩轴在前的 `b[k][n]`，`Tn` 是按输出行常驻的 `w[n][k]`。
///
/// 两种布局只在「权重怎么寻址」上不同，换算成同一组 `(基址, 内积步长, 列步长)` 后，
/// 标量尾块与打包扫描都只留一份实现。
#[derive(Clone, Copy)]
enum Layout {
    Nt,
    Tn,
}

/// 权重在 `w` 里的寻址三元组：`(基址, 内积步长, 列步长)`——两种布局在这一处收敛。
fn weight_stride(span: Span, k0: usize, layout: Layout) -> (usize, usize, usize) {
    match layout {
        Layout::Nt => (k0 * span.n + span.j, span.n, 1),
        Layout::Tn => (span.j * span.stride + k0, 1, span.stride),
    }
}

/// 标量尾块：`out[i..i+mr, j..j+nr]` 逐元素点积（与微内核同一组累加口径，容差内一致）。
fn edge(a: &[f32], w: &[f32], out: &mut [f32], span: Span, k0: usize, first: bool, layout: Layout) {
    let (base, dp, dc) = weight_stride(span, k0, layout);
    for r in 0..span.mr {
        for c in 0..span.nr {
            let slot = (span.i + r) * span.n + span.j + c;
            let mut acc = if first { 0.0f32 } else { out[slot] };
            let wc = base + c * dc;
            for p in 0..span.k {
                acc += a[(span.i + r) * span.stride + k0 + p] * w[wc + p * dp];
            }
            out[slot] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// i8 sdot 通道：MR=4 × NR=4，B 转置打包成「每列 k 连续」。
// 每个 k16 一次 4 行 A 加载 + 4 次 B 加载 + 16 次 `vdotq_s32`（256 MAC/24 指令），
// 且每块 B 瓦片被 4 行复用——B 访存比 MR=2/NR=8 再降一半。
// dotprod 运行时探测，缺失走标量回退；`unsafe` 仅限 aarch64 微内核。
// ---------------------------------------------------------------------------

/// i8 微内核行数（四行共享同一 B 向量，摊薄 B 加载）。
const I8_MR: usize = 4;

/// i8 微内核列数（4 个 int32x4 累加器 × 4 行 = 16 寄存器，避开溢出）。
const I8_NR: usize = 4;

/// 线程数：`LLM_THREADS` 环境变量优先，否则全核。
fn threads_override() -> usize {
    std::env::var("LLM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|x| *x > 0)
        .unwrap_or_else(|| {
            thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
        })
}

/// 打包阶段：本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包。
fn i8_pack(j: &Job, slot: usize) {
    let p = j.p;
    let groups = j.npad / I8_NR;
    #[allow(unsafe_code)]
    unsafe {
        let b = j.b as *const i8;
        let bp = j.buf as *mut i8;
        let n = j.n;
        let k = j.k;
        let mut g = slot;
        while g < groups {
            for nb in 0..I8_NR {
                let jj = g * I8_NR + nb;
                if jj < n {
                    let dst = bp.add((g * I8_NR + nb) * k);
                    let mut q = 0;
                    while q < k {
                        *dst.add(q) = *b.add(q * n + jj);
                        q += 1;
                    }
                }
            }
            g += p;
        }
    }
}

/// 计算阶段：只算本槽的行段，`bpack` 由 Barrier 保证整块就绪。
fn i8_calc(j: &Job, slot: usize) {
    let r0 = slot * j.rows_per;
    if r0 >= j.m {
        return;
    }
    let rows = (j.m - r0).min(j.rows_per);
    #[allow(unsafe_code)]
    unsafe {
        let a_l = std::slice::from_raw_parts(j.a as *const i8, j.m * j.k);
        let a_l = &a_l[r0 * j.k..(r0 + rows) * j.k];
        let b_l = std::slice::from_raw_parts(j.buf as *const i8, j.npad * j.k);
        let out_l = std::slice::from_raw_parts_mut(j.out as *mut i32, j.m * j.n);
        let out_l = &mut out_l[r0 * j.n..(r0 + rows) * j.n];
        compute_i8_packed(a_l, b_l, j.npad, rows, j.n, j.k, out_l);
    }
}

/// f32 打包体：本槽负责的 8 列组按 `p` 步长轮转，写入 `[jblk][p][NR]`（零初始化覆盖补零列）。
///
/// 转置布局（[`tn`]）与行主序布局共用同一条池通道：两者只差一次读步长，所以差异收在
/// [`pack::b_groups`] 与 [`tn::stage`] 里，池与闸门不必各备份。
#[allow(unsafe_code)]
fn f32_pack(j: &Job, slot: usize) {
    let (n, k, npad) = (j.n, j.k, j.npad);
    unsafe {
        let b = std::slice::from_raw_parts(j.b as *const f32, k * n);
        let dst = std::slice::from_raw_parts_mut(j.buf as *mut f32, npad * k);
        if j.tn {
            tn::stage(b, dst, n, k, npad, j.p, slot);
        } else {
            b_groups(b, dst, n, k, npad, j.p, slot);
        }
    }
}

/// f32 计算体：只算本槽的行段；B 面板由 Barrier 保证整块就绪。
#[allow(unsafe_code)]
fn f32_calc(j: &Job, slot: usize) {
    let r0 = slot * j.rows_per;
    if r0 >= j.m {
        return;
    }
    let rows = (j.m - r0).min(j.rows_per);
    unsafe {
        let a = std::slice::from_raw_parts(j.a as *const f32, j.m * j.k);
        let a_l = &a[r0 * j.k..(r0 + rows) * j.k];
        let bp = std::slice::from_raw_parts(j.buf as *const f32, j.npad * j.k);
        let b = std::slice::from_raw_parts(j.b as *const f32, j.k * j.n);
        let out = std::slice::from_raw_parts_mut(j.out as *mut f32, j.m * j.n);
        let out_l = &mut out[r0 * j.n..(r0 + rows) * j.n];
        if j.tn {
            tn::compute_tn(a_l, bp, b, rows, j.n, j.k, out_l);
        } else {
            compute(a_l, bp, b, rows, j.n, j.k, out_l);
        }
    }
}

/// 多线程 i8 GEMM：`out[m,n] = a[m,k] · b[k,n]`，i8 入 i32 累加。
///
/// # Panics
///
/// 三条缓冲长度不足时 panic（调用方先校验）。
pub fn gemm_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    assert!(
        a.len() >= m * k && b.len() >= k * n && out.len() >= m * n,
        "gemm_i8 尺寸不符"
    );
    let cpu = threads_override();
    let npad = n.div_ceil(I8_NR) * I8_NR;
    let groups = npad / I8_NR;
    let t = cpu.min(m.div_ceil(I8_MR)).min(groups).max(1);
    if t <= 1 {
        single_i8(a, b, m, n, k, out);
        return;
    }
    // 常驻池：一次发布 + 一次 Barrier，无 spawn 成本。
    let mut bpack = vec![0i8; npad * k];
    if run_pooled(
        pool(),
        Buffers {
            a: a.as_ptr() as *const u8,
            b: b.as_ptr() as *const u8,
            buf: bpack.as_mut_ptr() as *mut u8,
            out: out.as_mut_ptr() as *mut u8,
        },
        Shape {
            m,
            n,
            k,
            npad,
            tn: false,
        },
        I8_MR,
        t,
        i8_pack,
        i8_calc,
    ) {
        return;
    }
    scoped_i8(a, b, out, m, n, k, t);
}

/// 显式错误版对称 int8 GEMM：三条缓冲与 `(m, n, k)` 严格相等才跑，不符返回 [`Code::Invalid`]，
/// 不 panic。算术同 [`gemm_i8`]（i32 精确累加，与 [`crate::matmul_i8::forward_tiled`] 逐位一致）。
pub fn gemm_i8_checked(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [i32],
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "gemm_i8 缓冲尺寸不符：a={} b={} out={}，期望 m·k={} k·n={} m·n={}",
                a.len(),
                b.len(),
                out.len(),
                m * k,
                k * n,
                m * n
            ),
        ));
    }
    gemm_i8(a, b, m, n, k, out);
    Ok(())
}

/// 单线程路径：B 打包一次，直接算完整块。
fn single_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    let (bpack, npad) = pack_b(b, n, k);
    compute_i8_packed(a, &bpack, npad, m, n, k, out);
}

/// 兜底路径：池不可用（worker 起不来）时退回 scope 线程。
fn scoped_i8(a: &[i8], b: &[i8], out: &mut [i32], m: usize, n: usize, k: usize, t: usize) {
    let mut rows_per = m.div_ceil(t);
    // 每线程成组行，保证微内核分块不被切碎（缺则末线程走标量尾块）。
    rows_per = rows_per.div_ceil(I8_MR) * I8_MR;
    // B 只打包一次、全部线程共享：N 线程各打包一遍会让冗余带宽吞掉扩展性。
    let (bpack, npad) = pack_b_par(b, n, k);
    thread::scope(|s| {
        for (ti, block) in out[..m * n].chunks_mut(rows_per * n).enumerate() {
            let r0 = ti * rows_per;
            let rows = block.len() / n;
            let a_t = &a[r0 * k..(r0 + rows) * k];
            let bp = &bpack;
            s.spawn(move || compute_i8_packed(a_t, bp, npad, rows, n, k, block));
        }
    });
}

/// 单线程 i8 GEMM（基准确认内核质量用）。
pub fn gemm_i8_single(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    single_i8(a, b, m, n, k, out);
}

/// B 转置打包成 `[nblk][nb][k]`（同列 k 连续），返回 `(面板, n 补齐到 8)`。
fn pack_b(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
    let npad = n.div_ceil(I8_NR) * I8_NR;
    let mut bpack = vec![0i8; npad * k];
    for jj in 0..n {
        let base = ((jj / I8_NR) * I8_NR + jj % I8_NR) * k;
        for p in 0..k {
            bpack[base + p] = b[p * n + jj];
        }
    }
    (bpack, npad)
}

/// 并行 B 打包：转置是跨 `n` 字节步长的 cache-miss 密集写，小 `n` 大 `k` 时串行打包
/// 会独占整段耗时（如 256×256×4096 的 1MB 面板），这里按 8 列组切成线程。
fn pack_b_par(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
    let npad = n.div_ceil(I8_NR) * I8_NR;
    let groups = npad / I8_NR;
    let t = threads_override().min(groups);
    if t <= 1 || npad * k < (1 << 16) {
        return pack_b(b, n, k);
    }
    let mut bpack = vec![0i8; npad * k];
    let per = groups.div_ceil(t);
    let span = per * I8_NR * k;
    thread::scope(|s| {
        for (gi, chunk) in bpack.chunks_mut(span).enumerate() {
            let blk0 = gi * per;
            s.spawn(move || pack_b_span(chunk, b, n, k, blk0));
        }
    });
    (bpack, npad)
}

/// 一段连续的列组：`blk0` 起、`chunk` 覆盖的每组 8 列各搬一块。
fn pack_b_span(chunk: &mut [i8], b: &[i8], n: usize, k: usize, blk0: usize) {
    let blocks = chunk.len() / (I8_NR * k);
    for bi in 0..blocks {
        let tile = &mut chunk[bi * I8_NR * k..(bi + 1) * I8_NR * k];
        pack_b_tile(tile, b, n, k, (blk0 + bi) * I8_NR);
    }
}

/// 一组 8 列瓦片的转置写：`b`（行主、`n` 列）的第 `first..first+8` 列搬到 `dst`（列主块内偏移）。
/// 越界的列（`n` 不是 8 的倍数）留 0——打包缓冲本来就按 `npad` 补齐，乘加时该列权重为 0。
fn pack_b_tile(dst: &mut [i8], b: &[i8], n: usize, k: usize, first: usize) {
    for nb in 0..I8_NR {
        let jj = first + nb;
        if jj >= n {
            continue;
        }
        let off = nb * k;
        let mut p = 0;
        while p < k {
            dst[off + p] = b[p * n + jj];
            p += 1;
        }
    }
}

/// 单线程块计算（B 面板已备）：**N 分组在最外层**，每组 8×k 的 B 瓦片常驻 L1，
/// 被本线程所有行对复用——否则每行对都把整块 B 重读一遍，卡在内存带宽。
fn compute_i8_packed(
    a: &[i8],
    bpack: &[i8],
    npad: usize,
    rows: usize,
    n: usize,
    k: usize,
    out: &mut [i32],
) {
    let kvec = k - k % 16;
    // 行块要让 A 的 `rc × k` 字节留在 L1（约 128KB），否则每个列块都要从 L2 重拉整块 A。
    let rc = ((1usize << 17) / k.max(1)).clamp(I8_MR, 256);
    let rc = (rc / I8_MR) * I8_MR;
    let mut r0 = 0;
    while r0 < rows {
        let rend = (r0 + rc).min(rows);
        let mut blk = 0;
        while blk < npad {
            let mut r = r0;
            while r + I8_MR <= rend {
                pair_one(a, bpack, out, Gram { r, blk, n, k, kvec });
                r += I8_MR;
            }
            while r < rend {
                scalar_row_group(a, bpack, out, r, blk, n, k);
                r += 1;
            }
            blk += I8_NR;
        }
        r0 = rend;
    }
}

/// 一个 8 列块的两行：起点行 `r`、列块起点 `blk`，矩阵列宽 `n`、内积长 `k` 与向量化长 `kvec`。
///
/// 五个数合起来才说清「算哪一块」，所以同行进出一个值；`kvec` 与 `k` 分开是为了让标量尾段有
/// 一处明确的边界，而不是在微内核里再算一遍。
///
/// `kvec` 只有 aarch64 dotprod 通道读它（`blk` 沿 16 字节一拍走，余下 `kvec..k` 由标量补）：
/// x86_64 的 SIMD 通道服务 f32，这条 i8 路径在这边本来就是标量，用不上分开的边界。
#[derive(Clone, Copy)]
struct Gram {
    r: usize,
    blk: usize,
    n: usize,
    k: usize,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    kvec: usize,
}

/// 单个 8 列块、两行微内核分派：dotprod 可用且 k≥16 走向量，否则标量两行。
#[allow(unsafe_code)]
fn pair_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
    #[cfg(target_arch = "aarch64")]
    {
        if g.kvec >= 16 && std::arch::is_aarch64_feature_detected!("dotprod") {
            // 安全性：dotprod 已运行时探测。
            unsafe { dot_one(a, bpack, out, g) };
            return;
        }
    }
    scalar_row_group(a, bpack, out, g.r, g.blk, g.n, g.k);
    for mr in 1..I8_MR {
        scalar_row_group(a, bpack, out, g.r + mr, g.blk, g.n, g.k);
    }
}

/// aarch64 dotprod：一个 4 列块的 4 行（`unsafe` 岛）。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
unsafe fn dot_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
    unsafe {
        let ap = a.as_ptr().add(g.r * g.k);
        let bp = bpack.as_ptr();
        let accs = dot_block(ap, bp, g.blk, g.k, g.kvec);
        harvest(ap, bp, out, (g.r, g.n, g.k, g.kvec), g.blk, accs);
    }
}

/// 一个 4 列块的 4 行累加：`(4 行 × 4 列的整数累加和)`。
///
/// `blk` 是列块起点、`p` 沿 `kvec` 走 16 字节一拍；`ave[mr]` 是第 `r+mr` 行。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
unsafe fn dot_block(
    ap: *const i8,
    bp: *const i8,
    blk: usize,
    k: usize,
    kvec: usize,
) -> [[std::arch::aarch64::int32x4_t; I8_NR]; I8_MR] {
    use std::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_s32(0);
        // 列优先：一列一个内层迭代，块内推进不必同时索引四行。
        let mut accs = [[zero; I8_MR]; I8_NR];
        let mut p = 0;
        while p < kvec {
            let av: [int8x16_t; I8_MR] = [
                vld1q_s8(ap.add(p)),
                vld1q_s8(ap.add(k + p)),
                vld1q_s8(ap.add(2 * k + p)),
                vld1q_s8(ap.add(3 * k + p)),
            ];
            for (nb, column) in accs.iter_mut().enumerate() {
                let bv = vld1q_s8(bp.add((blk + nb) * k + p));
                for (cell, a) in column.iter_mut().zip(av) {
                    *cell = vdotq_s32(*cell, a, bv);
                }
            }
            p += 16;
        }
        accs
    }
}

/// 一个 4 列块的收口：水平求和 + 标量补 `kvec..k` 的余数，写回 4 行。
///
/// `jj >= n` 的列落在打包填充区，只算不写——填充列的乘积不参与任何输出元素。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
unsafe fn harvest(
    ap: *const i8,
    bp: *const i8,
    out: &mut [i32],
    (r, n, k, kvec): (usize, usize, usize, usize),
    blk: usize,
    accs: [[std::arch::aarch64::int32x4_t; I8_MR]; I8_NR],
) {
    use std::arch::aarch64::*;
    unsafe {
        for (nb, column) in accs.iter().enumerate() {
            let jj = blk + nb;
            if jj < n {
                for (mr, cell) in column.iter().enumerate() {
                    let mut s = vaddvq_s32(*cell);
                    let mut q = kvec;
                    while q < k {
                        s += *ap.add(mr * k + q) as i32 * *bp.add(jj * k + q) as i32;
                        q += 1;
                    }
                    out[(r + mr) * n + jj] = s;
                }
            }
        }
    }
}

/// 标量单行、单 4 列块（打包 B 取数，任意 n/k）。
fn scalar_row_group(
    a: &[i8],
    bpack: &[i8],
    out: &mut [i32],
    row: usize,
    blk: usize,
    n: usize,
    k: usize,
) {
    let ar = &a[row * k..(row + 1) * k];
    for nb in 0..I8_NR {
        let jj = blk + nb;
        if jj < n {
            let bp = &bpack[jj * k..jj * k + k];
            let mut acc = 0i32;
            for p in 0..k {
                acc += ar[p] as i32 * bp[p] as i32;
            }
            out[row * n + jj] = acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::{close_enough, matmul_scalar};

    fn sample_i8(len: usize, salt: u32) -> Vec<i8> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x % 251) as i32 - 125) as i8
            })
            .collect()
    }

    fn check_i8(m: usize, n: usize, k: usize) {
        let a = sample_i8(m * k, 1);
        let b = sample_i8(k * n, 2);
        let mut want = vec![0i32; m * n];
        crate::matmul_i8::forward_tiled(&a, &b, m, n, k, &mut want).unwrap();
        let mut got = vec![0i32; m * n];
        gemm_i8(&a, &b, m, n, k, &mut got);
        assert_eq!(got, want, "gemm_i8 与参考逐元素相等（整数无舍入）");
    }

    #[test]
    fn i8_matches_on_panel_multiples() {
        check_i8(64, 64, 96);
    }

    #[test]
    fn i8_matches_on_ragged_tails() {
        check_i8(37, 19, 33);
    }

    #[test]
    fn i8_matches_on_prefill_shape() {
        check_i8(256, 256, 4096);
    }

    #[test]
    fn i8_concurrent_calls_are_isolated() {
        // 多槽池：并发调用各租一组空闲槽，必须互不干扰且不挂死。
        let shapes = [(64usize, 64usize, 96usize), (80, 32, 128), (96, 48, 64)];
        thread::scope(|s| {
            for (m, n, k) in shapes {
                let a = sample_i8(m * k, 1);
                let b = sample_i8(k * n, 2);
                let mut want = vec![0i32; m * n];
                crate::matmul_i8::forward_tiled(&a, &b, m, n, k, &mut want).unwrap();
                s.spawn(move || {
                    for _ in 0..200 {
                        let mut got = vec![0i32; m * n];
                        gemm_i8(&a, &b, m, n, k, &mut got);
                        assert_eq!(got, want);
                    }
                });
            }
        });
    }

    pub(super) fn sample(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x >> 8) as f32) / 256.0 - 32.0
            })
            .collect()
    }

    /// 生成确定性输入、跑标量参考、调用 `compute` 后与参考对拍。
    /// `sme` 测试复用同一骨架，仅替换计算入口与断言文案。
    pub(super) fn check_gemm<F>(m: usize, n: usize, k: usize, compute: F, msg: &str)
    where
        F: FnOnce(&[f32], &[f32], usize, usize, usize, &mut [f32]),
    {
        let a = sample(m * k, 1);
        let b = sample(k * n, 2);
        let mut want = vec![0.0f32; m * n];
        matmul_scalar(&a, &b, m, n, k, &mut want);
        let mut got = vec![0.0f32; m * n];
        compute(&a, &b, m, n, k, &mut got);
        close_enough(&got, &want, 1e-3, 1e-3).expect(msg);
    }

    fn check(m: usize, n: usize, k: usize) {
        check_gemm(m, n, k, gemm_f32, "fast gemm 与标量参考容差内一致");
    }

    #[test]
    fn matches_scalar_on_tile_multiples() {
        check(64, 64, 96);
    }

    #[test]
    fn matches_scalar_on_ragged_edges() {
        check(100, 70, 4093);
    }

    #[test]
    fn matches_scalar_trivial() {
        check(1, 1, 1);
        check(7, 3, 5);
    }

    #[test]
    fn matches_scalar_on_prefill_shape() {
        check(256, 256, 4096);
    }

    /// 吞吐基线：`cargo test --release -p llm-op --lib fast::bench_throughput -- --ignored --nocapture`
    #[test]
    #[ignore = "吞吐基线，手动跑"]
    fn bench_throughput() {
        use std::time::Instant;
        let shapes = [
            (256usize, 256usize, 4096usize),
            (512, 512, 512),
            (1024, 1024, 1024),
        ];
        println!(
            "threads={}",
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
        );
        let g = |ns: f64, m: usize, n: usize, k: usize| 2.0 * m as f64 * n as f64 * k as f64 / ns;
        for (m, n, k) in shapes {
            let a = sample_i8(m * k, 1);
            let b = sample_i8(k * n, 2);
            let mut out = vec![0i32; m * n];
            gemm_i8(&a, &b, m, n, k, &mut out);
            let mut best = f64::INFINITY;
            for _ in 0..20 {
                let t = Instant::now();
                gemm_i8(&a, &b, m, n, k, &mut out);
                best = best.min(t.elapsed().as_nanos() as f64);
            }
            let mt = g(best, m, n, k);
            gemm_i8_single(&a, &b, m, n, k, &mut out);
            let mut best1 = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                gemm_i8_single(&a, &b, m, n, k, &mut out);
                best1 = best1.min(t.elapsed().as_nanos() as f64);
            }
            println!(
                "  i8 {m}x{n}x{k}: fast {mt:.1} GF/s | fast单 {:.1} GF/s | 单/多 {:.2}",
                g(best1, m, n, k),
                g(best1, m, n, k) / mt
            );
        }
        for (m, n, k) in shapes {
            let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.007).cos()).collect();
            let mut out = vec![0.0f32; m * n];
            gemm_f32(&a, &b, m, n, k, &mut out);
            let mut best = f64::INFINITY;
            for _ in 0..20 {
                let t = Instant::now();
                gemm_f32(&a, &b, m, n, k, &mut out);
                best = best.min(t.elapsed().as_nanos() as f64);
            }
            let mt = g(best, m, n, k);
            gemm_f32_single(&a, &b, m, n, k, &mut out);
            let mut best1 = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                gemm_f32_single(&a, &b, m, n, k, &mut out);
                best1 = best1.min(t.elapsed().as_nanos() as f64);
            }
            println!(
                "  f32 {m}x{n}x{k}: fast {mt:.1} GF/s | fast单 {:.1} GF/s | 单/多 {:.2}",
                g(best1, m, n, k),
                g(best1, m, n, k) / mt
            );
        }
    }
}

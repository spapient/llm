//! 高吞吐 CPU GEMM：多线程 + A 列面板打包 + NEON 8×8 寄存器分块。
//! 语义同 [`crate::matmul::matmul_scalar`]，浮点累加顺序不同，按 IEEE 容差对拍（不做逐位相等）。
//!
//! 关键在「打包」：A 行主序下按列取 8 行会跨 8 条 cache line、每条只用 4 字节，微内核会被访存打穿。
//! 先把每个 8 行块转置成 `[p][8]` 面板，微内核每条 `p` 只做两次**连续**向量加载，再用 `lane` 广播
//! 到 8 个累加器；`unsafe` 只在 aarch64 的 NEON 微内核这一小段（`#[allow(unsafe_code)]`）。

use std::thread;

/// 微内核行数。
const MR: usize = 8;

/// 微内核列数（两个 NEON 向量）。
const NR: usize = 8;

/// 多线程稠密 f32 GEMM：`out[m,n] = a[m,k] · b[k,n]`。
///
/// # Panics
///
/// 三条缓冲长度不足 `m·k` / `k·n` / `m·n` 时 panic（调用方先校验）。
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    assert!(
        a.len() >= m * k && b.len() >= k * n && out.len() >= m * n,
        "gemm_f32 尺寸不符"
    );
    let cpu = thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let t = cpu.min(m.div_ceil(MR)).max(1);
    let rows_per = m.div_ceil(t);
    thread::scope(|s| {
        for (ti, block) in out[..m * n].chunks_mut(rows_per * n).enumerate() {
            let r0 = ti * rows_per;
            let rows = block.len() / n;
            let a_t = &a[r0 * k..(r0 + rows) * k];
            s.spawn(move || compute(a_t, b, rows, n, k, block));
        }
    });
}

/// 单线程稠密 f32 GEMM（基准确认内核质量用）：`out[m,n] = a[m,k] · b[k,n]`。
pub fn gemm_f32_single(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    compute(a, b, m, n, k, out);
}

/// 单线程块计算：A 打包成 8 行面板，再 j 外层、面板内层跑微内核；残尾走标量。
fn compute(a: &[f32], b: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
    let full = rows / MR;
    let apack = pack_panels(a, full, k);
    sweep_columns(a, &apack, b, out, full, n, k);
    let tail = full * MR;
    if tail < rows {
        edge(a, b, out, tail, 0, rows - tail, n, n, k);
    }
}

/// A 的每个 8 行块转置成 `[p][8]` 面板：微内核每条 `p` 只需两次连续向量加载。
fn pack_panels(a: &[f32], full: usize, k: usize) -> Vec<f32> {
    let mut apack = vec![0.0f32; full * k * MR];
    for blk in 0..full {
        let src = &a[blk * MR * k..(blk * MR + MR) * k];
        let dst = &mut apack[blk * k * MR..blk * k * MR + k * MR];
        for p in 0..k {
            for r in 0..MR {
                dst[p * MR + r] = src[r * k + p];
            }
        }
    }
    apack
}

/// 沿列块扫描整行块：整块走微内核，最后一段残列走标量边（`edge` 读未打包的 `a`）。
fn sweep_columns(
    a: &[f32],
    apack: &[f32],
    b: &[f32],
    out: &mut [f32],
    full: usize,
    n: usize,
    k: usize,
) {
    let mut j = 0;
    while j < n {
        let nr = NR.min(n - j);
        if nr == NR {
            for blk in 0..full {
                micro_packed(&apack[blk * k * MR..], b, out, blk * MR, j, n, k);
            }
        } else {
            edge(a, b, out, 0, j, full * MR, nr, n, k);
        }
        j += NR;
    }
}

/// 打包面板上的 NEON 8×8 微内核（A 面板布局 `[p][8]`，一次算 8 行 8 列）。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn micro_packed(
    panel: &[f32],
    b: &[f32],
    out: &mut [f32],
    row0: usize,
    j: usize,
    n: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [zero; 16];
        let pp = panel.as_ptr();
        let bp = b.as_ptr();
        for p in 0..k {
            let av0 = vld1q_f32(pp.add(p * MR));
            let av1 = vld1q_f32(pp.add(p * MR + 4));
            let bv0 = vld1q_f32(bp.add(p * n + j));
            let bv1 = vld1q_f32(bp.add(p * n + j + 4));
            let s0 = vdupq_laneq_f32::<0>(av0);
            acc[0] = vfmaq_f32(acc[0], s0, bv0);
            acc[1] = vfmaq_f32(acc[1], s0, bv1);
            let s1 = vdupq_laneq_f32::<1>(av0);
            acc[2] = vfmaq_f32(acc[2], s1, bv0);
            acc[3] = vfmaq_f32(acc[3], s1, bv1);
            let s2 = vdupq_laneq_f32::<2>(av0);
            acc[4] = vfmaq_f32(acc[4], s2, bv0);
            acc[5] = vfmaq_f32(acc[5], s2, bv1);
            let s3 = vdupq_laneq_f32::<3>(av0);
            acc[6] = vfmaq_f32(acc[6], s3, bv0);
            acc[7] = vfmaq_f32(acc[7], s3, bv1);
            let s4 = vdupq_laneq_f32::<0>(av1);
            acc[8] = vfmaq_f32(acc[8], s4, bv0);
            acc[9] = vfmaq_f32(acc[9], s4, bv1);
            let s5 = vdupq_laneq_f32::<1>(av1);
            acc[10] = vfmaq_f32(acc[10], s5, bv0);
            acc[11] = vfmaq_f32(acc[11], s5, bv1);
            let s6 = vdupq_laneq_f32::<2>(av1);
            acc[12] = vfmaq_f32(acc[12], s6, bv0);
            acc[13] = vfmaq_f32(acc[13], s6, bv1);
            let s7 = vdupq_laneq_f32::<3>(av1);
            acc[14] = vfmaq_f32(acc[14], s7, bv0);
            acc[15] = vfmaq_f32(acc[15], s7, bv1);
        }
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

#[cfg(not(target_arch = "aarch64"))]
fn micro_packed(
    panel: &[f32],
    b: &[f32],
    out: &mut [f32],
    row0: usize,
    j: usize,
    n: usize,
    k: usize,
) {
    for r in 0..MR {
        for c in 0..NR {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += panel[p * MR + r] * b[p * n + j + c];
            }
            out[(row0 + r) * n + j + c] = acc;
        }
    }
}

/// 标量尾块：`out[i..i+mr, j..j+nr]` 逐元素点积（与微内核同一组累加口径，容差内一致）。
fn edge(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    i: usize,
    j: usize,
    mr: usize,
    nr: usize,
    n: usize,
    k: usize,
) {
    for r in 0..mr {
        for c in 0..nr {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[(i + r) * k + p] * b[p * n + j + c];
            }
            out[(i + r) * n + j + c] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// i8 sdot 通道：MR=2 × NR=8，B 转置打包成「每列 k 连续」。
// 每个 k16 一次 `vld1q_s8` A 两行 + 8 次 B 加载 + 16 次 `vdotq_s32`。
// dotprod 运行时探测，缺失走标量回退；`unsafe` 仅限 aarch64 微内核。
// ---------------------------------------------------------------------------

/// i8 微内核行数（两行共享同一 B 向量，摊薄 B 加载）。
const I8_MR: usize = 2;

/// i8 微内核列数（8 个 int32x4 累加器）。
const I8_NR: usize = 8;

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
    let t = cpu.min(m.div_ceil(I8_MR)).max(1);
    let mut rows_per = m.div_ceil(t);
    // 每线程偶数行，保证两行一对不被切碎（缺则末线程落单走标量行）。
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
    let (bpack, npad) = pack_b(b, n, k);
    compute_i8_packed(a, &bpack, npad, m, n, k, out);
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
            s.spawn(move || {
                let blocks = chunk.len() / (I8_NR * k);
                for bi in 0..blocks {
                    let blk = blk0 + bi;
                    for nb in 0..I8_NR {
                        let jj = blk * I8_NR + nb;
                        if jj < n {
                            let dst =
                                &mut chunk[bi * I8_NR * k + nb * k..bi * I8_NR * k + (nb + 1) * k];
                            let mut p = 0;
                            while p < k {
                                dst[p] = b[p * n + jj];
                                p += 1;
                            }
                        }
                    }
                }
            });
        }
    });
    (bpack, npad)
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
    let mut blk = 0;
    while blk < npad {
        let mut r = 0;
        while r + I8_MR <= rows {
            pair_one(a, bpack, out, r, blk, n, k, kvec);
            r += I8_MR;
        }
        while r < rows {
            scalar_row_group(a, bpack, out, r, blk, n, k);
            r += 1;
        }
        blk += I8_NR;
    }
}

/// 单个 8 列块、两行微内核分派：dotprod 可用且 k≥16 走向量，否则标量两行。
#[allow(unsafe_code)]
fn pair_one(
    a: &[i8],
    bpack: &[i8],
    out: &mut [i32],
    r: usize,
    blk: usize,
    n: usize,
    k: usize,
    kvec: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if kvec >= 16 && std::arch::is_aarch64_feature_detected!("dotprod") {
            // 安全性：dotprod 已运行时探测。
            unsafe { dot_one(a, bpack, out, r, blk, n, k, kvec) };
            return;
        }
    }
    scalar_row_group(a, bpack, out, r, blk, n, k);
    scalar_row_group(a, bpack, out, r + 1, blk, n, k);
}

/// aarch64 dotprod：一个 8 列块的两行（`unsafe` 岛）。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
unsafe fn dot_one(
    a: &[i8],
    bpack: &[i8],
    out: &mut [i32],
    r: usize,
    blk: usize,
    n: usize,
    k: usize,
    kvec: usize,
) {
    unsafe {
        let ap = a.as_ptr().add(r * k);
        let bp = bpack.as_ptr();
        let accs = dot_block(ap, bp, blk, k, kvec);
        harvest(ap, bp, out, (r, n, k, kvec), blk, accs);
    }
}

/// 一个 8 列块的两行累加：`(两行共 16 条 lane 的整数累加和)`。
///
/// `blk` 是列块起点、`p` 沿 `kvec` 走 16 字节一拍；两条 `ave` 分别是第 `r` 行与第 `r+1` 行。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
unsafe fn dot_block(
    ap: *const i8,
    bp: *const i8,
    blk: usize,
    k: usize,
    kvec: usize,
) -> [[std::arch::aarch64::int32x4_t; 8]; 2] {
    use std::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_s32(0);
        let mut accs = [[zero; 8]; 2];
        let mut p = 0;
        while p < kvec {
            let av0 = vld1q_s8(ap.add(p));
            let av1 = vld1q_s8(ap.add(k + p));
            for nb in 0..8 {
                let bv = vld1q_s8(bp.add((blk + nb) * k + p));
                accs[0][nb] = vdotq_s32(accs[0][nb], av0, bv);
                accs[1][nb] = vdotq_s32(accs[1][nb], av1, bv);
            }
            p += 16;
        }
        accs
    }
}

/// 一个 8 列块的收口：水平求和 + 标量补 `kvec..k` 的余数，写回两行。
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
    accs: [[std::arch::aarch64::int32x4_t; 8]; 2],
) {
    use std::arch::aarch64::*;
    unsafe {
        for nb in 0..8 {
            let jj = blk + nb;
            if jj < n {
                let mut s0 = vaddvq_s32(accs[0][nb]);
                let mut s1 = vaddvq_s32(accs[1][nb]);
                let mut q = kvec;
                while q < k {
                    let w = *bp.add(jj * k + q) as i32;
                    s0 += *ap.add(q) as i32 * w;
                    s1 += *ap.add(k + q) as i32 * w;
                    q += 1;
                }
                out[r * n + jj] = s0;
                out[(r + 1) * n + jj] = s1;
            }
        }
    }
}

/// 标量单行、单 8 列块（打包 B 取数，任意 n/k）。
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
    for nb in 0..8 {
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

    fn sample(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x >> 8) as f32) / 256.0 - 32.0
            })
            .collect()
    }

    fn check(m: usize, n: usize, k: usize) {
        let a = sample(m * k, 1);
        let b = sample(k * n, 2);
        let mut want = vec![0.0f32; m * n];
        matmul_scalar(&a, &b, m, n, k, &mut want);
        let mut got = vec![0.0f32; m * n];
        gemm_f32(&a, &b, m, n, k, &mut got);
        close_enough(&got, &want, 1e-3, 1e-3).expect("fast gemm 与标量参考容差内一致");
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
                "  i8 {m}x{n}x{k}: fast {mt:.1} GF/s | fast单 {:.1} GF/s | 扩展 {:.1}x",
                g(best1, m, n, k),
                g(best1, m, n, k) / mt
            );
        }
    }
}

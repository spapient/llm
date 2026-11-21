use super::{I8_MR, I8_NR};

/// 单线程块计算（B 面板已备）：**N 分组在最外层**，每组 8×k 的 B 瓦片常驻 L1，
/// 被本线程所有行对复用——否则每行对都把整块 B 重读一遍，卡在内存带宽。
pub(super) fn compute_i8_packed(
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
pub(super) struct Gram {
    r: usize,
    blk: usize,
    n: usize,
    k: usize,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    kvec: usize,
}

/// 单个 8 列块、两行微内核分派：dotprod 可用且 k≥16 走向量，否则标量两行。
#[allow(unsafe_code)]
pub(super) fn pair_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
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
pub(super) fn scalar_row_group(
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

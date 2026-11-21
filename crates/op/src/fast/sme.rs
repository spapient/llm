//! Apple SME 16×64 微内核通道：ZA 矩阵累加器外积（`fmopa`）+ 运行时探测。
//!
//! 与 [`super`] 里的 NEON 8×8 通道并列，只服务「16 行 × 64 列整铺」的大 GEMM。SME 的收益来自
//! 16×16 外积在 ZA 里就地累加、K 方向无需回写（四个瓦片交错隐藏 `fmopa` 延迟）；但 `smstart` /
//! `smstop` 与 A/B 面板打包的固定成本在小题上会吞掉收益，因此入口 [`gemm`] 里有规模门槛。
//!
//! `unsafe` 只出现在三段内联汇编（`enter_sm` / `sme_tile` / `exit_sm`），全部 `#[allow(unsafe_code)]`
//! 且由 `available` 的运行时探测保证只在 SME 存在时进入；非 aarch64 上整条内核不参与编译。

use std::thread;

#[cfg(target_arch = "aarch64")]
use std::arch::asm;

/// SME 微内核行数：一个 16×16 ZA 瓦片覆盖 16 行。
const MR: usize = 16;

/// SME 微内核列数：4 个 ZA 瓦片各 16 列，一次写回 64 列。
const NR: usize = 64;

/// 规模门槛（元素数）：低于此值的 GEMM，`smstart`/`smstop` 与打包的固定成本超过外积收益，
/// 交给 NEON 更划算。
const MIN_ELEMS: usize = 1 << 20;

/// 运行时探测：本机是否存在 SME（Apple M4/M5 起）。
#[cfg(target_arch = "aarch64")]
fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("sme")
}

#[cfg(not(target_arch = "aarch64"))]
fn available() -> bool {
    false
}

/// 16×64 整铺 SME GEMM：`out[m,n] = a[m,k] · b[k,n]`。
///
/// 形状不可整铺、规模低于门槛或本机无 SME 时返回 `false`，由调用方走 NEON；成功写入 `out`
/// 后返回 `true`。整铺形状下的数值与 [`crate::matmul::matmul_scalar`] 在容差内一致。
pub(super) fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) -> bool {
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (a, b, m, n, k, out);
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if m == 0
            || n == 0
            || k == 0
            || !m.is_multiple_of(MR)
            || !n.is_multiple_of(NR)
            || m * n * k < MIN_ELEMS
            || !available()
        {
            return false;
        }
        let t = super::threads_override().min(m / MR).min(n / NR).max(1);
        packed(a, b, m, n, k, out, t);
        true
    }
}

/// 一次整铺 SME GEMM：先并行打包 A/B 面板，再并行外积累加写回。
#[cfg(target_arch = "aarch64")]
fn packed(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32], t: usize) {
    let mblk = m / MR;
    let nblk = n / NR;
    let alen = mblk * k * MR;
    let blen = nblk * k * NR;
    super::scratch::with_panels(alen + blen, |buf| {
        let (abuf, bbuf) = buf.split_at_mut(alen);

        // 打包：A 按行块、B 按列组切给 t 个线程，一次 scope 里并行做完。
        thread::scope(|s| {
            let mut a_rest = &mut *abuf;
            let mut b_rest = &mut *bbuf;
            for slot in 0..t {
                let (a0, a1) = (slot * mblk / t, (slot + 1) * mblk / t);
                let (b0, b1) = (slot * nblk / t, (slot + 1) * nblk / t);
                let (a_chunk, a_tail) = a_rest.split_at_mut((a1 - a0) * k * MR);
                let (b_chunk, b_tail) = b_rest.split_at_mut((b1 - b0) * k * NR);
                a_rest = a_tail;
                b_rest = b_tail;
                s.spawn(move || {
                    pack_a(a, a_chunk, k, a0);
                    pack_b(b, b_chunk, n, k, b0);
                });
            }
        });

        // 计算：每个线程进入流式模式，算自己的行块组，退出前写回。
        let abuf: &[f32] = abuf;
        let bbuf: &[f32] = bbuf;
        thread::scope(|s| {
            let mut o_rest = &mut out[..];
            for slot in 0..t {
                let m0 = slot * mblk / t;
                let m1 = (slot + 1) * mblk / t;
                let (ochunk, o_tail) = o_rest.split_at_mut((m1 - m0) * MR * n);
                o_rest = o_tail;
                s.spawn(move || compute(abuf, bbuf, ochunk, n, k, m0));
            }
        });
    });
}

/// 打包 A 的一个行块段：`src[m,k]` → `dst[块][k][16]`，按 16×16 瓦片转置让微内核连续加载。
///
/// 逐列标量搬移在 16 行 × 4 KiB 跨距下受载入延迟拖累；改用 [`std::simd::f32x4`] 做 4×4
/// 子块转置（编译器直接下沉成 `zip1`/`zip2`），配合 `copy_from_slice` 的 128 位连续读写，
/// 全程安全代码，不触碰 `unsafe_code = "deny"` 的开岛边界。
#[cfg(target_arch = "aarch64")]
fn pack_a(src: &[f32], dst: &mut [f32], k: usize, mblk0: usize) {
    use std::simd::{f32x4, simd_swizzle};

    #[inline(always)]
    fn transpose4(a0: f32x4, a1: f32x4, a2: f32x4, a3: f32x4) -> (f32x4, f32x4, f32x4, f32x4) {
        let t0 = simd_swizzle!(a0, a1, [0, 4, 1, 5]);
        let t1 = simd_swizzle!(a0, a1, [2, 6, 3, 7]);
        let t2 = simd_swizzle!(a2, a3, [0, 4, 1, 5]);
        let t3 = simd_swizzle!(a2, a3, [2, 6, 3, 7]);
        (
            simd_swizzle!(t0, t2, [0, 1, 4, 5]),
            simd_swizzle!(t0, t2, [2, 3, 6, 7]),
            simd_swizzle!(t1, t3, [0, 1, 4, 5]),
            simd_swizzle!(t1, t3, [2, 3, 6, 7]),
        )
    }

    let blocks = dst.len() / (k * MR);
    for bi in 0..blocks {
        let rowbase = (mblk0 + bi) * MR * k;
        let dstbase = bi * k * MR;
        let mut kk0 = 0usize;
        while kk0 + 16 <= k {
            for r in 0..4usize {
                for c in 0..4usize {
                    let rr = r * 4;
                    let cc = kk0 + c * 4;
                    let a0 = f32x4::from_slice(&src[rowbase + rr * k + cc..][..4]);
                    let a1 = f32x4::from_slice(&src[rowbase + (rr + 1) * k + cc..][..4]);
                    let a2 = f32x4::from_slice(&src[rowbase + (rr + 2) * k + cc..][..4]);
                    let a3 = f32x4::from_slice(&src[rowbase + (rr + 3) * k + cc..][..4]);
                    let (b0, b1, b2, b3) = transpose4(a0, a1, a2, a3);
                    let orow = kk0 + c * 4;
                    dst[dstbase + orow * MR + rr..][..4].copy_from_slice(&b0.to_array());
                    dst[dstbase + (orow + 1) * MR + rr..][..4].copy_from_slice(&b1.to_array());
                    dst[dstbase + (orow + 2) * MR + rr..][..4].copy_from_slice(&b2.to_array());
                    dst[dstbase + (orow + 3) * MR + rr..][..4].copy_from_slice(&b3.to_array());
                }
            }
            kk0 += 16;
        }
        while kk0 < k {
            for r in 0..MR {
                dst[dstbase + kk0 * MR + r] = src[rowbase + r * k + kk0];
            }
            kk0 += 1;
        }
    }
}

/// 打包 B 的一个列组段：`src[k,n]` → `dst[组][k][64]`，按 k 逐片拷贝让微内核连续加载。
#[cfg(target_arch = "aarch64")]
fn pack_b(src: &[f32], dst: &mut [f32], n: usize, k: usize, nblk0: usize) {
    let blocks = dst.len() / (k * NR);
    for bi in 0..blocks {
        let g = nblk0 + bi;
        let base = bi * k * NR;
        for kk in 0..k {
            let off = base + kk * NR;
            let row = kk * n + g * NR;
            dst[off..off + NR].copy_from_slice(&src[row..row + NR]);
        }
    }
}

/// 计算一个行块段：进入流式模式 → 逐块外积累加 → 写回 → 退出流式模式。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn compute(abuf: &[f32], bbuf: &[f32], out: &mut [f32], n: usize, k: usize, mblk0: usize) {
    let nblk = bbuf.len() / (k * NR);
    let blocks = out.len() / (n * MR);
    unsafe {
        enter_sm();
        for bi in 0..blocks {
            let a_p = abuf.as_ptr().add((mblk0 + bi) * k * MR);
            let c_base = out.as_mut_ptr().add(bi * MR * n);
            for g in 0..nblk {
                let b_p = bbuf.as_ptr().add(g * k * NR);
                sme_tile(k, a_p, b_p, c_base.add(g * NR), n * 4);
            }
        }
        exit_sm();
    }
}

/// 16×16 外积微内核：`zero` → K 循环 `ld1w`+`fmopa` → 4 个 ZA 瓦片按切片号 `str za` 写回。
///
/// `str za[wX, #imm]` 的切片由 `wX` 的值选定（imm=0 时 `slice = (v & 3) << 4 | v >> 2`），
/// 寄存器名是惰性的：`w12..w15` 初始 0..3，每行写回后 `+4`，16 行后到达 64 覆盖全部切片。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sme")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn sme_tile(k: usize, a_p: *const f32, b_p: *const f32, c: *mut f32, n_stride: usize) {
    unsafe {
        asm!(
        "zero {{ za0.s }}",
        "zero {{ za1.s }}",
        "zero {{ za2.s }}",
        "zero {{ za3.s }}",
        "mov {a}, {a_in}",
        "mov {b}, {b_in}",
        "mov {k}, {kcnt}",
        "2:",
        "ld1w {{ z0.s }}, p0/z, [{a}]",
        "add {a}, {a}, #64",
        "ld1w {{ z1.s }}, p0/z, [{b}]",
        "add {b}, {b}, #64",
        "ld1w {{ z2.s }}, p0/z, [{b}]",
        "add {b}, {b}, #64",
        "ld1w {{ z3.s }}, p0/z, [{b}]",
        "add {b}, {b}, #64",
        "ld1w {{ z4.s }}, p0/z, [{b}]",
        "add {b}, {b}, #64",
        "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
        "fmopa za1.s, p0/m, p0/m, z0.s, z2.s",
        "fmopa za2.s, p0/m, p0/m, z0.s, z3.s",
        "fmopa za3.s, p0/m, p0/m, z0.s, z4.s",
        "subs {k}, {k}, #1",
        "b.ne 2b",
        "mov {c16}, {c}",
        "add {c16}, {c16}, #64",
        "mov {c32}, {c}",
        "add {c32}, {c32}, #128",
        "mov {c48}, {c}",
        "add {c48}, {c48}, #192",
        "mov w12, #0",
        "mov w13, #1",
        "mov w14, #2",
        "mov w15, #3",
        "3:",
        "str za[w12, 0], [{c}]",
        "str za[w13, 0], [{c16}]",
        "str za[w14, 0], [{c32}]",
        "str za[w15, 0], [{c48}]",
        "add {c}, {c}, {nstride}",
        "add {c16}, {c16}, {nstride}",
        "add {c32}, {c32}, {nstride}",
        "add {c48}, {c48}, {nstride}",
        "add w12, w12, #4",
        "add w13, w13, #4",
        "add w14, w14, #4",
        "add w15, w15, #4",
        "cmp w12, #64",
        "b.ne 3b",
        a = inout(reg) a_p => _,
        b = inout(reg) b_p => _,
        c = inout(reg) c => _,
        a_in = in(reg) a_p,
        b_in = in(reg) b_p,
        k = out(reg) _,
        kcnt = in(reg) k,
        nstride = in(reg) n_stride,
        c16 = out(reg) _, c32 = out(reg) _, c48 = out(reg) _,
        out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _,
        out("w12") _, out("w13") _, out("w14") _, out("w15") _,
        out("p0") _,
        options(nostack)
        );
    }
}

/// 进入流式模式并准备全真谓词 `p0`；必须在任何 SME 指令前调用，且每线程一次。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sme")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn enter_sm() {
    unsafe {
        asm!("smstart", "smstart za", "ptrue p0.s", options(nostack));
    }
}

/// 退出流式模式；与 [`enter_sm`] 成对，`sme_tile` 之间不得夹带 NEON。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sme")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn exit_sm() {
    unsafe {
        asm!("smstop za", "smstop", options(nostack));
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::gemm;
    use crate::matmul::{close_enough, matmul_scalar};

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

    /// 直接调用 SME 内核（越过 `gemm_f32` 的分派），用刚过规模门槛的小形状覆盖：
    /// 多行块 × 多列组、单列组、单行块三种铺法。
    fn check(m: usize, n: usize, k: usize) {
        let a = sample(m * k, 1);
        let b = sample(k * n, 2);
        let mut want = vec![0.0f32; m * n];
        matmul_scalar(&a, &b, m, n, k, &mut want);
        let mut got = vec![0.0f32; m * n];
        assert!(gemm(&a, &b, m, n, k, &mut got), "SME 应接住整铺形状");
        close_enough(&got, &want, 1e-3, 1e-3).expect("SME 与标量参考容差内一致");
    }

    #[test]
    fn sme_matches_on_tile_multiples() {
        check(128, 128, 128);
    }

    #[test]
    fn sme_matches_on_single_col_group() {
        check(128, 64, 128);
    }

    #[test]
    fn sme_matches_on_single_row_block() {
        check(16, 64, 1024);
    }
}

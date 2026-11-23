//! Apple SME 16×64 微内核通道：ZA 矩阵累加器外积（`fmopa`）+ 运行时探测。
//!
//! 瓦片形状是**量过**的，不是推出来的：2×2（32×32）每 4 条 `fmopa` 只摊 4 条 `ld1w`，
//! 按载入条数该比 1×4 快 7.6%，但实测 1 线程 1256.7 vs 1543.4 Gelem/s、16 线程 2063.7 vs
//! 2271.8（`sme_peak` 口径），**反而慢 19% / 9%**。载入条数因此不是约束项，
//! 2×2 的代价（A 面板按 32 宽写、每行分 8 次 16 B 散写）超过了它省下的那一条 `ld1w`。
//! 结论：1×4 保留，`port_mix` 探针留作这条否证的仪器。
//!
//! **本模块在仓里钉住的 stable 工具链上不编译**（`native/mod.rs` 以
//! `#[cfg(all(nightly, target_arch = "aarch64"))]` 门控，`sme` 的特征探测与 `target_feature`
//! 仍都是 `E0658`）。以上数字只说明 nightly 构建里两种瓦片形状的相对优劣，**不是出货 K7**；
//! 出货 K7 的分母是 `fma_peak`（见 `docs/math.md`）。

use std::thread;

use std::arch::asm;

use std::simd::{f32x4, simd_swizzle};

/// SME 微内核行数：一个 16×16 ZA 瓦片覆盖 16 行。
pub const MR: usize = 16;

/// SME 微内核列数：4 个 ZA 瓦片各 16 列，一次写回 64 列。
pub const NR: usize = 64;

/// 规模门槛（元素数）：低于此值的 GEMM，`smstart`/`smstop` 与打包的固定成本超过外积收益，
/// 交给 NEON 更划算。
const MIN_ELEMS: usize = 1 << 20;

/// 运行时探测：本机是否存在 SME（Apple M4/M5 起）。
fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("sme")
}

/// 16×64 整铺 SME GEMM：`out[m,n] = a[m,k] · b[k,n]`。
pub fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) -> bool {
    if rejects(m, n, k) {
        return false;
    }
    packed(a, b, m, n, k, out, threads_for(m, n));
    true
}

/// 线程数：行块（`m/MR`）与列组（`n/NR`）同时决定上限，两者都不能切出空槽。
pub fn threads_for(m: usize, n: usize) -> usize {
    super::threads_override().min(m / MR).min(n / NR).max(1)
}

/// 拒绝走 SME 的形状：零维、非整铺、低于规模门槛，或本机无 SME。
fn rejects(m: usize, n: usize, k: usize) -> bool {
    degenerate(m, n, k) || !tiled(m, n) || m * n * k < MIN_ELEMS || !available()
}

/// 任一维为零。
fn degenerate(m: usize, n: usize, k: usize) -> bool {
    m == 0 || n == 0 || k == 0
}

/// 行/列是否满足整铺（`MR` 行 × `NR` 列的倍数）。
fn tiled(m: usize, n: usize) -> bool {
    m.is_multiple_of(MR) && n.is_multiple_of(NR)
}

/// 一次整铺 SME GEMM：单次铺线程，先并行打包 A 面板，屏障同步后并行外积累加写回。
fn packed(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32], t: usize) {
    let mblk = m / MR;
    let alen = mblk * k * MR;
    super::with_panels(alen, |abuf| {
        let barrier = std::sync::Barrier::new(t);
        thread::scope(|s| {
            let barrier = &barrier;
            let mut a_rest = &mut *abuf;
            let mut o_rest = &mut *out;
            for slot in 0..t {
                let (r0, r1) = (slot * mblk / t, (slot + 1) * mblk / t);
                let (a_chunk, a_tail) = a_rest.split_at_mut((r1 - r0) * k * MR);
                let (o_chunk, o_tail) = o_rest.split_at_mut((r1 - r0) * MR * n);
                a_rest = a_tail;
                o_rest = o_tail;
                s.spawn(move || {
                    pack_a(a, a_chunk, k, r0);
                    barrier.wait();
                    compute_stage(&*a_chunk, b, o_chunk, n, k);
                });
            }
        });
    });
}

/// 打包 A 的一个行块段：`src[m,k]` → `dst[块][k][16]`，按 16×16 瓦片转置让微内核连续加载。
fn pack_a(src: &[f32], dst: &mut [f32], k: usize, mblk0: usize) {
    let blocks = dst.len() / (k * MR);
    for bi in 0..blocks {
        let rowbase = (mblk0 + bi) * MR * k;
        let dstbase = bi * k * MR;
        let kk0 = pack_tiles(src, dst, k, rowbase, dstbase);
        pack_tail(src, dst, k, rowbase, dstbase, kk0);
    }
}

/// 4×4 子块转置：两轮 `zip1`/`zip2` 交错，供 [`pack_tiles`] 连续写出。
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

/// 16 对齐的整段：按 16×16 瓦片做 4×4 转置，返回尾部起点。
fn pack_tiles(src: &[f32], dst: &mut [f32], k: usize, rowbase: usize, dstbase: usize) -> usize {
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
    kk0
}

/// 尾段（不足 16 列）：逐元素标量搬移。
fn pack_tail(
    src: &[f32],
    dst: &mut [f32],
    k: usize,
    rowbase: usize,
    dstbase: usize,
    mut kk0: usize,
) {
    while kk0 < k {
        for r in 0..MR {
            dst[dstbase + kk0 * MR + r] = src[rowbase + r * k + kk0];
        }
        kk0 += 1;
    }
}

/// 计算一段连续行块：进入流式模式 → 逐行块逐列组外积累加 → 写回 → 退出流式模式。
#[allow(unsafe_code)]
fn compute_stage(a_stage: &[f32], b: &[f32], o_stage: &mut [f32], n: usize, k: usize) {
    let nblk = n / NR;
    let blocks = a_stage.len() / (k * MR);
    if blocks == 0 {
        return;
    }
    unsafe {
        enter_sm();
        for bi in 0..blocks {
            let a_p = a_stage.as_ptr().add(bi * k * MR);
            let c_base = o_stage.as_mut_ptr().add(bi * MR * n);
            for g in 0..nblk {
                let b_p = b.as_ptr().add(g * NR);
                sme_tile(k, a_p, b_p, c_base.add(g * NR), n * 4, n * 4);
            }
        }
        exit_sm();
    }
}

/// 16×16 外积微内核：`zero` → K 循环 `ld1w`+`fmopa` → 4 个 ZA 瓦片按切片号 `str za` 写回。
#[target_feature(enable = "sme")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn sme_tile(
    k: usize,
    a_p: *const f32,
    b_p: *const f32,
    c: *mut f32,
    n_stride: usize,
    b_stride: usize,
) {
    let b_adv = b_stride - 3 * 64;
    unsafe {
        asm!(
        "zero {{ za0.s }}",
        "zero {{ za1.s }}",
        "zero {{ za2.s }}",
        "zero {{ za3.s }}",
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
        "add {b}, {b}, {b_adv}",
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
        a = inout(reg) a_p => _, b = inout(reg) b_p => _, c = inout(reg) c => _,
        k = inout(reg) k => _,
        nstride = in(reg) n_stride, b_adv = in(reg) b_adv,
        c16 = out(reg) _, c32 = out(reg) _, c48 = out(reg) _,
        out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _,
        out("w12") _, out("w13") _, out("w14") _, out("w15") _, out("p0") _,
        options(nostack),
        );
    }
}

/// 进入流式模式并准备全真谓词 `p0`；必须在任何 SME 指令前调用，且每线程一次。
#[target_feature(enable = "sme")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn enter_sm() {
    unsafe {
        asm!("smstart", "smstart za", "ptrue p0.s", options(nostack));
    }
}

/// 退出流式模式；与 [`enter_sm`] 成对，`sme_tile` 之间不得夹带 NEON。
#[target_feature(enable = "sme")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn exit_sm() {
    unsafe {
        asm!("smstop za", "smstop", options(nostack));
    }
}

/// SME 端口峰值探针：`rounds` 轮、每轮 4 条 `fmopa`，`z0`/`z1` 与 ZA 全部寄存器常驻，
/// **内层零 load、零 store**——把「数据从哪来」整个删掉，剩下的才是这块单元每秒能吃多少外积。
///
/// 每轮 4 × 16 × 16 = 1024 次乘加（`p0.s` 16 条泳道，`ZA.s` 瓦片 16×16），即 2048 FLOP；
/// 瓦片尺寸与 [`MR`]/[`NR`] 同源，由 `nn` 的 SME 整铺用例钉死。
///
/// 这是把 GF/s 换算成「端口峰值百分比」的分母：GEMM 每轮还要搬 320 B，搬不动与算不动是两件事，
/// 不分开量就只能猜瓶颈（F18）。
///
/// 本机无 SME 或 `rounds == 0` 时返回 `false` 且不执行任何 SME 指令——安全入口，调用方据此跳过。
/// 只在 `bench` 特性下编译，生产公开面不含本项。
#[cfg(feature = "bench")]
pub fn port_run(rounds: usize) -> bool {
    if rounds == 0 || !available() {
        return false;
    }
    #[allow(unsafe_code)]
    unsafe {
        port_rounds(rounds);
    }
    true
}

/// 纯外积循环：`smstart` → `rounds` 轮 4 条 `fmopa` → `smstop`，不碰任何内存。
#[cfg(feature = "bench")]
#[target_feature(enable = "sme")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn port_rounds(rounds: usize) {
    unsafe {
        asm!(
            "smstart",
            "smstart za",
            "ptrue p0.s",
            "fmov z0.s, #1.0",
            "fmov z1.s, #1.0",
            "zero {{ za0.s }}",
            "zero {{ za1.s }}",
            "zero {{ za2.s }}",
            "zero {{ za3.s }}",
            "2:",
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
            "fmopa za1.s, p0/m, p0/m, z0.s, z1.s",
            "fmopa za2.s, p0/m, p0/m, z0.s, z1.s",
            "fmopa za3.s, p0/m, p0/m, z0.s, z1.s",
            "subs {k}, {k}, #1",
            "b.ne 2b",
            "smstop za",
            "smstop",
            k = inout(reg) rounds => _,
            out("z0") _, out("z1") _, out("p0") _,
            options(nostack),
        );
    }
}

/// 取数混合探针：每轮 **1 条 `a` 载入 + 4 条 `b` 载入 + 4 条 `fmopa`**，与 [`sme_tile`] 内层
/// 的指令配比逐条同构（16×64 瓦片即 `a` 16 个 f32 = 64 B、`b` 64 个 f32 = 256 B）。
///
/// 与 [`port_run`] 的差就是「取数代价」：两者 `fmopa` 条数相同，只差这 5 条 load。
/// 地址**固定不推进**，即每一轮都命中同一批缓存行——这是取数端口的**上限**，真实内核走不同行
/// 只会更慢，所以据此算出的取数代价偏小、缺口偏小，拿它下结论不夸大（A3 保守值）。
///
/// 本机无 SME 或 `rounds == 0` 时返回 `false` 且不执行任何 SME 指令。只在 `bench` 特性下编译。
#[cfg(feature = "bench")]
pub fn port_mix(rounds: usize, a: &[f32; 16], b: &[f32; 64]) -> bool {
    if rounds == 0 || !available() {
        return false;
    }
    #[allow(unsafe_code)]
    unsafe {
        mix_rounds(rounds, a.as_ptr(), b.as_ptr());
    }
    true
}

/// 固定地址的外积 + 载入循环：`smstart` → `rounds` 轮 → `smstop`，读边界由调用方的数组长度保证。
#[cfg(feature = "bench")]
#[target_feature(enable = "sme")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn mix_rounds(rounds: usize, a: *const f32, b: *const f32) {
    let (b1, b2, b3) = unsafe { (b.add(16), b.add(32), b.add(48)) };
    unsafe {
        asm!(
            "smstart",
            "smstart za",
            "ptrue p0.s",
            "zero {{ za0.s }}",
            "zero {{ za1.s }}",
            "zero {{ za2.s }}",
            "zero {{ za3.s }}",
            "2:",
            "ld1w {{ z0.s }}, p0/z, [{a}]",
            "ld1w {{ z1.s }}, p0/z, [{b}]",
            "ld1w {{ z2.s }}, p0/z, [{b1}]",
            "ld1w {{ z3.s }}, p0/z, [{b2}]",
            "ld1w {{ z4.s }}, p0/z, [{b3}]",
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
            "fmopa za1.s, p0/m, p0/m, z0.s, z2.s",
            "fmopa za2.s, p0/m, p0/m, z0.s, z3.s",
            "fmopa za3.s, p0/m, p0/m, z0.s, z4.s",
            "subs {k}, {k}, #1",
            "b.ne 2b",
            "smstop za",
            "smstop",
            a = in(reg) a, b = in(reg) b, b1 = in(reg) b1, b2 = in(reg) b2, b3 = in(reg) b3,
            k = inout(reg) rounds => _,
            out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("p0") _,
            options(nostack),
        );
    }
}

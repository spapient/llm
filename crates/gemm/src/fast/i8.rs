// i8 sdot 通道：MR=4 × NR=4，B 转置打包成「每列 k 连续」。
// 每个 k16 一次 4 行 A 加载 + 4 次 B 加载 + 16 次 `vdotq_s32`（256 MAC/24 指令），
// 且每块 B 瓦片被 4 行复用——B 访存比 MR=2/NR=8 再降一半。
// dotprod 运行时探测，缺失走标量回退；`unsafe` 仅限 aarch64 微内核。
// ---------------------------------------------------------------------------

/// i8 微内核行数（四行共享同一 B 向量，摊薄 B 加载）。
use std::thread;

use diagnostic::{Code, Error, Result};

use super::compute;
use super::pack::b_groups;
use super::pool::{Buffers, Job, Shape, pool, run_pooled};
use super::tn;

pub(super) const I8_MR: usize = 4;

/// i8 微内核列数（4 个 int32x4 累加器 × 4 行 = 16 寄存器，避开溢出）。
pub(super) const I8_NR: usize = 4;

/// 线程数：`LLM_THREADS` 环境变量优先，否则全核。
pub(super) fn threads_override() -> usize {
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
pub(super) fn i8_pack(j: &Job, slot: usize) {
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
pub(super) fn i8_calc(j: &Job, slot: usize) {
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
#[allow(unsafe_code)]
pub(super) fn f32_pack(j: &Job, slot: usize) {
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
pub(super) fn f32_calc(j: &Job, slot: usize) {
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
pub(super) fn single_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    let (bpack, npad) = pack_b(b, n, k);
    compute_i8_packed(a, &bpack, npad, m, n, k, out);
}

/// 兜底路径：池不可用（worker 起不来）时退回 scope 线程。
pub(super) fn scoped_i8(
    a: &[i8],
    b: &[i8],
    out: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
    t: usize,
) {
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
mod kernel {
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
}

mod pack {
    use super::super::*;
    use super::I8_NR;

    pub(super) fn pack_b(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
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
    pub(super) fn pack_b_par(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
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
    pub(super) fn pack_b_span(chunk: &mut [i8], b: &[i8], n: usize, k: usize, blk0: usize) {
        let blocks = chunk.len() / (I8_NR * k);
        for bi in 0..blocks {
            let tile = &mut chunk[bi * I8_NR * k..(bi + 1) * I8_NR * k];
            pack_b_tile(tile, b, n, k, (blk0 + bi) * I8_NR);
        }
    }

    /// 一组 8 列瓦片的转置写：`b`（行主、`n` 列）的第 `first..first+8` 列搬到 `dst`（列主块内偏移）。
    /// 越界的列（`n` 不是 8 的倍数）留 0——打包缓冲本来就按 `npad` 补齐，乘加时该列权重为 0。
    pub(super) fn pack_b_tile(dst: &mut [i8], b: &[i8], n: usize, k: usize, first: usize) {
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
}
use kernel::compute_i8_packed;
use pack::{pack_b, pack_b_par};

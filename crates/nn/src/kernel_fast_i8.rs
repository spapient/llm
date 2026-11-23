//! 职责：i8 sdot 微内核的**打包、扫描与调度骨架**。
// i8 sdot 通道：MR=4 × NR=4，B 转置打包成「每列 k 连续」。瓦片本体（aarch64 dotprod
// `sdot_tile` / `dot_one` / `harvest`）与端口探针已上提 `prof::native::i8`；本文件只留
// 面板打包、行块扫描、标量兜底与多线程分派，不持 arch/feature 门与汇编片段。
// ---------------------------------------------------------------------------

/// i8 微内核行数（四行共享同一 B 向量，摊薄 B 加载）。
use std::thread;

use diagnostic::{Code, Error, Result};

use super::align_up;
use super::compute_gemm;
use super::pack::{b_groups, group_panel};
use super::pool::{Bufs, Job, Shape, pool, run_pooled};
use super::threads_override;
use super::tn;

// i8 sdot 瓦片几何：单一事实源在 `prof::native::i8_geo`，重导出别名供本模块与经 `super::` 引用的用例。
pub(super) use prof::native::i8_geo::{MR as I8_MR, NR as I8_NR};

// 瓦片坐标与 dotprod 分派入口住在 prof 岛：探测、`target_feature` 与 `unsafe` 全收在岛内。
use prof::native::i8::{Gram, dot_block};

/// 行块高度：让 A 的 `rc × k` 字节留在 L1（约 128KiB），否则每个列块都要从 L2 重拉整块 A。
///
/// 「留住」这个假设已被实测证伪：`rc` 扫 112 / 64 / 32 与默认 128 在噪声内无差
/// （默认 309.1、366.7 Gelem/s；`rc=112` 304.4、346.4），见 `docs/lld.md` v3.0.40。
fn row_chunk(k: usize) -> usize {
    ((1usize << 17) / k.max(1)).clamp(I8_MR, 256)
}

/// 打包阶段：本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包（列主面板走泛型轮转体）。
#[allow(unsafe_code)]
pub(super) fn i8_pack(j: &Job, slot: usize) {
    unsafe {
        let b = std::slice::from_raw_parts(j.b as *const i8, j.k * j.n);
        let bp = std::slice::from_raw_parts_mut(j.buf as *mut i8, j.npad * j.k);
        group_panel::<i8, I8_NR, true>(b, bp, j.n, j.k, j.npad, j.p, slot, false);
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
            compute_gemm(a_l, bp, b, rows, j.n, j.k, out_l);
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
    let npad = align_up(n, I8_NR);
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
        Bufs {
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

mod kernel {
    use super::{Gram, I8_MR, I8_NR, dot_block};

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
        let rc = (super::row_chunk(k) / I8_MR) * I8_MR;
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

    /// 单个 4 列块、四行分派：dotprod 通道接住走向量，否则标量四行。
    pub(super) fn pair_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
        if dot_block(a, bpack, out, g) {
            return;
        }
        scalar_row_group(a, bpack, out, g.r, g.blk, g.n, g.k);
        for mr in 1..I8_MR {
            scalar_row_group(a, bpack, out, g.r + mr, g.blk, g.n, g.k);
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
    use super::super::pack::{pack_serial, pack_span};
    use super::I8_NR;
    use super::align_up;
    use super::thread;
    use super::threads_override;

    pub(super) fn pack_b(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
        let npad = align_up(n, I8_NR);
        let mut bpack = vec![0i8; npad * k];
        pack_serial::<i8, I8_NR, true>(&mut bpack, b, n, k, false);
        (bpack, npad)
    }

    /// 并行 B 打包：转置是跨 `n` 字节步长的 cache-miss 密集写，小 `n` 大 `k` 时串行打包
    /// 会独占整段耗时（如 256×256×4096 的 1MB 面板），这里按 4 列组切成线程。
    pub(super) fn pack_b_par(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
        let npad = align_up(n, I8_NR);
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
                s.spawn(move || pack_span::<i8, I8_NR, true>(chunk, b, n, k, blk0, false));
            }
        });
        (bpack, npad)
    }
}
use kernel::compute_i8_packed;
use pack::{pack_b, pack_b_par};

//! 高吞吐 CPU GEMM：多线程 + A 列面板打包 + 8×8 寄存器分块。
//! 语义同 [`crate::matmul::matmul_scalar`]，浮点累加顺序不同，按 IEEE 容差对拍（不做逐位相等）。

use std::thread;

use diagnostic::{Code, Error, Result};
use tensor::View;

#[path = "pack.rs"]
mod pack;
use pack::micro;
#[path = "pool.rs"]
mod pool;
// 面板暂存池、线程数、对齐助手都是设备无关的公共件，本体单点住在 `cutile::native`，
// 这里只重导出别名供本族子模块经 `super::` 引用；nn 不持有这些实现（B-1 收口）。
pub(super) use cutile::native::{align_up, threads_override, with_panels};

// SME 通道本体在 `cutile::native::sme`（`target_feature` / 特性探测门收在 prof），本模块只经
// 协商面 `gemm_negotiate` 调用，不写 arch/feature 门；其余情形走 NEON 8×8 通道。
use pack::{pack_b_f32_into, pack_b_f32_par_into, pack_panels_into};
use pool::{Bufs, Shape, pool, run_pooled};

/// 端口峰值探针只在测量容器里导出（`bench` 特性），生产构建看不到它；本体在 `cutile::native::sme`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub use cutile::native::sme::{port_mix, port_run};

use pack::tn;
pub use tn::{MIN_ROWS, gemm_f32_tn};

// f32 NEON 瓦片几何：单一事实源在 `cutile::native::f32`，这里重导出别名供本族子模块经
// `super::MR/NR/KB` 引用；数值不在本 crate 再写一份。
pub(crate) use cutile::native::f32::{KB, MR, NR};

/// 多线程稠密 f32 GEMM：`out[m,n] = a[m,k] · b[k,n]`。
///
/// # Panics
///
/// 三条缓冲长度不足 `m·k` / `k·n` / `m·n` 时 panic（调用方先校验）。
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    check_dims(a, b, m, n, k, out);
    // 协商面：把 16×64 整铺的大 GEMM 交给注册表里能接住的专用通道（SME ZA 外积）；
    // 小题、不可整铺或通道不存在时返回 `false`，退回下方 NEON 8×8 通道。nn 不写任何 arch/feature 门。
    if cutile::native::gemm_negotiate(a, b, m, n, k, out) {
        return;
    }
    let blocks = m.div_ceil(MR);
    let t = threads_override().min(blocks).max(1);
    if t <= 1 {
        gemm_f32_single(a, b, m, n, k, out);
        return;
    }
    // 常驻池：打包与计算共用一次发布（一次 Barrier 隔开两阶段），无重复 spawn。
    let npad = align_up(n, NR);
    with_panels(npad * k, |buf| {
        if run_pooled(
            pool(),
            Bufs {
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
pub fn gemm_f32_checked(a: View<'_, f32>, b: View<'_, f32>, out: &mut [f32]) -> Result<()> {
    let (m, k) = super::matmul::dim2(a.dims(), "gemm_f32")?;
    let (k2, n) = super::matmul::dim2(b.dims(), "gemm_f32")?;
    if k != k2 {
        return Err(Error::new(
            Code::Invalid,
            format!("gemm_f32 收缩轴不符：a·k={k}，b·k={k2}"),
        ));
    }
    if out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "gemm_f32 输出缓冲不符：out={}，期望 m·n={}",
                out.len(),
                m * n
            ),
        ));
    }
    gemm_f32(a.as_slice(), b.as_slice(), m, n, k, out);
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
    let npad = align_up(n, NR);
    let blocks = m.div_ceil(MR);
    // 行块按 MR 对齐：否则 18 线程切 256 行会得到 15 行/线程，8 行走微内核、7 行掉回标量尾块。
    let per = blocks.div_ceil(t) * MR;
    with_panels(npad * k, |bpack| {
        pack_b_f32_par_into(bpack, b, n, k, t);
        let bp: &[f32] = bpack;
        thread::scope(|s| {
            for (ti, block) in out[..m * n].chunks_mut(per * n).enumerate() {
                let r0 = ti * per;
                let rows = block.len() / n;
                let a_t = &a[r0 * k..(r0 + rows) * k];
                s.spawn(move || compute_gemm(a_t, bp, b, rows, n, k, block));
            }
        });
    });
}

/// 单线程稠密 f32 GEMM（基准确认内核质量用）：`out[m,n] = a[m,k] · b[k,n]`。
pub fn gemm_f32_single(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    let npad = align_up(n, NR);
    with_panels(npad * k, |bpack| {
        pack_b_f32_into(bpack, b, n, k);
        compute_gemm(a, bpack, b, m, n, k, out);
    });
}

/// 行块计算的三元形状：行数、列宽与 K。
#[derive(Clone, Copy)]
struct Extent {
    rows: usize,
    n: usize,
    k: usize,
}

/// 一个行块的块计算：A 打包成 8 行面板，再 j 外层、面板内层跑微内核；残尾走标量。
///
/// `layout` 是**数据而非分支**：`Nt`（B 按列驻留）与 `Tn`（W 转置驻留）曾各写一份逐字
/// 相同的正文，只在 `Sweep.layout` / `edge` 的最后一个实参上不同；现在一份正文配两次调用。
fn compute_rows(
    a: &[f32],
    bpack: &[f32],
    w: &[f32],
    extent: Extent,
    out: &mut [f32],
    layout: Layout,
) {
    let Extent { rows, n, k } = extent;
    let full = rows / MR;
    // A 面板借用线程局部池：面板每次都被整块重写，池里残留的旧值不会被读到，
    // 于是每行块省下一次 `full · k · MR` 的分配与清零。
    with_panels(full * k * MR, |apack| {
        pack_panels_into(a, full, k, apack);
        let mut first = true;
        let mut k0 = 0;
        while k0 < k {
            let kb = KB.min(k - k0);
            sweep_columns(
                Sweep {
                    a,
                    apack,
                    bpack,
                    w,
                    full,
                    n,
                    k,
                    layout,
                },
                out,
                k0,
                kb,
                first,
            );
            first = false;
            k0 += kb;
        }
    });
    // 残行只占 ≤7 行，一次把整段 K 走完，不参与分块累加。
    let tail = full * MR;
    if tail < rows {
        edge(
            a,
            w,
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
            layout,
        );
    }
}

/// Nt 视角的块计算：B 按列驻留（父模块的两条入口用它）。
fn compute_gemm(
    a: &[f32],
    bpack: &[f32],
    b: &[f32],
    rows: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) {
    compute_rows(a, bpack, b, Extent { rows, n, k }, out, Layout::Nt);
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

mod i8 {
    //! 职责：i8 sdot 微内核的**打包、扫描与调度骨架**。
    // i8 sdot 通道：MR=4 × NR=4，B 转置打包成「每列 k 连续」。瓦片本体（aarch64 dotprod
    // `sdot_tile` / `dot_one` / `harvest`）与端口探针已上提 `cutile::native::i8`；本文件只留
    // 面板打包、行块扫描、标量兜底与多线程分派，不持 arch/feature 门与汇编片段。
    // ---------------------------------------------------------------------------

    /// i8 微内核行数（四行共享同一 B 向量，摊薄 B 加载）。
    use std::thread;

    use diagnostic::{Code, Error, Result};
    use tensor::View;

    use super::align_up;
    use super::compute_gemm;
    use super::pack::{b_groups, group_panel};
    use super::pool::{Bufs, Job, Shape, pool, run_pooled};
    use super::threads_override;
    use super::tn;

    // i8 sdot 瓦片几何：单一事实源在 `cutile::native::i8_geo`，重导出别名供本模块与经 `super::` 引用的用例。
    pub(super) use cutile::native::i8_geo::{MR as I8_MR, NR as I8_NR};

    // 瓦片坐标与 dotprod 分派入口住在 prof 岛：探测、`target_feature` 与 `unsafe` 全收在岛内。
    use cutile::native::i8::{Gram, dot_block};

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
    pub fn gemm_i8_checked(a: View<'_, i8>, b: View<'_, i8>, out: &mut [i32]) -> Result<()> {
        let (m, k) = crate::matmul::dim2(a.dims(), "gemm_i8")?;
        let (k2, n) = crate::matmul::dim2(b.dims(), "gemm_i8")?;
        if k != k2 {
            return Err(Error::new(
                Code::Invalid,
                format!("gemm_i8 收缩轴不符：a·k={k}，b·k={k2}"),
            ));
        }
        if out.len() != m * n {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "gemm_i8 输出缓冲不符：out={}，期望 m·n={}",
                    out.len(),
                    m * n
                ),
            ));
        }
        gemm_i8(a.as_slice(), b.as_slice(), m, n, k, out);
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
                        acc += i32::from(ar[p]) * i32::from(bp[p]);
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
}
/// i8 `sdot` 端口峰值探针（K8 的分母），只在测量容器里导出；本体在 `cutile::native::i8`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub use cutile::native::i8::{
    SDOT_BLOCK_MAC_PER_K, SDOT_MAC_PER_ROUND, port_sdot, port_sdot_block,
};

/// 一条端口峰值探针：**一条指令路径**的分母。
///
/// 四个字段就是本模块与测量容器之间的全部协议：组名（`{group}/{threads}{suffix}`）、
/// 一轮的操作次数（算吞吐的分子）、探针本身。清单由提供方给，见 [`probes`]。
#[cfg(feature = "bench")]
#[derive(Clone, Copy, Debug)]
pub struct Probe {
    /// 基准组名前缀，如 `i8_peak`。
    pub group: &'static str,
    /// 组名后缀，用来在同前缀下区分通道，如 `_block`。
    pub suffix: &'static str,
    /// 探针一轮做多少次操作（MAC 数）。
    pub ops_per_round: usize,
    /// 跑 `rounds` 轮；本机没有这条指令路径时返回 `false` 且一条都不执行。
    pub run: fn(usize) -> bool,
}

/// 本机可测的端口探针清单——**由提供方回答**，调用方不写 `cfg`。
///
/// `nightly` 是本 crate 的私有 cfg，下游不能可靠判断；缺失通道直接不进清单。
/// 每项只测同一条指令路径，SME、sdot 与 f32 FMA 的分母不混用。
#[cfg(feature = "bench")]
pub fn probes() -> &'static [Probe] {
    PROBES
}

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const SME_FLOPS_PER_ROUND: usize = 4 * 16 * 16 * 2;

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const SME_PURE: Probe = Probe {
    group: "sme_peak",
    suffix: "",
    ops_per_round: SME_FLOPS_PER_ROUND,
    run: cutile::native::sme::port_run,
};

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const SME_MIX: Probe = Probe {
    group: "sme_peak",
    suffix: "_mix",
    ops_per_round: SME_FLOPS_PER_ROUND,
    run: sme_mix_probe,
};

/// 混合探针复用固定缓存行，测取数端口上限；真实内核走不同行只会更慢。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
fn sme_mix_probe(rounds: usize) -> bool {
    static A: [f32; 16] = [1.0; 16];
    static B: [f32; 64] = [1.0; 64];
    cutile::native::sme::port_mix(rounds, &A, &B)
}

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const I8_PURE: Probe = Probe {
    group: "i8_peak",
    suffix: "",
    ops_per_round: 2 * cutile::native::i8::SDOT_MAC_PER_ROUND,
    run: cutile::native::i8::port_sdot,
};

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const FMA_PURE: Probe = Probe {
    group: "fma_peak",
    suffix: "",
    ops_per_round: 2 * cutile::native::gemm8x8::FMA_MAC_PER_K,
    run: cutile::native::gemm8x8::port_fma,
};

/// 固定 K 让地址在环内推进，避免 LLVM 把 load 整体提出循环。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const PROBE_K: usize = 1024;

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const I8_BLOCK: Probe = Probe {
    group: "i8_peak",
    suffix: "_block",
    ops_per_round: 2 * cutile::native::i8::SDOT_BLOCK_MAC_PER_K * PROBE_K,
    run: |rounds| {
        static A: [i8; 4 * PROBE_K] = [3; 4 * PROBE_K];
        static B: [i8; 4 * PROBE_K] = [5; 4 * PROBE_K];
        cutile::native::i8::port_sdot_block(rounds, &A, &B, PROBE_K)
    },
};

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const FMA_BLOCK: Probe = Probe {
    group: "fma_peak",
    suffix: "_block",
    ops_per_round: 2 * cutile::native::gemm8x8::FMA_MAC_PER_K * PROBE_K,
    run: |rounds| {
        static A: [f32; NR * PROBE_K] = [1.0; NR * PROBE_K];
        static B: [f32; NR * PROBE_K] = [1.0; NR * PROBE_K];
        cutile::native::gemm8x8::port_fma_block(rounds, &A, &B, PROBE_K)
    },
};

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const PROBES: &[Probe] = &[SME_PURE, SME_MIX, I8_PURE, I8_BLOCK, FMA_PURE, FMA_BLOCK];
#[cfg(all(feature = "bench", not(target_arch = "aarch64")))]
const PROBES: &[Probe] = &[];

use i8::{f32_calc, f32_pack};
pub use i8::{gemm_i8, gemm_i8_checked, gemm_i8_single};
// 扫描骨架只读侧（Sweep/Span/Layout/残块 edge）留 nn；8×8 微内核本体与 Tile 住在 cutile::native。
use cutile::native::gemm8x8::{Tile, micro_packed};
use micro::{Layout, Span, Sweep, edge};

#[cfg(test)]
mod tests {
    include!("../tests/kernel_fast_tests.rs");
    // SME 对拍：stable／nightly 共用 C 内核，运行时能力由 prof 安全桥决定。
    #[cfg(target_arch = "aarch64")]
    include!("../tests/kernel_sme_tests.rs");
}

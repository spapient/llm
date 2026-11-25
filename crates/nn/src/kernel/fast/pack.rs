//! 打包：把 A 的行块转成 `[p][MR]`、把 B 的列组转成 `[jblk][p][NR]` 面板。
use std::thread;

use super::{MR, NR, align_up};

/// A 的每个 8 行块转置成 `[p][8]` 面板：微内核每条 `p` 只需两次连续向量加载。
///
/// 写满 `apack` 的每一个元素（`full · k · MR`），因此可以安全复用一个更早借出的面板池：
/// 池里没有「上次剩下的半截」会被读到。
pub(crate) fn pack_panels_into(a: &[f32], full: usize, k: usize, apack: &mut [f32]) {
    debug_assert_eq!(apack.len(), full * k * MR);
    for blk in 0..full {
        let src = &a[blk * MR * k..(blk * MR + MR) * k];
        let dst = &mut apack[blk * k * MR..blk * k * MR + k * MR];
        for p in 0..k {
            for r in 0..MR {
                dst[p * MR + r] = src[r * k + p];
            }
        }
    }
}

/// 一个列组面板的内层转置：`first` 起 `NR` 列 × 全 `k` 行 → `dst[base..]`。
///
/// 面板布局由微内核取数方式决定，编译期固化，两种不能互替：
/// - 步主 `[q][NR]`（`COL_MAJOR=false`）：每个收缩步 `q` 的 `NR` 列连续（f32 NEON 向量按步加载）；
/// - 列主 `[NR][k]`（`COL_MAJOR=true`）：每列的 `k` 个元素连续（i8 sdot 按列主取数）。
///
/// 布局用 `bool` 作 const 标签：const 泛型只接受整数 / bool / char。`transposed` 只决定源下标
/// （仅步主布局有 Tn 视角）：
/// - `false`（Nt）：`src[q*n + first + c]`，B 按 `[k][n]` 驻留；
/// - `true`（Tn）：`src[(first+c)*k + q]`，W 按 `[n][k]` 驻留。
///
/// 元素类型、列宽与面板布局在编译期固化；整段、分片与池化三种外层遍历只差列组下标。
pub(crate) fn panel_tile<T: Copy, const NR: usize, const COL_MAJOR: bool>(
    src: &[T],
    dst: &mut [T],
    n: usize,
    k: usize,
    first: usize,
    base: usize,
    transposed: bool,
) {
    if COL_MAJOR {
        panel_col_major::<T, NR>(src, dst, n, k, first, base);
    } else {
        panel_row_major::<T, NR>(src, dst, n, k, first, base, transposed);
    }
}

/// 列主打包：列 `nb` 占连续 `k` 个槽；源恒为行主 B（Nt），无 Tn 视角。
fn panel_col_major<T: Copy, const NR: usize>(
    src: &[T],
    dst: &mut [T],
    n: usize,
    k: usize,
    first: usize,
    base: usize,
) {
    let cols = if first < n { NR.min(n - first) } else { 0 };
    for nb in 0..cols {
        let jj = first + nb;
        let off = base + nb * k;
        for q in 0..k {
            dst[off + q] = src[q * n + jj];
        }
    }
}

/// 行主打包：`transposed` 时源按 `[col][k]` 读（Tn），否则按行主 `[row][n]` 整块拷贝。
fn panel_row_major<T: Copy, const NR: usize>(
    src: &[T],
    dst: &mut [T],
    n: usize,
    k: usize,
    first: usize,
    base: usize,
    transposed: bool,
) {
    let cols = if first < n { NR.min(n - first) } else { 0 };
    for q in 0..k {
        let row = base + q * NR;
        if transposed {
            for (c, slot) in dst[row..row + cols].iter_mut().enumerate() {
                *slot = src[(first + c) * k + q];
            }
        } else {
            let src0 = q * n + first;
            dst[row..row + cols].copy_from_slice(&src[src0..src0 + cols]);
        }
    }
}

/// 整段串行打包：`n` 对齐到 `NR`，补零列由零初始化的面板缓冲承担。
pub(crate) fn pack_serial<T: Copy, const NR: usize, const COL_MAJOR: bool>(
    bpack: &mut [T],
    b: &[T],
    n: usize,
    k: usize,
    transposed: bool,
) {
    let npad = align_up(n, NR);
    debug_assert_eq!(bpack.len(), npad * k);
    for jb in 0..npad / NR {
        panel_tile::<T, NR, COL_MAJOR>(b, bpack, n, k, jb * NR, jb * k * NR, transposed);
    }
}

/// 一段连续列组分片：`chunk` 覆盖每组 `NR` 列，按 `base_group` 起。列组与面板槽一一对应，
/// 按 `bi` 串行推进（写入位置互不重叠），故此处保留循环而非整块批量搬运。
pub(crate) fn pack_span<T: Copy, const NR: usize, const COL_MAJOR: bool>(
    chunk: &mut [T],
    b: &[T],
    n: usize,
    k: usize,
    base_group: usize,
    transposed: bool,
) {
    let blocks = chunk.len() / (NR * k);
    for bi in 0..blocks {
        panel_tile::<T, NR, COL_MAJOR>(
            b,
            chunk,
            n,
            k,
            (base_group + bi) * NR,
            bi * k * NR,
            transposed,
        );
    }
}

/// 池化打包的槽轮转体：本槽从 `slot` 起按 `p` 步长接管列组，每组一次面板转置。
/// 整段（[`pack_serial`]）、分片（[`pack_span`]）与池化轮转三种外层遍历只差列组下标；
/// 布局与源视角仍由 `COL_MAJOR` / `transposed` 标签钉住。
#[allow(clippy::too_many_arguments)]
pub(crate) fn group_panel<T: Copy, const NR: usize, const COL_MAJOR: bool>(
    b: &[T],
    buf: &mut [T],
    n: usize,
    k: usize,
    npad: usize,
    p: usize,
    slot: usize,
    transposed: bool,
) {
    let groups = npad / NR;
    let mut g = slot;
    while g < groups {
        panel_tile::<T, NR, COL_MAJOR>(b, buf, n, k, g * NR, g * NR * k, transposed);
        g += p;
    }
}

/// f32 列组面板内层转置（Tn 入口复用，签名保持不变）。
pub(crate) fn pack_panel(
    src: &[f32],
    dst: &mut [f32],
    n: usize,
    k: usize,
    first: usize,
    base: usize,
    transposed: bool,
) {
    panel_tile::<f32, NR, false>(src, dst, n, k, first, base, transposed);
}

/// B 打包成 `[jblk][p][NR]`：微内核每条 `p` 只需两次连续向量加载（行主序 B 是每 `p` 跨一条行）。
pub(crate) fn pack_b_f32_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize) {
    pack_serial::<f32, NR, false>(bpack, b, n, k, false);
}

/// 单个并行分片：本槽承担的列组顺序打包进 `chunk`。
fn pack_b_chunk(chunk: &mut [f32], b: &[f32], n: usize, k: usize, base_group: usize) {
    pack_span::<f32, NR, false>(chunk, b, n, k, base_group, false);
}

/// 并行 B 打包：按 8 列组切线程。`k·n` 小的时候串行更划算（线程创建比转置贵）。
pub(crate) fn pack_b_f32_par_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize, t: usize) {
    let groups = n.div_ceil(NR);
    if t <= 1 || n * k < (1 << 16) {
        pack_b_f32_into(bpack, b, n, k);
        return;
    }
    let per = groups.div_ceil(t);
    let span = per * NR * k;
    thread::scope(|s| {
        for (gi, chunk) in bpack.chunks_mut(span).enumerate() {
            s.spawn(move || pack_b_chunk(chunk, b, n, k, gi * per));
        }
    });
}

/// 池化打包的分片体：本槽负责的 8 列组按 `p` 步长轮转，写入 `[jblk][p][NR]`（零初始化覆盖补零列）。
pub(crate) fn b_groups(
    b: &[f32],
    buf: &mut [f32],
    n: usize,
    k: usize,
    npad: usize,
    p: usize,
    slot: usize,
) {
    group_panel::<f32, NR, false>(b, buf, n, k, npad, p, slot, false);
}

pub(crate) mod micro {
    //! 微内核整行块扫描坐标与标量残块兜底；微内核本体在 `cutile::native`。

    #[derive(Clone, Copy)]
    pub(crate) struct Sweep<'a> {
        pub(crate) a: &'a [f32],
        pub(crate) apack: &'a [f32],
        pub(crate) bpack: &'a [f32],
        pub(crate) w: &'a [f32],
        pub(crate) full: usize,
        pub(crate) n: usize,
        pub(crate) k: usize,
        pub(crate) layout: Layout,
    }

    #[derive(Clone, Copy)]
    pub(crate) struct Span {
        pub(crate) i: usize,
        pub(crate) j: usize,
        pub(crate) mr: usize,
        pub(crate) nr: usize,
        pub(crate) n: usize,
        pub(crate) k: usize,
        pub(crate) stride: usize,
    }

    #[derive(Clone, Copy)]
    pub(crate) enum Layout {
        Nt,
        Tn,
    }

    pub(crate) fn weight_stride(span: Span, k0: usize, layout: Layout) -> (usize, usize, usize) {
        match layout {
            Layout::Nt => (k0 * span.n + span.j, span.n, 1),
            Layout::Tn => (span.j * span.stride + k0, 1, span.stride),
        }
    }

    pub(crate) fn edge(
        a: &[f32],
        w: &[f32],
        out: &mut [f32],
        span: Span,
        k0: usize,
        first: bool,
        layout: Layout,
    ) {
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
}

pub(in crate::kernel::fast) mod tn {
    //! 线性层布局（TN）的稠密 f32 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
    use diagnostic::Result;
    use tensor::View;

    use crate::kernel::fast::pack::{group_panel, pack_panel};
    use crate::kernel::fast::pool::{Bufs, Shape, pool, run_pooled};
    use crate::kernel::fast::{Layout, MR, NR, align_up, threads_override};

    /// 走这条入口的最少行数：一个完整微内核行块。再少的批只有残行（标量），打包的转置成本摊不回来，
    /// 调用方应当继续用按列驻留的 tile 核。
    pub const MIN_ROWS: usize = MR;

    /// 多线程 TN 稠密 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
    ///
    /// 语义同 [`crate::matmul::forward_tn_tiled`]（同一组累加口径，IEEE 容差内一致），吞吐按父模块的
    /// 8×8 寄存器分块 + 常驻池给。
    ///
    /// # Errors
    ///
    /// 输入不是合法的 TN 二维几何时返回诊断。
    ///
    /// # Panics
    ///
    /// 三条缓冲长度不足 `rows·k` / `n·k` / `rows·n` 时 panic（调用方先校验）。
    pub fn gemm_f32_tn(x: View<'_, f32>, w: View<'_, f32>, out: &mut [f32]) -> Result<()> {
        let (rows, n, k) = crate::matmul::tn_geom(x.dims(), w.dims())?;
        let (x, w) = (x.as_slice(), w.as_slice());
        crate::kernel::fast::check_dims(x, w, rows, n, k, out);
        let t = threads_override().min(rows.div_ceil(MR)).max(1);
        if t <= 1 {
            single_tn(x, w, rows, n, k, out);
            return Ok(());
        }
        if !pooled_tn(x, w, rows, n, k, out, t) {
            scoped_tn(x, w, out, rows, n, k, t);
        }
        Ok(())
    }

    /// 单线程：整段打包一次，再按行块算完整段 K。
    fn single_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
        let npad = align_up(n, NR);
        crate::kernel::fast::with_panels(npad * k, |bpack| {
            pack_tn_into(bpack, w, n, k);
            compute_tn(x, bpack, w, rows, n, k, out);
        });
    }

    /// 常驻池路径；池不可用（参与数不足 2）时返回 `false`，由 [`scoped_tn`] 兜底。
    fn pooled_tn(
        x: &[f32],
        w: &[f32],
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        t: usize,
    ) -> bool {
        let npad = align_up(n, NR);
        crate::kernel::fast::with_panels(npad * k, |buf| {
            run_pooled(
                pool(),
                Bufs {
                    a: x.as_ptr() as *const u8,
                    b: w.as_ptr() as *const u8,
                    buf: buf.as_mut_ptr() as *mut u8,
                    out: out.as_mut_ptr() as *mut u8,
                },
                Shape {
                    m: rows,
                    n,
                    k,
                    npad,
                    tn: true,
                },
                MR,
                t,
                crate::kernel::fast::f32_pack,
                crate::kernel::fast::f32_calc,
            )
        })
    }

    /// 兜底（池不可用）：打包一次后 shared 给 scope 线程，行块按 MR 对齐切。
    fn scoped_tn(x: &[f32], w: &[f32], out: &mut [f32], rows: usize, n: usize, k: usize, t: usize) {
        let npad = align_up(n, NR);
        let per = rows.div_ceil(MR).div_ceil(t) * MR;
        crate::kernel::fast::with_panels(npad * k, |bpack| {
            pack_tn_into(bpack, w, n, k);
            let bp: &[f32] = bpack;
            std::thread::scope(|s| {
                for (ti, block) in out[..rows * n].chunks_mut(per * n).enumerate() {
                    let r0 = ti * per;
                    let got = block.len() / n;
                    let a_l = &x[r0 * k..(r0 + got) * k];
                    s.spawn(move || compute_tn(a_l, bp, w, got, n, k, block));
                }
            });
        });
    }

    /// 整段打包（单槽、兜底路径与单线程用）：面板布局与父模块一致，只是源按 `[n][k]` 读。
    fn pack_tn_into(bpack: &mut [f32], w: &[f32], n: usize, k: usize) {
        let npad = align_up(n, NR);
        debug_assert_eq!(bpack.len(), npad * k);
        for g in 0..npad / NR {
            pack_panel(w, bpack, n, k, g * NR, g * k * NR, true);
        }
    }

    /// 本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包（与父模块同一套分工）。
    pub(in crate::kernel::fast) fn stage(
        w: &[f32],
        buf: &mut [f32],
        n: usize,
        k: usize,
        npad: usize,
        p: usize,
        slot: usize,
    ) {
        group_panel::<f32, NR, false>(w, buf, n, k, npad, p, slot, true);
    }

    /// 共用父模块行机器，只固定 TN 布局。
    pub(in crate::kernel::fast) fn compute_tn(
        a: &[f32],
        bpack: &[f32],
        w: &[f32],
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) {
        crate::kernel::fast::compute_rows(
            a,
            bpack,
            w,
            crate::kernel::fast::Extent { rows, n, k },
            out,
            Layout::Tn,
        );
    }

    #[cfg(test)]
    mod tests {
        include!("../../../tests/kernel_tn_tests.rs");
    }
}

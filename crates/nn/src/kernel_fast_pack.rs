    //! 打包：把 A 的行块转成 `[p][MR]`、把 B 的列组转成 `[jblk][p][NR]` 面板。
    use std::thread;

    use super::{MR, NR};

    /// A 的每个 8 行块转置成 `[p][8]` 面板：微内核每条 `p` 只需两次连续向量加载。
    ///
    /// 写满 `apack` 的每一个元素（`full · k · MR`），因此可以安全复用一个更早借出的面板池：
    /// 池里没有「上次剩下的半截」会被读到。
    pub(super) fn pack_panels_into(a: &[f32], full: usize, k: usize, apack: &mut [f32]) {
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
    pub(super) fn panel_tile<T: Copy, const NR: usize, const COL_MAJOR: bool>(
        src: &[T],
        dst: &mut [T],
        n: usize,
        k: usize,
        first: usize,
        base: usize,
        transposed: bool,
    ) {
        let cols = if first < n { NR.min(n - first) } else { 0 };
        if COL_MAJOR {
            // 列主：列 `nb` 占连续 `k` 个槽；源恒为行主 B（Nt），无 Tn 视角。
            for nb in 0..cols {
                let jj = first + nb;
                let off = base + nb * k;
                for q in 0..k {
                    dst[off + q] = src[q * n + jj];
                }
            }
        } else {
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
    }

    /// 整段串行打包：`n` 对齐到 `NR`，补零列由零初始化的面板缓冲承担。
    pub(super) fn pack_serial<T: Copy, const NR: usize, const COL_MAJOR: bool>(
        bpack: &mut [T],
        b: &[T],
        n: usize,
        k: usize,
        transposed: bool,
    ) {
        let npad = n.div_ceil(NR) * NR;
        debug_assert_eq!(bpack.len(), npad * k);
        for jb in 0..npad / NR {
            panel_tile::<T, NR, COL_MAJOR>(b, bpack, n, k, jb * NR, jb * k * NR, transposed);
        }
    }

    /// 一段连续列组分片：`chunk` 覆盖每组 `NR` 列，按 `base_group` 起。列组与面板槽一一对应，
    /// 按 `bi` 串行推进（写入位置互不重叠），故此处保留循环而非整块批量搬运。
    pub(super) fn pack_span<T: Copy, const NR: usize, const COL_MAJOR: bool>(
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
    pub(super) fn group_panel<T: Copy, const NR: usize, const COL_MAJOR: bool>(
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
    pub(super) fn pack_panel(
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
    pub(super) fn pack_b_f32_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize) {
        pack_serial::<f32, NR, false>(bpack, b, n, k, false);
    }

    /// 单个并行分片：本槽承担的列组顺序打包进 `chunk`。
    fn pack_b_chunk(
        chunk: &mut [f32],
        b: &[f32],
        n: usize,
        k: usize,
        base_group: usize,
    ) {
        pack_span::<f32, NR, false>(chunk, b, n, k, base_group, false);
    }

    /// 并行 B 打包：按 8 列组切线程。`k·n` 小的时候串行更划算（线程创建比转置贵）。
    pub(super) fn pack_b_f32_par_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize, t: usize) {
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
    pub(super) fn b_groups(
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

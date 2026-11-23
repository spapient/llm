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

    /// 一个列组面板的内层转置：`first` 起的 `cols` 列 × 全 `k` 行 → `dst[base..]` 的 `[q][NR]`。
    ///
    /// `transposed` 只决定源下标：
    /// - `false`（Nt）：`src[q*n + first + c]`，B 按 `[k][n]` 驻留，整行 `copy_from_slice`；
    /// - `true`（Tn）：`src[(first+c)*k + q]`，W 按 `[n][k]` 驻留，逐元素转置。
    ///
    /// 这是 B 打包的唯一算术：整段、分片与池化三种外层遍历只差列组下标，内层下标只有这一份。
    pub(super) fn pack_panel(
        src: &[f32],
        dst: &mut [f32],
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

    /// B 打包成 `[jblk][p][NR]`：微内核每条 `p` 只需两次连续向量加载（行主序 B 是每 `p` 跨一条行）。
    pub(super) fn pack_b_f32_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize) {
        let npad = n.div_ceil(NR) * NR;
        debug_assert_eq!(bpack.len(), npad * k);
        for jb in 0..npad / NR {
            pack_panel(b, bpack, n, k, jb * NR, jb * k * NR, false);
        }
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
                s.spawn(move || {
                    let blocks = chunk.len() / (NR * k);
                    for bi in 0..blocks {
                        pack_panel(b, chunk, n, k, (gi * per + bi) * NR, bi * k * NR, false);
                    }
                });
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
        let groups = npad / NR;
        let mut g = slot;
        while g < groups {
            pack_panel(b, buf, n, k, g * NR, g * k * NR, false);
            g += p;
        }
    }

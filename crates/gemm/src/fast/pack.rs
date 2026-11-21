//! 打包：把 A 的行块转成 `[p][MR]`、把 B 的列组转成 `[jblk][p][NR]` 面板。
use std::thread;

use super::{MR, NR};

/// A 的每个 8 行块转置成 `[p][8]` 面板：微内核每条 `p` 只需两次连续向量加载。
pub(super) fn pack_panels(a: &[f32], full: usize, k: usize) -> Vec<f32> {
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

/// B 打包成 `[jblk][p][NR]`：微内核每条 `p` 只需两次连续向量加载（行主序 B 是每 `p` 跨一条行）。
pub(super) fn pack_b_f32_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize) {
    let npad = n.div_ceil(NR) * NR;
    debug_assert_eq!(bpack.len(), npad * k);
    for jb in 0..npad / NR {
        let first = jb * NR;
        let cols = if first < n { NR.min(n - first) } else { 0 };
        for p in 0..k {
            let dst = (jb * k + p) * NR;
            bpack[dst..dst + cols].copy_from_slice(&b[p * n + first..p * n + first + cols]);
        }
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
            s.spawn(move || pack_b_f32_span(chunk, b, n, k, gi * per));
        }
    });
}

/// 一段连续的列组打包：`blk0` 起，每块 `NR · k`。
fn pack_b_f32_span(chunk: &mut [f32], b: &[f32], n: usize, k: usize, blk0: usize) {
    let blocks = chunk.len() / (NR * k);
    for bi in 0..blocks {
        let first = (blk0 + bi) * NR;
        let cols = if first < n { NR.min(n - first) } else { 0 };
        let tile = &mut chunk[bi * NR * k..(bi + 1) * NR * k];
        for p in 0..k {
            let dst = p * NR;
            tile[dst..dst + cols].copy_from_slice(&b[p * n + first..p * n + first + cols]);
        }
    }
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
        let first = g * NR;
        let cols = if first < n { NR.min(n - first) } else { 0 };
        let base = g * k * NR;
        for q in 0..k {
            let src = q * n + first;
            let dst = base + q * NR;
            buf[dst..dst + cols].copy_from_slice(&b[src..src + cols]);
        }
        g += p;
    }
}

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

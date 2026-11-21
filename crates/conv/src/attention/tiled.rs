//! 分块 tile 的两条自注意力内核：Flash（online softmax）与稠密（先落地分数、再按块加权）。
//!
//! 两条都与 [`crate::attention`] 的同名路径**逐位一致**——每个 query 行的列累加次序不变，收益只来自
//! 块内复用：[`TILE_S`]×`d` 的 K/V 块驻留缓存后被一个行块的查询共用，而不是每行重走一遍。分数与加权
//! 两段都按块推进，于是常驻的永远是「K/V 块 + 行块内的分数轨」，没有 `s×s` 的整块落地。
use crate::attention::check_decode_len;
use diagnostic::Result;

/// KV 分块 tile 的块宽：块内历史列数。
pub const TILE_S: usize = 64;

/// 稠密 tile 一次驻留的查询行数：块内查询共用同一段 K/V 分块。
pub const TILE_ROWS: usize = 16;

/// 行块分数轨的预算（元素数）：`s` 很大时按预算收窄行块。
///
/// 行块越大，K/V 被复用的次数越多；但分数轨是 `rows × s`，撑出缓存后每条分数都要回主存写一次。
/// 两条约束在 `s ≈ 4096` 处正好交于 [`TILE_ROWS`]，更长的序列自动收窄而不是硬撑。
const SCORE_BUDGET: usize = 1 << 16;

/// Flash 分块 tile 内核：外层按 [`TILE_S`] 切历史 K/V，内层遍历全部 query。
///
/// # Errors
///
/// 长度不符、头宽为 0，或序列长为 0 时返回说明。
pub fn flash_tiled_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    s: usize,
    d: usize,
) -> Result<()> {
    check_decode_len(q, k, v, out, s, s, d)?;
    let ctx = TiledCtx {
        q,
        k,
        v,
        d,
        scale: (d as f32).sqrt(),
    };

    // 每 query 的跨块状态；行主序 acc 一次分配，末段统一除归一化分母写出。
    let mut max = vec![f32::MIN; s];
    let mut sum = vec![0.0f32; s];
    let mut acc = vec![0.0f32; s * d];

    let mut start = 0;
    while start < s {
        let end = (start + TILE_S).min(s);
        for row in 0..s {
            ctx.update_row(row, start, end, &mut max, &mut sum, &mut acc);
        }
        start = end;
    }

    ctx.write_out(&acc, &sum, out);
    Ok(())
}

/// 稠密分块 tile 内核：分数按 K/V 块算进行块内的临时轨，行内 softmax 后再按 K/V 块加权。
///
/// 与 [`crate::attention::forward`] 逐位一致（分数、`exp`、归一化和与加权累加的次序都不变），差别只在
/// K/V 的读取次数：朴素路径每行读一遍整段 K/V，这里一个行块只读一遍。
///
/// # Errors
///
/// 长度不符、头宽为 0，或序列长为 0 时返回说明。
pub fn forward(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], s: usize, d: usize) -> Result<()> {
    check_decode_len(q, k, v, out, s, s, d)?;
    let ctx = TiledCtx {
        q,
        k,
        v,
        d,
        scale: (d as f32).sqrt(),
    };

    // `s ≥ 1` 已由长度校验保证，预算除法不会除零。
    let rows = TILE_ROWS.min(s).min((SCORE_BUDGET / s).max(1));
    let mut scores = vec![0.0f32; rows * s];

    let mut first = 0;
    while first < s {
        let end = (first + rows).min(s);
        let block = &mut scores[..(end - first) * s];
        ctx.score_rows(first, end, block);
        for (offset, line) in block.chunks_mut(s).enumerate() {
            let row = (first + offset) * d;
            ctx.weight_row(line, &mut out[row..][..d]);
        }
        first = end;
    }
    Ok(())
}

/// 分块 tile 的只读上下文：操作数、头宽与缩放因子；跨块可变状态由调用方持有。
struct TiledCtx<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    d: usize,
    scale: f32,
}

impl TiledCtx<'_> {
    /// 单 query 行在 `[start,end)` 块内的 online 更新；`max`/`sum`/`acc` 是跨块状态。
    fn update_row(
        &self,
        row: usize,
        start: usize,
        end: usize,
        max: &mut [f32],
        sum: &mut [f32],
        acc: &mut [f32],
    ) {
        let qrow = &self.q[row * self.d..][..self.d];
        let mut m = max[row];
        let mut l = sum[row];
        let arow = &mut acc[row * self.d..][..self.d];
        for col in start..end {
            let kcol = &self.k[col * self.d..][..self.d];
            let score = llm_gemm::matmul::dot_lanes(qrow, kcol) / self.scale;
            let new_max = m.max(score);
            let reweight = (m - new_max).exp();
            let prob = (score - new_max).exp();
            let vcol = &self.v[col * self.d..][..self.d];
            // `reweight` 只在最大值被这一列抬高时偏离 1；乘 1.0 是精确恒等，跳过不改变任何一位，
            // 却省掉每列 `d` 次乘法——块内绝大多数列都走这一支。
            if reweight == 1.0 {
                l += prob;
                for (slot, value) in arow.iter_mut().zip(vcol) {
                    *slot += prob * value;
                }
            } else {
                l = l * reweight + prob;
                for (slot, value) in arow.iter_mut().zip(vcol) {
                    *slot = *slot * reweight + prob * value;
                }
            }
            m = new_max;
        }
        max[row] = m;
        sum[row] = l;
    }

    /// 行块 `[first,end)` 的分数：K/V 列块在外、行块在内，一个列块只从主存读一遍。
    ///
    /// `scores` 按行主序存 `(end - first) × cols`；每行的列顺序仍是 `0..cols`，于是与朴素路径
    /// 逐位一致。
    fn score_rows(&self, first: usize, end: usize, scores: &mut [f32]) {
        let cols = self.k.len() / self.d;
        let mut start = 0;
        while start < cols {
            let stop = (start + TILE_S).min(cols);
            let keys = &self.k[start * self.d..stop * self.d];
            for (line, row) in scores.chunks_mut(cols).zip(first..end) {
                let qrow = &self.q[row * self.d..][..self.d];
                let span = &mut line[start..stop];
                for (slot, key) in span.iter_mut().zip(keys.chunks_exact(self.d)) {
                    *slot = llm_gemm::matmul::dot_lanes(qrow, key) / self.scale;
                }
            }
            start = stop;
        }
    }

    /// 一行分数 → 一行输出：`exp` 与归一化和一次算完，再按 K/V 列块做 AXPY。
    ///
    /// 列块内先按列把 `prob / sum` 除一次（所有头共用同一份权重，不是每头重除一遍），随后列在外、
    /// 头在内：`V` 的每一列连续读、`out` 连续写，每头沿列仍是升序累加——
    /// 与朴素路径逐位一致，但内层是可向量化的连续存取（固定头扫列会让每次读跨 `d` 个元素）。
    fn weight_row(&self, scores: &mut [f32], out: &mut [f32]) {
        let max = scores.iter().copied().fold(f32::MIN, f32::max);
        let mut sum = 0.0f32;
        for score in scores.iter_mut() {
            *score = (*score - max).exp();
            sum += *score;
        }

        out.fill(0.0);
        let mut start = 0;
        while start < scores.len() {
            let stop = (start + TILE_S).min(scores.len());
            for (offset, prob) in scores[start..stop].iter().enumerate() {
                let weight = *prob / sum;
                let vcol = &self.v[(start + offset) * self.d..][..self.d];
                for (slot, value) in out.iter_mut().zip(vcol) {
                    *slot += weight * value;
                }
            }
            start = stop;
        }
    }

    /// 末段：把跨块累加向量除归一化分母，写进 `out`。
    fn write_out(&self, acc: &[f32], sum: &[f32], out: &mut [f32]) {
        for row in 0..sum.len() {
            let arow = &acc[row * self.d..][..self.d];
            let orow = &mut out[row * self.d..][..self.d];
            for h in 0..self.d {
                orow[h] = arow[h] / sum[row];
            }
        }
    }
}

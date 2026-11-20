//! 分块 tile 的 Flash 自注意力：外层按 KV 块切、内层遍历 query。
//!
//! 与 [`crate::attention::flash_forward`] 的 online 路径**结果逐位一致**（每个 query 的 `col`
//! 累加顺序不变），收益是 KV 块（[`TILE_S`]×d）驻留缓存、被多 query 复用，prefill 大序列时
//! 降低 KV 的跨 query 重复读。跨块状态（running max / 归一化分母 / 累加向量）一次性分配。

use crate::attention::check_decode_len;
use diagnostic::Result;

/// KV 分块 tile 的块宽：块内历史列数。
pub const TILE_S: usize = 64;

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
            let score = crate::matmul::dot_lanes(qrow, kcol) / self.scale;
            let new_max = m.max(score);
            let reweight = (m - new_max).exp();
            let prob = (score - new_max).exp();
            l = l * reweight + prob;
            let vcol = &self.v[col * self.d..][..self.d];
            for h in 0..self.d {
                arow[h] = arow[h] * reweight + prob * vcol[h];
            }
            m = new_max;
        }
        max[row] = m;
        sum[row] = l;
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

//! DeepSeek MLA 的潜在 KV（§5.8.1）：每 token 每层只缓存 `c^KV` 与解耦 RoPE 键 `k^R`。

use crate::invalid;
use diagnostic::Result;
use op::scan::Scalar;

use crate::csa::{accumulate, dot, softmax_row};

/// 报错出处：几何、长度与寻址越界都在入口失败，不留到数值阶段。
const SITE: &str = "algorithm::mla";

/// V3 公开超参：解耦 RoPE 键宽 `d_h^R`。
pub const V3_ROPE: usize = 64;
/// V3 公开超参：潜在宽 `d_c`。
pub const V3_LATENT: usize = 512;
/// V3 公开超参：头宽 `d_h`。
pub const V3_HEAD_DIM: usize = 128;
/// V3 公开超参：头数 `n_h`。
pub const V3_HEADS: usize = 128;
/// V3 公开超参：层数（KV 记账按层累加）。
pub const V3_LAYERS: usize = 61;

/// 每个 token 每个位置的 MLA 行宽 `d_c + d_h^R`：这就是缓存要存的全部宽度。
pub fn row_width(latent: usize, rope: usize) -> usize {
    latent + rope
}

/// 同口径的稠密 KV 行宽 `2 · n_h · d_h`：MLA 的收益就是拿它当分母。
pub fn dense_width(heads: usize, head_dim: usize) -> usize {
    2 * heads * head_dim
}

/// 行宽之比 `(d_c + d_h^R) / (2 · n_h · d_h)`。
pub fn ratio(heads: usize, head_dim: usize, latent: usize, rope: usize) -> f64 {
    row_width(latent, rope) as f64 / dense_width(heads, head_dim) as f64
}

/// 计算同一次注意力的两条等价路径。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Path {
    /// 吸收式：`W^{UK}` 吸到查询侧、`W^{UV}` 吸到输出侧，每 token 只读 `d_c + d_h^R`。
    Absorb,
    /// 升维参考：把 `c` 还原成 `k_h` / `v_h` 再算标准注意力，供数值对拍。
    Upref,
}

impl Path {
    /// 稳定的短名，写进报告与参数表。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Absorb => "absorb",
            Self::Upref => "upref",
        }
    }
}

/// 潜在 KV 缓存：每 token 只有 `c^KV`（`d_c`）与解耦 RoPE 键 `k^R`（`d_h^R`）两段。
#[derive(Debug, Clone, PartialEq)]
pub struct Latent<S: Scalar = f32> {
    latent_width: usize,
    rope_width: usize,
    latent: Vec<S>,
    rope: Vec<S>,
}

impl<S: Scalar> Latent<S> {
    /// 空缓存（`seq = 0`）：只能由 [`Mla::empty_cache`] 产生，宽度与层绑定。
    fn empty(latent_width: usize, rope_width: usize) -> Self {
        Self {
            latent_width,
            rope_width,
            latent: Vec::new(),
            rope: Vec::new(),
        }
    }

    /// 已缓存的 token 数。
    pub fn seq(&self) -> usize {
        self.latent.len() / self.latent_width
    }

    /// 潜在段宽 `d_c`。
    pub fn latent_width(&self) -> usize {
        self.latent_width
    }

    /// RoPE 键宽 `d_h^R`。
    pub fn rope_width(&self) -> usize {
        self.rope_width
    }

    /// 一个 token 位置的缓存宽度 `d_c + d_h^R`。
    pub fn row_width(&self) -> usize {
        row_width(self.latent_width, self.rope_width)
    }

    /// 整段缓存的元素数：`seq · (d_c + d_h^R)`，这是 §5.8.1 那条 Must 的可核对形态。
    pub fn cache_elements(&self) -> usize {
        self.latent.len() + self.rope.len()
    }

    /// 两段的原始视图，都是行主序 `[seq, width]`。
    pub fn rows(&self) -> (&[S], &[S]) {
        (&self.latent, &self.rope)
    }

    /// 第 `t` 个位置的 `(c^KV, k^R)`；越界返回 `None`。
    pub fn row(&self, t: usize) -> Option<(&[S], &[S])> {
        if t >= self.seq() {
            return None;
        }
        let (latent, rope) = self.rows();
        Some((
            &latent[t * self.latent_width..(t + 1) * self.latent_width],
            &rope[t * self.rope_width..(t + 1) * self.rope_width],
        ))
    }

    /// 追加一个位置的 `(c^KV, k^R)`：解码每步调一次。
    ///
    /// # Errors
    ///
    /// 两段宽度必须正好是 `d_c` 与 `d_h^R`。宽度不符时**不追加任何一段**——只推进一半的缓存
    /// 会让 `seq` 在下一行对不上，而那种错要到很远的地方才暴露。
    pub fn push(&mut self, latent: &[S], rope: &[S]) -> Result<()> {
        if latent.len() != self.latent_width || rope.len() != self.rope_width {
            return Err(invalid(SITE, format!(
                    "潜在行宽度不符：给的是 ({}, {})，缓存要 ({}, {})",
                    latent.len(),
                    rope.len(),
                    self.latent_width,
                    self.rope_width
                )));
        }
        self.latent.extend_from_slice(latent);
        self.rope.extend_from_slice(rope);
        Ok(())
    }
}

/// MLA 层：超参 `(n_h, d_h, d_c, d_h^R)` 加上两条低秩投影 `W^{UK}` / `W^{UV}`。
#[derive(Debug, Clone, PartialEq)]
pub struct Mla<S: Scalar = f32> {
    heads: usize,
    head_dim: usize,
    latent: usize,
    rope: usize,
    up_key: Vec<S>,
    up_value: Vec<S>,
}

impl<S: Scalar> Mla<S> {
    /// 由超参与两条投影构造。
    ///
    /// # Errors
    ///
    /// 四个超参任一为零、或 `n_h · d_h · d_c` 溢出，或两个权重的长度不是它——都返回
    /// `Code::Invalid`。权重长度只是"个数对"还不够：形状错了的话每个头的块会整体错位，
    /// 输出的每个数都错，却不会有任何一处报错。
    pub fn new(
        heads: usize,
        head_dim: usize,
        latent: usize,
        rope: usize,
        up_key: &[S],
        up_value: &[S],
    ) -> Result<Self> {
        if heads.min(head_dim).min(latent).min(rope) == 0 {
            return Err(invalid(
                SITE,
                format!("MLA 超参必须非零（n_h = {heads}，d_h = {head_dim}，d_c = {latent}，d_h^R = {rope}）"),
            ));
        }
        let width = heads
            .checked_mul(head_dim)
            .and_then(|rows| rows.checked_mul(latent))
            .ok_or_else(|| {
                invalid(SITE, format!("n_h × d_h × d_c 溢出（{heads} × {head_dim} × {latent}）"))
            })?;
        if [up_key.len(), up_value.len()] != [width; 2] {
            return Err(invalid(SITE, format!(
                    "投影长度必须都是 n_h × d_h × d_c = {width}，得到 ({}, {})",
                    up_key.len(),
                    up_value.len()
                )));
        }
        Ok(Self {
            heads,
            head_dim,
            latent,
            rope,
            up_key: up_key.to_vec(),
            up_value: up_value.to_vec(),
        })
    }

    /// 头数 `n_h`。
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// 头宽 `d_h`。
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// 潜在宽 `d_c`。
    pub fn latent(&self) -> usize {
        self.latent
    }

    /// 解耦 RoPE 键宽 `d_h^R`。
    pub fn rope(&self) -> usize {
        self.rope
    }

    /// 一个 token 位置的缓存宽度 `d_c + d_h^R`。
    pub fn row_width(&self) -> usize {
        row_width(self.latent, self.rope)
    }

    /// 同口径的稠密 KV 行宽 `2 · n_h · d_h`。
    pub fn dense_width(&self) -> usize {
        dense_width(self.heads, self.head_dim)
    }

    /// 两条路径的行宽之比。
    pub fn ratio(&self) -> f64 {
        ratio(self.heads, self.head_dim, self.latent, self.rope)
    }

    /// 键侧投影 `W^{UK}`，行主序 `[n_h · d_h, d_c]`。
    pub fn up_key(&self) -> &[S] {
        &self.up_key
    }

    /// 值侧投影 `W^{UV}`，行主序 `[n_h · d_h, d_c]`。
    pub fn up_value(&self) -> &[S] {
        &self.up_value
    }

    /// 按本层几何开一张空缓存。
    pub fn empty_cache(&self) -> Latent<S> {
        Latent::empty(self.latent, self.rope)
    }

    /// 由两段现成数据装缓存：`latent` 是 `[seq, d_c]`、`rope` 是 `[seq, d_h^R]`。
    ///
    /// # Errors
    ///
    /// 任一段长度不是对应宽度的整数倍、或两段算出的 `seq` 不同，都返回 `Code::Invalid`。
    pub fn cache(&self, latent: &[S], rope: &[S]) -> Result<Latent<S>> {
        let seq = self.count(latent, self.latent)?;
        if self.count(rope, self.rope)? != seq {
            return Err(invalid(SITE, format!(
                    "两段的 seq 不一致：潜在段 {} 行、RoPE 段 {} 行",
                    seq,
                    self.count(rope, self.rope)?
                )));
        }
        Ok(Latent {
            latent_width: self.latent,
            rope_width: self.rope,
            latent: latent.to_vec(),
            rope: rope.to_vec(),
        })
    }

    /// 这一段能装几个位置；不是整数倍就是错的几何。
    fn count(&self, segment: &[S], width: usize) -> Result<usize> {
        if !segment.len().is_multiple_of(width) {
            return Err(invalid(SITE, format!("段长 {} 不是宽度 {width} 的整数倍", segment.len())));
        }
        Ok(segment.len() / width)
    }

    /// 查询 `(q^{no-nope}, q^R)` 在位置 `t` 的输出，`n_h · d_h` 个元素按头拼接。
    ///
    /// 因果性来自寻址本身：只累加 `0..=t`，`t` 之后的缓存行根本不读。两条路径给出同一个数
    /// （单测按 `1e-5` 对拍），差别只在"什么时候乘投影"：吸收式乘在查询/输出侧（各一次），
    /// 升维参考乘在每个 token 上（`2 · d_h · d_c` 每 token 每头）。
    ///
    /// # Errors
    ///
    /// 缓存不是本层的几何、`t` 超出已缓存位置、查询宽度不是 `n_h · d_h` / `n_h · d_h^R`——
    /// 都返回 `Code::Invalid`。位置越界不截断：截断会把"还没生成到那一步"读成"那一步已算完"。
    pub fn attend(
        &self,
        cache: &Latent<S>,
        q_nope: &[S],
        q_rope: &[S],
        t: usize,
        path: Path,
    ) -> Result<Vec<S>> {
        self.check_cache(cache, q_nope, q_rope, t)?;
        let mut out = vec![S::ZERO; self.heads * self.head_dim];
        for head in 0..self.heads {
            let mut scores = self.scores(head, cache, q_nope, q_rope, t, path);
            softmax_row(&mut scores);
            let block = self.combine(head, cache, &scores, path);
            out[head * self.head_dim..(head + 1) * self.head_dim].copy_from_slice(&block);
        }
        Ok(out)
    }

    /// 入口校验：缓存几何、位置与两条查询的宽度。
    fn check_cache(&self, cache: &Latent<S>, q_nope: &[S], q_rope: &[S], t: usize) -> Result<()> {
        if [cache.latent_width(), cache.rope_width()] != [self.latent, self.rope] {
            return Err(invalid(SITE, format!(
                    "缓存的几何 ({}, {}) 与本层 ({}, {}) 不同：不能拿别的层的潜在缓存来算",
                    cache.latent_width(),
                    cache.rope_width(),
                    self.latent,
                    self.rope
                )));
        }
        if t >= cache.seq()
            || q_nope.len() != self.heads * self.head_dim
            || q_rope.len() != self.heads * self.rope
        {
            return Err(invalid(SITE, format!(
                    "位置 {t} 或查询宽度 ({}, {}) 越界（seq = {}，要 ({}, {})）",
                    q_nope.len(),
                    q_rope.len(),
                    cache.seq(),
                    self.heads * self.head_dim,
                    self.heads * self.rope
                )));
        }
        Ok(())
    }

    /// 第 `h` 头在 `0..=t` 上的分数，末乘 `1/√(d_h + d_h^R)`。
    fn scores(
        &self,
        head: usize,
        cache: &Latent<S>,
        q_nope: &[S],
        q_rope: &[S],
        t: usize,
        path: Path,
    ) -> Vec<S> {
        let scale = S::ONE.div(S::from_f64(((self.head_dim + self.rope) as f64).sqrt()));
        let query = &q_nope[head * self.head_dim..(head + 1) * self.head_dim];
        let rope_query = &q_rope[head * self.rope..(head + 1) * self.rope];
        let key = self.head_key(head);
        let absorbed = match path {
            Path::Absorb => Some(matvec_t(key, self.head_dim, self.latent, query)),
            Path::Upref => None,
        };
        let (latent, rope) = cache.rows();
        (0..=t)
            .map(|i| {
                let c = &latent[i * self.latent..(i + 1) * self.latent];
                let nope = match &absorbed {
                    Some(absorbed) => dot(absorbed, c),
                    None => dot(query, &matvec(key, self.head_dim, self.latent, c)),
                };
                let decoupled = dot(rope_query, &rope[i * self.rope..(i + 1) * self.rope]);
                nope.add(decoupled).mul(scale)
            })
            .collect()
    }

    /// 第 `h` 头按权重合成 `d_h` 个输出元素。
    fn combine(&self, head: usize, cache: &Latent<S>, weights: &[S], path: Path) -> Vec<S> {
        let value = self.value_head(head);
        let (latent, _) = cache.rows();
        match path {
            // 吸收式：先在潜在空间加权求和（每 token 一次 `d_c` 线性组合），再乘一次 `W^{UV}`。
            Path::Absorb => {
                let mut latent_sum = vec![S::ZERO; self.latent];
                for (i, &weight) in weights.iter().enumerate() {
                    accumulate(
                        &mut latent_sum,
                        &latent[i * self.latent..(i + 1) * self.latent],
                        weight,
                    );
                }
                matvec(value, self.head_dim, self.latent, &latent_sum)
            }
            // 升维参考：每 token 先还原 `v_h` 再按标准注意力加权求和。
            Path::Upref => {
                let mut out = vec![S::ZERO; self.head_dim];
                for (i, &weight) in weights.iter().enumerate() {
                    let v = matvec(
                        value,
                        self.head_dim,
                        self.latent,
                        &latent[i * self.latent..(i + 1) * self.latent],
                    );
                    accumulate(&mut out, &v, weight);
                }
                out
            }
        }
    }

    /// 第 `h` 头的键侧块 `[d_h, d_c]`。
    fn head_key(&self, head: usize) -> &[S] {
        &self.up_key[self.offset(head)..self.offset(head + 1)]
    }

    /// 第 `h` 头的值侧块 `[d_h, d_c]`。
    fn value_head(&self, head: usize) -> &[S] {
        &self.up_value[self.offset(head)..self.offset(head + 1)]
    }

    /// 第 `head` 头块在扁平权重里的起点。
    fn offset(&self, head: usize) -> usize {
        head * self.head_dim * self.latent
    }
}

/// `W x`：`w` 是行主序 `[rows, cols]`，`x` 长 `cols`，返回 `rows` 个元素。
fn matvec<S: Scalar>(w: &[S], rows: usize, cols: usize, x: &[S]) -> Vec<S> {
    (0..rows)
        .map(|r| dot(&w[r * cols..(r + 1) * cols], x))
        .collect()
}

/// `Wᵀ x`：`w` 是行主序 `[rows, cols]`，`x` 长 `rows`，返回 `cols` 个元素（吸收式的全部秘密）。
fn matvec_t<S: Scalar>(w: &[S], rows: usize, cols: usize, x: &[S]) -> Vec<S> {
    let mut out = vec![S::ZERO; cols];
    for (r, &value) in x.iter().enumerate().take(rows) {
        accumulate(&mut out, &w[r * cols..(r + 1) * cols], value);
    }
    out
}

#[cfg(test)]
#[path = "../tests/mla_tests.rs"]
mod tests;

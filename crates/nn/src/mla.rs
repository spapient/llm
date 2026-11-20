//! DeepSeek MLA 的潜在 KV（§5.8.1）：每 token 每层只缓存 `c^KV` 与解耦 RoPE 键 `k^R`。
//!
//! 标准 MHA/GQA 缓存的是一个位置的 `K` 与 `V`（`2 · n_h · d_h` 个元素）。MLA 把这两段改写成
//! 两个低秩投影的**输入**：`k_h = W^{UK}_h c`、`v_h = W^{UV}_h c`，其中 `c ∈ R^{d_c}`；解耦的
//! RoPE 键 `k^R ∈ R^{d_h^R}` 不进 `W^{UK}`，直接加进分数。缓存里因此只剩 `d_c + d_h^R`——
//! `k^{no-nope}` 与 `V` 都由 `c` 现算，存它们等于把低秩投影的结果又存了一遍。
//!
//! 于是同一次注意力有两条**数学等价**的算法，本模块都实现：
//!
//! - [`Path::Upref`]（升维参考）：每步把 `c` 显式还原成 `k_h` / `v_h`，再按标准注意力算。
//!   它是数值对拍用的那条，也是"潜在缓存到底等不等价"唯一能验证的写法。
//! - [`Path::Absorb`]（吸收）：`q^T k_h = (W^{UK}_h{}^T q)^T c`，把 `W^{UK}` 吸到查询侧、
//!   `W^{UV}` 吸到输出侧；每 token 只做一次 `d_c` 点积与一次 `d_h^R` 点积，`V` 始终不必还原。
//!   这是解码期真跑的那条：吸收之后缓存的读取宽度就是 `d_c + d_h^R` 本身。
//!
//! 与 [`crate::Attention`] / [`crate::BlockTable`] 的分工：那两者分别压**头数**与**序列长度**，
//! 本模块压**每个 token 的 KV 宽度**。三者不叠加——一次前向只选一条（`docs/lld.md` 规则 0.9）。
//!
//! 纯几何量（行宽、比例）走模块级函数，与 [`crate::csa::effective_len`] 同形：V3 的行宽算得出来
//! 就不该先分配 8M 个权重再算。`f32` 供前向、`f64` 供对拍，共用一份泛型实现（[`op::scan::Scalar`]）。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::{accumulate, dot, softmax_row};

/// 报错出处：几何、长度与寻址越界都在入口失败，不留到数值阶段。
const SITE: &str = "nn::mla";

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
///
/// 分母为零（几何不成立）时返回 `f64::INFINITY` 而不是 panic：这是个记账函数，调用方要的是
/// "这个配置省不省"，不是"配置对不对"——配置对不对由 [`Mla::new`] 判。
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
///
/// 两段分开存而不是拼成一行，是因为它们的宽度由两个不同的超参给，且只有 `c^KV` 参与低秩投影——
/// 拼起来会让人误以为 `k^R` 也要过 `W^{UK}`。交给 `engine` 的页池时按
/// [`Latent::row_width`] 当行宽即可，两段本来就是连续的。
///
/// 从 [`Mla::empty_cache`] 起步、按 token [`Latent::push`]：解码每一步追加一行，这与 §5.8.1
/// "只缓存 `(c^KV, k^R)`" 是同一句话。
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
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "潜在行宽度不符：给的是 ({}, {})，缓存要 ({}, {})",
                    latent.len(),
                    rope.len(),
                    self.latent_width,
                    self.rope_width
                ),
            )
            .with(SITE));
        }
        self.latent.extend_from_slice(latent);
        self.rope.extend_from_slice(rope);
        Ok(())
    }
}

/// MLA 层：超参 `(n_h, d_h, d_c, d_h^R)` 加上两条低秩投影 `W^{UK}` / `W^{UV}`。
///
/// 两个权重都按 `[n_h · d_h, d_c]` 行主序存放（第 `h` 个头的块是 `[h·d_h, (h+1)·d_h)` 这
/// `d_h` 行）：`k_h = W^{UK}_h c`。`W^{UK}_h` 与 `W^{UV}_h` 各自的 `d_h` 行共享同一个 `c`，
/// 所以按行存就能一次把整头算完，不需要为每个头切一份子矩阵。
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
            return Err(Error::new(
                Code::Invalid,
                format!("MLA 超参必须非零（n_h = {heads}，d_h = {head_dim}，d_c = {latent}，d_h^R = {rope}）"),
            )
            .with(SITE));
        }
        let width = heads
            .checked_mul(head_dim)
            .and_then(|rows| rows.checked_mul(latent))
            .ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!("n_h × d_h × d_c 溢出（{heads} × {head_dim} × {latent}）"),
                )
                .with(SITE)
            })?;
        if [up_key.len(), up_value.len()] != [width; 2] {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "投影长度必须都是 n_h × d_h × d_c = {width}，得到 ({}, {})",
                    up_key.len(),
                    up_value.len()
                ),
            )
            .with(SITE));
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
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "两段的 seq 不一致：潜在段 {} 行、RoPE 段 {} 行",
                    seq,
                    self.count(rope, self.rope)?
                ),
            )
            .with(SITE));
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
            return Err(Error::new(
                Code::Invalid,
                format!("段长 {} 不是宽度 {width} 的整数倍", segment.len()),
            )
            .with(SITE));
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
        self.check(cache, q_nope, q_rope, t)?;
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
    fn check(&self, cache: &Latent<S>, q_nope: &[S], q_rope: &[S], t: usize) -> Result<()> {
        if [cache.latent_width(), cache.rope_width()] != [self.latent, self.rope] {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓存的几何 ({}, {}) 与本层 ({}, {}) 不同：不能拿别的层的潜在缓存来算",
                    cache.latent_width(),
                    cache.rope_width(),
                    self.latent,
                    self.rope
                ),
            )
            .with(SITE));
        }
        if t >= cache.seq()
            || q_nope.len() != self.heads * self.head_dim
            || q_rope.len() != self.heads * self.rope
        {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "位置 {t} 或查询宽度 ({}, {}) 越界（seq = {}，要 ({}, {})）",
                    q_nope.len(),
                    q_rope.len(),
                    cache.seq(),
                    self.heads * self.head_dim,
                    self.heads * self.rope
                ),
            )
            .with(SITE));
        }
        Ok(())
    }

    /// 第 `h` 头在 `0..=t` 上的分数，末乘 `1/√(d_h + d_h^R)`。
    ///
    /// 两条路径的分数**逐项相同**，只是乘投影的时机不同：吸收式先把 `W^{UK}` 的转置乘进查询，
    /// 于是每个 token 只剩一次 `d_c` 点积；升维参考每 token 先把 `c` 还原成 `k_h` 再点积。
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
mod tests {
    use super::*;

    /// 确定性有限取值：`[−0.5, 0.5)` 的整数派生物，避开 0 与 1 这类退化值。
    fn values<S: Scalar>(n: usize, salt: usize) -> Vec<S> {
        (0..n)
            .map(|i| S::from_f64(((i * 37 + salt * 11) % 13) as f64 / 13.0 - 0.5))
            .collect()
    }

    /// 第 `row` 行按 `scale` 缩放的取值：用来造"未来行被改过"的对照组。
    fn scaled<S: Scalar>(width: usize, row: usize, scale: f64) -> Vec<S> {
        values::<S>(width, row)
            .into_iter()
            .map(|v| v.mul(S::from_f64(scale)))
            .collect()
    }

    /// 2 头、`d_h = 3`、`d_c = 4`、`d_h^R = 2` 的小层：`n_h · d_h · d_c = 24`。
    fn layer() -> Mla<f32> {
        Mla::new(2, 3, 4, 2, &values(24, 1), &values(24, 2)).expect("层")
    }

    /// 顺序追加 `seq` 行；每行的取值随行号变。
    fn cache(mla: &Mla<f32>, seq: usize) -> Latent<f32> {
        let mut cache = mla.empty_cache();
        for t in 0..seq {
            cache
                .push(&scaled(mla.latent(), t, 1.0), &scaled(mla.rope(), t, 1.0))
                .expect("追加");
        }
        cache
    }

    /// V3 的行宽记账：`d_c + d_h^R = 576`，对稠密 `2 · n_h · d_h = 32768` 是 1.76%。
    #[test]
    fn the_v3_latent_row_is_a_fraction_of_the_dense_row() {
        assert_eq!(row_width(V3_LATENT, V3_ROPE), 576);
        assert_eq!(dense_width(V3_HEADS, V3_HEAD_DIM), 32_768);
        assert!((ratio(V3_HEADS, V3_HEAD_DIM, V3_LATENT, V3_ROPE) - 0.0176).abs() < 5e-5);
        // 61 层 × 576 个元素 × 2 字节 = 70272 字节/token，与 lld §5.8.1 的记账一致。
        assert_eq!(V3_LAYERS * row_width(V3_LATENT, V3_ROPE) * 2, 70_272);
        // 几何不成立时不 panic：记账函数只回答"省不省"，配置对不对由构造器判。
        assert!(ratio(0, 0, 4, 2).is_infinite());
    }

    /// 缓存里只有两段：潜在量与解耦 RoPE 键，每 token 正好 `d_c + d_h^R` 个元素。
    #[test]
    fn only_the_latent_and_the_rope_key_are_cached() {
        let mla = layer();
        let cache = cache(&mla, 5);
        assert_eq!(cache.seq(), 5);
        assert_eq!(cache.cache_elements(), 5 * (4 + 2));
        assert_eq!(cache.cache_elements(), 5 * mla.row_width());
        assert!(
            cache.cache_elements() < 5 * mla.dense_width(),
            "潜在行必须比同头的稠密 K/V 行窄"
        );
        // 行视图就是两段本身：没有 `k^{no-nope}`、也没有 `V` 的位置。
        let (latent, rope) = cache.row(2).expect("第 2 行");
        assert_eq!((latent.len(), rope.len()), (4, 2));
        assert_eq!(latent, scaled::<f32>(4, 2, 1.0).as_slice());
    }

    /// 吸收式与升维参考给出同一个数：这是"潜在缓存等价"的唯一可验证形态。
    #[test]
    fn absorb_matches_the_upref_reference() {
        let mla = layer();
        let cache = cache(&mla, 5);
        let q_nope = values::<f32>(6, 7);
        let q_rope = values::<f32>(4, 9);
        let absorbed = mla
            .attend(&cache, &q_nope, &q_rope, 4, Path::Absorb)
            .expect("吸收");
        let upref = mla
            .attend(&cache, &q_nope, &q_rope, 4, Path::Upref)
            .expect("升维");
        assert_eq!(absorbed.len(), 6);
        for (a, b) in absorbed.iter().zip(&upref) {
            assert!((a - b).abs() < 1e-5, "吸收式 {a} 与升维参考 {b} 不等");
        }
    }

    /// 同一份泛型实现在 `f64` 下同样等价：两条路径不因精度复制代码。
    #[test]
    fn the_two_paths_agree_in_f64_too() {
        let mla: Mla<f64> = Mla::new(2, 3, 4, 2, &values(24, 1), &values(24, 2)).expect("层");
        let mut cache = mla.empty_cache();
        for t in 0..5 {
            cache
                .push(&scaled(mla.latent(), t, 1.0), &scaled(mla.rope(), t, 1.0))
                .expect("追加");
        }
        let q_nope: Vec<f64> = values(6, 7);
        let q_rope: Vec<f64> = values(4, 9);
        let absorbed = mla
            .attend(&cache, &q_nope, &q_rope, 3, Path::Absorb)
            .expect("吸收");
        let upref = mla
            .attend(&cache, &q_nope, &q_rope, 3, Path::Upref)
            .expect("升维");
        for (a, b) in absorbed.iter().zip(&upref) {
            assert!(
                (a - b).abs() < 1e-12,
                "f64 下吸收式 {a} 与升维参考 {b} 不等"
            );
        }
    }

    /// 因果性：`t` 之后的行怎么改都不影响 `t` 的输出，`t` 之前（含 `t`）的一改就影响。
    #[test]
    fn future_tokens_cannot_reach_the_past() {
        let mla = layer();
        let q_nope = values::<f32>(6, 7);
        let q_rope = values::<f32>(4, 9);
        let base = cache(&mla, 5);

        let mut future_changed = mla.empty_cache();
        for t in 0..5 {
            let scale = if t > 2 { 7.0 } else { 1.0 };
            future_changed
                .push(
                    &scaled(mla.latent(), t, scale),
                    &scaled(mla.rope(), t, scale),
                )
                .expect("追加");
        }
        let mut past_changed = mla.empty_cache();
        for t in 0..5 {
            let scale = if t == 0 { 3.0 } else { 1.0 };
            past_changed
                .push(
                    &scaled(mla.latent(), t, scale),
                    &scaled(mla.rope(), t, scale),
                )
                .expect("追加");
        }
        for path in [Path::Absorb, Path::Upref] {
            assert_eq!(
                mla.attend(&base, &q_nope, &q_rope, 2, path).expect("基线"),
                mla.attend(&future_changed, &q_nope, &q_rope, 2, path)
                    .expect("未来改动"),
                "{} 路径读到了位置 2 之后的行",
                path.as_str()
            );
            assert_ne!(
                mla.attend(&base, &q_nope, &q_rope, 2, path).expect("基线"),
                mla.attend(&past_changed, &q_nope, &q_rope, 2, path)
                    .expect("过去改动"),
                "{} 路径没有读到位置 0 的行",
                path.as_str()
            );
        }
    }

    /// 解耦 RoPE 键直接进分数：换掉 `q^R` 就换输出，不能只看 `c^KV`。
    #[test]
    fn the_rope_key_enters_the_score_directly() {
        let mla = layer();
        let cache = cache(&mla, 3);
        let q_nope = values::<f32>(6, 7);
        let q_rope = values::<f32>(4, 9);
        let zeros = vec![0.0f32; 4];
        assert_ne!(
            mla.attend(&cache, &q_nope, &q_rope, 2, Path::Absorb)
                .expect("有 RoPE 查询"),
            mla.attend(&cache, &q_nope, &zeros, 2, Path::Absorb)
                .expect("无 RoPE 查询")
        );
    }

    /// 位置与查询宽度都在入口校验：越界不截断、宽度不符不猜。
    #[test]
    fn a_query_or_position_outside_the_geometry_is_rejected() {
        let mla = layer();
        let cache = cache(&mla, 5);
        let q_nope = values::<f32>(6, 7);
        let q_rope = values::<f32>(4, 9);
        assert!(
            mla.attend(&cache, &q_nope, &q_rope, 5, Path::Absorb)
                .is_err()
        );
        assert!(
            mla.attend(&cache, &q_nope[..5], &q_rope, 4, Path::Absorb)
                .is_err()
        );
        assert!(
            mla.attend(&cache, &q_nope, &q_rope[..3], 4, Path::Absorb)
                .is_err()
        );
        assert!(
            mla.attend(&mla.empty_cache(), &q_nope, &q_rope, 0, Path::Absorb)
                .is_err(),
            "空缓存没有可寻址的位置"
        );
        let other = Mla::new(2, 3, 3, 2, &values(18, 1), &values(18, 2)).expect("另一层");
        let foreign = other.cache(&values(18, 1), &values(12, 2)).expect("缓存");
        assert!(
            mla.attend(&foreign, &q_nope, &q_rope, 3, Path::Absorb)
                .is_err(),
            "别的层的潜在缓存不能拿来算"
        );
    }

    /// 构造期把长度与几何一次判清：错的形状活到寻址期只会整体错位。
    #[test]
    fn construction_checks_every_length() {
        let key = values::<f32>(24, 1);
        let value = values::<f32>(24, 2);
        assert!(Mla::new(0, 3, 4, 2, &key, &value).is_err());
        assert!(Mla::new(2, 0, 4, 2, &key, &value).is_err());
        assert!(Mla::new(2, 3, 0, 2, &key, &value).is_err());
        assert!(Mla::new(2, 3, 4, 0, &key, &value).is_err());
        assert!(Mla::new(2, 3, 4, 2, &key[..23], &value).is_err());
        assert!(Mla::new(2, 3, 4, 2, &key, &values::<f32>(25, 2)).is_err());

        let mla = layer();
        assert!(mla.cache(&values::<f32>(23, 1), &values(4, 2)).is_err());
        assert!(mla.cache(&values::<f32>(24, 1), &values(3, 2)).is_err());
        // 两段各自都是整数倍，但算出的 seq 不同：6 行 vs 4 行。
        assert!(mla.cache(&values::<f32>(24, 1), &values(8, 2)).is_err());

        let mut cache = mla.empty_cache();
        assert!(cache.push(&values::<f32>(3, 1), &values(2, 1)).is_err());
        assert!(cache.push(&values::<f32>(4, 1), &values(1, 1)).is_err());
        assert_eq!(cache.seq(), 0, "失败的追加不许推进一半的缓存");
        cache.push(&values(4, 1), &values(2, 1)).expect("追加");
        assert_eq!(cache.seq(), 1);
    }

    /// 路径短名稳定：报告与参数表按这个名字对齐。
    #[test]
    fn paths_have_stable_names() {
        assert_eq!(Path::Absorb.as_str(), "absorb");
        assert_eq!(Path::Upref.as_str(), "upref");
    }
}

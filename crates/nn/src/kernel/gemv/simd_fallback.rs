//! stable 退路：与 nightly 版的 `super::simd` 同名同签名，只是不做显式车道。
//!
//! 两边的差别只有「向量化没向量化」，语义一模一样：
//!
//! * [`Load::one`] + [`elementwise`] 保持与向量版**相同的分段形状**——`WIDE` 个元素一段、
//!   `CHAINS` 条独立累加链轮转。段内逐条 `mul_add`，于是每条链累加的元素序列与向量版逐车道
//!   一一对应，stable 上 LLVM 也能把这份源码自动向量化成近乎同一组指令；两份实现只差最后那
//!   一步归约（`reduce_sum` 是树形，这里是顺序）。
//! * 打包格式的通道（`admit` / `forward_batch`）在 stable 上恒为不可达：调用方
//!   `gemv::packed` 本来就有标量叶 `forward::forward_rows_scaled` 兜底，那条路径一直都在。

use core::convert::Infallible;

use super::ScaledRow;

/// 宽档车道数，与向量版一致：决定 [`elementwise`] 的分段宽度。
const WIDE: usize = 16;

/// 独立累加链数，与向量版一致：一条链的乘加延迟约 4 拍，4 条刚好把流水填满。
const CHAINS: usize = 4;

/// 行并行度：一次算 [`COLS`] 行，一份 `x` 段摊到这 [`COLS`] 行上。
///
/// 实测每多一条行流就多一条 DRAM 预取流，2026-11-22 扫描 2/4/8 得到 55.1／48.9／44.7 GiB/s，
/// 故取 2：`x` 复用与流数的最优点，再往上就是流数占主导。
pub(crate) const COLS: usize = 2;

/// 段宽：一条累加链一次推进的元素数，与 [`WIDE`] 同义但可独立调（见 [`COLS`] 的扫描）。
const SPAN: usize = 8;

/// 每条行的独立累加链数：行并行度只有 [`COLS`]，链不够就喂不满 4 拍的乘加延迟。
const DEPTH: usize = 2;

/// 逐元素通道的权重取法：稠密 `f32` 直取，对称 `int8` 按二补数重新解释。
///
/// 一条通道只由权重元素类型选，[`elementwise`] 因此对两种格式是同一段代码。
pub(crate) trait Load: Copy {
    /// 单个元素取成 `f32`。
    fn one(elem: Self) -> f32;
}

impl Load for f32 {
    fn one(elem: Self) -> f32 {
        elem
    }
}

impl Load for u8 {
    /// `u8 → i8` 是位模式重解释，`i8 → f32` 是有符号展宽。
    fn one(elem: Self) -> f32 {
        (elem as i8) as f32
    }
}

/// [`COLS`] 行并行点积：`w` 是连续 [`COLS`] 条行、每行 `m` 个元素，结果写进 `out` 的 [`COLS`] 个槽。
///
/// 形状是 `COLS × DEPTH × SPAN` 的累加器：一行一份 `x` 段摊给 [`COLS`] 行共用，行内再拆
/// [`DEPTH`] 条独立链去盖住乘加的 4 拍延迟。稳定版没有显式车道，靠的是循环体形状本身
/// 够整齐，让 LLVM 自己向量化——所以 `SPAN · DEPTH` 必须是 2 的幂，否则整段退化成标量
/// （实测 `COLS = 3` 掉到 30 GiB/s）。
///
/// 三个常数都不是猜的：2026-11-22 在同机同口径下扫过 `COLS ∈ {2,4,8}`（55.1／48.9／44.7）
/// 与 `SPAN ∈ {8,16,32,64}`、`DEPTH ∈ {1,2,3,4}`，定格在 2／8／2 —— 再大的 `COLS` 每多一条行流
/// 就多一条 DRAM 预取流，与 `x` 的流争抢，收益反转为负。
pub(crate) fn elementwise_four<T: Load>(w: &[T], x: &[f32], m: usize, out: &mut [f32]) {
    let mut acc = [[[0.0f32; SPAN]; DEPTH]; COLS];
    let group = SPAN * DEPTH;
    let full = m - m % group;
    let (x_head, x_tail) = x.split_at(full);
    for (g, xg) in x_head.as_chunks::<{ SPAN * DEPTH }>().0.iter().enumerate() {
        commit(&mut acc, w, m, xg, g * group);
    }
    for (r, (o, chain)) in out.iter_mut().zip(acc).enumerate() {
        *o = finish(w, m, r, full, x_tail, chain);
    }
}

/// 把一个 `x` 组（`SPAN · DEPTH` 个元素）乘进 [`COLS`] 行 × [`DEPTH`] 链的累加器。
///
/// 单独成函数只为把嵌套压到门禁的 5 以内；`#[inline]` 保证与内联版逐指令一致——这是
/// 排版，不是抽象层。
#[inline]
fn commit<T: Load>(
    acc: &mut [[[f32; SPAN]; DEPTH]; COLS],
    w: &[T],
    m: usize,
    xg: &[f32],
    at: usize,
) {
    for (r, row) in acc.iter_mut().enumerate() {
        for (d, chain) in row.iter_mut().enumerate() {
            let xs = &xg[d * SPAN..][..SPAN];
            let ws = &w[r * m + at + d * SPAN..][..SPAN];
            for (lane, (wv, xv)) in ws.iter().zip(xs).enumerate() {
                chain[lane] = T::one(*wv).mul_add(*xv, chain[lane]);
            }
        }
    }
}

/// 归约一条链并吃掉行尾（`m` 不是 `SPAN · DEPTH` 整数倍时剩下的那几个）。
#[inline]
fn finish<T: Load>(
    w: &[T],
    m: usize,
    r: usize,
    full: usize,
    x_tail: &[f32],
    chain: [[f32; SPAN]; DEPTH],
) -> f32 {
    let mut total: f32 = chain.iter().flatten().sum();
    for (wc, xc) in w[r * m + full..][..m - full].iter().zip(x_tail) {
        total = T::one(*wc).mul_add(*xc, total);
    }
    total
}

/// 连续点积：`CHAINS` 条独立累加链轮转，行尾归约一次。
pub(crate) fn elementwise<T: Load>(w: &[T], x: &[f32]) -> f32 {
    let (w_chunks, w_tail) = w.as_chunks::<WIDE>();
    let (x_chunks, x_tail) = x.as_chunks::<WIDE>();
    let mut acc = [[0.0f32; WIDE]; CHAINS];
    for (i, (w, x)) in w_chunks.iter().zip(x_chunks).enumerate() {
        let chain = &mut acc[i % CHAINS];
        for (k, (w, x)) in w.iter().zip(x).enumerate() {
            chain[k] = T::one(*w).mul_add(*x, chain[k]);
        }
    }
    let mut total: f32 = acc.iter().flatten().sum();
    for (w, x) in w_tail.iter().zip(x_tail) {
        total = T::one(*w).mul_add(*x, total);
    }
    total
}

/// 打包格式的通道参数：stable 上没有向量通道，恒为 `None`，调用方落到标量叶。
pub(crate) fn admit(
    _cpb: usize,
    _n: usize,
    _ctx: ScaledRow,
    _dequant: &[f32],
) -> Option<Infallible> {
    None
}

/// 只有 [`admit`] 返回 `Some` 才会走到这里，而 stable 上它永远不返回 `Some`。
pub(crate) fn forward_batch(
    _rows: &[&[f32]],
    _weights: &[u8],
    _scales: &[f32],
    never: Infallible,
    _out: &mut [f32],
) {
    match never {}
}

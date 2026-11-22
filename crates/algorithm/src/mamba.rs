//! Mamba-3 的指数梯形状态空间（§5.6，`exponential_trapezoid` 那一格）。
//!
//! 十一款标杆里这一款此前只有别的形态：`kda` 是通道级 delta 遗忘记忆（写入是「纠错替换」，一步只
//! 留最新那个键），本条是**累加**型状态空间——同一个输入写两次是叠加，不是覆盖。两者判据正交，
//! 因此不合并成一个模块，也不让其中一个借另一个的语义。
//!
//! # 三个决定
//!
//! **状态是复值的、存成 2×2 实值块**：`λ = λ_re + i·ω` 时状态不再是实标量，而是一对实值分量，
//! 一步就是 [`op::scan::PlaneStep`] 的一次施加（旋转 + 衰减）。复值状态不必引入复数类型，于是「状态宽减半、
//! 同困惑度」那条实证红利的实现形态是可测的：每个状态维占 2 个实值分量，`pairs` 取一半即同宽。
//!
//! **离散化取双抽头梯形**：[`exp_trapezoid`] 用 `x_{t−1}` 与 `x_t` 的平均，步内输入不是常数——这正是
//! `exponential_trapezoid` 这个名字所指的算式；左端常数那条（[`op::scan::exp_zero_order`]）是另一个名字、另一条
//! 式子，`scan_f32` 算子用它。层内记着上一个 token 的输入（[`Carry::taps`]），所以梯形不需要调用方
//! 额外传参。
//!
//! **选择是低秩多输入多输出**：`B`、`C` 各有 `rank` 组（各长 `pairs`）。状态更新对秩求和，
//! 输出对秩求和——`rank = 1` 就是单输入单输出的经典形态，`rank > 1` 是 Mamba-2 之后的低秩 MIMO。
//! 秩不能超过状态维：行多于列的选择矩阵必有线性相关，多出来的那几组只是同一个方向的重复。
//!
//! 读数用**本 token 更新之后**的状态（与 `kda` 的读法一致），因此单个 token 读得到自己写下的东西；
//! 跨 token 的因果由调用次序保证，不由本模块合成。

use crate::invalid;
use diagnostic::Result;
use op::scan::{Scalar, exp_trapezoid};

use crate::csa::finite;

/// 报错出处：几何、步长与选择量都在入口失败，不留到状态更新中途。
const SITE: &str = "algorithm::mamba";

/// 一层指数梯形状态空间：`width` 个通道，每通道 `pairs` 个复值状态维、`rank` 组选择。
#[derive(Debug, Clone, PartialEq)]
pub struct Mamba<S: Scalar> {
    width: usize,
    pairs: usize,
    rank: usize,
    decay: Vec<S>,
    phase: Vec<S>,
    skip: Vec<S>,
}

/// 跨步携带的状态：`width × pairs` 个复值块（`2` 个实值分量）加上上一个 token 的输入。
#[derive(Debug, Clone, PartialEq)]
pub struct Carry<S: Scalar> {
    planes: Vec<S>,
    taps: Vec<S>,
}

impl<S: Scalar> Carry<S> {
    /// 状态的实值视图：按 `[通道, 状态维, 实/虚]` 行主序铺开。
    pub fn planes(&self) -> &[S] {
        &self.planes
    }

    /// 上一个 token 的输入（双抽头梯形的左端）。
    pub fn taps(&self) -> &[S] {
        &self.taps
    }
}

impl<S: Scalar> Mamba<S> {
    /// 由宽度、状态维数、秩与三组逐维参数构造一层。
    ///
    /// # Errors
    ///
    /// 任一维为零、`rank > pairs`、`decay` / `phase` 长度不是 `width × pairs`、`skip` 长度不是
    /// `width`、`decay` 现正值、尺寸相乘溢出，或任一参数非有限——都返回 `Code::Invalid`。
    pub fn new(
        width: usize,
        pairs: usize,
        rank: usize,
        decay: Vec<S>,
        phase: Vec<S>,
        skip: Vec<S>,
    ) -> Result<Self> {
        check_shape(width, pairs, rank)?;
        check_params(width, pairs, &decay, &phase, &skip)?;
        Ok(Self {
            width,
            pairs,
            rank,
            decay,
            phase,
            skip,
        })
    }

    accessors! {
        /// 通道数 `d`。
        width -> usize = width,
        /// 每通道的状态维数 `N`（复值，占 `2N` 个实值分量）。
        pairs -> usize = pairs,
        /// 选择秩 `R`。
        rank -> usize = rank,
    }

    /// 零状态：一个 token 都没喂过。
    pub fn state(&self) -> Carry<S> {
        Carry {
            planes: vec![S::ZERO; self.width * self.pairs * 2],
            taps: vec![S::ZERO; self.width],
        }
    }

    /// 一步：逐通道更新复值状态并读出 `y = Re(C h_t) + D·x_t`。
    ///
    /// `delta` 是逐通道步长 `Δ`（必须为正：零步长读不到这一次输入，负步长把时间倒着走），
    /// `select_b` / `select_c` 是低秩选择量（各 `rank × pairs`，实值）。
    ///
    /// # Errors
    ///
    /// 状态几何与本层不符、输入/步长/选择量宽度不符、步长不为正，或任一输入非有限——都返回
    /// `Code::Invalid`，且**在更新之前**判定，失败的那一步不会留下半截状态。
    pub fn step(
        &self,
        carry: &mut Carry<S>,
        input: &[S],
        delta: &[S],
        select_b: &[S],
        select_c: &[S],
    ) -> Result<Vec<S>> {
        self.check_carry(carry)?;
        self.check_inputs(input, delta, select_b, select_c)?;
        let mut output = vec![S::ZERO; self.width];
        for (channel, slot) in output.iter_mut().enumerate() {
            *slot = self.advance(
                carry,
                channel,
                input[channel],
                delta[channel],
                select_b,
                select_c,
            );
        }
        carry.taps.copy_from_slice(input);
        Ok(output)
    }

    /// 单通道的一步：`pairs` 个块各自 `h' = ā h + b̄·Σ_r B_r`，读数再对秩求和。
    fn advance(
        &self,
        carry: &mut Carry<S>,
        channel: usize,
        input: S,
        delta: S,
        select_b: &[S],
        select_c: &[S],
    ) -> S {
        let base = channel * self.pairs;
        let tap = carry.taps[channel];
        let mut read = S::ZERO;
        for state in 0..self.pairs {
            let at = base + state;
            let block = exp_trapezoid(self.decay[at], self.phase[at], delta, tap, input);
            let slot = &mut carry.planes[2 * at..2 * at + 2];
            let next = block.apply(
                [slot[0], slot[1]],
                mixed(select_b, self.pairs, self.rank, state),
            );
            slot.copy_from_slice(&next);
            read = read.add(mixed(select_c, self.pairs, self.rank, state).mul(next[0]));
        }
        read.add(self.skip[channel].mul(input))
    }

    /// 状态几何必须与本层一致：错配的状态是别的层的，读出来的数没有意义。
    fn check_carry(&self, carry: &Carry<S>) -> Result<()> {
        let cells = self.width * self.pairs * 2;
        if carry.planes.len() == cells && carry.taps.len() == self.width {
            return Ok(());
        }
        err!(
            "状态几何必须是 [{} 个实值分量, {} 个输入]，得到 [{}, {}]",
            cells,
            self.width,
            carry.planes.len(),
            carry.taps.len()
        )
    }

    /// 输入契约：宽度、有限性与步长符号一起判，且都排在第一次写入之前。
    fn check_inputs(&self, input: &[S], delta: &[S], select_b: &[S], select_c: &[S]) -> Result<()> {
        let selects = self.rank * self.pairs;
        let lengths = [
            ("输入", input.len(), self.width),
            ("步长", delta.len(), self.width),
            ("选择量 b", select_b.len(), selects),
            ("选择量 c", select_c.len(), selects),
        ];
        check_widths(&lengths)?;
        check_finite_all(&[input, delta, select_b, select_c])?;
        check_positive(delta)
    }
}

/// 选择量在某个状态维上对秩求和：`Σ_r S[r][state]`。
fn mixed<S: Scalar>(select: &[S], pairs: usize, rank: usize, state: usize) -> S {
    (0..rank).fold(S::ZERO, |acc, r| acc.add(select[r * pairs + state]))
}

/// 几何三问：任一维为零、秩超过状态维、尺寸相乘溢出。
fn check_shape(width: usize, pairs: usize, rank: usize) -> Result<()> {
    if [width, pairs, rank].contains(&0) {
        return err!(
            "宽度、状态维数与秩都必须非零，得到 width = {width}、pairs = {pairs}、rank = {rank}"
        );
    }
    if rank > pairs {
        return err!("秩 {rank} 不能超过状态维数 {pairs}：行多于列的选择矩阵必有线性相关");
    }
    if width
        .checked_mul(pairs)
        .and_then(|n| n.checked_mul(2))
        .is_none()
    {
        return err!("状态尺寸溢出（{width} × {pairs} × 2）");
    }
    Ok(())
}

/// 参数三问：两组逐状态维的参数与一组逐通道的直连系数，长度都对上后各自判取值。
fn check_params<S: Scalar>(
    width: usize,
    pairs: usize,
    decay: &[S],
    phase: &[S],
    skip: &[S],
) -> Result<()> {
    let cells = width * pairs;
    let lengths = [
        ("decay", decay.len(), cells),
        ("phase", phase.len(), cells),
        ("skip", skip.len(), width),
    ];
    check_widths(&lengths)?;
    check_spectrum(decay, phase)?;
    every_finite(skip)
}

/// 逐状态维的 `λ` 必须有限且实部 `≤ 0`。
fn check_spectrum<S: Scalar>(decay: &[S], phase: &[S]) -> Result<()> {
    for (cell, (&real, &imaginary)) in decay.iter().zip(phase).enumerate() {
        check_lambda(cell, real, imaginary)?;
    }
    Ok(())
}

/// 一个状态维的稳定性：非有限值报错在前，正实部报错在后（两者都可能同时成立）。
fn check_lambda<S: Scalar>(cell: usize, real: S, imaginary: S) -> Result<()> {
    if !finite(real) || !finite(imaginary) {
        return err!("第 {cell} 个状态维的 λ 非有限");
    }
    if real > S::ZERO {
        return err!(
            "第 {cell} 个状态维的 λ 实部必须 ≤ 0（正实部每步放大，几步就到 inf），得到 {real:?}"
        );
    }
    Ok(())
}

/// 一组「叫什么、拿到多少、该是多少」的宽度断言，逐条给出第一个不符的名字。
fn check_widths(expected: &[(&str, usize, usize)]) -> Result<()> {
    expected
        .iter()
        .try_for_each(|&(what, got, want)| check_len(what, got, want))
}

/// 若干段向量都要有限。分组而不是逐段调用，同样是为了把分支数收在一处。
fn check_finite_all<S: Scalar>(groups: &[&[S]]) -> Result<()> {
    groups.iter().try_for_each(|group| every_finite(group))
}

/// 整段向量都有限。比较对 `NaN` 全为假，所以放行一个 `NaN` 会让状态一路静默变成 `NaN`。
fn every_finite<S: Scalar>(values: &[S]) -> Result<()> {
    if values.iter().all(|&v| finite(v)) {
        return Ok(());
    }
    err!("输入、步长或选择量含非有限值")
}

/// 步长必须严格为正：零步长读不到这一次输入，负步长把时间倒着走。
fn check_positive<S: Scalar>(delta: &[S]) -> Result<()> {
    if let Some(channel) = delta.iter().position(|&v| v <= S::ZERO) {
        return err!(
            "第 {channel} 个通道的步长必须为正，得到 {:?}",
            delta[channel]
        );
    }
    Ok(())
}

/// 宽度不符一律 `Code::Invalid`：短了会静默读到别的通道，长了是多出来的数被丢掉。
fn check_len(what: &str, got: usize, want: usize) -> Result<()> {
    if got == want {
        return Ok(());
    }
    err!("{what}宽度必须是 {want}，得到 {got}")
}

#[cfg(test)]
#[path = "../tests/mamba_tests.rs"]
mod tests;

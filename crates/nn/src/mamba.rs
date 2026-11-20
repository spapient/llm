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

use diagnostic::{Code, Error, Result};
use op::scan::{Scalar, exp_trapezoid};

use crate::csa::finite;

/// 报错出处：几何、步长与选择量都在入口失败，不留到状态更新中途。
const SITE: &str = "nn::mamba";

/// 一层指数梯形状态空间：`width` 个通道，每通道 `pairs` 个复值状态维、`rank` 组选择。
///
/// `decay` 是逐（通道, 状态维）的 `λ` 实部，必须 `≤ 0`——正实数部让状态每步放大，几步就到 `inf`；
/// `phase` 是同一个位置的 `ω`。`skip` 是逐通道的直连系数 `D`。
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
///
/// 没有公开构造函数：状态只能由 [`Mamba::state`] 给出，于是「状态的形状」与「层的形状」不可能
/// 不一致；长度也与序列长无关——这是状态空间模型与注意力的结构性差别，由用例钉住而不是由注释承诺。
#[derive(Debug, Clone, PartialEq)]
pub struct Carry<S: Scalar> {
    planes: Vec<S>,
    taps: Vec<S>,
}

impl<S: Scalar> Carry<S> {
    /// 状态的实值视图：按 `[通道, 状态维, 实/虚]` 行主序铺开。
    ///
    /// 快照与检查都读这一份；写入口只有 [`Mamba::step`]，所以不存在「视图改了但状态没跟上」。
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

    /// 通道数 `d`。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 每通道的状态维数 `N`（复值，占 `2N` 个实值分量）。
    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// 选择秩 `R`。
    pub fn rank(&self) -> usize {
        self.rank
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
        self.check(carry)?;
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
    ///
    /// 逐通道独立，因此内层没有分支、也没有跨通道的读写依赖；梯形左端取本层的 `taps`。
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
    fn check(&self, carry: &Carry<S>) -> Result<()> {
        let cells = self.width * self.pairs * 2;
        if carry.planes.len() == cells && carry.taps.len() == self.width {
            return Ok(());
        }
        Err(fail(format!(
            "状态几何必须是 [{} 个实值分量, {} 个输入]，得到 [{}, {}]",
            cells,
            self.width,
            carry.planes.len(),
            carry.taps.len()
        )))
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
///
/// 秩那一问另有语义：行多于列的选择矩阵必有线性相关，多出来的那几组只是同一个方向的重复。
fn check_shape(width: usize, pairs: usize, rank: usize) -> Result<()> {
    if [width, pairs, rank].contains(&0) {
        return Err(fail(format!(
            "宽度、状态维数与秩都必须非零，得到 width = {width}、pairs = {pairs}、rank = {rank}"
        )));
    }
    if rank > pairs {
        return Err(fail(format!(
            "秩 {rank} 不能超过状态维数 {pairs}：行多于列的选择矩阵必有线性相关"
        )));
    }
    if width
        .checked_mul(pairs)
        .and_then(|n| n.checked_mul(2))
        .is_none()
    {
        return Err(fail(format!("状态尺寸溢出（{width} × {pairs} × 2）")));
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
        return Err(fail(format!("第 {cell} 个状态维的 λ 非有限")));
    }
    if real > S::ZERO {
        return Err(fail(format!(
            "第 {cell} 个状态维的 λ 实部必须 ≤ 0（正实部每步放大，几步就到 inf），得到 {real:?}"
        )));
    }
    Ok(())
}

/// 一组「叫什么、拿到多少、该是多少」的宽度断言，逐条给出第一个不符的名字。
///
/// 收集成表再判，是为了让调用方不必为每条长度各写一个 `?`——每个 `?` 都是一条分支，
/// 八条长度合起来会把一个只做校验的函数的圈复杂度顶到门限之上。
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
    Err(fail("输入、步长或选择量含非有限值".to_string()))
}

/// 步长必须严格为正：零步长读不到这一次输入，负步长把时间倒着走。
fn check_positive<S: Scalar>(delta: &[S]) -> Result<()> {
    if let Some(channel) = delta.iter().position(|&v| v <= S::ZERO) {
        return Err(fail(format!(
            "第 {channel} 个通道的步长必须为正，得到 {:?}",
            delta[channel]
        )));
    }
    Ok(())
}

/// 宽度不符一律 `Code::Invalid`：短了会静默读到别的通道，长了是多出来的数被丢掉。
fn check_len(what: &str, got: usize, want: usize) -> Result<()> {
    if got == want {
        return Ok(());
    }
    Err(fail(format!("{what}宽度必须是 {want}，得到 {got}")))
}

/// 统一的失败构造：出处固定、码位固定，报错文案各自不同。
fn fail(message: String) -> Error {
    Error::new(Code::Invalid, message).with(SITE)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: usize = 3;

    /// 一层：λ 实部 `−0.5`、`ω = 0.4`、直连 `1.0`，参数逐维都相同。
    fn layer(pairs: usize, rank: usize) -> Mamba<f64> {
        let n = WIDTH * pairs;
        Mamba::new(
            WIDTH,
            pairs,
            rank,
            vec![-0.5; n],
            vec![0.4; n],
            vec![1.0; WIDTH],
        )
        .expect("参数合法")
    }

    /// 一层只跑一个通道（宽度 1），用来做闭式对照。
    fn single(decay: f64, phase: f64, skip: f64) -> Mamba<f64> {
        Mamba::new(1, 1, 1, vec![decay], vec![phase], vec![skip]).expect("参数合法")
    }

    fn units(n: usize) -> Vec<f64> {
        vec![1.0; n]
    }

    #[test]
    fn new_rejects_bad_geometry() {
        let ok = (WIDTH, 2, 1, 6usize, 3usize);
        assert!(Mamba::new(0, 2, 1, vec![0.0; 2], vec![0.0; 2], vec![0.0; ok.0]).is_err());
        assert!(Mamba::new(WIDTH, 0, 1, vec![], vec![], vec![0.0; WIDTH]).is_err());
        assert!(
            Mamba::new(
                WIDTH,
                1,
                0,
                vec![0.0; WIDTH],
                vec![0.0; WIDTH],
                vec![0.0; WIDTH]
            )
            .is_err()
        );
        // 秩超过状态维数。
        assert!(
            Mamba::new(
                WIDTH,
                1,
                2,
                vec![0.0; WIDTH],
                vec![0.0; WIDTH],
                vec![0.0; WIDTH]
            )
            .is_err()
        );
        // 三组参数各自短一格。
        assert!(Mamba::new(WIDTH, 2, 1, vec![0.0; 5], vec![0.0; ok.3], vec![0.0; WIDTH]).is_err());
        assert!(Mamba::new(WIDTH, 2, 1, vec![0.0; ok.3], vec![0.0; 5], vec![0.0; WIDTH]).is_err());
        assert!(
            Mamba::new(
                WIDTH,
                2,
                1,
                vec![0.0; ok.3],
                vec![0.0; ok.3],
                vec![0.0; ok.4 - 1]
            )
            .is_err()
        );
    }

    #[test]
    fn new_rejects_unstable_decay_and_non_finite() {
        assert!(Mamba::new(1, 1, 1, vec![0.1], vec![0.0], vec![0.0]).is_err());
        assert!(Mamba::new(1, 1, 1, vec![f64::NAN], vec![0.0], vec![0.0]).is_err());
        assert!(Mamba::new(1, 1, 1, vec![0.0], vec![f64::INFINITY], vec![0.0]).is_err());
        assert!(Mamba::new(1, 1, 1, vec![0.0], vec![0.0], vec![f64::NAN]).is_err());
        // λ 实部为 0 合法：纯旋转，不放大也不衰减（旋转用例要用它）。
        assert!(Mamba::new(1, 1, 1, vec![0.0], vec![0.7], vec![0.0]).is_ok());
    }

    #[test]
    fn state_is_zero_and_its_size_ignores_the_sequence() {
        let m = layer(4, 2);
        let mut carry = m.state();
        assert_eq!(carry.planes().len(), WIDTH * 4 * 2);
        assert!(carry.planes().iter().all(|&v| v == 0.0));
        assert_eq!(carry.taps(), &[0.0; WIDTH]);

        let delta = units(WIDTH);
        let (b, c) = (units(2 * 4), units(2 * 4));
        for _ in 0..128 {
            m.step(&mut carry, &units(WIDTH), &delta, &b, &c)
                .expect("合法一步");
        }
        // 跑过 128 个 token，状态长度一格不变——这是「与序列长无关」的可测形态。
        assert_eq!(carry.planes().len(), WIDTH * 4 * 2);
        assert_eq!(carry.taps().len(), WIDTH);
    }

    #[test]
    fn step_rejects_bad_widths_and_step_sign() {
        let m = layer(2, 1);
        let mut carry = m.state();
        let delta = units(WIDTH);
        let (b, c) = (units(2), units(2));
        assert!(
            m.step(&mut carry, &units(WIDTH - 1), &delta, &b, &c)
                .is_err()
        );
        assert!(
            m.step(&mut carry, &units(WIDTH), &units(WIDTH + 1), &b, &c)
                .is_err()
        );
        assert!(
            m.step(&mut carry, &units(WIDTH), &delta, &units(1), &c)
                .is_err()
        );
        assert!(
            m.step(&mut carry, &units(WIDTH), &delta, &b, &units(3))
                .is_err()
        );
        assert!(
            m.step(&mut carry, &units(WIDTH), &[0.5, 0.0, 0.5], &b, &c)
                .is_err()
        );
        assert!(
            m.step(&mut carry, &units(WIDTH), &[0.5, -0.5, 0.5], &b, &c)
                .is_err()
        );
        assert!(
            m.step(&mut carry, &[1.0, f64::NAN, 1.0], &delta, &b, &c)
                .is_err()
        );
    }

    #[test]
    fn a_failed_step_leaves_the_state_untouched() {
        let m = layer(2, 1);
        let mut carry = m.state();
        let delta = units(WIDTH);
        let (b, c) = (units(2), units(2));
        m.step(&mut carry, &units(WIDTH), &delta, &b, &c)
            .expect("合法一步");
        let before = carry.clone();
        // 步长非正：在更新之前就被拒。
        assert!(
            m.step(&mut carry, &units(WIDTH), &[1.0, 0.0, 1.0], &b, &c)
                .is_err()
        );
        assert_eq!(carry, before);
        // 选择量长度不符同样在写之前。
        assert!(
            m.step(&mut carry, &units(WIDTH), &delta, &units(9), &c)
                .is_err()
        );
        assert_eq!(carry, before);
    }

    #[test]
    fn state_from_another_layer_is_rejected() {
        let narrow = layer(1, 1);
        let wide = layer(3, 1);
        let mut carry = narrow.state();
        let delta = units(WIDTH);
        assert!(
            wide.step(&mut carry, &units(WIDTH), &delta, &units(3), &units(3))
                .is_err()
        );
    }

    /// 读数是**更新之后**的状态：从零状态出发，一步的闭式结果是 `C·(dt/2·x·B) + D·x`。
    #[test]
    fn output_reads_the_updated_state() {
        let m = single(-0.5, 0.4, 1.0);
        let mut carry = m.state();
        let (x, dt) = (2.0f64, 0.25f64);
        let y = m
            .step(&mut carry, &[x], &[dt], &[0.7], &[0.3])
            .expect("合法一步");
        // 左端为 0，故 b̄ 的实部是 (dt/2)·x，虚部为 0；直连项 D·x 另加。
        let expected = 0.3 * (dt / 2.0 * x * 0.7) + x;
        assert!((y[0] - expected).abs() < 1e-12, "{} vs {expected}", y[0]);
    }

    /// 未来不影响过去：改掉第 5 个 token 不影响第 2 个 token 的输出。
    #[test]
    fn later_tokens_do_not_change_earlier_outputs() {
        let m = layer(2, 1);
        let delta = units(WIDTH);
        let (b, c) = (units(2), units(2));
        let inputs = [0.5, -1.5, 2.0, 0.25, 3.0];
        let run = |tokens: &[f64]| {
            let mut carry = m.state();
            tokens
                .iter()
                .map(|&x| {
                    m.step(&mut carry, &[x; WIDTH], &delta, &b, &c)
                        .expect("合法一步")
                })
                .collect::<Vec<_>>()
        };
        let first = run(&inputs);
        let mut changed = inputs;
        changed[4] = -7.0;
        let second = run(&changed);
        assert_eq!(first[..4], second[..4]);
        assert_ne!(first[4], second[4]);
    }

    /// 同一个输入写两次是**叠加**而不是覆盖：这与 `kda` 的 delta 规则正好相反。
    #[test]
    fn two_writes_accumulate() {
        let m = single(-0.5, 0.4, 0.0);
        let mut carry = m.state();
        m.step(&mut carry, &[1.0], &[0.25], &[1.0], &[0.0])
            .expect("合法一步");
        let once = carry.planes()[0];
        m.step(&mut carry, &[1.0], &[0.25], &[1.0], &[0.0])
            .expect("合法一步");
        let twice = carry.planes()[0];
        assert!(twice > once, "{twice} 应大于 {once}");
    }

    /// 旋转只改相位：`λ_re = 0` 时第二次（选择量为零的）一步保持模长，`ω = 0` 时状态原样不动。
    #[test]
    fn rotation_keeps_the_state_magnitude() {
        for (phase, moves) in [(0.4f64, true), (0.0, false)] {
            let m = single(0.0, phase, 0.0);
            let mut carry = m.state();
            m.step(&mut carry, &[1.0], &[0.25], &[1.0], &[0.0])
                .expect("合法一步");
            let before = [carry.planes()[0], carry.planes()[1]];
            // 选择量为零：状态只吃 `ā h`，这正是「纯旋转」的那一步。
            m.step(&mut carry, &[1.0], &[0.25], &[0.0], &[0.0])
                .expect("合法一步");
            let after = [carry.planes()[0], carry.planes()[1]];
            let norm = |s: [f64; 2]| s[0].hypot(s[1]);
            assert!((norm(after) - norm(before)).abs() < 1e-12, "ω={phase}");
            if moves {
                assert!(after != before);
            } else {
                assert_eq!(after, before);
            }
        }
    }

    /// 虚部不是装饰：同一组参数，`ω = 0` 与 `ω ≠ 0` 读出的结果不同。
    ///
    /// 零状态上 `b̄` 是实数（左端为 0），旋转只在**已有状态**上才显出来，因此先写一步、
    /// 再以零选择量读那一步——读到的正是 `Re(ā h)`，相位不同就不同。
    #[test]
    fn the_imaginary_part_changes_the_output() {
        let mut ys = Vec::new();
        for phase in [0.0f64, 0.9] {
            let m = single(-0.2, phase, 0.0);
            let mut carry = m.state();
            m.step(&mut carry, &[1.0], &[0.5], &[1.0], &[0.0])
                .expect("合法一步");
            let y = m
                .step(&mut carry, &[0.0], &[0.5], &[0.0], &[1.0])
                .expect("合法一步");
            ys.push(y[0]);
        }
        assert!(ys[0] != ys[1], "{ys:?}");
    }

    /// 低秩 MIMO：第二组选择量为零时，与单秩层逐步给出相同的读数与相同的状态。
    #[test]
    fn a_silent_rank_does_not_change_anything() {
        let shape = (vec![-0.5; 2], vec![0.4; 2], vec![0.5]);
        let one = Mamba::new(1, 2, 1, shape.0.clone(), shape.1.clone(), shape.2.clone())
            .expect("参数合法");
        let two = Mamba::new(1, 2, 2, shape.0, shape.1, shape.2).expect("参数合法");
        let (mut a, mut b) = (one.state(), two.state());
        let (b1, c1) = ([0.7f64, 0.5], [0.3f64, -0.2]);
        // 第二组选择量整行为零：秩的**行**多一组，但这一组什么都没写、也没读到。
        let (b2, c2) = ([0.7f64, 0.5, 0.0, 0.0], [0.3f64, -0.2, 0.0, 0.0]);
        for x in [0.5f64, -1.0, 2.0] {
            let ya = one.step(&mut a, &[x], &[0.25], &b1, &c1).expect("合法一步");
            let yb = two.step(&mut b, &[x], &[0.25], &b2, &c2).expect("合法一步");
            assert_eq!(ya, yb);
        }
        assert_eq!(a.planes(), b.planes());
    }

    /// `f32` 上同一套算术跑得动：泛型不是只为 `f64` 测试写的。
    #[test]
    fn f32_runs_the_same_loop() {
        let m =
            Mamba::new(2, 2, 2, vec![-1.0f32; 4], vec![0.3; 4], vec![0.5; 2]).expect("参数合法");
        let mut carry = m.state();
        let mut last = Vec::new();
        for _ in 0..8 {
            last = m
                .step(&mut carry, &[1.0, -1.0], &[0.5, 0.5], &[1.0; 4], &[1.0; 4])
                .expect("合法一步");
        }
        assert!(last.iter().all(|y| y.is_finite()));
    }
}

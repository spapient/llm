//! 稀疏专家路由（MoE，§5.8.2，第 10 款标杆的专家轴）。

use std::cmp::Ordering;

use crate::invalid;
use diagnostic::Result;
use op::scan::Scalar;

use crate::csa::{accumulate, dot, finite};
use crate::policy::Calib;

/// 报错出处：路由的几何与取值都在入口失败，不留到矩阵乘的中途。
const SITE: &str = "algorithm::moe";

/// 第 10 款标杆（Kimi K3）的路由专家总数。
pub const K3_EXPERTS: usize = 896;

/// 同一次前向激活的专家数（K3：896 取 16）。
pub const K3_ACTIVE: usize = 16;

/// 稀疏专家层：路由矩阵 + 一组方阵专家 + 可选共享专家。
#[derive(Debug, Clone, PartialEq)]
pub struct Moe<S: Scalar = f32> {
    width: usize,
    active: usize,
    calib: Calib,
    router: Vec<S>,
    experts: Vec<Vec<S>>,
    shared: Option<Vec<S>>,
}

impl<S: Scalar> Moe<S> {
    /// 由宽度、激活数、路由矩阵 `[专家数, width]`、专家方阵与共享专家构造。
    ///
    /// 校准名默认取 [`Calib::Spherical`]（§1.11 的默认），要换走 [`Moe::with_calib`]。
    ///
    /// # Errors
    ///
    /// 宽度、专家数或激活数为零，激活数超过专家数，路由矩阵不是 `[专家数, width]`，任一专家或
    /// 共享专家不是 `[width, width]`，或这些乘积溢出——都返回 `Code::Invalid`。行数错了的话
    /// 每个专家的输出会整体错位，却不会有任何一处报错。
    pub fn new(
        width: usize,
        active: usize,
        router: Vec<S>,
        experts: Vec<Vec<S>>,
        shared: Option<Vec<S>>,
    ) -> Result<Self> {
        check_shape(width, active, &router, &experts)?;
        if let Some(matrix) = &shared {
            check_matrix("共享专家", width, matrix)?;
        }
        Ok(Self {
            width,
            active,
            calib: Calib::Spherical,
            router,
            experts,
            shared,
        })
    }

    /// 换校准名（分数 → 分布的同一端口，§1.11）。
    #[must_use]
    pub fn with_calib(mut self, calib: Calib) -> Self {
        self.calib = calib;
        self
    }

    accessors! {
        /// 宽度（每个 token 的隐藏维）。
        width -> usize = width,
    }

    /// 路由专家数（候选集合的大小）。
    pub fn experts(&self) -> usize {
        self.experts.len()
    }

    accessors! {
        /// 每次前向激活的专家数 `k`。
        active -> usize = active,
        /// 校准名。
        calib -> Calib = calib,
    }

    /// 路由矩阵，行主序 `[专家数, width]`。
    pub fn router(&self) -> &[S] {
        &self.router
    }

    /// 共享专家（恒参与、不占 `k`、不被路由权重缩放）。
    pub fn shared(&self) -> Option<&[S]> {
        self.shared.as_deref()
    }

    /// 一次路由：`(专家下标, 权重)`，权重降序、下标互不相同、和为 1。
    ///
    /// # Errors
    ///
    /// `x` 的宽度不是本层的宽度、输入含非有限值、或路由分布出现非有限值（分数大到内积溢出）
    /// 时返回 `Code::Invalid`——放行会把 `NaN` 一路带进前向，而报错点离原因很远。
    pub fn route(&self, x: &[S]) -> Result<Vec<(usize, S)>> {
        self.check_input(x)?;
        let scores: Vec<S> = self
            .router
            .chunks_exact(self.width)
            .map(|row| dot(row, x))
            .collect();
        let dist = self.calib.to_dist(&scores);
        if let Some(bad) = dist.iter().position(|&d| !finite(d)) {
            return err!(
                "路由分布第 {bad} 项非有限（分数 {:?}）：内积已溢出",
                scores[bad]
            );
        }
        Ok(renormalized(ranked(dist, self.active)))
    }

    /// 一次前向：选中专家的加权和，再加共享专家。
    ///
    /// # Errors
    ///
    /// 同 [`Moe::route`]。
    pub fn forward(&self, x: &[S]) -> Result<Vec<S>> {
        let mut out = vec![S::ZERO; self.width];
        for (expert, weight) in self.route(x)? {
            let y = apply(&self.experts[expert], self.width, x);
            accumulate(&mut out, &y, weight);
        }
        if let Some(matrix) = &self.shared {
            let y = apply(matrix, self.width, x);
            accumulate(&mut out, &y, S::ONE);
        }
        Ok(out)
    }

    /// 入口校验：宽度与非有限值。
    fn check_input(&self, x: &[S]) -> Result<()> {
        if x.len() != self.width {
            return err!("输入宽度必须是 {}，得到 {}", self.width, x.len());
        }
        if let Some(bad) = x.iter().position(|&v| !finite(v)) {
            return err!("输入第 {bad} 项非有限：路由分数会把这些值带进分布");
        }
        Ok(())
    }
}

/// 按 `(权重降序, 下标升序)` 排序后只留前 `k` 个。
fn ranked<S: Scalar>(dist: Vec<S>, k: usize) -> Vec<(usize, S)> {
    let mut ranked: Vec<(usize, S)> = dist.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    ranked.truncate(k);
    ranked
}

/// 在选中集合上重新归一：输出整体不再随 `k` 缩放。
fn renormalized<S: Scalar>(picks: Vec<(usize, S)>) -> Vec<(usize, S)> {
    let total = picks.iter().fold(S::ZERO, |acc, &(_, w)| acc.add(w));
    debug_assert!(total > S::ZERO, "分布已校验为有限，选中集的和必为正");
    picks
        .into_iter()
        .map(|(expert, weight)| (expert, weight.div(total)))
        .collect()
}

/// 一次矩阵乘：`matrix` 是行主序 `[rows, width]`，返回每行与 `x` 的内积。
fn apply<S: Scalar>(matrix: &[S], width: usize, x: &[S]) -> Vec<S> {
    matrix.chunks_exact(width).map(|row| dot(row, x)).collect()
}

/// 几何校验：宽度/专家数/激活数非零，路由矩阵与专家方阵尺寸正确。
fn check_shape<S: Scalar>(
    width: usize,
    active: usize,
    router: &[S],
    experts: &[Vec<S>],
) -> Result<()> {
    let count = experts.len();
    nonzero(width, count)?;
    in_range(active, count)?;
    let expected = product(count, width)?;
    if router.len() != expected {
        return err!(
            "路由矩阵必须是 [专家数 {count}, width {width}] = {expected} 个元素，得到 {}",
            router.len()
        );
    }
    check_experts(width, experts)
}

/// 宽度与专家数都非零。
fn nonzero(width: usize, count: usize) -> Result<()> {
    if width > 0 && count > 0 {
        return Ok(());
    }
    err!("宽度与专家数都必须非零，得到 width = {width}、专家 {count} 个")
}

/// 激活数在 `1..=count` 之间。
fn in_range(active: usize, count: usize) -> Result<()> {
    if (1..=count).contains(&active) {
        return Ok(());
    }
    err!("激活数必须在 1..={count} 之间，得到 {active}")
}

/// 每个专家都是一张 `[width, width]` 方阵。
fn check_experts<S: Scalar>(width: usize, experts: &[Vec<S>]) -> Result<()> {
    for (expert, matrix) in experts.iter().enumerate() {
        check_matrix(&format!("专家 {expert}"), width, matrix)?;
    }
    Ok(())
}

/// 方阵元素数 `width × width`，溢出即拒绝。
fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid!("方阵尺寸溢出（{a} × {b}）"))
}

/// 一张方阵的尺寸校验。
fn check_matrix<S: Scalar>(what: &str, width: usize, matrix: &[S]) -> Result<()> {
    let expected = product(width, width)?;
    if matrix.len() == expected {
        return Ok(());
    }
    err!(
        "{what} 必须是 [width, width] = {expected} 个元素，得到 {}（width = {width}）",
        matrix.len()
    )
}

#[cfg(test)]
#[path = "../tests/moe_tests.rs"]
mod moe_tests;

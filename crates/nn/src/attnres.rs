//! 注意力相对残差（AttnRes）：第 10 款标杆（Kimi K3）的深度轴。
//!
//! 深层的残差是**累加**：第 `l` 层的输入是前面所有层输出的和，于是越深，新增的那一层对总和的相对
//! 贡献越小——梯度与信号都被早期层淹没。AttnRes 把「往里加」换成「按注意力取」：深层不再累加，
//! 而是用一次注意力在**前几层的输出**上取一个凸组合，再并回主路。
//!
//! 三件套与它们的边界：
//!
//! - [`AttnRes`] 只做 `Σ w_j · V_j` 的凸组合，权重由 [`crate::csa::softmax_row`] 给出（与块表注意力、
//!   门控同一份 softmax，不因模块边界重写）。它不持参数、不写 KV：`V_j` 是调用方给的前几层输出。
//! - [`Sources`] 是这一层的**来源簿记**：哪些层号喂给本层、主路从哪一层并入。深度改配方时，读的是簿记
//!   而不是重算一遍下标。
//! - [`splice`] 是并回主路的那一步：`main + α·(mixed − main)`。`α = 0` 原样返回主路、`α = 1` 直接替换，
//!   `α ∈ (0, 1)` 是两者之间——`α ≤ 1` 因此保证这次并回**非扩张**。
//!
//! # 为什么来源按层号而不是按位置
//!
//! 位置（`values` 里的第几个）是这一层的私有视角，层号是模型级的名字。用层号记来源，才能让"第 40 层
//! 取第 1/2 层"这件事在装配侧可读、可测、可换配方；换成位置的话，任何一次重排都会悄无声息地改语义。
//! 层号必须严格递增且不与主路重复——重复的层会在一份自注意力里被算两次，等于偷偷改了权重。
//!
//! 空来源被拒：那等于「按注意力取空集」，softmax 无定义，残差也没有东西可并。

use diagnostic::{Code, Error, Result};
use op::scan::Scalar;

use crate::csa::{accumulate, dot, finite, softmax_row};

/// 报错出处：来源簿记、几何与取值都在入口失败，不留到混合中途。
const SITE: &str = "nn::attnres";

/// 注意力相对残差：在前几层输出上按一次注意力取凸组合。
///
/// 不持参数：查询、键、值都由调用方给（真实实现里各是一次投影）。方法对元素类型泛型，同一份几何
/// 既跑 `f32` 也跑 `f64`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnRes {
    width: usize,
}

impl AttnRes {
    /// 由隐藏维 `d` 构造。
    ///
    /// # Errors
    ///
    /// `width` 为零——返回 `Code::Invalid`。
    pub fn new(width: usize) -> Result<Self> {
        if width == 0 {
            return Err(Error::new(Code::Invalid, "宽度必须非零").with(SITE));
        }
        Ok(Self { width })
    }

    /// 隐藏维 `d`。
    pub fn width(&self) -> usize {
        self.width
    }

    /// 一次混合：`Σ w_j · values[j]`，`w = softmax(q·valuesᵀ / √d)`。
    ///
    /// # Errors
    ///
    /// 查询宽度不是 `d`、坐标集为空、坐标集里任一宽度不是 `d`、或任一输入非有限——返回
    /// `Code::Invalid`（放行会让 `NaN` 分数经 softmax 铺满整组权重）。
    pub fn mix<S: Scalar>(&self, query: &[S], values: &[&[S]]) -> Result<Vec<S>> {
        self.check(query, values, "坐标集")?;
        let scores: Vec<S> = values.iter().map(|value| dot(query, value)).collect();
        self.combine(&scores, values)
    }

    /// 门控变体：分数由**注意力日志**给出，而不是查询的内积。
    ///
    /// 这是给装配侧的直通口：真实 K3 层的分数来自注意力日志（`Q·K/√d` 的总和），本模块不该凭空
    /// 重算一遍。`logs` 与 `values` 一一对应，几何与有限性仍在入口判。
    ///
    /// # Errors
    ///
    /// 日志条数与坐标集不等、坐标集为空、任一宽度不是 `d`、任一输入非有限——返回 `Code::Invalid`。
    pub fn gated<S: Scalar>(&self, logs: &[S], values: &[&[S]]) -> Result<Vec<S>> {
        self.check_rows(values, "坐标集")?;
        if logs.len() != values.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "注意力日志必须与坐标集一一对应，得到 {} 条日志对 {} 个坐标",
                    logs.len(),
                    values.len()
                ),
            )
            .with(SITE));
        }
        if let Some(bad) = logs.iter().position(|&log| !finite(log)) {
            return Err(Error::new(
                Code::Invalid,
                format!("注意力日志第 {bad} 条非有限：softmax 会把它摊到整组权重上"),
            )
            .with(SITE));
        }
        self.combine(logs, values)
    }

    /// 分数 → 权重 → 凸组合。两个入口共用这一份，权重只有一处定义。
    fn combine<S: Scalar>(&self, scores: &[S], values: &[&[S]]) -> Result<Vec<S>> {
        // 温度 `√d` 与注意力同一约定：分数随宽度增长，不除的话 `d = 4096` 时 softmax 早被压成 one-hot。
        let temperature = S::from_f64(self.width as f64).sqrt();
        let mut weights: Vec<S> = scores.iter().map(|&score| score.div(temperature)).collect();
        softmax_row(&mut weights);
        let mut out = vec![S::ZERO; self.width];
        for (weight, value) in weights.iter().zip(values) {
            accumulate(&mut out, value, *weight);
        }
        Ok(out)
    }

    /// 入口校验：查询宽度与坐标集。
    fn check<S: Scalar>(&self, query: &[S], values: &[&[S]], what: &str) -> Result<()> {
        self.check_row(query, "查询")?;
        self.check_rows(values, what)
    }

    /// 坐标集非空、每个坐标的宽度正确、取值有限。
    fn check_rows<S: Scalar>(&self, values: &[&[S]], what: &str) -> Result<()> {
        if values.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                format!("{what}为空：softmax 无定义，残差也没有东西可并"),
            )
            .with(SITE));
        }
        for (index, value) in values.iter().enumerate() {
            self.check_row(value, &format!("{what}第 {index} 项"))?;
        }
        Ok(())
    }

    /// 单行几何 + 有限性。
    fn check_row<S: Scalar>(&self, row: &[S], what: &str) -> Result<()> {
        if row.len() != self.width {
            return Err(Error::new(
                Code::Invalid,
                format!("{what}宽度必须是 {}，得到 {}", self.width, row.len()),
            )
            .with(SITE));
        }
        match row.iter().position(|&value| !finite(value)) {
            Some(index) => Err(Error::new(
                Code::Invalid,
                format!("{what}第 {index} 项非有限：softmax 会把 NaN 摊到整组权重上"),
            )
            .with(SITE)),
            None => Ok(()),
        }
    }
}

/// 并回主路：`main + α·(mixed − main)`，`α ∈ [0, 1]`。
///
/// `α = 0` 原样返回主路、`α = 1` 直接替换。系数在 `[0, 1]` 内时并回是**凸组合**，因此非扩张：
/// 混合值有多大，主路就最多走那么远，不会把激活放大。
///
/// # Errors
///
/// 主路与混合值宽度不等、任一含非有限值、或系数不在 `[0, 1]`——返回 `Code::Invalid`。
pub fn splice<S: Scalar>(main: &[S], mixed: &[S], alpha: S) -> Result<Vec<S>> {
    check_splice(main, mixed, alpha)?;
    Ok(main
        .iter()
        .zip(mixed)
        .map(|(&keep, &mixed)| keep.add(alpha.mul(mixed.sub(keep))))
        .collect())
}

/// 并回的三处入口校验，按从便宜到贵的次序：宽度、系数、逐项有限性。
fn check_splice<S: Scalar>(main: &[S], mixed: &[S], alpha: S) -> Result<()> {
    check_width(main, mixed)?;
    check_alpha(alpha)?;
    check_pairs(main, mixed)
}

/// 主路与混合值等宽：并回是把同一维度的两段合起来。
fn check_width<S: Scalar>(main: &[S], mixed: &[S]) -> Result<()> {
    if main.len() == mixed.len() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "主路宽 {} 与混合值宽 {} 不等：并回是把同一维度的两段合起来",
            main.len(),
            mixed.len()
        ),
    )
    .with(SITE))
}

/// 系数 `α ∈ [0, 1]`：出界就不再是凸组合，主路会被拉长而不是内插。
fn check_alpha<S: Scalar>(alpha: S) -> Result<()> {
    if finite(alpha) && alpha >= S::ZERO && alpha <= S::ONE {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("系数是 {alpha:?}：必须在 [0, 1] 内（0 保留主路，1 换成混合值）"),
    )
    .with(SITE))
}

/// 两段逐项有限：任一项是 `NaN`，并回都会把它带进主路。
fn check_pairs<S: Scalar>(main: &[S], mixed: &[S]) -> Result<()> {
    match main
        .iter()
        .zip(mixed)
        .position(|(&a, &b)| !finite(a) || !finite(b))
    {
        Some(index) => Err(Error::new(
            Code::Invalid,
            format!("第 {index} 项非有限：并回会把 NaN 带进主路"),
        )
        .with(SITE)),
        None => Ok(()),
    }
}

/// 一层的来源簿记：喂本层的前几层层号 + 主路取自哪一层。
///
/// 层号是模型级的名字（位置是这一层的私有视角）：装配侧读得到自己能解释的东西，重排 `values` 也不会
/// 改语义。构造期就判——来源必须**严格递增**且**都不等于主路**；重复的层会在一份自注意力里被算两次，
/// 等于偷偷改了权重。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sources {
    layers: Vec<usize>,
    main: usize,
}

impl Sources {
    /// 由来源层号与主路层号构造。
    ///
    /// # Errors
    ///
    /// 来源为空、来源不是严格递增、或任一来源等于主路——返回 `Code::Invalid`。
    pub fn new(layers: Vec<usize>, main: usize) -> Result<Self> {
        if layers.is_empty() {
            return Err(Error::new(Code::Invalid, "来源为空：这一层没有可取的前层输出").with(SITE));
        }
        if let Some(gap) = layers.windows(2).position(|pair| pair[0] >= pair[1]) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "来源第 {} 与第 {} 项不是严格递增（{} ≥ {}）：重复的层会被算两次",
                    gap,
                    gap + 1,
                    layers[gap],
                    layers[gap + 1]
                ),
            )
            .with(SITE));
        }
        if layers.contains(&main) {
            return Err(Error::new(
                Code::Invalid,
                format!("主路第 {main} 层同时出现在来源里：同一条输出不能取两次"),
            )
            .with(SITE));
        }
        Ok(Self { layers, main })
    }

    /// 来源层号（严格递增）。
    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// 主路取自第几层。
    pub fn main(&self) -> usize {
        self.main
    }

    /// 来源层数：凸组合里的项数。
    pub fn depth(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
#[path = "../tests/attnres_tests.rs"]
mod attnres_tests;

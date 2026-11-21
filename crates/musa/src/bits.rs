//! 对拍：程序按位打印的输出 vs 标量参考，逐元素相对误差。
//!
//! 判据只有一条：**每个元素都要过阈值**，不是"平均误差够小"——平均会把一个错到 `1e9` 的
//! 元素平摊成"还行"。`NaN` 只有两边同时是 `NaN` 时才算相等（`NaN != NaN` 是 IEEE 的语义，
//! 但在这里"参考值是 NaN、设备也是 NaN"是同一条结果）。

use diagnostic::{Code, Error, Result};

/// 相对误差的分母下限：参考值接近 0 时避免把"绝对误差极小"放大成无穷大。
const FLOOR: f64 = 1e-6;

/// 对拍结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Outcome {
    /// 参与比较的元素个数。
    pub compared: usize,
    /// 超过阈值的元素个数。
    pub over: usize,
    /// 最大相对误差。
    pub worst: f32,
    /// 最大相对误差出现在第几个元素（0 基）。
    pub at: usize,
}

impl Outcome {
    /// 是否全部在阈值内。
    pub fn pass(&self) -> bool {
        self.over == 0
    }
}

/// 比较两份文本：每行一个十进制数，`printed` 是程序输出，`reference` 是标量参考。
///
/// 长度不等、行不是数、空输入都算失败：这三种情况下"全部通过"只会是假绿。
pub fn compare(printed: &str, reference: &str, tol: f64) -> Result<Outcome> {
    let got = parse(printed, "程序输出")?;
    let want = parse(reference, "标量参考")?;
    aligned(&got, &want)?;
    Ok(summarize(&want, &got, tol))
}

/// 长度不等与空输入两条硬门：任一不成立，“全部通过”只会是假绿。
fn aligned(got: &[f64], want: &[f64]) -> Result<()> {
    if got.len() != want.len() {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "元素数不一致：程序打印 {} 个，参考 {} 个（形状或末块守卫不同）",
                got.len(),
                want.len()
            ),
        ));
    }
    if want.is_empty() {
        return Err(Error::new(Code::Invalid, "没有可比对的元素"));
    }
    Ok(())
}

/// 逐元素误差的摘要：最大误差与超阈个数。
fn summarize(want: &[f64], got: &[f64], tol: f64) -> Outcome {
    let (at, worst) = largest(want, got);
    let over = want
        .iter()
        .zip(got)
        .filter(|(want, got)| deviation(**got, **want) > tol)
        .count();
    Outcome {
        compared: want.len(),
        over,
        worst: worst as f32,
        at,
    }
}

/// 最大相对误差及其下标。
fn largest(want: &[f64], got: &[f64]) -> (usize, f64) {
    want.iter()
        .zip(got)
        .map(|(want, got)| deviation(*got, *want))
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(at, worst), (index, value)| {
            if value > worst {
                (index, value)
            } else {
                (at, worst)
            }
        })
}

/// 一个元素的相对误差；位相同时为 0（覆盖 ±0 与同 `NaN`），任一侧是 `NaN` 时为无穷大。
fn deviation(got: f64, want: f64) -> f64 {
    if got.to_bits() == want.to_bits() {
        return 0.0;
    }
    if got.is_nan() || want.is_nan() {
        return f64::INFINITY;
    }
    (got - want).abs() / want.abs().max(FLOOR)
}

/// 逐行解析；空行跳过（`fprintf` 的换行与末尾换行都会产生一行空）。
fn parse(text: &str, what: &str) -> Result<Vec<f64>> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.trim().parse::<f64>().map_err(|_| {
                Error::new(
                    Code::Invalid,
                    format!("{what}里有一行不是十进制数：`{}`", line.trim()),
                )
            })
        })
        .collect()
}

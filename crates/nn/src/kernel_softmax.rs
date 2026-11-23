//! Softmax：`y[n] = exp(x[n] - max(x)) / sum_k exp(x[k] - max(x))`。
use crate::model::{Cost, Sample, ShapeCtx, cost_n};
use diagnostic::{Code, Error};
use tensor::View;

// 单轴定义与形状投影复用 unary 骨架：本文件只声明 softmax 特有的代价、样本与遍历。
pub use crate::unary::{AXES, project_shape};

/// 标量代价：三趟读 `x`、一趟写 `y`，每元素约 3 次运算（max、exp、除法）。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
    cost_n(shape, "softmax", 3.0, 3)
}

/// 确定性样本：`x`、`y` 两个等长缓冲。
pub fn sample(len: usize, _: usize) -> Sample {
    Sample::f32_io(len, len, len, len, vec![("N", len as i64)])
}

/// 标量参考：三趟，与内核同一公式。
///
/// # Errors
///
/// 输入为空时返回说明：空行没有可归一化的分母。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    if x.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            "softmax 输入不能为空：归一化分母无定义",
        ));
    }
    let max = x.iter().copied().fold(f32::MIN, f32::max);
    let mut sum = 0.0;
    let exps: Vec<f32> = x
        .iter()
        .map(|value| {
            let e = (value - max).exp();
            sum += e;
            e
        })
        .collect();
    Ok(exps.iter().map(|e| e / sum).collect())
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输入与输出等长，或输入为空时返回说明：空行没有可归一化的分母。
pub fn forward(x: View<'_, f32>, out: &mut [f32]) -> diagnostic::Result<()> {
    let x = x.as_slice();
    if x.len() != out.len() {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
        ));
    }
    if x.is_empty() {
        return Err(Error::new(
            Code::Invalid,
            "softmax 输入不能为空：归一化分母无定义",
        ));
    }
    let max = x.iter().copied().fold(f32::MIN, f32::max);
    // 一趟 `exp`：先用写入的指数值累加分母，再原地除掉。旧实现算了两遍 `exp`
    // （一遍求 `sum`、一遍求结果），而 `exp` 正是这里最贵的指令；`(x-max).exp()`
    // 的取值与从左到右的累加顺序都和旧实现一致，所以输出逐位不变。
    let mut sum = 0.0;
    for (out, x) in out.iter_mut().zip(x) {
        let e = (x - max).exp();
        sum += e;
        *out = e;
    }
    for out in out.iter_mut() {
        *out /= sum;
    }
    Ok(())
}

/// 标记类型：承载 `#[model]` 语义。
pub struct Softmax;

/// Softmax 的具体语义别名。
#[annotation::model(name = "softmax_f32", shape = [4096, 4096], dtype = "f32", skeleton = "softmax")]
pub type SoftmaxF32 = Softmax;

crate::operator_here!(Softmax, "softmax_f32", 1, (4096, 4096), true, reference);

#[cfg(test)]
mod tests {
    include!("../tests/kernel_softmax.rs");
}

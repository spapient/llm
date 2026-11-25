//! 投影格式策略：共享 [`WeightFormat`] 来自 `dtype`，这里只处理 `gemv` 选择与诊断。

use dtype::DType;
pub use dtype::WeightFormat;

use crate::Linear;

impl Linear {
    /// 本格式对应的已声明投影算子；`None` = 没有匹配格式与内核分组的算子。
    #[must_use]
    pub fn operator_for(format: WeightFormat) -> Option<&'static str> {
        match format.dtype() {
            DType::F32 => Some(Self::DENSE),
            _ => quantized(format)
                .filter(|quant| Some(quant.group) == quant.declared)
                .map(|quant| quant.op),
        }
    }

    /// 为什么这种编码取不到投影算子：说清缺算子、标度还是匹配的分组。
    #[must_use]
    pub fn unpriced_reason_for(format: WeightFormat) -> String {
        if let Some(quant) = quantized(format) {
            return quantized_reason(quant);
        }
        match format.dtype() {
            DType::U2 | DType::U4 => format!(
                "是 {} 位打包，但找不到它的标度条目，分组不明",
                format.dtype().bits()
            ),
            DType::F16 | DType::BF16 => {
                format!("编码是 {format}，还没有 16 位元素的投影算子")
            }
            _ => format!("编码是 {format}，还没有这个格式的投影算子"),
        }
    }
}

/// 带分组的量化格式：存储类型、文件分组、对应算子与内核声明分组。
struct Quant {
    dtype: DType,
    group: usize,
    op: &'static str,
    declared: Option<usize>,
}

/// 从共享格式提取量化事实，并用 `nn` 注册表补充分组兼容性。
fn quantized(format: WeightFormat) -> Option<Quant> {
    let group = format.group()?;
    let op = match format.dtype() {
        DType::U2 => crate::Linear::I2,
        DType::U4 => crate::Linear::OP,
        DType::I8 => crate::Linear::I8,
        _ => return None,
    };
    Some(Quant {
        dtype: format.dtype(),
        group,
        op,
        declared: kernel_group(op),
    })
}

/// 量化分组对不上内核声明档时的诊断文案。
fn quantized_reason(quant: Quant) -> String {
    match quant.declared {
        Some(declared) => format!(
            "按 {} 个元素一组量化（int{}），而内核声明的是 {declared} 个元素一组：分组不同不能共用同一条内核",
            quant.group,
            quant.dtype.bits()
        ),
        None => format!(
            "按 {} 个元素一组量化，但没有任何算子声明过 int{} 的分组",
            quant.group,
            quant.dtype.bits()
        ),
    }
}

/// 算子在语义表里声明的分组；`None` = 算子未注册或未声明分组。
fn kernel_group(op: &str) -> Option<usize> {
    crate::registry::model(op).map(|model| model.group as usize)
}

#[cfg(test)]
#[path = "../tests/format_tests.rs"]
mod format_tests;

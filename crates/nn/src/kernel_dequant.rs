//! 打包反量化骨架：每字节 `CPB` 个码，`y = scale · (code − zero)`。

use crate::model::{Buf, Cost, Sample, ShapeCtx, Values};
use diagnostic::{Code, Error};

// 单轴定义与形状投影复用 unary 骨架。
pub use crate::unary::{AXES, project_shape};

/// 反量化标度（2 的负幂网格，f32 精确）。
pub const SCALE: f32 = prof::epilogue::QUANT_SCALE;

/// 打包格式的前置约束：输出长度必须是输入字节数的 `cpb` 倍。
fn check_len(q: &[u8], out: &[f32], cpb: usize) -> Result<(), Error> {
    if out.len() != q.len() * cpb {
        return Err(Error::new(
            Code::Invalid,
            format!("缓冲长度不符：q {} 字节、out {} 元素", q.len(), out.len()),
        ));
    }
    Ok(())
}

/// 仿射参数校验：标度不能为零（否则整组塌成常数、失去量化口径），零点必须落在码域内。
///
/// 码域是 `0..2^width`：零点是码点到有符号值的平移量，越出码域意味着存在无法表示的码点，
/// 属于配置错误而不是数值极端。打包格式传自身位宽（i2=2、i4=4），逐元素格式传 16 档。
///
/// # Errors
///
/// 标度为零、零点不是有限值，或零点不在 `0..2^width` 内时返回 [`Code::Invalid`]。
pub fn check_affine(scale: f32, zero: f32, width: usize) -> Result<(), Error> {
    if scale == 0.0 || !scale.is_finite() {
        return Err(Error::new(Code::Invalid, "反量化标度必须是非零有限值"));
    }
    if !zero.is_finite() {
        return Err(Error::new(Code::Invalid, "反量化零点必须是有限值"));
    }
    let domain = 1u32 << width;
    if !(0.0..domain as f32).contains(&zero) {
        return Err(Error::new(
            Code::Invalid,
            format!("零点 {zero} 越出码域 0..{domain}（{width} 位码）"),
        ));
    }
    Ok(())
}

/// 标量代价：每元素一次减法一次乘法；读打包字节（`n / cpb` 个），写 `n` 个 f32。
///
/// # Errors
///
/// 形状缺轴时返回说明。
pub fn cost_packed(shape: &ShapeCtx, cpb: usize) -> Result<Cost, Error> {
    let n = shape
        .get('n')
        .ok_or_else(|| Error::new(Code::Invalid, "dequant 形状缺轴 `n`"))?;
    Ok(Cost {
        flops: (n as f64) * 2.0,
        read: (n / cpb) as u64,
        write: (n as u64) * 4,
    })
}

/// 确定性样本：`q`（打包字节，`len / cpb` 个）、`y`（`len` 个 f32）。
pub fn sample_packed(len: usize, cpb: usize) -> Sample {
    let bytes = len / cpb;
    Sample {
        m: len,
        n: len,
        scalars: vec![("N", len as i64)],
        bufs: vec![
            Buf {
                name: "q",
                values: Values::U8((0..bytes).map(|i| (i * 37 % 251) as u8).collect()),
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; len]),
            },
        ],
    }
}

/// 解包字节内第 `slot` 个码，按对称零点回退成有符号值。
pub fn unpack(byte: u8, slot: usize, cpb: usize, zero: f32) -> f32 {
    let width = crate::quant::width_cpb(cpb);
    let mask = (1u8 << width) - 1;
    ((byte >> (slot * width)) & mask) as f32 - zero
}

/// 标量参考：逐字节解 `cpb` 个码，与内核同一公式。
pub fn reference_packed(q: &[u8], cpb: usize, zero: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.len() * cpb);
    for byte in q {
        for slot in 0..cpb {
            out.push(SCALE * unpack(*byte, slot, cpb, zero));
        }
    }
    out
}

/// 内核本体：结果写进调用方给的缓冲。
///
/// # Errors
///
/// 输出长度不是输入字节数的 `cpb` 倍时返回说明。
pub fn forward_packed(
    scale: f32,
    zero: f32,
    q: &[u8],
    out: &mut [f32],
    cpb: usize,
) -> diagnostic::Result<()> {
    check_len(q, out, cpb)?;
    check_affine(scale, zero, crate::quant::width_cpb(cpb))?;
    for (byte, chunk) in q.iter().zip(out.chunks_mut(cpb)) {
        for (slot, place) in chunk.iter_mut().enumerate() {
            *place = scale * unpack(*byte, slot, cpb, zero);
        }
    }
    Ok(())
}

/// 一档打包反量化的配置：每字节码数与对称零点，唯一事实源是 [`crate::quant::Row`]。
///
/// 代价、样本、参考与内核都由这一行配置派生，因此新增一档打包精度只需实现
/// [`ROW`](Packed::ROW)；码域、每字节码数与零点都不必手写第二遍。
pub trait Packed {
    /// 配置表的一行：位宽、每字节码数与对称零点。
    const ROW: crate::quant::Row;

    /// 每字节码数（`8 / 位宽`）。
    fn cpb() -> usize {
        Self::ROW.cpb
    }

    /// 对称零点：无符号码点减去它得到有符号值。
    fn zero() -> f32 {
        Self::ROW.zero as f32
    }

    /// 码域上界 `2^bits`：零点必须落在 `0..domain` 内。
    fn domain() -> u32 {
        Self::ROW.domain()
    }

    /// 标量代价：每元素一次减法一次乘法；读打包字节（`n / cpb` 个），写 `n` 个 f32。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        cost_packed(shape, Self::cpb())
    }

    /// 确定性样本：`q`（打包字节，`len / cpb` 个）、`y`（`len` 个 f32）。
    fn sample(len: usize) -> Sample {
        sample_packed(len, Self::cpb())
    }

    /// 标量参考：逐字节解包，与内核同一公式。
    ///
    /// # Errors
    ///
    /// 样本缺 `q` 缓冲时返回说明。
    fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        Ok(reference_packed(sample.u8("q")?, Self::cpb(), Self::zero()))
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 缓冲长度与打包宽度不符、标度为零或零点越出码域时返回说明。
    fn forward(scale: f32, zero: f32, q: &[u8], out: tensor::ViewMut<'_, f32>) -> diagnostic::Result<()> {
        forward_packed(scale, zero, q, out.into_slice_mut(), Self::cpb())
    }
}

/// 生成一份「打包反量化」精度变体的完整公开面：`SCALE`/`ZERO` 常量、代价/样本/参考/
/// `forward` 入口、[`annotation::model`] 别名与测试挂载。行为全部来自 [`Packed`] 的默认
/// 实现，`dequant_i2`/`dequant_i4` 只差一行 [`crate::quant::Row`]。
#[macro_export]
macro_rules! dequant_packed {
    ($name:ident, $alias:ident, $op:tt, $row:ident) => {
        /// 反量化标度（2 的负幂网格，f32 精确）。
        pub const SCALE: f32 = $crate::dequant::SCALE;

        /// 打包对称零点（取自配置表 `$row`）。
        pub const ZERO: f32 = $row.zero as f32;

        // 单轴定义与形状投影复用打包骨架。
        pub use $crate::dequant::{AXES, project_shape};

        #[doc = concat!(
            "`", $op, "` 的标记类型：配置取自 `crate::quant::", stringify!($row),
            "`，行为由 `crate::dequant::Packed` 派生。"
        )]
        pub struct $name;

        impl $crate::dequant::Packed for $name {
            const ROW: $crate::quant::Row = $row;
        }

        /// 标量代价：每元素一次减法一次乘法；读打包字节，写 `n` 个 f32。
        pub fn cost(shape: &$crate::model::ShapeCtx) -> diagnostic::Result<$crate::model::Cost> {
            <$name as $crate::dequant::Packed>::cost(shape)
        }

        /// 确定性样本：`q`（打包字节）、`y`（`n` 个 f32）。
        pub fn sample(len: usize, _: usize) -> $crate::model::Sample {
            <$name as $crate::dequant::Packed>::sample(len)
        }

        /// 标量参考：逐字节解包，与内核同一公式。
        pub fn reference(sample: &$crate::model::Sample) -> diagnostic::Result<Vec<f32>> {
            <$name as $crate::dequant::Packed>::reference(sample)
        }

        /// 内核本体：结果写进调用方给的缓冲。
        pub fn forward(scale: f32, zero: f32, q: &[u8], out: tensor::ViewMut<'_, f32>) -> diagnostic::Result<()> {
            <$name as $crate::dequant::Packed>::forward(scale, zero, q, out)
        }

        // 基座契约（ADR 0050）：转发本模块的骨架入口；粒度直接取配置表的每字节码数。
        $crate::operator_via!(
            @impl $name,
            $op,
            1,
            (4096, 4096),
            false,
            $crate::quant::$row.cpb as u32,
            self::AXES,
            self::project_shape,
            self::cost,
            self::sample,
            self::reference
        );

        #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = $op)]
        #[doc = concat!("`", $op, "` 的语义别名。")]
        pub type $alias = $name;

        #[cfg(test)]
        mod tests {
            use super::*;
            $crate::dequant::test_support::dequant_packed_tests!(
                sample,
                reference,
                forward,
                SCALE,
                ZERO,
                <$name as $crate::dequant::Packed>::cpb(),
                <$name as $crate::dequant::Packed>::domain() as f32,
                -1.0,
                f32::NAN
            );
        }
    };
}

#[cfg(test)]
mod test_support {
    include!("../tests/kernel_dequant_test_support.rs");
}

pub mod dequant_i2 {
    //! 2 位反量化：打包字节 `q`（每字节四个 2 位码），`y = scale * (code - 2)`。

    use crate::quant::I2;

    crate::dequant_packed!(DequantI2, DequantI2F32, "dequant_i2", I2);
}

pub mod dequant_i4 {
    //! 4 位反量化：打包字节 `q`（每字节两个 nibble），`y = scale * (nibble - 8)`。

    use crate::quant::I4;

    crate::dequant_packed!(DequantI4, DequantI4F32, "dequant_i4", I4);
}

pub mod i8 {
    //! 8 位反量化：`y[n] = scale * (q[n] - zero)`。
    use crate::dequant;
    use crate::model::{Buf, Cost, Sample, ShapeCtx, Values, cost_n};
    use diagnostic::{Code, Error};
    use tensor::{View, ViewMut};

    // 单轴定义复用 unary 骨架。
    pub use crate::unary::AXES;

    /// 样本固定的反量化参数。
    pub const SCALE: f32 = 0.25;
    /// 样本固定的零点。
    pub const ZERO: f32 = 8.0;

    // 形状投影复用 unary 骨架。
    pub use crate::unary::project_shape;

    /// 标量代价：每元素一次减法一次乘法，读一次写一次。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        cost_n(shape, "dequant", 2.0, 1)
    }

    /// 确定性样本：`q`（f32 承载的量化整数）、`y`，标量 `scale`/`zero`。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            bufs: vec![
                Buf {
                    name: "q",
                    values: Values::F32((0..len).map(|i| (i % 16) as f32).collect()),
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
    }

    /// 标量参考：与内核同一公式。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let q = sample.f32("q")?;
        Ok(q.iter().map(|value| SCALE * (value - ZERO)).collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输入与输出等长、标度非零有限、零点在 `0..=15` 码域内，否则返回说明。
    pub fn forward(
        scale: f32,
        zero: f32,
        q: View<'_, f32>,
        out: ViewMut<'_, f32>,
    ) -> diagnostic::Result<()> {
        let q = q.as_slice();
        let out = out.into_slice_mut();
        if q.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：q {}、out {}", q.len(), out.len()),
            ));
        }
        dequant::check_affine(scale, zero, 4)?;
        for (out, q) in out.iter_mut().zip(q) {
            *out = scale * (q - zero);
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct DequantI8;

    // 基座契约（ADR 0050）：本模块自带具体入口，直接转发。
    crate::operator_here!(
        DequantI8,
        "dequant_i8",
        1,
        (4096, 4096),
        false,
        self::reference
    );

    /// 反量化的具体语义别名。
    #[annotation::model(name = "dequant_i8", shape = [4096, 4096], dtype = "f32", skeleton = "dequant")]
    pub type DequantI8F32 = DequantI8;

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel_dequant_i8_tests.rs");
    }
}

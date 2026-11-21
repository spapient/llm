//! 职责：conv2d 声明档输入形状常量（H/W）。
use super::*;

/// 声明档输入高。
pub const H: usize = 4;
/// 声明档输入宽。
pub const W: usize = 4;
/// 声明档输入通道。
pub const CIN: usize = 2;
/// 声明档输出通道。
pub const COUT: usize = 3;
/// 声明档核边长。
pub const K: usize = 3;
/// 声明档步长。
pub const S: usize = 1;
/// 声明档补零（`(K-1)/2`，保持 same 输出）。
pub const P: usize = 1;
/// 声明档输出高（same 卷积与输入同形）。
pub const OH: usize = 4;
/// 声明档输出宽。
pub const OW: usize = 4;
/// 每行 im2col 的收缩长度：`K·K·CIN`。
pub const PATCH: usize = K * K * CIN;

fn conv() -> crate::conv::Conv2d {
    crate::conv::Conv2d::new(
        CIN,
        COUT,
        ConvGeom {
            kh: K,
            kw: K,
            sh: S,
            sw: S,
            ph: P,
            pw: P,
        },
    )
}

/// 声明档样本：固定几何下的输入、权重、偏置与输出缓冲。
pub fn sample(_: usize, _: usize) -> Sample {
    Sample {
        m: OH * OW,
        n: COUT,
        scalars: vec![("N", 1)],
        buffers: vec![
            x_buffer(H * W * CIN),
            Buffer {
                name: "weight",
                values: Values::F32(
                    (0..PATCH * COUT)
                        .map(|i| 0.05 + (i % 7) as f32 * 0.03)
                        .collect(),
                ),
            },
            Buffer {
                name: "bias",
                values: Values::F32(vec![0.4; COUT]),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; OH * OW * COUT]),
            },
        ],
    }
}

/// 标量参考：复用 `im2col + GEMM` 的真实现算一遍（不是第二份卷积循环）。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let weight = sample.f32("weight")?;
    let bias = sample.f32("bias")?;
    let mut out = vec![0.0f32; OH * OW * COUT];
    conv().forward_f32(x, weight, bias, 1, (H, W), &mut out)?;
    Ok(out)
}

/// 声明档前向：单样本 `H×W×CIN → OH×OW×COUT`。
pub fn forward(x: &[f32], weight: &[f32], bias: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    conv().forward_f32(x, weight, bias, 1, (H, W), out)
}

/// 标量代价：每输出 `2·K²·CIN` 次浮点运算；读 x/weight/bias 三遍、写 y 一遍。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    crate::model::cost_n(shape, "conv2d_f32", 2.0 * PATCH as f64, 3)
}

classic_op!(Conv2dModel, Conv2d, "conv2d_f32", 3, 16, 3);

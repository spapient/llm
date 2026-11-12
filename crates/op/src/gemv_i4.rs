//! `gemv_i4` 的算子语义：代价、样本与标量参考。
//!
//! 模板在 `ops/gemv_i4.k`，本模块是它的第二份实现——标量参考（DESIGN §11.1）：片段只与这里
//! 对拍，片段之间不互相对拍。内核实参不在这里手写：它由 `.k` 的签名与样本派生（[`crate::sig`]）。

use crate::model::{Buffer, Model, Sample, Values};
use diagnostic::{Code, Error};
use einsum::{Bits, Cost, Expr, Extents, Formula, Operand};
use prof::atom::DType;

/// 量化分组。
///
/// 声明里 `needs.quant.group = ["32"]` 是**能力请求**（原语必须支持多大的组），这里是样本与
/// 代价实际采用的取值。`cutile` 的 `the_model_group_matches_the_declaration` 测试保证两者
/// 不漂移。
pub const GROUP: usize = 32;

/// 本算子的语义表。
pub const MODEL: Model = Model {
    group: GROUP as u32,
    // 生产形状：一份 4096 方阵的权重就是一次真实调用；`verify` 的形状矩阵也用它。
    shape: (4096, 4096),
    dtype: DType::F32,
    cost,
    sample,
    reference,
};

/// 本算子的 einsum 语义：`w[nm] · x[m] · scales[ng] → y[n]`。
///
/// 这条式子就是代价模型的全部来源：`flops` 由参与运算的轴（`n`、`m`）推出，`bytes` 由四个
/// 操作数各自的轴与位宽推出。`g` 是 `m` 每 [`GROUP`] 个元素切出来的派生轴，只计流量、不计
/// 运算次数——把它乘进去会让 `flops` 多出一个因子。
pub const EXPR: &str = "nm,m,ng,n->n";

/// 表达式 + 四个操作数的位宽/方向；解析一次，之后复用到任意形状。
fn formula() -> Result<Formula, Error> {
    Formula::new(
        Expr::parse(EXPR)?,
        vec![
            Operand::read("w", Bits::Four),
            Operand::read("x", Bits::ThirtyTwo),
            Operand::read("scales", Bits::ThirtyTwo),
            Operand::write("y", Bits::ThirtyTwo),
        ],
    )
    .map_err(|error| error.with("op::gemv_i4::formula"))
}

/// 轴长度：`n` 是输出、`m` 是规约元素，`g = m / GROUP` 由 `m` 推出。
fn extents(m: usize, n: usize) -> Result<Extents, Error> {
    Extents::new([('n', n), ('m', m)])
        .and_then(|extents| extents.group('g', 'm', GROUP))
        .map_err(|error| error.with("op::gemv_i4::extents"))
}

/// 标量代价：完全由 [`EXPR`] 与四个操作数的位宽推出，本函数不含任何流量常数。
///
/// # Errors
///
/// 形状与表达式不自洽（`n` 为 0、`m` 不是 [`GROUP`] 的整数倍）时返回说明。
pub fn cost(m: usize, n: usize) -> Result<Cost, Error> {
    formula()?
        .cost(&extents(m, n)?)
        .map_err(|error| error.with("op::gemv_i4::cost"))
}

/// 确定性样本：同一形状永远得到同一组输入（DESIGN §12.3 的"可复现"要求）。
pub fn sample(m: usize, n: usize) -> Sample {
    let x: Vec<f32> = (0..m)
        .map(|index| ((index * 7) % 13) as f32 * 0.25 - 1.0)
        .collect();
    let w: Vec<u8> = (0..n * (m / 2))
        .map(|index| ((index * 37) % 251) as u8)
        .collect();
    let groups = m / GROUP.max(1);
    let scales: Vec<f32> = (0..n * groups)
        .map(|index| 0.031_25 * (1 + (index % 4)) as f32)
        .collect();
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("N", n as i64), ("gsz", GROUP as i64)],
        buffers: vec![
            Buffer {
                name: "x",
                values: Values::F32(x),
            },
            Buffer {
                name: "w",
                values: Values::U8(w),
            },
            Buffer {
                name: "scales",
                values: Values::F32(scales),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; n]),
            },
        ],
    }
}

/// 标量参考：`y[n] = Σ_g Σ_k (nibble(w[n, g·G+k]) − 8) · scale[n, g] · x[g·G+k]`。
///
/// 分组数由 `scales` 的长度给出（`scales[n][m/G]`），因此参考实现不必另存一个组大小常数；
/// 组大小与 `m` 不整除即报错。
///
/// # Errors
///
/// 缓冲缺失、长度不足，或分组推不出偶数且能整除 `m` 的取值时返回说明。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let w = sample.u8("w")?;
    let scales = sample.f32("scales")?;
    let m = sample.m;
    let n = sample.n;

    if n == 0 {
        return Err(Error::new(Code::Invalid, "输出维度不能为 0"));
    }
    if !scales.len().is_multiple_of(n) {
        return Err(Error::new(
            Code::Invalid,
            format!("scale 长度 {} 不是 {n} 的整数倍", scales.len()),
        ));
    }
    let groups = scales.len() / n;
    if groups == 0 || !m.is_multiple_of(groups) {
        return Err(Error::new(
            Code::Invalid,
            format!("scale 长度 {} 推不出能整除 {m} 的分组数", scales.len()),
        ));
    }
    let group = m / groups;
    if !group.is_multiple_of(2) {
        return Err(Error::new(
            Code::Invalid,
            format!("分组 {group} 必须是偶数"),
        ));
    }
    if x.len() < m {
        return Err(Error::new(
            Code::Invalid,
            format!("输入长度 {} < {m}", x.len()),
        ));
    }
    if w.len() < n * (m / 2) {
        return Err(Error::new(
            Code::Invalid,
            format!("权重长度 {} < {}", w.len(), n * (m / 2)),
        ));
    }

    let mut y = vec![0.0f32; n];
    for (row, out) in y.iter_mut().enumerate() {
        let base = row * (m / 2);
        let mut acc = 0.0f32;
        for g in 0..groups {
            let scale = scales[row * groups + g];
            let packed = &w[base + g * (group / 2)..base + (g + 1) * (group / 2)];
            let xs = &x[g * group..(g + 1) * group];
            for (index, byte) in packed.iter().enumerate() {
                let low = ((i32::from(byte & 0x0F)) - 8) as f32 * scale;
                let high = ((i32::from((byte >> 4) & 0x0F)) - 8) as f32 * scale;
                acc += low * xs[2 * index];
                acc += high * xs[2 * index + 1];
            }
        }
        *out = acc;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(values: &[i8]) -> Vec<u8> {
        values
            .chunks(2)
            .map(|pair| {
                let low = (pair[0] + 8) as u8 & 0x0F;
                let high = (pair.get(1).copied().unwrap_or(0) + 8) as u8 & 0x0F;
                low | (high << 4)
            })
            .collect()
    }

    fn with(m: usize, n: usize, x: Vec<f32>, w: Vec<u8>, scales: Vec<f32>) -> Sample {
        Sample {
            m,
            n,
            scalars: Vec::new(),
            buffers: vec![
                Buffer {
                    name: "x",
                    values: Values::F32(x),
                },
                Buffer {
                    name: "w",
                    values: Values::U8(w),
                },
                Buffer {
                    name: "scales",
                    values: Values::F32(scales),
                },
            ],
        }
    }

    #[test]
    fn reference_dequantizes_and_dots() {
        // 1 行、1 组、4 个权重：[1, 2, 3, 4]，scale = 0.5，x = [1, 1, 1, 1]
        let w = packed(&[1, 2, 3, 4]);
        let sample = with(4, 1, vec![1.0; 4], w, vec![0.5]);
        let y = reference(&sample).expect("参考可算");
        assert_eq!(y.len(), 1);
        assert!((y[0] - 5.0).abs() < 1e-6, "0.5*(1+2+3+4)=5，实际 {}", y[0]);
    }

    #[test]
    fn reference_handles_multiple_groups_and_rows() {
        let m = 8;
        let n = 2;
        let w = packed(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 2, 2, 2, 2]);
        let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
        let sample = with(m, n, x, w, vec![0.25, 0.5, 1.0, 1.0]);
        let y = reference(&sample).expect("参考可算");
        assert_eq!(y.len(), n);
        assert!(y[0] > 0.0);
        assert!(y[1] > y[0], "第二行权重更大");
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        // 输入太短。
        assert!(reference(&with(4, 1, vec![1.0], vec![0u8; 2], vec![1.0])).is_err());
        // 权重太短。
        assert!(reference(&with(4, 1, vec![1.0; 4], vec![0u8; 1], vec![1.0])).is_err());
        // 没有分组。
        assert!(reference(&with(4, 1, vec![1.0; 4], vec![0u8; 2], vec![])).is_err());
        // 分组数为奇数：m=3、1 组 → 组大小 3。
        assert!(reference(&with(3, 1, vec![1.0; 3], vec![0u8; 2], vec![1.0])).is_err());
    }

    #[test]
    fn the_production_shape_is_a_legal_input() {
        let (m, n) = MODEL.shape;
        assert!(m.is_multiple_of(GROUP), "生产形状的 M 必须是组大小的整数倍");
        assert!(cost(m, n).is_ok(), "生产形状必须能算出代价");
        assert_eq!(MODEL.dtype, DType::F32, "样本的输出缓冲是 f32");
    }

    #[test]
    fn cost_comes_from_the_declared_expression_without_hand_written_constants() {
        let cost = self::cost(4096, 4096).expect("可估");
        assert_eq!(cost.flops, 2.0 * 4096.0 * 4096.0);
        assert_eq!(
            cost.bytes,
            4096 * 4096 / 2 + 4096 * 4 + 4096 * 128 * 4 + 4096 * 4
        );
    }

    #[test]
    fn the_expression_keeps_traffic_and_arithmetic_axes_apart() {
        let axes = formula()
            .expect("表达式与操作数配对")
            .axes(&extents(64, 4).expect("轴"))
            .expect("可分解");
        let reduced: Vec<char> = axes.reduced().map(|axis| axis.label()).collect();
        let grouped: Vec<char> = axes.grouped().map(|axis| axis.label()).collect();
        assert_eq!(reduced, ['m'], "只有输入维参与求和");
        assert_eq!(grouped, ['g'], "量化分组是派生轴");
        assert_eq!(axes.output_shape(), [4]);
    }

    #[test]
    fn a_small_shape_pins_every_term_so_a_factor_cannot_hide() {
        // 大形状里 `m/32` 之类的因子很容易互相抵消，小形状才能逐项钉住：
        // w = 4·64 个 int4（128 B）、x = 64·4、scales = 4·2·4、y = 4·4。
        let cost = self::cost(64, 4).expect("可估");
        assert_eq!(cost.flops, 2.0 * 64.0 * 4.0);
        assert_eq!(cost.bytes, 128 + 256 + 32 + 16);
    }

    #[test]
    fn a_shape_that_does_not_fit_the_grouping_is_an_error() {
        let error = self::cost(33, 4).expect_err("33 不是 32 的整数倍");
        assert!(error.to_string().contains("整数倍"), "{error}");
        assert_eq!(error.code(), diagnostic::Code::Invalid);
    }

    #[test]
    fn sample_is_deterministic_and_sized_to_the_group() {
        let first = sample(64, 4);
        assert_eq!(first, sample(64, 4));
        assert_eq!(first.u8("w").expect("w").len(), 4 * 32);
        assert_eq!(first.f32("scales").expect("scales").len(), 4 * (64 / GROUP));
        assert_eq!(first.f32("y").expect("y").len(), 4);
    }

    #[test]
    fn call_arguments_follow_the_template_signature() {
        let signature = crate::sig::Signature::parse("gemv_i4", include_str!("../ops/gemv_i4.k"))
            .expect("签名可解析");
        assert_eq!(
            signature.call_args(&sample(64, 4)).expect("实参"),
            "x, w, scales, y, 64, 4, 32"
        );
        assert_eq!(signature.output(), Some("y"));
    }
}

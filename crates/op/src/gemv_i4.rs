//! `gemv_i4` 的算子语义：代价、样本与标量参考。
//!
//! 模板在 `ops/gemv_i4.k`，本模块是它的第二份实现——标量参考（DESIGN §11.1）：片段只与这里
//! 对拍，片段之间不互相对拍。内核实参不在这里手写：它由 `.k` 的签名与样本派生（[`crate::sig`]）。

use crate::model::{AxisRole, AxisSpec, Buffer, Model, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};
use einsum::{Bits, Cost, Expr, Formula, Operand};
use prof::atom::DType;
use std::sync::OnceLock;

/// 量化分组。
///
/// 声明里 `needs.quant.group = ["32"]` 是**能力请求**（原语必须支持多大的组），这里是样本与
/// 代价实际采用的取值。`cutile` 的 `the_model_group_matches_the_declaration` 测试保证两者
/// 不漂移。
pub const GROUP: usize = 32;

/// 四位无符号码的解包结果：`nibble − 8`。
///
/// 查表而不是每次做 `i32 → f32` 转换：解包是这条内核最内层的循环，每元素省一次转换就是
/// 整个 GEMV 的省；16 个取值都是精确的小整数，表里没有舍入。
const DEQUANT: [f32; 16] = [
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
];

/// 本算子的基本轴：`m` 是规约轴（输入宽度），`n` 是输出轴。
///
/// 字母属于这条表达式，角色属于层：`nn::Linear` 按 [`AxisRole::Input`] / [`AxisRole::Output`]
/// 填宽度，不需要知道它们叫 `m` 和 `n`。
pub const AXES: &[AxisSpec] = &[
    AxisSpec {
        role: AxisRole::Input,
        label: 'm',
    },
    AxisSpec {
        role: AxisRole::Output,
        label: 'n',
    },
];

/// 本算子的语义表。
pub const MODEL: Model = Model {
    group: GROUP as u32,
    // 生产形状：一份 4096 方阵的权重就是一次真实调用；`verify` 的形状矩阵也用它。
    shape: (4096, 4096),
    dtype: DType::F32,
    axes: AXES,
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

/// 表达式 + 四个操作数的位宽/方向；**整个进程只解析一次**，之后复用到任意形状。
///
/// [`cost`] 会被层的下界（每层一次）、`doctor` 的 roofline 与形状矩阵分别调用，而表达式与
/// 形状无关：解析一次把结果留在静态里，热路径上只剩一次原子读。初始化失败也一起缓存——
/// 同一条常量表达式不会第二次成功。
fn formula() -> Result<&'static Formula, Error> {
    static FORMULA: OnceLock<Result<Formula, Error>> = OnceLock::new();
    FORMULA
        .get_or_init(|| {
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
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// 标量代价：完全由 [`EXPR`] 与四个操作数的位宽推出，本函数不含任何流量常数。
///
/// 调用方只给基本轴（`m`、`n`）；分组轴 `g = m / GROUP` 在这里声明——分组是算子语义，不是
/// 每个层都该抄一遍的常数。
///
/// # Errors
///
/// 形状与表达式不自洽（缺 `m` / `n`、`n` 为 0、`m` 不是 [`GROUP`] 的整数倍）时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    let extents = shape
        .extents_with_group('g', 'm', GROUP)
        .map_err(|error| error.with("op::gemv_i4::cost"))?;
    formula()?
        .cost(&extents)
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
/// 算术本身在 [`forward`]：运行时重放每一步都要算，不该为了"多一层包装"再付一次拷贝与一次
/// 分配；参考与重放共用同一段循环，对拍比较的才是同一份语义。
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
    forward(&x[..m], &w[..n * (m / 2)], scales, &mut y)?;
    Ok(y)
}

/// 内核本体：`m = x.len()`、`n = out.len()`，结果**写进调用方给的缓冲**。
///
/// 与 [`reference()`] 是同一段算术，区别只在形状从缓冲长度推出、输出不自带分配：重放时每一步
/// 都要算，稳定态不该再向分配器要一块新的 `Vec`。
///
/// 分组数仍由 `scales` 的长度给出（`scales[n · m/G]`），所以组大小不必另存常数。
///
/// # Errors
///
/// `out` 为空、`scales` 长度不是 `n` 的整数倍、分组数为 0、分组不是偶数、分组不能整除 `m`，
/// 或 `w` 长度不足 `n · m / 2` 时返回说明。
pub fn forward(x: &[f32], w: &[u8], scales: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    let m = x.len();
    let n = out.len();

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
    if w.len() < n * (m / 2) {
        return Err(Error::new(
            Code::Invalid,
            format!("权重长度 {} < {}", w.len(), n * (m / 2)),
        ));
    }

    // 每行只看自己那一段：切片一次算好，内层就不再做 `row × 列数 + g` 的乘加与边界检查。
    // 每个分组只乘一次 `scale`（整组共用一个），而不是每个元素各乘一次。
    let half = m / 2;
    let per_group = group / 2;
    for (row, value) in out.iter_mut().enumerate() {
        let w_row = &w[row * half..][..half];
        let s_row = &scales[row * groups..][..groups];
        let mut acc = 0.0f32;
        for ((packed, scale), xs) in w_row
            .chunks_exact(per_group)
            .zip(s_row)
            .zip(x.chunks_exact(group))
        {
            let mut dot = 0.0f32;
            for (byte, pair) in packed.iter().zip(xs.as_chunks::<2>().0) {
                dot += DEQUANT[usize::from(*byte & 0x0F)] * pair[0];
                dot += DEQUANT[usize::from(*byte >> 4)] * pair[1];
            }
            acc += dot * scale;
        }
        *value = acc;
    }
    Ok(())
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

    /// 测试里只关心 `(m, n)`：它们走的是层那条路径——把宽度填进算子声明的具名轴。
    fn at(m: usize, n: usize) -> Result<Cost, Error> {
        cost(&ShapeContext::new([('m', m), ('n', n)])?)
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
    fn forward_writes_into_the_caller_buffer_and_matches_the_reference() {
        let m = 8;
        let n = 2;
        let w = packed(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 2, 2, 2, 2]);
        let x: Vec<f32> = (0..m).map(|index| index as f32 + 1.0).collect();
        let scales = vec![0.25, 0.5, 1.0, 1.0];
        let expected =
            reference(&with(m, n, x.clone(), w.clone(), scales.clone())).expect("参考可算");

        // 复用一块脏缓冲：结果必须是覆盖写，读到的不能是上一次的残留。
        let mut out = vec![f32::NAN; n];
        forward(&x, &w, &scales, &mut out).expect("内核可算");
        assert_eq!(out, expected, "内核与参考必须是同一段算术");
    }

    /// 手写一遍求和，不经过本文件的任何循环：分组边界、`scale` 落在哪、行偏移都由测试自己算。
    ///
    /// 这是替换内层循环（查表 + 按组切片 + 每组一次乘法）之后仍然钉住算术的判据——
    /// `reference` 与 `forward` 是同一段代码，只对它俩对拍等于自己跟自己对。
    #[test]
    fn the_kernel_agrees_with_a_hand_written_sum() {
        let m = 8;
        let n = 2;
        // 打包只支持 [-8, 7]：`value + 8` 必须落在 4 位里，所以 8 会溢出成 -8。
        let w = packed(&[1, 2, 3, 4, 5, 6, 7, -8, -1, 0, 1, 2, 3, -4, 5, -6]);
        let x: Vec<f32> = (0..m).map(|index| index as f32 * 0.5 - 1.0).collect();
        // 行 0：组 0 = 0.25、组 1 = 0.5；行 1：组 0 = 1.0、组 1 = -1.0。
        let scales = vec![0.25, 0.5, 1.0, -1.0];

        let mut out = vec![f32::NAN; n];
        forward(&x, &w, &scales, &mut out).expect("内核可算");

        let row = |weights: [f32; 8], scale: [f32; 2]| {
            scale[0] * (0..4).map(|k| weights[k] * x[k]).sum::<f32>()
                + scale[1] * (0..4).map(|k| weights[4 + k] * x[4 + k]).sum::<f32>()
        };
        let expected = [
            row([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0], [0.25, 0.5]),
            row([-1.0, 0.0, 1.0, 2.0, 3.0, -4.0, 5.0, -6.0], [1.0, -1.0]),
        ];
        assert!((out[0] - expected[0]).abs() < 1e-5, "{:?}", out);
        assert!((out[1] - expected[1]).abs() < 1e-5, "{:?}", out);
    }

    #[test]
    fn forward_rejects_a_buffer_that_does_not_fit_the_shapes() {
        let error = forward(&[1.0; 4], &[0u8; 1], &[1.0], &mut [0.0; 1]).expect_err("权重太短");
        assert_eq!(error.code(), diagnostic::Code::Invalid);
        assert!(
            forward(&[1.0; 4], &[0u8; 2], &[1.0], &mut []).is_err(),
            "空输出"
        );
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
        assert!(at(m, n).is_ok(), "生产形状必须能算出代价");
        assert_eq!(MODEL.dtype, DType::F32, "样本的输出缓冲是 f32");
    }

    #[test]
    fn cost_comes_from_the_declared_expression_without_hand_written_constants() {
        let cost = at(4096, 4096).expect("可估");
        assert_eq!(cost.flops, 2.0 * 4096.0 * 4096.0);
        // 三个输入都是读：int4 权重、输入、缩放因子；只有输出是写。
        assert_eq!(
            cost.read,
            4096 * 4096 / 2 + 4096 * 4 + 4096 * 128 * 4,
            "权重 8 MiB + 输入 16 KiB + 缩放 2 MiB"
        );
        assert_eq!(cost.write, 4096 * 4, "写方向只有输出那一份");
        assert_eq!(
            cost.bytes(),
            4096 * 4096 / 2 + 4096 * 4 + 4096 * 128 * 4 + 4096 * 4,
            "合计与改造前逐字节相同"
        );
    }

    #[test]
    fn the_expression_keeps_traffic_and_arithmetic_axes_apart() {
        let axes = formula()
            .expect("表达式与操作数配对")
            .axes(
                &ShapeContext::new([('m', 64), ('n', 4)])
                    .expect("轴")
                    .extents_with_group('g', 'm', GROUP)
                    .expect("可分组"),
            )
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
        let cost = at(64, 4).expect("可估");
        assert_eq!(cost.flops, 2.0 * 64.0 * 4.0);
        assert_eq!(cost.read, 128 + 256 + 32);
        assert_eq!(cost.write, 16);
        assert_eq!(cost.bytes(), 128 + 256 + 32 + 16);
    }

    #[test]
    fn a_shape_that_does_not_fit_the_grouping_is_an_error() {
        let error = at(33, 4).expect_err("33 不是 32 的整数倍");
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

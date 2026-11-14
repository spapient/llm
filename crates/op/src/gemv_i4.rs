//! `gemv_i4` 的算子语义：4 位打包权重 + 每 [`GROUP`] 个元素一条标度。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"4 位打包"这一份格式契约：表达式、操作数位宽、声明粒度、
//! 标度取值与内核本体。模板在 `ops/gemv_i4.k`，本模块是它的第二份实现（cutile README §11.1）：
//! 片段只与这里对拍，片段之间不互相对拍。内核实参不在这里手写：它由 `.k` 的签名与样本派生
//! （[`crate::sig`]）。
//!
//! # 哪些是编译期的，哪些是数据
//!
//! [`I4`] 把**声明粒度**放进类型（`Quantized<32>`）：它决定代价里的派生轴除数、生产形状的
//! 合法性与样本里每行的标度个数，三件都在编译期定死。实际的**分组大小**不在这里——模板把
//! 它当 `int gsz` 实参收，所以 [`forward`] 仍按 `scales` 的长度推分组：一份 32 分组的权重与
//! 一份 8 分组的权重是同一个类型的两次调用，而不是两份单态化副本。

use crate::einsum::{Bits, FormulaCache, OperandSpec};
use crate::gemv::{self, Format};
use diagnostic::{Code, Error};

/// 量化分组。
///
/// 声明里 `needs.quant.group = ["32"]` 是**能力请求**（原语必须支持多大的组），这里是样本与
/// 代价实际采用的取值。`cutile` 的 `the_model_group_matches_the_declaration` 测试保证两者
/// 不漂移，`#[model]` 生成的编译期断言保证生产形状是它的整数倍。
pub const GROUP: usize = 32;

/// 四位无符号码的解包结果：`nibble − 8`。
///
/// 查表而不是每次做 `i32 → f32` 转换：解包是这条内核最内层的循环，每元素省一次转换就是
/// 整个 GEMV 的省；16 个取值都是精确的小整数，表里没有舍入。
const DEQUANT: [f32; 16] = [
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
];

/// 标度取值：`0.03125` 的 1..=4 倍。
///
/// 全部落在 2 的负幂网格上，因此"标度 × 整数"与累加在 f32 里都是精确的：对拍结果的
/// `max_rel_err` 反映的是实现是否一致，而不是浮点本来就会抖。
fn scale_of(index: usize) -> f32 {
    0.031_25 * (1 + (index % 4)) as f32
}

/// 4 位打包格式的语义类型：声明粒度 [`GROUP`]、每行 `m / 2` 个字节。
///
/// `#[model]` 由这份注解生成 [`MODEL`]：生产形状、输出类型与三个单态化入口。别名不实现
/// [`Format`] 就编译不过，所以"注解说的格式"与"实际实现的格式"不可能分家。
#[annotation::model(name = "gemv_i4", shape = [4096, 4096], dtype = "f32")]
pub type I4 = gemv::Quantized<GROUP>;

impl<const G: usize> Format for gemv::Quantized<G> {
    type Elem = u8;

    /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 [`GROUP`] 个元素切出来的派生轴，
    /// 只计流量、不计运算次数。
    const EXPR: &'static str = "nm,m,ng,n->n";

    /// 三个输入都是读，只有输出是写；位宽就是解包前后各占多少。
    const OPERANDS: &'static [OperandSpec] = &[
        OperandSpec::read("w", Bits::Four),
        OperandSpec::read("x", Bits::ThirtyTwo),
        OperandSpec::read("scales", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ];

    const GRAIN: usize = G;

    const DERIVED: Option<(char, char, usize)> = Some(('g', 'm', G));

    const SCALES: Option<fn(usize) -> f32> = Some(scale_of);

    const STEP: usize = G;

    fn cache() -> &'static FormulaCache {
        // 缓存落在这个 impl 里：`Quantized<G>` 的表达式与粒度无关，一份缓存覆盖所有实例。
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn row(m: usize) -> usize {
        // 两个 4 位码共用一个字节：粒度必须是正偶数，否则"每行 m / 2 字节"没有意义。
        // 单态化时求值，运行期一条指令都不留。
        const {
            assert!(
                G > 0 && G.is_multiple_of(2),
                "4 位打包的粒度必须是正偶数：两个元素共用一个字节"
            )
        };
        m / 2
    }

    fn weights(m: usize, n: usize) -> Vec<u8> {
        (0..n * (m / 2))
            .map(|index| ((index * 37) % 251) as u8)
            .collect()
    }

    fn forward(
        x: &[f32],
        weights: &[u8],
        scales: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        let m = x.len();
        let n = out.len();

        let (groups, group) = quantized_groups(m, n, scales.len(), weights.len())?;

        // 每行只看自己那一段：切片一次算好，内层就不再做 `row × 列数 + g` 的乘加与边界检查。
        let half = m / 2;
        let per_group = group / 2;
        for (row, value) in out.iter_mut().enumerate() {
            let w_row = &weights[row * half..][..half];
            let s_row = &scales[row * groups..][..groups];
            *value = dot_row(w_row, x, s_row, per_group, group);
        }
        Ok(())
    }
}

/// 量化格式的前置约束：骨架的分组推导 + 本节格式自己的"分组必须是偶数"。
///
/// 两个 4 位码共用一个字节，所以分组为奇数时"每行 m / 2 字节"没有意义。
fn quantized_groups(
    m: usize,
    n: usize,
    scales: usize,
    weights: usize,
) -> diagnostic::Result<(usize, usize)> {
    let (groups, group) = gemv::scaled_groups(m, n, scales)?;
    if !group.is_multiple_of(2) {
        return Err(Error::new(
            Code::Invalid,
            format!("分组 {group} 必须是偶数"),
        ));
    }
    gemv::required_weights(weights, n, m / 2)?;
    Ok((groups, group))
}

/// 一行的点积：按分组推进，组内解包两个 4 位码，整组共用同一个标度。
///
/// 累加顺序与朴素写法逐位一致（同一个字节的低半字节在前），否则舍入顺序一变，
/// 对拍就不再是同一个算术。
fn dot_row(packed_row: &[u8], x: &[f32], s_row: &[f32], per_group: usize, group: usize) -> f32 {
    let mut acc = 0.0f32;
    for ((packed, scale), xs) in packed_row
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
    acc
}

/// 内核本体：`m = x.len()`、`n = out.len()`，结果**写进调用方给的缓冲**。
///
/// 与 [`gemv::reference`] 是同一段算术，区别只在形状从缓冲长度推出、输出不自带分配：重放时
/// 每一步都要算，稳定态不该再向分配器要一块新的 `Vec`。分组仍由 `scales` 的长度给出，所以
/// 同一份权重按 8 或 32 分组调用都能算。
///
/// # Errors
///
/// `out` 为空、`scales` 长度不是 `n` 的整数倍、分组数为 0、分组不是偶数、分组不能整除 `m`，
/// 或 `weights` 长度不足 `n · m / 2` 时返回说明。
pub fn forward(
    x: &[f32],
    weights: &[u8],
    scales: &[f32],
    out: &mut [f32],
) -> diagnostic::Result<()> {
    <I4 as Format>::forward(x, weights, scales, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Buffer, Cost, Sample, ShapeContext, Values};
    use prof::atom::DType;

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
        gemv::cost::<I4>(&ShapeContext::new([('m', m), ('n', n)])?)
    }

    #[test]
    fn reference_dequantizes_and_dots() {
        // 1 行、1 组、4 个权重：[1, 2, 3, 4]，scale = 0.5，x = [1, 1, 1, 1]
        let w = packed(&[1, 2, 3, 4]);
        let sample = with(4, 1, vec![1.0; 4], w, vec![0.5]);
        let y = gemv::reference::<I4>(&sample).expect("参考可算");
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
        let y = gemv::reference::<I4>(&sample).expect("参考可算");
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
        let expected = gemv::reference::<I4>(&with(m, n, x.clone(), w.clone(), scales.clone()))
            .expect("参考可算");

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
        assert!(gemv::reference::<I4>(&with(4, 1, vec![1.0], vec![0u8; 2], vec![1.0])).is_err());
        // 权重太短。
        assert!(gemv::reference::<I4>(&with(4, 1, vec![1.0; 4], vec![0u8; 1], vec![1.0])).is_err());
        // 没有分组。
        assert!(gemv::reference::<I4>(&with(4, 1, vec![1.0; 4], vec![0u8; 2], vec![])).is_err());
        // 分组数为奇数：m=3、1 组 → 组大小 3。
        assert!(gemv::reference::<I4>(&with(3, 1, vec![1.0; 3], vec![0u8; 2], vec![1.0])).is_err());
    }

    #[test]
    fn the_production_shape_is_a_legal_input() {
        let (m, n) = MODEL.shape;
        assert!(m.is_multiple_of(GROUP), "生产形状的 M 必须是组大小的整数倍");
        assert!(at(m, n).is_ok(), "生产形状必须能算出代价");
        assert_eq!(MODEL.dtype, DType::F32, "样本的输出缓冲是 f32");
        assert_eq!(MODEL.group as usize, GROUP, "注册的粒度就是注解的粒度");
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
        let first = gemv::sample::<I4>(64, 4);
        assert_eq!(first, gemv::sample::<I4>(64, 4));
        assert_eq!(first.u8("w").expect("w").len(), 4 * 32);
        assert_eq!(first.f32("scales").expect("scales").len(), 4 * (64 / GROUP));
        assert_eq!(first.f32("y").expect("y").len(), 4);
    }

    #[test]
    fn call_arguments_follow_the_template_signature() {
        let signature = crate::sig::Signature::parse("gemv_i4", include_str!("../ops/gemv_i4.k"))
            .expect("签名可解析");
        assert_eq!(
            signature
                .call_args(&gemv::sample::<I4>(64, 4))
                .expect("实参"),
            "x, w, scales, y, 64, 4, 32"
        );
        assert_eq!(signature.output(), Some("y"));
    }
}

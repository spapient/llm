//! `gemv_f32` 的算子语义：稠密 f32 权重投影。
//!
//! 骨架在 [`crate::gemv`]，本文件只有"稠密权重"这一份格式契约：表达式、操作数位宽、分块宽度
//! 与内核本体。模板在 `ops/gemv_f32.k`，本模块是它的第二份实现（cutile README §11.1）：片段只与这里
//! 对拍，片段之间不互相对拍。内核实参不在这里手写：它由 `.k` 的签名与样本派生（[`crate::sig`]）。
//!
//! 与 [`crate::gemv_i4`] 的差别只有两点，而且两点都是**类型参数**：权重元素是 `f32` 而不是
//! 打包字节，形状粒度是 1 而不是量化分组。代价、样本、校验、遍历四条路径一份代码也没有重写。

use crate::einsum::{Bits, FormulaCache, OperandSpec};
use crate::gemv::{self, Format};
use diagnostic::{Code, Error};

/// 分块宽度：模板每轮处理这么多元素，标量参考按同一个宽度分块累加。
///
/// 它不是"形状粒度"：稠密权重没有分组约束，最后一块可以短一点（模板用 `len` 截断）。
/// 参考实现必须用同一个宽度分块，否则对拍比的是两种不同的舍入顺序。
pub const BLOCK: usize = 32;

/// 稠密格式的语义类型：分块宽度 [`BLOCK`]、任意输入宽度都合法。
///
/// `#[model]` 由这份注解生成 [`MODEL`]：生产形状、输出类型与三个单态化入口。别名不实现
/// [`Format`] 就编译不过，所以"注解说的格式"与"实际实现的格式"不可能分家。
#[annotation::model(name = "gemv_f32", shape = [4096, 4096], dtype = "f32")]
pub type F32 = gemv::Dense<BLOCK>;

impl<const B: usize> Format for gemv::Dense<B> {
    type Elem = f32;

    /// `w[nm] · x[m] → y[n]`：f32 权重没有派生轴，所以这里比 int4 少一个 `g`。
    const EXPR: &'static str = "nm,m,n->n";

    /// 三个操作数都是 f32：两个读、一个写。
    const OPERANDS: &'static [OperandSpec] = &[
        OperandSpec::read("w", Bits::ThirtyTwo),
        OperandSpec::read("x", Bits::ThirtyTwo),
        OperandSpec::write("y", Bits::ThirtyTwo),
    ];

    /// 稠密权重没有分组约束：任意输入宽度都合法。
    const GRAIN: usize = 1;

    const DERIVED: Option<(char, char, usize)> = None;

    const SCALES: Option<fn(usize) -> f32> = None;

    const STEP: usize = B;

    fn cache() -> &'static FormulaCache {
        // 缓存落在这个 impl 里：`Dense<B>` 的表达式与块宽无关，一份缓存覆盖所有实例。
        static CACHE: FormulaCache = FormulaCache::new();
        &CACHE
    }

    fn row(m: usize) -> usize {
        m
    }

    fn weights(m: usize, n: usize) -> Vec<f32> {
        (0..n * m)
            .map(|index| ((index * 37) % 251) as f32 * 0.125 - 8.0)
            .collect()
    }

    fn forward(
        x: &[f32],
        weights: &[f32],
        scales: &[f32],
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        // 块宽进源码就是常数：单态化后编译器看到的是 `chunks(32)` 而不是 `chunks(gsz)`。
        const { assert!(B > 0, "累加块宽必须是正数：`chunks(0)` 会 panic") };
        let m = x.len();
        let n = out.len();

        dense_checks(scales, weights.len(), m, n)?;

        // 每行只看自己那一段：切片一次算好，内层就不再做 `row × 列数 + base` 的乘加。
        for (row, value) in out.iter_mut().enumerate() {
            *value = dot_row::<B>(&weights[row * m..][..m], x);
        }
        Ok(())
    }
}

/// 稠密格式的前置校验：输出非空、没有标度、权重够长。
///
/// 「没有标度」是**格式**判据而不是骨架判据：拿着标度调用说明调用方把两条格式搞混了。
/// 静默忽略会让"换错算子"表现成数值正确，那比报错难查得多。
fn dense_checks(scales: &[f32], weights: usize, m: usize, n: usize) -> diagnostic::Result<()> {
    gemv::non_empty(n)?;
    if scales.is_empty() {
        gemv::required_weights(weights, n, m)?;
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("稠密格式没有标度，收到 {} 个", scales.len()),
    ))
}

/// 一行的点积：按块宽推进，块内交给迭代器展开。
fn dot_row<const B: usize>(w_row: &[f32], x: &[f32]) -> f32 {
    w_row
        .chunks(B)
        .zip(x.chunks(B))
        .map(|(w_block, x_block)| {
            w_block
                .iter()
                .zip(x_block)
                .map(|(weight, input)| weight * input)
                .sum::<f32>()
        })
        .sum()
}

/// 内核本体：`m = x.len()`、`n = out.len()`，结果**写进调用方给的缓冲**。
///
/// 与 [`gemv::reference`] 是同一段算术，区别只在形状从缓冲长度推出、输出不自带分配。
///
/// # Errors
///
/// `out` 为空或 `weights` 长度不足 `n · m` 时返回说明。
pub fn forward(x: &[f32], weights: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    <F32 as Format>::forward(x, weights, &[], out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ShapeContext, Values};
    use prof::atom::DType;

    fn shape(m: usize, n: usize) -> ShapeContext {
        MODEL.projection(m, n).expect("两个投影轴都声明了")
    }

    #[test]
    fn cost_reads_three_f32_operands_and_counts_one_multiply_add_per_element() {
        // 小形状逐项可算：w 是 m·n 个 f32、x 是 m 个、y 是 n 个；flops = 2·m·n。
        let cost = gemv::cost::<F32>(&shape(8, 4)).expect("合法形状");
        assert_eq!(cost.read, (32 + 8) * 4, "w 与 x 都是 f32");
        assert_eq!(cost.write, 4 * 4, "y 是 n 个 f32 输出");
        assert_eq!(cost.flops, 2.0 * 8.0 * 4.0);
    }

    #[test]
    fn a_zero_extent_is_rejected_instead_of_silently_priced() {
        let error = ShapeContext::new([('m', 8), ('n', 0)]).expect_err("n = 0");
        assert_eq!(error.code(), Code::Invalid);
        assert!(
            ShapeContext::new([('m', 0), ('n', 4)]).is_err(),
            "m = 0 同样不自洽"
        );
    }

    /// 稠密算子的形状粒度是 1：一条只为对齐 nibble 的约束不该溢出到 f32 权重上。
    #[test]
    fn any_width_is_a_legal_dense_shape() {
        assert_eq!(MODEL.group, 1);
        let odd = gemv::cost::<F32>(&shape(100, 3)).expect("100 不是 32 的倍数也合法");
        assert!(odd.read > 0);
        // 模板用 `len` 截断最后一块，参考实现也必须能算这个宽度。
        let sample = gemv::sample::<F32>(100, 3);
        let y = gemv::reference::<F32>(&sample).expect("分块余数不应报错");
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn sample_is_deterministic_and_sized_to_the_shape() {
        let first = gemv::sample::<F32>(64, 4);
        assert_eq!(first, gemv::sample::<F32>(64, 4));
        assert_eq!(first.f32("w").expect("w").len(), 4 * 64);
        assert_eq!(first.f32("x").expect("x").len(), 64);
        assert_eq!(first.f32("y").expect("y").len(), 4);
        assert_eq!(first.scalar("gsz"), Some(BLOCK as i64));
    }

    #[test]
    fn the_scalar_reference_matches_a_hand_computed_dot_product() {
        // 一行的权重是 1..=4，输入是 1、2、3、4 → 1+4+9+16 = 30。
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = [0.0f32; 1];
        forward(&x, &w, &mut y).expect("合法");
        assert_eq!(y, [30.0]);
    }

    #[test]
    fn the_reference_is_exact_on_the_deterministic_sample() {
        // 取值都在 2 的负幂网格上：累加在 f32 里精确，所以结果可以被逐位钉死。
        // 期望值由同一段算术算出，再用"换顺序必须等价"验一遍：分块边界的移动不改变结果。
        let sample = gemv::sample::<F32>(64, 2);
        let y = gemv::reference::<F32>(&sample).expect("合法");
        let x = sample.f32("x").expect("x");
        let w = sample.f32("w").expect("w");
        for (row, value) in y.iter().enumerate() {
            let flat: f32 = (0..64).map(|k| w[row * 64 + k] * x[k]).sum();
            assert_eq!(*value, flat, "精确网格上的分块累加必须等价于顺序累加");
        }
    }

    #[test]
    fn a_row_that_is_shorter_than_declared_is_rejected() {
        let x = [1.0f32; 4];
        let w = [1.0f32; 4];
        let mut y = [0.0f32; 2];
        let error = forward(&x, &w, &mut y).expect_err("两行只有一行权重");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("权重长度"), "{error}");
    }

    /// 稠密格式不接受标度：换错算子必须报错，而不是"算出一个看起来对的数"。
    #[test]
    fn the_dense_format_refuses_a_scales_buffer() {
        let x = [1.0f32; 4];
        let w = [1.0f32; 4];
        let mut y = [0.0f32; 1];
        let error = <F32 as Format>::forward(&x, &w, &[1.0], &mut y).expect_err("稠密格式没有标度");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("标度"), "{error}");
    }

    /// 样本的权重类型由格式决定：稠密格式的 `w` 是 f32，换进来的 u8 会被骨架拒绝。
    #[test]
    fn the_shared_skeleton_rejects_a_weight_of_the_wrong_element_type() {
        let mut sample = gemv::sample::<F32>(8, 2);
        sample
            .set("w", Values::U8(vec![0; 16]))
            .expect("长度相同即可换");
        let error = gemv::reference::<F32>(&sample).expect_err("u8 不是 f32 权重");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("f32"), "{error}");
    }

    #[test]
    fn call_arguments_follow_the_template_signature() {
        let signature = crate::sig::Signature::parse("gemv_f32", include_str!("../ops/gemv_f32.k"))
            .expect("签名可解析");
        assert_eq!(
            signature
                .call_args(&gemv::sample::<F32>(64, 4))
                .expect("实参"),
            "x, w, y, 64, 4, 32"
        );
        assert_eq!(signature.output(), Some("y"));
    }

    /// 这条算子与 `gemv_i4` 共用同一批原语，只少一个 `quant`：差异全在数据里。
    #[test]
    fn the_two_operators_differ_only_in_their_declared_quantization() {
        use crate::decl::OpDecl;
        use crate::model;
        use prof::atom::Prim;

        let dense = crate::builtin_named("gemv_f32").expect("内置");
        let quantized = crate::builtin_named("gemv_i4").expect("内置");
        let dense_decl =
            OpDecl::parse(dense.declaration(), dense.template()).expect("稠密声明合法");
        let quant_decl =
            OpDecl::parse(quantized.declaration(), quantized.template()).expect("量化声明合法");

        let prims =
            |decl: &OpDecl| -> Vec<Prim> { decl.needs.iter().map(|need| need.prim).collect() };
        assert!(!prims(&dense_decl).contains(&Prim::Quant), "稠密算子不解包");
        assert!(prims(&quant_decl).contains(&Prim::Quant));
        assert_eq!(dense_decl.skeleton, quant_decl.skeleton, "同一条骨架");
        assert_eq!(
            model("gemv_f32").map(|model| model.axes),
            model("gemv_i4").map(|model| model.axes),
            "两条算子按同一组角色取轴：层不必知道权重是 4 位还是 32 位"
        );
        assert_eq!(
            model("gemv_f32").map(|model| model.group),
            Some(1),
            "稠密的形状粒度是 1"
        );
        assert_eq!(MODEL.dtype, DType::F32, "样本的输出缓冲是 f32");
        assert_eq!(FORMAT_GRAIN, 1, "格式自己的声明粒度也是 1");
    }

    /// 与 [`MODEL`] 同源：证明 `Model::group` 确实来自 [`Format::GRAIN`]，而不是注解另写了一个数。
    const FORMAT_GRAIN: usize = <F32 as Format>::GRAIN;
}

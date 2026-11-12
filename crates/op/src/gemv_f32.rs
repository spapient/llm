//! `gemv_f32` 的算子语义：代价、样本与标量参考。
//!
//! 模板在 `ops/gemv_f32.k`，本模块是它的第二份实现（DESIGN §11.1）：片段只与这里对拍，
//! 片段之间不互相对拍。内核实参不在这里手写：它由 `.k` 的签名与样本派生（[`crate::sig`]）。
//!
//! 与 [`crate::gemv_i4`] 的差别只有两点，而且两点都是**数据**：求和表达式少了标度轴
//! （`nm,m,n->n` 而不是 `nm,m,ng,n->n`），样本里的权重是 `f32` 而不是打包字节。代价、样本、
//! 参考三条路径的代码形状完全一样——这正是"加一条算子不改机制"的样子。

use crate::model::{AxisRole, AxisSpec, Buffer, Model, Sample, ShapeContext, Values};
use diagnostic::{Code, Error};
use einsum::{Bits, Cost, Expr, Formula, Operand};
use prof::atom::DType;
use std::sync::OnceLock;

/// 分块宽度：模板每轮处理这么多元素，标量参考按同一个宽度分块累加。
///
/// 它不是"形状粒度"：稠密权重没有分组约束，最后一块可以短一点（模板用 `len` 截断）。
/// 参考实现必须用同一个宽度分块，否则对拍比的是两种不同的舍入顺序。
pub const BLOCK: usize = 32;

/// 本算子的基本轴：`m` 是规约轴（输入宽度），`n` 是输出轴。
///
/// 与 `gemv_i4` 逐字相同——两条算子是同一副骨架的两种输入格式，层的"输入宽度填哪条轴"
/// 不该因为权重从 4 位变成 32 位而改变。
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
///
/// `group = 1`：任意输入宽度都合法。量化算子那条"`m` 必须是分组整数倍"属于对齐 nibble 的
/// 约束，稠密算子不该继承它——否则"宽度 100 的层算不出下界"会变成一条莫名其妙的限制。
pub const MODEL: Model = Model {
    group: 1,
    // 生产形状：一份 4096 方阵的 f32 权重就是一次真实调用。
    shape: (4096, 4096),
    dtype: DType::F32,
    axes: AXES,
    cost,
    sample,
    reference,
};

/// 本算子的 einsum 语义：`w[nm] · x[m] → y[n]`。
///
/// 这条式子就是代价模型的全部来源：`flops` 由参与运算的轴（`n`、`m`）推出，`bytes` 由三个
/// 操作数各自的轴与位宽推出。f32 权重没有派生轴，所以这里比 int4 少一个 `g`。
pub const EXPR: &str = "nm,m,n->n";

/// 表达式 + 三个操作数的位宽/方向；整个进程只解析一次。
///
/// [`cost`] 会被层的下界（每层一次）、`doctor` 的 roofline 与形状矩阵分别调用，而表达式与
/// 形状无关：解析一次把结果留在静态里，热路径上只剩一次原子读。
fn formula() -> Result<&'static Formula, Error> {
    static FORMULA: OnceLock<Result<Formula, Error>> = OnceLock::new();
    FORMULA
        .get_or_init(|| {
            Formula::new(
                Expr::parse(EXPR)?,
                vec![
                    Operand::read("w", Bits::ThirtyTwo),
                    Operand::read("x", Bits::ThirtyTwo),
                    Operand::write("y", Bits::ThirtyTwo),
                ],
            )
            .map_err(|error| error.with("op::gemv_f32::formula"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// 标量代价：完全由 [`EXPR`] 与三个操作数的位宽推出，本函数不含任何流量常数。
///
/// 稠密权重没有派生轴，所以这里不像量化算子那样声明 `g`：调用方给的轴表直接就是表达式要的
/// 全部。**这也正是两条算子唯一的结构差异**——一条要切分组，一条不切。
///
/// # Errors
///
/// 形状与表达式不自洽（缺 `m` / `n`，或任一为 0）时返回说明。
pub fn cost(shape: &ShapeContext) -> Result<Cost, Error> {
    formula()?
        .cost(shape.extents())
        .map_err(|error| error.with("op::gemv_f32::cost"))
}

/// 确定性样本：同一形状永远得到同一组输入（DESIGN §12.3 的"可复现"要求）。
///
/// 取值刻意落在 2 的负幂网格上（权重 `k/8 - 8`、输入 `k/4 - 1`）：乘积与累加在 f32 里都是
/// 精确的，因此对拍结果的 `max_rel_err` 反映的是**实现是否一致**，而不是"浮点本来就抖"。
pub fn sample(m: usize, n: usize) -> Sample {
    let x: Vec<f32> = (0..m)
        .map(|index| ((index * 7) % 13) as f32 * 0.25 - 1.0)
        .collect();
    let w: Vec<f32> = (0..n * m)
        .map(|index| ((index * 37) % 251) as f32 * 0.125 - 8.0)
        .collect();
    Sample {
        m,
        n,
        scalars: vec![("M", m as i64), ("N", n as i64), ("gsz", BLOCK as i64)],
        buffers: vec![
            Buffer {
                name: "x",
                values: Values::F32(x),
            },
            Buffer {
                name: "w",
                values: Values::F32(w),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0; n]),
            },
        ],
    }
}

/// 标量参考：`y[n] = Σ_blocks dot(w[n, block], x[block])`。
///
/// 分块边界与模板一致（[`BLOCK`] 个元素一块，最后一块可以更短），因此比较的是同一份舍入顺序。
///
/// # Errors
///
/// 缓冲缺失或长度不足时返回说明。
pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
    let x = sample.f32("x")?;
    let w = sample.f32("w")?;
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
    if w.len() < n * m {
        return Err(Error::new(
            Code::Invalid,
            format!("权重长度 {} < {}", w.len(), n * m),
        ));
    }

    let mut y = vec![0.0f32; n];
    forward(&x[..m], &w[..n * m], &mut y)?;
    Ok(y)
}

/// 内核本体：`m = x.len()`、`n = out.len()`，结果**写进调用方给的缓冲**。
///
/// 与 [`reference()`] 是同一段算术，区别只在形状从缓冲长度推出、输出不自带分配：重放时每一步
/// 都要算，稳定态不该再向分配器要一块新的 `Vec`。
///
/// # Errors
///
/// `out` 为空或 `w` 长度不足 `n · m` 时返回说明。
pub fn forward(x: &[f32], w: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
    let m = x.len();
    let n = out.len();

    if n == 0 {
        return Err(Error::new(Code::Invalid, "输出维度不能为 0"));
    }
    if w.len() < n * m {
        return Err(Error::new(
            Code::Invalid,
            format!("权重长度 {} < {}", w.len(), n * m),
        ));
    }

    // 每行只看自己那一段：切片一次算好，内层就不再做 `row × 列数 + base` 的乘加。
    for (row, value) in out.iter_mut().enumerate() {
        let w_row = &w[row * m..][..m];
        let mut acc = 0.0f32;
        for (w_block, x_block) in w_row.chunks(BLOCK).zip(x.chunks(BLOCK)) {
            let mut dot = 0.0f32;
            for (weight, input) in w_block.iter().zip(x_block) {
                dot += weight * input;
            }
            acc += dot;
        }
        *value = acc;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(m: usize, n: usize) -> ShapeContext {
        MODEL.projection(m, n).expect("两个投影轴都声明了")
    }

    #[test]
    fn cost_reads_three_f32_operands_and_counts_one_multiply_add_per_element() {
        // 小形状逐项可算：w 是 m·n 个 f32、x 是 m 个、y 是 n 个；flops = 2·m·n。
        let cost = cost(&shape(8, 4)).expect("合法形状");
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
        let odd = cost(&shape(100, 3)).expect("100 不是 32 的倍数也合法");
        assert!(odd.read > 0);
        // 模板用 `len` 截断最后一块，参考实现也必须能算这个宽度。
        let sample = sample(100, 3);
        let y = reference(&sample).expect("分块余数不应报错");
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn sample_is_deterministic_and_sized_to_the_shape() {
        let first = sample(64, 4);
        assert_eq!(first, sample(64, 4));
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
        let sample = sample(64, 2);
        let y = reference(&sample).expect("合法");
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

    #[test]
    fn call_arguments_follow_the_template_signature() {
        let signature = crate::sig::Signature::parse("gemv_f32", include_str!("../ops/gemv_f32.k"))
            .expect("签名可解析");
        assert_eq!(
            signature.call_args(&sample(64, 4)).expect("实参"),
            "x, w, y, 64, 4, 32"
        );
        assert_eq!(signature.output(), Some("y"));
    }

    /// 这条算子与 `gemv_i4` 共用同一批原语，只少一个 `quant`：差异全在数据里。
    #[test]
    fn the_two_operators_differ_only_in_their_declared_quantization() {
        use crate::model;
        use prof::atom::Prim;

        let dense = crate::builtin_named("gemv_f32").expect("内置");
        let quantized = crate::builtin_named("gemv_i4").expect("内置");
        let dense_decl = crate::decl::OpDecl::parse(dense.declaration(), dense.template())
            .expect("稠密声明合法");
        let quant_decl = crate::decl::OpDecl::parse(quantized.declaration(), quantized.template())
            .expect("量化声明合法");

        let prims = |decl: &crate::decl::OpDecl| -> Vec<Prim> {
            decl.needs.iter().map(|need| need.prim).collect()
        };
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
    }
}

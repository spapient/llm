//! 层定义：[`Layer`]、[`LayerSpec`]、第一个具体层 [`Linear`]、权重编码 [`WeightFormat`]、
//! 层的性能下界 [`Bounds`] 与层栈总下界 [`Stack`]。
//!
//! **拥有**：`Layer`、`LayerSpec`、`Linear`、`WeightFormat`、`Bounds`、`Stack`。**不拥有**：算子声明与
//! 模板（`op`）、kernel 机制（`cutile`）、设备峰值（`prof`）、权重字节与解析（`weights`）。
//! **依赖**：`weights`、`tensor`、`op`、`diagnostic`。**被依赖**：`train`。
//!
//! `nn` 是 DSL 的**消费者**而不是并列层：层定义唯一的产出是一份 [`LayerSpec`]，里面的算子名
//! 必须已经存在于 `op::builtin()`。写错名字在这里就失败，而不是等到 emit 阶段才报"未知算子"。
//!
//! # 下界有三种状态，不是一个数
//!
//! [`Layer::bounds`] 返回 [`BoundsResult`]，三种情况刻意分开：
//!
//! | 返回值 | 含义 |
//! |---|---|
//! | `Ok(Some(bounds))` | 有下界，且它由算子语义推出（不是手写常数） |
//! | `Ok(None)` | 层已声明，但**没有可计价的形状**（例如注解层没给出输入宽度） |
//! | `Err(..)` | 有算子却算不出来：形状不合法（没有合法内核），或该 [`WeightFormat`] 还没有算子 |
//!
//! `Err` 不会被 [`Stack::floor`] 放大成"整张模型算不出下界"：它进无法计价清单，其余层照常相加。
//! 一个 LoRA 适配器或一张 f32 权重不该让 300 层的和归零，也不该因此得到一个偏低的数字。
//!
//! # 例子
//!
//! ```
//! use llm_nn::{Layer, Linear};
//!
//! let layer = Linear::new("blk.0.attn_q", 4096, 4096).expect("算子已声明");
//! assert_eq!(layer.output().dims(), [4096]);
//! assert_eq!(layer.spec().ops(), ["gemv_i4"]);
//!
//! // 下界按读/写分账：读侧是 int4 权重主导，写侧只有一份 f32 输出。
//! let floor = layer.bounds().expect("形状合法").expect("有算子");
//! assert!(floor.read() > floor.write());
//! ```

#![deny(unsafe_code)]

// 生成的代码用 `::nn::..` 指回本 crate，因此 crate 内部也必须能按名字解析到自己：
// 这样注解在 `nn` 自己身上（下面的 `OutputHead`）与在下游 crate 里是同一套展开。
extern crate self as nn;

#[doc(inline)]
pub use annotation::layer;

use diagnostic::{Code, Error, Result};

mod bounds;
mod format;
mod stack;

#[doc(inline)]
pub use bounds::Bounds;

#[doc(inline)]
pub use format::WeightFormat;

#[doc(inline)]
pub use stack::{Floor, Stack, Unpriced, WEIGHT_SUFFIX};

/// 形状由 `tensor` 拥有（`docs/adr/0002`），这里再导出：`Layer::output` 本来就返回它，
/// 消费者与 `#[layer]` 生成的代码都不该为此额外依赖 `tensor`。
pub use tensor::Shape;

/// 层下界的结果：[`Layer::bounds`] 的返回类型。
///
/// 单独起名是为了让 `#[layer]` 生成的方法签名不必让调用方依赖 `diagnostic`
/// （与 [`Shape`] 再导出是同一个理由）。
pub type BoundsResult = Result<Option<Bounds>>;

/// 层定义产出的声明：名字、输出形状、要用的算子名。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSpec {
    name: String,
    output: Shape,
    ops: Vec<String>,
}

impl LayerSpec {
    /// 校验并构造：`ops` 必须既存在于 `op::builtin()`，又有 `op::model()` 的语义表。
    ///
    /// 两条都要求是刻意的：只声明不实现的算子无法对拍，也就推不出它的[`Bounds`]，层一旦
    /// 引用它，下界与 `verify` 就都失去依据。把这条提前到构造期，比在 emit 或对拍阶段才发现
    /// 要便宜得多。
    ///
    /// 空 `ops` 是合法的：那表示"这一层存在，但它的编码还没有算子"（见 [`WeightFormat`]）。
    /// 这种情况不是校验失败——层仍然有自己的名字与输出形状，只是推不出下界。
    pub fn new(name: impl Into<String>, output: Shape, ops: Vec<String>) -> Result<Self> {
        let known = op::builtin();
        for wanted in &ops {
            if !known.iter().any(|declared| declared.name() == wanted) {
                return Err(Error::new(
                    Code::Missing,
                    format!(
                        "算子 `{wanted}` 没有声明（已声明：{}）",
                        known
                            .iter()
                            .map(op::Op::name)
                            .collect::<Vec<_>>()
                            .join("、")
                    ),
                )
                .with("nn::LayerSpec::new"));
            }
            if op::model(wanted).is_none() {
                return Err(Error::new(
                    Code::Missing,
                    format!("算子 `{wanted}` 没有语义表，层无法推出下界也无法对拍"),
                )
                .with("nn::LayerSpec::new"));
            }
        }
        Ok(Self {
            name: name.into(),
            output,
            ops,
        })
    }

    /// 层名（与权重条目的前缀一致）。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 输出形状。
    pub fn output(&self) -> &Shape {
        &self.output
    }

    /// 要用的算子名。空切片表示这一层的编码还没有算子，因此无法计价。
    pub fn ops(&self) -> &[String] {
        &self.ops
    }
}

/// 层契约：名字、输出形状、算子列表、它导出的声明，以及**自己的下界**。
pub trait Layer {
    /// 层名。
    fn name(&self) -> &str;

    /// 输出形状。
    fn output(&self) -> &Shape;

    /// 导出给 `cutile` 的声明。
    fn spec(&self) -> &LayerSpec;

    /// 本层的性能下界。
    ///
    /// 默认实现是 `Ok(None)`：一个层只知道名字、输出形状与算子名，**不知道规约轴有多宽**，就把
    /// 不出算子的 `(m, n)`。默认给 `None` 而不是 `Err`，是因为"没有可计价的形状"不是错误；它也
    /// 不会被静默吞掉——[`Stack::floor`] 把它列进无法计价清单。
    ///
    /// 有形状的层（[`Linear`]、带 `in = ..` 的注解层）必须覆盖它。
    fn bounds(&self) -> BoundsResult {
        Ok(None)
    }
}

/// 固定形状的输出头：形状、输入宽度与算子写在注解里，`impl Layer` 由 `#[layer]` 生成。
///
/// 与 [`Linear`] 的分工：宽度在编译期就定下来的层用注解（声明即契约），宽度要按模型配置
/// 计算出来的层用构造函数（`Linear::new` 里做校验）。两者走同一条 `LayerSpec` 校验路径，
/// 也走同一条下界路径：`in` 给了输入宽度，生成的 `bounds` 就把 `(in, out)` 交给算子语义表。
#[crate::layer(name = "lm_head", out = [4096], in = 4096, ops = ["gemv_i4"])]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OutputHead;

/// 线性层：`y = x · W`。用哪个算子由权重编码 [`WeightFormat`] 决定，而不是一个全局常量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linear {
    spec: LayerSpec,
    in_features: usize,
    out_features: usize,
    format: WeightFormat,
}

impl Linear {
    /// 默认编码（[`WeightFormat::INT4`]）对应的算子名。
    ///
    /// 构造与下界都从这一处取，层里不会出现第二个字符串；编码到算子名的映射见
    /// [`WeightFormat::operator`]。
    pub const OP: &str = "gemv_i4";

    /// 稠密 f32 权重对应的算子名。
    ///
    /// 与 [`Linear::OP`] 同属一个骨架、同一批原语，差别只在权重怎么取（解包 vs 直接点积）。
    /// 两条算子并存，是因为它们的**代价不一样**：同一份 4096 方阵，f32 的读下界是 int4 的
    /// 8 倍，拿一个当另一个的替代品就是在报一个错得很像对的数。
    pub const DENSE: &str = "gemv_f32";

    /// 由输入/输出宽度构造，按 int4 权重（[`WeightFormat::INT4`]）计价；两个宽度都必须非零。
    pub fn new(name: impl Into<String>, in_features: usize, out_features: usize) -> Result<Self> {
        Self::with_format(name, WeightFormat::INT4, in_features, out_features)
    }

    /// 按给定的权重编码构造。
    ///
    /// 编码没有对应算子时（例如 f16 / bf16 / i8），层的 `ops` 为空：它仍然是一个合法的投影
    /// 声明，只是推不出下界。把它做成 `Err` 反而会逼调用方"随便挑一个格式"来让构造成功。
    ///
    /// # Errors
    ///
    /// 任一宽度为零（[`Code::Invalid`]）；算子名由编码给出却没有通过 [`LayerSpec::new`] 的校验
    /// （只可能发生在编码与 `op` 注册表不一致时）。
    pub fn with_format(
        name: impl Into<String>,
        format: WeightFormat,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("线性层宽度必须非零（in = {in_features}，out = {out_features}）"),
            )
            .with("nn::Linear::with_format"));
        }
        let ops = format.operator().map(str::to_string).into_iter().collect();
        let spec = LayerSpec::new(name, Shape::new([out_features]), ops)?;
        Ok(Self {
            spec,
            in_features,
            out_features,
            format,
        })
    }

    /// 输入宽度。
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// 输出宽度。
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// 权重编码。它决定了本层用哪个算子，也决定了"这一层的下界是否可算"。
    pub fn format(&self) -> WeightFormat {
        self.format
    }

    /// 本层的性能下界，见 [`Layer::bounds`]。
    ///
    /// `(m, n)` 取 `(in_features, out_features)`：对本算子而言规约轴是输入宽度、输出轴是
    /// 输出宽度，与 `op` 里那条求和表达式 `nm,m,ng,n->n` 的轴顺序一致。
    pub fn bounds(&self) -> BoundsResult {
        <Self as Layer>::bounds(self)
    }
}

impl Layer for Linear {
    fn name(&self) -> &str {
        self.spec.name()
    }

    fn output(&self) -> &Shape {
        self.spec.output()
    }

    fn spec(&self) -> &LayerSpec {
        &self.spec
    }

    fn bounds(&self) -> BoundsResult {
        // 编码没有算子：不是"下界为零"，而是"这个格式还没有可计价的内核"。
        // 如实报出而不是拿另一个格式的算子凑一个数——那正是改造前 F32 权重按 int4
        // 报出 10 MiB 的原因。
        let Some(operator) = self.format.operator() else {
            // 理由由编码自己给：`Int4 { group }` 要说出"分组不同"，`Packed4` 要说出"缺标度条目"，
            // 否则读完只知道"没有算子"，不知道该改文件还是该换内核。
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "层 `{}` 的权重{}，无法推出下界",
                    self.spec.name(),
                    self.format.unpriced_reason()
                ),
            )
            .with("nn::Linear::bounds"));
        };
        Bounds::of_op(operator, self.in_features, self.out_features).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 只声明了输出、没有输入宽度：层合法，但没有可计价的形状。
    #[crate::layer(name = "no_axes", out = [8], ops = ["gemv_i4"])]
    struct NoAxes;

    #[test]
    fn an_unknown_operator_is_rejected_at_definition_time() {
        let error = LayerSpec::new(
            "bad",
            Shape::new([4]),
            vec!["gemv_i4".to_string(), "flash_attention".to_string()],
        )
        .expect_err("未声明的算子");
        assert_eq!(error.code(), Code::Missing);
        assert!(error.to_string().contains("flash_attention"), "{error}");
    }

    #[test]
    fn a_linear_layer_points_at_a_declared_operator() {
        let layer = Linear::new("blk.0.attn_q", 8, 4).expect("合法");
        assert_eq!(layer.name(), "blk.0.attn_q");
        assert_eq!(layer.output().dims(), [4]);
        assert_eq!(layer.spec().ops(), ["gemv_i4"]);
        assert_eq!(layer.out_features(), 4);
        assert_eq!(layer.format(), WeightFormat::INT4);
    }

    #[test]
    fn a_zero_width_layer_is_rejected() {
        assert_eq!(Linear::new("x", 0, 4).unwrap_err().code(), Code::Invalid);
        assert_eq!(Linear::new("x", 4, 0).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn a_linear_layer_declares_its_floor() {
        let layer = Linear::new("blk.0.attn_q", 4096, 4096).expect("合法");
        let floor = layer.bounds().expect("生产形状").expect("有算子");
        // 层自己不算代价：它必须与算子语义逐个数字一致，否则就是第二份公式。
        assert_eq!(
            floor,
            Bounds::of_op(Linear::OP, 4096, 4096).expect("同一份语义")
        );
        assert!(
            floor.read() > 8 * 1024 * 1024,
            "光 int4 权重就 8 MiB：{}",
            floor.read()
        );
    }

    #[test]
    fn a_layer_can_exist_without_a_legal_kernel_shape() {
        // 宽度非零，所以层能构造；但 8 不是量化分组的整数倍，它没有合法内核，也就没有下界。
        let layer = Linear::new("odd", 8, 4).expect("宽度非零即可构造");
        assert_eq!(
            layer.bounds().unwrap_err().code(),
            Code::Invalid,
            "形状非法仍然是错误，不是「无法计价」"
        );
    }

    #[test]
    fn a_format_without_an_operator_is_declared_but_unpriced() {
        // 同一份权重，文件里写的是 f16：层可以声明，但还没有 16 位元素的算子给它计价。
        let layer = Linear::with_format("blk.0.attn_q", WeightFormat::F16, 4096, 4096)
            .expect("宽度非零即可构造");
        assert!(layer.spec().ops().is_empty(), "没有算子可声明");
        let error = layer.bounds().unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("f16"), "{error}");
    }

    /// f32 权重现在有算子，而且报出的是它自己的代价（64 MiB 读），不是 int4 的 10.5 MB。
    #[test]
    fn a_dense_weight_is_priced_by_the_dense_operator() {
        let layer = Linear::with_format("blk.0.attn_q", WeightFormat::F32, 4096, 4096)
            .expect("宽度非零即可构造");
        assert_eq!(layer.spec().ops(), [Linear::DENSE]);
        let floor = layer.bounds().expect("合法").expect("有算子");
        assert_eq!(
            floor,
            Bounds::of_op(Linear::DENSE, 4096, 4096).expect("同一份语义")
        );
        // 读侧 = 权重（4096² 个 f32）+ 输入向量（4096 个 f32）：下界按方向分账，输入不是免费的。
        assert_eq!(
            floor.read(),
            64 * 1024 * 1024 + 4096 * 4,
            "f32 权重 + 一次输入向量的读下界"
        );
    }

    #[test]
    fn a_float_weight_is_not_priced_as_int4() {
        // 这就是改造前的隐性错误：同一个 [4096, 4096] 条目，f32 的存储本身是 64 MiB，
        // 而 int4 的读下界 ≈ 10.5 MB —— 差 6.4 倍，且旧代码会把前者报成后者。
        let int4 = Bounds::of_op(Linear::OP, 4096, 4096).expect("合法");
        let float_bytes = Shape::new([4096, 4096]).element_bytes(tensor::DType::F32);
        assert_eq!(float_bytes, 64 * 1024 * 1024);
        let ratio = float_bytes as f64 / int4.read() as f64;
        assert!((6.3..6.5).contains(&ratio), "f32 / int4 读侧 = {ratio}");
        // 两条算子的差不能被静默抹平：稠密那条报出自己的代价，量化那条报自己的。
        let dense = Linear::with_format("w", WeightFormat::F32, 4096, 4096).expect("可声明");
        let dense = dense.bounds().expect("合法").expect("有算子");
        assert_eq!(
            dense.read(),
            float_bytes as u64 + 4096 * 4,
            "f32 读下界 = 权重字节数 + 输入向量"
        );
        assert!(dense.read() > int4.read());
    }

    #[test]
    fn an_annotated_layer_is_a_checked_declaration() {
        let layer = OutputHead;
        assert_eq!(layer.name(), "lm_head");
        assert_eq!(layer.output().dims(), [4096]);
        assert_eq!(layer.spec().ops(), ["gemv_i4"]);
    }

    #[test]
    fn an_annotated_layer_prices_its_floor_from_the_same_semantics() {
        let floor = OutputHead.bounds().expect("形状合法").expect("有算子");
        assert_eq!(
            floor,
            Bounds::of_op(Linear::OP, 4096, 4096).expect("同一份语义"),
            "注解层的下界与 Linear 同源"
        );
        // 与手写的同宽度投影逐项相等：注解只是另一种声明方式，不是另一条公式。
        let linear = Linear::new(OutputHead.name(), 4096, 4096).expect("合法");
        assert_eq!(floor, linear.bounds().expect("合法").expect("有算子"));
    }

    #[test]
    fn an_annotated_layer_without_an_input_width_has_no_floor() {
        // 没有 `in`：层合法，但规约轴宽度未知，算子语义表无法被调用。
        assert_eq!(NoAxes.bounds().expect("不是错误"), None);
    }

    #[test]
    fn the_annotated_implementation_is_shared_by_every_instance() {
        let first = OutputHead.spec();
        let second = OutputHead.spec();
        assert!(core::ptr::eq(first, second), "规格只构造一次（OnceLock）");
    }

    #[test]
    fn a_spec_works_as_a_trait_object() {
        let layer = Linear::new("blk.0.attn_k", 32, 6).expect("合法");
        let dynamic: &dyn Layer = &layer;
        assert_eq!(dynamic.spec().name(), "blk.0.attn_k");
        assert!(dynamic.bounds().expect("合法").is_some());
    }
}

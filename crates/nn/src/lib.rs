//! 层定义：[`Layer`]、[`LayerSpec`]、第一个具体层 [`Linear`] 与层的性能下界 [`Bounds`]。
//!
//! **拥有**：`Layer`、`LayerSpec`、`Linear`、`Bounds`。**不拥有**：算子声明与模板（`op`）、kernel
//! 机制（`cutile`）、设备峰值（`prof`）、参数存储（`weights`）。**依赖**：`tensor`、`op`、
//! `diagnostic`。**被依赖**：`train`。
//!
//! `nn` 是 DSL 的**消费者**而不是并列层：层定义唯一的产出是一份 [`LayerSpec`]，里面的算子名
//! 必须已经存在于 `op::builtin()`。写错名字在这里就失败，而不是等到 emit 阶段才报"未知算子"。
//!
//! # 例子
//!
//! ```
//! use llm_nn::{Layer, Linear};
//!
//! let layer = Linear::new("blk.0.attn_q", 4096, 4096).expect("算子已声明");
//! assert_eq!(layer.output().dims(), [4096]);
//! assert_eq!(layer.spec().ops(), ["gemv_i4"]);
//! ```

#![deny(unsafe_code)]

// 生成的代码用 `::nn::..` 指回本 crate，因此 crate 内部也必须能按名字解析到自己：
// 这样注解在 `nn` 自己身上（下面的 `OutputHead`）与在下游 crate 里是同一套展开。
extern crate self as nn;

#[doc(inline)]
pub use annotation::layer;

use diagnostic::{Code, Error, Result};

mod bounds;

#[doc(inline)]
pub use bounds::Bounds;

/// 形状由 `tensor` 拥有（`docs/adr/0002`），这里再导出：`Layer::output` 本来就返回它，
/// 消费者与 `#[layer]` 生成的代码都不该为此额外依赖 `tensor`。
pub use tensor::Shape;

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

    /// 要用的算子名。
    pub fn ops(&self) -> &[String] {
        &self.ops
    }
}

/// 层契约：名字、输出形状、算子列表与它导出的声明。
pub trait Layer {
    /// 层名。
    fn name(&self) -> &str;

    /// 输出形状。
    fn output(&self) -> &Shape;

    /// 导出给 `cutile` 的声明。
    fn spec(&self) -> &LayerSpec;
}

/// 固定形状的输出头：形状与算子写在注解里，`impl Layer` 由 `#[layer]` 生成。
///
/// 与 [`Linear`] 的分工：宽度在编译期就定下来的层用注解（声明即契约），宽度要按模型配置
/// 计算出来的层用构造函数（`Linear::new` 里做校验）。两者走同一条 `LayerSpec` 校验路径。
#[crate::layer(name = "lm_head", out = [4096], ops = ["gemv_i4"])]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OutputHead;

/// 线性层：`y = x · W`，在 int4 权重上走 `gemv_i4`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linear {
    spec: LayerSpec,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// 本层使用的算子名：构造与下界都从这一处取，层里不会出现第二个字符串。
    pub const OP: &str = "gemv_i4";

    /// 由输入/输出宽度构造；两个宽度都必须非零。
    pub fn new(name: impl Into<String>, in_features: usize, out_features: usize) -> Result<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("线性层宽度必须非零（in = {in_features}，out = {out_features}）"),
            )
            .with("nn::Linear::new"));
        }
        let spec = LayerSpec::new(name, Shape::new([out_features]), vec![Self::OP.to_string()])?;
        Ok(Self {
            spec,
            in_features,
            out_features,
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

    /// 本层的性能下界：由算子语义在自己的形状上推出，不是手写常数。
    ///
    /// `(m, n)` 取 `(in_features, out_features)`：对本算子而言规约轴是输入宽度、输出轴是
    /// 输出宽度，与 `op` 里那条求和表达式 `nm,m,ng,n->n` 的轴顺序一致。
    ///
    /// # Errors
    ///
    /// 形状对该算子不合法时（`in_features` 不是量化分组的整数倍）返回 [`Code::Invalid`]：
    /// 层本身可以构造，但它没有合法内核，因此"至少要搬多少"无从谈起。
    pub fn bounds(&self) -> Result<Bounds> {
        Bounds::of_op(Self::OP, self.in_features, self.out_features)
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn a_zero_width_layer_is_rejected() {
        assert_eq!(Linear::new("x", 0, 4).unwrap_err().code(), Code::Invalid);
        assert_eq!(Linear::new("x", 4, 0).unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn a_linear_layer_declares_its_floor() {
        let layer = Linear::new("blk.0.attn_q", 4096, 4096).expect("合法");
        let floor = layer.bounds().expect("生产形状");
        // 层自己不算代价：它必须与算子语义逐个数字一致，否则就是第二份公式。
        assert_eq!(
            floor,
            Bounds::of_op(Linear::OP, 4096, 4096).expect("同一份语义")
        );
        assert!(
            floor.bytes() > 8 * 1024 * 1024,
            "光 int4 权重就 8 MiB：{}",
            floor.bytes()
        );
    }

    #[test]
    fn a_layer_can_exist_without_a_legal_kernel_shape() {
        // 宽度非零，所以层能构造；但 8 不是量化分组的整数倍，它没有合法内核，也就没有下界。
        let layer = Linear::new("odd", 8, 4).expect("宽度非零即可构造");
        assert_eq!(layer.bounds().unwrap_err().code(), Code::Invalid);
    }

    #[test]
    fn an_annotated_layer_is_a_checked_declaration() {
        let layer = OutputHead;
        assert_eq!(layer.name(), "lm_head");
        assert_eq!(layer.output().dims(), [4096]);
        assert_eq!(layer.spec().ops(), ["gemv_i4"]);
    }

    #[test]
    fn the_annotated_implementation_is_shared_by_every_instance() {
        let first = OutputHead.spec();
        let second = OutputHead.spec();
        assert!(core::ptr::eq(first, second), "规格只构造一次（OnceLock）");
    }

    #[test]
    fn a_spec_works_as_a_trait_object() {
        let layer = Linear::new("blk.0.attn_k", 6, 6).expect("合法");
        let dynamic: &dyn Layer = &layer;
        assert_eq!(dynamic.spec().name(), "blk.0.attn_k");
    }
}

//! 原先内联在 `src/pointwise.rs` 的测试。
//!
//! 用 `#[path]` 挂回原模块，模块树位置不变，故 `super::`/`crate::` 路径照旧。

use super::*;

use crate::{Bounds, Contract, ParamRole};
use diagnostic::Code;

fn assert_layer<L: Contract>(layer: &L, name: &str, op: &str, len: usize) {
    assert_eq!(layer.name(), name);
    assert_eq!(layer.output().dims(), [len]);
    assert_eq!(layer.spca().ops(), [op]);
    let floor = layer.bounds().expect("单轴形状合法").expect("有算子");
    // 单轴算子不写第二份公式：下界必须与直接问语义表逐项相等。
    let same = Bounds::of_ctx(op, len, 1).expect("同一份语义");
    assert_eq!(floor, same, "{op} 的下界来自算子语义表");
    assert!(floor.read() > 0 && floor.write() > 0, "{op} 有读有写");
}

#[test]
fn each_pointwise_layer_binds_its_declared_operator() {
    assert_layer(
        &RmsNorm::new("blk.0.norm", 128).expect("合法"),
        "blk.0.norm",
        RmsNorm::OP,
        128,
    );
    assert_layer(
        &Softmax::new("attn.softmax", 64).expect("合法"),
        "attn.softmax",
        Softmax::OP,
        64,
    );
    assert_layer(
        &Gelu::new("mlp.gelu", 128).expect("合法"),
        "mlp.gelu",
        Gelu::OP,
        128,
    );
    assert_layer(
        &Silu::new("mlp.silu", 128).expect("合法"),
        "mlp.silu",
        Silu::OP,
        128,
    );
    assert_layer(
        &Rope::new("attn.rope", 128).expect("合法"),
        "attn.rope",
        Rope::OP,
        128,
    );
    assert_layer(
        &Scale::new("proj.scale", 128).expect("合法"),
        "proj.scale",
        Scale::OP,
        128,
    );
    assert_layer(
        &Add::new("res.add", 128).expect("合法"),
        "res.add",
        Add::OP,
        128,
    );
    assert_layer(
        &Mul::new("gate.mul", 128).expect("合法"),
        "gate.mul",
        Mul::OP,
        128,
    );
    assert_layer(
        &ResidualAdd::new("residual.add", 128).expect("合法"),
        "residual.add",
        ResidualAdd::OP,
        128,
    );
}

#[test]
fn normalization_declares_the_norm_role_and_other_axis_layers_the_trainable_role() {
    let norm = RmsNorm::new("blk.0.norm", 128).expect("合法");
    assert_eq!(norm.spca().role(), ParamRole::Norm);
    assert!(norm.spca().role().wd_exempt());

    let activation = Gelu::new("mlp.gelu", 128).expect("合法");
    assert_eq!(activation.spca().role(), ParamRole::Trainable);
}

#[test]
fn a_zero_length_pointwise_layer_is_rejected() {
    assert_eq!(Gelu::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Silu::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(RmsNorm::new("x", 0).unwrap_err().code(), Code::Invalid);
}

#[test]
fn pointwise_layers_work_as_trait_objects() {
    let layer: &dyn Contract = &Silu::new("mlp.silu", 64).expect("合法");
    assert_eq!(layer.output().dims(), [64]);
    assert!(layer.bounds().expect("合法").is_some());
}

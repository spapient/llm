//! 单算子层的用例：逐元素/归一化/激活/查表/矩阵乘共用一条契约——
//! 层不写第二份流量常数，下界一律由算子语义表推出。

use super::*;

use crate::{Bounds, Contract};
use diagnostic::Code;

#[test]
fn a_reduction_emits_a_scalar_shape() {
    let sum = ReduceSum::new("loss.sum", 128).expect("合法");
    assert_eq!(sum.output().dims(), [], "归约输出是标量");
    assert_eq!(sum.spca().ops(), [ReduceSum::OP]);
    let floor = sum.bounds().expect("合法").expect("有算子");
    assert_eq!(
        floor,
        Bounds::of_ctx(ReduceSum::OP, 128, 1).expect("同一份语义")
    );
    // 读侧覆盖整个输入，写侧只有一个标量。
    assert!(floor.read() > floor.write());
}

#[test]
fn a_max_reduction_binds_its_operator() {
    let max = ReduceMax::new("pool.max", 64).expect("合法");
    assert_eq!(max.output().dims(), []);
    assert_eq!(max.spca().ops(), [ReduceMax::OP]);
    assert!(max.bounds().expect("合法").is_some());
}

#[test]
fn a_zero_length_reduction_is_rejected() {
    assert_eq!(ReduceSum::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(ReduceMax::new("x", 0).unwrap_err().code(), Code::Invalid);
}

#[test]
fn an_embedding_declares_its_table_and_output_shape() {
    let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
    assert_eq!(layer.name(), "tok_emb");
    assert_eq!(layer.vocab(), 32_000);
    assert_eq!(layer.dim(), 128);
    assert_eq!(layer.tokens(), 8);
    assert_eq!(layer.output().dims(), [8, 128]);
    assert_eq!(layer.spca().ops(), [Embedding::OP]);
    assert_eq!(layer.spca().role(), ParamRole::Embedding);
    assert!(layer.spca().role().wd_exempt());
}

#[test]
fn an_embedding_floor_follows_the_gather_semantics() {
    let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
    let floor = layer.bounds().expect("合法").expect("有算子");
    assert_eq!(
        floor,
        Bounds::of_ctx(Embedding::OP, 8, 1).expect("同一份语义")
    );
}

#[test]
fn a_zero_dimension_embedding_is_rejected() {
    assert_eq!(
        Embedding::new("x", 0, 128, 8).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Embedding::new("x", 32, 0, 8).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Embedding::new("x", 32, 128, 0).unwrap_err().code(),
        Code::Invalid
    );
}

#[test]
fn a_matmul_layer_binds_its_operator_and_shape() {
    let layer = MatMul::new("blk.0.matmul", 16, 12).expect("合法");
    assert_eq!(layer.name(), "blk.0.matmul");
    assert_eq!(layer.output().dims(), [16, 12]);
    assert_eq!(layer.spca().ops(), [MatMul::OP]);
    assert_eq!(layer.rows(), 16);
    assert_eq!(layer.cols(), 12);
}

#[test]
fn the_matmul_floor_follows_the_gemm_semantics() {
    let layer = MatMul::new("blk.0.matmul", 16, 12).expect("合法");
    let floor = layer.bounds().expect("合法").expect("有算子");
    // 层不写第二份公式：下界必须与直接问语义表逐项相等。
    let same = Bounds::of_ctx(MatMul::OP, 16, 12).expect("同一份语义");
    assert_eq!(floor, same, "矩阵乘下界来自算子语义表");
    assert!(floor.read() > 0 && floor.write() > 0 && floor.flops() > 0.0);
}

#[test]
fn a_zero_dimension_matmul_is_rejected() {
    assert_eq!(MatMul::new("x", 0, 12).unwrap_err().code(), Code::Invalid);
    assert_eq!(MatMul::new("x", 16, 0).unwrap_err().code(), Code::Invalid);
}

#[test]
fn a_matmul_works_as_a_trait_object() {
    let layer: &dyn Contract = &MatMul::new("proj", 16, 12).expect("合法");
    assert_eq!(layer.output().dims(), [16, 12]);
    assert!(layer.bounds().expect("合法").is_some());
}

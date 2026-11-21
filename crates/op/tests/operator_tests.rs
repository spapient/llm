// [`Operator`](crate::operator::Operator) 投影与注册表的一致性（ADR 0050 §2）。
//
// 过渡期 `MODEL` 仍由 `#[model]` 注解生成；本测试逐字段比对 `model_of::<T>()` 与注册表里的
// 那一份，任何一侧漂移都在这里失败，而不是等到 `verify` 用错语义表。

use crate::model::Model;
use crate::operator::{Operator, model_of};

/// 投影与注册表逐字段一致：常量字段直接比，行为字段在同一个形状上比输出。
fn assert_projection_matches_registry<T: Operator>() {
    let projected = model_of::<T>();
    let registered = crate::model(T::NAME).expect("算子必须已注册");
    assert_eq!(projected.group, registered.group, "{}: group", T::NAME);
    assert_eq!(projected.shape, registered.shape, "{}: shape", T::NAME);
    assert_eq!(projected.dtype, registered.dtype, "{}: dtype", T::NAME);
    assert_eq!(projected.axes, registered.axes, "{}: axes", T::NAME);
    assert_behavior_matches(projected, registered, T::NAME);
}

/// 代价、样本与参考：同一形状上下文的输出逐位相等。
fn assert_behavior_matches(projected: Model, registered: Model, name: &str) {
    let shape = (projected.shape_ctx)(64, 8).expect("形状上下文可构造");
    let sample = (projected.sample)(64, 8);
    assert_eq!(sample, (registered.sample)(64, 8), "{name}: sample");
    assert_eq!(
        (projected.cost)(&shape).expect("投影代价可计算"),
        (registered.cost)(&shape).expect("注册表代价可计算"),
        "{name}: cost"
    );
    assert_eq!(
        (projected.reference)(&sample).expect("投影参考可计算"),
        (registered.reference)(&sample).expect("注册表参考可计算"),
        "{name}: reference"
    );
}

#[test]
fn reduce_projection_matches_the_registered_model() {
    assert_projection_matches_registry::<crate::reduce_sum_f32::Sum>();
    assert_projection_matches_registry::<crate::reduce_max_f32::Max>();
}

#[test]
fn unary_projection_matches_the_registered_model() {
    assert_projection_matches_registry::<crate::unary::exp_f32::Exp>();
    assert_projection_matches_registry::<crate::unary::log_f32::Log>();
    assert_projection_matches_registry::<crate::unary::gelu_f32::Gelu>();
    assert_projection_matches_registry::<crate::unary::silu_f32::Silu>();
    assert_projection_matches_registry::<crate::unary::scale_f32::Scale>();
}

#[test]
fn binary_projection_matches_the_registered_model() {
    assert_projection_matches_registry::<crate::elementwise::add_f32::Add>();
    assert_projection_matches_registry::<crate::elementwise::sub_f32::Sub>();
    assert_projection_matches_registry::<crate::elementwise::mul_f32::Mul>();
    assert_projection_matches_registry::<crate::elementwise::div_f32::Div>();
    assert_projection_matches_registry::<crate::elementwise::max_f32::Max>();
    assert_projection_matches_registry::<crate::elementwise::min_f32::Min>();
    assert_projection_matches_registry::<crate::elementwise::residual_add_f32::Residual>();
}

#[test]
fn dequant_projection_matches_the_registered_model() {
    assert_projection_matches_registry::<crate::dequant_i2::DequantI2>();
    assert_projection_matches_registry::<crate::dequant_i4::DequantI4>();
    assert_projection_matches_registry::<crate::dequant_i8::DequantI8>();
}

#[test]
fn gemv_projection_matches_the_registered_model() {
    assert_projection_matches_registry::<crate::gemv::i2::I2>();
    assert_projection_matches_registry::<crate::gemv::i4::I4>();
    assert_projection_matches_registry::<crate::gemv::i8::I8>();
    assert_projection_matches_registry::<crate::gemv::f32::F32>();
}

#[test]
fn gemm_projection_matches_the_registered_model() {
    assert_projection_matches_registry::<crate::gemm::i2::GemmI2>();
    assert_projection_matches_registry::<crate::gemm::i4::GemmI4>();
    assert_projection_matches_registry::<crate::gemm::i8::GemmI8>();
    assert_projection_matches_registry::<crate::gemm::f32::Dense>();
}

#[test]
fn normalization_and_shape_projection_matches_the_registered_model() {
    assert_projection_matches_registry::<crate::softmax::f32::softmax::Softmax>();
    assert_projection_matches_registry::<crate::softmax::f32::rmsnorm::RmsNorm>();
    assert_projection_matches_registry::<crate::softmax::f32::rope::Rope>();
    assert_projection_matches_registry::<crate::softmax::f32::embed::Embed>();
    assert_projection_matches_registry::<crate::scan::f32::Scan>();
}

#[test]
fn attention_projection_matches_the_registered_model() {
    assert_projection_matches_registry::<crate::attention::Attention>();
    assert_projection_matches_registry::<crate::attention::FlashAttention>();
    assert_projection_matches_registry::<crate::attention::decode_attention::DecodeAttention>();
    assert_projection_matches_registry::<crate::attention::decode_attention::DecodeFlashAttention>(
    );
}

#[test]
fn matmul_projection_matches_the_registered_model() {
    assert_projection_matches_registry::<crate::matmul::Matmul>();
    assert_projection_matches_registry::<crate::matmul_i8::MatmulI8Op>();
}

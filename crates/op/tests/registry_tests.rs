//! 注册表 ↔ 内核的跨层核对：只有同时看得见声明层与内核的地方才能写这种测试。
//!
//! 内核 crate 不认识 [`REGISTRY`]（也不该认识），所以「一条算子登记的那份 `MODEL`
//! 与它实际消费的形状/类型一致吗」这一问留在这里回答。

use crate::decl::OpDecl;
use crate::model;
use llm_op_kernel::gemv;
use llm_op_kernel::gemv::f32::F32;
use llm_op_spec::model::DType;
use llm_op_spec::operator::model_of;
use prof::atom::Prim;

/// 这条算子与 `gemv_i4` 共用同一批原语，只少一个 `quant`：差异全在数据里。
#[test]
fn the_two_operators_differ_only_in_their_declared_quantization() {
    let dense = crate::builtin_named("gemv_f32").expect("内置");
    let quantized = crate::builtin_named("gemv_i4").expect("内置");
    let dense_decl = OpDecl::parse(dense.declaration(), dense.template()).expect("稠密声明合法");
    let quant_decl =
        OpDecl::parse(quantized.declaration(), quantized.template()).expect("量化声明合法");

    let prims = |decl: &OpDecl| -> Vec<Prim> { decl.needs.iter().map(|need| need.prim).collect() };
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
    assert_eq!(model_of::<F32>().dtype, DType::F32, "样本的输出缓冲是 f32");
    assert_eq!(FORMAT_GRAIN, 1, "格式自己的声明粒度也是 1");
}

/// 与 `model_of::<F32>()` 同源：证明 `Model::group` 确实来自 [`gemv::Format::GRAIN`]，
/// 而不是注解另写了一个数。
const FORMAT_GRAIN: usize = <F32 as gemv::Format>::GRAIN;

//! 算子注册表：名字 → 语义表（代价 / 样本 / 标量参考）+ 可微性。
//!
//! 与 [`prof::catalogue`] 分工：那里回答"有哪些算子、各自用哪份模板"，本表回答"谁来算"。
//! 两张表都按名字排同一顺序，错位在编译期失败；模板正文不在这里重复——要它的人问
//! `prof::catalogue::template(name)` 取主机参考实现；设备正文由发射层按 vendor 选择。

use prof::catalogue::Registration;

/// 算子清单：`名字: 内核类型`，一行一个，**唯一一次**声明内核类型。
///
/// 名字必须与 `Operator::NAME` 逐字节相同（下面有编译期断言）；模板 stem 不在这里写——
/// 那是 [`prof::for_each_template!`] 的事，两张表按名字同序，错位在编译期失败。
///
/// 导出是因为上层还要从同一张清单展开类型化 API（`train::OperatorSetOps`）：一张清单，
/// 注册表、类型化入口、样本三者不会各自漂移。
#[macro_export]
macro_rules! for_each_operator {
    ($m:ident) => {
        $m! {
            abs_f16: $crate::kernel::abs_f16::AbsF16,
            abs_f32: $crate::kernel::abs_f32::AbsF32,
            adaptive_pool_f32: $crate::conv::adaptive_pool::AdaptivePoolModel,
            add_f32: $crate::kernel::add_f32::AddF32,
            attention_f32: $crate::conv::attention::AttentionF32,
            avg_pool2d_f32: $crate::conv::avg_pool2d::AvgPool2dModel,
            batch_norm_f32: $crate::conv::batch_norm::BatchNormModel,
            ceil_f32: $crate::kernel::ceil_f32::CeilF32,
            clamp_f32: $crate::kernel::clamp_f32::ClampF32,
            conv2d_f32: $crate::conv::conv2d::Conv2dModel,
            cos_f32: $crate::kernel::cos_f32::CosF32,
            decode_attention_f32: $crate::conv::decode_attention::DecodeAttentionF32,
            decode_flash_attention_f32: $crate::conv::decode_attention::DecodeFlashAttentionF32,
            dequant_i2: $crate::kernel::dequant_i2::DequantI2F32,
            dequant_i4: $crate::kernel::dequant_i4::DequantI4F32,
            dequant_i8: $crate::kernel::dequant_i8::DequantI8F32,
            div_f32: $crate::kernel::div_f32::DivF32,
            embed_f32: $crate::kernel::embed_f32::EmbedF32,
            exp2_f32: $crate::kernel::exp2_f32::Exp2F32,
            exp_f32: $crate::kernel::exp_f32::ExpF32,
            flash_attention_f32: $crate::conv::attention::FlashAttentionF32,
            floor_f32: $crate::kernel::floor_f32::FloorF32,
            gelu_f32: $crate::kernel::gelu_f32::GeluF32,
            gemm_f32: $crate::kernel::gemm_f32::GemmF32,
            gemm_i2: $crate::kernel::gemm_i2::GemmI2Model,
            gemm_i4: $crate::kernel::gemm_i4::GemmI4Model,
            gemm_i8: $crate::kernel::gemm_i8::GemmI8Model,
            gemv_f32: $crate::kernel::gemv_f32::F32,
            gemv_i2: $crate::kernel::gemv_i2::I2,
            gemv_i4: $crate::kernel::gemv_i4::I4,
            gemv_i8: $crate::kernel::gemv_i8::I8,
            group_norm_f32: $crate::conv::group_norm::GroupNormModel,
            hard_sigmoid_f32: $crate::kernel::hard_sigmoid_f32::HardSigmoidF32,
            instance_norm_f32: $crate::conv::instance_norm::InstanceNormModel,
            interpolate_f32: $crate::conv::interpolate::InterpolateModel,
            latent_attention_f32: $crate::conv::attention::latent::LatentAttentionF32,
            layer_norm_f32: $crate::conv::layer_norm::LayerNormModel,
            leaky_relu_f32: $crate::kernel::leaky_relu_f32::LeakyReluF32,
            log1p_f32: $crate::kernel::log1p_f32::Log1pF32,
            log_f32: $crate::kernel::log_f32::LogF32,
            matmul_f32: $crate::kernel::matmul::MatmulF32,
            matmul_i8: $crate::kernel::matmul_i8::MatmulI8,
            max_f32: $crate::kernel::max_f32::MaxF32,
            max_pool1d_f32: $crate::conv::max_pool1d::MaxPool1dModel,
            max_pool2d_f32: $crate::conv::max_pool2d::MaxPool2dModel,
            min_f32: $crate::kernel::min_f32::MinF32,
            mul_f32: $crate::kernel::mul_f32::MulF32,
            neg_f32: $crate::kernel::neg_f32::NegF32,
            recip_f32: $crate::kernel::recip_f32::RecipF32,
            reduce_max_f32: $crate::kernel::reduce_max_f32::ReduceMaxF32,
            reduce_prod_f32: $crate::kernel::reduce_prod_f32::ReduceProdF32,
            reduce_sum_f32: $crate::kernel::reduce_sum_f32::ReduceSumF32,
            relu_f32: $crate::kernel::relu_f32::ReluF32,
            remainder_f32: $crate::kernel::remainder_f32::RemainderF32,
            residual_add_f32: $crate::kernel::residual_add_f32::ResidualAddF32,
            rmsnorm_f32: $crate::kernel::rmsnorm_f32::RmsNormF32,
            rope_f32: $crate::kernel::rope_f32::RopeF32,
            round_f32: $crate::kernel::round_f32::RoundF32,
            rsqrt_f32: $crate::kernel::rsqrt_f32::RsqrtF32,
            scale_f32: $crate::kernel::scale_f32::ScaleF32,
            scan_f32: $crate::conv::scan_f32::ScanF32,
            sigmoid_f32: $crate::kernel::sigmoid_f32::SigmoidF32,
            sign_f32: $crate::kernel::sign_f32::SignF32,
            silu_f32: $crate::kernel::silu_f32::SiluF32,
            sin_f32: $crate::kernel::sin_f32::SinF32,
            softmax_f32: $crate::kernel::softmax_f32::SoftmaxF32,
            sqrt_f32: $crate::kernel::sqrt_f32::SqrtF32,
            square_f32: $crate::kernel::square_f32::SquareF32,
            sub_f32: $crate::kernel::sub_f32::SubF32,
            tanh_f32: $crate::kernel::tanh_f32::TanhF32,
            trunc_f32: $crate::kernel::trunc_f32::TruncF32,
        }
    };
}

macro_rules! registry_impl {
    ($($name:ident: $ty:ty),* $(,)?) => {
        /// 编译期内嵌的算子注册表，顺序与 [`prof::catalogue::TEMPLATES`] 一致。
        pub const REGISTRY: [Registration; <[()]>::len(&[$(registry_impl!(@one $ty)),*])] = [
            $(Registration {
                name: <$ty as prof::operator::Operator>::NAME,
                model: prof::operator::model_of::<$ty>(),
                differentiable: <$ty as prof::operator::Operator>::DIFFERENTIABLE,
            }),*
        ];

        /// 两道编译期断言：清单名字与 `Operator::NAME` 一致，清单顺序与算子目录一致。
        const _: () = {
            let mut idx = 0;
            $(
                assert!(
                    prof::same_bytes(stringify!($name), <$ty as prof::operator::Operator>::NAME),
                    "算子清单的名字与 Operator::NAME 不一致",
                );
                assert!(
                    prof::same_bytes(stringify!($name), prof::catalogue::TEMPLATES[idx].0),
                    "算子注册表与算子目录的顺序或名字不一致",
                );
                idx += 1;
            )*
            let _ = idx;
        };
    };
    (@one $ty:ty) => { () };
}

for_each_operator!(registry_impl);

/// 名字是否在注册表里；const 上下文可用。
pub const fn contains(name: &str) -> bool {
    let mut idx = 0;
    while idx < REGISTRY.len() {
        if prof::same_bytes(name, REGISTRY[idx].name) {
            return true;
        }
        idx += 1;
    }
    false
}

/// 算子语义表：声明层有的算子，这里必须有它的代价、样本与标量参考。
///
/// 查表走二分而不是线性扫描：`REGISTRY` 与算子目录同序，且清单已按名字字典序排好（上面的
/// `const` 断言把这一不变量钉死），所以"有序"不是约定而是编译期事实。名字列表只有几十条，
/// 但装配一条几百层的模型会按层查它若干次，对数查找省下的是热路径上重复的比较。
pub fn model(name: &str) -> Option<prof::model::Model> {
    REGISTRY
        .binary_search_by(|entry| entry.name.cmp(name))
        .ok()
        .map(|idx| REGISTRY[idx].model)
}

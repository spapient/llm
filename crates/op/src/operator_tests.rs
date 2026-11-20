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
    let shape = (projected.shape_ctx)(64, 1).expect("形状上下文可构造");
    let sample = (projected.sample)(64, 1);
    assert_eq!(sample, (registered.sample)(64, 1), "{name}: sample");
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

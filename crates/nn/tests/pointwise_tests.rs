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

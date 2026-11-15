use diagnostic::Code;

use crate::{MemoryPolicy, Ordering};

use super::*;

#[test]
fn ids_are_allocated_monotonically() {
    let mut tape = Tape::new();
    let a = tape.leaf();
    let b = tape.parameter();
    assert_eq!((a.id(), b.id()), (0, 1));
    assert_eq!(tape.vars(), 2);
    assert!(tape.is_empty(), "叶子不是节点");
    assert_eq!(tape.requirement(a), Some(Requirement::None));
    assert_eq!(tape.requirement(b), Some(Requirement::Grad));
}

#[test]
fn clearing_resets_the_numbering() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    tape.record("relu", &[x]).expect("记录");
    tape.clear();
    assert!(tape.is_empty());
    assert_eq!(tape.vars(), 0);
    let fresh = tape.parameter();
    assert_eq!(fresh.id(), 0, "编号从 0 重新开始");
    assert_eq!(tape.requirement(fresh), Some(Requirement::Grad));
}

#[test]
fn backward_order_is_the_reverse_of_forward() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let w = tape.parameter();
    let y = tape.record("gemv_i4", &[x, w]).expect("记录");
    let loss = tape.record("reduce_sum", &[y]).expect("记录");
    assert_eq!(tape.len(), 2);
    assert_eq!(tape.backward_order(), [loss, y]);
    assert_eq!(tape.nodes()[1].inputs(), [y]);
    assert_eq!(tape.nodes()[0].op(), "gemv_i4");
    assert_eq!(tape.depth(y), Some(1));
    assert_eq!(tape.depth(loss), Some(2));
}

#[test]
fn referencing_a_missing_variable_is_rejected() {
    let mut tape = Tape::new();
    let x = tape.leaf();
    let ghost = Var::at(42);
    let error = tape.record("add", &[x, ghost]).expect_err("不存在的输入");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("42"), "{error}");
}

#[test]
fn the_requirement_is_inferred_from_the_inputs() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let constant = tape.leaf();
    let from_constant = tape.record("scale", &[constant]).expect("记录");
    let from_parameter = tape.record("relu", &[x]).expect("记录");
    assert_eq!(tape.requirement(from_constant), Some(Requirement::None));
    assert_eq!(
        tape.requirement(from_parameter),
        Some(Requirement::Backward)
    );
    assert_eq!(tape.depth(from_constant), Some(0), "不进图的节点深度是 0");
}

#[test]
fn detaching_cuts_the_path_to_the_inputs() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let frozen = tape
        .record_with("stop_gradient", &[x], Requirement::None)
        .expect("记录");
    let downstream = tape.record("relu", &[frozen]).expect("记录");
    assert_eq!(tape.requirement(downstream), Some(Requirement::None));
    assert_eq!(
        tape.backward(downstream).expect_err("不在图上").code(),
        Code::Invalid
    );
    assert!(tape.backward(x).expect("参数的梯度是 1").is_empty());
}

#[test]
fn a_requiring_node_without_requiring_inputs_is_rejected() {
    let mut tape = Tape::new();
    let constant = tape.leaf();
    let error = tape
        .record_with("relu", &[constant], Requirement::Grad)
        .expect_err("没有输入要梯度");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("Grad"), "{error}");
    assert!(tape.is_empty(), "失败的记录不留节点");
}

#[test]
fn an_empty_operator_name_is_rejected() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    assert_eq!(
        tape.record("  ", &[x]).expect_err("空算子名").code(),
        Code::Invalid
    );
    assert!(tape.is_empty());
}

#[test]
fn plan_config_follows_the_configured_ordering() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let y = tape.record("mul", &[x]).expect("记录");
    let z = tape.record("add", &[y]).expect("记录");
    let config = Config::new(Ordering::Insertion, MemoryPolicy::Auto);
    let mut schedule = Schedule::default();
    tape.plan_config(&mut schedule, z, &config).expect("规划");
    assert_eq!(schedule.ordering(), Ordering::Insertion);
}

#[test]
fn roles_separate_parameters_constants_and_activations() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let c = tape.leaf();
    let y = tape.record("relu", &[x]).expect("记录");
    assert_eq!(tape.role(x), Some(Role::Parameter));
    assert_eq!(tape.role(c), Some(Role::Constant));
    assert_eq!(tape.role(y), Some(Role::Activation));
    assert!(tape.validate().is_ok());
}

#[test]
fn retaining_an_unused_leaf_makes_it_an_endpoint() {
    let mut tape = Tape::new();
    let x = tape.leaf();
    tape.retain(x).expect("未被引用");
    assert_eq!(tape.requirement(x), Some(Requirement::Grad));
    tape.retain(x).expect("幂等");
    let y = tape.record("relu", &[x]).expect("记录");
    assert_eq!(
        tape.backward(y).expect("在图上").endpoints(),
        [x],
        "升级后的叶子是梯度终点"
    );
}

#[test]
fn retaining_is_rejected_once_the_leaf_is_used_or_computed() {
    let mut tape = Tape::new();
    let x = tape.leaf();
    let y = tape.record("relu", &[x]).expect("记录");
    assert_eq!(tape.retain(x).expect_err("已被引用").code(), Code::Invalid);
    assert_eq!(tape.retain(y).expect_err("算子输出").code(), Code::Invalid);
    assert_eq!(
        tape.retain(Var::at(7)).expect_err("不存在").code(),
        Code::Invalid
    );
}

#[test]
fn a_parameter_recorded_as_loss_has_an_empty_schedule() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let schedule = tape.backward(x).expect("参数在图上");
    assert!(schedule.is_empty());
    assert_eq!(schedule.endpoints(), [x]);
    assert_eq!(schedule.loss(), x);
}

#[test]
fn backward_rejects_variables_that_are_not_on_the_graph() {
    let mut tape = Tape::new();
    let constant = tape.leaf();
    let computed = tape.record("scale", &[constant]).expect("记录");
    let ghost = Var::at(9);
    assert_eq!(
        tape.backward(constant).expect_err("不进图").code(),
        Code::Invalid
    );
    assert_eq!(
        tape.backward(computed).expect_err("不进图").code(),
        Code::Invalid
    );
    assert_eq!(
        tape.backward(ghost).expect_err("不存在").code(),
        Code::Invalid
    );
}

#[test]
fn a_retained_intermediate_is_an_endpoint_with_a_step() {
    let mut tape = Tape::new();
    let x = tape.parameter();
    let hidden = tape
        .record_with("relu", &[x], Requirement::Grad)
        .expect("记录");
    let loss = tape.record("reduce_sum", &[hidden]).expect("记录");
    let schedule = tape.backward(loss).expect("在图上");
    assert_eq!(schedule.endpoints(), [x, hidden]);
    assert_eq!(schedule.len(), 2, "终点也要执行它自己的反向步骤");
}

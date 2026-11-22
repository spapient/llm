use super::*;
use crate::{Parallel, Topology};
use diagnostic::Code;

fn topology() -> Topology {
    Topology::new(2, 1, 2).expect("tp 2 · dp 2")
}

#[test]
fn a_group_must_be_sorted_and_unique() {
    let group = Group::new([0, 2, 4]).expect("升序");
    assert_eq!(group.ranks(), [0, 2, 4]);
    assert_eq!(group.size(), 3);
    assert_eq!(Group::new([]).unwrap_err().code(), Code::Invalid);
    assert_eq!(Group::new([2, 1]).unwrap_err().code(), Code::Invalid);
    assert_eq!(Group::new([1, 1]).unwrap_err().code(), Code::Invalid);
}

/// 端口在 crate 之外可实现：单进程组也能走完一次可观察的变换。
///
/// 恒等实现在空函数下也能通过，测不出任何东西，所以这里故意把缓冲乘以 2。
#[test]
fn the_port_is_implementable_outside_and_the_transform_lands() {
    struct Doubling;
    impl Collective for Doubling {
        fn all_reduce(&mut self, _group: &Group, reduce: Reduce, buf: &mut [f32]) -> Result<()> {
            if reduce == Reduce::Sum {
                for value in buf.iter_mut() {
                    *value *= 2.0;
                }
            }
            Ok(())
        }
    }
    let group = Group::new([0]).expect("单进程组");
    let mut buf = [1.0f32];
    Doubling
        .all_reduce(&group, Reduce::Sum, &mut buf)
        .expect("归约");
    assert_eq!(buf, [2.0], "变换必须真的落到缓冲上");
}

#[test]
fn every_kind_names_itself() {
    let names: Vec<&str> = Reduce::ALL.iter().map(|kind| kind.as_str()).collect();
    assert_eq!(names, ["sum", "max", "min", "prod"]);
}

/// 单位元必须中立：拿它去 `apply` 任何值都不能改变那个值。
///
/// `Max` 的单位元写成 `f32::MIN` 时这条会在负值上失败——那正是最容易被漏掉的输入。
#[test]
fn the_identity_is_neutral_for_every_kind() {
    let samples = [
        -1.5e30,
        -3.0,
        -0.0,
        0.0,
        1.0e-30,
        7.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    for kind in Reduce::ALL {
        for value in samples {
            assert_eq!(
                kind.apply(kind.identity(), value),
                value,
                "{kind:?} 的单位元吃掉了 {value}"
            );
            assert_eq!(
                kind.apply(value, kind.identity()),
                value,
                "{kind:?} 的单位元吃掉了 {value}"
            );
        }
    }
}

#[test]
fn apply_is_the_expected_arithmetic() {
    assert_eq!(Reduce::Sum.apply(2.0, 3.0), 5.0);
    assert_eq!(Reduce::Prod.apply(2.0, 3.0), 6.0);
    assert_eq!(Reduce::Max.apply(2.0, 3.0), 3.0);
    assert_eq!(Reduce::Min.apply(2.0, 3.0), 2.0);
    // 负值上的极值：`0.0` 不能是任何一个极值的单位元。
    assert_eq!(Reduce::Max.apply(-9.0, -2.0), -2.0);
    assert_eq!(Reduce::Min.apply(-9.0, -2.0), -9.0);
}

/// NaN 必须传播，而不是被 `f32::max` / `f32::min` 静默吃掉。
#[test]
fn a_non_finite_value_is_never_silently_dropped() {
    assert!(Reduce::Max.apply(f32::NAN, 1.0).is_nan());
    assert!(Reduce::Min.apply(1.0, f32::NAN).is_nan());
    assert!(Reduce::Sum.apply(f32::NAN, 1.0).is_nan());
    assert!(Reduce::Prod.apply(1.0, f32::NAN).is_nan());
}

/// 空输入没有答案：不许拿单位元冒充"这一组算完了"。
#[test]
fn folding_nothing_is_not_a_value() {
    for kind in Reduce::ALL {
        assert_eq!(kind.fold(&[]), None, "{kind:?} 给空输入编了一个值");
    }
    assert_eq!(Reduce::Sum.fold(&[1.0]), Some(1.0));
    assert_eq!(Reduce::Max.fold(&[-4.0, -7.0]), Some(-4.0));
    assert_eq!(Reduce::Prod.fold(&[2.0, 3.0, 4.0]), Some(24.0));
}

/// 占位远端后端拒绝任何集合操作；缓冲原样保留，不产生半个归约结果。
#[test]
fn all_reduce_is_unsupported_and_leaves_the_buf_untouched() {
    let mut remote = Remote::new();
    let group = Group::new([0, 1]).expect("双进程组");
    let mut buf = [1.0f32, 2.0, 3.0];

    let error = remote
        .all_reduce(&group, Reduce::Sum, &mut buf)
        .expect_err("远端未挂载应拒绝归约");
    assert_eq!(error.code(), Code::Unsupported);
    assert_eq!(buf, [1.0, 2.0, 3.0], "拒绝时不得改动调用方缓冲");
}

/// [`Local`] 与 [`Remote`] 能放进同一个 `Box<dyn Collective>`：端口契约不随后端切换。
#[test]
fn local_and_remote_share_the_collective_port() {
    let topology = Topology::new(2, 1, 1).expect("tp 2");
    let mut local: Box<dyn Collective> = Box::new(Local::new(topology, 2).expect("宽度非零"));
    let mut remote: Box<dyn Collective> = Box::new(Remote::new());

    let group = topology.group(Parallel::Tensor, 0).expect("唯一一组");
    let mut buf = [0.0f32; 2];
    local
        .all_reduce(&group, Reduce::Sum, &mut buf)
        .expect("进程内归约");
    assert_eq!(
        remote
            .all_reduce(&group, Reduce::Sum, &mut buf)
            .unwrap_err()
            .code(),
        Code::Unsupported
    );
}

/// TP 一次归约：两个 rank 各持部分和，归约后两边都拿到完整结果。
#[test]
fn a_tensor_parallel_group_reduces_into_every_member() {
    let topology = topology();
    let mut local = Local::new(topology, 3).expect("宽度非零");
    local.load(0, &[1.0, 2.0, 3.0]).expect("rank 0");
    local.load(1, &[10.0, 20.0, 30.0]).expect("rank 1");
    let group = topology.group(Parallel::Tensor, 0).expect("第 0 组");
    let mut buf = [1.0, 2.0, 3.0];
    local
        .all_reduce(&group, Reduce::Sum, &mut buf)
        .expect("归约");
    assert_eq!(buf, [11.0, 22.0, 33.0]);
    assert_eq!(local.replica(0).expect("rank 0"), buf);
    assert_eq!(local.replica(1).expect("rank 1"), buf);
    // 组外的 rank 不该被这次归约改动。
    assert_eq!(local.replica(2).expect("rank 2"), [0.0, 0.0, 0.0]);
}

/// 数据并行的组跨整个 `tp · pp`，成员是 `0` 与 `2`；归约只读自己组里的镜像。
#[test]
fn a_data_parallel_group_reduces_only_its_own_members() {
    let topology = topology();
    let mut local = Local::new(topology, 2).expect("宽度非零");
    for rank in 0..4 {
        let base = f32::from(rank as u8) + 1.0;
        local.load(rank, &[base, -base]).expect("rank");
    }
    let group = topology.group(Parallel::Data, 0).expect("第 0 组");
    assert_eq!(group.ranks(), [0, 2]);
    let mut buf = [0.0; 2];
    local
        .all_reduce(&group, Reduce::Max, &mut buf)
        .expect("归约");
    assert_eq!(buf, [3.0, -1.0], "max 按元素取，不和 rank 1 的 -2 混");
    assert_eq!(local.replica(2).expect("rank 2"), buf);
}

/// 归约顺序必须由组的升序决定，而不是由切片顺序决定：混合正负时能暴露取错顺序。
#[test]
fn reduction_order_follows_the_group_not_the_call_order() {
    let topology = Topology::new(3, 1, 1).expect("tp 3");
    let mut local = Local::new(topology, 1).expect("宽度非零");
    local.load(0, &[-5.0]).expect("rank 0");
    local.load(1, &[7.0]).expect("rank 1");
    local.load(2, &[-1.0]).expect("rank 2");
    let group = topology.group(Parallel::Tensor, 0).expect("唯一一组");
    assert_eq!(local.reduce(&group, Reduce::Min).expect("归约"), [-5.0]);
    assert_eq!(local.reduce(&group, Reduce::Max).expect("归约"), [7.0]);
}

#[test]
fn a_width_mismatch_is_refused_on_both_sides() {
    let mut local = Local::new(topology(), 2).expect("宽度非零");
    assert_eq!(local.load(0, &[1.0]).unwrap_err().code(), Code::Invalid);
    assert_eq!(
        local.load(4, &[1.0, 2.0]).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(local.replica(4).unwrap_err().code(), Code::Invalid);
    let group = topology().group(Parallel::Tensor, 0).expect("第 0 组");
    let mut short = [0.0; 1];
    assert_eq!(
        local
            .all_reduce(&group, Reduce::Sum, &mut short)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    assert_eq!(Local::new(topology(), 0).unwrap_err().code(), Code::Invalid);
}

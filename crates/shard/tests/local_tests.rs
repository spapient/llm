use super::*;
use crate::{Parallel, Topology};

fn topology() -> Topology {
    Topology::new(2, 1, 2).expect("tp 2 · dp 2")
}

/// TP 一次归约：两个 rank 各持部分和，归约后两边都拿到完整结果。
#[test]
fn a_tensor_parallel_group_reduces_into_every_member() {
    let topology = topology();
    let mut local = Local::new(topology, 3).expect("宽度非零");
    local.load(0, &[1.0, 2.0, 3.0]).expect("rank 0");
    local.load(1, &[10.0, 20.0, 30.0]).expect("rank 1");
    let group = topology.group(Parallel::Tensor, 0).expect("第 0 组");
    let mut buffer = [1.0, 2.0, 3.0];
    local
        .all_reduce(&group, Reduce::Sum, &mut buffer)
        .expect("归约");
    assert_eq!(buffer, [11.0, 22.0, 33.0]);
    assert_eq!(local.replica(0).expect("rank 0"), buffer);
    assert_eq!(local.replica(1).expect("rank 1"), buffer);
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
    let mut buffer = [0.0; 2];
    local
        .all_reduce(&group, Reduce::Max, &mut buffer)
        .expect("归约");
    assert_eq!(buffer, [3.0, -1.0], "max 按元素取，不和 rank 1 的 -2 混");
    assert_eq!(local.replica(2).expect("rank 2"), buffer);
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

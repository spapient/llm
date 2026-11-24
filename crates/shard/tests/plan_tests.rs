use super::*;
use diagnostic::Code;

#[test]
fn a_single_rank_plan_holds_the_world_and_full_layer_range() {
    let plan = Plan::single();
    assert_eq!(plan.tp_idx(), 0);
    assert_eq!(plan.pp_idx(), 0);
    assert_eq!(plan.dp_idx(), 0);
    assert_eq!(plan.tp_group().expect("TP 组").ranks(), [0]);
    assert_eq!(plan.pp_group().expect("PP 组").ranks(), [0]);
    assert_eq!(plan.dp_group().expect("DP 组").ranks(), [0]);
    assert_eq!(plan.layer_range(7).expect("整段"), (0, 7));
}

#[test]
fn each_axis_group_keeps_the_other_two_idxs_fixed() {
    let topology = Topology::new(2, 3, 2).expect("合法");
    // rank 2 = (tp 0, pp 1, dp 0)。
    let plan = Plan::new(topology, 2).expect("rank 在世界内");
    assert_eq!((plan.tp_idx(), plan.pp_idx(), plan.dp_idx()), (0, 1, 0));

    assert_eq!(plan.tp_group().expect("TP 组").ranks(), [2, 3]);
    assert_eq!(plan.pp_group().expect("PP 组").ranks(), [0, 2, 4]);
    assert_eq!(plan.dp_group().expect("DP 组").ranks(), [2, 8]);

    // PP 段：6 层按 3 段等分，第 1 段是 [2, 4)。
    assert_eq!(plan.layer_range(6).expect("层段"), (2, 4));
}

#[test]
fn layer_range_rejects_non_divisible_and_empty() {
    let topology = Topology::new(2, 3, 2).expect("合法");
    let plan = Plan::new(topology, 2).expect("rank 在世界内");
    assert_eq!(
        plan.layer_range(5).unwrap_err().code(),
        Code::Invalid,
        "5 层不能被 pp=3 整除"
    );
    assert_eq!(plan.layer_range(0).unwrap_err().code(), Code::Invalid);
}

#[test]
fn every_rank_belongs_to_its_three_axis_groups() {
    let topology = Topology::new(3, 2, 4).expect("合法");
    for rank in 0..topology.world() {
        let plan = Plan::new(topology, rank as u32).expect("rank 在世界内");
        for group in [
            plan.tp_group().expect("TP 组"),
            plan.pp_group().expect("PP 组"),
            plan.dp_group().expect("DP 组"),
        ] {
            assert!(
                group.ranks().contains(&(rank as u32)),
                "rank {rank} 必须在自己的组里"
            );
        }
    }
}

#[test]
fn layer_ranges_partition_the_stacked_layers_without_overlap() {
    let topology = Topology::new(1, 4, 1).expect("合法");
    let layers = 12;
    let mut seen = vec![0usize; layers];
    for pp_idx in 0..topology.pp() {
        let rank = topology.rank(0, pp_idx, 0).expect("rank");
        let plan = Plan::new(topology, rank).expect("rank 在世界内");
        let (start, end) = plan.layer_range(layers).expect("层段");
        assert_eq!(end - start, layers / topology.pp(), "各段层数一致");
        for layer in seen.iter_mut().skip(start).take(end - start) {
            *layer += 1;
        }
    }
    assert!(seen.iter().all(|hits| *hits == 1), "层段必须恰好划开一遍");
}

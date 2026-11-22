use super::*;
use diagnostic::Code;

#[test]
fn the_world_is_the_product_of_the_three_axes() {
    let topology = Topology::new(4, 2, 3).expect("合法");
    assert_eq!(topology.world(), 24);
    assert!(topology.check_world(24).is_ok());
    assert_eq!(topology.check_world(12).unwrap_err().code(), Code::Invalid);
    assert_eq!(Topology::new(0, 1, 1).unwrap_err().code(), Code::Invalid);
    assert_eq!(Topology::single(), Topology::new(1, 1, 1).expect("合法"));
}

#[test]
fn each_axis_has_its_own_group_stride() {
    let topology = Topology::new(2, 2, 2).expect("合法");
    let groups = |parallel| -> Vec<Vec<u32>> {
        (0..topology.world() / 2)
            .map(|idx| {
                topology
                    .group(parallel, idx)
                    .expect("组存在")
                    .ranks()
                    .to_vec()
            })
            .collect()
    };
    assert_eq!(
        groups(Parallel::Tensor),
        [vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]
    );
    assert_eq!(
        groups(Parallel::Pipeline),
        [vec![0, 2], vec![1, 3], vec![4, 6], vec![5, 7]]
    );
    assert_eq!(
        groups(Parallel::Data),
        [vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]]
    );
}

/// 三个维度必须对称：组内只沿**该维**移动，组大小就是该维的并行度。
///
/// 旧实现里 `Pipeline` 的组是整个 `tp · pp` 平面、组号当 `dp` 下标用，于是同一个
/// `Parallel::Pipeline` 有两套下标域：`rank` 按 `pp` 校验，`group` 却按 `dp` 编号
/// （3·5·7 上 `rank(_, 4, _)` 合法，而 `group(Pipeline, 4)` 指的是 `dp = 4`）。这条用反解
/// 钉住"只有 pp 那一维在动"，组数则回到 `world / pp`。
#[test]
fn each_axis_group_moves_only_along_that_axis() {
    let topology = Topology::new(2, 3, 2).expect("合法");
    let pipeline = topology.group(Parallel::Pipeline, 2).expect("组号在区间内");
    assert_eq!(pipeline.size(), topology.pp(), "组大小必须等于该维并行度");
    assert_eq!(pipeline.ranks(), [6, 8, 10]);
    let idxs: Vec<_> = pipeline
        .ranks()
        .iter()
        .map(|rank| topology.idxs(*rank).expect("反解"))
        .collect();
    assert_eq!(idxs, [(0, 0, 1), (0, 1, 1), (0, 2, 1)]);
    assert_eq!(
        topology
            .group(Parallel::Pipeline, topology.world() / topology.pp())
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

#[test]
fn an_idx_outside_the_axis_is_rejected() {
    let topology = Topology::new(2, 2, 2).expect("合法");
    for parallel in Parallel::ALL {
        assert_eq!(
            topology.group(parallel, 4).unwrap_err().code(),
            Code::Invalid
        );
    }
}

/// 乘积溢出时不能只把 `world()` 饱和掉：那样 `check` 会跟着接受一个不可能的进程数，
/// 而 `group` 又用回绕的乘积发出一串"看起来合法"的 rank。
#[test]
fn a_topology_whose_world_does_not_fit_in_u32_is_refused() {
    let error = Topology::new(usize::MAX, 2, 2).expect_err("溢出");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("u32"), "{error}");
    // 恰好超出 u32 也算无解：rank 是 u32，最后一个 rank 写不出来。
    assert_eq!(
        Topology::new(1 << 16, 1 << 16, 1).unwrap_err().code(),
        Code::Invalid
    );
    // 卡在 u32::MAX 之下的乘积仍然合法，且 world 是精确值而不是饱和值。
    let huge = Topology::new(65_535, 65_535, 1).expect("乘积 4_294_836_225 < u32::MAX");
    assert_eq!(huge.world(), 4_294_836_225);
    assert!(huge.check_world(4_294_836_225).is_ok());
    assert_eq!(
        huge.check_world(usize::MAX).unwrap_err().code(),
        Code::Invalid
    );
}

/// 三个维度的组必须各自把 `0..world` 恰切一遍：不多不少、不重叠。
/// 布局规则一旦算错，这里会先于训练发现。
#[test]
fn every_axis_partitions_the_ranks_exactly_once() {
    for (tp, pp, dp) in [(1, 1, 1), (2, 2, 2), (3, 5, 7), (4, 1, 3), (1, 6, 2)] {
        let topology = Topology::new(tp, pp, dp).expect("合法");
        let world = topology.world();
        // 组数 = 另两维的乘积：这一列与实现无关，是"恰好划开"要核对的那个数。
        let counts = [dp * pp, tp * dp, tp * pp];
        for (parallel, count) in Parallel::ALL.into_iter().zip(counts) {
            let mut seen = vec![0usize; world];
            for idx in 0..count {
                for &rank in topology.group(parallel, idx).expect("组存在").ranks() {
                    seen[rank as usize] += 1;
                }
            }
            assert!(
                seen.iter().all(|hits| *hits == 1),
                "{parallel:?} 在 {tp}·{pp}·{dp} 上没有恰好划开：{seen:?}"
            );
            assert_eq!(
                topology.group(parallel, count).unwrap_err().code(),
                Code::Invalid,
                "{parallel:?} 的第 {count} 组不存在"
            );
        }
    }
}

#[test]
fn axis_names_are_stable() {
    let names: Vec<&str> = Parallel::ALL.iter().map(|axis| axis.as_str()).collect();
    assert_eq!(names, ["tp", "pp", "dp"]);
}

/// 布局规则只有一个方向是可读的公式，反解必须与它逐点互逆——否则 `group` 与
/// "我在第几段"会各说一套。
#[test]
fn rank_and_idxs_are_inverses() {
    for (tp, pp, dp) in [(1, 1, 1), (2, 2, 2), (3, 1, 4), (1, 5, 2)] {
        let topology = Topology::new(tp, pp, dp).expect("合法");
        for rank in 0..topology.world() {
            let flat = u32::try_from(rank).expect("world 收窄");
            let (tp_idx, pp_idx, dp_idx) = topology.idxs(flat).expect("反解");
            assert!(tp_idx < tp && pp_idx < pp && dp_idx < dp);
            assert_eq!(
                topology.rank(tp_idx, pp_idx, dp_idx),
                Ok(flat),
                "{tp}·{pp}·{dp} 的 rank {rank} 反解后不再等于自己"
            );
        }
        assert_eq!(
            topology
                .idxs(u32::try_from(topology.world()).expect("world 收窄"))
                .unwrap_err()
                .code(),
            Code::Invalid
        );
    }
}

#[test]
fn an_idx_outside_its_own_degree_is_refused() {
    let topology = Topology::new(2, 3, 4).expect("合法");
    assert_eq!(topology.rank(2, 0, 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(topology.rank(0, 3, 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(topology.rank(0, 0, 4).unwrap_err().code(), Code::Invalid);
    assert_eq!(topology.rank(1, 2, 3), Ok(23));
}

use super::*;

#[test]
fn the_world_is_the_product_of_the_three_axes() {
    let topology = Topology::new(4, 2, 3).expect("合法");
    assert_eq!(topology.world(), 24);
    assert!(topology.check(24).is_ok());
    assert_eq!(topology.check(12).unwrap_err().code(), Code::Invalid);
    assert_eq!(Topology::new(0, 1, 1).unwrap_err().code(), Code::Invalid);
}

#[test]
fn each_axis_has_its_own_group_stride() {
    let topology = Topology::new(2, 2, 2).expect("合法");
    let tensor: Vec<Vec<u32>> = (0..4)
        .map(|index| {
            topology
                .group(Parallel::Tensor, index)
                .expect("张量组")
                .ranks()
                .to_vec()
        })
        .collect();
    assert_eq!(tensor, [vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]);
    assert_eq!(
        topology
            .group(Parallel::Pipeline, 1)
            .expect("流水组")
            .ranks(),
        [4, 5, 6, 7]
    );
    assert_eq!(
        topology.group(Parallel::Data, 1).expect("数据组").ranks(),
        [1, 5]
    );
}

#[test]
fn an_index_outside_the_axis_is_rejected() {
    let topology = Topology::new(2, 2, 2).expect("合法");
    assert_eq!(
        topology.group(Parallel::Tensor, 4).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        topology.group(Parallel::Data, 4).unwrap_err().code(),
        Code::Invalid
    );
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
    assert!(huge.check(4_294_836_225).is_ok());
    assert_eq!(huge.check(usize::MAX).unwrap_err().code(), Code::Invalid);
}

/// 三个维度的组必须各自把 `0..world` 恰切一遍：不多不少、不重叠。
/// 布局规则一旦算错，这里会先于训练发现。
#[test]
fn every_axis_partitions_the_ranks_exactly_once() {
    for (tp, pp, dp) in [(1, 1, 1), (2, 2, 2), (3, 5, 7), (4, 1, 3), (1, 6, 2)] {
        let topology = Topology::new(tp, pp, dp).expect("合法");
        let world = topology.world();
        for parallel in Parallel::ALL {
            let count = match parallel {
                Parallel::Tensor => dp * pp,
                Parallel::Pipeline => dp,
                Parallel::Data => tp * pp,
            };
            let mut seen = vec![0usize; world];
            for index in 0..count {
                for &rank in topology.group(parallel, index).expect("组存在").ranks() {
                    seen[rank as usize] += 1;
                }
            }
            assert!(
                seen.iter().all(|count| *count == 1),
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

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

/// 三个维度必须对称：组内只沿**该维**移动，组大小就是该维的并行度，组数就是 `world / 该维`。
///
/// 旧实现里 `Pipeline` 的组是整个 `tp · pp` 平面、组号当 `dp` 下标用，于是同一个
/// `Parallel::Pipeline` 有两套下标域：`rank` 按 `pp` 校验，`group` 却按 `dp` 编号
/// （3·5·7 上 `rank(_, 4, _)` 合法，而 `group(Pipeline, 4)` 指的是 `dp = 4`）。这张表在
/// 2·3·2 上逐轴同时核对成员与反解，把"只有本维在动"钉死。
#[test]
fn each_axis_group_moves_only_along_that_axis() {
    let topology = Topology::new(2, 3, 2).expect("合法");
    struct Row {
        parallel: Parallel,
        idx: usize,
        degree: usize,
        members: &'static [u32],
        idxs: &'static [(usize, usize, usize)],
    }
    let table = [
        Row {
            parallel: Parallel::Tensor,
            idx: 1,
            degree: 2,
            members: &[2, 3],
            idxs: &[(0, 1, 0), (1, 1, 0)],
        },
        Row {
            parallel: Parallel::Pipeline,
            idx: 2,
            degree: 3,
            members: &[6, 8, 10],
            idxs: &[(0, 0, 1), (0, 1, 1), (0, 2, 1)],
        },
        Row {
            parallel: Parallel::Data,
            idx: 1,
            degree: 2,
            members: &[1, 7],
            idxs: &[(1, 0, 0), (1, 0, 1)],
        },
    ];
    for Row {
        parallel,
        idx,
        degree,
        members,
        idxs,
    } in table
    {
        let group = topology.group(parallel, idx).expect("组号在区间内");
        assert_eq!(group.ranks(), members, "{parallel:?} 第 {idx} 组的成员");
        assert_eq!(
            group.size(),
            degree,
            "{parallel:?} 的组大小必须等于该维并行度"
        );
        let solved: Vec<_> = members
            .iter()
            .map(|rank| topology.idxs(*rank).expect("反解"))
            .collect();
        assert_eq!(solved, idxs, "{parallel:?} 第 {idx} 组反解出的下标");
        // 组数由另两维的乘积给出：第 `count` 组必须不存在。
        let count = topology.world() / degree;
        assert_eq!(
            topology.group(parallel, count).unwrap_err().code(),
            Code::Invalid,
            "{parallel:?} 的第 {count} 组不存在"
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
    for (tp_idx, pp_idx, dp_idx) in [(2, 0, 0), (0, 3, 0), (0, 0, 4)] {
        assert_eq!(
            topology.rank(tp_idx, pp_idx, dp_idx).unwrap_err().code(),
            Code::Invalid,
            "({tp_idx}, {pp_idx}, {dp_idx}) 越过了自己那一维"
        );
    }
    assert_eq!(topology.rank(1, 2, 3), Ok(23));
}

/// 行序 = 判别式：`degree` / `place` / `as_str` 都走 `NAMES_[self as usize]`，这条约定错了
/// 不会有编译错误，只会把组切到别的维上去。
#[test]
fn every_axis_row_matches_its_discriminant() {
    let topology = Topology::new(2, 3, 5).expect("合法");
    let table = [
        (Parallel::Tensor, "tp", topology.tp()),
        (Parallel::Pipeline, "pp", topology.pp()),
        (Parallel::Data, "dp", topology.dp()),
    ];
    for (row, (axis, name, degree)) in table.into_iter().enumerate() {
        assert_eq!(axis as usize, row, "{name} 的行号与判别式不一致");
        assert_eq!(axis.as_str(), name);
        // 组大小就是该维的并行度——判别式若指错了维，这里会拿到别的进制。
        assert_eq!(
            topology.group(axis, 0).expect("首组").size(),
            degree,
            "{name} 的组大小不是它自己的并行度"
        );
    }
}

/// 布局只有一条规则：rank 是 `[tp_idx, pp_idx, dp_idx]` 按 `[tp, pp, dp]` 编码的混合进制数。
/// 反解与它逐点互逆（上面那条钉住了），这里钉"编码本身就是那条公式"。
#[test]
fn the_rank_is_the_mixed_radix_code_of_its_axes() {
    for (tp, pp, dp) in [(2, 3, 5), (4, 1, 2), (1, 1, 7)] {
        let topology = Topology::new(tp, pp, dp).expect("合法");
        for dp_idx in 0..dp {
            for pp_idx in 0..pp {
                for tp_idx in 0..tp {
                    let expected = dp_idx * tp * pp + pp_idx * tp + tp_idx;
                    assert_eq!(
                        topology.rank(tp_idx, pp_idx, dp_idx),
                        Ok(u32::try_from(expected).expect("world ≤ u32::MAX")),
                        "{tp}·{pp}·{dp} 的 ({tp_idx}, {pp_idx}, {dp_idx}) 不是混合进制数"
                    );
                }
            }
        }
    }
}

/// [`Topology::group`] 的唯一依据现在是**一条等差序列**：第 `k` 个成员 = 首成员 + `k` · 本维步长。
///
/// 这条定律同时钉住两件事：组内只沿本维移动（另两维的下标逐点恒定），且组号的解码是
/// **压缩**进制（只在剩下的两维上数，而不是拿完整 rank 的高位当组号）。旧形态逐个走
/// `rank` 的三次下标校验，改错一位只体现为"某几组的成员错位"；这里在随机非方阵上逐轴逐组
/// 核对首成员加步长与真正的反解。
///
/// 同一条定律也钉住编码/解码这一对原语：`weigh(digits(flat)) == flat` 对 `0..world` 逐点成立。
mod progression {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(testkit::prop::config())]

        #[test]
        fn a_group_is_the_progression_of_its_axis(
            tp in 1usize..=4,
            pp in 1usize..=4,
            dp in 1usize..=4,
        ) {
            let topology = Topology::new(tp, pp, dp).expect("合法");
            let strides = topology.strides();
            for parallel in Parallel::ALL {
                let own = parallel as usize;
                let degree = topology.degree(parallel);
                let stride = strides[own];
                for idx in 0..topology.world() / degree {
                    let group = topology.group(parallel, idx).expect("组号在区间内");
                    prop_assert_eq!(group.size(), degree, "{} 的组大小", parallel.as_str());
                    let head = group.ranks()[0] as usize;
                    let base = topology.idxs(group.ranks()[0]).expect("反解");
                    let base = [base.0, base.1, base.2];
                    for (step, rank) in group.ranks().iter().enumerate() {
                        prop_assert_eq!(
                            *rank as usize,
                            head + step * stride,
                            "{} 第 {} 组的第 {} 个成员不在首成员加步长的位置上",
                            parallel.as_str(),
                            idx,
                            step,
                        );
                        let solved = topology.idxs(*rank).expect("反解");
                        let solved = [solved.0, solved.1, solved.2];
                        prop_assert_eq!(solved[own], step, "{} 的本维下标", parallel.as_str());
                        for axis in 0..DIM {
                            if axis != own {
                                prop_assert_eq!(
                                    solved[axis],
                                    base[axis],
                                    "{} 第 {} 组动到了第 {} 维",
                                    parallel.as_str(),
                                    idx,
                                    axis,
                                );
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn weighing_the_digits_of_a_rank_gives_it_back(
            tp in 1usize..=4,
            pp in 1usize..=4,
            dp in 1usize..=4,
        ) {
            let topology = Topology::new(tp, pp, dp).expect("合法");
            for flat in 0..topology.world() {
                prop_assert_eq!(topology.weigh(topology.digits(flat, None)), flat);
            }
        }
    }
}

// ---- plan_tests ----

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

// ---- policy_tests ----

/// 优先级短名、序数与判别式一一对应；序数越小越先调度。
#[test]
fn priority_names_and_ordinals_are_consistent() {
    let expected = [("high", 0u8), ("normal", 1u8), ("low", 2u8)];
    for (priority, (name, ordinal)) in Priority::ALL.iter().zip(expected) {
        assert_eq!(priority.as_str(), name, "短名不符");
        assert_eq!(priority.ordinal(), ordinal, "调度序数不符");
        assert_eq!(Priority::from_name(name), Some(*priority), "短名反解不符");
    }
    assert_eq!(
        Priority::from_name("nope"),
        None,
        "未知优先级不应当成合法值"
    );
    assert!(Priority::High < Priority::Normal && Priority::Normal < Priority::Low);
}

/// 单进程策略：1×1×1、求和、平均、普通优先级，恒合法。
#[test]
fn single_process_config_is_the_identity_policy() {
    let config = ParallelConfig::single();
    assert_eq!((config.tp(), config.pp(), config.dp()), (1, 1, 1));
    assert!(config.average());
    assert_eq!(config.priority(), Priority::Normal);
    let topology = config.topology().expect("单进程拓扑合法");
    assert_eq!(topology.world(), 1);
}

/// 三维度数进拓扑；派生世界数为三者乘积。
#[test]
fn config_derives_the_matching_topology() {
    let config = ParallelConfig::new(2, 2, 2).expect("2·2·2 合法");
    let topology = config.topology().expect("派生拓扑");
    assert_eq!(topology.world(), 8);
    assert_eq!((config.tp(), config.pp(), config.dp()), (2, 2, 2));
}

/// 零维度数在构造期拒绝，与 [`Topology::new`] 同一判据。
#[test]
fn zero_degree_is_rejected_at_construction() {
    assert!(ParallelConfig::new(0, 2, 2).is_err());
    assert!(ParallelConfig::new(2, 0, 2).is_err());
    assert!(ParallelConfig::new(2, 2, 0).is_err());
}

/// 旋钮链式覆盖：归约、平均、优先级各自独立可改。
#[test]
fn knobs_are_overridden_independently() {
    let config = ParallelConfig::new(1, 1, 2)
        .expect("dp 2 合法")
        .with_reduce(crate::Reduce::Max)
        .with_average(false)
        .with_priority(Priority::High);
    assert_eq!(config.reduce(), crate::Reduce::Max);
    assert!(!config.average());
    assert_eq!(config.priority(), Priority::High);
}

/// 非法度数上抛 [`Code::Invalid`]，不把拓扑错误留到装配期。
#[test]
fn rejected_config_carries_invalid_code() {
    let error = ParallelConfig::new(0, 1, 1).expect_err("零维应拒绝");
    assert_eq!(error.code(), Code::Invalid);
}

/// 策略度数与直接构造的拓扑一致：配置与拓扑不是两套可漂移的事实。
#[test]
fn config_topology_matches_direct_construction() {
    let config = ParallelConfig::new(2, 1, 3).expect("tp2 dp3 合法");
    let direct = Topology::new(2, 1, 3).expect("直接构造合法");
    assert_eq!(config.topology().expect("派生"), direct);
}

// ---- schedule_tests ----

#[test]
fn empty_queue_drains_to_nothing() {
    let queue: Queue<usize> = Queue::new();
    assert!(queue.is_empty());
    assert_eq!(queue.len(), 0);
    assert!(queue.drain().is_empty());
}

#[test]
fn drain_orders_by_priority_ordinal_within_a_batch() {
    let mut queue = Queue::new();
    // 入队顺序打乱优先级；载荷编号记录入队先后。
    queue.submit(Priority::Low, 1);
    queue.submit(Priority::High, 2);
    queue.submit(Priority::Normal, 3);
    queue.submit(Priority::High, 4);

    let drained = queue.drain();
    let order: Vec<_> = drained.iter().map(Job::priority).collect();
    assert_eq!(
        order,
        [
            Priority::High,
            Priority::High,
            Priority::Normal,
            Priority::Low
        ]
    );
    // 同档保持入队先后：High 档 2 先于 4。
    let payloads: Vec<_> = drained.into_iter().map(Job::into_payload).collect();
    assert_eq!(payloads, [2, 4, 3, 1]);
}

#[test]
fn jobs_carry_their_payload_through_the_queue() {
    let mut queue: Queue<(&str, Priority)> = Queue::new();
    queue.submit(Priority::Normal, ("meta", Priority::Normal));
    let job = queue.drain().pop().expect("一条作业");
    assert_eq!(job.into_payload(), ("meta", Priority::Normal));
}

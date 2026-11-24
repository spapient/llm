use super::*;
use crate::{Parallel, Topology};
use diagnostic::Code;

fn topology() -> Topology {
    Topology::new(2, 1, 2).expect("tp 2 · dp 2")
}

/// 两个 rank 各载入一段输入的 TP 2 局：返回局与唯一组。
fn loaded_local(a: [f32; 2], b: [f32; 2]) -> (Local, Group) {
    let topology = Topology::new(2, 1, 1).expect("tp 2");
    let mut local = Local::new(topology, 2).expect("宽度非零");
    local.load(0, &a).expect("rank 0");
    local.load(1, &b).expect("rank 1");
    let group = topology.group(Parallel::Tensor, 0).expect("唯一一组");
    (local, group)
}

/// 两个 rank 的文件远端：共享一份以 `stem` 命名的临时文件，各载入一段输入。
fn file_remotes(stem: &str) -> (std::path::PathBuf, FileRemote, Group) {
    let backing = std::env::temp_dir().join(format!("{stem}{}", std::process::id()));
    let r0 = FileRemote::open(0, 2, 2, &backing).expect("rank 0");
    let r1 = FileRemote::open(1, 2, 2, &backing).expect("rank 1");
    r0.load(0, &[5.0, 6.0]).expect("rank 0");
    r1.load(1, &[7.0, 8.0]).expect("rank 1");
    let group = Group::new([0, 1]).expect("双进程组");
    (backing, r0, group)
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

/// `Reduce` 的短名、单位元与运算都从同一张表按判别式取出，所以行序必须与枚举顺序一致。
///
/// 表把三件必须同时成立的事实写在一行（加一种归约只改一处），代价是行序成了隐式约定：
/// 这条把它变成可执行断言，行序错了就不会是"悄悄取到别的种类"。
#[test]
fn the_kind_table_follows_the_enum_order() {
    for (position, kind) in Reduce::ALL.iter().enumerate() {
        assert_eq!(
            *kind as usize, position,
            "{kind:?} 的判别式与 ALL 的顺序不符"
        );
        assert_eq!(
            kind.as_str(),
            Reduce::NAMES[position],
            "{kind:?} 的短名与表里第 {position} 行不符"
        );
    }
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
        assert_eq!(kind.fold::<f32>(&[]), None, "{kind:?} 给空输入编了一个值");
    }
    assert_eq!(Reduce::Sum.fold(&[1.0]), Some(1.0));
    assert_eq!(Reduce::Max.fold(&[-4.0, -7.0]), Some(-4.0));
    assert_eq!(Reduce::Prod.fold(&[2.0, 3.0, 4.0]), Some(24.0));
}

/// 占位远端后端拒绝任何集合操作；缓冲原样保留，不产生半个归约结果。
#[test]
fn all_reduce_is_unsupported_and_leaves_the_buf_untouched() {
    let mut remote = Remote;
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
    let mut remote: Box<dyn Collective> = Box::new(Remote);

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
    assert_eq!(
        Local::<f32>::new(topology(), 0).unwrap_err().code(),
        Code::Invalid
    );
}

/// [`Local::reduce_into`] 是 [`Local::reduce`] 不分配的那条路径，[`Collective::all_reduce`] 走它。
///
/// 两条路径必须逐元素同解，四种归约都要对：只测 `Sum` 时，"`Max` 的缓冲没被整段覆盖"这种
/// 缺陷照样能过。缓冲先灌成 NaN——归约必须先填单位元再折叠，任何一格没被写到都会立刻暴露。
/// 同时钉住归约的纯度：归约之后每个 rank 的镜像仍是原值。
#[test]
fn reducing_into_a_buf_agrees_with_allocating_a_vec() {
    let topology = Topology::new(2, 1, 1).expect("tp 2");
    let mut local = Local::new(topology, 3).expect("宽度非零");
    local.load(0, &[2.0, -3.0, 0.5]).expect("rank 0");
    local.load(1, &[4.0, -1.0, -2.0]).expect("rank 1");
    let group = topology.group(Parallel::Tensor, 0).expect("唯一一组");
    for reduce in Reduce::ALL {
        let expected = local.reduce(&group, reduce).expect("归约");
        let mut buf = [f32::NAN; 3];
        local
            .reduce_into(&group, reduce, &mut buf)
            .expect("归约进缓冲");
        assert_eq!(
            buf.as_slice(),
            expected.as_slice(),
            "{}：进缓冲与出 Vec 两条路径不同解",
            reduce.as_str()
        );
        assert_eq!(local.replica(0).expect("rank 0"), [2.0, -3.0, 0.5]);
        assert_eq!(local.replica(1).expect("rank 1"), [4.0, -1.0, -2.0]);
    }
    let mut short = [0.0; 2];
    assert_eq!(
        local
            .reduce_into(&group, Reduce::Sum, &mut short)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

/// 文件传输的远端归约：两个 rank 共享一份文件，各写各的槽位，归约后两边都拿到完整结果——
/// 这是 [`Remote`] 占位之外的真实远端实现（跨进程可用，无 `unsafe`）。
#[test]
fn file_remote_reduces_across_ranks() {
    let backing = std::env::temp_dir().join(format!("shard-collective-{}", std::process::id()));
    let mut r0 = FileRemote::open(0, 2, 2, &backing).expect("rank 0");
    let r1 = FileRemote::open(1, 2, 2, &backing).expect("rank 1");
    assert_eq!(r0.rank(), 0);
    r0.load(0, &[1.0, 2.0]).expect("rank 0 输入");
    r1.load(1, &[3.0, 4.0]).expect("rank 1 输入");
    let group = Group::new([0, 1]).expect("双进程组");
    let mut buf = [0.0; 2];
    r0.all_reduce(&group, Reduce::Sum, &mut buf).expect("归约");
    assert_eq!(buf, [4.0, 6.0]);
    assert_eq!(r1.replica(0).expect("rank 0"), buf, "结果写回每个成员");
    assert_eq!(r1.replica(1).expect("rank 1"), buf);
    let _ = std::fs::remove_file(&backing);
}

/// 越界与宽度不符在文件后端同样被拒：`load`、`replica`、`all_reduce` 各自守住自己的判据。
#[test]
fn file_remote_refuses_out_of_world_and_width_mismatch() {
    let backing = std::env::temp_dir().join(format!("shard-collective-bad-{}", std::process::id()));
    let mut r0 = FileRemote::open(0, 2, 2, &backing).expect("rank 0");
    assert_eq!(r0.load(2, &[1.0, 2.0]).unwrap_err().code(), Code::Invalid);
    assert_eq!(r0.load(0, &[1.0]).unwrap_err().code(), Code::Invalid);
    assert_eq!(r0.replica(2).unwrap_err().code(), Code::Invalid);
    let group = Group::new([0, 1]).expect("双进程组");
    let mut short = [0.0; 1];
    assert_eq!(
        r0.all_reduce(&group, Reduce::Sum, &mut short)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
    assert_eq!(
        FileRemote::open(2, 2, 2, &backing).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        FileRemote::open(0, 0, 2, &backing).unwrap_err().code(),
        Code::Invalid
    );
    let _ = std::fs::remove_file(&backing);
}

/// 全部收集按组内顺序拼接：每个 rank 的镜像写进自己那一分块，不是按调用顺序。
#[test]
fn all_gather_concatenates_the_group_in_order() {
    let (mut local, group) = loaded_local([1.0, 2.0], [3.0, 4.0]);
    let mut buf = [0.0; 4];
    local.all_gather(&group, &mut buf).expect("收集");
    assert_eq!(buf, [1.0, 2.0, 3.0, 4.0], "按组内 rank 顺序拼接");
    // 缓冲长度不符（不是 size × width）要拒绝。
    let mut short = [0.0; 3];
    assert_eq!(
        local.all_gather(&group, &mut short).unwrap_err().code(),
        Code::Invalid
    );
}

/// 广播：源 rank 的值复制进缓冲，与其它 rank 的镜像无关。
#[test]
fn broadcast_copies_the_source_rank() {
    let topology = Topology::new(2, 1, 1).expect("tp 2");
    let mut local = Local::new(topology, 2).expect("宽度非零");
    local.load(0, &[7.0, 8.0]).expect("rank 0");
    local.load(1, &[1.0, 1.0]).expect("rank 1");
    let group = topology.group(Parallel::Tensor, 0).expect("唯一一组");
    let mut buf = [0.0; 2];
    local.broadcast(&group, 0, &mut buf).expect("广播");
    assert_eq!(buf, [7.0, 8.0]);
    // 源不在组内要拒绝。
    let mut other = [0.0; 2];
    let bad = Group::new([1]).expect("单成员组");
    assert_eq!(
        local.broadcast(&bad, 0, &mut other).unwrap_err().code(),
        Code::Invalid
    );
}

/// 文件后端同样支持收集与广播：槽位是跨进程共享的，语义与 [`Local`] 同构。
#[test]
fn file_remote_all_gathers_and_broadcasts() {
    let (backing, mut r0, group) = file_remotes("shard-collective-ops-");
    let mut gathered = [0.0; 4];
    r0.all_gather(&group, &mut gathered).expect("收集");
    assert_eq!(gathered, [5.0, 6.0, 7.0, 8.0]);
    let mut broadcast = [0.0; 2];
    r0.broadcast(&group, 1, &mut broadcast).expect("广播");
    assert_eq!(broadcast, [7.0, 8.0]);
    let _ = std::fs::remove_file(&backing);
}

/// 占位 [`Remote`] 对新增的收集/广播/归约散布同样走默认 `Unsupported`，不改动缓冲。
#[test]
fn the_placeholder_rejects_gather_and_broadcast() {
    let mut remote = Remote;
    let group = Group::new([0, 1]).expect("双进程组");
    let mut buf = [1.0f32, 2.0];
    assert_eq!(
        remote.all_gather(&group, &mut buf).unwrap_err().code(),
        Code::Unsupported
    );
    assert_eq!(
        remote.broadcast(&group, 0, &mut buf).unwrap_err().code(),
        Code::Unsupported
    );
    let mut full = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(
        remote
            .reduce_scatter(&group, Reduce::Sum, &mut full)
            .unwrap_err()
            .code(),
        Code::Unsupported
    );
    assert_eq!(buf, [1.0, 2.0], "拒绝时不得改动缓冲");
}

/// 归约散布：每个 rank 的分片按组内顺序用 `reduce` 折进对应分块；分块不相交时退化为收集。
#[test]
fn reduce_scatter_folds_each_shard_into_its_chunk() {
    let (mut local, group) = loaded_local([1.0, 2.0], [3.0, 4.0]);
    let mut buf = [0.0; 4];
    local
        .reduce_scatter(&group, Reduce::Sum, &mut buf)
        .expect("归约散布");
    assert_eq!(buf, [1.0, 2.0, 3.0, 4.0], "每个分片折进自己的分块");
    let mut short = [0.0; 3];
    assert_eq!(
        local
            .reduce_scatter(&group, Reduce::Sum, &mut short)
            .unwrap_err()
            .code(),
        Code::Invalid
    );
}

/// 文件后端的归约散布与 [`Local`] 同构：槽位是跨进程共享的，折进顺序由组决定。
#[test]
fn file_remote_reduce_scatters() {
    let (backing, mut r0, group) = file_remotes("shard-collective-rs-");
    let mut buf = [0.0; 4];
    r0.reduce_scatter(&group, Reduce::Sum, &mut buf)
        .expect("归约散布");
    assert_eq!(buf, [5.0, 6.0, 7.0, 8.0]);
    let _ = std::fs::remove_file(&backing);
}

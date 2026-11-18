use crate::dist::Distributed;
use diagnostic::{Code, Result};
use shard::{Local, Topology};

/// 归约槽位宽度：每个槽梯度元素个数。
const WIDTH: usize = 3;

/// 两进程数据并行拓扑：`tp=1, pp=1, dp=2`，rank 0/1 同属一个数据组。
fn topology() -> Topology {
    Topology::new(1, 1, 2).expect("拓扑合法")
}

/// 进程内后端：两个 rank 各一份镜像，槽宽度 [`WIDTH`]。
fn local_backend(topology: Topology) -> Local {
    Local::new(topology, WIDTH).expect("归约宽度合法")
}

#[test]
fn single_process_identity_synchronizes_without_a_backend() -> Result<()> {
    let mut distributed: Distributed<crate::Remote> = Distributed::none();
    let mut grads = [vec![1.0, 2.0, 3.0]];
    let mut views: Vec<&mut [f32]> = grads.iter_mut().map(|g| g.as_mut_slice()).collect();

    distributed.sync_grads(&mut views)?;

    assert!(!distributed.is_distributed(), "恒等回路不挂通信后端");
    assert_eq!(views[0], &[1.0, 2.0, 3.0], "单进程梯度原样保留");
    Ok(())
}

#[test]
fn data_parallel_averages_reduced_gradients_across_the_group() -> Result<()> {
    let topology = topology();
    let mut local = local_backend(topology);
    local.load(0, &[2.0, 4.0, 6.0])?;
    local.load(1, &[4.0, 8.0, 10.0])?;

    let mut distributed = Distributed::new(topology, 0, local)?;
    let mut grads = [vec![2.0, 4.0, 6.0]];
    let mut views: Vec<&mut [f32]> = grads.iter_mut().map(|g| g.as_mut_slice()).collect();

    distributed.sync_grads(&mut views)?;

    // 求和 = [6, 12, 16]，组大小 2 取平均 → [3, 6, 8]。
    assert_eq!(views[0], &[3.0, 6.0, 8.0]);
    assert_eq!(distributed.data_parallel(), 2, "数据并行度 2");
    Ok(())
}

#[test]
fn summation_mode_keeps_the_group_sum_without_dividing_by_size() -> Result<()> {
    let topology = topology();
    let mut local = local_backend(topology);
    local.load(0, &[2.0, 4.0, 6.0])?;
    local.load(1, &[4.0, 8.0, 10.0])?;

    let mut distributed = Distributed::new(topology, 0, local)?.with_average(false);
    let mut grads = [vec![2.0, 4.0, 6.0]];
    let mut views: Vec<&mut [f32]> = grads
        .iter_mut()
        .map(|gradient| gradient.as_mut_slice())
        .collect();

    distributed.sync_grads(&mut views)?;

    assert_eq!(views[0], &[6.0, 12.0, 16.0], "求和模式不除组数");
    Ok(())
}

#[test]
fn a_rank_outside_the_topology_is_rejected_at_construction() -> Result<()> {
    let topology = topology();
    let local = local_backend(topology);
    let error = Distributed::new(topology, 2, local).expect_err("rank 2 超出 dp=2 的世界");
    assert_eq!(error.code(), Code::Invalid);
    Ok(())
}

/// 会话级端到端：`Session::with_distributed` 接 2-rank 进程内后端，`optimize` 在步进前先把
/// 参数槽梯度组内平均，再按平均梯度（而非本 rank 的原始梯度）更新参数。
#[test]
fn distributed_session_averages_gradients_before_the_step() -> Result<()> {
    use crate::{Reference, Session, Sgd, TrainConfig};
    use autograd::{Sizes, Tape};
    use nn::ParamRole;
    use tensor::{DType, Shape};

    let width = 3usize;
    let mut tape = Tape::new();
    let p = tape.parameter();
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([width]), DType::F32);

    let grad0 = [0.1_f32, 0.2, 0.3];
    let grad1 = [0.3_f32, 0.6, 0.9];
    let topology = topology();
    let mut local = Local::new(topology, width)?;
    local.load(0, &grad0)?;
    local.load(1, &grad1)?;
    let distributed = Distributed::new(topology, 0, local)?;

    let mut session = Session::<Reference, Sgd, Local>::with_distributed(
        tape,
        sizes,
        Reference,
        Sgd::new(),
        TrainConfig::default(),
        distributed,
    )?;
    assert!(
        session.distributed().is_distributed(),
        "会话应挂上真实通信后端"
    );
    session.register_parameter(p, ParamRole::Trainable, vec![1.0_f32, 2.0, 3.0])?;
    session.load_gradient(p, &grad0)?;

    session.optimize()?;

    // 平均梯度 = (grad0 + grad1) / 2 = [0.2, 0.4, 0.6]，默认 lr = 1e-3。
    let expected = [0.9998_f32, 1.9996, 2.9994];
    let actual = session.slots()[0].params();
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-5,
            "参数[{index}] 应按平均梯度更新为 {expected}，实际 {actual}"
        );
    }
    Ok(())
}

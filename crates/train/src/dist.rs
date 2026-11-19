//! 分布式训练编排：拓扑绑定 + 梯度同步后端，配置化挂到训练会话上。
//!
//! 训练侧只认 [`Collective`] 契约：数据并行组内把各槽梯度按 [`Reduce::Sum`] 归约（可选除组数
//! 取平均），张量/流水并行的组划分由 [`Topology`] 派生。后端实现不同（进程内 [`Local`]、
//! 远端占位 [`Remote`]、后续 NCCL/Gloo），上层调用点不变。
//!
//! 一条泛型装配路径：[`Distributed::none`] 是单进程恒等回路（不持有通信句柄）；接真实后端时
//! 构造 [`Distributed::new`]，把拓扑、本 rank、通信后端一起登记进会话编排。

use diagnostic::{Code, Error, Result};
use shard::{Collective, Group, Parallel, Reduce, Topology};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::Distributed";

/// 分布式编排配置：拓扑、本进程 rank 与通信后端；梯度同步口径随配置走。
///
/// `E` 是通信后端实现；单进程场景用 [`Distributed::none`]，不必套 `Option` 分支穿透会话。
#[derive(Debug, Clone, PartialEq)]
pub struct Distributed<E: Collective> {
    topology: Topology,
    rank: u32,
    collective: Option<E>,
    /// 归约后是否除以数据并行组数（`true` = 平均梯度，`false` = 求和梯度）。
    average: bool,
    reduce: Reduce,
}

impl Distributed<Remote> {
    /// 单进程恒等编排：无拓扑、无后端，梯度不同步。后续装配多进程时用 [`Self::new`]。
    pub fn none() -> Self {
        Self {
            topology: Topology::single(),
            rank: 0,
            collective: None,
            average: true,
            reduce: Reduce::Sum,
        }
    }
}

impl<E: Collective> Distributed<E> {
    /// 由拓扑、本 rank 与通信后端构造；rank 必须落在拓扑世界内，否则归约组派生时会越界。
    ///
    /// # Errors
    ///
    /// `rank` 超出 `0..world`（[`Code::Invalid`]）。
    pub fn new(topology: Topology, rank: u32, collective: E) -> Result<Self> {
        if rank as usize >= topology.world() {
            return Err(Error::new(
                Code::Invalid,
                format!("rank {rank} 超出拓扑世界 0..{}", topology.world()),
            )
            .with(SITE));
        }
        Ok(Self {
            topology,
            rank,
            collective: Some(collective),
            average: true,
            reduce: Reduce::Sum,
        })
    }

    /// 归约后取平均（默认）；`false` 保留求和梯度。
    pub fn with_average(mut self, average: bool) -> Self {
        self.average = average;
        self
    }

    /// 归约种类；默认 [`Reduce::Sum`]（梯度同步的标准选择）。
    pub fn with_reduce(mut self, reduce: Reduce) -> Self {
        self.reduce = reduce;
        self
    }

    /// 拓扑只读投影。
    pub fn topology(&self) -> Topology {
        self.topology
    }

    /// 本进程全局 rank。
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// 数据并行组：同一份权重、不同微批的进程集合；单进程退化成 `[0]`。
    pub fn data_group(&self) -> Result<Group> {
        let index = self.topology.indices(self.rank)?.2;
        self.topology.group(Parallel::Data, index)
    }

    /// 组大小（数据并行度）；单进程为 1。
    pub fn data_parallel(&self) -> usize {
        self.topology.dp()
    }

    /// 是否接了真实通信后端：`false` 时同步是恒等回路，不产生网络动作。
    pub fn is_distributed(&self) -> bool {
        self.collective.is_some()
    }

    /// 把全部槽位梯度在数据并行组内同步：先组内归约，再按组数取平均（配置开启时）。
    ///
    /// 缓冲长度与槽位对齐由调用方保证；后端返回的错误原样透传。单进程无后端时直接返回。
    ///
    /// # Errors
    ///
    /// 组派生越界、后端归约失败，或平均除数为 0 时返回对应 [`Code`]。
    pub fn sync_grads(&mut self, slots: &mut [&mut [f32]]) -> Result<()> {
        let group = self.data_group()?;
        let Some(collective) = self.collective.as_mut() else {
            return Ok(());
        };
        synchronize(collective, &group, self.reduce, self.average, slots)?;
        Ok(())
    }
}

/// 组内归约各槽梯度；配置开启平均时再按数据并行组数取均值。
fn synchronize<E: Collective>(
    collective: &mut E,
    group: &Group,
    reduce: Reduce,
    average: bool,
    slots: &mut [&mut [f32]],
) -> Result<()> {
    reduce_slots(collective, group, reduce, slots)?;
    if average {
        average_slots(slots, group)?;
    }
    Ok(())
}

/// 组内对每个槽位梯度按 [`Reduce`] 归约；后端错误原样透传。
fn reduce_slots<E: Collective>(
    collective: &mut E,
    group: &Group,
    reduce: Reduce,
    slots: &mut [&mut [f32]],
) -> Result<()> {
    for grads in slots.iter_mut() {
        collective.all_reduce(group, reduce, grads)?;
    }
    Ok(())
}

/// 归约后按数据并行组数取平均；组大小为 0 是装配口径错误。
fn average_slots(slots: &mut [&mut [f32]], group: &Group) -> Result<()> {
    let divisor = group.size() as f32;
    if divisor == 0.0 {
        return Err(Error::new(Code::Internal, "数据并行组大小为 0").with(SITE));
    }
    for grads in slots.iter_mut() {
        for slot in grads.iter_mut() {
            *slot /= divisor;
        }
    }
    Ok(())
}

/// 远端占位后端别名：单进程 [`Distributed::none`] 用它占位，不挂真实通信栈。
pub use shard::{Local, Remote};

#[cfg(test)]
mod tests {
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
}

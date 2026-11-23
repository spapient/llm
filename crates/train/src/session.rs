//! 训练会话：tape → 反传计划 → 梯度需求分类 → 梯度缓冲 → 优化器步的最小闭环；学习率调度
//! （[`Schedule`] / [`Anneal`]）与分布式同步配置（[`Distributed`]）与它同住一处——会话是它们唯一的
//! 装配者，调度曲线与同步口径都随 [`TrainConfig`] 走。

use autograd::{Requirement, Sizes, Tape, Var};
use diagnostic::{Code, Error, Result};
use nn::ParamRole;
use shard::{Collective, Group, Parallel, ParallelConfig, Priority, Reduce, Topology};
use tensor::{DType, Shape};

use crate::optimizer::{Optimizer, OwnedSlot, Slot};
use crate::pool::Bufs;
use crate::{Data, Kernel};
use storage::{GradAccum, Moments, SlotMoments, Ties};

pub use shard::{Local, Remote};
use weights::Snapshot;

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::Session";

/// 训练配置：学习率峰值、warmup、总步数、裁剪与梯度累积窗口。
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    /// 学习率峰值（warmup 结束时）。
    pub learning_rate: f32,
    /// warmup 步数。
    pub warmup_steps: u64,
    /// 计划总步数（调度分母）。
    pub max_steps: u64,
    /// 全局梯度范数上限；`None` 表示不裁剪。
    pub clip_norm: Option<f32>,
    /// 梯度累积窗口：多少个 micro-batch 汇总一次再步进；`1` 为每步更新。
    pub accumulation: u32,
    /// warmup 之后的学习率衰减曲线。
    pub anneal: Anneal,
    /// 谷底占峰值的比例；必须落在 `[0, 1]`，越界在装配调度时拒绝。
    pub min_ratio: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            warmup_steps: 0,
            max_steps: 1,
            clip_norm: None,
            accumulation: 1,
            anneal: Anneal::default(),
            min_ratio: 0.0,
        }
    }
}

impl TrainConfig {
    /// 构造并校验：总步数非零、warmup 不超过总步数、学习率非负、累积窗口非零。
    pub fn new(
        learning_rate: f32,
        warmup_steps: u64,
        max_steps: u64,
        clip_norm: Option<f32>,
        accumulation: u32,
    ) -> Result<Self> {
        if max_steps == 0 {
            return Err(Error::new(Code::Invalid, "总步数不能为 0").with(SITE));
        }
        if warmup_steps > max_steps {
            return Err(Error::new(
                Code::Invalid,
                format!("warmup {warmup_steps} 超过总步数 {max_steps}"),
            )
            .with(SITE));
        }
        if learning_rate < 0.0 {
            return Err(Error::new(Code::Invalid, "学习率不能为负").with(SITE));
        }
        if accumulation == 0 {
            return Err(Error::new(Code::Invalid, "累积窗口不能为 0").with(SITE));
        }
        Ok(Self {
            learning_rate,
            warmup_steps,
            max_steps,
            clip_norm,
            accumulation,
            anneal: Anneal::default(),
            min_ratio: 0.0,
        })
    }

    /// 按可选覆盖构造：未给的字段取 [`Default`]。
    pub fn with_overrides(
        learning_rate: Option<f32>,
        warmup_steps: Option<u64>,
        max_steps: Option<u64>,
        clip_norm: Option<f32>,
        accumulation: Option<u32>,
    ) -> Result<Self> {
        let base = Self::default();
        Self::new(
            learning_rate.unwrap_or(base.learning_rate),
            warmup_steps.unwrap_or(base.warmup_steps),
            max_steps.unwrap_or(base.max_steps),
            clip_norm.or(base.clip_norm),
            accumulation.unwrap_or(base.accumulation),
        )
    }

    /// 设置 warmup 之后的衰减曲线（builder 风格，覆盖默认的线性衰减）。
    pub fn with_anneal(mut self, anneal: Anneal) -> Self {
        self.anneal = anneal;
        self
    }

    /// 设置谷底占峰值的比例（builder 风格）；越界不静默修正，装配调度时拒绝。
    pub fn with_min_ratio(mut self, min_ratio: f32) -> Self {
        self.min_ratio = min_ratio;
        self
    }

    /// 衰减曲线。
    pub fn anneal(&self) -> Anneal {
        self.anneal
    }

    /// 谷底占峰值的比例。
    pub fn min_ratio(&self) -> f32 {
        self.min_ratio
    }

    /// 梯度累积窗口。
    pub fn accumulation(&self) -> u32 {
        self.accumulation
    }
}

/// 取视图时按槽号记下的元数据：变量、角色、矩阵形状。
type SlotMeta = (Var, ParamRole, Option<(usize, usize)>);

/// 一个已登记的参数槽：变量编号、角色与常驻参数值。
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSlot {
    var: Var,
    role: ParamRole,
    params: Vec<f32>,
    matrix: Option<(usize, usize)>,
}

impl ParamSlot {
    /// 参数对应的图变量。
    pub fn var(&self) -> Var {
        self.var
    }

    /// 参数角色：分组与衰减豁免的派生依据。
    pub fn role(&self) -> ParamRole {
        self.role
    }

    /// 矩阵形状 `(rows, cols)`：登记形状是二阶张量时给出，否则 `None`。
    pub fn matrix(&self) -> Option<(usize, usize)> {
        self.matrix
    }

    /// 当前参数值（只读）。
    pub fn params(&self) -> &[f32] {
        &self.params
    }

    /// 当前参数值（可变视图由优化器步进借用）。
    pub fn params_mut(&mut self) -> &mut [f32] {
        &mut self.params
    }
}

/// 一次训练会话：复用同一张图、同一批缓冲与同一个优化器，逐 micro-batch 推进。
pub struct Session<K: Kernel, O: Optimizer, C: Collective = shard::Remote> {
    tape: Tape,
    sizes: Sizes,
    bufs: Bufs<K>,
    optimizer: O,
    distributed: Distributed<C>,
    backward: autograd::Schedule,
    slots: Vec<ParamSlot>,
    lr_schedule: Schedule,
    config: TrainConfig,
    step: u64,
    /// 当前累积窗口内已汇总的 micro-batch 数；满 `accumulation` 才步进一次。
    accum: u32,
    /// 权重绑定对：`(源槽, 回填槽)` 下标；回填时源槽参数整块复制到回填槽。
    ties: Vec<(usize, usize)>,
    losses: Vec<f32>,
    batch_rows: usize,
    batch_cols: usize,
}

impl<K: Kernel, O: Optimizer> Session<K, O> {
    /// 由图、形状表、内核、优化器与配置构造；默认单进程恒等编排，学习率调度按 warmup 与
    /// 总步数建立。
    pub fn new(
        tape: Tape,
        sizes: Sizes,
        kernel: K,
        optimizer: O,
        config: TrainConfig,
    ) -> Result<Self> {
        Self::with_distributed(tape, sizes, kernel, optimizer, config, Distributed::none())
    }
}

impl<K: Kernel, O: Optimizer, C: Collective> Session<K, O, C> {
    /// 显式装配分布式编排：拓扑、本 rank 与通信后端一起登记；单进程场景用
    /// [`Distributed::none`]。
    pub fn with_distributed(
        tape: Tape,
        sizes: Sizes,
        kernel: K,
        optimizer: O,
        config: TrainConfig,
        distributed: Distributed<C>,
    ) -> Result<Self> {
        let lr_schedule = Schedule::with(
            config.warmup_steps,
            config.max_steps,
            config.anneal,
            config.min_ratio,
        )?;
        let bufs = Bufs::new(sizes.clone(), kernel);
        Ok(Self {
            tape,
            sizes,
            bufs,
            optimizer,
            distributed,
            backward: autograd::Schedule::default(),
            slots: Vec::new(),
            lr_schedule,
            config,
            step: 0,
            accum: 0,
            ties: Vec::new(),
            losses: Vec::new(),
            batch_rows: 0,
            batch_cols: 0,
        })
    }

    /// 分布式编排（只读）。
    pub fn distributed(&self) -> &Distributed<C> {
        &self.distributed
    }

    /// 登记一个可训练参数槽：变量必须是 [`Requirement::Grad`] 终点，初始值长度与登记形状一致。
    ///
    /// # Errors
    ///
    /// 变量不是梯度终点、未登记 f32 形状、或初始值长度不符（[`Code::Invalid`]）。
    pub fn register_parameter(
        &mut self,
        var: Var,
        role: ParamRole,
        initial: Vec<f32>,
    ) -> Result<()> {
        self.require_grad_endpoint(var)?;
        let (shape, dtype) = self.shape_for(var)?;
        self.require_f32_initial(var, shape, dtype, &initial)?;
        let matrix = matrix_of(shape);
        self.bufs
            .load(var, Data::F32(initial.clone()))
            .map_err(|error| error.with(SITE))?;
        self.slots.push(ParamSlot {
            var,
            role,
            params: initial,
            matrix,
        });
        Ok(())
    }

    /// 变量必须是图上的 [`Requirement::Grad`] 终点；其余需求分类不进参数槽。
    fn require_grad_endpoint(&self, var: Var) -> Result<()> {
        match self.tape.requirement(var) {
            Some(Requirement::Grad) => Ok(()),
            Some(other) => {
                let message = format!(
                    "变量 {} 的需求是 {other:?}，不是梯度终点：不能登记为参数槽",
                    var.id()
                );
                Err(Error::new(Code::Invalid, message).with(SITE))
            }
            None => {
                let message = format!("变量 {} 不在图上", var.id());
                Err(Error::new(Code::Invalid, message).with(SITE))
            }
        }
    }

    /// 取变量在形状表里的形状与元素类型；表缺条目是图装配口径问题（[`Code::Internal`]）。
    fn shape_for(&self, var: Var) -> Result<(&Shape, DType)> {
        let Some(entry) = self.sizes.get(var) else {
            let message = format!("变量 {} 没有登记形状与元素类型", var.id());
            return Err(Error::new(Code::Internal, message).with(SITE));
        };
        Ok(entry)
    }

    /// 参数槽只收 f32；初始值长度必须与形状元素数一致。
    fn require_f32_initial(
        &self,
        var: Var,
        shape: &Shape,
        dtype: DType,
        initial: &[f32],
    ) -> Result<()> {
        if dtype != DType::F32 {
            let message = format!("变量 {} 登记为 {dtype:?}，参数槽只收 f32", var.id());
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        if initial.len() != shape.numel() {
            let message = format!(
                "初始值长度 {} 与形状 {:?}（{} 元素）不一致",
                initial.len(),
                shape.dims(),
                shape.numel()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        Ok(())
    }

    /// 反传计划：只把 loss 可达的 [`Requirement::Grad`] 终点收进端点表。
    ///
    /// 已登记的参数必须在端点表里——连不到 loss 的参数不该被优化器更新。
    ///
    /// # Errors
    ///
    /// 规划失败、或某个已登记参数不可达（[`Code::Invalid`]）。
    pub fn plan(&mut self, loss: Var) -> Result<()> {
        self.tape
            .plan_into(&mut self.backward, loss)
            .map_err(|error| error.with(SITE))?;
        for slot in &self.slots {
            if !self.backward.endpoints().contains(&slot.var()) {
                let message = format!(
                    "参数 {} 不在 loss 的可达梯度终点里：它连不到 loss，不能进优化器",
                    slot.var().id()
                );
                return Err(Error::new(Code::Invalid, message).with(SITE));
            }
        }
        Ok(())
    }

    /// 本次计划的梯度终点（按编号升序）。
    pub fn endpoints(&self) -> &[Var] {
        self.backward.endpoints()
    }

    /// 按已规划的反向计划回传梯度：从 loss 上游 `1` 出发，逐节点分配到图上输入。
    ///
    /// 自动回算把参数槽梯度填好；调用方仍可在其后用 [`Self::load_gradient`] /
    /// [`Self::accumulate_gradient`] 覆盖或追加，闭包接缝保留为覆盖口。
    ///
    /// # Errors
    ///
    /// 前向值或上游梯度不在场、账本顺序不符，或遇到尚未接反向公式的算子时透传对应错误。
    pub fn backward(&mut self) -> Result<()> {
        crate::backprop::backprop(&mut self.bufs, &self.tape, &self.backward)
    }

    /// 已登记的参数槽。
    pub fn slots(&self) -> &[ParamSlot] {
        &self.slots
    }

    /// 激活/参数缓冲（前向装载与重算走这里）。
    pub fn bufs(&mut self) -> &mut Bufs<K> {
        &mut self.bufs
    }

    /// 装入一个参数槽的当前梯度；长度与形状校验在缓冲侧完成。
    pub fn load_gradient(&mut self, var: Var, grads: &[f32]) -> Result<()> {
        self.bufs
            .load_gradient(var, grads)
            .map_err(|error| error.with(SITE))
    }

    /// 把一批微批梯度累加进窗口：窗口未满前梯度逐批相加，不覆盖上一批。
    pub fn accumulate_gradient(&mut self, var: Var, grads: &[f32]) -> Result<()> {
        self.bufs
            .accumulate_gradient(var, grads)
            .map_err(|error| error.with(SITE))
    }

    /// 当前窗口内已汇总的 micro-batch 数（未满 [`TrainConfig::accumulation`] 不步进）。
    pub fn accumulated_microbatches(&self) -> u32 {
        self.accum
    }

    /// 当前累积窗口汇总：逐参数槽的汇总梯度（未装入的槽取空表）与已汇总 micro-batch 数，
    /// 供快照导出；槽位顺序与 [`Self::slots`] 对齐，口径与 [`GradAccum`] 一致。
    pub fn accum_snapshot(&self) -> GradAccum {
        let slots = self
            .slots
            .iter()
            .map(|slot| {
                self.bufs
                    .gradient(slot.var())
                    .map(|grads| grads.to_vec())
                    .unwrap_or_default()
            })
            .collect();
        GradAccum::new(slots, self.accum)
    }

    /// 已登记的权重绑定对：`(源槽下标, 回填槽下标)`，供快照导出与回填对账。
    pub fn ties(&self) -> &[(usize, usize)] {
        &self.ties
    }

    /// 当前登记的权重绑定表快照；下标按 [`Self::slots`] 顺序对齐。
    pub fn ties_snapshot(&self) -> Ties {
        Ties::new(self.ties.clone())
    }

    /// 清空全部梯度缓冲；参数值与激活不动。
    pub fn zero_gradients(&mut self) {
        self.bufs.clear_gradients();
    }

    /// 记录一条 loss（取 f32 缓冲的均值），供趋势对账。
    pub fn record_loss(&mut self, loss: Var) -> Result<f32> {
        let view = self.bufs.value(loss).ok_or_else(|| {
            Error::new(Code::Missing, format!("loss 变量 {} 不在场", loss.id())).with(SITE)
        })?;
        let Some(values) = view.f32() else {
            return Err(Error::new(Code::Invalid, "loss 不是 f32").with(SITE));
        };
        let loss = values.iter().sum::<f32>() / values.len().max(1) as f32;
        self.losses.push(loss);
        Ok(loss)
    }

    /// 已记录的 loss 序列（每步一条）。
    pub fn losses(&self) -> &[f32] {
        &self.losses
    }

    /// 当前步数。
    pub fn step(&self) -> u64 {
        self.step
    }

    /// 配置快照。
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// 登记一批的几何：批次形状往返一致——`rows * cols` 必须等于 token 数。
    pub fn accept_batch(&mut self, rows: usize, cols: usize, token_count: usize) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Err(Error::new(Code::Invalid, "批行列不能为 0").with(SITE));
        }
        if rows.checked_mul(cols) != Some(token_count) {
            return Err(Error::new(
                Code::Invalid,
                format!("批形状 {rows} x {cols} 与 {token_count} 个 token 不符"),
            )
            .with(SITE));
        }
        self.batch_rows = rows;
        self.batch_cols = cols;
        Ok(())
    }

    /// 最近一批的行数。
    pub fn batch_rows(&self) -> usize {
        self.batch_rows
    }

    /// 最近一批的列宽。
    pub fn batch_cols(&self) -> usize {
        self.batch_cols
    }

    /// 走一步优化器：累积窗口满后先做数据并行梯度同步，再按当前学习率更新全部参数槽，
    /// 更新后镜像回激活缓冲。窗口未满只累加、不同步、不步进。
    ///
    /// # Errors
    ///
    /// 某槽梯度未装入、后端归约失败，或优化器步进校验失败（[`Code::Invalid`] /
    /// [`Code::Missing`]）。
    pub fn optimize(&mut self) -> Result<()> {
        self.accum = self.accum.saturating_add(1);
        if self.accum < self.config.accumulation {
            // 累积窗口未满：保留已汇总的梯度，等下一批 micro-batch，本批不步进。
            return Ok(());
        }
        self.accum = 0;
        self.step_once()
    }

    /// 窗口已满：汇总梯度 → 组内同步 → 优化器步进 → 镜像参数 → 清空梯度，步号 +1。
    fn step_once(&mut self) -> Result<()> {
        let lr = self.lr_schedule.lr(self.config.learning_rate, self.step);
        let mut views = Self::collect_views(&mut self.slots, &mut self.bufs)?;
        let mut grads: Vec<&mut [f32]> = views.iter_mut().map(|slot| slot.grads.as_mut()).collect();
        self.distributed
            .sync_grads(&mut grads)
            .map_err(|error| error.with(SITE))?;
        let mut slots: Vec<Slot<'_>> = views.iter_mut().map(|slot| slot.view()).collect();
        self.optimizer
            .step(&mut slots, lr)
            .map_err(|error| error.with(SITE))?;
        // 视图参数是取走时的拷贝：步进结果整块写回槽位、梯度向量归还缓冲，再镜像参数。
        Self::return_views(&mut self.slots, &mut self.bufs, views);
        Self::mirror_params(&self.slots, &mut self.bufs)?;
        self.zero_gradients();
        self.step = self.step.saturating_add(1);
        Ok(())
    }

    /// 登记权重绑定：回填时 `tied` 槽整块复制 `source` 槽的当前参数值（嵌入表与输出头共享权重的接缝）。
    pub fn register_tie(&mut self, source: Var, tied: Var) -> Result<()> {
        let source = self.slot_idx(source)?;
        let tied = self.slot_idx(tied)?;
        self.check_tie_length(source, tied)?;
        self.ties.push((source, tied));
        Ok(())
    }

    /// 按登记把源槽参数回填到绑定槽，并镜像回激活缓冲。
    pub fn backfill_ties(&mut self) -> Result<()> {
        for &(source, tied) in &self.ties {
            // 先克隆源槽再整块 copy：拆开两个可变借用；复制走 `copy_from_slice`，不逐元素循环。
            let values = self.slots[source].params().to_vec();
            self.slots[tied].params_mut().copy_from_slice(&values);
        }
        Self::mirror_params(&self.slots, &mut self.bufs)?;
        Ok(())
    }

    /// 槽下标：未登记的变量不是参数槽（[`Code::Invalid`]）。
    fn slot_idx(&self, var: Var) -> Result<usize> {
        self.slots
            .iter()
            .position(|slot| slot.var() == var)
            .ok_or_else(|| {
                Error::new(Code::Invalid, format!("变量 {} 不是已登记参数槽", var.id())).with(SITE)
            })
    }
}

/// 登记形状 → 矩阵形状：只有二阶张量是矩阵，其余（标量/向量/更高阶）一律 `None`。
fn matrix_of(shape: &Shape) -> Option<(usize, usize)> {
    match shape.dims() {
        &[rows, cols] => Some((rows, cols)),
        _ => None,
    }
}

// ---- 可恢复检查点：只记"要复现这次训练必须知道的东西" ----

/// 一条训练步记录。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    idx: u64,
    loss: f32,
}

impl Step {
    /// 由步号与损失构造。
    pub fn new(idx: u64, loss: f32) -> Self {
        Self { idx, loss }
    }

    /// 步号（从 0 开始）。
    pub fn idx(&self) -> u64 {
        self.idx
    }

    /// 该步的平均损失。
    pub fn loss(&self) -> f32 {
        self.loss
    }
}

/// 可恢复检查点：只记"要复现这次训练必须知道的东西"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Checkpoint {
    step: u64,
    tape_nodes: usize,
    batch_rows: usize,
    batch_cols: usize,
    layers: usize,
    weight_entries: usize,
    world: usize,
}

impl Checkpoint {
    /// 由当前状态构造；`layers` 为 0 说明没有可训练的层，直接报错。
    pub fn new(
        step: u64,
        tape: &autograd::Tape,
        batch: &crate::Batch<'_, u32>,
        layers: usize,
        weights: &weights::Header,
        topology: shard::Topology,
    ) -> Result<Self> {
        if layers == 0 {
            return Err(Error::new(Code::Invalid, "没有可训练的层").with("train::Checkpoint::new"));
        }
        Ok(Self {
            step,
            tape_nodes: tape.len(),
            batch_rows: batch.rows(),
            batch_cols: batch.cols(),
            layers,
            weight_entries: weights.entries().len(),
            world: topology.world(),
        })
    }

    /// 步号。
    pub fn step(&self) -> u64 {
        self.step
    }

    /// 反向图节点数的内存诊断量；不进快照、恢复时不据此对账。图结构一致性由权重条目表
    /// 与层算子契约守卫，而不是靠节点数。
    pub fn tape_nodes(&self) -> usize {
        self.tape_nodes
    }

    /// 批的行数。
    pub fn batch_rows(&self) -> usize {
        self.batch_rows
    }

    /// 批的列宽。
    pub fn batch_cols(&self) -> usize {
        self.batch_cols
    }

    /// 训练层数。
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// 权重条目数。
    pub fn weight_entries(&self) -> usize {
        self.weight_entries
    }

    /// 恢复时要求的进程数。
    pub fn world(&self) -> usize {
        self.world
    }

    /// 把这些事实与当前权重表组合成可落盘的快照。
    pub fn snapshot(&self, weights: &weights::Header) -> Result<weights::Snapshot> {
        let facts = storage::Facts::new(
            self.step,
            self.world,
            self.layers,
            self.batch_rows,
            self.batch_cols,
        )?;
        Ok(weights::Snapshot::new(facts, weights.clone()))
    }
}

/// 一个检查点能否在同一结构上恢复。
pub fn resume(
    checkpoint: &Checkpoint,
    topology: &shard::Topology,
    layers: usize,
    batch_rows: usize,
    batch_cols: usize,
) -> Result<()> {
    topology.check_world(checkpoint.world()).map_err(|error| {
        error.with(format!(
            "检查点在第 {} 步、{} 进程下保存",
            checkpoint.step(),
            checkpoint.world()
        ))
    })?;
    let ctx = format!("检查点在第 {} 步保存", checkpoint.step());
    reconcile(&ctx, "训练层数", "层", checkpoint.layers(), layers)?;
    reconcile(&ctx, "批行数", "行", checkpoint.batch_rows(), batch_rows)?;
    reconcile(&ctx, "批列宽", "列", checkpoint.batch_cols(), batch_cols)?;
    Ok(())
}

/// 一项恢复事实必须与当前运行逐项相同；口径与 [`resume`] 一致，三条对账共用同一份判据。
fn reconcile(ctx: &str, label: &str, unit: &str, saved: usize, current: usize) -> Result<()> {
    if saved == current {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!("{label}不一致：检查点记录 {saved} {unit}，当前是 {current} {unit}"),
    )
    .with(ctx))
}

/// 训练层的算子引用：`nn` 的层定义必须先通过 `nn` 注册表的闭集校验。
pub fn operators(layer: &dyn nn::Contract) -> &[String] {
    layer.spca().ops()
}

// ---- 会话快照回载 ----------------------------------------------------------
//
// 矩状态、累积窗口与绑定表按槽位对齐校验后回填。
impl<K: Kernel, O: Optimizer, C: Collective> Session<K, O, C> {
    /// 把快照状态回载到这个**已装配**会话：矩状态走优化器 restore，累积窗口汇总、绑定表与
    /// 步数按槽位编号对齐校验。
    ///
    /// 本接缝不从快照重建 tape / 参数槽 / 内核 / 权重（真实权重通路属后续里程碑）；回载前
    /// 会话必须已按同一张图、同一槽位顺序装配。
    ///
    /// # Errors
    ///
    /// 矩表/累积表槽位数或槽内长度不符、ties 编号越界或两端长度不符，一律 [`Code::Invalid`]。
    pub fn restore_snapshot(&mut self, snapshot: &Snapshot) -> Result<()> {
        self.restore_moments(snapshot.moments())?;
        self.restore_accum(snapshot.accum())?;
        self.restore_ties(snapshot.ties())?;
        self.step = snapshot.facts().step();
        Ok(())
    }

    /// 矩状态回载：空表对应无矩优化器；非空表槽位数与每槽长度必须对齐会话参数槽。
    fn restore_moments(&mut self, moments: &Moments) -> Result<()> {
        let slots = moments.slots();
        self.check_moment_slots(slots)?;
        let momentum = slots.iter().map(|s| s.m().to_vec()).collect();
        let variance = slots.iter().map(|s| s.v().to_vec()).collect();
        self.optimizer
            .restore(momentum, variance, moments.step())
            .map_err(|error| error.with(SITE))?;
        Ok(())
    }

    /// 矩表槽位数与每槽长度对账；空表（无矩优化器）跳过槽数校验。
    fn check_moment_slots(&self, slots: &[SlotMoments]) -> Result<()> {
        if !slots.is_empty() && slots.len() != self.slots.len() {
            let message = format!(
                "矩状态槽位数不符：快照 {} 槽，会话 {} 槽",
                slots.len(),
                self.slots.len()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        for (idx, slot) in slots.iter().enumerate() {
            self.check_moment_slot_length(idx, slot)?;
        }
        Ok(())
    }

    /// 单槽矩长度对账：一阶/二阶矩元素数必须等于会话槽参数元素数。
    fn check_moment_slot_length(&self, idx: usize, slot: &SlotMoments) -> Result<()> {
        let want = self.slots[idx].params().len();
        if slot.m().len() != want || slot.v().len() != want {
            let message = format!(
                "第 {idx} 槽矩长度不符：快照 {} 元素，会话 {} 元素",
                slot.m().len(),
                want
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        Ok(())
    }

    /// 累积窗口回载：槽位数对齐；空梯度表表示该槽窗口内未装入，跳过不覆盖。
    fn restore_accum(&mut self, accum: &GradAccum) -> Result<()> {
        let slots = accum.slots();
        self.check_accum_slot_count(slots)?;
        for (idx, grads) in slots.iter().enumerate() {
            self.load_accum_slot(idx, grads)?;
        }
        self.accum = accum.count();
        Ok(())
    }

    /// 累积表槽位数对账。
    fn check_accum_slot_count(&self, slots: &[Vec<f32>]) -> Result<()> {
        if slots.len() != self.slots.len() {
            let message = format!(
                "累积梯度槽位数不符：快照 {} 槽，会话 {} 槽",
                slots.len(),
                self.slots.len()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        Ok(())
    }

    /// 单槽累积梯度回载：空表跳过；非空长度必须等于槽参数元素数，再装入缓冲。
    fn load_accum_slot(&mut self, idx: usize, grads: &[f32]) -> Result<()> {
        if grads.is_empty() {
            return Ok(());
        }
        let want = self.slots[idx].params().len();
        if grads.len() != want {
            let message = format!(
                "第 {idx} 槽累积梯度长度不符：快照 {} 元素，会话 {} 元素",
                grads.len(),
                want
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        self.bufs
            .load_gradient(self.slots[idx].var(), grads)
            .map_err(|error| error.with(SITE))?;
        Ok(())
    }

    /// 校验绑定两端参数槽长度一致；不符是 [`Code::Invalid`]。
    fn check_tie_length(&self, source: usize, tied: usize) -> Result<()> {
        if self.slots[source].params().len() == self.slots[tied].params().len() {
            return Ok(());
        }
        let message = format!(
            "绑定两端长度不一致：源槽 {} 元素，回填槽 {} 元素",
            self.slots[source].params().len(),
            self.slots[tied].params().len()
        );
        Err(Error::new(Code::Invalid, message).with(SITE))
    }

    /// 绑定表回载：编号必须落在会话槽位内，两端长度一致；回载替换既有绑定表。
    fn restore_ties(&mut self, ties: &Ties) -> Result<()> {
        let slot_count = self.slots.len();
        for &(source, tied) in ties.pairs() {
            if source >= slot_count || tied >= slot_count {
                let message =
                    format!("绑定对 ({source}, {tied}) 编号越界：会话只有 {slot_count} 个参数槽");
                return Err(Error::new(Code::Invalid, message).with(SITE));
            }
            self.check_tie_length(source, tied)?;
        }
        self.ties = ties.pairs().to_vec();
        Ok(())
    }
}

// ---- 优化器视图装配 --------------------------------------------------------
//
// 参数/梯度按槽整块取走、步进后整块归还与镜像。
impl<K: Kernel, O: Optimizer, C: Collective> Session<K, O, C> {
    /// 汇总各槽的参数/梯度可变视图；任一槽梯度未装入即报错。
    fn collect_views(slots: &mut [ParamSlot], bufs: &mut Bufs<K>) -> Result<Vec<OwnedSlot>> {
        let meta: Vec<SlotMeta> = slots
            .iter()
            .map(|slot| (slot.var(), slot.role(), slot.matrix()))
            .collect();
        let mut views = Vec::with_capacity(meta.len());
        for (idx, (var, role, shape)) in meta.into_iter().enumerate() {
            let Some(grads) = bufs.take_gradient(var) else {
                // 中途归还已取走的梯度：失败路径不留悬空向量。
                Self::return_views_prefix(slots, bufs, views);
                return Err(Error::new(
                    Code::Missing,
                    format!("参数 {} 的梯度还没装入缓冲", var.id()),
                )
                .with(SITE));
            };
            let params = slots[idx].params().to_vec();
            views.push(OwnedSlot {
                params,
                grads,
                role,
                shape,
            });
        }
        Ok(views)
    }

    /// 把 [`Self::collect_views`] 取走的视图按槽号归位：更新后的参数真值回填进槽，梯度向量
    /// 归还缓冲；是否清空梯度由步进收尾决定。
    fn return_views(slots: &mut [ParamSlot], bufs: &mut Bufs<K>, views: Vec<OwnedSlot>) {
        for (slot, view) in slots.iter_mut().zip(views) {
            slot.params.copy_from_slice(&view.params);
            bufs.put_back_gradient(slot.var(), view.grads);
        }
    }

    /// 收集视图失败时的局部归还：已装配的前 `views.len()` 个槽对应 `slots` 前缀。
    fn return_views_prefix(slots: &mut [ParamSlot], bufs: &mut Bufs<K>, views: Vec<OwnedSlot>) {
        Self::return_views(&mut slots[..views.len()], bufs, views);
    }

    /// 把优化器写回后的参数值镜像回激活缓冲，供下一轮前向复用。
    fn mirror_params(slots: &[ParamSlot], bufs: &mut Bufs<K>) -> Result<()> {
        for slot in slots {
            bufs.load(slot.var(), Data::F32(slot.params.clone()))
                .map_err(|error| error.with(SITE))?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../tests/session_tests.rs"]
mod tests;

// ---- 学习率调度：warmup 升到峰值，之后按曲线降到下限比例 ----

/// 衰减曲线闭集：warmup 段对所有策略一致，差别只在 warmup 之后怎么降到学习率下限。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Anneal {
    /// 线性衰减：与峰值成比例匀速下行，只有两个参数、`lr()` 是纯函数，可解释性优先。
    #[default]
    Linear,
    /// 余弦衰减：warmup 后按 `½(1 + cos(π·p))` 平滑下行，是 LLM 训练主流的末段收敛形态。
    Cosine,
    /// 恒定：warmup 后保持峰值，下限不参与下行。
    Constant,
}

impl Anneal {
    /// 曲线短名：诊断与报告里的稳定标签。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Cosine => "cosine",
            Self::Constant => "constant",
        }
    }

    /// 归一化形状：`1.0` 为峰值、`0.0` 为谷底；谷底比例由 [`factor`](Self::factor) 统一插值。
    fn shaped(self, progress: f32) -> f32 {
        match self {
            Self::Constant => 1.0,
            Self::Linear => 1.0 - progress,
            Self::Cosine => 0.5 * (1.0 + (core::f32::consts::PI * progress).cos()),
        }
    }

    /// 曲线在进度 `progress ∈ [0, 1]` 处的因子：`min_ratio` 为谷底，`1.0` 为峰值。
    fn factor(self, progress: f32, min_ratio: f32) -> f32 {
        let progress = progress.clamp(0.0, 1.0);
        min_ratio + (1.0 - min_ratio) * self.shaped(progress)
    }
}

/// 学习率调度：warmup 升到峰值，之后按 [`Anneal`] 曲线降到下限比例。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schedule {
    warmup: u64,
    total: u64,
    anneal: Anneal,
    min_ratio: f32,
}

impl Schedule {
    /// 线性衰减到 0：`total` 必须非零，`warmup` 不得超过 `total`。
    pub fn new(warmup: u64, total: u64) -> Result<Self> {
        Self::with(warmup, total, Anneal::Linear, 0.0)
    }

    /// 全参构造；`min_ratio` 是谷底占峰值的比例，必须落在 `[0, 1]`。
    pub fn with(warmup: u64, total: u64, anneal: Anneal, min_ratio: f32) -> Result<Self> {
        if total == 0 {
            return Err(Error::new(Code::Invalid, "总步数不能为 0").with("train::Schedule::with"));
        }
        if warmup > total {
            return Err(
                Error::new(Code::Invalid, format!("warmup {warmup} 超过总步数 {total}"))
                    .with("train::Schedule::with"),
            );
        }
        if !(0.0..=1.0).contains(&min_ratio) {
            return Err(
                Error::new(Code::Invalid, format!("下限比例 {min_ratio} 不在 [0, 1]"))
                    .with("train::Schedule::with"),
            );
        }
        Ok(Self {
            warmup,
            total,
            anneal,
            min_ratio,
        })
    }

    /// warmup 步数。
    pub fn warmup(&self) -> u64 {
        self.warmup
    }

    /// 总步数。
    pub fn total(&self) -> u64 {
        self.total
    }

    /// 衰减曲线。
    pub fn anneal(&self) -> Anneal {
        self.anneal
    }

    /// 谷底占峰值的比例。
    pub fn min_ratio(&self) -> f32 {
        self.min_ratio
    }

    /// 第 `step` 步的学习率：`peak` 是 warmup 结束时的值，超过 `total` 后为 0。
    pub fn lr(&self, peak: f32, step: u64) -> f32 {
        if step >= self.total {
            return 0.0;
        }
        if self.warmup > 0 && step < self.warmup {
            // 第 0 步已经有非零学习率，否则前若干步等于白跑。
            return peak * ((step + 1) as f32) / (self.warmup as f32);
        }
        let span = self.total - self.warmup;
        if span == 0 {
            return peak;
        }
        let progress = (step - self.warmup) as f32 / span as f32;
        peak * self.anneal.factor(progress, self.min_ratio)
    }
}

// ---- 分布式训练编排：拓扑绑定 + 梯度同步后端，配置化挂到训练会话上 ----

/// 分布式编排的诊断站点（站名保持 `"train::Distributed"`，与 [`SITE`] 区分）。
const DIST_SITE: &str = "train::Distributed";

/// 分布式编排配置：拓扑、本进程 rank、通信后端与一份并行策略口径；梯度同步口径随策略走。
#[derive(Debug, Clone, PartialEq)]
pub struct Distributed<E: Collective> {
    topology: Topology,
    rank: u32,
    collective: Option<E>,
    /// 归约种类、平均开关与通信优先级；三维度数与拓扑一致。
    policy: ParallelConfig,
}

impl Distributed<Remote> {
    /// 单进程恒等编排：无拓扑、无后端，梯度不同步。后续装配多进程时用 [`Self::new`]。
    pub fn none() -> Self {
        Self {
            topology: Topology::single(),
            rank: 0,
            collective: None,
            policy: ParallelConfig::single(),
        }
    }
}

impl<E: Collective> Distributed<E> {
    /// 由拓扑、本 rank 与通信后端构造；rank 必须落在拓扑世界内，否则归约组派生时会越界。
    ///
    /// 策略口径取默认（求和、平均、普通优先级）；要改归约种类/优先级走 [`Self::with_reduce`] /
    /// [`Self::with_priority`]，或直接用 [`Self::from_config`] 传入完整策略。
    ///
    /// # Errors
    ///
    /// `rank` 超出 `0..world`（[`Code::Invalid`]），或策略度数回校验失败。
    pub fn new(topology: Topology, rank: u32, collective: E) -> Result<Self> {
        Self::ensure_rank(&topology, rank)?;
        let policy = ParallelConfig::new(topology.tp(), topology.pp(), topology.dp())?;
        Ok(Self {
            topology,
            rank,
            collective: Some(collective),
            policy,
        })
    }

    /// 由完整并行策略、本 rank 与通信后端构造；拓扑由策略度数派生。
    ///
    /// # Errors
    ///
    /// 策略度数非法（零维/上界溢出），或 `rank` 超出世界范围。
    pub fn from_config(rank: u32, collective: E, policy: ParallelConfig) -> Result<Self> {
        let topology = policy.topology()?;
        Self::ensure_rank(&topology, rank)?;
        Ok(Self {
            topology,
            rank,
            collective: Some(collective),
            policy,
        })
    }

    /// `rank` 必须落在拓扑世界内，否则归约组派生时越界。
    fn ensure_rank(topology: &Topology, rank: u32) -> Result<()> {
        if rank as usize >= topology.world() {
            return Err(Error::new(
                Code::Invalid,
                format!("rank {rank} 超出拓扑世界 0..{}", topology.world()),
            )
            .with(DIST_SITE));
        }
        Ok(())
    }

    /// 归约后取平均（默认）；`false` 保留求和梯度。
    pub fn with_average(mut self, average: bool) -> Self {
        self.policy = self.policy.with_average(average);
        self
    }

    /// 覆盖组内归约种类。
    pub fn with_reduce(mut self, reduce: Reduce) -> Self {
        self.policy = self.policy.with_reduce(reduce);
        self
    }

    /// 覆盖通信调度优先级。
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.policy = self.policy.with_priority(priority);
        self
    }

    /// 并行策略只读投影。
    pub fn policy(&self) -> ParallelConfig {
        self.policy
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
        let idx = self.topology.idxs(self.rank)?.2;
        self.topology.group(Parallel::Data, idx)
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
        synchronize(
            collective,
            &group,
            self.policy.reduce(),
            self.policy.average(),
            slots,
        )?;
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
    for grads in slots.iter_mut() {
        collective.all_reduce(group, reduce, grads)?;
    }
    if average {
        scale(slots, group)?;
    }
    Ok(())
}

/// 按组大小缩小平坦后的梯度；组大小为 0 是构造错误，不是可降级情形。
///
/// 各槽元素互相独立、无串行依赖；但安全 Rust 边界内没有可调用的整块浮点向量化原语，
/// 故保留逐元素除法，不引入 rayon/unsafe SIMD。
fn scale(slots: &mut [&mut [f32]], group: &Group) -> Result<()> {
    let divisor = group.size() as f32;
    if divisor == 0.0 {
        return Err(Error::new(Code::Internal, "数据并行组大小为 0").with(DIST_SITE));
    }
    for grads in slots.iter_mut() {
        for slot in grads.iter_mut() {
            *slot /= divisor;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../tests/dist_tests.rs"]
mod dist_tests;

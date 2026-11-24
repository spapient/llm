//! 会话快照与视图装配：可恢复检查点的最小事实集、快照回载校验、优化器视图的取走/归还/镜像。

use super::{ParamSlot, SITE, Session, SlotMeta};
use crate::optimizer::{Optimizer, OwnedSlot};
use crate::pool::Bufs;
use crate::Kernel;
use autograd::Var;
use diagnostic::{Code, Error, Result};
use shard::Collective;
use storage::{GradAccum, Moments, SlotMoments, Ties};
use weights::Snapshot;

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

/// 独立梯度快照：按图变量自持一份梯度向量副本，脱离优化器累积窗口与缓冲生命周期。
///
/// 与 [`storage::GradAccum`] 的差别：`GradAccum` 按参数槽对齐、带 micro-batch 计数，
/// 服务窗口内汇总；本快照按 [`Var`] 取读、整块复制，不随 `zero_gradients` 或优化器步进失效。
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GradientSnapshot {
    entries: Vec<(Var, Vec<f32>)>,
}

impl GradientSnapshot {
    /// 由 `(变量, 梯度向量)` 条目构造；条目顺序沿用传入顺序。
    pub fn new(entries: Vec<(Var, Vec<f32>)>) -> Self {
        Self { entries }
    }

    /// 全部条目：变量与其梯度向量。
    #[must_use]
    pub fn entries(&self) -> &[(Var, Vec<f32>)] {
        &self.entries
    }

    /// 取一个变量的梯度向量；快照里没有该变量时为 `None`。
    #[must_use]
    pub fn get(&self, var: Var) -> Option<&[f32]> {
        self.entries
            .iter()
            .find(|(held, _)| *held == var)
            .map(|(_, grad)| grad.as_slice())
    }

    /// 快照条目数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否没有任何变量带梯度。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
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
    pub(super) fn check_tie_length(&self, source: usize, tied: usize) -> Result<()> {
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
    pub(super) fn collect_views(
        slots: &mut [ParamSlot],
        bufs: &mut Bufs<K>,
    ) -> Result<Vec<OwnedSlot>> {
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
            // 参数向量随视图整块移交；同步/步进后由原所有者归还，不再为每次更新复制一份。
            let params = std::mem::take(&mut slots[idx].params);
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
    pub(super) fn return_views(slots: &mut [ParamSlot], bufs: &mut Bufs<K>, views: Vec<OwnedSlot>) {
        for (slot, view) in slots.iter_mut().zip(views) {
            slot.params = view.params;
            bufs.put_back_gradient(slot.var(), view.grads);
        }
    }

    /// 收集视图失败时的局部归还：已装配的前 `views.len()` 个槽对应 `slots` 前缀。
    pub(super) fn return_views_prefix(
        slots: &mut [ParamSlot],
        bufs: &mut Bufs<K>,
        views: Vec<OwnedSlot>,
    ) {
        Self::return_views(&mut slots[..views.len()], bufs, views);
    }

    /// 把优化器写回后的参数值镜像回激活缓冲，供下一轮前向复用。
    pub(super) fn mirror_params(slots: &[ParamSlot], bufs: &mut Bufs<K>) -> Result<()> {
        for slot in slots {
            bufs.overwrite_f32(slot.var(), &slot.params)
                .map_err(|error| error.with(SITE))?;
        }
        Ok(())
    }
}

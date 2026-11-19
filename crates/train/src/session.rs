//! 训练会话：tape → 反传计划 → 梯度需求分类 → 梯度缓冲 → 优化器步的最小闭环。
//!
//! **拥有**：[`TrainConfig`]、[`ParamSlot`]、[`Session`]。**不拥有**：前向算子语义（`op`）、
//! 反向公式（autograd 只出 tape / plan / Replay）、分词与打包策略（`data`）。**依赖**：`autograd`、
//! `data`、`nn`、`diagnostic`、本层的 [`Buffers`](crate::Buffers) 与 [`Optimizer`](crate::Optimizer)。
//!
//! 职责收口：
//!
//! - 参数槽按 [`ParamRole`] 登记，角色从层声明派生，优化器不按字段名猜组；
//! - 梯度缓冲由 [`Buffers`](crate::Buffers) 扩展自持，autograd 不持梯度存储；
//! - 反传计划只取 [`Requirement::Grad`](autograd::Requirement::Grad) 终点；不可达参数不进优化器；
//! - 学习率走本层 [`Schedule`](crate::Schedule)，会话只在每步取当前值。

use autograd::{Requirement, Sizes, Tape, Var};
use diagnostic::{Code, Error, Result};
use nn::ParamRole;
use shard::Collective;
use tensor::{DType, Shape};

use crate::dist::Distributed;
use crate::optimizer::{Optimizer, OwnedSlot, Slot};
use crate::pool::Buffers;
use crate::schedule::Schedule as LrSchedule;
use crate::snapshot::{GradAccum, Moments, SlotMoments, Snapshot, Ties};
use crate::{Data, Kernel};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::Session";

/// 训练配置：学习率峰值、warmup、总步数、裁剪与梯度累积窗口。
///
/// 裁剪与累积窗口先收下口径；裁剪在优化器内执行，累积汇总见 T1-5。
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
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            warmup_steps: 0,
            max_steps: 1,
            clip_norm: None,
            accumulation: 1,
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
        })
    }

    /// 按可选覆盖构造：未给的字段取 [`Default`]。
    ///
    /// 默认值只在 [`Default`] 一份；调用方把缺项留成 `None` 透传，不复制第二份常数。校验仍走
    /// [`Self::new`]。
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

    /// 梯度累积窗口。
    pub fn accumulation(&self) -> u32 {
        self.accumulation
    }
}

/// 一个已登记的参数槽：变量编号、角色与常驻参数值。
///
/// 梯度不在这里，梯度缓冲由 [`Buffers`](crate::Buffers) 自持；参数值是优化器就地写回的对象，
/// 每步前向再镜像回激活缓冲。
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSlot {
    var: Var,
    role: ParamRole,
    params: Vec<f32>,
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
///
/// `C` 是分布式通信后端（[`Collective`]）；单进程恒等回路用 [`Distributed::none`] 占位。
pub struct Session<K: Kernel, O: Optimizer, C: Collective = shard::Remote> {
    tape: Tape,
    sizes: Sizes,
    buffers: Buffers<K>,
    optimizer: O,
    distributed: Distributed<C>,
    backward: autograd::Schedule,
    slots: Vec<ParamSlot>,
    lr_schedule: LrSchedule,
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
        let lr_schedule = LrSchedule::with(
            config.warmup_steps,
            config.max_steps,
            crate::Anneal::default(),
            0.0,
        )?;
        let buffers = Buffers::new(sizes.clone(), kernel);
        Ok(Self {
            tape,
            sizes,
            buffers,
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

    /// 分布式编排（可变：替换后端、调整归约口径走这里）。
    pub fn distributed_mut(&mut self) -> &mut Distributed<C> {
        &mut self.distributed
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
        self.buffers
            .load(var, Data::F32(initial.clone()))
            .map_err(|error| error.with(SITE))?;
        self.slots.push(ParamSlot {
            var,
            role,
            params: initial,
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
        crate::backprop::backprop(&mut self.buffers, &self.tape, &self.backward)
    }

    /// 已登记的参数槽。
    pub fn slots(&self) -> &[ParamSlot] {
        &self.slots
    }

    /// 激活/参数缓冲（前向装载与重算走这里）。
    pub fn buffers(&mut self) -> &mut Buffers<K> {
        &mut self.buffers
    }

    /// 装入一个参数槽的当前梯度；长度与形状校验在缓冲侧完成。
    pub fn load_gradient(&mut self, var: Var, grads: &[f32]) -> Result<()> {
        self.buffers
            .load_gradient(var, grads)
            .map_err(|error| error.with(SITE))
    }

    /// 把一批微批梯度累加进窗口：窗口未满前梯度逐批相加，不覆盖上一批。
    pub fn accumulate_gradient(&mut self, var: Var, grads: &[f32]) -> Result<()> {
        self.buffers
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
                self.buffers
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
        for (index, slot) in slots.iter().enumerate() {
            self.check_moment_slot_length(index, slot)?;
        }
        Ok(())
    }

    /// 单槽矩长度对账：一阶/二阶矩元素数必须等于会话槽参数元素数。
    fn check_moment_slot_length(&self, index: usize, slot: &SlotMoments) -> Result<()> {
        let want = self.slots[index].params().len();
        if slot.m().len() != want || slot.v().len() != want {
            let message = format!(
                "第 {index} 槽矩长度不符：快照 {} 元素，会话 {} 元素",
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
        for (index, grads) in slots.iter().enumerate() {
            self.load_accum_slot(index, grads)?;
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
    fn load_accum_slot(&mut self, index: usize, grads: &[f32]) -> Result<()> {
        if grads.is_empty() {
            return Ok(());
        }
        let want = self.slots[index].params().len();
        if grads.len() != want {
            let message = format!(
                "第 {index} 槽累积梯度长度不符：快照 {} 元素，会话 {} 元素",
                grads.len(),
                want
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
        self.buffers
            .load_gradient(self.slots[index].var(), grads)
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

    /// 清空全部梯度缓冲；参数值与激活不动。
    pub fn zero_gradients(&mut self) {
        self.buffers.clear_gradients();
    }

    /// 记录一条 loss（取 f32 缓冲的均值），供趋势对账。
    pub fn record_loss(&mut self, loss: Var) -> Result<f32> {
        let view = self.buffers.value(loss).ok_or_else(|| {
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
        let mut views = Self::collect_views(&mut self.slots, &mut self.buffers)?;
        let mut grads: Vec<&mut [f32]> = views.iter_mut().map(|slot| slot.grads.as_mut()).collect();
        self.distributed
            .sync_grads(&mut grads)
            .map_err(|error| error.with(SITE))?;
        let mut slots: Vec<Slot<'_>> = views.iter_mut().map(|slot| slot.view()).collect();
        self.optimizer
            .step(&mut slots, lr)
            .map_err(|error| error.with(SITE))?;
        // 视图参数是取走时的拷贝：步进结果整块写回槽位、梯度向量归还缓冲，再镜像参数。
        Self::return_views(&mut self.slots, &mut self.buffers, views);
        Self::mirror_params(&self.slots, &mut self.buffers)?;
        self.zero_gradients();
        self.step = self.step.saturating_add(1);
        Ok(())
    }

    /// 登记权重绑定：回填时 `tied` 槽整块复制 `source` 槽的当前参数值（嵌入表与输出头共享权重的接缝）。
    ///
    /// 两端都必须是已登记参数槽，且长度一致；长度不符是 [`Code::Invalid`]。
    pub fn register_tie(&mut self, source: Var, tied: Var) -> Result<()> {
        let source = self.slot_index(source)?;
        let tied = self.slot_index(tied)?;
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
        Self::mirror_params(&self.slots, &mut self.buffers)?;
        Ok(())
    }

    /// 槽下标：未登记的变量不是参数槽（[`Code::Invalid`]）。
    fn slot_index(&self, var: Var) -> Result<usize> {
        self.slots
            .iter()
            .position(|slot| slot.var() == var)
            .ok_or_else(|| {
                Error::new(Code::Invalid, format!("变量 {} 不是已登记参数槽", var.id())).with(SITE)
            })
    }

    /// 汇总各槽的参数/梯度可变视图；任一槽梯度未装入即报错。
    ///
    /// 梯度向量按槽整块取走（[`Buffers::take_gradient`]），步进结束后在
    /// [`Self::return_views`] 归还；参数槽与梯度表因此各只建立一条可变借用。
    fn collect_views(slots: &mut [ParamSlot], buffers: &mut Buffers<K>) -> Result<Vec<OwnedSlot>> {
        let meta: Vec<(Var, ParamRole)> = slots.iter().map(|s| (s.var(), s.role())).collect();
        let mut views = Vec::with_capacity(meta.len());
        for (index, (var, role)) in meta.into_iter().enumerate() {
            let Some(grads) = buffers.take_gradient(var) else {
                // 中途归还已取走的梯度：失败路径不留悬空向量。
                Self::return_views_prefix(slots, buffers, views);
                return Err(Error::new(
                    Code::Missing,
                    format!("参数 {} 的梯度还没装入缓冲", var.id()),
                )
                .with(SITE));
            };
            let params = slots[index].params().to_vec();
            views.push(OwnedSlot {
                params,
                grads,
                role,
            });
        }
        Ok(views)
    }

    /// 把 [`Self::collect_views`] 取走的视图按槽号归位：更新后的参数真值回填进槽，梯度向量
    /// 归还缓冲；是否清空梯度由步进收尾决定。
    fn return_views(slots: &mut [ParamSlot], buffers: &mut Buffers<K>, views: Vec<OwnedSlot>) {
        for (slot, view) in slots.iter_mut().zip(views) {
            slot.params.copy_from_slice(&view.params);
            buffers.put_back_gradient(slot.var(), view.grads);
        }
    }

    /// 收集视图失败时的局部归还：已装配的前 `views.len()` 个槽对应 `slots` 前缀。
    fn return_views_prefix(
        slots: &mut [ParamSlot],
        buffers: &mut Buffers<K>,
        views: Vec<OwnedSlot>,
    ) {
        Self::return_views(&mut slots[..views.len()], buffers, views);
    }

    /// 把优化器写回后的参数值镜像回激活缓冲，供下一轮前向复用。
    fn mirror_params(slots: &[ParamSlot], buffers: &mut Buffers<K>) -> Result<()> {
        for slot in slots {
            buffers
                .load(slot.var(), Data::F32(slot.params.clone()))
                .map_err(|error| error.with(SITE))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
use super::*;

use crate::snapshot::{Facts, GradAccum, Moments, SlotMoments, Snapshot, Ties};
use crate::{Data, Reference, Sgd};
use autograd::{Sizes, Store, Tape, Var};
use data::{Chunks, Dataset};
use diagnostic::{Code, Result};
use nn::ParamRole;
use tensor::{DType, Shape};
use weights::Header;

/// 夹具向量宽度。
const N: usize = 4;

/// `loss = add_f32(p, x)`：`p` 是梯度终点参数，`x` 是常量输入叶子。
fn fixture() -> (Tape, Sizes, Var, Var, Var) {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let x = tape.leaf();
    let loss = tape.record("add_f32", &[p, x]).expect("记录");
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([N]), DType::F32);
    (tape, sizes, p, x, loss)
}

/// 两层共享权重夹具：`h = add(x, w_source)`，`t = add(h, w_tied)`，`loss = add(t, u)`；
/// `w_source` / `w_tied` 待注册权重绑定，`u` 是未绑定参数，三层都可达 loss。
#[allow(clippy::type_complexity)]
fn two_layer_fixture() -> (Tape, Sizes, Var, Var, Var, Var, Var, Var, Var) {
    let mut tape = Tape::new();
    let w_source = tape.parameter();
    let w_tied = tape.parameter();
    let u = tape.parameter();
    let x = tape.leaf();
    let h = tape.record("add_f32", &[x, w_source]).expect("第一层");
    let t = tape.record("add_f32", &[h, w_tied]).expect("第二层");
    let loss = tape.record("add_f32", &[t, u]).expect("loss");
    let mut sizes = Sizes::new(tape.vars());
    for var in [w_source, w_tied, u, x, h, t, loss] {
        sizes.set(var, Shape::new([N]), DType::F32);
    }
    (tape, sizes, w_source, w_tied, u, x, h, t, loss)
}

/// 装配一个裸会话：优化器用默认 SGD，配置由用例给定。
fn open_session(tape: Tape, sizes: Sizes, config: TrainConfig) -> Session<Reference, Sgd> {
    Session::new(tape, sizes, Reference, Sgd::new(), config).expect("会话")
}

/// 接收一批 token、装输入、前向到终点并登记计划；批形状固定 1 × N。
fn forward_fixture(
    session: &mut Session<Reference, Sgd>,
    tokens: &[u32],
    op_inputs: &[Var],
    x: Var,
    loss: Var,
) -> Result<()> {
    session.accept_batch(1, N, tokens.len())?;
    let inputs: Vec<f32> = tokens.iter().map(|token| *token as f32 * 0.1).collect();
    session.buffers().load(x, Data::F32(inputs))?;
    session.buffers().compute("add_f32", op_inputs, loss)?;
    session.plan(loss)?;
    Ok(())
}

/// 登记两层夹具的三个参数并绑定源/回填槽；初值沿用夹具口径。
fn register_two_layer(
    session: &mut Session<Reference, Sgd>,
    w_source: Var,
    w_tied: Var,
    u: Var,
) -> Result<()> {
    session.register_parameter(w_source, ParamRole::Trainable, vec![1.0; N])?;
    session.register_parameter(w_tied, ParamRole::Trainable, vec![0.0; N])?;
    session.register_parameter(u, ParamRole::Trainable, vec![2.0; N])?;
    session.register_tie(w_source, w_tied)?;
    Ok(())
}

/// 两层夹具前向：常量输入 → 第一层 → 第二层 → 未绑定参数终点，并登记计划终点。
#[allow(clippy::too_many_arguments)]
fn forward_two_layer(
    session: &mut Session<Reference, Sgd>,
    x: Var,
    w_source: Var,
    h: Var,
    w_tied: Var,
    t: Var,
    u: Var,
    loss: Var,
) -> Result<()> {
    session.buffers().load(x, Data::F32(vec![0.1; N]))?;
    session.buffers().compute("add_f32", &[x, w_source], h)?;
    session.buffers().compute("add_f32", &[h, w_tied], t)?;
    session.buffers().compute("add_f32", &[t, u], loss)?;
    session.plan(loss)?;
    Ok(())
}

/// 夹具语料按批跑通指定步数：前向 → 计划分类终点 → 梯度缓冲 → 优化器步，loss 逐步入账。
#[test]
fn a_fixture_dataset_runs_the_declared_steps() {
    let (tape, sizes, p, x, loss) = fixture();
    let tokens: Vec<u32> = (0..12).collect();
    let mut chunks = Chunks::new(&tokens, 1, N).expect("批 1 x 4");
    let config = TrainConfig::new(1e-2, 0, 3, None, 1).expect("配置合法");
    let mut session = open_session(tape, sizes, config);
    session
        .register_parameter(p, ParamRole::Trainable, vec![1.0; N])
        .expect("登记参数");

    let mut steps = 0;
    while let Some(batch) = chunks.next_batch() {
        forward_fixture(&mut session, batch.tokens(), &[p, x], x, loss).expect("前向");
        session.record_loss(loss).expect("loss 在场");
        session.zero_gradients();
        session.load_gradient(p, &[1.0; N]).expect("梯度就位");
        session.optimize().expect("步进");
        steps += 1;
    }

    assert_eq!(steps, 3);
    assert_eq!(session.losses().len(), 3, "每步一条 loss");
    assert_eq!((session.batch_rows(), session.batch_cols()), (1, N));
    assert!(session.endpoints().contains(&p), "参数是可达梯度终点");
    assert!(session.slots()[0].params()[0] < 1.0, "参数沿梯度方向被更新");
}

/// 梯度累积：窗口未满前逐批汇总梯度、不步进；满窗后按汇总梯度走一次优化器。
#[test]
fn gradients_accumulate_over_the_window_before_one_step() {
    let (tape, sizes, p, x, loss) = fixture();
    let tokens: Vec<u32> = (0..8).collect();
    let mut chunks = Chunks::new(&tokens, 1, N).expect("批 1 x 4");
    let config = TrainConfig::new(1e-1, 0, 2, None, 2).expect("累积窗口 2");
    let mut session = open_session(tape, sizes, config);
    session
        .register_parameter(p, ParamRole::Trainable, vec![1.0; N])
        .expect("登记参数");

    // 第一批：梯度 1.0/元素累加进窗口；未满 2，参数不动。
    let batch = chunks.next_batch().expect("第一批");
    forward_fixture(&mut session, batch.tokens(), &[p, x], x, loss).expect("前向");
    session
        .accumulate_gradient(p, &[1.0; N])
        .expect("累加第一批");
    session.optimize().expect("第一批不步进");
    assert_eq!(session.accumulated_microbatches(), 1);
    assert_eq!(session.slots()[0].params(), &[1.0; N], "窗口未满不更新参数");

    // 第二批：再累加 1.0，汇总梯度 2.0/元素；窗口满，步进一次。
    let batch = chunks.next_batch().expect("第二批");
    forward_fixture(&mut session, batch.tokens(), &[p, x], x, loss).expect("前向");
    session
        .accumulate_gradient(p, &[1.0; N])
        .expect("累加第二批");
    session.optimize().expect("窗口满步进");
    assert_eq!(session.accumulated_microbatches(), 0);
    assert_eq!(session.step(), 1);
    // SGD lr 0.1，汇总梯度 2.0：1.0 − 0.1·2.0 = 0.8。
    assert!((session.slots()[0].params()[0] - 0.8).abs() < 1e-6);
}

/// 权重绑定：回填时源槽参数整块复制到绑定槽；未登记变量不能绑定。
#[test]
fn tied_parameter_copies_source_on_backfill() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let q = tape.parameter();
    let loss = tape.record("add_f32", &[p, q])?;
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(q, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([N]), DType::F32);
    let config = TrainConfig::default();
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config)?;
    session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
    session.register_parameter(q, ParamRole::Trainable, vec![0.0; N])?;
    session.register_tie(p, q)?;

    session.backfill_ties()?;
    assert_eq!(session.slots()[1].params(), &[1.0; N], "回填槽取源槽整块值");

    let error = session.register_tie(loss, p).expect_err("loss 不是参数槽");
    assert_eq!(error.code(), Code::Invalid);
    Ok(())
}

/// 两层真实装配夹具：源/回填槽经若干 micro-batch 各自更新分化后，回填把 tied 整块复制成
/// source；未绑定参数不被改动。
#[test]
fn tied_parameter_copies_source_across_a_two_layer_fixture() -> Result<()> {
    let (tape, sizes, w_source, w_tied, u, x, h, t, loss) = two_layer_fixture();
    let config = TrainConfig::new(1e-2, 0, 2, None, 1)?;
    let mut session = open_session(tape, sizes, config);
    register_two_layer(&mut session, w_source, w_tied, u)?;

    // 两个 micro-batch 各走一步：源/回填槽按自身梯度分化，u 梯度为 0 保持不动。
    for (source_grad, tied_grad) in [(1.0_f32, 2.0_f32), (0.5, 2.0)] {
        forward_two_layer(&mut session, x, w_source, h, w_tied, t, u, loss)?;
        session.accumulate_gradient(w_source, &[source_grad; N])?;
        session.accumulate_gradient(w_tied, &[tied_grad; N])?;
        session.accumulate_gradient(u, &[0.0; N])?;
        session.optimize()?;
    }

    let source_after_step = session.slots()[0].params().to_vec();
    let u_before = session.slots()[2].params().to_vec();
    session.backfill_ties()?;
    assert_eq!(
        session.slots()[1].params(),
        source_after_step,
        "回填槽逐元素取源槽值"
    );
    assert_eq!(
        session.slots()[2].params(),
        u_before,
        "未绑定参数不被回填改动"
    );
    Ok(())
}

/// 会话窗口状态入快照：多次累积后的梯度总和、micro-batch 计数与绑定表写进快照再读回，
/// 逐字段相等；空累积窗口路径由快照侧用例覆盖。
#[test]
fn session_window_state_round_trips_through_snapshot() -> Result<()> {
    let (tape, sizes, w_source, w_tied, u, x, h, t, loss) = two_layer_fixture();
    let config = TrainConfig::new(1e-2, 0, 4, None, 4)?;
    let mut session = open_session(tape, sizes, config);
    register_two_layer(&mut session, w_source, w_tied, u)?;

    // 三个 micro-batch 累积进同一窗口（窗口 4，未满不步进）。
    let source_batches = [1.0_f32, 0.5, 0.25];
    let tied_batches = [2.0_f32, 1.0, 0.5];
    for batch in 0..3 {
        forward_two_layer(&mut session, x, w_source, h, w_tied, t, u, loss)?;
        session.accumulate_gradient(w_source, &[source_batches[batch]; N])?;
        session.accumulate_gradient(w_tied, &[tied_batches[batch]; N])?;
        session.accumulate_gradient(u, &[0.0; N])?;
        session.optimize()?;
    }
    assert_eq!(session.accumulated_microbatches(), 3);

    let facts = Facts::new(session.step(), 1, 2, 1, N)?;
    let snapshot = Snapshot::new_with_all(
        facts,
        Header::new(Vec::new(), 0)?,
        Moments::empty(),
        session.accum_snapshot(),
        session.ties_snapshot(),
    );
    let restored = Snapshot::decode(&snapshot.encode()?)?;

    assert_eq!(restored.accum().count(), 3, "窗口批数往返一致");
    assert_eq!(restored.ties().pairs(), &[(0, 1)], "绑定关系往返一致");
    assert_eq!(restored.accum(), snapshot.accum(), "累积梯度逐槽往返一致");
    assert_eq!(restored.ties(), snapshot.ties(), "绑定表往返一致");

    // 显式对拍三批总和：源槽 1.75/元素，回填槽 3.5/元素，未绑定槽零梯度。
    let slots = restored.accum().slots();
    assert_eq!(slots.len(), 3);
    assert!(slots[0].iter().all(|g| (g - 1.75).abs() < 1e-6));
    assert!(slots[1].iter().all(|g| (g - 3.5).abs() < 1e-6));
    assert!(slots[2].iter().all(|g| *g == 0.0));
    Ok(())
}

/// 会话状态快照回载：源会话窗口汇总/绑定表/步号写进快照，回载到另一个同装配会话后
/// accum / ties / 步号逐字段一致；矩状态走优化器 restore（SGD 对应空表）。
#[test]
fn session_state_round_trips_through_restore_snapshot() -> Result<()> {
    let (tape, sizes, w_source, w_tied, u, x, h, t, loss) = two_layer_fixture();
    let config = TrainConfig::new(1e-2, 0, 4, None, 4)?;
    let mut source = open_session(tape, sizes, config);
    register_two_layer(&mut source, w_source, w_tied, u)?;

    // 三个 micro-batch 累积进同一窗口（窗口 4，未满不步进）。
    let source_batches = [1.0_f32, 0.5, 0.25];
    let tied_batches = [2.0_f32, 1.0, 0.5];
    for batch in 0..3 {
        forward_two_layer(&mut source, x, w_source, h, w_tied, t, u, loss)?;
        source.accumulate_gradient(w_source, &[source_batches[batch]; N])?;
        source.accumulate_gradient(w_tied, &[tied_batches[batch]; N])?;
        source.accumulate_gradient(u, &[0.0; N])?;
        source.optimize()?;
    }

    let facts = Facts::new(5, 1, 2, 1, N)?;
    let snapshot = Snapshot::new_with_all(
        facts,
        Header::new(Vec::new(), 0)?,
        Moments::empty(),
        source.accum_snapshot(),
        source.ties_snapshot(),
    );

    // 目标会话：同图同槽位顺序装配，初始窗口为空；回载不重建 tape / 槽位 / 权重。
    let (tape, sizes, w_source, w_tied, u, _x, _h, _t, _loss) = two_layer_fixture();
    let mut target = open_session(tape, sizes, TrainConfig::new(1e-2, 0, 4, None, 4)?);
    register_two_layer(&mut target, w_source, w_tied, u)?;
    target.restore_snapshot(&snapshot)?;

    assert_eq!(target.step(), 5, "步号回载一致");
    assert_eq!(
        target.accumulated_microbatches(),
        source.accumulated_microbatches(),
        "窗口批数回载一致"
    );
    assert_eq!(
        target.accum_snapshot(),
        *snapshot.accum(),
        "累积窗口逐槽回载"
    );
    assert_eq!(target.ties(), snapshot.ties().pairs(), "绑定表回载一致");
    Ok(())
}

/// 回载对账：累积槽位数错位、槽长度不符、ties 编号越界、矩槽位数/长度错位，一律
/// [`Code::Invalid`]，拒绝前不改写会话状态。
#[test]
fn restore_snapshot_rejects_mismatched_state() -> Result<()> {
    let (tape, sizes, w_source, w_tied, u, _x, _h, _t, _loss) = two_layer_fixture();
    let mut target = open_session(tape, sizes, TrainConfig::default());
    register_two_layer(&mut target, w_source, w_tied, u)?;

    let facts = Facts::new(0, 1, 2, 1, N)?;
    let header = Header::new(Vec::new(), 0)?;

    // 槽位数错位：累积表只有 2 槽，会话 3 槽。
    let fewer = GradAccum::new(vec![vec![1.0; N], vec![2.0; N]], 1);
    let snapshot = Snapshot::new_with_all(
        facts,
        header.clone(),
        Moments::empty(),
        fewer,
        Ties::empty(),
    );
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("累积槽位数不符");
    assert_eq!(error.code(), Code::Invalid);

    // 槽长度不符：第三槽梯度长 N − 1。
    let wrong_len = GradAccum::new(vec![vec![1.0; N], vec![2.0; N], vec![3.0; N - 1]], 1);
    let snapshot = Snapshot::new_with_all(
        facts,
        header.clone(),
        Moments::empty(),
        wrong_len,
        Ties::empty(),
    );
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("累积槽长度不符");
    assert_eq!(error.code(), Code::Invalid);

    // ties 编号越界：回填槽 5 超出 3 槽。
    let snapshot = Snapshot::new_with_all(
        facts,
        header.clone(),
        Moments::empty(),
        GradAccum::empty(),
        Ties::new(vec![(0, 5)]),
    );
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("ties 编号越界");
    assert_eq!(error.code(), Code::Invalid);

    // 矩表槽长度不符：第一槽矩向量长 N − 1，会话槽参数长 N。
    let moments = Moments::new(
        vec![
            SlotMoments::new(vec![0.0; N - 1], vec![0.0; N - 1])?,
            SlotMoments::new(vec![0.0; N], vec![0.0; N])?,
            SlotMoments::new(vec![0.0; N], vec![0.0; N])?,
        ],
        1,
    );
    let snapshot = Snapshot::new_with_all(
        facts,
        header.clone(),
        moments,
        GradAccum::empty(),
        Ties::empty(),
    );
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("矩槽长度不符");
    assert_eq!(error.code(), Code::Invalid);

    // 矩表槽位数错位：非空矩表只 1 槽，会话 3 槽。
    let moments = Moments::new(vec![SlotMoments::new(vec![0.0; N], vec![0.0; N])?], 1);
    let snapshot =
        Snapshot::new_with_all(facts, header, moments, GradAccum::empty(), Ties::empty());
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("矩槽位数不符");
    assert_eq!(error.code(), Code::Invalid);
    Ok(())
}

/// 非梯度终点的叶子不能登记为参数槽。
#[test]
fn a_non_endpoint_leaf_is_rejected_as_a_parameter() {
    let mut tape = Tape::new();
    let x = tape.leaf();
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([N]), DType::F32);
    let config = TrainConfig::default();
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config).expect("会话");

    let error = session
        .register_parameter(x, ParamRole::Trainable, vec![0.0; N])
        .expect_err("叶子不是终点");
    assert_eq!(error.code(), Code::Invalid);
}

/// 连不到 loss 的参数不进优化器：计划对账时拒绝。
#[test]
fn an_unreachable_parameter_is_rejected_at_plan_time() {
    let mut tape = Tape::new();
    let orphan = tape.parameter();
    let used = tape.parameter();
    let x = tape.leaf();
    let loss = tape.record("add_f32", &[used, x]).expect("记录");
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(orphan, Shape::new([N]), DType::F32);
    sizes.set(used, Shape::new([N]), DType::F32);
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(loss, Shape::new([N]), DType::F32);
    let config = TrainConfig::default();
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config).expect("会话");
    session
        .register_parameter(orphan, ParamRole::Trainable, vec![1.0; N])
        .expect("登记");
    session
        .register_parameter(used, ParamRole::Trainable, vec![1.0; N])
        .expect("登记");

    let error = session.plan(loss).expect_err("参数不可达");
    assert_eq!(error.code(), Code::Invalid);
}

/// 批形状与 token 数不符时拒绝，不把错位批次带进循环。
#[test]
fn a_mismatched_batch_shape_is_rejected() {
    let (tape, sizes, p, _x, _loss) = fixture();
    let config = TrainConfig::default();
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config).expect("会话");
    session
        .register_parameter(p, ParamRole::Trainable, vec![1.0; N])
        .expect("登记");

    let error = session.accept_batch(2, N, 7).expect_err("2 x 4 ≠ 7");
    assert_eq!(error.code(), Code::Invalid);
}

/// 梯度没装入就步进：缺梯度是一条 Missing，不做半截更新。
#[test]
fn stepping_without_a_gradient_is_rejected() -> Result<()> {
    let (tape, sizes, p, x, loss) = fixture();
    let config = TrainConfig::default();
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config)?;
    session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
    session.buffers().load(x, Data::F32(vec![0.0; N]))?;
    session.buffers().compute("add_f32", &[p, x], loss)?;
    session.plan(loss)?;

    let error = session.optimize().expect_err("梯度未装入");
    assert_eq!(error.code(), Code::Missing);
    assert_eq!(session.slots()[0].params(), &[1.0; N], "拒绝前不改写参数");
    Ok(())
}

/// 可选覆盖：缺项取默认值、给出的字段生效；非法值仍走 [`TrainConfig::new`] 校验。
#[test]
fn overrides_fall_back_to_default_without_duplicating_constants() {
    let base = TrainConfig::default();

    let all_default =
        TrainConfig::with_overrides(None, None, None, None, None).expect("缺省配置合法");
    assert_eq!(all_default, base, "全缺省应与默认配置逐字段一致");

    let overridden = TrainConfig::with_overrides(Some(2e-3), Some(2), Some(5), Some(0.5), Some(2))
        .expect("部分覆盖合法");
    assert_eq!(overridden.learning_rate, 2e-3);
    assert_eq!(overridden.warmup_steps, 2);
    assert_eq!(overridden.max_steps, 5);
    assert_eq!(overridden.clip_norm, Some(0.5));
    assert_eq!(overridden.accumulation, 2);

    let bad_steps = TrainConfig::with_overrides(None, None, Some(0), None, None);
    assert_eq!(bad_steps.expect_err("总步数不能为 0").code(), Code::Invalid);

    let bad_accum = TrainConfig::with_overrides(None, None, None, None, Some(0));
    assert_eq!(
        bad_accum.expect_err("累积窗口不能为 0").code(),
        Code::Invalid
    );
}
}


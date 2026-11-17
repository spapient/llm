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
use tensor::{DType, Shape};

use crate::optimizer::Optimizer;
use crate::pool::Buffers;
use crate::schedule::Schedule as LrSchedule;
use crate::snapshot::{GradAccum, Moments, SlotMoments, Snapshot, Ties};
use crate::{Data, Kernel, Slot};

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
pub struct Session<K: Kernel, O: Optimizer> {
    tape: Tape,
    sizes: Sizes,
    buffers: Buffers<K>,
    optimizer: O,
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
    /// 由图、形状表、内核、优化器与配置构造；学习率调度按 warmup 与总步数建立。
    pub fn new(
        tape: Tape,
        sizes: Sizes,
        kernel: K,
        optimizer: O,
        config: TrainConfig,
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

    /// 绑定表回载：编号必须落在会话槽位内，两端长度一致；回载替换既有绑定表。
    fn restore_ties(&mut self, ties: &Ties) -> Result<()> {
        let slot_count = self.slots.len();
        for &(source, tied) in ties.pairs() {
            if source >= slot_count || tied >= slot_count {
                let message =
                    format!("绑定对 ({source}, {tied}) 编号越界：会话只有 {slot_count} 个参数槽");
                return Err(Error::new(Code::Invalid, message).with(SITE));
            }
            if self.slots[source].params().len() != self.slots[tied].params().len() {
                let message = format!(
                    "绑定两端长度不一致：源槽 {} 元素，回填槽 {} 元素",
                    self.slots[source].params().len(),
                    self.slots[tied].params().len()
                );
                return Err(Error::new(Code::Invalid, message).with(SITE));
            }
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

    /// 走一步优化器：按当前学习率更新全部参数槽，更新后镜像回激活缓冲。
    ///
    /// # Errors
    ///
    /// 某槽梯度未装入、或优化器步进校验失败（[`Code::Invalid`] / [`Code::Missing`]）。
    pub fn optimize(&mut self) -> Result<()> {
        self.accum = self.accum.saturating_add(1);
        if self.accum < self.config.accumulation {
            // 累积窗口未满：保留已汇总的梯度，等下一批 micro-batch，本批不步进。
            return Ok(());
        }
        self.accum = 0;
        let lr = self.lr_schedule.lr(self.config.learning_rate, self.step);
        let mut views = Self::collect_views(&mut self.slots, &mut self.buffers)?;
        self.optimizer
            .step(&mut views, lr)
            .map_err(|error| error.with(SITE))?;
        Self::mirror_params(&self.slots, &mut self.buffers)?;
        // 窗口已步进：清空梯度，下一个窗口从空缓冲重新累加。
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
        if self.slots[source].params().len() != self.slots[tied].params().len() {
            let message = format!(
                "绑定两端长度不一致：源槽 {} 元素，回填槽 {} 元素",
                self.slots[source].params().len(),
                self.slots[tied].params().len()
            );
            return Err(Error::new(Code::Invalid, message).with(SITE));
        }
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
    fn collect_views<'a>(
        slots: &'a mut [ParamSlot],
        buffers: &'a mut Buffers<K>,
    ) -> Result<Vec<Slot<'a>>> {
        slots
            .iter_mut()
            .map(|slot| {
                let role = slot.role();
                let grads = buffers.gradient(slot.var()).ok_or_else(|| {
                    Error::new(
                        Code::Missing,
                        format!("参数 {} 的梯度还没装入缓冲", slot.var().id()),
                    )
                    .with(SITE)
                })?;
                Ok(Slot {
                    params: slot.params_mut(),
                    grads,
                    role,
                })
            })
            .collect::<Result<Vec<_>>>()
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
#[path = "../tests/session_tests.rs"]
mod tests;

//! 训练会话：tape → 反传计划 → 梯度需求分类 → 梯度缓冲 → 优化器步的最小闭环；分布式同步配置
//! （[`Distributed`]）与它同住一处——会话是它唯一的装配者，同步口径随 [`TrainConfig`] 走。学习率
//! 调度曲线在 [`schedule`](crate::schedule)；快照事实集与视图装配在 [`checkpoint`]。

use autograd::{Requirement, Sizes, Tape, Var};
use diagnostic::{Code, Error, Result};
use nn::ParamRole;
use shard::{Collective, Group, Parallel, ParallelConfig, Priority, Reduce, Topology};
use tensor::{DType, Shape};

use crate::loss::ObjectiveConfig;
use crate::optimizer::{Optimizer, Slot};
use crate::pool::Bufs;
use crate::schedule::{Anneal, Schedule};
use crate::{Data, Kernel, LoaderConfig};
use storage::{GradAccum, Ties};

pub use shard::{Local, Remote};
mod checkpoint;
pub use checkpoint::{Checkpoint, GradientSnapshot, Step, operators, resume};

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
    /// 构造并校验全部训练数值；公开字段的后续修改可由 [`Self::validate`] 再检查。
    pub fn new(
        learning_rate: f32,
        warmup_steps: u64,
        max_steps: u64,
        clip_norm: Option<f32>,
        accumulation: u32,
    ) -> Result<Self> {
        let config = Self {
            learning_rate,
            warmup_steps,
            max_steps,
            clip_norm,
            accumulation,
            anneal: Anneal::default(),
            min_ratio: 0.0,
        };
        config.validate()?;
        Ok(config)
    }

    /// 校验配置不变量；`Session` 装配时再次调用，防止公开字段被直接改成非法值。
    pub fn validate(&self) -> Result<()> {
        self.check_schedule()?;
        self.check_regularization()?;
        Ok(())
    }

    /// 调度与学习率口径：总步数、warmup 关系、学习率非负有限。
    fn check_schedule(&self) -> Result<()> {
        if self.max_steps == 0 {
            return Err(Error::new(Code::Invalid, "总步数不能为 0").with(SITE));
        }
        if self.warmup_steps > self.max_steps {
            return Err(Error::new(
                Code::Invalid,
                format!("warmup {} 超过总步数 {}", self.warmup_steps, self.max_steps),
            )
            .with(SITE));
        }
        if !self.learning_rate.is_finite() || self.learning_rate < 0.0 {
            return Err(Error::new(Code::Invalid, "学习率必须是有限非负数").with(SITE));
        }
        Ok(())
    }

    /// 正则与累积口径：裁剪范数为正有限、累积窗口非零、下限比例落在 `[0, 1]`。
    fn check_regularization(&self) -> Result<()> {
        if self
            .clip_norm
            .is_some_and(|clip| !clip.is_finite() || clip <= 0.0)
        {
            return Err(Error::new(Code::Invalid, "裁剪范数必须是有限正数").with(SITE));
        }
        if self.accumulation == 0 {
            return Err(Error::new(Code::Invalid, "累积窗口不能为 0").with(SITE));
        }
        if !(0.0..=1.0).contains(&self.min_ratio) {
            return Err(Error::new(Code::Invalid, "下限比例必须在 [0, 1]").with(SITE));
        }
        Ok(())
    }

    /// 按可选覆盖构造：未给的字段取 [`Default`]。
    pub fn with_overrides(
        learning_rate: Option<f32>,
        warmup_steps: Option<u64>,
        max_steps: Option<u64>,
        clip_norm: Option<f32>,
        accumulation: Option<u32>,
    ) -> Result<Self> {
        Self::default().overridden(
            learning_rate,
            warmup_steps,
            max_steps,
            clip_norm,
            accumulation,
        )
    }

    /// 在当前配置上应用可选覆盖，保留调度曲线等未覆盖字段。
    pub fn overridden(
        self,
        learning_rate: Option<f32>,
        warmup_steps: Option<u64>,
        max_steps: Option<u64>,
        clip_norm: Option<f32>,
        accumulation: Option<u32>,
    ) -> Result<Self> {
        let config = Self {
            learning_rate: learning_rate.unwrap_or(self.learning_rate),
            warmup_steps: warmup_steps.unwrap_or(self.warmup_steps),
            max_steps: max_steps.unwrap_or(self.max_steps),
            clip_norm: clip_norm.or(self.clip_norm),
            accumulation: accumulation.unwrap_or(self.accumulation),
            anneal: self.anneal,
            min_ratio: self.min_ratio,
        };
        config.validate()?;
        Ok(config)
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

/// 一次训练装配的静态配置；运行状态仍由 `Session`、`Loader` 等对象分别持有。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainSpec {
    /// 会话、学习率调度与梯度累积配置。
    pub session: TrainConfig,
    /// 数据批几何与互斥采样策略。
    pub loader: LoaderConfig,
    /// 静态目标种类与超参数。
    pub objective: ObjectiveConfig,
}

impl TrainSpec {
    /// 在装配边界统一校验各子配置；默认目标是分类。
    pub fn new(session: TrainConfig, loader: LoaderConfig) -> Result<Self> {
        Self::default()
            .with_session(session)?
            .with_loader(loader)
    }

    /// 覆盖会话配置并校验。
    pub fn with_session(mut self, session: TrainConfig) -> Result<Self> {
        session.validate()?;
        self.session = session;
        Ok(self)
    }

    /// 覆盖数据配置并校验。
    pub fn with_loader(mut self, loader: LoaderConfig) -> Result<Self> {
        loader.validate()?;
        self.loader = loader;
        Ok(self)
    }

    /// 覆盖静态目标并复用损失函数的数值判据。
    pub fn with_objective(mut self, objective: ObjectiveConfig) -> Result<Self> {
        objective.validate()?;
        self.objective = objective;
        Ok(self)
    }

    /// 拆成会话与数据的运行配置；目标仍可由 [`Self::objective`] 在拆分前读取。
    pub fn into_parts(self) -> (TrainConfig, LoaderConfig) {
        (self.session, self.loader)
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
        mut optimizer: O,
        config: TrainConfig,
        distributed: Distributed<C>,
    ) -> Result<Self> {
        config.validate()?;
        optimizer.set_clip_norm(config.clip_norm);
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
                    .map_or_default(<[f32]>::to_vec)
            })
            .collect();
        GradAccum::new(slots, self.accum)
    }

    /// 独立梯度快照：按 [`Var`] 复制一份当前梯度向量。与 [`Self::accum_snapshot`] 不同，
    /// 它不读累积窗口计数、不清缓冲、也不与参数槽下标绑定；副本在后续 `zero_gradients`
    /// 或优化器步进之后仍然有效。
    pub fn gradient_snapshot(&self) -> GradientSnapshot {
        let entries = self
            .slots
            .iter()
            .filter_map(|slot| {
                self.bufs
                    .gradient(slot.var())
                    .map(|grad| (slot.var(), grad.to_vec()))
            })
            .collect();
        GradientSnapshot::new(entries)
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
        let next = self.accum.saturating_add(1);
        if next < self.config.accumulation {
            // 累积窗口未满：保留已汇总的梯度，等下一批 micro-batch，本批不步进。
            self.accum = next;
            return Ok(());
        }
        self.step_once()?;
        // 只有完整步进成功才提交窗口状态；失败时调用方仍能看到未提交的窗口计数。
        self.accum = 0;
        Ok(())
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

#[cfg(test)]
#[path = "../tests/session_tests.rs"]
mod tests;

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

#[cfg(test)]
#[path = "../tests/parallel_config_tests.rs"]
mod parallel_config_tests;

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
use crate::optimizer::{Optimizer, Slot};
use crate::pool::Buffers;
use crate::schedule::Schedule as LrSchedule;
use crate::{Data, Kernel};
use snapshot::{GradAccum, Ties};

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

/// 取视图时按槽号记下的元数据：变量、角色、矩阵形状。
///
/// 单独起名的理由是可读性而不是洁癖：`Vec<(Var, ParamRole, Option<(usize, usize)>)>` 写出来
/// 读者要数三层括号才知道自己在看什么。
type SlotMeta = (Var, ParamRole, Option<(usize, usize)>);

/// 一个已登记的参数槽：变量编号、角色与常驻参数值。
///
/// 梯度不在这里，梯度缓冲由 [`Buffers`](crate::Buffers) 自持；参数值是优化器就地写回的对象，
/// 每步前向再镜像回激活缓冲。
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
    ///
    /// 它随参数槽一起走，因为 [`Optimizer`] 的矩阵类实现（[`crate::Muon`]）要在矩阵上做正交化；
    /// 形状只从图上的登记形状派生，不从参数长度反推（`4` 既可以是 `2 × 2`，也可以是 `4 × 1`）。
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
        let matrix = matrix_of(shape);
        self.buffers
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
}

mod restore;
mod views;

/// 登记形状 → 矩阵形状：只有二阶张量是矩阵，其余（标量/向量/更高阶）一律 `None`。
///
/// 不从元素数反推：`4` 既可以是 `2 × 2`，也可以是 `4 × 1`，反推出来的行列会让 [`crate::Muon`]
/// 对一个其实不是矩阵的参数做正交化，却不会有任何一处报错。
fn matrix_of(shape: &Shape) -> Option<(usize, usize)> {
    match shape.dims() {
        &[rows, cols] => Some((rows, cols)),
        _ => None,
    }
}

#[cfg(test)]
#[path = "../tests/session_tests.rs"]
mod tests;

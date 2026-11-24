//! 静态训练配置与可恢复检查点：学习率调度、命名覆盖、快照回载校验和优化器视图装配。

use super::{ParamSlot, SITE, Session, SlotMeta};
use crate::loss::ObjectiveConfig;
use crate::{Kernel, LoaderConfig};
use crate::optimizer::{Optimizer, OwnedSlot};
use crate::pool::Bufs;
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

    getters! {
        /// 步号（从 0 开始）。
        idx => idx: u64,
        /// 该步的平均损失。
        loss => loss: f32,
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
    config_fingerprint: u64,
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
            config_fingerprint: 0,
        })
    }

    builders! {
        /// 绑定保存时的训练配置指纹。
        with_config_fingerprint => config_fingerprint: u64,
    }
    getters! {
        /// 保存时的训练配置指纹。
        config_fingerprint => config_fingerprint: u64,
        /// 步号。
        step => step: u64,
        /// 反向图节点数的内存诊断量；不进快照，图结构由权重条目表与层算子契约守卫。
        tape_nodes => tape_nodes: usize,
        /// 批的行数。
        batch_rows => batch_rows: usize,
        /// 批的列宽。
        batch_cols => batch_cols: usize,
        /// 训练层数。
        layers => layers: usize,
        /// 权重条目数。
        weight_entries => weight_entries: usize,
        /// 恢复时要求的进程数。
        world => world: usize,
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
        Ok(weights::Snapshot::new(facts, weights.clone())
            .with_config_fingerprint(self.config_fingerprint))
    }
}

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

    getters! {
        /// warmup 步数。
        warmup => warmup: u64,
        /// 总步数。
        total => total: u64,
        /// 衰减曲线。
        anneal => anneal: Anneal,
        /// 谷底占峰值的比例。
        min_ratio => min_ratio: f32,
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

/// 训练覆盖参数：`None` 保留原配置，避免同类型位置参数在装配边界错配。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TrainOverrides {
    /// 学习率峰值（warmup 结束时）。
    pub learning_rate: Option<f32>,
    /// warmup 步数。
    pub warmup_steps: Option<u64>,
    /// 计划总步数（调度分母）。
    pub max_steps: Option<u64>,
    /// 全局梯度范数上限；不能通过覆盖清除已有的裁剪阈值。
    pub clip_norm: Option<f32>,
    /// 梯度累积窗口：多少个 micro-batch 汇总一次再步进。
    pub accumulation: Option<u32>,
}

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
    pub fn with_overrides(overrides: TrainOverrides) -> Result<Self> {
        Self::default().overridden(overrides)
    }

    /// 在当前配置上应用可选覆盖，保留调度曲线等未覆盖字段。
    pub fn overridden(self, overrides: TrainOverrides) -> Result<Self> {
        let config = Self {
            learning_rate: overrides.learning_rate.unwrap_or(self.learning_rate),
            warmup_steps: overrides.warmup_steps.unwrap_or(self.warmup_steps),
            max_steps: overrides.max_steps.unwrap_or(self.max_steps),
            clip_norm: overrides.clip_norm.or(self.clip_norm),
            accumulation: overrides.accumulation.unwrap_or(self.accumulation),
            anneal: self.anneal,
            min_ratio: self.min_ratio,
        };
        config.validate()?;
        Ok(config)
    }

    builders! {
        /// 设置 warmup 之后的衰减曲线（builder 风格，覆盖默认的线性衰减）。
        with_anneal => anneal: Anneal,
        /// 设置谷底比例；越界不静默修正，装配调度时拒绝。
        with_min_ratio => min_ratio: f32,
    }
    getters! {
        /// 衰减曲线。
        anneal => anneal: Anneal,
        /// 谷底占峰值的比例。
        min_ratio => min_ratio: f32,
        /// 梯度累积窗口。
        accumulation => accumulation: u32,
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
        let spec = Self {
            session,
            loader,
            objective: ObjectiveConfig::default(),
        };
        spec.validate()?;
        Ok(spec)
    }

    /// 覆盖静态目标并复用损失函数的数值判据。
    pub fn with_objective(mut self, objective: ObjectiveConfig) -> Result<Self> {
        objective.validate()?;
        self.objective = objective;
        Ok(self)
    }

    /// 校验当前完整配置，防止公开字段在构造后被改成非法值。
    pub fn validate(&self) -> Result<()> {
        self.session.validate()?;
        self.loader.validate()?;
        self.objective.validate()
    }

    /// 拆成会话与数据的运行配置；目标仍可由 [`Self::objective`] 在拆分前读取。
    pub fn into_parts(self) -> (TrainConfig, LoaderConfig) {
        (self.session, self.loader)
    }
}

impl TrainSpec {
    /// 完整训练配置的确定指纹；优化器类型与静态超参均纳入，浮点按 IEEE 位模式哈希。
    pub fn config_fingerprint<O: Optimizer + ?Sized>(&self, optimizer: &O) -> u64 {
        let mut hash = StableHash::new("train-spec-v1");
        hash.text(std::any::type_name::<O>());
        optimizer.write_config(&mut |value| hash.number(value));
        hash_config(&mut hash, &self.session);
        hash.number(self.loader.rows as u64);
        hash.number(self.loader.cols as u64);
        hash_sampling(&mut hash, &self.loader.sampling);
        hash.text(self.objective.as_str());
        hash_objective(&mut hash, &self.objective);
        hash.finish()
    }
}

/// 只有会话配置时的默认指纹；完整装配可用 [`TrainSpec::config_fingerprint`] 覆盖。
pub(super) fn session_fingerprint<O: Optimizer + ?Sized>(
    config: &TrainConfig,
    optimizer: &O,
) -> u64 {
    let mut hash = StableHash::new("train-session-v1");
    hash.text(std::any::type_name::<O>());
    optimizer.write_config(&mut |value| hash.number(value));
    hash_config(&mut hash, config);
    hash.finish()
}

fn hash_config(hash: &mut StableHash, config: &TrainConfig) {
    hash.number(config.learning_rate.to_bits().into());
    hash.number(config.warmup_steps);
    hash.number(config.max_steps);
    hash.number(config.clip_norm.is_some().into());
    if let Some(clip) = config.clip_norm {
        hash.number(clip.to_bits().into());
    }
    hash.number(config.accumulation.into());
    hash.text(config.anneal.as_str());
    hash.number(config.min_ratio.to_bits().into());
}

/// 数据采样策略按变体编号、种子及权重原位哈希，保持指纹输入顺序稳定。
fn hash_sampling(hash: &mut StableHash, sampling: &crate::Sampling) {
    match sampling {
        crate::Sampling::Sequential => hash.number(0),
        crate::Sampling::Shuffle { seed } => {
            hash.number(1);
            hash.number(*seed);
        }
        crate::Sampling::Weighted { weights, seed } => {
            hash.number(2);
            hash.number(*seed);
            hash.number(weights.len() as u64);
            for &weight in weights {
                hash.number(weight.to_bits().into());
            }
        }
    }
}

/// 静态目标的名称已先行写入，这里只追加各变体自己的数值参数。
fn hash_objective(hash: &mut StableHash, objective: &ObjectiveConfig) {
    match objective {
        ObjectiveConfig::Classify => {}
        ObjectiveConfig::Distill(config) => {
            hash.number(config.temperature().to_bits().into());
            hash.number(config.alpha().to_bits().into());
        }
        ObjectiveConfig::Grpo(config) => {
            hash.number(config.epsilon().to_bits().into());
            hash.number(config.beta().to_bits().into());
        }
    }
}

/// 格式固定的小端 FNV-1a：不依赖 `DefaultHasher` 的实现细节。
struct StableHash(u64);

impl StableHash {
    fn new(domain: &str) -> Self {
        let mut hash = Self(0xcbf2_9ce4_8422_2325);
        hash.text(domain);
        hash
    }

    fn number(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn text(&mut self, text: &str) {
        self.number(text.len() as u64);
        self.bytes(text.as_bytes());
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(self) -> u64 {
        // `0` 是旧快照的“未声明”哨兵，不让有效配置落入跳过校验的分支。
        if self.0 == 0 { 1 } else { self.0 }
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

    ref_getters! {
        /// 全部条目：变量与其梯度向量。
        #[must_use]
        entries => entries: &[(Var, Vec<f32>)],
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
        self.check_config_fingerprint(snapshot.config_fingerprint())?;
        self.restore_moments(snapshot.moments())?;
        self.restore_accum(snapshot.accum())?;
        self.restore_ties(snapshot.ties())?;
        self.step = snapshot.facts().step();
        Ok(())
    }

    /// 旧快照的零指纹表示未声明；非零指纹必须先于任何状态写入完成校验。
    fn check_config_fingerprint(&self, saved: u64) -> Result<()> {
        if saved != 0 && saved != self.config_fingerprint {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "训练配置指纹不一致：快照 {saved:#018x}，当前 {:#018x}",
                    self.config_fingerprint
                ),
            )
            .with(SITE));
        }
        Ok(())
    }

    /// 矩状态回载：空表对应无矩优化器；非空表槽位数与每槽长度必须对齐会话参数槽。
    fn restore_moments(&mut self, moments: &Moments) -> Result<()> {
        let slots = moments.slots();
        self.check_moment_slots(slots)?;
        let momentum = slots.iter().map(|s| s.m().to_vec()).collect();
        let variance = slots.iter().map(|s| s.v().to_vec()).collect();
        let shapes: Vec<_> = self.slots.iter().map(|slot| slot.matrix()).collect();
        self.optimizer
            .restore_shaped(momentum, variance, moments.step(), &shapes)
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

    /// 单槽矩长度对账：一阶矩必须等长；二阶矩可等长，或由无二阶矩的实现留空。
    fn check_moment_slot_length(&self, idx: usize, slot: &SlotMoments) -> Result<()> {
        let want = self.slots[idx].params().len();
        if slot.m().len() != want || !(slot.v().is_empty() || slot.v().len() == want) {
            let message = format!(
                "第 {idx} 槽矩长度不符：一阶 {} / 二阶 {} 元素，会话 {want} 元素",
                slot.m().len(),
                slot.v().len(),
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

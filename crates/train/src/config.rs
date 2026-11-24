//! 静态训练配置：学习率调度、命名覆盖、会话参数与数据批几何的装配边界。

use diagnostic::{Code, Error, Result};

use crate::LoaderConfig;
use crate::loss::ObjectiveConfig;

/// 配置校验沿用会话诊断站点，保持既有错误上下文稳定。
const SITE: &str = "train::Session";

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

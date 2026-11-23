//! 独立训练编排器：把数据集遍历、前向评估、指标入账与优化步收拢到一个驱动对象；训练样本
//! [`Batch`]/[`Dataset`]/[`Chunks`] 与它同住一处——[`Learner`] 是它们的唯一消费者，批几何与批总数
//! 只有一处算。
//!
//! [`Metric`] 是本文件的入账目标枚举，[`MetricPoint`]/[`MetricsBoard`] 是它的账表：
//! 编排与记账用同一个概念，故同住一处，而非各自成文件。
//!
//! # 训练样本
//!
//! **拥有**：`Batch`、`Dataset`、`Chunks`。**不拥有**：分词（`tokenizer`）、打乱、打包与训练/验证集
//! 切分（打乱与打包只改变 [`Chunks`] 里游标的推进方式，切分是调用方按行区间的选择）。
//!
//! 契约的重点是"形状即校验"：`rows * cols` 必须等于 token 数，否则训练循环会在第 N 步才发现
//! 偏移错位——那时已经烧掉了几个小时。两个构造函数共用同一份几何校验（行列非零、乘积不溢出），
//! 所以同一对非法行列不管走哪条路径，拿到的是同一个 `Code::Invalid`。
//!
//! ## 拥有与借用
//!
//! 这些类型**不拥有** token 缓冲：调用方持有它，本模块只给出一段视图。取一批因此没有分配、
//! 也没有拷贝，批次的生命周期停在调用方那份缓冲上，而不停在数据集对象上。
//!
//! ## 按批取样本
//!
//! [`Chunks`] 是 `Iterator`，并且它知道自己还剩几批——`ExactSizeIterator::len` 与 `size_hint`
//! 都取自 [`Dataset::remaining_batches`]，"还剩几条"只有一个来源：
//!
//! ```
//! use llm_train::{Chunks, Dataset};
//!
//! let tokens = [0, 1, 2, 3, 4, 5];
//! let mut chunks = Chunks::new(&tokens, 1, 2).expect("每批 1 行、每行 2 个 token");
//! assert_eq!(chunks.len(), 3);
//! let rows: usize = chunks.by_ref().map(|batch| batch.rows()).sum();
//! assert_eq!((rows, chunks.len()), (3, 0));
//! // `next()` 与 `Dataset::next_batch` 是同一个游标，混用不会跳批。
//! let mut again = Chunks::new(&tokens, 1, 2).expect("同上");
//! assert_eq!(again.next_batch().expect("第一批").tokens(), [0, 1]);
//! assert_eq!(again.next().expect("第二批").tokens(), [2, 3]);
//! ```
//!
//! 它也是**对象安全**的：`&mut dyn Dataset<'a>` 与具体类型走同一条取批路径，所以按批消费的
//! 代码可以只依赖 trait。
//!
//! ```
//! use llm_train::{Chunks, Dataset};
//!
//! fn drain<'a>(dataset: &mut dyn Dataset<'a>) -> usize {
//!     core::iter::from_fn(|| dataset.next_batch()).count()
//! }
//!
//! let tokens = [0, 1, 2, 3];
//! let mut chunks = Chunks::new(&tokens, 1, 2).expect("每批 1 行、每行 2 个 token");
//! assert_eq!(drain(&mut chunks), 2);
//! ```
//!
//! # 例子
//!
//! ```
//! use llm_train::Batch;
//!
//! let tokens = [1, 2, 3, 4, 5, 6];
//! let batch = Batch::new(&tokens, 2, 3).expect("2 x 3");
//! assert_eq!((batch.rows(), batch.cols()), (2, 3));
//! ```

use diagnostic::{Code, Error, Result};

use autograd::Var;
use shard::Collective;

use crate::loss::{Distill, Grpo, Outcome, classify, distill, grpo};
use crate::optimizer::Optimizer;
use crate::{Kernel, Session};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::Learner";

/// 指标入账的诊断站点：本文件承载"训练指标看板"这一轴，站名沿用看板自己的名字，
/// 使 `Learner` 编排失败与指标入参非法在诊断里仍可区分。
pub(crate) const METRICS_SITE: &str = "train::metrics";

/// 一次前向的两个评估终点：`logits` 用于算 NLL 与命中率，`loss` 用于反传计划对账。
#[derive(Debug, Clone, Copy)]
pub struct ForwardOut {
    /// 每个有效位置的词表分布，展平成 `[positions, vocab]`。
    pub logits: Var,
    /// 标量/向量 loss 终点，供会话规划梯度可达性。
    pub loss: Var,
    /// 可选的非分类度量来源：`None` 走分类 NLL + 命中率。
    pub metric: Option<Metric>,
}

/// 非分类度量来源：蒸馏/RL 度量所需的额外变量与配置。
#[derive(Debug, Clone, Copy)]
pub enum Metric {
    /// Hinton 蒸馏：软 KL + 硬 CE。
    Distill {
        /// 教师 logits（软标签来源），与 `logits` 同形状。
        teacher: Var,
        /// 蒸馏配置（温度 `T` 与软损失权重 `α`）。
        config: Distill,
    },
    /// GRPO：裁剪策略梯度 + 参考 KL 惩罚。
    Grpo {
        /// 当前策略逐 token log-prob。
        new_logp: Var,
        /// 行为策略逐 token log-prob（冻结）。
        old_logp: Var,
        /// 逐 token advantage（同 completion 共享）。
        advantages: Var,
        /// 逐 token 参考 KL。
        ref_kl: Var,
        /// GRPO 配置（裁剪 `ε` 与 KL 惩罚 `β`）。
        config: Grpo,
    },
}

/// 前向一步取出的评估数据：标签与同序位置的 logits；反传终点不在此处留存。
struct Prepared {
    labels: Vec<f32>,
    logits: Vec<f32>,
    metric: Option<Metric>,
}

/// 训练编排器：包住一个会话与一块指标看板，按 epoch 遍历数据集推进。
pub struct Learner<K: Kernel, O: Optimizer, C: Collective = shard::Remote> {
    session: Session<K, O, C>,
    board: MetricsBoard,
    vocab: usize,
}

impl<K: Kernel, O: Optimizer, C: Collective> Learner<K, O, C> {
    /// 由已装配会话与词表宽度构造；词表为 0 是 [`Code::Invalid`]。
    pub fn new(session: Session<K, O, C>, vocab: usize) -> Result<Self> {
        if vocab == 0 {
            return Err(Error::new(Code::Invalid, "词表宽度不能为 0").with(SITE));
        }
        Ok(Self {
            session,
            board: MetricsBoard::new(),
            vocab,
        })
    }

    /// 会话状态（只读）。
    pub fn session(&self) -> &Session<K, O, C> {
        &self.session
    }

    /// 指标看板（只读）。
    pub fn board(&self) -> &MetricsBoard {
        &self.board
    }

    /// 跑指定 epoch 数；每个完整批走一次前向评估、入账与优化步。
    ///
    /// `forward` 借会话与本批 token 装配前向，返回两个终点；`backward` 在入账后把梯度装进缓冲。
    /// 数据集残批（不足一行）由 [`Dataset`] 自身丢弃，编排器不补齐。
    ///
    /// # Errors
    ///
    /// 批几何、前向、计划、梯度入账或优化步任一步失败都会透传；评估几何不自洽也是
    /// [`Code::Invalid`]。
    pub fn run<'a, F, G>(
        &mut self,
        dataset: &mut dyn Dataset<'a>,
        epochs: u64,
        mut forward: F,
        mut backward: G,
    ) -> Result<()>
    where
        F: FnMut(&mut Session<K, O, C>, &[u32]) -> Result<ForwardOut>,
        G: FnMut(&mut Session<K, O, C>) -> Result<()>,
    {
        for _epoch in 0..epochs {
            while let Some(batch) = dataset.next_batch() {
                self.train_batch(
                    batch.rows(),
                    batch.cols(),
                    batch.tokens(),
                    &mut forward,
                    &mut backward,
                )?;
            }
        }
        Ok(())
    }

    /// 在独立验证集上跑一轮：只做前向与评估入账，不回传梯度、不步进优化器、不并入训练序列。
    ///
    /// 评估指标进看板的独立账表（[`MetricsBoard::eval_points`]），训练样本数/加权均值/最优
    /// loss 都不受验证批影响；跑完清空梯度缓冲，避免残留验证梯度混进下一个训练窗口。
    ///
    /// # Errors
    ///
    /// 批几何、前向或评估入账任一步失败都会透传；评估几何不自洽是 [`Code::Invalid`]。
    pub fn evaluate<'a, F>(&mut self, dataset: &mut dyn Dataset<'a>, mut forward: F) -> Result<()>
    where
        F: FnMut(&mut Session<K, O, C>, &[u32]) -> Result<ForwardOut>,
    {
        while let Some(batch) = dataset.next_batch() {
            let prepared =
                self.forward_pass(batch.rows(), batch.cols(), batch.tokens(), &mut forward)?;
            self.record_eval(&prepared)?;
        }
        // 验证批不回传：清空可能残留的激活梯度，下一个训练窗口从空缓冲重新累加。
        self.session.zero_gradients();
        Ok(())
    }

    /// 单个 micro-batch 的编排闭环：前向装配 → 评估入账 → 装梯度 → 优化步。
    fn train_batch<F, G>(
        &mut self,
        rows: usize,
        cols: usize,
        tokens: &[u32],
        forward: &mut F,
        backward: &mut G,
    ) -> Result<()>
    where
        F: FnMut(&mut Session<K, O, C>, &[u32]) -> Result<ForwardOut>,
        G: FnMut(&mut Session<K, O, C>) -> Result<()>,
    {
        let prepared = self.forward_pass(rows, cols, tokens, forward)?;
        self.finish(prepared, backward)?;
        Ok(())
    }

    /// 入账评估量、回传梯度、跑闭包覆盖口并走优化步。
    fn finish<G>(&mut self, prepared: Prepared, backward: &mut G) -> Result<()>
    where
        G: FnMut(&mut Session<K, O, C>) -> Result<()>,
    {
        self.record(&prepared)?;
        self.backward_with(backward)?;
        self.session.optimize()?;
        Ok(())
    }

    /// 自动回算参数梯度，再跑调用方闭包覆盖/追加口。
    fn backward_with<G>(&mut self, backward: &mut G) -> Result<()>
    where
        G: FnMut(&mut Session<K, O, C>) -> Result<()>,
    {
        self.session.backward()?;
        backward(&mut self.session)?;
        Ok(())
    }

    /// 接批、跑前向并取出评估所需的标签与 logits；反传终点已由前向闭包登记进计划。
    fn forward_pass<F>(
        &mut self,
        rows: usize,
        cols: usize,
        tokens: &[u32],
        forward: &mut F,
    ) -> Result<Prepared>
    where
        F: FnMut(&mut Session<K, O, C>, &[u32]) -> Result<ForwardOut>,
    {
        self.session.accept_batch(rows, cols, tokens.len())?;
        let out = forward(&mut self.session, tokens)?;
        let labels = next_token_labels(tokens, rows, cols);
        let logits = self.read_var(out.logits)?;
        Ok(Prepared {
            labels,
            logits,
            metric: out.metric,
        })
    }

    /// 读出变量终点的 f32 切片；不在场是 [`Code::Missing`]，lane 不符是 [`Code::Invalid`]。
    fn read_var(&mut self, var: Var) -> Result<Vec<f32>> {
        let view = self.session.bufs().value(var).ok_or_else(|| {
            Error::new(Code::Missing, format!("变量 {} 不在场", var.id())).with(SITE)
        })?;
        view.f32()
            .map(<[f32]>::to_vec)
            .ok_or_else(|| Error::new(Code::Invalid, "变量不是 f32 lane").with(SITE))
    }

    /// 按度量来源计算本步 (loss, 命中)；`None` 走分类，`Some` 走蒸馏/RL。
    fn compute_metric(&mut self, prepared: &Prepared) -> Result<Outcome> {
        match &prepared.metric {
            None => classify(&prepared.logits, &prepared.labels, self.vocab),
            Some(Metric::Distill { teacher, config }) => {
                let teacher = self.read_var(*teacher)?;
                distill(
                    &prepared.logits,
                    &teacher,
                    &prepared.labels,
                    self.vocab,
                    *config,
                )
            }
            Some(Metric::Grpo {
                new_logp,
                old_logp,
                advantages,
                ref_kl,
                config,
            }) => self.compute_grpo(*new_logp, *old_logp, *advantages, *ref_kl, *config),
        }
    }

    /// RL 度量：读出四个逐 token 变量后算 GRPO 组合损失。
    fn compute_grpo(
        &mut self,
        new_logp: Var,
        old_logp: Var,
        advantages: Var,
        ref_kl: Var,
        config: Grpo,
    ) -> Result<Outcome> {
        let new_logp = self.read_var(new_logp)?;
        let old_logp = self.read_var(old_logp)?;
        let advantages = self.read_var(advantages)?;
        let ref_kl = self.read_var(ref_kl)?;
        grpo(&new_logp, &old_logp, &advantages, &ref_kl, config)
    }

    /// 按度量算出一个待记账点：loss、命中、样本数与当前训练步。
    fn point(&mut self, prepared: &Prepared) -> Result<MetricPoint> {
        let outcome = self.compute_metric(prepared)?;
        Ok(MetricPoint {
            step: self.session.step(),
            samples: prepared.labels.len(),
            loss: outcome.loss,
            correct: outcome.correct,
        })
    }

    /// 按度量计算 loss 与命中并记进训练看板。
    fn record(&mut self, prepared: &Prepared) -> Result<()> {
        let point = self.point(prepared)?;
        self.board.record(point).map_err(|error| error.with(SITE))
    }

    /// 按度量计算 loss 与命中并记进评估独立账表；步号沿用当前训练步，仅作对账标记。
    fn record_eval(&mut self, prepared: &Prepared) -> Result<()> {
        let point = self.point(prepared)?;
        self.board
            .record_eval(point)
            .map_err(|error| error.with(SITE))
    }
}

/// next-token 标签：`(r, c)` 的标签取 `tokens[r, c+1]`；行末列无后继，剔除。
fn next_token_labels(tokens: &[u32], rows: usize, cols: usize) -> Vec<f32> {
    let positions = rows.saturating_mul(cols.saturating_sub(1));
    let mut labels = Vec::with_capacity(positions);
    // 行边界切断右移：外层按行推进；行内后继是一段连续 slice，整块映射进标签表。
    for row in 0..rows {
        let base = row * cols;
        labels.extend(
            tokens[base + 1..base + cols]
                .iter()
                .map(|&token| token as f32),
        );
    }
    labels
}

/// 一条步指标：步号、本步样本数、平均 loss、命中条数。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricPoint {
    /// 优化器步数（未步进的累积批不计步号）。
    pub step: u64,
    /// 本步参与评估的样本条数。
    pub samples: usize,
    /// 本步平均交叉熵。
    pub loss: f32,
    /// 本步命中条数，闭区间 `0..=samples`。
    pub correct: u64,
}

impl MetricPoint {
    /// 命中率 `correct / samples`，闭区间 `[0, 1]`。
    pub fn accuracy(&self) -> f32 {
        self.correct as f32 / self.samples as f32
    }
}

/// 一份账表：训练序列与评估序列各持一份。聚合规则逐字相同，故只写一次——
/// 两条序列的语义不可能各自漂移，加权均值/最优值的口径天然一致。
#[derive(Debug, Clone, Default, PartialEq)]
struct Ledger {
    points: Vec<MetricPoint>,
    samples: u64,
    loss_sum: f64,
    correct: u64,
    best: Option<f32>,
}

impl Ledger {
    /// 记一条指标：先校验，再累计样本、加权 loss 与命中条数，并刷新最优 loss。
    fn push(&mut self, point: MetricPoint) -> Result<()> {
        validate(&point)?;
        let samples = point.samples as u64;
        self.samples += samples;
        self.loss_sum += point.loss as f64 * samples as f64;
        self.correct += point.correct;
        self.best = Some(match self.best {
            Some(best) => best.min(point.loss),
            None => point.loss,
        });
        self.points.push(point);
        Ok(())
    }

    /// 样本加权平均 loss；无样本为 `None`。
    fn mean_loss(&self) -> Option<f32> {
        self.nonzero_samples()
            .map(|samples| (self.loss_sum / samples as f64) as f32)
    }

    /// 样本加权命中率；无样本为 `None`。
    fn mean_accuracy(&self) -> Option<f32> {
        self.nonzero_samples()
            .map(|samples| self.correct as f32 / samples as f32)
    }

    /// 累计样本数（大于 0 时才参与加权均值）。
    fn nonzero_samples(&self) -> Option<u64> {
        (self.samples > 0).then_some(self.samples)
    }
}

/// 指标累积表：训练与评估两条序列各自逐点入账 + 加权聚合计数。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricsBoard {
    train: Ledger,
    eval: Ledger,
}

impl MetricsBoard {
    /// 空看板。
    pub fn new() -> Self {
        Self::default()
    }

    /// 记一条训练步指标；累计样本、加权 loss 与命中条数，并刷新最优 loss。
    ///
    /// # Errors
    ///
    /// 样本数为 0、loss 非有限数，或命中条数超过样本数时返回 [`Code::Invalid`]。
    pub fn record(&mut self, point: MetricPoint) -> Result<()> {
        self.train.push(point)
    }

    /// 记一条评估步指标：独立账表，不并入训练序列的样本数、加权均值与最优 loss。
    ///
    /// # Errors
    ///
    /// 样本数为 0、loss 非有限数，或命中条数超过样本数时返回 [`Code::Invalid`]。
    pub fn record_eval(&mut self, point: MetricPoint) -> Result<()> {
        self.eval.push(point)
    }

    /// 已记账的训练指标点（按入账顺序）。
    pub fn points(&self) -> &[MetricPoint] {
        &self.train.points
    }

    /// 已记账的评估指标点（与训练序列分开）。
    pub fn eval_points(&self) -> &[MetricPoint] {
        &self.eval.points
    }

    /// 已记账的点数。
    pub fn len(&self) -> usize {
        self.train.points.len()
    }

    /// 看板是否还没有任何指标。
    pub fn is_empty(&self) -> bool {
        self.train.points.is_empty()
    }

    /// 最近一条指标；空看板为 `None`。
    pub fn latest(&self) -> Option<&MetricPoint> {
        self.train.points.last()
    }

    /// 样本加权平均 loss；无样本为 `None`。
    pub fn mean_loss(&self) -> Option<f32> {
        self.train.mean_loss()
    }

    /// 样本加权命中率；无样本为 `None`。
    pub fn mean_accuracy(&self) -> Option<f32> {
        self.train.mean_accuracy()
    }

    /// 各步最低训练 loss；空看板为 `None`。
    pub fn best_loss(&self) -> Option<f32> {
        self.train.best
    }

    /// 各步最低评估 loss；空看板为 `None`。
    pub fn best_eval_loss(&self) -> Option<f32> {
        self.eval.best
    }

    /// 渲染最近一条为可读行；空看板返回占位说明。
    pub fn render(&self) -> String {
        match self.latest() {
            Some(point) => format!(
                "step {:>4} | loss {:.4} | acc {:5.1}% | n={}",
                point.step,
                point.loss,
                point.accuracy() * 100.0,
                point.samples
            ),
            None => "step    - | loss     - | acc     -% | n=0".to_string(),
        }
    }
}

/// 一条指标的入参校验：样本非零、loss 有限、命中条数不超过样本数。
fn validate(point: &MetricPoint) -> Result<()> {
    if point.samples == 0 {
        return Err(Error::new(Code::Invalid, "指标样本数不能为 0").with(METRICS_SITE));
    }
    if !point.loss.is_finite() {
        let message = format!("loss {} 不是有限数（NaN / Infinity 不进看板）", point.loss);
        return Err(Error::new(Code::Invalid, message).with(METRICS_SITE));
    }
    if point.correct > point.samples as u64 {
        let message = format!("命中条数 {} 超过样本数 {}", point.correct, point.samples);
        return Err(Error::new(Code::Invalid, message).with(METRICS_SITE));
    }
    Ok(())
}

// ---- 训练样本：Batch / Dataset / Chunks ----

/// 批几何：`rows`、`cols` 非零且乘积不溢出，返回每批 token 数（`rows * cols`）。
fn stride(rows: usize, cols: usize, site: &str) -> Result<usize> {
    if rows == 0 {
        return Err(Error::new(Code::Invalid, "批行数不能为 0").with(site));
    }
    if cols == 0 {
        return Err(Error::new(Code::Invalid, "列宽不能为 0").with(site));
    }
    rows.checked_mul(cols).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            format!("rows * cols 溢出（rows = {rows}，cols = {cols}）"),
        )
        .with(site)
    })
}

/// 一个批：`rows x cols` 个 token，行主序；借用调用方的缓冲，不拥有。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch<'a> {
    tokens: &'a [u32],
    rows: usize,
    cols: usize,
}

impl<'a> Batch<'a> {
    /// 构造：`rows` 与 `cols` 都非零，且 `tokens.len() == rows * cols`。
    pub fn new(tokens: &'a [u32], rows: usize, cols: usize) -> Result<Self> {
        // 行列非零与乘积不溢出都由同一份几何校验给出，`Chunks::new` 走的是同一条。
        let expected = stride(rows, cols, "train::Batch::new")?;
        if tokens.len() != expected {
            return Err(Error::new(
                Code::Invalid,
                format!("token 数 {} 与 {rows} x {cols} 不符", tokens.len()),
            )
            .with("train::Batch::new"));
        }
        Ok(Self { tokens, rows, cols })
    }

    /// 不变量已由调用方在构造期保证（`tokens.len() == rows * cols` 且两者非零）时使用。
    fn exact(tokens: &'a [u32], rows: usize, cols: usize) -> Self {
        debug_assert_eq!(
            tokens.len(),
            rows * cols,
            "内部批次的长度必须等于 rows * cols"
        );
        Self { tokens, rows, cols }
    }

    /// token 序列（行主序）。
    pub fn tokens(&self) -> &'a [u32] {
        self.tokens
    }

    /// 行数。
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 列数。
    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// 数据集：按批取样本。三个计数语义互不重叠：列宽、完整批总数、还没取走的完整批数。
pub trait Dataset<'a> {
    /// 列宽（每行多少 token）。
    fn cols(&self) -> usize;

    /// 完整批的**总数**——不随消费推进。
    fn batches(&self) -> usize;

    /// **还没取走**的完整批数。
    fn remaining_batches(&self) -> usize;

    /// 取下一批；`None` 当且仅当剩余 token 不足一整批（尾部残批丢弃，不补齐）。
    fn next_batch(&mut self) -> Option<Batch<'a>>;
}

/// 最简实现：一段连续 token 流按固定几何切批。
#[derive(Debug, Clone)]
pub struct Chunks<'a> {
    tokens: &'a [u32],
    rows: usize,
    cols: usize,
    stride: usize,
    /// 完整批总数：`tokens.len() / stride`，构造期算好，不随消费推进。
    batches: usize,
    /// 已取走的批数（不是 token 数：游标永远是 `stride` 的整数倍，那是不变量的另一半）。
    taken: usize,
}

impl<'a> Chunks<'a> {
    /// 由 token 流、每批行数与列宽构造：三者任一不合法都报 `Code::Invalid`。
    pub fn new(tokens: &'a [u32], rows: usize, cols: usize) -> Result<Self> {
        let stride = stride(rows, cols, "train::Chunks::new")?;
        Ok(Self {
            tokens,
            rows,
            cols,
            stride,
            batches: tokens.len() / stride,
            taken: 0,
        })
    }
}

/// 迭代器：`next` 就是 [`Dataset::next_batch`]，所以「还剩几条」只有一个来源。
impl<'a> Iterator for Chunks<'a> {
    type Item = Batch<'a>;

    fn next(&mut self) -> Option<Batch<'a>> {
        self.next_batch()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_batches();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Chunks<'_> {}

impl core::iter::FusedIterator for Chunks<'_> {}

impl<'a> Dataset<'a> for Chunks<'a> {
    fn cols(&self) -> usize {
        self.cols
    }

    fn batches(&self) -> usize {
        self.batches
    }

    fn remaining_batches(&self) -> usize {
        self.batches - self.taken
    }

    fn next_batch(&mut self) -> Option<Batch<'a>> {
        if self.taken == self.batches {
            return None;
        }
        // `taken < batches == len / stride` 蕴含 `end <= len`：越界在结构上不可能，因此这里不需要
        // 一条只能表示「取完了」的 `?`（那会把「内部不变量破了」重新折成 `None`）。
        let start = self.taken * self.stride;
        let end = start + self.stride;
        self.taken += 1;
        Some(Batch::exact(&self.tokens[start..end], self.rows, self.cols))
    }
}

#[cfg(test)]
#[path = "../tests/learner_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/samples_tests.rs"]
mod samples_tests;

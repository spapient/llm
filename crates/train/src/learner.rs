//! 独立训练编排器：把数据集遍历、前向评估、指标入账与优化步收拢到一个驱动对象。
//!
//! [`Session`](crate::Session) 只持有会话状态与单步接缝，不关心数据集如何分批；本模块补上
//! 「遍历一批 → 前向 → 记 loss/accuracy → 回传梯度 → 优化步」的循环编排。模型拓扑不在这里硬编码：
//! 前向装配由闭包注入，梯度先由 [`Session::backward`] 按反向计划自动回算——
//!
//! - `forward`：装载本批输入并算出 `logits` 与 `loss` 两个终点变量；
//! - `backward`：自动回算之后的覆盖/追加口，把额外梯度装入或累加进会话缓冲。
//!
//! 标签按标准 next-token 设定由序列右移派生：位置 `(r, c)` 的标签是 `tokens[r, c+1]`；每行末列
//! 没有后继，剔除出评估集。logits 必须按同一行主序位置展开成 `[positions, vocab]`。

use diagnostic::{Code, Error, Result};

use autograd::Var;
use data::Dataset;
use shard::Collective;

use crate::distill::Distill;
use crate::metrics::{MetricPoint, MetricsBoard};
use crate::objective::{Outcome, classify, distill, grpo};
use crate::optimizer::Optimizer;
use crate::rl::Grpo;
use crate::{Kernel, Session};

/// 本模块诊断站点。
pub(crate) const SITE: &str = "train::Learner";

/// 一次前向的两个评估终点：`logits` 用于算 NLL 与命中率，`loss` 用于反传计划对账。
///
/// 默认度量是分类 NLL；`metric` 为 `Some` 时按蒸馏/RL 度量入账（见 [`Metric`]）。
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
///
/// 蒸馏只多一个教师 logits 变量；RL 额外带当前/行为策略 log-prob、advantage 与参考 KL 四个变量
/// （逐 token 对齐），配置随度量一起由前向闭包给出。
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
///
/// `C` 是通信后端，默认 [`Remote`](crate::Remote) 走单进程恒等回路；接分布式后端时显式传入，
/// 编排器本体对后端透明——梯度同步发生在 [`Session::optimize`] 内部，本层不感知。
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

    /// 会话状态（可变：登记参数、装权重、回载快照等装配动作走这里）。
    pub fn session_mut(&mut self) -> &mut Session<K, O, C> {
        &mut self.session
    }

    /// 指标看板（只读）。
    pub fn board(&self) -> &MetricsBoard {
        &self.board
    }

    /// 拆解回收会话：快照导出/回载等需要裸会话的场景取走会话本体。
    pub fn into_session(self) -> Session<K, O, C> {
        self.session
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
    ///
    /// 回传段见 [`Self::backward_with`]；`backward` 闭包在自动回算之后保留为覆盖/追加口。
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
        let view = self.session.buffers().value(var).ok_or_else(|| {
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

    /// 按度量计算 loss 与命中并记进训练看板。
    fn record(&mut self, prepared: &Prepared) -> Result<()> {
        let outcome = self.compute_metric(prepared)?;
        let point = MetricPoint {
            step: self.session.step(),
            samples: prepared.labels.len(),
            loss: outcome.loss,
            correct: outcome.correct,
        };
        self.board.record(point).map_err(|error| error.with(SITE))
    }

    /// 按度量计算 loss 与命中并记进评估独立账表；步号沿用当前训练步，仅作对账标记。
    fn record_eval(&mut self, prepared: &Prepared) -> Result<()> {
        let outcome = self.compute_metric(prepared)?;
        let point = MetricPoint {
            step: self.session.step(),
            samples: prepared.labels.len(),
            loss: outcome.loss,
            correct: outcome.correct,
        };
        self.board
            .record_eval(point)
            .map_err(|error| error.with(SITE))
    }
}

/// next-token 标签：`(r, c)` 的标签取 `tokens[r, c+1]`；行末列无后继，剔除。
///
/// 位置顺序与前向闭包产出的 logits 行主序一一对应；一列宽（`cols == 1`）没有可预测位置，
/// 返回空表，由下游损失函数拒绝。
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

#[cfg(test)]
mod tests {
    use crate::{
        Data, Distill, ForwardOut, Grpo, Learner, Metric, Reference, Session, Sgd, TrainConfig,
    };
    use autograd::{Sizes, Store, Tape, Var};
    use data::Chunks;
    use diagnostic::{Code, Error, Result};
    use nn::ParamRole;
    use tensor::{DType, Shape};

    /// 夹具参数宽度。
    const N: usize = 4;
    /// 词表宽度。
    const VOCAB: usize = 3;

    /// 造一张夹具图：`loss = add(p, x)`，`logits` 是独立叶子用于评估；返回图、形状表与四个变量。
    fn fixture() -> (Tape, Sizes, Var, Var, Var, Var) {
        let mut tape = Tape::new();
        let p = tape.parameter();
        let x = tape.leaf();
        let logits = tape.leaf();
        let loss = tape.record("add_f32", &[p, x]).expect("记录");
        let mut sizes = Sizes::new(tape.vars());
        sizes.set(p, Shape::new([N]), DType::F32);
        sizes.set(x, Shape::new([N]), DType::F32);
        sizes.set(logits, Shape::new([VOCAB]), DType::F32);
        sizes.set(loss, Shape::new([N]), DType::F32);
        (tape, sizes, p, x, logits, loss)
    }

    /// 注册参数 `p` 后包成 learner：四个测试共享的会话装配。
    fn make_learner(
        tape: Tape,
        sizes: Sizes,
        p: Var,
        config: TrainConfig,
    ) -> Result<Learner<Reference, Sgd>> {
        let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config)?;
        session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
        Learner::new(session, VOCAB)
    }

    /// 夹具前向：装 `x` 与 `logits`，跑 `loss = add(p, x)`，plan 后返回 logits + loss。
    fn forward(
        session: &mut Session<Reference, Sgd>,
        p: Var,
        x: Var,
        logits: Var,
        loss: Var,
        logits_values: &[f32],
    ) -> Result<ForwardOut> {
        session.buffers().load(x, Data::F32(vec![0.1; N]))?;
        session
            .buffers()
            .load(logits, Data::F32(logits_values.to_vec()))?;
        session.buffers().compute("add_f32", &[p, x], loss)?;
        session.plan(loss)?;
        Ok(ForwardOut {
            logits,
            loss,
            metric: None,
        })
    }

    /// 夹具回传：给参数 `p` 塞一份全 1 梯度。
    fn load_one_gradient(session: &mut Session<Reference, Sgd>, p: Var) -> Result<()> {
        session.load_gradient(p, &[1.0; N])
    }

    /// 跑一个"带度量"的对齐批：装配 learner 后，`metric` 在装完 `x`/`logits` 后完成度量专属装载
    /// 并返回挂在 `ForwardOut.metric` 上的度量（蒸馏 / GRPO 各自自建叶子）。返回入账后的 learner。
    fn run_metric_batch(
        tape: Tape,
        sizes: Sizes,
        p: Var,
        x: Var,
        logits: Var,
        loss: Var,
        metric: impl Fn(&mut Session<Reference, Sgd>) -> Result<Metric>,
    ) -> Result<Learner<Reference, Sgd>> {
        let config = TrainConfig::new(1e-2, 0, 3, None, 1)?;
        let mut learner = make_learner(tape, sizes, p, config)?;
        let tokens = [0u32, 1];
        let mut chunks = Chunks::new(&tokens, 1, 2)?;
        learner.run(
            &mut chunks,
            1,
            |session, _tokens| {
                session.buffers().load(x, Data::F32(vec![0.1; N]))?;
                session
                    .buffers()
                    .load(logits, Data::F32(vec![0.0, 10.0, 0.0]))?;
                let metric = metric(session)?;
                session.buffers().compute("add_f32", &[p, x], loss)?;
                session.plan(loss)?;
                Ok(ForwardOut {
                    logits,
                    loss,
                    metric: Some(metric),
                })
            },
            |session| load_one_gradient(session, p),
        )?;
        Ok(learner)
    }

    /// 跑一个对齐批次：标签取后继 token，argmax 命中标签，看板记一条接近 0 的 loss。
    #[test]
    fn learner_runs_a_batch_and_records_metrics() -> Result<()> {
        let (tape, sizes, p, x, logits, loss) = fixture();
        let tokens = [0u32, 1];
        let config = TrainConfig::new(1e-2, 0, 3, None, 1)?;
        let mut learner = make_learner(tape, sizes, p, config)?;

        // 类别 1 的 logit 顶到 10；后继 token 是 1，argmax 命中。
        let logits_values = vec![0.0, 10.0, 0.0];
        let mut chunks = Chunks::new(&tokens, 1, 2)?;

        learner.run(
            &mut chunks,
            1,
            |session, _tokens| forward(session, p, x, logits, loss, &logits_values),
            |session| load_one_gradient(session, p),
        )?;

        let board = learner.board();
        assert_eq!(board.len(), 1, "一个 micro-batch 记一条指标");
        let point = board.latest().expect("已有指标");
        assert_eq!(point.samples, 1, "一行一列后继位置 = 1 个样本");
        assert_eq!(point.accuracy(), 1.0, "argmax 命中标签");
        assert!(point.loss < 1e-3, "对齐类别 NLL 趋近 0");
        assert_eq!(learner.session().step(), 1, "窗口 1，每批步进一次");
        Ok(())
    }

    /// logits 元素数与「有效位置 × 词表」不符时，编排器在入账前拒绝，不带错值进优化步。
    #[test]
    fn rejects_mismatched_logits_geometry() -> Result<()> {
        let (tape, sizes, p, x, logits, loss) = fixture();
        let tokens = [0u32, 1];
        let config = TrainConfig::default();
        let mut learner = make_learner(tape, sizes, p, config)?;
        let mut chunks = Chunks::new(&tokens, 1, 2)?;

        // 只给 2 个 logit，词表是 3：长度不符。
        let short = vec![0.0, 10.0];
        let error = learner
            .run(
                &mut chunks,
                1,
                |session, _tokens| forward(session, p, x, logits, loss, &short),
                |session| load_one_gradient(session, p),
            )
            .expect_err("logits 长度不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(learner.board().is_empty(), "拒绝前不写指标");
        Ok(())
    }

    /// 一批一条指标：两个 micro-batch 各入账一条，窗口 1 下各步进一次。
    #[test]
    fn records_one_metric_per_batch_across_epochs() -> Result<()> {
        let (tape, sizes, p, x, logits, loss) = fixture();
        let tokens = [0u32, 1, 0, 1];
        let config = TrainConfig::new(1e-2, 0, 4, None, 1)?;
        let mut learner = make_learner(tape, sizes, p, config)?;
        let mut chunks = Chunks::new(&tokens, 1, 2)?;

        learner.run(
            &mut chunks,
            1,
            |session, _tokens| forward(session, p, x, logits, loss, &[0.0, 10.0, 0.0]),
            |session| load_one_gradient(session, p),
        )?;

        assert_eq!(learner.board().len(), 2, "两个 micro-batch 各记一条");
        assert_eq!(learner.session().step(), 2, "窗口 1，两批各步进一次");
        Ok(())
    }

    /// 单列批没有后继可预测位置：入账前拒绝，看板不写入。
    #[test]
    fn rejects_single_column_with_no_successor() -> Result<()> {
        let (tape, sizes, p, x, logits, loss) = fixture();
        let tokens = [5u32];
        let config = TrainConfig::default();
        let mut learner = make_learner(tape, sizes, p, config)?;
        let mut chunks = Chunks::new(&tokens, 1, 1)?;

        let error = learner
            .run(
                &mut chunks,
                1,
                |session, _tokens| forward(session, p, x, logits, loss, &[0.0, 10.0, 0.0]),
                |session| load_one_gradient(session, p),
            )
            .expect_err("单列无可预测位置");
        assert_eq!(error.code(), Code::Invalid);
        assert!(learner.board().is_empty(), "拒绝前不写指标");
        Ok(())
    }

    /// 验证集独立循环：只前向评估入账，不回传梯度、不步进优化器，也不污染训练指标序列。
    #[test]
    fn evaluate_runs_a_validation_pass_without_gradients_or_steps() -> Result<()> {
        let (tape, sizes, p, x, logits, loss) = fixture();
        let train_tokens = [0u32, 1];
        let valid_tokens = [1u32, 0];
        let config = TrainConfig::new(1e-2, 0, 3, None, 1)?;
        let mut learner = make_learner(tape, sizes, p, config)?;

        let mut train = Chunks::new(&train_tokens, 1, 2)?;
        let mut valid = Chunks::new(&valid_tokens, 1, 2)?;

        learner.run(
            &mut train,
            1,
            |session, _tokens| forward(session, p, x, logits, loss, &[0.0, 10.0, 0.0]),
            |session| load_one_gradient(session, p),
        )?;

        // 验证批：标签是 0，而 logits 顶在类别 1；独立账表记一条，训练账表不追加。
        learner.evaluate(&mut valid, |session, _tokens| {
            forward(session, p, x, logits, loss, &[0.0, 10.0, 0.0])
        })?;

        let board = learner.board();
        assert_eq!(board.len(), 1, "训练序列仍是一条，验证不追加");
        assert_eq!(board.eval_points().len(), 1, "验证独立账表一条");
        assert_eq!(learner.session().step(), 1, "验证批不步进优化器");
        assert!(
            learner.session().accumulated_microbatches() == 0,
            "验证后梯度缓冲已清空"
        );

        let eval = board.eval_points().first().expect("已有评估点");
        assert_eq!(eval.samples, 1, "一行一列后继位置 = 1 个样本");
        assert_eq!(eval.accuracy(), 0.0, "验证标签 0 未命中类别 1");
        Ok(())
    }

    /// 前向闭包失败时错误原样透传，未入账任何指标。
    #[test]
    fn forwards_forward_closure_error_without_recording() -> Result<()> {
        let (tape, sizes, p, _x, _logits, _loss) = fixture();
        let tokens = [0u32, 1];
        let config = TrainConfig::default();
        let mut learner = make_learner(tape, sizes, p, config)?;
        let mut chunks = Chunks::new(&tokens, 1, 2)?;

        let error =
            learner
                .run(
                    &mut chunks,
                    1,
                    |_session, _tokens| {
                        Err(Error::new(Code::Internal, "夹具前向故意失败").with("test"))
                    },
                    |_session| Ok(()),
                )
                .expect_err("前向错误透传");
        assert_eq!(error.code(), Code::Internal);
        assert!(learner.board().is_empty(), "前向失败前不写指标");
        Ok(())
    }

    /// 蒸馏度量：`ForwardOut.metric` 带 teacher 时看板记蒸馏损失（含 KL），命中仍按硬标签。
    #[test]
    fn records_distillation_loss_when_metric_is_distill() -> Result<()> {
        let (mut tape, mut sizes, p, x, logits, loss) = fixture();
        let teacher = tape.leaf();
        sizes.set(teacher, Shape::new([VOCAB]), DType::F32);

        let learner = run_metric_batch(tape, sizes, p, x, logits, loss, |session| {
            session
                .buffers()
                .load(teacher, Data::F32(vec![1.0; VOCAB]))?;
            Ok(Metric::Distill {
                teacher,
                config: Distill::new(2.0, 0.5),
            })
        })?;

        let point = learner.board().latest().expect("已有指标");
        assert!(point.loss > 0.0, "教师均匀分布引入 KL>0，蒸馏损失应为正");
        assert_eq!(point.accuracy(), 1.0, "argmax 仍命中硬标签");
        Ok(())
    }

    /// RL 度量：GRPO 组合损失入账，命中恒 0（RL 无分类命中）。
    #[test]
    fn records_grpo_loss_when_metric_is_grpo() -> Result<()> {
        let (mut tape, mut sizes, p, x, logits, loss) = fixture();
        let (new_logp, old_logp, advantages, ref_kl) =
            (tape.leaf(), tape.leaf(), tape.leaf(), tape.leaf());
        sizes.set(new_logp, Shape::new([1]), DType::F32);
        sizes.set(old_logp, Shape::new([1]), DType::F32);
        sizes.set(advantages, Shape::new([1]), DType::F32);
        sizes.set(ref_kl, Shape::new([1]), DType::F32);

        let learner = run_metric_batch(tape, sizes, p, x, logits, loss, |session| {
            session.buffers().load(new_logp, Data::F32(vec![-0.1]))?;
            session.buffers().load(old_logp, Data::F32(vec![-0.2]))?;
            session.buffers().load(advantages, Data::F32(vec![1.0]))?;
            session.buffers().load(ref_kl, Data::F32(vec![0.05]))?;
            Ok(Metric::Grpo {
                new_logp,
                old_logp,
                advantages,
                ref_kl,
                config: Grpo::new(0.2, 0.1),
            })
        })?;

        let point = learner.board().latest().expect("已有指标");
        assert!(point.loss < 0.0, "正 advantage 改善策略，GRPO 损失为负");
        assert_eq!(point.correct, 0, "RL 度量无分类命中");
        Ok(())
    }
}

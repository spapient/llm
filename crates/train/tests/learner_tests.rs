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

    let error = learner
        .run(
            &mut chunks,
            1,
            |_session, _tokens| Err(Error::new(Code::Internal, "夹具前向故意失败").with("test")),
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
    let mut tape = Tape::new();
    let p = tape.parameter();
    let x = tape.leaf();
    let logits = tape.leaf();
    let teacher = tape.leaf();
    let loss = tape.record("add_f32", &[p, x]).expect("记录");
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(logits, Shape::new([VOCAB]), DType::F32);
    sizes.set(teacher, Shape::new([VOCAB]), DType::F32);
    sizes.set(loss, Shape::new([N]), DType::F32);

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
    let mut tape = Tape::new();
    let p = tape.parameter();
    let x = tape.leaf();
    let logits = tape.leaf();
    let (new_logp, old_logp, advantages, ref_kl) =
        (tape.leaf(), tape.leaf(), tape.leaf(), tape.leaf());
    let loss = tape.record("add_f32", &[p, x]).expect("记录");
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(p, Shape::new([N]), DType::F32);
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(logits, Shape::new([VOCAB]), DType::F32);
    sizes.set(new_logp, Shape::new([1]), DType::F32);
    sizes.set(old_logp, Shape::new([1]), DType::F32);
    sizes.set(advantages, Shape::new([1]), DType::F32);
    sizes.set(ref_kl, Shape::new([1]), DType::F32);
    sizes.set(loss, Shape::new([N]), DType::F32);

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

use crate::{Data, ForwardOut, Learner, Reference, Session, Sgd, TrainConfig};
use autograd::{Sizes, Store, Tape, Var};
use data::Chunks;
use diagnostic::{Code, Result};
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

/// 跑一个对齐批次：标签取后继 token，argmax 命中标签，看板记一条接近 0 的 loss。
#[test]
fn learner_runs_a_batch_and_records_metrics() -> Result<()> {
    let (tape, sizes, p, x, logits, loss) = fixture();
    let tokens = [0u32, 1];
    let config = TrainConfig::new(1e-2, 0, 3, None, 1)?;
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config)?;
    session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
    let mut learner = Learner::new(session, VOCAB)?;

    // 类别 1 的 logit 顶到 10；后继 token 是 1，argmax 命中。
    let logits_values = vec![0.0, 10.0, 0.0];
    let mut chunks = Chunks::new(&tokens, 1, 2)?;

    learner.run(
        &mut chunks,
        1,
        |session, _tokens| {
            session.buffers().load(x, Data::F32(vec![0.1; N]))?;
            session
                .buffers()
                .load(logits, Data::F32(logits_values.clone()))?;
            session.buffers().compute("add_f32", &[p, x], loss)?;
            session.plan(loss)?;
            Ok(ForwardOut { logits, loss })
        },
        |session| {
            session.load_gradient(p, &[1.0; N])?;
            Ok(())
        },
    )?;

    let board = learner.board();
    assert_eq!(board.len(), 1, "一个 micro-batch 记一条指标");
    let point = board.latest().expect("已有指标");
    assert_eq!(point.samples, 1, "一行一列后继位置 = 1 个样本");
    assert_eq!(point.accuracy, 1.0, "argmax 命中标签");
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
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config)?;
    session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
    let mut learner = Learner::new(session, VOCAB)?;
    let mut chunks = Chunks::new(&tokens, 1, 2)?;

    // 只给 2 个 logit，词表是 3：长度不符。
    let short = vec![0.0, 10.0];
    let error = learner
        .run(
            &mut chunks,
            1,
            |session, _tokens| {
                session.buffers().load(x, Data::F32(vec![0.1; N]))?;
                session.buffers().load(logits, Data::F32(short.clone()))?;
                session.buffers().compute("add_f32", &[p, x], loss)?;
                session.plan(loss)?;
                Ok(ForwardOut { logits, loss })
            },
            |session| {
                session.load_gradient(p, &[1.0; N])?;
                Ok(())
            },
        )
        .expect_err("logits 长度不符");
    assert_eq!(error.code(), Code::Invalid);
    assert!(learner.board().is_empty(), "拒绝前不写指标");
    Ok(())
}

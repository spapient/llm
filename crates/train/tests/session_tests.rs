use super::*;

use crate::{Data, Reference, Sgd};
use autograd::{Sizes, Store, Tape, Var};
use data::{Chunks, Dataset};
use diagnostic::{Code, Result};
use nn::ParamRole;
use tensor::{DType, Shape};

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

/// 夹具语料按批跑通指定步数：前向 → 计划分类终点 → 梯度缓冲 → 优化器步，loss 逐步入账。
#[test]
fn a_fixture_dataset_runs_the_declared_steps() {
    let (tape, sizes, p, x, loss) = fixture();
    let tokens: Vec<u32> = (0..12).collect();
    let mut chunks = Chunks::new(&tokens, 1, N).expect("批 1 x 4");
    let config = TrainConfig::new(1e-2, 0, 3, None, 1).expect("配置合法");
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config).expect("会话");
    session
        .register_parameter(p, ParamRole::Trainable, vec![1.0; N])
        .expect("登记参数");

    let mut steps = 0;
    while let Some(batch) = chunks.next_batch() {
        session
            .accept_batch(batch.rows(), batch.cols(), batch.tokens().len())
            .expect("批形状往返一致");
        let inputs: Vec<f32> = batch
            .tokens()
            .iter()
            .map(|token| *token as f32 * 0.1)
            .collect();
        session
            .buffers()
            .load(x, Data::F32(inputs))
            .expect("装载输入");
        session
            .buffers()
            .compute("add_f32", &[p, x], loss)
            .expect("前向可算");
        session.plan(loss).expect("参数可达");
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
    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config).expect("会话");
    session
        .register_parameter(p, ParamRole::Trainable, vec![1.0; N])
        .expect("登记参数");

    // 第一批：梯度 1.0/元素累加进窗口；未满 2，参数不动。
    let batch = chunks.next_batch().expect("第一批");
    session
        .accept_batch(batch.rows(), batch.cols(), batch.tokens().len())
        .expect("批形状往返一致");
    let inputs: Vec<f32> = batch
        .tokens()
        .iter()
        .map(|token| *token as f32 * 0.1)
        .collect();
    session
        .buffers()
        .load(x, Data::F32(inputs))
        .expect("装载输入");
    session
        .buffers()
        .compute("add_f32", &[p, x], loss)
        .expect("前向可算");
    session.plan(loss).expect("参数可达");
    session
        .accumulate_gradient(p, &[1.0; N])
        .expect("累加第一批");
    session.optimize().expect("第一批不步进");
    assert_eq!(session.accumulated_microbatches(), 1);
    assert_eq!(session.slots()[0].params(), &[1.0; N], "窗口未满不更新参数");

    // 第二批：再累加 1.0，汇总梯度 2.0/元素；窗口满，步进一次。
    let batch = chunks.next_batch().expect("第二批");
    session
        .accept_batch(batch.rows(), batch.cols(), batch.tokens().len())
        .expect("批形状往返一致");
    let inputs: Vec<f32> = batch
        .tokens()
        .iter()
        .map(|token| *token as f32 * 0.1)
        .collect();
    session
        .buffers()
        .load(x, Data::F32(inputs))
        .expect("装载输入");
    session
        .buffers()
        .compute("add_f32", &[p, x], loss)
        .expect("前向可算");
    session.plan(loss).expect("参数可达");
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

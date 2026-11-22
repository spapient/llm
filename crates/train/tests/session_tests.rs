use super::*;

use crate::{Chunks, Data, Dataset, Reference, Sgd};
use autograd::{Sizes, Store, Tape, Var};
use diagnostic::{Code, Result};
use nn::ParamRole;
use snapshot::{Facts, GradAccum, Moments, SlotMoments, Snapshot, Ties};
use tensor::{DType, Shape};
use weights::Header;

use crate::support;

/// 夹具向量宽度。
const N: usize = 4;

/// `loss = add_f32(p, x)`：`p` 是梯度终点参数，`x` 是常量输入叶子。
fn fixture() -> (Tape, Sizes, Var, Var, Var) {
    support::add_graph(N)
}

/// 两层共享权重夹具：`h = add(x, w_source)`，`t = add(h, w_tied)`，`loss = add(t, u)`；
/// `w_source` / `w_tied` 待注册权重绑定，`u` 是未绑定参数，三层都可达 loss。
#[allow(clippy::type_complexity)]
fn two_layer_fixture() -> (Tape, Sizes, Var, Var, Var, Var, Var, Var, Var) {
    let mut tape = Tape::new();
    let w_source = tape.parameter();
    let w_tied = tape.parameter();
    let u = tape.parameter();
    let x = tape.leaf();
    let h = tape.record("add_f32", &[x, w_source]).expect("第一层");
    let t = tape.record("add_f32", &[h, w_tied]).expect("第二层");
    let loss = tape.record("add_f32", &[t, u]).expect("loss");
    let mut sizes = Sizes::new(tape.vars());
    for var in [w_source, w_tied, u, x, h, t, loss] {
        sizes.set(var, Shape::new([N]), DType::F32);
    }
    (tape, sizes, w_source, w_tied, u, x, h, t, loss)
}

/// 装配一个裸会话：优化器用默认 SGD，配置由用例给定。
fn open_session(tape: Tape, sizes: Sizes, config: TrainConfig) -> Session<Reference, Sgd> {
    Session::new(tape, sizes, Reference, Sgd::new(), config).expect("会话")
}

/// 接收一批 token、装输入、前向到终点并登记计划；批形状固定 1 × N。
fn forward_fixture(
    session: &mut Session<Reference, Sgd>,
    tokens: &[u32],
    op_inputs: &[Var],
    x: Var,
    loss: Var,
) -> Result<()> {
    session.accept_batch(1, N, tokens.len())?;
    let inputs: Vec<f32> = tokens.iter().map(|token| *token as f32 * 0.1).collect();
    session.bufs().load(x, Data::F32(inputs))?;
    session.bufs().compute_step("add_f32", op_inputs, loss)?;
    session.plan(loss)?;
    Ok(())
}

/// 登记两层夹具的三个参数并绑定源/回填槽；初值沿用夹具口径。
fn register_two_layer(
    session: &mut Session<Reference, Sgd>,
    w_source: Var,
    w_tied: Var,
    u: Var,
) -> Result<()> {
    session.register_parameter(w_source, ParamRole::Trainable, vec![1.0; N])?;
    session.register_parameter(w_tied, ParamRole::Trainable, vec![0.0; N])?;
    session.register_parameter(u, ParamRole::Trainable, vec![2.0; N])?;
    session.register_tie(w_source, w_tied)?;
    Ok(())
}

/// 两层夹具前向：常量输入 → 第一层 → 第二层 → 未绑定参数终点，并登记计划终点。
#[allow(clippy::too_many_arguments)]
fn forward_two_layer(
    session: &mut Session<Reference, Sgd>,
    x: Var,
    w_source: Var,
    h: Var,
    w_tied: Var,
    t: Var,
    u: Var,
    loss: Var,
) -> Result<()> {
    session.bufs().load(x, Data::F32(vec![0.1; N]))?;
    session.bufs().compute_step("add_f32", &[x, w_source], h)?;
    session.bufs().compute_step("add_f32", &[h, w_tied], t)?;
    session.bufs().compute_step("add_f32", &[t, u], loss)?;
    session.plan(loss)?;
    Ok(())
}

/// 夹具语料按批跑通指定步数：前向 → 计划分类终点 → 梯度缓冲 → 优化器步，loss 逐步入账。
#[test]
fn a_fixture_dataset_runs_the_declared_steps() {
    let (tape, sizes, p, x, loss) = fixture();
    let tokens: Vec<u32> = (0..12).collect();
    let mut chunks = Chunks::new(&tokens, 1, N).expect("批 1 x 4");
    let config = TrainConfig::new(1e-2, 0, 3, None, 1).expect("配置合法");
    let mut session = open_session(tape, sizes, config);
    session
        .register_parameter(p, ParamRole::Trainable, vec![1.0; N])
        .expect("登记参数");

    let mut steps = 0;
    while let Some(batch) = chunks.next_batch() {
        forward_fixture(&mut session, batch.tokens(), &[p, x], x, loss).expect("前向");
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
    let mut session = open_session(tape, sizes, config);
    session
        .register_parameter(p, ParamRole::Trainable, vec![1.0; N])
        .expect("登记参数");

    // 第一批：梯度 1.0/元素累加进窗口；未满 2，参数不动。
    let batch = chunks.next_batch().expect("第一批");
    forward_fixture(&mut session, batch.tokens(), &[p, x], x, loss).expect("前向");
    session
        .accumulate_gradient(p, &[1.0; N])
        .expect("累加第一批");
    session.optimize().expect("第一批不步进");
    assert_eq!(session.accumulated_microbatches(), 1);
    assert_eq!(session.slots()[0].params(), &[1.0; N], "窗口未满不更新参数");

    // 第二批：再累加 1.0，汇总梯度 2.0/元素；窗口满，步进一次。
    let batch = chunks.next_batch().expect("第二批");
    forward_fixture(&mut session, batch.tokens(), &[p, x], x, loss).expect("前向");
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

/// 两层真实装配夹具：源/回填槽经若干 micro-batch 各自更新分化后，回填把 tied 整块复制成
/// source；未绑定参数不被改动。
#[test]
fn tied_parameter_copies_source_across_a_two_layer_fixture() -> Result<()> {
    let (tape, sizes, w_source, w_tied, u, x, h, t, loss) = two_layer_fixture();
    let config = TrainConfig::new(1e-2, 0, 2, None, 1)?;
    let mut session = open_session(tape, sizes, config);
    register_two_layer(&mut session, w_source, w_tied, u)?;

    // 两个 micro-batch 各走一步：源/回填槽按自身梯度分化，u 梯度为 0 保持不动。
    for (source_grad, tied_grad) in [(1.0_f32, 2.0_f32), (0.5, 2.0)] {
        forward_two_layer(&mut session, x, w_source, h, w_tied, t, u, loss)?;
        session.accumulate_gradient(w_source, &[source_grad; N])?;
        session.accumulate_gradient(w_tied, &[tied_grad; N])?;
        session.accumulate_gradient(u, &[0.0; N])?;
        session.optimize()?;
    }

    let source_after_step = session.slots()[0].params().to_vec();
    let u_before = session.slots()[2].params().to_vec();
    session.backfill_ties()?;
    assert_eq!(
        session.slots()[1].params(),
        source_after_step,
        "回填槽逐元素取源槽值"
    );
    assert_eq!(
        session.slots()[2].params(),
        u_before,
        "未绑定参数不被回填改动"
    );
    Ok(())
}

/// 会话窗口状态入快照：多次累积后的梯度总和、micro-batch 计数与绑定表写进快照再读回，
/// 逐字段相等；空累积窗口路径由快照侧用例覆盖。
#[test]
fn session_window_state_round_trips_through_snapshot() -> Result<()> {
    let (tape, sizes, w_source, w_tied, u, x, h, t, loss) = two_layer_fixture();
    let config = TrainConfig::new(1e-2, 0, 4, None, 4)?;
    let mut session = open_session(tape, sizes, config);
    register_two_layer(&mut session, w_source, w_tied, u)?;

    // 三个 micro-batch 累积进同一窗口（窗口 4，未满不步进）。
    let source_batches = [1.0_f32, 0.5, 0.25];
    let tied_batches = [2.0_f32, 1.0, 0.5];
    for batch in 0..3 {
        forward_two_layer(&mut session, x, w_source, h, w_tied, t, u, loss)?;
        session.accumulate_gradient(w_source, &[source_batches[batch]; N])?;
        session.accumulate_gradient(w_tied, &[tied_batches[batch]; N])?;
        session.accumulate_gradient(u, &[0.0; N])?;
        session.optimize()?;
    }
    assert_eq!(session.accumulated_microbatches(), 3);

    let facts = Facts::new(session.step(), 1, 2, 1, N)?;
    let snapshot = Snapshot::new_with_all(
        facts,
        Header::new(Vec::new(), 0)?,
        Moments::empty(),
        session.accum_snapshot(),
        session.ties_snapshot(),
    );
    let restored = Snapshot::decode(&snapshot.encode()?)?;

    assert_eq!(restored.accum().count(), 3, "窗口批数往返一致");
    assert_eq!(restored.ties().pairs(), &[(0, 1)], "绑定关系往返一致");
    assert_eq!(restored.accum(), snapshot.accum(), "累积梯度逐槽往返一致");
    assert_eq!(restored.ties(), snapshot.ties(), "绑定表往返一致");

    // 显式对拍三批总和：源槽 1.75/元素，回填槽 3.5/元素，未绑定槽零梯度。
    let slots = restored.accum().slots();
    assert_eq!(slots.len(), 3);
    assert!(slots[0].iter().all(|g| (g - 1.75).abs() < 1e-6));
    assert!(slots[1].iter().all(|g| (g - 3.5).abs() < 1e-6));
    assert!(slots[2].iter().all(|g| *g == 0.0));
    Ok(())
}

/// 会话状态快照回载：源会话窗口汇总/绑定表/步号写进快照，回载到另一个同装配会话后
/// accum / ties / 步号逐字段一致；矩状态走优化器 restore（SGD 对应空表）。
#[test]
fn session_state_round_trips_through_restore_snapshot() -> Result<()> {
    let (tape, sizes, w_source, w_tied, u, x, h, t, loss) = two_layer_fixture();
    let config = TrainConfig::new(1e-2, 0, 4, None, 4)?;
    let mut source = open_session(tape, sizes, config);
    register_two_layer(&mut source, w_source, w_tied, u)?;

    // 三个 micro-batch 累积进同一窗口（窗口 4，未满不步进）。
    let source_batches = [1.0_f32, 0.5, 0.25];
    let tied_batches = [2.0_f32, 1.0, 0.5];
    for batch in 0..3 {
        forward_two_layer(&mut source, x, w_source, h, w_tied, t, u, loss)?;
        source.accumulate_gradient(w_source, &[source_batches[batch]; N])?;
        source.accumulate_gradient(w_tied, &[tied_batches[batch]; N])?;
        source.accumulate_gradient(u, &[0.0; N])?;
        source.optimize()?;
    }

    let facts = Facts::new(5, 1, 2, 1, N)?;
    let snapshot = Snapshot::new_with_all(
        facts,
        Header::new(Vec::new(), 0)?,
        Moments::empty(),
        source.accum_snapshot(),
        source.ties_snapshot(),
    );

    // 目标会话：同图同槽位顺序装配，初始窗口为空；回载不重建 tape / 槽位 / 权重。
    let (tape, sizes, w_source, w_tied, u, _x, _h, _t, _loss) = two_layer_fixture();
    let mut target = open_session(tape, sizes, TrainConfig::new(1e-2, 0, 4, None, 4)?);
    register_two_layer(&mut target, w_source, w_tied, u)?;
    target.restore_snapshot(&snapshot)?;

    assert_eq!(target.step(), 5, "步号回载一致");
    assert_eq!(
        target.accumulated_microbatches(),
        source.accumulated_microbatches(),
        "窗口批数回载一致"
    );
    assert_eq!(
        target.accum_snapshot(),
        *snapshot.accum(),
        "累积窗口逐槽回载"
    );
    assert_eq!(target.ties(), snapshot.ties().pairs(), "绑定表回载一致");
    Ok(())
}

/// 回载对账：累积槽位数错位、槽长度不符、ties 编号越界、矩槽位数/长度错位，一律
/// [`Code::Invalid`]，拒绝前不改写会话状态。
#[test]
fn restore_snapshot_rejects_mismatched_state() -> Result<()> {
    let (tape, sizes, w_source, w_tied, u, _x, _h, _t, _loss) = two_layer_fixture();
    let mut target = open_session(tape, sizes, TrainConfig::default());
    register_two_layer(&mut target, w_source, w_tied, u)?;

    let facts = Facts::new(0, 1, 2, 1, N)?;
    let header = Header::new(Vec::new(), 0)?;

    // 槽位数错位：累积表只有 2 槽，会话 3 槽。
    let fewer = GradAccum::new(vec![vec![1.0; N], vec![2.0; N]], 1);
    let snapshot = Snapshot::new_with_all(
        facts,
        header.clone(),
        Moments::empty(),
        fewer,
        Ties::empty(),
    );
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("累积槽位数不符");
    assert_eq!(error.code(), Code::Invalid);

    // 槽长度不符：第三槽梯度长 N − 1。
    let wrong_len = GradAccum::new(vec![vec![1.0; N], vec![2.0; N], vec![3.0; N - 1]], 1);
    let snapshot = Snapshot::new_with_all(
        facts,
        header.clone(),
        Moments::empty(),
        wrong_len,
        Ties::empty(),
    );
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("累积槽长度不符");
    assert_eq!(error.code(), Code::Invalid);

    // ties 编号越界：回填槽 5 超出 3 槽。
    let snapshot = Snapshot::new_with_all(
        facts,
        header.clone(),
        Moments::empty(),
        GradAccum::empty(),
        Ties::new(vec![(0, 5)]),
    );
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("ties 编号越界");
    assert_eq!(error.code(), Code::Invalid);

    // 矩表槽长度不符：第一槽矩向量长 N − 1，会话槽参数长 N。
    let moments = Moments::new(
        vec![
            SlotMoments::new(vec![0.0; N - 1], vec![0.0; N - 1])?,
            SlotMoments::new(vec![0.0; N], vec![0.0; N])?,
            SlotMoments::new(vec![0.0; N], vec![0.0; N])?,
        ],
        1,
    );
    let snapshot = Snapshot::new_with_all(
        facts,
        header.clone(),
        moments,
        GradAccum::empty(),
        Ties::empty(),
    );
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("矩槽长度不符");
    assert_eq!(error.code(), Code::Invalid);

    // 矩表槽位数错位：非空矩表只 1 槽，会话 3 槽。
    let moments = Moments::new(vec![SlotMoments::new(vec![0.0; N], vec![0.0; N])?], 1);
    let snapshot =
        Snapshot::new_with_all(facts, header, moments, GradAccum::empty(), Ties::empty());
    let error = target
        .restore_snapshot(&snapshot)
        .expect_err("矩槽位数不符");
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
    session.bufs().load(x, Data::F32(vec![0.0; N]))?;
    session.bufs().compute_step("add_f32", &[p, x], loss)?;
    session.plan(loss)?;

    let error = session.optimize().expect_err("梯度未装入");
    assert_eq!(error.code(), Code::Missing);
    assert_eq!(session.slots()[0].params(), &[1.0; N], "拒绝前不改写参数");
    Ok(())
}

/// 可选覆盖：缺项取默认值、给出的字段生效；非法值仍走 [`TrainConfig::new`] 校验。
#[test]
fn overrides_fall_back_to_default_without_duplicating_constants() {
    let base = TrainConfig::default();

    let all_default =
        TrainConfig::with_overrides(None, None, None, None, None).expect("缺省配置合法");
    assert_eq!(all_default, base, "全缺省应与默认配置逐字段一致");

    let overridden = TrainConfig::with_overrides(Some(2e-3), Some(2), Some(5), Some(0.5), Some(2))
        .expect("部分覆盖合法");
    assert_eq!(overridden.learning_rate, 2e-3);
    assert_eq!(overridden.warmup_steps, 2);
    assert_eq!(overridden.max_steps, 5);
    assert_eq!(overridden.clip_norm, Some(0.5));
    assert_eq!(overridden.accumulation, 2);

    let bad_steps = TrainConfig::with_overrides(None, None, Some(0), None, None);
    assert_eq!(bad_steps.expect_err("总步数不能为 0").code(), Code::Invalid);

    let bad_accum = TrainConfig::with_overrides(None, None, None, None, Some(0));
    assert_eq!(
        bad_accum.expect_err("累积窗口不能为 0").code(),
        Code::Invalid
    );
}

/// 检查点事实集：恢复所需的最小事实集与偏差拒绝。
mod checkpoint {
    use crate::session::checkpoint::*;

    use diagnostic::Code;
    use tensor::Shape;

    /// 造一个可恢复的最小检查点：`gemv_i4` 单节点图 + 2×2 批 + 2 副本拓扑，返回检查点、头部与拓扑。
    fn fixture() -> (Checkpoint, weights::Header, shard::Topology) {
        let mut tape = autograd::Tape::new();
        let x = tape.leaf();
        tape.record("gemv_i4", &[x]).expect("记录");
        let tokens = [1, 2, 3, 4];
        let batch = crate::Batch::new(&tokens, 2, 2).expect("批");
        let header = weights::Header::new(
            vec![weights::Entry::new("w", tensor::DType::F32, Shape::new([2])).at(64)],
            64,
        )
        .expect("头部");
        let topology = shard::Topology::new(1, 1, 2).expect("拓扑");
        let checkpoint = Checkpoint::new(7, &tape, &batch, 2, &header, topology).expect("检查点");
        (checkpoint, header, topology)
    }

    #[test]
    fn a_checkpoint_records_the_minimum_state_needed_to_resume() {
        let (checkpoint, _header, topology) = fixture();
        assert_eq!(checkpoint.step(), 7);
        assert_eq!(checkpoint.tape_nodes(), 1);
        assert_eq!(checkpoint.batch_rows(), 2);
        assert_eq!(checkpoint.batch_cols(), 2);
        assert_eq!(checkpoint.weight_entries(), 1);
        assert_eq!(checkpoint.world(), 2);
        resume(&checkpoint, &topology, 2, 2, 2).expect("同结构可恢复");
        let other = shard::Topology::new(1, 1, 4).expect("拓扑");
        assert_eq!(
            resume(&checkpoint, &other, 2, 2, 2).unwrap_err().code(),
            Code::Invalid
        );
        assert_eq!(
            resume(&checkpoint, &topology, 3, 2, 2).unwrap_err().code(),
            Code::Invalid,
            "层数不一致必须拒绝"
        );
        assert_eq!(
            resume(&checkpoint, &topology, 2, 4, 2).unwrap_err().code(),
            Code::Invalid,
            "批行数不一致必须拒绝"
        );
        assert_eq!(
            resume(&checkpoint, &topology, 2, 2, 4).unwrap_err().code(),
            Code::Invalid,
            "批列宽不一致必须拒绝"
        );
    }

    #[test]
    fn a_checkpoint_becomes_a_snapshot_that_round_trips() {
        let (checkpoint, header, _topology) = fixture();
        let snapshot = checkpoint.snapshot(&header).expect("快照");
        let bytes = snapshot.encode().expect("编码");
        let restored = snapshot::Snapshot::decode(&bytes).expect("解码");
        let facts = restored.facts();
        assert_eq!(facts.step(), checkpoint.step());
        assert_eq!(facts.world(), checkpoint.world());
        assert_eq!(facts.layers(), checkpoint.layers());
        assert_eq!(facts.batch_rows(), checkpoint.batch_rows());
        assert_eq!(facts.batch_cols(), checkpoint.batch_cols());
        assert_eq!(
            restored.weights().entries().len(),
            checkpoint.weight_entries()
        );
    }

    #[test]
    fn a_checkpoint_without_layers_is_rejected() {
        let tape = autograd::Tape::new();
        let tokens = [1];
        let batch = crate::Batch::new(&tokens, 1, 1).expect("批");
        let header = weights::Header::new(Vec::new(), 0).expect("空头部");
        let topology = shard::Topology::new(1, 1, 1).expect("拓扑");
        assert_eq!(
            Checkpoint::new(0, &tape, &batch, 0, &header, topology)
                .unwrap_err()
                .code(),
            Code::Invalid
        );
    }

    #[test]
    fn a_layer_exposes_the_operators_it_was_validated_against() {
        let layer = nn::Linear::new("blk.0.attn_q", 4, 4).expect("层");
        assert_eq!(operators(&layer), ["gemv_i4"]);
    }
}

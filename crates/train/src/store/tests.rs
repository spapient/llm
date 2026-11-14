//! `store` 的测试：装载、重放、峰值与池的复用，全部走 `Reference` 参考实现。

use super::*;
use autograd::{Budget, Footprint, Plan, Replay, Tape};

/// 每层都是 `y = gemv_i4(x, w, scales)`：`m = n = 8`，`scales` 只有一组（组大小 8）。
const WIDTH: usize = 8;
/// 三个常驻值的字节数：`x`（8×f32）、`w`（8×8 个 int4 = 32 B）、`scales`（8×f32）。
const RESIDENT: u64 = 96;

fn input() -> Data {
    Data::F32(
        (0..WIDTH)
            .map(|i| ((i * 7) % 13) as f32 * 0.25 - 1.0)
            .collect(),
    )
}

fn weights() -> Data {
    Data::U8(
        (0..WIDTH * (WIDTH / 2))
            .map(|i| ((i * 37) % 251) as u8)
            .collect(),
    )
}

fn scales() -> Data {
    Data::F32(
        (0..WIDTH)
            .map(|i| 0.031_25 * (1 + (i % 4)) as f32)
            .collect(),
    )
}

/// `var` 现在的 f32 值：没有值或不是浮点都是测试自己写错了。
fn f32s(store: &Buffers<Reference>, var: Var) -> Vec<f32> {
    let view = store.value(var).expect("有值");
    view.f32().expect("f32").to_vec()
}

/// 一条 `layers` 层的链：形状表、叶子与 loss 一起给出，测试不必各自重算。
struct Fixture {
    tape: Tape,
    sizes: Sizes,
    x: Var,
    w: Var,
    scales: Var,
    loss: Var,
}

impl Fixture {
    fn new(layers: usize) -> Self {
        let mut tape = Tape::new();
        let (x, w, scales) = (tape.parameter(), tape.leaf(), tape.leaf());
        let mut loss = x;
        for _ in 0..layers {
            loss = tape.record("gemv_i4", &[loss, w, scales]).expect("记录");
        }
        let mut sizes = Sizes::new(tape.vars());
        sizes.set(x, Shape::new([WIDTH]), DType::F32);
        sizes.set(w, Shape::new([WIDTH, WIDTH]), DType::U4);
        sizes.set(scales, Shape::new([WIDTH]), DType::F32);
        for node in tape.nodes() {
            sizes.set(node.output(), Shape::new([WIDTH]), DType::F32);
        }
        Self {
            tape,
            sizes,
            x,
            w,
            scales,
            loss,
        }
    }

    fn store(&self) -> Buffers<Reference> {
        let mut store = Buffers::new(self.sizes.clone(), Reference);
        store.load(self.x, input()).expect("输入");
        store.load(self.w, weights()).expect("权重");
        store.load(self.scales, scales()).expect("scale");
        store
    }

    /// 一步 `gemv_i4` 前向：`(x, w, scales) → output`。
    fn step(&self, store: &mut Buffers<Reference>, output: Var) -> Result<()> {
        store.compute("gemv_i4", &[self.x, self.w, self.scales], output)
    }

    /// 按记录顺序算一遍全图，返回每个变量当时的值（叶子也在内）。
    fn forward(&self, store: &mut Buffers<Reference>) -> Vec<Option<Data>> {
        for node in self.tape.nodes() {
            store
                .compute(node.op(), node.inputs(), node.output())
                .expect("前向可算");
        }
        (0..self.tape.vars())
            .map(|id| {
                let view = store.value(self.tape.var(id).expect("编号有效"));
                view.map(|view| match view {
                    View::F32(values) => Data::F32(values.to_vec()),
                    View::U8(values) => Data::U8(values.to_vec()),
                })
            })
            .collect()
    }

    /// 把前向留下的激活收敛到计划保留的那一份：检查点只能是前向没丢掉的那批。
    fn rewind(&self, store: &mut Buffers<Reference>, plan: &Plan) {
        for node in self.tape.nodes() {
            if !plan.keep(node.output()) {
                store.free(node.output());
            }
        }
        assert_eq!(
            store.live(),
            RESIDENT + plan.retained(),
            "重放开始前只剩常驻值与检查点"
        );
    }
}

#[test]
fn compute_matches_the_operator_reference_bit_for_bit() {
    let fixture = Fixture::new(1);
    let mut store = fixture.store();
    fixture.step(&mut store, fixture.loss).expect("可算");

    let mut expected = vec![0.0f32; WIDTH];
    let (x, w, s) = (input(), weights(), scales());
    let (xv, wv, sv) = (
        x.view().f32().expect("f32"),
        w.view().u8().expect("u8"),
        s.view().f32().expect("f32"),
    );
    op::gemv_i4::forward(xv, wv, sv, &mut expected).expect("参考可算");
    assert_eq!(
        f32s(&store, fixture.loss),
        expected,
        "算术必须与 np 的参考同一份"
    );
}

#[test]
fn freeing_returns_the_buffer_and_lowers_live_immediately() {
    let fixture = Fixture::new(1);
    let mut store = fixture.store();
    let activation = (WIDTH * 4) as u64;

    assert_eq!(store.live(), RESIDENT, "只有常驻值");
    fixture.step(&mut store, fixture.loss).expect("可算");
    assert_eq!(store.live(), RESIDENT + activation);
    assert_eq!(store.peak(), RESIDENT + activation);

    store.free(fixture.loss);
    assert_eq!(store.live(), RESIDENT, "交还当场生效，不等作用域结束");
    assert_eq!(store.pooled(), 1, "块进了空闲表");

    fixture.step(&mut store, fixture.loss).expect("可算");
    assert_eq!(store.pooled(), 0, "第二次算复用同一块");
    assert_eq!(store.live(), RESIDENT + activation);
}

#[test]
fn replaying_into_a_reused_buffer_reproduces_the_same_values() {
    let fixture = Fixture::new(1);
    let mut store = fixture.store();
    let run = |store: &mut Buffers<Reference>| {
        fixture.step(store, fixture.loss).expect("可算");
        f32s(store, fixture.loss)
    };
    let first = run(&mut store);
    store.free(fixture.loss);
    assert_eq!(first, run(&mut store), "复用块不得留下上一轮的任何一项");
}

#[test]
fn a_released_input_is_never_read_as_a_value() {
    let fixture = Fixture::new(2);
    let mut store = fixture.store();
    let first = fixture.tape.nodes()[0].output();
    fixture.step(&mut store, first).expect("可算");
    store.free(first);

    let error = store
        .compute("gemv_i4", &[first, fixture.w, fixture.scales], fixture.loss)
        .expect_err("输入已释放");
    assert_eq!(error.code(), Code::Missing);
    assert_eq!(store.live(), RESIDENT, "失败的这一次没有留下任何占用");
}

#[test]
fn a_variable_without_a_registered_shape_is_rejected() {
    let fixture = Fixture::new(1);
    let mut store = Buffers::new(Sizes::new(fixture.tape.vars()), Reference);
    let error = fixture
        .step(&mut store, fixture.loss)
        .expect_err("没有形状");
    assert_eq!(error.code(), Code::Internal);
}

#[test]
fn an_element_type_without_scalar_storage_is_unsupported() {
    let mut pool = Pool::default();
    let error = pool
        .block(DType::F16, &Shape::new([4]))
        .expect_err("没有 f16 存储");
    assert_eq!(error.code(), Code::Unsupported);
}

#[test]
fn a_loaded_block_must_match_what_the_shape_table_says() {
    let fixture = Fixture::new(1);
    let mut store = fixture.store();
    let too_short = Data::U8(vec![0; WIDTH]); // 登记的是 8×8 个 int4 = 32 B
    let error = store.load(fixture.w, too_short).expect_err("长度不符");
    assert_eq!(error.code(), Code::Invalid);
    let error = store
        .load(fixture.w, Data::F32(vec![0.0; 64]))
        .expect_err("形态不符");
    assert_eq!(error.code(), Code::Invalid, "浮点缓冲不能装进整型槽位");
    assert_eq!(store.live(), RESIDENT, "失败没有破坏原来的常驻值");
}

/// 没登记形状的变量不能绕过校验：否则任意长度、任意类型的数据都能占住一个槽位。
#[test]
fn a_variable_without_registered_metadata_cannot_be_loaded() {
    let mut tape = Tape::new();
    let (first, gap) = (tape.parameter(), tape.parameter());
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(first, Shape::new([1]), DType::F32);
    let mut store = Buffers::new(sizes, Reference);

    let error = store.load(gap, Data::F32(vec![0.0])).expect_err("没登记");
    assert_eq!(error.code(), Code::Internal, "这是调用方漏了 Sizes::set");
    assert!(error.to_string().contains("没有登记"), "{error}");
    assert_eq!(store.live(), 0, "被拒绝的值不能占住槽位");
    assert!(store.value(gap).is_none());
}

/// 大到分配不出来的形状必须是错误，而不是把宿主进程 abort 掉：形状可能就是从文件头读来的。
#[test]
fn an_impossibly_large_output_shape_is_an_error_rather_than_an_abort() {
    let mut tape = Tape::new();
    let (x, y) = (tape.parameter(), tape.parameter());
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([1]), DType::F32);
    sizes.set(y, Shape::new([usize::MAX]), DType::F32);
    let mut store = Buffers::new(sizes, Reference);
    store.load(x, Data::F32(vec![1.0])).expect("输入");

    let error = store
        .compute("gemv_i4", &[x], y)
        .expect_err("分配不出来就必须报错");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("个元素"), "{error}");
    assert_eq!(store.live(), 4, "失败的那次不留下半块缓冲区");
    assert!(store.value(y).is_none());
}

#[test]
fn an_unknown_operator_is_unsupported_rather_than_silently_zero() {
    let fixture = Fixture::new(1);
    let mut store = fixture.store();
    let error = store
        .compute("relu", &[fixture.x], fixture.loss)
        .expect_err("没有 relu 的内核");
    assert_eq!(error.code(), Code::Unsupported);
    assert!(
        error.to_string().contains("gemv_f32") && error.to_string().contains("gemv_i4"),
        "`unsupported` 要说出表里覆盖了什么：{error}"
    );
    assert_eq!(store.live(), RESIDENT);
}

#[test]
fn a_plan_drives_the_store_and_every_replay_matches_the_forward() {
    let fixture = Fixture::new(12);
    let mut store = fixture.store();
    let truth = fixture.forward(&mut store);
    let total: u64 = fixture
        .tape
        .nodes()
        .iter()
        .map(|n| fixture.sizes.bytes(n.output()))
        .sum();
    let schedule = fixture.tape.backward(fixture.loss).expect("在图上");
    let budget = Budget::new(total * 2 / 3);
    let plan = Plan::of(&fixture.tape, &schedule, &fixture.sizes, budget).expect("预算够");
    fixture.rewind(&mut store, &plan);
    // 前向自己也占过内存；峰值要按"重放这一段"量，才对得上计划承诺的数。
    store.reset_peak();
    let replay = Replay::of(&fixture.tape, &schedule, &plan).expect("可执行");

    let mut checked = 0usize;
    let computed = replay
        .run(&fixture.tape, &schedule, &mut store, |store, step| {
            for input in step.inputs() {
                let actual = store.value(*input).expect("重放后输入必须在场");
                let expected = truth[input.id() as usize].clone().expect("前向算过");
                let same = match (actual, expected) {
                    (View::F32(actual), Data::F32(expected)) => actual == expected.as_slice(),
                    (View::U8(actual), Data::U8(expected)) => actual == expected.as_slice(),
                    _ => false,
                };
                assert!(
                    same,
                    "第 {} 步的输入 {} 与前向不一致",
                    step.output().id(),
                    input.id()
                );
                checked += 1;
            }
            Ok(())
        })
        .expect("可重放");

    assert!(computed > 0, "紧预算必须真的重算");
    assert_eq!(computed, plan.replayed(), "实测与计划承诺必须一致");
    assert!(checked > 0);
    assert_eq!(
        store.live(),
        RESIDENT,
        "执行完只该剩常驻值：检查点与临时值都到了释放点"
    );
    let peak = store.peak() - RESIDENT;
    assert!(
        peak <= plan.peak(),
        "实测峰值 {peak} 超过计划 {}",
        plan.peak()
    );
}

#[test]
fn many_rounds_do_not_grow_the_pool() {
    let fixture = Fixture::new(1);
    let mut store = fixture.store();
    for round in 0..50 {
        fixture
            .step(&mut store, fixture.loss)
            .unwrap_or_else(|e| panic!("第 {round} 轮：{e}"));
        store.free(fixture.loss);
        assert_eq!(store.live(), RESIDENT, "第 {round} 轮");
        assert_eq!(store.pooled(), 1, "第 {round} 轮：空闲表不增长");
    }
}

#[test]
fn clear_returns_everything_and_resets_the_peak() {
    let fixture = Fixture::new(1);
    let mut store = fixture.store();
    fixture.step(&mut store, fixture.loss).expect("可算");
    assert!(store.peak() > RESIDENT);

    store.clear();
    assert_eq!((store.live(), store.peak(), store.pooled()), (0, 0, 0));
    assert!(store.value(fixture.x).is_none(), "常驻值也要还回去");

    // 归零之后同一个实例可以接着用：形状表还在，值要重新装。
    store.load(fixture.x, input()).expect("重新装入");
    assert_eq!(store.live(), WIDTH as u64 * 4, "只算刚装进来的那一块");
}

/// 注册表里有 `gemv_f32`，参考内核就必须有它——两张表由编译期断言锁在一起。
#[test]
fn the_second_registered_operator_replays_like_the_first() {
    const N: usize = WIDTH;
    let mut tape = Tape::new();
    let (x, w) = (tape.parameter(), tape.leaf());
    let y = tape.record("gemv_f32", &[x, w]).expect("记录");
    let mut sizes = Sizes::new(tape.vars());
    sizes.set(x, Shape::new([N]), DType::F32);
    sizes.set(w, Shape::new([N, N]), DType::F32);
    sizes.set(y, Shape::new([N]), DType::F32);
    let mut store = Buffers::new(sizes, Reference);

    let input: Vec<f32> = (0..N).map(|i| i as f32 * 0.25 - 1.0).collect();
    let weights: Vec<f32> = (0..N * N)
        .map(|i| ((i * 37) % 251) as f32 * 0.125 - 8.0)
        .collect();
    store.load(x, Data::F32(input.clone())).expect("输入");
    store.load(w, Data::F32(weights.clone())).expect("权重");
    store.compute("gemv_f32", &[x, w], y).expect("可算");

    let mut expected = vec![0.0f32; N];
    op::gemv_f32::forward(&input, &weights, &mut expected).expect("参考可算");
    assert_eq!(
        f32s(&store, y),
        expected,
        "第二条算子的重放必须与它自己的参考逐位相同"
    );
}

/// 元数与 lane 的校验只写一份（`Inputs` 的单态化拆包），且失败的那次不留下占用。
#[test]
fn a_kernel_signature_mismatch_names_the_slot() {
    let fixture = Fixture::new(1);
    let mut store = fixture.store();

    let error = store.compute("gemv_i4", &[fixture.x, fixture.w], fixture.loss);
    let error = error.expect_err("少一个输入");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("必须是"), "{error}");

    let arity = &[fixture.x, fixture.x, fixture.scales];
    let error = store
        .compute("gemv_i4", arity, fixture.loss)
        .expect_err("权重是 f32");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("第 1 个输入"), "{error}");
    assert_eq!(store.live(), RESIDENT, "失败的这一次没有留下占用");
}

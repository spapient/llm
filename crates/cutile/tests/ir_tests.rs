use super::pass::{nodes, normalize};
use super::*;

#[crate::dsl]
#[allow(dead_code)]
fn affine(input: &[f32], output: &mut [f32], scale: f32, extent: usize) {
    for position in 0..extent {
        output[position] = input[position] * scale + (2.0 + 1.0);
    }
}

/// 归约形状由同一宏前端翻译：`out[0] += a[k] * b[k]` 应落成 `Reduce` 而不是 `Kernel`。
#[crate::dsl]
#[allow(dead_code)]
fn dot_reduce(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    for k in 0..len {
        out[0] += a[k] * b[k];
    }
}

fn buf(name: &str, access: Access) -> Parameter {
    Parameter {
        name: name.to_string(),
        ty: ValueType::Buf {
            element: ScalarType::F32,
            access,
        },
    }
}

fn scalar(name: &str, ty: ScalarType) -> Parameter {
    Parameter {
        name: name.to_string(),
        ty: ValueType::Scalar(ty),
    }
}

/// 只有 `output` + `extent` 的最小合法内核，别的判据都从它派生出来。
fn minimal() -> Kernel {
    Kernel::map(
        "copy",
        vec![
            Parameter {
                name: "output".to_string(),
                ty: ValueType::Buf {
                    element: ScalarType::F32,
                    access: Access::ReadWrite,
                },
            },
            Parameter {
                name: "extent".to_string(),
                ty: ValueType::Scalar(ScalarType::Usize),
            },
        ],
        "i",
        "extent",
        "output",
        Expr::F32(0.0f32.to_bits()),
    )
}

/// 最小点积归约：`out[0] += a[k] * b[k]`，两个只读缓冲 + 一个输出槽 + 一个长度。
fn dot() -> Kernel {
    Kernel::reduce(
        "dot",
        vec![
            Parameter {
                name: "a".to_string(),
                ty: ValueType::Buf {
                    element: ScalarType::F32,
                    access: Access::Read,
                },
            },
            Parameter {
                name: "b".to_string(),
                ty: ValueType::Buf {
                    element: ScalarType::F32,
                    access: Access::Read,
                },
            },
            Parameter {
                name: "out".to_string(),
                ty: ValueType::Buf {
                    element: ScalarType::F32,
                    access: Access::ReadWrite,
                },
            },
            Parameter {
                name: "len".to_string(),
                ty: ValueType::Scalar(ScalarType::Usize),
            },
        ],
        "k",
        "len",
        "out",
        0,
        Expr::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expr::Load("a".to_string())),
            right: Box::new(Expr::Load("b".to_string())),
        },
    )
}

#[test]
fn dsl_builds_verified_optimized_ir_and_portable_source() {
    let kernel = __dsl_ir_affine();
    kernel.verify().expect("macro output has valid types");
    let source = kernel.emit_c().expect("verified IR emits C");
    assert!(source.contains("int affine(const float *input, size_t cutile_len_0, float *output, size_t cutile_len_1, float scale, size_t extent)"), "{source}");
    assert!(
        source.contains("if (extent > cutile_len_0) return 1;"),
        "{source}"
    );
    assert!(
        source.contains("output[position] = ((input[position] * scale) + 3.0f);"),
        "{source}"
    );
}

#[test]
fn dsl_builds_a_reduce_ir_from_compound_assignment() {
    let reduce = __dsl_ir_dot_reduce();
    reduce
        .verify()
        .expect("macro reduction output has valid types");
    let source = reduce.emit_c().expect("verified reduction emits C");
    assert!(
        source.contains(
            "int dot_reduce(const float *a, size_t cutile_len_0, const float *b, size_t cutile_len_1, \
             float *out, size_t cutile_len_2, size_t len)"
        ),
        "{source}"
    );
    assert!(
        source.contains("if (0 >= cutile_len_2) return 1;"),
        "{source}"
    );
    assert!(source.contains("out[0] += (a[k] * b[k]);"), "{source}");
}

#[test]
fn verifier_rejects_non_mutable_outputs_and_invalid_extents() {
    let mut kernel = minimal();
    kernel.params[0].ty = ValueType::Buf {
        element: ScalarType::F32,
        access: Access::Read,
    };
    let error = kernel
        .verify()
        .expect_err("wrong output access is rejected");
    assert!(error.to_string().contains("mutable f32 buffer"));

    kernel.params[0].ty = ValueType::Buf {
        element: ScalarType::F32,
        access: Access::ReadWrite,
    };
    kernel.verify().expect("valid signature passes");
    kernel.params[1].ty = ValueType::Scalar(ScalarType::F32);
    let error = kernel.verify().expect_err("non-usize extent is rejected");
    assert!(error.to_string().contains("loop extent"));
}

/// 名字要原样写进 C 片段，所以 C 关键字与非法形状都在发射前被打回。
#[test]
fn c_keywords_and_odd_names_are_rejected() {
    let mut kernel = minimal();
    for name in ["float", "1out", "out-put", ""] {
        kernel.name = name.to_string();
        let error = kernel.verify().expect_err("non-portable name is rejected");
        assert!(
            error.to_string().contains("portable C identifier"),
            "{name}"
        );
    }
}

#[test]
fn generated_length_names_cannot_collide_with_parameters() {
    let mut kernel = minimal();
    kernel.params.insert(
        0,
        Parameter {
            name: "cutile_len_0".to_string(),
            ty: ValueType::Buf {
                element: ScalarType::F32,
                access: Access::Read,
            },
        },
    );
    let error = kernel
        .verify()
        .expect_err("clashing generated name is rejected");
    assert!(error.to_string().contains("generated buffer length"));
}

/// `usize` 缓冲本身合法（还能当形参），但它装不下 `f32`，必须在校验期就被挡下。
#[test]
fn non_f32_bufs_are_rejected() {
    let mut kernel = minimal();
    kernel.params[0].ty = ValueType::Buf {
        element: ScalarType::Usize,
        access: Access::ReadWrite,
    };
    let error = kernel.verify().expect_err("usize buffer is rejected");
    assert!(error.to_string().contains("must contain f32 elements"));
}

/// 折叠只吃掉能算出有限值的常量：`1.0 / 0.0` 会变成 `inf`，所以原样留下。
#[test]
fn optimizer_keeps_operations_that_leave_the_finite_range() {
    let mut kernel = minimal();
    kernel.write = Write::Map(Expr::Binary {
        op: BinaryOp::Div,
        left: Box::new(Expr::F32(1.0f32.to_bits())),
        right: Box::new(Expr::F32(0.0f32.to_bits())),
    });
    let source = kernel.emit_c().expect("non-finite folding still emits C");
    assert!(source.contains("(1.0f / 0.0f)"), "{source}");

    kernel.write = Write::Map(Expr::Neg(Box::new(Expr::F32(2.0f32.to_bits()))));
    let source = kernel.emit_c().expect("finite negation folds");
    assert!(source.contains("= -2.0f;"), "{source}");
}

#[test]
fn loads_must_name_a_buf_and_scalars_are_typed() {
    let mut kernel = minimal();
    kernel.params.push(Parameter {
        name: "scale".to_string(),
        ty: ValueType::Scalar(ScalarType::F32),
    });
    kernel.write = Write::Map(Expr::Load("scale".to_string()));
    let error = kernel.verify().expect_err("scalar cannot be loaded");
    assert!(error.to_string().contains("cannot be loaded as a buffer"));

    kernel.write = Write::Map(Expr::Load("missing".to_string()));
    let error = kernel.verify().expect_err("unknown buffer is rejected");
    assert!(error.to_string().contains("unknown buffer"));

    kernel.write = Write::Map(Expr::Scalar("missing".to_string()));
    let error = kernel.verify().expect_err("unknown scalar is rejected");
    assert!(error.to_string().contains("unknown scalar parameter"));
}

/// 归约发射：输入缓冲用归约长度做上界，输出缓冲只看固定槽位是否越界。
#[test]
fn reduce_emits_an_accumulating_loop_into_a_fixed_slot() {
    let source = dot().emit_c().expect("reduction IR emits C");
    assert!(
        source.contains(
            "int dot(const float *a, size_t cutile_len_0, const float *b, size_t cutile_len_1, \
             float *out, size_t cutile_len_2, size_t len)"
        ),
        "{source}"
    );
    assert!(
        source.contains("if (len > cutile_len_0) return 1;"),
        "{source}"
    );
    assert!(
        source.contains("if (len > cutile_len_1) return 1;"),
        "{source}"
    );
    assert!(
        source.contains("if (0 >= cutile_len_2) return 1;"),
        "{source}"
    );
    assert!(source.contains("out[0] += (a[k] * b[k]);"), "{source}");
}

/// 归约与逐点共用同一套校验：输出仍是可变 f32 缓冲，值仍是 f32 表达式。
#[test]
fn reduce_rejects_non_mutable_outputs() {
    let mut reduce = dot();
    reduce.output = "missing".to_string();
    let error = reduce.verify().expect_err("unknown output is rejected");
    assert!(error.to_string().contains("mutable f32 buffer"));

    let mut reduce = dot();
    reduce.write = Write::Reduce {
        at: 0,
        value: Expr::Scalar("len".to_string()),
    };
    let error = reduce.verify().expect_err("usize scalar is not f32");
    assert!(error.to_string().contains("must have type f32"));
}

/// 优化通道删掉死形参：发射的 C 签名里没有它，行为不变。
#[test]
fn optimize_drops_dead_params_before_emission() {
    let mut kernel = minimal();
    kernel.params.push(Parameter {
        name: "dead".to_string(),
        ty: ValueType::Scalar(ScalarType::F32),
    });
    let optimized = kernel.optimize().expect("optimize succeeds");
    assert!(
        optimized
            .params
            .iter()
            .all(|parameter| parameter.name != "dead"),
        "{optimized:?}"
    );
    let source = kernel
        .emit_c()
        .expect("dead param is dropped, C still emits");
    assert!(!source.contains("dead"), "{source}");
}

/// 一个幂等的改名 pass：只认 `from`，改到 `to`，之后再跑什么都不变。
#[derive(Clone, Copy)]
struct Rename {
    name: &'static str,
    from: &'static str,
    to: &'static str,
}

impl Pass for Rename {
    fn name(&self) -> &'static str {
        self.name
    }

    fn run(&self, mut kernel: Kernel) -> Result<Kernel, VerificationError> {
        if kernel.name == self.from {
            kernel.name = self.to.to_string();
        }
        Ok(kernel)
    }
}

/// 把 `-(0 - y)` 降成 `-(-y)`：一条「暴露双重取负」的 pass，自己严格收缩。
struct LowerSubtraction;

impl Pass for LowerSubtraction {
    fn name(&self) -> &'static str {
        "lower-subtraction"
    }

    fn run(&self, mut kernel: Kernel) -> Result<Kernel, VerificationError> {
        let Write::Map(Expr::Neg(inner)) = &kernel.write else {
            return Ok(kernel);
        };
        let Expr::Binary {
            op: BinaryOp::Sub,
            left,
            right,
        } = inner.as_ref()
        else {
            return Ok(kernel);
        };
        let Expr::F32(bits) = left.as_ref() else {
            return Ok(kernel);
        };
        if *bits != 0.0f32.to_bits() {
            return Ok(kernel);
        }
        let value = Expr::Neg(Box::new(right.as_ref().clone()));
        kernel.write = Write::Map(Expr::Neg(Box::new(value)));
        Ok(kernel)
    }
}

/// 撤掉 `-(-y)`。
struct CancelDoubleNegation;

impl Pass for CancelDoubleNegation {
    fn name(&self) -> &'static str {
        "cancel-double-negation"
    }

    fn run(&self, mut kernel: Kernel) -> Result<Kernel, VerificationError> {
        let Write::Map(Expr::Neg(inner)) = &kernel.write else {
            return Ok(kernel);
        };
        let Expr::Neg(value) = inner.as_ref() else {
            return Ok(kernel);
        };
        kernel.write = Write::Map(value.as_ref().clone());
        Ok(kernel)
    }
}

/// `Pass` 可组合。寄存器要求的是「结果稳定」，不是「调用方排好依赖」：互相交换的 pass 无论怎么
/// 注册都会收敛到同一个范式（上面反序那条已证），所以顺序只在两个 pass **真正冲突**时才决定结果。
/// 这里让两条改名对 `middle` 各持一词，正序得 `left`，反序得 `right`。
#[test]
fn pass_manager_applies_custom_passes_in_order() {
    let towards_left = Rename {
        name: "middle-to-left",
        from: "middle",
        to: "left",
    };
    let towards_right = Rename {
        name: "middle-to-right",
        from: "middle",
        to: "right",
    };
    let mut kernel = minimal();
    kernel.name = "middle".to_string();
    let forward = PassManager::new()
        .push(towards_left)
        .push(towards_right)
        .run(kernel.clone())
        .expect("custom passes run");
    assert_eq!(forward.name, "left");
    let backward = PassManager::new()
        .push(towards_right)
        .push(towards_left)
        .run(kernel)
        .expect("custom passes run");
    assert_eq!(backward.name, "right");
}

/// 管理器不要求调用方排好依赖：`CancelDoubleNegation` 注册在 `LowerSubtraction` **之前**，而它要
/// 消掉的 `-(-x)` 只有后者才造得出来，所以这条恒等只能靠下一轮扫描完成。两轮各自严格收缩，
/// 因此驱动一定会重扫——这正是「一轮算完」测不出来的性质。
#[test]
fn a_pass_sees_work_created_by_a_later_pass_on_the_next_sweep() {
    let kernel = Kernel::map(
        "peel",
        vec![
            buf("out", Access::ReadWrite),
            scalar("extent", ScalarType::Usize),
        ],
        "i",
        "extent",
        "out",
        Expr::Neg(Box::new(Expr::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expr::F32(0.0f32.to_bits())),
            right: Box::new(Expr::F32(2.0f32.to_bits())),
        })),
    );
    let peeled = PassManager::new()
        .push(CancelDoubleNegation)
        .push(LowerSubtraction)
        .run(kernel)
        .expect("two sweeps reach the normal form");
    assert_eq!(peeled.write, Write::Map(Expr::F32(2.0f32.to_bits())));
}

/// 只加不减的 pass 不会被悄悄截断：上限（初始 `cost`）耗尽后 `run` 报错，调用方拿到的是
/// 「没收敛」，而不是一段半成品内核。
#[test]
fn a_pipeline_that_never_settles_is_an_error() {
    struct Grow;
    impl Pass for Grow {
        fn name(&self) -> &'static str {
            "grow"
        }

        fn run(&self, mut kernel: Kernel) -> Result<Kernel, VerificationError> {
            let Write::Map(value) = &kernel.write else {
                return Ok(kernel);
            };
            let wrapped = Expr::Neg(Box::new(value.clone()));
            kernel.write = Write::Map(wrapped);
            Ok(kernel)
        }
    }
    let error = PassManager::new()
        .push(Grow)
        .run(minimal())
        .expect_err("a growing pass never settles");
    assert!(error.to_string().contains("did not settle"), "{error}");
    // 诊断要点名是哪条流水线在震荡，否则调用方只能猜。
    assert!(error.to_string().contains("[grow]"), "{error}");
}

/// 坏内核由 pass 产出时，报错点名那条 pass：入站校验能过，调用方就不会误以为是自己写错了，
/// 而且发布构建里也不会让它溜到发射器。
#[test]
fn an_invalid_kernel_from_a_pass_names_the_pass() {
    struct Broken;
    impl Pass for Broken {
        fn name(&self) -> &'static str {
            "broken"
        }

        fn run(&self, mut kernel: Kernel) -> Result<Kernel, VerificationError> {
            kernel.output = "missing".to_string();
            Ok(kernel)
        }
    }
    let error = PassManager::new()
        .push(Broken)
        .run(minimal())
        .expect_err("a pass that breaks the signature must be reported");
    let message = error.to_string();
    assert!(message.contains("`broken`"), "{message}");
    assert!(message.contains("invalid kernel"), "{message}");
}

/// 化简一路走到发射器：`input * (0.5 + 0.5)` 在 C 里只剩 `input`。
#[test]
fn the_fixpoint_reaches_the_emitted_c() {
    let kernel = Kernel::map(
        "half",
        vec![
            buf("out", Access::ReadWrite),
            scalar("extent", ScalarType::Usize),
            buf("input", Access::Read),
        ],
        "i",
        "extent",
        "out",
        Expr::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expr::Load("input".to_string())),
            right: Box::new(Expr::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expr::F32(0.5f32.to_bits())),
                right: Box::new(Expr::F32(0.5f32.to_bits())),
            }),
        },
    );
    let source = kernel.emit_c().expect("emits");
    assert!(source.contains("= input[i];"), "{source}");
    assert!(!source.contains("0.5"), "{source}");
}

// 定律的差分执行证据：这里只执行、只比较结果位；结构判据在 `ir::pass` 的单测里。
fn literal(value: f32) -> Expr {
    Expr::F32(value.to_bits())
}

fn load(name: &str) -> Expr {
    Expr::Load(name.to_string())
}

fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// 定律的**语义**判据：逐位相同（`NaN` 之间不比较 payload——核里不产生 sNaN，payload 本就不
/// 属于契约）。两个 `NaN` 视为一致。
fn same_result(left: f32, right: f32) -> bool {
    left.to_bits() == right.to_bits() || (left.is_nan() && right.is_nan())
}

/// 解释执行：定律的判据是「化简前后在同一组输入上逐位同值」，所以必须真的算一遍。
fn evaluate(expression: &Expr, inputs: [f32; 2]) -> f32 {
    let slot = |name: &str| match name {
        "y" => inputs[1],
        _ => inputs[0],
    };
    match expression {
        Expr::F32(bits) => f32::from_bits(*bits),
        Expr::Load(name) | Expr::Scalar(name) => slot(name),
        Expr::Neg(value) => -evaluate(value, inputs),
        Expr::Binary { op, left, right } => {
            op.apply(evaluate(left, inputs), evaluate(right, inputs))
        }
    }
}

/// 复用同一组特殊值：`±0` 能证伪 `x + 0.0`，`inf` / `NaN` 能证伪「看起来显然」的折叠。
const INPUTS: [[f32; 2]; 6] = [
    [0.0, -0.0],
    [-0.0, 0.0],
    [1.0, -1.0],
    [f32::from_bits(1), 3.5],
    [f32::INFINITY, f32::NEG_INFINITY],
    [f32::NAN, 0.5],
];

/// 确定性伪随机（xorshift），不引依赖：同一种子每次跑出同一棵表达式树。
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn pick<T: Copy>(&mut self, options: &[T]) -> T {
        options[(self.next() % options.len() as u64) as usize]
    }
}

/// 从字面量池里取一个常量：含 `±0` 与 `1.0`，好让恒等定律真的被触发。
fn any_literal(rng: &mut Rng) -> Expr {
    let pool = [0.0f32, -0.0, 1.0, 0.5, 2.0, -1.0, -2.0];
    literal(rng.pick(&pool))
}

/// 随机表达式，深度受限；叶子是字面量或两个缓冲读，保证至少有一种输入依赖。
fn any_expr(rng: &mut Rng, depth: u32) -> Expr {
    if depth == 0 || rng.next().is_multiple_of(3) {
        let leaf = rng.next() % 4;
        return match leaf {
            0 => any_literal(rng),
            1 => load("x"),
            2 => load("y"),
            _ => Expr::Scalar("s".to_string()),
        };
    }
    if rng.next().is_multiple_of(5) {
        return Expr::Neg(Box::new(any_expr(rng, depth - 1)));
    }
    let op = rng.pick(&[BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div]);
    binary(op, any_expr(rng, depth - 1), any_expr(rng, depth - 1))
}

/// 代表操作数：两个缓冲读、一个标量，加一批常量（含 `±0`、`1.0` 这些恒等律的判据），再加两种
/// 取负——定律的形状几乎都落在这里。
fn operands() -> Vec<Expr> {
    let mut operands = vec![load("x"), load("y"), Expr::Scalar("s".to_string())];
    for value in [0.0f32, -0.0, 0.5, 1.0, 2.0, -1.0, 0.75] {
        operands.push(literal(value));
    }
    operands.push(Expr::Neg(Box::new(load("x"))));
    operands.push(Expr::Neg(Box::new(literal(0.0))));
    operands
}

/// 待判定的形状：算子 × 操作数对**全组合**，每个节点再叠一层和两层取负。
///
/// 这一层不是抽样，是枚举：`x * 1.0`、`x / 1.0`、`x - 0.0`、`-(-x)` 的每个字面量组合都必然被
/// 跑到，所以「某条律在某个常量上不成立」不可能靠运气漏掉。
fn shapes() -> Vec<Expr> {
    let operands = operands();
    let mut shapes = Vec::new();
    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div] {
        for left in &operands {
            shapes.push(Expr::Neg(Box::new(left.clone())));
            shapes.push(Expr::Neg(Box::new(Expr::Neg(Box::new(left.clone())))));
            for right in &operands {
                let node = binary(op, left.clone(), right.clone());
                shapes.push(Expr::Neg(Box::new(node.clone())));
                shapes.push(node);
            }
        }
    }
    shapes
}

/// 一条形状的裁决：不长大、已是范式、每组输入逐位同值。
fn assert_sound(original: &Expr, simplified: &Expr) {
    assert!(
        nodes(simplified) <= nodes(original),
        "simplification grew {original:?}"
    );
    assert_eq!(
        normalize(simplified.clone()),
        simplified.clone(),
        "not a normal form: {original:?}"
    );
    for inputs in INPUTS {
        let before = evaluate(original, inputs);
        let after = evaluate(simplified, inputs);
        assert!(
            same_result(before, after),
            "{original:?} -> {simplified:?} changed {inputs:?}: {before} vs {after}"
        );
    }
}

/// 定律的差分判据：化简前后在每组输入（含 `±0`、次正规、`inf`、`NaN`）上逐位同值。
///
/// 先枚举每条律的形状（见 `shapes`），再用随机树补交互广度。把任何一条「逐位不成立」的律塞
/// 进表里——比如 `x + 0.0 → x`（`-0.0 + 0.0 = +0.0`）——枚举这一层立刻红，这比在注释里断言
/// 「成立」强得多。
#[test]
fn every_law_preserves_every_bit_it_touches() {
    for original in shapes() {
        assert_sound(&original, &normalize(original.clone()));
    }
    let mut rng = Rng(0x5eed_1234_9abc_def0);
    for _ in 0..300 {
        let original = any_expr(&mut rng, 4);
        assert_sound(&original, &normalize(original.clone()));
    }
}

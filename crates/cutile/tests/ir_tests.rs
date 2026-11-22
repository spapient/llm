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

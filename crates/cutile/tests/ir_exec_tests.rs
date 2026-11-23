//! 编译并执行 `emit_c` 的输出：与 Rust 参考实现逐位对拍。
//!
//! 字符串断言只能证明发射器写了哪些字；这里把源码交给 `cc` 编译、运行，再按 IEEE-754
//! 位模式比较，覆盖 ±0、次正规数、无穷、NaN、溢出与长度守卫。归约还额外锁住累加顺序。

use crate::kernel::{Access, Arith, Expr, Kernel, Parameter, ScalarType, ValueType};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

/// 每次对拍的固定长度：短到能在测试里逐位看完，长到能覆盖尾部与随机段。
const LEN: usize = 37;

fn buf(name: &str, access: Access) -> Parameter {
    Parameter {
        name: name.to_string(),
        ty: ValueType::Buf {
            element: ScalarType::F32,
            access,
        },
    }
}

fn length(name: &str) -> Parameter {
    Parameter {
        name: name.to_string(),
        ty: ValueType::Scalar(ScalarType::Usize),
    }
}

fn scalar(name: &str) -> Parameter {
    Parameter {
        name: name.to_string(),
        ty: ValueType::Scalar(ScalarType::F32),
    }
}

fn binary(op: Arith, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn load(name: &str) -> Expr {
    Expr::Load(name.to_string())
}

/// `output[i] = input[i] * scale + 3.0`，三种运算齐上：乘、加、标量参数。
fn affine() -> Kernel {
    Kernel::map(
        "affine",
        vec![
            buf("input", Access::Read),
            buf("output", Access::ReadWrite),
            scalar("scale"),
            length("extent"),
        ],
        "i",
        "extent",
        "output",
        binary(
            Arith::Add,
            binary(Arith::Mul, load("input"), Expr::Scalar("scale".to_string())),
            Expr::F32(3.0f32.to_bits()),
        ),
    )
}

/// `output[i] = -(input[i] / 3.0)`：负号与除法单独成核，避免只测一条表达式路径。
fn scaled_negation() -> Kernel {
    Kernel::map(
        "scaled_negation",
        vec![
            buf("input", Access::Read),
            buf("output", Access::ReadWrite),
            length("extent"),
        ],
        "i",
        "extent",
        "output",
        Expr::Neg(Box::new(binary(
            Arith::Div,
            load("input"),
            Expr::F32(3.0f32.to_bits()),
        ))),
    )
}

/// `out[i] = max(min(abs(a[i]), 2.0), -1.0)`：绝对值 + 二元 min/max 一条链，覆盖新组合子的发射对拍。
fn clipped() -> Kernel {
    Kernel::map(
        "clipped",
        vec![
            buf("a", Access::Read),
            buf("output", Access::ReadWrite),
            length("extent"),
        ],
        "i",
        "extent",
        "output",
        Expr::Max {
            left: Box::new(Expr::Min {
                left: Box::new(Expr::Abs(Box::new(load("a")))),
                right: Box::new(Expr::F32(2.0f32.to_bits())),
            }),
            right: Box::new(Expr::F32((-1.0f32).to_bits())),
        },
    )
}

/// `out[0] += a[k] * b[k]`：归约族的写回形状。
fn dot() -> Kernel {
    Kernel::reduce(
        "dot",
        vec![
            buf("a", Access::Read),
            buf("b", Access::Read),
            buf("out", Access::ReadWrite),
            length("len"),
        ],
        "k",
        "len",
        "out",
        0,
        binary(Arith::Mul, load("a"), load("b")),
    )
}

/// 覆盖 `-0.0`、次正规数、最大有限值与三类非有限值；其余位置用固定种子的 xorshift 填满。
fn adversarial() -> Vec<f32> {
    let mut values = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        -0.1,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1.0e-45,
        -1.0e-45,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let mut state = 0x9e37_79b9u32;
    while values.len() < LEN {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        values.push((state >> 8) as f32 / 16_777_216.0 * 8.0 - 4.0);
    }
    values
}

/// 旋转输入，让归约的两个操作数不共享同一组位模式。
fn rotated(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .cycle()
        .skip(5)
        .take(values.len())
        .copied()
        .collect()
}

fn c_float(value: f32) -> String {
    if value.is_nan() {
        return "NAN".to_string();
    }
    if value == f32::INFINITY {
        return "INFINITY".to_string();
    }
    if value == f32::NEG_INFINITY {
        return "(-INFINITY)".to_string();
    }
    format!("{value:?}f")
}

fn c_array(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| c_float(*value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn fixture(name: &str) -> PathBuf {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "cutile-ir-exec-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create fixture dir");
    dir
}

struct Run {
    rc: i32,
    bits: Vec<u32>,
}

/// 写入源码、用 `cc -O2 -ffp-contract=off` 编译、运行并解析 `rc` 与位模式。
fn run(name: &str, kernel: &str, main: &str) -> Run {
    let dir = fixture(name);
    let program = format!(
        "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include <math.h>\n\n\
         static uint32_t bits(float value) {{\n    uint32_t word;\n    memcpy(&word, &value, sizeof word);\n    return word;\n}}\n\n{kernel}\n{main}\n"
    );
    let source = dir.join("main.c");
    fs::write(&source, program).expect("write fixture source");
    let exe = dir.join("main");
    let compiled = Command::new("cc")
        .args(["-std=c11", "-O2", "-ffp-contract=off", "-o"])
        .arg(&exe)
        .arg(&source)
        .status()
        .expect("`cc` is required to execute emitted kernels");
    assert!(compiled.success(), "cc failed for {name}");
    let output = Command::new(&exe).output().expect("run fixture");
    assert!(
        output.status.success(),
        "fixture {name} exited with failure"
    );
    let text = String::from_utf8(output.stdout).expect("fixture stdout is utf8");
    let mut lines = text.lines();
    let first = lines.next().expect("fixture prints a status line");
    let rc = first
        .strip_prefix("rc ")
        .expect("status line starts with `rc `")
        .parse::<i32>()
        .expect("status line carries an integer");
    let bits = lines
        .map(|line| u32::from_str_radix(line, 16).expect("fixture prints hex bit patterns"))
        .collect();
    fs::remove_dir_all(&dir).expect("clean fixture dir");
    Run { rc, bits }
}

fn pointwise_main(call: &str, input: &[f32]) -> String {
    let mut main = String::new();
    let _ = writeln!(main, "int main(void) {{");
    let _ = writeln!(
        main,
        "    static const float input[] = {{{}}};",
        c_array(input)
    );
    let _ = writeln!(main, "    float output[{LEN}];");
    let _ = writeln!(
        main,
        "    for (size_t i = 0; i < {LEN}; ++i) output[i] = -12345.0f;"
    );
    let _ = writeln!(main, "    int rc = {call};");
    let _ = writeln!(main, "    printf(\"rc %d\\n\", rc);");
    let _ = writeln!(
        main,
        "    for (size_t i = 0; i < {LEN}; ++i) printf(\"%08x\\n\", bits(output[i]));"
    );
    let _ = writeln!(main, "    return 0;\n}}");
    main
}

fn reduce_main(name: &str, a: &[f32], b: &[f32], extent: usize) -> String {
    let mut main = String::new();
    let _ = writeln!(main, "int main(void) {{");
    let _ = writeln!(main, "    static const float a[] = {{{}}};", c_array(a));
    let _ = writeln!(main, "    static const float b[] = {{{}}};", c_array(b));
    let _ = writeln!(main, "    float out[1] = {{0.0f}};");
    let _ = writeln!(
        main,
        "    int rc = {name}(a, {LEN}, b, {LEN}, out, 1, {extent});"
    );
    let _ = writeln!(main, "    printf(\"rc %d\\n\", rc);");
    let _ = writeln!(main, "    printf(\"%08x\\n\", bits(out[0]));");
    let _ = writeln!(main, "    return 0;\n}}");
    main
}

/// 位模式相等；NaN 只要求两侧都是 NaN，因为载荷不参与语言语义。
fn assert_same(actual: &[u32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: value count");
    for (idx, (bits, reference)) in actual.iter().zip(expected).enumerate() {
        let value = f32::from_bits(*bits);
        let same = if value.is_nan() || reference.is_nan() {
            value.is_nan() && reference.is_nan()
        } else {
            value.to_bits() == reference.to_bits()
        };
        assert!(
            same,
            "{label}[{idx}]: emitted {value:?} ({bits:08x}) != reference {reference:?}"
        );
    }
}

#[test]
fn pointwise_emitted_c_matches_rust_reference_bit_for_bit() {
    let input = adversarial();
    let call = format!("affine(input, {LEN}, output, {LEN}, 1.5f, {LEN})");
    let run = run(
        "affine",
        &affine().emit_c().expect("affine emits C"),
        &pointwise_main(&call, &input),
    );
    assert_eq!(run.rc, 0);
    let expected: Vec<f32> = input.iter().map(|value| value * 1.5f32 + 3.0f32).collect();
    assert_same(&run.bits, &expected, "affine");
}

#[test]
fn negation_and_division_match_rust_including_signed_zero() {
    let input = adversarial();
    let call = format!("scaled_negation(input, {LEN}, output, {LEN}, {LEN})");
    let run = run(
        "scaled_negation",
        &scaled_negation().emit_c().expect("negation emits C"),
        &pointwise_main(&call, &input),
    );
    assert_eq!(run.rc, 0);
    let expected: Vec<f32> = input.iter().map(|value| -(value / 3.0f32)).collect();
    assert_same(&run.bits, &expected, "scaled_negation");
}

// 参考侧必须显式 `min`/`max` 镜像 IR 节点；adversarial 输入含 NaN，`clamp` 会 panic，故不替换。
#[allow(clippy::manual_clamp)]
#[test]
fn abs_min_max_emitted_c_matches_rust_bit_for_bit() {
    let input = adversarial();
    let call = format!("clipped(input, {LEN}, output, {LEN}, {LEN})");
    let run = run(
        "clipped",
        &clipped().emit_c().expect("clipped emits C"),
        &pointwise_main(&call, &input),
    );
    assert_eq!(run.rc, 0);
    let expected: Vec<f32> = input
        .iter()
        .map(|value| value.abs().min(2.0f32).max(-1.0f32))
        .collect();
    assert_same(&run.bits, &expected, "clipped");
}

#[test]
fn reduce_emitted_c_accumulates_in_the_same_order() {
    let a = adversarial();
    let b = rotated(&a);
    let run = run(
        "dot",
        &dot().emit_c().expect("reduce emits C"),
        &reduce_main("dot", &a, &b, LEN),
    );
    assert_eq!(run.rc, 0);
    let mut expected = 0.0f32;
    for (left, right) in a.iter().zip(&b) {
        expected += left * right;
    }
    assert_same(&run.bits, &[expected], "dot");
}

/// 生产者：`mid[i] = input[i] * scale + 3.0`，中间缓冲按可写声明。
fn producer() -> Kernel {
    Kernel::map(
        "producer",
        vec![
            buf("input", Access::Read),
            buf("mid", Access::ReadWrite),
            scalar("scale"),
            length("extent"),
        ],
        "i",
        "extent",
        "mid",
        binary(
            Arith::Add,
            binary(Arith::Mul, load("input"), Expr::Scalar("scale".to_string())),
            Expr::F32(3.0f32.to_bits()),
        ),
    )
}

/// 消费者：`output[i] = mid[i] * 2.0 - 1.0`；中间缓冲在这里是只读，融合才有链接可消。
fn consumer() -> Kernel {
    Kernel::map(
        "consumer",
        vec![
            buf("mid", Access::Read),
            buf("output", Access::ReadWrite),
            length("extent"),
        ],
        "i",
        "extent",
        "output",
        binary(
            Arith::Sub,
            binary(Arith::Mul, load("mid"), Expr::F32(2.0f32.to_bits())),
            Expr::F32(1.0f32.to_bits()),
        ),
    )
}

/// 只读某个名字的消费者：分别命中「没有链接」与「链接可写」两闸。
fn consumer_of(name: &str, access: Access) -> Kernel {
    Kernel::map(
        "after",
        vec![
            buf(name, access),
            buf("out", Access::ReadWrite),
            length("extent"),
        ],
        "i",
        "extent",
        "out",
        binary(Arith::Mul, load(name), Expr::F32(2.0f32.to_bits())),
    )
}

/// 融合消掉中间缓冲、只留一趟循环，结果与「先写中间、再读中间」逐位相同。
#[test]
fn fusion_removes_the_intermediate_buf_and_keeps_every_bit() {
    let input = adversarial();
    let fused = producer().fuse(&consumer()).expect("pointwise chain fuses");
    let source = fused.emit_c().expect("fused kernel emits C");
    assert!(
        !source.contains("mid"),
        "intermediate buf must be gone:\n{source}"
    );
    let call = format!("producer_consumer(input, {LEN}, 1.5f, {LEN}, output, {LEN})");
    let run = run("fused", &source, &pointwise_main(&call, &input));
    assert_eq!(run.rc, 0);
    let expected: Vec<f32> = input
        .iter()
        .map(|value| {
            let mid = value * 1.5f32 + 3.0f32;
            mid * 2.0f32 - 1.0f32
        })
        .collect();
    assert_same(&run.bits, &expected, "fused");
}

/// 无法拼接的链在执行前就被挡住：归约不可融、缺链接、链接可写。
#[test]
fn fusion_rejects_chains_it_cannot_join() {
    let error = affine().fuse(&dot()).expect_err("reductions cannot fuse");
    assert!(
        error.to_string().contains("two pointwise kernels"),
        "{error}"
    );

    let error = affine()
        .fuse(&consumer_of("tail", Access::Read))
        .expect_err("consumer must read the producer output");
    assert!(
        error.to_string().contains("does not read `output`"),
        "{error}"
    );

    let error = affine()
        .fuse(&consumer_of("output", Access::ReadWrite))
        .expect_err("link must be read-only");
    assert!(
        error.to_string().contains("read-only f32 buffer"),
        "{error}"
    );
}

#[test]
fn length_guards_reject_short_bufs_before_any_write() {
    let input = adversarial();
    let sentinel = (-12345.0f32).to_bits();
    let call = format!("affine(input, {LEN}, output, {}, 1.5f, {LEN})", LEN - 1);
    let run = run(
        "short",
        &affine().emit_c().expect("affine emits C"),
        &pointwise_main(&call, &input),
    );
    assert_eq!(run.rc, 1, "short output buffer must be reported");
    assert!(
        run.bits.iter().all(|bits| *bits == sentinel),
        "no element may be written"
    );
}

#[test]
fn zero_extent_writes_nothing_and_reports_success() {
    let input = adversarial();
    let sentinel = (-12345.0f32).to_bits();
    let call = format!("affine(input, {LEN}, output, {LEN}, 1.5f, 0)");
    let run = run(
        "empty",
        &affine().emit_c().expect("affine emits C"),
        &pointwise_main(&call, &input),
    );
    assert_eq!(run.rc, 0);
    assert!(
        run.bits.iter().all(|bits| *bits == sentinel),
        "zero extent writes nothing"
    );
}

/// 前端宏与手写 IR 走同一条执行链：宏只负责把 Rust 形状翻译成 `Kernel`。
#[crate::dsl]
#[allow(dead_code)]
fn fused(input: &[f32], output: &mut [f32], scale: f32, extent: usize) {
    for i in 0..extent {
        output[i] = input[i] * scale + 3.0;
    }
}

#[test]
fn macro_front_end_output_matches_rust_after_execution() {
    let input = adversarial();
    let call = format!("fused(input, {LEN}, output, {LEN}, 1.5f, {LEN})");
    let run = run(
        "fused",
        &__dsl_ir_fused().emit_c().expect("macro IR emits C"),
        &pointwise_main(&call, &input),
    );
    assert_eq!(run.rc, 0);
    let expected: Vec<f32> = input.iter().map(|value| value * 1.5f32 + 3.0f32).collect();
    assert_same(&run.bits, &expected, "fused");
}

/// 归约宏前端与手写 `Kernel` 走同一条执行链，输出位模式逐位一致。
#[crate::dsl]
#[allow(dead_code)]
fn dot_reduce(a: &[f32], b: &[f32], out: &mut [f32], len: usize) {
    for k in 0..len {
        out[0] += a[k] * b[k];
    }
}

/// 宏前端把 `.abs()` / `.min()` / `.max()` 翻成同一组新组合子，执行链与手写 IR 逐位一致。
#[crate::dsl]
#[allow(dead_code, clippy::manual_clamp)]
fn clipped_dsl(input: &[f32], output: &mut [f32], extent: usize) {
    for i in 0..extent {
        output[i] = input[i].abs().min(2.0).max(-1.0);
    }
}

#[allow(clippy::manual_clamp)]
#[test]
fn macro_method_front_end_matches_rust_after_execution() {
    let input = adversarial();
    let call = format!("clipped_dsl(input, {LEN}, output, {LEN}, {LEN})");
    let run = run(
        "clipped_dsl",
        &__dsl_ir_clipped_dsl()
            .emit_c()
            .expect("macro method IR emits C"),
        &pointwise_main(&call, &input),
    );
    assert_eq!(run.rc, 0);
    let expected: Vec<f32> = input
        .iter()
        .map(|value| value.abs().min(2.0f32).max(-1.0f32))
        .collect();
    assert_same(&run.bits, &expected, "clipped_dsl");
}

#[test]
fn macro_reduce_front_end_matches_rust_after_execution() {
    let a = adversarial();
    let b = rotated(&a);
    let run = run(
        "dot_reduce",
        &__dsl_ir_dot_reduce()
            .emit_c()
            .expect("macro reduce emits C"),
        &reduce_main("dot_reduce", &a, &b, LEN),
    );
    assert_eq!(run.rc, 0);
    let mut expected = 0.0f32;
    for (left, right) in a.iter().zip(&b) {
        expected += left * right;
    }
    assert_same(&run.bits, &[expected], "dot_reduce");
}

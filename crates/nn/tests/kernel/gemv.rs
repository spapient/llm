use super::*;
use crate::gemv::Format;
use crate::model::{ShapeCtx, Values};
use diagnostic::Code;

fn shape(m: usize, n: usize) -> ShapeCtx {
    MODEL.projection(m, n).expect("两个投影轴都声明了")
}

#[test]
fn cost_reads_three_f32_operands_and_counts_one_multiply_add_per_element() {
    // 小形状逐项可算：w 是 m·n 个 f32、x 是 m 个、y 是 n 个；flops = 2·m·n。
    let cost = gemv::cost::<F32>(&shape(8, 4)).expect("合法形状");
    assert_eq!(cost.read, (32 + 8) * 4, "w 与 x 都是 f32");
    assert_eq!(cost.write, 4 * 4, "y 是 n 个 f32 输出");
    assert_eq!(cost.flops, 2.0 * 8.0 * 4.0);
}

#[test]
fn a_zero_extent_is_rejected_instead_of_silently_priced() {
    let error = ShapeCtx::new([('m', 8), ('n', 0)]).expect_err("n = 0");
    assert_eq!(error.code(), Code::Invalid);
    assert!(
        ShapeCtx::new([('m', 0), ('n', 4)]).is_err(),
        "m = 0 同样不自洽"
    );
}

/// 稠密算子的形状粒度是 1：一条只为对齐 nibble 的约束不该溢出到 f32 权重上。
#[test]
fn any_width_is_a_legal_dense_shape() {
    assert_eq!(MODEL.group, 1);
    let odd = gemv::cost::<F32>(&shape(100, 3)).expect("100 不是 32 的倍数也合法");
    assert!(odd.read > 0);
    // 模板用 `len` 截断最后一块，参考实现也必须能算这个宽度。
    let sample = gemv::sample::<F32>(100, 3);
    let y = gemv::reference::<F32>(&sample).expect("分块余数不应报错");
    assert_eq!(y.len(), 3);
}

#[test]
fn sample_is_deterministic_and_sized_to_the_shape() {
    let first = gemv::sample::<F32>(64, 4);
    assert_eq!(first, gemv::sample::<F32>(64, 4));
    assert_eq!(first.f32("w").expect("w").len(), 4 * 64);
    assert_eq!(first.f32("x").expect("x").len(), 64);
    assert_eq!(first.f32("y").expect("y").len(), 4);
    assert_eq!(first.scalar("gsz"), Some(BLOCK as i64));
}

#[test]
fn the_scalar_reference_matches_a_hand_computed_dot_product() {
    // 一行的权重是 1..=4，输入是 1、2、3、4 → 1+4+9+16 = 30。
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let w = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = [0.0f32; 1];
    forward(&x, &w, &mut y).expect("合法");
    assert_eq!(y, [30.0]);
}

#[test]
fn the_reference_is_exact_on_the_deterministic_sample() {
    // 取值都在 2 的负幂网格上：累加在 f32 里精确，所以结果可以被逐位钉死。
    // 期望值由同一段算术算出，再用"换顺序必须等价"验一遍：分块边界的移动不改变结果。
    let sample = gemv::sample::<F32>(64, 2);
    let y = gemv::reference::<F32>(&sample).expect("合法");
    let x = sample.f32("x").expect("x");
    let w = sample.f32("w").expect("w");
    for (row, value) in y.iter().enumerate() {
        let flat: f32 = (0..64).map(|k| w[row * 64 + k] * x[k]).sum();
        assert_eq!(*value, flat, "精确网格上的分块累加必须等价于顺序累加");
    }
}

/// 内核本体与标量参考在同一份确定性样本上逐位一致（softmax / rope 同口径）。
#[test]
fn forward_matches_the_scalar_reference_on_the_deterministic_sample() {
    let sample = gemv::sample::<F32>(64, 4);
    let expected = gemv::reference::<F32>(&sample).expect("参考可计算");
    let x = sample.f32("x").expect("x");
    let w = sample.f32("w").expect("w");
    let mut out = vec![0.0f32; expected.len()];
    forward(x, w, &mut out).expect("合法形状");
    assert_eq!(out, expected, "稠密 GEMV 内核与参考必须逐位一致");
}

#[test]
fn a_row_that_is_shorter_than_declared_is_rejected() {
    let x = [1.0f32; 4];
    let w = [1.0f32; 4];
    let mut y = [0.0f32; 2];
    let error = forward(&x, &w, &mut y).expect_err("两行只有一行权重");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("权重长度"), "{error}");
}

/// 稠密格式不接受标度：换错算子必须报错，而不是"算出一个看起来对的数"。
#[test]
fn the_dense_format_refuses_a_scales_buf() {
    let x = [1.0f32; 4];
    let w = [1.0f32; 4];
    let mut y = [0.0f32; 1];
    let error = <F32 as Format>::forward(&x, &w, &[1.0], &mut y).expect_err("稠密格式没有标度");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("标度"), "{error}");
}

/// 样本的权重类型由格式决定：稠密格式的 `w` 是 f32。两道闸都要在——`Sample::set` 拒绝换类型
/// （桥接期的闸），骨架自己也拒绝（直接改出来的 `[u8]` 权重不能按 f32 读，那会算出一个看起来
/// 对的数）。
#[test]
fn the_shared_skeleton_rejects_a_weight_of_the_wrong_element_type() {
    let mut sample = gemv::sample::<F32>(8, 2);
    let error = sample
        .set("w", Values::U8(vec![0; 16]))
        .expect_err("float 槽位不能换成字节");
    assert_eq!(error.code(), Code::Invalid);
    let wrong = sample
        .bufs
        .iter_mut()
        .find(|buf| buf.name == "w")
        .expect("稠密样本有 w");
    wrong.values = Values::U8(vec![0; 16]);
    let error = gemv::reference::<F32>(&sample).expect_err("u8 不是 f32 权重");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("f32"), "{error}");
}

#[test]
fn call_arguments_follow_the_template_signature() {
    let signature = crate::sig::Signature::parse(
        "gemv_f32",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../prof/src/ops/gemv_f32.k")),
    )
    .expect("签名可解析");
    assert_eq!(
        signature
            .call_args(&gemv::sample::<F32>(64, 4))
            .expect("实参"),
        "x, w, y, 64, 4, 32"
    );
    assert_eq!(signature.output(), Some("y"));
}

/// `Model::group` 必须来自 [`Format::GRAIN`]，不是注解另写的一个数。
///
/// 这条断言原先写在注册表侧（那里同时看得见 `MODEL` 与格式），拆 crate 后注册表看不见内核类型，
/// 于是它回到内核：格式的粒度由格式自己回答，注册表只需照抄。
fn grain_is_what_the_model_declares<G: Format + crate::operator::Operator>() {
    assert_eq!(
        crate::operator::model_of::<G>().group,
        u32::try_from(<G as Format>::GRAIN).expect("粒度落在 u32 内")
    );
}

#[test]
fn the_declared_group_is_the_format_grain() {
    grain_is_what_the_model_declares::<F32>();
    grain_is_what_the_model_declares::<crate::gemv::i2::I2>();
    grain_is_what_the_model_declares::<crate::gemv::i4::I4>();
    grain_is_what_the_model_declares::<crate::gemv::i8::I8>();
}

/// A1 的真证据：`simd::elementwise` 有 `CHAINS · WIDE` 条独立累加链，求和顺序与顺序标量不同，**不可能逐位相等**；
/// 于是按 A1 的另一半要求钉住误差上界。
///
/// 上面的 `forward_matches_the_scalar_reference_on_the_deterministic_sample` 不能当证据：
/// `gemv::reference` 内部就是 `F::forward`，那是自己跟自己比（F6）。这里用 f64 顺序累加当
/// 真参考——f32×f32 落进 f64 是精确的（24+24 ≤ 53 位），4096 项 f64 求和的相对误差
/// ≈ 4096·2⁻⁵³ ≈ 4.5e-13，比 f32 的界小 9 个数量级，故 `exact as f32` 就是可用的真值。
#[test]
fn the_dense_dot_stays_inside_the_gamma_bound_against_a_wider_reference() {
    const M: usize = 4096;
    // 3/7 与 1/3 在二进制里都不精确，累加不可精确 → 重排误差真实可见。
    let weight: Vec<f32> = (0..M)
        .map(|k| ((k * 37 % 101) as f32 - 50.0) / 7.0)
        .collect();
    let x: Vec<f32> = (0..M)
        .map(|k| ((k * 11 % 53) as f32 - 26.0) / 3.0)
        .collect();
    let exact: f64 = (0..M).map(|k| f64::from(weight[k]) * f64::from(x[k])).sum();
    let magnitude: f64 = (0..M)
        .map(|k| (f64::from(weight[k]) * f64::from(x[k])).abs())
        .sum();
    let mut y = [0.0f32; 1];
    forward(&x, &weight, &mut y).expect("合法形状");
    // γ_M = M·u/(1−M·u)，u = 2⁻²⁴。逐行版的重排更松，γ_M 是上界（误差方向保守）。
    let u = f64::from(f32::EPSILON) / 2.0;
    let gamma = M as f64 * u / (1.0 - (M as f64) * u);
    let error = (f64::from(y[0]) - exact).abs();
    let ulps = (y[0] as f64 - exact).abs() / (exact.abs() * u);
    assert!(
        error <= gamma * magnitude,
        "实测 {error:e} 超出 γ_M·Σ|wx| = {:e}（≈{ulps:.1} ulp）",
        gamma * magnitude
    );
    assert!(magnitude > 0.0, "样本不能退化成零");
}

/// 行分块路径（`Plain::dot_four`）的 A1 证据：`forward_dense` 把前 `n − n mod COLS` 行走
/// [`COLS`](crate::gemv::simd::COLS) 行并行、余数行走单行 `dot`，两条路径的求和顺序不同。
///
/// 这里故意取奇数 `N = 5`：若块宽推导写成 `n · m` 而不是 `COLS · m`，第 4 行会读到别行的权重，
/// 或者尾行拿不到数据——这正是本用例要钉住的边界，而单行版测试覆盖不到。
#[test]
fn the_row_blocked_forward_stays_inside_the_gamma_bound_with_an_odd_row_count() {
    const M: usize = 512;
    const N: usize = 5;
    let weight: Vec<f32> = (0..M * N)
        .map(|k| ((k * 41 % 97) as f32 - 48.0) / 5.0)
        .collect();
    let x: Vec<f32> = (0..M)
        .map(|k| ((k * 13 % 59) as f32 - 29.0) / 3.0)
        .collect();
    let mut y = vec![0.0f32; N];
    forward(&x, &weight, &mut y).expect("合法形状");
    let u = f64::from(f32::EPSILON) / 2.0;
    let gamma = M as f64 * u / (1.0 - (M as f64) * u);
    for r in 0..N {
        let row = &weight[r * M..][..M];
        let exact: f64 = (0..M).map(|k| f64::from(row[k]) * f64::from(x[k])).sum();
        let magnitude: f64 = (0..M)
            .map(|k| (f64::from(row[k]) * f64::from(x[k])).abs())
            .sum();
        let error = (f64::from(y[r]) - exact).abs();
        assert!(
            error <= gamma * magnitude,
            "第 {r} 行实测 {error:e} 超出 γ_M·Σ|wx| = {:e}",
            gamma * magnitude
        );
        assert!(magnitude > 0.0, "第 {r} 行样本退化");
    }
}

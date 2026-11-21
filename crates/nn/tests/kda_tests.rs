//! `kda` 的用例。
//!
//! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。

use super::*;

/// 3 通道键、2 维值的层：手算跟得上。
fn kda() -> Kda {
    Kda::new(3, 2).expect("几何合法")
}

/// 单位门：什么都不忘。
fn keep() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}

/// 零记忆。
fn zeroed() -> Memory<f64> {
    kda().memory()
}

/// 单位键 `k / ‖k‖`：写入与读取都用它，精确性才可比。
fn unit(key: [f64; 3]) -> [f64; 3] {
    let norm = key.iter().map(|value| value * value).sum::<f64>().sqrt();
    [key[0] / norm, key[1] / norm, key[2] / norm]
}

/// 两个正交的单位键：写一个不该影响另一个。
fn axes() -> ([f64; 3], [f64; 3]) {
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
}

/// 两个正交键各记一条关联：键 `first` 记 `[1, 0]`、键 `second` 记 `[0, 4]`；
/// 返回层、写完的记忆与两个键。
fn two_records() -> (Kda, Memory<f64>, [f64; 3], [f64; 3]) {
    let layer = kda();
    let mut memory = zeroed();
    let (first, second) = axes();
    layer
        .step(&mut memory, &keep(), 1.0, &first, &[1.0, 0.0])
        .expect("几何合法");
    layer
        .step(&mut memory, &keep(), 1.0, &second, &[0.0, 4.0])
        .expect("几何合法");
    (layer, memory, first, second)
}

/// 写一次、读一次：读出就是写进去的值（`q = k̂` 时的精确性）。
#[test]
fn a_write_is_exactly_recalled() {
    let layer = kda();
    let mut memory = zeroed();
    let key = unit([0.3, 0.4, 0.5]);
    let value = [1.0, -2.0];
    layer
        .step(&mut memory, &keep(), 1.0, &key, &value)
        .expect("几何合法");
    let got = layer.read(&memory, &key).expect("几何合法");
    for (index, (read, wrote)) in got.iter().zip(&value).enumerate() {
        assert!(
            (read - wrote).abs() < 1e-12,
            "第 {index} 维读出 {read}，写进去的是 {wrote}"
        );
    }
}

/// 同一个键写两次、值不同，读出是**后一次**的值：这是 delta 规则与指数梯形的分界。
#[test]
fn the_delta_rule_overwrites_instead_of_accumulating() {
    let layer = kda();
    let mut memory = zeroed();
    let (key, _) = axes();
    layer
        .step(&mut memory, &keep(), 1.0, &key, &[1.0, 2.0])
        .expect("几何合法");
    layer
        .step(&mut memory, &keep(), 1.0, &key, &[-3.0, 0.5])
        .expect("几何合法");
    let got = layer.read(&memory, &key).expect("几何合法");
    assert!(
        (got[0] + 3.0).abs() < 1e-12,
        "第 0 维是 {}，累加的话会是 -2",
        got[0]
    );
    assert!(
        (got[1] - 0.5).abs() < 1e-12,
        "第 1 维是 {}，累加的话会是 2.5",
        got[1]
    );
}

/// 正交的两个键各记各的：写第二个不扰动第一个。
#[test]
fn two_orthogonal_keys_keep_their_own_values() {
    let (layer, memory, first, second) = two_records();
    let a = layer.read(&memory, &first).expect("几何合法");
    let b = layer.read(&memory, &second).expect("几何合法");
    assert!((a[0] - 1.0).abs() < 1e-12 && a[1].abs() < 1e-12, "{a:?}");
    assert!(b[0].abs() < 1e-12 && (b[1] - 4.0).abs() < 1e-12, "{b:?}");
}

/// 门是逐通道的：关掉通道 0 只让键 0 的关联失效，通道 1 的照旧。
#[test]
fn the_gate_forgets_one_channel_at_a_time() {
    let (layer, mut memory, first, second) = two_records();
    // 只在通道 0 上遗忘，这一次不写（β = 0）。
    layer
        .step(&mut memory, &[0.0, 1.0, 1.0], 0.0, &first, &[0.0, 0.0])
        .expect("几何合法");
    let gone = layer.read(&memory, &first).expect("几何合法");
    assert!(
        gone[0].abs() < 1e-12,
        "通道 0 被清空，键 0 的关联应当读不出来，得到 {}",
        gone[0]
    );
    let kept = layer.read(&memory, &second).expect("几何合法");
    assert!(
        (kept[1] - 4.0).abs() < 1e-12,
        "通道 1 没忘，键 1 的关联还该在，得到 {}",
        kept[1]
    );
}

/// 写强度按比例缩放纠错量：`β = 0.25` 读到的就是满写的四分之一。
#[test]
fn the_write_strength_scales_the_correction() {
    let layer = kda();
    let (key, _) = axes();
    let value = [3.0, -1.0];
    let mut full = zeroed();
    layer
        .step(&mut full, &keep(), 1.0, &key, &value)
        .expect("几何合法");
    let mut quarter = zeroed();
    layer
        .step(&mut quarter, &keep(), 0.25, &key, &value)
        .expect("几何合法");
    let want = layer.read(&full, &key).expect("几何合法");
    let got = layer.read(&quarter, &key).expect("几何合法");
    for (index, (read, full)) in got.iter().zip(&want).enumerate() {
        assert!(
            (read - full * 0.25).abs() < 1e-12,
            "第 {index} 维 {read} 不是满写的四分之一（{full}）"
        );
    }
}

/// 写强度为零时只留下遗忘：`α = 0.5` 且不写，读出减半。
#[test]
fn a_zero_write_strength_only_forgets() {
    let layer = kda();
    let mut memory = zeroed();
    let (key, _) = axes();
    layer
        .step(&mut memory, &keep(), 1.0, &key, &[2.0, -2.0])
        .expect("几何合法");
    layer
        .step(&mut memory, &[0.5; 3], 0.0, &key, &[0.0, 0.0])
        .expect("几何合法");
    let got = layer.read(&memory, &key).expect("几何合法");
    assert!(
        (got[0] - 1.0).abs() < 1e-12 && (got[1] + 1.0).abs() < 1e-12,
        "{got:?}"
    );
}

/// 键的范数不影响写入强度：归一化在入口做，缩放键不会改变记忆。
#[test]
fn the_key_is_normalized_at_the_door() {
    let layer = kda();
    let mut short = zeroed();
    let mut long = zeroed();
    let key = [0.2, -0.3, 0.6];
    let stretched = [1.2, -1.8, 3.6];
    layer
        .step(&mut short, &keep(), 1.0, &key, &[1.5, -0.5])
        .expect("几何合法");
    layer
        .step(&mut long, &keep(), 1.0, &stretched, &[1.5, -0.5])
        .expect("几何合法");
    for (cell, (a, b)) in short.cells().iter().zip(long.cells()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "第 {cell} 格 {a} 与 {b} 不同：键的范数不该改变写入强度"
        );
    }
}

/// 零键被拒：归一化它只会得到 `NaN`，而残差会退化成 `v`。
#[test]
fn a_zero_key_is_refused() {
    let mut memory = zeroed();
    let refused = kda()
        .step(&mut memory, &keep(), 1.0, &[0.0; 3], &[1.0, 2.0])
        .expect_err("零键不能归一");
    assert_eq!(refused.code(), Code::Invalid);
    assert_eq!(memory.cells(), &[0.0; 6], "被拒的一次不该改动记忆");
}

/// 非有限输入在入口被拒，且记忆保持原样。
#[test]
fn non_finite_inputs_are_refused() {
    let layer = kda();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut memory = zeroed();
        let key = layer
            .step(&mut memory, &keep(), 1.0, &[1.0, bad, 0.0], &[1.0, 2.0])
            .expect_err("键非有限");
        assert_eq!(key.code(), Code::Invalid);
        let value = layer
            .step(&mut memory, &keep(), 1.0, &[1.0, 0.0, 0.0], &[bad, 2.0])
            .expect_err("值非有限");
        assert_eq!(value.code(), Code::Invalid);
        let query = layer
            .read(&memory, &[bad, 0.0, 0.0])
            .expect_err("查询非有限");
        assert_eq!(query.code(), Code::Invalid);
        assert_eq!(memory.cells(), &[0.0; 6]);
    }
}

/// 门是遗忘系数：越界拒绝，两个端点合法。
#[test]
fn the_gate_is_a_forget_gate() {
    let layer = kda();
    let key = [1.0, 0.0, 0.0];
    let value = [1.0, 1.0];
    for bad in [-0.5, 1.5, f64::NAN] {
        let mut memory = zeroed();
        let refused = layer
            .step(&mut memory, &[bad, 1.0, 1.0], 1.0, &key, &value)
            .expect_err("门越界");
        assert_eq!(refused.code(), Code::Invalid);
    }
    for edge in [0.0, 1.0] {
        let mut memory = zeroed();
        layer
            .step(&mut memory, &[edge; 3], 1.0, &key, &value)
            .expect("两个端点都是合法遗忘系数");
    }
}

/// 写强度是比例：越界拒绝，端点合法。
#[test]
fn the_write_strength_is_a_fraction() {
    let layer = kda();
    let key = [1.0, 0.0, 0.0];
    let value = [1.0, 1.0];
    for bad in [-0.5, 1.5, f64::NAN] {
        let mut memory = zeroed();
        let refused = layer
            .step(&mut memory, &keep(), bad, &key, &value)
            .expect_err("写强度越界");
        assert_eq!(refused.code(), Code::Invalid);
    }
    for edge in [0.0, 1.0] {
        let mut memory = zeroed();
        layer
            .step(&mut memory, &keep(), edge, &key, &value)
            .expect("两个端点都是合法写强度");
    }
}

/// 构造期非零校验，运行期每处宽度与状态归属都在入口判定。
#[test]
fn geometry_is_checked_at_the_door() {
    assert_eq!(Kda::new(0, 2).expect_err("键维为零").code(), Code::Invalid);
    assert_eq!(Kda::new(2, 0).expect_err("值维为零").code(), Code::Invalid);

    let layer = kda();
    let key = [1.0, 0.0, 0.0];
    let value = [1.0, 1.0];
    let mut memory = zeroed();
    let narrow_gate = layer
        .step(&mut memory, &[1.0, 1.0], 1.0, &key, &value)
        .expect_err("门窄了一格");
    assert_eq!(narrow_gate.code(), Code::Invalid);
    let narrow_key = layer
        .step(&mut memory, &keep(), 1.0, &[1.0, 0.0], &value)
        .expect_err("键窄了一格");
    assert_eq!(narrow_key.code(), Code::Invalid);
    let narrow_value = layer
        .step(&mut memory, &keep(), 1.0, &key, &[1.0])
        .expect_err("值窄了一格");
    assert_eq!(narrow_value.code(), Code::Invalid);
    let narrow_query = layer.read(&memory, &[1.0, 0.0]).expect_err("查询窄了一格");
    assert_eq!(narrow_query.code(), Code::Invalid);

    let mut foreign = Kda::new(2, 3).expect("几何合法").memory::<f64>();
    let mismatched = layer
        .step(&mut foreign, &keep(), 1.0, &key, &value)
        .expect_err("记忆属于别的层");
    assert_eq!(mismatched.code(), Code::Invalid);
}

/// 记忆是行主序 `[值维, 键维]`：一个通道上的写只落在对应列。
#[test]
fn the_memory_is_row_major_value_by_key() {
    let mut memory = zeroed();
    kda()
        .step(&mut memory, &keep(), 1.0, &[1.0, 0.0, 0.0], &[1.0, -2.0])
        .expect("几何合法");
    assert_eq!(memory.cells(), &[1.0, 0.0, 0.0, -2.0, 0.0, 0.0]);
    assert_eq!(memory.row(0), Some(&[1.0, 0.0, 0.0][..]));
    assert_eq!(memory.row(1), Some(&[-2.0, 0.0, 0.0][..]));
    assert_eq!(memory.row(2), None);
    assert_eq!(memory.key_dim(), 3);
    assert_eq!(memory.value_dim(), 2);
}

/// 没写过的记忆读出零，而不是残缺的记忆。
#[test]
fn an_untouched_memory_reads_zero() {
    let got = kda().read(&zeroed(), &[1.0, 2.0, 3.0]).expect("几何合法");
    assert_eq!(got, vec![0.0, 0.0]);
}

/// `f32` 与 `f64` 同一套算术：宽窄差别只在精度。
#[test]
fn f32_and_f64_agree() {
    let layer = kda();
    let key64: [f64; 3] = unit([0.3, -0.4, 0.5]);
    let key32: [f32; 3] = [key64[0] as f32, key64[1] as f32, key64[2] as f32];
    let mut wide = zeroed();
    layer
        .step(&mut wide, &[0.75; 3], 0.75, &key64, &[1.5, -0.25])
        .expect("几何合法");
    layer
        .step(&mut wide, &[0.9, 0.5, 0.25], 0.5, &key64, &[-1.0, 2.0])
        .expect("几何合法");
    let mut narrow: Memory<f32> = layer.memory();
    layer
        .step(&mut narrow, &[0.75; 3], 0.75, &key32, &[1.5, -0.25])
        .expect("几何合法");
    layer
        .step(&mut narrow, &[0.9, 0.5, 0.25], 0.5, &key32, &[-1.0, 2.0])
        .expect("几何合法");
    for (cell, (a, b)) in wide.cells().iter().zip(narrow.cells()).enumerate() {
        assert!(
            (a - f64::from(*b)).abs() < 1e-6,
            "第 {cell} 格 f64 是 {a}、f32 是 {b}"
        );
    }
}

//! gemm 数值内核的**定律**测试：每条性质都必须由数学定义推出，而不是复述实现。
//!
//! 判据分三档，按"能有多强就用多强"取：
//!
//! - **精确相等**（[`prop::assert_exact`]）：整数累加、取反律、量化网格落点、`pos = 0` 的恒等。
//!   这些地方的容差只会掩盖错误。
//! - **相对误差**（[`prop::assert_within_rel`]）：不同累加顺序、`f32` 三角函数这类"数学上相等、
//!   浮点上只到 `1e-7`"的定律。
//! - **稳定性**：宽动态范围输入下不得出 `NaN`/`inf`、且排序关系（`argmax`）保持——这是
//!   `max` 减去、在线归一化这类写法唯一能被自动抓住的地方。

use crate::matmul::matmul_scalar;
use crate::quant::{I2, I4, width_cpb};
use crate::{
    dequant, dequant_i2, dequant_i4, fast, gemv_f32, gemv_i2, gemv_i4, gemv_i8, rmsnorm_f32,
    rope_f32, softmax_f32,
};
use proptest::prelude::*;
use std::rc::Rc;
use testkit::prop;

/// 走一遍 softmax 并把结果拿出来；长度合法由调用方保证。
fn softmax(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    softmax_f32::forward(tensor::View::one(x), &mut out).expect("等长");
    out
}

fn rmsnorm(x: &[f32], gamma: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    rmsnorm_f32::forward(tensor::View::one(x), tensor::View::one(gamma), &mut out).expect("等长");
    out
}

fn rope(pos: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    rope_f32::forward(pos, tensor::View::one(x), &mut out).expect("等长");
    out
}

/// 把任意向量缩放成 `mean(x²) ≈ 1`：均方根归一化的输入应当近似原样穿过 rmsnorm。
fn unit_mean_square(values: &[f32]) -> Vec<f32> {
    let energy: f64 = values.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let scale = (values.len() as f64 / energy).sqrt() as f32;
    values.iter().map(|v| v * scale).collect()
}

fn matrix(rows: usize, cols: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(prop::finite(), rows * cols)
}

/// 小整数取值：乘积与部分和都落在 f32 精确整数区间内，于是线性定律可断言逐位相等。
fn small_int() -> impl Strategy<Value = f32> {
    (-4i8..=4).prop_map(f32::from)
}

fn ints(len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(small_int(), len)
}

/// 幅值落在 `[1/8, 2)` 的标量：两个这样的数相乘 ≥ 1/64，`m ≤ 96` 的部分和 ≤ 384，
/// 既不下溢成次正规数也不上溢 —— `γ_m` 界成立的前提（每步都正常舍入）就满足了。
fn scaled_scalar() -> impl Strategy<Value = f32> {
    (prop::finite(), any::<bool>()).prop_map(|(v, negative)| {
        let magnitude = 0.125 + v.abs() * 1.875;
        if negative { -magnitude } else { magnitude }
    })
}

fn scaled(len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(scaled_scalar(), len)
}

/// 稠密 GEMV 一行定义：`out[j] = Σ_k w[j·m+k]·x[k]`。
fn gemv_f32_of(x: &[f32], w: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    gemv_f32::forward(tensor::View::one(x), w, &mut out).expect("形状非空");
    out
}

/// 稠密定义，逐行顺序点积（与内核的累加顺序不同，只用来比舍入）。
fn dense_reference(x: &[f32], w: &[f32], n: usize) -> Vec<f32> {
    let m = x.len();
    (0..n)
        .map(|j| (0..m).map(|k| w[j * m + k] * x[k]).sum())
        .collect()
}

/// i8 定义：分组内点积、整组乘标度、跨组相加；权重字节按 `i8` 解释。
fn i8_reference(x: &[f32], w: &[u8], s: &[f32], n: usize, groups: usize) -> Vec<f32> {
    let m = x.len();
    let group = m / groups;
    (0..n)
        .map(|j| {
            (0..groups)
                .map(|g| {
                    let dot: f32 = (0..group)
                        .map(|k| f32::from(w[j * m + g * group + k] as i8) * x[g * group + k])
                        .sum();
                    dot * s[j * groups + g]
                })
                .sum()
        })
        .collect()
}

/// 打包格式定义：低位槽在前按 `cpb` 位宽解包、查码表还原有符号码、整组共用一条标度。
fn packed_reference(
    x: &[f32],
    w: &[u8],
    s: &[f32],
    n: usize,
    groups: usize,
    cpb: usize,
    table: &[f32],
) -> Vec<f32> {
    let group = x.len() / groups;
    let row = x.len() / cpb;
    let width = 8 / cpb;
    let mask = (1u8 << width) - 1;
    (0..n)
        .map(|j| {
            (0..groups)
                .map(|g| {
                    let dot: f32 = (0..group)
                        .map(|k| {
                            let elem = g * group + k;
                            let byte = w[j * row + elem / cpb];
                            let code = usize::from((byte >> ((elem % cpb) * width)) & mask);
                            table[code] * x[elem]
                        })
                        .sum();
                    dot * s[j * groups + g]
                })
                .sum()
        })
        .collect()
}

proptest! {
    #![proptest_config(prop::config())]

    /// 概率分布的**定义**：非负、和为 1、每项不超过 1。这是 softmax 唯一不可让步的性质。
    #[test]
    fn softmax_rows_form_a_probability_distribution(x in prop::vector(1..=64)) {
        let out = softmax(&x);
        prop_assert!(prop::all_finite(&out), "输出出现非有限值");
        prop_assert!(out.iter().all(|p| *p > 0.0 && *p <= 1.0), "越出 (0, 1]");
        let total: f32 = out.iter().sum();
        prop::assert_within_rel(&[total], &[1.0], 1e-5, "softmax 行和为 1");
    }

    /// 平移不变：`softmax(x + c) = softmax(x)`。缺了减最大值的写法会在这里溢出。
    #[test]
    fn softmax_is_shift_invariant(x in prop::vector(1..=32), shift in -10.0f32..=10.0) {
        let shifted: Vec<f32> = x.iter().map(|v| v + shift).collect();
        prop::assert_within_rel(&softmax(&shifted), &softmax(&x), 1e-5, "softmax(x + c)");
    }

    /// 保序：`x[i] ≤ x[j] ⟹ out[i] ≤ out[j]`。exp 单调，任何"近似"实现都可能悄悄破坏它。
    #[test]
    fn softmax_is_order_preserving(x in prop::vector(2..=32)) {
        let out = softmax(&x);
        for i in 0..x.len() {
            for j in 0..x.len() {
                if x[i] <= x[j] {
                    prop_assert!(out[i] <= out[j] + 1e-7, "第 {i}/{j} 项逆序");
                }
            }
        }
    }

    /// 置换等变：重排输入就重排输出，逐元素相同。归一化分母不该依赖顺序（仅容差的累加差异）。
    #[test]
    fn softmax_is_equivariant_under_cyclic_shift(x in prop::vector(1..=32), offset in any::<usize>()) {
        let n = x.len();
        let step = offset % n;
        let rotated: Vec<f32> = (0..n).map(|i| x[(i + step) % n]).collect();
        let base = softmax(&x);
        let expected: Vec<f32> = (0..n).map(|i| base[(i + step) % n]).collect();
        prop::assert_within_rel(&softmax(&rotated), &expected, 1e-5, "循环移位后的 softmax");
    }

    /// 全等输入必须给均匀分布：任一项偏离 `1/n` 都是分母算错。
    #[test]
    fn softmax_is_uniform_on_uniform_inputs(value in prop::finite(), len in 1usize..=32) {
        let out = softmax(&vec![value; len]);
        let expected = vec![1.0 / len as f32; len];
        prop::assert_within_rel(&out, &expected, 1e-5, "均匀输入的 softmax");
    }

    /// 稳定性：跨度 60 个数量级的输入下仍须有限、和为 1，且**非严格**保序。
    ///
    /// 这是"减去最大值"是否真的做了的证据——直接 `exp` 的实现会在这里变 `NaN`。
    /// 保序只能到 `<=`：`x[i]` 与 `x[j]` 相差低于 `f32::EPSILON` 时，两者概率被舍入成同一个数，
    /// 此时谁"更大"在 f32 里没有意义（`exp(2.7e-12)` 就是 `1.0`）。
    #[test]
    fn softmax_survives_extreme_magnitudes(x in prop::wide_vector(1..=16)) {
        let out = softmax(&x);
        prop_assert!(prop::all_finite(&out), "极端量级下输出非有限");
        let total: f32 = out.iter().sum();
        prop::assert_within_rel(&[total], &[1.0], 1e-5, "极端量级下的行和");
        for i in 0..x.len() {
            for j in 0..x.len() {
                if x[i] < x[j] {
                    prop_assert!(
                        out[i] <= out[j],
                        "保序被破坏：x[{i}]={} < x[{j}]={} 却 out[{i}]={} > out[{j}]={}",
                        x[i], x[j], out[i], out[j]
                    );
                }
            }
        }
    }

    /// 保范数：rmsnorm 是逐元素除以同一个正数，输出与输入的**方向**必须一致到 `1e-6` 以内。
    #[test]
    fn rmsnorm_preserves_direction(x in prop::vector(1..=32), gamma in prop::vector(1..=32)) {
        let n = x.len().min(gamma.len());
        let (x, gamma) = (&x[..n], &gamma[..n]);
        let out = rmsnorm(x, gamma);
        prop_assert!(prop::all_finite(&out), "输出出现非有限值");
        // 定律：`out = (γ ⊙ x) / rms`，即 out 与 γ ⊙ x 之间只差一个**同一个**正标量。
        // 用"支点比例"而不是 cosine：cosine 要先算范数平方，量级小到次正规时会整体下溢成 0，
        // 那样得到的 `None` 是算术分辨率不足，不是实现违规。
        let scaled: Vec<f32> = gamma.iter().zip(x).map(|(g, v)| g * v).collect();
        let pivot = (0..n)
            .max_by(|&i, &j| scaled[i].abs().total_cmp(&scaled[j].abs()))
            .expect("n >= 1");
        if scaled[pivot] == 0.0 {
            // 内核按 `gamma * x / rms` 求值，乘积先下溢成 0；rms >= sqrt(eps) 为有限正数，故输出必为零。
            prop_assert!(
                out.iter().all(|v| *v == 0.0),
                "γ⊙x 全部下溢时输出必须为零，实得 {:?}",
                out
            );
            return Ok(());
        }
        let base = out[pivot];
        prop_assert_ne!(base, 0.0, "支点分量不该下溢成零");
        for i in 0..n {
            // 低于支点 4 个数量级的分量，其相对信息已被支点自身的舍入淹没，不作判据。
            if scaled[i].abs() < scaled[pivot].abs() * 1e-4 {
                continue;
            }
            prop::assert_within_rel(
                &[out[i] / base],
                &[scaled[i] / scaled[pivot]],
                1e-4,
                "out 与 γ⊙x 必须只差一个公共正标量",
            );
        }
    }

    /// 奇对称**精确**成立：`rmsnorm(−x) = −rmsnorm(x)`。平方和逐位相同，符号翻转逐位精确。
    #[test]
    fn rmsnorm_is_odd(x in prop::vector(1..=32), gamma in prop::vector(1..=32)) {
        let n = x.len().min(gamma.len());
        let (x, gamma) = (&x[..n], &gamma[..n]);
        let negated: Vec<f32> = x.iter().map(|v| -v).collect();
        let expected: Vec<f32> = rmsnorm(x, gamma).iter().map(|v| -v).collect();
        prop::assert_exact("rmsnorm(−x) = −rmsnorm(x)", &rmsnorm(&negated, gamma), &expected);
    }

    /// 对 `gamma` 的**线性**精确成立：`rmsnorm(x, c·γ) = c·rmsnorm(x, γ)`，取 `c` 为 2 的幂
    /// 时不引入舍入，所以这里断言逐位相等。
    #[test]
    fn rmsnorm_scales_exactly_with_gamma(x in prop::vector(1..=32), gamma in prop::vector(1..=32)) {
        let n = x.len().min(gamma.len());
        let (x, gamma) = (&x[..n], &gamma[..n]);
        let halved: Vec<f32> = gamma.iter().map(|g| g * 0.5).collect();
        let got = rmsnorm(x, &halved);
        let expected: Vec<f32> = rmsnorm(x, gamma).iter().map(|v| v * 0.5).collect();
        // 逐位相等的前置条件是**整条缩放链都在正规数区**：`0.5×` 在次正规数上本身就要舍入
        // （`2.07e-38 / 2` 已经低于 `MIN_POSITIVE`，结果 1.46e-38 仍差 1 ulp），
        // 那属于浮点分辨率而非实现缺陷，只能用"误差不超过一个最小正规数"地板来判。
        for (i, (g, w)) in got.iter().zip(&expected).enumerate() {
            if w.is_normal() && (gamma[i] * 0.5).is_normal() {
                prop_assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "正规区内 2 的幂缩放必须逐位相等（第 {} 项：{} 对 {}）",
                    i,
                    g,
                    w
                );
            } else {
                prop_assert!(
                    (g - w).abs() <= f32::MIN_POSITIVE,
                    "次正规区缩放偏差超出分辨率地板（第 {} 项：{} 对 {}）",
                    i,
                    g,
                    w
                );
            }
        }
    }

    /// 均方根为 1 的输入近似原样穿过：`eps` 的影响是 `1 − eps/(2·mean)` 量级，容差 `1e-5` 足够。
    #[test]
    fn rmsnorm_passes_unit_rms_inputs_through(x in prop::vector(2..=32), gamma in prop::vector(2..=32)) {
        let n = x.len().min(gamma.len());
        let (x, gamma) = (unit_mean_square(&x[..n]), &gamma[..n]);
        let expected: Vec<f32> = x.iter().zip(gamma).map(|(v, g)| g * v).collect();
        prop::assert_within_rel(&rmsnorm(&x, gamma), &expected, 1e-5, "单位均方输入");
    }

    /// 零输入给零输出（`0 / sqrt(eps) = 0`）：不能出现 `NaN`。精确成立。
    #[test]
    fn rmsnorm_of_zero_is_zero(gamma in prop::vector(1..=32)) {
        prop::assert_exact("rmsnorm(0)", &rmsnorm(&vec![0.0; gamma.len()], &gamma), &vec![0.0; gamma.len()]);
    }

    /// 旋转是**等距**：`Σ y² = Σ x²`。写错符号、转置、漏掉一个分量都会打破它。
    #[test]
    fn rope_is_an_isometry(x in prop::vector(2..=32), pos in 0usize..=64) {
        let y = rope(pos, &x);
        let before: f32 = x.iter().map(|v| v * v).sum();
        let after: f32 = y.iter().map(|v| v * v).sum();
        prop::assert_within_rel(&[after], &[before], 1e-5, "RoPE 保范数");
    }

    /// 角度可加：`rope(p) ∘ rope(q) = rope(p + q)`。这是"频率表按对下标取、不按元素下标取"的
    /// 直接推论，写错频率会让两次旋转与一次旋转对不上。
    #[test]
    fn rope_angles_add_up(x in prop::vector(2..=32), first in 0usize..=32, second in 0usize..=32) {
        let twice = rope(second, &rope(first, &x));
        prop::assert_within_rel(&twice, &rope(first + second, &x), 1e-5, "rope(p+q)");
    }

    /// `pos = 0` 是恒等映射：`cos 0 = 1`、`sin 0 = 0`，逐位成立。
    #[test]
    fn rope_at_origin_is_the_identity(x in any::<Vec<f32>>()) {
        prop_assert!(x.iter().all(|v| v.is_finite()), "只对有限输入断言恒等");
        prop::assert_exact("rope(0)", &rope(0, &x), &x);
    }

    /// 奇数长度末尾那个无法配对的元素原样透传（契约明写，不能静默丢弃或复制）。
    #[test]
    fn rope_passes_the_odd_tail_through(head in prop::vector(1..=16), tail in prop::finite(), pos in 0usize..=64) {
        let mut x: Vec<f32> = head.iter().take(head.len() / 2 * 2).copied().collect();
        x.push(tail);
        let y = rope(pos, &x);
        prop::assert_exact("奇数尾巴", &y[y.len() - 1..], &[tail]);
    }

    /// 频率表：`freq[i] = BASE^(−2i/d)`，且第二次取用同一份（thread-local 缓存命中）。
    #[test]
    fn rope_frequencies_follow_the_inverse_power_table(d in 2usize..=64) {
        let cached = rope_f32::frequencies(d);
        let expected: Vec<f32> = (0..d / 2)
            .map(|i| (f64::from(rope_f32::BASE).powf(-2.0 * i as f64 / d as f64)) as f32)
            .collect();
        prop::assert_within_rel(&cached, &expected, 1e-6, "频率表");
        prop_assert!(Rc::ptr_eq(&cached, &rope_f32::frequencies(d)), "同一 d 应当命中缓存");
    }

    /// 反量化落点必须是**精确**的仿射网格：`y = SCALE·(code − ZERO)`，`code` 是整数。
    /// 位提取写错（移位数、掩码、槽序）会立刻让 `y / SCALE` 出现小数。
    #[test]
    fn dequant_lands_exactly_on_the_quantization_grid(byte in any::<u8>()) {
        for (cpb, scale, zero) in [(4, dequant_i2::SCALE, dequant_i2::ZERO), (2, dequant_i4::SCALE, dequant_i4::ZERO)] {
            let mut out = vec![0.0; cpb];
            dequant::forward_packed(scale, zero, &[byte], &mut out, cpb).expect("长度与仿射参数合法");
            prop_assert!(prop::all_finite(&out), "反量化出现非有限值");
            for value in &out {
                prop_assert!((value / scale).fract() == 0.0, "{value} 不在 {scale} 的整数网格上");
                let codes = (1u32 << width_cpb(cpb)) as f32;
                prop_assert!(*value >= -zero * scale && *value <= (codes - 1.0 - zero) * scale, "{value} 越出码域");
            }
        }
    }

    /// 全部 256 个字节必须覆盖**整个**码空间：每个码点都该在某处出现（掩码少一位就覆盖不到）。
    #[test]
    fn dequant_byte_space_covers_every_code(scale in prop::positive(), zero in 0.0f32..=8.0) {
        let cpb = 2;
        let width = width_cpb(cpb);
        let codes = 1u32 << width;
        if zero >= codes as f32 {
            return Ok(());
        }
        let bytes: Vec<u8> = (0..=255u8).collect();
        let mut out = vec![0.0; bytes.len() * cpb];
        dequant::forward_packed(scale, zero, &bytes, &mut out, cpb).expect("长度与仿射参数合法");
        let lowest = -zero * scale;
        let highest = (codes as f32 - 1.0 - zero) * scale;
        let min = out.iter().copied().fold(f32::INFINITY, f32::min);
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        prop::assert_within_rel(&[min], &[lowest], 1e-6, "最小码点未取到");
        prop::assert_within_rel(&[max], &[highest], 1e-6, "最大码点未取到");
    }

    /// 缓冲长度不是 `q.len()·cpb` 时必须报错，而不是越界或静默截断。
    #[test]
    fn dequant_rejects_mismatched_lengths(bytes in proptest::collection::vec(any::<u8>(), 1..=8), extra in 0usize..=8) {
        let cpb = 2;
        let mut out = vec![0.0; bytes.len() * cpb + extra + 1];
        prop_assert!(dequant::forward_packed(dequant_i4::SCALE, dequant_i4::ZERO, &bytes, &mut out, cpb).is_err());
    }

    /// 仿射参数的定义域：零标度/非有限零点/零点越出码域一律拒绝，域内一律接受。
    #[test]
    fn dequant_affine_domain_is_enforced(width in 1usize..=8, zero in -4.0f32..=300.0) {
        let codes = (1u32 << width) as f32;
        let accepted = dequant::check_affine(0.25, zero, width).is_ok();
        prop_assert_eq!(accepted, (0.0..codes).contains(&zero), "零点 {} 的判定与码域 0..{} 不符", zero, codes);
        prop_assert!(dequant::check_affine(0.0, zero, width).is_err(), "零标度必须拒绝");
        prop_assert!(dequant::check_affine(0.25, f32::NAN, width).is_err(), "NaN 零点必须拒绝");
        prop_assert!(dequant::check_affine(0.25, f32::INFINITY, width).is_err(), "inf 零点必须拒绝");
    }

    /// 向量化 GEMM 与标量参考的**语义**相同：差异只来自累加顺序，所以按 IEEE 容差对拍。
    #[test]
    fn fast_gemm_matches_the_scalar_reference(
        m in 1usize..=6, n in 1usize..=6, k in 1usize..=6,
        a in matrix(6, 6), b in matrix(6, 6),
    ) {
        let a: Vec<f32> = a[..m * k].to_vec();
        let b: Vec<f32> = b[..k * n].to_vec();
        let mut want = vec![0.0; m * n];
        matmul_scalar::<f32>(&a, &b, m, n, k, &mut want);
        let mut got = vec![0.0; m * n];
        fast::gemm_f32(&a, &b, m, n, k, &mut got);
        prop_assert!(prop::all_finite(&got), "fast 路径出现非有限值");
        prop::assert_within_rel(&got, &want, 1e-5, "fast::gemm_f32 对 matmul_scalar");
    }

    /// 单线程与多线程两条路径必须给同一个答案：并行分块不该改变数值口径。
    #[test]
    fn fast_gemm_paths_agree(m in 1usize..=6, n in 1usize..=6, k in 1usize..=8, a in matrix(6, 8), b in matrix(8, 6)) {
        let a: Vec<f32> = a[..m * k].to_vec();
        let b: Vec<f32> = b[..k * n].to_vec();
        let mut single = vec![0.0; m * n];
        fast::gemm_f32_single(&a, &b, m, n, k, &mut single);
        let mut multi = vec![0.0; m * n];
        fast::gemm_f32(&a, &b, m, n, k, &mut multi);
        prop::assert_within_rel(&multi, &single, 1e-6, "多线程对单线程");
    }

    /// `gemm_f32_tn` 用的是线性层布局：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
    /// 把 `w` 转置成 `k×n` 后必须与标量参考一致——这是布局约定唯一的机检手段。
    #[test]
    fn fast_gemm_tn_matches_the_transposed_scalar_product(
        rows in 1usize..=8, n in 1usize..=6, k in 1usize..=6,
        x in matrix(8, 6), w in matrix(6, 6),
    ) {
        let x: Vec<f32> = x[..rows * k].to_vec();
        let w: Vec<f32> = w[..n * k].to_vec();
        let transposed: Vec<f32> = (0..k * n).map(|idx| w[(idx % n) * k + idx / n]).collect();
        let mut want = vec![0.0; rows * n];
        matmul_scalar::<f32>(&x, &transposed, rows, n, k, &mut want);
        let mut got = vec![0.0; rows * n];
        fast::gemm_f32_tn(
            tensor::View::matrix(rows, k, &x).expect("x 二维"),
            tensor::View::matrix(n, k, &w).expect("w 二维"),
            &mut got,
        );
        prop::assert_within_rel(&got, &want, 1e-5, "gemm_f32_tn 对转置后的标量积");
    }

    /// i8 累加是**整数**运算：不同实现之间必须逐位相等，容差在这里毫无意义。
    #[test]
    fn fast_i8_gemm_is_exact(m in 1usize..=4, n in 1usize..=4, k in 1usize..=16,
        a in proptest::collection::vec(any::<i8>(), 4 * 16), b in proptest::collection::vec(any::<i8>(), 16 * 4)) {
        let a: Vec<i8> = a[..m * k].to_vec();
        let b: Vec<i8> = b[..k * n].to_vec();
        let mut want = vec![0i32; m * n];
        matmul_scalar::<i8>(&a, &b, m, n, k, &mut want);
        let mut got = vec![0i32; m * n];
        fast::gemm_i8(&a, &b, m, n, k, &mut got);
        prop_assert_eq!(got, want, "i8 GEMM 与标量参考必须逐位相等");
    }

    // ===== GEMV 定律 =====
    //
    // GEMV 是 `x ↦ W·x` 的线性映射。小整数输入 + 2 的幂标度下，每条乘积与部分和
    // 都是 f32 精确整数，于是"线性""尺度不变"这两条代数定律可以断言**逐位相等**，
    // 不必退到容差。

    /// 加性：`gemv(x₁ + x₂) = gemv(x₁) + gemv(x₂)`。
    #[test]
    fn gemv_f32_is_additive_in_x(
        m in 1usize..=96, n in 1usize..=8,
        w in ints(96 * 8), a in ints(96), b in ints(96),
    ) {
        let (w, a, b) = (&w[..m * n], &a[..m], &b[..m]);
        let summed: Vec<f32> = a.iter().zip(b).map(|(p, q)| p + q).collect();
        let (left, right) = (gemv_f32_of(a, w, n), gemv_f32_of(b, w, n));
        let want: Vec<f32> = left.iter().zip(&right).map(|(p, q)| p + q).collect();
        prop::assert_exact("gemv(x₁+x₂) = gemv(x₁)+gemv(x₂)", &gemv_f32_of(&summed, w, n), &want);
    }

    /// 齐次性（2 的幂）：`gemv(2x) = 2·gemv(x)`，逐位成立。
    #[test]
    fn gemv_f32_scales_exactly_with_powers_of_two(
        m in 1usize..=96, n in 1usize..=8, w in ints(96 * 8), x in ints(96),
    ) {
        let (w, x) = (&w[..m * n], &x[..m]);
        let doubled: Vec<f32> = x.iter().map(|v| v * 2.0).collect();
        let want: Vec<f32> = gemv_f32_of(x, w, n).iter().map(|v| v * 2.0).collect();
        prop::assert_exact("gemv(2x) = 2·gemv(x)", &gemv_f32_of(&doubled, w, n), &want);
    }

    /// 内核与"逐行顺序点积"的定义一致：任意有限输入下只差累加顺序的舍入。
    #[test]
    fn gemv_f32_matches_the_row_wise_dot_product(
        m in 1usize..=96, n in 1usize..=8, w in matrix(96, 8), x in matrix(96, 1),
    ) {
        let (w, x) = (&w[..m * n], &x[..m]);
        let got = gemv_f32_of(x, w, n);
        prop_assert!(prop::all_finite(&got), "gemv_f32 出现非有限值");
        prop::assert_within_rel(&got, &dense_reference(x, w, n), 1e-4, "gemv_f32 对逐行顺序点积");
    }

    /// 稠密点积的**误差上界**：`simd::elementwise` 把求和拆进多条累加链，顺序与标量不同，
    /// 逐位相等不可能；于是对 f64 真值钉 `γ_m = m·u/(1−m·u)`。
    ///
    /// `f32×f32` 落进 `f64` 是精确的（24+24 ≤ 53 位），96 项 `f64` 求和的相对误差
    /// ≈ 96·2⁻⁵³ ≈ 1e-14，比 `f32` 的界小 7 个数量级 —— `exact as f32` 就是可用的真值。
    #[test]
    fn gemv_f32_dot_stays_inside_the_gamma_bound(
        m in 1usize..=96, n in 1usize..=5, w in scaled(96 * 5), x in scaled(96),
    ) {
        let (w, x) = (&w[..m * n], &x[..m]);
        let got = gemv_f32_of(x, w, n);
        let u = f64::from(f32::EPSILON) / 2.0;
        let gamma = m as f64 * u / (1.0 - m as f64 * u);
        for j in 0..n {
            let products = (0..m).map(|k| f64::from(w[j * m + k]) * f64::from(x[k]));
            let (exact, magnitude) =
                products.fold((0.0f64, 0.0f64), |(sum, mag), p| (sum + p, mag + p.abs()));
            let error = (f64::from(got[j]) - exact).abs();
            prop_assert!(
                error <= gamma * magnitude,
                "第 {j} 行：{error:e} 超出 γ_{m}·Σ|wx| = {:e}",
                gamma * magnitude
            );
        }
    }

    /// 行批与逐行调用给同一个答案：并行遍历不该改变数值口径。
    #[test]
    fn gemv_f32_batch_equals_row_by_row_calls(
        m in 1usize..=96, n in 1usize..=6, rows in 1usize..=3,
        w in ints(96 * 6), x in ints(96 * 3),
    ) {
        let (w, head) = (&w[..m * n], &x[..rows * m]);
        let mut batched = vec![0.0f32; rows * n];
        crate::gemv::forward_plain_batch::<crate::gemv::Dense<32>>(
            head,
            w,
            &[],
            crate::gemv::Geometry::new(rows, m, n),
            &mut batched,
        )
        .expect("行批形状合法");
        let mut stepwise = Vec::with_capacity(rows * n);
        for r in 0..rows {
            stepwise.extend(gemv_f32_of(&head[r * m..][..m], w, n));
        }
        prop::assert_exact("行批 = 逐行", &batched, &stepwise);
    }

    /// i8 输出必须等于定义：分组解包、整组乘标度、再跨组相加。
    /// 1 字节/元素、整数权重 × 整数输入 × 2 的幂标度 → 全程精确，逐位相等。
    #[test]
    fn gemv_i8_matches_the_dequantized_dot(
        n in 1usize..=5, groups in 1usize..=4, x in ints(128),
    ) {
        let m = crate::gemv::GROUP * groups;
        let x = &x[..m];
        let w = crate::gemv::packed::weights(m, n, 1);
        let s: Vec<f32> = (0..n * groups).map(|i| [0.25f32, 0.5, 1.0, 2.0][i % 4]).collect();
        let mut got = vec![0.0f32; n];
        gemv_i8::forward(tensor::View::one(x), &w, tensor::View::one(&s), &mut got).expect("形状合法");
        prop::assert_exact("gemv_i8 对定义", &got, &i8_reference(x, &w, &s, n, groups));
    }

    /// i8 同样是线性映射：分组标度不破坏加性。
    #[test]
    fn gemv_i8_is_additive_in_x(
        n in 1usize..=4, groups in 1usize..=4, a in ints(128), b in ints(128),
    ) {
        let m = crate::gemv::GROUP * groups;
        let (a, b) = (&a[..m], &b[..m]);
        let summed: Vec<f32> = a.iter().zip(b).map(|(p, q)| p + q).collect();
        let w = crate::gemv::packed::weights(m, n, 1);
        let s: Vec<f32> = vec![1.0; n * groups];
        let (mut left, mut right, mut both) =
            (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        gemv_i8::forward(tensor::View::one(a), &w, tensor::View::one(&s), &mut left).expect("形状合法");
        gemv_i8::forward(tensor::View::one(b), &w, tensor::View::one(&s), &mut right).expect("形状合法");
        gemv_i8::forward(tensor::View::one(&summed), &w, tensor::View::one(&s), &mut both).expect("形状合法");
        let want: Vec<f32> = left.iter().zip(&right).map(|(p, q)| p + q).collect();
        prop::assert_exact("gemv_i8(x₁+x₂)", &both, &want);
    }

    /// i4 打包：与"独立解包 + 查码表"的定义逐位相等，覆盖 SIMD（n≥4）与标量（n<4）两条路径。
    #[test]
    fn gemv_i4_matches_the_packed_definition(
        n in 1usize..=5, groups in 1usize..=4, x in ints(128),
    ) {
        let m = crate::gemv::GROUP * groups;
        let x = &x[..m];
        let w = crate::gemv::packed::weights(m, n, 2);
        let s: Vec<f32> = (0..n * groups).map(|i| [0.5f32, 1.0, 2.0, 0.25][i % 4]).collect();
        let mut got = vec![0.0f32; n];
        gemv_i4::forward(tensor::View::one(x), &w, tensor::View::one(&s), &mut got).expect("形状合法");
        let want = packed_reference(x, &w, &s, n, groups, 2, I4.table);
        prop::assert_exact("gemv_i4 对打包定义", &got, &want);
    }

    /// i2 打包：同上，位宽 2、每组 4 个码。
    #[test]
    fn gemv_i2_matches_the_packed_definition(
        n in 1usize..=5, groups in 1usize..=4, x in ints(128),
    ) {
        let m = crate::gemv::GROUP * groups;
        let x = &x[..m];
        let w = crate::gemv::packed::weights(m, n, 4);
        let s: Vec<f32> = (0..n * groups).map(|i| [1.0f32, 0.5][i % 2]).collect();
        let mut got = vec![0.0f32; n];
        gemv_i2::forward(tensor::View::one(x), &w, tensor::View::one(&s), &mut got).expect("形状合法");
        let want = packed_reference(x, &w, &s, n, groups, 4, I2.table);
        prop::assert_exact("gemv_i2 对打包定义", &got, &want);
    }

    // ===== GEMV 守卫：拒绝的边界 =====
    //
    // 上面几条钉的是"算得对"，这一组钉的是"什么时候不许算"：守卫要把形状错误挡在内核之外，
    // 而不是让 `&weights[idx * row..]` 在上越界处 panic。每条都同时断言该拒的拒、该收的收——
    // 只断言拒绝一侧，整体收紧成"一律拒绝"也能过。

    /// 输出维度是三条格式共同的前置约束：`n = 0` 没有可写的元素，必须报错而不是静默成功。
    #[test]
    fn gemv_rejects_an_empty_output(m in 1usize..=32, x in ints(32)) {
        let x = &x[..m];
        prop_assert!(gemv_f32::forward(tensor::View::one(x), &[], &mut []).is_err(), "f32 的 n = 0 必须拒绝");
        prop_assert!(
            gemv_i8::forward(tensor::View::one(x), &[], tensor::View::one(&[]), &mut []).is_err(),
            "i8 的 n = 0 必须拒绝"
        );
    }

    /// 权重下界是 `n·m`：少一个就拒，恰好够就收。
    #[test]
    fn gemv_f32_needs_exactly_n_times_m_weights(
        m in 1usize..=48, n in 1usize..=8, w in ints(48 * 8),
    ) {
        let x = &w[..m];
        let (need, mut out) = (n * m, vec![0.0f32; n]);
        prop_assert!(
            gemv_f32::forward(tensor::View::one(x), &w[..need - 1], &mut out).is_err(),
            "权重 {} < {need} 必须拒绝",
            need - 1
        );
        prop_assert!(
            gemv_f32::forward(tensor::View::one(x), &w[..need], &mut out).is_ok(),
            "权重恰好 {need} 必须接受"
        );
    }

    /// 稠密格式没有标度轴：空切片才合法，收到任何一条标度都是调用方的形状错误。
    #[test]
    fn gemv_f32_dense_has_no_scale_axis(
        m in 1usize..=32, n in 1usize..=8, w in matrix(8, 32), s in prop::finite(),
    ) {
        let x: Vec<f32> = (0..m).map(|k| k as f32 * 0.25).collect();
        let mut out = vec![0.0f32; n];
        prop_assert!(
            <crate::gemv::f32::F32 as crate::gemv::Format>::forward(tensor::View::one(&x), &w[..n * m], tensor::View::one(&[]), &mut out).is_ok(),
            "空标度必须接受"
        );
        prop_assert!(
            <crate::gemv::f32::F32 as crate::gemv::Format>::forward(tensor::View::one(&x), &w[..n * m], tensor::View::one(&[s]), &mut out).is_err(),
            "稠密格式收到标度必须拒绝"
        );
    }

    /// 标度长度要同时说清两件事：每行整数个组，且组数能整除规约轴。两条缺一不可。
    #[test]
    fn gemv_i8_needs_scales_that_divide_both_axes(
        n in 1usize..=4, groups in 1usize..=4, len in 0usize..=17,
    ) {
        let m = crate::gemv::GROUP * groups;
        let w = crate::gemv::packed::weights(m, n, 1);
        let (x, s, mut out) = (vec![1.0f32; m], vec![1.0f32; len], vec![0.0f32; n]);
        let per_row = len / n;
        let ok = len.is_multiple_of(n) && per_row > 0 && m.is_multiple_of(per_row);
        prop_assert_eq!(
            gemv_i8::forward(tensor::View::one(&x), &w, tensor::View::one(&s), &mut out).is_ok(),
            ok,
            "{} 个标度配 n = {}、m = {} 的可接受性",
            len, n, m
        );
    }

    /// 非有限标度会把整组乘积带进 `NaN`/`inf`，且与格式无关：必须在累加前拒掉。
    #[test]
    fn gemv_i8_rejects_non_finite_scales(
        n in 1usize..=4, groups in 1usize..=4, slot in 0usize..=16, which in 0usize..3,
    ) {
        let m = crate::gemv::GROUP * groups;
        let w = crate::gemv::packed::weights(m, n, 1);
        let (x, mut s, mut out) = (vec![1.0f32; m], vec![1.0f32; n * groups], vec![0.0f32; n]);
        prop_assert!(gemv_i8::forward(tensor::View::one(&x), &w, tensor::View::one(&s), &mut out).is_ok(), "全有限必须接受");
        let at = slot % s.len();
        s[at] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][which];
        prop_assert!(
            gemv_i8::forward(tensor::View::one(&x), &w, tensor::View::one(&s), &mut out).is_err(),
            "第 {at} 个标度非有限必须拒绝"
        );
    }

    /// 行批的几何是三个数一起说的：`x` 必须正好 `rows·m`、`out` 必须正好 `rows·n`。
    #[test]
    fn gemv_i8_batch_rejects_inconsistent_geometry(
        rows in 1usize..=4, n in 1usize..=4, mode in 0usize..4,
    ) {
        let m = crate::gemv::GROUP;
        let w = crate::gemv::packed::weights(m, n, 1);
        let (s, x) = (vec![1.0f32; n], vec![1.0f32; rows * m]);
        let mut good = vec![0.0f32; rows * n];
        let xv = tensor::View::matrix(rows, m, &x).expect("几何一致");
        prop_assert!(
            gemv_i8::forward_batch(xv, &w, tensor::View::one(&s), &mut good).is_ok(),
            "几何一致必须接受"
        );
        let rejected = match mode {
            0 => (|| {
                let xv = tensor::View::matrix(0, m, &x[..0])?;
                gemv_i8::forward_batch(xv, &w, tensor::View::one(&s), &mut vec![0.0f32; rows * n])
            })()
            .is_err(),
            1 => (|| {
                let xv = tensor::View::matrix(rows, m, &x[..rows * m - 1])?;
                gemv_i8::forward_batch(xv, &w, tensor::View::one(&s), &mut vec![0.0f32; rows * n])
            })()
            .is_err(),
            _ => (|| {
                let xv = tensor::View::matrix(rows, m, &x)?;
                gemv_i8::forward_batch(
                    xv,
                    &w,
                    tensor::View::one(&s),
                    &mut vec![0.0f32; rows * n + 1],
                )
            })()
            .is_err(),
        };
        prop_assert!(rejected, "mode {mode} 的几何必须拒绝");
    }
}

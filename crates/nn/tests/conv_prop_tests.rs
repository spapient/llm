// conv 数值内核的**定律**测试：每条性质都从算子的数学定义推出，不复述实现。
//
// 挑选原则与 `gemm/tests/prop_numeric.rs` 一致——按"能有多强就用多强"分档：
//
// - **精确相等**：`w = 0` 时输出就是 `bias`、`conv` 的奇对称、最大池化的"取值必被取到"。
// 这些地方用容差只会掩盖错误。
// - **相对误差**：仿射性、线性、两个独立实现的差分（`flash` 对 `naive`）。
// - **结构性**：注意力的输出必须落在 `v` 的凸包内、平均/插值输出必须落在窗口界内。
// 这类性质不依赖任何实现细节，只依赖算子的定义，且能抓住"权重没归一化"这类错。

use crate::conv::attention;
use crate::conv::classic::{
    adaptive_pool, avg_pool2d, conv2d, interpolate, max_pool1d, max_pool2d,
};
use proptest::prelude::*;
use testkit::prop;

/// 定长浮点向量：几何是声明档常量，不能由生成器决定长度。
fn fixed(len: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(prop::finite(), len)
}

/// `k = 3, s = 1, p = 1` 的 same 输出共用窗口：返回该输出位置**有效**（非填充）元素的下标。
fn same_window(idx: usize, h: usize, w: usize) -> Vec<usize> {
    let (oh, ow) = ((idx / w) as i32, (idx % w) as i32);
    let mut idx_set = Vec::with_capacity(9);
    for dh in -1..=1 {
        for dw in -1..=1 {
            let (ih, iw) = (oh + dh, ow + dw);
            if (0..h as i32).contains(&ih) && (0..w as i32).contains(&iw) {
                idx_set.push(ih as usize * w + iw as usize);
            }
        }
    }
    idx_set
}

/// 自适应池化 `4 → 2` 的分块：输出格 `(oh, ow)` 覆盖 `2×2` 输入块。
fn adaptive_window(idx: usize, oh: usize, ow: usize, h: usize, w: usize) -> Vec<usize> {
    let (bh, bw) = (h / oh, w / ow);
    let (row, col) = (idx / ow, idx % ow);
    let mut idx_set = Vec::with_capacity(bh * bw);
    for dh in 0..bh {
        for dw in 0..bw {
            idx_set.push((row * bh + dh) * w + col * bw + dw);
        }
    }
    idx_set
}

/// 把下标集合映射成值的上下界（`min`/`max` 对有限输入是全域定义的）。
fn bounds(values: &[f32], idx_set: &[usize]) -> (f32, f32) {
    idx_set
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), &i| {
            (low.min(values[i]), high.max(values[i]))
        })
}

fn conv2d_run(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; conv2d::OH * conv2d::OW * conv2d::COUT];
    conv2d::forward(x, weight, bias, &mut out).expect("定长缓冲");
    out
}

/// 一维窗口族与插值的统一调用口径：输入定长、输出定长、无参数。
fn window_run(
    forward: fn(&[f32], &mut [f32]) -> diagnostic::Result<()>,
    x: &[f32],
    len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    forward(x, &mut out).expect("定长缓冲");
    out
}

fn attention_run(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; s * d];
    attention::forward(q, k, v, &mut out, s, d).expect("等长");
    out
}

fn flash_run(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; s * d];
    attention::flash_forward(q, k, v, &mut out, s, d).expect("等长");
    out
}

proptest! {
    #![proptest_config(prop::config())]

    // ---------------------------------------------------------------- 注意力

    /// 注意力的定义：输出第 `i` 行是 `v` 各行按行和为 1 的非负权重的凸组合。
    /// 于是每个分量必须落在 `v` 该列的最小值与最大值之间——这条律不依赖 softmax、
    /// 不依赖 `1/√d`、不依赖是否分块，只依赖"凸组合"这一个定义。
    #[test]
    fn attention_output_lies_in_the_convex_hull_of_values(
        q in fixed(4 * 3), k in fixed(4 * 3), v in fixed(4 * 3)
    ) {
        let (s, d) = (4, 3);
        let out = attention_run(&q, &k, &v, s, d);
        prop_assert!(prop::all_finite(&out), "输出出现非有限值");
        for row in 0..s {
            for col in 0..d {
                let column: Vec<f32> = (0..s).map(|j| v[j * d + col]).collect();
                let (low, high) = bounds(&column, &(0..s).collect::<Vec<_>>());
                let got = out[row * d + col];
                prop_assert!(
                    got >= low - 1e-6 && got <= high + 1e-6,
                    "第 {row} 行第 {col} 列越出凸包：[{low}, {high}] 之外得 {got}"
                );
            }
        }
    }

    /// 对 `v` 的线性：注意力权重只由 `q`、`k` 决定，所以 `v` 方向是线性的。
    #[test]
    fn attention_is_linear_in_values(
        q in fixed(3 * 2), k in fixed(3 * 2), v1 in fixed(3 * 2), v2 in fixed(3 * 2),
        alpha in prop::finite(), beta in prop::finite()
    ) {
        let (s, d) = (3, 2);
        let mixed: Vec<f32> = v1.iter().zip(&v2).map(|(a, b)| alpha * a + beta * b).collect();
        let got = attention_run(&q, &k, &mixed, s, d);
        let left = attention_run(&q, &k, &v1, s, d);
        let right = attention_run(&q, &k, &v2, s, d);
        let want: Vec<f32> = left.iter().zip(&right).map(|(a, b)| alpha * a + beta * b).collect();
        prop::assert_within_rel(&got, &want, 1e-5, "注意力对 v 的线性");
    }

    /// softmax 的参数不变性：给**所有** `k` 行加同一个向量，等价于给每行 logit 加同一个常数
    /// `q_i · c`，行内相对大小不变，因此注意力输出不变。这条律盯的是"是否真的按行做 softmax"。
    #[test]
    fn attention_ignores_a_common_shift_of_keys(
        q in fixed(3 * 2), k in fixed(3 * 2), v in fixed(3 * 2), shift in fixed(2)
    ) {
        let (s, d) = (3, 2);
        let shifted: Vec<f32> = k
            .iter()
            .enumerate()
            .map(|(i, value)| value + shift[i % d])
            .collect();
        prop::assert_within_rel(
            &attention_run(&q, &shifted, &v, s, d),
            &attention_run(&q, &k, &v, s, d),
            1e-5,
            "所有 k 行同加一个向量不改变注意力",
        );
    }

    /// `v` 各行相同时，凸组合只能是那个值本身：权重是否归一化在这里无处可藏。
    #[test]
    fn attention_on_constant_values_reproduces_them(
        q in fixed(3 * 2), k in fixed(3 * 2), value in fixed(2)
    ) {
        let (s, d) = (3, 2);
        let v: Vec<f32> = (0..s).flat_map(|_| value.clone()).collect();
        let out = attention_run(&q, &k, &v, s, d);
        let want: Vec<f32> = (0..s).flat_map(|_| value.clone()).collect();
        prop::assert_within_rel(&out, &want, 1e-5, "常量 v 必须原样穿过注意力");
    }

    /// 差分律：`flash`（online softmax）与 `naive`（先求全行再归一化）是两条独立路径，
    /// 数学上同值。分块与不分块、单遍与两遍的差别只能落在 `f32` 舍入上。
    #[test]
    fn flash_and_naive_attention_agree(
        q in fixed(5 * 3), k in fixed(5 * 3), v in fixed(5 * 3)
    ) {
        let (s, d) = (5, 3);
        prop::assert_within_rel(
            &flash_run(&q, &k, &v, s, d),
            &attention_run(&q, &k, &v, s, d),
            1e-5,
            "flash 与 naive 注意力",
        );
    }

    // ---------------------------------------------------------------- 二维卷积

    /// `w = 0` 时卷积核没有任何贡献，输出必须**恰好**是 `bias` 的广播。
    /// 用精确相等：零累加后再加 `bias` 不引入任何舍入。
    #[test]
    fn conv_with_zero_weights_is_the_bias(x in fixed(4 * 4 * 2), bias in fixed(3)) {
        let weight = vec![0.0f32; 3 * 3 * 2 * 3];
        let out = conv2d_run(&x, &weight, &bias);
        let want: Vec<f32> = (0..conv2d::OH * conv2d::OW)
            .flat_map(|_| bias.clone())
            .collect();
        prop::assert_exact("w = 0 的卷积", &out, &want);
    }

    /// 输入全零时同理：输出必须**恰好**是 `bias`。
    #[test]
    fn conv_of_zero_input_is_the_bias(weight in fixed(3 * 3 * 2 * 3), bias in fixed(3)) {
        let x = vec![0.0f32; 4 * 4 * 2];
        let out = conv2d_run(&x, &weight, &bias);
        let want: Vec<f32> = (0..conv2d::OH * conv2d::OW)
            .flat_map(|_| bias.clone())
            .collect();
        prop::assert_exact("x = 0 的卷积", &out, &want);
    }

    /// 对输入的仿射性：`conv(x₁ + x₂) = conv(x₁) + conv(x₂) − bias`。
    /// 卷积是线性算子，"减掉多加的那一份 bias"是它区别于纯线性的全部内容。
    #[test]
    fn conv_is_affine_in_its_input(
        x1 in fixed(4 * 4 * 2), x2 in fixed(4 * 4 * 2),
        weight in fixed(3 * 3 * 2 * 3), bias in fixed(3)
    ) {
        let sum: Vec<f32> = x1.iter().zip(&x2).map(|(a, b)| a + b).collect();
        let got = conv2d_run(&sum, &weight, &bias);
        let want: Vec<f32> = conv2d_run(&x1, &weight, &bias)
            .iter()
            .zip(conv2d_run(&x2, &weight, &bias))
            .zip((0..conv2d::OH * conv2d::OW).flat_map(|_| bias.clone()))
            .map(|((a, b), c)| a + b - c)
            .collect();
        prop::assert_within_rel(&got, &want, 1e-5, "卷积对输入的仿射性");
    }

    /// 奇对称**逐位**成立：`conv(−x, w, −b) = −conv(x, w, b)`。
    /// 每个乘积取反是精确的、累加顺序不变，所以左右两侧的舍入完全一致；这里用精确断言。
    #[test]
    fn conv_is_odd_when_input_and_bias_both_flip(
        x in fixed(4 * 4 * 2), weight in fixed(3 * 3 * 2 * 3), bias in fixed(3)
    ) {
        let negated: Vec<f32> = x.iter().map(|v| -v).collect();
        let flipped_bias: Vec<f32> = bias.iter().map(|v| -v).collect();
        let want: Vec<f32> = conv2d_run(&x, &weight, &bias).iter().map(|v| -v).collect();
        prop::assert_exact("conv(−x, w, −b) = −conv(x, w, b)", &conv2d_run(&negated, &weight, &flipped_bias), &want);
    }

    // ---------------------------------------------------------------- 池化

    /// 最大池化 `3×3, s=1, p=1` 含中心点，所以输出每个位置必**不小于**输入同位置。
    /// 这条律直接盯住"填充值取 0"这类错误：全负输入下它会立刻破防。
    #[test]
    fn max_pool2d_never_goes_below_its_input(x in fixed(4 * 4)) {
        let out = window_run(max_pool2d::forward, &x, max_pool2d::H * max_pool2d::W);
        for (i, (got, source)) in out.iter().zip(&x).enumerate() {
            prop_assert!(got >= source, "第 {i} 个位置：池化结果 {got} 小于输入 {source}");
        }
    }

    /// 最大值必被**取到**：每个输出都恰好等于其有效窗口里的某个输入元素（逐位相等）。
    /// 这比"不小于窗口内所有元素"更强——它同时排除了"多加了什么"。
    #[test]
    fn max_pool2d_output_is_attained_in_its_window(x in fixed(4 * 4)) {
        let (h, w) = (max_pool2d::H, max_pool2d::W);
        let out = window_run(max_pool2d::forward, &x, h * w);
        for (i, got) in out.iter().enumerate() {
            let window = same_window(i, h, w);
            prop_assert!(
                window.iter().any(|&j| x[j].to_bits() == got.to_bits()),
                "第 {i} 个输出 {got} 不是其窗口中任何元素的原值"
            );
        }
    }

    /// 常量场上最大池化原样返回（逐位相等）：窗口内所有元素相同，最大值就是它。
    #[test]
    fn max_pool2d_of_a_constant_field_is_that_constant(value in prop::finite()) {
        let x = vec![value; max_pool2d::H * max_pool2d::W];
        let out = window_run(max_pool2d::forward, &x, x.len());
        prop::assert_exact("常量场的最大池化", &out, &x);
    }

    /// 一维最大池化 `k=2, s=2` 同样"取值必被取到"：每个输出必等于其两个输入之一。
    #[test]
    fn max_pool1d_output_is_one_of_its_pair(x in fixed(16)) {
        let out = window_run(max_pool1d::forward, &x, max_pool1d::OL);
        for (i, got) in out.iter().enumerate() {
            prop_assert!(
                x[2 * i].to_bits() == got.to_bits() || x[2 * i + 1].to_bits() == got.to_bits(),
                "第 {i} 个输出 {got} 既不是 x[{}] 也不是 x[{}]",
                2 * i,
                2 * i + 1
            );
        }
    }

    /// 平均池化 `3×3, s=1, p=1`、`count_include_pad = false`：常量场必须原样穿过。
    /// 若填充被计入分母，角上的输出会变成 `4c/9`——这条律对那种错误是致命的。
    #[test]
    fn avg_pool2d_of_a_constant_field_is_that_constant(value in prop::finite()) {
        let x = vec![value; avg_pool2d::H * avg_pool2d::W];
        let out = window_run(avg_pool2d::forward, &x, x.len());
        let want = vec![value; x.len()];
        prop::assert_within_rel(&out, &want, 1e-6, "常量场的平均池化");
    }

    /// 平均池化输出必须落在其有效窗口的上下界之内（平均值不会越界）。
    #[test]
    fn avg_pool2d_stays_within_its_window_bounds(x in fixed(4 * 4)) {
        let (h, w) = (avg_pool2d::H, avg_pool2d::W);
        let out = window_run(avg_pool2d::forward, &x, h * w);
        for (i, got) in out.iter().enumerate() {
            let (low, high) = bounds(&x, &same_window(i, h, w));
            prop_assert!(
                *got >= low - 1e-6 && *got <= high + 1e-6,
                "第 {i} 个输出 {got} 越出窗口界 [{low}, {high}]"
            );
        }
    }

    /// 自适应池化 `4 → 2`：每个输出是其 `2×2` 分块的均值，常量场必须原样穿过。
    #[test]
    fn adaptive_pool_of_a_constant_field_is_that_constant(value in prop::finite()) {
        let x = vec![value; adaptive_pool::H * adaptive_pool::W];
        let out = window_run(adaptive_pool::forward, &x, adaptive_pool::OH * adaptive_pool::OW);
        let want = vec![value; out.len()];
        prop::assert_within_rel(&out, &want, 1e-6, "常量场的自适应池化");
    }

    /// 自适应池化输出必须落在其分块的上下界之内。
    #[test]
    fn adaptive_pool_stays_within_its_block_bounds(x in fixed(4 * 4)) {
        let (h, w) = (adaptive_pool::H, adaptive_pool::W);
        let (oh, ow) = (adaptive_pool::OH, adaptive_pool::OW);
        let out = window_run(adaptive_pool::forward, &x, oh * ow);
        for (i, got) in out.iter().enumerate() {
            let (low, high) = bounds(&x, &adaptive_window(i, oh, ow, h, w));
            prop_assert!(
                *got >= low - 1e-6 && *got <= high + 1e-6,
                "第 {i} 个输出 {got} 越出分块界 [{low}, {high}]"
            );
        }
    }

    // ---------------------------------------------------------------- 双线性插值

    /// 双线性插值的权重是单位分解（该点两侧权重和为 1），所以常量场必须原样穿过。
    /// 这条律盯的是"权重和是否为 1"——`align_corners`、边界钳制都会在这里露出来。
    #[test]
    fn interpolation_of_a_constant_field_is_that_constant(value in prop::finite()) {
        let x = vec![value; interpolate::H * interpolate::W];
        let out = window_run(interpolate::forward, &x, interpolate::OH * interpolate::OW);
        let want = vec![value; out.len()];
        prop::assert_within_rel(&out, &want, 1e-6, "常量场的双线性插值");
    }

    /// 双线性插值是线性算子：`interp(α·x₁ + β·x₂) = α·interp(x₁) + β·interp(x₂)`。
    #[test]
    fn interpolation_is_linear(
        x1 in fixed(4), x2 in fixed(4), alpha in prop::finite(), beta in prop::finite()
    ) {
        let len = interpolate::OH * interpolate::OW;
        let mixed: Vec<f32> = x1.iter().zip(&x2).map(|(a, b)| alpha * a + beta * b).collect();
        let got = window_run(interpolate::forward, &mixed, len);
        let left = window_run(interpolate::forward, &x1, len);
        let right = window_run(interpolate::forward, &x2, len);
        let want: Vec<f32> = left.iter().zip(&right).map(|(a, b)| alpha * a + beta * b).collect();
        prop::assert_within_rel(&got, &want, 1e-5, "双线性插值的线性");
    }

    /// 平均池化同样是线性算子，且线性律与几何无关（不像"窗口均值"那样复述实现）。
    #[test]
    fn avg_pool2d_is_linear(
        x1 in fixed(16), x2 in fixed(16), alpha in prop::finite(), beta in prop::finite()
    ) {
        let len = avg_pool2d::H * avg_pool2d::W;
        let mixed: Vec<f32> = x1.iter().zip(&x2).map(|(a, b)| alpha * a + beta * b).collect();
        let got = window_run(avg_pool2d::forward, &mixed, len);
        let left = window_run(avg_pool2d::forward, &x1, len);
        let right = window_run(avg_pool2d::forward, &x2, len);
        let want: Vec<f32> = left.iter().zip(&right).map(|(a, b)| alpha * a + beta * b).collect();
        prop::assert_within_rel(&got, &want, 1e-5, "平均池化的线性");
    }
}

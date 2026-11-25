//! `nn` 的属性测试：装配层的代数与"声明的可计价性"。
//!
//! 这些性质不依赖任何具体形状或机器，是**任何**合法层清单都必须满足的：
//!
//! | 性质 | 若被违反说明什么 |
//! |---|---|
//! | `Bounds` 加法的交换/结合律 | 总和取决于层在文件里的先后顺序 |
//! | `bytes = read + write`、`intensity = flops / bytes` | 派生量与账本脱钩 |
//! | 下界对自身 `moves_within` | 融合判定里"自己不高于自己"不成立 |
//! | `Stack::floor` 与顺序无关、且等于逐层之和 | 装配侧算的总量不是各层之和 |
//! | 一层不可计价不清零其余层 | 一张 LoRA 表把 300 层的和归零 |
//! | 重名被拒、异名被收 | 按名字对齐的遍历变成猜测 |
//! | 分组对齐宽度必有下界 | 构造成功却推不出数，或反之 |
//!
//! 与 `testkit::compare` 的关系：那套容差判据只接受 `&[f32]`，而 `Bounds` 的 `flops` 是 `f64`。
//! 本文件对 `f64` 用同一口径的投影（`|got − want| ≤ tol · max(1, |want|)`），不是第二套真理。
//!
//! 分组的对齐口径（N8 信任清单）：本文件只断言**诚实形状**——宽度取分组因子的整数倍，
//! 因此不对"非对齐宽度该报哪条 `Code`"作断言；那是 `prof` 的代价模型自己的契约。

use super::*;
use diagnostic::Code;
use proptest::prelude::*;
use testkit::assert_near;

/// 分组因子的整数倍：任何量化格式的分组都能整除它，f32 格式也不受影响。
fn aligned(units: usize) -> usize {
    32 * units
}

fn units() -> impl Strategy<Value = usize> {
    1usize..=64
}

/// 唯一层名：`Stack` 按名字对齐，重名会让整条遍历变成猜测。
fn named(idx: usize) -> String {
    format!("blk.0.layer_{idx}")
}

/// 一组层：`(输入宽度, 输出宽度)`，宽度都对齐到分组因子。
fn pool() -> impl Strategy<Value = Vec<(usize, usize)>> {
    prop::collection::vec((units(), units()), 1..=6).prop_map(|pairs| {
        pairs
            .into_iter()
            .map(|(a, b)| (aligned(a), aligned(b)))
            .collect()
    })
}

fn priced(pairs: &[(usize, usize)]) -> Vec<Linear> {
    pairs
        .iter()
        .enumerate()
        .map(|(idx, &(ins, out))| {
            Linear::new(named(idx), ins, out).expect("分组对齐的宽度必然可构造")
        })
        .collect()
}

fn bounds(pairs: &[(usize, usize)]) -> Vec<Bounds> {
    priced(pairs)
        .iter()
        .map(|layer| layer.bounds().expect("形状合法").expect("编码有算子"))
        .collect()
}

fn total(values: &[Bounds]) -> Bounds {
    values.iter().fold(Bounds::ZERO, |acc, b| acc.plus(*b))
}

proptest! {
    #![proptest_config(testkit::prop::config())]

    /// 总和与层在清单里的先后无关：`bytes` 是加法，不是"最后一层说了算"。
    #[test]
    fn floor_is_order_independent(pairs in pool(), seed in any::<u64>()) {
        let forward = Stack::new(priced(&pairs)).expect("名字唯一").floor().total();
        let mut reversed = priced(&pairs);
        reversed.reverse();
        let backward = Stack::new(reversed).expect("名字唯一").floor().total();
        prop_assert_eq!(forward, backward);

        // 再按 seed 洗一次：只有加法满足交换律时才会一致。
        let mut shuffled = priced(&pairs);
        let len = shuffled.len();
        for i in 0..len {
            let j = (seed.rotate_left(i as u32) as usize) % len;
            shuffled.swap(i, j);
        }
        let mixed = Stack::new(shuffled).expect("名字唯一").floor().total();
        prop_assert_eq!(forward, mixed);
    }

    /// 装配侧的总量必须等于逐层下界之和——否则"总和"不是算出来的。
    #[test]
    fn floor_is_the_sum_of_its_layers(pairs in pool()) {
        let per_layer = bounds(&pairs);
        let stack = Stack::new(priced(&pairs)).expect("名字唯一");
        let floor = stack.floor();
        prop_assert_eq!(floor.total(), total(&per_layer));
        prop_assert_eq!(floor.total().bytes(), total(&per_layer).bytes());
        prop_assert!(floor.unpriced().is_empty(), "对齐宽度的层都可计价");
    }

    /// 一层不可计价的层只进清单，不清零其余层。
    #[test]
    fn an_unpriced_layer_does_not_zero_the_floor(pairs in pool()) {
        let layers = priced(&pairs);
        let priced_total = Stack::new(layers.clone()).expect("名字唯一").floor().total();
        let mut mixed: Vec<Box<dyn Contract>> = vec![Box::new(Ghost::new("blk.0.norm"))];
        mixed.extend(layers.into_iter().map(|layer| Box::new(layer) as Box<dyn Contract>));
        let floor = Stack::of(mixed).expect("名字唯一").floor();
        prop_assert_eq!(floor.total(), priced_total, "不可计价的层不贡献也不清零");
        prop_assert_eq!(floor.unpriced().len(), 1);
        prop_assert_eq!(floor.unpriced()[0].name(), "blk.0.norm");
    }

    /// `Bounds` 的加法是交换且结合的：任意组合顺序得到同一本账。
    #[test]
    fn bounds_addition_is_commutative_and_associative(pairs in pool()) {
        let values = bounds(&pairs);
        if values.len() < 2 {
            return Ok(());
        }
        let (a, b) = (values[0], values[1]);
        prop_assert_eq!(a.plus(b), b.plus(a));
        let c = total(&values);
        prop_assert_eq!(a.plus(b).plus(c), a.plus(b.plus(c)));
    }

    /// `bytes` 是派生量：读加写，而不是第三个数。
    #[test]
    fn bytes_are_the_read_write_sum(pairs in pool()) {
        for b in bounds(&pairs) {
            prop_assert_eq!(b.bytes(), b.read() + b.write());
            prop_assert!(b.read() > 0, "投影至少要读一遍权重");
            prop_assert!(b.write() > 0, "投影至少要写一份输出");
        }
    }

    /// 算术强度就是 `flops / bytes`：roofline 的横轴不允许有第二份定义。
    #[test]
    fn intensity_is_flops_over_bytes(pairs in pool()) {
        for b in bounds(&pairs) {
            assert_near(b.intensity(), b.flops() / b.bytes() as f64, 1e-15, "算术强度");
        }
    }

    /// 融合判定必须自反：没有哪个下界比它自己更省。
    #[test]
    fn every_floor_moves_within_itself(pairs in pool()) {
        for b in bounds(&pairs) {
            prop_assert!(b.moves_within(b));
            prop_assert!(Bounds::ZERO.moves_within(b), "零下界落在任何下界之内");
        }
    }

    /// 分组对齐的三种编码都必须推得出数：构造成功却推不出下界，等于层白声明。
    #[test]
    fn group_aligned_widths_are_priceable(ins in units(), out in units(), format in 0usize..3) {
        let format = match format {
            0 => WeightFormat::INT4,
            1 => WeightFormat::INT8,
            _ => WeightFormat::F32,
        };
        let layer = Linear::with_format("blk.0.proj", format, aligned(ins), aligned(out))
            .expect("分组对齐的形状可构造");
        let floor = layer.bounds().expect("形状合法").expect("这三种编码都有算子");
        prop_assert!(floor.read() > 0);
        prop_assert_eq!(layer.output().dims(), [aligned(out)]);
        prop_assert_eq!(layer.spca().ops().len(), 1);
    }

    /// 零宽度不是层：退化形状必须显式报错，而不是构造出一个零流量层。
    #[test]
    fn zero_width_is_rejected(ins in units(), out in units(), zero_in in any::<bool>()) {
        let ins = if zero_in { 0 } else { aligned(ins) };
        let out = if zero_in { aligned(out) } else { 0 };
        let err = Linear::new("blk.0.proj", ins, out).expect_err("零宽度必须被拒");
        prop_assert_eq!(err.code(), Code::Invalid);
    }

    /// 重名被拒、异名被收：名字是装配侧唯一的对齐键。
    #[test]
    fn duplicate_names_are_rejected(pairs in pool()) {
        let mut layers = priced(&pairs);
        let clash = layers[0].clone();
        let first = clash.name().to_string();
        layers.push(clash);
        let err = Stack::new(layers).expect_err("重名必须被拒");
        prop_assert_eq!(err.code(), Code::Invalid);
        prop_assert_eq!(first, "blk.0.layer_0");
    }

    /// 同一批层经类型擦除（`Stack::of`）与直接构造（`Stack::new`）得到同一本账。
    #[test]
    fn type_erasure_preserves_the_floor(pairs in pool()) {
        let direct = Stack::new(priced(&pairs)).expect("名字唯一").floor().total();
        let erased = Stack::of(
            priced(&pairs)
                .into_iter()
                .map(|layer| Box::new(layer) as Box<dyn Contract>)
                .collect(),
        )
        .expect("名字唯一")
        .floor()
        .total();
        prop_assert_eq!(direct, erased);
    }
}

/// 一个只声明名字与输出形状的层：`Contract::bounds` 走默认实现，得到 `Ok(None)`。
///
/// 它复刻"注解层没给出输入宽度"的真实情况，用来验证不可计价的层不会污染总和。
struct Ghost {
    spec: ContractSpec,
}

impl Ghost {
    fn new(name: &str) -> Self {
        Self {
            spec: ContractSpec::new(name, Shape::new([8]), core::iter::empty::<&str>())
                .expect("空算子列表合法"),
        }
    }
}

impl Contract for Ghost {
    fn name(&self) -> &str {
        self.spec.name()
    }

    fn output(&self) -> &Shape {
        self.spec.output()
    }

    fn spca(&self) -> &ContractSpec {
        &self.spec
    }
}

proptest! {
    /// 循环层的参数字节数必须能由门表独立重算：每层每方向有输入侧矩阵
    /// `门数·隐宽 × 输入宽`、隐侧矩阵 `门数·隐宽 × 隐宽`、以及两条偏置 `门数·隐宽`。
    /// 若被违反，说明 `param_bytes` 里藏了第二份门数、第二份层宽口径，或漏算了一条偏置。
    #[test]
    fn recurrent_parameters_follow_the_gate_table(
        pick in 0usize..3,
        input in units(),
        hidden in units(),
        layers in 1usize..=4,
    ) {
        let cell = Cell::ALL[pick];
        let layer = Recurrent::new("r", cell, input, hidden, layers, 1, true).expect("合法");
        let gates = cell.gates() as u64;
        let mut want = 0u64;
        for level in 0..layers {
            let width = if level == 0 { input } else { hidden };
            want += gates * hidden as u64 * (width as u64 + hidden as u64 + 2);
        }
        prop_assert_eq!(layer.param_bytes(), want * 4);
    }

    /// 加一层只能更贵：深度是最没理由让下界变小的维度。
    #[test]
    fn a_deeper_recurrent_stack_is_never_cheaper(
        pick in 0usize..3,
        input in units(),
        hidden in units(),
        layers in 1usize..=3,
    ) {
        let cell = Cell::ALL[pick];
        let floor = |depth| {
            Recurrent::new("r", cell, input, hidden, depth, 1, true)
                .expect("合法")
                .bounds()
                .expect("合法")
                .expect("有算子")
        };
        let (shallow, deep) = (floor(layers), floor(layers + 1));
        prop_assert!(deep.read() > shallow.read(), "读侧随深度不减");
        prop_assert!(deep.write() > shallow.write(), "写侧随深度不减");
        prop_assert!(deep.flops() > shallow.flops(), "算术量随深度不减");
    }

    /// 双向精确是单向的两倍：两个方向除了拼接之外不共享任何代价。
    #[test]
    fn a_bidirectional_recurrent_layer_costs_exactly_twice(
        pick in 0usize..3,
        input in units(),
        hidden in units(),
    ) {
        let cell = Cell::ALL[pick];
        let floor = |dirs| {
            Recurrent::new("r", cell, input, hidden, 1, dirs, true)
                .expect("合法")
                .bounds()
                .expect("合法")
                .expect("有算子")
        };
        let (uni, bi) = (floor(1), floor(2));
        prop_assert_eq!(bi.read(), uni.read() * 2);
        prop_assert_eq!(bi.write(), uni.write() * 2);
        prop_assert_eq!(bi.flops(), uni.flops() * 2.0);
    }
}

mod conv_property {
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
        conv2d::forward(
            tensor::View::one(x),
            tensor::View::one(weight),
            tensor::View::one(bias),
            &mut out,
        )
        .expect("定长缓冲");
        out
    }

    /// 一维窗口族与插值的统一调用口径：输入定长、输出定长、无参数。
    fn window_run(
        forward: fn(tensor::View<'_, f32>, &mut [f32]) -> diagnostic::Result<()>,
        x: &[f32],
        len: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; len];
        forward(tensor::View::one(x), &mut out).expect("定长缓冲");
        out
    }

    fn attention_run(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; s * d];
        attention::forward(
            tensor::View::matrix(s, d, q).expect("q 二维"),
            tensor::View::matrix(s, d, k).expect("k 二维"),
            tensor::View::matrix(s, d, v).expect("v 二维"),
            &mut out,
        )
        .expect("等长");
        out
    }

    fn flash_run(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; s * d];
        attention::flash_forward(
            tensor::View::matrix(s, d, q).expect("q 二维"),
            tensor::View::matrix(s, d, k).expect("k 二维"),
            tensor::View::matrix(s, d, v).expect("v 二维"),
            &mut out,
        )
        .expect("等长");
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
}

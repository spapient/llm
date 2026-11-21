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
//! 与 `testkit::rel` 的关系：那套容差判据只接受 `&[f32]`，而 `Bounds` 的 `flops` 是 `f64`。
//! 本文件对 `f64` 用同一口径的投影（`|got − want| ≤ tol · max(1, |want|)`），不是第二套真理。
//!
//! 分组的对齐口径（N8 信任清单）：本文件只断言**诚实形状**——宽度取分组因子的整数倍，
//! 因此不对"非对齐宽度该报哪条 `Code`"作断言；那是 `op` 的代价模型自己的契约。

use super::*;
use diagnostic::Code;
use proptest::prelude::*;

/// 分组因子的整数倍：任何量化格式的分组都能整除它，f32 格式也不受影响。
fn aligned(units: usize) -> usize {
    32 * units
}

fn units() -> impl Strategy<Value = usize> {
    1usize..=64
}

/// 唯一层名：`Stack` 按名字对齐，重名会让整条遍历变成猜测。
fn named(index: usize) -> String {
    format!("blk.0.layer_{index}")
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
        .map(|(index, &(ins, out))| {
            Linear::new(named(index), ins, out).expect("分组对齐的宽度必然可构造")
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

/// `|got − want| ≤ tol · max(1, |want|)`：`testkit::rel` 的 f64 投影。
fn assert_near(got: f64, want: f64, tol: f64, what: &str) {
    let scale = want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol * scale,
        "{what}: got {got}, want {want}（tol = {tol}，scale = {scale}）"
    );
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
            spec: ContractSpec::new(name, Shape::new([8]), Vec::new()).expect("空算子列表合法"),
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

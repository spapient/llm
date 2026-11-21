//! 样本统计：均值、标准差、最近秩分位，以及「对任意统计量重采样」的区间。
//!
//! 纯函数，零领域类型：只吃 `&[f64]` 与一个「把这列样本聚成一个数」的闭包——选哪个统计量、
//! 门槛在哪，都是协议卡的事（[`crate::card`]）。重采样与分位的写法全仓只有这一份。
//!
//! 不引入 `rng`：这里要的是一条**可复现的下标序列**（同一份样本必须逐位对拍），不是随机数契约；
//! 为六行 splitmix64 换一条依赖边不划算，也不该让静态分析器链入运行时的随机源。

/// 重采样次数：协议卡报的是 95% 区间，两千次抽样的分位点已经稳定到卡片精度之内。
pub const RESAMPLES: usize = 2000;

/// 重采样种子固定：**同一份样本只会得到一段区间**，复核时能逐位对拍，报告也不随机漂移。
pub const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// 样本均值；空列返回 0（调用方在此之前已排除空列）。
#[must_use]
pub fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// 样本标准差（`n − 1` 分母）；少于 2 个样本时返回 0。
#[must_use]
pub fn sd(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let center = mean(samples);
    let sum: f64 = samples.iter().map(|value| (value - center).powi(2)).sum();
    (sum / (samples.len() - 1) as f64).sqrt()
}

/// 最近秩分位：`q` 落进哪个秩就取哪个秩，不做插值（样本量小的时候插值会让分位看起来更稳）。
///
/// `sorted` 必须已按 `f64::total_cmp` 升序排好。
#[must_use]
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len() as f64;
    let rank = (q * last).ceil().clamp(1.0, last);
    sorted[rank as usize - 1]
}

/// 对统计量做有放回重采样，返回每次抽样的取值（升序）；无定义的抽样直接丢弃。
///
/// `draw` 就是「一列样本 → 一个数」的统计量本身，所以重采样与统计量同坐标。
#[must_use]
pub fn resample(samples: &[f64], draw: impl Fn(&[f64]) -> Option<f64>) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut state = SEED;
    let mut drawn = Vec::with_capacity(samples.len());
    let mut values = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        drawn.clear();
        drawn.extend(
            samples
                .iter()
                .map(|_| samples[index(&mut state, samples.len())]),
        );
        if let Some(value) = draw(&drawn) {
            values.push(value);
        }
    }
    values.sort_by(f64::total_cmp);
    values
}

/// 统计量的重采样区间（升序）；统计量在整列上无定义、或有效抽样不到一半时 `None`。
#[must_use]
pub fn interval(samples: &[f64], draw: impl Fn(&[f64]) -> Option<f64>) -> Option<Vec<f64>> {
    draw(samples)?;
    let draws = resample(samples, draw);
    (draws.len() >= RESAMPLES / 2).then_some(draws)
}

/// splitmix64：状态只有 64 位，重采样可复现，不引入随机依赖。
fn index(state: &mut u64, len: usize) -> usize {
    *state = state.wrapping_add(SEED);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let z = z ^ (z >> 31);
    (z % len as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1..=5 的均值 3、样本标准差 √2.5：查表可算，不靠实现自证。
    #[test]
    fn mean_and_sample_deviation_match_the_closed_form() {
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&samples), 3.0);
        assert!((sd(&samples) - 2.5_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn a_single_sample_has_no_spread() {
        assert_eq!(sd(&[7.0]), 0.0);
        assert_eq!(mean(&[]), 0.0);
    }

    /// 最近秩：偶数个样本时中位取较低那个秩，不插值成中间值。
    #[test]
    fn quantiles_are_nearest_rank_not_interpolated() {
        let sorted = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&sorted, 0.50), 2.0);
        assert_eq!(quantile(&sorted, 0.95), 4.0);
        assert_eq!(quantile(&sorted, 0.0), 1.0, "下界夹住在最小秩");
    }

    /// 同种子同样本 ⇒ 同一段区间：这张卡才能被复核。
    #[test]
    fn resampling_is_reproducible() {
        let samples = [9.0, 11.0, 10.5, 9.4, 12.0, 10.1, 9.9, 10.8];
        let first = interval(&samples, |drawn| Some(mean(drawn)));
        let second = interval(&samples, |drawn| Some(mean(drawn)));
        assert_eq!(first, second);
        let draws = first.expect("区间");
        assert!(draws.windows(2).all(|pair| pair[0] <= pair[1]), "必须升序");
        assert!(
            draws[0] <= 10.0 && 10.0 <= draws[draws.len() - 1],
            "区间应罩住真值"
        );
    }

    /// 统计量在整列上无定义时没有区间——不能拿空抽样假装「很稳」。
    #[test]
    fn an_undefined_statistic_has_no_interval() {
        assert_eq!(interval(&[1.0, -1.0], |_| None), None);
        assert!(resample(&[], |_| Some(0.0)).is_empty());
    }
}

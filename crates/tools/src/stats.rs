//! 样本统计：均值、标准差、最近秩分位，以及「对任意统计量重采样」的区间。

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
#[must_use]
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len() as f64;
    let rank = (q * last).ceil().clamp(1.0, last);
    sorted[rank as usize - 1]
}

/// 对统计量做有放回重采样，返回每次抽样的取值（升序）；无定义的抽样直接丢弃。
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

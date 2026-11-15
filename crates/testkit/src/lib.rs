//! 测试约定：浮点对拍、真机门控、临时目录。**只作 dev-dependency**。
//!
//! **拥有**：`approx_eq`、`device_enabled`、`scratch_dir`。**不拥有**：任何生产逻辑——它是
//! 测试夹具，不进任何二进制。**依赖**：无（夹具不该把被测 crate 拉进来）。**被依赖**：各成员
//! 的 `[dev-dependencies]`。
//!
//! 这里的 [`DEVICE_ENV`] 是**测试门控变量，不是产品配置**：它不进 `cli::settings` 的键表，
//! 生产代码不得读取它。真机用例的形态固定为"默认跳过、显式开门才跑"：
//!
//! ```
//! if llm_testkit::device_enabled("nvidia") {
//!     // 需要 nvcc + 真卡的用例
//! }
//! ```

#![deny(unsafe_code)]

use std::path::PathBuf;

/// 真机门控环境变量（测试专用，不是产品配置）。
///
/// 取值是逗号分隔的 vendor 列表（`cpu,nvidia`），`all` 表示全部打开；未设置表示全部关闭。
pub const DEVICE_ENV: &str = "LLM_TEST_DEVICE";

/// 逐个元素比较，`NaN` 与 `NaN` 视为相等（对拍时它两边同源，不该因为 `NaN != NaN` 失败）。
///
/// 两个**相同的无限大**也算相等：`inf - inf` 是 `NaN`，只看差值的判定会把「两边同为无穷」
/// 判成不等，而它与 `NaN` 对 `NaN` 是同一个理由。
///
/// 长度不同直接返回 `false`：对拍的首要条件是形状一致。
pub fn approx_eq(left: &[f32], right: &[f32], tolerance: f32) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right).all(|(a, b)| {
        if a == b || (a.is_nan() && b.is_nan()) {
            return true;
        }
        (a - b).abs() <= tolerance
    })
}

/// 这台设备的真机用例是否被显式打开。
pub fn device_enabled(vendor: &str) -> bool {
    match std::env::var(DEVICE_ENV) {
        Ok(spec) => matches(&spec, vendor),
        Err(_) => false,
    }
}

/// 逗号列表匹配：`all` 全开，空白被忽略，空列表等于没开。
fn matches(spec: &str, vendor: &str) -> bool {
    spec.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .any(|item| item == "all" || item == vendor)
}

/// 一个已创建的空临时目录，标签用于区分用例。
///
/// 目录名带进程号，避免并行测试互相踩。建不出来不在这里终止测试进程：返回的路径照样交给
/// 调用方，它紧接着的写入会带出带路径与 io 类别的错误，比一句「创建失败」更有用。
pub fn scratch_dir(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("llm-testkit-{}-{tag}", std::process::id()));
    if let Err(error) = std::fs::create_dir_all(&path) {
        eprintln!("testkit: 无法创建临时目录 `{}`：{error}", path.display());
    }
    path
}

/// 确定性 PRNG（xorshift）：测试共享，不给依赖表加 `rand`，同时让失败可复现。
pub struct Rng {
    state: u32,
}

impl Rng {
    /// 固定种子。
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// 取 `[0, bound)` 内的确定性值。
    pub fn below(&mut self, bound: u32) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state % bound.max(1)
    }
}

/// CRC-32 反射多项式（IEEE 802.3 的 `0x04C1_1DB7` 反射形式）。
const CRC_POLY: u32 = 0xEDB8_8320;

/// CRC-32 的逐字节参考定义：与各 crate 的查表实现独立，供对拍共享，免得每份测试重写一遍。
#[must_use]
pub fn crc32_bitwise(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ CRC_POLY
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// 两条同长向量的余弦相似度；长度不同、或任一向量范数为零时返回 [`None`]。
///
/// 回归判据用：量化前向的 logits 与 f32 参考比对方向一致性（不要求逐元素相等）。累加在 `f64`
/// 完成，避免长向量上 `f32` 平方和与点积的累加误差把相似度压低；返回值落在 `[-1, 1]`。
#[must_use]
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut left_sq = 0.0f64;
    let mut right_sq = 0.0f64;
    for (a, b) in left.iter().zip(right) {
        let a = f64::from(*a);
        let b = f64::from(*b);
        dot += a * b;
        left_sq += a * a;
        right_sq += b * b;
    }
    let denom = (left_sq * right_sq).sqrt();
    if denom == 0.0 {
        return None;
    }
    Some((dot / denom) as f32)
}

#[cfg(test)]
mod tests;

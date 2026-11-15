//! 测试约定：浮点对拍、真机门控、临时目录、跳过登记。**只作 dev-dependency**。
//!
//! **拥有**：`approx_eq`、`device_enabled`、`scratch_dir`、`Rng`、
//! `cosine_similarity`、回归阈值 `COSINE_GUARD` 与跳过登记 [`skip`]。**不拥有**：任何生产
//! 逻辑——它是测试夹具，不进任何二进制。**依赖**：无（夹具不该把被测 crate 拉进来）。
//! **被依赖**：各成员的 `[dev-dependencies]`。
//!
//! 这里的真机门控变量是**测试门控，不是产品配置**：它不进 `cli::settings` 的键表，
//! 生产代码不得读取它。真机用例的形态固定为"默认跳过、显式开门才跑"：
//!
//! ```
//! if llm_testkit::device_enabled("nvidia") {
//!     // 需要 nvcc + 真卡的用例
//! }
//! ```
//!
//! 前提不满足的用例必须走 [`skip`] 登记，禁止裸 `return` 空转：跳过原因进进程内计数与
//! 结构化日志，设置 `LLM_TEST_SKIP_DIR` 时还会追加到该目录的 `skips.log`，供 CI 汇总。

#![deny(unsafe_code)]

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// 真机门控环境变量名（测试专用，不是产品配置）。取值是逗号分隔的 vendor 列表
/// （`cpu,nvidia`），`all` 表示全部打开；未设置表示全部关闭。
const DEVICE_ENV: &str = "LLM_TEST_DEVICE";

/// 量化 logits 与 f32 参考的余弦回归下界（方向一致性，尺度不变）。
///
/// 合成基准与未来的真实样本集共用这一条阈值，避免各用例各抄一个数字、判据悄悄漂移。
pub const COSINE_GUARD: f32 = 0.999;

/// 逐个元素**绝对误差**比较：两套同尺度算法变体对拍走这一档。`NaN` 与 `NaN` 视为相等
/// （对拍时它两边同源，不该因为 `NaN != NaN` 失败）。
///
/// 两个**相同的无限大**也算相等：`inf - inf` 是 `NaN`，只看差值的判定会把「两边同为无穷」
/// 判成不等，而它与 `NaN` 对 `NaN` 是同一个理由。
///
/// 长度不同直接返回 `false`：对拍的首要条件是形状一致。
///
/// 判据分三档，不要混用：同尺度逐元素对拍用本函数；量化 logits 的方向一致性用
/// [`cosine_similarity`] + [`COSINE_GUARD`]；解析梯度 vs 数值梯度这类随量级缩放的比对，
/// 误差预算是「绝对 + 相对」双项，由用例本地按步长写 slack，不套绝对容差。
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

/// 进程内跳过计数与原因清单：标准测试框架没有正式 skip 信号，集中登记后才能在报告里
/// 区分"通过"与"前提不满足而未运行"。
static SKIPPED: AtomicUsize = AtomicUsize::new(0);
static SKIP_REASONS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// 登记一条"环境前提不满足"的跳过；恒返回 `true`，调用方据此立即 `return`。
///
/// 每条跳过写一行结构化记录 `llm-testkit[skip]: <原因>`，并计入进程内计数；若设置了
/// `LLM_TEST_SKIP_DIR`，同时追加到该目录的 `skips.log`，跨测试二进制汇总时按文件读取。
/// 原因要写清缺什么（设备 / 编译器 / 样本目录 / 旧基线二进制），不留"不知道为什么没跑"。
pub fn skip(reason: impl Into<String>) -> bool {
    let reason = reason.into();
    SKIPPED.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut reasons) = SKIP_REASONS.lock() {
        reasons.push(reason.clone());
    }
    eprintln!("llm-testkit[skip]: {reason}");
    if let Some(dir) = std::env::var_os("LLM_TEST_SKIP_DIR") {
        let dir = PathBuf::from(dir);
        if std::fs::create_dir_all(&dir).is_ok() {
            let path = dir.join("skips.log");
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                use std::io::Write;
                let _ = writeln!(file, "{reason}");
            }
        }
    }
    true
}

/// 登记一条跳过并返回 `None`：供 `Option` 链式表达式（如 `ok().or_else(...)`）内联使用，
/// 避免在测试臂里再开一个语句块；类型 `T` 由上下文推断。
#[must_use]
pub fn skip_none<T>(reason: impl Into<String>) -> Option<T> {
    skip(reason);
    None
}

/// 本进程已登记的跳过条数。
#[must_use]
pub fn skip_count() -> usize {
    SKIPPED.load(Ordering::Relaxed)
}

/// 本进程已登记的跳过原因（按登记顺序）。
#[must_use]
pub fn skipped_reasons() -> Vec<String> {
    SKIP_REASONS
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default()
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

//! 测试约定：浮点对拍、真机门控、临时目录、随机数、位级 CRC 参考。**只作 dev-dependency**。
//!
//! **拥有**：`approx_eq`、`close`、`all_finite`、`cosine_similarity`、`COSINE_GUARD`
//! （[`compare`]）；`device_enabled`、`device_or_skip`、`skip`、`skip_none`、`skip_count`、
//! `skipped_reasons`（[`gating`]）；`scratch_dir`、`scratch_clean`（[`scratch`]）；
//! [`Rng`]（[`rng`]）；`crc32_bitwise`（[`hash`]）；`safetensors_bytes` 与 `pack_codes`
//! （本文件，格式夹具）。**不拥有**：任何生产逻辑——它是测试
//! 夹具，不进任何二进制。**依赖**：无（夹具不该把被测 crate 拉进来）。
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

pub mod compare;
pub mod gating;
pub mod hash {
    //! 位级 CRC-32 参考定义：与各 crate 的查表实现独立，供对拍共享，免得每份测试重写一遍。

    /// CRC-32 反射多项式（IEEE 802.3 的 `0x04C1_1DB7` 反射形式）。
    const CRC_POLY: u32 = 0xEDB8_8320;

    /// CRC-32 的逐字节参考定义。
    ///
    /// 逐位循环是 CRC 移位寄存器的固有串行依赖：每一位的余数由上一位移出的比特决定，
    /// 无法整块查表；这一份刻意保留位级写法，作为查表实现的独立对拍基准。
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

    /// 把某个 crate 的查表 CRC-32 实现与位级参考对拍：`0..=200` 递增序列、空串与标准 check 值。
    pub fn assert_table_matches_bitwise(crc32: impl Fn(&[u8]) -> u32) {
        let data: Vec<u8> = (0u8..=200).collect();
        assert_eq!(crc32(&data), crc32_bitwise(&data));
        assert_eq!(crc32(b""), crc32_bitwise(b""));
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926, "标准 check 值");
    }
}

pub mod rng {
    //! 确定性伪随机数：测试共享一个不带 `rand` 依赖的 xorshift32，让失败可复现。

    /// 确定性 PRNG（xorshift32）：测试共享，不给依赖表加 `rand`，同时让失败可复现。
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

        /// `±[0.5, 2]` 的确定性值：远离 0，避免平方类算子的差分为 0 造成无谓抵消。
        pub fn value(&mut self) -> f64 {
            let magnitude = 0.5 + 1.5 * f64::from(self.below(1_000)) / 1_000.0;
            match self.below(2) {
                0 => magnitude,
                _ => -magnitude,
            }
        }

        /// `[0, 1)` 内的确定性 f32。
        pub fn next_f32(&mut self) -> f32 {
            self.below(1_000_000) as f32 / 1_000_000.0
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rng_is_deterministic_and_bounded() {
            let mut left = Rng::new(1234);
            let mut right = Rng::new(1234);
            for _ in 0..16 {
                let value = left.below(10);
                assert_eq!(value, right.below(10), "同种子序列一致");
                assert!(value < 10, "结果落在界内");
            }
            assert_eq!(Rng::new(7).below(1), 0, "bound=1 恒为 0");
        }

        #[test]
        fn rng_value_stays_away_from_zero_and_is_signable() {
            let mut rng = Rng::new(42);
            for _ in 0..16 {
                let value = rng.value();
                assert!(value.abs() >= 0.5 && value.abs() <= 2.0);
            }
        }

        #[test]
        fn rng_next_f32_is_in_unit_interval() {
            let mut rng = Rng::new(7);
            for _ in 0..16 {
                let value = rng.next_f32();
                assert!((0.0..1.0).contains(&value));
            }
        }
    }
}

pub mod scratch {
    //! 测试临时目录：按标签取一个进程隔离的空目录。

    use std::path::PathBuf;

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

    /// 先删掉旧目录再建一个空目录：进入用例时目录里不该带上一次运行的残留。
    ///
    /// 与 [`scratch_dir`] 的复用语义不同，这里每次都清空；适用于要逐条枚举目录内容
    /// （如"写回后目录里只有这一个产物"）的对拍。目录名带进程号，并行测试互不踩。
    pub fn scratch_clean(tag: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("llm-testkit-clean-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        if let Err(error) = std::fs::create_dir_all(&path) {
            eprintln!("testkit: 无法创建临时目录 `{}`：{error}", path.display());
        }
        path
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_same_tag_reuses_one_directory() {
            let first = scratch_dir("a");
            let second = scratch_dir("a");
            assert!(first.is_dir());
            assert_eq!(first, second, "同标签同进程应复用同一目录");
            assert_ne!(first, scratch_dir("b"), "不同标签必须各自一个目录");
            let _ = std::fs::remove_dir_all(&first);
            let _ = std::fs::remove_dir_all(scratch_dir("b"));
        }

        #[test]
        fn scratch_clean_wipes_prior_content() {
            let dir = scratch_clean("clean");
            std::fs::write(dir.join("leftover.txt"), "x").expect("写入残留");
            assert!(dir.join("leftover.txt").exists());
            let again = scratch_clean("clean");
            assert!(again.is_dir());
            assert!(
                !again.join("leftover.txt").exists(),
                "第二次取同标签必须先清空残留"
            );
            let _ = std::fs::remove_dir_all(&again);
        }
    }
}

/// 按 safetensors 布局拼一份完整文件：8 字节小端头长 + 头部 JSON + 依序拼接的数据段。
///
/// `data` 的顺序必须与 `json` 里 `data_offsets` 的偏移一致；条目名用于在调试构建里核对头部
/// 确实声明了这段数据（不引入 JSON 解析器，守住本 crate 的零依赖）。
#[must_use]
pub fn safetensors_bytes(json: &str, data: &[(&str, Vec<u8>)]) -> Vec<u8> {
    for (name, _) in data {
        debug_assert!(json.contains(name), "头部 JSON 必须声明数据段 `{name}`");
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json.as_bytes());
    for (_, chunk) in data {
        bytes.extend_from_slice(chunk);
    }
    bytes
}

/// 把 `bits` 位有符号码按低 `bits` 位打包：`zero = 2^(bits-1)` 平移，低 `bits` 位落在字节内。
///
/// 一个字节按 `8 / bits` 个槽从低到高排；尾组不足一槽时补零码。位宽取 1..=8（整除 8 的
/// 2/4/8 是常规用法），与 `op` 的量化 GEMV 参考共用同一口径。
#[must_use]
pub fn pack_codes(bits: usize, values: &[i8]) -> Vec<u8> {
    let per_byte = 8 / bits;
    let zero = 1i16 << (bits - 1);
    let mask = u8::MAX >> (8 - bits);
    values
        .chunks(per_byte)
        .map(|group| {
            (0..per_byte).fold(0u8, |byte, slot| {
                let code = (i16::from(group.get(slot).copied().unwrap_or(0)) + zero) as u8 & mask;
                byte | (code << (slot * bits))
            })
        })
        .collect()
}

pub use compare::{COSINE_GUARD, all_finite, approx_eq, close, cosine_similarity};
pub use gating::{device_enabled, device_or_skip, skip, skip_count, skip_none, skipped_reasons};
pub use hash::{assert_table_matches_bitwise, crc32_bitwise};
pub use rng::Rng;
pub use scratch::{scratch_clean, scratch_dir};

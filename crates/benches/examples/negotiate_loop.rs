//! 协商热路径的独立采样用例：不起 criterion，在 release 紧循环里反复调用 `cap::negotiate`。
//!
//! 用途是给采样器（`sample` / `samply`）提供可解析的栈帧：criterion 二进制在预热与测量框架
//! 上占掉大量样本，且 release 默认剥掉调试信息，1 µs 量级的被测函数坐实不了热点。这个用例把
//! 样本集中到 `negotiate` 本身；它不产出基准数字，耗时判定仍归 criterion 与 `tool::regress`。
//!
//! 编译时保留符号（默认 release 会 `strip = "debuginfo"`）：
//!
//! ```sh
//! CARGO_PROFILE_RELEASE_DEBUG=1 CARGO_PROFILE_RELEASE_STRIP=false \
//!   cargo build -p llm-benches --example negotiate_loop --release
//! ```

use cutile::builtin::Catalogue;
use cutile::cap;
use std::hint::black_box;

/// 紧循环次数：按约 1 µs/次估算，留出 2–3 s 的可采样窗口；采样时可用环境变量拉长。
const DEFAULT_ITERATIONS: u64 = 2_000_000;

fn main() {
    let iterations: u64 = std::env::var("ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_ITERATIONS);
    let catalogue = Catalogue::try_builtin().expect("内置目录");
    let op = catalogue.op("gemv_i4").expect("内置算子 gemv_i4");
    let profile = catalogue.profile("cpu").expect("内置 cpu 档案");

    let started = std::time::Instant::now();
    for _ in 0..iterations {
        let path = cap::negotiate(black_box(op), black_box(profile)).expect("cpu 档案应命中");
        black_box(&path);
    }
    let elapsed = started.elapsed();
    eprintln!(
        "negotiate_loop: {iterations} 次，耗时 {elapsed:?}，单次 {:.2} ns",
        elapsed.as_nanos() as f64 / iterations as f64
    );
}

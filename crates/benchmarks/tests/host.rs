//! 主机档案查询、harness 构造与 criterion 估读。

use super::*;

/// 一台机器的档案：标识、机型、系统、工具链。
///
/// 机型写成数据而不是注释，是因为它必须随报告归档：上一轮的注释把 M5 Pro 写成 M4 Pro，
/// 而 `os-algorithm` 完全看不出这个差别。没有档案的主机不判——跨机数字不可比。档案经构造器填进
/// [`Harness`]，本结构不再直接进 JSON 渲染。
pub(crate) struct HostProfile {
    /// `os-algorithm` 标识，与 [`host`] 的返回值一致。
    pub(crate) host: &'static str,
    /// 机型：登记基线时从本机读（macOS `sysctl -n machdep.cpu.brand_string`）。
    pub(crate) cpu: &'static str,
    /// 系统版本（macOS `sw_vers`）。
    pub(crate) os: &'static str,
    /// 采集基线时的工具链（`rustc --version`）；换编译器就重新登记。
    pub(crate) toolchain: &'static str,
}

/// 已登记的主机档案。
pub(crate) const HOSTS: &[HostProfile] = &[HostProfile {
    host: "macos-aarch64",
    cpu: "Apple M5 Pro",
    os: "macOS 27.0.1 (26A434)",
    toolchain: "rustc 1.100.0-nightly (0dfb098f3 2026-08-31)",
}];

/// 一档存储上的搬运上界：同机、同工作集规模下实测的最快搬法。
///
/// 三档不是猜出来的硬件缓存容量，而是三个**测量点**：16 KiB、1 MiB、64 MiB。场景按工作集
/// 落到最近的一档（见 [`tier`]），档与档之间留了四倍余量，因此「按 l1 判」意思是"工作集比
/// 16 KiB 的测量点小一个量级"，不是宣称这台机器的 L1 有 64 KiB。
pub(crate) struct Ceiling {
    pub(crate) host: &'static str,
    /// 档位：`l1` / `l2` / `dram`。
    pub(crate) tier: &'static str,
    /// 搬运方向：`read` / `write`。读、写上限不是同一个数（只读权重的 GEMV 可以跑过拷贝上限）。
    pub(crate) traffic: &'static str,
    /// criterion 分组与用例名，与 `src/memory.rs` 逐字一致。
    pub(crate) group: &'static str,
    pub(crate) benchmark: &'static str,
    /// 每迭代搬运的字节数，与基准里 `Throughput::Bytes` 同源。
    pub(crate) bytes: u64,
    /// 实测 ns/iter（criterion mean）。
    pub(crate) ns: f64,
    /// 允许的相对回退百分比。
    pub(crate) budget_percent: f64,
}

/// 各主机登记的三档读 / 写上界；数值来自 `cargo bench -p llm-benchmarks --bench mem_stream`。
pub(crate) const CEILINGS: &[Ceiling] = &[
    // Apple M5 Pro（2026-11-16）：16 KiB 读 64 趟 = 1 MiB / 迭代。
    Ceiling {
        host: "macos-aarch64",
        tier: "l1",
        traffic: "read",
        group: "mem_stream",
        benchmark: "l1_read_16kib",
        bytes: 64 * 16 * 1024,
        ns: 7_302.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-16）：16 KiB 写 64 趟 = 1 MiB / 迭代。
    Ceiling {
        host: "macos-aarch64",
        tier: "l1",
        traffic: "write",
        group: "mem_stream",
        benchmark: "l1_write_16kib",
        bytes: 64 * 16 * 1024,
        ns: 8_665.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-16）：1 MiB 读写各 8 趟 = 8 MiB / 迭代。
    Ceiling {
        host: "macos-aarch64",
        tier: "l2",
        traffic: "read",
        group: "mem_stream",
        benchmark: "l2_read_1mib",
        bytes: 8 * 1024 * 1024,
        ns: 64_824.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-16）：1 MiB 读写各 8 趟 = 8 MiB / 迭代。
    Ceiling {
        host: "macos-aarch64",
        tier: "l2",
        traffic: "write",
        group: "mem_stream",
        benchmark: "l2_write_1mib",
        bytes: 8 * 1024 * 1024,
        ns: 64_051.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-16）：64 MiB 读写各一趟。
    Ceiling {
        host: "macos-aarch64",
        tier: "dram",
        traffic: "read",
        group: "mem_stream",
        benchmark: "dram_read_64mib",
        bytes: 64 * 1024 * 1024,
        ns: 769_021.0,
        budget_percent: 10.0,
    },
    Ceiling {
        host: "macos-aarch64",
        tier: "dram",
        traffic: "write",
        group: "mem_stream",
        benchmark: "dram_write_64mib",
        bytes: 64 * 1024 * 1024,
        ns: 506_825.0,
        budget_percent: 10.0,
    },
];

/// 工作集不超过这一档的测量点时按 `l1` 判（四倍余量，见 [`Ceiling`]）。
pub(crate) const L1_LIMIT: u64 = 64 * 1024;

/// 工作集不超过这一档的测量点时按 `l2` 判。
pub(crate) const L2_LIMIT: u64 = 2 * 1024 * 1024;

/// 一秒等于多少纳秒：把字节 / ns 换算成 MB/s。
pub(crate) const NS_PER_SECOND: f64 = 1e9;

/// 一兆字节：把字节 / ns 换算成 MB/s。
pub(crate) const BYTES_PER_MB: f64 = 1e6;

/// 一吉字节（十进制）：把 MB/s 换算成 GB/s，填进记录的上界带宽字段。
pub(crate) const BYTES_PER_GB: f64 = 1e9;

/// 工作集落在哪一档：`l1` / `l2` / `dram`。
pub(crate) fn tier(working_set: u64) -> &'static str {
    if working_set <= L1_LIMIT {
        "l1"
    } else if working_set <= L2_LIMIT {
        "l2"
    } else {
        "dram"
    }
}

/// 本机标识：`os-algorithm`。criterion 的绝对值只在同一台机器上可比，标识也就到这一层。
pub(crate) fn host() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// 查某主机的档案；未登记返回 `None`。
pub(crate) fn profile_for(host: &str) -> Option<&'static HostProfile> {
    HOSTS.iter().find(|profile| profile.host == host)
}

/// 查某主机某后端某场景的登记基线；未登记返回 `None`。
pub(crate) fn baseline_for(host: &str, backend: &str, scenario: &str) -> Option<&'static Baseline> {
    BASELINES
        .iter()
        .find(|b| b.host == host && b.backend == backend && b.scenario == scenario)
}

/// 查某主机某档某方向的搬运上界；未登记返回 `None`。
pub(crate) fn ceiling_for(host: &str, tier: &str, traffic: &str) -> Option<&'static Ceiling> {
    CEILINGS
        .iter()
        .find(|c| c.host == host && c.tier == tier && c.traffic == traffic)
}

/// 由主机档案与后端标识构造记录的 [`Harness`]：algorithm 取运行环境，driver 填后端标识。
pub(crate) fn harness_of(host: &HostProfile, driver: &str) -> Harness {
    Harness::new(
        host.host,
        host.cpu,
        host.os,
        std::env::consts::ARCH,
        host.toolchain,
        driver,
    )
}

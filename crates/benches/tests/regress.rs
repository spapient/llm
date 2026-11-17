//! 把真实算子的实测值接进 `tool::regress` 的基线判定，并把「测到了什么」归档成一份可入 CI 的报告。
//!
//! 分工是刻意的：criterion 只负责测（预热、采样、区间估计），`tool::regress` 只负责判（方向、
//! 相对预算、"回退"与"改进"），`nn::Bounds` 只负责给出**下界**（这层至少要搬多少、算多少），
//! 本文件只负责**搬运**与**归档**。判定规则不在这里重写，代价公式也不在这里写第二份。
//!
//! # 三条"不可比"的处理
//!
//! - **跨机不可比**：基线按 `os-arch` 登记（见 [`baseline_for`]）。没有本机基线时跳过，而不是
//!   拿别的机器的数判——那会把"A 机的单核速度"当成"这台机器的验收线"。
//! - **跨口径不可比**：判的是 `ns/iter` 而不是 FLOPs 或 GiB/s。`gemv` 的二进制位宽随权重格式
//!   变（f32 / i4 / 打包），同一形状的理论算力能差 4 倍；时间才是可比的量。
//! - **跨版本不可比**：基线与实测必须在同一工具链下取；换了编译器就重新登记（档案里记了工具链）。
//!
//! # roofline 的分母也是测量
//!
//! 归档里除了实测，还有「这条基准最少该花多久」：`read` / `write` 两个方向分开由 `nn::Bounds`
//! 从算子语义推出，再除以**同机、同工作集档位**的实测搬运速率（[`CEILINGS`]，来自
//! `criterion/mem_stream.rs`）。不用厂商峰值：那只是规格，不是这台机器上测得到的速率
//! （`docs/lld.md` E6 把「仅有未验证的峰值规格」判为不可计算）。
//!
//! 于是报告能回答三个问题，而不只是一个 ns 数：这层卡在带宽上吗（`achieved_percent`）、卡在
//! 哪一侧（`bottleneck_side`）、离本机同口径上界还有多少余量。归档本身仍然只含整数，因此可以
//! 被仓库唯一的 JSON 解析器读回来自检。
//!
//! # 被测面与主机档案都写在表里
//!
//! 被测面在 [`SCENARIOS`] 里逐场景声明（criterion 分组 / 用例名与 `criterion/*.rs` 逐字一致，
//! 后端标识也是其中一列）；机型、系统、工具链在 [`HOSTS`] 里声明。新增一条基准或一个新后端都
//! 只是加一行，不复制判定流程；新增一台机器则连同上界一起登记，否则这份报告无法被别人复核。
//!
//! 归档在 `target/bench-report/<host>/<group>.json`，由 CI 的 bench job 收成 artifact。

use std::path::{Path, PathBuf};
use tool::regress::{Case, Direction, Judgement};

/// 一条回归基线场景：criterion 被测面 + 归档元数据。
///
/// 基线数值不写在结构里（基线随主机变化），由 [`baseline_for`] 按 `(host, backend, scenario)`
/// 查 [`BASELINES`]；结构只描述"测哪条基准、报告怎么写"。
struct Scenario {
    /// criterion 分组名（`target/criterion/<group>/...`）。
    group: &'static str,
    /// criterion 用例名（`.../<benchmark>/new/estimates.json`）。
    benchmark: &'static str,
    /// 场景名：`<算子>/<形状>`，同时是 regress `Case` 名与归档口径的一部分。
    scenario: &'static str,
    /// 后端标识：同一算子在不同后端上各占一行，基线按它区分（不是写死的常量）。
    backend: &'static str,
    /// `op` 注册表里的算子名：`nn::Bounds` 用它取语义表，roofline 的分子由它推出。
    op: &'static str,
    /// 形状（归档报告的 `shape` 字段）。
    shape: &'static [usize],
    /// 元素类型（归档报告的 `dtype` 字段）。
    dtype: &'static str,
    /// 批大小：1 行 = 一个 decode 步，与 prefill 的多行批差三个数量级。
    batch: u32,
    /// 交给 [`nn::Bounds::of_context`] 的 `(a, b)`：算子自己的语义表把它翻成具名轴。
    bounds: (usize, usize),
}

/// 全部接入基线判定的真实算子基准；分组 / 用例名与 `criterion/*.rs` 逐字一致。
///
/// 七个场景的后端都是 `cpu-reference`：它们调用的是 `op` 里的标量参考实现。加真实后端时
/// 只是在这张表里多一行同形状、不同 `backend` 的记录，判定与归档不动。
const SCENARIOS: &[Scenario] = &[
    Scenario {
        group: "gemv_f32",
        benchmark: "decode_1x4096x4096",
        scenario: "gemv_f32/decode_1x4096x4096",
        backend: "cpu-reference",
        op: "gemv_f32",
        shape: &[4096, 4096],
        dtype: "f32",
        batch: 1,
        bounds: (4096, 4096),
    },
    Scenario {
        group: "softmax_f32",
        benchmark: "decode_1x4096",
        scenario: "softmax_f32/decode_1x4096",
        backend: "cpu-reference",
        op: "softmax_f32",
        shape: &[4096],
        dtype: "f32",
        batch: 1,
        bounds: (4096, 1),
    },
    Scenario {
        group: "rope_f32",
        benchmark: "decode_1x4096",
        scenario: "rope_f32/decode_1x4096",
        backend: "cpu-reference",
        op: "rope_f32",
        shape: &[4096],
        dtype: "f32",
        batch: 1,
        bounds: (4096, 1),
    },
    Scenario {
        group: "rmsnorm_f32",
        benchmark: "decode_1x4096",
        scenario: "rmsnorm_f32/decode_1x4096",
        backend: "cpu-reference",
        op: "rmsnorm_f32",
        shape: &[4096],
        dtype: "f32",
        batch: 1,
        bounds: (4096, 1),
    },
    Scenario {
        group: "attention_f32",
        benchmark: "flash_decode_128x64",
        scenario: "attention_f32/flash_decode_128x64",
        backend: "cpu-reference",
        op: "attention_f32",
        shape: &[128, 64],
        dtype: "f32",
        batch: 1,
        bounds: (128, 64),
    },
    Scenario {
        group: "decode_attention_f32",
        benchmark: "naive_hist128x64",
        scenario: "decode_attention_f32/naive_hist128x64",
        backend: "cpu-reference",
        op: "decode_attention_f32",
        shape: &[128, 64],
        dtype: "f32",
        batch: 1,
        bounds: (1, 64),
    },
    Scenario {
        group: "decode_attention_f32",
        benchmark: "flash_hist128x64",
        scenario: "decode_attention_f32/flash_hist128x64",
        backend: "cpu-reference",
        op: "decode_flash_attention_f32",
        shape: &[128, 64],
        dtype: "f32",
        batch: 1,
        bounds: (1, 64),
    },
    Scenario {
        group: "gemm_f32",
        benchmark: "prefill_256x256_k8",
        scenario: "gemm_f32/prefill_256x256_k8",
        backend: "cpu-reference",
        op: "gemm_f32",
        shape: &[256, 256],
        dtype: "f32",
        batch: 1,
        bounds: (256, 256),
    },
    Scenario {
        group: "silu_f32",
        benchmark: "decode_1x4096",
        scenario: "silu_f32/decode_1x4096",
        backend: "cpu-reference",
        op: "silu_f32",
        shape: &[4096],
        dtype: "f32",
        batch: 1,
        bounds: (4096, 1),
    },
];

/// 一条登记基线：哪台机器、哪个后端、哪个场景、多少 ns、允许多少相对偏差。
///
/// 用结构而不是多元组，是因为三个字符串字段挨着放时，`(host, backend, scenario)` 的顺序错了
/// 编译照样通过。
struct Baseline {
    host: &'static str,
    backend: &'static str,
    scenario: &'static str,
    /// 基线 ns/iter：`cargo bench -p llm-benches --bench <group>` 的 mean 点估计，取整到 ns。
    ns: f64,
    /// 允许的相对回退百分比。
    budget_percent: f64,
}

/// 各主机登记的基线；数值来自真实测量，注释里写清机型与日期——"基线"必须能追溯到一次测量。
const BASELINES: &[Baseline] = &[
    // Apple M5 Pro（2026-11-15，reference 实现 BLOCK=32）：100 次采样。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "gemv_f32/decode_1x4096x4096",
        ns: 2_186_061.0,
        budget_percent: 5.0,
    },
    // Apple M5 Pro（2026-11-16，三趟标量参考）：噪声相对均值约 4.5%，预算取 10%。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "softmax_f32/decode_1x4096",
        ns: 15_492.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-16，成对旋转参考）：噪声相对均值约 6.4%，预算取 10%。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "rope_f32/decode_1x4096",
        ns: 9_487.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-16，两趟标量参考）：噪声相对均值约 1.5%，预算取 10%。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "rmsnorm_f32/decode_1x4096",
        ns: 2_374.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-16，online flash 参考 128×64）：噪声相对均值约 1.5%，预算取 10%。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "attention_f32/flash_decode_128x64",
        ns: 517_442.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-16，三重循环参考 256×256，K=8）：噪声相对均值约 1.5%，预算取 10%。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "gemm_f32/prefill_256x256_k8",
        ns: 16_599.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-16，逐元素 SiLU 参考）：噪声相对均值约 1.5%，预算取 10%。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "silu_f32/decode_1x4096",
        ns: 3_287.0,
        budget_percent: 10.0,
    },
];

/// 一台机器的档案：标识、机型、系统、工具链。
///
/// 机型写成数据而不是注释，是因为它必须随报告归档：上一轮的注释把 M5 Pro 写成了 M4 Pro，
/// 而 `os-arch` 完全看不出这个差别。没有档案的主机不判——跨机数字不可比。
struct HostProfile {
    /// `os-arch` 标识，与 [`host`] 的返回值一致。
    host: &'static str,
    /// 机型：登记基线时从本机读（macOS `sysctl -n machdep.cpu.brand_string`）。
    cpu: &'static str,
    /// 系统版本（macOS `sw_vers`）。
    os: &'static str,
    /// 采集基线时的工具链（`rustc --version`）；换编译器就重新登记。
    toolchain: &'static str,
}

/// 已登记的主机档案。
const HOSTS: &[HostProfile] = &[HostProfile {
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
struct Ceiling {
    host: &'static str,
    /// 档位：`l1` / `l2` / `dram`。
    tier: &'static str,
    /// 搬运方向：`read` / `write`。读、写上限不是同一个数（只读权重的 GEMV 可以跑过拷贝上限）。
    traffic: &'static str,
    /// criterion 分组与用例名，与 `criterion/mem_stream.rs` 逐字一致。
    group: &'static str,
    benchmark: &'static str,
    /// 每迭代搬运的字节数，与基准里 `Throughput::Bytes` 同源。
    bytes: u64,
    /// 实测 ns/iter（criterion mean）。
    ns: f64,
    /// 允许的相对回退百分比。
    budget_percent: f64,
}

/// 各主机登记的三档读 / 写上界；数值来自 `cargo bench -p llm-benches --bench mem_stream`。
const CEILINGS: &[Ceiling] = &[
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
const L1_LIMIT: u64 = 64 * 1024;

/// 工作集不超过这一档的测量点时按 `l2` 判。
const L2_LIMIT: u64 = 2 * 1024 * 1024;

/// 一秒等于多少纳秒：把字节 / ns 换算成 MB/s。
const NS_PER_SECOND: f64 = 1e9;

/// 一兆字节：把字节 / ns 换算成 MB/s。
const BYTES_PER_MB: f64 = 1e6;

/// 工作集落在哪一档：`l1` / `l2` / `dram`。
fn tier(working_set: u64) -> &'static str {
    if working_set <= L1_LIMIT {
        "l1"
    } else if working_set <= L2_LIMIT {
        "l2"
    } else {
        "dram"
    }
}

/// 本机标识：`os-arch`。criterion 的绝对值只在同一台机器上可比，标识也就到这一层。
fn host() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// 查某主机的档案；未登记返回 `None`。
fn profile_for(host: &str) -> Option<&'static HostProfile> {
    HOSTS.iter().find(|profile| profile.host == host)
}

/// 查某主机某后端某场景的登记基线；未登记返回 `None`。
fn baseline_for(host: &str, backend: &str, scenario: &str) -> Option<&'static Baseline> {
    BASELINES
        .iter()
        .find(|b| b.host == host && b.backend == backend && b.scenario == scenario)
}

/// 查某主机某档某方向的搬运上界；未登记返回 `None`。
fn ceiling_for(host: &str, tier: &str, traffic: &str) -> Option<&'static Ceiling> {
    CEILINGS
        .iter()
        .find(|c| c.host == host && c.tier == tier && c.traffic == traffic)
}

/// criterion 的 mean 点估计（`ns/iter`）。
///
/// 只依赖 `estimates.json` 里两个键名（`mean` 与 `point_estimate`）：criterion 的统计口径演进过，
/// 但这两个名字是它对外报告的核心。读不出来就明说"读不出"，不猜一个数。
fn mean_ns(text: &str) -> Option<f64> {
    let mean = text.find("\"mean\"")?;
    let key = text[mean..].find("\"point_estimate\"")?;
    let tail = &text[mean + key + "\"point_estimate\"".len()..];
    let start = tail.find(|c: char| c == '-' || c.is_ascii_digit())?;
    let number: String = tail[start..]
        .chars()
        .take_while(|c| {
            *c == '-' || *c == '.' || c.is_ascii_digit() || *c == 'e' || *c == '+' || *c == 'E'
        })
        .collect();
    number.parse().ok()
}

/// criterion 的输出目录：`target/criterion/<group>/<benchmark>/new/estimates.json`。
///
/// 目标目录在**工作区根**（`crates/benches/../..`），不在本成员的目录下：criterion 由
/// `cargo bench` 调用，写的是调用方的 `target/`。
fn estimates(manifest: &Path, group: &str, benchmark: &str) -> PathBuf {
    manifest
        .join("../..")
        .join("target/criterion")
        .join(group)
        .join(benchmark)
        .join("new/estimates.json")
}

/// 实测值；还没跑过基准就登记跳过并返回 `None`。
fn observed_ns(path: &Path) -> Option<f64> {
    let Ok(text) = std::fs::read_to_string(path) else {
        testkit::skip(format!(
            "benches: 还没有基准测量（{}），先跑 cargo bench -p llm-benches",
            path.display()
        ));
        return None;
    };
    Some(mean_ns(&text).unwrap_or_else(|| {
        panic!(
            "读不出 {} 的 mean.point_estimate：criterion 的输出格式变了，判定口径要跟着改",
            path.display()
        )
    }))
}

/// 搬 `bytes` 字节在该档上界下至少需要多少 ns。
fn bytes_in_ns(bytes: u64, ceiling: &Ceiling) -> f64 {
    bytes as f64 * ceiling.ns / ceiling.bytes as f64
}

/// 某档实测上界的 MB/s。
fn mbps(ceiling: &Ceiling) -> f64 {
    ceiling.bytes as f64 / ceiling.ns * NS_PER_SECOND / BYTES_PER_MB
}

/// 一条基准的 roofline 口径：下界流量、所落的存储档位、该档实测上界、达成率。
struct Roofline {
    /// 下界读入字节（`nn::Bounds`，由算子语义推出，不是手写常数）。
    read: u64,
    /// 下界写出字节。
    write: u64,
    /// 下界浮点运算次数。
    flops: f64,
    /// 工作集落在哪一档：`l1` / `l2` / `dram`。
    tier: &'static str,
    /// 该档实测读上界（MB/s）。
    read_ceiling_mbps: f64,
    /// 该档实测写上界（MB/s）。
    write_ceiling_mbps: f64,
    /// roofline 上界时间（ns）：两个方向各自换算后取慢的那一侧。
    ns: f64,
    /// 慢的那一侧：`read` / `write`。
    side: &'static str,
    /// 达成率（%）：上界时间 / 实测时间。超过 100 说明上界登记过期，不是"超越硬件"。
    achieved_percent: f64,
    /// 实测算力（MFLOP/s）：带宽达成率低时，这个数才说得清"代价花在算上"。
    mflops_per_s: f64,
}

impl Roofline {
    /// 由算子语义与本机上界推出一条基准的 roofline 口径。
    ///
    /// 工作集决定档位：16 KiB 量级的行宽算子按 `l1` 判，几百 KiB 的 `gemm` / `attention` 按
    /// `l2` 判，`gemv` 那 64 MiB 权重按 `dram` 判。档位定错会把"L1 里的往返"拿去比主存带宽，
    /// 达成率就没有意义。
    fn of(host: &HostProfile, scenario: &Scenario, observed_ns: f64) -> Self {
        let bounds = nn::Bounds::of_context(scenario.op, scenario.bounds.0, scenario.bounds.1)
            .unwrap_or_else(|error| {
                panic!(
                    "{}：算子 `{}` 推不出下界（{}），roofline 就没有分子",
                    scenario.scenario, scenario.op, error
                )
            });
        let tier = tier(bounds.bytes());
        let read = ceiling_for(host.host, tier, "read").unwrap_or_else(|| {
            panic!(
                "主机 `{}` 没有登记 {tier} 档的读上界：\
                 先跑 cargo bench -p llm-benches --bench mem_stream",
                host.host
            )
        });
        let write = ceiling_for(host.host, tier, "write").unwrap_or_else(|| {
            panic!(
                "主机 `{}` 没有登记 {tier} 档的写上界：\
                 先跑 cargo bench -p llm-benches --bench mem_stream",
                host.host
            )
        });

        let read_ns = bytes_in_ns(bounds.read(), read);
        let write_ns = bytes_in_ns(bounds.write(), write);
        let (ns, side) = if read_ns >= write_ns {
            (read_ns, "read")
        } else {
            (write_ns, "write")
        };

        Self {
            read: bounds.read(),
            write: bounds.write(),
            flops: bounds.flops(),
            tier,
            read_ceiling_mbps: mbps(read),
            write_ceiling_mbps: mbps(write),
            ns,
            side,
            achieved_percent: if observed_ns > 0.0 {
                ns / observed_ns * 100.0
            } else {
                0.0
            },
            mflops_per_s: if observed_ns > 0.0 {
                bounds.flops() / observed_ns * 1e3
            } else {
                0.0
            },
        }
    }

    /// 是否已经顶到带宽上界：85% 的门槛与 `docs/lld.md` E7 的带宽达成率同一口径。
    fn bandwidth_bound(&self) -> bool {
        self.achieved_percent >= 85.0
    }

    /// 瓶颈分类：只有顶到带宽上界才写 `bandwidth`，否则明确写 `not-bandwidth`。
    ///
    /// 不猜"是算力还是延迟"：那需要 prof 采样，达成率只够否定"带宽主导"。
    fn bottleneck(&self) -> &'static str {
        if self.bandwidth_bound() {
            "bandwidth"
        } else {
            "not-bandwidth"
        }
    }
}

/// 一份归档报告：元数据 + 实测 + 结论 + roofline 口径。
struct Report<'a> {
    scenario: &'a Scenario,
    host: &'a HostProfile,
    roofline: Roofline,
    verdict: &'static str,
    observed_ns: f64,
    baseline_ns: f64,
    budget_percent: f64,
}

/// 把形状渲染成 JSON 整数数组：`[4096, 4096]` / `[4096]`。
fn shape_json(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|dim| dim.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

impl Report<'_> {
    /// 渲染成整数数值的 JSON（见模块文档最后一段）。
    fn render(&self) -> String {
        let scenario = self.scenario;
        let roofline = &self.roofline;
        format!(
            "{{\n  \"surface\": \"{}\",\n  \"benchmark\": \"{}\",\n  \
             \"scenario\": \"{}\",\n  \"op\": \"{}\",\n  \"backend\": \"{}\",\n  \
             \"shape\": {},\n  \"batch\": {},\n  \"dtype\": \"{}\",\n  \
             \"read_bytes\": {},\n  \"write_bytes\": {},\n  \"working_set_bytes\": {},\n  \
             \"cache_tier\": \"{}\",\n  \"flops\": {},\n  \
             \"metric\": \"ns_per_iter\",\n  \"direction\": \"{}\",\n  \
             \"budget_percent\": {},\n  \"baseline_ns\": {},\n  \"observed_ns\": {},\n  \
             \"verdict\": \"{}\",\n  \
             \"host\": \"{}\",\n  \"cpu\": \"{}\",\n  \"os\": \"{}\",\n  \
             \"toolchain\": \"{}\",\n  \"arch\": \"{}\",\n  \
             \"read_ceiling_mbps\": {},\n  \"write_ceiling_mbps\": {},\n  \
             \"roofline_ns\": {},\n  \"bottleneck_side\": \"{}\",\n  \
             \"achieved_percent\": {},\n  \"mflops_per_s\": {},\n  \"bottleneck\": \"{}\",\n  \
             \"profile\": \"bench\"\n}}\n",
            scenario.group,
            scenario.benchmark,
            scenario.scenario,
            scenario.op,
            scenario.backend,
            shape_json(scenario.shape),
            scenario.batch,
            scenario.dtype,
            roofline.read,
            roofline.write,
            roofline.read + roofline.write,
            roofline.tier,
            roofline.flops.round() as i64,
            Direction::Lower.as_str(),
            self.budget_percent.round() as i64,
            self.baseline_ns.round() as i64,
            self.observed_ns.round() as i64,
            self.verdict,
            self.host.host,
            self.host.cpu,
            self.host.os,
            self.host.toolchain,
            std::env::consts::ARCH,
            roofline.read_ceiling_mbps.round() as i64,
            roofline.write_ceiling_mbps.round() as i64,
            roofline.ns.round() as i64,
            roofline.side,
            roofline.achieved_percent.round() as i64,
            roofline.mflops_per_s.round() as i64,
            roofline.bottleneck(),
        )
    }
}

/// 判定单个场景：建 `Case`、比对实测、归档报告；回退则 panic。
fn judge_one(manifest: &Path, host: &HostProfile, scenario: &Scenario) -> Option<()> {
    let baseline = baseline_for(host.host, scenario.backend, scenario.scenario)?;
    let observed = observed_ns(&estimates(manifest, scenario.group, scenario.benchmark))?;

    let case = Case::new(
        scenario.scenario,
        baseline.ns,
        baseline.budget_percent,
        Direction::Lower,
    )
    .expect("基线合法");
    let judgement = case.judge(observed);

    let dir = manifest
        .join("../..")
        .join("target/bench-report")
        .join(host.host);
    std::fs::create_dir_all(&dir).expect("归档目录可建");
    let report = Report {
        scenario,
        host,
        roofline: Roofline::of(host, scenario, observed),
        verdict: judgement.as_str(),
        observed_ns: observed,
        baseline_ns: baseline.ns,
        budget_percent: baseline.budget_percent,
    };
    let rendered = report.render();
    // 自检：归档必须是能被仓库唯一 JSON 解析器读懂的文档（整数约定见模块文档）。
    weights::json::parse(&rendered).expect("归档必须是合法 JSON");
    std::fs::write(dir.join(format!("{}.json", scenario.group)), &rendered).expect("归档可写");

    if judgement.is_pass() {
        return Some(());
    }
    panic!(
        "{} 回退：实测 {observed:.0} ns，基线 {:.0} ns，预算 {:.0}%（上限 {:.0} ns）。\
         先确认不是这台机器在忙；同一口径下确实回退了，要么修，要么按新测量重新登记基线",
        scenario.scenario,
        baseline.ns,
        baseline.budget_percent,
        case.limit()
    );
}

/// 一份上界归档：这一档的实测搬运速率本身也是一条被守护的基线。
struct CeilingReport<'a> {
    ceiling: &'a Ceiling,
    host: &'a HostProfile,
    verdict: &'static str,
    observed_ns: f64,
}

impl CeilingReport<'_> {
    /// 渲染成整数数值的 JSON。
    fn render(&self) -> String {
        let ceiling = self.ceiling;
        format!(
            "{{\n  \"surface\": \"{}\",\n  \"benchmark\": \"{}\",\n  \
             \"scenario\": \"mem_stream/{}_{}\",\n  \"kind\": \"ceiling\",\n  \
             \"tier\": \"{}\",\n  \"traffic\": \"{}\",\n  \"bytes_per_iter\": {},\n  \
             \"metric\": \"ns_per_iter\",\n  \"direction\": \"{}\",\n  \
             \"budget_percent\": {},\n  \"baseline_ns\": {},\n  \"observed_ns\": {},\n  \
             \"verdict\": \"{}\",\n  \
             \"host\": \"{}\",\n  \"cpu\": \"{}\",\n  \"os\": \"{}\",\n  \
             \"toolchain\": \"{}\",\n  \"mb_per_s\": {},\n  \"profile\": \"bench\"\n}}\n",
            ceiling.group,
            ceiling.benchmark,
            ceiling.tier,
            ceiling.traffic,
            ceiling.tier,
            ceiling.traffic,
            ceiling.bytes,
            Direction::Lower.as_str(),
            ceiling.budget_percent.round() as i64,
            ceiling.ns.round() as i64,
            self.observed_ns.round() as i64,
            self.verdict,
            self.host.host,
            self.host.cpu,
            self.host.os,
            self.host.toolchain,
            mbps(ceiling).round() as i64,
        )
    }
}

/// 判定单个上界：搬运速率掉了，说明机器或工具链变了，roofline 的分母必须重新登记。
fn judge_ceiling(manifest: &Path, host: &HostProfile, ceiling: &Ceiling) -> Option<()> {
    let observed = observed_ns(&estimates(manifest, ceiling.group, ceiling.benchmark))?;
    let case = Case::new(
        ceiling.benchmark,
        ceiling.ns,
        ceiling.budget_percent,
        Direction::Lower,
    )
    .expect("上界合法");
    let judgement = case.judge(observed);

    let dir = manifest
        .join("../..")
        .join("target/bench-report")
        .join(host.host);
    std::fs::create_dir_all(&dir).expect("归档目录可建");
    let rendered = CeilingReport {
        ceiling,
        host,
        verdict: judgement.as_str(),
        observed_ns: observed,
    }
    .render();
    weights::json::parse(&rendered).expect("归档必须是合法 JSON");
    let path = dir.join(format!(
        "{}_{}_{}.json",
        ceiling.group, ceiling.tier, ceiling.traffic
    ));
    std::fs::write(&path, &rendered).expect("归档可写");

    if judgement.is_pass() {
        return Some(());
    }
    panic!(
        "上界 {} 回退：实测 {observed:.0} ns，基线 {:.0} ns（上限 {:.0} ns）。\
         分母变了，所有达成率都要按新测量重算并重新登记",
        ceiling.benchmark,
        ceiling.ns,
        case.limit()
    );
}

/// 本机档案；有基线却没有档案时直接失败，而不是拿一个没有机型的 host 字符串判。
fn local_profile(host: &str) -> Option<&'static HostProfile> {
    let profile = profile_for(host);
    if profile.is_none() && BASELINES.iter().any(|b| b.host == host) {
        panic!(
            "主机 `{host}` 有登记基线却没有档案：补上机型、系统与工具链（见 HOSTS），\
             否则归档里的数字无法被复核"
        );
    }
    profile
}

#[test]
fn registered_baselines_stay_within_their_budget() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();
    let Some(profile) = local_profile(&host) else {
        testkit::skip(format!(
            "benches: 主机 `{host}` 没有建档（跨机绝对值不可比），先在靶机上跑 cargo bench 后建档"
        ));
        return;
    };

    let ready: Vec<&Scenario> = SCENARIOS
        .iter()
        .filter(|sc| baseline_for(&host, sc.backend, sc.scenario).is_some())
        .collect();
    if ready.is_empty() {
        testkit::skip(format!(
            "benches: 主机 `{host}` 没有登记任何基线，先在靶机上跑 cargo bench 后登记"
        ));
        return;
    }

    for scenario in ready {
        if judge_one(manifest, profile, scenario).is_none() {
            return;
        }
    }
}

#[test]
fn registered_ceilings_stay_within_their_budget() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();
    let Some(profile) = local_profile(&host) else {
        testkit::skip(format!(
            "benches: 主机 `{host}` 没有建档，上界无从判定（跨机绝对值不可比）"
        ));
        return;
    };

    let ready: Vec<&Ceiling> = CEILINGS.iter().filter(|c| c.host == host).collect();
    if ready.is_empty() {
        testkit::skip(format!(
            "benches: 主机 `{host}` 没有登记搬运上界，roofline 的达成率无法计算"
        ));
        return;
    }

    for ceiling in ready {
        if judge_ceiling(manifest, profile, ceiling).is_none() {
            return;
        }
    }
}

#[test]
fn every_registered_host_is_fully_described() {
    for baseline in BASELINES {
        assert!(
            profile_for(baseline.host).is_some(),
            "主机 `{}` 有基线却没有档案：机型、系统、工具链必须随数字一起归档",
            baseline.host
        );
    }
    for profile in HOSTS {
        for tier in ["l1", "l2", "dram"] {
            for traffic in ["read", "write"] {
                assert!(
                    ceiling_for(profile.host, tier, traffic).is_some(),
                    "主机 `{}` 缺少 {tier} 档的 {traffic} 上界",
                    profile.host
                );
            }
        }
    }
}

#[test]
fn the_report_carries_every_field_a_reader_needs_to_judge_it() {
    let host = &HOSTS[0];
    for scenario in SCENARIOS {
        let report = Report {
            scenario,
            host,
            roofline: Roofline::of(host, scenario, 2_000_000.0),
            verdict: Judgement::Improved {
                observed: 1.0,
                baseline: 2.0,
            }
            .as_str(),
            observed_ns: 1_500_000.0,
            baseline_ns: 2_000_000.0,
            budget_percent: 5.0,
        };
        let rendered = report.render();
        for field in [
            "\"surface\"",
            "\"shape\"",
            "\"dtype\"",
            "\"backend\"",
            "\"metric\"",
            "\"direction\"",
            "\"budget_percent\"",
            "\"baseline_ns\"",
            "\"observed_ns\"",
            "\"verdict\"",
            "\"host\"",
            "\"cpu\"",
            "\"os\"",
            "\"toolchain\"",
            "\"read_bytes\"",
            "\"write_bytes\"",
            "\"working_set_bytes\"",
            "\"cache_tier\"",
            "\"flops\"",
            "\"read_ceiling_mbps\"",
            "\"write_ceiling_mbps\"",
            "\"roofline_ns\"",
            "\"bottleneck_side\"",
            "\"achieved_percent\"",
            "\"mflops_per_s\"",
            "\"bottleneck\"",
        ] {
            assert!(
                rendered.contains(field),
                "报告缺少 {field}：读者无法自己判断结论"
            );
        }
        let parsed = weights::json::parse(&rendered).expect("合法 JSON");
        assert_eq!(
            parsed.get("batch").and_then(|value| value.integer().ok()),
            Some(scenario.batch as i64),
            "批大小是口径的一部分：1 行 = decode 步，与 prefill 的多行批差三个数量级"
        );
        assert_eq!(
            parsed
                .get("working_set_bytes")
                .and_then(|value| value.integer().ok()),
            Some((report.roofline.read + report.roofline.write) as i64),
            "工作集是档位的判据：写错档位，达成率就没有意义"
        );
    }
}

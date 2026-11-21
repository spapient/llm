//! 把真实算子的实测值接进 `tools::regress` 的基线判定，并把「测到了什么」归档成一份可入 CI 的报告。
//!
//! 分工是刻意的：criterion 只负责测（预热、采样、区间估计），`tools::regress` 只负责判（方向、
//! 相对预算、"回退"与"改进"），`nn::Bounds` 只负责给出**下界**（这层至少要搬多少、算多少），
//! 本文件只负责**搬运**与**归档**。判定规则不在这里重写，代价公式也不在这里写第二份。
//!
//! # 记录与分析件的分工
//!
//! 每条基准先构造一条 [`tool::bench::BenchRecord`]：身份（场景名 / 后端）、几何（形状 / 批次 /
//! 精度）、主机档案（[`Harness`]）、上界（[`RecordRoofline`]）与回填后的实测；达成率、瓶颈侧、
//! MB/s、flops、档位、verdict、基线与预算这些**本地分析件**仍由 Report 侧派生，不塞进记录。
//! [`Scenario`] 保留为 criterion 侧描述符：分组 / 用例名 / 算子 / 上下界这些测量定位信息不进
//! BenchRecord。未测得的整机峰值带宽 / 算力在记录里留 `None`，不拿标称值顶替。
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
use tool::bench::{BenchRecord, EntryKind, Geometry, Harness, Roofline as RecordRoofline};
use tool::regress::{Case, Direction, Judgement};

/// 一条回归基线场景：criterion 被测面 + 归档元数据。
///
/// 基线数值不写在结构里（基线随主机变化），由 [`baseline_for`] 按 `(host, backend, scenario)`
/// 查 [`BASELINES`]；结构只描述"测哪条基准、报告怎么写"。分组 / 用例名 / 算子 / 上下界是
/// criterion 侧的测量定位信息，不进 [`BenchRecord`]。
struct Scenario {
    /// criterion 分组名（`target/criterion/<group>/...`）。
    group: &'static str,
    /// criterion 用例名（`.../<benchmark>/new/estimates.json`）。
    benchmark: &'static str,
    /// 场景名：`<算子>/<形状>`，同时是 regress `Case` 名、记录名与归档口径的一部分。
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
    // 量化 GEMM 的分块执行路径：位流解一遍、再按列块整数累加（旧遍历逐元素重解位流）。
    Scenario {
        group: "gemm_quant",
        benchmark: "tiled_i2_256x256_k8",
        scenario: "gemm_i2/prefill_256x256_k8",
        backend: "cpu-reference",
        op: "gemm_i2",
        shape: &[256, 256],
        dtype: "u8",
        batch: 1,
        bounds: (256, 256),
    },
    Scenario {
        group: "gemm_quant",
        benchmark: "tiled_i4_256x256_k8",
        scenario: "gemm_i4/prefill_256x256_k8",
        backend: "cpu-reference",
        op: "gemm_i4",
        shape: &[256, 256],
        dtype: "u8",
        batch: 1,
        bounds: (256, 256),
    },
    Scenario {
        group: "gemm_quant",
        benchmark: "tiled_i8_256x256_k8",
        scenario: "gemm_i8/prefill_256x256_k8",
        backend: "cpu-reference",
        op: "gemm_i8",
        shape: &[256, 256],
        dtype: "u8",
        batch: 1,
        bounds: (256, 256),
    },
    Scenario {
        group: "matmul_f32_256x256_k4096",
        benchmark: "tiled",
        scenario: "matmul_f32/prefill_256x256_k4096_tiled",
        backend: "cpu-reference",
        op: "matmul_f32",
        shape: &[256, 256, 4096],
        dtype: "f32",
        batch: 1,
        bounds: (256, 256),
    },
    Scenario {
        group: "matmul_i8_256x256_k4096",
        benchmark: "tiled",
        scenario: "matmul_i8/prefill_256x256_k4096_tiled",
        backend: "cpu-reference",
        op: "matmul_i8",
        shape: &[256, 256, 4096],
        dtype: "i8",
        batch: 1,
        bounds: (256, 256),
    },
    Scenario {
        group: "attention_prefill_f32",
        benchmark: "tiled_1024x64",
        scenario: "attention_f32/prefill_1024x64_tiled",
        backend: "cpu-reference",
        op: "attention_f32",
        shape: &[1024, 64],
        dtype: "f32",
        batch: 1,
        bounds: (1024, 64),
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
    /// 基线 ns/iter：`cargo bench -p llm-benchmarks --bench <group>` 的 mean 点估计，取整到 ns。
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
    // Apple M5 Pro（2026-11-20，三位流解一遍 + 列块整数累加，20 次采样）：逐元素重解 494.6 µs → tile 50.7 µs（9.8×）。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "gemm_i2/prefill_256x256_k8",
        ns: 50_670.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-20，同一次采样）：逐元素重解 483.1 µs → tile 50.4 µs（9.6×）。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "gemm_i4/prefill_256x256_k8",
        ns: 50_420.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-20，同一次采样）：逐元素重解 133.1 µs → tile 48.3 µs（2.8×）；8 位每字节只有一个码，解包本来就便宜。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "gemm_i8/prefill_256x256_k8",
        ns: 48_330.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-20，tile 内核 256×256×4096，100 次采样）：scalar 307.33 ms → tiled 18.87 ms（16.3×）。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "matmul_f32/prefill_256x256_k4096_tiled",
        ns: 18_867_825.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-20，tile 内核 256×256×4096，100 次采样）：scalar 148.03 ms → tiled 20.76 ms（7.1×）。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "matmul_i8/prefill_256x256_k4096_tiled",
        ns: 20_760_328.0,
        budget_percent: 10.0,
    },
    // Apple M5 Pro（2026-11-20，分块 tile 1024×64，100 次采样）：online 18.40 ms → tiled 12.35 ms（1.49×）。
    Baseline {
        host: "macos-aarch64",
        backend: "cpu-reference",
        scenario: "attention_f32/prefill_1024x64_tiled",
        ns: 12_350_014.0,
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

/// 装载基准夹具镜像：条目数 / 每条元素数，与 `criterion/weights_load.rs` 同源。
///
/// 台账复制基准常量而不是反过来读夹具：基准二进制不被测试目标依赖，分组 / 用例名与形状
/// 同样逐字镜像（见 [`SCENARIOS`]）。
const LOAD_ENTRIES: usize = 8;
const LOAD_ELEMS: usize = 256;
/// 每条 F32 张量字节数（4 字节 / 元素）。
const LOAD_BYTES_FIRST: u64 = (LOAD_ELEMS * 4) as u64;
/// 数据段总字节数（全量装载）。
const LOAD_BYTES_ALL: u64 = LOAD_ENTRIES as u64 * LOAD_BYTES_FIRST;

/// 整模型一步基准夹具镜像：隐藏宽 / 输出宽，与 `criterion/model_decode.rs` 同源。
const MODEL_DIM: usize = 8;
const MODEL_OUT: usize = 8;

/// 一条装载基准场景：IO 路径 × 首条 / 全量 × 冷开 / 热复用。
///
/// 不进基线表：跨机绝对值不可比，记录只承载本机一次测量与身份（见模块文档「三条不可比」）。
struct LoadScenario {
    /// criterion 分组名。
    group: &'static str,
    /// criterion 用例名。
    benchmark: &'static str,
    /// 记录名：`<分组>/<用例>`。
    scenario: &'static str,
    /// IO 路径后端标识：文件流 / 映射。
    backend: &'static str,
    /// 形状（每条 F32 张量宽度）。
    shape: &'static [usize],
    /// 批大小：首条 1，全量为条目数。
    batch: u32,
    /// 元素精度。
    dtype: &'static str,
    /// 每迭代装载字节数，与基准 `Throughput::Bytes` 同源。
    bytes: u64,
}

/// 装载基准台账；分组 / 用例名与 `criterion/weights_load.rs` 逐字一致。
///
/// mmap 条目仅在 Unix 目标编译，与基准里的 `#[cfg(unix)]` 门控同源。
const LOAD_SCENARIOS: &[LoadScenario] = &[
    LoadScenario {
        group: "weights/load",
        benchmark: "read_all_reopen",
        scenario: "weights/load/read_all_reopen",
        backend: "file-stream",
        shape: &[LOAD_ELEMS],
        batch: LOAD_ENTRIES as u32,
        dtype: "f32",
        bytes: LOAD_BYTES_ALL,
    },
    LoadScenario {
        group: "weights/load",
        benchmark: "read_first_reopen",
        scenario: "weights/load/read_first_reopen",
        backend: "file-stream",
        shape: &[LOAD_ELEMS],
        batch: 1,
        dtype: "f32",
        bytes: LOAD_BYTES_FIRST,
    },
    LoadScenario {
        group: "weights/load",
        benchmark: "read_all_open",
        scenario: "weights/load/read_all_open",
        backend: "file-stream",
        shape: &[LOAD_ELEMS],
        batch: LOAD_ENTRIES as u32,
        dtype: "f32",
        bytes: LOAD_BYTES_ALL,
    },
    LoadScenario {
        group: "weights/load",
        benchmark: "read_first_open",
        scenario: "weights/load/read_first_open",
        backend: "file-stream",
        shape: &[LOAD_ELEMS],
        batch: 1,
        dtype: "f32",
        bytes: LOAD_BYTES_FIRST,
    },
    #[cfg(unix)]
    LoadScenario {
        group: "weights/load",
        benchmark: "mmap_all_reopen",
        scenario: "weights/load/mmap_all_reopen",
        backend: "mmap",
        shape: &[LOAD_ELEMS],
        batch: LOAD_ENTRIES as u32,
        dtype: "f32",
        bytes: LOAD_BYTES_ALL,
    },
    #[cfg(unix)]
    LoadScenario {
        group: "weights/load",
        benchmark: "mmap_first_reopen",
        scenario: "weights/load/mmap_first_reopen",
        backend: "mmap",
        shape: &[LOAD_ELEMS],
        batch: 1,
        dtype: "f32",
        bytes: LOAD_BYTES_FIRST,
    },
    #[cfg(unix)]
    LoadScenario {
        group: "weights/load",
        benchmark: "mmap_all_open",
        scenario: "weights/load/mmap_all_open",
        backend: "mmap",
        shape: &[LOAD_ELEMS],
        batch: LOAD_ENTRIES as u32,
        dtype: "f32",
        bytes: LOAD_BYTES_ALL,
    },
    #[cfg(unix)]
    LoadScenario {
        group: "weights/load",
        benchmark: "mmap_first_open",
        scenario: "weights/load/mmap_first_open",
        backend: "mmap",
        shape: &[LOAD_ELEMS],
        batch: 1,
        dtype: "f32",
        bytes: LOAD_BYTES_FIRST,
    },
];

/// 一条整模型一步基准场景：单序列 / 同批四条。
///
/// 不进基线表：整模型一步的 TPOT 与吞吐只承载本机一次测量，回退判定留给后续刀。
struct DecodeScenario {
    /// criterion 分组名。
    group: &'static str,
    /// criterion 用例名。
    benchmark: &'static str,
    /// 记录名：`<分组>/<用例>`。
    scenario: &'static str,
    /// 批大小：同批序列数。
    batch: u32,
    /// 每迭代产出 token 数，与基准 `Throughput::Elements` 同源。
    tokens: u64,
    /// 准入后页池 `in_use()` 可观测账目：每条提示预留一页（提示长 1、页宽 4）。
    pages: usize,
}

/// 整模型一步基准台账；分组 / 用例名与 `criterion/model_decode.rs` 逐字一致。
const DECODE_SCENARIOS: &[DecodeScenario] = &[
    DecodeScenario {
        group: "model_decode",
        benchmark: "single_step",
        scenario: "model_decode/single_step",
        batch: 1,
        tokens: 1,
        pages: 1,
    },
    DecodeScenario {
        group: "model_decode",
        benchmark: "batch4_step",
        scenario: "model_decode/batch4_step",
        batch: 4,
        tokens: 4,
        pages: 4,
    },
];

/// 一台机器的档案：标识、机型、系统、工具链。
///
/// 机型写成数据而不是注释，是因为它必须随报告归档：上一轮的注释把 M5 Pro 写成 M4 Pro，
/// 而 `os-arch` 完全看不出这个差别。没有档案的主机不判——跨机数字不可比。档案经构造器填进
/// [`Harness`]，本结构不再直接进 JSON 渲染。
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

/// 各主机登记的三档读 / 写上界；数值来自 `cargo bench -p llm-benchmarks --bench mem_stream`。
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

/// 一吉字节（十进制）：把 MB/s 换算成 GB/s，填进记录的上界带宽字段。
const BYTES_PER_GB: f64 = 1e9;

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
/// 目标目录在**工作区根**（`crates/benchmarks/../..`），不在本成员的目录下：criterion 由
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
            "benchmarks: 还没有基准测量（{}），先跑 cargo bench -p llm-benchmarks",
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

/// 由主机档案与后端标识构造记录的 [`Harness`]：arch 取运行环境，driver 填后端标识。
fn harness_of(host: &HostProfile, driver: &str) -> Harness {
    Harness::new(
        host.host,
        host.cpu,
        host.os,
        std::env::consts::ARCH,
        host.toolchain,
        driver,
    )
}

/// 由场景描述符构造几何（形状 / 批次 / 精度），构造自检失败属于登记数据错误，直接 panic。
fn geometry_of(scenario: &Scenario) -> Geometry {
    Geometry::new(scenario.shape.to_vec(), scenario.batch, scenario.dtype)
        .expect("场景几何合法：形状 / 批次 / 精度通过构造自检")
}

/// 构造一条算子基准的统一记录，并回填实测 ns/iter。
///
/// 整机峰值带宽 / 算力没有在本机测得：上界字段留 [`None`]，不拿标称值或他机数字顶替。分档
/// 搬运输送率属于 Report 侧的 roofline 分析件，不重复塞进记录。
fn record_for(host: &HostProfile, scenario: &Scenario, observed: f64) -> BenchRecord {
    let record = BenchRecord::new(
        EntryKind::Op,
        scenario.scenario,
        scenario.backend,
        geometry_of(scenario),
        harness_of(host, scenario.backend),
        RecordRoofline::default(),
    )
    .expect("算子记录合法：身份 / 几何 / 后端通过构造自检");
    record.with_observed(observed).expect("实测有限")
}

/// 构造一条上界基准的统一记录：实测搬运速率登记进峰值带宽字段，峰值算力未测得留 [`None`]。
fn ceiling_record(host: &HostProfile, ceiling: &Ceiling, observed: f64) -> BenchRecord {
    let geometry = Geometry::new(vec![ceiling.bytes as usize], 1, "u8")
        .expect("上界几何合法：每迭代字节数作为一维");
    // MB/s（十进制）→ GB/s：记录上界字段按 GB/s 口径。
    let peak_bw_gb_s = mbps(ceiling) * BYTES_PER_MB / BYTES_PER_GB;
    let roofline = RecordRoofline::new(Some(peak_bw_gb_s), None).expect("上界带宽非负有限");
    let name = format!("mem_stream/{}_{}", ceiling.tier, ceiling.traffic);
    let record = BenchRecord::new(
        EntryKind::Op,
        name,
        "cpu-reference",
        geometry,
        harness_of(host, "cpu-reference"),
        roofline,
    )
    .expect("上界记录合法：身份 / 几何 / 后端通过构造自检");
    record.with_observed(observed).expect("实测有限")
}

/// 构造一条装载基准的统一记录（算子级）。
///
/// 装载是算子级 IO 路径，记 [`EntryKind::Op`]；整机峰值带宽 / 算力未在本机测得，上界留
/// [`None`]，不拿标称值顶替。字节吞吐是 Report 侧由字节账目与实测 ns 派生的分析件，
/// 不塞进记录。
fn load_record(host: &HostProfile, scenario: &LoadScenario, observed: f64) -> BenchRecord {
    let geometry = Geometry::new(scenario.shape.to_vec(), scenario.batch, scenario.dtype)
        .expect("装载几何合法：形状 / 批次 / 精度通过构造自检");
    let record = BenchRecord::new(
        EntryKind::Op,
        scenario.scenario,
        scenario.backend,
        geometry,
        harness_of(host, scenario.backend),
        RecordRoofline::default(),
    )
    .expect("装载记录合法：身份 / 几何 / 后端通过构造自检");
    record.with_observed(observed).expect("实测有限")
}

/// 构造一条整模型一步基准的统一记录（模型级）。
///
/// 整模型一步记 [`EntryKind::Model`]；整机峰值 BW / GFLOPS 未测得，上界留 [`None`]。
/// tokens/s 与页数是 Report 侧账目 / 派生件，不塞进记录。
fn model_record(host: &HostProfile, scenario: &DecodeScenario, observed: f64) -> BenchRecord {
    let geometry = Geometry::new(vec![MODEL_DIM, MODEL_OUT], scenario.batch, "f32")
        .expect("模型几何合法：形状 / 批次 / 精度通过构造自检");
    let record = BenchRecord::new(
        EntryKind::Model,
        scenario.scenario,
        "cpu-reference",
        geometry,
        harness_of(host, "cpu-reference"),
        RecordRoofline::default(),
    )
    .expect("模型记录合法：身份 / 几何 / 后端通过构造自检");
    record.with_observed(observed).expect("实测有限")
}

/// 字节吞吐：bytes/iter ÷ ns/iter，十进制 GB/s（1 GB = 1e9 B）。
fn gb_per_s(bytes: u64, ns: f64) -> f64 {
    if ns > 0.0 {
        bytes as f64 / ns * NS_PER_SECOND / BYTES_PER_GB
    } else {
        0.0
    }
}

/// token 吞吐：tokens/iter ÷ ns/iter，tokens/s。
fn tokens_per_s(tokens: u64, ns: f64) -> f64 {
    if ns > 0.0 {
        tokens as f64 / ns * NS_PER_SECOND
    } else {
        0.0
    }
}

/// 一条基准的 roofline 口径：下界流量、所落的存储档位、该档实测上界、达成率。
///
/// 这是 Report 侧的本地分析件，不进 [`BenchRecord`]：达成率、瓶颈侧、MB/s、flops、档位都由
/// 算子语义与本机上界在此派生。
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
                 先跑 cargo bench -p llm-benchmarks --bench mem_stream",
                host.host
            )
        });
        let write = ceiling_for(host.host, tier, "write").unwrap_or_else(|| {
            panic!(
                "主机 `{}` 没有登记 {tier} 档的写上界：\
                 先跑 cargo bench -p llm-benchmarks --bench mem_stream",
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

/// 一份归档报告：统一记录（身份 / 几何 / 后端 / 上界 / 实测）+ 本地分析件 + 结论。
struct Report<'a> {
    scenario: &'a Scenario,
    record: BenchRecord,
    roofline: Roofline,
    verdict: &'static str,
    baseline_ns: f64,
    budget_percent: f64,
}

/// 把形状渲染成 JSON 整数数组：`[4096, 4096]` / `[4096]`。
fn shape_json(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|dim| dim.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// 归档 JSON 字段的标量取值：字符串加引号、整数按整数约定直出，`Raw` 是预渲染的 JSON 片段。
#[derive(Clone)]
enum Field {
    Str(String),
    Int(i64),
    Raw(String),
}

impl Field {
    /// 渲染成 JSON 右值（不含键名）。
    fn render(&self) -> String {
        match self {
            Field::Str(value) => format!("\"{value}\""),
            Field::Int(value) => value.to_string(),
            Field::Raw(value) => value.clone(),
        }
    }
}

/// 一份归档报告从记录侧取齐的公共取值：身份 / 几何 / 主机档案与实测。
///
/// 装载与模型两组报告共用这一层；各自独有的账目字段（字节吞吐 / tokens、页数）留在各自
/// 报告里，不进记录也不进这里。
struct ReportRecord<'a> {
    name: &'a str,
    kind: &'a str,
    backend: &'a str,
    shape: String,
    batch: u32,
    dtype: &'a str,
    observed: f64,
    host: &'a str,
    cpu: &'a str,
    os: &'a str,
    toolchain: &'a str,
    arch: &'a str,
}

impl<'a> ReportRecord<'a> {
    /// 从回填过实测的记录上一次取齐公共字段。
    fn from(record: &'a BenchRecord) -> Self {
        let geometry = record.geometry();
        let harness = record.harness();
        Self {
            name: record.name(),
            kind: record.kind().as_str(),
            backend: record.backend(),
            shape: shape_json(geometry.shape()),
            batch: geometry.batch(),
            dtype: geometry.dtype(),
            observed: record.observed().expect("记录已回填实测"),
            host: harness.host(),
            cpu: harness.cpu(),
            os: harness.os(),
            toolchain: harness.toolchain(),
            arch: harness.arch(),
        }
    }
}

/// 渲染一份无基线归档 JSON：公共身份 / 几何 / 主机字段 + 两组各自的账目字段。
///
/// `front` 插在 `dtype` 之后（每迭代账目），`tail` 插在 `observed_ns` 之后（派生速率）；
/// 键序沿用既有报告书写顺序，解析器按键名读回，不依赖顺序。
fn report_json(
    group: &str,
    benchmark: &str,
    meta: &ReportRecord<'_>,
    front: &[(&str, Field)],
    tail: &[(&str, Field)],
) -> String {
    let mut fields: Vec<(&str, Field)> = vec![
        ("surface", Field::Str(group.to_string())),
        ("benchmark", Field::Str(benchmark.to_string())),
        ("scenario", Field::Str(meta.name.to_string())),
        ("kind", Field::Str(meta.kind.to_string())),
        ("backend", Field::Str(meta.backend.to_string())),
        ("shape", Field::Raw(meta.shape.clone())),
        ("batch", Field::Int(meta.batch as i64)),
        ("dtype", Field::Str(meta.dtype.to_string())),
    ];
    for (key, value) in front {
        fields.push((key, value.clone()));
    }
    fields.push(("metric", Field::Str("ns_per_iter".to_string())));
    fields.push(("observed_ns", Field::Int(meta.observed.round() as i64)));
    for (key, value) in tail {
        fields.push((key, value.clone()));
    }
    fields.push(("host", Field::Str(meta.host.to_string())));
    fields.push(("cpu", Field::Str(meta.cpu.to_string())));
    fields.push(("os", Field::Str(meta.os.to_string())));
    fields.push(("toolchain", Field::Str(meta.toolchain.to_string())));
    fields.push(("arch", Field::Str(meta.arch.to_string())));
    fields.push(("profile", Field::Str("bench".to_string())));

    let inner = fields
        .iter()
        .map(|(key, value)| format!("  \"{key}\": {}", value.render()))
        .collect::<Vec<_>>()
        .join(",\n");
    format!("{{\n{inner}\n}}\n")
}

/// 一条记录对拍所期望的身份 / 几何 / 实测；主机档案取自 [`HostProfile`]。
struct RecordExpectation<'a> {
    name: &'a str,
    backend: &'a str,
    shape: &'a [usize],
    batch: u32,
    dtype: &'a str,
    observed: f64,
}

impl<'a> RecordExpectation<'a> {
    fn new(
        name: &'a str,
        backend: &'a str,
        shape: &'a [usize],
        batch: u32,
        dtype: &'a str,
        observed: f64,
    ) -> Self {
        Self {
            name,
            backend,
            shape,
            batch,
            dtype,
            observed,
        }
    }
}

/// 逐字段核对一条记录的身份 / 几何 / 主机档案与实测；屋顶上界按未测得留 [`None`]。
///
/// 族别（Op / Model）由调用方单独断言，其余字段三条归档口径一致；driver 与后端同值。
fn assert_record_fields(record: &BenchRecord, host: &HostProfile, expect: &RecordExpectation<'_>) {
    assert_eq!(record.name(), expect.name);
    assert_eq!(record.backend(), expect.backend);
    assert_eq!(record.geometry().shape(), expect.shape);
    assert_eq!(record.geometry().batch(), expect.batch);
    assert_eq!(record.geometry().dtype(), expect.dtype);
    assert_eq!(record.harness().host(), host.host);
    assert_eq!(record.harness().cpu(), host.cpu);
    assert_eq!(record.harness().os(), host.os);
    assert_eq!(record.harness().toolchain(), host.toolchain);
    assert_eq!(record.harness().driver(), expect.backend);
    assert!(record.roofline().peak_bw_gb_s().is_none());
    assert!(record.roofline().peak_gflops().is_none());
    assert_eq!(record.observed(), Some(expect.observed));
}

/// 核对一条算子级归档记录：族别 Op，身份 / 几何取自台账描述符，主机档案逐字段对上。
fn assert_op_record(
    host: &HostProfile,
    record: &BenchRecord,
    fields: (&str, &str, &[usize], u32, &str),
    observed: f64,
) {
    let (name, backend, shape, batch, dtype) = fields;
    assert_eq!(record.kind(), EntryKind::Op);
    assert_record_fields(
        record,
        host,
        &RecordExpectation::new(name, backend, shape, batch, dtype, observed),
    );
}

/// 断言归档 JSON 含全部读者必需字段；`what` 是报告类别前缀，进失败文案。
fn assert_present(rendered: &str, fields: &[&str], what: &str) {
    for field in fields {
        assert!(
            rendered.contains(field),
            "{what}报告缺少 {field}：读者无法自己判断结论"
        );
    }
}

/// 读回归档 JSON：必须能被仓库唯一解析器读懂（整数约定见模块文档）。
fn parsed_report(rendered: &str) -> weights::json::Value {
    weights::json::parse(rendered).expect("合法 JSON")
}

/// 取归档 JSON 里的批大小整数；键缺失或非整数返回 `None`。
fn parsed_batch(parsed: &weights::json::Value) -> Option<i64> {
    parsed.get("batch").and_then(|value| value.integer().ok())
}

impl Report<'_> {
    /// 渲染成整数数值的 JSON（见模块文档最后一段）。
    fn render(&self) -> String {
        let scenario = self.scenario;
        let record = &self.record;
        let roofline = &self.roofline;
        let harness = record.harness();
        let geometry = record.geometry();
        let observed = record.observed().expect("记录已回填实测");
        format!(
            "{{\n  \"surface\": \"{}\",\n  \"benchmark\": \"{}\",\n  \
             \"scenario\": \"{}\",\n  \"op\": \"{}\",\n  \"backend\": \"{}\",\n  \
             \"shape\": {},\n  \"batch\": {},\n  \"dtype\": \"{}\",\n  \
             \"read_bytes\": {},\n  \"write_bytes\": {},\n  \"working_set_bytes\": {},\n  \
             \"cache_tier\": \"{}\",\n  \"flops\": {},\n  \
             \"metric\": \"ns_per_iter\",\n  \"direction\": \"{}\",\n             \"budget_percent\": {},\n             \"baseline_ns\": {},\n             \"observed_ns\": {},\n             \"verdict\": \"{}\",\n             \"host\": \"{}\",\n             \"cpu\": \"{}\",\n             \"os\": \"{}\",\n             \"toolchain\": \"{}\",\n             \"arch\": \"{}\",\n             \"read_ceiling_mbps\": {},\n             \"write_ceiling_mbps\": {},\n             \"roofline_ns\": {},\n             \"bottleneck_side\": \"{}\",\n             \"achieved_percent\": {},\n             \"mflops_per_s\": {},\n             \"bottleneck\": \"{}\",\n             \
             \"profile\": \"bench\"\n}}\n",
            scenario.group,
            scenario.benchmark,
            record.name(),
            scenario.op,
            record.backend(),
            shape_json(geometry.shape()),
            geometry.batch(),
            geometry.dtype(),
            roofline.read,
            roofline.write,
            roofline.read + roofline.write,
            roofline.tier,
            roofline.flops.round() as i64,
            Direction::Lower.as_str(),
            self.budget_percent.round() as i64,
            self.baseline_ns.round() as i64,
            observed.round() as i64,
            self.verdict,
            harness.host(),
            harness.cpu(),
            harness.os(),
            harness.toolchain(),
            harness.arch(),
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
        record: record_for(host, scenario, observed),
        roofline: Roofline::of(host, scenario, observed),
        verdict: judgement.as_str(),
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

/// 一份上界归档：统一记录（含实测带宽上界）+ 结论。
struct CeilingReport<'a> {
    ceiling: &'a Ceiling,
    record: BenchRecord,
    verdict: &'static str,
}

impl CeilingReport<'_> {
    /// 渲染成整数数值的 JSON。
    fn render(&self) -> String {
        let ceiling = self.ceiling;
        let harness = self.record.harness();
        let observed = self.record.observed().expect("记录已回填实测");
        format!(
            "{{\n  \"surface\": \"{}\",\n  \"benchmark\": \"{}\",\n  \
             \"scenario\": \"mem_stream/{}_{}\",\n  \"kind\": \"ceiling\",\n  \
             \"tier\": \"{}\",\n  \"traffic\": \"{}\",\n             \"bytes_per_iter\": {},\n  \
             \"metric\": \"ns_per_iter\",\n  \"direction\": \"{}\",\n             \"budget_percent\": {},\n             \"baseline_ns\": {},\n             \"observed_ns\": {},\n             \"verdict\": \"{}\",\n             \"host\": \"{}\",\n             \"cpu\": \"{}\",\n             \"os\": \"{}\",\n             \"toolchain\": \"{}\",\n             \"mb_per_s\": {},\n             \"profile\": \"bench\"\n}}\n",
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
            observed.round() as i64,
            self.verdict,
            harness.host(),
            harness.cpu(),
            harness.os(),
            harness.toolchain(),
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
        record: ceiling_record(host, ceiling, observed),
        verdict: judgement.as_str(),
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

/// 一份装载归档：统一记录 + 字节账目；无基线、无判定结论（跨机绝对值不可比）。
struct LoadReport<'a> {
    scenario: &'a LoadScenario,
    record: BenchRecord,
}

impl LoadReport<'_> {
    /// 渲染成整数数值的 JSON（整数约定见 [`Report`]）。
    fn render(&self) -> String {
        let scenario = self.scenario;
        let meta = ReportRecord::from(&self.record);
        let bytes = scenario.bytes;
        report_json(
            scenario.group,
            scenario.benchmark,
            &meta,
            &[("bytes_per_iter", Field::Int(bytes as i64))],
            &[(
                "gb_per_s",
                Field::Int(gb_per_s(bytes, meta.observed).round() as i64),
            )],
        )
    }
}

/// 一份整模型一步归档：统一记录 + token 与页数账目；无基线、无判定结论。
struct DecodeReport<'a> {
    scenario: &'a DecodeScenario,
    record: BenchRecord,
}

impl DecodeReport<'_> {
    /// 渲染成整数数值的 JSON（整数约定见 [`Report`]）。
    fn render(&self) -> String {
        let scenario = self.scenario;
        let meta = ReportRecord::from(&self.record);
        let tokens = scenario.tokens;
        let pages = scenario.pages;
        report_json(
            scenario.group,
            scenario.benchmark,
            &meta,
            &[
                ("tokens_per_iter", Field::Int(tokens as i64)),
                ("pages_in_use", Field::Int(pages as i64)),
            ],
            &[(
                "tokens_per_s",
                Field::Int(tokens_per_s(tokens, meta.observed).round() as i64),
            )],
        )
    }
}

/// 归档一条无基线基准：读本机实测，建记录，写归档 JSON；没有测量则跳过。
///
/// `make` 用实测 ns 构造记录，`render` 把记录渲染成归档报告，`file` 是归档文件名；
/// 装载与模型两组只在记录构造、独有账目字段与文件名上不同。
fn archive_record(
    manifest: &Path,
    host: &HostProfile,
    group: &str,
    benchmark: &str,
    file: String,
    make: impl FnOnce(&HostProfile, f64) -> BenchRecord,
    render: impl FnOnce(BenchRecord) -> String,
) -> Option<()> {
    let observed = observed_ns(&estimates(manifest, group, benchmark))?;
    let record = make(host, observed);
    let dir = manifest
        .join("../..")
        .join("target/bench-report")
        .join(host.host);
    std::fs::create_dir_all(&dir).expect("归档目录可建");
    let rendered = render(record);
    weights::json::parse(&rendered).expect("归档必须是合法 JSON");
    std::fs::write(dir.join(file), rendered).expect("归档可写");
    Some(())
}

/// 归档一条装载基准：读本机实测，建记录，写归档 JSON；没有测量则跳过。
fn archive_load(manifest: &Path, host: &HostProfile, scenario: &LoadScenario) -> Option<()> {
    archive_record(
        manifest,
        host,
        scenario.group,
        scenario.benchmark,
        format!("weights_load_{}.json", scenario.benchmark),
        |host, observed| load_record(host, scenario, observed),
        |record| LoadReport { scenario, record }.render(),
    )
}

/// 归档一条整模型一步基准：读本机实测，建记录，写归档 JSON；没有测量则跳过。
fn archive_model(manifest: &Path, host: &HostProfile, scenario: &DecodeScenario) -> Option<()> {
    archive_record(
        manifest,
        host,
        scenario.group,
        scenario.benchmark,
        format!("model_decode_{}.json", scenario.benchmark),
        |host, observed| model_record(host, scenario, observed),
        |record| DecodeReport { scenario, record }.render(),
    )
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
            "benchmarks: 主机 `{host}` 没有建档（跨机绝对值不可比），先在靶机上跑 cargo bench 后建档"
        ));
        return;
    };

    let ready: Vec<&Scenario> = SCENARIOS
        .iter()
        .filter(|sc| baseline_for(&host, sc.backend, sc.scenario).is_some())
        .collect();
    if ready.is_empty() {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有登记任何基线，先在靶机上跑 cargo bench 后登记"
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
            "benchmarks: 主机 `{host}` 没有建档，上界无从判定（跨机绝对值不可比）"
        ));
        return;
    };

    let ready: Vec<&Ceiling> = CEILINGS.iter().filter(|c| c.host == host).collect();
    if ready.is_empty() {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有登记搬运上界，roofline 的达成率无法计算"
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
        let record = record_for(host, scenario, 1_500_000.0);
        assert_op_record(
            host,
            &record,
            (
                scenario.scenario,
                scenario.backend,
                scenario.shape,
                scenario.batch,
                scenario.dtype,
            ),
            1_500_000.0,
        );

        let report = Report {
            scenario,
            record,
            roofline: Roofline::of(host, scenario, 2_000_000.0),
            verdict: Judgement::Improved {
                observed: 1.0,
                baseline: 2.0,
            }
            .as_str(),
            baseline_ns: 2_000_000.0,
            budget_percent: 5.0,
        };
        let rendered = report.render();
        assert_present(
            &rendered,
            &[
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
            ],
            "",
        );
        let parsed = parsed_report(&rendered);
        assert_eq!(
            parsed_batch(&parsed),
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

#[test]
fn registered_load_records_archive_when_measured() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();
    let Some(profile) = local_profile(&host) else {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有建档，装载记录无从归档（跨机绝对值不可比）"
        ));
        return;
    };

    for scenario in LOAD_SCENARIOS {
        if archive_load(manifest, profile, scenario).is_none() {
            return;
        }
    }
}

#[test]
fn load_records_carry_every_field_a_reader_needs_to_judge_it() {
    let host = &HOSTS[0];
    let expected = if cfg!(unix) { 8 } else { 4 };
    assert_eq!(
        LOAD_SCENARIOS.len(),
        expected,
        "装载台账条目数与基准用例逐条对齐：File 四条 + mmap 四条（仅 Unix）"
    );
    for scenario in LOAD_SCENARIOS {
        let record = load_record(host, scenario, 1_500.0);
        assert_op_record(
            host,
            &record,
            (
                scenario.scenario,
                scenario.backend,
                scenario.shape,
                scenario.batch,
                scenario.dtype,
            ),
            1_500.0,
        );

        let report = LoadReport { scenario, record };
        let rendered = report.render();
        assert_present(
            &rendered,
            &[
                "\"surface\"",
                "\"shape\"",
                "\"dtype\"",
                "\"backend\"",
                "\"metric\"",
                "\"bytes_per_iter\"",
                "\"observed_ns\"",
                "\"gb_per_s\"",
                "\"host\"",
                "\"cpu\"",
                "\"os\"",
                "\"toolchain\"",
                "\"arch\"",
            ],
            "装载",
        );
        let parsed = parsed_report(&rendered);
        assert_eq!(parsed_batch(&parsed), Some(scenario.batch as i64));
        assert_eq!(
            parsed
                .get("bytes_per_iter")
                .and_then(|value| value.integer().ok()),
            Some(scenario.bytes as i64),
            "字节账目与基准 Throughput::Bytes 同源"
        );
    }
}

#[test]
fn registered_model_records_archive_when_measured() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = host();
    let Some(profile) = local_profile(&host) else {
        testkit::skip(format!(
            "benchmarks: 主机 `{host}` 没有建档，模型记录无从归档（跨机绝对值不可比）"
        ));
        return;
    };

    for scenario in DECODE_SCENARIOS {
        if archive_model(manifest, profile, scenario).is_none() {
            return;
        }
    }
}

#[test]
fn model_records_carry_every_field_a_reader_needs_to_judge_it() {
    let host = &HOSTS[0];
    assert_eq!(DECODE_SCENARIOS.len(), 2, "单序列与同批四条各一条");
    for scenario in DECODE_SCENARIOS {
        let record = model_record(host, scenario, 1_500_000.0);
        // 记录身份 / 几何 / 后端必须与场景描述符逐字段对上，避免归档元数据串台。
        assert_eq!(record.kind(), EntryKind::Model);
        assert_record_fields(
            &record,
            host,
            &RecordExpectation::new(
                scenario.scenario,
                "cpu-reference",
                &[MODEL_DIM, MODEL_OUT],
                scenario.batch,
                "f32",
                1_500_000.0,
            ),
        );

        let report = DecodeReport { scenario, record };
        let rendered = report.render();
        assert_present(
            &rendered,
            &[
                "\"surface\"",
                "\"shape\"",
                "\"dtype\"",
                "\"backend\"",
                "\"metric\"",
                "\"tokens_per_iter\"",
                "\"pages_in_use\"",
                "\"observed_ns\"",
                "\"tokens_per_s\"",
                "\"host\"",
                "\"cpu\"",
                "\"os\"",
                "\"toolchain\"",
                "\"arch\"",
            ],
            "模型",
        );
        let parsed = parsed_report(&rendered);
        assert_eq!(
            parsed_batch(&parsed),
            Some(scenario.batch as i64),
            "批大小是口径的一部分：1 行 = decode 步，与 prefill 的多行批差三个数量级"
        );
        assert_eq!(
            parsed
                .get("tokens_per_iter")
                .and_then(|value| value.integer().ok()),
            Some(scenario.tokens as i64)
        );
        assert_eq!(
            parsed
                .get("pages_in_use")
                .and_then(|value| value.integer().ok()),
            Some(scenario.pages as i64),
            "页数是页池 in_use() 的可观测账目，不虚构峰值字节"
        );
    }
}

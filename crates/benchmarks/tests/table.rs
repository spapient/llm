//! 登记表：场景 / 基线 / 负载与解码场景 / 主机档案 / 上界。

/// 一条回归基线场景：criterion 被测面 + 归档元数据。
///
/// 基线数值不写在结构里（基线随主机变化），由 [`baseline_for`] 按 `(host, backend, scenario)`
/// 查 [`BASELINES`]；结构只描述"测哪条基准、报告怎么写"。分组 / 用例名 / 算子 / 上下界是
/// criterion 侧的测量定位信息，不进 [`BenchRecord`]。
pub(crate) struct Scenario {
    /// criterion 分组名（`target/criterion/<group>/...`）。
    pub(crate) group: &'static str,
    /// criterion 用例名（`.../<benchmark>/new/estimates.json`）。
    pub(crate) benchmark: &'static str,
    /// 场景名：`<算子>/<形状>`，同时是 regress `Case` 名、记录名与归档口径的一部分。
    pub(crate) scenario: &'static str,
    /// 后端标识：同一算子在不同后端上各占一行，基线按它区分（不是写死的常量）。
    pub(crate) backend: &'static str,
    /// `op` 注册表里的算子名：`nn::Bounds` 用它取语义表，roofline 的分子由它推出。
    pub(crate) op: &'static str,
    /// 形状（归档报告的 `shape` 字段）。
    pub(crate) shape: &'static [usize],
    /// 元素类型（归档报告的 `dtype` 字段）。
    pub(crate) dtype: &'static str,
    /// 批大小：1 行 = 一个 decode 步，与 prefill 的多行批差三个数量级。
    pub(crate) batch: u32,
    /// 交给 [`nn::Bounds::of_context`] 的 `(a, b)`：算子自己的语义表把它翻成具名轴。
    pub(crate) bounds: (usize, usize),
}

/// 全部接入基线判定的真实算子基准；分组 / 用例名与 `criterion/*.rs` 逐字一致。
///
/// 七个场景的后端都是 `cpu-reference`：它们调用的是 `op` 里的标量参考实现。加真实后端时
/// 只是在这张表里多一行同形状、不同 `backend` 的记录，判定与归档不动。
pub(crate) const SCENARIOS: &[Scenario] = &[
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
pub(crate) struct Baseline {
    pub(crate) host: &'static str,
    pub(crate) backend: &'static str,
    pub(crate) scenario: &'static str,
    /// 基线 ns/iter：`cargo bench -p llm-benchmarks --bench <group>` 的 mean 点估计，取整到 ns。
    pub(crate) ns: f64,
    /// 允许的相对回退百分比。
    pub(crate) budget_percent: f64,
}

/// 各主机登记的基线；数值来自真实测量，注释里写清机型与日期——"基线"必须能追溯到一次测量。
pub(crate) const BASELINES: &[Baseline] = &[
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

/// 装载基准夹具镜像：条目数 / 每条元素数，与 `criterion/weights.rs` 同源。
///
/// 台账复制基准常量而不是反过来读夹具：基准二进制不被测试目标依赖，分组 / 用例名与形状
/// 同样逐字镜像（见 [`SCENARIOS`]）。
pub(crate) const LOAD_ENTRIES: usize = 8;
pub(crate) const LOAD_ELEMS: usize = 256;
/// 每条 F32 张量字节数（4 字节 / 元素）。
pub(crate) const LOAD_BYTES_FIRST: u64 = (LOAD_ELEMS * 4) as u64;
/// 数据段总字节数（全量装载）。
pub(crate) const LOAD_BYTES_ALL: u64 = LOAD_ENTRIES as u64 * LOAD_BYTES_FIRST;

/// 整模型一步基准夹具镜像：隐藏宽 / 输出宽，与 `criterion/model.rs` 同源。
pub(crate) const MODEL_DIM: usize = 8;
pub(crate) const MODEL_OUT: usize = 8;

/// 一条装载基准场景：IO 路径 × 首条 / 全量 × 冷开 / 热复用。
///
/// 不进基线表：跨机绝对值不可比，记录只承载本机一次测量与身份（见模块文档「三条不可比」）。
pub(crate) struct LoadScenario {
    /// criterion 分组名。
    pub(crate) group: &'static str,
    /// criterion 用例名。
    pub(crate) benchmark: &'static str,
    /// 记录名：`<分组>/<用例>`。
    pub(crate) scenario: &'static str,
    /// IO 路径后端标识：文件流 / 映射。
    pub(crate) backend: &'static str,
    /// 形状（每条 F32 张量宽度）。
    pub(crate) shape: &'static [usize],
    /// 批大小：首条 1，全量为条目数。
    pub(crate) batch: u32,
    /// 元素精度。
    pub(crate) dtype: &'static str,
    /// 每迭代装载字节数，与基准 `Throughput::Bytes` 同源。
    pub(crate) bytes: u64,
}

/// 装载基准台账；分组 / 用例名与 `criterion/weights.rs` 逐字一致。
///
/// mmap 条目仅在 Unix 目标编译，与基准里的 `#[cfg(unix)]` 门控同源。
pub(crate) const LOAD_SCENARIOS: &[LoadScenario] = &[
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
pub(crate) struct DecodeScenario {
    /// criterion 分组名。
    pub(crate) group: &'static str,
    /// criterion 用例名。
    pub(crate) benchmark: &'static str,
    /// 记录名：`<分组>/<用例>`。
    pub(crate) scenario: &'static str,
    /// 批大小：同批序列数。
    pub(crate) batch: u32,
    /// 每迭代产出 token 数，与基准 `Throughput::Elements` 同源。
    pub(crate) tokens: u64,
    /// 准入后页池 `in_use()` 可观测账目：每条提示预留一页（提示长 1、页宽 4）。
    pub(crate) pages: usize,
}

/// 整模型一步基准台账；分组 / 用例名与 `criterion/model.rs` 逐字一致。
pub(crate) const DECODE_SCENARIOS: &[DecodeScenario] = &[
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

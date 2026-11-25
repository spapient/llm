//! 主机原生微内核（`prof::native`）：设备无关的 CPU 高吞吐内核与它们的协商入口。
//!
//! `prof` 是抽象隔离层（DSL 侧）：它把「在什么设备上、用哪条微内核算一块 GEMM」收成
//! 一张**编译期配置表** `GEMM_KERNELS` 加一个协商函数 `gemm_negotiate`。调用方（`nn` 的
//! GEMM 骨架、未来的 `cutile` tile DSL IR）只问一句「这块谁能算」，不必写任何
//! `cfg(target_arch)` / `cfg(nightly)` / 特性探测——那些门全部收在本模块里。
//!
//! FMA、dotprod 与 SME 算法统一住在 `ops/gemm_micro.k`，这里只保留公开兼容面、
//! 面板缓存和协商表；调用方路径不变，运行时特性与 ABI 全由安全桥收口。

use std::thread;

/// f32 8×8 微内核的兼容公开面；算法由 `gemm_micro.k` 静态链接。
pub mod gemm8x8 {
    /// 一个待算的 8×8 微内核块。
    #[derive(Clone, Copy)]
    pub struct Tile {
        /// 块左上角行。
        pub row0: usize,
        /// 块左上角列。
        pub j: usize,
        /// 矩阵列宽。
        pub n: usize,
        /// 本块内积分长度。
        pub kb: usize,
        /// 首块从零累加，否则读回输出续算。
        pub first: bool,
    }

    /// 保持原 `void` 调用面；非法缓冲与既有 slice 入口一样触发断言。
    pub fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
        assert!(
            crate::runtime::gemm_micro::fma_micro(
                panel, b, out, (tile.row0, tile.j, tile.n, tile.kb), tile.first,
            ).is_ok(),
            "FMA 微内核的面板或输出缓冲不足"
        );
    }

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    pub use crate::runtime::gemm_micro::{port_fma, port_fma_block};

    /// FMA 探针每 K 步的 MAC：由配置几何派生。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    pub const FMA_MAC_PER_K: usize =
        crate::runtime::gemm_micro::FMA_MR * crate::runtime::gemm_micro::FMA_NR;
}

/// i8 `sdot` 4×4 微内核的兼容公开面；无 dotprod 时仍返回 `false`。
pub mod i8 {
    /// 一个 sdot 4 列块的坐标与 K 向量段。
    #[derive(Clone, Copy)]
    pub struct Gram {
        /// 块起点行。
        pub r: usize,
        /// 块起点列。
        pub blk: usize,
        /// 矩阵列宽。
        pub n: usize,
        /// 内积长度。
        pub k: usize,
        /// 向量化段长度，必须是 16 的非零倍数才进入 sdot。
        pub kvec: usize,
    }

    /// 保持原布尔拒绝语义；非法长度或溢出同样拒绝，不进入 C ABI。
    pub fn dot_block(a: &[i8], bpack: &[i8], out: &mut [i32], gram: Gram) -> bool {
        crate::runtime::gemm_micro::sdot_micro(
            a, bpack, out, (gram.r, gram.blk, gram.n, gram.k, gram.kvec),
        ).unwrap_or(false)
    }

    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    pub use crate::runtime::gemm_micro::{port_sdot, port_sdot_block};

    /// 纯 sdot 探针一轮的 MAC：配置链数 × 16。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    pub const SDOT_MAC_PER_ROUND: usize = crate::runtime::gemm_micro::SDOT_CHAINS * 16;

    /// 取数探针每 K 步的 MAC：由 4×4 配置几何派生。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    pub const SDOT_BLOCK_MAC_PER_K: usize =
        crate::runtime::gemm_micro::SDOT_MR * crate::runtime::gemm_micro::SDOT_NR;
}

/// Apple SME 16×64 微内核：C 函数级目标特性使 stable 与 nightly 共用同一入口。
#[cfg(target_arch = "aarch64")]
pub mod sme {
    pub use super::sme_geo::{MIN_ELEMS, MR, NR};

    /// 整铺 SME GEMM；形状、缓冲或设备能力不满足时返回 `false`。
    pub fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) -> bool {
        crate::runtime::gemm_micro::sme_gemm(a, b, m, n, k, out).unwrap_or(false)
    }

    /// 线程上限同时受行块、列组和用户配置约束。
    pub fn threads_for(m: usize, n: usize) -> usize {
        crate::runtime::gemm_micro::sme_threads(m, n)
    }

    /// 测量容器复用生产 SME 外积核心。
    #[cfg(feature = "bench")]
    pub use crate::runtime::gemm_micro::{port_mix, port_run};
}

/// f32 NEON 寄存器瓦片几何（8×8 块）——编译期常量的单一事实源。
pub mod f32 {
    /// 微内核行数，来自 `gemm_micro.fma` 配置。
    pub const MR: usize = crate::runtime::gemm_micro::FMA_MR;
    /// 微内核列数，来自同一配置。
    pub const NR: usize = crate::runtime::gemm_micro::FMA_NR;
    /// K 方向内积分块宽度：K=4096 时按 256 切块以驻留 L1/L2。
    pub const KB: usize = crate::runtime::gemm_micro::FMA_KB;
}

/// i8 sdot 寄存器瓦片几何：配置常量的单一事实源，瓦片本体在 `i8`。
pub mod i8_geo {
    /// 微内核行数：四行共享同一 B 向量，摊薄 B 加载。
    pub const MR: usize = crate::runtime::gemm_micro::SDOT_MR;
    /// 微内核列数：4 个 int32x4 累加器 × 4 行 = 16 寄存器，避开溢出。
    pub const NR: usize = crate::runtime::gemm_micro::SDOT_NR;
}

/// Apple SME ZA 外积瓦片几何：数值只来自 `[gemm_micro.sme]`。
pub mod sme_geo {
    /// 微内核行数：一个 16×16 ZA 瓦片覆盖 16 行。
    pub const MR: usize = crate::runtime::gemm_micro::SME_MR;
    /// 微内核列数：4 个 ZA 瓦片各 16 列，一次写回 64 列。
    pub const NR: usize = crate::runtime::gemm_micro::SME_NR;
    /// 规模门槛（元素数）：低于此值 `smstart`/`smstop` 与打包固定成本超过外积收益，交回 NEON。
    pub const MIN_ELEMS: usize = crate::runtime::gemm_micro::SME_MIN_ELEMS;
}

// 打包面板的暂存区：按线程复用，用完即还。
//
// 打包出来的是「整段 B 面板」，`n = k = 4096` 时是 `4096 × 4096 × 4 B = 67 MB`。每次调用现分配，
// 光 `calloc` 与首次触页就要几十毫秒，而 decode 与 prefill 都是**同一线程反复调**同一个几何，
// 这块内存每步都用得上，却每步被丢掉一次。这里让每个调用线程留一块只增不减的缓冲：
//
// - **只按线程共享，不跨线程**：并发调用各占一个线程私有的缓冲，不存在两次调用同时写同一块；
//   池里的 worker 只是拿到主线程的裸指针去写自己的分片，收回发生在 [`with_panels`] 返回之前。
// - **只扩容，不缩小**：几何变了就扩容一次，之后小几何沿用它前面的部分。常驻量等于本线程见过的
//   最大面板，与「每次分配再释放」的峰值一致（面板本来就要整块驻留）。
// - **重入退化为临时缓冲**：若某个打包路径在一次借用期间又调 [`with_panels`]，`RefCell` 已经在
//   借出状态，这里退化成一次性 `Vec`——宁可多分配一次，也不 panic。
thread_local! {
    /// 本线程的面板暂存区；长度只增不减。
    static PANELS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 借一块长度 `len` 的 f32 暂存区（不足则扩容并清零新增部分），返回 `f` 的结果。
pub fn with_panels<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    PANELS.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            if buf.len() < len {
                buf.resize(len, 0.0);
            }
            f(&mut buf[..len])
        }
        // 重入（同一次调用里又借一次）：这一次用临时缓冲，语义不变。
        Err(_) => f(&mut vec![0.0f32; len]),
    })
}

/// 线程数：`LLM_THREADS` 环境变量优先，否则全核。四档 GEMM（f32 Nt/Tn、i8、SME）共用，
/// 与具体精度无关，故住在 native 根而不是某个精度模块里。
pub fn threads_override() -> usize {
    std::env::var("LLM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|x| *x > 0)
        .unwrap_or_else(|| thread::available_parallelism().map_or(1, std::num::NonZero::get))
}

/// 向上对齐到 `block` 的整数倍：面板列宽补齐到 NR/I8_NR，补零列由零初始化面板承担。
pub const fn align_up(x: usize, block: usize) -> usize {
    x.div_ceil(block) * block
}

/// 一条微内核的签名：算 `out[m,n] = a[m,k] · b[k,n]`，形状不可整铺/不划算时返回 `false`。
pub type Gemm = fn(&[f32], &[f32], usize, usize, usize, &mut [f32]) -> bool;

/// 编译期登记的一条微内核：名字、瓦片形状（`mr × nr`）与入口。
pub struct Native {
    /// 内核名（如 `"sme"`），给日志/调优留一个稳定标识。
    pub name: &'static str,
    /// 微内核行数。
    pub mr: usize,
    /// 微内核列数。
    pub nr: usize,
    /// 计算入口：成功写回 `out` 返回 `true`，拒绝该形状返回 `false`。
    pub run: Gemm,
}

/// 可用的主机原生微内核表——**编译期配置**：哪一行存在由工具链与目标架构决定，调用方看不到 `cfg`。
///
/// 目前只登记 SME 这一条「整铺大 GEMM」的专用通道；NEON/AVX2 8×8 是 `gemm8x8` 里恒在的兜底，
/// 不登记（它没有拒绝条件，`gemm_negotiate` 命中前先由它接单没有意义）。
pub const GEMM_KERNELS: &[Native] = &[
    #[cfg(target_arch = "aarch64")]
    Native {
        name: "sme",
        mr: sme::MR,
        nr: sme::NR,
        run: sme::gemm,
    },
];

/// 协商入口：按表序问「这块谁能算」，第一个接住的写回 `out` 并返回 `true`；无人接返回 `false`。
///
/// 这是 `nn` / `cutile` 与设备解耦的唯一面：它们只问结果，不关心哪条通道、哪套特性门。
pub fn gemm_negotiate(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) -> bool {
    GEMM_KERNELS
        .iter()
        .any(|kernel| (kernel.run)(a, b, m, n, k, out))
}

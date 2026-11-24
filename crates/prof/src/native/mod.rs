//! 主机原生微内核（`prof::native`）：设备无关的 CPU 高吞吐内核与它们的协商入口。
//!
//! `prof` 是抽象隔离层（DSL 侧）：它把「在什么设备上、用哪条微内核算一块 GEMM」收成
//! 一张**编译期配置表** `GEMM_KERNELS` 加一个协商函数 `gemm_negotiate`。调用方（`nn` 的
//! GEMM 骨架、未来的 `cutile` tile DSL IR）只问一句「这块谁能算」，不必写任何
//! `cfg(target_arch)` / `cfg(nightly)` / 特性探测——那些门全部收在本模块里。
//!
//! 目录布局照 `prof/src/nvidia/cvt.cu` 的形态：一个设备/通道一个专门文件
//! (`gemm8x8` 是 NEON/AVX2/scalar 的 8×8 寄存器块，`i8` 是 aarch64 dotprod 4×4
//! `sdot` 块，`sme` 是 Apple SME 16×64 ZA 外积)，本文件只放**公共方法**：暂存面板、
//! 线程数、对齐，与那张协商表。

use std::thread;

/// f32 8×8 寄存器块微内核（NEON lane-indexed FMA / AVX2+FMA / scalar 兜底）。
#[path = "gemm8x8.rs"]
pub mod gemm8x8;

/// i8 `sdot` 4×4 寄存器块微内核（aarch64 dotprod）与端口探针；非 aarch64 交回标量兜底。
#[path = "i8.rs"]
pub mod i8;

/// Apple SME 16×64 微内核：nightly + aarch64 才存在（`target_feature(enable = "sme")` 仍不稳定）。
#[cfg(all(nightly, target_arch = "aarch64"))]
#[path = "sme.rs"]
pub mod sme;

/// f32 NEON 寄存器瓦片几何（8×8 块）——编译期常量的单一事实源。
pub mod f32 {
    /// 微内核行数。
    pub const MR: usize = 8;
    /// 微内核列数（两个 NEON 向量 / 一个 AVX2 向量）。
    pub const NR: usize = 8;
    /// K 方向内积分块宽度：`K = 4096` 时整段 B 面板有 4MB，按 256 切块让 A 面板与 B 瓦片同时
    /// 驻留 L1/L2。
    pub const KB: usize = 256;
}

/// i8 sdot 寄存器瓦片几何（4×4 块）：配置常量的单一事实源，瓦片本体在 `i8`。
pub mod i8_geo {
    /// 微内核行数：四行共享同一 B 向量，摊薄 B 加载。
    pub const MR: usize = 4;
    /// 微内核列数：4 个 int32x4 累加器 × 4 行 = 16 寄存器，避开溢出。
    pub const NR: usize = 4;
}

/// Apple SME ZA 外积瓦片几何（16×64）。
pub mod sme_geo {
    /// 微内核行数：一个 16×16 ZA 瓦片覆盖 16 行。
    pub const MR: usize = 16;
    /// 微内核列数：4 个 ZA 瓦片各 16 列，一次写回 64 列。
    pub const NR: usize = 64;
    /// 规模门槛（元素数）：低于此值 `smstart`/`smstop` 与打包固定成本超过外积收益，交回 NEON。
    pub const MIN_ELEMS: usize = 1 << 20;
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
    #[cfg(all(nightly, target_arch = "aarch64"))]
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

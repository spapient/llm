//! 后端能力表：`骨架族 × 元素档` 查表裁决，不写第二张 `match` 分支（ADR 0054 §3）。
//!
//! 一张行表登记「某副骨架支持哪个精度档」；[`supports`] 按两维查表，缺项返回 `false`，
//! 由上层按缺的是哪一族、哪一档定位，而不是回一个泛化 `Unsupported`。
//!
//! ## 边界（ADR 0050）
//!
//! 本层是 L2 契约层，**不认识** L3/L6 的设备字面量：第三维 `backend`（哪台机器能跑）
//! 留到 P3 接 `Executor`/后端真实实现逐条对账后再补，不在此引入设备名。表内只放骨架族与
//! 元素档，二者都来自算子目录行表，不另造一份清单。
//!
//! 骨架键取 [`Skeleton`] 闭集枚举而非自由字符串：族名只有声明层一张表可解析，编译期就
//! 堵死悬空族名；未落地的族（`Scan`）与未核通的精度档不登记，不凭目录名单签发能力。

use crate::decl::Skeleton;
use crate::model::DType;

/// 一条能力登记：一副骨架族支持一个元素档。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Capability {
    /// 骨架族标签：闭集变体，与算子目录 `[op].skeleton` 同源解析。
    pub skeleton: Skeleton,
    /// 该骨架当前落地的元素档。
    pub dtype: DType,
}

/// 能力行表：顺序无关，[`supports`] 全表扫描；条目只有三族一档，列表即查表，不引入索引结构。
///
/// 收敛方向（P3）：本表从算子目录行表派生，避免与 `REGISTRY` 漂移；backend 维对账后在此扩列。
pub const CAPABILITIES: &[Capability] = &[
    Capability {
        skeleton: Skeleton::Elementwise,
        dtype: DType::F32,
    },
    Capability {
        skeleton: Skeleton::Reduce,
        dtype: DType::F32,
    },
    Capability {
        skeleton: Skeleton::Gemm,
        dtype: DType::F32,
    },
];

/// `skeleton` × `dtype` 是否登记为受支持；未登记即缺项，调用方据两维定位，不静默回退。
pub fn supports(skeleton: Skeleton, dtype: DType) -> bool {
    CAPABILITIES
        .iter()
        .any(|cap| cap.skeleton == skeleton && cap.dtype == dtype)
}

/// 一副骨架族的编译期默认派发几何：网格与线程块各三轴。
///
/// 形状相关的网格大小待 G2 几何通道按 `extent` 实算；当前只是族级保守占位种子。
/// 它不参与 tile 打分、不改 `select` 的声明顺序（ADR 0056 G4/G5），真机计时择优（autotune）
/// 仍 opt-in 留 G5。未落地的族（`Scan`）不登记，与 [`CAPABILITIES`] 同口径。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GeometrySeed {
    /// 骨架族标签。
    pub skeleton: Skeleton,
    /// 网格三轴占位；按形状展开的网格推导待几何通道接入。
    pub grid: [u32; 3],
    /// 线程块三轴占位。
    pub block: [u32; 3],
}

/// 族级默认几何种子行表：顺序无关，[`geometry_seed`] 全表扫描；列表即查表，不引入索引结构。
pub const GEOMETRY_SEEDS: &[GeometrySeed] = &[
    GeometrySeed {
        skeleton: Skeleton::Elementwise,
        grid: [1, 1, 1],
        block: [256, 1, 1],
    },
    GeometrySeed {
        skeleton: Skeleton::Reduce,
        grid: [1, 1, 1],
        block: [256, 1, 1],
    },
    GeometrySeed {
        skeleton: Skeleton::Gemm,
        grid: [1, 1, 1],
        block: [16, 16, 1],
    },
];

/// 取一族的默认几何种子；未登记族返回 `None`，不猜一个泛化块大小。
pub fn geometry_seed(skeleton: Skeleton) -> Option<&'static GeometrySeed> {
    GEOMETRY_SEEDS.iter().find(|seed| seed.skeleton == skeleton)
}

pub mod native {
    //! 主机微内核的公开兼容与协商面；算法和 unsafe 均在运行时安全桥。
    //! 主机原生微内核（`prof::native`）：设备无关的 CPU 高吞吐内核与它们的协商入口。
    //!
    //! `prof` 是抽象隔离层（DSL 侧）：它把「在什么设备上、用哪条微内核算一块 GEMM」收成
    //! 一张**编译期配置表** `GEMM_KERNELS` 加一个协商函数 `gemm_negotiate`。调用方（`nn` 的
    //! GEMM 骨架、未来的 `cutile` tile DSL IR）只问一句「这块谁能算」，不必写任何
    //! `cfg(target_arch)` / `cfg(nightly)` / 特性探测——那些门全部收在本模块里。
    //!
    //! FMA、dotprod 与 SME 算法统一住在 `ops/gemm_micro.k`，这里只保留公开兼容面、
    //! 面板缓存和协商表；调用方路径不变，运行时特性与 ABI 全由安全桥收口。

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
                    panel,
                    b,
                    out,
                    (tile.row0, tile.j, tile.n, tile.kb),
                    tile.first,
                )
                .is_ok(),
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

        /// 保持原布尔拒绝语义；非法长度或溢出同样拒绝，校验通过后才进入 intrinsic 桥。
        pub fn dot_block(a: &[i8], bpack: &[i8], out: &mut [i32], gram: Gram) -> bool {
            crate::runtime::gemm_micro::sdot_micro(
                a,
                bpack,
                out,
                (gram.r, gram.blk, gram.n, gram.k, gram.kvec),
            )
            .unwrap_or(false)
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
            let Some(panel_len) = m.checked_mul(k) else {
                return false;
            };
            super::with_panels(panel_len, |panel| {
                crate::runtime::gemm_micro::sme_gemm(a, b, m, n, k, out, panel).unwrap_or(false)
            })
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

    /// 线程配置唯一实现由安全桥持有，公开路径不变。
    pub use crate::runtime::gemm_micro::threads_override;

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
    pub fn gemm_negotiate(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) -> bool {
        GEMM_KERNELS
            .iter()
            .any(|kernel| (kernel.run)(a, b, m, n, k, out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DType;

    #[cfg(test)]
    testkit::test_cases! {
    fn registered_f32_families_supported() {
        for skeleton in [Skeleton::Elementwise, Skeleton::Reduce, Skeleton::Gemm] {
            assert!(supports(skeleton, DType::F32), "{skeleton:?}×f32 应在表内");
        }
    }

    fn unknown_family_or_dtype_is_a_missing_capability() {
        // Scan 族与低比特档尚未核通实现覆盖，不登记为能力。
        assert!(!supports(Skeleton::Scan, DType::F32), "scan 族尚未登记");
        assert!(
            !supports(Skeleton::Elementwise, DType::I8),
            "骨架当前只有 f32 档"
        );
    }

    fn table_keys_stay_within_the_skeleton_closed_set() {
        for cap in CAPABILITIES {
            assert!(
                Skeleton::ALL.contains(&cap.skeleton),
                "未知骨架 {}",
                cap.skeleton.as_str()
            );
        }
    }

    fn geometry_seed_keys_stay_within_the_closed_set() {
        for seed in GEOMETRY_SEEDS {
            assert!(
                Skeleton::ALL.contains(&seed.skeleton),
                "种子表骨架 {} 不在闭集内",
                seed.skeleton.as_str()
            );
        }
    }

    fn landed_families_have_a_default_seed_scan_does_not() {
        for skeleton in [Skeleton::Elementwise, Skeleton::Reduce, Skeleton::Gemm] {
            assert!(
                geometry_seed(skeleton).is_some(),
                "已落地族 {skeleton:?} 应有默认几何种子"
            );
        }
        assert!(
            geometry_seed(Skeleton::Scan).is_none(),
            "Scan 族尚未落地，不登记占位几何"
        );
    }
    }
}

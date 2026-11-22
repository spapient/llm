//! 职责：端口峰值探针清单——哪几条指令路径在本机可测，由这里统一回答。

#![allow(unsafe_code)]

/// 一条端口峰值探针：**一条指令路径**的分母。
///
/// 四个字段就是本模块与测量容器之间的全部协议：组名（`{group}/{threads}{suffix}`）、
/// 一轮的操作次数（算吞吐的分子）、探针本身。清单由提供方给，见 [`probes`]。
#[derive(Clone, Copy, Debug)]
pub struct Probe {
    /// 基准组名前缀，如 `i8_peak`。
    pub group: &'static str,
    /// 组名后缀，用来在同前缀下区分通道，如 `_block`。
    pub suffix: &'static str,
    /// 探针一轮做多少次操作（MAC 数）。
    pub ops_per_round: usize,
    /// 跑 `rounds` 轮；本机没有这条指令路径时返回 `false` 且一条都不执行。
    pub run: fn(usize) -> bool,
}

/// 本机可测的端口探针清单——**由提供方回答**，调用方不写 `cfg`。
///
/// 为什么不把 `cfg(nightly)` 留给调用方：`nightly` 是本 crate 的私有 cfg（`build.rs` 发出），
/// 别的 crate 写它只会拿到 `unexpected_cfgs` 警告；而「这台机器上 SME 通道存不存在」本就是
/// gemm 的判断。缺的通道**直接不在清单里**，而不是留在清单里再让每个调用方判空。
///
/// 每项都必须是**同一条指令路径**的峰值：K7 的 SME 分母不能拿 NEON 峰值凑，i8 的 `sdot`
/// 也不能拿 f32 的 `fma` 凑（F2 口径混用）。
#[cfg(feature = "bench")]
pub fn probes() -> &'static [Probe] {
    PROBES
}

/// SME 一轮的浮点次数：4 条 `fmopa` × 16×16 的 f32 瓦片 × 2 flop（乘 + 加）。
#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
const SME_FLOPS_PER_ROUND: usize = 4 * 16 * 16 * 2;

/// 纯 `fmopa` 与「`fmopa` + 每轮 5 条 `ld1w`」两条曲线：两者 `fmopa` 条数相同，
/// 差额就是取数代价。
#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
const SME_PURE: Probe = Probe {
    group: "sme_peak",
    suffix: "",
    ops_per_round: SME_FLOPS_PER_ROUND,
    run: super::sme::port_run,
};
#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
const SME_MIX: Probe = Probe {
    group: "sme_peak",
    suffix: "_mix",
    ops_per_round: SME_FLOPS_PER_ROUND,
    run: sme_mix_probe,
};

/// 混合探针的固定输入：地址不推进，即每轮命中同一批缓存行——这是取数端口的**上限**，
/// 真实内核走不同行只会更慢，据此算出的取数代价偏小、缺口偏小，拿它下结论不夸大（A3）。
#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
fn sme_mix_probe(rounds: usize) -> bool {
    static A: [f32; 16] = [1.0; 16];
    static B: [f32; 64] = [1.0; 64];
    super::sme::port_mix(rounds, &A, &B)
}

/// 纯 `sdot` 端口：零 load 的理想上限。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const I8_PURE: Probe = Probe {
    group: "i8_peak",
    suffix: "",
    ops_per_round: 2 * super::i8::SDOT_MAC_PER_ROUND,
    run: super::i8::port_sdot,
};

/// `k` 取 1024 使地址在环内单调推进——地址不随循环变化时 LLVM 会把 load 整体提出循环，
/// 那样测的是纯端口而非「取数受限」端口（第一版就是这么错的，见 [`super::i8::port_sdot_block`]）。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const PROBE_K: usize = 1024;

/// 取数受限的 `sdot` 端口：每 `k16` 步 4 条 A load + 4 条 B load 配 16 条 `sdot`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const I8_BLOCK: Probe = Probe {
    group: "i8_peak",
    suffix: "_block",
    ops_per_round: 2 * super::i8::SDOT_BLOCK_MAC_PER_K * PROBE_K,
    run: |rounds| {
        static A: [i8; 4 * PROBE_K] = [3; 4 * PROBE_K];
        static B: [i8; 4 * PROBE_K] = [5; 4 * PROBE_K];
        super::i8::port_sdot_block(rounds, &A, &B, PROBE_K)
    },
};

#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
const PROBES: &[Probe] = &[SME_PURE, SME_MIX, I8_PURE, I8_BLOCK];
#[cfg(all(
    feature = "bench",
    not(all(nightly, target_arch = "aarch64")),
    target_arch = "aarch64"
))]
const PROBES: &[Probe] = &[I8_PURE, I8_BLOCK];
/// 非 aarch64 没有 `sdot` 微内核，一条探针都没有。
#[cfg(all(feature = "bench", not(target_arch = "aarch64")))]
const PROBES: &[Probe] = &[];

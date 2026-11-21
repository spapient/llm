//! 探针表：框架产不出的裸微基准，给框架的数字一个外部参照。
//!
//! 三条探针各自回答一个"只能真机测"的问题，源码进二进制（[`Probe::source`]），
//! 在目标机落盘后交给 [`super::Recipe`] 编译——仓库不存二进制，也不依赖目标机上有源码树。
//!
//! 结果不当结论用：探针测的是**硬件与工具链**（FMA 峰值、手写分块 GEMM、muBLAS），
//! 框架的数是"生成器 + 调度 + 内核"的合成量；两者只做数量级对照，见 `docs/MUSA.md`。

/// 一支探针：名字 + 编译车道 + 源码。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Probe {
    /// 命令行上用的名字。
    pub name: &'static str,
    /// 一句话说明它测什么（`musa list` 打印）。
    pub about: &'static str,
    /// 交给 [`super::Recipe::at`] 的车道名。
    pub lane: &'static str,
    /// 单文件源码。
    pub source: &'static str,
}

/// 全部探针。
pub const PROBES: [Probe; 3] = [FMA, GEMM, MUBLAS];

/// FMA 峰值：`fmaf` 乘加链，测的是本机算力上限。
pub const FMA: Probe = Probe {
    name: "fma",
    about: "FP32 FMA 峰值（fmaf 乘加链，一条发射）",
    lane: "plain",
    source: include_str!("../../assets/musa/fma_peak.cu"),
};

/// 手写分块 GEMM：`128×128` 分块 + 寄存器块，给生成器的循环注入一个手工上限。
pub const GEMM: Probe = Probe {
    name: "simt-gemm",
    about: "手写 128×128 分块 FP32 GEMM（寄存器块 + 共享内存）",
    lane: "plain",
    source: include_str!("../../assets/musa/simt_gemm.cu"),
};

/// muBLAS 对照：厂商库在同一个形状上的数，用来判断差距是"我们的内核"还是"整机"。
pub const MUBLAS: Probe = Probe {
    name: "mublas",
    about: "muBLAS FP32 GEMM（厂商库对照）",
    lane: "mublas",
    source: include_str!("../../assets/musa/mublas_gemm.cu"),
};

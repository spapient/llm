//! 高吞吐 CPU GEMM：多线程 + A 列面板打包 + 8×8 寄存器分块。
//! 语义同 [`crate::matmul::matmul_scalar`]，浮点累加顺序不同，按 IEEE 容差对拍（不做逐位相等）。

use std::thread;

use diagnostic::{Code, Error, Result};
use tensor::View;

mod pack;
mod pool;

// 面板暂存池、线程数、对齐助手都是设备无关的公共件，本体单点住在 `cutile::native`，
// 这里只重导出别名供本族子模块经 `super::` 引用；nn 不持有这些实现（B-1 收口）。
pub(super) use cutile::native::{align_up, threads_override, with_panels};

// SME 通道本体在 `cutile::native::sme`（`target_feature` / 特性探测门收在 prof），本模块只经
// 协商面 `gemm_negotiate` 调用，不写 arch/feature 门；其余情形走 NEON 8×8 通道。
use pack::{pack_b_f32_into, pack_b_f32_par_into, pack_panels_into};
use pool::{Bufs, Shape, pool, run_pooled};

/// 端口峰值探针只在测量容器里导出（`bench` 特性），生产构建看不到它；本体在 `cutile::native::sme`。
#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
pub use cutile::native::sme::{port_mix, port_run};

mod tn {
    //! 线性层布局（TN）的稠密 f32 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
    use tensor::View;

    use super::pack::{group_panel, pack_panel};
    use super::pool::{Bufs, Shape, pool, run_pooled};
    use super::{Layout, MR, NR, align_up, threads_override};

    /// 走这条入口的最少行数：一个完整微内核行块。再少的批只有残行（标量），打包的转置成本摊不回来，
    /// 调用方应当继续用按列驻留的 tile 核。
    pub const MIN_ROWS: usize = MR;

    /// 多线程 TN 稠密 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
    ///
    /// 语义同 [`crate::matmul::forward_tn_tiled`]（同一组累加口径，IEEE 容差内一致），吞吐按父模块的
    /// 8×8 寄存器分块 + 常驻池给。
    ///
    /// # Panics
    ///
    /// 三条缓冲长度不足 `rows·k` / `n·k` / `rows·n` 时 panic（调用方先校验）。
    pub fn gemm_f32_tn(x: View<'_, f32>, w: View<'_, f32>, out: &mut [f32]) {
        let (rows, n, k) =
            crate::matmul::tn_geom(x.dims(), w.dims()).expect("gemm_f32_tn 几何不符");
        let (x, w) = (x.as_slice(), w.as_slice());
        super::check_dims(x, w, rows, n, k, out);
        let t = threads_override().min(rows.div_ceil(MR)).max(1);
        if t <= 1 {
            single_tn(x, w, rows, n, k, out);
            return;
        }
        if !pooled_tn(x, w, rows, n, k, out, t) {
            scoped_tn(x, w, out, rows, n, k, t);
        }
    }

    /// 单线程：整段打包一次，再按行块算完整段 K。
    fn single_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
        let npad = align_up(n, NR);
        super::with_panels(npad * k, |bpack| {
            pack_tn_into(bpack, w, n, k);
            compute_tn(x, bpack, w, rows, n, k, out);
        });
    }

    /// 常驻池路径；池不可用（参与数不足 2）时返回 `false`，由 [`scoped_tn`] 兜底。
    fn pooled_tn(
        x: &[f32],
        w: &[f32],
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        t: usize,
    ) -> bool {
        let npad = align_up(n, NR);
        super::with_panels(npad * k, |buf| {
            run_pooled(
                pool(),
                Bufs {
                    a: x.as_ptr() as *const u8,
                    b: w.as_ptr() as *const u8,
                    buf: buf.as_mut_ptr() as *mut u8,
                    out: out.as_mut_ptr() as *mut u8,
                },
                Shape {
                    m: rows,
                    n,
                    k,
                    npad,
                    tn: true,
                },
                MR,
                t,
                super::f32_pack,
                super::f32_calc,
            )
        })
    }

    /// 兜底（池不可用）：打包一次后 shared 给 scope 线程，行块按 MR 对齐切。
    fn scoped_tn(x: &[f32], w: &[f32], out: &mut [f32], rows: usize, n: usize, k: usize, t: usize) {
        let npad = align_up(n, NR);
        let per = rows.div_ceil(MR).div_ceil(t) * MR;
        super::with_panels(npad * k, |bpack| {
            pack_tn_into(bpack, w, n, k);
            let bp: &[f32] = bpack;
            std::thread::scope(|s| {
                for (ti, block) in out[..rows * n].chunks_mut(per * n).enumerate() {
                    let r0 = ti * per;
                    let got = block.len() / n;
                    let a_l = &x[r0 * k..(r0 + got) * k];
                    s.spawn(move || compute_tn(a_l, bp, w, got, n, k, block));
                }
            });
        });
    }

    /// 整段打包（单槽、兜底路径与单线程用）：面板布局与父模块一致，只是源按 `[n][k]` 读。
    fn pack_tn_into(bpack: &mut [f32], w: &[f32], n: usize, k: usize) {
        let npad = align_up(n, NR);
        debug_assert_eq!(bpack.len(), npad * k);
        for g in 0..npad / NR {
            pack_panel(w, bpack, n, k, g * NR, g * k * NR, true);
        }
    }

    /// 本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包（与父模块同一套分工）。
    pub(super) fn stage(
        w: &[f32],
        buf: &mut [f32],
        n: usize,
        k: usize,
        npad: usize,
        p: usize,
        slot: usize,
    ) {
        group_panel::<f32, NR, false>(w, buf, n, k, npad, p, slot, true);
    }

    /// 共用父模块行机器，只固定 TN 布局。
    pub(super) fn compute_tn(
        a: &[f32],
        bpack: &[f32],
        w: &[f32],
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) {
        super::compute_rows(a, bpack, w, super::Extent { rows, n, k }, out, Layout::Tn);
    }

    #[cfg(test)]
    mod tests {
        include!("../../tests/kernel_tn_tests.rs");
    }
}
pub use tn::{MIN_ROWS, gemm_f32_tn};

// f32 NEON 瓦片几何：单一事实源在 `cutile::native::f32`，这里重导出别名供本族子模块经
// `super::MR/NR/KB` 引用；数值不在本 crate 再写一份。
pub(crate) use cutile::native::f32::{KB, MR, NR};

/// 多线程稠密 f32 GEMM：`out[m,n] = a[m,k] · b[k,n]`。
///
/// # Panics
///
/// 三条缓冲长度不足 `m·k` / `k·n` / `m·n` 时 panic（调用方先校验）。
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    check_dims(a, b, m, n, k, out);
    // 协商面：把 16×64 整铺的大 GEMM 交给注册表里能接住的专用通道（SME ZA 外积）；
    // 小题、不可整铺或通道不存在时返回 `false`，退回下方 NEON 8×8 通道。nn 不写任何 arch/feature 门。
    if cutile::native::gemm_negotiate(a, b, m, n, k, out) {
        return;
    }
    let blocks = m.div_ceil(MR);
    let t = threads_override().min(blocks).max(1);
    if t <= 1 {
        gemm_f32_single(a, b, m, n, k, out);
        return;
    }
    // 常驻池：打包与计算共用一次发布（一次 Barrier 隔开两阶段），无重复 spawn。
    let npad = align_up(n, NR);
    with_panels(npad * k, |buf| {
        if run_pooled(
            pool(),
            Bufs {
                a: a.as_ptr() as *const u8,
                b: b.as_ptr() as *const u8,
                buf: buf.as_mut_ptr() as *mut u8,
                out: out.as_ptr() as *mut u8,
            },
            Shape {
                m,
                n,
                k,
                npad,
                tn: false,
            },
            MR,
            t,
            f32_pack,
            f32_calc,
        ) {
            return;
        }
        scoped_f32(a, b, out, m, n, k, t);
    });
}

/// 显式错误版稠密 f32 GEMM：三条缓冲与 `(m, n, k)` 严格相等才跑，不符返回 [`Code::Invalid`]，
/// 不 panic。算术同 [`gemm_f32`]（NEON 微内核 + 多槽池）。
pub fn gemm_f32_checked(a: View<'_, f32>, b: View<'_, f32>, out: &mut [f32]) -> Result<()> {
    let (m, k) = super::matmul::dim2(a.dims(), "gemm_f32")?;
    let (k2, n) = super::matmul::dim2(b.dims(), "gemm_f32")?;
    if k != k2 {
        return Err(Error::new(
            Code::Invalid,
            format!("gemm_f32 收缩轴不符：a·k={k}，b·k={k2}"),
        ));
    }
    if out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "gemm_f32 输出缓冲不符：out={}，期望 m·n={}",
                out.len(),
                m * n
            ),
        ));
    }
    gemm_f32(a.as_slice(), b.as_slice(), m, n, k, out);
    Ok(())
}

/// 尺寸校验：三条缓冲不足 `m·k` / `k·n` / `m·n` 时 panic。
fn check_dims(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &[f32]) {
    assert!(
        a.len() >= m * k && b.len() >= k * n && out.len() >= m * n,
        "gemm_f32 尺寸不符"
    );
}

/// 兜底（池不可用）：B 打包一次后 shared 给 scope 线程。
fn scoped_f32(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize, t: usize) {
    let npad = align_up(n, NR);
    let blocks = m.div_ceil(MR);
    // 行块按 MR 对齐：否则 18 线程切 256 行会得到 15 行/线程，8 行走微内核、7 行掉回标量尾块。
    let per = blocks.div_ceil(t) * MR;
    with_panels(npad * k, |bpack| {
        pack_b_f32_par_into(bpack, b, n, k, t);
        let bp: &[f32] = bpack;
        thread::scope(|s| {
            for (ti, block) in out[..m * n].chunks_mut(per * n).enumerate() {
                let r0 = ti * per;
                let rows = block.len() / n;
                let a_t = &a[r0 * k..(r0 + rows) * k];
                s.spawn(move || compute_gemm(a_t, bp, b, rows, n, k, block));
            }
        });
    });
}

/// 单线程稠密 f32 GEMM（基准确认内核质量用）：`out[m,n] = a[m,k] · b[k,n]`。
pub fn gemm_f32_single(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    let npad = align_up(n, NR);
    with_panels(npad * k, |bpack| {
        pack_b_f32_into(bpack, b, n, k);
        compute_gemm(a, bpack, b, m, n, k, out);
    });
}

/// 行块计算的三元形状：行数、列宽与 K。
#[derive(Clone, Copy)]
struct Extent {
    rows: usize,
    n: usize,
    k: usize,
}

/// 一个行块的块计算：A 打包成 8 行面板，再 j 外层、面板内层跑微内核；残尾走标量。
///
/// `layout` 是**数据而非分支**：`Nt`（B 按列驻留）与 `Tn`（W 转置驻留）曾各写一份逐字
/// 相同的正文，只在 `Sweep.layout` / `edge` 的最后一个实参上不同；现在一份正文配两次调用。
fn compute_rows(
    a: &[f32],
    bpack: &[f32],
    w: &[f32],
    extent: Extent,
    out: &mut [f32],
    layout: Layout,
) {
    let Extent { rows, n, k } = extent;
    let full = rows / MR;
    // A 面板借用线程局部池：面板每次都被整块重写，池里残留的旧值不会被读到，
    // 于是每行块省下一次 `full · k · MR` 的分配与清零。
    with_panels(full * k * MR, |apack| {
        pack_panels_into(a, full, k, apack);
        let mut first = true;
        let mut k0 = 0;
        while k0 < k {
            let kb = KB.min(k - k0);
            sweep_columns(
                Sweep {
                    a,
                    apack,
                    bpack,
                    w,
                    full,
                    n,
                    k,
                    layout,
                },
                out,
                k0,
                kb,
                first,
            );
            first = false;
            k0 += kb;
        }
    });
    // 残行只占 ≤7 行，一次把整段 K 走完，不参与分块累加。
    let tail = full * MR;
    if tail < rows {
        edge(
            a,
            w,
            out,
            Span {
                i: tail,
                j: 0,
                mr: rows - tail,
                nr: n,
                n,
                k,
                stride: k,
            },
            0,
            true,
            layout,
        );
    }
}

/// Nt 视角的块计算：B 按列驻留（父模块的两条入口用它）。
fn compute_gemm(
    a: &[f32],
    bpack: &[f32],
    b: &[f32],
    rows: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) {
    compute_rows(a, bpack, b, Extent { rows, n, k }, out, Layout::Nt);
}

/// 沿列块扫描整行块：整块走微内核，最后一段残列走标量边（`edge` 读未打包的 `a`）。
fn sweep_columns(sweep: Sweep<'_>, out: &mut [f32], k0: usize, kb: usize, first: bool) {
    let Sweep {
        a,
        apack,
        bpack,
        w,
        full,
        n,
        k,
        layout,
    } = sweep;
    let mut j = 0;
    while j < n {
        let nr = NR.min(n - j);
        if nr == NR {
            let start = (j / NR) * k * NR + k0 * NR;
            let b_tile = &bpack[start..start + kb * NR];
            for blk in 0..full {
                micro_packed(
                    &apack[blk * k * MR + k0 * MR..],
                    b_tile,
                    out,
                    Tile {
                        row0: blk * MR,
                        j,
                        n,
                        kb,
                        first,
                    },
                );
            }
        } else {
            edge(
                a,
                w,
                out,
                Span {
                    i: 0,
                    j,
                    mr: full * MR,
                    nr,
                    n,
                    k: kb,
                    stride: k,
                },
                k0,
                first,
                layout,
            );
        }
        j += NR;
    }
}

mod i8;
mod micro {
    //! 微内核整行块扫描坐标与标量残块兜底；微内核本体在 `cutile::native`。

    #[derive(Clone, Copy)]
    pub(super) struct Sweep<'a> {
        pub(super) a: &'a [f32],
        pub(super) apack: &'a [f32],
        pub(super) bpack: &'a [f32],
        pub(super) w: &'a [f32],
        pub(super) full: usize,
        pub(super) n: usize,
        pub(super) k: usize,
        pub(super) layout: Layout,
    }

    #[derive(Clone, Copy)]
    pub(super) struct Span {
        pub(super) i: usize,
        pub(super) j: usize,
        pub(super) mr: usize,
        pub(super) nr: usize,
        pub(super) n: usize,
        pub(super) k: usize,
        pub(super) stride: usize,
    }

    #[derive(Clone, Copy)]
    pub(super) enum Layout {
        Nt,
        Tn,
    }

    pub(super) fn weight_stride(span: Span, k0: usize, layout: Layout) -> (usize, usize, usize) {
        match layout {
            Layout::Nt => (k0 * span.n + span.j, span.n, 1),
            Layout::Tn => (span.j * span.stride + k0, 1, span.stride),
        }
    }

    pub(super) fn edge(
        a: &[f32],
        w: &[f32],
        out: &mut [f32],
        span: Span,
        k0: usize,
        first: bool,
        layout: Layout,
    ) {
        let (base, dp, dc) = weight_stride(span, k0, layout);
        for r in 0..span.mr {
            for c in 0..span.nr {
                let slot = (span.i + r) * span.n + span.j + c;
                let mut acc = if first { 0.0f32 } else { out[slot] };
                let wc = base + c * dc;
                for p in 0..span.k {
                    acc += a[(span.i + r) * span.stride + k0 + p] * w[wc + p * dp];
                }
                out[slot] = acc;
            }
        }
    }
}

/// i8 `sdot` 端口峰值探针（K8 的分母），只在测量容器里导出；本体在 `cutile::native::i8`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub use cutile::native::i8::{
    SDOT_BLOCK_MAC_PER_K, SDOT_MAC_PER_ROUND, port_sdot, port_sdot_block,
};

/// 一条端口峰值探针：**一条指令路径**的分母。
///
/// 四个字段就是本模块与测量容器之间的全部协议：组名（`{group}/{threads}{suffix}`）、
/// 一轮的操作次数（算吞吐的分子）、探针本身。清单由提供方给，见 [`probes`]。
#[cfg(feature = "bench")]
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
/// `nightly` 是本 crate 的私有 cfg，下游不能可靠判断；缺失通道直接不进清单。
/// 每项只测同一条指令路径，SME、sdot 与 f32 FMA 的分母不混用。
#[cfg(feature = "bench")]
pub fn probes() -> &'static [Probe] {
    PROBES
}

#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
const SME_FLOPS_PER_ROUND: usize = 4 * 16 * 16 * 2;

#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
const SME_PURE: Probe = Probe {
    group: "sme_peak",
    suffix: "",
    ops_per_round: SME_FLOPS_PER_ROUND,
    run: cutile::native::sme::port_run,
};

#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
const SME_MIX: Probe = Probe {
    group: "sme_peak",
    suffix: "_mix",
    ops_per_round: SME_FLOPS_PER_ROUND,
    run: sme_mix_probe,
};

/// 混合探针复用固定缓存行，测取数端口上限；真实内核走不同行只会更慢。
#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
fn sme_mix_probe(rounds: usize) -> bool {
    static A: [f32; 16] = [1.0; 16];
    static B: [f32; 64] = [1.0; 64];
    cutile::native::sme::port_mix(rounds, &A, &B)
}

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const I8_PURE: Probe = Probe {
    group: "i8_peak",
    suffix: "",
    ops_per_round: 2 * cutile::native::i8::SDOT_MAC_PER_ROUND,
    run: cutile::native::i8::port_sdot,
};

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const FMA_PURE: Probe = Probe {
    group: "fma_peak",
    suffix: "",
    ops_per_round: 2 * cutile::native::gemm8x8::FMA_MAC_PER_K,
    run: cutile::native::gemm8x8::port_fma,
};

/// 固定 K 让地址在环内推进，避免 LLVM 把 load 整体提出循环。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const PROBE_K: usize = 1024;

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const I8_BLOCK: Probe = Probe {
    group: "i8_peak",
    suffix: "_block",
    ops_per_round: 2 * cutile::native::i8::SDOT_BLOCK_MAC_PER_K * PROBE_K,
    run: |rounds| {
        static A: [i8; 4 * PROBE_K] = [3; 4 * PROBE_K];
        static B: [i8; 4 * PROBE_K] = [5; 4 * PROBE_K];
        cutile::native::i8::port_sdot_block(rounds, &A, &B, PROBE_K)
    },
};

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
const FMA_BLOCK: Probe = Probe {
    group: "fma_peak",
    suffix: "_block",
    ops_per_round: 2 * cutile::native::gemm8x8::FMA_MAC_PER_K * PROBE_K,
    run: |rounds| {
        static A: [f32; NR * PROBE_K] = [1.0; NR * PROBE_K];
        static B: [f32; NR * PROBE_K] = [1.0; NR * PROBE_K];
        cutile::native::gemm8x8::port_fma_block(rounds, &A, &B, PROBE_K)
    },
};

#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
const PROBES: &[Probe] = &[SME_PURE, SME_MIX, I8_PURE, I8_BLOCK, FMA_PURE, FMA_BLOCK];
#[cfg(all(
    feature = "bench",
    not(all(nightly, target_arch = "aarch64")),
    target_arch = "aarch64"
))]
const PROBES: &[Probe] = &[I8_PURE, I8_BLOCK, FMA_PURE, FMA_BLOCK];
#[cfg(all(feature = "bench", not(target_arch = "aarch64")))]
const PROBES: &[Probe] = &[];

use i8::{f32_calc, f32_pack};
pub use i8::{gemm_i8, gemm_i8_checked, gemm_i8_single};
// 扫描骨架只读侧（Sweep/Span/Layout/残块 edge）留 nn；8×8 微内核本体与 Tile 住在 cutile::native。
use cutile::native::gemm8x8::{Tile, micro_packed};
use micro::{Layout, Span, Sweep, edge};

#[cfg(test)]
mod tests {
    include!("../../tests/kernel_fast_tests.rs");
    // SME 对拍：参考算术在 nn，被测内核入口在 cutile::native::sme；仅 nightly + aarch64 编译。
    #[cfg(all(nightly, target_arch = "aarch64"))]
    include!("../../tests/kernel_sme_tests.rs");
}

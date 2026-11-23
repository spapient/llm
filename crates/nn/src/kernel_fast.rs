//! 高吞吐 CPU GEMM：多线程 + A 列面板打包 + 8×8 寄存器分块。
//! 语义同 [`crate::matmul::matmul_scalar`]，浮点累加顺序不同，按 IEEE 容差对拍（不做逐位相等）。

use std::thread;

use diagnostic::{Code, Error, Result};
use tensor::{View, ViewMut};

#[path = "kernel_fast_pack.rs"]
mod pack;
#[path = "kernel_fast_pool.rs"]
mod pool;

// 面板暂存池、线程数、对齐助手都是设备无关的公共件，本体单点住在 `prof::native`，
// 这里只重导出别名供本族子模块经 `super::` 引用；nn 不持有这些实现（B-1 收口）。
pub(super) use prof::native::{align_up, threads_override, with_panels};

// SME 通道本体在 `prof::native::sme`（`target_feature` / 特性探测门收在 prof），本模块只经
// 协商面 `gemm_negotiate` 调用，不写 arch/feature 门；其余情形走 NEON 8×8 通道。
use pack::{pack_b_f32_into, pack_b_f32_par_into, pack_panels_into};
use pool::{Bufs, Shape, pool, run_pooled};

/// 端口峰值探针只在测量容器里导出（`bench` 特性），生产构建看不到它；本体在 `prof::native::sme`。
#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
pub use prof::native::sme::{port_mix, port_run};

#[path = "kernel_fast_tn.rs"]
mod tn;
pub use tn::{MIN_ROWS, gemm_f32_tn};

// f32 NEON 瓦片几何：单一事实源在 `prof::native::f32`，这里重导出别名供本族子模块经
// `super::MR/NR/KB` 引用；数值不在本 crate 再写一份。
pub(crate) use prof::native::f32::{KB, MR, NR};

/// 多线程稠密 f32 GEMM：`out[m,n] = a[m,k] · b[k,n]`。
///
/// # Panics
///
/// 三条缓冲长度不足 `m·k` / `k·n` / `m·n` 时 panic（调用方先校验）。
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    check_dims(a, b, m, n, k, out);
    // 协商面：把 16×64 整铺的大 GEMM 交给注册表里能接住的专用通道（SME ZA 外积）；
    // 小题、不可整铺或通道不存在时返回 `false`，退回下方 NEON 8×8 通道。nn 不写任何 arch/feature 门。
    if prof::native::gemm_negotiate(a, b, m, n, k, out) {
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
pub fn gemm_f32_checked(a: View<'_, f32>, b: View<'_, f32>, out: ViewMut<'_, f32>) -> Result<()> {
    let (m, k) = super::matmul::dim2(a.dims(), "gemm_f32")?;
    let (k2, n) = super::matmul::dim2(b.dims(), "gemm_f32")?;
    if k != k2 {
        return Err(Error::new(
            Code::Invalid,
            format!("gemm_f32 收缩轴不符：a·k={k}，b·k={k2}"),
        ));
    }
    let out = out.into_slice_mut();
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

#[path = "kernel_fast_i8.rs"]
mod i8;
#[path = "kernel_fast_micro.rs"]
mod micro;

/// i8 `sdot` 端口峰值探针（K8 的分母），只在测量容器里导出；本体在 `prof::native::i8`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub use prof::native::i8::{SDOT_BLOCK_MAC_PER_K, SDOT_MAC_PER_ROUND, port_sdot, port_sdot_block};

/// 端口探针清单：哪几条指令路径可测，由这里统一回答（调用方不写 `cfg`）。
#[cfg(feature = "bench")]
#[path = "kernel_fast_probe.rs"]
mod probe;
#[cfg(feature = "bench")]
pub use probe::{Probe, probes};

use i8::{f32_calc, f32_pack};
pub use i8::{gemm_i8, gemm_i8_checked, gemm_i8_single};
// 扫描骨架只读侧（Sweep/Span/Layout/残块 edge）留 nn；8×8 微内核本体与 Tile 住在 prof::native。
use micro::{Layout, Span, Sweep, edge};
use prof::native::gemm8x8::{Tile, micro_packed};

#[cfg(test)]
mod tests {
    include!("../tests/kernel_fast_tests.rs");
    // SME 对拍：参考算术在 nn，被测内核入口在 prof::native::sme；仅 nightly + aarch64 编译。
    #[cfg(all(nightly, target_arch = "aarch64"))]
    include!("../tests/kernel_sme_tests.rs");
}

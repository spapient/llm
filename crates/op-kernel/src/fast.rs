//! 高吞吐 CPU GEMM：多线程 + A 列面板打包 + 8×8 寄存器分块。
//! 语义同 [`crate::matmul::matmul_scalar`]，浮点累加顺序不同，按 IEEE 容差对拍（不做逐位相等）。
//!
//! 关键在「打包」：A 行主序下按列取 8 行会跨 8 条 cache line、每条只用 4 字节，微内核会被访存打穿。
//! 先把每个 8 行块转置成 `[p][8]` 面板，微内核每条 `p` 只做两次**连续**向量加载，再用 `lane` 广播
//! 到 8 个累加器；微内核按目标架构选择：aarch64 走 NEON，x86_64 走 AVX2+FMA（运行时探测，缺
//! 能力回退标量），其余架构走标量。`unsafe` 只在这两段 SIMD 微内核（`#[allow(unsafe_code)]`）。

use std::thread;

use diagnostic::{Code, Error, Result};

mod pack {
    //! 打包：把 A 的行块转成 `[p][MR]`、把 B 的列组转成 `[jblk][p][NR]` 面板。
    //!
    //! 微内核每条 `p` 只做两次**连续**加载，前提就是这里的重排。面板写进调用方的暂存区
    //! （[`super::scratch`]，已清零），不在这里分配：`n = k = 4096` 时这块是 67 MB。
    use std::thread;

    use super::{MR, NR};

    /// A 的每个 8 行块转置成 `[p][8]` 面板：微内核每条 `p` 只需两次连续向量加载。
    pub(super) fn pack_panels(a: &[f32], full: usize, k: usize) -> Vec<f32> {
        let mut apack = vec![0.0f32; full * k * MR];
        for blk in 0..full {
            let src = &a[blk * MR * k..(blk * MR + MR) * k];
            let dst = &mut apack[blk * k * MR..blk * k * MR + k * MR];
            for p in 0..k {
                for r in 0..MR {
                    dst[p * MR + r] = src[r * k + p];
                }
            }
        }
        apack
    }

    /// B 打包成 `[jblk][p][NR]`：微内核每条 `p` 只需两次连续向量加载（行主序 B 是每 `p` 跨一条行）。
    ///
    /// `n` 不整除的部分补 0，只有列残尾（`nr < NR`）才回退到 `edge`。写进调用方的面板（已清零），
    /// 不在这里分配：`n = k = 4096` 时这块是 67 MB，每次调用现分配是第 5 条要扇掉的成本。
    pub(super) fn pack_b_f32_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize) {
        let npad = n.div_ceil(NR) * NR;
        debug_assert_eq!(bpack.len(), npad * k);
        for jb in 0..npad / NR {
            let first = jb * NR;
            let cols = if first < n { NR.min(n - first) } else { 0 };
            for p in 0..k {
                let dst = (jb * k + p) * NR;
                bpack[dst..dst + cols].copy_from_slice(&b[p * n + first..p * n + first + cols]);
            }
        }
    }

    /// 并行 B 打包：按 8 列组切线程。`k·n` 小的时候串行更划算（线程创建比转置贵）。
    pub(super) fn pack_b_f32_par_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize, t: usize) {
        let groups = n.div_ceil(NR);
        if t <= 1 || n * k < (1 << 16) {
            pack_b_f32_into(bpack, b, n, k);
            return;
        }
        let per = groups.div_ceil(t);
        let span = per * NR * k;
        thread::scope(|s| {
            for (gi, chunk) in bpack.chunks_mut(span).enumerate() {
                s.spawn(move || pack_b_f32_span(chunk, b, n, k, gi * per));
            }
        });
    }

    /// 一段连续的列组打包：`blk0` 起，每块 `NR · k`。
    fn pack_b_f32_span(chunk: &mut [f32], b: &[f32], n: usize, k: usize, blk0: usize) {
        let blocks = chunk.len() / (NR * k);
        for bi in 0..blocks {
            let first = (blk0 + bi) * NR;
            let cols = if first < n { NR.min(n - first) } else { 0 };
            let tile = &mut chunk[bi * NR * k..(bi + 1) * NR * k];
            for p in 0..k {
                let dst = p * NR;
                tile[dst..dst + cols].copy_from_slice(&b[p * n + first..p * n + first + cols]);
            }
        }
    }

    /// 池化打包的分片体：本槽负责的 8 列组按 `p` 步长轮转，写入 `[jblk][p][NR]`（零初始化覆盖补零列）。
    pub(super) fn b_groups(
        b: &[f32],
        buf: &mut [f32],
        n: usize,
        k: usize,
        npad: usize,
        p: usize,
        slot: usize,
    ) {
        let groups = npad / NR;
        let mut g = slot;
        while g < groups {
            let first = g * NR;
            let cols = if first < n { NR.min(n - first) } else { 0 };
            let base = g * k * NR;
            for q in 0..k {
                let src = q * n + first;
                let dst = base + q * NR;
                buf[dst..dst + cols].copy_from_slice(&b[src..src + cols]);
            }
            g += p;
        }
    }
}
mod pool;

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
///
/// 清零只针对**新增**的那一段：面板的每个字节都会被打包阶段覆写，但列残尾之外的补零区
/// （`n % NR` 的补齐列）与 `edge` 的读路径依赖 0，复用旧内容时必须在扩容时清一次。
fn with_panels<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
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

mod sme;
use pack::{pack_b_f32_into, pack_b_f32_par_into, pack_panels};
use pool::{Buffers, Shape, pool, run_pooled};

mod tn {
    //! 线性层布局（TN）的稠密 f32 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
    //!
    //! 与父模块是同一套 8×8 微内核与同一条常驻池，差别只在一处：权重按**输出行**存放（`w[j][q]`），
    //! 而微内核要的是「每个 `q` 上连续 8 个输出列」。所以打包阶段要做一次转置——`w` 的 8 行、每行
    //! 一段 `k`，转成面板 `[q][8]`；这也正是打包本来就要做的重排，只是读的步长从 `n` 变成 `k`。
    //!
    //! 存在的理由：prefill 的投影走的就是这份布局（权重按输出行常驻），而父模块只认收缩轴在前的
    //! 权重。没有这个入口，服务端只能退回逐列 `forward_tn_tiled`——那份内核每个权重元素被读一遍没错，
    //! 但没有寄存器分块与多线程，实测差 20 倍以上。
    use super::pack::pack_panels;
    use super::pool::{Buffers, Shape, pool, run_pooled};
    use super::{KB, Layout, MR, NR, Span, Sweep, edge, sweep_columns, threads_override};

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
    pub fn gemm_f32_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
        check_dims_tn(x, w, rows, n, k, out);
        let t = threads_override().min(rows.div_ceil(MR)).max(1);
        if t <= 1 {
            single_tn(x, w, rows, n, k, out);
            return;
        }
        if !pooled_tn(x, w, rows, n, k, out, t) {
            scoped_tn(x, w, out, rows, n, k, t);
        }
    }

    /// 尺寸校验：三条缓冲不足 `rows·k` / `n·k` / `rows·n` 时 panic。
    fn check_dims_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &[f32]) {
        assert!(
            x.len() >= rows * k && w.len() >= n * k && out.len() >= rows * n,
            "gemm_f32_tn 尺寸不符"
        );
    }

    /// 单线程：整段打包一次，再按行块算完整段 K。
    fn single_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
        let npad = n.div_ceil(NR) * NR;
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
        let npad = n.div_ceil(NR) * NR;
        super::with_panels(npad * k, |buf| {
            run_pooled(
                pool(),
                Buffers {
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
        let npad = n.div_ceil(NR) * NR;
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
    ///
    /// 写进调用方的暂存区（已清零），不在这里分配：`n = k = 4096` 时这块是 67 MB。
    fn pack_tn_into(bpack: &mut [f32], w: &[f32], n: usize, k: usize) {
        let npad = n.div_ceil(NR) * NR;
        debug_assert_eq!(bpack.len(), npad * k);
        for g in 0..npad / NR {
            let base = g * k * NR;
            fill_panel(&mut bpack[base..base + k * NR], w, n, k, g * NR);
        }
    }

    /// 一个列组的转置打包：`w` 的 `first..first+NR` 行、每行 `k` 个元素 → 面板 `[q][NR]`。
    ///
    /// 读是 8 条步长 `k` 的流、写是连续面板；`k` 远大于 cache line 时读端是 8 条并发预取流，
    /// 与父模块「读 8 条连续、写跨步」正好互补，实际耗时同一量级。
    fn fill_panel(panel: &mut [f32], w: &[f32], n: usize, k: usize, first: usize) {
        let cols = if first < n { NR.min(n - first) } else { 0 };
        for q in 0..k {
            let dst = &mut panel[q * NR..q * NR + cols];
            for (c, slot) in dst.iter_mut().enumerate() {
                *slot = w[(first + c) * k + q];
            }
        }
    }

    /// 本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包（与父模块同一套分工）。
    ///
    /// 读 `w` 的步长是 `k`（按输出行存放），写面板是连续段：这正是与行主序打包唯一的差别。
    pub(super) fn stage(
        w: &[f32],
        buf: &mut [f32],
        n: usize,
        k: usize,
        npad: usize,
        p: usize,
        slot: usize,
    ) {
        let groups = npad / NR;
        let mut g = slot;
        while g < groups {
            let base = g * k * NR;
            fill_panel(&mut buf[base..base + k * NR], w, n, k, g * NR);
            g += p;
        }
    }

    /// 一段行的计算：A 面板打包 + 按 [`KB`] 分块扫列 + 行残尾。
    ///
    /// 与父模块的 `compute` 逐行对应，只有两处同构偏移：完整列块的微内核吃的是同一份 B 面板，
    /// 所以直接复用；列残尾与行残尾要对**原始权重**取值，那边索引从 `w[q*n + j]` 换成 `w[j*k + q]`。
    pub(super) fn compute_tn(
        a: &[f32],
        bpack: &[f32],
        w: &[f32],
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) {
        let full = rows / MR;
        let apack = pack_panels(a, full, k);
        let mut first = true;
        let mut k0 = 0;
        while k0 < k {
            let kb = KB.min(k - k0);
            sweep_columns(
                Sweep {
                    a,
                    apack: &apack,
                    bpack,
                    w,
                    full,
                    n,
                    k,
                    layout: Layout::Tn,
                },
                out,
                k0,
                kb,
                first,
            );
            first = false;
            k0 += kb;
        }
        // 残行（≤ MR−1 行）一次走完整段 K，不参与分块累加——与父模块同一条边界。
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
                Layout::Tn,
            );
        }
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/tn_tests.rs");
    }
}
pub use tn::{MIN_ROWS, gemm_f32_tn};

/// 微内核行数。
const MR: usize = 8;

/// 微内核列数（两个 NEON 向量）。
const NR: usize = 8;

/// 内积分块长度：`K = 4096` 时整段 B 面板有 4MB，按 256 切块让 A 面板与 B 瓦片同时驻留 L1/L2。
const KB: usize = 256;

/// 多线程稠密 f32 GEMM：`out[m,n] = a[m,k] · b[k,n]`。
///
/// # Panics
///
/// 三条缓冲长度不足 `m·k` / `k·n` / `m·n` 时 panic（调用方先校验）。
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    check_dims(a, b, m, n, k, out);
    // SME 通道：16×64 整铺的大 GEMM 交给 ZA 外积；小题或不可整铺时退回下方 NEON 通道。
    if sme::gemm(a, b, m, n, k, out) {
        return;
    }
    let blocks = m.div_ceil(MR);
    let t = threads_override().min(blocks).max(1);
    if t <= 1 {
        gemm_f32_single(a, b, m, n, k, out);
        return;
    }
    // 常驻池：打包与计算共用一次发布（一次 Barrier 隔开两阶段），无重复 spawn。
    let npad = n.div_ceil(NR) * NR;
    with_panels(npad * k, |buf| {
        if run_pooled(
            pool(),
            Buffers {
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
pub fn gemm_f32_checked(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [f32],
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "gemm_f32 缓冲尺寸不符：a={} b={} out={}，期望 m·k={} k·n={} m·n={}",
                a.len(),
                b.len(),
                out.len(),
                m * k,
                k * n,
                m * n
            ),
        ));
    }
    gemm_f32(a, b, m, n, k, out);
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
    let npad = n.div_ceil(NR) * NR;
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
                s.spawn(move || compute(a_t, bp, b, rows, n, k, block));
            }
        });
    });
}

/// 单线程稠密 f32 GEMM（基准确认内核质量用）：`out[m,n] = a[m,k] · b[k,n]`。
pub fn gemm_f32_single(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    let npad = n.div_ceil(NR) * NR;
    with_panels(npad * k, |bpack| {
        pack_b_f32_into(bpack, b, n, k);
        compute(a, bpack, b, m, n, k, out);
    });
}

/// 单线程块计算：A 打包成 8 行面板，再 j 外层、面板内层跑微内核；残尾走标量。
fn compute(a: &[f32], bpack: &[f32], b: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
    let full = rows / MR;
    let apack = pack_panels(a, full, k);
    let mut first = true;
    let mut k0 = 0;
    while k0 < k {
        let kb = KB.min(k - k0);
        sweep_columns(
            Sweep {
                a,
                apack: &apack,
                bpack,
                w: b,
                full,
                n,
                k,
                layout: Layout::Nt,
            },
            out,
            k0,
            kb,
            first,
        );
        first = false;
        k0 += kb;
    }
    // 残行只占 ≤7 行，一次把整段 K 走完，不参与分块累加。
    let tail = full * MR;
    if tail < rows {
        edge(
            a,
            b,
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
            Layout::Nt,
        );
    }
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
mod micro;

use i8::{f32_calc, f32_pack, threads_override};
pub use i8::{gemm_i8, gemm_i8_checked, gemm_i8_single};
use micro::{Layout, Span, Sweep, Tile, edge, micro_packed};

#[cfg(test)]
#[path = "../tests/fast_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/scratch_tests.rs"]
mod scratch_tests;

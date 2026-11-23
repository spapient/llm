
//! 职责：i8 sdot 微内核：MR×NR 打包与 dotprod 探测回退。
// i8 sdot 通道：MR=4 × NR=4，B 转置打包成「每列 k 连续」。
// 每个 k16 一次 4 行 A 加载 + 4 次 B 加载 + 16 次 `vdotq_s32`（256 MAC/24 指令），
// 且每块 B 瓦片被 4 行复用——B 访存比 MR=2/NR=8 再降一半。
// dotprod 运行时探测，缺失走标量回退；`unsafe` 仅限 aarch64 微内核。
// ---------------------------------------------------------------------------

/// i8 微内核行数（四行共享同一 B 向量，摊薄 B 加载）。
use std::thread;

use diagnostic::{Code, Error, Result};

use super::align_up;
use super::compute_gemm;
use super::pack::{b_groups, group_panel};
use super::pool::{Bufs, Job, Shape, pool, run_pooled};
use super::threads_override;
use super::tn;

// i8 sdot 瓦片几何：单一事实源在 `prof::native::i8`，重导出别名供本模块与经 `super::` 引用的用例。
pub(super) use prof::native::i8::{MR as I8_MR, NR as I8_NR};

/// 行块高度：让 A 的 `rc × k` 字节留在 L1（约 128KiB），否则每个列块都要从 L2 重拉整块 A。
///
/// 「留住」这个假设已被实测证伪：`rc` 扫 112 / 64 / 32 与默认 128 在噪声内无差
/// （默认 309.1、366.7 Gelem/s；`rc=112` 304.4、346.4），见 `docs/lld.md` v3.0.40。
fn row_chunk(k: usize) -> usize {
    ((1usize << 17) / k.max(1)).clamp(I8_MR, 256)
}

/// 打包阶段：本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包（列主面板走泛型轮转体）。
#[allow(unsafe_code)]
pub(super) fn i8_pack(j: &Job, slot: usize) {
    unsafe {
        let b = std::slice::from_raw_parts(j.b as *const i8, j.k * j.n);
        let bp = std::slice::from_raw_parts_mut(j.buf as *mut i8, j.npad * j.k);
        group_panel::<i8, I8_NR, true>(b, bp, j.n, j.k, j.npad, j.p, slot, false);
    }
}

/// 计算阶段：只算本槽的行段，`bpack` 由 Barrier 保证整块就绪。
pub(super) fn i8_calc(j: &Job, slot: usize) {
    let r0 = slot * j.rows_per;
    if r0 >= j.m {
        return;
    }
    let rows = (j.m - r0).min(j.rows_per);
    #[allow(unsafe_code)]
    unsafe {
        let a_l = std::slice::from_raw_parts(j.a as *const i8, j.m * j.k);
        let a_l = &a_l[r0 * j.k..(r0 + rows) * j.k];
        let b_l = std::slice::from_raw_parts(j.buf as *const i8, j.npad * j.k);
        let out_l = std::slice::from_raw_parts_mut(j.out as *mut i32, j.m * j.n);
        let out_l = &mut out_l[r0 * j.n..(r0 + rows) * j.n];
        compute_i8_packed(a_l, b_l, j.npad, rows, j.n, j.k, out_l);
    }
}

/// f32 打包体：本槽负责的 8 列组按 `p` 步长轮转，写入 `[jblk][p][NR]`（零初始化覆盖补零列）。
#[allow(unsafe_code)]
pub(super) fn f32_pack(j: &Job, slot: usize) {
    let (n, k, npad) = (j.n, j.k, j.npad);
    unsafe {
        let b = std::slice::from_raw_parts(j.b as *const f32, k * n);
        let dst = std::slice::from_raw_parts_mut(j.buf as *mut f32, npad * k);
        if j.tn {
            tn::stage(b, dst, n, k, npad, j.p, slot);
        } else {
            b_groups(b, dst, n, k, npad, j.p, slot);
        }
    }
}

/// f32 计算体：只算本槽的行段；B 面板由 Barrier 保证整块就绪。
#[allow(unsafe_code)]
pub(super) fn f32_calc(j: &Job, slot: usize) {
    let r0 = slot * j.rows_per;
    if r0 >= j.m {
        return;
    }
    let rows = (j.m - r0).min(j.rows_per);
    unsafe {
        let a = std::slice::from_raw_parts(j.a as *const f32, j.m * j.k);
        let a_l = &a[r0 * j.k..(r0 + rows) * j.k];
        let bp = std::slice::from_raw_parts(j.buf as *const f32, j.npad * j.k);
        let b = std::slice::from_raw_parts(j.b as *const f32, j.k * j.n);
        let out = std::slice::from_raw_parts_mut(j.out as *mut f32, j.m * j.n);
        let out_l = &mut out[r0 * j.n..(r0 + rows) * j.n];
        if j.tn {
            tn::compute_tn(a_l, bp, b, rows, j.n, j.k, out_l);
        } else {
            compute_gemm(a_l, bp, b, rows, j.n, j.k, out_l);
        }
    }
}

/// 多线程 i8 GEMM：`out[m,n] = a[m,k] · b[k,n]`，i8 入 i32 累加。
///
/// # Panics
///
/// 三条缓冲长度不足时 panic（调用方先校验）。
pub fn gemm_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    assert!(
        a.len() >= m * k && b.len() >= k * n && out.len() >= m * n,
        "gemm_i8 尺寸不符"
    );
    let cpu = threads_override();
    let npad = align_up(n, I8_NR);
    let groups = npad / I8_NR;
    let t = cpu.min(m.div_ceil(I8_MR)).min(groups).max(1);
    if t <= 1 {
        single_i8(a, b, m, n, k, out);
        return;
    }
    // 常驻池：一次发布 + 一次 Barrier，无 spawn 成本。
    let mut bpack = vec![0i8; npad * k];
    if run_pooled(
        pool(),
        Bufs {
            a: a.as_ptr() as *const u8,
            b: b.as_ptr() as *const u8,
            buf: bpack.as_mut_ptr() as *mut u8,
            out: out.as_mut_ptr() as *mut u8,
        },
        Shape {
            m,
            n,
            k,
            npad,
            tn: false,
        },
        I8_MR,
        t,
        i8_pack,
        i8_calc,
    ) {
        return;
    }
    scoped_i8(a, b, out, m, n, k, t);
}

/// 显式错误版对称 int8 GEMM：三条缓冲与 `(m, n, k)` 严格相等才跑，不符返回 [`Code::Invalid`]，
/// 不 panic。算术同 [`gemm_i8`]（i32 精确累加，与 [`crate::matmul_i8::forward_tiled`] 逐位一致）。
pub fn gemm_i8_checked(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    out: &mut [i32],
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || out.len() != m * n {
        return Err(Error::new(
            Code::Invalid,
            format!(
                "gemm_i8 缓冲尺寸不符：a={} b={} out={}，期望 m·k={} k·n={} m·n={}",
                a.len(),
                b.len(),
                out.len(),
                m * k,
                k * n,
                m * n
            ),
        ));
    }
    gemm_i8(a, b, m, n, k, out);
    Ok(())
}

/// 单线程路径：B 打包一次，直接算完整块。
pub(super) fn single_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    let (bpack, npad) = pack_b(b, n, k);
    compute_i8_packed(a, &bpack, npad, m, n, k, out);
}

/// 兜底路径：池不可用（worker 起不来）时退回 scope 线程。
pub(super) fn scoped_i8(
    a: &[i8],
    b: &[i8],
    out: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
    t: usize,
) {
    let mut rows_per = m.div_ceil(t);
    // 每线程成组行，保证微内核分块不被切碎（缺则末线程走标量尾块）。
    rows_per = rows_per.div_ceil(I8_MR) * I8_MR;
    // B 只打包一次、全部线程共享：N 线程各打包一遍会让冗余带宽吞掉扩展性。
    let (bpack, npad) = pack_b_par(b, n, k);
    thread::scope(|s| {
        for (ti, block) in out[..m * n].chunks_mut(rows_per * n).enumerate() {
            let r0 = ti * rows_per;
            let rows = block.len() / n;
            let a_t = &a[r0 * k..(r0 + rows) * k];
            let bp = &bpack;
            s.spawn(move || compute_i8_packed(a_t, bp, npad, rows, n, k, block));
        }
    });
}

/// 单线程 i8 GEMM（基准确认内核质量用）。
pub fn gemm_i8_single(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    single_i8(a, b, m, n, k, out);
}

/// `sdot` 端口峰值探针一轮的乘加次数：12 条互不依赖的 `sdot` × 16 项。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub const SDOT_MAC_PER_ROUND: usize = 12 * 16;

/// i8 `sdot` 端口峰值：`rounds` 轮 × 12 条 `sdot`，操作数与累加器全常驻，**内层零 load 零 store**。
///
/// 这是 K8 的分母：i8 路径走的是 NEON `sdot`（每指令 4 组 × 4 项 = 16 次乘加），不是 SME
/// 的 `fmopa`，所以 f32 FMA 峰值与 SME 峰值**都不能**当分母——那会得到 >100% 的无意义百分比（F2/F18）。
///
/// **12 条链是实测平台起点**（链数扫描，单线程 GMAC/s：4 → 91、8 → 182、12 → 272、
/// 16 → 274、20 → 273、24 → 237、32 → 232；两次重复误差 <1%）。平台是 12–20 条，
/// ≥24 条因累加器溢出寄存器回落到 235 一带。滑下去的 8 条链版本低于平台 1.5 倍，
/// 曾把 GEMM 误判成超峰值——分母偏低会让百分比虚高，所以这里取平台**左侧**的 12 条而不是最高点。
/// 延迟/发射模型解释不了这个数值（274 GMAC/s ÷ 16 MAC = 17 G 条/s），故只报实测值，不报推导值。
///
/// 本机无 `dotprod` 或 `rounds == 0` 时返回 `false` 且不执行任何向量指令。只在 `bench` 特性下编译。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub fn port_sdot(rounds: usize) -> bool {
    if rounds == 0 || !std::arch::is_aarch64_feature_detected!("dotprod") {
        return false;
    }
    // 安全性：dotprod 已运行时探测；函数体内只有寄存器运算，不触碰内存。
    #[allow(unsafe_code)]
    unsafe {
        sdot_rounds(rounds);
    }
    true
}

/// 纯 `sdot` 循环：12 个累加器互不依赖，累加结果进 `black_box` 防整段消除。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn sdot_rounds(rounds: usize) {
    use std::arch::aarch64::*;
    let a = vdupq_n_s8(3);
    let b = vdupq_n_s8(5);
    let mut acc = [vdupq_n_s32(0); 12];
    for _ in 0..rounds {
        for chain in &mut acc {
            *chain = vdotq_s32(*chain, a, b);
        }
    }
    std::hint::black_box(acc);
}

/// `sdot` **取数受限**峰值一轮的 k16 步数换算系数：每步 16 条 `sdot`，故一轮供 `16 * k` 次乘加。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub const SDOT_BLOCK_MAC_PER_K: usize = 16;

/// `sdot` **取数受限**峰值：一轮 = `k/16` 个 k16 步，每步 4 条 A load + 4 条 B load + 16 条
/// `sdot`，与生产微内核 [`sdot_tile`] 逐指令一致（同一份代码，不会漂移）。
///
/// **为什么要它**：纯端口探针零 load，是理想上限；生产核每 16 条 `sdot` 要发 8 条 load。
/// 两者之差就是取数代价，也是「K8 的缺口里多少是结构注定、多少是核写得差」的分界线。
///
/// **第一版是错探针，必须记下来**：当时把 A/B 写成固定 64 字节、地址用常量偏移
/// `bp.add(nb * 16)`，而循环变量不进地址，于是 LLVM 把 8 条 load **整体提到循环外**——
/// `otool -tV` 实测循环体只剩 16 条 `sdot` 加 `subs`/`b.ne`，`ldp q16,q17,[x0]` 只出现一次。
/// 那样的「混合峰值 ≈ 纯峰值」是空证据：它根本没在环内取数。改成与生产相同的 `base + p`
/// 单调推进后，取数被迫留在环内，实测比值才是真的（同一台机、同一编译参数，中位 0.875）。
///
/// 但仓内 `criterion` 口径与那支独立程序不同，且**仓内才是权威**：同一会话
/// `i8_peak/1_block/port` 539.37 vs `i8_peak/1/port` 543.19 = **0.993**——单线程取数几乎免费；
/// 18 线程 5667.3 vs 6736.0 = **0.841**，即瓦片体自身能到端口峰值的 84%。
/// 故「4×4 tile 注定只能到 87.5%」被证伪（单线程天花板 99%、18 线程 84%）。
///
/// **它现在与生产共用 [`sdot_tile`]，因此不再是独立仪器**：N5 要求探针与生产同源、不许漂移，
/// 代价是它的绝对值不能再横向比"换了一版取数写法"的旧数（旧独立版单线程 493.3、18 线程 4032.8）。
/// 它仍然回答一个没有别的口径能回答的问题：**同一个瓦片体放进零记账的合成循环里能跑多快**，
/// 于是"生产离瓦片体上限差多少"= 内存系统 + 瓦片外记账，而不是核写得差。
///
/// A/B 与 `k` 的形状校验：`k` 非零且是 16 的倍数，两侧各留足 `4k`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
fn block_shaped(a: &[i8], b: &[i8], k: usize) -> bool {
    k != 0 && k.is_multiple_of(16) && a.len() >= 4 * k && b.len() >= 4 * k
}

/// 本机无 `dotprod`、`rounds == 0`、形状不合时返回 `false` 且不执行任何向量指令。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub fn port_sdot_block(rounds: usize, a: &[i8], b: &[i8], k: usize) -> bool {
    if rounds == 0 || !block_shaped(a, b, k) || !std::arch::is_aarch64_feature_detected!("dotprod")
    {
        return false;
    }
    // 安全性：dotprod 已运行时探测；A/B 各 4k 字节已在上面校验。
    #[allow(unsafe_code)]
    unsafe {
        sdot_block_rounds(rounds, a.as_ptr(), b.as_ptr(), k);
    }
    true
}

/// 取数受限循环：4 行 × 4 列累加器，每 k16 步 4 条 A load + 4 条 B load + 16 条 `sdot`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
#[inline(never)]
unsafe fn sdot_block_rounds(rounds: usize, ap: *const i8, bp: *const i8, k: usize) {
    unsafe {
        for _ in 0..rounds {
            std::hint::black_box(sdot_tile(ap, bp, k, k));
        }
    }
}

/// 一个 4 行 × 4 列瓦片的整数累加：`bp` 指向**本瓦片首列**在 B 面板里的起点，只累加 `p < kvec`。
///
/// 生产微内核（[`kernel::dot_one`]）与取数受限探针（[`sdot_block_rounds`]）共用**这一份**：
/// 探针的全部意义是"与生产逐指令一致"，两份写法必然漂移，一份才有保证（N5/可替换）。
/// `kvec` 是向量化段长度，标量尾巴由 [`kernel::harvest`] 补，不在这里。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
#[inline]
unsafe fn sdot_tile(
    ap: *const i8,
    bp: *const i8,
    k: usize,
    kvec: usize,
) -> [[std::arch::aarch64::int32x4_t; I8_MR]; I8_NR] {
    use std::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_s32(0);
        // 列优先：一列一个内层迭代，块内推进不必同时索引四行。
        let mut accs = [[zero; I8_MR]; I8_NR];
        let mut p = 0;
        while p < kvec {
            let av: [int8x16_t; I8_MR] = [
                vld1q_s8(ap.add(p)),
                vld1q_s8(ap.add(k + p)),
                vld1q_s8(ap.add(2 * k + p)),
                vld1q_s8(ap.add(3 * k + p)),
            ];
            for (nb, column) in accs.iter_mut().enumerate() {
                let bv = vld1q_s8(bp.add(nb * k + p));
                for (cell, a) in column.iter_mut().zip(av) {
                    *cell = vdotq_s32(*cell, a, bv);
                }
            }
            p += 16;
        }
        accs
    }
}

/// B 转置打包成 `[nblk][nb][k]`（同列 k 连续），返回 `(面板, n 补齐到 8)`。
mod kernel {
    use super::{I8_MR, I8_NR, sdot_tile};

    /// 单线程块计算（B 面板已备）：**N 分组在最外层**，每组 8×k 的 B 瓦片常驻 L1，
    /// 被本线程所有行对复用——否则每行对都把整块 B 重读一遍，卡在内存带宽。
    pub(super) fn compute_i8_packed(
        a: &[i8],
        bpack: &[i8],
        npad: usize,
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [i32],
    ) {
        let kvec = k - k % 16;
        let rc = (super::row_chunk(k) / I8_MR) * I8_MR;
        let mut r0 = 0;
        while r0 < rows {
            let rend = (r0 + rc).min(rows);
            let mut blk = 0;
            while blk < npad {
                let mut r = r0;
                while r + I8_MR <= rend {
                    pair_one(a, bpack, out, Gram { r, blk, n, k, kvec });
                    r += I8_MR;
                }
                while r < rend {
                    scalar_row_group(a, bpack, out, r, blk, n, k);
                    r += 1;
                }
                blk += I8_NR;
            }
            r0 = rend;
        }
    }

    /// 一个 8 列块的两行：起点行 `r`、列块起点 `blk`，矩阵列宽 `n`、内积长 `k` 与向量化长 `kvec`。
    #[derive(Clone, Copy)]
    pub(super) struct Gram {
        r: usize,
        blk: usize,
        n: usize,
        k: usize,
        #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
        kvec: usize,
    }

    /// 单个 8 列块、两行微内核分派：dotprod 可用且 k≥16 走向量，否则标量两行。
    #[allow(unsafe_code)]
    pub(super) fn pair_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
        #[cfg(target_arch = "aarch64")]
        {
            if g.kvec >= 16 && std::arch::is_aarch64_feature_detected!("dotprod") {
                // 安全性：dotprod 已运行时探测。
                unsafe { dot_one(a, bpack, out, g) };
                return;
            }
        }
        scalar_row_group(a, bpack, out, g.r, g.blk, g.n, g.k);
        for mr in 1..I8_MR {
            scalar_row_group(a, bpack, out, g.r + mr, g.blk, g.n, g.k);
        }
    }

    /// aarch64 dotprod：一个 4 列块的 4 行（`unsafe` 岛）。
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "dotprod")]
    #[allow(unsafe_code)]
    unsafe fn dot_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
        unsafe {
            let ap = a.as_ptr().add(g.r * g.k);
            let bp = bpack.as_ptr();
            let accs = sdot_tile(ap, bp.add(g.blk * g.k), g.k, g.kvec);
            harvest(ap, bp, out, (g.r, g.n, g.k, g.kvec), g.blk, accs);
        }
    }

    /// 一个 4 列块的收口：水平求和 + 标量补 `kvec..k` 的余数，写回 4 行。
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "dotprod")]
    #[allow(unsafe_code)]
    unsafe fn harvest(
        ap: *const i8,
        bp: *const i8,
        out: &mut [i32],
        (r, n, k, kvec): (usize, usize, usize, usize),
        blk: usize,
        accs: [[std::arch::aarch64::int32x4_t; I8_MR]; I8_NR],
    ) {
        use std::arch::aarch64::*;
        unsafe {
            for (nb, column) in accs.iter().enumerate() {
                let jj = blk + nb;
                if jj < n {
                    for (mr, cell) in column.iter().enumerate() {
                        let mut s = vaddvq_s32(*cell);
                        let mut q = kvec;
                        while q < k {
                            s += *ap.add(mr * k + q) as i32 * *bp.add(jj * k + q) as i32;
                            q += 1;
                        }
                        out[(r + mr) * n + jj] = s;
                    }
                }
            }
        }
    }

    /// 标量单行、单 4 列块（打包 B 取数，任意 n/k）。
    pub(super) fn scalar_row_group(
        a: &[i8],
        bpack: &[i8],
        out: &mut [i32],
        row: usize,
        blk: usize,
        n: usize,
        k: usize,
    ) {
        let ar = &a[row * k..(row + 1) * k];
        for nb in 0..I8_NR {
            let jj = blk + nb;
            if jj < n {
                let bp = &bpack[jj * k..jj * k + k];
                let mut acc = 0i32;
                for p in 0..k {
                    acc += ar[p] as i32 * bp[p] as i32;
                }
                out[row * n + jj] = acc;
            }
        }
    }
}

mod pack {
    use super::super::pack::{pack_serial, pack_span};
    use super::I8_NR;
    use super::align_up;
    use super::thread;
    use super::threads_override;

    pub(super) fn pack_b(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
        let npad = align_up(n, I8_NR);
        let mut bpack = vec![0i8; npad * k];
        pack_serial::<i8, I8_NR, true>(&mut bpack, b, n, k, false);
        (bpack, npad)
    }

    /// 并行 B 打包：转置是跨 `n` 字节步长的 cache-miss 密集写，小 `n` 大 `k` 时串行打包
    /// 会独占整段耗时（如 256×256×4096 的 1MB 面板），这里按 4 列组切成线程。
    pub(super) fn pack_b_par(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
        let npad = align_up(n, I8_NR);
        let groups = npad / I8_NR;
        let t = threads_override().min(groups);
        if t <= 1 || npad * k < (1 << 16) {
            return pack_b(b, n, k);
        }
        let mut bpack = vec![0i8; npad * k];
        let per = groups.div_ceil(t);
        let span = per * I8_NR * k;
        thread::scope(|s| {
            for (gi, chunk) in bpack.chunks_mut(span).enumerate() {
                let blk0 = gi * per;
                s.spawn(move || pack_span::<i8, I8_NR, true>(chunk, b, n, k, blk0, false));
            }
        });
        (bpack, npad)
    }
}
use kernel::compute_i8_packed;
use pack::{pack_b, pack_b_par};

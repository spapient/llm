//! 高吞吐 CPU GEMM：多线程 + A 列面板打包 + NEON 8×8 寄存器分块。
//! 语义同 [`crate::matmul::matmul_scalar`]，浮点累加顺序不同，按 IEEE 容差对拍（不做逐位相等）。
//!
//! 关键在「打包」：A 行主序下按列取 8 行会跨 8 条 cache line、每条只用 4 字节，微内核会被访存打穿。
//! 先把每个 8 行块转置成 `[p][8]` 面板，微内核每条 `p` 只做两次**连续**向量加载，再用 `lane` 广播
//! 到 8 个累加器；`unsafe` 只在 aarch64 的 NEON 微内核这一小段（`#[allow(unsafe_code)]`）。

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex, OnceLock};
use std::thread;

/// 微内核行数。
const MR: usize = 8;

/// 微内核列数（两个 NEON 向量）。
const NR: usize = 8;

/// 多线程稠密 f32 GEMM：`out[m,n] = a[m,k] · b[k,n]`。
///
/// # Panics
///
/// 三条缓冲长度不足 `m·k` / `k·n` / `m·n` 时 panic（调用方先校验）。
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    assert!(
        a.len() >= m * k && b.len() >= k * n && out.len() >= m * n,
        "gemm_f32 尺寸不符"
    );
    let cpu = thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let t = cpu.min(m.div_ceil(MR)).max(1);
    let rows_per = m.div_ceil(t);
    thread::scope(|s| {
        for (ti, block) in out[..m * n].chunks_mut(rows_per * n).enumerate() {
            let r0 = ti * rows_per;
            let rows = block.len() / n;
            let a_t = &a[r0 * k..(r0 + rows) * k];
            s.spawn(move || compute(a_t, b, rows, n, k, block));
        }
    });
}

/// 单线程稠密 f32 GEMM（基准确认内核质量用）：`out[m,n] = a[m,k] · b[k,n]`。
pub fn gemm_f32_single(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    compute(a, b, m, n, k, out);
}

/// 单线程块计算：A 打包成 8 行面板，再 j 外层、面板内层跑微内核；残尾走标量。
fn compute(a: &[f32], b: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
    let full = rows / MR;
    let apack = pack_panels(a, full, k);
    sweep_columns(a, &apack, b, out, full, n, k);
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
            },
        );
    }
}

/// A 的每个 8 行块转置成 `[p][8]` 面板：微内核每条 `p` 只需两次连续向量加载。
fn pack_panels(a: &[f32], full: usize, k: usize) -> Vec<f32> {
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

/// 沿列块扫描整行块：整块走微内核，最后一段残列走标量边（`edge` 读未打包的 `a`）。
fn sweep_columns(
    a: &[f32],
    apack: &[f32],
    b: &[f32],
    out: &mut [f32],
    full: usize,
    n: usize,
    k: usize,
) {
    let mut j = 0;
    while j < n {
        let nr = NR.min(n - j);
        if nr == NR {
            for blk in 0..full {
                micro_packed(&apack[blk * k * MR..], b, out, blk * MR, j, n, k);
            }
        } else {
            edge(
                a,
                b,
                out,
                Span {
                    i: 0,
                    j,
                    mr: full * MR,
                    nr,
                    n,
                    k,
                },
            );
        }
        j += NR;
    }
}

/// 打包面板上的 NEON 8×8 微内核（A 面板布局 `[p][8]`，一次算 8 行 8 列）。
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn micro_packed(
    panel: &[f32],
    b: &[f32],
    out: &mut [f32],
    row0: usize,
    j: usize,
    n: usize,
    k: usize,
) {
    use std::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let mut acc = [zero; 16];
        let pp = panel.as_ptr();
        let bp = b.as_ptr();
        for p in 0..k {
            let av0 = vld1q_f32(pp.add(p * MR));
            let av1 = vld1q_f32(pp.add(p * MR + 4));
            let bv0 = vld1q_f32(bp.add(p * n + j));
            let bv1 = vld1q_f32(bp.add(p * n + j + 4));
            let s0 = vdupq_laneq_f32::<0>(av0);
            acc[0] = vfmaq_f32(acc[0], s0, bv0);
            acc[1] = vfmaq_f32(acc[1], s0, bv1);
            let s1 = vdupq_laneq_f32::<1>(av0);
            acc[2] = vfmaq_f32(acc[2], s1, bv0);
            acc[3] = vfmaq_f32(acc[3], s1, bv1);
            let s2 = vdupq_laneq_f32::<2>(av0);
            acc[4] = vfmaq_f32(acc[4], s2, bv0);
            acc[5] = vfmaq_f32(acc[5], s2, bv1);
            let s3 = vdupq_laneq_f32::<3>(av0);
            acc[6] = vfmaq_f32(acc[6], s3, bv0);
            acc[7] = vfmaq_f32(acc[7], s3, bv1);
            let s4 = vdupq_laneq_f32::<0>(av1);
            acc[8] = vfmaq_f32(acc[8], s4, bv0);
            acc[9] = vfmaq_f32(acc[9], s4, bv1);
            let s5 = vdupq_laneq_f32::<1>(av1);
            acc[10] = vfmaq_f32(acc[10], s5, bv0);
            acc[11] = vfmaq_f32(acc[11], s5, bv1);
            let s6 = vdupq_laneq_f32::<2>(av1);
            acc[12] = vfmaq_f32(acc[12], s6, bv0);
            acc[13] = vfmaq_f32(acc[13], s6, bv1);
            let s7 = vdupq_laneq_f32::<3>(av1);
            acc[14] = vfmaq_f32(acc[14], s7, bv0);
            acc[15] = vfmaq_f32(acc[15], s7, bv1);
        }
        let op = out.as_mut_ptr();
        let r = row0;
        vst1q_f32(op.add(r * n + j), acc[0]);
        vst1q_f32(op.add(r * n + j + 4), acc[1]);
        vst1q_f32(op.add((r + 1) * n + j), acc[2]);
        vst1q_f32(op.add((r + 1) * n + j + 4), acc[3]);
        vst1q_f32(op.add((r + 2) * n + j), acc[4]);
        vst1q_f32(op.add((r + 2) * n + j + 4), acc[5]);
        vst1q_f32(op.add((r + 3) * n + j), acc[6]);
        vst1q_f32(op.add((r + 3) * n + j + 4), acc[7]);
        vst1q_f32(op.add((r + 4) * n + j), acc[8]);
        vst1q_f32(op.add((r + 4) * n + j + 4), acc[9]);
        vst1q_f32(op.add((r + 5) * n + j), acc[10]);
        vst1q_f32(op.add((r + 5) * n + j + 4), acc[11]);
        vst1q_f32(op.add((r + 6) * n + j), acc[12]);
        vst1q_f32(op.add((r + 6) * n + j + 4), acc[13]);
        vst1q_f32(op.add((r + 7) * n + j), acc[14]);
        vst1q_f32(op.add((r + 7) * n + j + 4), acc[15]);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn micro_packed(
    panel: &[f32],
    b: &[f32],
    out: &mut [f32],
    row0: usize,
    j: usize,
    n: usize,
    k: usize,
) {
    for r in 0..MR {
        for c in 0..NR {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += panel[p * MR + r] * b[p * n + j + c];
            }
            out[(row0 + r) * n + j + c] = acc;
        }
    }
}

/// 一个待算的标量尾块：左上角 `(i, j)`、覆盖 `mr × nr`，矩阵列宽 `n`、内积长 `k`。
///
/// 六个数合起来才描述得清「哪一块」，所以它们同行进出一个值——拆回参数会让每处调用把同一组
/// 坐标重拼一遍，也让「块多大」这件事在两处各写一份。
#[derive(Clone, Copy)]
struct Span {
    i: usize,
    j: usize,
    mr: usize,
    nr: usize,
    n: usize,
    k: usize,
}

/// 标量尾块：`out[i..i+mr, j..j+nr]` 逐元素点积（与微内核同一组累加口径，容差内一致）。
fn edge(a: &[f32], b: &[f32], out: &mut [f32], span: Span) {
    for r in 0..span.mr {
        for c in 0..span.nr {
            let mut acc = 0.0f32;
            for p in 0..span.k {
                acc += a[(span.i + r) * span.k + p] * b[p * span.n + span.j + c];
            }
            out[(span.i + r) * span.n + span.j + c] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// i8 sdot 通道：MR=4 × NR=4，B 转置打包成「每列 k 连续」。
// 每个 k16 一次 4 行 A 加载 + 4 次 B 加载 + 16 次 `vdotq_s32`（256 MAC/24 指令），
// 且每块 B 瓦片被 4 行复用——B 访存比 MR=2/NR=8 再降一半。
// dotprod 运行时探测，缺失走标量回退；`unsafe` 仅限 aarch64 微内核。
// ---------------------------------------------------------------------------

/// i8 微内核行数（四行共享同一 B 向量，摊薄 B 加载）。
const I8_MR: usize = 4;

/// i8 微内核列数（4 个 int32x4 累加器 × 4 行 = 16 寄存器，避开溢出）。
const I8_NR: usize = 4;

/// 线程数：`LLM_THREADS` 环境变量优先，否则全核。
fn threads_override() -> usize {
    std::env::var("LLM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|x| *x > 0)
        .unwrap_or_else(|| {
            thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
        })
}

// --- 常驻线程池 -------------------------------------------------------------
// 每次调用都 `thread::scope` 会产生「打包一次 + 计算一次」两拨线程创建；小形状
// （如 512×512×512，0.27 GFLOP）里这个固定开销能占到近半。这里改成：进程内
// 常驻 worker，任务只发布指针 + 一次 Barrier，spawn 成本归零。
// `unsafe` 仅用于把本次调用的裸指针交给 worker（主线程自旋等全部完成才返回）。

struct Job {
    a: *const i8,
    b: *const i8,
    bpack: *mut i8,
    out: *mut i32,
    m: usize,
    n: usize,
    k: usize,
    npad: usize,
    rows_per: usize,
    p: usize,
    barrier: Barrier,
    done: AtomicUsize,
}

// 安全性：裸指针指向主线程栈上的输入与堆上的输出；主线程在 `done` 齐之前不返回，
// 期间 worker 只读写互不重叠的分片。
#[allow(unsafe_code)]
unsafe impl Send for Job {}
#[allow(unsafe_code)]
unsafe impl Sync for Job {}

/// 一个 worker 的私有槽：只装一个任务 `(job, 本槽在 job 里的编号)`。
struct Cell {
    job: Mutex<Option<(Arc<Job>, usize)>>,
    cv: Condvar,
}

struct Pool {
    /// 每个常驻 worker 一个私有槽，多槽互不争抢——并发调用可各租一组空闲槽真并行。
    cells: Vec<Cell>,
    /// 空闲槽位图：bit `i` 为 1 表示 worker `i` 空闲；并发发布靠 CAS 租借，杜绝重叠。
    free: AtomicU64,
    /// 保护「等空闲槽」的休眠/唤醒。
    free_lock: Mutex<()>,
    free_cv: Condvar,
    /// 实际起成功的 worker 数（≤ `cells.len()`，也 ≤ 64）。
    live: AtomicUsize,
}

static POOL: OnceLock<&'static Pool> = OnceLock::new();

/// 常驻池（首次调用时创建 `cpu-1` 个常驻 worker，之后只唤醒不复建）。
fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let want = thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .saturating_sub(1)
            .min(64);
        let cells = (0..want)
            .map(|_| Cell {
                job: Mutex::new(None),
                cv: Condvar::new(),
            })
            .collect();
        let p: &'static Pool = Box::leak(Box::new(Pool {
            cells,
            free: AtomicU64::new(0),
            free_lock: Mutex::new(()),
            free_cv: Condvar::new(),
            live: AtomicUsize::new(0),
        }));
        let mut live = 0usize;
        for id in 0..want {
            if thread::Builder::new()
                .name(format!("llm-gemm-{id}"))
                .spawn(move || spin(id, p))
                .is_ok()
            {
                live += 1;
            }
        }
        p.live.store(live, Ordering::Release);
        p.free.store((1u64 << live) - 1, Ordering::Release);
        p
    })
}

/// worker 主循环：等自己的槽有任务 → 取走 → 执行 → 归还槽位并唤醒等槽者。
fn spin(id: usize, pool: &'static Pool) {
    let cell = &pool.cells[id];
    loop {
        let task = {
            let mut g = cell.job.lock().unwrap_or_else(|e| e.into_inner());
            while g.is_none() {
                g = cell.cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            g.take()
        };
        let (j, slot) = task.expect("槽被唤醒必有任务");
        part(&j, slot);
        j.done.fetch_add(1, Ordering::AcqRel);
        // 归还槽位：先置回空闲位，再通知等待空闲槽的发布者。
        pool.free.fetch_or(1u64 << id, Ordering::AcqRel);
        {
            let _g = pool.free_lock.lock().unwrap_or_else(|e| e.into_inner());
            pool.free_cv.notify_all();
        }
    }
}

/// 从空闲位图里挑 `need` 个置位，返回这些位的掩码（不足返回 `None`）。
fn pick_free(free: u64, need: usize) -> Option<u64> {
    let mut mask = 0u64;
    let mut f = free;
    for _ in 0..need {
        if f == 0 {
            return None;
        }
        let bit = 1u64 << f.trailing_zeros();
        mask |= bit;
        f &= !bit;
    }
    Some(mask)
}

/// 一次任务里单个参与者的全部工作：打包自己那份 B 组 → Barrier → 算自己那段行。
fn part(j: &Job, slot: usize) {
    pack_part(j, slot);
    j.barrier.wait();
    compute_part(j, slot);
}

/// 打包阶段：本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包。
fn pack_part(j: &Job, slot: usize) {
    let p = j.p;
    let groups = j.npad / I8_NR;
    #[allow(unsafe_code)]
    unsafe {
        let b = j.b;
        let bp = j.bpack;
        let n = j.n;
        let k = j.k;
        let mut g = slot;
        while g < groups {
            for nb in 0..I8_NR {
                let jj = g * I8_NR + nb;
                if jj < n {
                    let dst = bp.add((g * I8_NR + nb) * k);
                    let mut q = 0;
                    while q < k {
                        *dst.add(q) = *b.add(q * n + jj);
                        q += 1;
                    }
                }
            }
            g += p;
        }
    }
}

/// 计算阶段：只算本槽的行段，`bpack` 由 Barrier 保证整块就绪。
fn compute_part(j: &Job, slot: usize) {
    let r0 = slot * j.rows_per;
    if r0 >= j.m {
        return;
    }
    let rows = (j.m - r0).min(j.rows_per);
    #[allow(unsafe_code)]
    unsafe {
        let a_l = std::slice::from_raw_parts(j.a.add(r0 * j.k), rows * j.k);
        let b_l = std::slice::from_raw_parts(j.bpack as *const i8, j.npad * j.k);
        let out_l = std::slice::from_raw_parts_mut(j.out.add(r0 * j.n), rows * j.n);
        compute_i8_packed(a_l, b_l, j.npad, rows, j.n, j.k, out_l);
    }
}

/// 一次 i8 调用的几何：`npad` 是 `n` 对齐到微内核列数后的补零宽。
#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    n: usize,
    k: usize,
    npad: usize,
}

/// 池化执行；参与数不足 2 时返回 `false`，由调用方走 scope 兜底。
#[allow(unsafe_code)]
fn run_pooled(
    pool: &'static Pool,
    a: &[i8],
    b: &[i8],
    out: &mut [i32],
    shape: Shape,
    t: usize,
) -> bool {
    let Shape { m, n, k, npad } = shape;
    let live = pool.live.load(Ordering::Acquire);
    let p = t.min(live + 1);
    if p < 2 {
        return false;
    }
    let need = p - 1;

    // 从空闲位图租 `need` 个 worker 槽；不足则睡到有槽归还。多调用并发时各自租不同组。
    let mut free = pool.free.load(Ordering::Acquire);
    let mask = loop {
        match pick_free(free, need) {
            Some(m) => match pool
                .free
                .compare_exchange(free, free & !m, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break m,
                Err(actual) => free = actual,
            },
            None => {
                let g = pool.free_lock.lock().unwrap_or_else(|e| e.into_inner());
                let _g = pool.free_cv.wait(g).unwrap_or_else(|e| e.into_inner());
                free = pool.free.load(Ordering::Acquire);
            }
        }
    };

    let mut bpack = vec![0i8; npad * k];
    let rows_per = m.div_ceil(p).div_ceil(I8_MR) * I8_MR;
    let job = Arc::new(Job {
        a: a.as_ptr(),
        b: b.as_ptr(),
        bpack: bpack.as_mut_ptr(),
        out: out.as_mut_ptr(),
        m,
        n,
        k,
        npad,
        rows_per,
        p,
        barrier: Barrier::new(p),
        done: AtomicUsize::new(0),
    });

    // 把任务发到租来的每个槽；槽号 1..=need 对应 worker 在 job 里的分片编号。
    let mut mm = mask;
    let mut slot = 1usize;
    while mm != 0 {
        let id = mm.trailing_zeros() as usize;
        mm &= mm - 1;
        {
            let mut g = pool.cells[id].job.lock().unwrap_or_else(|e| e.into_inner());
            *g = Some((job.clone(), slot));
        }
        pool.cells[id].cv.notify_one();
        slot += 1;
    }

    part(&job, 0);
    while job.done.load(Ordering::Acquire) < p - 1 {
        std::hint::spin_loop();
    }
    true
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
    let npad = n.div_ceil(I8_NR) * I8_NR;
    let groups = npad / I8_NR;
    let t = cpu.min(m.div_ceil(I8_MR)).min(groups).max(1);
    if t <= 1 {
        single_i8(a, b, m, n, k, out);
        return;
    }
    // 常驻池：一次发布 + 一次 Barrier，无 spawn 成本。
    if run_pooled(pool(), a, b, out, Shape { m, n, k, npad }, t) {
        return;
    }
    scoped_i8(a, b, out, m, n, k, t);
}

/// 单线程路径：B 打包一次，直接算完整块。
fn single_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
    let (bpack, npad) = pack_b(b, n, k);
    compute_i8_packed(a, &bpack, npad, m, n, k, out);
}

/// 兜底路径：池不可用（worker 起不来）时退回 scope 线程。
fn scoped_i8(a: &[i8], b: &[i8], out: &mut [i32], m: usize, n: usize, k: usize, t: usize) {
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

/// B 转置打包成 `[nblk][nb][k]`（同列 k 连续），返回 `(面板, n 补齐到 8)`。
fn pack_b(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
    let npad = n.div_ceil(I8_NR) * I8_NR;
    let mut bpack = vec![0i8; npad * k];
    for jj in 0..n {
        let base = ((jj / I8_NR) * I8_NR + jj % I8_NR) * k;
        for p in 0..k {
            bpack[base + p] = b[p * n + jj];
        }
    }
    (bpack, npad)
}

/// 并行 B 打包：转置是跨 `n` 字节步长的 cache-miss 密集写，小 `n` 大 `k` 时串行打包
/// 会独占整段耗时（如 256×256×4096 的 1MB 面板），这里按 8 列组切成线程。
fn pack_b_par(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
    let npad = n.div_ceil(I8_NR) * I8_NR;
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
            s.spawn(move || pack_b_span(chunk, b, n, k, blk0));
        }
    });
    (bpack, npad)
}

/// 一段连续的列组：`blk0` 起、`chunk` 覆盖的每组 8 列各搬一块。
fn pack_b_span(chunk: &mut [i8], b: &[i8], n: usize, k: usize, blk0: usize) {
    let blocks = chunk.len() / (I8_NR * k);
    for bi in 0..blocks {
        let tile = &mut chunk[bi * I8_NR * k..(bi + 1) * I8_NR * k];
        pack_b_tile(tile, b, n, k, (blk0 + bi) * I8_NR);
    }
}

/// 一组 8 列瓦片的转置写：`b`（行主、`n` 列）的第 `first..first+8` 列搬到 `dst`（列主块内偏移）。
/// 越界的列（`n` 不是 8 的倍数）留 0——打包缓冲本来就按 `npad` 补齐，乘加时该列权重为 0。
fn pack_b_tile(dst: &mut [i8], b: &[i8], n: usize, k: usize, first: usize) {
    for nb in 0..I8_NR {
        let jj = first + nb;
        if jj >= n {
            continue;
        }
        let off = nb * k;
        let mut p = 0;
        while p < k {
            dst[off + p] = b[p * n + jj];
            p += 1;
        }
    }
}

/// 单线程块计算（B 面板已备）：**N 分组在最外层**，每组 8×k 的 B 瓦片常驻 L1，
/// 被本线程所有行对复用——否则每行对都把整块 B 重读一遍，卡在内存带宽。
fn compute_i8_packed(
    a: &[i8],
    bpack: &[i8],
    npad: usize,
    rows: usize,
    n: usize,
    k: usize,
    out: &mut [i32],
) {
    let kvec = k - k % 16;
    let mut blk = 0;
    while blk < npad {
        let mut r = 0;
        while r + I8_MR <= rows {
            pair_one(a, bpack, out, Gram { r, blk, n, k, kvec });
            r += I8_MR;
        }
        while r < rows {
            scalar_row_group(a, bpack, out, r, blk, n, k);
            r += 1;
        }
        blk += I8_NR;
    }
}

/// 一个 8 列块的两行：起点行 `r`、列块起点 `blk`，矩阵列宽 `n`、内积长 `k` 与向量化长 `kvec`。
///
/// 五个数合起来才说清「算哪一块」，所以同行进出一个值；`kvec` 与 `k` 分开是为了让标量尾段有
/// 一处明确的边界，而不是在微内核里再算一遍。
#[derive(Clone, Copy)]
struct Gram {
    r: usize,
    blk: usize,
    n: usize,
    k: usize,
    kvec: usize,
}

/// 单个 8 列块、两行微内核分派：dotprod 可用且 k≥16 走向量，否则标量两行。
#[allow(unsafe_code)]
fn pair_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
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
        let accs = dot_block(ap, bp, g.blk, g.k, g.kvec);
        harvest(ap, bp, out, (g.r, g.n, g.k, g.kvec), g.blk, accs);
    }
}

/// 一个 4 列块的 4 行累加：`(4 行 × 4 列的整数累加和)`。
///
/// `blk` 是列块起点、`p` 沿 `kvec` 走 16 字节一拍；`ave[mr]` 是第 `r+mr` 行。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(unsafe_code)]
unsafe fn dot_block(
    ap: *const i8,
    bp: *const i8,
    blk: usize,
    k: usize,
    kvec: usize,
) -> [[std::arch::aarch64::int32x4_t; I8_NR]; I8_MR] {
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
                let bv = vld1q_s8(bp.add((blk + nb) * k + p));
                for (cell, a) in column.iter_mut().zip(av) {
                    *cell = vdotq_s32(*cell, a, bv);
                }
            }
            p += 16;
        }
        accs
    }
}

/// 一个 4 列块的收口：水平求和 + 标量补 `kvec..k` 的余数，写回 4 行。
///
/// `jj >= n` 的列落在打包填充区，只算不写——填充列的乘积不参与任何输出元素。
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
fn scalar_row_group(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul::{close_enough, matmul_scalar};

    fn sample_i8(len: usize, salt: u32) -> Vec<i8> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x % 251) as i32 - 125) as i8
            })
            .collect()
    }

    fn check_i8(m: usize, n: usize, k: usize) {
        let a = sample_i8(m * k, 1);
        let b = sample_i8(k * n, 2);
        let mut want = vec![0i32; m * n];
        crate::matmul_i8::forward_tiled(&a, &b, m, n, k, &mut want).unwrap();
        let mut got = vec![0i32; m * n];
        gemm_i8(&a, &b, m, n, k, &mut got);
        assert_eq!(got, want, "gemm_i8 与参考逐元素相等（整数无舍入）");
    }

    #[test]
    fn i8_matches_on_panel_multiples() {
        check_i8(64, 64, 96);
    }

    #[test]
    fn i8_matches_on_ragged_tails() {
        check_i8(37, 19, 33);
    }

    #[test]
    fn i8_matches_on_prefill_shape() {
        check_i8(256, 256, 4096);
    }

    #[test]
    fn i8_concurrent_calls_are_isolated() {
        // 多槽池：并发调用各租一组空闲槽，必须互不干扰且不挂死。
        let shapes = [(64usize, 64usize, 96usize), (80, 32, 128), (96, 48, 64)];
        thread::scope(|s| {
            for (m, n, k) in shapes {
                let a = sample_i8(m * k, 1);
                let b = sample_i8(k * n, 2);
                let mut want = vec![0i32; m * n];
                crate::matmul_i8::forward_tiled(&a, &b, m, n, k, &mut want).unwrap();
                s.spawn(move || {
                    for _ in 0..200 {
                        let mut got = vec![0i32; m * n];
                        gemm_i8(&a, &b, m, n, k, &mut got);
                        assert_eq!(got, want);
                    }
                });
            }
        });
    }

    fn sample(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x >> 8) as f32) / 256.0 - 32.0
            })
            .collect()
    }

    fn check(m: usize, n: usize, k: usize) {
        let a = sample(m * k, 1);
        let b = sample(k * n, 2);
        let mut want = vec![0.0f32; m * n];
        matmul_scalar(&a, &b, m, n, k, &mut want);
        let mut got = vec![0.0f32; m * n];
        gemm_f32(&a, &b, m, n, k, &mut got);
        close_enough(&got, &want, 1e-3, 1e-3).expect("fast gemm 与标量参考容差内一致");
    }

    #[test]
    fn matches_scalar_on_tile_multiples() {
        check(64, 64, 96);
    }

    #[test]
    fn matches_scalar_on_ragged_edges() {
        check(100, 70, 4093);
    }

    #[test]
    fn matches_scalar_trivial() {
        check(1, 1, 1);
        check(7, 3, 5);
    }

    #[test]
    fn matches_scalar_on_prefill_shape() {
        check(256, 256, 4096);
    }

    /// 吞吐基线：`cargo test --release -p llm-op --lib fast::bench_throughput -- --ignored --nocapture`
    #[test]
    #[ignore = "吞吐基线，手动跑"]
    fn bench_throughput() {
        use std::time::Instant;
        let shapes = [
            (256usize, 256usize, 4096usize),
            (512, 512, 512),
            (1024, 1024, 1024),
        ];
        println!(
            "threads={}",
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
        );
        let g = |ns: f64, m: usize, n: usize, k: usize| 2.0 * m as f64 * n as f64 * k as f64 / ns;
        for (m, n, k) in shapes {
            let a = sample_i8(m * k, 1);
            let b = sample_i8(k * n, 2);
            let mut out = vec![0i32; m * n];
            gemm_i8(&a, &b, m, n, k, &mut out);
            let mut best = f64::INFINITY;
            for _ in 0..20 {
                let t = Instant::now();
                gemm_i8(&a, &b, m, n, k, &mut out);
                best = best.min(t.elapsed().as_nanos() as f64);
            }
            let mt = g(best, m, n, k);
            gemm_i8_single(&a, &b, m, n, k, &mut out);
            let mut best1 = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                gemm_i8_single(&a, &b, m, n, k, &mut out);
                best1 = best1.min(t.elapsed().as_nanos() as f64);
            }
            println!(
                "  i8 {m}x{n}x{k}: fast {mt:.1} GF/s | fast单 {:.1} GF/s | 单/多 {:.2}",
                g(best1, m, n, k),
                g(best1, m, n, k) / mt
            );
        }
    }
}

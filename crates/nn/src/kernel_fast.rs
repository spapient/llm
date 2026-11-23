//! 高吞吐 CPU GEMM：多线程 + A 列面板打包 + 8×8 寄存器分块。
//! 语义同 [`crate::matmul::matmul_scalar`]，浮点累加顺序不同，按 IEEE 容差对拍（不做逐位相等）。

use std::thread;

use diagnostic::{Code, Error, Result};

mod pack {
    //! 打包：把 A 的行块转成 `[p][MR]`、把 B 的列组转成 `[jblk][p][NR]` 面板。
    use std::thread;

    use super::{MR, NR};

    /// A 的每个 8 行块转置成 `[p][8]` 面板：微内核每条 `p` 只需两次连续向量加载。
    ///
    /// 写满 `apack` 的每一个元素（`full · k · MR`），因此可以安全复用一个更早借出的面板池：
    /// 池里没有「上次剩下的半截」会被读到。
    pub(super) fn pack_panels_into(a: &[f32], full: usize, k: usize, apack: &mut [f32]) {
        debug_assert_eq!(apack.len(), full * k * MR);
        for blk in 0..full {
            let src = &a[blk * MR * k..(blk * MR + MR) * k];
            let dst = &mut apack[blk * k * MR..blk * k * MR + k * MR];
            for p in 0..k {
                for r in 0..MR {
                    dst[p * MR + r] = src[r * k + p];
                }
            }
        }
    }

    /// 一个列组面板的内层转置：`first` 起的 `cols` 列 × 全 `k` 行 → `dst[base..]` 的 `[q][NR]`。
    ///
    /// `transposed` 只决定源下标：
    /// - `false`（Nt）：`src[q*n + first + c]`，B 按 `[k][n]` 驻留，整行 `copy_from_slice`；
    /// - `true`（Tn）：`src[(first+c)*k + q]`，W 按 `[n][k]` 驻留，逐元素转置。
    ///
    /// 这是 B 打包的唯一算术：整段、分片与池化三种外层遍历只差列组下标，内层下标只有这一份。
    pub(super) fn pack_panel(
        src: &[f32],
        dst: &mut [f32],
        n: usize,
        k: usize,
        first: usize,
        base: usize,
        transposed: bool,
    ) {
        let cols = if first < n { NR.min(n - first) } else { 0 };
        for q in 0..k {
            let row = base + q * NR;
            if transposed {
                for (c, slot) in dst[row..row + cols].iter_mut().enumerate() {
                    *slot = src[(first + c) * k + q];
                }
            } else {
                let src0 = q * n + first;
                dst[row..row + cols].copy_from_slice(&src[src0..src0 + cols]);
            }
        }
    }

    /// B 打包成 `[jblk][p][NR]`：微内核每条 `p` 只需两次连续向量加载（行主序 B 是每 `p` 跨一条行）。
    pub(super) fn pack_b_f32_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize) {
        let npad = n.div_ceil(NR) * NR;
        debug_assert_eq!(bpack.len(), npad * k);
        for jb in 0..npad / NR {
            pack_panel(b, bpack, n, k, jb * NR, jb * k * NR, false);
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
                s.spawn(move || {
                    let blocks = chunk.len() / (NR * k);
                    for bi in 0..blocks {
                        pack_panel(b, chunk, n, k, (gi * per + bi) * NR, bi * k * NR, false);
                    }
                });
            }
        });
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
            pack_panel(b, buf, n, k, g * NR, g * k * NR, false);
            g += p;
        }
    }
}
mod pool {
    //! 职责：常驻 GEMM 线程池——任务只发布裸指针 + 一次闸门，spawn 成本归零。

    use std::any::Any;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex, OnceLock};
    use std::thread;

    /// 一次 GEMM 调用的几何：`npad` 是 `n` 对齐到微内核列数后的补零宽（f32 与 i8 共用）。
    #[derive(Clone, Copy)]
    pub(super) struct Shape {
        pub(super) m: usize,
        pub(super) n: usize,
        pub(super) k: usize,
        pub(super) npad: usize,
        pub(super) tn: bool,
    }

    /// 一次任务的四条数据指针：输入 A/B、打包面板、输出。一并发布，池只认这一个入口。
    #[derive(Clone, Copy)]
    pub(super) struct Bufs {
        pub(super) a: *const u8,
        pub(super) b: *const u8,
        pub(super) buf: *mut u8,
        pub(super) out: *mut u8,
    }

    pub(super) struct Job {
        pub(super) a: *const u8,
        pub(super) b: *const u8,
        /// 打包后的 B 面板（i8 或 f32，由 `stage`/`calc` 决定解释方式）。
        pub(super) buf: *mut u8,
        pub(super) out: *mut u8,
        pub(super) m: usize,
        pub(super) n: usize,
        pub(super) k: usize,
        pub(super) npad: usize,
        pub(super) tn: bool,
        pub(super) rows_per: usize,
        pub(super) p: usize,
        /// 打包阶段的分片体；类型差异只落在这两个函数里，池机制完全共享。
        stage: fn(&Job, usize),
        calc: fn(&Job, usize),
        /// 打包 → 计算的两阶段闸门；可被 panic 中止以放行所有等待者。
        gate: Gate,
        done: AtomicUsize,
        /// 首个 panic 载荷：worker 记入，主线程收尾时 `resume_unwind`。
        panic: Mutex<Option<Box<dyn Any + Send + 'static>>>,
    }

    // 安全性：裸指针指向主线程栈上的输入与堆上的输出；主线程在 `done` 齐之前不返回，
    // 期间 worker 只读写互不重叠的分片。
    #[allow(unsafe_code)]
    unsafe impl Send for Job {}
    #[allow(unsafe_code)]
    unsafe impl Sync for Job {}

    /// 可中止的参与闸门：`p` 个参与者到齐即放行；任一参与者 `abort()` 立即放行其余所有人。
    struct Gate {
        state: Mutex<GateState>,
        cv: Condvar,
    }

    struct GateState {
        arrived: usize,
        generation: u64,
        aborted: bool,
    }

    impl Gate {
        fn new() -> Self {
            Gate {
                state: Mutex::new(GateState {
                    arrived: 0,
                    generation: 0,
                    aborted: false,
                }),
                cv: Condvar::new(),
            }
        }

        /// 到齐放行，返回 `false`；被 `abort()` 提前放行时返回 `true`。
        fn wait(&self, p: usize) -> bool {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let start_gen = s.generation;
            s.arrived += 1;
            if s.arrived == p {
                s.arrived = 0;
                s.generation += 1;
                self.cv.notify_all();
                return false;
            }
            while s.generation == start_gen && !s.aborted {
                s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
            }
            s.aborted
        }

        /// 中止本次闸门：放行所有已在等待的参与者，并让后续 `wait` 也立即返回 `true`。
        fn abort(&self) {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            s.aborted = true;
            self.cv.notify_all();
        }
    }

    /// 一个 worker 的私有槽：只装一个任务 `(job, 本槽在 job 里的编号)`。
    struct Cell {
        job: Mutex<Option<(Arc<Job>, usize)>>,
        cv: Condvar,
    }

    pub(super) struct Pool {
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

    /// 造 `n` 个空 worker 槽位（每个槽一个 job 锁 + 一个私有条件变量）。
    fn make_cells(n: usize) -> Vec<Cell> {
        (0..n)
            .map(|_| Cell {
                job: Mutex::new(None),
                cv: Condvar::new(),
            })
            .collect()
    }

    /// 由给定槽位建常驻池、起 worker（命名前缀 `prefix`），置好 `live`/`free` 后返回。
    fn spawn_pool(cells: Vec<Cell>, prefix: &str) -> &'static Pool {
        let n = cells.len();
        let p: &'static Pool = Box::leak(Box::new(Pool {
            cells,
            free: AtomicU64::new(0),
            free_lock: Mutex::new(()),
            free_cv: Condvar::new(),
            live: AtomicUsize::new(0),
        }));
        let mut live = 0usize;
        for id in 0..n {
            if thread::Builder::new()
                .name(format!("{prefix}-{id}"))
                .spawn(move || spin(id, p))
                .is_ok()
            {
                live += 1;
            }
        }
        p.live.store(live, Ordering::Release);
        p.free.store((1u64 << live) - 1, Ordering::Release);
        p
    }

    /// 常驻池（首次调用时创建 `cpu-1` 个常驻 worker，之后只唤醒不复建）。
    pub(super) fn pool() -> &'static Pool {
        POOL.get_or_init(|| {
            let want = thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
                .saturating_sub(1)
                .min(64);
            spawn_pool(make_cells(want), "llm-gemm")
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
            let Some((j, slot)) = task else {
                continue;
            };
            // 恐慌隔离：worker 的 panic 不杀死常驻线程；记录后中止闸门，其余参与者与发布者
            // 都能退出而不是卡在闸门 / `done` 自旋上。`done` 与槽位归还在成功与 panic 两条
            // 路径上都无条件执行。
            if let Err(payload) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| part(&j, slot)))
            {
                j.panic
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_or_insert(payload);
                j.gate.abort();
            }
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

    /// 从空闲位图租 `need` 个槽的掩码；不足则睡到有槽归还。多调用并发时各自租不同组。
    fn lease_free(pool: &'static Pool, need: usize) -> u64 {
        let mut free = pool.free.load(Ordering::Acquire);
        loop {
            match pick_free(free, need) {
                Some(m) => match pool.free.compare_exchange(
                    free,
                    free & !m,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return m,
                    Err(actual) => free = actual,
                },
                None => {
                    // 拿锁后必须重查位图：worker 是「先置位、再取锁、再 notify」的，若置位与本次
                    // 加锁之间发生归还，那次 notify 落在「还没人睡下」的空档里，唤醒就丢了——
                    // 睡下去要等到下一次任务结束才醒，并发调用者会互等到卡住。
                    // 所以先持锁复核一次，确认仍然没有空槽，才允许睡。
                    let guard = pool.free_lock.lock().unwrap_or_else(|e| e.into_inner());
                    free = pool.free.load(Ordering::Acquire);
                    if pick_free(free, need).is_some() {
                        drop(guard);
                        continue;
                    }
                    let guard = pool.free_cv.wait(guard).unwrap_or_else(|e| e.into_inner());
                    drop(guard);
                    free = pool.free.load(Ordering::Acquire);
                }
            }
        }
    }

    /// 一次任务里单个参与者的全部工作：打包自己那份 B 组 → 闸门 → 算自己那段行。
    /// 闸门被中止（有参与者 panic）时跳过 `calc`：打包不完整，结果将被丢弃。
    fn part(j: &Job, slot: usize) {
        (j.stage)(j, slot);
        if !j.gate.wait(j.p) {
            (j.calc)(j, slot);
        }
    }

    /// 把任务发到租来的每个槽：槽号 `1..=need` 对应 worker 在 job 里的分片编号。
    fn dispatch(pool: &'static Pool, job: &Arc<Job>, mut mask: u64) {
        let mut slot = 1usize;
        while mask != 0 {
            let id = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            {
                let mut g = pool.cells[id].job.lock().unwrap_or_else(|e| e.into_inner());
                *g = Some((job.clone(), slot));
            }
            pool.cells[id].cv.notify_one();
            slot += 1;
        }
    }

    /// 一个参与者（主线程或 worker）的恐慌隔离执行：panic 记录后中止闸门，其余参与者与发布者
    /// 都能退出而不是卡在闸门 / `done` 自旋上。成功与 panic 两条路径都由调用方无条件收尾。
    fn run_part(job: &Job, slot: usize) {
        if let Err(payload) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| part(job, slot)))
        {
            job.panic
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_or_insert(payload);
            job.gate.abort();
        }
    }

    /// 等全部 worker 完成（成功或 panic 都 `done += 1`），再进入收尾。
    fn await_done(job: &Job, p: usize) {
        while job.done.load(Ordering::Acquire) < p - 1 {
            std::hint::spin_loop();
        }
    }

    /// 等全部槽归还后再把 panic 抛回调用方：此刻没有 worker 再碰 job 的裸指针。
    fn resume_panic(job: &Job) {
        if let Some(payload) = job.panic.lock().unwrap_or_else(|e| e.into_inner()).take() {
            std::panic::resume_unwind(payload);
        }
    }

    /// 池化执行；参与数不足 2 时返回 `false`，由调用方走 scope 兜底。
    #[allow(unsafe_code)]
    pub(super) fn run_pooled(
        pool: &'static Pool,
        bufs: Bufs,
        shape: Shape,
        mr: usize,
        t: usize,
        stage: fn(&Job, usize),
        calc: fn(&Job, usize),
    ) -> bool {
        let Bufs { a, b, buf, out } = bufs;
        let Shape { m, n, k, npad, tn } = shape;
        let live = pool.live.load(Ordering::Acquire);
        let p = t.min(live + 1);
        if p < 2 {
            return false;
        }
        let need = p - 1;
        let mask = lease_free(pool, need);

        let rows_per = m.div_ceil(p).div_ceil(mr) * mr;
        let job = Arc::new(Job {
            a,
            b,
            buf,
            out,
            m,
            n,
            k,
            npad,
            tn,
            rows_per,
            p,
            stage,
            calc,
            gate: Gate::new(),
            done: AtomicUsize::new(0),
            panic: Mutex::new(None),
        });

        dispatch(pool, &job, mask);
        run_part(&job, 0);
        await_done(&job, p);
        resume_panic(&job);
        true
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel_pool_tests.rs");
    }
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

/// SME 通道要 `#[target_feature(enable = "sme")]` 与 `is_aarch64_feature_detected!("sme")`，
/// 两者都还没稳定，因此整块只在「nightly + aarch64」存在；其余情形 `gemm_f32` 走 NEON 通道，
/// 结果一模一样，只是没有 ZA 外积。
#[cfg(all(nightly, target_arch = "aarch64"))]
mod sme {
    //! Apple SME 16×64 微内核通道：ZA 矩阵累加器外积（`fmopa`）+ 运行时探测。
    //!
    //! 瓦片形状是**量过**的，不是推出来的：2×2（32×32）每 4 条 `fmopa` 只摊 4 条 `ld1w`，
    //! 按载入条数该比 1×4 快 7.6%，但实测 1 线程 1256.7 vs 1543.4 Gelem/s、16 线程 2063.7 vs
    //! 2271.8（`sme_peak` 口径），**反而慢 19% / 9%**。载入条数因此不是约束项，
    //! 2×2 的代价（A 面板按 32 宽写、每行分 8 次 16 B 散写）超过了它省下的那一条 `ld1w`。
    //! 结论：1×4 保留，`port_mix` 探针留作这条否证的仪器。
    //!
    //! **本模块在仓里钉住的 stable 工具链上不编译**（`fast.rs` 以 `#[cfg(all(nightly, …))]`
    //! 门控，`sme` 的特征探测与 `target_feature` 仍都是 `E0658`）。以上数字只说明 nightly 构建里
    //! 两种瓦片形状的相对优劣，**不是出货 K7**；出货 K7 的分母是 `fma_peak`（见 `docs/math.md`）。

    use std::thread;

    use std::arch::asm;

    use std::simd::{f32x4, simd_swizzle};

    /// SME 微内核行数：一个 16×16 ZA 瓦片覆盖 16 行。
    const MR: usize = 16;

    /// SME 微内核列数：4 个 ZA 瓦片各 16 列，一次写回 64 列。
    const NR: usize = 64;

    /// 规模门槛（元素数）：低于此值的 GEMM，`smstart`/`smstop` 与打包的固定成本超过外积收益，
    /// 交给 NEON 更划算。
    const MIN_ELEMS: usize = 1 << 20;

    /// 运行时探测：本机是否存在 SME（Apple M4/M5 起）。
    fn available() -> bool {
        std::arch::is_aarch64_feature_detected!("sme")
    }

    /// 16×64 整铺 SME GEMM：`out[m,n] = a[m,k] · b[k,n]`。
    pub(super) fn gemm(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) -> bool {
        if rejects(m, n, k) {
            return false;
        }
        packed(a, b, m, n, k, out, threads_for(m, n));
        true
    }

    /// 线程数：行块（`m/MR`）与列组（`n/NR`）同时决定上限，两者都不能切出空槽。
    fn threads_for(m: usize, n: usize) -> usize {
        super::threads_override().min(m / MR).min(n / NR).max(1)
    }

    /// 拒绝走 SME 的形状：零维、非整铺、低于规模门槛，或本机无 SME。
    fn rejects(m: usize, n: usize, k: usize) -> bool {
        degenerate(m, n, k) || !tiled(m, n) || m * n * k < MIN_ELEMS || !available()
    }

    /// 任一维为零。
    fn degenerate(m: usize, n: usize, k: usize) -> bool {
        m == 0 || n == 0 || k == 0
    }

    /// 行/列是否满足整铺（`MR` 行 × `NR` 列的倍数）。
    fn tiled(m: usize, n: usize) -> bool {
        m.is_multiple_of(MR) && n.is_multiple_of(NR)
    }

    /// 一次整铺 SME GEMM：单次铺线程，先并行打包 A 面板，屏障同步后并行外积累加写回。
    fn packed(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32], t: usize) {
        let mblk = m / MR;
        let alen = mblk * k * MR;
        super::with_panels(alen, |abuf| {
            let barrier = std::sync::Barrier::new(t);
            thread::scope(|s| {
                let barrier = &barrier;
                let mut a_rest = &mut *abuf;
                let mut o_rest = &mut *out;
                for slot in 0..t {
                    let (r0, r1) = (slot * mblk / t, (slot + 1) * mblk / t);
                    let (a_chunk, a_tail) = a_rest.split_at_mut((r1 - r0) * k * MR);
                    let (o_chunk, o_tail) = o_rest.split_at_mut((r1 - r0) * MR * n);
                    a_rest = a_tail;
                    o_rest = o_tail;
                    s.spawn(move || {
                        pack_a(a, a_chunk, k, r0);
                        barrier.wait();
                        compute_stage(&*a_chunk, b, o_chunk, n, k);
                    });
                }
            });
        });
    }

    /// 打包 A 的一个行块段：`src[m,k]` → `dst[块][k][16]`，按 16×16 瓦片转置让微内核连续加载。
    fn pack_a(src: &[f32], dst: &mut [f32], k: usize, mblk0: usize) {
        let blocks = dst.len() / (k * MR);
        for bi in 0..blocks {
            let rowbase = (mblk0 + bi) * MR * k;
            let dstbase = bi * k * MR;
            let kk0 = pack_tiles(src, dst, k, rowbase, dstbase);
            pack_tail(src, dst, k, rowbase, dstbase, kk0);
        }
    }

    /// 4×4 子块转置：两轮 `zip1`/`zip2` 交错，供 [`pack_tiles`] 连续写出。
    #[inline(always)]
    fn transpose4(a0: f32x4, a1: f32x4, a2: f32x4, a3: f32x4) -> (f32x4, f32x4, f32x4, f32x4) {
        let t0 = simd_swizzle!(a0, a1, [0, 4, 1, 5]);
        let t1 = simd_swizzle!(a0, a1, [2, 6, 3, 7]);
        let t2 = simd_swizzle!(a2, a3, [0, 4, 1, 5]);
        let t3 = simd_swizzle!(a2, a3, [2, 6, 3, 7]);
        (
            simd_swizzle!(t0, t2, [0, 1, 4, 5]),
            simd_swizzle!(t0, t2, [2, 3, 6, 7]),
            simd_swizzle!(t1, t3, [0, 1, 4, 5]),
            simd_swizzle!(t1, t3, [2, 3, 6, 7]),
        )
    }

    /// 16 对齐的整段：按 16×16 瓦片做 4×4 转置，返回尾部起点。
    fn pack_tiles(src: &[f32], dst: &mut [f32], k: usize, rowbase: usize, dstbase: usize) -> usize {
        let mut kk0 = 0usize;
        while kk0 + 16 <= k {
            for r in 0..4usize {
                for c in 0..4usize {
                    let rr = r * 4;
                    let cc = kk0 + c * 4;
                    let a0 = f32x4::from_slice(&src[rowbase + rr * k + cc..][..4]);
                    let a1 = f32x4::from_slice(&src[rowbase + (rr + 1) * k + cc..][..4]);
                    let a2 = f32x4::from_slice(&src[rowbase + (rr + 2) * k + cc..][..4]);
                    let a3 = f32x4::from_slice(&src[rowbase + (rr + 3) * k + cc..][..4]);
                    let (b0, b1, b2, b3) = transpose4(a0, a1, a2, a3);
                    let orow = kk0 + c * 4;
                    dst[dstbase + orow * MR + rr..][..4].copy_from_slice(&b0.to_array());
                    dst[dstbase + (orow + 1) * MR + rr..][..4].copy_from_slice(&b1.to_array());
                    dst[dstbase + (orow + 2) * MR + rr..][..4].copy_from_slice(&b2.to_array());
                    dst[dstbase + (orow + 3) * MR + rr..][..4].copy_from_slice(&b3.to_array());
                }
            }
            kk0 += 16;
        }
        kk0
    }

    /// 尾段（不足 16 列）：逐元素标量搬移。
    fn pack_tail(
        src: &[f32],
        dst: &mut [f32],
        k: usize,
        rowbase: usize,
        dstbase: usize,
        mut kk0: usize,
    ) {
        while kk0 < k {
            for r in 0..MR {
                dst[dstbase + kk0 * MR + r] = src[rowbase + r * k + kk0];
            }
            kk0 += 1;
        }
    }

    /// 计算一段连续行块：进入流式模式 → 逐行块逐列组外积累加 → 写回 → 退出流式模式。
    #[allow(unsafe_code)]
    fn compute_stage(a_stage: &[f32], b: &[f32], o_stage: &mut [f32], n: usize, k: usize) {
        let nblk = n / NR;
        let blocks = a_stage.len() / (k * MR);
        if blocks == 0 {
            return;
        }
        unsafe {
            enter_sm();
            for bi in 0..blocks {
                let a_p = a_stage.as_ptr().add(bi * k * MR);
                let c_base = o_stage.as_mut_ptr().add(bi * MR * n);
                for g in 0..nblk {
                    let b_p = b.as_ptr().add(g * NR);
                    sme_tile(k, a_p, b_p, c_base.add(g * NR), n * 4, n * 4);
                }
            }
            exit_sm();
        }
    }

    /// 16×16 外积微内核：`zero` → K 循环 `ld1w`+`fmopa` → 4 个 ZA 瓦片按切片号 `str za` 写回。
    #[target_feature(enable = "sme")]
    #[allow(unsafe_code)]
    #[inline(never)]
    unsafe fn sme_tile(
        k: usize,
        a_p: *const f32,
        b_p: *const f32,
        c: *mut f32,
        n_stride: usize,
        b_stride: usize,
    ) {
        let b_adv = b_stride - 3 * 64;
        unsafe {
            asm!(
            "zero {{ za0.s }}",
            "zero {{ za1.s }}",
            "zero {{ za2.s }}",
            "zero {{ za3.s }}",
            "2:",
            "ld1w {{ z0.s }}, p0/z, [{a}]",
            "add {a}, {a}, #64",
            "ld1w {{ z1.s }}, p0/z, [{b}]",
            "add {b}, {b}, #64",
            "ld1w {{ z2.s }}, p0/z, [{b}]",
            "add {b}, {b}, #64",
            "ld1w {{ z3.s }}, p0/z, [{b}]",
            "add {b}, {b}, #64",
            "ld1w {{ z4.s }}, p0/z, [{b}]",
            "add {b}, {b}, {b_adv}",
            "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
            "fmopa za1.s, p0/m, p0/m, z0.s, z2.s",
            "fmopa za2.s, p0/m, p0/m, z0.s, z3.s",
            "fmopa za3.s, p0/m, p0/m, z0.s, z4.s",
            "subs {k}, {k}, #1",
            "b.ne 2b",
            "mov {c16}, {c}",
            "add {c16}, {c16}, #64",
            "mov {c32}, {c}",
            "add {c32}, {c32}, #128",
            "mov {c48}, {c}",
            "add {c48}, {c48}, #192",
            "mov w12, #0",
            "mov w13, #1",
            "mov w14, #2",
            "mov w15, #3",
            "3:",
            "str za[w12, 0], [{c}]",
            "str za[w13, 0], [{c16}]",
            "str za[w14, 0], [{c32}]",
            "str za[w15, 0], [{c48}]",
            "add {c}, {c}, {nstride}",
            "add {c16}, {c16}, {nstride}",
            "add {c32}, {c32}, {nstride}",
            "add {c48}, {c48}, {nstride}",
            "add w12, w12, #4",
            "add w13, w13, #4",
            "add w14, w14, #4",
            "add w15, w15, #4",
            "cmp w12, #64",
            "b.ne 3b",
            a = inout(reg) a_p => _, b = inout(reg) b_p => _, c = inout(reg) c => _,
            k = inout(reg) k => _,
            nstride = in(reg) n_stride, b_adv = in(reg) b_adv,
            c16 = out(reg) _, c32 = out(reg) _, c48 = out(reg) _,
            out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _,
            out("w12") _, out("w13") _, out("w14") _, out("w15") _, out("p0") _,
            options(nostack),
            );
        }
    }

    /// 进入流式模式并准备全真谓词 `p0`；必须在任何 SME 指令前调用，且每线程一次。
    #[target_feature(enable = "sme")]
    #[allow(unsafe_code)]
    #[inline(never)]
    unsafe fn enter_sm() {
        unsafe {
            asm!("smstart", "smstart za", "ptrue p0.s", options(nostack));
        }
    }

    /// 退出流式模式；与 [`enter_sm`] 成对，`sme_tile` 之间不得夹带 NEON。
    #[target_feature(enable = "sme")]
    #[allow(unsafe_code)]
    #[inline(never)]
    unsafe fn exit_sm() {
        unsafe {
            asm!("smstop za", "smstop", options(nostack));
        }
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel_sme_tests.rs");
    }

    /// SME 端口峰值探针：`rounds` 轮、每轮 4 条 `fmopa`，`z0`/`z1` 与 ZA 全部寄存器常驻，
    /// **内层零 load、零 store**——把「数据从哪来」整个删掉，剩下的才是这块单元每秒能吃多少外积。
    ///
    /// 每轮 4 × 16 × 16 = 1024 次乘加（`p0.s` 16 条泳道，`ZA.s` 瓦片 16×16），即 2048 FLOP；
    /// 瓦片尺寸与 [`MR`]/[`NR`] 同源，由 `sme_tests.rs` 的整铺用例钉死。
    ///
    /// 这是把 GF/s 换算成「端口峰值百分比」的分母：GEMM 每轮还要搬 320 B，搬不动与算不动是两件事，
    /// 不分开量就只能猜瓶颈（F18）。
    ///
    /// 本机无 SME 或 `rounds == 0` 时返回 `false` 且不执行任何 SME 指令——安全入口，调用方据此跳过。
    /// 只在 `bench` 特性下编译，生产公开面不含本项。
    #[cfg(feature = "bench")]
    pub fn port_run(rounds: usize) -> bool {
        if rounds == 0 || !available() {
            return false;
        }
        #[allow(unsafe_code)]
        unsafe {
            port_rounds(rounds);
        }
        true
    }

    /// 纯外积循环：`smstart` → `rounds` 轮 4 条 `fmopa` → `smstop`，不碰任何内存。
    #[cfg(feature = "bench")]
    #[target_feature(enable = "sme")]
    #[allow(unsafe_code)]
    #[inline(never)]
    unsafe fn port_rounds(rounds: usize) {
        unsafe {
            asm!(
                "smstart",
                "smstart za",
                "ptrue p0.s",
                "fmov z0.s, #1.0",
                "fmov z1.s, #1.0",
                "zero {{ za0.s }}",
                "zero {{ za1.s }}",
                "zero {{ za2.s }}",
                "zero {{ za3.s }}",
                "2:",
                "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
                "fmopa za1.s, p0/m, p0/m, z0.s, z1.s",
                "fmopa za2.s, p0/m, p0/m, z0.s, z1.s",
                "fmopa za3.s, p0/m, p0/m, z0.s, z1.s",
                "subs {k}, {k}, #1",
                "b.ne 2b",
                "smstop za",
                "smstop",
                k = inout(reg) rounds => _,
                out("z0") _, out("z1") _, out("p0") _,
                options(nostack),
            );
        }
    }

    /// 取数混合探针：每轮 **1 条 `a` 载入 + 4 条 `b` 载入 + 4 条 `fmopa`**，与 [`sme_tile`] 内层
    /// 的指令配比逐条同构（16×64 瓦片即 `a` 16 个 f32 = 64 B、`b` 64 个 f32 = 256 B）。
    ///
    /// 与 [`port_run`] 的差就是「取数代价」：两者 `fmopa` 条数相同，只差这 5 条 load。
    /// 地址**固定不推进**，即每一轮都命中同一批缓存行——这是取数端口的**上限**，真实内核走不同行
    /// 只会更慢，所以据此算出的取数代价偏小、缺口偏小，拿它下结论不夸大（A3 保守值）。
    ///
    /// 本机无 SME 或 `rounds == 0` 时返回 `false` 且不执行任何 SME 指令。只在 `bench` 特性下编译。
    #[cfg(feature = "bench")]
    pub fn port_mix(rounds: usize, a: &[f32; 16], b: &[f32; 64]) -> bool {
        if rounds == 0 || !available() {
            return false;
        }
        #[allow(unsafe_code)]
        unsafe {
            mix_rounds(rounds, a.as_ptr(), b.as_ptr());
        }
        true
    }

    /// 固定地址的外积 + 载入循环：`smstart` → `rounds` 轮 → `smstop`，读边界由调用方的数组长度保证。
    #[cfg(feature = "bench")]
    #[target_feature(enable = "sme")]
    #[allow(unsafe_code)]
    #[inline(never)]
    unsafe fn mix_rounds(rounds: usize, a: *const f32, b: *const f32) {
        let (b1, b2, b3) = unsafe { (b.add(16), b.add(32), b.add(48)) };
        unsafe {
            asm!(
                "smstart",
                "smstart za",
                "ptrue p0.s",
                "zero {{ za0.s }}",
                "zero {{ za1.s }}",
                "zero {{ za2.s }}",
                "zero {{ za3.s }}",
                "2:",
                "ld1w {{ z0.s }}, p0/z, [{a}]",
                "ld1w {{ z1.s }}, p0/z, [{b}]",
                "ld1w {{ z2.s }}, p0/z, [{b1}]",
                "ld1w {{ z3.s }}, p0/z, [{b2}]",
                "ld1w {{ z4.s }}, p0/z, [{b3}]",
                "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
                "fmopa za1.s, p0/m, p0/m, z0.s, z2.s",
                "fmopa za2.s, p0/m, p0/m, z0.s, z3.s",
                "fmopa za3.s, p0/m, p0/m, z0.s, z4.s",
                "subs {k}, {k}, #1",
                "b.ne 2b",
                "smstop za",
                "smstop",
                a = in(reg) a, b = in(reg) b, b1 = in(reg) b1, b2 = in(reg) b2, b3 = in(reg) b3,
                k = inout(reg) rounds => _,
                out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("p0") _,
                options(nostack),
            );
        }
    }
}

use pack::{pack_b_f32_into, pack_b_f32_par_into, pack_panels_into};
use pool::{Bufs, Shape, pool, run_pooled};

/// 端口峰值探针只在测量容器里导出（`bench` 特性），生产构建看不到它。
#[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
pub use sme::{port_mix, port_run};

mod tn {
    //! 线性层布局（TN）的稠密 f32 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
    use super::pack::pack_panel;
    use super::pool::{Bufs, Shape, pool, run_pooled};
    use super::{Layout, MR, NR, threads_override};

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
    fn pack_tn_into(bpack: &mut [f32], w: &[f32], n: usize, k: usize) {
        let npad = n.div_ceil(NR) * NR;
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
        let groups = npad / NR;
        let mut g = slot;
        while g < groups {
            pack_panel(w, buf, n, k, g * NR, g * k * NR, true);
            g += p;
        }
    }

    /// 一段行的计算：A 面板打包 + 按 [`KB`] 分块扫列 + 行残尾。
    ///
    /// 正文与父模块的 Nt 视角**共用一份**（`fast::compute_rows`），这里只钉住 `Layout::Tn`。
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
        include!("../tests/kernel_tn_tests.rs");
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
    #[cfg(all(nightly, target_arch = "aarch64"))]
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
                s.spawn(move || compute_gemm(a_t, bp, b, rows, n, k, block));
            }
        });
    });
}

/// 单线程稠密 f32 GEMM（基准确认内核质量用）：`out[m,n] = a[m,k] · b[k,n]`。
pub fn gemm_f32_single(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
    let npad = n.div_ceil(NR) * NR;
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

mod i8 {
    //! 职责：i8 sdot 微内核：MR×NR 打包与 dotprod 探测回退。
    // i8 sdot 通道：MR=4 × NR=4，B 转置打包成「每列 k 连续」。
    // 每个 k16 一次 4 行 A 加载 + 4 次 B 加载 + 16 次 `vdotq_s32`（256 MAC/24 指令），
    // 且每块 B 瓦片被 4 行复用——B 访存比 MR=2/NR=8 再降一半。
    // dotprod 运行时探测，缺失走标量回退；`unsafe` 仅限 aarch64 微内核。
    // ---------------------------------------------------------------------------

    /// i8 微内核行数（四行共享同一 B 向量，摊薄 B 加载）。
    use std::thread;

    use diagnostic::{Code, Error, Result};

    use super::compute_gemm;
    use super::pack::b_groups;
    use super::pool::{Bufs, Job, Shape, pool, run_pooled};
    use super::tn;

    pub(super) const I8_MR: usize = 4;

    /// 行块高度：让 A 的 `rc × k` 字节留在 L1（约 128KiB），否则每个列块都要从 L2 重拉整块 A。
    ///
    /// 「留住」这个假设已被实测证伪：`rc` 扫 112 / 64 / 32 与默认 128 在噪声内无差
    /// （默认 309.1、366.7 Gelem/s；`rc=112` 304.4、346.4），见 `docs/lld.md` v3.0.40。
    fn row_chunk(k: usize) -> usize {
        ((1usize << 17) / k.max(1)).clamp(I8_MR, 256)
    }

    /// i8 微内核列数（4 个 int32x4 累加器 × 4 行 = 16 寄存器，避开溢出）。
    pub(super) const I8_NR: usize = 4;

    /// 线程数：`LLM_THREADS` 环境变量优先，否则全核。
    pub(super) fn threads_override() -> usize {
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

    /// 打包阶段：本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包。
    pub(super) fn i8_pack(j: &Job, slot: usize) {
        let p = j.p;
        let groups = j.npad / I8_NR;
        #[allow(unsafe_code)]
        unsafe {
            let b = j.b as *const i8;
            let bp = j.buf as *mut i8;
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
        let npad = n.div_ceil(I8_NR) * I8_NR;
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
        if rounds == 0
            || !block_shaped(a, b, k)
            || !std::arch::is_aarch64_feature_detected!("dotprod")
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
        use super::super::*;
        use super::I8_NR;

        pub(super) fn pack_b(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
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
        pub(super) fn pack_b_par(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
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
        pub(super) fn pack_b_span(chunk: &mut [i8], b: &[i8], n: usize, k: usize, blk0: usize) {
            let blocks = chunk.len() / (I8_NR * k);
            for bi in 0..blocks {
                let tile = &mut chunk[bi * I8_NR * k..(bi + 1) * I8_NR * k];
                pack_b_tile(tile, b, n, k, (blk0 + bi) * I8_NR);
            }
        }

        /// 一组 8 列瓦片的转置写：`b`（行主、`n` 列）的第 `first..first+8` 列搬到 `dst`（列主块内偏移）。
        /// 越界的列（`n` 不是 8 的倍数）留 0——打包缓冲本来就按 `npad` 补齐，乘加时该列权重为 0。
        pub(super) fn pack_b_tile(dst: &mut [i8], b: &[i8], n: usize, k: usize, first: usize) {
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
    }
    use kernel::compute_i8_packed;
    use pack::{pack_b, pack_b_par};
}
mod micro {
    //! 职责：微内核整行块扫描的只读侧（Sweep）。
    use super::{MR, NR};

    /// 一次整行块扫描的只读侧：指针、面板与三个尺寸一次带走（实参不散成一长串）。
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

    /// 一个待算的 8×8 微内核块：左上角 `(row0, j)`、矩阵列宽 `n`、内积分块长 `kb`、是否首块。
    #[derive(Clone, Copy)]
    pub(super) struct Tile {
        pub(super) row0: usize,
        pub(super) j: usize,
        pub(super) n: usize,
        pub(super) kb: usize,
        pub(super) first: bool,
    }

    /// 打包面板上的 NEON 8×8 微内核（A 面板 `[p][8]`、B 面板 `[p][8]`，一次算 8 行 8 列）。
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    pub(super) fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
        let Tile {
            row0,
            j,
            n,
            kb,
            first,
        } = tile;
        use std::arch::aarch64::*;
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let pp = panel.as_ptr();
            let bp = b.as_ptr();
            // 后续内积分块接着上一块的累加器算：首块用 0 起手，否则读回 `out` 的 16 个向量。
            let mut acc = if first {
                [zero; 16]
            } else {
                micro_load(out, row0, n, j)
            };
            fma_step(&mut acc, pp, bp, kb);
            micro_store(out, row0, n, j, acc);
        }
    }

    /// 一个收缩步的 8×8 瓦片体：4 条 `vld1q` 取数 + 16 条 lane-indexed FMA。
    ///
    /// 生产微内核 [`micro_packed`] 与取数受限端口探针 [`fma_block_rounds`] 共用**这一份**——
    /// 探针的全部意义是「与生产逐指令一致」，两份写法必然漂移（N5）。`pp` 是 A 面板每步
    /// `MR` 个 f32、`bp` 是 B 面板每步 `NR` 个 f32，两者步距都是 8 即 32 字节。
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    #[inline]
    unsafe fn fma_step(
        acc: &mut [std::arch::aarch64::float32x4_t; 16],
        pp: *const f32,
        bp: *const f32,
        kb: usize,
    ) {
        use std::arch::aarch64::*;
        unsafe {
            for p in 0..kb {
                let av0 = vld1q_f32(pp.add(p * MR));
                let av1 = vld1q_f32(pp.add(p * MR + 4));
                let bv0 = vld1q_f32(bp.add(p * NR));
                let bv1 = vld1q_f32(bp.add(p * NR + 4));
                // lane-indexed FMA 把广播并进乘加：每条 `p` 从 24 条指令降到 20 条（省 8 条 dup）。
                acc[0] = vfmaq_laneq_f32::<0>(acc[0], bv0, av0);
                acc[1] = vfmaq_laneq_f32::<0>(acc[1], bv1, av0);
                acc[2] = vfmaq_laneq_f32::<1>(acc[2], bv0, av0);
                acc[3] = vfmaq_laneq_f32::<1>(acc[3], bv1, av0);
                acc[4] = vfmaq_laneq_f32::<2>(acc[4], bv0, av0);
                acc[5] = vfmaq_laneq_f32::<2>(acc[5], bv1, av0);
                acc[6] = vfmaq_laneq_f32::<3>(acc[6], bv0, av0);
                acc[7] = vfmaq_laneq_f32::<3>(acc[7], bv1, av0);
                acc[8] = vfmaq_laneq_f32::<0>(acc[8], bv0, av1);
                acc[9] = vfmaq_laneq_f32::<0>(acc[9], bv1, av1);
                acc[10] = vfmaq_laneq_f32::<1>(acc[10], bv0, av1);
                acc[11] = vfmaq_laneq_f32::<1>(acc[11], bv1, av1);
                acc[12] = vfmaq_laneq_f32::<2>(acc[12], bv0, av1);
                acc[13] = vfmaq_laneq_f32::<2>(acc[13], bv1, av1);
                acc[14] = vfmaq_laneq_f32::<3>(acc[14], bv0, av1);
                acc[15] = vfmaq_laneq_f32::<3>(acc[15], bv1, av1);
            }
        }
    }

    /// 一个收缩步的乘加次数：16 条 FMA × 每向量 4 个 f32 通道。以 `k` 为单位命名，
    /// 因为取数受限探针按 `k` 步展开，纯端口探针正好一步。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    pub const FMA_MAC_PER_K: usize = 16 * 4;

    /// f32 FMA 端口峰值：`rounds` 步 × 16 条 lane-indexed FMA，**环内零 load 零 store**。
    ///
    /// 这是 K7 分母里「理想上限」那一半，指令形状与生产瓦片体逐条一致（同 16 个累加器、
    /// 同 lane-indexed FMA），只差那 4 条 `vld1q`。`machine_peak` 虽也是 f32 峰值，但它是
    /// 自动向量化的标量循环（4 条链 × 16 lane），链数与寄存器分配都不同于手写瓦片，拿它当
    /// 「瓦片体的端口上限」是口径混用；本探针才是同一个形状。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    pub fn port_fma(rounds: usize) -> bool {
        if rounds == 0 {
            return false;
        }
        // 安全性：aarch64 基线含 NEON，函数体内只有寄存器运算，不触碰内存。
        #[allow(unsafe_code)]
        unsafe {
            fma_pure_rounds(rounds);
        }
        true
    }

    /// 纯 FMA 循环：16 个累加器互不依赖，结果进 `black_box` 防整段消除。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    #[allow(unsafe_code)]
    #[inline(never)]
    unsafe fn fma_pure_rounds(rounds: usize) {
        use std::arch::aarch64::*;
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); 16];
            let av = vdupq_n_f32(1.000_000_1);
            for _ in 0..rounds {
                // 16 个累加器互不依赖：每条只读自己，不构成串行链。
                for chain in &mut acc {
                    *chain = vfmaq_f32(*chain, av, av);
                }
            }
            std::hint::black_box(acc);
        }
    }

    /// `fma_step` 的形状校验：`k` 非零，A/B 各留足 `NR * k` 个 f32。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    fn block_shaped(a: &[f32], b: &[f32], k: usize) -> bool {
        k != 0 && a.len() >= NR * k && b.len() >= NR * k
    }

    /// f32 **取数受限**端口：与生产瓦片体逐指令一致，每步 4 条 `vld1q` + 16 条 FMA。
    ///
    /// **为什么要它**：纯端口探针环内零 load，是理想上限；生产核每步要发 4 条 load。
    /// 两者之差就是取数代价，也是「K7 的缺口里多少是结构注定、多少是核外记账」的分界线。
    /// K8 的同一对照给出 i8 瓦片体单线程 99.3%、18 线程 84.1%，故那 84% 就是瓦片体的真实天花板。
    ///
    /// `k` 取 1024 使地址在环内单调推进——地址不随循环变化时 LLVM 会把 load 整体提出循环，
    /// 那样测的是纯端口而非「取数受限」端口（i8 侧第一版就是这么错的，见 [`super::i8::port_sdot_block`]）。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    pub fn port_fma_block(rounds: usize, a: &[f32], b: &[f32], k: usize) -> bool {
        if rounds == 0 || !block_shaped(a, b, k) {
            return false;
        }
        // 安全性：A/B 各 `NR * k` 个 f32 已在上面校验。
        #[allow(unsafe_code)]
        unsafe {
            fma_block_rounds(rounds, a.as_ptr(), b.as_ptr(), k);
        }
        true
    }

    /// 取数受限循环：每轮跑一次完整瓦片体，**零记账**（不读回 `out`、不写回 `out`）。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    #[allow(unsafe_code)]
    #[inline(never)]
    unsafe fn fma_block_rounds(rounds: usize, ap: *const f32, bp: *const f32, k: usize) {
        use std::arch::aarch64::*;
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); 16];
            for _ in 0..rounds {
                fma_step(&mut acc, ap, bp, k);
            }
            std::hint::black_box(acc);
        }
    }

    /// 读回上一步的 16 个累加向量（非首块续算用）。
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    pub(super) fn micro_load(
        out: &[f32],
        row0: usize,
        n: usize,
        j: usize,
    ) -> [std::arch::aarch64::float32x4_t; 16] {
        unsafe {
            use std::arch::aarch64::*;
            let mut acc = [vdupq_n_f32(0.0); 16];
            let prev = out.as_ptr();
            let r = row0;
            acc[0] = vld1q_f32(prev.add(r * n + j));
            acc[1] = vld1q_f32(prev.add(r * n + j + 4));
            acc[2] = vld1q_f32(prev.add((r + 1) * n + j));
            acc[3] = vld1q_f32(prev.add((r + 1) * n + j + 4));
            acc[4] = vld1q_f32(prev.add((r + 2) * n + j));
            acc[5] = vld1q_f32(prev.add((r + 2) * n + j + 4));
            acc[6] = vld1q_f32(prev.add((r + 3) * n + j));
            acc[7] = vld1q_f32(prev.add((r + 3) * n + j + 4));
            acc[8] = vld1q_f32(prev.add((r + 4) * n + j));
            acc[9] = vld1q_f32(prev.add((r + 4) * n + j + 4));
            acc[10] = vld1q_f32(prev.add((r + 5) * n + j));
            acc[11] = vld1q_f32(prev.add((r + 5) * n + j + 4));
            acc[12] = vld1q_f32(prev.add((r + 6) * n + j));
            acc[13] = vld1q_f32(prev.add((r + 6) * n + j + 4));
            acc[14] = vld1q_f32(prev.add((r + 7) * n + j));
            acc[15] = vld1q_f32(prev.add((r + 7) * n + j + 4));
            acc
        }
    }

    /// 把 16 个累加向量写回 `out` 的 8×8 块。
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    pub(super) fn micro_store(
        out: &mut [f32],
        row0: usize,
        n: usize,
        j: usize,
        acc: [std::arch::aarch64::float32x4_t; 16],
    ) {
        unsafe {
            use std::arch::aarch64::*;
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

    /// 标量微内核：无 SIMD 的架构（以及 x86_64 上未检出 AVX2/FMA 的机器）的兜底。
    #[cfg(not(target_arch = "aarch64"))]
    pub(super) fn micro_packed_scalar(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
        let Tile {
            row0,
            j,
            n,
            kb,
            first,
        } = tile;
        for r in 0..MR {
            for c in 0..NR {
                let slot = (row0 + r) * n + j + c;
                let mut acc = if first { 0.0f32 } else { out[slot] };
                for p in 0..kb {
                    acc += panel[p * MR + r] * b[p * NR + c];
                }
                out[slot] = acc;
            }
        }
    }

    /// x86_64 的 8×8 微内核：AVX2 + FMA，8 个 `__m256` 累加器（每行一个、8 列一个向量）。
    #[cfg(target_arch = "x86_64")]
    pub(super) fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            micro_packed_avx2(panel, b, out, tile);
        } else {
            micro_packed_scalar(panel, b, out, tile);
        }
    }

    /// 8×8 AVX2+FMA 微内核：每行一个 `__m256` 累加器，逐收缩步广播 A 值 × B 向量做 FMA。
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    pub(super) fn micro_packed_avx2(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
        let Tile {
            row0,
            j,
            n,
            kb,
            first,
        } = tile;
        use std::arch::x86_64::*;
        unsafe {
            let zero = _mm256_setzero_ps();
            let pp = panel.as_ptr();
            let bp = b.as_ptr();
            let mut acc = if first {
                [zero; 8]
            } else {
                micro_load(out, row0, n, j)
            };
            for p in 0..kb {
                let bv = _mm256_loadu_ps(bp.add(p * NR));
                // 手写 FMA 微内核：下标形式比迭代器形式更贴近指令序列，便于逐条核对。
                #[allow(clippy::needless_range_loop)]
                for r in 0..MR {
                    acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(*pp.add(p * MR + r)), bv, acc[r]);
                }
            }
            micro_store(out, row0, n, j, acc);
        }
    }

    /// 读回上一步的 8 个累加向量（x86_64 非首块续算用）。
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    pub(super) fn micro_load(
        out: &[f32],
        row0: usize,
        n: usize,
        j: usize,
    ) -> [std::arch::x86_64::__m256; 8] {
        unsafe {
            use std::arch::x86_64::*;
            let prev = out.as_ptr();
            let r = row0;
            [
                _mm256_loadu_ps(prev.add(r * n + j)),
                _mm256_loadu_ps(prev.add((r + 1) * n + j)),
                _mm256_loadu_ps(prev.add((r + 2) * n + j)),
                _mm256_loadu_ps(prev.add((r + 3) * n + j)),
                _mm256_loadu_ps(prev.add((r + 4) * n + j)),
                _mm256_loadu_ps(prev.add((r + 5) * n + j)),
                _mm256_loadu_ps(prev.add((r + 6) * n + j)),
                _mm256_loadu_ps(prev.add((r + 7) * n + j)),
            ]
        }
    }

    /// 把 8 个累加向量写回 `out` 的 8×8 块（x86_64）。
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    pub(super) fn micro_store(
        out: &mut [f32],
        row0: usize,
        n: usize,
        j: usize,
        acc: [std::arch::x86_64::__m256; 8],
    ) {
        unsafe {
            use std::arch::x86_64::*;
            let op = out.as_mut_ptr();
            let r = row0;
            for (i, v) in acc.into_iter().enumerate() {
                _mm256_storeu_ps(op.add((r + i) * n + j), v);
            }
        }
    }

    /// 无 SIMD 的架构：直接走标量微内核（`micro_packed_scalar` 在这些目标上不会被别处引用）。
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    pub(super) fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
        micro_packed_scalar(panel, b, out, tile);
    }

    /// 一个待算的标量尾块：左上角 `(i, j)`、覆盖 `mr × nr`，矩阵列宽 `n`、内积长 `k`。
    #[derive(Clone, Copy)]
    pub(super) struct Span {
        pub(super) i: usize,
        pub(super) j: usize,
        pub(super) mr: usize,
        pub(super) nr: usize,
        pub(super) n: usize,
        pub(super) k: usize,
        /// `a` 的整行宽：内积分块时 `k` 只是当前块，跨块偏移要按整行宽算。
        pub(super) stride: usize,
    }

    /// 权重布局：`Nt` 是收缩轴在前的 `b[k][n]`，`Tn` 是按输出行常驻的 `w[n][k]`。
    #[derive(Clone, Copy)]
    pub(super) enum Layout {
        Nt,
        Tn,
    }

    /// 权重在 `w` 里的寻址三元组：`(基址, 内积步长, 列步长)`——两种布局在这一处收敛。
    pub(super) fn weight_stride(span: Span, k0: usize, layout: Layout) -> (usize, usize, usize) {
        match layout {
            Layout::Nt => (k0 * span.n + span.j, span.n, 1),
            Layout::Tn => (span.j * span.stride + k0, 1, span.stride),
        }
    }

    /// 标量尾块：`out[i..i+mr, j..j+nr]` 逐元素点积（与微内核同一组累加口径，容差内一致）。
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

/// i8 `sdot` 端口峰值探针（K8 的分母），只在测量容器里导出。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
pub use i8::{SDOT_BLOCK_MAC_PER_K, SDOT_MAC_PER_ROUND, port_sdot, port_sdot_block};

/// 端口探针清单：哪几条指令路径可测，由这里统一回答（调用方不写 `cfg`）。
#[cfg(feature = "bench")]
mod probe {
    //! 职责：端口峰值探针清单——哪几条指令路径在本机可测，由这里统一回答。

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

    /// 纯 f32 FMA 端口：16 个累加器、零 load，与生产瓦片体同形状（K7 的理想上限）。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    const FMA_PURE: Probe = Probe {
        group: "fma_peak",
        suffix: "",
        ops_per_round: 2 * super::micro::FMA_MAC_PER_K,
        run: super::micro::port_fma,
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

    /// 取数受限的 f32 端口：每步 2 条 A load + 2 条 B load 配 16 条 lane-indexed FMA，
    /// 与生产瓦片体 [`super::micro::fma_step`] 逐指令一致。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    const FMA_BLOCK: Probe = Probe {
        group: "fma_peak",
        suffix: "_block",
        ops_per_round: 2 * super::micro::FMA_MAC_PER_K * PROBE_K,
        run: |rounds| {
            static A: [f32; super::NR * PROBE_K] = [1.0; super::NR * PROBE_K];
            static B: [f32; super::NR * PROBE_K] = [1.0; super::NR * PROBE_K];
            super::micro::port_fma_block(rounds, &A, &B, PROBE_K)
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
    /// 非 aarch64 没有 `sdot` 微内核，一条探针都没有。
    #[cfg(all(feature = "bench", not(target_arch = "aarch64")))]
    const PROBES: &[Probe] = &[];
}
#[cfg(feature = "bench")]
pub use probe::{Probe, probes};

use i8::{f32_calc, f32_pack, threads_override};
pub use i8::{gemm_i8, gemm_i8_checked, gemm_i8_single};
use micro::{Layout, Span, Sweep, Tile, edge, micro_packed};

#[cfg(test)]
mod tests {
    include!("../tests/kernel_fast_tests.rs");
}

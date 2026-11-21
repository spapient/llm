//! 职责：常驻 GEMM 线程池——任务只发布裸指针 + 一次闸门，spawn 成本归零。
//!
//! 每次调用都 `thread::scope` 会产生「打包一次 + 计算一次」两拨线程创建；小形状
//! （如 512×512×512，0.27 GFLOP）里这个固定开销能占到近半。这里改成：进程内
//! 常驻 worker，任务只发布指针 + 一次闸门，spawn 成本归零。
//! `unsafe` 仅用于把本次调用的裸指针交给 worker（主线程自旋等全部完成才返回）。
//! worker 的任何 panic 都被就地隔离：记录后中止闸门，其余参与者与发布者都能退出，
//! 常驻线程不丢、槽位不泄，panic 最终由主线程 `resume_unwind` 抛回调用方。

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
///
/// `std::sync::Barrier` 要求 `p` 个参与者**全部**到齐才放行——若某个 worker 在打包段 panic，
/// 它到不了闸门，其余 `p-1` 人永久阻塞。这里改成：panic 处理方调 `abort()` 广播放行，
/// `wait` 返回「本次是否被中止」，被中止的参与者跳过 `calc`（结果反正要被丢弃）。
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

/// 常驻池（首次调用时创建 `cpu-1` 个常驻 worker，之后只唤醒不复建）。
pub(super) fn pool() -> &'static Pool {
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
                let g = pool.free_lock.lock().unwrap_or_else(|e| e.into_inner());
                let _g = pool.free_cv.wait(g).unwrap_or_else(|e| e.into_inner());
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

/// 池化执行；参与数不足 2 时返回 `false`，由调用方走 scope 兜底。
#[allow(unsafe_code)]
pub(super) fn run_pooled(
    pool: &'static Pool,
    a: *const u8,
    b: *const u8,
    buf: *mut u8,
    out: *mut u8,
    shape: Shape,
    mr: usize,
    t: usize,
    stage: fn(&Job, usize),
    calc: fn(&Job, usize),
) -> bool {
    let Shape { m, n, k, npad } = shape;
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
        rows_per,
        p,
        stage,
        calc,
        gate: Gate::new(),
        done: AtomicUsize::new(0),
        panic: Mutex::new(None),
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

    // 主线程自己的分片（slot 0）也纳入恐慌隔离：主线程 panic 若不中止闸门，其余 worker
    // 会卡在闸门上；记录后与 worker panic 走同一条收尾路径。
    if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| part(&job, 0))) {
        job.panic
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert(payload);
        job.gate.abort();
    }
    while job.done.load(Ordering::Acquire) < p - 1 {
        std::hint::spin_loop();
    }
    // 等全部槽归还后再把 panic 抛回调用方：此刻没有 worker 再碰 job 的裸指针。
    if let Some(payload) = job.panic.lock().unwrap_or_else(|e| e.into_inner()).take() {
        std::panic::resume_unwind(payload);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    /// 造一个专用池（固定 worker 数，与机器核数无关），测完即弃。
    fn build_pool(n: usize) -> &'static Pool {
        let cells: Vec<Cell> = (0..n)
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
        for id in 0..n {
            if thread::Builder::new()
                .name(format!("llm-gemm-test-{id}"))
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

    /// 以最小几何发布一次任务；`stage`/`calc` 由测试注入（可为 panic / no-op）。
    fn run(pool: &'static Pool, stage: fn(&Job, usize), calc: fn(&Job, usize), t: usize) -> bool {
        let a = vec![0u8; 64];
        let b = vec![0u8; 64];
        let mut buf = vec![0u8; 64];
        let mut out = vec![0u8; 64];
        let shape = Shape {
            m: 1,
            n: 1,
            k: 1,
            npad: 1,
        };
        run_pooled(
            pool,
            a.as_ptr(),
            b.as_ptr(),
            buf.as_mut_ptr(),
            out.as_mut_ptr(),
            shape,
            1,
            t,
            stage,
            calc,
        )
    }

    fn noop(_: &Job, _slot: usize) {}

    fn panic_in_worker(_: &Job, slot: usize) {
        if slot == 1 {
            panic!("注入：worker 分片 panic");
        }
    }

    fn panic_in_caller(_: &Job, slot: usize) {
        if slot == 0 {
            panic!("注入：调用方分片 panic");
        }
    }

    /// D2：worker 在打包段 panic → 其余参与者不被闸门卡死，panic 抛回调用方；
    /// D3：panicked worker 的槽位被归还，池仍可复用。
    fn assert_panic_propagates_and_pool_survives(stage: fn(&Job, usize), calc: fn(&Job, usize)) {
        let p = build_pool(3);
        assert!(
            p.live.load(Ordering::Acquire) >= 1,
            "本机需至少能起 1 个常驻 worker 才有可测的 panic 隔离面"
        );
        let caught =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(p, stage, calc, 4)));
        assert!(caught.is_err(), "参与方 panic 必须抛回调用方，不得静默吞掉");
        let again =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(p, noop, noop, 4)));
        assert!(
            matches!(again, Ok(true)),
            "panic 后所有槽位必须已归还，池可复用（否则槽位泄漏）"
        );
    }

    #[test]
    fn worker_stage_panic_propagates_and_pool_survives() {
        assert_panic_propagates_and_pool_survives(panic_in_worker, noop);
    }

    #[test]
    fn worker_calc_panic_propagates_and_pool_survives() {
        assert_panic_propagates_and_pool_survives(noop, panic_in_worker);
    }

    #[test]
    fn caller_panic_releases_workers_and_pool_survives() {
        assert_panic_propagates_and_pool_survives(panic_in_caller, noop);
    }
}

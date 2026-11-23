use super::*;
use std::sync::atomic::Ordering;

/// 造一个专用池（固定 worker 数，与机器核数无关），测完即弃。
fn build_pool(n: usize) -> &'static Pool {
    spawn_pool(make_cells(n), "llm-gemm-test")
}

/// 以最小几何发布一次任务；`stage`/`calc` 由测试注入（可为 panic / no-op）。
fn run(pool: &'static Pool, stage: fn(&Job, usize), calc: fn(&Job, usize), t: usize) -> bool {
    let a = [0u8; 64];
    let b = [0u8; 64];
    let mut buf = [0u8; 64];
    let mut out = [0u8; 64];
    let shape = Shape {
        m: 1,
        n: 1,
        k: 1,
        npad: 1,
        tn: false,
    };
    run_pooled(
        pool,
        Bufs {
            a: a.as_ptr(),
            b: b.as_ptr(),
            buf: buf.as_mut_ptr(),
            out: out.as_mut_ptr(),
        },
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
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(p, stage, calc, 4)));
    assert!(caught.is_err(), "参与方 panic 必须抛回调用方，不得静默吞掉");
    let again = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(p, noop, noop, 4)));
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

/// 并发发布时「归还 → 唤醒」不得丢。
///
/// 两个调用方同时要满池：后者必然睡在 `free_cv` 上，等前者归还。若唤醒丢失，它会一直睡到
/// 下一次任务结束——单次任务下就是永久等待。用带超时的收信把「挂死」判成「失败」，不让
/// 门禁被一个死锁拖住（这正是 `lease_free` 拿锁后重查位图要挡的那条路径）。
#[test]
fn concurrent_callers_do_not_miss_wakeups() {
    let p = build_pool(4);
    let live = p.live.load(Ordering::Acquire);
    if live < 2 {
        return; // 起不到两个 worker 就没有可争的槽，跳过而不是误报
    }
    let t = live + 1;
    let (tx, rx) = std::sync::mpsc::channel();
    for id in 0..2 {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for _ in 0..40 {
                assert!(run(p, noop, noop, t), "池可用时必须走池化路径");
            }
            let _ = tx.send(id);
        });
    }
    drop(tx);
    for _ in 0..2 {
        assert!(
            rx.recv_timeout(std::time::Duration::from_secs(20)).is_ok(),
            "并发调用方在 20s 内未全部返回：唤醒丢失，等槽者卡在 condvar 上"
        );
    }
}

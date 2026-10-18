//! Deterministic micro-benchmark harness (no criterion).
//!
//! Each benchmark case is a closure run repeatedly under [`Instant`] timing;
//! the iteration count adapts to a target wall time and the reported number is
//! the median per-iteration duration, which is robust to scheduler noise. The
//! harness reports one repeatable number per case — the repository's
//! acceptance criteria require measured evidence, not statistical inference,
//! so no confidence-interval machinery is imported (docs/engineering.md 性能
//! 工程: 区分延迟/吞吐, 记录基线, 调查噪声).
//!
//! Usage, from a benchmark executable under `benches/`:
//!
//! ```ignore
//! use ::bench::harness::{report, run};
//!
//! fn main() {
//!     report(&run("kernel/relu", || {
//!         black_box(cpu::relu(&x));
//!     }));
//! }
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

/// A measured benchmark case.
pub struct sample {
    /// Case name, printed verbatim.
    pub name: String,
    /// Median per-iteration duration.
    pub median: Duration,
    /// Number of iterations the median was taken over.
    pub iters: u64,
}

/// Times `f` over enough iterations to fill a 300 ms target, returning the
/// median per-iteration duration.
///
/// The name may be static or dynamically constructed (e.g. `format!` inside a
/// loop); it is owned by the returned sample.
///
/// One warm-up/estimate run precedes the timed loop; its cost is excluded from
/// the sample so first-call and cold-cache effects do not pollute the median.
pub fn run<R>(name: &str, f: impl FnMut() -> R) -> sample {
    run_target(name, Duration::from_millis(300), f)
}

/// Same as [`run`] with an explicit target wall time.
pub fn run_target<R>(name: &str, target: Duration, mut f: impl FnMut() -> R) -> sample {
    // Warm-up and per-iteration cost estimate.
    let t0 = Instant::now();
    black_box(f());
    let estimate = t0.elapsed();
    let target_nanos = target.as_nanos().max(1);
    let estimate_nanos = estimate.as_nanos().max(1);
    let iters = (target_nanos / estimate_nanos).clamp(1, 1_000_000) as u64;

    let mut times = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t0 = Instant::now();
        black_box(f());
        times.push(t0.elapsed());
    }
    times.sort_unstable();
    sample {
        name: name.to_string(),
        median: times[times.len() / 2],
        iters,
    }
}

/// Prints a sample in a stable, greppable format.
pub fn report(s: &sample) {
    println!("{}: {:?} ({} iters)", s.name, s.median, s.iters);
}

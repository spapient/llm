//! Relocated unit tests for the `context` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/context/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/context/src/budget.rs`.
mod budget {
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    use context::contexterror;
    use context::memorybudget;

    #[test]
    fn reserves_and_reports_accounting() {
        let budget = memorybudget::new(100);
        assert_eq!(budget.getcapacity(), 100);
        assert_eq!(budget.getavailable(), 100);
        let guard = budget.tryreserve(40).unwrap();
        assert_eq!(guard.getbytes(), 40);
        assert_eq!(budget.getreserved(), 40);
        assert_eq!(budget.getavailable(), 60);
        drop(guard);
        assert_eq!(budget.getreserved(), 0);
        assert_eq!(budget.getavailable(), 100);
    }

    #[test]
    fn rejects_requests_past_capacity() {
        let budget = memorybudget::new(100);
        assert_eq!(
            budget.tryreserve(101).map(|_| ()),
            Err(contexterror::budgetexceeded {
                requested: 101,
                reserved: 0,
                capacity: 100,
            })
        );
        assert_eq!(budget.getreserved(), 0);
    }

    #[test]
    fn accepts_request_equal_to_capacity() {
        let budget = memorybudget::new(64);
        let guard = budget.tryreserve(64).unwrap();
        assert_eq!(budget.getavailable(), 0);
        drop(guard);
        assert_eq!(budget.getavailable(), 64);
    }

    #[test]
    fn rejects_counter_overflow() {
        let budget = memorybudget::new(usize::MAX);
        let guard = budget.tryreserve(usize::MAX).unwrap();
        assert_eq!(
            budget.tryreserve(1).map(|_| ()),
            Err(contexterror::budgetoverflow)
        );
        drop(guard);
    }

    #[test]
    fn releases_on_error_paths() {
        let budget = memorybudget::new(100);
        let outcome = (|| {
            let _guard = budget.tryreserve(80)?;
            // Simulate a later failure: the guard is dropped on this early return.
            Err::<(), contexterror>(contexterror::cancelled)
        })();
        assert_eq!(outcome, Err(contexterror::cancelled));
        assert_eq!(budget.getreserved(), 0);
    }

    #[test]
    fn concurrent_reservations_never_exceed_capacity() {
        let budget = memorybudget::new(100);
        let barrier = Barrier::new(4);
        let successes = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let guard = budget.tryreserve(40);
                    if guard.is_ok() {
                        successes.fetch_add(1, Ordering::AcqRel);
                    }
                    // Hold every reservation until all four attempts have run.
                    barrier.wait();
                    drop(guard);
                });
            }
        });
        assert_eq!(successes.load(Ordering::Acquire), 2);
        assert_eq!(budget.getreserved(), 0);
    }
}

/// Tests relocated from `crates/context/src/cancel.rs`.
mod cancel {
    use context::cancellation;
    use context::contexterror;

    #[test]
    fn starts_uncancelled() {
        let handle = cancellation::new();
        assert!(!handle.iscancelled());
        assert_eq!(handle.checkpoint(), Ok(()));
    }

    #[test]
    fn cancel_propagates_to_all_clones() {
        let handle = cancellation::new();
        let observer = handle.clone();
        assert!(!observer.iscancelled());
        handle.cancel();
        assert!(observer.iscancelled());
        assert_eq!(observer.checkpoint(), Err(contexterror::cancelled));
    }

    #[test]
    fn cancel_is_idempotent() {
        let handle = cancellation::new();
        handle.cancel();
        handle.cancel();
        assert!(handle.iscancelled());
    }

    #[test]
    fn propagates_across_threads() {
        let handle = cancellation::new();
        let worker = handle.clone();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                while !worker.iscancelled() {
                    std::thread::yield_now();
                }
                assert_eq!(worker.checkpoint(), Err(contexterror::cancelled));
            });
            handle.cancel();
        });
    }
}

/// Tests relocated from `crates/context/src/error.rs`.
mod error {
    use context::contexterror;

    #[test]
    fn carries_structured_context() {
        assert_eq!(
            contexterror::stepbudgetexceeded { used: 2, max: 2 }.to_string(),
            "step budget exceeded: used 2, maximum 2"
        );
        assert_eq!(
            contexterror::budgetexceeded {
                requested: 40,
                reserved: 80,
                capacity: 100,
            }
            .to_string(),
            "memory budget exceeded: requested 40, already reserved 80, capacity 100"
        );
    }

    #[test]
    fn distinguishes_causes() {
        assert_ne!(contexterror::cancelled, contexterror::budgetoverflow);
        assert_ne!(
            contexterror::invalidconfig { detail: "a" },
            contexterror::invalidconfig { detail: "b" }
        );
    }
}

/// Tests relocated from `crates/context/src/execution.rs`.
mod execution {
    use std::sync::Arc;

    use context::collectingmetrics;
    use context::contexterror;
    use context::{executionconfig, executioncontext};
    use lib::requestid;

    fn config(maxsteps: usize) -> executionconfig {
        executionconfig::new(1024, 4, maxsteps).unwrap()
    }

    #[test]
    fn rejects_zero_workers() {
        assert_eq!(
            executionconfig::new(1024, 0, 10),
            Err(contexterror::invalidconfig {
                detail: "maxworkers must be at least 1",
            })
        );
    }

    #[test]
    fn exposes_configuration() {
        let config = executionconfig::new(2048, 2, 16).unwrap();
        assert_eq!(config.getmemorybytes(), 2048);
        assert_eq!(config.getmaxworkers(), 2);
        assert_eq!(config.getmaxsteps(), 16);
    }

    #[test]
    fn checkpoint_advances_and_enforces_step_budget() {
        let exec = executioncontext::new(requestid::new(1), config(2));
        assert_eq!(exec.getsteps(), 0);
        assert_eq!(exec.checkpoint(), Ok(()));
        assert_eq!(exec.checkpoint(), Ok(()));
        assert_eq!(exec.getsteps(), 2);
        assert_eq!(
            exec.checkpoint(),
            Err(contexterror::stepbudgetexceeded { used: 2, max: 2 })
        );
    }

    #[test]
    fn cancellation_takes_precedence_over_step_budget() {
        let exec = executioncontext::new(requestid::new(2), config(0));
        exec.getcancellation().cancel();
        assert_eq!(exec.checkpoint(), Err(contexterror::cancelled));
    }

    #[test]
    fn budget_is_sized_from_configuration() {
        let exec = executioncontext::new(requestid::new(3), config(10));
        assert_eq!(exec.getid(), requestid::new(3));
        assert_eq!(exec.getbudget().getcapacity(), 1024);
    }

    #[test]
    fn forwards_metrics_to_the_sink() {
        let sink = Arc::new(collectingmetrics::new());
        let exec = executioncontext::withmetrics(requestid::new(4), config(10), sink.clone());
        exec.recordcounter("steps", 1);
        exec.recordgauge("depth", -3);
        assert_eq!(sink.getcounter("steps"), Some(1));
        assert_eq!(sink.getgauge("depth"), Some(-3));
    }

    #[test]
    fn metrics_recording_is_optional() {
        let exec = executioncontext::new(requestid::new(5), config(10));
        // Must not panic when no sink is installed.
        exec.recordcounter("ignored", 1);
        exec.recordgauge("ignored", 1);
    }
}

/// Tests relocated from `crates/context/src/metric.rs`.
mod metric {
    use context::{collectingmetrics, metrics, noopmetrics};

    #[test]
    fn accumulates_counters_and_overwrites_gauges() {
        let sink = collectingmetrics::new();
        sink.recordcounter("tokens", 3);
        sink.recordcounter("tokens", 4);
        sink.recordgauge("depth", 5);
        sink.recordgauge("depth", 2);
        assert_eq!(sink.getcounter("tokens"), Some(7));
        assert_eq!(sink.getgauge("depth"), Some(2));
    }

    #[test]
    fn absent_metrics_are_none() {
        let sink = collectingmetrics::new();
        assert_eq!(sink.getcounter("missing"), None);
        assert_eq!(sink.getgauge("missing"), None);
    }

    #[test]
    fn noop_sink_records_nothing_visible() {
        let sink = noopmetrics;
        sink.recordcounter("tokens", 1);
        sink.recordgauge("depth", 1);
    }
}

/// Tests relocated from `crates/context/src/worker.rs`.
mod worker {
    use context::contexterror;
    use context::{executionconfig, executioncontext};
    use context::{workererror, workergroup};
    use lib::requestid;

    #[derive(Debug, PartialEq, Eq)]
    enum joberror {
        boom,
    }

    fn exec(maxsteps: usize) -> executioncontext {
        let config = executionconfig::new(1 << 20, 4, maxsteps).unwrap();
        executioncontext::new(requestid::new(1), config)
    }

    #[test]
    fn rejects_zero_workers() {
        assert_eq!(
            workergroup::new(0).map(|_| ()),
            Err(contexterror::invalidconfig {
                detail: "worker count must be at least 1",
            })
        );
    }

    #[test]
    fn runs_zero_jobs() {
        let group = workergroup::new(2).unwrap();
        let exec = exec(16);
        let out = group.run(0, &exec, Ok::<usize, joberror>);
        assert_eq!(out.unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn returns_results_in_index_order() {
        let group = workergroup::new(2).unwrap();
        let exec = exec(16);
        let out = group
            .run(4, &exec, |index| Ok::<usize, joberror>(index * 10))
            .unwrap();
        assert_eq!(out, vec![0, 10, 20, 30]);
        assert_eq!(group.getmaxworkers(), 2);
    }

    #[test]
    fn propagates_job_errors_with_single_worker() {
        let group = workergroup::new(1).unwrap();
        let exec = exec(16);
        let out = group.run(4, &exec, |index| {
            if index == 2 {
                Err(joberror::boom)
            } else {
                Ok::<usize, joberror>(index)
            }
        });
        assert_eq!(
            out,
            Err(workererror::job {
                index: 2,
                source: joberror::boom,
            })
        );
    }

    #[test]
    fn reports_step_budget_exhaustion() {
        let group = workergroup::new(1).unwrap();
        let exec = exec(2);
        let out = group.run(4, &exec, Ok::<usize, joberror>);
        assert_eq!(
            out,
            Err(workererror::context {
                source: contexterror::stepbudgetexceeded { used: 2, max: 2 },
            })
        );
    }

    #[test]
    fn reports_cancellation_before_any_job() {
        let group = workergroup::new(2).unwrap();
        let exec = exec(16);
        exec.getcancellation().cancel();
        let out = group.run(4, &exec, Ok::<usize, joberror>);
        assert_eq!(
            out,
            Err(workererror::context {
                source: contexterror::cancelled,
            })
        );
    }
}

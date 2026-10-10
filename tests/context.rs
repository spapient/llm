//! Cross-crate integration tests for the public `context` API.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use ::context::{
    collectingmetrics, contexterror, executionconfig, executioncontext, memorybudget, workererror,
    workergroup,
};
use lib::requestid;

#[derive(Debug, PartialEq, Eq)]
enum joberror {
    boom,
}

fn config(memorybytes: usize, maxworkers: usize, maxsteps: usize) -> executionconfig {
    executionconfig::new(memorybytes, maxworkers, maxsteps).unwrap()
}

#[test]
fn cancellation_and_step_budget_are_distinguishable() {
    let exec = executioncontext::new(requestid::new(1), config(1024, 2, 1));
    assert_eq!(exec.checkpoint(), Ok(()));
    assert_eq!(
        exec.checkpoint(),
        Err(contexterror::stepbudgetexceeded { used: 1, max: 1 })
    );
    exec.getcancellation().cancel();
    assert_eq!(exec.checkpoint(), Err(contexterror::cancelled));
}

#[test]
fn memory_budget_is_explicit_and_reclaimed() {
    let budget = memorybudget::new(64);
    let guard = budget.tryreserve(64).unwrap();
    assert_eq!(budget.getreserved(), 64);
    assert_eq!(
        budget.tryreserve(1).map(|_| ()),
        Err(contexterror::budgetexceeded {
            requested: 1,
            reserved: 64,
            capacity: 64,
        })
    );
    drop(guard);
    assert_eq!(budget.getreserved(), 0);
}

#[test]
fn worker_group_returns_ordered_results() {
    let group = workergroup::new(2).unwrap();
    let exec = executioncontext::new(requestid::new(2), config(1 << 16, 2, 64));
    let out = group
        .run(5, &exec, |index| Ok::<usize, joberror>(index + 1))
        .unwrap();
    assert_eq!(out, vec![1, 2, 3, 4, 5]);
}

#[test]
fn worker_group_reports_the_failing_job() {
    let group = workergroup::new(1).unwrap();
    let exec = executioncontext::new(requestid::new(3), config(1 << 16, 1, 64));
    let out = group.run(3, &exec, |index| {
        if index == 1 {
            Err(joberror::boom)
        } else {
            Ok::<usize, joberror>(index)
        }
    });
    assert_eq!(
        out,
        Err(workererror::job {
            index: 1,
            source: joberror::boom,
        })
    );
}

#[test]
fn metrics_sink_receives_execution_records() {
    let sink = Arc::new(collectingmetrics::new());
    let exec = executioncontext::withmetrics(requestid::new(4), config(1024, 1, 8), sink.clone());
    exec.recordcounter("checkpoints", 2);
    exec.recordgauge("available", 1024);
    assert_eq!(sink.getcounter("checkpoints"), Some(2));
    assert_eq!(sink.getgauge("available"), Some(1024));
}

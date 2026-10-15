//! Cache-aware submission forward: the single-token decode step recorded into a
//! submission must reproduce the compute-driven cache-aware forward bit for bit.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::common::{model, provider};
use api::submission;
use cpu::cpusubmission;
use infer::kvstate;
use models::causallm;

#[test]
fn forwardsubmission_cached_matches_forward_cached() {
    let exec = provider();
    let model = model(42);
    let config = model.getconfig();
    let prompt = vec![0usize, 1, 2];
    let next = 3usize;

    let mut cache =
        kvstate::new(config.getnlayers(), config.getkvwidth(), config.getmaxseq()).unwrap();
    for &token in &prompt {
        model.forward_cached(&exec, &mut cache, token).unwrap();
    }

    let want = model
        .forward_cached(&exec, &mut cache.clone(), next)
        .unwrap();

    let mut sub = cpusubmission::new(exec.clone());
    let handles = model
        .forwardsubmission_cached(&mut sub, &cache, next)
        .unwrap();
    let outputs = sub.apply(&[handles.logits]).unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].getdata(), &want);
}

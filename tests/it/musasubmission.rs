//! MUSA submission inference chain.
//!
//! The model-level prefill and decode chains record every operator into a
//! device-resident submission (embedding, normalization, matmul, rope, attention
//! and the elementwise chain), so this suite checks the recorded logits against
//! the compute path on the real device. The decode case additionally appends the
//! downloaded key/value rows back into the cache and keeps decoding, which is
//! the loop the inference engine performs.
#![cfg(all(target_os = "linux", musa_toolkit))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use api::submission;
use context::{executionconfig, executioncontext};
use infer::{
    collectingsink, generationengine, generationparams, generationrequest, kvstate, samplingparams,
    stopcondition, submissionengine,
};
use lib::{limits, requestid};
use models::{causallm, kvcache};
use musa::musaexecutor;
use testkit::assertions::assertallclose;

/// TF32 input rounding propagated through the model chain.
const TOL: f32 = 2.0e-3;

/// Reports whether a MUSA device is reachable, failing when it is required.
fn hasdevice(test: &str) -> bool {
    testkit::probe::requiredevice(test, testkit::probe::gpuprobe::musa)
}

/// A MUSA executor bounded for the tiny test model.
fn engine() -> musaexecutor {
    musaexecutor::new(limits::new(1 << 24, 1 << 28, 1 << 12, 1 << 8))
}

/// The full prefill chain recorded into one submission matches `forward`.
#[test]
fn forwardsubmission_matches_forward_on_musa() {
    if !hasdevice("forwardsubmission_matches_forward_on_musa") {
        return;
    }
    let exec = engine();
    let model = crate::common::model(42);
    let tokens = vec![0usize, 1, 2, 3];
    let want = model.forward(&exec, &tokens).unwrap();
    let mut sub = exec.newsubmission();
    let logits = model.forwardsubmission(&mut sub, &tokens).unwrap();
    let got = sub.apply(&[logits]).unwrap();
    assertallclose(got[0].getdata(), want.getdata(), TOL, TOL);
}

/// One cache-aware decode step recorded into a submission matches the compute
/// path.
#[test]
fn forwardsubmission_cached_matches_forward_cached_on_musa() {
    if !hasdevice("forwardsubmission_cached_matches_forward_cached_on_musa") {
        return;
    }
    let exec = engine();
    let model = crate::common::model(42);
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

    let mut sub = exec.newsubmission();
    let handles = model
        .forwardsubmission_cached(&mut sub, &cache, next)
        .unwrap();
    let got = sub.apply(&[handles.logits]).unwrap();
    assertallclose(got[0].getdata(), &want, TOL, TOL);
}

/// A prefill and two decode steps recorded through submissions match the
/// compute path step for step, with the new key/value rows appended back into
/// the cache exactly as the inference loop does.
#[test]
fn submission_decode_chain_matches_compute_on_musa() {
    if !hasdevice("submission_decode_chain_matches_compute_on_musa") {
        return;
    }
    let exec = engine();
    let model = crate::common::model(42);
    let config = model.getconfig();
    let layers = config.getnlayers();
    let width = config.getkvwidth();
    let prompt = vec![0usize, 1, 2];
    let next = [3usize, 4];

    // Compute reference: prefill then two cached decode steps.
    let mut refcache = kvstate::new(layers, width, config.getmaxseq()).unwrap();
    for &token in &prompt {
        model.forward_cached(&exec, &mut refcache, token).unwrap();
    }
    let want: Vec<Vec<f32>> = next
        .iter()
        .map(|&token| model.forward_cached(&exec, &mut refcache, token).unwrap())
        .collect();

    // Submission path: prefill with the compute cache-aware forward (the
    // recorded decode requires a populated cache), then record each decode
    // step, apply the logits plus every layer's new key/value row, and append
    // those rows to the cache.
    let mut cache = kvstate::new(layers, width, config.getmaxseq()).unwrap();
    for &token in &prompt {
        model.forward_cached(&exec, &mut cache, token).unwrap();
    }
    let record = |cache: &mut kvstate, token: usize| {
        let mut sub = exec.newsubmission();
        let handles = model
            .forwardsubmission_cached(&mut sub, cache, token)
            .unwrap();
        let mut requested = Vec::with_capacity(1 + 2 * layers);
        requested.push(handles.logits);
        requested.extend(handles.keys.iter().copied());
        requested.extend(handles.values.iter().copied());
        let mut out = sub.apply(&requested).unwrap();
        let mut keyrows = out.split_off(1);
        let valuerows = keyrows.split_off(layers);
        for layer in 0..layers {
            cache
                .append(layer, &keyrows[layer], &valuerows[layer])
                .unwrap();
        }
        out.remove(0).getdata().to_vec()
    };

    for (step, &token) in next.iter().enumerate() {
        let got = record(&mut cache, token);
        assertallclose(&got, &want[step], TOL, TOL);
    }
}

/// The submission generation engine (compute prefill, device-resident decode)
/// selects the same greedy tokens as the compute engine.
#[test]
fn submission_engine_matches_compute_engine_on_musa() {
    if !hasdevice("submission_engine_matches_compute_engine_on_musa") {
        return;
    }
    let exec = engine();
    let model = crate::common::model(42);
    let stop = stopcondition::new(vec![], 3).unwrap();
    let params = generationparams::formodel(model.getconfig(), stop, samplingparams::greedy());
    let prompt = vec![1usize, 2, 3];
    let mut compute_request = generationrequest::new(prompt.clone(), params.clone()).unwrap();
    let mut submission_request = generationrequest::new(prompt, params).unwrap();
    let compute_context = executioncontext::new(
        requestid::new(1),
        executionconfig::new(1 << 20, 1, 8).unwrap(),
    );
    let submission_context = executioncontext::new(
        requestid::new(2),
        executionconfig::new(1 << 20, 1, 8).unwrap(),
    );
    let compute_engine = generationengine::new(&model, &exec);
    let submission_engine = submissionengine::new(&model, &exec, || exec.newsubmission());
    let mut compute_sink = collectingsink::new();
    let mut submission_sink = collectingsink::new();

    let want = compute_engine
        .generate(&compute_context, &mut compute_request, &mut compute_sink)
        .unwrap();
    let got = submission_engine
        .generate(
            &submission_context,
            &mut submission_request,
            &mut submission_sink,
        )
        .unwrap();

    assert_eq!(got.gettokens(), want.gettokens());
    assert_eq!(got.getreason(), want.getreason());
}

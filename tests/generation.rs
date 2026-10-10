//! Deterministic generation, stop conditions, streaming, and cancellation for
//! the `infer` engine.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::{argmaxlast, model, provider};
use context::{cancellation, executionconfig, executioncontext};
use infer::{
    collectingsink, generationengine, generationparams, generationrequest, infererror,
    samplingparams, stopcondition, stopreason, streamsink,
};
use lib::requestid;
use models::{causallm, llamamodel};

fn makeexec(id: u64, maxsteps: usize) -> executioncontext {
    executioncontext::new(
        requestid::new(id),
        executionconfig::new(1 << 20, 1, maxsteps).unwrap(),
    )
}

fn request(
    model: &llamamodel,
    prompt: Vec<usize>,
    eos: Vec<usize>,
    maxnew: usize,
) -> generationrequest {
    let stop = stopcondition::new(eos, maxnew).unwrap();
    let params = generationparams::formodel(model.getconfig(), stop, samplingparams::greedy());
    generationrequest::new(prompt, params).unwrap()
}

/// A sink that refuses the first token, requesting an immediate stop.
struct stopsink;

impl streamsink for stopsink {
    fn ongenerated(&mut self, _index: usize, _token: usize) -> bool {
        false
    }

    fn onstop(&mut self, _reason: stopreason) {}
}

/// A sink that cancels a shared handle on the first token.
struct cancelsink {
    handle: cancellation,
}

impl streamsink for cancelsink {
    fn ongenerated(&mut self, _index: usize, _token: usize) -> bool {
        self.handle.cancel();
        true
    }

    fn onstop(&mut self, _reason: stopreason) {}
}

#[test]
fn greedy_generation_matches_an_independent_recomputation() {
    let model = model(1);
    let provider = provider();
    let engine = generationengine::new(&model, &provider);
    let exec = makeexec(1, 8);
    let prompt = vec![1usize, 2, 3];
    let mut request = request(&model, prompt.clone(), vec![], 3);
    let mut sink = collectingsink::new();

    let result = engine.generate(&exec, &mut request, &mut sink).unwrap();

    assert_eq!(result.getreason(), stopreason::maxlength);
    assert_eq!(result.getcount(), 3);
    assert_eq!(result.getsteps(), 3);

    let mut context = prompt;
    let mut expected = Vec::new();
    for _ in 0..3 {
        let next = argmaxlast(&model, &provider, &context);
        expected.push(next);
        context.push(next);
    }
    assert_eq!(result.gettokens(), expected.as_slice());
    assert_eq!(sink.gettokens(), result.gettokens());
    assert_eq!(sink.getreason(), Some(stopreason::maxlength));
}

#[test]
fn generation_is_deterministic_across_requests() {
    let model = model(2);
    let provider = provider();
    let engine = generationengine::new(&model, &provider);

    let run = |id: u64| {
        let exec = makeexec(id, 8);
        let mut request = request(&model, vec![1, 2], vec![], 3);
        let mut sink = collectingsink::new();
        engine
            .generate(&exec, &mut request, &mut sink)
            .unwrap()
            .gettokens()
            .to_vec()
    };

    assert_eq!(run(1), run(2));
}

#[test]
fn an_eos_token_ends_generation_with_no_tokens() {
    let model = model(3);
    let provider = provider();
    let engine = generationengine::new(&model, &provider);
    let exec = makeexec(1, 8);
    let prompt = vec![1usize, 2, 3];
    let first = argmaxlast(&model, &provider, &prompt);
    let mut request = request(&model, prompt, vec![first], 4);
    let mut sink = collectingsink::new();

    let result = engine.generate(&exec, &mut request, &mut sink).unwrap();

    assert_eq!(result.getreason(), stopreason::endoftoken);
    assert_eq!(result.getcount(), 0);
    assert_eq!(result.getsteps(), 1);
    assert!(sink.gettokens().is_empty());
    assert_eq!(sink.getreason(), Some(stopreason::endoftoken));
}

#[test]
fn the_new_token_cap_ends_generation() {
    let model = model(4);
    let provider = provider();
    let engine = generationengine::new(&model, &provider);
    let exec = makeexec(1, 8);
    let mut request = request(&model, vec![1, 2], vec![], 2);
    let mut sink = collectingsink::new();

    let result = engine.generate(&exec, &mut request, &mut sink).unwrap();

    assert_eq!(result.getreason(), stopreason::maxlength);
    assert_eq!(result.getcount(), 2);
}

#[test]
fn a_sink_can_request_a_stop() {
    let model = model(5);
    let provider = provider();
    let engine = generationengine::new(&model, &provider);
    let exec = makeexec(1, 8);
    let mut request = request(&model, vec![1, 2], vec![], 4);
    let mut sink = stopsink;

    let result = engine.generate(&exec, &mut request, &mut sink).unwrap();

    assert_eq!(result.getreason(), stopreason::stopped);
    assert_eq!(result.getcount(), 1);
}

#[test]
fn a_pre_cancelled_execution_fails_before_any_forward() {
    let model = model(6);
    let provider = provider();
    let engine = generationengine::new(&model, &provider);
    let exec = makeexec(1, 8);
    exec.getcancellation().cancel();
    let mut request = request(&model, vec![1, 2], vec![], 3);
    let mut sink = collectingsink::new();

    let error = engine.generate(&exec, &mut request, &mut sink).unwrap_err();

    assert_eq!(error, infererror::context(context::contexterror::cancelled));
    assert!(sink.getreason().is_none());
    assert_eq!(request.getgeneratedlen(), 0);
}

#[test]
fn mid_generation_cancellation_stops_and_leaves_the_request_reusable() {
    let model = model(7);
    let provider = provider();
    let engine = generationengine::new(&model, &provider);

    let exec = makeexec(1, 8);
    let mut request = request(&model, vec![1, 2, 3], vec![], 3);
    let mut sink = cancelsink {
        handle: exec.getcancellation().clone(),
    };
    let error = engine.generate(&exec, &mut request, &mut sink).unwrap_err();
    assert_eq!(error, infererror::context(context::contexterror::cancelled));
    // The first token was emitted before cancellation was observed, so a
    // failure never appears as a completed result.
    assert_eq!(request.getgeneratedlen(), 1);

    // After a reset the same request runs to completion deterministically.
    request.reset();
    assert_eq!(request.getgeneratedlen(), 0);
    let fresh = makeexec(2, 8);
    let mut sink = collectingsink::new();
    let result = engine.generate(&fresh, &mut request, &mut sink).unwrap();
    assert_eq!(result.getreason(), stopreason::maxlength);
    assert_eq!(result.getcount(), 3);
}

#[test]
fn an_out_of_vocabulary_prompt_token_is_rejected() {
    let model = model(8);
    let provider = provider();
    let engine = generationengine::new(&model, &provider);
    let exec = makeexec(1, 8);
    let mut request = request(&model, vec![1, common::VOCAB], vec![], 3);
    let mut sink = collectingsink::new();

    let error = engine.generate(&exec, &mut request, &mut sink).unwrap_err();

    assert_eq!(
        error,
        infererror::tokenoutofrange {
            token: common::VOCAB,
            vocab: common::VOCAB,
        }
    );
}

#[test]
fn an_empty_prompt_is_rejected_at_construction() {
    let model = model(9);
    let stop = stopcondition::new(vec![], 3).unwrap();
    let params = generationparams::formodel(model.getconfig(), stop, samplingparams::greedy());
    assert!(matches!(
        generationrequest::new(vec![], params),
        Err(infererror::invalidrequest {
            field: "prompt",
            ..
        })
    ));
}

//! Request isolation: independent requests, request-owned caches, and
//! concurrent generation over one shared model.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::common;
use crate::common::{argmaxlast, model, provider};
use ::tensor::{shape, tensor};
use context::{executionconfig, executioncontext};
use cpu::cpuexecutor;
use infer::{
    collectingsink, generationengine, generationparams, generationrequest, samplingparams,
    stopcondition,
};
use lib::requestid;
use models::{causallm, kvcache, llamamodel};

fn run(model: &llamamodel, provider: &cpuexecutor, prompt: &[usize]) -> Vec<usize> {
    let engine = generationengine::new(model, provider);
    let exec = executioncontext::new(
        requestid::new(1),
        executionconfig::new(1 << 20, 1, 8).unwrap(),
    );
    let stop = stopcondition::new(vec![], 3).unwrap();
    let params = generationparams::formodel(model.getconfig(), stop, samplingparams::greedy());
    let mut request = generationrequest::new(prompt.to_vec(), params).unwrap();
    let mut sink = collectingsink::new();
    engine
        .generate(&exec, &mut request, &mut sink)
        .unwrap()
        .gettokens()
        .to_vec()
}

fn request(model: &llamamodel, prompt: Vec<usize>) -> generationrequest {
    let stop = stopcondition::new(vec![], 3).unwrap();
    let params = generationparams::formodel(model.getconfig(), stop, samplingparams::greedy());
    generationrequest::new(prompt, params).unwrap()
}

/// Independently recomputes the three-token greedy continuation of `prompt`.
fn expected(model: &llamamodel, provider: &cpuexecutor, prompt: &[usize]) -> Vec<usize> {
    let mut context = prompt.to_vec();
    let mut tokens = Vec::new();
    for _ in 0..3 {
        let next = argmaxlast(model, provider, &context);
        tokens.push(next);
        context.push(next);
    }
    tokens
}

#[test]
fn independent_requests_do_not_share_generated_state() {
    let model = model(11);
    let provider = provider();

    let first = run(&model, &provider, &[1, 2, 3]);
    let other = run(&model, &provider, &[4, 1, 2]);
    let again = run(&model, &provider, &[1, 2, 3]);

    // Re-running an earlier prompt reproduces its tokens exactly.
    assert_eq!(first, again);
    // Every result matches an independent recomputation for its own prompt,
    // so no token from one request leaks into another.
    assert_eq!(first, expected(&model, &provider, &[1, 2, 3]));
    assert_eq!(other, expected(&model, &provider, &[4, 1, 2]));
}

#[test]
fn request_owned_caches_are_isolated() {
    let model = model(12);
    let a = request(&model, vec![1, 2]);
    let mut b = request(&model, vec![3, 4]);

    assert_eq!(a.getkvcache().getlayers(), model.getconfig().getnlayers());
    assert_eq!(a.getkvcache().getmaxseq(), common::MAXSEQ);
    assert_eq!(a.getkvcache().getwidth(), model.getconfig().getkvwidth());
    assert_eq!(a.getkvcache().getlen(), 0);
    assert_eq!(b.getkvcache().getlen(), 0);

    let row = tensor::new(
        shape::new(vec![1, model.getconfig().getkvwidth()]).unwrap(),
        vec![0.0_f32; model.getconfig().getkvwidth()],
    )
    .unwrap();
    b.getkvcachemut().appendall(&row, &row).unwrap();

    assert_eq!(b.getkvcache().getlen(), 1);
    assert_eq!(a.getkvcache().getlen(), 0);
}

#[test]
fn concurrent_requests_over_one_model_match_sequential_results() {
    let model = model(13);
    let provider = provider();
    let prompts = [vec![1usize, 2, 3], vec![4, 1, 2]];

    let sequential: Vec<Vec<usize>> = prompts
        .iter()
        .map(|prompt| run(&model, &provider, prompt))
        .collect();

    let modelref = &model;
    let providerref = &provider;
    let concurrent: Vec<Vec<usize>> = std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .iter()
            .cloned()
            .map(|prompt| scope.spawn(move || run(modelref, providerref, &prompt)))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    assert_eq!(concurrent, sequential);
}

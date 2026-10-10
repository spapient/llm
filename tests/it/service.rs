//! Service-level tests: routing, validation, admission, cancellation mapping,
//! generation mapping, and request isolation.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::common;

use std::sync::Barrier;
use std::sync::mpsc;

use context::{executionconfig, executioncontext};
use cpu::cpuexecutor;
use infer::{
    collectingsink, generationengine, generationparams, generationrequest, samplingparams,
    stopcondition,
};
use lib::requestid;
use models::{causallm, llamamodel};
use serve::inferenceservice;

/// Independently runs the generation engine for the same parameters.
fn expected(
    model: &llamamodel,
    provider: &cpuexecutor,
    prompt: &[usize],
    maxnew: usize,
) -> Vec<usize> {
    let exec = executioncontext::new(
        requestid::new(1),
        executionconfig::new(1 << 20, 1, 128).unwrap(),
    );
    let stop = stopcondition::new(vec![], maxnew).unwrap();
    let params = generationparams::formodel(model.getconfig(), stop, samplingparams::greedy());
    let mut request = generationrequest::new(prompt.to_vec(), params).unwrap();
    let mut sink = collectingsink::new();
    generationengine::new(model, provider)
        .generate(&exec, &mut request, &mut sink)
        .unwrap()
        .gettokens()
        .to_vec()
}

#[test]
fn health_route_returns_ok() {
    let model = common::model(1);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));

    let response = service.handle(&common::parse(b"GET /health HTTP/1.1\r\nhost: x\r\n\r\n"));
    assert_eq!(response.getstatus(), 200);
    assert_eq!(response.getbodytext(), Some("ok"));
}

#[test]
fn unknown_route_is_not_found() {
    let model = common::model(2);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));

    let response = service.handle(&common::parse(b"GET /nope HTTP/1.1\r\n\r\n"));
    assert_eq!(response.getstatus(), 404);
}

#[test]
fn wrong_method_is_rejected() {
    let model = common::model(3);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));

    let response = service.handle(&common::parse(b"GET /generate HTTP/1.1\r\n\r\n"));
    assert_eq!(response.getstatus(), 405);
}

#[test]
fn missing_field_is_rejected_before_model_work() {
    let model = common::model(4);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));

    let empty = common::parse(b"POST /generate HTTP/1.1\r\n\r\n");
    let response = service.handle(&empty);
    assert_eq!(response.getstatus(), 400);
    assert!(response.getbodytext().unwrap_or("").contains("prompt"));
}

#[test]
fn out_of_range_prompt_is_rejected() {
    let model = common::model(5);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));

    // A prompt longer than the model's maximum sequence is rejected when the
    // generation request is constructed.
    let prompt: Vec<usize> = (0..common::MAXSEQ).collect();
    let response = service.handle(&common::generaterequest(&prompt, 32));
    assert_eq!(response.getstatus(), 400);
}

#[test]
fn unsupported_media_type_is_rejected() {
    let model = common::model(6);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));

    let body = "prompt=1&maxnewtokens=2";
    let raw = format!(
        "POST /generate HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let response = service.handle(&common::parse(raw.as_bytes()));
    assert_eq!(response.getstatus(), 415);
}

#[test]
fn generation_matches_the_engine() {
    let model = common::model(7);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));

    let response = service.handle(&common::generaterequest(&[1, 2, 3], 3));
    assert_eq!(response.getstatus(), 200);
    assert_eq!(
        common::tokensof(&response),
        expected(&model, &provider, &[1, 2, 3], 3)
    );
    assert_eq!(common::reasonof(&response), "maxlength");
}

#[test]
fn transport_cancellation_maps_to_a_timeout_response() {
    let model = common::model(8);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));

    let response = service.handlewith(&common::generaterequest(&[1, 2], 2), |cancel| {
        cancel.cancel();
    });
    assert_eq!(response.getstatus(), 408);
    assert!(response.getbodytext().unwrap_or("").contains("cancelled"));
}

#[test]
fn admission_rejects_requests_over_the_ceiling() {
    let model = common::model(9);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(1));
    let started = Barrier::new(2);
    let (release_tx, release_rx) = mpsc::channel::<()>();

    std::thread::scope(|scope| {
        let service = &service;
        let started = &started;
        scope.spawn(move || {
            let occupied = common::generaterequest(&[1, 2], 2);
            let response = service.handlewith(&occupied, move |_| {
                started.wait();
                let _ = release_rx.recv();
            });
            assert_eq!(response.getstatus(), 200);
        });
        started.wait();
        let blocked = service.handle(&common::generaterequest(&[3, 4], 2));
        assert_eq!(blocked.getstatus(), 429);
        let _ = release_tx.send(());
    });
}

#[test]
fn concurrent_requests_are_isolated() {
    let model = common::model(10);
    let provider = common::provider();
    let service = inferenceservice::new(&model, &provider, common::serviceconfig(4));
    let prompts = [vec![1usize, 2, 3], vec![4, 1, 2]];

    let sequential: Vec<Vec<usize>> = prompts
        .iter()
        .map(|prompt| common::tokensof(&service.handle(&common::generaterequest(prompt, 3))))
        .collect();

    let concurrent: Vec<Vec<usize>> = std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .iter()
            .map(|prompt| {
                let service = &service;
                let prompt = prompt.clone();
                scope.spawn(move || {
                    common::tokensof(&service.handle(&common::generaterequest(&prompt, 3)))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    assert_eq!(concurrent, sequential);
}

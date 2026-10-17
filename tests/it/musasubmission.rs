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

use crate::support::build;
use ::tensor::shape;
use api::{backenderror, compute, operand, recorder, session, stepref, submission};
use context::{executionconfig, executioncontext};
use infer::{
    collectingsink, generationengine, generationparams, generationrequest, kvstate, samplingparams,
    stopcondition, submissionengine,
};
use lib::{limits, requestid};
use models::{causallm, devicekvcache, kvcache};
use musa::{musaexecutor, musasession};
use testkit::assertions::assertallclose;
use testkit::reference;

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

    // Submission path: prefill and decode both record one submission per
    // token, apply the logits plus every layer's new key/value row, and append
    // those rows to the cache. The first prompt token records against an empty
    // cache.
    let mut cache = kvstate::new(layers, width, config.getmaxseq()).unwrap();
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

    for &token in &prompt {
        record(&mut cache, token);
    }
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
    let submission_engine = submissionengine::new(&model, || exec.newpersistent());
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

/// A recorded fused RMSNorm+matmul matches the compute path.
#[test]
fn submission_rmsnormmatmul_matches_compute_on_musa() {
    if !hasdevice("submission_rmsnormmatmul_matches_compute_on_musa") {
        return;
    }
    let exec = engine();
    let (m, k, n) = (31usize, 64usize, 48usize);
    let input: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
    let weights: Vec<f32> = (0..k).map(|i| (i as f32 * 0.17).cos() + 1.5).collect();
    let rhs: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.19).sin()).collect();
    let x = build(vec![m, k], input);
    let wv = build(vec![k], weights);
    let b = build(vec![k, n], rhs);
    let want = exec
        .rmsnormmatmul(&x.getview(), &wv.getview(), &b.getview(), 1.0e-5)
        .unwrap();
    let mut sess = exec.newpersistent();
    let step = sess
        .recordrmsnormmatmul(
            operand::input(x.getview()),
            operand::input(wv.getview()),
            operand::input(b.getview()),
            1.0e-5,
        )
        .unwrap();
    let out = sess.apply(&[step]).unwrap();
    assertallclose(out[0].getdata(), want.getdata(), 2.0e-3, 2.0e-3);
}

/// A device-resident KV cache used by the decode test: each layer owns one
/// zeroed `[maxseq, width]` key buffer and one value buffer, and the current
/// prefix is exposed as a `[len, width]` view.
struct musadevicecache {
    layers: usize,
    width: usize,
    maxseq: usize,
    len: usize,
    keys: Vec<stepref>,
    values: Vec<stepref>,
}

impl musadevicecache {
    fn new(session: &mut musasession, layers: usize, width: usize, maxseq: usize) -> Self {
        let buffer = shape::new(vec![maxseq, width]).unwrap();
        let mut keys = Vec::with_capacity(layers);
        let mut values = Vec::with_capacity(layers);
        for _ in 0..layers {
            keys.push(session.zeroed(buffer.clone()).unwrap());
            values.push(session.zeroed(buffer.clone()).unwrap());
        }
        Self {
            layers,
            width,
            maxseq,
            len: 0,
            keys,
            values,
        }
    }
}

impl devicekvcache for musadevicecache {
    fn getlayers(&self) -> usize {
        self.layers
    }

    fn getlen(&self) -> usize {
        self.len
    }

    fn getmaxseq(&self) -> usize {
        self.maxseq
    }

    fn keys<R: recorder>(
        &mut self,
        sub: &mut R,
        layer: usize,
    ) -> Result<Option<stepref>, backenderror> {
        if self.len == 0 {
            return Ok(None);
        }
        let view =
            shape::new(vec![self.len, self.width]).map_err(|_| backenderror::sizeoverflow)?;
        Ok(Some(sub.recordview(self.keys[layer], view)?))
    }

    fn values<R: recorder>(
        &mut self,
        sub: &mut R,
        layer: usize,
    ) -> Result<Option<stepref>, backenderror> {
        if self.len == 0 {
            return Ok(None);
        }
        let view =
            shape::new(vec![self.len, self.width]).map_err(|_| backenderror::sizeoverflow)?;
        Ok(Some(sub.recordview(self.values[layer], view)?))
    }

    fn append<R: recorder>(
        &mut self,
        sub: &mut R,
        layer: usize,
        key: stepref,
        value: stepref,
    ) -> Result<(), backenderror> {
        sub.recordappend(
            operand::result(self.keys[layer]),
            operand::result(key),
            self.len,
        )?;
        sub.recordappend(
            operand::result(self.values[layer]),
            operand::result(value),
            self.len,
        )
    }

    fn advance(&mut self) {
        self.len += 1;
    }
}

/// The device KV cache keeps every cached row on the device and reproduces the
/// compute decode logits.
#[test]
fn device_cache_decode_chain_matches_compute_on_musa() {
    if !hasdevice("device_cache_decode_chain_matches_compute_on_musa") {
        return;
    }
    let exec = engine();
    let model = crate::common::model(42);
    let config = model.getconfig();
    let layers = config.getnlayers();
    let width = config.getkvwidth();
    let prompt = vec![0usize, 1, 2];
    let next = [3usize, 4];

    let mut refcache = kvstate::new(layers, width, config.getmaxseq()).unwrap();
    for &token in &prompt {
        model.forward_cached(&exec, &mut refcache, token).unwrap();
    }
    let want: Vec<Vec<f32>> = next
        .iter()
        .map(|&token| model.forward_cached(&exec, &mut refcache, token).unwrap())
        .collect();

    let mut sess = exec.newpersistent();
    let mut cache = musadevicecache::new(&mut sess, layers, width, config.getmaxseq());
    let mut got = Vec::new();
    for (step, &token) in prompt.iter().chain(next.iter()).enumerate() {
        let handles = model
            .forwardsubmission_cached_device(&mut sess, &mut cache, token)
            .unwrap();
        let out = sess.apply(&[handles.logits]).unwrap();
        if step >= prompt.len() {
            got.push(out[0].getdata().to_vec());
        }
    }
    assert_eq!(got.len(), want.len());
    for (step, (actual, expected)) in got.iter().zip(want.iter()).enumerate() {
        assertallclose(actual, expected, TOL, TOL);
        let _ = step;
    }
}

/// A persistent session exposes both scan outputs, and the state handle feeds
/// the next cached step without a host round trip.
#[test]
fn submission_scan_pair_round_trips_state_on_musa() {
    if !hasdevice("submission_scan_pair_round_trips_state_on_musa") {
        return;
    }
    let exec = engine();
    let n = 8usize;
    let d = 3usize;
    let statewidth = 4usize;
    let input: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.37).sin()).collect();
    let init: Vec<f32> = (0..d * statewidth)
        .map(|i| (i as f32 * 0.13).cos())
        .collect();
    let a: Vec<f32> = (0..d * statewidth)
        .map(|i| 0.5 + 0.5 * (i as f32 * 0.71).cos().abs())
        .collect();
    let b: Vec<f32> = (0..d * statewidth)
        .map(|i| (i as f32 * 0.29).sin())
        .collect();
    let c: Vec<f32> = (0..d * statewidth)
        .map(|i| (i as f32 * 0.19).cos())
        .collect();
    let skip: Vec<f32> = (0..d).map(|i| (i as f32 * 0.11).sin()).collect();
    let input_t = build(vec![n, d], input.clone());
    let init_t = build(vec![d, statewidth], init.clone());
    let a_t = build(vec![d, statewidth], a.clone());
    let b_t = build(vec![d, statewidth], b.clone());
    let c_t = build(vec![d, statewidth], c.clone());
    let skip_t = build(vec![d], skip.clone());
    let next: Vec<f32> = (0..d).map(|i| (i as f32 * 0.53).cos()).collect();
    let next_t = build(vec![1, d], next.clone());

    let mut sess = exec.newpersistent();
    let (outref, stateref) = sess
        .recordscan_pair(
            operand::input(input_t.getview()),
            operand::input(init_t.getview()),
            operand::input(a_t.getview()),
            operand::input(b_t.getview()),
            operand::input(c_t.getview()),
            operand::input(skip_t.getview()),
        )
        .unwrap();
    let (out2ref, state2ref) = sess
        .recordscan_cached_pair(
            operand::input(next_t.getview()),
            operand::result(stateref),
            operand::input(a_t.getview()),
            operand::input(b_t.getview()),
            operand::input(c_t.getview()),
            operand::input(skip_t.getview()),
        )
        .unwrap();
    let outputs = sess.apply(&[outref, stateref, out2ref, state2ref]).unwrap();

    let (want_out, want_state) =
        reference::scan(&input, &init, &a, &b, &c, &skip, n, d, statewidth);
    assertallclose(outputs[0].getdata(), &want_out, 1e-5, 1e-5);
    assertallclose(outputs[1].getdata(), &want_state, 1e-5, 1e-5);

    let mut wantstate = want_state.clone();
    let mut wantout = vec![0.0_f32; d];
    for i in 0..d {
        let x = next[i];
        let mut acc = 0.0_f32;
        for j in 0..statewidth {
            let idx = i * statewidth + j;
            wantstate[idx] = a[idx] * wantstate[idx] + b[idx] * x;
            acc += c[idx] * wantstate[idx];
        }
        wantout[i] = acc + skip[i] * x;
    }
    assertallclose(outputs[2].getdata(), &wantout, 1e-5, 1e-5);
    assertallclose(outputs[3].getdata(), &wantstate, 1e-5, 1e-5);
}

//! Device-resident KV cache: the `devicecache` contract implementation and its
//! equivalence against the host cache-aware forward.
//!
//! The device forward records every token of the prompt and the decode step
//! into one submission chain — `recordpersist` registers each layer's buffer
//! once (the CPU reference deduplicates on `(pointer, length)`), `recordview`
//! exposes the growing `[position, width]` window, and `recordappend` writes
//! each new row in place — then applies once and requires the final logits to
//! match the host `forward_cached` path bit for bit.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::common::{model, provider};
use api::submission;
use cpu::cpusubmission;
use models::causallm;
use models::devicecache;
use models::devicekvcache;

/// `devicecache::new` rejects zero dimensions.
#[test]
fn devicecache_rejects_zero_dimensions() {
    assert!(devicecache::new(0, 8, 64).is_err());
    assert!(devicecache::new(2, 0, 64).is_err());
    assert!(devicecache::new(2, 8, 0).is_err());
}

/// At position zero the cache exposes no cached rows, so a device forward of
/// the first token attends over the empty key/value operands.
#[test]
fn devicecache_position0_has_no_handles() {
    let mut cache = devicecache::new(2, 8, 64).unwrap();
    let mut sub = cpusubmission::new(provider());
    assert!(cache.keys(&mut sub, 0).unwrap().is_none());
    assert!(cache.values(&mut sub, 0).unwrap().is_none());
    assert_eq!(cache.getlen(), 0);
}

/// Appended rows become visible through the leading view: position advances,
/// the key view exposes exactly `[position, width]`, and the window contents
/// equal the appended rows.
#[test]
fn devicecache_append_then_view_roundtrip() {
    let mut cache = devicecache::new(1, 4, 8).unwrap();
    let mut sub = cpusubmission::new(provider());
    let k0 = vec![1.0, 2.0, 3.0, 4.0];
    let v0 = vec![5.0, 6.0, 7.0, 8.0];
    let (kh, vh) = (
        sub.recordpersist(&k0, ::tensor::shape::new(vec![1, 4]).unwrap())
            .unwrap(),
        sub.recordpersist(&v0, ::tensor::shape::new(vec![1, 4]).unwrap())
            .unwrap(),
    );
    cache.append(&mut sub, 0, kh, vh).unwrap();
    cache.advance();
    let k1 = vec![9.0, 10.0, 11.0, 12.0];
    let (kh, vh) = (
        sub.recordpersist(&k1, ::tensor::shape::new(vec![1, 4]).unwrap())
            .unwrap(),
        sub.recordpersist(&v1(), ::tensor::shape::new(vec![1, 4]).unwrap())
            .unwrap(),
    );
    cache.append(&mut sub, 0, kh, vh).unwrap();
    cache.advance();

    let handle = cache.keys(&mut sub, 0).unwrap().unwrap();
    let view = sub
        .recordview(handle, ::tensor::shape::new(vec![2, 4]).unwrap())
        .unwrap();
    let outputs = sub.apply(&[view]).unwrap();
    let got = outputs[0].getdata();
    let want = vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0];
    assert_eq!(got, &want);
}

fn v1() -> Vec<f32> {
    vec![13.0, 14.0, 15.0, 16.0]
}

/// Appending past `maxseq` fails: the row lands outside the registered buffer.
#[test]
fn devicecache_append_out_of_range() {
    let mut cache = devicecache::new(1, 4, 2).unwrap();
    let mut sub = cpusubmission::new(provider());
    for row in 0..2usize {
        let data = vec![row as f32; 4];
        let handle = sub
            .recordpersist(&data, ::tensor::shape::new(vec![1, 4]).unwrap())
            .unwrap();
        cache.append(&mut sub, 0, handle, handle).unwrap();
        cache.advance();
    }
    let overflow = vec![9.0; 4];
    let handle = sub
        .recordpersist(&overflow, ::tensor::shape::new(vec![1, 4]).unwrap())
        .unwrap();
    assert!(cache.append(&mut sub, 0, handle, handle).is_err());
}

/// `clear` resets the position so a reused cache starts a fresh sequence.
#[test]
fn devicecache_clear_resets_position() {
    let mut cache = devicecache::new(1, 4, 8).unwrap();
    let mut sub = cpusubmission::new(provider());
    let row = vec![1.0; 4];
    let handle = sub
        .recordpersist(&row, ::tensor::shape::new(vec![1, 4]).unwrap())
        .unwrap();
    cache.append(&mut sub, 0, handle, handle).unwrap();
    cache.advance();
    assert_eq!(cache.getlen(), 1);
    cache.clear();
    assert_eq!(cache.getlen(), 0);
    assert!(cache.keys(&mut sub, 0).unwrap().is_none());
}

/// The device-resident forward reproduces the host cache-aware forward bit for
/// bit: same prompt, same decode token, same logits.
#[test]
fn device_forward_matches_host_cached() {
    let exec = provider();
    let model = model(42);
    let config = model.getconfig();
    let prompt = vec![0usize, 1, 2];
    let next = 3usize;

    let mut host =
        infer::kvstate::new(config.getnlayers(), config.getkvwidth(), config.getmaxseq()).unwrap();
    for &token in &prompt {
        model.forward_cached(&exec, &mut host, token).unwrap();
    }
    let want = model.forward_cached(&exec, &mut host, next).unwrap();

    let mut cache =
        devicecache::new(config.getnlayers(), config.getkvwidth(), config.getmaxseq()).unwrap();
    let mut sub = cpusubmission::new(exec);
    let mut last = None;
    for &token in prompt.iter().chain(core::iter::once(&next)) {
        let step = model
            .forwardsubmission_cached_device(&mut sub, &mut cache, token)
            .unwrap();
        last = Some(step.logits);
    }
    let outputs = sub.apply(&[last.unwrap()]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].getdata(), &want);
}

/// The device-resident forward also matches a full recompute of the whole
/// prompt plus the decode token.
#[test]
fn device_forward_matches_full_recompute() {
    let exec = provider();
    let model = model(42);
    let config = model.getconfig();
    let prompt = vec![0usize, 1, 2];
    let next = 3usize;
    let mut full = prompt.clone();
    full.push(next);
    let full_out = model.forward(&exec, &full).unwrap();
    let vocab = config.getvocab();
    let want = &full_out.getdata()[full_out.getdata().len() - vocab..];

    let mut cache =
        devicecache::new(config.getnlayers(), config.getkvwidth(), config.getmaxseq()).unwrap();
    let mut sub = cpusubmission::new(exec);
    let mut last = None;
    for &token in &full {
        let step = model
            .forwardsubmission_cached_device(&mut sub, &mut cache, token)
            .unwrap();
        last = Some(step.logits);
    }
    let outputs = sub.apply(&[last.unwrap()]).unwrap();
    assert_eq!(outputs[0].getdata(), want);
}

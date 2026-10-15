//! Cache-aware forward: prefill the prompt token by token into a KV cache, then
//! decode one more token and require the result to match a full recompute bit
//! for bit.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::common::{model, provider};
use infer::kvstate;
use models::causallm;

#[test]
fn cached_decode_matches_full_recompute() {
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
        kvstate::new(config.getnlayers(), config.getkvwidth(), config.getmaxseq()).unwrap();
    for &token in &prompt {
        model.forward_cached(&exec, &mut cache, token).unwrap();
    }
    let got = model.forward_cached(&exec, &mut cache, next).unwrap();

    assert_eq!(got, want);
}

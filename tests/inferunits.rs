//! Relocated unit tests for the `infer` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/infer/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/infer/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::infer::infererror;
    use context::contexterror;
    use models::modelerror;

    #[test]
    fn carries_structured_request_context() {
        assert_eq!(
            infererror::invalidrequest {
                field: "prompt",
                detail: "must be non-empty".to_string(),
            }
            .to_string(),
            "invalid generation request for prompt: must be non-empty"
        );
    }

    #[test]
    fn token_and_logits_variants_are_distinct() {
        assert_ne!(infererror::emptylogits, infererror::degeneratelogits);
        assert_eq!(
            infererror::tokenoutofrange { token: 9, vocab: 5 }.to_string(),
            "token 9 is outside the vocabulary of size 5"
        );
    }

    #[test]
    fn wraps_model_and_context_errors() {
        let m: infererror = modelerror::config("dim", "bad").into();
        assert!(matches!(m, infererror::model(_)));
        let c: infererror = contexterror::cancelled.into();
        assert_eq!(c, infererror::context(contexterror::cancelled));
    }
}

/// Tests relocated from `crates/infer/src/kvcache.rs`.
mod kvcache {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::infer::kvstate;
    use ::tensor::{shape, tensor};
    use models::kvcache;

    fn row(values: [f32; 2]) -> tensor<f32> {
        tensor::new(shape::new(vec![1, 2]).unwrap(), values.to_vec()).unwrap()
    }

    #[test]
    fn rejects_zero_dimensions() {
        assert!(kvstate::new(0, 2, 8).is_err());
        assert!(kvstate::new(2, 0, 8).is_err());
        assert!(kvstate::new(2, 2, 0).is_err());
    }

    #[test]
    fn appends_and_reports_state() {
        let mut c = kvstate::new(2, 2, 8).unwrap();
        assert_eq!(c.getlayers(), 2);
        assert_eq!(c.getwidth(), 2);
        assert_eq!(c.getmaxseq(), 8);
        assert_eq!(c.getlen(), 0);
        assert!(c.keys(0).is_none());
        c.append(0, &row([1.0, 2.0]), &row([3.0, 4.0])).unwrap();
        c.append(1, &row([1.0, 2.0]), &row([7.0, 8.0])).unwrap();
        assert_eq!(c.getlen(), 1);
        assert_eq!(c.keys(0).unwrap().getdata(), &[1.0, 2.0]);
        assert_eq!(c.values(1).unwrap().getdata(), &[7.0, 8.0]);
        assert_eq!(c.keys(0).unwrap().getshape().getdims(), &[1, 2]);
        c.append(0, &row([1.0, 2.0]), &row([3.0, 4.0])).unwrap();
        c.append(1, &row([1.0, 2.0]), &row([3.0, 4.0])).unwrap();
        assert_eq!(c.getlen(), 2);
        assert_eq!(c.keys(0).unwrap().getshape().getdims(), &[2, 2]);
        assert_eq!(c.keys(0).unwrap().getdata(), &[1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn appendall_advances_every_layer_together() {
        let mut c = kvstate::new(3, 2, 8).unwrap();
        c.appendall(&row([1.0, 2.0]), &row([3.0, 4.0])).unwrap();
        assert_eq!(c.getlen(), 1);
        for layer in 0..3 {
            assert_eq!(c.keys(layer).unwrap().getdata(), &[1.0, 2.0]);
            assert_eq!(c.values(layer).unwrap().getdata(), &[3.0, 4.0]);
        }
    }

    #[test]
    fn rejects_out_of_range_layers_and_bad_shapes() {
        let mut c = kvstate::new(1, 2, 8).unwrap();
        assert!(c.append(2, &row([0.0, 0.0]), &row([0.0, 0.0])).is_err());
        let wide = tensor::new(shape::new(vec![1, 3]).unwrap(), vec![0.0; 3]).unwrap();
        assert!(c.append(0, &wide, &row([0.0, 0.0])).is_err());
        let flat = tensor::new(shape::new(vec![2]).unwrap(), vec![0.0; 2]).unwrap();
        assert!(c.append(0, &flat, &row([0.0, 0.0])).is_err());
    }

    #[test]
    fn rejects_appends_beyond_capacity() {
        let mut c = kvstate::new(1, 2, 1).unwrap();
        c.append(0, &row([1.0, 2.0]), &row([3.0, 4.0])).unwrap();
        assert!(c.append(0, &row([1.0, 2.0]), &row([3.0, 4.0])).is_err());
    }

    #[test]
    fn clear_resets_every_layer() {
        let mut c = kvstate::new(2, 2, 8).unwrap();
        c.appendall(&row([1.0, 2.0]), &row([3.0, 4.0])).unwrap();
        assert_eq!(c.getlen(), 1);
        c.clear();
        assert_eq!(c.getlen(), 0);
        assert!(c.keys(0).is_none());
        assert!(c.values(1).is_none());
    }
}

/// Tests relocated from `crates/infer/src/request.rs`.
mod request {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::infer::generationparams;
    use ::infer::generationrequest;
    use ::infer::infererror;
    use ::infer::samplingparams;
    use ::infer::stopcondition;
    use models::kvcache;

    fn params(maxnewtokens: usize, maxseq: usize) -> generationparams {
        generationparams::new(
            stopcondition::new(vec![], maxnewtokens).unwrap(),
            samplingparams::greedy(),
            maxseq,
            2,
            4,
        )
        .unwrap()
    }

    #[test]
    fn rejects_an_empty_prompt() {
        assert_eq!(
            generationrequest::new(vec![], params(2, 8)).map(|_| ()),
            Err(infererror::invalidrequest {
                field: "prompt",
                detail: "must be non-empty".to_string(),
            })
        );
    }

    #[test]
    fn rejects_a_prompt_that_cannot_fit_the_cap() {
        assert!(generationrequest::new(vec![1, 2, 3], params(6, 8)).is_err());
    }

    #[test]
    fn tracks_context_growth_and_resets() {
        let mut request = generationrequest::new(vec![1, 2], params(4, 8)).unwrap();
        assert_eq!(request.getcontext(), &[1, 2]);
        assert_eq!(request.getgeneratedlen(), 0);
        request.pushgenerated(3).unwrap();
        assert_eq!(request.getcontext(), &[1, 2, 3]);
        assert_eq!(request.getgenerated(), &[3]);
        assert_eq!(request.getgeneratedlen(), 1);
        request.reset();
        assert_eq!(request.getcontext(), &[1, 2]);
        assert_eq!(request.getgeneratedlen(), 0);
    }

    #[test]
    fn pushgenerated_enforces_the_context_bound() {
        let mut request = generationrequest::new(vec![1], params(2, 3)).unwrap();
        request.pushgenerated(1).unwrap();
        request.pushgenerated(1).unwrap();
        assert!(request.pushgenerated(1).is_err());
    }

    #[test]
    fn owns_a_request_scoped_cache() {
        let mut request = generationrequest::new(vec![1], params(2, 8)).unwrap();
        assert_eq!(request.getkvcache().getlayers(), 2);
        request.getkvcachemut().clear();
        assert_eq!(request.getkvcache().getlen(), 0);
    }

    #[test]
    fn exposes_its_parameters() {
        let request = generationrequest::new(vec![1], params(3, 9)).unwrap();
        assert_eq!(request.getprompt(), &[1]);
        assert_eq!(request.getmaxnewtokens(), 3);
        assert_eq!(request.getmaxseq(), 9);
        assert_eq!(request.getstop().geteostokens(), &[] as &[usize]);
    }
}

/// Tests relocated from `crates/infer/src/sampling.rs`.
mod sampling {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::infer::infererror;
    use ::infer::sampler;
    use ::infer::samplingparams;
    use lib::seed;

    fn params(temp: f32, topk: Option<usize>, topp: Option<f32>, s: u64) -> samplingparams {
        samplingparams::new(temp, topk, topp, seed::new(s)).unwrap()
    }

    #[test]
    fn greedy_selects_the_largest_logit_and_breaks_ties_by_index() {
        let mut s = sampler::new(samplingparams::greedy());
        assert_eq!(s.getsample(&[1.0, 3.0, 2.0]).unwrap(), 1);
        assert_eq!(s.getsample(&[5.0, 5.0, 1.0]).unwrap(), 0);
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(samplingparams::new(-1.0, None, None, seed::new(0)).is_err());
        assert!(samplingparams::new(f32::NAN, None, None, seed::new(0)).is_err());
        assert!(samplingparams::new(1.0, Some(0), None, seed::new(0)).is_err());
        assert!(samplingparams::new(1.0, None, Some(0.0), seed::new(0)).is_err());
        assert!(samplingparams::new(1.0, None, Some(1.5), seed::new(0)).is_err());
        assert!(samplingparams::new(1.0, None, Some(0.5), seed::new(0)).is_ok());
    }

    #[test]
    fn rejects_empty_and_non_finite_logits() {
        let mut s = sampler::new(params(1.0, None, None, 0));
        assert_eq!(s.getsample(&[]), Err(infererror::emptylogits));
        assert_eq!(
            s.getsample(&[0.0, f32::INFINITY]),
            Err(infererror::degeneratelogits)
        );
    }

    #[test]
    fn topp_one_is_a_plain_categorical_draw() {
        let mut s = sampler::new(params(1.0, None, Some(1.0), 7));
        assert!(s.getsample(&[0.0, 0.0, 0.0]).is_ok());
    }

    #[test]
    fn topk_one_always_selects_the_largest_logit() {
        let mut s = sampler::new(params(1.0, Some(1), None, 3));
        for _ in 0..8 {
            assert_eq!(s.getsample(&[0.0, 9.0, 1.0]).unwrap(), 1);
        }
    }

    #[test]
    fn a_dominant_logit_survives_a_tiny_top_p() {
        let mut s = sampler::new(params(1.0, None, Some(0.01), 5));
        assert_eq!(s.getsample(&[0.0, 40.0, 0.0]).unwrap(), 1);
    }

    #[test]
    fn sampling_is_reproducible_for_a_fixed_seed() {
        let logits = [0.1_f32, 0.2, 0.3, 0.4];
        let mut a = sampler::new(params(1.0, None, None, 99));
        let mut b = sampler::new(params(1.0, None, None, 99));
        for _ in 0..16 {
            assert_eq!(a.getsample(&logits).unwrap(), b.getsample(&logits).unwrap());
        }
    }

    #[test]
    fn reset_rewinds_the_stream() {
        let logits = [0.1_f32, 0.2, 0.3, 0.4];
        let mut s = sampler::new(params(1.0, None, None, 11));
        let first = s.getsample(&logits).unwrap();
        let second = s.getsample(&logits).unwrap();
        s.reset();
        assert_eq!(s.getsample(&logits).unwrap(), first);
        assert_eq!(s.getsample(&logits).unwrap(), second);
    }
}

/// Tests relocated from `crates/infer/src/stopping.rs`.
mod stopping {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::infer::infererror;
    use ::infer::stopcondition;

    #[test]
    fn rejects_a_zero_new_token_cap() {
        assert_eq!(
            stopcondition::new(vec![1], 0),
            Err(infererror::invalidrequest {
                field: "maxnewtokens",
                detail: "must be non-zero".to_string(),
            })
        );
    }

    #[test]
    fn exposes_its_criteria_and_matches_eos() {
        let stop = stopcondition::new(vec![2, 4], 8).unwrap();
        assert_eq!(stop.getmaxnewtokens(), 8);
        assert_eq!(stop.geteostokens(), &[2, 4]);
        assert!(stop.iseos(4));
        assert!(!stop.iseos(3));
    }
}

/// Tests relocated from `crates/infer/src/stream.rs`.
mod stream {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::infer::collectingsink;
    use ::infer::stopreason;
    use ::infer::streamsink;

    #[test]
    fn records_tokens_and_the_terminal_reason() {
        let mut sink = collectingsink::new();
        assert!(sink.ongenerated(0, 7));
        assert!(sink.ongenerated(1, 9));
        sink.onstop(stopreason::maxlength);
        assert_eq!(sink.gettokens(), &[7, 9]);
        assert_eq!(sink.getreason(), Some(stopreason::maxlength));
    }
}

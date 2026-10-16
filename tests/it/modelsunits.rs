//! Relocated unit tests for the `models` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/models/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/models/src/causal.rs`.
mod causal {
    use ::models::kvcache;
    use ::models::modelerror;
    use ::tensor::{shape, tensor};
    use std::collections::BTreeMap;

    /// An in-test cache used to prove the contract is implementable.
    #[derive(Default)]
    struct stubcache {
        layers: usize,
        maxseq: usize,
        keys: BTreeMap<usize, tensor<f32>>,
        values: BTreeMap<usize, tensor<f32>>,
    }

    impl kvcache for stubcache {
        fn getlayers(&self) -> usize {
            self.layers
        }

        fn getlen(&self) -> usize {
            self.keys.values().map(|t| t.getshape().getdims()[0]).sum()
        }

        fn getmaxseq(&self) -> usize {
            self.maxseq
        }

        fn append(
            &mut self,
            layer: usize,
            key: &tensor<f32>,
            value: &tensor<f32>,
        ) -> Result<(), modelerror> {
            if layer >= self.layers {
                return Err(modelerror::config("layer", "out of range"));
            }
            self.keys.insert(layer, key.clone());
            self.values.insert(layer, value.clone());
            Ok(())
        }

        fn keys(&self, layer: usize) -> Option<&tensor<f32>> {
            self.keys.get(&layer)
        }

        fn values(&self, layer: usize) -> Option<&tensor<f32>> {
            self.values.get(&layer)
        }
    }

    #[test]
    fn a_cache_contract_is_implementable_and_reports_state() {
        let mut c = stubcache {
            layers: 2,
            maxseq: 8,
            ..stubcache::default()
        };
        assert_eq!(c.getlayers(), 2);
        assert_eq!(c.getmaxseq(), 8);
        assert_eq!(c.getlen(), 0);
        let k = tensor::new(shape::new(vec![1, 2]).unwrap(), vec![1.0, 2.0]).unwrap();
        let v = tensor::new(shape::new(vec![1, 2]).unwrap(), vec![3.0, 4.0]).unwrap();
        c.append(0, &k, &v).unwrap();
        c.append(1, &k, &v).unwrap();
        assert_eq!(c.getlen(), 2);
        assert_eq!(c.keys(0).unwrap().getdata(), &[1.0, 2.0]);
        assert!(c.append(2, &k, &v).is_err());
        assert!(c.keys(9).is_none());
    }
}

/// Tests relocated from `crates/models/src/error.rs`.
mod error {
    use ::models::modelerror;
    use ::tensor::tensorerror;

    #[test]
    fn converts_from_tensorerror() {
        let e: modelerror = tensorerror::emptyshape.into();
        assert_eq!(e.to_string(), "a shape must have at least one dimension");
    }

    #[test]
    fn conversions_keep_underlying_context() {
        let e: modelerror = ::api::backenderror::sizeoverflow.into();
        assert_eq!(e.to_string(), "size overflow");
    }

    #[test]
    fn config_and_unknown_and_missing_are_distinct() {
        assert_eq!(
            modelerror::config("dim", "must be non-zero").to_string(),
            "invalid configuration for dim: must be non-zero"
        );
        assert_eq!(
            modelerror::unknownarchitecture {
                name: "gpt".to_string()
            }
            .to_string(),
            "unknown architecture: gpt"
        );
        assert_eq!(
            modelerror::missingweight {
                name: "norm.weight".to_string()
            }
            .to_string(),
            "missing weight: norm.weight"
        );
    }
}

/// Tests relocated from `crates/models/src/llama/block.rs`.
///
/// The former unit test read the block's private operator fields. The block's
/// public contract is exercised through construction; the operator wiring
/// (per-head rotary for query and key, GQA) is covered end to end by
/// `tests/modelforward.rs`.
mod llama_block {
    use ::models::{llamablock, llamaconfig};

    #[test]
    fn new_builds_operators_from_a_valid_configuration() {
        let c = llamaconfig::new(5, 8, 1, 4, 2, 16, 1.0e-5, 10_000.0, 8).unwrap();
        assert!(llamablock::new(&c).is_ok());
    }

    #[test]
    fn new_rejects_an_inconsistent_configuration() {
        // dim = 6 is not divisible by nheads = 4; the config itself is rejected.
        assert!(llamaconfig::new(5, 6, 1, 4, 1, 8, 1.0e-5, 10_000.0, 8).is_err());
    }
}

/// Tests relocated from `crates/models/src/llama/config.rs`.
mod llama_config {
    use ::models::llamaconfig;

    fn tiny() -> llamaconfig {
        llamaconfig::new(5, 4, 1, 1, 1, 8, 1.0e-5, 10_000.0, 8).unwrap()
    }

    #[test]
    fn derived_dimensions_are_computed() {
        let c = tiny();
        assert_eq!(c.getheaddim(), 4);
        assert_eq!(c.getkvwidth(), 4);
        assert_eq!(c.getvocab(), 5);
        assert_eq!(c.getmaxseq(), 8);
    }

    #[test]
    fn grouped_query_head_dimensions() {
        let c = llamaconfig::new(10, 8, 2, 4, 2, 16, 1.0e-5, 10_000.0, 16).unwrap();
        assert_eq!(c.getheaddim(), 2);
        assert_eq!(c.getkvwidth(), 4);
    }

    #[test]
    fn invalid_dimensions_are_rejected() {
        assert!(llamaconfig::new(0, 4, 1, 1, 1, 8, 1.0e-5, 1.0, 8).is_err());
        assert!(llamaconfig::new(5, 4, 1, 3, 1, 8, 1.0e-5, 1.0, 8).is_err());
        assert!(llamaconfig::new(5, 4, 1, 4, 3, 8, 1.0e-5, 1.0, 8).is_err());
        assert!(llamaconfig::new(5, 6, 1, 2, 1, 8, 1.0e-5, 1.0, 8).is_err());
        assert!(llamaconfig::new(5, 4, 1, 1, 1, 8, -1.0, 1.0, 8).is_err());
        assert!(llamaconfig::new(5, 4, 1, 1, 1, 8, 1.0e-5, 0.0, 8).is_err());
    }
}

/// Tests relocated from `crates/models/src/llama/model.rs`.
mod llama_model {
    use ::models::causallm;
    use ::models::llamaconfig;
    use ::models::llamamodel;
    use ::models::llamaweights;
    use lib::seed;

    fn tiny() -> llamaconfig {
        llamaconfig::new(5, 4, 1, 1, 1, 8, 1.0e-5, 10_000.0, 8).unwrap()
    }

    #[test]
    fn model_reports_the_configuration_it_was_built_with() {
        let c = tiny();
        let w = llamaweights::random(&c, seed::new(1)).unwrap();
        let m = llamamodel::new(c, w).unwrap();
        assert_eq!(m.getconfig().getvocab(), 5);
        assert_eq!(m.getweights().getlayers().len(), 1);
    }

    #[test]
    fn model_rejects_a_weight_layer_count_mismatch() {
        let c = tiny();
        let w = llamaweights::random(&c, seed::new(1)).unwrap();
        let other = llamaconfig::new(5, 4, 2, 1, 1, 8, 1.0e-5, 10_000.0, 8).unwrap();
        assert!(llamamodel::new(other, w).is_err());
    }
}

/// Tests relocated from `crates/models/src/llama/weights.rs`.
mod llama_weights {
    use ::models::llamaconfig;
    use ::models::llamaweights;
    use lib::seed;

    fn tiny() -> llamaconfig {
        llamaconfig::new(5, 4, 2, 1, 1, 8, 1.0e-5, 10_000.0, 8).unwrap()
    }

    #[test]
    fn random_weights_are_reproducible() {
        let c = tiny();
        let a = llamaweights::random(&c, seed::new(11)).unwrap();
        let b = llamaweights::random(&c, seed::new(11)).unwrap();
        let d = llamaweights::random(&c, seed::new(12)).unwrap();
        assert_eq!(a.getlayers().len(), 2);
        assert_eq!(a.getembedding().getvocab(), 5);
        assert_ne!(
            a.getoutput().getweight().getvalue().getdata(),
            d.getoutput().getweight().getvalue().getdata()
        );
        assert_eq!(
            b.getoutput().getweight().getvalue().getdata(),
            a.getoutput().getweight().getvalue().getdata()
        );
    }

    #[test]
    fn collect_produces_hierarchical_names() {
        use nn::module;
        let c = tiny();
        let w = llamaweights::random(&c, seed::new(3)).unwrap();
        let d = w.getstatedict().unwrap();
        assert!(d.get("embedding.weight").is_some());
        assert!(d.get("layers.0.wq.weight").is_some());
        assert!(d.get("layers.1.ffn.gate.weight").is_some());
        assert!(d.get("norm.weight").is_some());
        assert!(d.get("output.weight").is_some());
    }
}

/// Tests relocated from `crates/models/src/registry.rs`.
mod registry {
    use ::models::architecture;
    use ::models::causallm;
    use ::models::getarchitecture;
    use ::models::modelerror;
    use ::models::{llamaconfig, llamaweights};
    use lib::seed;

    #[test]
    fn known_names_resolve_and_report_themselves() {
        let a = getarchitecture("llama").unwrap();
        assert_eq!(a, architecture::llama);
        assert_eq!(a.getname(), "llama");
    }

    #[test]
    fn unknown_names_are_rejected() {
        assert_eq!(
            getarchitecture("gpt").map(|_| ()),
            Err(modelerror::unknownarchitecture {
                name: "gpt".to_string()
            })
        );
    }

    #[test]
    fn newmodel_produces_a_model() {
        let c = llamaconfig::new(5, 4, 1, 1, 1, 8, 1.0e-5, 10_000.0, 8).unwrap();
        let w = llamaweights::random(&c, seed::new(1)).unwrap();
        let m = getarchitecture("llama").unwrap().newmodel(c, w).unwrap();
        assert_eq!(m.getconfig().getvocab(), 5);
    }

    #[test]
    fn mamba_resolves_and_refuses_newmodel() {
        let a = getarchitecture("mamba").unwrap();
        assert_eq!(a, architecture::mamba);
        assert_eq!(a.getname(), "mamba");
        let c = llamaconfig::new(5, 4, 1, 1, 1, 8, 1.0e-5, 10_000.0, 8).unwrap();
        let w = llamaweights::random(&c, seed::new(1)).unwrap();
        assert!(matches!(a.newmodel(c, w), Err(modelerror::invalidconfig { .. })));
    }
}

//! Relocated unit tests for the `nn` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/nn/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/nn/src/dictionary.rs`.
mod dictionary {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::joinname;
    use ::nn::nnerror;
    use ::nn::parameter;
    use ::nn::statedict;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn joinname_composes_and_omits_an_empty_prefix() {
        assert_eq!(joinname("", "weight"), "weight");
        assert_eq!(joinname("q_proj", "weight"), "q_proj.weight");
    }

    #[test]
    fn insert_rejects_duplicates() {
        let mut d = statedict::new();
        d.insert("w", build(vec![1], vec![1.0])).unwrap();
        assert_eq!(
            d.insert("w", build(vec![1], vec![2.0])),
            Err(nnerror::duplicateparameter {
                name: "w".to_string()
            })
        );
        assert_eq!(d.getlen(), 1);
    }

    #[test]
    fn add_prefixes_the_leaf_name() {
        let mut d = statedict::new();
        d.add("block", parameter::new("weight", build(vec![1], vec![3.0])))
            .unwrap();
        assert!(d.get("block.weight").is_some());
        assert!(d.get("weight").is_none());
    }

    #[test]
    fn names_are_sorted_and_iteration_is_deterministic() {
        let mut d = statedict::new();
        d.insert("b", build(vec![1], vec![1.0])).unwrap();
        d.insert("a", build(vec![1], vec![2.0])).unwrap();
        assert_eq!(d.names().collect::<Vec<_>>(), vec!["a", "b"]);
        assert!(!d.isempty());
    }

    #[test]
    fn getmut_updates_in_place() {
        let mut d = statedict::new();
        d.insert("w", build(vec![2], vec![1.0, 2.0])).unwrap();
        d.getmut("w").unwrap().getdatamut()[0] = 9.0;
        assert_eq!(d.get("w").unwrap().getdata(), &[9.0, 2.0]);
    }
}

/// Tests relocated from `crates/nn/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::nnerror;
    use ::tensor::{shape, tensorerror};
    use api::backenderror;

    #[test]
    fn converts_from_backenderror() {
        let e: nnerror = backenderror::sizeoverflow.into();
        assert_eq!(e.to_string(), "size overflow");
    }

    #[test]
    fn converts_from_tensorerror() {
        let e: nnerror = tensorerror::emptyshape.into();
        assert_eq!(e.to_string(), "a shape must have at least one dimension");
    }

    #[test]
    fn shape_helper_reports_both_shapes() {
        let expected = shape::new(vec![2, 4]).unwrap();
        let actual = shape::new(vec![3, 4]).unwrap();
        assert_eq!(
            nnerror::shapes("linear", &expected, &actual).to_string(),
            "layer linear expected shape [2, 4] but received [3, 4]"
        );
    }

    #[test]
    fn config_helper_reports_field_and_detail() {
        assert_eq!(
            nnerror::config("heads", "must be non-zero").to_string(),
            "invalid configuration for heads: must be non-zero"
        );
    }

    #[test]
    fn duplicate_and_missing_names_are_distinct() {
        let d = nnerror::duplicateparameter {
            name: "w".to_string(),
        };
        let m = nnerror::missingparameter {
            name: "w".to_string(),
        };
        assert_eq!(d.to_string(), "duplicate parameter name: w");
        assert_eq!(m.to_string(), "missing parameter name: w");
        assert_ne!(d.to_string(), m.to_string());
    }
}

/// Tests relocated from `crates/nn/src/init.rs`.
mod init {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::fillnormal;
    use ::nn::fillone;
    use ::nn::filluniform;
    use ::nn::fillvalue;
    use ::nn::fillzero;
    use ::nn::initxavier;
    use ::tensor::shape;
    use lib::seed;

    #[test]
    fn fillzero_and_fillone_fill_every_element() {
        assert_eq!(
            fillzero(shape::new(vec![2, 3]).unwrap()).unwrap().getdata(),
            &[0.0; 6]
        );
        assert_eq!(
            fillone(shape::new(vec![2]).unwrap()).unwrap().getdata(),
            &[1.0, 1.0]
        );
        assert_eq!(
            fillvalue(shape::new(vec![1]).unwrap(), 7.0)
                .unwrap()
                .getdata(),
            &[7.0]
        );
    }

    #[test]
    fn filluniform_is_reproducible_for_equal_seeds() {
        let s = shape::new(vec![8]).unwrap();
        let a = filluniform(s.clone(), -1.0, 1.0, seed::new(42)).unwrap();
        let b = filluniform(s.clone(), -1.0, 1.0, seed::new(42)).unwrap();
        let c = filluniform(s, -1.0, 1.0, seed::new(43)).unwrap();
        assert_eq!(a.getdata(), b.getdata());
        assert_ne!(a.getdata(), c.getdata());
    }

    #[test]
    fn filluniform_stays_within_bounds() {
        let t = filluniform(shape::new(vec![256]).unwrap(), -2.0, 5.0, seed::new(7)).unwrap();
        assert!(t.getdata().iter().all(|&v| (-2.0..5.0).contains(&v)));
    }

    #[test]
    fn fillnormal_is_reproducible_and_finite() {
        let s = shape::new(vec![64]).unwrap();
        let a = fillnormal(s.clone(), 0.0, 1.0, seed::new(1)).unwrap();
        let b = fillnormal(s, 0.0, 1.0, seed::new(1)).unwrap();
        assert_eq!(a.getdata(), b.getdata());
        assert!(a.getdata().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn xavier_limit_matches_the_fan_sum() {
        let t = initxavier(shape::new(vec![4, 12]).unwrap(), seed::new(9)).unwrap();
        let limit = (6.0_f32 / 16.0).sqrt();
        assert!(t.getdata().iter().all(|&v| (-limit..limit).contains(&v)));
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        assert!(filluniform(shape::new(vec![1]).unwrap(), 2.0, 1.0, seed::new(0)).is_err());
        assert!(fillnormal(shape::new(vec![1]).unwrap(), 0.0, -1.0, seed::new(0)).is_err());
        assert!(initxavier(shape::new(vec![4]).unwrap(), seed::new(0)).is_err());
    }
}

/// Tests relocated from `crates/nn/src/layers/gated.rs`.
mod layers_gated {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::linear;
    use ::nn::mlp;
    use ::nn::module;
    use ::nn::parameter;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    fn projection(name: &str, in_features: usize, out_features: usize) -> linear {
        linear::new(
            parameter::new(
                name,
                build(
                    vec![in_features, out_features],
                    vec![0.0; in_features * out_features],
                ),
            ),
            None,
        )
        .unwrap()
    }

    #[test]
    fn reports_features_and_collects_prefixed_names() {
        let m = mlp::new(
            projection("weight", 4, 8),
            projection("weight", 4, 8),
            projection("weight", 8, 4),
        )
        .unwrap();
        assert_eq!(m.getfeatures(), 4);
        assert_eq!(m.gethidden(), 8);
        assert_eq!(m.getname(), "mlp");
        let d = m.getstatedict().unwrap();
        assert_eq!(d.getlen(), 3);
        assert!(d.get("gate.weight").is_some());
        assert!(d.get("up.weight").is_some());
        assert!(d.get("down.weight").is_some());
    }

    #[test]
    fn rejects_mismatched_projection_shapes() {
        assert!(
            mlp::new(
                projection("weight", 4, 8),
                projection("weight", 4, 9),
                projection("weight", 8, 4),
            )
            .is_err()
        );
        assert!(
            mlp::new(
                projection("weight", 4, 8),
                projection("weight", 4, 8),
                projection("weight", 7, 4),
            )
            .is_err()
        );
    }
}

/// Tests relocated from `crates/nn/src/layers/lookup.rs`.
mod layers_lookup {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::embedding;
    use ::nn::module;
    use ::nn::parameter;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn reports_vocab_and_width() {
        let w = parameter::new("weight", build(vec![7, 3], vec![0.0; 21]));
        let e = embedding::new(w).unwrap();
        assert_eq!(e.getvocab(), 7);
        assert_eq!(e.getwidth(), 3);
        assert_eq!(e.getname(), "embedding");
    }

    #[test]
    fn rejects_a_rank_one_table() {
        let w = parameter::new("weight", build(vec![4], vec![0.0; 4]));
        assert!(embedding::new(w).is_err());
    }

    #[test]
    fn collects_the_table() {
        let w = parameter::new("weight", build(vec![2, 2], vec![0.0; 4]));
        let e = embedding::new(w).unwrap();
        let d = e.getstatedict().unwrap();
        assert_eq!(d.getlen(), 1);
        assert!(d.get("weight").is_some());
    }
}

/// Tests relocated from `crates/nn/src/layers/mod.rs`.
///
/// The former unit tests targeted crate-private helpers (`leadingcount`,
/// `expandrows`, `reshape`). They are covered here through the public
/// `linear::forward` seam: leading batch dimensions, the explicit bias
/// broadcast, and input-shape rejection. The helpers' overflow paths have no
/// public trigger, so they carry no contract test.
mod layers {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::cpu::cpuexecutor;
    use ::lib::limits;
    use ::nn::{linear, parameter};
    use ::tensor::{shape, tensor};

    fn engine() -> cpuexecutor {
        cpuexecutor::new(limits::new(1 << 20, 1 << 24, 1 << 12, 1 << 8))
    }

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn forward_broadcasts_one_bias_row_across_a_batch() {
        // Weight [2, 3] (in=2, out=3), bias [3]; every batch row shares the bias.
        let w = parameter::new("weight", build(vec![2, 3], vec![1.0; 6]));
        let b = parameter::new("bias", build(vec![3], vec![10.0, 20.0, 30.0]));
        let layer = linear::new(w, Some(b)).unwrap();
        let input = build(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]);
        let out = layer.forward(&engine(), &input).unwrap();
        assert_eq!(out.getshape().getdims(), &[2, 3]);
        assert_eq!(out.getdata(), &[12.0, 22.0, 32.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn forward_preserves_multiple_leading_dimensions() {
        let w = parameter::new("weight", build(vec![2, 3], vec![1.0; 6]));
        let layer = linear::new(w, None).unwrap();
        let input = build(vec![2, 2, 2], vec![1.0; 8]);
        let out = layer.forward(&engine(), &input).unwrap();
        assert_eq!(out.getshape().getdims(), &[2, 2, 3]);
        assert_eq!(out.getdata(), &[2.0; 12]);
    }

    #[test]
    fn forward_rejects_a_feature_mismatch() {
        let w = parameter::new("weight", build(vec![2, 3], vec![1.0; 6]));
        let layer = linear::new(w, None).unwrap();
        let input = build(vec![4], vec![1.0; 4]);
        assert!(layer.forward(&engine(), &input).is_err());
    }
}

/// Tests relocated from `crates/nn/src/layers/multihead.rs`.
mod layers_multihead {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::attention;
    use ::nn::module;

    #[test]
    fn reports_configuration() {
        let a = attention::new(8, 2, true).unwrap();
        assert_eq!(a.getheads(), 8);
        assert_eq!(a.getkvheads(), 2);
        assert!(a.getcausal());
        assert_eq!(a.getname(), "attention");
    }

    #[test]
    fn rejects_inconsistent_head_counts() {
        assert!(attention::new(0, 1, true).is_err());
        assert!(attention::new(4, 3, true).is_err());
    }
}

/// Tests relocated from `crates/nn/src/layers/normalize.rs`.
mod layers_normalize {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::DEFAULTEPS;
    use ::nn::module;
    use ::nn::parameter;
    use ::nn::rmsnorm;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn reports_features_and_epsilon() {
        let w = parameter::new("weight", build(vec![4], vec![1.0; 4]));
        let n = rmsnorm::new(w, DEFAULTEPS).unwrap();
        assert_eq!(n.getfeatures(), 4);
        assert_eq!(n.geteps(), DEFAULTEPS);
        assert_eq!(n.getname(), "rmsnorm");
    }

    #[test]
    fn rejects_a_rank_two_weight_and_bad_epsilon() {
        let w = parameter::new("weight", build(vec![2, 2], vec![0.0; 4]));
        assert!(rmsnorm::new(w, DEFAULTEPS).is_err());
        let w = parameter::new("weight", build(vec![4], vec![0.0; 4]));
        assert!(rmsnorm::new(w, -1.0).is_err());
    }
}

/// Tests relocated from `crates/nn/src/layers/projection.rs`.
mod layers_projection {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::linear;
    use ::nn::module;
    use ::nn::parameter;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn reports_feature_counts() {
        let w = parameter::new("weight", build(vec![3, 5], vec![0.0; 15]));
        let l = linear::new(w, None).unwrap();
        assert_eq!(l.getinfeatures(), 3);
        assert_eq!(l.getoutfeatures(), 5);
        assert_eq!(l.getname(), "linear");
    }

    #[test]
    fn rejects_a_rank_one_weight() {
        let w = parameter::new("weight", build(vec![4], vec![0.0; 4]));
        assert!(linear::new(w, None).is_err());
    }

    #[test]
    fn rejects_a_mismatched_bias() {
        let w = parameter::new("weight", build(vec![3, 5], vec![0.0; 15]));
        let b = parameter::new("bias", build(vec![4], vec![0.0; 4]));
        assert!(linear::new(w, Some(b)).is_err());
    }

    #[test]
    fn collects_weight_and_bias() {
        let w = parameter::new("weight", build(vec![3, 5], vec![0.0; 15]));
        let b = parameter::new("bias", build(vec![5], vec![0.0; 5]));
        let l = linear::new(w, Some(b)).unwrap();
        let d = l.getstatedict().unwrap();
        assert_eq!(d.getlen(), 2);
        assert!(d.get("weight").is_some());
        assert!(d.get("bias").is_some());
    }
}

/// Tests relocated from `crates/nn/src/layers/rotate.rs`.
mod layers_rotate {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::DEFAULTBASE;
    use ::nn::module;
    use ::nn::rotary;

    #[test]
    fn reports_heads_and_base() {
        let r = rotary::new(4, DEFAULTBASE).unwrap();
        assert_eq!(r.getheads(), 4);
        assert_eq!(r.getbase(), DEFAULTBASE);
        assert_eq!(r.getname(), "rotary");
    }

    #[test]
    fn rejects_zero_heads_and_bad_base() {
        assert!(rotary::new(0, DEFAULTBASE).is_err());
        assert!(rotary::new(2, 0.0).is_err());
    }
}

/// Tests relocated from `crates/nn/src/modules.rs`.
mod modules {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::module;
    use ::nn::nnerror;
    use ::nn::parameter;
    use ::nn::statedict;
    use ::tensor::{shape, tensor};

    /// A module holding one parameter, used to exercise the default method.
    struct onew {
        value: parameter,
    }

    impl module for onew {
        fn getname(&self) -> &'static str {
            "onew"
        }

        fn collect(&self, prefix: &str, dict: &mut statedict) -> Result<(), nnerror> {
            dict.add(prefix, self.value.clone())
        }
    }

    #[test]
    fn default_statedict_uses_the_root_prefix() {
        let value = parameter::new(
            "weight",
            tensor::new(shape::new(vec![1]).unwrap(), vec![1.0]).unwrap(),
        );
        let m = onew { value };
        assert_eq!(m.getname(), "onew");
        let dict = m.getstatedict().unwrap();
        assert_eq!(dict.getlen(), 1);
        assert!(dict.get("weight").is_some());
    }
}

/// Tests relocated from `crates/nn/src/parameters.rs`.
mod parameters {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::nn::parameter;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    #[test]
    fn reports_name_shape_and_length() {
        let p = parameter::new("weight", build(vec![2, 3], vec![0.0; 6]));
        assert_eq!(p.getname(), "weight");
        assert_eq!(p.getshape().getdims(), &[2, 3]);
        assert_eq!(p.getlen(), 6);
        assert_eq!(p.getvalue().getdata(), &[0.0; 6]);
    }

    #[test]
    fn equality_covers_name_and_value() {
        let a = parameter::new("w", build(vec![1], vec![1.0]));
        let b = parameter::new("w", build(vec![1], vec![1.0]));
        let c = parameter::new("w", build(vec![1], vec![2.0]));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}

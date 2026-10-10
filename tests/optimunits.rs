//! Relocated unit tests for the `optim` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/optim/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/decision-records/blackboxtesting.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/optim/src/adamws.rs`.
mod adamws {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::optim::adamw;
    use ::optim::adamwconfig;
    use ::optim::clipconfig;
    use ::optim::optimizer;
    use ::optim::{optimizerstate, statetensor};
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    fn config() -> adamwconfig {
        adamwconfig::new(0.1, 0.9, 0.999, 1e-8, 0.0).unwrap()
    }

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn first_update_matches_the_documented_equation() {
        let mut opt = adamw::new(config());
        let mut params = [build(vec![1], vec![0.0])];
        let mut grads = [build(vec![1], vec![2.0])];
        opt.update(&mut params, &mut grads).unwrap();
        // m_hat / (sqrt(v_hat) + eps) = 2 / (2 + eps) ~= 1, so p ~= -lr.
        assert!(approx(params[0].getdata()[0], -0.1, 1e-6));
        assert_eq!(opt.getstep(), 1);
        assert_eq!(opt.getparamcount(), 1);
    }

    #[test]
    fn constant_gradient_steps_by_exactly_the_learning_rate() {
        let mut opt = adamw::new(config());
        let mut params = [build(vec![1], vec![0.0])];
        let mut grads = [build(vec![1], vec![2.0])];
        for _ in 0..3 {
            opt.update(&mut params, &mut grads).unwrap();
        }
        assert!(approx(params[0].getdata()[0], -0.3, 1e-5));
        assert_eq!(opt.getstep(), 3);
    }

    #[test]
    fn zero_gradient_only_applies_decoupled_weight_decay() {
        let cfg = adamwconfig::new(0.1, 0.9, 0.999, 1e-8, 0.1).unwrap();
        let mut opt = adamw::new(cfg);
        let mut params = [build(vec![1], vec![1.0])];
        let mut grads = [build(vec![1], vec![0.0])];
        opt.update(&mut params, &mut grads).unwrap();
        // m_hat = 0, v_hat = 0 -> momentum term 0; decay term 0.1 * 1.0.
        assert!(approx(params[0].getdata()[0], 0.99, 1e-6));
    }

    #[test]
    fn update_clips_gradients_when_configured() {
        let cfg = config().withclip(clipconfig::bynorm(1.0).unwrap());
        let mut opt = adamw::new(cfg);
        let mut params = [build(vec![2], vec![0.0, 0.0])];
        let mut grads = [build(vec![2], vec![3.0, 4.0])];
        opt.update(&mut params, &mut grads).unwrap();
        // The gradient is scaled to norm 1 before use.
        assert!(approx(grads[0].getdata()[0], 0.6, 1e-6));
        assert!(approx(grads[0].getdata()[1], 0.8, 1e-6));
        assert!(params[0].getdata()[0] < 0.0);
    }

    #[test]
    fn setlr_changes_the_effective_rate() {
        let mut opt = adamw::new(config());
        assert_eq!(opt.getlr(), 0.1);
        opt.setlr(0.25).unwrap();
        assert_eq!(opt.getlr(), 0.25);
        assert!(opt.setlr(-0.1).is_err());
    }

    #[test]
    fn update_rejects_operand_mismatches_without_mutating() {
        let mut opt = adamw::new(config());
        let mut params = [build(vec![2], vec![1.0, 2.0])];
        let mut grads = [build(vec![3], vec![0.0, 0.0, 0.0])];
        assert!(opt.update(&mut params, &mut grads).is_err());
        assert_eq!(params[0].getdata(), &[1.0, 2.0]);
        assert_eq!(opt.getstep(), 0);
        assert_eq!(opt.getparamcount(), 0);
    }

    #[test]
    fn update_rejects_a_changed_parameter_set() {
        let mut opt = adamw::new(config());
        let mut params = [build(vec![2], vec![0.0, 0.0])];
        let mut grads = [build(vec![2], vec![1.0, 1.0])];
        opt.update(&mut params, &mut grads).unwrap();
        let mut other = [build(vec![3], vec![0.0, 0.0, 0.0])];
        let mut othergrads = [build(vec![3], vec![1.0, 1.0, 1.0])];
        assert!(opt.update(&mut other, &mut othergrads).is_err());
    }

    #[test]
    fn state_roundtrip_restores_identical_updates() {
        let mut original = adamw::new(config());
        let mut params = [build(vec![2], vec![1.0, -1.0])];
        let mut grads = [build(vec![2], vec![0.5, 0.25])];
        original.update(&mut params, &mut grads).unwrap();
        let snapshot = original.getstate();
        assert_eq!(snapshot.getstep(), 1);
        assert_eq!(snapshot.getparamcount(), 1);
        assert_eq!(snapshot.gettensors().len(), 2);

        let mut restored = adamw::new(config());
        restored.restore(&snapshot).unwrap();

        // Continue both optimizers from the same post-step state.
        let mut a_params = [params[0].clone()];
        let mut a_grads = [grads[0].clone()];
        let mut b_params = [params[0].clone()];
        let mut b_grads = [grads[0].clone()];
        original.update(&mut a_params, &mut a_grads).unwrap();
        restored.update(&mut b_params, &mut b_grads).unwrap();
        assert_eq!(b_params[0].getdata(), a_params[0].getdata());
    }

    #[test]
    fn empty_snapshot_restores_cleanly() {
        let mut opt = adamw::new(config());
        let state = opt.getstate();
        assert_eq!(state.getparamcount(), 0);
        assert!(opt.restore(&state).is_ok());
    }

    #[test]
    fn restore_rejects_a_tensor_count_mismatch() {
        let mut opt = adamw::new(config());
        let tensor = statetensor::new("m.0", shape::new(vec![1]).unwrap(), vec![0.0]).unwrap();
        let state = optimizerstate::new(1, 0.1, 1, vec![tensor]);
        assert!(matches!(
            opt.restore(&state),
            Err(::optim::opterror::invalidstate { .. })
        ));
    }

    #[test]
    fn restore_rejects_an_unknown_slot_name() {
        let mut opt = adamw::new(config());
        let a = statetensor::new("m.0", shape::new(vec![1]).unwrap(), vec![0.0]).unwrap();
        let b = statetensor::new("q.0", shape::new(vec![1]).unwrap(), vec![0.0]).unwrap();
        let state = optimizerstate::new(1, 0.1, 1, vec![a, b]);
        assert!(opt.restore(&state).is_err());
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        assert!(adamwconfig::new(-0.1, 0.9, 0.999, 1e-8, 0.0).is_err());
        assert!(adamwconfig::new(0.1, 1.0, 0.999, 1e-8, 0.0).is_err());
        assert!(adamwconfig::new(0.1, 0.9, 1.0, 1e-8, 0.0).is_err());
        assert!(adamwconfig::new(0.1, 0.9, 0.999, 0.0, 0.0).is_err());
        assert!(adamwconfig::new(0.1, 0.9, 0.999, 1e-8, -1.0).is_err());
    }
}

/// Tests relocated from `crates/optim/src/error.rs`.
mod error {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::optim::opterror;
    use ::tensor::tensorerror;

    #[test]
    fn converts_from_tensorerror() {
        let e: opterror = tensorerror::emptyshape.into();
        assert_eq!(e.to_string(), "a shape must have at least one dimension");
    }

    #[test]
    fn state_and_hyperparameter_helpers_report_their_detail() {
        assert_eq!(
            opterror::state("slot 0 shape mismatch").to_string(),
            "invalid optimizer state: slot 0 shape mismatch"
        );
        assert_eq!(
            opterror::hyperparameter("lr", "must be finite and non-negative").to_string(),
            "invalid hyperparameter lr: must be finite and non-negative"
        );
    }

    #[test]
    fn variants_are_distinguishable_by_value() {
        let a = opterror::mismatchedcount {
            params: 1,
            grads: 2,
        };
        let b = opterror::mismatchedcount {
            params: 1,
            grads: 3,
        };
        assert_ne!(a, b);
        assert_eq!(a.to_string(), "received 1 parameters but 2 gradients");
    }
}

/// Tests relocated from `crates/optim/src/gradientclip.rs`.
mod gradientclip {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::optim::clipconfig;
    use ::optim::clipgradients;
    use ::optim::opterror;
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn none_leaves_gradients_unchanged() {
        let mut grads = [build(vec![2], vec![3.0, -4.0])];
        clipgradients(&mut grads, clipconfig::none());
        assert_eq!(grads[0].getdata(), &[3.0, -4.0]);
        assert!(!clipconfig::none().isenabled());
    }

    #[test]
    fn byvalue_clamps_each_element() {
        let mut grads = [build(vec![3], vec![-2.0, 0.5, 3.0])];
        clipgradients(&mut grads, clipconfig::byvalue(-1.0, 1.0).unwrap());
        assert_eq!(grads[0].getdata(), &[-1.0, 0.5, 1.0]);
        assert!(clipconfig::byvalue(-1.0, 1.0).unwrap().isenabled());
    }

    #[test]
    fn bynorm_scales_the_global_vector() {
        let mut grads = [build(vec![2], vec![3.0, 4.0])];
        clipgradients(&mut grads, clipconfig::bynorm(1.0).unwrap());
        assert!(approx(grads[0].getdata()[0], 0.6, 1e-6));
        assert!(approx(grads[0].getdata()[1], 0.8, 1e-6));
    }

    #[test]
    fn bynorm_uses_one_norm_across_all_gradients() {
        let mut grads = [build(vec![1], vec![3.0]), build(vec![1], vec![4.0])];
        clipgradients(&mut grads, clipconfig::bynorm(1.0).unwrap());
        assert!(approx(grads[0].getdata()[0], 0.6, 1e-6));
        assert!(approx(grads[1].getdata()[0], 0.8, 1e-6));
    }

    #[test]
    fn bynorm_leaves_a_within_bound_gradient_unchanged() {
        let mut grads = [build(vec![2], vec![0.3, 0.4])];
        clipgradients(&mut grads, clipconfig::bynorm(1.0).unwrap());
        assert_eq!(grads[0].getdata(), &[0.3, 0.4]);
    }

    #[test]
    fn bynorm_zero_norm_and_zero_limit_do_not_divide_by_zero() {
        let mut zero = [build(vec![1], vec![0.0])];
        clipgradients(&mut zero, clipconfig::bynorm(0.0).unwrap());
        assert_eq!(zero[0].getdata(), &[0.0]);

        let mut nonzero = [build(vec![2], vec![1.0, 1.0])];
        clipgradients(&mut nonzero, clipconfig::bynorm(0.0).unwrap());
        assert_eq!(nonzero[0].getdata(), &[0.0, 0.0]);
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        assert!(matches!(
            clipconfig::bynorm(-1.0),
            Err(opterror::invalidhyperparameter { .. })
        ));
        assert!(clipconfig::bynorm(f32::NAN).is_err());
        assert!(matches!(
            clipconfig::byvalue(2.0, 1.0),
            Err(opterror::invalidhyperparameter { .. })
        ));
        assert!(clipconfig::byvalue(f32::INFINITY, 1.0).is_err());
    }
}

/// Tests relocated from `crates/optim/src/optimizers.rs`.
///
/// The crate-private validators are exercised indirectly: the operand checks
/// through `optimizer::update` and the hyperparameter checks through
/// `sgdconfig::new`. Internal-only helpers with no public seam (`zerolike`,
/// `slotindex`) are covered by the state round-trip tests instead.
mod optimizers {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::optim::{opterror, optimizer, sgd, sgdconfig};
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    fn sgd_with_momentum() -> sgd {
        sgd::new(sgdconfig::new(0.1, 0.9, 0.0).unwrap())
    }

    #[test]
    fn matching_operands_are_accepted() {
        let mut opt = sgd_with_momentum();
        let mut params = [build(vec![2], vec![1.0, 2.0])];
        let mut grads = [build(vec![2], vec![0.1, 0.2])];
        assert_eq!(opt.update(&mut params, &mut grads), Ok(()));
    }

    #[test]
    fn count_mismatch_is_reported() {
        let mut opt = sgd_with_momentum();
        let mut params = [build(vec![1], vec![1.0])];
        let mut grads: [tensor<f32>; 0] = [];
        assert_eq!(
            opt.update(&mut params, &mut grads),
            Err(opterror::mismatchedcount {
                params: 1,
                grads: 0,
            })
        );
    }

    #[test]
    fn shape_mismatch_reports_both_shapes() {
        let mut opt = sgd_with_momentum();
        let mut params = [build(vec![2], vec![0.0, 0.0])];
        let mut grads = [build(vec![1], vec![0.0])];
        assert_eq!(
            opt.update(&mut params, &mut grads),
            Err(opterror::shapemismatch {
                index: 0,
                param: vec![2],
                grad: vec![1],
            })
        );
    }

    #[test]
    fn slot_mismatch_is_rejected() {
        // A second update over differently shaped parameters is rejected by the
        // already-initialized momentum buffers and mutates nothing.
        let mut opt = sgd_with_momentum();
        let mut params = [build(vec![2], vec![0.0, 0.0])];
        let mut grads = [build(vec![2], vec![1.0, 1.0])];
        opt.update(&mut params, &mut grads).unwrap();
        let mut wider = [build(vec![3], vec![0.0, 0.0, 0.0])];
        let mut wider_grads = [build(vec![3], vec![1.0, 1.0, 1.0])];
        assert!(matches!(
            opt.update(&mut wider, &mut wider_grads),
            Err(opterror::invalidstate { .. })
        ));
    }

    #[test]
    fn invalid_hyperparameters_are_rejected() {
        assert!(sgdconfig::new(-0.1, 0.0, 0.0).is_err());
        assert!(sgdconfig::new(f32::NAN, 0.0, 0.0).is_err());
        assert!(sgdconfig::new(0.1, -1.0, 0.0).is_err());
        assert!(sgdconfig::new(0.1, 0.0, -1.0).is_err());
        assert!(sgdconfig::new(0.0, 0.0, 0.0).is_ok());
    }

    #[test]
    fn name_and_learning_rate_are_reported() {
        let mut opt = sgd_with_momentum();
        assert_eq!(opt.getname(), "sgd");
        assert_eq!(opt.getlr(), 0.1);
        opt.setlr(0.05).unwrap();
        assert_eq!(opt.getlr(), 0.05);
    }
}

/// Tests relocated from `crates/optim/src/scheduler.rs`.
mod scheduler {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::optim::lrschedule;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn constant_never_changes() {
        let s = lrschedule::constant(0.1).unwrap();
        assert_eq!(s.getlr(0), 0.1);
        assert_eq!(s.getlr(1_000_000), 0.1);
    }

    #[test]
    fn steplr_decays_every_interval() {
        let s = lrschedule::steplr(1.0, 10, 0.5, 0.0).unwrap();
        assert!(approx(s.getlr(0), 1.0, 1e-6));
        assert!(approx(s.getlr(9), 1.0, 1e-6));
        assert!(approx(s.getlr(10), 0.5, 1e-6));
        assert!(approx(s.getlr(20), 0.25, 1e-6));
    }

    #[test]
    fn steplr_respects_the_minimum() {
        let s = lrschedule::steplr(1.0, 1, 0.5, 0.2).unwrap();
        assert!(s.getlr(100) >= 0.2);
        assert!(approx(s.getlr(100), 0.2, 1e-6));
    }

    #[test]
    fn exponential_decays_each_step() {
        let s = lrschedule::exponential(1.0, 0.5, 0.0).unwrap();
        assert!(approx(s.getlr(0), 1.0, 1e-6));
        assert!(approx(s.getlr(3), 0.125, 1e-6));
    }

    #[test]
    fn cosine_annealing_hits_its_boundaries() {
        let s = lrschedule::cosineannealing(1.0, 0.0, 100).unwrap();
        assert!(approx(s.getlr(0), 1.0, 1e-6));
        assert!(approx(s.getlr(50), 0.5, 1e-6));
        assert!(approx(s.getlr(100), 0.0, 1e-6));
        // Past the period the rate stays at the minimum.
        assert!(approx(s.getlr(500), 0.0, 1e-6));
    }

    #[test]
    fn warmup_cosine_rises_then_decays() {
        let s = lrschedule::linearwarmupcosine(0.0, 1.0, 0.0, 10, 100).unwrap();
        assert!(approx(s.getlr(0), 0.0, 1e-6));
        assert!(approx(s.getlr(5), 0.5, 1e-6));
        assert!(approx(s.getlr(10), 1.0, 1e-6));
        // At half of the decay window the rate is the midpoint.
        assert!(approx(s.getlr(60), 0.5, 1e-6));
        assert!(approx(s.getlr(110), 0.0, 1e-6));
    }

    #[test]
    fn invalid_schedules_are_rejected() {
        assert!(lrschedule::constant(-0.1).is_err());
        assert!(lrschedule::constant(f32::NAN).is_err());
        assert!(lrschedule::steplr(1.0, 0, 0.5, 0.0).is_err());
        assert!(lrschedule::steplr(1.0, 10, 2.0, 0.0).is_err());
        assert!(lrschedule::exponential(1.0, -0.5, 0.0).is_err());
        assert!(lrschedule::cosineannealing(1.0, 0.0, 0).is_err());
        assert!(lrschedule::linearwarmupcosine(0.0, 1.0, 0.0, 11, 10).is_err());
        assert!(lrschedule::linearwarmupcosine(0.0, 1.0, 0.0, 0, 0).is_err());
    }
}

/// Tests relocated from `crates/optim/src/sgds.rs`.
mod sgds {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::optim::clipconfig;
    use ::optim::optimizer;
    use ::optim::sgd;
    use ::optim::sgdconfig;
    use ::optim::{optimizerstate, statetensor};
    use ::tensor::{shape, tensor};

    fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(dims).unwrap(), data).unwrap()
    }

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn plain_sgd_moves_against_the_gradient() {
        let cfg = sgdconfig::new(0.1, 0.0, 0.0).unwrap();
        let mut opt = sgd::new(cfg);
        let mut params = [build(vec![1], vec![1.0])];
        let mut grads = [build(vec![1], vec![2.0])];
        opt.update(&mut params, &mut grads).unwrap();
        assert!(approx(params[0].getdata()[0], 0.8, 1e-6));
        assert_eq!(opt.getstep(), 1);
        assert_eq!(opt.getname(), "sgd");
    }

    #[test]
    fn momentum_accumulates_the_velocity() {
        let cfg = sgdconfig::new(0.1, 0.9, 0.0).unwrap();
        let mut opt = sgd::new(cfg);
        let mut params = [build(vec![1], vec![0.0])];
        let mut grads = [build(vec![1], vec![1.0])];
        opt.update(&mut params, &mut grads).unwrap();
        assert!(approx(params[0].getdata()[0], -0.1, 1e-6));
        opt.update(&mut params, &mut grads).unwrap();
        assert!(approx(params[0].getdata()[0], -0.29, 1e-6));
        opt.update(&mut params, &mut grads).unwrap();
        assert!(approx(params[0].getdata()[0], -0.561, 1e-6));
    }

    #[test]
    fn weight_decay_is_added_to_the_gradient() {
        let cfg = sgdconfig::new(0.1, 0.0, 0.1).unwrap();
        let mut opt = sgd::new(cfg);
        let mut params = [build(vec![1], vec![1.0])];
        let mut grads = [build(vec![1], vec![0.0])];
        opt.update(&mut params, &mut grads).unwrap();
        assert!(approx(params[0].getdata()[0], 0.99, 1e-6));
    }

    #[test]
    fn update_clips_gradients_when_configured() {
        let cfg = sgdconfig::new(0.1, 0.0, 0.0)
            .unwrap()
            .withclip(clipconfig::byvalue(-0.5, 0.5).unwrap());
        let mut opt = sgd::new(cfg);
        let mut params = [build(vec![2], vec![0.0, 0.0])];
        let mut grads = [build(vec![2], vec![2.0, -2.0])];
        opt.update(&mut params, &mut grads).unwrap();
        assert_eq!(grads[0].getdata(), &[0.5, -0.5]);
        assert!(approx(params[0].getdata()[0], -0.05, 1e-6));
        assert!(approx(params[0].getdata()[1], 0.05, 1e-6));
    }

    #[test]
    fn state_roundtrip_restores_identical_updates() {
        let cfg = sgdconfig::new(0.1, 0.9, 0.0).unwrap();
        let mut original = sgd::new(cfg);
        let mut params = [build(vec![2], vec![1.0, -1.0])];
        let mut grads = [build(vec![2], vec![0.5, 0.25])];
        original.update(&mut params, &mut grads).unwrap();
        let snapshot = original.getstate();
        assert_eq!(snapshot.getparamcount(), 1);
        assert_eq!(snapshot.gettensors().len(), 1);
        assert_eq!(snapshot.gettensors()[0].getname(), "momentum.0");

        let mut restored = sgd::new(cfg);
        restored.restore(&snapshot).unwrap();

        // Continue both optimizers from the same post-step state.
        let mut a_params = [params[0].clone()];
        let mut a_grads = [grads[0].clone()];
        let mut b_params = [params[0].clone()];
        let mut b_grads = [grads[0].clone()];
        original.update(&mut a_params, &mut a_grads).unwrap();
        restored.update(&mut b_params, &mut b_grads).unwrap();
        assert_eq!(b_params[0].getdata(), a_params[0].getdata());
    }

    #[test]
    fn restore_rejects_a_count_mismatch() {
        let mut opt = sgd::new(sgdconfig::new(0.1, 0.0, 0.0).unwrap());
        let a = statetensor::new("momentum.0", shape::new(vec![1]).unwrap(), vec![0.0]).unwrap();
        let b = statetensor::new("momentum.1", shape::new(vec![1]).unwrap(), vec![0.0]).unwrap();
        let state = optimizerstate::new(1, 0.1, 1, vec![a, b]);
        assert!(opt.restore(&state).is_err());
    }

    #[test]
    fn invalid_configurations_are_rejected() {
        assert!(sgdconfig::new(-0.1, 0.0, 0.0).is_err());
        assert!(sgdconfig::new(0.1, 1.0, 0.0).is_err());
        assert!(sgdconfig::new(0.1, -0.1, 0.0).is_err());
        assert!(sgdconfig::new(0.1, 0.0, -1.0).is_err());
    }
}

/// Tests relocated from `crates/optim/src/state.rs`.
mod state {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use ::optim::opterror;
    use ::optim::optimizerstate;
    use ::optim::statetensor;
    use ::tensor::shape;

    fn s(dims: &[usize]) -> shape {
        shape::new(dims.to_vec()).unwrap()
    }

    #[test]
    fn new_accepts_a_matching_length() {
        let t = statetensor::new("m.0", s(&[2, 3]), vec![0.0; 6]).unwrap();
        assert_eq!(t.getname(), "m.0");
        assert_eq!(t.getshape().getdims(), &[2, 3]);
        assert_eq!(t.getdata(), &[0.0; 6]);
    }

    #[test]
    fn new_rejects_a_length_mismatch() {
        assert!(matches!(
            statetensor::new("m.0", s(&[2, 3]), vec![0.0; 5]),
            Err(opterror::invalidstate { .. })
        ));
    }

    #[test]
    fn byte_roundtrip_preserves_values() {
        let t = statetensor::new("v.0", s(&[3]), vec![-1.5, 0.0, f32::MAX]).unwrap();
        let bytes = t.tobytes();
        assert_eq!(bytes.len(), 12);
        let back = statetensor::frombytes("v.0", s(&[3]), &bytes).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn frombytes_rejects_a_short_buffer() {
        assert!(matches!(
            statetensor::frombytes("v.0", s(&[3]), &[0u8; 8]),
            Err(opterror::invalidstate { .. })
        ));
    }

    #[test]
    fn state_reports_its_scalars_and_tensors() {
        let t = statetensor::new("m.0", s(&[1]), vec![1.0]).unwrap();
        let st = optimizerstate::new(7, 0.25, 1, vec![t]);
        assert_eq!(st.getstep(), 7);
        assert_eq!(st.getlr(), 0.25);
        assert_eq!(st.getparamcount(), 1);
        assert_eq!(st.gettensors().len(), 1);
    }
}

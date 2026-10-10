//! Relocated unit tests for the `train` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/train/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/train/src/error.rs`.
mod error {
    use ::tensor::tensorerror;
    use ::train::trainerror;
    use optim::opterror;

    #[test]
    fn tensor_errors_convert_transparently() {
        let error: trainerror = tensorerror::emptyshape.into();
        assert_eq!(
            error.to_string(),
            "a shape must have at least one dimension"
        );
    }

    #[test]
    fn optimizer_errors_convert_transparently() {
        let error: trainerror = opterror::state("bad snapshot").into();
        assert_eq!(error.to_string(), "invalid optimizer state: bad snapshot");
    }

    #[test]
    fn state_helper_reports_its_detail() {
        assert_eq!(
            trainerror::state("step ahead of optimizer").to_string(),
            "invalid training state: step ahead of optimizer"
        );
    }

    #[test]
    fn cancellation_is_distinguishable() {
        assert!(matches!(trainerror::cancelled, trainerror::cancelled));
        assert_eq!(trainerror::cancelled.to_string(), "training was cancelled");
    }
}

/// Tests relocated from `crates/train/src/evaluation.rs`.
mod evaluation {
    use crate::support::build;
    use ::train::evalsample;
    use ::train::evaluate;
    use ::train::trainerror;

    #[test]
    fn evaluation_is_row_weighted() {
        let small = evalsample::new(build(vec![1, 2], vec![0.0, 0.0]), vec![0]);
        let large = evalsample::new(build(vec![3, 2], vec![0.0; 6]), vec![0, 1, 0]);
        let result = evaluate(&[small, large]).unwrap();
        assert_eq!(result.getrows(), 4);
        assert!((result.getloss() - (2.0_f32).ln()).abs() < 1e-6);
        assert!((result.getperplexity() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn empty_samples_are_rejected() {
        assert!(matches!(
            evaluate(&[]),
            Err(trainerror::invalidstate { .. })
        ));
    }
}

/// Tests relocated from `crates/train/src/gradientaccumulation.rs`.
mod gradientaccumulation {
    use crate::support::build;
    use ::train::gradientaccumulator;
    use ::train::trainerror;

    #[test]
    fn zero_capacity_is_rejected() {
        let params = [build(vec![2], vec![1.0, 2.0])];
        assert!(matches!(
            gradientaccumulator::new(&params, 0),
            Err(trainerror::invalidaccumulation { found: 0 })
        ));
    }

    #[test]
    fn empty_parameters_are_rejected() {
        assert!(matches!(
            gradientaccumulator::new(&[], 2),
            Err(trainerror::emptyparameters)
        ));
    }

    #[test]
    fn mean_of_accumulated_gradients_is_arithmetic() {
        let params = [build(vec![2], vec![0.0, 0.0])];
        let mut accumulator = gradientaccumulator::new(&params, 2).unwrap();
        accumulator
            .accumulate(&[build(vec![2], vec![1.0, 3.0])])
            .unwrap();
        assert!(!accumulator.isfull());
        accumulator
            .accumulate(&[build(vec![2], vec![3.0, 1.0])])
            .unwrap();
        assert!(accumulator.isfull());
        let mean = accumulator.take().unwrap();
        assert_eq!(mean[0].getdata(), &[2.0, 2.0]);
        assert!(accumulator.isempty());
    }

    #[test]
    fn take_after_reset_returns_zeroed_buffer() {
        let params = [build(vec![1], vec![0.0])];
        let mut accumulator = gradientaccumulator::new(&params, 1).unwrap();
        accumulator
            .accumulate(&[build(vec![1], vec![5.0])])
            .unwrap();
        let _ = accumulator.take().unwrap();
        accumulator
            .accumulate(&[build(vec![1], vec![5.0])])
            .unwrap();
        assert_eq!(accumulator.take().unwrap()[0].getdata(), &[5.0]);
    }

    #[test]
    fn overflowing_the_window_is_rejected() {
        let params = [build(vec![1], vec![0.0])];
        let mut accumulator = gradientaccumulator::new(&params, 1).unwrap();
        accumulator
            .accumulate(&[build(vec![1], vec![1.0])])
            .unwrap();
        assert!(matches!(
            accumulator.accumulate(&[build(vec![1], vec![1.0])]),
            Err(trainerror::accumulatorfull)
        ));
    }

    #[test]
    fn wrong_gradient_count_is_rejected() {
        let params = [build(vec![1], vec![0.0])];
        let mut accumulator = gradientaccumulator::new(&params, 2).unwrap();
        assert!(matches!(
            accumulator.accumulate(&[]),
            Err(trainerror::gradientcountmismatch {
                expected: 1,
                actual: 0
            })
        ));
        assert!(accumulator.isempty());
    }

    #[test]
    fn wrong_gradient_shape_is_rejected() {
        let params = [build(vec![2], vec![0.0, 0.0])];
        let mut accumulator = gradientaccumulator::new(&params, 2).unwrap();
        assert!(matches!(
            accumulator.accumulate(&[build(vec![3], vec![0.0, 0.0, 0.0])]),
            Err(trainerror::gradientshapemismatch { index: 0, .. })
        ));
    }

    #[test]
    fn take_on_an_empty_window_is_rejected() {
        let params = [build(vec![1], vec![0.0])];
        let mut accumulator = gradientaccumulator::new(&params, 2).unwrap();
        assert!(matches!(
            accumulator.take(),
            Err(trainerror::invalidstate { .. })
        ));
    }
}

/// Tests relocated from `crates/train/src/loss.rs`.
mod loss {
    use crate::support::build;
    use ::train::crossentropy;
    use ::train::trainerror;

    #[test]
    fn uniform_logits_produce_log_classes() {
        let logits = build(vec![2, 4], vec![0.0; 8]);
        let result = crossentropy(&logits, &[0, 3]).unwrap();
        assert!((result.getloss() - (4.0_f32).ln()).abs() < 1e-6);
        let grad = result.getgradient().getdata();
        for row in grad.as_chunks::<4>().0 {
            let sum: f32 = row.iter().sum();
            assert!(sum.abs() < 1e-6, "row gradient did not sum to zero: {sum}");
        }
    }

    #[test]
    fn confident_correct_prediction_has_lower_loss() {
        let confident = build(vec![1, 3], vec![8.0, 0.0, 0.0]);
        let unsure = build(vec![1, 3], vec![0.0, 0.0, 0.0]);
        let a = crossentropy(&confident, &[0]).unwrap();
        let b = crossentropy(&unsure, &[0]).unwrap();
        assert!(a.getloss() < b.getloss());
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let logits = build(vec![2, 3], vec![0.5, -1.0, 2.0, -0.25, 0.75, 0.1]);
        let targets = [2_usize, 0_usize];
        let analytic = crossentropy(&logits, &targets).unwrap();
        let epsilon = 1e-3_f32;
        for index in 0..logits.getdata().len() {
            let mut plus = logits.getdata().to_vec();
            let mut minus = logits.getdata().to_vec();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let lp = crossentropy(&build(vec![2, 3], plus), &targets)
                .unwrap()
                .getloss();
            let lm = crossentropy(&build(vec![2, 3], minus), &targets)
                .unwrap()
                .getloss();
            let numeric = (lp - lm) / (2.0 * epsilon);
            let analytic = analytic.getgradient().getdata()[index];
            assert!(
                (numeric - analytic).abs() < 1e-3,
                "index {index}: numeric {numeric} vs analytic {analytic}"
            );
        }
    }

    #[test]
    fn large_logits_do_not_overflow() {
        let logits = build(vec![1, 3], vec![1000.0, 0.0, -1000.0]);
        let result = crossentropy(&logits, &[0]).unwrap();
        assert!(result.getloss().is_finite());
        assert!((result.getloss() - 0.0).abs() < 1e-3);
    }

    #[test]
    fn non_matrix_logits_are_rejected() {
        let logits = build(vec![3], vec![0.0, 0.0, 0.0]);
        assert!(matches!(
            crossentropy(&logits, &[0]),
            Err(trainerror::lossnotmatrix { .. })
        ));
    }

    #[test]
    fn empty_matrix_is_rejected() {
        let logits = build(vec![0, 3], Vec::new());
        assert!(matches!(
            crossentropy(&logits, &[]),
            Err(trainerror::lossemptymatrix { .. })
        ));
    }

    #[test]
    fn target_count_mismatch_is_rejected() {
        let logits = build(vec![2, 3], vec![0.0; 6]);
        assert!(matches!(
            crossentropy(&logits, &[0]),
            Err(trainerror::targetcountmismatch {
                rows: 2,
                targets: 1
            })
        ));
    }

    #[test]
    fn out_of_range_target_is_rejected() {
        let logits = build(vec![1, 3], vec![0.0; 3]);
        assert!(matches!(
            crossentropy(&logits, &[3]),
            Err(trainerror::targetoutofrange {
                index: 0,
                label: 3,
                classes: 3
            })
        ));
    }
}

/// Tests relocated from `crates/train/src/step.rs`.
///
/// The former unit tests called the crate-private `applyschedule` and
/// `stepresult::new`. The schedule behavior is verified through `trainer::step`,
/// which reports the applied rate in its `stepresult`; the `stepresult` fields
/// are covered by the `trainers` block.
mod step {
    use crate::support::build;
    use ::optim::{lrschedule, sgd, sgdconfig};
    use ::tensor::tensor;
    use ::train::{trainconfig, trainer, trainerror};

    /// Loss `w^2` with gradient `2 * w` on a single-element parameter.
    fn quadratic(params: &[tensor<f32>]) -> Result<(f32, Vec<tensor<f32>>), trainerror> {
        let value = params[0].getdata()[0];
        Ok((value * value, vec![build(vec![1], vec![2.0 * value])]))
    }

    fn trainerwith(schedule: Option<lrschedule>, lr: f32) -> trainer<sgd> {
        let config = trainconfig::new(1, 0, schedule, lib::seed::new(0)).unwrap();
        trainer::new(
            vec![build(vec![1], vec![1.0])],
            sgd::new(sgdconfig::new(lr, 0.0, 0.0).unwrap()),
            config,
        )
        .unwrap()
    }

    #[test]
    fn a_schedule_sets_the_applied_learning_rate() {
        // steplr(initial=1.0, steplen=1, gamma=0.5) halves once per step index.
        let schedule = lrschedule::steplr(1.0, 1, 0.5, 0.0).unwrap();
        let mut trainer = trainerwith(Some(schedule), 0.1);
        let first = trainer.step(quadratic).unwrap();
        assert!((first.getlearningrate() - 1.0).abs() < 1e-6);
        let second = trainer.step(quadratic).unwrap();
        assert!((second.getlearningrate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn without_a_schedule_the_configured_rate_is_kept() {
        let mut trainer = trainerwith(None, 0.2);
        let result = trainer.step(quadratic).unwrap();
        assert!((result.getlearningrate() - 0.2).abs() < 1e-6);
    }
}

/// Tests relocated from `crates/train/src/trainconfigs.rs`.
mod trainconfigs {
    use ::train::trainconfig;
    use lib::seed;
    use optim::lrschedule;

    #[test]
    fn new_accepts_a_valid_configuration() {
        let schedule = lrschedule::constant(0.1).unwrap();
        let config = trainconfig::new(4, 100, Some(schedule), seed::new(7)).unwrap();
        assert_eq!(config.getaccumulationsteps(), 4);
        assert_eq!(config.getmaxsteps(), 100);
        assert_eq!(config.getschedule(), Some(schedule));
        assert_eq!(config.getseed(), seed::new(7));
    }

    #[test]
    fn zero_accumulation_is_rejected() {
        assert!(matches!(
            trainconfig::new(0, 10, None, seed::new(1)),
            Err(::train::trainerror::invalidaccumulation { found: 0 })
        ));
    }

    #[test]
    fn single_uses_one_microbatch_and_no_schedule() {
        let config = trainconfig::single(5, seed::new(3)).unwrap();
        assert_eq!(config.getaccumulationsteps(), 1);
        assert_eq!(config.getmaxsteps(), 5);
        assert_eq!(config.getschedule(), None);
        assert_eq!(config.getseed(), seed::new(3));
    }
}

/// Tests relocated from `crates/train/src/trainers.rs`.
mod trainers {
    use crate::support::build;
    use core::cell::Cell;

    use ::tensor::tensor;
    use context::cancellation;
    use optim::{optimizer, sgd, sgdconfig};

    use ::train::trainconfig;
    use ::train::trainer;
    use ::train::trainerror;

    fn sgdoptimizer(lr: f32) -> sgd {
        sgd::new(sgdconfig::new(lr, 0.0, 0.0).unwrap())
    }

    fn quadraticconfig(accumulation: usize, maxsteps: u64) -> trainconfig {
        trainconfig::new(accumulation, maxsteps, None, lib::seed::new(0)).unwrap()
    }

    /// Loss `sum(w^2)` on a single-element parameter with gradient `2 * w`.
    fn quadratic(params: &[tensor<f32>]) -> Result<(f32, Vec<tensor<f32>>), trainerror> {
        let value = params[0].getdata()[0];
        let loss = value * value;
        let grad = build(vec![1], vec![2.0 * value]);
        Ok((loss, vec![grad]))
    }

    #[test]
    fn empty_parameters_are_rejected() {
        assert!(matches!(
            trainer::new(Vec::new(), sgdoptimizer(0.1), quadraticconfig(1, 0)),
            Err(trainerror::emptyparameters)
        ));
    }

    #[test]
    fn a_step_decreases_a_quadratic_loss() {
        let params = vec![build(vec![1], vec![4.0])];
        let mut trainer = trainer::new(params, sgdoptimizer(0.1), quadraticconfig(1, 0)).unwrap();
        let mut previous = f32::INFINITY;
        for expected in 1..=20_u64 {
            let result = trainer.step(quadratic).unwrap();
            assert_eq!(result.getstep(), expected);
            assert_eq!(result.getmicrobatches(), 1);
            assert!(result.getloss() < previous);
            previous = result.getloss();
        }
        assert!(trainer.getparams()[0].getdata()[0] < 4.0);
    }

    #[test]
    fn accumulation_averages_before_a_single_update() {
        let params = vec![build(vec![1], vec![1.0])];
        let mut trainer = trainer::new(params, sgdoptimizer(0.1), quadraticconfig(3, 0)).unwrap();
        let calls = Cell::new(0_usize);
        let closure = |params: &[tensor<f32>]| {
            calls.set(calls.get() + 1);
            quadratic(params)
        };
        let result = trainer.step(closure).unwrap();
        assert_eq!(calls.get(), 3);
        assert_eq!(result.getmicrobatches(), 3);
        assert_eq!(result.getstep(), 1);
        // The mean gradient over three identical micro-batches is `2 * w`, and
        // one SGD update with rate 0.1 yields w = 1 - 0.2 = 0.8.
        assert!((trainer.getparams()[0].getdata()[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn step_requires_an_empty_window() {
        let params = vec![build(vec![1], vec![1.0])];
        let mut trainer = trainer::new(params, sgdoptimizer(0.1), quadraticconfig(2, 0)).unwrap();
        assert!(trainer.accumulatemicrobatch(quadratic).unwrap().is_none());
        assert!(matches!(
            trainer.step(quadratic),
            Err(trainerror::invalidstate { .. })
        ));
    }

    #[test]
    fn cancellation_does_not_advance_state() {
        let token = cancellation::new();
        token.cancel();
        let params = vec![build(vec![1], vec![4.0])];
        let mut trainer = trainer::new(params, sgdoptimizer(0.1), quadraticconfig(1, 0))
            .unwrap()
            .withcancellation(token);
        assert!(matches!(
            trainer.step(quadratic),
            Err(trainerror::cancelled)
        ));
        assert_eq!(trainer.getstep(), 0);
        assert_eq!(trainer.getoptimizer().getstep(), 0);
        assert!((trainer.getparams()[0].getdata()[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn a_non_finite_loss_does_not_advance_state() {
        let params = vec![build(vec![1], vec![1.0])];
        let mut trainer = trainer::new(params, sgdoptimizer(0.1), quadraticconfig(1, 0)).unwrap();
        let result = trainer.step(|params: &[tensor<f32>]| {
            Ok((f32::NAN, vec![build(vec![1], vec![params[0].getdata()[0]])]))
        });
        assert!(matches!(
            result,
            Err(trainerror::nonfiniteloss { step: 0, .. })
        ));
        assert_eq!(trainer.getstep(), 0);
    }

    #[test]
    fn the_step_budget_is_enforced() {
        let params = vec![build(vec![1], vec![1.0])];
        let mut trainer = trainer::new(params, sgdoptimizer(0.1), quadraticconfig(1, 2)).unwrap();
        assert!(trainer.step(quadratic).is_ok());
        assert!(trainer.step(quadratic).is_ok());
        assert!(!trainer.canstep());
        assert!(matches!(
            trainer.step(quadratic),
            Err(trainerror::steplimitreached {
                completed: 2,
                max: 2
            })
        ));
    }

    #[test]
    fn wrong_gradient_count_resets_the_window() {
        let params = vec![build(vec![1], vec![1.0])];
        let mut trainer = trainer::new(params, sgdoptimizer(0.1), quadraticconfig(2, 0)).unwrap();
        let result = trainer.step(|_params: &[tensor<f32>]| Ok((1.0, Vec::new())));
        assert!(matches!(
            result,
            Err(trainerror::gradientcountmismatch { .. })
        ));
        assert!(trainer.getaccumulator().isempty());
        assert_eq!(trainer.getstep(), 0);
    }
}

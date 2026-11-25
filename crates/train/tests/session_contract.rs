use super::{TrainConfig, TrainOverrides, TrainSpec};
use crate::{LoaderConfig, ObjectiveConfig, Sampling, Sgd};
use diagnostic::{Code, Result};

/// TrainSpec 只组合纯数据，并在装配边界重新校验公开字段。
#[test]
fn train_spec_validates_session_and_loader_together() -> Result<()> {
    let session = TrainConfig::new(0.2, 1, 3, Some(0.5), 2)?;
    let loader = LoaderConfig::new(2, 3)?;
    let spec = TrainSpec::new(session.clone(), loader.clone())?;
    assert_eq!(spec.into_parts(), (session, loader));

    let invalid = TrainConfig {
        learning_rate: f32::NAN,
        ..TrainConfig::default()
    };
    let error = TrainSpec::new(invalid, LoaderConfig::default())
        .expect_err("公开字段绕过构造器后仍要在装配边界拒绝");
    assert_eq!(error.code(), Code::Invalid);
    Ok(())
}

/// 可选覆盖：缺项取默认值、给出的字段生效；非法值仍走 [`TrainConfig::new`] 校验。
#[test]
fn overrides_fall_back_to_default_without_duplicating_constants() {
    let base = TrainConfig::default();

    let all_default = TrainConfig::with_overrides(TrainOverrides::default()).expect("缺省配置合法");
    assert_eq!(all_default, base, "全缺省应与默认配置逐字段一致");

    let overridden = TrainConfig::with_overrides(TrainOverrides {
        learning_rate: Some(2e-3),
        warmup_steps: Some(2),
        max_steps: Some(5),
        clip_norm: Some(0.5),
        accumulation: Some(2),
    })
    .expect("全部覆盖合法");
    assert_eq!(overridden.learning_rate, 2e-3);
    assert_eq!(overridden.warmup_steps, 2);
    assert_eq!(overridden.max_steps, 5);
    assert_eq!(overridden.clip_norm, Some(0.5));
    assert_eq!(overridden.accumulation, 2);

    let bad_steps = TrainConfig::with_overrides(TrainOverrides {
        max_steps: Some(0),
        ..TrainOverrides::default()
    });
    assert_eq!(bad_steps.expect_err("总步数不能为 0").code(), Code::Invalid);

    let bad_accum = TrainConfig::with_overrides(TrainOverrides {
        accumulation: Some(0),
        ..TrainOverrides::default()
    });
    assert_eq!(
        bad_accum.expect_err("累积窗口不能为 0").code(),
        Code::Invalid
    );
}

/// 配置指纹稳定且覆盖调度、目标、数据采样、优化器类型与静态超参。
#[test]
fn train_spec_fingerprint_is_stable_and_field_sensitive() {
    let base = TrainSpec::default();
    let sgd = Sgd::new();
    let fingerprint = base.config_fingerprint(&sgd);
    assert_eq!(base.config_fingerprint(&sgd), fingerprint);
    assert_ne!(base.config_fingerprint(&crate::Adam::new()), fingerprint);
    assert_ne!(
        base.config_fingerprint(&Sgd::new().with_weight_decay(0.1)),
        fingerprint,
        "优化器超参漂移也必须拒绝"
    );
    let mut changed = base.clone();
    changed.session.anneal = crate::Anneal::Cosine;
    assert_ne!(changed.config_fingerprint(&sgd), fingerprint);
    let mut changed = base.clone();
    changed.session.clip_norm = Some(1.0);
    assert_ne!(changed.config_fingerprint(&sgd), fingerprint);
    let mut changed = base.clone();
    changed.objective = ObjectiveConfig::Distill(crate::Distill::new(2.0, 0.5));
    assert_ne!(changed.config_fingerprint(&sgd), fingerprint);
    let mut changed = base.clone();
    changed.loader.sampling = Sampling::Shuffle { seed: 7 };
    assert_ne!(changed.config_fingerprint(&sgd), fingerprint);
    let mut changed = base;
    changed.loader.sampling = Sampling::Weighted {
        weights: vec![0.25, 0.75],
        seed: 11,
    };
    assert_ne!(changed.config_fingerprint(&sgd), fingerprint);
}

/// 会话装配必须重验优化器配置，非法裁剪值不能变成 None。
#[test]
fn session_rejects_invalid_optimizer_configuration() {
    let (tape, sizes, ..) = super::fixture();
    let error = crate::Session::new(
        tape,
        sizes,
        crate::Reference,
        Sgd::new().with_weight_decay(-1.0),
        TrainConfig::default(),
    )
    .err()
    .expect("非法优化器配置");
    assert_eq!(error.code(), Code::Invalid);
}

struct Reject;
impl crate::Optimizer for Reject {
    fn step(&mut self, _: &mut [crate::Slot<'_>], _: f32) -> Result<()> {
        Err(diagnostic::Error::new(Code::Invalid, "注入的优化器失败"))
    }
    fn set_clip_norm(&mut self, _: Option<f32>) {}
    fn restore(&mut self, _: Vec<Vec<f32>>, _: Vec<Vec<f32>>, _: u64) -> Result<()> {
        Ok(())
    }
}

/// 优化器失败后参数与梯度所有权仍归还，窗口与步号不提交。
#[test]
fn failed_optimizer_returns_parameter_and_gradient_views() -> Result<()> {
    let (tape, sizes, p, ..) = super::fixture();
    let config = TrainConfig::new(0.1, 0, 1, None, 2)?;
    let mut session = crate::Session::new(tape, sizes, crate::Reference, Reject, config)?;
    session.register_parameter(p, nn::ParamRole::Trainable, vec![1.0; super::N])?;
    session.load_gradient(p, &[0.5; super::N])?;
    session.optimize()?;
    assert_eq!(
        session.optimize().expect_err("注入失败").code(),
        Code::Invalid
    );
    assert_eq!(session.slots()[0].params(), &[1.0; super::N]);
    assert_eq!(session.bufs().gradient(p), Some(&[0.5; super::N][..]));
    assert_eq!((session.step(), session.accumulated_microbatches()), (0, 1));
    Ok(())
}

/// 晚期 ties 错误不能提前提交矩状态、梯度、窗口、绑定或步号。
#[test]
fn restore_validation_failure_leaves_every_state_component_unchanged() -> Result<()> {
    use crate::Optimizer;
    let (tape, sizes, source, tied, other, ..) = super::two_layer_fixture();
    let config = TrainConfig::new(0.1, 0, 4, None, 2)?;
    let mut target =
        crate::Session::new(tape, sizes, crate::Reference, crate::Adam::new(), config)?;
    super::register_two_layer(&mut target, source, tied, other)?;
    target.optimizer.restore(
        vec![vec![0.25; super::N]; 3],
        vec![vec![0.5; super::N]; 3],
        7,
    )?;
    target.optimize()?;
    target.load_gradient(source, &[9.0; super::N])?;
    let slot = storage::SlotMoments::new(vec![0.75; super::N], vec![0.875; super::N])?;
    let snapshot = weights::Snapshot::new_with_all(
        storage::Facts::new(5, 1, 2, 1, super::N)?,
        weights::Header::new(Vec::new(), 0)?,
        storage::Moments::new(vec![slot; 3], 9),
        storage::GradAccum::new(vec![vec![1.0; super::N]; 3], 1),
        storage::Ties::new(vec![(0, 5)]),
    );
    assert_eq!(
        target
            .restore_snapshot(&snapshot)
            .expect_err("ties 越界")
            .code(),
        Code::Invalid
    );
    assert_eq!(
        target.optimizer.momentum().to_vec(),
        vec![vec![0.25; super::N]; 3]
    );
    assert_eq!(
        target.optimizer.variance().to_vec(),
        vec![vec![0.5; super::N]; 3]
    );
    assert_eq!(target.optimizer.step_count(), 7);
    assert_eq!(target.bufs().gradient(source), Some(&[9.0; super::N][..]));
    assert_eq!(target.accumulated_microbatches(), 1);
    assert_eq!(target.ties(), &[(0, 1)]);
    assert_eq!(target.step(), 0);
    Ok(())
}

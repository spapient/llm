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
    assert_eq!(changed.config_fingerprint(&sgd), fingerprint);
}

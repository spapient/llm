use super::*;

/// 收集一个数据集全部批的 token 序列（每批压成一段）。
fn drain_tokens<'a>(dataset: &mut dyn Dataset<'a, Item = u32>) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    while let Some(batch) = dataset.next_batch() {
        out.push(batch.tokens().to_vec());
    }
    out
}

#[test]
fn same_shuffle_seed_replays_the_same_order() {
    let tokens: Vec<u32> = (0..8).collect();
    let config = LoaderConfig::new(1, 1).expect("几何合法");
    let seeded = LoaderConfig {
        sampling: Sampling::Shuffle { seed: 7 },
        ..config
    };

    let first = drain_tokens(&mut Loader::with_config(&tokens, seeded.clone()).expect("装配"));
    let second = drain_tokens(&mut Loader::with_config(&tokens, seeded).expect("装配"));

    assert_eq!(first, second, "同种子必须重放出同一条批序");
}

#[test]
fn different_shuffle_seeds_yield_different_orders() {
    let tokens: Vec<u32> = (0..8).collect();
    let make = |seed| {
        let config = LoaderConfig::new(1, 1).expect("几何合法");
        let seeded = LoaderConfig {
            sampling: Sampling::Shuffle { seed },
            ..config
        };
        drain_tokens(&mut Loader::with_config(&tokens, seeded).expect("装配"))
    };

    let a = make(0);
    let b = make(1);
    assert_ne!(a, b, "不同种子的批序应当不同");
}

#[test]
fn shuffle_keeps_every_batch_exactly_once() {
    let tokens: Vec<u32> = (0..8).collect();
    let config = LoaderConfig::new(1, 1).expect("几何合法");
    let seeded = LoaderConfig {
        sampling: Sampling::Shuffle { seed: 3 },
        ..config
    };

    let collected = drain_tokens(&mut Loader::with_config(&tokens, seeded).expect("装配"));
    let mut flat: Vec<u32> = collected.iter().flatten().copied().collect();
    flat.sort_unstable();

    assert_eq!(flat, tokens, "洗牌不得丢批也不得重复批");
    assert_eq!(collected.len(), 8, "批数与顺序流一致");
}

#[test]
fn reset_returns_to_the_first_batch_of_the_round() {
    let tokens: Vec<u32> = (0..6).collect();
    let mut loader = Loader::new(&tokens, 1, 1).expect("顺序流");

    let first_round = drain_tokens(&mut loader);
    loader.reset();
    let again = drain_tokens(&mut loader);

    assert_eq!(first_round, again, "顺序流重置后必须重放同一条序");
}

#[test]
fn two_rounds_yield_twice_the_declared_batches() {
    let tokens: Vec<u32> = (0..6).collect();
    let mut loader = Loader::new(&tokens, 1, 1).expect("顺序流");
    let batches = loader.batches();

    let mut count = 0;
    for _round in 0..2 {
        loader.reset();
        while loader.next_batch().is_some() {
            count += 1;
        }
    }

    assert_eq!(count, 2 * batches, "每轮重置后批数回到声明的总数");
}

#[test]
fn weighted_rejects_illegal_weights_at_construction() {
    let tokens: Vec<u32> = (0..4).collect();
    let config = LoaderConfig::new(1, 1).expect("几何合法");

    let wrong_len = LoaderConfig {
        sampling: Sampling::Weighted {
            weights: vec![1.0, 2.0],
            seed: 0,
        },
        ..config
    };
    assert_eq!(
        Loader::with_config(&tokens, wrong_len).unwrap_err().code(),
        Code::Invalid,
        "权重条数与批数不符必须在构造期拒绝"
    );

    let negative = LoaderConfig {
        sampling: Sampling::Weighted {
            weights: vec![1.0, -1.0, 1.0, 1.0],
            seed: 0,
        },
        ..config
    };
    assert_eq!(
        Loader::with_config(&tokens, negative).unwrap_err().code(),
        Code::Invalid,
        "负权重必须拒绝"
    );

    let nan = LoaderConfig {
        sampling: Sampling::Weighted {
            weights: vec![f32::NAN; 4],
            seed: 0,
        },
        ..config
    };
    assert_eq!(
        Loader::with_config(&tokens, nan).unwrap_err().code(),
        Code::Invalid,
        "NaN 权重必须拒绝"
    );

    let zero = LoaderConfig {
        sampling: Sampling::Weighted {
            weights: vec![0.0; 4],
            seed: 0,
        },
        ..config
    };
    assert_eq!(
        Loader::with_config(&tokens, zero).unwrap_err().code(),
        Code::Invalid,
        "权重总质量为正才允许抽样"
    );
}

#[test]
fn weighted_draws_one_batch_per_declared_batch() {
    let tokens: Vec<u32> = (0..4).collect();
    let config = LoaderConfig::new(1, 1).expect("几何合法");
    let weighted = LoaderConfig {
        sampling: Sampling::Weighted {
            weights: vec![1.0, 1.0, 1.0, 1.0],
            seed: 0,
        },
        ..config
    };
    let mut loader = Loader::with_config(&tokens, weighted).expect("装配");

    let drawn: Vec<_> = (0..loader.batches())
        .filter_map(|_| loader.next_batch())
        .collect();
    assert_eq!(drawn.len(), 4, "每轮抽出的批数等于声明批数");
}

#[test]
fn high_weight_batch_is_sampled_more_often() {
    let tokens: Vec<u32> = (0..2).collect();
    let config = LoaderConfig::new(1, 1).expect("几何合法");
    let weighted = LoaderConfig {
        sampling: Sampling::Weighted {
            weights: vec![1.0, 100.0],
            seed: 0,
        },
        ..config
    };
    let mut loader = Loader::with_config(&tokens, weighted).expect("装配");

    let mut counts = [0usize, 0];
    for _round in 0..20 {
        loader.reset();
        while let Some(batch) = loader.next_batch() {
            counts[batch.tokens()[0] as usize] += 1;
        }
    }

    assert!(
        counts[1] > counts[0],
        "高权重批（1）应比低权重批（0）更常被抽到：{counts:?}"
    );
}

#[test]
fn config_rejects_bad_geometry_before_assembly() {
    assert_eq!(LoaderConfig::new(0, 2).unwrap_err().code(), Code::Invalid);
    assert_eq!(LoaderConfig::new(2, 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(
        LoaderConfig::new(usize::MAX, 2).unwrap_err().code(),
        Code::Invalid
    );
}

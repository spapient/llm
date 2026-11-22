//! 接线台账的自证：族名与树文件一一对上、阻塞必须具名、具名的算子必须真的不存在，
//! 已接通的族必须真的落下层。

use std::collections::BTreeSet;
use std::path::Path;

use crate::wire::{self, Blocked, Wiring};
use crate::{Hybrid, Mhc, Mla, Moe};
use nn::Contract;

/// 递归收集 `src/` 下的模块名（文件名去掉 `.rs`）。
fn walk(dir: &Path, out: &mut BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            out.insert(stem.to_owned());
        }
    }
}

#[test]
fn every_family_in_the_ledger_still_exists_in_the_tree_and_is_named_once() {
    let mut files = BTreeSet::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    let ledger = wire::ledger();
    let names: BTreeSet<&str> = ledger.iter().map(|family| family.name).collect();
    assert_eq!(names.len(), ledger.len(), "台账里族名不能重复");
    for family in &ledger {
        assert!(
            files.contains(family.name),
            "台账写了 `{}`，但 `src/` 里没有同名文件——台账过期了",
            family.name
        );
    }
    assert_eq!(ledger.len(), 17, "17 族的数量是这份台账的意义所在");
}

#[test]
fn no_family_is_blocked_without_saying_what_is_missing() {
    for family in wire::ledger() {
        let name = family.name;
        let wiring = family.wiring;
        if wiring.is_wired() {
            assert!(wiring.layers() > 0, "`{name}` 说接通了却零个层");
            continue;
        }
        let reason = match wiring.blocked_on() {
            Some(Blocked::Operator(op)) => op,
            Some(Blocked::Layer(text) | Blocked::Geometry(text)) => text,
            None => panic!("`{name}` 既没接通又没写缺什么"),
        };
        assert!(!reason.is_empty(), "`{name}` 的阻塞理由是空串");
    }
}

/// 反过期：族声称缺某个算子时，`op` 注册表里必须真的没有它。
///
/// 这条守卫直接吃过自己的亏——`latent_attention_f32` 落地后 `mla` 仍写着"缺算子"，
/// 只有手工翻台账才发现。现在把这件事交给注册表判定。
#[test]
fn no_family_claims_a_missing_operator_that_is_already_registered() {
    for family in wire::ledger() {
        let Some(Blocked::Operator(op)) = family.wiring.blocked_on() else {
            continue;
        };
        assert!(
            prof::catalogue::named(op).is_none(),
            "`{}` 声称缺算子 `{op}`，但该算子已注册：台账该翻成接通或换阻塞理由",
            family.name
        );
    }
}

/// 夹具：`n_h = 2`、`d_h = 4`、`d_c = 8`、`d_h^R = 2`，所以主干宽 8。
fn fixtures() -> (Mla<f32>, Moe<f32>, Mhc<f32>) {
    let (heads, head_dim, latent, rope) = (2, 4, 8, 2);
    let up = vec![0.5f32; heads * head_dim * latent];
    let mla = Mla::new(heads, head_dim, latent, rope, &up, &up).expect("MLA 超参合法");
    let width = heads * head_dim;
    let moe = Moe::new(
        width,
        1,
        vec![0.5; 2 * width],
        vec![vec![0.1; width * width]; 2],
        None,
    )
    .expect("专家几何合法");
    let mhc = Mhc::identity(2).expect("流数非零");
    (mla, moe, mhc)
}

#[test]
fn the_wired_families_produce_layers_whose_operators_are_registered() {
    let (mla, moe, mhc) = fixtures();
    let hybrid = Hybrid::new(mla.clone(), moe, mhc).expect("混合块几何一致");
    let attention = [hybrid.clone(), hybrid.clone()];

    // 台账层数按单块计，所以这里带上块数换算。
    let cases = [
        (
            1,
            "mla",
            vec![
                Box::new(wire::mla_layer(&mla, 4, "mla.attn").expect("MLA 落层"))
                    as Box<dyn Contract>,
            ],
        ),
        (
            1,
            "hybrid",
            wire::hybrid_layer(&hybrid, 4, 0).expect("混合块落层"),
        ),
        (
            attention.len(),
            "tower",
            wire::tower_layers(&attention, 4).expect("塔落层"),
        ),
    ];
    let ledger = wire::ledger();
    for (blocks, name, layers) in cases {
        let family = ledger
            .iter()
            .find(|family| family.name == name)
            .expect("台账有这一族");
        assert_eq!(
            family.wiring.layers() * blocks,
            layers.len(),
            "`{name}` 台账写的层数与实际不符"
        );
        let unique: BTreeSet<&str> = layers.iter().map(|layer| layer.name()).collect();
        assert_eq!(unique.len(), layers.len(), "`{name}` 落层出现重名");
        for layer in &layers {
            for op in layer.spca().ops() {
                assert!(
                    prof::catalogue::named(op).is_some(),
                    "`{name}` 的层引用了未注册算子 `{op}`"
                );
            }
        }
    }
    assert_eq!(
        ledger
            .iter()
            .filter(|family| family.wiring.is_wired())
            .count(),
        2,
        "当前只有 `mla` 与 `mhc` 整族接通"
    );
    assert!(Mhc::<f32>::identity(0).is_err(), "流数为零必须被拒");
}

#[test]
fn mhc_lands_on_a_stream_square_matrix() {
    let mhc = Mhc::<f32>::identity(4).expect("流数非零");
    let layer = wire::mhc_layer(&mhc, "mhc.mix").expect("流重组落层");
    assert_eq!((layer.rows(), layer.cols()), (4, 4));
}

/// `Wiring` 的三种构造各自守住自己的不变量。
#[test]
fn wiring_constructors_keep_their_shape() {
    assert_eq!(Wiring::wired(3).layers(), 3);
    assert!(Wiring::wired(3).is_wired());
    assert_eq!(Wiring::blocked(Blocked::Operator("x")).layers(), 0);
    let partial = Wiring::partial(2, Blocked::Operator("x"));
    assert_eq!(partial.layers(), 2);
    assert!(!partial.is_wired());
}

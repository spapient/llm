use super::*;
fn f16_bf16() -> DTypeSet {
    DTypeSet::of([DType::F16, DType::BF16])
}
#[test]
fn closed_sets_round_trip() {
    assert_eq!(Prim::ALL.len(), 16);
    for prim in Prim::ALL.iter().copied() {
        assert_eq!(Prim::parse(prim.as_str()), Some(prim));
    }
    for dt in DType::ALL.iter().copied() {
        assert_eq!(DType::parse_name(dt.name()), Some(dt));
    }
    for space in Space::ALL.iter().copied() {
        assert_eq!(Space::parse(space.as_str()), Some(space));
    }
    for key in AttrKey::ALL.iter().copied() {
        assert_eq!(AttrKey::parse(key.as_str()), Some(key));
    }
    assert_eq!(Prim::parse("tma_bulk"), None);
}
#[test]
fn index_is_the_position_in_the_closed_set() {
    // 位集下标就是 `ALL` 里的位置：这条不变量让“加一个原语只改一个 impl”成立，
    // 也让 `Set<E, 1>` 的容量自证有意义（超过 64 个元素要显式加宽）。
    fn positions<T: Closed>(label: &str) {
        for (position, item) in T::ALL.iter().copied().enumerate() {
            assert_eq!(item.index(), position, "{label}: {}", item.as_str());
        }
    }
    positions::<Prim>("prim");
    positions::<DType>("dtype");
    positions::<Space>("space");
    positions::<AttrKey>("attr");
    positions::<Grade>("grade");
}
#[test]
fn a_set_holds_every_element_of_its_closed_set() {
    let all = SpaceSet::of(Space::ALL.iter().copied());
    assert_eq!(all.len(), 3);
    assert!(all.iter().eq(Space::ALL.iter().copied()));
    assert_eq!(all.names(), vec!["GM", "FAST", "REG"]);
    assert_eq!(all.to_string(), "GM,FAST,REG");
    assert!(SpaceSet::empty().is_subset(all));
    assert!(!all.is_subset(SpaceSet::empty()));
    assert_eq!(all.difference(all).len(), 0);
    assert_eq!(SpaceSet::of([Space::Fast]).difference(all).len(), 0);
    assert_eq!(
        all.difference(SpaceSet::of([Space::Fast])).to_string(),
        "GM,REG"
    );
    // 宽度自证：`Set<E, 1>` 装得下所有闭集，因此常量构造在编译期就通过检查。
    const EMPTY: SpaceSet = SpaceSet::empty();
    assert!(EMPTY.is_empty());
}
#[test]
fn span_parses_four_forms() {
    assert_eq!(Span::parse("16").unwrap(), Span::exact(16));
    assert_eq!(Span::parse("16..64").unwrap(), Span::range(16, 64).unwrap());
    assert_eq!(Span::parse("<=512").unwrap(), Span::at_most(512));
    assert_eq!(Span::parse("32..").unwrap(), Span::at_least(32));
    assert!(Span::parse("64..16").is_err());
    assert!(Span::parse("abc").is_err());
}
#[test]
fn covers_requires_range_containment() {
    let have = Span::range(16, 256).unwrap();
    assert!(have.covers(Span::range(16, 64).unwrap()));
    assert!(have.covers(Span::exact(256)));
    assert!(
        !have.covers(Span::range(8, 16).unwrap()),
        "下界越界不算覆盖"
    );
    assert!(!have.covers(Span::any()), "不限的需求只能由不限的供给覆盖");
}
#[test]
fn diff_reports_every_missing_field() {
    let need = Need::of(Prim::Mma)
        .with_extent(Span::range(16, 64).unwrap())
        .with_n(Span::range(8, 128).unwrap())
        .with_k(Span::exact(16))
        .with_dt(f16_bf16())
        .with_space(SpaceSet::of([Space::Fast]))
        .with_attr(Attr::new(AttrKey::Op, ["sum"]));
    let have = Provide::of(Prim::Mma, "cube")
        .with_extent(Span::exact(16))
        .with_n(Span::exact(16))
        .with_k(Span::exact(16))
        .with_dt(DTypeSet::of([DType::F16]))
        .with_space(SpaceSet::of([Space::Fast]))
        .with_grade(Grade::L2);
    assert!(!have.covers(&need));
    let diff = have.diff(&need);
    let text = diff.iter().map(ToString::to_string).collect::<Vec<_>>();
    // 两条都必须是真的陈述：`Span` 两个方向都可能不覆盖，只打印上界会写出
    // “16 < 8” 这种数学上为假的一句话。
    assert!(
        text.contains(&"m: 需求 16..64 不被 16 覆盖".to_string()),
        "{text:?}"
    );
    assert!(
        text.contains(&"n: 需求 8..128 不被 16 覆盖".to_string()),
        "{text:?}"
    );
    assert!(text.contains(&"dt 缺 bf16".to_string()));
    assert!(text.contains(&"op 缺 sum".to_string()));
    assert!(!text.iter().any(|t| t.starts_with("space")));
}
#[test]
fn exact_supply_covers_identical_need() {
    let need = Need::of(Prim::Load)
        .with_extent(Span::at_most(512))
        .with_space(SpaceSet::of([Space::Gm, Space::Fast]));
    let have = Provide::of(Prim::Load, "neon")
        .with_extent(Span::at_most(1024))
        .with_space(SpaceSet::of([Space::Gm, Space::Fast]));
    assert!(have.covers(&need));
    let too_wide = Provide::of(Prim::Load, "neon")
        .with_extent(Span::at_most(256))
        .with_space(SpaceSet::of([Space::Gm, Space::Fast]));
    assert_eq!(
        too_wide.diff(&need),
        vec![Mismatch::ExtentTooSmall {
            label: "bytes",
            need: Span::at_most(512),
            have: Span::at_most(256),
        }]
    );
}
#[test]
fn an_alignment_demand_is_matched_by_a_supply_of_the_same_or_wider_align() {
    let need = Need::of(Prim::Load).with_attr(Attr::new(AttrKey::Align, ["16"]));
    let scalar = Provide::of(Prim::Load, "memcpy").with_attr(Attr::new(AttrKey::Align, ["1"]));
    assert!(!scalar.covers(&need));
    assert_eq!(
        scalar.diff(&need),
        vec![Mismatch::AttrMissing {
            key: AttrKey::Align,
            missing: vec!["16".to_string()],
        }]
    );
    let vector = Provide::of(Prim::Load, "neon").with_attr(Attr::new(AttrKey::Align, ["1", "16"]));
    assert!(vector.covers(&need));
}
#[test]
fn missing_prim_is_reported_first() {
    let need = Need::of(Prim::LoadAsync);
    let have = Provide::of(Prim::Load, "memcpy");
    assert_eq!(
        have.diff(&need),
        vec![Mismatch::PrimAbsent {
            prim: Prim::LoadAsync
        }]
    );
}
#[test]
fn signatures_render_like_design_diagnostic() {
    let need = Need::of(Prim::Mma)
        .with_extent(Span::range(16, 64).unwrap())
        .with_n(Span::range(8, 128).unwrap())
        .with_k(Span::exact(16))
        .with_dt(f16_bf16())
        .with_space(SpaceSet::of([Space::Fast]));
    assert_eq!(
        need.signature(),
        "mma{m:16..64, n:8..128, k:16, dt:[f16,bf16], space:[FAST]}"
    );
    let have = Provide::of(Prim::Load, "neon")
        .with_extent(Span::at_most(1024))
        .with_grade(Grade::L2);
    assert_eq!(have.summary(), "load{bytes:<=1024, impl:neon, grade:L2}");
}
#[test]
fn grade_gate_is_l2() {
    assert!(!Grade::L0.is_verified());
    assert!(!Grade::L1.is_verified());
    assert!(Grade::L2.is_verified());
    assert!(Grade::L3.is_verified());
    assert!(Grade::L3 > Grade::L2);
}

/// 等级闭集常驻 L0–L3：全量、名字往返与下标顺序冻结；闭集外等级解析为 `None`，不许旁路自造等级。
#[test]
fn grade_closed_set_runs_l0_to_l3() {
    assert_eq!(Grade::ALL.len(), 4);
    let names: Vec<&str> = Grade::ALL
        .iter()
        .map(|&grade| Grade::as_str(grade))
        .collect();
    assert_eq!(names, ["L0", "L1", "L2", "L3"]);
    for grade in Grade::ALL.iter().copied() {
        assert_eq!(Grade::parse(grade.as_str()), Some(grade));
        assert_eq!(
            grade.index(),
            Grade::ALL.iter().position(|&item| item == grade).unwrap()
        );
    }
    assert_eq!(Grade::parse("L4"), None);
    assert_eq!(Grade::parse("verified"), None);
}
#[test]
fn dtype_set_ops() {
    let a = DTypeSet::of([DType::F16]);
    let b = f16_bf16();
    assert!(a.is_subset(b));
    assert!(!b.is_subset(a));
    assert_eq!(b.difference(a).names(), vec!["bf16"]);
    assert_eq!(DType::F4.bits(), 4);
}

#[test]
fn kernel_key_packs_the_seven_dimensions() {
    let key = KernelKey::new(
        Prim::Dot,
        "scalar_f32",
        DType::F32,
        Space::Fast,
        4,
        "cpu",
        16,
    );
    assert_eq!(key.prim, Prim::Dot);
    assert_eq!(key.impl_name, "scalar_f32");
    assert_eq!(key.dtype, DType::F32);
    assert_eq!(key.space, Space::Fast);
    assert_eq!(key.lanes, 4);
    assert_eq!(key.arch, "cpu");
    assert_eq!(key.block, 16);
}

#[test]
fn kernel_key_normalizes_zero_lanes_and_block_to_scalar() {
    let key = KernelKey::new(Prim::Mma, "mma", DType::F16, Space::Gm, 0, "cuda", 0);
    assert_eq!(key.lanes, 1);
    assert_eq!(key.block, 1);
}

#[test]
fn kernel_key_signature_keeps_the_adrd_order() {
    let key = KernelKey::new(
        Prim::Dot,
        "scalar_f32",
        DType::F32,
        Space::Fast,
        4,
        "cpu",
        16,
    );
    assert_eq!(
        key.signature(),
        "dot{dt:f32, space:FAST, lanes:4, arch:cpu, block:16}/impl:scalar_f32"
    );
}

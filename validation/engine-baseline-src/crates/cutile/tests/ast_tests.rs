use super::*;
use crate::builtin::Catalogue;
use crate::kernel::{Arith, Expr, Kernel, Write};
use crate::substitute;

/// 双射门：全部内置算子 × 每个 tile × 每台设备，AST 路径与字符串路径逐字节相等。
#[test]
fn ast_renders_byte_identical_text() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    for profile in catalogue.profiles() {
        let fragments =
            prof::builtin(profile.meta.vendor.as_str(), profile.meta.emit.ext()).expect("片段层");
        for op in catalogue.ops() {
            for combo in crate::decl::expand(op).expect("tile 候选") {
                let direct = substitute::render(op, &combo, &fragments)
                    .unwrap_or_else(|err| panic!("字符串路径失败：{err}"));
                let nodes = build(op, &combo, &fragments).expect("AST 构建");
                let via_ast = render(&nodes);
                assert_eq!(
                    direct, via_ast,
                    "`{}`/`{}`/`{combo}` 双射破裂",
                    profile.meta.vendor, op.name
                );
            }
        }
    }
}

/// 规范编码确定且对字段敏感：改任一节点都换编码。
#[test]
fn encode_is_stable_and_field_sensitive() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let op = catalogue.op("gemv_i4").expect("算子");
    let combo = crate::decl::expand(op).expect("tile").remove(0);
    let fragments = prof::builtin("cpu", "c").expect("片段");
    let nodes = build(op, &combo, &fragments).expect("AST");

    assert_eq!(encode(&nodes), encode(&nodes), "同一序列编码稳定");

    let mut changed = nodes.clone();
    if let Some(Node::Slot(value)) = changed.iter_mut().find(|n| matches!(n, Node::Slot(_))) {
        value.push('7');
    }
    assert_ne!(encode(&nodes), encode(&changed), "改槽位必须换编码");

    let mut changed = nodes.clone();
    if let Some(Node::Prim(call)) = changed.iter_mut().find(|n| matches!(n, Node::Prim(_))) {
        call.body.push(' ');
    }
    assert_ne!(encode(&nodes), encode(&changed), "改片段体必须换编码");
}

/// 节点往返：owned 参数与借用参数视图一致，splice 结果不依赖走哪条路径。
#[test]
fn owned_args_splice_like_borrowed_args() {
    let borrowed = [
        Arg::Expr("a"),
        Arg::Named {
            key: "along",
            value: "32",
        },
    ];
    let owned: Vec<CallArg> = borrowed.iter().map(CallArg::from_ref).collect();
    let view: Vec<Arg<'_>> = owned.iter().map(CallArg::as_ref).collect();
    assert_eq!(&borrowed[..], &view[..]);
}

/// 确定性 fuzz：对真实片段 body 随机插入空白 / 换行 / 分号，重建片段层后两条路径零差异。
///
/// 不依赖外部随机源：固定种子的 LCG 在任何机器上派生同一批扰动，缩进对齐与分号裁剪的边界
/// 因此被反复对拍。只插入空白与分号、不碰 `$` 占位符，保证 splice 语义不变。
#[test]
fn fuzzed_fragments_keep_the_two_paths_identical() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let op = catalogue.op("gemv_i4").expect("算子");
    let combos = crate::decl::expand(op).expect("tile 候选");
    let base = prof::builtin("cpu", "c").expect("真实片段");

    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..300 {
        let mut set = FragmentSet::new("cpu", "c");
        for &prim in Prim::ALL {
            let Some(fragment) = base.fragment(prim) else {
                continue;
            };
            let body = perturb(fragment.body(), &mut state);
            let source = format!("/* {} {prim}/z */\n{body}", prof::SIGN_MARKER);
            set.insert(prim, source).expect("合法签名片段");
        }
        for combo in &combos {
            let direct = substitute::render(op, combo, &set).expect("字符串路径");
            let nodes = build(op, combo, &set).expect("AST 构建");
            assert_eq!(direct, render(&nodes), "fuzz 双射破裂（{combo}）");
        }
    }
}

/// 确定性 LCG：无外部依赖的伪随机源。
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// 在随机位置插入空白 / 换行 / 分号，制造缩进与终结符的边界情况。
///
/// `$` 占位符的名字整体保留：不在 `$ax` 与 `is` 之间插入字符，否则会把 `$axis`
/// 拆成不存在的 `$ax`——那是生成器在破坏输入，不是被测代码的问题。
fn perturb(body: &str, state: &mut u64) -> String {
    const TOKENS: [&str; 5] = [" ", "  ", "\n", "\n    ", ";"];
    let mut out = String::with_capacity(body.len() + 16);
    let mut chars = body.chars().peekable();
    while let Some(ch) = chars.next() {
        out.push(ch);
        if ch == '$' {
            while let Some(&next) = chars.peek() {
                if next.is_ascii_alphanumeric() || next == '_' {
                    out.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            continue;
        }
        if lcg(state).is_multiple_of(7) {
            out.push_str(TOKENS[(lcg(state) as usize) % TOKENS.len()]);
        }
    }
    out
}

/// 取一台设备上某算子的 lower 产物；缺片段 / 展开 / 构建任一步都返回 `None`。
fn lower_nodes(catalogue: &Catalogue, name: &str) -> Option<Kernel> {
    let op = catalogue.op(name)?;
    let combo = crate::decl::expand(op).ok()?.remove(0);
    let fragments = prof::builtin("cpu", "c")?;
    let nodes = build(op, &combo, &fragments).ok()?;
    lower(op, &nodes)
}

#[test]
fn lower_pointwise_arithmetic_to_ir() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    for (name, arith) in [
        ("add_f32", Arith::Add),
        ("sub_f32", Arith::Sub),
        ("mul_f32", Arith::Mul),
        ("div_f32", Arith::Div),
    ] {
        let kernel = lower_nodes(&catalogue, name).expect("四则应 lower 成 IR");
        kernel.verify().expect("lower 产物过闸");
        let Write::Map(Expr::Binary { op, left, right }) = &kernel.write else {
            panic!("`{name}` 逐点应为 Binary Map");
        };
        assert_eq!(*op, arith);
        assert!(matches!(left.as_ref(), Expr::Load(_)), "左操作数应是缓冲读");
        assert!(
            matches!(right.as_ref(), Expr::Load(_)),
            "右操作数应是缓冲读"
        );
    }
}

#[test]
fn lower_reduce_sum_to_accumulating_ir() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let kernel = lower_nodes(&catalogue, "reduce_sum_f32").expect("求和归约应 lower");
    kernel.verify().expect("lower 产物过闸");
    let Write::Reduce { at, value } = &kernel.write else {
        panic!("求和归约应为 Reduce");
    };
    assert_eq!(*at, 0);
    assert!(matches!(value, Expr::Load(buf) if buf == "a"));
}

#[test]
fn templates_outside_the_lower_closed_set_stay_none() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let fragments = prof::builtin("cpu", "c").expect("CPU 片段");
    const LOWERABLE: [&str; 6] = [
        "add_f32",
        "sub_f32",
        "mul_f32",
        "div_f32",
        "reduce_sum_f32",
        "residual_add_f32",
    ];
    for op in catalogue.ops() {
        let Ok(combo) = crate::decl::expand(op) else {
            continue;
        };
        let Some(combo) = combo.into_iter().next() else {
            continue;
        };
        let Ok(nodes) = build(op, &combo, &fragments) else {
            continue;
        };
        if lower(op, &nodes).is_some() {
            assert!(
                LOWERABLE.contains(&op.name.as_str()),
                "`{}` 不在 lower 闭集内却被识别",
                op.name
            );
        }
    }
}

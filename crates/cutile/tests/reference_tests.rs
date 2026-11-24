//! 对拍口径的单元测试：误差量级、长度路径、阈值分档与非有限值分类。

use super::*;

#[test]
fn relative_error_uses_the_biggest_magnitude() {
    assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    let err = max_rel_err(&[1.0, 2.1], &[1.0, 2.0]);
    assert!((err - 0.05).abs() < 1e-6, "{err}");
    assert!(within(&[1.0, 2.0], &[1.0, 2.0], DType::F32));
    assert!(!within(&[1.0, 2.5], &[1.0, 2.0], DType::F32));
}

#[test]
fn every_length_takes_the_same_path_to_the_same_answer() {
    // 8 路展开后有三条路径：整块、整块 + 尾部、只有尾部。三者对同一批数必须一致。
    let want: Vec<f32> = (0..19u32).map(|i| i as f32 * 0.5).collect();
    let mut got = want.clone();
    got[17] += 0.25;
    let expected = 0.25 / 9.0;
    assert!((max_rel_err(&got, &want) - expected).abs() < 1e-9);
    // 长度不足一组：只走尾部。
    assert_eq!(max_rel_err(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    assert_eq!(max_rel_err(&[], &[]), 0.0);
    // 不是同一批数时是无穷，而不是某个"很大"的误差。
    assert_eq!(max_rel_err(&[1.0], &[1.0, 2.0]), f64::INFINITY);
}

#[test]
fn tolerance_follows_dtype() {
    assert_eq!(tolerance(DType::F16), 1e-3);
    assert_eq!(tolerance(DType::F32), 1e-5);
    assert_eq!(tolerance(DType::I8), 0.0);
}

#[test]
fn non_finite_values_are_classified_not_averaged_away() {
    // 同是 NaN、同号 Inf 算对上；其余非有限组合是不一致，不能被 max 当缺席吞掉。
    assert!(within(&[f32::NAN], &[f32::NAN], DType::F32));
    assert!(within(&[f32::INFINITY], &[f32::INFINITY], DType::F32));
    assert!(!within(&[f32::NAN], &[0.0], DType::F32));
    assert!(!within(&[f32::INFINITY], &[f32::NEG_INFINITY], DType::F32));
    assert!(!within(&[f32::INFINITY], &[1.0], DType::F32));
    assert_eq!(max_rel_err(&[f32::NAN], &[0.0]), f64::INFINITY);
}

#[test]
fn every_accepted_fusion_is_either_testable_or_named_unrealizable() {
    // README §9 的契约：`plan::fusions` 接受的每一条融合都必须能生成一条对拍。这里在整张内置
    // 注册表上把契约变成可执行断言，不留"悄悄没测"的结局：
    //   Ok          → 每条缝都桥接了（`bridged.len() == seams.len()`）；
    //   Invalid     → 几何上拼不出来（gemv 的 `[N]` 喂不进 elementwise 的 `[M*N]`），必须说形状；
    //   Unsupported → 多输出 / U8 缝，标量参考答不了；
    //   跳过        → 样本面上没有生产者的输入，这条融合的 ABI 压根没构造出来。
    // 唯独 `Missing` 是 bug：样本按消费者声明构造，代数若点出样本里没有的槽位，就是发明了名字。
    let sources = prof::catalogue::all();
    let decls: Vec<crate::decl::OpDecl> = sources
        .iter()
        .map(|source| {
            crate::decl::OpDecl::parse(source.declaration(), source.template()).expect("声明自洽")
        })
        .collect();
    let models: Vec<(&str, Model)> = sources
        .iter()
        .filter_map(|source| nn::registry::model(source.name()).map(|model| (source.name(), model)))
        .collect();

    let accepted = crate::plan::fusions(&decls, 12);
    assert!(!accepted.is_empty(), "整张注册表里必须有能拼的对");
    let (mut testable, mut unrealizable, mut unshaped) = (0usize, 0usize, 0usize);
    for fusion in &accepted {
        assert!(!fusion.is_self(), "自融合不该进候选：{fusion:?}");
        assert!(!fusion.seams.is_empty(), "空缝不是融合：{fusion:?}");
        let (Some((_, producer)), Some((_, consumer))) = (
            models.iter().find(|(name, _)| *name == fusion.producer),
            models.iter().find(|(name, _)| *name == fusion.consumer),
        ) else {
            continue; // 没有语义投影的算子无从对拍，代数与语义两张表的差异另有测试管。
        };
        // 样本按消费者声明构造：融合核的 ABI 就是消费者那一侧，缝写进去的也是它的槽位。
        let mut sample = (consumer.sample)(consumer.shape.0, consumer.shape.1);
        let declared = decls
            .iter()
            .find(|decl| decl.name == fusion.producer)
            .is_some_and(|decl| {
                decl.phases
                    .iter()
                    .flat_map(|phase| phase.reads.iter())
                    .all(|name| sample.buf(name).is_some())
            });
        if !declared {
            unshaped += 1;
            continue;
        }
        match verify_fusion(producer, consumer, fusion, &mut sample) {
            Ok(oracle) => {
                assert_eq!(
                    oracle.bridged.len(),
                    fusion.seams.len(),
                    "{} → {} 只桥了一部分缝",
                    fusion.producer,
                    fusion.consumer
                );
                assert_eq!(oracle.got.len(), consumer.shape.1.max(1), "产出长度");
                testable += 1;
            }
            Err(error) => {
                assert_ne!(
                    error.code(),
                    Code::Missing,
                    "代数发明了样本里没有的槽位（{} → {}）：{error}",
                    fusion.producer,
                    fusion.consumer
                );
                if error.code() == Code::Invalid {
                    assert!(
                        error.to_string().contains("形状对不上"),
                        "只有几何不匹配能算 Invalid：{error}"
                    );
                    unrealizable += 1;
                }
            }
        }
    }
    assert_eq!(
        testable + unrealizable + unshaped,
        accepted.len(),
        "被接受的融合必须逐条落到某一种结局"
    );
    // 实测基线：内置注册表的 ABI 多数互不相同——`gemv_f32` 的 `y` 是 `[N]`，elementwise 是 `[M*N]`；
    // 代数只看名字与类型、不看形状。当前唯一几何可实现的融合是 `abs_f32 → add_f32`：同为逐元素、
    // 同秩同长，abs 产出行连续桥进 add 的 `a` 输入槽，oracle 链路由融合对拍用例独立背书。
    // 这是快照绊线：再增 elementwise 邻近对、或把形状纳入代数时，逐条复核 oracle 后再调基线。
    assert_eq!(
        testable, 1,
        "几何可实现融合条数偏离当前快照（可测 {testable} / 几何不可行 {unrealizable} / 未构造样本 {unshaped}）"
    );
}

/// 建一条真实的声明：内置算子的 TOML + 模板，过一遍拼装期校验。
fn declared(name: &str) -> crate::decl::OpDecl {
    let source = prof::catalogue::named(name).expect("内置算子");
    crate::decl::OpDecl::parse(source.declaration(), source.template()).expect("声明自洽")
}

#[test]
fn the_fused_oracle_bridges_every_seam_not_just_one() {
    let mul = nn::registry::model("mul_f32").expect("mul_f32 已登记");
    let add = nn::registry::model("add_f32").expect("add_f32 已登记");
    let fusion = crate::decl::compose(&declared("mul_f32"), &declared("add_f32")).expect("可拼");
    assert_eq!(fusion.outputs(), ["y"], "生产者只有一条输出");
    assert_eq!(fusion.seams.len(), 2, "add 的两个读槽位各一条缝");

    let mut sample = (mul.sample)(8, 8);
    let a = sample.f32("a").expect("样本有 a").to_vec();
    let b = sample.f32("b").expect("样本有 b").to_vec();
    // 完整桥接：mul 的结果同时喂给 add 的两路输入。
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y + x * y).collect();
    let oracle = verify_fusion(&mul, &add, &fusion, &mut sample).expect("融合参考可算");
    assert_eq!(oracle.bridged.len(), fusion.seams.len(), "每条缝都必须桥接");
    assert!(within(&oracle.got, &want, DType::F32), "对拍失败");

    // 只桥一条缝（旧口径）算的是另一个程序：b 仍是原值而不是生产者产出。
    let half: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y + y).collect();
    assert!(
        !within(&oracle.got, &half, DType::F32),
        "旧口径必须与完整桥接不同"
    );
}

#[test]
fn the_fused_oracle_refuses_what_it_cannot_bridge() {
    let mul = nn::registry::model("mul_f32").expect("mul_f32 已登记");
    let add = nn::registry::model("add_f32").expect("add_f32 已登记");
    let real = crate::decl::compose(&declared("mul_f32"), &declared("add_f32")).expect("可拼");
    let seam = |from: &str, to: &str, ctype| crate::decl::Seam {
        from: from.to_string(),
        to: to.to_string(),
        ctype,
    };
    let code_of = |fusion: &crate::decl::Fusion| {
        verify_fusion(&mul, &add, fusion, &mut (mul.sample)(8, 8))
            .err()
            .expect("必须拒绝")
            .code()
    };

    let mut self_fusion = real.clone();
    self_fusion.consumer = self_fusion.producer.clone();
    assert_eq!(code_of(&self_fusion), Code::Invalid, "自融合是循环携带依赖");

    let mut two_outputs = real.clone();
    two_outputs.seams.push(seam("z", "a", CType::F32));
    assert_eq!(
        code_of(&two_outputs),
        Code::Unsupported,
        "标量参考只给一条输出"
    );

    let mut u8_seam = real.clone();
    for seam in &mut u8_seam.seams {
        seam.ctype = CType::U8;
    }
    assert_eq!(code_of(&u8_seam), Code::Unsupported, "U8 没有字节级 oracle");

    let mut missing = real.clone();
    missing.seams = vec![seam("y", "nope", CType::F32)];
    assert_eq!(code_of(&missing), Code::Missing, "槽位必须真的在样本里");

    let mut empty = real;
    empty.seams.clear();
    assert_eq!(code_of(&empty), Code::Invalid, "没有缝就没有可对拍的程序");
}

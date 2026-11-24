use crate::sch;

use super::*;

fn builtin() -> OpDecl {
    // 按名字取 gemv_f32：注册表按名字排序，新增算子后不应再假设索引 0 是哪条。
    let op = crate::catalogue::named("gemv_f32").expect("gemv_f32 内置");
    op.to_decl().expect("内置声明必须自洽")
}

/// 计数与展开必须同一个口径：装载期只数不物化，调用方仍然可以拿到组合。
#[test]
fn the_tile_count_matches_the_expanded_combinations() {
    let op = builtin();
    assert_eq!(
        tiles(&op).expect("未超限"),
        expand(&op).expect("可展开").len()
    );
}

#[test]
fn a_tile_overflow_names_the_axis_to_prune() {
    let mut op = builtin();
    // 把首轴 GMAX 撑到 9 个取值：候选数 9 超过上限 8。
    op.axes[0].values = vec![1, 2, 4, 8, 16, 32, 64, 128, 256];
    let error = tiles(&op).expect_err("9 个候选超过上限 8");
    assert_eq!(error.code(), Code::Invalid);
    let text = error.to_string();
    assert!(text.contains("GMAX"), "{text}");
    assert!(text.contains("超过上限 8"), "{text}");
    assert!(text.contains("9 个 tile 候选"), "{text}");
}

/// 属性键属于原语自己的签名：`sat` 是 `cvt` 的参数，写在 `store` 上既不会被协商检查，
/// 也不会被片段消费。判据必须点名原语、键与该原语真正允许的参数。
#[test]
fn an_attr_key_must_belong_to_its_primitive() {
    let store = Prim::Store;
    assert!(!store.accepts(AttrKey::Sat), "`store` 不接受 `sat`");
    let mut allowed: Vec<&str> = store.attr_keys().iter().map(|key| key.as_str()).collect();
    allowed.push(AttrKey::Exec.as_str());
    let text = allowed.join(" | ");
    assert_eq!(text, "src | dst | align | exec");
}

/// 降级链的单调性判据接受借用名：装载期不为它拷一份 `String`。
#[test]
fn the_monotonic_chain_is_checked_on_borrowed_names() {
    let chain = [("first", 0.5), ("second", 0.5)];
    assert!(sch::check_monotonic(&chain).is_ok());
    let reversed = [("first", 0.4), ("second", 0.9)];
    assert!(sch::check_monotonic(&reversed).is_err());
}

/// 手搓一个最小算子：只填 `compose`/`reads`/`writes` 用得到的字段，签名从模板解析。
fn minimal(name: &str, template: &str, phases: Vec<Phase>) -> OpDecl {
    OpDecl {
        name: name.to_string(),
        skeleton: Skeleton::Elementwise,
        summary: String::new(),
        axes: Vec::new(),
        needs: Vec::new(),
        layouts: Vec::new(),
        grid: None,
        chunk: None,
        consts: Vec::new(),
        variants: Vec::new(),
        template: template.to_string(),
        signature: crate::sig::Signature::parse(name, template).expect("签名"),
        phases,
    }
}

/// 槽位代数：生产者写出的缓冲与消费者读入的同类型缓冲就是一条缝。
#[test]
fn compose_pairs_a_producer_write_with_a_consumer_read_of_the_same_dtype() {
    let producer = crate::catalogue::named("gemv_f32")
        .map(|op| op.to_decl().expect("gemv_f32 自洽"))
        .expect("gemv_f32 内置");
    let consumer = crate::catalogue::named("gelu_f32")
        .map(|op| op.to_decl().expect("gelu_f32 自洽"))
        .expect("gelu_f32 内置");
    let fusion = compose(&producer, &consumer).expect("写→读同型可拼");
    assert_eq!(fusion.producer, "gemv_f32");
    assert_eq!(fusion.consumer, "gelu_f32");
    assert!(!fusion.seams.is_empty(), "至少一条缝");
    assert!(
        fusion
            .seams
            .iter()
            .all(|seam| seam.ctype == crate::sig::CType::F32)
    );
}

/// 缺省阶段：单内核 = 单阶段，名字取算子名，读/写槽位从签名派生。
#[test]
fn the_default_phase_is_derived_from_the_signature() {
    let op = builtin();
    assert_eq!(
        op.phases.len(),
        1,
        "gemv_f32 没有显式 [[phases]]，应派生单阶段"
    );
    let phase = &op.phases[0];
    assert_eq!(phase.name, "gemv_f32");
    assert_eq!(phase.writes, ["y"], "可写输出缓冲恰好一个");
    assert_eq!(
        op.signature.buf_names(false),
        phase.reads,
        "读槽位与签名 const 缓冲一致"
    );
}

/// 类型不一致不拼：生产者只写 `u8`，消费者只读 `f32`，没有任何缝。
#[test]
fn compose_rejects_a_dtype_mismatch() {
    let producer = minimal(
        "u8w",
        "void cutile_u8w_kernel(const float *x, u8 *y) { }\n",
        vec![Phase {
            name: "u8w".to_string(),
            reads: vec!["x".to_string()],
            writes: vec!["y".to_string()],
        }],
    );
    let consumer = minimal(
        "f32r",
        "void cutile_f32r_kernel(const float *x, float *y) { }\n",
        vec![Phase {
            name: "f32r".to_string(),
            reads: vec!["x".to_string()],
            writes: vec!["y".to_string()],
        }],
    );
    assert!(compose(&producer, &consumer).is_none(), "u8 写 ≠ f32 读");
}

/// 自融合不拼：`gemv_f32` 写 `y`、读 `w`/`x`，全是 F32，所以"gemv→gemv"在槽位上看起来有两条
/// 缝（`y→w`、`y→x`）——但那是循环携带依赖，拼出来的融合核无法定义执行顺序，还会白占一个预算
/// 名额。槽位代数必须自己排除它，不能指望调用方记得去重。
#[test]
fn compose_rejects_a_self_fusion() {
    let gemv = builtin();
    assert!(compose(&gemv, &gemv).is_none(), "同一算子不是一条缝");
    let single_output = minimal(
        "doubler",
        "void cutile_doubler_kernel(const float *x, float *y) { }\n",
        vec![Phase {
            name: "doubler".to_string(),
            writes: vec!["y".to_string()],
            reads: vec!["x".to_string()],
        }],
    );
    assert!(
        compose(&single_output, &single_output).is_none(),
        "名字相同即自融合，与槽位是否对得上无关"
    );
}

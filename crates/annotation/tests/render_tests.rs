//! 渲染层的单元测试：生成源码的每一处契约（以及它**不该**出现的那些东西）。
//!
//! 断言的是文本，不是行为：渲染器只拼字符串，于是"生成的代码长什么样"可以被逐字钉住。

use crate::decl::{
    AttrModel, AxisModel, DTYPES, Declaration, ModelDecl, NeedModel, OpModel, RuntimeDecl,
    SKELETONS, SpanModel, VariantModel, parse, parse_model, parse_parallel, parse_runtime,
};
use crate::fixture::lex;
use crate::render::{
    MODEL_ROOM, RUNTIME_ROOM, STATIC_ROOM, render, render_model, render_ops, render_parallel,
    render_runtime,
};

fn declaration(text: &str) -> Declaration {
    parse(&lex(text)).expect("合法注解")
}

fn runtime_declaration(text: &str) -> RuntimeDecl {
    parse_runtime(&lex(text)).expect("合法运行时注解")
}

/// 一个最小线性层的生成结果：多条用例共享这份夹具，各自只加一条契约。
fn q() -> String {
    render(
        &declaration(r#"name = "q", out = [8], ops = ["gemv_i4"]"#),
        "Q",
    )
}

#[test]
fn a_runtime_declaration_generates_a_constructor_macro() {
    let code = render_runtime(
        &runtime_declaration(r#"name = $name, out = [$tokens, $dim], ops = [$op], role = $role"#),
        "Embedding",
    );
    assert!(
        code.contains("macro_rules! __layer_spec_Embedding"),
        "{code}"
    );
    assert!(
        code.contains("($name:expr, $tokens:expr, $dim:expr, $op:expr, $role:expr)"),
        "{code}"
    );
    assert!(
        code.contains("::nn::ContractSpec::new_with_role("),
        "{code}"
    );
    assert!(code.contains("::nn::Shape::new([$tokens, $dim])"), "{code}");
    assert!(code.contains("::std::vec![$op.to_string()]"), "{code}");
    assert!(code.contains("$role,"), "{code}");
}

#[test]
fn a_runtime_declaration_dedupes_axes_and_defaults_the_role() {
    let code = render_runtime(
        &runtime_declaration(r#"name = $name, out = [$dim, $dim], ops = [$op]"#),
        "Square",
    );
    assert!(
        code.contains("($name:expr, $dim:expr, $op:expr)"),
        "同一轴宽只占一个宏参数：{code}"
    );
    assert!(code.contains("::nn::Shape::new([$dim, $dim])"), "{code}");
    assert!(
        code.contains("::nn::ContractSpec::new("),
        "没有角色时走 `new`，与手写构造同一路径：{code}"
    );
    assert!(!code.contains("$role"), "{code}");
}

#[test]
fn the_generated_code_implements_the_contract_of_the_annotated_type() {
    let code = render(
        &declaration(r#"name = "lm_head", out = [4096], ops = ["gemv_i4"]"#),
        "LmHead",
    );
    assert!(code.contains("impl LmHead {"), "{code}");
    assert!(code.contains("impl ::nn::Contract for LmHead {"), "{code}");
    assert!(
        code.contains("fn __layer_spec() -> &'static ::nn::ContractSpec"),
        "{code}"
    );
    assert!(
    code.contains(r#"::nn::ContractSpec::new("lm_head", ::nn::Shape::new([4096usize]), ::std::vec!["gemv_i4".to_string()])"#),
    "{code}"
);
}

#[test]
fn a_static_role_is_rendered_into_the_checked_spec_constructor() {
    let code = render(
        &declaration(
            r#"name = "norm", out = [8], ops = ["rmsnorm_f32"], role = "norm""#,
        ),
        "Norm",
    );
    assert!(code.contains("::nn::ContractSpec::new_with_role("), "{code}");
    assert!(code.contains("::nn::ParamRole::Norm"), "{code}");
}

#[test]
fn an_input_width_generates_the_same_floor_path_as_a_linear_layer() {
    let code = render(
        &declaration(r#"name = "lm_head", out = [4096], in = 4096, ops = ["gemv_i4"]"#),
        "LmHead",
    );
    assert!(
        code.contains("fn bounds(&self) -> ::nn::BoundsResult {"),
        "{code}"
    );
    assert!(
        code.contains("::nn::Bounds::of_ops(<Self>::__layer_spec().ops(), 4096usize, 4096usize)"),
        "下界必须走与 Linear 同一条路径：{code}"
    );
}

#[test]
fn without_an_input_width_no_bounds_method_is_generated() {
    let code = q();
    assert!(
        !code.contains("fn bounds("),
        "没给形状就不要生成一个“空”实现：契约的默认值已经说明了这件事：{code}"
    );
}

#[test]
fn a_scalar_shape_is_written_with_an_explicit_element_type() {
    let code = render(
        &declaration(r#"name = "loss", out = [], ops = ["dot"]"#),
        "Loss",
    );
    assert!(
        code.contains("::std::vec::Vec::<usize>::new()"),
        "空形状必须能推断出元素类型：{code}"
    );
    assert!(!code.contains("Shape::new([])"), "{code}");
}

#[test]
fn the_check_module_runs_only_under_cfg_test() {
    let code = q();
    assert!(
        code.contains("#[cfg(test)]\n#[allow(non_snake_case)]\nmod __layer_check_Q {"),
        "{code}"
    );
    assert!(
        code.contains("fn the_annotation_still_matches_the_declared_operators()"),
        "{code}"
    );
    assert!(code.contains("#[test]"), "{code}");
}

#[test]
fn generated_identifiers_are_prefixed_so_they_cannot_collide_with_user_methods() {
    let code = q();
    assert!(code.contains("__layer_spec"), "{code}");
    assert!(
        !code.contains("fn declaration("),
        "不该占用 `declaration` 这类普通名字：{code}"
    );
}

#[test]
fn the_check_module_calls_the_inherent_helper_not_a_receiver_less_trait_method() {
    let code = q();
    assert!(
        code.contains("let spec = <super::Q>::__layer_spec();"),
        "{code}"
    );
    assert!(
        !code.contains("as ::nn::Contract>::spca()"),
        "`spca` 需要 &self，不能用无接收者的 UFCS：{code}"
    );
}

/// 生成代码的三种分隔符必须严格配对。
///
/// 只数花括号会漏掉 `assert_eq!(a, b), "…")` 这类**多出来的 `)`**：它让整份生成物不是合法 token
/// 流，而 `contains` 断言照样全绿，直到某个真实的 `#[annotation::model]` 消费方编译才暴露。
fn assert_balanced(code: &str) {
    let mut open: Vec<char> = Vec::new();
    let mut chars = code.chars();
    while let Some(character) = chars.next() {
        match character {
            '"' => quoted(&mut chars),
            '(' | '[' | '{' => open.push(character),
            ')' | ']' | '}' => assert_eq!(
                open.pop().map(closing),
                Some(character),
                "分隔符不配对（`{character}`）：{code}"
            ),
            _ => {}
        }
    }
    assert!(open.is_empty(), "分隔符没有闭合：{code}");
}

/// 开括号期待的闭括号。
fn closing(open: char) -> char {
    match open {
        '(' => ')',
        '[' => ']',
        _ => '}',
    }
}

/// 跳过一段字符串字面量：里面的分隔符是文案，不是代码结构。
fn quoted(chars: &mut std::str::Chars<'_>) {
    while let Some(character) = chars.next() {
        match character {
            '\\' => {
                chars.next();
            }
            '"' => return,
            _ => {}
        }
    }
}

/// 三种形态、每个骨架、每种 dtype 都要配平：生成物会被原样喂给 `TokenStream::from_str`，
/// 所以"多一个 `)`"与"少一个 `}`"是同一级别的故障。只挑一个代表骨架会漏掉分支差异。
#[test]
fn every_form_renders_balanced_delimiters() {
    assert_balanced(&render(
        &declaration(r#"name = "q", out = [2, 3], ops = ["gemv_i4"]"#),
        "Q",
    ));
    assert_balanced(&render_runtime(
        &runtime_declaration(r#"name = $name, out = [$a, $b], ops = [$op, $op2], role = $r"#),
        "R",
    ));
    for skeleton in SKELETONS {
        // `gemm` 骨架用注册名选模块，名字必须落在 gemm 家族里；其余骨架用自身名字即可。
        let name = if skeleton == "gemm" {
            "gemm_f32"
        } else {
            skeleton
        };
        for dtype in DTYPES {
            let text = format!(
                r#"name = "{name}", shape = [8, 4], dtype = "{dtype}", skeleton = "{skeleton}""#
            );
            assert_balanced(&render_model(&model(&text), "M"));
        }
    }
}

/// "渲染一次只分配一次"是可验证的结构事实，不是计时结论：`String::with_capacity(n)` 的容量恰好
/// 是 `n`，一旦写超就翻倍增长。所以 `capacity()` 仍等于预留值，就证明最长的一层也没溢出——
/// 把预留量调小、或某块渲染偷偷多存一次，这条断言都会红。
#[test]
fn one_render_allocates_exactly_once() {
    let layer = declaration(
        r#"name = "multi_head_attention_projection_layer",
           out = [512, 64, 12, 4096],
           ops = ["operator_0", "operator_1", "operator_2", "operator_3", "operator_4", "operator_5"]"#,
    );
    let code = render(&layer, "crate::layers::attention::MultiHeadAttention");
    assert_eq!(
        code.capacity(),
        STATIC_ROOM,
        "静态渲染写超了预留容量（多一次 realloc，长度 {}）",
        code.len()
    );

    let runtime = runtime_declaration(
        r#"name = $name, out = [$dim, $dim], ops = [$matmul, $bias], role = $role"#,
    );
    let code = render_runtime(&runtime, "crate::MultiHeadAttention");
    assert_eq!(
        code.capacity(),
        RUNTIME_ROOM,
        "运行时渲染写超了预留容量（多一次 realloc，长度 {}）",
        code.len()
    );

    let projection = model(r#"name = "gemm_f32", shape = [4096, 4096], dtype = "f32""#);
    let code = render_model(&projection, "crate::layers::attention::Projection");
    assert_eq!(
        code.capacity(),
        MODEL_ROOM,
        "语义表渲染写超了预留容量（多一次 realloc，长度 {}）",
        code.len()
    );
}

#[test]
fn a_missing_operator_is_reported_with_the_layer_name() {
    let code = q();
    assert!(
        code.contains(r#"::std::panic!("注解声明的层"#),
        "运行期失败也要说清是哪个层：{code}"
    );
}
fn model(text: &str) -> ModelDecl {
    parse_model(&lex(text)).expect("合法注解")
}

#[test]
fn the_model_renderer_projects_the_operator_contract() {
    let code = render_model(
        &model(r#"name = "gemv_i4", shape = [4096, 4096], dtype = "f32""#),
        "I4",
    );
    assert!(
        code.contains(
            "pub const MODEL: ::prof::model::Model = ::prof::operator::model_of::<I4>();"
        ),
        "MODEL 必须是 Operator 的投影：{code}"
    );
    assert!(
        code.contains(r#"::prof::same_bytes(<I4 as ::prof::operator::Operator>::NAME, "gemv_i4")"#),
        "注解注册名必须在编译期与 Operator::NAME 对齐：{code}"
    );
    assert!(
        code.contains(
            r#"::prof::same_bytes(<I4 as ::prof::operator::Operator>::DTYPE.name(), "f32")"#
        ),
        "注解里的 `dtype` 必须在编译期与 Operator::DTYPE 对齐（不再靠运行期解析）：{code}"
    );
    assert!(
        code.contains("<I4 as ::prof::operator::Operator>::SHAPE.0 == 4096usize"),
        "注解生产形状必须在编译期与 Operator::SHAPE 对齐：{code}"
    );
    assert!(
        code.contains("let group = <I4 as ::prof::operator::Operator>::GROUP as usize;")
            && code.contains("4096usize / group * group == 4096usize"),
        "粒度来自 Operator::GROUP，生产形状整除性仍是编译期断言：{code}"
    );
    assert!(
        !code.contains("group: <I4"),
        "MODEL 不再逐字段渲染，粒度不该出现第二份：{code}"
    );
    assert!(
        !code.contains("allow("),
        "生成的代码是固定标识符，不需要豁免任何 lint：{code}"
    );
}

/// 注解只认识契约 crate：展开到任何 crate 都成立。它不该提到注册表（`builtin_named` / `model(..)`）、
/// 某个执行核（`gemv::Format`）或 `.k` 签名——那些事实分别属于 `nn` 与 `prof`，由各自的测试核对。
#[test]
fn the_model_expansion_names_only_the_contract_crate() {
    let code = render_model(
        &model(r#"name = "gemv_i4", shape = [4096, 4096], dtype = "f32""#),
        "I4",
    );
    assert!(
        !code.contains("::llm_nn::"),
        "展开里不该出现注册表 crate 的路径：{code}"
    );
    assert!(
        !code.contains("builtin_named") && !code.contains("Signature::parse"),
        "注解不核对注册表与签名：{code}"
    );
    assert!(
        !code.contains("__model_check") && !code.contains("#[cfg(test)]"),
        "注解不生成测试模块（注册表驱动的一致性是 `nn` 的测试）：{code}"
    );
}

/// `const = ".."` 让同模块里的第二个算子拿一个不撞名的语义常量与测试模块。
#[test]
fn a_named_const_gives_two_models_distinct_names_in_one_module() {
    let code = render_model(
        &model(
            r#"name = "flash_attention_f32", shape = [8, 8], dtype = "f32", skeleton = "attention", const = "FLASH_MODEL""#,
        ),
        "FlashAttention",
    );
    assert!(
        code.contains("pub const FLASH_MODEL: ::prof::model::Model"),
        "{code}"
    );
    assert!(
        !code.contains("pub const MODEL:"),
        "写了 const 键就不该再生成默认名：{code}"
    );
    assert!(
        code.contains("::prof::operator::model_of::<FlashAttention>()"),
        "const 名只改常量标识符，投影的仍是同一个 Operator：{code}"
    );
}

/// 闭集里的每个骨架都必须有生成分支：加一个骨架名却忘了加分支，会在用户 crate 里变成
/// 一条 `compile_error!`，而这条测试让它先在注解层自己红掉。
#[test]
fn every_skeleton_projects_through_the_same_contract() {
    for skeleton in SKELETONS {
        let name = if skeleton == "gemm" {
            "gemm_f32"
        } else {
            "op_under_test"
        };
        let code = render_model(
            &model(&format!(
                r#"name = "{name}", shape = [8, 4], dtype = "f32", skeleton = "{skeleton}""#
            )),
            "M",
        );
        assert!(
            !code.contains("compile_error!"),
            "骨架 `{skeleton}` 没有投影出 MODEL：{code}"
        );
        assert!(
            code.contains("::prof::operator::model_of::<M>()"),
            "所有骨架走同一条投影：{code}"
        );
    }
}

#[test]
fn the_generated_model_code_has_balanced_braces() {
    let code = render_model(&model(r#"name = "q", shape = [8, 4], dtype = "f16""#), "Q");
    assert_balanced(&code);
}

/// 闭集一旦被绕过（例如手写构造了一个未映射的 `dtype`），生成器早退回 `compile_error!`，
/// 而不是悄悄挑一个别的变体绑进 `MODEL`。
#[test]
fn an_unmapped_dtype_falls_back_to_a_compile_error() {
    let decl = ModelDecl {
        name: "x".to_string(),
        shape: [8, 4],
        dtype: "f96".to_string(),
        skeleton: "gemv".to_string(),
        const_name: "MODEL".to_string(),
    };
    let code = render_model(&decl, "M");
    assert!(code.contains("compile_error!"), "{code}");
    assert!(code.contains("DType"), "{code}");
    assert_balanced(&code);
}

/// 骨架表里没有的名字同样早退回编译错误：不允许落到一个默认分支去绑错模型。
#[test]
fn an_unlisted_skeleton_falls_back_to_a_compile_error() {
    let decl = ModelDecl {
        name: "x".to_string(),
        shape: [8, 4],
        dtype: "f32".to_string(),
        skeleton: "ghost".to_string(),
        const_name: "MODEL".to_string(),
    };
    let code = render_model(&decl, "M");
    assert!(code.contains("compile_error!"), "{code}");
    assert!(code.contains("骨架"), "{code}");
    assert_balanced(&code);
}

/// 全量策略：度数、归约变体、平均开关、优先级变体逐字落进访问器。
#[test]
fn a_parallel_declaration_renders_the_policy_accessor() {
    let decl = parse_parallel(&lex(
        r#"tp = 2, pp = 1, dp = 4, reduce = "max", average = false, priority = "high""#,
    ))
    .expect("合法");
    let code = render_parallel(&decl, "Block");
    assert!(
        code.contains("::shard::ParallelConfig::new(2, 1, 4)"),
        "{code}"
    );
    assert!(code.contains("::shard::Reduce::Max"), "{code}");
    assert!(code.contains(".with_average(false)"), "{code}");
    assert!(code.contains("::shard::Priority::High"), "{code}");
    assert!(code.contains("fn __parallel_config()"), "{code}");
    assert_balanced(&code);
}

/// 缺省策略回落求和 / 平均 / 普通优先级，变体名与单进程口径一致。
#[test]
fn a_default_parallel_declaration_renders_sum_normal() {
    let decl = parse_parallel(&lex("")).expect("合法");
    let code = render_parallel(&decl, "Block");
    assert!(
        code.contains("::shard::ParallelConfig::new(1, 1, 1)"),
        "{code}"
    );
    assert!(code.contains("::shard::Reduce::Sum"), "{code}");
    assert!(code.contains(".with_average(true)"), "{code}");
    assert!(code.contains("::shard::Priority::Normal"), "{code}");
    assert_balanced(&code);
}

/// 一条最小逐元素算子夹具：与 `add_f32.toml` 同形。
fn sample_op() -> OpModel {
    OpModel {
        name: "add_f32".to_string(),
        skeleton: "elementwise".to_string(),
        summary: "y[n] = a[n] + b[n]".to_string(),
        axes: vec![AxisModel {
            name: "N".to_string(),
            values: vec![32],
        }],
        needs: vec![NeedModel {
            prim: "load".to_string(),
            extent: SpanModel::AtMost(4096),
            n: SpanModel::Any,
            k: SpanModel::Any,
            dt: vec![],
            space: vec!["GM".to_string(), "FAST".to_string()],
            attrs: vec![AttrModel {
                key: "src".to_string(),
                values: vec!["GM".to_string()],
            }],
        }],
        layouts: vec![("a".to_string(), "row".to_string())],
        grid: None,
        chunk: None,
        consts: vec![("binop".to_string(), "+".to_string())],
        variants: vec![VariantModel {
            name: "direct".to_string(),
            order: 1,
            schedule: "direct".to_string(),
            mma_share: 0.0,
            extra_needs: vec![],
        }],
        phases: vec![],
    }
}

/// 算子表渲染出强类型 `OpView` 字面量：闭集短名逐条映射到 prof 变体路径。
#[test]
fn ops_table_renders_typed_op_views() {
    let code = render_ops(&[sample_op()]);
    assert!(
        code.contains("pub const OPS: [::llm_prof::decl::OpView; 1] = ["),
        "{code}"
    );
    assert!(
        code.contains("::llm_prof::decl::Skeleton::Elementwise"),
        "{code}"
    );
    assert!(code.contains("::llm_prof::atom::Prim::Load"), "{code}");
    assert!(code.contains("::llm_prof::atom::Space::Gm"), "{code}");
    assert!(code.contains("::llm_prof::atom::Space::Fast"), "{code}");
    assert!(code.contains("::llm_prof::atom::AttrKey::Src"), "{code}");
    assert!(
        code.contains("::llm_prof::atom::Span::at_most(4096)"),
        "{code}"
    );
    assert!(
        code.contains("::llm_prof::sch::layout::LayoutSpec::Row"),
        "{code}"
    );
    assert!(code.contains("::llm_prof::sch::Schedule::Direct"), "{code}");
    assert!(code.contains("values: &[32u32]"), "{code}");
    assert_balanced(&code);
}

/// 流水线调度与列布局带参构造逐字落进字面量。
#[test]
fn ops_table_renders_parameterized_schedule_and_layout() {
    let mut op = sample_op();
    op.variants[0].schedule = "pipe<2>".to_string();
    op.layouts[0] = ("w".to_string(), "col".to_string());
    let code = render_ops(&[op]);
    assert!(
        code.contains("::llm_prof::sch::Schedule::Pipe { stages: 2 }"),
        "{code}"
    );
    assert!(
        code.contains("::llm_prof::sch::layout::LayoutSpec::Col"),
        "{code}"
    );
    assert_balanced(&code);
}

/// 闭集外的骨架名不允许落到默认变体；整份生成物退一条 `compile_error!`。
#[test]
fn an_unmapped_op_skeleton_falls_back_to_compile_error() {
    let mut op = sample_op();
    op.skeleton = "ghost".to_string();
    let code = render_ops(&[op]);
    assert!(code.contains("compile_error!"), "{code}");
    assert!(code.contains("骨架"), "{code}");
    assert_balanced(&code);
}

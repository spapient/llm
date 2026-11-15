use super::*;
use crate::fixture::lex;
use crate::syntax;

fn declaration(text: &str) -> Declaration {
    syntax::parse(&lex(text)).expect("合法注解")
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
    let code = render(
        &declaration(r#"name = "q", out = [8], ops = ["gemv_i4"]"#),
        "Q",
    );
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
    let code = render(
        &declaration(r#"name = "q", out = [8], ops = ["gemv_i4"]"#),
        "Q",
    );
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
    let code = render(
        &declaration(r#"name = "q", out = [8], ops = ["gemv_i4"]"#),
        "Q",
    );
    assert!(code.contains("__layer_spec"), "{code}");
    assert!(
        !code.contains("fn declaration("),
        "不该占用 `declaration` 这类普通名字：{code}"
    );
}

#[test]
fn the_check_module_calls_the_inherent_helper_not_a_receiver_less_trait_method() {
    let code = render(
        &declaration(r#"name = "q", out = [8], ops = ["gemv_i4"]"#),
        "Q",
    );
    assert!(
        code.contains("let spec = <super::Q>::__layer_spec();"),
        "{code}"
    );
    assert!(
        !code.contains("as ::nn::Contract>::spca()"),
        "`spca` 需要 &self，不能用无接收者的 UFCS：{code}"
    );
}

/// 生成代码的花括号必须配平，且中途不出现多余的 `}`。
fn assert_balanced_braces(code: &str) {
    let mut depth: i32 = 0;
    for character in code.chars() {
        match character {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        assert!(depth >= 0, "出现了多余的 `}}`：{code}");
    }
    assert_eq!(depth, 0, "花括号没有配平：{code}");
}

#[test]
fn the_generated_braces_are_balanced() {
    let code = render(
        &declaration(r#"name = "q", out = [2, 3], ops = ["gemv_i4"]"#),
        "Q",
    );
    assert_balanced_braces(&code);
}

#[test]
fn a_missing_operator_is_reported_with_the_layer_name() {
    let code = render(
        &declaration(r#"name = "q", out = [8], ops = ["gemv_i4"]"#),
        "Q",
    );
    assert!(
        code.contains(r#"::std::panic!("注解声明的层"#),
        "运行期失败也要说清是哪个层：{code}"
    );
}
fn model(text: &str) -> ModelDecl {
    syntax::parse_model(&lex(text)).expect("合法注解")
}

#[test]
fn the_model_renderer_binds_the_constant_to_the_annotated_alias() {
    let code = render_model(
        &model(r#"name = "gemv_i4", shape = [4096, 4096], dtype = "f32""#),
        "I4",
    );
    assert!(
        code.contains("pub const MODEL: ::llm_op::model::Model"),
        "{code}"
    );
    assert!(
        code.contains("group: <I4 as ::llm_op::gemv::Format>::GRAIN as u32"),
        "粒度必须取类型上的常量，而不是注解里再写一个数：{code}"
    );
    assert!(code.contains("cost: ::llm_op::gemv::cost::<I4>"), "{code}");
    assert!(
        code.contains("dtype: ::llm_op::model::DType::F32"),
        "{code}"
    );
    assert!(
        code.contains("4096usize / <I4 as ::llm_op::gemv::Format>::GRAIN * <I4 as ::llm_op::gemv::Format>::GRAIN == 4096usize"),
        "生产形状与粒度的关系必须是编译期断言：{code}"
    );
    assert!(
        code.contains("const _: () = <I4 as ::llm_op::gemv::Format>::CHECK;"),
        "格式自证要由格式自己声明、在别名上被求值，宏不另抄一份规则：{code}"
    );
    assert!(
        !code.contains("GRAIN > 0"),
        "粒度与派生轴的规则属于 `Format::CHECK`，生成代码不该再写第二遍：{code}"
    );
    assert!(
        code.contains("signature.output_type().and_then(|ctype| ctype.dtype())")
            && code.contains("Some(super::MODEL.dtype)"),
        "注解的 dtype 必须与 `.k` 的可写输出交叉核对：{code}"
    );
    assert!(code.contains("mod __model_check {"), "{code}");
    assert!(
        code.contains(r#"::llm_op::model::DType::parse_name("f32")"#),
        "注解里的 `dtype` 必须解析成 `MODEL` 用的那个变体：{code}"
    );
    assert!(
        !code.contains("allow("),
        "生成的模块名是固定标识符，不需要豁免任何 lint：{code}"
    );
}

/// 闭集里的每个骨架都必须有生成分支：加一个骨架名却忘了加分支，会在用户 crate 里变成
/// 一条 `compile_error!`，而这条测试让它先在注解层自己红掉。
#[test]
fn every_skeleton_has_a_branch() {
    for skeleton in syntax::SKELETONS {
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
            "骨架 `{skeleton}` 在 `render_model` 里没有分支：{code}"
        );
        assert!(
            code.contains("    axes: ::llm_op::") && code.contains("AXES,\n"),
            "骨架 `{skeleton}` 必须绑定一份轴表：{code}"
        );
    }
    for row in SKELETON_TABLE {
        assert!(
            syntax::SKELETONS.contains(&row.name),
            "骨架表里的 `{}` 不在闭集里：闭集是可选值的唯一来源，反向漏一行就等于允许了它",
            row.name
        );
    }
    assert_eq!(
        SKELETON_TABLE.len(),
        syntax::SKELETONS.len(),
        "表与闭集必须一一对应：多一行是死数据，少一行是生成不出 MODEL 的骨架"
    );
}

/// 表里三行不显然的事实各钉一条：具体骨架**不带** turbofish、`dequant_i4` 的粒度是 2、
/// flash attention 走另一条参考实现。三条都是"改一个字符就静默换掉生成代码"的点——它们
/// 原本散在三段 `else if` 里，现在集中在表里，所以判据也集中在这里。
#[test]
fn the_table_keeps_the_rows_that_are_easy_to_get_wrong() {
    let concrete = render_model(
        &model(r#"name = "rmsnorm_f32", shape = [8, 4], dtype = "f32", skeleton = "rmsnorm""#),
        "M",
    );
    assert!(
        concrete.contains("    cost: ::llm_op::rmsnorm_f32::cost,\n")
            && concrete.contains("    reference: ::llm_op::rmsnorm_f32::reference,\n"),
        "具体骨架的函数不是泛型的，不能带 `::<T>`：{concrete}"
    );
    let packed = render_model(
        &model(r#"name = "dequant_i4", shape = [8, 4], dtype = "f32", skeleton = "dequant_i4""#),
        "M",
    );
    assert!(
        packed.contains("    group: 2u32,\n"),
        "int4 每 2 个输出元素占一个字节，粒度写 1 会让 `verify` 从奇数形状起步：{packed}"
    );
    let flash = render_model(
        &model(
            r#"name = "flash_attention_f32", shape = [8, 4], dtype = "f32", skeleton = "attention""#,
        ),
        "M",
    );
    assert!(
        flash.contains("    reference: ::llm_op::attention::flash_reference,\n")
            && !flash.contains("    reference: ::llm_op::attention::reference,\n"),
        "flash 版与朴素版共用骨架，只有 `reference` 分叉：{flash}"
    );
    let plain = render_model(
        &model(r#"name = "attention_f32", shape = [8, 4], dtype = "f32", skeleton = "attention""#),
        "M",
    );
    assert!(
        plain.contains("    reference: ::llm_op::attention::reference,\n"),
        "非 flash 的 attention 走朴素实现：{plain}"
    );
}

#[test]
fn every_dtype_name_maps_to_the_expected_variant() {
    for (name, variant) in [("f16", "F16"), ("bf16", "BF16"), ("f32", "F32")] {
        assert_eq!(dtype_variant(name).as_deref(), Some(variant), "{name}");
    }
}

#[test]
fn the_generated_model_code_has_balanced_braces() {
    let code = render_model(&model(r#"name = "q", shape = [8, 4], dtype = "f16""#), "Q");
    assert_balanced_braces(&code);
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
    };
    let code = render_model(&decl, "M");
    assert!(code.contains("compile_error!"), "{code}");
    assert!(code.contains("DType"), "{code}");
}

/// 骨架表里没有的名字同样早退回编译错误：不允许落到一个默认分支去绑错模型。
#[test]
fn an_unlisted_skeleton_falls_back_to_a_compile_error() {
    let decl = ModelDecl {
        name: "x".to_string(),
        shape: [8, 4],
        dtype: "f32".to_string(),
        skeleton: "ghost".to_string(),
    };
    let code = render_model(&decl, "M");
    assert!(code.contains("compile_error!"), "{code}");
    assert!(code.contains("骨架"), "{code}");
}

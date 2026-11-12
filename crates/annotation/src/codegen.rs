//! 代码生成：把 [`Declaration`] 渲染成 `impl nn::Layer` 与一条校验测试。
//!
//! 这里只做字符串拼接、不维护任何状态，也不接触 `proc_macro`，因此可以用纯文本断言测试。
//! 生成的代码是确定的三块（被标注项由调用方原样拼接）：
//!
//! 1. 一个私有构造函数 `__layer_spec()`，用 `OnceLock` 保证 `LayerSpec` 只构造一次；
//! 2. 一份 `impl ::nn::Layer`，只把三个访问器转发给它；
//! 3. 一个 `#[cfg(test)]` 模块，在 `cargo test` 时校验"注解里的算子仍然存在"。
//!
//! 生成的名字一律带 `__layer_` 前缀：宏展开不该占用一个用户可能想用的普通名字
//! （`declaration` 这类名字太容易撞车，撞了之后报错会指向用户自己的方法）。

use crate::syntax::Declaration;

/// 渲染实现；`ty` 必须是 [`crate::syntax::type_name`] 给出的类型名。
pub(crate) fn render(declaration: &Declaration, ty: &str) -> String {
    let name = literal(declaration.name());
    let shape = shape(declaration.out());
    let ops = declaration
        .ops()
        .iter()
        .map(|op| format!("{}.to_string()", literal(op)))
        .collect::<Vec<_>>()
        .join(", ");

    let mut code = String::new();
    code.push_str("#[automatically_derived]\n");
    code.push_str(&format!("impl {ty} {{\n"));
    code.push_str("    #[doc(hidden)]\n");
    code.push_str("    fn __layer_spec() -> &'static ::nn::LayerSpec {\n");
    code.push_str(
        "        static SPEC: ::std::sync::OnceLock<::nn::LayerSpec> = ::std::sync::OnceLock::new();\n",
    );
    code.push_str("        SPEC.get_or_init(|| {\n");
    code.push_str(&format!(
        "            let declaration = ::nn::LayerSpec::new({name}, {shape}, ::std::vec![{ops}]);\n"
    ));
    code.push_str("            match declaration {\n");
    code.push_str("                ::core::result::Result::Ok(spec) => spec,\n");
    code.push_str(&format!(
        "                ::core::result::Result::Err(error) => {{\n\
         \x20                   ::std::panic!(\"注解声明的层 `{{}}` 引用了未声明的算子：{{}}\", {name}, error)\n\
         \x20               }}\n"
    ));
    code.push_str("            }\n");
    code.push_str("        })\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");

    code.push_str("#[automatically_derived]\n");
    code.push_str(&format!("impl ::nn::Layer for {ty} {{\n"));
    code.push_str("    fn name(&self) -> &str {\n        <Self>::__layer_spec().name()\n    }\n");
    code.push_str(
        "    fn output(&self) -> &::nn::Shape {\n        <Self>::__layer_spec().output()\n    }\n",
    );
    code.push_str(
        "    fn spec(&self) -> &::nn::LayerSpec {\n        <Self>::__layer_spec()\n    }\n",
    );
    // `in` 给了规约轴宽度，下界就有得算；没给就沿用契约里的默认实现（`Ok(None)`），
    // 而不是生成一个“返回空值”的方法把“这一层不懂自己的形状”藏起来。
    if let Some(width) = declaration.input() {
        let out = declaration.out()[0];
        code.push_str("    fn bounds(&self) -> ::nn::BoundsResult {\n");
        code.push_str(&format!(
            "        ::nn::Bounds::of_ops(<Self>::__layer_spec().ops(), {width}usize, {out}usize).map(::core::option::Option::Some)\n"
        ));
        code.push_str("    }\n");
    }
    code.push_str("}\n\n");

    code.push_str("#[cfg(test)]\n#[allow(non_snake_case)]\n");
    code.push_str(&format!("mod __layer_check_{ty} {{\n"));
    code.push_str("    #[test]\n    fn the_annotation_still_matches_the_declared_operators() {\n");
    code.push_str(&format!(
        "        let spec = <super::{ty}>::__layer_spec();\n"
    ));
    code.push_str(&format!("        assert_eq!(spec.name(), {name});\n"));
    code.push_str(&format!(
        "        assert_eq!(spec.output().dims(), &{:?});\n",
        declaration.out()
    ));
    code.push_str(&format!(
        "        assert_eq!(spec.ops(), &{:?});\n",
        declaration.ops()
    ));
    code.push_str("    }\n}\n");
    code
}

/// 形状表达式：空切片表示标量，必须显式写类型，否则 `new([])` 推断不出元素类型。
fn shape(dims: &[usize]) -> String {
    if dims.is_empty() {
        return "::nn::Shape::new(::std::vec::Vec::<usize>::new())".to_string();
    }
    let axes = dims
        .iter()
        .map(|dim| format!("{dim}usize"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("::nn::Shape::new([{axes}])")
}

/// Rust 字符串字面量。解析层已拒绝转义序列，因此 `{:?}` 的输出与输入逐字相同。
fn literal(text: &str) -> String {
    format!("{text:?}")
}

#[cfg(test)]
mod tests {
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
        assert!(code.contains("impl ::nn::Layer for LmHead {"), "{code}");
        assert!(
            code.contains("fn __layer_spec() -> &'static ::nn::LayerSpec"),
            "{code}"
        );
        assert!(
            code.contains(r#"::nn::LayerSpec::new("lm_head", ::nn::Shape::new([4096usize]), ::std::vec!["gemv_i4".to_string()])"#),
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
            code.contains(
                "::nn::Bounds::of_ops(<Self>::__layer_spec().ops(), 4096usize, 4096usize)"
            ),
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
            !code.contains("as ::nn::Layer>::spec()"),
            "`spec` 需要 &self，不能用无接收者的 UFCS：{code}"
        );
    }

    #[test]
    fn the_generated_braces_are_balanced() {
        let code = render(
            &declaration(r#"name = "q", out = [2, 3], ops = ["gemv_i4"]"#),
            "Q",
        );
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
}

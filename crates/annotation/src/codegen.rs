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

use crate::syntax::{Declaration, Frontend, ModelDecl};

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

/// 把 `#[frontend(..)]` 渲染成 `crate::witness::Language` 的实现 + 两条编译期断言。
///
/// 生成的东西只有三块，都是常量：语言（生成 `Emit` 的具体变体）、后缀、缓存标签。
/// 注解不负责执行——执行是手写的 `Device` 实现，由 `Frontend: Language + Device` 这条
/// 超 trait 绑死：只标注解不写执行，类型就拿不到 `Frontend`，泛型入口也就不接受它。
///
/// 另外生成一个 `#[cfg(test)]` 模块：把注解里的**字符串**与词汇表里的**变体**对齐
/// （`Emit::parse(emit) == Language::EMIT`）。它是“注解与词汇表不脱节”在**这个调用点**
/// 的证据；全量与映射规则由 `tests/contracts.rs` 统一核对。
///
/// 两条断言是注解与数据之间的契约，全部在编译期成立：
///
/// 1. `ext` 必须等于 `Emit::ext()`——写错会把见证编成一门语言、却按另一门的后缀找文件；
/// 2. `tag` 不能为空——它进产物身份，空标签会让所有程序的缓存条目挤在一起。
///
/// 路径写成 `crate::witness::..`：本注解只作用于 `audit` 的执行前端，不引入跨 crate 的
/// 隐式依赖（与 `#[layer]` 指向 `::nn::..` 是同一个约定：注解绑定它所属契约的路径）。
pub(crate) fn render_frontend(frontend: &Frontend, ty: &str) -> String {
    let emit = frontend.emit();
    let emit_literal = literal(emit);
    let ext = literal(frontend.ext());
    let tag = literal(frontend.tag());
    let variant = language_variant(emit);

    let mut code = String::new();
    code.push_str("#[automatically_derived]\n");
    code.push_str(&format!("impl crate::witness::Language for {ty} {{\n"));
    code.push_str(&format!(
        "    const EMIT: crate::witness::Emit = crate::witness::Emit::{variant};\n"
    ));
    code.push_str(&format!("    const EXT: &'static str = {ext};\n"));
    code.push_str(&format!("    const TAG: &'static str = {tag};\n"));
    code.push_str("}\n\n");
    code.push_str("const _: () = {\n");
    code.push_str(&format!(
        "    assert!(crate::witness::same_text(\n\
         \x20       <{ty} as crate::witness::Language>::EXT,\n\
         \x20       <{ty} as crate::witness::Language>::EMIT.ext(),\n\
         \x20   ));\n"
    ));
    code.push_str(&format!(
        "    assert!(!<{ty} as crate::witness::Language>::TAG.is_empty());\n"
    ));
    code.push_str("};\n\n");

    // 注解里的 `emit` 是**字符串**，`EMIT` 是**变体**，两者由「首字母大写」这条规则联系。
    // 这条测试把`首字母大写`与词汇表的名字钉在一起：`Emit::parse("opencl")` 必须就是
    // `Language::EMIT` 那个值。词汇表改名而注解不改，这里就会红；另一头（全量、顺序、
    // 变体名）在 `tests/contracts.rs` 里统一核对。
    code.push_str("#[cfg(test)]\n#[allow(non_snake_case)]\n");
    code.push_str(&format!("mod __frontend_check_{ty} {{\n"));
    code.push_str("    #[test]\n");
    code.push_str("    fn the_annotation_names_a_language_in_the_vocabulary() {\n");
    code.push_str(&format!(
        "        assert_eq!(\n\
         \x20           crate::witness::Emit::parse({emit_literal}),\n\
         \x20           Some(<super::{ty} as crate::witness::Language>::EMIT),\n\
         \x20           \"注解里的 `emit` 必须解析成 `Language::EMIT` 用的那个变体（词汇表是唯一名字源）\",\n\
         \x20       );\n"
    ));
    code.push_str("    }\n}\n");
    code
}

/// 语言名 → `Emit` 的变体名：首字母大写、其余原样。
///
/// 不做第二张"语言 → 变体"表：文法层已经用 [`crate::syntax::LANGUAGES`] 校验过闭集，
/// 这里再抄一份就有两个事实源，早晚会分家。`opencl → Opencl`、`ascendc → Ascendc` 也落在这条规则里。
fn language_variant(emit: &str) -> String {
    capitalised(emit)
}

/// `dtype` 名 → `prof::atom::DType` 的变体名。
///
/// 与 [`language_variant`] 同一条规则：闭集已在文法层（[`crate::syntax::DTYPES`]）校验过，
/// `f16 → F16`、`bf16 → Bf16`、`f32 → F32` 都落在"首字母大写"里。
fn dtype_variant(dtype: &str) -> String {
    capitalised(dtype)
}

/// 首字母大写、其余原样：两道闭集（语言、输出类型）共用这一条命名规则。
fn capitalised(word: &str) -> String {
    let mut characters = word.chars();
    match characters.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + characters.as_str(),
        None => String::new(),
    }
}

/// 把 `#[model(..)]` 渲染成 `MODEL` + 两条编译期断言 + 一条校验测试。
///
/// 生成的只有一块**数据**：`cost` / `sample` / `reference` 都是 `<类型 as Format>` 的单态化
/// 实例，`group` 直接取 `Format::GRAIN`。注解里的字面量因此只管注册名、生产形状与输出类型，
/// "这份语义是哪份格式"由类型参数保证——别名不实现 `Format` 就编译不过。
///
/// 两条断言是注解与类型之间的契约，全部在编译期成立：
///
/// 1. 形状粒度必须大于 0——`sample` 按它切标度，0 会让除法失真；
/// 2. 生产形状的 `M` 必须是它的整数倍——量化算子按分组对齐，写错在 `cargo check` 就失败，
///    而不是等 `doctor` 算代价时才报。
///
/// 生成的测试把四份事实对齐：注册表里的语义就是这份 `MODEL`、样本与 `.k` 签名一致、生产形状
/// 能算出代价、注解里的 `dtype` 字符串解析成同一个变体。任何一处漂移都在 `cargo test` 里
/// 失败，而不是在审计报告里表现成一个奇怪的数。
pub(crate) fn render_model(declaration: &ModelDecl, ty: &str) -> String {
    let name = literal(declaration.name());
    let [m, n] = declaration.shape();
    let dtype_name = literal(declaration.dtype());
    let dtype = dtype_variant(declaration.dtype());

    let mut code = String::new();
    code.push_str(&format!(
        "#[doc = \"`{ty}` 的语义表（`#[model]` 生成）：生产形状、输出类型与三条单态化入口。\"]\n"
    ));
    code.push_str("pub const MODEL: ::llm_op::model::Model = ::llm_op::model::Model {\n");
    code.push_str(&format!(
        "    group: <{ty} as ::llm_op::gemv::Format>::GRAIN as u32,\n"
    ));
    code.push_str(&format!("    shape: ({m}usize, {n}usize),\n"));
    code.push_str(&format!("    dtype: ::llm_op::model::DType::{dtype},\n"));
    code.push_str("    axes: ::llm_op::gemv::AXES,\n");
    code.push_str(&format!("    cost: ::llm_op::gemv::cost::<{ty}>,\n"));
    code.push_str(&format!("    sample: ::llm_op::gemv::sample::<{ty}>,\n"));
    code.push_str(&format!(
        "    reference: ::llm_op::gemv::reference::<{ty}>,\n"
    ));
    code.push_str("};\n\n");

    // 格式自证交给 `Format::CHECK`：粒度与派生轴除数同源这条规则属于格式契约，不属于宏。
    // 注解唯一要做的，是让它在**生产别名上**被求值——不被引用的关联常量不会报错。
    code.push_str(&format!(
        "const _: () = <{ty} as ::llm_op::gemv::Format>::CHECK;\n\n"
    ));
    code.push_str("const _: () = {\n");
    code.push_str(&format!(
        // 用"除回去乘回来"而不是 `%`：粒度可以是 1（稠密格式），`x % 1` 会被 clippy 的
        // `modulo_one` 判为恒真，而这里要表达的正是"形状能被粒度整除"。
        "    assert!({m}usize / <{ty} as ::llm_op::gemv::Format>::GRAIN * <{ty} as ::llm_op::gemv::Format>::GRAIN == {m}usize, \"生产形状的 M 必须是形状粒度的整数倍\");\n"
    ));
    code.push_str("};\n\n");

    code.push_str("#[cfg(test)]\nmod __model_check {\n");
    code.push_str("    #[test]\n");
    code.push_str("    fn the_annotation_agrees_with_the_registered_kernel() {\n");
    code.push_str(&format!(
        "        let op = ::llm_op::builtin_named({name}).expect(\"算子已注册\");\n"
    ));
    code.push_str(&format!(
        "        let signature = ::llm_op::sig::Signature::parse({name}, op.template()).expect(\"签名可解析\");\n"
    ));
    code.push_str(&format!(
        "        let registered = ::llm_op::model({name}).expect(\"语义已注册\");\n"
    ));
    code.push_str(
        "        assert_eq!(registered.group, super::MODEL.group, \"注册表必须指向这份语义\");\n",
    );
    code.push_str(
        "        assert_eq!(registered.shape, super::MODEL.shape, \"注册表必须指向这份语义\");\n",
    );
    code.push_str(
        "        assert_eq!(registered.dtype, super::MODEL.dtype, \"注册表必须指向这份语义\");\n",
    );
    code.push_str(
        "        let sample = (registered.sample)(registered.shape.0, registered.shape.1);\n",
    );
    code.push_str("        signature.check(&sample).expect(\"样本与 .k 签名一致\");\n");
    code.push_str(
        "        assert_eq!((registered.sample)(registered.shape.0, registered.shape.1), sample, \"样本必须可复现\");\n",
    );
    code.push_str(
        "        let shape = registered.projection(registered.shape.0, registered.shape.1).expect(\"两条投影轴都声明了\");\n",
    );
    code.push_str("        (registered.cost)(&shape).expect(\"生产形状必须能算出代价\");\n");
    // 注解里的 `dtype` 是**字符串**，`MODEL.dtype` 是**变体**：这一句就是两者之间的绑定点。
    // 词汇表改名、或生成代码挑错变体，都会在这里失败，而不是等到 `verify` 对一个错宽度的
    // 缓冲对拍。全量与映射规则由 `tests/contracts.rs` 统一核对。
    code.push_str("        assert_eq!(\n");
    code.push_str(&format!(
        "            ::llm_op::model::DType::parse({dtype_name}),\n"
    ));
    code.push_str("            Some(super::MODEL.dtype),\n");
    code.push_str(
        "            \"注解里的 `dtype` 必须解析成 `MODEL` 用的那个变体（词汇表是唯一名字源）\",\n",
    );
    code.push_str("        );\n");
    // 注解的 `dtype` 与 `.k` 的可写输出是同一件事的两种说法：前者说"回读按什么解释"，
    // 后者说"内核写的是什么"。只剩一半时（`.k` 写 `u8 *y` 而注解写 f32，或反过来）
    // 对拍会按另一种宽度解读缓冲，所以两者必须是同一个类型。`Model::dtype` 的阈值与
    // `harness` 的回读都取自它，这是它们在装载期唯一的交叉点。
    code.push_str("        assert!(\n");
    code.push_str(
        "            signature.output_type().and_then(|ctype| ctype.dtype()) == Some(super::MODEL.dtype),\n",
    );
    code.push_str(&format!(
        "            \"注解写 `dtype = {}`，但 `.k` 的可写输出是 `{{}}`：回读按 `MODEL.dtype` 解释，两者必须是同一个类型\",\n",
        declaration.dtype(),
    ));
    code.push_str("            signature.output_type().map_or(\"-\", |ctype| ctype.c_name()),\n");
    code.push_str("        );\n");
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

    /// 渲染出来的东西必须是常量 + 编译期断言：语言名机械映射到 `Emit` 变体，
    /// `ext` 与 `Emit::ext()` 的一致性、以及"标签非空"都在编译期判定。
    #[test]
    fn the_generated_front_end_binds_the_annotation_to_the_language_trait() {
        let frontend =
            syntax::parse_frontend(&lex(r#"emit = "opencl", ext = "cl", tag = "witness""#))
                .expect("合法前端注解");
        let code = render_frontend(&frontend, "ClBackend");
        assert!(
            code.contains("impl crate::witness::Language for ClBackend"),
            "{code}"
        );
        assert!(
            code.contains("crate::witness::Emit::Opencl"),
            "语言名要机械映射到变体，不能另抄一张表：{code}"
        );
        assert!(code.contains("const EXT: &'static str = \"cl\";"), "{code}");
        assert!(
            code.contains("crate::witness::same_text"),
            "`ext` 必须与 `Emit::ext()` 在编译期比对：{code}"
        );
        assert!(
            code.contains("TAG.is_empty()"),
            "空标签会让所有程序挤进同一组缓存条目：{code}"
        );
        assert!(
            code.contains("mod __frontend_check_ClBackend"),
            "每个调用点都要核对注解里的名字与词汇表变体：{code}"
        );
        assert!(
            code.contains(r#"crate::witness::Emit::parse("opencl")"#),
            "绑定必须走词汇表的 `parse`，而不是注解自己再抄一张映射表：{code}"
        );
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
            code.contains(r#"::llm_op::model::DType::parse("f32")"#),
            "注解里的 `dtype` 必须解析成 `MODEL` 用的那个变体：{code}"
        );
        assert!(
            !code.contains("allow("),
            "生成的模块名是固定标识符，不需要豁免任何 lint：{code}"
        );
    }

    #[test]
    fn every_dtype_name_maps_to_the_expected_variant() {
        for (name, variant) in [("f16", "F16"), ("bf16", "Bf16"), ("f32", "F32")] {
            assert_eq!(dtype_variant(name), variant, "{name}");
        }
    }

    #[test]
    fn the_generated_model_code_has_balanced_braces() {
        let code = render_model(&model(r#"name = "q", shape = [8, 4], dtype = "f16""#), "Q");
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
}

//! 渲染：把 [`Declaration`] 渲染成 `impl nn::Contract` 与一条校验测试。
//!
//! 这里只做字符串拼接、不维护任何状态，也不接触 `proc_macro`，因此可以用纯文本断言测试。
//! 生成的代码是确定的三块（被标注项由调用方原样拼接）：
//!
//! 1. 一个私有构造函数 `__layer_spec()`，用 `OnceLock` 保证 `ContractSpec` 只构造一次；
//! 2. 一份 `impl ::nn::Contract`，只把三个访问器转发给它；
//! 3. 一个 `#[cfg(test)]` 模块，在 `cargo test` 时校验"注解里的算子仍然存在"。
//!
//! 生成的名字一律带 `__layer_` 前缀：宏展开不该占用一个用户可能想用的普通名字
//! （`declaration` 这类名字太容易撞车，撞了之后报错会指向用户自己的方法）。

use crate::syntax;
use crate::syntax::{Declaration, ModelDecl, RuntimeDecl};

/// 渲染实现；`ty` 必须是 [`crate::syntax::target_name`] 给出的类型名。
pub(crate) fn render(declaration: &Declaration, ty: &str) -> String {
    let mut code = String::new();
    render_spec(&mut code, declaration, ty);
    render_layer(&mut code, declaration, ty);
    render_layer_check(&mut code, declaration, ty);
    code
}

/// 第一块：私有构造函数 `__layer_spec()`，`OnceLock` 保证 `ContractSpec` 只构造一次。
fn render_spec(code: &mut String, declaration: &Declaration, ty: &str) {
    let name = literal(&declaration.name);
    let shape = shape(&declaration.out);
    let ops = declaration
        .ops
        .iter()
        .map(|op| format!("{}.to_string()", literal(op)))
        .collect::<Vec<_>>()
        .join(", ");

    code.push_str("#[automatically_derived]\n");
    code.push_str(&format!("impl {ty} {{\n"));
    code.push_str("    #[doc(hidden)]\n");
    code.push_str("    fn __layer_spec() -> &'static ::nn::ContractSpec {\n");
    code.push_str(
        "        static SPEC: ::std::sync::OnceLock<::nn::ContractSpec> = ::std::sync::OnceLock::new();\n",
    );
    code.push_str("        SPEC.get_or_init(|| {\n");
    code.push_str(&format!(
        "            let declaration = ::nn::ContractSpec::new({name}, {shape}, ::std::vec![{ops}]);\n"
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
}

/// 第二块：`impl nn::Contract`，只把三个访问器转发给 [`render_spec`] 生成的那份 `ContractSpec`。
fn render_layer(code: &mut String, declaration: &Declaration, ty: &str) {
    code.push_str("#[automatically_derived]\n");
    code.push_str(&format!("impl ::nn::Contract for {ty} {{\n"));
    code.push_str("    fn name(&self) -> &str {\n        <Self>::__layer_spec().name()\n    }\n");
    code.push_str(
        "    fn output(&self) -> &::nn::Shape {\n        <Self>::__layer_spec().output()\n    }\n",
    );
    code.push_str(
        "    fn spca(&self) -> &::nn::ContractSpec {\n        <Self>::__layer_spec()\n    }\n",
    );
    // `in` 给了规约轴宽度，下界就有得算；没给就沿用契约里的默认实现（`Ok(None)`），
    // 而不是生成一个“返回空值”的方法把“这一层不懂自己的形状”藏起来。
    if let Some(width) = declaration.input {
        let out = declaration.out[0];
        code.push_str("    fn bounds(&self) -> ::nn::BoundsResult {\n");
        code.push_str(&format!(
            "        ::nn::Bounds::of_ops(<Self>::__layer_spec().ops(), {width}usize, {out}usize).map(::core::option::Option::Some)\n"
        ));
        code.push_str("    }\n");
    }
    code.push_str("}\n\n");
}

/// 第三块：`#[cfg(test)]` 校验模块，在 `cargo test` 时核对"注解里的算子仍然存在"。
fn render_layer_check(code: &mut String, declaration: &Declaration, ty: &str) {
    let name = literal(&declaration.name);
    code.push_str("#[cfg(test)]\n#[allow(non_snake_case)]\n");
    code.push_str(&format!("mod __layer_check_{ty} {{\n"));
    code.push_str("    #[test]\n    fn the_annotation_still_matches_the_declared_operators() {\n");
    code.push_str(&format!(
        "        let spec = <super::{ty}>::__layer_spec();\n"
    ));
    code.push_str(&format!("        assert_eq!(spec.name(), {name});\n"));
    code.push_str(&format!(
        "        assert_eq!(spec.output().dims(), &{:?});\n",
        declaration.out
    ));
    code.push_str(&format!(
        "        assert_eq!(spec.ops(), &{:?});\n",
        declaration.ops
    ));
    code.push_str("    }\n}\n");
}

/// 运行时形态只生成一个构造宏：`spec` 字段由适配层注入，`impl ::nn::Contract` 的三件访问器
/// 仍由 `spec_accessors!` 从 `self.spec` 读取，展开与静态形态逐字相同。
///
/// 生成宏而不是函数，是因为错误类型 `diagnostic::Error` 不在注解的依赖面里：宏展开到调用点，
/// `?` 直接对着被标注 crate 自己的构造函数签名推断，生成的代码只指向 `::nn` 与 `::std`。
pub(crate) fn render_runtime(declaration: &RuntimeDecl, ty: &str) -> String {
    let constructor = if declaration.role.is_some() {
        "new_with_role"
    } else {
        "new"
    };
    let mut code = format!(
        "#[doc(hidden)]\n#[allow(non_snake_case, unused_macros)]\nmacro_rules! __layer_spec_{ty} {{\n    ({}) => {{\n        ::nn::ContractSpec::{constructor}(\n            ${},\n            {},\n            {},\n",
        runtime_params(declaration),
        declaration.name,
        runtime_shape(&declaration.out),
        runtime_ops(&declaration.ops),
    );
    if let Some(role) = &declaration.role {
        code.push_str(&format!("            ${role},\n"));
    }
    code.push_str("        )\n    };\n}\n");
    code
}

/// 构造宏的参数表：名字、轴宽、算子、角色，按首次出现去重；同一个轴宽可以出现在多根轴上。
fn runtime_params(declaration: &RuntimeDecl) -> String {
    let holes = std::iter::once(&declaration.name)
        .chain(declaration.out.iter())
        .chain(declaration.ops.iter())
        .chain(declaration.role.iter());
    unique(holes)
        .into_iter()
        .map(|hole| format!("${hole}:expr"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// 去重且保持首次出现顺序：`out = [$dim, $dim]` 只占一个宏参数，但在形状里写两次。
fn unique<'a>(names: impl Iterator<Item = &'a String>) -> Vec<&'a str> {
    let mut seen: Vec<&str> = Vec::new();
    for name in names {
        if !seen.contains(&name.as_str()) {
            seen.push(name);
        }
    }
    seen
}

/// 输出形状：空形状与静态形态同一条写法（`[]` 推不出元素类型，显式给 `Vec<usize>`）。
fn runtime_shape(axes: &[String]) -> String {
    if axes.is_empty() {
        return "::nn::Shape::new(::std::vec::Vec::<usize>::new())".to_string();
    }
    let axes = axes
        .iter()
        .map(|axis| format!("${axis}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("::nn::Shape::new([{axes}])")
}

/// 算子列表：每个洞在调用点求值一次，再转成 `ContractSpec` 要求的 `Vec<String>`。
fn runtime_ops(ops: &[String]) -> String {
    let ops = ops
        .iter()
        .map(|op| format!("${op}.to_string()"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("::std::vec![{ops}]")
}

/// 把 `#[model(..)]` 渲染成 `MODEL` + 编译期契约断言 + 一条校验测试。
///
/// `MODEL` 不再是逐字段拼出来的第二份事实：它就是
/// [`Operator`](llm_op::operator::Operator) 的投影（`model_of::<T>()`）。注解里的注册名、
/// 生产形状与输出类型仍要写，但只在编译期与 `Operator` 的常量对齐——写错是 `cargo check`
/// 失败，而不是生成一份绑错算子的表。
///
/// 三条断言：
/// 1. `Operator::NAME` 与注解注册名逐字节相同；
/// 2. `Operator::SHAPE` 与注解生产形状相同，且 `M` 是 `Operator::GROUP` 的整数倍；
/// 3. `gemv` 骨架额外要求在别名上求值 `Format::CHECK`（格式自证）。
///
/// 生成的测试再把注册表、`.k` 签名与注解的 `dtype` 字符串对齐。
pub(crate) fn render_model(declaration: &ModelDecl, ty: &str) -> String {
    if !syntax::DTYPES.contains(&declaration.dtype.as_str()) {
        return format!(
            "compile_error!(\"注解层内部错误：输出类型 `{}` 不在 DType 闭集内\");\n",
            declaration.dtype
        );
    }
    if !syntax::SKELETONS.contains(&declaration.skeleton.as_str()) {
        return format!(
            "compile_error!(\"注解层内部错误：骨架 `{}` 不在闭集内\");\n",
            declaration.skeleton
        );
    }

    let [m, n] = declaration.shape;
    let mut code = String::new();
    code.push_str(&format!(
        "#[doc = \"`{ty}` 的语义表（`#[model]` 生成）：`Operator` 的投影。\"]\n"
    ));
    code.push_str(&format!(
        "pub const {}: ::llm_op::model::Model = ::llm_op::operator::model_of::<{ty}>();\n\n",
        declaration.const_name
    ));
    render_contract_asserts(&mut code, declaration, ty, (m, n));
    render_model_check(&mut code, declaration, ty);
    code
}

/// 注解字面量与 [`Operator`](llm_op::operator::Operator) 常量之间的编译期契约。
fn render_contract_asserts(
    code: &mut String,
    declaration: &ModelDecl,
    ty: &str,
    dims: (usize, usize),
) {
    let (m, n) = dims;
    let name = literal(&declaration.name);
    code.push_str(&format!(
        "const _: () = assert!(::llm_op::same_bytes(<{ty} as ::llm_op::operator::Operator>::NAME, {name}), \"注解注册名必须与 Operator::NAME 一致\");\n\n"
    ));
    code.push_str(&format!(
        "const _: () = {{\n    assert!(<{ty} as ::llm_op::operator::Operator>::SHAPE.0 == {m}usize, \"注解生产形状必须与 Operator::SHAPE 一致\");\n    assert!(<{ty} as ::llm_op::operator::Operator>::SHAPE.1 == {n}usize, \"注解生产形状必须与 Operator::SHAPE 一致\");\n    let group = <{ty} as ::llm_op::operator::Operator>::GROUP as usize;\n    assert!(group > 0, \"形状粒度必须大于 0\");\n    assert!({m}usize / group * group == {m}usize, \"生产形状的 M 必须是形状粒度的整数倍\");\n}};\n\n"
    ));
    if declaration.skeleton == "gemv" {
        code.push_str(&format!(
            "const _: () = <{ty} as ::llm_op::gemv::Format>::CHECK;\n\n"
        ));
    }
}

/// `#[cfg(test)] mod __model_check_*`：把注解、注册表与 `.k` 签名三份事实对齐。模块名带
/// `const` 名，多算子同模块时两个检查各占一个测试模块，不撞名。
fn render_model_check(code: &mut String, declaration: &ModelDecl, ty: &str) {
    let check = format!(
        "__model_check_{}",
        declaration.const_name.to_ascii_lowercase()
    );
    code.push_str("#[cfg(test)]\n");
    code.push_str(&format!("mod {check} {{\n"));
    code.push_str("    #[test]\n");
    code.push_str("    fn the_annotation_agrees_with_the_registered_kernel() {\n");
    render_registry_check(code, declaration, ty);
    render_dtype_check(code, declaration);
    code.push_str("    }\n}\n");
}

/// 注册表 ↔ 注解声明 ↔ `Operator` 投影：名字、形状、样本与代价。
fn render_registry_check(code: &mut String, declaration: &ModelDecl, ty: &str) {
    let name = literal(&declaration.name);
    let model = &declaration.const_name;
    let [m, n] = declaration.shape;
    code.push_str(&format!(
        "        let op = ::llm_op::builtin_named({name}).expect(\"算子已注册\");\n"
    ));
    code.push_str(&format!(
        "        let signature = ::llm_op::sig::Signature::parse({name}, op.template()).expect(\"签名可解析\");\n"
    ));
    code.push_str(&format!(
        "        let registered = ::llm_op::model({name}).expect(\"语义已注册\");\n"
    ));
    // 注解的注册名与生产形状属于**声明**；`MODEL` 现在由 `Operator` 投影，所以这两条把声明
    // 与投影绑在一起，防止只改一侧。
    code.push_str(&format!(
        "        assert_eq!(<super::{ty} as ::llm_op::operator::Operator>::NAME, {name}, \"注解注册名必须与 Operator::NAME 一致\");\n"
    ));
    code.push_str(&format!(
        "        assert_eq!(super::{model}.shape, ({m}usize, {n}usize), \"注解生产形状必须与 Operator::SHAPE 一致\");\n"
    ));
    code.push_str(&format!(
        "        assert_eq!(registered.group, super::{model}.group, \"注册表必须指向这份语义\");\n",
    ));
    code.push_str(&format!(
        "        assert_eq!(registered.shape, super::{model}.shape, \"注册表必须指向这份语义\");\n",
    ));
    code.push_str(&format!(
        "        assert_eq!(registered.dtype, super::{model}.dtype, \"注册表必须指向这份语义\");\n",
    ));
    code.push_str(
        "        let sample = (registered.sample)(registered.shape.0, registered.shape.1);\n",
    );
    code.push_str("        signature.check(&sample).expect(\"样本与 .k 签名一致\");\n");
    code.push_str(
        "        assert_eq!((registered.sample)(registered.shape.0, registered.shape.1), sample, \"样本必须可复现\");\n",
    );
    code.push_str(
        "        let shape = (registered.shape_ctx)(registered.shape.0, registered.shape.1).expect(\"形状上下文可构造\");\n",
    );
    code.push_str("        (registered.cost)(&shape).expect(\"生产形状必须能算出代价\");\n");
}

/// 注解的 `dtype` 字符串 ↔ `MODEL.dtype` 变体 ↔ `.k` 的可写输出类型。
fn render_dtype_check(code: &mut String, declaration: &ModelDecl) {
    let dtype_name = literal(&declaration.dtype);
    let model = &declaration.const_name;
    // 注解里的 `dtype` 是**字符串**，`MODEL.dtype` 是**变体**：这一句就是两者之间的绑定点。
    // 词汇表改名、或生成代码挑错变体，都会在这里失败，而不是等到 `verify` 对一个错宽度的
    // 缓冲对拍。全量与映射规则由 `tests/contracts.rs` 统一核对。
    code.push_str("        assert_eq!(\n");
    code.push_str(&format!(
        "            ::llm_op::model::DType::parse_name({dtype_name}),\n"
    ));
    code.push_str(&format!("            Some(super::{model}.dtype),\n",));
    code.push_str(
        "            \"注解里的 `dtype` 必须解析成 `MODEL` 用的那个变体（词汇表是唯一名字源）\",\n",
    );
    code.push_str("        );\n");
    // 注解的 `dtype` 与 `.k` 的可写输出是同一件事的两种说法：前者说"回读按什么解释"，
    // 后者说"内核写的是什么"。只剩一半时（`.k` 写 `u8 *y` 而注解写 f32，或反过来）
    // 对拍会按另一种宽度解读缓冲，所以两者必须是同一个类型。`Model::dtype` 的阈值与
    // `harness` 的回读都取自它，这是它们在装载期唯一的交叉点。
    code.push_str("        assert!(\n");
    code.push_str(&format!(
        "            signature.output_type().and_then(|ctype| ctype.dtype()) == Some(super::{model}.dtype),\n",
    ));
    code.push_str(&format!(
        "            \"注解写 `dtype = {}`，但 `.k` 的可写输出是 `{{}}`：回读按 `MODEL.dtype` 解释，两者必须是同一个类型\",\n",
        declaration.dtype,
    ));
    code.push_str("            signature.output_type().map_or(\"-\", |ctype| ctype.c_name()),\n");
    code.push_str("        );\n");
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
#[path = "render_tests.rs"]
mod tests;

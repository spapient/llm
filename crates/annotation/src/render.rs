//! 渲染：把一份声明拼成源码。只做字符串拼接、不维护状态、不接触 `proc_macro`，因此可以用纯文本断言测试。
//!
//! 三块输出共用一条规则：**生成的名字一律带 `__layer_` / `__model_` 前缀**，宏展开不该占用用户可能
//! 想用的普通名字。拼接只往一个预留了容量的 `String` 里写（`write!` 而非 `format!` + 拷贝），
//! 于是"渲染一份"是一次分配。

use crate::decl::{DTYPES, Declaration, ModelDecl, RuntimeDecl, SKELETONS};
use std::fmt::Write;

/// 静态形态：私有构造函数 + `impl ::nn::Contract` + 一条 `#[cfg(test)]` 校验。
pub(crate) fn render(declaration: &Declaration, ty: &str) -> String {
    let mut code = String::with_capacity(1536);
    spec(&mut code, declaration, ty);
    contract(&mut code, declaration, ty);
    check(&mut code, declaration, ty);
    code
}

/// 第一块：`__layer_spec()` 用 `OnceLock` 保证 `ContractSpec` 只构造一次。
fn spec(code: &mut String, declaration: &Declaration, ty: &str) {
    code.push_str("#[automatically_derived]\n");
    let _ = writeln!(code, "impl {ty} {{");
    code.push_str("    #[doc(hidden)]\n");
    code.push_str("    fn __layer_spec() -> &'static ::nn::ContractSpec {\n");
    code.push_str(CONST);
    code.push('\n');
    code.push_str("        SPEC.get_or_init(|| {\n");
    code.push_str("            let declaration = ::nn::ContractSpec::new(");
    literal(code, &declaration.name);
    code.push_str(", ");
    shape_of(code, &declaration.out, |code, dim| {
        let _ = write!(code, "{dim}usize");
    });
    code.push_str(", ::std::vec![");
    comma(&mut code, declaration.ops.iter(), |code, op| {
        literal(code, op);
        code.push_str(".to_string()");
    });
    code.push_str("]);\n");
    code.push_str("            match declaration {\n");
    code.push_str("                ::core::result::Result::Ok(spec) => spec,\n");
    code.push_str(
        "                ::core::result::Result::Err(error) => { ::std::panic!(\"注解声明的层 `{}` 引用了未声明的算子：{}\", ",
    );
    literal(code, &declaration.name);
    code.push_str(", error) }\n");
    code.push_str("            }\n        })\n    }\n}\n\n");
}

/// 第二块：三个访问器只转发给 [`spec`] 生成的那份 `ContractSpec`。
fn contract(code: &mut String, declaration: &Declaration, ty: &str) {
    let _ = writeln!(
        code,
        "#[automatically_derived]\nimpl ::nn::Contract for {ty} {{"
    );
    code.push_str("    fn name(&self) -> &str {\n        <Self>::__layer_spec().name()\n    }\n");
    code.push_str(
        "    fn output(&self) -> &::nn::Shape {\n        <Self>::__layer_spec().output()\n    }\n",
    );
    code.push_str(
        "    fn spca(&self) -> &::nn::ContractSpec {\n        <Self>::__layer_spec()\n    }\n",
    );
    // `in` 给了规约轴宽度，下界就有得算；没给就不生成该方法，沿用契约里的默认实现（`Ok(None)`），
    // 而不是生成一个"返回空值"的方法把"这一层不懂自己的形状"藏起来。
    if let Some(width) = declaration.input {
        let out = declaration.out[0];
        code.push_str("    fn bounds(&self) -> ::nn::BoundsResult {\n        ");
        let _ = writeln!(
            code,
            "::nn::Bounds::of_ops(<Self>::__layer_spec().ops(), {width}usize, {out}usize).map(::core::option::Option::Some)"
        );
        code.push_str("    }\n");
    }
    code.push_str("}\n\n");
}

/// 第三块：`#[cfg(test)]` 校验模块，核对"注解里的算子仍然存在"。
fn check(code: &mut String, declaration: &Declaration, ty: &str) {
    code.push_str("#[cfg(test)]\n#[allow(non_snake_case)]\n");
    let _ = writeln!(code, "mod __layer_check_{ty} {{");
    code.push_str("    #[test]\n    fn the_annotation_still_matches_the_declared_operators() {\n");
    let _ = writeln!(code, "        let spec = <super::{ty}>::__layer_spec();");
    code.push_str("        assert_eq!(spec.name(), ");
    literal(code, &declaration.name);
    code.push_str(");\n");
    let _ = writeln!(
        code,
        "        assert_eq!(spec.output().dims(), &{:?});",
        declaration.out
    );
    let _ = writeln!(
        code,
        "        assert_eq!(spec.ops(), &{:?});",
        declaration.ops
    );
    code.push_str("    }\n}\n");
}

/// 运行时形态只生成一个构造宏：`spec` 字段由适配层注入，`impl ::nn::Contract` 仍由
/// `spec_accessors!` 从 `self.spec` 读取，与静态形态逐字相同。
///
/// 生成宏而不是函数，是因为错误类型 `diagnostic::Error` 不在注解的依赖面里：宏展开到调用点，
/// `?` 对着被标注 crate 自己的构造函数签名推断，生成的代码只指向 `::nn` 与 `::std`。
pub(crate) fn render_runtime(declaration: &RuntimeDecl, ty: &str) -> String {
    let mut code = String::with_capacity(512);
    code.push_str("#[doc(hidden)]\n#[allow(non_snake_case, unused_macros)]\n");
    let _ = writeln!(code, "macro_rules! __layer_spec_{ty} {{");
    code.push_str("    (");
    comma(&mut code, unique(holes(declaration)), |code, hole| {
        let _ = write!(code, "${hole}:expr");
    });
    code.push_str(") => {\n");
    let constructor = match declaration.role {
        Some(_) => "new_with_role",
        None => "new",
    };
    let _ = writeln!(code, "        ::nn::ContractSpec::{constructor}(");
    let _ = writeln!(code, "            ${},", declaration.name);
    code.push_str("            ");
    shape_of(&mut code, &declaration.out, |code, axis| {
        let _ = write!(code, "${axis}");
    });
    code.push_str(",\n");
    code.push_str("            ::std::vec![");
    comma(&mut code, declaration.ops.iter(), |code, op| {
        let _ = write!(code, "${op}.to_string()");
    });
    code.push_str("],\n");
    if let Some(role) = &declaration.role {
        let _ = writeln!(code, "            ${role},");
    }
    code.push_str("        )\n    };\n}\n");
    code
}

/// 宏参数的空位：名字、轴宽、算子、角色，按首次出现顺序排好，交给 [`unique`] 去重。
fn holes(declaration: &RuntimeDecl) -> impl Iterator<Item = &String> {
    std::iter::once(&declaration.name)
        .chain(declaration.out.iter())
        .chain(declaration.ops.iter())
        .chain(declaration.role.iter())
}

/// 去重且保持首次出现顺序：`out = [$dim, $dim]` 只占一个宏参数，却在形状里写两次。
fn unique<'a>(names: impl Iterator<Item = &'a String>) -> Vec<&'a str> {
    let mut seen: Vec<&str> = Vec::new();
    for name in names {
        if !seen.contains(&name.as_str()) {
            seen.push(name);
        }
    }
    seen
}

/// 把 `#[model(..)]` 渲染成 `MODEL` + 编译期契约断言 + 一条校验测试。
///
/// `MODEL` 不是逐字段拼出来的第二份事实：它就是 `Operator` 的投影（`model_of::<T>()`）。注解里的
/// 注册名、生产形状与输出类型仍要写，但只在编译期与 `Operator` 的常量对齐——写错是 `cargo check`
/// 失败，而不是生成一份绑错算子的表。
pub(crate) fn render_model(declaration: &ModelDecl, ty: &str) -> String {
    if let Some(closure) = closed(declaration) {
        return closure;
    }
    let mut code = String::with_capacity(3072);
    let _ = writeln!(
        code,
        "#[doc = \"`{ty}` 的语义表（`#[model]` 生成）：`Operator` 的投影。\"]"
    );
    let _ = writeln!(
        code,
        "pub const {}: ::llm_op::model::Model = ::llm_op::operator::model_of::<{ty}>();\n",
        declaration.const_name
    );
    asserts(&mut code, declaration, ty);
    model_check(&mut code, declaration, ty);
    code
}

/// 闭集被绕过时（手写构造了一个未映射的 `dtype` / 未列出的骨架）生成一条 `compile_error!`：
/// 让注解层自己先红，而不是生成一份指向不存在模块的代码。
fn closed(declaration: &ModelDecl) -> Option<String> {
    let message = match (
        DTYPES.contains(&declaration.dtype.as_str()),
        SKELETONS.contains(&declaration.skeleton.as_str()),
    ) {
        (true, true) => return None,
        (false, _) => format!(
            "注解层内部错误：`{}` 不是可对拍的 DType（可用：{}）",
            declaration.dtype,
            DTYPES.join("、")
        ),
        (_, false) => format!(
            "注解层内部错误：骨架 `{}` 不在闭集内（可用：{}）",
            declaration.skeleton,
            SKELETONS.join("、")
        ),
    };
    Some(format!("compile_error!({});\n", Quoted(&message)))
}

/// 注解字面量与 `Operator` 常量之间的编译期契约：名字逐字节相同、形状相同、`M` 是粒度整数倍。
fn asserts(code: &mut String, declaration: &ModelDecl, ty: &str) {
    let [m, n] = declaration.shape;
    let _ = writeln!(
        code,
        "const _: () = assert!(::llm_op::same_bytes(<{ty} as ::llm_op::operator::Operator>::NAME, {}), \"注解注册名必须与 Operator::NAME 一致\");",
        Quoted(&declaration.name)
    );
    code.push_str("const _: () = {\n    ");
    let _ = writeln!(
        code,
        "assert!(<{ty} as ::llm_op::operator::Operator>::SHAPE.0 == {m}usize, \"注解生产形状必须与 Operator::SHAPE 一致\");"
    );
    let _ = writeln!(
        code,
        "    assert!(<{ty} as ::llm_op::operator::Operator>::SHAPE.1 == {n}usize, \"注解生产形状必须与 Operator::SHAPE 一致\");"
    );
    let _ = writeln!(
        code,
        "    let group = <{ty} as ::llm_op::operator::Operator>::GROUP as usize;"
    );
    code.push_str("    assert!(group > 0, \"形状粒度必须大于 0\");\n");
    let _ = writeln!(
        code,
        "    assert!({m}usize / group * group == {m}usize, \"生产形状的 M 必须是形状粒度的整数倍\");\n}};\n"
    );
    if declaration.skeleton == "gemv" {
        let _ = writeln!(
            code,
            "const _: () = <{ty} as ::llm_op::gemv::Format>::CHECK;\n"
        );
    }
}

/// `#[cfg(test)] mod __model_check_*`：把注解、注册表与 `.k` 签名三份事实对齐。模块名带 `const`
/// 名，多算子同模块时两个检查各占一个测试模块，不撞名。
fn model_check(code: &mut String, declaration: &ModelDecl, ty: &str) {
    let check = declaration.const_name.to_ascii_lowercase();
    code.push_str("#[cfg(test)]\n");
    let _ = writeln!(code, "mod __model_check_{check} {{");
    code.push_str("    #[test]\n    fn the_annotation_agrees_with_the_registered_kernel() {\n");
    registry(code, declaration, ty);
    dtype(code, declaration);
    code.push_str("    }\n}\n");
}

/// 注册表 ↔ 注解声明 ↔ `Operator` 投影：名字、形状、样本与代价。
fn registry(code: &mut String, declaration: &ModelDecl, ty: &str) {
    let model = &declaration.const_name;
    let [m, n] = declaration.shape;
    let _ = writeln!(
        code,
        "        let op = ::llm_op::builtin_named({}).expect(\"算子已注册\");",
        Quoted(&declaration.name)
    );
    let _ = writeln!(
        code,
        "        let signature = ::llm_op::sig::Signature::parse({}, op.template()).expect(\"签名可解析\");",
        Quoted(&declaration.name)
    );
    let _ = writeln!(
        code,
        "        let registered = ::llm_op::model({}).expect(\"语义已注册\");",
        Quoted(&declaration.name)
    );
    // 注册名与生产形状属于**声明**；`MODEL` 由 `Operator` 投影，所以这两条把声明与投影绑在一起，
    // 防止只改一侧。
    let _ = writeln!(
        code,
        "        assert_eq!(<super::{ty} as ::llm_op::operator::Operator>::NAME, {}), \"注解注册名必须与 Operator::NAME 一致\");",
        Quoted(&declaration.name)
    );
    let _ = writeln!(
        code,
        "        assert_eq!(super::{model}.shape, ({m}usize, {n}usize), \"注解生产形状必须与 Operator::SHAPE 一致\");"
    );
    for field in ["group", "shape", "dtype"] {
        let _ = writeln!(
            code,
            "        assert_eq!(registered.{field}, super::{model}.{field}, \"注册表必须指向这份语义\");"
        );
    }
    code.push_str(
        "        let sample = (registered.sample)(registered.shape.0, registered.shape.1);\n",
    );
    code.push_str("        signature.check(&sample).expect(\"样本与 .k 签名一致\");\n");
    code.push_str("        assert_eq!((registered.sample)(registered.shape.0, registered.shape.1), sample, \"样本必须可复现\");\n");
    code.push_str("        let shape = (registered.shape_ctx)(registered.shape.0, registered.shape.1).expect(\"形状上下文可构造\");\n");
    code.push_str("        (registered.cost)(&shape).expect(\"生产形状必须能算出代价\");\n");
}

/// 注解的 `dtype` 字符串 ↔ `MODEL.dtype` 变体 ↔ `.k` 的可写输出类型：三者是同一件事的三种说法。
/// 只剩一半时（`.k` 写 `u8 *y` 而注解写 f32，或反过来）对拍会按另一种宽度解读缓冲。
fn dtype(code: &mut String, declaration: &ModelDecl) {
    let model = &declaration.const_name;
    code.push_str("        assert_eq!(\n");
    let _ = writeln!(
        code,
        "            ::llm_op::model::DType::parse_name({}),\n            Some(super::{model}.dtype),\n            \"注解里的 `dtype` 必须解析成 `MODEL` 用的那个变体（词汇表是唯一名字源）\",\n        );",
        Quoted(&declaration.dtype)
    );
    code.push_str("        assert!(\n");
    let _ = writeln!(
        code,
        "            signature.output_type().and_then(|ctype| ctype.dtype()) == Some(super::{model}.dtype),"
    );
    let _ = writeln!(
        code,
        "            \"注解写 `dtype = {}`，但 `.k` 的可写输出是 `{{}}`：回读按 `MODEL.dtype` 解释，两者必须是同一个类型\",",
        declaration.dtype
    );
    code.push_str(
        "            signature.output_type().map_or(\"-\", |ctype| ctype.c_name()),\n        );\n",
    );
}

/// `__layer_spec()` 里那行 `OnceLock` 静态量；单独提出来只为让 [`spec`] 的 nloc 留在门禁内。
const CONST: &str = "        static SPEC: ::std::sync::OnceLock<::nn::ContractSpec> = ::std::sync::OnceLock::new();";

/// 形状表达式：空形状必须显式给 `Vec<usize>`——`Shape::new([])` 推断不出元素类型。
fn shape_of<T>(code: &mut String, axes: &[T], each: impl FnMut(&mut String, &T)) {
    match axes {
        [] => code.push_str("::nn::Shape::new(::std::vec::Vec::<usize>::new())"),
        _ => {
            code.push_str("::nn::Shape::new([");
            comma(code, axes.iter(), each);
            code.push_str("])");
        }
    }
}

/// 逗号连接：调用点只描述"每一项长什么样"，中间不留一层 `Vec<String>`。
fn comma<T>(code: &mut String, items: impl Iterator<Item = T>, mut each: impl FnMut(&mut String, T)) {
    for (index, item) in items.enumerate() {
        if index > 0 {
            code.push_str(", ");
        }
        each(code, item);
    }
}

/// 借用视图：`Display` 即 `{:?}`，把字面量零分配地嵌进 `write!` 的格式串。
#[derive(Clone, Copy)]
struct Quoted<'a>(&'a str);

impl std::fmt::Display for Quoted<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}", self.0)
    }
}

/// Rust 字符串字面量。解析层已拒绝转义序列，因此 `{:?}` 的输出与输入逐字相同。
fn literal(code: &mut String, text: &str) {
    let _ = write!(code, "{}", Quoted(text));
}

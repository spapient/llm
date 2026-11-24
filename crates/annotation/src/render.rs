//! 渲染：把一份声明拼成源码。只做字符串拼接、不维护状态、不接触 `proc_macro`，因此可以用纯文本断言测试。

use crate::decl::{DTYPES, Declaration, ModelDecl, ParallelDecl, RuntimeDecl, SKELETONS};
use std::fmt::Write;

/// 三块输出的预留容量：按最长的真实声明留足余量，于是"渲染一次 = 一次分配"真的成立。
pub(crate) const STATIC_ROOM: usize = 3072;
pub(crate) const MODEL_ROOM: usize = 1536;
pub(crate) const RUNTIME_ROOM: usize = 512;

/// 静态形态：私有构造函数 + `impl ::nn::Contract` + 一条 `#[cfg(test)]` 校验。
pub(crate) fn render(declaration: &Declaration, ty: &str) -> String {
    let mut code = String::with_capacity(STATIC_ROOM);
    spec(&mut code, declaration, ty);
    contract(&mut code, declaration, ty);
    check_module(&mut code, declaration, ty);
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
    comma(code, declaration.ops.iter(), |code, op| {
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
fn check_module(code: &mut String, declaration: &Declaration, ty: &str) {
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
pub(crate) fn render_runtime(declaration: &RuntimeDecl, ty: &str) -> String {
    let mut code = String::with_capacity(RUNTIME_ROOM);
    code.push_str("#[doc(hidden)]\n#[allow(non_snake_case, unused_macros)]\n");
    let _ = writeln!(code, "macro_rules! __layer_spec_{ty} {{");
    code.push_str("    (");
    comma(
        &mut code,
        unique(holes(declaration)).into_iter(),
        |code, hole| {
            let _ = write!(code, "${hole}:expr");
        },
    );
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
    let mut seen: Vec<&str> = Vec::with_capacity(names.size_hint().1.unwrap_or(0));
    for name in names {
        if !seen.contains(&name.as_str()) {
            seen.push(name);
        }
    }
    seen
}

/// 把 `#[model(..)]` 渲染成 `MODEL` + 编译期契约断言（不生成运行期测试，见 [`asserts`]）。
pub(crate) fn render_model(declaration: &ModelDecl, ty: &str) -> String {
    if let Some(closure) = closed(declaration) {
        return closure;
    }
    let mut code = String::with_capacity(MODEL_ROOM);
    let _ = writeln!(
        code,
        "#[doc = \"`{ty}` 的语义表（`#[model]` 生成）：`Operator` 的投影。\"]"
    );
    let _ = writeln!(
        code,
        "pub const {}: ::prof::model::Model = ::prof::operator::model_of::<{ty}>();\n",
        declaration.const_name
    );
    asserts(&mut code, declaration, ty);
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

/// 注解字面量与 `Operator` 常量之间的编译期契约：名字与输出类型逐字节相同、形状相同、`M` 是
/// 粒度整数倍。
fn asserts(code: &mut String, declaration: &ModelDecl, ty: &str) {
    let [m, n] = declaration.shape;
    let _ = writeln!(
        code,
        "const _: () = assert!(::prof::same_bytes(<{ty} as ::prof::operator::Operator>::NAME, {}), \"注解注册名必须与 Operator::NAME 一致\");",
        Quoted(&declaration.name)
    );
    let _ = writeln!(
        code,
        "const _: () = assert!(::prof::same_bytes(<{ty} as ::prof::operator::Operator>::DTYPE.name(), {}), \"注解里的 `dtype` 必须与 Operator::DTYPE 一致\");",
        Quoted(&declaration.dtype)
    );
    code.push_str("const _: () = {\n    ");
    let _ = writeln!(
        code,
        "assert!(<{ty} as ::prof::operator::Operator>::SHAPE.0 == {m}usize, \"注解生产形状必须与 Operator::SHAPE 一致\");"
    );
    let _ = writeln!(
        code,
        "    assert!(<{ty} as ::prof::operator::Operator>::SHAPE.1 == {n}usize, \"注解生产形状必须与 Operator::SHAPE 一致\");"
    );
    let _ = writeln!(
        code,
        "    let group = <{ty} as ::prof::operator::Operator>::GROUP as usize;"
    );
    code.push_str("    assert!(group > 0, \"形状粒度必须大于 0\");\n");
    let _ = writeln!(
        code,
        "    assert!({m}usize / group * group == {m}usize, \"生产形状的 M 必须是形状粒度的整数倍\");\n}};\n"
    );
}

/// `__layer_spec()` 里那行 `OnceLock` 静态量；单独提出来只为让 [`spec`] 的 nloc 留在门禁内。
const CONST: &str = "        static SPEC: ::std::sync::OnceLock<::nn::ContractSpec> = ::std::sync::OnceLock::new();";

/// 形状表达式：空形状必须显式给 `Vec<usize>`——`Shape::new([])` 推断不出元素类型。
fn shape_of<T>(code: &mut String, axes: &[T], each: impl FnMut(&mut String, &T)) {
    if axes.is_empty() { code.push_str("::nn::Shape::new(::std::vec::Vec::<usize>::new())") } else {
        code.push_str("::nn::Shape::new([");
        comma(code, axes.iter(), each);
        code.push_str("])");
    }
}

/// 逗号连接：调用点只描述"每一项长什么样"，中间不留一层 `Vec<String>`。
fn comma<T>(
    code: &mut String,
    items: impl Iterator<Item = T>,
    mut each: impl FnMut(&mut String, T),
) {
    for (idx, item) in items.enumerate() {
        if idx > 0 {
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

/// 把 `#[parallel(..)]` 渲染成策略访问器：度数走 `ParallelConfig::new`，口径走三条 `with_*` 覆盖链。
///
/// 注解只声明静态意图；访问器在首次调用时构造 [`ParallelConfig`]，不复制字段表，也不发起通信。
/// 闭集短名在解析层已校验，这里的变体兜底分支不可达。
pub(crate) fn render_parallel(declaration: &ParallelDecl, ty: &str) -> String {
    let reduce = reduce_variant(&declaration.reduce);
    let priority = priority_variant(&declaration.priority);
    let average = if declaration.average { "true" } else { "false" };
    let mut code = String::with_capacity(PARALLEL_ROOM);
    code.push_str("#[automatically_derived]\n");
    let _ = writeln!(code, "impl {ty} {{");
    code.push_str("    #[doc(hidden)]\n    #[allow(dead_code)]\n");
    code.push_str("    fn __parallel_config() -> ::shard::ParallelConfig {\n");
    let _ = writeln!(
        code,
        "        ::shard::ParallelConfig::new({}, {}, {})",
        declaration.tp, declaration.pp, declaration.dp
    );
    code.push_str("            .expect(\"注解度数已在编译期校验非零；此处仅拒绝 u32 上界溢出\")\n");
    code.push_str(&format!(
        "            .with_reduce(::shard::Reduce::{reduce})\n"
    ));
    code.push_str(&format!("            .with_average({average})\n"));
    code.push_str(&format!(
        "            .with_priority(::shard::Priority::{priority})\n"
    ));
    code.push_str("    }\n}\n");
    code
}

/// 归约短名 → `Reduce` 变体名。
fn reduce_variant(short: &str) -> &str {
    match short {
        "max" => "Max",
        "min" => "Min",
        "prod" => "Prod",
        _ => "Sum",
    }
}

/// 优先级短名 → `Priority` 变体名。
fn priority_variant(short: &str) -> &str {
    match short {
        "high" => "High",
        "low" => "Low",
        _ => "Normal",
    }
}

/// 并行策略访问器的预留容量。
pub(crate) const PARALLEL_ROOM: usize = 768;

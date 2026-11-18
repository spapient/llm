//! 代码生成：把 [`Declaration`] 渲染成 `impl nn::Contract` 与一条校验测试。
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

use crate::syntax::{Declaration, ModelDecl};

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

/// `dtype` 名 → 变体名。
///
/// 不再手写「名字 → 变体」表：`dtype` 的 [`DType::variant`] 是变体名的唯一事实源，
/// 缩写词的全大写规则（`bf16 → BF16`）也住在那里。闭集已在文法层（[`crate::syntax::DTYPES`]）
/// 校验过，这里只做名字到变体的查找。
fn dtype_variant(dtype: &str) -> Option<String> {
    dtype::DType::parse_name(dtype).map(|dtype| dtype.variant().to_string())
}

/// 把 `#[model(..)]` 渲染成 `MODEL` + 两条编译期断言 + 一条校验测试。
///
/// 生成的只有一块**数据**：`cost` / `sample` / `reference` 都是 `<类型 as Format>` 的单态化
/// 实例，`group` 直接取 `Format::GRAIN`。注解里的字面量因此只管注册名、生产形状与输出类型，
/// "这份语义是哪份格式"由类型参数保证——别名不实现 `Format` 就编译不过。
///
/// 十二条骨架的 `MODEL` 字段逐字相同，差别只有四件：`op` 模块名、`group` 的写法、入口是否按
/// 类型参数单态化、以及是否附带粒度断言。这四件写在 [`SKELETON_TABLE`] 里，"加一条骨架"
/// 因此是加一行——[`crate::syntax::SKELETONS`] 与 `every_skeleton_has_a_branch` 会一起要求它。
///
/// 两条断言是注解与类型之间的契约，全部在编译期成立，但**只由 `gemv` 骨架生成**——只有它拿
/// 得到 `Format::GRAIN` 这个粒度（其余骨架的粒度写在 `op` 的语义表里，不经过类型参数）：
///
/// 1. 形状粒度必须大于 0——`sample` 按它切标度，0 会让除法失真；
/// 2. 生产形状的 `M` 必须是它的整数倍——量化算子按分组对齐，写错在 `cargo check` 就失败，
///    而不是等 `doctor` 算代价时才报。
///
/// 生成的测试把四份事实对齐：注册表里的语义就是这份 `MODEL`、样本与 `.k` 签名一致、生产形状
/// 能算出代价、注解里的 `dtype` 字符串解析成同一个变体。任何一处漂移都在 `cargo test` 里
/// 失败，而不是在审计报告里表现成一个奇怪的数。
pub(crate) fn render_model(declaration: &ModelDecl, ty: &str) -> String {
    let [m, n] = declaration.shape;

    let Some(dtype) = dtype_variant(&declaration.dtype) else {
        // 与骨架同一类失配：文法层挡在 `DTYPES` 闭集外，走到这里说明闭集加了成员、词汇表没有。
        return format!(
            "compile_error!(\"注解层内部错误：输出类型 `{}` 没有 `DType` 变体\");\n",
            declaration.dtype
        );
    };
    let Some(shape) = skeleton_of(&declaration.skeleton) else {
        // 文法层保证 `skeleton` 在闭集里，所以走到这里只剩一个原因：闭集加了成员、表没加行。
        // 早退回一条编译错误，而不是悄悄挑一个别的骨架——那会生成一份编译得过、却绑错格式的
        // `MODEL`。
        return format!(
            "compile_error!(\"注解层内部错误：骨架 `{}` 没有生成分支\");\n",
            declaration.skeleton
        );
    };

    let mut code = String::new();
    code.push_str(&format!(
        "#[doc = \"`{ty}` 的语义表（`#[model]` 生成）：生产形状、输出类型与三条单态化入口。\"]\n"
    ));
    code.push_str("pub const MODEL: ::llm_op::model::Model = ::llm_op::model::Model {\n");
    render_fields(&mut code, shape, &declaration.name, ty, (m, n), &dtype);
    render_model_check(&mut code, declaration);
    code
}

/// 一条骨架在生成器眼里的全部差异。
///
/// 早先这里是一串 `else if declaration.skeleton() == ".."`，每条各写 8 行逐字相同的字段
/// 列表（`group` / `shape` / `dtype` / `axes` / `shape_ctx` / `cost` / `sample` / `reference`）。
/// 加一条骨架要改三处，而漏改一处的后果不是编译错误，是一份**编译得过、却绑错模型**的
/// `MODEL`——十一段拷贝换不来任何区分度。
struct Skeleton {
    /// 注解里 `skeleton = ".."` 的可选值（[`crate::syntax::SKELETONS`] 的成员）。
    name: &'static str,
    /// `op` 里承载这套语义的模块名；`None` 表示"用注册名"（按算子名各自的模块寻址）。
    module: Option<&'static str>,
    /// `MODEL.group` 的写法，`{ty}` 在渲染时替换成类型参数。
    group: &'static str,
    /// 入口按 `ty` 单态化（`cost::<T>`），还是模块里的具体函数（`module::cost`）。
    generic: bool,
    /// 附带粒度契约的编译期断言（只有拿得到 `Format::GRAIN` 的骨架有）。
    grain: bool,
}

/// 表项构造：`module` 用 `Option` 而不是空串哨兵，与 [`Skeleton`] 的字段名一一对应。
const fn row(
    name: &'static str,
    module: Option<&'static str>,
    group: &'static str,
    generic: bool,
    grain: bool,
) -> Skeleton {
    Skeleton {
        name,
        module,
        group,
        generic,
        grain,
    }
}

/// 十四条骨架的差异表，顺序与 [`crate::syntax::SKELETONS`] 一致。
///
/// `dequant_i4` 的 `group` 是 2、`dequant_i2` 的 `group` 是 4：每 2（或 4）个输出元素占一个
/// 字节，`verify` 从对齐这个约束的形状起步。`gemv` 的 `group` 指向类型参数上的
/// `Format::GRAIN`——粒度是格式的编译期事实，注解里不该再写一个数。
const SKELETON_TABLE: [Skeleton; 14] = [
    row(
        "gemv",
        Some("gemv"),
        "<{ty} as ::llm_op::gemv::Format>::GRAIN as u32",
        true,
        true,
    ),
    row("gemm", Some("gemm"), "1u32", true, false),
    row("elementwise", Some("elementwise"), "1u32", true, false),
    row("unary", Some("unary"), "1u32", true, false),
    row("reduce", Some("reduce"), "1u32", true, false),
    row("rmsnorm", Some("rmsnorm_f32"), "1u32", false, false),
    row("softmax", Some("softmax_f32"), "1u32", false, false),
    row("rope", Some("rope_f32"), "1u32", false, false),
    row("embed", Some("embed_f32"), "1u32", false, false),
    row("dequant", Some("dequant_i8"), "1u32", false, false),
    row("dequant_i4", Some("dequant_i4"), "2u32", false, false),
    row("dequant_i2", Some("dequant_i2"), "4u32", false, false),
    row("attention", Some("attention"), "1u32", false, false),
    row("scan", Some("scan_f32"), "1u32", false, false),
];

/// 按注册名覆盖 `reference` 入口与语义模块的例外表：一行三列 `(注册名, 参考覆盖, 模块覆盖)`。
///
/// 两列缺省都回落骨架默认：`reference` 缺省 `module::reference`；`module` 缺省骨架自带模块、
/// 再回注册名本身。两条差异都按**注册名**描述，不属于骨架表那一行的属性：flash 版与朴素版共用
/// 骨架、只有参考实现分叉；decode 两条算子样本几何不同，各自独立模块。闭集只有这一张表，查找
/// 走 [`override_for`]，不再有第二份按名循环。
const OP_OVERRIDES: [(&str, Option<&str>, Option<&str>); 3] = [
    (
        "flash_attention_f32",
        Some("::llm_op::attention::flash_reference"),
        None,
    ),
    ("decode_attention_f32", None, Some("decode_attention")),
    (
        "decode_flash_attention_f32",
        Some("::llm_op::decode_attention::decode_flash_reference"),
        Some("decode_flash_attention"),
    ),
];

/// 按注解里写的骨架名取表项。
fn skeleton_of(name: &str) -> Option<&'static Skeleton> {
    SKELETON_TABLE.iter().find(|row| row.name == name)
}

/// 按注册名在例外表里闭集查找；未登记的名字回落骨架默认。
fn override_for(name: &str) -> Option<(&'static str, Option<&'static str>, Option<&'static str>)> {
    OP_OVERRIDES.iter().find(|(op, _, _)| *op == name).copied()
}

/// 模块名：先取例外表的模块覆盖，再回落到骨架自带模块，最后用注册名本身。
fn module_for<'a>(skeleton: Option<&'a str>, name: &'a str) -> &'a str {
    override_for(name)
        .and_then(|(_, _, module)| module)
        .unwrap_or_else(|| skeleton.unwrap_or(name))
}

/// 渲染 `MODEL` 的八个字段，以及（`gemv`）随字段一起求值的粒度断言。
fn render_fields(
    code: &mut String,
    shape: &Skeleton,
    name: &str,
    ty: &str,
    dims: (usize, usize),
    dtype: &str,
) {
    let (m, n) = dims;
    let module = module_for(shape.module, name);
    let suffix = if shape.generic {
        format!("::<{ty}>")
    } else {
        String::new()
    };
    code.push_str(&format!(
        "    group: {},\n",
        shape.group.replace("{ty}", ty)
    ));
    code.push_str(&format!("    shape: ({m}usize, {n}usize),\n"));
    code.push_str(&format!("    dtype: ::llm_op::model::DType::{dtype},\n"));
    code.push_str(&format!("    axes: ::llm_op::{module}::AXES,\n"));
    code.push_str(&format!(
        "    shape_ctx: ::llm_op::{module}::project_shape,\n"
    ));
    code.push_str(&format!("    cost: ::llm_op::{module}::cost{suffix},\n"));
    code.push_str(&format!(
        "    sample: ::llm_op::{module}::sample{suffix},\n"
    ));
    code.push_str(&format!(
        "    reference: {},\n",
        reference_path(module, &suffix, name)
    ));
    code.push_str("};\n\n");
    if shape.grain {
        render_grain_asserts(code, ty, m);
    }
}

/// `reference` 字段的路径：默认 `module::reference`，例外表里登记的入口优先。
fn reference_path(module: &str, suffix: &str, name: &str) -> String {
    match override_for(name).and_then(|(_, reference, _)| reference) {
        Some(path) => format!("{path}{suffix}"),
        None => format!("::llm_op::{module}::reference{suffix}"),
    }
}

/// `gemv` 的两条编译期断言，在**生产别名上**求值。
///
/// 格式自证交给 `Format::CHECK`：粒度与派生轴除数同源这条规则属于格式契约，不属于宏。宏唯一
/// 要做的，是让它在生产别名上被求值——不被引用的关联常量不会报错。
fn render_grain_asserts(code: &mut String, ty: &str, m: usize) {
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
}

/// `#[cfg(test)] mod __model_check`：把注解、注册表与 `.k` 签名三份事实对齐。
fn render_model_check(code: &mut String, declaration: &ModelDecl) {
    let name = literal(&declaration.name);
    let dtype_name = literal(&declaration.dtype);
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
        "        let shape = (registered.shape_ctx)(registered.shape.0, registered.shape.1).expect(\"形状上下文可构造\");\n",
    );
    code.push_str("        (registered.cost)(&shape).expect(\"生产形状必须能算出代价\");\n");
    // 注解里的 `dtype` 是**字符串**，`MODEL.dtype` 是**变体**：这一句就是两者之间的绑定点。
    // 词汇表改名、或生成代码挑错变体，都会在这里失败，而不是等到 `verify` 对一个错宽度的
    // 缓冲对拍。全量与映射规则由 `tests/contracts.rs` 统一核对。
    code.push_str("        assert_eq!(\n");
    code.push_str(&format!(
        "            ::llm_op::model::DType::parse_name({dtype_name}),\n"
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
        declaration.dtype,
    ));
    code.push_str("            signature.output_type().map_or(\"-\", |ctype| ctype.c_name()),\n");
    code.push_str("        );\n");
    code.push_str("    }\n}\n");
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
#[path = "../tests/codegen_tests.rs"]
mod tests;

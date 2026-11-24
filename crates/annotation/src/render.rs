//! 渲染：把一份声明拼成源码。只做字符串拼接、不维护状态、不接触 `proc_macro`，因此可以用纯文本断言测试。

use crate::decl::{
    AttrModel, AxisModel, ChunkModel, DTYPES, Declaration, GridModel, ModelDecl, NeedModel,
    OpModel, ParallelDecl, PhaseModel, RuntimeDecl, SKELETONS, SpanModel, VariantModel,
};
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
    if axes.is_empty() {
        code.push_str("::nn::Shape::new(::std::vec::Vec::<usize>::new())")
    } else {
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

/// 整条算子表的预留容量：按最宽一份视图留足余量，渲染一次不二次分配。
pub(crate) const OPS_ROOM: usize = 8192;

/// 把一批算子模型渲染成 `pub const OPS: [OpView; N] = [...]` 字面量流。
///
/// 任一闭集短名无法映射到 prof 变体时，整份生成物退一条 `compile_error!`（不产出指向
/// 不存在变体的代码）；与 [`render_model`] 的闭集兜底同一口径。
pub(crate) fn render_ops(ops: &[OpModel]) -> String {
    match first_error(ops) {
        Some(message) => format!("compile_error!({:?});\n", message),
        None => render_ops_unchecked(ops),
    }
}

/// 遍历整批，返回第一条无法映射的闭集 / 非法写法文案；全部可映射时为 `None`。
fn first_error(ops: &[OpModel]) -> Option<String> {
    ops.iter().find_map(|op| {
        if skeleton_variant(&op.skeleton).is_none() {
            return Some(op_error("骨架", &op.skeleton));
        }
        needs_error(&op.needs)
            .err()
            .or_else(|| layouts_error(&op.layouts))
            .or_else(|| variants_error(&op.variants))
    })
}

/// 需求段逐条校验：原语 / 类型 / 存储 / 属性键 / 区间写法都要可映射。
fn needs_error(needs: &[NeedModel]) -> Result<(), String> {
    for need in needs {
        check_need(need)?;
    }
    Ok(())
}

/// 单条需求的全部闭集与区间校验。
fn check_need(need: &NeedModel) -> Result<(), String> {
    if prim_variant(&need.prim).is_none() {
        return Err(op_error("原语", &need.prim));
    }
    space_error(&need.space)?;
    dtype_error(&need.dt)?;
    attrs_error(&need.attrs)?;
    span_error(&need.extent)?;
    span_error(&need.n)?;
    span_error(&need.k)
}

/// 原语未映射时返回带原语名的错误；已映射为 `Ok(())`。
fn prim_variant(prim: &str) -> Option<&'static str> {
    match prim {
        "mma" => Some("Mma"),
        "dot" => Some("Dot"),
        "load" => Some("Load"),
        "load_async" => Some("LoadAsync"),
        "commit_wait" => Some("CommitWait"),
        "store" => Some("Store"),
        "bar" => Some("Bar"),
        "shuffle" => Some("Shuffle"),
        "broadcast" => Some("Broadcast"),
        "reduce" => Some("Reduce"),
        "scan" => Some("Scan"),
        "exp2" => Some("Exp2"),
        "rcp" => Some("Rcp"),
        "cvt" => Some("Cvt"),
        "quant" => Some("Quant"),
        "atomic" => Some("Atomic"),
        _ => None,
    }
}

/// 骨架短名 → `Skeleton` 变体名。
fn skeleton_variant(short: &str) -> Option<&'static str> {
    match short {
        "elementwise" => Some("Elementwise"),
        "reduce" => Some("Reduce"),
        "gemm" => Some("Gemm"),
        "scan" => Some("Scan"),
        _ => None,
    }
}

/// 存储短名 → `Space` 变体名。
fn space_variant(short: &str) -> Option<&'static str> {
    match short {
        "GM" => Some("Gm"),
        "FAST" => Some("Fast"),
        "REG" => Some("Reg"),
        _ => None,
    }
}

/// 类型短名 → `DType` 变体名（覆盖语料出现的定点 / 浮点档）。
fn dtype_variant(short: &str) -> Option<&'static str> {
    match short {
        "f32" => Some("F32"),
        "f16" => Some("F16"),
        "bf16" => Some("BF16"),
        "i2" => Some("I2"),
        "i4" => Some("I4"),
        "i8" => Some("I8"),
        "i32" => Some("I32"),
        _ => None,
    }
}

/// 属性键短名 → `AttrKey` 变体名。
fn attr_key_variant(short: &str) -> Option<&'static str> {
    match short {
        "op" => Some("Op"),
        "axis" => Some("Axis"),
        "kind" => Some("Kind"),
        "sat" => Some("Sat"),
        "accuracy" => Some("Accuracy"),
        "group" => Some("Group"),
        "lanes" => Some("Lanes"),
        "src" => Some("Src"),
        "dst" => Some("Dst"),
        "stages" => Some("Stages"),
        "exec" => Some("Exec"),
        "align" => Some("Align"),
        _ => None,
    }
}

/// 存储列表逐项校验。
fn space_error(space: &[String]) -> Result<(), String> {
    space
        .iter()
        .find_map(|name| space_variant(name).map_or_else(|| Some(op_error("存储", name)), |_| None))
        .map_or(Ok(()), Err)
}

/// 类型列表逐项校验。
fn dtype_error(dt: &[String]) -> Result<(), String> {
    dt.iter()
        .find_map(|name| dtype_variant(name).map_or_else(|| Some(op_error("类型", name)), |_| None))
        .map_or(Ok(()), Err)
}

/// 属性段逐条校验键。
fn attrs_error(attrs: &[AttrModel]) -> Result<(), String> {
    attrs
        .iter()
        .find_map(|attr| {
            attr_key_variant(&attr.key)
                .map_or_else(|| Some(op_error("属性键", &attr.key)), |_| None)
        })
        .map_or(Ok(()), Err)
}

/// 区间写法校验：闭区间暂无可 const 构造子。
fn span_error(span: &SpanModel) -> Result<(), String> {
    match span {
        SpanModel::Range(..) => Err(op_error("区间", "闭区间 `lo..hi` 暂无 const 构造子")),
        _ => Ok(()),
    }
}

/// 布局列表逐项校验写法可解析。
fn layouts_error(layouts: &[(String, String)]) -> Option<String> {
    layouts.iter().find_map(|(_, spec)| layout_expr(spec).err())
}

/// 变体段逐条校验调度写法可解析。
fn variants_error(variants: &[VariantModel]) -> Option<String> {
    variants
        .iter()
        .find_map(|variant| schedule_expr(&variant.schedule).err())
}

/// 闭集未映射的统一错误文案。
fn op_error(kind: &str, value: &str) -> String {
    format!("注解层内部错误：{kind} `{value}` 不在 prof 闭集内")
}

/// 校验通过后的表渲染。
fn render_ops_unchecked(ops: &[OpModel]) -> String {
    let mut code = String::with_capacity(OPS_ROOM);
    let _ = writeln!(
        code,
        "pub const OPS: [::llm_prof::decl::OpView; {}] = [",
        ops.len()
    );
    for op in ops {
        render_op(&mut code, op);
    }
    code.push_str("];\n");
    code
}

/// 一条算子的 `OpView` 字面量。
fn render_op(code: &mut String, op: &OpModel) {
    code.push_str("    ::llm_prof::decl::OpView {\n");
    field_str(code, "name", &op.name);
    skeleton_field(code, &op.skeleton);
    field_str(code, "summary", &op.summary);
    render_axes(code, &op.axes);
    render_needs(code, &op.needs);
    render_layouts(code, &op.layouts);
    render_grid(code, &op.grid);
    render_chunk(code, &op.chunk);
    render_consts(code, &op.consts);
    render_variants(code, &op.variants);
    field_str(code, "template", &op.template);
    render_phases(code, &op.phases);
    code.push_str("    },\n");
}

/// 字符串字段：`name: "..",`。
fn field_str(code: &mut String, field: &str, value: &str) {
    code.push_str(&format!("        {field}: "));
    literal(code, value);
    code.push_str(",\n");
}

/// 骨架字段：拼 `Skeleton::Variant`。
fn skeleton_field(code: &mut String, skeleton: &str) {
    let variant = skeleton_variant(skeleton).expect("闭集已在 first_error 校验");
    code.push_str(&format!(
        "        skeleton: ::llm_prof::decl::Skeleton::{variant},\n"
    ));
}

/// 轴列表：`axes: &[AxisView { .. }, ..],`。
fn render_axes(code: &mut String, axes: &[AxisModel]) {
    code.push_str("        axes: &[");
    comma(code, axes.iter(), |code, axis| {
        code.push_str("::llm_prof::decl::AxisView { name: ");
        literal(code, &axis.name);
        code.push_str(", values: &[");
        comma(code, axis.values.iter(), |code, value| {
            let _ = write!(code, "{value}u32");
        });
        code.push_str("] }");
    });
    code.push_str("],\n");
}

/// 需求列表。
fn render_needs(code: &mut String, needs: &[NeedModel]) {
    code.push_str("        needs: &[");
    comma(code, needs.iter(), render_need);
    code.push_str("],\n");
}

/// 单条需求的 `NeedView` 字面量。
fn render_need(code: &mut String, need: &NeedModel) {
    let prim = prim_variant(&need.prim).expect("闭集已校验");
    code.push_str("::llm_prof::decl::NeedView { prim: ::llm_prof::atom::Prim::");
    code.push_str(prim);
    code.push_str(", extent: ");
    span_expr(code, &need.extent);
    code.push_str(", n: ");
    span_expr(code, &need.n);
    code.push_str(", k: ");
    span_expr(code, &need.k);
    code.push_str(", dt: &[");
    comma(code, need.dt.iter(), |code, name| {
        let variant = dtype_variant(name).expect("闭集已校验");
        let _ = write!(code, "::llm_prof::atom::DType::{variant}");
    });
    code.push_str("], space: &[");
    comma(code, need.space.iter(), |code, name| {
        let variant = space_variant(name).expect("闭集已校验");
        let _ = write!(code, "::llm_prof::atom::Space::{variant}");
    });
    code.push_str("], attrs: &[");
    comma(code, need.attrs.iter(), render_attr);
    code.push_str("] }");
}

/// 一条属性的 `AttrView` 字面量。
fn render_attr(code: &mut String, attr: &AttrModel) {
    let key = attr_key_variant(&attr.key).expect("闭集已校验");
    code.push_str("::llm_prof::decl::AttrView { key: ::llm_prof::atom::AttrKey::");
    code.push_str(key);
    code.push_str(", values: &[");
    comma(code, attr.values.iter(), |code, value| literal(code, value));
    code.push_str("] }");
}

/// 区间表达式：选 const 构造子。
fn span_expr(code: &mut String, span: &SpanModel) {
    match span {
        SpanModel::Any => code.push_str("::llm_prof::atom::Span::any()"),
        SpanModel::Exact(value) => {
            let _ = write!(code, "::llm_prof::atom::Span::exact({value})");
        }
        SpanModel::AtMost(value) => {
            let _ = write!(code, "::llm_prof::atom::Span::at_most({value})");
        }
        SpanModel::AtLeast(value) => {
            let _ = write!(code, "::llm_prof::atom::Span::at_least({value})");
        }
        SpanModel::Range(..) => code.push_str("::llm_prof::atom::Span::any()"),
    }
}

/// 布局列表：`layouts: &[("a", Row), ..],`。
fn render_layouts(code: &mut String, layouts: &[(String, String)]) {
    code.push_str("        layouts: &[");
    comma(code, layouts.iter(), |code, (slot, spec)| {
        literal(code, slot);
        code.push_str(", ");
        code.push_str(&layout_expr(spec).expect("写法已校验"));
    });
    code.push_str("],\n");
}

/// 网格映射；缺省为 `None`。
fn render_grid(code: &mut String, grid: &Option<GridModel>) {
    code.push_str("        grid: ");
    match grid {
        Some(grid) => {
            code.push_str("Some(::llm_prof::decl::GridView { dims: &[");
            comma(code, grid.dims.iter(), |code, (thread, scalar)| {
                literal(code, thread);
                code.push_str(", ");
                literal(code, scalar);
            });
            code.push_str("] }),");
        }
        None => code.push_str("None,"),
    }
    code.push('\n');
}

/// 分块并行；缺省为 `None`。
fn render_chunk(code: &mut String, chunk: &Option<ChunkModel>) {
    code.push_str("        chunk: ");
    match chunk {
        Some(chunk) => {
            let _ = write!(
                code,
                "Some(::llm_prof::decl::ChunkView {{ size: {} }}),",
                chunk.size
            );
        }
        None => code.push_str("None,"),
    }
    code.push('\n');
}

/// 模板常量列表。
fn render_consts(code: &mut String, consts: &[(String, String)]) {
    code.push_str("        consts: &[");
    comma(code, consts.iter(), |code, (key, value)| {
        literal(code, key);
        code.push_str(", ");
        literal(code, value);
    });
    code.push_str("],\n");
}

/// 变体链。
fn render_variants(code: &mut String, variants: &[VariantModel]) {
    code.push_str("        variants: &[");
    comma(code, variants.iter(), render_variant);
    code.push_str("],\n");
}

/// 单条变体的 `VariantView` 字面量。
fn render_variant(code: &mut String, variant: &VariantModel) {
    code.push_str("::llm_prof::decl::VariantView { name: ");
    literal(code, &variant.name);
    let _ = write!(code, ", order: {}u32, schedule: ", variant.order);
    code.push_str(&schedule_expr(&variant.schedule).expect("写法已校验"));
    let _ = write!(
        code,
        ", mma_share: {}, extra_needs: &[] }}",
        variant.mma_share
    );
}

/// 阶段列表。
fn render_phases(code: &mut String, phases: &[PhaseModel]) {
    code.push_str("        phases: &[");
    comma(code, phases.iter(), |code, phase| {
        code.push_str("::llm_prof::decl::PhaseView { name: ");
        literal(code, &phase.name);
        code.push_str(", reads: &[");
        comma(code, phase.reads.iter(), |code, value| literal(code, value));
        code.push_str("], writes: &[");
        comma(code, phase.writes.iter(), |code, value| {
            literal(code, value)
        });
        code.push_str("] }");
    });
    code.push_str("],\n");
}

/// 调度写法 → `Schedule` 构造表达式；写法不在闭集内时返回错误文案。
fn schedule_expr(schedule: &str) -> Result<String, String> {
    let text = schedule.trim();
    if text == "direct" {
        return Ok("::llm_prof::sch::Schedule::Direct".to_string());
    }
    let Some((name, args)) = split_args(text) else {
        return Err(op_error("调度", schedule));
    };
    match (name, split_numbers(&args)?.as_slice()) {
        ("pipe", [stages]) => Ok(format!(
            "::llm_prof::sch::Schedule::Pipe {{ stages: {stages} }}"
        )),
        ("ws", [producer, consumer]) => Ok(format!(
            "::llm_prof::sch::Schedule::WarpSpecialized {{ producer: {producer}, consumer: {consumer} }}"
        )),
        ("pingpong", [groups]) => Ok(format!(
            "::llm_prof::sch::Schedule::PingPong {{ groups: {groups} }}"
        )),
        _ => Err(op_error("调度", schedule)),
    }
}

/// 布局写法 → `LayoutSpec` 构造表达式；写法不在闭集内时返回错误文案。
fn layout_expr(text: &str) -> Result<String, String> {
    match text.trim() {
        "row" => Ok("::llm_prof::sch::layout::LayoutSpec::Row".to_string()),
        "col" => Ok("::llm_prof::sch::layout::LayoutSpec::Col".to_string()),
        other => {
            if let Some(rest) = other.strip_prefix("swz<") {
                let [b, m, s] = split_widths(
                    rest.strip_suffix('>')
                        .ok_or_else(|| op_error("布局", text))?,
                )?;
                return Ok(format!(
                    "::llm_prof::sch::layout::LayoutSpec::Swz {{ b: {b}, m: {m}, s: {s} }}"
                ));
            }
            if let Some(rest) = other.strip_prefix("pad(") {
                let pad = rest
                    .strip_suffix(')')
                    .ok_or_else(|| op_error("布局", text))?
                    .trim();
                let pad: u32 = pad.parse().map_err(|_| op_error("布局", text))?;
                return Ok(format!("::llm_prof::sch::layout::LayoutSpec::Pad({pad})"));
            }
            Err(op_error("布局", text))
        }
    }
}

/// 拆 `name<args>`：返回名字与尖括号内参数串。
fn split_args(text: &str) -> Option<(&str, String)> {
    let open = text.find('<')?;
    let close = text.rfind('>')?;
    if open >= close {
        return None;
    }
    Some((text[..open].trim(), text[open + 1..close].to_string()))
}

/// 逗号分隔的无符号整数参数列表。
fn split_numbers(args: &str) -> Result<Vec<u32>, String> {
    args.split(',')
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .map_err(|_| op_error("调度参数", part))
        })
        .collect()
}

/// `swz<b,m,s>` 的三个位宽（u8）。
fn split_widths(inner: &str) -> Result<[u8; 3], String> {
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    let [b, m, s] = parts.try_into().map_err(|_| op_error("布局", inner))?;
    let width = |value: &str| value.parse::<u8>().map_err(|_| op_error("位宽", value));
    Ok([width(b)?, width(m)?, width(s)?])
}

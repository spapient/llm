//! 职责：单 kernel 级结构化 AST（ADR 0017 的 L1）。

use crate::decl::{OpDecl, Skeleton, TileCombo};
use crate::kernel::{Access, Arith, Expr, Kernel, Parameter, ScalarType, ValueType};
use crate::substitute::{self, Arg, FragmentSet, Segment, error, lex};
use core::fmt::{self, Write as _};
use prof::atom::Prim;
use prof::sig::CType;

/// 模板级节点：拥有所有权，不借用模板正文。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    /// 原样透传的 C 文本。
    Text(String),
    /// 槽位解析后的最终值（算子名 / 轴值 / 布局构造子）。
    Slot(String),
    /// 结构化的原语调用。
    Prim(PrimCall),
}

/// 一次 `@原语(...)` 调用：原语、参数与当前后端片段的展开文本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimCall {
    /// 原语（闭集之一）。
    pub prim: Prim,
    /// 调用参数（owned）。
    pub args: Vec<CallArg>,
    /// 片段经占位符替换后的文本；分号裁剪与缩进留给 [`render`]。
    pub body: String,
}

/// 原语调用参数的 owned 版本，对应 [`prof::tpl::Arg`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallArg {
    /// 位置参数。
    Expr(String),
    /// 命名参数（`key: value`）。
    Named {
        /// 键。
        key: String,
        /// 值。
        value: String,
    },
}

impl CallArg {
    /// 从借用的词法参数构造 owned 参数（不做槽位替换，仅测试核对用）。
    #[cfg(test)]
    fn from_ref(arg: &Arg<'_>) -> Self {
        match arg {
            Arg::Expr(text) => Self::Expr((*text).to_string()),
            Arg::Named { key, value } => Self::Named {
                key: (*key).to_string(),
                value: (*value).to_string(),
            },
        }
    }

    /// 借用视图：owned 参数转回词法参数，供 [`substitute::splice`] 直接消费，不需要第二套占位符规则。
    fn as_ref(&self) -> Arg<'_> {
        match self {
            Self::Expr(text) => Arg::Expr(text),
            Self::Named { key, value } => Arg::Named { key, value },
        }
    }
}

/// 把一份声明 + tile + 片段层提升成节点序列。
///
/// # Errors
///
/// 槽位无法解析、片段缺失或占位符替换失败时，返回与 [`substitute::render`] 同源的诊断。
pub fn build(
    op: &OpDecl,
    combo: &TileCombo,
    fragments: &FragmentSet,
) -> diagnostic::Result<Vec<Node>> {
    let segments = lex(&op.template);
    let mut nodes = Vec::with_capacity(segments.len());
    for segment in &segments {
        nodes.push(node(op, combo, fragments, segment)?);
    }
    Ok(nodes)
}

/// 一个词法片段 → 一个节点：三种片段各有唯一的一条转换。
fn node(
    op: &OpDecl,
    combo: &TileCombo,
    fragments: &FragmentSet,
    segment: &Segment<'_>,
) -> diagnostic::Result<Node> {
    match segment {
        Segment::Text(part) => Ok(Node::Text((*part).to_string())),
        Segment::Slot { name, line } => Ok(Node::Slot(slot_value(op, combo, name, *line)?)),
        Segment::Call { prim, args, line } => prim_node(op, combo, fragments, *prim, args, *line),
    }
}

/// 槽位取值：算子名 / 轴取值 / 布局构造子 / 模板常量，四者之外的槽位是声明错误。
fn slot_value(
    op: &OpDecl,
    combo: &TileCombo,
    name: &str,
    line: usize,
) -> diagnostic::Result<String> {
    if name == "op" {
        return Ok(op.name.clone());
    }
    combo
        .get(name)
        .map(|value| value.to_string())
        .or_else(|| op.layout(name).map(|spec| spec.as_str()))
        .or_else(|| op.const_value(name).map(str::to_string))
        .ok_or_else(|| {
            error(
                line,
                format!("槽位 `{{{{{name}}}}}` 既不是轴、布局槽位，也不是模板常量"),
            )
        })
}

/// 原语调用节点：片段必须存在，占位符必须能全部解析；实参里的 `{{槽位}}` 先解析再拼接。
fn prim_node(
    op: &OpDecl,
    combo: &TileCombo,
    fragments: &FragmentSet,
    prim: Prim,
    args: &[Arg<'_>],
    line: usize,
) -> diagnostic::Result<Node> {
    let Some(fragment) = fragments.get(prim) else {
        return Err(substitute::missing_fragment(line, prim, fragments));
    };
    let resolved: Vec<CallArg> = args
        .iter()
        .map(|arg| resolve_arg(op, combo, arg, line))
        .collect::<diagnostic::Result<_>>()?;
    let borrowed: Vec<Arg<'_>> = resolved.iter().map(CallArg::as_ref).collect();
    let body = substitute::splice(fragment, &borrowed, line)?;
    Ok(Node::Prim(PrimCall {
        prim,
        args: resolved,
        body,
    }))
}

/// 一个实参的 `{{槽位}}` 替换：位置参数与命名参数的值都过一遍 [`resolve_slots`]。
fn resolve_arg(
    op: &OpDecl,
    combo: &TileCombo,
    arg: &Arg<'_>,
    line: usize,
) -> diagnostic::Result<CallArg> {
    match arg {
        Arg::Expr(text) => Ok(CallArg::Expr(resolve_slots(op, combo, text, line)?)),
        Arg::Named { key, value } => Ok(CallArg::Named {
            key: (*key).to_string(),
            value: resolve_slots(op, combo, value, line)?,
        }),
    }
}

/// 把一段文本里的 `{{槽位}}` 逐个换成取值；多数实参不含槽位，无 `{{` 时原样返回免分配。
pub(crate) fn resolve_slots(
    op: &OpDecl,
    combo: &TileCombo,
    text: &str,
    line: usize,
) -> diagnostic::Result<String> {
    if !text.contains("{{") {
        return Ok(text.to_string());
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let inner = &rest[start + 2..];
        let Some(end) = inner.find("}}") else {
            out.push_str(&rest[start..]);
            break;
        };
        let offset = text.len() - rest.len() + start;
        let hole_line = line + text[..offset].bytes().filter(|byte| *byte == b'\n').count();
        out.push_str(&slot_value(op, combo, inner[..end].trim(), hole_line)?);
        rest = &inner[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

/// 把节点序列打印回源文本：[`substitute::render`] 就是 [`build`] + 本函数的薄包装。
#[must_use]
pub fn render(nodes: &[Node]) -> String {
    let mut text = String::new();
    for node in nodes {
        match node {
            Node::Text(part) => text.push_str(part),
            Node::Slot(value) => text.push_str(value),
            Node::Prim(call) => crate::substitute::place(&mut text, &call.body),
        }
    }
    text
}

/// 节点序列的确定性规范编码（缓存指纹输入）。
#[must_use]
pub fn encode(nodes: &[Node]) -> String {
    let mut out = String::new();
    for node in nodes {
        // 写入 `String` 永远不会失败：只有格式化实现本身出错才会走到 `Err`。
        let _ = write_node(&mut out, node);
    }
    out
}

/// 一个节点一行：`T`/`S`/`P` 三种前缀，字段一律经 [`escape`] 转义。
fn write_node(out: &mut String, node: &Node) -> fmt::Result {
    match node {
        Node::Text(part) => writeln!(out, "T {}", escape(part)),
        Node::Slot(value) => writeln!(out, "S {}", escape(value)),
        Node::Prim(call) => writeln!(
            out,
            "P {} {} {}",
            call.prim.as_str(),
            encode_args(&call.args),
            escape(&call.body)
        ),
    }
}

/// 参数序列：位置参数 `e…|`、命名参数 `n…=…|`，两者都以 `|` 结尾。
fn encode_args(args: &[CallArg]) -> String {
    args.iter()
        .map(|arg| match arg {
            CallArg::Expr(value) => format!("e{}|", escape(value)),
            CallArg::Named { key, value } => format!("n{}={}|", escape(key), escape(value)),
        })
        .collect()
}

/// 转义：反斜杠、换行与字段分隔符各有唯一编码，保证编码可逆且无歧义。
fn escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match escape_char(ch) {
            Some(coded) => out.push_str(coded),
            None => out.push(ch),
        }
    }
    out
}

/// 单个字符的编码；`None` 表示原样输出。闭集只有四个，多一个就多一种解码可能。
fn escape_char(ch: char) -> Option<&'static str> {
    match ch {
        '\\' => Some("\\\\"),
        '\n' => Some("\\n"),
        '|' => Some("\\p"),
        '=' => Some("\\q"),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Token<'a> {
    pub(crate) text: &'a str,
    pub(crate) line: usize,
}

pub(crate) fn lex_c(source: &str) -> diagnostic::Result<Vec<Token<'_>>> {
    let mut tokens = Vec::new();
    let mut at = 0;
    let mut line = 1;
    while at < source.len() {
        at = skip_trivia(source, at, &mut line)?;
        if at == source.len() {
            break;
        }
        let rest = &source[at..];
        let len = token_len(rest);
        if len == 0 {
            return Err(error(
                line,
                format!("无法识别字符 `{}`", rest.chars().next().unwrap_or_default()),
            ));
        }
        tokens.push(Token {
            text: &rest[..len],
            line,
        });
        at += len;
    }
    Ok(tokens)
}

fn skip_trivia(source: &str, mut at: usize, line: &mut usize) -> diagnostic::Result<usize> {
    while at < source.len() {
        let rest = &source[at..];
        let byte = rest.as_bytes()[0];
        if byte.is_ascii_whitespace() {
            *line += usize::from(byte == b'\n');
            at += 1;
            continue;
        }
        if comment_start(rest) {
            at += comment(rest, line)?;
            continue;
        }
        break;
    }
    Ok(at)
}

fn comment_start(text: &str) -> bool {
    text.starts_with("//") || text.starts_with("/*")
}

fn comment(text: &str, line: &mut usize) -> diagnostic::Result<usize> {
    if text.starts_with("//") {
        return Ok(text.find('\n').unwrap_or(text.len()));
    }
    let end = text
        .find("*/")
        .ok_or_else(|| error(*line, "块注释未闭合"))?;
    *line += text[..end].bytes().filter(|byte| *byte == b'\n').count();
    Ok(end + 2)
}

fn token_len(text: &str) -> usize {
    let bytes = text.as_bytes();
    if identifier_byte(bytes[0]) {
        return take(text, |byte| byte.is_ascii_alphanumeric() || byte == b'_');
    }
    if number_start(bytes) {
        return number_len(text);
    }
    if quote_byte(bytes[0]) {
        return quoted_len(text, bytes[0]);
    }
    [
        "++", "--", "+=", "-=", "*=", "/=", "%=", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>",
        "->",
    ]
    .iter()
    .find(|op| text.starts_with(**op))
    .map_or(1, |op| op.len())
}

fn identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn number_start(bytes: &[u8]) -> bool {
    bytes[0].is_ascii_digit() || (bytes[0] == b'.' && bytes.get(1).is_some_and(u8::is_ascii_digit))
}

fn quote_byte(byte: u8) -> bool {
    byte == b'\'' || byte == b'"'
}

fn take(text: &str, predicate: impl Fn(u8) -> bool) -> usize {
    text.bytes().take_while(|byte| predicate(*byte)).count()
}

fn number_len(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut at = 0;
    while number_byte(bytes, at) {
        at += 1;
    }
    at
}

fn number_byte(bytes: &[u8], at: usize) -> bool {
    bytes.get(at).is_some_and(|byte| {
        byte.is_ascii_alphanumeric() || *byte == b'.' || exponent_sign(bytes, at)
    })
}

fn exponent_sign(bytes: &[u8], at: usize) -> bool {
    matches!(bytes.get(at), Some(b'+' | b'-'))
        && at > 0
        && matches!(bytes.get(at - 1), Some(b'e' | b'E'))
}

fn quoted_len(text: &str, quote: u8) -> usize {
    let mut escaped = false;
    for (idx, byte) in text.bytes().enumerate().skip(1) {
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == quote {
            return idx + 1;
        }
    }
    0
}

#[derive(Clone, Copy)]
struct Conditional {
    parent: bool,
    active: bool,
    in_else: bool,
}

pub(crate) fn preprocess(
    source: &str,
    evaluate: impl Fn(&str) -> diagnostic::Result<i64>,
) -> diagnostic::Result<String> {
    let mut out = String::with_capacity(source.len());
    let mut stack = Vec::new();
    for (idx, line) in source.split_inclusive('\n').enumerate() {
        preprocess_line(&mut out, line, idx + 1, &mut stack, &evaluate)?;
    }
    if !stack.is_empty() {
        return Err(error(1, "条件编译缺少 #endif"));
    }
    Ok(out)
}

fn preprocess_line(
    out: &mut String,
    line: &str,
    line_no: usize,
    stack: &mut Vec<Conditional>,
    evaluate: &impl Fn(&str) -> diagnostic::Result<i64>,
) -> diagnostic::Result<()> {
    let trimmed = line.trim();
    if pragma_line(out, line, trimmed, stack) {
        return Ok(());
    }
    directive_line(out, line, trimmed, stack, evaluate)?;
    active_error(trimmed, line_no, stack)
}

fn directive_line(
    out: &mut String,
    line: &str,
    trimmed: &str,
    stack: &mut Vec<Conditional>,
    evaluate: &impl Fn(&str) -> diagnostic::Result<i64>,
) -> diagnostic::Result<()> {
    if let Some(expression) = trimmed.strip_prefix("#if ") {
        return start_conditional(expression, stack, evaluate);
    }
    if trimmed.starts_with('#') {
        return directive(trimmed, stack);
    }
    if stack.last().is_none_or(|frame| frame.active) {
        out.push_str(line);
    }
    Ok(())
}

fn active_error(trimmed: &str, line: usize, stack: &[Conditional]) -> diagnostic::Result<()> {
    let active = stack.last().is_none_or(|frame| frame.active);
    if active && trimmed.starts_with("#error") {
        return Err(error(line, trimmed.to_string()));
    }
    Ok(())
}

fn pragma_line(out: &mut String, line: &str, trimmed: &str, stack: &[Conditional]) -> bool {
    let active = stack.last().is_none_or(|frame| frame.active);
    if !(active && trimmed.starts_with("#pragma ")) {
        return false;
    }
    out.push_str(line);
    true
}

fn directive(text: &str, stack: &mut Vec<Conditional>) -> diagnostic::Result<()> {
    match text {
        "#else" => else_directive(stack),
        "#endif" => end_conditional(stack),
        _ if text.starts_with("#error") => Ok(()),
        _ => Err(error(1, format!("不支持的预处理指令 `{text}`"))),
    }
}

fn start_conditional(
    expression: &str,
    stack: &mut Vec<Conditional>,
    evaluate: &impl Fn(&str) -> diagnostic::Result<i64>,
) -> diagnostic::Result<()> {
    let parent = stack.last().is_none_or(|frame| frame.active);
    let active = parent && evaluate(expression)? != 0;
    stack.push(Conditional {
        parent,
        active,
        in_else: false,
    });
    Ok(())
}

fn end_conditional(stack: &mut Vec<Conditional>) -> diagnostic::Result<()> {
    stack.pop().ok_or_else(|| error(1, "多余的 #endif"))?;
    Ok(())
}

fn else_directive(stack: &mut [Conditional]) -> diagnostic::Result<()> {
    let frame = stack.last_mut().ok_or_else(|| error(1, "多余的 #else"))?;
    if frame.in_else {
        return Err(error(1, "重复或错位的 #else"));
    }
    frame.active = frame.parent && !frame.active;
    frame.in_else = true;
    Ok(())
}

// ── AST → typed IR（P1 受限 lower）─────────────────────────────────────
/// 四则运算符槽位到 IR 算术族的同源映射；槽位是唯一单字符运算槽。
const BINOP_ARITH: [(&str, Arith); 4] = [
    ("+", Arith::Add),
    ("-", Arith::Sub),
    ("*", Arith::Mul),
    ("/", Arith::Div),
];

/// 把节点序列 lower 成 typed IR。
///
/// 只认两条已核通的闭集：四则逐点 `y[n] = a[n] op b[n]` 与求和归约 `y[0] += a[k]`。
/// 其余模板（一元三元、厂商数学函数、多临时变量、非 f32 档）识别不了返回 [`None`]，
/// 调用方回退 [`render`] 的文本双射；这里不猜半截 IR（ADR 0056 G6）。
pub fn lower(op: &OpDecl, nodes: &[Node]) -> Option<Kernel> {
    let params = ir_params(op)?;
    match op.skeleton {
        Skeleton::Elementwise => lower_pointwise(op, &params, nodes),
        Skeleton::Reduce => lower_reduce(op, &params, nodes),
        Skeleton::Gemm | Skeleton::Scan => None,
    }
}

/// 签名形参 → IR 参数；逐项委托 [`ir_param`]，非 f32 档不在 IR 闭集内。
fn ir_params(op: &OpDecl) -> Option<Vec<Parameter>> {
    op.signature.params().iter().map(ir_param).collect()
}

/// 单个形参：f32 缓冲按读写定访问模式，整数成 `usize` 标量，其余拒绝。
fn ir_param(param: &prof::sig::Param) -> Option<Parameter> {
    let ty = match param.ctype {
        CType::F32 => ValueType::Buf {
            element: ScalarType::F32,
            access: buf_access(param.mutable),
        },
        CType::Int => ValueType::Scalar(ScalarType::Usize),
        _ => return None,
    };
    Some(Parameter {
        name: param.name.clone(),
        ty,
    })
}

/// 缓冲读写访问模式：可写缓冲进写回通道，只读缓冲进读通道。
fn buf_access(mutable: bool) -> Access {
    if mutable {
        Access::ReadWrite
    } else {
        Access::Read
    }
}

/// 唯一整数标量即循环长度；IR 的 extent 必须是 `usize` 形参。
fn extent(params: &[Parameter]) -> Option<String> {
    let mut names = params
        .iter()
        .filter(|param| param.ty == ValueType::Scalar(ScalarType::Usize))
        .map(|param| param.name.clone());
    let first = names.next()?;
    names.next().is_none().then_some(first)
}

/// 四则逐点：两个 `@load` 读 + 一个 `@store` 写回，右值是两读之间的四则运算。
fn lower_pointwise(op: &OpDecl, params: &[Parameter], nodes: &[Node]) -> Option<Kernel> {
    let loads = load_aliases(nodes);
    let (left, right) = pair2(&loads)?;
    let store = store_call(nodes)?;
    let binop = binop_slot(nodes)?;
    let value = Expr::Binary {
        op: binop,
        left: Box::new(Expr::Load(left.1.clone())),
        right: Box::new(Expr::Load(right.1.clone())),
    };
    let kernel = Kernel::map(
        op.name.clone(),
        params.to_vec(),
        left.2.clone(),
        extent(params)?,
        store.0,
        value,
    );
    finish(kernel)
}

/// 求和归约：`@reduce(out, a, N, 1, op: sum, …)` 落成每迭代累加 `a[k]`；乘/最大归约没有对应族。
fn lower_reduce(op: &OpDecl, params: &[Parameter], nodes: &[Node]) -> Option<Kernel> {
    let buf = reduce_source(nodes)?;
    let store = store_call(nodes)?;
    let at = store.1.parse().ok()?;
    let kernel = Kernel::reduce(
        op.name.clone(),
        params.to_vec(),
        "k".to_string(),
        extent(params)?,
        store.0,
        at,
        Expr::Load(buf.to_string()),
    );
    finish(kernel)
}

/// 产物过 IR 校验闸后返回；校验不通过的半成品不许流出 lower。
fn finish(kernel: Kernel) -> Option<Kernel> {
    kernel.verify().ok()?;
    Some(kernel)
}

/// 归约原语缓冲名：只认 `op: sum`；其余归约族在 IR 没有对应累加模型。
fn reduce_source(nodes: &[Node]) -> Option<&str> {
    let call = reduce_call(nodes)?;
    if named_value(call, "op")? != "sum" {
        return None;
    }
    expr_at(call, 1)
}

/// 节点序列里的归约原语调用。
fn reduce_call(nodes: &[Node]) -> Option<&PrimCall> {
    nodes.iter().find_map(|node| match node {
        Node::Prim(call) if call.prim == Prim::Reduce => Some(call),
        _ => None,
    })
}

/// 恰好两个读别名（逐点二元）；左右按 `@load` 出现顺序，Sub/Div 顺序敏感。
fn pair2<T>(loads: &[T]) -> Option<(&T, &T)> {
    let [left, right] = loads else { return None };
    Some((left, right))
}

/// 全部 `@load(alias, buf + idx)`：别名指针 → 真实缓冲与循环下标。
fn load_aliases(nodes: &[Node]) -> Vec<(String, String, String)> {
    nodes.iter().filter_map(load_row).collect()
}

/// 取出匹配指定原语的调用；非 Prim 或原语不符都返回 `None`。
fn prim_row(node: &Node, prim: Prim) -> Option<&PrimCall> {
    let Node::Prim(call) = node else { return None };
    (call.prim == prim).then_some(call)
}

/// 单条 load 原语解析；非 load 或目标形不成 `buf + idx` 都跳过。
fn load_row(node: &Node) -> Option<(String, String, String)> {
    let call = prim_row(node, Prim::Load)?;
    let alias = expr_at(call, 0)?;
    let target = expr_at(call, 1)?;
    let (buf, idx) = split_target(target)?;
    Some((alias.to_string(), buf.to_string(), idx.to_string()))
}

/// `buf + idx` 拆成缓冲名与下标；两段都必须非空。
fn split_target(target: &str) -> Option<(&str, &str)> {
    let (buf, idx) = target.split_once('+')?;
    let buf = buf.trim();
    let idx = idx.trim();
    (!buf.is_empty() && !idx.is_empty()).then_some((buf, idx))
}

/// 节点序列里的 `@store(out, idx, var)`：输出缓冲、写回槽位下标与结果变量名。
fn store_call(nodes: &[Node]) -> Option<(String, String, String)> {
    nodes.iter().find_map(store_row)
}

/// 单条 store 原语解析；非 store 或形参不全都跳过。
fn store_row(node: &Node) -> Option<(String, String, String)> {
    let call = prim_row(node, Prim::Store)?;
    let out = expr_at(call, 0)?;
    let idx = expr_at(call, 1)?;
    let var = expr_at(call, 2)?;
    Some((out.to_string(), idx.to_string(), var.to_string()))
}

/// 模板槽位里的四则运算符；逐点模板的 `{{binop}}}` 是唯一单字符运算槽。
fn binop_slot(nodes: &[Node]) -> Option<Arith> {
    nodes.iter().find_map(|node| match node {
        Node::Slot(text) => BINOP_ARITH
            .iter()
            .find(|(op, _)| *op == text.as_str())
            .map(|(_, arith)| *arith),
        _ => None,
    })
}

/// 位置实参文本；命名实参在该位置上不是表达式。
fn expr_at(call: &PrimCall, idx: usize) -> Option<&str> {
    match call.args.get(idx)? {
        CallArg::Expr(text) => Some(text),
        CallArg::Named { .. } => None,
    }
}

/// 命名实参取值（`key: value`）。
fn named_value<'a>(call: &'a PrimCall, key: &str) -> Option<&'a str> {
    call.args.iter().find_map(|arg| match arg {
        CallArg::Named { key: name, value } if name == key => Some(value.as_str()),
        _ => None,
    })
}

#[cfg(test)]
#[path = "../tests/ast_tests.rs"]
mod tests;

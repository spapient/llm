//! 档案解析：把编译期展开的表逐段搬进 [`Profile`]（`parse` 与它的子函数）。
//!
//! 这里只回答"表里的字节是什么形状"，闭集与见证是否成立留给 [`super::check`]。
use super::*;
impl MatchRule {
    /// 解析规则文本：`contains:` → 择一 → 集合 → 区间 → 精确逐级降级，只回答规则种类。
    pub fn parse(text: &str) -> Self {
        let text = text.trim();
        if let Some(tokens) = text.strip_prefix("contains:") {
            return Self::Contains(split(tokens, '|'));
        }
        if text.contains('|') {
            return Self::OneOf(text.split('|').map(Self::parse).collect());
        }
        Self::parse_narrow(text)
    }
    /// 剩下三种：集合、区间、精确；没有 `|` 也没有 `contains:` 时才轮到这里。
    fn parse_narrow(text: &str) -> Self {
        if let Some((base, members)) = brace_set(text) {
            return Self::Set { base, members };
        }
        if let Some((lo, hi)) = text.split_once("..") {
            let (lo, hi) = (lo.trim().to_string(), hi.trim().to_string());
            return Self::Range { lo, hi };
        }
        Self::Exact(text.to_string())
    }
}
/// 花括号集合语法 `base{a,b}`：括号必须成对且有序，否则不认（退回区间/精确）。
fn brace_set(text: &str) -> Option<(String, Vec<String>)> {
    let (open, close) = (text.find('{')?, text.rfind('}')?);
    if open >= close {
        return None;
    }
    let base = text[..open].trim().to_string();
    Some((base, split(&text[open + 1..close], ',')))
}
fn split(text: &str, delimiter: char) -> Vec<String> {
    let parts = text.split(delimiter).map(|part| part.trim().to_string());
    parts.filter(|part| !part.is_empty()).collect()
}
impl Profile {
    /// 从构建期展开的表解析（表由 [`crate::embedded`] 给出）；闭集与见证校验留给 [`Profile::validate`]。
    pub fn parse(table: &Table) -> Result<Self, ProfileError> {
        let (meta, fingerprint) = parse_head(table)?;
        let (spaces, units) = parse_floor(table)?;
        let (provides, peak) = parse_capability(table)?;
        let (flags, preamble) = parse_compile(table);
        Ok(Self {
            meta,
            fingerprint,
            spaces,
            mem: parse_mem(table),
            simt: parse_simt(table),
            units,
            provides,
            peak,
            flags,
            preamble,
        })
    }
}
/// `[meta]` 与 `[fingerprint]` 两段。
fn parse_head(table: &Table) -> Result<(Meta, Vec<(String, MatchRule)>), ProfileError> {
    let meta = table.table("meta").ok_or_else(|| missing("meta"))?;
    let meta = parse_meta(meta)?;
    Ok((meta, parse_fingerprint(table)?))
}
/// 静态形状的两半：存储映射与执行单元。
type Floor = (Vec<(Space, String)>, Vec<Unit>);
/// `[space]` 与 `[units]` 两段。
fn parse_floor(table: &Table) -> Result<Floor, ProfileError> {
    let spaces = parse_spaces(table)?;
    Ok((spaces, parse_units(table)?))
}
/// `[[provides]]` 与 `[peak]` 两段。
fn parse_capability(table: &Table) -> Result<(Vec<Provide>, Peak), ProfileError> {
    let provides = parse_provides(table)?;
    Ok((provides, parse_peak(table)?))
}
/// `[meta]` 段：厂商、型号、输出语言、工具名与核对日期；不核对工具链与本机是否一致。
fn parse_meta(table: &Table) -> Result<Meta, ProfileError> {
    let (vendor, id) = meta_identity(table)?;
    let (emit, tool) = meta_toolchain(table)?;
    Ok(Meta {
        vendor,
        id,
        emit,
        tool,
        tool_ver: or_dash(table, "tool_ver"),
        checked: or_dash(table, "checked"),
    })
}
/// 厂商与型号：两个字符串字段。
fn meta_identity(table: &Table) -> Result<(String, String), ProfileError> {
    let vendor = text(table, "vendor", "meta.vendor")?.to_string();
    Ok((vendor, text(table, "id", "meta.id")?.to_string()))
}
/// 输出语言与工具名：`emit` 必须在 [`Emit`] 闭集里。
fn meta_toolchain(table: &Table) -> Result<(Emit, String), ProfileError> {
    let emit = closed::<Emit>(text(table, "emit", "meta.emit")?, "meta.emit")?;
    Ok((emit, text(table, "tool", "meta.tool")?.to_string()))
}
/// `[fingerprint]` 段：规则直接写在段下，或写在 `match = { ... }` 里（README §5.3）。
fn parse_fingerprint(table: &Table) -> Result<Vec<(String, MatchRule)>, ProfileError> {
    let section = table.table("fingerprint");
    let rules = section.map(|section| section.table("match").unwrap_or(section));
    let fingerprint = rules.map_or(Ok(Vec::new()), |rules| entries(rules, fingerprint_rule))?;
    if fingerprint.is_empty() {
        return Err(missing("fingerprint.match"));
    }
    Ok(fingerprint)
}
/// 一条指纹规则：值必须是字符串（规则文法由 [`MatchRule::parse`] 解释）。
fn fingerprint_rule(key: &str, value: &Value) -> Result<(String, MatchRule), ProfileError> {
    let Value::Str(text) = value else {
        return Err(bad(format!("fingerprint.match.{key}"), "只接受字符串规则"));
    };
    Ok((key.to_string(), MatchRule::parse(text)))
}
/// `[space]` 段：语义存储名 → 物理名，三级存储必须齐全。
fn parse_spaces(table: &Table) -> Result<Vec<(Space, String)>, ProfileError> {
    let section = table.table("space");
    let spaces = section.map_or(Ok(Vec::new()), |map| entries(map, space_entry))?;
    require_spaces(&spaces)?;
    Ok(spaces)
}
/// 一条存储映射：键是闭集里的语义名，值是物理名。
fn space_entry(key: &str, value: &Value) -> Result<(Space, String), ProfileError> {
    let space = closed::<Space>(key, "space")?;
    let Value::Str(name) = value else {
        return Err(bad(format!("space.{key}"), "只接受字符串"));
    };
    Ok((space, name.clone()))
}
/// 三级存储的语义名必须齐全：缺一条就拼不出完整的物理名映射。
fn require_spaces(spaces: &[(Space, String)]) -> Result<(), ProfileError> {
    let has = |wanted: Space| spaces.iter().any(|(have, _)| *have == wanted);
    let unfilled = Space::ALL.iter().copied().find(|space| !has(*space));
    unfilled.map_or(Ok(()), |space| Err(unfilled_space(space)))
}
fn unfilled_space(space: Space) -> ProfileError {
    let name = space.as_str();
    missing(&format!("space.{name}（三级存储的语义名必须齐全）"))
}
/// `[mem]` 段：容量与 bank 数，缺项按 0。
fn parse_mem(table: &Table) -> Mem {
    let mem = table.table("mem");
    let fast = mem.and_then(|mem| mem.table("FAST"));
    let reg = mem.and_then(|mem| mem.table("REG"));
    Mem {
        fast_bytes: count(fast, "bytes") as u64,
        fast_bank: count(fast, "bank") as u32,
        reg_bytes: count(reg, "bytes") as u64,
    }
}
fn count(table: Option<&Table>, key: &str) -> i64 {
    table.and_then(|table| table.int(key)).unwrap_or(0).max(0)
}
/// `[simt]` 段：缺段表示"不是 SIMT"，不能拿默认值 1 当作单 lane 的证据。
fn parse_simt(table: &Table) -> Simt {
    let simt = table.table("simt");
    Simt {
        group: count(simt, "group").max(1) as u32,
        groups_max: count(simt, "groups_max").max(1) as u32,
        declared: simt.is_some(),
    }
}
/// `[units.<name>]` 段：执行单元及其可访问的存储层；缺段表示档案不区分单元。
fn parse_units(table: &Table) -> Result<Vec<Unit>, ProfileError> {
    table
        .table("units")
        .map_or(Ok(Vec::new()), |section| entries(section, unit_entry))
}
/// 一个执行单元：必须是 `[units.<name>]` 形式的表，`space` 必填。
fn unit_entry(name: &str, value: &Value) -> Result<Unit, ProfileError> {
    let Value::Table(inner) = value else {
        return Err(bad(format!("units.{name}"), UNIT_DETAIL));
    };
    let path = format!("units.{name}.space");
    let space = parse_space_set(&path, &list(inner, "space", &path)?)?;
    Ok(Unit {
        name: name.to_string(),
        space,
    })
}
/// `[[provides]]` 数组：至少一条，否则协商没有可选的能力。
fn parse_provides(table: &Table) -> Result<Vec<Provide>, ProfileError> {
    let tables = table.table_array("provides").unwrap_or_default();
    let each = |(index, entry): (usize, &&Table)| parse_provide(entry, index);
    let provides: Vec<Provide> = tables
        .iter()
        .enumerate()
        .map(each)
        .collect::<Result<_, _>>()?;
    if provides.is_empty() {
        return Err(missing("provides"));
    }
    Ok(provides)
}
/// `[peak]` 段：roofline 用的峰值常数，两项都可以缺。
fn parse_peak(table: &Table) -> Result<Peak, ProfileError> {
    let empty = Table::new();
    let peak = table.table("peak").unwrap_or(&empty);
    Ok(Peak {
        flops_f32: parse_constant(peak, "flops_f32")?,
        bw: parse_constant(peak, "bw")?,
    })
}
/// `[compile]` 段：参与缓存指纹的参数与源文件前置声明，缺项给空。
fn parse_compile(table: &Table) -> (Vec<String>, String) {
    let compile = table.table("compile");
    let flags = compile
        .and_then(|table| table.string_array("flags"))
        .unwrap_or_default();
    let flags = flags.into_iter().map(str::to_string).collect();
    let preamble = compile
        .and_then(|table| table.string("preamble"))
        .unwrap_or("");
    (flags, preamble.to_string())
}
/// 峰值常数：`value` 必须等于 `src` 的推导式结果；抄来的数字与带变量名的公式都不收。
fn parse_constant(table: &Table, key: &str) -> Result<Option<Constant>, ProfileError> {
    table
        .table(key)
        .map(|inner| constant(inner, &format!("peak.{key}")))
        .transpose()
}
/// 一条峰值常数：两个必填项加上一次对拍。
fn constant(table: &Table, field: &str) -> Result<Constant, ProfileError> {
    let path = format!("{field}.value");
    let value = table.float("value").ok_or(missing(&path))?;
    let src = text(table, "src", &format!("{field}.src"))?.to_string();
    let derived = derive(field, &src)?;
    // 允许字面量的常规舍入（如 `2.1e12`），但容不下 10% 量级的"抄错一位"。
    if (derived - value).abs() > value.abs().max(1.0) * 1e-6 {
        let detail = format!("推导式 `{src}` = {derived:e}，与 value = {value:e} 不一致");
        return Err(bad(format!("{field}.src"), detail));
    }
    Ok(Constant { value, src })
}
/// 推导式求值；失败一律是 `invalid`（推导式是档案的一部分，写错就是声明写错）。
fn derive(field: &str, src: &str) -> Result<f64, ProfileError> {
    let fail = |error: Error| {
        let detail = format!("推导式 `{src}` 求值失败：{}", error.message());
        bad(format!("{field}.src"), detail)
    };
    eval_expression(src).map_err(fail)
}
/// 推导式写错：位置与非法内容由调用方给全。
fn invalid(detail: impl Into<String>) -> Error {
    Error::new(Code::Invalid, detail)
}
/// 求值一个只用数值字面量、`+ - * /` 与括号的推导式。
pub(super) fn eval_expression(text: &str) -> diagnostic::Result<f64> {
    let bytes = text.as_bytes();
    let mut cursor = 0usize;
    let value = eval_sum(text, bytes, &mut cursor)?;
    skip_spaces(bytes, &mut cursor);
    if cursor != bytes.len() {
        return Err(invalid(format!("位置 {cursor} 起有多余内容")));
    }
    Ok(value)
}
fn skip_spaces(bytes: &[u8], cursor: &mut usize) {
    while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
        *cursor += 1;
    }
}
/// 加减层：左结合的 `+`/`-`。
fn eval_sum(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    let mut value = eval_product(text, bytes, cursor)?;
    while let Some(operator) = operator(bytes, cursor, b"+-") {
        let operand = eval_product(text, bytes, cursor)?;
        value = added(value, operator, operand);
    }
    Ok(value)
}
/// 乘除层：`*` 与 `/` 优先于加减，除数为 0 是声明写错。
fn eval_product(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    let mut value = eval_atom(text, bytes, cursor)?;
    while let Some(operator) = operator(bytes, cursor, b"*/") {
        let operand = eval_atom(text, bytes, cursor)?;
        value = multiplied(value, operator, operand)?;
    }
    Ok(value)
}
/// 当前位置的二元运算符（消费掉它）；不在 `allowed` 里就返回 `None` 且不动游标。
fn operator(bytes: &[u8], cursor: &mut usize, allowed: &[u8]) -> Option<u8> {
    skip_spaces(bytes, cursor);
    let found = *bytes.get(*cursor)?;
    if !allowed.contains(&found) {
        return None;
    }
    *cursor += 1;
    Some(found)
}
fn added(value: f64, operator: u8, operand: f64) -> f64 {
    if operator == b'+' {
        value + operand
    } else {
        value - operand
    }
}
/// 一次乘除；除数为 0 时给出诊断，否则原样折叠。
fn multiplied(value: f64, operator: u8, operand: f64) -> diagnostic::Result<f64> {
    if operator == b'*' {
        return Ok(value * operand);
    }
    if operand == 0.0 {
        return Err(invalid("除数为 0"));
    }
    Ok(value / operand)
}
/// 原子层：括号、数字字面量，以及"这里不该有东西"。
fn eval_atom(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    skip_spaces(bytes, cursor);
    match bytes.get(*cursor) {
        Some(b'(') => eval_group(text, bytes, cursor),
        None => Err(invalid("表达式意外结束")),
        Some(byte) => eval_scalar(text, bytes, cursor, *byte),
    }
}
/// 括号表达式：括号必须闭合，里面按完整表达式求值。
fn eval_group(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    *cursor += 1;
    let value = eval_sum(text, bytes, cursor)?;
    skip_spaces(bytes, cursor);
    if bytes.get(*cursor) != Some(&b')') {
        return Err(invalid("括号没有闭合"));
    }
    *cursor += 1;
    Ok(value)
}
/// 一个数字、或者一个不允许出现在这里的字符。
fn eval_scalar(text: &str, bytes: &[u8], cursor: &mut usize, byte: u8) -> diagnostic::Result<f64> {
    if byte.is_ascii_digit() || byte == b'.' {
        return eval_number(text, bytes, cursor);
    }
    let found = char::from(byte);
    Err(invalid(format!("位置 {cursor} 处不允许字符 `{found}`")))
}
/// 数值字面量：按最长前缀切出来再交给 `f64`，非法字面量连位置一起报出。
fn eval_number(text: &str, bytes: &[u8], cursor: &mut usize) -> diagnostic::Result<f64> {
    let start = *cursor;
    scan_number(bytes, cursor);
    let literal = &text[start..*cursor];
    let illegal = |error: std::num::ParseFloatError| {
        invalid(format!("位置 {start} 的数字 `{literal}` 非法：{error}"))
    };
    literal.parse::<f64>().map_err(illegal)
}
/// 扫描数值字面量：指数只认第一个 `e`/`E`，其后的 `+`/`-` 属于指数。
fn scan_number(bytes: &[u8], cursor: &mut usize) {
    let mut exponent = false;
    while let Some(&byte) = bytes.get(*cursor) {
        match byte {
            b'0'..=b'9' | b'.' => *cursor += 1,
            b'e' | b'E' if !exponent => {
                exponent = true;
                *cursor += 1;
                skip_sign(bytes, cursor);
            }
            _ => break,
        }
    }
}
fn skip_sign(bytes: &[u8], cursor: &mut usize) {
    let sign = bytes.get(*cursor);
    if sign.is_some_and(|s| matches!(s, b'+' | b'-')) {
        *cursor += 1;
    }
}
/// 一个"名字集合"字段：标量字符串或字符串数组，与 `[needs]` 侧同一口径，形状不对直接报错。
fn names(path: &str, value: &Value) -> Result<Vec<String>, ProfileError> {
    match value {
        Value::Str(text) => Ok(vec![text.clone()]),
        Value::Array(items) => items.iter().map(|item| one_name(path, item)).collect(),
        _ => Err(bad(path, "只接受字符串或字符串数组")),
    }
}
fn one_name(path: &str, item: &Value) -> Result<String, ProfileError> {
    match item {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(bad(path, "数组里只能是字符串")),
    }
}
/// 一组存储层名字：每个都必须在 [`Space`] 闭集里。
fn parse_space_set(field: &str, items: &[&str]) -> Result<SpaceSet, ProfileError> {
    let mut set = SpaceSet::empty();
    for item in items {
        set.insert(closed::<Space>(item, field)?);
    }
    Ok(set)
}
/// 一条 `[[provides]]`：认得的键各有分支，其余一律当属性（键还要过 [`AttrKey`] 闭集）。
fn parse_provide(entry: &Table, index: usize) -> Result<Provide, ProfileError> {
    let path = at(index, "prim");
    let prim = closed::<Prim>(text(entry, "prim", &path)?, &path)?;
    let mut provide = Provide::of(prim, entry.string("impl").unwrap_or("-"));
    provide_shape(entry, index, &mut provide)?;
    provide_sets(entry, index, &mut provide)?;
    provide_tail(entry, index, provide)
}
/// 三条区间：主维度（`m` / `bytes` / `extent` 三选一）与 `n` / `k`。
fn provide_shape(entry: &Table, index: usize, provide: &mut Provide) -> Result<(), ProfileError> {
    provide.extent = span_of(entry, index, &["m", "bytes", "extent"], EXTENT_DETAIL)?;
    provide.n = span_of(entry, index, &["n"], AXIS_DETAIL)?;
    provide.k = span_of(entry, index, &["k"], AXIS_DETAIL)?;
    Ok(())
}
/// `dt` / `space` / `grade`：前两项缺省为空集，`grade` 必填。
fn provide_sets(entry: &Table, index: usize, provide: &mut Provide) -> Result<(), ProfileError> {
    provide.dt = dtype_set(entry, index)?;
    provide.space = space_set(entry, index)?;
    provide.grade = grade_of(entry, index)?;
    Ok(())
}
/// `witness` 与其余属性：两者都只在签名写完之后才看得懂。
fn provide_tail(
    entry: &Table,
    index: usize,
    mut provide: Provide,
) -> Result<Provide, ProfileError> {
    provide.witness = witness_of(entry, index)?;
    attrs(entry, index, &mut provide)?;
    Ok(provide)
}
/// 一组同义维度：没写就是不限，多写就是错，写了就要能解析。
fn span_of(
    entry: &Table,
    index: usize,
    names: &[&str],
    detail: &str,
) -> Result<Span, ProfileError> {
    let hit = |name: &&str| entry.contains(name);
    let used: Vec<&str> = names.iter().copied().filter(hit).collect();
    let Some(alias) = used.first().copied() else {
        return Ok(Span::any());
    };
    if used.len() > 1 {
        let count = used.len();
        let aliases = "主维度有三个同义名字（`m` / `bytes` / `extent`），只能写一个";
        let detail = format!("{aliases}，这里写了 {count}");
        return Err(bad(at(index, alias), detail));
    }
    let field = at(index, alias);
    let Some(text) = entry.string(alias) else {
        return Err(bad(&field, detail));
    };
    parse_span(&field, text)
}
/// `dt`：标量或数组都接受，闭集外取值报 `Unknown`。
fn dtype_set(entry: &Table, index: usize) -> Result<DTypeSet, ProfileError> {
    let Some(value) = entry.get("dt") else {
        return Ok(DTypeSet::empty());
    };
    let path = at(index, "dt");
    let mut set = DTypeSet::empty();
    for item in names(&path, value)? {
        set.insert(closed_dtype(&path, &item)?);
    }
    Ok(set)
}
/// 一个数据类型名：`DType` 是 L1 外部类型，按它的规范名解析。
fn closed_dtype(path: &str, name: &str) -> Result<DType, ProfileError> {
    DType::parse_name(name).ok_or_else(|| unknown(path, name, &dtype_names()))
}
/// `space`：标量或数组都接受，缺省是空集（无存储层约束）。
fn space_set(entry: &Table, index: usize) -> Result<SpaceSet, ProfileError> {
    let Some(value) = entry.get("space") else {
        return Ok(SpaceSet::empty());
    };
    let path = at(index, "space");
    let owned = names(&path, value)?;
    let items: Vec<&str> = owned.iter().map(String::as_str).collect();
    parse_space_set(&path, &items)
}
/// `grade`：必填，闭集外取值报 `Unknown`。
fn grade_of(entry: &Table, index: usize) -> Result<Grade, ProfileError> {
    let path = at(index, "grade");
    closed::<Grade>(text(entry, "grade", &path)?, &path)
}
/// `witness`：内联表 `{ file, func, expect }`；只有路径的见证无法执行。
fn witness_of(entry: &Table, index: usize) -> Result<Option<Witness>, ProfileError> {
    let Some(table) = entry.table("witness") else {
        if entry.contains("witness") {
            return Err(bad(at(index, "witness"), WITNESS_DETAIL));
        }
        return Ok(None);
    };
    Ok(Some(witness_fields(table, index)?))
}
/// 见证的三个字段：`file` / `func` 必填，`expect` 是数值数组。
fn witness_fields(table: &Table, index: usize) -> Result<Witness, ProfileError> {
    let file = text(table, "file", &witness_at(index, "file"))?.to_string();
    let func = text(table, "func", &witness_at(index, "func"))?.to_string();
    let expect = expect_values(table, index)?;
    Ok(Witness { file, func, expect })
}
/// `expect`：数值数组，按顺序与见证输出对拍。
fn expect_values(table: &Table, index: usize) -> Result<Vec<f32>, ProfileError> {
    let path = witness_at(index, "expect");
    let items = table.array("expect").ok_or_else(|| missing(&path))?;
    let number = |value: &Value| expect_number(value, &path);
    items.iter().map(number).collect()
}
fn expect_number(value: &Value, path: &str) -> Result<f32, ProfileError> {
    match value {
        Value::Float(number) => Ok(*number as f32),
        Value::Int(number) => Ok(*number as f32),
        _ => Err(bad(path, "只能是数值")),
    }
}
/// 其余键一律视为枚举型属性；闭集外的键直接报错，避免"写了但没人看"。
fn attrs(entry: &Table, index: usize, provide: &mut Provide) -> Result<(), ProfileError> {
    for (key, value) in entry.entries() {
        if !parsed_key(key) {
            provide.attrs.push(one_attr(index, key, value)?);
        }
    }
    Ok(())
}
/// 已经有专属解析分支的键：再当属性处理就会重复计入（与诊断同一种 `a | b` 写法）。
fn parsed_key(key: &str) -> bool {
    let parsed = "prim|impl|m|bytes|extent|n|k|dt|space|grade|witness";
    parsed.split('|').any(|name| name == key)
}
/// 一条属性：键在 [`AttrKey`] 闭集里，取值按标量或数组展开。
fn one_attr(index: usize, key: &str, value: &Value) -> Result<Attr, ProfileError> {
    let attr_key = closed::<AttrKey>(key, &at(index, key))?;
    Ok(Attr::new(attr_key, attr_values(index, key, value)?))
}
/// 属性取值：标量或数组；表在这里没有意义。
fn attr_values(index: usize, key: &str, value: &Value) -> Result<Vec<String>, ProfileError> {
    let path = at(index, key);
    match value {
        Value::Array(items) => items.iter().map(|item| one_string(&path, item)).collect(),
        Value::Table(_) => Err(bad(path, "属性不能是表")),
        scalar => one_scalar(&path, scalar),
    }
}
/// 单个标量属性值：字符串原样，整数/浮点/布尔按字面写。
fn one_scalar(path: &str, value: &Value) -> Result<Vec<String>, ProfileError> {
    match value {
        Value::Str(text) => Ok(vec![text.clone()]),
        Value::Int(number) => Ok(vec![number.to_string()]),
        Value::Float(number) => Ok(vec![number.to_string()]),
        Value::Bool(flag) => Ok(vec![flag.to_string()]),
        _ => Err(bad(path, "属性不能是表")),
    }
}
fn one_string(path: &str, item: &Value) -> Result<String, ProfileError> {
    match item {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(bad(path, "属性数组里只能是字符串")),
    }
}
/// 区间文本：解析失败时把原话带上，字段路径由调用方给全。
fn parse_span(field: &str, text: &str) -> Result<Span, ProfileError> {
    Span::parse(text).map_err(|error| bad(field.to_string(), error.message().to_string()))
}

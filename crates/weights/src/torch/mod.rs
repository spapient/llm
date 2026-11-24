//! 职责：torch.save 检查点（`.bin`）的只读解析——ZIP 归档（STORED 条目，含 ZIP64 回退）里，
//! `data.pkl` 是 pickle 流、`data/<key>` 是各 storage 的原始字节。栈式 pickle 虚拟机解释流，
//! 拦截 `_rebuild_tensor_v2` 重建与 `BINPERSID` persistent storage，还原「名字 + 类型 + 形状 +
//! 字节区间」；原始字节按需借用，未知 opcode 与不可识别的可调用对象一律显式报错，不执行任意
//! 代码、不猜结果。
//!
//! 门面、pickle 虚拟机与 ZIP 解析原为三个独立模块（且门面↔虚拟机互为引用），现平铺进同一模块，
//! 引用环随之消除。

use diagnostic::Result;
use dtype::DType;
use storage::{bad, or_bad};

use crate::cursor::{Cursor, slice};

/// 张量目录条目：名字、元素类型、形状，加存储区在归档内的字节区间。
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) name: String,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<u64>,
    pub(crate) storage_key: String,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl Tensor {
    /// 张量名，取自 `_rebuild_tensor_v2` 所在字典的键。
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 元素类型，由 storage 类名决定。
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 形状，按书写轴序。
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

/// 一个已解析的检查点：归档句柄 + 张量目录。
pub struct Checkpoint<'a> {
    zip: Zip<'a>,
    tensors: Vec<Tensor>,
}

impl<'a> Checkpoint<'a> {
    /// 解析归档并跑完 pickle 流；`data.pkl` 的两种条目名都试。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let zip = Zip::parse(bytes)?;
        let pkl = zip.data_suffix("data.pkl");
        let pkl = pkl.or_else(|_| zip.data_suffix("/data.pkl"))?;
        let tensors = Vm::run(pkl)?;
        Ok(Self { zip, tensors })
    }

    /// 张量目录，按 pickle 流里出现的先后顺序。
    #[must_use]
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// 按名字查张量；没有该名字返回 `None`。
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// 张量字节：`/data/<key>` 与 `data/<key>` 两种条目名都试；区间越界由共享的
    /// `cursor::slice` 判据统一拒——偏移相加必须先查溢出，否则绕回小值
    /// 会让越界区间变成「空的成功切片」。
    pub fn data(&self, tensor: &Tensor) -> Result<&'a [u8]> {
        let key = &tensor.storage_key;
        let storage = self
            .zip
            .data_suffix(&format!("/data/{key}"))
            .or_else(|_| self.zip.data_suffix(&format!("data/{key}")))?;
        slice(storage, tensor.offset, tensor.len)
    }
}

// —— pickle 虚拟机（原 `pickle`，与门面互为引用，平铺后消除环）——
//
// 读取一律走 [`Cursor`]（借用调用方切片，不复制整份流）；每个 opcode 一个自由函数，整张
// [`OPS`] 表就是能力清单——表的顺序即文件里处理函数的顺序，找一条 opcode 不必跨文件。

#[derive(Clone)]
pub(crate) enum Value {
    Str(String),
    Int(i64),
    Scalar,
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global { module: String, name: String },
    Storage { key: String, dtype: DType },
    Tensor,
    Marker,
}

pub(crate) struct Vm<'a> {
    stack: Vec<Value>,
    memo: Vec<Value>,
    cursor: Cursor<'a>,
    tensors: Vec<Tensor>,
}

/// 一个 opcode 的处理函数。opcode 字节始终传入：`K`/`M`/`J`、`X`/`\x8c`、`q`/`r`/`h`/`j`
/// 都是一套处理覆盖多个码，统一签名后整张表才只是「opcode → 函数」的平表。
type Handler = fn(&mut Vm<'_>, u8) -> Result<()>;

/// opcode → 处理函数。整张表就是 pickle 的能力清单：删掉一行，那条 opcode 就退化成「未知」错误。
static OPS: [Option<Handler>; 256] = ops();

/// 把「处理函数 → opcode 集合」的平铺声明填进表：表本身就是能力清单，
/// 按协议分组写比逐条 `table[..] = Some(..)` 更贴近语义，也不易漏码。
macro_rules! ops_table {
    ($table:ident, $($handler:ident => $($code:literal),+;)+) => {
        $($(
            $table[$code as usize] = Some($handler);
        )+)+
    };
}

const fn ops() -> [Option<Handler>; 256] {
    let mut table: [Option<Handler>; 256] = [None; 256];
    ops_table!(table,
        // 协议框架：版本号 1 字节、帧长 8 字节，都不影响重建结果。
        opaque => b'\x80', b'\x95';
        // 空容器与标记：`(` 是 MARK，其余是空 dict/list 与 `()`。
        placeholder => b'(', b'}', b']', b')', b'\x89', b'\x88';
        newobj => b'\x81';
        // 文本：长度前缀 4 字节（`X`）或 1 字节（`\x8c`）。
        string => b'X', b'\x8c';
        // 浮点与全局名。
        opaque => b'G';
        global => b'c', b'\x93';
        // 整数：`K`/`M` 无符号 1/2 字节，`J` 补码 4 字节，`\x8a`/`\x8b` 长度前缀 1/4 字节。
        integer => b'K', b'M', b'J', b'\x8a', b'\x8b';
        // 元组：`t` 读 MARK 起的一段，`\x85..\x87` 按 1/2/3 个成员。
        tuple => b't', b'\x85', b'\x86', b'\x87';
        // memo 与对象图。
        memo_op => b'q', b'r', b'\x94', b'h', b'j';
        persistent_load => b'Q';
        reduce => b'R';
        // 容器写入与出栈。
        set_item => b'u', b's';
        append => b'a', b'e';
        discard => b'b';
    );
    table
}

impl<'a> Vm<'a> {
    pub(crate) fn run(pkl: &'a [u8]) -> Result<Vec<Tensor>> {
        let mut vm = Vm {
            stack: Vec::new(),
            memo: Vec::new(),
            cursor: Cursor::new(pkl),
            tensors: Vec::new(),
        };
        vm.execute()?;
        Ok(vm.tensors)
    }

    fn execute(&mut self) -> Result<()> {
        loop {
            let op = self.cursor.byte()?;
            if op == b'.' {
                return Ok(());
            }
            self.dispatch(op)?;
        }
    }

    /// 查表并调用处理函数；未知 opcode 在这里统一拒。
    fn dispatch(&mut self, op: u8) -> Result<()> {
        let handler =
            OPS[usize::from(op)].ok_or_else(|| bad(format!("未知 pickle opcode 0x{op:02X}")))?;
        handler(self, op)
    }

    fn pops(&mut self, n: usize) -> Result<Vec<Value>> {
        let at = self
            .stack
            .len()
            .checked_sub(n)
            .ok_or_else(|| bad("栈下溢"))?;
        Ok(self.stack.split_off(at))
    }

    fn pop(&mut self) -> Result<Value> {
        or_bad(self.stack.pop(), "栈下溢")
    }

    fn pop_until_marker(&mut self) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        loop {
            match self.pop()? {
                Value::Marker => {
                    items.reverse();
                    return Ok(items);
                }
                other => items.push(other),
            }
        }
    }

    fn store_memo(&mut self, idx: usize) -> Result<()> {
        let value = or_bad(self.stack.last(), "空栈无法写 memo")?.clone();
        // memo 槽位由更早的存槽 opcode 占用，索引不可能超过流长度；谎报大索引先拒绝，避免稀疏巨表。
        if idx > self.cursor.total() {
            return Err(bad("memo 索引超出 pickle 流长度"));
        }
        if idx >= self.memo.len() {
            self.memo.resize(idx + 1, Value::Scalar);
        }
        self.memo[idx] = value;
        Ok(())
    }

    fn recall_memo(&mut self, idx: usize) -> Result<()> {
        let value = or_bad(self.memo.get(idx), "memo 索引缺失")?.clone();
        self.stack.push(value);
        Ok(())
    }

    /// `REDUCE` 的张量重建：实参 → 张量，名字取栈顶，并把 `Tensor` 占位压回栈。
    fn rebuild_tensor(&mut self, args: Value) -> Result<()> {
        let items = as_tuple(args, "rebuild 参数不是元组")?;
        let mut tensor = tensor_of(&items)?;
        tensor.name = self.tensor_name();
        self.tensors.push(tensor);
        self.stack.push(Value::Tensor);
        Ok(())
    }

    /// 张量名：`REDUCE` 的实参与可调用对象弹出后，栈顶若是字符串，它就是 `state_dict`
    /// 里紧邻这个值的键；否则按出现序号命名，避免借用上一个张量残留的键把两个张量
    /// 叫成同一个名字（`torch.save([t1, t2])` 就是这一形状）。
    fn tensor_name(&self) -> String {
        match self.stack.last() {
            Some(Value::Str(name)) => name.clone(),
            _ => self.tensors.len().to_string(),
        }
    }

    fn line(&mut self) -> Result<String> {
        let len = or_bad(
            self.cursor.tail().iter().position(|&byte| byte == b'\n'),
            "GLOBAL 缺换行",
        )?;
        let text = self.cursor.text(len)?;
        self.cursor.byte()?;
        Ok(text)
    }
}

/// 不参与重建、但字节必须完整存在的定宽字段：`PROTO` 1 字节版本号、`FRAME` 8 字节帧长、
/// `BINFLOAT` 8 字节大端双精度；只有 `BINFLOAT` 往栈上留一个占位。
fn opaque(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    vm.cursor
        .take(if op == b'\x80' { 1 } else { 8 })
        .map(drop)?;
    if op == b'G' {
        vm.stack.push(Value::Scalar);
    }
    Ok(())
}

fn newobj(vm: &mut Vm<'_>, _op: u8) -> Result<()> {
    vm.pop()?;
    vm.pop()?;
    Err(bad("NEWOBJ 不支持的类"))
}

fn placeholder(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let value = match op {
        b'}' => Value::Dict(Vec::new()),
        b']' => Value::List(Vec::new()),
        b'(' => Value::Marker,
        _ => Value::Scalar,
    };
    vm.stack.push(value);
    Ok(())
}

fn string(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let len = usize::try_from(vm.cursor.uint(if op == b'X' { 4 } else { 1 })?)
        .map_err(|_| bad("字符串长度溢出"))?;
    let text = vm.cursor.text(len)?;
    vm.stack.push(Value::Str(text));
    Ok(())
}

/// 全局名：模块在前、名字在后。`GLOBAL` 直接读两行（模块、类名）；`STACK_GLOBAL` 的两个
/// 字符串压栈时是「模块、类名」，所以弹出的第一个是类名、第二个才是模块。
fn global(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let (module, name) = match op {
        b'c' => line_global(vm)?,
        _ => stacked_global(vm)?,
    };
    vm.stack.push(Value::Global { module, name });
    Ok(())
}

/// `GLOBAL`：模块与类名各占一行。
fn line_global(vm: &mut Vm<'_>) -> Result<(String, String)> {
    Ok((vm.line()?, vm.line()?))
}

/// `STACK_GLOBAL`：先弹的是类名，后弹的才是模块。
fn stacked_global(vm: &mut Vm<'_>) -> Result<(String, String)> {
    let name = into_str(&vm.pop()?)?;
    Ok((into_str(&vm.pop()?)?, name))
}

/// 整数：`BININT1`/`BININT2` 无符号 1/2 字节、`BININT` 补码 4 字节、`LONG1`/`LONG4`
/// 1/4 字节长度前缀 + 小端补码正文。
fn integer(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let value = match op {
        b'M' => widen(vm, 2, false),
        b'J' => widen(vm, 4, true),
        b'\x8a' | b'\x8b' => long_width(vm, op).and_then(|width| widen(vm, width, true)),
        _ => widen(vm, 1, false),
    }?;
    vm.stack.push(Value::Int(value));
    Ok(())
}

/// `LONG1`/`LONG4` 的长度前缀：1 或 4 字节。
fn long_width(vm: &mut Vm<'_>, op: u8) -> Result<usize> {
    let size = if op == b'\x8b' { 4 } else { 1 };
    let len = vm.cursor.uint(size)?;
    usize::try_from(len).map_err(|_| bad("整数长度超出 usize"))
}

/// 定宽小端整数：`signed` 按最高位符号扩展，否则零扩展；超过 8 字节装不进 `i64`，显式失败。
fn widen(vm: &mut Vm<'_>, width: usize, signed: bool) -> Result<i64> {
    if width > 8 {
        return Err(bad("整数超过 i64 宽度"));
    }
    let raw = vm.cursor.take(width)?;
    let negative = signed && raw.last().is_some_and(|&last| last & 0x80 != 0);
    let mut buf = [if negative { 0xFF } else { 0 }; 8];
    buf[..width].copy_from_slice(raw);
    Ok(i64::from_le_bytes(buf))
}

fn tuple(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let items = match op {
        b't' => vm.pop_until_marker()?,
        _ => vm.pops(usize::from(op - 0x84))?,
    };
    vm.stack.push(Value::Tuple(items));
    Ok(())
}

/// `POP`：丢弃栈顶。
fn discard(vm: &mut Vm<'_>, _op: u8) -> Result<()> {
    vm.pop().map(drop)
}

/// memo 的 5 条 opcode：`q`/`r`/`\x94` 存当前栈顶，`h`/`j` 取回。索引宽度随 opcode 变：
/// `q`/`h` 读 1 字节，`r`/`j` 读 4 字节，`\x94` 直接用当前栈长。
fn memo_op(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let idx = match op {
        b'q' | b'h' => memo_idx(vm, 1)?,
        b'r' | b'j' => memo_idx(vm, 4)?,
        _ => vm.memo.len(),
    };
    recall(vm, idx, matches!(op, b'h' | b'j'))
}

/// memo 索引：`size` 字节无符号。
fn memo_idx(vm: &mut Vm<'_>, size: usize) -> Result<usize> {
    usize::try_from(vm.cursor.uint(size)?).map_err(|_| bad("memo 索引超出 usize"))
}

/// 取回（`h`/`j`）还是存入（`q`/`r`/`\x94`）。
fn recall(vm: &mut Vm<'_>, idx: usize, get: bool) -> Result<()> {
    if get {
        return vm.recall_memo(idx);
    }
    vm.store_memo(idx)
}

/// `BINPERSID`：栈顶是 persistent id 元组，换成 [`Value::Storage`] 留在栈上。
fn persistent_load(vm: &mut Vm<'_>, _op: u8) -> Result<()> {
    let items = as_tuple(vm.pop()?, "persistent id 不是元组")?;
    let (key, dtype) = storage_handle(&items)?;
    vm.stack.push(Value::Storage { key, dtype });
    Ok(())
}

/// `REDUCE`：弹出实参与可调用对象，只认张量重建与 `OrderedDict` 两种构造——
/// 可调用对象必须是 `GLOBAL`，其余一律显式失败。
fn reduce(vm: &mut Vm<'_>, _op: u8) -> Result<()> {
    let args = vm.pop()?;
    let Value::Global { module, name } = vm.pop()? else {
        return Err(bad("REDUCE 的可调用对象不是 GLOBAL"));
    };
    invoke(vm, &module, &name, args)
}

/// 两种可调用对象：张量重建与空 `OrderedDict`；其余一律显式失败。
fn invoke(vm: &mut Vm<'_>, module: &str, name: &str, args: Value) -> Result<()> {
    match (module, name) {
        ("torch._utils", "_rebuild_tensor_v2") => vm.rebuild_tensor(args),
        ("collections", "OrderedDict") => {
            vm.stack.push(Value::Dict(Vec::new()));
            Ok(())
        }
        _ => Err(bad(format!("不支持的可调用对象 {module}.{name}"))),
    }
}

/// 栈顶 dict 写入：`SETITEM` 弹一队键值，`SETITEMS` 弹 MARK 起的一段（项数必须为偶数）。
fn set_item(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let (pairs, what) = if op == b's' {
        let value = vm.pop()?;
        (vec![(vm.pop()?, value)], "SETITEM")
    } else {
        (split_items(vm)?, "SETITEMS")
    };
    write_dict(vm, pairs, what)
}

/// `SETITEMS` 的键值对：MARK 起的一段，项数必须为偶数。
fn split_items(vm: &mut Vm<'_>) -> Result<Vec<(Value, Value)>> {
    let items = vm.pop_until_marker()?;
    if !items.len().is_multiple_of(2) {
        return Err(bad("SETITEMS 的项数为奇数"));
    }
    Ok(items
        .chunks(2)
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect())
}

/// 栈顶必须是 dict，否则按 `what` 报错。
fn write_dict(vm: &mut Vm<'_>, pairs: Vec<(Value, Value)>, what: &str) -> Result<()> {
    match vm.stack.last_mut() {
        Some(Value::Dict(entries)) => {
            entries.extend(pairs);
            Ok(())
        }
        _ => Err(bad(format!("{what} 目标不是 dict"))),
    }
}

/// 栈顶 list 追加：`APPEND` 弹一值，`APPENDS` 弹 MARK 起的一段。
fn append(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let (items, label) = if op == b'a' {
        (vec![vm.pop()?], "APPEND")
    } else {
        (vm.pop_until_marker()?, "APPENDS")
    };
    match vm.stack.last_mut() {
        Some(Value::List(list)) => {
            list.extend(items);
            Ok(())
        }
        _ => Err(bad(format!("{label} 目标不是 list"))),
    }
}

// —— torch 张量与存储句柄：`BINPERSID` 三元组与 `_rebuild_tensor_v2` 五要素解析；
// 不持有虚拟机状态，输入输出都是 [`Value`]/[`Tensor`]。两个入口各管一件事：`storage_handle`
// 解 `BINPERSID` 的三元组，`tensor_of` 解 `_rebuild_tensor_v2` 实参的五要素。布局判据只有
// `contiguous` 一条：非连续张量无法用单个连续字节区间表示，显式失败好过按连续布局读出错误字节。

/// 元组视图：`REDUCE` 的实参与 persistent id 都必须是元组。
pub(crate) fn as_tuple(value: Value, message: &str) -> Result<Vec<Value>> {
    match value {
        Value::Tuple(items) => Ok(items),
        _ => Err(bad(message)),
    }
}

/// 字符串视图。
pub(crate) fn into_str(value: &Value) -> Result<String> {
    match value {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(bad("期望字符串")),
    }
}

/// persistent id：`("storage", <类名>, <key>)` → 存库名与元素类型。
pub(crate) fn storage_handle(items: &[Value]) -> Result<(String, DType)> {
    tagged(items)?;
    let class = storage_class(items)?;
    Ok((text_at(items, 2, "key")?, storage_dtype(class)?))
}

/// `_rebuild_tensor_v2` 的实参 → 张量：五要素齐备且布局连续。
pub(crate) fn tensor_of(items: &[Value]) -> Result<Tensor> {
    let (key, dtype, wide) = head(items)?;
    let (shape, count) = layout(items)?;
    let width = dtype.bytes_for(1) as usize;
    let offset = scaled(usize::try_from(wide).ok(), width, "offset 溢出")?;
    Ok(Tensor {
        name: String::new(),
        dtype,
        shape,
        storage_key: key.clone(),
        offset,
        len: scaled(Some(count), width, "长度溢出")?,
    })
}

/// pid 的首项必须是 `"storage"`：别种 persistent id 本模块不认识，静默当空会更糟。
fn tagged(items: &[Value]) -> Result<()> {
    let Some(Value::Str(tag)) = items.first() else {
        return Err(bad("pid 空"));
    };
    match tag.as_str() {
        "storage" => Ok(()),
        other => Err(bad(format!("不支持的 persistent id `{other}`"))),
    }
}

/// pid 的 storage 类名。
fn storage_class(items: &[Value]) -> Result<&String> {
    let Some(Value::Global { name: class, .. }) = items.get(1) else {
        return Err(bad("pid 缺 storage 类"));
    };
    Ok(class)
}

/// storage 类名 → 元素类型。
fn storage_dtype(class: &str) -> Result<DType> {
    STORAGE_DTYPES
        .iter()
        .find_map(|&(name, dtype)| (name == class).then_some(dtype))
        .ok_or_else(|| bad(format!("不支持的 storage 类型 `{class}`")))
}

/// 第 `idx` 项的字符串视图。
fn text_at(items: &[Value], idx: usize, what: &str) -> Result<String> {
    into_str(field(items, idx, what)?)
}

/// rebuild 的前三要素：storage 键、元素类型、offset 原文。
fn head(items: &[Value]) -> Result<(&String, DType, i64)> {
    let Some(Value::Storage { key, dtype }) = items.first() else {
        return Err(bad("rebuild 缺 storage"));
    };
    let Some(Value::Int(wide)) = items.get(1) else {
        return Err(bad("rebuild 缺 offset"));
    };
    Ok((key, *dtype, *wide))
}

/// 后两要素：形状与 stride 推出元素总数。
fn layout(items: &[Value]) -> Result<(Vec<u64>, usize)> {
    let shape = shape(items)?;
    let steps = strides(items)?;
    let count = contiguous(&shape, &steps)?;
    Ok((shape, count))
}

fn shape(items: &[Value]) -> Result<Vec<u64>> {
    tuple_at(items, 2, "shape")?.iter().map(dim).collect()
}

fn strides(items: &[Value]) -> Result<Vec<usize>> {
    tuple_at(items, 3, "stride")?.iter().map(step).collect()
}

/// 元素数 × 字节宽：任何一步失败都是同一句"溢出"。
fn scaled(count: Option<usize>, width: usize, what: &str) -> Result<usize> {
    or_bad(count.and_then(|n| n.checked_mul(width)), what)
}

/// 第 `idx` 项。
fn field<'v>(items: &'v [Value], idx: usize, missing: &str) -> Result<&'v Value> {
    items
        .get(idx)
        .ok_or_else(|| bad(format!("rebuild 缺 {missing}")))
}

/// 第 `idx` 项元组视图。
fn tuple_at<'v>(items: &'v [Value], idx: usize, what: &str) -> Result<&'v [Value]> {
    match field(items, idx, what)? {
        Value::Tuple(fields) => Ok(fields),
        _ => Err(bad(format!("rebuild 缺 {what}"))),
    }
}

/// 形状维度：非负整数，负数与错型分别是溢出与类型错误。
fn dim(value: &Value) -> Result<u64> {
    match value {
        Value::Int(wide) => u64::try_from(*wide).map_err(|_| bad("形状维度溢出")),
        _ => Err(bad("形状维度不是整数")),
    }
}

/// stride 与维度是同一概念（非负整数），只多一步收窄到 `usize`。
fn step(value: &Value) -> Result<usize> {
    let wide = dim(value).map_err(|_| bad("stride 不是非负整数"))?;
    usize::try_from(wide).map_err(|_| bad("stride 溢出"))
}

/// row-major 连续判据：尺寸为 1 的维不参与（torch 允许其 stride 任意）；返回元素总数。
fn contiguous(shape: &[u64], steps: &[usize]) -> Result<usize> {
    if shape.len() != steps.len() {
        return Err(bad("stride 与 shape 维数不符"));
    }
    let mut count = 1usize;
    for (&size, &step) in shape.iter().zip(steps).rev() {
        count = advance(count, size, step)?;
    }
    Ok(count)
}

/// 单一维的连续校验与累乘：尺寸为 1 跳过 stride 校验，乘积溢出显式失败。
fn advance(count: usize, size: u64, step: usize) -> Result<usize> {
    let width = usize::try_from(size).map_err(|_| bad("形状维度溢出"))?;
    if width > 1 && step != count {
        return Err(bad(format!(
            "非连续张量：stride {step}，连续布局应为 {count}"
        )));
    }
    or_bad(count.checked_mul(width), "形状乘积溢出")
}

/// torch 的 storage 类名 → 元素类型（Torch 与 `dtype` 的命名对应表）。
const STORAGE_DTYPES: [(&str, DType); 9] = [
    ("FloatStorage", DType::F32),
    ("HalfStorage", DType::F16),
    ("BFloat16Storage", DType::BF16),
    ("DoubleStorage", DType::F64),
    ("LongStorage", DType::I64),
    ("IntStorage", DType::I32),
    ("ShortStorage", DType::I16),
    ("CharStorage", DType::I8),
    ("ByteStorage", DType::U8),
];

pub(crate) mod zip;

pub(crate) use zip::Zip;
#[cfg(test)]
pub(crate) use zip::zip64_extra;

#[cfg(test)]
#[path = "../../tests/torch.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/pickle.rs"]
mod pickle_tests;

#[cfg(test)]
#[path = "../../tests/zip.rs"]
mod zip_tests;

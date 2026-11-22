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
use wire::{bad, or_bad};

use crate::cursor::{slice, u16_at, u32_at, u64_at, Cursor};

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
enum Value {
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

const fn ops() -> [Option<Handler>; 256] {
    let mut table: [Option<Handler>; 256] = [None; 256];
    // 协议框架：版本号 1 字节、帧长 8 字节，都不影响重建结果。
    table[b'\x80' as usize] = Some(opaque);
    table[b'\x95' as usize] = Some(opaque);
    // 空容器与标记：`(` 是 MARK，其余是空 dict/list 与 `()`。
    table[b'(' as usize] = Some(placeholder);
    table[b'}' as usize] = Some(placeholder);
    table[b']' as usize] = Some(placeholder);
    table[b')' as usize] = Some(placeholder);
    table[b'\x89' as usize] = Some(placeholder);
    table[b'\x88' as usize] = Some(placeholder);
    table[b'\x81' as usize] = Some(newobj);
    // 文本：长度前缀 4 字节（`X`）或 1 字节（`\x8c`）。
    table[b'X' as usize] = Some(string);
    table[b'\x8c' as usize] = Some(string);
    // 浮点与全局名。
    table[b'G' as usize] = Some(opaque);
    table[b'c' as usize] = Some(global);
    table[b'\x93' as usize] = Some(global);
    // 整数：`K`/`M` 无符号 1/2 字节，`J` 补码 4 字节，`\x8a`/`\x8b` 长度前缀 1/4 字节。
    table[b'K' as usize] = Some(integer);
    table[b'M' as usize] = Some(integer);
    table[b'J' as usize] = Some(integer);
    table[b'\x8a' as usize] = Some(integer);
    table[b'\x8b' as usize] = Some(integer);
    // 元组：`t` 读 MARK 起的一段，`\x85..\x87` 按 1/2/3 个成员。
    table[b't' as usize] = Some(tuple);
    table[b'\x85' as usize] = Some(tuple);
    table[b'\x86' as usize] = Some(tuple);
    table[b'\x87' as usize] = Some(tuple);
    // memo 与对象图。
    table[b'q' as usize] = Some(memo_op);
    table[b'r' as usize] = Some(memo_op);
    table[b'\x94' as usize] = Some(memo_op);
    table[b'h' as usize] = Some(memo_op);
    table[b'j' as usize] = Some(memo_op);
    table[b'Q' as usize] = Some(persistent_load);
    table[b'R' as usize] = Some(reduce);
    // 容器写入与出栈。
    table[b'u' as usize] = Some(set_item);
    table[b's' as usize] = Some(set_item);
    table[b'a' as usize] = Some(append);
    table[b'e' as usize] = Some(append);
    table[b'b' as usize] = Some(discard);
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

// ---- torch 张量与存储句柄 ----
//
// 两个入口各管一件事：`storage_handle` 解 `BINPERSID` 的三元组，`tensor_of` 解
// `_rebuild_tensor_v2` 实参的五要素（storage、offset、shape、stride、元素总数）。布局判据
// 只有 `contiguous` 一条：非连续张量无法用单个连续字节区间表示，显式失败好过按连续
// 布局读出错误字节。

/// 元组视图：`REDUCE` 的实参与 persistent id 都必须是元组。
fn as_tuple(value: Value, message: &str) -> Result<Vec<Value>> {
    match value {
        Value::Tuple(items) => Ok(items),
        _ => Err(bad(message)),
    }
}

/// 字符串视图。
fn into_str(value: &Value) -> Result<String> {
    match value {
        Value::Str(text) => Ok(text.clone()),
        _ => Err(bad("期望字符串")),
    }
}

/// persistent id：`("storage", <类名>, <key>)` → 存库名与元素类型。
fn storage_handle(items: &[Value]) -> Result<(String, DType)> {
    tagged(items)?;
    let class = storage_class(items)?;
    Ok((text_at(items, 2, "key")?, storage_dtype(class)?))
}

/// `_rebuild_tensor_v2` 的实参 → 张量：五要素齐备且布局连续。
fn tensor_of(items: &[Value]) -> Result<Tensor> {
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
// —— ZIP 归档解析（原 `zip`）：STORED 条目，含 ZIP64 回退 ——
//
// 每个函数对应 APPNOTE 的一段：`directory`（尾部 EOCD 或 ZIP64 回退）、`entry`（一条中央目录
// 记录）、`sizes`（32 位尺寸字段与其 ZIP64 槽位）、`zip64_extra`（extra 字段头遍历）。

const EOCD_SIG: u32 = 0x0605_4B50;
const CD_SIG: u32 = 0x0201_4B50;
const LOCAL_SIG: u32 = 0x0403_4B50;
const ZIP64_SIG: u32 = 0x0606_4B50;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4B50;
/// extra 字段里 ZIP64 的 header id。
const ZIP64_EXTRA_ID: u16 = 0x0001;
const STORED: u16 = 0;
/// 条目最小 46+1 字节；预分配上限 4096，谎报计数先按剩余字节粗查。
const MIN_ENTRY_BYTES: usize = 47;
const MAX_ENTRY_PREALLOC: usize = 4096;
/// 中央目录尺寸字段被截断时的 32 位哨兵。
const WIDE_SENTINEL: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub(crate) struct ZipEntry {
    name: String,
    method: u16,
    start: usize,
    len: usize,
}

pub(crate) struct Zip<'a> {
    bytes: &'a [u8],
    entries: Vec<ZipEntry>,
}

impl<'a> Zip<'a> {
    pub(crate) fn parse(bytes: &'a [u8]) -> Result<Self> {
        let (mut at, total) = directory(bytes)?;
        if total > bytes.len() / MIN_ENTRY_BYTES {
            return Err(bad("条目数超出归档能容纳的规模"));
        }
        let mut entries = Vec::with_capacity(total.min(MAX_ENTRY_PREALLOC));
        for _ in 0..total {
            let (entry, next) = entry(bytes, at)?;
            entries.push(entry);
            at = next;
        }
        Ok(Self { bytes, entries })
    }

    #[must_use]
    pub(crate) fn get_suffix(&self, suffix: &str) -> Option<&ZipEntry> {
        self.entries.iter().find(|e| e.name.ends_with(suffix))
    }

    pub(crate) fn data_suffix(&self, suffix: &str) -> Result<&'a [u8]> {
        let entry = self
            .get_suffix(suffix)
            .ok_or_else(|| bad(format!("无后缀条目 `{suffix}`")))?;
        self.data(entry)
    }

    fn data(&self, entry: &ZipEntry) -> Result<&'a [u8]> {
        if entry.method != STORED {
            return Err(bad("条目经过压缩，零依赖读取器只支持 STORED"));
        }
        slice(self.bytes, entry.start, entry.len)
    }
}

/// 中央目录起点与条目总数：EOCD 的两个字段任一为 `0xFFFF…` 就走 ZIP64 回退。
fn directory(bytes: &[u8]) -> Result<(usize, usize)> {
    let eocd = locate_eocd(bytes)?;
    narrow_or_wide(
        bytes,
        eocd,
        u32_at(bytes, eocd + 16)?,
        u16_at(bytes, eocd + 10)?,
    )
}

/// 32 位字段够用就按它换算，任一被截断（`0xFFFF…`）就走 ZIP64 回退。
fn narrow_or_wide(bytes: &[u8], eocd: usize, wide: u32, total: u16) -> Result<(usize, usize)> {
    if wide == WIDE_SENTINEL || total == u16::MAX {
        return zip64_directory(bytes, eocd);
    }
    Ok((
        narrow(u64::from(wide), "中央目录偏移超出 usize")?,
        usize::from(total),
    ))
}

/// ZIP64 回退：定位器在 EOCD 前 20 字节，定位器 +8 指向 64 位 EOCD；
/// 条目数与该 EOCD 的 +32、中央目录偏移在 +48。
fn zip64_directory(bytes: &[u8], eocd: usize) -> Result<(usize, usize)> {
    let z64 = zip64_eocd(bytes, eocd)?;
    let total = wide_value(bytes, z64 + 32, "条目数超出 usize")?;
    let at = wide_value(bytes, z64 + 48, "中央目录偏移超出 usize")?;
    Ok((at, total))
}

/// 定位器的第一跳：它在 EOCD 前 20 字节，定位器 +8 指向 64 位 EOCD。
fn zip64_eocd(bytes: &[u8], eocd: usize) -> Result<usize> {
    let locator = locator_at(eocd)?;
    check_sig(bytes, locator, EOCD64_LOCATOR_SIG, "ZIP64 定位器签名不符")?;
    let z64 = wide_value(bytes, locator + 8, "ZIP64 EOCD 偏移超出 usize")?;
    check_sig(bytes, z64, ZIP64_SIG, "ZIP64 EOCD 签名不符")?;
    Ok(z64)
}

/// 定位器起点：EOCD 前 20 字节，越界说明这个归档没有 ZIP64 段。
fn locator_at(eocd: usize) -> Result<usize> {
    match eocd.checked_sub(20) {
        Some(at) => Ok(at),
        None => Err(bad("ZIP64 定位器不完整")),
    }
}

/// 一个 64 位字段换算成 `usize`：越界即错，不截断。
fn wide_value(bytes: &[u8], at: usize, msg: &str) -> Result<usize> {
    narrow(u64_at(bytes, at)?, msg)
}

/// 64 位真值收窄成 `usize`：收不下就显式失败。
fn narrow(value: u64, msg: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| bad(msg))
}

fn check_sig(bytes: &[u8], at: usize, sig: u32, msg: &str) -> Result<()> {
    if u32_at(bytes, at)? == sig {
        return Ok(());
    }
    Err(bad(msg))
}

/// 一条中央目录记录：名字、方法、数据区间与下一条记录的起点。
/// 记录长 46 字节 + 名字 + extra + 注释依次排开；区间必须落在归档内，否则显式失败。
fn entry(bytes: &[u8], at: usize) -> Result<(ZipEntry, usize)> {
    let lens = Lens::of(cd_head(bytes, at)?)?;
    let (csize, local) = sizes(bytes, at, lens.name)?;
    build(bytes, at, lens, csize, local)
}

/// 固定 46 字节头：签名对得上才有后面的字段，所以先整块切出来。
fn cd_head(bytes: &[u8], at: usize) -> Result<&[u8]> {
    let fixed = slice(bytes, at, 46)?;
    check_sig(fixed, 0, CD_SIG, "中央目录条目签名不符")?;
    Ok(fixed)
}

/// 装配一条记录：名字、数据区间与下一条记录的起点。
fn build(
    bytes: &[u8],
    at: usize,
    lens: Lens,
    csize: u64,
    local: usize,
) -> Result<(ZipEntry, usize)> {
    let name = entry_name(bytes, at, lens.name)?;
    let (start, len) = span(bytes, local, csize)?;
    let next = lens.next(at)?;
    Ok((
        ZipEntry {
            name,
            method: lens.method,
            start,
            len,
        },
        next,
    ))
}

/// 固定头里的方法码与三段长度：条目名、extra、注释依次排开。
struct Lens {
    method: u16,
    name: usize,
    extra: usize,
    comment: usize,
}

impl Lens {
    fn of(fixed: &[u8]) -> Result<Self> {
        Ok(Self {
            method: u16_at(fixed, 10)?,
            name: usize::from(u16_at(fixed, 28)?),
            extra: usize::from(u16_at(fixed, 30)?),
            comment: usize::from(u16_at(fixed, 32)?),
        })
    }

    /// 下一条记录的起点：46 字节固定头 + 名字 + extra + 注释。
    fn next(&self, at: usize) -> Result<usize> {
        match at.checked_add(46 + self.name + self.extra + self.comment) {
            Some(next) => Ok(next),
            None => Err(bad("中央目录偏移溢出")),
        }
    }
}

/// 条目名：中央目录里的名字是 UTF-8 闭集，收不下就是归档坏了。
fn entry_name(bytes: &[u8], at: usize, len: usize) -> Result<String> {
    String::from_utf8(slice(bytes, at + 46, len)?.to_vec()).map_err(|_| bad("条目名不是 UTF-8"))
}

/// 数据区间（起点、长度）：本地头给出起点，区间必须落在归档内。
fn span(bytes: &[u8], local: usize, csize: u64) -> Result<(usize, usize)> {
    let start = local_start(bytes, local)?;
    let len = narrow(csize, "条目长度超出 usize")?;
    if start.checked_add(len).is_none_or(|stop| stop > bytes.len()) {
        return Err(bad("条目数据超出归档"));
    }
    Ok((start, len))
}

/// 中央目录的尺寸字段：32 位真值，或被截断成 `0xFFFF_FFFF` 时回退到 ZIP64 extra。
/// 真值按 APPNOTE 的固定顺序（原长、压缩长、本地头偏移）落在 extra 里，且**只**收下被
/// 截断的那些——所以槽位是「前面被截断的字段个数」，不是字段序号。返回（压缩长，本地头偏移）。
fn sizes(bytes: &[u8], at: usize, name_len: usize) -> Result<(u64, usize)> {
    let extra = at + 46 + name_len;
    let extra_len = usize::from(u16_at(bytes, at + 30)?);
    let mut slot = 0usize;
    let mut wide = [0u64; 3];
    for (idx, offset) in [24, 20, 42].into_iter().enumerate() {
        wide[idx] = wide_field(
            bytes,
            at + offset,
            Extra {
                extra,
                extra_len,
                slot: &mut slot,
            },
        )?;
    }
    Ok((wide[1], narrow(wide[2], "本地头偏移超出 usize")?))
}

/// ZIP64 extra 字段体的位置、声明长度与「前面截断过几个字段」的槽位。
struct Extra<'a> {
    extra: usize,
    extra_len: usize,
    slot: &'a mut usize,
}

/// 一个尺寸字段：32 位真值，或被截断成 `0xFFFF_FFFF` 时按槽位读 ZIP64 extra。
fn wide_field(bytes: &[u8], at: usize, extra: Extra<'_>) -> Result<u64> {
    let narrow = u32_at(bytes, at)?;
    if narrow != WIDE_SENTINEL {
        return Ok(u64::from(narrow));
    }
    let slot = *extra.slot;
    *extra.slot = slot + 1;
    zip64_extra(bytes, extra.extra, extra.extra_len, slot)
}

/// 本地头起点：签名对得上，30 字节固定段 + 名字 + extra 之后就是数据。
fn local_start(bytes: &[u8], local: usize) -> Result<usize> {
    check_sig(bytes, local, LOCAL_SIG, "本地头签名不符")?;
    let name_len = usize::from(u16_at(bytes, local + 26)?);
    let extra_len = usize::from(u16_at(bytes, local + 28)?);
    let Some(start) = local.checked_add(30 + name_len + extra_len) else {
        return Err(bad("本地头偏移溢出"));
    };
    Ok(start)
}

/// ZIP64 extra 里第 `field` 个 8 字节真值：逐个 extra 字段头（id u16 + 长 u16）往前挪，
/// 找到 id 0x0001 就按槽位取；声明长度装不下该槽位显式失败。
fn zip64_extra(bytes: &[u8], extra_start: usize, extra_len: usize, field: usize) -> Result<u64> {
    let stop = extra_start.saturating_add(extra_len);
    let mut at = extra_start;
    while at.checked_add(4).is_some_and(|end| end <= stop) {
        let id = u16_at(bytes, at)?;
        let size = usize::from(u16_at(bytes, at + 2)?);
        let body = at + 4;
        if id == ZIP64_EXTRA_ID {
            return slot_value(bytes, body, size, field);
        }
        at = body.saturating_add(size);
    }
    Err(bad("缺少 ZIP64 extra 字段"))
}

/// 第 `field` 个 8 字节真值：声明长度装不下该槽位就显式失败（截断不许读成 0）。
fn slot_value(bytes: &[u8], body: usize, size: usize, field: usize) -> Result<u64> {
    let end = field
        .checked_add(1)
        .and_then(|slots| slots.checked_mul(8))
        .ok_or_else(|| bad("ZIP64 字段序号溢出"))?;
    if end > size {
        return Err(bad("ZIP64 extra 声明长度不足，字段被截断"));
    }
    u64_at(bytes, body.saturating_add(end - 8))
}

/// EOCD 在文件尾部 22 + 65535 字节窗口内反向查找；窗口外一律拒（不猜其他签名）。
fn locate_eocd(bytes: &[u8]) -> Result<usize> {
    let from = bytes.len().saturating_sub(22 + 65535);
    let mut at = bytes.len().saturating_sub(4);
    while at >= from {
        if u32_at(bytes, at)? == EOCD_SIG {
            return Ok(at);
        }
        if at == 0 {
            break;
        }
        at -= 1;
    }
    Err(bad("找不到 ZIP 签名，可能不是 ZIP 归档"))
}

#[cfg(test)]
#[path = "../tests/torch.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/pickle.rs"]
mod pickle_tests;

#[cfg(test)]
#[path = "../tests/zip.rs"]
mod zip_tests;
//! 职责：栈式 pickle 虚拟机——解释 `data.pkl` 流，`_rebuild_tensor_v2` 重建与
//! `BINPERSID` persistent storage 在此被拦截，还原「名字 + 类型 + 形状 + 字节区间」。
//!
//! 读取一律走 [`Cursor`]（借用调用方切片，不复制整份流）；每个 opcode 一个自由函数，整张
//! [`OPS`] 表就是能力清单——表的顺序即文件里处理函数的顺序，找一条 opcode 不必跨文件。

mod tensor;

use diagnostic::Result;
use dtype::DType;
use wire::{bad, or_bad};

use crate::cursor::Cursor;
use crate::torch::Tensor;
use tensor::{as_tuple, into_str, storage_handle, tensor_of};

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
    pub(crate) stack: Vec<Value>,
    pub(crate) memo: Vec<Value>,
    pub(crate) cursor: Cursor<'a>,
    pub(crate) tensors: Vec<Tensor>,
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

    pub(crate) fn execute(&mut self) -> Result<()> {
        loop {
            let op = self.cursor.byte()?;
            if op == b'.' {
                return Ok(());
            }
            self.dispatch(op)?;
        }
    }

    /// 查表并调用处理函数；未知 opcode 在这里统一拒。
    pub(crate) fn dispatch(&mut self, op: u8) -> Result<()> {
        let handler =
            OPS[usize::from(op)].ok_or_else(|| bad(format!("未知 pickle opcode 0x{op:02X}")))?;
        handler(self, op)
    }

    pub(crate) fn pops(&mut self, n: usize) -> Result<Vec<Value>> {
        let at = self
            .stack
            .len()
            .checked_sub(n)
            .ok_or_else(|| bad("栈下溢"))?;
        Ok(self.stack.split_off(at))
    }

    pub(crate) fn pop(&mut self) -> Result<Value> {
        or_bad(self.stack.pop(), "栈下溢")
    }

    pub(crate) fn pop_until_marker(&mut self) -> Result<Vec<Value>> {
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
    pub(crate) fn tensor_name(&self) -> String {
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
        b'q' | b'h' => memo_index(vm, 1)?,
        b'r' | b'j' => memo_index(vm, 4)?,
        _ => vm.memo.len(),
    };
    recall(vm, idx, matches!(op, b'h' | b'j'))
}

/// memo 索引：`size` 字节无符号。
fn memo_index(vm: &mut Vm<'_>, size: usize) -> Result<usize> {
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
pub(crate) fn append(vm: &mut Vm<'_>, op: u8) -> Result<()> {
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

#[cfg(test)]
#[path = "../tests/pickle.rs"]
mod tests;

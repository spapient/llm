//! 职责：栈式 pickle 虚拟机——解释 `data.pkl` 流，拦截 `_rebuild_tensor_v2` 重建与
//! `BINPERSID` persistent storage，还原「名字 + 类型 + 形状 + 字节区间」。
//!
//! 读取一律走 [`Cursor`]（借用调用方切片，不复制整份流）；每个 opcode 一个自由函数，
//! 整张 [`OPS`] 表就是能力清单。

use diagnostic::Result;
use dtype::DType;
use wire::{bad, or_bad};

use crate::bin::Tensor;
use crate::bytes::Cursor;
use crate::pickle_stack::{append, memo_op, persistent_load, reduce, set_item};
use crate::pickle_storage::into_str;

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
    table[b'\x80' as usize] = Some(skip);
    table[b'\x95' as usize] = Some(skip);
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
    table[b'G' as usize] = Some(float);
    table[b'c' as usize] = Some(global);
    table[b'\x93' as usize] = Some(global);
    // 整数：`K`/`M` 无符号 1/2 字节，`J` 补码 4 字节，`\x8a`/`\x8b` 长度前缀 1/4 字节。
    table[b'K' as usize] = Some(push_int);
    table[b'M' as usize] = Some(push_int);
    table[b'J' as usize] = Some(push_int);
    table[b'\x8a' as usize] = Some(long);
    table[b'\x8b' as usize] = Some(long);
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
            let handler = OPS[usize::from(op)]
                .ok_or_else(|| bad(format!("未知 pickle opcode 0x{op:02X}")))?;
            handler(self, op)?;
        }
    }

    /// `GLOBAL` 的两行名字：读到下一个 `\n` 为止（换行本身被消费）。
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

/// `PROTO` 的 1 字节版本号与 `FRAME` 的 8 字节帧长：值不参与重建，字节必须完整存在。
fn skip(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    vm.cursor.take(if op == b'\x95' { 8 } else { 1 }).map(drop)
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

/// `BINFLOAT`：8 字节**大端** IEEE 双精度；张量重建不使用这个值，但字节必须完整存在。
fn float(vm: &mut Vm<'_>, _op: u8) -> Result<()> {
    vm.cursor.take(8).map(drop)?;
    vm.stack.push(Value::Scalar);
    Ok(())
}

/// `LONG1`/`LONG4`：长度分别是 1/4 字节，正文是变长小端补码；超过 8 字节装不进 `i64`，
/// 由 [`integer`] 统一拒掉。
fn long(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let len = usize::try_from(vm.cursor.uint(if op == b'\x8b' { 4 } else { 1 })?)
        .map_err(|_| bad("整数长度超出 usize"))?;
    integer(vm, len, true)
}

/// 全局名：`GLOBAL` 从流里读两行，`STACK_GLOBAL` 从栈上取两个字符串（模块在前）。
fn global(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let (module, name) = if op == b'c' {
        (vm.line()?, vm.line()?)
    } else {
        let name = into_str(&vm.pop()?)?;
        (into_str(&vm.pop()?)?, name)
    };
    vm.stack.push(Value::Global { module, name });
    Ok(())
}

/// `BININT1`/`BININT2`（无符号）与 `BININT`（4 字节补码）共用一套符号规则。
fn push_int(vm: &mut Vm<'_>, op: u8) -> Result<()> {
    let (width, signed) = match op {
        b'M' => (2, false),
        b'J' => (4, true),
        _ => (1, false),
    };
    integer(vm, width, signed)
}

/// 定宽小端补码字段 → `i64`：`signed` 按最高位符号扩展（`BININT`/`LONG*`），否则零扩展
/// （`BININT1`/`BININT2`）；超过 8 字节装不进 `i64`，显式失败。
fn integer(vm: &mut Vm<'_>, width: usize, signed: bool) -> Result<()> {
    if width > 8 {
        return Err(bad("整数超过 i64 宽度"));
    }
    let raw = vm.cursor.take(width)?;
    let negative = signed && raw.last().is_some_and(|&last| last & 0x80 != 0);
    let mut buf = [if negative { 0xFF } else { 0 }; 8];
    buf[..width].copy_from_slice(raw);
    vm.stack.push(Value::Int(i64::from_le_bytes(buf)));
    Ok(())
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

#[cfg(test)]
#[path = "../tests/bin_pickle.rs"]
mod tests;

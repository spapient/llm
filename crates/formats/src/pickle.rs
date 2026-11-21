//! 栈式 pickle 虚拟机：解释 `data.pkl` 流，拦截 `_rebuild_tensor_v2` 重建与 `BINPERSID`
//! persistent storage，还原「名字 + 类型 + 形状 + 字节区间」。

use diagnostic::Result;
use dtype::DType;
use wire::bad;

use crate::bin::Tensor;
use crate::bytes::slice;
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

#[derive(Default)]
pub(crate) struct Vm<'a> {
    pub(crate) stack: Vec<Value>,
    pub(crate) memo: Vec<Value>,
    pub(crate) at: usize,
    pub(crate) pkl: &'a [u8],
    pub(crate) tensors: Vec<Tensor>,
}

/// 一个 opcode 的处理函数。opcode 字节本身也要传进去：`K`/`M`、`X`/`\x8c`、`q`/`r`/`h`/`j`
/// 都是一套处理覆盖多个码。
type Handler = for<'v> fn(&mut Vm<'v>, u8) -> Result<()>;

/// opcode → 处理函数。整张表就是 pickle 的能力清单：删掉一行，那条 opcode 就退化成「未知」错误。
static OPS: [Option<Handler>; 256] = ops();

const fn ops() -> [Option<Handler>; 256] {
    let mut table: [Option<Handler>; 256] = [None; 256];
    // 协议框架：版本号 1 字节、帧长 8 字节，都不影响重建结果。
    table[b'\x80' as usize] = Some(Vm::skip);
    table[b'\x95' as usize] = Some(Vm::skip);
    // 空容器与标记：`(` 是 MARK，其余是空 dict/list 与 `()`。
    table[b'(' as usize] = Some(Vm::placeholder);
    table[b'}' as usize] = Some(Vm::placeholder);
    table[b']' as usize] = Some(Vm::placeholder);
    table[b')' as usize] = Some(Vm::placeholder);
    table[b'\x89' as usize] = Some(Vm::placeholder);
    table[b'\x88' as usize] = Some(Vm::placeholder);
    table[b'\x81' as usize] = Some(Vm::newobj);
    // 文本：长度前缀 4 字节（`X`）或 1 字节（`\x8c`）。
    table[b'X' as usize] = Some(Vm::string);
    table[b'\x8c' as usize] = Some(Vm::string);
    // 全局名与不可达值。
    table[b'G' as usize] = Some(Vm::float);
    table[b'c' as usize] = Some(Vm::global);
    table[b'\x93' as usize] = Some(Vm::global_stack);
    // 整数：`K`/`M` 无符号 1/2 字节，`J` 补码 4 字节，`\x8a`/`\x8b` 变长补码。
    table[b'K' as usize] = Some(Vm::push_uint);
    table[b'M' as usize] = Some(Vm::push_uint);
    table[b'J' as usize] = Some(Vm::push_binint);
    table[b'\x8a' as usize] = Some(Vm::long);
    table[b'\x8b' as usize] = Some(Vm::long);
    // 元组：`t` 读 MARK 起的一段，`\x85..\x87` 按 1/2/3 个成员。
    table[b't' as usize] = Some(Vm::tuple);
    table[b'\x85' as usize] = Some(Vm::tuple);
    table[b'\x86' as usize] = Some(Vm::tuple);
    table[b'\x87' as usize] = Some(Vm::tuple);
    // memo 与对象图。
    table[b'q' as usize] = Some(Vm::memo_op);
    table[b'r' as usize] = Some(Vm::memo_op);
    table[b'\x94' as usize] = Some(Vm::memo_op);
    table[b'h' as usize] = Some(Vm::memo_op);
    table[b'j' as usize] = Some(Vm::memo_op);
    table[b'Q' as usize] = Some(Vm::persistent_load);
    table[b'R' as usize] = Some(Vm::reduce);
    // 容器写入与出栈。
    table[b'u' as usize] = Some(Vm::set_item);
    table[b's' as usize] = Some(Vm::set_item);
    table[b'a' as usize] = Some(Vm::append);
    table[b'e' as usize] = Some(Vm::append);
    table[b'b' as usize] = Some(Vm::discard);
    table
}

impl<'a> Vm<'a> {
    pub(crate) fn run(pkl: &'a [u8]) -> Result<Vec<Tensor>> {
        let mut vm = Vm {
            pkl,
            ..Vm::default()
        };
        vm.execute()?;
        Ok(vm.tensors)
    }

    pub(crate) fn execute(&mut self) -> Result<()> {
        loop {
            let op = self.byte()?;
            if op == b'.' {
                return Ok(());
            }
            let handler = OPS[usize::from(op)]
                .ok_or_else(|| bad(format!("未知 pickle opcode 0x{op:02X}")))?;
            handler(self, op)?;
        }
    }

    pub(crate) fn byte(&mut self) -> Result<u8> {
        let b = *diag::required(self.pkl.get(self.at), "pickle 意外结束")?;
        self.at += 1;
        Ok(b)
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let out = slice(self.pkl, self.at, n)?;
        self.at += n;
        Ok(out)
    }

    pub(crate) fn uint(&mut self, n: usize) -> Result<u64> {
        let raw = self.bytes(n)?;
        Ok(raw
            .iter()
            .rev()
            .fold(0u64, |acc, &b| (acc << 8) | u64::from(b)))
    }

    pub(crate) fn text(&mut self, n: usize) -> Result<String> {
        let raw = self.bytes(n)?;
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|_| bad("字符串不是 UTF-8"))
    }

    pub(crate) fn line(&mut self) -> Result<String> {
        let stop = self
            .pkl
            .get(self.at..)
            .and_then(|tail| tail.iter().position(|&b| b == b'\n'));
        let raw = self.bytes(diag::required(stop, "GLOBAL 缺换行")?)?;
        self.byte()?; // 消费换行
        std::str::from_utf8(raw)
            .map(str::to_owned)
            .map_err(|_| bad("GLOBAL 不是 UTF-8"))
    }

    /// `PROTO` 的 1 字节版本号与 `FRAME` 的 8 字节帧长：值不参与重建，字节必须完整存在。
    pub(crate) fn skip(&mut self, op: u8) -> Result<()> {
        self.bytes(if op == b'\x95' { 8 } else { 1 }).map(drop)
    }

    pub(crate) fn newobj(&mut self) -> Result<()> {
        self.pop()?;
        self.pop()?;
        Err(bad("NEWOBJ 不支持的类"))
    }

    pub(crate) fn placeholder(&mut self, op: u8) -> Result<()> {
        let value = match op {
            b'}' => Value::Dict(Vec::new()),
            b']' => Value::List(Vec::new()),
            b'(' => Value::Marker,
            _ => Value::Scalar,
        };
        self.stack.push(value);
        Ok(())
    }

    pub(crate) fn string(&mut self, op: u8) -> Result<()> {
        let len = self.uint(if op == b'X' { 4 } else { 1 })? as usize;
        let text = self.text(len)?;
        self.stack.push(Value::Str(text));
        Ok(())
    }

    /// `BINFLOAT`：8 字节**大端** IEEE 双精度；张量重建不使用这个值，但字节必须完整存在。
    pub(crate) fn float(&mut self) -> Result<()> {
        self.bytes(8).map(drop)?;
        self.stack.push(Value::Scalar);
        Ok(())
    }

    /// `LONG1`/`LONG4`：长度分别是 1/4 字节，正文是变长小端补码；超过 8 字节装不进 `i64`，
    /// 由 [`Vm::integer`] 统一拒掉。
    pub(crate) fn long(&mut self, op: u8) -> Result<()> {
        let len = self.uint(if op == b'\x8b' { 4 } else { 1 })? as usize;
        let raw = self.bytes(len)?;
        self.integer(raw, true)
    }

    pub(crate) fn global(&mut self) -> Result<()> {
        let module = self.line()?;
        let name = self.line()?;
        self.stack.push(Value::Global { module, name });
        Ok(())
    }

    pub(crate) fn global_stack(&mut self) -> Result<()> {
        let name = into_str(&self.pop()?)?;
        let module = into_str(&self.pop()?)?;
        self.stack.push(Value::Global { module, name });
        Ok(())
    }

    /// `BININT1`/`BININT2`（无符号）与 `BININT`（4 字节补码）共用一套符号规则。
    pub(crate) fn push_uint(&mut self, op: u8) -> Result<()> {
        let raw = self.bytes(if op == b'M' { 2 } else { 1 })?;
        self.integer(raw, false)
    }

    pub(crate) fn push_binint(&mut self) -> Result<()> {
        let raw = self.bytes(4)?;
        self.integer(raw, true)
    }

    /// 小端补码字节 → `i64`：`signed` 时按最高位符号扩展（`BININT`/`LONG*`），否则零扩展
    /// （`BININT1`/`BININT2`）；超过 8 字节装不进 `i64`，显式失败。
    fn integer(&mut self, raw: &[u8], signed: bool) -> Result<()> {
        if raw.len() > 8 {
            return Err(bad("整数超过 i64 宽度"));
        }
        let negative = signed && raw.last().is_some_and(|&last| last & 0x80 != 0);
        let mut buf = [if negative { 0xFF } else { 0 }; 8];
        buf[..raw.len()].copy_from_slice(raw);
        self.stack.push(Value::Int(i64::from_le_bytes(buf)));
        Ok(())
    }

    pub(crate) fn tuple(&mut self, op: u8) -> Result<()> {
        let items = match op {
            b't' => self.pop_until_marker()?,
            _ => self.pops(usize::from(op - 0x84))?,
        };
        self.stack.push(Value::Tuple(items));
        Ok(())
    }

    /// `POP`：丢弃栈顶。
    pub(crate) fn discard(&mut self) -> Result<()> {
        self.pop().map(drop)
    }
}

#[cfg(test)]
#[path = "../tests/bin_pickle.rs"]
mod tests;

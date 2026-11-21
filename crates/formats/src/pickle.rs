//! 栈式 pickle 虚拟机：解释 `data.pkl` 流，拦截 `_rebuild_tensor_v2` 重建与 `BINPERSID`
//! persistent storage，还原「名字 + 类型 + 形状 + 字节区间」。
//!
//! 未知 opcode / 不可识别可调用对象显式报错；不执行任意代码、不猜结果。

pub(crate) mod stack;
pub(crate) mod storage;

use wire::{bad, or_bad};
use diagnostic::Result;
use dtype::DType;

use crate::bin::Tensor;
use crate::pickle::storage::into_str;

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
pub(crate) struct Vm {
    pub(crate) stack: Vec<Value>,
    pub(crate) memo: Vec<Value>,
    pub(crate) at: usize,
    pub(crate) pkl: Vec<u8>,
    pub(crate) tensors: Vec<Tensor>,
}

impl Vm {
    pub(crate) fn run(pkl: &[u8]) -> Result<Vec<Tensor>> {
        let mut vm = Vm {
            pkl: pkl.to_vec(),
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
            self.step(op)?;
        }
    }

    pub(crate) fn byte(&mut self) -> Result<u8> {
        let b = *or_bad(self.pkl.get(self.at), "pickle 意外结束")?;
        self.at += 1;
        Ok(b)
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let slice = or_bad(self.pkl.get(self.at..self.at + n), "pickle 字段越界")?;
        self.at += n;
        Ok(slice.to_vec())
    }

    pub(crate) fn uint(&mut self, n: usize) -> Result<u64> {
        let raw = self.bytes(n)?;
        let value = raw
            .iter()
            .rev()
            .fold(0u64, |acc, &b| (acc << 8) | u64::from(b));
        Ok(value)
    }

    pub(crate) fn text(&mut self, n: usize) -> Result<String> {
        let raw = self.bytes(n)?;
        String::from_utf8(raw).map_err(|_| bad("字符串不是 UTF-8"))
    }

    pub(crate) fn line(&mut self) -> Result<String> {
        let stop = self
            .pkl
            .get(self.at..)
            .and_then(|s| s.iter().position(|&b| b == b'\n'));
        let stop = or_bad(stop, "GLOBAL 缺换行")?;
        let raw = self.bytes(stop)?;
        self.byte()?; // 消费换行
        String::from_utf8(raw).map_err(|_| bad("GLOBAL 不是 UTF-8"))
    }

    pub(crate) fn step(&mut self, op: u8) -> Result<()> {
        match op {
            b'\x80' => self.bytes(1).map(drop),
            b'\x95' => self.bytes(8).map(drop),
            b'(' | b'}' | b']' | b')' | b'\x89' | b'\x88' => self.placeholder(op),
            b'\x81' => self.newobj(),
            _ => self.step_text(op),
        }
    }

    pub(crate) fn step_text(&mut self, op: u8) -> Result<()> {
        match op {
            b'X' | b'\x8c' => self.string(op),
            b'G' => self.float(),
            b'c' => self.global(),
            _ => self.step_global(op),
        }
    }

    pub(crate) fn step_global(&mut self, op: u8) -> Result<()> {
        match op {
            b'\x93' => self.global_stack(),
            b'K' | b'M' => self.push_uint(op),
            b'J' => self.push_binint(),
            b't' | b'\x85'..=b'\x87' => self.tuple(op),
            _ => self.step_tuple(op),
        }
    }

    pub(crate) fn step_tuple(&mut self, op: u8) -> Result<()> {
        match op {
            b'q' | b'r' | b'\x94' | b'h' | b'j' => self.memo_op(op),
            b'Q' => self.persistent_load(),
            b'R' => self.reduce(),
            _ => self.step_command(op),
        }
    }

    pub(crate) fn step_command(&mut self, op: u8) -> Result<()> {
        match op {
            b'u' => self.set_items(),
            b's' => self.set_item(),
            b'a' | b'e' => self.append(op),
            b'b' => self.pop().map(drop),
            other => Err(bad(format!("未知 pickle opcode 0x{other:02X}"))),
        }
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

    pub(crate) fn float(&mut self) -> Result<()> {
        self.bytes(8).map(drop)?;
        self.stack.push(Value::Scalar);
        Ok(())
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

    pub(crate) fn push_uint(&mut self, op: u8) -> Result<()> {
        let width = if op == b'M' { 2 } else { 1 };
        let value = self.uint(width)? as i64;
        self.stack.push(Value::Int(value));
        Ok(())
    }

    pub(crate) fn push_binint(&mut self) -> Result<()> {
        let raw = self.uint(4)? as u32 as i32;
        self.stack.push(Value::Int(i64::from(raw)));
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
}

#[cfg(test)]
#[path = "../tests/bin_pickle.rs"]
mod tests;

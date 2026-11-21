//! pickle 的栈与对象图：出栈、memo、`REDUCE` 与 `_rebuild_tensor_v2` 的重建。
//!
//! 从 [`crate::pickle`] 拆出：那边只管 opcode 分派，这里只管「栈上的值怎么被组合成张量」。

use diagnostic::Result;

use crate::pickle::storage::{as_tuple, storage_handle, tensor_of};
use crate::pickle::{Value, Vm};
use wire::{bad, or_bad};

impl Vm {
    pub(crate) fn pops(&mut self, n: usize) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        for _ in 0..n {
            items.push(self.pop()?);
        }
        items.reverse();
        Ok(items)
    }

    pub(crate) fn memo_index(&mut self, op: u8) -> Result<usize> {
        match op {
            b'q' | b'h' => Ok(usize::from(self.byte()?)),
            b'r' | b'j' => Ok(self.uint(4)? as usize),
            _ => Ok(self.memo.len()),
        }
    }

    pub(crate) fn memo_op(&mut self, op: u8) -> Result<()> {
        let idx = self.memo_index(op)?;
        match op {
            b'h' | b'j' => self.recall_memo(idx),
            _ => self.store_memo(idx),
        }
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

    pub(crate) fn store_memo(&mut self, idx: usize) -> Result<()> {
        let value = or_bad(self.stack.last(), "空栈无法写 memo")?.clone();
        // memo 槽位由更早的存槽 opcode 占用，索引不可能超过流长度；谎报大索引先拒绝，避免稀疏巨表。
        if idx > self.pkl.len() {
            return Err(bad("memo 索引超出 pickle 流长度"));
        }
        if idx >= self.memo.len() {
            self.memo.resize(idx + 1, Value::Scalar);
        }
        self.memo[idx] = value;
        Ok(())
    }

    pub(crate) fn recall_memo(&mut self, idx: usize) -> Result<()> {
        let value = or_bad(self.memo.get(idx), "memo 索引缺失")?.clone();
        self.stack.push(value);
        Ok(())
    }

    pub(crate) fn persistent_load(&mut self) -> Result<()> {
        let items = as_tuple(self.pop()?, "persistent id 不是元组")?;
        let (key, dtype) = storage_handle(&items)?;
        self.stack.push(Value::Storage { key, dtype });
        Ok(())
    }

    pub(crate) fn reduce(&mut self) -> Result<()> {
        let args = self.pop()?;
        let (module, name) = match self.pop()? {
            Value::Global { module, name } => (module, name),
            _ => return Err(bad("REDUCE 的可调用对象不是 GLOBAL")),
        };
        self.reduced(&module, &name, args)
    }

    pub(crate) fn reduced(&mut self, module: &str, name: &str, args: Value) -> Result<()> {
        match (module, name) {
            ("torch._utils", "_rebuild_tensor_v2") => self.rebuild_tensor(args),
            ("collections", "OrderedDict") => {
                self.stack.push(Value::Dict(Vec::new()));
                Ok(())
            }
            _ => Err(bad(format!("不支持的可调用对象 {module}.{name}"))),
        }
    }

    pub(crate) fn rebuild_tensor(&mut self, args: Value) -> Result<()> {
        let items = as_tuple(args, "rebuild 参数不是元组")?;
        let mut tensor = tensor_of(&items)?;
        tensor.name = self.tensor_name()?;
        self.tensors.push(tensor);
        self.stack.push(Value::Tensor);
        Ok(())
    }

    pub(crate) fn tensor_name(&self) -> Result<String> {
        let name = self.stack.iter().rev().find_map(|v| match v {
            Value::Str(s) => Some(s),
            _ => None,
        });
        or_bad(name.cloned(), "找不到张量名")
    }

    pub(crate) fn pop_pairs(&mut self) -> Result<Vec<(Value, Value)>> {
        let items = self.pop_until_marker()?;
        if items.len() % 2 != 0 {
            return Err(bad("SETITEMS 的项数为奇数"));
        }
        Ok(items
            .chunks(2)
            .map(|pair| (pair[0].clone(), pair[1].clone()))
            .collect())
    }

    pub(crate) fn set_items(&mut self) -> Result<()> {
        let pairs = self.pop_pairs()?;
        let dict = self
            .stack
            .last_mut()
            .ok_or_else(|| bad("SETITEMS 缺目标 dict"))?;
        if let Value::Dict(entries) = dict {
            entries.extend(pairs);
            Ok(())
        } else {
            Err(bad("SETITEMS 目标不是 dict"))
        }
    }

    pub(crate) fn set_item(&mut self) -> Result<()> {
        let value = self.pop()?;
        let key = self.pop()?;
        match self.stack.last_mut() {
            Some(Value::Dict(entries)) => {
                entries.push((key, value));
                Ok(())
            }
            _ => Err(bad("SETITEM 目标不是 dict")),
        }
    }

    pub(crate) fn append(&mut self, op: u8) -> Result<()> {
        let (items, label) = if op == b'a' {
            (vec![self.pop()?], "APPEND")
        } else {
            (self.pop_until_marker()?, "APPENDS")
        };
        self.extend_list(items, label)
    }

    pub(crate) fn extend_list(&mut self, items: Vec<Value>, label: &str) -> Result<()> {
        match self.stack.last_mut() {
            Some(Value::List(list)) => {
                list.extend(items);
                Ok(())
            }
            _ => Err(bad(format!("{label} 目标不是 list"))),
        }
    }
}

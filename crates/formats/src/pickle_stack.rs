//! 职责：pickle 的栈与对象图——出栈、memo、`REDUCE` 与 `_rebuild_tensor_v2` 的重建。

use diagnostic::Result;

use crate::pickle::{Value, Vm};
use crate::pickle_storage::{as_tuple, storage_handle, tensor_of};
use wire::{bad, or_bad};

impl Vm {
    /// 弹出栈顶 `n` 个值并保持原来的压栈顺序。
    pub(crate) fn pops(&mut self, n: usize) -> Result<Vec<Value>> {
        let at = self
            .stack
            .len()
            .checked_sub(n)
            .ok_or_else(|| bad("栈下溢"))?;
        Ok(self.stack.split_off(at))
    }

    /// memo 的 5 条 opcode：`q`/`r`/`\x94` 存当前栈顶，`h`/`j` 取回；`r`/`j` 的索引是 4 字节。
    pub(crate) fn memo_op(&mut self, op: u8) -> Result<()> {
        let idx = match op {
            b'q' | b'h' => usize::from(self.byte()?),
            b'r' | b'j' => {
                usize::try_from(self.uint(4)?).map_err(|_| bad("memo 索引超出 usize"))?
            }
            _ => self.memo.len(),
        };
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

    pub(crate) fn persistent_load(&mut self, _op: u8) -> Result<()> {
        let items = as_tuple(self.pop()?, "persistent id 不是元组")?;
        let (key, dtype) = storage_handle(&items)?;
        self.stack.push(Value::Storage { key, dtype });
        Ok(())
    }

    /// `REDUCE`：弹出实参与可调用对象，只认张量重建与 `OrderedDict` 两种构造。
    pub(crate) fn reduce(&mut self, _op: u8) -> Result<()> {
        let args = self.pop()?;
        let (module, name) = match self.pop()? {
            Value::Global { module, name } => (module, name),
            _ => return Err(bad("REDUCE 的可调用对象不是 GLOBAL")),
        };
        match (module.as_str(), name.as_str()) {
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

    /// 栈顶 dict 写入：`SETITEM` 弹一队键值，`SETITEMS` 弹 MARK 起的一段（项数必须为偶数）。
    pub(crate) fn set_item(&mut self, op: u8) -> Result<()> {
        let (pairs, label) = if op == b's' {
            let value = self.pop()?;
            (vec![(self.pop()?, value)], "SETITEM")
        } else {
            let items = self.pop_until_marker()?;
            if items.len() % 2 != 0 {
                return Err(bad("SETITEMS 的项数为奇数"));
            }
            let pairs = items
                .chunks(2)
                .map(|pair| (pair[0].clone(), pair[1].clone()))
                .collect();
            (pairs, "SETITEMS")
        };
        match self.stack.last_mut() {
            Some(Value::Dict(entries)) => {
                entries.extend(pairs);
                Ok(())
            }
            _ => Err(bad(format!("{label} 目标不是 dict"))),
        }
    }

    /// 栈顶 list 追加：`APPEND` 弹一值，`APPENDS` 弹 MARK 起的一段。
    pub(crate) fn append(&mut self, op: u8) -> Result<()> {
        let (items, label) = if op == b'a' {
            (vec![self.pop()?], "APPEND")
        } else {
            (self.pop_until_marker()?, "APPENDS")
        };
        match self.stack.last_mut() {
            Some(Value::List(list)) => {
                list.extend(items);
                Ok(())
            }
            _ => Err(bad(format!("{label} 目标不是 list"))),
        }
    }
}

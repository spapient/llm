//! 栈式 pickle 虚拟机：解释 `data.pkl` 流，拦截 `_rebuild_tensor_v2` 重建与 `BINPERSID`
//! persistent storage，还原「名字 + 类型 + 形状 + 字节区间」。
//!
//! 未知 opcode / 不可识别可调用对象显式报错；不执行任意代码、不猜结果。

use diagnostic::Result;
use dtype::DType;

use super::{Tensor, bad, or_bad};

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

#[derive(Default)]
pub(crate) struct Vm {
    stack: Vec<Value>,
    memo: Vec<Value>,
    at: usize,
    pkl: Vec<u8>,
    tensors: Vec<Tensor>,
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

    fn execute(&mut self) -> Result<()> {
        loop {
            let op = self.byte()?;
            if op == b'.' {
                return Ok(());
            }
            self.step(op)?;
        }
    }

    fn byte(&mut self) -> Result<u8> {
        let b = *or_bad(self.pkl.get(self.at), "pickle 意外结束")?;
        self.at += 1;
        Ok(b)
    }

    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let slice = or_bad(self.pkl.get(self.at..self.at + n), "pickle 字段越界")?;
        self.at += n;
        Ok(slice.to_vec())
    }

    fn uint(&mut self, n: usize) -> Result<u64> {
        let raw = self.bytes(n)?;
        let value = raw
            .iter()
            .rev()
            .fold(0u64, |acc, &b| (acc << 8) | u64::from(b));
        Ok(value)
    }

    fn text(&mut self, n: usize) -> Result<String> {
        let raw = self.bytes(n)?;
        String::from_utf8(raw).map_err(|_| bad("字符串不是 UTF-8"))
    }

    fn line(&mut self) -> Result<String> {
        let stop = self
            .pkl
            .get(self.at..)
            .and_then(|s| s.iter().position(|&b| b == b'\n'));
        let stop = or_bad(stop, "GLOBAL 缺换行")?;
        let raw = self.bytes(stop)?;
        self.byte()?; // 消费换行
        String::from_utf8(raw).map_err(|_| bad("GLOBAL 不是 UTF-8"))
    }

    fn step(&mut self, op: u8) -> Result<()> {
        match op {
            b'\x80' => self.bytes(1).map(drop),
            b'\x95' => self.bytes(8).map(drop),
            b'(' | b'}' | b']' | b')' | b'\x89' | b'\x88' => self.placeholder(op),
            b'\x81' => self.newobj(),
            _ => self.step_text(op),
        }
    }

    fn step_text(&mut self, op: u8) -> Result<()> {
        match op {
            b'X' | b'\x8c' => self.string(op),
            b'G' => self.float(),
            b'c' => self.global(),
            _ => self.step_global(op),
        }
    }

    fn step_global(&mut self, op: u8) -> Result<()> {
        match op {
            b'\x93' => self.global_stack(),
            b'K' | b'M' => self.push_uint(op),
            b'J' => self.push_binint(),
            b't' | b'\x85'..=b'\x87' => self.tuple(op),
            _ => self.step_tuple(op),
        }
    }

    fn step_tuple(&mut self, op: u8) -> Result<()> {
        match op {
            b'q' | b'r' | b'\x94' | b'h' | b'j' => self.memo_op(op),
            b'Q' => self.persistent_load(),
            b'R' => self.reduce(),
            _ => self.step_command(op),
        }
    }

    fn step_command(&mut self, op: u8) -> Result<()> {
        match op {
            b'u' => self.set_items(),
            b's' => self.set_item(),
            b'a' | b'e' => self.append(op),
            b'b' => self.pop().map(drop),
            other => Err(bad(&format!("未知 pickle opcode 0x{other:02X}"))),
        }
    }

    fn newobj(&mut self) -> Result<()> {
        self.pop()?;
        self.pop()?;
        Err(bad("NEWOBJ 不支持的类"))
    }

    fn placeholder(&mut self, op: u8) -> Result<()> {
        let value = match op {
            b'}' => Value::Dict(Vec::new()),
            b']' => Value::List(Vec::new()),
            b'(' => Value::Marker,
            _ => Value::Scalar,
        };
        self.stack.push(value);
        Ok(())
    }

    fn string(&mut self, op: u8) -> Result<()> {
        let len = self.uint(if op == b'X' { 4 } else { 1 })? as usize;
        let text = self.text(len)?;
        self.stack.push(Value::Str(text));
        Ok(())
    }

    fn float(&mut self) -> Result<()> {
        self.bytes(8).map(drop)?;
        self.stack.push(Value::Scalar);
        Ok(())
    }

    fn global(&mut self) -> Result<()> {
        let module = self.line()?;
        let name = self.line()?;
        self.stack.push(Value::Global { module, name });
        Ok(())
    }

    fn global_stack(&mut self) -> Result<()> {
        let name = into_str(&self.pop()?)?;
        let module = into_str(&self.pop()?)?;
        self.stack.push(Value::Global { module, name });
        Ok(())
    }

    fn push_uint(&mut self, op: u8) -> Result<()> {
        let width = if op == b'M' { 2 } else { 1 };
        let value = self.uint(width)? as i64;
        self.stack.push(Value::Int(value));
        Ok(())
    }

    fn push_binint(&mut self) -> Result<()> {
        let raw = self.uint(4)? as u32 as i32;
        self.stack.push(Value::Int(i64::from(raw)));
        Ok(())
    }

    fn tuple(&mut self, op: u8) -> Result<()> {
        let items = match op {
            b't' => self.pop_until_marker()?,
            _ => self.pops(usize::from(op - 0x84))?,
        };
        self.stack.push(Value::Tuple(items));
        Ok(())
    }

    fn pops(&mut self, n: usize) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        for _ in 0..n {
            items.push(self.pop()?);
        }
        items.reverse();
        Ok(items)
    }

    fn memo_index(&mut self, op: u8) -> Result<usize> {
        match op {
            b'q' | b'h' => Ok(usize::from(self.byte()?)),
            b'r' | b'j' => Ok(self.uint(4)? as usize),
            _ => Ok(self.memo.len()),
        }
    }

    fn memo_op(&mut self, op: u8) -> Result<()> {
        let idx = self.memo_index(op)?;
        match op {
            b'h' | b'j' => self.recall_memo(idx),
            _ => self.store_memo(idx),
        }
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
        if idx > self.pkl.len() {
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

    fn persistent_load(&mut self) -> Result<()> {
        let items = as_tuple(self.pop()?, "persistent id 不是元组")?;
        let (key, dtype) = storage_handle(&items)?;
        self.stack.push(Value::Storage { key, dtype });
        Ok(())
    }

    fn reduce(&mut self) -> Result<()> {
        let args = self.pop()?;
        let (module, name) = match self.pop()? {
            Value::Global { module, name } => (module, name),
            _ => return Err(bad("REDUCE 的可调用对象不是 GLOBAL")),
        };
        self.reduced(&module, &name, args)
    }

    fn reduced(&mut self, module: &str, name: &str, args: Value) -> Result<()> {
        match (module, name) {
            ("torch._utils", "_rebuild_tensor_v2") => self.rebuild_tensor(args),
            ("collections", "OrderedDict") => {
                self.stack.push(Value::Dict(Vec::new()));
                Ok(())
            }
            _ => Err(bad(&format!("不支持的可调用对象 {module}.{name}"))),
        }
    }

    fn rebuild_tensor(&mut self, args: Value) -> Result<()> {
        let items = as_tuple(args, "rebuild 参数不是元组")?;
        let mut tensor = tensor_of(&items)?;
        tensor.name = self.tensor_name()?;
        self.tensors.push(tensor);
        self.stack.push(Value::Tensor);
        Ok(())
    }

    fn tensor_name(&self) -> Result<String> {
        let name = self.stack.iter().rev().find_map(|v| match v {
            Value::Str(s) => Some(s),
            _ => None,
        });
        or_bad(name.cloned(), "找不到张量名")
    }

    fn pop_pairs(&mut self) -> Result<Vec<(Value, Value)>> {
        let items = self.pop_until_marker()?;
        if items.len() % 2 != 0 {
            return Err(bad("SETITEMS 的项数为奇数"));
        }
        Ok(items
            .chunks(2)
            .map(|pair| (pair[0].clone(), pair[1].clone()))
            .collect())
    }

    fn set_items(&mut self) -> Result<()> {
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

    fn set_item(&mut self) -> Result<()> {
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

    fn append(&mut self, op: u8) -> Result<()> {
        let (items, label) = if op == b'a' {
            (vec![self.pop()?], "APPEND")
        } else {
            (self.pop_until_marker()?, "APPENDS")
        };
        self.extend_list(items, label)
    }

    fn extend_list(&mut self, items: Vec<Value>, label: &str) -> Result<()> {
        match self.stack.last_mut() {
            Some(Value::List(list)) => {
                list.extend(items);
                Ok(())
            }
            _ => Err(bad(&format!("{label} 目标不是 list"))),
        }
    }
}

fn as_tuple(value: Value, message: &str) -> Result<Vec<Value>> {
    match value {
        Value::Tuple(items) => Ok(items),
        _ => Err(bad(message)),
    }
}

fn at<'v>(items: &'v [Value], index: usize, message: &str) -> Result<&'v Value> {
    or_bad(items.get(index), message)
}

fn at_str(items: &[Value], index: usize, message: &str) -> Result<String> {
    into_str(at(items, index, message)?)
}

fn check_tag(items: &[Value]) -> Result<()> {
    let tag = into_str(at(items, 0, "pid 空")?)?;
    if tag == "storage" {
        return Ok(());
    }
    Err(bad(&format!("不支持的 persistent id `{tag}`")))
}

fn storage_class(items: &[Value]) -> Result<&str> {
    match at(items, 1, "pid 缺 storage 类")? {
        Value::Global { name, .. } => Ok(name.as_str()),
        _ => Err(bad("storage 类不是 GLOBAL")),
    }
}

fn storage_handle(items: &[Value]) -> Result<(String, DType)> {
    check_tag(items)?;
    let dtype = storage_dtype(storage_class(items)?)?;
    let key = at_str(items, 2, "pid 缺 key")?;
    Ok((key, dtype))
}

fn storage_of(items: &[Value]) -> Result<(String, DType)> {
    match items.first() {
        Some(Value::Storage { key, dtype }) => Ok((key.clone(), *dtype)),
        _ => Err(bad("rebuild 缺 storage")),
    }
}

fn offset_of(items: &[Value]) -> Result<usize> {
    if let Some(Value::Int(n)) = items.get(1) {
        return usize::try_from(*n).map_err(|_| bad("offset 溢出"));
    }
    Err(bad("rebuild 缺 offset"))
}

fn shape_of(items: &[Value]) -> Result<(Vec<u64>, usize)> {
    let dims = match at(items, 2, "rebuild 缺 shape")? {
        Value::Tuple(dims) => dims,
        _ => return Err(bad("rebuild 缺 shape")),
    };
    let shape = dims.iter().map(dim).collect::<Result<Vec<u64>>>()?;
    let count = shape.iter().try_fold(1usize, |acc, &d| {
        usize::try_from(d).ok().and_then(|n| acc.checked_mul(n))
    });
    Ok((shape, or_bad(count, "形状乘积溢出")?))
}

fn dim(value: &Value) -> Result<u64> {
    match value {
        Value::Int(n) => u64::try_from(*n).map_err(|_| bad("形状维度溢出")),
        _ => Err(bad("形状维度不是整数")),
    }
}

fn tensor_of(items: &[Value]) -> Result<Tensor> {
    let (storage_key, dtype) = storage_of(items)?;
    let offset = offset_of(items)?;
    let (shape, count) = shape_of(items)?;
    let (offset, len) = span(offset, count, dtype.bytes_for(1) as usize)?;
    Ok(Tensor {
        name: String::new(),
        dtype,
        shape,
        storage_key,
        offset,
        len,
    })
}

fn into_str(value: &Value) -> Result<String> {
    match value {
        Value::Str(s) => Ok(s.clone()),
        _ => Err(bad("期望字符串")),
    }
}

fn span(offset: usize, count: usize, width: usize) -> Result<(usize, usize)> {
    let offset = or_bad(offset.checked_mul(width), "offset 溢出")?;
    let len = or_bad(count.checked_mul(width), "长度溢出")?;
    Ok((offset, len))
}

fn storage_dtype(name: &str) -> Result<DType> {
    STORAGE_DTYPES
        .iter()
        .find_map(|&(class, dtype)| (class == name).then_some(dtype))
        .ok_or_else(|| bad(&format!("不支持的 storage 类型 `{name}`")))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_odd_setitems_is_an_error_not_a_panic() {
        let error = Vm::run(b"}(K\x01u.").expect_err("奇数项必须报错");
        assert!(error.to_string().contains("奇数"), "{error}");
    }

    #[test]
    fn a_huge_memo_index_is_rejected_not_resized() {
        let error = Vm::run(b"K\x01r\xff\xff\xff\xff.").expect_err("memo 索引必须报错");
        assert!(error.to_string().contains("memo"), "{error}");
    }
}

//! bin（torch.save 产出的 .bin）权重只读解析，零第三方依赖、纯安全 Rust。
//!
//! 该格式是一个 ZIP_STORED 归档：`model/data.pkl` 是 pickle 流，`model/data/<key>`
//! 是各 storage 的原始字节。本模块先经 `crate::zip` 解归档，再用一个栈式 pickle 虚拟机解释
//! data.pkl：拦截 `torch._utils._rebuild_tensor_v2` 的重建与 `BINPERSID` 的 persistent storage，
//! 把每个张量还原成「名字 + 元素类型 + 形状 + 指向 data 区的字节区间」。原始字节不复制，按需借用。
//!
//! 虚拟机只处理 pickle 数据模型与 torch 重建这两类有界操作；未知 opcode 与无法识别的可调用对象
//! 显式报错，不静默执行任意代码，也不猜测结果。

use diagnostic::{Code, Error, Result};
use dtype::DType;

use crate::zip::Zip;

/// 一个张量的元数据；字节落在所属 storage 的 data 文件中。
#[derive(Debug, Clone)]
pub struct Tensor {
    name: String,
    dtype: DType,
    shape: Vec<u64>,
    storage_key: String,
    /// 相对 storage 数据起点的字节偏移。
    offset: usize,
    /// 字节长度（由形状与元素宽度推出）。
    len: usize,
}

impl Tensor {
    /// 张量名。
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Storage 元素类型（全精度闭集的直接成员）。
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 形状。
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// 元素总数。
    #[must_use]
    pub fn count(&self) -> usize {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &d| {
                usize::try_from(d).ok().and_then(|n| acc.checked_mul(n))
            })
            .unwrap_or(0)
    }
}

/// 已解析的 .bin，持有调用方字节切片的借用。
pub struct Bin<'a> {
    zip: Zip<'a>,
    tensors: Vec<Tensor>,
}

impl<'a> Bin<'a> {
    /// 解析归档与 pickle，重建全部张量。
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let zip = Zip::parse(bytes)?;
        // 顶层目录名随文件名变化，按后缀定位 data.pkl。
        let pkl = zip.data_suffix("data.pkl");
        let pkl = pkl.or_else(|_| zip.data_suffix("/data.pkl"))?;
        let tensors = Vm::run(pkl)?;
        Ok(Self { zip, tensors })
    }

    /// 全部张量。
    #[must_use]
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    /// 按名取张量。
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// 张量的原始字节（借用，不复制）。
    pub fn data(&self, tensor: &Tensor) -> Result<&'a [u8]> {
        // 顶层目录名不固定，按 `/data/<key>` 后缀定位所属 storage。
        let suffix = format!("/data/{}", tensor.storage_key);
        let storage = self.zip.data_suffix(&suffix);
        let fallback = format!("data/{}", tensor.storage_key);
        let storage = storage.or_else(|_| self.zip.data_suffix(&fallback))?;
        storage
            .get(tensor.offset..tensor.offset + tensor.len)
            .ok_or_else(|| bad("张量字节区间越界"))
    }

    /// 按名取原始字节。
    pub fn data_named(&self, name: &str) -> Result<&'a [u8]> {
        let tensor = self.tensor(name);
        let tensor = tensor.ok_or_else(|| bad(&format!("无张量 `{name}`")))?;
        self.data(tensor)
    }
}

// ------------------------------------------------ pickle 虚拟机

#[derive(Clone)]
enum Value {
    Str(String),
    Int(i64),
    /// NEWTRUE、NEWFALSE 与 BINFLOAT 的占位：权重路径只保持栈平衡，不消费其值。
    Scalar,
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global {
        module: String,
        name: String,
    },
    Storage {
        key: String,
        dtype: DType,
    },
    Tensor,
    Marker,
}

#[derive(Default)]
struct Vm {
    stack: Vec<Value>,
    memo: Vec<Value>,
    at: usize,
    pkl: Vec<u8>,
    tensors: Vec<Tensor>,
}

impl Vm {
    fn run(pkl: &[u8]) -> Result<Vec<Tensor>> {
        let mut vm = Vm {
            pkl: pkl.to_vec(),
            ..Vm::default()
        };
        vm.execute()?;
        Ok(vm.tensors)
    }

    /// 逐个 opcode 执行到 `.`（STOP）。
    fn execute(&mut self) -> Result<()> {
        loop {
            let op = self.byte()?;
            if op == b'.' {
                return Ok(());
            }
            self.step(op)?;
        }
    }

    /// 取一个字节；越界即「pickle 意外结束」。
    fn byte(&mut self) -> Result<u8> {
        let b = *or_bad(self.pkl.get(self.at), "pickle 意外结束")?;
        self.at += 1;
        Ok(b)
    }

    /// 取 n 字节并前移游标；越界即「pickle 字段越界」。
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let slice = or_bad(self.pkl.get(self.at..self.at + n), "pickle 字段越界")?;
        self.at += n;
        Ok(slice.to_vec())
    }

    /// 小端读 n 字节无符号数（n ≤ 8）。
    fn uint(&mut self, n: usize) -> Result<u64> {
        let raw = self.bytes(n)?;
        let value = raw
            .iter()
            .rev()
            .fold(0u64, |acc, &b| (acc << 8) | u64::from(b));
        Ok(value)
    }

    /// n 字节 UTF-8 文本。
    fn text(&mut self, n: usize) -> Result<String> {
        let raw = self.bytes(n)?;
        String::from_utf8(raw).map_err(|_| bad("字符串不是 UTF-8"))
    }

    /// 读到换行为止的一行，换行本身被消费。
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

    /// 分派帧头、占位容器与 NEWOBJ，其余交给 [`Vm::step_text`]。
    fn step(&mut self, op: u8) -> Result<()> {
        match op {
            b'\x80' => self.bytes(1).map(drop),
            b'\x95' => self.bytes(8).map(drop),
            b'(' | b'}' | b']' | b')' | b'\x89' | b'\x88' => self.placeholder(op),
            b'\x81' => self.newobj(),
            _ => self.step_text(op),
        }
    }

    /// 分派字符串与 GLOBAL，其余交给 [`Vm::step_global`]。
    fn step_text(&mut self, op: u8) -> Result<()> {
        match op {
            b'X' | b'\x8c' => self.string(op),
            b'G' => self.float(),
            b'c' => self.global(),
            _ => self.step_global(op),
        }
    }

    /// 分派 STACK_GLOBAL 与整数，其余交给 [`Vm::step_tuple`]。
    fn step_global(&mut self, op: u8) -> Result<()> {
        match op {
            b'\x93' => self.global_stack(),
            b'K' | b'M' => self.push_uint(op),
            b'J' => self.push_binint(),
            b't' | b'\x85'..=b'\x87' => self.tuple(op),
            _ => self.step_tuple(op),
        }
    }

    /// 分派 memo 读写与张量重建，其余交给 [`Vm::step_command`]。
    fn step_tuple(&mut self, op: u8) -> Result<()> {
        match op {
            b'q' | b'r' | b'\x94' | b'h' | b'j' => self.memo_op(op),
            b'Q' => self.persistent_load(),
            b'R' => self.reduce(),
            _ => self.step_command(op),
        }
    }

    /// 分派容器写入、列表追加、BUILD 与未知 opcode。
    fn step_command(&mut self, op: u8) -> Result<()> {
        match op {
            b'u' => self.set_items(),
            b's' => self.set_item(),
            b'a' | b'e' => self.append(op),
            b'b' => self.pop().map(drop),
            other => Err(bad(&format!("未知 pickle opcode 0x{other:02X}"))),
        }
    }

    /// NEWOBJ 不构造任意类：弹掉两个操作数后报错。
    fn newobj(&mut self) -> Result<()> {
        self.pop()?;
        self.pop()?;
        Err(bad("NEWOBJ 不支持的类"))
    }

    /// 六个只压一个初值的 opcode：三种空容器、标记与布尔占位。
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

    /// BINUNICODE 的长度是 4 字节小端，SHORT_BINUNICODE 是 1 字节。
    fn string(&mut self, op: u8) -> Result<()> {
        let len = self.uint(if op == b'X' { 4 } else { 1 })? as usize;
        let text = self.text(len)?;
        self.stack.push(Value::Str(text));
        Ok(())
    }

    /// BINFLOAT 的 8 字节大端值不参与重建，只占栈位。
    fn float(&mut self) -> Result<()> {
        self.bytes(8).map(drop)?;
        self.stack.push(Value::Scalar);
        Ok(())
    }

    /// GLOBAL 在磁盘上是两行：module 与 name。
    fn global(&mut self) -> Result<()> {
        let module = self.line()?;
        let name = self.line()?;
        self.stack.push(Value::Global { module, name });
        Ok(())
    }

    /// STACK_GLOBAL 从栈上取同样两项：先 name，后 module。
    fn global_stack(&mut self) -> Result<()> {
        let name = into_str(&self.pop()?)?;
        let module = into_str(&self.pop()?)?;
        self.stack.push(Value::Global { module, name });
        Ok(())
    }

    /// BININT1/BININT2 是 1/2 字节小端无符号。
    fn push_uint(&mut self, op: u8) -> Result<()> {
        let width = if op == b'M' { 2 } else { 1 };
        let value = self.uint(width)? as i64;
        self.stack.push(Value::Int(value));
        Ok(())
    }

    /// BININT 是 4 字节小端补码。
    fn push_binint(&mut self) -> Result<()> {
        let raw = self.uint(4)? as u32 as i32;
        self.stack.push(Value::Int(i64::from(raw)));
        Ok(())
    }

    /// TUPLE 弹到标记为止，TUPLE1/2/3 弹固定个数。
    fn tuple(&mut self, op: u8) -> Result<()> {
        let items = match op {
            b't' => self.pop_until_marker()?,
            _ => self.pops(usize::from(op - 0x84))?,
        };
        self.stack.push(Value::Tuple(items));
        Ok(())
    }

    /// 弹 n 项，返回时恢复压栈顺序。
    fn pops(&mut self, n: usize) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        for _ in 0..n {
            items.push(self.pop()?);
        }
        items.reverse();
        Ok(items)
    }

    /// memo 下标：BINPUT/BINGET 1 字节，LONG_* 4 字节，MEMOIZE 取末位。
    fn memo_index(&mut self, op: u8) -> Result<usize> {
        match op {
            b'q' | b'h' => Ok(usize::from(self.byte()?)),
            b'r' | b'j' => Ok(self.uint(4)? as usize),
            _ => Ok(self.memo.len()),
        }
    }

    /// BINPUT/BINPUT_LONG/MEMOIZE 写 memo，BINGET/BINGET_LONG 回读。
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

    /// BINPERSID：栈顶是 persistent id 元组，转换为 storage 句柄。
    fn persistent_load(&mut self) -> Result<()> {
        let items = as_tuple(self.pop()?, "persistent id 不是元组")?;
        let (key, dtype) = storage_handle(&items)?;
        self.stack.push(Value::Storage { key, dtype });
        Ok(())
    }

    /// REDUCE：可调用对象必须是 GLOBAL，随后按 `module.name` 分派。
    fn reduce(&mut self) -> Result<()> {
        let args = self.pop()?;
        let (module, name) = match self.pop()? {
            Value::Global { module, name } => (module, name),
            _ => return Err(bad("REDUCE 的可调用对象不是 GLOBAL")),
        };
        self.reduced(&module, &name, args)
    }

    /// 只接 `_rebuild_tensor_v2` 与 `OrderedDict` 两个可调用对象。
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

    /// `_rebuild_tensor_v2`：把 `(storage, offset, shape, stride)` 还原成张量。
    fn rebuild_tensor(&mut self, args: Value) -> Result<()> {
        let items = as_tuple(args, "rebuild 参数不是元组")?;
        let mut tensor = tensor_of(&items)?;
        // 张量名在调用前已作为 dict 键压栈：取栈顶最近的字符串。
        tensor.name = self.tensor_name()?;
        self.tensors.push(tensor);
        self.stack.push(Value::Tensor);
        Ok(())
    }

    /// 栈顶最近的字符串即张量名。
    fn tensor_name(&self) -> Result<String> {
        let name = self.stack.iter().rev().find_map(|v| match v {
            Value::Str(s) => Some(s),
            _ => None,
        });
        or_bad(name.cloned(), "找不到张量名")
    }

    /// 弹出标记之前的全部项，按「键值成对」切开；奇数项说明 pickle 自相矛盾，必须报错。
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

    /// APPEND 追加一项，APPENDS 追加标记之前的一整段。
    fn append(&mut self, op: u8) -> Result<()> {
        let (items, label) = if op == b'a' {
            (vec![self.pop()?], "APPEND")
        } else {
            (self.pop_until_marker()?, "APPENDS")
        };
        self.extend_list(items, label)
    }

    /// 把一批项接在栈顶 list 之后。
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

/// 取出元组，否则报给定文案。
fn as_tuple(value: Value, message: &str) -> Result<Vec<Value>> {
    match value {
        Value::Tuple(items) => Ok(items),
        _ => Err(bad(message)),
    }
}

/// 取第 `index` 项，缺失时报给定文案。
fn at<'v>(items: &'v [Value], index: usize, message: &str) -> Result<&'v Value> {
    or_bad(items.get(index), message)
}

/// 取第 `index` 项并要求是字符串。
fn at_str(items: &[Value], index: usize, message: &str) -> Result<String> {
    into_str(at(items, index, message)?)
}

/// persistent id 的第一项必须是 `storage`。
fn check_tag(items: &[Value]) -> Result<()> {
    let tag = into_str(at(items, 0, "pid 空")?)?;
    if tag == "storage" {
        return Ok(());
    }
    Err(bad(&format!("不支持的 persistent id `{tag}`")))
}

/// storage 类的 GLOBAL 名（`torch.FloatStorage` 等）。
fn storage_class(items: &[Value]) -> Result<&str> {
    match at(items, 1, "pid 缺 storage 类")? {
        Value::Global { name, .. } => Ok(name.as_str()),
        _ => Err(bad("storage 类不是 GLOBAL")),
    }
}

/// persistent id `("storage", 类, key)` → (storage 键, 元素类型)。
fn storage_handle(items: &[Value]) -> Result<(String, DType)> {
    check_tag(items)?;
    let dtype = storage_dtype(storage_class(items)?)?;
    let key = at_str(items, 2, "pid 缺 key")?;
    Ok((key, dtype))
}

/// rebuild 参数的第 0 项：storage 键与元素类型。
fn storage_of(items: &[Value]) -> Result<(String, DType)> {
    match items.first() {
        Some(Value::Storage { key, dtype }) => Ok((key.clone(), *dtype)),
        _ => Err(bad("rebuild 缺 storage")),
    }
}

/// rebuild 参数的第 1 项：以 storage 元素数计的偏移。
fn offset_of(items: &[Value]) -> Result<usize> {
    if let Some(Value::Int(n)) = items.get(1) {
        return usize::try_from(*n).map_err(|_| bad("offset 溢出"));
    }
    Err(bad("rebuild 缺 offset"))
}

/// rebuild 参数的第 2 项：形状与元素总数（逐维转 `u64`，乘积溢出即报错）。
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

/// 一个形状维度：整数且装得进 `u64`。
fn dim(value: &Value) -> Result<u64> {
    match value {
        Value::Int(n) => u64::try_from(*n).map_err(|_| bad("形状维度溢出")),
        _ => Err(bad("形状维度不是整数")),
    }
}

/// 由 rebuild 参数造张量元数据；名字由调用方稍后补上。
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

/// 偏移与长度都按元素宽度放大；两段各自的溢出文案不同。
fn span(offset: usize, count: usize, width: usize) -> Result<(usize, usize)> {
    let offset = or_bad(offset.checked_mul(width), "offset 溢出")?;
    let len = or_bad(count.checked_mul(width), "长度溢出")?;
    Ok((offset, len))
}

/// torch 的 storage 类名 → 元素类型（`torch.save` 只写这九种）。
fn storage_dtype(name: &str) -> Result<DType> {
    STORAGE_DTYPES
        .iter()
        .find_map(|&(class, dtype)| (class == name).then_some(dtype))
        .ok_or_else(|| bad(&format!("不支持的 storage 类型 `{name}`")))
}

/// 九种 storage 类名与元素类型的对应表。
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

/// `Option` → `Result`：缺失时报给定文案。
fn or_bad<T>(value: Option<T>, message: &str) -> Result<T> {
    value.ok_or_else(|| bad(message))
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("bin")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_bytes() -> Option<Vec<u8>> {
        std::fs::read("/tmp/pt/model.bin").ok()
    }

    #[test]
    fn parses_real_torch_bin() {
        let Some(raw) = real_bytes() else {
            return;
        };
        let bin = Bin::parse(&raw).expect("解析");
        assert_eq!(bin.tensors().len(), 4);

        let w = bin.tensor("layer.weight").expect("张量");
        assert_eq!(w.dtype(), DType::F32);
        assert_eq!(w.shape(), &[4, 6]);
        assert_eq!(bin.data(w).expect("字节").len(), 96);

        let embed = bin.tensor("embed").expect("f16");
        assert_eq!(embed.dtype(), DType::F16);
        assert_eq!(bin.data(embed).expect("字节").len(), 24);

        let count = bin.tensor("count").expect("i64");
        assert_eq!(count.dtype(), DType::I64);
        assert_eq!(bin.data(count).expect("字节").len(), 24);
    }

    #[test]
    fn bytes_match_torch_storage() {
        let Some(raw) = real_bytes() else {
            return;
        };
        let bin = Bin::parse(&raw).expect("解析");
        let bias = bin.tensor("layer.bias").expect("bias");
        let bytes = bin.data(bias).expect("字节");
        // 4 个 f32 = 16 字节，且与 data/1 原始字节一致。
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn rejects_non_bin() {
        assert!(Bin::parse(b"definitely not a torch bin").is_err());
    }

    #[test]
    fn an_odd_setitems_is_an_error_not_a_panic() {
        // `}` 建 dict、`(` 起项、`K\x01` 一项、`u` 为 SETITEMS：项数由文件决定，奇数即自相矛盾。
        let error = Vm::run(b"}(K\x01u.").expect_err("奇数项必须报错");
        assert!(error.to_string().contains("奇数"), "{error}");
    }
}

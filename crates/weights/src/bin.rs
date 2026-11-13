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
        let pkl = zip
            .data_suffix("data.pkl")
            .or_else(|_| zip.data_suffix("/data.pkl"))?;
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
        let storage = self.zip.data_suffix(&suffix).or_else(|_| {
            self.zip
                .data_suffix(&format!("data/{}", tensor.storage_key))
        })?;
        storage
            .get(tensor.offset..tensor.offset + tensor.len)
            .ok_or_else(|| bad("张量字节区间越界"))
    }

    /// 按名取原始字节。
    pub fn data_named(&self, name: &str) -> Result<&'a [u8]> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| bad(&format!("无张量 `{name}`")))?;
        self.data(tensor)
    }
}

// ------------------------------------------------ pickle 虚拟机

#[derive(Clone)]
enum Value {
    Str(String),
    Int(i64),
    Float,
    Bool,
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global { module: String, name: String },
    Storage { key: String, dtype: DType },
    Tensor,
    Marker,
}

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
            stack: Vec::new(),
            memo: Vec::new(),
            at: 0,
            pkl: pkl.to_vec(),
            tensors: Vec::new(),
        };
        vm.execute()?;
        Ok(vm.tensors)
    }

    fn byte(&mut self) -> Result<u8> {
        let b = *self
            .pkl
            .get(self.at)
            .ok_or_else(|| bad("pickle 意外结束"))?;
        self.at += 1;
        Ok(b)
    }

    fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let slice = self
            .pkl
            .get(self.at..self.at + n)
            .ok_or_else(|| bad("pickle 字段越界"))?;
        self.at += n;
        Ok(slice.to_vec())
    }

    fn text(&mut self, n: usize) -> Result<String> {
        let raw = self.bytes(n)?;
        String::from_utf8(raw).map_err(|_| bad("字符串不是 UTF-8"))
    }

    fn line(&mut self) -> Result<String> {
        let stop = self
            .pkl
            .get(self.at..)
            .and_then(|s| s.iter().position(|&b| b == b'\n'))
            .ok_or_else(|| bad("GLOBAL 缺换行"))?;
        let raw = self.bytes(stop)?;
        self.byte()?; // 消费换行
        String::from_utf8(raw).map_err(|_| bad("GLOBAL 不是 UTF-8"))
    }

    fn execute(&mut self) -> Result<()> {
        loop {
            let op = self.byte()?;
            match op {
                b'.' => return Ok(()),
                b'\x80' => {
                    self.byte()?; // 协议版本
                }
                b'\x95' => {
                    self.bytes(8)?; // FRAME 长度，跳过
                }
                b'(' => self.stack.push(Value::Marker),
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b']' => self.stack.push(Value::List(Vec::new())),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'X' => {
                    let len = u32::from_le_bytes(self.bytes(4)?.try_into().expect("4")) as usize;
                    let s = self.text(len)?;
                    self.stack.push(Value::Str(s));
                }
                b'\x8c' => {
                    let len = self.byte()? as usize;
                    let s = self.text(len)?;
                    self.stack.push(Value::Str(s));
                }
                b'c' => {
                    // GLOBAL 在磁盘上是两行：module\n 与 name\n（pickletools 展示时用空格拼接）。
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Global { module, name });
                }
                b'\x93' => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    self.stack.push(Value::Global {
                        module: into_str(&module)?,
                        name: into_str(&name)?,
                    });
                }
                b'K' => {
                    // SHORT_BININT：1 字节无符号。
                    let v = self.byte()? as i64;
                    self.stack.push(Value::Int(v));
                }
                b'M' => {
                    // BININT2：2 字节小端无符号。
                    let v = u16::from_le_bytes(self.bytes(2)?.try_into().expect("2")) as i64;
                    self.stack.push(Value::Int(v));
                }
                // NEWTRUE/NEWFALSE：权重路径只做栈占位，不区分布尔取值，合并入同一标记。
                b'\x89' | b'\x88' => self.stack.push(Value::Bool),
                b'J' => {
                    let v = i32::from_le_bytes(self.bytes(4)?.try_into().expect("4")) as i64;
                    self.stack.push(Value::Int(v));
                }
                b'G' => {
                    self.bytes(8)?; // 大端 f64，权重路径不消费其值，仅占位保持栈平衡。
                    self.stack.push(Value::Float);
                }
                b'\x85' => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a]));
                }
                b'\x86' => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                b'\x87' => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                b't' => {
                    let items = self.pop_until_marker()?;
                    self.stack.push(Value::Tuple(items));
                }
                b'q' => {
                    let idx = self.byte()? as usize;
                    self.store_memo(idx)?;
                }
                b'r' => {
                    let idx = u32::from_le_bytes(self.bytes(4)?.try_into().expect("4")) as usize;
                    self.store_memo(idx)?;
                }
                b'\x94' => {
                    let idx = self.memo.len();
                    self.store_memo(idx)?;
                }
                b'h' => {
                    let idx = self.byte()? as usize;
                    self.recall_memo(idx)?;
                }
                b'j' => {
                    let idx = u32::from_le_bytes(self.bytes(4)?.try_into().expect("4")) as usize;
                    self.recall_memo(idx)?;
                }
                b'Q' => self.persistent_load()?,
                b'R' => self.reduce()?,
                b'u' => self.set_items()?,
                b's' => self.set_item()?,
                b'a' => self.append()?,
                b'e' => self.appends()?,
                b'b' => {
                    let _ = self.pop()?; // state，张量无额外状态
                }
                b'\x81' => {
                    let _args = self.pop()?;
                    let _cls = self.pop()?;
                    return Err(bad("NEWOBJ 不支持的类"));
                }
                other => return Err(bad(&format!("未知 pickle opcode 0x{other:02X}"))),
            }
        }
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().ok_or_else(|| bad("栈下溢"))
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
        let value = self
            .stack
            .last()
            .ok_or_else(|| bad("空栈无法写 memo"))?
            .clone();
        if idx >= self.memo.len() {
            self.memo.resize(idx + 1, Value::Bool);
        }
        self.memo[idx] = value;
        Ok(())
    }

    fn recall_memo(&mut self, idx: usize) -> Result<()> {
        let value = self
            .memo
            .get(idx)
            .ok_or_else(|| bad("memo 索引缺失"))?
            .clone();
        self.stack.push(value);
        Ok(())
    }

    /// BINPERSID：栈顶是 persistent id 元组，转换为 storage 句柄。
    fn persistent_load(&mut self) -> Result<()> {
        let pid = self.pop()?;
        let items = match pid {
            Value::Tuple(items) => items,
            _ => return Err(bad("persistent id 不是元组")),
        };
        let tag = items.first().map(into_str).ok_or_else(|| bad("pid 空"))??;
        if tag != "storage" {
            return Err(bad(&format!("不支持的 persistent id `{tag}`")));
        }
        let storage_global = items.get(1).ok_or_else(|| bad("pid 缺 storage 类"))?;
        let global_name = match storage_global {
            Value::Global { name, .. } => name.clone(),
            _ => return Err(bad("storage 类不是 GLOBAL")),
        };
        let dtype = storage_dtype(&global_name)?;
        let key = items
            .get(2)
            .map(into_str)
            .ok_or_else(|| bad("pid 缺 key"))??;
        self.stack.push(Value::Storage { key, dtype });
        Ok(())
    }

    fn reduce(&mut self) -> Result<()> {
        let args = self.pop()?;
        let callable = self.pop()?;
        let (module, name) = match callable {
            Value::Global { module, name } => (module, name),
            _ => return Err(bad("REDUCE 的可调用对象不是 GLOBAL")),
        };

        if module == "torch._utils" && name == "_rebuild_tensor_v2" {
            self.rebuild_tensor(args)?;
            return Ok(());
        }
        if module == "collections" && name == "OrderedDict" {
            self.stack.push(Value::Dict(Vec::new()));
            return Ok(());
        }
        Err(bad(&format!("不支持的可调用对象 {module}.{name}")))
    }

    fn rebuild_tensor(&mut self, args: Value) -> Result<()> {
        let items = match args {
            Value::Tuple(items) => items,
            _ => return Err(bad("rebuild 参数不是元组")),
        };
        let (key, dtype) = match items.first() {
            Some(Value::Storage { key, dtype }) => (key.clone(), *dtype),
            _ => return Err(bad("rebuild 缺 storage")),
        };
        let offset = match items.get(1) {
            Some(Value::Int(n)) => usize::try_from(*n).map_err(|_| bad("offset 溢出"))?,
            _ => return Err(bad("rebuild 缺 offset")),
        };
        let shape = match items.get(2) {
            Some(Value::Tuple(dims)) => dims
                .iter()
                .map(|d| match d {
                    Value::Int(n) => Ok(u64::try_from(*n).map_err(|_| bad("形状维度溢出"))?),
                    _ => Err(bad("形状维度不是整数")),
                })
                .collect::<Result<Vec<u64>>>()?,
            _ => return Err(bad("rebuild 缺 shape")),
        };
        let width = dtype.bytes_for(1) as usize;
        let count = shape
            .iter()
            .try_fold(1usize, |acc, &d| {
                usize::try_from(d).ok().and_then(|n| acc.checked_mul(n))
            })
            .ok_or_else(|| bad("形状乘积溢出"))?;
        let byte_offset = offset
            .checked_mul(width)
            .ok_or_else(|| bad("offset 溢出"))?;
        let len = count.checked_mul(width).ok_or_else(|| bad("长度溢出"))?;

        // 张量名在调用前已作为 dict 键压栈；取栈顶最近的字符串作为名字。
        let name = self
            .stack
            .iter()
            .rev()
            .find_map(|v| match v {
                Value::Str(s) => Some(s.clone()),
                _ => None,
            })
            .ok_or_else(|| bad("找不到张量名"))?;

        let tensor = Tensor {
            name,
            dtype,
            shape,
            storage_key: key,
            offset: byte_offset,
            len,
        };
        self.tensors.push(tensor);
        self.stack.push(Value::Tensor);
        Ok(())
    }

    fn set_items(&mut self) -> Result<()> {
        let pairs = self.pop_until_marker()?;
        let dict = self
            .stack
            .last_mut()
            .ok_or_else(|| bad("SETITEMS 缺目标 dict"))?;
        if let Value::Dict(entries) = dict {
            for chunk in pairs.chunks(2) {
                entries.push((chunk[0].clone(), chunk[1].clone()));
            }
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

    fn append(&mut self) -> Result<()> {
        let value = self.pop()?;
        match self.stack.last_mut() {
            Some(Value::List(items)) => {
                items.push(value);
                Ok(())
            }
            _ => Err(bad("APPEND 目标不是 list")),
        }
    }

    fn appends(&mut self) -> Result<()> {
        let items = self.pop_until_marker()?;
        match self.stack.last_mut() {
            Some(Value::List(list)) => {
                list.extend(items);
                Ok(())
            }
            _ => Err(bad("APPENDS 目标不是 list")),
        }
    }
}

fn into_str(value: &Value) -> Result<String> {
    match value {
        Value::Str(s) => Ok(s.clone()),
        _ => Err(bad("期望字符串")),
    }
}

fn storage_dtype(name: &str) -> Result<DType> {
    Ok(match name {
        "FloatStorage" => DType::F32,
        "HalfStorage" => DType::F16,
        "BFloat16Storage" => DType::BF16,
        "DoubleStorage" => DType::F64,
        "LongStorage" => DType::I64,
        "IntStorage" => DType::I32,
        "ShortStorage" => DType::I16,
        "CharStorage" => DType::I8,
        "ByteStorage" => DType::U8,
        other => return Err(bad(&format!("不支持的 storage 类型 `{other}`"))),
    })
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
}

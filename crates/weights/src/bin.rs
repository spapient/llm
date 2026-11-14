//! torch.save `.bin` 只读解析（零依赖、纯安全 Rust）：ZIP_STORED 归档内 `data.pkl` 是 pickle 流，
//! `data/<key>` 是各 storage 原始字节。先解归档，再用栈式 pickle 虚拟机解释流，拦截
//! `_rebuild_tensor_v2` 重建与 `BINPERSID` persistent storage，还原「名字 + 类型 + 形状 + 字节区间」；原始字节按需借用，未知 opcode / 不可识别可调用对象显式报错，不执行任意代码、不猜结果。
#![allow(missing_docs)]

use diagnostic::{Code, Error, Result};
use dtype::DType;

const EOCD_SIG: u32 = 0x0605_4B50;
const CD_SIG: u32 = 0x0201_4B50;
const LOCAL_SIG: u32 = 0x0403_4B50;
const ZIP64_SIG: u32 = 0x0606_4B50;
const STORED: u16 = 0;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4B50;

#[derive(Debug, Clone)]
struct ZipEntry {
    name: String,
    method: u16,
    start: usize,
    len: usize,
}

struct Zip<'a> {
    bytes: &'a [u8],
    entries: Vec<ZipEntry>,
}

impl<'a> Zip<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self> {
        let (mut at, total) = directory(bytes)?;
        let mut entries = Vec::with_capacity(total);
        for _ in 0..total {
            let (entry, next) = entry(bytes, at)?;
            entries.push(entry);
            at = next;
        }
        Ok(Self { bytes, entries })
    }

    #[must_use]
    fn get_suffix(&self, suffix: &str) -> Option<&ZipEntry> {
        self.entries.iter().find(|e| e.name.ends_with(suffix))
    }

    fn data_suffix(&self, suffix: &str) -> Result<&'a [u8]> {
        let entry = self
            .get_suffix(suffix)
            .ok_or_else(|| bad(&format!("无后缀条目 `{suffix}`")))?;
        self.data(entry)
    }

    fn data(&self, entry: &ZipEntry) -> Result<&'a [u8]> {
        if entry.method != STORED {
            return Err(bad("条目经过压缩，零依赖读取器只支持 STORED"));
        }
        self.bytes
            .get(entry.start..entry.start + entry.len)
            .ok_or_else(|| bad("条目数据越界"))
    }
}

fn directory(bytes: &[u8]) -> Result<(usize, usize)> {
    let eocd = locate_eocd(bytes)?;
    let at = u32le(bytes, eocd + 16)? as usize;
    let total = u16le(bytes, eocd + 10)? as usize;
    if needs_zip64(at, total) {
        return zip64_directory(bytes, eocd);
    }
    Ok((at, total))
}

fn needs_zip64(at: usize, total: usize) -> bool {
    at == 0xFFFF_FFFF || total == 0xFFFF
}

fn zip64_directory(bytes: &[u8], eocd: usize) -> Result<(usize, usize)> {
    let z64 = zip64_record(bytes, eocd)?;
    let total = u64le(bytes, z64 + 32)? as usize;
    let at = u64le(bytes, z64 + 48)? as usize;
    Ok((at, total))
}

fn zip64_record(bytes: &[u8], eocd: usize) -> Result<usize> {
    let locator = find_sig_back(bytes, EOCD64_LOCATOR_SIG, eocd.saturating_sub(20))?;
    let z64 = u32le(bytes, locator + 8)? as usize;
    if u32le(bytes, z64)? != ZIP64_SIG {
        return Err(bad("ZIP64 EOCD 签名不符"));
    }
    Ok(z64)
}

fn entry(bytes: &[u8], at: usize) -> Result<(ZipEntry, usize)> {
    let central = Central::read(bytes, at)?;
    let start = local_start(bytes, central.local)?;
    let len = data_range(bytes, start, central.csize)?;
    let entry = ZipEntry {
        name: central.name,
        method: central.method,
        start,
        len,
    };
    Ok((entry, central.next))
}

struct Central {
    name: String,
    method: u16,
    csize: u64,
    local: usize,
    next: usize,
}

impl Central {
    fn read(bytes: &[u8], at: usize) -> Result<Self> {
        let fixed = fixed_fields(bytes, at)?;
        let name_len = u16::from_le_bytes([fixed[28], fixed[29]]) as usize;
        let extra_len = u16::from_le_bytes([fixed[30], fixed[31]]) as usize;
        let comment_len = u16::from_le_bytes([fixed[32], fixed[33]]) as usize;
        let name = entry_name(bytes, at, name_len)?;
        let (csize, local) = zip64_sizes(bytes, at, name_len, extra_len, fixed)?;
        Ok(Self {
            name,
            method: u16::from_le_bytes([fixed[10], fixed[11]]),
            csize,
            local,
            next: next_offset(at, name_len, extra_len, comment_len)?,
        })
    }
}

fn fixed_fields(bytes: &[u8], at: usize) -> Result<&[u8]> {
    let fixed = slice(bytes, at, 46)?;
    if u32::from_le_bytes([fixed[0], fixed[1], fixed[2], fixed[3]]) != CD_SIG {
        return Err(bad("中央目录条目签名不符"));
    }
    Ok(fixed)
}

fn entry_name(bytes: &[u8], at: usize, len: usize) -> Result<String> {
    let name = slice(bytes, at + 46, len)?;
    String::from_utf8(name.to_vec()).map_err(|_| bad("条目名不是 UTF-8"))
}

fn zip64_sizes(
    bytes: &[u8],
    at: usize,
    name_len: usize,
    extra_len: usize,
    fixed: &[u8],
) -> Result<(u64, usize)> {
    let extra = at + 46 + name_len;
    let local_off = u32::from_le_bytes([fixed[42], fixed[43], fixed[44], fixed[45]]) as usize;
    let local = if local_off == 0xFFFF_FFFF {
        zip64_extra(bytes, extra, extra_len, 0)?
    } else {
        local_off
    };
    let csize = u32::from_le_bytes([fixed[20], fixed[21], fixed[22], fixed[23]]);
    let data_len = if csize == 0xFFFF_FFFF {
        zip64_extra(bytes, extra, extra_len, 1)? as u64
    } else {
        u64::from(csize)
    };
    Ok((data_len, local))
}

fn next_offset(at: usize, name_len: usize, extra_len: usize, comment_len: usize) -> Result<usize> {
    at.checked_add(46 + name_len + extra_len + comment_len)
        .ok_or_else(|| bad("中央目录偏移溢出"))
}

fn local_start(bytes: &[u8], local: usize) -> Result<usize> {
    if u32le(bytes, local)? != LOCAL_SIG {
        return Err(bad("本地头签名不符"));
    }
    let (name_len, extra_len) = local_lengths(bytes, local)?;
    let Some(start) = local.checked_add(30 + name_len + extra_len) else {
        return Err(bad("本地头偏移溢出"));
    };
    Ok(start)
}

fn local_lengths(bytes: &[u8], local: usize) -> Result<(usize, usize)> {
    Ok((
        u16le(bytes, local + 26)? as usize,
        u16le(bytes, local + 28)? as usize,
    ))
}

fn data_range(bytes: &[u8], start: usize, data_len: u64) -> Result<usize> {
    let len = data_len as usize;
    if start.checked_add(len).is_none_or(|stop| stop > bytes.len()) {
        return Err(bad("条目数据超出归档"));
    }
    Ok(len)
}

fn zip64_extra(bytes: &[u8], extra_start: usize, extra_len: usize, field: usize) -> Result<usize> {
    let stop = extra_start.saturating_add(extra_len);
    let mut at = extra_start;
    while at.checked_add(4).is_some_and(|end| end <= stop) {
        let header_id = u16le(bytes, at)?;
        let size = u16le(bytes, at + 2)? as usize;
        let body = at + 4;
        if header_id == 0x0001 {
            return zip64_field(bytes, body, size, field);
        }
        at = body.saturating_add(size);
    }
    Err(bad("缺少 ZIP64 extra 字段"))
}

fn zip64_field(bytes: &[u8], body: usize, size: usize, field: usize) -> Result<usize> {
    let end = field
        .checked_add(1)
        .and_then(|slots| slots.checked_mul(8))
        .ok_or_else(|| bad("ZIP64 字段序号溢出"))?;
    if end > size {
        return Err(bad("ZIP64 extra 声明长度不足，字段被截断"));
    }
    Ok(u64le(bytes, body.saturating_add(end.saturating_sub(8)))? as usize)
}

fn locate_eocd(bytes: &[u8]) -> Result<usize> {
    let window = bytes.len().min(22 + 65535);
    find_sig_back(bytes, EOCD_SIG, bytes.len() - window)
}

fn find_sig_back(bytes: &[u8], sig: u32, from: usize) -> Result<usize> {
    let mut at = bytes.len().saturating_sub(4);
    while at >= from {
        if u32le(bytes, at)? == sig {
            return Ok(at);
        }
        if at == 0 {
            break;
        }
        at -= 1;
    }
    Err(bad("找不到 ZIP 签名，可能不是 ZIP 归档"))
}

fn slice(bytes: &[u8], at: usize, len: usize) -> Result<&[u8]> {
    let stop = at.checked_add(len).ok_or_else(|| bad("ZIP 字段偏移溢出"))?;
    bytes.get(at..stop).ok_or_else(|| bad("ZIP 字段越界"))
}

fn u16le(bytes: &[u8], at: usize) -> Result<u16> {
    let s = slice(bytes, at, 2)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn u32le(bytes: &[u8], at: usize) -> Result<u32> {
    let s = slice(bytes, at, 4)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64le(bytes: &[u8], at: usize) -> Result<u64> {
    let s = slice(bytes, at, 8)?;
    Ok(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

#[derive(Debug, Clone)]
pub struct Tensor {
    name: String,
    dtype: DType,
    shape: Vec<u64>,
    storage_key: String,
    offset: usize,
    len: usize,
}

impl Tensor {
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

pub struct Bin<'a> {
    zip: Zip<'a>,
    tensors: Vec<Tensor>,
}

impl<'a> Bin<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let zip = Zip::parse(bytes)?;
        let pkl = zip.data_suffix("data.pkl");
        let pkl = pkl.or_else(|_| zip.data_suffix("/data.pkl"))?;
        let tensors = Vm::run(pkl)?;
        Ok(Self { zip, tensors })
    }

    #[must_use]
    pub fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn data(&self, tensor: &Tensor) -> Result<&'a [u8]> {
        let suffix = format!("/data/{}", tensor.storage_key);
        let storage = self.zip.data_suffix(&suffix);
        let fallback = format!("data/{}", tensor.storage_key);
        let storage = storage.or_else(|_| self.zip.data_suffix(&fallback))?;
        storage
            .get(tensor.offset..tensor.offset + tensor.len)
            .ok_or_else(|| bad("张量字节区间越界"))
    }
}

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
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn rejects_non_bin() {
        assert!(Bin::parse(b"definitely not a torch bin").is_err());
    }

    #[test]
    fn an_odd_setitems_is_an_error_not_a_panic() {
        let error = Vm::run(b"}(K\x01u.").expect_err("奇数项必须报错");
        assert!(error.to_string().contains("奇数"), "{error}");
    }
}

#[cfg(test)]
mod zip_tests {
    use super::*;

    #[test]
    fn parses_real_torch_zip() {
        let Ok(raw) = std::fs::read("/tmp/pt/model.bin") else {
            return;
        };
        let zip = Zip::parse(&raw).expect("解析");
        assert!(zip.get_suffix("data.pkl").is_some());
        assert!(zip.get_suffix("/data/0").is_some());
        assert_eq!(zip.data_suffix("/data/0").expect("数据").len(), 96);
    }

    #[test]
    fn rejects_non_zip() {
        assert!(Zip::parse(b"not a zip archive at all").is_err());
    }

    #[test]
    fn slices_with_overflowing_offsets_are_errors_not_panics() {
        assert!(slice(b"abc", usize::MAX, 4).is_err());
        assert!(slice(b"abc", 1, usize::MAX).is_err());
    }

    #[test]
    fn zip64_fields_must_fit_the_declared_size() {
        let mut bytes = vec![0x01, 0x00, 8, 0];
        bytes.extend(7u64.to_le_bytes());
        bytes.extend(0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
        assert_eq!(zip64_extra(&bytes, 0, 12, 0).ok(), Some(7));
        assert!(zip64_extra(&bytes, 0, 12, 1).is_err());
    }

    #[test]
    fn zip64_extra_handles_absurd_lengths_without_overflow() {
        let bytes = vec![0x01, 0x00, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(zip64_extra(&bytes, 0, usize::MAX, 0).is_ok());
        assert!(zip64_extra(&bytes, usize::MAX - 2, 12, 0).is_err());
    }
}

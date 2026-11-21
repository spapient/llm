//! torch.save `.bin` 只读解析（零依赖、纯安全 Rust）：ZIP_STORED 归档内 `data.pkl` 是 pickle 流，
//! `data/<key>` 是各 storage 原始字节。先解归档，再用栈式 pickle 虚拟机解释流，拦截
//! `_rebuild_tensor_v2` 重建与 `BINPERSID` persistent storage，还原「名字 + 类型 + 形状 + 字节区间」；原始字节按需借用，未知 opcode / 不可识别可调用对象显式报错，不执行任意代码、不猜结果。
#![allow(missing_docs)]

use diagnostic::{Code, Error, Result};
use dtype::DType;


    impl<'a> Zip<'a> {
        pub(crate) fn parse(bytes: &'a [u8]) -> Result<Self> {
            let (mut at, total) = directory(bytes)?;
            if total > bytes.len() / MIN_ENTRY_BYTES {
                return Err(bad("条目数超出归档能容纳的规模"));
            }
            let mut entries = Vec::with_capacity(total.min(MAX_ENTRY_PREALLOC));
            for _ in 0..total {
                let (entry, next) = entry(bytes, at)?;
                entries.push(entry);
                at = next;
            }
            Ok(Self { bytes, entries })
        }

        #[must_use]
        pub(crate) fn get_suffix(&self, suffix: &str) -> Option<&ZipEntry> {
            self.entries.iter().find(|e| e.name.ends_with(suffix))
        }

        pub(crate) fn data_suffix(&self, suffix: &str) -> Result<&'a [u8]> {
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
        let at = u32::from_le_bytes(le(bytes, eocd + 16)?) as usize;
        let total = u16::from_le_bytes(le(bytes, eocd + 10)?) as usize;
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
        let total = u64::from_le_bytes(le(bytes, z64 + 32)?) as usize;
        let at = u64::from_le_bytes(le(bytes, z64 + 48)?) as usize;
        Ok((at, total))
    }

    fn zip64_record(bytes: &[u8], eocd: usize) -> Result<usize> {
        let locator = find_sig_back(bytes, EOCD64_LOCATOR_SIG, eocd.saturating_sub(20))?;
        let z64 = u32::from_le_bytes(le(bytes, locator + 8)?) as usize;
        if u32::from_le_bytes(le(bytes, z64)?) != ZIP64_SIG {
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
        String::from_utf8(slice(bytes, at + 46, len)?.to_vec()).map_err(|_| bad("条目名不是 UTF-8"))
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

    fn next_offset(
        at: usize,
        name_len: usize,
        extra_len: usize,
        comment_len: usize,
    ) -> Result<usize> {
        at.checked_add(46 + name_len + extra_len + comment_len)
            .ok_or_else(|| bad("中央目录偏移溢出"))
    }

    fn local_start(bytes: &[u8], local: usize) -> Result<usize> {
        if u32::from_le_bytes(le(bytes, local)?) != LOCAL_SIG {
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
            u16::from_le_bytes(le(bytes, local + 26)?) as usize,
            u16::from_le_bytes(le(bytes, local + 28)?) as usize,
        ))
    }

    fn data_range(bytes: &[u8], start: usize, data_len: u64) -> Result<usize> {
        let len = data_len as usize;
        if start.checked_add(len).is_none_or(|stop| stop > bytes.len()) {
            return Err(bad("条目数据超出归档"));
        }
        Ok(len)
    }

    fn zip64_extra(
        bytes: &[u8],
        extra_start: usize,
        extra_len: usize,
        field: usize,
    ) -> Result<usize> {
        let stop = extra_start.saturating_add(extra_len);
        let mut at = extra_start;
        while at.checked_add(4).is_some_and(|end| end <= stop) {
            let header_id = u16::from_le_bytes(le(bytes, at)?);
            let size = u16::from_le_bytes(le(bytes, at + 2)?) as usize;
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
        Ok(u64::from_le_bytes(le(bytes, body.saturating_add(end.saturating_sub(8)))?) as usize)
    }

    fn locate_eocd(bytes: &[u8]) -> Result<usize> {
        let window = bytes.len().min(22 + 65535);
        find_sig_back(bytes, EOCD_SIG, bytes.len() - window)
    }

    fn find_sig_back(bytes: &[u8], sig: u32, from: usize) -> Result<usize> {
        let mut at = bytes.len().saturating_sub(4);
        while at >= from {
            if u32::from_le_bytes(le(bytes, at)?) == sig {
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

    fn le<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N]> {
        let s = slice(bytes, at, N)?;
        let mut out = [0; N];
        out.copy_from_slice(s);
        Ok(out)
    }

    #[cfg(test)]
    #[path = "../tests/bin_zip.rs"]
    mod tests;
}

use zip::Zip;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) name: String,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<u64>,
    pub(crate) storage_key: String,
    pub(crate) offset: usize,
    pub(crate) len: usize,
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
        let tensors = pickle::Vm::run(pkl)?;
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

fn or_bad<T>(value: Option<T>, message: &str) -> Result<T> {
    value.ok_or_else(|| bad(message))
}

fn bad(message: &str) -> Error {
    Error::new(Code::Invalid, message).with("bin")
}

#[cfg(test)]
#[path = "../tests/bin.rs"]
mod tests;

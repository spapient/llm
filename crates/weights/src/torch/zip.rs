//! ZIP 归档解析：STORED 条目，含 ZIP64 回退；与 pickle 虚拟机分属两个内聚族。

use diagnostic::Result;
use storage::bad;

use crate::cursor::{slice, u16_at, u32_at, u64_at};

// —— ZIP 归档解析（原 `zip`）：STORED 条目，含 ZIP64 回退 ——
//
// 每个函数对应 APPNOTE 的一段：`directory`（尾部 EOCD 或 ZIP64 回退）、`entry`（一条中央目录
// 记录）、`sizes`（32 位尺寸字段与其 ZIP64 槽位）、`zip64_extra`（extra 字段头遍历）。

const EOCD_SIG: u32 = 0x0605_4B50;
const CD_SIG: u32 = 0x0201_4B50;
const LOCAL_SIG: u32 = 0x0403_4B50;
const ZIP64_SIG: u32 = 0x0606_4B50;
const EOCD64_LOCATOR_SIG: u32 = 0x0706_4B50;
/// extra 字段里 ZIP64 的 header id。
const ZIP64_EXTRA_ID: u16 = 0x0001;
const STORED: u16 = 0;
/// 条目最小 46+1 字节；预分配上限 4096，谎报计数先按剩余字节粗查。
const MIN_ENTRY_BYTES: usize = 47;
const MAX_ENTRY_PREALLOC: usize = 4096;
/// 中央目录尺寸字段被截断时的 32 位哨兵。
const WIDE_SENTINEL: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub(crate) struct ZipEntry {
    name: String,
    method: u16,
    start: usize,
    len: usize,
}

pub(crate) struct Zip<'a> {
    bytes: &'a [u8],
    entries: Vec<ZipEntry>,
}

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
            .ok_or_else(|| bad(format!("无后缀条目 `{suffix}`")))?;
        self.data(entry)
    }

    fn data(&self, entry: &ZipEntry) -> Result<&'a [u8]> {
        if entry.method != STORED {
            return Err(bad("条目经过压缩，零依赖读取器只支持 STORED"));
        }
        slice(self.bytes, entry.start, entry.len)
    }
}

/// 中央目录起点与条目总数：EOCD 的两个字段任一为 `0xFFFF…` 就走 ZIP64 回退。
fn directory(bytes: &[u8]) -> Result<(usize, usize)> {
    let eocd = locate_eocd(bytes)?;
    narrow_or_wide(
        bytes,
        eocd,
        u32_at(bytes, eocd + 16)?,
        u16_at(bytes, eocd + 10)?,
    )
}

/// 32 位字段够用就按它换算，任一被截断（`0xFFFF…`）就走 ZIP64 回退。
fn narrow_or_wide(bytes: &[u8], eocd: usize, wide: u32, total: u16) -> Result<(usize, usize)> {
    if wide == WIDE_SENTINEL || total == u16::MAX {
        return zip64_directory(bytes, eocd);
    }
    Ok((
        narrow(u64::from(wide), "中央目录偏移超出 usize")?,
        usize::from(total),
    ))
}

/// ZIP64 回退：定位器在 EOCD 前 20 字节，定位器 +8 指向 64 位 EOCD；
/// 条目数与该 EOCD 的 +32、中央目录偏移在 +48。
fn zip64_directory(bytes: &[u8], eocd: usize) -> Result<(usize, usize)> {
    let z64 = zip64_eocd(bytes, eocd)?;
    let total = wide_value(bytes, z64 + 32, "条目数超出 usize")?;
    let at = wide_value(bytes, z64 + 48, "中央目录偏移超出 usize")?;
    Ok((at, total))
}

/// 定位器的第一跳：它在 EOCD 前 20 字节，定位器 +8 指向 64 位 EOCD。
fn zip64_eocd(bytes: &[u8], eocd: usize) -> Result<usize> {
    let locator = locator_at(eocd)?;
    check_sig(bytes, locator, EOCD64_LOCATOR_SIG, "ZIP64 定位器签名不符")?;
    let z64 = wide_value(bytes, locator + 8, "ZIP64 EOCD 偏移超出 usize")?;
    check_sig(bytes, z64, ZIP64_SIG, "ZIP64 EOCD 签名不符")?;
    Ok(z64)
}

/// 定位器起点：EOCD 前 20 字节，越界说明这个归档没有 ZIP64 段。
fn locator_at(eocd: usize) -> Result<usize> {
    match eocd.checked_sub(20) {
        Some(at) => Ok(at),
        None => Err(bad("ZIP64 定位器不完整")),
    }
}

/// 一个 64 位字段换算成 `usize`：越界即错，不截断。
fn wide_value(bytes: &[u8], at: usize, msg: &str) -> Result<usize> {
    narrow(u64_at(bytes, at)?, msg)
}

/// 64 位真值收窄成 `usize`：收不下就显式失败。
fn narrow(value: u64, msg: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| bad(msg))
}

fn check_sig(bytes: &[u8], at: usize, sig: u32, msg: &str) -> Result<()> {
    if u32_at(bytes, at)? == sig {
        return Ok(());
    }
    Err(bad(msg))
}

/// 一条中央目录记录：名字、方法、数据区间与下一条记录的起点。
/// 记录长 46 字节 + 名字 + extra + 注释依次排开；区间必须落在归档内，否则显式失败。
fn entry(bytes: &[u8], at: usize) -> Result<(ZipEntry, usize)> {
    let lens = Lens::of(cd_head(bytes, at)?)?;
    let (csize, local) = sizes(bytes, at, lens.name)?;
    build(bytes, at, lens, csize, local)
}

/// 固定 46 字节头：签名对得上才有后面的字段，所以先整块切出来。
fn cd_head(bytes: &[u8], at: usize) -> Result<&[u8]> {
    let fixed = slice(bytes, at, 46)?;
    check_sig(fixed, 0, CD_SIG, "中央目录条目签名不符")?;
    Ok(fixed)
}

/// 装配一条记录：名字、数据区间与下一条记录的起点。
fn build(
    bytes: &[u8],
    at: usize,
    lens: Lens,
    csize: u64,
    local: usize,
) -> Result<(ZipEntry, usize)> {
    let name = entry_name(bytes, at, lens.name)?;
    let (start, len) = span(bytes, local, csize)?;
    let next = lens.next(at)?;
    Ok((
        ZipEntry {
            name,
            method: lens.method,
            start,
            len,
        },
        next,
    ))
}

/// 固定头里的方法码与三段长度：条目名、extra、注释依次排开。
struct Lens {
    method: u16,
    name: usize,
    extra: usize,
    comment: usize,
}

impl Lens {
    fn of(fixed: &[u8]) -> Result<Self> {
        Ok(Self {
            method: u16_at(fixed, 10)?,
            name: usize::from(u16_at(fixed, 28)?),
            extra: usize::from(u16_at(fixed, 30)?),
            comment: usize::from(u16_at(fixed, 32)?),
        })
    }

    /// 下一条记录的起点：46 字节固定头 + 名字 + extra + 注释。
    fn next(&self, at: usize) -> Result<usize> {
        match at.checked_add(46 + self.name + self.extra + self.comment) {
            Some(next) => Ok(next),
            None => Err(bad("中央目录偏移溢出")),
        }
    }
}

/// 条目名：中央目录里的名字是 UTF-8 闭集，收不下就是归档坏了。
fn entry_name(bytes: &[u8], at: usize, len: usize) -> Result<String> {
    String::from_utf8(slice(bytes, at + 46, len)?.to_vec()).map_err(|_| bad("条目名不是 UTF-8"))
}

/// 数据区间（起点、长度）：本地头给出起点，区间必须落在归档内。
fn span(bytes: &[u8], local: usize, csize: u64) -> Result<(usize, usize)> {
    let start = local_start(bytes, local)?;
    let len = narrow(csize, "条目长度超出 usize")?;
    if start.checked_add(len).is_none_or(|stop| stop > bytes.len()) {
        return Err(bad("条目数据超出归档"));
    }
    Ok((start, len))
}

/// 中央目录的尺寸字段：32 位真值，或被截断成 `0xFFFF_FFFF` 时回退到 ZIP64 extra。
/// 真值按 APPNOTE 的固定顺序（原长、压缩长、本地头偏移）落在 extra 里，且**只**收下被
/// 截断的那些——所以槽位是「前面被截断的字段个数」，不是字段序号。返回（压缩长，本地头偏移）。
fn sizes(bytes: &[u8], at: usize, name_len: usize) -> Result<(u64, usize)> {
    let extra = at + 46 + name_len;
    let extra_len = usize::from(u16_at(bytes, at + 30)?);
    let mut slot = 0usize;
    let mut wide = [0u64; 3];
    for (idx, offset) in [24, 20, 42].into_iter().enumerate() {
        wide[idx] = wide_field(
            bytes,
            at + offset,
            Extra {
                extra,
                extra_len,
                slot: &mut slot,
            },
        )?;
    }
    Ok((wide[1], narrow(wide[2], "本地头偏移超出 usize")?))
}

/// ZIP64 extra 字段体的位置、声明长度与「前面截断过几个字段」的槽位。
struct Extra<'a> {
    extra: usize,
    extra_len: usize,
    slot: &'a mut usize,
}

/// 一个尺寸字段：32 位真值，或被截断成 `0xFFFF_FFFF` 时按槽位读 ZIP64 extra。
fn wide_field(bytes: &[u8], at: usize, extra: Extra<'_>) -> Result<u64> {
    let narrow = u32_at(bytes, at)?;
    if narrow != WIDE_SENTINEL {
        return Ok(u64::from(narrow));
    }
    let slot = *extra.slot;
    *extra.slot = slot + 1;
    zip64_extra(bytes, extra.extra, extra.extra_len, slot)
}

/// 本地头起点：签名对得上，30 字节固定段 + 名字 + extra 之后就是数据。
fn local_start(bytes: &[u8], local: usize) -> Result<usize> {
    check_sig(bytes, local, LOCAL_SIG, "本地头签名不符")?;
    let name_len = usize::from(u16_at(bytes, local + 26)?);
    let extra_len = usize::from(u16_at(bytes, local + 28)?);
    let Some(start) = local.checked_add(30 + name_len + extra_len) else {
        return Err(bad("本地头偏移溢出"));
    };
    Ok(start)
}

/// ZIP64 extra 里第 `field` 个 8 字节真值：逐个 extra 字段头（id u16 + 长 u16）往前挪，
/// 找到 id 0x0001 就按槽位取；声明长度装不下该槽位显式失败。
pub(crate) fn zip64_extra(
    bytes: &[u8],
    extra_start: usize,
    extra_len: usize,
    field: usize,
) -> Result<u64> {
    let stop = extra_start.saturating_add(extra_len);
    let mut at = extra_start;
    while at.checked_add(4).is_some_and(|end| end <= stop) {
        let id = u16_at(bytes, at)?;
        let size = usize::from(u16_at(bytes, at + 2)?);
        let body = at + 4;
        if id == ZIP64_EXTRA_ID {
            return slot_value(bytes, body, size, field);
        }
        at = body.saturating_add(size);
    }
    Err(bad("缺少 ZIP64 extra 字段"))
}

/// 第 `field` 个 8 字节真值：声明长度装不下该槽位就显式失败（截断不许读成 0）。
fn slot_value(bytes: &[u8], body: usize, size: usize, field: usize) -> Result<u64> {
    let end = field
        .checked_add(1)
        .and_then(|slots| slots.checked_mul(8))
        .ok_or_else(|| bad("ZIP64 字段序号溢出"))?;
    if end > size {
        return Err(bad("ZIP64 extra 声明长度不足，字段被截断"));
    }
    u64_at(bytes, body.saturating_add(end - 8))
}

/// EOCD 在文件尾部 22 + 65535 字节窗口内反向查找；窗口外一律拒（不猜其他签名）。
fn locate_eocd(bytes: &[u8]) -> Result<usize> {
    let from = bytes.len().saturating_sub(22 + 65535);
    let mut at = bytes.len().saturating_sub(4);
    while at >= from {
        if u32_at(bytes, at)? == EOCD_SIG {
            return Ok(at);
        }
        if at == 0 {
            break;
        }
        at -= 1;
    }
    Err(bad("找不到 ZIP 签名，可能不是 ZIP 归档"))
}

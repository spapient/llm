//! 文件传输的远端集合通信：[`FileRemote`]。
//!
//! 它与进程内的 [`crate::Local`] 是同一个 [`crate::Collective`] 端口的两份实现，但只有它碰盘：
//! `world × width` 个 f32 槽位落在一份共享文件里，每个 rank 一个句柄指向同一份文件。
//! 单独成篇的理由不是行数——是"唯一有 I/O 的实现"：这一份要读/写/预扩容文件、要定字节序，
//! 出问题时要看的是文件布局；其余实现只在内存里搬数。

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use crate::Code;
use diagnostic::{Error, Result};

use crate::collective::{Collective, Group, Reduce};

/// 文件传输的远端集合通信：一份共享文件承载 `world × width` 个 f32 槽位，每个 rank 一个
/// 句柄指向同一份文件。缓冲在进程外，因此跨进程可用；无 `unsafe`，传输就是 `std::fs::File`
/// 的按槽读/写（小端字节序，跨机器共享文件时字节布局确定）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRemote {
    rank: u32,
    world: u32,
    width: usize,
    backing: PathBuf,
}

impl FileRemote {
    /// 打开第 `rank` 个进程的句柄：`backing` 是全部 rank 共享的文件路径，槽位按
    /// `rank × width × 4` 字节排列。文件在第一次 `open` 时预留到满长，未写过的槽位读作 0。
    ///
    /// # Errors
    ///
    /// `world` 或 `width` 为零、`rank` 超出 `0..world`、或文件不可创建 / 不可预留。
    pub fn open(rank: u32, world: u32, width: usize, backing: impl Into<PathBuf>) -> Result<Self> {
        if world == 0 {
            fail!("shard::FileRemote::open", "进程数 world 不能为 0");
        }
        if rank >= world {
            fail!("shard::FileRemote::open", "rank {rank} 超出 0..{world}");
        }
        if width == 0 {
            fail!("shard::FileRemote::open", "归约宽度不能为 0");
        }
        let remote = Self {
            rank,
            world,
            width,
            backing: backing.into(),
        };
        remote.resize()?;
        Ok(remote)
    }

    /// 本进程的 rank。
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// 把某个 rank 的输入写进它的槽位；[`Collective::all_reduce`] 只读已写进的槽位。
    ///
    /// # Errors
    ///
    /// 长度不等于归约宽度、`rank` 超出 `0..world`、或文件不可写。
    pub fn load(&self, rank: u32, values: &[f32]) -> Result<()> {
        self.check_width("shard::FileRemote::load", values.len())?;
        self.check_rank("shard::FileRemote::load", rank)?;
        self.write(rank, values)
    }

    /// 某个 rank 当前槽位的值（`load` 或 `all_reduce` 之后）。
    ///
    /// # Errors
    ///
    /// `rank` 超出 `0..world`，或文件不可读。
    pub fn replica(&self, rank: u32) -> Result<Vec<f32>> {
        self.check_rank("shard::FileRemote::replica", rank)?;
        self.read(rank)
    }

    /// 把组内归约写进调用方给的缓冲；长度必须等于归约宽度。与 `Local::reduce_into` 同口径。
    ///
    /// # Errors
    ///
    /// 缓冲长度与归约宽度不一致，或组里出现本 world 之外的 rank。
    pub fn reduce_into(&self, group: &Group, reduce: Reduce, out: &mut [f32]) -> Result<()> {
        self.check_width("shard::FileRemote::reduce_into", out.len())?;
        out.fill(reduce.identity());
        let mut slot = vec![0.0f32; self.width];
        for rank in group.ranks() {
            self.read_into(*rank, &mut slot)?;
            for (acc, value) in out.iter_mut().zip(&slot) {
                *acc = reduce.apply(*acc, *value);
            }
        }
        Ok(())
    }

    fn check_width(&self, site: &'static str, len: usize) -> Result<()> {
        if len == self.width {
            return Ok(());
        }
        refuse_at!(site, "宽度 {len} 与归约宽度 {} 不一致", self.width)
    }

    fn check_rank(&self, site: &'static str, rank: u32) -> Result<()> {
        if rank < self.world {
            return Ok(());
        }
        refuse_at!(site, "rank {rank} 超出 0..{}", self.world)
    }

    /// 字节总长：`world × width × 4`。`open` 时预留，未写过的槽位读作 0。
    fn len_bytes(&self) -> u64 {
        u64::from(self.world) * (self.width as u64) * 4
    }

    /// 第 `rank` 个槽位的字节偏移。
    fn offset(&self, rank: u32) -> u64 {
        u64::from(rank) * (self.width as u64) * 4
    }

    /// 把一次文件动作的 `io::Error` 折成带动作名与路径的 [`Code::Io`] 错误。
    fn io(&self, action: &str, error: std::io::Error) -> Error {
        Error::new(
            Code::Io,
            format!("{action} `{}` 失败：{error}", self.backing.display()),
        )
    }

    /// 只读打开共享文件。
    fn open_read(&self) -> Result<File> {
        File::open(&self.backing).map_err(|error| self.io("打开", error))
    }

    /// 写打开（不创建）共享文件。
    fn open_write(&self) -> Result<File> {
        OpenOptions::new()
            .write(true)
            .open(&self.backing)
            .map_err(|error| self.io("打开", error))
    }

    /// 把文件定位到第 `rank` 个槽位的起点。
    fn seek_slot(&self, file: &mut File, rank: u32) -> Result<()> {
        file.seek(SeekFrom::Start(self.offset(rank)))
            .map_err(|error| self.io("定位", error))
            .map(|_| ())
    }

    fn resize(&self) -> Result<()> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.backing)
            .map_err(|error| self.io("创建", error))?;
        file.set_len(self.len_bytes())
            .map_err(|error| self.io("预留", error))
    }

    fn read_into(&self, rank: u32, out: &mut [f32]) -> Result<()> {
        let mut file = self.open_read()?;
        self.seek_slot(&mut file, rank)?;
        let mut bytes = vec![0u8; self.width * 4];
        file.read_exact(&mut bytes)
            .map_err(|error| self.io("读", error))?;
        for (slot, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *slot = f32::from_le_bytes(*chunk);
        }
        Ok(())
    }

    fn read(&self, rank: u32) -> Result<Vec<f32>> {
        let mut slot = vec![0.0f32; self.width];
        self.read_into(rank, &mut slot)?;
        Ok(slot)
    }

    fn write(&self, rank: u32, values: &[f32]) -> Result<()> {
        let mut file = self.open_write()?;
        self.seek_slot(&mut file, rank)?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        file.write_all(&bytes).map_err(|error| self.io("写", error))
    }
}

impl Collective for FileRemote {
    fn all_reduce(&mut self, group: &Group, reduce: Reduce, buf: &mut [f32]) -> Result<()> {
        self.reduce_into(group, reduce, buf)?;
        for rank in group.ranks() {
            self.write(*rank, buf)?;
        }
        Ok(())
    }

    fn all_gather(&mut self, group: &Group, buf: &mut [f32]) -> Result<()> {
        let width = self.width;
        if buf.len() != group.size() * width {
            return refuse_at!(
                "shard::FileRemote::all_gather",
                "缓冲长度 {} 与 all_gather 期望的 {} 不一致",
                buf.len(),
                group.size() * width
            );
        }
        for (idx, rank) in group.ranks().iter().enumerate() {
            self.read_into(*rank, &mut buf[idx * width..(idx + 1) * width])?;
        }
        Ok(())
    }

    fn broadcast(&mut self, group: &Group, src: u32, buf: &mut [f32]) -> Result<()> {
        self.check_width("shard::FileRemote::broadcast", buf.len())?;
        if !group.ranks().contains(&src) {
            return refuse_at!("shard::FileRemote::broadcast", "源 rank {src} 不在组内");
        }
        buf.copy_from_slice(&self.read(src)?);
        Ok(())
    }

    fn reduce_scatter(&mut self, group: &Group, reduce: Reduce, buf: &mut [f32]) -> Result<()> {
        let width = self.width;
        if buf.len() != group.size() * width {
            return refuse_at!(
                "shard::FileRemote::reduce_scatter",
                "缓冲长度 {} 与 reduce_scatter 期望的 {} 不一致",
                buf.len(),
                group.size() * width
            );
        }
        buf.fill(reduce.identity());
        let mut slot = vec![0.0f32; self.width];
        for (idx, rank) in group.ranks().iter().enumerate() {
            self.read_into(*rank, &mut slot)?;
            for (out, value) in buf[idx * width..(idx + 1) * width].iter_mut().zip(&slot) {
                *out = reduce.apply(*out, *value);
            }
        }
        Ok(())
    }
}

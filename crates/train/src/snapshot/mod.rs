//! 检查点文件：[`Snapshot`] 把一次训练恢复所需的事实与权重条目表编成字节，字节自带校验。
//!
//! **拥有**：`Format`、`Facts`、`Snapshot`——字节布局、完整性校验与原子落盘。**不拥有**：训练
//! 循环状态（[`crate::Checkpoint`] 是内存里的最小事实集，本模块只负责它怎么落盘、怎么读回）、
//! 参数数据（`weights::Header` 被引用，条目表不复制）、张量存储（`tensor`）、检查点存在哪个
//! 路径（`llm-config` 决定根目录）。**依赖**：`weights`、`tensor`、`diagnostic`。
//! **被依赖**：[`crate::Checkpoint`]（本模块并入后与它同 crate）。
//!
//! 它原是一个独立成员（`llm-checkpoint`），唯一消费者是训练循环；并入后"恢复需要知道哪些事实"
//! 只有一处定义：`Checkpoint::snapshot` 与 `Snapshot::decode` 之间不再隔着 crate 边界，
//! `Facts` 也不会出现"声明了却没人校验"的字段。
//!
//! 四个取舍，都是"错了会很贵"那一类：
//!
//! - **字节自证**：`LLMC` 魔数 + 版本 + **CRC-32 尾部**。检查点是上次训练的结论；读回一个被
//!   截断或改过一个比特的文件却照常恢复，比读不出来贵得多——训练会在第 N 步之后才发现参数
//!   与步号对不上，而那时旧状态已经被覆盖。
//! - **编码紧凑且是单射**：长度、计数、形状与偏移全用 LEB128，偏移按条目顺序存**增量**
//!   （[`weights::Header`] 已保证条目按偏移有序）；条目名与上一条共享**前后缀**——真实权重表
//!   里相邻名字往往只差一个词（`blk.3.attn_q.weight` 紧跟着 `blk.3.attn_k.weight`），于是每条
//!   只写"共享前缀长度 + 共享后缀长度 + 中间那几个字节"，名字从"每条一整串"变成"每条几个
//!   字节"。元素类型与形状也相对上一条：同一层里 `gate` / `up` / `down` 往往同型同形，于是每条
//!   的布局只花一个字节（标签 0）。共享长度与"布局重复"各只有一种写法（前缀取最长、后缀取
//!   剩下那一段里最长、同布局必写 0），解码器对每条重算一遍并拒掉任何"还能省却没这么写"的
//!   编码，再拒掉非规范整数与超过 `u64` 的值。于是同一份快照只有一种字节形态：
//!   `decode(encode(x)) == x` 与 `encode(decode(b)) == b` 同时成立。
//! - **长度先算准，写盘不留副本**：`encoded_len` 与 `encode` 走同一个编码器（零重复算术），
//!   数长度只数正文、不顺手再算一遍校验和；`write` 把同一个编码器直接接到同目录临时文件上，
//!   整份检查点的字节从不落进一块"文件大小"的缓冲：写多大的文件，额外内存都只是一个固定
//!   大小的写缓冲。承诺的字节数仍是 `metadata().len()`，与 `autograd::Plan` 的"计划峰值必须
//!   能被实测对上"是同一条约定。
//! - **恢复有两道前置校验**：[`Snapshot::verify`] 只验字节（魔数、版本、CRC-32，不建表），
//!   [`Snapshot::check_weights`] 再确认这份检查点引用的条目表与手头的权重文件逐条相同。
//!   形状对、内容错位是这里最贵的一类失败：它不在读取时炸，而是在训练若干步之后才表现成
//!   loss 不降，那时旧的检查点往往已经被覆盖。
//!
//! 解码不是"读出来就用"：`weights::Header` 会重新校验名字唯一、区间不重叠、起点不早于数据段，
//! 因此手改过的检查点不会等到恢复后的第 N 步才炸。
//!
//! # 例子
//!
//! ```
//! use llm_train::snapshot::{Facts, Snapshot};
//! use tensor::{DType, Shape};
//! use weights::{Entry, Header};
//!
//! let header = Header::new(
//!     vec![Entry::new("blk.0.attn_q.weight", DType::F32, Shape::new([2, 3])).at(64)],
//!     64,
//! )
//! .expect("头部");
//! let facts = Facts::new(7, 2, 2, 3).expect("恢复事实");
//! let snapshot = Snapshot::new(facts, header);
//!
//! let bytes = snapshot.encode();
//! assert_eq!(bytes.len(), snapshot.encoded_len());
//! Snapshot::verify(&bytes).expect("字节自证");
//!
//! let restored = Snapshot::decode(&bytes).expect("解码");
//! assert_eq!(restored.facts(), facts);
//! assert_eq!(restored.weights().entries().len(), 1);
//!
//! // 恢复前的第二道关：这份检查点描述的条目表必须与手头的权重文件逐条相同。
//! let weights = restored.weights().clone();
//! restored.check_weights(&weights).expect("同一份权重表");
//! ```

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use diagnostic::{Code, Error, Result};
use tensor::{DType, Shape};
use weights::{Entry, Header};

/// 解码失败的上下文：错误只由这一层产生，排查时不必再猜是哪一段。
const CONTEXT: &str = "train::snapshot::Snapshot::decode";

/// 一个权重条目在字节里最少占多少：共享前缀、共享后缀、中段长度、布局标签、偏移增量各 1 字节
/// （前三个都可以是 0，但每个字段仍要一个字节）。
///
/// 解码据此把"声明的条目数"与"剩下多少字节"对账，伪造的计数撑不大内存。
const MIN_ENTRY: usize = 5;

/// 写盘时的缓冲：够把几千条目的头部一次交给内核，又不至于为一个小检查点占太多内存。
const WRITE_BUFFER: usize = 8 * 1024;

/// 条目表的预分配上限：更大的表改为按需增长。
///
/// 声明的条目数只有在字节能撑住时才可信（见 [`MIN_ENTRY`]），但"能撑住"仍允许几十倍的放大
/// ——一个 [`Entry`] 比 5 字节大得多，一个 1 MB 的伪造文件能声明 20 万个条目。预分到这么多
/// 之后改为按需增长，于是伪造的计数最多多要一块固定大小的内存，真表也只有几次摊还扩容。
const PREALLOC: usize = 1024;

/// 检查点文件头：4 字节魔数 + 2 字节小端版本号。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format;

impl Format {
    /// 魔数 `LLMC`。
    pub const MAGIC: [u8; 4] = *b"LLMC";

    /// 当前格式版本：字段集、编码顺序或校验方式变化时必须递增。
    ///
    /// 第 2 版相对第 1 版换了整套编码：LEB128 变长整数、偏移增量、CRC-32 尾部；第 3 版把条目名
    /// 改成"与上一条共享前后缀"、把布局改成"同型同形只写一个标签"。版本之间不兼容，也只支持
    /// 当前版本——读一个"看起来能读"的旧文件比读不出来更危险。
    pub const VERSION: u16 = 3;

    /// 文件头字节数（魔数 + 版本）。
    pub const HEADER_BYTES: usize = Self::MAGIC.len() + 2;

    /// 尾部校验和字节数（小端 `u32`）。
    pub const FOOTER_BYTES: usize = 4;

    /// 校验文件头：魔数一致且版本相同。
    ///
    /// 版本不符时说清是哪一边旧：读端旧要重存，写端旧要升级，这两件事的处理方式完全不同。
    ///
    /// # Errors
    ///
    /// 不足文件头长度、魔数不符或版本不同时返回 [`Code::Invalid`]。
    pub fn check(bytes: &[u8]) -> Result<()> {
        let magic = bytes.get(..Self::MAGIC.len()).ok_or_else(|| truncated(0))?;
        if magic != Self::MAGIC.as_slice() {
            return Err(
                Error::new(Code::Invalid, format!("检查点魔数不是 `LLMC`：{magic:?}"))
                    .with("train::snapshot::Format::check"),
            );
        }
        let raw = bytes
            .get(Self::MAGIC.len()..Self::HEADER_BYTES)
            .ok_or_else(|| truncated(Self::MAGIC.len()))?;
        let version = u16::from_le_bytes([raw[0], raw[1]]);
        if version == Self::VERSION {
            return Ok(());
        }
        let message = if version > Self::VERSION {
            format!(
                "检查点版本 {version} 比当前版本 {} 新：请用写它的那个版本读取",
                Self::VERSION
            )
        } else {
            format!(
                "检查点版本 {version} 比当前版本 {} 旧：本 crate 只读第 {} 版，请重存一次",
                Self::VERSION,
                Self::VERSION
            )
        };
        Err(Error::new(Code::Invalid, message).with("train::snapshot::Format::check"))
    }
}

/// 一次训练恢复后必须逐项相同的事实：步号、并行度、层数与批行数。
///
/// 放在一个类型里不是样式问题：四个数全是整数，位置化参数一旦传反，得到的仍是一个"合法"的
/// 检查点。作为整体比较（`restored.facts() == saved.facts()`）才是恢复前的判据，也是
/// [`Snapshot`] 里唯一必须逐项对上的那部分。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Facts {
    step: u64,
    world: usize,
    layers: usize,
    batch_rows: usize,
}

impl Facts {
    /// 构造：并行度、层数与批行数必须非零。
    ///
    /// 步号可以为 0——"刚初始化就存盘"是合法状态，零并不表示信息缺失。
    ///
    /// # Errors
    ///
    /// 三个非零字段中任意一个为 0 时返回 [`Code::Invalid`]。
    pub fn new(step: u64, world: usize, layers: usize, batch_rows: usize) -> Result<Self> {
        for (name, value) in [("并行度", world), ("层数", layers), ("批行数", batch_rows)] {
            if value == 0 {
                return Err(Error::new(Code::Invalid, format!("检查点的{name}不能为 0"))
                    .with("train::snapshot::Facts::new"));
            }
        }
        Ok(Self {
            step,
            world,
            layers,
            batch_rows,
        })
    }

    /// 步号。
    pub fn step(&self) -> u64 {
        self.step
    }

    /// 保存时的并行度（恢复时必须一致）。
    pub fn world(&self) -> usize {
        self.world
    }

    /// 训练层数。
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// 批的行数。
    pub fn batch_rows(&self) -> usize {
        self.batch_rows
    }
}

/// 一次落盘的快照：恢复事实 + 权重条目表。
///
/// 字节布局（全部小端，见 [`Snapshot::encode`]）：
///
/// ```text
/// LLMC | 版本 u16
/// 步号 | 并行度 | 层数 | 批行数 | 权重数据段起点 | 条目数            （LEB128）
/// 每条目：共享前缀 | 共享后缀 | 中段长度 | 中段
///         | 布局 | [类型 u8 | 秩 | 各维度…] | 偏移增量              （LEB128 + 字节）
/// CRC-32 u32                                                         （覆盖以上全部）
/// ```
///
/// 两处"相对上一条"，都是把已经存在的结构写进格式：
///
/// - **名字**：`共享前缀 + 中段 + 共享后缀`。前两段来自**上一条**名字（第一条的上一条是空名字，
///   因此两个共享长度都是 0），所以一条名字只需要写它跟上一条不一样的那几个字节。
/// - **布局**：`0` 表示"与上一条同类型同形状"（同一层的 `gate` / `up` / `down` 就是这种情形），
///   否则写 `秩 + 1`，后面跟类型与各维度。两者都只有一种写法（共享取最长、同布局必写 0），
///   这才保住了"同一份快照只有一种字节形态"。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    facts: Facts,
    weights: Header,
}

impl Snapshot {
    /// 由事实与权重条目表构造。
    ///
    /// 不会失败：两个入参各自已自证合法（[`Facts::new`] 管零值，[`Header::new`] 管名字与区间），
    /// 这一层没有第二条判据可加——把校验放在拥有不变量的类型里，而不是在每条路径上重复一遍。
    #[must_use]
    pub fn new(facts: Facts, weights: Header) -> Self {
        Self { facts, weights }
    }

    /// 恢复事实（四个数整体比较）。
    pub fn facts(&self) -> Facts {
        self.facts
    }

    /// 步号。
    pub fn step(&self) -> u64 {
        self.facts.step()
    }

    /// 权重条目表：解码时已经重新校验过偏移与重名。
    pub fn weights(&self) -> &Header {
        &self.weights
    }

    /// 恢复前的前置校验：检查点引用的条目表必须与手头的权重表逐条相同。
    ///
    /// 检查点只描述"参数长什么样、在第几个字节"，它不持有参数。于是"文件都对、但配错权重"
    /// 这一种失败在字节层面完全看不出来：条目数与形状可能刚好一致，读出来的参数却是另一次
    /// 训练的。恢复之前必须先过这一关——它比"读得出来"强，也比"跑起来不报错"强。
    ///
    /// 比较数据段起点、条目数，以及每个条目的名字、元素类型、形状与偏移；差异按**第一条**
    /// 不同的记录报出，几千个条目的表里不必自己去二分。
    ///
    /// # Errors
    ///
    /// 任一字段不同时返回 [`Code::Invalid`]，消息里带上条目序号与两边的值。
    pub fn check_weights(&self, weights: &Header) -> Result<()> {
        if self.weights.data_start() != weights.data_start() {
            return Err(mismatch(format!(
                "权重数据段起点：检查点里是 {}，权重文件里是 {}",
                self.weights.data_start(),
                weights.data_start()
            )));
        }
        let saved = self.weights.entries();
        let live = weights.entries();
        if saved.len() != live.len() {
            return Err(mismatch(format!(
                "检查点引用 {} 个权重条目，权重文件有 {} 个",
                saved.len(),
                live.len()
            )));
        }
        for (index, (want, have)) in saved.iter().zip(live).enumerate() {
            if want.name() != have.name() {
                return Err(mismatch(format!(
                    "第 {index} 个条目：检查点里是 `{}`，权重文件里是 `{}`",
                    want.name(),
                    have.name()
                )));
            }
            if want.dtype() != have.dtype() {
                return Err(mismatch(format!(
                    "条目 `{}` 的元素类型：检查点里是 {:?}，权重文件里是 {:?}",
                    want.name(),
                    want.dtype(),
                    have.dtype()
                )));
            }
            if want.shape() != have.shape() {
                return Err(mismatch(format!(
                    "条目 `{}` 的形状：检查点里是 {:?}，权重文件里是 {:?}",
                    want.name(),
                    want.shape().dims(),
                    have.shape().dims()
                )));
            }
            if want.offset() != have.offset() {
                return Err(mismatch(format!(
                    "条目 `{}` 的偏移：检查点里是 {}，权重文件里是 {}",
                    want.name(),
                    want.offset(),
                    have.offset()
                )));
            }
        }
        Ok(())
    }

    /// 编码后的精确字节数。
    ///
    /// 与 [`Snapshot::encode`] 走同一个编码器，只是把字节"写"进一个只做加法的计数器，因此
    /// 不存在第二份长度算术、也不会两边漂移。调用方据此一次分配，或判断磁盘上是否放得下。
    ///
    /// 计数**不算校验和**：长度与 CRC 是两份工作，而校验和固定 [`Format::FOOTER_BYTES`] 字节，
    /// 所以"正文长度 + 校验和长度"就是整份文件。写正文的路径只有 `Snapshot::write_body` 一处，
    /// 新增字段若写到它外面，既逃出 CRC 覆盖、也会让这条长度承诺当场对不上。
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        let mut count = Count(0);
        self.write_body(&mut count);
        count.0 + Format::FOOTER_BYTES
    }

    /// 编码成字节，容量正好等于 [`Snapshot::encoded_len`]。
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let capacity = self.encoded_len();
        let mut bytes = Vec::with_capacity(capacity);
        self.write_to(&mut VecSink(&mut bytes));
        debug_assert_eq!(bytes.len(), capacity, "长度算术与写出的字节必须一致");
        bytes
    }

    /// 写进调用方给的缓冲区，返回写入的字节数。
    ///
    /// `out` 至少要有 [`Snapshot::encoded_len`] 字节；多出来的部分不会被碰。失败时 `out` 的
    /// 内容没有意义——编码器一旦发现放不下就停止追加，不会越界。
    ///
    /// # Errors
    ///
    /// 缓冲区装不下整份检查点时返回 [`Code::Invalid`]（这是调用方的错，不能截断写入）。
    pub fn encode_into(&self, out: &mut [u8]) -> Result<usize> {
        let capacity = out.len();
        let mut sink = SliceSink {
            out,
            at: 0,
            overflow: false,
        };
        self.write_to(&mut sink);
        if sink.overflow {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲区 {capacity} 字节，检查点需要 {} 字节",
                    self.encoded_len()
                ),
            )
            .with("train::snapshot::Snapshot::encode_into"));
        }
        Ok(sink.at)
    }

    /// 原子地写到一个路径：同目录临时文件 → 落盘 → 改名。
    ///
    /// 检查点的写入必须是"全有或全无"：中途崩溃既不能让上一次的好检查点消失，也不能让恢复端
    /// 读到一个半截文件。`rename` 在同一文件系统上原子，所以临时文件放在目标目录里——放 `/tmp`
    /// 会退化成跨文件系统的拷贝，"原子"就没了。
    ///
    /// 字节是**边编码边写**的：编码器直接接到文件上的缓冲，整份检查点从不进内存。条目表大到
    /// 几十万条时，这一条就是"写得下"与"写不下"的区别，而不是快慢的区别。
    ///
    /// # Errors
    ///
    /// 目录不存在、写盘或改名失败时返回 [`Code::Io`]；临时文件会被清掉。
    pub fn write(&self, path: &Path) -> Result<()> {
        let temp = temp_path(path);
        if let Err(error) = self.write_temp(&temp, path) {
            // 半截临时文件比错误本身更麻烦：它会被下一次 `ls` 当成一个检查点。
            let _ = std::fs::remove_file(&temp);
            return Err(
                Error::new(Code::Io, format!("写入 {} 失败：{error}", path.display()))
                    .with("train::snapshot::Snapshot::write"),
            );
        }
        // 目录 fsync 尽力而为：改名已经生效，有的文件系统（容器 overlay、部分网络盘）不支持
        // 目录 fsync；为一个"可能不持久"的目录项把成功改判成失败，比不 fsync 更糟。
        if let Some(parent) = path.parent()
            && let Ok(directory) = std::fs::File::open(parent)
        {
            let _ = directory.sync_all();
        }
        Ok(())
    }

    /// 写临时文件、刷盘、改名为目标路径。
    ///
    /// 编码器一次走完：字节追加给写缓冲，缓冲再交给内核，没有任何一处需要先知道总长度。
    /// 三段式（写 → 刷盘 → 改名）也正是"改名前任何失败都不影响旧检查点"的原因。
    fn write_temp(&self, temp: &Path, path: &Path) -> io::Result<()> {
        let mut sink = FileSink {
            inner: io::BufWriter::with_capacity(WRITE_BUFFER, std::fs::File::create(temp)?),
            written: 0,
            error: None,
        };
        self.write_to(&mut sink);
        // 第一次写失败之后剩下的追加都是空操作，错误在这里报一次就够。
        if let Some(error) = sink.error {
            return Err(error);
        }
        // 长度承诺在改名**之前**对账：调试期一旦对不上，不要留下一个已经换成正式名字的文件。
        debug_assert_eq!(
            sink.written,
            self.encoded_len(),
            "写下的字节数必须等于承诺的长度"
        );
        let file = sink
            .inner
            .into_inner()
            .map_err(io::IntoInnerError::into_error)?;
        file.sync_all()?;
        std::fs::rename(temp, path)?;
        Ok(())
    }

    /// 从一个路径读出并解码。
    ///
    /// 路径不存在是 [`Code::Missing`]、存在但读不出来是 [`Code::Io`]——这个分类由
    /// [`Error::from_io`] 给，本层不重写一遍；解出来的错误补上路径，排查时不必回头查调用栈。
    ///
    /// # Errors
    ///
    /// 读文件失败时返回 `missing` / `io`；字节不合法时返回 [`Snapshot::decode`] 的错误。
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| Error::from_io(path, &error))?;
        Self::decode(&bytes).map_err(|error| error.with(path.display().to_string()))
    }

    /// 只验字节：魔数、版本与 CRC-32，不解析条目表。
    ///
    /// 与 [`Snapshot::decode`] 的关系是"先判能不能读，再判读到什么"：想确认一个文件是不是完好
    /// 的检查点（巡检、备份、下载后自检）不必付出一次完整解码的代价，也不需要一块条目表大小的
    /// 内存。
    ///
    /// # Errors
    ///
    /// 文件头不符、长度不足文件头加校验和、校验和不符时返回 [`Code::Invalid`]。
    pub fn verify(bytes: &[u8]) -> Result<()> {
        Format::check(bytes)?;
        let Some(end) = bytes.len().checked_sub(Format::FOOTER_BYTES) else {
            return Err(truncated(bytes.len()));
        };
        // 连文件头都装不下时，问题不是"校验和对不上"而是"少了一截"；把话说准。
        if end < Format::HEADER_BYTES {
            return Err(truncated(end));
        }
        let (payload, footer) = bytes.split_at(end);
        let recorded = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
        let actual = weights::crc::crc32(payload);
        if recorded != actual {
            return Err(invalid(format!(
                "检查点校验和不符（记录 {recorded:#010x}，实测 {actual:#010x}）：\
                 文件被截断或被改动"
            )));
        }
        Ok(())
    }

    /// 从字节解码。
    ///
    /// 顺序是刻意的：先由 [`Snapshot::verify`] 判文件头与校验和（截断与改动在那里一次拦下），
    /// 再解析正文——此时出错只剩"拿着正确工具重封了一个坏检查点"这一种可能，所以条目表还要
    /// 再过一次 [`weights::Header`] 的自证。
    ///
    /// # Errors
    ///
    /// 文件头不符、校验和不符、正文截断/超宽/非规范、条目表自证失败（重名或区间重叠）或
    /// 恢复事实非法时返回 [`Code::Invalid`]。
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Self::verify(bytes)?;
        let payload = &bytes[..bytes.len() - Format::FOOTER_BYTES];

        let mut cursor = Cursor::new(payload, Format::HEADER_BYTES);
        let step = cursor.varint()?;
        let world = cursor.count("并行度")?;
        let layers = cursor.count("层数")?;
        let batch_rows = cursor.count("批行数")?;
        let data_start = cursor.count("权重数据段起点")?;
        let count = cursor.count("权重条目数")?;
        // 先对账再分配：每个条目至少 `MIN_ENTRY` 字节，所以声明值超过剩余字节能容纳的上限就
        // 一定读不完。否则一个伪造的计数就能让解码器按声明值要一大块内存。
        if count > cursor.remaining() / MIN_ENTRY {
            return Err(invalid(format!(
                "检查点声明了 {count} 个权重条目，但剩余 {} 字节最多容纳 {} 个",
                cursor.remaining(),
                cursor.remaining() / MIN_ENTRY
            )));
        }
        let mut entries = Vec::with_capacity(count.min(PREALLOC));
        // 名字与布局都"相对上一条"，而上一条就是 `entries` 的最后一条：不必另存一份副本，也不必
        // 为它克隆任何东西——名字只借用它的字节，布局只在需要时拷一份维度。
        let mut rebuilt: Vec<u8> = Vec::new();
        let mut previous = 0usize;
        for index in 0..count {
            let previous_name = entries
                .last()
                .map_or(&[][..], |entry: &Entry| entry.name().as_bytes());
            let front = cursor.count("名字的共享前缀").map_err(context(index))?;
            let back = cursor.count("名字的共享后缀").map_err(context(index))?;
            let shared = previous_name.len();
            if front > shared || back > shared - front {
                return Err(context(index)(invalid(format!(
                    "上一条目名只有 {shared} 字节，却要共享前 {front} 后 {back} 字节"
                ))));
            }
            let len = cursor.count("条目名中段长度").map_err(context(index))?;
            let middle = cursor.take(len).map_err(context(index))?;
            rebuilt.clear();
            rebuilt.extend_from_slice(&previous_name[..front]);
            rebuilt.extend_from_slice(middle);
            rebuilt.extend_from_slice(&previous_name[shared - back..]);
            // 共享长度必须是最长写法：前缀取最长、后缀取剩下那一段里最长。少了这一步，一份
            // 快照就能有多种字节形态（"全写"与"共享"都合法），`encode(decode(b)) == b` 随之
            // 不成立，校验和也不再等于"这份字节就是这个检查点"。
            let want_front = common_prefix(previous_name, &rebuilt);
            let want_back = common_suffix(&previous_name[want_front..], &rebuilt[want_front..]);
            if (front, back) != (want_front, want_back) {
                return Err(context(index)(invalid(format!(
                    "条目名的共享不是最长写法（写了前 {front} 后 {back}，\
                     应为前 {want_front} 后 {want_back}）"
                ))));
            }
            let text = std::str::from_utf8(&rebuilt)
                .map_err(|error| context(index)(invalid(format!("条目名不是 UTF-8：{error}"))))?;
            // 布局标签：`0` = 与上一条同型同形，否则是 `秩 + 1`。
            let layout = cursor.count("布局标签").map_err(context(index))?;
            let (kind, dims) = if layout == 0 {
                let Some(previous) = entries.last() else {
                    return Err(context(index)(invalid(
                        "第一个条目不能与上一条共享布局（还没有上一条）",
                    )));
                };
                (previous.dtype(), previous.shape().dims().to_vec())
            } else {
                let kind = dtype(cursor.u8().map_err(context(index))?)?;
                let rank = layout - 1;
                // 放大攻击的另一个入口：几百字节的伪造正文就能声明一个上百万维的形状。预分到
                // [`PREALLOC`] 个维度为止、其余按需增长，于是真实形状（秩远小于它）一次分够，
                // 伪造的秩也只能骗到一块固定大小的内存。
                let mut dims = Vec::with_capacity(rank.min(PREALLOC));
                for _ in 0..rank {
                    dims.push(cursor.count("维度").map_err(context(index))?);
                }
                // 与上一条同型同形就必须写成 `0`：否则同一份快照既有"重复"写法又有"重写"
                // 写法，字节不再唯一，校验和也就不再等于"这份字节就是这个检查点"。
                if let Some(previous) = entries.last()
                    && previous.dtype() == kind
                    && previous.shape().dims() == dims.as_slice()
                {
                    return Err(context(index)(invalid(
                        "布局与上一条相同却没有写成重复（标签 0）",
                    )));
                }
                (kind, dims)
            };
            let delta = cursor.count("偏移增量").map_err(context(index))?;
            let offset = previous.checked_add(delta).ok_or_else(|| {
                context(index)(invalid(format!("偏移 {previous} + {delta} 溢出")))
            })?;
            previous = offset;
            entries.push(Entry::new(text, kind, Shape::new(dims)).at(offset));
        }
        if cursor.remaining() > 0 {
            return Err(invalid(format!(
                "检查点尾部多出 {} 字节",
                cursor.remaining()
            )));
        }
        let weights = Header::new(entries, data_start).map_err(|error| error.with(CONTEXT))?;
        let facts =
            Facts::new(step, world, layers, batch_rows).map_err(|error| error.with(CONTEXT))?;
        Ok(Self::new(facts, weights))
    }

    /// 编码正文与校验和：正文写进一个"密封"的接收器，于是**写出的每个字节都自动进入 CRC**，
    /// 新增字段不可能忘了更新校验范围。
    fn write_to(&self, sink: &mut impl Sink) {
        let crc = {
            let mut sealed = Sealed {
                inner: sink,
                crc: !0,
            };
            self.write_body(&mut sealed);
            !sealed.crc
        };
        sink.push(&crc.to_le_bytes());
    }

    /// 写文件头与正文（不含校验和）。
    ///
    /// 两处"只写差值"：条目名相对上一条名共享前后缀，偏移相对上一条偏移存增量。两者都不是
    /// 压缩技巧，而是**把已经存在的结构写进格式**——`Header` 保证条目按偏移有序，真实权重表的
    /// 命名则天然按层、按部件连续，相邻名字的公共部分本来就是冗余的。
    fn write_body(&self, sink: &mut impl Sink) {
        let facts = self.facts;
        sink.push(Format::MAGIC.as_slice());
        sink.push(&Format::VERSION.to_le_bytes());
        varint(sink, facts.step());
        varint(sink, facts.world() as u64);
        varint(sink, facts.layers() as u64);
        varint(sink, facts.batch_rows() as u64);
        varint(sink, self.weights.data_start() as u64);

        let entries = self.weights.entries();
        varint(sink, entries.len() as u64);
        let mut previous_name: &[u8] = &[];
        let mut previous_offset = 0usize;
        let mut previous_layout: Option<(DType, &[usize])> = None;
        for entry in entries {
            let name = entry.name().as_bytes();
            let front = common_prefix(previous_name, name);
            // 后缀只从"两条名字都还没被前缀吃掉"的那一段里取，于是前后两段永远不会重叠。
            let back = common_suffix(&previous_name[front..], &name[front..]);
            varint(sink, front as u64);
            varint(sink, back as u64);
            varint(sink, (name.len() - front - back) as u64);
            sink.push(&name[front..name.len() - back]);
            // 布局（元素类型 + 形状）也相对上一条：同一层里 `gate` / `up` / `down` 往往同形同型，
            // "与上一条相同"只占一个字节，不同的时候才写 `秩 + 1`、类型与各维度。
            let dims = entry.shape().dims();
            if previous_layout == Some((entry.dtype(), dims)) {
                varint(sink, 0);
            } else {
                varint(sink, (dims.len() + 1) as u64);
                sink.push(&[tag(entry.dtype()).expect("快照条目类型在标签闭集内")]);
                for dim in dims {
                    varint(sink, *dim as u64);
                }
                previous_layout = Some((entry.dtype(), dims));
            }
            // 增量而不是绝对值：`Header` 按偏移排序，相邻差值就是"中间的空隙 + 上一条的长度"，
            // 比绝对偏移少几个字节；条目有序这一条也因此写进了格式本身。
            debug_assert!(entry.offset() >= previous_offset, "`Header` 必须按偏移排序");
            varint(sink, entry.offset().saturating_sub(previous_offset) as u64);
            previous_offset = entry.offset();
            previous_name = name;
        }
    }
}

/// 编码器的唯一接口：追加一段字节。
///
/// 三个消费者共用同一份编码逻辑——统计长度（[`Count`]）、写进缓冲区（[`SliceSink`]）、
/// 或分配一块 `Vec`（[`VecSink`]）。编码本身没有可失败的操作（所有整数都是变长、所有名字
/// 都按字节计），所以这里没有 `Result`：缓冲区放不下是接收器的事，由 [`Snapshot::encode_into`]
/// 在最后统一判一次。
trait Sink {
    /// 追加一段字节；接收器放不下时置位而不是返回错误。
    fn push(&mut self, bytes: &[u8]);
}

/// 只做加法：`encoded_len` 用它与 `encode` 共用同一份算术。
struct Count(usize);

impl Sink for Count {
    fn push(&mut self, bytes: &[u8]) {
        self.0 += bytes.len();
    }
}

/// 写进一块 `Vec`。
struct VecSink<'a>(&'a mut Vec<u8>);

impl Sink for VecSink<'_> {
    fn push(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }
}

/// 写进调用方的固定缓冲区，放不下就置 `overflow`。
struct SliceSink<'a> {
    out: &'a mut [u8],
    at: usize,
    overflow: bool,
}

impl Sink for SliceSink<'_> {
    fn push(&mut self, bytes: &[u8]) {
        let Some(end) = self.at.checked_add(bytes.len()) else {
            self.overflow = true;
            return;
        };
        let Some(slot) = self.out.get_mut(self.at..end) else {
            self.overflow = true;
            return;
        };
        slot.copy_from_slice(bytes);
        self.at = end;
    }
}

/// 直接写进文件缓冲：整份检查点从不进内存。
///
/// 错误只记一次：`Sink::push` 不返回 `Result`（编码本身没有可失败的操作），所以第一次写失败
/// 之后剩下的追加全部变成空操作，由 [`Snapshot::write`] 在最后统一报出来——一个坏掉的块设备
/// 不该产生一千条一模一样的错误。
struct FileSink<W> {
    inner: W,
    written: usize,
    error: Option<io::Error>,
}

impl<W: io::Write> Sink for FileSink<W> {
    fn push(&mut self, bytes: &[u8]) {
        if self.error.is_some() {
            return;
        }
        match self.inner.write_all(bytes) {
            Ok(()) => self.written += bytes.len(),
            Err(error) => self.error = Some(error),
        }
    }
}

/// 一边写一边更新 CRC：正文的每个字节都会经过这里，校验范围不可能漏掉字段。
struct Sealed<'s, S> {
    inner: &'s mut S,
    crc: u32,
}

impl<S: Sink> Sink for Sealed<'_, S> {
    fn push(&mut self, bytes: &[u8]) {
        self.crc = weights::crc::update(self.crc, bytes);
        self.inner.push(bytes);
    }
}

/// 写一个 LEB128 变长整数（每字节 7 位，小端，最高位是续位）。
#[inline]
fn varint(sink: &mut impl Sink, mut value: u64) {
    // 一个条目里有五六个整数，而绝大多数（秩、维度、名字长度）都小于 128：单字节这条路
    // 占九成，就不必为一个字节走一遭循环、也不必占用那个 10 字节的栈缓冲。
    if value < 0x80 {
        sink.push(&[value as u8]);
        return;
    }
    let mut buffer = [0u8; 10];
    let mut len = 0usize;
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buffer[len] = byte;
            len += 1;
            break;
        }
        buffer[len] = byte | 0x80;
        len += 1;
    }
    sink.push(&buffer[..len]);
}

/// 逐字节读取器：每次读取都检查边界，越界时报出当前的偏移。
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], offset: usize) -> Self {
        Self { bytes, offset }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| truncated(self.offset))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| truncated(self.offset))?;
        self.offset = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// 读一个 LEB128 变长整数。
    ///
    /// 两条拒绝：超过 `u64` 宽度，以及**非规范写法**（例如 `0x87 0x00` 也等于 7，但只允许
    /// `0x07`）。后者让编码成为单射——同一份快照不会有两种字节形态，校验和也就真正对得上
    /// "这份字节就是这个检查点"。
    fn varint(&mut self) -> Result<u64> {
        // 直接在切片上走：`take(1)` 每次都把"再切一刀 + 判空"做一遭，而解码一条条目要读
        // 五六个整数。这里的边界检查只做在真正取字节的那一处，报错时的偏移仍然是精确的。
        let bytes = self.bytes;
        let start = self.offset;
        let mut at = start;
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let Some(&byte) = bytes.get(at) else {
                self.offset = at;
                return Err(truncated(at));
            };
            at += 1;
            let low = u64::from(byte & 0x7F);
            if shift == 63 && low > 1 {
                self.offset = at;
                return Err(invalid("检查点里的整数超过 u64 上限"));
            }
            value |= low << shift;
            if byte & 0x80 == 0 {
                self.offset = at;
                let width = at - start;
                if width > 1 && low == 0 {
                    return Err(invalid(format!(
                        "检查点里的整数用了 {width} 字节的非规范写法（末字节为 0）"
                    )));
                }
                return Ok(value);
            }
            shift += 7;
            if shift >= 64 {
                self.offset = at;
                return Err(invalid("检查点里的整数超过 10 字节"));
            }
        }
    }

    /// 读一个不能超过 `usize` 的计数；`what` 是字段名，用来把错误指到具体位置。
    fn count(&mut self, what: &str) -> Result<usize> {
        let value = self.varint()?;
        usize::try_from(value)
            .map_err(|_| invalid(format!("检查点的{what} {value} 超出 usize 上限")))
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

/// 条目错误的上下文：一个几千条目的检查点里，"第 3 个条目"比"某个偏移"有用得多。
fn context(index: usize) -> impl Fn(Error) -> Error {
    move |error| error.with(format!("第 {index} 个权重条目"))
}

/// 两条名字共有的最大字节前缀。
///
/// 按**字节**而不是按字符比：名字的共享长度最终要能直接切出字节区间，而 UTF-8 的一个字符
/// 占几个字节并不重要——两条不同的合法名字的公共前缀确实可能切在一个字符中间，那一段中间
/// 字节与拼回来的完整名字一起才是 UTF-8，所以判 UTF-8 要放到拼接之后。
fn common_prefix(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut at = 0;
    while at < limit && left[at] == right[at] {
        at += 1;
    }
    at
}

/// 两条名字共有的最大字节后缀。
///
/// 调用方传进来的总是**已经被前缀吃掉**的那两段尾巴：这样前后两段永远不会重叠，也就不需要
/// 一条额外的"前三后四不能超过总长"的不等式。
fn common_suffix(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut at = 0;
    while at < limit && left[left.len() - 1 - at] == right[right.len() - 1 - at] {
        at += 1;
    }
    at
}

/// 元素类型在快照字节里的标签：快照格式只支持计算层 5 种，闭集外的类型显式报错。
fn tag(dtype: DType) -> Result<u8> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::I8 => Ok(3),
        DType::U4 => Ok(4),
        other => Err(invalid(format!("快照格式不支持 {other:?} 元素类型"))),
    }
}

fn dtype(tag: u8) -> Result<DType> {
    match tag {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        3 => Ok(DType::I8),
        4 => Ok(DType::U4),
        other => Err(invalid(format!("未知的元素类型标签 {other}"))),
    }
}

/// 临时文件名：同目录 + 进程号 + 进程内序号。
///
/// 两个写者同时存同一个路径时，各自的临时文件不能互相覆盖——否则先改名的一方会把后来者的
/// 半截文件搬走。
fn temp_path(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    path.with_file_name(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn truncated(offset: usize) -> Error {
    Error::new(Code::Invalid, format!("检查点在偏移 {offset} 处截断")).with(CONTEXT)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with(CONTEXT)
}

/// 检查点与手头的权重表不一致：只由 [`Snapshot::check_weights`] 产生。
fn mismatch(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message).with("train::snapshot::Snapshot::check_weights")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> Facts {
        Facts::new(7, 2, 2, 3).expect("恢复事实")
    }

    fn snapshot() -> Snapshot {
        let weights = Header::new(
            vec![
                Entry::new("blk.0.attn_q.weight", DType::F16, Shape::new([2, 3])).at(64),
                Entry::new("blk.0.scales", DType::U4, Shape::new([4])).at(88),
            ],
            64,
        )
        .expect("头部");
        Snapshot::new(facts(), weights)
    }

    /// 用任意正文重新封上校验和：只有测试会造"CRC 正确但正文有问题"的文件。
    fn seal(payload: &[u8]) -> Vec<u8> {
        let mut bytes = payload.to_vec();
        bytes.extend_from_slice(&weights::crc::crc32(payload).to_le_bytes());
        bytes
    }

    fn push_varint(bytes: &mut Vec<u8>, value: u64) {
        varint(&mut VecSink(bytes), value);
    }

    fn payload_of(snapshot: &Snapshot) -> Vec<u8> {
        let bytes = snapshot.encode();
        bytes[..bytes.len() - Format::FOOTER_BYTES].to_vec()
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    /// 一整套承诺：长度是算准的、写进缓冲区的字节与 `encode` 一致、缓冲区多出来的部分不被碰、
    /// 解码回来逐项相同、再编码逐字节相同。
    fn assert_round_trip(snapshot: &Snapshot) {
        let bytes = snapshot.encode();
        assert_eq!(
            bytes.len(),
            snapshot.encoded_len(),
            "承诺的长度就是写出的长度"
        );

        let mut buffer = vec![0xAAu8; snapshot.encoded_len() + 16];
        let written = snapshot.encode_into(&mut buffer).expect("缓冲够大");
        assert_eq!(written, snapshot.encoded_len());
        assert_eq!(&buffer[..written], bytes.as_slice());
        assert!(
            buffer[written..].iter().all(|byte| *byte == 0xAA),
            "缓冲区多出来的部分不能被碰"
        );

        let restored = Snapshot::decode(&bytes).expect("解码");
        assert_eq!(&restored, snapshot);
        assert_eq!(restored.encode(), bytes, "同一份快照只有一种写法");
    }

    #[test]
    fn a_snapshot_round_trips_through_its_bytes() {
        let snapshot = snapshot();
        assert_round_trip(&snapshot);
        let restored = Snapshot::decode(&snapshot.encode()).expect("解码");
        assert_eq!(restored.facts(), facts());
        assert_eq!(restored.step(), 7);
        assert_eq!(restored.weights().entries().len(), 2);
        assert_eq!(restored.weights().entries()[1].name(), "blk.0.scales");
    }

    /// 变长整数的每个宽度边界都要对上：错一档就是半个检查点。
    #[test]
    fn the_length_promise_holds_at_every_varint_boundary() {
        let steps = [0u64, 1, 127, 128, 16_383, 16_384, 4_294_967_295, u64::MAX];
        for (index, step) in steps.into_iter().enumerate() {
            let offset = 64 + index * 65_536;
            let header = Header::new(
                vec![Entry::new("w", DType::F32, Shape::new(vec![1, 127, 128, 16_384])).at(offset)],
                64,
            )
            .expect("头部");
            let size = index + 1;
            let facts = Facts::new(step, size, size, size).expect("事实");
            assert_round_trip(&Snapshot::new(facts, header));
        }
    }

    #[test]
    fn a_snapshot_without_entries_still_round_trips() {
        let header = Header::new(Vec::new(), 0).expect("空头部");
        assert_round_trip(&Snapshot::new(facts(), header));
    }

    /// 名字长度按**字节**算：按字符算就会在解码时切进 UTF-8 中间。
    #[test]
    fn a_unicode_name_is_measured_in_bytes() {
        let header = Header::new(
            vec![Entry::new("层.0.注意_权重", DType::BF16, Shape::new([3])).at(64)],
            64,
        )
        .expect("头部");
        let snapshot = Snapshot::new(facts(), header);
        assert_round_trip(&snapshot);
        let name = "层.0.注意_权重".as_bytes();
        assert!(
            snapshot
                .encode()
                .windows(name.len())
                .any(|window| window == name),
            "名字原样进字节"
        );
    }

    /// 恢复前的第二道关：文件都是好的，但配的是另一次训练的权重。
    ///
    /// 这一种失败在字节层面完全看不出来——它不会在读取时炸，而是在训练若干步之后才表现成
    /// loss 不降。所以每一条差异都要被逐个拦住，而不是只比一个总数。
    #[test]
    fn check_weights_reports_the_first_entry_that_differs() {
        let snapshot = snapshot();
        let same = snapshot.weights().clone();
        snapshot.check_weights(&same).expect("同一份表");

        let two = || {
            vec![
                Entry::new("blk.0.attn_q.weight", DType::F16, Shape::new([2, 3])).at(64),
                Entry::new("blk.0.scales", DType::U4, Shape::new([4])).at(88),
            ]
        };
        let edited = |name: &str, dtype: DType, shape: [usize; 2], offset: usize| {
            Header::new(
                vec![
                    Entry::new(name, dtype, Shape::new(shape)).at(offset),
                    two()[1].clone(),
                ],
                64,
            )
            .expect("头部")
        };
        let cases: [(Header, [&str; 2]); 6] = [
            (Header::new(two(), 60).expect("头部"), ["数据段起点", "60"]),
            (
                Header::new(vec![two()[0].clone()], 64).expect("头部"),
                ["引用 2 个", "有 1 个"],
            ),
            (
                edited("blk.0.attn_k.weight", DType::F16, [2, 3], 64),
                ["attn_k", "attn_q"],
            ),
            (
                edited("blk.0.attn_q.weight", DType::BF16, [2, 3], 64),
                ["元素类型", "BF16"],
            ),
            (
                edited("blk.0.attn_q.weight", DType::F16, [3, 3], 64),
                ["形状", "[3, 3]"],
            ),
            (
                edited("blk.0.attn_q.weight", DType::F16, [2, 3], 72),
                ["偏移", "72"],
            ),
        ];
        for (weights, wants) in cases {
            let error = snapshot.check_weights(&weights).expect_err(wants[0]);
            assert_eq!(error.code(), Code::Invalid);
            for want in wants {
                assert!(error.to_string().contains(want), "{want}: {error}");
            }
        }
    }

    /// 检查点最贵的失败不是"读不出来"，而是"读出来了但不是当初写的那份"。校验和把每一种
    /// 一个比特的改动都变成解码失败，这条测试把整个文件逐位翻一遍。
    #[test]
    fn every_flipped_bit_is_rejected() {
        let bytes = snapshot().encode();
        for offset in 0..bytes.len() {
            for bit in 0..8 {
                let mut broken = bytes.clone();
                broken[offset] ^= 1 << bit;
                assert!(
                    Snapshot::decode(&broken).is_err(),
                    "第 {offset} 字节第 {bit} 位翻转后被接受了"
                );
            }
        }
    }

    /// 校验和本身也要能被独立复算：表是编译期生成的，另写一份逐位定义来对。
    #[test]
    fn the_checksum_matches_its_bitwise_definition() {
        assert_eq!(
            weights::crc::crc32(b"123456789"),
            0xCBF4_3926,
            "IEEE 802.3 的标准检验值"
        );

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let bytes: Vec<u8> = (0..2048).map(|_| (lcg(&mut state) >> 33) as u8).collect();
        let mut crc = !0u32;
        for byte in &bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        assert_eq!(weights::crc::crc32(&bytes), !crc);
    }

    #[test]
    fn a_foreign_magic_is_rejected() {
        let mut bytes = snapshot().encode();
        bytes[0] = b'X';
        let error = Snapshot::decode(&bytes).expect_err("魔数不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("LLMC"), "{error}");
    }

    /// 版本错误要说清是哪一边旧：读端旧要重存，写端旧要升级。
    #[test]
    fn another_version_says_which_side_is_old() {
        let mut newer = snapshot().encode();
        newer[4..6].copy_from_slice(&(Format::VERSION + 1).to_le_bytes());
        let error = Snapshot::decode(&newer).expect_err("更新");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("新"), "{error}");

        let mut older = snapshot().encode();
        older[4..6].copy_from_slice(&1u16.to_le_bytes());
        let error = Snapshot::decode(&older).expect_err("更旧");
        assert!(error.to_string().contains("旧"), "{error}");
    }

    #[test]
    fn a_truncated_file_is_a_checksum_mismatch_not_a_half_snapshot() {
        let bytes = snapshot().encode();
        let error = Snapshot::decode(&bytes[..bytes.len() - 3]).expect_err("截断");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("校验和"), "{error}");

        let error = Snapshot::decode(&[]).expect_err("连文件头都没有");
        assert!(error.to_string().contains("截断"), "{error}");
    }

    /// 巡检、备份、下载后自检只想知道"这份字节能自证吗"：不必建出一整张条目表。
    #[test]
    fn verify_checks_the_bytes_without_building_the_snapshot() {
        let bytes = snapshot().encode();
        Snapshot::verify(&bytes).expect("完好");

        let mut changed = bytes.clone();
        let last = changed.len() - 1;
        changed[last] ^= 0x80;
        let error = Snapshot::verify(&changed).expect_err("校验和被改");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("校验和"), "{error}");

        // 比文件头还短：这不是"校验和对不上"，而是少了一截。
        let error = Snapshot::verify(&bytes[..Format::HEADER_BYTES]).expect_err("只有一个头");
        assert!(error.to_string().contains("截断"), "{error}");

        let mut foreign = bytes.clone();
        foreign[0] = b'X';
        assert!(Snapshot::verify(&foreign).is_err(), "魔数不符也要拦下");
    }

    /// 校验和正确但正文不够：此时只有游标能说清缺在哪个偏移。
    #[test]
    fn a_sealed_but_short_payload_names_the_offset() {
        let bytes = snapshot().encode();
        let payload = &bytes[..Format::HEADER_BYTES + 1];
        let error = Snapshot::decode(&seal(payload)).expect_err("正文不够");
        let offset = Format::HEADER_BYTES + 1;
        assert!(
            error.to_string().contains(&format!("偏移 {offset}")),
            "{error}"
        );
    }

    /// 手工拼一份正文：用来造编码器**永远不会**产生的字节（两条目同名、共享长度写得比实际
    /// 少、布局明明相同却不写重复）。解码器的判据必须比编码器更强，所以测试要能绕过编码器
    /// 直接说话。
    ///
    /// 条目是 `(共享前缀, 共享后缀, 中段, 偏移, 布局标签)`；标签为 `k > 0` 时后面跟类型 `F32`
    /// 与 `k - 1` 个值为 1 的维度。
    fn handcrafted(entries: &[(u64, u64, &str, usize, u64)]) -> Vec<u8> {
        let mut payload = Format::MAGIC.to_vec();
        payload.extend_from_slice(&Format::VERSION.to_le_bytes());
        for value in [7, 1, 1, 1, 64] {
            push_varint(&mut payload, value);
        }
        push_varint(&mut payload, entries.len() as u64);
        let mut previous = 0usize;
        for (front, back, middle, offset, layout) in entries {
            push_varint(&mut payload, *front);
            push_varint(&mut payload, *back);
            push_varint(&mut payload, middle.len() as u64);
            payload.extend_from_slice(middle.as_bytes());
            push_varint(&mut payload, *layout);
            if *layout > 0 {
                payload.push(tag(DType::F32).expect("f32 在快照标签闭集内"));
                for _ in 0..*layout - 1 {
                    push_varint(&mut payload, 1);
                }
            }
            push_varint(&mut payload, (*offset - previous) as u64);
            previous = *offset;
        }
        seal(&payload)
    }

    /// 校验和挡得住"被改动"，挡不住"拿着正确工具重封了一个坏检查点"：条目表的判据必须在
    /// 解码期也跑一遍，否则手改过的检查点会等到恢复后的第 N 步才炸。
    #[test]
    fn a_sealed_checkpoint_still_has_its_entry_table_validated() {
        // 名字都写全（共享长度 0）、布局按规则写、校验和也对，不合法的只剩"名字重复"。
        let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (3, 0, "", 80, 0)]);
        let error = Snapshot::decode(&bytes).expect_err("重名");
        assert!(error.to_string().contains("重复"), "{error}");
    }

    /// 共享是按**字节**算的：两条名字的公共前缀可以正好切在一个 UTF-8 字符中间。`块.0.注意_é`
    /// 与 `块.0.注意_è` 只差最后一个字节，共享长度落在那个两字节字符的正中间——拼回来的整串
    /// 仍然是合法 UTF-8，所以判 UTF-8 必须放在**拼接之后**，不能对每一段单独判。
    #[test]
    fn a_shared_prefix_may_cut_through_a_character() {
        let header = Header::new(
            vec![
                Entry::new("块.0.注意_é", DType::F32, Shape::new([1])).at(64),
                Entry::new("块.0.注意_è", DType::F32, Shape::new([1])).at(80),
            ],
            64,
        )
        .expect("头部");
        let snapshot = Snapshot::new(facts(), header);
        assert_round_trip(&snapshot);

        // 判据落在**名字块**上而不是整份文件：整份文件还含事实、布局、偏移与 CRC，
        // 拿文件总长去比名字字节总和永远不成立。
        let bytes = snapshot.encode();
        let payload = &bytes[..bytes.len() - Format::FOOTER_BYTES];
        let mut cursor = Cursor::new(payload, 0);
        cursor.take(4 + 2).expect("魔数与版本");
        for _ in 0..5 {
            cursor.varint().expect("头部事实");
        }
        assert_eq!(cursor.count("条目数").expect("条目数"), 2);

        // 第一条：名字全写，随后跳过布局与偏移。
        let (f1, b1, m1) = (
            cursor.count("前缀").expect("前缀"),
            cursor.count("后缀").expect("后缀"),
            cursor.count("中段").expect("中段长"),
        );
        let previous = cursor.take(m1).expect("第一条名字").to_vec();
        assert_eq!((f1, b1, m1), (0, 0, previous.len()));
        let layout = cursor.count("布局").expect("布局");
        cursor.u8().expect("类型标签");
        for _ in 0..layout - 1 {
            cursor.varint().expect("形状维度");
        }
        cursor.varint().expect("偏移");

        // 第二条：记录名字块实际占用的字节区间，再按前后缀拼回整串。
        let start = cursor.offset;
        let front = cursor.count("前缀").expect("前缀");
        let back = cursor.count("后缀").expect("后缀");
        let middle_len = cursor.count("中段").expect("中段长");
        let middle = cursor.take(middle_len).expect("第二条名字中段");
        let block_bytes = cursor.offset - start;

        let mut rebuilt = previous[..front].to_vec();
        rebuilt.extend_from_slice(middle);
        rebuilt.extend_from_slice(&previous[previous.len() - back..]);

        assert_eq!(front, 14, "公共前缀按字节算，切在两字节字符的中间字节");
        assert_eq!(back, 0);
        assert_eq!(middle_len, 1, "第二条名字只该写一个中段字节");
        assert_eq!(
            rebuilt,
            "块.0.注意_è".as_bytes(),
            "拼接后才判 UTF-8，整串合法"
        );
        let full_block = 3 + rebuilt.len();
        assert!(
            block_bytes < full_block,
            "共享后名字块 {block_bytes} 字节，全写要 {full_block} 字节"
        );
    }

    /// 共享长度只有一种写法：同一份快照不能既"全写"又"共享"，否则 `encode(decode(b)) == b`
    /// 不成立，校验和也就不再等于"这份字节就是这个检查点"。
    #[test]
    fn a_non_maximal_name_share_is_rejected() {
        // `aaa` → `aab`：公共前缀是 2，写成 1（自留 `ab` 当中段）同样能拼出 `aab`。
        let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (1, 0, "ab", 80, 0)]);
        let error = Snapshot::decode(&bytes).expect_err("前缀少写");
        assert!(error.to_string().contains("最长"), "{error}");

        // 后缀同理：`aaa` → `baa` 的公共后缀是 2，只写 1 就必须多带一个 `a`。
        let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (0, 1, "ba", 80, 0)]);
        let error = Snapshot::decode(&bytes).expect_err("后缀少写");
        assert!(error.to_string().contains("最长"), "{error}");

        // 写满就过，拼回来的名字与直写完全一样。
        let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (2, 0, "c", 80, 0)]);
        assert_eq!(
            Snapshot::decode(&bytes)
                .expect("最长写法")
                .weights()
                .entries()[1]
                .name(),
            "aac"
        );
        // 空中段也合法：`aaa` 后面还是 `aaa` 时前后缀就已拼满（重名由 `Header` 拦下）。
        let bytes = handcrafted(&[(0, 0, "aaa", 64, 2), (3, 0, "", 80, 0)]);
        let error = Snapshot::decode(&bytes).expect_err("重名");
        assert!(error.to_string().contains("重复"), "{error}");
    }

    /// 布局同样只有一种写法：与上一条同类型同形状就必须写标签 `0`，否则同一份快照可以既有
    /// "重复"写法又有"重写"写法。第一个条目没有上一条，标签 `0` 无从引用。
    #[test]
    fn a_repeated_layout_must_be_written_as_a_repeat() {
        let bytes = handcrafted(&[(0, 0, "a", 64, 2), (0, 0, "b", 80, 2)]);
        let error = Snapshot::decode(&bytes).expect_err("同布局重写");
        assert!(error.to_string().contains("布局"), "{error}");

        let bytes = handcrafted(&[(0, 0, "a", 64, 0), (0, 0, "b", 80, 0)]);
        let error = Snapshot::decode(&bytes).expect_err("第一个条目");
        assert!(error.to_string().contains("第一个条目"), "{error}");

        // 标签 2 = 秩 1 的形状 `[1]`：两条都这么写就合法，条目各自拿到同一个形状。
        let bytes = handcrafted(&[(0, 0, "a", 64, 2), (0, 0, "b", 80, 0)]);
        let restored = Snapshot::decode(&bytes).expect("重复布局");
        assert_eq!(restored.weights().entries()[1].shape().dims(), [1]);
        assert_eq!(restored.weights().entries()[1].dtype(), DType::F32);
    }

    #[test]
    fn trailing_bytes_after_the_entry_table_are_rejected() {
        let mut payload = payload_of(&snapshot());
        payload.push(0);
        let error = Snapshot::decode(&seal(&payload)).expect_err("尾部多余");
        assert!(error.to_string().contains("尾部多出 1 字节"), "{error}");
    }

    /// 一个值只能有一种写法，否则"同一份快照只有一种字节形态"就不成立。
    #[test]
    fn a_non_canonical_integer_is_rejected() {
        let mut payload = payload_of(&snapshot());
        assert_eq!(
            payload[Format::HEADER_BYTES],
            7,
            "步号 7 的正常写法是单字节"
        );
        payload[Format::HEADER_BYTES] = 0x87;
        payload.insert(Format::HEADER_BYTES + 1, 0x00);
        let error = Snapshot::decode(&seal(&payload)).expect_err("非规范");
        assert!(error.to_string().contains("规范"), "{error}");
    }

    #[test]
    fn an_integer_wider_than_u64_is_rejected() {
        let mut payload = snapshot().encode()[..Format::HEADER_BYTES].to_vec();
        payload.extend_from_slice(&[0xFF; 9]);
        payload.push(0x02);
        let error = Snapshot::decode(&seal(&payload)).expect_err("超宽");
        assert!(error.to_string().contains("u64"), "{error}");
    }

    /// 先对账再分配：伪造的条目数不能换来一次大分配。
    #[test]
    fn an_entry_count_beyond_the_bytes_is_rejected_before_allocating() {
        let mut payload = snapshot().encode()[..Format::HEADER_BYTES].to_vec();
        for value in [0u64, 1, 1, 1, 0] {
            push_varint(&mut payload, value);
        }
        push_varint(&mut payload, 1_000_000);
        let error = Snapshot::decode(&seal(&payload)).expect_err("计数超限");
        assert!(error.to_string().contains("权重条目"), "{error}");
    }

    #[test]
    fn a_checkpoint_without_layers_is_rejected() {
        for (world, layers, rows) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
            let error = Facts::new(0, world, layers, rows).expect_err("零值");
            assert_eq!(error.code(), Code::Invalid);
        }
    }

    #[test]
    fn a_short_buffer_is_refused_not_silently_filled() {
        let snapshot = snapshot();
        let mut buffer = vec![0u8; snapshot.encoded_len() - 1];
        let error = snapshot.encode_into(&mut buffer).expect_err("放不下");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("需要"), "{error}");
    }

    /// 随机条目表（名字、秩、维度、间隔各不同）压一遍长度与往返承诺。
    #[test]
    fn random_snapshots_keep_the_length_promise_and_the_bytes() {
        let mut state = 0x2545_F491_4F6C_DD1D;
        for round in 0..200u64 {
            let count = (lcg(&mut state) % 8) as usize;
            let mut entries = Vec::new();
            let mut offset = 64usize;
            for index in 0..count {
                let rank = (lcg(&mut state) % 4) as usize;
                let dims: Vec<usize> = (0..rank)
                    .map(|_| (lcg(&mut state) % 4096) as usize)
                    .collect();
                let bytes = dims.iter().product::<usize>() * 4;
                offset += (lcg(&mut state) % 512) as usize;
                entries.push(
                    Entry::new(
                        format!("blk.{round}.w{index}"),
                        DType::F32,
                        Shape::new(dims),
                    )
                    .at(offset),
                );
                offset += bytes.max(1);
            }
            let header = Header::new(entries, 64).expect("头部");
            let size = |state: &mut u64| 1 + (lcg(state) % 64) as usize;
            let facts = Facts::new(
                lcg(&mut state),
                size(&mut state),
                size(&mut state),
                size(&mut state),
            )
            .expect("事实");
            assert_round_trip(&Snapshot::new(facts, header));
        }
    }

    /// 写盘是边编码边写的：超过写缓冲的检查点必须逐字节对得上——这一条走的是 `BufWriter`
    /// 真正的刷写路径，而不是"先编进一块内存再拷贝"。
    ///
    /// 顺带锁住名字的前后缀共享：同一种名字写 4096 遍，字节数必须**远小于**把每个名字整串写下
    /// 去的体积，否则共享在某一轮优化里被悄悄关掉，只有文件变大而测试仍是绿的。
    #[test]
    fn a_snapshot_larger_than_the_write_buffer_lands_byte_for_byte() {
        let mut entries = Vec::new();
        let mut offset = 4096usize;
        let mut plain = 0usize;
        for layer in 0..4096 {
            let name = format!("blk.{layer}.attn_qkv.weight");
            plain += name.len();
            let entry = Entry::new(name, DType::F16, Shape::new([64, 64])).at(offset);
            offset += entry.len();
            entries.push(entry);
        }
        let header = Header::new(entries, 4096).expect("头部");
        let snapshot = Snapshot::new(facts(), header);
        assert!(
            snapshot.encoded_len() > WRITE_BUFFER,
            "这一条必须真的跨过写缓冲"
        );
        assert!(
            snapshot.encoded_len() < plain,
            "名字共享必须省下字节：{} 字节 vs 名字整串写下的 {plain} 字节",
            snapshot.encoded_len()
        );

        let dir = scratch("stream");
        let path = dir.join("step-7.llmc");
        snapshot.write(&path).expect("写");
        let meta = std::fs::metadata(&path).expect("文件存在");
        assert_eq!(meta.len() as usize, snapshot.encoded_len());
        assert_eq!(std::fs::read(&path).expect("字节"), snapshot.encode());
        assert_eq!(Snapshot::read(&path).expect("读"), snapshot);

        let names: Vec<String> = std::fs::read_dir(&dir)
            .expect("目录")
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["step-7.llmc"], "跨缓冲刷写也不留临时文件");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("llm-checkpoint-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("临时目录");
        dir
    }

    #[test]
    fn write_then_read_restores_the_snapshot_and_leaves_no_residue() {
        let dir = scratch("write");
        let path = dir.join("step-7.llmc");
        let snapshot = snapshot();
        snapshot.write(&path).expect("写");

        let meta = std::fs::metadata(&path).expect("文件存在");
        assert_eq!(
            meta.len() as usize,
            snapshot.encoded_len(),
            "写下的字节数就是承诺的字节数"
        );
        assert_eq!(Snapshot::read(&path).expect("读"), snapshot);

        let names: Vec<String> = std::fs::read_dir(&dir)
            .expect("目录")
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["step-7.llmc"], "成功后不留临时文件");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_checkpoint_is_missing_and_a_failed_write_is_io() {
        let dir = scratch("io");
        let error = Snapshot::read(&dir.join("absent.llmc")).expect_err("不存在");
        assert_eq!(error.code(), Code::Missing);

        let error = snapshot()
            .write(&dir.join("no-such-dir/step.llmc"))
            .expect_err("目录不存在");
        assert_eq!(error.code(), Code::Io);
        assert!(error.to_string().contains("写入"), "{error}");
        let names: Vec<String> = std::fs::read_dir(&dir)
            .expect("目录")
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.is_empty(), "失败不留残骸：{names:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupted_file_names_its_path() {
        let dir = scratch("corrupt");
        let path = dir.join("step-7.llmc");
        snapshot().write(&path).expect("写");

        let mut bytes = std::fs::read(&path).expect("读");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("覆盖");

        let error = Snapshot::read(&path).expect_err("校验和不符");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("step-7.llmc"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

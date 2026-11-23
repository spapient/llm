//! 设备后端契约：cutile 只依赖这条缝，不认识 Metal / CUDA / C 的具体实现。

use crate::toolchain::Toolchain;
use crate::witness::{Scope, Selected};
use cutile::decl::Grid;
use cutile::dirs::Dirs;
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error, Result};
use prof::model::Sample;
use prof::sig::Signature;

/// 一次执行回读出来的东西：每条见证写出的个数 + 按顺序回读的 f32 值。
pub type Readback = (Vec<i64>, Vec<f32>);

/// 一段要编译并执行的生产态程序源（verify / bench 的加速器车道）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// 单源文本（C / CUDA）：编译成可执行文件直接跑。
    Single(String),
    /// 双源文本（Metal）：设备源（.metal）编成 metallib，host 驱动（.m）编成可执行文件并发射。
    Pair {
        /// 设备源（MSL）。
        device: String,
        /// host 驱动（ObjC）。
        host: String,
    },
}

/// 并行网格程序的请求：降低形态 + 派发几何。
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    /// 降低形态。
    pub kind: Lower,
    /// 每个维度的工作项数：逐点是元素数、分块族是块数、循环注入是 `[grid]` 各维的样本取值。
    pub items: [usize; 3],
    /// 一个线程组里放几个工作项：逐点取 [`crate::execution::POINT_THREADS`]，其余族一工作项一线程。
    pub group: usize,
}

/// 张量核 GEMM 的几何：输出块的行/列跨度、规约块尺寸，以及每组挂几个 simdgroup。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorGeometry {
    /// 输出块的行跨度（`M` 方向）。
    pub tile_m: usize,
    /// 输出块的列跨度（`N` 方向），也是线程组内存的唯一占用方。
    pub tile_n: usize,
    /// 规约轴切片长度（装载按 16 元素一组）。
    pub k_tile: usize,
    /// 每个线程组挂的 simdgroup 数，组内线程数为 `32·simdgroups`。
    pub simdgroups: usize,
}

impl TensorGeometry {
    /// 显式指定四要素：后端选型、基准对照与专用形状都从这里进来。
    #[must_use]
    pub const fn new(tile_m: usize, tile_n: usize, k_tile: usize, simdgroups: usize) -> Self {
        Self {
            tile_m,
            tile_n,
            k_tile,
            simdgroups,
        }
    }

    /// 规约块是否整除 `K`：整除时生成的内核没有尾块分支。
    #[must_use]
    pub const fn exact(&self, k: usize) -> bool {
        k.is_multiple_of(self.k_tile)
    }

    /// 输出块跨度是否落在矩阵乘单元支持的档位里（`32`、`64`、`128`）。
    #[must_use]
    pub fn spans_ok(&self) -> bool {
        matches!(self.tile_m, 32 | 64 | 128) && matches!(self.tile_n, 32 | 64 | 128)
    }

    /// 规约块形状与 simdgroup 数是否合法。
    #[must_use]
    pub fn block_ok(&self) -> bool {
        self.k_tile >= 32 && self.k_tile.is_multiple_of(16) && matches!(self.simdgroups, 4 | 8)
    }

    /// B 片（`tile_n × k_tile` 的 half，按 4 元素补齐）是否装得进 32 KiB 线程组内存。
    #[must_use]
    pub fn fits_threadgroup_memory(&self) -> bool {
        self.tile_n * (self.k_tile + 4) * 2 <= 32 * 1024
    }
}

/// 生产态内核 lowering 请求：骨架分派后的中性描述，vendor 后端据此生成设备源。
#[derive(Debug, Clone, PartialEq)]
pub enum Lower {
    /// 逐点网格：每线程一个元素，唯一标量传 `1`。
    Pointwise,
    /// 循环注入 / 串行：`dims` 为空即串行兜底（零维网格）。
    Loop(Grid),
    /// CUDA/MUSA int4 GEMV：一个 32-lane warp 协作完成一行，并行解包、缩放与规约。
    QuantizedGemvI4 {
        /// 协作线程数；当前落地要求完整的 32-lane warp。
        threads: usize,
    },
    /// 分块归约：每线程归约一块、主机折叠部分结果。
    ChunkReduce {
        /// 每块元素数。
        chunk: usize,
        /// 软件归约原语名（如 `_cutile_sum_f32`）。
        func: String,
    },
    /// 分块扫描：两段内核（局部扫描 + 校正）。
    ChunkScan {
        /// 每块元素数。
        chunk: usize,
        /// 扫描衰减 `a`（几何级数步长）。
        decay: f32,
    },
    /// 分块 RMSNorm：两段内核（部分平方和 + 归一化）。
    ChunkRmsnorm {
        /// 每块元素数。
        chunk: usize,
    },
    /// 分块 Softmax：三段内核（最大 + 指数和 + 归一化）。
    ChunkSoftmax {
        /// 每块元素数。
        chunk: usize,
    },
    /// 共享内存分块 f32 GEMM：A/B 按 `tile×tile` 块协同搬进 threadgroup 内存，规约轴在共享块上
    /// 复用，语义与标量三重循环逐元素相同。要求 `M/N/K` 均能被 `tile` 整除——部分线程组会破坏
    /// `threadgroup_barrier` 的统一控制流，不满足时由调用方回落到网格/串行路径。
    TiledGemm {
        /// 输出块边长（同时是每维 threadgroup 线程数）。
        tile: usize,
        /// 每个线程计算的输出块边长（`1` = 每线程一输出；`4` = 4×4 寄存器分块）。
        per_thread: usize,
    },
    /// threadgroup 协作的分块注意力：一个线程组负责一个查询行，线程分摊 S 个分数与 D 个输出维，
    /// 行内最大值/指数和走线程组树归约，完整分数留在 threadgroup 内存。要求线程数为 2 的幂，
    /// 且 `S·4` 字节的分数块落在设备线程组内存上限内；不满足时由调用方回落到网格/串行路径。
    TiledAttention {
        /// 每个查询行使用的线程数（2 的幂）。
        threads: usize,
    },
    /// simdgroup 矩阵指令 GEMM：每个 simdgroup（32 线程）用 `8×8×8` 矩阵乘指令算
    /// `row_frags × col_frags` 个 `8×8` 输出片段，K 轴每次前进 8。要求 `K % 8 == 0`、
    /// `M` 被 `8·row_frags` 整除、`N` 被 `8·col_frags` 整除；不满足时由调用方回落。
    SimdGroupGemm {
        /// 每个 simdgroup 沿 M 的输出片段数（每片段 8 行）。
        row_frags: usize,
        /// 每个 simdgroup 沿 N 的输出片段数（每片段 8 列）。
        col_frags: usize,
    },
    /// 张量核矩阵乘 GEMM：一个 `tile_m × tile_n` 输出块由 `32·simdgroups` 个线程协同算完，
    /// 规约轴按 `k_tile` 切片，B 片经线程组内存中转。乘加走硬件的矩阵乘单元而非标量 FMA，
    /// 因此要求 B 预先转成 `N×K` 的低精度布局（布局前置不属于 lowering 本身）。
    /// 要求 `M` 被 `tile_m` 整除、两个块跨度 ∈ {32, 64, 128}、`k_tile` 是 16 的倍数、
    /// `simdgroups ∈ {4, 8}`；不满足时由调用方回落。
    TensorGemm {
        /// 输出块跨度、规约块与 simdgroup 数。
        geometry: TensorGeometry,
    },
    /// flash attention 分块：一个线程组负责一个查询行，K/V 按 `block` 个键分块流式扫描，
    /// 在线 softmax（running max/sum）把输出累加在寄存器，不保留完整分数向量。
    /// 要求线程数为 2 的幂、`block` 是线程数的倍数、`D ≤ 64`；不满足时由调用方回落。
    FlashAttention {
        /// 每个查询行使用的线程数（2 的幂）。
        threads: usize,
        /// 每次流式扫描的键块大小（线程数的倍数）。
        block: usize,
    },
    /// 共享内存分块的对称 int8 GEMM：`i8` 输入、`i32` 精确累加、`f32` 输出。
    Int8Gemm {
        /// 输出块边长，同时是每维 threadgroup 线程数；`tile² ≤ 1024`。
        tile: usize,
    },
    /// 张量核特化的 f32 GEMM（WMMA）：整块 `64×64`、`BK=8`、每 CTA 4 个 `32×32×8` 累积片段。
    /// 形状在生成期已知，越界访问在展开时按整除性决定是否加守卫；不满足 `K % 8 == 0` 的
    /// 形状由调用方回落到循环注入或串行路径。
    WmmaGemm,
}

/// 生成程序文本：纯函数，见证清单（结构化中性输入）→ 一门语言的源文本，可离线对拍。
pub trait Codegen {
    /// 这门语言（[`Emit`] 闭集）。
    fn emit(&self) -> Emit;
    /// 生成「跑一次全部见证」的程序文本；写出上限定在 ABI 常量 [`crate::witness::MAX_OUT`]。
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String>;
    /// 生产态内核 lowering：把一份 `.k` 内核按 [`Lower`] 降成一份或多份设备源（扫描两段、
    /// softmax 三段，其余一段）。host C 后端不实现（生产态设备内核只有 Metal / CUDA 有）。
    fn lower(
        &self,
        _kernel: &str,
        _signature: &Signature,
        _sample: &Sample,
        _kind: &Lower,
    ) -> Result<Vec<String>> {
        Err(Error::new(
            Code::Unsupported,
            "该后端没有生产态 lowering".to_string(),
        ))
    }
    /// 基准封装：把一份 `.k` 内核 + 签名 + 样本降成「跑一次并回读」的完整程序源（verify 的
    /// 加速器车道）。host C 后端不实现（C 基准走产物缓存，不经 [`Driver::compile_run`]；
    /// 设备侧 `Metal` / `Cuda` / `Ascendc` 实现）。
    fn harness(&self, _kernel: &str, _signature: &Signature, _sample: &Sample) -> Result<Source> {
        Err(Error::new(
            Code::Unsupported,
            "该后端没有基准封装".to_string(),
        ))
    }
    /// 并行网格程序：按 [`Launch`] 降源并拼上**按网格发射**的 host 驱动——加速器车道跑的是生产态
    /// 形态（每线程一元素 / 每线程一块 / `[grid]` 各维一线程），不是单线程语义门。
    fn parallel(
        &self,
        _kernel: &str,
        _signature: &Signature,
        _sample: &Sample,
        _launch: &Launch,
    ) -> Result<Source> {
        Err(Error::new(
            Code::Unsupported,
            "该后端没有并行网格程序".to_string(),
        ))
    }
    /// 生产态计时封装：把 lowered kernel 和设备专属 launch geometry 变成带设备计时的程序。
    fn benchmark_harness(
        &self,
        _profile: &Profile,
        _kernel: &str,
        _signature: &Signature,
        _sample: &Sample,
        _kind: &Lower,
        _repeat: usize,
    ) -> Result<Source> {
        Err(Error::new(
            Code::Unsupported,
            "该后端没有生产态设备计时封装".to_string(),
        ))
    }
}

/// 编译 / 发射 / 回读：唯一碰工具链、设备 FFI 与子进程的一半，由设备后端实现。
pub trait Driver {
    /// 设备是否在本机（工具链 / 驱动就位）。
    fn available(&self, profile: &Profile) -> bool;
    /// 让程序跑起来，返回 `(每条写入个数, 按顺序回读的值)`。
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        max: usize,
        scope: Scope,
    ) -> Result<Readback>;
    /// 这条通道实测的工具链（进产物身份 / 报告）；拿不到（缺编译器）时为 `None`。
    fn toolchain(&self) -> Option<Toolchain>;
    /// 生产态内核的「编译 + 执行 + 回读」，返回 `(stdout, stderr)` 原文。与见证 [`run`](Self::run)
    /// 分开：这条是原始子进程路径，`verify` 的加速器车道（CUDA / Metal）从这取回读并解析位模式。
    fn compile_run(
        &self,
        _profile: &Profile,
        _dirs: &Dirs,
        _source: &Source,
        _label: &str,
    ) -> Result<(String, String)> {
        Err(Error::new(
            Code::Unsupported,
            "该后端没有生产态编译车道".to_string(),
        ))
    }
}

/// 一台设备的执行后端：`Codegen` 出源 + `Driver` 跑源。不加新方法，只做「能分派」的并集约束。
/// `Sync` 让后端能被写进进程级注册表（[`register`]）后从任意线程取用。
pub trait Backend: Codegen + Driver + Sync {}

/// 装配一台后端：L7（`cli`）在启动时注入 Metal / CUDA；cutile 已内置 host C 参考后端。
pub fn register(backend: &'static dyn Backend) {
    crate::device::register(backend);
}

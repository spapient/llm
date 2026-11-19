//! 设备后端契约：cutile 只依赖这条缝，不认识 Metal / CUDA / C 的具体实现。
//!
//! 对标 Triton 的 `Backend = Codegen + Driver`：把「生成程序文本」与「编译 / 发射 / 回读」拆成两条
//! trait，各自可独立实现、独立测试。[`Codegen`] 是纯函数（见证清单 → 一门语言的源），可离线对拍；
//! [`Driver`] 是唯一碰工具链与设备 FFI / 子进程的一半。[`Backend`] 只是两者的并集，cutile 只经
//! `&dyn Backend` 分派，不枚举后端。
//!
//! cutile 保留 [`Codegen`] 与 C 宿主参考后端（本机 CPU，不是设备厂商）；Metal / CUDA 的 [`Driver`]
//! 由 L6（`engine`）实现、L7（`cli`）装配注入（`docs/ARCHITECTURE.md` 的 L3 契约：机制层不出现
//! 厂商特判与运行时依赖）。装配走 [`register`]，cutile 的机制层只从这张表取后端，不硬编码后端清单。

use crate::decl::Grid;
use crate::dirs::Dirs;
use crate::prof::{Emit, Profile};
use crate::toolchain::Toolchain;
use crate::witness::{Scope, Selected};
use diagnostic::{Code, Error, Result};
use op::model::Sample;
use op::sig::Signature;

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

/// 生产态内核 lowering 请求：骨架分派后的中性描述，vendor 后端据此生成设备源。
///
/// 它只描述「怎么降」（网格 / 分块 / 串行），不含任何厂商语法；Metal / CUDA 的语法差异收在
/// 各自后端的实现里（`engine`），cutile 的机制层不出现厂商字面量（L3 契约）。
#[derive(Debug, Clone, PartialEq)]
pub enum Lower {
    /// 逐点网格：每线程一个元素，唯一标量传 `1`。
    Pointwise,
    /// 循环注入 / 串行：`dims` 为空即串行兜底（零维网格）。
    Loop(Grid),
    /// 分块归约：每线程归约一块、主机折叠部分结果。
    ChunkReduce { chunk: usize, func: String },
    /// 分块扫描：两段内核（局部扫描 + 校正）。
    ChunkScan { chunk: usize, decay: f32 },
    /// 分块 RMSNorm：两段内核（部分平方和 + 归一化）。
    ChunkRmsnorm { chunk: usize },
    /// 分块 Softmax：三段内核（最大 + 指数和 + 归一化）。
    ChunkSoftmax { chunk: usize },
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
    /// 加速器车道）。host C 后端不实现（C 基准走产物缓存，不经 [`Driver::compile_run`]）。
    fn harness(
        &self,
        _kernel: &str,
        _signature: &Signature,
        _sample: &Sample,
    ) -> Result<Source> {
        Err(Error::new(
            Code::Unsupported,
            "该后端没有基准封装".to_string(),
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
    ///
    /// 把「探针」暴露成方法而不是让机制层直接调 `detect_nvcc` / `detect_metal`：这样后端实现
    /// 搬去 `engine` 后，[`crate::verify`] 仍只经这条缝取工具链身份，不依赖任何厂商检测函数。
    fn toolchain(&self) -> Option<Toolchain>;
    /// 生产态内核的「编译 + 执行 + 回读」，返回 `(stdout, stderr)` 原文。与见证 [`run`](Self::run)
    /// 分开：这条是原始子进程路径，`verify` 的加速器车道（CUDA / Metal）从这取回读并解析位模式。
    ///
    /// 宿主 C 的 `verify` 走产物缓存（`build::execute`），不经这条路径，故默认不支持；Metal / CUDA
    /// 覆盖它。
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
///
/// 后端注册进 cutile 的进程级表后，机制层（`witness` / `verify`）按 `emit` 从表里取，不硬编码
/// 后端清单。注册是幂等的（同一条后端重复注册只留一份，见 [`crate::device::register`]）。
pub fn register(backend: &'static dyn Backend) {
    crate::device::register(backend);
}

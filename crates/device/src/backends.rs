//! Metal / CUDA 设备后端：实现 [`cutile_run::backend::Backend`]（`Codegen` 出源 + `Driver` 跑源）。
//!
//! 这是 L6 的设备落地：唯一碰 `nvcc` / `xcrun metal` / `metallib` / MSVC `cl.exe` 的地方。
//! 厂商编译器的**探测**在 `crate::toolchain`，本模块只按档案取探测结果、拼命令行、跑源。
//! cutile（L3 机制层）只认 [`cutile_run::backend::Backend`] 缝，不认识这里的任何厂商符号；本模块经
//! [`install`] 把 Metal / CUDA 后端注册进 cutile 的进程级表，由 L7（`cli`）在启动时调用。
//!
//! 双源说明：Metal 是「设备源（.metal）+ host 驱动（.m）」两段文本，CUDA 是单段 `.cu`；这个差异
//! 收在 [`cutile_run::backend::Source`] 里，由各自的 [`Driver::compile_run`] 解包，不外泄到机制层。
//!
//! 三条厂商链各自成文件（[`metal`] / [`cuda`] / [`ascend`]）——它们之间没有共享状态，只有基准几何
//! 与见证回读这两段字面共用，因此留在本模块；本模块还剩装配（[`install`]）与三家的原路径再导出。

pub mod cuda;
pub mod metal;

pub use crate::metal::host::{compile_run_metal, metal_driver, metal_kernel};
pub use cuda::compile_run_cuda;
// 三条厂商链共用的导入面：子模块经 `use super::*` 取用，避免各写一份重复块。
pub(crate) use cutile::dirs::Dirs;
pub(crate) use cutile::prof::{Emit, Profile};
pub(crate) use cutile_run::backend::Lower::*;
pub(crate) use cutile_run::backend::{Backend, Codegen, Driver, Launch, Lower, Readback, Source};
pub(crate) use cutile_run::driver;
pub(crate) use cutile_run::toolchain::Toolchain;
pub(crate) use cutile_run::witness::{MAX_OUT, Scope, Selected};
pub(crate) use diagnostic::{Code, Error, Result};
pub(crate) use op::model::Sample;
pub(crate) use op::sig::Signature;

// ---------------------------------------------------------------------------
// 基准几何与见证回读：三条厂商链共用，因此不归任何一家
// ---------------------------------------------------------------------------

fn benchmark_output_len(signature: &Signature, sample: &Sample) -> Result<usize> {
    let output = cutile::harness::output_name(signature)?;
    sample
        .buffer(output)
        .map(|buffer| buffer.values.len())
        .ok_or_else(|| Error::new(Code::Internal, "签名输出缓冲与样本不匹配"))
}

fn benchmark_grid(grid: &op::decl::Grid, sample: &Sample) -> Result<[usize; 3]> {
    if grid.dims.is_empty() || grid.dims.len() > 3 {
        return Err(Error::new(
            Code::Unsupported,
            "设备基准要求一到三个明确的网格维度",
        ));
    }
    let mut dims = [1usize; 3];
    for (index, (_, scalar)) in grid.dims.iter().enumerate() {
        dims[index] = grid_extent(scalar, sample)?;
    }
    Ok(dims)
}

/// 一个网格标量就是这一维的长度：必须是正数的样本取值。
fn grid_extent(scalar: &str, sample: &Sample) -> Result<usize> {
    let value = sample
        .scalar(scalar)
        .ok_or_else(|| Error::new(Code::Invalid, format!("网格标量 `{scalar}` 没有样本取值")))?;
    usize::try_from(value)
        .ok()
        .filter(|extent| *extent > 0)
        .ok_or_else(|| Error::new(Code::Invalid, format!("网格标量 `{scalar}` 必须为正数")))
}

/// CUDA 与昇腾的 host 驱动共用这一段字面：同一个 `driver::readback` 在两种产物上必须读出同一种
/// 文本，各写一份时改一处就会漂。
fn witness_readback(max: usize) -> String {
    format!(
        "\x20   for (int t = 0; t < total; ++t) {{\n\
         \x20       for (int i = 0; i < h_counts[t]; ++i) {{\n\
         \x20           union {{ float f; unsigned int u; }} v;\n\
         \x20           v.f = h_out[t * {max} + i];\n\
         \x20           printf(\"%u\\n\", v.u);\n\
         \x20       }}\n\
         \x20       fprintf(stderr, \"witness %d\\n\", h_counts[t]);\n\
         \x20   }}\n"
    )
}

// ---------------------------------------------------------------------------
// 装配：注册进 cutile 的进程级表
// ---------------------------------------------------------------------------

pub(crate) use cuda::CudaBackend;
pub(crate) use metal::MetalBackend;

use crate::toolchain::{ascend_device, detect_cce};
use std::path::Path;
use std::process::Command;

static METAL_BACKEND: MetalBackend = MetalBackend;
static CUDA_BACKEND: CudaBackend = CudaBackend;
static ASCEND_BACKEND: AscendBackend = AscendBackend;

/// 把 Metal / CUDA / 昇腾后端注册进 cutile 的进程级表（幂等）。L7（`cli`）在启动时调用一次。
pub fn install() {
    cutile_run::backend::register(&METAL_BACKEND);
    cutile_run::backend::register(&CUDA_BACKEND);
    cutile_run::backend::register(&ASCEND_BACKEND);
}

#[cfg(test)]
#[path = "../tests/backends.rs"]
mod tests;

/// 昇腾的可用性：编译器与设备都在（`ccec` 认得出设备，但没有 NPU 也发射不了）。
fn ascend_ready() -> bool {
    detect_cce().is_some() && ascend_device()
}

/// AscendC 是**单源**方言：host 驱动与核函数在同一个翻译单元里，`<<<>>>` 由 ccec 拆成两段。
/// 它与 CUDA 的差别只在三处——核函数前缀（`extern "C" __global__ __aicore__`）、显存指针
/// （`GM_ADDR` / `__gm__`）、以及 host 侧用 AscendCL 而不是 `cudaMalloc` / `cudaMemcpy`。
pub(crate) struct AscendBackend;

impl Codegen for AscendBackend {
    fn emit(&self) -> Emit {
        Emit::Ascendc
    }
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        let _ = scope; // 与 CUDA 同：见证只有设备执行这一条范围
        ascend_program(profile, dirs, selected)
    }

    /// 语义门：`.k` 内核 → `__aicore__` 单源 AscendC 程序（内核 + 单块发射 + AscendCL 回读）。
    ///
    /// 这是 `L2` 那条证据链在昇腾上的最后一段：少这一处，`verify` 的加速器车道在 `ascend` 上
    /// 必然 `Unsupported`，“与参考对拍通过”就只能靠人手贴一张图。它与 `lower` / `parallel` 不同——
    /// 单块标量执行不粉饰性能，只回答“算得对不对”，所以现在就能给。
    fn harness(&self, kernel: &str, signature: &Signature, sample: &Sample) -> Result<Source> {
        crate::ascend::ascend_harness(kernel, signature, sample).map(Source::Single)
    }
}

impl Driver for AscendBackend {
    fn available(&self, _profile: &Profile) -> bool {
        ascend_ready()
    }
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        _count: usize,
        _scope: Scope,
    ) -> Result<Readback> {
        let (stdout, stderr) = compile_run_ascend(profile, dirs, program, "witness")?;
        Ok(driver::readback(&stdout, &stderr))
    }
    fn toolchain(&self) -> Option<Toolchain> {
        detect_cce()
    }
    /// 生产态内核（`lower` / `harness` / `parallel`）仍是空：P0 片段只到标量 C 语义，没有一条
    /// 触及 MTE / Cube / Vector，把它们编成 AscendC 只会得到"能跑但和 CPU 一样慢"的核函数。
    /// 这里如实不支持，调用方按"已声明但没有并行实现"回落——等 P3 的 `DataCopy` / `Mmad` 特化落地。
    ///
    /// 语义门 [`harness`] 是**例外**：它不走这条入口，而是自己拼一份单源 AscendC 程序
    /// （[`crate::ascend::ascend_harness`]），所以走的是上一个分支。
    fn compile_run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        source: &Source,
        label: &str,
    ) -> Result<(String, String)> {
        match source {
            Source::Single(text) => compile_run_ascend(profile, dirs, text, label),
            Source::Pair { .. } => Err(Error::new(
                Code::Unsupported,
                "昇腾后端需要单源（host 驱动 + 核函数在同一份 .cpp）".to_string(),
            )),
        }
    }
}

impl Backend for AscendBackend {}

fn ascend_program(profile: &Profile, dirs: &Dirs, selected: &[Selected<'_>]) -> Result<String> {
    let mut program = ascend_head(profile);
    let files = as_ascend_device(driver::witness_files(selected, dirs, driver::C_PARAMS)?);
    driver::push_sources(&mut program, &files);
    push_ascend_kernel(&mut program, MAX_OUT, selected);
    push_ascend_main(&mut program, MAX_OUT, selected.len());
    Ok(program)
}

/// 头：AscendCL（host 侧分配 / 发射 / 回读）与 AscendC 的内核入口（`__aicore__` / `GM_ADDR` /
/// `GetBlockIdx`）。与 CUDA 的 `<cuda_runtime.h>` 是同一位置的一件事。
fn ascend_head(profile: &Profile) -> String {
    let mut head = String::from(
        "#include \"acl/acl.h\"\n\
         #include \"kernel_operator.h\"\n\
         #include <cstdio>\n\
         #include <cstdlib>\n",
    );
    if !profile.preamble.is_empty() {
        head.push_str(&profile.preamble);
        head.push('\n');
    }
    head
}

/// 见证函数改成 AscendC 设备函数：与 CUDA 的 `__device__` 同一个位置。不加这一层，函数只有宿主
/// 链接性，核函数里调用它会被当成未声明的设备符号；见证本身是纯标量 C，不需要别的改写。
fn as_ascend_device(files: driver::Sources) -> driver::Sources {
    files
        .into_iter()
        .map(|(file, text)| (file, text.replace("int witness", "__aicore__ int witness")))
        .collect()
}

/// 一个见证占一个 AI Core：`GetBlockIdx()` 就是见证序号（与 CUDA 的 `blockIdx.x` 同位）。
fn push_ascend_kernel(program: &mut String, max: usize, selected: &[Selected<'_>]) {
    program.push_str(&format!(
        "extern \"C\" __global__ __aicore__ void cutile_witness(GM_ADDR out_raw, GM_ADDR counts_raw,\n\
         \x20   int total) {{\n\
         \x20   const int tid = (int)AscendC::GetBlockIdx();\n\
         \x20   if (tid >= total) return;\n\
         \x20   __gm__ float *out = (__gm__ float *)out_raw;\n\
         \x20   __gm__ int *counts = (__gm__ int *)counts_raw;\n\
         \x20   float local[{max}];\n"
    ));
    driver::push_dispatch(program, max, selected, "}\n\n");
}

/// host 驱动：AscendCL 分配 GM、同步发射、回读。回读口径与 CUDA 版逐字一致（位模式 + `witness N`），
/// 否则同一个 `driver::readback` 在两台设备上会读出两种东西。
///
/// 调用守卫取自 [`crate::ascend::ACL_GUARD`]：语义门 harness 用的是同一份，两处各写一套时，
/// 加一个 API 调用就会漏掉其中一处。
fn push_ascend_main(program: &mut String, max: usize, total: usize) {
    program.push_str(crate::ascend::ACL_GUARD);
    program.push_str(&format!(
        "\nint main() {{\n\
         \x20   const int total = {total};\n\
         \x20   const size_t out_bytes = (size_t)total * {max} * sizeof(float);\n\
         \x20   const size_t cnt_bytes = (size_t)total * sizeof(int);\n\
         \x20   CUTILE_ACL(aclInit(nullptr));\n\
         \x20   CUTILE_ACL(aclrtSetDevice(0));\n\
         \x20   aclrtStream stream = nullptr;\n\
         \x20   CUTILE_ACL(aclrtCreateStream(&stream));\n\
         \x20   uint8_t *d_out = nullptr;\n\
         \x20   uint8_t *d_counts = nullptr;\n\
         \x20   CUTILE_ACL(aclrtMalloc((void **)&d_out, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST));\n\
         \x20   CUTILE_ACL(aclrtMalloc((void **)&d_counts, cnt_bytes, ACL_MEM_MALLOC_HUGE_FIRST));\n\
         \x20   cutile_witness<<<total, nullptr, stream>>>((GM_ADDR)d_out, (GM_ADDR)d_counts, total);\n\
         \x20   CUTILE_ACL(aclrtSynchronizeStream(stream));\n\
         \x20   float *h_out = (float *)malloc(out_bytes);\n\
         \x20   int *h_counts = (int *)malloc(cnt_bytes);\n\
         \x20   if (h_out == nullptr || h_counts == nullptr) {{\n\
         \x20       fprintf(stderr, \"host malloc\\n\");\n\
         \x20       return 1;\n\
         \x20   }}\n\
         \x20   CUTILE_ACL(aclrtMemcpy(h_out, out_bytes, d_out, out_bytes, ACL_MEMCPY_DEVICE_TO_HOST));\n\
         \x20   CUTILE_ACL(aclrtMemcpy(h_counts, cnt_bytes, d_counts, cnt_bytes, ACL_MEMCPY_DEVICE_TO_HOST));\n"
    ));
    program.push_str(&witness_readback(max));
    program.push_str(
        "\x20   free(h_out); free(h_counts);\n\
         \x20   CUTILE_ACL(aclrtFree(d_out));\n\
         \x20   CUTILE_ACL(aclrtFree(d_counts));\n\
         \x20   CUTILE_ACL(aclrtDestroyStream(stream));\n\
         \x20   CUTILE_ACL(aclrtResetDevice(0));\n\
         \x20   CUTILE_ACL(aclFinalize());\n\
         \x20   return 0;\n\
         }\n",
    );
}

/// 编译并运行任意 AscendC 程序，返回 `(stdout, stderr)` 原文：临时目录、ccec、执行。
pub fn compile_run_ascend(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    label: &str,
) -> Result<(String, String)> {
    let dir = driver::scratch(dirs, &format!("{label}-ascend"));
    let outcome = ascend_in_dir(&dir, profile, program, label, dirs);
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn ascend_in_dir(
    dir: &Path,
    profile: &Profile,
    program: &str,
    label: &str,
    dirs: &Dirs,
) -> Result<(String, String)> {
    let src = dir.join(format!("{label}.cpp"));
    let exe = dir.join(format!("{label}.exe"));
    driver::write_text(&src, program)?;
    ascend_compile(&exe, &src, profile, dirs)?;
    let mut command = Command::new(&exe);
    driver::run_exe(&mut command, &exe, "昇腾执行失败")
}

fn ascend_compile(exe: &Path, src: &Path, profile: &Profile, dirs: &Dirs) -> Result<()> {
    let tool = profile.meta.tool.as_str();
    let mut command = Command::new(tool);
    // `-xcce` / `--cce-aicore-algorithm` 归档案的 `[compile].flags`：那是"哪个核芯"的声明，本层不猜
    // （`dav-c220` 只对 `id = "910b"` 成立）。唯一的分拣是顺序：`-l` 类参数被转给链接器，排在源
    // 文件之前就是"undefined reference to aclInit"，所以它们挪到源文件之后。
    let flags = cutile_run::compile_flags(profile, dirs)?;
    let (links, options): (Vec<String>, Vec<String>) =
        flags.into_iter().partition(|flag| flag.starts_with("-l"));
    command
        .args(options)
        .arg(src)
        .arg("-o")
        .arg(exe)
        .args(links);
    driver::run_tool(tool, &mut command, src)
}

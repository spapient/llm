//! Metal / CUDA 设备后端：实现 [`llm_optim::backend::Backend`]（`Codegen` 出源 + `Driver` 跑源）。

pub use crate::metal::{compile_run_metal, metal_driver, metal_kernel};
// 三条厂商链共用的导入面：避免各写一份重复块。
pub(crate) use cutile::dirs::Dirs;
pub(crate) use cutile::prof::{Emit, Profile};
pub(crate) use diagnostic::{Code, Error, Result};
pub(crate) use llm_optim::backend::Lower::*;
pub(crate) use llm_optim::backend::{Backend, Codegen, Driver, Launch, Lower, Readback, Source};
pub(crate) use llm_optim::driver;
pub(crate) use llm_optim::toolchain::Toolchain;
pub(crate) use llm_optim::witness::{MAX_OUT, Scope, Selected};
pub(crate) use nn::model::Sample;
pub(crate) use nn::sig::Signature;

// ---------------------------------------------------------------------------
// 基准几何与见证回读：三条厂商链共用，因此不归任何一家
// ---------------------------------------------------------------------------

fn benchmark_output_len(signature: &Signature, sample: &Sample) -> Result<usize> {
    let output = cutile::harness::output_name(signature)?;
    sample
        .buf(output)
        .map(|buf| buf.values.len())
        .ok_or_else(|| Error::new(Code::Internal, "签名输出缓冲与样本不匹配"))
}

fn benchmark_grid(grid: &prof::decl::Grid, sample: &Sample) -> Result<[usize; 3]> {
    if grid.dims.is_empty() || grid.dims.len() > 3 {
        return Err(Error::new(
            Code::Unsupported,
            "设备基准要求一到三个明确的网格维度",
        ));
    }
    let mut dims = [1usize; 3];
    for (idx, (_, scalar)) in grid.dims.iter().enumerate() {
        dims[idx] = grid_extent(scalar, sample)?;
    }
    Ok(dims)
}

/// 一个必须为正的整数标量：网格维（[`grid_extent`]）与输出宽度（[`warp_rows`]）共用这一判据，
/// 缺取值、负数与零全在这一处报。
fn positive_scalar(sample: &Sample, scalar: &str, what: &str) -> Result<usize> {
    let value = sample
        .scalar(scalar)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{what} `{scalar}` 没有样本取值")))?;
    usize::try_from(value)
        .ok()
        .filter(|extent| *extent > 0)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{what} `{scalar}` 必须为正数")))
}

/// 一个网格维的长度：逐点网格与分块网格选型共用。
fn grid_extent(scalar: &str, sample: &Sample) -> Result<usize> {
    positive_scalar(sample, scalar, "网格标量")
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

use crate::metal::run_metal;
use crate::platform::{
    ascend_device, detect_cce, detect_metal, detect_musa, detect_nvcc, find_cl, musa_device,
};
use std::path::Path;
use std::process::Command;

static METAL_BACKEND: MetalBackend = MetalBackend;
static CUDA_BACKEND: CudaBackend = CudaBackend;
static ASCEND_BACKEND: AscendBackend = AscendBackend;

/// 把 Metal / CUDA / 昇腾后端注册进 cutile 的进程级表（幂等）。L7（`cli`）在启动时调用一次。
pub fn install() {
    llm_optim::backend::register(&METAL_BACKEND);
    llm_optim::backend::register(&CUDA_BACKEND);
    llm_optim::backend::register(&ASCEND_BACKEND);
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
    fn harness(&self, kernel: &str, signature: &Signature, sample: &Sample) -> Result<Source> {
        ascend_harness(kernel, signature, sample).map(Source::Single)
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
    let mut program = witness_head(
        "#include \"acl/acl.h\"\n#include \"kernel_operator.h\"\n#include <cstdio>\n#include <cstdlib>\n",
        profile,
    );
    let files = device_fn(
        driver::witness_files(selected, dirs, driver::C_PARAMS)?,
        "__aicore__",
    );
    driver::push_sources(&mut program, &files);
    push_ascend_kernel(&mut program, MAX_OUT, selected);
    push_ascend_main(&mut program, MAX_OUT, selected.len());
    Ok(program)
}

/// 见证程序头：包含行 + 档案 preamble（非空才追加，与三家编译器的 `-include` 同位）。
fn witness_head(includes: &str, profile: &Profile) -> String {
    let mut head = String::from(includes);
    if !profile.preamble.is_empty() {
        head.push_str(&profile.preamble);
        head.push('\n');
    }
    head
}

/// 见证函数加设备函数前缀（CUDA `__device__` / AscendC `__aicore__`）：不加这一层，函数只有宿主
/// 链接性，核函数里调用它会被当成未声明的设备符号；见证本身是纯标量 C，不需要别的改写。
fn device_fn(files: driver::Sources, prefix: &str) -> driver::Sources {
    files
        .into_iter()
        .map(|(file, text)| {
            (
                file,
                text.replace("int witness", &format!("{prefix} int witness")),
            )
        })
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
fn push_ascend_main(program: &mut String, max: usize, total: usize) {
    program.push_str(ACL_GUARD);
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
    let flags = llm_optim::compile_flags(profile, dirs)?;
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

// ---------------------------------------------------------------------------
// CUDA / MUSA 后端：`Codegen` 出 `.cu`、`Driver` 跑 `nvcc`。
// ---------------------------------------------------------------------------

fn cuda_bench_geometry(
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<([usize; 3], [usize; 3], String)> {
    match kind {
        Lower::Pointwise => pointwise_wrap(signature, sample),
        Lower::Loop(grid) => loop_wrap(signature, sample, grid),
        Lower::QuantizedGemvI4 { threads } => warp_wrap(signature, sample, *threads),
        _ => Err(Error::new(
            Code::Unsupported,
            "该 lowering 尚无单 kernel 设备计时封装",
        )),
    }
}

/// 逐点族：一条线程一个输出元素，块宽 256。
fn pointwise_wrap(
    signature: &Signature,
    sample: &Sample,
) -> Result<([usize; 3], [usize; 3], String)> {
    let len = benchmark_output_len(signature, sample)?;
    Ok((
        [len.div_ceil(256), 1, 1],
        [256, 1, 1],
        crate::cuda::buf_args(signature),
    ))
}

/// 多段族：网格就是声明里的网格，一个线程一段。
fn loop_wrap(
    signature: &Signature,
    sample: &Sample,
    grid: &prof::decl::Grid,
) -> Result<([usize; 3], [usize; 3], String)> {
    Ok((
        benchmark_grid(grid, sample)?,
        [1, 1, 1],
        crate::dialect::launch_args(sample, signature.params())?,
    ))
}

/// i4 GEMV 一行一个 warp：`grid = [N, 1, 1]`、`block = [threads, 1, 1]`。
fn warp_wrap(
    signature: &Signature,
    sample: &Sample,
    threads: usize,
) -> Result<([usize; 3], [usize; 3], String)> {
    let rows = warp_rows(sample)?;
    Ok((
        [rows, 1, 1],
        [warp_lanes(threads)?, 1, 1],
        crate::dialect::launch_args(sample, signature.params())?,
    ))
}

/// GEMV 的规约按 lane 展开，因此 32-lane warp 是硬条件而非默认值。
fn warp_lanes(threads: usize) -> Result<usize> {
    if threads != 32 {
        return Err(Error::new(Code::Unsupported, "CUDA GEMV 要求 32-lane warp"));
    }
    Ok(threads)
}

/// 输出宽度 `N` 必须存在且为正整数：零行或负行没有可写的输出。
fn warp_rows(sample: &Sample) -> Result<usize> {
    positive_scalar(sample, "N", "gemv_i4 输出宽度")
}

pub(crate) struct CudaBackend;

impl Codegen for CudaBackend {
    fn emit(&self) -> Emit {
        Emit::Cuda
    }
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        let _ = scope; // CUDA 只走设备执行
        cuda_program(profile, dirs, selected)
    }
    fn lower(
        &self,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        kind: &Lower,
    ) -> Result<Vec<String>> {
        cuda_lower(kernel, signature, sample, kind)
    }
    fn harness(&self, kernel: &str, signature: &Signature, sample: &Sample) -> Result<Source> {
        Ok(Source::Single(crate::cuda::cuda_harness(
            kernel, signature, sample,
        )?))
    }
    fn parallel(
        &self,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        launch: &Launch,
    ) -> Result<Source> {
        cuda(kernel, signature, sample, launch).map(Source::Single)
    }
    fn benchmark_harness(
        &self,
        profile: &Profile,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        kind: &Lower,
        repeat: usize,
    ) -> Result<Source> {
        let lowered = cuda_lower(kernel, signature, sample, kind)?;
        let [device] = lowered.as_slice() else {
            return Err(Error::new(
                Code::Unsupported,
                "CUDA 基准当前只支持单段生产态 kernel",
            ));
        };
        let (grid, threads, args) = cuda_bench_geometry(signature, sample, kind)?;
        let source = crate::cuda::cuda_benchmark_driver(
            device,
            signature,
            sample,
            grid,
            threads,
            &args,
            repeat,
            profile.meta.tool == "mcc",
        )?;
        Ok(Source::Single(source))
    }
}

impl Driver for CudaBackend {
    /// 可用性按**档案声明的工具名**判，不按厂商判：同一份 `Emit::Cuda` 后端既是 `nvcc` 的落地，
    /// 也是 `mcc` 的落地。`nvcc` 在 Windows 上还要 MSVC 宿主；`mcc` 不调宿主 cl（它自己就是 clang）。
    fn available(&self, profile: &Profile) -> bool {
        match profile.meta.tool.as_str() {
            "nvcc" => nvcc_ready(),
            "mcc" => musa_ready(),
            _ => false,
        }
    }
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        _count: usize,
        _scope: Scope,
    ) -> Result<Readback> {
        let (stdout, stderr) = compile_run_cuda(profile, dirs, program, "witness")?;
        Ok(driver::readback(&stdout, &stderr))
    }
    /// 缓存指纹里的工具链：按现有工具选一份（同一台机器上不会同时是 CUDA 与 MUSA 的编译机，
    /// 真同时有时取先命中的那份，两个档案的缓存键因此天然分开）。
    fn toolchain(&self) -> Option<Toolchain> {
        detect_nvcc().or_else(detect_musa)
    }
    fn compile_run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        source: &Source,
        label: &str,
    ) -> Result<(String, String)> {
        match source {
            Source::Single(text) => compile_run_cuda(profile, dirs, text, label),
            Source::Pair { .. } => Err(Error::new(
                Code::Unsupported,
                "CUDA 后端需要单源（.cu）".to_string(),
            )),
        }
    }
}

impl Backend for CudaBackend {}

/// 把中性的 [`Lower`] 描述降成 CUDA：逐点网格与分块归约已落地，其余骨架尚未覆盖。
fn cuda_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        // 逐块归一化两族与扫描自己就产出多段源码，不经过下面的单文件包装。
        ChunkRmsnorm { chunk } => {
            crate::cuda::cuda_chunk_rmsnorm(kernel, signature, sample, *chunk)
        }
        ChunkSoftmax { chunk } => {
            crate::cuda::cuda_chunk_softmax(kernel, signature, sample, *chunk)
        }
        ChunkScan { chunk, decay } => {
            crate::cuda::cuda_chunk_scan(kernel, signature, sample, *chunk, *decay)
        }
        QuantizedGemvI4 { threads } => {
            crate::cuda::cuda_quantized_gemv_i4(signature, *threads).map(|source| vec![source])
        }
        // 其余骨架都只吐一段源码；分派单独一层，是为了让这一层只回答「哪一族」。
        _ => cuda_single_source(kind, kernel, signature, sample).map(|source| vec![source]),
    }
}

/// 单段源码骨架的分派：pointwise / chunk-reduce / loop / wmma 各产出一段源码。
fn cuda_single_source(
    kind: &Lower,
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    match kind {
        Pointwise => crate::cuda::cuda_grid_kernel(kernel, signature, sample),
        ChunkReduce { chunk, func } => {
            crate::cuda::cuda_chunk_reduce(kernel, signature, sample, *chunk, func)
        }
        Loop(grid) => crate::cuda::cuda_loop_kernel(kernel, signature, sample, grid),
        WmmaGemm => crate::cuda::gemm_kernel(signature, sample),
        _ => Err(Error::new(
            Code::Internal,
            "该骨架产出多段源码，应在 cuda_lower 分派，不应落到单段兜底".to_string(),
        )),
    }
}

fn cuda_program(profile: &Profile, dirs: &Dirs, selected: &[Selected<'_>]) -> Result<String> {
    let mut program = witness_head("#include <cuda_runtime.h>\n#include <cstdio>\n", profile);
    let files = device_fn(
        driver::witness_files(selected, dirs, driver::C_PARAMS)?,
        "__device__",
    );
    driver::push_sources(&mut program, &files);
    push_cuda_kernel(&mut program, MAX_OUT, selected);
    push_cuda_main(&mut program, MAX_OUT, selected.len());
    Ok(program)
}

fn push_cuda_kernel(program: &mut String, max: usize, selected: &[Selected<'_>]) {
    program.push_str(&format!(
        "__global__ void cutile_witness(float *out, int *counts, int total) {{\n\
         \x20   int tid = blockIdx.x;\n\
         \x20   if (tid >= total) return;\n\
         \x20   float local[{max}];\n"
    ));
    driver::push_dispatch(program, max, selected, "}\n\n");
}

/// 回读口径：逐见证打印位模式，再打一行 `witness N`。
fn push_cuda_main(program: &mut String, max: usize, total: usize) {
    program.push_str(&format!(
        "int main() {{\n\
         \x20   const int total = {total};\n\
         \x20   float *d_out = 0; int *d_counts = 0;\n\
         \x20   cudaMalloc((void **)&d_out, total * {max} * sizeof(float));\n\
         \x20   cudaMalloc((void **)&d_counts, total * sizeof(int));\n\
         \x20   cutile_witness<<<total, 1>>>(d_out, d_counts, total);\n\
         \x20   cudaError_t error = cudaDeviceSynchronize();\n\
         \x20   if (error != cudaSuccess) {{\n\
         \x20       fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error));\n\
         \x20       return 1;\n\
         \x20   }}\n\
         \x20   float h_out[total * {max}]; int h_counts[total];\n\
         \x20   cudaMemcpy(h_out, d_out, total * {max} * sizeof(float), cudaMemcpyDeviceToHost);\n\
         \x20   cudaMemcpy(h_counts, d_counts, total * sizeof(int), cudaMemcpyDeviceToHost);\n"
    ));
    program.push_str(&witness_readback(max));
    program.push_str(
        "\x20   cudaFree(d_out); cudaFree(d_counts);\n\
         \x20   return 0;\n\
         \x20   }\n",
    );
}

/// 编译并运行任意 CUDA 程序，返回 `(stdout, stderr)` 原文：临时目录、nvcc（Windows 用 `-ccbin`
/// 指向 cl）、执行。见证与算子 verify 共用这一条；写、编、跑三段分开。
pub fn compile_run_cuda(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    label: &str,
) -> Result<(String, String)> {
    let dir = driver::scratch(dirs, &format!("{label}-cuda"));
    let (src, exe) = (
        dir.join(format!("{label}.cu")),
        dir.join(format!("{label}.exe")),
    );
    let outcome = (|| {
        driver::write_text(&src, program)?;
        cuda_compile(&exe, &src, profile, dirs)?;
        let mut command = Command::new(&exe);
        driver::run_exe(&mut command, &exe, "CUDA 执行失败")
    })();
    let _ = std::fs::remove_dir_all(&dir);
    outcome
}

fn cuda_compile(exe: &Path, src: &Path, profile: &Profile, dirs: &Dirs) -> Result<()> {
    let tool = profile.meta.tool.as_str();
    let mut command = Command::new(tool);
    // 头文件垫片归档案（`[compile.headers]`）：生成程序里的 `<cuda_runtime.h>` 与工具链实际提供的
    // `<musa_runtime.h>` 之间的映射写在 `devices/musa.toml`，本层只负责把它变成 `-I`。
    command.args(llm_optim::compile_flags(profile, dirs)?);
    cuda_ccbin(tool, &mut command)?;
    command.arg("-o").arg(exe).arg(src);
    driver::run_tool(tool, &mut command, src)
}

/// nvcc 的可用性：编译器在，且 Windows 上还能找到 MSVC 的 `cl.exe`。
fn nvcc_ready() -> bool {
    detect_nvcc().is_some() && (std::env::consts::OS != "windows" || find_cl().is_some())
}

/// MUSA 的可用性：编译器与设备都在。
fn musa_ready() -> bool {
    detect_musa().is_some() && musa_device()
}

/// Windows 上的 nvcc 需要显式指定 MSVC 编译器。
fn cuda_ccbin(tool: &str, command: &mut Command) -> Result<()> {
    if !(tool == "nvcc" && std::env::consts::OS == "windows") {
        return Ok(());
    }
    let Some(cl) = find_cl() else {
        return Err(Error::new(
            Code::Unsupported,
            "找不到 MSVC cl.exe：nvcc 设备编译需要它".to_string(),
        ));
    };
    command.arg("-ccbin").arg(cl);
    Ok(())
}

// ---------------------------------------------------------------------------
// Metal 后端：`Codegen` 出 MSL、`Driver` 跑 `xcrun metal` + `metallib`。
// ---------------------------------------------------------------------------

/// MSL 的形参必须带地址空间：MSL 没有默认地址空间，裸 `float *` 在那里是语法错误。
const MSL_PARAMS: &str = "thread float *";

fn metal_bench_geometry(
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<([usize; 3], [usize; 3])> {
    match kind {
        Lower::Pointwise => Ok((
            [benchmark_output_len(signature, sample)?, 1, 1],
            [256, 1, 1],
        )),
        Lower::Loop(grid) => Ok((benchmark_grid(grid, sample)?, [1, 1, 1])),
        _ => Err(Error::new(
            Code::Unsupported,
            "该 lowering 尚无单 kernel 设备计时封装",
        )),
    }
}

pub(crate) struct MetalBackend;

impl Codegen for MetalBackend {
    fn emit(&self) -> Emit {
        Emit::Metal
    }
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        let _ = scope; // Metal 只走设备执行，host 自检由 C 后端承担
        metal_program(profile, dirs, selected)
    }
    fn lower(
        &self,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        kind: &Lower,
    ) -> Result<Vec<String>> {
        metal_lower(kernel, signature, sample, kind)
    }
    fn harness(&self, kernel: &str, signature: &Signature, sample: &Sample) -> Result<Source> {
        let (device, host) = crate::metal::metal_harness(kernel, signature, sample)?;
        Ok(Source::Pair { device, host })
    }
    fn parallel(
        &self,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        launch: &Launch,
    ) -> Result<Source> {
        metal(kernel, signature, sample, launch)
    }
    fn benchmark_harness(
        &self,
        _profile: &Profile,
        kernel: &str,
        signature: &Signature,
        sample: &Sample,
        kind: &Lower,
        repeat: usize,
    ) -> Result<Source> {
        let lowered = metal_lower(kernel, signature, sample, kind)?;
        let [device] = lowered.as_slice() else {
            return Err(Error::new(
                Code::Unsupported,
                "Metal 基准当前只支持单段生产态 kernel",
            ));
        };
        let (grid, threads) = metal_bench_geometry(signature, sample, kind)?;
        let host = crate::metal::metal_benchmark_driver(signature, sample, grid, threads, repeat)?;
        Ok(Source::Pair {
            device: device.clone(),
            host,
        })
    }
}

impl Driver for MetalBackend {
    fn available(&self, _: &Profile) -> bool {
        detect_metal().is_some()
    }
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        count: usize,
        _scope: Scope,
    ) -> Result<Readback> {
        run_metal(program, count, &profile.flags, dirs)
    }
    fn toolchain(&self) -> Option<Toolchain> {
        detect_metal()
    }
    fn compile_run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        source: &Source,
        label: &str,
    ) -> Result<(String, String)> {
        match source {
            Source::Pair { device, host } => compile_run_metal(profile, dirs, device, host, label),
            Source::Single(_) => Err(Error::new(
                Code::Unsupported,
                "Metal 后端需要双源（MSL + host 驱动）".to_string(),
            )),
        }
    }
}

impl Backend for MetalBackend {}

/// 把中性的 [`Lower`] 描述降成 MSL：逐点/循环走网格或串行，分块族（归约/扫描/归一化）走分块内核。
fn metal_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        Pointwise => crate::metal::metal_grid_kernel(kernel, signature, sample).map(|s| vec![s]),
        QuantizedGemvI4 { .. } => Err(Error::new(
            Code::Unsupported,
            "合作式 int4 GEMV 当前仅落地在 CUDA/MUSA",
        )),
        Loop(grid) => {
            crate::metal::metal_loop_kernel(kernel, signature, sample, grid).map(|s| vec![s])
        }
        ChunkReduce { .. } | ChunkScan { .. } | ChunkRmsnorm { .. } | ChunkSoftmax { .. } => {
            metal_chunk_lower(kernel, signature, sample, kind)
        }
        TiledGemm { .. }
        | SimdGroupGemm { .. }
        | TensorGemm { .. }
        | Int8Gemm { .. }
        | TiledAttention { .. }
        | FlashAttention { .. }
        | WmmaGemm => metal_special(kernel, signature, sample, kind),
    }
}

/// 特殊 lowering 家族（分块 / simdgroup GEMM、分块 / flash 注意力、WMMA）：单独一个 match 让 [`metal_lower`] 的
/// 骨架分派保持在门禁臂数内。
fn metal_special(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        TiledGemm { .. }
        | SimdGroupGemm { .. }
        | TensorGemm { .. }
        | Int8Gemm { .. }
        | WmmaGemm => metal_gemm_lower(kernel, signature, sample, kind),
        TiledAttention { .. } | FlashAttention { .. } => {
            metal_attention_lower(kernel, signature, sample, kind)
        }
        _ => Err(Error::new(
            Code::Internal,
            "非特殊 lowering 不应进入 metal_special".to_string(),
        )),
    }
}

/// GEMM 家族（共享内存分块 / simdgroup 矩阵指令 / WMMA 张量核）的 Metal 落地。
fn metal_gemm_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        TiledGemm { tile, per_thread } => {
            crate::tiled::metal_tiled_gemm(kernel, signature, sample, *tile, *per_thread)
                .map(|s| vec![s])
        }
        SimdGroupGemm {
            row_frags,
            col_frags,
        } => crate::tiled::metal_simdgroup_gemm(kernel, signature, sample, *row_frags, *col_frags)
            .map(|s| vec![s]),
        TensorGemm { geometry } => {
            crate::tiled::metal_tensor_gemm(kernel, signature, sample, *geometry).map(|s| vec![s])
        }
        Int8Gemm { tile } => {
            crate::tiled::metal_int8_gemm(kernel, signature, sample, *tile).map(|s| vec![s])
        }
        other => Err(gemm_unsupported(other)),
    }
}

/// GEMM lowering 不在 Metal 生产态内的错误：WMMA 属未覆盖，其余是调用方走错入口。
fn gemm_unsupported(kind: &Lower) -> Error {
    match kind {
        WmmaGemm => Error::new(
            Code::Unsupported,
            "Metal 生产态 lowering 未覆盖 WMMA 张量核特化".to_string(),
        ),
        _ => Error::new(
            Code::Internal,
            "非 GEMM lowering 不应进入 metal_gemm_lower".to_string(),
        ),
    }
}

/// 注意力家族（共享内存分块 / flash 在线 softmax）的 Metal 落地。
fn metal_attention_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        TiledAttention { threads } => {
            crate::tiled::metal_tiled_attention(kernel, signature, sample, *threads)
                .map(|s| vec![s])
        }
        FlashAttention { threads, block } => {
            crate::tiled::metal_flash_attention(kernel, signature, sample, *threads, *block)
                .map(|s| vec![s])
        }
        _ => Err(Error::new(
            Code::Internal,
            "非注意力 lowering 不应进入 metal_attention_lower".to_string(),
        )),
    }
}

/// 分块归约/扫描/逐行规约的 Metal lowering：4 种分块骨架各自降成 1–3 段设备源。
/// 只应由 [`metal_lower`] 在已确认分块后调用；非分块算子返回内部错误而非 panic。
fn metal_chunk_lower(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    kind: &Lower,
) -> Result<Vec<String>> {
    match kind {
        ChunkReduce { chunk, func } => {
            crate::metal::metal_chunk_reduce(kernel, signature, sample, *chunk, func)
                .map(|s| vec![s])
        }
        ChunkScan { chunk, decay } => {
            crate::metal::metal_chunk_scan(kernel, signature, sample, *chunk, *decay)
                .map(|(local, correct)| vec![local, correct])
        }
        ChunkRmsnorm { chunk } => {
            crate::metal::metal_chunk_rmsnorm(kernel, signature, sample, *chunk)
                .map(|(partial, map)| vec![partial, map])
        }
        ChunkSoftmax { chunk } => {
            crate::metal::metal_chunk_softmax(kernel, signature, sample, *chunk)
                .map(|(max, expsum, map)| vec![max, expsum, map])
        }
        Pointwise
        | Loop(_)
        | QuantizedGemvI4 { .. }
        | TiledGemm { .. }
        | SimdGroupGemm { .. }
        | TensorGemm { .. }
        | Int8Gemm { .. }
        | TiledAttention { .. }
        | FlashAttention { .. }
        | WmmaGemm => Err(Error::new(
            Code::Internal,
            "非分块算子不应进入 Metal 分块 lowering",
        )),
    }
}

fn metal_program(profile: &Profile, dirs: &Dirs, selected: &[Selected<'_>]) -> Result<String> {
    let mut program = metal_head(profile);
    driver::push_sources(
        &mut program,
        &driver::witness_files(selected, dirs, MSL_PARAMS)?,
    );
    push_metal_kernel(&mut program, MAX_OUT, selected);
    Ok(program)
}

fn metal_head(profile: &Profile) -> String {
    let mut head = profile.preamble.clone();
    driver::normalize_newline(&mut head);
    head
}

fn push_metal_kernel(program: &mut String, max: usize, selected: &[Selected<'_>]) {
    program.push_str(&format!(
        "kernel void cutile_witness(device float *out [[buffer(0)]],\n\
         \x20                          device int *counts [[buffer(1)]],\n\
         \x20                          uint tid [[thread_position_in_grid]]) {{\n\
         \x20   thread float local[{max}];\n"
    ));
    driver::push_dispatch(program, max, selected, "}\n");
}

// ---------------------------------------------------------------------------
// 昇腾侧的设备源与主机驱动拼装：`__aicore__` 单源、`<<<blocks, nullptr, stream>>>` 发射、
// `aclrtMalloc` / `aclrtMemcpy` / `aclrtFree` 骨架与回读
// ---------------------------------------------------------------------------

use crate::cuda::host_buf_decl;
use crate::dialect::readback_bits;
use cutile::harness::{c_type, output_of};
use nn::model::Values;
use nn::sig::Param;

/// AscendCL 的调用守卫：失败时报出是哪个调用并返回非零。
pub(crate) const ACL_GUARD: &str = "static int cutile_acl(const char *what, aclError code) {\n\
     \x20   if (code == ACL_SUCCESS) return 0;\n\
     \x20   fprintf(stderr, \"acl %s: %d\\n\", what, (int)code);\n\
     \x20   return -1;\n\
     }\n\
     #define CUTILE_ACL(call) do { if (cutile_acl(#call, (call)) != 0) return 1; } while (0)\n";

/// 拼出可在 NPU 上真实执行的 AscendC 程序：`.k` 内核转 `__aicore__`，单块 `<<<1, nullptr, stream>>>`
/// 包装里调用，host `main` 负责分配 / 拷贝 / 回读——**语义对拍门**，性能另由 roofline 门承载。
///
/// 单块（`block = 1`）与 CUDA 的 `<<<1,1>>>` 是同一件事：这一档只回答"算出来的数对不对"，
/// 不回答"跑得多快"；并行形态要等 P3 的 `DataCopy` / `Mmad` 特化。
///
/// # Errors
///
/// 缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致时返回说明。
pub fn ascend_harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let kernel = signature.kernel();
    let mut text = format!(
        "#include \"acl/acl.h\"\n\
         #include \"kernel_operator.h\"\n\
         #include <cstdio>\n\
         #include <cstdlib>\n\
         /* cutile ascend harness | kernel={kernel} m={} n={} | 单块设备执行（语义门，非并行性能门） */\n",
        sample.m, sample.n
    );
    let device = as_ascend_source(kernel_source, &kernel);
    text.push_str(&device);
    if !device.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&ascend_wrapper(&kernel, signature, sample)?);
    text.push_str(&ascend_main(signature, sample, output, values.len())?);
    Ok(text)
}

/// 把模板翻译单元改成设备源：内核函数补 `__aicore__`，`[compile].preamble` 里的软件实现
/// （`_cutile_sum_f32` 一族）同样要带——它们是 `@reduce` 的 `$func` 与几个软件数学函数的落点，
/// 不加就是"AI Core 函数调用了宿主函数"，编译期会被拒。
pub(crate) fn as_ascend_source(text: &str, kernel: &str) -> String {
    text.replace(
        &format!("void {kernel}("),
        &format!("__aicore__ void {kernel}("),
    )
    .replace("static inline ", "static __aicore__ inline ")
}

/// 设备侧入口：`extern "C" __global__ __aicore__` 的核函数，形参用 `GM_ADDR` 收显存指针。
fn ascend_wrapper(kernel: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let params = signature
        .params()
        .iter()
        .map(|param| {
            if param.ctype.is_buf() {
                format!("GM_ADDR d_{}", param.name)
            } else {
                format!("{} {}", param.ctype.c_name(), param.name)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let call = signature
        .params()
        .iter()
        .map(|param| ascend_argument(param, sample))
        .collect::<Result<Vec<_>>>()?
        .join(", ");
    Ok(format!(
        "\nextern \"C\" __global__ __aicore__ void cutile_launch({params}) {{\n    {kernel}({call});\n}}\n"
    ))
}

/// 内核实参：缓冲传补好元素类型的 `__gm__` 指针，标量取样本值（与 CUDA 的 `launch_args` 同序同值）。
fn ascend_argument(param: &Param, sample: &Sample) -> Result<String> {
    if param.ctype.is_buf() {
        let base = param.ctype.c_name();
        return Ok(if param.mutable {
            format!("(__gm__ {base} *)d_{}", param.name)
        } else {
            format!("(__gm__ const {base} *)d_{}", param.name)
        });
    }
    let value = sample.scalar(&param.name).ok_or_else(|| {
        Error::new(
            Code::Internal,
            format!("标量实参 `{}` 没有取值", param.name),
        )
    })?;
    Ok(value.to_string())
}

/// host 驱动：AscendCL 分配 GM、单块发射、回读、释放。
fn ascend_main(
    signature: &Signature,
    sample: &Sample,
    output: &str,
    output_len: usize,
) -> Result<String> {
    let mut text = String::from(ACL_GUARD);
    text.push_str(
        "\nint main(void) {\n\
         \x20   CUTILE_ACL(aclInit(nullptr));\n\
         \x20   CUTILE_ACL(aclrtSetDevice(0));\n\
         \x20   aclrtStream stream = nullptr;\n\
         \x20   CUTILE_ACL(aclrtCreateStream(&stream));\n",
    );
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
    {
        text.push_str(&ascend_buf(param, sample, output, output_len)?);
    }
    text.push_str(&format!(
        "    cutile_launch<<<1, nullptr, stream>>>({});\n\
         \x20   CUTILE_ACL(aclrtSynchronizeStream(stream));\n\
         \x20   CUTILE_ACL(aclrtMemcpy(h_{output}, {output_len} * sizeof(float), d_{output}, \
         {output_len} * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));\n{}",
        ascend_launch_args(signature, sample)?,
        readback_bits(&format!("h_{output}"), output_len, "    ")
    ));
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
    {
        text.push_str(&format!("    CUTILE_ACL(aclrtFree(d_{}));\n", param.name));
    }
    text.push_str(
        "    CUTILE_ACL(aclrtDestroyStream(stream));\n\
         \x20   CUTILE_ACL(aclrtResetDevice(0));\n\
         \x20   CUTILE_ACL(aclFinalize());\n\
         \x20   return 0;\n\
         }\n",
    );
    Ok(text)
}

/// 单个缓冲的 host 声明 + GM 分配 + （只读时）入向拷贝；输出缓冲的长度取 `output_len`
/// （与 CUDA 同口径：分块归约的产物长度可以大于签名的单元素 `y`）。
fn ascend_buf(param: &Param, sample: &Sample, output: &str, output_len: usize) -> Result<String> {
    let Some(buf) = sample.buf(&param.name) else {
        return Err(Error::new(Code::Internal, "签名与样本已对齐（漂移）"));
    };
    let values: &Values = &buf.values;
    let ctype = c_type(values);
    let size = if param.name == output {
        output_len
    } else {
        values.len()
    };
    let mut text = host_buf_decl(param, values, ctype, size);
    text.push_str(&format!(
        "    {ctype} *d_{} = nullptr;\n\
         \x20   CUTILE_ACL(aclrtMalloc((void **)&d_{}, {size} * sizeof({ctype}), \
         ACL_MEM_MALLOC_HUGE_FIRST));\n",
        param.name, param.name
    ));
    if !param.mutable {
        text.push_str(&format!(
            "    CUTILE_ACL(aclrtMemcpy(d_{}, {size} * sizeof({ctype}), h_{}, \
             {size} * sizeof({ctype}), ACL_MEMCPY_HOST_TO_DEVICE));\n",
            param.name, param.name
        ));
    }
    Ok(text)
}

/// 派发实参：缓冲传 `(GM_ADDR)d_<name>`，标量取样本值——与设备入口的形参表同序。
fn ascend_launch_args(signature: &Signature, sample: &Sample) -> Result<String> {
    signature
        .params()
        .iter()
        .map(|param| {
            if param.ctype.is_buf() {
                Ok(format!("(GM_ADDR)d_{}", param.name))
            } else {
                ascend_argument(param, sample)
            }
        })
        .collect::<Result<Vec<_>>>()
        .map(|args| args.join(", "))
}

// ---------------------------------------------------------------------------
// 声明 → 并行程序：把 cutile 推出的 `Launch` 装配成「设备源 + 按网格发射的 host 驱动」
// ---------------------------------------------------------------------------

use prof::decl::Grid;

#[cfg(test)]
#[path = "../../testkit/tests/kernel.rs"]
mod cutile_x_kernel;

fn fold_of(func: &str) -> &'static str {
    if func.contains("max") { "max" } else { "sum" }
}

/// 「这台后端没有这个形态的并行程序」：调用方按它**显式回落**到语义门，不把缺实现说成已验证。
fn unsupported(vendor: &str, kind: &Lower) -> Error {
    Error::new(
        Code::Unsupported,
        format!("{vendor} 没有 `{kind:?}` 的并行程序"),
    )
    .with("engine::backends")
}

#[cfg(test)]
#[path = "../tests/program.rs"]
mod program_tests;

/// CUDA 并行程序：按形态装配设备源与 host 驱动，整份可直接 `mcc` / `nvcc` 编译执行。
///
/// # Errors
///
/// 该形态在 CUDA 上没有并行实现（tile / flash / WMMA 仍走各自的生产态入口），或样本与签名不一致。
fn cuda(kernel: &str, signature: &Signature, sample: &Sample, launch: &Launch) -> Result<String> {
    match &launch.kind {
        Lower::Pointwise | Lower::Loop(_) => cuda_single(kernel, signature, sample, launch),
        Lower::ChunkReduce { chunk, func } => {
            cuda_reduce(kernel, signature, sample, launch, *chunk, func)
        }
        _ => cuda_segments(kernel, signature, sample, launch),
    }
}

/// 单源族：一份设备源 + 一份主机驱动，两族的差别只在几何怎么折算。
fn cuda_single(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<String> {
    let Lower::Loop(grid) = &launch.kind else {
        let device = crate::cuda::cuda_grid_kernel(kernel, signature, sample)?;
        // CUDA 的网格是**块数**：工作项按线程组大小折算，块内再放 `group` 个线程。
        let blocks = launch.items[0].div_ceil(launch.group);
        return crate::cuda::cuda_point_driver(&device, signature, sample, blocks, launch.group);
    };
    // 零维网格就是串行：语义门那一份单线程 harness 已是它的并行程序（1 个线程）。
    if grid.dims.is_empty() {
        return crate::cuda::cuda_harness(kernel, signature, sample);
    }
    let device = crate::cuda::cuda_loop_kernel(kernel, signature, sample, grid)?;
    crate::cuda::cuda_loop_driver(&device, signature, sample, &launch.items[..grid.dims.len()])
}

/// 归约族：单段设备源 + 主机折叠，折叠算子在 `fold_of` 一处派生。
fn cuda_reduce(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
    chunk: usize,
    func: &str,
) -> Result<String> {
    let device = crate::cuda::cuda_chunk_reduce(kernel, signature, sample, chunk, func)?;
    crate::cuda::cuda_chunk_reduce_driver(
        &device,
        signature,
        sample,
        launch.items[0],
        fold_of(func),
    )
}

/// 多段族：段数与主机折叠是一体的，所以每臂只调各自模块的合成入口，这里不再拼第二份段表。
fn cuda_segments(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<String> {
    match &launch.kind {
        Lower::ChunkRmsnorm { chunk } => {
            crate::cuda::cuda_rmsnorm(kernel, signature, sample, *chunk)
        }
        Lower::ChunkSoftmax { chunk } => {
            crate::cuda::cuda_softmax(kernel, signature, sample, *chunk)
        }
        Lower::ChunkScan { chunk, decay } => {
            crate::cuda::cuda_scan(kernel, signature, sample, *chunk, *decay)
        }
        other => Err(unsupported("CUDA", other)),
    }
}

/// Metal 并行程序：单源族（逐点 / 循环注入）各有一份 `kernel_main`，分块族是多段源 + 主机折叠。
///
/// 分块族的段数与折叠是一体的，因此每臂只调 [`crate::metal`] 里各自的合成入口，
/// 与 CUDA 侧的 `cuda_segments` 分法一致：这层只管「什么形态配哪份程序」。
///
/// # Errors
///
/// 该形态没有 Metal 并行实现（调用方据此回落到语义门），或样本与签名不一致。
fn metal(kernel: &str, signature: &Signature, sample: &Sample, launch: &Launch) -> Result<Source> {
    match metal_segments(kernel, signature, sample, launch) {
        Some(source) => source,
        None => metal_dispatch(kernel, signature, sample, launch),
    }
}

/// 分块族的四份合成入口；不属分块族时 `None`（交给单源族）。
fn metal_segments(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Option<Result<Source>> {
    let kind = &launch.kind;
    if let Lower::ChunkReduce { chunk, func } = kind {
        return Some(crate::metal::reduce(
            kernel, signature, sample, *chunk, func,
        ));
    }
    if let Lower::ChunkRmsnorm { chunk } = kind {
        return Some(crate::metal::rmsnorm(kernel, signature, sample, *chunk));
    }
    if let Lower::ChunkSoftmax { chunk } = kind {
        return Some(crate::metal::softmax(kernel, signature, sample, *chunk));
    }
    if let Lower::ChunkScan { chunk, decay } = kind {
        return Some(crate::metal::scan(
            kernel, signature, sample, *chunk, *decay,
        ));
    }
    None
}

/// 单源族：设备源 + 按网格发射的驱动。
fn metal_dispatch(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<Source> {
    let (device, block) = metal_device(kernel, signature, sample, launch)?;
    let host =
        crate::metal::metal_dispatch_driver(signature, sample, "kernel_main", launch.items, block)?;
    Ok(Source::Pair { device, host })
}

/// 设备源与线程组：`dispatchThreads` 收的是**总线程数**（不是块数），工作项直接当线程；
/// 循环注入是「一维一线程」，组内 1 个。逐点内核带长度守卫，末组多出的线程自己跳过。
fn metal_device(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    launch: &Launch,
) -> Result<(String, [usize; 3])> {
    match &launch.kind {
        Lower::Pointwise => metal_pointwise(kernel, signature, sample, launch.group),
        Lower::Loop(grid) => metal_loop(kernel, signature, sample, grid),
        // 其余形态（分块族 / tile / flash / 波内特化）不是单源 Metal 程序：报「没有实现」而不是
        // 拿去按逐点 lowering 试一遍——那样签名校验会先报 `Invalid`，把「缺实现」说成「签名错」，
        // 调用方据此无法回落。
        other => Err(unsupported("Metal", other)),
    }
}

/// 逐点网格：工作项就是元素，一个元素一个线程，线程组为声明里的 `group`。
fn metal_pointwise(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    group: usize,
) -> Result<(String, [usize; 3])> {
    let device = crate::metal::metal_grid_kernel(kernel, signature, sample)?;
    Ok((device, [group, 1, 1]))
}

/// 循环注入：网格各维即线程形状（一维一线程），组内 1 个。
fn metal_loop(
    kernel: &str,
    signature: &Signature,
    sample: &Sample,
    grid: &Grid,
) -> Result<(String, [usize; 3])> {
    if grid.dims.is_empty() {
        return Err(Error::new(
            Code::Unsupported,
            "Metal 没有零维网格的并行程序".to_string(),
        )
        .with("engine::backends"));
    }
    let device = crate::metal::metal_loop_kernel(kernel, signature, sample, grid)?;
    Ok((device, [1, 1, 1]))
}

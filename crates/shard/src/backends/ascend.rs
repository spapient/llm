//! 昇腾（AscendC）后端：单源方言，`.cpp` 里同时放核函数与 AscendCL host 驱动，`ccec` 拆两段。

use crate::backend::{Backend, Codegen, Driver, Readback, Source};
use crate::backends::common::{device_fn, witness_head, witness_readback};
use crate::driver;
use crate::platform::{ascend_device, detect_cce};
use crate::toolchain::Toolchain;
use crate::witness::{MAX_OUT, Scope, Selected};
use cutile::dirs::Dirs;
use diagnostic::{Code, Error, Result};
use prof::prof::Emit;
use prof::prof::Profile;
use std::path::Path;
use std::process::Command;

/// AscendCL 的调用守卫：失败时报出是哪个调用并返回非零。
const ACL_GUARD: &str = "static int cutile_acl(const char *what, aclError code) {\n\
     \x20   if (code == ACL_SUCCESS) return 0;\n\
     \x20   fprintf(stderr, \"acl %s: %d\\n\", what, (int)code);\n\
     \x20   return -1;\n\
     }\n\
     #define CUTILE_ACL(call) do { if (cutile_acl(#call, (call)) != 0) return 1; } while (0)\n";

/// 昇腾的可用性：编译器与设备都在（`ccec` 认得出设备，但没有 NPU 也发射不了）。
pub(crate) fn ascend_ready() -> bool {
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
    let flags = crate::compile_flags(profile, dirs)?;
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

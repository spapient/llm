//! 见证设备：一台统一的执行器，按 [`Emit`] 在描述符表里分发到对应编排行。
//!
//! 片段本体来自 `prim`、能力档案来自 `prof`，本层只消费：把见证源码拼进模板、编译、起子进程、
//! 回读对拍。各语言的差异全部是表内函数指针行（探测 / 程序生成 / 执行回读），没有按厂商各写一套
//! Rust 前端类型；新增设备只在 [`SPECS`] 加一行，不改任何 `match`。
//!
//! 共享的装载、命名、起进程与回读各只有一份；编译器探测在 [`crate::toolchain`]，编译/执行原语在
//! [`crate::runner`]。CUDA 编排在 [`crate::device_cuda`]，Metal 编排在 [`crate::device_metal`]。

pub use crate::device_cuda::compile_run_cuda;

use crate::atom::{Prim, Witness};
use crate::build;
use crate::device_cuda::{cuda_available, cuda_program, cuda_run};
use crate::device_metal::{metal_available, metal_program, metal_run};
use crate::dirs::Dirs;
use crate::prof::{Emit, Profile};
use crate::toolchain;
use crate::witness::{MAX_OUT, Scope, Selected};
use diagnostic::{Code, Error, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 见证函数在 C 侧（C 与 CUDA 前端）的形参文本。
pub(crate) const C_PARAMS: &str = "float *";

/// 本次要拼进程序的见证源码：文件名 → 文本。
pub(crate) type Sources = BTreeMap<String, String>;

/// 一次执行回读出来的东西：每条见证写出的个数 + 按顺序回读的 f32 值。
pub(crate) type Readback = (Vec<i64>, Vec<f32>);

/// 一台设备的执行描述：差异全是表内的函数指针，运行期只多一次“按 emit 找行”，没有枚举匹配、没有虚表。
pub(crate) struct Spec {
    /// 这门语言。
    pub(crate) emit: Emit,
    /// 设备是否在本机。
    pub(crate) available: fn(&Profile) -> bool,
    /// 生成“跑一次全部见证”的程序文本；写出上限定在 ABI 常量 [`MAX_OUT`]。
    pub(crate) program: fn(&Profile, &Dirs, &[Selected<'_>], Scope) -> Result<String>,
    /// 让程序跑起来，返回 `(每条写入个数, 按顺序回读的值)`。
    pub(crate) run: fn(&Profile, &Dirs, &str, usize, Scope) -> Result<Readback>,
}

/// 全部设备：**加设备就加一行**，不需要改任何 `match`。
const SPECS: &[Spec] = &[
    Spec {
        emit: Emit::C,
        available: c_available,
        program: c_program,
        run: c_run,
    },
    Spec {
        emit: Emit::Metal,
        available: metal_available,
        program: metal_program,
        run: metal_run,
    },
    Spec {
        emit: Emit::Cuda,
        available: cuda_available,
        program: cuda_program,
        run: cuda_run,
    },
];

/// 按 `emit` 找设备描述；找不到就是“本机没有这门语言的执行前端”。
pub(crate) fn spec(emit: Emit) -> Option<&'static Spec> {
    SPECS.iter().find(|entry| entry.emit == emit)
}

/// 已登记的全部执行语言：错误报告里列出“本机接了哪些”。
pub(crate) fn specs() -> &'static [Spec] {
    SPECS
}

// ---------------------------------------------------------------------------
// 共享装载、命名、起进程与回读
// ---------------------------------------------------------------------------

/// 一份见证源文件：读出来，并确认里面有这个前端能调用的函数（签名由编译器回答）。
/// 错误里带原语名，是因为同一个文件被多条见证共用时，光有文件名定位不了是哪一条。
fn witness_source(prim: &Prim, witness: &Witness, dirs: &Dirs, params: &str) -> Result<String> {
    let text = crate::witness::source(witness, dirs)
        .map_err(|error| error.with(format!("`{prim}` 的")))?;
    if !text.contains(&format!("{}(", witness.func)) {
        let message = format!(
            "`{prim}` 的见证 `{}` 里没有函数 `{}({params})`",
            witness.file, witness.func
        );
        return Err(Error::new(Code::Invalid, message));
    }
    Ok(text)
}

/// 本次执行的见证源码：按文件去重，同一文件里的多条见证只出现一次。
pub(crate) fn witness_files(
    selected: &[Selected<'_>],
    dirs: &Dirs,
    params: &str,
) -> Result<Sources> {
    let mut files = Sources::new();
    for (prim, _, witness) in selected {
        if files.contains_key(&witness.file) {
            continue;
        }
        files.insert(
            witness.file.clone(),
            witness_source(prim, witness, dirs, params)?,
        );
    }
    Ok(files)
}

/// 源码段接在头之后：每段补齐换行再空一行，段与段之间因此总有分隔。
pub(crate) fn push_sources(program: &mut String, files: &Sources) {
    for text in files.values() {
        program.push_str(text);
        if !program.ends_with('\n') {
            program.push('\n');
        }
        program.push('\n');
    }
}

/// 让一段文本以换行结尾（空文本保持为空）：下一段可以直接接上。
pub(crate) fn normalize_newline(text: &mut String) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
}

/// 写一个文件，并把它的上层目录建出来：一次性构建目录都只有一层。
pub(crate) fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            Error::new(Code::Io, format!("创建 {} 失败：{error}", parent.display()))
        })?;
    }
    std::fs::write(path, text)
        .map_err(|error| Error::new(Code::Io, format!("写 {} 失败：{error}", path.display())))
}

/// 跑一条编译器命令；非零退出码时把命令、完整输出与源文件路径一并带进诊断。
pub(crate) fn run_tool(name: &str, command: &mut Command, source: &Path) -> Result<()> {
    let output = command
        .output()
        .map_err(|error| Error::new(Code::Io, format!("调用 `{name}` 失败：{error}")))?;
    if output.status.success() {
        return Ok(());
    }
    Err(Error::new(
        Code::Invalid,
        format!(
            "`{name}` 执行失败：\n{}\n{}\n源文件：{}",
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim(),
            source.display()
        ),
    ))
}

/// 跑一个已经编好的可执行文件，返回 `(stdout, stderr)` 原文。非零退出码报 [`Code::Unsupported`]：编得过却跑不起来说明这台机器现在跑不了，不是见证写错了。
pub(crate) fn run_exe(
    command: &mut Command,
    path: &Path,
    failure: &str,
) -> Result<(String, String)> {
    let output = command
        .output()
        .map_err(|error| Error::new(Code::Io, format!("运行 {} 失败：{error}", path.display())))?;
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "{failure}（退出码 {:?}）：{}",
                output.status.code(),
                stderr.trim()
            ),
        ));
    }
    Ok((String::from_utf8_lossy(&output.stdout).to_string(), stderr))
}

/// 从 stderr 读出每条见证声明的写入个数（按调用顺序）。
pub(crate) fn parse_counts(stderr: &str) -> Vec<i64> {
    stderr
        .lines()
        .filter_map(|line| line.trim().strip_prefix("witness "))
        .filter_map(|value| value.trim().parse::<i64>().ok())
        .collect()
}

/// 回读解码：个数走 stderr（`witness N`），值走 stdout 的 f32 位模式；两条通道必须分开，
/// 否则 [`crate::runner::parse_bits`] 会把计数当成值解出来。
pub(crate) fn readback(stdout: &str, stderr: &str) -> Readback {
    (parse_counts(stderr), crate::runner::parse_bits(stdout))
}

/// 一个新的一次性目录：`<work_dir>/<tag>-<pid>-<序号>`，调用方负责删除。pid 隔开进程、序号
/// 隔开同进程内的多次调用：同时跑的两个 `cutile` 不会互相覆盖对方的中间产物。
pub(crate) fn scratch(dirs: &Dirs, tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    dirs.work().join(format!(
        "{tag}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

/// 两个后端 kernel 共享的分派段：按 `tid` 的 `switch`、cap 截断与回写循环。`close` 是 kernel
/// 收尾文本（Metal 与 CUDA 的换行数不同）：`local` 声明留在各自函数里，这里只拼逐字相同的部分。
pub(crate) fn push_dispatch(
    program: &mut String,
    max: usize,
    selected: &[Selected<'_>],
    close: &str,
) {
    for (index, (_, _, witness)) in selected.iter().enumerate() {
        program.push_str(&format!(
            "        case {index}: n = {}(local); break;\n",
            witness.func
        ));
    }
    program.push_str(&format!(
        "        default: break;\n\
         \x20   }}\n\
         \x20   int cap = n < 0 ? 0 : (n > {max} ? {max} : n);\n\
         \x20   counts[tid] = cap;\n\
         \x20   for (int i = 0; i < {max}; ++i) out[tid * {max} + i] = i < cap ? local[i] : 0.0f;\n\
         {close}"
    ));
}

// ---------------------------------------------------------------------------
// C（host CPU）编排：见证本身就是一段 host C，`cc` 把它与 harness 编成可执行文件跑一遍。
// 它证明“这段 C 在这个 ABI 下编得过、跑得出 `expect`”，不证明设备具备该能力。
// ---------------------------------------------------------------------------

/// 自检用的 host C 参数：设备档案的 `[compile].flags` 在本机要么指向不存在的头，要么只对真机编译器有意义；它与设备执行是两个参数集。
pub(crate) const HOST_FLAGS: [&str; 2] = ["-O2", "-std=c99"];

/// C 编排的编译输入：前置声明 + 参数。自检只走 host C ABI，理由见 [`HOST_FLAGS`]。
fn c_inputs(profile: &Profile, scope: Scope) -> (String, Vec<String>) {
    match scope {
        Scope::Device => (profile.preamble.clone(), profile.flags.clone()),
        Scope::Host => (
            String::new(),
            HOST_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        ),
    }
}

fn c_available(_: &Profile) -> bool {
    // `c` 的设备就是本机 CPU：它永远在。
    true
}

fn c_program(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
    scope: Scope,
) -> Result<String> {
    let mut program = c_head(profile, scope, selected);
    push_sources(&mut program, &witness_files(selected, dirs, C_PARAMS)?);
    program.push_str(&c_harness::<MAX_OUT>(selected));
    Ok(program)
}

fn c_run(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    _count: usize,
    scope: Scope,
) -> Result<Readback> {
    let (_, flags) = c_inputs(profile, scope);
    // 设备执行与自检是两个不同的缓存命名空间：同一段文本在两套参数下编出的产物不同。
    let tag = match scope {
        Scope::Device => "witness",
        Scope::Host => "self",
    };
    // 见证程序也进产物缓存：第二次 `verify` 只在第一次调用编译器。
    let (outcome, _artifact) =
        build::run_program(toolchain::detect().as_ref(), &flags, tag, program, dirs)?;
    // stdout 已由 `runner` 解码成 f32，这里只补上 stderr 里的写入个数。
    Ok((parse_counts(&outcome.stderr), outcome.bits))
}

/// C 前端的头：档案前置声明 + `stdio.h`；自检时再吞掉 MSL 的地址空间限定符。`#define thread` 只服务
/// 于自检，设备前端走真正的 `xcrun metal`，所以这里的放宽不会被当成设备证据。
fn c_head(profile: &Profile, scope: Scope, selected: &[Selected<'_>]) -> String {
    let (mut head, _) = c_inputs(profile, scope);
    let metal = selected
        .iter()
        .any(|(_, _, witness)| witness.file.ends_with(".metal"));
    normalize_newline(&mut head);
    head.push_str("#include <stdio.h>\n\n");
    if scope == Scope::Host && metal {
        head.push_str("#define thread\n");
    }
    head
}

/// C 前端的内核包装与主函数：`main()` 逐条调用见证函数，值走 stdout，个数走 stderr。
fn c_harness<const OUT: usize>(selected: &[Selected<'_>]) -> String {
    let mut text = format!("\nint main(void) {{\n    float out[{OUT}];\n    int n;\n");
    // 每条见证是一句互不相同的调用语句，顺序由见证清单决定：这里只能逐条生成，不能批处理。
    for (_, _, witness) in selected {
        push_witness(&mut text, &witness.func, OUT);
    }
    text.push_str("    return 0;\n}\n");
    text
}

/// 一次见证调用：算一遍、报个数、按写出的个数打印位模式；`max` 是 const 泛型传下来的字面量，截断在生成的代码里是常数比较。
fn push_witness(text: &mut String, func: &str, max: usize) {
    text.push_str("    n = ");
    text.push_str(func);
    text.push_str("(out);\n");
    text.push_str("    fprintf(stderr, \"witness %d\\n\", n);\n");
    text.push_str("    if (n < 0) n = 0;\n");
    text.push_str(&format!("    if (n > {max}) n = {max};\n"));
    text.push_str("    for (int i = 0; i < n; ++i) {\n");
    text.push_str("        union { float f; unsigned int u; } v;\n");
    text.push_str("        v.f = out[i];\n");
    text.push_str("        printf(\"%u\\n\", v.u);\n");
    text.push_str("    }\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin::Catalogue;

    /// 生成各前端的见证源码。默认只断言生成非空；`CUTILE_CAPTURE_GOLDEN` 显式打开时才把副本
    /// 写到 `/tmp/cutile-golden` 供人工比对，不常驻产物。
    #[test]
    fn golden_capture() {
        let capture = std::env::var("CUTILE_CAPTURE_GOLDEN").is_ok();
        let dir = std::path::PathBuf::from("/tmp/cutile-golden");
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-golden-work"));
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        for (vendor, file) in [
            ("cpu", "witness_c.txt"),
            ("metal", "witness_metal.txt"),
            ("nvidia", "witness_cuda.txt"),
        ] {
            let Some(profile) = catalogue.profile(vendor).cloned() else {
                continue;
            };
            let Some(front) = spec(profile.meta.emit) else {
                continue;
            };
            let selected = crate::witness::declared_witnesses(&profile).expect("见证");
            if selected.is_empty() {
                continue;
            }
            let text = (front.program)(&profile, &dirs, &selected, Scope::Device).expect("可生成");
            assert!(!text.is_empty(), "{vendor} 生成文本不能为空");
            if capture {
                std::fs::create_dir_all(&dir).expect("目录");
                std::fs::write(dir.join(file), text).expect("写");
            }
        }
        let metal = catalogue.profile("metal").cloned().expect("档案");
        let selected = crate::witness::declared_witnesses(&metal).expect("见证");
        let c_front = spec(Emit::C).expect("c 前端");
        let text = (c_front.program)(&metal, &dirs, &selected, Scope::Host).expect("可生成");
        assert!(!text.is_empty(), "metal 档案的 host 自检文本不能为空");
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join("witness_metal_host.txt"), text).expect("写");
        }
    }

    #[test]
    fn scratch_directories_do_not_collide() {
        let dirs = Dirs::under(std::env::temp_dir());
        let first = scratch(&dirs, "gate");
        let second = scratch(&dirs, "gate");
        assert_ne!(first, second);
        assert_eq!(first.parent(), Some(dirs.work()));
    }

    /// 执行前端登记表：每个 `emit` 恰好一行，且表不许被清空。
    ///
    /// 旧的「trait 实现 + 语言注解 + 装配表」三位一体契约已由描述符表取代：语言身份就是
    /// [`Spec::emit`]（`Emit` 闭集保证取值合法），三指针的接线由类型系统保证。这里只守两类
    /// 文本/表结构事故：同一语言重复登记（`spec(..)` 命中第一行，后一行静默失效），以及表被
    /// 清空/拆坏后测试误判为"没有违规"。
    #[test]
    fn specs_register_each_execution_frontend_once() {
        let mut seen = std::collections::BTreeSet::new();
        for entry in SPECS {
            assert!(
                seen.insert(entry.emit),
                "执行前端 `{}` 在 SPECS 里重复登记：后一行会被 `spec(..)` 静默忽略",
                entry.emit.as_str()
            );
            assert!(
                spec(entry.emit).is_some(),
                "表内 `{}` 的查找行与登记表分家",
                entry.emit.as_str()
            );
        }
        assert!(
            SPECS.len() >= 3,
            "执行前端登记表行数 {} 低于基线 3：扫描布局或表被拆坏",
            SPECS.len()
        );
    }
}

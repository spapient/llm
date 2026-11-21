//! 见证源码的宿主执行：把设备 crate 的 `witness.<ext>` 编成可执行文件，逐条调用
//! `int func(float *out)` 并与档案里的 `witness.expect` 对拍。

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

/// 见证 ABI 的写出上限（与 `prof::atom::Witness::MAX_OUT` 同一个数）。
pub const MAX_OUT: usize = 8;

/// 一条待执行的见证：入口名与按顺序读回的期望值。
#[derive(Debug, Clone, Copy)]
pub struct Call<'a> {
    /// 入口函数名（`witness_*`）。
    pub func: &'a str,
    /// 期望输出，长度必须等于函数的返回值。
    pub expect: &'a [f32],
}

/// 一条见证的读数：个数与值分两路，与 `cutile::witness` 的回读口径一致。
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    /// 入口函数名。
    pub func: String,
    /// 函数返回的元素个数（即它写进 `out` 的个数）。
    pub count: i32,
    /// 读回的值，按写入顺序。
    pub values: Vec<f32>,
    /// 与期望值的最大相对误差；个数不符时是 [`f64::INFINITY`]。
    pub max_rel_err: f64,
}

/// 编译并执行一份见证源码，按 `calls` 的顺序逐条回读。
///
/// `ext` 决定语言：`cpp` 走 C++，其余（`c` / `cl`）走 C。编译器取 `CC` / `CXX`，缺省是
/// `cc` / `c++`；两者都不在 `PATH` 上时返回 `Err`，由调用方登记跳过而不是静默通过。
///
/// 源码写进临时目录时一律叫 `witness.h`：驱动文件的扩展名才是宿主编译器认的语言依据，
/// 而 `.cl` 会被 clang 直接当成 OpenCL（于是拒绝 `-std=c99`、也链接不出可执行文件）。
/// 见证正文因此按扩展名写、按宿主语言编，两者分离；设备前端是否接受原名那份文件，由
/// `cutile::witness` 的语言门单独回答。
///
/// # Errors
///
/// 临时目录不可写、编译器缺席、编译失败、程序非零退出、输出行数不足时返回可读的原因。
pub fn run(source: &str, ext: &str, calls: &[Call<'_>]) -> Result<Vec<Reading>, String> {
    let dir = scratch(&format!("{ext}-{}", main_ext(ext)));
    let _guard = Guard(dir.clone());
    let main = layout(&dir, ext, source, calls)?;
    let binary = dir.join("witness");
    compile(ext, &main, &binary)?;
    let stdout = execute(&binary)?;
    read(&stdout, calls)
}

/// 驱动文件的扩展名：C++ 见证要按 C++ 编，其余按 C 编。
fn main_ext(ext: &str) -> &'static str {
    if ext == "cpp" { "cpp" } else { "c" }
}

/// 把见证源码与驱动程序写进临时目录，返回驱动程序的路径。
fn layout(
    dir: &std::path::Path,
    ext: &str,
    source: &str,
    calls: &[Call<'_>],
) -> Result<PathBuf, String> {
    let main = dir.join(format!("main.{}", main_ext(ext)));
    write(&dir.join("witness.h"), source)?;
    write(&main, &program(calls))?;
    Ok(main)
}

/// 写文件；失败时带上路径。
fn write(path: &std::path::Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|error| format!("`{}` 不可写：{error}", path.display()))
}

/// 编成可执行文件；编译器取 `CC` / `CXX`，缺省是 `cc` / `c++`。
fn compile(ext: &str, main: &std::path::Path, binary: &std::path::Path) -> Result<(), String> {
    let cpp = ext == "cpp";
    let compiler = compiler(cpp);
    let standard = standard(cpp);
    let compiled = Command::new(&compiler)
        .arg("-O1")
        .arg(standard)
        .arg("-o")
        .arg(binary)
        .arg(main)
        .output()
        .map_err(|error| format!("编译器 `{}` 不可执行：{error}", compiler.to_string_lossy()))?;
    if compiled.status.success() {
        return Ok(());
    }
    Err(format!(
        "`{} {standard}` 拒绝了见证源码：{}",
        compiler.to_string_lossy(),
        String::from_utf8_lossy(&compiled.stderr).trim()
    ))
}

/// 主机上该语言的编译器：`CXX`/`CC` 优先，缺席时用 `c++`/`cc`。
fn compiler(cpp: bool) -> std::ffi::OsString {
    match cpp {
        true => std::env::var_os("CXX"),
        false => std::env::var_os("CC"),
    }
    .unwrap_or_else(|| default_compiler(cpp).into())
}

/// 该语言的方言旗标：C++17 与 C99。
fn standard(cpp: bool) -> &'static str {
    match cpp {
        true => "-std=c++17",
        false => "-std=c99",
    }
}

/// 没有环境变量时的编译器名。
fn default_compiler(cpp: bool) -> &'static str {
    match cpp {
        true => "c++",
        false => "cc",
    }
}

/// 跑一遍编出来的程序，返回它的标准输出。
fn execute(binary: &std::path::Path) -> Result<String, String> {
    let executed = Command::new(binary)
        .output()
        .map_err(|error| format!("见证程序不可执行 `{}`：{error}", binary.display()))?;
    if !executed.status.success() {
        return Err(format!(
            "见证程序非零退出：{}",
            String::from_utf8_lossy(&executed.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&executed.stdout).into_owned())
}

/// 生成的程序：把见证源码原样 `#include` 进来，于是辅助函数与 `static` 都保持同一份。
fn program(calls: &[Call<'_>]) -> String {
    let mut text = String::from("/* 由 testkit::witness 生成：一条见证一行读数。 */\n");
    let _ = writeln!(text, "#define MAX_OUT {MAX_OUT}");
    text.push_str("#include <stdio.h>\n");
    let _ = writeln!(text, "#include \"witness.h\"");
    text.push_str("\nint main(void) {\n float out[MAX_OUT];\n int n;\n");
    for call in calls {
        let _ = writeln!(text, " n = {}(out);", call.func);
        text.push_str(" printf(\"%d\", n);\n");
        text.push_str(
            " for (int i = 0; i < n && i < MAX_OUT; ++i) printf(\" %.9g\", (double)out[i]);\n",
        );
        text.push_str(" printf(\"\\n\");\n");
    }
    text.push_str(" return 0;\n}\n");
    text
}

/// 逐行解析：首列是个数，其余是值；行数由 `calls` 决定，少一行就是见证没被调用。
fn read(stdout: &str, calls: &[Call<'_>]) -> Result<Vec<Reading>, String> {
    let mut lines = stdout.lines();
    calls
        .iter()
        .map(|call| {
            let line = lines
                .next()
                .ok_or_else(|| format!("`{}` 没有回报读数：程序输出比见证少", call.func))?;
            reading(call, line)
        })
        .collect()
}

/// 一行读数：首列是个数，其余是值。
fn reading(call: &Call<'_>, line: &str) -> Result<Reading, String> {
    let mut fields = line.split_whitespace();
    let count: i32 = fields
        .next()
        .and_then(|field| field.parse().ok())
        .ok_or_else(|| format!("`{}` 的读数首列不是个数：`{line}`", call.func))?;
    let values: Vec<f32> = fields.filter_map(|field| field.parse().ok()).collect();
    let max_rel_err = deviation(&values, call.expect, count);
    Ok(Reading {
        func: call.func.to_string(),
        count,
        values,
        max_rel_err,
    })
}

/// 与期望值的最大相对误差：个数不符时是无穷大（"少写一个元素"与"值错了"必须可区分）。
fn deviation(values: &[f32], expect: &[f32], count: i32) -> f64 {
    let matches = usize::try_from(count).is_ok_and(|count| count == expect.len());
    if matches && values.len() == expect.len() {
        crate::rel::max_rel_err(values, expect)
    } else {
        f64::INFINITY
    }
}

/// 一次执行的临时目录；`Drop` 时整棵删掉，编译失败也不留垃圾。
struct Guard(PathBuf);

impl Drop for Guard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(tag: &str) -> PathBuf {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    crate::scratch::scratch_dir(&format!(
        "witness-{tag}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

#[cfg(test)]
#[path = "../tests/witness_tests.rs"]
mod witness_tests;

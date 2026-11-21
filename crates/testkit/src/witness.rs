//! 见证源码的宿主执行：把设备 crate 的 `witness.<ext>` 编成可执行文件，逐条调用
//! `int func(float *out)` 并与档案里的 `witness.expect` 对拍。
//!
//! 只回答"这份见证写的算法在本机算得对"：它不是设备证据，也不替代语言门——真机执行与
//! `ccec` / `clang -x cl` 的语法门由 `cutile::witness` 驱动。夹具刻意不依赖 `prof`：
//! 调用方把 `(入口名, 期望值)` 传进来，`testkit` 因此仍然只有 `std` 一个依赖。
//!
//! 只接受按宿主 ABI 写成的见证（`emit = c` / `opencl` / `ascendc` 三类的 `witness.*`）。
//! `__global__` 一类的设备扩展不在这里编译——那是设备执行器的职责。

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
    let cpp = ext == "cpp";
    let main_ext = if cpp { "cpp" } else { "c" };
    let dir = scratch(&format!("{ext}-{main_ext}"));
    let _guard = Guard(dir.clone());
    let witness = dir.join("witness.h");
    let main = dir.join(format!("main.{main_ext}"));
    std::fs::write(&witness, source)
        .map_err(|error| format!("见证源码不可写 `{}`：{error}", witness.display()))?;
    std::fs::write(&main, program(calls))
        .map_err(|error| format!("临时程序不可写 `{}`：{error}", main.display()))?;

    let binary = dir.join("witness");
    let compiler = if cpp {
        std::env::var_os("CXX").unwrap_or_else(|| "c++".into())
    } else {
        std::env::var_os("CC").unwrap_or_else(|| "cc".into())
    };
    let standard = if cpp { "-std=c++17" } else { "-std=c99" };
    let compiled = Command::new(&compiler)
        .arg("-O1")
        .arg(standard)
        .arg("-o")
        .arg(&binary)
        .arg(&main)
        .output()
        .map_err(|error| format!("编译器 `{}` 不可执行：{error}", compiler.to_string_lossy()))?;
    if !compiled.status.success() {
        return Err(format!(
            "`{} {standard}` 拒绝了见证源码：{}",
            compiler.to_string_lossy(),
            String::from_utf8_lossy(&compiled.stderr).trim()
        ));
    }

    let executed = Command::new(&binary)
        .output()
        .map_err(|error| format!("见证程序不可执行 `{}`：{error}", binary.display()))?;
    if !executed.status.success() {
        return Err(format!(
            "见证程序非零退出：{}",
            String::from_utf8_lossy(&executed.stderr).trim()
        ));
    }
    read(&String::from_utf8_lossy(&executed.stdout), calls)
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
    let mut readings = Vec::with_capacity(calls.len());
    for call in calls {
        let line = lines
            .next()
            .ok_or_else(|| format!("`{}` 没有回报读数：程序输出比见证少", call.func))?;
        let mut fields = line.split_whitespace();
        let count: i32 = fields
            .next()
            .and_then(|field| field.parse().ok())
            .ok_or_else(|| format!("`{}` 的读数首列不是个数：`{line}`", call.func))?;
        let values: Vec<f32> = fields.filter_map(|field| field.parse().ok()).collect();
        let matches = usize::try_from(count).is_ok_and(|count| count == call.expect.len());
        let max_rel_err = if matches && values.len() == call.expect.len() {
            crate::rel::max_rel_err(&values, call.expect)
        } else {
            f64::INFINITY
        };
        readings.push(Reading {
            func: call.func.to_string(),
            count,
            values,
            max_rel_err,
        });
    }
    Ok(readings)
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
mod tests {
    use super::*;

    /// 夹具自身：算得对的见证回读到同值，算错的一条被相对误差抓到。
    #[test]
    fn a_host_witness_is_compiled_and_compared() {
        let source = "int witness_ok(float *out) { out[0] = 2.0f; return 1; }\n\
                      int witness_off(float *out) { out[0] = 2.5f; return 1; }\n";
        let calls = [
            Call {
                func: "witness_ok",
                expect: &[2.0],
            },
            Call {
                func: "witness_off",
                expect: &[2.0],
            },
        ];
        let Ok(readings) = run(source, "c", &calls) else {
            assert!(crate::gating::skip("本机没有可用的 C 编译器"));
            return;
        };
        assert_eq!(readings[0].count, 1);
        assert_eq!(readings[0].values, [2.0]);
        assert!(readings[0].max_rel_err <= 1e-6, "{:?}", readings[0]);
        assert!(readings[1].max_rel_err > 1e-3, "{:?}", readings[1]);
    }

    /// 少写一个值必须按个数判失败，而不是拿前几个值凑过去。
    #[test]
    fn a_count_mismatch_is_not_a_near_miss() {
        let source = "int witness_short(float *out) { out[0] = 1.0f; return 1; }\n";
        let calls = [Call {
            func: "witness_short",
            expect: &[1.0, 1.0],
        }];
        let Ok(readings) = run(source, "c", &calls) else {
            assert!(crate::gating::skip("本机没有可用的 C 编译器"));
            return;
        };
        assert_eq!(readings[0].count, 1);
        assert!(readings[0].max_rel_err.is_infinite());
    }

    /// 编译器不认的源码是错误，不是"没跑到"。
    #[test]
    fn a_witness_the_host_rejects_is_an_error() {
        let source = "int witness_broken(float *out) { out[0] = ; return 1; }\n";
        let calls = [Call {
            func: "witness_broken",
            expect: &[0.0],
        }];
        let Err(error) = run(source, "c", &calls) else {
            panic!("语法错误的见证必须编译失败");
        };
        assert!(error.contains("拒绝了见证源码"), "{error}");
    }
}

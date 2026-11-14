//! C（host CPU）前端：见证本身就是一段 host C，`cc` 把它与 harness 编成可执行文件跑一遍。
//!
//! 它证明"这段 C 在这个 ABI 下编得过、跑得出 `expect`"，**不**证明设备具备该能力。
//! 身份由 `#[frontend]` 生成；共享装载、起进程与回读在 [`crate::device`]。

use crate::build;
use crate::dirs::Dirs;
use crate::prof::Profile;
use crate::toolchain;
use crate::witness::{Device, Language, Scope, Selected};
use diagnostic::Result;

/// 自检用的 host C 参数：设备档案的 `[compile].flags` 在本机要么指向不存在的头，要么只对真机编译器有意义；它与设备执行是两个参数集。
pub(crate) const HOST_FLAGS: [&str; 2] = ["-O2", "-std=c99"];

/// C 前端的编译输入：前置声明 + 参数。自检只走 host C ABI，理由见 [`HOST_FLAGS`]。
pub(crate) fn c_compile_inputs(profile: &Profile, scope: Scope) -> (String, Vec<String>) {
    match scope {
        Scope::Device => (profile.preamble.clone(), profile.flags.clone()),
        Scope::Host => (
            String::new(),
            HOST_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        ),
    }
}

/// C 前端：见证本身就是一段 host C，`cc` 把它与 harness 编成可执行文件跑一遍。它证明"这段 C 在这个
/// ABI 下编得过、跑得出 `expect`"，**不**证明设备具备该能力。身份由 `#[frontend]` 生成。
#[annotation::frontend(emit = "c", ext = "c", tag = "witness")]
pub struct CpuDevice;

impl Device for CpuDevice {
    fn available(_profile: &Profile) -> bool {
        // `c` 的设备就是本机 CPU：它永远在。
        true
    }

    fn program<const OUT: usize>(
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        let mut program = c_head(profile, scope, selected);
        crate::device::push_sources(
            &mut program,
            &crate::device::witness_files(selected, dirs, crate::device::C_PARAMS)?,
        );
        program.push_str(&harness::<OUT>(selected));
        Ok(program)
    }

    fn run(
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        _count: usize,
        scope: Scope,
    ) -> Result<crate::device::Readback> {
        let (_, flags) = c_compile_inputs(profile, scope);
        // 设备执行与自检是两个不同的缓存命名空间：同一段文本在两套参数下编出的产物不同。
        let tag = match scope {
            Scope::Device => Self::TAG,
            Scope::Host => "self",
        };
        // 见证程序也进产物缓存：第二次 `verify` 只在第一次调用编译器。
        let (outcome, _artifact) =
            build::run_program(toolchain::detect().as_ref(), &flags, tag, program, dirs)?;
        // stdout 已由 `runner` 解码成 f32，这里只补上 stderr 里的写入个数。
        Ok((crate::device::parse_counts(&outcome.stderr), outcome.bits))
    }
}

/// C 前端的头：档案前置声明 + `stdio.h`；自检时再吞掉 MSL 的地址空间限定符。`#define thread` 只服务
/// 于自检，设备前端走真正的 `xcrun metal`，所以这里的放宽不会被当成设备证据。
fn c_head(profile: &Profile, scope: Scope, selected: &[Selected<'_>]) -> String {
    let (mut head, _) = c_compile_inputs(profile, scope);
    let metal = selected
        .iter()
        .any(|(_, _, witness)| witness.file.ends_with(".metal"));
    crate::device::normalize_newline(&mut head);
    head.push_str("#include <stdio.h>\n\n");
    if scope == Scope::Host && metal {
        head.push_str("#define thread\n");
    }
    head
}

/// C 前端的内核包装与主函数：`main()` 逐条调用见证函数，值走 stdout，个数走 stderr。
pub(crate) fn harness<const OUT: usize>(selected: &[Selected<'_>]) -> String {
    let mut text = format!("\nint main(void) {{\n    float out[{OUT}];\n    int n;\n");
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

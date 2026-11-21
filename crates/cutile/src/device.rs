//! 宿主 C 参考后端 + 设备后端注册表。
//!
//! 本文件只保留「机制」：C 宿主参考后端（见证在 host CPU 上编过、跑出 `expect`，不证明设备能力）
//! 与一张按 `emit` 分派的注册表。Metal / CUDA 的设备后端实现已搬到 `engine`（L6），经
//! [`crate::backend::register`] 注入这张表；cutile 的机制层只认 [`crate::backend::Backend`] 缝，
//! 不出现任何厂商字面量（`docs/ARCHITECTURE.md` 的 L3 契约）。

use crate::backend::{Backend, Codegen, Driver, Readback};
use crate::build;
use crate::dirs::Dirs;
use crate::prof::{Emit, Profile};
use crate::runner::driver;
use crate::toolchain;
use crate::witness::{MAX_OUT, Scope, Selected};
use diagnostic::Result;
use std::sync::{Mutex, OnceLock};

/// 自检用的 host C 参数：设备档案的 `[compile].flags` 在本机要么指向不存在的头，要么只对真机编译器有意义；它与设备执行是两个参数集。
pub(crate) const HOST_FLAGS: [&str; 2] = ["-O2", "-std=c99"];

/// 宿主 C 参考后端：本机 CPU 永远在，证明「这段 C 编得过、跑得出 `expect`」，不是设备证据。
struct CBackend;

impl Codegen for CBackend {
    fn emit(&self) -> Emit {
        Emit::C
    }
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        c_program(profile, dirs, selected, scope)
    }
}

impl Driver for CBackend {
    fn available(&self, profile: &Profile) -> bool {
        c_available(profile)
    }
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        max: usize,
        scope: Scope,
    ) -> Result<Readback> {
        c_run(profile, dirs, program, max, scope)
    }
    fn toolchain(&self) -> Option<toolchain::Toolchain> {
        toolchain::detect()
    }
}

impl Backend for CBackend {}

/// 后端注册表：cutile 内置 host C 参考后端，Metal / CUDA 由 `engine` 经 [`crate::backend::register`]
/// 注入。进程级一张表，机制层按 `emit` 取，不硬编码后端清单。
static REGISTRY: OnceLock<Mutex<Vec<&'static dyn Backend>>> = OnceLock::new();

fn registry() -> &'static Mutex<Vec<&'static dyn Backend>> {
    REGISTRY.get_or_init(|| Mutex::new(vec![&CBackend]))
}

/// 注册一台后端；同一条后端重复注册只留一份（幂等）。注册表锁若已中毒，本次注册静默放弃——
/// 后端注入发生在启动期，宁可少一台也不 panic。
pub(crate) fn register(backend: &'static dyn Backend) {
    let Ok(mut table) = registry().lock() else {
        return;
    };
    if !table.iter().any(|entry| entry.emit() == backend.emit()) {
        table.push(backend);
    }
}

/// 按 `emit` 找设备后端；找不到就是「本机没有这门语言的执行前端」。
pub(crate) fn spec(emit: Emit) -> Option<&'static dyn Backend> {
    let Ok(table) = registry().lock() else {
        return None;
    };
    table.iter().find(|entry| entry.emit() == emit).copied()
}

/// 已登记的全部执行语言：错误报告里列出「本机接了哪些」。
pub(crate) fn specs() -> Vec<&'static dyn Backend> {
    let Ok(table) = registry().lock() else {
        return Vec::new();
    };
    table.clone()
}

// ---------------------------------------------------------------------------
// C（host CPU）编排：见证本身就是一段 host C，`cc` 把它与 harness 编成可执行文件跑一遍。
// 它证明「这段 C 在这个 ABI 下编得过、跑得出 `expect`」，不证明设备具备该能力。
// ---------------------------------------------------------------------------

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
    driver::push_sources(
        &mut program,
        &driver::witness_files(selected, dirs, driver::C_PARAMS)?,
    );
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
    Ok((driver::parse_counts(&outcome.stderr), outcome.bits))
}

/// C 前端的头：档案前置声明 + `stdio.h`；自检时再吞掉 MSL 的地址空间限定符。`#define thread` 只服务
/// 于自检，设备前端走真正的 `xcrun metal`，所以这里的放宽不会被当成设备证据。
fn c_head(profile: &Profile, scope: Scope, selected: &[Selected<'_>]) -> String {
    let (mut head, _) = c_inputs(profile, scope);
    let metal = selected
        .iter()
        .any(|(_, _, witness)| witness.file.ends_with(".metal"));
    driver::normalize_newline(&mut head);
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

    /// 一条 mock 后端（SPIRV，本机必然缺席）：只用来验注册表的「去重 + 按 emit 分派」机制，不碰任何
    /// 厂商工具链。真机 Metal / CUDA 后端在 `engine`（L6），其 Codegen / Driver 测试也搬到了那边。
    struct Mock;

    impl Codegen for Mock {
        fn emit(&self) -> Emit {
            Emit::Spirv
        }
        fn program(
            &self,
            _: &Profile,
            _: &Dirs,
            _: &[Selected<'_>],
            _: Scope,
        ) -> diagnostic::Result<String> {
            Ok(String::new())
        }
    }

    impl Driver for Mock {
        fn available(&self, _: &Profile) -> bool {
            false
        }
        fn run(
            &self,
            _: &Profile,
            _: &Dirs,
            _: &str,
            _: usize,
            _: Scope,
        ) -> diagnostic::Result<Readback> {
            Ok((Vec::new(), Vec::new()))
        }
        fn toolchain(&self) -> Option<crate::toolchain::Toolchain> {
            None
        }
    }

    impl Backend for Mock {}

    static MOCK: Mock = Mock;

    /// 生成 host C 参考后端的见证源码（设备文本 + 设备档案的 host 自检文本）。默认只断言非空；
    /// `CUTILE_CAPTURE_GOLDEN` 显式打开时才把副本写到 `/tmp/cutile-golden` 供人工比对。
    /// Metal / CUDA 的设备文本由 `engine` 生成，不再从 cutile 的注册表里取。
    #[test]
    fn golden_capture() {
        let capture = std::env::var("CUTILE_CAPTURE_GOLDEN").is_ok();
        let dir = std::path::PathBuf::from("/tmp/cutile-golden");
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-golden-work"));
        let catalogue = Catalogue::try_builtin().expect("内置数据");

        let cpu = catalogue.profile("cpu").cloned().expect("cpu 档案");
        let front = spec(Emit::C).expect("c 前端");
        let selected = crate::witness::declared_witnesses(&cpu).expect("见证");
        let text = front
            .program(&cpu, &dirs, &selected, Scope::Device)
            .expect("可生成");
        assert!(!text.is_empty(), "cpu 生成文本不能为空");
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join("witness_c.txt"), text).expect("写");
        }

        let metal = catalogue.profile("metal").cloned().expect("metal 档案");
        let selected = crate::witness::declared_witnesses(&metal).expect("见证");
        let host = front
            .program(&metal, &dirs, &selected, Scope::Host)
            .expect("可生成");
        assert!(!host.is_empty(), "metal 档案的 host 自检文本不能为空");
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join("witness_metal_host.txt"), host).expect("写");
        }
    }

    #[test]
    fn scratch_directories_do_not_collide() {
        let dirs = Dirs::under(std::env::temp_dir());
        let first = crate::runner::driver::scratch(&dirs, "gate");
        let second = crate::runner::driver::scratch(&dirs, "gate");
        assert_ne!(first, second);
        assert_eq!(first.parent(), Some(dirs.work()));
    }

    /// 执行后端注册表：宿主 C 参考后端种子在场、每个 `emit` 恰好一行、同 emit 重复注册被去重（
    /// `spec(..)` 命中第一行，后一行静默失效），表不许被清空。
    #[test]
    fn specs_register_each_execution_frontend_once() {
        assert!(spec(Emit::C).is_some(), "宿主 C 参考后端必须种子注册");
        crate::device::register(&MOCK);
        crate::device::register(&MOCK);

        let mut seen = std::collections::BTreeSet::new();
        for entry in specs() {
            assert!(
                seen.insert(entry.emit()),
                "执行前端 `{}` 在注册表里重复登记：后一行会被 `spec(..)` 静默忽略",
                entry.emit().as_str()
            );
            assert!(
                spec(entry.emit()).is_some(),
                "表内 `{}` 的查找行与登记表分家",
                entry.emit().as_str()
            );
        }
        assert!(!seen.is_empty(), "执行前端注册表不得被清空");
    }
}

pub mod probe;

pub mod space;

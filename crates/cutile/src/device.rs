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

pub mod probe {
    //! 设备探测：只如实探测，不做判断（README §5.3、§13 的 `probe`）。
    //!
    //! P0 只有 CPU 是真实可用的；GPU 档案保留下来，但探测不到设备时协商会拒绝它们。

    use crate::prof::{Detected, Profile};

    /// 探测本机 CPU。
    ///
    /// 指令集用 `is_x86_feature_detected!` / `is_aarch64_feature_detected!` 在**运行期**问 CPU，
    /// 不用 `cfg!(target_feature)`：后者回答的是“这个二进制按什么指令集编的”。默认 baseline 构建
    /// 下它永远报不出 `avx2`，而被 `-C target-cpu=native` 编出来的二进制拷到别的机器又会虚报能力
    /// ——指纹要描述这台机器，不是这个产物。
    pub fn detect_cpu() -> Detected {
        let arch = std::env::consts::ARCH;
        let mut features = vec![arch.to_string()];
        #[cfg(target_arch = "x86_64")]
        {
            for (name, present) in [
                ("avx2", std::arch::is_x86_feature_detected!("avx2")),
                ("avx512", std::arch::is_x86_feature_detected!("avx512f")),
            ] {
                if present {
                    features.push(name.to_string());
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                features.push("neon".to_string());
            }
        }
        Detected::new()
            .with("arch", arch)
            .with("features", features.join("+"))
    }

    /// 探测到的全部设备（P0 只有 CPU）。
    pub fn detect() -> Vec<Detected> {
        vec![detect_cpu()]
    }

    /// 按指纹挑档案：匹配、具体度排序与平局判定都归 `prof::select`，本模块只做本机探测。
    pub fn select<'a>(
        profiles: &'a [Profile],
        detected: &Detected,
    ) -> diagnostic::Result<&'a Profile> {
        crate::prof::select(profiles, detected)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::builtin::Catalogue;
        use crate::prof::MatchRule;
        use diagnostic::Code;

        /// 假指纹：回归测试替身。指纹从档案自己的 `[fingerprint]` 反查，不另写厂商表——规则语义只在
        /// `prof` 里定义一次，这里只填一个能满足规则的值；档案里没有该 vendor 时退回 `arch = <vendor>`，
        /// [`select`] 照常报"没有档案匹配"，不静默给能命中的指纹。
        fn fake(vendor: &str) -> Detected {
            let Some(profile) = ::prof::names()
                .iter()
                .filter_map(|name| ::prof::embedded(name))
                .filter_map(|table| Profile::parse(table).ok())
                .find(|profile| profile.meta.vendor == vendor)
            else {
                return Detected::new().with("arch", vendor);
            };
            profile
                .fingerprint
                .iter()
                .fold(Detected::new(), |detected, (key, rule)| {
                    detected.with(key, example_of(rule))
                })
        }

        /// 一条匹配规则的一个满足值。
        fn example_of(rule: &MatchRule) -> String {
            match rule {
                MatchRule::Exact(text) => text.clone(),
                MatchRule::Range { lo, .. } => lo.clone(),
                MatchRule::Contains(tokens) => tokens.first().cloned().unwrap_or_default(),
                MatchRule::Set { base, members } => format!("{base}{}", members.join("")),
                MatchRule::OneOf(branches) => branches.first().map_or_else(String::new, example_of),
            }
        }

        #[test]
        fn cpu_detection_matches_the_cpu_profile() {
            let catalogue = Catalogue::try_builtin().expect("内置数据");
            let detected = detect_cpu();
            assert_eq!(detected.get("arch"), Some(std::env::consts::ARCH));
            let profile = select(catalogue.profiles(), &detected).expect("必须命中");
            assert_eq!(profile.meta.vendor, "cpu");
        }

        #[test]
        fn fake_fingerprints_come_from_the_profiles_themselves() {
            let catalogue = Catalogue::try_builtin().expect("内置数据");
            // 逐份档案反查：加一台设备不需要改这里的任何一行。
            for profile in catalogue.profiles() {
                let detected = fake(&profile.meta.vendor);
                let chosen = select(catalogue.profiles(), &detected).expect("必须命中");
                assert_eq!(
                    chosen.meta.vendor, profile.meta.vendor,
                    "`{}` 的假指纹落到了别的档案：{detected}",
                    profile.meta.vendor
                );
            }
            // 值取自 `[fingerprint]` 的规则，而不是手写表：改档案里的上下界，假指纹跟着变。
            let metal = fake("metal");
            assert_eq!(metal.get("name"), Some("Apple M"), "contains 取第一个子串");
            assert_eq!(metal.get("gpu_cores"), Some("10"), "range 取下界");
        }

        #[test]
        fn cpu_features_are_probed_at_runtime_not_at_build_time() {
            let detected = detect_cpu();
            let features = detected.get("features").expect("features 必须有值");
            assert_eq!(features.split('+').next(), Some(std::env::consts::ARCH));
            // 与 cpuid 对拍：把 `cfg!(target_feature)` 换回来会让本机有 avx2 的 baseline 构建变红。
            #[cfg(target_arch = "x86_64")]
            assert_eq!(
                features.split('+').any(|feature| feature == "avx2"),
                std::arch::is_x86_feature_detected!("avx2"),
                "指纹必须描述这台机器的 cpuid：{features}"
            );
        }

        #[test]
        fn unknown_fingerprint_is_an_error_not_a_silent_fallback() {
            let catalogue = Catalogue::try_builtin().expect("内置数据");
            let error = select(catalogue.profiles(), &fake("sparc")).unwrap_err();
            assert_eq!(error.code(), Code::Unsupported);
            assert!(error.to_string().contains("没有档案匹配"), "{error}");
        }

        #[test]
        fn equally_specific_profiles_are_ambiguous() {
            let catalogue = Catalogue::try_builtin().expect("内置数据");
            let mut profiles = catalogue.profiles().to_vec();
            let cpu = profiles
                .iter()
                .find(|profile| profile.meta.vendor == "cpu")
                .expect("内置 cpu 档案")
                .clone();
            profiles.push(cpu);
            let error = select(&profiles, &detect_cpu()).unwrap_err();
            assert_eq!(error.code(), Code::Invalid);
            assert!(error.to_string().contains("同等具体"), "{error}");
        }
    }
}

pub mod space {
    //! 三级存储：语义名映射、容量校验、对齐校验（README §6.3）。

    use crate::atom::Space;
    use crate::prof::Profile;
    use diagnostic::{Code, Error};

    /// 一个存储层的用量。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SpaceUse {
        /// 语义存储层。
        pub space: Space,
        /// 字节数。
        pub bytes: u64,
    }

    impl SpaceUse {
        /// 构造。
        pub const fn new(space: Space, bytes: u64) -> Self {
            Self { space, bytes }
        }
    }

    /// 语义名映射到本设备的物理名；档案缺项时退回语义名本身。
    pub fn physical(profile: &Profile, space: Space) -> &str {
        profile.physical_space(space).unwrap_or(space.as_str())
    }

    /// 档案声明的容量；`None` 表示这一层不受档案限制（`GM`，或档案没有声明字节数）。
    ///
    /// 容量只有这一个出处：`check_capacity` 用它判定，报告用它打印，两者不会各算一份。
    pub fn capacity(profile: &Profile, space: Space) -> Option<u64> {
        let bytes = match space {
            Space::Fast => profile.mem.fast_bytes,
            Space::Reg => profile.mem.reg_bytes,
            Space::Gm => return None,
        };
        (bytes > 0).then_some(bytes)
    }

    /// 容量校验：`FAST` 与 `REG` 的用量不得超过档案声明的字节数。
    ///
    /// 超出容量不是“实现写得不好”，而是“这份声明与这台设备的组合不存在”——类别是
    /// [`Code::Unsupported`]，与协商拒绝同一个意思。
    pub fn check_capacity(profile: &Profile, uses: &[SpaceUse]) -> diagnostic::Result<()> {
        for usage in uses {
            let Some(limit) = capacity(profile, usage.space) else {
                continue;
            };
            if usage.bytes > limit {
                return Err(Error::new(
                    Code::Unsupported,
                    format!(
                        "{} 需要 {} B，档案只声明 {} B（物理层 `{}`）",
                        usage.space,
                        usage.bytes,
                        limit,
                        physical(profile, usage.space)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// 对齐校验：swizzle 的 bank 粒度不得超过档案声明的 bank 数。
    pub fn check_alignment(profile: &Profile, swizzle_bits: u8) -> diagnostic::Result<()> {
        let bank = profile.mem.fast_bank;
        if bank == 0 {
            return Ok(());
        }
        // `LayoutSpec::parse` 接受 `u8` 到 255，所以 `swz<40,…>` 是数据能写出来的。
        // 先挡住移位溢出，再比较：2^32 个 bank 不存在，任何 ≥ 32 的粒度都必然超限。
        let Some(needed) = 1u64.checked_shl(u32::from(swizzle_bits)) else {
            return Err(Error::new(
                Code::Unsupported,
                format!(
                    "swz<{swizzle_bits},…> 的 bank 粒度超出可表示范围，档案只有 {bank} 个 bank"
                ),
            ));
        };
        if needed > u64::from(bank) {
            return Err(Error::new(
                Code::Unsupported,
                format!("swz<{swizzle_bits},…> 需要 {needed} 个 bank，档案只有 {bank} 个"),
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn profile() -> Profile {
            Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("CPU 档案可解析")
        }

        #[test]
        fn maps_semantic_names_to_physical_ones() {
            let profile = profile();
            assert_eq!(physical(&profile, Space::Gm), "main");
            assert_eq!(physical(&profile, Space::Fast), "L2");
            assert_eq!(physical(&profile, Space::Reg), "stack");
        }

        #[test]
        fn capacity_is_the_archive_declaration() {
            let profile = profile();
            assert_eq!(capacity(&profile, Space::Fast), Some(524_288));
            assert_eq!(capacity(&profile, Space::Reg), Some(65_536));
            assert_eq!(capacity(&profile, Space::Gm), None, "GM 不受限额");
        }

        #[test]
        fn capacity_is_enforced_against_profile() {
            let profile = profile();
            assert!(check_capacity(&profile, &[SpaceUse::new(Space::Fast, 1024)]).is_ok());
            let error =
                check_capacity(&profile, &[SpaceUse::new(Space::Fast, 1 << 30)]).unwrap_err();
            assert_eq!(error.code(), Code::Unsupported);
            assert!(error.to_string().contains("L2"), "{error}");
        }

        #[test]
        fn global_memory_is_unbounded_in_the_profile() {
            let profile = profile();
            assert!(check_capacity(&profile, &[SpaceUse::new(Space::Gm, u64::MAX / 2)]).is_ok());
        }

        #[test]
        fn alignment_uses_declared_bank_count() {
            let profile = profile();
            assert!(check_alignment(&profile, 4).is_ok());
            assert!(check_alignment(&profile, 6).is_err());
        }

        #[test]
        fn alignment_rejects_granularities_that_overflow() {
            // `swz<40,…>` 是声明里能写出来的，必须报错而不是 panic。
            let error = check_alignment(&profile(), 40).unwrap_err();
            assert!(error.to_string().contains("40"), "{error}");
        }
    }
}

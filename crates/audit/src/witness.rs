//! 见证执行器（DESIGN §11.3）：每条 `grade ≥ L2` 的供给都要真的跑一次自己的见证。
//!
//! 档案校验只能证明"文件在、函数名写在源码里"；本模块把见证源码编成可执行文件，逐条调用
//! `int func(float *out)` 并与 `witness.expect` 对拍。没有跑通的提供不会通过 `verify`，
//! 因此 `grade` 不再是作者的自述：它是这次执行的结论。
//!
//! 执行分两种范围（[`Scope`]）：[`run_all`] 是**设备执行**（`emit = c` 时设备就是本机）；
//! [`self_check`] 是**本机自检**，把每条已声明的见证按 host C ABI 编译执行一遍——它证明
//! "见证写了什么、算得对不对"，不证明设备具备该能力，所以不改变任何 `grade`。

use cutile::atom::{DType, Prim, Witness};
use cutile::backend;
use cutile::build;
use cutile::prof::Profile;
use cutile::reference;
use diagnostic::{Code, Error};
use llm_config::Settings;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// 一次见证执行能证明什么：范围不同，结论的效力就不同。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// 设备执行：`emit = c` 时设备就是本机，通过即为 L2 证据。
    Device,
    /// 本机自检：用 host C 前端编译并执行见证函数，证明源码可编译、`expect` 算得对；
    /// **不**证明设备具备这条能力，所以不改变等级。
    Host,
}

/// 一条见证的执行结论。
#[derive(Debug, Clone, PartialEq)]
pub struct WitnessRun {
    /// 原语。
    pub prim: Prim,
    /// 实现名。
    pub impl_name: String,
    /// 入口函数名。
    pub func: String,
    /// 实际写入的元素个数。
    pub values: usize,
    /// 与 `expect` 的相对误差。
    pub max_rel_err: f64,
    /// 这次结论来自设备执行还是本机自检。
    pub scope: Scope,
}

/// 执行档案里所有 `grade ≥ L2` 的见证。
///
/// 一台设备的 `provides` 通常共用一份见证源码，因此整份文件只编译一次、逐函数调用。
///
/// # Errors
///
/// 缺编译器、见证文件不可读、函数不存在、编译/执行失败、返回值个数不符或对拍超出容差时，
/// 返回带原语与函数名的说明。
pub fn run_all(profile: &Profile, settings: &Settings) -> diagnostic::Result<Vec<WitnessRun>> {
    let selected = verified_witnesses(profile)?;
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    if profile.meta.emit != cutile::prof::Emit::C {
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "设备见证只能由本机执行 C 档案（`{}` 的 emit 是 {}）；非 C 设备可先跑 `self_check` 的 host 自检，但自检不构成 L2",
                profile.meta.vendor, profile.meta.emit
            ),
        ));
    }
    execute(profile, settings, &selected, Scope::Device)
}

/// 对**每条已声明**的见证做本机自检，不要求 `grade ≥ L2`。
///
/// 见证的 ABI 是 host C（`int func(float *)`），与片段语言无关：这里跳过档案的
/// `[compile].preamble` 与 `flags`（`#include <cuda_fp16.h>`、`-arch=sm_100a` 这类只对
/// 真机编译器有意义），用 host 编译器把见证源码与 harness 编成可执行文件跑一遍。
///
/// 结论的边界写在 [`Scope::Host`] 里：自检通过只说明"见证源码能编译、`expect` 算得对"，
/// **不**说明设备具备这条能力。因此自检不改写任何 `grade`——L2 仍要设备执行（[`run_all`]）。
///
/// # Errors
///
/// 缺 host C 编译器、见证文件不可读、函数不存在、编译/执行失败、返回值个数不符或对拍
/// 超出容差时，返回带设备、原语与函数名的说明。
pub fn self_check(profile: &Profile, settings: &Settings) -> diagnostic::Result<Vec<WitnessRun>> {
    let selected = declared_witnesses(profile);
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    execute(profile, settings, &selected, Scope::Host).map_err(|error| {
        error.with(format!(
            "`{}({})` 的见证自检不成立",
            profile.meta.vendor, profile.meta.id
        ))
    })
}

/// `grade ≥ L2` 的供给：按 DESIGN §11.3，等级是执行的结论，缺见证是错误而不是跳过。
fn verified_witnesses(profile: &Profile) -> diagnostic::Result<Vec<(Prim, String, &Witness)>> {
    let mut selected = Vec::new();
    for provide in &profile.provides {
        if !provide.grade.is_verified() {
            continue;
        }
        let Some(witness) = provide.witness.as_ref() else {
            return Err(Error::new(
                Code::Missing,
                format!(
                    "`{}`（{}）声明了 {}，但没有可执行的见证：等级必须是跑出来的结论",
                    provide.prim, provide.impl_name, provide.grade
                ),
            ));
        };
        selected.push((provide.prim, provide.impl_name.clone(), witness));
    }
    Ok(selected)
}

/// 声明了见证的供给：自检只看"写了什么"，不看等级。
fn declared_witnesses(profile: &Profile) -> Vec<(Prim, String, &Witness)> {
    profile
        .provides
        .iter()
        .filter_map(|provide| {
            provide
                .witness
                .as_ref()
                .map(|witness| (provide.prim, provide.impl_name.clone(), witness))
        })
        .collect()
}

/// 编译一次、逐条调用。
///
/// 两个调用方只差三件事：是否使用设备 preamble/flags（自检必须跳过）、工作目录标签与结果里
/// 的 [`Scope`]；编译、计数与对拍的逻辑完全共享。
fn execute(
    profile: &Profile,
    settings: &Settings,
    selected: &[(Prim, String, &Witness)],
    scope: Scope,
) -> diagnostic::Result<Vec<WitnessRun>> {
    let tool = backend::detect_toolchain().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            "缺少 C 编译器（cc/clang/gcc 都不在 PATH 上），无法执行见证",
        )
    })?;
    let (preamble, flags, tag) = match scope {
        Scope::Device => (profile.preamble.clone(), profile.flags.clone(), "witness"),
        // 自检只走 host C ABI：设备的 `[compile]` 前置声明与参数在这台机器上要么不存在
        // （`#include <cuda_fp16.h>`），要么只对真机编译器有意义（`-arch=sm_100a`）。
        Scope::Host => (
            String::new(),
            vec!["-O2".to_string(), "-std=c99".to_string()],
            "self",
        ),
    };
    let mut sources: BTreeMap<String, String> = BTreeMap::new();
    for (prim, _, witness) in selected {
        if sources.contains_key(&witness.file) {
            continue;
        }
        let text = source(witness, settings).map_err(|error| error.with(format!("`{prim}` 的")))?;
        if !text.contains(&format!("{}(", witness.func)) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的见证 `{}` 里没有函数 `{}(float *)`",
                    witness.file, witness.func
                ),
            ));
        }
        sources.insert(witness.file.clone(), text);
    }

    let mut program = preamble;
    if !program.is_empty() && !program.ends_with('\n') {
        program.push('\n');
    }
    program.push_str("#include <stdio.h>\n\n");
    for text in sources.values() {
        program.push_str(text);
        if !program.ends_with('\n') {
            program.push('\n');
        }
        program.push('\n');
    }
    program.push_str(&harness(selected));

    // 见证程序也进产物缓存：第二次 `verify` 只在第一次调用编译器。
    let (outcome, _artifact) = build::run_program(&tool, &flags, tag, &program, settings)?;
    let counts = parse_counts(&outcome.stderr);
    if counts.len() != selected.len() {
        return Err(Error::new(
            Code::Internal,
            format!(
                "见证执行只回报了 {} 条计数，期望 {} 条：可执行文件没有按顺序调用每条见证",
                counts.len(),
                selected.len()
            ),
        ));
    }

    let tolerance = reference::tolerance(DType::F32);
    let mut offset = 0usize;
    let mut runs = Vec::with_capacity(selected.len());
    for ((prim, impl_name, witness), &count) in selected.iter().zip(&counts) {
        let written = usize::try_from(count).unwrap_or(usize::MAX);
        if written != witness.expect.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的见证 `{}` 写了 {count} 个值，档案声明期望 {} 个",
                    witness.func,
                    witness.expect.len()
                ),
            ));
        }
        let end = offset + written;
        let got = outcome.bits.get(offset..end).ok_or_else(|| {
            Error::new(
                Code::Internal,
                format!(
                    "`{prim}` 的见证 `{}` 只回来了 {} 个值（第 {} 段需要 {written} 个）",
                    witness.func,
                    outcome.bits.len(),
                    runs.len() + 1
                ),
            )
        })?;
        let error = reference::max_rel_err(got, &witness.expect);
        if error > tolerance {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{prim}` 的见证 `{}` 不成立：得到 {got:?}，期望 {:?}，max_rel_err {error:.3e} > {tolerance:.0e}",
                    witness.func, witness.expect
                ),
            ));
        }
        runs.push(WitnessRun {
            prim: *prim,
            impl_name: impl_name.clone(),
            func: witness.func.clone(),
            values: written,
            max_rel_err: error,
            scope,
        });
        offset = end;
    }
    if offset != outcome.bits.len() {
        return Err(Error::new(
            Code::Internal,
            format!(
                "见证输出多出 {} 个值：计数行与实际打印不一致",
                outcome.bits.len() - offset
            ),
        ));
    }
    Ok(runs)
}

/// 见证源码：内置设备读编译期内嵌副本（发布机没有源码树），`data.root` 或未登记的设备走磁盘。
fn source(witness: &Witness, settings: &Settings) -> diagnostic::Result<String> {
    let ext = std::path::Path::new(&witness.file)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let vendor = witness.file.split('/').next().unwrap_or("");
    if settings.data_root().is_none()
        && let Some(text) = prim::builtin_witness(vendor, ext)
    {
        return Ok(text.to_string());
    }
    let path = witness_root(settings).join(&witness.file);
    std::fs::read_to_string(&path).map_err(|error| {
        Error::new(
            Code::Io,
            format!("见证 `{}` 不可读：{error}", path.display()),
        )
    })
}

/// 见证源码的根目录：与 `Catalogue` 的装载根保持一致（`data.root` 优先）。
fn witness_root(settings: &Settings) -> PathBuf {
    settings
        .data_root()
        .map_or_else(prim::dir, prim::layer_root)
}

/// 生成 `main()`：逐条调用见证函数，值走 stdout，个数走 stderr。
///
/// 个数必须单独走一条通道：stdout 只放 f32 位模式，`backend::parse_bits` 才能原样解码；
/// 把计数混进 stdout 会让"写了几个"与"值是多少"互相污染。
fn harness(selected: &[(Prim, String, &Witness)]) -> String {
    let max = Witness::MAX_OUT;
    let mut text = format!("\nint main(void) {{\n    float out[{max}];\n    int n;\n");
    for (_, _, witness) in selected {
        text.push_str("    n = ");
        text.push_str(&witness.func);
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
    text.push_str("    return 0;\n}\n");
    text
}

/// 从 stderr 读出每条见证声明的写入个数（按调用顺序）。
fn parse_counts(stderr: &str) -> Vec<i64> {
    stderr
        .lines()
        .filter_map(|line| line.trim().strip_prefix("witness "))
        .filter_map(|value| value.trim().parse::<i64>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile::builtin::Catalogue;

    fn settings() -> Settings {
        let roots = llm_config::Roots::new(std::env::temp_dir().join("cutile-witness-tests"));
        Settings::<llm_config::Cutile>::defaults(&roots)
    }

    fn cpu() -> Profile {
        Catalogue::try_builtin()
            .expect("内置数据")
            .profile("cpu")
            .cloned()
            .expect("cpu 档案")
    }

    #[test]
    fn every_l2_provide_executes_its_own_witness() {
        if backend::detect_toolchain().is_none() {
            return;
        }
        let profile = cpu();
        let expected = profile
            .provides
            .iter()
            .filter(|provide| provide.grade.is_verified())
            .count();
        let runs = run_all(&profile, &settings()).expect("见证必须全部通过");
        assert_eq!(runs.len(), expected, "每条 L2 供给都要有一条执行结果");
        assert!(runs.iter().all(|run| run.max_rel_err <= 1e-5), "{runs:?}");
    }

    #[test]
    fn a_wrong_expectation_is_a_failure() {
        if backend::detect_toolchain().is_none() {
            return;
        }
        let mut profile = cpu();
        profile.provides[0].witness.as_mut().expect("有见证").expect[0] += 1.0;
        let error = run_all(&profile, &settings()).expect_err("期望值不对必须失败");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("quant"), "{error}");
    }

    #[test]
    fn a_witness_must_write_exactly_the_declared_number_of_values() {
        if backend::detect_toolchain().is_none() {
            return;
        }
        let mut profile = cpu();
        profile.provides[0]
            .witness
            .as_mut()
            .expect("有见证")
            .expect
            .truncate(1);
        let error = run_all(&profile, &settings()).expect_err("个数不符必须失败");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("写了 2 个值"), "{error}");
    }

    #[test]
    fn a_missing_function_is_a_failure_without_compiling() {
        let mut profile = cpu();
        profile.provides[0].witness.as_mut().expect("有见证").func = "witness_absent".to_string();
        let error = run_all(&profile, &settings()).expect_err("函数不存在必须失败");
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("witness_absent"), "{error}");
    }

    fn device(vendor: &str) -> Profile {
        Catalogue::try_builtin()
            .expect("内置数据")
            .profile(vendor)
            .cloned()
            .unwrap_or_else(|| panic!("{vendor} 档案"))
    }

    /// 四台加速器的见证虽然进不了协商，但它们是写下来的声明：每一条都要能编译、能算出
    /// 档案里的 `expect`。写错一个期望值不该拖到真机接入那天才暴露。
    #[test]
    fn every_declared_accelerator_witness_passes_the_host_self_check() {
        if backend::detect_toolchain().is_none() {
            return;
        }
        for vendor in ["nvidia", "intel", "ascend", "metal"] {
            let profile = device(vendor);
            let declared = profile
                .provides
                .iter()
                .filter(|provide| provide.witness.is_some())
                .count();
            let runs = self_check(&profile, &settings())
                .unwrap_or_else(|error| panic!("{vendor} 的见证自检失败：{error}"));
            assert_eq!(runs.len(), declared, "{vendor}: 每条声明的见证都要跑一次");
            assert!(
                runs.iter().all(|run| run.max_rel_err <= 1e-5),
                "{vendor}: {runs:?}"
            );
            assert!(
                runs.iter().all(|run| run.scope == Scope::Host),
                "{vendor}: 自检的结论不能被当成设备证据"
            );
        }
    }

    /// 自检走 host C ABI：设备的 `[compile]` 前置声明与参数不能泄进来（这台机器上既没有
    /// `cuda_fp16.h`，也没有 `nvcc`）。把 preamble 换成一个必然编不过的 `#error`，自检仍必须通过。
    #[test]
    fn the_self_check_ignores_the_device_preamble_and_flags() {
        if backend::detect_toolchain().is_none() {
            return;
        }
        let mut profile = device("nvidia");
        profile.preamble = "#error 设备专属前置声明\n".to_string();
        profile.flags = vec!["-arch=sm_100a".to_string(), "-nocudainc".to_string()];
        let runs = self_check(&profile, &settings()).expect("自检不得使用设备 preamble/flags");
        assert!(!runs.is_empty(), "nvidia 有已声明的见证");
    }

    /// 自检是硬判据：期望值错一位就失败，而不是“没设备所以不知道”。
    #[test]
    fn a_wrong_expectation_fails_even_the_host_self_check() {
        if backend::detect_toolchain().is_none() {
            return;
        }
        let mut profile = device("ascend");
        profile.provides[0].witness.as_mut().expect("有见证").expect[0] += 1.0;
        let error = self_check(&profile, &settings()).expect_err("期望值不对必须失败");
        assert!(error.to_string().contains("见证自检不成立"), "{error}");
        assert!(error.to_string().contains("ascend"), "{error}");
    }

    #[test]
    fn counts_are_read_from_stderr_in_order() {
        assert_eq!(parse_counts("witness 1\nwitness 2\n"), vec![1, 2]);
        assert_eq!(parse_counts("noise\n"), Vec::<i64>::new());
    }
}

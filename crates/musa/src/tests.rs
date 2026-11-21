use std::path::Path;

use cutile::builtin::Catalogue;
use cutile::dirs::Dirs;
use diagnostic::{Code, Error, Result};

use super::*;

/// 全新空目录：没有任何 `verify` 产物。生成不读本机证据，用它才测得出这件事。
fn scratch(tag: &str) -> Dirs {
    let root = std::env::temp_dir().join(format!("llm-musa-{tag}-{}", std::process::id()));
    Dirs::under(root)
}

#[test]
fn a_pointwise_job_yields_a_runnable_program() -> Result<()> {
    let job = Job::of(&pointwise()?).shape(64, 64);
    let generated = program(&job, &scratch("gen"))?;
    assert!(
        generated.device.contains("<<<"),
        "设备源里没有内核发射：驱动没有拼进 host 段"
    );
    let text = lines(&generated.reference);
    let outcome = compare(&text, &text, 0.0)?;
    assert!(outcome.pass());
    assert!(outcome.compared > 0, "参考值为空，等于没测");
    Ok(())
}

#[test]
fn generation_does_not_need_local_execution_evidence() -> Result<()> {
    // 空目录 = 这台机器没跑过 `verify`。生成仍要成立，否则"在这里生成、拿到目标机编译"就没了。
    let empty = scratch("evidence");
    let generated = program(&Job::of(&pointwise()?).shape(32, 32), &empty)?;
    assert!(!generated.reference.is_empty(), "没生成参考值");
    Ok(())
}

#[test]
fn a_job_takes_its_whole_shape_at_once() {
    let job = Job::of("add_f32").shape(8, 4);
    assert_eq!((job.m, job.n), (8, 4));
    assert_eq!(Job::of("add_f32"), Job::of("add_f32"));
}

#[test]
fn timing_wraps_one_launch_and_reports_on_stderr() -> Result<()> {
    let generated = program(&Job::of(&pointwise()?).shape(64, 64), &scratch("timing"))?;
    let timed = timed(&generated.device, 16)?;
    assert!(timed.contains("for (int rep = 0; rep < 16; ++rep) {"));
    assert!(timed.contains("bench: %.4f ms/iter"));
    // 注入只夹住发射与回读之间：首段发射必须落在循环体里。
    let launch = timed.find("<<<").expect("发射");
    let loop_head = timed.find("for (int rep").expect("循环头");
    assert!(loop_head < launch, "发射没有被包进计时循环");
    Ok(())
}

#[test]
fn timing_refuses_what_it_cannot_wrap() {
    assert!(timed("int main() { return 0; }", 8).is_err());
    assert!(timed("<<<", 0).is_err());
}

#[test]
fn a_single_element_over_threshold_fails_the_whole_run() -> Result<()> {
    let outcome = compare("1.0\n2.0001\n", "1.0\n2.0\n", 1e-6)?;
    assert!(!outcome.pass());
    assert_eq!(outcome.over, 1);
    assert_eq!(outcome.at, 1);

    let loose = compare("1.0\n2.0001\n", "1.0\n2.0\n", 1e-3)?;
    assert!(loose.pass());
    Ok(())
}

#[test]
fn length_mismatch_and_empty_runs_are_errors() {
    assert!(compare("1.0\n", "1.0\n2.0\n", 1e-6).is_err());
    assert!(compare("\n", "\n", 1e-6).is_err());
    assert!(compare("1.0\n", "abc\n", 1e-6).is_err());
}

#[test]
fn nan_agrees_with_nan_but_not_with_a_number() -> Result<()> {
    let same = compare("NaN\n", "NaN\n", 0.0)?;
    assert!(same.pass());
    let differs = compare("1.0\n", "NaN\n", 1.0)?;
    assert!(!differs.pass());
    Ok(())
}

#[test]
fn a_lane_adds_only_its_own_flags() -> Result<()> {
    let root = Path::new("/opt/musa");
    let plain = Recipe::at(root, "plain")?;
    let tensor = Recipe::at(root, "wmma")?;
    let blas = Recipe::at(root, "mublas")?;
    let (source, binary, shim) = (Path::new("a.cu"), Path::new("a"), Path::new("shim"));

    let plain_args = plain.argv(source, binary, shim).join(" ");
    assert!(!plain_args.contains("mp_22"), "普通车道不该锁张量核架构");
    assert!(!plain_args.contains("-D__MUSA_QY2_MMA__"));

    let tensor_args = tensor.argv(source, binary, shim).join(" ");
    assert!(tensor_args.contains("--cuda-gpu-arch=mp_22"));
    assert!(tensor_args.contains("-D__MUSA_QY2_MMA__=1"));

    let blas_args = blas.argv(source, binary, shim).join(" ");
    assert!(blas_args.contains("-lmublas"));
    assert!(!blas_args.contains("mp_22"));

    assert!(Recipe::at(root, "opencl").is_err());
    Ok(())
}

#[test]
fn probes_are_named_and_carry_a_main() {
    for entry in &PROBES {
        let found = probe(entry.name).expect("按名字可取");
        assert_eq!(found.name, entry.name);
        assert!(found.source.contains("int main"), "{} 没有入口", entry.name);
        assert!(!found.about.is_empty());
    }
    assert!(probe("no-such-probe").is_none());
}

#[test]
fn shim_writes_both_headers_verbatim() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("llm-musa-shim-{}", std::process::id()));
    shim(&dir)?;
    shim(&dir)?;
    let runtime = std::fs::read_to_string(dir.join("cuda_runtime.h"))
        .map_err(|err| Error::new(Code::Io, format!("{err}")))?;
    assert!(runtime.contains("#define cudaMalloc musaMalloc"));
    assert!(runtime.contains("#define cudaEventElapsedTime musaEventElapsedTime"));
    let fp16 = dir.join("cuda_fp16.h");
    assert!(fp16.is_file(), "{} 没写出来", fp16.display());
    Ok(())
}

/// 从全量算子表里挑一个真逐点算子：名字写死在测试里会在算子改名后静默变成"跳过"。
fn pointwise() -> Result<String> {
    let catalogue = Catalogue::try_builtin()?;
    catalogue
        .ops()
        .iter()
        .find(|decl| decl.pointwise() && op::model(&decl.name).is_some())
        .map(|decl| decl.name.clone())
        .ok_or_else(|| Error::new(Code::Missing, "算子表里没有逐点算子"))
}

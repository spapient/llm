use super::*;
use std::path::Path;

/// 装配 Metal / CUDA 后端（幂等，进程级只装一次）。引擎测试直接调 `MetalBackend`/`CudaBackend` 的
/// `Codegen`/`Driver` 方法；经 `cutile::witness::device_available` 判断真机在场时，先注入注册表。
fn ensure_backends() {
    use std::sync::OnceLock;
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(install);
}

fn metal_profile() -> Profile {
    cutile::builtin::Catalogue::try_builtin()
        .expect("内置数据")
        .profile("metal")
        .cloned()
        .expect("metal 档案")
}

// ---------------------------------------------------------------------------
// 厂商工具链（MSVC cl.exe）：从 cutile 搬来，`find_cl`/`msvc_version` 只服务 Windows 上的 nvcc。
// ---------------------------------------------------------------------------

#[test]
fn paths_without_an_msvc_segment_parse_as_empty() {
    let mut rng = testkit::Rng::new(0xC12);
    for _ in 0..32 {
        let path = format!(r"C:\Tools\{}\bin\cl.exe", rng.below(9999));
        assert!(msvc_version(Path::new(&path)).is_empty());
    }
}

#[test]
fn the_newest_msvc_wins_by_version_not_by_string_order() {
    let candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\9.29.30133\bin\Hostx64\x64\cl.exe",
        r"D:\BuildTools\VC\Tools\MSVC\17.10.0\bin\Hostx64\x64\cl.exe",
    ];
    let newest = candidates
        .into_iter()
        .max_by_key(|cl| msvc_version(Path::new(cl)))
        .expect("候选非空");
    assert!(
        newest.contains("17.10.0"),
        "字符串序会把 9.29 排到 17.10 之后：{newest}"
    );
}

#[test]
fn a_path_without_a_version_loses_to_one_with_it() {
    assert!(msvc_version(Path::new(r"C:\msvc\cl.exe")).is_empty());
    assert_eq!(
        msvc_version(Path::new(
            r"C:\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"
        )),
        vec![14, 29, 30133]
    );
}

// ---------------------------------------------------------------------------
// Metal 见证驱动：宽度取宏、非空。
// ---------------------------------------------------------------------------

#[test]
fn the_metal_driver_takes_its_width_from_the_witness_limit() {
    let driver = metal_driver(MAX_OUT);
    assert!(driver.contains(&format!("#define WITNESS_MAX_OUT {MAX_OUT}")));
    assert_eq!(
        driver.matches("WITNESS_MAX_OUT").count(),
        5,
        "驱动里还有写死的宽度"
    );
    for literal in ["count * 8", "out[i * 8", "(n > 8)"] {
        assert!(!driver.contains(literal), "仍是字面量：{literal}");
    }
}

#[test]
fn metal_driver_wrapper_is_nonempty() {
    let driver = metal_driver(MAX_OUT);
    assert!(!driver.is_empty());
}

// ---------------------------------------------------------------------------
// Metal 见证真机 + 前端签名（从 cutile/witness 搬来）。
// ---------------------------------------------------------------------------

#[test]
fn every_metal_witness_runs_on_the_gpu() {
    ensure_backends();
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 设备不在本机，GPU 见证用例未运行");
        return;
    }
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-witness"));
    let runs = cutile::witness::run_all(&profile, &dirs).expect("Metal 见证必须全部通过");
    assert_eq!(runs.len(), profile.provides.len(), "每条声明都要有结果");
    assert!(
        runs.iter().all(|run| run.scope == Scope::Device),
        "GPU 上跑出来的结论不能被降级成自检"
    );
    assert!(runs.iter().all(|run| run.max_rel_err <= 1e-5), "{runs:?}");
}

#[test]
fn a_plain_c_signature_does_not_survive_the_metal_front_end() {
    ensure_backends();
    if !cutile::witness::device_available(&metal_profile()) {
        testkit::skip("engine: Metal 设备不在本机，MSL 前端用例未运行");
        return;
    }
    let program = "#include <metal_stdlib>\nusing namespace metal;\n\
                   int witness_bad(float *out) { out[0] = 1.0f; return 1; }\n";
    let profile = metal_profile();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-frontend"));
    let error = MetalBackend
        .run(&profile, &dirs, program, 1, Scope::Device)
        .expect_err("MSL 必须要求显式地址空间");
    let text = error.to_string();
    assert!(text.contains("xcrun metal"), "{text}");
    assert!(text.contains("address space"), "{text}");
}

// ---------------------------------------------------------------------------
// 设备生成文本的花括号配对：Metal / CUDA 两套 Codegen 都要守住。
// ---------------------------------------------------------------------------

#[test]
fn generated_device_program_braces_are_balanced() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-brace-work"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    for (vendor, front) in [
        ("nvidia", &CUDA_BACKEND as &dyn Codegen),
        ("metal", &METAL_BACKEND as &dyn Codegen),
    ] {
        let Some(profile) = catalogue.profile(vendor).cloned() else {
            continue;
        };
        let selected = cutile::witness::declared_witnesses(&profile).expect("见证");
        let text = front
            .program(&profile, &dirs, &selected, Scope::Device)
            .expect("可生成");
        let open = text.chars().filter(|c| *c == '{').count();
        let close = text.chars().filter(|c| *c == '}').count();
        assert_eq!(
            open, close,
            "{vendor} 生成文本花括号不配对：{open} != {close}"
        );
    }
}

// ---------------------------------------------------------------------------
// 生产态 Metal 网格内核编译：把一条真实算子走完整条流水线，再经 `xcrun metal` 编译证明合法。
// ---------------------------------------------------------------------------

#[test]
fn production_grid_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-grid"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，生产态内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("silu_f32").expect("silu_f32 语义表").shape;
    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, "silu_f32", m, n)
        .expect("生产态网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert_metal_compiles("engine-metal-grid", &msl, &profile.flags);
}

#[test]
fn production_reduce_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-reduce"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，归约内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("reduce_sum_f32")
        .expect("reduce_sum_f32 语义表")
        .shape;
    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, "reduce_sum_f32", m, n)
        .expect("归约网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(msl.contains("thread_position_in_grid"), "{msl}");
    assert!(msl.contains("_cutile_sum_f32"), "{msl}");
    assert_metal_compiles("engine-metal-reduce", &msl, &profile.flags);
}

#[test]
fn production_scan_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-scan"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，扫描内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("scan_f32").expect("scan_f32 语义表").shape;
    let (local, correct) =
        cutile::execution::metal_scan_program(&catalogue, &dirs, "scan_f32", m, n)
            .expect("分块扫描内核应能生成");
    assert!(local.contains("kernel_main"), "{local}");
    assert!(local.contains("thread_position_in_grid"), "{local}");
    assert!(local.contains("start + len"), "{local}");
    assert!(correct.contains("kernel_main"), "{correct}");
    assert!(correct.contains("d_c"), "{correct}");
    assert_metal_compiles("engine-metal-scan", &local, &profile.flags);
    assert_metal_compiles("engine-metal-scan", &correct, &profile.flags);
}

#[test]
fn production_rmsnorm_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-rmsnorm"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，两趟内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("rmsnorm_f32").expect("rmsnorm_f32 语义表").shape;
    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, "rmsnorm_f32", m, n)
        .expect("两趟内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(!msl.contains("thread_position_in_grid"), "{msl}");
    let (partial, map) =
        cutile::execution::metal_rmsnorm_program(&catalogue, &dirs, "rmsnorm_f32", m, n)
            .expect("分块 rmsnorm 内核应能生成");
    assert!(partial.contains("_cutile_sumsq_f32"), "{partial}");
    assert!(
        map.contains("d_rms") && map.contains("thread_position_in_grid"),
        "{map}"
    );
    assert_metal_compiles("engine-metal-rmsnorm", &msl, &profile.flags);
    assert_metal_compiles("engine-metal-rmsnorm", &partial, &profile.flags);
    assert_metal_compiles("engine-metal-rmsnorm", &map, &profile.flags);
}

#[test]
fn production_softmax_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-softmax"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，softmax 内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("softmax_f32").expect("softmax_f32 语义表").shape;
    let (max, expsum, map) =
        cutile::execution::metal_softmax_program(&catalogue, &dirs, "softmax_f32", m, n)
            .expect("分块 softmax 内核应能生成");
    assert!(max.contains("_cutile_max_f32"), "{max}");
    assert!(
        expsum.contains("_cutile_exp_sum_f32") && expsum.contains("d_max"),
        "{expsum}"
    );
    assert!(
        map.contains("_cutile_exp_norm_f32") && map.contains("d_sum"),
        "{map}"
    );
    assert_metal_compiles("engine-metal-softmax", &max, &profile.flags);
    assert_metal_compiles("engine-metal-softmax", &expsum, &profile.flags);
    assert_metal_compiles("engine-metal-softmax", &map, &profile.flags);
}

#[test]
fn production_gemv_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-gemv"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，gemv 内核编译用例未运行");
        return;
    }
    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, "gemv_f32", 64, 4)
        .expect("gemv 网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(msl.contains("thread_position_in_grid"), "{msl}");
    assert!(
        msl.contains("for (int n = (int)tid.x; n < (int)tid.x + 1; ++n)"),
        "{msl}"
    );
    assert_metal_compiles("engine-metal-gemv", &msl, &profile.flags);
}

#[test]
fn production_gemm_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-gemm"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，gemm 内核编译用例未运行");
        return;
    }
    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, "gemm_f32", 16, 16)
        .expect("gemm 网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        msl.contains("uint3 tid [[thread_position_in_grid]]"),
        "{msl}"
    );
    assert!(
        msl.contains("for (int i = (int)tid.x; i < (int)tid.x + 1; ++i)"),
        "{msl}"
    );
    assert!(
        msl.contains("for (int j = (int)tid.y; j < (int)tid.y + 1; ++j)"),
        "{msl}"
    );
    assert_metal_compiles("engine-metal-gemm", &msl, &profile.flags);
}

#[test]
fn production_gemv_i2_kernel_compiles_for_a_real_op() {
    production_quantized_gemv_compiles("gemv_i2");
}

#[test]
fn production_gemv_i4_kernel_compiles_for_a_real_op() {
    production_quantized_gemv_compiles("gemv_i4");
}

#[test]
fn production_gemv_i8_kernel_compiles_for_a_real_op() {
    production_quantized_gemv_compiles("gemv_i8");
}

#[test]
fn production_attention_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-attention"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，attention 内核编译用例未运行");
        return;
    }
    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, "attention_f32", 8, 8)
        .expect("attention 网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        msl.contains("for (int i = (int)tid.x; i < (int)tid.x + 1; ++i)"),
        "{msl}"
    );
    assert_metal_compiles("engine-metal-attention", &msl, &profile.flags);
}

#[test]
fn production_flash_attention_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-flash-attention"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，flash_attention 内核编译用例未运行");
        return;
    }
    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, "flash_attention_f32", 8, 8)
        .expect("flash_attention 网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        msl.contains("for (int i = (int)tid.x; i < (int)tid.x + 1; ++i)"),
        "{msl}"
    );
    assert_metal_compiles("engine-metal-flash-attention", &msl, &profile.flags);
}

fn production_quantized_gemv_compiles(op_name: &str) {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-gemv-quant"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip(format!(
            "engine: Metal 工具链不在本机，{op_name} 内核编译用例未运行"
        ));
        return;
    }
    let msl = cutile::execution::metal_grid_program(&catalogue, &dirs, op_name, 64, 4)
        .unwrap_or_else(|error| panic!("{op_name} 网格内核应能生成：{error}"));
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        msl.contains("for (int n = (int)tid.x; n < (int)tid.x + 1; ++n)"),
        "{msl}"
    );
    assert_metal_compiles(&format!("engine-metal-{op_name}"), &msl, &profile.flags);
}

/// 写 MSL 进一次性目录、经 [`metal_kernel`] 编成 metallib、删目录；编译失败即 panic。
fn assert_metal_compiles(stem: &str, msl: &str, flags: &[String]) {
    let dir = std::env::temp_dir().join(format!("{stem}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("目录");
    let compiled = metal_kernel(&dir, msl, flags);
    let _ = std::fs::remove_dir_all(&dir);
    compiled.unwrap_or_else(|error| panic!("MSL 应能通过 xcrun metal 编译：{error}"));
}

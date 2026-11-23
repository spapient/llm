//! 生产态 Metal 网格内核编译：把一条真实算子走完整条流水线，再经 `xcrun metal` 编译证明合法。

use super::*;

use cutile::dirs::Dirs;
use llm_optim::execution::Backend::Metal;
use llm_optim::execution::{Mode, program};

#[test]
fn production_grid_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-grid"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，生产态内核编译用例未运行");
        return;
    }
    let (m, n) = nn::registry::model("silu_f32")
        .expect("silu_f32 语义表")
        .shape;
    let msl = program(&catalogue, &dirs, "silu_f32", Metal, Mode::declared(), m, n)
        .and_then(|stages| stages.one("降源"))
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
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，归约内核编译用例未运行");
        return;
    }
    let (m, n) = nn::registry::model("reduce_sum_f32")
        .expect("reduce_sum_f32 语义表")
        .shape;
    let msl = program(
        &catalogue,
        &dirs,
        "reduce_sum_f32",
        Metal,
        Mode::declared(),
        m,
        n,
    )
    .and_then(|stages| stages.one("降源"))
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
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，扫描内核编译用例未运行");
        return;
    }
    let (m, n) = nn::registry::model("scan_f32")
        .expect("scan_f32 语义表")
        .shape;
    let (local, correct) = program(
        &catalogue,
        &dirs,
        "scan_f32",
        Metal,
        Mode::segmented(),
        m,
        n,
    )
    .and_then(|stages| stages.two("扫描"))
    .expect("分块扫描内核应能生成");
    assert!(local.contains("kernel void kernel_scan_local"), "{local}");
    assert!(local.contains("thread_position_in_grid"), "{local}");
    assert!(local.contains("start + len"), "{local}");
    assert!(
        correct.contains("kernel void kernel_scan_correct"),
        "{correct}"
    );
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
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，两趟内核编译用例未运行");
        return;
    }
    let (m, n) = nn::registry::model("rmsnorm_f32")
        .expect("rmsnorm_f32 语义表")
        .shape;
    let msl = program(
        &catalogue,
        &dirs,
        "rmsnorm_f32",
        Metal,
        Mode::declared(),
        m,
        n,
    )
    .and_then(|stages| stages.one("降源"))
    .expect("两趟内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(!msl.contains("thread_position_in_grid"), "{msl}");
    let (partial, map) = program(
        &catalogue,
        &dirs,
        "rmsnorm_f32",
        Metal,
        Mode::segmented(),
        m,
        n,
    )
    .and_then(|stages| stages.two("归一化"))
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
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，softmax 内核编译用例未运行");
        return;
    }
    let (m, n) = nn::registry::model("softmax_f32")
        .expect("softmax_f32 语义表")
        .shape;
    let (max, expsum, map) = program(
        &catalogue,
        &dirs,
        "softmax_f32",
        Metal,
        Mode::segmented(),
        m,
        n,
    )
    .and_then(|stages| stages.three("softmax"))
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
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，gemv 内核编译用例未运行");
        return;
    }
    let msl = program(
        &catalogue,
        &dirs,
        "gemv_f32",
        Metal,
        Mode::declared(),
        64,
        4,
    )
    .and_then(|stages| stages.one("降源"))
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
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，gemm 内核编译用例未运行");
        return;
    }
    let msl = program(
        &catalogue,
        &dirs,
        "gemm_f32",
        Metal,
        Mode::declared(),
        16,
        16,
    )
    .and_then(|stages| stages.one("降源"))
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
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，attention 内核编译用例未运行");
        return;
    }
    let msl = program(
        &catalogue,
        &dirs,
        "attention_f32",
        Metal,
        Mode::declared(),
        8,
        8,
    )
    .and_then(|stages| stages.one("降源"))
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
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，flash_attention 内核编译用例未运行");
        return;
    }
    let msl = program(
        &catalogue,
        &dirs,
        "flash_attention_f32",
        Metal,
        Mode::declared(),
        8,
        8,
    )
    .and_then(|stages| stages.one("降源"))
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
    if !llm_optim::witness::device_available(&profile) {
        testkit::skip(format!(
            "engine: Metal 工具链不在本机，{op_name} 内核编译用例未运行"
        ));
        return;
    }
    let msl = program(&catalogue, &dirs, op_name, Metal, Mode::declared(), 64, 4)
        .and_then(|stages| stages.one("降源"))
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

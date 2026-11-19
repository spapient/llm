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

#[test]
fn the_metal_driver_takes_its_width_from_the_witness_limit() {
    let driver = metal_driver(MAX_OUT);
    assert!(driver.contains(&format!("#define WITNESS_MAX_OUT {MAX_OUT}")));
    // 一处定义 + 四处使用（上限判断那行出现两次）：每个宽度用点都必须走宏。
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

/// 设备生成文本的花括号必须配对。无 nvcc/Metal 后端的环境编不到设备端产物，CI 只能在文本层
/// 守住这一条：kernel 函数体与 switch 各闭合一次，多余的 `}` 会把后续声明顶到全局作用域。
#[test]
fn generated_device_program_braces_are_balanced() {
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-brace-work"));
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    for vendor in ["nvidia", "metal"] {
        let Some(profile) = catalogue.profile(vendor).cloned() else {
            continue;
        };
        let Some(front) = spec(profile.meta.emit) else {
            continue;
        };
        let selected = crate::witness::declared_witnesses(&profile).expect("见证");
        let text = (front.program)(&profile, &dirs, &selected, Scope::Device).expect("可生成");
        let open = text.chars().filter(|c| *c == '{').count();
        let close = text.chars().filter(|c| *c == '}').count();
        assert_eq!(
            open, close,
            "{vendor} 生成文本花括号不配对：{open} != {close}"
        );
    }
}

/// 生产态 Metal 网格内核：把一条真实逐点算子（silu_f32）走完整条流水线（见证 → 协商 → prepare →
/// 网格包装），产出 `kernel_main`，再通过 `xcrun metal` 编译，证明「模板 → 拼接正文 → 网格包装」
/// 产出的是合法 Metal（不是只有空体占位的结构断言）。走公开入口 [`execution::metal_grid_program`]，
/// 即 L7 组合根注入设备后端时用的那条链路。
#[test]
fn production_grid_kernel_compiles_for_a_real_op() {
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-metal-grid"));
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("metal").cloned().expect("metal 档案");
    if !crate::witness::device_available(&profile) {
        testkit::skip("cutile: Metal 工具链不在本机，生产态内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("silu_f32").expect("silu_f32 语义表").shape;
    let msl = crate::execution::metal_grid_program(&catalogue, &dirs, "silu_f32", m, n)
        .expect("生产态网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");

    let dir = std::env::temp_dir().join(format!("cutile-metal-grid-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("目录");
    let compiled = metal_kernel(&dir, &msl, &profile.flags);
    let _ = std::fs::remove_dir_all(&dir);
    compiled.expect("生产态 MSL 应能通过 xcrun metal 编译");
}

/// 生产态归约网格内核：`reduce_sum_f32` 走整条流水线后收敛成单线程 `kernel_main`（无
/// `thread_position_in_grid`，缓冲原样透传、标量取样本实际值），再通过 `xcrun metal` 编译。
#[test]
fn production_reduce_kernel_compiles_for_a_real_op() {
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-metal-reduce"));
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("metal").cloned().expect("metal 档案");
    if !crate::witness::device_available(&profile) {
        testkit::skip("cutile: Metal 工具链不在本机，归约内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("reduce_sum_f32")
        .expect("reduce_sum_f32 语义表")
        .shape;
    let msl = crate::execution::metal_grid_program(&catalogue, &dirs, "reduce_sum_f32", m, n)
        .expect("归约网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        !msl.contains("thread_position_in_grid"),
        "单线程归约不应有线程下标: {msl}"
    );

    let dir = std::env::temp_dir().join(format!("cutile-metal-reduce-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("目录");
    let compiled = metal_kernel(&dir, &msl, &profile.flags);
    let _ = std::fs::remove_dir_all(&dir);
    compiled.expect("归约 MSL 应能通过 xcrun metal 编译");
}

/// 生产态两趟内核：`rmsnorm_f32`（带 `@dot` 归约的逐元素骨架）走串行 lowering——单线程
/// `kernel_main`、无 `thread_position_in_grid`，再通过 `xcrun metal` 编译证明合法。
#[test]
fn production_rmsnorm_kernel_compiles_for_a_real_op() {
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-metal-rmsnorm"));
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("metal").cloned().expect("metal 档案");
    if !crate::witness::device_available(&profile) {
        testkit::skip("cutile: Metal 工具链不在本机，两趟内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("rmsnorm_f32").expect("rmsnorm_f32 语义表").shape;
    let msl = crate::execution::metal_grid_program(&catalogue, &dirs, "rmsnorm_f32", m, n)
        .expect("两趟内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        !msl.contains("thread_position_in_grid"),
        "单线程两趟不应有线程下标: {msl}"
    );

    let dir = std::env::temp_dir().join(format!("cutile-metal-rmsnorm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("目录");
    let compiled = metal_kernel(&dir, &msl, &profile.flags);
    let _ = std::fs::remove_dir_all(&dir);
    compiled.expect("两趟 MSL 应能通过 xcrun metal 编译");
}

/// 生产态 gemv 行网格内核：`gemv_f32` 走 `[grid]` 声明——模板循环 `for (int n = 0; n < N; ++n)`
/// 的头绑定到 `tid.x`（每线程一行），再通过 `xcrun metal` 编译证明合法。
/// 用小形状（m=64, n=4）避免按生产形状 4096×4096 分配 64MB 权重样本。
#[test]
fn production_gemv_kernel_compiles_for_a_real_op() {
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-metal-gemv"));
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("metal").cloned().expect("metal 档案");
    if !crate::witness::device_available(&profile) {
        testkit::skip("cutile: Metal 工具链不在本机，gemv 内核编译用例未运行");
        return;
    }
    let msl = crate::execution::metal_grid_program(&catalogue, &dirs, "gemv_f32", 64, 4)
        .expect("gemv 网格内核应能生成");

    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(msl.contains("thread_position_in_grid"), "{msl}");
    assert!(
        msl.contains("for (int n = (int)tid.x; n < (int)tid.x + 1; ++n)"),
        "行循环应绑定到 tid.x: {msl}"
    );

    let dir = std::env::temp_dir().join(format!("cutile-metal-gemv-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("目录");
    let compiled = metal_kernel(&dir, &msl, &profile.flags);
    let _ = std::fs::remove_dir_all(&dir);
    compiled.expect("gemv MSL 应能通过 xcrun metal 编译");
}

/// 生产态 gemm 二维网格内核：`gemm_f32` 走 `[grid]` 声明——模板循环 `for (int i = 0; i < M; ++i)`、
/// `for (int j = 0; j < N; ++j)` 的头分别绑定到 `tid.x` / `tid.y`（每线程一个输出元素），规约维 `p`
/// 循环保留；再通过 `xcrun metal` 编译证明合法。形状 [16,16]、K=8。
#[test]
fn production_gemm_kernel_compiles_for_a_real_op() {
    let dirs = Dirs::under(std::env::temp_dir().join("cutile-metal-gemm"));
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("metal").cloned().expect("metal 档案");
    if !crate::witness::device_available(&profile) {
        testkit::skip("cutile: Metal 工具链不在本机，gemm 内核编译用例未运行");
        return;
    }
    let msl = crate::execution::metal_grid_program(&catalogue, &dirs, "gemm_f32", 16, 16)
        .expect("gemm 网格内核应能生成");

    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(msl.contains("uint3 tid [[thread_position_in_grid]]"), "{msl}");
    assert!(
        msl.contains("for (int i = (int)tid.x; i < (int)tid.x + 1; ++i)"),
        "行循环应绑定到 tid.x: {msl}"
    );
    assert!(
        msl.contains("for (int j = (int)tid.y; j < (int)tid.y + 1; ++j)"),
        "列循环应绑定到 tid.y: {msl}"
    );

    let dir = std::env::temp_dir().join(format!("cutile-metal-gemm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("目录");
    let compiled = metal_kernel(&dir, &msl, &profile.flags);
    let _ = std::fs::remove_dir_all(&dir);
    compiled.expect("gemm MSL 应能通过 xcrun metal 编译");
}

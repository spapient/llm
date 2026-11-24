use diagnostic::data::Value;

/// 生成表按文件名升序（旧的 `prof::load(dir)` 也这样排序）。
#[test]
fn embedded_names_are_sorted() {
    let generated = crate::names();
    assert!(
        generated.windows(2).all(|pair| pair[0] < pair[1]),
        "{generated:?}"
    );
    for name in generated {
        assert!(crate::embedded(name).is_some(), "{name} 有名字却没有表");
    }
}

/// 每份生成表都是一份设备档案：有 `[meta]`，也有能力区间。
#[test]
fn every_embedded_entry_is_a_device_archive() {
    for name in crate::names() {
        let table = crate::embedded(name).expect("名字都能取到表");
        assert!(table.table("meta").is_some(), "{name} 缺 `[meta]`");
        let provides = table.array("provides").expect("档案必须有 `[[provides]]`");
        assert!(!provides.is_empty(), "{name} 没有任何能力区间");
        assert!(
            provides
                .iter()
                .all(|entry| matches!(entry, Value::Table(_))),
            "{name} 的 `provides` 元素必须是表"
        );
    }
}

#[test]
fn embedded_reports_a_missing_archive() {
    assert!(crate::embedded("no-such-device").is_none());
}

#[test]
fn select_picks_the_profile_matching_local_arch() {
    let profiles = crate::prof::builtin_profiles();
    let detected = crate::prof::Detected::new().with("arch", std::env::consts::ARCH);
    let chosen = crate::prof::select(profiles, &detected).expect("本机架构应命中一份档案");
    assert_eq!(chosen.meta.vendor, "cpu");
}

#[test]
fn select_reports_unsupported_when_nothing_matches() {
    let profiles = crate::prof::builtin_profiles();
    let detected = crate::prof::Detected::new().with("arch", "no-such-arch");
    let error = crate::prof::select(profiles, &detected).expect_err("无命中必须报错");
    assert_eq!(error.code(), diagnostic::Code::Unsupported);
}

/// 每个后端都实现完整算子矩阵，且只替换正文、不改变共享 ABI。
#[test]
fn every_device_kernel_matches_the_shared_operator_contract() {
    for vendor in crate::names() {
        let kernels = crate::catalogue::all_for(vendor);
        assert_eq!(
            kernels.len(),
            crate::OPS.len(),
            "`{vendor}` 的内核矩阵不完整"
        );
        for view in crate::OPS {
            let reference = crate::catalogue::named(view.name)
                .expect("CPU 参考算子")
                .to_decl()
                .expect("CPU 参考声明自洽");
            let specialized = crate::catalogue::named_for(vendor, view.name)
                .unwrap_or_else(|| panic!("`{vendor}` 缺 `{}`", view.name))
                .to_decl()
                .unwrap_or_else(|error| panic!("`{vendor}`/`{}`：{error}", view.name));
            assert_eq!(
                specialized.signature, reference.signature,
                "`{vendor}` ABI 漂移"
            );
            assert_eq!(
                specialized.needs, reference.needs,
                "`{vendor}` 能力需求漂移"
            );
            assert_eq!(
                specialized.layouts, reference.layouts,
                "`{vendor}` 布局漂移"
            );
        }
    }
}

/// 后端表按 `(vendor, op)` 严格升序，二分查找不会遇到重复键或乱序。
#[test]
fn device_templates_are_sorted_and_complete() {
    assert_eq!(
        crate::DEVICE_TEMPLATES.len(),
        crate::OPS.len() * crate::names().len()
    );
    assert!(
        crate::DEVICE_TEMPLATES
            .windows(2)
            .all(|pair| { (pair[0].0, pair[0].1) < (pair[1].0, pair[1].1) })
    );
    assert!(crate::catalogue::template_for("no-such-device", "add_f32").is_none());
    assert!(crate::catalogue::template_for("cpu", "no-such-op").is_none());
}

/// 发射选择真正读取后端调优结果；参考实现保持零拷贝，调优正文才拥有替换后的声明。
#[test]
fn device_specialization_selects_the_backend_without_changing_abi() {
    let op = crate::catalogue::named("add_f32")
        .expect("逐元素参考算子")
        .to_decl()
        .expect("参考声明");
    let cpu = crate::catalogue::for_device(&op, "cpu").expect("CPU 内核存在");
    assert!(matches!(cpu, std::borrow::Cow::Borrowed(_)));

    let nvidia = crate::catalogue::for_device(&op, "nvidia").expect("NVIDIA 内核存在");
    assert!(matches!(nvidia, std::borrow::Cow::Owned(_)));
    assert_eq!(nvidia.signature, op.signature);
    assert!(nvidia.template.contains("#pragma unroll 4"));
    assert_eq!(
        nvidia.template,
        crate::catalogue::template_for("nvidia", &op.name).expect("NVIDIA 正文")
    );

    let error = crate::catalogue::for_device(&op, "no-such-device").expect_err("未知后端必须失败");
    assert_eq!(error.code(), diagnostic::Code::Missing);
}

/// 未特化且未调优的算子在所有后端共享参考正文，克隆声明不会引入字符串分配。
#[test]
fn untuned_kernels_reuse_the_shared_reference_declaration() {
    let op = crate::catalogue::named("gemm_f32")
        .expect("未调优的参考算子")
        .to_decl()
        .expect("参考声明");
    for vendor in crate::names() {
        let selected = crate::catalogue::for_device(&op, vendor).expect("有效后端内核");
        assert!(
            matches!(selected, std::borrow::Cow::Borrowed(_)),
            "`{vendor}` 应零拷贝复用参考声明"
        );
    }
}

/// 展开配置以模板 stem 为键：同模板算子共享策略，不同结构不会因声明继承而被误展开。
#[test]
fn unroll_configuration_follows_the_template_stem() {
    const HOT: &[&str] = &[
        "add_f32",
        "mul_f32",
        "abs_f32",
        "sin_f32",
        "matmul_f32",
        "gemm_i4",
        "gemm_i2",
        "gemm_i8",
        "matmul_i8",
    ];
    for name in HOT {
        let reference = crate::catalogue::template_for("cpu", name).expect("CPU 参考正文");
        assert!(reference.contains("/* cutile-unroll */"), "`{name}` 缺锚点");
        assert!(
            !reference.contains("#pragma unroll"),
            "`{name}` 的参考档不应展开"
        );
        for vendor in crate::names().iter().copied().filter(|name| *name != "cpu") {
            let source = crate::catalogue::template_for(vendor, name).expect("后端有效正文");
            if source.contains("cublasSgemm") {
                assert_eq!(*name, "matmul_f32");
                assert!(vendor.starts_with("nvidia"));
                assert!(!source.contains("/* cutile-unroll */"));
                assert!(!source.contains("#pragma unroll"));
                continue;
            }
            assert_eq!(
                source.matches("#pragma unroll 4").count(),
                1,
                "`{vendor}`/`{name}` 必须且只能展开一次"
            );
        }
    }

    for name in ["adaptive_pool_f32", "remainder_f32", "gemv_i4"] {
        for vendor in crate::names() {
            let source = crate::catalogue::template_for(vendor, name).expect("后端有效正文");
            assert!(
                !source.contains("#pragma unroll"),
                "`{vendor}`/`{name}` 不应继承其他模板的展开策略"
            );
        }
    }
}

/// MUSA 特化必须保持共享 ABI，并保留真实设备内核、分块同步与三侧边界守卫。
/// 这里只检查源码契约；是否可由 mcc 编译及真机数值/性能由设备测试另行证明。
#[test]
fn musa_matmul_specialization_preserves_abi_and_device_contract() {
    let reference = crate::catalogue::named("matmul_f32")
        .expect("共享矩阵乘声明")
        .to_decl()
        .expect("共享声明有效");
    let specialized = crate::catalogue::for_device(&reference, "musa").expect("MUSA 矩阵乘特化");
    assert!(matches!(specialized, std::borrow::Cow::Owned(_)));
    assert_eq!(specialized.signature, reference.signature);
    assert_eq!(specialized.needs, reference.needs);
    assert_eq!(specialized.layouts, reference.layouts);

    let source = &specialized.template;
    for required in [
        "__global__",
        "__shared__",
        "__syncthreads()",
        "<<<grid, block>>>",
        "row < M && k < K",
        "bk < K && col < N",
        "row < M && col < N",
        "cudaGetLastError()",
        "cudaDeviceSynchronize()",
    ] {
        assert!(source.contains(required), "MUSA 特化缺 `{required}`");
    }
    assert_eq!(source.matches("#pragma unroll 4").count(), 1);
    assert!(!source.contains("#include <wmma"));
    assert!(!source.contains("nvcuda::wmma"));
}

/// NVIDIA 使用严格 FP32 cuBLAS，缓存按设备失效且并发安全；公开 ABI 仍与共享声明一致。
#[test]
fn nvidia_matmul_uses_pedantic_cublas_and_cached_device_storage() {
    let reference = crate::catalogue::named("matmul_f32")
        .expect("共享矩阵乘声明")
        .to_decl()
        .expect("共享声明有效");
    let specialized =
        crate::catalogue::for_device(&reference, "nvidia").expect("NVIDIA cuBLAS 矩阵乘特化");
    assert!(matches!(specialized, std::borrow::Cow::Owned(_)));
    assert_eq!(specialized.signature, reference.signature);
    assert_eq!(specialized.needs, reference.needs);
    assert_eq!(specialized.layouts, reference.layouts);

    let source = &specialized.template;
    for required in [
        "#include <cublas_v2.h>",
        "CUBLAS_PEDANTIC_MATH",
        "CUBLAS_POINTER_MODE_HOST",
        "std::mutex",
        "std::lock_guard<std::mutex>",
        "cudaGetDevice(&device)",
        "state.device != device",
        "cudaMalloc",
        "cublasSgemm",
        "N, M, K, &alpha, state.b, N, state.a, K, &beta, state.y, N",
        "cudaMemcpyDeviceToHost",
    ] {
        assert!(
            source.contains(required),
            "NVIDIA cuBLAS 特化缺 `{required}`"
        );
    }
    assert_eq!(source.matches("cublasSgemm(").count(), 1);
    assert!(!source.contains("CUBLAS_COMPUTE_32F_FAST_TF32"));
    assert!(!source.contains("cudaDeviceSynchronize"));

    let profile = crate::prof::builtin_profiles()
        .into_iter()
        .find(|profile| profile.meta.vendor == "nvidia")
        .expect("NVIDIA 档案");
    assert!(profile.flags.iter().any(|flag| flag == "-lcublas"));
}

/// A3000 是独立指纹与编译架构，但复用 NVIDIA 的 cuBLAS 正文和片段，不复制设备族语料。
#[test]
fn a3000_variant_selects_sm86_and_reuses_the_nvidia_backend() {
    let detected = crate::prof::Detected::new()
        .with("cc", "8.6")
        .with("name", "NVIDIA RTX A3000 12GB Laptop GPU");
    let chosen = crate::prof::select(crate::prof::builtin_profiles(), &detected)
        .expect("A3000 实测指纹必须命中变体");
    assert_eq!(chosen.meta.vendor, "nvidia_a3000");
    assert_eq!(chosen.meta.id, "sm86");
    assert_eq!(chosen.mem.fast_bytes, 49_152);
    assert_eq!(chosen.mem.reg_bytes, 262_144);
    assert_eq!(chosen.simt.group, 32);
    assert_eq!(chosen.simt.groups_max, 1_536);
    assert!(chosen.flags.iter().any(|flag| flag == "-arch=sm_86"));
    assert!(chosen.flags.iter().any(|flag| flag == "-lcublas"));
    assert!(!chosen.flags.iter().any(|flag| flag == "-arch=sm_75"));
    assert!(!chosen.preamble.is_empty(), "必须继承 NVIDIA 前置声明");
    assert!(!chosen.provides.is_empty(), "必须继承 NVIDIA 原语供给");

    for name in ["matmul_f32", "add_f32", "gemm_i8"] {
        assert_eq!(
            crate::catalogue::template_for("nvidia_a3000", name),
            crate::catalogue::template_for("nvidia", name),
            "`{name}` 必须复用 NVIDIA 后端正文"
        );
    }
    assert!(crate::device::dir_of("nvidia_a3000").is_some_and(|root| root.ends_with("/nvidia")));

    let t600 = crate::prof::Detected::new()
        .with("cc", "7.5")
        .with("name", "NVIDIA T600");
    let chosen = crate::prof::select(crate::prof::builtin_profiles(), &t600)
        .expect("T600 仍必须命中原档案");
    assert_eq!(chosen.meta.vendor, "nvidia");
}

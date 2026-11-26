//! `cutile::build` 的用例：编译参数与产物指纹。
//!
//! 生产文件只留执行与判据，用例集中于此。

use super::*;
use crate::fixtures::profile;
#[cfg(unix)]
use crate::toolchain;
use nn::model::Sample;

/// 只改 owned 声明里的某一处（模板不变）：用来构造"缓存键不变、判定结果变了"的输入。
fn declared(mutate: impl FnOnce(&mut OpDecl)) -> OpDecl {
    let mut op = builtin();
    mutate(&mut op);
    op
}

/// 把所有搬运原语的 `bytes` 上限改到 `limit`。
fn declared_bytes(limit: u32) -> OpDecl {
    declared(|op| {
        for need in &mut op.needs {
            if matches!(need.prim, Prim::Load | Prim::Store | Prim::LoadAsync) {
                need.extent = cutile::atom::Span::at_most(limit);
            }
        }
    })
}

fn declared_swizzle() -> OpDecl {
    declared(|op| {
        for (slot, layout) in &mut op.layouts {
            if slot == "w" {
                *layout = LayoutSpec::Swz { b: 40, m: 3, s: 3 };
            }
        }
    })
}

fn builtin() -> OpDecl {
    prof::catalogue::named("gemv_i4")
        .expect("内置")
        .to_decl()
        .expect("内置声明可解析")
}

fn cpu_path(op: &OpDecl) -> Path {
    cutile::cap::negotiate(op, &profile("cpu")).expect("CPU 应命中首档")
}

/// 不经协商的路径：契约测试要故意构造协商会拒绝的声明。
fn declared_path(op: &OpDecl) -> Path {
    let variant = &op.variants[0];
    Path {
        variant: variant.name.clone(),
        idx: 0,
        schedule: variant.schedule,
        mma_share: variant.mma_share,
    }
}

fn first_tile(op: &OpDecl) -> TileCombo {
    cutile::decl::expand(op)
        .expect("轴可展开")
        .into_iter()
        .next()
        .expect("至少一个 tile 取值")
}

fn dirs(root: &std::path::Path) -> Dirs {
    Dirs::under(root)
}

testkit::test_cases! {
fn declared_traffic_beyond_the_fast_capacity_fails_the_build() {
    let op = declared_bytes(1_048_576);
    let error = contract(&op, &profile("cpu"), &declared_path(&op)).unwrap_err();
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.to_string().contains("FAST"), "{error}");
    assert!(error.to_string().contains("524288"), "{error}");
}

fn swizzle_beyond_the_bank_count_fails_the_build() {
    let op = declared_swizzle();
    let error = contract(&op, &profile("cpu"), &declared_path(&op)).unwrap_err();
    assert_eq!(error.code(), Code::Unsupported);
    assert!(error.to_string().contains("布局槽位 `w`"), "{error}");
}

fn the_contract_reports_the_numbers_it_compared() {
    let op = builtin();
    let uses = contract(&op, &profile("cpu"), &cpu_path(&op)).expect("内置声明在容量内");
    assert!(
        uses.iter().any(|usage| usage.prim == Prim::Load
            && usage.space == Space::Fast
            && usage.bytes == 4096),
        "{uses:?}"
    );
    assert!(
        uses.iter().all(|usage| usage.space != Space::Reg),
        "没有声明字节数的原语不该凭空占用 REG：{uses:?}"
    );
}

fn a_cached_source_cannot_exempt_the_declaration_from_the_contract() {
    // 声明的字节数只影响判定、不影响源文本，所以它不在缓存键里。先按合法声明落一次盘，
    // 再只把字节数改到超出容量：键不变、缓存必然命中——契约仍必须重新判定。
    let root = std::env::temp_dir().join(format!("cutile-contract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dirs = dirs(&root);
    let profile = profile("cpu");
    let op = builtin();
    let path = cpu_path(&op);
    let tile = first_tile(&op);
    source(&op, &profile, &path, &tile, &dirs).expect("合法声明可构建");

    let invalid = declared_bytes(1_048_576);
    let error = source(&invalid, &profile, &path, &tile, &dirs).unwrap_err();
    assert!(
        error.to_string().contains("524288"),
        "缓存命中不能跳过构建期契约：{error}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
}

/// 本机实测的工具链（本机没有编译器时返回 `None`，用例跳过）。
fn tool() -> Option<Toolchain> {
    crate::toolchain::detect()
}

/// 一份带主函数的内核源码与形状：产物身份要用的两个输入。
fn built(op: &OpDecl, dirs: &Dirs) -> (CacheKey, String, Sample) {
    let profile = profile("cpu");
    let path = cpu_path(op);
    let tile = first_tile(op);
    let (source, key) = source(op, &profile, &path, &tile, dirs).expect("可构建");
    let sample = (nn::registry::model(&op.name).expect("算子模型").sample)(32, 1);
    op.signature.check_sample(&sample).expect("ABI 对齐");
    let harness =
        cutile::harness::harness(&source.text, &op.signature, &sample, 1).expect("可生成");
    (key, harness, sample)
}

testkit::test_cases! {
#[cfg(unix)]
fn a_second_execution_needs_neither_the_compiler_nor_a_rebuild() {
    // 用一份"转调真正的 cc"的脚本充当编译器：第一次真的编，删掉脚本后第二次必须命中
    // 产物——真去找编译器就会以 `io` 失败，于是"没调用编译器"这件事是可观测的。
    let Some(tool) = tool() else {
        return;
    };
    let root = std::env::temp_dir().join(format!("cutile-execute-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dirs = dirs(&root);
    let profile = profile("cpu");
    let op = builtin();
    let (key, harness, sample) = built(&op, &dirs);

    let wrapper = root.join("fake-cc.sh");
    std::fs::create_dir_all(&root).expect("临时目录");
    std::fs::write(
        &wrapper,
        format!("#!/bin/sh\nexec {} \"$@\"\n", tool.command),
    )
    .expect("写脚本");
    let mut perms = std::fs::metadata(&wrapper).expect("脚本").permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
    std::fs::set_permissions(&wrapper, perms).expect("可执行");
    // 身份也是探测出来的：脚本 `--version` 转调真正的编译器，所以版本是真的，
    // 命令名是脚本路径——两者合起来就是这份产物的来历。
    let wrapper = toolchain::probe(&wrapper.to_string_lossy()).expect("脚本可探测");

    let (first, artifact) =
        execute(&profile, Some(&wrapper), &key, &harness, &dirs).expect("第一次应真的编译");
    assert!(!artifact.reused, "{artifact:?}");
    assert_eq!(first.bits.len(), sample.n);
    assert_eq!(
        artifact.key.tool,
        wrapper.id(),
        "产物身份必须带上实测工具链"
    );

    std::fs::remove_file(&wrapper.command).expect("删掉编译器");
    let (second, hit) =
        execute(&profile, Some(&wrapper), &key, &harness, &dirs).expect("应复用产物");
    assert!(hit.reused, "{hit:?}");
    assert_eq!(second.bits, first.bits, "复用同一份产物，结果必须逐位相同");

    // `None` 就是"这台机器上没有编译器"：产物旁的身份文件足以认回同一份文件。
    let (third, from_identity) =
        execute(&profile, None, &key, &harness, &dirs).expect("没有编译器也应复用产物");
    assert!(from_identity.reused, "{from_identity:?}");
    assert_eq!(from_identity.path, hit.path, "按身份找到的还是同一份产物");
    assert_eq!(third.bits, first.bits, "同一份产物，结果逐位相同");
    let _ = std::fs::remove_dir_all(&root);
}

#[cfg(unix)]
fn a_new_compiler_version_does_not_reuse_the_old_artifact() {
    // 同一份源文本、同一个主函数，只把 `--version` 换掉：必须重编。这是"产物身份带实测
    // 工具链"的可观测后果——此前身份只有命令名，升级编译器会命中旧代码生成。
    let Some(tool) = tool() else {
        return;
    };
    let root = std::env::temp_dir().join(format!("cutile-version-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dirs = dirs(&root);
    std::fs::create_dir_all(&root).expect("临时目录");
    let profile = profile("cpu");
    let op = builtin();
    let (key, harness, _) = built(&op, &dirs);

    let fake = |name: &str, version: &str| {
        let path = root.join(name);
        std::fs::write(
        &path,
        format!(
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'Fake clang version {version}'; exit 0; fi\nexec {} \"$@\"\n",
            tool.command
        ),
    )
    .expect("写脚本");
        let mut perms = std::fs::metadata(&path).expect("脚本").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
        std::fs::set_permissions(&path, perms).expect("可执行");
        toolchain::probe(&path.to_string_lossy()).expect("脚本可探测")
    };
    let old = fake("cc17.sh", "17.0.0");
    let new = fake("cc21.sh", "21.0.0");
    assert_ne!(old.id(), new.id(), "前提：两条工具链身份不同");

    let (first, older) = execute(&profile, Some(&old), &key, &harness, &dirs).expect("第一次编译");
    assert!(!older.reused, "{older:?}");
    let (second, newer) = execute(&profile, Some(&new), &key, &harness, &dirs).expect("第二次编译");
    assert!(!newer.reused, "换工具链版本必须重新编译：{newer:?}");
    assert_ne!(older.path, newer.path, "两份产物不能共用路径");
    assert_eq!(first.bits, second.bits, "语义相同，结果必须逐位相同");

    // 再跑一次新版本：这次才该复用。
    let (_, again) = execute(&profile, Some(&new), &key, &harness, &dirs).expect("应复用");
    assert!(again.reused, "{again:?}");
    assert_eq!(again.path, newer.path);
    let _ = std::fs::remove_dir_all(&root);
}

/// 换一组编译参数不许复用旧产物：见证没有 `CacheKey`，参数只能由调用方给；少了这一项，档案里改一次
/// `[compile].flags` 就会复用一个不再会被编出来的程序。
#[cfg(unix)]
fn a_new_flag_set_does_not_reuse_the_witness_binary() {
    let Some(tool) = tool() else {
        return;
    };
    let root = std::env::temp_dir().join(format!("cutile-flags-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dirs = dirs(&root);
    let program = "#include <stdio.h>\nint main(void){ printf(\"0\\n\"); return 0; }\n";
    let args = |values: &[&str]| -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    };
    let o2 = args(&["-O2", "-std=c99"]);
    let o0 = args(&["-O0", "-std=c99"]);

    let (_, first) = run_program(Some(&tool), &o2, "flags", program, &dirs).expect("编译");
    assert!(!first.reused, "{first:?}");
    let (_, again) = run_program(Some(&tool), &o2, "flags", program, &dirs).expect("复用");
    assert!(again.reused, "同一组参数应复用：{again:?}");

    let (_, relaid) = run_program(Some(&tool), &o0, "flags", program, &dirs).expect("编译");
    assert!(!relaid.reused, "换参数必须重新编译：{relaid:?}");
    assert_ne!(relaid.path, again.path, "两份产物不能共用路径");
    assert_ne!(
        relaid.key.params, again.key.params,
        "参数必须是产物身份的一部分"
    );

    // 没有编译器时也不许拿另一组参数的产物顶替：那不是"同一份产物"的副本。
    let error = run_program(None, &args(&["-O1"]), "flags", program, &dirs).unwrap_err();
    assert_eq!(error.code(), Code::Unsupported, "{error}");
    assert!(error.to_string().contains("编译参数"), "{error}");
    let _ = std::fs::remove_dir_all(&root);
}

fn the_artifact_cache_leaves_the_work_directory_empty() {
    let Some(tool) = tool() else {
        return;
    };
    let root = std::env::temp_dir().join(format!("cutile-sandbox-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dirs = dirs(&root);
    let op = builtin();
    let (key, harness, _) = built(&op, &dirs);
    execute(&profile("cpu"), Some(&tool), &key, &harness, &dirs).expect("应能编译执行");
    let leftovers: Vec<_> = std::fs::read_dir(dirs.work())
        .map_or_default(|items| items.flatten().map(|item| item.file_name()).collect());
    assert!(
        leftovers.is_empty(),
        "产物拷进缓存后沙箱就该删掉，否则每次编译都留一份可执行文件：{leftovers:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

fn without_a_compiler_a_missing_artifact_is_an_explicit_failure() {
    let root = std::env::temp_dir().join(format!("cutile-no-artifact-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dirs = dirs(&root);
    let op = builtin();
    let (key, harness, _) = built(&op, &dirs);

    let error = execute(&profile("cpu"), None, &key, &harness, &dirs).unwrap_err();
    assert_eq!(error.code(), Code::Unsupported, "{error}");
    assert!(error.to_string().contains("没有这份产物"), "{error}");
    let _ = std::fs::remove_dir_all(&root);
}

#[cfg(unix)]
fn two_compilers_worth_of_artifacts_are_not_chosen_between() {
    // 没有编译器时只在"唯一一份"产物上复用：多份候选就是替用户选编译器，也就替 `bench`
    // 决定了耗时的来历。用两份假产物把这条判定单独钉住（不需要真编译器）。
    let root = std::env::temp_dir().join(format!("cutile-ambiguous-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dirs = dirs(&root);
    let op = builtin();
    let (key, harness, _) = built(&op, &dirs);
    let cache = Cache::from_dirs(&dirs);

    let exe = root.join("fake-exe");
    std::fs::create_dir_all(&root).expect("临时目录");
    std::fs::write(&exe, b"#!/bin/sh\nexit 0\n").expect("写假产物");
    let mut perms = std::fs::metadata(&exe).expect("产物").permissions();
    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
    std::fs::set_permissions(&exe, perms).expect("可执行");
    for candidate in ["cc@17.0.0", "clang@18.1.3"] {
        cache
            .put_artifact(
                &ArtifactKey::keyed(key.hex(), candidate, key.flags_hash, &harness),
                &exe,
            )
            .expect("落产物");
    }

    let error = execute(&profile("cpu"), None, &key, &harness, &dirs).unwrap_err();
    assert_eq!(error.code(), Code::Unsupported, "{error}");
    assert!(error.to_string().contains("cc@17.0.0"), "{error}");
    assert!(error.to_string().contains("clang@18.1.3"), "{error}");
    let _ = std::fs::remove_dir_all(&root);
}

/// 同一参考声明经后端调优后，实际发射正文与 AST 缓存指纹同步变化。
fn source_applies_backend_tuning_and_separates_cache_fingerprints() {
    let root = std::env::temp_dir().join(format!("cutile-backend-kernel-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dirs = dirs(&root);
    let op = prof::catalogue::named("add_f32")
        .expect("热点参考算子")
        .to_decl()
        .expect("参考声明可解析");
    let path = declared_path(&op);
    let tile = first_tile(&op);
    let cpu = profile("cpu");
    let nvidia =
        Profile::parse(prof::embedded("nvidia").expect("NVIDIA 档案")).expect("NVIDIA 档案可解析");

    let (host, host_key) = source(&op, &cpu, &path, &tile, &dirs).expect("CPU 可发射");
    let (device, device_key) = source(&op, &nvidia, &path, &tile, &dirs).expect("NVIDIA 可发射");
    assert!(!host.text.contains("#pragma unroll"));
    assert_eq!(device.text.matches("#pragma unroll 4").count(), 1);
    assert_ne!(host_key.ast_hash, device_key.ast_hash);

    let _ = std::fs::remove_dir_all(&root);
}
}

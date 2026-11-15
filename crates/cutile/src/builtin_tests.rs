use super::*;
use crate::decl::expand;
use crate::probe;
use crate::substitute;

#[test]
fn builtin_data_is_valid() {
    let catalogue = Catalogue::try_builtin().expect("内置数据必须合法");
    assert_eq!(catalogue.ops().len(), op::REGISTRY.len());
    assert_eq!(catalogue.profiles().len(), prof::names().len());
}

#[test]
fn cached_catalogue_is_identical_across_calls() {
    let first = Catalogue::try_builtin().expect("内置数据");
    let second = Catalogue::try_builtin().expect("内置数据");
    assert_eq!(first, second, "进程内缓存返回的必须是同一份目录的克隆");
}

#[test]
fn builtin_witness_files_exist() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    catalogue
        .validate(Some(&prim::dir()))
        .expect("见证文件必须存在");
}

/// 每台设备的每条 `provides` 都要有见证声明；等级不是门槛：`L0`/`L1` 的声明同样要给出"证据是什么"。
#[test]
fn every_provide_declares_a_witness() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    for profile in catalogue.profiles() {
        let vendor = profile.meta.vendor.as_str();
        for provide in &profile.provides {
            assert!(
                provide.witness.is_some(),
                "`{vendor}`/`{}` 缺少见证声明",
                provide.prim
            );
        }
    }
}

/// 见证矩阵：闭集的 16 个原语在**每台设备**上都要有带见证的签名。只守"P0 用到的 9 个"会留下
/// 自相矛盾的档案。空白格只在数据能解释时才合法（如单 lane 设备没有子组内交换的对象，也就不声明
/// `shuffle`/`broadcast`）；其余补上的条目等级都是 `L0`——矩阵补的是覆盖，不是等级。
#[test]
fn every_device_covers_the_primitive_set_or_explains_the_gap() {
    use crate::atom::Prim;
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    for profile in catalogue.profiles() {
        let vendor = profile.meta.vendor.as_str();
        for prim in Prim::ALL.iter().copied() {
            if profile.simt.declared
                && profile.simt.group <= 1
                && matches!(prim, Prim::Shuffle | Prim::Broadcast)
            {
                assert!(
                    profile.providers(prim).next().is_none(),
                    "`{vendor}` 声明了 `[simt].group = 1`，却又有 `{prim}`"
                );
                continue;
            }
            let mut provides = profile.providers(prim).peekable();
            assert!(
                provides.peek().is_some(),
                "`{vendor}` 没有声明 `{prim}`：闭集的空白格等于没有证据"
            );
            assert!(
                provides.all(|provide| provide.witness.is_some()),
                "`{vendor}` 的 `{prim}` 有一条签名没有见证"
            );
        }
    }
}

/// 片段层与模板的调用签名对齐：每条 `@原语(...)` 都能被该设备的片段拼出来。参数个数或命名参数对不上
/// 时，`substitute` 要到构建期才报错，那时离写错的地方已经很远。
#[test]
fn every_device_fragment_set_splices_the_p0_template() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let op = catalogue.op("gemv_i4").expect("内置算子");
    let combos = expand(op).expect("tile 候选");
    for profile in catalogue.profiles() {
        let vendor = profile.meta.vendor.as_str();
        let ext = profile.meta.emit.ext();
        let fragments = prim::builtin(vendor, ext)
            .unwrap_or_else(|| panic!("`{vendor}` 的片段层没有按 `. {ext}` 登记"));
        for combo in &combos {
            if let Err(error) = substitute::render(op, combo, &fragments) {
                panic!("`{vendor}` 的片段与模板签名不同构（{combo}）：{error}");
            }
        }
    }
}

/// 片段层声明的实现，必须正是协商会为这台设备挑中的那条。拼错一条不会让拼接失败，只会让生成的程序算别的算法。
#[test]
fn every_fragment_declares_the_implementation_negotiation_picks() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    for profile in catalogue.profiles() {
        let vendor = profile.meta.vendor.as_str();
        let set = prim::builtin(vendor, profile.meta.emit.ext())
            .unwrap_or_else(|| panic!("`{vendor}` 的片段层没有登记"));
        for op in catalogue.ops() {
            let gaps = cap::fragment_gaps(op, profile, |prim| set.implementation(prim));
            assert!(gaps.is_empty(), "`{vendor}`/`{}`：{gaps:?}", op.name);
        }
    }
}

/// 声明错一条签名就是装载期错误：不必等到构建期，更不能静默用错算法。
#[test]
fn a_fragment_declaring_another_implementation_fails_the_load() {
    let root = std::env::temp_dir().join(format!("cutile-sign-{}", std::process::id()));
    let prim_root = root.join("prim");
    copy_dir(&prim::dir(), &prim_root);
    // `nvidia` 的 `dot` 协商会挑中 `fma`；把文本改成声明 `dp4a`（另一条签名，属 P3）。
    let path = prim_root.join("nvidia").join("dot.cu");
    let text = std::fs::read_to_string(&path).expect("片段可读");
    assert!(text.contains("dot/fma"), "{text}");
    std::fs::write(&path, text.replacen("dot/fma", "dot/dp4a", 1)).expect("写片段");

    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let errors = catalogue
        .validate(Some(&prim_root))
        .expect_err("片段声明另一条签名必须报错");
    let rendered: Vec<String> = errors.iter().map(ToString::to_string).collect();
    let rendered = rendered.join("\n");
    assert!(rendered.contains("`nvidia`/`gemv_i4`"), "{rendered}");
    assert!(rendered.contains("dp4a"), "{rendered}");
    assert!(rendered.contains("fma"), "{rendered}");
    std::fs::remove_dir_all(&root).ok();
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("建目录");
    for entry in std::fs::read_dir(from).expect("读目录") {
        let entry = entry.expect("目录项");
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("复制文件");
        }
    }
}

#[test]
fn a_declaration_name_must_match_its_file_name() {
    let source = op::builtin_named("add_f32").expect("内置");
    let op = Op::new("renamed", source.declaration(), source.template());
    let profiles = Catalogue::try_builtin().expect("内置数据").profiles;
    let error = Catalogue::assemble(vec![op], profiles).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("文件名即算子名"), "{error}");
}

/// 空片段层根下每条失败都是 `missing`：类别由数据层给，装载层不再一律压成 `invalid`。
#[test]
fn validation_keeps_the_category_of_the_failing_layer() {
    let root = std::env::temp_dir().join(format!("cutile-prim-empty-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("建临时目录");
    let errors = Catalogue::try_builtin()
        .expect("内置数据")
        .validate(Some(&root))
        .unwrap_err();
    assert!(!errors.is_empty(), "空片段层根必须报错");
    assert!(
        errors
            .iter()
            .any(|error| error.code() == Code::Missing && error.to_string().contains("见证文件")),
        "见证文件不在应是 `missing`：{errors:?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.code() == Code::Missing && error.to_string().contains("读取")),
        "片段层目录不在应是 `missing`，不是 `invalid`：{errors:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn duplicate_profiles_are_rejected() {
    let mut catalogue = Catalogue::try_builtin().expect("内置数据");
    let clone = catalogue.profiles[0].clone();
    catalogue.profiles.push(clone);
    let errors = catalogue.validate(Some(&prim::dir())).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.to_string().contains("两份档案"))
    );
}

#[test]
fn detects_the_local_cpu_profile() {
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let detected = probe::detect_cpu();
    let profile = catalogue.select(&detected).expect("必须命中 CPU 档案");
    assert_eq!(profile.meta.vendor, "cpu");
}

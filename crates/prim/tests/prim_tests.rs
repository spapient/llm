use super::*;
use diagnostic::Code;
use std::path::{Path, PathBuf};

/// 临时目录：`<tmp>/prim-<name>-<pid>`。
fn temp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("prim-{name}-{}", std::process::id()))
}

/// 内嵌表与磁盘一致：生成器漏扫一个目录，这里就会失败。
#[test]
fn builtin_matches_disk() {
    for entry in BUILTIN {
        let dir = dir_of(entry.vendor).expect("每台设备都有源码根");
        let disk = load(Path::new(dir), entry.vendor, entry.ext).expect("磁盘片段可装载");
        assert_eq!(
            disk.len(),
            entry.fragments.len(),
            "`{}` 的磁盘片段数与内嵌不一致",
            entry.vendor
        );
        for (prim, source) in entry.fragments {
            assert_eq!(
                disk.fragment(*prim).map(Fragment::source),
                Some(*source),
                "`{}` 的 `{prim}` 与磁盘内容不一致",
                entry.vendor
            );
        }
    }
}

#[test]
fn lookup_needs_both_vendor_and_ext() {
    assert!(builtin("cpu", "c").is_some());
    assert!(builtin("cpu", "cu").is_none(), "扩展名也参与登记键");
    assert!(builtin("nvidia", "c").is_none(), "未登记的设备走磁盘");
}

#[test]
fn builtin_witness_matches_disk() {
    let layout = Layout::builtin();
    for (vendor, ext, source) in WITNESS {
        let path = layout
            .witness(&format!("{vendor}/witness.{ext}"))
            .expect("每台设备都有见证路径");
        let disk = std::fs::read_to_string(&path).expect("磁盘见证可读");
        assert_eq!(disk, source, "`{vendor}` 的内嵌见证与磁盘不一致");
    }
}

#[test]
fn witness_is_not_a_fragment() {
    let cpu = BUILTIN
        .iter()
        .find(|entry| entry.vendor == "cpu")
        .expect("内置 cpu 片段");
    let dir = dir_of("cpu").expect("cpu 有源码根");
    let set = load(Path::new(dir), "cpu", "c").expect("witness.c 必须被跳过");
    assert_eq!(set.len(), cpu.fragments.len());
    assert!(set.get(Prim::Load).is_some());
}

/// 四张表按下标同源：`DEVICES` 的一行、`BUILTIN` 的一项、`WITNESS` / `ROOTS` 的一项必须指同一台设备。
///
/// 合成是手写的 `const` 数组：漏行编译器会数出来，但写错顺序不会——这条用例守的就是顺序。
#[test]
fn the_tables_line_up_by_index() {
    let len = devices().len();
    assert_eq!(BUILTIN.len(), len, "片段表与设备表不等长");
    assert_eq!(WITNESS.len(), len, "见证表与设备表不等长");
    assert_eq!(ROOTS.len(), len, "源码根表与设备表不等长");
    for (index, device) in devices().iter().enumerate() {
        assert_eq!(BUILTIN[index].vendor, device.vendor);
        assert_eq!(WITNESS[index].0, device.vendor);
        assert_eq!(ROOTS[index].0, device.vendor);
    }
}

/// 每台设备的源码根必须真的存在，且根里有它的见证：拆成六个 crate 之后，"片段层根"是六条
/// 路径，写错一条就是发布机读不到见证——这里在测试期把它读一遍。
#[test]
fn every_source_root_holds_its_witness() {
    let layout = Layout::builtin();
    for device in devices() {
        assert!(dir_of(device.vendor).is_some(), "每台设备都要有源码根");
        let file = format!("{}/witness.{}", device.vendor, device.ext);
        let path = layout.witness(&file).expect("内置见证有路径");
        assert_eq!(
            std::fs::read_to_string(&path).ok().as_deref(),
            Some(device.source),
            "`{}` 的见证不在它的源码根里",
            device.vendor
        );
    }
}

/// `data.root` 覆盖：`<root>/prim/<vendor>/` 下的同名文件替换内置片段，见证也走同一套解析。
#[test]
fn a_data_root_layout_reads_the_override() {
    let root = temp("layout");
    let vendor_dir = layer_root(&root).join("cpu");
    std::fs::create_dir_all(&vendor_dir).expect("建临时目录");
    std::fs::write(
        vendor_dir.join("load.c"),
        "/* cutile-sign: load/override */\n{ $0 = 1; }\n",
    )
    .expect("写片段");
    let layout = Layout::at(&root);
    let set = layout.fragments("cpu", "c").expect("覆盖层可装载");
    assert_eq!(set.len(), 1, "覆盖层只有写进去的那一份");
    assert_eq!(set.implementation(Prim::Load), Some("override"));
    assert_eq!(
        layout.witness("cpu/witness.c"),
        Some(vendor_dir.join("witness.c"))
    );
    std::fs::remove_dir_all(&root).ok();
}

/// 不认识的 vendor 没有布局：`missing` 带类别，不是"读到了空目录"。
#[test]
fn an_unknown_vendor_has_no_layout() {
    let layout = Layout::builtin();
    let error = layout
        .fragments("absent", "c")
        .expect_err("未知 vendor 必须报错");
    assert_eq!(error.code(), Code::Missing, "{error}");
    assert!(layout.witness("absent/witness.c").is_none());
    assert!(
        layout.witness("cpu").is_none(),
        "缺 `<vendor>/` 前缀时没有路径"
    );
}

/// 每段片段都声明自己的签名：一个原语可以有多条签名，文本必须说清写的是哪一条。
#[test]
fn builtin_signs_are_declared() {
    for entry in BUILTIN {
        let set = Layout::builtin()
            .fragments(entry.vendor, entry.ext)
            .expect("内置布局可装载");
        for (prim, _) in entry.fragments {
            assert!(
                set.implementation(*prim).is_some_and(|imp| !imp.is_empty()),
                "`{}` 的 `{prim}` 没有声明签名",
                entry.vendor
            );
        }
    }
}

#[test]
fn a_file_name_must_be_in_the_primitive_closed_set() {
    let vendor_dir = temp("names").join("fake");
    std::fs::create_dir_all(&vendor_dir).expect("建临时目录");
    std::fs::write(
        vendor_dir.join("load.c"),
        "/* cutile-sign: load/memcpy */\n",
    )
    .expect("写片段");
    std::fs::write(vendor_dir.join("not_a_prim.c"), "").expect("写非法片段");
    let error = load(&vendor_dir, "fake", "c").expect_err("闭集外的文件名必须报错");
    assert_eq!(
        error.code(),
        Code::Invalid,
        "文件名是声明，不是 io：{error}"
    );
    assert!(error.to_string().contains("not_a_prim"), "{error}");
    std::fs::remove_dir_all(vendor_dir.parent().expect("有父目录")).ok();
}

/// 片段第一行必须是签名声明：没有它的片段无法与档案交叉校验，所以装载期就拒绝。
#[test]
fn a_fragment_must_declare_its_signature() {
    let vendor_dir = temp("sign").join("fake");
    std::fs::create_dir_all(&vendor_dir).expect("建临时目录");
    std::fs::write(vendor_dir.join("load.c"), "{ $0 = $1; }\n").expect("写片段");
    let error = load(&vendor_dir, "fake", "c").expect_err("没有签名声明的片段必须报错");
    assert!(error.to_string().contains(SIGN_MARKER), "{error}");
    std::fs::remove_dir_all(vendor_dir.parent().expect("有父目录")).ok();
}

/// 声明里的原语名与文件名不一致：多半是复制粘贴，装载期就要报。
#[test]
fn a_signature_must_name_the_file_primitive() {
    let mut set = FragmentSet::new("fake", "c");
    let error = set
        .insert(Prim::Load, "/* cutile-sign: dot/fma */\n{ $0 += $1; }\n")
        .expect_err("原语名不一致必须报错");
    assert!(error.to_string().contains("dot"), "{error}");
    assert!(error.to_string().contains("load"), "{error}");
}

/// 别名不能空着：`impl` 就是档案里那条 `provides.impl`，空名无法对齐。
#[test]
fn a_signature_needs_an_implementation_name() {
    let mut set = FragmentSet::new("fake", "c");
    let error = set
        .insert(Prim::Load, "/* cutile-sign: load/ */\n{ $0 = $1; }\n")
        .expect_err("缺 impl 必须报错");
    assert!(error.to_string().contains("impl"), "{error}");
}

/// 同一原语可以登记多条实现：按实现名取对应片段，首份留给未筛选的兼容入口。
#[test]
fn same_primitive_multiple_implementations_are_selectable() {
    let mut set = FragmentSet::new("fake", "c");
    set.insert(Prim::Load, "/* cutile-sign: load/memcpy */\n{ $0 = $1; }\n")
        .expect("首份可登记");
    set.insert(
        Prim::Load,
        "/* cutile-sign: load/vector */\n{ $0 += $1; }\n",
    )
    .expect("第二份可登记");
    assert_eq!(set.len(), 2);
    assert_eq!(set.implementation(Prim::Load), Some("memcpy"));
    let memcpy = set.get_impl(Prim::Load, "memcpy").expect("memcpy 片段在");
    let vector = set.get_impl(Prim::Load, "vector").expect("vector 片段在");
    assert!(memcpy.contains("$0 = $1"), "{memcpy}");
    assert!(vector.contains("$0 += $1"), "{vector}");
    assert!(set.get_impl(Prim::Load, "absent").is_none());
}

/// 同一原语同名实现不能放两份：否则协商无法对齐到唯一片段。
#[test]
fn duplicate_implementation_is_rejected() {
    let mut set = FragmentSet::new("fake", "c");
    set.insert(Prim::Load, "/* cutile-sign: load/memcpy */\n{ $0 = $1; }\n")
        .expect("首份可登记");
    let error = set
        .insert(Prim::Load, "/* cutile-sign: load/memcpy */\n{ $0 = $1; }\n")
        .expect_err("同名实现重复必须报错");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("memcpy"), "{error}");
}

/// 去掉注释并压缩空白：跨设备只比对可执行语句，注释语言与设备说明不参与同构判定。
fn normalize_body(body: &str) -> String {
    let chars: Vec<char> = body.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        match (chars[i], chars.get(i + 1).copied()) {
            ('/', Some('*')) => {
                i += 2;
                while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                    i += 1;
                }
                i = (i + 2).min(chars.len());
            }
            ('/', Some('/')) => {
                i += 2;
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            (c, _) => {
                out.push(c);
                i += 1;
            }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 同一 `(原语, 实现)` 在多台设备上的片段必须同构：改点积/解包口径要同时改到，不能只改一台。
///
/// 比对按 `impl` 分组（ADR 0027 §4）：标量循环、NEON/DP4A/WMMA 等设备特化指令是不同实现，
/// 彼此不强制同构；只有声明了同一条签名的片段才要求正文同源。
#[test]
fn same_primitive_bodies_are_isomorphic_across_vendors() {
    use std::collections::BTreeMap;

    let vendors: Vec<(&str, &str)> = devices()
        .iter()
        .map(|device| (device.vendor, device.ext))
        .collect();
    let mut groups: BTreeMap<(Prim, String), Vec<String>> = BTreeMap::new();
    for prim in Prim::ALL {
        for (vendor, ext) in &vendors {
            let Some(set) = builtin(vendor, ext) else {
                continue;
            };
            let Some(body) = set.get(*prim) else {
                continue;
            };
            let Some(implementation) = set.implementation(*prim) else {
                continue;
            };
            groups
                .entry((*prim, implementation.to_string()))
                .or_default()
                .push(normalize_body(body));
        }
    }
    for ((prim, implementation), bodies) in &groups {
        if bodies.len() > 1 {
            let first = &bodies[0];
            assert!(
                bodies.iter().all(|body| body == first),
                "`{prim}/{implementation}` 跨设备片段正文不一致：同实现口径必须同源"
            );
        }
    }
}

/// 可执行的提供必须有正文：`L2` 及以上的 `provides` 会被协商选中并真的 emit，没有片段就等于
/// 声明了一条本仓库并不存在的指令。`L0`/`L1` 是能力目录（"未验证视为不具备"），允许只有见证、
/// 没有正文；本用例把这条界限钉住，防止空口补一条 `grade = "L2"`。
#[test]
fn every_verified_provide_has_a_fragment_body() {
    let mut matched = 0;
    for profile in prof::prof::builtin_profiles() {
        let Some(device) = devices()
            .iter()
            .find(|device| device.vendor == profile.meta.vendor)
        else {
            continue;
        };
        matched += 1;
        let Some(set) = builtin(device.vendor, device.ext) else {
            continue;
        };
        for provide in profile
            .provides
            .iter()
            .filter(|provide| provide.grade.is_verified())
        {
            assert_eq!(
                set.implementation(provide.prim),
                Some(provide.impl_name.as_str()),
                "`{}` 档案把 `{}/{}` 记为已验证，却没有这条正文",
                profile.meta.vendor,
                provide.prim,
                provide.impl_name
            );
        }
    }
    assert_eq!(matched, devices().len(), "档案与设备应一一对应");
}

/// 归约片段必须真的折叠：`($0) = ($1)` 只在"归约长度 = 1"时成立，对 `softmax` / `reduce_*` 的
/// 多元素归约会静默返回首元素。它与实现名无关（`scalar_loop` / `shfl.bfly` / `sub_group_reduce`
/// 都是同一条语义），所以单列一条用例——按 `impl` 分组的同构判定看不到这种漂移。
#[test]
fn a_reduction_fragment_must_fold_through_func() {
    for device in devices() {
        let Some(set) = builtin(device.vendor, device.ext) else {
            continue;
        };
        let Some(body) = set.get(Prim::Reduce) else {
            continue;
        };
        let normalized = normalize_body(body);
        assert!(
            normalized.contains("$func("),
            "{} 的 reduce 片段没有走 `$func`：{normalized}",
            device.vendor
        );
    }
}

/// `device-image` 开在本 crate 上必须真的转发给设备 crate：漏转发时设备 crate 自己编了镜像，
/// 而 `prim` 合成出来的表还是纯源码——两边说的不是同一件事。开启后 `metal` 有配方，所以它不再
/// 停在 [`Status::Source`]（编成 [`Status::Compiled`] 还是 [`Status::Skipped`] 取决于本机工具链）。
#[cfg(feature = "device-image")]
#[test]
fn the_device_image_feature_reaches_the_composed_table() {
    let metal = device("metal").expect("内置 metal");
    assert_ne!(
        metal.status,
        Status::Source,
        "`device-image` 没有转发到设备 crate"
    );
}

/// 关着 `device-image` 时每台设备都只有源码：镜像缺席是"这一步没走"，不是配置错误。
#[cfg(not(feature = "device-image"))]
#[test]
fn without_the_feature_every_device_is_source_only() {
    for device in devices() {
        assert_eq!(
            device.status,
            Status::Source,
            "`{}` 不该有镜像",
            device.vendor
        );
        assert!(
            !device.has_image(),
            "`{}` 没有镜像就不该说它有",
            device.vendor
        );
    }
}

/// 六种写回形式在每台设备上都切得出来：装配按语法顺序各取一段，切错或切空要到装配期
/// 才会暴露，所以在这里就按形式逐个取。
#[test]
fn every_device_splits_the_six_epilogue_sections() {
    for device in devices() {
        for form in prof::atom::Epilogue::ALL.iter().copied() {
            let text = device
                .epilogue_of(form)
                .unwrap_or_else(|| panic!("`{}` 没有 `{form}` 段", device.vendor));
            assert!(
                !text.trim().is_empty(),
                "`{}` 的 `{form}` 段是空的",
                device.vendor
            );
            assert!(
                !text.contains("/* epi: "),
                "`{}` 的 `{form}` 段跨进了下一段",
                device.vendor
            );
        }
    }
}

/// 六段互不重叠、合起来覆盖正文：形式标记是唯一的切分点，少一段就是装配缺一步。
#[test]
fn the_epilogue_sections_are_disjoint_and_cover_the_body() {
    for device in devices() {
        let markers = prof::atom::Epilogue::ALL
            .iter()
            .map(|form| prof::device::epi_marker(*form))
            .collect::<Vec<_>>();
        let mut rest = device.epilogue;
        for marker in markers {
            let (_, tail) = rest
                .split_once(marker.as_str())
                .unwrap_or_else(|| panic!("`{}` 缺标记 `{marker}`", device.vendor));
            rest = tail;
        }
        assert!(
            !rest.contains("/* epi: "),
            "`{}` 有多余的写回段",
            device.vendor
        );
    }
}

/// 标记缺席时返回 `None` 而不是切出整段：构建期已经保证六段齐全，运行期不该靠猜。
#[test]
fn a_missing_epilogue_marker_yields_none() {
    let mut bare = *devices().first().expect("至少一台设备");
    bare.epilogue = "/* epi: bias */\n$0[$1] = $2;\n";
    assert!(bare.epilogue_of(prof::atom::Epilogue::Bias).is_some());
    assert!(bare.epilogue_of(prof::atom::Epilogue::Scale).is_none());
}

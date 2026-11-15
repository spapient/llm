use super::*;

/// 内嵌表与磁盘一致：生成器漏扫一个目录，这里就会失败。
#[test]
fn builtin_matches_disk() {
    for entry in BUILTIN {
        let disk = load(&dir(), entry.vendor, entry.ext).expect("磁盘片段可装载");
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
    for (vendor, ext, source) in WITNESS {
        let path = dir().join(vendor).join(format!("witness.{ext}"));
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
    let set = load(&dir(), "cpu", "c").expect("witness.c 必须被跳过");
    assert_eq!(set.len(), cpu.fragments.len());
    assert!(set.get(Prim::Load).is_some());
}

/// 每段片段都声明自己的签名：一个原语可以有多条签名，文本必须说清写的是哪一条。
#[test]
fn builtin_signs_are_declared() {
    for entry in BUILTIN {
        let set = builtin(entry.vendor, entry.ext).expect("内嵌片段可装载");
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
    let root = std::env::temp_dir().join(format!("prim-test-{}", std::process::id()));
    let vendor_dir = root.join("fake");
    std::fs::create_dir_all(&vendor_dir).expect("建临时目录");
    std::fs::write(
        vendor_dir.join("load.c"),
        "/* cutile-sign: load/memcpy */\n",
    )
    .expect("写片段");
    std::fs::write(vendor_dir.join("not_a_prim.c"), "").expect("写非法片段");
    let error = load(&root, "fake", "c").expect_err("闭集外的文件名必须报错");
    assert_eq!(
        error.code(),
        Code::Invalid,
        "文件名是声明，不是 io：{error}"
    );
    assert!(error.to_string().contains("not_a_prim"), "{error}");
    std::fs::remove_dir_all(&root).ok();
}

/// 片段第一行必须是签名声明：没有它的片段无法与档案交叉校验，所以装载期就拒绝。
#[test]
fn a_fragment_must_declare_its_signature() {
    let root = std::env::temp_dir().join(format!("prim-sign-test-{}", std::process::id()));
    let vendor_dir = root.join("fake");
    std::fs::create_dir_all(&vendor_dir).expect("建临时目录");
    std::fs::write(vendor_dir.join("load.c"), "{ $0 = $1; }\n").expect("写片段");
    let error = load(&root, "fake", "c").expect_err("没有签名声明的片段必须报错");
    assert!(error.to_string().contains(SIGN_MARKER), "{error}");
    std::fs::remove_dir_all(&root).ok();
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

/// 同一原语在多台设备上的标量片段必须同构：改点积/解包口径要同时改到，不能只改一台。
///
/// 这只约束当前的标量实现；P3 出现设备特化指令（dp4a / mmad / wmma）后，按 `impl`
/// 分组比对，不同实现之间不再强制同构（ADR 0027 §4）。
#[test]
fn same_primitive_bodies_are_isomorphic_across_vendors() {
    let vendors: Vec<(&str, &str)> = devices()
        .iter()
        .map(|device| (device.vendor, device.ext))
        .collect();
    for prim in Prim::ALL {
        let mut bodies = Vec::new();
        for (vendor, ext) in &vendors {
            let Some(set) = builtin(vendor, ext) else {
                continue;
            };
            let Some(body) = set.get(*prim) else {
                continue;
            };
            bodies.push(normalize_body(body));
        }
        if bodies.len() > 1 {
            let first = &bodies[0];
            assert!(
                bodies.iter().all(|body| body == first),
                "`{prim}` 跨设备片段正文不一致：标量口径必须同源"
            );
        }
    }
}

use super::*;
use std::collections::BTreeSet;

/// 磁盘上的设备目录：有 `witness.<ext>` 的目录才算。
fn on_disk() -> BTreeSet<String> {
    let root = crate::dir();
    std::fs::read_dir(&root)
        .expect("片段层可读")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_dir()
                && std::fs::read_dir(path)
                    .map(|files| {
                        files.flatten().any(|file| {
                            file.path().file_stem().and_then(|v| v.to_str()) == Some("witness")
                        })
                    })
                    .unwrap_or(false)
        })
        .map(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .expect("目录名")
                .to_string()
        })
        .collect()
}

#[test]
fn every_device_directory_has_a_row() {
    let disk = on_disk();
    let generated: BTreeSet<String> = devices()
        .iter()
        .map(|device| device.vendor.to_string())
        .collect();
    assert_eq!(
        disk, generated,
        "`<vendor>/` 目录与生成的 `DEVICES` 不一致：加目录不会漏，漏的是构建脚本没重跑"
    );
}

#[test]
fn every_device_names_its_own_archive() {
    for device in devices() {
        let table =
            prof::embedded(device.vendor).unwrap_or_else(|| panic!("`{}` 没有档案", device.vendor));
        let profile = prof::prof::Profile::parse(table).expect("档案可解析");
        assert_eq!(profile.meta.vendor, device.vendor);
        assert_eq!(profile.meta.emit.as_str(), device.emit);
        assert_eq!(profile.meta.emit.ext(), device.ext);
        assert_eq!(profile.meta.tool, device.tool);
        assert!(
            !device.source.is_empty(),
            "`{}` 没有见证源码",
            device.vendor
        );
    }
}

/// 映射表指向的入口必须真的在那份见证源码里：否则镜像里查不到这个名字。
#[test]
fn every_entry_is_a_function_in_the_witness_source() {
    for device in devices() {
        assert!(!device.prims.is_empty(), "`{}` 没有原语", device.vendor);
        for (prim, entry) in device.prims {
            assert!(
                device.source.contains(&format!("int {entry}(")),
                "`{}` 的 `{prim}` 指向 `{entry}`，但见证源码里没有 `int {entry}(`",
                device.vendor
            );
        }
    }
}

#[test]
fn the_primitive_list_is_ordered_and_unique() {
    for device in devices() {
        let prims: Vec<Prim> = device.prims().collect();
        let mut sorted = prims.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            prims, sorted,
            "`{}` 的原语枚举应按闭集顺序且不重复",
            device.vendor
        );
    }
}

/// 原语枚举与片段表同源：同一台设备的两个视图不能各自列出不同的原语。
#[test]
fn the_primitive_list_matches_the_fragment_table() {
    for device in devices() {
        let set = crate::builtin(device.vendor, device.ext).expect("内嵌片段可装载");
        let from_fragments: Vec<Prim> = Prim::ALL
            .iter()
            .copied()
            .filter(|prim| set.get(*prim).is_some())
            .collect();
        assert_eq!(
            device.prims().collect::<Vec<_>>(),
            from_fragments,
            "`{}` 的原语枚举与片段表不一致",
            device.vendor
        );
    }
}

#[test]
fn the_status_explains_the_image() {
    for device in devices() {
        assert_eq!(
            device.image.is_some(),
            device.status == Status::Compiled,
            "`{}` 的 `status` 与 `image` 必须一致",
            device.vendor
        );
        if let Some(image) = &device.image {
            assert!(!image.is_empty(), "`{}` 的镜像是空的", device.vendor);
            assert!(
                !image.format().is_empty(),
                "`{}` 的镜像没有格式名",
                device.vendor
            );
        }
        if let Status::Skipped { reason } = device.status {
            assert!(!reason.is_empty(), "`{}` 跳过了却没有原因", device.vendor);
            assert!(!reason.contains('\n'), "原因只能是一行：{reason}");
        }
    }
}

/// 没有开启 `device-image` 时一个镜像都不该有：两种模式不能悄悄混在一起。
#[cfg(not(feature = "device-image"))]
#[test]
fn the_debug_mode_carries_no_image() {
    for device in devices() {
        assert!(device.image.is_none(), "`{}` 不该有镜像", device.vendor);
        assert_eq!(
            device.status,
            Status::Source,
            "`{}` 应是调试模式",
            device.vendor
        );
    }
}

/// 打开了 `device-image`：编出来的必须是真产物，不是占位。
#[cfg(feature = "device-image")]
#[test]
fn a_compiled_image_is_a_real_device_binary() {
    for device in devices() {
        let Some(image) = &device.image else {
            continue;
        };
        assert!(
            image.len() > 16,
            "`{}` 的镜像只有 {} 字节",
            device.vendor,
            image.len()
        );
        assert!(
            image.symbols() > 0,
            "`{}` 的镜像里一个符号都没数到",
            device.vendor
        );
        assert!(
            image.symbols() >= device.prims.len(),
            "`{}` 的镜像里符号数少于原语数",
            device.vendor
        );
        // `metallib` 的魔数：文件头就是格式，不需要外部工具才能验。
        if image.format() == "metallib" {
            assert_eq!(
                &image.bytes()[..4],
                b"MTLB",
                "`{}` 不是 metallib",
                device.vendor
            );
        }
    }
}

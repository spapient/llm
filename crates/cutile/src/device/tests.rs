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

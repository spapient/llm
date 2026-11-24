use super::*;
use crate::toolchain;
use cutile::builtin::Catalogue;
use cutile::{decl, emit, substitute};

/// 一台设备全部算子 × 分块组合的最终源文件：`{算子}_{序号}` → 文本。
///
/// 走的是执行层同一条路（`substitute::render` 拼片段层 → `emit::assemble` 补 preamble），
/// 所以这里解析的文本就是最终落盘的那份。协商判据也与生产链路同源：设备接不住的档
/// （如 CPU 前端不具备的 f16/bf16）不发射到它的语法前端，只对协商可行的算子做收下校验。
fn units_of(
    catalogue: &Catalogue,
    profile: &Profile,
    fragments: &prof::FragmentSet,
) -> Vec<(String, String)> {
    let mut units = Vec::new();
    for op in catalogue.ops() {
        if cutile::cap::negotiate(op, profile).is_err() {
            continue;
        }
        let combos = decl::expand(op).expect("tile 候选");
        let variant = op.variants.first().expect("降级链至少一档");
        for (idx, combo) in combos.iter().enumerate() {
            let body = substitute::render(op, combo, fragments).expect("片段拼得进去");
            let source = emit::assemble(profile, op, variant, combo, &body);
            units.push((format!("{}_{idx}", op.name), source.text));
        }
    }
    units
}

/// 片段层只有 `witness.*` 过门，正文从来没被任何编译器看过——声明里写 `barrier`、正文是一句
/// “留给 P3”，门一样绿。这条用例把判据钉在正文上：片段拼进翻译单元之后，这台设备**自己声明的
/// 那门语言**必须收下它，否则片段就是占位。
#[test]
fn every_fragment_composes_into_a_translation_unit_its_language_accepts() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let dir = std::env::temp_dir().join(format!("cutile-fragments-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("临时目录");
    let mut languages = 0;
    let mut units = 0;
    for profile in catalogue.profiles() {
        let Some(spec) = config()
            .gates
            .iter()
            .find(|entry| entry.emit == profile.meta.emit)
        else {
            eprintln!(
                "skip: `{}` 的 `{}` 没有登记语法前端，它的片段未被解析",
                profile.meta.id, profile.meta.emit
            );
            continue;
        };
        if front(spec).is_none() {
            eprintln!(
                "skip: 本机没有 `{}`，`{}` 的片段未被解析",
                spec.tool.join(" "),
                profile.meta.id
            );
            continue;
        }
        let arguments = args_for(spec, profile);
        let fragments =
            prof::builtin(profile.meta.vendor.as_str(), profile.meta.emit.ext()).expect("片段层");
        for (name, text) in units_of(&catalogue, profile, &fragments) {
            let path = dir.join(format!(
                "{}-{name}.{}",
                profile.meta.id,
                profile.meta.emit.ext()
            ));
            std::fs::write(&path, &text).expect("写翻译单元");
            if let Err(error) = parse_one(spec, &arguments, &path) {
                let _ = std::fs::remove_dir_all(&dir);
                panic!(
                    "`{}` 的 `{}` 收不下 `{name}`：{error}",
                    profile.meta.id, spec.emit
                );
            }
            units += 1;
        }
        languages += 1;
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(languages > 0, "没有任何设备走到解析：这条门等于没开");
    eprintln!("{languages} 门语言 / {units} 份翻译单元");
}

/// 反向验证：正文改坏之后判据必须变红，否则上一条的 green 不说明门在看正文。
#[test]
fn a_fragment_body_the_language_rejects_is_a_defect() {
    if toolchain::skip_without_host_compiler() {
        return;
    }
    let catalogue = Catalogue::try_builtin().expect("内置数据");
    let profile = catalogue.profile("intel").expect("intel 档案");
    let spec = config()
        .gates
        .iter()
        .find(|entry| entry.emit == profile.meta.emit)
        .expect("OpenCL 语法前端已登记");
    if front(spec).is_none() {
        eprintln!("skip: 本机没有 `clang`");
        return;
    }
    let arguments = args_for(spec, profile);
    let honest =
        prof::builtin(profile.meta.vendor.as_str(), profile.meta.emit.ext()).expect("片段层");
    // `gemv_f32` 用到 `bar`：改坏它，拼进去的翻译单元就该被拒。
    let op = catalogue
        .ops()
        .iter()
        .find(|op| op.name == "gemv_f32")
        .expect("gemv_f32 已声明");
    let combo = decl::expand(op)
        .expect("tile 候选")
        .into_iter()
        .next()
        .expect("至少一组分块");
    let variant = op.variants.first().expect("降级链至少一档");
    let dir = std::env::temp_dir().join(format!("cutile-fragments-broken-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("临时目录");
    let parse = |fragments: &prof::FragmentSet, tag: &str| {
        let body = substitute::render(op, &combo, fragments).expect("片段拼得进去");
        let source = emit::assemble(profile, op, variant, &combo, &body);
        let path = dir.join(format!("{tag}.cl"));
        std::fs::write(&path, &source.text).expect("写翻译单元");
        parse_one(spec, &arguments, &path)
    };
    parse(&honest, "honest").expect("原样的片段必须过门");
    let broken = corrupted(&honest, prof::Prim::Dot);
    let rejected = parse(&broken, "broken").expect_err("改坏的正文必须被语言拒绝");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(rejected.to_string().contains("broken.cl"), "{rejected}");
}

/// 复制一份片段层，把 `target` 的正文换成语言收不下的文本（声明行不动，拼进去的仍是一段片段）。
fn corrupted(base: &prof::FragmentSet, target: prof::Prim) -> prof::FragmentSet {
    let mut set = prof::FragmentSet::new(base.vendor(), base.ext());
    for prim in prof::Prim::ALL {
        for fragment in base.fragments_of(*prim) {
            let source = fragment.source();
            let text = if *prim == target {
                format!("{source}\nint cutile_broken_body = ;\n")
            } else {
                source.to_string()
            };
            set.insert(*prim, text).expect("签名声明没动，登记照旧");
        }
    }
    set
}

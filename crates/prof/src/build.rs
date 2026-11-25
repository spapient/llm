//! 构建期一次生成两张静态表：强类型档案视图（`$OUT_DIR/embedded.rs`）与片段表
//! （`$OUT_DIR/tables.rs`）；算子声明由 `prof_ops!()` 展开成强类型 `OpView`。
//!
//! 语料都在构建期解析一次（ADR 0022）：`<vendor>.toml` 回答设备档案，`ops/*.{toml,k}`
//! 回答共享算子契约与参考正文，`kernels.toml` 回答模板级调优；运行期只遍历生成后的强类型表。
//!
//! 生成器必须在这里，因为 cargo 不允许一个包当自己的构建依赖：它曾经在独立 crate `frag` 里
//! （`use prof::…`），代价是一个 crate、一个 feature 与一张手写的六设备清单。搬进来之后
//! "哪台设备"由 `src/<vendor>/<vendor>.toml` 这个文件名回答——目录里有几份档案就有几台设备。
//!
//! 构建期校验替代运行期断言：签名声明对不上文件名、`(prim, impl)` 在档案里没有供给、见证入口
//! 缺失、尾块段不齐，全部在这里失败。运行期只遍历生成后的强类型静态视图。
//!
//! 两处闭集不在这里核，因为构建脚本拿不到运行期类型：尾块六种形式的名字由
//! `prim/tests/fragments.rs` 对着 `Epilogue::ALL` 核（这里只核结构：恰六段、互不相同、六台
//! 设备一致）；原语名由生成的 `Prim::<变体>` 交给编译器核，写错的那一行在生成代码里编不过。

#[path = "sign.rs"]
mod sign;

#[path = "buildops.rs"]
mod buildops;

#[path = "nativebuild.rs"]
mod nativebuild;

#[path = "buildtables.rs"]
mod buildtables;

#[path = "buildimage.rs"]
mod buildimage;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use diagnostic::buildgen;
use sign::{EPILOGUE_STEM, PREAMBLE_STEM, WITNESS_STEM};

fn main() {
    let manifest = variable("CARGO_MANIFEST_DIR");
    let src = PathBuf::from(&manifest).join("src");
    let out = PathBuf::from(variable("OUT_DIR"));

    // 整棵树都算输入：新增一段片段、换一份见证都必须让构建脚本重跑。
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DEVICE_IMAGE");
    println!("cargo:rerun-if-env-changed=CUTILE_REQUIRE_IMAGE");

    let devices = devices_of(&src);
    let images = env::var_os("CARGO_FEATURE_DEVICE_IMAGE").is_some();
    let tables = buildtables::tables_of(&devices, &out, images);
    let (template_list, templates_table) =
        buildops::template_tables_of(Path::new(&manifest), &devices);
    nativebuild::build_native(Path::new(&manifest), &out);

    for (name, text) in [
        ("embedded.rs", buildtables::archives_of(&devices)),
        ("tables.rs", tables),
        ("template_list.rs", template_list),
        ("templates_table.rs", templates_table),
    ] {
        let path = out.join(name);
        fs::write(&path, text)
            .unwrap_or_else(|error| fail(format!("写 {} 失败：{error}", path.display())));
    }
}

// ── 设备清单 ──────────────────────────────────────────────────────────

/// 一台设备：档案元数据、片段与路径（正文由生成的 `include_str!` 读，脚本里不再存一份）。
struct Device {
    /// 唯一档案键：基档是 `nvidia`，变体是 `nvidia_a3000`。
    name: String,
    /// 共享片段、见证与后端内核的目录名；变体不复制设备族语料。
    family: String,
    /// 片段与见证的扩展名，由目录里的见证文件决定。
    ext: String,
    /// 档案 `[meta]` 里要写进设备行的那三项。
    emit: String,
    tool: String,
    tool_ver: String,
    /// 原语名 → 片段路径，按原语名升序。
    fragments: Vec<(String, PathBuf)>,
    /// 原语名 → 见证入口，顺序同 [`Device::fragments`]。
    entries: Vec<(String, String)>,
    /// 写回尾块片段：一份文本，六种形式按 `/* epi: <form> */` 分段。
    epilogue: PathBuf,
    /// 源文件前置声明（`#include` 与软件数学函数）：`preamble.<ext>`，缺它设备仍可用。
    preamble: Option<PathBuf>,
    /// 见证源码与它的档案：生成的代码按路径读，构建期再按路径校。
    witness: PathBuf,
    /// 已合并基档与变体的完整档案；校验和内嵌必须共用这一份。
    archive: toml::Table,
    /// 尾块六段的名字，按出现顺序。
    forms: Vec<String>,
}

/// 设备目录：`<family>.toml` 是基档，其余 `<variant>.toml` 是同族硬件变体。
/// 变体生成独立档案键 `<family>_<variant>`，但共享目录里的片段、见证与后端特化。
fn devices_of(src: &Path) -> Vec<Device> {
    let mut dirs: Vec<(PathBuf, Vec<PathBuf>)> = dirs_in(src)
        .into_iter()
        .filter(|(_, files)| has_witness(files))
        .collect();
    dirs.sort_by(|left, right| left.0.cmp(&right.0));
    if dirs.is_empty() {
        fail(format!("`{}` 里一台设备都没有", src.display()));
    }

    let mut out: Vec<Device> = Vec::new();
    for (dir, files) in dirs {
        let family = family_of(&dir);
        if !identifier(family) {
            fail(format!("设备目录名 `{family}` 不是可移植的 Rust 标识符"));
        }
        let variants = variants_of(&files, family);
        collect_devices(&dir, family, &variants, &files, &mut out);
    }
    out.sort_by(|left, right| left.name.cmp(&right.name));
    out
}

/// 设备目录名：目录名即设备族名，必须可作 UTF-8 标识符。
fn family_of(dir: &Path) -> &str {
    dir.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_else(|| fail(format!("`{}` 的目录名不是合法 UTF-8", dir.display())))
}

/// 同族变体清单：除基档外的其余 `.toml` 档案名。
fn variants_of(files: &[PathBuf], family: &str) -> Vec<String> {
    let mut variants: Vec<String> = files
        .iter()
        .filter(|path| {
            buildgen::extension(path).as_deref() == Some("toml")
                && buildgen::file_stem(path) != Some(family)
        })
        .map(|path| {
            buildgen::file_stem(path)
                .unwrap_or_else(|| fail(format!("`{}` 没有档案名", path.display())))
                .to_string()
        })
        .collect();
    variants.sort();
    variants
}

/// 基档名为设备族，变体名为 `<family>_<variant>`；所有入口共用同一校验。
fn device_name(family: &str, variant: Option<&str>) -> String {
    let name = variant.map_or_else(|| family.to_string(), |v| format!("{family}_{v}"));
    if !identifier(&name) {
        fail(format!("设备档案键 `{name}` 不是可移植的 Rust 标识符"));
    }
    name
}

/// 为一个设备族收集基档与全部变体设备行，并校验同族尾块分段一致。
fn collect_devices(
    dir: &Path,
    family: &str,
    variants: &[String],
    files: &[PathBuf],
    out: &mut Vec<Device>,
) {
    for variant in std::iter::once(None).chain(variants.iter().map(|name| Some(name.as_str()))) {
        let name = device_name(family, variant);
        let device = scan(dir, family, variant, files.to_vec());
        if let Some(first) = out.first().filter(|first| first.forms != device.forms) {
            fail(format!(
                "`{name}` 的尾块分段是 [{}]，与 `{}` 的 [{}] 不一致",
                device.forms.join(", "),
                first.name,
                first.forms.join(", ")
            ));
        }
        out.push(device);
    }
}

/// 扫描设备族目录，并为基档或一个变体生成独立设备行。
/// 所有变体共用同一份片段清单；档案正文在构建期合并，运行期没有继承分支。
fn scan(dir: &Path, family: &str, variant: Option<&str>, files: Vec<PathBuf>) -> Device {
    let name = device_name(family, variant);
    let witness = expect(
        files
            .iter()
            .find(|path| buildgen::file_stem(path) == Some(WITNESS_STEM))
            .cloned(),
        format!(
            "`{}` 里没有 `{WITNESS_STEM}.<ext>`：设备目录必须带上见证，否则映射表没有入口可指",
            dir.display()
        ),
    );
    let ext = expect(
        buildgen::extension(&witness),
        format!("`{}` 没有扩展名", witness.display()),
    );

    let doc = archive_of(dir, family, variant);
    let meta = expect(
        doc.get("meta").and_then(|value| value.as_table()),
        format!("`{name}` 的档案没有 `[meta]` 表"),
    );
    let provides = provides_of(&doc, family, &ext);

    let (fragments, entries, epilogue) = fragment_scan(&files, &witness, &ext, &provides);
    if fragments.is_empty() {
        fail(format!("`{}` 里一段片段都没有", dir.display()));
    }
    let epilogue = expect(
        epilogue,
        format!(
            "`{}` 里没有 `{EPILOGUE_STEM}.{ext}`：六种写回形式没有片段，装配无从下手",
            dir.display()
        ),
    );
    // 前置声明可有可无：只有写了 `preamble.<ext>` 的设备才内嵌它，其余退回档案内联文本。
    let preamble = files
        .iter()
        .find(|path| buildgen::file_stem(path) == Some(PREAMBLE_STEM))
        .cloned();
    let mut fragments = fragments;
    let mut entries = entries;
    fragments.sort();
    entries.sort();

    let forms = forms_of(&read(&epilogue), &epilogue);
    Device {
        name: name.clone(),
        family: family.to_string(),
        ext,
        emit: text(meta, "emit", &name),
        tool: text(meta, "tools", &name),
        tool_ver: meta
            .get("tool_ver")
            .and_then(|value| value.as_str())
            .unwrap_or("-")
            .to_string(),
        fragments,
        entries,
        epilogue,
        preamble,
        witness,
        archive: doc.clone(),
        forms,
    }
}

/// 一趟设备目录扫描的产物：原语片段清单、条目与尾块。
/// 以别名承载，避免返回类型逐字展开成嵌套三元组（`clippy::type_complexity`）。
type FragmentScan = (
    Vec<(String, PathBuf)>,
    Vec<(String, String)>,
    Option<PathBuf>,
);

/// 一趟扫完设备目录：剔除档案/见证/生成代码，按扩展名归入原语片段，并单独记下尾块。
fn fragment_scan(
    files: &[PathBuf],
    witness: &Path,
    ext: &str,
    provides: &[Provide],
) -> FragmentScan {
    let mut fragments = Vec::new();
    let mut entries = Vec::new();
    let mut epilogue = None;
    for path in files {
        if is_non_fragment(path, witness) {
            continue;
        }
        let prim = expect(
            buildgen::file_stem(path),
            format!("`{}` 的文件名不是合法 UTF-8", path.display()),
        )
        .to_string();
        if buildgen::extension(path).as_deref() != Some(ext) {
            fail(format!(
                "`{}` 的扩展名与同目录的 `{WITNESS_STEM}.{ext}` 不一致：一台设备只有一种输出语言",
                path.display()
            ));
        }
        if prim == EPILOGUE_STEM {
            epilogue = Some(path.clone());
            continue;
        }
        let implementation = signature(&prim, &read(path), path);
        entries.push((
            prim.clone(),
            entry_of(provides, &prim, &implementation, path),
        ));
        fragments.push((prim, path.clone()));
    }
    (fragments, entries, epilogue)
}

/// 见证、档案、生成代码与算子特化正文都不进原语片段。
fn is_non_fragment(path: &Path, witness: &Path) -> bool {
    path == witness
        || buildgen::file_stem(path) == Some(PREAMBLE_STEM)
        || matches!(
            buildgen::extension(path).as_deref(),
            Some("toml" | "rs" | "k")
        )
        || is_operator_body(path, witness)
}

/// 共享参考同名或显式标记的设备正文是算子特化，不是原语片段。
fn is_operator_body(path: &Path, witness: &Path) -> bool {
    let Some(stem) = buildgen::file_stem(path) else {
        return false;
    };
    witness
        .parent()
        .and_then(Path::parent)
        .is_some_and(|src| src.join("ops").join(format!("{stem}.k")).is_file())
        || sign::is_native_template(&read(path), stem)
}

/// 读取基档与可选变体覆盖表。表递归合并、数组整体替换，编译 flags 不会拼接漂移。
fn archive_of(dir: &Path, family: &str, variant: Option<&str>) -> toml::Table {
    let base = dir.join(format!("{family}.toml"));
    if !base.is_file() {
        fail(format!("`{}` 缺设备族基档 `{family}.toml`", dir.display()));
    }
    let mut doc = buildgen::read(&base).unwrap_or_else(|error| fail(format!("{error}")));
    let name = device_name(family, variant);
    if let Some(variant) = variant {
        let overlay = dir.join(format!("{variant}.toml"));
        let overlay = buildgen::read(&overlay).unwrap_or_else(|error| fail(format!("{error}")));
        buildgen::merge_tables(&mut doc, overlay);
    }
    let vendor = expect(
        doc.get("meta")
            .and_then(|value| value.as_table())
            .and_then(|meta| meta.get("vendor"))
            .and_then(|value| value.as_str()),
        format!("`{name}` 的档案没有 `meta.vendor`"),
    );
    if vendor != name {
        fail(format!(
            "设备档案 `{name}` 的 `meta.vendor` 是 `{vendor}`：两者必须一致"
        ));
    }
    doc
}

/// 一条 `[[provides]]` 里生成期要核的三件事：签名、见证入口、见证文件。
struct Provide {
    prim: String,
    imp: String,
    func: String,
}

fn provides_of(doc: &toml::Table, name: &str, ext: &str) -> Vec<Provide> {
    let want = format!("{name}/{WITNESS_STEM}.{ext}");
    doc.get("provides")
        .and_then(|value| value.as_array())
        .unwrap_or_else(|| fail(format!("`{name}` 的档案没有 `[[provides]]`")))
        .iter()
        .map(|entry| {
            let table = entry
                .as_table()
                .unwrap_or_else(|| fail(format!("`{name}` 的 `provides` 里有非表条目")));
            let witness = table
                .get("witness")
                .and_then(|value| value.as_table())
                .unwrap_or_else(|| {
                    fail(format!(
                        "`{name}` 的档案有 `provides` 缺 `witness`：写回证据必须可指"
                    ))
                });
            let file = text(witness, "file", name);
            if file != want {
                fail(format!(
                    "`{name}` 的 `witness.file` 是 `{file}`：必须是 `{want}`（档案与目录必须同名）"
                ));
            }
            Provide {
                prim: text(table, "prim", name),
                imp: text(table, "impl", name),
                func: text(witness, "func", name),
            }
        })
        .collect()
}

/// 原语在设备镜像里的入口：档案里 `(prim, impl)` 那条供给的见证函数名。
fn entry_of(provides: &[Provide], prim: &str, implementation: &str, path: &Path) -> String {
    provides
        .iter()
        .find(|provide| provide.prim == prim && provide.imp == implementation)
        .map_or_else(
            || {
                let declared: Vec<&str> = provides
                    .iter()
                    .filter(|provide| provide.prim == prim)
                    .map(|provide| provide.imp.as_str())
                    .collect();
                fail(format!(
                    "`{}` 声明的签名 `{prim}/{implementation}` 在档案里没有对应 `provides`\
                 （档案里 `{prim}` 的实现：{}）",
                    path.display(),
                    if declared.is_empty() {
                        "一条都没有".to_string()
                    } else {
                        declared.join(", ")
                    }
                ))
            },
            |provide| provide.func.clone(),
        )
}

/// 片段第一行的签名声明：`/* cutile-sign: <prim>/<impl> */`。返回 `impl` 名。
fn signature(prim: &str, source: &str, path: &Path) -> String {
    let first = source.lines().next().unwrap_or_default();
    let declared = sign::strip_marker(first).unwrap_or_else(|| {
        fail(format!(
            "`{}` 的第一行必须是 `/* {} {prim}/<impl> */`，实际是 `{first}`",
            path.display(),
            sign::MARKER
        ))
    });
    let (declared_prim, implementation) = sign::split_slash(declared).unwrap_or_else(|| {
        fail(format!(
            "`{}` 的签名声明不是 `<prim>/<impl>`",
            path.display()
        ))
    });
    if declared_prim.trim() != prim {
        fail(format!(
            "`{}` 的签名声明写的是 `{}`：文件名即原语名，两者必须一致",
            path.display(),
            declared_prim.trim()
        ));
    }
    let implementation = implementation.trim();
    if implementation.is_empty() {
        fail(format!("`{}` 的签名声明缺少 `impl` 名", path.display()));
    }
    implementation.to_string()
}

/// 尾块正文里的六段标记：数量与名字在这里核，名字属于闭集由 `prof` 的测试核。
fn forms_of(source: &str, path: &Path) -> Vec<String> {
    let mut forms: Vec<String> = Vec::new();
    for (idx, marker) in source.match_indices("/* epi: ").collect::<Vec<_>>() {
        let _ = marker;
        let name = source[idx + "/* epi: ".len()..]
            .split(" */")
            .next()
            .unwrap_or_default()
            .trim();
        if name.is_empty() {
            fail(format!("`{}` 有空的分段标记", path.display()));
        }
        if forms.iter().any(|have| have == name) {
            fail(format!(
                "`{}` 的尾块分段 `{name}` 出现了两次",
                path.display()
            ));
        }
        forms.push(name.to_string());
    }
    if forms.len() != 6 {
        fail(format!(
            "`{}` 有 {} 段尾块（`/* epi: <form> */`），六种写回形式必须齐全",
            path.display(),
            forms.len()
        ));
    }
    forms
}

// ── 文件与环境变量 ────────────────────────────────────────────────────

/// 列出目录里的文件并排序；单个条目读失败显式失败——静默跳过会让生成的表缺项而构建仍绿。
fn files_in(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)
        .unwrap_or_else(|error| fail(format!("读取 {} 失败：{error}", dir.display())))
    {
        let entry = entry
            .unwrap_or_else(|error| fail(format!("读取 {} 的目录项失败：{error}", dir.display())));
        let path = entry.path();
        if path.is_file() {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// 只有带见证文件的目录才是设备目录：`src/` 里除了六台设备的语料，还有 `decl`/`ops` 这类
/// 非设备子树（声明解析与算子语料），它们不含 `witness.<ext>`，不得被当成设备扫描。
fn has_witness(files: &[PathBuf]) -> bool {
    files
        .iter()
        .any(|path| buildgen::file_stem(path) == Some(WITNESS_STEM))
}

/// `src` 下每个子目录与其文件清单（一次读盘，判据与扫描共用同一份清单）。
fn dirs_in(src: &Path) -> Vec<(PathBuf, Vec<PathBuf>)> {
    let mut out = Vec::new();
    for entry in fs::read_dir(src)
        .unwrap_or_else(|error| fail(format!("读取 {} 失败：{error}", src.display())))
    {
        let entry = entry
            .unwrap_or_else(|error| fail(format!("读取 {} 的目录项失败：{error}", src.display())));
        let path = entry.path();
        if path.is_dir() {
            out.push((path.clone(), files_in(&path)));
        }
    }
    out
}

/// 读取一份文件正文，I/O 错误直接终止构建并带上路径；构建期子 mod 经 `super::` 复用。
fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| fail(format!("读取 {} 失败：{error}", path.display())))
}

fn text(table: &toml::Table, key: &str, name: &str) -> String {
    table
        .get(key)
        .and_then(|value| value.as_str())
        .unwrap_or_else(|| fail(format!("`{name}` 的档案缺少字符串键 `{key}`")))
        .to_string()
}

/// 构建脚本必带的环境变量；缺了说明调用方不是 `build.rs`，直接失败而不是猜一个目录。
fn variable(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| fail(format!("{name} 未设置：只有 build.rs 会读它")))
}

/// 目录名要能直接写进生成的代码：标识符形状，且不是 Rust 关键字。
fn identifier(name: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move",
        "mut", "pub", "ref", "return", "self", "static", "struct", "super", "trait", "true",
        "type", "unsafe", "use", "where", "while",
    ];
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && chars.all(|rest| rest == '_' || rest.is_ascii_alphanumeric())
        && !KEYWORDS.contains(&name)
}

/// 文件名里的原语名 → `Prim` 变体名：`load_async` → `LoadAsync`。
///
/// 名字对不对不靠这里判断：生成的代码写的是 `Prim::<变体>`，写错的原语在这一行编不过。
fn variant(name: &str) -> String {
    if name.eq_ignore_ascii_case("bf16") {
        return "BF16".to_string();
    }
    pascal(&normalize_upper(name))
}

/// 全大写档先归一到小写；已是混合大小写的原样保留。
fn normalize_upper(name: &str) -> String {
    if name.chars().all(|ch| !ch.is_ascii_lowercase()) {
        return name.to_ascii_lowercase();
    }
    name.to_string()
}

/// 蛇形/小写名转 Rust 变体驼峰：下划线后首字母大写。
fn pascal(name: &str) -> String {
    let mut out = String::new();
    let mut upper = true;
    for ch in name.chars() {
        if ch == '_' {
            upper = true;
        } else {
            out.push(if upper { ch.to_ascii_uppercase() } else { ch });
            upper = false;
        }
    }
    out
}

/// 构建期失败：报给 `cargo` 并以非零码结束，`build.rs` 因此失败。
fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("prof::build: {message}");
    std::process::exit(1)
}

/// 拆开 `Option`，缺值直接以 [`fail`] 终止构建；调用点不必再写 `||` 闭包。
fn expect<T>(value: Option<T>, message: impl std::fmt::Display) -> T {
    match value {
        Some(value) => value,
        None => fail(message),
    }
}

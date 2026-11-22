//! 构建期一次生成三棵树：档案表（`$OUT_DIR/embedded.rs`）、片段表（`$OUT_DIR/tables.rs`）
//! 与算子声明表（`$OUT_DIR/declared.rs`）。
//!
//! 三份语料同性质——都是「编译期解析一次、运行期只遍历 `Table`」（ADR 0022）——所以只有一个
//! 构建脚本：`src/<vendor>/<vendor>.toml` 回答"哪台设备"，`ops/*.toml` 回答"哪些算子"。
//!
//! 生成器必须在这里，因为 cargo 不允许一个包当自己的构建依赖：它曾经在独立 crate `frag` 里
//! （`use prof::…`），代价是一个 crate、一个 feature 与一张手写的六设备清单。搬进来之后
//! "哪台设备"由 `src/<vendor>/<vendor>.toml` 这个文件名回答——目录里有几份档案就有几台设备。
//!
//! 构建期校验替代运行期断言：签名声明对不上文件名、`(prim, impl)` 在档案里没有供给、见证入口
//! 缺失、尾块段不齐，全部在这里失败。ADR 0022：语料编译期解析一次，运行期只遍历 [`Table`]。
//!
//! 两处闭集不在这里核，因为构建脚本拿不到运行期类型：尾块六种形式的名字由
//! `prim/tests/fragments.rs` 对着 `Epilogue::ALL` 核（这里只核结构：恰六段、互不相同、六台
//! 设备一致）；原语名由生成的 `Prim::<变体>` 交给编译器核，写错的那一行在生成代码里编不过。

#[path = "src/sign.rs"]
mod sign;

#[path = "src/buildops.rs"]
mod buildops;

#[path = "src/buildtables.rs"]
mod buildtables;

#[path = "src/buildimage.rs"]
mod buildimage;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use diagnostic::buildgen;
use sign::{EPILOGUE_STEM, WITNESS_STEM};

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
    let (declared, template_list, templates_table) =
        buildops::declarations_of(Path::new(&manifest));

    for (name, text) in [
        ("embedded.rs", buildtables::archives_of(&devices)),
        ("tables.rs", tables),
        ("declared.rs", declared),
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
    /// vendor 名，同时是 `src/` 下的目录名与档案文件名。
    name: String,
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
    /// 见证源码与它的档案：生成的代码按路径读，构建期再按路径校。
    witness: PathBuf,
    /// 档案路径：生成的代码按路径读，生成器也拿它当输入登记。
    archive: PathBuf,
    /// 尾块六段的名字，按出现顺序。
    forms: Vec<String>,
}

/// 目录里的设备：一个子目录一台，顺序按目录名升序（产物稳定）。
///
/// 每个目录只列一次：判据是「目录里有没有见证文件」，而判完要读的文件清单正是同一份。
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
        let name = dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_else(|| fail(format!("`{}` 的目录名不是合法 UTF-8", dir.display())));
        if !identifier(name) {
            fail(format!("设备目录名 `{name}` 不是可移植的 Rust 标识符"));
        }
        let device = scan(&dir, name, files);
        // 六台设备的尾块必须是同一组形式：装配器按名字取段，缺一处就只在那一台上炸。
        match out.first() {
            Some(first) if first.forms != device.forms => fail(format!(
                "`{name}` 的尾块分段是 [{}]，与 `{}` 的 [{}] 不一致",
                device.forms.join(", "),
                first.name,
                first.forms.join(", ")
            )),
            _ => out.push(device),
        }
    }
    out
}

/// 扫描一台设备的目录：七段原语片段 + `epilogue.<ext>` + `witness.<ext>` + 同名档案。
///
/// `files` 由调用方一次性列出并传入：判「这个目录算不算设备」与「它有哪些文件」是同一回读盘。
fn scan(dir: &Path, name: &str, files: Vec<PathBuf>) -> Device {
    let witness = files
        .iter()
        .find(|path| stem(path) == Some(WITNESS_STEM))
        .cloned()
        .unwrap_or_else(|| {
            fail(format!(
                "`{}` 里没有 `{WITNESS_STEM}.<ext>`：设备目录必须带上见证，否则映射表没有入口可指",
                dir.display()
            ))
        });
    let ext =
        extension(&witness).unwrap_or_else(|| fail(format!("`{}` 没有扩展名", witness.display())));

    let doc = archive_of(dir, name);
    let meta = doc
        .get("meta")
        .and_then(|value| value.as_table())
        .unwrap_or_else(|| fail(format!("`{name}` 的档案没有 `[meta]` 表")));
    let provides = provides_of(&doc, name, &ext);
    let archive = dir.join(format!("{name}.toml"));

    let mut fragments = Vec::new();
    let mut entries = Vec::new();
    let mut epilogue = None;
    for path in files {
        if path == witness || path == archive {
            continue;
        }
        let prim = stem(&path)
            .unwrap_or_else(|| fail(format!("`{}` 的文件名不是合法 UTF-8", path.display())))
            .to_string();
        if extension(&path).as_deref() != Some(ext.as_str()) {
            fail(format!(
                "`{}` 的扩展名与同目录的 `{WITNESS_STEM}.{ext}` 不一致：一台设备只有一种输出语言",
                path.display()
            ));
        }
        if prim == EPILOGUE_STEM {
            epilogue = Some(path);
            continue;
        }
        let implementation = signature(&prim, &read(&path), &path);
        entries.push((
            prim.clone(),
            entry_of(&provides, &prim, &implementation, &path),
        ));
        fragments.push((prim, path));
    }
    if fragments.is_empty() {
        fail(format!("`{}` 里一段片段都没有", dir.display()));
    }
    let epilogue = epilogue.unwrap_or_else(|| {
        fail(format!(
            "`{}` 里没有 `{EPILOGUE_STEM}.{ext}`：六种写回形式没有片段，装配无从下手",
            dir.display()
        ))
    });
    fragments.sort();
    entries.sort();

    let forms = forms_of(&read(&epilogue), &epilogue);
    Device {
        name: name.to_string(),
        ext,
        emit: text(meta, "emit", name),
        tool: text(meta, "tools", name),
        tool_ver: meta
            .get("tool_ver")
            .and_then(|value| value.as_str())
            .unwrap_or("-")
            .to_string(),
        fragments,
        entries,
        epilogue,
        witness,
        archive,
        forms,
    }
}

/// 目录对应的档案：`src/<vendor>/<vendor>.toml`，目录名与 `meta.vendor` 必须一致。
fn archive_of(dir: &Path, name: &str) -> toml::Table {
    let path = dir.join(format!("{name}.toml"));
    if !path.is_file() {
        fail(format!(
            "`{}` 没有同名档案 `{name}.toml`：目录名就是 vendor，档案必须同名",
            dir.display()
        ));
    }
    let doc = buildgen::read(&path).unwrap_or_else(|error| fail(format!("{error}")));
    let vendor = doc
        .get("meta")
        .and_then(|value| value.as_table())
        .and_then(|meta| meta.get("vendor"))
        .and_then(|value| value.as_str())
        .unwrap_or_else(|| fail(format!("`{name}` 的档案没有 `meta.vendor`")));
    if vendor != name {
        fail(format!(
            "档案 `{name}.toml` 的 `meta.vendor` 是 `{vendor}`：目录名与 vendor 必须一致"
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
        .map(|provide| provide.func.clone())
        .unwrap_or_else(|| {
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
        })
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
    files.iter().any(|path| stem(path) == Some(WITNESS_STEM))
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

fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| fail(format!("读取 {} 失败：{error}", path.display())))
}

fn stem(path: &Path) -> Option<&str> {
    path.file_stem().and_then(|value| value.to_str())
}

fn extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|value| value.to_str())
        .map(str::to_string)
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

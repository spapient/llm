//! 构建期代码生成：一个设备目录 → 一份编译期表（`$OUT_DIR/tables.rs`）。

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::atom::{Epilogue, Prim};
use crate::prof::{Emit, Profile};
use crate::sign::EPILOGUE_STEM;

// ── scan ──────────────────────────────────────────────────────────────

/// 生成器入口：`CARGO_MANIFEST_DIR/src` 里的片段 → `$OUT_DIR/tables.rs`。
pub fn device(vendor: &str) {
    let dir = PathBuf::from(variable("CARGO_MANIFEST_DIR")).join("src");
    let out = PathBuf::from(variable("OUT_DIR"));
    let images = env::var_os("CARGO_FEATURE_DEVICE_IMAGE").is_some();

    // 整个 crate 目录都算输入：新增一段片段、换一份见证都必须让构建脚本重跑。
    println!("cargo:rerun-if-changed={}", dir.display());
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DEVICE_IMAGE");
    println!("cargo:rerun-if-env-changed=CUTILE_REQUIRE_IMAGE");

    let device = scan(&dir, vendor);
    // 镜像结局只算一次：生成的表与 `CUTILE_REQUIRE_IMAGE` 门禁读的是同一次编译的结果。
    let outcome = outcome(&device, &out, images);

    let mut text = String::new();
    header(&mut text);
    fragments(&mut text, &device);
    epilogues(&mut text, &device);
    device_line(&mut text, &device, &outcome);
    fs::write(out.join("tables.rs"), text)
        .unwrap_or_else(|error| fail(format!("写 tables.rs 失败：{error}")));

    require(&device, &outcome);
}

/// 构建脚本必带的环境变量；缺了说明调用方不是 `build.rs`，直接失败而不是猜一个目录。
fn variable(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| fail(format!("{name} 未设置：只有 build.rs 会调用这个函数")))
}

/// 一台设备：片段与路径（正文由生成的 `include_str!` 读）。
struct Device {
    /// vendor 名，与 `crates/prof/devices/<vendor>.toml` 的文件名一致。
    name: String,
    /// 片段与见证的扩展名，由 `emit` 语言决定。
    ext: String,
    /// 档案：工具链、编译参数与每条签名的见证入口都从这里来。
    profile: Profile,
    /// 片段与路径，顺序与 `Prim::ALL` 一致。
    fragments: Vec<(Prim, PathBuf)>,
    /// 写回尾块片段：一份文本，六种形式按 `/* epi: <form> */` 分段。
    epilogue: PathBuf,
    /// 写回形式 → 见证入口，顺序与 `Epilogue::ALL` 一致。
    epis: Vec<(Epilogue, String)>,
    /// 见证源码路径（正文由生成的 `include_str!` 读，脚本里不再存一份）。
    witness: PathBuf,
    /// 原语 → 见证入口（`int func(float *out)` 的函数名）。
    entries: Vec<(Prim, String)>,
}

/// 列出目录下的文件；目录打不开、或单个条目读失败（权限/竞态）都显式失败——
/// 静默跳过条目会让生成的表缺项而构建仍然绿。
fn files_in(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|error| fail(format!("读取 {} 失败：{error}", dir.display())));
    for entry in entries {
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

/// 扫描一台设备的目录：`<prim>.<ext>` 若干 + 一份 `witness.<ext>`。
fn scan(dir: &Path, vendor: &str) -> Device {
    let files = files_in(dir);
    let witness = witness_of(&files, dir);
    let ext = fragment_ext(&witness);
    let profile = profile_of(vendor, &ext);

    // 见证与清单不是片段（扩展名不符的被跳过）；每段片段带上自己的见证入口。
    let mut fragments = Vec::new();
    let mut entries = Vec::new();
    let mut epilogue = None;
    for path in &files {
        if *path == witness {
            continue;
        }
        if let Some(epis) = epilogue_of(path, &ext, &profile) {
            epilogue = Some((path.clone(), epis));
            continue;
        }
        if let Some((prim, entry)) = fragment_of(path, &ext, &profile) {
            fragments.push((prim, path.clone()));
            entries.push((prim, entry));
        }
    }
    if fragments.is_empty() {
        fail(format!("`{}` 里一段片段都没有", dir.display()));
    }
    fragments.sort_by_key(|(prim, _)| index(*prim));
    entries.sort_by_key(|(prim, _)| index(*prim));
    let Some((epilogue, epis)) = epilogue else {
        fail(format!(
            "`{}` 里没有 `{EPILOGUE_STEM}.{ext}`：六种写回形式没有片段，装配无从下手",
            dir.display()
        ))
    };

    Device {
        name: vendor.to_string(),
        ext,
        profile,
        fragments,
        epilogue,
        epis,
        witness,
        entries,
    }
}

/// 目录里那份见证：每台设备必须有，否则映射表没有入口可指。
fn witness_of(files: &[PathBuf], dir: &Path) -> PathBuf {
    files
        .iter()
        .find(|path| stem(path) == Some("witness"))
        .cloned()
        .unwrap_or_else(|| {
            fail(format!(
                "`{}` 里没有 `witness.<ext>`：设备目录必须带上见证，否则映射表没有入口可指",
                dir.display()
            ))
        })
}

/// 这台设备的输出语言：扩展名由见证给出，片段必须与它一致。
fn fragment_ext(witness: &Path) -> String {
    extension(witness).unwrap_or_else(|| fail(format!("`{}` 没有扩展名", witness.display())))
}

/// 一个文件是不是这台设备的片段：是就返回 `(原语, 见证入口)`，不是就 `None`。
fn fragment_of(path: &Path, ext: &str, profile: &Profile) -> Option<(Prim, String)> {
    let prim = Prim::parse(stem(path)?);
    if extension(path).as_deref() != Some(ext) {
        if prim.is_some() {
            fail(format!(
                "`{}` 的扩展名与同目录的 `witness.{ext}` 不一致：一台设备只有一种输出语言",
                path.display()
            ));
        }
        return None;
    }
    let prim = prim.unwrap_or_else(|| {
        fail(format!(
            "`{}` 的文件名不在原语闭集内：片段层只认 `<prim>.{ext}` 与 `witness.{ext}`",
            path.display()
        ))
    });
    let implementation = signature(prim, &read(path), path);
    let entry = entry_of(profile, prim, &implementation, path);
    Some((prim, entry))
}

/// 一个文件是不是写回尾块片段：是就返回六种形式的见证入口，不是就 `None`。
///
fn profile_of(vendor: &str, ext: &str) -> Profile {
    let table = crate::embedded(vendor).unwrap_or_else(|| {
        fail(format!(
            "`{vendor}` 没有对应的档案 `crates/prof/devices/{vendor}.toml`：\
             设备目录与档案必须同名"
        ))
    });
    let profile = Profile::parse(table)
        .unwrap_or_else(|error| fail(format!("`{vendor}` 的档案解析失败：{error}")));
    if profile.meta.vendor != vendor {
        fail(format!(
            "档案 `devices/{vendor}.toml` 的 `meta.vendor` 是 `{}`：目录名与 vendor 必须一致",
            profile.meta.vendor
        ));
    }
    if profile.meta.emit.ext() != ext {
        fail(format!(
            "`{vendor}` 的档案声明 `emit = {}`（扩展名 `{}`），目录里却是 `.{ext}`",
            profile.meta.emit,
            profile.meta.emit.ext()
        ));
    }
    profile
}

/// 片段第一行的签名声明：`/* cutile-sign: <prim>/<impl> */`。
fn signature(prim: Prim, source: &str, path: &Path) -> String {
    let first = source.lines().next().unwrap_or_default();
    let declared = crate::sign::strip_marker(first).unwrap_or_else(|| {
        fail(format!(
            "`{}` 的第一行必须是 `/* {} <prim>/<impl> */`，实际是 `{first}`",
            path.display(),
            crate::sign::MARKER
        ))
    });
    let (declared_prim, implementation) = crate::sign::split_slash(declared).unwrap_or_else(|| {
        fail(format!(
            "`{}` 的签名声明不是 `<prim>/<impl>`",
            path.display()
        ))
    });
    if Prim::parse(declared_prim.trim()) != Some(prim) {
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

/// 原语在设备镜像里的入口：档案里 `(prim, impl)` 那条供给的见证函数名。
fn entry_of(profile: &Profile, prim: Prim, implementation: &str, path: &Path) -> String {
    let provide = profile
        .provides
        .iter()
        .find(|provide| provide.prim == prim && provide.impl_name == implementation)
        .unwrap_or_else(|| {
            let declared: Vec<&str> = profile
                .provides
                .iter()
                .filter(|provide| provide.prim == prim)
                .map(|provide| provide.impl_name.as_str())
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
        });
    provide
        .witness
        .as_ref()
        .map(|witness| witness.func.clone())
        .unwrap_or_else(|| {
            fail(format!(
                "`{}` 声明的签名 `{prim}/{implementation}` 没有见证：映射表要指向镜像里的一个入口，\
                 先给这条签名写见证",
                path.display()
            ))
        })
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

fn index(prim: Prim) -> usize {
    Prim::ALL
        .iter()
        .position(|candidate| *candidate == prim)
        .unwrap_or(Prim::ALL.len())
}

/// 构建期失败：报给 `cargo` 并以非零码结束，`build.rs` 因此失败。
fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("prof::codegen: {message}");
    std::process::exit(1)
}

// ── image ─────────────────────────────────────────────────────────────

/// 一台设备的镜像结局。
enum Outcome {
    /// 未开启 `device-image`：只有源码。
    Source,
    /// 这台设备的输出语言还没有构建配方。
    NoRecipe,
    /// 有配方，但工具链不在本机或编译失败。
    Skipped(String),
    /// 编好了。
    Compiled { path: PathBuf, symbols: usize },
}

/// 编不出来的一行原因：它是 `Status::Skipped { reason }` 的**数据**载荷，要写进生成的代码，
/// 不是诊断——这条路径在构建期，不在 L7 的错误边界里，没有类别与退出码要承担。
struct Reason(String);

/// 按输出语言编设备镜像：`witness.<ext>` → 中间产物 → 镜像。
fn outcome(device: &Device, out: &Path, images: bool) -> Outcome {
    if !images {
        return Outcome::Source;
    }
    let Some(recipe) = recipe(device.profile.meta.emit) else {
        return Outcome::NoRecipe;
    };
    match compile(device, &recipe, out) {
        Ok(outcome) => outcome,
        Err(Reason(reason)) => Outcome::Skipped(reason),
    }
}

/// 编译一台设备的见证：`[tools] <compile> <link> <inspect>`。
fn compile(device: &Device, recipe: &Recipe, out: &Path) -> Result<Outcome, Reason> {
    let Some((program, prefix)) = tool(&device.profile.meta.tool) else {
        return Err(Reason(format!(
            "`{}` 不是可执行的工具名",
            device.profile.meta.tool
        )));
    };
    let object = out.join(format!("{}.witness.{}", device.name, recipe.object_ext));
    let image = out.join(format!("{}.witness.{}", device.name, recipe.format));

    let mut args: Vec<String> = prefix;
    args.extend(
        recipe
            .compile
            .iter()
            .map(|arg| expand(arg, &device.witness, &object, &image)),
    );
    run(&program, &args)?;

    if let Some((linker, link_args)) = recipe.link {
        let args: Vec<String> = link_args
            .iter()
            .map(|arg| expand(arg, &device.witness, &object, &image))
            .collect();
        run(linker, &args)?;
    }

    let symbols = recipe
        .inspect
        .and_then(|(inspector, inspect_args)| {
            let args: Vec<String> = inspect_args
                .iter()
                .map(|arg| expand(arg, &device.witness, &object, &image))
                .collect();
            run(inspector, &args).ok()
        })
        .map(|stdout| stdout.lines().filter(|line| line.contains(" T ")).count())
        .unwrap_or(0);

    Ok(Outcome::Compiled {
        path: image,
        symbols,
    })
}

/// 一条语言级的构建配方：怎么把一份翻译单元变成设备镜像。
struct Recipe {
    /// 编译参数（`$src` / `$obj`）。
    compile: &'static [&'static str],
    /// 链接一步：程序与参数（`$obj` / `$out`）。
    link: Option<(&'static str, &'static [&'static str])>,
    /// 中间产物的扩展名。
    object_ext: &'static str,
    /// 镜像格式名。
    format: &'static str,
    /// 符号统计：程序与参数（`$obj`），每行一个符号，` T ` 是已定义的文本符号。
    inspect: Option<(&'static str, &'static [&'static str])>,
}

fn recipe(emit: Emit) -> Option<Recipe> {
    match emit {
        Emit::Metal => Some(Recipe {
            compile: &["-c", "$src", "-o", "$obj"],
            link: Some(("xcrun", &["metallib", "$obj", "-o", "$out"])),
            object_ext: "air",
            format: "metallib",
            inspect: Some(("xcrun", &["metal-nm", "$obj"])),
        }),
        _ => None,
    }
}

/// 这台设备的镜像格式名；没有配方时是 `""`（该分支不会生成镜像）。
fn format(emit: Emit) -> &'static str {
    recipe(emit).map_or("", |recipe| recipe.format)
}

/// `tools` 拆成程序与固定前缀参数（档案里写 `xcrun metal`）。
fn tool(text: &str) -> Option<(String, Vec<String>)> {
    let mut parts = text.split_whitespace();
    let program = parts.next()?.to_string();
    Some((program, parts.map(str::to_string).collect()))
}

/// 占位符替换：`$src` / `$obj` / `$out`。
fn expand(arg: &str, src: &Path, object: &Path, image: &Path) -> String {
    arg.replace("$src", &src.display().to_string())
        .replace("$obj", &object.display().to_string())
        .replace("$out", &image.display().to_string())
}

/// 跑一个子进程；失败时把第一行 stderr 作为原因交回（构建不因此失败）。
fn run(program: &str, args: &[String]) -> Result<String, Reason> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| Reason(format!("`{program}` 不可执行：{error}")))?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let first = stderr
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    Err(Reason(format!(
        "`{program} {}` 失败（{}）：{}",
        args.join(" "),
        output.status,
        shorten(first)
    )))
}

/// 原因只留一行、最多 160 字符：它要进生成的代码。
fn shorten(text: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= 160 {
        return text.to_string();
    }
    let cut: String = text.chars().take(157).collect();
    format!("{cut}...")
}

/// `CUTILE_REQUIRE_IMAGE` 要求了这台设备就必须真的带上镜像。
fn require(device: &Device, outcome: &Outcome) {
    let required = env::var("CUTILE_REQUIRE_IMAGE").ok().is_some_and(|text| {
        text.split(',')
            .map(str::trim)
            .any(|name| name == device.name)
    });
    if !required {
        return;
    }
    match outcome {
        Outcome::Compiled { .. } => {}
        Outcome::Source => fail(format!(
            "`CUTILE_REQUIRE_IMAGE` 要求 `{}` 的设备镜像，但构建没有开 `device-image`",
            device.name
        )),
        Outcome::NoRecipe => fail(format!(
            "`CUTILE_REQUIRE_IMAGE` 要求 `{}` 的设备镜像，但它的输出语言还没有构建配方",
            device.name
        )),
        Outcome::Skipped(reason) => fail(format!(
            "`CUTILE_REQUIRE_IMAGE` 要求 `{}` 的设备镜像，但它没有编出来：{reason}",
            device.name
        )),
    }
}

// ── table ─────────────────────────────────────────────────────────────

/// 文件头：说明这是生成物，并把 `Prim` 拉进作用域。
fn header(text: &mut String) {
    let _ = writeln!(
        text,
        "// 由 `crates/prof/src/codegen` 生成，不要手改。\n\
         //\n\
         // 一台设备的片段、入口与设备行；`prim` 把六份这样的表合成 `BUILTIN` / `WITNESS` /\n\
         // `DEVICES`。类型定义在 `src/` 与 `prof`，所以生成代码不改变所有权表。\n\
         #[allow(unused_imports)]\n\
         use prof::atom::{{Epilogue, Prim}};"
    );
}

/// 生成片段表：`(原语, 文本)`，顺序与 `Prim::ALL` 一致。
fn fragments(text: &mut String, device: &Device) {
    let _ = writeln!(
        text,
        "/// 片段：原语 → 文本，顺序与 `Prim::ALL` 一致。\n\
         pub const FRAGMENTS: [(Prim, &str); {}] = [",
        device.fragments.len()
    );
    for (prim, path) in &device.fragments {
        let _ = writeln!(
            text,
            "    (Prim::{:?}, include_str!({:?})),",
            prim,
            path.display().to_string()
        );
    }
    let _ = writeln!(text, "];\n");
}

/// 生成写回尾块表：一份文本 + 六种形式各自的见证入口。
fn epilogues(text: &mut String, device: &Device) {
    let _ = writeln!(
        text,
        "/// 写回尾块片段：`/* epi: <form> */` 分成六段，装配按段取用。\n\
         pub const EPILOGUE: &str = include_str!({:?});\n",
        device.epilogue.display().to_string()
    );
    let _ = writeln!(
        text,
        "/// 写回形式 → 见证里的入口名，顺序与 `Epilogue::ALL` 一致。\n\
         pub const EPIS: [(Epilogue, &str); {}] = [",
        device.epis.len()
    );
    for (form, entry) in &device.epis {
        let _ = writeln!(text, "    (Epilogue::{:?}, {entry:?}),", form);
    }
    let _ = writeln!(text, "];\n");
}

/// 生成设备行：入口表 + 档案元数据 + 见证源码 + （可选的）镜像。
fn device_line(text: &mut String, device: &Device, outcome: &Outcome) {
    prims(text, device);

    let _ = writeln!(
        text,
        "/// 设备行：`prim` 把它连同另外五台设备合成 `DEVICES`。\n\
         pub const DEVICE: prof::Device = prof::Device {{"
    );
    let _ = writeln!(text, "    vendor: {:?},", device.name);
    let _ = writeln!(text, "    ext: {:?},", device.ext);
    let _ = writeln!(text, "    emit: {:?},", device.profile.meta.emit.as_str());
    let _ = writeln!(text, "    tool: {:?},", device.profile.meta.tool);
    let _ = writeln!(text, "    tool_ver: {:?},", device.profile.meta.tool_ver);
    let _ = writeln!(text, "    prims: &PRIMS,");
    let _ = writeln!(text, "    epilogue: EPILOGUE,");
    let _ = writeln!(text, "    epis: &EPIS,");
    let _ = writeln!(
        text,
        "    source: include_str!({:?}),",
        device.witness.display().to_string()
    );
    image_line(text, device, outcome);
    let _ = writeln!(text, "    status: {},", status(outcome));
    let _ = writeln!(text, "}};");
}

/// 入口表：原语 → 见证里的入口名，顺序与 `Prim::ALL` 一致。
fn prims(text: &mut String, device: &Device) {
    let _ = writeln!(
        text,
        "/// 原语 → 见证里的入口名：镜像里查的就是这个名字。\n\
         pub const PRIMS: [(Prim, &str); {}] = [",
        device.entries.len()
    );
    for (prim, entry) in &device.entries {
        let _ = writeln!(text, "    (Prim::{:?}, {entry:?}),", prim);
    }
    let _ = writeln!(text, "];\n");
}

/// 镜像行：只有本机真编出镜像的设备带 `Some(prof::Image::new(..))`，其余写 `None`。
fn image_line(text: &mut String, device: &Device, outcome: &Outcome) {
    match outcome {
        Outcome::Compiled { path, symbols } => {
            let format = format(device.profile.meta.emit);
            let _ = writeln!(
                text,
                "    image: Some(prof::Image::new({format:?}, include_bytes!({:?}), {symbols})),",
                path.display().to_string()
            );
        }
        _ => {
            let _ = writeln!(text, "    image: None,");
        }
    }
}

/// 结局闭集 → `prof::Status` 的构造表达式（"为什么没有镜像"永远有答案）。
fn status(outcome: &Outcome) -> String {
    match outcome {
        Outcome::Source => "prof::Status::Source".to_string(),
        Outcome::NoRecipe => "prof::Status::NoRecipe".to_string(),
        Outcome::Skipped(reason) => format!("prof::Status::Skipped {{ reason: {reason:?} }}"),
        Outcome::Compiled { .. } => "prof::Status::Compiled".to_string(),
    }
}

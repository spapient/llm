//! 构建期生成器：目录 → 枚举与映射表；能编就编成设备镜像。
//!
//! 手写登记的代价是漂移：`BUILTIN` / `WITNESS` 每加一台设备都要补一行 `include_str!`，漏一行就是
//! "发布出去的二进制里没有它"，而这类错误只有测试能发现。构建脚本在编译期就能读目录，所以三张表
//! ——片段（`BUILTIN`）、见证（`WITNESS`）、每台设备的原语枚举与映射表（`DEVICES`）——由扫描生成，
//! 加设备只需要放文件。
//!
//! 生成与校验走同一趟，数据错在构建期就失败：
//!
//! 1. 文件名必须是 `prof::atom::Prim` 闭集里的名字（运行期 [`prim::load`] 也这么判，但那时已经是
//!    用户的机器上）；
//! 2. 片段第一行的签名声明 `/* cutile-sign: <prim>/<impl> */` 必须在档案里有对应的 `provides`；
//! 3. 那条供给必须带见证——映射表指向镜像里的一个入口，指不到就不该生成。
//!
//! `device-image` feature 打开时，再按输出语言把 `<vendor>/witness.<ext>` 编成设备镜像（`metal`：
//! `xcrun metal -c` + `xcrun metallib`）并嵌入。工具链不在本机、或编译失败时只记
//! `Skipped { reason }`，构建照常成功——源码永远可用。发布流水线要"缺镜像即失败"时，设
//! `CUTILE_REQUIRE_IMAGE=<vendor>[,<vendor>]` 即可，它把"缺镜像"升级成构建失败。
//!
//! 片段本身不进镜像：片段是语句块，`$0..$3` 的绑定由算子模板的 `@prim(...)` 给出，脱离模板单独
//! 编译只能编出一段与真实调用无关的代码。能编成设备镜像的最小单位是 `<vendor>/witness.<ext>`
//! ——一份完整翻译单元、固定 ABI、每条签名一个入口。

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use prof::atom::Prim;
use prof::prof::{Emit, Profile};

// 签名剥壳与切分规则与 crate 装载期共用同一份纯函数源码（ADR 0027 §5）：
// 构建脚本不依赖正在构建的 crate 编译产物，只 include 这份源码文本，规则因此只有一份。
include!("src/signature.rs");

/// 生成器入口。
fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let images = env::var_os("CARGO_FEATURE_DEVICE_IMAGE").is_some();
    let required = required_images();

    // 整个 crate 目录都算输入：新增一台设备只放文件，构建脚本必须因此重跑。
    println!("cargo:rerun-if-changed={}", manifest.display());
    // 档案层由 `prof` 在构建期生成，但“目录名即档案名”这条约束在这里校验，档案改了也要重跑。
    let profiles = manifest
        .parent()
        .unwrap_or(&manifest)
        .join("prof")
        .join("devices");
    println!("cargo:rerun-if-changed={}", profiles.display());
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DEVICE_IMAGE");
    println!("cargo:rerun-if-env-changed=CUTILE_REQUIRE_IMAGE");

    let vendors = scan(&manifest);

    let mut text = String::new();
    header(&mut text);
    builtin(&mut text, &vendors);
    witness(&mut text, &vendors);
    let compiled = devices(&mut text, &vendors, &out, images);
    fs::write(out.join("tables.rs"), text).expect("写 tables.rs");

    for vendor in required {
        match compiled.iter().find(|(name, _)| *name == vendor) {
            None => fail(format!(
                "`CUTILE_REQUIRE_IMAGE` 里的 `{vendor}` 不是片段层里的设备"
            )),
            Some((_, Some(_))) => {}
            Some((_, None)) => fail(format!(
                "`CUTILE_REQUIRE_IMAGE` 要求 `{vendor}` 的设备镜像，但它没有编出来\
                 （需要 `--features device-image`，且本机有该设备的工具链）"
            )),
        }
    }
}

/// 片段层里的一个设备目录：`<vendor>/<prim>.<ext>` 加一份 `witness.<ext>`。
struct Vendor {
    /// 目录名，与 `prof/devices/<vendor>.toml` 的文件名一致。
    name: String,
    /// 片段与见证的扩展名，由 `emit` 语言决定。
    ext: String,
    /// 档案：工具链、编译参数与每条签名的见证入口都从这里来。
    profile: Profile,
    /// 片段与路径，顺序与 `Prim::ALL` 一致（正文由生成的 `include_str!` 读）。
    fragments: Vec<(Prim, PathBuf)>,
    /// 见证源码路径（正文由生成的 `include_str!` 读，脚本里不再存一份）。
    witness: PathBuf,
    /// 原语 → 见证入口（`int func(float *out)` 的函数名）。
    entries: Vec<(Prim, String)>,
}

/// 列出目录下的条目路径；`want_files` 为真收文件，否则收子目录。
///
/// 目录打不开、或单个条目读失败（权限/竞态）都显式失败：静默跳过条目会让生成表缺项
/// 而构建仍然绿，漏一台设备只能等到运行期才暴露。
fn read_dir_paths(dir: &Path, want_files: bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|error| fail(format!("读取 {} 失败：{error}", dir.display())));
    for entry in entries {
        let entry = entry
            .unwrap_or_else(|error| fail(format!("读取 {} 的目录项失败：{error}", dir.display())));
        let path = entry.path();
        if path.is_file() == want_files {
            out.push(path);
        }
    }
    out
}

/// 扫描 `<root>/<vendor>/`：有见证的目录才算设备目录，其余（`src`、`target`）跳过。
fn scan(root: &Path) -> Vec<Vendor> {
    let mut dirs = read_dir_paths(root, false);
    dirs.sort();

    let mut vendors = Vec::new();
    for dir in dirs {
        let files = read_dir_paths(&dir, true);
        let witness = files
            .iter()
            .find(|path| stem(path) == Some("witness"))
            .cloned();
        let Some(witness) = witness else {
            // 没有见证的目录不是设备目录；但如果它放着片段，那就是漏了见证。
            if let Some(path) = files.iter().find(|path| prim_of(path).is_some()) {
                fail(format!(
                    "`{}` 看起来是片段（文件名在原语闭集内），但同目录没有 `witness.<ext>`：\
                     设备目录必须带上见证，否则映射表没有入口可指",
                    path.display()
                ));
            }
            continue;
        };
        let name = dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_else(|| fail(format!("`{}` 的目录名不是合法 UTF-8", dir.display())))
            .to_string();
        let ext = extension(&witness)
            .unwrap_or_else(|| fail(format!("`{}` 没有扩展名", witness.display())));

        let profile = profile_of(&name, &ext);
        let mut fragments = Vec::new();
        let mut entries = Vec::new();
        for path in &files {
            if *path == witness {
                continue;
            }
            let Some(prim) = prim_of(path) else {
                fail(format!(
                    "`{}` 的文件名不在原语闭集内：片段层只认 `<prim>.<ext>` 与 `witness.<ext>`",
                    path.display()
                ));
            };
            if extension(path).as_deref() != Some(ext.as_str()) {
                fail(format!(
                    "`{}` 的扩展名与同目录的 `witness.{ext}` 不一致：一台设备只有一种输出语言",
                    path.display()
                ));
            }
            let source = read(path);
            let implementation = signature(prim, &source, path);
            let entry = entry_of(&profile, prim, &implementation, path);
            fragments.push((prim, path.clone()));
            entries.push((prim, entry));
        }
        if fragments.is_empty() {
            fail(format!("`{}` 里一段片段都没有", dir.display()));
        }
        fragments.sort_by_key(|(prim, _)| index(*prim));
        entries.sort_by_key(|(prim, _)| index(*prim));

        vendors.push(Vendor {
            name,
            ext,
            profile,
            fragments,
            witness,
            entries,
        });
    }
    if vendors.is_empty() {
        fail(format!(
            "`{}` 里没有任何设备目录（`<vendor>/` + `witness.<ext>`）",
            root.display()
        ));
    }
    vendors
}

/// 目录对应的档案：名字即文件名，`vendor` / `emit` 必须与目录一致。
fn profile_of(vendor: &str, ext: &str) -> Profile {
    let table = prof::embedded(vendor).unwrap_or_else(|| {
        fail(format!(
            "`{vendor}/` 没有对应的档案 `crates/prof/devices/{vendor}.toml`：\
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
///
/// 剥壳与切分走 [`include!`] 进来的共享规则（ADR 0027 §5）；这里只补原语比对与构建失败。
fn signature(prim: Prim, source: &str, path: &Path) -> String {
    let first = source.lines().next().unwrap_or_default();
    let declared = strip_marker(first).unwrap_or_else(|| {
        fail(format!(
            "`{}` 的第一行必须是 `/* {MARKER} <prim>/<impl> */`，实际是 `{first}`",
            path.display()
        ))
    });
    let (declared_prim, implementation) = split_slash(declared).unwrap_or_else(|| {
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

/// 生成 `BUILTIN`：`(vendor, ext, 原语 → 片段文本)`。
fn builtin(text: &mut String, vendors: &[Vendor]) {
    let _ = writeln!(
        text,
        "/// 编译期内嵌的片段层：`build.rs` 在编译期扫描 `<vendor>/<prim>.<ext>` 得到。\n\
         ///\n\
         /// 顺序按目录名，片段按 `Prim::ALL`；新增设备只需要放文件，不必再改这张表。"
    );
    let _ = writeln!(text, "pub const BUILTIN: [Builtin; {}] = [", vendors.len());
    for vendor in vendors {
        let _ = writeln!(text, "    Builtin {{");
        let _ = writeln!(text, "        vendor: {:?},", vendor.name);
        let _ = writeln!(text, "        ext: {:?},", vendor.ext);
        let _ = writeln!(text, "        fragments: &[");
        for (prim, path) in &vendor.fragments {
            let _ = writeln!(
                text,
                "            (Prim::{:?}, include_str!({:?})),",
                prim,
                path.display().to_string()
            );
        }
        let _ = writeln!(text, "        ],");
        let _ = writeln!(text, "    }},");
    }
    let _ = writeln!(text, "];\n");
}

/// 生成 `WITNESS`：`(vendor, ext, 见证源码)`。
fn witness(text: &mut String, vendors: &[Vendor]) {
    let _ = writeln!(
        text,
        "/// 编译期内嵌的见证源码：`build.rs` 扫描 `<vendor>/witness.<ext>` 得到。\n\
         ///\n\
         /// 与 [`BUILTIN`] 分开：见证不是片段（装载时按名字跳过），但发布机上没有源码树，\n\
         /// 执行 L2 证据与设备镜像都还需要它。"
    );
    let _ = writeln!(
        text,
        "pub const WITNESS: [(&str, &str, &str); {}] = [",
        vendors.len()
    );
    for vendor in vendors {
        let _ = writeln!(
            text,
            "    ({:?}, {:?}, include_str!({:?})),",
            vendor.name,
            vendor.ext,
            vendor.witness.display().to_string()
        );
    }
    let _ = writeln!(text, "];\n");
}

/// 生成 `DEVICES`：每台设备的原语枚举 + 映射表，以及（可选）编译好的设备镜像。
///
/// 返回每台设备"这次是否带上了镜像"，供 `CUTILE_REQUIRE_IMAGE` 判定。
fn devices(
    text: &mut String,
    vendors: &[Vendor],
    out: &Path,
    images: bool,
) -> Vec<(String, Option<PathBuf>)> {
    let _ = writeln!(
        text,
        "/// 每台设备这次构建带上的东西：原语枚举 + 映射表，以及可选的设备镜像。\n\
         ///\n\
         /// `prims` 与 [`BUILTIN`] 同源（同一个目录扫描），它多出来的是**入口名**：原语在这台设备的\n\
         /// 镜像里叫什么。`source` 永远在；`image` 只在 `device-image` feature 打开且本机工具链真的\n\
         /// 编出了产物时才有值，`status` 说明为什么没有。"
    );
    let _ = writeln!(text, "pub static DEVICES: [Device; {}] = [", vendors.len());
    let mut compiled = Vec::new();
    for vendor in vendors {
        let (status, image) = match image_of(vendor, out, images) {
            Outcome::Source => ("Status::Source".to_string(), None),
            Outcome::NoRecipe => ("Status::NoRecipe".to_string(), None),
            Outcome::Skipped(reason) => {
                (format!("Status::Skipped {{ reason: {:?} }}", reason), None)
            }
            Outcome::Compiled { path, symbols } => (
                "Status::Compiled".to_string(),
                Some((
                    path,
                    symbols,
                    recipe(vendor.profile.meta.emit).map_or("", |r| r.format),
                )),
            ),
        };
        compiled.push((
            vendor.name.clone(),
            image.as_ref().map(|(path, _, _)| path.clone()),
        ));

        let _ = writeln!(text, "    Device {{");
        let _ = writeln!(text, "        vendor: {:?},", vendor.name);
        let _ = writeln!(text, "        ext: {:?},", vendor.ext);
        let _ = writeln!(
            text,
            "        emit: {:?},",
            vendor.profile.meta.emit.as_str()
        );
        let _ = writeln!(text, "        tools: {:?},", vendor.profile.meta.tool);
        let _ = writeln!(
            text,
            "        tool_ver: {:?},",
            vendor.profile.meta.tool_ver
        );
        let _ = writeln!(text, "        prims: &[");
        for (prim, entry) in &vendor.entries {
            let _ = writeln!(text, "            (Prim::{:?}, {entry:?}),", prim);
        }
        let _ = writeln!(text, "        ],");
        let _ = writeln!(
            text,
            "        source: include_str!({:?}),",
            vendor.witness.display().to_string()
        );
        match &image {
            None => {
                let _ = writeln!(text, "        image: None,");
            }
            Some((path, symbols, format)) => {
                let _ = writeln!(
                    text,
                    "        image: Some(Image::new({format:?}, include_bytes!({:?}), {symbols})),",
                    path.display().to_string()
                );
            }
        }
        let _ = writeln!(text, "        status: {status},");
        let _ = writeln!(text, "    }},");
    }
    let _ = writeln!(text, "];");
    compiled
}

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

/// 按输出语言编设备镜像：`<vendor>/witness.<ext>` → 中间产物 → 镜像。
fn image_of(vendor: &Vendor, out: &Path, images: bool) -> Outcome {
    if !images {
        return Outcome::Source;
    }
    let Some(recipe) = recipe(vendor.profile.meta.emit) else {
        return Outcome::NoRecipe;
    };
    match compile_image(vendor, &recipe, out) {
        Ok(outcome) => outcome,
        Err(reason) => Outcome::Skipped(reason),
    }
}

/// 编译一台设备的见证：`[tools] <compile> <link> <inspect>`。
///
/// 不传 `[compile].flags` / `preamble`：见证是 **host C ABI**（`int func(float *out)`），与片段
/// 语言无关，`cutile verify` 的本机自检也因为同一个原因跳过它们。`metal` 的 `-std=metal3.1`
/// 会要求指针写明地址空间，而见证里的 `float *` 故意没有——那是 host C 的写法。
fn compile_image(vendor: &Vendor, recipe: &Recipe, out: &Path) -> Result<Outcome, String> {
    let Some((program, prefix)) = tool(&vendor.profile.meta.tool) else {
        return Err(format!("`{}` 不是可执行的工具名", vendor.profile.meta.tool));
    };
    let object = out.join(format!("{}.witness.{}", vendor.name, recipe.object_ext));
    let image = out.join(format!("{}.witness.{}", vendor.name, recipe.format));

    let mut args: Vec<String> = prefix;
    args.extend(
        recipe
            .compile
            .iter()
            .map(|arg| expand(arg, &vendor.witness, &object, &image)),
    );
    run(&program, &args)?;

    if let Some((linker, link_args)) = recipe.link {
        let args: Vec<String> = link_args
            .iter()
            .map(|arg| expand(arg, &vendor.witness, &object, &image))
            .collect();
        run(linker, &args)?;
    }

    let symbols = recipe
        .inspect
        .and_then(|(inspector, inspect_args)| {
            let args: Vec<String> = inspect_args
                .iter()
                .map(|arg| expand(arg, &vendor.witness, &object, &image))
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
///
/// 只有 `metal` 有配方，原因是数据本身：见证是 `int func(float *out)` 的完整翻译单元，
/// MSL 允许这样的普通函数进 `metallib`；CUDA / OpenCL / AscendC 的镜像只收 `__global__` /
/// `__kernel__` 入口，把 host 见证编进去只会得到一份空产物——那比"没有"更糟。
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
fn run(program: &str, args: &[String]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("`{program}` 不可执行：{error}"))?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let first = stderr
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    Err(format!(
        "`{program} {}` 失败（{}）：{}",
        args.join(" "),
        output.status,
        shorten(first)
    ))
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

/// `CUTILE_REQUIRE_IMAGE` 里的设备名。
fn required_images() -> Vec<String> {
    env::var("CUTILE_REQUIRE_IMAGE")
        .ok()
        .map(|text| {
            text.split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn header(text: &mut String) {
    let _ = writeln!(
        text,
        "// 由 `crates/prim/build.rs` 生成，不要手改。\n\
         //\n\
         // 三张表都来自目录扫描（`<vendor>/<prim>.<ext>` + `<vendor>/witness.<ext>`），\n\
         // 类型定义在 `src/`，所以生成代码不改变所有权表。\n\
         #[allow(unused_imports)]\n\
         use crate::{{Builtin, Device, Image, Status}};\n\
         use prof::atom::Prim;"
    );
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

/// 文件名对应的原语；闭集外返回 `None`。
fn prim_of(path: &Path) -> Option<Prim> {
    stem(path).and_then(Prim::parse)
}

fn index(prim: Prim) -> usize {
    Prim::ALL
        .iter()
        .position(|candidate| *candidate == prim)
        .unwrap_or(Prim::ALL.len())
}

fn fail(message: impl std::fmt::Display) -> ! {
    panic!("prim/build.rs: {message}")
}

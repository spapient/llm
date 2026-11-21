// 构建期共享生成器：一个设备目录 → 一份编译期表。
//
// 六台设备各有自己的 crate（`cuda` / `cpu` / `opencl` / `metal` / `musa` / `rocm`），每个 crate 的
// `build.rs` 各 `include!` 一次这份源码——构建脚本不能依赖正在编译的 crate，只能共用源码文本
// （与 `src/signature.rs` 的同一手法，ADR 0027 §5）。于是"扫描 → 校验 → 生成"只有一份，
// 加一台设备仍然只是放文件；每个 `build.rs` 只多出十行 `main`。
//
// 校验与生成同一趟，数据错在构建期就失败：
//
// 1. 文件名必须是 `prof::atom::Prim` 闭集里的名字（运行期 `prim::load` 也这么判，但那时已经是
//    用户的机器上）；
// 2. 片段第一行的签名声明 `/* cutile-sign: <prim>/<impl> */` 必须在档案里有对应的 `provides`；
// 3. 那条供给必须带见证——映射表指向一个入口，指不到就不该生成。
//
// `device-image` feature 打开时，再按输出语言把 `witness.<ext>` 编成设备镜像（`metal`：
// `xcrun metal -c` + `xcrun metallib`）并嵌入。工具链不在本机、或编译失败时只记
// `Skipped { reason }`，构建照常成功——源码永远可用。发布流水线要"缺镜像即失败"时，设
// `CUTILE_REQUIRE_IMAGE=<vendor>[,<vendor>]` 即可，它把"这台设备缺镜像"升级成构建失败。
//
// 片段本身不进镜像：片段是语句块，`$0..$3` 的绑定由算子模板的 `@prim(...)` 给出，脱离模板单独
// 编译只能编出一段与真实调用无关的代码。能编成设备镜像的最小单位是 `witness.<ext>`——一份完整
// 翻译单元、固定 ABI、每条签名一个入口。
//
// 本文件被 `include!` 进构建脚本，所以只能用 `//` 注释：`//!` 与 `///` 在 include 之后非法。

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use prof::atom::Prim;
use prof::prof::{Emit, Profile};

// 签名剥壳与切分规则与 crate 装载期共用同一份纯函数源码（ADR 0027 §5）。
include!("signature.rs");

/// 生成器入口：`dir` 里的片段 → `out/tables.rs`。
///
/// `vendor` 必须与 `prof/devices/<vendor>.toml` 同名——它是内嵌表的键，也是档案里 `witness.file`
/// 的前缀。目录名可以与之不同（`crates/opencl` 装的是 `intel` 的 OpenCL 片段），
/// 所以这里显式传入，不从路径推断。
fn generate(dir: &Path, vendor: &str, out: &Path, images: bool) {
    // 整个 crate 目录都算输入：新增一段片段、换一份见证都必须让构建脚本重跑。
    println!("cargo:rerun-if-changed={}", dir.display());
    // 档案由 `prof` 在构建期生成，但"目录名即档案名"这条约束在这里校验，档案改了也要重跑。
    let profiles = dir
        .parent()
        .unwrap_or(dir)
        .join("prof")
        .join("devices");
    println!("cargo:rerun-if-changed={}", profiles.display());
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DEVICE_IMAGE");
    println!("cargo:rerun-if-env-changed=CUTILE_REQUIRE_IMAGE");

    let device = scan(dir, vendor);
    // 镜像结局只算一次：生成的表与 `CUTILE_REQUIRE_IMAGE` 门禁读的是同一次编译的结果。
    let outcome = image_of(&device, out, images);

    let mut text = String::new();
    header(&mut text);
    fragments(&mut text, &device);
    device_table(&mut text, &device, &outcome);
    fs::write(out.join("tables.rs"), text).expect("写 tables.rs");

    require(&device, &outcome);
}

/// 一台设备：片段与路径（正文由生成的 `include_str!` 读）。
struct Device {
    /// vendor 名，与 `prof/devices/<vendor>.toml` 的文件名一致。
    name: String,
    /// 片段与见证的扩展名，由 `emit` 语言决定。
    ext: String,
    /// 档案：工具链、编译参数与每条签名的见证入口都从这里来。
    profile: Profile,
    /// 片段与路径，顺序与 `Prim::ALL` 一致。
    fragments: Vec<(Prim, PathBuf)>,
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
///
/// 同目录还放着清单、构建脚本与文档，它们不是片段层的一部分：扩展名不符的一律跳过，
/// 但**名字在原语闭集内而扩展名不符**的要报——那是把片段放错了语言。
fn scan(dir: &Path, vendor: &str) -> Device {
    let files = files_in(dir);
    let witness = files
        .iter()
        .find(|path| stem(path) == Some("witness"))
        .cloned()
        .unwrap_or_else(|| {
            fail(format!(
                "`{}` 里没有 `witness.<ext>`：设备目录必须带上见证，否则映射表没有入口可指",
                dir.display()
            ))
        });
    let ext = extension(&witness).unwrap_or_else(|| fail(format!("`{}` 没有扩展名", witness.display())));

    let profile = profile_of(vendor, &ext);
    let mut fragments = Vec::new();
    let mut entries = Vec::new();
    for path in &files {
        if *path == witness {
            continue;
        }
        let Some(name) = stem(path) else { continue };
        let prim = Prim::parse(name);
        if extension(path).as_deref() != Some(ext.as_str()) {
            if prim.is_some() {
                fail(format!(
                    "`{}` 的扩展名与同目录的 `witness.{ext}` 不一致：一台设备只有一种输出语言",
                    path.display()
                ));
            }
            continue;
        }
        let Some(prim) = prim else {
            fail(format!(
                "`{}` 的文件名不在原语闭集内：片段层只认 `<prim>.{ext}` 与 `witness.{ext}`",
                path.display()
            ));
        };
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

    Device {
        name: vendor.to_string(),
        ext,
        profile,
        fragments,
        witness,
        entries,
    }
}

/// 目录对应的档案：`vendor` / `emit` 必须与目录一致。
fn profile_of(vendor: &str, ext: &str) -> Profile {
    let table = prof::embedded(vendor).unwrap_or_else(|| {
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
///
/// 剥壳与切分走 `include!` 进来的共享规则（ADR 0027 §5）；这里只补原语比对与构建失败。
fn signature(prim: Prim, source: &str, path: &Path) -> String {
    let first = source.lines().next().unwrap_or_default();
    let declared = strip_marker(first).unwrap_or_else(|| {
        fail(format!(
            "`{}` 的第一行必须是 `/* {MARKER} <prim>/<impl> */`，实际是 `{first}`",
            path.display()
        ))
    });
    let (declared_prim, implementation) = split_slash(declared).unwrap_or_else(|| {
        fail(format!("`{}` 的签名声明不是 `<prim>/<impl>`", path.display()))
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

/// 生成片段表：`(原语, 文本)`，顺序与 `Prim::ALL` 一致。
fn fragments(text: &mut String, device: &Device) {
    let _ = writeln!(
        text,
        "// 片段：原语 → 文本，顺序与 `Prim::ALL` 一致。\n\
         pub const FRAGMENTS: [(Prim, &'static str); {}] = [",
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

/// 生成设备行：入口表 + 档案元数据 + 见证源码 + （可选的）镜像。
fn device_table(text: &mut String, device: &Device, outcome: &Outcome) {
    let _ = writeln!(
        text,
        "// 原语 → 见证里的入口名：镜像里查的就是这个名字。\n\
         pub const PRIMS: [(Prim, &'static str); {}] = [",
        device.entries.len()
    );
    for (prim, entry) in &device.entries {
        let _ = writeln!(text, "    (Prim::{:?}, {entry:?}),", prim);
    }
    let _ = writeln!(text, "];\n");

    let (status, image) = match outcome {
        Outcome::Source => ("prof::Status::Source".to_string(), None),
        Outcome::NoRecipe => ("prof::Status::NoRecipe".to_string(), None),
        Outcome::Skipped(reason) => {
            (format!("prof::Status::Skipped {{ reason: {reason:?} }}"), None)
        }
        Outcome::Compiled { path, symbols } => (
            "prof::Status::Compiled".to_string(),
            Some((
                path,
                *symbols,
                recipe(device.profile.meta.emit).map_or("", |r| r.format),
            )),
        ),
    };

    let _ = writeln!(
        text,
        "// 设备行：`prim` 把它连同另外五台设备合成 `DEVICES`。\n\
         pub const DEVICE: prof::Device = prof::Device {{"
    );
    let _ = writeln!(text, "    vendor: {:?},", device.name);
    let _ = writeln!(text, "    ext: {:?},", device.ext);
    let _ = writeln!(text, "    emit: {:?},", device.profile.meta.emit.as_str());
    let _ = writeln!(text, "    tool: {:?},", device.profile.meta.tool);
    let _ = writeln!(text, "    tool_ver: {:?},", device.profile.meta.tool_ver);
    let _ = writeln!(text, "    prims: &PRIMS,");
    let _ = writeln!(
        text,
        "    source: include_str!({:?}),",
        device.witness.display().to_string()
    );
    match &image {
        None => {
            let _ = writeln!(text, "    image: None,");
        }
        Some((path, symbols, format)) => {
            let _ = writeln!(
                text,
                "    image: Some(prof::Image::new({format:?}, include_bytes!({:?}), {symbols})),",
                path.display().to_string()
            );
        }
    }
    let _ = writeln!(text, "    status: {status},");
    let _ = writeln!(text, "}};");
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

/// 按输出语言编设备镜像：`witness.<ext>` → 中间产物 → 镜像。
fn image_of(device: &Device, out: &Path, images: bool) -> Outcome {
    if !images {
        return Outcome::Source;
    }
    let Some(recipe) = recipe(device.profile.meta.emit) else {
        return Outcome::NoRecipe;
    };
    match compile_image(device, &recipe, out) {
        Ok(outcome) => outcome,
        Err(reason) => Outcome::Skipped(reason),
    }
}

/// 编译一台设备的见证：`[tools] <compile> <link> <inspect>`。
///
/// 不传 `[compile].flags` / `preamble`：见证是 **host C ABI**（`int func(float *out)`），与片段
/// 语言无关，`cutile verify` 的本机自检也因为同一个原因跳过它们。`metal` 的 `-std=metal3.1`
/// 会要求指针写明地址空间，而见证里的 `float *` 故意没有——那是 host C 的写法。
fn compile_image(device: &Device, recipe: &Recipe, out: &Path) -> Result<Outcome, String> {
    let Some((program, prefix)) = tool(&device.profile.meta.tool) else {
        return Err(format!("`{}` 不是可执行的工具名", device.profile.meta.tool));
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

/// `CUTILE_REQUIRE_IMAGE` 要求了这台设备就必须真的带上镜像。
///
/// 清单里别的设备由它们自己的 crate 负责——一次构建只跑一个 crate 的 `build.rs`。
fn require(device: &Device, outcome: &Outcome) {
    let required = env::var("CUTILE_REQUIRE_IMAGE")
        .ok()
        .is_some_and(|text| text.split(',').map(str::trim).any(|name| name == device.name));
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

fn header(text: &mut String) {
    let _ = writeln!(
        text,
        "// 由 `crates/prim/generate.rs` 生成，不要手改。\n\
         //\n\
         // 一台设备的片段、入口与设备行；`prim` 把六份这样的表合成 `BUILTIN` / `WITNESS` /\n\
         // `DEVICES`。类型定义在 `src/` 与 `prof`，所以生成代码不改变所有权表。\n\
         #[allow(unused_imports)]\n\
         use prof::atom::Prim;"
    );
}

fn read(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| fail(format!("读取 {} 失败：{error}", path.display())))
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

fn fail(message: impl std::fmt::Display) -> ! {
    panic!("prim/generate.rs: {message}")
}

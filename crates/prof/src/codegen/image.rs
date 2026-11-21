//! 设备镜像：`witness.<ext>` → 中间产物 → 镜像，以及"这次构建到底带上了什么"。
//!
//! 工具链不在本机、或编译失败时只记 [`Outcome::Skipped`]，构建照常成功——源码永远可用。发布
//! 流水线要"缺镜像即失败"时，设 `CUTILE_REQUIRE_IMAGE=<vendor>[,<vendor>]` 即可，它把"这台设备
//! 缺镜像"升级成构建失败。

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::prof::Emit;

use super::scan::{Device, fail};

/// 一台设备的镜像结局。
pub(super) enum Outcome {
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
pub(super) struct Reason(pub(super) String);

/// 按输出语言编设备镜像：`witness.<ext>` → 中间产物 → 镜像。
pub(super) fn outcome(device: &Device, out: &Path, images: bool) -> Outcome {
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
///
/// 不传 `[compile].flags` / `preamble`：见证是 **host C ABI**（`int func(float *out)`），与片段
/// 语言无关，`cutile verify` 的本机自检也因为同一个原因跳过它们。`metal` 的 `-std=metal3.1`
/// 会要求指针写明地址空间，而见证里的 `float *` 故意没有——那是 host C 的写法。
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

/// 这台设备的镜像格式名；没有配方时是 `""`（该分支不会生成镜像）。
pub(super) fn format(emit: Emit) -> &'static str {
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
///
/// 清单里别的设备由它们自己的 crate 负责——一次构建只跑一个 crate 的 `build.rs`。
pub(super) fn require(device: &Device, outcome: &Outcome) {
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

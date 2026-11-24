//! 设备镜像：按输出语言把见证编成二进制镜像（`device-image` feature）。

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use diagnostic::{Code, Error};

use crate::{Device, fail};

// ── 设备镜像 ──────────────────────────────────────────────────────────

/// 一台设备的镜像结局。
pub enum Outcome {
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
pub fn outcome(device: &Device, out: &Path, images: bool) -> Outcome {
    if !images {
        return Outcome::Source;
    }
    match recipe(&device.emit) {
        None => Outcome::NoRecipe,
        Some(recipe) => {
            compile(device, recipe, out).unwrap_or_else(|error| Outcome::Skipped(error.to_string()))
        }
    }
}

/// 编译一台设备的见证：`[meta].tools` 给出的程序 + 配方里的参数。
fn compile(device: &Device, recipe: &Recipe, out: &Path) -> diagnostic::Result<Outcome> {
    let mut parts = device.tool.split_whitespace();
    let Some(program) = parts.next() else {
        return Err(Error::new(
            Code::Invalid,
            format!("`{}` 不是可执行的工具名", device.tool),
        ));
    };
    let object = out.join(format!("{}.witness.{}", device.name, recipe.object_ext));
    let image = out.join(format!("{}.witness.{}", device.name, recipe.format));

    let mut args: Vec<String> = parts.map(str::to_string).collect();
    args.extend(
        recipe
            .compile
            .iter()
            .map(|arg| expand(arg, &device.witness, &object, &image)),
    );
    run(program, &args)?;

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
        .map_or(0, |stdout| {
            stdout.lines().filter(|line| line.contains(" T ")).count()
        });

    Ok(Outcome::Compiled {
        path: image,
        symbols,
    })
}

/// 一条语言级的构建配方：怎么把一份翻译单元变成设备镜像。
pub struct Recipe {
    /// 编译参数（`$src` / `$obj`）。
    pub compile: &'static [&'static str],
    /// 链接一步：程序与参数（`$obj` / `$out`）。
    pub link: Option<(&'static str, &'static [&'static str])>,
    /// 中间产物的扩展名。
    pub object_ext: &'static str,
    /// 镜像格式名。
    pub format: &'static str,
    /// 符号统计：程序与参数（`$obj`），每行一个符号，` T ` 是已定义的文本符号。
    pub inspect: Option<(&'static str, &'static [&'static str])>,
}

/// 有构建配方的输出语言。其余语言只有源码：`Status::NoRecipe` 是它们的诚实答案。
const RECIPES: &[(&str, Recipe)] = &[(
    "metal",
    Recipe {
        compile: &["-c", "$src", "-o", "$obj"],
        link: Some(("xcrun", &["metallib", "$obj", "-o", "$out"])),
        object_ext: "air",
        format: "metallib",
        inspect: Some(("xcrun", &["metal-nm", "$obj"])),
    },
)];

pub fn recipe(emit: &str) -> Option<&'static Recipe> {
    RECIPES
        .iter()
        .find(|(language, _)| *language == emit)
        .map(|(_, recipe)| recipe)
}

/// 占位符替换：`$src` / `$obj` / `$out`。
fn expand(arg: &str, src: &Path, object: &Path, image: &Path) -> String {
    arg.replace("$src", &src.display().to_string())
        .replace("$obj", &object.display().to_string())
        .replace("$out", &image.display().to_string())
}

/// 跑一个子进程；失败时把第一行 stderr 作为原因交回（构建不因此失败）。
fn run(program: &str, args: &[String]) -> diagnostic::Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| Error::new(Code::Unsupported, format!("`{program}` 不可执行：{error}")))?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let first = stderr
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    Err(Error::new(
        Code::Unsupported,
        format!(
            "`{program} {}` 失败（{}）：{}",
            args.join(" "),
            output.status,
            shorten(first)
        ),
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
pub fn require(device: &Device, outcome: &Outcome) {
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

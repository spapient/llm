//! 构建期代码生成：一个设备目录 → 一份编译期表（`$OUT_DIR/tables.rs`）。
//!
//! 六台设备各在自己的 crate 里（`cuda` / `cpu` / `opencl` / `metal` / `musa` / `rocm`），每个 crate
//! 的 `build.rs` 只有一行 [`device`]：扫描、校验、生成与（可选的）镜像编译在仓库里只有这一份。
//!
//! 生成器住在本层，理由只有依赖方向：六个设备 crate 都依赖 `prof`，反之不成立；而"一台设备这次
//! 构建带上了什么"恰好就是 [`crate::Device`] 的定义。于是生成器既不必 `include!` 别处的源码文本，
//! 也不必从路径推断档案在哪——档案就在本层的 [`crate::embedded`] 里。本模块只在 `build` feature
//! 下编译，运行期二进制里没有它。
//!
//! 校验与生成同一趟，数据错在构建期就失败：
//!
//! 1. 文件名必须是 [`Prim`] 闭集里的名字（运行期 `prim::load` 也这么判，但那时已经在用户机器上）；
//! 2. 片段第一行的签名声明 `/* cutile-sign: <prim>/<impl> */` 必须在档案里有对应的 `provides`；
//! 3. 那条供给必须带见证——映射表指向一个入口，指不到就不该生成；
//! 4. 目录的厂商名与扩展名必须等于档案的 `meta.vendor` / `meta.emit.ext()`。
//!
//! `device-image` feature 打开时，再按输出语言把 `witness.<ext>` 编成设备镜像并嵌入。
//!
//! 片段本身不进镜像：片段是语句块，`$0..$3` 的绑定由算子模板的 `@prim(...)` 给出，脱离模板单独
//! 编译只能编出一段与真实调用无关的代码。能编成设备镜像的最小单位是 `witness.<ext>`——一份完整
//! 翻译单元、固定 ABI、每条签名一个入口。
//!
//! 重跑条件里没有档案目录：档案由 `prof` 自己的 `build.rs` 管（它已声明 `rerun-if-changed=devices`），
//! 而本模块是 `prof` 的构建依赖——档案一变 `prof` 就重建，设备 crate 的构建脚本随之重跑。设备目录
//! 则整个算输入：新增一段片段、换一份见证都必须重跑。

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::atom::Prim;
use crate::prof::Profile;

mod image;
mod table;

/// 生成器入口：`CARGO_MANIFEST_DIR` 里的片段 → `$OUT_DIR/tables.rs`。
///
/// 只给 `build.rs` 用：它读 `CARGO_MANIFEST_DIR` / `OUT_DIR` / `CARGO_FEATURE_DEVICE_IMAGE`，并往
/// stdout 写 `cargo:` 指令。`vendor` 必须与 `crates/prof/devices/<vendor>.toml` 同名——它是内嵌表的
/// 键，也是档案里 `witness.file` 的前缀；目录名可以与之不同（`crates/opencl` 装的是 `intel` 的
/// OpenCL 片段），所以显式传入，不从路径推断。
pub fn device(vendor: &str) {
    let dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR：只有 build.rs 会调用这个函数"),
    );
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR：只有 build.rs 会调用这个函数"));
    let images = env::var_os("CARGO_FEATURE_DEVICE_IMAGE").is_some();

    // 整个 crate 目录都算输入：新增一段片段、换一份见证都必须让构建脚本重跑。
    println!("cargo:rerun-if-changed={}", dir.display());
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DEVICE_IMAGE");
    println!("cargo:rerun-if-env-changed=CUTILE_REQUIRE_IMAGE");

    let device = scan(&dir, vendor);
    // 镜像结局只算一次：生成的表与 `CUTILE_REQUIRE_IMAGE` 门禁读的是同一次编译的结果。
    let outcome = image::outcome(&device, &out, images);

    let mut text = String::new();
    table::header(&mut text);
    table::fragments(&mut text, &device);
    table::device_line(&mut text, &device, &outcome);
    fs::write(out.join("tables.rs"), text).expect("写 tables.rs");

    image::require(&device, &outcome);
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
    let ext =
        extension(&witness).unwrap_or_else(|| fail(format!("`{}` 没有扩展名", witness.display())));

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
///
/// 剥壳与切分走 [`crate::sign`] 的共享规则（ADR 0027 §5）；这里只补原语比对与构建失败。
fn signature(prim: Prim, source: &str, path: &Path) -> String {
    let first = source.lines().next().unwrap_or_default();
    let declared = crate::sign::strip_marker(first).unwrap_or_else(|| {
        fail(format!(
            "`{}` 的第一行必须是 `/* {} <prim>/<impl> */`，实际是 `{first}`",
            path.display(),
            crate::sign::MARKER
        ))
    });
    let (declared_prim, implementation) = crate::sign::split_slash(declared)
        .unwrap_or_else(|| fail(format!("`{}` 的签名声明不是 `<prim>/<impl>`", path.display())));
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

fn fail(message: impl std::fmt::Display) -> ! {
    panic!("prof::codegen: {message}")
}

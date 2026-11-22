//! 职责：写回尾块片段与档案 `[[epilogue]]` 的逐形式对齐。
//!
//! 拥有"一份 `<stem>.<ext>` 文本 → 六条见证入口"的核对；不拥有目录扫描、原语片段与编译参数，
//! 那三件事在父模块。依赖 `spec::epilogue` 的闭集语法与 `Profile::epilogue` 的供给声明。

use super::{extension, fail, read, stem};
use crate::atom::Epilogue;
use crate::prof::Profile;
use crate::sign::EPILOGUE_STEM;
use std::path::Path;

/// 一个文件是不是写回尾块片段：是就返回六种形式的见证入口，不是就 `None`。
///
/// 六种形式必须齐全——`spec::epilogue` 的语法是闭集，少一种就说明这台设备的写回根本不完整；
/// 每段以 `/* epi: <form> */` 开头，装配按语法顺序各取一段。
pub(super) fn epilogue_of(
    path: &Path,
    ext: &str,
    profile: &Profile,
) -> Option<Vec<(Epilogue, String)>> {
    if stem(path) != Some(EPILOGUE_STEM) || extension(path).as_deref() != Some(ext) {
        return None;
    }
    let source = read(path);
    family_signature(&source, path);
    let entries = Epilogue::ALL
        .iter()
        .map(|form| (*form, epi_entry_of(profile, *form, path, &source)))
        .collect();
    Some(entries)
}

/// 写回尾块片段的第一行：`/* cutile-sign: epilogue */`。
///
/// 它不带 `/`：原语片段的签名是"这一段实现哪条签名"，而一份尾块文本实现的是六种形式，
/// 逐个形式写签名会让同一份文件出现六条声明。
fn family_signature(source: &str, path: &Path) {
    let first = source.lines().next().unwrap_or_default();
    let declared = crate::sign::strip_marker(first).unwrap_or_else(|| {
        fail(format!(
            "`{}` 的第一行必须是 `/* {} {EPILOGUE_STEM} */`，实际是 `{first}`",
            path.display(),
            crate::sign::MARKER
        ))
    });
    if declared.trim() != EPILOGUE_STEM {
        fail(format!(
            "`{}` 的签名声明是 `{}`：写回尾块只能声明 `{EPILOGUE_STEM}`（不带 `/`）",
            path.display(),
            declared.trim()
        ));
    }
}

/// 写回形式在档案里的见证入口；缺供给、缺段标记或缺见证都是构建期失败。
fn epi_entry_of(profile: &Profile, form: Epilogue, path: &Path, source: &str) -> String {
    let marker = crate::device::epi_marker(form);
    if !source.contains(marker.as_str()) {
        fail(format!(
            "`{}` 缺少 `{marker}` 段：六种写回形式必须齐全",
            path.display()
        ));
    }
    let provide = profile
        .epilogue
        .iter()
        .find(|provide| provide.form == form)
        .unwrap_or_else(|| {
            fail(format!(
                "`{}` 有 `{form}` 段，档案里却没有对应的 `[[epilogue]]`（`form = \"{form}\"`）",
                path.display()
            ))
        });
    provide
        .witness
        .as_ref()
        .map(|witness| witness.func.clone())
        .unwrap_or_else(|| {
            fail(format!(
                "`{}` 的 `{form}` 段没有见证：装配表要指向镜像里的一个入口，先给这条写回写见证",
                path.display()
            ))
        })
}

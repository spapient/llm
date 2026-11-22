//! 职责：认出写回尾块片段，并核对六种形式的段标记齐全。
//!
//! 拥有"一份 `epilogue.<ext>` 文本 → 六段可装配的正文"的核对；不拥有目录扫描、原语片段与
//! 编译参数，那三件事在父模块。闭集由 `spec::epilogue` 的语法与 [`Epilogue::ALL`] 共同钉住。

use super::{extension, fail, read, stem};
use crate::atom::Epilogue;
use crate::sign::EPILOGUE_STEM;
use std::path::{Path, PathBuf};

/// 一个文件是不是写回尾块片段：是就返回路径，不是就 `None`。
///
/// 六种形式必须齐全——`spec::epilogue` 的语法是闭集，少一种就说明这台设备的写回根本不完整。
/// 段标记在**正文**上核对，而不在档案的清单上：装配器取的就是这份正文，它是唯一的事实源，
/// 所以"有没有这一段"必须由正文自己回答。
pub(super) fn epilogue_of(path: &Path, ext: &str) -> Option<PathBuf> {
    if stem(path) != Some(EPILOGUE_STEM) || extension(path).as_deref() != Some(ext) {
        return None;
    }
    let source = read(path);
    family_signature(&source, path);
    every_form(&source, path);
    Some(path.to_path_buf())
}

/// 六段标记齐全，缺一段即构建期失败。
fn every_form(source: &str, path: &Path) {
    for form in Epilogue::ALL {
        let marker = crate::device::epi_marker(*form);
        if !source.contains(marker.as_str()) {
            fail(format!(
                "`{}` 缺少 `{marker}` 段：六种写回形式必须齐全",
                path.display()
            ));
        }
    }
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

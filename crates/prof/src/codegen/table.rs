//! 把扫出来的事实写成 `tables.rs`：片段表（`FRAGMENTS`）与设备行（`PRIMS` + `DEVICE`）。
//!
//! 生成的三项都是 `pub const`，所以不改变所有权表；每个都带 `///`——设备 crate 打开
//! `missing_docs`，生成的公开常量也得有文档。路径用绝对路径写进 `include_str!` / `include_bytes!`，
//! 因为 `tables.rs` 落在 `$OUT_DIR`，相对路径会以那里为基准。

use std::fmt::Write as _;

use super::image::{self, Outcome};
use super::scan::Device;

/// 文件头：说明这是生成物，并把 `Prim` 拉进作用域。
pub(super) fn header(text: &mut String) {
    let _ = writeln!(
        text,
        "// 由 `crates/prof/src/codegen` 生成，不要手改。\n\
         //\n\
         // 一台设备的片段、入口与设备行；`prim` 把六份这样的表合成 `BUILTIN` / `WITNESS` /\n\
         // `DEVICES`。类型定义在 `src/` 与 `prof`，所以生成代码不改变所有权表。\n\
         #[allow(unused_imports)]\n\
         use prof::atom::Prim;"
    );
}

/// 生成片段表：`(原语, 文本)`，顺序与 `Prim::ALL` 一致。
pub(super) fn fragments(text: &mut String, device: &Device) {
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

/// 生成设备行：入口表 + 档案元数据 + 见证源码 + （可选的）镜像。
pub(super) fn device_line(text: &mut String, device: &Device, outcome: &Outcome) {
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

    let (status, image) = match outcome {
        Outcome::Source => ("prof::Status::Source".to_string(), None),
        Outcome::NoRecipe => ("prof::Status::NoRecipe".to_string(), None),
        Outcome::Skipped(reason) => (
            format!("prof::Status::Skipped {{ reason: {reason:?} }}"),
            None,
        ),
        Outcome::Compiled { path, symbols } => (
            "prof::Status::Compiled".to_string(),
            Some((path, *symbols, image::format(device.profile.meta.emit))),
        ),
    };

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
    let _ = writeln!(
        text,
        "    source: include_str!({:?}),",
        device.witness.display().to_string()
    );
    match image {
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

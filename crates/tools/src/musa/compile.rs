//! 编译：把设备源交给 `mcc`，车道决定开关。
//!
//! 命令行只有一种形状（`mcc [开关] 源 -o 产物 -I 垫片 -L 库 [链接]`），车道只贡献两件事：
//! 编译开关与链接库。工具链根目录由调用方显式给出——库不认识环境变量，「用哪套 mcc」
//! 是入口的决定。

use std::path::{Path, PathBuf};
use std::process::Command;

use diagnostic::{Code, Error, Result};

/// 两个转发头：(文件名, 正文)。
///
/// 生成的程序只写 `cuda*` 名字；这两个文件把它们映射到 `musa*`。实测**只需这两个**——
/// 这条通道能跨工具链的原因就在这里，不在任何 crate 结构里。
const SHIM: [(&str, &str); 2] = [
    (
        "cuda_runtime.h",
        include_str!("../../assets/musa/shim/cuda_runtime.h"),
    ),
    (
        "cuda_fp16.h",
        include_str!("../../assets/musa/shim/cuda_fp16.h"),
    ),
];

/// 编译车道：决定开关与链接库。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lane {
    /// 普通 CUDA 语法的设备源（`-x musa` + `-lmusart`）。
    Plain,
    /// 张量核特化：额外要 mp_22 与 tf32 的完整头。
    Tensor,
    /// muBLAS 对照：额外链 `-lmublas`。
    Mublas,
}

impl Lane {
    /// 名字 → 车道；未知名字报 `Unsupported`（不静默降级成 `Plain`，那会把张量核代码编成
    /// mp_21 再报一个与原因无关的错）。
    fn of(name: &str) -> Result<Self> {
        match name {
            "plain" => Ok(Self::Plain),
            "wmma" | "tensor" => Ok(Self::Tensor),
            "mublas" => Ok(Self::Mublas),
            other => Err(Error::new(
                Code::Unsupported,
                format!("未知编译车道 `{other}`（可选：plain / wmma / mublas）"),
            )),
        }
    }
}

/// 一次编译：工具链根目录 + 车道。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    root: PathBuf,
    lane: Lane,
}

impl Recipe {
    /// 显式构造：`root` 是工具链根目录（如 `/usr/local/musa`，其下有 `bin/mcc`、`lib/`）。
    pub fn at(root: &Path, lane: &str) -> Result<Self> {
        Ok(Self {
            root: root.to_path_buf(),
            lane: Lane::of(lane)?,
        })
    }

    /// 命令行（不含程序名 `mcc` 本身），顺序按 `mcc` 的惯例：开关、源、产物、头、库、链接。
    pub fn argv(&self, source: &Path, binary: &Path, shim: &Path) -> Vec<String> {
        let mut args = vec!["-O3".to_string(), "-x".to_string(), "musa".to_string()];
        args.extend(self.flags());
        args.push(source.display().to_string());
        args.push("-o".to_string());
        args.push(binary.display().to_string());
        args.extend(self.headers(shim));
        args.push("-L".to_string());
        args.push(self.root.join("lib").display().to_string());
        args.push("-lmusart".to_string());
        args.extend(self.links());
        args
    }

    /// 编译；失败时把完整命令行带进错误（`mcc` 的诊断在 stderr，用户要能自己重跑一遍）。
    pub fn compile(&self, source: &Path, binary: &Path, shim: &Path) -> Result<()> {
        let argv = self.argv(source, binary, shim);
        let mcc = self.root.join("bin/mcc");
        let status = Command::new(&mcc)
            .args(&argv)
            .status()
            .map_err(|err| io(&mcc.display().to_string(), err))?;
        if status.success() {
            return Ok(());
        }
        Err(Error::new(
            Code::Unsupported,
            format!("mcc 退出 {status}：{} {}", mcc.display(), argv.join(" ")),
        ))
    }

    /// 车道开关。
    fn flags(&self) -> Vec<String> {
        match self.lane {
            Lane::Tensor => ["-D__MUSA_QY2_MMA__=1", "--cuda-gpu-arch=mp_22"]
                .map(str::to_string)
                .to_vec(),
            Lane::Plain | Lane::Mublas => Vec::new(),
        }
    }

    /// 头文件目录：垫片永远要；张量核还要工具链的完整头（`$ROOT/include` 的 `mublas.h`
    /// 是残缺包装，`precision::tf32` 只在 `musa-5.2.0/include` 里声明）。
    fn headers(&self, shim: &Path) -> Vec<String> {
        let mut args = vec!["-I".to_string(), shim.display().to_string()];
        if self.lane == Lane::Tensor {
            args.push("-I".to_string());
            args.push(self.complete_headers().display().to_string());
        }
        args
    }

    /// 完整头的目录：优先 `<根目录的父目录>/musa-5.2.0/include`，不存在时退回 `<根目录>/include`。
    fn complete_headers(&self) -> PathBuf {
        let versioned = self.root.parent().map(|parent| parent.join("musa-5.2.0/include"));
        versioned
            .filter(|dir| dir.is_dir())
            .unwrap_or_else(|| self.root.join("include"))
    }

    /// 车道额外链接的库。
    fn links(&self) -> Vec<String> {
        match self.lane {
            Lane::Mublas => vec!["-lmublas".to_string()],
            Lane::Plain | Lane::Tensor => Vec::new(),
        }
    }
}

/// 写两个转发头到 `dir`（幂等，可反复调用）。编译前必须先写。
pub fn shim(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|err| io(&dir.display().to_string(), err))?;
    for (name, body) in SHIM {
        let path = dir.join(name);
        std::fs::write(&path, body).map_err(|err| io(&path.display().to_string(), err))?;
    }
    Ok(())
}

/// IO 失败：带上出错的路径。
fn io(what: &str, err: std::io::Error) -> Error {
    Error::new(Code::Io, format!("{what}：{err}"))
}

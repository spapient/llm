//! 内置声明与档案：随 crate 编译，不依赖工作目录。
//!
//! 磁盘上的 `ops/`、`devices/`、`prim/` 才是事实源；这里是它们的编译期内嵌副本，
//! 让库调用方与测试无需设置路径即可跑通整条流水线。

use crate::atom::Prim;
use crate::lower::expand::{OpDecl, OpError};
use crate::probe::select;
use crate::prof::Detected;
use crate::prof::{Profile, ProfileError};
use std::path::{Path, PathBuf};

/// 内置算子：`(名字, 声明, 模板)`。
pub const OPS: [(&str, &str, &str); 1] = [(
    "gemv_i4",
    include_str!("../ops/gemv_i4.toml"),
    include_str!("../ops/gemv_i4.k"),
)];

/// 内置档案：`(vendor, 文本)`。
pub const PROFILES: [(&str, &str); 3] = [
    ("cpu", include_str!("../devices/cpu.toml")),
    ("nvidia", include_str!("../devices/nvidia.toml")),
    ("intel", include_str!("../devices/intel.toml")),
];

/// 声明 + 档案的集合。
#[derive(Debug, Clone, PartialEq)]
pub struct Catalogue {
    ops: Vec<OpDecl>,
    profiles: Vec<Profile>,
}

impl Catalogue {
    /// 内置集合。
    ///
    /// # Panics
    ///
    /// 内置数据解析失败即 crate 缺陷；由 `cargo test` 的 `builtin_data_is_valid` 兜住。
    pub fn builtin() -> Self {
        Self::try_builtin().expect("内置声明与档案必须可解析")
    }

    /// 内置集合（不 panic 版本）。
    pub fn try_builtin() -> Result<Self, String> {
        let mut ops = Vec::new();
        for (name, declaration, template) in OPS {
            let op = OpDecl::parse(declaration, template)
                .map_err(|error| format!("内置算子 `{name}`：{error}"))?;
            ops.push(op);
        }
        let mut profiles = Vec::new();
        for (name, text) in PROFILES {
            let profile =
                Profile::parse(text).map_err(|error| format!("内置档案 `{name}`：{error}"))?;
            profiles.push(profile);
        }
        Ok(Self { ops, profiles })
    }

    /// 从仓库目录读取（`ops/*.toml` + 同名 `.k`，`devices/*.toml`）。
    pub fn from_dir(root: &Path) -> Result<Self, String> {
        let mut ops = Vec::new();
        for entry in read_dir(&root.join("ops"))? {
            let path = entry;
            if path.extension().and_then(|value| value.to_str()) != Some("toml") {
                continue;
            }
            let declaration = read(&path)?;
            let template_path = path.with_extension("k");
            if !template_path.is_file() {
                return Err(format!(
                    "`{}` 缺少同名模板 `{}`",
                    path.display(),
                    template_path.display()
                ));
            }
            let template = read(&template_path)?;
            ops.push(
                OpDecl::parse(&declaration, &template)
                    .map_err(|error| format!("`{}`：{error}", path.display()))?,
            );
        }
        let mut profiles = Vec::new();
        for path in read_dir(&root.join("devices"))? {
            if path.extension().and_then(|value| value.to_str()) != Some("toml") {
                continue;
            }
            profiles.push(
                Profile::parse(&read(&path)?)
                    .map_err(|error| format!("`{}`：{error}", path.display()))?,
            );
        }
        Ok(Self { ops, profiles })
    }

    /// 取算子。
    pub fn op(&self, name: &str) -> Option<&OpDecl> {
        self.ops.iter().find(|op| op.name == name)
    }

    /// 取档案：先按 vendor，再按 id。
    pub fn profile(&self, name: &str) -> Option<&Profile> {
        self.profiles
            .iter()
            .find(|profile| profile.meta.vendor == name || profile.meta.id == name)
    }

    /// 全部算子。
    pub fn ops(&self) -> &[OpDecl] {
        &self.ops
    }

    /// 全部档案。
    pub fn profiles(&self) -> &[Profile] {
        &self.profiles
    }

    /// 按探测到的指纹挑档案：命中的规则条数最多者胜出；同等具体多个即报错。
    pub fn select(&self, detected: &Detected) -> Result<&Profile, String> {
        select(&self.profiles, detected)
    }

    /// 全量校验：重复标识、重复原语、见证文件存在性。
    pub fn validate(&self, root: Option<&Path>) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        for (index, profile) in self.profiles.iter().enumerate() {
            for other in &self.profiles[index + 1..] {
                if profile.meta.vendor == other.meta.vendor && profile.meta.id == other.meta.id {
                    errors.push(format!(
                        "`{}`/`{}` 有两份档案，无法决定用哪一份",
                        profile.meta.vendor, profile.meta.id
                    ));
                }
            }
            let mut seen: Vec<Prim> = Vec::new();
            for provide in &profile.provides {
                if seen.contains(&provide.prim) {
                    errors.push(format!(
                        "`{}`/`{}` 的 `{}` 声明了两条，协商无法取舍",
                        profile.meta.vendor, profile.meta.id, provide.prim
                    ));
                }
                seen.push(provide.prim);
            }
            if let Err(profile_errors) = profile.validate(root) {
                for error in profile_errors {
                    errors.push(format!("`{}`：{error}", profile.meta.vendor));
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// 仓库根目录（本 crate 的 `CARGO_MANIFEST_DIR`）。
    pub fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }
}

fn read_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries =
        std::fs::read_dir(dir).map_err(|error| format!("读取 {} 失败：{error}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .collect();
    paths.sort();
    Ok(paths)
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|error| format!("读取 {} 失败：{error}", path.display()))
}

/// 内置档案里的声明层错误（供 CLI 打印）。
pub fn op_error(error: &OpError) -> String {
    error.to_string()
}

/// 内置档案里的档案层错误（供 CLI 打印）。
pub fn profile_error(error: &ProfileError) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe;

    #[test]
    fn builtin_data_is_valid() {
        let catalogue = Catalogue::try_builtin().expect("内置数据必须合法");
        assert_eq!(catalogue.ops().len(), OPS.len());
        assert_eq!(catalogue.profiles().len(), PROFILES.len());
    }

    #[test]
    fn builtin_witness_files_exist() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        catalogue
            .validate(Some(&Catalogue::root()))
            .expect("见证文件必须存在");
    }

    #[test]
    fn on_disk_data_matches_embedded_data() {
        let catalogue = Catalogue::from_dir(&Catalogue::root()).expect("磁盘数据可解析");
        assert_eq!(
            catalogue.op("gemv_i4").map(|op| op.name.as_str()),
            Some("gemv_i4")
        );
        assert!(catalogue.profile("cpu").is_some());
        catalogue
            .validate(Some(&Catalogue::root()))
            .expect("磁盘数据通过校验");
    }

    #[test]
    fn duplicate_profiles_are_rejected() {
        let mut catalogue = Catalogue::try_builtin().expect("内置数据");
        let clone = catalogue.profiles[0].clone();
        catalogue.profiles.push(clone);
        let errors = catalogue.validate(Some(&Catalogue::root())).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("两份档案")));
    }

    #[test]
    fn detects_the_local_cpu_profile() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let detected = probe::detect_cpu();
        let profile = catalogue.select(&detected).expect("必须命中 CPU 档案");
        assert_eq!(profile.meta.vendor, "cpu");
    }
}

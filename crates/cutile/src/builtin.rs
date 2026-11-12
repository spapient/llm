//! 内置声明与档案：随 crate 编译，不依赖工作目录。
//!
//! 三层数据各自住在自己的 crate：声明层（`ops/`）由 [`op`]、档案层（`devices/`）由 [`prof`]、
//! 片段层（`prim/`）由 [`prim`]。本模块只负责把前两层解析成 [`OpDecl`] 与 [`Profile`]，并把
//! 它们的交叉约束（重复档案、同一原语的同名实现、见证文件存在性）合成一次校验；内嵌副本让库调用方与
//! 测试无需设置路径即可跑通整条流水线。

use crate::cap;
use crate::decl::OpDecl;
use crate::probe::select;
use crate::prof::Detected;
use crate::prof::Profile;
use diagnostic::{Code, Error};
use op::Op;
use std::path::Path;

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
    pub fn try_builtin() -> diagnostic::Result<Self> {
        let mut profiles = Vec::new();
        for archive in prof::builtin() {
            let profile = Profile::parse(archive.text()).map_err(|error| {
                Error::new(
                    Code::Invalid,
                    format!("内置档案 `{}`：{error}", archive.name()),
                )
            })?;
            profiles.push(profile);
        }
        Self::assemble(op::builtin(), profiles)
    }

    /// 从数据根读取：`<root>/ops/*.toml` + 同名 `.k`，`<root>/devices/*.toml`。
    ///
    /// 数据根是"不改代码替换数据"的入口：三层数据在仓库里分属 `op`、`prof` 与本 crate，
    /// 但一份 `data.root` 可以把它们整体替换掉。装载规则分别由 [`op::load`] 与 [`prof::load`]
    /// 定义，本 crate 只解析内容。
    pub fn from_dir(root: &Path) -> diagnostic::Result<Self> {
        Self::from_layers(&root.join("ops"), &root.join("devices"))
    }

    /// 仓库内的磁盘副本：声明层在 `crates/op/ops`，档案层在 `crates/prof/devices`。
    ///
    /// 只用于对拍"磁盘事实源"与"编译期内嵌副本"：它依赖构建机的源码树，发布路径不得调用。
    #[cfg(test)]
    pub fn from_repo() -> diagnostic::Result<Self> {
        Self::from_layers(&op::dir(), &prof::dir())
    }

    /// 按配置装载：设了 `data.root` 就读磁盘的三层数据，否则用编译期内嵌副本。
    ///
    /// 内置数据是编译期内嵌的：发布机上没有源码树，因此只校验结构，不校验见证文件路径。
    /// 见证源码住在片段层，所以校验根是 `<data.root>/prim`，而不是数据根本身。
    ///
    /// # Errors
    ///
    /// 目录装载失败或交叉校验不通过时返回逐条说明。
    pub fn load(settings: &llm_config::Settings) -> diagnostic::Result<Self> {
        let (witness_root, catalogue) = match settings.data_root() {
            Some(root) => (Some(prim::layer_root(root)), Self::from_dir(root)?),
            None => (None, Self::try_builtin()?),
        };
        catalogue
            .validate(witness_root.as_deref())
            .map_err(|mut errors| {
                // 首条决定类别，其余作为同级诊断逐行保留各自的类别：一次报全，但不把
                // `missing`（文件不在）与 `invalid`（内容写错）压成同一个码。
                let mut errors = errors.drain(..);
                let first = errors.next().expect("validate 失败即至少一条");
                errors.fold(first, |error, other| error.also(&other))
            })
            .map_err(|error| error.with("cutile::builtin::load"))?;
        Ok(catalogue)
    }

    /// 按名字或指纹取档案；`auto` 走设备探测。
    ///
    /// # Errors
    ///
    /// 名字不存在或探测不出可用档案时返回说明。
    pub fn resolve(&self, device: &str) -> diagnostic::Result<&Profile> {
        if device == "auto" {
            return self.select(&crate::probe::detect_cpu());
        }
        self.profile(device)
            .ok_or_else(|| Error::new(Code::Missing, format!("没有设备档案 `{device}`")))
    }

    fn from_layers(ops_dir: &Path, devices_dir: &Path) -> diagnostic::Result<Self> {
        let mut profiles = Vec::new();
        // 档案层与声明层现在都用自己的类别报错：边界处只补"是哪一层在读"，不重新给类别，
        // 否则 `Missing`（文件不在）与 `Invalid`（内容写错）会被压成同一个码。
        for archive in prof::load(devices_dir)? {
            profiles.push(Profile::parse(archive.text()).map_err(|error| {
                Error::new(Code::Invalid, format!("`{}`：{error}", archive.name()))
            })?);
        }
        Self::assemble(op::load(ops_dir)?, profiles)
    }

    /// 解析声明层并按名字对齐：文件名即算子名，与 `[op] name` 不一致直接报错。
    fn assemble(ops: Vec<Op>, profiles: Vec<Profile>) -> diagnostic::Result<Self> {
        let mut parsed = Vec::new();
        for op in &ops {
            let declaration = OpDecl::parse(op.declaration(), op.template())
                .map_err(|error| error.with(format!("`{}`", op.name())))?;
            if declaration.name != op.name() {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "`{}` 的 `[op] name` 是 `{}`：文件名即算子名，两者必须一致",
                        op.name(),
                        declaration.name
                    ),
                ));
            }
            parsed.push(declaration);
        }
        Ok(Self {
            ops: parsed,
            profiles,
        })
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
    pub fn select(&self, detected: &Detected) -> diagnostic::Result<&Profile> {
        select(&self.profiles, detected)
    }

    /// 全量校验：重复标识、同一原语的同名实现、见证文件存在性。
    ///
    /// 返回逐条错误而不是第一个：装载要一次报全，否则改一条重跑一次。
    pub fn validate(&self, root: Option<&Path>) -> Result<(), Vec<Error>> {
        let mut errors = Vec::new();
        for (index, profile) in self.profiles.iter().enumerate() {
            for other in &self.profiles[index + 1..] {
                if profile.meta.vendor == other.meta.vendor && profile.meta.id == other.meta.id {
                    errors.push(Error::new(
                        Code::Invalid,
                        format!(
                            "`{}`/`{}` 有两份档案，无法决定用哪一份",
                            profile.meta.vendor, profile.meta.id
                        ),
                    ));
                }
            }
            if let Err(profile_errors) = profile.validate(root) {
                for error in profile_errors {
                    // 类别由档案层给（`ProfileError::code`），这里只补"哪份档案"。
                    errors.push(Error::from(error).with(format!("`{}`", profile.meta.vendor)));
                }
            }
        }
        // `provides ↔ 片段文本`：协商会挑中的那条实现，片段必须真的写了。
        //
        // 缺这一段时，一份"协商可行"的档案配一段"存在但写的是另一条签名"的文本能通过全部
        // 校验，直到构建期才炸——甚至把 `dp4a` 的文本当 `fma` 用。判定与 `cap::negotiate`
        // 同源，所以"能协商"与"能生成"不会各算一份。
        for profile in &self.profiles {
            let set = match fragments_for(profile, root) {
                Ok(set) => set,
                Err(error) => {
                    errors.push(error.with(format!("`{}`", profile.meta.vendor)));
                    continue;
                }
            };
            for op in &self.ops {
                for gap in cap::fragment_gaps(op, profile, |prim| set.implementation(prim)) {
                    // 文本在、但写的是另一条签名是"数据不一致"；文本根本不在是"缺件"。
                    let code = if gap.declared.is_none() {
                        Code::Missing
                    } else {
                        Code::Invalid
                    };
                    errors.push(
                        Error::new(code, gap.to_string())
                            .with(format!("`{}`/`{}`", profile.meta.vendor, op.name)),
                    );
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// 校验用的片段层：与 `build::fragments` 的解析次序一致——内嵌优先，指定了片段层根则读磁盘。
///
/// 两条路径必须同源，否则"装载期校验通过"与"构建期取到的文本"可以是两份不同的数据。
fn fragments_for(profile: &Profile, root: Option<&Path>) -> diagnostic::Result<prim::FragmentSet> {
    let ext = profile.meta.emit.ext();
    match root {
        Some(root) => prim::load(root, &profile.meta.vendor, ext),
        None => prim::builtin(&profile.meta.vendor, ext).ok_or_else(|| {
            Error::new(
                Code::Missing,
                format!(
                    "内置片段层里没有 `{}`（`.{ext}`）：档案登记了这台设备，片段层却没有对应条目",
                    profile.meta.vendor
                ),
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decl::expand;
    use crate::probe;
    use crate::substitute;

    #[test]
    fn builtin_data_is_valid() {
        let catalogue = Catalogue::try_builtin().expect("内置数据必须合法");
        assert_eq!(catalogue.ops().len(), op::BUILTIN.len());
        assert_eq!(catalogue.profiles().len(), prof::BUILTIN.len());
    }

    #[test]
    fn builtin_witness_files_exist() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        catalogue
            .validate(Some(&prim::dir()))
            .expect("见证文件必须存在");
    }

    /// 每台设备的每条 `provides` 都要有见证声明——包括能在本机产出 L2 证据的 `cpu`。
    ///
    /// 等级不是门槛：`L0`/`L1` 的声明同样要给出"证据是什么"，否则补见证就没有完成。
    #[test]
    fn every_provide_declares_a_witness() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        for profile in catalogue.profiles() {
            let vendor = profile.meta.vendor.as_str();
            for provide in &profile.provides {
                assert!(
                    provide.witness.is_some(),
                    "`{vendor}`/`{}` 缺少见证声明",
                    provide.prim
                );
            }
        }
    }

    /// 见证矩阵：闭集的 16 个原语在**每台设备**上都要有带见证的签名。
    ///
    /// 只守"P0 用到的 9 个"会留下自相矛盾的档案：写了几十 TFLOPS 的 FMA 口径，却没有一条
    /// `mma`——设备声称的算力与它声明的能力对不上。空白格只在数据能解释时才合法：显式写了
    /// `[simt].group = 1` 的单 lane 设备没有子组内交换的对象，`shuffle`/`broadcast` 因此不声明
    /// （`prof` 装载期会直接拒绝这种声明）。区分"单 lane"与"没有 `[simt]` 段"要看
    /// `simt.declared`：昇腾的 lane 交换由向量单元（`vgather`/`vbroadcast`）承担，本来就不是
    /// SIMT，默认值 1 不是它单 lane 的证据。其余补上的条目等级都是 `L0`：有见证源码，没有
    /// 片段文本，协商照样把它们当作不具备——矩阵补的是覆盖，不是等级。
    #[test]
    fn every_device_covers_the_primitive_set_or_explains_the_gap() {
        use crate::atom::Prim;
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        for profile in catalogue.profiles() {
            let vendor = profile.meta.vendor.as_str();
            for prim in Prim::ALL {
                if profile.simt.declared
                    && profile.simt.group <= 1
                    && matches!(prim, Prim::Shuffle | Prim::Broadcast)
                {
                    assert!(
                        profile.providers(prim).next().is_none(),
                        "`{vendor}` 声明了 `[simt].group = 1`，却又有 `{prim}`"
                    );
                    continue;
                }
                let mut provides = profile.providers(prim).peekable();
                assert!(
                    provides.peek().is_some(),
                    "`{vendor}` 没有声明 `{prim}`：闭集的空白格等于没有证据"
                );
                assert!(
                    provides.all(|provide| provide.witness.is_some()),
                    "`{vendor}` 的 `{prim}` 有一条签名没有见证"
                );
            }
        }
    }

    /// 片段层与模板的调用签名对齐：每条 `@原语(...)` 都能被该设备的片段拼出来。
    ///
    /// 三层签名到此闭合——`needs`↔`provides` 由 `cap` 的测试守着，`provides`↔片段文本由这条
    /// 守着：参数个数或命名参数对不上时，`substitute` 要到构建期才报错，那时离写错的地方
    /// 已经很远。加速器片段同样参与：它们的等级仍是 L0/L1，但"怎么写"必须先是可拼接的文本。
    #[test]
    fn every_device_fragment_set_splices_the_p0_template() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let op = catalogue.op("gemv_i4").expect("内置算子");
        let combos = expand(op).expect("tile 候选");
        for profile in catalogue.profiles() {
            let vendor = profile.meta.vendor.as_str();
            let ext = profile.meta.emit.ext();
            let fragments = prim::builtin(vendor, ext)
                .unwrap_or_else(|| panic!("`{vendor}` 的片段层没有按 `. {ext}` 登记"));
            for combo in &combos {
                if let Err(error) = substitute::render(op, combo, &fragments) {
                    panic!("`{vendor}` 的片段与模板签名不同构（{combo}）：{error}");
                }
            }
        }
    }

    /// 片段层声明的实现，必须正是协商会为这台设备挑中的那条。
    ///
    /// 前一条测试只保证"文本能被拼进来"；这条保证"拼进来的是对的签名"。`dot` 在四台设备上
    /// 都有两条签名（`dp4a` / 标量），挑错一条不会让拼接失败，只会让生成的程序算别的算法。
    #[test]
    fn every_fragment_declares_the_implementation_negotiation_picks() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        for profile in catalogue.profiles() {
            let vendor = profile.meta.vendor.as_str();
            let set = prim::builtin(vendor, profile.meta.emit.ext())
                .unwrap_or_else(|| panic!("`{vendor}` 的片段层没有登记"));
            for op in catalogue.ops() {
                let gaps = cap::fragment_gaps(op, profile, |prim| set.implementation(prim));
                assert!(gaps.is_empty(), "`{vendor}`/`{}`：{gaps:?}", op.name);
            }
        }
    }

    /// 声明错一条签名就是装载期错误：不必等到构建期，更不能静默用错算法。
    #[test]
    fn a_fragment_declaring_another_implementation_fails_the_load() {
        let root = std::env::temp_dir().join(format!("cutile-sign-{}", std::process::id()));
        let prim_root = root.join("prim");
        copy_dir(&prim::dir(), &prim_root);
        // `nvidia` 的 `dot` 协商会挑中 `fma`；把文本改成声明 `dp4a`（另一条签名，属 P3）。
        let path = prim_root.join("nvidia").join("dot.cu");
        let text = std::fs::read_to_string(&path).expect("片段可读");
        assert!(text.contains("dot/fma"), "{text}");
        std::fs::write(&path, text.replacen("dot/fma", "dot/dp4a", 1)).expect("写片段");

        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let errors = catalogue
            .validate(Some(&prim_root))
            .expect_err("片段声明另一条签名必须报错");
        let rendered: Vec<String> = errors.iter().map(ToString::to_string).collect();
        let rendered = rendered.join("\n");
        assert!(rendered.contains("`nvidia`/`gemv_i4`"), "{rendered}");
        assert!(rendered.contains("dp4a"), "{rendered}");
        assert!(rendered.contains("fma"), "{rendered}");
        std::fs::remove_dir_all(&root).ok();
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).expect("建目录");
        for entry in std::fs::read_dir(from).expect("读目录") {
            let entry = entry.expect("目录项");
            let target = to.join(entry.file_name());
            if entry.path().is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).expect("复制文件");
            }
        }
    }

    #[test]
    fn on_disk_data_matches_embedded_data() {
        let catalogue = Catalogue::from_repo().expect("磁盘数据可解析");
        assert_eq!(
            catalogue.op("gemv_i4").map(|op| op.name.as_str()),
            Some("gemv_i4")
        );
        assert!(catalogue.profile("cpu").is_some());
        assert_eq!(
            catalogue.ops().len(),
            Catalogue::try_builtin().expect("内置数据").ops().len(),
            "`crates/op/ops/` 与内嵌声明数量不一致"
        );
        catalogue
            .validate(Some(&prim::dir()))
            .expect("磁盘数据通过校验");
    }

    #[test]
    fn a_declaration_name_must_match_its_file_name() {
        let op = Op::new("renamed", op::BUILTIN[0].1, op::BUILTIN[0].2);
        let profiles = Catalogue::try_builtin().expect("内置数据").profiles;
        let error = Catalogue::assemble(vec![op], profiles).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("文件名即算子名"), "{error}");
    }

    /// 空片段层根下每条失败都是 `missing`：类别由数据层给，装载层不再一律压成 `invalid`。
    #[test]
    fn validation_keeps_the_category_of_the_failing_layer() {
        let root = std::env::temp_dir().join(format!("cutile-prim-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("建临时目录");
        let errors = Catalogue::try_builtin()
            .expect("内置数据")
            .validate(Some(&root))
            .unwrap_err();
        assert!(!errors.is_empty(), "空片段层根必须报错");
        assert!(
            errors.iter().any(
                |error| error.code() == Code::Missing && error.to_string().contains("见证文件")
            ),
            "见证文件不在应是 `missing`：{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.code() == Code::Missing && error.to_string().contains("读取")),
            "片段层目录不在应是 `missing`，不是 `invalid`：{errors:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn duplicate_profiles_are_rejected() {
        let mut catalogue = Catalogue::try_builtin().expect("内置数据");
        let clone = catalogue.profiles[0].clone();
        catalogue.profiles.push(clone);
        let errors = catalogue.validate(Some(&prim::dir())).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.to_string().contains("两份档案"))
        );
    }

    #[test]
    fn detects_the_local_cpu_profile() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let detected = probe::detect_cpu();
        let profile = catalogue.select(&detected).expect("必须命中 CPU 档案");
        assert_eq!(profile.meta.vendor, "cpu");
    }
}

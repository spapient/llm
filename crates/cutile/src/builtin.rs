//! 职责：内置声明与档案：随 crate 编译，不依赖工作目录。

use crate::cap;
use crate::decl::OpDecl;
use crate::probe::select;
use crate::prof::Detected;
use crate::prof::Profile;
use ::prof::catalogue::Op;
use diagnostic::{Code, Error};
use std::sync::{Arc, OnceLock};

/// 保留的设备名：不走档案查找，按本机探测挑一份档案。命令行默认值在 `cli` 侧另存一份，这里是库内比对口径。
pub const AUTO_DEVICE: &str = "auto";

/// 内置集合解析结果的进程内缓存：声明表与档案表都是编译期内嵌的只读数据，
/// 重复调用不该把每条 `OpDecl` / `Profile` 再解析一遍。类型全是自有数据，可跨线程共享。
static CATALOGUE: OnceLock<diagnostic::Result<Catalogue>> = OnceLock::new();

/// 声明 + 档案的集合。
#[derive(Debug, Clone, PartialEq)]
pub struct Catalogue {
    ops: Arc<Vec<OpDecl>>,
    profiles: Arc<Vec<Profile>>,
}

impl Catalogue {
    /// 内置集合。失败原样返回：本 crate 的生产代码没有 panic 边界，`builtin_data_is_valid` 才是"内置数据必须能解析"的执行者。
    ///
    /// 两份表都存在 [`Arc`] 后面，重复调用只做一次引用计数递增，不再逐条深拷贝整张声明表。
    pub fn try_builtin() -> diagnostic::Result<Self> {
        CATALOGUE
            .get_or_init(|| Self::assemble(::prof::catalogue::all(), Self::builtin_profiles()?))
            .clone()
    }

    fn builtin_profiles() -> diagnostic::Result<Vec<Profile>> {
        ::prof::prof::builtin_profiles()
            .map(<[Profile]>::to_vec)
            .map_err(|error| Error::new(Code::Invalid, format!("内置档案：{error}")))
    }

    /// 按配置装载：两层语料都是编译期内嵌的生成表，`data.root` 只决定片段与见证的校验根
    /// （`<data.root>/prim`）。磁盘上的文件只是构建输入：运行期不读目录、也没有 TOML 解析器（ADR 0022）。
    ///
    /// # Errors
    ///
    /// 内置表缺项或交叉校验不通过时返回逐条说明。
    pub fn load(dirs: &crate::dirs::Dirs) -> diagnostic::Result<Self> {
        let layout = dirs.prim();
        let catalogue = Self::try_builtin()?;
        catalogue
            .validate(&layout)
            .map_err(merge)
            .map_err(|error| error.with("cutile::builtin::load"))?;
        Ok(catalogue)
    }

    /// 按名字或指纹取档案；`auto` 走设备探测。
    ///
    /// # Errors
    ///
    /// 名字不存在或探测不出可用档案时返回说明。
    pub fn resolve(&self, device: &str) -> diagnostic::Result<&Profile> {
        if device == AUTO_DEVICE {
            return crate::probe::select_detected(self.profiles(), &crate::probe::detect());
        }
        self.profile(device)
            .ok_or_else(|| Error::new(Code::Missing, format!("没有设备档案 `{device}`")))
    }

    /// 解析声明层并按名字对齐：文件名即算子名，与 `[op] name` 不一致直接报错。
    fn assemble(ops: Vec<Op>, profiles: Vec<Profile>) -> diagnostic::Result<Self> {
        let mut parsed = Vec::new();
        for op in &ops {
            let declaration = op
                .to_decl()
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
            ops: Arc::new(parsed),
            profiles: Arc::new(profiles),
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
    pub fn validate(&self, layout: &prof::Layout<'_>) -> Result<(), Vec<Error>> {
        let mut errors = self.duplicates();
        errors.extend(self.profile_errors(layout));
        errors.extend(self.fragment_errors(layout));
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// 重复标识：`vendor`/`id` 都对上的两份档案谁都不该赢。
    fn duplicates(&self) -> Vec<Error> {
        let mut errors = Vec::new();
        for (idx, profile) in self.profiles.iter().enumerate() {
            for other in &self.profiles[idx + 1..] {
                if same_profile(profile, other) {
                    errors.push(Error::new(
                        Code::Invalid,
                        format!(
                            "`{}`/`{}` 有两份档案，无法决定用哪一份",
                            profile.meta.vendor, profile.meta.id
                        ),
                    ));
                }
            }
        }
        errors
    }

    /// 档案自身的问题：类别由档案层给（`ProfileError::code`），这里只补"哪份档案"。
    fn profile_errors(&self, layout: &prof::Layout<'_>) -> Vec<Error> {
        let mut errors = Vec::new();
        for profile in self.profiles.iter() {
            let witnesses = layout as &dyn prof::prof::WitnessLookup;
            if let Err(profile_errors) = profile.validate(Some(witnesses)) {
                for error in profile_errors {
                    errors.push(Error::from(error).with(format!("`{}`", profile.meta.vendor)));
                }
            }
        }
        errors
    }

    /// `provides ↔ 片段文本`：协商会挑中的那条实现，片段必须真的写了。否则一份"协商可行"的档案配一段
    /// 写错签名的文本能通过全部校验，直到构建期才炸。判定与 `cap::negotiate` 同源。
    fn fragment_errors(&self, layout: &prof::Layout<'_>) -> Vec<Error> {
        let mut errors = Vec::new();
        for profile in self.profiles.iter() {
            let ext = profile.meta.emit.ext();
            let set = match layout.fragments(&profile.meta.vendor, ext) {
                Ok(set) => set,
                Err(error) => {
                    errors.push(error.with(format!("`{}`", profile.meta.vendor)));
                    continue;
                }
            };
            for op in self.ops.iter() {
                errors.extend(gaps_of(op, profile, &set));
            }
        }
        errors
    }
}

/// 同一份档案的标识：`vendor` 与 `id` 都对上才算重复。
fn same_profile(profile: &Profile, other: &Profile) -> bool {
    profile.meta.vendor == other.meta.vendor && profile.meta.id == other.meta.id
}

/// 一条声明在一份档案上的片段缺口，附上定位用的"档案/算子"帧。
fn gaps_of(op: &OpDecl, profile: &Profile, set: &prof::FragmentSet) -> Vec<Error> {
    let site = format!("`{}`/`{}`", profile.meta.vendor, op.name);
    cap::fragment_gaps(op, profile, |prim| set.implementation(prim))
        .into_iter()
        .map(|gap| {
            // 文本在、但写的是另一条签名是"数据不一致"；文本根本不在是"缺件"。
            let code = if gap.declared.is_none() {
                Code::Missing
            } else {
                Code::Invalid
            };
            Error::new(code, gap.to_string()).with(site.clone())
        })
        .collect()
}

/// 逐条错误合成一条：首条决定类别，其余作为同级诊断逐行保留；不把 `missing` 与 `invalid` 压成同一个码。
fn merge(errors: Vec<Error>) -> Error {
    let mut rest = errors.into_iter();
    match rest.next() {
        Some(first) => rest.fold(first, |first, other| first.also(&other)),
        // `validate` 只在有错时返回 `Err`；空列表是调用方违约，给一条显式内部诊断。
        None => Error::new(Code::Internal, "校验失败却没有给出任何一条诊断"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decl::expand;
    use crate::probe;
    use crate::substitute;
    use std::path::Path;

    testkit::test_cases! {
    fn builtin_data_is_valid() {
        let catalogue = Catalogue::try_builtin().expect("内置数据必须合法");
        assert_eq!(catalogue.ops().len(), nn::registry::REGISTRY.len());
        assert_eq!(catalogue.profiles().len(), prof::names().len());
    }

    fn cached_catalogue_is_identical_across_calls() {
        let first = Catalogue::try_builtin().expect("内置数据");
        let second = Catalogue::try_builtin().expect("内置数据");
        assert_eq!(first, second, "进程内缓存返回的必须是同一份目录的克隆");
    }

    fn builtin_witness_files_exist() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        catalogue
            .validate(&prof::Layout::builtin())
            .expect("见证文件必须存在");
    }

    /// 每台设备的每条 `provides` 都要有见证声明；等级不是门槛：`L0`/`L1` 的声明同样要给出"证据是什么"。
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

    /// 见证矩阵：闭集的 16 个原语在**每台设备**上都要有带见证的签名。只守"P0 用到的 9 个"会留下
    /// 自相矛盾的档案。空白格只在数据能解释时才合法（如单 lane 设备没有子组内交换的对象，也就不声明
    /// `shuffle`/`broadcast`）；其余补上的条目等级都是 `L0`——矩阵补的是覆盖，不是等级。
    fn every_device_covers_the_primitive_set_or_explains_the_gap() {
        use crate::atom::Prim;
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        for profile in catalogue.profiles() {
            let vendor = profile.meta.vendor.as_str();
            for prim in Prim::ALL.iter().copied() {
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

    /// 片段层与模板的调用签名对齐：每条 `@原语(...)` 都能被该设备的片段拼出来。参数个数或命名参数对不上
    /// 时，`substitute` 要到构建期才报错，那时离写错的地方已经很远。
    fn every_device_fragment_set_splices_the_p0_template() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let op = catalogue.op("gemv_i4").expect("内置算子");
        let combos = expand(op).expect("tile 候选");
        for profile in catalogue.profiles() {
            let vendor = profile.meta.vendor.as_str();
            let ext = profile.meta.emit.ext();
            let fragments = prof::builtin(vendor, ext)
                .unwrap_or_else(|| panic!("`{vendor}` 的片段层没有按 `. {ext}` 登记"));
            let backend = prof::catalogue::named_for(vendor, &op.name)
                .expect("后端必须有特化或继承得到的有效微内核")
                .to_decl()
                .expect("后端声明自洽");
            for combo in &combos {
                if let Err(error) = substitute::render(&backend, combo, &fragments) {
                    panic!("`{vendor}` 的片段与微内核签名不同构（{combo}）：{error}");
                }
            }
        }
    }

    /// 片段层声明的实现，必须正是协商会为这台设备挑中的那条。拼错一条不会让拼接失败，只会让生成的程序算别的算法。
    fn every_fragment_declares_the_implementation_negotiation_picks() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        for profile in catalogue.profiles() {
            let vendor = profile.meta.vendor.as_str();
            let set = prof::builtin(vendor, profile.meta.emit.ext())
                .unwrap_or_else(|| panic!("`{vendor}` 的片段层没有登记"));
            for op in catalogue.ops() {
                let gaps = cap::fragment_gaps(op, profile, |prim| set.implementation(prim));
                assert!(gaps.is_empty(), "`{vendor}`/`{}`：{gaps:?}", op.name);
            }
        }
    }

    /// 声明错一条签名就是装载期错误：不必等到构建期，更不能静默用错算法。
    fn a_fragment_declaring_another_implementation_fails_the_load() {
        let root = std::env::temp_dir().join(format!("cutile-sign-{}", std::process::id()));
        let prim_root = prof::layer_root(&root);
        for device in prof::devices() {
            let dir = prof::dir_of(device.vendor).expect("每台设备都有源码根");
            copy_dir(Path::new(dir), &prim_root.join(device.vendor));
        }
        // `nvidia` 的 `dot` 协商会挑中 `fma`；把文本改成声明 `dp4a`（另一条签名，属 P3）。
        let path = prim_root.join("nvidia").join("dot.cu");
        let text = std::fs::read_to_string(&path).expect("片段可读");
        assert!(text.contains("dot/fma"), "{text}");
        std::fs::write(&path, text.replacen("dot/fma", "dot/dp4a", 1)).expect("写片段");

        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let errors = catalogue
            .validate(&prof::Layout::at(&root))
            .expect_err("片段声明另一条签名必须报错");
        let rendered: Vec<String> = errors.iter().map(ToString::to_string).collect();
        let rendered = rendered.join("\n");
        assert!(rendered.contains("`nvidia`/`gemv_i4`"), "{rendered}");
        assert!(rendered.contains("dp4a"), "{rendered}");
        assert!(rendered.contains("fma"), "{rendered}");
        std::fs::remove_dir_all(&root).ok();
    }
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

    testkit::test_cases! {
    fn a_declaration_name_must_match_its_file_name() {
        let source = prof::catalogue::named("add_f32").expect("内置");
        let op = Op::new("renamed", source.view(), source.template());
        let profiles = Catalogue::try_builtin()
            .expect("内置数据")
            .profiles
            .to_vec();
        let error = Catalogue::assemble(vec![op], profiles).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("文件名即算子名"), "{error}");
    }

    /// 空片段层根下每条失败都是 `missing`：类别由数据层给，装载层不再一律压成 `invalid`。
    fn validation_keeps_the_category_of_the_failing_layer() {
        let root = std::env::temp_dir().join(format!("cutile-prim-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("建临时目录");
        let errors = Catalogue::try_builtin()
            .expect("内置数据")
            .validate(&prof::Layout::at(&root))
            .unwrap_err();
        assert!(!errors.is_empty(), "空片段层根必须报错");
        assert!(
            errors
                .iter()
                .any(|error| error.code() == Code::Missing && error.to_string().contains("见证文件")),
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

    fn duplicate_profiles_are_rejected() {
        let mut catalogue = Catalogue::try_builtin().expect("内置数据");
        let clone = catalogue.profiles[0].clone();
        Arc::make_mut(&mut catalogue.profiles).push(clone);
        let errors = catalogue.validate(&prof::Layout::builtin()).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.to_string().contains("两份档案"))
        );
    }

    fn detects_the_local_cpu_profile() {
        let catalogue = Catalogue::try_builtin().expect("内置数据");
        let detected = probe::detect_cpu();
        let profile = catalogue.select(&detected).expect("必须命中 CPU 档案");
        assert_eq!(profile.meta.vendor, "cpu");
    }
    }
}

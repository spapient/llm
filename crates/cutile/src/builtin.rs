//! 内置声明与档案：随 crate 编译，不依赖工作目录。

use crate::cap;
use crate::decl::OpDecl;
use crate::probe::select;
use crate::prof::Detected;
use crate::prof::Profile;
use diagnostic::{Code, Error};
use op::Op;
use std::sync::OnceLock;

/// 保留的设备名：不走档案查找，按本机探测挑一份档案。命令行默认值在 `cli` 侧另存一份，这里是库内比对口径。
pub const AUTO_DEVICE: &str = "auto";

/// 内置集合解析结果的进程内缓存：声明表与档案表都是编译期内嵌的只读数据，
/// 重复调用不该把每条 `OpDecl` / `Profile` 再解析一遍。类型全是自有数据，可跨线程共享。
static CATALOGUE: OnceLock<diagnostic::Result<Catalogue>> = OnceLock::new();

/// 声明 + 档案的集合。
#[derive(Debug, Clone, PartialEq)]
pub struct Catalogue {
    ops: Vec<OpDecl>,
    profiles: Vec<Profile>,
}

impl Catalogue {
    /// 内置集合。失败原样返回：本 crate 的生产代码没有 panic 边界，`builtin_data_is_valid` 才是"内置数据必须能解析"的执行者。
    pub fn try_builtin() -> diagnostic::Result<Self> {
        CATALOGUE
            .get_or_init(|| Self::assemble(op::builtin(), Self::builtin_profiles()?))
            .clone()
    }

    fn builtin_profiles() -> diagnostic::Result<Vec<Profile>> {
        let mut profiles = Vec::new();
        for name in prof::names() {
            let table = prof::embedded(name).ok_or_else(|| {
                Error::new(
                    Code::Missing,
                    format!("内置档案 `{name}` 不在生成的档案表里：构建脚本漏掉了它"),
                )
            })?;
            let profile = Profile::parse(table).map_err(|error| {
                Error::new(Code::Invalid, format!("内置档案 `{name}`：{error}"))
            })?;
            profiles.push(profile);
        }
        Ok(profiles)
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
            return self.select(&crate::probe::detect_cpu());
        }
        self.profile(device)
            .ok_or_else(|| Error::new(Code::Missing, format!("没有设备档案 `{device}`")))
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
    pub fn validate(&self, layout: &prim::Layout<'_>) -> Result<(), Vec<Error>> {
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
        for (index, profile) in self.profiles.iter().enumerate() {
            for other in &self.profiles[index + 1..] {
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
    fn profile_errors(&self, layout: &prim::Layout<'_>) -> Vec<Error> {
        let mut errors = Vec::new();
        for profile in &self.profiles {
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
    fn fragment_errors(&self, layout: &prim::Layout<'_>) -> Vec<Error> {
        let mut errors = Vec::new();
        for profile in &self.profiles {
            let ext = profile.meta.emit.ext();
            let set = match layout.fragments(&profile.meta.vendor, ext) {
                Ok(set) => set,
                Err(error) => {
                    errors.push(error.with(format!("`{}`", profile.meta.vendor)));
                    continue;
                }
            };
            for op in &self.ops {
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
fn gaps_of(op: &OpDecl, profile: &Profile, set: &prim::FragmentSet) -> Vec<Error> {
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
#[path = "../tests/builtin_tests.rs"]
mod tests;

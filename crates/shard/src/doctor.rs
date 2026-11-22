//! `doctor`：把整条流水线的每一步都摊开，包括协商为什么失败（README §16 的可解释性）。

use crate::build;
use crate::pipeline::Run;
use crate::toolchain;
use crate::witness::{self, Evidence};
use cutile::builtin::Catalogue;
use cutile::cache::{Cache, CacheKey, Changed, Entry};
use cutile::cap::{self, NoPath, Path, VariantCheck};
use cutile::dirs::Dirs;
use cutile::emit::Fragment;
use cutile::plan::Plan;
use cutile::prof::Profile;
use cutile::space;
use diagnostic::{Code, Error};
use op::decl::{OpDecl, expand};
use prof::atom::Space;
use std::path::PathBuf;

/// 一次 `doctor` 的结论：每一步的产物，排版交给前端。
#[derive(Debug, Clone, PartialEq)]
pub struct Doctor {
    /// 声明层头部。
    pub op: OpInfo,
    /// 档案层头部。
    pub device: DeviceInfo,
    /// 这次执行到的证据：三条命令共用同一份；协商用的是它的结论（`basis` 说明来自执行还是自述）。
    pub evidence: Evidence,
    /// 本次估值与 plan 用的形状 `(m, n)`：算子的生产形状（`Model::shape`），不是命令参数。
    pub shape: (usize, usize),
    /// 逐变体的需求复核：成功路径用它解释"为何选它、其它差在哪"。
    pub variants: Vec<VariantCheck>,
    /// 协商失败的诊断；成功时为 `None`，此时由 [`Doctor::variants`] 解释。
    pub no_path: Option<Box<NoPath>>,
    /// 协商结果；无可行变体时为 `None`，此时后面几项都没有产物。
    pub path: Option<Path>,
    /// ③ select 的产物。
    pub plan: Option<Plan>,
    /// 布局槽位的声明式。
    pub layouts: Vec<LayoutRow>,
    /// 逐层的容量与声明用量（`GM`/`FAST`/`REG` 三层都报）。
    pub storage: Vec<Storage>,
    /// ⑤ emit 的产物摘要。
    pub source: Option<Built>,
}

/// 一层存储：`capacity` 是判定的分母，`declared` 是分子的来源，合成一个数就两头都不是。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Storage {
    /// 语义存储层。
    pub space: Space,
    /// 本设备的物理名。
    pub physical: String,
    /// 档案声明的容量（字节）；`None` 表示档案没有声明，这一层不受限额。
    pub capacity: Option<u64>,
    /// 构建期契约比较过的声明用量；空表示没有原语声明过这一层。
    pub declared: Vec<Declared>,
}

/// 一条声明用量：哪条原语、要多少字节。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declared {
    /// 原语名（声明层写法）。
    pub prim: String,
    /// 字节数（声明给出的上界）。
    pub bytes: u64,
}

/// 声明层的头部信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpInfo {
    /// 算子名。
    pub name: String,
    /// 骨架。
    pub skeleton: String,
    /// 摘要。
    pub summary: String,
    /// 算子级需求条数。
    pub needs: usize,
    /// 变体数。
    pub variants: usize,
    /// 轴展开后的 tile 候选数。
    pub tiles: usize,
}

/// 档案层的头部信息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// 厂商。
    pub vendor: String,
    /// 档案 id（决定缓存目录）。
    pub id: String,
    /// emit 语言。
    pub emit: String,
    /// 编译工具（档案声明的命令名）。
    pub tool: String,
    /// 档案登记的工具版本（人写的证据出处，不参与缓存键）。
    pub tool_ver: String,
    /// 实测的工具链身份（`命令@版本`）：探测的是档案声明的那个命令（`[meta].tools`，可带固定前缀
    /// 参数），本机没装它时为 `None`。`verify` 报的是另一件事——真正编出产物的那条。
    pub toolchain: Option<String>,
    /// 编译参数（档案的 `[compile].flags`）：能力是「片段 + 工具链 + 参数」三元组。
    pub flags: String,
    /// `GM` / `FAST` / `REG` 的容量（档案原文，单位自带）。
    pub spaces: [String; 3],
    /// 执行单元（名字与可访问的存储层）。
    pub units: Vec<(String, String)>,
}

/// 一个布局槽位的声明：名字与声明式。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutRow {
    /// 槽位名。
    pub slot: String,
    /// 声明层写法。
    pub spec: String,
}

/// 一次 emit 的产物摘要。源文件本身不重复返回：它已经落在缓存里，按 `cache` 取。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Built {
    /// 行数与 §16 的口径。
    pub fragment: Fragment,
    /// 扩展名。
    pub ext: String,
    /// 缓存键：十六进制形式与逐项说明都由它自己给出。
    pub key: CacheKey,
    /// 缓存目录（尚未落盘时为 `None`）。
    pub cache_dir: Option<PathBuf>,
}

/// 判据：每一步的产物是什么、协商为什么不行；不回答"跑得对不对"——它不执行内核。
pub fn doctor(
    catalogue: &Catalogue,
    dirs: &Dirs,
    op_name: &str,
    device: &str,
) -> diagnostic::Result<Doctor> {
    let (op, info, profile) = target(catalogue, op_name, device)?;
    // 证据先跑：协商用的不是档案里的 `grade`，而是这次执行得到的 `Proof`。
    let evidence = witness::establish(profile, dirs);
    let run = Run::new(op, profile, evidence.proof(profile))?;
    let mut report = Doctor {
        op: info,
        device: device_info(profile),
        evidence,
        shape: run.model.shape,
        variants: Vec::new(),
        no_path: None,
        path: None,
        plan: None,
        layouts: Vec::new(),
        storage: Vec::new(),
        source: None,
    };
    explain(&run, dirs, &mut report)?;
    Ok(report)
}

/// 判据：名字与设备目录都存在吗；不回答"它可行不可行"——那是 [`explain`] 的事。
fn target<'a>(
    catalogue: &'a Catalogue,
    op_name: &str,
    device: &str,
) -> diagnostic::Result<(&'a OpDecl, OpInfo, &'a Profile)> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    Ok((op, op_info(op)?, catalogue.resolve(device)?))
}

/// 判据：这份声明有多大；不回答"它在这个形状下要多少代价"——估值在 `plan` 里。
fn op_info(op: &OpDecl) -> diagnostic::Result<OpInfo> {
    Ok(OpInfo {
        name: op.name.clone(),
        skeleton: op.skeleton.to_string(),
        summary: op.summary.clone(),
        needs: op.needs.len(),
        variants: op.variants.len(),
        tiles: expand(op)?.len(),
    })
}

/// 判据：用的是哪份档案、它指向哪条工具链；不回答"这份档案在本机跑不跑得动"（证据另报）。
fn device_info(profile: &Profile) -> DeviceInfo {
    DeviceInfo {
        vendor: profile.meta.vendor.to_string(),
        id: profile.meta.id.to_string(),
        emit: profile.meta.emit.to_string(),
        tool: profile.meta.tool.to_string(),
        tool_ver: profile.meta.tool_ver.to_string(),
        toolchain: toolchain::probe_declared(&profile.meta.tool).map(|tool| tool.id()),
        flags: profile.flags.join(" "),
        spaces: [
            space::physical(profile, Space::Gm).to_string(),
            space::physical(profile, Space::Fast).to_string(),
            space::physical(profile, Space::Reg).to_string(),
        ],
        units: profile
            .units
            .iter()
            .map(|unit| (unit.name.clone(), unit.space.to_string()))
            .collect(),
    }
}

/// 判据：协商为什么行 / 为什么不行；不回答"选中的变体能编得过吗"——协商失败不是错误。
fn explain(run: &Run<'_>, dirs: &Dirs, report: &mut Doctor) -> diagnostic::Result<()> {
    match run.negotiate() {
        Err(no_path) => report.no_path = Some(no_path),
        Ok(path) => filled(run, dirs, report, path)?,
    }
    Ok(())
}

/// 判据：逐变体复核、布局、容量与用量、emit 摘要各是什么；不回答"跑得对不对"。
fn filled(run: &Run<'_>, dirs: &Dirs, report: &mut Doctor, path: Path) -> diagnostic::Result<()> {
    report.variants = cap::inspect_with(run.op, run.profile, &run.proof);
    let (m, n) = run.model.shape;
    let prepared = run.prepare(&path, dirs, m, n)?;
    // 每一层都报容量与用量：判定比较的就是这两个数，报告不能只给第三个。
    let uses = build::contract(run.op, run.profile, &path)?;
    report.layouts = run
        .op
        .layouts
        .iter()
        .map(|(slot, spec)| LayoutRow {
            slot: slot.clone(),
            spec: spec.to_string(),
        })
        .collect();
    report.storage = Space::ALL
        .iter()
        .copied()
        .map(|space| Storage {
            space,
            physical: space::physical(run.profile, space).to_string(),
            capacity: space::capacity(run.profile, space),
            declared: declared(&uses, space),
        })
        .collect();
    report.source = Some(Built {
        fragment: Fragment::new(prepared.source.lines()),
        ext: prepared.source.ext().to_string(),
        cache_dir: build::dir(&prepared.key, dirs),
        key: prepared.key.clone(),
    });
    report.plan = Some(prepared.plan);
    report.path = Some(path);
    Ok(())
}

/// 判据：谁声明了多少字节；不回答"这些声明合起来是否超容量"——那由构建期契约判定。
fn declared(uses: &[build::Usage], space: Space) -> Vec<Declared> {
    uses.iter()
        .filter(|usage| usage.space == space)
        .map(|usage| Declared {
            prim: usage.prim.to_string(),
            bytes: usage.bytes,
        })
        .collect()
}

/// `doctor --why` 的结论：本次构建的键、缓存里那份记录、以及两者的逐项差异。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheWhy {
    /// 本次构建算出的键。
    pub current: String,
    /// 该键在缓存里的记录；`None` 表示从未落盘或已被回收。
    pub entry: Option<Entry>,
    /// 逐项差异；空表示这份缓存就是当前输入对应的产物。
    pub changes: Vec<Changed>,
}

impl CacheWhy {
    /// 判据：从未落盘 / 半截落盘 / 命中 / 有差异；不回答"差异在哪一项"（见 [`Self::changes`]）。
    #[must_use]
    pub fn state(&self) -> CacheState {
        let Some(entry) = &self.entry else {
            return CacheState::Never;
        };
        if !entry.complete() {
            return CacheState::Partial;
        }
        if self.changes.is_empty() {
            CacheState::Match
        } else {
            CacheState::Differs
        }
    }
}

/// `doctor --why` 的结论类别（文本在驱动方）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    /// 从未落盘，或已被回收。
    Never,
    /// 只有半截落盘（缺 `source`）：写到一半就没了，永远不会命中。
    Partial,
    /// 就是本次输入对应的产物。
    Match,
    /// 落盘过，但字段与本次构建不同。
    Differs,
}

/// 解释一个缓存键：本次构建的键与缓存里那份差在哪几个字段。
#[must_use]
pub fn why(dirs: &Dirs, current: &CacheKey, hex: &str) -> CacheWhy {
    let entry = Cache::from_dirs(dirs).entry(hex);
    let changes = current.changes(entry.as_ref().map_or(&[][..], |entry| &entry.fields));
    CacheWhy {
        current: current.hex(),
        entry,
        changes,
    }
}

#[cfg(test)]
#[path = "../tests/doctor_tests.rs"]
mod tests;

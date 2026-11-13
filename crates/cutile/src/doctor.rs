//! `doctor`：把整条流水线的每一步都摊开，包括协商为什么失败。

use crate::backend;
use crate::build;
use crate::builtin::Catalogue;
use crate::cache::{Cache, CacheKey, Changed, Entry};
use crate::cap::{self, NoPath, Path, VariantCheck};
use crate::emit::Fragment;
use crate::pipeline::Run;
use crate::plan::Plan;
use crate::space;
use crate::witness::{self, Evidence};
use diagnostic::{Code, Error};
use llm_config::Settings;
use op::decl::expand;
use prof::atom::Space;
use std::path::PathBuf;

/// 一次 `doctor` 的结论：每一步的产物，排版交给前端。
#[derive(Debug, Clone, PartialEq)]
pub struct Doctor {
    /// 声明层头部。
    pub op: OpInfo,
    /// 档案层头部。
    pub device: DeviceInfo,
    /// 这次执行到的证据：三条命令共用同一份。
    ///
    /// 协商用的是它的结论，不是档案里的 `grade`；`basis` 说明这份结论来自执行还是来自自述。
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

/// 一层存储：档案声明的容量 + 构建期契约比较过的声明用量。
///
/// 两件事必须分开报：`capacity` 是判定的分母，`declared` 是分子的来源。把它们合成一个
/// “字节数”只会打印出既不是容量、也不是用量的第三个数。全是数据，排版在前端。
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
    /// 实测的工具链身份（`命令@版本`）：探测的是档案声明的那个命令（`[meta].tool`），
    /// 因为 `doctor` 回答的是"这台设备按这份档案要用什么工具链"；本机没装它时为 `None`。
    ///
    /// `verify` 报的是另一件事——真正编出产物的那条（`backend::detect` 依次试 `cc`/`clang`/`gcc`）。
    /// 两者在只装了 `clang` 的机器上会不同，这正是要分别写出来的原因。
    pub toolchain: Option<String>,
    /// 编译参数（档案的 `[compile].flags`，按声明顺序）。
    ///
    /// 能力是「片段 + 工具链 + 参数」三元组：同一份片段在这一组参数下被证明可用，换一组就是
    /// 另一次代码生成。报告里写出它，读者才能回答"这份证据适用于哪次编译"。
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

/// 一次 emit 的产物摘要。源文件本身不重复返回：它已经落在缓存里，按 `cache_dir` 取。
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

/// 协商失败不是错误：`doctor` 的职责就是解释"为什么这台设备不行"，因此照样返回报告，
/// 只是 `path`/`plan`/`source` 为 `None`。
///
/// # Errors
///
/// 算子或设备档案不存在、配置读不出来、emit 失败时返回说明。
pub fn doctor(
    catalogue: &Catalogue,
    settings: &Settings,
    op_name: &str,
    device: &str,
) -> diagnostic::Result<Doctor> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let profile = catalogue.resolve(device)?;

    let info = OpInfo {
        name: op.name.clone(),
        skeleton: op.skeleton.to_string(),
        summary: op.summary.clone(),
        needs: op.needs.len(),
        variants: op.variants.len(),
        tiles: expand(op)?.len(),
    };
    let device = DeviceInfo {
        vendor: profile.meta.vendor.to_string(),
        id: profile.meta.id.to_string(),
        emit: profile.meta.emit.to_string(),
        tool: profile.meta.tool.to_string(),
        tool_ver: profile.meta.tool_ver.to_string(),
        toolchain: backend::probe(&profile.meta.tool).map(|tool| tool.id()),
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
    };

    // 证据先跑：协商用的不是档案里的 `grade`，而是这次执行得到的 `Proof`。见证不成立时协商
    // 必然失败（拒绝理由写成“L2 但没有这次执行的见证”），而 `doctor` 仍交报告——它的职责是
    // 解释为什么不行，不是把原因吞成一个 `Err`。
    let evidence = witness::establish(profile, settings);
    let run = Run::new(op, profile, evidence.proof(profile))?;
    let mut report = Doctor {
        op: info,
        device,
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

    // 协商只跑一次：失败时把诊断原样带回去，成功时再逐变体复核一遍用于解释。
    let path = match run.negotiate() {
        Ok(path) => path,
        Err(no_path) => {
            report.no_path = Some(no_path);
            return Ok(report);
        }
    };
    report.variants = cap::inspect_with(run.op, run.profile, &run.proof);
    let (m, n) = run.model.shape;
    let prepared = run.prepare(&path, settings, m, n)?;

    report.layouts = run
        .op
        .layouts
        .iter()
        .map(|(slot, spec)| LayoutRow {
            slot: slot.clone(),
            spec: spec.to_string(),
        })
        .collect();
    // 每一层都报容量与用量：判定比较的就是这两个数，报告不能只给第三个。
    let uses = build::contract(run.op, run.profile, &path)?;
    report.storage = Space::ALL
        .iter()
        .copied()
        .map(|space| Storage {
            space,
            physical: space::physical(profile, space).to_string(),
            capacity: space::capacity(profile, space),
            declared: uses
                .iter()
                .filter(|usage| usage.space == space)
                .map(|usage| Declared {
                    prim: usage.prim.to_string(),
                    bytes: usage.bytes,
                })
                .collect(),
        })
        .collect();
    report.source = Some(Built {
        fragment: Fragment::new(prepared.source.lines()),
        ext: prepared.source.ext().to_string(),
        cache_dir: build::dir(&prepared.key, settings),
        key: prepared.key,
    });
    report.plan = Some(prepared.plan);
    report.path = Some(path);
    Ok(report)
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
    /// 这份缓存与本次构建的关系。
    ///
    /// 必须先把"半截落盘"分出来：它没有 `source`，逐项差异自然为空，但那不代表命中——
    /// 按差异判会把它说成"一致"，而它根本读不出源文本。
    #[must_use]
    pub fn state(&self) -> CacheState {
        match &self.entry {
            None => CacheState::Never,
            Some(entry) if !entry.complete() => CacheState::Partial,
            Some(_) if self.changes.is_empty() => CacheState::Match,
            Some(_) => CacheState::Differs,
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
///
/// 字段名与取值都来自 [`CacheKey::fields`]，缓存侧来自 `meta.txt`——两边是同一份定义，
/// 所以不需要拿两行拼接文本做人肉 diff。
#[must_use]
pub fn why(settings: &Settings, current: &CacheKey, hex: &str) -> CacheWhy {
    let entry = Cache::from_settings(settings).entry(hex);
    let changes = current.changes(entry.as_ref().map_or(&[][..], |entry| &entry.fields));
    CacheWhy {
        current: current.hex(),
        entry,
        changes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::witness::Basis;
    use prof::atom::Prim;

    fn settings() -> Settings {
        let roots = llm_config::Roots::new(std::env::temp_dir().join("cutile-doctor-tests"));
        Settings::<llm_config::Cutile>::defaults(&roots)
    }

    fn catalogue() -> crate::builtin::Catalogue {
        crate::builtin::Catalogue::try_builtin().expect("内置数据")
    }

    /// 独立缓存目录：`--why` 会读真目录，不能与其他用例共用。
    fn cache_settings(name: &str) -> Settings {
        let root = std::env::temp_dir().join(format!("cutile-why-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut settings = Settings::<llm_config::Cutile>::defaults(&llm_config::Roots::new(&root));
        settings
            .apply_override(&format!("cache.dir={}", root.join("cache").display()))
            .expect("cache.dir 覆盖合法");
        settings
    }

    #[test]
    fn doctor_reports_the_full_chain() {
        let report = doctor(&catalogue(), &settings(), "gemv_i4", "cpu").expect("doctor 应成功");
        assert_eq!(report.path.as_ref().expect("有路径").variant, "pipe2");
        assert!(!report.layouts.is_empty(), "{report:?}");
        let source = report.source.as_ref().expect("有 emit 产物");
        assert!(source.fragment.lines > 0);
        assert!(source.fragment.within_limit(), "{source:?}");
        let hex = source.key.hex();
        assert_eq!(hex.len(), 16, "缓存键是 16 位十六进制");
        assert!(hex.chars().all(|digit| digit.is_ascii_hexdigit()));
    }

    #[test]
    fn doctor_reports_each_variant_as_data() {
        let report = doctor(&catalogue(), &settings(), "gemv_i4", "cpu").expect("doctor 应成功");
        assert!(report.no_path.is_none(), "{report:?}");
        assert_eq!(report.variants.len(), 2);
        assert!(
            report
                .variants
                .iter()
                .all(|check| check.failures.is_empty())
        );
    }

    #[test]
    fn doctor_explains_why_a_gpu_is_rejected() {
        let report = doctor(&catalogue(), &settings(), "gemv_i4", "nvidia").expect("仍应返回报告");
        assert!(report.path.is_none(), "{report:?}");
        let no_path = report.no_path.as_ref().expect("失败要带回诊断");
        assert!(no_path.to_string().contains("L2"), "{no_path}");
    }

    /// 协商用的是这次执行到的证据：条数与依据都要能对上档案。
    #[test]
    fn doctor_reports_the_evidence_it_ran() {
        if crate::backend::detect().is_none() {
            return;
        }
        let catalogue = catalogue();
        let report = doctor(&catalogue, &settings(), "gemv_i4", "cpu").expect("doctor 应成功");
        let profile = catalogue.profile("cpu").expect("cpu 档案");
        assert_eq!(
            report.evidence.basis,
            Basis::Executed,
            "C 档案在本机就是设备"
        );
        assert!(
            report.evidence.failure.is_none(),
            "证据不成立：{:?}",
            report.evidence.failure
        );
        assert_eq!(
            report.evidence.device,
            profile.provides.len(),
            "设备就在本机：每一条已声明的见证都要在这次证据里（不按 `grade` 筛）"
        );
        assert_eq!(report.path.as_ref().expect("有路径").variant, "pipe2");
    }

    /// 本机执行不了的设备：依据必须写成"档案自述"，不能假装执行过。
    #[test]
    fn a_device_that_cannot_run_here_is_negotiated_on_declarations() {
        let report = doctor(&catalogue(), &settings(), "gemv_i4", "nvidia").expect("仍应返回报告");
        assert_eq!(report.evidence.basis, Basis::Declared);
        assert_eq!(report.evidence.device, 0, "本机没有设备执行");
        assert!(
            report.evidence.host > 0,
            "自检仍要跑：见证源码算得对不对在这里是可验证的"
        );
    }

    /// 见证对不上时能力就不成立：同一份档案只把一处期望值写错，协商也必须全线不可行。
    #[test]
    fn a_witness_that_does_not_hold_removes_the_capability() {
        if crate::backend::detect().is_none() {
            return;
        }
        let mut profile = crate::prof::Profile::parse(prof::embedded("cpu").expect("cpu 档案"))
            .expect("档案可解析");
        // `quant` 的见证解包 `0xF0` 得 `-4.0`：把期望值改一位，执行器必须发现。
        let quant = profile
            .provides
            .iter_mut()
            .find(|provide| provide.prim == Prim::Quant)
            .expect("cpu 有 quant");
        quant.witness.as_mut().expect("quant 有见证").expect[0] = -5.0;

        let evidence = witness::establish(&profile, &settings());
        assert!(
            evidence.failure.is_some(),
            "见证对不上就是证据不成立：{evidence:?}"
        );
        assert!(evidence.proof(&profile).is_empty(), "一条都不该被证明");

        let op = catalogue().op("gemv_i4").expect("算子").clone();
        let error =
            crate::cap::negotiate_with(&op, &profile, &evidence.proof(&profile)).unwrap_err();
        assert!(error.to_string().contains("没有这次执行的见证"), "{error}");
    }

    #[test]
    fn storage_reports_the_archive_capacity_next_to_the_declared_traffic() {
        let report = doctor(&catalogue(), &settings(), "gemv_i4", "cpu").expect("doctor 应成功");
        assert_eq!(report.storage.len(), 3, "三层都要报：{report:?}");
        let fast = report
            .storage
            .iter()
            .find(|row| row.space == prof::atom::Space::Fast)
            .expect("有 FAST 行");
        assert_eq!(fast.capacity, Some(524_288), "容量取档案声明，不是中转缓冲");
        assert!(
            fast.declared
                .iter()
                .any(|usage| usage.prim == "load" && usage.bytes == 4096),
            "用量取构建期契约比较过的声明：{fast:?}"
        );
        let reg = report
            .storage
            .iter()
            .find(|row| row.space == prof::atom::Space::Reg)
            .expect("有 REG 行");
        assert_eq!(reg.capacity, Some(65_536));
        assert!(
            reg.declared.is_empty(),
            "没有声明字节数的原语不占 REG：{reg:?}"
        );
        let gm = report
            .storage
            .iter()
            .find(|row| row.space == prof::atom::Space::Gm)
            .expect("有 GM 行");
        assert_eq!(gm.capacity, None, "GM 不受限额，不编一个容量");
    }

    #[test]
    fn the_planned_shape_comes_from_the_operator_model() {
        let model = crate::pipeline::model_for("gemv_i4").expect("有模型");
        let report = doctor(&catalogue(), &settings(), "gemv_i4", "cpu").expect("doctor 应成功");
        assert_eq!(report.shape, model.shape);
        assert!(
            report.layouts.iter().all(|row| row.spec == "row"),
            "{report:?}"
        );
    }

    #[test]
    fn why_separates_half_written_from_never_written() {
        let settings = cache_settings("cache-why-partial");
        let report = doctor(&catalogue(), &settings, "gemv_i4", "cpu").expect("doctor 应成功");
        let key = report.source.expect("有产物").key;
        assert_eq!(why(&settings, &key, &key.hex()).state(), CacheState::Match);

        let dir = settings.cache_dir().join(&key.hex()[..2]).join(key.hex());
        std::fs::remove_file(dir.join("source")).expect("拿掉 source");
        let partial = why(&settings, &key, &key.hex());
        assert_eq!(partial.state(), CacheState::Partial, "{partial:?}");
        assert!(partial.entry.is_some(), "半截目录仍是一条记录：{partial:?}");
        assert!(
            partial.changes.is_empty(),
            "没有 source 不代表键不同：{partial:?}"
        );

        assert_eq!(
            why(&settings, &key, "0000000000000000").state(),
            CacheState::Never
        );
        let _ = std::fs::remove_dir_all(settings.cache_dir());
    }

    #[test]
    fn why_names_the_field_that_invalidated_the_cache() {
        let settings = cache_settings("cache-why");
        let report = doctor(&catalogue(), &settings, "gemv_i4", "cpu").expect("doctor 应成功");
        let key = report.source.expect("有产物").key;

        let hit = why(&settings, &key, &key.hex());
        assert!(hit.entry.is_some(), "doctor 已落盘：{hit:?}");
        assert!(hit.changes.is_empty(), "同一份键不应有差异：{hit:?}");

        let mut other = key.clone();
        other.tile = "GMAX=64".to_string();
        let miss = why(&settings, &other, &key.hex());
        let fields: Vec<&str> = miss
            .changes
            .iter()
            .map(|change| change.field.as_str())
            .collect();
        assert_eq!(fields, ["tile"], "只该指出真正变了的那项：{miss:?}");
        assert_eq!(miss.current, other.hex());

        let absent = why(&settings, &key, "0000000000000000");
        assert!(absent.entry.is_none(), "{absent:?}");
        assert!(
            !absent.changes.is_empty(),
            "没有缓存记录就不能声称无差异：{absent:?}"
        );
        let _ = std::fs::remove_dir_all(settings.cache_dir());
    }
}

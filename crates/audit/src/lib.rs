//! 内核审计：`doctor` / `verify` / `bench` 的**计算**（DESIGN §16 的 P0 交付物）。
//!
//! 三条命令共用同一条流水线：声明层 → 协商 → tile 取值 → [`cutile::build`] 的拼接与 emit →
//! 编译执行。任何一步失败都带上下文返回 `Err`，不写"跳过"（DESIGN §11.2 对旧工程的修正）。
//!
//! 本 crate 只回答"算出来是什么"：结果全是结构化报告，没有 `println!`，也不拼装人读排版——
//! 排版属于前端（`llm-cli` 的语法 + 驱动方）。
//!
//! 之所以不在 `cutile` 里：内核只做机制（协商、拼接、emit、编译执行），这里做的是**命令策略**
//! ——形状矩阵、重复次数、工作目录命名、达成率口径。两者混在一起时，内核的预算表只能靠一张
//! "实测偏差"的脚注解释，"内核有多大"也就没有答案了（DESIGN §13.1）。
//!
//! 数据从哪读也不在这里（[`Catalogue::load`] / [`Catalogue::resolve`]）：装载与交叉校验属于内核。

#![deny(unsafe_code)]

pub mod witness;

use cutile::backend;
use cutile::build;
use cutile::builtin::Catalogue;
use cutile::cache::{Artifact, Cache, CacheKey, Changed, Entry, Gc};
use cutile::cap::{self, NoPath, Path, VariantCheck};
use cutile::emit::Source;
use cutile::plan::{self, Plan};
use cutile::reference;
use cutile::space;
use diagnostic::{Code, Error};
use llm_config::Settings;
use op::decl::{OpDecl, expand};
use op::model::{Model, Sample, Values};
use prof::atom::Space;
use prof::prof::Profile;
use std::path::PathBuf;

/// L3 达成率门槛（DESIGN §12.3）：低于它就是"没有可用证据"。
pub const L3_THRESHOLD: f64 = 0.7;

/// 一次 `doctor` 的结论：每一步的产物，排版交给前端。
#[derive(Debug, Clone, PartialEq)]
pub struct Doctor {
    /// 声明层头部。
    pub op: OpInfo,
    /// 档案层头部。
    pub device: DeviceInfo,
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
/// “字节数”只会打印出既不是容量、也不是用量的第三个数——`FAST=L2 (128 B)` 曾把一层
/// 512 KiB 的缓存说成 128 B。全是数据，排版在前端。
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
    /// 编译工具。
    pub tool: String,
    /// 工具版本。
    pub tool_ver: String,
    /// `GM` / `FAST` / `REG` 的容量（档案原文，单位自带）。
    pub spaces: [String; 3],
    /// 执行单元（名字与可访问的存储层）。
    pub units: Vec<(String, String)>,
}

/// 一个布局槽位的声明：名字与声明式。
///
/// 构建期没有具体形状，正规形（步长/填充展开）无从谈起：`doctor` 打印的必须是缓存键里那一个
/// 声明式，否则"为什么 key 变了"会指向一个从未参与编译的步长。
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
    /// 行数。
    pub lines: usize,
    /// 扩展名。
    pub ext: String,
    /// 缓存键：十六进制形式与逐项说明都由它自己给出。
    pub key: CacheKey,
    /// 缓存目录（尚未落盘时为 `None`）。
    pub cache_dir: Option<PathBuf>,
}

/// 一台设备通过 L2 的证据。
#[derive(Debug, Clone, PartialEq)]
pub struct Verified {
    /// 用户给的设备名（`auto` 或档案名）。
    pub device: String,
    /// 厂商。
    pub vendor: String,
    /// 档案 id。
    pub id: String,
    /// 变体。
    pub variant: String,
    /// tile。
    pub tile: String,
    /// 实际跑过的形状（`verify` 的形状矩阵）。
    pub shapes: Vec<(u64, u64)>,
    /// 执行通过的见证条数（每条 `grade ≥ L2` 的供给一条）。
    pub witnesses: usize,
    /// 与标量参考的相对误差（取全部形状的最大值）。
    pub max_rel_err: f64,
    /// 源文件行数。
    pub lines: usize,
    /// 缓存键（十六进制）。
    pub key: String,
    /// 编译产物：形状矩阵的每一项一份（`reused` 为真表示那次没调用编译器）。
    pub artifacts: Vec<Artifact>,
}

/// 一台设备的结果。
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyEntry {
    /// 编译、执行、对拍都在容差内。
    Passed(Verified),
    /// 未通过；`label` 是档案的呈现名（`vendor(id)`，没有档案时就是用户给的设备名）。
    Failed {
        /// 呈现名。
        label: String,
        /// 失败原因。
        error: Error,
    },
}

impl VerifyEntry {
    /// 失败原因；通过时为 `None`。
    pub fn error(&self) -> Option<&Error> {
        match self {
            Self::Passed(_) => None,
            Self::Failed { error, .. } => Some(error),
        }
    }
}

/// 一次 `verify` 的逐设备结果。
///
/// "缺设备即失败"是调用方据 [`VerifyReport::failures`] 决定的（DESIGN §11.2），本模块只
/// 如实记录每台设备的结果——把失败变成"跳过"正是旧工程的错误。
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyReport {
    /// 算子名。
    pub op: String,
    /// 请求验证的设备（原样，含 `auto`）。
    pub devices: Vec<String>,
    /// 与 `devices` 一一对应的结果。
    pub entries: Vec<VerifyEntry>,
}

impl VerifyReport {
    /// 未通过的项。
    pub fn failures(&self) -> Vec<&Error> {
        self.entries.iter().filter_map(VerifyEntry::error).collect()
    }
}

/// 一次 `bench` 的实测口径（DESIGN §12.3 的台账）。
#[derive(Debug, Clone, PartialEq)]
pub struct Bench {
    /// 算子名。
    pub op: String,
    /// 形状 `(m, n)`。
    pub shape: (u64, u64),
    /// 重复次数（= 计时样本数；warmup 不计入）。
    pub repeat: usize,
    /// ③ select 的产物（预测值就在这里）。
    pub plan: Plan,
    /// 每次调用的耗时（纳秒）：全部样本的 min（最好情况）。
    pub per_run_ns: f64,
    /// 每次调用的耗时中位数（纳秒）：达成率按它计算，对单次抖动与频率状态更稳。
    pub p50_ns: f64,
    /// 实测吞吐。
    pub measured: Throughput,
    /// 档案声明的带宽峰值与来源。
    pub peak: Option<Peak>,
    /// 输入工作集（字节）：权重 + scales + 激活，不含输出。
    ///
    /// 达成率离开它就没有意义：工作集小于缓存时测到的是缓存带宽，不是档案里的
    /// 主存峰值（DESIGN §12.3 要求每次达成率都附口径）。
    pub working_set: u64,
    /// 档案声明的 FAST 容量（字节，0 表示未声明）。
    pub fast_bytes: u64,
    /// 缓存目录。
    pub cache_dir: Option<PathBuf>,
    /// 这次跑的那份编译产物：`reused` 为真表示没调用编译器。
    pub artifact: Artifact,
}

impl Bench {
    /// 口径结论：达成率只在 `Reached`/`Below` 时参与 L3 判定。
    pub fn verdict(&self) -> Verdict {
        classify(self.measured.bandwidth_ratio, self.in_fast())
    }

    /// 工作集是否落在档案声明的 FAST 容量内（重复执行会命中缓存）。
    pub fn in_fast(&self) -> bool {
        self.fast_bytes > 0 && self.working_set <= self.fast_bytes
    }
}

/// `bench` 的口径结论（DESIGN §12.1/§12.3）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// 档案没有带宽峰值：无法判定。
    Unknown,
    /// 实测带宽超过档案峰值：数据驻留在缓存中，这个达成率不是同一把尺子。
    OverPeak,
    /// 工作集落在 FAST 容量内：测到的是缓存带宽，与档案的主存峰值不可比。
    CacheResident,
    /// 达到 L3 门槛（≥ 70%）。
    Reached,
    /// 低于 L3 门槛。
    Below,
}

/// 由达成率与工作集判定口径。
///
/// `> 1` 不是"更好"，而是说明测量与档案峰值不是同一口径；工作集装得进缓存时，
/// 即便达成率没超过 100%，分母（主存峰值）也不是这次测量的口径。
fn classify(ratio: Option<f64>, cache_resident: bool) -> Verdict {
    match ratio {
        None => Verdict::Unknown,
        Some(ratio) if ratio > 1.0 => Verdict::OverPeak,
        _ if cache_resident => Verdict::CacheResident,
        Some(ratio) if ratio >= L3_THRESHOLD => Verdict::Reached,
        Some(_) => Verdict::Below,
    }
}

/// 一次 bench 的实测口径。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Throughput {
    /// 带宽（GB/s）。
    pub gb_s: f64,
    /// 算力（GFLOP/s）。
    pub gflops: f64,
    /// 带宽达成率（0–1）；档案没有峰值或耗时为零时为 `None`。
    pub bandwidth_ratio: Option<f64>,
}

/// 档案里的一个峰值：取值与人可读的来源（DESIGN §12.1 要求每个数都带推导式来源）。
#[derive(Debug, Clone, PartialEq)]
pub struct Peak {
    /// 以 B/s 计的取值。
    pub value: f64,
    /// 来源。
    pub src: String,
}

/// `doctor`：把整条流水线的每一步都摊开，包括协商为什么失败。
///
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
    let model = model_for(&op.name)?;
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

    let mut report = Doctor {
        op: info,
        device,
        shape: model.shape,
        variants: Vec::new(),
        no_path: None,
        path: None,
        plan: None,
        layouts: Vec::new(),
        storage: Vec::new(),
        source: None,
    };

    // 协商只跑一次：失败时把诊断原样带回去，成功时再逐变体复核一遍用于解释。
    let path = match cap::negotiate(op, profile) {
        Ok(path) => path,
        Err(no_path) => {
            report.no_path = Some(no_path);
            return Ok(report);
        }
    };
    report.variants = cap::inspect(op, profile);
    let (m, n) = model.shape;
    let cost = (model.cost)(m, n)?;
    let plan = plan::select(op, profile, &path, cost)?;
    let (source, key) = build::source(op, profile, &path, &plan.tile, settings)?;

    report.layouts = op
        .layouts
        .iter()
        .map(|(slot, spec)| LayoutRow {
            slot: slot.clone(),
            spec: spec.to_string(),
        })
        .collect();
    // 每一层都报容量与用量：判定比较的就是这两个数，报告不能只给第三个。
    // `build::source` 会再判一次契约——它必须能独立保证产物合法，而这里要的是判定用的输入。
    let uses = build::contract(op, profile, &path)?;
    report.storage = Space::ALL
        .into_iter()
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
        lines: source.lines(),
        ext: source.ext().to_string(),
        cache_dir: build::dir(&key, settings),
        key,
    });
    report.plan = Some(plan);
    report.path = Some(path);
    Ok(report)
}

/// `verify`：逐设备执行见证 + 协商 + 拼接 + 编译 + 与标量参考对拍。
///
/// 见证先跑：一条 `grade ≥ L2` 的供给如果没有自己的执行结果，后面的形状对拍就不再
/// 是证据链的一部分（DESIGN §11.3）。
///
/// # Errors
///
/// 算子不存在时返回说明；设备失败不算错误（写在 [`VerifyReport`] 里）。
pub fn verify(
    catalogue: &Catalogue,
    settings: &Settings,
    op_name: &str,
    devices: &[String],
) -> diagnostic::Result<VerifyReport> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let mut entries = Vec::with_capacity(devices.len());
    for device in devices {
        let profile = match catalogue.resolve(device) {
            Ok(profile) => profile,
            Err(error) => {
                entries.push(VerifyEntry::Failed {
                    label: device.clone(),
                    error,
                });
                continue;
            }
        };
        let label = format!("{}({})", profile.meta.vendor, profile.meta.id);
        let witnesses = match witness::run_all(profile, settings) {
            Ok(runs) => runs,
            Err(error) => {
                entries.push(VerifyEntry::Failed {
                    label: label.clone(),
                    error,
                });
                continue;
            }
        };
        // 非 C 设备在本机没有设备，但它的见证已经写成了 host C ABI：先把能执行的证据执行掉，
        // 再如实报“缺设备”。自检失败是硬错误——不能退化成“没设备所以没结论”。
        let self_checked = if profile.meta.emit == cutile::prof::Emit::C {
            None
        } else {
            match witness::self_check(profile, settings) {
                Ok(runs) => Some(runs.len()),
                Err(error) => {
                    entries.push(VerifyEntry::Failed {
                        label: label.clone(),
                        error,
                    });
                    continue;
                }
            }
        };
        entries.push(match verify_one(op, profile, settings) {
            Ok(mut verified) => {
                verified.device = device.clone();
                verified.witnesses = witnesses.len();
                VerifyEntry::Passed(verified)
            }
            Err(error) => VerifyEntry::Failed {
                label,
                error: match self_checked {
                    Some(count) => error.with(format!(
                        "见证自检 {count} 条通过（host C ABI，不含设备执行）"
                    )),
                    None => error,
                },
            },
        });
    }
    Ok(VerifyReport {
        op: op_name.to_string(),
        devices: devices.to_vec(),
        entries,
    })
}

/// 一台设备的 L2 证据：逐形状编译、执行、与参考对拍。
///
/// 形状矩阵见 `verify_shapes`：最小合法形状、多行非方形、生产形状。矩阵里的每个形状都要
/// 独立过一遍编译与对拍——只测一个形状时，行偏移（`w + n*(M/2)`）和单组边界都不会被覆盖。
///
/// # Errors
///
/// 设备不可编译、缺编译器、编译/执行失败、输出个数不符或对拍超出容差时返回说明。
pub fn verify_one(
    op: &OpDecl,
    profile: &Profile,
    settings: &Settings,
) -> diagnostic::Result<Verified> {
    let model = model_for(&op.name)?;
    if profile.meta.emit != cutile::prof::Emit::C {
        let declared: Vec<String> = profile
            .provides
            .iter()
            .map(|have| format!("{}={}", have.prim, have.grade))
            .collect();
        return Err(Error::new(
            Code::Unsupported,
            format!(
                "本机没有该设备，P0 只能真机验证；档案声明的等级：{}",
                declared.join(" ")
            ),
        ));
    }

    let path = cap::negotiate(op, profile)
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    let group = u64::from(model.group.max(1));

    // 负例也是判据：组大小约束被放宽时，这里必须先在 `verify` 里失败，而不是等到错数据。
    for (m, n) in rejected_shapes(group) {
        let shape = format!("{m}x{n}");
        if parse_shape(&shape, group).is_ok() {
            return Err(Error::new(
                Code::Internal,
                format!("形状 `{shape}` 必须被拒绝：M 必须是组大小 {group} 的整数倍"),
            ));
        }
    }

    let tool = backend::detect_toolchain().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            "缺少 C 编译器（cc/clang/gcc 都不在 PATH 上），无法产生 L2 证据",
        )
    })?;
    let tolerance = reference::tolerance(model.dtype);

    let mut max_rel_err = 0.0f64;
    let mut shapes = Vec::new();
    let mut artifacts = Vec::new();
    let mut lines = 0usize;
    let mut key = String::new();
    let mut tile = String::new();

    for (m, n) in verify_shapes(&model) {
        let cost = (model.cost)(m as usize, n as usize)?;
        let plan = plan::select(op, profile, &path, cost)?;
        // 签名与样本先对齐：ABI 不一致的错误必须在 emit 与编译之前出现。
        let sample = (model.sample)(m as usize, n as usize);
        op.signature.check(&sample)?;
        let (source, source_key) = build::source(op, profile, &path, &plan.tile, settings)?;
        let harness = harness(&source, op, &sample, 1)?;
        // 产物进缓存：命中就不调用编译器，L2 证据因此不必每次重编一遍。
        let (outcome, artifact) = build::execute(profile, &tool, &source_key, &harness, settings)?;
        if outcome.bits.len() != sample.n {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "形状 {m}x{n}：内核只输出 {} 个数，期望 {}",
                    outcome.bits.len(),
                    sample.n
                ),
            ));
        }
        let want = (model.reference)(&sample)?;
        let error = reference::max_rel_err(&outcome.bits, &want);
        if error > tolerance {
            // 对拍失败不是“环境问题”：产物不满足它自己声明的语义，类别是 `invalid`。
            return Err(Error::new(
                Code::Invalid,
                format!("形状 {m}x{n} 对拍失败：max_rel_err {error:.3e} > {tolerance:.0e}"),
            ));
        }
        max_rel_err = max_rel_err.max(error);
        lines = source.lines();
        key = source_key.hex();
        tile = plan.tile.to_string();
        shapes.push((m, n));
        artifacts.push(artifact);
    }

    Ok(Verified {
        device: String::new(),
        vendor: profile.meta.vendor.to_string(),
        id: profile.meta.id.to_string(),
        variant: path.variant,
        tile,
        shapes,
        witnesses: 0,
        max_rel_err,
        lines,
        key,
        artifacts,
    })
}

/// `verify` 的形状矩阵：最小合法、多行非方形、生产形状。
///
/// 三项全部来自算子模型：组大小（`model.group`）与生产形状（`model.shape`）都由算子声明，
/// 换一个分组策略或换一个算子时矩阵随它变，负例也不会跟着漂移。审计不写第二个方阵。
fn verify_shapes(model: &Model) -> Vec<(u64, u64)> {
    let group = u64::from(model.group.max(1));
    let (m, n) = model.shape;
    vec![(group, 1), (group * 2, 3), (m as u64, n as u64)]
}

/// 必须被拒绝的形状：`M` 不是组大小的整数倍。
fn rejected_shapes(group: u64) -> Vec<(u64, u64)> {
    if group <= 1 {
        return Vec::new();
    }
    vec![(1, group), (group + 1, 8)]
}

/// `bench`：跑一次并报告达成率（DESIGN §12.3）。
///
/// `shape` 是命令行覆盖；`None` 时用算子的生产形状（`Model::shape`），审计不持有一个缺省方阵。
///
/// # Errors
///
/// 形状非法、协商失败、缺编译器、编译/执行失败或内核没有给出计时样本时返回说明。
pub fn bench(
    catalogue: &Catalogue,
    settings: &Settings,
    op_name: &str,
    shape: Option<&str>,
    repeat: usize,
) -> diagnostic::Result<Bench> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| Error::new(Code::Missing, format!("没有算子 `{op_name}`")))?;
    let model = model_for(&op.name)?;
    let (m, n) = match shape {
        Some(text) => parse_shape(text, u64::from(model.group.max(1)))?,
        None => {
            let (m, n) = model.shape;
            (m as u64, n as u64)
        }
    };
    let profile = catalogue.resolve("auto")?;
    let path = cap::negotiate(op, profile)
        .map_err(|error| Error::new(Code::Unsupported, error.to_string()))?;
    let cost = (model.cost)(m as usize, n as usize)?;
    let plan = plan::select(op, profile, &path, cost)?;
    // 签名与样本先对齐：ABI 不一致的错误必须在 emit 与编译之前出现。
    let sample = (model.sample)(m as usize, n as usize);
    op.signature.check(&sample)?;
    let (source, key) = build::source(op, profile, &path, &plan.tile, settings)?;

    let tool = backend::detect_toolchain().ok_or_else(|| {
        Error::new(
            Code::Unsupported,
            "缺少 C 编译器（cc/clang/gcc 都不在 PATH 上）",
        )
    })?;
    let repeat = repeat.max(1);
    let harness = harness(&source, op, &sample, repeat)?;
    // 重复次数在主函数里，所以它也是产物身份的一部分：换 `--repeat` 就换一份产物。
    let (outcome, artifact) = build::execute(profile, &tool, &key, &harness, settings)?;
    let per_run_ns = outcome
        .timing
        .min_ns()
        .ok_or_else(|| Error::new(Code::Internal, "内核没有报告可用的计时样本"))?;
    // 中位数只用于报告抖动；样本缺失时退回 min，不让"没有 p50"变成第二个失败条件。
    let p50_ns = outcome.timing.p50_ns().unwrap_or(per_run_ns);
    let peak = profile.peak.bw.as_ref().map(|peak| Peak {
        value: peak.value,
        src: peak.src.clone(),
    });
    Ok(Bench {
        op: op_name.to_string(),
        shape: (m, n),
        repeat,
        measured: throughput(
            plan.bytes,
            plan.flops,
            p50_ns,
            peak.as_ref().map(|peak| peak.value),
        ),
        plan,
        per_run_ns,
        p50_ns,
        peak,
        working_set: working_set(&sample, op.signature.output().unwrap_or_default()),
        fast_bytes: profile.mem.fast_bytes,
        cache_dir: build::dir(&key, settings),
        artifact,
    })
}

/// 输入工作集：除输出外的全部缓冲（权重、scales、激活）。
fn working_set(sample: &Sample, output: &str) -> u64 {
    sample
        .buffers
        .iter()
        .filter(|buffer| buffer.name != output)
        .map(|buffer| match &buffer.values {
            Values::F32(values) => 4 * values.len() as u64,
            Values::U8(values) => values.len() as u64,
        })
        .sum()
}

/// 缓存容量上限的缺省值（DESIGN §10.3）。
pub const CACHE_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

/// `cache gc`：把缓存按 LRU 回收到上限以内；`limit` 为 `None` 时用 [`CACHE_LIMIT`]。
///
/// 删哪些条目全由缓存自己决定（时间戳与锁文件在那里），这里只决定"上限是多少"——
/// 它是命令策略，不进内核。
///
/// # Errors
///
/// 删除某个条目失败时返回带路径的说明（缓存目录不可写之类）。
pub fn cache_gc(settings: &Settings, limit: Option<u64>) -> diagnostic::Result<Gc> {
    Cache::from_settings(settings).gc(limit.unwrap_or(CACHE_LIMIT))
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

/// `bytes / ns` 直接就是 GB/s、`flops / ns` 直接就是 GFLOP/s（两者都以 1e9 为单位），
/// 但达成率必须换算回 B/s 再除以档案峰值——直接用量纲不同的 GB/s 去除会差 1e9 倍。
pub fn throughput(bytes: u64, flops: f64, per_run_ns: f64, peak_bw: Option<f64>) -> Throughput {
    if per_run_ns <= 0.0 {
        return Throughput {
            gb_s: 0.0,
            gflops: 0.0,
            bandwidth_ratio: None,
        };
    }
    let gb_s = bytes as f64 / per_run_ns;
    let bandwidth_ratio = peak_bw
        .filter(|peak| *peak > 0.0)
        .map(|peak| gb_s * 1e9 / peak);
    Throughput {
        gb_s,
        gflops: flops / per_run_ns,
        bandwidth_ratio,
    }
}

/// 拼出可执行的 `main()`：内核源码 + 样本 + 签名派生的实参 + 重复次数。
fn harness(
    source: &Source,
    op: &OpDecl,
    sample: &Sample,
    repeat: usize,
) -> diagnostic::Result<String> {
    backend::harness(&source.text, &op.signature, sample, repeat)
}

fn parse_shape(shape: &str, group: u64) -> diagnostic::Result<(u64, u64)> {
    let (m, n) = shape
        .split_once('x')
        .ok_or_else(|| Error::new(Code::Invalid, format!("`{shape}` 不是 `MxN` 形式")))?;
    let m = m
        .trim()
        .parse::<u64>()
        .map_err(|_| Error::new(Code::Invalid, format!("`{m}` 不是整数")))?;
    let n = n
        .trim()
        .parse::<u64>()
        .map_err(|_| Error::new(Code::Invalid, format!("`{n}` 不是整数")))?;
    if m == 0 || n == 0 || !m.is_multiple_of(group) {
        return Err(Error::new(
            Code::Invalid,
            format!("形状 {shape} 非法：M 必须是 {group} 的正整数倍"),
        ));
    }
    Ok((m, n))
}

/// 取算子的语义模型（代价、样本、标量参考）。声明层新增了算子却没有模型时，
/// `doctor`/`verify`/`bench` 都无法给出计划或对拍结果。
fn model_for(name: &str) -> diagnostic::Result<Model> {
    op::model(name).ok_or_else(|| {
        Error::new(
            Code::Missing,
            format!(
                "算子 `{name}` 没有语义模型：`crates/op/src/lib.rs` 的 `model()` 注册表里没有它"
            ),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use prof::atom::Prim;

    fn settings() -> Settings {
        let roots = llm_config::Roots::new(std::env::temp_dir().join("llm-audit-tests"));
        Settings::<llm_config::Cutile>::defaults(&roots)
    }

    fn catalogue() -> Catalogue {
        Catalogue::try_builtin().expect("内置数据")
    }

    #[test]
    fn doctor_reports_the_full_chain() {
        let report = doctor(&catalogue(), &settings(), "gemv_i4", "cpu").expect("doctor 应成功");
        assert_eq!(report.path.as_ref().expect("有路径").variant, "pipe2");
        assert!(!report.layouts.is_empty(), "{report:?}");
        let source = report.source.as_ref().expect("有 emit 产物");
        assert!(source.lines > 0);
        let hex = source.key.hex();
        assert_eq!(hex.len(), 16, "缓存键是 16 位十六进制");
        assert!(hex.chars().all(|digit| digit.is_ascii_hexdigit()));
    }

    /// 独立缓存目录：GC 会真删东西，不能与其他用例共用。
    fn cache_settings(name: &str) -> Settings {
        let root = std::env::temp_dir().join(format!("llm-audit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut settings = Settings::<llm_config::Cutile>::defaults(&llm_config::Roots::new(&root));
        settings
            .apply_override(&format!("cache.dir={}", root.join("cache").display()))
            .expect("cache.dir 覆盖合法");
        settings
    }

    #[test]
    fn cache_gc_reclaims_through_the_settings_directory() {
        let settings = cache_settings("cache-gc");
        let report = doctor(&catalogue(), &settings, "gemv_i4", "cpu").expect("doctor 应成功");
        assert!(report.source.is_some(), "doctor 应已落盘");
        let before = cache_gc(&settings, None).expect("可回收");
        assert_eq!(before.limit, CACHE_LIMIT, "缺省上限是命令策略");
        assert!(before.before > 0, "doctor 刚落过盘：{before:?}");
        assert_eq!(before.removed, 0, "未超上限时不删：{before:?}");

        let after = cache_gc(&settings, Some(0)).expect("可回收");
        assert_eq!(after.removed, 1, "{after:?}");
        assert_eq!(after.after, 0, "{after:?}");
        let _ = std::fs::remove_dir_all(settings.cache_dir());
    }

    #[test]
    fn why_separates_half_written_from_never_written() {
        // 半截落盘的目录没有 `source`，逐项差异自然为空；按差异判会把它说成“一致”。
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

    #[test]
    fn doctor_reports_each_variant_as_data() {
        // 报告里只有判定结果：拼成哪一行、标记怎么放由前端决定。
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

    #[test]
    fn verify_runs_witnesses_and_the_whole_shape_matrix() {
        if backend::detect_toolchain().is_none() {
            return;
        }
        let devices = vec!["cpu".to_string()];
        let report = verify(&catalogue(), &settings(), "gemv_i4", &devices).expect("算子存在");
        let verified = match &report.entries[0] {
            VerifyEntry::Passed(verified) => verified,
            VerifyEntry::Failed { error, .. } => panic!("验证失败：{error}"),
        };
        assert_eq!(
            verified.shapes,
            vec![(32, 1), (64, 3), (4096, 4096)],
            "形状矩阵必须全部跑到"
        );
        assert_eq!(
            verified.witnesses,
            catalogue()
                .profile("cpu")
                .expect("cpu 档案")
                .provides
                .iter()
                .filter(|provide| provide.grade.is_verified())
                .count(),
            "每条 L2 供给都要有自己的见证结果"
        );
    }

    /// 产物也进缓存：第二次 `verify` 不该再调用编译器。
    #[test]
    fn verify_reuses_its_artifacts_on_the_second_run() {
        if backend::detect_toolchain().is_none() {
            return;
        }
        let settings = cache_settings("verify-artifacts");
        let devices = vec!["cpu".to_string()];
        let first = verify(&catalogue(), &settings, "gemv_i4", &devices).expect("算子存在");
        let first = match &first.entries[0] {
            VerifyEntry::Passed(verified) => verified,
            VerifyEntry::Failed { error, .. } => panic!("验证失败：{error}"),
        };
        assert_eq!(
            first.artifacts.len(),
            3,
            "形状矩阵的每一项都有自己的产物：{:?}",
            first.artifacts
        );
        assert!(
            first.artifacts.iter().all(|artifact| !artifact.reused),
            "空缓存里第一次必须全部现编：{:?}",
            first.artifacts
        );

        let second = verify(&catalogue(), &settings, "gemv_i4", &devices).expect("算子存在");
        let second = match &second.entries[0] {
            VerifyEntry::Passed(verified) => verified,
            VerifyEntry::Failed { error, .. } => panic!("验证失败：{error}"),
        };
        assert!(
            second.artifacts.iter().all(|artifact| artifact.reused),
            "第二次必须全部命中产物（命中即不调用编译器）：{:?}",
            second.artifacts
        );
        assert_eq!(second.max_rel_err, first.max_rel_err, "复用的产物结论不变");
        let _ = std::fs::remove_dir_all(settings.cache_dir());
    }

    /// 本机没有的加速器：见证自检先跑，缺设备是最后一句而不是唯一一句。
    #[test]
    fn verify_reports_the_witness_self_check_for_a_device_that_is_not_here() {
        if backend::detect_toolchain().is_none() {
            return;
        }
        let devices = vec!["nvidia".to_string()];
        let report = verify(&catalogue(), &settings(), "gemv_i4", &devices).expect("算子存在");
        let error = match &report.entries[0] {
            VerifyEntry::Failed { error, .. } => error,
            VerifyEntry::Passed(verified) => panic!("本机没有 nvidia，不能通过：{verified:?}"),
        };
        let declared = catalogue()
            .profile("nvidia")
            .expect("nvidia 档案")
            .provides
            .iter()
            .filter(|provide| provide.witness.is_some())
            .count();
        assert!(
            error
                .to_string()
                .contains(&format!("见证自检 {declared} 条通过")),
            "{error}"
        );
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("本机没有该设备"), "{error}");
    }

    /// 把三层数据复制到临时 `data.root`，再按需要改一份档案。
    fn data_root(name: &str, patch: impl Fn(&std::path::Path)) -> std::path::PathBuf {
        fn copy(from: &std::path::Path, to: &std::path::Path) {
            std::fs::create_dir_all(to).expect("建目录");
            for entry in std::fs::read_dir(from).expect("读目录") {
                let entry = entry.expect("目录项");
                let target = to.join(entry.file_name());
                if entry.path().is_dir() {
                    copy(&entry.path(), &target);
                } else {
                    std::fs::copy(entry.path(), &target).expect("复制文件");
                }
            }
        }
        let root = std::env::temp_dir().join(format!("cutile-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        copy(&op::dir(), &root.join("ops"));
        copy(&prof::dir(), &root.join("devices"));
        copy(&prim::dir(), &root.join("prim"));
        patch(&root);
        root
    }

    fn with_data_root(root: &std::path::Path) -> Settings {
        let mut settings = settings();
        settings
            .apply_override(&format!("data.root={}", root.display()))
            .expect("data.root 覆盖合法");
        settings
    }

    #[test]
    fn a_stale_stage_declaration_fails_verify() {
        // `[needs].load_async` 要 `stages = ["2"]`，两个变体的模板都调用 `@load_async`；
        // 把档案改成只提供 7 级后，协商必须失败，而不是"降级到 direct 就当作通过"。
        let root = data_root("verify-stages", |root| {
            let path = root.join("devices").join("cpu.toml");
            let text = std::fs::read_to_string(&path)
                .expect("读 cpu.toml")
                .replace("stages  = [\"2\", \"3\"]", "stages  = [\"7\"]");
            std::fs::write(&path, text).expect("写 cpu.toml");
        });
        let settings = with_data_root(&root);
        let catalogue = Catalogue::load(&settings).expect("临时数据可装载");
        let report =
            verify(&catalogue, &settings, "gemv_i4", &["cpu".to_string()]).expect("算子存在");
        let failures = report.failures();
        assert!(!failures.is_empty(), "{report:?}");
        assert!(
            failures
                .iter()
                .any(|error| error.to_string().contains("stages")),
            "{failures:?}"
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn verify_fails_when_the_device_is_absent() {
        let devices = vec!["nvidia".to_string()];
        let report = verify(&catalogue(), &settings(), "gemv_i4", &devices).expect("算子存在");
        assert_eq!(report.failures().len(), 1, "{report:?}");
        assert!(report.failures()[0].to_string().contains("只能真机验证"));
    }

    #[test]
    fn bandwidth_attainment_is_a_byte_ratio_not_a_gigabyte_one() {
        // 0.7 MB / 1.467 ms ≈ 0.48 GB/s，占 68 GB/s 的 0.7%；除以 GB/s 会算出 0.0%。
        let measured = throughput(700_000, 2.1e6, 1.467e6, Some(68.0e9));
        assert!((measured.gb_s - 0.477).abs() < 0.01, "{measured:?}");
        assert!((measured.gflops - 1.43).abs() < 0.01, "{measured:?}");
        assert!((measured.bandwidth_ratio.expect("有峰值") - 0.0070).abs() < 1e-4);
        assert_eq!(throughput(1, 1.0, 0.0, Some(1.0)).bandwidth_ratio, None);
    }

    #[test]
    fn an_attainment_above_one_is_an_invalid_measurement_not_a_pass() {
        // 数据驻留在缓存里时 `bytes/ns` 会超过档案的主存峰值：这不是"超过 100% 的实现质量"。
        assert_eq!(classify(Some(1.42), false), Verdict::OverPeak);
        // 装得进缓存时，分母（主存峰值）本来就不是这次测量的口径。
        assert_eq!(classify(Some(0.19), true), Verdict::CacheResident);
        assert_eq!(classify(Some(1.0), false), Verdict::Reached);
        assert_eq!(classify(Some(0.7), false), Verdict::Reached);
        assert_eq!(classify(Some(0.69), false), Verdict::Below);
        assert_eq!(classify(None, false), Verdict::Unknown);
    }

    #[test]
    fn working_set_is_every_input_buffer_but_not_the_output() {
        let sample = Sample {
            m: 1,
            n: 2,
            scalars: Vec::new(),
            buffers: vec![
                op::model::Buffer {
                    name: "w",
                    values: Values::U8(vec![0; 8]),
                },
                op::model::Buffer {
                    name: "x",
                    values: Values::F32(vec![1.0; 4]),
                },
                op::model::Buffer {
                    name: "y",
                    values: Values::F32(vec![0.0; 64]),
                },
            ],
        };
        assert_eq!(working_set(&sample, "y"), 8 + 16);
    }

    #[test]
    fn storage_reports_the_archive_capacity_next_to_the_declared_traffic() {
        let report = doctor(&catalogue(), &settings(), "gemv_i4", "cpu").expect("doctor 应成功");
        assert_eq!(report.storage.len(), 3, "三层都要报：{report:?}");
        let fast = report
            .storage
            .iter()
            .find(|row| row.space == Space::Fast)
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
            .find(|row| row.space == Space::Reg)
            .expect("有 REG 行");
        assert_eq!(reg.capacity, Some(65_536));
        assert!(
            reg.declared.is_empty(),
            "没有声明字节数的原语不占 REG：{reg:?}"
        );
        let gm = report
            .storage
            .iter()
            .find(|row| row.space == Space::Gm)
            .expect("有 GM 行");
        assert_eq!(gm.capacity, None, "GM 不受限额，不编一个容量");
    }

    #[test]
    fn the_planned_shape_comes_from_the_operator_model() {
        let model = model_for("gemv_i4").expect("有模型");
        let report = doctor(&catalogue(), &settings(), "gemv_i4", "cpu").expect("doctor 应成功");
        assert_eq!(report.shape, model.shape);
        assert!(
            report.layouts.iter().all(|row| row.spec == "row"),
            "{report:?}"
        );
    }

    #[test]
    fn the_shape_matrix_follows_the_operator_model() {
        let model = model_for("gemv_i4").expect("有模型");
        assert_eq!(verify_shapes(&model), vec![(32, 1), (64, 3), (4096, 4096)]);
        // 换一个生产形状，矩阵跟着变：审计里没有第二个方阵。
        let other = Model {
            shape: (128, 64),
            ..model
        };
        assert_eq!(verify_shapes(&other), vec![(32, 1), (64, 3), (128, 64)]);
    }

    #[test]
    fn shape_validation_matches_the_operator_constraints() {
        let group = u64::from(model_for("gemv_i4").expect("有模型").group);
        assert_eq!(parse_shape("4096x4096", group).unwrap(), (4096, 4096));
        assert!(parse_shape("4096", group).is_err());
        assert!(parse_shape("1000x8", group).is_err());
        assert!(parse_shape("0x8", group).is_err());
    }

    #[test]
    fn the_model_group_matches_the_declaration() {
        // 声明里的 `needs.quant.group` 是能力请求，模型里的 `group` 是样本与代价的取值；
        // 两者漂移会让 verify 在一组原语支撑不了的组大小上"通过"。
        let catalogue = catalogue();
        let op = catalogue.op("gemv_i4").expect("内置算子");
        let declared = op
            .needs
            .iter()
            .filter(|need| need.prim == Prim::Quant)
            .flat_map(|need| need.attrs.iter())
            .find(|attr| attr.key() == prof::atom::AttrKey::Group)
            .and_then(|attr| attr.values().first())
            .expect("声明里有 quant.group")
            .parse::<u32>()
            .expect("组大小是整数");
        assert_eq!(declared, model_for("gemv_i4").expect("有模型").group);
    }
}

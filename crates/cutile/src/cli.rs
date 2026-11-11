//! `doctor` / `verify` / `bench`（DESIGN §16 的 P0 交付物）。
//!
//! 三条命令共用同一条流水线：声明层 → 协商 → tile 选择 → 片段拼接 → emit → 编译执行。
//! 任何一步失败都带上下文退出，不写"跳过"（DESIGN §11.2 对旧工程的修正）。

use crate::SCHEMA_VER;
use crate::atom::{DType, Prim};
use crate::backend::{self, GemvData};
use crate::builtin::Catalogue;
use crate::cache::{Cache, CacheKey};
use crate::cap::{self, Path};
use crate::layout::LayoutSpec;
use crate::lower::substitute::{self, FragmentSet};
use crate::lower::{Source, emit};
use crate::plan;
use crate::probe;
use crate::prof::{Emit, Profile};
use crate::reference;
use crate::space;
use std::path::PathBuf;

/// 可执行的子命令。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// 打印协商链、布局契约、emit 结果与缓存指纹。
    Doctor {
        /// 算子名。
        op: String,
        /// 设备名（`auto` 表示按指纹选）。
        device: String,
    },
    /// 逐设备验证：协商 + 拼接 + 编译 + 与参考对拍。
    Verify {
        /// 算子名。
        op: String,
        /// 设备列表。
        devices: Vec<String>,
        /// 是否把台账写到 `devices/<vendor>.capability.md`。
        ledger: bool,
    },
    /// 跑一次并报告达成率。
    Bench {
        /// 算子名。
        op: String,
        /// 形状 `MxN`。
        shape: String,
        /// 重复次数。
        repeat: usize,
    },
}

/// 解析命令行。
pub fn parse(args: &[String]) -> Result<Command, String> {
    let (head, rest) = args.split_first().ok_or_else(usage)?;
    let mut op = None;
    let mut device = "auto".to_string();
    let mut devices = Vec::new();
    let mut shape = "4096x4096".to_string();
    let mut repeat = 50usize;
    let mut ledger = false;

    let mut index = 0;
    while index < rest.len() {
        let arg = rest[index].as_str();
        let mut take = || -> Result<String, String> {
            index += 1;
            rest.get(index)
                .cloned()
                .ok_or_else(|| format!("`{arg}` 缺少取值"))
        };
        match arg {
            "--op" => op = Some(take()?),
            "--device" => device = take()?,
            "--devices" => devices = take()?.split(',').map(str::to_string).collect(),
            "--shape" => shape = take()?,
            "--repeat" => {
                repeat = take()?
                    .parse::<usize>()
                    .map_err(|_| "`--repeat` 需要整数")?
            }
            "--emit-ledger" => ledger = true,
            other => return Err(format!("未知参数 `{other}`\n{}", usage())),
        }
        index += 1;
    }

    let op = op.unwrap_or_else(|| "gemv_i4".to_string());
    match head.as_str() {
        "doctor" => Ok(Command::Doctor { op, device }),
        "verify" => Ok(Command::Verify {
            op,
            devices: if devices.is_empty() {
                vec!["cpu".to_string()]
            } else {
                devices
            },
            ledger,
        }),
        "bench" => Ok(Command::Bench { op, shape, repeat }),
        other => Err(format!("未知子命令 `{other}`\n{}", usage())),
    }
}

/// 用法。
pub fn usage() -> String {
    "用法：\n  cutile doctor --op gemv_i4 --device auto\n  cutile verify --op gemv_i4 --devices cpu[,cuda] [--emit-ledger]\n  cutile bench  --op gemv_i4 --shape 4096x4096 --repeat 50".to_string()
}

/// 命令行入口；返回进程退出码。
pub fn main(args: &[String]) -> i32 {
    match parse(args).and_then(|command| run(&command)) {
        Ok(report) => {
            println!("{report}");
            0
        }
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
}

/// 执行一个子命令，返回可打印的报告；失败即 `Err`（退出码 1）。
pub fn run(command: &Command) -> Result<String, String> {
    let catalogue = Catalogue::try_builtin()?;
    catalogue
        .validate(Some(&Catalogue::root()))
        .map_err(|errors| errors.join("\n"))?;

    match command {
        Command::Doctor { op, device } => doctor(&catalogue, op, device),
        Command::Verify {
            op,
            devices,
            ledger,
        } => verify(&catalogue, op, devices, *ledger),
        Command::Bench { op, shape, repeat } => bench(&catalogue, op, shape, *repeat),
    }
}

fn doctor(catalogue: &Catalogue, op_name: &str, device: &str) -> Result<String, String> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| format!("没有算子 `{op_name}`"))?;
    let profile = resolve(catalogue, device)?;

    let mut out = String::new();
    out.push_str(&format!(
        "cutile schema v{SCHEMA_VER} | op={} | skeleton={} | {}\n",
        op.name, op.skeleton, op.summary
    ));
    out.push_str(&format!(
        "device={}({}) emit={} tool={} {} | spaces GM={} FAST={} REG={}\n",
        profile.meta.vendor,
        profile.meta.id,
        profile.meta.emit,
        profile.meta.tool,
        profile.meta.tool_ver,
        space::physical(profile, crate::atom::Space::Gm),
        space::physical(profile, crate::atom::Space::Fast),
        space::physical(profile, crate::atom::Space::Reg)
    ));
    out.push_str(&format!(
        "needs={} variants={} tiles={}\n",
        op.needs.len(),
        op.variants.len(),
        crate::lower::expand::expand(op)
            .map_err(|error| error.to_string())?
            .len()
    ));

    out.push_str("\n[协商]\n");
    out.push_str(&cap::explain(op, profile));

    let path = match cap::negotiate(op, profile) {
        Ok(path) => path,
        Err(_) => {
            out.push_str("\n[emit] 无可行变体，未生成源文件\n");
            return Ok(out);
        }
    };
    let plan = plan::select(op, profile, &path, 4096, 4096, 32)?;
    out.push_str(&format!("\n[plan] {plan}\n"));

    out.push_str("\n[布局契约]\n");
    for (slot, spec) in &op.layouts {
        let normal = spec.instantiate(plan.tile.values()[0].1, 64);
        out.push_str(&format!("  {slot} = {spec} → {normal}\n"));
        if let LayoutSpec::Swz { b, .. } = spec
            && let Err(error) = space::check_alignment(profile, *b)
        {
            out.push_str(&format!("    ✗ {error}\n"));
        }
    }

    out.push_str("\n[存储]\n");
    out.push_str(&format!(
        "  {}\n",
        space::describe(
            profile,
            space::SpaceUse::new(
                crate::atom::Space::Fast,
                u64::from(plan.tile.values()[0].1) * 4
            )
        )
    ));

    let (source, key) = build_source(op, profile, &path, &plan.tile)?;
    out.push_str(&format!(
        "\n[emit] {} 行，{} | cache key {}\n",
        source.lines(),
        source.ext(),
        key.hex()
    ));
    out.push_str(&format!("  {}\n", key.describe()));

    if let Some(cache) = cache_dir(&key) {
        out.push_str(&format!("  cache dir {}\n", cache.display()));
    }
    Ok(out)
}

fn verify(
    catalogue: &Catalogue,
    op_name: &str,
    devices: &[String],
    ledger: bool,
) -> Result<String, String> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| format!("没有算子 `{op_name}`"))?;
    let mut report = format!("verify {op_name} @ [{}]\n", devices.join(","));
    let mut failures = Vec::new();

    for device in devices {
        let profile = match resolve(catalogue, device) {
            Ok(profile) => profile,
            Err(error) => {
                report.push_str(&format!("  {device}: ✗ {error}\n"));
                failures.push(error);
                continue;
            }
        };
        match verify_one(op, profile) {
            Ok(line) => {
                report.push_str(&line);
                if ledger {
                    report.push_str(&write_ledger(profile, &line)?);
                }
            }
            Err(error) => {
                report.push_str(&format!(
                    "  {}({}): ✗ {error}\n",
                    profile.meta.vendor, profile.meta.id
                ));
                failures.push(error);
            }
        }
    }

    if failures.is_empty() {
        Ok(report)
    } else {
        Err(format!(
            "{report}\n缺设备即失败（DESIGN §11.2）：{} 项未通过",
            failures.len()
        ))
    }
}

fn verify_one(op: &crate::lower::OpDecl, profile: &Profile) -> Result<String, String> {
    if profile.meta.emit != Emit::C {
        let declared: Vec<String> = profile
            .provides
            .iter()
            .map(|have| format!("{}={}", have.prim, have.grade))
            .collect();
        return Err(format!(
            "本机没有该设备，P0 只能真机验证；档案声明的等级：{}",
            declared.join(" ")
        ));
    }

    let path = cap::negotiate(op, profile).map_err(|error| error.render())?;
    let plan = plan::select(op, profile, &path, 4096, 4096, 32)?;
    let (source, key) = build_source(op, profile, &path, &plan.tile)?;

    let tool = backend::detect_toolchain().ok_or_else(|| {
        "缺少 C 编译器（cc/clang/gcc 都不在 PATH 上），无法产生 L2 证据".to_string()
    })?;
    let data = sample_data(4096, 4096, 32);
    let kernel = format!("cutile_{}_kernel", op.name);
    let harness = backend::harness(&source.text, &kernel, &data, 1);

    let workdir = std::env::temp_dir().join(format!(
        "cutile-verify-{}-{}",
        profile.meta.vendor,
        std::process::id()
    ));
    let outcome = backend::compile_and_run(&tool, &harness, &workdir)?;
    if outcome.bits.len() != data.n {
        return Err(format!(
            "内核只输出 {} 个数，期望 {}",
            outcome.bits.len(),
            data.n
        ));
    }
    let want = reference::gemv_i4(&data.x, &data.w, &data.scales, data.m, data.n, data.group)?;
    let error = reference::max_rel_err(&outcome.bits, &want);
    let tolerance = reference::tolerance(DType::F32);
    if error > tolerance {
        return Err(format!(
            "对拍失败：max_rel_err {error:.3e} > {tolerance:.0e}"
        ));
    }
    Ok(format!(
        "  {}({}): ✓ L2 | variant {} | tile {} | max_rel_err {:.3e} | source {} 行 | key {}\n",
        profile.meta.vendor,
        profile.meta.id,
        path.variant,
        plan.tile,
        error,
        source.lines(),
        key.hex()
    ))
}

fn write_ledger(profile: &Profile, line: &str) -> Result<String, String> {
    let path = PathBuf::from(format!(
        "{}/devices/{}.capability.md",
        env!("CARGO_MANIFEST_DIR"),
        profile.meta.vendor
    ));
    let mut text = format!(
        "# {} 能力台账（由 `cutile verify --emit-ledger` 生成）\n\n",
        profile.meta.vendor
    );
    text.push_str(&format!(
        "- 档案：`{}` ({})\n",
        profile.meta.id, profile.meta.checked
    ));
    text.push_str(&format!("- 结论：{}", line.trim()));
    text.push_str("\n\n| 原语 | 实现 | 等级 | 见证 |\n|---|---|---|---|\n");
    for have in &profile.provides {
        text.push_str(&format!(
            "| `{}` | {} | {} | {} |\n",
            have.prim,
            have.impl_name,
            have.grade,
            have.witness.as_deref().unwrap_or("—")
        ));
    }
    std::fs::write(&path, text).map_err(|error| format!("写 {} 失败：{error}", path.display()))?;
    Ok(format!("    ledger → {}\n", path.display()))
}

fn bench(
    catalogue: &Catalogue,
    op_name: &str,
    shape: &str,
    repeat: usize,
) -> Result<String, String> {
    let op = catalogue
        .op(op_name)
        .ok_or_else(|| format!("没有算子 `{op_name}`"))?;
    let (m, n) = parse_shape(shape)?;
    let profile = resolve(catalogue, "auto")?;
    let path = cap::negotiate(op, profile).map_err(|error| error.render())?;
    let plan = plan::select(op, profile, &path, m, n, 32)?;
    let (source, _key) = build_source(op, profile, &path, &plan.tile)?;

    let tool = backend::detect_toolchain()
        .ok_or_else(|| "缺少 C 编译器（cc/clang/gcc 都不在 PATH 上）".to_string())?;
    let data = sample_data(m as usize, n as usize, 32);
    let kernel = format!("cutile_{}_kernel", op.name);
    let harness = backend::harness(&source.text, &kernel, &data, repeat.max(1));
    let workdir = std::env::temp_dir().join(format!("cutile-bench-{}", std::process::id()));
    let outcome = backend::compile_and_run(&tool, &harness, &workdir)?;
    let ns = outcome.ns.ok_or_else(|| "内核没有报告耗时".to_string())?;
    let per_run = ns / repeat.max(1) as f64;
    let bytes = plan.bytes as f64;
    let ach_bw = bytes / per_run * 1e9 / 1e9;
    let ach_gflops = plan.flops / per_run * 1e9 / 1e9;

    let mut out = format!("bench {op_name} {m}x{n} repeat={repeat}\n");
    out.push_str(&format!("  plan  {plan}\n"));
    out.push_str(&format!(
        "  实测  {:.3} ms/次 → {:.1} GB/s, {:.1} GFLOP/s\n",
        per_run / 1e6,
        ach_bw,
        ach_gflops
    ));
    if let Some(peak) = profile.peak.bw.as_ref() {
        let ratio = bytes / per_run / peak.value;
        out.push_str(&format!(
            "  带宽达成率 {:.1}%（峰值 {:.1} GB/s，口径：{}）\n",
            ratio * 100.0,
            peak.value / 1e9,
            peak.src
        ));
        if ratio < 0.7 {
            out.push_str(
                "  ⚠ 低于 L3 门槛（70%）：P0 的 CPU 片段是标量顺序实现，未做向量化/分块\n",
            );
        }
    }
    out.push_str(&format!(
        "  缓存  {}\n",
        cache_dir(&_key)
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "-".to_string())
    ));
    Ok(out)
}

fn parse_shape(shape: &str) -> Result<(u64, u64), String> {
    let (m, n) = shape
        .split_once('x')
        .ok_or_else(|| format!("`{shape}` 不是 `MxN` 形式"))?;
    let m = m
        .trim()
        .parse::<u64>()
        .map_err(|_| format!("`{m}` 不是整数"))?;
    let n = n
        .trim()
        .parse::<u64>()
        .map_err(|_| format!("`{n}` 不是整数"))?;
    if m == 0 || n == 0 || m % 32 != 0 || m % 2 != 0 {
        return Err(format!("形状 {shape} 非法：M 必须是 32 的正整数倍"));
    }
    Ok((m, n))
}

fn resolve<'a>(catalogue: &'a Catalogue, device: &str) -> Result<&'a Profile, String> {
    if device == "auto" {
        return catalogue.select(&probe::detect_cpu());
    }
    catalogue
        .profile(device)
        .ok_or_else(|| format!("没有设备档案 `{device}`"))
}

fn fragments_for(profile: &Profile) -> Result<FragmentSet, String> {
    if profile.meta.vendor == "cpu" {
        return Ok(FragmentSet::builtin_cpu());
    }
    FragmentSet::from_dir(
        &Catalogue::root(),
        &profile.meta.vendor,
        profile.meta.emit.ext(),
    )
    .map_err(|error| format!("`{}` 的片段层缺失：{error}", profile.meta.vendor))
}

fn build_source(
    op: &crate::lower::OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &crate::lower::TileCombo,
) -> Result<(Source, CacheKey), String> {
    let key = cache_key(op, profile, path, tile);
    let cache = Cache::from_env();
    if let Some(text) = cache.get(&key) {
        return Ok((
            Source {
                lang: profile.meta.emit,
                text,
            },
            key,
        ));
    }

    let _lock = cache.lock(&key).ok();
    let fragments = fragments_for(profile)?;
    let variant = op
        .variant(&path.variant)
        .ok_or_else(|| format!("变体 `{}` 不存在", path.variant))?;
    let rendered =
        substitute::render(op, variant, tile, &fragments).map_err(|error| format!("{error}"))?;
    let source = emit::assemble(profile, op, variant, tile, &rendered.text);
    let _ = cache.put(&key, &source.text);
    Ok((source, key))
}

fn cache_key(
    op: &crate::lower::OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &crate::lower::TileCombo,
) -> CacheKey {
    let mut flags = profile.flags.join(" ");
    flags.push_str(&format!("|{}", profile.preamble));
    let layouts: Vec<(String, String)> = op
        .layouts
        .iter()
        .map(|(slot, spec)| {
            let d0 = tile.values().first().map_or(1, |(_, value)| *value);
            (slot.clone(), spec.instantiate(d0, 64).summary())
        })
        .collect();
    CacheKey {
        schema_ver: SCHEMA_VER,
        emit: profile.meta.emit,
        tool: profile.meta.tool.clone(),
        tool_ver: profile.meta.tool_ver.clone(),
        flags_hash: emit::fnv1a(&flags),
        arch: profile.meta.id.clone(),
        template_hash: emit::fnv1a(&op.template),
        fragments: fragments_for(profile)
            .map(|set| {
                Prim::ALL
                    .into_iter()
                    .filter_map(|prim| set.get(prim).map(|text| (prim, emit::fnv1a(text))))
                    .collect()
            })
            .unwrap_or_default(),
        layouts,
        variant: path.variant.clone(),
        tile: tile.summary(),
        dtype: "f32".to_string(),
    }
}

fn cache_dir(key: &CacheKey) -> Option<PathBuf> {
    let cache = Cache::from_env();
    let dir = cache.dir_for(key);
    dir.is_dir().then_some(dir)
}

/// 确定性样本数据：同一形状永远得到同一组输入，便于复现（DESIGN §12.3 的"可复现"要求）。
pub fn sample_data(m: usize, n: usize, group: usize) -> GemvData {
    let x: Vec<f32> = (0..m)
        .map(|index| ((index * 7) % 13) as f32 * 0.25 - 1.0)
        .collect();
    let w: Vec<u8> = (0..n * (m / 2))
        .map(|index| ((index * 37) % 251) as u8)
        .collect();
    let groups = m / group.max(1);
    let scales: Vec<f32> = (0..n * groups)
        .map(|index| 0.031_25 * (1 + (index % 4)) as f32)
        .collect();
    GemvData {
        m,
        n,
        group,
        x,
        w,
        scales,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|item| item.to_string()).collect()
    }

    #[test]
    fn parses_each_subcommand() {
        assert_eq!(
            parse(&args(&["doctor", "--op", "gemv_i4", "--device", "cpu"])).unwrap(),
            Command::Doctor {
                op: "gemv_i4".to_string(),
                device: "cpu".to_string()
            }
        );
        assert_eq!(
            parse(&args(&["verify", "--devices", "cpu,cuda"])).unwrap(),
            Command::Verify {
                op: "gemv_i4".to_string(),
                devices: vec!["cpu".to_string(), "cuda".to_string()],
                ledger: false
            }
        );
        assert_eq!(
            parse(&args(&["bench", "--shape", "1024x64", "--repeat", "3"])).unwrap(),
            Command::Bench {
                op: "gemv_i4".to_string(),
                shape: "1024x64".to_string(),
                repeat: 3
            }
        );
    }

    #[test]
    fn unknown_arguments_are_rejected() {
        let error = parse(&args(&["doctor", "--nope"])).unwrap_err();
        assert!(error.contains("未知参数"), "{error}");
        let error = parse(&args(&["nope"])).unwrap_err();
        assert!(error.contains("未知子命令"), "{error}");
        assert!(parse(&[]).is_err());
    }

    #[test]
    fn shape_validation_matches_the_operator_constraints() {
        assert_eq!(parse_shape("4096x4096").unwrap(), (4096, 4096));
        assert!(parse_shape("4096").is_err());
        assert!(parse_shape("1000x8").is_err());
        assert!(parse_shape("0x8").is_err());
    }

    #[test]
    fn doctor_reports_the_full_chain() {
        let report = run(&Command::Doctor {
            op: "gemv_i4".to_string(),
            device: "cpu".to_string(),
        })
        .expect("doctor 应成功");
        assert!(report.contains("path: pipe2"), "{report}");
        assert!(report.contains("[布局契约]"), "{report}");
        assert!(report.contains("[emit]"), "{report}");
        assert!(report.contains("cache key"), "{report}");
    }

    #[test]
    fn doctor_explains_why_a_gpu_is_rejected() {
        let report = run(&Command::Doctor {
            op: "gemv_i4".to_string(),
            device: "nvidia".to_string(),
        })
        .expect("doctor 仍应打印诊断");
        assert!(report.contains("没有可行变体"), "{report}");
        assert!(report.contains("L2"), "{report}");
    }

    #[test]
    fn verify_fails_when_the_device_is_absent() {
        let error = run(&Command::Verify {
            op: "gemv_i4".to_string(),
            devices: vec!["nvidia".to_string()],
            ledger: false,
        })
        .unwrap_err();
        assert!(error.contains("缺设备即失败"), "{error}");
    }

    #[test]
    fn sample_data_is_deterministic() {
        let first = sample_data(64, 4, 32);
        let second = sample_data(64, 4, 32);
        assert_eq!(first, second);
        assert_eq!(first.w.len(), 4 * 32);
        assert_eq!(first.scales.len(), 4 * 2);
    }
}

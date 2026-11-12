//! ⑤ emit + cache：指纹 → 查缓存 → 取锁 → 拼接 → emit → 落盘（DESIGN §10.3）。
//!
//! 输入全是数据：算子声明、协商出的变体、tile 取值与配置；输出是源文本与它的缓存键。
//! 键覆盖真正喂给 `cc` 的参数、片段文本、布局声明式与声明的数据类型集合，因此"命中缓存"
//! 与"重新生成"必然等价。
//!
//! 本模块不认识任何命令：审计（`llm-audit`）与将来的落地流程都从 [`source`] 取文本，
//! 命令策略（形状矩阵、重复次数、工作目录命名）不在内核里。

use crate::SCHEMA_VER;
use crate::atom::{DType, DTypeSet, Prim, Space};
use crate::backend::{self, Outcome};
use crate::cache::{Artifact, ArtifactKey, Cache, CacheKey, Lock};
use crate::cap::Path;
use crate::decl::{OpDecl, TileCombo};
use crate::emit::{self, Source};
use crate::layout::LayoutSpec;
use crate::prof::Profile;
use crate::space;
use crate::substitute::{self, FragmentSet};
use diagnostic::{Code, Error};
use llm_config::Settings;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// ⑤ 的完整流程：指纹 → 查缓存 → 取锁 → 拼接 → emit → 落盘。
///
/// 缓存命中时直接返回已落盘的文本——键相同意味着源文本相同，重新拼一遍不会得到别的东西。
///
/// # Errors
///
/// 片段层缺失、变体不存在、构建期契约（容量/对齐）不通过、取锁失败或落盘失败时返回说明。
pub fn source(
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &TileCombo,
    settings: &Settings,
) -> diagnostic::Result<(Source, CacheKey)> {
    // 契约先于缓存：声明里的搬运量只影响判定、不影响源文本，所以它不在缓存键里。
    // 如果把判定放在命中之后，一份已落盘的旧源文本就成了“声明越界也免检”的通行证。
    contract(op, profile, path)?;
    let key = key(op, profile, path, tile, settings);
    let cache = Cache::from_settings(settings);
    if let Some(text) = cache.get(&key) {
        return Ok((
            Source {
                lang: profile.meta.emit,
                text,
            },
            key,
        ));
    }

    let fragments = fragments(profile, settings)?;
    let variant = op
        .variant(&path.variant)
        .ok_or_else(|| Error::new(Code::Invalid, format!("变体 `{}` 不存在", path.variant)))?;

    // 锁必须真的拿到：静默继续会让两个进程同时写同一份 `source`（DESIGN §10.3）。
    // 但"另一个进程正在为同一个 key 编译"不是错误——它写出的就是我们要的文本，等它落盘
    // 即可（并行的 `doctor`/`verify` 对同一个 key 是常态）；等不到才报错。
    let lock = wait_for_lock(&cache, &key)?;
    if let Some(text) = cache.get(&key) {
        return Ok((
            Source {
                lang: profile.meta.emit,
                text,
            },
            key,
        ));
    }
    let text = substitute::render(op, tile, &fragments)?;
    let source = emit::assemble(profile, op, variant, tile, &text);
    cache.put(&key, &source.text).map_err(|error| {
        Error::new(
            Code::Io,
            format!("写缓存 {} 失败：{error}", cache.dir_for(&key).display()),
        )
    })?;
    drop(lock);
    Ok((source, key))
}

/// ⑤ 的编译执行：产物已在缓存里就直接跑，否则编一次再落盘。
///
/// 产物的身份是 [`ArtifactKey`]（源文本 + 编译器 + 主函数），因此"命中产物"与"重新编译"
/// 必然等价——形状、样本数据与重复次数都在主函数里，换一个就换一份产物。命中时连编译器都
/// 不需要：一份已经验证过的产物不该因为这台机器上没有 `cc` 而失效。
///
/// 编译在一次性的沙箱里进行（`work.dir` 下按产物键 + 进程命名），产物拷进缓存后沙箱即删，
/// 于是"编译过一次"不会在 `work.dir` 里留下第二份可执行文件。
///
/// # Errors
///
/// 编译器缺失或编译失败、产物不可执行、缓存不可写时返回说明；运行失败由
/// [`crate::backend::run`] 分类。
pub fn execute(
    profile: &Profile,
    tool: &str,
    key: &CacheKey,
    harness: &str,
    settings: &Settings,
) -> diagnostic::Result<(Outcome, Artifact)> {
    let cache = Cache::from_settings(settings);
    cached_run(
        &cache,
        ArtifactKey::new(key, tool, harness),
        tool,
        &profile.flags,
        harness,
        settings,
    )
}

/// 一份自建程序的编译执行（见证与将来的探测）：与 [`execute`] 共用同一条产物缓存。
///
/// 源文本不来自算子声明，所以条目按程序文本定名（[`Cache::put_program`]）；`flags` 也由
/// 调用方给——见证自检走的 host C 参数与设备档案的 `[compile].flags` 并不是同一套。
///
/// # Errors
///
/// 程序不可写盘、编译器缺失或编译失败、产物不可执行、缓存不可写时返回说明。
pub fn run_program(
    tool: &str,
    flags: &[String],
    label: &str,
    program: &str,
    settings: &Settings,
) -> diagnostic::Result<(Outcome, Artifact)> {
    let cache = Cache::from_settings(settings);
    let hex = cache
        .put_program(label, program, tool)
        .map_err(|error| Error::new(Code::Io, format!("写程序缓存失败：{error}")))?;
    cached_run(
        &cache,
        ArtifactKey::keyed(hex, tool, program),
        tool,
        flags,
        program,
        settings,
    )
}

/// 查产物 → 命中就直接跑；否则编译一次、落盘，再跑缓存里的那一份。
///
/// 跑的是缓存里那份而不是沙箱里那份：被缓存的、被报告的、被执行的必须是同一个文件。
fn cached_run(
    cache: &Cache,
    key: ArtifactKey,
    tool: &str,
    flags: &[String],
    program: &str,
    settings: &Settings,
) -> diagnostic::Result<(Outcome, Artifact)> {
    if let Some(artifact) = cache.artifact(&key) {
        return Ok((backend::run(&artifact.path)?, artifact));
    }
    let sandbox = settings
        .work_dir()
        .join(format!("{}-{}", key.hex(), std::process::id()));
    let exe = backend::compile(tool, flags, program, &sandbox)?;
    let artifact = cache.put_artifact(&key, &exe).map_err(|error| {
        Error::new(
            Code::Io,
            format!(
                "写产物缓存 {} 失败：{error}",
                cache.artifact_path(&key).display()
            ),
        )
    })?;
    let _ = std::fs::remove_dir_all(&sandbox);
    Ok((backend::run(&artifact.path)?, artifact))
}

/// 等锁的上限：另一个进程正常写完缓存就会释放锁，等到超时说明锁文件是残留的。
const LOCK_WAIT: Duration = Duration::from_secs(5);

/// 取锁：另一个进程正在为同一个 key 编译时等它（它的产物就是我们要的文本），等不到才报错。
///
/// 不静默继续，也不因为并行就先失败：`doctor`/`verify` 会为同一个 key 并发取锁，谁先写完
/// 都一样。每次失败都先看锁是否陈旧（[`STALE_LOCK`](crate::cache::STALE_LOCK)）：`O_EXCL`
/// 锁是文件而不是 flock，持有者被 `kill` 后没人能释放它，不破掉这个键就永久卡在"等待超时"
/// 上；活着的锁一律不抢，等满 `LOCK_WAIT` 才报错，错误里给出锁文件路径。
fn wait_for_lock(cache: &Cache, key: &CacheKey) -> diagnostic::Result<Lock> {
    let deadline = Instant::now() + LOCK_WAIT;
    loop {
        match cache.lock(key) {
            Ok(lock) => return Ok(lock),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                // 陈旧锁（持有者崩溃）没有谁会来释放它：先破掉再重试，不必等到超时。
                if cache.break_stale_lock(key) {
                    continue;
                }
                if Instant::now() >= deadline {
                    return Err(Error::new(
                        Code::Io,
                        format!(
                            "等待缓存 key `{}` 超时：{error}\n锁文件 {}；确认没有并发编译后删除它再重试",
                            key.hex(),
                            cache.dir_for(key).join("build.lock").display()
                        ),
                    ));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => {
                return Err(Error::new(
                    Code::Io,
                    format!("为缓存 key `{}` 取锁失败：{error}", key.hex()),
                ));
            }
        }
    }
}

/// 缓存目录：只有真正落过盘才返回（尚未生成时 `None`，报告里不出现指向空目录的路径）。
pub fn dir(key: &CacheKey, settings: &Settings) -> Option<PathBuf> {
    let cache = Cache::from_settings(settings);
    let path = cache.dir_for(key);
    path.is_dir().then_some(path)
}

/// 取片段层：配置了 `data.root` 就一律读磁盘（"不改代码替换数据"的入口），
/// 否则先查内嵌片段表（按 vendor + 扩展名，不写死任何一个厂商），再退回磁盘的 `<vendor>/`。
fn fragments(profile: &Profile, settings: &Settings) -> diagnostic::Result<FragmentSet> {
    let ext = profile.meta.emit.ext();
    if settings.data_root().is_none()
        && let Some(set) = prim::builtin(&profile.meta.vendor, ext)
    {
        return Ok(set);
    }
    let root = settings
        .data_root()
        .map_or_else(prim::dir, prim::layer_root);
    prim::load(&root, &profile.meta.vendor, ext).map_err(|error| {
        Error::new(
            Code::Missing,
            format!("`{}` 的片段层缺失：{error}", profile.meta.vendor),
        )
    })
}

/// 一条声明的搬运量：原语、存储层、字节数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    /// 原语。
    pub prim: Prim,
    /// 存储层。
    pub space: Space,
    /// 字节数（声明的 `extent` 上界）。
    pub bytes: u64,
}

/// 构建期存储契约：声明的搬运量与 swizzle 粒度都必须落在档案容量内（DESIGN §6.3）。
///
/// 声明的 `extent` 上界就是“一次搬运多少字节”（`load bytes<=4096`）；`*` 表示不限，不参与判定。
/// 返回值就是判定时真正比较过的那些数：审计把同一份数据打进 `doctor` 报告，报告与门禁
/// 因此不会各算一份（“为什么它通过了”与“它到底算了什么”是同一组值）。
///
/// # Errors
///
/// 变体不存在、某层的声明搬运量超过档案容量、或 `swz` 的 bank 粒度超过档案声明时
/// 返回说明。
pub fn contract(op: &OpDecl, profile: &Profile, path: &Path) -> diagnostic::Result<Vec<Usage>> {
    let variant = op
        .variant(&path.variant)
        .ok_or_else(|| Error::new(Code::Invalid, format!("变体 `{}` 不存在", path.variant)))?;
    let uses: Vec<Usage> = variant
        .needs(op)
        .filter(|need| need.extent.hi() != u32::MAX)
        .flat_map(|need| {
            let bytes = u64::from(need.extent.hi());
            Space::ALL
                .into_iter()
                .filter(move |unit| need.space.contains(*unit))
                .map(move |space| Usage {
                    prim: need.prim,
                    space,
                    bytes,
                })
        })
        .collect();
    let charges: Vec<space::SpaceUse> = uses
        .iter()
        .map(|usage| space::SpaceUse::new(usage.space, usage.bytes))
        .collect();
    space::check_capacity(profile, &charges)?;

    for (slot, spec) in &op.layouts {
        if let LayoutSpec::Swz { b, .. } = spec {
            space::check_alignment(profile, *b)
                .map_err(|error| error.with(format!("布局槽位 `{slot}`")))?;
        }
    }
    Ok(uses)
}

/// 缓存指纹（DESIGN §10.3）：键覆盖的每一项都必须是"改了它就会改变产物"的输入。
///
/// `flags` 与 `preamble` 一起哈希：编译参数既影响产物也影响执行，档案改一个字节就要换键。
/// 布局与数据类型也取声明而不是编一个具体形状：声明式会插进模板（`{{w}}`），而
/// `w[n, m/2]` 的步长不会——后者只在给定形状时才有意义，不属于构建期输入。
fn key(
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &TileCombo,
    settings: &Settings,
) -> CacheKey {
    let mut flags = profile.flags.join(" ");
    flags.push_str(&format!("|{}", profile.preamble));
    let layouts: Vec<(String, String)> = op
        .layouts
        .iter()
        .map(|(slot, spec)| (slot.clone(), spec.as_str()))
        .collect();
    let dtype = op
        .variant(&path.variant)
        .map(|variant| {
            DTypeSet::of(
                DType::ALL
                    .into_iter()
                    .filter(|dt| variant.needs(op).any(|need| need.dt.contains(*dt))),
            )
        })
        .unwrap_or_default();
    CacheKey {
        schema_ver: SCHEMA_VER,
        emit: profile.meta.emit,
        tool: profile.meta.tool.clone(),
        tool_ver: profile.meta.tool_ver.clone(),
        flags_hash: emit::fnv1a(&flags),
        arch: profile.meta.id.clone(),
        template_hash: emit::fnv1a(&op.template),
        fragments: fragments(profile, settings)
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
        dtype,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op::model::Sample;

    fn cpu_profile() -> Profile {
        Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案可解析")
    }

    /// 只改声明里的某一处（模板不变）：用来构造"缓存键不变、判定结果变了"的输入。
    fn declared(from_to: (&str, &str)) -> OpDecl {
        let source = &op::builtin()[0];
        let text = source.declaration().replace(from_to.0, from_to.1);
        OpDecl::parse(&text, source.template()).expect("声明可解析")
    }

    fn builtin() -> OpDecl {
        let source = &op::builtin()[0];
        OpDecl::parse(source.declaration(), source.template()).expect("内置声明可解析")
    }

    fn cpu_path(op: &OpDecl) -> Path {
        crate::cap::negotiate(op, &cpu_profile()).expect("CPU 应命中首档")
    }

    /// 不经协商的路径：契约测试要故意构造协商会拒绝的声明。
    fn declared_path(op: &OpDecl) -> Path {
        let variant = &op.variants[0];
        Path {
            variant: variant.name.clone(),
            index: 0,
            schedule: variant.schedule,
            mma_share: variant.mma_share,
        }
    }

    fn first_tile(op: &OpDecl) -> TileCombo {
        crate::decl::expand(op)
            .expect("轴可展开")
            .into_iter()
            .next()
            .expect("至少一个 tile 取值")
    }

    fn settings(root: &std::path::Path) -> Settings {
        let roots = llm_config::Roots::new(root.to_path_buf());
        let mut settings = Settings::defaults(&roots);
        settings
            .apply_override(&format!("cache.dir={}", root.join("cache").display()))
            .expect("cache.dir 覆盖合法");
        settings
    }

    #[test]
    fn declared_traffic_beyond_the_fast_capacity_fails_the_build() {
        let op = declared((r#"bytes = "<=4096""#, r#"bytes = "<=1048576""#));
        let error = contract(&op, &cpu_profile(), &declared_path(&op)).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("FAST"), "{error}");
        assert!(error.to_string().contains("524288"), "{error}");
    }

    #[test]
    fn swizzle_beyond_the_bank_count_fails_the_build() {
        let op = declared((r#"w = "row""#, r#"w = "swz<40,3,3>""#));
        let error = contract(&op, &cpu_profile(), &declared_path(&op)).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("布局槽位 `w`"), "{error}");
    }

    #[test]
    fn the_contract_reports_the_numbers_it_compared() {
        let op = builtin();
        let uses = contract(&op, &cpu_profile(), &cpu_path(&op)).expect("内置声明在容量内");
        assert!(
            uses.iter().any(|usage| usage.prim == Prim::Load
                && usage.space == Space::Fast
                && usage.bytes == 4096),
            "{uses:?}"
        );
        assert!(
            uses.iter().all(|usage| usage.space != Space::Reg),
            "没有声明字节数的原语不该凭空占用 REG：{uses:?}"
        );
    }

    #[test]
    fn a_cached_source_cannot_exempt_the_declaration_from_the_contract() {
        // 声明的字节数只影响判定、不影响源文本，所以它不在缓存键里。先按合法声明落一次盘，
        // 再只把字节数改到超出容量：键不变、缓存必然命中——契约仍必须重新判定。
        let root = std::env::temp_dir().join(format!("cutile-contract-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let settings = settings(&root);
        let profile = cpu_profile();
        let op = builtin();
        let path = cpu_path(&op);
        let tile = first_tile(&op);
        source(&op, &profile, &path, &tile, &settings).expect("合法声明可构建");

        let invalid = declared((r#"bytes = "<=4096""#, r#"bytes = "<=1048576""#));
        let error = source(&invalid, &profile, &path, &tile, &settings).unwrap_err();
        assert!(
            error.to_string().contains("524288"),
            "缓存命中不能跳过构建期契约：{error}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 真实的编译器名字（本机没有编译器时返回 `None`，用例跳过）。
    fn tool() -> Option<String> {
        crate::backend::detect_toolchain()
    }

    /// 一份带主函数的内核源码与形状：产物身份要用的两个输入。
    fn built(op: &OpDecl, settings: &Settings) -> (CacheKey, String, Sample) {
        let profile = cpu_profile();
        let path = cpu_path(op);
        let tile = first_tile(op);
        let (source, key) = source(op, &profile, &path, &tile, settings).expect("可构建");
        let sample = (op::model(&op.name).expect("算子模型").sample)(32, 1);
        op.signature.check(&sample).expect("ABI 对齐");
        let harness =
            crate::backend::harness(&source.text, &op.signature, &sample, 1).expect("可生成");
        (key, harness, sample)
    }

    #[cfg(unix)]
    #[test]
    fn a_second_execution_needs_neither_the_compiler_nor_a_rebuild() {
        // 用一份"转调真正的 cc"的脚本充当编译器：第一次真的编，删掉脚本后第二次必须命中
        // 产物——真去找编译器就会以 `io` 失败，于是"没调用编译器"这件事是可观测的。
        let Some(tool) = tool() else {
            return;
        };
        let root = std::env::temp_dir().join(format!("cutile-execute-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let settings = settings(&root);
        let profile = cpu_profile();
        let op = builtin();
        let (key, harness, sample) = built(&op, &settings);

        let wrapper = root.join("fake-cc.sh");
        std::fs::create_dir_all(&root).expect("临时目录");
        std::fs::write(&wrapper, format!("#!/bin/sh\nexec {tool} \"$@\"\n")).expect("写脚本");
        let mut perms = std::fs::metadata(&wrapper).expect("脚本").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
        std::fs::set_permissions(&wrapper, perms).expect("可执行");
        let wrapper = wrapper.to_string_lossy().to_string();

        let (first, artifact) =
            execute(&profile, &wrapper, &key, &harness, &settings).expect("第一次应真的编译");
        assert!(!artifact.reused, "{artifact:?}");
        assert_eq!(first.bits.len(), sample.n);

        std::fs::remove_file(&wrapper).expect("删掉编译器");
        let (second, hit) =
            execute(&profile, &wrapper, &key, &harness, &settings).expect("应复用产物");
        assert!(hit.reused, "{hit:?}");
        assert_eq!(second.bits, first.bits, "复用同一份产物，结果必须逐位相同");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_artifact_cache_leaves_the_work_directory_empty() {
        let Some(tool) = tool() else {
            return;
        };
        let root = std::env::temp_dir().join(format!("cutile-sandbox-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let settings = settings(&root);
        let op = builtin();
        let (key, harness, _) = built(&op, &settings);
        execute(&cpu_profile(), &tool, &key, &harness, &settings).expect("应能编译执行");
        let leftovers: Vec<_> = std::fs::read_dir(settings.work_dir())
            .map(|items| items.flatten().map(|item| item.file_name()).collect())
            .unwrap_or_default();
        assert!(
            leftovers.is_empty(),
            "产物拷进缓存后沙箱就该删掉，否则每次编译都留一份可执行文件：{leftovers:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}

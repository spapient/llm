//! ⑤ emit + cache：指纹 → 查缓存 → 取锁 → 拼接 → emit → 落盘（README §10.3）。

use crate::config::config;
use crate::runner::{self, Outcome};
use crate::toolchain::Toolchain;
use cutile::SCHEMA_VER;
use cutile::atom::{DType, DTypeSet, Prim, Space};
use cutile::cache::{Artifact, ArtifactKey, Cache, CacheKey, Lock};
use cutile::cap::{AtomTable, Path};
use cutile::decl::{OpDecl, TileCombo};
use cutile::dirs::Dirs;
use cutile::emit::{self, Source};
use cutile::kernel::Program;
use cutile::layout::LayoutSpec;
use cutile::parser;
use cutile::prof::Profile;
use cutile::space;
use cutile::substitute::FragmentSet;
use diagnostic::{Code, Error};
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path as StdPath, PathBuf};
use std::time::{Duration, Instant};

/// ⑤ 的完整流程：指纹 → 查缓存 → 取锁 → 拼接 → emit → 落盘。缓存命中时直接返回已落盘的文本——
/// 键相同意味着源文本相同。
///
/// # Errors
///
/// 片段层缺失、变体不存在、构建期契约（容量/对齐）不通过、取锁失败或落盘失败时返回说明。
pub fn source(
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &TileCombo,
    dirs: &Dirs,
) -> diagnostic::Result<(Source, CacheKey)> {
    // 缓存键与 IR 都从这次真正选择的类型化原子派生。
    let specialized = prof::catalogue::for_device(op, &profile.meta.vendor)?;
    let op = specialized.as_ref();
    // 契约先于缓存：已落盘的旧源文本不能绕过容量、能力或对齐检查。
    contract(op, profile, path)?;
    let fragments = fragments(profile, dirs);
    let prepared = Prepared::new(op, profile, path, tile, &fragments);
    let key = key(op, profile, path, tile, &prepared);
    let cache = Cache::from_dirs(dirs);
    if let Some(source) = hit(&cache, &key, profile) {
        return Ok((source, key));
    }
    let source = publish(&cache, &key, op, profile, path, tile, &prepared)?;
    Ok((source, key))
}

/// 键判定前完成一次 IR lower；显式原生模板不进入 DSL 子集解析。
struct Prepared<'a> {
    fragments: &'a diagnostic::Result<FragmentSet>,
    program: Option<diagnostic::Result<Program<'a>>>,
}

impl<'a> Prepared<'a> {
    fn new(
        op: &OpDecl,
        profile: &'a Profile,
        path: &Path,
        tile: &TileCombo,
        fragments: &'a diagnostic::Result<FragmentSet>,
    ) -> Self {
        let program = (!prof::sign::is_native_template(&op.template, &op.name))
            .then(|| lower(op, profile, path, tile, fragments));
        Self { fragments, program }
    }
}

/// 协商当前变体 → 绑定原子 arena → 解析完整模板；没有文本回退。
fn lower<'a>(
    op: &OpDecl,
    profile: &'a Profile,
    path: &Path,
    tile: &TileCombo,
    fragments: &'a diagnostic::Result<FragmentSet>,
) -> diagnostic::Result<Program<'a>> {
    let set = fragments.as_ref().map_err(Clone::clone)?;
    let variant = op
        .variant(&path.variant)
        .ok_or_else(|| Error::new(Code::Invalid, format!("变体 `{}` 不存在", path.variant)))?;
    let atoms = AtomTable::new(op, variant, profile, set)?;
    parser::parse(op, tile, &atoms)
}

/// 未命中时的一次生成：取锁 → 再查一次 → 拼接 → 落盘。锁必须真的拿到（否则两个进程同时写同一份
/// `source`），但"另一个进程正在为同一个 key 编译"不是错误：它写出的就是我们要的文本，所以锁内
/// 要**再查一次**，查的是等锁期间别人落盘的那一份。
fn publish(
    cache: &Cache,
    key: &CacheKey,
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &TileCombo,
    prepared: &Prepared<'_>,
) -> diagnostic::Result<Source> {
    let _lock = wait_for_lock(cache, key)?;
    match hit(cache, key, profile) {
        Some(source) => Ok(source),
        None => build_source(cache, key, op, profile, path, tile, prepared),
    }
}

/// 真正生成一次并落盘；普通模板走 IR，只有显式标记的原生模板保留设备翻译单元。
fn build_source(
    cache: &Cache,
    key: &CacheKey,
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &TileCombo,
    prepared: &Prepared<'_>,
) -> diagnostic::Result<Source> {
    let variant = op
        .variant(&path.variant)
        .ok_or_else(|| Error::new(Code::Invalid, format!("变体 `{}` 不存在", path.variant)))?;
    let text = render_body(op, profile, tile, prepared)?;
    generate(op, profile, variant, tile, &text, cache, key)
}

/// 根据显式模板类别选择正文来源；普通模板解析失败不会转入原生路径。
fn render_body(
    op: &OpDecl,
    profile: &Profile,
    tile: &TileCombo,
    prepared: &Prepared<'_>,
) -> diagnostic::Result<String> {
    match &prepared.program {
        Some(program) => render_ir(program, profile),
        None => render_native(op, tile, prepared.fragments),
    }
}

fn render_ir(
    program: &diagnostic::Result<Program<'_>>,
    profile: &Profile,
) -> diagnostic::Result<String> {
    program
        .as_ref()
        .map_err(Clone::clone)?
        .emit(profile.meta.emit)
        .map_err(|error| Error::new(Code::Invalid, error.to_string()))
}

fn render_native(
    op: &OpDecl,
    tile: &TileCombo,
    fragments: &diagnostic::Result<FragmentSet>,
) -> diagnostic::Result<String> {
    cutile::substitute::render(op, tile, fragments.as_ref().map_err(Clone::clone)?)
}

/// 缓存命中：键相同意味着源文本相同。
fn hit(cache: &Cache, key: &CacheKey, profile: &Profile) -> Option<Source> {
    cache.get(key).map(|text| Source {
        lang: profile.meta.emit,
        text,
    })
}

/// 未命中时统一组装并落盘；两条正文来源由显式模板类别决定，不在解析失败后回退。
fn generate(
    op: &OpDecl,
    profile: &Profile,
    variant: &cutile::decl::Variant,
    tile: &TileCombo,
    body: &str,
    cache: &Cache,
    key: &CacheKey,
) -> diagnostic::Result<Source> {
    let source = emit::assemble(profile, op, variant, tile, body);
    cache.put(key, &source.text).map_err(|error| {
        Error::new(
            Code::Io,
            format!("写缓存 {} 失败：{error}", cache.dir_for(key).display()),
        )
    })?;
    Ok(source)
}

/// ⑤ 的编译执行：产物已在缓存里就直接跑，否则编一次再落盘。产物身份是 [`ArtifactKey`]（源文本 +
/// **实测工具链身份** + 主函数），所以"命中产物"与"重新编译"必然等价。
///
/// `tools` 可以是 `None`：一份验证过的产物不该因为本机没有 `cc` 而失效——按产物旁的 `.key` 认人即可，
/// 只有"既没编译器、缓存里也没有这份产物"才失败。编译在一次性的沙箱里进行（`work.dir` 下按产物键 +
/// 进程命名），产物拷进缓存后沙箱即删。
///
/// # Errors
///
/// 编译器缺失（且缓存里没有这份产物）、编译失败、产物不可执行、缓存不可写时返回说明；运行失败由
/// `runner::run` 分类。
pub fn execute(
    profile: &Profile,
    tool: Option<&Toolchain>,
    key: &CacheKey,
    harness: &str,
    dirs: &Dirs,
) -> diagnostic::Result<(Outcome, Artifact)> {
    let cache = Cache::from_dirs(dirs);
    let flags = runner::compile_flags(profile, dirs)?;
    cached_run(
        &cache,
        &key.hex(),
        key.flags_hash,
        tool,
        &flags,
        harness,
        dirs,
    )
}

/// 一份自建程序的编译执行（见证与将来的探测）：与 [`execute`] 共用同一条产物缓存。源文本不来自算子
/// 声明，所以条目按程序文本定名（[`Cache::put_program`]），`flags` 也由调用方给；`tools` 为 `None`
/// 表示本机没有编译器，此时只能复用已落盘的产物。
///
/// # Errors
///
/// 程序不可写盘、编译器缺失（且缓存里没有这份产物）、编译失败、产物不可执行或缓存不可写时返回说明。
pub fn run_program(
    tool: Option<&Toolchain>,
    flags: &[String],
    label: &str,
    program: &str,
    dirs: &Dirs,
) -> diagnostic::Result<(Outcome, Artifact)> {
    let cache = Cache::from_dirs(dirs);
    let hex = cache
        .put_program(label, program)
        .map_err(|error| Error::new(Code::Io, format!("写程序缓存失败：{error}")))?;
    cached_run(
        &cache,
        &hex,
        ArtifactKey::params(flags),
        tool,
        flags,
        program,
        dirs,
    )
}

/// 查产物 → 命中就直接跑；否则编译一次、落盘，再跑缓存里的那一份（被缓存的、被报告的、被执行的必须是同一个文件）。
fn cached_run(
    cache: &Cache,
    source: &str,
    params: u64,
    tool: Option<&Toolchain>,
    flags: &[String],
    program: &str,
    dirs: &Dirs,
) -> diagnostic::Result<(Outcome, Artifact)> {
    let Some(tool) = tool else {
        return cached_only(cache, source, params, program);
    };
    // 产物身份用**实测**的工具链与**这次**的编译参数：升一次编译器、换一组参数都是另一套
    // 代码生成，旧产物不再冒充这一轮的产物。
    let key = ArtifactKey::keyed(source, &tool.id(), params, program);
    match cache.artifact(&key) {
        Some(artifact) => Ok((runner::run(&artifact.path)?, artifact)),
        None => compile_and_store(cache, tool, flags, program, &key, dirs),
    }
}

/// 编一次、落盘、跑缓存里那一份：被缓存的、被报告的、被执行的是同一个文件。
fn compile_and_store(
    cache: &Cache,
    tool: &Toolchain,
    flags: &[String],
    program: &str,
    key: &ArtifactKey,
    dirs: &Dirs,
) -> diagnostic::Result<(Outcome, Artifact)> {
    // 同一份产物可能被并行用例同时编译：先取产物键锁，锁内再查一次——等锁期间别人落的那一份
    // 就是我们要的，重编会和正在执行的二进制争同一个路径（Linux 下表现为 ETXTBSY）。锁持有到
    // 本次执行结束，避免另一条路径在我们 exec 期间替换文件。
    let _lock = wait_for_artifact_lock(cache, key)?;
    if let Some(reused) = reuse_cached(cache, key)? {
        return Ok(reused);
    }
    compile_fresh(cache, tool, flags, program, key, dirs)
}

/// 锁内复查缓存：等锁期间别人落的那一份直接复用；没有命中返回 `None` 走新编。
fn reuse_cached(
    cache: &Cache,
    key: &ArtifactKey,
) -> diagnostic::Result<Option<(Outcome, Artifact)>> {
    let Some(artifact) = cache.artifact(key) else {
        return Ok(None);
    };
    let outcome = runner::run(&artifact.path)?;
    Ok(Some((outcome, artifact)))
}

/// 未命中：新编一次、落盘产物缓存，再跑落盘那一份。
fn compile_fresh(
    cache: &Cache,
    tool: &Toolchain,
    flags: &[String],
    program: &str,
    key: &ArtifactKey,
    dirs: &Dirs,
) -> diagnostic::Result<(Outcome, Artifact)> {
    runner::driver::with_scratch(dirs, &key.hex(), |sandbox| {
        let exe = runner::compile(&tool.command, flags, program, sandbox)?;
        let artifact = cache.put_artifact(key, &exe).map_err(|error| {
            Error::new(
                Code::Io,
                format!(
                    "写产物缓存 {} 失败：{error}",
                    cache.artifact_path(key).display()
                ),
            )
        })?;
        Ok((runner::run(&artifact.path)?, artifact))
    })
}

/// 没有编译器时的唯一出路：复用已经落盘的同一份产物。主函数与**编译参数**必须逐字相同，工具链可以
/// 不同——但那时会有多份候选，选哪一份都是替用户决定"用谁的代码生成"，所以不猜：列出工具链身份
/// （`命令@版本`），让用户装一个或清掉多余的产物。
fn cached_only(
    cache: &Cache,
    source: &str,
    params: u64,
    program: &str,
) -> diagnostic::Result<(Outcome, Artifact)> {
    let harness = emit::fnv1a(program);
    // 主函数与编译参数都是这次请求的一部分：前者决定形状与样本，后者决定这份二进制是怎么
    // 编出来的。参数不同的那份来自另一条命令行，不是"同一份产物"的另一个副本。
    let candidates: Vec<Artifact> = cache
        .artifacts_in(source)
        .into_iter()
        .filter(|artifact| artifact.key.harness == harness && artifact.key.params == params)
        .collect();
    match candidates.as_slice() {
        [one] => Ok((runner::run(&one.path)?, one.clone())),
        [] => Err(Error::new(
            Code::Unsupported,
            format!(
                "缺少 C 编译器（cc/clang/gcc 都不在 PATH 上），缓存条目 `{source}` 里也没有这份产物（主函数或编译参数 `{params:016x}` 不同）\n先在有编译器的机器上跑一次，或把编译器装上"
            ),
        )),
        many => {
            let tools: Vec<&str> = many
                .iter()
                .map(|artifact| artifact.key.tool.as_str())
                .collect();
            Err(Error::new(
                Code::Unsupported,
                format!(
                    "缺少 C 编译器（cc/clang/gcc 都不在 PATH 上），而缓存里这份产物有 {} 个工具链留下的版本（{}）：无法判定复用哪一份\n装上其中一个（它会被优先使用），或删掉多余的产物",
                    tools.len(),
                    tools.join(" / ")
                ),
            ))
        }
    }
}

/// 取源码缓存锁：别人正在为同一个 key 编译时等它（它的产物就是我们要的文本），等不到才报错。
fn wait_for_lock(cache: &Cache, key: &CacheKey) -> diagnostic::Result<Lock> {
    wait_for_lock_at(cache, &cache.dir_for(key).join("build.lock"))
}

/// 取产物缓存锁：同一产物键的并行编译在这里排队，锁内复查缓存命中（见 [`compile_and_store`]）。
fn wait_for_artifact_lock(cache: &Cache, key: &ArtifactKey) -> diagnostic::Result<Lock> {
    wait_for_lock_at(cache, &cache.artifact_lock_path(key))
}

/// 取锁：别人正在为同一个 key 编译时等它（它的产物就是我们要的文本），等不到才报错。每次失败都先看
/// 锁是否陈旧（[`STALE_LOCK`](cutile::cache::STALE_LOCK)）：`O_EXCL` 锁是文件，持有者被 `kill` 后没人
/// 能释放它；活着的锁一律不抢，等满 `LOCK_WAIT` 才报错，错误里给出锁文件路径。
fn wait_for_lock_at(cache: &Cache, path: &StdPath) -> diagnostic::Result<Lock> {
    let deadline = Instant::now() + config().lock_wait;
    loop {
        match cache.lock_at(path) {
            Ok(lock) => return Ok(lock),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                await_lock_at(cache, path, &error, deadline)?;
            }
            Err(error) => {
                return Err(Error::new(
                    Code::Io,
                    format!("取锁失败：{error}（{}）", path.display()),
                ));
            }
        }
    }
}

/// 锁已存在时的一次等待：陈旧锁先破掉（没有谁会来释放它），活锁小睡一会儿，等满期限才报错。
fn await_lock_at(
    cache: &Cache,
    path: &StdPath,
    busy: &IoError,
    deadline: Instant,
) -> diagnostic::Result<()> {
    if cache.break_stale_lock_at(path) {
        return Ok(());
    }
    if Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
        return Ok(());
    }
    Err(Error::new(
        Code::Io,
        format!(
            "等待缓存锁超时：{busy}\n锁文件 {}；确认没有并发编译后删除它再重试",
            path.display()
        ),
    ))
}

/// 缓存目录：只有真正落过盘才返回（尚未生成时 `None`，报告里不出现指向空目录的路径）。
pub fn dir(key: &CacheKey, dirs: &Dirs) -> Option<PathBuf> {
    let cache = Cache::from_dirs(dirs);
    let path = cache.dir_for(key);
    path.is_dir().then_some(path)
}

/// 取片段层：配置了 `data.root` 就一律读磁盘（"不改代码替换数据"的入口），
/// 否则读该设备自己的 crate（按 vendor + 扩展名，不写死任何一个厂商）。
fn fragments(profile: &Profile, dirs: &Dirs) -> diagnostic::Result<FragmentSet> {
    let ext = profile.meta.emit.ext();
    dirs.prim()
        .fragments(&profile.meta.vendor, ext)
        .map_err(|error| error.with(format!("`{}` 的片段层", profile.meta.vendor)))
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

/// 构建期存储契约：声明的搬运量与 swizzle 粒度都必须落在档案容量内（README §6.3）。返回值就是判定时
/// 真正比较过的那些数，审计把它打进 `doctor` 报告，于是"为什么它通过了"与"它到底算了什么"是同一组值。
///
/// # Errors
///
/// 变体不存在、某层的声明搬运量超过档案容量、或 `swz` 的 bank 粒度超过档案声明时返回说明。
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
                .iter()
                .copied()
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
    check_layouts(op, profile)?;
    Ok(uses)
}

/// 布局契约：`swz` 的 bank 粒度必须落在档案声明内；其余构造子不参与这条判定。
fn check_layouts(op: &OpDecl, profile: &Profile) -> diagnostic::Result<()> {
    for (slot, spec) in &op.layouts {
        check_layout(slot, spec, profile)?;
    }
    Ok(())
}

fn check_layout(slot: &str, spec: &LayoutSpec, profile: &Profile) -> diagnostic::Result<()> {
    let LayoutSpec::Swz { b, .. } = spec else {
        return Ok(());
    };
    space::check_alignment(profile, *b).map_err(|error| error.with(format!("布局槽位 `{slot}`")))
}

/// 内核实现标签：参与展开片段的实现名并集排序去重；片段装载失败时回退空标签。
fn impl_tag(fragments: &diagnostic::Result<FragmentSet>) -> String {
    fragments.as_ref().map_or_default(|set| {
        let mut names: Vec<&str> = Prim::ALL
            .iter()
            .copied()
            .filter_map(|prim| set.implementation(prim))
            .collect();
        names.sort();
        names.dedup();
        names.join(",")
    })
}

/// 缓存指纹（README §10.3）：键覆盖的每一项都必须是"改了它就会改变产物"的输入。`flags` 与 `preamble`
/// 一起哈希；布局与数据类型取声明而不是编一个具体形状——声明式会插进模板，给定形状才有的步长不属于
/// 构建期输入。
fn key(
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &TileCombo,
    prepared: &Prepared<'_>,
) -> CacheKey {
    let mut flags = profile.flags.join(" ");
    flags.push_str(&format!("|{}", profile.preamble));
    for (name, text) in &profile.headers {
        flags.push_str(&format!("|{name}={text}"));
    }
    let layouts: Vec<(String, String)> = op
        .layouts
        .iter()
        .map(|(slot, spec)| (slot.clone(), spec.as_str()))
        .collect();
    let dtype = op.variant(&path.variant).map_or_default(|variant| {
        DTypeSet::of(
            DType::ALL
                .iter()
                .copied()
                .filter(|dt| variant.needs(op).any(|need| need.dt.contains(*dt))),
        )
    });

    let fragment_hashes: Vec<(Prim, u64)> = prepared.fragments.as_ref().map_or_default(|set| {
        Prim::ALL
            .iter()
            .copied()
            .filter_map(|prim| set.get(prim).map(|text| (prim, emit::fnv1a(text))))
            .collect()
    });
    let impl_name = impl_tag(prepared.fragments);
    // 正常产物使用优化后 IR 的规范编码；显式原生模板以原文本身标识缓存。
    let ast_hash = match &prepared.program {
        Some(Ok(program)) => program.encode().map_or_else(
            |_| emit::fnv1a(&op.template),
            |encoded| emit::fnv1a(&encoded),
        ),
        None | Some(Err(_)) => emit::fnv1a(&op.template),
    };

    CacheKey {
        schema_ver: SCHEMA_VER,
        emit: profile.meta.emit,
        flags_hash: emit::fnv1a(&flags),
        arch: profile.meta.id.clone(),
        impl_name,
        block: "1".to_string(),
        ast_hash,
        fragments: fragment_hashes,
        layouts,
        variant: path.variant.clone(),
        tile: tile.summary(),
        dtype,
    }
}

#[cfg(test)]
#[path = "../tests/build_tests.rs"]
mod tests;

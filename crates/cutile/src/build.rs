//! ⑤ emit + cache：指纹 → 查缓存 → 取锁 → 拼接 → emit → 落盘（README §10.3）。
//!
//! 输入全是数据：算子声明、协商出的变体、tile 取值与配置；输出是源文本与它的缓存键。键覆盖真正喂给
//! `cc` 的参数、片段文本、布局声明式与数据类型集合，因此"命中缓存"与"重新生成"必然等价。
//! 本模块不认识任何命令：审计（`llm-audit`）与将来的落地流程都从 [`source`] 取文本，命令策略不在内核里。

use crate::SCHEMA_VER;
use crate::ast;
use crate::atom::{DType, DTypeSet, Prim, Space};
use crate::cache::{Artifact, ArtifactKey, Cache, CacheKey, Lock};
use crate::cap::Path;
use crate::decl::{OpDecl, TileCombo};
use crate::dirs::Dirs;
use crate::emit::{self, Source};
use crate::layout::LayoutSpec;
use crate::prof::Profile;
use crate::runner::{self, Outcome};
use crate::space;
use crate::substitute::FragmentSet;
use crate::toolchain::Toolchain;
use diagnostic::{Code, Error};
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path as StdPath, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// 沙箱目录的进程内序号：同一个 key 会被同一进程的两条路径同时编译（`verify` 的自检与
/// 形状矩阵、两条用例），`pid` 只区分进程，不区分线程。
static SANDBOX_SEQ: AtomicUsize = AtomicUsize::new(0);

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
    // 契约先于缓存：声明里的搬运量只影响判定、不影响源文本，所以它不在缓存键里。
    // 如果把判定放在命中之后，一份已落盘的旧源文本就成了“声明越界也免检”的通行证。
    contract(op, profile, path)?;
    // 片段层与 AST 在键判定前各算一次：命中与未命中复用同一份，未命中不再重读磁盘、
    // 不再重跑 AST 构建。片段层不可用时保留旧的容错：键回退到模板指纹，命中仍可直接取用。
    let prepared = Prepared::new(op, profile, tile, dirs);
    let key = key(op, profile, path, tile, &prepared);
    let cache = Cache::from_dirs(dirs);
    if let Some(source) = hit(&cache, &key, profile) {
        return Ok((source, key));
    }
    let source = publish(&cache, &key, op, profile, path, tile, &prepared)?;
    Ok((source, key))
}

/// 键判定前算好的一份输入：片段层与提升后的节点序列，命中与未命中共用同一份。
struct Prepared {
    fragments: diagnostic::Result<FragmentSet>,
    nodes: diagnostic::Result<Vec<ast::Node>>,
}

impl Prepared {
    fn new(op: &OpDecl, profile: &Profile, tile: &TileCombo, dirs: &Dirs) -> Self {
        let fragments = fragments(profile, dirs);
        let nodes = match &fragments {
            Ok(set) => ast::build(op, tile, set),
            Err(error) => Err(error.clone()),
        };
        Self { fragments, nodes }
    }
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
    prepared: &Prepared,
) -> diagnostic::Result<Source> {
    let _lock = wait_for_lock(cache, key)?;
    match hit(cache, key, profile) {
        Some(source) => Ok(source),
        None => build_source(cache, key, op, profile, path, tile, prepared),
    }
}

/// 真正拼一次并落盘。变体必须在声明的变体表里：协商给出的名字与声明脱节时，这里是唯一的检查点。
fn build_source(
    cache: &Cache,
    key: &CacheKey,
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &TileCombo,
    prepared: &Prepared,
) -> diagnostic::Result<Source> {
    let variant = op
        .variant(&path.variant)
        .ok_or_else(|| Error::new(Code::Invalid, format!("变体 `{}` 不存在", path.variant)))?;
    // 片段层错误在此透传；片段本身不再参与后续生成——渲染只需要已提升好的节点序列。
    if let Err(error) = &prepared.fragments {
        return Err(error.clone());
    }
    let nodes = prepared.nodes.clone().map_err(|error| error.clone())?;
    generate(op, profile, variant, tile, &nodes, cache, key)
}

/// 缓存命中：键相同意味着源文本相同，重新拼一遍不会得到别的东西。
fn hit(cache: &Cache, key: &CacheKey, profile: &Profile) -> Option<Source> {
    cache.get(key).map(|text| Source {
        lang: profile.meta.emit,
        text,
    })
}

/// 未命中时的一次生成：AST → 文本 → 成文 → 落盘。数据流经过结构化 AST（ADR 0017 L1），双射测试
/// 保证它与直接 `substitute::render` 逐字节相等；字符串路径保留用于回滚与对拍。
fn generate(
    op: &OpDecl,
    profile: &Profile,
    variant: &op::decl::Variant,
    tile: &TileCombo,
    nodes: &[ast::Node],
    cache: &Cache,
    key: &CacheKey,
) -> diagnostic::Result<Source> {
    let text = ast::render(nodes);
    let source = emit::assemble(profile, op, variant, tile, &text);
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
/// `tool` 可以是 `None`：一份验证过的产物不该因为本机没有 `cc` 而失效——按产物旁的 `.key` 认人即可，
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
    cached_run(
        &cache,
        &key.hex(),
        key.flags_hash,
        tool,
        &profile.flags,
        harness,
        dirs,
    )
}

/// 一份自建程序的编译执行（见证与将来的探测）：与 [`execute`] 共用同一条产物缓存。源文本不来自算子
/// 声明，所以条目按程序文本定名（[`Cache::put_program`]），`flags` 也由调用方给；`tool` 为 `None`
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
    let sandbox = sandbox(dirs, key);
    let exe = runner::compile(&tool.command, flags, program, &sandbox)?;
    let artifact = cache.put_artifact(key, &exe).map_err(|error| {
        Error::new(
            Code::Io,
            format!(
                "写产物缓存 {} 失败：{error}",
                cache.artifact_path(key).display()
            ),
        )
    })?;
    let _ = std::fs::remove_dir_all(&sandbox);
    Ok((runner::run(&artifact.path)?, artifact))
}

/// 编译沙箱：目录名必须唯一，先编译完的那条会删掉它，共用目录会让另一条的 `cc -o`
/// 失去输出路径（表现为 linker 报 "cannot open output file"）。
fn sandbox(dirs: &Dirs, key: &ArtifactKey) -> PathBuf {
    dirs.work().join(format!(
        "{}-{}-{}",
        key.hex(),
        std::process::id(),
        SANDBOX_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
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

/// 等锁的上限：另一个进程正常写完缓存就会释放锁，等到超时说明锁文件是残留的。
const LOCK_WAIT: Duration = Duration::from_secs(5);

/// 取源码缓存锁：别人正在为同一个 key 编译时等它（它的产物就是我们要的文本），等不到才报错。
fn wait_for_lock(cache: &Cache, key: &CacheKey) -> diagnostic::Result<Lock> {
    wait_for_lock_at(cache, &cache.dir_for(key).join("build.lock"))
}

/// 取产物缓存锁：同一产物键的并行编译在这里排队，锁内复查缓存命中（见 [`compile_and_store`]）。
fn wait_for_artifact_lock(cache: &Cache, key: &ArtifactKey) -> diagnostic::Result<Lock> {
    wait_for_lock_at(cache, &cache.artifact_lock_path(key))
}

/// 取锁：别人正在为同一个 key 编译时等它（它的产物就是我们要的文本），等不到才报错。每次失败都先看
/// 锁是否陈旧（[`STALE_LOCK`](crate::cache::STALE_LOCK)）：`O_EXCL` 锁是文件，持有者被 `kill` 后没人
/// 能释放它；活着的锁一律不抢，等满 `LOCK_WAIT` 才报错，错误里给出锁文件路径。
fn wait_for_lock_at(cache: &Cache, path: &StdPath) -> diagnostic::Result<Lock> {
    let deadline = Instant::now() + LOCK_WAIT;
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
/// 否则先查内嵌片段表（按 vendor + 扩展名，不写死任何一个厂商），再退回磁盘的 `<vendor>/`。
fn fragments(profile: &Profile, dirs: &Dirs) -> diagnostic::Result<FragmentSet> {
    let ext = profile.meta.emit.ext();
    if dirs.data().is_none()
        && let Some(set) = prim::builtin(&profile.meta.vendor, ext)
    {
        return Ok(set);
    }
    let root = dirs.data().map_or_else(prim::dir, prim::layer_root);
    prim::load(&root, &profile.meta.vendor, ext)
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
    fragments
        .as_ref()
        .map(|set| {
            let mut names: Vec<&str> = Prim::ALL
                .iter()
                .copied()
                .filter_map(|prim| set.implementation(prim))
                .collect();
            names.sort();
            names.dedup();
            names.join(",")
        })
        .unwrap_or_default()
}

/// 缓存指纹（README §10.3）：键覆盖的每一项都必须是"改了它就会改变产物"的输入。`flags` 与 `preamble`
/// 一起哈希；布局与数据类型取声明而不是编一个具体形状——声明式会插进模板，给定形状才有的步长不属于
/// 构建期输入。
fn key(
    op: &OpDecl,
    profile: &Profile,
    path: &Path,
    tile: &TileCombo,
    prepared: &Prepared,
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
                    .iter()
                    .copied()
                    .filter(|dt| variant.needs(op).any(|need| need.dt.contains(*dt))),
            )
        })
        .unwrap_or_default();

    // 片段层只取一次：片段哈希与 AST 指纹都从同一份输入派生。
    let fragment_hashes: Vec<(Prim, u64)> = prepared
        .fragments
        .as_ref()
        .map(|set| {
            Prim::ALL
                .iter()
                .copied()
                .filter_map(|prim| set.get(prim).map(|text| (prim, emit::fnv1a(text))))
                .collect()
        })
        .unwrap_or_default();
    // 内核实现标签：参与展开片段的实现名并集排序去重；同原语多实现时各自成独立缓存条目（ADR 0027 §6）。
    let impl_name = impl_tag(&prepared.fragments);
    // 缓存指纹锚定结构化 AST：FNV(AST 规范编码)。AST 不可用时回退到模板哈希，
    // 保持 key() 不返回 Result 的容错契约；正常路径下两条路径已由双射测试守着。
    let ast_hash = match &prepared.nodes {
        Ok(nodes) => emit::fnv1a(&ast::encode(nodes)),
        Err(_) => emit::fnv1a(&op.template),
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
mod tests {
    use super::*;
    #[cfg(unix)]
    use crate::toolchain;
    use diagnostic::data::{Table, Value};
    use op::model::Sample;

    fn cpu_profile() -> Profile {
        Profile::parse(prof::embedded("cpu").expect("cpu 档案")).expect("档案可解析")
    }

    /// 只改声明表里的某一处（模板不变）：用来构造"缓存键不变、判定结果变了"的输入。
    fn declared(mutate: impl FnOnce(&mut Table)) -> OpDecl {
        let source = op::builtin_named("gemv_i4").expect("内置");
        let mut declaration = source.declaration().clone();
        mutate(&mut declaration);
        OpDecl::parse(&declaration, source.template()).expect("声明可解析")
    }

    /// 把所有搬运原语的 `bytes` 上限改到 `limit`。
    fn declared_bytes(limit: &str) -> OpDecl {
        declared(|declaration| {
            for prim in ["load", "store", "load_async"] {
                declaration.set(&format!("needs.{prim}.bytes"), Value::text(limit));
            }
        })
    }

    fn declared_swizzle() -> OpDecl {
        declared(|declaration| declaration.set("layouts.w", Value::text("swz<40,3,3>")))
    }

    fn builtin() -> OpDecl {
        let source = &op::builtin_named("gemv_i4").expect("内置");
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

    fn dirs(root: &std::path::Path) -> Dirs {
        Dirs::under(root)
    }

    #[test]
    fn declared_traffic_beyond_the_fast_capacity_fails_the_build() {
        let op = declared_bytes("<=1048576");
        let error = contract(&op, &cpu_profile(), &declared_path(&op)).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("FAST"), "{error}");
        assert!(error.to_string().contains("524288"), "{error}");
    }

    #[test]
    fn swizzle_beyond_the_bank_count_fails_the_build() {
        let op = declared_swizzle();
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
        let dirs = dirs(&root);
        let profile = cpu_profile();
        let op = builtin();
        let path = cpu_path(&op);
        let tile = first_tile(&op);
        source(&op, &profile, &path, &tile, &dirs).expect("合法声明可构建");

        let invalid = declared_bytes("<=1048576");
        let error = source(&invalid, &profile, &path, &tile, &dirs).unwrap_err();
        assert!(
            error.to_string().contains("524288"),
            "缓存命中不能跳过构建期契约：{error}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 本机实测的工具链（本机没有编译器时返回 `None`，用例跳过）。
    fn tool() -> Option<Toolchain> {
        crate::toolchain::detect()
    }

    /// 一份带主函数的内核源码与形状：产物身份要用的两个输入。
    fn built(op: &OpDecl, dirs: &Dirs) -> (CacheKey, String, Sample) {
        let profile = cpu_profile();
        let path = cpu_path(op);
        let tile = first_tile(op);
        let (source, key) = source(op, &profile, &path, &tile, dirs).expect("可构建");
        let sample = (op::model(&op.name).expect("算子模型").sample)(32, 1);
        op.signature.check(&sample).expect("ABI 对齐");
        let harness =
            crate::harness::harness(&source.text, &op.signature, &sample, 1).expect("可生成");
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
        let dirs = dirs(&root);
        let profile = cpu_profile();
        let op = builtin();
        let (key, harness, sample) = built(&op, &dirs);

        let wrapper = root.join("fake-cc.sh");
        std::fs::create_dir_all(&root).expect("临时目录");
        std::fs::write(
            &wrapper,
            format!("#!/bin/sh\nexec {} \"$@\"\n", tool.command),
        )
        .expect("写脚本");
        let mut perms = std::fs::metadata(&wrapper).expect("脚本").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
        std::fs::set_permissions(&wrapper, perms).expect("可执行");
        // 身份也是探测出来的：脚本 `--version` 转调真正的编译器，所以版本是真的，
        // 命令名是脚本路径——两者合起来就是这份产物的来历。
        let wrapper = toolchain::probe(&wrapper.to_string_lossy()).expect("脚本可探测");

        let (first, artifact) =
            execute(&profile, Some(&wrapper), &key, &harness, &dirs).expect("第一次应真的编译");
        assert!(!artifact.reused, "{artifact:?}");
        assert_eq!(first.bits.len(), sample.n);
        assert_eq!(
            artifact.key.tool,
            wrapper.id(),
            "产物身份必须带上实测工具链"
        );

        std::fs::remove_file(&wrapper.command).expect("删掉编译器");
        let (second, hit) =
            execute(&profile, Some(&wrapper), &key, &harness, &dirs).expect("应复用产物");
        assert!(hit.reused, "{hit:?}");
        assert_eq!(second.bits, first.bits, "复用同一份产物，结果必须逐位相同");

        // `None` 就是"这台机器上没有编译器"：产物旁的身份文件足以认回同一份文件。
        let (third, from_identity) =
            execute(&profile, None, &key, &harness, &dirs).expect("没有编译器也应复用产物");
        assert!(from_identity.reused, "{from_identity:?}");
        assert_eq!(from_identity.path, hit.path, "按身份找到的还是同一份产物");
        assert_eq!(third.bits, first.bits, "同一份产物，结果逐位相同");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn a_new_compiler_version_does_not_reuse_the_old_artifact() {
        // 同一份源文本、同一个主函数，只把 `--version` 换掉：必须重编。这是"产物身份带实测
        // 工具链"的可观测后果——此前身份只有命令名，升级编译器会命中旧代码生成。
        let Some(tool) = tool() else {
            return;
        };
        let root = std::env::temp_dir().join(format!("cutile-version-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dirs = dirs(&root);
        std::fs::create_dir_all(&root).expect("临时目录");
        let profile = cpu_profile();
        let op = builtin();
        let (key, harness, _) = built(&op, &dirs);

        let fake = |name: &str, version: &str| {
            let path = root.join(name);
            std::fs::write(
            &path,
            format!(
                "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'Fake clang version {version}'; exit 0; fi\nexec {} \"$@\"\n",
                tool.command
            ),
        )
        .expect("写脚本");
            let mut perms = std::fs::metadata(&path).expect("脚本").permissions();
            std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
            std::fs::set_permissions(&path, perms).expect("可执行");
            toolchain::probe(&path.to_string_lossy()).expect("脚本可探测")
        };
        let old = fake("cc17.sh", "17.0.0");
        let new = fake("cc21.sh", "21.0.0");
        assert_ne!(old.id(), new.id(), "前提：两条工具链身份不同");

        let (first, older) =
            execute(&profile, Some(&old), &key, &harness, &dirs).expect("第一次编译");
        assert!(!older.reused, "{older:?}");
        let (second, newer) =
            execute(&profile, Some(&new), &key, &harness, &dirs).expect("第二次编译");
        assert!(!newer.reused, "换工具链版本必须重新编译：{newer:?}");
        assert_ne!(older.path, newer.path, "两份产物不能共用路径");
        assert_eq!(first.bits, second.bits, "语义相同，结果必须逐位相同");

        // 再跑一次新版本：这次才该复用。
        let (_, again) = execute(&profile, Some(&new), &key, &harness, &dirs).expect("应复用");
        assert!(again.reused, "{again:?}");
        assert_eq!(again.path, newer.path);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// 换一组编译参数不许复用旧产物：见证没有 `CacheKey`，参数只能由调用方给；少了这一项，档案里改一次
    /// `[compile].flags` 就会复用一个不再会被编出来的程序。
    #[cfg(unix)]
    #[test]
    fn a_new_flag_set_does_not_reuse_the_witness_binary() {
        let Some(tool) = tool() else {
            return;
        };
        let root = std::env::temp_dir().join(format!("cutile-flags-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dirs = dirs(&root);
        let program = "#include <stdio.h>\nint main(void){ printf(\"0\\n\"); return 0; }\n";
        let args = |values: &[&str]| -> Vec<String> {
            values.iter().map(|value| (*value).to_string()).collect()
        };
        let o2 = args(&["-O2", "-std=c99"]);
        let o0 = args(&["-O0", "-std=c99"]);

        let (_, first) = run_program(Some(&tool), &o2, "flags", program, &dirs).expect("编译");
        assert!(!first.reused, "{first:?}");
        let (_, again) = run_program(Some(&tool), &o2, "flags", program, &dirs).expect("复用");
        assert!(again.reused, "同一组参数应复用：{again:?}");

        let (_, relaid) = run_program(Some(&tool), &o0, "flags", program, &dirs).expect("编译");
        assert!(!relaid.reused, "换参数必须重新编译：{relaid:?}");
        assert_ne!(relaid.path, again.path, "两份产物不能共用路径");
        assert_ne!(
            relaid.key.params, again.key.params,
            "参数必须是产物身份的一部分"
        );

        // 没有编译器时也不许拿另一组参数的产物顶替：那不是"同一份产物"的副本。
        let error = run_program(None, &args(&["-O1"]), "flags", program, &dirs).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported, "{error}");
        assert!(error.to_string().contains("编译参数"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_artifact_cache_leaves_the_work_directory_empty() {
        let Some(tool) = tool() else {
            return;
        };
        let root = std::env::temp_dir().join(format!("cutile-sandbox-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dirs = dirs(&root);
        let op = builtin();
        let (key, harness, _) = built(&op, &dirs);
        execute(&cpu_profile(), Some(&tool), &key, &harness, &dirs).expect("应能编译执行");
        let leftovers: Vec<_> = std::fs::read_dir(dirs.work())
            .map(|items| items.flatten().map(|item| item.file_name()).collect())
            .unwrap_or_default();
        assert!(
            leftovers.is_empty(),
            "产物拷进缓存后沙箱就该删掉，否则每次编译都留一份可执行文件：{leftovers:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn without_a_compiler_a_missing_artifact_is_an_explicit_failure() {
        let root = std::env::temp_dir().join(format!("cutile-no-artifact-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dirs = dirs(&root);
        let op = builtin();
        let (key, harness, _) = built(&op, &dirs);

        let error = execute(&cpu_profile(), None, &key, &harness, &dirs).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported, "{error}");
        assert!(error.to_string().contains("没有这份产物"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn two_compilers_worth_of_artifacts_are_not_chosen_between() {
        // 没有编译器时只在"唯一一份"产物上复用：多份候选就是替用户选编译器，也就替 `bench`
        // 决定了耗时的来历。用两份假产物把这条判定单独钉住（不需要真编译器）。
        let root = std::env::temp_dir().join(format!("cutile-ambiguous-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dirs = dirs(&root);
        let op = builtin();
        let (key, harness, _) = built(&op, &dirs);
        let cache = Cache::from_dirs(&dirs);

        let exe = root.join("fake-exe");
        std::fs::create_dir_all(&root).expect("临时目录");
        std::fs::write(&exe, b"#!/bin/sh\nexit 0\n").expect("写假产物");
        let mut perms = std::fs::metadata(&exe).expect("产物").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o755);
        std::fs::set_permissions(&exe, perms).expect("可执行");
        for candidate in ["cc@17.0.0", "clang@18.1.3"] {
            cache
                .put_artifact(
                    &ArtifactKey::keyed(key.hex(), candidate, key.flags_hash, &harness),
                    &exe,
                )
                .expect("落产物");
        }

        let error = execute(&cpu_profile(), None, &key, &harness, &dirs).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported, "{error}");
        assert!(error.to_string().contains("cc@17.0.0"), "{error}");
        assert!(error.to_string().contains("clang@18.1.3"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }
}

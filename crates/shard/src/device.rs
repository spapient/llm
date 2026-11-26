//! 宿主 C 参考后端 + 设备后端注册表。

use crate::backend::{Backend, Codegen, Driver, Readback};
use crate::build;
use crate::config::config;
use crate::runner::driver;
use crate::toolchain;
use crate::witness::{MAX_OUT, Scope, Selected};
use cutile::dirs::Dirs;
use cutile::prof::{Emit, Profile};
use diagnostic::{Code, Error, Result};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// 宿主 C 参考后端：本机 CPU 永远在，证明「这段 C 编得过、跑得出 `expect`」，不是设备证据。
struct CBackend;

impl Codegen for CBackend {
    fn emit(&self) -> Emit {
        Emit::C
    }
    fn program(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        selected: &[Selected<'_>],
        scope: Scope,
    ) -> Result<String> {
        c_program(profile, dirs, selected, scope)
    }
}

impl Driver for CBackend {
    fn available(&self, profile: &Profile) -> bool {
        c_available(profile)
    }
    fn run(
        &self,
        profile: &Profile,
        dirs: &Dirs,
        program: &str,
        max: usize,
        scope: Scope,
    ) -> Result<Readback> {
        c_run(profile, dirs, program, max, scope)
    }
    fn toolchain(&self) -> Option<toolchain::Toolchain> {
        toolchain::detect()
    }
}

impl Backend for CBackend {}

/// 后端注册表：cutile 内置 host C 参考后端，Metal / CUDA 由 `engine` 经 [`crate::backend::register`]
/// 注入。进程级一张表，机制层按 `emit` 取，不硬编码后端清单。
static REGISTRY: OnceLock<Mutex<Vec<&'static dyn Backend>>> = OnceLock::new();

fn registry() -> &'static Mutex<Vec<&'static dyn Backend>> {
    REGISTRY.get_or_init(|| Mutex::new(vec![&CBackend]))
}

/// 注册一台后端；同一条后端重复注册只留一份（幂等）。注册表锁若已中毒，本次注册静默放弃——
/// 后端注入发生在启动期，宁可少一台也不 panic。
pub(crate) fn register(backend: &'static dyn Backend) {
    let Ok(mut table) = registry().lock() else {
        return;
    };
    if !table.iter().any(|entry| entry.emit() == backend.emit()) {
        table.push(backend);
    }
}

/// 按 `emit` 找设备后端；找不到就是「本机没有这门语言的执行前端」。
pub(crate) fn spec(emit: Emit) -> Option<&'static dyn Backend> {
    let Ok(table) = registry().lock() else {
        return None;
    };
    table.iter().find(|entry| entry.emit() == emit).copied()
}

/// 已登记的全部执行语言：错误报告里列出「本机接了哪些」。
pub(crate) fn specs() -> Vec<&'static dyn Backend> {
    let Ok(table) = registry().lock() else {
        return Vec::new();
    };
    table.clone()
}

// ---------------------------------------------------------------------------
// C（host CPU）编排：见证本身就是一段 host C，`cc` 把它与 harness 编成可执行文件跑一遍。
// 它证明「这段 C 在这个 ABI 下编得过、跑得出 `expect`」，不证明设备具备该能力。
// ---------------------------------------------------------------------------

/// C 编排的编译输入：前置声明 + 参数。自检只走 host C ABI，理由见 [`HOST_FLAGS`]。
fn c_inputs(profile: &Profile, scope: Scope) -> (String, Vec<String>) {
    match scope {
        Scope::Device => (profile.preamble.clone(), profile.flags.clone()),
        Scope::Host => (String::new(), config().host_flags.clone()),
    }
}

fn c_available(_: &Profile) -> bool {
    // `c` 的设备就是本机 CPU：它永远在。
    true
}

fn c_program(
    profile: &Profile,
    dirs: &Dirs,
    selected: &[Selected<'_>],
    scope: Scope,
) -> Result<String> {
    let mut program = c_head(profile, scope, selected);
    driver::push_sources(
        &mut program,
        &driver::witness_files(selected, dirs, driver::C_PARAMS)?,
    );
    program.push_str(&c_harness::<MAX_OUT>(selected));
    Ok(program)
}

fn c_run(
    profile: &Profile,
    dirs: &Dirs,
    program: &str,
    _count: usize,
    scope: Scope,
) -> Result<Readback> {
    let (_, flags) = c_inputs(profile, scope);
    // 设备执行与自检是两个不同的缓存命名空间：同一段文本在两套参数下编出的产物不同。
    let tag = match scope {
        Scope::Device => "witness",
        Scope::Host => "self",
    };
    // 见证程序也进产物缓存：第二次 `verify` 只在第一次调用编译器。
    let (outcome, _artifact) =
        build::run_program(toolchain::detect().as_ref(), &flags, tag, program, dirs)?;
    // stdout 已由 `runner` 解码成 f32，这里只补上 stderr 里的写入个数。
    Ok((driver::parse_counts(&outcome.stderr), outcome.bits))
}

/// C 前端的头：档案前置声明 + `stdio.h`；自检时再吞掉 MSL 的地址空间限定符。`#define thread` 只服务
/// 于自检，设备前端走真正的 `xcrun metal`，所以这里的放宽不会被当成设备证据。
fn c_head(profile: &Profile, scope: Scope, selected: &[Selected<'_>]) -> String {
    let (mut head, _) = c_inputs(profile, scope);
    let metal = selected
        .iter()
        .any(|(_, _, witness)| witness.file.ends_with(".metal"));
    driver::normalize_newline(&mut head);
    head.push_str("#include <stdio.h>\n\n");
    if scope == Scope::Host && metal {
        head.push_str("#define thread\n");
    }
    head
}

/// C 前端的内核包装与主函数：`main()` 逐条调用见证函数，值走 stdout，个数走 stderr。
fn c_harness<const OUT: usize>(selected: &[Selected<'_>]) -> String {
    let mut text = format!("\nint main(void) {{\n    float out[{OUT}];\n    int n;\n");
    // 每条见证是一句互不相同的调用语句，顺序由见证清单决定：这里只能逐条生成，不能批处理。
    for (_, _, witness) in selected {
        push_witness(&mut text, &witness.func, OUT);
    }
    text.push_str("    return 0;\n}\n");
    text
}

/// 一次见证调用：算一遍、报个数、按写出的个数打印位模式；`max` 是 const 泛型传下来的字面量，截断在生成的代码里是常数比较。
fn push_witness(text: &mut String, func: &str, max: usize) {
    text.push_str("    n = ");
    text.push_str(func);
    text.push_str("(out);\n");
    text.push_str("    fprintf(stderr, \"witness %d\\n\", n);\n");
    text.push_str("    if (n < 0) n = 0;\n");
    text.push_str(&format!("    if (n > {max}) n = {max};\n"));
    text.push_str("    for (int i = 0; i < n; ++i) {\n");
    text.push_str("        union { float f; unsigned int u; } v;\n");
    text.push_str("        v.f = out[i];\n");
    text.push_str("        printf(\"%u\\n\", v.u);\n");
    text.push_str("    }\n");
}

// ---- Metal host 驱动与 OpenCL 设备实现 ----

const METAL_HEAD: &str = r#"#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: driver <metallib> <count>\n");
            return 2;
        }
        const int count = atoi(argv[2]);
"#;

const METAL_TAIL: &str = r#"        id<MTLBuffer> values = [device newBufferWithLength:(NSUInteger)count * WITNESS_MAX_OUT * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> counts = [device newBufferWithLength:(NSUInteger)count * sizeof(int)
                                                   options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:values offset:0 atIndex:0];
        [encoder setBuffer:counts offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)count, 1, 1)];
        [encoder endEncoding];
        [buffer commit];
        [buffer waitUntilCompleted];
        if (buffer.error) {
            fprintf(stderr, "内核执行失败：%s\n", buffer.error.localizedDescription.UTF8String);
            return 7;
        }
        const int *written = (const int *)counts.contents;
        const float *out = (const float *)values.contents;
        for (int i = 0; i < count; ++i) {
            fprintf(stderr, "witness %d\n", written[i]);
        }
        fflush(stderr);
        for (int i = 0; i < count; ++i) {
            int n = written[i];
            if (n < 0) n = 0;
            if (n > WITNESS_MAX_OUT) n = WITNESS_MAX_OUT;
            for (int j = 0; j < n; ++j) {
                union { float f; unsigned int u; } v;
                v.f = out[i * WITNESS_MAX_OUT + j];
                printf("%u\n", v.u);
            }
        }
    }
    return 0;
}
"#;

/// 生成 Metal 见证 host 驱动；写出上限由 `max` 参数注入。
#[must_use]
pub fn metal_driver(max: usize) -> String {
    let library = r#"        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "本机没有 Metal 设备\n"); return 3; }
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) { fprintf(stderr, "metallib 装载失败：%s\n", error.localizedDescription.UTF8String); return 4; }
"#;
    format!(
        "#define WITNESS_MAX_OUT {max}\n{METAL_HEAD}{library}        id<MTLFunction> function = [library newFunctionWithName:@\"cutile_witness\"];
        if (!function) {{ fprintf(stderr, \"metallib 里没有 cutile_witness\\n\"); return 5; }}
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {{ fprintf(stderr, \"管线创建失败：%s\\n\", error.localizedDescription.UTF8String); return 6; }}
{METAL_TAIL}"
    )
}

#[cfg(windows)]
pub(crate) mod opencl {
    use super::*;
    use crate::toolchain::Toolchain;
    use prof::device::opencl::{OpenClBuf, OpenClDevice};

    pub(crate) struct OpenClBackend;

    impl Codegen for OpenClBackend {
        fn emit(&self) -> Emit {
            Emit::Opencl
        }

        fn program(
            &self,
            profile: &Profile,
            dirs: &Dirs,
            selected: &[Selected<'_>],
            _scope: Scope,
        ) -> Result<String> {
            let mut program = profile.preamble.clone();
            driver::normalize_newline(&mut program);
            program.push_str("#define CUTILE_INTEL_DEVICE 1\n");
            driver::push_sources(
                &mut program,
                &driver::witness_files(selected, dirs, driver::C_PARAMS)?,
            );
            program.push_str(&format!(
                "__kernel void cutile_witness(__global float *out, __global int *counts) {{\n\
                 \x20   int tid = get_group_id(0);\n\
                 \x20   int lane = get_local_id(0);\n\
                 \x20   out += tid * {MAX_OUT};\n\
                 \x20   int n = 0;\n\
                 \x20   switch (tid) {{\n"
            ));
            for (idx, (prim, _, witness)) in selected.iter().enumerate() {
                let cooperative = matches!(
                    *prim,
                    cutile::atom::Prim::Shuffle
                        | cutile::atom::Prim::Broadcast
                        | cutile::atom::Prim::Scan
                );
                let guard = if cooperative { "" } else { "if (lane == 0) " };
                program.push_str(&format!(
                    "case {idx}: {guard}n = {}(out); break;\n",
                    witness.func
                ));
            }
            program.push_str(&format!(
                "default: break;\n}}\n\
                 \x20   if (lane == 0) counts[tid] = n < 0 ? 0 : (n > {MAX_OUT} ? {MAX_OUT} : n);\n}}\n"
            ));
            Ok(program)
        }
    }

    fn bufs(device: &OpenClDevice, total: usize) -> Result<(OpenClBuf, OpenClBuf)> {
        let output = device.alloc(total * MAX_OUT)?;
        let counts = device.alloc_i32(total)?;
        Ok((output, counts))
    }

    fn readback(device: &OpenClDevice, output: &OpenClBuf, counts: &OpenClBuf) -> Result<Readback> {
        let counts = device.read_i32(counts)?;
        let values = device.read_f32(output)?;
        let readback = counts
            .iter()
            .enumerate()
            .flat_map(|(idx, &count)| {
                let count = usize::try_from(count).unwrap_or(0).min(MAX_OUT);
                let start = idx * MAX_OUT;
                values[start..start + count].iter().copied()
            })
            .collect();
        Ok((counts.into_iter().map(i64::from).collect(), readback))
    }

    impl Driver for OpenClBackend {
        fn available(&self, _: &Profile) -> bool {
            OpenClDevice::probe().is_some()
        }

        fn run(
            &self,
            profile: &Profile,
            _: &Dirs,
            program: &str,
            total: usize,
            _: Scope,
        ) -> Result<Readback> {
            let device = OpenClDevice::new()?;
            let kernel = device.compile(program, &profile.flags.join(" "), "cutile_witness")?;
            let (output, counts) = bufs(&device, total)?;
            device.launch(
                &kernel,
                &[&output, &counts],
                [total * 16, 1, 1],
                Some([16, 1, 1]),
            )?;
            readback(&device, &output, &counts)
        }

        fn toolchain(&self) -> Option<Toolchain> {
            OpenClDevice::probe().map(|info| Toolchain {
                command: "OpenCL".to_string(),
                version: Some(info.driver),
                banner: format!("{} · {}", info.name, info.version),
            })
        }
    }

    impl Backend for OpenClBackend {}
}

// ---- 文件槽存储：集合算法由 collective 统一提供 ----

/// 文件传输集合通信：共享文件承载 `world × width` 个小端 `f32` 槽位。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRemote {
    rank: u32,
    world: u32,
    width: usize,
    bytes: u64,
    backing: PathBuf,
}

impl FileRemote {
    /// 打开一个 rank 的文件句柄，并将共享文件预留到完整长度。
    ///
    /// # Errors
    ///
    /// `world` 或 `width` 为零、rank 越界、文件长度溢出或文件不可创建。
    pub fn open(rank: u32, world: u32, width: usize, backing: impl Into<PathBuf>) -> Result<Self> {
        Self::check_open_config(rank, world, width)?;
        let bytes = Self::file_bytes(world, width)?;
        let remote = Self {
            rank,
            world,
            width,
            bytes,
            backing: backing.into(),
        };
        remote.resize()?;
        Ok(remote)
    }

    /// 打开配置只负责进程拓扑与非零宽度，和文件字节换算分离。
    fn check_open_config(rank: u32, world: u32, width: usize) -> Result<()> {
        if world == 0 {
            fail!("shard::FileRemote::open", "进程数 world 不能为 0");
        }
        if rank >= world {
            fail!("shard::FileRemote::open", "rank {rank} 超出 0..{world}");
        }
        if width == 0 {
            fail!("shard::FileRemote::open", "归约宽度不能为 0");
        }
        Ok(())
    }

    /// 共享文件包含 `world × width` 个 `f32`，所有换算必须受溢出保护。
    fn file_bytes(world: u32, width: usize) -> Result<u64> {
        let slot_bytes = width
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| invalid!("shard::FileRemote::open", "槽位字节数溢出"))?;
        u64::try_from(slot_bytes)
            .ok()
            .and_then(|slot| slot.checked_mul(u64::from(world)))
            .ok_or_else(|| invalid!("shard::FileRemote::open", "共享文件字节数溢出"))
    }

    accessors! {
        /// 本进程的 rank。
        rank -> u32 = |s| s.rank;
    }

    /// 将某个 rank 的输入写入对应槽位。
    ///
    /// # Errors
    ///
    /// 长度不符、rank 越界或文件不可写。
    pub fn load(&self, rank: u32, values: &[f32]) -> Result<()> {
        self.check_width("shard::FileRemote::load", values.len())?;
        self.check_rank("shard::FileRemote::load", rank)?;
        self.write_slot(rank, values)
    }

    /// 读取某个 rank 的当前槽位。
    ///
    /// # Errors
    ///
    /// rank 越界或文件不可读。
    pub fn replica(&self, rank: u32) -> Result<Vec<f32>> {
        self.check_rank("shard::FileRemote::replica", rank)?;
        let mut slot = vec![0.0; self.width];
        self.read_slot(rank, &mut slot)?;
        Ok(slot)
    }

    pub(crate) const fn width(&self) -> usize {
        self.width
    }

    fn check_rank(&self, site: &'static str, rank: u32) -> Result<()> {
        if rank < self.world {
            return Ok(());
        }
        refuse_at!(site, "rank {rank} 超出 0..{}", self.world)
    }

    /// 槽宽校验：`load` 的输入长度必须等于本镜像的归约宽度。
    fn check_width(&self, site: &'static str, len: usize) -> Result<()> {
        if len == self.width {
            return Ok(());
        }
        refuse_at!(site, "宽度 {len} 与归约宽度 {} 不一致", self.width)
    }

    fn offset(&self, rank: u32) -> u64 {
        u64::from(rank) * (self.width as u64) * size_of::<f32>() as u64
    }

    fn io(&self, action: &str, error: std::io::Error) -> Error {
        Error::new(
            Code::Io,
            format!("{action} `{}` 失败：{error}", self.backing.display()),
        )
    }

    fn seek_slot(&self, file: &mut File, rank: u32) -> Result<()> {
        file.seek(SeekFrom::Start(self.offset(rank)))
            .map_err(|error| self.io("定位", error))
            .map(|_| ())
    }

    fn resize(&self) -> Result<()> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&self.backing)
            .map_err(|error| self.io("创建", error))?
            .set_len(self.bytes)
            .map_err(|error| self.io("预留", error))
    }

    pub(crate) fn read_slot(&self, rank: u32, out: &mut [f32]) -> Result<()> {
        let bytes = self.read_bytes(rank)?;
        for (slot, chunk) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *slot = f32::from_le_bytes(*chunk);
        }
        Ok(())
    }

    /// 文件定位与完整读取共用一条路径，调用方只负责协议解码。
    fn read_bytes(&self, rank: u32) -> Result<Vec<u8>> {
        self.check_rank("shard::FileRemote::read", rank)?;
        let mut file = File::open(&self.backing).map_err(|error| self.io("打开", error))?;
        self.seek_slot(&mut file, rank)?;
        let mut bytes = vec![0u8; self.width * size_of::<f32>()];
        file.read_exact(&mut bytes)
            .map_err(|error| self.io("读", error))?;
        Ok(bytes)
    }

    pub(crate) fn write_slot(&self, rank: u32, values: &[f32]) -> Result<()> {
        self.check_rank("shard::FileRemote::write", rank)?;
        let mut file = OpenOptions::new()
            .write(true)
            .open(&self.backing)
            .map_err(|error| self.io("打开", error))?;
        self.seek_slot(&mut file, rank)?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        file.write_all(&bytes).map_err(|error| self.io("写", error))
    }
}

#[cfg(test)]
mod tests {
    //! `cutile::device` 的用例：设备枚举与能力查询。
    //!
    //! 生产文件只留执行与判据，用例集中于此。

    use super::*;
    use crate::fixtures::catalogue;

    /// 一条 mock 后端（SPIRV，本机必然缺席）：只用来验注册表的「去重 + 按 emit 分派」机制，不碰任何
    /// 厂商工具链。真机 Metal / CUDA 后端在 `engine`（L6），其 Codegen / Driver 测试也搬到了那边。
    struct Mock;

    impl Codegen for Mock {
        fn emit(&self) -> Emit {
            Emit::Spirv
        }
        fn program(
            &self,
            _: &Profile,
            _: &Dirs,
            _: &[Selected<'_>],
            _: Scope,
        ) -> diagnostic::Result<String> {
            Ok(String::new())
        }
    }

    impl Driver for Mock {
        fn available(&self, _: &Profile) -> bool {
            false
        }
        fn run(
            &self,
            _: &Profile,
            _: &Dirs,
            _: &str,
            _: usize,
            _: Scope,
        ) -> diagnostic::Result<Readback> {
            Ok((Vec::new(), Vec::new()))
        }
        fn toolchain(&self) -> Option<crate::toolchain::Toolchain> {
            None
        }
    }

    impl Backend for Mock {}

    static MOCK: Mock = Mock;

    testkit::test_cases! {
    /// 生成 host C 参考后端的见证源码（设备文本 + 设备档案的 host 自检文本）。默认只断言非空；
    /// `CUTILE_CAPTURE_GOLDEN` 显式打开时才把副本写到 `/tmp/cutile-golden` 供人工比对。
    /// Metal / CUDA 的设备文本由 `engine` 生成，不再从 cutile 的注册表里取。
    fn golden_capture() {
        let capture = std::env::var("CUTILE_CAPTURE_GOLDEN").is_ok();
        let dir = std::path::PathBuf::from("/tmp/cutile-golden");
        let dirs = Dirs::under(std::env::temp_dir().join("cutile-golden-work"));
        let catalogue = catalogue();

        let cpu = catalogue.profile("cpu").cloned().expect("cpu 档案");
        let front = spec(Emit::C).expect("c 前端");
        let selected = crate::witness::declared_witnesses(&cpu).expect("见证");
        let text = front
            .program(&cpu, &dirs, &selected, Scope::Device)
            .expect("可生成");
        assert!(!text.is_empty(), "cpu 生成文本不能为空");
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join("witness_c.txt"), text).expect("写");
        }

        let metal = catalogue.profile("metal").cloned().expect("metal 档案");
        let selected = crate::witness::declared_witnesses(&metal).expect("见证");
        let host = front
            .program(&metal, &dirs, &selected, Scope::Host)
            .expect("可生成");
        assert!(!host.is_empty(), "metal 档案的 host 自检文本不能为空");
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join("witness_metal_host.txt"), host).expect("写");
        }
    }

    fn scratch_directories_do_not_collide() {
        let dirs = Dirs::under(std::env::temp_dir());
        let first = crate::runner::driver::scratch(&dirs, "gate");
        let second = crate::runner::driver::scratch(&dirs, "gate");
        assert_ne!(first, second);
        assert_eq!(first.parent(), Some(dirs.work()));
    }

    /// 执行后端注册表：宿主 C 参考后端种子在场、每个 `emit` 恰好一行、同 emit 重复注册被去重（
    /// `spec(..)` 命中第一行，后一行静默失效），表不许被清空。
    fn specs_register_each_execution_frontend_once() {
        assert!(spec(Emit::C).is_some(), "宿主 C 参考后端必须种子注册");
        crate::device::register(&MOCK);
        crate::device::register(&MOCK);

        let mut seen = std::collections::BTreeSet::new();
        for entry in specs() {
            assert!(
                seen.insert(entry.emit()),
                "执行前端 `{}` 在注册表里重复登记：后一行会被 `spec(..)` 静默忽略",
                entry.emit().as_str()
            );
            assert!(
                spec(entry.emit()).is_some(),
                "表内 `{}` 的查找行与登记表分家",
                entry.emit().as_str()
            );
        }
        assert!(!seen.is_empty(), "执行前端注册表不得被清空");
    }

    fn scratch_action_cleans_up_on_success_and_failure() {
        let root = std::env::temp_dir().join(format!("shard-scratch-cleanup-{}", std::process::id()));
        let dirs = Dirs::under(&root);
        let mut success_path = None;
        let value = crate::runner::driver::with_scratch(&dirs, "success", |dir| {
            success_path = Some(dir.to_path_buf());
            std::fs::create_dir_all(dir).expect("临时目录");
            Ok(7)
        });
        assert_eq!(value.expect("成功值"), 7);
        assert!(!success_path.expect("路径").exists(), "成功后必须清理");

        let mut failure_path = None;
        let result: diagnostic::Result<()> =
            crate::runner::driver::with_scratch(&dirs, "failure", |dir| {
                failure_path = Some(dir.to_path_buf());
                std::fs::create_dir_all(dir).expect("临时目录");
                Err(diagnostic::Error::new(
                    diagnostic::Code::Invalid,
                    "故意失败",
                ))
            });
        assert!(result.is_err());
        assert!(!failure_path.expect("路径").exists(), "失败后也必须清理");
        let _ = std::fs::remove_dir_all(root);
    }
    }
}

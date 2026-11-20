//! Metal 设备边界：手写 FFI，设备、命令队列、共享缓冲、计算管线句柄与 [`MetalExecutor`]（M3 第二刀）。
//!
//! 越界理由：`MTLCreateSystemDefaultDevice` 与 Objective-C 消息派发（`objc_msgSend`）触及平台 FFI，
//! 属于 ADR 0035 §1「FFI 调用」白名单。与 `cutile` 的差异：`cutile` 生成 ObjC driver 交给 clang 编译、
//! 起子进程执行，本身零 unsafe；而 `Executor::submit` / `Allocator::alloc` 是进程内、有状态、逐步调用，
//! 必须在本进程内直接向 Metal 对象发消息，因此这条边界走手写 `objc_msgSend`（ADR 0035 §3 FFI 默认策略）。
//!
//! 安全不变量：
//! - `MTLCreateSystemDefaultDevice`、`newCommandQueue`、`newBufferWithLength:options:`、`newLibraryWithSource:options:error:`、
//!   `newFunctionWithName:`、`newComputePipelineStateWithFunction:error:` 都返回 **+1 保留** 的对象，
//!   句柄 `Drop` 时恰好 `-release` 一次；编译路径里 `library` / `function` 在管线建立后立即释放（管线已自包含）；
//! - 句柄指针私有、不暴露、不跨线程；`objc_msgSend` 按形状逐条登记强转（`(id, SEL) -> id` / `-> void` /
//!   带 `usize`、指针或结构体参数）；`dispatchThreads:threadsPerThreadgroup:` 的两个 `MTLSize`（24 字节，
//!   `#[repr(C)]`）在 arm64 / x86_64 下按间接指针传参，与 clang 的 `objc_msgSend` 调用 ABI 一致；
//! - `commandBuffer` / `computeCommandEncoder` 返回自动释放对象，`execute` 全程包在
//!   `objc_autoreleasePoolPush` / `Pop` 内，`waitUntilCompleted` 返回后池弹栈统一释放，不手动 `release`；
//! - 缓冲取 `MTLResourceStorageModeShared`（统一内存 CPU/GPU 双向可见），`contents` 指针在 `release` 前
//!   稳定有效、读写走同一块内存；不预写 `Managed` / `Private` 存储模式；
//! - 编译路径用 `objc_autoreleasePoolPush` / `Pop` 包住 `stringWithUTF8String:`（返回自动释放对象），
//!   `NSString` 生命周期与池同界，不在 Rust 侧手动 `release` 自动释放对象。
//!
//! 全仓默认 `deny(unsafe_code)`；本模块是 `engine` 内的越界岛（同 `storage::mmap` 形态），对整个模块
//! 放开、只在 macOS 目标编译（ADR 0035 §2）。设备后端不单独成 crate：`Executor` / `Program` / `Device`
//! 的主人就是 `engine`，Metal 实现与 trait 同住一 crate，跨成员依赖边与所有权登记各少一份。

#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};
use std::ptr::null_mut;

use diagnostic::{Code, Error, Result};

use super::exec::{Device, Executor, Graph, Node, Queued, Stream, check_operands, enqueue};

// Objective-C runtime：消息发送、selector 注册、类查找与自动释放池（`libobjc`）。
#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    fn objc_msgSend();
    fn sel_registerName(name: *const c_char) -> *mut c_void;
    fn objc_getClass(name: *const c_char) -> *mut c_void;
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

// Metal.framework：创建系统默认 GPU 设备，失败返回 null（+1 保留）。
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
}

/// Metal 三维尺寸：`dispatchThreads:threadsPerThreadgroup:` 的两个按值参数（`MTLSize`）。
#[repr(C)]
#[derive(Clone, Copy)]
struct MTLSize {
    width: usize,
    height: usize,
    depth: usize,
}

/// 生成 `objc_msgSend` 的按形状强转包装：把裸 `objc_msgSend` 转成一个具体签名的
/// `extern "C" fn` 再调用。所有形状共享同一份 ABI 纪律——`(id, SEL)` 打头，余下参数按值或
/// 按指针，返回 `id`（对象/指针）或 `void`。每个形状对应的 ObjC 方法在调用点标注。
macro_rules! msg {
    ($name:ident, [$($arg:ident: $ty:ty),*]) => {
        unsafe fn $name(
            receiver: *mut c_void,
            selector: *mut c_void,
            $($arg: $ty,)*
        ) -> *mut c_void {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void $(, $ty)*) -> *mut c_void =
                unsafe { std::mem::transmute(objc_msgSend as *const ()) };
            unsafe { send(receiver, selector $(, $arg)*) }
        }
    };
    (void $name:ident, [$($arg:ident: $ty:ty),*]) => {
        unsafe fn $name(receiver: *mut c_void, selector: *mut c_void, $($arg: $ty,)*) {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void $(, $ty)*) =
                unsafe { std::mem::transmute(objc_msgSend as *const ()) };
            unsafe { send(receiver, selector $(, $arg)*) };
        }
    };
}

// `(id, SEL) -> id`：0 参返回对象/指针（如 newCommandQueue）。
msg!(msg_id, []);
// `(id, SEL) -> void`：无返回值（如 -release）。
msg!(void msg_void, []);
// `(id, SEL, usize, usize) -> id`：两个 NSUInteger 参数（如 newBufferWithLength:options:）。
msg!(msg_id_2u, [a: usize, b: usize]);
// `(id, SEL, p) -> id`：一个指针参数（如 stringWithUTF8String:）。
msg!(msg_id_1p, [p: *mut c_void]);
// `(id, SEL, p1, p2) -> id`：两个指针参数（如 newComputePipelineStateWithFunction:error:）。
msg!(msg_id_2p, [p1: *mut c_void, p2: *mut c_void]);
// `(id, SEL, p1, p2, p3) -> id`：三个指针参数。
msg!(msg_id_3p, [p1: *mut c_void, p2: *mut c_void, p3: *mut c_void]);
// `(id, SEL, p) -> void`：一个指针参数的无返回值方法。
msg!(void msg_void_1p, [p: *mut c_void]);
// `(id, SEL, p, usize, usize) -> void`：setBuffer:offset:atIndex:。
msg!(void msg_void_p2u, [p: *mut c_void, a: usize, b: usize]);
// `(id, SEL, MTLSize, MTLSize) -> void`：dispatchThreads:threadsPerThreadgroup:。
// MTLSize 24 字节（>16），arm64/x86_64 下 extern "C" 按间接指针传参，无需 objc_msgSend_stret。
msg!(void msg_void_2size, [a: MTLSize, b: MTLSize]);

/// 注册一个 selector；`name` 是带结尾 `\0` 的 C 字符串。
fn selector(name: &[u8]) -> *mut c_void {
    unsafe { sel_registerName(name.as_ptr().cast()) }
}

/// 释放一个 ObjC 对象（`-release`），只在 `Drop` 调用。
unsafe fn release(object: *mut c_void) {
    unsafe { msg_void(object, selector(b"release\0")) };
}

/// 用 `stringWithUTF8String:` 构造一个自动释放的 `NSString`；只在编译路径的池内调用。
unsafe fn nsstring(cstr: *const c_char) -> *mut c_void {
    let class = unsafe { objc_getClass(c"NSString".as_ptr()) };
    unsafe {
        msg_id_1p(
            class,
            selector(b"stringWithUTF8String:\0"),
            cstr as *mut c_void,
        )
    }
}

/// 本机 Metal 默认设备句柄（+1 保留，`Drop` 释放）。
pub struct MetalDevice {
    raw: *mut c_void,
}

impl MetalDevice {
    /// 取系统默认设备；本机无 Metal 时返回 `None`。
    pub fn system_default() -> Option<Self> {
        let raw = unsafe { MTLCreateSystemDefaultDevice() };
        (!raw.is_null()).then_some(Self { raw })
    }

    /// 新建一条命令队列（+1 保留，`Drop` 释放）。
    pub fn new_command_queue(&self) -> Option<MetalCommandQueue> {
        let raw = unsafe { msg_id(self.raw, selector(b"newCommandQueue\0")) };
        (!raw.is_null()).then_some(MetalCommandQueue { raw })
    }

    /// 分配一条共享模式缓冲（+1 保留，`Drop` 释放）；`MTLResourceStorageModeShared` 使 CPU 可直接读写。
    pub fn new_buffer(&self, bytes: usize) -> Option<MetalBuffer> {
        // `newBufferWithLength:options:`：长度 + 存储模式（0 = Shared）。
        let raw = unsafe {
            msg_id_2u(
                self.raw,
                selector(b"newBufferWithLength:options:\0"),
                bytes,
                0,
            )
        };
        if raw.is_null() {
            return None;
        }
        let contents = unsafe { msg_id(raw, selector(b"contents\0")) };
        Some(MetalBuffer {
            raw,
            contents: contents.cast(),
            len: bytes,
        })
    }

    /// 把一段 MSL 源码编译成计算管线（+1 保留，`Drop` 释放）；入口函数名固定 `kernel_main`。
    ///
    /// 编译失败（源码非法 / 无 `kernel_main`）返回 `None`；本刀不把 `NSError` 文本带回 Rust 侧。
    pub fn compile(&self, source: &str) -> Option<MetalPipeline> {
        let src = CString::new(source).ok()?;
        let fname = CString::new("kernel_main").ok()?;
        // `stringWithUTF8String:` 返回自动释放对象，编译路径包一层池，池弹栈时统一释放。
        let pool = unsafe { objc_autoreleasePoolPush() };
        let result = self.compile_with_pool(src.as_ptr(), fname.as_ptr());
        unsafe { objc_autoreleasePoolPop(pool) };
        result
    }

    /// 池内的编译主体：`newLibraryWithSource → newFunctionWithName → newComputePipelineState`。
    fn compile_with_pool(&self, src: *const c_char, fname: *const c_char) -> Option<MetalPipeline> {
        unsafe {
            let source = nsstring(src);
            let library = msg_id_3p(
                self.raw,
                selector(b"newLibraryWithSource:options:error:\0"),
                source,
                null_mut(),
                null_mut(),
            );
            if library.is_null() {
                return None;
            }
            let function_name = nsstring(fname);
            let function = msg_id_1p(library, selector(b"newFunctionWithName:\0"), function_name);
            if function.is_null() {
                release(library);
                return None;
            }
            let pipeline = msg_id_2p(
                self.raw,
                selector(b"newComputePipelineStateWithFunction:error:\0"),
                function,
                null_mut(),
            );
            // 管线建立后 library / function 即自包含，立即释放，只持有 pipeline。
            release(library);
            release(function);
            (!pipeline.is_null()).then_some(MetalPipeline { raw: pipeline })
        }
    }
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// 一条 Metal 命令队列句柄（+1 保留，`Drop` 释放）。
pub struct MetalCommandQueue {
    raw: *mut c_void,
}

impl MetalCommandQueue {
    /// 执行一条计算管线：把 `buffers` 按数组下标绑定（`setBuffer:offset:atIndex:`，offset=0），
    /// 以 `(grid, block)` 几何派发（`dispatchThreads:threadsPerThreadgroup:`），`commit` 后
    /// `waitUntilCompleted` 同步等完。
    ///
    /// 成功返回 `true`；任一步拿不到命令缓冲/编码器返回 `false`。缓冲走共享存储，
    /// `waitUntilCompleted` 返回后 CPU 侧 `contents` 即可读到 GPU 写入结果。
    pub fn execute(
        &self,
        pipeline: &MetalPipeline,
        buffers: &[&MetalBuffer],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> bool {
        // `commandBuffer` / `computeCommandEncoder` 返回自动释放对象，包一层池统一释放。
        let pool = unsafe { objc_autoreleasePoolPush() };
        let ok = self.execute_with_pool(pipeline, buffers, grid, block);
        unsafe { objc_autoreleasePoolPop(pool) };
        ok
    }

    /// 池内的派发主体：`commandBuffer → computeCommandEncoder → setPipeline → setBuffers →
    /// dispatchThreads → endEncoding → commit → waitUntilCompleted`。
    fn execute_with_pool(
        &self,
        pipeline: &MetalPipeline,
        buffers: &[&MetalBuffer],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> bool {
        unsafe {
            let cmd = msg_id(self.raw, selector(b"commandBuffer\0"));
            if cmd.is_null() {
                return false;
            }
            let encoder = msg_id(cmd, selector(b"computeCommandEncoder\0"));
            if encoder.is_null() {
                return false;
            }
            msg_void_1p(
                encoder,
                selector(b"setComputePipelineState:\0"),
                pipeline.raw,
            );
            for (i, buffer) in buffers.iter().enumerate() {
                msg_void_p2u(
                    encoder,
                    selector(b"setBuffer:offset:atIndex:\0"),
                    buffer.raw,
                    0,
                    i,
                );
            }
            let grid = MTLSize {
                width: grid[0] as usize,
                height: grid[1] as usize,
                depth: grid[2] as usize,
            };
            let block = MTLSize {
                width: block[0] as usize,
                height: block[1] as usize,
                depth: block[2] as usize,
            };
            msg_void_2size(
                encoder,
                selector(b"dispatchThreads:threadsPerThreadgroup:\0"),
                grid,
                block,
            );
            msg_void(encoder, selector(b"endEncoding\0"));
            msg_void(cmd, selector(b"commit\0"));
            msg_void(cmd, selector(b"waitUntilCompleted\0"));
            true
        }
    }
}

impl Drop for MetalCommandQueue {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// 一条 Metal 共享缓冲句柄（+1 保留，`Drop` 释放）；`contents` 是 CPU 可访问的字节视图。
pub struct MetalBuffer {
    raw: *mut c_void,
    contents: *mut u8,
    len: usize,
}

impl MetalBuffer {
    /// CPU 可访问的字节长度。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// CPU 可访问的可变字节视图；共享存储读写两侧是同一块内存。
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        // Safety: `contents` 来自 `-contents`，共享缓冲在 `release` 前稳定有效且长度即分配长度。
        unsafe { std::slice::from_raw_parts_mut(self.contents, self.len) }
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// 一条 Metal 计算管线句柄（+1 保留，`Drop` 释放）。
pub struct MetalPipeline {
    raw: *mut c_void,
}

impl Drop for MetalPipeline {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// Metal 设备执行器：实现 [`Executor`]，把一张图按序派发到本机 Metal GPU。
///
/// 与 [`CpuExecutor`](super::exec::CpuExecutor) 的差异只在「排水」：CPU 把节点交给
/// [`Program`](super::exec::Program) 就地计算，Metal 把节点交给已编译的 [`MetalPipeline`] 派发。
/// 提交仍只入队、[`Executor::sync`] 才真派发（同一流内按图序，别的流不动）。缓冲是共享存储
/// （[`MetalBuffer`]）：输入在提交前由调用方写入、输出在 `sync` 后由调用方回读，全程不经第二份拷贝。
///
/// 两张运行期表：内核表按算子名登记（[`MetalExecutor::register`]，进程内编译 MSL），操作数表按编号
/// 绑定设备缓冲（[`MetalExecutor::bind`]）。本层不引用 `op` 算子闭集、也不调用 `cutile` 生成 kernel
/// （L6 禁止列），MSL 源码由装配层注入——本类型只负责「把图提交到设备」。
///
/// 句柄指针私有、不跨线程（四个句柄类型都不派生 `Send` / `Sync`），执行器整体也是单线程后端。
pub struct MetalExecutor {
    device: Device,
    metal: MetalDevice,
    queue: MetalCommandQueue,
    kernels: BTreeMap<String, MetalPipeline>,
    buffers: BTreeMap<u32, MetalBuffer>,
    pending: Vec<Queued>,
}

impl MetalExecutor {
    /// 绑定本机默认 Metal 设备与其命令队列，构造空执行器；本机无 Metal 返回 `None`。
    ///
    /// `vendor` / `arch` 是登记标签（同 [`Device`]），不派生热路径分发。
    pub fn new(vendor: &str, arch: &str) -> Option<Self> {
        let metal = MetalDevice::system_default()?;
        let queue = metal.new_command_queue()?;
        Some(Self {
            device: Device::new(vendor, arch),
            metal,
            queue,
            kernels: BTreeMap::new(),
            buffers: BTreeMap::new(),
            pending: Vec::new(),
        })
    }

    /// 登记一段 MSL 内核源：进程内编译成计算管线并按 `name` 索引；同名已登记返回 `Invalid`。
    ///
    /// 编译入口函数名固定 `kernel_main`（[`MetalDevice::compile`] 的口径）；源由装配层注入，
    /// 本方法不生成任何算子内核。
    pub fn register(&mut self, name: &str, source: &str) -> Result<()> {
        if self.kernels.contains_key(name) {
            return Err(Error::new(Code::Invalid, format!("内核 `{name}` 已登记"))
                .with("engine::MetalExecutor::register"));
        }
        let pipeline = self.metal.compile(source).ok_or_else(|| {
            Error::new(Code::Invalid, format!("内核 `{name}` 编译失败"))
                .with("engine::MetalExecutor::register")
        })?;
        self.kernels.insert(name.to_owned(), pipeline);
        Ok(())
    }

    /// 绑定一个操作数编号到设备缓冲；返回被替换掉的旧缓冲。
    pub fn bind(&mut self, id: u32, buffer: MetalBuffer) -> Option<MetalBuffer> {
        self.buffers.insert(id, buffer)
    }

    /// 分配一段 `len` 字节的共享缓冲并绑定到编号；返回被替换掉的旧缓冲，分配失败返回 `Internal`。
    ///
    /// 成功后调用方可经 [`MetalExecutor::buffer_mut`] 写入输入、`sync` 后回读输出。
    pub fn bind_bytes(&mut self, id: u32, len: usize) -> Result<Option<MetalBuffer>> {
        let buffer = self.metal.new_buffer(len).ok_or_else(|| {
            Error::new(Code::Internal, format!("分配 {len} 字节设备缓冲失败"))
                .with("engine::MetalExecutor::bind_bytes")
        })?;
        Ok(self.buffers.insert(id, buffer))
    }

    /// 某个编号的缓冲（只读）。
    pub fn buffer(&self, id: u32) -> Option<&MetalBuffer> {
        self.buffers.get(&id)
    }

    /// 某个编号的缓冲（可写）：调用方在提交前写入输入、在 `sync` 后回读输出。
    pub fn buffer_mut(&mut self, id: u32) -> Option<&mut MetalBuffer> {
        self.buffers.get_mut(&id)
    }

    /// 尚未同步的提交条数。
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// 队列是否为空。
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// 派发一个节点：校验操作数、查内核、按「输入后输出」顺序绑定缓冲、派发、`waitUntilCompleted`。
    fn run(&mut self, node: &Node) -> Result<()> {
        self.check_operands(node)?;
        let pipeline = self.resolve_pipeline(node)?;
        let buffers = self.bind_buffers(node)?;
        if !self
            .queue
            .execute(pipeline, &buffers, node.launch().grid, node.launch().block)
        {
            return Err(
                Error::new(Code::Internal, format!("节点 `{}` 派发失败", node.name()))
                    .with("engine::MetalExecutor::run"),
            );
        }
        Ok(())
    }

    /// 按名字取已登记的内核管线。
    fn resolve_pipeline(&self, node: &Node) -> Result<&MetalPipeline> {
        let Some(pipeline) = self.kernels.get(node.name()) else {
            return Err(
                Error::new(Code::Invalid, format!("内核 `{}` 未登记", node.name()))
                    .with("engine::MetalExecutor::run"),
            );
        };
        Ok(pipeline)
    }

    /// 按「输入后输出」顺序绑定缓冲；`check_operands` 已保证编号都在表里，这里兜底防御。
    fn bind_buffers(&self, node: &Node) -> Result<Vec<&MetalBuffer>> {
        let mut buffers = Vec::with_capacity(node.inputs().len() + node.outputs().len());
        for operand in node.inputs().iter().chain(node.outputs()) {
            let Some(buffer) = self.buffers.get(&operand.id()) else {
                return Err(
                    Error::new(Code::Internal, format!("操作数 {} 未绑定", operand.id()))
                        .with("engine::MetalExecutor::run"),
                );
            };
            buffers.push(buffer);
        }
        Ok(buffers)
    }

    /// 节点自己的操作数约束：每个编号都要已绑定，且输入与输出不能是同一个编号。
    fn check_operands(&self, node: &Node) -> Result<()> {
        check_operands(
            node,
            |id| !self.buffers.contains_key(&id),
            "engine::MetalExecutor::run",
        )
    }
}

impl Executor for MetalExecutor {
    fn device(&self) -> &Device {
        &self.device
    }

    fn submit(&mut self, graph: &Graph, stream: Stream) -> Result<()> {
        enqueue(&mut self.pending, graph, stream);
        Ok(())
    }

    /// 排这一条流上已提交的节点：按入队顺序派发，成功一条才出队一条。
    ///
    /// 别的流不动（同 [`CpuExecutor::sync`](super::exec::CpuExecutor::sync) 口径）。失败时出错的那条
    /// 留在队列里，调用方能看到「它还没跑完」，而不是被悄悄丢掉。
    fn sync(&mut self, stream: Stream) -> Result<()> {
        let mut index = 0;
        while index < self.pending.len() {
            if self.pending[index].stream() != stream {
                index += 1;
                continue;
            }
            let node = self.pending[index].node().clone();
            self.run(&node)?;
            self.pending.remove(index);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 设备 → 命令队列 的进程内对象模型打通：能建、能释放，反复建放不崩溃、不重复释放。
    #[test]
    fn device_and_queue_round_trip_without_crashing() {
        for _ in 0..2 {
            let Some(device) = MetalDevice::system_default() else {
                // 无 Metal 的本机（如 CI Linux）跳过而非失败：本测试只证明"有设备时路径成立"。
                return;
            };
            let queue = device.new_command_queue();
            assert!(queue.is_some(), "有设备必有命令队列");
        }
    }

    /// 共享缓冲分配 + CPU 读写回读：写进去的字节原样读回，证明设备内存与 CPU 视图是同一块内存。
    #[test]
    fn shared_buffer_round_trip_reads_written_bytes() {
        let Some(device) = MetalDevice::system_default() else {
            return;
        };
        let Some(mut buf) = device.new_buffer(16) else {
            panic!("分配共享缓冲失败");
        };
        let expect: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        buf.as_mut_bytes().copy_from_slice(&expect);
        assert_eq!(buf.as_mut_bytes(), &expect);
    }

    /// 一段合法 MSL 源码在进程内编译出计算管线：不依赖 xcrun 子进程，直接 `newLibraryWithSource`。
    #[test]
    fn compiles_a_trivial_kernel_in_process() {
        let Some(device) = MetalDevice::system_default() else {
            return;
        };
        let source = r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = 42.0f;
}
"#;
        let pipeline = device.compile(source);
        assert!(pipeline.is_some(), "合法 MSL 应能编译出管线");
    }

    /// 派发一个写常量的内核并回读：4 个线程各写 `out[id] = 42.0`，`waitUntilCompleted` 后 CPU 读到 4 个 42.0。
    #[test]
    fn dispatches_a_kernel_and_reads_back() {
        let Some(device) = MetalDevice::system_default() else {
            return;
        };
        let source = r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = 42.0f;
}
"#;
        let Some(pipeline) = device.compile(source) else {
            panic!("合法 MSL 应能编译出管线");
        };
        let Some(queue) = device.new_command_queue() else {
            panic!("有设备必有命令队列");
        };
        // 4 个 f32 = 16 字节。
        let mut out = device.new_buffer(16).expect("分配输出缓冲失败");
        assert!(
            queue.execute(&pipeline, &[&out], [4, 1, 1], [1, 1, 1]),
            "派发应成功"
        );

        let bytes = out.as_mut_bytes();
        assert_eq!(bytes.len(), 16);
        let got: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(got, vec![42.0f32; 4], "GPU 应把 42.0 写进每个元素");
    }

    /// 执行器把一张图的节点派发到 Metal 并回读：注册 `kernel_main` 双倍内核，绑定输入/输出，
    /// `submit` → `sync` 后输出 = 输入的 2 倍，证明「提交 → 派发 → 同步」整条 [`Executor`] 契约在真机成立。
    #[test]
    fn executor_runs_a_registered_kernel_through_the_graph() {
        use super::super::exec::Launch;

        let Some(mut exec) = MetalExecutor::new("metal", "apple") else {
            return;
        };
        exec.register(
            "double",
            r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2.0f;
}
"#,
        )
        .expect("登记内核失败");
        exec.bind_bytes(0, 16).expect("绑定输入缓冲失败");
        exec.bind_bytes(1, 16).expect("绑定输出缓冲失败");

        let input = [1.0f32, 2.0, 3.0, 4.0];
        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_ne_bytes()).collect();
        exec.buffer_mut(0)
            .expect("输入缓冲应已绑定")
            .as_mut_bytes()
            .copy_from_slice(&input_bytes);

        let mut graph = Graph::new();
        let mut node = Node::new("double", Launch::new([4, 1, 1], [1, 1, 1]));
        node.bind_input(0).bind_output(1);
        graph.push(node);

        exec.submit(&graph, Stream::new(0)).expect("提交失败");
        exec.sync(Stream::new(0)).expect("同步失败");

        let (chunks, _) = exec
            .buffer_mut(1)
            .expect("输出缓冲应已绑定")
            .as_mut_bytes()
            .as_chunks::<4>();
        let output: Vec<f32> = chunks.iter().map(|c| f32::from_ne_bytes(*c)).collect();
        assert_eq!(output, vec![2.0f32, 4.0, 6.0, 8.0], "GPU 应把输入翻倍写回");
    }

    /// simdgroup 矩阵指令探测：编译并派发 8×8×8 矩阵乘（`simdgroup_float8x8` /
    /// `simdgroup_load` / `simdgroup_multiply_accumulate` / `simdgroup_store`），与 CPU 参考对拍。
    /// 探测失败只报原因，后续路线退回寄存器分块。
    #[test]
    #[ignore = "Metal simdgroup 探测：cargo test -p llm-engine --lib -- metal_simdgroup --release --ignored --nocapture"]
    fn metal_simdgroup_probe() {
        let Some(device) = MetalDevice::system_default() else {
            eprintln!("Metal 设备不在本机，simdgroup 探测未运行");
            return;
        };
        let Some(queue) = device.new_command_queue() else {
            panic!("有设备必有命令队列");
        };
        let source = r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device const float* A [[buffer(0)]],
                        device const float* B [[buffer(1)]],
                        device float* C [[buffer(2)]],
                        uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float zeros[64];
    for (uint i = lid.x; i < 64; i += 32) zeros[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 a, b, acc;
    simdgroup_load(a, A, 8);
    simdgroup_load(b, B, 8);
    simdgroup_load(acc, zeros, 8);
    simdgroup_multiply_accumulate(acc, a, b, acc);
    simdgroup_store(acc, C, 8);
}
"#;
        let Some(pipeline) = device.compile(source) else {
            eprintln!("simdgroup 探测：MSL 编译失败，本机走寄存器分块路线");
            return;
        };
        let a: Vec<f32> = (0..64).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
        let b: Vec<f32> = (0..64).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
        let reference = cpu_matmul(&a, &b, 8, 8, 8);
        let mut a_buf = device.new_buffer(64 * 4).expect("A 缓冲");
        let mut b_buf = device.new_buffer(64 * 4).expect("B 缓冲");
        let mut c_buf = device.new_buffer(64 * 4).expect("C 缓冲");
        a_buf.as_mut_bytes().copy_from_slice(&as_bytes(&a));
        b_buf.as_mut_bytes().copy_from_slice(&as_bytes(&b));
        let ok = queue.execute(&pipeline, &[&a_buf, &b_buf, &c_buf], [32, 1, 1], [32, 1, 1]);
        assert!(ok, "simdgroup 探测派发失败");
        assert_matmul_matches("simdgroup-8x8", &reference, c_buf.as_mut_bytes());
        eprintln!("simdgroup 探测通过：8×8×8 矩阵乘 32 线程单线程组，输出与 CPU 参考一致");
    }

    /// Metal 真机 matmul 前向基准：同一份 `M×N×K` 稠密 f32 在两条路径上跑同一算术，
    /// 对拍 GPU 与 CPU 参考、并报 naive（每线程一输出）与 tiled（threadgroup 共享内存分块）
    /// 的真实端到端延迟与 GFLOP/s。
    ///
    /// 本用例只出测量：`cargo test -p llm-engine --lib -- metal_matmul_forward --release --ignored --nocapture`。
    /// 无 Metal 设备时静默跳过；产品级 E1/E3（7B / 100K / NVIDIA A800·3090·T600）不在本机、保持 OPEN。
    #[test]
    #[ignore = "Metal matmul 前向基准：cargo test -p llm-engine --lib -- metal_matmul_forward --release --ignored --nocapture"]
    fn metal_matmul_forward() {
        let Some(device) = MetalDevice::system_default() else {
            eprintln!("Metal 设备不在本机，matmul 前向基准未运行");
            return;
        };
        let Some(queue) = device.new_command_queue() else {
            panic!("有设备必有命令队列");
        };

        const M: usize = 512;
        const N: usize = 512;
        const K: usize = 512;

        let a: Vec<f32> = (0..M * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
        let b: Vec<f32> = (0..K * N).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
        let reference = cpu_matmul(&a, &b, M, N, K);

        let mut a_buf = device.new_buffer(a.len() * 4).expect("A 缓冲");
        let mut b_buf = device.new_buffer(b.len() * 4).expect("B 缓冲");
        a_buf.as_mut_bytes().copy_from_slice(&as_bytes(&a));
        b_buf.as_mut_bytes().copy_from_slice(&as_bytes(&b));

        let naive = device
            .compile(&matmul_msl(false, M, N, K))
            .expect("naive matmul 应能编译");
        let tiled = device
            .compile(&matmul_msl(true, M, N, K))
            .expect("tiled matmul 应能编译");

        let mut c_buf = device.new_buffer(M * N * 4).expect("C 缓冲");
        let grid = [N as u32, M as u32, 1];

        // 先各跑一次对拍，再进入计时轮。
        queue.execute(&naive, &[&a_buf, &b_buf, &c_buf], grid, [1, 1, 1]);
        assert_matmul_matches("naive", &reference, c_buf.as_mut_bytes());
        queue.execute(&tiled, &[&a_buf, &b_buf, &c_buf], grid, [16, 16, 1]);
        assert_matmul_matches("tiled", &reference, c_buf.as_mut_bytes());

        let flop = 2.0 * (M * N * K) as f64;
        let naive_ns = measure(&queue, &naive, &[&a_buf, &b_buf, &c_buf], grid, [1, 1, 1]);
        let tiled_ns = measure(&queue, &tiled, &[&a_buf, &b_buf, &c_buf], grid, [16, 16, 1]);

        eprintln!(
            "Metal matmul 512×512×512：naive={naive_ns:.1} ns（{:.2} GFLOP/s） tiled={tiled_ns:.1} ns（{:.2} GFLOP/s） 加速={:.2}×",
            flop / naive_ns,
            flop / tiled_ns,
            naive_ns / tiled_ns,
        );
    }

    /// 生产态分块 GEMM 对照：同一份 `matmul_f32`（K=4096）分别经
    /// [`cutile::execution::metal_grid_program`]（标量网格基线）与
    /// [`cutile::execution::metal_tiled_gemm_program`]（共享内存分块）生成 MSL，在真机 GPU 上跑
    /// 同一组输入、对拍 CPU 参考并报延迟与 GFLOP/s。
    ///
    /// 本用例只出测量：`cargo test -p llm-engine --lib -- metal_tiled_gemm_k4096 --release --ignored --nocapture`。
    /// 无 Metal 设备时静默跳过；产品级 E1/E3 不在本机、保持 OPEN。
    #[test]
    #[ignore = "Metal 生产态分块 GEMM：cargo test -p llm-engine --lib -- metal_tiled_gemm_k4096 --release --ignored --nocapture"]
    fn metal_tiled_gemm_k4096() {
        let Some(device) = MetalDevice::system_default() else {
            eprintln!("Metal 设备不在本机，分块 GEMM 基准未运行");
            return;
        };
        let Some(queue) = device.new_command_queue() else {
            panic!("有设备必有命令队列");
        };
        crate::backends::install();

        const M: usize = 256;
        const N: usize = 256;
        const K: usize = 4096;

        let a: Vec<f32> = (0..M * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
        let b: Vec<f32> = (0..K * N).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
        let reference = cpu_matmul(&a, &b, M, N, K);

        let mut a_buf = device.new_buffer(M * K * 4).expect("A 缓冲");
        let mut b_buf = device.new_buffer(K * N * 4).expect("B 缓冲");
        a_buf.as_mut_bytes().copy_from_slice(&as_bytes(&a));
        b_buf.as_mut_bytes().copy_from_slice(&as_bytes(&b));
        let mut y_buf = device.new_buffer(M * N * 4).expect("Y 缓冲");

        let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
        let dirs =
            cutile::dirs::Dirs::under(std::env::temp_dir().join("engine-metal-matmul-k4096"));
        let grid_msl = cutile::execution::metal_grid_program(&catalogue, &dirs, "matmul_f32", M, N)
            .expect("标量网格 MSL 应能生成");
        let grid_pipe = device.compile(&grid_msl).expect("标量网格 MSL 应能编译");

        let grid = [M as u32, N as u32, 1];
        queue.execute(&grid_pipe, &[&a_buf, &b_buf, &y_buf], grid, [1, 1, 1]);
        assert_matmul_matches("grid", &reference, y_buf.as_mut_bytes());

        let flop = 2.0 * (M * N * K) as f64;
        let grid_ns = measure(
            &queue,
            &grid_pipe,
            &[&a_buf, &b_buf, &y_buf],
            grid,
            [1, 1, 1],
        );
        eprintln!(
            "Metal 生产态 GEMM 256×256×4096：网格标量={grid_ns:.1} ns（{:.2} GFLOP/s）",
            flop / grid_ns,
        );
        for (tile, per_thread) in [(8usize, 1usize), (16, 1), (32, 1), (32, 4)] {
            let msl = cutile::execution::metal_tiled_gemm_program(
                &catalogue,
                &dirs,
                "matmul_f32",
                M,
                N,
                tile,
                per_thread,
            )
            .expect("分块 MSL 应能生成");
            let pipe = device.compile(&msl).expect("分块 MSL 应能编译");
            let threads = tile / per_thread;
            let (launch_grid, block) = if per_thread == 1 {
                (grid, [tile as u32, tile as u32, 1])
            } else {
                (
                    [(M / per_thread) as u32, (N / per_thread) as u32, 1],
                    [threads as u32, threads as u32, 1],
                )
            };
            queue.execute(&pipe, &[&a_buf, &b_buf, &y_buf], launch_grid, block);
            assert_matmul_matches("tiled", &reference, y_buf.as_mut_bytes());
            let ns = measure(&queue, &pipe, &[&a_buf, &b_buf, &y_buf], launch_grid, block);
            eprintln!(
                "  分块 tile={tile} pt={per_thread}：{ns:.1} ns（{:.2} GFLOP/s） 加速={:.2}×",
                flop / ns,
                grid_ns / ns,
            );
        }
        for (row_frags, col_frags) in [(1usize, 1usize), (2, 2), (4, 4)] {
            let msl = cutile::execution::metal_simdgroup_gemm_program(
                &catalogue,
                &dirs,
                "matmul_f32",
                M,
                N,
                row_frags,
                col_frags,
            )
            .expect("simdgroup MSL 应能生成");
            let pipe = device.compile(&msl).expect("simdgroup MSL 应能编译");
            let simd_grid = [
                ((M / (8 * row_frags)) * 32) as u32,
                (N / (8 * col_frags)) as u32,
                1,
            ];
            let block = [32, 1, 1];
            queue.execute(&pipe, &[&a_buf, &b_buf, &y_buf], simd_grid, block);
            assert_matmul_matches("simdgroup", &reference, y_buf.as_mut_bytes());
            let ns = measure(
                &queue,
                &pipe,
                &[&a_buf, &b_buf, &y_buf],
                simd_grid,
                block,
            );
            eprintln!(
                "  simdgroup {row_frags}×{col_frags} 片段：{ns:.1} ns（{:.2} GFLOP/s） 加速={:.2}×",
                flop / ns,
                grid_ns / ns,
            );
        }
    }

    /// 生产态分块注意力对照：同一份 `attention_f32`（S=512、D=64）分别经
    /// [`cutile::execution::metal_grid_program`]（每查询行一线程）与
    /// [`cutile::execution::metal_tiled_attention_program`]（每行一个 threadgroup）生成 MSL，
    /// 在真机 GPU 上跑同一组 Q/K/V、对拍 CPU 参考并报延迟与吞吐。
    ///
    /// 本用例只出测量：`cargo test -p llm-engine --lib -- metal_tiled_attention_s512 --release --ignored --nocapture`。
    #[test]
    #[ignore = "Metal 生产态分块注意力：cargo test -p llm-engine --lib -- metal_tiled_attention_s512 --release --ignored --nocapture"]
    fn metal_tiled_attention_s512() {
        let Some(device) = MetalDevice::system_default() else {
            eprintln!("Metal 设备不在本机，分块注意力基准未运行");
            return;
        };
        let Some(queue) = device.new_command_queue() else {
            panic!("有设备必有命令队列");
        };
        crate::backends::install();

        const S: usize = 512;
        const D: usize = 64;
        const THREADS: usize = 64;

        let q: Vec<f32> = (0..S * D).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
        let k: Vec<f32> = (0..S * D).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
        let v: Vec<f32> = (0..S * D).map(|i| ((i % 3) as f32) * 0.3 - 0.3).collect();
        let reference = cpu_attention(&q, &k, &v, S, D);

        let mut q_buf = device.new_buffer(S * D * 4).expect("Q 缓冲");
        let mut k_buf = device.new_buffer(S * D * 4).expect("K 缓冲");
        let mut v_buf = device.new_buffer(S * D * 4).expect("V 缓冲");
        q_buf.as_mut_bytes().copy_from_slice(&as_bytes(&q));
        k_buf.as_mut_bytes().copy_from_slice(&as_bytes(&k));
        v_buf.as_mut_bytes().copy_from_slice(&as_bytes(&v));
        let mut y_buf = device.new_buffer(S * D * 4).expect("Y 缓冲");

        let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
        let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join("engine-metal-attn-s512"));
        let grid_msl =
            cutile::execution::metal_grid_program(&catalogue, &dirs, "attention_f32", S, D)
                .expect("网格注意力 MSL 应能生成");
        let tiled_msl = cutile::execution::metal_tiled_attention_program(
            &catalogue,
            &dirs,
            "attention_f32",
            S,
            D,
            THREADS,
        )
        .expect("分块注意力 MSL 应能生成");
        let grid_pipe = device.compile(&grid_msl).expect("网格注意力 MSL 应能编译");
        let tiled_pipe = device.compile(&tiled_msl).expect("分块注意力 MSL 应能编译");

        queue.execute(
            &grid_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            [S as u32, 1, 1],
            [1, 1, 1],
        );
        assert_matmul_matches("attn-grid", &reference, y_buf.as_mut_bytes());
        let tiled_grid = [(S * THREADS) as u32, 1, 1];
        queue.execute(
            &tiled_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            tiled_grid,
            [THREADS as u32, 1, 1],
        );
        assert_matmul_matches("attn-tiled", &reference, y_buf.as_mut_bytes());

        let grid_ns = measure(
            &queue,
            &grid_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            [S as u32, 1, 1],
            [1, 1, 1],
        );
        let tiled_ns = measure(
            &queue,
            &tiled_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            tiled_grid,
            [THREADS as u32, 1, 1],
        );
        eprintln!(
            "Metal 生产态 attention 512×64：网格行线程={grid_ns:.1} ns 分块线程组={tiled_ns:.1} ns（每行 {THREADS} 线程） 加速={:.2}×",
            grid_ns / tiled_ns,
        );
        for block in [64usize, 128, 256, 512] {
            let flash_msl = cutile::execution::metal_flash_attention_program(
                &catalogue,
                &dirs,
                "attention_f32",
                S,
                D,
                THREADS,
                block,
            )
            .expect("flash attention MSL 应能生成");
            let flash_pipe = device
                .compile(&flash_msl)
                .expect("flash attention MSL 应能编译");
            queue.execute(
                &flash_pipe,
                &[&q_buf, &k_buf, &v_buf, &y_buf],
                tiled_grid,
                [THREADS as u32, 1, 1],
            );
            assert_matmul_matches("attn-flash", &reference, y_buf.as_mut_bytes());
            let flash_ns = measure(
                &queue,
                &flash_pipe,
                &[&q_buf, &k_buf, &v_buf, &y_buf],
                tiled_grid,
                [THREADS as u32, 1, 1],
            );
            eprintln!(
                "  flash block={block}：{flash_ns:.1} ns（每行 {THREADS} 线程） 对网格={:.2}× 对分块={:.2}×",
                grid_ns / flash_ns,
                tiled_ns / flash_ns,
            );
        }
    }

    /// 长序列 flash attention：S=2048、D=64，验证 K/V 分块流式扫描不保留完整分数向量，
    /// 与「整行分数留在 threadgroup 内存」的分块路径对拍并报延迟。
    ///
    /// 本用例只出测量：`cargo test -p llm-engine --lib -- metal_flash_attention_long --release --ignored --nocapture`。
    #[test]
    #[ignore = "Metal 长序列 flash attention：cargo test -p llm-engine --lib -- metal_flash_attention_long --release --ignored --nocapture"]
    fn metal_flash_attention_long() {
        let Some(device) = MetalDevice::system_default() else {
            eprintln!("Metal 设备不在本机，长序列 flash attention 基准未运行");
            return;
        };
        let Some(queue) = device.new_command_queue() else {
            panic!("有设备必有命令队列");
        };
        crate::backends::install();

        const S: usize = 2048;
        const D: usize = 64;
        const THREADS: usize = 64;

        let q: Vec<f32> = (0..S * D).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
        let k: Vec<f32> = (0..S * D).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
        let v: Vec<f32> = (0..S * D).map(|i| ((i % 3) as f32) * 0.3 - 0.3).collect();
        let reference = cpu_attention(&q, &k, &v, S, D);

        let mut q_buf = device.new_buffer(S * D * 4).expect("Q 缓冲");
        let mut k_buf = device.new_buffer(S * D * 4).expect("K 缓冲");
        let mut v_buf = device.new_buffer(S * D * 4).expect("V 缓冲");
        q_buf.as_mut_bytes().copy_from_slice(&as_bytes(&q));
        k_buf.as_mut_bytes().copy_from_slice(&as_bytes(&k));
        v_buf.as_mut_bytes().copy_from_slice(&as_bytes(&v));
        let mut y_buf = device.new_buffer(S * D * 4).expect("Y 缓冲");

        let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
        let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join("engine-metal-flash-long"));
        let grid_msl =
            cutile::execution::metal_grid_program(&catalogue, &dirs, "attention_f32", S, D)
                .expect("网格注意力 MSL 应能生成");
        let grid_pipe = device.compile(&grid_msl).expect("网格注意力 MSL 应能编译");
        let launch = [(S * THREADS) as u32, 1, 1];
        let block = [THREADS as u32, 1, 1];
        queue.execute(
            &grid_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            [S as u32, 1, 1],
            [1, 1, 1],
        );
        assert_matmul_matches("attn-long-grid", &reference, y_buf.as_mut_bytes());
        let grid_ns = measure(
            &queue,
            &grid_pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            [S as u32, 1, 1],
            [1, 1, 1],
        );
        eprintln!("Metal 长序列 attention {S}×{D}：网格行线程={grid_ns:.1} ns");

        for key_block in [64usize, 128, 256] {
            let msl = cutile::execution::metal_flash_attention_program(
                &catalogue,
                &dirs,
                "attention_f32",
                S,
                D,
                THREADS,
                key_block,
            )
            .expect("flash attention MSL 应能生成");
            let pipe = device.compile(&msl).expect("flash attention MSL 应能编译");
            queue.execute(
                &pipe,
                &[&q_buf, &k_buf, &v_buf, &y_buf],
                launch,
                block,
            );
            assert_matmul_matches("attn-long-flash", &reference, y_buf.as_mut_bytes());
            let ns = measure(&queue, &pipe, &[&q_buf, &k_buf, &v_buf, &y_buf], launch, block);
            eprintln!(
                "  flash block={key_block}：{ns:.1} ns 对网格={:.2}×",
                grid_ns / ns,
            );
        }
    }

    /// 注意力 CPU 参考：分数 f64 累加、行内 max/exp 归一化、输出 f64 累加。
    fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
        let scale = (d as f32).sqrt();
        let mut out = vec![0.0f32; s * d];
        let mut scores = vec![0.0f32; s];
        for i in 0..s {
            let mut row_max = f32::MIN;
            for j in 0..s {
                let mut dot = 0.0f64;
                for p in 0..d {
                    dot += f64::from(q[i * d + p]) * f64::from(k[j * d + p]);
                }
                scores[j] = (dot as f32) / scale;
                row_max = row_max.max(scores[j]);
            }
            let mut sum = 0.0f64;
            for score in &mut scores {
                *score = (*score - row_max).exp();
                sum += f64::from(*score);
            }
            for h in 0..d {
                let mut acc = 0.0f64;
                for j in 0..s {
                    acc += f64::from(scores[j]) * f64::from(v[j * d + h]);
                }
                out[i * d + h] = (acc / sum) as f32;
            }
        }
        out
    }

    /// 朴素 MSL：每线程算一个输出元素；`tiled` 为真时用 threadgroup 共享内存把 A/B 按 16×16 分块，
    /// 每个线程组 16×16 线程协同搬运一次整块，规约轴复用共享块，避免反复从设备全局内存重读。
    fn matmul_msl(tiled: bool, m: usize, n: usize, k: usize) -> String {
        if !tiled {
            return format!(
                "#include <metal_stdlib>\n\
                 using namespace metal;\n\
                 kernel void kernel_main(\n\
                 \x20    device const float* A [[buffer(0)]],\n\
                 \x20    device const float* B [[buffer(1)]],\n\
                 \x20    device float* C [[buffer(2)]],\n\
                 \x20    uint2 gid [[thread_position_in_grid]]) {{\n\
                 \x20    const uint M = {m}, N = {n}, K = {k};\n\
                 \x20    float acc = 0.0f;\n\
                 \x20    for (uint p = 0; p < K; ++p) acc += A[gid.y * K + p] * B[p * N + gid.x];\n\
                 \x20    C[gid.y * N + gid.x] = acc;\n\
                 }}"
            );
        }
        format!(
            "#include <metal_stdlib>\n\
             using namespace metal;\n\
             kernel void kernel_main(\n\
             \x20    device const float* A [[buffer(0)]],\n\
             \x20    device const float* B [[buffer(1)]],\n\
             \x20    device float* C [[buffer(2)]],\n\
             \x20    uint2 gid [[thread_position_in_grid]],\n\
             \x20    uint2 lid [[thread_position_in_threadgroup]]) {{\n\
             \x20    const uint M = {m}, N = {n}, K = {k};\n\
             \x20    threadgroup float As[16][16];\n\
             \x20    threadgroup float Bs[16][16];\n\
             \x20    float acc = 0.0f;\n\
             \x20    for (uint kk = 0; kk < K; kk += 16) {{\n\
             \x20        As[lid.y][lid.x] = A[gid.y * K + kk + lid.x];\n\
             \x20        Bs[lid.y][lid.x] = B[(kk + lid.y) * N + gid.x];\n\
             \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \x20        for (uint p = 0; p < 16; ++p) acc += As[lid.y][p] * Bs[p][lid.x];\n\
             \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \x20    }}\n\
             \x20    C[gid.y * N + gid.x] = acc;\n\
             }}"
        )
    }

    /// 同一算术的 CPU 参考（f64 累加，保证对拍基准的精度只受量化级控制，不引入累加顺序差异）。
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
                }
                out[i * n + j] = acc as f32;
            }
        }
        out
    }

    fn as_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|f| f.to_ne_bytes()).collect()
    }

    fn assert_matmul_matches(label: &str, reference: &[f32], output: &[u8]) {
        let (chunks, _) = output.as_chunks::<4>();
        let mut max_rel = 0.0f32;
        for (got, want) in chunks
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .zip(reference.iter())
        {
            let denom = want.abs().max(1.0f32);
            max_rel = max_rel.max((got - want).abs() / denom);
        }
        assert!(
            max_rel < 1e-3,
            "{label} GPU 输出与 CPU 参考偏差过大：max_rel_err={max_rel}"
        );
    }

    /// 同一内核连跑 50 次，返回中位数延迟（ns）；预热一次不计入。
    fn measure(
        queue: &MetalCommandQueue,
        pipeline: &MetalPipeline,
        buffers: &[&MetalBuffer],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> f64 {
        queue.execute(pipeline, buffers, grid, block);
        let mut samples = [0u128; 50];
        for slot in &mut samples {
            let start = std::time::Instant::now();
            queue.execute(pipeline, buffers, grid, block);
            *slot = start.elapsed().as_nanos();
        }
        samples.sort_unstable();
        samples[samples.len() / 2] as f64
    }
}

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

use super::exec::{Device, Executor, Graph, Node, Queued, Stream, enqueue};

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

/// `objc_msgSend` 按 `(id, SEL) -> id` 强转：0 参、返回对象/指针的 ObjC 方法。
unsafe fn msg_id(receiver: *mut c_void, selector: *mut c_void) -> *mut c_void {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { send(receiver, selector) }
}

/// `objc_msgSend` 按 `(id, SEL) -> void` 强转：`-release` 这类无返回值方法。
unsafe fn msg_void(receiver: *mut c_void, selector: *mut c_void) {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void) =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { send(receiver, selector) };
}

/// `objc_msgSend` 按 `(id, SEL, usize, usize) -> id` 强转：两个 `NSUInteger` 参数的方法。
unsafe fn msg_id_2u(
    receiver: *mut c_void,
    selector: *mut c_void,
    a: usize,
    b: usize,
) -> *mut c_void {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, usize, usize) -> *mut c_void =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { send(receiver, selector, a, b) }
}

/// `objc_msgSend` 按 `(id, SEL, p) -> id` 强转：一个指针参数的方法。
unsafe fn msg_id_1p(receiver: *mut c_void, selector: *mut c_void, p: *mut c_void) -> *mut c_void {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { send(receiver, selector, p) }
}

/// `objc_msgSend` 按 `(id, SEL, p1, p2) -> id` 强转：两个指针参数的方法。
unsafe fn msg_id_2p(
    receiver: *mut c_void,
    selector: *mut c_void,
    p1: *mut c_void,
    p2: *mut c_void,
) -> *mut c_void {
    let send: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> *mut c_void = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { send(receiver, selector, p1, p2) }
}

/// `objc_msgSend` 按 `(id, SEL, p1, p2, p3) -> id` 强转：三个指针参数的方法。
unsafe fn msg_id_3p(
    receiver: *mut c_void,
    selector: *mut c_void,
    p1: *mut c_void,
    p2: *mut c_void,
    p3: *mut c_void,
) -> *mut c_void {
    let send: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> *mut c_void = unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { send(receiver, selector, p1, p2, p3) }
}

/// Metal 三维尺寸：`dispatchThreads:threadsPerThreadgroup:` 的两个按值参数（`MTLSize`）。
#[repr(C)]
#[derive(Clone, Copy)]
struct MTLSize {
    width: usize,
    height: usize,
    depth: usize,
}

/// `objc_msgSend` 按 `(id, SEL, p) -> void` 强转：一个指针参数的无返回值方法。
unsafe fn msg_void_1p(receiver: *mut c_void, selector: *mut c_void, p: *mut c_void) {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { send(receiver, selector, p) };
}

/// `objc_msgSend` 按 `(id, SEL, p, usize, usize) -> void` 强转：`setBuffer:offset:atIndex:` 的形状。
unsafe fn msg_void_p2u(
    receiver: *mut c_void,
    selector: *mut c_void,
    p: *mut c_void,
    a: usize,
    b: usize,
) {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, usize, usize) =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { send(receiver, selector, p, a, b) };
}

/// `objc_msgSend` 按 `(id, SEL, MTLSize, MTLSize) -> void` 强转：`dispatchThreads:threadsPerThreadgroup:` 的形状。
///
/// `MTLSize` 24 字节（>16），arm64 / x86_64 下 `extern "C"` 按间接指针传参，与 clang 生成的
/// `objc_msgSend` 调用 ABI 一致，无需 `objc_msgSend_stret`。
unsafe fn msg_void_2size(receiver: *mut c_void, selector: *mut c_void, a: MTLSize, b: MTLSize) {
    let send: unsafe extern "C" fn(*mut c_void, *mut c_void, MTLSize, MTLSize) =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { send(receiver, selector, a, b) };
}

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
        for operand in node.inputs().iter().chain(node.outputs()) {
            if !self.buffers.contains_key(&operand.id()) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("节点 `{}` 的操作数 {} 未绑定", node.name(), operand.id()),
                )
                .with("engine::MetalExecutor::run"));
            }
        }
        let clash = node
            .outputs()
            .iter()
            .any(|output| node.inputs().iter().any(|input| input.id() == output.id()));
        if clash {
            return Err(Error::new(
                Code::Invalid,
                format!("节点 `{}` 有一个编号既是输入又是输出", node.name()),
            )
            .with("engine::MetalExecutor::run"));
        }
        Ok(())
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
}

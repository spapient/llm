//! Metal 设备边界：手写 FFI，设备、命令队列、共享缓冲、计算管线句柄与 [`MetalExecutor`]（M3 第二刀）。

#![allow(unsafe_code)]

use diagnostic::{Code, Error, Result};
use engine::exec::{Device, Executor, Graph, Node, Queued, Stream, check_operands, enqueue};
use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};
use std::ptr::null_mut;

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
    (usize $name:ident, []) => {
        unsafe fn $name(receiver: *mut c_void, selector: *mut c_void) -> usize {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
                unsafe { std::mem::transmute(objc_msgSend as *const ()) };
            unsafe { send(receiver, selector) }
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
// `(id, SEL, usize) -> void`：一个 NSUInteger 参数（如 setLanguageVersion:）。
msg!(void msg_void_1u, [a: usize]);
// `(id, SEL) -> usize`：返回 NSUInteger 的 0 参方法（如 -status）。
msg!(usize msg_usize, []);
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
    pub(super) raw: *mut c_void,
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
    pub fn new_buf(&self, bytes: usize) -> Option<MetalBuf> {
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
        Some(MetalBuf {
            raw,
            contents: contents.cast(),
            len: bytes,
        })
    }

    /// 把一段 MSL 源码编译成计算管线（+1 保留，`Drop` 释放）；入口函数名由源码自己说。
    pub fn compile(&self, source: &str) -> Option<MetalPipeline> {
        let entry = entry_of(source)?;
        let src = CString::new(source).ok()?;
        let fname = CString::new(entry).ok()?;
        // `stringWithUTF8String:` 返回自动释放对象，编译路径包一层池，池弹栈时统一释放。
        let pool = unsafe { objc_autoreleasePoolPush() };
        let result = self.compile_with_pool(src.as_ptr(), fname.as_ptr(), metal4(source));
        unsafe { objc_autoreleasePoolPop(pool) };
        result
    }

    /// 依据是否抬到 Metal 4.0 选择编译选项：非 4.0 传空指针（沿用设备默认版本）。
    fn compile_options(metal4: bool) -> *mut c_void {
        if metal4 {
            unsafe { metal4_options() }
        } else {
            null_mut()
        }
    }

    /// 池内的编译主体：`newLibraryWithSource → newFunctionWithName → newComputePipelineState`。
    fn compile_with_pool(
        &self,
        src: *const c_char,
        fname: *const c_char,
        metal4: bool,
    ) -> Option<MetalPipeline> {
        unsafe {
            let source = nsstring(src);
            let options = Self::compile_options(metal4);
            let mut error: *mut c_void = null_mut();
            let library = msg_id_3p(
                self.raw,
                selector(b"newLibraryWithSource:options:error:\0"),
                source,
                options,
                (&raw mut error).cast(),
            );
            if !options.is_null() {
                release(options);
            }
            if library.is_null() {
                report(&error);
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
                (&raw mut error).cast(),
            );
            // 管线建立后 library / function 即自包含，立即释放，只持有 pipeline。
            release(library);
            release(function);
            if pipeline.is_null() {
                report(&error);
            }
            (!pipeline.is_null()).then_some(MetalPipeline { raw: pipeline })
        }
    }
}

/// 源码要求 Metal 4.0 语言版本：张量核心内建（`metal_tensor` / `mpp::tensor_ops`）的门槛。
pub(crate) fn metal4(source: &str) -> bool {
    source.contains("<metal_tensor>") || source.contains("mpp::tensor_ops")
}

/// `MTLCommandBufferStatusCompleted`：命令缓冲正常跑完的唯一取值。
pub(crate) const COMMAND_BUF_COMPLETED: usize = 4;

/// Metal 4.0 对应的 `MTLLanguageVersion` 数值（`MTLLanguageVersion4_0`）。
pub(crate) const METAL4_VERSION: usize = 0x0004_0000;

/// 构造抬到 4.0 的 `MTLCompileOptions`（+1 保留，调用方负责 `release`）。
pub(crate) unsafe fn metal4_options() -> *mut c_void {
    let class = unsafe { objc_getClass(c"MTLCompileOptions".as_ptr()) };
    if class.is_null() {
        return null_mut();
    }
    let options = unsafe { msg_id(class, selector(b"new\0")) };
    if options.is_null() {
        return null_mut();
    }
    unsafe { msg_void_1u(options, selector(b"setLanguageVersion:\0"), METAL4_VERSION) };
    options
}

/// 把 `NSError` 的 `localizedDescription` 打到 stderr；拿不到就只说一句失败。
pub(crate) unsafe fn report(error: &*mut c_void) {
    let text = unsafe {
        let message = msg_id(*error, selector(b"localizedDescription\0"));
        if message.is_null() {
            None
        } else {
            let bytes = msg_id(message, selector(b"UTF8String\0"));
            (!bytes.is_null()).then(|| std::ffi::CStr::from_ptr(bytes.cast()).to_string_lossy())
        }
    };
    match text {
        Some(text) => eprintln!("metal 编译失败: {text}"),
        None => eprintln!("metal 编译失败: 无 NSError 描述"),
    }
}

/// 源码里的入口函数名：`kernel void <name>(` 的 `<name>`。
pub(crate) fn entry_of(source: &str) -> Option<&str> {
    const MARK: &str = "kernel void ";
    let rest = source.split_once(MARK)?.1.trim_start();
    let name = rest[..rest.find('(')?].trim();
    (!name.is_empty()).then_some(name)
}

impl Drop for MetalDevice {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// Metal 设备执行器：实现 [`Executor`]，把一张图按序派发到本机 Metal GPU。
pub struct MetalExecutor {
    pub(super) device: Device,
    pub(super) metal: MetalDevice,
    pub(super) queue: MetalCommandQueue,
    pub(super) kernels: BTreeMap<String, MetalPipeline>,
    pub(super) bufs: BTreeMap<u32, MetalBuf>,
    pub(super) pending: Vec<Queued>,
}

impl MetalExecutor {
    /// 绑定本机默认 Metal 设备与其命令队列，构造空执行器；本机无 Metal 返回 `None`。
    pub fn new(vendor: &str, arch: &str) -> Option<Self> {
        let metal = MetalDevice::system_default()?;
        let queue = metal.new_command_queue()?;
        Some(Self {
            device: Device::new(vendor, arch),
            metal,
            queue,
            kernels: BTreeMap::new(),
            bufs: BTreeMap::new(),
            pending: Vec::new(),
        })
    }

    /// 登记一段 MSL 内核源：进程内编译成计算管线并按 `name` 索引；同名已登记返回 `Invalid`。
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
    pub fn bind(&mut self, id: u32, buf: MetalBuf) -> Option<MetalBuf> {
        self.bufs.insert(id, buf)
    }

    /// 分配一段 `len` 字节的共享缓冲并绑定到编号；返回被替换掉的旧缓冲，分配失败返回 `Internal`。
    pub fn bind_bytes(&mut self, id: u32, len: usize) -> Result<Option<MetalBuf>> {
        let buf = self.metal.new_buf(len).ok_or_else(|| {
            Error::new(Code::Internal, format!("分配 {len} 字节设备缓冲失败"))
                .with("engine::MetalExecutor::bind_bytes")
        })?;
        Ok(self.bufs.insert(id, buf))
    }

    /// 某个编号的缓冲（只读）。
    pub fn buf(&self, id: u32) -> Option<&MetalBuf> {
        self.bufs.get(&id)
    }

    /// 某个编号的缓冲（可写）：调用方在提交前写入输入、在 `sync` 后回读输出。
    pub fn buf_mut(&mut self, id: u32) -> Option<&mut MetalBuf> {
        self.bufs.get_mut(&id)
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
        let bufs = self.bind_bufs(node)?;
        if !self
            .queue
            .execute(pipeline, &bufs, node.launch().grid, node.launch().block)
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
    fn bind_bufs(&self, node: &Node) -> Result<Vec<&MetalBuf>> {
        let mut bufs = Vec::with_capacity(node.inputs().len() + node.outputs().len());
        for operand in node.inputs().iter().chain(node.outputs()) {
            let Some(buf) = self.bufs.get(&operand.id()) else {
                return Err(
                    Error::new(Code::Internal, format!("操作数 {} 未绑定", operand.id()))
                        .with("engine::MetalExecutor::run"),
                );
            };
            bufs.push(buf);
        }
        Ok(bufs)
    }

    /// 节点自己的操作数约束：每个编号都要已绑定，且输入与输出不能是同一个编号。
    fn check_operands(&self, node: &Node) -> Result<()> {
        check_operands(
            node,
            |id| !self.bufs.contains_key(&id),
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
    fn sync(&mut self, stream: Stream) -> Result<()> {
        let mut idx = 0;
        while idx < self.pending.len() {
            if self.pending[idx].stream() != stream {
                idx += 1;
                continue;
            }
            let node = self.pending[idx].node().clone();
            self.run(&node)?;
            self.pending.remove(idx);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/metal_executor_harness.rs"]
mod harness;
#[cfg(test)]
#[path = "../../tests/metal/mod.rs"]
mod tests;

/// 一条 Metal 共享缓冲句柄（+1 保留，`Drop` 释放）；`contents` 是 CPU 可访问的字节视图。
pub struct MetalBuf {
    pub(super) raw: *mut c_void,
    pub(super) contents: *mut u8,
    pub(super) len: usize,
}

impl MetalBuf {
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

impl Drop for MetalBuf {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// 一条 Metal 计算管线句柄（+1 保留，`Drop` 释放）。
pub struct MetalPipeline {
    pub(super) raw: *mut c_void,
}

impl Drop for MetalPipeline {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

/// 一条 Metal 命令队列句柄（+1 保留，`Drop` 释放）。
pub struct MetalCommandQueue {
    pub(super) raw: *mut c_void,
}

impl MetalCommandQueue {
    /// 执行一条计算管线：把 `bufs` 按数组下标绑定（`setBuffer:offset:atIndex:`，offset=0），
    /// 以 `(grid, block)` 几何派发（`dispatchThreads:threadsPerThreadgroup:`），`commit` 后
    /// `waitUntilCompleted` 同步等完。
    pub fn execute(
        &self,
        pipeline: &MetalPipeline,
        bufs: &[&MetalBuf],
        grid: [u32; 3],
        block: [u32; 3],
    ) -> bool {
        // `commandBuffer` / `computeCommandEncoder` 返回自动释放对象，包一层池统一释放。
        let pool = unsafe { objc_autoreleasePoolPush() };
        let ok = self.execute_with_pool(pipeline, bufs, grid, block);
        unsafe { objc_autoreleasePoolPop(pool) };
        ok
    }

    /// 池内的派发主体：`commandBuffer → computeCommandEncoder → setPipeline → setBuffers →
    /// dispatchThreads → endEncoding → commit → waitUntilCompleted`。
    fn execute_with_pool(
        &self,
        pipeline: &MetalPipeline,
        bufs: &[&MetalBuf],
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
            for (i, buf) in bufs.iter().enumerate() {
                msg_void_p2u(
                    encoder,
                    selector(b"setBuffer:offset:atIndex:\0"),
                    buf.raw,
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
            // 派发本身成功不代表内核跑成了：越界、共享内存超限、张量布局不合法都会让命令
            // 缓冲以 error 收场，而缓冲内容保持原样。不查状态就会把「没写」当成「写错」。
            if msg_usize(cmd, selector(b"status\0")) == COMMAND_BUF_COMPLETED {
                return true;
            }
            report(&msg_id(cmd, selector(b"error\0")));
            false
        }
    }
}

impl Drop for MetalCommandQueue {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
    }
}

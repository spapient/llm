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

mod device;

pub use device::*;

#[cfg(test)]
#[path = "../../tests/metal_executor_harness.rs"]
mod harness;
#[cfg(test)]
#[path = "../../tests/metal/mod.rs"]
mod tests;

/// 一条 Metal 共享缓冲句柄（+1 保留，`Drop` 释放）；`contents` 是 CPU 可访问的字节视图。
pub struct MetalBuffer {
    pub(super) raw: *mut c_void,
    pub(super) contents: *mut u8,
    pub(super) len: usize,
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
    /// 执行一条计算管线：把 `buffers` 按数组下标绑定（`setBuffer:offset:atIndex:`，offset=0），
    /// 以 `(grid, block)` 几何派发（`dispatchThreads:threadsPerThreadgroup:`），`commit` 后
    /// `waitUntilCompleted` 同步等完。
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
            // 派发本身成功不代表内核跑成了：越界、共享内存超限、张量布局不合法都会让命令
            // 缓冲以 error 收场，而缓冲内容保持原样。不查状态就会把「没写」当成「写错」。
            if msg_usize(cmd, selector(b"status\0")) == COMMAND_BUFFER_COMPLETED {
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

//! Metal 设备边界：手写 FFI，设备与命令队列句柄（M3 第二刀）。
//!
//! 越界理由：`MTLCreateSystemDefaultDevice` 与 Objective-C 消息派发（`objc_msgSend`）触及平台 FFI，
//! 属于 ADR 0035 §1「FFI 调用」白名单。与 `cutile` 的差异：`cutile` 生成 ObjC driver 交给 clang 编译、
//! 起子进程执行，本身零 unsafe；而 `Executor::submit` / `Allocator::alloc` 是进程内、有状态、逐步调用，
//! 必须在本进程内直接向 Metal 对象发消息，因此这条边界走手写 `objc_msgSend`（ADR 0035 §3 FFI 默认策略）。
//!
//! 安全不变量：
//! - `MTLCreateSystemDefaultDevice` 与 `newCommandQueue` 都返回 **+1 保留** 的对象，句柄 `Drop` 时恰好
//!   `-release` 一次，不泄漏、不重复释放；
//! - 句柄指针私有、不暴露、不跨线程；本模块只转发两类最安全的 `objc_msgSend`（`(id, SEL) -> id` 与
//!   `(id, SEL) -> void`），含结构体参数/返回值的方法（`dispatchThreads` 等）随内核刀按签名逐条登记；
//! - selector 名都是无参方法（`newCommandQueue` / `release`），无冒号、无参数打包，不会因 selector 拼错
//!   而走到未识别消息。
//!
//! 全仓默认 `deny(unsafe_code)`；本模块是边界 crate 的越界岛，对整个模块放开（ADR 0035 §2）。

#![allow(unsafe_code)]

use std::ffi::{c_char, c_void};

// Objective-C runtime：消息发送与 selector 注册（`libobjc`）。
#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    fn objc_msgSend();
    fn sel_registerName(name: *const c_char) -> *mut c_void;
}

// Metal.framework：创建系统默认 GPU 设备，失败返回 null（+1 保留）。
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut c_void;
}

/// `objc_msgSend` 按 `(id, SEL) -> id` 强转：0 参、返回对象的 ObjC 方法。
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

/// 注册一个 selector；`name` 是带结尾 `\0` 的 C 字符串。
fn selector(name: &[u8]) -> *mut c_void {
    unsafe { sel_registerName(name.as_ptr().cast()) }
}

/// 释放一个 ObjC 对象（`-release`），只在 `Drop` 调用。
unsafe fn release(object: *mut c_void) {
    unsafe { msg_void(object, selector(b"release\0")) };
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

impl Drop for MetalCommandQueue {
    fn drop(&mut self) {
        unsafe { release(self.raw) };
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
}

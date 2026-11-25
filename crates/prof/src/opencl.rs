//! Intel OpenCL 设备边界：设备探测、JIT、类型化缓冲、发射、回读与事件计时。
//!
//! C 符号由 `link_name` 精确保留，Rust 标识符遵守本仓命名；所有裸指针与 FFI 都收在本模块。

#![allow(unsafe_code)]

use diagnostic::{Code, Error, Result};
use std::ffi::{CString, c_char, c_void};
use std::mem::{MaybeUninit, size_of};
use std::ptr::{null, null_mut};

type Raw = *mut c_void;
type Release = unsafe extern "system" fn(Raw) -> i32;

const DEVICE_NOT_FOUND: i32 = -1;
const DEVICE_TYPE_GPU: u64 = 1 << 2;
const MEM_READ_WRITE: u64 = 1;
const BLOCKING: u32 = 1;
const WORK_DIMS: u32 = 3;
const QUEUE_PROPERTIES: usize = 0x1093;
const QUEUE_PROFILING_ENABLE: u64 = 1 << 1;
const KERNEL_NUM_ARGS: u32 = 0x1191;
const PROGRAM_BUILD_LOG: u32 = 0x1183;
const PROFILING_COMMAND_START: u32 = 0x1282;
const PROFILING_COMMAND_END: u32 = 0x1283;
const DEVICE_COMPUTE_UNITS: u32 = 0x1002;
const DEVICE_MAX_WORK_GROUP_SIZE: u32 = 0x1004;
const DEVICE_MAX_CLOCK: u32 = 0x100C;
const DEVICE_GLOBAL_MEMORY: u32 = 0x101F;
const DEVICE_LOCAL_MEMORY: u32 = 0x1023;
const DEVICE_NAME: u32 = 0x102B;
const DEVICE_DRIVER: u32 = 0x102D;
const DEVICE_VERSION: u32 = 0x102F;
const DEVICE_EXTENSIONS: u32 = 0x1030;
const DEVICE_DOT_CAPABILITIES: u32 = 0x1073;
const DEVICE_DOT_ACCELERATION_8BIT: u32 = 0x1074;

#[repr(C)]
#[derive(Clone, Copy)]
struct DotI8Props {
    signed_accelerated: u32,
    _unsigned_accelerated: u32,
    _mixed_signedness_accelerated: u32,
    _accumulating_saturating_signed: u32,
    _accumulating_saturating_unsigned: u32,
    _accumulating_saturating_mixed: u32,
}

#[link(name = "OpenCL")]
unsafe extern "system" {
    #[link_name = "clGetPlatformIDs"]
    fn cl_get_platform_ids(num: u32, ids: *mut Raw, count: *mut u32) -> i32;
    #[link_name = "clGetDeviceIDs"]
    fn cl_get_device_ids(platform: Raw, kind: u64, num: u32, ids: *mut Raw, count: *mut u32)
    -> i32;
    #[link_name = "clGetDeviceInfo"]
    fn cl_get_device_info(
        device: Raw,
        key: u32,
        size: usize,
        value: *mut c_void,
        written: *mut usize,
    ) -> i32;
    #[link_name = "clCreateContext"]
    fn cl_create_context(
        properties: *const isize,
        devices: u32,
        device: *const Raw,
        notify: *mut c_void,
        data: *mut c_void,
        status: *mut i32,
    ) -> Raw;
    #[link_name = "clCreateCommandQueueWithProperties"]
    fn cl_create_queue(
        context: Raw,
        device: Raw,
        properties: *const usize,
        status: *mut i32,
    ) -> Raw;
    #[link_name = "clCreateBuffer"]
    fn cl_create_buf(
        context: Raw,
        flags: u64,
        size: usize,
        host: *mut c_void,
        status: *mut i32,
    ) -> Raw;
    #[link_name = "clCreateProgramWithSource"]
    fn cl_create_program(
        context: Raw,
        count: u32,
        strings: *const *const c_char,
        lengths: *const usize,
        status: *mut i32,
    ) -> Raw;
    #[link_name = "clBuildProgram"]
    fn cl_build_program(
        program: Raw,
        devices: u32,
        device: *const Raw,
        options: *const c_char,
        notify: *mut c_void,
        data: *mut c_void,
    ) -> i32;
    #[link_name = "clGetProgramBuildInfo"]
    fn cl_get_build_info(
        program: Raw,
        device: Raw,
        key: u32,
        size: usize,
        value: *mut c_void,
        written: *mut usize,
    ) -> i32;
    #[link_name = "clCreateKernel"]
    fn cl_create_kernel(program: Raw, name: *const c_char, status: *mut i32) -> Raw;
    #[link_name = "clGetKernelInfo"]
    fn cl_get_kernel_info(
        kernel: Raw,
        key: u32,
        size: usize,
        value: *mut c_void,
        written: *mut usize,
    ) -> i32;
    #[link_name = "clSetKernelArg"]
    fn cl_set_kernel_arg(kernel: Raw, idx: u32, size: usize, value: *const c_void) -> i32;
    #[link_name = "clEnqueueWriteBuffer"]
    fn cl_enqueue_write_buf(
        queue: Raw,
        buf: Raw,
        blocking: u32,
        offset: usize,
        size: usize,
        host: *const c_void,
        waiting: u32,
        events: *const Raw,
        event: *mut Raw,
    ) -> i32;
    #[link_name = "clEnqueueReadBuffer"]
    fn cl_enqueue_read_buf(
        queue: Raw,
        buf: Raw,
        blocking: u32,
        offset: usize,
        size: usize,
        host: *mut c_void,
        waiting: u32,
        events: *const Raw,
        event: *mut Raw,
    ) -> i32;
    #[link_name = "clEnqueueCopyBuffer"]
    fn cl_enqueue_copy_buf(
        queue: Raw,
        source: Raw,
        destination: Raw,
        source_offset: usize,
        destination_offset: usize,
        size: usize,
        waiting: u32,
        events: *const Raw,
        event: *mut Raw,
    ) -> i32;
    #[link_name = "clEnqueueNDRangeKernel"]
    fn cl_enqueue_kernel(
        queue: Raw,
        kernel: Raw,
        dimensions: u32,
        offset: *const usize,
        global: *const usize,
        local: *const usize,
        waiting: u32,
        events: *const Raw,
        event: *mut Raw,
    ) -> i32;
    #[link_name = "clFinish"]
    fn cl_finish(queue: Raw) -> i32;
    #[link_name = "clGetEventProfilingInfo"]
    fn cl_get_event_info(
        event: Raw,
        key: u32,
        size: usize,
        value: *mut c_void,
        written: *mut usize,
    ) -> i32;
    #[link_name = "clReleaseEvent"]
    fn cl_release_event(handle: Raw) -> i32;
    #[link_name = "clReleaseKernel"]
    fn cl_release_kernel(handle: Raw) -> i32;
    #[link_name = "clReleaseProgram"]
    fn cl_release_program(handle: Raw) -> i32;
    #[link_name = "clReleaseMemObject"]
    fn cl_release_mem(handle: Raw) -> i32;
    #[link_name = "clReleaseCommandQueue"]
    fn cl_release_queue(handle: Raw) -> i32;
    #[link_name = "clReleaseContext"]
    fn cl_release_context(handle: Raw) -> i32;
}

/// 一台 Intel GPU 的实测身份与 Roofline 参数。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenClInfo {
    /// 设备全名。
    pub name: String,
    /// 设备支持的 OpenCL 版本。
    pub version: String,
    /// 当前驱动版本。
    pub driver: String,
    /// 执行单元数量。
    pub compute_units: u32,
    /// 设备自报的最高时钟（MHz）。
    pub max_clock_mhz: u32,
    /// 全局内存容量（字节）。
    pub global_memory_bytes: u64,
    /// 单个工作组可用的本地内存（字节）。
    pub local_memory_bytes: u64,
    /// 最大工作组尺寸。
    pub max_work_group_size: usize,
    /// 设备扩展闭集原文。
    pub extensions: String,
    /// `cl_khr_integer_dot_product` 能力位掩码。
    pub dot_i8_capabilities: u64,
    /// 驱动是否声明有符号 8-bit 点积硬件加速。
    pub dot_i8_accelerated: bool,
}

/// 拥有上下文与命令队列的 Intel OpenCL 设备。
pub struct OpenClDevice {
    raw: Raw,
    // 字段按声明序析构：队列先于它所属的上下文释放。
    queue: Handle,
    context: Handle,
    info: OpenClInfo,
}

impl OpenClDevice {
    /// 探测首台 Intel OpenCL GPU，不创建上下文。
    #[must_use]
    pub fn probe() -> Option<OpenClInfo> {
        static INFO: std::sync::OnceLock<Option<OpenClInfo>> = std::sync::OnceLock::new();
        INFO.get_or_init(|| find_device().and_then(|(_, raw)| device_info(raw).ok()))
            .clone()
    }

    /// 创建首台 Intel GPU 的上下文与带性能分析能力的顺序队列。
    ///
    /// # Errors
    /// 没有 Intel GPU、驱动调用失败或上下文/队列创建失败时返回诊断。
    pub fn new() -> Result<Self> {
        let (_, raw) = find_device().ok_or_else(|| {
            Error::new(Code::Unsupported, "没有可用的 Intel OpenCL GPU")
                .with("prof::runtime::opencl::OpenClDevice::new")
        })?;
        Self::with_device(raw)
    }

    fn with_device(raw: Raw) -> Result<Self> {
        let info = device_info(raw)?;
        let context = create_context(raw)?;
        let queue = create_queue(context.raw, raw)?;
        Ok(Self {
            raw,
            queue,
            context,
            info,
        })
    }

    /// 设备身份与能力。
    #[must_use]
    pub fn info(&self) -> &OpenClInfo {
        &self.info
    }

    /// 从 OpenCL C 源文本编译一个入口；失败诊断包含完整构建日志。
    ///
    /// # Errors
    /// 源码含 NUL、程序创建/构建失败或入口不存在时返回诊断。
    pub fn compile(&self, source: &str, options: &str, entry: &str) -> Result<OpenClKernel> {
        let [source, options, entry] = kernel_cstrings(source, options, entry)?;
        let program = self.create_program(&source)?;
        self.build_program(&program, &options)?;
        let kernel = self.create_kernel(&program, &entry)?;
        Ok(OpenClKernel {
            kernel,
            _program: program,
        })
    }

    fn create_program(&self, source: &CString) -> Result<Handle> {
        let mut status = 0;
        let pointer = source.as_ptr();
        // SAFETY: 上下文有效，源码指针在调用期间有效，状态指针可写。
        let raw = unsafe { cl_create_program(self.context.raw, 1, &pointer, null(), &mut status) };
        check(status, "clCreateProgramWithSource")?;
        Handle::new(raw, cl_release_program)
    }

    fn build_program(&self, program: &Handle, options: &CString) -> Result<()> {
        // SAFETY: 程序、设备与选项有效；不使用回调或用户数据。
        let built = unsafe {
            cl_build_program(
                program.raw,
                1,
                &self.raw,
                options.as_ptr(),
                null_mut(),
                null_mut(),
            )
        };
        if built == 0 {
            return Ok(());
        }
        let log = self.build_log(program.raw)?;
        Err(Error::new(
            Code::Invalid,
            format!("OpenCL 内核构建失败（{built}）：\n{log}"),
        )
        .with("prof::runtime::opencl::OpenClDevice::compile"))
    }

    fn create_kernel(&self, program: &Handle, entry: &CString) -> Result<Handle> {
        let mut status = 0;
        // SAFETY: 程序已构建成功，入口名以 NUL 结尾，状态指针可写。
        let raw = unsafe { cl_create_kernel(program.raw, entry.as_ptr(), &mut status) };
        check(status, "clCreateKernel")?;
        Handle::new(raw, cl_release_kernel)
    }

    fn build_log(&self, program: Raw) -> Result<String> {
        // SAFETY: 程序与设备有效；查询助手保证尺寸和数据写入指针有效。
        let text = query_bytes("clGetProgramBuildInfo", |size, value, written| unsafe {
            cl_get_build_info(program, self.raw, PROGRAM_BUILD_LOG, size, value, written)
        })?;
        Ok(decode_text(&text))
    }

    /// 分配一段 `f32` 设备缓冲。
    ///
    /// # Errors
    /// 元素数为零、字节数溢出或驱动分配失败时返回诊断。
    pub fn alloc(&self, elements: usize) -> Result<OpenClBuf> {
        self.alloc_typed::<f32>(elements)
    }

    /// 分配一段 `i32` 设备缓冲，用于计数页。
    ///
    /// # Errors
    /// 元素数为零、字节数溢出或驱动分配失败时返回诊断。
    pub fn alloc_i32(&self, elements: usize) -> Result<OpenClBuf> {
        self.alloc_typed::<i32>(elements)
    }

    fn alloc_typed<T: OpenClScalar>(&self, elements: usize) -> Result<OpenClBuf> {
        let bytes = Self::buf_bytes::<T>(elements)?;
        let handle = self.create_buf(bytes)?;
        Ok(OpenClBuf {
            handle,
            kind: T::KIND,
            bytes,
        })
    }

    /// 零长度与字节溢出共用同一次受检计算。
    fn buf_bytes<T>(elements: usize) -> Result<usize> {
        elements
            .checked_mul(size_of::<T>())
            .filter(|&bytes| bytes > 0)
            .ok_or_else(|| invalid("OpenCL 缓冲长度必须大于零且字节数不溢出"))
    }

    fn create_buf(&self, bytes: usize) -> Result<Handle> {
        let mut status = 0;
        // SAFETY: 上下文有效，长度非零且已受溢出检查，不使用主机指针。
        let raw = unsafe {
            cl_create_buf(
                self.context.raw,
                MEM_READ_WRITE,
                bytes,
                null_mut(),
                &mut status,
            )
        };
        check(status, "clCreateBuffer")?;
        Handle::new(raw, cl_release_mem)
    }

    /// 把主机 `f32` 切片完整写入设备缓冲。
    ///
    /// # Errors
    /// 类型或切片长度不符、入队失败时返回诊断。
    pub fn write_f32(&self, buf: &mut OpenClBuf, values: &[f32]) -> Result<()> {
        self.write(buf, values)
    }

    fn write<T: OpenClScalar>(&self, buf: &mut OpenClBuf, values: &[T]) -> Result<()> {
        if buf.kind != T::KIND {
            return Err(invalid("OpenCL 缓冲元素类型不符"));
        }
        let bytes = size_of_val(values);
        if bytes != buf.bytes {
            return Err(invalid(format!(
                "写入 {bytes} 字节，OpenCL 缓冲是 {} 字节",
                buf.bytes
            )));
        }
        // SAFETY: 类型和字节数已核对，阻塞写返回前主机切片保持有效。
        check(
            unsafe {
                cl_enqueue_write_buf(
                    self.queue.raw,
                    buf.handle.raw,
                    BLOCKING,
                    0,
                    bytes,
                    values.as_ptr().cast(),
                    0,
                    null(),
                    null_mut(),
                )
            },
            "clEnqueueWriteBuffer",
        )
    }

    /// 回读整个 `f32` 设备缓冲。
    ///
    /// # Errors
    /// 类型不符或回读失败时返回诊断。
    pub fn read_f32(&self, buf: &OpenClBuf) -> Result<Vec<f32>> {
        self.read_typed(buf)
    }

    /// 回读整个 `i32` 设备缓冲。
    ///
    /// # Errors
    /// 类型不符或回读失败时返回诊断。
    pub fn read_i32(&self, buf: &OpenClBuf) -> Result<Vec<i32>> {
        self.read_typed(buf)
    }

    fn read_typed<T: OpenClScalar>(&self, buf: &OpenClBuf) -> Result<Vec<T>> {
        if buf.kind != T::KIND {
            return Err(invalid("OpenCL 缓冲元素类型不符"));
        }
        let width = size_of::<T>();
        if width == 0 || !buf.bytes.is_multiple_of(width) {
            return Err(invalid("OpenCL 缓冲长度与元素类型不对齐"));
        }
        let mut values = vec![T::default(); buf.bytes / width];
        // SAFETY: 类型化 Vec 拥有对齐、可写的完整输出范围，阻塞读不保留指针。
        check(
            unsafe {
                cl_enqueue_read_buf(
                    self.queue.raw,
                    buf.handle.raw,
                    BLOCKING,
                    0,
                    buf.bytes,
                    values.as_mut_ptr().cast(),
                    0,
                    null(),
                    null_mut(),
                )
            },
            "clEnqueueReadBuffer",
        )?;
        Ok(values)
    }

    /// 发射内核并等待完成。
    ///
    /// # Errors
    /// 参数绑定、发射或同步失败时返回诊断。
    pub fn launch(
        &self,
        kernel: &OpenClKernel,
        bufs: &[&OpenClBuf],
        global: [usize; 3],
        local: Option<[usize; 3]>,
    ) -> Result<()> {
        let _event = self.enqueue(kernel, bufs, global, local)?;
        // SAFETY: 队列有效，完成等待后再释放事件。
        check(unsafe { cl_finish(self.queue.raw) }, "clFinish")
    }

    /// 发射内核，等待完成并返回设备事件的 `END - START` 纳秒数。
    ///
    /// # Errors
    /// 参数绑定、发射、同步或性能事件读取失败时返回诊断。
    pub fn launch_timed(
        &self,
        kernel: &OpenClKernel,
        bufs: &[&OpenClBuf],
        global: [usize; 3],
        local: Option<[usize; 3]>,
    ) -> Result<u64> {
        let event = self.enqueue(kernel, bufs, global, local)?;
        // SAFETY: 队列有效，完成后再读取事件时间。
        check(unsafe { cl_finish(self.queue.raw) }, "clFinish")?;
        event_elapsed(event.raw)
    }

    /// 用驱动原生缓冲拷贝复制整个输入，并返回设备事件纳秒数。
    ///
    /// # Errors
    /// 类型或长度不同、拷贝入队、同步或性能事件读取失败时返回诊断。
    pub fn copy_timed(&self, input: &OpenClBuf, output: &mut OpenClBuf) -> Result<u64> {
        if input.bytes != output.bytes || input.kind != output.kind {
            return Err(invalid("OpenCL 原生拷贝要求两个缓冲同类型且等长"));
        }
        let event = self.copy(input, output)?;
        // SAFETY: 队列有效，完成后事件时间才可读取。
        check(unsafe { cl_finish(self.queue.raw) }, "clFinish")?;
        event_elapsed(event.raw)
    }

    fn copy(&self, input: &OpenClBuf, output: &mut OpenClBuf) -> Result<Handle> {
        let mut event = null_mut();
        // SAFETY: 两个同类型缓冲等长，源与目的范围完整有效，事件指针可写。
        let status = unsafe {
            cl_enqueue_copy_buf(
                self.queue.raw,
                input.handle.raw,
                output.handle.raw,
                0,
                0,
                input.bytes,
                0,
                null(),
                &mut event,
            )
        };
        check(status, "clEnqueueCopyBuffer")?;
        Handle::new(event, cl_release_event)
    }

    fn enqueue(
        &self,
        kernel: &OpenClKernel,
        bufs: &[&OpenClBuf],
        global: [usize; 3],
        local: Option<[usize; 3]>,
    ) -> Result<Handle> {
        check_launch_shape(global, local.as_ref())?;
        check_arg_count(kernel, bufs.len())?;
        bind_args(kernel, bufs)?;
        let local = local.as_ref().map_or(null(), |shape| shape.as_ptr());
        let mut event = null_mut();
        // SAFETY: 工作尺寸与参数均已验证，数组在入队调用期间有效。
        check(
            unsafe {
                cl_enqueue_kernel(
                    self.queue.raw,
                    kernel.kernel.raw,
                    WORK_DIMS,
                    null(),
                    global.as_ptr(),
                    local,
                    0,
                    null(),
                    &mut event,
                )
            },
            "clEnqueueNDRangeKernel",
        )?;
        Handle::new(event, cl_release_event)
    }
}

/// 设备支持的标量配置：类型决定元素宽度与缓冲档，不在热路径做动态分派。
trait OpenClScalar: Copy + Default + Sized + 'static {
    const KIND: BufKind;
}

impl OpenClScalar for f32 {
    const KIND: BufKind = BufKind::F32;
}

impl OpenClScalar for i32 {
    const KIND: BufKind = BufKind::I32;
}

/// 类型化设备缓冲的元素档。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BufKind {
    F32,
    I32,
}

/// 拥有元素档、字节数与释放责任的 OpenCL 设备缓冲。
pub struct OpenClBuf {
    handle: Handle,
    kind: BufKind,
    bytes: usize,
}

impl OpenClBuf {
    /// 缓冲字节数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// 缓冲是否为空；分配器拒绝零长缓冲，因此正常实例恒为 `false`。
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
}

/// 编译后的 OpenCL 内核；程序句柄与内核同寿命。
pub struct OpenClKernel {
    kernel: Handle,
    _program: Handle,
}

struct Handle {
    raw: Raw,
    release: Release,
}

impl Handle {
    fn new(raw: Raw, release: Release) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::new(Code::Internal, "OpenCL 返回空句柄"));
        }
        Ok(Self { raw, release })
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: 句柄由本对象独占且只释放一次。
        unsafe { (self.release)(self.raw) };
    }
}

fn create_context(device: Raw) -> Result<Handle> {
    let mut status = 0;
    // SAFETY: 设备来自成功探测，设备与状态指针均有效。
    let raw = unsafe { cl_create_context(null(), 1, &device, null_mut(), null_mut(), &mut status) };
    check(status, "clCreateContext")?;
    Handle::new(raw, cl_release_context)
}

fn create_queue(context: Raw, device: Raw) -> Result<Handle> {
    let mut status = 0;
    let properties = [QUEUE_PROPERTIES, QUEUE_PROFILING_ENABLE as usize, 0];
    // SAFETY: 上下文与设备有效，属性数组以 0 终止，状态指针可写。
    let raw = unsafe { cl_create_queue(context, device, properties.as_ptr(), &mut status) };
    check(status, "clCreateCommandQueueWithProperties")?;
    Handle::new(raw, cl_release_queue)
}

fn kernel_cstrings(source: &str, options: &str, entry: &str) -> Result<[CString; 3]> {
    Ok([
        CString::new(source).map_err(|_| invalid("OpenCL 源文本包含 NUL"))?,
        CString::new(options).map_err(|_| invalid("OpenCL 编译参数包含 NUL"))?,
        CString::new(entry).map_err(|_| invalid("OpenCL 入口名包含 NUL"))?,
    ])
}

fn check_launch_shape(global: [usize; 3], local: Option<&[usize; 3]>) -> Result<()> {
    if global.contains(&0) || local.is_some_and(|shape| shape.contains(&0)) {
        return Err(invalid("OpenCL 全局与局部工作尺寸必须非零"));
    }
    if local.is_some_and(|shape| (0..3).any(|axis| !global[axis].is_multiple_of(shape[axis]))) {
        return Err(invalid("OpenCL 全局工作尺寸必须能被局部尺寸整除"));
    }
    Ok(())
}

fn check_arg_count(kernel: &OpenClKernel, actual: usize) -> Result<()> {
    let expected: u32 = query_scalar("clGetKernelInfo", |value| {
        // SAFETY: 内核有效，value 指向一个 u32 的可写存储。
        unsafe {
            cl_get_kernel_info(
                kernel.kernel.raw,
                KERNEL_NUM_ARGS,
                size_of::<u32>(),
                value,
                null_mut(),
            )
        }
    })?;
    if expected as usize == actual {
        return Ok(());
    }
    Err(invalid(format!(
        "OpenCL 内核要 {expected} 个参数，实际给了 {actual} 个"
    )))
}

fn bind_args(kernel: &OpenClKernel, bufs: &[&OpenClBuf]) -> Result<()> {
    for (idx, buf) in bufs.iter().enumerate() {
        let idx = u32::try_from(idx).map_err(|_| invalid("OpenCL 参数个数超过 u32 上限"))?;
        let raw = buf.handle.raw;
        // SAFETY: 内核有效，raw 指向存活的设备句柄，大小与指针类型一致。
        check(
            unsafe {
                cl_set_kernel_arg(
                    kernel.kernel.raw,
                    idx,
                    size_of::<Raw>(),
                    (&raw as *const Raw).cast(),
                )
            },
            "clSetKernelArg",
        )?;
    }
    Ok(())
}

fn event_elapsed(event: Raw) -> Result<u64> {
    let start = event_info(event, PROFILING_COMMAND_START)?;
    let end = event_info(event, PROFILING_COMMAND_END)?;
    end.checked_sub(start)
        .ok_or_else(|| invalid("OpenCL 事件结束时间早于开始时间"))
}

fn find_device() -> Option<(Raw, Raw)> {
    platforms().ok()?.into_iter().find_map(|platform| {
        devices(platform).ok()?.into_iter().find_map(|device| {
            device_string(device, DEVICE_NAME)
                .ok()?
                .contains("Intel")
                .then_some((platform, device))
        })
    })
}

fn platforms() -> Result<Vec<Raw>> {
    let mut count = 0;
    // SAFETY: 空指针形式只查询数量，count 指向有效可写整数。
    check(
        unsafe { cl_get_platform_ids(0, null_mut(), &mut count) },
        "clGetPlatformIDs(count)",
    )?;
    let mut values = vec![null_mut(); count as usize];
    // SAFETY: values 拥有 count 个句柄槽位，驱动不得越界写入。
    check(
        unsafe { cl_get_platform_ids(count, values.as_mut_ptr(), null_mut()) },
        "clGetPlatformIDs",
    )?;
    Ok(values)
}

fn devices(platform: Raw) -> Result<Vec<Raw>> {
    let mut count = 0;
    // SAFETY: 平台有效，空指针形式只查询 GPU 数量。
    let status = unsafe { cl_get_device_ids(platform, DEVICE_TYPE_GPU, 0, null_mut(), &mut count) };
    if status == DEVICE_NOT_FOUND {
        return Ok(Vec::new());
    }
    check(status, "clGetDeviceIDs(count)")?;
    let mut values = vec![null_mut(); count as usize];
    // SAFETY: values 拥有 count 个句柄槽位，驱动不得越界写入。
    check(
        unsafe {
            cl_get_device_ids(
                platform,
                DEVICE_TYPE_GPU,
                count,
                values.as_mut_ptr(),
                null_mut(),
            )
        },
        "clGetDeviceIDs",
    )?;
    Ok(values)
}

fn device_info(raw: Raw) -> Result<OpenClInfo> {
    let [name, version, driver, extensions] = device_text(raw)?;
    let (compute_units, max_clock_mhz) = device_compute(raw)?;
    let (global_memory_bytes, local_memory_bytes, max_work_group_size) = device_memory(raw)?;
    let (dot_i8_capabilities, dot_i8_accelerated) = device_dot(raw, &extensions)?;
    Ok(OpenClInfo {
        name,
        version,
        driver,
        compute_units,
        max_clock_mhz,
        global_memory_bytes,
        local_memory_bytes,
        max_work_group_size,
        extensions,
        dot_i8_capabilities,
        dot_i8_accelerated,
    })
}

fn device_text(raw: Raw) -> Result<[String; 4]> {
    Ok([
        device_string(raw, DEVICE_NAME)?,
        device_string(raw, DEVICE_VERSION)?,
        device_string(raw, DEVICE_DRIVER)?,
        device_string(raw, DEVICE_EXTENSIONS)?,
    ])
}

fn device_compute(raw: Raw) -> Result<(u32, u32)> {
    Ok((
        device_scalar(raw, DEVICE_COMPUTE_UNITS)?,
        device_scalar(raw, DEVICE_MAX_CLOCK)?,
    ))
}

fn device_memory(raw: Raw) -> Result<(u64, u64, usize)> {
    Ok((
        device_scalar(raw, DEVICE_GLOBAL_MEMORY)?,
        device_scalar(raw, DEVICE_LOCAL_MEMORY)?,
        device_scalar(raw, DEVICE_MAX_WORK_GROUP_SIZE)?,
    ))
}

fn device_dot(raw: Raw, extensions: &str) -> Result<(u64, bool)> {
    if !extensions.contains("cl_khr_integer_dot_product") {
        return Ok((0, false));
    }
    let capabilities = device_scalar(raw, DEVICE_DOT_CAPABILITIES)?;
    let props: DotI8Props = query_scalar("clGetDeviceInfo(dot_i8)", |value| {
        // SAFETY: value 指向完整的六字段 cl_bool 属性结构。
        unsafe {
            cl_get_device_info(
                raw,
                DEVICE_DOT_ACCELERATION_8BIT,
                size_of::<DotI8Props>(),
                value,
                null_mut(),
            )
        }
    })?;
    Ok((capabilities, props.signed_accelerated != 0))
}

fn device_string(raw: Raw, key: u32) -> Result<String> {
    // SAFETY: 设备有效；查询助手保证两次调用使用正确的尺寸与指针。
    let value = query_bytes("clGetDeviceInfo", |size, output, written| unsafe {
        cl_get_device_info(raw, key, size, output, written)
    })?;
    Ok(decode_text(&value))
}

fn decode_text(value: &[u8]) -> String {
    String::from_utf8_lossy(value)
        .trim_end_matches('\0')
        .to_string()
}

fn query_bytes(
    operation: &str,
    mut query: impl FnMut(usize, *mut c_void, *mut usize) -> i32,
) -> Result<Vec<u8>> {
    let mut bytes = 0;
    check(query(0, null_mut(), &mut bytes), operation)?;
    let mut value = vec![0_u8; bytes];
    check(
        query(bytes, value.as_mut_ptr().cast(), null_mut()),
        operation,
    )?;
    Ok(value)
}

fn device_scalar<T: Copy>(raw: Raw, key: u32) -> Result<T> {
    query_scalar("clGetDeviceInfo(scalar)", |value| {
        // SAFETY: value 指向 T 大小的可写存储，成功后由查询助手读取。
        unsafe { cl_get_device_info(raw, key, size_of::<T>(), value, null_mut()) }
    })
}

fn event_info(event: Raw, key: u32) -> Result<u64> {
    query_scalar("clGetEventProfilingInfo", |value| {
        // SAFETY: 事件有效，value 指向一个 u64 的可写存储。
        unsafe { cl_get_event_info(event, key, size_of::<u64>(), value, null_mut()) }
    })
}

fn query_scalar<T: Copy>(operation: &str, query: impl FnOnce(*mut c_void) -> i32) -> Result<T> {
    let mut value = MaybeUninit::<T>::uninit();
    check(query(value.as_mut_ptr().cast()), operation)?;
    // SAFETY: 驱动按 key 对应的 T 大小初始化了该值。
    Ok(unsafe { value.assume_init() })
}

fn check(status: i32, operation: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    Err(Error::new(
        Code::Internal,
        format!("{operation} 失败，OpenCL 错误码 {status}"),
    )
    .with("prof::runtime::opencl"))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(Code::Invalid, message.into()).with("prof::runtime::opencl")
}

//! Intel OpenCL 设备边界：设备探测、运行期编译、缓冲、发射、回读与事件计时。
//!
//! 所有 OpenCL FFI 与裸指针都收在本模块；调用方只接触拥有所有权的句柄和类型化切片。

#![allow(unsafe_code)]

use diagnostic::{Code, Error, Result};
use std::ffi::{c_char, c_void};
use std::mem::{MaybeUninit, size_of};
#[cfg(test)]
use std::path::Path;
use std::ptr::{null, null_mut};

const DEVICE_NOT_FOUND: i32 = -1;
const DEVICE_TYPE_GPU: u64 = 1 << 2;
const QUEUE_PROPERTIES: usize = 0x1093;
const QUEUE_PROFILING_ENABLE: u64 = 1 << 1;
const PROFILING_COMMAND_START: u32 = 0x1282;
const PROFILING_COMMAND_END: u32 = 0x1283;

#[link(name = "OpenCL")]
unsafe extern "system" {
    fn clGetPlatformIDs(num: u32, ids: *mut *mut c_void, count: *mut u32) -> i32;
    fn clGetDeviceIDs(
        platform: *mut c_void,
        kind: u64,
        num: u32,
        ids: *mut *mut c_void,
        count: *mut u32,
    ) -> i32;
    fn clGetDeviceInfo(
        device: *mut c_void,
        key: u32,
        size: usize,
        value: *mut c_void,
        written: *mut usize,
    ) -> i32;
    fn clCreateContext(
        properties: *const isize,
        devices: u32,
        device: *const *mut c_void,
        notify: *mut c_void,
        data: *mut c_void,
        status: *mut i32,
    ) -> *mut c_void;
    fn clCreateCommandQueueWithProperties(
        context: *mut c_void,
        device: *mut c_void,
        properties: *const usize,
        status: *mut i32,
    ) -> *mut c_void;
    fn clCreateBuffer(
        context: *mut c_void,
        flags: u64,
        size: usize,
        host: *mut c_void,
        status: *mut i32,
    ) -> *mut c_void;
    fn clCreateProgramWithSource(
        context: *mut c_void,
        count: u32,
        strings: *const *const c_char,
        lengths: *const usize,
        status: *mut i32,
    ) -> *mut c_void;
    fn clBuildProgram(
        program: *mut c_void,
        devices: u32,
        device: *const *mut c_void,
        options: *const c_char,
        notify: *mut c_void,
        data: *mut c_void,
    ) -> i32;
    fn clGetProgramBuildInfo(
        program: *mut c_void,
        device: *mut c_void,
        key: u32,
        size: usize,
        value: *mut c_void,
        written: *mut usize,
    ) -> i32;
    fn clCreateKernel(program: *mut c_void, name: *const c_char, status: *mut i32) -> *mut c_void;
    fn clSetKernelArg(kernel: *mut c_void, index: u32, size: usize, value: *const c_void) -> i32;
    fn clEnqueueWriteBuffer(
        queue: *mut c_void,
        buffer: *mut c_void,
        blocking: u32,
        offset: usize,
        size: usize,
        host: *const c_void,
        waiting: u32,
        events: *const *mut c_void,
        event: *mut *mut c_void,
    ) -> i32;
    fn clEnqueueReadBuffer(
        queue: *mut c_void,
        buffer: *mut c_void,
        blocking: u32,
        offset: usize,
        size: usize,
        host: *mut c_void,
        waiting: u32,
        events: *const *mut c_void,
        event: *mut *mut c_void,
    ) -> i32;
    fn clEnqueueNDRangeKernel(
        queue: *mut c_void,
        kernel: *mut c_void,
        dimensions: u32,
        offset: *const usize,
        global: *const usize,
        local: *const usize,
        waiting: u32,
        events: *const *mut c_void,
        event: *mut *mut c_void,
    ) -> i32;
    fn clFinish(queue: *mut c_void) -> i32;
    fn clGetEventProfilingInfo(
        event: *mut c_void,
        key: u32,
        size: usize,
        value: *mut c_void,
        written: *mut usize,
    ) -> i32;
    fn clReleaseEvent(handle: *mut c_void) -> i32;
    fn clReleaseKernel(handle: *mut c_void) -> i32;
    fn clReleaseProgram(handle: *mut c_void) -> i32;
    fn clReleaseMemObject(handle: *mut c_void) -> i32;
    fn clReleaseCommandQueue(handle: *mut c_void) -> i32;
    fn clReleaseContext(handle: *mut c_void) -> i32;
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
}

/// 拥有上下?文与命令队列的 Intel OpenCL 设备。
pub struct OpenClDevice {
    raw: *mut c_void,
    context: Handle,
    queue: Handle,
    info: OpenClInfo,
}

impl OpenClDevice {
    /// 探测首台 Intel OpenCL GPU，不创建上下文。
    #[must_use]
    pub fn probe() -> Option<OpenClInfo> {
        find_device().and_then(|(_, raw)| device_info(raw).ok())
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
        let info = device_info(raw)?;
        let mut status = 0;
        let context =
            unsafe { clCreateContext(null(), 1, &raw, null_mut(), null_mut(), &mut status) };
        check(status, "clCreateContext")?;
        let context = Handle::new(context, clReleaseContext)?;
        let properties = [QUEUE_PROPERTIES, QUEUE_PROFILING_ENABLE as usize, 0];
        let queue = unsafe {
            clCreateCommandQueueWithProperties(context.raw, raw, properties.as_ptr(), &mut status)
        };
        check(status, "clCreateCommandQueueWithProperties")?;
        Ok(Self {
            raw,
            context,
            queue: Handle::new(queue, clReleaseCommandQueue)?,
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
        let source =
            std::ffi::CString::new(source).map_err(|_| invalid("OpenCL 源文本包含 NUL"))?;
        let options =
            std::ffi::CString::new(options).map_err(|_| invalid("OpenCL 编译参数包含 NUL"))?;
        let entry = std::ffi::CString::new(entry).map_err(|_| invalid("OpenCL 入口名包含 NUL"))?;
        let mut status = 0;
        let pointer = source.as_ptr();
        let program = unsafe {
            clCreateProgramWithSource(self.context.raw, 1, &pointer, null(), &mut status)
        };
        check(status, "clCreateProgramWithSource")?;
        let program = Handle::new(program, clReleaseProgram)?;
        let built = unsafe {
            clBuildProgram(
                program.raw,
                1,
                &self.raw,
                options.as_ptr(),
                null_mut(),
                null_mut(),
            )
        };
        if built != 0 {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "OpenCL 内核构建失败（{built}）：\n{}",
                    self.build_log(program.raw)?
                ),
            )
            .with("prof::runtime::opencl::OpenClDevice::compile"));
        }
        let kernel = unsafe { clCreateKernel(program.raw, entry.as_ptr(), &mut status) };
        check(status, "clCreateKernel")?;
        Ok(OpenClKernel {
            kernel: Handle::new(kernel, clReleaseKernel)?,
            _program: program,
        })
    }

    fn build_log(&self, program: *mut c_void) -> Result<String> {
        let mut bytes = 0;
        check(
            unsafe { clGetProgramBuildInfo(program, self.raw, 0x1183, 0, null_mut(), &mut bytes) },
            "clGetProgramBuildInfo(size)",
        )?;
        let mut text = vec![0_u8; bytes];
        check(
            unsafe {
                clGetProgramBuildInfo(
                    program,
                    self.raw,
                    0x1183,
                    bytes,
                    text.as_mut_ptr().cast(),
                    null_mut(),
                )
            },
            "clGetProgramBuildInfo",
        )?;
        Ok(String::from_utf8_lossy(&text)
            .trim_end_matches('\0')
            .to_string())
    }

    /// 分配一段可读写的设备缓冲。
    ///
    /// # Errors
    /// 长度为零或驱动分配失败时返回诊断。
    pub fn buffer(&self, bytes: usize) -> Result<OpenClBuffer> {
        if bytes == 0 {
            return Err(invalid("OpenCL 缓冲长度必须大于零"));
        }
        let mut status = 0;
        let raw =
            unsafe { clCreateBuffer(self.context.raw, 1 << 0, bytes, null_mut(), &mut status) };
        check(status, "clCreateBuffer")?;
        Ok(OpenClBuffer {
            handle: Handle::new(raw, clReleaseMemObject)?,
            bytes,
        })
    }

    /// 把主机 `f32` 切片完整写入设备缓冲。
    ///
    /// # Errors
    /// 切片字节数与缓冲不一致或入队失败时返回诊断。
    pub fn write_f32(&self, buffer: &OpenClBuffer, values: &[f32]) -> Result<()> {
        self.write(buffer, values)
    }

    /// 回读整个 `f32` 设备缓冲。
    ///
    /// # Errors
    /// 缓冲不是完整的 `f32` 数组或回读失败时返回诊断。
    pub fn read_f32(&self, buffer: &OpenClBuffer) -> Result<Vec<f32>> {
        self.read(buffer)
    }

    /// 回读整个 `i32` 设备缓冲。
    ///
    /// # Errors
    /// 缓冲不是完整的 `i32` 数组或回读失败时返回诊断。
    pub fn read_i32(&self, buffer: &OpenClBuffer) -> Result<Vec<i32>> {
        self.read(buffer)
    }

    fn write<T>(&self, buffer: &OpenClBuffer, values: &[T]) -> Result<()> {
        let bytes = size_of_val(values);
        if bytes != buffer.bytes {
            return Err(invalid(format!(
                "写入 {bytes} 字节，OpenCL 缓冲是 {} 字节",
                buffer.bytes
            )));
        }
        check(
            unsafe {
                clEnqueueWriteBuffer(
                    self.queue.raw,
                    buffer.handle.raw,
                    1,
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

    fn read<T: Copy>(&self, buffer: &OpenClBuffer) -> Result<Vec<T>> {
        if !buffer.bytes.is_multiple_of(size_of::<T>()) {
            return Err(invalid("OpenCL 缓冲长度与元素类型不对齐"));
        }
        let mut values = vec![0_u8; buffer.bytes];
        check(
            unsafe {
                clEnqueueReadBuffer(
                    self.queue.raw,
                    buffer.handle.raw,
                    1,
                    0,
                    buffer.bytes,
                    values.as_mut_ptr().cast(),
                    0,
                    null(),
                    null_mut(),
                )
            },
            "clEnqueueReadBuffer",
        )?;
        let count = buffer.bytes / size_of::<T>();
        let pointer = values.as_mut_ptr().cast::<T>();
        // SAFETY: 缓冲区已成功写满，按 T 对齐且拥有 count * size_of::<T>() 字节；读取后立即复制。
        let typed = unsafe { std::slice::from_raw_parts(pointer, count) }.to_vec();
        Ok(typed)
    }

    /// 发射内核并等待完成。
    ///
    /// # Errors
    /// 参数绑定、发射或同步失败时返回诊断。
    pub fn launch(
        &self,
        kernel: &OpenClKernel,
        buffers: &[&OpenClBuffer],
        global: [usize; 3],
        local: Option<[usize; 3]>,
    ) -> Result<()> {
        self.dispatch(kernel, buffers, global, local, null_mut())?;
        check(unsafe { clFinish(self.queue.raw) }, "clFinish")
    }

    /// 发射内核，等待完成并返回设备事件的 `END - START` 纳秒数。
    ///
    /// # Errors
    /// 参数绑定、发射、同步或性能事件读取失败时返回诊断。
    pub fn launch_timed(
        &self,
        kernel: &OpenClKernel,
        buffers: &[&OpenClBuffer],
        global: [usize; 3],
        local: Option<[usize; 3]>,
    ) -> Result<u64> {
        let mut raw = null_mut();
        self.dispatch(kernel, buffers, global, local, &mut raw)?;
        let event = Handle::new(raw, clReleaseEvent)?;
        check(unsafe { clFinish(self.queue.raw) }, "clFinish")?;
        let start = event_info(event.raw, PROFILING_COMMAND_START)?;
        let end = event_info(event.raw, PROFILING_COMMAND_END)?;
        end.checked_sub(start)
            .ok_or_else(|| invalid("OpenCL 事件结束时间早于开始时间"))
    }

    fn dispatch(
        &self,
        kernel: &OpenClKernel,
        buffers: &[&OpenClBuffer],
        global: [usize; 3],
        local: Option<[usize; 3]>,
        event: *mut *mut c_void,
    ) -> Result<()> {
        if global.contains(&0) || local.is_some_and(|shape| shape.contains(&0)) {
            return Err(invalid("OpenCL 全局与局部工作尺寸必须非零"));
        }
        if local.is_some_and(|shape| (0..3).any(|axis| !global[axis].is_multiple_of(shape[axis]))) {
            return Err(invalid("OpenCL 全局工作尺寸必须能被局部尺寸整除"));
        }
        for (index, buffer) in buffers.iter().enumerate() {
            let raw = buffer.handle.raw;
            check(
                unsafe {
                    clSetKernelArg(
                        kernel.kernel.raw,
                        index as u32,
                        size_of::<*mut c_void>(),
                        (&raw as *const *mut c_void).cast(),
                    )
                },
                "clSetKernelArg",
            )?;
        }
        let local = local.as_ref().map_or(null(), |shape| shape.as_ptr());
        check(
            unsafe {
                clEnqueueNDRangeKernel(
                    self.queue.raw,
                    kernel.kernel.raw,
                    3,
                    null(),
                    global.as_ptr(),
                    local,
                    0,
                    null(),
                    event,
                )
            },
            "clEnqueueNDRangeKernel",
        )
    }
}

/// 拥有长度的 OpenCL 设备缓冲。
pub struct OpenClBuffer {
    handle: Handle,
    bytes: usize,
}

impl OpenClBuffer {
    /// 缓冲字节数。
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// 缓冲是否为空；OpenCL 分配器拒绝零长缓冲，因此正常实例恒为 `false`。
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
    raw: *mut c_void,
    release: unsafe extern "system" fn(*mut c_void) -> i32,
}

impl Handle {
    fn new(
        raw: *mut c_void,
        release: unsafe extern "system" fn(*mut c_void) -> i32,
    ) -> Result<Self> {
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

fn find_device() -> Option<(*mut c_void, *mut c_void)> {
    for platform in platforms().ok()? {
        for device in devices(platform).ok()? {
            let name = device_string(device, 0x102B).ok()?;
            if name.contains("Intel") {
                return Some((platform, device));
            }
        }
    }
    None
}

fn platforms() -> Result<Vec<*mut c_void>> {
    let mut count = 0;
    check(
        unsafe { clGetPlatformIDs(0, null_mut(), &mut count) },
        "clGetPlatformIDs(count)",
    )?;
    let mut values = vec![null_mut(); count as usize];
    check(
        unsafe { clGetPlatformIDs(count, values.as_mut_ptr(), null_mut()) },
        "clGetPlatformIDs",
    )?;
    Ok(values)
}

fn devices(platform: *mut c_void) -> Result<Vec<*mut c_void>> {
    let mut count = 0;
    let status = unsafe { clGetDeviceIDs(platform, DEVICE_TYPE_GPU, 0, null_mut(), &mut count) };
    if status == DEVICE_NOT_FOUND {
        return Ok(Vec::new());
    }
    check(status, "clGetDeviceIDs(count)")?;
    let mut values = vec![null_mut(); count as usize];
    check(
        unsafe {
            clGetDeviceIDs(
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

fn device_info(raw: *mut c_void) -> Result<OpenClInfo> {
    Ok(OpenClInfo {
        name: device_string(raw, 0x102B)?,
        version: device_string(raw, 0x102F)?,
        driver: device_string(raw, 0x102D)?,
        compute_units: device_scalar(raw, 0x1002)?,
        max_clock_mhz: device_scalar(raw, 0x100C)?,
        global_memory_bytes: device_scalar(raw, 0x101F)?,
        local_memory_bytes: device_scalar(raw, 0x1023)?,
        max_work_group_size: device_scalar(raw, 0x1004)?,
        extensions: device_string(raw, 0x1030)?,
    })
}

fn device_string(raw: *mut c_void, key: u32) -> Result<String> {
    let mut bytes = 0;
    check(
        unsafe { clGetDeviceInfo(raw, key, 0, null_mut(), &mut bytes) },
        "clGetDeviceInfo(size)",
    )?;
    let mut value = vec![0_u8; bytes];
    check(
        unsafe { clGetDeviceInfo(raw, key, bytes, value.as_mut_ptr().cast(), null_mut()) },
        "clGetDeviceInfo",
    )?;
    Ok(String::from_utf8_lossy(&value)
        .trim_end_matches('\0')
        .to_string())
}

fn device_scalar<T: Copy>(raw: *mut c_void, key: u32) -> Result<T> {
    let mut value = MaybeUninit::<T>::uninit();
    check(
        unsafe {
            clGetDeviceInfo(
                raw,
                key,
                size_of::<T>(),
                value.as_mut_ptr().cast(),
                null_mut(),
            )
        },
        "clGetDeviceInfo(scalar)",
    )?;
    // SAFETY: OpenCL 已按键对应的 T 大小初始化该值。
    Ok(unsafe { value.assume_init() })
}

fn event_info(event: *mut c_void, key: u32) -> Result<u64> {
    let mut value = MaybeUninit::<u64>::uninit();
    check(
        unsafe {
            clGetEventProfilingInfo(
                event,
                key,
                size_of::<u64>(),
                value.as_mut_ptr().cast(),
                null_mut(),
            )
        },
        "clGetEventProfilingInfo",
    )?;
    // SAFETY: 成功的性能查询恰好写入一个 cl_ulong。
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intel_gpu_executes_and_times_a_real_kernel() {
        let Some(info) = OpenClDevice::probe() else {
            return;
        };
        assert!(info.name.contains("Intel"));
        assert!(info.compute_units > 0);
        let device = OpenClDevice::new().expect("创建 Intel OpenCL 设备");
        let source = "__kernel void copy(__global const float *input, __global float *output) { size_t i = get_global_id(0); output[i] = input[i] * 2.0f; }";
        let kernel = device
            .compile(source, "-cl-std=CL2.0", "copy")
            .expect("编译 OpenCL 内核");
        let input = device.buffer(4 * size_of::<f32>()).expect("输入缓冲");
        let output = device.buffer(4 * size_of::<f32>()).expect("输出缓冲");
        device
            .write_f32(&input, &[1.0, 2.0, 3.0, 4.0])
            .expect("写入输入");
        device
            .launch_timed(&kernel, &[&input, &output], [4, 1, 1], None)
            .expect("发射内核");
        assert_eq!(
            device.read_f32(&output).expect("回读输出"),
            vec![2.0, 4.0, 6.0, 8.0]
        );
    }

    #[test]
    fn windows_opencl_library_is_not_a_source_path() {
        assert!(!Path::new("OpenCL.lib").is_absolute());
    }
}

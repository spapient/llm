use super::*;

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
pub(crate) const COMMAND_BUFFER_COMPLETED: usize = 4;

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
    pub(super) buffers: BTreeMap<u32, MetalBuffer>,
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
            buffers: BTreeMap::new(),
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
    pub fn bind(&mut self, id: u32, buffer: MetalBuffer) -> Option<MetalBuffer> {
        self.buffers.insert(id, buffer)
    }

    /// 分配一段 `len` 字节的共享缓冲并绑定到编号；返回被替换掉的旧缓冲，分配失败返回 `Internal`。
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

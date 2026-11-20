//! 厂商语法面 + 后端无关的 lowering 原语：把「同一条内核 body 拼成不同后端」的语法差异与
//! lowering 逻辑都收进 [`Dialect`] trait。
//!
//! Metal 与 CUDA 两个 vendor 各给一份**手写实现**（[`Metal`] / [`Cuda`] 单元结构体），只声明
//! 厂商差异面（地址空间 / 设备函数 / 缓冲声明 / 入口关键字 / 线程形参与下标语句）；后端无关的
//! lowering 原语（逐点 / 循环注入 / 分块归约 / 分块扫描 / 串行）作为 trait 的**默认方法**在这里
//! 一次实现。vendor 字面量（Metal 的 `device` / `kernel`、CUDA 的 `__device__` / `__global__`）
//! 只经关联常量与方法进出，不散落任何一处。

use diagnostic::{Code, Error, Result};
use op::decl::Grid;
use op::model::Sample;
use op::sig::{Param, Signature};

use cutile::harness::output_name;

/// 分块多段程序的入口名：同一份源里每个核只能出现一次，因此段与段必须分开命名——
/// CUDA 侧与 Metal 侧的 `PARTIAL` / `EXPSUM` / `NORM` / `SCAN_*` 同义，多段驱动按这些名字取
/// pipeline / 派发。
pub(crate) const PARTIAL: &str = "kernel_partial";
pub(crate) const EXPSUM: &str = "kernel_expsum";
pub(crate) const NORM: &str = "kernel_norm";
pub(crate) const SCAN_LOCAL: &str = "kernel_scan_local";
pub(crate) const SCAN_CORRECT: &str = "kernel_scan_correct";

// ---------------------------------------------------------------------------
// 后端无关的渲染助手：不持有任何厂商字面量，只回答「从签名 / 样本算出什么文本」。
// ---------------------------------------------------------------------------

/// 长度标量：签名里唯一非缓冲参数（扫描/归约/归一化的 `N`），取样本值。
pub(crate) fn length_scalar(signature: &Signature, sample: &Sample) -> Result<i64> {
    let scalar = scalar_name(signature)?;
    sample.scalar(scalar).ok_or(Error::new(
        Code::Invalid,
        format!("长度标量 `{scalar}` 没有样本取值"),
    ))
}

/// 唯一长度标量名（非缓冲参数）：归约/扫描/循环注入共用。
pub(crate) fn scalar_name(signature: &Signature) -> Result<&str> {
    signature
        .params()
        .iter()
        .find(|param| !param.ctype.is_buffer())
        .map(|param| param.name.as_str())
        .ok_or(Error::new(Code::Internal, "内核没有长度标量").with("engine::dialect::scalar_name"))
}

/// 只读输入缓冲名（唯一非可写缓冲）：归约/扫描/循环注入三族 lowering 共用这一条查表。
pub(crate) fn read_input(signature: &Signature) -> Result<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
        .find(|param| !param.mutable)
        .map(|param| param.name.as_str())
        .ok_or(
            Error::new(Code::Internal, "内核没有只读输入缓冲").with("engine::dialect::read_input"),
        )
}

/// launch 实参：缓冲传设备指针 `d_<name>`，标量取样本值。
pub(crate) fn launch_args(sample: &Sample, params: &[Param]) -> Result<String> {
    let mut args = Vec::with_capacity(params.len());
    for param in params {
        if param.ctype.is_buffer() {
            args.push(format!("d_{}", param.name));
        } else {
            let value = sample.scalar(&param.name).ok_or_else(|| {
                Error::new(
                    Code::Internal,
                    format!("标量实参 `{}` 没有取值", param.name),
                )
            })?;
            args.push(value.to_string());
        }
    }
    Ok(args.join(", "))
}

/// 生产态实参：缓冲取 `d_{name} + n`（线程处理的元素）、标量取 `1`（每线程一个元素）。
fn grid_args(params: &[Param]) -> String {
    params
        .iter()
        .map(|param| {
            if param.ctype.is_buffer() {
                format!("d_{} + n", param.name)
            } else {
                "1".to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// 定位内核签名：返回 `(右括号位置, 函数体起始位置)`，供追加形参（[`thread_param_typed`]）与
/// 注入块头（[`inject_chunk_preamble`]）共用，避免两处各写一遍 `void {kernel}(` 查找。
/// 体起始是 `) {` 之后的第一个字符——注入时把 `{` 一并留在前缀里。
fn kernel_span(body: &str, kernel: &str) -> Result<(usize, usize)> {
    let marker = format!("void {kernel}(");
    let Some(start) = body.find(&marker) else {
        return Err(
            Error::new(Code::Invalid, format!("找不到内核函数 `void {kernel}(`"))
                .with("engine::dialect::kernel_span"),
        );
    };
    let Some(close) = body[start..].find(") {") else {
        return Err(
            Error::new(Code::Invalid, format!("内核函数 `{kernel}` 签名不完整"))
                .with("engine::dialect::kernel_span"),
        );
    };
    Ok((start + close, start + close + 3))
}

/// 给内部内核函数追加线程下标形参：`void {kernel}(...) {` → `void {kernel}(..., {param}) {`。
/// Metal 传 `uint3 tid`、CUDA 传 `int3 tid`（由 `kernel_main` 从 `blockIdx` 装配，见
/// [`Dialect::TID_INIT`]），其余改写规则一致。
pub(crate) fn thread_param_typed(body: &str, kernel: &str, param: &str) -> Result<String> {
    let (paren_at, _) = kernel_span(body, kernel)?;
    let mut out = String::with_capacity(body.len() + param.len() + 2);
    out.push_str(&body[..paren_at]);
    out.push_str(", ");
    out.push_str(param);
    out.push_str(&body[paren_at..]);
    Ok(out)
}

/// 给内部内核函数体注入块头：`g`/`start`/`len` 三个局部量，供循环头改写与分块寻址引用。
pub(crate) fn inject_chunk_preamble(
    body: &str,
    kernel: &str,
    chunk: usize,
    scalar: &str,
) -> Result<String> {
    let (_, body_at) = kernel_span(body, kernel)?;
    let preamble = format!(
        "    int g = (int)tid.x;\n    int start = g * {chunk};\n    int len = min({chunk}, {scalar} - start);\n"
    );
    let mut out = String::with_capacity(body.len() + preamble.len());
    out.push_str(&body[..body_at]);
    out.push_str(&preamble);
    out.push_str(&body[body_at..]);
    Ok(out)
}

/// 把以 `scalar` 为上界的串行循环头改写到块内边界：`for (int v = 0; v < {scalar}; ++v)` →
/// `for (int v = start; v < start + len; ++v)`。循环变量从模板读出，不对名字硬编码。
pub(crate) fn bind_chunk_loop(body: &str, scalar: &str) -> Result<String> {
    let bound = format!(" < {scalar}; ");
    let Some(bound_at) = body.find(&bound) else {
        return Err(Error::new(
            Code::Invalid,
            format!("模板里找不到以 `{scalar}` 为上界的串行循环"),
        )
        .with("engine::dialect::bind_chunk_loop"));
    };
    let prefix = &body[..bound_at];
    let Some(open) = prefix.rfind("for (int ") else {
        return Err(Error::new(Code::Invalid, "循环上界不在 `for (int …)` 头里")
            .with("engine::dialect::bind_chunk_loop"));
    };
    let var = prefix[open + "for (int ".len()..]
        .split('=')
        .next()
        .unwrap_or("")
        .trim();
    let old_head = format!("for (int {var} = 0; {var} < {scalar}; ++{var})");
    let new_head = format!("for (int {var} = start; {var} < start + len; ++{var})");
    if !body.contains(&old_head) {
        return Err(
            Error::new(Code::Invalid, format!("模板里找不到串行循环 `{old_head}`"))
                .with("engine::dialect::bind_chunk_loop"),
        );
    }
    Ok(body.replace(&old_head, &new_head))
}

/// 把每个网格维的串行循环头绑定到线程下标，轴取 `x`/`y`/`z`。
pub(crate) fn bind_grid_loops(body: String, grid: &Grid) -> Result<String> {
    const AXES: [&str; 3] = ["x", "y", "z"];
    let mut out = body;
    for (dim, (thread, scalar)) in grid.dims.iter().enumerate() {
        let axis = AXES.get(dim).ok_or(Error::new(
            Code::Invalid,
            format!("网格维 {} 超过 Metal 上限 3", grid.dims.len()),
        ))?;
        out = bind_grid_loop(&out, thread, scalar, axis)?;
    }
    Ok(out)
}

/// 把一个网格维的串行循环头绑定到线程下标：`for (int {thread} = 0; {thread} < {scalar}; ++{thread})`
/// → `for (int {thread} = (int)tid.{axis}; {thread} < (int)tid.{axis} + 1; ++{thread})`。
fn bind_grid_loop(body: &str, thread: &str, scalar: &str, axis: &str) -> Result<String> {
    let head = format!("for (int {thread} = 0; {thread} < {scalar}; ++{thread})");
    let bound =
        format!("for (int {thread} = (int)tid.{axis}; {thread} < (int)tid.{axis} + 1; ++{thread})");
    if !body.contains(&head) {
        return Err(
            Error::new(Code::Invalid, format!("模板里找不到网格循环 `{head}`"))
                .with("engine::dialect::bind_grid_loop"),
        );
    }
    Ok(body.replace(&head, &bound))
}

/// 厂商语法面：后端无关的 lowering 原语里需要厂商语法的那几处，抽成方法 + 关联常量，并把
/// lowering 原语本身作为**默认方法**在这里一次实现。
///
/// 每个厂商差异成员只回答一个问题：
///
/// - [`address_space`](Dialect::address_space)：给一条 C 声明补地址空间限定词（Metal 在指针类型
///   前插 `device`；CUDA 恒等）；
/// - [`device_fn`](Dialect::device_fn)：给设备函数名返回限定词（CUDA `__device__`；Metal 恒等）；
/// - [`buffer_decl`](Dialect::buffer_decl)：渲染一条缓冲形参声明（Metal 带 `[[buffer(slot)]]`）；
/// - 关联常量们：入口关键字、逐点 / 网格的线程形参与线程下标语句，以及循环注入 / 分块扫描共用
///   的线程下标形参与装配语句。
///
/// 默认方法（lowering 原语）只面向 trait，不持有任何厂商字面量：`Metal::grid_kernel(...)` 与
/// `Cuda::grid_kernel(...)` 同一条 body，只经上面的差异成员换语法。
pub(crate) trait Dialect {
    /// 给一条 C 声明补地址空间：Metal 在指针类型前插 `device`（`\u{1}` 哨兵先占住 `*`），CUDA
    /// 原样返回。限定词为空即恒等，所以 CUDA 的「无地址空间」与 Metal 的「六型规则」共用同一份
    /// 调用代码。
    fn address_space(decl: &str) -> String;

    /// 设备函数限定词：CUDA 返回 `__device__`，Metal 返回空串（无操作）。
    fn device_fn(name: &str) -> String;

    /// 渲染一条缓冲形参声明；`slot` 是驱动侧 `setBuffer:atIndex:` / 拷贝的槽位序。
    fn buffer_decl(param: &Param, slot: usize) -> String;

    /// 内核入口关键字：Metal `kernel`，CUDA `__global__`。
    const ENTRY: &'static str;
    /// 逐点网格线程形参（Metal `uint n [[thread_position_in_grid]]`；CUDA 空，用 [`Dialect::POINT_STMT`]）。
    const POINT_PARAM: &'static str;
    /// 逐点网格线程下标语句（CUDA 装配 `blockIdx` / `threadIdx`；Metal 空）。
    const POINT_STMT: &'static str;
    /// 分块 / 循环网格线程形参（Metal `uint3 tid [[thread_position_in_grid]]`；CUDA 空）。
    const GRID_PARAM: &'static str;
    /// 分块 / 循环网格线程下标语句（CUDA 装配；Metal `int g = (int)tid.x;`）。
    const GRID_STMT: &'static str;
    /// 循环注入 / 分块扫描里追加给内部内核的线程下标形参（Metal `uint3 tid`；CUDA `int3 tid`）。
    const THREAD_PARAM: &'static str;
    /// `kernel_main` 里装配线程下标的语句：CUDA 从 `blockIdx` 拼 `int3 tid`；Metal 空（线程下标
    /// 由形参 [`Dialect::GRID_PARAM`] 提供）。
    const TID_INIT: &'static str;

    /// 方言化的缓冲声明段：按签名序逐条问 [`Dialect::buffer_decl`]。Metal 带 `[[buffer(slot)]]`，
    /// CUDA 不带；槽位序与驱动侧 `setBuffer:atIndex:` 同源。
    fn buffers(signature: &Signature) -> String {
        signature
            .params()
            .iter()
            .filter(|param| param.ctype.is_buffer())
            .enumerate()
            .map(|(slot, param)| Self::buffer_decl(param, slot))
            .collect::<Vec<_>>()
            .join(",\n")
    }

    /// 方言化的地址空间改写：把六种裸指针拼法逐条交给 [`Dialect::address_space`]，命中就替换。
    /// 限定词为空的方言（CUDA）每条都是恒等、整体无操作；Metal 的六型规则经 `\u{1}` 哨兵与
    /// `device` 前缀逐字节一致。
    fn address_text(text: &str) -> String {
        const SPELLINGS: [&str; 6] = [
            "const float *",
            "const unsigned char *",
            "const u8 *",
            "float *",
            "unsigned char *",
            "u8 *",
        ];
        let mut out = text.to_owned();
        for spelling in SPELLINGS {
            let qualified = Self::address_space(spelling);
            if qualified != spelling {
                out = out.replace(spelling, &qualified);
            }
        }
        out.replace("\u{1}", " *")
    }

    /// 方言化前置：地址空间改写之后，再按 [`Dialect::device_fn`] 补设备函数限定词。Metal 的
    /// 限定词为空 → 只剩地址空间；CUDA 的地址空间为空 → 只剩 `__device__` 注入。
    fn prelude(text: &str, kernel: &str) -> String {
        let text = Self::address_text(text);
        let qual = Self::device_fn(kernel);
        if qual.is_empty() {
            text
        } else {
            text.replace(&format!("void {kernel}("), &format!("{qual} void {kernel}("))
                .replace("static ", &format!("{qual} static "))
        }
    }

    /// 具名入口：`prelude` 变换后的 body 接入口关键字、缓冲段、线程形参与线程下标语句。
    /// 多段设备源同处一个文件时入口名不同，其余拼装规则一致。
    fn kernel_entry(
        source: &str,
        kernel: &str,
        buffers: &str,
        tid: (&str, &str),
        body: &str,
        entry: &str,
    ) -> String {
        let (tid_param, tid_stmt) = tid;
        let param = if tid_param.is_empty() {
            String::new()
        } else {
            format!(",\n    {tid_param}")
        };
        let stmt = if tid_stmt.is_empty() {
            String::new()
        } else {
            format!("    {tid_stmt}\n")
        };
        format!(
            "{}\n\n{} void {entry}(\n{buffers}{param}) {{\n{stmt}{body}\n}}",
            Self::prelude(source, kernel),
            Self::ENTRY,
        )
    }

    /// 生产态逐点网格内核：把逐点（pointwise）串行内核包成按网格派发的 `kernel_main`，后端差异
    /// 收进 `Self`。每个线程处理一个元素：网格索引 `n` 落到每个缓冲的 `+ n` 偏移、唯一的长度标量
    /// 传 `1`。因此**只对逐点族成立**——所有缓冲按元素步长 1 索引、唯一标量是长度 `N`。
    fn grid_kernel(
        kernel_source: &str,
        signature: &Signature,
        guard_len: Option<i64>,
    ) -> Result<String> {
        let scalars = signature
            .params()
            .iter()
            .filter(|param| !param.ctype.is_buffer())
            .count();
        if scalars != 1 {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "`{}` 不是逐点签名（标量 {scalars} 项）：行跨步/归约 lowering 不在此列",
                    signature.op(),
                ),
            )
            .with("engine::dialect::grid_kernel"));
        }
        let call = grid_args(signature.params());
        let buffers = Self::buffers(signature);
        let body = match guard_len {
            Some(len) => format!(
                "    if (n < {len}) {{\n        {}({call});\n    }}",
                signature.kernel()
            ),
            None => format!("    {}({call});", signature.kernel()),
        };
        Ok(Self::kernel_entry(
            kernel_source,
            &signature.kernel(),
            &buffers,
            (Self::POINT_PARAM, Self::POINT_STMT),
            &body,
            "kernel_main",
        ))
    }

    /// 生产态分块归约内核（后端无关）：把长轴切成 `chunk` 元素一块，网格 = `ceil(N / chunk)`，
    /// 每线程用设备 preamble 里的软件归约原语串行归约自己那一块，把**块级部分结果**写到输出缓冲
    /// 的第 `g` 个槽。归约的跨元素依赖不能靠逐元素拆线程，只能「切块 + 主机折叠」。
    fn chunk_reduce(
        kernel_source: &str,
        signature: &Signature,
        sample: &Sample,
        chunk: usize,
        reduce_func: &str,
    ) -> Result<String> {
        Self::chunk_reduce_entry(
            kernel_source,
            signature,
            sample,
            chunk,
            reduce_func,
            "kernel_main",
        )
    }

    /// [`Dialect::chunk_reduce`] 的具名版本：多段归一化 lowering 里部分归约段的入口名与其他段区分。
    fn chunk_reduce_entry(
        kernel_source: &str,
        signature: &Signature,
        sample: &Sample,
        chunk: usize,
        reduce_func: &str,
        entry: &str,
    ) -> Result<String> {
        let input = read_input(signature)?;
        let output = output_name(signature)?;
        let len = length_scalar(signature, sample)?;
        let buffers = Self::buffers(signature);
        let body = format!(
            "    int start = g * {chunk};\n    int len = min({chunk}, {len} - start);\n    d_{output}[g] = {reduce_func}(d_{input} + start, len, 1);"
        );
        Ok(Self::kernel_entry(
            kernel_source,
            &signature.kernel(),
            &buffers,
            (Self::GRID_PARAM, Self::GRID_STMT),
            &body,
            entry,
        ))
    }

    /// 串行（零维网格）：不追加 `tid`、不绑循环，缓冲原样透传、标量取样本值——串行就是循环注入的
    /// 零维特例，不再单列一条 lowering。
    fn serial_kernel(
        kernel_source: &str,
        signature: &Signature,
        sample: &Sample,
    ) -> Result<String> {
        let call = launch_args(sample, signature.params())?;
        let buffers = Self::buffers(signature);
        let body = format!("    {}({call});", signature.kernel());
        Ok(Self::kernel_entry(
            kernel_source,
            &signature.kernel(),
            &buffers,
            ("", ""),
            &body,
            "kernel_main",
        ))
    }

    /// 带线程下标的内核包裹：给内部内核透传 `tid`、声明线程下标形参，循环注入与分块扫描两条
    /// lowering 共用（串行零维用 [`Dialect::serial_kernel`]，不追加 `tid`）。线程下标形参取
    /// [`Dialect::GRID_PARAM`]、装配语句取 [`Dialect::TID_INIT`]、入口关键字取 [`Dialect::ENTRY`]：
    /// CUDA 的 `GRID_PARAM` 为空，`tid` 由 `TID_INIT` 在函数体内装配。
    #[doc(hidden)]
    fn tid_kernel_main(
        body: &str,
        signature: &Signature,
        sample: &Sample,
        entry: &str,
    ) -> Result<String> {
        let call = format!("{}, tid", launch_args(sample, signature.params())?);
        let buffers = Self::buffers(signature);
        let param = if Self::GRID_PARAM.is_empty() {
            String::new()
        } else {
            format!(",\n    {}", Self::GRID_PARAM)
        };
        Ok(format!(
            "{body}\n\n{} void {entry}(\n{buffers}{param}) {{\n{}    {}({call});\n}}",
            Self::ENTRY,
            Self::TID_INIT,
            signature.kernel(),
        ))
    }

    /// 循环注入（非零网格）：给内部内核追加线程下标形参、绑循环头到线程下标、调用点透传 `tid`。
    fn injected_kernel(
        kernel_source: &str,
        signature: &Signature,
        sample: &Sample,
        grid: &Grid,
    ) -> Result<String> {
        let mut body = Self::prelude(kernel_source, &signature.kernel());
        body = thread_param_typed(&body, &signature.kernel(), Self::THREAD_PARAM)?;
        body = bind_grid_loops(body, grid)?;
        Self::tid_kernel_main(&body, signature, sample, "kernel_main")
    }

    /// 局部扫描内核（方言化）：把模板里的时间轴循环头改写到块内 `[start, start+len)`，`tid` 透传进
    /// 内部内核。Metal 与 CUDA 共用：地址空间 / 设备函数 / 线程下标形参与装配由 `Self` 提供。
    fn chunk_scan_local(
        kernel_source: &str,
        signature: &Signature,
        sample: &Sample,
        chunk: usize,
        scalar: &str,
    ) -> Result<String> {
        let kernel = signature.kernel();
        let mut body = Self::prelude(kernel_source, &kernel);
        body = thread_param_typed(&body, &kernel, Self::THREAD_PARAM)?;
        body = inject_chunk_preamble(&body, &kernel, chunk, scalar)?;
        body = bind_chunk_loop(&body, scalar)?;
        Self::tid_kernel_main(&body, signature, sample, SCAN_LOCAL)
    }
}

/// Metal 方言标记：地址空间用 `device`、入口用 `kernel`，线程下标由 `thread_position_in_grid`
/// 形参提供。
pub(crate) struct Metal;

impl Dialect for Metal {
    fn address_space(decl: &str) -> String {
        format!("device {decl}").replace(" *", "\u{1}")
    }

    fn device_fn(_name: &str) -> String {
        String::new()
    }

    fn buffer_decl(param: &Param, slot: usize) -> String {
        format!(
            "    device {}* d_{} [[buffer({})]]",
            param.ctype.c_name(),
            param.name,
            slot
        )
    }

    const ENTRY: &'static str = "kernel";
    const POINT_PARAM: &'static str = "uint n [[thread_position_in_grid]]";
    const POINT_STMT: &'static str = "";
    const GRID_PARAM: &'static str = "uint3 tid [[thread_position_in_grid]]";
    const GRID_STMT: &'static str = "int g = (int)tid.x;";
    const THREAD_PARAM: &'static str = "uint3 tid";
    const TID_INIT: &'static str = "";
}

/// CUDA 方言标记：无地址空间、设备函数用 `__device__`、入口用 `__global__`，线程下标在
/// `kernel_main` 内由 `blockIdx` 装配。
pub(crate) struct Cuda;

impl Dialect for Cuda {
    fn address_space(decl: &str) -> String {
        decl.to_owned()
    }

    fn device_fn(_name: &str) -> String {
        "__device__".to_owned()
    }

    fn buffer_decl(param: &Param, _slot: usize) -> String {
        format!("    {}* d_{}", param.ctype.c_name(), param.name)
    }

    const ENTRY: &'static str = "__global__";
    const POINT_PARAM: &'static str = "";
    const POINT_STMT: &'static str = "int n = blockIdx.x * blockDim.x + threadIdx.x;";
    const GRID_PARAM: &'static str = "";
    const GRID_STMT: &'static str = "int g = blockIdx.x * blockDim.x + threadIdx.x;";
    const THREAD_PARAM: &'static str = "int3 tid";
    const TID_INIT: &'static str =
        "    int3 tid; tid.x = blockIdx.x; tid.y = blockIdx.y; tid.z = blockIdx.z;\n";
}

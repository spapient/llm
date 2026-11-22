//! 职责：方言源码文本：CUDA/Metal 源串拼接。
use super::Dialect;
use diagnostic::{Code, Error, Result};
use op::decl::Grid;
use op::model::{Sample, Values};
use op::sig::{Param, Signature};

use cutile::harness::output_name;

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
        .find(|param| !param.ctype.is_buf())
        .map(|param| param.name.as_str())
        .ok_or(Error::new(Code::Internal, "内核没有长度标量").with("engine::dialect::scalar_name"))
}

/// 只读输入缓冲名（唯一非可写缓冲）：归约/扫描/循环注入三族 lowering 共用这一条查表。
pub(crate) fn read_input(signature: &Signature) -> Result<&str> {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
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
        if param.ctype.is_buf() {
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

/// 回读的浮点位模式打印循环：`union` 保持 float→uint 位级保真，`src[i]` 是第 i 个元素的取值
/// 表达式（CUDA/昇腾是 `h_<name>[i]`，Metal 是 `out[i]`）。三端共用：回读口径必须逐字一致，
/// 否则 `llm_optim::runner::parse_bits` 会读出不同的位模式。
pub(crate) fn readback_bits(src: &str, len: usize, indent: &str) -> String {
    format!(
        "{indent}for (int i = 0; i < {len}; ++i) {{\n\
         {indent}    union {{ float f; unsigned int u; }} v; v.f = {src}[i];\n\
         {indent}    printf(\"%u\\n\", v.u);\n\
         {indent}}}\n"
    )
}

/// 输出缓冲的（名字，元素数）：只接受非空 f32，回读路径按 float 解释字节。三端 benchmark/回读共用。
pub(crate) fn f32_output_shape(signature: &Signature, sample: &Sample) -> Result<(String, usize)> {
    let output = output_name(signature)?;
    let values = sample
        .buf(output)
        .ok_or(Error::new(Code::Internal, "签名输出缓冲与样本不匹配"))?;
    if !matches!(&values.values, Values::F32(_)) || values.values.is_empty() {
        return Err(Error::new(
            Code::Unsupported,
            "设备回读当前要求非空 f32 输出",
        ));
    }
    Ok((output.to_owned(), values.values.len()))
}

/// RMSNorm 分母稳定项：与 `.k` 模板、CUDA/Metal 折叠同口径（`sqrt(Σx²/N + ε)`）。
pub(crate) const RMS_EPS: f32 = 1e-6;

/// 分块循环的块头：块起点 `start`、块内长度 `len`（末块可能不足 `chunk`）。与 `.k` 模板、
/// CUDA/Metal 的逐字同口径，三端共用；`indent` 是每行缩进（设备侧 4 空格，宿主折叠更深）。
pub(crate) fn chunk_head(chunk: usize, len: i64, indent: &str) -> String {
    format!(
        "{indent}int start = g * {chunk};\n\
         {indent}int rest = {len} - start;\n\
         {indent}int len = ({chunk} < rest) ? {chunk} : rest;\n"
    )
}

/// 携带前缀折叠的 C 循环体（f64）：`C_0 = 0`、`C_{g+1} = a^len·C_g + y[start+len-1]`。
///
/// 与 `.k` 模板同口径；`a^len` 用平方乘（块长最多 `chunk` 次乘法），不引入 `math.h`/`-lm`。
/// `y` / `c` 是已绑定的 C 数组名（CUDA 传 `h_<out>`/`h_c`，Metal 传局部指针 `y`/`c`），
/// `blocks` 是块数的 C 表达式（CUDA 是变量 `G`，Metal 是代入的数字），`indent` 是循环体缩进。
pub(crate) fn carry_fold(
    y: &str,
    c: &str,
    blocks: &str,
    chunk: usize,
    len: i64,
    decay: f32,
    indent: &str,
) -> String {
    let inner = format!("{indent}    ");
    let mut s = format!("{indent}double carry = 0.0;\n");
    s.push_str(&format!("{indent}for (int g = 0; g < {blocks}; ++g) {{\n"));
    s.push_str(&chunk_head(chunk, len, &inner));
    s.push_str(&format!(
        "{inner}double p = 1.0, base = (double){decay}f;\n\
         {inner}int e = len;\n\
         {inner}while (e > 0) {{ if (e & 1) p *= base; base *= base; e >>= 1; }}\n\
         {inner}{c}[g] = (float)carry;\n\
         {inner}carry = p * carry + (double){y}[start + len - 1];\n"
    ));
    s.push_str(&format!("{indent}}}\n"));
    s
}

/// 校正段正文：`c = d_c[g]` 起，把携带态 `C_g·a^(j+1)` 按几何级数 `pow *= a` 加回
/// `d_{output}[start+j]`（就地累加）。CUDA 与 Metal 的校正内核共用同一份，`indent` 是每行缩进，
/// 末行（`for` 的 `}`）**不带**尾换行，由调用方补。
pub(crate) fn scan_correct_body(output: &str, decay: f32, indent: &str) -> String {
    format!(
        r#"{indent}float c = d_c[g];
{indent}float pow = {decay}f;
{indent}for (int j = 0; j < len; ++j) {{
{indent}    d_{output}[start + j] += c * pow;
{indent}    pow *= {decay}f;
{indent}}}"#
    )
}

/// 生产态实参：缓冲取 `d_{name} + n`（线程处理的元素）、标量取 `1`（每线程一个元素）。
pub(crate) fn grid_args(params: &[Param]) -> String {
    params
        .iter()
        .map(|param| {
            if param.ctype.is_buf() {
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

    fn buf_decl(param: &Param, slot: usize) -> String {
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

    fn buf_decl(param: &Param, _slot: usize) -> String {
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

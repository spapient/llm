//! 内核 body 改写：追线程下标形参、注入块头、把串行循环头绑到块内边界或线程下标。

use diagnostic::{Code, Error, Result};
use prof::decl::Grid;

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
/// Metal 传 `uint3 tid`、CUDA 传 `int3 tid`（由 `kernel_main` 从 `blockIdx` 装配），其余改写规则一致。
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

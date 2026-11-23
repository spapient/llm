//! CUDA host 驱动公共骨架：前导/收尾字面、MUSA 垫片、分配守卫、计时循环、缓冲声明与分配、
//! 单次派发公共装配（[`Driver`]）、发射与位级回读。各形态子模块只换派发那一行，不另拼第二份
//! host 流程。

use crate::dialect::{Cuda, readback_bits};
use cutile::harness::{c_literal, c_type, output_name};
use diagnostic::{Code, Error, Result};
use prof::model::{Sample, Values};
use prof::sig::{Param, Signature};

pub(crate) const CUDA_CHECK_PROLOGUE: &str = r#"
#define CUTILE_CUDA_CHECK(expr) do { \
    cudaError_t cutile_error = (expr); \
    if (cutile_error != cudaSuccess) { \
        fprintf(stderr, "cuda: %s\n", cudaGetErrorString(cutile_error)); \
        return 1; \
    } \
} while (0)
int main() {
"#;

/// 生产态驱动的收尾：事件对象销毁 + 正常返回。
pub(crate) const CUDA_EPILOGUE: &str = "    CUTILE_CUDA_CHECK(cudaEventDestroy(start));\n\
     \x20   CUTILE_CUDA_CHECK(cudaEventDestroy(stop));\n\
     \x20   return 0;\n}\n";

/// MUSA 兼容层：把 CUDA event / error API 映射到 MUSA 同名接口；非 MUSA 目标返回空串。
pub(crate) fn musa_shims(musa: bool) -> &'static str {
    if musa {
        "#define cudaEvent_t musaEvent_t\n\
         #define cudaEventCreate musaEventCreate\n\
         #define cudaEventRecord musaEventRecord\n\
         #define cudaEventSynchronize musaEventSynchronize\n\
         #define cudaEventElapsedTime musaEventElapsedTime\n\
         #define cudaEventDestroy musaEventDestroy\n\
         #define cudaGetLastError musaGetLastError\n"
    } else {
        ""
    }
}

/// 每个缓冲参数的分配失败检查：任何一次分配失败都必须立刻返回 1，不能让后续 launch 读到空指针。
pub(crate) fn alloc_guards(signature: &Signature) -> String {
    let mut out = String::new();
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
    {
        out.push_str(&format!(
            "    if (!d_{}) {{ fprintf(stderr, \"CUDA 缓冲分配失败: {}\\n\"); return 1; }}\n",
            param.name, param.name
        ));
    }
    out
}

/// 计时循环：先热身两轮，再按 `repeat` 轮计时，每轮固定 16 次 launch 后报单次纳秒。
pub(crate) fn timing_loop(
    grid: [usize; 3],
    threads: [usize; 3],
    launch_args: &str,
    repeat: usize,
) -> String {
    format!(
        r#"    const dim3 grid({}, {}, {});
    const dim3 block({}, {}, {});
    cudaEvent_t start, stop;
    CUTILE_CUDA_CHECK(cudaEventCreate(&start));
    CUTILE_CUDA_CHECK(cudaEventCreate(&stop));
    for (int warm = 0; warm < 2; ++warm) {{
        for (int batch = 0; batch < 16; ++batch) kernel_main<<<grid, block>>>({launch_args});
    }}
    CUTILE_CUDA_CHECK(cudaGetLastError());
    CUTILE_CUDA_CHECK(cudaDeviceSynchronize());
    for (int sample = 0; sample < {repeat}; ++sample) {{
        CUTILE_CUDA_CHECK(cudaEventRecord(start));
        for (int batch = 0; batch < 16; ++batch) kernel_main<<<grid, block>>>({launch_args});
        CUTILE_CUDA_CHECK(cudaGetLastError());
        CUTILE_CUDA_CHECK(cudaEventRecord(stop));
        CUTILE_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUTILE_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        if (!(elapsed_ms > 0.0f)) {{ fprintf(stderr, "CUDA event 时间戳不可用\n"); return 2; }}
        fprintf(stderr, "ns %.3f\n", (double)elapsed_ms * 1.0e6 / 16.0);
    }}
"#,
        grid[0], grid[1], grid[2], threads[0], threads[1], threads[2],
    )
}

pub(crate) struct Driver<'a> {
    signature: &'a Signature,
    sample: &'a Sample,
    output: &'a str,
    output_len: usize,
    fold: Option<&'a str>,
}

impl<'a> Driver<'a> {
    /// 由签名、样本与输出缓冲长度（分块归约取部分结果数，其余取样本长度）装配。
    pub(crate) fn new(
        signature: &'a Signature,
        sample: &'a Sample,
        output: &'a str,
        output_len: usize,
    ) -> Self {
        Self {
            signature,
            sample,
            output,
            output_len,
            fold: None,
        }
    }

    /// 主机折叠算子（只有分块归约用）：`max` 取大、其余求和。读到的是 `len` 个部分结果，折叠后仍
    /// 只回读一个值——输出个数与签名声明一致，折叠律与设备侧 `_cutile_<op>_f32` 同源。
    pub(crate) fn folding(mut self, fold: &'a str) -> Self {
        self.fold = Some(fold);
        self
    }

    /// 拼完整程序：`head` 是文件头（include + 身份注释 + 设备源，方言由调用方给），`emit` 是派发那一行。
    pub(crate) fn assemble(&self, head: &str, emit: &str) -> Result<String> {
        let mut text = String::from(head);
        text.push_str("\nint main(void) {\n");
        text.push_str(&cuda_host_bufs_grid(
            self.sample,
            self.signature.params(),
            self.output,
            self.output_len,
        )?);
        text.push_str(emit);
        let readback = cuda_readback(self.output, self.output_len, self.fold, false);
        text.push_str(&readback);
        text.push_str(&cuda_release(self.signature));
        text.push_str("    return 0;\n}\n");
        Ok(text)
    }
}

/// 文件头一行 include 段 + 设备源码；设备源码缺尾换行时补一个，免得 `main` 粘在最后一行上。
pub(crate) fn cuda_head(includes: &str, device_source: &str) -> String {
    let mut text = String::from(includes);
    text.push_str(device_source);
    if !device_source.ends_with('\n') {
        text.push('\n');
    }
    text
}

/// CUDA 前导：运行时与 stdio 两行 include + 设备源。
pub(crate) fn cuda_prologue(device_source: &str) -> String {
    cuda_head(
        "#include <cuda_runtime.h>\n#include <cstdio>\n",
        device_source,
    )
}

/// 单次派发的完整 host 程序：前导（include + 设备源）、`main`、缓冲、一次发射、回读、释放。
/// 逐点 / 网格 / 循环注入三种单次派发只在发射那一行上不同。
pub(crate) fn program(
    device_source: &str,
    signature: &Signature,
    sample: &Sample,
    output: &str,
    output_len: usize,
    emit: &str,
) -> Result<String> {
    let head = cuda_prologue(device_source);
    Driver::new(signature, sample, output, output_len).assemble(&head, emit)
}

/// 一次派发 + 同步失败即报错：内核名、网格、线程数与实参由调用方给，报错那一行只有这一份。
pub(crate) fn cuda_launch(kernel: &str, grid: &str, threads: usize, args: &str) -> String {
    format!(
        "    {kernel}<<<{grid}, {threads}>>>({args});\n\
         \x20   cudaError_t error = cudaDeviceSynchronize();\n\
         \x20   if (error != cudaSuccess) {{ fprintf(stderr, \"cuda: %s\\n\", cudaGetErrorString(error)); return 1; }}\n"
    )
}

/// 结果回读：设备 → 主机，再按 `float` 位模式逐元素打印（打印循环下沉到
/// [`readback_bits`]，三端共用同一份 `union` 位级保真代码）。
///
/// `fold` 给定时把 `len` 个部分结果在主机折成一个输出（`max` 取大、其余求和）；`checked` 决定
/// memcpy 是否套 `CUTILE_CUDA_CHECK`（基准驱动套，语义门 harness 不套）。
pub(crate) fn cuda_readback(output: &str, len: usize, fold: Option<&str>, checked: bool) -> String {
    let copy = if checked {
        format!(
            "    CUTILE_CUDA_CHECK(cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost));\n"
        )
    } else {
        format!(
            "    cudaMemcpy(h_{output}, d_{output}, {len} * sizeof(float), cudaMemcpyDeviceToHost);\n"
        )
    };
    let Some(fold) = fold else {
        return format!(
            "{copy}{}",
            readback_bits(&format!("h_{output}"), len, "    ")
        );
    };
    let (init, combine) = if fold == "max" {
        (
            "-3.402823466e38f",
            format!("if (h_{output}[i] > acc) acc = h_{output}[i];"),
        )
    } else {
        ("0.0f", format!("acc += h_{output}[i];"))
    };
    format!(
        "{copy}    float acc = {init};\n    for (int i = 0; i < {len}; ++i) {{\n        {combine}\n    }}\n    union {{ float f; unsigned int u; }} v; v.f = acc;\n    printf(\"%u\\n\", v.u);\n"
    )
}

/// 生产态分块归约程序：`blocks` 个线程各归约一块，回读部分结果后由主机折叠成一个输出。
///
/// # Errors：缺输出缓冲、输出不是 f32、样本缺缓冲时返回说明。

pub(crate) fn cuda_host_bufs_grid(
    sample: &Sample,
    params: &[Param],
    output: &str,
    output_len: usize,
) -> Result<String> {
    let mut text = String::new();
    for param in params.iter().filter(|param| param.ctype.is_buf()) {
        let Some(buf) = sample.buf(&param.name) else {
            return Err(Error::new(Code::Internal, "签名与样本已对齐（漂移）"));
        };
        text.push_str(&cuda_buf_host(param, &buf.values, output, output_len));
    }
    Ok(text)
}

/// C 系 host 侧缓冲声明：只读落静态初值、可写开数组。CUDA 与昇腾逐字同源（`h_` 主机数组 +
/// 设备侧 `d_` 指针），Metal 的 ObjC host 不用这套（见 [`crate::metal`]）。
pub(crate) fn host_buf_decl(param: &Param, values: &Values, ctype: &str, size: usize) -> String {
    if param.mutable {
        format!("    {ctype} h_{}[{size}];\n", param.name)
    } else {
        format!(
            "    static const {ctype} h_{}[{size}] = {{{}}};\n",
            param.name,
            c_literal(values)
        )
    }
}

/// 单个缓冲的 host 声明 + 设备分配 + （只读时）入向拷贝；`output` 与 `output_len` 只用于把输出
/// 缓冲的尺寸换成部分结果数，其余缓冲尺寸取样本原值。
pub(crate) fn cuda_buf_host(
    param: &Param,
    values: &Values,
    output: &str,
    output_len: usize,
) -> String {
    let ctype = c_type(values);
    let size = if param.name == output {
        output_len
    } else {
        values.len()
    };
    let mut text = host_buf_decl(param, values, ctype, size);
    text.push_str(&format!(
        "    {ctype} *d_{} = 0;\n    cudaMalloc((void **)&d_{}, {size} * sizeof({ctype}));\n",
        param.name, param.name
    ));
    if !param.mutable {
        text.push_str(&format!(
            "    cudaMemcpy(d_{}, h_{}, {size} * sizeof({ctype}), cudaMemcpyHostToDevice);\n",
            param.name, param.name
        ));
    }
    text
}

pub(crate) fn cuda_launch_params(signature: &Signature) -> String {
    let scalars = signature
        .params()
        .iter()
        .filter(|param| !param.ctype.is_buf())
        .map(|param| format!("{} {}", param.ctype.c_name(), param.name))
        .collect::<Vec<_>>()
        .join(", ");
    let bufs = Cuda::bufs(signature);
    if scalars.is_empty() {
        bufs
    } else {
        format!("{bufs},\n    {scalars}")
    }
}

/// 收尾：释放每个缓冲参数对应的设备端指针（只释放真分配过的，标量是值传递）。
pub(crate) fn cuda_release(signature: &Signature) -> String {
    let mut text = String::new();
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
    {
        text.push_str(&format!("    cudaFree(d_{});\n", param.name));
    }
    text
}

/// 启动实参：只传缓冲（形状是编译期常量，不进内核形参）。
pub(crate) fn buf_args(signature: &Signature) -> String {
    signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buf())
        .map(|param| format!("d_{}", param.name))
        .collect::<Vec<_>>()
        .join(", ")
}

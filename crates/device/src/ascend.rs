//! 昇腾侧的设备源与主机驱动拼装：`__aicore__` 单源翻译单元、`<<<blocks, nullptr, stream>>>`
//! 发射、`aclrtMalloc` / `aclrtMemcpy` / `aclrtFree` 骨架与回读。

use crate::cuda::host_buffer_decl;
use crate::dialect::readback_bits;
use cutile::harness::{c_type, output_of};
use diagnostic::{Code, Error, Result};
use op::model::{Sample, Values};
use op::sig::{Param, Signature};

/// AscendCL 的调用守卫：失败时报出是哪个调用并返回非零。
pub(crate) const ACL_GUARD: &str = "static int cutile_acl(const char *what, aclError code) {\n\
     \x20   if (code == ACL_SUCCESS) return 0;\n\
     \x20   fprintf(stderr, \"acl %s: %d\\n\", what, (int)code);\n\
     \x20   return -1;\n\
     }\n\
     #define CUTILE_ACL(call) do { if (cutile_acl(#call, (call)) != 0) return 1; } while (0)\n";

/// 拼出可在 NPU 上真实执行的 AscendC 程序：`.k` 内核转 `__aicore__`，单块 `<<<1, nullptr, stream>>>`
/// 包装里调用，host `main` 负责分配 / 拷贝 / 回读——**语义对拍门**，性能另由 roofline 门承载。
///
/// 单块（`block = 1`）与 CUDA 的 `<<<1,1>>>` 是同一件事：这一档只回答"算出来的数对不对"，
/// 不回答"跑得多快"；并行形态要等 P3 的 `DataCopy` / `Mmad` 特化。
///
/// # Errors
///
/// 缺输出缓冲、输出不是 f32、样本缺缓冲，或签名与样本不一致时返回说明。
pub fn ascend_harness(
    kernel_source: &str,
    signature: &Signature,
    sample: &Sample,
) -> Result<String> {
    let (output, values) = output_of(signature, sample)?;
    let kernel = signature.kernel();
    let mut text = format!(
        "#include \"acl/acl.h\"\n\
         #include \"kernel_operator.h\"\n\
         #include <cstdio>\n\
         #include <cstdlib>\n\
         /* cutile ascend harness | kernel={kernel} m={} n={} | 单块设备执行（语义门，非并行性能门） */\n",
        sample.m, sample.n
    );
    let device = ascend_device(kernel_source, &kernel);
    text.push_str(&device);
    if !device.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&ascend_wrapper(&kernel, signature, sample)?);
    text.push_str(&ascend_main(signature, sample, output, values.len())?);
    Ok(text)
}

/// 把模板翻译单元改成设备源：内核函数补 `__aicore__`，`[compile].preamble` 里的软件实现
/// （`_cutile_sum_f32` 一族）同样要带——它们是 `@reduce` 的 `$func` 与几个软件数学函数的落点，
/// 不加就是"AI Core 函数调用了宿主函数"，编译期会被拒。
pub(crate) fn ascend_device(text: &str, kernel: &str) -> String {
    text.replace(
        &format!("void {kernel}("),
        &format!("__aicore__ void {kernel}("),
    )
    .replace("static inline ", "static __aicore__ inline ")
}

/// 设备侧入口：`extern "C" __global__ __aicore__` 的核函数，形参用 `GM_ADDR` 收显存指针。
fn ascend_wrapper(kernel: &str, signature: &Signature, sample: &Sample) -> Result<String> {
    let params = signature
        .params()
        .iter()
        .map(|param| {
            if param.ctype.is_buffer() {
                format!("GM_ADDR d_{}", param.name)
            } else {
                format!("{} {}", param.ctype.c_name(), param.name)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let call = signature
        .params()
        .iter()
        .map(|param| ascend_argument(param, sample))
        .collect::<Result<Vec<_>>>()?
        .join(", ");
    Ok(format!(
        "\nextern \"C\" __global__ __aicore__ void cutile_launch({params}) {{\n    {kernel}({call});\n}}\n"
    ))
}

/// 内核实参：缓冲传补好元素类型的 `__gm__` 指针，标量取样本值（与 CUDA 的 `launch_args` 同序同值）。
fn ascend_argument(param: &Param, sample: &Sample) -> Result<String> {
    if param.ctype.is_buffer() {
        let base = param.ctype.c_name();
        return Ok(if param.mutable {
            format!("(__gm__ {base} *)d_{}", param.name)
        } else {
            format!("(__gm__ const {base} *)d_{}", param.name)
        });
    }
    let value = sample.scalar(&param.name).ok_or_else(|| {
        Error::new(
            Code::Internal,
            format!("标量实参 `{}` 没有取值", param.name),
        )
    })?;
    Ok(value.to_string())
}

/// host 驱动：AscendCL 分配 GM、单块发射、回读、释放。
fn ascend_main(
    signature: &Signature,
    sample: &Sample,
    output: &str,
    output_len: usize,
) -> Result<String> {
    let mut text = String::from(ACL_GUARD);
    text.push_str(
        "\nint main(void) {\n\
         \x20   CUTILE_ACL(aclInit(nullptr));\n\
         \x20   CUTILE_ACL(aclrtSetDevice(0));\n\
         \x20   aclrtStream stream = nullptr;\n\
         \x20   CUTILE_ACL(aclrtCreateStream(&stream));\n",
    );
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
    {
        text.push_str(&ascend_buffer(param, sample, output, output_len)?);
    }
    text.push_str(&format!(
        "    cutile_launch<<<1, nullptr, stream>>>({});\n\
         \x20   CUTILE_ACL(aclrtSynchronizeStream(stream));\n\
         \x20   CUTILE_ACL(aclrtMemcpy(h_{output}, {output_len} * sizeof(float), d_{output}, \
         {output_len} * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));\n{}",
        ascend_launch_args(signature, sample)?,
        readback_bits(&format!("h_{output}"), output_len, "    ")
    ));
    for param in signature
        .params()
        .iter()
        .filter(|param| param.ctype.is_buffer())
    {
        text.push_str(&format!("    CUTILE_ACL(aclrtFree(d_{}));\n", param.name));
    }
    text.push_str(
        "    CUTILE_ACL(aclrtDestroyStream(stream));\n\
         \x20   CUTILE_ACL(aclrtResetDevice(0));\n\
         \x20   CUTILE_ACL(aclFinalize());\n\
         \x20   return 0;\n\
         }\n",
    );
    Ok(text)
}

/// 单个缓冲的 host 声明 + GM 分配 + （只读时）入向拷贝；输出缓冲的长度取 `output_len`
/// （与 CUDA 同口径：分块归约的产物长度可以大于签名的单元素 `y`）。
fn ascend_buffer(
    param: &Param,
    sample: &Sample,
    output: &str,
    output_len: usize,
) -> Result<String> {
    let Some(buffer) = sample.buffer(&param.name) else {
        return Err(Error::new(Code::Internal, "签名与样本已对齐（漂移）"));
    };
    let values: &Values = &buffer.values;
    let ctype = c_type(values);
    let size = if param.name == output {
        output_len
    } else {
        values.len()
    };
    let mut text = host_buffer_decl(param, values, ctype, size);
    text.push_str(&format!(
        "    {ctype} *d_{} = nullptr;\n\
         \x20   CUTILE_ACL(aclrtMalloc((void **)&d_{}, {size} * sizeof({ctype}), \
         ACL_MEM_MALLOC_HUGE_FIRST));\n",
        param.name, param.name
    ));
    if !param.mutable {
        text.push_str(&format!(
            "    CUTILE_ACL(aclrtMemcpy(d_{}, {size} * sizeof({ctype}), h_{}, \
             {size} * sizeof({ctype}), ACL_MEMCPY_HOST_TO_DEVICE));\n",
            param.name, param.name
        ));
    }
    Ok(text)
}

/// 派发实参：缓冲传 `(GM_ADDR)d_<name>`，标量取样本值——与设备入口的形参表同序。
fn ascend_launch_args(signature: &Signature, sample: &Sample) -> Result<String> {
    signature
        .params()
        .iter()
        .map(|param| {
            if param.ctype.is_buffer() {
                Ok(format!("(GM_ADDR)d_{}", param.name))
            } else {
                ascend_argument(param, sample)
            }
        })
        .collect::<Result<Vec<_>>>()
        .map(|args| args.join(", "))
}

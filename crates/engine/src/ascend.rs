//! 昇腾侧的设备源与主机驱动拼装：`__aicore__` 单源翻译单元、`<<<blocks, nullptr, stream>>>`
//! 发射、`aclrtMalloc` / `aclrtMemcpy` / `aclrtFree` 骨架与回读。
//!
//! 与 [`crate::cuda`] 是同一层的一件事：那里把 `.k` 循环体换成 `__device__` 单线程内核、
//! 用 `cudaMalloc` 驱动；这里换成 `__aicore__` 内核、用 AscendCL 驱动。只换方言，不改语义：
//! 回读口径（`float` 位模式逐行打印）必须与 CUDA 逐字一致，否则共用同一份
//! [`cutile::runner::parse_bits`] 的两台设备会读出两种东西。
//!
//! 这是 `L2`（与参考对拍）在昇腾上的**入口**：见证程序只回答"片段在设备上跑得起来"，
//! `verify` 的形状矩阵走的是这里——内核 + 单块发射 + 回读。等级仍是 `L0`：这份文本还没在
//! 任何 910B 上编过，本节只是把那条证据链的最后一段接上。

use cutile::harness::{c_literal, c_type, output_of};
use diagnostic::{Code, Error, Result};
use op::model::{Sample, Values};
use op::sig::{Param, Signature};

use crate::dialect::readback_bits;

/// AscendCL 的调用守卫：失败时报出是哪个调用并返回非零。
///
/// 见证程序（`backends` 的 `push_ascend_main`）与本模块的语义门 harness 共用这一份：两处各写
/// 一套时，加一个 API 调用就会漏掉其中一处，而漏掉的那处失败会以"回读全是 0"的形式出现。
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
///
/// 只改**函数**，不改变量：命名空间里的 `static const float` 是编译期可折叠的常量，给它加
/// `__aicore__`（那是函数属性）只会把本来能编的源码改坏。与 CUDA 的
/// [`crate::dialect::Dialect::device_fn`] 同一套写法，只是属性换名。
pub(crate) fn ascend_device(text: &str, kernel: &str) -> String {
    text.replace(
        &format!("void {kernel}("),
        &format!("__aicore__ void {kernel}("),
    )
    .replace("static inline ", "static __aicore__ inline ")
}

/// 设备侧入口：`extern "C" __global__ __aicore__` 的核函数，形参用 `GM_ADDR` 收显存指针。
///
/// 形参声明成 `GM_ADDR` 是为了让 `<<<>>>` 的实参形状与 CUDA 的 `void *` 同构；交给 P0 内核前
/// 要补回元素类型的 `__gm__` 指针（`GM_ADDR` 只是 `__gm__ uint8_t *` 的别名，内核签名要的是
/// `__gm__ const float *` 这类有类型的指针）。
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
///
/// 次序与 CUDA 的 `Driver::assemble` 一一对应（分配 → 入向拷贝 → 一次派发 → 回读 → 释放），
/// 差别只在 API 名字：`aclrtMalloc` / `aclrtMemcpy` / `aclrtSynchronizeStream`。
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
    let mut text = if param.mutable {
        format!("    {ctype} h_{}[{size}];\n", param.name)
    } else {
        format!(
            "    static const {ctype} h_{}[{size}] = {{{}}};\n",
            param.name,
            c_literal(values)
        )
    };
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

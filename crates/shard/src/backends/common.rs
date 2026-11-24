//! 三条厂商链共用的几何与见证回读：输出长度、网格维校验、正整数标量、位模式回读字面。
//! 这些原语与厂商无关，单列一份，避免 CUDA / 昇腾 / Metal 各抄一份漂移。

use crate::driver;
use diagnostic::{Code, Error, Result};
use prof::model::Sample;
use prof::prof::Profile;
use prof::sig::Signature;

/// 基准输出缓冲长度：取样本里输出缓冲的元素数（分块归约的部分结果数由调用方另给）。
pub(crate) fn benchmark_output_len(signature: &Signature, sample: &Sample) -> Result<usize> {
    let output = cutile::harness::output_name(signature)?;
    sample
        .buf(output)
        .map(|buf| buf.values.len())
        .ok_or_else(|| Error::new(Code::Internal, "签名输出缓冲与样本不匹配"))
}

/// 网格各维从样本取值并校验在一到三维内；返回补到三维的尺寸。
pub(crate) fn benchmark_grid(grid: &prof::decl::Grid, sample: &Sample) -> Result<[usize; 3]> {
    if grid.dims.is_empty() || grid.dims.len() > 3 {
        return Err(Error::new(
            Code::Unsupported,
            "设备基准要求一到三个明确的网格维度",
        ));
    }
    let mut dims = [1usize; 3];
    for (idx, (_, scalar)) in grid.dims.iter().enumerate() {
        dims[idx] = grid_extent(scalar, sample)?;
    }
    Ok(dims)
}

/// 一个必须为正的整数标量：网格维与输出宽度共用这一判据，缺取值、负数与零全在这一处报。
pub(crate) fn positive_scalar(sample: &Sample, scalar: &str, what: &str) -> Result<usize> {
    let value = sample
        .scalar(scalar)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{what} `{scalar}` 没有样本取值")))?;
    usize::try_from(value)
        .ok()
        .filter(|extent| *extent > 0)
        .ok_or_else(|| Error::new(Code::Invalid, format!("{what} `{scalar}` 必须为正数")))
}

/// 一个网格维的长度：逐点网格与分块网格选型共用。
pub(crate) fn grid_extent(scalar: &str, sample: &Sample) -> Result<usize> {
    positive_scalar(sample, scalar, "网格标量")
}

/// CUDA 与昇腾的 host 驱动共用这一段字面：同一个 `driver::readback` 在两种产物上必须读出同一种
/// 文本，各写一份时改一处就会漂。
pub(crate) fn witness_readback(max: usize) -> String {
    format!(
        "\x20   for (int t = 0; t < total; ++t) {{\n\
         \x20       for (int i = 0; i < h_counts[t]; ++i) {{\n\
         \x20           union {{ float f; unsigned int u; }} v;\n\
         \x20           v.f = h_out[t * {max} + i];\n\
         \x20           printf(\"%u\\n\", v.u);\n\
         \x20       }}\n\
         \x20       fprintf(stderr, \"witness %d\\n\", h_counts[t]);\n\
         \x20   }}\n"
    )
}

/// 见证程序头：包含行 + 档案 preamble（非空才追加，与三家编译器的 `-include` 同位）。
pub(crate) fn witness_head(includes: &str, profile: &Profile) -> String {
    let mut head = String::from(includes);
    if !profile.preamble.is_empty() {
        head.push_str(&profile.preamble);
        head.push('\n');
    }
    head
}

/// 见证函数加设备函数前缀（CUDA `__device__` / AscendC `__aicore__`）：不加这一层，函数只有宿主
/// 链接性，核函数里调用它会被当成未声明的设备符号；见证本身是纯标量 C，不需要别的改写。
pub(crate) fn device_fn(files: driver::Sources, prefix: &str) -> driver::Sources {
    files
        .into_iter()
        .map(|(file, text)| {
            (
                file,
                text.replace("int witness", &format!("{prefix} int witness")),
            )
        })
        .collect()
}

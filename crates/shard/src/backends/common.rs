//! 三条厂商链共用的见证装配与回读字面。
//! 这些原语与厂商无关，单列一份，避免 CUDA / 昇腾 / Metal 各抄一份漂移。

use crate::driver;
use prof::prof::Profile;

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

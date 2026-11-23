//! 设备源文本装配：回读片段、模板填空、分块头、携带折叠与校正段正文。纯字符串生成，不带签名/样本。

/// RMSNorm 分母稳定项：与 `.k` 模板、CUDA/Metal 折叠同口径（`sqrt(Σx²/N + ε)`）。
pub(crate) const RMS_EPS: f32 = 1e-6;

/// 回读的浮点位模式打印循环：`union` 保持 float→uint 位级保真，`src[i]` 是第 i 个元素的取值
/// 表达式（CUDA/昇腾是 `h_<name>[i]`，Metal 是 `out[i]`）。三端共用：回读口径必须逐字一致，
/// 否则 `llm_optim::runner::parse_bits` 会读出不同的位模式。
pub(crate) fn readback_bits(src: &str, len: usize, indent: &str) -> String {
    format!(
        "{indent}for (int i = 0; i < {len}; ++i) {{\n\
         {indent}    union {{ float f; unsigned int u; }} v; v.f = {src}[i];\n\
         {indent}    printf(\"%u\\n\", v.u);\n\
         }}\n"
    )
}

/// 模板填空：`@name` 逐个替换成对应文本。设备源以「原始文本 + `@占位`」的形式保存成数据，装配处只
/// 做一次填空——`format!` 的花括号转义不进模板，模板可逐字与厂商语法对照。全 crate 只有这一处。
///
/// 单趟扫描：占位名到第一个非标识符字符为止（`@a@b`、`@a,`、`@a ` 都是合法写法），已插入的文本
/// 不会被二次替换，未知占位原样输出——Objective-C 的 `@"..."` 因此安然通过，写错的名字也不会被
/// 静默吞掉。渲染结果里的裸 `@` 另有 host 程序的专项用例兜住。
pub(crate) fn fill(template: &str, pairs: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(at) = rest.find('@') {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 1..];
        // 占位名到第一个非标识符字符为止：`@sets ` / `@bytes[` / `@a,` 都是同一个名字的词法边界。
        let end = tail
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(tail.len());
        let (name, after) = tail.split_at(end);
        match pairs.iter().find(|(key, _)| *key == name) {
            Some((_, value)) => out.push_str(value),
            // 未知占位（含 Objective-C 的 `@"..."`）原样输出：不静默吞掉调用方写错的名字。
            None => {
                out.push('@');
                out.push_str(name);
            }
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

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

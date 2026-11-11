//! ⑤ 编译与执行：`c` 后端通过子进程调用 `cc`，不涉及 FFI，也不需要 `unsafe`。
//!
//! 只有"能跑出数"才算 L2 证据（DESIGN §11.2）：这里负责把生成的源文件编成可执行文件、
//! 跑一遍、把 f32 结果的位模式读回来交给 [`crate::reference`] 对拍。

use std::path::Path;
use std::process::Command;

/// 一次执行的产物。
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// 内核输出（按位模式解码后的 f32）。
    pub bits: Vec<f32>,
    /// 内核耗时（纳秒，来自 C 侧 `clock()`）。
    pub ns: Option<f64>,
    /// 编译器/运行期的 stderr。
    pub stderr: String,
}

/// GEMV 的形状与数据，供 harness 生成静态数组。
#[derive(Debug, Clone, PartialEq)]
pub struct GemvData {
    /// 规约维度。
    pub m: usize,
    /// 输出行数。
    pub n: usize,
    /// 量化分组。
    pub group: usize,
    /// 输入向量。
    pub x: Vec<f32>,
    /// 打包的 int4 权重。
    pub w: Vec<u8>,
    /// 分组缩放。
    pub scales: Vec<f32>,
}

/// 生成带数据的 `main()`：跑 `repeat` 次、打印结果位模式与耗时。
pub fn harness(kernel_source: &str, kernel: &str, data: &GemvData, repeat: usize) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "/* cutile harness | kernel={kernel} m={} n={} group={} repeat={repeat} */\n",
        data.m, data.n, data.group
    ));
    out.push_str(kernel_source);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("\n#include <stdio.h>\n#include <time.h>\n\n");

    out.push_str(&format!("static const float X[{}] = {{", data.x.len()));
    for (index, value) in data.x.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{value:?}f"));
    }
    out.push_str("};\n\n");

    out.push_str(&format!(
        "static const unsigned char W[{}] = {{",
        data.w.len()
    ));
    for (index, value) in data.w.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("0x{value:02x}u"));
    }
    out.push_str("};\n\n");

    out.push_str(&format!("static const float S[{}] = {{", data.scales.len()));
    for (index, value) in data.scales.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{value:?}f"));
    }
    out.push_str("};\n\n");

    out.push_str(&format!(
        "int main(void) {{
    float y[{n}];
    for (int i = 0; i < {n}; ++i) {{ y[i] = 0.0f; }}
    clock_t t0 = clock();
    for (int r = 0; r < {repeat}; ++r) {{
        {kernel}(X, W, S, y, {m}, {n}, {group});
    }}
    clock_t t1 = clock();
    for (int i = 0; i < {n}; ++i) {{
        union {{ float f; unsigned int u; }} v;
        v.f = y[i];
        printf(\"%u\\n\", v.u);
    }}
    fprintf(stderr, \"ns %.0f\\n\", 1e9 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC);
    return 0;
}}
",
        n = data.n,
        m = data.m,
        group = data.group,
        repeat = repeat
    ));
    out
}

/// 探测一个可用的 C 编译器（`cc` → `clang` → `gcc`）。
pub fn detect_toolchain() -> Option<String> {
    for candidate in ["cc", "clang", "gcc"] {
        let ok = Command::new(candidate)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        if ok {
            return Some(candidate.to_string());
        }
    }
    None
}

/// 编译并运行；`workdir` 由调用方给（通常是缓存目录）。
pub fn compile_and_run(tool: &str, source: &str, workdir: &Path) -> Result<Outcome, String> {
    std::fs::create_dir_all(workdir)
        .map_err(|error| format!("创建 {} 失败：{error}", workdir.display()))?;
    let src = workdir.join("kernel.c");
    let exe = workdir.join("kernel");
    std::fs::write(&src, source).map_err(|error| format!("写 {} 失败：{error}", src.display()))?;

    let compile = Command::new(tool)
        .arg("-O2")
        .arg("-std=c99")
        .arg("-o")
        .arg(&exe)
        .arg(&src)
        .output()
        .map_err(|error| format!("调用 `{tool}` 失败：{error}"))?;
    if !compile.status.success() {
        return Err(format!(
            "`{tool}` 编译失败：\n{}\n{}\n源文件：{}",
            String::from_utf8_lossy(&compile.stdout),
            String::from_utf8_lossy(&compile.stderr),
            src.display()
        ));
    }

    let run = Command::new(&exe)
        .output()
        .map_err(|error| format!("运行 {} 失败：{error}", exe.display()))?;
    let stderr = String::from_utf8_lossy(&run.stderr).to_string();
    if !run.status.success() {
        return Err(format!(
            "内核运行失败（退出码 {:?}）：{stderr}",
            run.status.code()
        ));
    }
    Ok(Outcome {
        bits: parse_bits(&String::from_utf8_lossy(&run.stdout)),
        ns: parse_ns(&stderr),
        stderr,
    })
}

/// 把 `printf("%u\n", …)` 的输出解码回 f32。
pub fn parse_bits(stdout: &str) -> Vec<f32> {
    stdout
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok().map(f32::from_bits))
        .collect()
}

fn parse_ns(stderr: &str) -> Option<f64> {
    stderr.lines().find_map(|line| {
        line.trim()
            .strip_prefix("ns ")
            .and_then(|value| value.trim().parse::<f64>().ok())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> GemvData {
        GemvData {
            m: 4,
            n: 2,
            group: 4,
            x: vec![1.0, 1.0, 1.0, 1.0],
            w: vec![0x98, 0x76],
            scales: vec![0.5, 0.5],
        }
    }

    #[test]
    fn harness_embeds_data_and_calls_the_kernel() {
        let source = harness("void cutile_x(void) {}\n", "cutile_x", &data(), 3);
        assert!(source.contains("static const float X[4]"));
        assert!(source.contains("static const unsigned char W[2]"));
        assert!(source.contains("static const float S[2]"));
        assert!(source.contains("cutile_x(X, W, S, y, 4, 2, 4);"));
        assert!(source.contains("for (int r = 0; r < 3; ++r)"));
        assert!(source.contains("ns %.0f"));
    }

    #[test]
    fn floats_round_trip_through_the_literal_format() {
        let value = 0.1f32;
        let literal = format!("{value:?}");
        let parsed: f64 = literal.parse().expect("可解析");
        assert_eq!(parsed as f32, value);
    }

    #[test]
    fn bit_patterns_decode_back_to_floats() {
        let bits = f32::to_bits(1.5);
        assert_eq!(parse_bits(&format!("{bits}\n")), vec![1.5]);
    }

    #[test]
    fn missing_compiler_is_not_an_error_by_itself() {
        // 只是探测：返回 None 也算合法结果，调用方负责报"缺设备即失败"。
        let _ = detect_toolchain();
    }
}

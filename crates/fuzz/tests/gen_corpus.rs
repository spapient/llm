//! 种子语料生成器：把库编码的合法范例写到 `<out>/<target>/seed.bin`。
//!
//! 这是 `harness = false` 的独立二进制，不走 libtest：CI 在 fuzz job 里先跑它生成语料，再把
//! 语料目录挂给 libFuzzer。范例与 sweep 遍历共用 [`seeds`]，二进制格式不经手抄。被 test runner
//! 无参拉起时不落盘，避免普通 `cargo test` 产生源码树副作用。

#[path = "support/seeds.rs"]
mod seeds;

use std::fs;
use std::path::PathBuf;

fn main() {
    let Some(out) = std::env::args().nth(1) else {
        return;
    };
    for (target, bytes) in seeds::exemplars() {
        let dir = PathBuf::from(&out).join(target);
        fs::create_dir_all(&dir).expect("语料目录可创建");
        fs::write(dir.join("seed.bin"), bytes).expect("种子可写入");
        for (name, edge) in seeds::edges() {
            fs::write(dir.join(format!("{name}.bin")), edge).expect("边界样本可写入");
        }
    }
}

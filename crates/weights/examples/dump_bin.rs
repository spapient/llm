//! 对拍辅助：解析 .bin，把每个张量的元数据与原始字节导出到指定目录。
//! 用法：`cargo run -p llm-weights --example dump_bin -- <input.bin> <out_dir>`。
//! 每个张量写出 `<out_dir>/<name>.bin`（原始字节）与 `<out_dir>/meta.json`。

use std::fs;
use std::path::Path;

use llm_weights::bin::Bin;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).ok_or("缺输入路径")?;
    let out_dir = args.get(2).ok_or("缺输出目录")?;

    let raw = fs::read(input)?;
    let bin = Bin::parse(&raw)?;
    let out = Path::new(out_dir);
    fs::create_dir_all(out)?;

    let mut meta = String::from("{");
    for (i, tensor) in bin.tensors().iter().enumerate() {
        let safe = tensor.name().replace('.', "_");
        let path = out.join(format!("{safe}.bin"));
        fs::write(&path, bin.data(tensor)?)?;
        if i > 0 {
            meta.push(',');
        }
        let shape: Vec<String> = tensor.shape().iter().map(u64::to_string).collect();
        meta.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{:?}\",\"shape\":[{}]}}",
            tensor.name(),
            tensor.dtype(),
            shape.join(",")
        ));
    }
    meta.push('}');
    fs::write(out.join("meta.json"), meta)?;
    println!("dumped {} tensors", bin.tensors().len());
    Ok(())
}

//! A3000 实测指纹到真实 execution 流水线的远程验证入口。

use cutile::builtin::Catalogue;
use cutile::dirs::Dirs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    llm_shard::backends::install();
    let root = std::env::var("A3000_WORK")
        .unwrap_or_else(|_| r"C:\Users\fei\a3000-auto-verify-work".to_string());
    let dirs = Dirs::under(root);
    let catalogue = Catalogue::load(&dirs)?;
    let profile = catalogue.resolve("auto")?;
    println!("selected={}({})", profile.meta.vendor, profile.meta.id);
    println!("flags={}", profile.flags.join(" "));

    let cases = [
        (1, 1, 0),
        (3, 5, 0),
        (1, 1, 1),
        (1, 17, 3),
        (17, 1, 19),
        (15, 17, 31),
        (16, 16, 16),
        (17, 17, 17),
        (31, 33, 35),
        (64, 65, 63),
        (65, 64, 129),
        (128, 3, 65),
        (2, 4096, 1),
        (256, 256, 4096),
    ];
    for (m, n, k) in cases {
        let sample = nn::kernel::matmul::sample_k(m, n, k);
        let pass = llm_shard::execution::execute(
            &catalogue,
            &dirs,
            "matmul_f32",
            &profile.meta.vendor,
            &sample,
        )?;
        println!(
            "case={m}x{n}x{k} PASS variant={} tile={} max_rel_err={:.6e} outputs={}",
            pass.variant,
            pass.tile,
            pass.max_rel_err,
            pass.output.len()
        );
    }
    Ok(())
}

//! A3000 真实 pipeline 的给定样本性能验证入口。

use cutile::builtin::Catalogue;
use cutile::dirs::Dirs;
use cutile::prof::Detected;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    llm_shard::backends::install();
    let root = std::env::var("A3000_WORK")
        .unwrap_or_else(|_| r"C:\Users\fei\a3000-real-benchmark-work".to_string());
    let dirs = Dirs::under(root);
    let catalogue = Catalogue::load(&dirs)?;
    let detected = Detected::new()
        .with("cc", "8.6")
        .with("name", "NVIDIA RTX A3000 12GB Laptop GPU");
    let profile = catalogue.select(&detected)?;
    println!("selected={}({}) flags={}", profile.meta.vendor, profile.meta.id, profile.flags.join(" "));

    for (m, n, k, repeat) in [(256, 256, 4096, 50), (1024, 1024, 1024, 20), (4096, 256, 4096, 10)] {
        let sample = nn::kernel::matmul::sample_k(m, n, k);
        let timed = llm_shard::pipeline::execute_with_repeat(
            &catalogue,
            &dirs,
            "matmul_f32",
            &profile.meta.vendor,
            &sample,
            repeat,
        )?;
        let summary = timed.timing.summary().ok_or("真实执行没有计时样本")?;
        let tflops = 2.0 * m as f64 * n as f64 * k as f64 / (summary.p50_ns * 1e3);
        println!(
            "case={m}x{n}x{k} repeat={} samples={} max_rel_err={:.6e} min_ns={:.3} p50_ns={:.3} p95_ns={:.3} tflops={:.3}",
            repeat,
            timed.timing.samples_ns.len(),
            timed.execution.max_rel_err,
            summary.min_ns,
            summary.p50_ns,
            summary.p95_ns,
            tflops,
        );
    }
    Ok(())
}

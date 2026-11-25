use diagnostic::Result;
use engine::exec::{CpuExecutor, Device, Launch, Node, Program, View, ViewMut};
use std::hint::black_box;
use std::time::Instant;

struct Noop(&'static str);

impl Program for Noop {
    fn name(&self) -> &str {
        self.0
    }

    fn run(&self, _: Launch, _: &[View<'_>], _: &mut [ViewMut<'_>]) -> Result<()> {
        Ok(())
    }
}

const NAMES: [&str; 27] = [
    "add_f32", "attention_f32", "decode_attention_f32", "decode_flash_attention_f32",
    "dequant_i2", "dequant_i4", "dequant_i8", "embed_f32", "flash_attention_f32", "gelu_f32",
    "gemm_i2", "gemm_i4", "gemm_i8", "gemv_f32", "gemv_i2", "gemv_i4", "gemv_i8", "mul_f32",
    "reduce_max_f32", "reduce_sum_f32", "residual_add_f32", "rmsnorm_f32", "rope_f32", "scale",
    "silu_f32", "softmax_f32", "gemm_f32",
];

fn main() {
    let mut executor = CpuExecutor::new(Device::new("cpu", "generic"));
    for name in NAMES.iter().rev() {
        executor.register(Box::new(Noop(name))).unwrap();
    }
    let nodes: Vec<Node> = NAMES
        .iter()
        .map(|name| Node::new(*name, Launch::new([1, 1, 1], [1, 1, 1])))
        .collect();
    for node in nodes.iter().cycle().take(10_000) {
        executor.run(node).unwrap();
    }
    let mut samples = Vec::new();
    for _ in 0..11 {
        let start = Instant::now();
        for node in nodes.iter().cycle().take(2_700_000) {
            executor.run(black_box(node)).unwrap();
        }
        samples.push(start.elapsed().as_nanos() as f64 / 2_700_000.0);
    }
    samples.sort_by(f64::total_cmp);
    println!("median_ns={:.3} min_ns={:.3} max_ns={:.3}", samples[5], samples[0], samples[10]);
    println!("samples_ns={samples:?}");
}

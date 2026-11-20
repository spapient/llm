//! 走本仓公开流水线生成 MUSA 可编译的 CUDA 程序 + 标量参考。
//! 逐点: cuda_grid_program + cuda_grid_driver;归一化: cuda_rmsnorm/softmax_program + engine::norm 驱动。
//! 用法: gen-cuda-win <op> [m] [n]
fn main() -> diagnostic::Result<()> {
    let mut args = std::env::args().skip(1);
    let name = args.next().expect("用法: <op> [m] [n]");
    let m: usize = args.next().map_or(64, |v| v.parse().expect("m"));
    let n: usize = args.next().map_or(1, |v| v.parse().expect("n"));

    engine::backends::install();
    let catalogue = cutile::builtin::Catalogue::try_builtin()?;
    let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join(format!("llm-gen-{name}")));
    let decl = catalogue.op(&name).expect("算子声明");
    let model = op::model(&name).expect("语义表");
    let sample = (model.sample)(m, n);
    let output = decl.signature.output().expect("可写输出");
    let out_len = sample.f32(output).expect("f32 输出").len();

    let program = match name.as_str() {
        "rmsnorm_f32" => {
            let (partial, norm) =
                cutile::execution::cuda_rmsnorm_program(&catalogue, &dirs, &name, m, n)?;
            let chunk = decl.chunk.as_ref().expect("[chunk] 声明").size;
            engine::norm::cuda_rmsnorm_driver(&[partial, norm], &decl.signature, &sample, chunk)?
        }
        "softmax_f32" => {
            let (max, expsum, norm) =
                cutile::execution::cuda_softmax_program(&catalogue, &dirs, &name, m, n)?;
            let chunk = decl.chunk.as_ref().expect("[chunk] 声明").size;
            engine::norm::cuda_softmax_driver(&[max, expsum, norm], &decl.signature, &sample, chunk)?
        }
        _ => {
            let device = cutile::execution::cuda_grid_program(&catalogue, &dirs, &name, m, n)?;
            engine::codegen::cuda_grid_driver(
                &device, &decl.signature, &sample, out_len, out_len,
            )?
        }
    };
    std::fs::write("C:/Users/fei/fw.cu", &program)
        .map_err(|e| diagnostic::Error::new(diagnostic::Code::Io, format!("{e}")))?;

    let want = (model.reference)(&sample)?;
    let mut refs = String::new();
    for v in &want {
        refs.push_str(&format!("{v:.9e}\n"));
    }
    std::fs::write("C:/Users/fei/fw_ref.txt", refs)
        .map_err(|e| diagnostic::Error::new(diagnostic::Code::Io, format!("{e}")))?;
    println!("ok out_len={out_len} refs={}", want.len());
    Ok(())
}

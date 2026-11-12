//! 组装点：把各层连起来，**不定义任何新类型**（`docs/ARCHITECTURE.md` 的 L7）。
//!
//! 现在它只做一件事：构造每一层的最小实例并打印一份摘要。这不是演示，而是"依赖图真的能编译、
//! 各层公共 API 真的可用"的端到端证据——依赖方向只是评审约定，能不能跑只能靠这里真正组装。
//!
//! 真正的子命令（`run` / `serve` / `train`）在 P1 接入；在那之前，任何新类型都应该长在
//! 它所属的层里，而不是长在这里。

use diagnostic::Result;
use nn::Layer as _;
use std::process::ExitCode;
use tokenizer::Tokenizer as _;
use tool::Report;

/// 由注解声明的固定形状层：形状与算子只写一次，`impl` 与校验测试由宏生成。
#[nn::layer(name = "lm_head", out = [4096], ops = ["gemv_i4"])]
struct LmHead;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let invocation = match cli::Invocation::parse(&args) {
        Ok(invocation) => invocation,
        Err(error) => {
            eprintln!("✗ {error}");
            eprint!("{}", cli::USAGE);
            return ExitCode::from(2);
        }
    };
    match invocation.command() {
        cli::Command::Help => {
            print!("{}", cli::USAGE);
            ExitCode::SUCCESS
        }
        cli::Command::Status => match status() {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("✗ {error}");
                ExitCode::FAILURE
            }
        },
        cli::Command::Verify { .. } | cli::Command::Bench { .. } => {
            // 组装点只分派：这两个审计命令今天由 `cutile` 二进制提供，`llm` 不复制实现。
            eprintln!("✗ 该命令当前由 `cargo run --bin cutile -- ...` 提供（`llm` 只负责组装）");
            ExitCode::from(2)
        }
    }
}

/// 逐层构造一个最小实例：任何一层不可用都会让这个函数失败。
fn status() -> Result<()> {
    let ops = op::builtin();
    let profiles = prof::builtin();
    let mut report = Report::new("llm —— 组装点（各层最小实例）")
        .row(
            "声明层  op",
            format!(
                "{} 个算子（{}）",
                ops.len(),
                tool::join_names(ops.iter().map(op::Op::name))
            ),
        )
        .row(
            "档案层  prof",
            format!(
                "{} 份设备档案（{}）",
                profiles.len(),
                tool::join_names(profiles.iter().map(prof::Archive::name))
            ),
        );

    report = report.row(
        "层定义  nn",
        format!(
            "{} → {:?}，算子 {}",
            LmHead.name(),
            LmHead.output().dims(),
            LmHead.spec().ops().join("、")
        ),
    );

    // 层的性能下界：由算子语义推出（不是手写常数），与内核实测对账时才谈得上"优化了多少"。
    let projection = nn::Linear::new("blk.0.attn_q", 4096, 4096)?;
    let floor = projection.bounds()?;
    report = report.row(
        "下界    nn::Bounds",
        format!(
            "{} 4096 → ≥ {:.2} MB / {:.1} MFLOP，强度 {:.2} flop/byte",
            projection.name(),
            floor.bytes() as f64 / 1e6,
            floor.flops() / 1e6,
            floor.intensity()
        ),
    );

    let tensor = tensor::Tensor::new(tensor::DType::F32, tensor::Shape::new([2, 2]), &[0u8; 16])?;
    report = report.row(
        "张量    tensor",
        format!(
            "{:?} {:?} = {} 元素，连续布局：{}",
            tensor.dtype(),
            tensor.shape().dims(),
            tensor.numel(),
            tensor.is_contiguous()
        ),
    );

    let buffer = vec![0.5f32, 0.25, 0.125, 0.0625];
    let view = tensor::View::new(tensor::Shape::new([2, 2]), &buffer)?;
    report = report.row(
        "视图    tensor::View",
        format!(
            "{:?} {:?}，第 1 行和 = {:.3}（借用调用方 {} 个元素，不复制）",
            view.dtype(),
            view.shape().dims(),
            view.row(1)
                .map(|row| row.iter().sum::<f32>())
                .unwrap_or(0.0),
            view.numel()
        ),
    );

    let header = weights::Header::new(Vec::new(), 0)?;
    report = report.row(
        "权重    weights",
        format!(
            "{} 条条目，数据段 {} 字节",
            header.entries().len(),
            header.data_bytes()
        ),
    );

    let schedule = train::Schedule::new(10, 100)?;
    report = report.row(
        "训练    train",
        format!(
            "warmup {}/{} 步，第 0 步 lr = {:.3}",
            schedule.warmup(),
            schedule.total(),
            schedule.lr(1.0, 0)
        ),
    );

    let route = serve::Route::parse("POST /v1/completions")?;
    let tokens = tokenizer::Bytes.encode("hi")?;
    report = report.row(
        "服务    serve",
        format!(
            "{} {}，提示 \"hi\" → {} 个 token",
            route.method(),
            route.path(),
            tokens.len()
        ),
    );

    let mut scheduler = engine::Scheduler::new(kv::Cache::new(4, 16)?);
    let sequence = engine::Sequence::new(0, 8, 4, 16, exec::Stream::new(0))?;
    let response = serve::admit(&mut scheduler, &sequence)?;
    report = report.row(
        "调度    engine",
        format!(
            "准入 {} 个序列，占用 {} 页（页宽 {}），响应 {}",
            scheduler.running(),
            scheduler.cache().in_use(),
            scheduler.cache().page_size(),
            response.status().code()
        ),
    );

    print!("{}", report.render());
    Ok(())
}

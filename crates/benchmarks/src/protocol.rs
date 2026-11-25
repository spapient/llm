//! 协议层基准：`Span::parse` 的四种写法与 `cap::negotiate` 的一次能力协商。

use crate::fixtures::{checked_or_exit, found};
use criterion::Criterion;
use cutile::builtin::Catalogue;
use cutile::cap;
use prof::atom::Span;
use std::hint::black_box;

/// `(用例名, 字面量)`：精确、上界、区间、开区间。
const SPANS: [(&str, &str); 4] = [
    ("exact", "1024"),
    ("at_most", "<=512"),
    ("range", "16..64"),
    ("open_range", "16.."),
];

/// 四种写法各测一次解析。
pub(crate) fn span(c: &mut Criterion) {
    let mut group = c.benchmark_group("span_parse");
    for (name, text) in SPANS {
        checked_or_exit(Span::parse(text), "Span 写法");
        group.bench_function(name, |bench| {
            bench.iter(|| black_box(Span::parse(black_box(text))))
        });
    }
    group.finish();
}

/// 内置目录在 `cpu` 档案上协商 `gemv_i4`：应命中首档。
pub(crate) fn negotiate(c: &mut Criterion) {
    let catalogue = checked_or_exit(Catalogue::try_builtin(), "内置目录");
    let op = found(catalogue.op("gemv_i4"), "内置算子 gemv_i4");
    let profile = found(catalogue.profile("cpu"), "内置 cpu 档案");

    let mut group = c.benchmark_group("cutile/negotiate");
    group.bench_function("gemv_i4_on_cpu", |bench| {
        bench.iter(|| black_box(cap::negotiate(black_box(op), black_box(profile))))
    });
    group.finish();
}

/// 协议卡的真实 decode 测量装置；断言保留在回归测试中，判定规则仍属 `tool::card`。
#[cfg(test)]
pub(crate) mod decode_card {
    use std::hint::black_box;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use engine::exec::Stream;
    use engine::kv::{Cache, Pager};
    use engine::{Batch, Sequence};
    use serve::Predict;
    use serve::model::CpuModel;
    use tool::bench::Harness;
    use tool::card::{Card, Direction, Protocol, Statistic};

    use crate::fixtures::PROMPT;

    pub(crate) const WINDOWS: usize = 32;
    const WARMUP: usize = 4;
    const PAGES: usize = 128;
    const PAGE_SIZE: usize = 4;

    /// 合成模型、批与序列跨窗口复用，前向内不重新分配权重。
    fn fixture() -> (CpuModel, Batch, Sequence) {
        let model = crate::fixtures::synthetic_model();
        let pager = Pager::new(Cache::new(PAGES, PAGE_SIZE).expect("页池"), 2).expect("行宽");
        let mut batch = Batch::new(pager);
        let sequence = Sequence::new(0, PROMPT, 2, PAGE_SIZE, Stream::new(0)).expect("序列");
        batch.admit(sequence.clone()).expect("准入");
        (model, batch, sequence)
    }

    /// 热身后逐窗口测纯前向墙钟，不混入调度与序列化时间。
    fn timed(model: &mut CpuModel, batch: &mut Batch, sequence: &Sequence) -> Vec<f64> {
        for _ in 0..WARMUP {
            black_box(model.logits(sequence, batch).expect("热身前向"));
        }
        (0..WINDOWS)
            .map(|_| {
                let started = Instant::now();
                black_box(model.logits(sequence, batch).expect("一步前向"));
                started.elapsed().as_nanos() as f64
            })
            .collect()
    }

    /// 三个协议卡用例共享同一种真实样本采集路径。
    pub(crate) fn samples() -> Vec<f64> {
        let (mut model, mut batch, sequence) = fixture();
        timed(&mut model, &mut batch, &sequence)
    }

    /// 本机装置身份：未知机型不猜，工具链取运行环境。
    pub(crate) fn harness(host: String) -> Harness {
        Harness::new(
            host,
            "unknown",
            std::env::consts::OS,
            std::env::consts::ARCH,
            "rustc",
            "cpu-reference",
        )
    }

    /// 声明变异系数门槛与样本下限。
    pub(crate) fn protocol(name: &'static str, threshold: f64, least: usize) -> Protocol {
        Protocol::of(
            name,
            "decode_step_ns",
            Statistic::Cv,
            Direction::Lower,
            threshold,
        )
        .expect("协议")
        .with_least(least)
        .expect("样本下限")
    }

    /// `E10`：95% 区间上界低于 5% 才通过。
    pub(crate) fn jitter() -> Protocol {
        protocol("E10", 0.05, WINDOWS)
    }

    /// 卡片归档在工作区根，CI 才能收集；不落到成员目录。
    pub(crate) fn archive(card: &Card, host: String) -> PathBuf {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("target/bench-report")
            .join(host);
        std::fs::create_dir_all(&directory).expect("归档目录");
        let path = directory.join(format!("card-{}.json", card.protocol().name()));
        std::fs::write(&path, card.to_json()).expect("写卡");
        path
    }
}

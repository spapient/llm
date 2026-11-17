//! P0-6：tokenizer / op / prof 三族纯函数接进 [`sweep`] 契约。
//!
//! 既有 9 个解析器目标在 `parsers.rs` 遍历；本文件按同一口径接线三族的边界纯函数：词表查找、
//! 特殊句柄、字节/流式解码、表达式与轴长度校验、调度解析、归一化/归约/逐元素/查表内核，以及
//! 区间与闭集解析。种子范例在 [`seeds`]，适配在 [`p06_drivers`]；每条目标跑同一份确定性扰动，
//! 核对"不 panic / 结论稳定 / 拒绝有话说"三条契约。
//!
//! 形状类内核（softmax / rmsnorm / 归约 / embed / rope）的合法种子覆盖非空、对齐长度；空输入、
//! 非对齐长度与越界索引由种子的确定性变异派生，路径必须返回带说明的拒绝而不是 panic。

#[path = "support/drivers.rs"]
mod drivers;
#[path = "support/p06_drivers.rs"]
mod p06_drivers;
#[path = "support/seeds.rs"]
mod seeds;

use llm_fuzz::{Seed, Target, sweep};
use p06_drivers::*;

/// 每个目标跑多少条派生用例；与 `parsers.rs` 同档 512，三族合计 18 目标 × 512。
const CASES: u64 = 512;

/// 全部 P0-6 目标及其合法种子；顺序与 [`p06_targets`] 的适配器一一对应。
fn targets() -> [(&'static dyn Target, Seed); 20] {
    let adapters: [&'static dyn Target; 20] = [
        &VocabLookup,
        &SpecialHandles,
        &BytesDecode,
        &StreamDecode,
        &ExprParse,
        &ExtentsNew,
        &ScheduleParse,
        &SoftmaxF32,
        &RmsnormF32,
        &ReduceSumF32,
        &ReduceMaxF32,
        &AddF32,
        &MulF32,
        &ScaleF32,
        &SiluF32,
        &EmbedF32,
        &RopeF32,
        &SpanParse,
        &DequantI4,
        &DequantI8,
    ];
    let seeds = p06_seeds();
    std::array::from_fn(|i| (adapters[i], Seed::new(seeds[i].1.clone())))
}

/// 合法范例：与适配器顺序一致；二进制/数值格式一律经库里的构造路径产出，不手抄字节。
fn p06_seeds() -> [(&'static str, Vec<u8>); 20] {
    [
        ("p06_vocab", seeds::p06_vocab()),
        ("p06_special", vec![1u8, 2, 3]),
        ("p06_bytes", (0u32..4).flat_map(u32::to_le_bytes).collect()),
        ("p06_stream", "hi".as_bytes().to_vec()),
        ("p06_expr", seeds::P06_EXPR.to_vec()),
        ("p06_extents", vec![0x10, b'n']),
        ("p06_schedule", seeds::P06_SCHEDULE.to_vec()),
        ("p06_softmax", seeds::p06_f32_row()),
        ("p06_rmsnorm", seeds::p06_f32_pair()),
        ("p06_reduce_sum", seeds::p06_f32_row()),
        ("p06_reduce_max", seeds::p06_f32_row()),
        ("p06_add", seeds::p06_f32_pair()),
        ("p06_mul", seeds::p06_f32_pair()),
        ("p06_scale", seeds::p06_f32_row()),
        ("p06_silu", seeds::p06_f32_row()),
        ("p06_embed", seeds::p06_embed()),
        ("p06_rope", {
            let mut out = vec![7u8];
            out.extend_from_slice(&seeds::p06_f32_row());
            out
        }),
        ("p06_span", seeds::P06_SPAN.to_vec()),
        ("p06_dequant_i4", seeds::P06_DEQUANT_I4.to_vec()),
        ("p06_dequant_i8", seeds::p06_dequant_i8()),
    ]
}

#[test]
fn every_p06_target_turns_arbitrary_bytes_into_a_stable_verdict() {
    for (target, seed) in targets() {
        if let Err(finding) = sweep(target, &seed, CASES) {
            panic!("{finding}");
        }
    }
}

/// 种子本身必须被接受：否则遍历只在错误路径上打转，"合法输入也只许一种结论"没人验。
#[test]
fn every_p06_seed_is_a_valid_input() {
    for (target, seed) in targets() {
        assert!(
            target.run(seed.bytes()).is_ok(),
            "{} 的种子应当被接受：{}",
            target.name(),
            llm_fuzz::repro(seed.bytes())
        );
    }
}

/// 闭集解析目标（原语 / 存储 / 类型）不强制种子被接受：它们把未知名归一成空结论，
/// 契约只要求不 panic、结论稳定、拒绝有话说。
#[test]
fn closed_set_parsers_are_stable_on_arbitrary_bytes() {
    let parsers: [&'static dyn Target; 3] = [&PrimParse, &SpaceParse, &DTypeParse];
    let seeds = [
        Seed::new(seeds::P06_PRIM),
        Seed::new(seeds::P06_SPACE),
        Seed::new(seeds::P06_DTYPE),
    ];
    for (target, seed) in parsers.iter().zip(&seeds) {
        if let Err(finding) = sweep(*target, seed, CASES) {
            panic!("{finding}");
        }
    }
}

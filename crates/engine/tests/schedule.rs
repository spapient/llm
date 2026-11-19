//! 黑盒契约：调度器作为"第三方"使用时的准入、推进、腾页与错误类别。
//!
//! 内联测试直接摸页表与序列内部字段；这里只经 `Scheduler` / `Sequence` / `Progress` 的公开面
//! 断言对外承诺：准入回答"能不能进"，推进回答"生成没生成"，`Blocked` 不修改状态且可原样重试，
//! 三个状态不会互相越界。
//!
//! 另含 vendor codegen 白盒对拍：CUDA harness 与 Metal 网格内核的文本判据。

use diagnostic::Code;
use llm_engine::codegen::cuda_harness;
use llm_engine::codegen::metal_grid_kernel;
use llm_engine::exec::Stream;
use llm_engine::kv::{Cache, Pager};
use llm_engine::{Progress, Scheduler, Sequence};
use op::model::{Buffer, Sample, Values};
use op::sig::Signature;

/// 页池：`pages` 个页，每页 `page_size` 个 token、每行 2 个元素。
fn scheduler(pages: usize, page_size: usize) -> Scheduler {
    let pager =
        Pager::new(Cache::new(pages, page_size).expect("页池参数合法"), 2).expect("行宽参数合法");
    Scheduler::new(pager)
}

fn sequence(id: u64, prompt: usize, max_new: usize, page_size: usize) -> Sequence {
    let prompt = vec![0u32; prompt];
    Sequence::new(id, &prompt, max_new, page_size, Stream::new(id as u32)).expect("序列参数合法")
}

/// 准入判定：`None` 是"现在装不下"，`Some` 是能开始。
fn admitted(scheduler: &mut Scheduler, sequence: &Sequence, why: &str) -> bool {
    scheduler
        .admit(sequence)
        .unwrap_or_else(|_| panic!("{why}"))
        .is_some()
}

#[test]
fn a_sequence_generates_one_token_per_step_until_done() {
    let mut scheduler = scheduler(4, 16);
    let mut sequence = sequence(1, 4, 2, 16);
    assert!(
        admitted(&mut scheduler, &sequence, "准入"),
        "空页池必须能放下短序列"
    );
    assert_eq!(sequence.produced(), 0);

    assert_eq!(
        scheduler.advance(&mut sequence).expect("推进"),
        Progress::Running,
        "两步里只走了一步"
    );
    assert_eq!(
        scheduler.advance(&mut sequence).expect("推进"),
        Progress::Done
    );
    assert_eq!(sequence.produced(), 2);
    assert!(sequence.is_done(), "`Done` 与 `is_done` 必须同一个判据");
    assert!(scheduler.finish(sequence.id()).expect("释放"));
}

#[test]
fn a_sequence_that_cannot_run_to_completion_is_never_admitted() {
    let mut scheduler = scheduler(2, 8);
    // 最坏情况：提示 1 + 生成 64 个 token，需要 9 页，而池只有 2 页。
    let sequence = sequence(1, 1, 64, 8);
    assert!(
        !scheduler.fits(&sequence),
        "独占页池也跑不完的请求不算放得下"
    );
    assert!(
        !admitted(&mut scheduler, &sequence, "准入"),
        "放不下就不能进批"
    );
    assert_eq!(scheduler.running(), 0);
}

#[test]
fn a_sequence_occupies_only_the_pages_its_current_length_needs() {
    let mut scheduler = scheduler(4, 4);
    let mut sequence = sequence(1, 4, 8, 4);
    assert!(admitted(&mut scheduler, &sequence, "准入"));
    assert_eq!(
        scheduler.cache().in_use(),
        1,
        "长度 4 正好一页，不为最坏情况预占"
    );

    // 生成到长度 5：跨进第二页。
    assert_eq!(
        scheduler.advance(&mut sequence).expect("推进"),
        Progress::Running
    );
    assert_eq!(sequence.len(), 5);
    assert_eq!(scheduler.cache().in_use(), 2, "跨页即增长一页");
}

#[test]
fn a_blocked_step_neither_generates_nor_moves_pages() {
    // 两个页、每页 2 个 token：两条序列各需 2 页时池刚好占满，第三步必然有人要不到页。
    let mut scheduler = scheduler(2, 2);
    let mut first = sequence(1, 2, 2, 2);
    let second = sequence(2, 2, 2, 2);
    assert!(admitted(&mut scheduler, &first, "准入"));
    assert!(admitted(&mut scheduler, &second, "准入"));

    // 推进到某一条进入 Done 或 Blocked 为止：两条都占满页池时，任意一条继续都要新页。
    let before = scheduler.cache().in_use();
    let outcome = scheduler.advance(&mut first).expect("推进");
    let after = scheduler.cache().in_use();
    match outcome {
        Progress::Blocked => {
            assert_eq!(first.produced(), 0, "`Blocked` 不许生成 token");
            assert_eq!(after, before, "`Blocked` 不许改页账");
            assert!(
                scheduler.blocked().contains(&first.id()),
                "刚被阻塞的序列必须在等待清单里，调用方据此挑牺牲者"
            );
        }
        Progress::Running | Progress::Done => {
            assert_eq!(first.produced(), 1, "非阻塞结局必须真的生成一个 token");
            assert_eq!(
                after,
                before.max(scheduler.cache().in_use()),
                "页账只增不减"
            );
        }
    }
    assert!(scheduler.finish(second.id()).expect("腾页"));
    assert!(scheduler.finish(first.id()).expect("腾页"));
    assert_eq!(scheduler.cache().in_use(), 0, "两条都结束后页必须全部归还");
}

#[test]
fn parameters_that_cannot_work_are_rejected_with_categories() {
    let error = Sequence::new(0, &[0; 1], 0, 4, Stream::new(0)).expect_err("max_new 为 0");
    assert_eq!(error.code().as_str(), "invalid");
    let error = Sequence::new(0, &[0; 1], 1, 0, Stream::new(0)).expect_err("页宽为 0");
    assert_eq!(error.code().as_str(), "invalid");
    let error = Cache::new(0, 4).expect_err("零页池");
    assert_eq!(error.code().as_str(), "invalid");
}

#[test]
fn finishing_an_unknown_sequence_reports_absence_without_moving_pages() {
    // `finish` 的对外契约是"返回它是否在批里"，不是"未知 id 就报错"：把不在批里的 id 当成错误，
    // 会让幂等收尾（两条路径都调一次 finish）变成必须先查再删。
    let mut scheduler = scheduler(2, 4);
    let sequence = sequence(1, 2, 1, 4);
    assert!(admitted(&mut scheduler, &sequence, "准入"));
    assert!(scheduler.finish(sequence.id()).expect("在批里"));
    let after_first = scheduler.cache().in_use();
    assert_eq!(after_first, 0, "结束即归还，页数回到零");

    assert!(
        !scheduler.finish(42).expect("未知 id 不报错"),
        "从未准入的 id 必须如实回答「不在批里」"
    );
    assert_eq!(
        scheduler.cache().in_use(),
        after_first,
        "未知 id 不得动页账"
    );
    assert_eq!(scheduler.running(), 0);
}

#[test]
fn admitting_the_same_id_twice_is_rejected_with_a_category() {
    let mut scheduler = scheduler(4, 4);
    let sequence = sequence(7, 2, 2, 4);
    assert!(admitted(&mut scheduler, &sequence, "首次准入"));
    let error = scheduler
        .admit(&sequence)
        .expect_err("重复 id 必须拒绝，而不是覆盖前一条");
    assert_eq!(
        error.code().as_str(),
        "invalid",
        "重复 id 是请求自己不成立，重试不会好"
    );
}

/// KV 也经页表进出：写进去的行按页寻址读回，页不存在或序列已离开批时读写都失败。
///
/// 这是"模型不必自己拿页表再算一次偏移"的对外承诺——调用方只给 id 与位置。
#[test]
fn kv_rows_are_addressed_through_the_page_table() {
    let mut scheduler = scheduler(4, 4);
    let sequence = sequence(1, 5, 2, 4);
    assert!(admitted(&mut scheduler, &sequence, "准入"));
    // 提示 5 个 token 占 2 页（8 个位置），所以第 0..8 个位置都已经有页。
    scheduler.write_kv(1, 0, &[1.5, 2.5]).expect("写第一行");
    scheduler
        .write_kv(1, 5, &[3.5, 4.5])
        .expect("写跨页的那一行");
    assert_eq!(scheduler.kv(1, 0).expect("读第一行"), [1.5, 2.5]);
    assert_eq!(scheduler.kv(1, 5).expect("读跨页的那一行"), [3.5, 4.5]);

    // 第 8 个位置落在第三页上，而这条序列只占了两页：没有地址可写。
    assert_eq!(
        scheduler
            .write_kv(1, 8, &[0.0, 0.0])
            .unwrap_err()
            .code()
            .as_str(),
        "invalid"
    );
    // 行宽必须恰好等于页池的行宽：半行会让同一页里的后续行错位。
    assert_eq!(
        scheduler
            .write_kv(1, 1, &[1.0])
            .unwrap_err()
            .code()
            .as_str(),
        "invalid"
    );
    // 不在批里的 id 没有页表，也就没有地址。
    assert_eq!(scheduler.kv(2, 0).unwrap_err().code().as_str(), "invalid");

    assert!(scheduler.finish(1).expect("结束"));
    assert_eq!(
        scheduler.kv(1, 0).unwrap_err().code().as_str(),
        "invalid",
        "离开批的序列不能再用旧地址读"
    );
}

/// 页复用时读到的是零：上一条序列的 KV 不会漏给下一条。
///
/// 这条断言必须经调度器走完整回路（准入 → 写 → 结束 → 再准入），因为清零发生在归还那一刻。
#[test]
fn a_reused_page_does_not_leak_the_previous_sequences_kv() {
    let mut scheduler = scheduler(1, 4);
    let first = sequence(1, 3, 1, 4);
    assert!(admitted(&mut scheduler, &first, "准入"));
    scheduler.write_kv(1, 0, &[9.0, 9.0]).expect("写第一行");
    assert!(scheduler.finish(1).expect("结束"));
    assert_eq!(
        scheduler.cache().in_use(),
        0,
        "页已归还，会被下一条序列复用"
    );

    let second = sequence(2, 3, 1, 4);
    assert!(admitted(&mut scheduler, &second, "同一页再借出来"));
    assert_eq!(
        scheduler.kv(2, 0).expect("读"),
        [0.0, 0.0],
        "复用页不能带出上一条序列的字节"
    );
}

// ---------------------------------------------------------------------------
// vendor codegen 白盒对拍
// ---------------------------------------------------------------------------
//
// 生产态 Metal / CUDA 内核 lowering 已从 cutile 搬到 `engine::codegen`（M3 治理：机制层
// 不出现厂商字面量），这些文本判据随之迁来。签名 / 样本夹具在本地复制，不依赖 cutile 的
// harness 内部夹具——两者测的是不同层的产物：cutile 测 C 宿主基准，这里测 vendor 生产态源。

fn signature() -> Signature {
    Signature::parse(
        "x",
        "void cutile_x_kernel(const float *x, const unsigned char *w, const float *scales,\n\
         float *y, int M, int N, int gsz) {}\n",
    )
    .expect("签名")
}

fn sample() -> Sample {
    let buffers = [
        ("x", Values::F32(vec![1.0, 1.0, 1.0, 1.0])),
        ("w", Values::U8(vec![0x98, 0x76])),
        ("scales", Values::F32(vec![0.5, 0.5])),
        ("y", Values::F32(vec![0.0, 0.0])),
    ];
    Sample {
        m: 4,
        n: 2,
        scalars: vec![("M", 4), ("N", 2), ("gsz", 4)],
        buffers: buffers
            .into_iter()
            .map(|(name, values)| Buffer { name, values })
            .collect(),
    }
}

#[test]
fn cuda_harness_makes_device_kernel_launch_and_readback() {
    let source =
        cuda_harness("void cutile_x_kernel(void) {}\n", &signature(), &sample()).expect("可生成");
    assert!(source.contains("__device__ void cutile_x_kernel("));
    assert!(source.contains("__global__ void cutile_launch("));
    assert!(source.contains("cutile_launch<<<1, 1>>>("));
    assert!(source.contains("cudaMemcpy(h_y, d_y, 2 * sizeof(float)"));
    assert!(source.contains("cudaFree(d_x)"));
    assert!(source.contains("cudaFree(d_w)"));
    assert!(source.contains("cudaFree(d_scales)"));
}

#[test]
fn cuda_missing_scalar_argument_is_an_internal_error() {
    let sample = Sample {
        m: 4,
        n: 2,
        scalars: vec![("M", 4), ("N", 2)],
        buffers: vec![
            Buffer {
                name: "x",
                values: Values::F32(vec![1.0, 1.0]),
            },
            Buffer {
                name: "w",
                values: Values::U8(vec![0x98]),
            },
            Buffer {
                name: "scales",
                values: Values::F32(vec![0.5]),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0]),
            },
        ],
    };
    let error = cuda_harness("void cutile_x_kernel(void) {}\n", &signature(), &sample)
        .expect_err("缺 gsz 标量");
    assert!(error.to_string().contains("gsz"), "{error}");
}

#[test]
fn cuda_missing_buffer_is_an_internal_error() {
    // 签名与样本漂移：某个缓冲在样本里不存在，host 分配段无法拼出声明。
    let sample = Sample {
        m: 4,
        n: 2,
        scalars: vec![("M", 4), ("N", 2), ("gsz", 4)],
        buffers: vec![
            Buffer {
                name: "x",
                values: Values::F32(vec![1.0, 1.0]),
            },
            Buffer {
                name: "y",
                values: Values::F32(vec![0.0]),
            },
        ],
    };
    let error = cuda_harness("void cutile_x_kernel(void) {}\n", &signature(), &sample)
        .expect_err("缺 w 缓冲");
    assert!(error.to_string().contains("漂移"), "{error}");
}

#[test]
fn metal_grid_kernel_emits_a_grid_dispatched_main() {
    let sig = Signature::parse(
        "scale_f32",
        "void cutile_scale_f32_kernel(const float *a, float *y, int N) {}\n",
    )
    .expect("签名");
    let source = "void cutile_scale_f32_kernel(const float *a, float *y, int N) {}\n";
    let msl = metal_grid_kernel(source, &sig).expect("逐点签名应可生成");
    assert!(msl.contains("kernel void kernel_main("), "{msl}");
    assert!(msl.contains("device float* d_a [[buffer(0)]]"), "{msl}");
    assert!(msl.contains("device float* d_y [[buffer(1)]]"), "{msl}");
    assert!(msl.contains("uint n [[thread_position_in_grid]]"), "{msl}");
    assert!(
        msl.contains("cutile_scale_f32_kernel(d_a + n, d_y + n, 1)"),
        "{msl}"
    );
    assert!(
        !msl.contains("cutile_launch"),
        "生产态入口应是 kernel_main，不是单线程见证：{msl}"
    );
}

#[test]
fn metal_grid_kernel_rejects_a_non_pointwise_signature() {
    // 三标量签名（gemv 形）：行跨步与分组标量不能套「每线程一个元素」的逐点 lowering。
    let source = "void cutile_x_kernel(const float *x, const unsigned char *w, const float *scales,\n\
         float *y, int M, int N, int gsz) {}\n";
    let error = metal_grid_kernel(source, &signature()).expect_err("三标量签名应被拒绝");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("不是逐点"), "{error}");
}

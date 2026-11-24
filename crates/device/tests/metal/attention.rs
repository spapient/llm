//! 职责：长序列 flash attention 的真机对拍与延迟基准。

use super::*;

/// 长序列 flash attention：S=2048、D=64，验证 K/V 分块流式扫描不保留完整分数向量，
/// 与「整行分数留在 threadgroup 内存」的分块路径对拍并报延迟。
///
/// 本用例只出测量：`cargo test -p llm-device --test metal_suite -- metal_flash_attention_long --release --ignored --nocapture`。
#[test]
#[ignore = "Metal 长序列 flash attention：cargo test -p llm-device --test metal_suite -- metal_flash_attention_long --release --ignored --nocapture"]
fn metal_flash_attention_long() {
    let Some((device, queue)) = bench_device("长序列 flash attention 基准") else {
        return;
    };
    crate::backends::install();

    const S: usize = 2048;
    const D: usize = 64;
    const THREADS: usize = 64;

    let AttentionBench(q_buf, k_buf, v_buf, mut y_buf, reference) = attention_bench(&device, S, D);

    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-flash-long");
    let grid_pipe = grid_pipeline(
        &device,
        &catalogue,
        &dirs,
        "attention_f32",
        S,
        D,
        "网格注意力",
    );
    let launch = [(S * THREADS) as u32, 1, 1];
    let block = [THREADS as u32, 1, 1];
    queue.execute(
        &grid_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        [S as u32, 1, 1],
        [1, 1, 1],
    );
    assert_matmul_matches("attn-long-grid", &reference, y_buf.as_mut_bytes());
    let grid_ns = measure(
        &queue,
        &grid_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        [S as u32, 1, 1],
        [1, 1, 1],
    );
    eprintln!("Metal 长序列 attention {S}×{D}：网格行线程={grid_ns:.1} ns");

    let tiled_msl = program(
        &catalogue,
        &dirs,
        "attention_f32",
        Metal,
        Mode::tiled_attention(THREADS),
        S,
        D,
    )
    .and_then(|stages| stages.one("分块注意力"))
    .expect("分块注意力 MSL 应能生成");
    let tiled_pipe = device.compile(&tiled_msl).expect("分块注意力 MSL 应能编译");
    queue.execute(
        &tiled_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        launch,
        block,
    );
    assert_matmul_matches("attn-long-tiled", &reference, y_buf.as_mut_bytes());
    let tiled_ns = measure(
        &queue,
        &tiled_pipe,
        &[&q_buf, &k_buf, &v_buf, &y_buf],
        launch,
        block,
    );
    eprintln!(
        "  整行分数分块线程组={tiled_ns:.1} ns 对网格={:.2}×",
        grid_ns / tiled_ns
    );

    for key_block in [64usize, 128, 256] {
        let msl = program(
            &catalogue,
            &dirs,
            "attention_f32",
            Metal,
            Mode::flash(THREADS, key_block),
            S,
            D,
        )
        .and_then(|stages| stages.one("flash"))
        .expect("flash attention MSL 应能生成");
        let pipe = device.compile(&msl).expect("flash attention MSL 应能编译");
        queue.execute(&pipe, &[&q_buf, &k_buf, &v_buf, &y_buf], launch, block);
        assert_matmul_matches("attn-long-flash", &reference, y_buf.as_mut_bytes());
        let ns = measure(
            &queue,
            &pipe,
            &[&q_buf, &k_buf, &v_buf, &y_buf],
            launch,
            block,
        );
        eprintln!(
            "  flash block={key_block}：{ns:.1} ns 对网格={:.2}× 对分块={:.2}×",
            grid_ns / ns,
            tiled_ns / ns,
        );
    }
}

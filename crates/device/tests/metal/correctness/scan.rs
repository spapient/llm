//! 职责：decode 与 latent 注意力扫过单块容量的历史轴判据。

use super::*;

/// decode 的显式历史长度样本：模型自带的 `sample(q_rows, d)` 把 `HIST` 钉在 4，
/// 而「历史比分块容量长」正是要验的那一档，只能由用例自己给出。
fn decode_sample(q_rows: usize, hist: usize, d: usize) -> (Sample, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(0x5eed_0000_0000_0005 ^ (hist as u64));
    let q = rng.values(q_rows * d);
    let k = rng.values(hist * d);
    let v = rng.values(hist * d);
    let buf = |name: &'static str, values: Vec<f32>| Buf {
        name,
        values: Values::F32(values),
    };
    let sample = Sample {
        m: q_rows,
        n: d,
        scalars: vec![
            ("Q_ROWS", q_rows as i64),
            ("HIST", hist as i64),
            ("D", d as i64),
        ],
        bufs: vec![
            buf("q", q.clone()),
            buf("k", k.clone()),
            buf("v", v.clone()),
            buf("y", vec![0.0; q_rows * d]),
        ],
    };
    (sample, q, k, v)
}

/// 模块级用例钩子：`decode_attention_f32` 的历史长度可以超过单个分块（模板原来写死
/// `float scores[64]` 并按 `j < HIST` 无界写入，HIST > 64 时越界写栈），必须扫完整个历史。
///
/// 判据与 `attention_f32` 同源：相对 Frobenius 误差对齐 `conv` 的 online-softmax 参考
/// ——内核是朴素三阶段、参考是单遍 online，两者算法不同，比的才是数学而不是同一段代码。
#[test]
fn decode_attention_scans_history_past_one_block() {
    const Q_ROWS: usize = 2;
    const HIST: usize = 128;
    const D: usize = 8;

    let Some((device, queue)) = bench_device("decode 历史扫描") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-decode");

    let (sample, q, k, v) = decode_sample(Q_ROWS, HIST, D);
    let want = want_from(&op::attention::decode_attention::reference(&sample).expect("CPU 参考"));
    let (q_buf, k_buf, v_buf, mut y) = attention_bufs(&device, &q, &k, &v, Q_ROWS * D);
    let pipe = grid_pipeline_sample(
        &device,
        &catalogue,
        &dirs,
        "decode_attention_f32",
        &sample,
        "decode 网格",
    );
    assert!(
        queue.execute(
            &pipe,
            &[&q_buf, &k_buf, &v_buf, &y],
            [Q_ROWS as u32, 1, 1],
            [1, 1, 1]
        ),
        "decode 网格派发应成功"
    );

    let got = read_back(y.as_mut_bytes());
    let error = frobenius(&got, &want);
    assert!(
        error < 1e-4,
        "decode HIST={HIST}（两个满块）的相对误差 {error:.2e} 超过 1e-4"
    );
    println!("decode 注意力 Q_ROWS={Q_ROWS} HIST={HIST} D={D}：相对误差 {error:.2e}");
}

/// 同一类越界写：`latent_attention_f32` 的时间轴 `T` 也吃着写死的 `scores[64]`。
/// 该算子的键行数与查询行数前视同构（`T = R`），所以 `R = T = 128` 就把 `T > 64` 走通。
#[test]
fn latent_attention_scans_time_axis_past_one_block() {
    const R: usize = 128;
    const W: usize = 8;

    let Some((device, queue)) = bench_device("latent 时间轴扫描") else {
        return;
    };
    crate::backends::install();
    let (catalogue, dirs) = catalogue_and_dirs("engine-metal-latent");

    let sample = op::attention::latent::sample(R, W);
    let read = |name: &str| sample.f32(name).expect("样本缓冲").to_vec();
    let (q, k, v) = (read("q"), read("k"), read("v"));
    let dc = sample.scalar("DC").expect("样本缺 DC") as usize;
    let want = want_from(&op::attention::latent::reference(&sample).expect("CPU 参考"));
    let (q_buf, k_buf, v_buf, mut y) = attention_bufs(&device, &q, &k, &v, R * dc);
    let pipe = grid_pipeline_sample(
        &device,
        &catalogue,
        &dirs,
        "latent_attention_f32",
        &sample,
        "latent 网格",
    );
    assert!(
        queue.execute(
            &pipe,
            &[&q_buf, &k_buf, &v_buf, &y],
            [R as u32, 1, 1],
            [1, 1, 1]
        ),
        "latent 网格派发应成功"
    );

    let got = read_back(y.as_mut_bytes());
    let error = frobenius(&got, &want);
    assert!(
        error < 1e-4,
        "latent T={R}（两个满块）的相对误差 {error:.2e} 超过 1e-4"
    );
    println!("latent 注意力 R={R} T={R} W={W} DC={dc}：相对误差 {error:.2e}");
}

use super::*;
use op::model::{Buffer, Values};

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

/// 生成 C 封装文本；`CUTILE_CAPTURE_GOLDEN` 显式打开时才把副本写到 `/tmp/cutile-golden`。
#[test]
fn golden_capture() {
    let capture = std::env::var("CUTILE_CAPTURE_GOLDEN").is_ok();
    let dir = std::path::PathBuf::from("/tmp/cutile-golden");
    for repeat in [1usize, 3, 50] {
        let text = harness(
            "void cutile_x_kernel(void) {}\n",
            &signature(),
            &sample(),
            repeat,
        )
        .expect("可生成");
        assert!(!text.is_empty(), "封装文本不能为空");
        assert!(text.contains("cutile_x_kernel("), "封装里要调用内核");
        if capture {
            std::fs::create_dir_all(&dir).expect("目录");
            std::fs::write(dir.join(format!("c_harness_{repeat}.txt")), text).expect("写");
        }
    }
    let kernel = "void cutile_x_kernel(const float *x, const unsigned char *w, const float *scales,\n\
         float *y, int M, int N, int gsz) {\n}\n";
    let text = harness(kernel, &signature(), &sample(), 3).expect("可生成");
    assert!(
        text.contains("cutile_init_"),
        "输入初值要内联进封装，否则常量输入会被折叠"
    );
    if capture {
        std::fs::create_dir_all(&dir).expect("目录");
        std::fs::write(dir.join("c_harness_kernel.txt"), text).expect("写");
    }
}

#[test]
fn harness_embeds_data_and_calls_the_kernel() {
    let source = harness(
        "void cutile_x_kernel(void) {}\n",
        &signature(),
        &sample(),
        3,
    )
    .expect("可生成");
    assert!(source.contains("static volatile const float cutile_init_x[4]"));
    assert!(source.contains("static volatile const unsigned char cutile_init_w[2]"));
    assert!(source.contains("static volatile const float cutile_init_scales[2]"));
    assert!(source.contains("x[cutile_i] = cutile_init_x[cutile_i];"));
    assert!(source.contains("#define CUTILE_BARRIER()"));
    assert!(source.contains("CUTILE_BARRIER();"));
    assert!(source.contains("volatile float cutile_sink"));
    assert!(
        source.contains("cutile_sink = y[cutile_s];"),
        "整个输出都要保活进 sink"
    );
    assert!(source.contains("float y[2]"));
    assert!(
        !source.contains("static volatile const float cutile_init_y"),
        "输出不能是只读"
    );
    assert!(source.contains("cutile_x_kernel(x, w, scales, y, 4, 2, 4);"));
    assert!(source.contains("for (int cutile_r = 0; cutile_r < 3; ++cutile_r)"));
    assert!(source.contains("clock_gettime(CLOCK_MONOTONIC"));
    assert!(
        source.contains("for (int cutile_warm = 0; cutile_warm < 1; ++cutile_warm)"),
        "warmup 至少一次，且不能与缓冲名撞车"
    );
    assert!(
        source.contains("int cutile_batch = (int)(1e6 /"),
        "小内核要批量计时，否则样本会被时钟分辨率抹平"
    );
    assert!(source.contains("fprintf(stderr, \"ns %.3f\\n\""));
    assert!(
        source.starts_with("#define _POSIX_C_SOURCE"),
        "POSIX 声明必须在系统头之前：{source}"
    );
}

#[test]
fn warmup_is_a_tenth_of_the_measurement_and_never_zero() {
    assert_eq!(warmup_runs(1), 1);
    assert_eq!(warmup_runs(9), 1);
    assert_eq!(warmup_runs(50), 5);
}

#[test]
fn harness_rejects_a_missing_or_mistyped_output() {
    let mut sample = sample();
    sample.buffers.retain(|buffer| buffer.name != "y");
    let error = harness("", &signature(), &sample, 1).unwrap_err();
    assert_eq!(error.code(), Code::Missing);
    assert!(error.to_string().contains("输出缓冲"), "{error}");

    let mut sample = self::sample();
    sample.buffers.retain(|buffer| buffer.name != "y");
    sample.buffers.push(Buffer {
        name: "y",
        values: Values::U8(vec![0]),
    });
    let error = harness("", &signature(), &sample, 1).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("必须是 f32"), "{error}");
}

/// 回读只认 `float`：可写指针不是 `float *` 时要在拼 C 之前报，否则只有编译器的一句 `incompatible pointer type`。
#[test]
fn a_non_float_output_is_rejected_before_any_c_is_generated() {
    let signature = Signature::parse(
        "x",
        "void cutile_x_kernel(const float *x, const unsigned char *w, const float *scales,\n\
         unsigned char *y, int M, int N, int gsz) {}\n",
    )
    .expect("u8 是闭集内的缓冲类型");
    let error = harness("", &signature, &sample(), 1).unwrap_err();
    assert_eq!(error.code(), Code::Unsupported);
    let text = error.to_string();
    assert!(text.contains("`x`"), "算子名要在诊断里：{text}");
    assert!(text.contains("`y`"), "缓冲名要在诊断里：{text}");
    assert!(
        text.contains("unsigned char"),
        "错误要说清 `.k` 实际写的是什么：{text}"
    );
    assert!(text.contains("float"), "{text}");
}

#[test]
fn floats_round_trip_through_the_literal_format() {
    let value = 0.1f32;
    let literal = format!("{value:?}");
    let parsed: f64 = literal.parse().expect("可解析");
    assert_eq!(parsed as f32, value);
}

#[test]
fn f32_literals_round_trip_for_random_finite_values() {
    let mut rng = testkit::Rng::new(0xF17A);
    for _ in 0..48 {
        let value = f32::from_bits(rng.below(u32::MAX));
        if !value.is_finite() {
            continue;
        }
        let literal = format!("{value:?}f");
        let parsed: f32 = literal.trim_end_matches('f').parse().expect("字面量可解析");
        assert_eq!(parsed, value, "往返失真：{value}");
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
    assert!(error.to_string().contains("不是逐点"), "{error}");
}

//! engine 的 vendor codegen 白盒对拍：CUDA harness 与 Metal 网格内核的文本判据。
//!
//! 生产态 Metal / CUDA 内核 lowering 已从 cutile 搬到 `engine::codegen`（M3 治理：机制层
//! 不出现厂商字面量），这些文本判据随之迁来。签名 / 样本夹具在本地复制，不依赖 cutile 的
//! harness 内部夹具——两者测的是不同层的产物：cutile 测 C 宿主基准，这里测 vendor 生产态源。

use diagnostic::Code;
use llm_engine::codegen::cuda::cuda_harness;
use llm_engine::codegen::metal::metal_grid_kernel;
use op::model::{Buffer, Sample, Values};
use op::sig::Signature;

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

use super::*;
use llm_optim::backend::TensorGeometry;
use llm_optim::execution::Backend::Metal;
use llm_optim::execution::{Mode, program, program_sample};
use tens::View;

pub(crate) fn bench_device(hint: &str) -> Option<(MetalDevice, MetalCommandQueue)> {
    let Some(device) = MetalDevice::system_default() else {
        eprintln!("Metal 设备不在本机，{hint}未运行");
        return None;
    };
    let Some(queue) = device.new_command_queue() else {
        panic!("有设备必有命令队列");
    };
    Some((device, queue))
}

/// 内置算子目录 + 该基准自己的临时工作目录：一次基准里的所有 MSL 生成都落在这上面。
pub(crate) fn catalogue_and_dirs(cache: &str) -> (cutile::builtin::Catalogue, cutile::dirs::Dirs) {
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let dirs = cutile::dirs::Dirs::under(std::env::temp_dir().join(cache));
    (catalogue, dirs)
}

/// matmul 测点的确定性输入 + CPU 参考：A、B 各取一套固定取模模式，参考由 f64 累加给出。
pub(crate) fn matmul_inputs(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (a, b) = matmul_values(m, n, k);
    let reference = cpu_matmul(&a, &b, m, n, k);
    (a, b, reference)
}

/// matmul 测点的确定性输入（不含参考）：大形状的 f64 参考要跑上分钟，吞吐测点只要输入。
pub(crate) fn matmul_values(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    (a, b)
}

/// 满尾数样本：每个值都用足 f32 的 24 位有效位（`[0.5, 1)` 内的规约数再缩放到 `(0.7, 1)`
/// 区间）。这种输入进入 half 必然要舍入，因而能验出「张量核确实按 half 精度舍入」而不是
/// 靠夹具恰好可精确表示而得出的假零误差。
pub(crate) fn full_mantissa_values(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    // xorshift64*：确定性、无外部依赖，低位也参与高位运算，避免 `i % 7` 那种短周期。
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u32 << 24) as f32
    };
    let a = (0..m * k).map(|_| 0.7 + next() * 0.3).collect();
    let b = (0..k * n).map(|_| 0.7 + next() * 0.3).collect();
    (a, b)
}

/// 注意力测点的确定性输入 + CPU 参考：Q/K/V 各取一套固定取模模式，参考由 [`cpu_attention`] 给出。
pub(crate) fn attention_inputs(s: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> = (0..s * d).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect();
    let k: Vec<f32> = (0..s * d).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect();
    let v: Vec<f32> = (0..s * d).map(|i| ((i % 3) as f32) * 0.3 - 0.3).collect();
    let reference = cpu_attention(&q, &k, &v, s, d);
    (q, k, v, reference)
}

/// 注意力基准的缓冲套件：Q/K/V 输入缓冲、Y 输出缓冲与 CPU 参考。
pub(crate) struct AttentionBench(
    pub(crate) MetalBuf,
    pub(crate) MetalBuf,
    pub(crate) MetalBuf,
    pub(crate) MetalBuf,
    pub(crate) Vec<f32>,
);

/// 上传 [`attention_inputs`] 的 Q/K/V，再开一路只有 GPU 写的 Y 输出。
pub(crate) fn attention_bench(device: &MetalDevice, s: usize, d: usize) -> AttentionBench {
    let (q, k, v, reference) = attention_inputs(s, d);
    AttentionBench(
        upload(device, &q, "Q"),
        upload(device, &k, "K"),
        upload(device, &v, "V"),
        output_buf(device, s * d, "Y"),
        reference,
    )
}

/// 上传一路输入：共享缓冲按元素数 × 4 分配，`values` 原样拷进去（`label` 只进失败消息）。
pub(crate) fn upload(device: &MetalDevice, values: &[f32], label: &str) -> MetalBuf {
    let mut buf = device
        .new_buf(values.len() * 4)
        .unwrap_or_else(|| panic!("{label} 缓冲"));
    buf.as_mut_bytes().copy_from_slice(&as_bytes(values));
    buf
}

/// 上传一段原始字节（int8 输入走这里，按字节而不是 f32 分配）。
pub(crate) fn upload_bytes(device: &MetalDevice, bytes: &[u8], label: &str) -> MetalBuf {
    let mut buf = device
        .new_buf(bytes.len())
        .unwrap_or_else(|| panic!("{label} 缓冲"));
    buf.as_mut_bytes().copy_from_slice(bytes);
    buf
}

/// 开一路只由 GPU 写的输出缓冲：`len` 个 f32，内容留白（`label` 只进失败消息）。
pub(crate) fn output_buf(device: &MetalDevice, len: usize, label: &str) -> MetalBuf {
    device
        .new_buf(len * 4)
        .unwrap_or_else(|| panic!("{label} 缓冲"))
}

/// 标量网格程序：按 `kernel` 与 `m×n` 生成 MSL 再编译成管线；`what` 只进失败消息。
pub(crate) fn grid_pipeline(
    device: &MetalDevice,
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    kernel: &str,
    m: usize,
    n: usize,
    what: &str,
) -> MetalPipeline {
    let msl = program(catalogue, dirs, kernel, Metal, Mode::declared(), m, n)
        .and_then(|stages| stages.one("降源"))
        .unwrap_or_else(|_| panic!("{what} MSL 应能生成"));
    device
        .compile(&msl)
        .unwrap_or_else(|| panic!("{what} MSL 应能编译"))
}

/// 同 [`grid_pipeline`]，但形状由样本显式给出（`K` 不是 `(m, n)` 的函数，只能这样传）。
pub(crate) fn grid_pipeline_sample(
    device: &MetalDevice,
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    kernel: &str,
    sample: &nn::model::Sample,
    what: &str,
) -> MetalPipeline {
    let msl = program_sample(catalogue, dirs, kernel, Metal, Mode::declared(), sample)
        .and_then(|stages| stages.one("降源"))
        .unwrap_or_else(|_| panic!("{what} MSL 应能生成"));
    device
        .compile(&msl)
        .unwrap_or_else(|| panic!("{what} MSL 应能编译"))
}

/// simdgroup GEMM 的一个片段形状：生成 MSL、编译成管线，并给出对应的网格（每线程组 32 线程）。
pub(crate) fn simdgroup_gemm_pipeline(
    device: &MetalDevice,
    catalogue: &cutile::builtin::Catalogue,
    dirs: &cutile::dirs::Dirs,
    m: usize,
    n: usize,
    row_frags: usize,
    col_frags: usize,
) -> (MetalPipeline, [u32; 3]) {
    let msl = program(
        catalogue,
        dirs,
        "matmul_f32",
        Metal,
        Mode::simdgroup(row_frags, col_frags),
        m,
        n,
    )
    .and_then(|stages| stages.one("simdgroup"))
    .expect("simdgroup MSL 应能生成");
    let pipe = device.compile(&msl).expect("simdgroup MSL 应能编译");
    let grid = [
        ((m / (8 * row_frags)) * 32) as u32,
        (n / (8 * col_frags)) as u32,
        1,
    ];
    (pipe, grid)
}

pub(crate) struct Int8Gemm {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) a: Vec<u8>,
    pub(crate) b: Vec<u8>,
    pub(crate) reference: Vec<f32>,
}

/// 一次 CPU 参考（`nn::matmul_i8::forward_tiled`）给出 `f32` 参考列，设备侧只与它比。
pub(crate) fn int8_gemm() -> Int8Gemm {
    let (m, n, k) = (64usize, 256usize, 4096usize);
    let a: Vec<u8> = (0..m * k)
        .map(|i| (((i % 17) as i32 - 8) as i8) as u8)
        .collect();
    let b: Vec<u8> = (0..k * n)
        .map(|i| (((i % 19) as i32 - 9) as i8) as u8)
        .collect();
    let (sa, sb): (Vec<i8>, Vec<i8>) = (
        a.iter().map(|&x| x as i8).collect(),
        b.iter().map(|&x| x as i8).collect(),
    );
    let mut acc = vec![0i32; m * n];
    nn::matmul_i8::forward_tiled(
        View::matrix(m, k, &sa).expect("a 二维"),
        View::matrix(k, n, &sb).expect("b 二维"),
        &mut acc,
    )
    .expect("CPU 参考");
    Int8Gemm {
        m,
        n,
        k,
        a,
        b,
        reference: acc.iter().map(|&v| v as f32).collect(),
    }
}

/// int8 GEMM 基准的设备侧：`int8_gemm` 的 A/B/Y 三路缓冲 + 网格。
///
/// 两个 int8 用例的输入、输出与网格完全一致，只是 MSL 来源不同（本地生成 / cutile 生产入口），
/// 因此缓冲与网格只在这里建一次。
pub(crate) struct Int8GemmBench {
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) reference: Vec<f32>,
    pub(crate) a: MetalBuf,
    pub(crate) b: MetalBuf,
    pub(crate) y: MetalBuf,
    pub(crate) grid: [u32; 3],
}

/// int8 GEMM 基准的固定几何与输入：64×256×4096，`i8` 输入、`i32` 精确累加、`f32` 参考。
///
/// 两个 int8 用例共用同一份样本，它们对拍的对象因此逐位相同——换一组数不该改变结论。
///
/// 上传 [`int8_gemm`] 的 A/B，再开一路只有 GPU 写的 Y 输出。
pub(crate) fn int8_gemm_bench(device: &MetalDevice) -> Int8GemmBench {
    let fixture = int8_gemm();
    let (m, n, k) = (fixture.m, fixture.n, fixture.k);
    let a = upload_bytes(device, &fixture.a, "A-int8");
    let b = upload_bytes(device, &fixture.b, "B-int8");
    let y = output_buf(device, m * n, "Y-int8");
    Int8GemmBench {
        m,
        n,
        k,
        reference: fixture.reference,
        a,
        b,
        y,
        grid: [m as u32, n as u32, 1],
    }
}
pub(crate) fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], s: usize, d: usize) -> Vec<f32> {
    let scale = (d as f32).sqrt();
    let mut out = vec![0.0f32; s * d];
    let mut scores = vec![0.0f32; s];
    for i in 0..s {
        let mut row_max = f32::MIN;
        for j in 0..s {
            let mut dot = 0.0f64;
            for p in 0..d {
                dot += f64::from(q[i * d + p]) * f64::from(k[j * d + p]);
            }
            scores[j] = (dot as f32) / scale;
            row_max = row_max.max(scores[j]);
        }
        let mut sum = 0.0f64;
        for score in &mut scores {
            *score = (*score - row_max).exp();
            sum += f64::from(*score);
        }
        for h in 0..d {
            let mut acc = 0.0f64;
            for j in 0..s {
                acc += f64::from(scores[j]) * f64::from(v[j * d + h]);
            }
            out[i * d + h] = (acc / sum) as f32;
        }
    }
    out
}

/// 两版 matmul MSL 共用的前导：A、B、C 三路接口声明，`M`/`N`/`K` 由调用方随后内联。
pub(crate) const MATMUL_MSL_HEAD: &str = "#include <metal_stdlib>\n\
     using namespace metal;\n\
     kernel void kernel_main(\n\
     \x20    device const float* A [[buffer(0)]],\n\
     \x20    device const float* B [[buffer(1)]],\n\
     \x20    device float* C [[buffer(2)]],\n";

/// 朴素 MSL：每线程算一个输出元素；`tiled` 为真时用 threadgroup 共享内存把 A/B 按 16×16 分块，
/// 每个线程组 16×16 线程协同搬运一次整块，规约轴复用共享块，避免反复从设备全局内存重读。
pub(crate) fn matmul_msl(tiled: bool, m: usize, n: usize, k: usize) -> String {
    if !tiled {
        return format!(
            "{MATMUL_MSL_HEAD}\
             \x20    uint2 gid [[thread_position_in_grid]]) {{\n\
             \x20    const uint M = {m}, N = {n}, K = {k};\n\
             \x20    float acc = 0.0f;\n\
             \x20    for (uint p = 0; p < K; ++p) acc += A[gid.y * K + p] * B[p * N + gid.x];\n\
             \x20    C[gid.y * N + gid.x] = acc;\n\
             }}"
        );
    }
    format!(
        "{MATMUL_MSL_HEAD}\
         \x20    uint2 gid [[thread_position_in_grid]],\n\
         \x20    uint2 lid [[thread_position_in_threadgroup]]) {{\n\
         \x20    const uint M = {m}, N = {n}, K = {k};\n\
         \x20    threadgroup float As[16][16];\n\
         \x20    threadgroup float Bs[16][16];\n\
         \x20    float acc = 0.0f;\n\
         \x20    for (uint kk = 0; kk < K; kk += 16) {{\n\
         \x20        As[lid.y][lid.x] = A[gid.y * K + kk + lid.x];\n\
         \x20        Bs[lid.y][lid.x] = B[(kk + lid.y) * N + gid.x];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20        for (uint p = 0; p < 16; ++p) acc += As[lid.y][p] * Bs[p][lid.x];\n\
         \x20        threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20    }}\n\
         \x20    C[gid.y * N + gid.x] = acc;\n\
         }}"
    )
}

/// 同一算术的 CPU 参考（f64 累加，保证对拍基准的精度只受量化级控制，不引入累加顺序差异）。
pub(crate) fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

pub(crate) fn as_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

/// 设备输出与 CPU 参考对拍：阈值 1e-3，分母按每元素 `max(|want|, 1)`（[`testkit::Scale::Unit`]）。
///
/// 这里选 `Unit` 而非夹具层默认的整批口径，是因为本文件的输出量级差得远：GEMM 的元素是上百上千项
/// 乘积之和，量级在 1 以上；注意力输出是 softmax 权重乘 V 的和，整批最大幅度只有 1e-4 量级。
/// 整批口径会把注意力输出上 1e-5 的绝对差放大成 1e0 的相对误差，那不是「算错了」而是「分母选错了」。
/// `Unit` 给每元素留 1 的绝对地板，正对这两类输出的共同下限；最差元素的下标与两侧取值仍由夹具层报出。
pub(crate) fn assert_matmul_matches(label: &str, reference: &[f32], output: &[u8]) {
    let got = read_back(output);
    testkit::assert_rel_with(
        testkit::Scale::Unit,
        &got,
        reference,
        1e-3,
        &format!("{label} 的 GPU 输出与 CPU 参考"),
    );
}

/// 把设备缓冲回读成 f32 序列（缓冲按 f32 分配、按 4 字节切块）。
pub(crate) fn read_back(output: &[u8]) -> Vec<f32> {
    let (chunks, _) = output.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect()
}

/// 每元素相对误差的最大值，分母按 `max(|want|, 1)`（[`testkit::Scale::Unit`] 口径），只用于打印。
pub(crate) fn max_relative(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(got, want)| f64::from((got - want).abs()) / f64::from(want.abs()).max(1.0))
        .fold(0.0, f64::max)
}

/// 同一内核连跑 50 次，返回中位数延迟（ns）；预热一次不计入。
/// 同一派发连跑 `SAMPLES` 次取中位，口径对齐 llm.u 的 `measure_kernel`。
pub(crate) const SAMPLES: usize = 30;

pub(crate) fn measure(
    queue: &MetalCommandQueue,
    pipeline: &MetalPipeline,
    bufs: &[&MetalBuf],
    grid: [u32; 3],
    block: [u32; 3],
) -> f64 {
    queue.execute(pipeline, bufs, grid, block);
    let mut samples = [0u128; SAMPLES];
    for slot in &mut samples {
        let start = std::time::Instant::now();
        queue.execute(pipeline, bufs, grid, block);
        *slot = start.elapsed().as_nanos();
    }
    samples.sort_unstable();
    samples[samples.len() / 2] as f64
}

/// 把 `value` 按张量核输入精度舍入：10 位尾数、就近取偶，等价于 f32→half→f32 的往返。
///
/// 只用于造 CPU 参考——`relaxed` 路径的 A、B 都只有 10 位尾数（half / tf32 同级），拿全精度
/// 参考去对拍只会把「精度档不同」误报成「算错了」。指数域不动：测点取值落在 half 正规区间。
pub(crate) fn half_precision(value: f32) -> f32 {
    let bits = value.to_bits();
    let sign = bits & 0x8000_0000;
    let exponent = bits & 0x7f80_0000;
    let mantissa = bits & 0x007f_ffff;
    let dropped = mantissa & 0x1fff;
    let mut kept = mantissa >> 13;
    if dropped > 0x1000 || (dropped == 0x1000 && kept & 1 == 1) {
        kept += 1;
    }
    // 进位要**加**进指数域而不是或进去：`kept` 进位到 1024 时 `kept << 13` 落在指数域最低位，
    // 指数为奇数（如 [0.25,0.5) 的 125）时该位已置 1，或运算会把它吞掉——`0.4999` 会算成 `0.25`。
    f32::from_bits(sign + exponent + (kept << 13))
}

/// 张量核 `relaxed` 口径的对拍阈值：输入只有 10 位尾数、硬件还可能低精度累加，故放到 5e-3；
/// 同形的全 f32 路径（分块 / simdgroup）仍按 1e-3 判。
pub(crate) const TENSOR_TOLERANCE: f64 = 5e-3;

/// 张量核输出与「按同精度舍入后的 f64 参考」对拍。
pub(crate) fn assert_relaxed_matches(label: &str, reference: &[f32], output: &[u8]) -> f64 {
    let got = read_back(output);
    let error = max_relative(&got, reference);
    assert!(
        error < TENSOR_TOLERANCE,
        "{label} 的 GPU 输出与 CPU 参考偏差 {error:.2e} 超过 {TENSOR_TOLERANCE:.0e}"
    );
    error
}

/// 张量核基准的每档形状夹具：`A`/`B` 上传、`B` 预转置成 half 的 `Bt`、输出缓冲、可选 CPU 参考。
pub(crate) struct TensorStage {
    pub(crate) a: MetalBuf,
    pub(crate) bt: MetalBuf,
    pub(crate) y: MetalBuf,
    pub(crate) reference: Option<Vec<f32>>,
}

impl TensorStage {
    /// 与 CPU 参考对拍并返回误差（无参考时 NaN）；拆借 `y` 与 `reference` 两个字段避免借用冲突。
    pub(crate) fn error(&mut self, label: &str) -> f64 {
        let Self { y, reference, .. } = self;
        reference.as_ref().map_or(f64::NAN, |want| {
            assert_relaxed_matches(label, want, y.as_mut_bytes())
        })
    }
}

/// 上传 + 布局前置 + 可选参考：算子装配与裸内核两档基准共用同一套输入准备。
pub(crate) fn tensor_stage(
    device: &MetalDevice,
    queue: &MetalCommandQueue,
    shape: (usize, usize, usize),
    want_reference: bool,
) -> TensorStage {
    let (m, n, k) = shape;
    let (a, b) = matmul_values(m, n, k);
    tensor_stage_from(device, queue, shape, (a, b), want_reference)
}

/// 同 [`tensor_stage`]，输入由调用方给定：样本值能否精确表示成 half，直接决定这条用例
/// 能不能验出舍入误差，因而必须可由用例自己掌握而不是钉在夹具里。
pub(crate) fn tensor_stage_from(
    device: &MetalDevice,
    queue: &MetalCommandQueue,
    (m, n, k): (usize, usize, usize),
    (a, b): (Vec<f32>, Vec<f32>),
    want_reference: bool,
) -> TensorStage {
    let a_buf = upload(device, &a, "A");
    let b_buf = upload(device, &b, "B");
    let bt_buf = device.new_buf(n * k * 2).expect("Bt 缓冲");
    let y_buf = output_buf(device, m * n, "Y");
    // 布局前置：`B[K][N]` 的 f32 → `Bt[N][K]` 的 half，每个形状只跑一次。
    let transpose = crate::tiled::metal_transpose_half(k, n).expect("转置 MSL 应能生成");
    let transpose = device.compile(&transpose).expect("转置 MSL 应能编译");
    assert!(
        queue.execute(
            &transpose,
            &[&b_buf, &bt_buf],
            [(n as u32).div_ceil(32) * 32, (k as u32).div_ceil(32) * 8, 1],
            [32, 8, 1],
        ),
        "转置派发应成功"
    );
    // 参考两路输入也按 half 舍入后再乘，与 GPU 侧口径一致。
    let reference = want_reference.then(|| {
        cpu_matmul(
            &a.iter().copied().map(half_precision).collect::<Vec<f32>>(),
            &b.iter().copied().map(half_precision).collect::<Vec<f32>>(),
            m,
            n,
            k,
        )
    });
    TensorStage {
        a: a_buf,
        bt: bt_buf,
        y: y_buf,
        reference,
    }
}

/// 一条已编译的张量核管线及其派发参数：连跑取中位时复用，不重复编译。
pub(crate) struct Pipe {
    pub(crate) label: String,
    pub(crate) pipe: MetalPipeline,
    pub(crate) grid: [u32; 3],
    pub(crate) block: [u32; 3],
}

/// 编译一个几何档并首跑一次；标签由 `(tile_m, tile_n, k_tile, simdgroups)` 生成。
pub(crate) fn tensor_pipe(
    device: &MetalDevice,
    queue: &MetalCommandQueue,
    stage: &TensorStage,
    (m, n): (usize, usize),
    geometry: TensorGeometry,
    msl: &str,
) -> Pipe {
    let TensorGeometry {
        tile_m,
        tile_n,
        k_tile,
        simdgroups,
    } = geometry;
    let pipe = device.compile(msl).expect("张量核 MSL 应能编译");
    let grid = [
        ((n / tile_n) * 32) as u32,
        ((m / tile_m) * simdgroups) as u32,
        1,
    ];
    let block = [32, simdgroups as u32, 1];
    assert!(
        queue.execute(&pipe, &[&stage.a, &stage.bt, &stage.y], grid, block),
        "GEMM 派发应成功"
    );
    Pipe {
        label: format!("{tile_m}·{tile_n}·{k_tile}·{simdgroups}"),
        pipe,
        grid,
        block,
    }
}

/// 同一批管线连跑 3 轮、每档取中位：频率漂移靠「整轮重扫」消除，而不是打散单档采样。
pub(crate) fn sweep(
    queue: &MetalCommandQueue,
    stage: &TensorStage,
    pipes: &[Pipe],
    (m, n, k): (usize, usize, usize),
    tag: &str,
) {
    let flops = 2.0 * (m * n * k) as f64;
    let mut passes = vec![Vec::new(); pipes.len()];
    for _ in 0..3 {
        for (slot, pipe) in passes.iter_mut().zip(pipes) {
            slot.push(measure(
                queue,
                &pipe.pipe,
                &[&stage.a, &stage.bt, &stage.y],
                pipe.grid,
                pipe.block,
            ));
        }
    }
    for (slot, Pipe { label, .. }) in passes.iter_mut().zip(pipes) {
        slot.sort_by(f64::total_cmp);
        let ns = slot[slot.len() / 2];
        println!(
            "{tag} {label}: {:.3} ms → {:.0} GF/s（最差 {:.0}）",
            ns / 1e6,
            flops / ns,
            flops / slot[slot.len() - 1]
        );
    }
}

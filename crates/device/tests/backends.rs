use super::*;
use crate::toolchain::msvc_version;
use cutile::backend::{Codegen, Driver, Source};
use cutile::dirs::Dirs;
use cutile::prof::Profile;
use cutile::witness::{MAX_OUT, Scope};
use std::path::Path;

/// 装配 Metal / CUDA 后端（幂等，进程级只装一次）。引擎测试直接调 `MetalBackend`/`CudaBackend` 的
/// `Codegen`/`Driver` 方法；经 `cutile::witness::device_available` 判断真机在场时，先注入注册表。
fn ensure_backends() {
    use std::sync::OnceLock;
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(install);
}

fn metal_profile() -> Profile {
    cutile::builtin::Catalogue::try_builtin()
        .expect("内置数据")
        .profile("metal")
        .cloned()
        .expect("metal 档案")
}

fn ascend_profile() -> Profile {
    cutile::builtin::Catalogue::try_builtin()
        .expect("内置数据")
        .profile("ascend")
        .cloned()
        .expect("ascend 档案")
}

// ---------------------------------------------------------------------------
// 厂商工具链（MSVC cl.exe）：从 cutile 搬来，`find_cl`/`msvc_version` 只服务 Windows 上的 nvcc。
// ---------------------------------------------------------------------------

#[test]
fn paths_without_an_msvc_segment_parse_as_empty() {
    let mut rng = testkit::Rng::new(0xC12);
    for _ in 0..32 {
        let path = format!(r"C:\Tools\{}\bin\cl.exe", rng.below(9999));
        assert!(msvc_version(Path::new(&path)).is_empty());
    }
}

#[test]
fn the_newest_msvc_wins_by_version_not_by_string_order() {
    let candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\9.29.30133\bin\Hostx64\x64\cl.exe",
        r"D:\BuildTools\VC\Tools\MSVC\17.10.0\bin\Hostx64\x64\cl.exe",
    ];
    let newest = candidates
        .into_iter()
        .max_by_key(|cl| msvc_version(Path::new(cl)))
        .expect("候选非空");
    assert!(
        newest.contains("17.10.0"),
        "字符串序会把 9.29 排到 17.10 之后：{newest}"
    );
}

#[test]
fn a_path_without_a_version_loses_to_one_with_it() {
    assert!(msvc_version(Path::new(r"C:\msvc\cl.exe")).is_empty());
    assert_eq!(
        msvc_version(Path::new(
            r"C:\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"
        )),
        vec![14, 29, 30133]
    );
}

// ---------------------------------------------------------------------------
// Metal 见证驱动：宽度取宏、非空。
// ---------------------------------------------------------------------------

#[test]
fn the_metal_driver_takes_its_width_from_the_witness_limit() {
    let driver = metal_driver(MAX_OUT);
    assert!(driver.contains(&format!("#define WITNESS_MAX_OUT {MAX_OUT}")));
    assert_eq!(
        driver.matches("WITNESS_MAX_OUT").count(),
        5,
        "驱动里还有写死的宽度"
    );
    for literal in ["count * 8", "out[i * 8", "(n > 8)"] {
        assert!(!driver.contains(literal), "仍是字面量：{literal}");
    }
}

#[test]
fn metal_driver_wrapper_is_nonempty() {
    let driver = metal_driver(MAX_OUT);
    assert!(!driver.is_empty());
}

#[test]
fn device_benchmark_drivers_measure_gpu_time_and_emit_output() {
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置算子");
    let op = catalogue.op("gemv_f32").expect("gemv_f32 声明");
    let model = op::model("gemv_f32").expect("gemv_f32 模型");
    let sample = (model.sample)(2, 4);
    op.signature.check(&sample).expect("样本与签名一致");

    let i4 = catalogue.op("gemv_i4").expect("gemv_i4 声明");
    let cuda_kernel = crate::cuda::cuda_quantized_gemv_i4(&i4.signature, 32).expect("CUDA 内核");
    assert!(cuda_kernel.contains("__shfl_down_sync"));

    let cuda_args =
        crate::dialect::launch_args(&sample, op.signature.params()).expect("CUDA 启动实参");
    let cuda = crate::cuda::cuda_benchmark_driver(
        "__global__ void kernel_main() {}",
        &op.signature,
        &sample,
        [2, 1, 1],
        [1, 1, 1],
        &cuda_args,
        3,
        false,
    )
    .expect("CUDA 驱动");
    assert!(cuda.contains("cudaEventRecord(start)"));
    assert!(cuda.contains("cudaEventElapsedTime"));
    assert!(cuda.contains("for (int sample = 0; sample < 3; ++sample)"));
    assert!(cuda.contains("ns %.3f"));
    let musa = crate::cuda::cuda_benchmark_driver(
        "__global__ void kernel_main() {}",
        &op.signature,
        &sample,
        [2, 1, 1],
        [1, 1, 1],
        &cuda_args,
        3,
        true,
    )
    .expect("MUSA 驱动");
    assert!(musa.contains("#define cudaEvent_t musaEvent_t"));
    assert!(musa.contains("#define cudaEventRecord musaEventRecord"));

    let metal =
        crate::metal::metal_benchmark_driver(&op.signature, &sample, [2, 1, 1], [1, 1, 1], 3)
            .expect("Metal 驱动");
    assert!(metal.contains("GPUStartTime"));
    assert!(metal.contains("GPUEndTime"));
    assert!(metal.contains("dispatchThreads:"));
    assert!(metal.contains("ns %.3f"));
}

// ---------------------------------------------------------------------------
// Metal 见证真机 + 前端签名（从 cutile/witness 搬来）。
// ---------------------------------------------------------------------------

#[test]
fn every_metal_witness_runs_on_the_gpu() {
    ensure_backends();
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 设备不在本机，GPU 见证用例未运行");
        return;
    }
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-witness"));
    let runs = cutile::witness::run_all(&profile, &dirs).expect("Metal 见证必须全部通过");
    assert_eq!(runs.len(), profile.provides.len(), "每条声明都要有结果");
    assert!(
        runs.iter().all(|run| run.scope == Scope::Device),
        "GPU 上跑出来的结论不能被降级成自检"
    );
    assert!(runs.iter().all(|run| run.max_rel_err <= 1e-5), "{runs:?}");
}

#[test]
fn the_metal_front_end_demands_an_explicit_address_space() {
    ensure_backends();
    if !cutile::witness::device_available(&metal_profile()) {
        testkit::skip("engine: Metal 设备不在本机，MSL 前端用例未运行");
        return;
    }
    // 必须带 `kernel`：不带限定符的函数在 MSL 里是普通 C 函数，`xcrun metal` 会照收不误、
    // 只是不会生成入口点，于是失败被推迟到执行期（"metallib 里没有 cutile_witness"），
    // 那条路径测不到"显式地址空间"这条不变量。
    let program = "#include <metal_stdlib>\nusing namespace metal;\n\
               kernel void cutile_witness(float *out) { out[0] = 1.0f; }\n";
    let profile = metal_profile();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-frontend"));
    let error = MetalBackend
        .run(&profile, &dirs, program, 1, Scope::Device)
        .expect_err("MSL 必须要求显式地址空间");
    let text = error.to_string();
    assert!(text.contains("xcrun metal"), "{text}");
    assert!(text.contains("address space"), "{text}");
}

// ---------------------------------------------------------------------------
// 设备生成文本的花括号配对：Metal / CUDA 两套 Codegen 都要守住。
// ---------------------------------------------------------------------------

#[test]
fn generated_device_program_braces_are_balanced() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-brace-work"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    for (vendor, front) in [
        ("nvidia", &CUDA_BACKEND as &dyn Codegen),
        ("metal", &METAL_BACKEND as &dyn Codegen),
        ("ascend", &ASCEND_BACKEND as &dyn Codegen),
    ] {
        let Some(profile) = catalogue.profile(vendor).cloned() else {
            continue;
        };
        let selected = cutile::witness::declared_witnesses(&profile).expect("见证");
        let text = front
            .program(&profile, &dirs, &selected, Scope::Device)
            .expect("可生成");
        let open = text.chars().filter(|c| *c == '{').count();
        let close = text.chars().filter(|c| *c == '}').count();
        assert_eq!(
            open, close,
            "{vendor} 生成文本花括号不配对：{open} != {close}"
        );
    }
}

// ---------------------------------------------------------------------------
// 昇腾后端：AscendC 单源文本的形状，以及“缺编译器就如实说缺谁”。
// ---------------------------------------------------------------------------

/// 昇腾的见证程序必须是一份**能被 ccec 处理的 AscendC 翻译单元**：缺 `__aicore__` 的设备函数、
/// 缺 `GM_ADDR` / `__gm__` 的显存指针、或缺 AscendCL 的 host 驱动，产出就不属于任何设备，
/// 只是把字段 `emit = "ascendc"` 贴在了件普通 C++ 上。
#[test]
fn an_ascend_program_is_a_single_ascendc_translation_unit() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-ascend-program"));
    let profile = ascend_profile();
    let selected = cutile::witness::declared_witnesses(&profile).expect("见证");
    let text = ASCEND_BACKEND
        .program(&profile, &dirs, &selected, Scope::Device)
        .expect("可生成");

    // 单源：host 驱动与核函数在同一份文本里，`<<<>>>` 由 ccec 拆成两段。
    assert!(
        text.contains("extern \"C\" __global__ __aicore__ void cutile_witness("),
        "核函数入口缺 AscendC 前缀：{text}"
    );
    assert!(text.contains("GM_ADDR"), "核函数入口没有 GM 指针：{text}");
    assert!(
        text.contains("__gm__ float *out"),
        "返回值缓冲区不是全局内存：{text}"
    );
    assert!(
        text.contains("AscendC::GetBlockIdx()"),
        "核索引不是 AscendC 的读法：{text}"
    );
    assert!(
        text.contains("#include \"acl/acl.h\""),
        "host 侧没有 AscendCL：{text}"
    );
    assert!(
        text.contains("aclrtMalloc") && text.contains("aclrtMemcpy"),
        "host 驱动没有分配 / 回读：{text}"
    );
    assert!(
        text.contains("<<<total, nullptr, stream>>>"),
        "发射不是单源的 `<<<>>>`：{text}"
    );
    // 档案 preamble 与见证在同一份翻译单元里：`@reduce` 的 `$func` 就指到这里。
    assert!(
        text.contains("_cutile_sum_f32"),
        "preamble 没进翻译单元：{text}"
    );
    // 一条见证一个设备函数：分派调到的每个函数都要真的带 `__aicore__`，否则它只是宿主函数。
    assert_eq!(
        text.matches("n = witness_").count(),
        selected.len(),
        "分派条数与见证条数不一致"
    );
    for (_, _, witness) in &selected {
        assert!(
            text.contains(&format!("__aicore__ int {}(", witness.func)),
            "`{}` 没有改成设备函数",
            witness.func
        );
    }
}

/// 昇腾可用性是两个独立事实的合取：编译器在、设备也在。合成一个布尔就丢掉了唯一的行动信息
/// （装 CANN 还是换机器）；把缺的那个报成另一个也是假绿——`ascend` 的等级仍因此恒不可用。
#[test]
fn the_ascend_backend_needs_both_the_compiler_and_the_device() {
    ensure_backends();
    let profile = ascend_profile();
    let ready = detect_cce().is_some() && ascend_device();
    assert_eq!(ASCEND_BACKEND.available(&profile), ready);
    assert_eq!(
        cutile::witness::device_available(&profile),
        ready,
        "注册表里的可用性要与后端自己的判定一致"
    );
    assert!(
        ASCEND_BACKEND
            .toolchain()
            .is_none_or(|tool| tool.command == "ccec"),
        "缓存的指纹只能是档案声明的 `ccec`"
    );
    if !ready {
        // 注册是进程级的一张表：没注册时 `device::spec(ascendc)` 是 `None`，报的是"本机没有该设备"
        // 并列出已接的前端。注册之后同一台机器仍不可用，只可能因为它自己判定的那个理由。
        let dirs = Dirs::under(std::env::temp_dir().join("engine-ascend-registry"));
        let error = cutile::witness::run_all(&profile, &dirs).expect_err("本机没有 ccec / NPU");
        let text = error.to_string();
        assert!(
            !text.contains("本机没有该设备"),
            "AscendBackend 没注册进 cutile 的后端表：{text}"
        );
        assert!(text.contains("ccec"), "要报缺哪个程序：{text}");
    }
}

/// 本机没有 `ccec` 时，报的必须是「编译器不在」而不是一条 C++ 语法错误：那是 `nvcc` 那一类
/// “把缺工具链伪装成编译失败”的旧病。
#[test]
fn an_ascend_run_without_ccec_names_the_missing_compiler() {
    ensure_backends();
    if detect_cce().is_some() {
        testkit::skip("engine: 本机装了 `ccec`，缺席分支无从构造");
        return;
    }
    let dirs = Dirs::under(std::env::temp_dir().join("engine-ascend-missing"));
    let error = ASCEND_BACKEND
        .run(
            &ascend_profile(),
            &dirs,
            "int main() { return 0; }",
            1,
            Scope::Device,
        )
        .expect_err("缺 ccec 就编不出来");
    let text = error.to_string();
    assert!(text.contains("ccec"), "要报缺哪个程序：{text}");
}

/// 语义门的产物也必须是 AscendC：`verify` 的 `L2` 就靠它把"算得对不对"钉在设备上。见证只证明
/// 片段跑得起来，这里证明**内核**在设备上被调用一次并回读。
///
/// 内核源不需要真装配也能断言这件事：装配器（`build::source`）要一条协商出来的 `Path`，而 `ascend`
/// 现在全 `L0`、协商给不出路径（也正因为如此这条链才需要先升等级）。所以这里按装配后的**形状**
/// 手写一段：真档案 preamble + 真签名 + 一段调用 `$func` 的内核体。机制层要证的就是这个形状——
/// `@prim` 换完之后剩下的东西是否仍然是一个可编译的 AscendC 翻译单元。
#[test]
fn an_ascend_harness_is_a_single_ascendc_translation_unit() {
    ensure_backends();
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = ascend_profile();
    let decl = catalogue.op("gemv_i4").expect("gemv_i4 声明").clone();
    let model = op::model("gemv_i4").expect("gemv_i4 语义表");
    let (m, n) = model.shape;
    let sample = (model.sample)(m, n);
    // 装配后的内核体：`@load` / `@dot` / `@reduce` 都已换成片段，只剩 C。
    let body = "typedef unsigned char u8;\n\
         \n\
         void cutile_gemv_i4_kernel(const float *x, const u8 *w, const float *scales,\n\
         \x20                          float *y, int M, int N, int gsz) {\n\
         \x20   for (int n = 0; n < N; ++n) {\n\
         \x20       const u8 *row = w + (long)n * (M / 2);\n\
         \x20       float acc = _cutile_sum_f32(x, M / gsz, 1);\n\
         \x20       y[n] = acc * (float)(row[0] & 0x0F) * scales[n];\n\
         \x20   }\n\
         }\n";
    let kernel_source = format!("{}\n{}", profile.preamble, body);
    let program = match ASCEND_BACKEND
        .harness(&kernel_source, &decl.signature, &sample)
        .expect("语义门应能拼出单源 AscendC")
    {
        Source::Single(text) => text,
        Source::Pair { .. } => panic!("昇腾是单源方言，host 驱动与核函数必须同在一份翻译单元里"),
    };

    // 入口：`extern "C" __global__ __aicore__` + `GM_ADDR` 形参（形参名跟签名走）。
    assert!(
        program.contains("extern \"C\" __global__ __aicore__ void cutile_launch(GM_ADDR d_"),
        "设备入口不是 AscendC 核函数：{program}"
    );
    // 内核本体与 preamble 里的软件实现同样要落在 AI Core 上。
    assert!(
        program.contains(&format!("__aicore__ void {}(", decl.signature.kernel())),
        "内核 `{}` 没转成设备函数：{program}",
        decl.signature.kernel()
    );
    assert!(
        program.contains("static __aicore__ inline float _cutile_sum_f32("),
        "preamble 的 `$func` 落点没转成设备函数：{program}"
    );
    // 只改函数，别的一概不动：`typedef` 与 `static const` 不是函数，给它们加 `__aicore__`
    // 会把本来能编的源码改坏（也是 CUDA 那边同一条边界）。
    assert!(
        program.contains("typedef unsigned char u8;"),
        "`typedef` 被改动了：{program}"
    );
    assert!(
        program.contains("static const float _cutile_exp_log2e"),
        "编译期常量被改动了：{program}"
    );
    // 显存指针：`GM_ADDR` 只是 `__gm__ uint8_t *`，交给 P0 内核前要补回元素类型。
    assert!(
        program.contains("(__gm__ const float *)d_"),
        "只读缓冲没有补回 `__gm__` 类型：{program}"
    );
    // 发射与回读：单块 `<<<1, nullptr, stream>>>`（与 CUDA 的 `<<<1,1>>>` 同位），AscendCL 回读。
    assert!(
        program.contains("cutile_launch<<<1, nullptr, stream>>>("),
        "发射不是单源 `<<<>>>`：{program}"
    );
    assert!(
        program.contains("ACL_MEMCPY_HOST_TO_DEVICE")
            && program.contains("ACL_MEMCPY_DEVICE_TO_HOST"),
        "缓冲没有入向 / 回读：{program}"
    );
    // 回读口径必须与 CUDA 逐字一致：位模式逐行打印，共用同一份 `driver::readback`。
    assert!(
        program.contains("printf(\"%u\\n\", v.u)"),
        "回读不是位模式：{program}"
    );
}

// ---------------------------------------------------------------------------
// 生产态 Metal 网格内核编译：把一条真实算子走完整条流水线，再经 `xcrun metal` 编译证明合法。
// ---------------------------------------------------------------------------

#[test]
fn production_grid_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-grid"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，生产态内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("silu_f32").expect("silu_f32 语义表").shape;
    let msl = cutile::execution::program(&catalogue, &dirs, "silu_f32", cutile::execution::Backend::Metal, Mode::loop_injected(), m, n).and_then(|stages| stages.one("网格"))
        .expect("生产态网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert_metal_compiles("engine-metal-grid", &msl, &profile.flags);
}

#[test]
fn production_reduce_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-reduce"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，归约内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("reduce_sum_f32")
        .expect("reduce_sum_f32 语义表")
        .shape;
    let msl = cutile::execution::program(&catalogue, &dirs, "reduce_sum_f32", cutile::execution::Backend::Metal, Mode::loop_injected(), m, n).and_then(|stages| stages.one("网格"))
        .expect("归约网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(msl.contains("thread_position_in_grid"), "{msl}");
    assert!(msl.contains("_cutile_sum_f32"), "{msl}");
    assert_metal_compiles("engine-metal-reduce", &msl, &profile.flags);
}

#[test]
fn production_scan_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-scan"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，扫描内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("scan_f32").expect("scan_f32 语义表").shape;
    let (local, correct) =
        cutile::execution::program(&catalogue, &dirs, "scan_f32", cutile::execution::Backend::Metal, Mode::segmented(), m, n).and_then(|stages| stages.two("扫描"))
            .expect("分块扫描内核应能生成");
    assert!(local.contains("kernel void kernel_scan_local"), "{local}");
    assert!(local.contains("thread_position_in_grid"), "{local}");
    assert!(local.contains("start + len"), "{local}");
    assert!(
        correct.contains("kernel void kernel_scan_correct"),
        "{correct}"
    );
    assert!(correct.contains("d_c"), "{correct}");
    assert_metal_compiles("engine-metal-scan", &local, &profile.flags);
    assert_metal_compiles("engine-metal-scan", &correct, &profile.flags);
}

#[test]
fn production_rmsnorm_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-rmsnorm"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，两趟内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("rmsnorm_f32").expect("rmsnorm_f32 语义表").shape;
    let msl = cutile::execution::program(&catalogue, &dirs, "rmsnorm_f32", cutile::execution::Backend::Metal, Mode::loop_injected(), m, n).and_then(|stages| stages.one("网格"))
        .expect("两趟内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(!msl.contains("thread_position_in_grid"), "{msl}");
    let (partial, map) =
        cutile::execution::program(&catalogue, &dirs, "rmsnorm_f32", cutile::execution::Backend::Metal, Mode::segmented(), m, n).and_then(|stages| stages.two("归一化"))
            .expect("分块 rmsnorm 内核应能生成");
    assert!(partial.contains("_cutile_sumsq_f32"), "{partial}");
    assert!(
        map.contains("d_rms") && map.contains("thread_position_in_grid"),
        "{map}"
    );
    assert_metal_compiles("engine-metal-rmsnorm", &msl, &profile.flags);
    assert_metal_compiles("engine-metal-rmsnorm", &partial, &profile.flags);
    assert_metal_compiles("engine-metal-rmsnorm", &map, &profile.flags);
}

#[test]
fn production_softmax_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-softmax"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，softmax 内核编译用例未运行");
        return;
    }
    let (m, n) = op::model("softmax_f32").expect("softmax_f32 语义表").shape;
    let (max, expsum, map) =
        cutile::execution::program(&catalogue, &dirs, "softmax_f32", cutile::execution::Backend::Metal, Mode::segmented(), m, n).and_then(|stages| stages.three("softmax"))
            .expect("分块 softmax 内核应能生成");
    assert!(max.contains("_cutile_max_f32"), "{max}");
    assert!(
        expsum.contains("_cutile_exp_sum_f32") && expsum.contains("d_max"),
        "{expsum}"
    );
    assert!(
        map.contains("_cutile_exp_norm_f32") && map.contains("d_sum"),
        "{map}"
    );
    assert_metal_compiles("engine-metal-softmax", &max, &profile.flags);
    assert_metal_compiles("engine-metal-softmax", &expsum, &profile.flags);
    assert_metal_compiles("engine-metal-softmax", &map, &profile.flags);
}

#[test]
fn production_gemv_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-gemv"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，gemv 内核编译用例未运行");
        return;
    }
    let msl = cutile::execution::program(&catalogue, &dirs, "gemv_f32", cutile::execution::Backend::Metal, Mode::loop_injected(), 64, 4).and_then(|stages| stages.one("网格"))
        .expect("gemv 网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(msl.contains("thread_position_in_grid"), "{msl}");
    assert!(
        msl.contains("for (int n = (int)tid.x; n < (int)tid.x + 1; ++n)"),
        "{msl}"
    );
    assert_metal_compiles("engine-metal-gemv", &msl, &profile.flags);
}

#[test]
fn production_gemm_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-gemm"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，gemm 内核编译用例未运行");
        return;
    }
    let msl = cutile::execution::program(&catalogue, &dirs, "gemm_f32", cutile::execution::Backend::Metal, Mode::loop_injected(), 16, 16).and_then(|stages| stages.one("网格"))
        .expect("gemm 网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        msl.contains("uint3 tid [[thread_position_in_grid]]"),
        "{msl}"
    );
    assert!(
        msl.contains("for (int i = (int)tid.x; i < (int)tid.x + 1; ++i)"),
        "{msl}"
    );
    assert!(
        msl.contains("for (int j = (int)tid.y; j < (int)tid.y + 1; ++j)"),
        "{msl}"
    );
    assert_metal_compiles("engine-metal-gemm", &msl, &profile.flags);
}

#[test]
fn production_gemv_i2_kernel_compiles_for_a_real_op() {
    production_quantized_gemv_compiles("gemv_i2");
}

#[test]
fn production_gemv_i4_kernel_compiles_for_a_real_op() {
    production_quantized_gemv_compiles("gemv_i4");
}

#[test]
fn production_gemv_i8_kernel_compiles_for_a_real_op() {
    production_quantized_gemv_compiles("gemv_i8");
}

#[test]
fn production_attention_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-attention"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，attention 内核编译用例未运行");
        return;
    }
    let msl = cutile::execution::program(&catalogue, &dirs, "attention_f32", cutile::execution::Backend::Metal, Mode::loop_injected(), 8, 8).and_then(|stages| stages.one("网格"))
        .expect("attention 网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        msl.contains("for (int i = (int)tid.x; i < (int)tid.x + 1; ++i)"),
        "{msl}"
    );
    assert_metal_compiles("engine-metal-attention", &msl, &profile.flags);
}

#[test]
fn production_flash_attention_kernel_compiles_for_a_real_op() {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-flash-attention"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip("engine: Metal 工具链不在本机，flash_attention 内核编译用例未运行");
        return;
    }
    let msl = cutile::execution::program(&catalogue, &dirs, "flash_attention_f32", cutile::execution::Backend::Metal, Mode::loop_injected(), 8, 8).and_then(|stages| stages.one("网格"))
        .expect("flash_attention 网格内核应能生成");
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        msl.contains("for (int i = (int)tid.x; i < (int)tid.x + 1; ++i)"),
        "{msl}"
    );
    assert_metal_compiles("engine-metal-flash-attention", &msl, &profile.flags);
}

fn production_quantized_gemv_compiles(op_name: &str) {
    ensure_backends();
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-gemv-quant"));
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置数据");
    let profile = metal_profile();
    if !cutile::witness::device_available(&profile) {
        testkit::skip(format!(
            "engine: Metal 工具链不在本机，{op_name} 内核编译用例未运行"
        ));
        return;
    }
    let msl = cutile::execution::program(&catalogue, &dirs, op_name, cutile::execution::Backend::Metal, Mode::loop_injected(), 64, 4).and_then(|stages| stages.one("网格"))
        .unwrap_or_else(|error| panic!("{op_name} 网格内核应能生成：{error}"));
    assert!(msl.contains("kernel_main"), "{msl}");
    assert!(
        msl.contains("for (int n = (int)tid.x; n < (int)tid.x + 1; ++n)"),
        "{msl}"
    );
    assert_metal_compiles(&format!("engine-metal-{op_name}"), &msl, &profile.flags);
}

/// 写 MSL 进一次性目录、经 [`metal_kernel`] 编成 metallib、删目录；编译失败即 panic。
fn assert_metal_compiles(stem: &str, msl: &str, flags: &[String]) {
    let dir = std::env::temp_dir().join(format!("{stem}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("目录");
    let compiled = metal_kernel(&dir, msl, flags);
    let _ = std::fs::remove_dir_all(&dir);
    compiled.unwrap_or_else(|error| panic!("MSL 应能通过 xcrun metal 编译：{error}"));
}

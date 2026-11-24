use super::*;
use crate::backend::{Codegen, Source};
use crate::platform::msvc_version;
use crate::witness::{MAX_OUT, Scope};
use cutile::dirs::Dirs;
use cutile::prof::Profile;
use std::path::Path;

/// 装配 Metal / CUDA 后端（幂等，进程级只装一次）。引擎测试直接调 `MetalBackend`/`CudaBackend` 的
/// `Codegen`/`Driver` 方法；经 `crate::witness::device_available` 判断真机在场时，先注入注册表。
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
    let model = nn::registry::model("gemv_f32").expect("gemv_f32 模型");
    let sample = (model.sample)(2, 4);
    op.signature.check_sample(&sample).expect("样本与签名一致");

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
    if !crate::witness::device_available(&profile) {
        testkit::skip("engine: Metal 设备不在本机，GPU 见证用例未运行");
        return;
    }
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-witness"));
    let runs = crate::witness::run_all(&profile, &dirs).expect("Metal 见证必须全部通过");
    assert_eq!(runs.len(), profile.provides.len(), "每条声明都要有结果");
    assert!(
        runs.iter().all(|run| run.scope == Scope::Device),
        "GPU 上跑出来的结论不能被降级成自检"
    );
    assert!(runs.iter().all(|run| run.max_rel_err <= 1e-5), "{runs:?}");
}

fn nvidia_profile() -> Profile {
    cutile::builtin::Catalogue::try_builtin()
        .expect("内置数据")
        .profile("nvidia")
        .cloned()
        .expect("nvidia 档案")
}

#[test]
fn every_nvidia_witness_runs_on_the_cuda_device() {
    ensure_backends();
    let profile = nvidia_profile();
    if !crate::witness::device_available(&profile) {
        testkit::skip("engine: CUDA/nvcc 工具链或 MSVC 宿主不在本机，GPU 见证用例未运行");
        return;
    }
    let dirs = Dirs::under(std::env::temp_dir().join("engine-cuda-witness"));
    let runs = crate::witness::run_all(&profile, &dirs).expect("CUDA 见证必须全部通过");
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
    if !crate::witness::device_available(&metal_profile()) {
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
        let selected = crate::witness::declared_witnesses(&profile).expect("见证");
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
    let selected = crate::witness::declared_witnesses(&profile).expect("见证");
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
        crate::witness::device_available(&profile),
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
        let error = crate::witness::run_all(&profile, &dirs).expect_err("本机没有 ccec / NPU");
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
    let model = nn::registry::model("gemv_i4").expect("gemv_i4 语义表");
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
// 生产态 Metal 网格内核编译（见 `backends_production.rs`）。
// ---------------------------------------------------------------------------

#[path = "backends_production.rs"]
mod production;

/// `@` 在数据模板里是占位符前导；渲染后仍留下裸 `@` 说明某个占位漏填，而 Objective-C 只认
/// `@autoreleasepool` 与 `@"..."`——这类残留在编译器眼里是语法错，在这里先拦下。
fn stray_placeholder(text: &str) -> Option<&str> {
    text.lines().find(|line| {
        let mut rest = *line;
        while let Some(at) = rest.find('@') {
            let tail = &rest[at + 1..];
            if !(tail.starts_with("autoreleasepool") || tail.starts_with('"')) {
                return true;
            }
            rest = tail;
        }
        false
    })
}

#[test]
fn metal_host_programs_leave_no_template_placeholders() {
    let catalogue = cutile::builtin::Catalogue::try_builtin().expect("内置算子");
    let op = catalogue.op("gemv_f32").expect("gemv_f32 声明");
    let model = nn::registry::model("gemv_f32").expect("gemv_f32 模型");
    let sample = (model.sample)(2, 4);
    let (_, harness) = crate::metal::metal_harness("", &op.signature, &sample).expect("语义门");
    let dispatch = crate::metal::metal_dispatch_driver(
        &op.signature,
        &sample,
        "kernel_main",
        [2, 1, 1],
        [1, 1, 1],
    )
    .expect("派发");
    let bench =
        crate::metal::metal_benchmark_driver(&op.signature, &sample, [2, 1, 1], [1, 1, 1], 3)
            .expect("基准");
    let rop = catalogue.op("rmsnorm_f32").expect("rmsnorm 声明");
    let rmodel = nn::registry::model("rmsnorm_f32").expect("rmsnorm 模型");
    let rsample = (rmodel.sample)(2, 8);
    let Source::Pair { host, .. } =
        crate::metal::rmsnorm("", &rop.signature, &rsample, 4).expect("多段")
    else {
        panic!("多段驱动应是 Pair");
    };
    for (name, program) in [
        ("语义门", &harness),
        ("派发", &dispatch),
        ("基准", &bench),
        ("多段", &host),
    ] {
        assert_eq!(
            stray_placeholder(program),
            None,
            "{name} host 程序有未填的模板占位：{program}"
        );
        assert_eq!(
            program.matches("int main(").count(),
            1,
            "{name} 应只有一个 main"
        );
    }
}

/// 模板填空的契约：单趟扫描、相邻占位合法、未知占位原样保留（Objective-C 的 `@"..."` 靠这条）。
#[test]
fn template_fill_scans_once_and_keeps_unknown_placeholders() {
    let fill = crate::dialect::fill;
    assert_eq!(
        fill("@a@b|@c.@", &[("a", "1"), ("b", "2"), ("c", "3")]),
        "12|3.@"
    );
    assert_eq!(fill("@a", &[("a", "@b"), ("b", "X")]), "@b");
    assert_eq!(fill("@\"x\"", &[]), "@\"x\"");
}

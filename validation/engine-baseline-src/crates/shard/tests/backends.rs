use super::*;
use crate::backend::Codegen;
use crate::fixtures::profile;
use crate::platform::msvc_version;
use crate::witness::{MAX_OUT, Scope};
use cutile::dirs::Dirs;
use std::path::Path;

/// 装配 Metal / CUDA 后端（幂等，进程级只装一次）。引擎测试直接调 `MetalBackend`/`CudaBackend` 的
/// `Codegen`/`Driver` 方法；经 `crate::witness::device_available` 判断真机在场时，先注入注册表。
fn ensure_backends() {
    use std::sync::OnceLock;
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(install);
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

// ---------------------------------------------------------------------------
// Metal 见证真机 + 前端签名（从 cutile/witness 搬来）。
// ---------------------------------------------------------------------------

/// 两条真机路径共用同一组判据：逐条有结果、不得降级成自检、误差必须入预算。
fn check_device_witnesses(profile: &cutile::prof::Profile, dirs: &Dirs, label: &str) {
    let runs = crate::witness::run_all(profile, dirs)
        .unwrap_or_else(|error| panic!("{label} 见证必须全部通过：{error}"));
    assert_eq!(runs.len(), profile.provides.len(), "每条声明都要有结果");
    assert!(
        runs.iter().all(|run| run.scope == Scope::Device),
        "GPU 上跑出来的结论不能被降级成自检"
    );
    assert!(runs.iter().all(|run| run.max_rel_err <= 1e-5), "{runs:?}");
}

#[test]
fn every_metal_witness_runs_on_the_gpu() {
    ensure_backends();
    let profile = profile("metal");
    if !crate::witness::device_available(&profile) {
        testkit::skip("engine: Metal 设备不在本机，GPU 见证用例未运行");
        return;
    }
    let dirs = Dirs::under(std::env::temp_dir().join("engine-metal-witness"));
    check_device_witnesses(&profile, &dirs, "Metal");
}

#[test]
fn every_nvidia_witness_runs_on_the_cuda_device() {
    ensure_backends();
    let profile = profile("nvidia");
    if !crate::witness::device_available(&profile) {
        testkit::skip("engine: CUDA/nvcc 工具链或 MSVC 宿主不在本机，GPU 见证用例未运行");
        return;
    }
    let dirs = Dirs::under(std::env::temp_dir().join("engine-cuda-witness"));
    check_device_witnesses(&profile, &dirs, "CUDA");
}

#[test]
fn the_metal_front_end_demands_an_explicit_address_space() {
    ensure_backends();
    if !crate::witness::device_available(&profile("metal")) {
        testkit::skip("engine: Metal 设备不在本机，MSL 前端用例未运行");
        return;
    }
    // 必须带 `kernel`：不带限定符的函数在 MSL 里是普通 C 函数，`xcrun metal` 会照收不误、
    // 只是不会生成入口点，于是失败被推迟到执行期（"metallib 里没有 cutile_witness"），
    // 那条路径测不到"显式地址空间"这条不变量。
    let program = "#include <metal_stdlib>\nusing namespace metal;\n\
               kernel void cutile_witness(float *out) { out[0] = 1.0f; }\n";
    let profile = profile("metal");
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
    let catalogue = crate::fixtures::catalogue();
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
    let profile = profile("ascend");
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
    let profile = profile("ascend");
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
            &profile("ascend"),
            &dirs,
            "int main() { return 0; }",
            1,
            Scope::Device,
        )
        .expect_err("缺 ccec 就编不出来");
    let text = error.to_string();
    assert!(text.contains("ccec"), "要报缺哪个程序：{text}");
}

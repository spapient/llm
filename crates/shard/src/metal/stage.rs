//! Metal 多段并行驱动：一次装载 metallib，段间插入主机折叠，最后回读输出。

use cutile::harness::output_name;
use diagnostic::Result;
use prof::model::Sample;
use prof::sig::Signature;

use crate::metal::host::{IMPORTS_MATH, Wiring, host_library, host_program, output_alloc};

/// 一段内核：入口名、缓冲槽位名、网格、段后主机折叠文本。
pub(crate) struct MetalStage<'a> {
    pub entry: &'a str,
    pub bufs: &'a [&'a str],
    pub grid: [usize; 3],
    pub fold: &'a str,
}

/// 多段并行驱动：装载一次 metallib，按段依次发射，段间插入主机折叠，最后回读输出。
///
/// `capacity` 是**输出缓冲按元素开的长度**：分块族的部分结果先写进输出缓冲（`y[g]`），
/// 块数可能大于最终输出长度（归约是 1 个标量），因此分配与回读分开给。
/// `scratch` 是折叠用的中间缓冲（`rms` / `max` / `sum` / 携带态），一律 f32。
///
/// # Errors
///
/// 签名与样本不一致、或某段找不到入口函数时返回说明（入口函数缺失由 host 侧返回码 5 报出）。
pub(crate) fn metal_multistage_driver(
    signature: &Signature,
    sample: &Sample,
    capacity: usize,
    read: usize,
    scratch: &[(&str, usize)],
    stages: &[MetalStage<'_>],
) -> Result<String> {
    let output = output_name(signature)?;
    let wiring = Wiring::of(signature, sample, Some(capacity), true, 12)?;
    let mut allocs = wiring.allocs;
    for (name, len) in scratch {
        allocs.push_str(&output_alloc(name, "float", *len, true));
    }
    let body = format!(
        "        id<MTLCommandQueue> queue = [device newCommandQueue];\n{}",
        metal_stages(stages)
    );
    Ok(host_program(
        IMPORTS_MATH,
        &wiring.data,
        host_library(),
        &allocs,
        &body,
        output,
        read,
    ))
}

/// 段体：每段一份发射 + 该段自己的主机折叠（折叠串由调用方给，不在驱动里拼第二份算式）。
fn metal_stages(stages: &[MetalStage<'_>]) -> String {
    let mut body = String::new();
    for (idx, stage) in stages.iter().enumerate() {
        body.push_str(&metal_stage(idx, stage));
        body.push_str(stage.fold);
    }
    body
}

/// 单段发射：取入口函数、建管线、设缓冲、派发、提交并等待。
fn metal_stage(idx: usize, stage: &MetalStage<'_>) -> String {
    let mut sets = String::new();
    for (slot, name) in stage.bufs.iter().enumerate() {
        sets.push_str(&format!(
            "            [encoder{idx} setBuffer:b_{name} offset:0 atIndex:{slot}];\n"
        ));
    }
    format!(
        r#"        {{
            id<MTLFunction> function{idx} = [library newFunctionWithName:@"{entry}"];
            if (!function{idx}) {{ fprintf(stderr, "metallib 里没有 {entry}\n"); return 5; }}
            id<MTLComputePipelineState> pipeline{idx} = [device newComputePipelineStateWithFunction:function{idx} error:&error];
            if (!pipeline{idx}) {{ fprintf(stderr, "管线创建失败：%s\n", error.localizedDescription.UTF8String); return 6; }}
            id<MTLCommandBuffer> buffer{idx} = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder{idx} = [buffer{idx} computeCommandEncoder];
            [encoder{idx} setComputePipelineState:pipeline{idx}];
{sets}            [encoder{idx} dispatchThreads:MTLSizeMake({gx}, {gy}, {gz}) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [encoder{idx} endEncoding];
            [buffer{idx} commit];
            [buffer{idx} waitUntilCompleted];
            if (buffer{idx}.error) {{ fprintf(stderr, "内核执行失败：%s\n", buffer{idx}.error.localizedDescription.UTF8String); return 7; }}
        }}
"#,
        entry = stage.entry,
        sets = sets,
        gx = stage.grid[0],
        gy = stage.grid[1],
        gz = stage.grid[2],
    )
}

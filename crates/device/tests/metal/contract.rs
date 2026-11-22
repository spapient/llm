//! 职责：Metal 执行器契约冒烟——设备/队列/缓冲/管线/派发五步在真机成立。

use super::*;

/// 设备 → 命令队列 的进程内对象模型打通：能建、能释放，反复建放不崩溃、不重复释放。
#[test]
fn device_and_queue_round_trip_without_crashing() {
    for _ in 0..2 {
        let Some(device) = MetalDevice::system_default() else {
            // 无 Metal 的本机（如 CI Linux）跳过而非失败：本测试只证明"有设备时路径成立"。
            return;
        };
        let queue = device.new_command_queue();
        assert!(queue.is_some(), "有设备必有命令队列");
    }
}

/// 共享缓冲分配 + CPU 读写回读：写进去的字节原样读回，证明设备内存与 CPU 视图是同一块内存。
#[test]
fn shared_buffer_round_trip_reads_written_bytes() {
    let Some(device) = MetalDevice::system_default() else {
        return;
    };
    let Some(mut buf) = device.new_buffer(16) else {
        panic!("分配共享缓冲失败");
    };
    let expect: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    buf.as_mut_bytes().copy_from_slice(&expect);
    assert_eq!(buf.as_mut_bytes(), &expect);
}

/// 一段合法 MSL 源码在进程内编译出计算管线：不依赖 xcrun 子进程，直接 `newLibraryWithSource`。
#[test]
fn compiles_a_trivial_kernel_in_process() {
    let Some(device) = MetalDevice::system_default() else {
        return;
    };
    let source = r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = 42.0f;
}
"#;
    let pipeline = device.compile(source);
    assert!(pipeline.is_some(), "合法 MSL 应能编译出管线");
}

/// 派发一个写常量的内核并回读：4 个线程各写 `out[id] = 42.0`，`waitUntilCompleted` 后 CPU 读到 4 个 42.0。
#[test]
fn dispatches_a_kernel_and_reads_back() {
    let Some(device) = MetalDevice::system_default() else {
        return;
    };
    let source = r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    out[id] = 42.0f;
}
"#;
    let Some(pipeline) = device.compile(source) else {
        panic!("合法 MSL 应能编译出管线");
    };
    let Some(queue) = device.new_command_queue() else {
        panic!("有设备必有命令队列");
    };
    // 4 个 f32 = 16 字节。
    let mut out = device.new_buffer(16).expect("分配输出缓冲失败");
    assert!(
        queue.execute(&pipeline, &[&out], [4, 1, 1], [1, 1, 1]),
        "派发应成功"
    );

    let bytes = out.as_mut_bytes();
    assert_eq!(bytes.len(), 16);
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(got, vec![42.0f32; 4], "GPU 应把 42.0 写进每个元素");
}

/// 执行器把一张图的节点派发到 Metal 并回读：注册 `kernel_main` 双倍内核，绑定输入/输出，
/// `submit` → `sync` 后输出 = 输入的 2 倍，证明「提交 → 派发 → 同步」整条 [`Executor`] 契约在真机成立。
#[test]
fn executor_runs_a_registered_kernel_through_the_graph() {
    use engine::exec::Launch;

    let Some(mut exec) = MetalExecutor::new("metal", "apple") else {
        return;
    };
    exec.register(
        "double",
        r#"#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float* in [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2.0f;
}
"#,
    )
    .expect("登记内核失败");
    exec.bind_bytes(0, 16).expect("绑定输入缓冲失败");
    exec.bind_bytes(1, 16).expect("绑定输出缓冲失败");

    let input = [1.0f32, 2.0, 3.0, 4.0];
    let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_ne_bytes()).collect();
    exec.buffer_mut(0)
        .expect("输入缓冲应已绑定")
        .as_mut_bytes()
        .copy_from_slice(&input_bytes);

    let mut graph = Graph::new();
    let mut node = Node::new("double", Launch::new([4, 1, 1], [1, 1, 1]));
    node.bind_input(0).bind_output(1);
    graph.push(node);

    exec.submit(&graph, Stream::new(0)).expect("提交失败");
    exec.sync(Stream::new(0)).expect("同步失败");

    let (chunks, _) = exec
        .buffer_mut(1)
        .expect("输出缓冲应已绑定")
        .as_mut_bytes()
        .as_chunks::<4>();
    let output: Vec<f32> = chunks.iter().map(|c| f32::from_ne_bytes(*c)).collect();
    assert_eq!(output, vec![2.0f32, 4.0, 6.0, 8.0], "GPU 应把输入翻倍写回");
}

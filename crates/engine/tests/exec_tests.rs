use super::*;
use storage::CpuAllocator;

struct Counting {
    device: Device,
    submitted: usize,
}

impl Executor for Counting {
    fn device(&self) -> &Device {
        &self.device
    }

    fn submit(&mut self, graph: &Graph, _stream: Stream) -> Result<()> {
        self.submitted += graph.len();
        Ok(())
    }
}

#[test]
fn total_threads_multiplies_grid_and_block() {
    let launch = Launch::new([2, 3, 1], [4, 5, 1]);
    assert_eq!(launch.total_threads(), 120);
    assert_eq!(Launch::new([0, 1, 1], [1, 1, 1]).total_threads(), 0);
}

#[test]
fn a_graph_keeps_insertion_order() {
    let mut graph = Graph::new();
    assert!(graph.is_empty());
    graph.push(Node::new("a", Launch::new([1, 1, 1], [1, 1, 1])));
    graph.push(Node::new("b", Launch::new([1, 1, 1], [1, 1, 1])));
    assert_eq!(graph.len(), 2);
    let names: Vec<&str> = graph.nodes().iter().map(Node::name).collect();
    assert_eq!(names, ["a", "b"]);
}

#[test]
fn an_executor_only_needs_submit() {
    let mut graph = Graph::new();
    graph.push(Node::new("gemv_i4", Launch::new([8, 1, 1], [64, 1, 1])));
    let device = Device::new("cpu", "generic");
    let mut executor = Counting {
        device,
        submitted: 0,
    };
    executor.submit(&graph, Stream::new(0)).expect("提交");
    assert_eq!(executor.submitted, 1);
    assert_eq!(executor.device().vendor(), "cpu");
    executor.sync(Stream::new(0)).expect("CPU 同步是空操作");
}

fn graph(names: &[&str]) -> Graph {
    let mut graph = Graph::new();
    for name in names {
        graph.push(Node::new(*name, Launch::new([1, 1, 1], [1, 1, 1])));
    }
    graph
}

#[test]
fn local_executor_keeps_submission_order_within_a_stream() {
    let mut executor =
        LocalExecutor::new(Device::new("cpu", "generic"), Box::new(CpuAllocator::new()));
    executor
        .submit(&graph(&["a", "b"]), Stream::new(0))
        .expect("提交第一幅图");
    executor
        .submit(&graph(&["c"]), Stream::new(0))
        .expect("提交第二幅图");

    let names: Vec<&str> = executor
        .pending()
        .iter()
        .map(|queued| queued.node().name())
        .collect();
    assert_eq!(names, ["a", "b", "c"], "同流内必须按图序与节点序入队");
    assert_eq!(executor.len(), 3);
}

#[test]
fn local_sync_drains_only_the_named_stream() {
    let mut executor =
        LocalExecutor::new(Device::new("cpu", "generic"), Box::new(CpuAllocator::new()));
    executor
        .submit(&graph(&["x", "y"]), Stream::new(0))
        .expect("流 0 入队");
    executor
        .submit(&graph(&["z"]), Stream::new(1))
        .expect("流 1 入队");

    executor.sync(Stream::new(0)).expect("排干流 0");

    let remaining: Vec<&str> = executor
        .pending()
        .iter()
        .map(|queued| queued.node().name())
        .collect();
    assert_eq!(remaining, ["z"], "流 1 的提交不许被流 0 的同步带走");
    assert_eq!(executor.pending()[0].stream(), Stream::new(1));
}

#[test]
fn local_executor_empty_graph_submits_nothing() {
    let mut executor =
        LocalExecutor::new(Device::new("cpu", "generic"), Box::new(CpuAllocator::new()));
    executor
        .submit(&Graph::new(), Stream::new(0))
        .expect("空图提交");
    assert!(executor.is_empty(), "空图不应入队任何节点");
}

#[test]
fn local_executor_reports_bound_device() {
    let executor = LocalExecutor::new(Device::new("cpu", "aarch64"), Box::new(CpuAllocator::new()));
    assert_eq!(executor.device().vendor(), "cpu");
    assert_eq!(executor.device().arch(), "aarch64");
}

#[test]
fn local_executor_bounds_injected_allocator() {
    let mut executor =
        LocalExecutor::new(Device::new("cpu", "generic"), Box::new(CpuAllocator::new()));
    let buffer = executor.allocator_mut().alloc(16).expect("分配 16 字节");
    assert_eq!(buffer.len(), 16);
    executor.allocator_mut().free(buffer);

    let empty = executor.allocator_mut().alloc(0).expect("零字节分配");
    assert!(empty.is_empty());
}

#[test]
fn a_new_node_has_no_operands() {
    let node = Node::new("softmax_f32", Launch::new([1, 1, 1], [1, 1, 1]));
    assert!(node.inputs().is_empty());
    assert!(node.outputs().is_empty());
}

#[test]
fn node_binds_input_and_output_operands() {
    let mut node = Node::new("gemv_f32", Launch::new([1, 1, 1], [1, 1, 1]));
    node.bind_input(0).bind_input(1).bind_output(2);

    let inputs: Vec<u32> = node.inputs().iter().map(Operand::id).collect();
    let outputs: Vec<u32> = node.outputs().iter().map(Operand::id).collect();
    assert_eq!(inputs, [0, 1]);
    assert_eq!(outputs, [2]);
    assert_eq!(node.inputs()[0].role(), OperandRole::Input);
    assert_eq!(node.outputs()[0].role(), OperandRole::Output);
}

#[test]
fn queued_node_keeps_its_operand_bindings() {
    let mut graph = Graph::new();
    let mut node = Node::new("rope_f32", Launch::new([1, 1, 1], [1, 1, 1]));
    node.bind_input(3).bind_output(4);
    graph.push(node);

    let mut executor =
        LocalExecutor::new(Device::new("cpu", "generic"), Box::new(CpuAllocator::new()));
    executor.submit(&graph, Stream::new(0)).expect("提交");

    let queued = executor.pending();
    assert_eq!(queued[0].node().inputs()[0].id(), 3);
    assert_eq!(queued[0].node().outputs()[0].id(), 4);
}

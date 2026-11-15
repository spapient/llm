use super::*;

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
    let mut executor = LocalExecutor::new(Device::new("cpu", "generic"));
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
    let mut executor = LocalExecutor::new(Device::new("cpu", "generic"));
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
    let mut executor = LocalExecutor::new(Device::new("cpu", "generic"));
    executor
        .submit(&Graph::new(), Stream::new(0))
        .expect("空图提交");
    assert!(executor.is_empty(), "空图不应入队任何节点");
}

#[test]
fn local_executor_reports_bound_device() {
    let executor = LocalExecutor::new(Device::new("cpu", "aarch64"));
    assert_eq!(executor.device().vendor(), "cpu");
    assert_eq!(executor.device().arch(), "aarch64");
}

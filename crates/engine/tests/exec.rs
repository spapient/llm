use super::*;
use diagnostic::{Code, Error, Result};
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
fn scalar_slots_start_empty() {
    let launch = Launch::new([1, 1, 1], [1, 1, 1]);
    assert!(launch.scalars().is_empty());
    assert_eq!(launch.scalars().rope_position(), None);
}

#[test]
fn a_rope_position_round_trips_through_the_node() {
    let mut scalars = ScalarSlots::new();
    scalars.set_rope_position(7);
    let launch = Launch::new([1, 1, 1], [8, 1, 1]).with_scalars(scalars);
    let node = Node::new("rope_f32", launch);
    assert_eq!(node.scalars().rope_position(), Some(7));
    assert_eq!(node.launch().scalars().rope_position(), Some(7));

    let mut other = Node::new("rope_f32", Launch::new([1, 1, 1], [8, 1, 1]));
    other.scalars_mut().set_rope_position(12);
    assert_eq!(other.scalars().rope_position(), Some(12));
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

/// 装饰器叠在基座上：转发不改语义，提交序被测到，同步只排被装饰执行器的那条流。
#[test]
fn trace_decorates_any_executor_and_records_submissions() {
    let inner = LocalExecutor::new(Device::new("cpu", "generic"), Box::new(CpuAllocator::new()));
    let mut traced = Trace::new(inner);
    traced
        .submit(&graph(&["a", "b"]), Stream::new(0))
        .expect("提交第一幅图");
    traced
        .submit(&graph(&["c"]), Stream::new(1))
        .expect("提交第二幅图");

    assert_eq!(
        traced.submitted(),
        [vec!["a".to_owned(), "b".to_owned()], vec!["c".to_owned()],],
        "记录的是每次提交的节点名与顺序"
    );
    traced.sync(Stream::new(0)).expect("同步转发");
    assert_eq!(
        traced.inner().pending().len(),
        1,
        "流 1 的节点原样留在被装饰执行器里"
    );
}

#[test]
fn local_executor_bounds_injected_allocator() {
    let mut executor =
        LocalExecutor::new(Device::new("cpu", "generic"), Box::new(CpuAllocator::new()));
    let buf = executor.allocator_mut().alloc(16).expect("分配 16 字节");
    assert_eq!(buf.len(), 16);
    executor.allocator_mut().free(buf);

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

/// 最小内核：`out = in * factor`。它只认"一进一出"，因此操作数顺序错了会立刻报错。
struct Scale {
    factor: f32,
}

impl Program for Scale {
    fn name(&self) -> &str {
        "scale"
    }

    fn run(&self, _launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let (Some(input), Some(output)) = (inputs.first(), outputs.first_mut()) else {
            return Err(Error::new(Code::Invalid, "scale 需要一进一出").with("test::Scale"));
        };
        let input = input.f32().ok_or_else(|| {
            Error::new(Code::Invalid, "scale 输入不是浮点缓冲").with("test::Scale")
        })?;
        let output = output.f32_mut().ok_or_else(|| {
            Error::new(Code::Invalid, "scale 输出不是浮点缓冲").with("test::Scale")
        })?;
        if input.len() != output.len() {
            return Err(Error::new(Code::Invalid, "形状不一致").with("test::Scale"));
        }
        for (slot, value) in output.iter_mut().zip(input) {
            *slot = value * self.factor;
        }
        Ok(())
    }
}

/// 最小内核：`out = a + b`；两路输入都只读，输出可写。
struct Add;

impl Program for Add {
    fn name(&self) -> &str {
        "add"
    }

    fn run(&self, _launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> {
        let (Some(left), Some(right), Some(output)) =
            (inputs.first(), inputs.get(1), outputs.first_mut())
        else {
            return Err(Error::new(Code::Invalid, "add 需要两进一出").with("test::Add"));
        };
        let left = left
            .f32()
            .ok_or_else(|| Error::new(Code::Invalid, "add 左输入不是浮点缓冲").with("test::Add"))?;
        let right = right
            .f32()
            .ok_or_else(|| Error::new(Code::Invalid, "add 右输入不是浮点缓冲").with("test::Add"))?;
        let output = output
            .f32_mut()
            .ok_or_else(|| Error::new(Code::Invalid, "add 输出不是浮点缓冲").with("test::Add"))?;
        if left.len() != right.len() || left.len() != output.len() {
            return Err(Error::new(Code::Invalid, "形状不一致").with("test::Add"));
        }
        for ((slot, a), b) in output.iter_mut().zip(left).zip(right) {
            *slot = a + b;
        }
        Ok(())
    }
}

fn cpu() -> CpuExecutor {
    CpuExecutor::new(Device::new("cpu", "generic"))
}

#[test]
fn a_cpu_executor_writes_through_the_bound_operands() {
    let mut executor = cpu();
    executor
        .register(Box::new(Scale { factor: 2.0 }))
        .expect("登记");
    executor.bind(0, vec![1.0, 2.0, 3.0]);
    executor.bind(1, vec![0.0; 3]);

    let mut node = Node::new("scale", Launch::new([1, 1, 1], [3, 1, 1]));
    node.bind_input(0).bind_output(1);
    executor.run(&node).expect("执行");

    assert_eq!(executor.get(0).expect("输入还在"), [1.0, 2.0, 3.0]);
    assert_eq!(executor.get(1).expect("输出写回"), [2.0, 4.0, 6.0]);
}

/// 图序就是执行序：第二个节点读到的是第一个节点写过的值。
#[test]
fn a_graph_chains_through_the_same_bindings() {
    let mut executor = cpu();
    executor
        .register(Box::new(Scale { factor: 3.0 }))
        .expect("登记");
    executor.register(Box::new(Add)).expect("登记");
    executor.bind(0, vec![1.0, 1.0]);
    executor.bind(1, vec![0.0; 2]);
    executor.bind(2, vec![10.0, 20.0]);
    executor.bind(3, vec![0.0; 2]);

    let mut scaled = Node::new("scale", Launch::new([1, 1, 1], [2, 1, 1]));
    scaled.bind_input(0).bind_output(1);
    let mut summed = Node::new("add", Launch::new([1, 1, 1], [2, 1, 1]));
    summed.bind_input(1).bind_input(2).bind_output(3);
    let mut graph = Graph::new();
    graph.push(scaled);
    graph.push(summed);

    executor.submit(&graph, Stream::new(0)).expect("入队");
    assert_eq!(executor.len(), 2, "提交只入队：两条都还没跑");
    assert_eq!(
        executor.get(3).expect("输出缓冲"),
        [0.0, 0.0],
        "同步之前不写结果"
    );

    executor.sync(Stream::new(0)).expect("执行");
    assert!(executor.is_empty(), "同步后队列排空");
    assert_eq!(executor.get(3).expect("最后一步输出"), [13.0, 23.0]);
    assert_eq!(executor.bindings().len(), 4, "每个编号都放回了表里");
}

#[test]
fn an_unregistered_kernel_is_refused_by_name() {
    let mut executor = cpu();
    executor.bind(0, vec![1.0]);
    let mut node = Node::new("softmax_f32", Launch::new([1, 1, 1], [1, 1, 1]));
    node.bind_output(0);
    let error = executor.run(&node).unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("softmax_f32"), "{error}");
}

#[test]
fn a_second_registration_of_the_same_name_is_refused() {
    let mut executor = cpu();
    executor
        .register(Box::new(Scale { factor: 1.0 }))
        .expect("登记");
    let error = executor
        .register(Box::new(Scale { factor: 2.0 }))
        .unwrap_err();
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("已登记"), "{error}");
    assert_eq!(executor.bindings().len(), 0);
}

#[test]
fn an_unbound_operand_is_refused() {
    let mut executor = cpu();
    executor
        .register(Box::new(Scale { factor: 1.0 }))
        .expect("登记");
    executor.bind(0, vec![1.0]);
    let mut node = Node::new("scale", Launch::new([1, 1, 1], [1, 1, 1]));
    node.bind_input(0).bind_output(9);
    let error = executor.run(&node).unwrap_err();
    assert!(error.to_string().contains('9'), "{error}");
}

/// 一个编号不能同时是输入与输出：借出/放回会拿到两次同一块缓冲，语义上就已经错了。
#[test]
fn the_same_operand_cannot_be_both_input_and_output() {
    let mut executor = cpu();
    executor
        .register(Box::new(Scale { factor: 2.0 }))
        .expect("登记");
    executor.bind(0, vec![1.0]);
    let mut node = Node::new("scale", Launch::new([1, 1, 1], [1, 1, 1]));
    node.bind_input(0).bind_output(0);
    let error = executor.run(&node).unwrap_err();
    assert!(error.to_string().contains("既是输入又是输出"), "{error}");
}

/// 内核失败时缓冲必须放回：否则下一次执行会看到"操作数未绑定"，把真正的错误盖掉。
#[test]
fn a_failing_kernel_still_returns_its_bufs() {
    struct Wrong;

    impl Program for Wrong {
        fn name(&self) -> &str {
            "wrong"
        }

        fn run(
            &self,
            _launch: Launch,
            _inputs: &[View<'_>],
            _outputs: &mut [ViewMut<'_>],
        ) -> Result<()> {
            Err(Error::new(Code::Internal, "内核自己失败了").with("test::Wrong"))
        }
    }

    let mut executor = cpu();
    executor.register(Box::new(Wrong)).expect("登记");
    executor.bind(0, vec![5.0]);
    let mut node = Node::new("wrong", Launch::new([1, 1, 1], [1, 1, 1]));
    node.bind_input(0);
    assert!(executor.run(&node).is_err());
    assert_eq!(executor.get(0).expect("缓冲已放回"), [5.0]);
}

#[test]
fn bindings_replace_and_report_their_size() {
    let mut bindings = Bindings::new();
    assert!(bindings.is_empty());
    assert_eq!(bindings.bind_f32(0, vec![1.0]), None);
    assert_eq!(bindings.bind_f32(0, vec![2.0]), Some(vec![1.0]));
    assert_eq!(bindings.get(0), Some(&[2.0][..]));
    assert_eq!(bindings.get(1), None);
    assert_eq!(bindings.len(), 1);
}

/// 同步只排指定流：别的流上已提交的节点原样留着，等它自己被同步。
#[test]
fn a_cpu_sync_drains_only_the_named_stream() {
    let mut executor = cpu();
    executor
        .register(Box::new(Scale { factor: 2.0 }))
        .expect("登记");
    executor.bind(0, vec![1.0, 1.0]);
    executor.bind(1, vec![0.0; 2]);
    executor.bind(2, vec![0.0; 2]);

    let mut first = Node::new("scale", Launch::new([1, 1, 1], [2, 1, 1]));
    first.bind_input(0).bind_output(1);
    let mut second = Node::new("scale", Launch::new([1, 1, 1], [2, 1, 1]));
    second.bind_input(0).bind_output(2);
    let mut nodes = Graph::new();
    nodes.push(first);
    nodes.push(second);
    executor.submit(&nodes, Stream::new(0)).expect("流 0 入队");
    executor
        .submit(&graph(&["scale"]), Stream::new(1))
        .expect("流 1 入队");
    assert_eq!(executor.len(), 3);

    executor.sync(Stream::new(0)).expect("只排流 0");
    assert_eq!(executor.len(), 1, "流 1 的那一条还在队列里");
    assert_eq!(executor.pending()[0].stream(), Stream::new(1));
    assert_eq!(executor.get(2).expect("流 0 的结果"), [2.0, 2.0]);
}

/// 执行失败的条目留在队列里：它没有跑完，调用方查得到，而不是被静默丢掉。
#[test]
fn a_failed_node_stays_in_the_queue() {
    let mut executor = cpu();
    executor.bind(0, vec![1.0]);
    let mut node = Node::new("scale", Launch::new([1, 1, 1], [1, 1, 1]));
    node.bind_input(0).bind_output(1);
    let mut graph = Graph::new();
    graph.push(node);
    executor.submit(&graph, Stream::new(0)).expect("入队不校验");
    assert_eq!(executor.len(), 1);

    let error = executor.sync(Stream::new(0)).unwrap_err();
    assert_eq!(error.code(), Code::Invalid, "缺的内核由执行时报出来");
    assert_eq!(executor.len(), 1, "没跑完的条目不许出队");
    assert_eq!(executor.get(0).expect("输入原样放回"), [1.0]);
}

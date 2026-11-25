use super::*;
use crate::exec::Buf;
use diagnostic::{Code, Error, Result};
use storage::CpuAllocator;

/// 每个条目仍生成独立 `#[test]`；只复用测试声明，不合并断言。
macro_rules! tests {
    ($($(#[$meta:meta])* $name:ident $body:block)*) => {
        $($(#[$meta])* #[test] fn $name() $body)*
    };
}

/// 单表达式测试夹具也只声明一次；多步校验不进入这个宏。
macro_rules! functions {
    ($($(#[$meta:meta])* fn $name:ident($($args:tt)*) $(->$ret:ty)? = $body:expr;)*) => {
        $($(#[$meta])* fn $name($($args)*) $(->$ret)? { $body })*
    };
}

functions! {
    launch() -> Launch = Launch::new([1, 1, 1], [1, 1, 1]);
    node(name: &str) -> Node = Node::new(name, launch());
    bound(name: &str, inputs: &[u32], outputs: &[u32]) -> Node = {
        let node = inputs.iter().fold(node(name), |mut node, id| { node.bind_input(*id); node });
        outputs.iter().fold(node, |mut node, id| { node.bind_output(*id); node })
    };
    graph_of(nodes: impl IntoIterator<Item = Node>) -> Graph = nodes.into_iter().fold(Graph::new(), |mut graph, node| { graph.push(node); graph });
    graph(names: &[&str]) -> Graph = graph_of(names.iter().map(|name| node(name)));
    local() -> LocalExecutor = LocalExecutor::new(Device::new("cpu", "generic"), Box::new(CpuAllocator::new()));
    pending_names(executor: &LocalExecutor) -> Vec<&str> = executor.pending().iter().map(|queued| queued.node().name()).collect();
    cpu() -> CpuExecutor = CpuExecutor::new(Device::new("cpu", "generic"));
    scaled(factor: f32) -> CpuExecutor = {
        let mut executor = cpu();
        executor.register(Box::new(Scale { factor })).expect("登记");
        executor
    };
}

fn submit<E: Executor>(executor: &mut E, names: &[&str], stream: u32) {
    executor
        .submit(&graph(names), Stream::new(stream))
        .expect("提交");
}

struct Counting {
    device: Device,
    submitted: usize,
}

/// 测试 trait 的单表达式实现复用同一声明骨架。
macro_rules! trait_methods {
    ($trait:ident { $($type:ident { $($name:ident($($args:tt)*) $(->$ret:ty)? = $body:expr; )* }) }) => {
        $(impl $trait for $type { $(fn $name($($args)*) $(->$ret)? { $body })* })*
    };
}

/// 测试内核共享操作数数量、dtype 与等长校验；实际数值闭包保持独立。
fn map_f32(
    inputs: &[View<'_>],
    outputs: &mut [ViewMut<'_>],
    arity: usize,
    site: &'static str,
    apply: impl FnOnce(&[&[f32]], &mut [f32]),
) -> Result<()> {
    if inputs.len() != arity || outputs.len() != 1 {
        return Err(Error::new(Code::Invalid, "操作数数量不符").with(site));
    }
    let typed: Vec<&[f32]> = inputs
        .iter()
        .map(|input| {
            input
                .f32()
                .ok_or_else(|| Error::new(Code::Invalid, "输入不是 f32").with(site))
        })
        .collect::<Result<_>>()?;
    let output = outputs[0]
        .f32_mut()
        .ok_or_else(|| Error::new(Code::Invalid, "输出不是 f32").with(site))?;
    if typed.iter().any(|input| input.len() != output.len()) {
        return Err(Error::new(Code::Invalid, "形状不一致").with(site));
    }
    apply(&typed, output);
    Ok(())
}

struct Scale {
    factor: f32,
}
struct Add;
struct Fail(&'static str);

trait_methods! {
    Executor {
        Counting {
            device(&self) -> &Device = &self.device;
            submit(&mut self, graph: &Graph, _stream: Stream) -> Result<()> = { self.submitted += graph.len(); Ok(()) };
        }
    }
}

trait_methods! {
    Program {
        Scale {
            name(&self) -> &str = "scale";
            run(&self, _launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> = map_f32(inputs, outputs, 1, "test::Scale", |inputs, out| out.iter_mut().zip(inputs[0]).for_each(|(slot, value)| *slot = value * self.factor));
        }
        Add {
            name(&self) -> &str = "add";
            run(&self, _launch: Launch, inputs: &[View<'_>], outputs: &mut [ViewMut<'_>]) -> Result<()> = map_f32(inputs, outputs, 2, "test::Add", |inputs, out| out.iter_mut().zip(inputs[0]).zip(inputs[1]).for_each(|((slot, a), b)| *slot = a + b));
        }
        Fail {
            name(&self) -> &str = self.0;
            run(&self, _launch: Launch, _inputs: &[View<'_>], _outputs: &mut [ViewMut<'_>]) -> Result<()> = Err(Error::new(Code::Internal, "内核自己失败了").with("test::Fail"));
        }
    }
}

tests! {
    total_threads_multiplies_grid_and_block {
        assert_eq!(Launch::new([2, 3, 1], [4, 5, 1]).total_threads(), 120);
        assert_eq!(Launch::new([0, 1, 1], [1, 1, 1]).total_threads(), 0);
    }

    scalar_slots_start_empty {
        let launch = launch();
        assert!(launch.scalars().is_empty());
        assert_eq!(launch.scalars().rope_position(), None);
    }

    a_rope_position_round_trips_through_the_node {
        let mut scalars = ScalarSlots::new();
        scalars.set_rope_position(7);
        let node = Node::new("rope_f32", launch().with_scalars(scalars));
        assert_eq!(node.scalars().rope_position(), Some(7));
        assert_eq!(node.launch().scalars().rope_position(), Some(7));
        let mut other = node("rope_f32");
        other.scalars_mut().set_rope_position(12);
        assert_eq!(other.scalars().rope_position(), Some(12));
    }

    a_graph_keeps_insertion_order {
        let mut graph = Graph::new();
        assert!(graph.is_empty());
        graph.push(node("a"));
        graph.push(node("b"));
        assert_eq!(graph.len(), 2);
        let names: Vec<&str> = graph.nodes().iter().map(Node::name).collect();
        assert_eq!(names, ["a", "b"]);
    }

    cloned_graph_writes_are_independent {
        let mut original = graph(&["a"]);
        let mut cloned = original.clone();
        cloned.push(node("b"));
        assert_eq!(original.len(), 1);
        assert_eq!(cloned.len(), 2);
        original.nodes_mut()[0].scalars_mut().set_rope_position(9);
        assert!(cloned.nodes()[0].scalars().rope_position().is_none());
    }

    an_executor_only_needs_submit {
        let mut executor = Counting { device: Device::new("cpu", "generic"), submitted: 0 };
        executor.submit(&graph(&["gemv_i4"]), Stream::new(0)).expect("提交");
        assert_eq!(executor.submitted, 1);
        assert_eq!(executor.device().vendor(), "cpu");
        executor.sync(Stream::new(0)).expect("CPU 同步是空操作");
    }

    local_executor_keeps_submission_order_within_a_stream {
        let mut executor = local();
        submit(&mut executor, &["a", "b"], 0);
        submit(&mut executor, &["c"], 0);
        assert_eq!(pending_names(&executor), ["a", "b", "c"]);
        assert_eq!(executor.len(), 3);
    }

    local_sync_drains_only_the_named_stream {
        let mut executor = local();
        submit(&mut executor, &["x", "y"], 0);
        submit(&mut executor, &["z"], 1);
        executor.sync(Stream::new(0)).expect("排干流 0");
        assert_eq!(pending_names(&executor), ["z"]);
        assert_eq!(executor.pending()[0].stream(), Stream::new(1));
    }

    local_executor_empty_graph_submits_nothing {
        let mut executor = local();
        executor.submit(&Graph::new(), Stream::new(0)).expect("空图提交");
        assert!(executor.is_empty());
    }

    local_executor_reports_bound_device {
        let executor = LocalExecutor::new(Device::new("cpu", "aarch64"), Box::new(CpuAllocator::new()));
        assert_eq!(executor.device().vendor(), "cpu");
        assert_eq!(executor.device().arch(), "aarch64");
    }

    /// 装饰器转发不改语义，提交序被记录，同步只排被装饰执行器的那条流。
    trace_decorates_any_executor_and_records_submissions {
        let mut traced = Trace::new(local());
        submit(&mut traced, &["a", "b"], 0);
        submit(&mut traced, &["c"], 1);
        assert_eq!(traced.submitted(), [vec!["a".to_owned(), "b".to_owned()], vec!["c".to_owned()]]);
        traced.sync(Stream::new(0)).expect("同步转发");
        assert_eq!(traced.inner().pending().len(), 1);
    }

    local_executor_bounds_injected_allocator {
        let mut executor = local();
        let buf = executor.allocator_mut().alloc(16).expect("分配 16 字节");
        assert_eq!(buf.len(), 16);
        executor.allocator_mut().free(buf);
        assert!(executor.allocator_mut().alloc(0).expect("零字节分配").is_empty());
    }

    a_new_node_has_no_operands {
        let node = node("softmax_f32");
        assert!(node.inputs().is_empty());
        assert!(node.outputs().is_empty());
    }

    node_binds_input_and_output_operands {
        let node = bound("gemv_f32", &[0, 1], &[2]);
        let inputs: Vec<u32> = node.inputs().iter().map(Operand::id).collect();
        let outputs: Vec<u32> = node.outputs().iter().map(Operand::id).collect();
        assert_eq!(inputs, [0, 1]);
        assert_eq!(outputs, [2]);
        assert_eq!(node.inputs()[0].role(), OperandRole::Input);
        assert_eq!(node.outputs()[0].role(), OperandRole::Output);
    }

    queued_node_keeps_its_operand_bindings {
        let mut graph = Graph::new();
        graph.push(bound("rope_f32", &[3], &[4]));
        let mut executor = local();
        executor.submit(&graph, Stream::new(0)).expect("提交");
        let queued = executor.pending();
        assert_eq!(queued[0].node().inputs()[0].id(), 3);
        assert_eq!(queued[0].node().outputs()[0].id(), 4);
    }

    a_cpu_executor_writes_through_the_bound_operands {
        let mut executor = scaled(2.0);
        executor.bind(0, Buf::F32(vec![1.0, 2.0, 3.0]));
        executor.bind(1, Buf::F32(vec![0.0; 3]));
        executor.run(&bound("scale", &[0], &[1])).expect("执行");
        assert_eq!(executor.get(0).expect("输入还在"), [1.0, 2.0, 3.0]);
        assert_eq!(executor.get(1).expect("输出写回"), [2.0, 4.0, 6.0]);
    }

    /// 图序就是执行序：第二个节点读到第一个节点写过的值，同步前不写结果。
    a_graph_chains_through_the_same_bindings {
        let mut executor = scaled(3.0);
        executor.register(Box::new(Add)).expect("登记");
        for (id, values) in [(0, [1.0, 1.0]), (1, [0.0; 2]), (2, [10.0, 20.0]), (3, [0.0; 2])] {
            executor.bind(id, Buf::F32(values.to_vec()));
        }
        let graph = graph_of([bound("scale", &[0], &[1]), bound("add", &[1, 2], &[3])]);
        executor.submit(&graph, Stream::new(0)).expect("入队");
        assert_eq!(executor.len(), 2);
        assert_eq!(executor.get(3).expect("输出缓冲"), [0.0, 0.0]);
        executor.sync(Stream::new(0)).expect("执行");
        assert!(executor.is_empty());
        assert_eq!(executor.get(3).expect("最后一步输出"), [13.0, 23.0]);
        assert_eq!(executor.bindings().len(), 4);
    }

    an_unregistered_kernel_is_refused_by_name {
        let mut executor = cpu();
        executor.bind(0, Buf::F32(vec![1.0]));
        let error = executor.run(&bound("softmax_f32", &[], &[0])).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("softmax_f32"), "{error}");
    }

    a_second_registration_of_the_same_name_is_refused {
        let mut executor = scaled(1.0);
        let error = executor.register(Box::new(Scale { factor: 2.0 })).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert!(error.to_string().contains("已登记"), "{error}");
        assert_eq!(executor.bindings().len(), 0);
    }

    an_unbound_operand_is_refused {
        let mut executor = scaled(1.0);
        executor.bind(0, Buf::F32(vec![1.0]));
        let error = executor.run(&bound("scale", &[0], &[9])).unwrap_err();
        assert!(error.to_string().contains('9'), "{error}");
    }

    /// 一个编号不能同时是输入与输出，否则借出与放回会拿到同一块缓冲两次。
    the_same_operand_cannot_be_both_input_and_output {
        let mut executor = scaled(2.0);
        executor.bind(0, Buf::F32(vec![1.0]));
        let error = executor.run(&bound("scale", &[0], &[0])).unwrap_err();
        assert!(error.to_string().contains("既是输入又是输出"), "{error}");
    }

    /// 内核失败时缓冲必须放回，避免下一次执行把原始错误覆盖成“未绑定”。
    a_failing_kernel_still_returns_its_bufs {
        let mut executor = cpu();
        executor.register(Box::new(Fail("wrong"))).expect("登记");
        executor.bind(0, Buf::F32(vec![5.0]));
        assert!(executor.run(&bound("wrong", &[0], &[])).is_err());
        assert_eq!(executor.get(0).expect("缓冲已放回"), [5.0]);
    }

    bindings_replace_and_report_their_size {
        let mut bindings = Bindings::new();
        assert!(bindings.is_empty());
        assert_eq!(bindings.bind(0, Buf::F32(vec![1.0])), None);
        assert_eq!(bindings.bind(0, Buf::F32(vec![2.0])).as_ref().and_then(Buf::f32), Some(&[1.0][..]));
        assert_eq!(bindings.get(0), Some(&[2.0][..]));
        assert_eq!(bindings.get(1), None);
        assert_eq!(bindings.len(), 1);
    }

    /// 同步只排指定流，别的流上已提交节点原样保留。
    a_cpu_sync_drains_only_the_named_stream {
        let mut executor = scaled(2.0);
        for id in 0..3 {
            executor.bind(id, Buf::F32(vec![1.0, 1.0]));
        }
        let nodes = graph_of([bound("scale", &[0], &[1]), bound("scale", &[0], &[2])]);
        executor.submit(&nodes, Stream::new(0)).expect("流 0 入队");
        submit(&mut executor, &["scale"], 1);
        assert_eq!(executor.len(), 3);
        executor.sync(Stream::new(0)).expect("只排流 0");
        assert_eq!(executor.len(), 1);
        assert_eq!(executor.pending()[0].stream(), Stream::new(1));
        assert_eq!(executor.get(2).expect("流 0 的结果"), [2.0, 2.0]);
    }

    /// 执行失败的条目留在队列里，输入缓冲也必须归还。
    a_failed_node_stays_in_the_queue {
        let mut executor = cpu();
        executor.bind(0, Buf::F32(vec![1.0]));
        let graph = graph_of([bound("scale", &[0], &[1])]);
        executor.submit(&graph, Stream::new(0)).expect("入队不校验");
        let error = executor.sync(Stream::new(0)).unwrap_err();
        assert_eq!(error.code(), Code::Invalid);
        assert_eq!(executor.len(), 1);
        assert_eq!(executor.get(0).expect("输入原样放回"), [1.0]);
    }

    /// 未接入运行时的后端只登记厂商标签，所有运行面明确返回 Unsupported。
    vacant_executor_rejects_every_runtime_surface {
        let mut executor = VacantExecutor::new(Device::new("cuda", "sm_90"));
        assert_eq!(executor.device().vendor(), "cuda");
        assert_eq!(executor.vendor(), "cuda");
        let error = executor.submit(&graph(&["gemv_i4"]), Stream::new(0)).unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        assert!(error.to_string().contains("cuda"), "{error}");
        let program: Box<dyn Program> = Box::new(Fail("placeholder"));
        assert_eq!(executor.register_program(program).unwrap_err().code(), Code::Unsupported);
        assert!(executor.bind(0, Buf::F32(vec![1.0])).is_none());
        assert!(executor.bind(0, Buf::U8(vec![1u8])).is_none());
        assert!(executor.fetch(0).is_none());
    }
}

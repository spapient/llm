use super::*;

struct Counting {
    submitted: usize,
}

impl Executor for Counting {
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
    let mut executor = Counting { submitted: 0 };
    executor.submit(&graph, Stream::new(0)).expect("提交");
    assert_eq!(executor.submitted, 1);
}

use super::*;

#[test]
fn counts_separate_roles_and_graph_edges() {
    let mut tape = Tape::new();
    let w = tape.parameter();
    let c = tape.leaf();
    let hidden = tape.record("relu", &[w]).expect("记录");
    let _unused = tape.record("relu", &[c]).expect("记录");
    let loss = tape.record("add", &[hidden]).expect("记录");

    let summary = Summary::of(&tape);
    assert_eq!(summary.vars(), 5);
    assert_eq!(summary.nodes(), 3);
    assert_eq!(summary.edges(), 3);
    assert_eq!(summary.parameters(), 1);
    assert_eq!(summary.constants(), 1);
    assert_eq!(summary.activations(), 3);
    assert_eq!(summary.max_depth(), 2);
    assert_eq!(summary.steps(), 0);

    let schedule = tape.backward(loss).expect("在图上");
    let planned = Summary::for_schedule(&tape, &schedule);
    assert_eq!(planned.steps(), 2, "loss 可达 hidden 与 loss 两步");
    assert_eq!(planned.endpoints(), 1);
    assert!(planned.reachable() >= planned.endpoints());
    assert_eq!(
        planned.dead(),
        summary.vars() as usize - planned.reachable()
    );
}

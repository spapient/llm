use super::*;

#[test]
fn empty_queue_drains_to_nothing() {
    let queue: Queue<usize> = Queue::new();
    assert!(queue.is_empty());
    assert_eq!(queue.len(), 0);
    assert!(queue.drain().is_empty());
}

#[test]
fn drain_orders_by_priority_ordinal_within_a_batch() {
    let mut queue = Queue::new();
    // 入队顺序打乱优先级；载荷编号记录入队先后。
    queue.submit(Priority::Low, 1);
    queue.submit(Priority::High, 2);
    queue.submit(Priority::Normal, 3);
    queue.submit(Priority::High, 4);

    let drained = queue.drain();
    let order: Vec<_> = drained.iter().map(Job::priority).collect();
    assert_eq!(
        order,
        [
            Priority::High,
            Priority::High,
            Priority::Normal,
            Priority::Low
        ]
    );
    // 同档保持入队先后：High 档 2 先于 4。
    let payloads: Vec<_> = drained.into_iter().map(Job::into_payload).collect();
    assert_eq!(payloads, [2, 4, 3, 1]);
}

#[test]
fn jobs_carry_their_payload_through_the_queue() {
    let mut queue: Queue<(&str, Priority)> = Queue::new();
    queue.submit(Priority::Normal, ("meta", Priority::Normal));
    let job = queue.drain().pop().expect("一条作业");
    assert_eq!(job.into_payload(), ("meta", Priority::Normal));
}

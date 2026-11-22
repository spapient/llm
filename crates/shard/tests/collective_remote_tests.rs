use crate::{Collective, Group, Local, Parallel, Reduce, Remote, Topology};
use diagnostic::Code;

/// 占位远端后端拒绝任何集合操作；缓冲原样保留，不产生半个归约结果。
#[test]
fn all_reduce_is_unsupported_and_leaves_the_buf_untouched() {
    let mut remote = Remote::new();
    let group = Group::new([0, 1]).expect("双进程组");
    let mut buf = [1.0f32, 2.0, 3.0];

    let error = remote
        .all_reduce(&group, Reduce::Sum, &mut buf)
        .expect_err("远端未挂载应拒绝归约");
    assert_eq!(error.code(), Code::Unsupported);
    assert_eq!(buf, [1.0, 2.0, 3.0], "拒绝时不得改动调用方缓冲");
}

/// [`Local`] 与 [`Remote`] 能放进同一个 `Box<dyn Collective>`：端口契约不随后端切换。
#[test]
fn local_and_remote_share_the_collective_port() {
    let topology = Topology::new(2, 1, 1).expect("tp 2");
    let mut local: Box<dyn Collective> = Box::new(Local::new(topology, 2).expect("宽度非零"));
    let mut remote: Box<dyn Collective> = Box::new(Remote::new());

    let group = topology.group(Parallel::Tensor, 0).expect("唯一一组");
    let mut buf = [0.0f32; 2];
    local
        .all_reduce(&group, Reduce::Sum, &mut buf)
        .expect("进程内归约");
    assert_eq!(
        remote
            .all_reduce(&group, Reduce::Sum, &mut buf)
            .unwrap_err()
            .code(),
        Code::Unsupported
    );
}

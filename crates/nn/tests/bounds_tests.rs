use super::*;

#[test]
fn a_projection_declares_its_floor() {
    // 4096 方阵的 int4 权重：光是"读一遍权重"就已经这么多字节，
    // 再加上输入、每个 32 元素的缩放因子与输出。
    let floor = Bounds::of_op("gemv_i4", 4096, 4096).expect("生产形状");
    assert_eq!(
        floor.read(),
        4096 * 4096 / 2 + 4096 * 4 + 4096 * (4096 / 32) * 4
    );
    assert_eq!(floor.write(), 4096 * 4, "写方向只有输出那一份");
    assert_eq!(
        floor.bytes(),
        4096 * 4096 / 2 + 4096 * 4 + 4096 * (4096 / 32) * 4 + 4096 * 4,
        "合计与改造前逐字节相同"
    );
    assert_eq!(floor.flops(), 2.0 * 4096.0 * 4096.0);
    // 量化让"每搬一个字节能算几次"落在 3.19，而不是同一层用 f32 权重时的 1.0。
    assert_eq!((floor.intensity() * 100.0).round() / 100.0, 3.19);
}

#[test]
fn layer_geometry_preserves_projection_and_named_shape_mappings() {
    for (input, output) in [(64, 4), (32, 1), (4096, 64)] {
        let unified = Bounds::of_geometry("gemv_i4", LayerGeometry::projection(input, output))
            .expect("合法投影");
        assert_eq!(
            unified,
            Bounds::of_op("gemv_i4", input, output).expect("合法投影")
        );
    }

    let gather = Bounds::of_geometry("embed_f32", LayerGeometry::shape(8, 128))
        .expect("8 个 token、每条 128 宽");
    assert_eq!(gather.read(), 8 * 128 * 4);
    assert_eq!(gather.write(), 8 * 128 * 4);
    assert_eq!(gather.flops(), 8.0);
    assert_eq!(LayerGeometry::axis(64), LayerGeometry::shape(64, 1));
}

#[test]
fn the_floor_is_exactly_the_operator_semantics() {
    let model = nn::registry::model("gemv_i4").expect("有语义表");
    for (m, n) in [(64, 4), (32, 1), (4096, 64)] {
        let floor = Bounds::of_op("gemv_i4", m, n).expect("合法形状");
        let shape = model.projection(m, n).expect("合法形状");
        let cost = (model.cost)(&shape).expect("合法形状");
        assert_eq!(floor.read(), cost.read, "{m}x{n}");
        assert_eq!(floor.write(), cost.write, "{m}x{n}");
        assert_eq!(floor.flops(), cost.flops, "{m}x{n}");
    }
}

#[test]
fn the_two_directions_are_not_collapsed() {
    let floor = Bounds::of_op("gemv_i4", 4096, 4096).expect("生产形状");
    assert!(
        floor.read() > 8 * 1024 * 1024,
        "读侧是权重主导：{}",
        floor.read()
    );
    assert!(
        floor.write() < floor.read() / 100,
        "写侧只有一个 f32 输出向量：{}",
        floor.write()
    );
}

#[test]
fn shifting_traffic_between_directions_does_not_move_within() {
    // 总量相同，只是把读挪成写：这不是"更省"，只是换了一个瓶颈。
    let read_heavy = Bounds {
        read: 100,
        write: 0,
        flops: 1.0,
    };
    let write_heavy = Bounds {
        read: 0,
        write: 100,
        flops: 1.0,
    };
    assert_eq!(read_heavy.bytes(), write_heavy.bytes());
    assert!(!read_heavy.moves_within(write_heavy));
    assert!(!write_heavy.moves_within(read_heavy));
    assert!(read_heavy.moves_within(read_heavy), "自己落于自己之内");
}

#[test]
fn a_fused_layer_moves_within_the_sum_of_its_parts() {
    let weights = Bounds::of_op("gemv_i4", 4096, 4096).expect("合法");
    let sum = weights.plus(weights);
    let fused = Bounds {
        read: weights.read,
        write: weights.write,
        flops: 2.0 * weights.flops,
    };
    assert!(fused.moves_within(sum), "融合省掉了中间那份流量");
    assert!(!sum.moves_within(fused), "反过来不成立：和不会更小");
}

#[test]
fn a_shape_the_operator_rejects_has_no_floor() {
    // 层本身可以构造（宽度非零），但 8 不是量化分组的整数倍：
    // 这个形状根本没有合法内核，因此谈不上"至少搬多少"。
    let error = Bounds::of_op("gemv_i4", 8, 4).expect_err("形状非法");
    assert_eq!(error.code(), Code::Invalid);
    assert!(error.to_string().contains("整数倍"), "{error}");
}

#[test]
fn an_operator_without_semantics_has_no_floor() {
    let error = Bounds::of_op("flash_attention", 64, 4).expect_err("没有语义表");
    assert_eq!(error.code(), Code::Missing);
    assert!(error.to_string().contains("flash_attention"), "{error}");
}

#[test]
fn floors_add_up_across_a_stack() {
    let one = Bounds::of_op("gemv_i4", 64, 4).expect("合法");
    let two = one.plus(one);
    assert_eq!(two.read(), 2 * one.read());
    assert_eq!(two.write(), 2 * one.write());
    assert_eq!(two.bytes(), 2 * one.bytes());
    assert_eq!(two.flops(), 2.0 * one.flops());
    assert_eq!(two.intensity(), one.intensity(), "同构的两层强度不变");
    assert_eq!(Bounds::ZERO.plus(one), one, "零元");
    assert_eq!(Bounds::ZERO.intensity(), 0.0, "除零不返回 NaN");
}

#[test]
fn a_composite_layer_is_the_sum_of_its_operators() {
    let one = vec!["gemv_i4".to_string()];
    let twice = vec!["gemv_i4".to_string(), "gemv_i4".to_string()];
    assert_eq!(Bounds::of_ops(&[], 64, 4).expect("空列表"), Bounds::ZERO);
    assert_eq!(
        Bounds::of_ops(&twice, 64, 4).expect("合法"),
        Bounds::of_ops(&one, 64, 4)
            .expect("合法")
            .plus(Bounds::of_ops(&one, 64, 4).expect("合法"))
    );
    let error = Bounds::of_ops(&twice, 8, 4).expect_err("形状非法");
    assert!(error.to_string().contains("gemv_i4"), "{error}");
}

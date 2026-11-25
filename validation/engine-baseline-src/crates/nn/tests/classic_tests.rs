//! 经典几何层族（`spatial` 与 `norm`）的用例。
//!
//! 生产文件只留执行与判据，用例集中于此；N3 行数门限不再受用例增删影响。
//! 两族的用例同处一份：它们校验的是同一类构造期判据（非零、整除、eps 为正）。

use super::*;

use crate::{BatchNorm2d, Bounds, Contract, GroupNorm, InstanceNorm, LayerNorm};

fn geom3() -> ConvGeom {
    ConvGeom {
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        ph: 1,
        pw: 1,
    }
}

#[test]
fn conv2d_registers_its_shape_and_conv_operator() {
    let layer = Conv2d::new("stem.conv", 3, 64, geom3(), 32, 32).expect("合法");
    assert_eq!(layer.name(), "stem.conv");
    assert_eq!(layer.output().dims(), [32, 32, 64]);
    assert_eq!(layer.spca().ops(), [CONV]);
    assert_eq!(layer.shape(), (32, 32));
    // 下界来自 conv2d_f32 的语义表（每输出 2·k²·cin、三读一写），因此有正 FLOPs 与正字节。
    let floor = layer.bounds().expect("合法").expect("有算子");
    assert!(floor.flops() > 0.0 && floor.bytes() > 0);
}

#[test]
fn spatial_floors_are_the_output_shape_rule() {
    // 这一族的 `impl Contract` 都由 `priced!` 生成，下界实参不再写在各层里：行 = 输出 numel / 末轴，
    // 宽 = 末轴。口径收成一份之后，"抄错一个乘积"的风险也收成了一份、且编译器看不见——本用例把
    // 这条规则钉在具体层上：任何一层换了字段、换了轴序，这里立刻不等。
    let conv = Conv2d::new("stem.conv", 3, 64, geom3(), 32, 32).expect("合法");
    assert_eq!(
        conv.bounds().expect("合法").expect("可计价"),
        Bounds::of_ctx(CONV, 32 * 32, 64).expect("合法")
    );

    // 二维池化由 `planar!` 生成、一维池化手写：两条路径必须给出同一个口径。
    let pool2 = MaxPool2d::new("p2", geom3(), 8, 8, 4).expect("合法");
    assert_eq!(
        pool2.bounds().expect("合法").expect("可计价"),
        Bounds::of_ctx("max_pool2d_f32", 8 * 8, 4).expect("合法")
    );
    let pool1 = MaxPool1d::new("p1", 2, 2, 0, 8, 4).expect("合法");
    assert_eq!(
        pool1.bounds().expect("合法").expect("可计价"),
        Bounds::of_ctx("max_pool1d_f32", 4, 4).expect("合法")
    );

    // 反卷积的输出通道存在算子结构体里（`deconv.cout`），与"末轴"是不是同一个数要由这里保证。
    let up = ConvTranspose1d::new("up", 4, 6, 2, 2, 0, 9).expect("合法");
    assert_eq!(
        up.bounds().expect("合法").expect("可计价"),
        Bounds::of_ctx(GEMM, 18, 6).expect("合法")
    );
}

#[test]
fn conv2d_rejects_zero_channels() {
    assert!(Conv2d::new("x", 0, 8, geom3(), 8, 8).is_err());
}

#[test]
fn conv1d_and_transpose_lengths_follow_the_geometry() {
    // 一维卷积只吃 `kw/sw/pw`：核 2、步长 1、无补零，长度 9 → 8。
    let layer = Conv1d::new(
        "a",
        4,
        8,
        ConvGeom {
            kh: 1,
            kw: 2,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
        },
        9,
    )
    .expect("合法");
    assert_eq!(layer.len(), 8);
    assert_eq!(layer.output().dims(), [8, 8]);

    let up = ConvTranspose1d::new("up", 4, 6, 2, 2, 0, 9).expect("合法");
    assert_eq!(up.len(), 18);
    assert_eq!(up.output().dims(), [18, 6]);

    let up2 = ConvTranspose2d::new("up2", 4, 6, geom3(), 8, 8).expect("合法");
    assert_eq!(up2.shape(), (8, 8));
    assert_eq!(up2.output().dims(), [8, 8, 6]);
}

#[test]
fn pool_and_interpolate_layers_bind_one_real_operator() {
    let pool = MaxPool1d::new("p1", 2, 2, 0, 8, 4).expect("合法");
    assert_eq!(pool.output().dims(), [4, 4]);
    assert_eq!(pool.spca().ops(), ["max_pool1d_f32"]);
    // 有真算子就能推出下界：读一遍输入、写一遍输出。
    let lower = pool.bounds().expect("合法").expect("可计价");
    assert!(lower.read() > 0 && lower.write() > 0 && lower.flops() > 0.0);

    let pool2 = MaxPool2d::new("p2", geom3(), 8, 8, 4).expect("合法");
    assert_eq!(pool2.output().dims(), [8, 8, 4]);
    assert_eq!(pool2.spca().ops(), ["max_pool2d_f32"]);

    let avg = AvgPool2d::new("p3", geom3(), false, 8, 8, 4).expect("合法");
    assert_eq!(avg.output().dims(), [8, 8, 4]);
    assert_eq!(avg.spca().ops(), ["avg_pool2d_f32"]);

    let adp = AdaptiveAvgPool2d::new("p4", 2, 2, 4).expect("合法");
    assert_eq!(adp.output().dims(), [2, 2, 4]);
    assert_eq!(adp.spca().ops(), ["adaptive_pool_f32"]);

    let up = Interpolate::new("i", crate::conv::Sampler::Linear, 16, 16, 4).expect("合法");
    assert_eq!(up.output().dims(), [16, 16, 4]);
    assert_eq!(up.spca().ops(), ["interpolate_f32"]);
}

#[test]
fn norm_layers_check_their_geometry() {
    let bn = BatchNorm2d::new("bn", 64, 1e-5, 0.1).expect("合法");
    assert_eq!(bn.output().dims(), [64]);
    assert_eq!(bn.spca().ops(), ["batch_norm_f32"]);
    assert!(bn.bounds().expect("合法").is_some());
    // 超参存了就必须能读回：否则它只是会被拷贝、会被比较、却永不参与判断的状态。
    assert_eq!((bn.norm().eps, bn.norm().momentum), (1e-5, 0.1));
    assert!(BatchNorm2d::new("bn", 64, 0.0, 0.1).is_err());
    assert!(BatchNorm2d::new("bn", 64, f32::INFINITY, 0.1).is_err());
    assert!(BatchNorm2d::new("bn", 64, f32::NEG_INFINITY, 0.1).is_err());
    assert!(BatchNorm2d::new("bn", 64, f32::NAN, 0.1).is_err());
    assert!(BatchNorm2d::new("bn", 64, 1e-5, 1.5).is_err());

    let gn = GroupNorm::new("gn", 8, 64, 1e-5).expect("合法");
    assert_eq!(gn.output().dims(), [64]);
    assert_eq!(gn.norm().groups, 8);
    assert!(GroupNorm::new("gn", 8, 60, 1e-5).is_err());

    let ins = InstanceNorm::new("in", 32, 1e-5).expect("合法");
    assert_eq!(ins.output().dims(), [32]);

    let ln = LayerNorm::new("ln", 512, 1e-5).expect("合法");
    assert_eq!(ln.output().dims(), [512]);
    assert!(LayerNorm::new("ln", 512, -1.0).is_err());
}

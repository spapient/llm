//! `src/single.rs` 的全部用例：逐元素/归一化/激活、归约、查表、矩阵乘（精度与档位）、
//! 前缀扫描与反量化共用一条契约——层不写第二份流量常数，下界一律由算子语义表推出。
//!
//! 逐元素层原先住在 `src/pointwise.rs`，测试因此分出去一份；该文件已并入 `single.rs`，
//! 两者又挂同一个模块，分开只剩两个同义助手，故合并。

use super::*;

use crate::{Bounds, Contract, ParamRole};
use diagnostic::Code;

#[test]
fn an_embedding_declares_its_table_and_output_shape() {
    let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
    assert_eq!(layer.name(), "tok_emb");
    assert_eq!(layer.vocab(), 32_000);
    assert_eq!(layer.dim(), 128);
    assert_eq!(layer.tokens(), 8);
    assert_eq!(layer.output().dims(), [8, 128]);
    assert_eq!(layer.spca().ops(), [Embedding::OP]);
    assert_eq!(layer.spca().role(), ParamRole::Embedding);
    assert!(layer.spca().role().wd_exempt());
}

#[test]
fn an_embedding_floor_follows_the_gather_semantics() {
    let layer = Embedding::new("tok_emb", 32_000, 128, 8).expect("合法");
    let floor = layer.bounds().expect("合法").expect("有算子");
    // 独立按 gather 定义手算：8 个 token 各搬一条 128 宽的 f32 行。
    let elements = (8 * 128) as u64;
    assert_eq!(floor.read(), elements * 4, "读 8×128 个 f32");
    assert_eq!(floor.write(), elements * 4, "写 8×128 个 f32");
    assert_eq!(floor.flops(), 8.0, "每个 token 一次下标转换");
}

#[test]
fn a_zero_dimension_embedding_is_rejected() {
    assert_eq!(
        Embedding::new("x", 0, 128, 8).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Embedding::new("x", 32, 0, 8).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        Embedding::new("x", 32, 128, 0).unwrap_err().code(),
        Code::Invalid
    );
}

#[test]
fn a_matmul_layer_binds_its_operator_and_shape() {
    let layer = MatMul::new("blk.0.matmul", 16, 12).expect("合法");
    assert_eq!(layer.name(), "blk.0.matmul");
    assert_eq!(layer.output().dims(), [16, 12]);
    assert_eq!(layer.spca().ops(), [MatMul::OP]);
    assert_eq!(layer.rows(), 16);
    assert_eq!(layer.cols(), 12);
}

#[test]
fn the_matmul_floor_follows_the_gemm_semantics() {
    let layer = MatMul::new("blk.0.matmul", 16, 12).expect("合法");
    let floor = layer.bounds().expect("合法").expect("有算子");
    // 层不写第二份公式：下界必须与直接问语义表逐项相等。
    let same = Bounds::of_ctx(MatMul::OP, 16, 12).expect("同一份语义");
    assert_eq!(floor, same, "矩阵乘下界来自算子语义表");
    assert!(floor.read() > 0 && floor.write() > 0 && floor.flops() > 0.0);
}

#[test]
fn a_zero_dimension_matmul_is_rejected() {
    assert_eq!(MatMul::new("x", 0, 12).unwrap_err().code(), Code::Invalid);
    assert_eq!(MatMul::new("x", 16, 0).unwrap_err().code(), Code::Invalid);
}

/// 单轴层与矩阵乘层都必须能当 `&dyn Contract` 用：装配侧只认这个视角。
#[test]
fn layers_work_as_trait_objects() {
    let axis: &dyn Contract = &Silu::new("mlp.silu", 64).expect("合法");
    assert_eq!(axis.output().dims(), [64]);
    assert!(axis.bounds().expect("合法").is_some());

    let matmul: &dyn Contract = &MatMul::new("proj", 16, 12).expect("合法");
    assert_eq!(matmul.output().dims(), [16, 12]);
    assert!(matmul.bounds().expect("合法").is_some());
}

/// 权重衰减豁免与可训练判定都从角色派生：归一化的系数不衰减，激活没有参数轴之外的例外。
#[test]
fn a_norm_layer_declares_the_norm_role_and_an_activation_the_trainable_role() {
    let norm = RmsNorm::new("blk.0.norm", 128).expect("合法");
    assert_eq!(norm.spca().role(), ParamRole::Norm);
    assert!(norm.spca().role().wd_exempt());

    let activation = Gelu::new("mlp.gelu", 128).expect("合法");
    assert_eq!(activation.spca().role(), ParamRole::Trainable);
}

/// 单轴层的通用断言：名字、形状、绑定的算子名，以及与语义表逐项相等的下界、非零的两笔流量。
fn assert_layer<L: Contract>(layer: &L, name: &str, op: &str, len: usize) {
    assert_eq!(layer.name(), name);
    assert_eq!(layer.output().dims(), [len]);
    assert_eq!(layer.spca().ops(), [op]);
    let floor = layer.bounds().expect("单轴形状合法").expect("有算子");
    // 单轴算子不写第二份公式：下界必须与直接问语义表逐项相等。
    assert_eq!(
        floor,
        Bounds::of_ctx(op, len, 1).expect("同一份语义"),
        "{op} 的下界来自算子语义表"
    );
    assert!(floor.read() > 0 && floor.write() > 0, "{op} 有读有写");
}

/// 每个层只剩「类型 + 名字 + 长度」三项事实：构造、名字、算子名、形状、下界都由这两行给出，
/// 于是加一个单轴层不必再抄五遍同样的断言。
macro_rules! layer {
    ($ty:ident, $name:literal, $len:literal) => {
        assert_layer(&$ty::new($name, $len).expect("合法"), $name, $ty::OP, $len)
    };
}

/// 逐元素 / 归一化 / 激活层各自绑定自己的算子、各自从语义表取价。
/// 这张清单也是「哪些层存在」的唯一一处：加层时这里必须多一行，否则新层没有任何断言。
#[test]
fn every_pointwise_layer_binds_its_declared_operator() {
    layer!(RmsNorm, "blk.0.norm", 128);
    layer!(Softmax, "attn.softmax", 64);
    layer!(Gelu, "mlp.gelu", 128);
    layer!(Silu, "mlp.silu", 128);
    layer!(LeakyRelu, "act.leaky", 128);
    layer!(HardSigmoid, "act.hard", 128);
    layer!(Rope, "attn.rope", 128);
    layer!(Scale, "proj.scale", 128);
    layer!(Add, "res.add", 128);
    layer!(Mul, "gate.mul", 128);
    layer!(Relu, "act.relu", 128);
    layer!(Sigmoid, "act.sigmoid", 128);
    layer!(Tanh, "act.tanh", 128);
    // 除 / 减 / 取大 / 取小 / 指数 / 对数此前算子表里有、层集里没有：并进来一并钉住。
    layer!(Sub, "blk.0.sub", 128);
    layer!(Div, "blk.0.div", 128);
    layer!(Max, "blk.0.max", 128);
    layer!(Min, "blk.0.min", 128);
    layer!(Exp, "blk.0.exp", 128);
    layer!(Log, "blk.0.log", 128);
}

/// 所有单轴层沿用同一条「零长度拒绝」口径，不留一个可以构造出空轴的后门。
#[test]
fn a_zero_length_axis_tensor_is_rejected() {
    assert_eq!(Gelu::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Silu::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(RmsNorm::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Sub::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Div::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Max::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Min::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Exp::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Log::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Relu::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Sigmoid::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Tanh::new("x", 0).unwrap_err().code(), Code::Invalid);
}

/// 收缩轴档位是算子表声明的固定档（`gemm_*` = 8、`matmul_*` = 4096），不是层能编造的尺寸。
/// 实测两档的读侧与 flops 都精确相差 512 倍——这条用例锁的就是「档位真的进了计价」，
/// 而不是层拿小档算子给大档层报一条偏小 512 倍的下界。
#[test]
fn the_matmul_gauge_is_declared_and_reaches_the_price() {
    let short = MatMul::with_precision("m", 16, 12, Precision::F32, Gauge::Short).expect("合法");
    let long = MatMul::with_precision("m", 16, 12, Precision::F32, Gauge::Long).expect("合法");
    assert_eq!(short.spca().ops(), ["gemm_f32"]);
    assert_eq!(long.spca().ops(), ["matmul_f32"]);
    let short = short.bounds().expect("合法").expect("有算子");
    let long = long.bounds().expect("合法").expect("有算子");
    assert_eq!(
        long.read(),
        short.read() * 512,
        "K 8 → 4096 是 512 倍的读数"
    );
    assert_eq!(
        long.flops(),
        short.flops() * 512.0,
        "同一条乘加，K 放大 512 倍"
    );
}

/// 精度决定读写字节、不决定算术量：同一档位下 int8 / int4 / int2 的读侧按元素宽度精确缩到
/// 1/4、1/8、1/16，而 flops 与 f32 逐位相同——「几个数的乘加还是那么多次」。
#[test]
fn the_matmul_precision_scales_reads_and_never_the_arithmetic() {
    let floor = |precision| {
        MatMul::with_precision("m", 16, 12, precision, Gauge::Short)
            .expect("合法")
            .bounds()
            .expect("合法")
            .expect("有算子")
    };
    let (f32, i8, i4, i2) = (
        floor(Precision::F32),
        floor(Precision::Int8),
        floor(Precision::Int4),
        floor(Precision::Int2),
    );
    assert_eq!(i8.read() * 4, f32.read());
    assert_eq!(i4.read() * 8, f32.read());
    assert_eq!(i2.read() * 16, f32.read());
    for (label, other) in [("int8", i8), ("int4", i4), ("int2", i2)] {
        assert_eq!(other.flops(), f32.flops(), "{label} 与 f32 做同样多次乘加");
        assert_eq!(other.write(), f32.write(), "{label} 的输出仍是 f32");
    }
}

/// 算子表只注册了 int2 / int4 × `K = 8`：大收缩轴的这两个精度不存在。层的回答是
/// 「没有这个组合」并说清缺哪一半，而不是退回小 K 的兄弟算子凑一条下界。
#[test]
fn an_unregistered_precision_rung_says_which_half_is_missing() {
    for precision in [Precision::Int2, Precision::Int4] {
        let layer =
            MatMul::with_precision("m", 16, 12, precision, Gauge::Long).expect("仍是合法声明");
        assert!(layer.spca().ops().is_empty(), "没有内核就没有算子名");
        let error = layer.bounds().unwrap_err();
        assert_eq!(error.code(), Code::Unsupported);
        let message = error.to_string();
        assert!(message.contains(precision.name()), "{message}");
        assert!(message.contains("K = 4096"), "{message}");
    }
}

/// 前缀扫描层：绑 `scan_f32`，读写覆盖整条时间轴，算术量是每个元素的递推。
#[test]
fn the_prefix_scan_binds_its_recurrence_operator_and_prices_it() {
    let scan = Scan::new("state", 16).expect("合法");
    assert_eq!(scan.spca().ops(), [Scan::OP]);
    assert_eq!(scan.output().dims(), [16], "扫描不改变长度");
    let floor = scan.bounds().expect("合法").expect("有算子");
    assert_eq!(floor, Bounds::of_ctx(Scan::OP, 16, 1).expect("同一份语义"));
    // 递推每元素读一次、写一次，两者都是 f32 满宽。
    assert_eq!(floor.read(), floor.write(), "读写各一次，各按 f32 满宽");
    assert_eq!(floor.read(), 16 * 4, "16 个 f32");
    assert!(floor.flops() > 0.0, "递推有算术，不是纯搬运");
    assert!(Scan::new("state", 0).is_err(), "零长度在构造期拒绝");
}

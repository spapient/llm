//! 单算子层的用例：逐元素/归一化/激活/查表/矩阵乘共用一条契约——
//! 层不写第二份流量常数，下界一律由算子语义表推出。

use super::*;

use crate::{Bounds, Contract};
use diagnostic::Code;

#[test]
fn a_reduction_emits_a_scalar_shape() {
    let sum = ReduceSum::new("loss.sum", 128).expect("合法");
    assert_eq!(sum.output().dims(), [], "归约输出是标量");
    assert_eq!(sum.spca().ops(), [ReduceSum::OP]);
    let floor = sum.bounds().expect("合法").expect("有算子");
    assert_eq!(
        floor,
        Bounds::of_ctx(ReduceSum::OP, 128, 1).expect("同一份语义")
    );
    // 读侧覆盖整个输入，写侧只有一个标量。
    assert!(floor.read() > floor.write());
}

#[test]
fn a_max_reduction_binds_its_operator() {
    let max = ReduceMax::new("pool.max", 64).expect("合法");
    assert_eq!(max.output().dims(), []);
    assert_eq!(max.spca().ops(), [ReduceMax::OP]);
    assert!(max.bounds().expect("合法").is_some());
}

#[test]
fn a_zero_length_reduction_is_rejected() {
    assert_eq!(ReduceSum::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(ReduceMax::new("x", 0).unwrap_err().code(), Code::Invalid);
}

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
    assert_eq!(
        floor,
        Bounds::of_ctx(Embedding::OP, 8, 1).expect("同一份语义")
    );
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

#[test]
fn a_matmul_works_as_a_trait_object() {
    let layer: &dyn Contract = &MatMul::new("proj", 16, 12).expect("合法");
    assert_eq!(layer.output().dims(), [16, 12]);
    assert!(layer.bounds().expect("合法").is_some());
}

/// 单轴层的通用断言：绑定的算子名、以及与语义表逐项相等的下界。
fn assert_elementwise<L: Contract>(layer: &L, op: &str, len: usize) {
    assert_eq!(layer.spca().ops(), [op]);
    let floor = layer.bounds().expect("单轴形状合法").expect("有算子");
    assert_eq!(
        floor,
        Bounds::of_ctx(op, len, 1).expect("同一份语义"),
        "{op} 的下界来自算子语义表"
    );
}

/// 除 / 减 / 取大 / 取小 / 指数 / 对数六个算子此前在 `op` 注册表里有、在层集里没有：
/// 元素级 API 的主体缺了一角。这条用例把它们钉住，同时验证与 `add_f32` / `gelu_f32` 同价骨架无关——
/// 层不写第二份公式，下界一律由各自的语义表给出。
#[test]
fn the_six_elementwise_operators_that_had_no_layer_now_have_one() {
    assert_elementwise(&Sub::new("blk.0.sub", 128).expect("合法"), Sub::OP, 128);
    assert_elementwise(&Div::new("blk.0.div", 128).expect("合法"), Div::OP, 128);
    assert_elementwise(&Max::new("blk.0.max", 128).expect("合法"), Max::OP, 128);
    assert_elementwise(&Min::new("blk.0.min", 128).expect("合法"), Min::OP, 128);
    assert_elementwise(&Exp::new("blk.0.exp", 128).expect("合法"), Exp::OP, 128);
    assert_elementwise(&Log::new("blk.0.log", 128).expect("合法"), Log::OP, 128);
}

/// 六个新层沿用同一条「零长度拒绝」口径，不留一个可以构造出空轴的后门。
#[test]
fn a_zero_length_elementwise_tensor_is_rejected() {
    assert_eq!(Sub::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Div::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Max::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Min::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Exp::new("x", 0).unwrap_err().code(), Code::Invalid);
    assert_eq!(Log::new("x", 0).unwrap_err().code(), Code::Invalid);
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

/// 反量化层：三档各绑自己的算子；写侧与算术量**逐位相同**——反量化不改输出格式（仍是 f32），
/// 只改读侧的打包宽度。i2 / i4 的读恰好是满位打包（`len·2/8` 与 `len·4/8`）。
///
/// i8 这一档的读按 f32 宽度出，而非 1 字节/元素：`crates/op/ops/dequant_i8.k` 把 `q` 声明成
/// `const float *`，声明表也没有 `dt`，两处自洽但语义上不是「读 8 位码」。故本用例如实锁住
/// **现状**，并在注释里点名——修 op 侧模板时应连带把这条断言改成 `len·1`。
#[test]
fn the_dequant_layer_undoes_each_packed_width_at_its_own_cost() {
    let len = 32;
    let i2 = Dequant::new("d", len, Precision::Int2).expect("合法");
    let i4 = Dequant::new("d", len, Precision::Int4).expect("合法");
    let i8 = Dequant::new("d", len, Precision::Int8).expect("合法");
    assert_eq!(i2.spca().ops(), ["dequant_i2"]);
    assert_eq!(i4.spca().ops(), ["dequant_i4"]);
    assert_eq!(i8.spca().ops(), ["dequant_i8"]);

    let read = |layer: &Dequant| layer.bounds().expect("合法").expect("有算子").read();
    assert_eq!(read(&i2), (len * 2 / 8) as u64, "2 位码满位打包");
    assert_eq!(read(&i4), (len * 4 / 8) as u64, "4 位码满位打包");
    assert_eq!(
        read(&i8),
        (len * 4) as u64,
        "现状：i8 模板读 float（见用例文档）"
    );
    assert!(read(&i2) < read(&i4), "位宽更窄，读更少");
    assert!(read(&i4) < read(&i8), "位宽更宽，读更多");

    let floor = i2.bounds().expect("合法").expect("有算子");
    for other in [&i4, &i8] {
        let probe = other.bounds().expect("合法").expect("有算子");
        assert_eq!(probe.write(), floor.write(), "输出一律 f32");
        assert_eq!(probe.flops(), floor.flops(), "反量化的算术量与位宽无关");
    }
    assert!(Dequant::new("d", 0, Precision::Int4).is_err(), "零长度拒绝");
}

/// f32 没有可还原的编码：这一档不是「下界为零」，是没有这一步。
#[test]
fn an_f32_dequant_declares_nothing_to_undo() {
    let layer = Dequant::new("d", 16, Precision::F32).expect("仍是合法声明");
    assert!(layer.spca().ops().is_empty(), "没有内核就没有算子名");
    let error = layer.bounds().unwrap_err();
    assert_eq!(error.code(), Code::Unsupported);
    let message = error.to_string();
    assert!(message.contains("f32"), "{message}");
    assert!(message.contains("没有可还原的编码"), "{message}");
}

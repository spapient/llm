//! typed IR → Lower / 设备 dispatch 桥的白盒单测：`kind` 路由、`source` 的 `.k` 发射与签名
//! 解析、`lower` 的派发接缝。不依赖任何真机——设备路径的文本产出在 `device` 的 backends 测试里
//! 用 Metal 再兜一道。

use super::{kind, lower, source, REDUCE_CHUNK, REDUCE_FUNC};
use crate::backend::Lower;
use cutile::ir::{Access, BinaryOp, Expr, Kernel, Parameter, ScalarType, ValueType};
use cutile::prof::Emit;
use diagnostic::Code;
use op::decl::Grid;
use op::model::{Buf, Sample, Values};
use op::sig::CType;

fn read(name: &str) -> Parameter {
    Parameter {
        name: name.to_string(),
        ty: ValueType::Buf {
            element: ScalarType::F32,
            access: Access::Read,
        },
    }
}

fn write(name: &str) -> Parameter {
    Parameter {
        name: name.to_string(),
        ty: ValueType::Buf {
            element: ScalarType::F32,
            access: Access::ReadWrite,
        },
    }
}

fn len(name: &str) -> Parameter {
    Parameter {
        name: name.to_string(),
        ty: ValueType::Scalar(ScalarType::Usize),
    }
}

fn copy_kernel() -> Kernel {
    Kernel::map(
        "copy",
        vec![read("a"), write("y"), len("N")],
        "i",
        "N",
        "y",
        Expr::Load("a".to_string()),
    )
}

fn sum_kernel() -> Kernel {
    Kernel::reduce(
        "sum",
        vec![read("a"), write("y"), len("N")],
        "i",
        "N",
        "y",
        0,
        Expr::Load("a".to_string()),
    )
}

fn sample(len: i64) -> Sample {
    Sample {
        m: 1,
        n: len as usize,
        scalars: vec![("N", len)],
        bufs: vec![
            Buf {
                name: "a",
                values: Values::F32(vec![1.0; len as usize]),
            },
            Buf {
                name: "y",
                values: Values::F32(vec![0.0; len as usize]),
            },
        ],
    }
}

#[test]
fn map_with_single_scalar_routes_to_pointwise() {
    assert_eq!(kind(&copy_kernel()).unwrap(), Lower::Pointwise);
}

#[test]
fn map_with_extra_scalar_routes_to_serial() {
    let kernel = Kernel::map(
        "scale",
        vec![read("a"), write("y"), len("N"), Parameter {
            name: "s".to_string(),
            ty: ValueType::Scalar(ScalarType::F32),
        }],
        "i",
        "N",
        "y",
        Expr::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expr::Load("a".to_string())),
            right: Box::new(Expr::Scalar("s".to_string())),
        },
    );
    assert_eq!(kind(&kernel).unwrap(), Lower::Loop(Grid { dims: vec![] }));
}

#[test]
fn plain_sum_reduce_routes_to_chunk_reduce() {
    assert_eq!(
        kind(&sum_kernel()).unwrap(),
        Lower::ChunkReduce {
            chunk: REDUCE_CHUNK,
            func: REDUCE_FUNC.to_string(),
        }
    );
}

#[test]
fn dot_reduce_with_two_inputs_routes_to_serial() {
    let kernel = Kernel::reduce(
        "dot",
        vec![read("a"), read("b"), write("y"), len("N")],
        "i",
        "N",
        "y",
        0,
        Expr::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expr::Load("a".to_string())),
            right: Box::new(Expr::Load("b".to_string())),
        },
    );
    assert_eq!(kind(&kernel).unwrap(), Lower::Loop(Grid { dims: vec![] }));
}

#[test]
fn map_emit_k_round_trips_through_signature() {
    let (text, signature) = source(&copy_kernel(), "").unwrap();
    assert_eq!(signature.op(), "copy");
    assert_eq!(signature.params().len(), 3);

    let [a, y, n] = signature.params() else {
        panic!("copy 应有三个形参");
    };
    assert_eq!((a.name.as_str(), a.ctype, a.mutable), ("a", CType::F32, false));
    assert_eq!((y.name.as_str(), y.ctype, y.mutable), ("y", CType::F32, true));
    assert_eq!((n.name.as_str(), n.ctype), ("N", CType::Int));
    assert_eq!(signature.output(), Some("y"));

    assert!(
        text.contains("void cutile_copy_kernel(const float *a, float *y, int N)"),
        "{text}"
    );
    assert!(text.contains("y[i] = a[i];"), "{text}");
}

#[test]
fn reduce_source_prepends_preamble() {
    let (text, signature) = source(&sum_kernel(), "/* pre */").unwrap();
    assert_eq!(signature.op(), "sum");
    assert!(text.starts_with("/* pre */void cutile_sum_kernel("), "{text}");
    assert!(text.contains("y[0] += a[i];"), "{text}");
}

#[test]
fn lower_reports_unsupported_for_host_c_backend() {
    let error = lower(Emit::C, &copy_kernel(), "", &sample(4))
        .expect_err("host C 后端没有生产态 lowering");
    assert_eq!(error.code(), Code::Unsupported);
}

//! 稠密计算核：矩阵乘族（`gemm` / `gemv` / `matmul`）、激活（`embed`/`rmsnorm`/`rope`/`softmax`）、归一化（`norm`）、
//! 反量化（`dequant`）、逐元素（`elementwise` / `unary`）、归约（`reduce`）与 SIMD/SME 通道（`fast`）。

// 不稳定特性一律挂在 `cfg(nightly)` 上（由本 crate 的 `build.rs` 按 `rustc --version` 判定）：
// nightly 上走手写车道，stable 上走 `gemv/simd_fallback.rs` 的等价退路，两边的能力差异只在
// 「向量化没向量化」，不在语义。这样 `rust-toolchain.toml` 不必把整个工作区钉在某个 nightly。
//
// 三个特性各自还有额外的门：
// - `aarch64_unstable_target_feature`：放行 `#[target_feature(enable = "sme")]`。
// - `stdarch_aarch64_feature_detection`：放行运行时探测 `is_aarch64_feature_detected!("sme")`。
//   上面这两个特性名**只在 aarch64 上存在**：x86_64 等其它目标根本不认识，直接声明会 E0635
//   （unknown feature）。特性门不是惰性的，名字在编译目标上不认识就是硬错误，因此还要按架构门控。
// - `portable_simd`：`std::simd`，用在 `fast::sme::pack_a` 的分块转置（只存在于 aarch64）与
//   `gemv::simd` 的解包点积。后者**不按目标门控**，因为它是架构无关的 16 车道源码：aarch64 上
//   落成 NEON、x86_64 上落成 SSE2 基线，两个目标编译的是同一份算术。

pub mod dequant {
    //! 打包反量化骨架：每字节 `CPB` 个码，`y = scale · (code − zero)`。

    use crate::model::{Buf, Cost, Sample, ShapeCtx, Values};
    use diagnostic::{Code, Error};

    // 单轴定义与形状投影复用 unary 骨架。
    pub use crate::unary::{AXES, project_shape};

    /// 反量化标度（2 的负幂网格，f32 精确）。
    pub const SCALE: f32 = prof::epilogue::QUANT_SCALE;

    /// 打包格式的前置约束：输出长度必须是输入字节数的 `cpb` 倍。
    fn check_len(q: &[u8], out: &[f32], cpb: usize) -> Result<(), Error> {
        if out.len() != q.len() * cpb {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不符：q {} 字节、out {} 元素", q.len(), out.len()),
            ));
        }
        Ok(())
    }

    /// 仿射参数校验：标度不能为零（否则整组塌成常数、失去量化口径），零点必须落在码域内。
    ///
    /// 码域是 `0..2^width`：零点是码点到有符号值的平移量，越出码域意味着存在无法表示的码点，
    /// 属于配置错误而不是数值极端。打包格式传自身位宽（i2=2、i4=4），逐元素格式传 16 档。
    ///
    /// # Errors
    ///
    /// 标度为零、零点不是有限值，或零点不在 `0..2^width` 内时返回 [`Code::Invalid`]。
    pub fn check_affine(scale: f32, zero: f32, width: usize) -> Result<(), Error> {
        if scale == 0.0 || !scale.is_finite() {
            return Err(Error::new(Code::Invalid, "反量化标度必须是非零有限值"));
        }
        if !zero.is_finite() {
            return Err(Error::new(Code::Invalid, "反量化零点必须是有限值"));
        }
        let domain = 1u32 << width;
        if !(0.0..domain as f32).contains(&zero) {
            return Err(Error::new(
                Code::Invalid,
                format!("零点 {zero} 越出码域 0..{domain}（{width} 位码）"),
            ));
        }
        Ok(())
    }

    /// 标量代价：每元素一次减法一次乘法；读打包字节（`n / cpb` 个），写 `n` 个 f32。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost_packed(shape: &ShapeCtx, cpb: usize) -> Result<Cost, Error> {
        let n = shape
            .get('n')
            .ok_or_else(|| Error::new(Code::Invalid, "dequant 形状缺轴 `n`"))?;
        Ok(Cost {
            flops: (n as f64) * 2.0,
            read: (n / cpb) as u64,
            write: (n as u64) * 4,
        })
    }

    /// 确定性样本：`q`（打包字节，`len / cpb` 个）、`y`（`len` 个 f32）。
    pub fn sample_packed(len: usize, cpb: usize) -> Sample {
        let bytes = len / cpb;
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            bufs: vec![
                Buf {
                    name: "q",
                    values: Values::U8((0..bytes).map(|i| (i * 37 % 251) as u8).collect()),
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
    }

    /// 解包字节内第 `slot` 个码，按对称零点回退成有符号值。
    pub fn unpack(byte: u8, slot: usize, cpb: usize, zero: f32) -> f32 {
        let width = crate::quant::width_cpb(cpb);
        let mask = (1u8 << width) - 1;
        ((byte >> (slot * width)) & mask) as f32 - zero
    }

    /// 标量参考：逐字节解 `cpb` 个码，与内核同一公式。
    pub fn reference_packed(q: &[u8], cpb: usize, zero: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(q.len() * cpb);
        for byte in q {
            for slot in 0..cpb {
                out.push(SCALE * unpack(*byte, slot, cpb, zero));
            }
        }
        out
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输出长度不是输入字节数的 `cpb` 倍时返回说明。
    pub fn forward_packed(
        scale: f32,
        zero: f32,
        q: &[u8],
        out: &mut [f32],
        cpb: usize,
    ) -> diagnostic::Result<()> {
        check_len(q, out, cpb)?;
        check_affine(scale, zero, crate::quant::width_cpb(cpb))?;
        for (byte, chunk) in q.iter().zip(out.chunks_mut(cpb)) {
            for (slot, place) in chunk.iter_mut().enumerate() {
                *place = scale * unpack(*byte, slot, cpb, zero);
            }
        }
        Ok(())
    }

    /// 生成一份「打包反量化」精度变体的完整公开面：`SCALE`/`ZERO` 常量、代价/样本/参考/
    /// `forward` 入口、[`annotation::model`] 别名与测试挂载。`dequant_i2`/`dequant_i4` 只差一行
    /// [`crate::quant::Row`]；新增一档打包反量化加一次调用即可。
    #[macro_export]
    macro_rules! dequant_packed {
        ($name:ident, $alias:ident, $op:tt, $neg:expr, $nonfinite:expr, $row:ident) => {
            /// 反量化标度（2 的负幂网格，f32 精确）。
            pub const SCALE: f32 = $crate::dequant::SCALE;

            /// 打包对称零点（取自配置表 `$row`）。
            pub const ZERO: f32 = $row.zero as f32;

            // 单轴定义与形状投影复用打包骨架。
            pub use $crate::dequant::{AXES, project_shape};

            /// 标量代价：每元素一次减法一次乘法；读打包字节，写 `n` 个 f32。
            pub fn cost(
                shape: &$crate::model::ShapeCtx,
            ) -> diagnostic::Result<$crate::model::Cost> {
                $crate::dequant::cost_packed(shape, $row.cpb)
            }

            /// 确定性样本：`q`（打包字节）、`y`（`n` 个 f32）。
            pub fn sample(len: usize, _: usize) -> $crate::model::Sample {
                $crate::dequant::sample_packed(len, $row.cpb)
            }

            /// 标量参考：逐字节解包，与内核同一公式。
            pub fn reference(sample: &$crate::model::Sample) -> diagnostic::Result<Vec<f32>> {
                let q = sample.u8("q")?;
                Ok($crate::dequant::reference_packed(
                    q,
                    $row.cpb,
                    $row.zero as f32,
                ))
            }

            /// 内核本体：结果写进调用方给的缓冲。
            pub fn forward(scale: f32, zero: f32, q: &[u8], out: &mut [f32]) -> diagnostic::Result<()> {
                $crate::dequant::forward_packed(scale, zero, q, out, $row.cpb)
            }

            #[doc = concat!("`", $op, "` 的标记类型：位宽与零点见配置表，骨架见 `crate::dequant`。")]
            pub struct $name;

            // 基座契约（ADR 0050）：转发本模块的骨架入口；粒度直接取配置表的每字节码数。
            $crate::operator_via!(
                @impl $name,
                $op,
                1,
                (4096, 4096),
                false,
                $crate::quant::$row.cpb as u32,
                self::AXES,
                self::project_shape,
                self::cost,
                self::sample,
                self::reference
            );

            #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = $op)]
            #[doc = concat!("`", $op, "` 的语义别名。")]
            pub type $alias = $name;

            #[cfg(test)]
            mod tests {
                use super::*;
                $crate::dequant::test_support::dequant_packed_tests!(
                    sample,
                    reference,
                    forward,
                    SCALE,
                    ZERO,
                    $row.cpb,
                    (1u32 << $crate::quant::width_cpb($row.cpb)) as f32,
                    $neg,
                    $nonfinite
                );
            }
        };
    }

    #[cfg(test)]
    mod test_support {
        include!("../tests/kernel/dequant_test_support.rs");
    }

    pub mod dequant_i2 {
        //! 2 位反量化：打包字节 `q`（每字节四个 2 位码），`y = scale * (code - 2)`。

        use crate::quant::I2;

        crate::dequant_packed!(DequantI2, DequantI2F32, "dequant_i2", -0.5, f32::NAN, I2);
    }

    pub mod dequant_i4 {
        //! 4 位反量化：打包字节 `q`（每字节两个 nibble），`y = scale * (nibble - 8)`。

        use crate::quant::I4;

        crate::dequant_packed!(
            DequantI4,
            DequantI4F32,
            "dequant_i4",
            -1.0,
            f32::INFINITY,
            I4
        );
    }

    pub mod i8 {
        //! 8 位反量化：`y[n] = scale * (q[n] - zero)`。
        use crate::dequant;
        use crate::model::{Buf, Cost, Sample, ShapeCtx, Values, cost_n};
        use diagnostic::{Code, Error};

        // 单轴定义复用 unary 骨架。
        pub use crate::unary::AXES;

        /// 样本固定的反量化参数。
        pub const SCALE: f32 = 0.25;
        /// 样本固定的零点。
        pub const ZERO: f32 = 8.0;

        // 形状投影复用 unary 骨架。
        pub use crate::unary::project_shape;

        /// 标量代价：每元素一次减法一次乘法，读一次写一次。
        ///
        /// # Errors
        ///
        /// 形状缺轴时返回说明。
        pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
            cost_n(shape, "dequant", 2.0, 1)
        }

        /// 确定性样本：`q`（f32 承载的量化整数）、`y`，标量 `scale`/`zero`。
        pub fn sample(len: usize, _: usize) -> Sample {
            Sample {
                m: len,
                n: len,
                scalars: vec![("N", len as i64)],
                bufs: vec![
                    Buf {
                        name: "q",
                        values: Values::F32((0..len).map(|i| (i % 16) as f32).collect()),
                    },
                    Buf {
                        name: "y",
                        values: Values::F32(vec![0.0; len]),
                    },
                ],
            }
        }

        /// 标量参考：与内核同一公式。
        pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            let q = sample.f32("q")?;
            Ok(q.iter().map(|value| SCALE * (value - ZERO)).collect())
        }

        /// 内核本体：结果写进调用方给的缓冲。
        ///
        /// # Errors
        ///
        /// 输入与输出等长、标度非零有限、零点在 `0..=15` 码域内，否则返回说明。
        pub fn forward(scale: f32, zero: f32, q: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
            if q.len() != out.len() {
                return Err(Error::new(
                    Code::Invalid,
                    format!("缓冲长度不一致：q {}、out {}", q.len(), out.len()),
                ));
            }
            dequant::check_affine(scale, zero, 4)?;
            for (out, q) in out.iter_mut().zip(q) {
                *out = scale * (q - zero);
            }
            Ok(())
        }

        /// 标记类型：承载 `#[model]` 语义。
        pub struct DequantI8;

        // 基座契约（ADR 0050）：本模块自带具体入口，直接转发。
        crate::operator_here!(
            DequantI8,
            "dequant_i8",
            1,
            (4096, 4096),
            false,
            self::reference
        );

        /// 反量化的具体语义别名。
        #[annotation::model(name = "dequant_i8", shape = [4096, 4096], dtype = "f32", skeleton = "dequant")]
        pub type DequantI8F32 = DequantI8;

        #[cfg(test)]
        mod tests {
            include!("../tests/kernel/dequant_i8_tests.rs");
        }
    }
}
pub mod elementwise {
    //! 逐元素二元骨架：`y[n] = op(a[n], b[n])`。

    use crate::formula::FormulaOwner;
    use crate::model::{Buf, Cost, Sample, ShapeCtx, Values};
    use diagnostic::{Code, Error};

    // 单轴定义与形状投影复用一元骨架（与 dequant / softmax / rmsnorm / rope / scan 同一模式）。
    pub use crate::unary::{AXES, project_shape};

    /// 逐元素二元算子随语义变化的那部分。
    pub trait Binary: FormulaOwner {
        /// 逐元素运算：两个输入值，一个输出值。
        fn apply(x: f32, y: f32) -> f32;

        /// 第一路输入的确定性取值：默认线性夹具 `−1.0 + 0.5·i`；需要不同取值时覆盖。
        fn input_a(len: usize) -> Vec<f32> {
            crate::model::linear(len, 0.5, -1.0)
        }

        /// 第二路输入的确定性取值：默认线性夹具 `0.5 + 0.25·i`；需要不同取值时覆盖。
        fn input_b(len: usize) -> Vec<f32> {
            crate::model::linear(len, 0.25, 0.5)
        }
    }

    /// 标量代价：完全由表达式与操作数位宽推出，本函数不含流量常数。
    pub fn cost<B: Binary>(shape: &ShapeCtx) -> Result<Cost, Error> {
        B::formula()?
            .cost(shape.extents())
            .map_err(|error| error.with("nn::elementwise::cost"))
    }

    /// 确定性样本：第一个参数是元素数（轴 `n`），第二个参数逐元素算子不用。
    pub fn sample<B: Binary>(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            bufs: vec![
                Buf {
                    name: "a",
                    values: Values::F32(B::input_a(len)),
                },
                Buf {
                    name: "b",
                    values: Values::F32(B::input_b(len)),
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
    }

    /// 标量参考：逐元素应用 [`Binary::apply`]，不依赖任何设备。
    pub fn reference<B: Binary>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let a = sample.f32("a")?;
        let b = sample.f32("b")?;
        if a.len() != b.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("两个输入长度不一致：{} 与 {}", a.len(), b.len()),
            ));
        }
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| B::apply(*x, *y))
            .collect())
    }

    /// 内核本体：结果写进调用方给的缓冲，不自带分配。
    ///
    /// # Errors
    ///
    /// 两个输入等长、且与输出等长，否则返回说明。
    pub fn forward<B: Binary>(a: &[f32], b: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        if a.len() != b.len() || a.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲长度不一致：a {}、b {}、out {}",
                    a.len(),
                    b.len(),
                    out.len()
                ),
            ));
        }
        // 三个切片 zip：连续内存、无下标边界检查，编译器最容易自动向量化。
        for ((out, x), y) in out.iter_mut().zip(a).zip(b) {
            *out = B::apply(*x, *y);
        }
        Ok(())
    }

    /// 生成一份 f32 二元逐元素算子的公开面：标记类型、[`Binary`] 实现、[`annotation::model`] 别名、
    /// `forward` 入口与测试挂载。`add` / `mul` / `residual_add` 只差一行 `apply`（可选样本夹具）；
    /// 新增一种二元运算加一次调用即可。
    #[macro_export]
    macro_rules! binary_f32 {
        (
            $name:ident, $alias:ident, $op:tt, $kind:ident,
            $apply:expr
            $(, $input_a:expr, $input_b:expr)?
        ) => {
            #[doc = concat!("`", $op, "` 的标记类型：逐元素二元运算见 `apply`，骨架见 `crate::elementwise`。")]
            pub struct $name;

            impl $crate::formula::FormulaOwner for $name {
                const EXPR: &'static str = "n,n,n->n";

                const OPERANDS: &'static [$crate::einsum::OperandSpec] =
                    &$crate::einsum::f32_binary_operands();

                const FRAME: &'static str = "nn::elementwise::formula";

                fn cache() -> &'static $crate::einsum::FormulaCache {
                    $crate::formula_cell!()
                }
            }

            impl $crate::elementwise::Binary for $name {
                fn apply(x: f32, y: f32) -> f32 {
                    const APPLY: fn(f32, f32) -> f32 = $apply;
                    APPLY(x, y)
                }

                $( fn input_a(len: usize) -> Vec<f32> {
                    const INPUT: fn(usize) -> Vec<f32> = $input_a;
                    INPUT(len)
                }

                fn input_b(len: usize) -> Vec<f32> {
                    const INPUT: fn(usize) -> Vec<f32> = $input_b;
                    INPUT(len)
                } )?
            }

            // 基座契约（ADR 0050）：只转发族骨架，不写第二份算术。
            $crate::operator_via!(
                $name,
                $op,
                2,
                (4096, 4096),
                true,
                super,
                $crate::elementwise::AXES
            );

            #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = "elementwise")]
            #[doc = concat!("`", $op, "` 的语义别名。")]
            pub type $alias = $name;

            /// 内核入口：结果写进调用方给的缓冲。
            ///
            /// # Errors
            ///
            /// 三个缓冲等长，否则返回说明。
            pub fn forward(a: &[f32], b: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
                $crate::elementwise::forward::<$name>(a, b, out)
            }

            #[cfg(test)]
            mod tests {
                use super::*;
                $crate::elementwise::test_support::$kind!($name, forward);
            }
        };
    }

    #[cfg(test)]
    mod test_support {
        include!("../tests/kernel/elementwise_test_support.rs");
    }

    // 声明式算子表：同一副二元骨架的七个算子，每个算子一个 `pub mod` 保持 `nn::<name>::forward`
    // 路径不变（`forward` 同名，必须隔在各自的模块里）。新增二元运算在此补一段。

    pub mod add_f32 {
        //! 逐元素加法：`y[n] = a[n] + b[n]`。

        crate::binary_f32!(Add, AddF32, "add_f32", elementwise_binary_tests, |x, y| x
            + y);
    }

    pub mod sub_f32 {
        //! 逐元素减法：`y[n] = a[n] - b[n]`。

        crate::binary_f32!(Sub, SubF32, "sub_f32", elementwise_binary_tests, |x, y| x
            - y);
    }

    pub mod mul_f32 {
        //! 逐元素乘法：`y[n] = a[n] * b[n]`。

        crate::binary_f32!(Mul, MulF32, "mul_f32", elementwise_binary_tests, |x, y| x
            * y);
    }

    pub mod div_f32 {
        //! 逐元素除法：`y[n] = a[n] / b[n]`。

        crate::binary_f32!(Div, DivF32, "div_f32", elementwise_div_tests, |x, y| x / y);
    }

    pub mod max_f32 {
        //! 逐元素取大：`y[n] = max(a[n], b[n])`。

        crate::binary_f32!(Max, MaxF32, "max_f32", elementwise_binary_tests, |x, y| x
            .max(y));
    }

    pub mod min_f32 {
        //! 逐元素取小：`y[n] = min(a[n], b[n])`。

        crate::binary_f32!(Min, MinF32, "min_f32", elementwise_binary_tests, |x, y| x
            .min(y));
    }

    pub mod residual_add_f32 {
        //! 残差相加：`y[n] = a[n] + b[n]`。

        crate::binary_f32!(
            Residual,
            ResidualAddF32,
            "residual_add_f32",
            elementwise_binary_tests,
            |x, y| x + y,
            |len| (0..len).map(|i| (i as f32) * 0.5).collect(),
            |len| (0..len).map(|i| (i as f32) * 0.25).collect()
        );
    }
}
pub mod embed {
    //! Embedding 行查表：`y[n, :] = table[idx[n], :]`（行向量 gather）。
    use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
    use diagnostic::{Code, Error};

    /// 权重表行数（闭集内的固定档）。
    pub const VOCAB: usize = 64;

    /// 两条基本轴：token 轴 `n`（输入）、行宽轴 `d`（输出）。
    pub const AXES: &[AxisSpec] = &[
        AxisSpec {
            role: AxisRole::Input,
            label: 'n',
        },
        AxisSpec {
            role: AxisRole::Output,
            label: 'd',
        },
    ];

    /// `(m, n)` 即 `(token 数 n, 行宽 d)`。
    ///
    /// # Errors
    ///
    /// 当前实现不会失败。
    pub fn project_shape(tokens: usize, dim: usize) -> Result<ShapeCtx, Error> {
        ShapeCtx::new([('n', tokens), ('d', dim)])
    }

    /// 标量代价：每个 token 搬一行 `D`（读表一行 + 写输出一行），下标转换按 1 次/token 估算。
    ///
    /// 常驻表按被访问的行数计价，不把整张词表读一遍（表流量口径属于代价模型，层不重复推）。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        let n = shape
            .get('n')
            .ok_or_else(|| Error::new(Code::Invalid, "embed 形状缺轴 `n`"))?;
        let d = shape
            .get('d')
            .ok_or_else(|| Error::new(Code::Invalid, "embed 形状缺轴 `d`"))?;
        let elements = (n as u64) * (d as u64);
        Ok(Cost {
            flops: n as f64,
            read: elements * 4,
            write: elements * 4,
        })
    }

    /// 确定性样本：`idx`（f32 承载的整数下标，长度 n）、`table`（长度 `V·D`）、`y`（长度 `n·D`）。
    pub fn sample(tokens: usize, dim: usize) -> Sample {
        let table = (0..VOCAB * dim)
            .map(|idx| (idx % VOCAB) as f32 * 0.5)
            .collect();
        Sample {
            m: tokens,
            n: dim,
            scalars: vec![("N", tokens as i64), ("V", VOCAB as i64), ("D", dim as i64)],
            bufs: vec![
                Buf {
                    name: "idx",
                    values: Values::F32((0..tokens).map(|i| (i % VOCAB) as f32).collect()),
                },
                Buf {
                    name: "table",
                    values: Values::F32(table),
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; tokens * dim]),
                },
            ],
        }
    }

    /// 标量参考：与内核同一行 gather 公式。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let idx = sample.f32("idx")?;
        let table = sample.f32("table")?;
        gather(idx, table, sample.n)
    }

    /// 共享行查表：参考与内核同一套；`dim` 为行宽，表长必须是 `dim` 的整数倍，行数由此推出。
    fn gather(idx: &[f32], table: &[f32], dim: usize) -> diagnostic::Result<Vec<f32>> {
        if dim == 0 {
            return Err(Error::new(Code::Invalid, "embed 行宽不能为零"));
        }
        if !table.len().is_multiple_of(dim) {
            return Err(Error::new(
                Code::Invalid,
                format!("embedding 表长 {} 不是行宽 {dim} 的整数倍", table.len()),
            ));
        }
        let rows = table.len() / dim;
        let mut out = Vec::with_capacity(idx.len() * dim);
        for value in idx {
            let k = *value as usize;
            if k >= rows {
                return Err(Error::new(
                    Code::Invalid,
                    format!("embedding 下标 {k} 越界（词表 {rows}）"),
                ));
            }
            let row = k * dim;
            out.extend_from_slice(&table[row..row + dim]);
        }
        Ok(out)
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// 行宽 `D` 由输出长度与 token 数推出：`out.len() == idx.len()·D`；表长必须是 `D` 的整数倍。
    ///
    /// # Errors
    ///
    /// 缓冲长度不自洽、表长不符或下标越界时返回说明。
    pub fn forward(idx: &[f32], table: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        let tokens = idx.len();
        if tokens == 0 {
            return Err(Error::new(Code::Invalid, "embed 索引为空"));
        }
        if !out.len().is_multiple_of(tokens) {
            return Err(Error::new(
                Code::Invalid,
                format!("输出长度 {} 不能按 token 数 {tokens} 整除出行宽", out.len()),
            ));
        }
        let dim = out.len() / tokens;
        let gathered = gather(idx, table, dim)?;
        out.copy_from_slice(&gathered);
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Embed;

    /// Embedding 的具体语义别名。
    #[annotation::model(name = "embed_f32", shape = [8, 64], dtype = "f32", skeleton = "embed")]
    pub type EmbedF32 = Embed;

    crate::operator_here!(Embed, "embed_f32", 2, (8, 64), true, reference);

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel/embed.rs");
    }
}
pub mod epilogue {
    //! 写回尾块执行：把 `prof::epilogue::Step` 闭集按声明顺序施加到 f32 瓦片，算术只转发 `gemm` 的拥有方。
    //!
    //! 六步闭集来自 [`prof::epilogue::Step`]：`bias` / `act` / `residual` / `scale` / `quant` / `store`。
    //! 本模块不持有任何算术——激活逐个取 [`crate::unary`] 的标量核，量化用 [`crate::dequant::SCALE`]
    //! 做对称 round-trip，残差与缩放直接就地累加。契约层（`spec`）只声明"有哪些步"，执行层在这里
    //! 决定"每一步落在哪个既有内核上"。

    use diagnostic::{Code, Error};
    use prof::atom::{ActKind, DType};
    use prof::epilogue::Step;

    /// 按 `steps` 的顺序把写回尾块施加到 `out`（行主序，列宽 `cols`）。
    ///
    /// * `bias`：长度 `cols` 的列偏置，按列广播；缺省跳过。
    /// * `residual`：长度等于 `out` 的残差输入；缺省跳过。
    /// * `act:<kind>`：逐元素施加 `crate::{kind}_f32` 的标量核，算术归 [`crate::unary`]。
    /// * `scale:<α>`：逐元素乘 `α`。
    /// * `quant:<i8|i4|i2>`：量化到对称整数网格再反量化（步长 [`crate::dequant::SCALE`]，误差 ≤ 半步长）。
    /// * `store`：无副作用标记，本层空操作。
    ///
    /// # Errors
    ///
    /// `bias` / `residual` 长度不符、`quant` 位宽超出 `i8|i4|i2` 时返回 [`Code::Invalid`]。
    pub fn apply(
        steps: &[Step],
        out: &mut [f32],
        cols: usize,
        bias: Option<&[f32]>,
        residual: Option<&[f32]>,
    ) -> diagnostic::Result<()> {
        for step in steps {
            apply_step(step, out, cols, bias, residual)?;
        }
        Ok(())
    }

    /// 单步分派：前三个是几何/标量步（`bias` / `residual` / `scale`），其余交给 [`apply_tail`]，
    /// 把六臂 match 压成两段、每段 ≤5 圈复杂度。
    fn apply_step(
        step: &Step,
        out: &mut [f32],
        cols: usize,
        bias: Option<&[f32]>,
        residual: Option<&[f32]>,
    ) -> diagnostic::Result<()> {
        match step {
            Step::Bias => add_bias(out, cols, bias),
            Step::Residual => add_residual(out, residual),
            Step::Scale(alpha) => scale_tile(out, *alpha),
            _ => apply_tail(step, out),
        }
    }

    /// 后三个是激活 / 量化 / 无副作用步；`_` 臂不可达（已由 [`apply_step`] 收走），防御性报错。
    fn apply_tail(step: &Step, out: &mut [f32]) -> diagnostic::Result<()> {
        match step {
            Step::Act(kind) => {
                activate(out, *kind);
                Ok(())
            }
            Step::Quant(dtype) => quant_round_trip(out, *dtype),
            Step::Store => Ok(()),
            _ => Err(Error::new(Code::Invalid, "写回步骤分派不可达")),
        }
    }

    /// 列偏置广播：`out[r][c] += bias[c]`。缺省即空操作。
    fn add_bias(out: &mut [f32], cols: usize, bias: Option<&[f32]>) -> diagnostic::Result<()> {
        match bias {
            Some(bias) => add_bias_cols(out, cols, bias),
            None => Ok(()),
        }
    }

    /// 有偏置的广播加：先校验列宽，再逐行累加。
    fn add_bias_cols(out: &mut [f32], cols: usize, bias: &[f32]) -> diagnostic::Result<()> {
        if bias.len() != cols {
            return Err(shape_err("bias", cols, bias.len()));
        }
        for row in out.chunks_mut(cols) {
            for (x, b) in row.iter_mut().zip(bias) {
                *x += *b;
            }
        }
        Ok(())
    }

    /// 残差加：`out[i] += residual[i]`。缺省即空操作。
    fn add_residual(out: &mut [f32], residual: Option<&[f32]>) -> diagnostic::Result<()> {
        let Some(residual) = residual else {
            return Ok(());
        };
        if residual.len() != out.len() {
            return Err(shape_err("residual", out.len(), residual.len()));
        }
        for (x, r) in out.iter_mut().zip(residual) {
            *x += *r;
        }
        Ok(())
    }

    /// 标量缩放：`out[i] *= alpha`。
    fn scale_tile(out: &mut [f32], alpha: f32) -> diagnostic::Result<()> {
        for x in out.iter_mut() {
            *x *= alpha;
        }
        Ok(())
    }

    /// 激活分派：按 [`SCALARS`] 取标量核，就地施加（不建副本、不另分配）。
    fn activate(out: &mut [f32], kind: ActKind) {
        let Some((_, apply)) = SCALARS.iter().find(|(have, _)| *have == kind) else {
            return;
        };
        for x in out.iter_mut() {
            *x = apply(*x);
        }
    }

    /// 一个标量激活核：`f32 -> f32`，与 [`crate::unary::Unary::apply`] 同签名。
    type Scalar = fn(f32) -> f32;

    /// 五个标量核的登记表：`ActKind` 的闭集成员与 `crate::unary` 的 `Unary::apply` 一一对上，
    /// 覆盖关系由 `epilogue_tests` 的 `every_act_kind_has_a_scalar_kernel` 钉住。
    const SCALARS: [(ActKind, Scalar); 5] = [
        (
            ActKind::Relu,
            <crate::relu_f32::ReluF32 as crate::unary::Unary>::apply,
        ),
        (
            ActKind::Gelu,
            <crate::gelu_f32::GeluF32 as crate::unary::Unary>::apply,
        ),
        (
            ActKind::Silu,
            <crate::silu_f32::SiluF32 as crate::unary::Unary>::apply,
        ),
        (
            ActKind::LeakyRelu,
            <crate::leaky_relu_f32::LeakyReluF32 as crate::unary::Unary>::apply,
        ),
        (
            ActKind::HardSigmoid,
            <crate::hard_sigmoid_f32::HardSigmoidF32 as crate::unary::Unary>::apply,
        ),
    ];

    /// 量化到对称整数网格再反量化（round-trip）：`out[i] = SCALE * round(out[i] / SCALE)`，饱和到
    /// `[-(2^(b-1)), 2^(b-1)-1]`。步长 [`crate::dequant::SCALE`]，最大误差 0.5 步长。
    fn quant_round_trip(out: &mut [f32], dtype: DType) -> diagnostic::Result<()> {
        let (min, max) = quant_bounds(dtype)?;
        for x in out.iter_mut() {
            let code = (*x / crate::dequant::SCALE).round().clamp(min, max);
            *x = crate::dequant::SCALE * code;
        }
        Ok(())
    }

    /// 对称网格边界：`i2 → [-2, 1]`、`i4 → [-8, 7]`、`i8 → [-128, 127]`；其余位宽拒绝。
    fn quant_bounds(dtype: DType) -> diagnostic::Result<(f32, f32)> {
        // 定义域是这条对称网格接受的三个精度；位宽从 `DType::bits` 取，不在此再写一遍字面量。
        let bits = match dtype {
            DType::I2 | DType::I4 | DType::I8 => dtype.bits(),
            _ => {
                return Err(Error::new(
                    Code::Invalid,
                    format!("`quant` 只接受 i8/i4/i2，`{}` 不是", dtype.name()),
                ));
            }
        };
        let half = 1u32 << (bits - 1);
        Ok((-(half as f32), (half - 1) as f32))
    }

    /// 形状不符的错误。
    fn shape_err(what: &str, expected: usize, got: usize) -> Error {
        Error::new(
            Code::Invalid,
            format!("{what} 长度不一致：期望 {expected}、实际 {got}"),
        )
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel/epilogue_tests.rs");
    }

    #[cfg(test)]
    mod device_tests {
        include!("../tests/kernel/epilogue_device_tests.rs");
    }
}
pub mod fast {
    //! 高吞吐 CPU GEMM：多线程 + A 列面板打包 + 8×8 寄存器分块。
    //! 语义同 [`crate::matmul::matmul_scalar`]，浮点累加顺序不同，按 IEEE 容差对拍（不做逐位相等）。

    use std::thread;

    use diagnostic::{Code, Error, Result};

    mod pack {
        //! 打包：把 A 的行块转成 `[p][MR]`、把 B 的列组转成 `[jblk][p][NR]` 面板。
        use std::thread;

        use super::{MR, NR};

        /// A 的每个 8 行块转置成 `[p][8]` 面板：微内核每条 `p` 只需两次连续向量加载。
        pub(super) fn pack_panels(a: &[f32], full: usize, k: usize) -> Vec<f32> {
            let mut apack = vec![0.0f32; full * k * MR];
            for blk in 0..full {
                let src = &a[blk * MR * k..(blk * MR + MR) * k];
                let dst = &mut apack[blk * k * MR..blk * k * MR + k * MR];
                for p in 0..k {
                    for r in 0..MR {
                        dst[p * MR + r] = src[r * k + p];
                    }
                }
            }
            apack
        }

        /// 一个列组面板的内层转置：`first` 起的 `cols` 列 × 全 `k` 行 → `dst[base..]` 的 `[q][NR]`。
        ///
        /// `transposed` 只决定源下标：
        /// - `false`（Nt）：`src[q*n + first + c]`，B 按 `[k][n]` 驻留，整行 `copy_from_slice`；
        /// - `true`（Tn）：`src[(first+c)*k + q]`，W 按 `[n][k]` 驻留，逐元素转置。
        ///
        /// 这是 B 打包的唯一算术：整段、分片与池化三种外层遍历只差列组下标，内层下标只有这一份。
        pub(super) fn pack_panel(
            src: &[f32],
            dst: &mut [f32],
            n: usize,
            k: usize,
            first: usize,
            base: usize,
            transposed: bool,
        ) {
            let cols = if first < n { NR.min(n - first) } else { 0 };
            for q in 0..k {
                let row = base + q * NR;
                if transposed {
                    for (c, slot) in dst[row..row + cols].iter_mut().enumerate() {
                        *slot = src[(first + c) * k + q];
                    }
                } else {
                    let src0 = q * n + first;
                    dst[row..row + cols].copy_from_slice(&src[src0..src0 + cols]);
                }
            }
        }

        /// B 打包成 `[jblk][p][NR]`：微内核每条 `p` 只需两次连续向量加载（行主序 B 是每 `p` 跨一条行）。
        pub(super) fn pack_b_f32_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize) {
            let npad = n.div_ceil(NR) * NR;
            debug_assert_eq!(bpack.len(), npad * k);
            for jb in 0..npad / NR {
                pack_panel(b, bpack, n, k, jb * NR, jb * k * NR, false);
            }
        }

        /// 并行 B 打包：按 8 列组切线程。`k·n` 小的时候串行更划算（线程创建比转置贵）。
        pub(super) fn pack_b_f32_par_into(bpack: &mut [f32], b: &[f32], n: usize, k: usize, t: usize) {
            let groups = n.div_ceil(NR);
            if t <= 1 || n * k < (1 << 16) {
                pack_b_f32_into(bpack, b, n, k);
                return;
            }
            let per = groups.div_ceil(t);
            let span = per * NR * k;
            thread::scope(|s| {
                for (gi, chunk) in bpack.chunks_mut(span).enumerate() {
                    s.spawn(move || {
                        let blocks = chunk.len() / (NR * k);
                        for bi in 0..blocks {
                            pack_panel(b, chunk, n, k, (gi * per + bi) * NR, bi * k * NR, false);
                        }
                    });
                }
            });
        }

        /// 池化打包的分片体：本槽负责的 8 列组按 `p` 步长轮转，写入 `[jblk][p][NR]`（零初始化覆盖补零列）。
        pub(super) fn b_groups(
            b: &[f32],
            buf: &mut [f32],
            n: usize,
            k: usize,
            npad: usize,
            p: usize,
            slot: usize,
        ) {
            let groups = npad / NR;
            let mut g = slot;
            while g < groups {
                pack_panel(b, buf, n, k, g * NR, g * k * NR, false);
                g += p;
            }
        }
    }
    mod pool {
        //! 职责：常驻 GEMM 线程池——任务只发布裸指针 + 一次闸门，spawn 成本归零。

        use std::any::Any;
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
        use std::sync::{Arc, Condvar, Mutex, OnceLock};
        use std::thread;

        /// 一次 GEMM 调用的几何：`npad` 是 `n` 对齐到微内核列数后的补零宽（f32 与 i8 共用）。
        #[derive(Clone, Copy)]
        pub(super) struct Shape {
            pub(super) m: usize,
            pub(super) n: usize,
            pub(super) k: usize,
            pub(super) npad: usize,
            pub(super) tn: bool,
        }

        /// 一次任务的四条数据指针：输入 A/B、打包面板、输出。一并发布，池只认这一个入口。
        #[derive(Clone, Copy)]
        pub(super) struct Bufs {
            pub(super) a: *const u8,
            pub(super) b: *const u8,
            pub(super) buf: *mut u8,
            pub(super) out: *mut u8,
        }

        pub(super) struct Job {
            pub(super) a: *const u8,
            pub(super) b: *const u8,
            /// 打包后的 B 面板（i8 或 f32，由 `stage`/`calc` 决定解释方式）。
            pub(super) buf: *mut u8,
            pub(super) out: *mut u8,
            pub(super) m: usize,
            pub(super) n: usize,
            pub(super) k: usize,
            pub(super) npad: usize,
            pub(super) tn: bool,
            pub(super) rows_per: usize,
            pub(super) p: usize,
            /// 打包阶段的分片体；类型差异只落在这两个函数里，池机制完全共享。
            stage: fn(&Job, usize),
            calc: fn(&Job, usize),
            /// 打包 → 计算的两阶段闸门；可被 panic 中止以放行所有等待者。
            gate: Gate,
            done: AtomicUsize,
            /// 首个 panic 载荷：worker 记入，主线程收尾时 `resume_unwind`。
            panic: Mutex<Option<Box<dyn Any + Send + 'static>>>,
        }

        // 安全性：裸指针指向主线程栈上的输入与堆上的输出；主线程在 `done` 齐之前不返回，
        // 期间 worker 只读写互不重叠的分片。
        #[allow(unsafe_code)]
        unsafe impl Send for Job {}
        #[allow(unsafe_code)]
        unsafe impl Sync for Job {}

        /// 可中止的参与闸门：`p` 个参与者到齐即放行；任一参与者 `abort()` 立即放行其余所有人。
        struct Gate {
            state: Mutex<GateState>,
            cv: Condvar,
        }

        struct GateState {
            arrived: usize,
            generation: u64,
            aborted: bool,
        }

        impl Gate {
            fn new() -> Self {
                Gate {
                    state: Mutex::new(GateState {
                        arrived: 0,
                        generation: 0,
                        aborted: false,
                    }),
                    cv: Condvar::new(),
                }
            }

            /// 到齐放行，返回 `false`；被 `abort()` 提前放行时返回 `true`。
            fn wait(&self, p: usize) -> bool {
                let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
                let start_gen = s.generation;
                s.arrived += 1;
                if s.arrived == p {
                    s.arrived = 0;
                    s.generation += 1;
                    self.cv.notify_all();
                    return false;
                }
                while s.generation == start_gen && !s.aborted {
                    s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
                }
                s.aborted
            }

            /// 中止本次闸门：放行所有已在等待的参与者，并让后续 `wait` 也立即返回 `true`。
            fn abort(&self) {
                let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
                s.aborted = true;
                self.cv.notify_all();
            }
        }

        /// 一个 worker 的私有槽：只装一个任务 `(job, 本槽在 job 里的编号)`。
        struct Cell {
            job: Mutex<Option<(Arc<Job>, usize)>>,
            cv: Condvar,
        }

        pub(super) struct Pool {
            /// 每个常驻 worker 一个私有槽，多槽互不争抢——并发调用可各租一组空闲槽真并行。
            cells: Vec<Cell>,
            /// 空闲槽位图：bit `i` 为 1 表示 worker `i` 空闲；并发发布靠 CAS 租借，杜绝重叠。
            free: AtomicU64,
            /// 保护「等空闲槽」的休眠/唤醒。
            free_lock: Mutex<()>,
            free_cv: Condvar,
            /// 实际起成功的 worker 数（≤ `cells.len()`，也 ≤ 64）。
            live: AtomicUsize,
        }

        static POOL: OnceLock<&'static Pool> = OnceLock::new();

        /// 造 `n` 个空 worker 槽位（每个槽一个 job 锁 + 一个私有条件变量）。
        fn make_cells(n: usize) -> Vec<Cell> {
            (0..n)
                .map(|_| Cell {
                    job: Mutex::new(None),
                    cv: Condvar::new(),
                })
                .collect()
        }

        /// 由给定槽位建常驻池、起 worker（命名前缀 `prefix`），置好 `live`/`free` 后返回。
        fn spawn_pool(cells: Vec<Cell>, prefix: &str) -> &'static Pool {
            let n = cells.len();
            let p: &'static Pool = Box::leak(Box::new(Pool {
                cells,
                free: AtomicU64::new(0),
                free_lock: Mutex::new(()),
                free_cv: Condvar::new(),
                live: AtomicUsize::new(0),
            }));
            let mut live = 0usize;
            for id in 0..n {
                if thread::Builder::new()
                    .name(format!("{prefix}-{id}"))
                    .spawn(move || spin(id, p))
                    .is_ok()
                {
                    live += 1;
                }
            }
            p.live.store(live, Ordering::Release);
            p.free.store((1u64 << live) - 1, Ordering::Release);
            p
        }

        /// 常驻池（首次调用时创建 `cpu-1` 个常驻 worker，之后只唤醒不复建）。
        pub(super) fn pool() -> &'static Pool {
            POOL.get_or_init(|| {
                let want = thread::available_parallelism()
                    .map(|x| x.get())
                    .unwrap_or(1)
                    .saturating_sub(1)
                    .min(64);
                spawn_pool(make_cells(want), "llm-gemm")
            })
        }

        /// worker 主循环：等自己的槽有任务 → 取走 → 执行 → 归还槽位并唤醒等槽者。
        fn spin(id: usize, pool: &'static Pool) {
            let cell = &pool.cells[id];
            loop {
                let task = {
                    let mut g = cell.job.lock().unwrap_or_else(|e| e.into_inner());
                    while g.is_none() {
                        g = cell.cv.wait(g).unwrap_or_else(|e| e.into_inner());
                    }
                    g.take()
                };
                let Some((j, slot)) = task else {
                    continue;
                };
                // 恐慌隔离：worker 的 panic 不杀死常驻线程；记录后中止闸门，其余参与者与发布者
                // 都能退出而不是卡在闸门 / `done` 自旋上。`done` 与槽位归还在成功与 panic 两条
                // 路径上都无条件执行。
                if let Err(payload) =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| part(&j, slot)))
                {
                    j.panic
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .get_or_insert(payload);
                    j.gate.abort();
                }
                j.done.fetch_add(1, Ordering::AcqRel);
                // 归还槽位：先置回空闲位，再通知等待空闲槽的发布者。
                pool.free.fetch_or(1u64 << id, Ordering::AcqRel);
                {
                    let _g = pool.free_lock.lock().unwrap_or_else(|e| e.into_inner());
                    pool.free_cv.notify_all();
                }
            }
        }

        /// 从空闲位图里挑 `need` 个置位，返回这些位的掩码（不足返回 `None`）。
        fn pick_free(free: u64, need: usize) -> Option<u64> {
            let mut mask = 0u64;
            let mut f = free;
            for _ in 0..need {
                if f == 0 {
                    return None;
                }
                let bit = 1u64 << f.trailing_zeros();
                mask |= bit;
                f &= !bit;
            }
            Some(mask)
        }

        /// 从空闲位图租 `need` 个槽的掩码；不足则睡到有槽归还。多调用并发时各自租不同组。
        fn lease_free(pool: &'static Pool, need: usize) -> u64 {
            let mut free = pool.free.load(Ordering::Acquire);
            loop {
                match pick_free(free, need) {
                    Some(m) => match pool.free.compare_exchange(
                        free,
                        free & !m,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => return m,
                        Err(actual) => free = actual,
                    },
                    None => {
                        // 拿锁后必须重查位图：worker 是「先置位、再取锁、再 notify」的，若置位与本次
                        // 加锁之间发生归还，那次 notify 落在「还没人睡下」的空档里，唤醒就丢了——
                        // 睡下去要等到下一次任务结束才醒，并发调用者会互等到卡住。
                        // 所以先持锁复核一次，确认仍然没有空槽，才允许睡。
                        let guard = pool.free_lock.lock().unwrap_or_else(|e| e.into_inner());
                        free = pool.free.load(Ordering::Acquire);
                        if pick_free(free, need).is_some() {
                            drop(guard);
                            continue;
                        }
                        let guard = pool.free_cv.wait(guard).unwrap_or_else(|e| e.into_inner());
                        drop(guard);
                        free = pool.free.load(Ordering::Acquire);
                    }
                }
            }
        }

        /// 一次任务里单个参与者的全部工作：打包自己那份 B 组 → 闸门 → 算自己那段行。
        /// 闸门被中止（有参与者 panic）时跳过 `calc`：打包不完整，结果将被丢弃。
        fn part(j: &Job, slot: usize) {
            (j.stage)(j, slot);
            if !j.gate.wait(j.p) {
                (j.calc)(j, slot);
            }
        }

        /// 把任务发到租来的每个槽：槽号 `1..=need` 对应 worker 在 job 里的分片编号。
        fn dispatch(pool: &'static Pool, job: &Arc<Job>, mut mask: u64) {
            let mut slot = 1usize;
            while mask != 0 {
                let id = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                {
                    let mut g = pool.cells[id].job.lock().unwrap_or_else(|e| e.into_inner());
                    *g = Some((job.clone(), slot));
                }
                pool.cells[id].cv.notify_one();
                slot += 1;
            }
        }

        /// 一个参与者（主线程或 worker）的恐慌隔离执行：panic 记录后中止闸门，其余参与者与发布者
        /// 都能退出而不是卡在闸门 / `done` 自旋上。成功与 panic 两条路径都由调用方无条件收尾。
        fn run_part(job: &Job, slot: usize) {
            if let Err(payload) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| part(job, slot)))
            {
                job.panic
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_or_insert(payload);
                job.gate.abort();
            }
        }

        /// 等全部 worker 完成（成功或 panic 都 `done += 1`），再进入收尾。
        fn await_done(job: &Job, p: usize) {
            while job.done.load(Ordering::Acquire) < p - 1 {
                std::hint::spin_loop();
            }
        }

        /// 等全部槽归还后再把 panic 抛回调用方：此刻没有 worker 再碰 job 的裸指针。
        fn resume_panic(job: &Job) {
            if let Some(payload) = job.panic.lock().unwrap_or_else(|e| e.into_inner()).take() {
                std::panic::resume_unwind(payload);
            }
        }

        /// 池化执行；参与数不足 2 时返回 `false`，由调用方走 scope 兜底。
        #[allow(unsafe_code)]
        pub(super) fn run_pooled(
            pool: &'static Pool,
            bufs: Bufs,
            shape: Shape,
            mr: usize,
            t: usize,
            stage: fn(&Job, usize),
            calc: fn(&Job, usize),
        ) -> bool {
            let Bufs { a, b, buf, out } = bufs;
            let Shape { m, n, k, npad, tn } = shape;
            let live = pool.live.load(Ordering::Acquire);
            let p = t.min(live + 1);
            if p < 2 {
                return false;
            }
            let need = p - 1;
            let mask = lease_free(pool, need);

            let rows_per = m.div_ceil(p).div_ceil(mr) * mr;
            let job = Arc::new(Job {
                a,
                b,
                buf,
                out,
                m,
                n,
                k,
                npad,
                tn,
                rows_per,
                p,
                stage,
                calc,
                gate: Gate::new(),
                done: AtomicUsize::new(0),
                panic: Mutex::new(None),
            });

            dispatch(pool, &job, mask);
            run_part(&job, 0);
            await_done(&job, p);
            resume_panic(&job);
            true
        }

        #[cfg(test)]
        mod tests {
            include!("../tests/kernel/pool_tests.rs");
        }
    }

    // 打包面板的暂存区：按线程复用，用完即还。
    //
    // 打包出来的是「整段 B 面板」，`n = k = 4096` 时是 `4096 × 4096 × 4 B = 67 MB`。每次调用现分配，
    // 光 `calloc` 与首次触页就要几十毫秒，而 decode 与 prefill 都是**同一线程反复调**同一个几何，
    // 这块内存每步都用得上，却每步被丢掉一次。这里让每个调用线程留一块只增不减的缓冲：
    //
    // - **只按线程共享，不跨线程**：并发调用各占一个线程私有的缓冲，不存在两次调用同时写同一块；
    //   池里的 worker 只是拿到主线程的裸指针去写自己的分片，收回发生在 [`with_panels`] 返回之前。
    // - **只扩容，不缩小**：几何变了就扩容一次，之后小几何沿用它前面的部分。常驻量等于本线程见过的
    //   最大面板，与「每次分配再释放」的峰值一致（面板本来就要整块驻留）。
    // - **重入退化为临时缓冲**：若某个打包路径在一次借用期间又调 [`with_panels`]，`RefCell` 已经在
    //   借出状态，这里退化成一次性 `Vec`——宁可多分配一次，也不 panic。
    thread_local! {
        /// 本线程的面板暂存区；长度只增不减。
        static PANELS: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// 借一块长度 `len` 的 f32 暂存区（不足则扩容并清零新增部分），返回 `f` 的结果。
    fn with_panels<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
        PANELS.with(|cell| match cell.try_borrow_mut() {
            Ok(mut buf) => {
                if buf.len() < len {
                    buf.resize(len, 0.0);
                }
                f(&mut buf[..len])
            }
            // 重入（同一次调用里又借一次）：这一次用临时缓冲，语义不变。
            Err(_) => f(&mut vec![0.0f32; len]),
        })
    }

    /// SME 通道要 `#[target_feature(enable = "sme")]` 与 `is_aarch64_feature_detected!("sme")`，
    /// 两者都还没稳定，因此整块只在「nightly + aarch64」存在；其余情形 `gemm_f32` 走 NEON 通道，
    /// 结果一模一样，只是没有 ZA 外积。
    #[cfg(all(nightly, target_arch = "aarch64"))]
    mod sme {
        //! Apple SME 16×64 微内核通道：ZA 矩阵累加器外积（`fmopa`）+ 运行时探测。
        //!
        //! 瓦片形状是**量过**的，不是推出来的：2×2（32×32）每 4 条 `fmopa` 只摊 4 条 `ld1w`，
        //! 按载入条数该比 1×4 快 7.6%，但实测 1 线程 1256.7 vs 1543.4 Gelem/s、16 线程 2063.7 vs
        //! 2271.8（`sme_peak` 口径），**反而慢 19% / 9%**。载入条数因此不是约束项，
        //! 2×2 的代价（A 面板按 32 宽写、每行分 8 次 16 B 散写）超过了它省下的那一条 `ld1w`。
        //! 结论：1×4 保留，`port_mix` 探针留作这条否证的仪器。
        //!
        //! **本模块在仓里钉住的 stable 工具链上不编译**（`fast.rs` 以 `#[cfg(all(nightly, …))]`
        //! 门控，`sme` 的特征探测与 `target_feature` 仍都是 `E0658`）。以上数字只说明 nightly 构建里
        //! 两种瓦片形状的相对优劣，**不是出货 K7**；出货 K7 的分母是 `fma_peak`（见 `docs/math.md`）。

        use std::thread;

        use std::arch::asm;

        use std::simd::{f32x4, simd_swizzle};

        /// SME 微内核行数：一个 16×16 ZA 瓦片覆盖 16 行。
        const MR: usize = 16;

        /// SME 微内核列数：4 个 ZA 瓦片各 16 列，一次写回 64 列。
        const NR: usize = 64;

        /// 规模门槛（元素数）：低于此值的 GEMM，`smstart`/`smstop` 与打包的固定成本超过外积收益，
        /// 交给 NEON 更划算。
        const MIN_ELEMS: usize = 1 << 20;

        /// 运行时探测：本机是否存在 SME（Apple M4/M5 起）。
        fn available() -> bool {
            std::arch::is_aarch64_feature_detected!("sme")
        }

        /// 16×64 整铺 SME GEMM：`out[m,n] = a[m,k] · b[k,n]`。
        pub(super) fn gemm(
            a: &[f32],
            b: &[f32],
            m: usize,
            n: usize,
            k: usize,
            out: &mut [f32],
        ) -> bool {
            if rejects(m, n, k) {
                return false;
            }
            packed(a, b, m, n, k, out, threads_for(m, n));
            true
        }

        /// 线程数：行块（`m/MR`）与列组（`n/NR`）同时决定上限，两者都不能切出空槽。
        fn threads_for(m: usize, n: usize) -> usize {
            super::threads_override().min(m / MR).min(n / NR).max(1)
        }

        /// 拒绝走 SME 的形状：零维、非整铺、低于规模门槛，或本机无 SME。
        fn rejects(m: usize, n: usize, k: usize) -> bool {
            degenerate(m, n, k) || !tiled(m, n) || m * n * k < MIN_ELEMS || !available()
        }

        /// 任一维为零。
        fn degenerate(m: usize, n: usize, k: usize) -> bool {
            m == 0 || n == 0 || k == 0
        }

        /// 行/列是否满足整铺（`MR` 行 × `NR` 列的倍数）。
        fn tiled(m: usize, n: usize) -> bool {
            m.is_multiple_of(MR) && n.is_multiple_of(NR)
        }

        /// 一次整铺 SME GEMM：单次铺线程，先并行打包 A 面板，屏障同步后并行外积累加写回。
        fn packed(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32], t: usize) {
            let mblk = m / MR;
            let alen = mblk * k * MR;
            super::with_panels(alen, |abuf| {
                let barrier = std::sync::Barrier::new(t);
                thread::scope(|s| {
                    let barrier = &barrier;
                    let mut a_rest = &mut *abuf;
                    let mut o_rest = &mut *out;
                    for slot in 0..t {
                        let (r0, r1) = (slot * mblk / t, (slot + 1) * mblk / t);
                        let (a_chunk, a_tail) = a_rest.split_at_mut((r1 - r0) * k * MR);
                        let (o_chunk, o_tail) = o_rest.split_at_mut((r1 - r0) * MR * n);
                        a_rest = a_tail;
                        o_rest = o_tail;
                        s.spawn(move || {
                            pack_a(a, a_chunk, k, r0);
                            barrier.wait();
                            compute_stage(&*a_chunk, b, o_chunk, n, k);
                        });
                    }
                });
            });
        }

        /// 打包 A 的一个行块段：`src[m,k]` → `dst[块][k][16]`，按 16×16 瓦片转置让微内核连续加载。
        fn pack_a(src: &[f32], dst: &mut [f32], k: usize, mblk0: usize) {
            let blocks = dst.len() / (k * MR);
            for bi in 0..blocks {
                let rowbase = (mblk0 + bi) * MR * k;
                let dstbase = bi * k * MR;
                let kk0 = pack_tiles(src, dst, k, rowbase, dstbase);
                pack_tail(src, dst, k, rowbase, dstbase, kk0);
            }
        }

        /// 4×4 子块转置：两轮 `zip1`/`zip2` 交错，供 [`pack_tiles`] 连续写出。
        #[inline(always)]
        fn transpose4(a0: f32x4, a1: f32x4, a2: f32x4, a3: f32x4) -> (f32x4, f32x4, f32x4, f32x4) {
            let t0 = simd_swizzle!(a0, a1, [0, 4, 1, 5]);
            let t1 = simd_swizzle!(a0, a1, [2, 6, 3, 7]);
            let t2 = simd_swizzle!(a2, a3, [0, 4, 1, 5]);
            let t3 = simd_swizzle!(a2, a3, [2, 6, 3, 7]);
            (
                simd_swizzle!(t0, t2, [0, 1, 4, 5]),
                simd_swizzle!(t0, t2, [2, 3, 6, 7]),
                simd_swizzle!(t1, t3, [0, 1, 4, 5]),
                simd_swizzle!(t1, t3, [2, 3, 6, 7]),
            )
        }

        /// 16 对齐的整段：按 16×16 瓦片做 4×4 转置，返回尾部起点。
        fn pack_tiles(src: &[f32], dst: &mut [f32], k: usize, rowbase: usize, dstbase: usize) -> usize {
            let mut kk0 = 0usize;
            while kk0 + 16 <= k {
                for r in 0..4usize {
                    for c in 0..4usize {
                        let rr = r * 4;
                        let cc = kk0 + c * 4;
                        let a0 = f32x4::from_slice(&src[rowbase + rr * k + cc..][..4]);
                        let a1 = f32x4::from_slice(&src[rowbase + (rr + 1) * k + cc..][..4]);
                        let a2 = f32x4::from_slice(&src[rowbase + (rr + 2) * k + cc..][..4]);
                        let a3 = f32x4::from_slice(&src[rowbase + (rr + 3) * k + cc..][..4]);
                        let (b0, b1, b2, b3) = transpose4(a0, a1, a2, a3);
                        let orow = kk0 + c * 4;
                        dst[dstbase + orow * MR + rr..][..4].copy_from_slice(&b0.to_array());
                        dst[dstbase + (orow + 1) * MR + rr..][..4].copy_from_slice(&b1.to_array());
                        dst[dstbase + (orow + 2) * MR + rr..][..4].copy_from_slice(&b2.to_array());
                        dst[dstbase + (orow + 3) * MR + rr..][..4].copy_from_slice(&b3.to_array());
                    }
                }
                kk0 += 16;
            }
            kk0
        }

        /// 尾段（不足 16 列）：逐元素标量搬移。
        fn pack_tail(
            src: &[f32],
            dst: &mut [f32],
            k: usize,
            rowbase: usize,
            dstbase: usize,
            mut kk0: usize,
        ) {
            while kk0 < k {
                for r in 0..MR {
                    dst[dstbase + kk0 * MR + r] = src[rowbase + r * k + kk0];
                }
                kk0 += 1;
            }
        }

        /// 计算一段连续行块：进入流式模式 → 逐行块逐列组外积累加 → 写回 → 退出流式模式。
        #[allow(unsafe_code)]
        fn compute_stage(a_stage: &[f32], b: &[f32], o_stage: &mut [f32], n: usize, k: usize) {
            let nblk = n / NR;
            let blocks = a_stage.len() / (k * MR);
            if blocks == 0 {
                return;
            }
            unsafe {
                enter_sm();
                for bi in 0..blocks {
                    let a_p = a_stage.as_ptr().add(bi * k * MR);
                    let c_base = o_stage.as_mut_ptr().add(bi * MR * n);
                    for g in 0..nblk {
                        let b_p = b.as_ptr().add(g * NR);
                        sme_tile(k, a_p, b_p, c_base.add(g * NR), n * 4, n * 4);
                    }
                }
                exit_sm();
            }
        }

        /// 16×16 外积微内核：`zero` → K 循环 `ld1w`+`fmopa` → 4 个 ZA 瓦片按切片号 `str za` 写回。
        #[target_feature(enable = "sme")]
        #[allow(unsafe_code)]
        #[inline(never)]
        unsafe fn sme_tile(
            k: usize,
            a_p: *const f32,
            b_p: *const f32,
            c: *mut f32,
            n_stride: usize,
            b_stride: usize,
        ) {
            let b_adv = b_stride - 3 * 64;
            unsafe {
                asm!(
                "zero {{ za0.s }}",
                "zero {{ za1.s }}",
                "zero {{ za2.s }}",
                "zero {{ za3.s }}",
                "2:",
                "ld1w {{ z0.s }}, p0/z, [{a}]",
                "add {a}, {a}, #64",
                "ld1w {{ z1.s }}, p0/z, [{b}]",
                "add {b}, {b}, #64",
                "ld1w {{ z2.s }}, p0/z, [{b}]",
                "add {b}, {b}, #64",
                "ld1w {{ z3.s }}, p0/z, [{b}]",
                "add {b}, {b}, #64",
                "ld1w {{ z4.s }}, p0/z, [{b}]",
                "add {b}, {b}, {b_adv}",
                "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
                "fmopa za1.s, p0/m, p0/m, z0.s, z2.s",
                "fmopa za2.s, p0/m, p0/m, z0.s, z3.s",
                "fmopa za3.s, p0/m, p0/m, z0.s, z4.s",
                "subs {k}, {k}, #1",
                "b.ne 2b",
                "mov {c16}, {c}",
                "add {c16}, {c16}, #64",
                "mov {c32}, {c}",
                "add {c32}, {c32}, #128",
                "mov {c48}, {c}",
                "add {c48}, {c48}, #192",
                "mov w12, #0",
                "mov w13, #1",
                "mov w14, #2",
                "mov w15, #3",
                "3:",
                "str za[w12, 0], [{c}]",
                "str za[w13, 0], [{c16}]",
                "str za[w14, 0], [{c32}]",
                "str za[w15, 0], [{c48}]",
                "add {c}, {c}, {nstride}",
                "add {c16}, {c16}, {nstride}",
                "add {c32}, {c32}, {nstride}",
                "add {c48}, {c48}, {nstride}",
                "add w12, w12, #4",
                "add w13, w13, #4",
                "add w14, w14, #4",
                "add w15, w15, #4",
                "cmp w12, #64",
                "b.ne 3b",
                a = inout(reg) a_p => _, b = inout(reg) b_p => _, c = inout(reg) c => _,
                k = inout(reg) k => _,
                nstride = in(reg) n_stride, b_adv = in(reg) b_adv,
                c16 = out(reg) _, c32 = out(reg) _, c48 = out(reg) _,
                out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _,
                out("w12") _, out("w13") _, out("w14") _, out("w15") _, out("p0") _,
                options(nostack),
                );
            }
        }

        /// 进入流式模式并准备全真谓词 `p0`；必须在任何 SME 指令前调用，且每线程一次。
        #[target_feature(enable = "sme")]
        #[allow(unsafe_code)]
        #[inline(never)]
        unsafe fn enter_sm() {
            unsafe {
                asm!("smstart", "smstart za", "ptrue p0.s", options(nostack));
            }
        }

        /// 退出流式模式；与 [`enter_sm`] 成对，`sme_tile` 之间不得夹带 NEON。
        #[target_feature(enable = "sme")]
        #[allow(unsafe_code)]
        #[inline(never)]
        unsafe fn exit_sm() {
            unsafe {
                asm!("smstop za", "smstop", options(nostack));
            }
        }

        #[cfg(test)]
        mod tests {
            include!("../tests/kernel/sme_tests.rs");
        }

        /// SME 端口峰值探针：`rounds` 轮、每轮 4 条 `fmopa`，`z0`/`z1` 与 ZA 全部寄存器常驻，
        /// **内层零 load、零 store**——把「数据从哪来」整个删掉，剩下的才是这块单元每秒能吃多少外积。
        ///
        /// 每轮 4 × 16 × 16 = 1024 次乘加（`p0.s` 16 条泳道，`ZA.s` 瓦片 16×16），即 2048 FLOP；
        /// 瓦片尺寸与 [`MR`]/[`NR`] 同源，由 `sme_tests.rs` 的整铺用例钉死。
        ///
        /// 这是把 GF/s 换算成「端口峰值百分比」的分母：GEMM 每轮还要搬 320 B，搬不动与算不动是两件事，
        /// 不分开量就只能猜瓶颈（F18）。
        ///
        /// 本机无 SME 或 `rounds == 0` 时返回 `false` 且不执行任何 SME 指令——安全入口，调用方据此跳过。
        /// 只在 `bench` 特性下编译，生产公开面不含本项。
        #[cfg(feature = "bench")]
        pub fn port_run(rounds: usize) -> bool {
            if rounds == 0 || !available() {
                return false;
            }
            #[allow(unsafe_code)]
            unsafe {
                port_rounds(rounds);
            }
            true
        }

        /// 纯外积循环：`smstart` → `rounds` 轮 4 条 `fmopa` → `smstop`，不碰任何内存。
        #[cfg(feature = "bench")]
        #[target_feature(enable = "sme")]
        #[allow(unsafe_code)]
        #[inline(never)]
        unsafe fn port_rounds(rounds: usize) {
            unsafe {
                asm!(
                    "smstart",
                    "smstart za",
                    "ptrue p0.s",
                    "fmov z0.s, #1.0",
                    "fmov z1.s, #1.0",
                    "zero {{ za0.s }}",
                    "zero {{ za1.s }}",
                    "zero {{ za2.s }}",
                    "zero {{ za3.s }}",
                    "2:",
                    "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
                    "fmopa za1.s, p0/m, p0/m, z0.s, z1.s",
                    "fmopa za2.s, p0/m, p0/m, z0.s, z1.s",
                    "fmopa za3.s, p0/m, p0/m, z0.s, z1.s",
                    "subs {k}, {k}, #1",
                    "b.ne 2b",
                    "smstop za",
                    "smstop",
                    k = inout(reg) rounds => _,
                    out("z0") _, out("z1") _, out("p0") _,
                    options(nostack),
                );
            }
        }

        /// 取数混合探针：每轮 **1 条 `a` 载入 + 4 条 `b` 载入 + 4 条 `fmopa`**，与 [`sme_tile`] 内层
        /// 的指令配比逐条同构（16×64 瓦片即 `a` 16 个 f32 = 64 B、`b` 64 个 f32 = 256 B）。
        ///
        /// 与 [`port_run`] 的差就是「取数代价」：两者 `fmopa` 条数相同，只差这 5 条 load。
        /// 地址**固定不推进**，即每一轮都命中同一批缓存行——这是取数端口的**上限**，真实内核走不同行
        /// 只会更慢，所以据此算出的取数代价偏小、缺口偏小，拿它下结论不夸大（A3 保守值）。
        ///
        /// 本机无 SME 或 `rounds == 0` 时返回 `false` 且不执行任何 SME 指令。只在 `bench` 特性下编译。
        #[cfg(feature = "bench")]
        pub fn port_mix(rounds: usize, a: &[f32; 16], b: &[f32; 64]) -> bool {
            if rounds == 0 || !available() {
                return false;
            }
            #[allow(unsafe_code)]
            unsafe {
                mix_rounds(rounds, a.as_ptr(), b.as_ptr());
            }
            true
        }

        /// 固定地址的外积 + 载入循环：`smstart` → `rounds` 轮 → `smstop`，读边界由调用方的数组长度保证。
        #[cfg(feature = "bench")]
        #[target_feature(enable = "sme")]
        #[allow(unsafe_code)]
        #[inline(never)]
        unsafe fn mix_rounds(rounds: usize, a: *const f32, b: *const f32) {
            let (b1, b2, b3) = unsafe { (b.add(16), b.add(32), b.add(48)) };
            unsafe {
                asm!(
                    "smstart",
                    "smstart za",
                    "ptrue p0.s",
                    "zero {{ za0.s }}",
                    "zero {{ za1.s }}",
                    "zero {{ za2.s }}",
                    "zero {{ za3.s }}",
                    "2:",
                    "ld1w {{ z0.s }}, p0/z, [{a}]",
                    "ld1w {{ z1.s }}, p0/z, [{b}]",
                    "ld1w {{ z2.s }}, p0/z, [{b1}]",
                    "ld1w {{ z3.s }}, p0/z, [{b2}]",
                    "ld1w {{ z4.s }}, p0/z, [{b3}]",
                    "fmopa za0.s, p0/m, p0/m, z0.s, z1.s",
                    "fmopa za1.s, p0/m, p0/m, z0.s, z2.s",
                    "fmopa za2.s, p0/m, p0/m, z0.s, z3.s",
                    "fmopa za3.s, p0/m, p0/m, z0.s, z4.s",
                    "subs {k}, {k}, #1",
                    "b.ne 2b",
                    "smstop za",
                    "smstop",
                    a = in(reg) a, b = in(reg) b, b1 = in(reg) b1, b2 = in(reg) b2, b3 = in(reg) b3,
                    k = inout(reg) rounds => _,
                    out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("p0") _,
                    options(nostack),
                );
            }
        }
    }

    use pack::{pack_b_f32_into, pack_b_f32_par_into, pack_panels};
    use pool::{Bufs, Shape, pool, run_pooled};

    /// 端口峰值探针只在测量容器里导出（`bench` 特性），生产构建看不到它。
    #[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
    pub use sme::{port_mix, port_run};

    mod tn {
        //! 线性层布局（TN）的稠密 f32 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
        use super::pack::pack_panel;
        use super::pool::{Bufs, Shape, pool, run_pooled};
        use super::{Layout, MR, NR, threads_override};

        /// 走这条入口的最少行数：一个完整微内核行块。再少的批只有残行（标量），打包的转置成本摊不回来，
        /// 调用方应当继续用按列驻留的 tile 核。
        pub const MIN_ROWS: usize = MR;

        /// 多线程 TN 稠密 GEMM：`out[rows,n] = x[rows,k] · w[n,k]ᵀ`。
        ///
        /// 语义同 [`crate::matmul::forward_tn_tiled`]（同一组累加口径，IEEE 容差内一致），吞吐按父模块的
        /// 8×8 寄存器分块 + 常驻池给。
        ///
        /// # Panics
        ///
        /// 三条缓冲长度不足 `rows·k` / `n·k` / `rows·n` 时 panic（调用方先校验）。
        pub fn gemm_f32_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
            check_dims_tn(x, w, rows, n, k, out);
            let t = threads_override().min(rows.div_ceil(MR)).max(1);
            if t <= 1 {
                single_tn(x, w, rows, n, k, out);
                return;
            }
            if !pooled_tn(x, w, rows, n, k, out, t) {
                scoped_tn(x, w, out, rows, n, k, t);
            }
        }

        /// 尺寸校验：三条缓冲不足 `rows·k` / `n·k` / `rows·n` 时 panic。
        fn check_dims_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &[f32]) {
            assert!(
                x.len() >= rows * k && w.len() >= n * k && out.len() >= rows * n,
                "gemm_f32_tn 尺寸不符"
            );
        }

        /// 单线程：整段打包一次，再按行块算完整段 K。
        fn single_tn(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
            let npad = n.div_ceil(NR) * NR;
            super::with_panels(npad * k, |bpack| {
                pack_tn_into(bpack, w, n, k);
                compute_tn(x, bpack, w, rows, n, k, out);
            });
        }

        /// 常驻池路径；池不可用（参与数不足 2）时返回 `false`，由 [`scoped_tn`] 兜底。
        fn pooled_tn(
            x: &[f32],
            w: &[f32],
            rows: usize,
            n: usize,
            k: usize,
            out: &mut [f32],
            t: usize,
        ) -> bool {
            let npad = n.div_ceil(NR) * NR;
            super::with_panels(npad * k, |buf| {
                run_pooled(
                    pool(),
                    Bufs {
                        a: x.as_ptr() as *const u8,
                        b: w.as_ptr() as *const u8,
                        buf: buf.as_mut_ptr() as *mut u8,
                        out: out.as_mut_ptr() as *mut u8,
                    },
                    Shape {
                        m: rows,
                        n,
                        k,
                        npad,
                        tn: true,
                    },
                    MR,
                    t,
                    super::f32_pack,
                    super::f32_calc,
                )
            })
        }

        /// 兜底（池不可用）：打包一次后 shared 给 scope 线程，行块按 MR 对齐切。
        fn scoped_tn(x: &[f32], w: &[f32], out: &mut [f32], rows: usize, n: usize, k: usize, t: usize) {
            let npad = n.div_ceil(NR) * NR;
            let per = rows.div_ceil(MR).div_ceil(t) * MR;
            super::with_panels(npad * k, |bpack| {
                pack_tn_into(bpack, w, n, k);
                let bp: &[f32] = bpack;
                std::thread::scope(|s| {
                    for (ti, block) in out[..rows * n].chunks_mut(per * n).enumerate() {
                        let r0 = ti * per;
                        let got = block.len() / n;
                        let a_l = &x[r0 * k..(r0 + got) * k];
                        s.spawn(move || compute_tn(a_l, bp, w, got, n, k, block));
                    }
                });
            });
        }

        /// 整段打包（单槽、兜底路径与单线程用）：面板布局与父模块一致，只是源按 `[n][k]` 读。
        fn pack_tn_into(bpack: &mut [f32], w: &[f32], n: usize, k: usize) {
            let npad = n.div_ceil(NR) * NR;
            debug_assert_eq!(bpack.len(), npad * k);
            for g in 0..npad / NR {
                pack_panel(w, bpack, n, k, g * NR, g * k * NR, true);
            }
        }

        /// 本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包（与父模块同一套分工）。
        pub(super) fn stage(
            w: &[f32],
            buf: &mut [f32],
            n: usize,
            k: usize,
            npad: usize,
            p: usize,
            slot: usize,
        ) {
            let groups = npad / NR;
            let mut g = slot;
            while g < groups {
                pack_panel(w, buf, n, k, g * NR, g * k * NR, true);
                g += p;
            }
        }

        /// 一段行的计算：A 面板打包 + 按 [`KB`] 分块扫列 + 行残尾。
        ///
        /// 正文与父模块的 Nt 视角**共用一份**（`fast::compute_rows`），这里只钉住 `Layout::Tn`。
        pub(super) fn compute_tn(
            a: &[f32],
            bpack: &[f32],
            w: &[f32],
            rows: usize,
            n: usize,
            k: usize,
            out: &mut [f32],
        ) {
            super::compute_rows(a, bpack, w, super::Extent { rows, n, k }, out, Layout::Tn);
        }

        #[cfg(test)]
        mod tests {
            include!("../tests/kernel/tn_tests.rs");
        }
    }
    pub use tn::{MIN_ROWS, gemm_f32_tn};

    /// 微内核行数。
    const MR: usize = 8;

    /// 微内核列数（两个 NEON 向量）。
    const NR: usize = 8;

    /// 内积分块长度：`K = 4096` 时整段 B 面板有 4MB，按 256 切块让 A 面板与 B 瓦片同时驻留 L1/L2。
    const KB: usize = 256;

    /// 多线程稠密 f32 GEMM：`out[m,n] = a[m,k] · b[k,n]`。
    ///
    /// # Panics
    ///
    /// 三条缓冲长度不足 `m·k` / `k·n` / `m·n` 时 panic（调用方先校验）。
    pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
        check_dims(a, b, m, n, k, out);
        // SME 通道：16×64 整铺的大 GEMM 交给 ZA 外积；小题或不可整铺时退回下方 NEON 通道。
        #[cfg(all(nightly, target_arch = "aarch64"))]
        if sme::gemm(a, b, m, n, k, out) {
            return;
        }
        let blocks = m.div_ceil(MR);
        let t = threads_override().min(blocks).max(1);
        if t <= 1 {
            gemm_f32_single(a, b, m, n, k, out);
            return;
        }
        // 常驻池：打包与计算共用一次发布（一次 Barrier 隔开两阶段），无重复 spawn。
        let npad = n.div_ceil(NR) * NR;
        with_panels(npad * k, |buf| {
            if run_pooled(
                pool(),
                Bufs {
                    a: a.as_ptr() as *const u8,
                    b: b.as_ptr() as *const u8,
                    buf: buf.as_mut_ptr() as *mut u8,
                    out: out.as_ptr() as *mut u8,
                },
                Shape {
                    m,
                    n,
                    k,
                    npad,
                    tn: false,
                },
                MR,
                t,
                f32_pack,
                f32_calc,
            ) {
                return;
            }
            scoped_f32(a, b, out, m, n, k, t);
        });
    }

    /// 显式错误版稠密 f32 GEMM：三条缓冲与 `(m, n, k)` 严格相等才跑，不符返回 [`Code::Invalid`]，
    /// 不 panic。算术同 [`gemm_f32`]（NEON 微内核 + 多槽池）。
    pub fn gemm_f32_checked(
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || out.len() != m * n {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "gemm_f32 缓冲尺寸不符：a={} b={} out={}，期望 m·k={} k·n={} m·n={}",
                    a.len(),
                    b.len(),
                    out.len(),
                    m * k,
                    k * n,
                    m * n
                ),
            ));
        }
        gemm_f32(a, b, m, n, k, out);
        Ok(())
    }

    /// 尺寸校验：三条缓冲不足 `m·k` / `k·n` / `m·n` 时 panic。
    fn check_dims(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &[f32]) {
        assert!(
            a.len() >= m * k && b.len() >= k * n && out.len() >= m * n,
            "gemm_f32 尺寸不符"
        );
    }

    /// 兜底（池不可用）：B 打包一次后 shared 给 scope 线程。
    fn scoped_f32(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize, t: usize) {
        let npad = n.div_ceil(NR) * NR;
        let blocks = m.div_ceil(MR);
        // 行块按 MR 对齐：否则 18 线程切 256 行会得到 15 行/线程，8 行走微内核、7 行掉回标量尾块。
        let per = blocks.div_ceil(t) * MR;
        with_panels(npad * k, |bpack| {
            pack_b_f32_par_into(bpack, b, n, k, t);
            let bp: &[f32] = bpack;
            thread::scope(|s| {
                for (ti, block) in out[..m * n].chunks_mut(per * n).enumerate() {
                    let r0 = ti * per;
                    let rows = block.len() / n;
                    let a_t = &a[r0 * k..(r0 + rows) * k];
                    s.spawn(move || compute_gemm(a_t, bp, b, rows, n, k, block));
                }
            });
        });
    }

    /// 单线程稠密 f32 GEMM（基准确认内核质量用）：`out[m,n] = a[m,k] · b[k,n]`。
    pub fn gemm_f32_single(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, out: &mut [f32]) {
        let npad = n.div_ceil(NR) * NR;
        with_panels(npad * k, |bpack| {
            pack_b_f32_into(bpack, b, n, k);
            compute_gemm(a, bpack, b, m, n, k, out);
        });
    }

    /// 行块计算的三元形状：行数、列宽与 K。
    #[derive(Clone, Copy)]
    struct Extent {
        rows: usize,
        n: usize,
        k: usize,
    }

    /// 一个行块的块计算：A 打包成 8 行面板，再 j 外层、面板内层跑微内核；残尾走标量。
    ///
    /// `layout` 是**数据而非分支**：`Nt`（B 按列驻留）与 `Tn`（W 转置驻留）曾各写一份逐字
    /// 相同的正文，只在 `Sweep.layout` / `edge` 的最后一个实参上不同；现在一份正文配两次调用。
    fn compute_rows(
        a: &[f32],
        bpack: &[f32],
        w: &[f32],
        extent: Extent,
        out: &mut [f32],
        layout: Layout,
    ) {
        let Extent { rows, n, k } = extent;
        let full = rows / MR;
        let apack = pack_panels(a, full, k);
        let mut first = true;
        let mut k0 = 0;
        while k0 < k {
            let kb = KB.min(k - k0);
            sweep_columns(
                Sweep {
                    a,
                    apack: &apack,
                    bpack,
                    w,
                    full,
                    n,
                    k,
                    layout,
                },
                out,
                k0,
                kb,
                first,
            );
            first = false;
            k0 += kb;
        }
        // 残行只占 ≤7 行，一次把整段 K 走完，不参与分块累加。
        let tail = full * MR;
        if tail < rows {
            edge(
                a,
                w,
                out,
                Span {
                    i: tail,
                    j: 0,
                    mr: rows - tail,
                    nr: n,
                    n,
                    k,
                    stride: k,
                },
                0,
                true,
                layout,
            );
        }
    }

    /// Nt 视角的块计算：B 按列驻留（父模块的两条入口用它）。
    fn compute_gemm(
        a: &[f32],
        bpack: &[f32],
        b: &[f32],
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) {
        compute_rows(a, bpack, b, Extent { rows, n, k }, out, Layout::Nt);
    }

    /// 沿列块扫描整行块：整块走微内核，最后一段残列走标量边（`edge` 读未打包的 `a`）。
    fn sweep_columns(sweep: Sweep<'_>, out: &mut [f32], k0: usize, kb: usize, first: bool) {
        let Sweep {
            a,
            apack,
            bpack,
            w,
            full,
            n,
            k,
            layout,
        } = sweep;
        let mut j = 0;
        while j < n {
            let nr = NR.min(n - j);
            if nr == NR {
                let start = (j / NR) * k * NR + k0 * NR;
                let b_tile = &bpack[start..start + kb * NR];
                for blk in 0..full {
                    micro_packed(
                        &apack[blk * k * MR + k0 * MR..],
                        b_tile,
                        out,
                        Tile {
                            row0: blk * MR,
                            j,
                            n,
                            kb,
                            first,
                        },
                    );
                }
            } else {
                edge(
                    a,
                    w,
                    out,
                    Span {
                        i: 0,
                        j,
                        mr: full * MR,
                        nr,
                        n,
                        k: kb,
                        stride: k,
                    },
                    k0,
                    first,
                    layout,
                );
            }
            j += NR;
        }
    }

    mod i8 {
        //! 职责：i8 sdot 微内核：MR×NR 打包与 dotprod 探测回退。
        // i8 sdot 通道：MR=4 × NR=4，B 转置打包成「每列 k 连续」。
        // 每个 k16 一次 4 行 A 加载 + 4 次 B 加载 + 16 次 `vdotq_s32`（256 MAC/24 指令），
        // 且每块 B 瓦片被 4 行复用——B 访存比 MR=2/NR=8 再降一半。
        // dotprod 运行时探测，缺失走标量回退；`unsafe` 仅限 aarch64 微内核。
        // ---------------------------------------------------------------------------

        /// i8 微内核行数（四行共享同一 B 向量，摊薄 B 加载）。
        use std::thread;

        use diagnostic::{Code, Error, Result};

        use super::compute_gemm;
        use super::pack::b_groups;
        use super::pool::{Bufs, Job, Shape, pool, run_pooled};
        use super::tn;

        pub(super) const I8_MR: usize = 4;

        /// 行块高度：让 A 的 `rc × k` 字节留在 L1（约 128KiB），否则每个列块都要从 L2 重拉整块 A。
        ///
        /// 「留住」这个假设已被实测证伪：`rc` 扫 112 / 64 / 32 与默认 128 在噪声内无差
        /// （默认 309.1、366.7 Gelem/s；`rc=112` 304.4、346.4），见 `docs/lld.md` v3.0.40。
        fn row_chunk(k: usize) -> usize {
            ((1usize << 17) / k.max(1)).clamp(I8_MR, 256)
        }

        /// i8 微内核列数（4 个 int32x4 累加器 × 4 行 = 16 寄存器，避开溢出）。
        pub(super) const I8_NR: usize = 4;

        /// 线程数：`LLM_THREADS` 环境变量优先，否则全核。
        pub(super) fn threads_override() -> usize {
            std::env::var("LLM_THREADS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|x| *x > 0)
                .unwrap_or_else(|| {
                    thread::available_parallelism()
                        .map(|x| x.get())
                        .unwrap_or(1)
                })
        }

        /// 打包阶段：本槽负责的列组按 `p` 步长轮转，别槽的列组由别槽打包。
        pub(super) fn i8_pack(j: &Job, slot: usize) {
            let p = j.p;
            let groups = j.npad / I8_NR;
            #[allow(unsafe_code)]
            unsafe {
                let b = j.b as *const i8;
                let bp = j.buf as *mut i8;
                let n = j.n;
                let k = j.k;
                let mut g = slot;
                while g < groups {
                    for nb in 0..I8_NR {
                        let jj = g * I8_NR + nb;
                        if jj < n {
                            let dst = bp.add((g * I8_NR + nb) * k);
                            let mut q = 0;
                            while q < k {
                                *dst.add(q) = *b.add(q * n + jj);
                                q += 1;
                            }
                        }
                    }
                    g += p;
                }
            }
        }

        /// 计算阶段：只算本槽的行段，`bpack` 由 Barrier 保证整块就绪。
        pub(super) fn i8_calc(j: &Job, slot: usize) {
            let r0 = slot * j.rows_per;
            if r0 >= j.m {
                return;
            }
            let rows = (j.m - r0).min(j.rows_per);
            #[allow(unsafe_code)]
            unsafe {
                let a_l = std::slice::from_raw_parts(j.a as *const i8, j.m * j.k);
                let a_l = &a_l[r0 * j.k..(r0 + rows) * j.k];
                let b_l = std::slice::from_raw_parts(j.buf as *const i8, j.npad * j.k);
                let out_l = std::slice::from_raw_parts_mut(j.out as *mut i32, j.m * j.n);
                let out_l = &mut out_l[r0 * j.n..(r0 + rows) * j.n];
                compute_i8_packed(a_l, b_l, j.npad, rows, j.n, j.k, out_l);
            }
        }

        /// f32 打包体：本槽负责的 8 列组按 `p` 步长轮转，写入 `[jblk][p][NR]`（零初始化覆盖补零列）。
        #[allow(unsafe_code)]
        pub(super) fn f32_pack(j: &Job, slot: usize) {
            let (n, k, npad) = (j.n, j.k, j.npad);
            unsafe {
                let b = std::slice::from_raw_parts(j.b as *const f32, k * n);
                let dst = std::slice::from_raw_parts_mut(j.buf as *mut f32, npad * k);
                if j.tn {
                    tn::stage(b, dst, n, k, npad, j.p, slot);
                } else {
                    b_groups(b, dst, n, k, npad, j.p, slot);
                }
            }
        }

        /// f32 计算体：只算本槽的行段；B 面板由 Barrier 保证整块就绪。
        #[allow(unsafe_code)]
        pub(super) fn f32_calc(j: &Job, slot: usize) {
            let r0 = slot * j.rows_per;
            if r0 >= j.m {
                return;
            }
            let rows = (j.m - r0).min(j.rows_per);
            unsafe {
                let a = std::slice::from_raw_parts(j.a as *const f32, j.m * j.k);
                let a_l = &a[r0 * j.k..(r0 + rows) * j.k];
                let bp = std::slice::from_raw_parts(j.buf as *const f32, j.npad * j.k);
                let b = std::slice::from_raw_parts(j.b as *const f32, j.k * j.n);
                let out = std::slice::from_raw_parts_mut(j.out as *mut f32, j.m * j.n);
                let out_l = &mut out[r0 * j.n..(r0 + rows) * j.n];
                if j.tn {
                    tn::compute_tn(a_l, bp, b, rows, j.n, j.k, out_l);
                } else {
                    compute_gemm(a_l, bp, b, rows, j.n, j.k, out_l);
                }
            }
        }

        /// 多线程 i8 GEMM：`out[m,n] = a[m,k] · b[k,n]`，i8 入 i32 累加。
        ///
        /// # Panics
        ///
        /// 三条缓冲长度不足时 panic（调用方先校验）。
        pub fn gemm_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
            assert!(
                a.len() >= m * k && b.len() >= k * n && out.len() >= m * n,
                "gemm_i8 尺寸不符"
            );
            let cpu = threads_override();
            let npad = n.div_ceil(I8_NR) * I8_NR;
            let groups = npad / I8_NR;
            let t = cpu.min(m.div_ceil(I8_MR)).min(groups).max(1);
            if t <= 1 {
                single_i8(a, b, m, n, k, out);
                return;
            }
            // 常驻池：一次发布 + 一次 Barrier，无 spawn 成本。
            let mut bpack = vec![0i8; npad * k];
            if run_pooled(
                pool(),
                Bufs {
                    a: a.as_ptr() as *const u8,
                    b: b.as_ptr() as *const u8,
                    buf: bpack.as_mut_ptr() as *mut u8,
                    out: out.as_mut_ptr() as *mut u8,
                },
                Shape {
                    m,
                    n,
                    k,
                    npad,
                    tn: false,
                },
                I8_MR,
                t,
                i8_pack,
                i8_calc,
            ) {
                return;
            }
            scoped_i8(a, b, out, m, n, k, t);
        }

        /// 显式错误版对称 int8 GEMM：三条缓冲与 `(m, n, k)` 严格相等才跑，不符返回 [`Code::Invalid`]，
        /// 不 panic。算术同 [`gemm_i8`]（i32 精确累加，与 [`crate::matmul_i8::forward_tiled`] 逐位一致）。
        pub fn gemm_i8_checked(
            a: &[i8],
            b: &[i8],
            m: usize,
            n: usize,
            k: usize,
            out: &mut [i32],
        ) -> Result<()> {
            if a.len() != m * k || b.len() != k * n || out.len() != m * n {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "gemm_i8 缓冲尺寸不符：a={} b={} out={}，期望 m·k={} k·n={} m·n={}",
                        a.len(),
                        b.len(),
                        out.len(),
                        m * k,
                        k * n,
                        m * n
                    ),
                ));
            }
            gemm_i8(a, b, m, n, k, out);
            Ok(())
        }

        /// 单线程路径：B 打包一次，直接算完整块。
        pub(super) fn single_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
            let (bpack, npad) = pack_b(b, n, k);
            compute_i8_packed(a, &bpack, npad, m, n, k, out);
        }

        /// 兜底路径：池不可用（worker 起不来）时退回 scope 线程。
        pub(super) fn scoped_i8(
            a: &[i8],
            b: &[i8],
            out: &mut [i32],
            m: usize,
            n: usize,
            k: usize,
            t: usize,
        ) {
            let mut rows_per = m.div_ceil(t);
            // 每线程成组行，保证微内核分块不被切碎（缺则末线程走标量尾块）。
            rows_per = rows_per.div_ceil(I8_MR) * I8_MR;
            // B 只打包一次、全部线程共享：N 线程各打包一遍会让冗余带宽吞掉扩展性。
            let (bpack, npad) = pack_b_par(b, n, k);
            thread::scope(|s| {
                for (ti, block) in out[..m * n].chunks_mut(rows_per * n).enumerate() {
                    let r0 = ti * rows_per;
                    let rows = block.len() / n;
                    let a_t = &a[r0 * k..(r0 + rows) * k];
                    let bp = &bpack;
                    s.spawn(move || compute_i8_packed(a_t, bp, npad, rows, n, k, block));
                }
            });
        }

        /// 单线程 i8 GEMM（基准确认内核质量用）。
        pub fn gemm_i8_single(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
            single_i8(a, b, m, n, k, out);
        }

        /// `sdot` 端口峰值探针一轮的乘加次数：12 条互不依赖的 `sdot` × 16 项。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        pub const SDOT_MAC_PER_ROUND: usize = 12 * 16;

        /// i8 `sdot` 端口峰值：`rounds` 轮 × 12 条 `sdot`，操作数与累加器全常驻，**内层零 load 零 store**。
        ///
        /// 这是 K8 的分母：i8 路径走的是 NEON `sdot`（每指令 4 组 × 4 项 = 16 次乘加），不是 SME
        /// 的 `fmopa`，所以 f32 FMA 峰值与 SME 峰值**都不能**当分母——那会得到 >100% 的无意义百分比（F2/F18）。
        ///
        /// **12 条链是实测平台起点**（链数扫描，单线程 GMAC/s：4 → 91、8 → 182、12 → 272、
        /// 16 → 274、20 → 273、24 → 237、32 → 232；两次重复误差 <1%）。平台是 12–20 条，
        /// ≥24 条因累加器溢出寄存器回落到 235 一带。滑下去的 8 条链版本低于平台 1.5 倍，
        /// 曾把 GEMM 误判成超峰值——分母偏低会让百分比虚高，所以这里取平台**左侧**的 12 条而不是最高点。
        /// 延迟/发射模型解释不了这个数值（274 GMAC/s ÷ 16 MAC = 17 G 条/s），故只报实测值，不报推导值。
        ///
        /// 本机无 `dotprod` 或 `rounds == 0` 时返回 `false` 且不执行任何向量指令。只在 `bench` 特性下编译。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        pub fn port_sdot(rounds: usize) -> bool {
            if rounds == 0 || !std::arch::is_aarch64_feature_detected!("dotprod") {
                return false;
            }
            // 安全性：dotprod 已运行时探测；函数体内只有寄存器运算，不触碰内存。
            #[allow(unsafe_code)]
            unsafe {
                sdot_rounds(rounds);
            }
            true
        }

        /// 纯 `sdot` 循环：12 个累加器互不依赖，累加结果进 `black_box` 防整段消除。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        #[target_feature(enable = "dotprod")]
        #[allow(unsafe_code)]
        #[inline(never)]
        unsafe fn sdot_rounds(rounds: usize) {
            use std::arch::aarch64::*;
            let a = vdupq_n_s8(3);
            let b = vdupq_n_s8(5);
            let mut acc = [vdupq_n_s32(0); 12];
            for _ in 0..rounds {
                for chain in &mut acc {
                    *chain = vdotq_s32(*chain, a, b);
                }
            }
            std::hint::black_box(acc);
        }

        /// `sdot` **取数受限**峰值一轮的 k16 步数换算系数：每步 16 条 `sdot`，故一轮供 `16 * k` 次乘加。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        pub const SDOT_BLOCK_MAC_PER_K: usize = 16;

        /// `sdot` **取数受限**峰值：一轮 = `k/16` 个 k16 步，每步 4 条 A load + 4 条 B load + 16 条
        /// `sdot`，与生产微内核 [`sdot_tile`] 逐指令一致（同一份代码，不会漂移）。
        ///
        /// **为什么要它**：纯端口探针零 load，是理想上限；生产核每 16 条 `sdot` 要发 8 条 load。
        /// 两者之差就是取数代价，也是「K8 的缺口里多少是结构注定、多少是核写得差」的分界线。
        ///
        /// **第一版是错探针，必须记下来**：当时把 A/B 写成固定 64 字节、地址用常量偏移
        /// `bp.add(nb * 16)`，而循环变量不进地址，于是 LLVM 把 8 条 load **整体提到循环外**——
        /// `otool -tV` 实测循环体只剩 16 条 `sdot` 加 `subs`/`b.ne`，`ldp q16,q17,[x0]` 只出现一次。
        /// 那样的「混合峰值 ≈ 纯峰值」是空证据：它根本没在环内取数。改成与生产相同的 `base + p`
        /// 单调推进后，取数被迫留在环内，实测比值才是真的（同一台机、同一编译参数，中位 0.875）。
        ///
        /// 但仓内 `criterion` 口径与那支独立程序不同，且**仓内才是权威**：同一会话
        /// `i8_peak/1_block/port` 539.37 vs `i8_peak/1/port` 543.19 = **0.993**——单线程取数几乎免费；
        /// 18 线程 5667.3 vs 6736.0 = **0.841**，即瓦片体自身能到端口峰值的 84%。
        /// 故「4×4 tile 注定只能到 87.5%」被证伪（单线程天花板 99%、18 线程 84%）。
        ///
        /// **它现在与生产共用 [`sdot_tile`]，因此不再是独立仪器**：N5 要求探针与生产同源、不许漂移，
        /// 代价是它的绝对值不能再横向比"换了一版取数写法"的旧数（旧独立版单线程 493.3、18 线程 4032.8）。
        /// 它仍然回答一个没有别的口径能回答的问题：**同一个瓦片体放进零记账的合成循环里能跑多快**，
        /// 于是"生产离瓦片体上限差多少"= 内存系统 + 瓦片外记账，而不是核写得差。
        ///
        /// A/B 与 `k` 的形状校验：`k` 非零且是 16 的倍数，两侧各留足 `4k`。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        fn block_shaped(a: &[i8], b: &[i8], k: usize) -> bool {
            k != 0 && k.is_multiple_of(16) && a.len() >= 4 * k && b.len() >= 4 * k
        }

        /// 本机无 `dotprod`、`rounds == 0`、形状不合时返回 `false` 且不执行任何向量指令。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        pub fn port_sdot_block(rounds: usize, a: &[i8], b: &[i8], k: usize) -> bool {
            if rounds == 0
                || !block_shaped(a, b, k)
                || !std::arch::is_aarch64_feature_detected!("dotprod")
            {
                return false;
            }
            // 安全性：dotprod 已运行时探测；A/B 各 4k 字节已在上面校验。
            #[allow(unsafe_code)]
            unsafe {
                sdot_block_rounds(rounds, a.as_ptr(), b.as_ptr(), k);
            }
            true
        }

        /// 取数受限循环：4 行 × 4 列累加器，每 k16 步 4 条 A load + 4 条 B load + 16 条 `sdot`。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        #[target_feature(enable = "dotprod")]
        #[allow(unsafe_code)]
        #[inline(never)]
        unsafe fn sdot_block_rounds(rounds: usize, ap: *const i8, bp: *const i8, k: usize) {
            unsafe {
                for _ in 0..rounds {
                    std::hint::black_box(sdot_tile(ap, bp, k, k));
                }
            }
        }

        /// 一个 4 行 × 4 列瓦片的整数累加：`bp` 指向**本瓦片首列**在 B 面板里的起点，只累加 `p < kvec`。
        ///
        /// 生产微内核（[`kernel::dot_one`]）与取数受限探针（[`sdot_block_rounds`]）共用**这一份**：
        /// 探针的全部意义是"与生产逐指令一致"，两份写法必然漂移，一份才有保证（N5/可替换）。
        /// `kvec` 是向量化段长度，标量尾巴由 [`kernel::harvest`] 补，不在这里。
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "dotprod")]
        #[allow(unsafe_code)]
        #[inline]
        unsafe fn sdot_tile(
            ap: *const i8,
            bp: *const i8,
            k: usize,
            kvec: usize,
        ) -> [[std::arch::aarch64::int32x4_t; I8_MR]; I8_NR] {
            use std::arch::aarch64::*;
            unsafe {
                let zero = vdupq_n_s32(0);
                // 列优先：一列一个内层迭代，块内推进不必同时索引四行。
                let mut accs = [[zero; I8_MR]; I8_NR];
                let mut p = 0;
                while p < kvec {
                    let av: [int8x16_t; I8_MR] = [
                        vld1q_s8(ap.add(p)),
                        vld1q_s8(ap.add(k + p)),
                        vld1q_s8(ap.add(2 * k + p)),
                        vld1q_s8(ap.add(3 * k + p)),
                    ];
                    for (nb, column) in accs.iter_mut().enumerate() {
                        let bv = vld1q_s8(bp.add(nb * k + p));
                        for (cell, a) in column.iter_mut().zip(av) {
                            *cell = vdotq_s32(*cell, a, bv);
                        }
                    }
                    p += 16;
                }
                accs
            }
        }

        /// B 转置打包成 `[nblk][nb][k]`（同列 k 连续），返回 `(面板, n 补齐到 8)`。
        mod kernel {
            use super::{I8_MR, I8_NR, sdot_tile};

            /// 单线程块计算（B 面板已备）：**N 分组在最外层**，每组 8×k 的 B 瓦片常驻 L1，
            /// 被本线程所有行对复用——否则每行对都把整块 B 重读一遍，卡在内存带宽。
            pub(super) fn compute_i8_packed(
                a: &[i8],
                bpack: &[i8],
                npad: usize,
                rows: usize,
                n: usize,
                k: usize,
                out: &mut [i32],
            ) {
                let kvec = k - k % 16;
                let rc = (super::row_chunk(k) / I8_MR) * I8_MR;
                let mut r0 = 0;
                while r0 < rows {
                    let rend = (r0 + rc).min(rows);
                    let mut blk = 0;
                    while blk < npad {
                        let mut r = r0;
                        while r + I8_MR <= rend {
                            pair_one(a, bpack, out, Gram { r, blk, n, k, kvec });
                            r += I8_MR;
                        }
                        while r < rend {
                            scalar_row_group(a, bpack, out, r, blk, n, k);
                            r += 1;
                        }
                        blk += I8_NR;
                    }
                    r0 = rend;
                }
            }

            /// 一个 8 列块的两行：起点行 `r`、列块起点 `blk`，矩阵列宽 `n`、内积长 `k` 与向量化长 `kvec`。
            #[derive(Clone, Copy)]
            pub(super) struct Gram {
                r: usize,
                blk: usize,
                n: usize,
                k: usize,
                #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
                kvec: usize,
            }

            /// 单个 8 列块、两行微内核分派：dotprod 可用且 k≥16 走向量，否则标量两行。
            #[allow(unsafe_code)]
            pub(super) fn pair_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
                #[cfg(target_arch = "aarch64")]
                {
                    if g.kvec >= 16 && std::arch::is_aarch64_feature_detected!("dotprod") {
                        // 安全性：dotprod 已运行时探测。
                        unsafe { dot_one(a, bpack, out, g) };
                        return;
                    }
                }
                scalar_row_group(a, bpack, out, g.r, g.blk, g.n, g.k);
                for mr in 1..I8_MR {
                    scalar_row_group(a, bpack, out, g.r + mr, g.blk, g.n, g.k);
                }
            }

            /// aarch64 dotprod：一个 4 列块的 4 行（`unsafe` 岛）。
            #[cfg(target_arch = "aarch64")]
            #[target_feature(enable = "dotprod")]
            #[allow(unsafe_code)]
            unsafe fn dot_one(a: &[i8], bpack: &[i8], out: &mut [i32], g: Gram) {
                unsafe {
                    let ap = a.as_ptr().add(g.r * g.k);
                    let bp = bpack.as_ptr();
                    let accs = sdot_tile(ap, bp.add(g.blk * g.k), g.k, g.kvec);
                    harvest(ap, bp, out, (g.r, g.n, g.k, g.kvec), g.blk, accs);
                }
            }

            /// 一个 4 列块的收口：水平求和 + 标量补 `kvec..k` 的余数，写回 4 行。
            #[cfg(target_arch = "aarch64")]
            #[target_feature(enable = "dotprod")]
            #[allow(unsafe_code)]
            unsafe fn harvest(
                ap: *const i8,
                bp: *const i8,
                out: &mut [i32],
                (r, n, k, kvec): (usize, usize, usize, usize),
                blk: usize,
                accs: [[std::arch::aarch64::int32x4_t; I8_MR]; I8_NR],
            ) {
                use std::arch::aarch64::*;
                unsafe {
                    for (nb, column) in accs.iter().enumerate() {
                        let jj = blk + nb;
                        if jj < n {
                            for (mr, cell) in column.iter().enumerate() {
                                let mut s = vaddvq_s32(*cell);
                                let mut q = kvec;
                                while q < k {
                                    s += *ap.add(mr * k + q) as i32 * *bp.add(jj * k + q) as i32;
                                    q += 1;
                                }
                                out[(r + mr) * n + jj] = s;
                            }
                        }
                    }
                }
            }

            /// 标量单行、单 4 列块（打包 B 取数，任意 n/k）。
            pub(super) fn scalar_row_group(
                a: &[i8],
                bpack: &[i8],
                out: &mut [i32],
                row: usize,
                blk: usize,
                n: usize,
                k: usize,
            ) {
                let ar = &a[row * k..(row + 1) * k];
                for nb in 0..I8_NR {
                    let jj = blk + nb;
                    if jj < n {
                        let bp = &bpack[jj * k..jj * k + k];
                        let mut acc = 0i32;
                        for p in 0..k {
                            acc += ar[p] as i32 * bp[p] as i32;
                        }
                        out[row * n + jj] = acc;
                    }
                }
            }
        }

        mod pack {
            use super::super::*;
            use super::I8_NR;

            pub(super) fn pack_b(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
                let npad = n.div_ceil(I8_NR) * I8_NR;
                let mut bpack = vec![0i8; npad * k];
                for jj in 0..n {
                    let base = ((jj / I8_NR) * I8_NR + jj % I8_NR) * k;
                    for p in 0..k {
                        bpack[base + p] = b[p * n + jj];
                    }
                }
                (bpack, npad)
            }

            /// 并行 B 打包：转置是跨 `n` 字节步长的 cache-miss 密集写，小 `n` 大 `k` 时串行打包
            /// 会独占整段耗时（如 256×256×4096 的 1MB 面板），这里按 8 列组切成线程。
            pub(super) fn pack_b_par(b: &[i8], n: usize, k: usize) -> (Vec<i8>, usize) {
                let npad = n.div_ceil(I8_NR) * I8_NR;
                let groups = npad / I8_NR;
                let t = threads_override().min(groups);
                if t <= 1 || npad * k < (1 << 16) {
                    return pack_b(b, n, k);
                }
                let mut bpack = vec![0i8; npad * k];
                let per = groups.div_ceil(t);
                let span = per * I8_NR * k;
                thread::scope(|s| {
                    for (gi, chunk) in bpack.chunks_mut(span).enumerate() {
                        let blk0 = gi * per;
                        s.spawn(move || pack_b_span(chunk, b, n, k, blk0));
                    }
                });
                (bpack, npad)
            }

            /// 一段连续的列组：`blk0` 起、`chunk` 覆盖的每组 8 列各搬一块。
            pub(super) fn pack_b_span(chunk: &mut [i8], b: &[i8], n: usize, k: usize, blk0: usize) {
                let blocks = chunk.len() / (I8_NR * k);
                for bi in 0..blocks {
                    let tile = &mut chunk[bi * I8_NR * k..(bi + 1) * I8_NR * k];
                    pack_b_tile(tile, b, n, k, (blk0 + bi) * I8_NR);
                }
            }

            /// 一组 8 列瓦片的转置写：`b`（行主、`n` 列）的第 `first..first+8` 列搬到 `dst`（列主块内偏移）。
            /// 越界的列（`n` 不是 8 的倍数）留 0——打包缓冲本来就按 `npad` 补齐，乘加时该列权重为 0。
            pub(super) fn pack_b_tile(dst: &mut [i8], b: &[i8], n: usize, k: usize, first: usize) {
                for nb in 0..I8_NR {
                    let jj = first + nb;
                    if jj >= n {
                        continue;
                    }
                    let off = nb * k;
                    let mut p = 0;
                    while p < k {
                        dst[off + p] = b[p * n + jj];
                        p += 1;
                    }
                }
            }
        }
        use kernel::compute_i8_packed;
        use pack::{pack_b, pack_b_par};
    }
    mod micro {
        //! 职责：微内核整行块扫描的只读侧（Sweep）。
        use super::{MR, NR};

        /// 一次整行块扫描的只读侧：指针、面板与三个尺寸一次带走（实参不散成一长串）。
        #[derive(Clone, Copy)]
        pub(super) struct Sweep<'a> {
            pub(super) a: &'a [f32],
            pub(super) apack: &'a [f32],
            pub(super) bpack: &'a [f32],
            pub(super) w: &'a [f32],
            pub(super) full: usize,
            pub(super) n: usize,
            pub(super) k: usize,
            pub(super) layout: Layout,
        }

        /// 一个待算的 8×8 微内核块：左上角 `(row0, j)`、矩阵列宽 `n`、内积分块长 `kb`、是否首块。
        #[derive(Clone, Copy)]
        pub(super) struct Tile {
            pub(super) row0: usize,
            pub(super) j: usize,
            pub(super) n: usize,
            pub(super) kb: usize,
            pub(super) first: bool,
        }

        /// 打包面板上的 NEON 8×8 微内核（A 面板 `[p][8]`、B 面板 `[p][8]`，一次算 8 行 8 列）。
        #[cfg(target_arch = "aarch64")]
        #[allow(unsafe_code)]
        pub(super) fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
            let Tile {
                row0,
                j,
                n,
                kb,
                first,
            } = tile;
            use std::arch::aarch64::*;
            unsafe {
                let zero = vdupq_n_f32(0.0);
                let pp = panel.as_ptr();
                let bp = b.as_ptr();
                // 后续内积分块接着上一块的累加器算：首块用 0 起手，否则读回 `out` 的 16 个向量。
                let mut acc = if first {
                    [zero; 16]
                } else {
                    micro_load(out, row0, n, j)
                };
                fma_step(&mut acc, pp, bp, kb);
                micro_store(out, row0, n, j, acc);
            }
        }

        /// 一个收缩步的 8×8 瓦片体：4 条 `vld1q` 取数 + 16 条 lane-indexed FMA。
        ///
        /// 生产微内核 [`micro_packed`] 与取数受限端口探针 [`fma_block_rounds`] 共用**这一份**——
        /// 探针的全部意义是「与生产逐指令一致」，两份写法必然漂移（N5）。`pp` 是 A 面板每步
        /// `MR` 个 f32、`bp` 是 B 面板每步 `NR` 个 f32，两者步距都是 8 即 32 字节。
        #[cfg(target_arch = "aarch64")]
        #[allow(unsafe_code)]
        #[inline]
        unsafe fn fma_step(
            acc: &mut [std::arch::aarch64::float32x4_t; 16],
            pp: *const f32,
            bp: *const f32,
            kb: usize,
        ) {
            use std::arch::aarch64::*;
            unsafe {
                for p in 0..kb {
                    let av0 = vld1q_f32(pp.add(p * MR));
                    let av1 = vld1q_f32(pp.add(p * MR + 4));
                    let bv0 = vld1q_f32(bp.add(p * NR));
                    let bv1 = vld1q_f32(bp.add(p * NR + 4));
                    // lane-indexed FMA 把广播并进乘加：每条 `p` 从 24 条指令降到 20 条（省 8 条 dup）。
                    acc[0] = vfmaq_laneq_f32::<0>(acc[0], bv0, av0);
                    acc[1] = vfmaq_laneq_f32::<0>(acc[1], bv1, av0);
                    acc[2] = vfmaq_laneq_f32::<1>(acc[2], bv0, av0);
                    acc[3] = vfmaq_laneq_f32::<1>(acc[3], bv1, av0);
                    acc[4] = vfmaq_laneq_f32::<2>(acc[4], bv0, av0);
                    acc[5] = vfmaq_laneq_f32::<2>(acc[5], bv1, av0);
                    acc[6] = vfmaq_laneq_f32::<3>(acc[6], bv0, av0);
                    acc[7] = vfmaq_laneq_f32::<3>(acc[7], bv1, av0);
                    acc[8] = vfmaq_laneq_f32::<0>(acc[8], bv0, av1);
                    acc[9] = vfmaq_laneq_f32::<0>(acc[9], bv1, av1);
                    acc[10] = vfmaq_laneq_f32::<1>(acc[10], bv0, av1);
                    acc[11] = vfmaq_laneq_f32::<1>(acc[11], bv1, av1);
                    acc[12] = vfmaq_laneq_f32::<2>(acc[12], bv0, av1);
                    acc[13] = vfmaq_laneq_f32::<2>(acc[13], bv1, av1);
                    acc[14] = vfmaq_laneq_f32::<3>(acc[14], bv0, av1);
                    acc[15] = vfmaq_laneq_f32::<3>(acc[15], bv1, av1);
                }
            }
        }

        /// 一个收缩步的乘加次数：16 条 FMA × 每向量 4 个 f32 通道。以 `k` 为单位命名，
        /// 因为取数受限探针按 `k` 步展开，纯端口探针正好一步。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        pub const FMA_MAC_PER_K: usize = 16 * 4;

        /// f32 FMA 端口峰值：`rounds` 步 × 16 条 lane-indexed FMA，**环内零 load 零 store**。
        ///
        /// 这是 K7 分母里「理想上限」那一半，指令形状与生产瓦片体逐条一致（同 16 个累加器、
        /// 同 lane-indexed FMA），只差那 4 条 `vld1q`。`machine_peak` 虽也是 f32 峰值，但它是
        /// 自动向量化的标量循环（4 条链 × 16 lane），链数与寄存器分配都不同于手写瓦片，拿它当
        /// 「瓦片体的端口上限」是口径混用；本探针才是同一个形状。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        pub fn port_fma(rounds: usize) -> bool {
            if rounds == 0 {
                return false;
            }
            // 安全性：aarch64 基线含 NEON，函数体内只有寄存器运算，不触碰内存。
            #[allow(unsafe_code)]
            unsafe {
                fma_pure_rounds(rounds);
            }
            true
        }

        /// 纯 FMA 循环：16 个累加器互不依赖，结果进 `black_box` 防整段消除。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        #[allow(unsafe_code)]
        #[inline(never)]
        unsafe fn fma_pure_rounds(rounds: usize) {
            use std::arch::aarch64::*;
            unsafe {
                let mut acc = [vdupq_n_f32(0.0); 16];
                let av = vdupq_n_f32(1.000_000_1);
                for _ in 0..rounds {
                    // 16 个累加器互不依赖：每条只读自己，不构成串行链。
                    for chain in &mut acc {
                        *chain = vfmaq_f32(*chain, av, av);
                    }
                }
                std::hint::black_box(acc);
            }
        }

        /// `fma_step` 的形状校验：`k` 非零，A/B 各留足 `NR * k` 个 f32。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        fn block_shaped(a: &[f32], b: &[f32], k: usize) -> bool {
            k != 0 && a.len() >= NR * k && b.len() >= NR * k
        }

        /// f32 **取数受限**端口：与生产瓦片体逐指令一致，每步 4 条 `vld1q` + 16 条 FMA。
        ///
        /// **为什么要它**：纯端口探针环内零 load，是理想上限；生产核每步要发 4 条 load。
        /// 两者之差就是取数代价，也是「K7 的缺口里多少是结构注定、多少是核外记账」的分界线。
        /// K8 的同一对照给出 i8 瓦片体单线程 99.3%、18 线程 84.1%，故那 84% 就是瓦片体的真实天花板。
        ///
        /// `k` 取 1024 使地址在环内单调推进——地址不随循环变化时 LLVM 会把 load 整体提出循环，
        /// 那样测的是纯端口而非「取数受限」端口（i8 侧第一版就是这么错的，见 [`super::i8::port_sdot_block`]）。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        pub fn port_fma_block(rounds: usize, a: &[f32], b: &[f32], k: usize) -> bool {
            if rounds == 0 || !block_shaped(a, b, k) {
                return false;
            }
            // 安全性：A/B 各 `NR * k` 个 f32 已在上面校验。
            #[allow(unsafe_code)]
            unsafe {
                fma_block_rounds(rounds, a.as_ptr(), b.as_ptr(), k);
            }
            true
        }

        /// 取数受限循环：每轮跑一次完整瓦片体，**零记账**（不读回 `out`、不写回 `out`）。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        #[allow(unsafe_code)]
        #[inline(never)]
        unsafe fn fma_block_rounds(rounds: usize, ap: *const f32, bp: *const f32, k: usize) {
            use std::arch::aarch64::*;
            unsafe {
                let mut acc = [vdupq_n_f32(0.0); 16];
                for _ in 0..rounds {
                    fma_step(&mut acc, ap, bp, k);
                }
                std::hint::black_box(acc);
            }
        }

        /// 读回上一步的 16 个累加向量（非首块续算用）。
        #[cfg(target_arch = "aarch64")]
        #[allow(unsafe_code)]
        pub(super) fn micro_load(
            out: &[f32],
            row0: usize,
            n: usize,
            j: usize,
        ) -> [std::arch::aarch64::float32x4_t; 16] {
            unsafe {
                use std::arch::aarch64::*;
                let mut acc = [vdupq_n_f32(0.0); 16];
                let prev = out.as_ptr();
                let r = row0;
                acc[0] = vld1q_f32(prev.add(r * n + j));
                acc[1] = vld1q_f32(prev.add(r * n + j + 4));
                acc[2] = vld1q_f32(prev.add((r + 1) * n + j));
                acc[3] = vld1q_f32(prev.add((r + 1) * n + j + 4));
                acc[4] = vld1q_f32(prev.add((r + 2) * n + j));
                acc[5] = vld1q_f32(prev.add((r + 2) * n + j + 4));
                acc[6] = vld1q_f32(prev.add((r + 3) * n + j));
                acc[7] = vld1q_f32(prev.add((r + 3) * n + j + 4));
                acc[8] = vld1q_f32(prev.add((r + 4) * n + j));
                acc[9] = vld1q_f32(prev.add((r + 4) * n + j + 4));
                acc[10] = vld1q_f32(prev.add((r + 5) * n + j));
                acc[11] = vld1q_f32(prev.add((r + 5) * n + j + 4));
                acc[12] = vld1q_f32(prev.add((r + 6) * n + j));
                acc[13] = vld1q_f32(prev.add((r + 6) * n + j + 4));
                acc[14] = vld1q_f32(prev.add((r + 7) * n + j));
                acc[15] = vld1q_f32(prev.add((r + 7) * n + j + 4));
                acc
            }
        }

        /// 把 16 个累加向量写回 `out` 的 8×8 块。
        #[cfg(target_arch = "aarch64")]
        #[allow(unsafe_code)]
        pub(super) fn micro_store(
            out: &mut [f32],
            row0: usize,
            n: usize,
            j: usize,
            acc: [std::arch::aarch64::float32x4_t; 16],
        ) {
            unsafe {
                use std::arch::aarch64::*;
                let op = out.as_mut_ptr();
                let r = row0;
                vst1q_f32(op.add(r * n + j), acc[0]);
                vst1q_f32(op.add(r * n + j + 4), acc[1]);
                vst1q_f32(op.add((r + 1) * n + j), acc[2]);
                vst1q_f32(op.add((r + 1) * n + j + 4), acc[3]);
                vst1q_f32(op.add((r + 2) * n + j), acc[4]);
                vst1q_f32(op.add((r + 2) * n + j + 4), acc[5]);
                vst1q_f32(op.add((r + 3) * n + j), acc[6]);
                vst1q_f32(op.add((r + 3) * n + j + 4), acc[7]);
                vst1q_f32(op.add((r + 4) * n + j), acc[8]);
                vst1q_f32(op.add((r + 4) * n + j + 4), acc[9]);
                vst1q_f32(op.add((r + 5) * n + j), acc[10]);
                vst1q_f32(op.add((r + 5) * n + j + 4), acc[11]);
                vst1q_f32(op.add((r + 6) * n + j), acc[12]);
                vst1q_f32(op.add((r + 6) * n + j + 4), acc[13]);
                vst1q_f32(op.add((r + 7) * n + j), acc[14]);
                vst1q_f32(op.add((r + 7) * n + j + 4), acc[15]);
            }
        }

        /// 标量微内核：无 SIMD 的架构（以及 x86_64 上未检出 AVX2/FMA 的机器）的兜底。
        #[cfg(not(target_arch = "aarch64"))]
        pub(super) fn micro_packed_scalar(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
            let Tile {
                row0,
                j,
                n,
                kb,
                first,
            } = tile;
            for r in 0..MR {
                for c in 0..NR {
                    let slot = (row0 + r) * n + j + c;
                    let mut acc = if first { 0.0f32 } else { out[slot] };
                    for p in 0..kb {
                        acc += panel[p * MR + r] * b[p * NR + c];
                    }
                    out[slot] = acc;
                }
            }
        }

        /// x86_64 的 8×8 微内核：AVX2 + FMA，8 个 `__m256` 累加器（每行一个、8 列一个向量）。
        #[cfg(target_arch = "x86_64")]
        pub(super) fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                micro_packed_avx2(panel, b, out, tile);
            } else {
                micro_packed_scalar(panel, b, out, tile);
            }
        }

        /// 8×8 AVX2+FMA 微内核：每行一个 `__m256` 累加器，逐收缩步广播 A 值 × B 向量做 FMA。
        #[cfg(target_arch = "x86_64")]
        #[allow(unsafe_code)]
        pub(super) fn micro_packed_avx2(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
            let Tile {
                row0,
                j,
                n,
                kb,
                first,
            } = tile;
            use std::arch::x86_64::*;
            unsafe {
                let zero = _mm256_setzero_ps();
                let pp = panel.as_ptr();
                let bp = b.as_ptr();
                let mut acc = if first {
                    [zero; 8]
                } else {
                    micro_load(out, row0, n, j)
                };
                for p in 0..kb {
                    let bv = _mm256_loadu_ps(bp.add(p * NR));
                    // 手写 FMA 微内核：下标形式比迭代器形式更贴近指令序列，便于逐条核对。
                    #[allow(clippy::needless_range_loop)]
                    for r in 0..MR {
                        acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(*pp.add(p * MR + r)), bv, acc[r]);
                    }
                }
                micro_store(out, row0, n, j, acc);
            }
        }

        /// 读回上一步的 8 个累加向量（x86_64 非首块续算用）。
        #[cfg(target_arch = "x86_64")]
        #[allow(unsafe_code)]
        pub(super) fn micro_load(
            out: &[f32],
            row0: usize,
            n: usize,
            j: usize,
        ) -> [std::arch::x86_64::__m256; 8] {
            unsafe {
                use std::arch::x86_64::*;
                let prev = out.as_ptr();
                let r = row0;
                [
                    _mm256_loadu_ps(prev.add(r * n + j)),
                    _mm256_loadu_ps(prev.add((r + 1) * n + j)),
                    _mm256_loadu_ps(prev.add((r + 2) * n + j)),
                    _mm256_loadu_ps(prev.add((r + 3) * n + j)),
                    _mm256_loadu_ps(prev.add((r + 4) * n + j)),
                    _mm256_loadu_ps(prev.add((r + 5) * n + j)),
                    _mm256_loadu_ps(prev.add((r + 6) * n + j)),
                    _mm256_loadu_ps(prev.add((r + 7) * n + j)),
                ]
            }
        }

        /// 把 8 个累加向量写回 `out` 的 8×8 块（x86_64）。
        #[cfg(target_arch = "x86_64")]
        #[allow(unsafe_code)]
        pub(super) fn micro_store(
            out: &mut [f32],
            row0: usize,
            n: usize,
            j: usize,
            acc: [std::arch::x86_64::__m256; 8],
        ) {
            unsafe {
                use std::arch::x86_64::*;
                let op = out.as_mut_ptr();
                let r = row0;
                for (i, v) in acc.into_iter().enumerate() {
                    _mm256_storeu_ps(op.add((r + i) * n + j), v);
                }
            }
        }

        /// 无 SIMD 的架构：直接走标量微内核（`micro_packed_scalar` 在这些目标上不会被别处引用）。
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        pub(super) fn micro_packed(panel: &[f32], b: &[f32], out: &mut [f32], tile: Tile) {
            micro_packed_scalar(panel, b, out, tile);
        }

        /// 一个待算的标量尾块：左上角 `(i, j)`、覆盖 `mr × nr`，矩阵列宽 `n`、内积长 `k`。
        #[derive(Clone, Copy)]
        pub(super) struct Span {
            pub(super) i: usize,
            pub(super) j: usize,
            pub(super) mr: usize,
            pub(super) nr: usize,
            pub(super) n: usize,
            pub(super) k: usize,
            /// `a` 的整行宽：内积分块时 `k` 只是当前块，跨块偏移要按整行宽算。
            pub(super) stride: usize,
        }

        /// 权重布局：`Nt` 是收缩轴在前的 `b[k][n]`，`Tn` 是按输出行常驻的 `w[n][k]`。
        #[derive(Clone, Copy)]
        pub(super) enum Layout {
            Nt,
            Tn,
        }

        /// 权重在 `w` 里的寻址三元组：`(基址, 内积步长, 列步长)`——两种布局在这一处收敛。
        pub(super) fn weight_stride(span: Span, k0: usize, layout: Layout) -> (usize, usize, usize) {
            match layout {
                Layout::Nt => (k0 * span.n + span.j, span.n, 1),
                Layout::Tn => (span.j * span.stride + k0, 1, span.stride),
            }
        }

        /// 标量尾块：`out[i..i+mr, j..j+nr]` 逐元素点积（与微内核同一组累加口径，容差内一致）。
        pub(super) fn edge(
            a: &[f32],
            w: &[f32],
            out: &mut [f32],
            span: Span,
            k0: usize,
            first: bool,
            layout: Layout,
        ) {
            let (base, dp, dc) = weight_stride(span, k0, layout);
            for r in 0..span.mr {
                for c in 0..span.nr {
                    let slot = (span.i + r) * span.n + span.j + c;
                    let mut acc = if first { 0.0f32 } else { out[slot] };
                    let wc = base + c * dc;
                    for p in 0..span.k {
                        acc += a[(span.i + r) * span.stride + k0 + p] * w[wc + p * dp];
                    }
                    out[slot] = acc;
                }
            }
        }
    }

    /// i8 `sdot` 端口峰值探针（K8 的分母），只在测量容器里导出。
    #[cfg(all(feature = "bench", target_arch = "aarch64"))]
    pub use i8::{SDOT_BLOCK_MAC_PER_K, SDOT_MAC_PER_ROUND, port_sdot, port_sdot_block};

    /// 端口探针清单：哪几条指令路径可测，由这里统一回答（调用方不写 `cfg`）。
    #[cfg(feature = "bench")]
    mod probe {
        //! 职责：端口峰值探针清单——哪几条指令路径在本机可测，由这里统一回答。

        /// 一条端口峰值探针：**一条指令路径**的分母。
        ///
        /// 四个字段就是本模块与测量容器之间的全部协议：组名（`{group}/{threads}{suffix}`）、
        /// 一轮的操作次数（算吞吐的分子）、探针本身。清单由提供方给，见 [`probes`]。
        #[derive(Clone, Copy, Debug)]
        pub struct Probe {
            /// 基准组名前缀，如 `i8_peak`。
            pub group: &'static str,
            /// 组名后缀，用来在同前缀下区分通道，如 `_block`。
            pub suffix: &'static str,
            /// 探针一轮做多少次操作（MAC 数）。
            pub ops_per_round: usize,
            /// 跑 `rounds` 轮；本机没有这条指令路径时返回 `false` 且一条都不执行。
            pub run: fn(usize) -> bool,
        }

        /// 本机可测的端口探针清单——**由提供方回答**，调用方不写 `cfg`。
        ///
        /// 为什么不把 `cfg(nightly)` 留给调用方：`nightly` 是本 crate 的私有 cfg（`build.rs` 发出），
        /// 别的 crate 写它只会拿到 `unexpected_cfgs` 警告；而「这台机器上 SME 通道存不存在」本就是
        /// gemm 的判断。缺的通道**直接不在清单里**，而不是留在清单里再让每个调用方判空。
        ///
        /// 每项都必须是**同一条指令路径**的峰值：K7 的 SME 分母不能拿 NEON 峰值凑，i8 的 `sdot`
        /// 也不能拿 f32 的 `fma` 凑（F2 口径混用）。
        #[cfg(feature = "bench")]
        pub fn probes() -> &'static [Probe] {
            PROBES
        }

        /// SME 一轮的浮点次数：4 条 `fmopa` × 16×16 的 f32 瓦片 × 2 flop（乘 + 加）。
        #[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
        const SME_FLOPS_PER_ROUND: usize = 4 * 16 * 16 * 2;

        /// 纯 `fmopa` 与「`fmopa` + 每轮 5 条 `ld1w`」两条曲线：两者 `fmopa` 条数相同，
        /// 差额就是取数代价。
        #[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
        const SME_PURE: Probe = Probe {
            group: "sme_peak",
            suffix: "",
            ops_per_round: SME_FLOPS_PER_ROUND,
            run: super::sme::port_run,
        };
        #[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
        const SME_MIX: Probe = Probe {
            group: "sme_peak",
            suffix: "_mix",
            ops_per_round: SME_FLOPS_PER_ROUND,
            run: sme_mix_probe,
        };

        /// 混合探针的固定输入：地址不推进，即每轮命中同一批缓存行——这是取数端口的**上限**，
        /// 真实内核走不同行只会更慢，据此算出的取数代价偏小、缺口偏小，拿它下结论不夸大（A3）。
        #[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
        fn sme_mix_probe(rounds: usize) -> bool {
            static A: [f32; 16] = [1.0; 16];
            static B: [f32; 64] = [1.0; 64];
            super::sme::port_mix(rounds, &A, &B)
        }

        /// 纯 `sdot` 端口：零 load 的理想上限。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        const I8_PURE: Probe = Probe {
            group: "i8_peak",
            suffix: "",
            ops_per_round: 2 * super::i8::SDOT_MAC_PER_ROUND,
            run: super::i8::port_sdot,
        };

        /// 纯 f32 FMA 端口：16 个累加器、零 load，与生产瓦片体同形状（K7 的理想上限）。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        const FMA_PURE: Probe = Probe {
            group: "fma_peak",
            suffix: "",
            ops_per_round: 2 * super::micro::FMA_MAC_PER_K,
            run: super::micro::port_fma,
        };

        /// `k` 取 1024 使地址在环内单调推进——地址不随循环变化时 LLVM 会把 load 整体提出循环，
        /// 那样测的是纯端口而非「取数受限」端口（第一版就是这么错的，见 [`super::i8::port_sdot_block`]）。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        const PROBE_K: usize = 1024;

        /// 取数受限的 `sdot` 端口：每 `k16` 步 4 条 A load + 4 条 B load 配 16 条 `sdot`。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        const I8_BLOCK: Probe = Probe {
            group: "i8_peak",
            suffix: "_block",
            ops_per_round: 2 * super::i8::SDOT_BLOCK_MAC_PER_K * PROBE_K,
            run: |rounds| {
                static A: [i8; 4 * PROBE_K] = [3; 4 * PROBE_K];
                static B: [i8; 4 * PROBE_K] = [5; 4 * PROBE_K];
                super::i8::port_sdot_block(rounds, &A, &B, PROBE_K)
            },
        };

        /// 取数受限的 f32 端口：每步 2 条 A load + 2 条 B load 配 16 条 lane-indexed FMA，
        /// 与生产瓦片体 [`super::micro::fma_step`] 逐指令一致。
        #[cfg(all(feature = "bench", target_arch = "aarch64"))]
        const FMA_BLOCK: Probe = Probe {
            group: "fma_peak",
            suffix: "_block",
            ops_per_round: 2 * super::micro::FMA_MAC_PER_K * PROBE_K,
            run: |rounds| {
                static A: [f32; super::NR * PROBE_K] = [1.0; super::NR * PROBE_K];
                static B: [f32; super::NR * PROBE_K] = [1.0; super::NR * PROBE_K];
                super::micro::port_fma_block(rounds, &A, &B, PROBE_K)
            },
        };

        #[cfg(all(feature = "bench", nightly, target_arch = "aarch64"))]
        const PROBES: &[Probe] = &[SME_PURE, SME_MIX, I8_PURE, I8_BLOCK, FMA_PURE, FMA_BLOCK];
        #[cfg(all(
            feature = "bench",
            not(all(nightly, target_arch = "aarch64")),
            target_arch = "aarch64"
        ))]
        const PROBES: &[Probe] = &[I8_PURE, I8_BLOCK, FMA_PURE, FMA_BLOCK];
        /// 非 aarch64 没有 `sdot` 微内核，一条探针都没有。
        #[cfg(all(feature = "bench", not(target_arch = "aarch64")))]
        const PROBES: &[Probe] = &[];
    }
    #[cfg(feature = "bench")]
    pub use probe::{Probe, probes};

    use i8::{f32_calc, f32_pack, threads_override};
    pub use i8::{gemm_i8, gemm_i8_checked, gemm_i8_single};
    use micro::{Layout, Span, Sweep, Tile, edge, micro_packed};

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel/fast_tests.rs");
    }

    #[cfg(test)]
    mod scratch_tests {
        include!("../tests/kernel/scratch_tests.rs");
    }
}
pub mod gemm {
    //! GEMM 家族共享的轴契约、形状投影与格式无关的矩阵乘骨架。

    use crate::einsum::Bits;
    use crate::gemv::Element;
    use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
    use crate::quant::Row;
    use diagnostic::{Code, Error};

    /// 收缩轴长度（固定档）。
    pub const K: usize = 8;

    /// 输出列块宽度：B 的一个列块（`K × TILE_N` 个码）解包一次后被整段行复用。
    pub const TILE_N: usize = 64;

    /// 每字节打包码数：位宽 ≤ 8 时按 `8 / bits` 算；8 位逐元素为 1。
    pub const fn codes_per_byte(bits: Bits) -> usize {
        8 / bits.bits() as usize
    }

    /// 三轴：`m`/`k` 输入，`n` 输出。
    pub const AXES: &[AxisSpec] = &[
        AxisSpec {
            role: AxisRole::Input,
            label: 'm',
        },
        AxisSpec {
            role: AxisRole::Input,
            label: 'k',
        },
        AxisSpec {
            role: AxisRole::Output,
            label: 'n',
        },
    ];

    /// 形状：输出 `m×n`，收缩轴 `k` 固定。
    ///
    /// # Errors
    ///
    /// 当前实现不会失败。
    pub fn project_shape(m: usize, n: usize) -> Result<ShapeCtx, Error> {
        ShapeCtx::new([('m', m), ('k', K), ('n', n)])
    }

    /// 从形状上下文取一条轴；缺轴时按 `op` 标签报 [`Code::Invalid`]。
    pub fn axis(shape: &ShapeCtx, label: char, op: &str) -> Result<usize, Error> {
        shape
            .get(label)
            .ok_or_else(|| Error::new(Code::Invalid, format!("{op} 形状缺轴 `{label}`")))
    }

    /// 矩阵乘格式：一条 GEMM 里随精度变化的那部分。
    pub trait Gemm {
        /// 样本缓冲里的元素类型（[`Values`] 的两个闭集变体）。
        type Elem: Element;

        /// 诊断里点名的算子名：缺轴、尺寸不符时进错误文案。
        const OP: &'static str;

        /// A / B 输入元素位宽；输出恒为 f32。
        const BITS: Bits;

        /// 打包格式的静态配置（[`Row`]）：每字节码数、零点、样本偏移、对称码表单点收敛。
        fn config() -> Option<Row> {
            None
        }

        /// A（`m×K`）确定性取值：打包格式走 `packed_sample_a`；逐元素格式自行实现本函数。
        fn sample_a(m: usize) -> Vec<Self::Elem> {
            let Some(row) = Self::config() else {
                return Vec::new();
            };
            let conf = Conf::from(&row);
            Self::Elem::from_bytes(packed_sample_a(m, &conf))
        }

        /// B（`K×n`）确定性取值（口径同 [`sample_a`](Self::sample_a)）。
        fn sample_b(n: usize) -> Vec<Self::Elem> {
            let Some(row) = Self::config() else {
                return Vec::new();
            };
            let conf = Conf::from(&row);
            Self::Elem::from_bytes(packed_sample_b(n, &conf))
        }

        /// A 缓冲应有的元素个数（打包后按 [`Elem`](Self::Elem) 计）。
        fn a_len(m: usize) -> usize {
            let Some(row) = Self::config() else { return 0 };
            packed_a_len(m, &Conf::from(&row))
        }

        /// B 缓冲应有的元素个数。
        fn b_len(n: usize) -> usize {
            let Some(row) = Self::config() else { return 0 };
            packed_b_len(n, &Conf::from(&row))
        }

        /// 输出位置 `(i, j)` 的点积：A 第 `i` 行（连续 `K` 个元素）与 B 第 `j` 列（跨步 `n`）。
        fn dot(a: &[Self::Elem], b: &[Self::Elem], n: usize, i: usize, j: usize) -> f32 {
            let Some(row) = Self::config() else {
                return 0.0;
            };
            let Some(a) = Element::borrow_u8(a) else {
                return 0.0;
            };
            let Some(b) = Element::borrow_u8(b) else {
                return 0.0;
            };
            dot(a, b, n, i, j, &Conf::from(&row))
        }

        /// 内核本体：这份格式的 tile 矩阵乘，结果写满 `out`（`m × n`）。
        fn mul(a: &[Self::Elem], b: &[Self::Elem], m: usize, n: usize, out: &mut [f32]);

        /// 逐元素格式（不打包）的码值：测试侧闭式重算用；打包格式由骨架按位流解包，返回空切片。
        fn codes(values: &[Self::Elem]) -> Vec<i32> {
            let _ = values;
            Vec::new()
        }
    }

    /// 从样本里取输入缓冲：名字固定 `a` / `b`，元素类型由格式给出。
    fn take<'a, G: Gemm>(sample: &'a Sample, name: &str) -> Result<&'a [G::Elem], Error> {
        let buf = sample
            .buf(name)
            .ok_or_else(|| Error::new(Code::Missing, format!("样本里没有缓冲 `{name}`")))?;
        G::Elem::borrow(&buf.values).ok_or_else(|| {
            Error::new(
                Code::Invalid,
                format!(
                    "缓冲 `{name}` 不是 {}：这份格式的输入元素类型",
                    G::Elem::NAME
                ),
            )
        })
    }

    /// 标量代价：`flops` 固定 `2·m·k·n`；输入按 [`Gemm::BITS`] 计费，输出每元素 4 字节。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明（带 [`Gemm::OP`] 算子名）。
    pub fn cost<G: Gemm>(shape: &ShapeCtx) -> Result<Cost, Error> {
        let m = axis(shape, 'm', G::OP)?;
        let k = axis(shape, 'k', G::OP)?;
        let n = axis(shape, 'n', G::OP)?;
        let read = G::BITS.bytes_for((m * k) as u128) + G::BITS.bytes_for((k * n) as u128);
        Ok(Cost {
            flops: (2 * m * k * n) as f64,
            read: read as u64,
            write: (m * n) as u64 * 4,
        })
    }

    /// 确定性样本：A、B 由格式取值，Y 为 f32 零缓冲。
    pub fn sample<G: Gemm>(m: usize, n: usize) -> Sample {
        Sample {
            m,
            n,
            scalars: vec![("M", m as i64), ("K", K as i64), ("N", n as i64)],
            bufs: vec![
                G::Elem::buf("a", G::sample_a(m)),
                G::Elem::buf("b", G::sample_b(n)),
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; m * n]),
                },
            ],
        }
    }

    /// 打包配置：取样/长度/点积共用一行 [`Row`](crate::quant::Row) 的位宽事实。
    pub struct Conf {
        /// 每字节码数。
        pub cpb: usize,
        /// 对称零点（无符号码点回退有符号整数的平移量）。
        pub zero: i32,
        /// 样本码值偏移，长度等于 `cpb`。
        pub offsets: &'static [u32],
    }

    impl From<&crate::quant::Row> for Conf {
        fn from(row: &crate::quant::Row) -> Self {
            Self {
                cpb: row.cpb,
                zero: row.zero,
                offsets: row.offsets,
            }
        }
    }

    /// 解包字节内第 `slot` 个码，按对称零点回退成有符号整数。
    pub fn unpack(byte: u8, slot: usize, cpb: usize, zero: i32) -> i8 {
        let width = crate::quant::width_cpb(cpb);
        let mask = (1u8 << width) - 1;
        let value = (byte >> (slot * width)) & mask;
        (value as i32 - zero) as i8
    }

    /// 确定性打包位模式：`len` 个码按 `cpb` 每字节装一行；码值取
    /// `(字节下标 · seed + offsets[c]) mod 2^width`。
    pub fn pack(len: usize, seed: usize, cpb: usize, offsets: &[u32]) -> Vec<u8> {
        let width = crate::quant::width_cpb(cpb);
        let modulus = 1usize << width;
        (0..len / cpb)
            .map(|byte| {
                (0..cpb).fold(0u8, |acc, code| {
                    let value = ((byte * seed + offsets[code] as usize) % modulus) as u8;
                    acc | (value << (code * width))
                })
            })
            .collect()
    }

    /// 打包矩阵点积：A 第 `i` 行与 B 第 `j` 列，收缩轴固定 [`K`]，整数累加后转 f32。
    pub fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize, conf: &Conf) -> f32 {
        let bytes_per_row = K / conf.cpb;
        let mut acc: i32 = 0;
        for p in 0..K {
            let x = unpack(
                a[i * bytes_per_row + p / conf.cpb],
                p % conf.cpb,
                conf.cpb,
                conf.zero,
            ) as i32;
            let y_pos = p * n + j;
            let y = unpack(b[y_pos / conf.cpb], y_pos % conf.cpb, conf.cpb, conf.zero) as i32;
            acc += x * y;
        }
        acc as f32
    }

    /// A（`m×K`）确定性打包取样；seed 固定，同一形状取值可复现。
    pub fn packed_sample_a(m: usize, conf: &Conf) -> Vec<u8> {
        pack(m * K, 37, conf.cpb, conf.offsets)
    }

    /// B（`K×n`）确定性打包取样。
    pub fn packed_sample_b(n: usize, conf: &Conf) -> Vec<u8> {
        pack(K * n, 53, conf.cpb, conf.offsets)
    }

    /// A 打包后字节数：`m·K/cpb`。
    pub fn packed_a_len(m: usize, conf: &Conf) -> usize {
        m * K / conf.cpb
    }

    /// B 打包后字节数：`K·n/cpb`。
    pub fn packed_b_len(n: usize, conf: &Conf) -> usize {
        K * n / conf.cpb
    }

    /// 按元素位流解包一串打包字节：每字节 `cpb` 个码，低槽在前，码值减 `zero`。
    pub fn unpack_stream(bytes: &[u8], cpb: usize, zero: i32) -> Vec<i32> {
        bytes
            .iter()
            .flat_map(|&byte| (0..cpb).map(move |slot| unpack(byte, slot, cpb, zero) as i32))
            .collect()
    }

    /// 位流 → 整数码：每个码一个 `i32`，按行主序连续铺开（A 行宽 [`K`]、B 行宽 `n`）。
    pub fn decode_stream(codes: &mut [i32], bytes: &[u8], conf: &Conf) {
        for (idx, code) in codes.iter_mut().enumerate() {
            let byte = bytes[idx / conf.cpb];
            *code = unpack(byte, idx % conf.cpb, conf.cpb, conf.zero) as i32;
        }
    }

    /// 整数码的 tile 累加：外层按 [`TILE_N`] 走列块，块内 `p` 在外、列在内。
    pub fn accumulate_tiled(codes_a: &[i32], codes_b: &[i32], m: usize, n: usize, out: &mut [f32]) {
        let mut acc = vec![0i32; TILE_N.min(n)];
        let mut start = 0;
        while start < n {
            let width = TILE_N.min(n - start);
            let acc = &mut acc[..width];
            let (rows, _) = codes_a.as_chunks::<K>();
            for (row, arow) in rows.iter().take(m).enumerate() {
                acc.fill(0);
                for (p, &x) in arow.iter().enumerate() {
                    let segment = &codes_b[p * n + start..][..width];
                    rank1_update(acc, x, segment);
                }
                write_back(out, row, n, start, width, acc);
            }
            start += width;
        }
    }

    /// 块内 rank-1 更新：`acc[j] += x · segment[j]`。
    fn rank1_update(acc: &mut [i32], x: i32, segment: &[i32]) {
        for (slot, &y) in acc.iter_mut().zip(segment) {
            *slot += x * y;
        }
    }

    /// 把 `acc` 写回输出行：`out[row, start..start+width] = acc as f32`。
    fn write_back(out: &mut [f32], row: usize, n: usize, start: usize, width: usize, acc: &[i32]) {
        let target = &mut out[row * n + start..][..width];
        for (slot, &value) in target.iter_mut().zip(acc.iter()) {
            *slot = value as f32;
        }
    }

    /// 逐元素格式的整数码：`u8` 位模式按 `i8` 解释，与 [`Gemm::dot`] 的取码口径同一份。
    pub fn signed_codes(values: &[u8]) -> Vec<i32> {
        values.iter().map(|&byte| byte as i8 as i32).collect()
    }

    /// 打包格式的本体：两条位流各解一遍，再走 [`accumulate_tiled`]。
    pub fn packed_tiled(a: &[u8], b: &[u8], m: usize, n: usize, conf: &Conf, out: &mut [f32]) {
        let mut codes_a = vec![0i32; m * K];
        let mut codes_b = vec![0i32; K * n];
        decode_stream(&mut codes_a, a, conf);
        decode_stream(&mut codes_b, b, conf);
        accumulate_tiled(&codes_a, &codes_b, m, n, out);
    }

    /// 矩阵遍历：外层只写一份，内层点积落在 [`Gemm::dot`]。
    pub fn matmul<G: Gemm>(a: &[G::Elem], b: &[G::Elem], m: usize, n: usize, out: &mut [f32]) {
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = G::dot(a, b, n, i, j);
            }
        }
    }

    /// 标量参考：逐 `(i, j)` 走 [`Gemm::dot`]，是与 tile 内核**相互独立**的对拍真值。
    pub fn reference<G: Gemm>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let m = sample.m;
        let n = sample.n;
        let a = take::<G>(sample, "a")?;
        let b = take::<G>(sample, "b")?;
        let mut out = vec![0.0; m * n];
        matmul::<G>(a, b, m, n, &mut out);
        Ok(out)
    }

    /// 内核本体：先校验打包长度，再把结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 三项缓冲长度不符时返回说明（带 [`Gemm::OP`] 算子名）。
    pub fn forward<G: Gemm>(
        a: &[G::Elem],
        b: &[G::Elem],
        m: usize,
        n: usize,
        out: &mut [f32],
    ) -> diagnostic::Result<()> {
        if a.len() != G::a_len(m) || b.len() != G::b_len(n) || out.len() != m * n {
            return Err(Error::new(Code::Invalid, format!("{} 缓冲尺寸不符", G::OP)));
        }
        G::mul(a, b, m, n, out);
        Ok(())
    }

    /// 各打包 / 不打包格式共用的对拍：内核算出的结果必须与标量参考逐位一致。
    #[cfg(test)]
    pub(crate) fn matches_reference<G: Gemm>() {
        let sample = sample::<G>(16, 12);
        let expected = reference::<G>(&sample).expect("参考可计算");
        let a = take::<G>(&sample, "a").expect("a");
        let b = take::<G>(&sample, "b").expect("b");
        let mut out = vec![0.0; 16 * 12];
        forward::<G>(a, b, 16, 12, &mut out).expect("尺寸符");
        assert_eq!(out, expected);
    }

    /// 多形状对拍矩阵：每条格式在一档 `(m, n)` 配置上重放参考与内核，结果逐位一致。
    #[cfg(test)]
    pub(crate) fn matches_reference_over_shapes<G: Gemm>(shapes: &[(usize, usize)]) {
        for &(m, n) in shapes {
            let sample = sample::<G>(m, n);
            let expected = reference::<G>(&sample).expect("参考可计算");
            let a = take::<G>(&sample, "a").expect("a");
            let b = take::<G>(&sample, "b").expect("b");
            let mut out = vec![0.0; m * n];
            forward::<G>(a, b, m, n, &mut out).expect("尺寸符");
            assert_eq!(out, expected, "m={m}、n={n} 必须与参考逐位一致");
        }
    }

    /// 取样本 `a`/`b` 并按 `m×n` 跑一遍生产 [`forward`]：两条闭式校验的公共装配，
    /// 这一段必须逐字一致，否则 [`hand_closed`] 与 [`dense_closed`] 比的不是同一次调用。
    #[cfg(test)]
    fn closed_pair<'a, G: Gemm>(
        sample: &'a Sample,
        m: usize,
        n: usize,
        out: &mut [f32],
    ) -> (&'a [G::Elem], &'a [G::Elem]) {
        let a = take::<G>(sample, "a").expect("a");
        let b = take::<G>(sample, "b").expect("b");
        forward::<G>(a, b, m, n, out).expect("尺寸符");
        (a, b)
    }

    /// 整数点积的独立闭式校验：不经过 `G::dot` 与 `matmul`，测试侧自己按收缩轴重算。
    #[cfg(test)]
    pub(crate) fn hand_closed<G: Gemm>() {
        const M: usize = 5;
        const N: usize = 7;
        let sample = sample::<G>(M, N);
        let mut out = vec![0.0; M * N];
        let (a, b) = closed_pair::<G>(&sample, M, N, &mut out);

        let unpacked: Vec<i32> = sample_unpacked::<G>(a, b, M, N);
        for i in 0..M {
            for j in 0..N {
                let dot: i32 = (0..K)
                    .map(|p| unpacked[i * K + p] * unpacked[M * K + p * N + j])
                    .sum();
                assert_eq!(out[i * N + j], dot as f32, "({i}, {j}) 闭式点积不符");
            }
        }
    }

    /// 稠密格式的独立闭式校验：**不经过** `G::dot` / `matmul` / 生产 `mul`，测试侧在 f64 里
    /// 自己按收缩轴重算，再用前向误差界 `γ_k·Σ|a·b|`（`γ_k = k·u/(1−k·u)`、`u = 2^-24`）判。
    ///
    /// 这条校验之所以必要：`gemm_f32` 的 `mul` 与 [`reference`] 走的是**同一个** `matmul`，
    /// 两者逐位相等是被构造出来的，`matches_the_scalar_reference` 对稠密格式的证据为零（F6）。
    /// f64 重算与生产路径不共享任何一行算术，因此它能抓到 `mul` 自身的错。
    ///
    /// 界取**半个** `γ_k·Σ|a·b|`（`MARGIN = 0.5`）而不是整个：整个 `γ_k` 对任何重排求和几乎恒真，
    /// 断言它等于没断言。实测最坏比值 **0.1683**（`k=8` 样本、全 `γ_k` 口径），半个界即 3.0 倍余量。
    #[cfg(test)]
    pub(crate) fn dense_closed<G: Gemm>() {
        const M: usize = 16;
        const N: usize = 12;
        /// 实测界的收紧系数：留余量但不留到与 `γ_k` 同宽。
        const MARGIN: f64 = 0.5;
        let sample = sample::<G>(M, N);
        let mut out = vec![0.0; M * N];
        let (a, b) = closed_pair::<G>(&sample, M, N, &mut out);

        let (af, bf) = (f32_values::<G>(a), f32_values::<G>(b));
        let k = af.len() / M;
        assert_eq!(bf.len(), k * N, "B 的元素数不是 k×n");
        for i in 0..M {
            for j in 0..N {
                let (sum, mass) = exact(&af, &bf, k, N, i, j);
                let bound = MARGIN * test_support::gamma(k) * mass;
                let err = (f64::from(out[i * N + j]) - sum).abs();
                // 样本全零时界与误差同时为 0，断言恒真——那是最容易被忽略的空转，先把它钉掉。
                assert!(bound > 0.0, "({i}, {j}) 闭式界退化为 0：样本无判别力");
                assert!(
                    err <= bound,
                    "({i}, {j}) 前向误差 {err:e} 超过 {MARGIN}·γ_{k}·Σ|a·b| = {bound:e}"
                );
            }
        }
    }

    /// f64 里的精确点积与 `|a·b|` 的质量和——重排误差界的分子与分母。
    #[cfg(test)]
    fn exact(af: &[f32], bf: &[f32], k: usize, n: usize, i: usize, j: usize) -> (f64, f64) {
        (0..k).fold((0.0, 0.0), |(sum, mass), p| {
            let prod = f64::from(af[i * k + p]) * f64::from(bf[p * n + j]);
            (sum + prod, mass + prod.abs())
        })
    }

    /// 把元素读成 f32；打包格式（`u8` 元素）返回空，由调用方按 `k = 0` 自行失败。
    #[cfg(test)]
    fn f32_values<G: Gemm>(values: &[G::Elem]) -> Vec<f32> {
        match G::Elem::buf("v", values.to_vec()).values {
            Values::F32(v) => v,
            _ => Vec::new(),
        }
    }

    /// 把 A、B 解包成有符号整数码；测试侧独立读数，供 [`hand_closed`] 闭式重算。
    #[cfg(test)]
    fn sample_unpacked<G: Gemm>(a: &[G::Elem], b: &[G::Elem], m: usize, n: usize) -> Vec<i32> {
        let mut values = Vec::with_capacity(m * K + K * n);
        match G::config() {
            Some(row) => {
                let a: &[u8] = Element::borrow_u8(a).expect("打包格式元素为 u8");
                let b: &[u8] = Element::borrow_u8(b).expect("打包格式元素为 u8");
                values.extend(unpack_stream(a, row.cpb, row.zero));
                values.extend(unpack_stream(b, row.cpb, row.zero));
            }
            None => {
                values.extend(G::codes(a));
                values.extend(G::codes(b));
            }
        }
        values
    }

    /// 各格式共用的测试生成：打包格式（i2/i4/i8）跑参考、形状矩阵与独立闭式点积三则；
    /// 稠密 f32 跑独立 f64 闭式校验（整数闭式不适用，但那不等于只能自我对拍）。形状矩阵只定义一次。
    #[cfg(test)]
    mod test_support {
        include!("../tests/kernel/gemm_test_support.rs");
    }

    /// 生成一份 GEMM 精度变体的完整公开面：标记类型、[`Gemm`] 实现、`forward` 入口、
    /// [`annotation::model`] 注解与测试挂载。差异只随后面的关联项（取样 / 打包 / 点积契约）变化；
    /// 新增一档精度只需加一行调用，不另写样板。
    #[macro_export]
    macro_rules! gemm_variant {
        (
            $name:ident, $alias:ident,
            $op:tt, $kind:ident, $differentiable:literal,
            $elem:ty, $bits:expr,
            $( $methods:item ),* $(,)?
        ) => {
            #[doc = concat!("`", $op, "` 的标记类型：格式契约（取样 / 打包 / 点积）见 `crate::gemm` 的 `Gemm` 骨架。")]
            pub struct $name;

            impl $crate::gemm::Gemm for $name {
                type Elem = $elem;

                const OP: &'static str = $op;

                const BITS: $crate::einsum::Bits = $bits;

                $( $methods )*
            }

            // 基座契约（ADR 0050）：三条行为从 `Gemm` 骨架转发，粒度由注解的骨架默认值给出。
            $crate::operator_via!($name, $op, 2, (16, 16), $differentiable, super, $crate::gemm::AXES);

            #[doc = concat!("`", $op, "` 内核本体：结果写进调用方给的缓冲。")]
            pub fn forward(
                a: &[$elem],
                b: &[$elem],
                m: usize,
                n: usize,
                out: &mut [f32],
            ) -> diagnostic::Result<()> {
                $crate::gemm::forward::<$name>(a, b, m, n, out)
            }

            #[annotation::model(name = $op, shape = [16, 16], dtype = "f32", skeleton = "gemm")]
            #[doc = concat!("`", $op, "` 矩阵乘的语义别名。")]
            pub type $alias = $name;

            #[cfg(test)]
            mod tests {
                use super::*;
                $crate::gemm::test_support::$kind!($name);
            }
        };
    }

    pub mod i2 {
        //! `gemm_i2`：对称 int2 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int2"这一份格式
        //! 契约：A、B 是打包 2 位码（每字节 4 个，低 2 位在前），对称零点取配置表 [`I2`]；取样、
        //! 打包长度与整数点积由骨架按 [`I2`] 一行配置读出，点积在 `i32` 上精确累加后转 f32。
        pub use crate::gemm::{AXES, K, project_shape};

        use crate::quant::I2;

        crate::gemm_variant!(
            GemmI2,
            GemmI2Model,
            "gemm_i2",
            gemm_packed_tests,
            false,
            u8,
            I2.bits,
            fn config() -> Option<crate::quant::Row> {
                Some(I2)
            },
            fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
                crate::gemm::packed_tiled(a, b, m, n, &crate::gemm::Conf::from(&I2), out)
            },
        );
    }

    pub mod i4 {
        //! `gemm_i4`：对称 int4 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int4"这一份格式
        //! 契约：A、B 是打包 nibble（每字节 2 个，低 4 位在前、高 4 位在后），对称零点取配置表
        //! [`I4`]；取样、打包长度与整数点积由骨架按 [`I4`] 一行配置读出，点积在 `i32` 上精确累加。
        pub use crate::gemm::{AXES, K, project_shape};

        use crate::quant::I4;

        crate::gemm_variant!(
            GemmI4,
            GemmI4Model,
            "gemm_i4",
            gemm_packed_tests,
            false,
            u8,
            I4.bits,
            fn config() -> Option<crate::quant::Row> {
                Some(I4)
            },
            fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
                crate::gemm::packed_tiled(a, b, m, n, &crate::gemm::Conf::from(&I4), out)
            },
        );
    }

    pub mod i8 {
        //! `gemm_i8`：对称 int8 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"对称 int8"这一份格式
        //! 契约：`u8` 承载 int8 位模式，点积先在 `i32` 上精确累加（对应 CUDA `dp4a`：每 4 对 int8
        //! 乘积累加进一个 int32），再收敛成 f32。CPU 没有 dp4a，退化为标量整数点积。
        pub use crate::gemm::{AXES, K, project_shape};

        use crate::quant::I8;

        crate::gemm_variant!(
            GemmI8,
            GemmI8Model,
            "gemm_i8",
            gemm_packed_tests,
            false,
            u8,
            I8.bits,
            fn sample_a(m: usize) -> Vec<u8> {
                (0..m * K)
                    .map(|i| (((i * 37) % 16) as i8 - 8) as u8)
                    .collect()
            },
            fn sample_b(n: usize) -> Vec<u8> {
                (0..K * n)
                    .map(|i| (((i * 53) % 16) as i8 - 8) as u8)
                    .collect()
            },
            fn a_len(m: usize) -> usize {
                m * K
            },
            fn b_len(n: usize) -> usize {
                K * n
            },
            fn dot(a: &[u8], b: &[u8], n: usize, i: usize, j: usize) -> f32 {
                let row = &a[i * K..][..K];
                let acc: i32 = row
                    .iter()
                    .enumerate()
                    .map(|(p, packed)| (*packed as i8) as i32 * (b[p * n + j] as i8) as i32)
                    .sum();
                acc as f32
            },
            fn codes(values: &[u8]) -> Vec<i32> {
                crate::gemm::signed_codes(values)
            },
            fn mul(a: &[u8], b: &[u8], m: usize, n: usize, out: &mut [f32]) {
                let codes_a = crate::gemm::signed_codes(a);
                let codes_b = crate::gemm::signed_codes(b);
                crate::gemm::accumulate_tiled(&codes_a, &codes_b, m, n, out)
            },
        );
    }

    pub mod f32 {
        //! `gemm_f32`：稠密 f32 矩阵乘。骨架在 [`crate::gemm`]，本模块只有"稠密 f32"这一份格式
        //! 契约：元素类型、位宽、样本取值与点积算术。CPU 片段没有 MMA 原语，算术走通用稠密 tile
        //! （[`crate::matmul::matmul_tiled`]），与标量双重循环逐位一致。
        pub use crate::gemm::{AXES, K, project_shape};

        crate::gemm_variant!(
            Dense,
            GemmF32,
            "gemm_f32",
            gemm_dense_tests,
            true,
            f32,
            crate::einsum::Bits::ThirtyTwo,
            fn sample_a(m: usize) -> Vec<f32> {
                (0..m * K).map(|i| ((i % 5) as f32) * 0.25 - 0.5).collect()
            },
            fn sample_b(n: usize) -> Vec<f32> {
                (0..K * n).map(|i| ((i % 7) as f32) * 0.2 - 0.4).collect()
            },
            fn a_len(m: usize) -> usize {
                m * K
            },
            fn b_len(n: usize) -> usize {
                K * n
            },
            fn dot(a: &[f32], b: &[f32], n: usize, i: usize, j: usize) -> f32 {
                let row = &a[i * K..][..K];
                row.iter().enumerate().map(|(p, x)| x * b[p * n + j]).sum()
            },
            fn mul(a: &[f32], b: &[f32], m: usize, n: usize, out: &mut [f32]) {
                // 稠密 f32 不分块：`K = 8` 时一行 A 的 8 个权重就在寄存器里，分块（64 列块或
                // `matmul_tiled` 的三级块）只会多出累加器驻留与写回，实测反而慢 2~3 倍
                // （`gemm_f32` 基准的三档对照）。顺序累加同时保证与标量参考逐位相等。
                crate::gemm::matmul::<Self>(a, b, m, n, out)
            },
        );
    }
}
pub mod gemv {
    //! 一条 GEMV 骨架 + 权重格式契约。
    //!
    //! `gemv_i4` 与 `gemv_f32` 是同一副骨架的两种权重格式：求和表达式只差一条派生轴，样本只差
    //! 权重的取值，参考只差内层循环。差异全部收进 [`Format`] 的关联常量与方法，骨架（表达式缓存、
    //! 代价、样本、校验、遍历）只有一份——按格式**单态化**，因此 `Model` 里仍是三个 `fn` 指针，
    //! 每个指针指向一条独立的机器码，而不是一个在热路径上查 `match` 的分发器。
    //!
    //! # 为什么是 trait + const 泛型，而不是 `enum`
    //!
    //! 格式在编译期就确定：量化格式的每个元素占半个字节、稠密格式没有标度轴、累加按固定块宽推进。
    //! 写成运行时枚举会让 `forward` 在每行开始前多一次分支，也会让"分组必须是偶数""每行字节数
    //! 折半"这类约束从编译期退化成运行期检查。
    //!
    //! 哪些东西进类型、哪些留在数据里，是按**内核自己怎么收**划的：
    //!
    //! | 事实 | 落在哪 | 理由 |
    //! |---|---|---|
    //! | 权重元素类型、位宽、解包方式 | [`Format::Elem`] / [`FormulaOwner::OPERANDS`] / `forward` | 只有代码能表达，且必须进单态化 |
    //! | 稠密格式的分块宽度 | [`Dense`] 的 const 参数 | 它出现在最内层循环的 `chunks` 上，是真正的热路径常数 |
    //! | 量化格式的声明粒度 | [`Quantized`] 的 const 参数 | 它决定派生轴除数与样本里的标度个数（代价与样本，冷路径） |
    //! | 实际分组大小 | 运行时数据（`scales` 长度 / `gsz` 实参） | 模板本身就是 `int gsz` 的运行时参数，钉进类型会让训练侧按 8 元素分组的调用编译不过 |
    //!
    //! 最后一行是刻意的：把"内核已经做成运行时参数"的东西改成类型参数，换来的是每次不同分组都要
    //! 一份单态化副本与一次调用点上的匹配，而不是省下一条分支。

    use crate::model::{AxisRole, AxisSpec};

    /// 本骨架的基本轴：`m` 是规约轴（输入宽度），`n` 是输出轴。
    pub const AXES: &[AxisSpec] = &[
        AxisSpec {
            role: AxisRole::Input,
            label: 'm',
        },
        AxisSpec {
            role: AxisRole::Output,
            label: 'n',
        },
    ];

    /// 样本缓冲里能装的元素类型：闭集 `f32` / `u8`（[`Values`] 的两个变体）。
    macro_rules! scaled_operands {
        ($bits:expr) => {
            /// 三个输入都是读、只有输出是写；权重位宽取配置，其余为 32 位浮点。
            const OPERANDS: &'static [$crate::einsum::OperandSpec] =
                &$crate::gemv::scaled_operands($bits);
        };
    }

    /// 带标度格式的量化几何：粒度、派生轴、标度取值与步长。
    macro_rules! scaled_grain {
        ($grain:expr) => {
            const GRAIN: usize = $grain;

            const DERIVED: Option<(char, char, usize)> = Some(('g', 'm', $grain));

            const SCALES: Option<fn(usize) -> f32> = Some($crate::gemv::scale_of);

            const STEP: usize = $grain;
        };
    }

    mod format {
        //! 权重格式：把「一行权重怎么摆」抽象成 trait，解包与分组约束都由格式自己负责。
        use crate::einsum::{Bits, FormulaCache, OperandSpec};
        use crate::formula::FormulaOwner;
        use crate::model::{Buf, Values};
        use crate::quant::Row;

        use super::forward::{forward_dense, forward_rows, scaled_dot};
        use super::packed;
        use super::shape::non_empty;

        /// 权重元素：进样本、读缓冲、拼 `_Generic` 三处都按它分派。
        pub trait Element: Copy + PartialEq + std::fmt::Debug + 'static {
            /// 诊断里怎么称呼它。
            const NAME: &'static str;

            /// 从样本缓冲里借出同类元素；类型不符返回 `None`。
            fn borrow(values: &Values) -> Option<&[Self]>;

            /// 直接从 `u8` 切片借出本元素；`u8` 元素返回原样，`f32` 返回 `None`。
            fn borrow_u8(slice: &[Self]) -> Option<&[u8]>;

            /// 从打包字节构造本元素向量；`u8` 元素原样回收，`f32` 不可用。
            fn from_bytes(bytes: Vec<u8>) -> Vec<Self>;

            /// 装成同名缓冲。
            fn buf(name: &'static str, values: Vec<Self>) -> Buf;
        }

        impl Element for f32 {
            const NAME: &'static str = "f32";

            fn borrow(values: &Values) -> Option<&[Self]> {
                match values {
                    Values::F32(values) => Some(values),
                    Values::U8(_) => None,
                }
            }

            fn borrow_u8(_slice: &[Self]) -> Option<&[u8]> {
                None
            }

            fn from_bytes(_bytes: Vec<u8>) -> Vec<Self> {
                // f32 逐元素格式不参与打包取样：此路径在单态化时不可达，留空切片作安全占位。
                Vec::new()
            }

            fn buf(name: &'static str, values: Vec<Self>) -> Buf {
                Buf {
                    name,
                    values: Values::F32(values),
                }
            }
        }

        impl Element for u8 {
            const NAME: &'static str = "u8";

            fn borrow(values: &Values) -> Option<&[Self]> {
                match values {
                    Values::F32(_) => None,
                    Values::U8(values) => Some(values),
                }
            }

            fn borrow_u8(slice: &[Self]) -> Option<&[u8]> {
                Some(slice)
            }

            fn from_bytes(bytes: Vec<u8>) -> Vec<Self> {
                bytes
            }

            fn buf(name: &'static str, values: Vec<Self>) -> Buf {
                Buf {
                    name,
                    values: Values::U8(values),
                }
            }
        }

        /// 格式自证：派生轴的除数必须就是形状粒度。
        pub const fn derived_matches_grain(grain: usize, derived: Option<(char, char, usize)>) -> bool {
            match derived {
                Some((_, _, divisor)) => divisor == grain,
                None => true,
            }
        }

        /// 带分组标度的操作数表：权重按 `bits` 读，输入与标度都是 32 位浮点，输出写 32 位。
        pub const fn scaled_operands(bits: Bits) -> [OperandSpec; 4] {
            [
                OperandSpec::read("w", bits),
                OperandSpec::read("x", Bits::ThirtyTwo),
                OperandSpec::read("scales", Bits::ThirtyTwo),
                OperandSpec::write("y", Bits::ThirtyTwo),
            ]
        }

        /// 权重格式：一条 GEMV 里随精度变化的那部分。
        pub trait Format: FormulaOwner {
            /// 权重在样本里的元素类型。
            type Elem: Element;

            /// 形状粒度：`m` 必须是它的正整数倍；`1` 表示任意宽度。
            const GRAIN: usize;

            /// 派生轴 `(派生轴, 基准轴, 除数)`：量化格式声明一条（`g = m / GRAIN`），稠密格式没有。
            const DERIVED: Option<(char, char, usize)>;

            /// 标度取值函数；`None` 表示这份格式没有标度缓冲（每个标度不带参数）。
            const SCALES: Option<fn(usize) -> f32>;

            /// 模板每轮推进的元素数，也是样本里 `gsz` 的取值。
            const STEP: usize;

            /// 契约自证：本格式的声明必须自洽。
            const CHECK: () = {
                assert!(Self::GRAIN > 0, "形状粒度必须大于 0");
                assert!(
                    derived_matches_grain(Self::GRAIN, Self::DERIVED),
                    "派生轴的除数必须等于形状粒度：代价里的分组与样本里的分组必须同源"
                );
            };

            /// 一行权重占多少个元素。
            fn row(m: usize) -> usize;

            /// 确定性权重：同一形状永远得到同一组取值（cutile README §12.3 的"可复现"要求）。
            fn weights(m: usize, n: usize) -> Vec<Self::Elem>;

            /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
            fn forward(
                x: &[f32],
                weights: &[Self::Elem],
                scales: &[f32],
                out: &mut [f32],
            ) -> diagnostic::Result<()>;
        }

        /// 4 位打包配置标记：骨架测试按不同粒度复用这一份配置；生产 i4 lane 用同名 [`I4`](crate::gemv_i4::I4) 别名。
        pub struct QuantI4;

        impl PackedRow for QuantI4 {
            const ROW: Row = crate::quant::I4;

            fn cache() -> &'static FormulaCache {
                crate::formula_cell!()
            }
        }

        /// 量化格式：`GRAIN` 个元素共用一条标度，每两个元素占一个字节（4 位一元素）。
        pub type Quantized<const GRAIN: usize> = Packed<QuantI4, GRAIN>;

        /// 稠密逐元素格式的配置标记：每 `BLOCK` 个元素一块累加（最后一块可以短），无打包、无标度。
        pub struct Dense<const BLOCK: usize>;

        /// 逐元素对称 int8 格式的配置标记：不打包（每元素 1 字节、无 nibble 解包），有 per-group
        /// 标度。`u8` 承载 int8 的位模式，算术时按 `i8` 解释。`impl Plain` 在 `gemv_i8`。
        pub struct SymI8<const GRAIN: usize>;

        /// 打包格式的静态配置提供者：把一档精度的 [`Row`] 挂到一个标记类型上。
        pub trait PackedRow {
            /// 这档精度的静态配置（位宽 / 码数 / 零点 / 码表 / 偏移）。
            const ROW: Row;

            /// 这档精度自己的表达式缓存单元格。
            fn cache() -> &'static FormulaCache;
        }

        /// 配置驱动的打包权重格式：`GRAIN` 个元素共用一条标度，打包密度由 [`PackedRow::ROW`] 给出。
        pub struct Packed<P: PackedRow, const G: usize> {
            _config: core::marker::PhantomData<P>,
        }

        impl<P: PackedRow, const G: usize> FormulaOwner for Packed<P, G> {
            /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 `G` 个元素切出来的派生轴。
            const EXPR: &'static str = "nm,m,ng,n->n";

            scaled_operands!(P::ROW.bits);

            const FRAME: &'static str = "nn::gemv::formula";

            fn cache() -> &'static FormulaCache {
                // 缓存单元格由标记类型各自持有，避免泛型 static 被合并成跨配置共享的一份。
                P::cache()
            }
        }

        impl<P: PackedRow, const G: usize> Format for Packed<P, G> {
            type Elem = u8;

            scaled_grain!(G);

            fn row(m: usize) -> usize {
                // 粒度必须是打包码数的倍数，否则"每行 m/cpb 字节"没有意义；单态化时求值。
                const {
                    assert!(
                        G > 0 && G.is_multiple_of(P::ROW.cpb),
                        "打包粒度必须是正的每字节码数倍数"
                    )
                };
                m / P::ROW.cpb
            }

            fn weights(m: usize, n: usize) -> Vec<u8> {
                packed::weights(m, n, P::ROW.cpb)
            }

            fn forward(
                x: &[f32],
                weights: &[u8],
                scales: &[f32],
                out: &mut [f32],
            ) -> diagnostic::Result<()> {
                packed::forward_rows(x, weights, scales, out, P::ROW.cpb, P::ROW.table)
            }
        }

        /// 逐元素（不打包）GEMV 的静态配置：元素类型、量化几何、样本与元素反量化。
        pub trait Plain: FormulaOwner {
            /// 权重元素类型。
            type Elem: Element;

            /// 形状粒度：`m` 必须是它的正整数倍；`1` 表示任意宽度。
            const GRAIN: usize;

            /// 派生轴 `(派生轴, 基准轴, 除数)`；无派生轴为 `None`。
            const DERIVED: Option<(char, char, usize)>;

            /// 标度取值；`None` 表示没有标度缓冲。
            const SCALES: Option<fn(usize) -> f32>;

            /// 模板每轮推进元素数（样本 `gsz`；稠密同时是累加分块）。
            const STEP: usize;

            /// 确定性权重。
            fn weights(m: usize, n: usize) -> Vec<Self::Elem>;

            /// 元素反量化到 f32：int8 按 `i8` 解释，f32 恒等。
            fn to_f32(elem: Self::Elem) -> f32;

            /// 一段权重与输入的逐元素点积：默认逐元素标量推进。
            ///
            /// `f32` 与对称 `int8` 都覆盖成 [`simd::elementwise`] 的向量通道——这里是唯一一处
            /// 配置能自己决定"用哪台机器算这一行"，其余算术全部锁在骨架里。
            fn dot(w_row: &[Self::Elem], x: &[f32]) -> f32 {
                w_row
                    .iter()
                    .zip(x)
                    .map(|(elem, xv)| Self::to_f32(*elem) * xv)
                    .sum()
            }

            /// [`COLS`](super::simd::COLS) 行并行点积：`w` 是连续 [`COLS`](super::simd::COLS) 条行、
            /// 每行 `m` 个元素，结果写进 `out` 的 [`COLS`](super::simd::COLS) 个槽。
            ///
            /// 默认逐行转发 [`Plain::dot`]；稠密 `f32` 覆盖成 [`simd::elementwise_four`]，
            /// 一份 `x` 段摊到多行。行数由 `COLS` 编译期定死，没有运行期分支。
            fn dot_four(w: &[Self::Elem], x: &[f32], m: usize, out: &mut [f32]) {
                for (r, o) in out.iter_mut().enumerate() {
                    *o = Self::dot(&w[r * m..][..m], x);
                }
            }
        }

        /// 配置驱动的逐元素权重格式：把一条 [`Plain`] 配置挂进 [`Format`] 骨架。
        pub struct PlainFmt<C: Plain> {
            _config: core::marker::PhantomData<C>,
        }

        impl<C: Plain> FormulaOwner for PlainFmt<C> {
            const EXPR: &'static str = C::EXPR;

            const OPERANDS: &'static [OperandSpec] = C::OPERANDS;

            const FRAME: &'static str = C::FRAME;

            fn cache() -> &'static FormulaCache {
                // 缓存单元格由配置各自持有，避免泛型 static 被合并成跨配置共享的一份。
                C::cache()
            }
        }

        impl<C: Plain> Format for PlainFmt<C> {
            type Elem = C::Elem;

            const GRAIN: usize = C::GRAIN;

            const DERIVED: Option<(char, char, usize)> = C::DERIVED;

            const SCALES: Option<fn(usize) -> f32> = C::SCALES;

            const STEP: usize = C::STEP;

            fn row(m: usize) -> usize {
                // 逐元素不打包：一行权重就是 m 个元素。
                m
            }

            fn weights(m: usize, n: usize) -> Vec<Self::Elem> {
                C::weights(m, n)
            }

            fn forward(
                x: &[f32],
                weights: &[Self::Elem],
                scales: &[f32],
                out: &mut [f32],
            ) -> diagnostic::Result<()> {
                non_empty(out.len())?;
                if C::SCALES.is_some() {
                    forward_rows(x, weights, scales, out, 1, scaled_dot::<C>)
                } else {
                    forward_dense::<C>(x, weights, scales, out)
                }
            }
        }
    }
    mod forward {
        //! 前向内核：按行（解码）与按批（prefill）两条路径，点积本身分「带标度」与「稠密」两种。
        use diagnostic::{Code, Error};

        use super::format::Plain;
        use super::shape::{finite_scales, required_weights, scaled_groups};
        use super::simd::COLS;

        /// 带标度行骨架：分组、粒度、权重与标度守卫连同行切片统一收口，逐行点积由 `row_dot` 给出。
        pub(super) fn forward_rows<E>(
            x: &[f32],
            weights: &[E],
            scales: &[f32],
            out: &mut [f32],
            cpb: usize,
            row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
        ) -> diagnostic::Result<()> {
            let ctx = scaled_row_ctx(x.len(), out.len(), scales, weights.len(), cpb)?;
            forward_rows_scaled(x, weights, scales, out, ctx, row_dot);
            Ok(())
        }

        /// 逐行遍历一遍，算术由 `row_dot` 给：守卫已经过了（`ctx` 就是证据），这里只剩切片与调用。
        pub(crate) fn forward_rows_scaled<E>(
            x: &[f32],
            weights: &[E],
            scales: &[f32],
            out: &mut [f32],
            ctx: ScaledRow,
            row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
        ) {
            for (idx, value) in out.iter_mut().enumerate() {
                let w_row = &weights[idx * ctx.row..][..ctx.row];
                let s_row = &scales[idx * ctx.groups..][..ctx.groups];
                *value = row_dot(w_row, x, s_row, ctx.group);
            }
        }

        /// 行批骨架：输出行在外、输入行在里，一行权重只读一遍。
        fn forward_batch_rows<E>(
            x: &[f32],
            weights: &[E],
            scales: &[f32],
            geo: Geometry,
            out: &mut [f32],
            cpb: usize,
            row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
        ) -> diagnostic::Result<()> {
            let ctx = scaled_row_ctx(geo.m, geo.n, scales, weights.len(), cpb)?;
            forward_batch_rows_scaled(x, weights, scales, geo, out, ctx, row_dot);
            Ok(())
        }

        /// 稠密行批：无标度、输出行在外使一行权重只读一遍，守卫与 [`forward_dense`] 逐条对齐。
        fn forward_batch_dense<C: Plain>(
            x: &[f32],
            weights: &[C::Elem],
            scales: &[f32],
            geo: Geometry,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            dense_scales(scales)?;
            required_weights(weights.len(), geo.n, geo.m)?;
            for row in 0..geo.n {
                let w_row = &weights[row * geo.m..][..geo.m];
                for r in 0..geo.rows {
                    out[r * geo.n + row] = C::dot(w_row, &x[r * geo.m..][..geo.m]);
                }
            }
            Ok(())
        }

        /// 稠密格式的标度守卫：稠密没有标度，收到任何一条都是调用方的形状错误。
        fn dense_scales(scales: &[f32]) -> diagnostic::Result<()> {
            if scales.is_empty() {
                return Ok(());
            }
            Err(Error::new(
                Code::Invalid,
                format!("稠密格式没有标度，收到 {} 个", scales.len()),
            ))
        }

        /// 行批的逐行遍历，算术由 `row_dot` 给：与 [`forward_rows_scaled`] 同一个"守卫已过"约定。
        pub(crate) fn forward_batch_rows_scaled<E>(
            x: &[f32],
            weights: &[E],
            scales: &[f32],
            geo: Geometry,
            out: &mut [f32],
            ctx: ScaledRow,
            row_dot: impl Fn(&[E], &[f32], &[f32], usize) -> f32,
        ) {
            for j in 0..geo.n {
                let w_row = &weights[j * ctx.row..][..ctx.row];
                let s_row = &scales[j * ctx.groups..][..ctx.groups];
                for r in 0..geo.rows {
                    out[r * geo.n + j] = row_dot(w_row, &x[r * geo.m..][..geo.m], s_row, ctx.group);
                }
            }
        }

        /// 带标度格式的行守卫：分组、粒度倍数、权重长度与标度有限四项一次判齐，返回 `(groups, group)`。
        pub(super) fn scaled_row_ctx(
            m: usize,
            n: usize,
            scales: &[f32],
            weights: usize,
            cpb: usize,
        ) -> diagnostic::Result<ScaledRow> {
            let (groups, group) = scaled_groups(m, n, scales.len())?;
            if !group.is_multiple_of(cpb) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("分组 {group} 必须是 {cpb} 的倍数"),
                ));
            }
            required_weights(weights, n, m / cpb)?;
            finite_scales(scales, groups)?;
            Ok(ScaledRow {
                groups,
                group,
                row: m / cpb,
            })
        }

        /// 已过守卫的带标度行上下文：每行多少组、每组多少元素、每行多少个权重单位。
        #[derive(Clone, Copy, Debug)]
        pub(crate) struct ScaledRow {
            /// 每行多少组。
            pub groups: usize,
            /// 每组多少元素（必须是 `cpb` 的倍数）。
            pub group: usize,
            /// 每行多少个权重单位（打包格式是 `m / cpb` 个字节）。
            pub row: usize,
        }

        /// 行批逐元素入口：按配置选择带标度 / 无标度两条骨架，与 [`Format::forward`] 同一分叉。
        pub fn forward_plain_batch<C: Plain>(
            x: &[f32],
            weights: &[C::Elem],
            scales: &[f32],
            geo: Geometry,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            batch_geometry(x.len(), out.len(), geo)?;
            if C::SCALES.is_some() {
                forward_batch_rows(x, weights, scales, geo, out, 1, scaled_dot::<C>)
            } else {
                forward_batch_dense::<C>(x, weights, scales, geo, out)
            }
        }

        /// 行批几何：`rows` 行输入、输入行宽 `m`、输出行宽 `n`。
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct Geometry {
            /// 输入行数。
            pub rows: usize,
            /// 输入行宽（权重行宽）。
            pub m: usize,
            /// 输出行宽。
            pub n: usize,
        }

        impl Geometry {
            /// 由三个数直接构造；合法性留给入口守卫，构造本身不报错。
            pub const fn new(rows: usize, m: usize, n: usize) -> Self {
                Self { rows, m, n }
            }
        }

        /// 行批几何：行数非零，输入与输出的长度分别对上 `rows × m` 与 `rows × n`。
        pub(super) fn batch_geometry(x: usize, out: usize, geo: Geometry) -> diagnostic::Result<()> {
            if geo.rows == 0 {
                return Err(Error::new(Code::Invalid, "行批至少一行"));
            }
            let (want_x, want_out) = (
                geo.rows.saturating_mul(geo.m),
                geo.rows.saturating_mul(geo.n),
            );
            if x != want_x || out != want_out {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "行批尺寸不符：x={x}，out={out}，期望 {}×{}={want_x}、{}×{}={want_out}",
                        geo.rows, geo.m, geo.rows, geo.n
                    ),
                ));
            }
            Ok(())
        }

        /// 无标度逐元素格式的行遍历：换错算子（给了标度）必须报错，而不是静默忽略。
        pub(super) fn forward_dense<C: Plain>(
            x: &[f32],
            weights: &[C::Elem],
            scales: &[f32],
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            dense_scales(scales)?;
            let (m, n) = (x.len(), out.len());
            required_weights(weights.len(), n, m)?;
            let full = n - n % COLS;
            for j in (0..full).step_by(COLS) {
                let block = &weights[j * m..][..COLS * m];
                C::dot_four(block, x, m, &mut out[j..][..COLS]);
            }
            for j in full..n {
                out[j] = C::dot(&weights[j * m..][..m], x);
            }
            Ok(())
        }

        /// 一行的带标度点积：按组推进，每组解包点积一次乘标度（[`Plain::dot`] 承担单组点积）。
        pub(super) fn scaled_dot<C: Plain>(
            w_row: &[C::Elem],
            x: &[f32],
            s_row: &[f32],
            group: usize,
        ) -> f32 {
            let mut acc = 0.0f32;
            for ((w_chunk, scale), x_chunk) in w_row.chunks(group).zip(s_row).zip(x.chunks(group)) {
                acc += C::dot(w_chunk, x_chunk) * scale;
            }
            acc
        }
    }
    pub mod packed {
        //! 打包权重（每字节 `cpb` 个码）的取样与解包点积助手。
        /// 确定性打包权重：`n` 行、每行 `m / cpb` 个字节，同一形状取值可复现。
        pub fn weights(m: usize, n: usize, cpb: usize) -> Vec<u8> {
            (0..n * (m / cpb))
                .map(|idx| ((idx * 37) % 251) as u8)
                .collect()
        }

        /// 一行的点积：按分组推进，组内解包 `cpb` 个码查 `dequant` 表，整组共用同一个标度。
        pub fn dot_row(
            packed_row: &[u8],
            x: &[f32],
            s_row: &[f32],
            per_group: usize,
            group: usize,
            cpb: usize,
            dequant: &[f32],
        ) -> f32 {
            // 位宽走配置表派生：`width = 8 / cpb`，与 [`crate::quant::width_cpb`] 同一公式，不在这里第二份除法。
            let width = crate::quant::width_cpb(cpb);
            let mask = (1u8 << width) - 1;
            let mut acc = 0.0f32;
            for ((packed, scale), xs) in packed_row
                .chunks_exact(per_group)
                .zip(s_row)
                .zip(x.chunks_exact(group))
            {
                let mut dot = 0.0f32;
                for (byte, codes) in packed.iter().zip(xs.chunks(cpb)) {
                    for (slot, &code_slot) in codes.iter().enumerate() {
                        let code = usize::from((byte >> (slot * width)) & mask);
                        dot += dequant[code] * code_slot;
                    }
                }
                acc += dot * scale;
            }
            acc
        }

        /// 把骨架的逐行点积接口接到 [`dot_row`]：骨架只给分组元素数 `group`，解包点积还要每组字节数
        /// `group / cpb`、每字节码数与反量化表。
        fn row_dot(cpb: usize, dequant: &[f32]) -> impl Fn(&[u8], &[f32], &[f32], usize) -> f32 + '_ {
            move |w_row, x, s_row, group| dot_row(w_row, x, s_row, group / cpb, group, cpb, dequant)
        }

        /// 打包格式的单行入口：行宽按 `cpb` 折算，逐行点积走 `row_dot`。
        pub fn forward_rows(
            x: &[f32],
            weights: &[u8],
            scales: &[f32],
            out: &mut [f32],
            cpb: usize,
            dequant: &[f32],
        ) -> diagnostic::Result<()> {
            let ctx = super::forward::scaled_row_ctx(x.len(), out.len(), scales, weights.len(), cpb)?;
            if let Some(affine) = super::simd::admit(cpb, out.len(), ctx, dequant) {
                super::simd::forward_batch(&[x], weights, scales, affine, out);
                return Ok(());
            }
            super::forward::forward_rows_scaled(x, weights, scales, out, ctx, row_dot(cpb, dequant));
            Ok(())
        }

        /// 打包格式的行批入口：与 [`forward_rows`] 同一套算术与同一份 `row_dot`，只换遍历嵌套。
        pub fn forward_batch(
            x: &[f32],
            weights: &[u8],
            scales: &[f32],
            geo: super::forward::Geometry,
            out: &mut [f32],
            cpb: usize,
            dequant: &[f32],
        ) -> diagnostic::Result<()> {
            super::forward::batch_geometry(x.len(), out.len(), geo)?;
            let ctx = super::forward::scaled_row_ctx(geo.m, geo.n, scales, weights.len(), cpb)?;
            if let Some(affine) = super::simd::admit(cpb, geo.n, ctx, dequant) {
                let rows: Vec<&[f32]> = x.chunks(geo.m).collect();
                super::simd::forward_batch(&rows, weights, scales, affine, out);
                return Ok(());
            }
            super::forward::forward_batch_rows_scaled(
                x,
                weights,
                scales,
                geo,
                out,
                ctx,
                row_dot(cpb, dequant),
            );
            Ok(())
        }
    }
    mod shape {
        //! 形状上下文、确定性样本与标量参考：`#[annotation::model]` 要的三件产物。
        use diagnostic::{Code, Error};

        use crate::model::{Buf, Cost, Sample, ShapeCtx, Values};

        use super::format::{Element, Format};

        /// 与权重格式无关的输入向量：两条算子共用同一份取值。
        fn input(m: usize) -> Vec<f32> {
            (0..m)
                .map(|idx| ((idx * 7) % 13) as f32 * 0.25 - 1.0)
                .collect()
        }

        /// 量化格式共用的标度取值：`0.03125` 的 1..=4 倍，落在 2 的负幂网格，f32 精确。
        pub(crate) fn scale_of(idx: usize) -> f32 {
            0.031_25 * (1 + (idx % 4)) as f32
        }

        /// 把形状对翻译成投影骨架的具名轴上下文：`m` 是规约轴、`n` 是输出轴。
        pub fn project_shape(m: usize, n: usize) -> Result<ShapeCtx, Error> {
            ShapeCtx::new([('m', m), ('n', n)])
        }

        /// 标量代价：完全由 [`FormulaOwner::EXPR`] 与操作数位宽推出，本函数不含任何流量常数。
        ///
        /// 调用方只给基本轴；派生轴由格式自己声明——分组是算子语义，不是每个层都该抄一遍的常数。
        ///
        /// # Errors
        ///
        /// 形状与表达式不自洽（缺轴、长度为 0，或 `m` 不是 [`Format::GRAIN`] 的整数倍）时返回说明。
        pub fn cost<F: Format>(shape: &ShapeCtx) -> Result<Cost, Error> {
            let extents = match F::DERIVED {
                Some((label, base, divisor)) => shape.extents_with_group(label, base, divisor)?,
                None => shape.extents().clone(),
            };
            F::formula()?
                .cost(&extents)
                .map_err(|error| error.with("nn::gemv::cost"))
        }

        /// 确定性样本：同一形状永远得到同一组输入。
        pub fn sample<F: Format>(m: usize, n: usize) -> Sample {
            let mut bufs = vec![
                Buf {
                    name: "x",
                    values: Values::F32(input(m)),
                },
                F::Elem::buf("w", F::weights(m, n)),
            ];
            if let Some(scale) = F::SCALES {
                let groups = m / F::GRAIN.max(1);
                bufs.push(Buf {
                    name: "scales",
                    values: Values::F32((0..n * groups).map(scale).collect()),
                });
            }
            bufs.push(Buf {
                name: "y",
                values: Values::F32(vec![0.0; n]),
            });
            Sample {
                m,
                n,
                scalars: vec![("M", m as i64), ("N", n as i64), ("gsz", F::STEP as i64)],
                bufs,
            }
        }

        /// 从样本里取权重：名字固定是 `w`，元素类型由格式给出。
        fn weights<F: Format>(sample: &Sample) -> diagnostic::Result<&[F::Elem]> {
            let buf = sample
                .buf("w")
                .ok_or_else(|| Error::new(Code::Missing, "样本里没有缓冲 `w`"))?;
            F::Elem::borrow(&buf.values).ok_or_else(|| {
                Error::new(
                    Code::Invalid,
                    format!("缓冲 `w` 不是 {}：这份格式的权重元素类型", F::Elem::NAME),
                )
            })
        }

        /// 输出维度不能为 0：三条格式与参考实现共同的前置约束。
        pub(crate) fn non_empty(n: usize) -> Result<(), Error> {
            if n == 0 {
                return Err(Error::new(Code::Invalid, "输出维度不能为 0"));
            }
            Ok(())
        }

        /// 带标度格式的公共前置约束：`scales` 能按行均分、分组能整除输入宽度。
        pub(crate) fn scaled_groups(
            m: usize,
            n: usize,
            scales: usize,
        ) -> Result<(usize, usize), Error> {
            non_empty(n)?;
            if !scales.is_multiple_of(n) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("scale 长度 {scales} 不是 {n} 的整数倍"),
                ));
            }
            let groups = scales / n;
            if groups == 0 || !m.is_multiple_of(groups) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("scale 长度 {scales} 推不出能整除 {m} 的分组数"),
                ));
            }
            Ok((groups, m / groups))
        }

        /// 标度值守卫：每组分标度必须是有限值；非有限标度会把整组乘积带进 NaN/Inf，且与格式无关。
        pub(crate) fn finite_scales(scales: &[f32], groups: usize) -> Result<(), Error> {
            let bad = scales.iter().any(|scale| !scale.is_finite());
            if bad {
                return Err(Error::new(
                    Code::Invalid,
                    format!("标度含非有限值（{groups} 组）"),
                ));
            }
            Ok(())
        }

        /// 权重至少要够 `n` 行、每行 `row` 个元素，返回这个下界。
        pub(crate) fn required_weights(weights: usize, n: usize, row: usize) -> Result<usize, Error> {
            let want = n.saturating_mul(row);
            if weights < want {
                return Err(Error::new(
                    Code::Invalid,
                    format!("权重长度 {weights} < {want}"),
                ));
            }
            Ok(want)
        }

        /// 标量参考：校验由骨架给出，算术由格式给出。
        ///
        /// 骨架负责的部分与格式无关——输出非空、输入够长、权重够长、结果写进一块新缓冲；格式负责
        /// 自己的解包与分组约束。于是"再加一份精度"不需要重写这些检查。
        ///
        /// # Errors
        ///
        /// 缓冲缺失、元素类型不符、长度不足，或格式自己的约束（分组、标度个数）不成立时返回说明。
        pub fn reference<F: Format>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
            let x = sample.f32("x")?;
            let weights = weights::<F>(sample)?;
            let scales = scales::<F>(sample)?;
            let (m, n) = (sample.m, sample.n);
            gauge(x.len(), m, n)?;
            run::<F>(&x[..m], weights, scales, m, n)
        }

        /// 标度缓冲：只有格式声明了 [`Format::SCALES`] 才读；没有标度的格式拿空切片。
        fn scales<F: Format>(sample: &Sample) -> diagnostic::Result<&[f32]> {
            match F::SCALES {
                Some(_) => sample.f32("scales"),
                None => Ok(&[]),
            }
        }

        /// 骨架侧的尺寸判据：输出非空、输入够长。只问答不答格式——权重够不够由 [`run`] 判。
        fn gauge(x: usize, m: usize, n: usize) -> diagnostic::Result<()> {
            non_empty(n)?;
            if x < m {
                return Err(Error::new(Code::Invalid, format!("输入长度 {x} < {m}")));
            }
            Ok(())
        }

        /// 权重下界够了之后，把结果交给格式自己的算术（分配也在这里，格式只写缓冲）。
        fn run<F: Format>(
            x: &[f32],
            weights: &[F::Elem],
            scales: &[f32],
            m: usize,
            n: usize,
        ) -> diagnostic::Result<Vec<f32>> {
            let want = required_weights(weights.len(), n, F::row(m))?;
            let mut y = vec![0.0f32; n];
            F::forward(x, &weights[..want], scales, &mut y)?;
            Ok(y)
        }
    }
    #[cfg(nightly)]
    mod simd {
        //! 真 SIMD 点积，两条通道：
        //!
        //! * [`forward`]／[`forward_batch`]——打包格式（一字节 `CPB` 个码）。解包、偏移、乘加全在
        //!   `LANES` 宽的车道上，[`COLS`] 行权重并行，一份 `x` 块摊到 4 行；`x` 就地转置，不额外分配。
        //! * [`elementwise`]——逐元素格式（稠密 `f32`、对称 `int8`）。`LANES` 条车道 × [`CHAINS`] 条
        //!   独立累加链，权重按 [`Load`] 取成车道。
        //!
        //! 两条通道的行尾都只归约一次，于是舍入只发生在"每个元素乘加一次"这一层。

        use std::simd::prelude::{SimdFloat, SimdInt, SimdUint};
        use std::simd::{Simd, StdFloat};

        use super::ScaledRow;

        /// 4 列累加器：一次算 4 行，4 条向量链各持一组寄存器，一份 `x` 块摊到 4 行。
        pub(crate) const COLS: usize = 4;

        /// 宽档车道数：一条 AVX2 `ymm`、四条 NEON 寄存器。
        const WIDE: usize = 16;

        /// 窄档车道数：一组凑不满 [`WIDE`] 时用，半个 `ymm`、两条 NEON 寄存器。
        const NARROW: usize = 8;

        /// 逐元素通道的独立累加链数：一条车道链的乘加延迟约 4 拍，4 条刚好把流水填满。
        const CHAINS: usize = 4;

        /// 反量化表的仿射零：`table[code] == code - zero` 时返回 `zero`，否则 `None`（走标量叶）。
        pub(crate) fn affine_zero(dequant: &[f32], codes: usize) -> Option<f32> {
            if dequant.len() != codes {
                return None;
            }
            let zero = -dequant.first()?;
            let affine = dequant
                .iter()
                .enumerate()
                .all(|(code, &value)| value == code as f32 - zero);
            affine.then_some(zero)
        }

        /// 车道数：一组里每槽位的字节数被 [`WIDE`] 整除取宽档，被 [`NARROW`] 整除取窄档，都不行就不走本通道。
        ///
        /// 车道数整除一组是硬前提：真整除才能让索引算术完全无分支、无掩码、无零头。真实分组
        /// （[`GROUP`](super::GROUP) 的倍数配 2 / 4 个码一字节）永远落在宽档或窄档，其余形状走标量叶，
        /// 算术一模一样，只是不向量化。
        fn lanes_of(per_group: usize) -> Option<usize> {
            if per_group.is_multiple_of(WIDE) {
                Some(WIDE)
            } else if per_group.is_multiple_of(NARROW) {
                Some(NARROW)
            } else {
                None
            }
        }

        /// 通过校验之后的通道参数：形状（[`ScaledRow`]）、码宽、车道数、反量化表的仿射零点。
        #[derive(Clone, Copy, Debug)]
        pub(crate) struct Affine {
            ctx: ScaledRow,
            cpb: usize,
            lanes: usize,
            zero: f32,
        }

        /// 校验并构造 [`Affine`]：`n` 是输出宽度（权重行数），`ctx` 是入口已经量好的行形状。
        pub(crate) fn admit(cpb: usize, n: usize, ctx: ScaledRow, dequant: &[f32]) -> Option<Affine> {
            // 码表长度 `1 << width`，`width = 8 / cpb`（与 `quant::width_cpb` 同一公式）。
            let zero = affine_zero(dequant, 1 << crate::quant::width_cpb(cpb))?;
            let lanes = lanes_of(ctx.group / cpb)?;
            (n >= COLS).then_some(Affine {
                ctx,
                cpb,
                lanes,
                zero,
            })
        }

        /// 唯一入口：一次带上所有输入行，单行只是 `rows` 只有一个元素。
        pub(crate) fn forward_batch(
            rows: &[&[f32]],
            weights: &[u8],
            scales: &[f32],
            affine: Affine,
            out: &mut [f32],
        ) {
            run(
                &Plan {
                    weights,
                    inputs: rows,
                    affine,
                    scales,
                },
                out,
            );
        }

        /// 内核的全部输入：权重、输入行、通道参数、标度。
        struct Plan<'a> {
            weights: &'a [u8],
            inputs: &'a [&'a [f32]],
            affine: Affine,
            scales: &'a [f32],
        }

        /// 两个轴一起单态化：`cpb` 决定槽位循环与位移量，车道数决定向量宽度，编译期全摊平。
        fn run(plan: &Plan<'_>, out: &mut [f32]) {
            match (plan.affine.cpb, plan.affine.lanes == WIDE) {
                (2, true) => columns::<2, WIDE>(plan, out),
                (2, false) => columns::<2, NARROW>(plan, out),
                (_, true) => columns::<4, WIDE>(plan, out),
                (_, false) => columns::<4, NARROW>(plan, out),
            }
        }

        /// 列遍历：整块走 `R = COLS`，不足一块的尾部逐列走 `R = 1`（同一个 [`dot`]、同一副累加器
        /// 形态），所以 `n` 不必是 [`COLS`] 的倍数。
        fn columns<const CPB: usize, const LANES: usize>(plan: &Plan<'_>, out: &mut [f32]) {
            let n = out.len() / plan.inputs.len();
            let mut j = 0;
            while j + COLS <= n {
                strip::<CPB, LANES, COLS>(plan, out, n, j);
                j += COLS;
            }
            if j < n {
                strip::<CPB, LANES, 1>(plan, out, n, j);
            }
        }

        /// 一列块：`R` 行权重与同一段 `x` 窗口在 L1 里轮转；权重与标度切片与行无关，只在块首取一次。
        fn strip<const CPB: usize, const LANES: usize, const R: usize>(
            plan: &Plan<'_>,
            out: &mut [f32],
            n: usize,
            j: usize,
        ) {
            let ctx = plan.affine.ctx;
            let (w, s) = (
                &plan.weights[j * ctx.row..][..R * ctx.row],
                &plan.scales[j * ctx.groups..][..R * ctx.groups],
            );
            for (r, input) in plan.inputs.iter().enumerate() {
                let dest = &mut out[r * n + j..][..R];
                dest.copy_from_slice(&dot::<CPB, LANES, R>(w, s, input, ctx, plan.affine.zero));
            }
        }

        /// `R` 行的点积：一组一条 `LANES` 车道链，`R` 行同时推进，**每行只在行尾归约一次**。
        ///
        /// `ctx.groups` 不是从 `input.len()` 反推出来的，而是入口 [`admit`] 量好的：形状守卫已经保证
        /// `input.len() == ctx.groups · ctx.group`，所以这里没有"一组都凑不满"这种需要断言的中间态。
        fn dot<const CPB: usize, const LANES: usize, const R: usize>(
            w: &[u8],
            s: &[f32],
            input: &[f32],
            ctx: ScaledRow,
            zero: f32,
        ) -> [f32; R] {
            let per_group = ctx.group / CPB;
            let shift = Simd::<f32, LANES>::splat(zero);
            let mut running = [Simd::<f32, LANES>::splat(0.0); R];
            for g in 0..ctx.groups {
                let at = g * per_group;
                let acc = group_acc::<CPB, LANES, R>(w, ctx.row, input, at, per_group, shift);
                for (r, (acc, running)) in acc.iter().zip(running.iter_mut()).enumerate() {
                    *running = acc.mul_add(Simd::splat(s[r * ctx.groups + g]), *running);
                }
            }
            running.map(SimdFloat::reduce_sum)
        }

        /// 一组：`CPB` 个槽位 × `per_group / LANES` 段，`R` 条车道链各累加一份。
        ///
        /// `x` 在这里就地转置——每段只取一次连续的 `LANES · CPB` 个元素，摊成 `CPB` 条车道向量，
        /// 于是 `x` 每元素只读一遍，入口那一遍整行重排连同它的分配一起消失。
        #[inline]
        fn group_acc<const CPB: usize, const LANES: usize, const R: usize>(
            w: &[u8],
            row: usize,
            input: &[f32],
            at: usize,
            per_group: usize,
            shift: Simd<f32, LANES>,
        ) -> [Simd<f32, LANES>; R] {
            let mut acc = [Simd::<f32, LANES>::splat(0.0); R];
            for byte in (at..at + per_group).step_by(LANES) {
                let block = &input[byte * CPB..][..LANES * CPB];
                for slot in 0..CPB {
                    let xs = slot_lanes::<CPB, LANES>(block, slot);
                    for (r, acc) in acc.iter_mut().enumerate() {
                        let bytes = Simd::<u8, LANES>::from_slice(&w[r * row + byte..][..LANES]);
                        let value = codes::<CPB, LANES>(bytes, slot).cast::<f32>() - shift;
                        *acc = value.mul_add(xs, *acc);
                    }
                }
            }
            acc
        }

        /// 一段输入的某个槽位，摊成 `LANES` 条车道：槽位 `s` 取第 `i · CPB + s` 个元素。
        #[inline]
        fn slot_lanes<const CPB: usize, const LANES: usize>(
            block: &[f32],
            slot: usize,
        ) -> Simd<f32, LANES> {
            let mut lane = [0.0f32; LANES];
            for (i, value) in lane.iter_mut().enumerate() {
                *value = block[i * CPB + slot];
            }
            Simd::from_array(lane)
        }

        /// 第 `slot` 个码的 `LANES` 个无符号码值：低位槽在前，`mask = (1 << width) - 1`。
        fn codes<const CPB: usize, const LANES: usize>(
            bytes: Simd<u8, LANES>,
            slot: usize,
        ) -> Simd<u8, LANES> {
            let width = 8 / CPB as u8;
            (bytes >> Simd::splat(width * slot as u8)) & Simd::splat((1 << width) - 1)
        }

        /// 逐元素通道的权重取法：稠密 `f32` 直取，对称 `int8` 按二补数重新解释。
        ///
        /// 一条通道只由权重元素类型选，[`elementwise`] 因此对两种格式是同一段代码。
        pub(crate) trait Load: Copy {
            /// [`WIDE`] 个元素一次取成 `f32` 车道。
            fn wide(block: &[Self]) -> Simd<f32, WIDE>;

            /// 单个元素：不足一条车道的零头用。
            fn one(elem: Self) -> f32;
        }

        impl Load for f32 {
            #[inline]
            fn wide(block: &[Self]) -> Simd<f32, WIDE> {
                Simd::from_slice(block)
            }

            #[inline]
            fn one(elem: Self) -> f32 {
                elem
            }
        }

        impl Load for u8 {
            /// `u8 → i8` 是位模式重解释，`i8 → f32` 是有符号展宽，两条都是单条指令。
            #[inline]
            fn wide(block: &[Self]) -> Simd<f32, WIDE> {
                Simd::<u8, WIDE>::from_slice(block)
                    .cast::<i8>()
                    .cast::<f32>()
            }

            #[inline]
            fn one(elem: Self) -> f32 {
                (elem as i8) as f32
            }
        }

        /// 连续点积骨架：`CHAINS` 条独立累加链轮转，权重按 [`Load`] 取成车道，行尾归约一次。
        ///
        /// 这是 `f32` 与对称 `int8` 共用的唯一一份点积——两种格式的差别只剩 [`Load`] 的取法。
        ///
        /// 链按 [`CHAINS`] 分段常量展开，`acc` 的下标因此是编译期常量、累加器整份留在向量寄存器里。
        /// 分段之后块 `g·CHAINS + c` 落在链 `c`，与"块下标取模"是同一张映射，故逐位不变——2026-11-22
        /// 实测两者同为 `gemv_f32/decode` 39.8–40.2 GiB/s（离散 0.5%），即常量下标只是把意图写明，
        /// 并没有换来速度：这条路径只在 `cfg(nightly)` 下编译，那台机器上的实测数字属于 stable 退路
        /// （`gemv/simd_fallback.rs`），本文件的热路径结论必须在那条通道上取。
        pub(crate) fn elementwise<T: Load>(w: &[T], x: &[f32]) -> f32 {
            let (w_groups, w_tail) = w.as_chunks::<{ WIDE * CHAINS }>();
            let (x_groups, x_tail) = x.as_chunks::<{ WIDE * CHAINS }>();
            let mut acc = [Simd::<f32, WIDE>::splat(0.0); CHAINS];
            for (wg, xg) in w_groups.iter().zip(x_groups) {
                for c in 0..CHAINS {
                    let (wc, xc) = (&wg[c * WIDE..][..WIDE], &xg[c * WIDE..][..WIDE]);
                    acc[c] = T::wide(wc).mul_add(Simd::from_slice(xc), acc[c]);
                }
            }
            let mut total = acc.iter().copied().map(SimdFloat::reduce_sum).sum::<f32>();
            for (w, x) in w_tail.iter().zip(x_tail) {
                total += T::one(*w) * *x;
            }
            total
        }

        /// [`COLS`] 行并行点积：一份 `x` 段摊到 [`COLS`] 行，权重每 64 B 的取数从 8 次降到 5 次。
        ///
        /// 行内按 [`WIDE`] 整除的部分走向量，零头逐行标量收口——零头重读 `x` 的代价可忽略，换来行宽
        /// 不必是 [`WIDE`] 的倍数。每行只有一条累加链：链的独立性由"行"提供（[`COLS`] 条独立链），
        /// 而不是由链数提供，所以累加器刚好 `COLS` 份，全部留在寄存器里。
        pub(crate) fn elementwise_four<T: Load>(w: &[T], x: &[f32], m: usize, out: &mut [f32]) {
            let mut acc = [Simd::<f32, WIDE>::splat(0.0); COLS];
            let full = m - m % WIDE;
            let row = |r: usize| &w[r * m..][..m];
            let chunks: [&[[T; WIDE]]; COLS] = [0, 1, 2, 3].map(|r| row(r).as_chunks::<WIDE>().0);
            let (x_head, x_tail) = x.split_at(full);
            for (k, xv) in x_head.as_chunks::<WIDE>().0.iter().enumerate() {
                let xv = Simd::from_slice(xv);
                for r in 0..COLS {
                    acc[r] = T::wide(&chunks[r][k]).mul_add(xv, acc[r]);
                }
            }
            for (r, o) in out.iter_mut().enumerate() {
                let mut total = acc[r].reduce_sum();
                for (wc, xc) in row(r)[full..].iter().zip(x_tail) {
                    total += T::one(*wc) * *xc;
                }
                *o = total;
            }
        }

        #[cfg(test)]
        mod tests {
            include!("../tests/kernel/simd_tests.rs");
        }
    }
    /// stable 退路：同名同签名，只是把车道换成标量。
    #[cfg(not(nightly))]
    mod simd {
        //! stable 退路：与 nightly 版的 `super::simd` 同名同签名，只是不做显式车道。
        //!
        //! 两边的差别只有「向量化没向量化」，语义一模一样：
        //!
        //! * [`Load::one`] + [`elementwise`] 保持与向量版**相同的分段形状**——`WIDE` 个元素一段、
        //!   `CHAINS` 条独立累加链轮转。段内逐条 `mul_add`，于是每条链累加的元素序列与向量版逐车道
        //!   一一对应，stable 上 LLVM 也能把这份源码自动向量化成近乎同一组指令；两份实现只差最后那
        //!   一步归约（`reduce_sum` 是树形，这里是顺序）。
        //! * 打包格式的通道（`admit` / `forward_batch`）在 stable 上恒为不可达：调用方
        //!   `gemv::packed` 本来就有标量叶 `forward::forward_rows_scaled` 兜底，那条路径一直都在。

        use core::convert::Infallible;

        use super::ScaledRow;

        /// 宽档车道数，与向量版一致：决定 [`elementwise`] 的分段宽度。
        const WIDE: usize = 16;

        /// 独立累加链数，与向量版一致：一条链的乘加延迟约 4 拍，4 条刚好把流水填满。
        const CHAINS: usize = 4;

        /// 行并行度：一次算 [`COLS`] 行，一份 `x` 段摊到这 [`COLS`] 行上。
        ///
        /// 实测每多一条行流就多一条 DRAM 预取流，2026-11-22 扫描 2/4/8 得到 55.1／48.9／44.7 GiB/s，
        /// 故取 2：`x` 复用与流数的最优点，再往上就是流数占主导。
        pub(crate) const COLS: usize = 2;

        /// 段宽：一条累加链一次推进的元素数，与 [`WIDE`] 同义但可独立调（见 [`COLS`] 的扫描）。
        const SPAN: usize = 8;

        /// 每条行的独立累加链数：行并行度只有 [`COLS`]，链不够就喂不满 4 拍的乘加延迟。
        const DEPTH: usize = 2;

        /// 逐元素通道的权重取法：稠密 `f32` 直取，对称 `int8` 按二补数重新解释。
        ///
        /// 一条通道只由权重元素类型选，[`elementwise`] 因此对两种格式是同一段代码。
        pub(crate) trait Load: Copy {
            /// 单个元素取成 `f32`。
            fn one(elem: Self) -> f32;
        }

        impl Load for f32 {
            fn one(elem: Self) -> f32 {
                elem
            }
        }

        impl Load for u8 {
            /// `u8 → i8` 是位模式重解释，`i8 → f32` 是有符号展宽。
            fn one(elem: Self) -> f32 {
                (elem as i8) as f32
            }
        }

        /// [`COLS`] 行并行点积：`w` 是连续 [`COLS`] 条行、每行 `m` 个元素，结果写进 `out` 的 [`COLS`] 个槽。
        ///
        /// 形状是 `COLS × DEPTH × SPAN` 的累加器：一行一份 `x` 段摊给 [`COLS`] 行共用，行内再拆
        /// [`DEPTH`] 条独立链去盖住乘加的 4 拍延迟。稳定版没有显式车道，靠的是循环体形状本身
        /// 够整齐，让 LLVM 自己向量化——所以 `SPAN · DEPTH` 必须是 2 的幂，否则整段退化成标量
        /// （实测 `COLS = 3` 掉到 30 GiB/s）。
        ///
        /// 三个常数都不是猜的：2026-11-22 在同机同口径下扫过 `COLS ∈ {2,4,8}`（55.1／48.9／44.7）
        /// 与 `SPAN ∈ {8,16,32,64}`、`DEPTH ∈ {1,2,3,4}`，定格在 2／8／2 —— 再大的 `COLS` 每多一条行流
        /// 就多一条 DRAM 预取流，与 `x` 的流争抢，收益反转为负。
        pub(crate) fn elementwise_four<T: Load>(w: &[T], x: &[f32], m: usize, out: &mut [f32]) {
            let mut acc = [[[0.0f32; SPAN]; DEPTH]; COLS];
            let group = SPAN * DEPTH;
            let full = m - m % group;
            let (x_head, x_tail) = x.split_at(full);
            for (g, xg) in x_head.as_chunks::<{ SPAN * DEPTH }>().0.iter().enumerate() {
                commit(&mut acc, w, m, xg, g * group);
            }
            for (r, (o, chain)) in out.iter_mut().zip(acc).enumerate() {
                *o = finish(w, m, r, full, x_tail, chain);
            }
        }

        /// 把一个 `x` 组（`SPAN · DEPTH` 个元素）乘进 [`COLS`] 行 × [`DEPTH`] 链的累加器。
        ///
        /// 单独成函数只为把嵌套压到门禁的 5 以内；`#[inline]` 保证与内联版逐指令一致——这是
        /// 排版，不是抽象层。
        #[inline]
        fn commit<T: Load>(
            acc: &mut [[[f32; SPAN]; DEPTH]; COLS],
            w: &[T],
            m: usize,
            xg: &[f32],
            at: usize,
        ) {
            for (r, row) in acc.iter_mut().enumerate() {
                for (d, chain) in row.iter_mut().enumerate() {
                    let xs = &xg[d * SPAN..][..SPAN];
                    let ws = &w[r * m + at + d * SPAN..][..SPAN];
                    for (lane, (wv, xv)) in ws.iter().zip(xs).enumerate() {
                        chain[lane] = T::one(*wv).mul_add(*xv, chain[lane]);
                    }
                }
            }
        }

        /// 归约一条链并吃掉行尾（`m` 不是 `SPAN · DEPTH` 整数倍时剩下的那几个）。
        #[inline]
        fn finish<T: Load>(
            w: &[T],
            m: usize,
            r: usize,
            full: usize,
            x_tail: &[f32],
            chain: [[f32; SPAN]; DEPTH],
        ) -> f32 {
            let mut total: f32 = chain.iter().flatten().sum();
            for (wc, xc) in w[r * m + full..][..m - full].iter().zip(x_tail) {
                total = T::one(*wc).mul_add(*xc, total);
            }
            total
        }

        /// 连续点积：`CHAINS` 条独立累加链轮转，行尾归约一次。
        pub(crate) fn elementwise<T: Load>(w: &[T], x: &[f32]) -> f32 {
            let (w_chunks, w_tail) = w.as_chunks::<WIDE>();
            let (x_chunks, x_tail) = x.as_chunks::<WIDE>();
            let mut acc = [[0.0f32; WIDE]; CHAINS];
            for (i, (w, x)) in w_chunks.iter().zip(x_chunks).enumerate() {
                let chain = &mut acc[i % CHAINS];
                for (k, (w, x)) in w.iter().zip(x).enumerate() {
                    chain[k] = T::one(*w).mul_add(*x, chain[k]);
                }
            }
            let mut total: f32 = acc.iter().flatten().sum();
            for (w, x) in w_tail.iter().zip(x_tail) {
                total = T::one(*w).mul_add(*x, total);
            }
            total
        }

        /// 打包格式的通道参数：stable 上没有向量通道，恒为 `None`，调用方落到标量叶。
        pub(crate) fn admit(
            _cpb: usize,
            _n: usize,
            _ctx: ScaledRow,
            _dequant: &[f32],
        ) -> Option<Infallible> {
            None
        }

        /// 只有 [`admit`] 返回 `Some` 才会走到这里，而 stable 上它永远不返回 `Some`。
        pub(crate) fn forward_batch(
            _rows: &[&[f32]],
            _weights: &[u8],
            _scales: &[f32],
            never: Infallible,
            _out: &mut [f32],
        ) {
            match never {}
        }
    }
    pub use format::{
        Dense, Element, Format, Packed, PackedRow, Plain, PlainFmt, QuantI4, Quantized, SymI8,
        derived_matches_grain, scaled_operands,
    };
    pub use forward::{Geometry, forward_plain_batch};
    pub use shape::{cost, project_shape, reference, sample};

    pub(crate) use forward::ScaledRow;
    pub(crate) use shape::scale_of;

    #[cfg(test)]
    mod packed_test_support {
        include!("../tests/kernel/packed_test_support.rs");
    }

    /// 打包量化的分组长度：一行权重的元素个数是它的整数倍，标度按它分组记账。
    pub const GROUP: usize = 32;

    /// 生成一份「打包量化」GEMV 精度变体的完整公开面：`GROUP` 常量、`PackedRow` 标记、
    /// [`annotation::model`] 别名、`forward` 入口与测试挂载。`i2`/`i4` 只差一行 [`Row`]；
    /// 新增一档打包精度加一次调用即可。
    #[macro_export]
    macro_rules! gemv_packed {
        (
            $conf:ident, $alias:ident, $op:tt, $row:ident, $bits:expr,
            $w_bytes:expr, $sample_w:expr, $w_note:literal,
            $dequant_vals:expr, $dequant_expect:expr,
            $w_group:expr, $scales_group:expr, $ref_assert:expr,
            $w_sum:expr, $row0:expr, $row1:expr
        ) => {
            /// 量化分组（声明粒度）：唯一事实源在 [`GROUP`](crate::gemv::GROUP)。
            pub const GROUP: usize = $crate::gemv::GROUP;

            /// 打包配置标记：把一行 [`Row`](crate::quant::Row) 挂进 [`Packed`](crate::gemv::Packed)。
            pub struct $conf;

            impl $crate::gemv::PackedRow for $conf {
                const ROW: $crate::quant::Row = $row;

                fn cache() -> &'static $crate::einsum::FormulaCache {
                    $crate::formula_cell!()
                }
            }

            // 基座契约（ADR 0050）：粒度与三条行为全部从格式契约转发。
            $crate::operator_via!(
                grained $crate::gemv::Packed<$conf, { $crate::gemv::GROUP }>,
                $op,
                3,
                (4096, 4096),
                false,
                super,
                $crate::gemv::AXES
            );

            #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32")]
            #[doc = concat!("`", $op, "` 的语义类型：声明粒度 [`GROUP`]、每行按位宽打包。")]
            pub type $alias = $crate::gemv::Packed<$conf, GROUP>;

            /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
            pub fn forward(
                x: &[f32],
                weights: &[u8],
                scales: &[f32],
                out: &mut [f32],
            ) -> diagnostic::Result<()> {
                <$alias as $crate::gemv::Format>::forward(x, weights, scales, out)
            }

            /// 行批内核：`x` 是 `rows × m`、`out` 是 `rows × n`，一行权重只解包一次服务所有输入行。
            pub fn forward_batch(
                x: &[f32],
                weights: &[u8],
                scales: &[f32],
                geo: $crate::gemv::Geometry,
                out: &mut [f32],
            ) -> diagnostic::Result<()> {
                $crate::gemv::packed::forward_batch(x, weights, scales, geo, out, $row.cpb, $row.table)
            }

            #[cfg(test)]
            mod tests {
                use super::*;
                use $crate::gemv::packed_test_support::{at, multi_group, pack, with};

                $crate::gemv::packed_test_support::packed_gemv_tests!(
                    $alias, MODEL, GROUP, forward, $w_bytes, $sample_w
                );
                $crate::gemv::packed_test_support::packed_gemv_config_tests!(
                    $alias,
                    $op,
                    $bits,
                    $w_note,
                    $dequant_vals,
                    $dequant_expect
                );
                $crate::gemv::packed_test_support::packed_gemv_batch_tests!(
                    forward,
                    forward_batch,
                    $row.cpb,
                    GROUP
                );
                $crate::gemv::packed_test_support::packed_gemv_value_tests!(
                    $alias,
                    $bits,
                    $w_group,
                    $scales_group,
                    $ref_assert,
                    $w_sum,
                    $row0,
                    $row1
                );
            }
        };
    }

    pub mod f32 {
        //! `gemv_f32` 的算子语义：稠密 f32 权重投影。
        use crate::einsum::{Bits, FormulaCache, OperandSpec};
        use crate::gemv::{self, Plain, PlainFmt};

        /// 分块宽度：模板每轮处理这么多元素，标量参考按同一个宽度分块累加。
        pub const BLOCK: usize = 32;

        /// 稠密逐元素的配置：`f32` 权重、无标度、任意输入宽度，按 [`BLOCK`] 分块累加。
        impl<const B: usize> crate::formula::FormulaOwner for gemv::Dense<B> {
            /// `w[nm] · x[m] → y[n]`：无派生轴，宽度不参与分组。
            const EXPR: &'static str = "nm,m,n->n";

            /// 三个操作数都是 f32：`w` 是 `m·n` 个元素、`x` 是 `m` 个、`y` 是 `n` 个。
            const OPERANDS: &'static [OperandSpec] = &[
                OperandSpec::read("w", Bits::ThirtyTwo),
                OperandSpec::read("x", Bits::ThirtyTwo),
                OperandSpec::write("y", Bits::ThirtyTwo),
            ];

            const FRAME: &'static str = "nn::gemv::formula";

            fn cache() -> &'static FormulaCache {
                crate::formula_cell!()
            }
        }

        /// 稠密 f32 的样本与反量化（恒等）：骨架的一切算术都由 [`PlainFmt`] 那一份给出。
        impl<const B: usize> Plain for gemv::Dense<B> {
            type Elem = f32;

            /// 稠密算子的形状粒度是 1：一条只为对齐 nibble 的约束不该溢出到 f32 权重上。
            const GRAIN: usize = 1;

            const DERIVED: Option<(char, char, usize)> = None;

            const SCALES: Option<fn(usize) -> f32> = None;

            /// 模板每轮推进的元素数就是分块宽度（样本里的 `gsz` 与分块累加的宽度同源）。
            const STEP: usize = B;

            fn weights(m: usize, n: usize) -> Vec<f32> {
                (0..n * m)
                    .map(|idx| ((idx * 37) % 251) as f32 * 0.125 - 8.0)
                    .collect()
            }

            fn to_f32(elem: f32) -> f32 {
                elem
            }

            /// 稠密 `f32` 走 [`simd::elementwise`]：`WIDE` 条车道 × 独立累加链，整段乘加编译成向量指令。
            fn dot(w_row: &[f32], x: &[f32]) -> f32 {
                super::simd::elementwise(w_row, x)
            }

            /// 行并行：一份 `x` 段摊到 [`COLS`](super::simd::COLS) 行，解码阶段每 64 B 权重的取数
            /// 从单行版的 8 次降到 6 次，是 K9 从 40 GiB/s 抬到 66 GiB/s 的主因。
            fn dot_four(w: &[f32], x: &[f32], m: usize, out: &mut [f32]) {
                super::simd::elementwise_four(w, x, m, out)
            }
        }

        /// 稠密格式的语义类型：分块宽度 [`BLOCK`]、任意输入宽度都合法。
        #[annotation::model(name = "gemv_f32", shape = [4096, 4096], dtype = "f32")]
        pub type F32 = PlainFmt<gemv::Dense<BLOCK>>;

        // 基座契约（ADR 0050）：粒度与三条行为从格式契约转发。
        crate::operator_via!(grained F32, "gemv_f32", 2, (4096, 4096), true, super, gemv::AXES);

        /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
        pub fn forward(x: &[f32], weights: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
            <F32 as gemv::Format>::forward(x, weights, &[], out)
        }

        #[cfg(test)]
        mod tests {
            include!("../tests/kernel/gemv.rs");
        }
    }
    pub mod i2 {
        //! `gemv_i2` 的算子语义：2 位打包权重 + 每 [`GROUP`] 个元素一条标度。
        use crate::quant::I2;

        crate::gemv_packed!(
            I2Conf,
            I2,
            "gemv_i2",
            I2,
            2,
            1,
            16,
            "权重 4 MiB + 输入 16 KiB + 缩放 2 MiB",
            [1, -1, 0, 1],
            0.5,
            [1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1],
            [0.25, 0.5, 1.0, -1.0],
            |y: &[f32]| {
                assert!(y[0].is_finite());
                assert!(y[1].is_finite());
            },
            [1, -1, 0, 1, 1, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1],
            [1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0],
            [-2.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, 1.0]
        );
    }
    pub mod i4 {
        //! `gemv_i4` 的算子语义：4 位打包权重 + 每 [`GROUP`] 个元素一条标度。
        use crate::quant::I4;

        crate::gemv_packed!(
            I4Conf,
            I4,
            "gemv_i4",
            I4,
            4,
            2,
            32,
            "权重 8 MiB + 输入 16 KiB + 缩放 2 MiB",
            [1, 2, 3, 4],
            5.0,
            [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 2, 2, 2, 2],
            [0.25, 0.5, 1.0, 1.0],
            |y: &[f32]| {
                assert!(y[0] > 0.0);
                assert!(y[1] > y[0], "第二行权重更大");
            },
            [1, 2, 3, 4, 5, 6, 7, -8, -1, 0, 1, 2, 3, -4, 5, -6],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0],
            [-1.0, 0.0, 1.0, 2.0, 3.0, -4.0, 5.0, -6.0]
        );
    }
    pub mod i8 {
        //! `gemv_i8` 的算子语义：对称 8 位权重（不打包、零点 0）+ 每 [`GROUP`] 个元素一条标度。
        use crate::einsum::FormulaCache;
        use crate::gemv::{self, Plain, PlainFmt};
        use crate::quant::I8;

        /// 量化分组（声明粒度）：唯一事实源在 [`GROUP`](crate::gemv::GROUP)。
        pub const GROUP: usize = gemv::GROUP;

        /// 逐元素对称 int8 的配置：`u8` 承载位模式、按 `i8` 反量化、每 [`GROUP`] 个元素一条标度。
        impl<const G: usize> crate::formula::FormulaOwner for gemv::SymI8<G> {
            /// `w[nm] · x[m] · scales[ng] → y[n]`：`g` 是 `m` 每 [`GROUP`] 个元素切出的派生轴。
            const EXPR: &'static str = "nm,m,ng,n->n";

            // 权重是 8 位（每元素 1 字节），其余同打包格式：操作数表走同一条展开。
            scaled_operands!(I8.bits);

            const FRAME: &'static str = "nn::gemv::formula";

            fn cache() -> &'static FormulaCache {
                crate::formula_cell!()
            }
        }

        /// 逐元素对称 int8 的配置：样本取对称小整数，反量化按 `i8` 解释位模式。
        impl<const G: usize> Plain for gemv::SymI8<G> {
            type Elem = u8;

            scaled_grain!(G);

            fn weights(m: usize, n: usize) -> Vec<u8> {
                // 取对称小整数（-8..7）的 int8 位模式，避免大整数在 f32 网格外引入舍入。
                (0..n * m)
                    .map(|idx| ((idx * 37) % 16) as i8 - 8)
                    .map(|value| value as u8)
                    .collect()
            }

            fn to_f32(elem: u8) -> f32 {
                (elem as i8) as f32
            }

            /// 对称 `int8` 走 [`simd::elementwise`]：位模式原样重解释成 `i8` 再展宽，零点 0 无需偏移。
            fn dot(w_row: &[u8], x: &[f32]) -> f32 {
                super::simd::elementwise(w_row, x)
            }
        }

        /// 对称 int8 格式的语义类型：声明粒度 [`GROUP`]、每行 `m` 个字节。
        #[annotation::model(name = "gemv_i8", shape = [4096, 4096], dtype = "f32")]
        pub type I8 = PlainFmt<gemv::SymI8<GROUP>>;

        // 基座契约（ADR 0050）：粒度与三条行为从格式契约转发；量化算子不可微。
        crate::operator_via!(grained I8, "gemv_i8", 3, (4096, 4096), false, super, gemv::AXES);

        /// 内核本体：`m = x.len()`、`n = out.len()`，结果写进调用方给的缓冲。
        pub fn forward(
            x: &[f32],
            weights: &[u8],
            scales: &[f32],
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            <I8 as gemv::Format>::forward(x, weights, scales, out)
        }

        /// 行批内核：`x` 是 `rows × m`、`out` 是 `rows × n`，一行权重只读一遍。
        pub fn forward_batch(
            x: &[f32],
            weights: &[u8],
            scales: &[f32],
            geo: gemv::Geometry,
            out: &mut [f32],
        ) -> diagnostic::Result<()> {
            gemv::forward_plain_batch::<gemv::SymI8<GROUP>>(x, weights, scales, geo, out)
        }

        #[cfg(test)]
        mod tests {
            include!("../tests/kernel/gemv_i8.rs");
        }
    }

    /// 每个格式别名的契约自证都必须在**本 crate**里被求值一次。
    const _: () = <f32::F32 as Format>::CHECK;
    const _: () = <i2::I2 as Format>::CHECK;
    const _: () = <i4::I4 as Format>::CHECK;
    const _: () = <i8::I8 as Format>::CHECK;
}
pub mod matmul {
    //! 大 K 稠密 GEMM（可变收缩轴）：`Y[m,n] = Σ_k A[m,k] · B[k,n]`。
    //!
    //! 与 [`crate::gemm`] 的 `K=8` 量化骨架互补：本模块的收缩轴长度 `k` 由调用方给定，覆盖 LLM
    //! prefill 主链路的 GEMM（如 `k=4096`）。契约四件套同源：
    //! - 声明：`Y[m,n] = Σ_k A[m,k]·B[k,n]`，行主序，输出 `f32`；
    //! - 语义：浮点代价 `2·m·n·k` FLOP，读入 `(m·k + k·n)·4` 字节；
    //! - 标量参考：[`matmul_scalar`]，朴素 `i→j→p` 三重循环，每个输出元素独立按 `p` 顺序累加；
    //! - tile 实现：[`matmul_tiled`]，分块 rank-1 更新（GotoBLAS 风格的简化 CPU tile）。
    //!
    //! 两条路径累加顺序不同，`f32` 加法不结合，因此对拍按 IEEE 容差（[`close_enough`]，对齐
    //! lld §10.4 的 IEEE `rtol/atol` 口径），不做逐位相等。
    //!
    //! # 两种权重布局，两个核
    //!
    //! [`matmul_tiled`] 吃 `B [k, n]`（收缩轴在外），适合转置过的中间矩阵；线性层（`nn::MatMul` 与
    //! `serve` 的投影）的权重是 `w [n, k]`——每个输出通道一整行。那份布局下按 `[k, n]` 分块等于每次
    //! 都跨行取数，所以另有一个核 [`matmul_tn_tiled`]：输出列块 × 输出行块 × 收缩块三级分块，块内
    //! 是「一整行权重 × 连续行块」的点积，两段都连续。行批投影必须走这个核——逐行调 `gemv` 会把整份
    //! 权重按行数重复读一遍。
    //!
    //! # 一副骨架，两档精度
    //!
    //! 收缩轴可变的 GEMM 有两档：[`Precision for f32`] 稠密（累加器 `f32`，对拍按 IEEE 容差）与
    //! [`Precision for i8`] 对称 int8（累加器 `i32`，对拍逐位相等，见 [`matmul_i8`]）。两者的分块遍历、
    //! 尺寸校验、轴表与生产形状全同，只差「元素类型 + 累加器 + 一次乘加」，因此骨架只写这一份、
    //! 按元素类型单态化——[`Precision`] 是那份差异的唯一出处。

    use std::mem::size_of;
    use std::ops::Range;

    use diagnostic::{Code, Error, Result};

    use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};

    /// 输出 tile 行数。
    pub const TILE_M: usize = 64;

    /// 输出 tile 列数。
    pub const TILE_N: usize = 64;

    /// 收缩轴 tile 宽度。
    pub const TILE_K: usize = 32;

    /// 线性层布局（`x [rows, k]` × `w [n, k]`）的 tile 行数。
    pub const TN_ROWS: usize = 16;

    /// 线性层 tile 的输出列数：一列权重块要同时在累加器里驻留，不能大于 `TN_ROWS × TN_COLS` 的 L1 预算。
    pub const TN_COLS: usize = 32;

    /// 线性层 tile 的收缩块宽度。
    pub const TN_K: usize = 256;

    /// 收缩轴可变 GEMM 的精度契约：两档只差元素类型、累加器与一次乘加。
    pub trait Precision: Copy {
        /// 累加器与输出元素：`f32` 档就是 [`f32`]，`i8` 档是 [`i32`]（转回 f32 由注册侧做）。
        type Acc: Copy + Default;

        /// 注册名：尺寸校验与代价标签都取它，入口不必再传一遍名字。
        const NAME: &'static str;

        /// 一次乘加：`*acc += a·b`。
        fn mac(acc: &mut Self::Acc, a: Self, b: Self);
    }

    impl Precision for f32 {
        type Acc = f32;
        const NAME: &'static str = "matmul_f32";

        fn mac(acc: &mut f32, a: f32, b: f32) {
            *acc += a * b;
        }
    }

    impl Precision for i8 {
        type Acc = i32;
        const NAME: &'static str = "matmul_i8";

        fn mac(acc: &mut i32, a: i8, b: i8) {
            *acc += i32::from(a) * i32::from(b);
        }
    }

    /// 标量参考：`out[i,j] = Σ_p a[i,p]·b[p,j]`，每个元素按 `p` 顺序独立累加；两档精度共用。
    pub fn matmul_scalar<P: Precision>(
        a: &[P],
        b: &[P],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [P::Acc],
    ) {
        for i in 0..m {
            let arow = &a[i * k..][..k];
            let orow = &mut out[i * n..][..n];
            for j in 0..n {
                let mut acc = P::Acc::default();
                for (p, &x) in arow.iter().enumerate() {
                    P::mac(&mut acc, x, b[p * n + j]);
                }
                orow[j] = acc;
            }
        }
    }

    /// 分块 tile：按 [`TILE_M`]×[`TILE_N`] 切输出、[`TILE_K`] 切收缩轴，块内 rank-1 更新。
    pub fn matmul_tiled<P: Precision>(
        a: &[P],
        b: &[P],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [P::Acc],
    ) {
        out.fill(P::Acc::default());
        let mut bi = 0;
        while bi < m {
            let i_end = (bi + TILE_M).min(m);
            let mut bk = 0;
            while bk < k {
                let k_end = (bk + TILE_K).min(k);
                let mut bj = 0;
                while bj < n {
                    let j_end = (bj + TILE_N).min(n);
                    let tile = Tile {
                        rows: bi..i_end,
                        contraction: bk..k_end,
                        cols: bj..j_end,
                    };
                    accumulate_tile(a, b, out, &tile, n, k);
                    bj += TILE_N;
                }
                bk += TILE_K;
            }
            bi += TILE_M;
        }
    }

    /// 一个 tile 的三条区间：输出行、收缩轴、输出列。
    struct Tile {
        rows: Range<usize>,
        contraction: Range<usize>,
        cols: Range<usize>,
    }

    /// 块内 rank-1 更新：`out[i, j] += Σ_p a[i, p]·b[p, j]`，三条下标都取 `tile` 给的区间。
    fn accumulate_tile<P: Precision>(
        a: &[P],
        b: &[P],
        out: &mut [P::Acc],
        tile: &Tile,
        n: usize,
        k: usize,
    ) {
        for i in tile.rows.clone() {
            let start = i * k + tile.contraction.start;
            let arow = &a[start..i * k + tile.contraction.end];
            for (p, &x) in arow.iter().enumerate() {
                let base = (tile.contraction.start + p) * n;
                let brow = &b[base + tile.cols.start..base + tile.cols.end];
                let orow = &mut out[i * n + tile.cols.start..i * n + tile.cols.end];
                for (o, &y) in orow.iter_mut().zip(brow) {
                    P::mac(o, x, y);
                }
            }
        }
    }

    /// 三条缓冲的尺寸校验，两档精度的四个 `forward` 共用；算子名取自 [`Precision::NAME`]。
    fn check_dims<P: Precision>(
        a: &[P],
        b: &[P],
        m: usize,
        n: usize,
        k: usize,
        out: &[P::Acc],
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || out.len() != m * n {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "{} 缓冲尺寸不符：a={}，b={}，out={}，期望 m·k={}，k·n={}，m·n={}",
                    P::NAME,
                    a.len(),
                    b.len(),
                    out.len(),
                    m * k,
                    k * n,
                    m * n
                ),
            ));
        }
        Ok(())
    }

    /// 标量参考入口：校验尺寸后写 `out`。
    ///
    /// # Errors
    ///
    /// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
    pub fn forward_scalar<P: Precision>(
        a: &[P],
        b: &[P],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [P::Acc],
    ) -> Result<()> {
        check_dims(a, b, m, n, k, out)?;
        matmul_scalar(a, b, m, n, k, out);
        Ok(())
    }

    /// tile 实现入口：校验尺寸后写 `out`。
    ///
    /// # Errors
    ///
    /// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
    pub fn forward_tiled<P: Precision>(
        a: &[P],
        b: &[P],
        m: usize,
        n: usize,
        k: usize,
        out: &mut [P::Acc],
    ) -> Result<()> {
        check_dims(a, b, m, n, k, out)?;
        matmul_tiled(a, b, m, n, k, out);
        Ok(())
    }

    /// 线性层布局（TN）的标量参考：`out[r, j] = Σ_p x[r, p]·w[j, p]`，每个元素按 `p` 顺序独立累加。
    pub fn matmul_tn_scalar(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
        for r in 0..rows {
            let xrow = &x[r * k..][..k];
            let orow = &mut out[r * n..][..n];
            for (j, slot) in orow.iter_mut().enumerate() {
                let wrow = &w[j * k..][..k];
                let mut acc = 0.0f32;
                for (&a, &b) in xrow.iter().zip(wrow) {
                    acc += a * b;
                }
                *slot = acc;
            }
        }
    }

    /// 线性层布局的分块 tile：[`TN_ROWS`] 行 × [`TN_COLS`] 列的输出块，收缩轴按 [`TN_K`] 切。
    pub fn matmul_tn_tiled(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &mut [f32]) {
        let mut block = [0.0f32; TN_ROWS * TN_COLS];
        let mut j0 = 0;
        while j0 < n {
            let j1 = (j0 + TN_COLS).min(n);
            let mut r0 = 0;
            while r0 < rows {
                let r1 = (r0 + TN_ROWS).min(rows);
                let acc = &mut block[..(r1 - r0) * TN_COLS];
                acc.fill(0.0);
                let mut p0 = 0;
                while p0 < k {
                    let p1 = (p0 + TN_K).min(k);
                    let tile = TnTile {
                        rows: r0..r1,
                        contraction: p0..p1,
                        cols: j0..j1,
                    };
                    accumulate_tn_tile(x, w, acc, &tile, k);
                    p0 = p1;
                }
                for row in r0..r1 {
                    let start = (row - r0) * TN_COLS;
                    let len = j1 - j0;
                    out[row * n + j0..row * n + j1].copy_from_slice(&acc[start..start + len]);
                }
                r0 = r1;
            }
            j0 = j1;
        }
    }

    /// 线性层布局的一个 tile：输出行、收缩轴、输出列三条区间（与 [`Tile`] 同构，含义不同）。
    struct TnTile {
        rows: Range<usize>,
        contraction: Range<usize>,
        cols: Range<usize>,
    }

    /// 块内累加：`acc[row, j] += Σ_p x[row, p]·w[j, p]`，`acc` 的行宽是 [`TN_COLS`]。
    fn accumulate_tn_tile(x: &[f32], w: &[f32], acc: &mut [f32], tile: &TnTile, k: usize) {
        for j in tile.cols.clone() {
            let wchunk = &w[j * k + tile.contraction.start..j * k + tile.contraction.end];
            let slot = j - tile.cols.start;
            for row in tile.rows.clone() {
                let xchunk = &x[row * k + tile.contraction.start..row * k + tile.contraction.end];
                acc[(row - tile.rows.start) * TN_COLS + slot] += dot_lanes(xchunk, wchunk);
            }
        }
    }

    /// 累加链数：8 条独立的部分和，乘加不排成一条依赖链（编译后是一整条 SIMD 乘加指令）。
    const LANES: usize = 8;

    /// 多点积：`LANES` 条独立累加链。
    pub fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let (x_chunks, x_tail) = a[..len].as_chunks::<LANES>();
        let (y_chunks, y_tail) = b[..len].as_chunks::<LANES>();
        let mut lanes = [0.0f32; LANES];
        for (x, y) in x_chunks.iter().zip(y_chunks) {
            let mut lane = 0;
            while lane < LANES {
                lanes[lane] += x[lane] * y[lane];
                lane += 1;
            }
        }
        let mut sum = lanes.iter().sum::<f32>();
        for (x, y) in x_tail.iter().zip(y_tail) {
            sum += x * y;
        }
        sum
    }

    /// 线性层布局的尺寸校验：输入 `rows × k`、权重 `n × k`、输出 `rows × n`。
    fn check_tn_dims(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize, out: &[f32]) -> Result<()> {
        if x.len() != rows * k || w.len() != n * k || out.len() != rows * n {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "线性层 GEMM 缓冲尺寸不符：x={}，w={}，out={}，期望 rows·k={}，n·k={}，rows·n={}",
                    x.len(),
                    w.len(),
                    out.len(),
                    rows * k,
                    n * k,
                    rows * n
                ),
            ));
        }
        Ok(())
    }

    /// 线性层布局的标量入口：校验尺寸后写 `out`。
    ///
    /// # Errors
    ///
    /// 三条缓冲长度与 `(rows, n, k)` 不符时返回说明。
    pub fn forward_tn_scalar(
        x: &[f32],
        w: &[f32],
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) -> Result<()> {
        check_tn_dims(x, w, rows, n, k, out)?;
        matmul_tn_scalar(x, w, rows, n, k, out);
        Ok(())
    }

    /// 线性层布局的 tile 入口：校验尺寸后写 `out`。这是服务侧行批投影的唯一生产路径。
    ///
    /// # Errors
    ///
    /// 三条缓冲长度与 `(rows, n, k)` 不符时返回说明。
    pub fn forward_tn_tiled(
        x: &[f32],
        w: &[f32],
        rows: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
    ) -> Result<()> {
        check_tn_dims(x, w, rows, n, k, out)?;
        matmul_tn_tiled(x, w, rows, n, k, out);
        Ok(())
    }

    /// IEEE 容差对拍：`|a - e| ≤ atol + rtol·|e|`。`rtol/atol` 与 lld §10.4 的 IEEE 口径同源。
    pub fn close_enough(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) -> Result<()> {
        if actual.len() != expected.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "对拍长度不符：实际 {}，参考 {}",
                    actual.len(),
                    expected.len()
                ),
            ));
        }
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let tol = atol + rtol * e.abs();
            if (a - e).abs() > tol {
                return Err(Error::new(
                    Code::Invalid,
                    format!(
                        "第 {i} 个元素 {a} 与参考 {e} 差 {}，超过容差 {tol}",
                        (a - e).abs()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// 收缩轴长度（生产档）：prefill 主链路 GEMM 的 K。
    pub const K: usize = 4096;

    /// 三轴：`m`/`k` 输入，`n` 输出（角色与 [`crate::gemm::AXES`] 一致，收缩轴长度由 [`K`] 给定）。
    pub const AXES: &[AxisSpec] = &[
        AxisSpec {
            role: AxisRole::Input,
            label: 'm',
        },
        AxisSpec {
            role: AxisRole::Input,
            label: 'k',
        },
        AxisSpec {
            role: AxisRole::Output,
            label: 'n',
        },
    ];

    /// 形状投影：输出 `m×n`，收缩轴 `k = K`（生产档）。
    ///
    /// # Errors
    ///
    /// 轴表不合法时由 [`ShapeCtx::new`] 报出。
    pub fn project_shape(m: usize, n: usize) -> Result<ShapeCtx> {
        ShapeCtx::new([('m', m), ('k', K), ('n', n)])
    }

    /// 标量代价：`2·m·K·n` FLOP，读 `(m·K + K·n)·size_of::<P>()` 字节，写 `m·n·4` 字节。
    ///
    /// 读数由精度定（`f32` 每元素 4 字节、`i8` 1 字节），写出恒为 f32；轴名与算子名取自
    /// [`Precision::NAME`]。
    ///
    /// # Errors
    ///
    /// 形状上下文缺轴时返回带算子名的说明。
    fn cost_of<P: Precision>(shape: &ShapeCtx) -> Result<Cost> {
        let axis = |label: char| {
            shape
                .get(label)
                .ok_or_else(|| Error::new(Code::Invalid, format!("{} 形状缺轴 `{label}`", P::NAME)))
        };
        let m = axis('m')?;
        let k = axis('k')?;
        let n = axis('n')?;
        Ok(Cost {
            flops: (2 * m * k * n) as f64,
            read: ((m * k + k * n) * size_of::<P>()) as u64,
            write: (m * n * 4) as u64,
        })
    }

    /// 稠密档的代价：输入每元素 4 字节，口径见 `cost_of`。
    ///
    /// # Errors
    ///
    /// 形状上下文缺轴时返回带算子名的说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost> {
        cost_of::<f32>(shape)
    }

    /// 三缓冲样本的公共骨架：`a` / `b` 的取值由精度给，`y` 恒为 f32 零缓冲，标量 `M`/`K`/`N`
    /// 与 `.k` 签名一致。
    fn sample_of(m: usize, n: usize, k: usize, a: Values, b: Values) -> Sample {
        Sample {
            m,
            n,
            scalars: vec![("M", m as i64), ("K", k as i64), ("N", n as i64)],
            bufs: vec![
                Buf {
                    name: "a",
                    values: a,
                },
                Buf {
                    name: "b",
                    values: b,
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; m * n]),
                },
            ],
        }
    }

    /// 确定性样本：`a` / `b` 由同一份采样序列给出，`y` 为零缓冲，收缩轴取生产形状 [`K`]。
    pub fn sample(m: usize, n: usize) -> Sample {
        sample_k(m, n, K)
    }

    /// 显式收缩轴长度的确定性样本：与 [`sample`] 同构，只有 `K` 由调用方给定。
    ///
    /// 形状前提（尾块、不整除）只有把 `K` 拿在手里才验得了——`K` 不是 `(m, n)` 的函数，
    /// [`sample`] 把它写死成生产形状，`3×5×7` 这种形状就无从表达。
    pub fn sample_k(m: usize, n: usize, k: usize) -> Sample {
        sample_of(
            m,
            n,
            k,
            Values::F32(deterministic(m * k, 1)),
            Values::F32(deterministic(k * n, 2)),
        )
    }

    /// 标量参考：读 `a` / `b`，按 [`matmul_scalar`] 计算输出。
    ///
    /// # Errors
    ///
    /// 样本缺 `a` / `b` 时返回说明。
    pub fn reference(sample: &Sample) -> Result<Vec<f32>> {
        let a = sample.f32("a")?;
        let b = sample.f32("b")?;
        let mut out = vec![0.0f32; sample.m * sample.n];
        matmul_scalar(a, b, sample.m, sample.n, K, &mut out);
        Ok(out)
    }

    /// 稠密矩阵乘标记类型：只用来承载语义表，没有实例。
    pub struct Matmul;

    /// 语义表由注解生成：生产形状 `256×256`，收缩轴 [`K`]，输出 f32。
    #[annotation::model(name = "matmul_f32", shape = [256, 256], dtype = "f32", skeleton = "matmul")]
    pub type MatmulF32 = Matmul;

    crate::operator_here!(Matmul, "matmul_f32", 2, (256, 256), false, reference);

    /// 确定性采样（与测试同源）：`(len, salt)` 决定序列，不引入外部依赖。
    fn deterministic(len: usize, salt: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(salt.wrapping_mul(40503));
                ((x >> 8) as f32) / 256.0 - 32.0
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel/matmul_tests.rs");
    }

    /// 大 K 对称 int8 GEMM 的语义与精度档：`Y[m, n] = Σ_k A[m, k] · B[k, n]`，累加器 `i32`。
    pub mod i8 {
        //! 职责：i8 矩阵乘的标量/分块出口与形状投影。
        use diagnostic::Result;

        use crate::model::{Cost, Sample, ShapeCtx, Values};

        pub use super::{
            AXES, K, TILE_K, TILE_M, TILE_N, forward_scalar, matmul_scalar, matmul_tiled, project_shape,
        };

        /// 输出块行数：4 行共用每一条 `B` 行块的读取。
        const BLOCK_ROWS: usize = 4;

        /// 输出块列数：一行的累加器是 16 个 `i32`，4 行共 8 个 NEON 向量寄存器。
        const BLOCK_COLS: usize = 16;

        /// 分块入口：校验尺寸后走 4×16 寄存器分块。
        ///
        /// 整数累加与遍历次序无关，因此结果与 [`matmul_scalar`] 逐位一致；收益来自 `B` 行块
        /// 一次读取被 4 行复用（原 rank-1 更新每行都要重读一遍 `B`）。
        ///
        /// # Errors
        ///
        /// 三条缓冲长度与 `(m, n, k)` 不符时返回说明。
        pub fn forward_tiled(
            a: &[i8],
            b: &[i8],
            m: usize,
            n: usize,
            k: usize,
            out: &mut [i32],
        ) -> Result<()> {
            super::check_dims(a, b, m, n, k, out)?;
            blocked(a, b, m, n, k, out);
            Ok(())
        }

        /// 4×16 块 + 行尾/列尾：块内累加器留在寄存器，尾块退回标量但结果不变。
        fn blocked(a: &[i8], b: &[i8], m: usize, n: usize, k: usize, out: &mut [i32]) {
            let mut i0 = 0;
            while i0 + BLOCK_ROWS <= m {
                let mut j0 = 0;
                while j0 + BLOCK_COLS <= n {
                    block(a, b, out, (i0, j0), n, k);
                    j0 += BLOCK_COLS;
                }
                edge_cols(a, b, out, i0, j0, n, k);
                i0 += BLOCK_ROWS;
            }
            edge_rows(a, b, out, i0, m, n, k);
        }

        /// 一个 `4×16` 输出块：`B` 的每个 p 行块只读一次、只拓宽一次，再广播给 4 行。
        fn block(a: &[i8], b: &[i8], out: &mut [i32], at: (usize, usize), n: usize, k: usize) {
            let (i0, j0) = at;
            let mut acc = [[0i32; BLOCK_COLS]; BLOCK_ROWS];
            for p in 0..k {
                let wide = widen(&b[p * n + j0..][..BLOCK_COLS]);
                for r in 0..BLOCK_ROWS {
                    let x = i32::from(a[(i0 + r) * k + p]);
                    for j in 0..BLOCK_COLS {
                        acc[r][j] += x * wide[j];
                    }
                }
            }
            for r in 0..BLOCK_ROWS {
                out[(i0 + r) * n + j0..][..BLOCK_COLS].copy_from_slice(&acc[r]);
            }
        }

        /// `B` 的一行块拓宽一次：这里就是「只拓宽一次」的唯一位置，逐行重做会让向量化只剩一半。
        fn widen(row: &[i8]) -> [i32; BLOCK_COLS] {
            let mut wide = [0i32; BLOCK_COLS];
            for (slot, &value) in wide.iter_mut().zip(row) {
                *slot = i32::from(value);
            }
            wide
        }

        /// 列尾（`j0..n`）：块内按行标量累加，结果与分块逐位一致。
        fn edge_cols(
            a: &[i8],
            b: &[i8],
            out: &mut [i32],
            i0: usize,
            mut j0: usize,
            n: usize,
            k: usize,
        ) {
            while j0 < n {
                for r in 0..BLOCK_ROWS {
                    let mut acc = 0i32;
                    for p in 0..k {
                        acc += i32::from(a[(i0 + r) * k + p]) * i32::from(b[p * n + j0]);
                    }
                    out[(i0 + r) * n + j0] = acc;
                }
                j0 += 1;
            }
        }

        /// 行尾（`i0..m`）：不足 [`BLOCK_ROWS`] 的行逐元素累加。
        fn edge_rows(a: &[i8], b: &[i8], out: &mut [i32], mut i0: usize, m: usize, n: usize, k: usize) {
            while i0 < m {
                for j in 0..n {
                    let mut acc = 0i32;
                    for p in 0..k {
                        acc += i32::from(a[i0 * k + p]) * i32::from(b[p * n + j]);
                    }
                    out[i0 * n + j] = acc;
                }
                i0 += 1;
            }
        }

        /// 标量代价：`2·m·K·n` FLOP；输入为 i8（每元素 1 字节），输出 f32（每元素 4 字节）。
        ///
        /// # Errors
        ///
        /// 形状上下文缺轴时返回带算子名的说明。
        pub fn cost(shape: &ShapeCtx) -> Result<Cost> {
            super::cost_of::<i8>(shape)
        }

        /// 确定性样本：`a` / `b` 为对称 i8（u8 字节承载有符号码），`y` 为 f32 零缓冲。
        pub fn sample(m: usize, n: usize) -> Sample {
            sample_k(m, n, K)
        }

        /// 显式收缩轴长度的确定性样本：与 [`sample`] 同构，只有 `K` 由调用方给定。
        pub fn sample_k(m: usize, n: usize, k: usize) -> Sample {
            super::sample_of(
                m,
                n,
                k,
                Values::U8(deterministic_i8(m * k, 1)),
                Values::U8(deterministic_i8(k * n, 2)),
            )
        }

        /// 标量参考：读 `a` / `b`（u8 承载的有符号码），i32 精确累加后转 f32 输出。
        ///
        /// # Errors
        ///
        /// 样本缺 `a` / `b` 时返回说明。
        pub fn reference(sample: &Sample) -> Result<Vec<f32>> {
            let a: Vec<i8> = sample.u8("a")?.iter().map(|&x| x as i8).collect();
            let b: Vec<i8> = sample.u8("b")?.iter().map(|&x| x as i8).collect();
            let mut acc = vec![0i32; sample.m * sample.n];
            matmul_scalar(&a, &b, sample.m, sample.n, K, &mut acc);
            Ok(acc.into_iter().map(|x| x as f32).collect())
        }

        /// 对称 int8 矩阵乘标记类型：只用来承载语义表，没有实例。
        pub struct MatmulI8Op;

        /// 语义表由注解生成：生产形状 `256×256`，收缩轴 [`K`]，`i32` 精确累加后输出 f32。
        #[annotation::model(name = "matmul_i8", shape = [256, 256], dtype = "f32", skeleton = "matmul")]
        pub type MatmulI8 = MatmulI8Op;

        crate::operator_here!(MatmulI8Op, "matmul_i8", 2, (256, 256), false, reference);

        /// 确定性 i8 采样：覆盖正负与零，`(len, salt)` 固定即序列固定。
        fn deterministic_i8(len: usize, salt: u32) -> Vec<u8> {
            (0..len)
                .map(|i| {
                    let x = (i as u32)
                        .wrapping_mul(2654435761)
                        .wrapping_add(salt.wrapping_mul(40503));
                    ((x % 251) as i32 - 125) as i8 as u8
                })
                .collect()
        }

        #[cfg(test)]
        mod tests {
            include!("../tests/kernel/matmul_i8_tests.rs");
        }
    }
}
pub mod norm {
    //! 归一化层：`BatchNorm2d`、`GroupNorm`、`InstanceNorm`、`LayerNorm`（NHWC，精度由 `S: Scalar` 决定）。

    use crate::Scalar;
    use crate::guard::check_all;
    use diagnostic::{Code, Error, Result};

    /// 二维批归一化：通道数、数值稳定项与滑动统计的动量。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct BatchNorm2d<S: Scalar> {
        /// 通道数。
        pub channels: usize,
        /// `sqrt(var + eps)` 的稳定项。
        pub eps: S,
        /// 滑动统计的动量：`new = (1−m)·old + m·batch`。
        pub momentum: S,
    }

    /// 组归一化：把通道分成 `groups` 组，每组在自己的 `(C/G,H,W)` 内统计。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct GroupNorm<S: Scalar> {
        /// 组数（必须整除通道数）。
        pub groups: usize,
        /// 通道数。
        pub channels: usize,
        /// 数值稳定项。
        pub eps: S,
    }

    /// 实例归一化：`GroupNorm` 在 `groups == channels` 时的特例，单独列出是为了让调用方不必
    /// 记住"传 C 个组"这件事。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct InstanceNorm<S: Scalar> {
        /// 通道数。
        pub channels: usize,
        /// 数值稳定项。
        pub eps: S,
    }

    /// 层归一化：在最后一维（宽度 `width`）上统计，逐行独立。
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct LayerNorm<S: Scalar> {
        /// 归一化宽度（最后一维）。
        pub width: usize,
        /// 数值稳定项。
        pub eps: S,
    }

    impl<S: Scalar> BatchNorm2d<S> {
        /// 训练期：用批内统计归一化，并把批统计按 `momentum` 混入滑动统计。
        ///
        /// # Errors
        ///
        /// 形状不符或 `n·h·w == 0` 时返回说明。
        #[allow(clippy::too_many_arguments)]
        pub fn forward_train(
            &self,
            x: &[S],
            gamma: &[S],
            beta: &[S],
            n: usize,
            h: usize,
            w: usize,
            running_mean: &mut [S],
            running_var: &mut [S],
            out: &mut [S],
        ) -> Result<()> {
            let (c, count) = self.geometry(x, gamma, beta, n, h, w, out)?;
            let (mean, var) = batch_stats(x, c, count);
            for ci in 0..c {
                running_mean[ci] =
                    (S::ONE - self.momentum) * running_mean[ci] + self.momentum * mean[ci];
                // 方差用无偏口径进滑动统计（与 PyTorch 一致），归一化本身用有偏方差。
                let unbiased = if count > 1 {
                    var[ci] * S::from_usize(count) / S::from_usize(count - 1)
                } else {
                    var[ci]
                };
                running_var[ci] = (S::ONE - self.momentum) * running_var[ci] + self.momentum * unbiased;
            }
            affine(x, gamma, beta, &mean, &var, self.eps, c, count, out);
            Ok(())
        }

        /// 推理期：只用滑动统计，不改状态。
        ///
        /// # Errors
        ///
        /// 形状不符时返回说明。
        #[allow(clippy::too_many_arguments)]
        pub fn forward_eval(
            &self,
            x: &[S],
            gamma: &[S],
            beta: &[S],
            n: usize,
            h: usize,
            w: usize,
            running_mean: &[S],
            running_var: &[S],
            out: &mut [S],
        ) -> Result<()> {
            let (c, count) = self.geometry(x, gamma, beta, n, h, w, out)?;
            let mean = running_mean.to_vec();
            let var = running_var.to_vec();
            affine(x, gamma, beta, &mean, &var, self.eps, c, count, out);
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn geometry(
            &self,
            x: &[S],
            gamma: &[S],
            beta: &[S],
            n: usize,
            h: usize,
            w: usize,
            out: &[S],
        ) -> Result<(usize, usize)> {
            let c = self.channels;
            let count = n * h * w;
            check_all(&[
                (x.len(), count * c, "x"),
                (gamma.len(), c, "gamma"),
                (beta.len(), c, "beta"),
                (out.len(), count * c, "out"),
            ])?;
            if count == 0 {
                return Err(Error::new(Code::Invalid, "BatchNorm 不接受 0 个空间位置"));
            }
            Ok((c, count))
        }
    }

    impl<S: Scalar> GroupNorm<S> {
        /// 前向：按组统计后归一化。
        ///
        /// # Errors
        ///
        /// `groups` 不整除通道数或形状不符时返回说明。
        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            &self,
            x: &[S],
            gamma: &[S],
            beta: &[S],
            n: usize,
            spatial: usize,
            out: &mut [S],
        ) -> Result<()> {
            let c = self.channels;
            if self.groups == 0 || !c.is_multiple_of(self.groups) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("GroupNorm: {c} 个通道不能分成 {} 组", self.groups),
                ));
            }
            check_all(&[
                (x.len(), n * spatial * c, "x"),
                (gamma.len(), c, "gamma"),
                (beta.len(), c, "beta"),
                (out.len(), n * spatial * c, "out"),
            ])?;
            normalize_groups(
                x,
                gamma,
                beta,
                out,
                n,
                spatial,
                c,
                c / self.groups,
                self.groups,
                self.eps,
            );
            Ok(())
        }
    }

    impl<S: Scalar> InstanceNorm<S> {
        /// 前向：等价于 `GroupNorm{groups: channels}`。
        ///
        /// # Errors
        ///
        /// 形状不符时返回说明。
        pub fn forward(
            &self,
            x: &[S],
            gamma: &[S],
            beta: &[S],
            n: usize,
            spatial: usize,
            out: &mut [S],
        ) -> Result<()> {
            let gn = GroupNorm {
                groups: self.channels,
                channels: self.channels,
                eps: self.eps,
            };
            gn.forward(x, gamma, beta, n, spatial, out)
        }
    }

    impl<S: Scalar> LayerNorm<S> {
        /// 前向：逐行在最后一维上减均值除标准差。
        ///
        /// # Errors
        ///
        /// 缓冲长度不是 `width` 整数倍或仿射参数长度不符时返回说明。
        pub fn forward(&self, x: &[S], gamma: &[S], beta: &[S], out: &mut [S]) -> Result<()> {
            let width = self.width;
            if width == 0 || !x.len().is_multiple_of(width) {
                return Err(Error::new(
                    Code::Invalid,
                    format!("LayerNorm: {} 不是宽度 {width} 的整数倍", x.len()),
                ));
            }
            check_all(&[
                (gamma.len(), width, "gamma"),
                (beta.len(), width, "beta"),
                (out.len(), x.len(), "out"),
            ])?;
            for row in 0..x.len() / width {
                normalize_row(x, gamma, beta, out, width, row, self.eps);
            }
            Ok(())
        }
    }

    /// 批内按通道的均值与（有偏）方差。
    fn batch_stats<S: Scalar>(x: &[S], c: usize, count: usize) -> (Vec<S>, Vec<S>) {
        let mut mean = vec![S::ZERO; c];
        let mut sq = vec![S::ZERO; c];
        for p in 0..count {
            for ci in 0..c {
                let v = x[p * c + ci];
                mean[ci] = mean[ci] + v;
                sq[ci] = sq[ci] + v * v;
            }
        }
        let count_s = S::from_usize(count);
        for ci in 0..c {
            mean[ci] = mean[ci] / count_s;
            let m2 = sq[ci] / count_s - mean[ci] * mean[ci];
            sq[ci] = m2.max(S::ZERO);
        }
        (mean, sq)
    }

    /// 统一仿射：`(x−mean)/√(var+eps)·γ+β`，`mean`/`var` 按通道索引。
    #[allow(clippy::too_many_arguments)]
    fn affine<S: Scalar>(
        x: &[S],
        gamma: &[S],
        beta: &[S],
        mean: &[S],
        var: &[S],
        eps: S,
        c: usize,
        count: usize,
        out: &mut [S],
    ) {
        let inv: Vec<S> = var.iter().map(|v| S::ONE / (*v + eps).sqrt()).collect();
        for p in 0..count {
            for ci in 0..c {
                let idx = p * c + ci;
                out[idx] = (x[idx] - mean[ci]) * inv[ci] * gamma[ci] + beta[ci];
            }
        }
    }

    /// 按组遍历并逐组归一化（`GroupNorm` 的骨架）。
    #[allow(clippy::too_many_arguments)]
    fn normalize_groups<S: Scalar>(
        x: &[S],
        gamma: &[S],
        beta: &[S],
        out: &mut [S],
        n: usize,
        spatial: usize,
        c: usize,
        per: usize,
        groups: usize,
        eps: S,
    ) {
        for bi in 0..n {
            for g in 0..groups {
                normalize_group(x, gamma, beta, out, spatial, c, per, bi, g * per, eps);
            }
        }
    }

    /// 单组 `(C/G,H,W)` 内的减均值、除标准差、仿射。
    #[allow(clippy::too_many_arguments)]
    fn normalize_group<S: Scalar>(
        x: &[S],
        gamma: &[S],
        beta: &[S],
        out: &mut [S],
        spatial: usize,
        c: usize,
        per: usize,
        bi: usize,
        c0: usize,
        eps: S,
    ) {
        let mut sum = S::ZERO;
        let mut sq = S::ZERO;
        let len = S::from_usize(per * spatial);
        for s in 0..spatial {
            for ci in c0..c0 + per {
                let v = x[(bi * spatial + s) * c + ci];
                sum = sum + v;
                sq = sq + v * v;
            }
        }
        let mean = sum / len;
        let var = (sq / len - mean * mean).max(S::ZERO);
        let inv = S::ONE / (var + eps).sqrt();
        for s in 0..spatial {
            for ci in c0..c0 + per {
                let idx = (bi * spatial + s) * c + ci;
                out[idx] = (x[idx] - mean) * inv * gamma[ci] + beta[ci];
            }
        }
    }

    /// 单行 LayerNorm；方差用 f64 累加以保 f32 在宽行上的精度。
    fn normalize_row<S: Scalar>(
        x: &[S],
        gamma: &[S],
        beta: &[S],
        out: &mut [S],
        width: usize,
        row: usize,
        eps: S,
    ) {
        let src = &x[row * width..(row + 1) * width];
        let dst = &mut out[row * width..(row + 1) * width];
        let sum: f64 = src.iter().map(|v| v.to_f64()).sum();
        let mean = S::from_f64(sum / width as f64);
        let sq: f64 = src
            .iter()
            .map(|v| {
                let d = v.to_f64() - mean.to_f64();
                d * d
            })
            .sum();
        let inv = S::ONE / (S::from_f64(sq / width as f64) + eps).sqrt();
        for i in 0..width {
            dst[i] = (src[i] - mean) * inv * gamma[i] + beta[i];
        }
    }

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel/norm_tests.rs");
    }
}
pub mod reduce {
    //! 归约骨架：`y[0] = fold(a[0..n])`，输出单个元素。

    use crate::formula::FormulaOwner;
    use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
    use diagnostic::{Code, Error};

    /// 归约算子随语义变化的部分。
    pub trait Reduce: FormulaOwner {
        /// 累加初值。
        fn start() -> f32;

        /// 一次折叠：把当前累加值与下一个元素合并。
        fn fold(acc: f32, x: f32) -> f32;

        /// 确定性输入：默认线性夹具 `−1.0 + 0.5·i`；需要不同取值时在实现里覆盖。
        fn input(len: usize) -> Vec<f32> {
            crate::model::linear(len, 0.5, -1.0)
        }
    }

    /// 两条轴：输入轴 `n`（被规约），输出轴 `o`。
    pub const AXES: &[AxisSpec] = &[
        AxisSpec {
            role: AxisRole::Input,
            label: 'n',
        },
        AxisSpec {
            role: AxisRole::Output,
            label: 'o',
        },
    ];

    /// 形状：输入轴 `n`=len，输出轴 `o` 恒为 1。
    ///
    /// # Errors
    ///
    /// 当前实现不会失败。
    pub fn project_shape(len: usize, _: usize) -> Result<ShapeCtx, Error> {
        ShapeCtx::new([('n', len), ('o', 1)])
    }

    /// 标量代价。
    pub fn cost<R: Reduce>(shape: &ShapeCtx) -> Result<Cost, Error> {
        R::formula()?
            .cost(shape.extents())
            .map_err(|error| error.with("nn::reduce::cost"))
    }

    /// 确定性样本：输入 len 个，输出 1 个。
    pub fn sample<R: Reduce>(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: 1,
            scalars: vec![("N", len as i64)],
            bufs: vec![
                Buf {
                    name: "a",
                    values: Values::F32(R::input(len)),
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0]),
                },
            ],
        }
    }

    /// 标量参考：折叠整个输入。
    pub fn reference<R: Reduce>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let a = sample.f32("a")?;
        let mut acc = R::start();
        for x in a {
            acc = R::fold(acc, *x);
        }
        Ok(vec![acc])
    }

    /// 内核本体：结果写进调用方给的单元素缓冲。
    ///
    /// # Errors
    ///
    /// 输出缓冲不是单元素时返回说明。
    pub fn forward<R: Reduce>(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        if out.len() != 1 {
            return Err(Error::new(
                Code::Invalid,
                format!("归约输出必须是单元素，得到 {} 个", out.len()),
            ));
        }
        let mut acc = R::start();
        for x in a {
            acc = R::fold(acc, *x);
        }
        out[0] = acc;
        Ok(())
    }

    /// 生成一份 f32 归约算子的公开面：标记类型、[`Reduce`] 实现、[`annotation::model`] 别名、
    /// `forward` 入口与测试挂载。`reduce_max` / `reduce_sum` 只差初值与折叠闭包；
    /// 新增一种归约加一次调用即可。
    #[macro_export]
    macro_rules! reduce_f32 {
        ($name:ident, $alias:ident, $op:tt, $cases:expr, $start:expr, $fold:expr) => {
            #[doc = concat!("`", $op, "` 的标记类型：初值与折叠见 `start`/`fold`，骨架见 `crate::reduce`。")]
            pub struct $name;

            impl $crate::formula::FormulaOwner for $name {
                const EXPR: &'static str = "n,o->o";

                const OPERANDS: &'static [$crate::einsum::OperandSpec] =
                    &$crate::einsum::f32_io_operands();

                const FRAME: &'static str = "nn::reduce::formula";

                fn cache() -> &'static $crate::einsum::FormulaCache {
                    $crate::formula_cell!()
                }
            }

            impl $crate::reduce::Reduce for $name {
                fn start() -> f32 {
                    $start
                }

                fn fold(acc: f32, x: f32) -> f32 {
                    const FOLD: fn(f32, f32) -> f32 = $fold;
                    FOLD(acc, x)
                }
            }

            // 基座契约（ADR 0050）：只转发族骨架，不写第二份算术。迁移期与 `#[model]` 生成的
            // `MODEL` 并存，两者一致性由 `operator::tests` 逐字段锁住。
            $crate::operator_via!($name, $op, 1, (4096, 1), true, super, $crate::reduce::AXES);

            #[annotation::model(name = $op, shape = [4096, 1], dtype = "f32", skeleton = "reduce")]
            #[doc = concat!("`", $op, "` 的语义别名。")]
            pub type $alias = $name;

            /// 内核入口：结果写进调用方给的单元素缓冲。
            ///
            /// # Errors
            ///
            /// 输出不是单元素时返回说明。
            pub fn forward(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
                $crate::reduce::forward::<$name>(a, out)
            }

            #[cfg(test)]
            mod tests {
                use super::*;
                $crate::reduce::test_support::reduce_op_tests!($name, forward, $start);
                $crate::reduce::test_support::reduce_saturating_tests!($name, forward, $cases);
            }
        };
    }

    #[cfg(test)]
    mod test_support {
        include!("../tests/kernel/reduce_test_support.rs");
    }

    // 声明式算子表：同一副归约骨架的两种算子（最大值 / 求和），每个算子一个 `pub mod` 保持
    // `nn::<name>::MODEL` / `nn::<name>::forward` 路径不变（`forward` 同名，必须隔在各自的模块里）。
    // 新增一种归约在此补一段。

    pub mod reduce_max_f32 {
        //! 最大值归约：`y[0] = max(a[0..n])`。

        crate::reduce_f32!(
            Max,
            ReduceMaxF32,
            "reduce_max_f32",
            &[
                &[0.0, 0.0, 0.0],
                &[f32::MIN, f32::MAX, -1.0],
                &[f32::MIN, f32::MIN],
                &[2.0, 2.0, 2.0],
                &[-3.0, -1.0, -2.0],
            ],
            f32::MIN,
            |acc, x| if x > acc { x } else { acc }
        );
    }

    pub mod reduce_sum_f32 {
        //! 求和归约：`y[0] = sum(a[0..n])`。

        crate::reduce_f32!(
            Sum,
            ReduceSumF32,
            "reduce_sum_f32",
            &[
                &[0.0, 0.0, 0.0],
                &[f32::MAX, f32::MAX],
                &[f32::MIN, f32::MIN],
                &[1.0, 1.0, 1.0, 1.0],
                &[-3.0, 0.0, 3.0],
            ],
            0.0,
            |acc, x| acc + x
        );
    }
}
pub mod rmsnorm {
    //! RMSNorm：`y[n] = gamma[n] * x[n] / sqrt(mean(x[0..n]^2) + eps)`。
    use crate::model::{Buf, Cost, Sample, ShapeCtx, Values, cost_n, x_buf};
    use diagnostic::{Code, Error};

    /// 数值稳定项：均方根里加的小常数。
    pub const EPS: f32 = 1e-6;

    // 单轴定义与形状投影复用 unary 骨架：本文件只声明 rmsnorm 特有的代价、样本与遍历。
    pub use crate::unary::{AXES, project_shape};

    /// 标量代价：两趟读 `x`、一趟读 `gamma`、一趟写 `y`，运算按每元素约 4 次（平方、加法、
    /// 除法、乘法）估算。这里给出可审计的显式口径，不伪装成一条 einsum。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        cost_n(shape, "rmsnorm", 4.0, 3)
    }

    /// 确定性样本：`x`、`gamma`、`y` 三个等长缓冲。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            bufs: vec![
                x_buf(len),
                Buf {
                    name: "gamma",
                    values: Values::F32((0..len).map(|i| 0.5 + (i as f32) * 0.01).collect()),
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
    }

    /// 参考两参前置：长度一致且非空（空行没有可统计的均方根）。
    fn check_ref(x: &[f32], gamma: &[f32]) -> diagnostic::Result<()> {
        if x.len() != gamma.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("x 与 gamma 长度不一致：{} 与 {}", x.len(), gamma.len()),
            ));
        }
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "rmsnorm 输入不能为空：均方根无定义",
            ));
        }
        Ok(())
    }

    /// 内核三参前置：等长且非空。
    fn check_kernel(x: &[f32], gamma: &[f32], out: &[f32]) -> diagnostic::Result<()> {
        if x.len() != gamma.len() || x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "缓冲长度不一致：x {}、gamma {}、out {}",
                    x.len(),
                    gamma.len(),
                    out.len()
                ),
            ));
        }
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "rmsnorm 输入不能为空：均方根无定义",
            ));
        }
        Ok(())
    }

    /// 归一化第一趟：平方和。累加是串行依赖，保留逐元素循环；参考与内核共用同一份口径。
    fn sum_sq(x: &[f32]) -> f32 {
        let mut sum = 0.0;
        for value in x {
            sum += value * value;
        }
        sum
    }

    /// 标量参考：两趟，与内核同一公式。
    ///
    /// # Errors
    ///
    /// 输入为空时返回说明：空行没有可统计的均方根。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let gamma = sample.f32("gamma")?;
        check_ref(x, gamma)?;
        let rms = (sum_sq(x) / (x.len() as f32) + EPS).sqrt();
        Ok(x.iter()
            .zip(gamma.iter())
            .map(|(value, weight)| weight * value / rms)
            .collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 三个缓冲等长，或输入为空时返回说明：空行没有可统计的均方根。
    pub fn forward(x: &[f32], gamma: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        check_kernel(x, gamma, out)?;
        let rms = (sum_sq(x) / (x.len() as f32) + EPS).sqrt();
        for ((out, x), gamma) in out.iter_mut().zip(x).zip(gamma) {
            *out = gamma * x / rms;
        }
        Ok(())
    }

    /// 批段前置：行宽非空、gamma 与行宽等长、x/out 等长且为行宽整数倍；返回批内段数。
    fn rmsnorm_batch_segments(
        x: &[f32],
        gamma: &[f32],
        out: &[f32],
        row_dim: usize,
    ) -> Result<usize, Error> {
        if row_dim == 0 {
            return Err(Error::new(Code::Invalid, "rmsnorm 行宽不能为 0"));
        }
        if gamma.len() != row_dim {
            return Err(Error::new(
                Code::Invalid,
                format!("gamma 长度 {} 不等于行宽 {row_dim}", gamma.len()),
            ));
        }
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("x {} 与 out {} 长度不一致", x.len(), out.len()),
            ));
        }
        check_batch_rows(x, row_dim)?;
        Ok(x.len() / row_dim)
    }

    /// 批行校验：缓冲非空且为行宽整数倍；空行没有可统计的均方根。
    fn check_batch_rows(x: &[f32], row_dim: usize) -> Result<(), Error> {
        if x.is_empty() || !x.len().is_multiple_of(row_dim) {
            return Err(Error::new(Code::Invalid, "rmsnorm 批缓冲不是行宽整数倍"));
        }
        Ok(())
    }

    /// 批量入口：`[batch, row_dim]` 段连续，每段独立 RMS 归一化，gamma 按行宽复用。
    ///
    /// # Errors
    ///
    /// 行宽为 0、gamma 与行宽不等长、x/out 不等长、缓冲非行宽倍数或为空时返回说明。
    pub fn forward_batch(
        x: &[f32],
        gamma: &[f32],
        out: &mut [f32],
        row_dim: usize,
    ) -> diagnostic::Result<()> {
        let batch = rmsnorm_batch_segments(x, gamma, out, row_dim)?;
        for b in 0..batch {
            let start = b * row_dim;
            forward(
                &x[start..start + row_dim],
                gamma,
                &mut out[start..start + row_dim],
            )?;
        }
        Ok(())
    }

    /// RMSNorm 的具体语义别名：`#[model]` 在它上面生成 `MODEL`。
    #[annotation::model(name = "rmsnorm_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rmsnorm")]
    pub type RmsNormF32 = RmsNorm;

    crate::operator_here!(RmsNorm, "rmsnorm_f32", 2, (4096, 4096), true, reference);

    /// 标记类型：`#[model]` 需要一个类型别名承载语义，这里用单元结构。
    pub struct RmsNorm;

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel/rmsnorm.rs");
    }
}
pub mod rope {
    //! RoPE（旋转位置编码）：按位置 `p` 对一对对分量做二维旋转。
    //!
    //! 对偶数下标 `2i` 与奇数下标 `2i+1`，频率 `freq_i = base^(-2i/d)`，角度 `angle = p*freq_i`：
    //!
    //! ```text
    //! y[2i]   = x[2i] cos(angle) - x[2i+1] sin(angle)
    //! y[2i+1] = x[2i] sin(angle) + x[2i+1] cos(angle)
    //! ```
    //!
    //! 常量 `base = 10000`。本模块自带形状、代价、样本、标量参考与遍历，`#[annotation::model]` 装配。
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::model::{Cost, Sample, ShapeCtx, cost_n};
    use diagnostic::{Code, Error};

    /// 频率底数。
    pub const BASE: f32 = 10000.0;

    thread_local! {
        /// 段长 → 频率表。`freq_i = base^(-2i/d)` 只依赖 `d`，与位置、数据都无关。
        static FREQUENCIES: RefCell<Option<(usize, Rc<[f32]>)>> = const { RefCell::new(None) };
    }

    /// 段长 `d` 的频率表：`d/2` 条，`freq[i] = base^(-2i/d)`。
    pub fn frequencies(d: usize) -> Rc<[f32]> {
        FREQUENCIES.with(|slot| {
            let mut slot = slot.borrow_mut();
            if let Some((cached, table)) = slot.as_ref()
                && *cached == d
            {
                return Rc::clone(table);
            }
            let table: Rc<[f32]> = (0..d / 2)
                .map(|i| BASE.powf(-((2 * i) as f32) / d as f32))
                .collect();
            *slot = Some((d, Rc::clone(&table)));
            table
        })
    }

    // 单轴定义与形状投影复用 unary 骨架。
    pub use crate::unary::{AXES, project_shape};

    /// 标量代价：每对分量约 6 次运算（两对乘加 + 一次 sin/cos 视为共享），按元素约 4 次估算。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        cost_n(shape, "rope", 4.0, 1)
    }

    /// 确定性样本：`x`、`y` 两个等长（偶数）缓冲，位置固定为 7。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample::f32_io(len, len, len, len, vec![("N", len as i64), ("pos", 7)])
    }

    /// 标量参考：与内核同一公式，位置取样本标量 `P`。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        let pos = sample
            .scalar("pos")
            .ok_or_else(|| Error::new(Code::Invalid, "rope 样本缺标量 `pos`"))? as f32;
        rotate(x, pos)
    }

    /// 共享旋转：参考与内核的同一套数学。末尾若有无法配对的元素，原样透传。
    fn rotate(x: &[f32], pos: f32) -> diagnostic::Result<Vec<f32>> {
        let mut out = vec![0.0; x.len()];
        rotate_into(x, pos, &mut out)?;
        Ok(out)
    }

    /// 旋转写进给定缓冲：参考与内核共用的算术本体。内核路径由此直写调用方缓冲，
    /// 不再先落一份临时 `Vec` 再整块拷贝；末尾不成对的元素原样透传。
    fn rotate_into(x: &[f32], pos: f32, out: &mut [f32]) -> diagnostic::Result<()> {
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
            ));
        }
        rotate_rows(x, pos, frequencies(x.len()).as_ref(), out);
        Ok(())
    }

    /// 按已备好的频率表旋转：`freq[i]` 就是第 `i` 对的频率，长度校验已在调用方做完。
    fn rotate_rows(x: &[f32], pos: f32, freq: &[f32], out: &mut [f32]) {
        let d = x.len();
        for (i, &f) in freq.iter().enumerate() {
            let (sin, cos) = (pos * f).sin_cos();
            let (e, o) = (x[2 * i], x[2 * i + 1]);
            out[2 * i] = e * cos - o * sin;
            out[2 * i + 1] = e * sin + o * cos;
        }
        if d % 2 == 1 {
            out[d - 1] = x[d - 1];
        }
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输入与输出长度不一致时返回说明；末尾无法配对的奇数元素原样透传（不报错）。
    pub fn forward(pos: usize, x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        rotate_into(x, pos as f32, out)
    }

    /// 校验多头旋转长度并派生头数；非法头宽、不等长或非头宽倍数在此一次性拦下。
    fn rope_heads(x: &[f32], out: &[f32], head_dim: usize) -> diagnostic::Result<usize> {
        if head_dim == 0 || x.len() != out.len() || !x.len().is_multiple_of(head_dim) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "RoPE 头宽 {head_dim} 非法，或长度 {} 与其/输出长度不符",
                    x.len()
                ),
            ));
        }
        Ok(x.len() / head_dim)
    }

    /// 多头旋转：头优先块布局 `[heads, head_dim]`，每段独立按头宽重计频率（段内下标从 0 起），
    /// 不把整段长度当一个头宽；段内旋转算术复用 `rotate_into`，这里只做头分段编排。
    ///
    /// # Errors
    ///
    /// 头宽为 0、缓冲不是头宽的整数倍，或输入输出长度不一致时返回说明。
    pub fn forward_heads(
        pos: usize,
        x: &[f32],
        out: &mut [f32],
        head_dim: usize,
    ) -> diagnostic::Result<()> {
        rope_heads(x, out, head_dim)?;
        rotate_heads(pos, x, out, head_dim, frequencies(head_dim).as_ref());
        Ok(())
    }

    /// 按已备好的频率表旋转整段头优先块：`heads` 个头，每头 `head_dim`，段内下标从 0 重计。
    fn rotate_heads(pos: usize, x: &[f32], out: &mut [f32], head_dim: usize, freq: &[f32]) {
        for start in (0..x.len()).step_by(head_dim) {
            let span = start..start + head_dim;
            rotate_rows(&x[span.clone()], pos as f32, freq, &mut out[span]);
        }
    }

    /// 批量校验：返回单序列段长 `heads·head_dim`；批数取 `positions.len()`，段内头宽/倍数守卫
    /// 交给 [`forward_heads`]。
    fn rope_batch_segments(
        positions: &[usize],
        x: &[f32],
        out: &[f32],
        head_dim: usize,
    ) -> diagnostic::Result<usize> {
        let batch = positions.len();
        if batch == 0 || head_dim == 0 {
            return Err(Error::new(
                Code::Invalid,
                format!("批量序列数 {batch} 与头宽 {head_dim} 都必须非零"),
            ));
        }
        if x.len() != out.len() || !x.len().is_multiple_of(batch * head_dim) {
            return Err(Error::new(
                Code::Invalid,
                format!(
                    "x 长度 {} 不是 batch·head_dim（{}）的整数倍，或与输出不等长",
                    x.len(),
                    batch * head_dim
                ),
            ));
        }
        Ok(x.len() / batch)
    }

    /// 批量旋转：`[batch, heads, head_dim]` 头优先块，第 `b` 段连续 `heads·head_dim`，每序列
    /// 独立位置 `positions[b]`；段内复用 [`forward_heads`] 头分段编排，段间互不串段。
    ///
    /// # Errors
    ///
    /// `positions` 为空、头宽为 0、输入输出不等长或长度不是批段整数倍时返回说明。
    pub fn forward_batch(
        positions: &[usize],
        x: &[f32],
        out: &mut [f32],
        head_dim: usize,
    ) -> diagnostic::Result<()> {
        let seg = rope_batch_segments(positions, x, out, head_dim)?;
        // 先把每个序列的段形状校验完，再统一写：任一序列非法时不会留下写了一半的输出。
        for start in (0..x.len()).step_by(seg) {
            let span = start..start + seg;
            rope_heads(&x[span.clone()], &out[span], head_dim)?;
        }
        let freq = frequencies(head_dim);
        for (seq, &pos) in positions.iter().enumerate() {
            let start = seq * seg;
            let span = start..start + seg;
            rotate_heads(
                pos,
                &x[span.clone()],
                &mut out[span],
                head_dim,
                freq.as_ref(),
            );
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Rope;

    /// RoPE 的具体语义别名。
    #[annotation::model(name = "rope_f32", shape = [4096, 4096], dtype = "f32", skeleton = "rope")]
    pub type RopeF32 = Rope;

    crate::operator_here!(Rope, "rope_f32", 1, (4096, 4096), true, reference);

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel/rope.rs");
    }
}
pub mod scalar {
    //! 浮点标量闭集：一份算术骨架为 `f32` / `f64` / `half::f16` / `half::bf16` 实现。

    use core::fmt::Debug;

    /// 浮点标量闭集：`f32` / `f64` / `half::f16` / `half::bf16` 满足；`nn` 的脉冲 / 合成 / 决策头复用本 trait。
    pub trait Scalar:
        Copy
        + PartialEq
        + PartialOrd
        + Debug
        + core::ops::Add<Output = Self>
        + core::ops::Sub<Output = Self>
        + core::ops::Mul<Output = Self>
        + core::ops::Div<Output = Self>
    {
        /// 加法幺元。
        const ZERO: Self;
        /// 乘法幺元。
        const ONE: Self;
        /// 负无穷（DSA top-k 掩码：`softmax(−∞)=0`，实现真稀疏注意力）。
        const NEG_INFINITY: Self;

        /// 自然指数 `exp(x)`。
        fn exp(self) -> Self;
        /// `exp(x) - 1`（无 `x≈0` 时的灾难性消去）。
        fn exp_m1(self) -> Self;
        /// 平方根。
        fn sqrt(self) -> Self;
        /// 绝对值。
        fn abs(self) -> Self;
        /// 取两数较大者。
        fn max(self, rhs: Self) -> Self;
        /// 取两数较小者。
        fn min(self, rhs: Self) -> Self;
        /// 向下取整（池化/插值的坐标下取整）。
        fn floor(self) -> Self;
        /// 自然对数（自信息 `−ln p` 与交叉熵的唯一来源，不另写近似）。
        fn ln(self) -> Self;
        /// 正弦（复值状态的相位，单位弧度）。
        fn sin(self) -> Self;
        /// 余弦（复值状态的相位，单位弧度）。
        fn cos(self) -> Self;
        /// 由 `f64` 字面量构造（仅用于常量，`f32` 下可能有损）。
        fn from_f64(v: f64) -> Self;
        /// 由 `usize` 构造：形状/索引数学（步长、核尺寸、面积）的通用入口，不绕道 `f64`。
        fn from_usize(v: usize) -> Self;
        /// 取回 `usize`（截断向零）：只用于形状/索引数学，调用方保证值非负且落在 `usize` 范围内。
        fn to_usize(self) -> usize;
        /// 就近取整，半分远离零（`f64::round` 的口径）。
        fn round(self) -> Self;
        /// 取回 `f64`。
        fn to_f64(self) -> f64;
    }

    /// 一份算术骨架为 `f32` / `f64` / `half::f16` / `half::bf16` 四个标量实现，能变的部分只有类型、
    /// 零/一/负无穷常量与 `f64`/`usize` 的进出转换式；逐字写四份会把 14 个方法的加/减/乘/除与
    /// 超越函数复制成四胞胎（jscpd 的克隆源）。数学函数统一走 [`num_traits::Float`]：`f32`/`f64`
    /// 由该 trait 直接委托给原生实现，`half` 类型由 half 自带实现经 `f32` 往返得到，语义与精度由
    /// half 侧保证，`op` 侧不复制任何一份算术。
    macro_rules! impl_scalar {
        ($ty:ty, $zero:expr, $one:expr, $neg_inf:expr, $from_f64:expr, $from_usize:expr, $to_f64:expr, $to_usize:expr) => {
            impl Scalar for $ty {
                const ZERO: Self = $zero;
                const ONE: Self = $one;
                const NEG_INFINITY: Self = $neg_inf;

                fn exp(self) -> Self {
                    num_traits::Float::exp(self)
                }
                fn exp_m1(self) -> Self {
                    num_traits::Float::exp_m1(self)
                }
                fn sqrt(self) -> Self {
                    num_traits::Float::sqrt(self)
                }
                fn abs(self) -> Self {
                    num_traits::Float::abs(self)
                }
                fn max(self, rhs: Self) -> Self {
                    num_traits::Float::max(self, rhs)
                }
                fn min(self, rhs: Self) -> Self {
                    num_traits::Float::min(self, rhs)
                }
                fn floor(self) -> Self {
                    num_traits::Float::floor(self)
                }
                fn ln(self) -> Self {
                    num_traits::Float::ln(self)
                }
                fn sin(self) -> Self {
                    num_traits::Float::sin(self)
                }
                fn cos(self) -> Self {
                    num_traits::Float::cos(self)
                }
                fn from_f64(v: f64) -> Self {
                    ($from_f64)(v)
                }
                fn from_usize(v: usize) -> Self {
                    ($from_usize)(v)
                }
                fn round(self) -> Self {
                    num_traits::Float::round(self)
                }
                fn to_f64(self) -> f64 {
                    ($to_f64)(self)
                }
                fn to_usize(self) -> usize {
                    ($to_usize)(self)
                }
            }
        };
    }

    impl_scalar!(
        f32,
        0.0_f32,
        1.0_f32,
        f32::NEG_INFINITY,
        |v| v as f32,
        |v| v as f32,
        |v| v as f64,
        |v| v as usize
    );
    impl_scalar!(
        f64,
        0.0_f64,
        1.0_f64,
        f64::NEG_INFINITY,
        |v| v,
        |v| v as f64,
        |v| v,
        |v| v as usize
    );
    impl_scalar!(
        half::f16,
        half::f16::ZERO,
        half::f16::ONE,
        half::f16::NEG_INFINITY,
        half::f16::from_f64,
        |v| half::f16::from_f64(v as f64),
        half::f16::to_f64,
        |v| half::f16::to_f64(v) as usize
    );
    impl_scalar!(
        half::bf16,
        half::bf16::ZERO,
        half::bf16::ONE,
        half::bf16::NEG_INFINITY,
        half::bf16::from_f64,
        |v| half::bf16::from_f64(v as f64),
        half::bf16::to_f64,
        |v| half::bf16::to_f64(v) as usize
    );
}
pub mod softmax {
    //! Softmax：`y[n] = exp(x[n] - max(x)) / sum_k exp(x[k] - max(x))`。
    use crate::model::{Cost, Sample, ShapeCtx, cost_n};
    use diagnostic::{Code, Error};

    // 单轴定义与形状投影复用 unary 骨架：本文件只声明 softmax 特有的代价、样本与遍历。
    pub use crate::unary::{AXES, project_shape};

    /// 标量代价：三趟读 `x`、一趟写 `y`，每元素约 3 次运算（max、exp、除法）。
    ///
    /// # Errors
    ///
    /// 形状缺轴时返回说明。
    pub fn cost(shape: &ShapeCtx) -> Result<Cost, Error> {
        cost_n(shape, "softmax", 3.0, 3)
    }

    /// 确定性样本：`x`、`y` 两个等长缓冲。
    pub fn sample(len: usize, _: usize) -> Sample {
        Sample::f32_io(len, len, len, len, vec![("N", len as i64)])
    }

    /// 标量参考：三趟，与内核同一公式。
    ///
    /// # Errors
    ///
    /// 输入为空时返回说明：空行没有可归一化的分母。
    pub fn reference(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let x = sample.f32("x")?;
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "softmax 输入不能为空：归一化分母无定义",
            ));
        }
        let max = x.iter().copied().fold(f32::MIN, f32::max);
        let mut sum = 0.0;
        let exps: Vec<f32> = x
            .iter()
            .map(|value| {
                let e = (value - max).exp();
                sum += e;
                e
            })
            .collect();
        Ok(exps.iter().map(|e| e / sum).collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输入与输出等长，或输入为空时返回说明：空行没有可归一化的分母。
    pub fn forward(x: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        if x.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：x {}、out {}", x.len(), out.len()),
            ));
        }
        if x.is_empty() {
            return Err(Error::new(
                Code::Invalid,
                "softmax 输入不能为空：归一化分母无定义",
            ));
        }
        let max = x.iter().copied().fold(f32::MIN, f32::max);
        // 一趟 `exp`：先用写入的指数值累加分母，再原地除掉。旧实现算了两遍 `exp`
        // （一遍求 `sum`、一遍求结果），而 `exp` 正是这里最贵的指令；`(x-max).exp()`
        // 的取值与从左到右的累加顺序都和旧实现一致，所以输出逐位不变。
        let mut sum = 0.0;
        for (out, x) in out.iter_mut().zip(x) {
            let e = (x - max).exp();
            sum += e;
            *out = e;
        }
        for out in out.iter_mut() {
            *out /= sum;
        }
        Ok(())
    }

    /// 标记类型：承载 `#[model]` 语义。
    pub struct Softmax;

    /// Softmax 的具体语义别名。
    #[annotation::model(name = "softmax_f32", shape = [4096, 4096], dtype = "f32", skeleton = "softmax")]
    pub type SoftmaxF32 = Softmax;

    crate::operator_here!(Softmax, "softmax_f32", 1, (4096, 4096), true, reference);

    #[cfg(test)]
    mod tests {
        include!("../tests/kernel/softmax.rs");
    }
}
pub mod unary {
    //! 逐元素一元骨架：`y[n] = f(a[n])`。

    use crate::formula::FormulaOwner;
    use crate::model::{AxisRole, AxisSpec, Buf, Cost, Sample, ShapeCtx, Values};
    use diagnostic::{Code, Error};

    /// 逐元素一元算子随语义变化的那部分。
    pub trait Unary: FormulaOwner {
        /// 逐元素函数：一个输入值，一个输出值。
        fn apply(x: f32) -> f32;

        /// 输入的确定性取值：默认线性夹具 `−4.0 + 0.25·i`；需要不同取值时在实现里覆盖。
        fn input(len: usize) -> Vec<f32> {
            crate::model::linear(len, 0.25, -4.0)
        }
    }

    /// 一元骨架只有一条轴：既是输入轴也是输出轴。
    pub const AXES: &[AxisSpec] = &[AxisSpec {
        role: AxisRole::Input,
        label: 'n',
    }];

    /// 单轴形状：轴 `n` 取第一个参数（元素数）。
    ///
    /// # Errors
    ///
    /// 形状上下文构造失败时返回说明（当前实现不会失败）。
    pub fn project_shape(len: usize, _: usize) -> Result<ShapeCtx, Error> {
        ShapeCtx::new([('n', len)])
    }

    /// 标量代价：完全由表达式与操作数位宽推出。
    pub fn cost<U: Unary>(shape: &ShapeCtx) -> Result<Cost, Error> {
        U::formula()?
            .cost(shape.extents())
            .map_err(|error| error.with("nn::unary::cost"))
    }

    /// 确定性样本：第一个参数是元素数。
    pub fn sample<U: Unary>(len: usize, _: usize) -> Sample {
        Sample {
            m: len,
            n: len,
            scalars: vec![("N", len as i64)],
            bufs: vec![
                Buf {
                    name: "a",
                    values: Values::F32(U::input(len)),
                },
                Buf {
                    name: "y",
                    values: Values::F32(vec![0.0; len]),
                },
            ],
        }
    }

    /// 标量参考：逐元素应用 [`Unary::apply`]。
    pub fn reference<U: Unary>(sample: &Sample) -> diagnostic::Result<Vec<f32>> {
        let a = sample.f32("a")?;
        Ok(a.iter().map(|x| U::apply(*x)).collect())
    }

    /// 内核本体：结果写进调用方给的缓冲。
    ///
    /// # Errors
    ///
    /// 输入与输出等长，否则返回说明。
    pub fn forward<U: Unary>(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
        if a.len() != out.len() {
            return Err(Error::new(
                Code::Invalid,
                format!("缓冲长度不一致：a {}、out {}", a.len(), out.len()),
            ));
        }
        // 切片 zip：连续内存、无下标边界检查，便于自动向量化。
        for (out, x) in out.iter_mut().zip(a) {
            *out = U::apply(*x);
        }
        Ok(())
    }

    /// 生成一份 f32 一元逐元素算子的公开面：标记类型、[`Unary`] 实现、[`annotation::model`] 别名、
    /// `forward` 入口与测试挂载。`scale` / `gelu` / `silu` 只差一行 `apply`（可选样本夹具）；
    /// 新增一种一元运算加一次调用即可。
    #[macro_export]
    macro_rules! unary_f32 {
        (
            $name:ident, $alias:ident, $op:tt,
            $kind:ident, $edge:tt,
            $apply:expr
            $(, $input:expr)?
        ) => {
            #[doc = concat!("`", $op, "` 的标记类型：逐元素一元运算见 `apply`，骨架见 `crate::unary`。")]
            pub struct $name;

            impl $crate::formula::FormulaOwner for $name {
                const EXPR: &'static str = "n,n->n";

                const OPERANDS: &'static [$crate::einsum::OperandSpec] =
                    &$crate::einsum::f32_io_operands();

                const FRAME: &'static str = "nn::unary::formula";

                fn cache() -> &'static $crate::einsum::FormulaCache {
                    $crate::formula_cell!()
                }
            }

            impl $crate::unary::Unary for $name {
                fn apply(x: f32) -> f32 {
                    const APPLY: fn(f32) -> f32 = $apply;
                    APPLY(x)
                }

                $( fn input(len: usize) -> Vec<f32> {
                    const INPUT: fn(usize) -> Vec<f32> = $input;
                    INPUT(len)
                } )?
            }

            // 基座契约（ADR 0050）：只转发族骨架，不写第二份算术。
            $crate::operator_via!($name, $op, 1, (4096, 4096), true, super, $crate::unary::AXES);

            #[annotation::model(name = $op, shape = [4096, 4096], dtype = "f32", skeleton = "unary")]
            #[doc = concat!("`", $op, "` 的语义别名。")]
            pub type $alias = $name;

            /// 内核入口：结果写进调用方给的缓冲。
            ///
            /// # Errors
            ///
            /// 输入与输出等长，否则返回说明。
            pub fn forward(a: &[f32], out: &mut [f32]) -> diagnostic::Result<()> {
                $crate::unary::forward::<$name>(a, out)
            }

            #[cfg(test)]
            mod tests {
                use super::*;
                $crate::unary::test_support::$kind!($name, forward);

                #[test]
                fn edge_cases_match_reference() {
                    $edge
                }
            }
        };
    }

    #[cfg(test)]
    mod test_support {
        include!("../tests/kernel/unary_test_support.rs");
    }

    // 声明式算子表：同一副一元骨架的五个算子，每个算子一个 `pub mod` 保持 `nn::<name>::MODEL` /
    // `nn::<name>::forward` 路径不变（`forward` 同名，必须隔在各自的模块里）。新增一元运算在此补一段。

    pub mod exp_f32 {
        //! 逐元素指数：`y[n] = exp(a[n])`。

        crate::unary_f32!(
            Exp,
            ExpF32,
            "exp_f32",
            unary_pair_tests,
            {
                // 边界：exp(0)=1、大负数下溢为 0、exp(1)=e。
                let mut out = vec![0.0; 3];
                assert!(forward(&[0.0, -1000.0, 1.0], &mut out).is_ok(), "等长");
                assert_eq!(out[0], 1.0, "exp(0)=1");
                assert_eq!(out[1], 0.0, "exp(-1000) 下溢为 0");
                assert!((out[2] - std::f32::consts::E).abs() < 1e-6, "exp(1)=e");
            },
            |x| x.exp(),
            |len| (0..len).map(|i| ((i % 65) as f32) * 0.25 - 8.0).collect()
        );
    }

    pub mod log_f32 {
        //! 逐元素自然对数：`y[n] = ln(a[n])`。

        crate::unary_f32!(
            Log,
            LogF32,
            "log_f32",
            unary_pair_tests,
            {
                // 边界：ln(1)=0、ln(e)=1。
                let mut out = vec![0.0; 2];
                assert!(
                    forward(&[1.0, std::f32::consts::E], &mut out).is_ok(),
                    "等长"
                );
                assert_eq!(out[0], 0.0, "ln(1)=0");
                assert!((out[1] - 1.0).abs() < 1e-6, "ln(e)=1");
            },
            |x| x.ln(),
            |len| (0..len).map(|i| 0.5 + i as f32 * 0.25).collect()
        );
    }

    pub mod gelu_f32 {
        //! 逐元素 GELU 激活（精确 erf 版）：`y[n] = 0.5*a[n]*(1 + erf(a[n]/sqrt(2)))`。

        /// 标量误差函数：Abramowitz-Stegun 手册公式 7.1.26（`|误差| < 1.5e-7`）。
        pub fn erf(x: f32) -> f32 {
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            let x = x.abs();
            let p = 0.327_591_1;
            let a1 = 0.254_829_6;
            let a2 = -0.284_496_72;
            let a3 = 1.421_413_8;
            let a4 = -1.453_152_1;
            let a5 = 1.061_405_4;
            let t = 1.0 / (1.0 + p * x);
            let series = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
            let y = 1.0 - series * (-x * x).exp();
            sign * y
        }

        crate::unary_f32!(
            Gelu,
            GeluF32,
            "gelu_f32",
            unary_pair_tests,
            {
                // 精确 erf 的已知值钉住：Abramowitz-Stegun 7.1.26 闭式，两侧绝对误差 < 1.5e-7。
                assert!((erf(0.0) - 0.0).abs() < 1e-6);
                assert!((erf(1.0) - 0.842_700_8).abs() < 1.5e-6);
                assert!((erf(-1.0) + 0.842_700_8).abs() < 1.5e-6);
            },
            |x| 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
        );
    }

    pub mod silu_f32 {
        //! 逐元素 SiLU 激活：`y[n] = a[n] * sigmoid(a[n])`。

        crate::unary_f32!(
            Silu,
            SiluF32,
            "silu_f32",
            unary_op_tests,
            {
                // silu(1) = sigmoid(1)：1/(1+e^-1) 的已知值钉住。
                let mut out = [0.0; 1];
                assert!(forward(&[1.0], &mut out).is_ok(), "等长");
                assert!(
                    (out[0] - 0.731_058_6).abs() < 1e-6,
                    "silu(1) = {out}",
                    out = out[0]
                );
            },
            |x| x / (1.0 + (-x).exp())
        );
    }

    pub mod leaky_relu_f32 {
        //! 逐元素 LeakyReLU：`y[n] = a[n]`（非负）或 `slope·a[n]`（负）。

        /// 斜率常量：与内核模板、PyTorch 默认值一致。
        pub const SLOPE: f32 = 0.01;

        crate::unary_f32!(
            LeakyRelu,
            LeakyReluF32,
            "leaky_relu_f32",
            unary_op_tests,
            {
                // 三个锚点：正区恒等、负区按 slope 缩放、零点两侧连续。
                let mut out = [0.0; 3];
                assert!(forward(&[-2.0, 0.0, 2.0], &mut out).is_ok(), "等长");
                assert!((out[0] - (-2.0 * SLOPE)).abs() < 1e-6, "负区 {}", out[0]);
                assert_eq!(out[1], 0.0, "零点");
                assert_eq!(out[2], 2.0, "正区恒等");
            },
            |x| if x >= 0.0 { x } else { SLOPE * x }
        );
    }

    pub mod relu_f32 {
        //! 逐元素 ReLU：`y[n] = max(a[n], 0)`。

        crate::unary_f32!(
            Relu,
            ReluF32,
            "relu_f32",
            unary_op_tests,
            {
                // 三个锚点：负区归零、零保持、正区恒等。
                let mut out = [0.0; 3];
                assert!(forward(&[-2.0, 0.0, 2.0], &mut out).is_ok(), "等长");
                assert_eq!(out[0], 0.0, "负区归零");
                assert_eq!(out[1], 0.0, "零点");
                assert_eq!(out[2], 2.0, "正区恒等");
            },
            |x| x.max(0.0)
        );
    }

    pub mod sigmoid_f32 {
        //! 逐元素 Sigmoid：`y[n] = 1 / (1 + exp(-a[n]))`。

        crate::unary_f32!(
            Sigmoid,
            SigmoidF32,
            "sigmoid_f32",
            unary_op_tests,
            {
                // 三个锚点：σ(0)=0.5、σ(1)、两端饱和（不溢出成 NaN）。
                let mut out = [0.0; 4];
                assert!(
                    forward(&[0.0, 1.0, -1000.0, 1000.0], &mut out).is_ok(),
                    "等长"
                );
                assert_eq!(out[0], 0.5, "σ(0)");
                assert!((out[1] - 0.731_058_6).abs() < 1e-6, "σ(1) = {}", out[1]);
                assert_eq!(out[2], 0.0, "下饱和不溢出");
                assert_eq!(out[3], 1.0, "上饱和不溢出");
            },
            |x| 1.0 / (1.0 + (-x).exp())
        );
    }

    pub mod tanh_f32 {
        //! 逐元素双曲正切：`y[n] = 2 / (1 + exp(-2·a[n])) - 1`。

        crate::unary_f32!(
            Tanh,
            TanhF32,
            "tanh_f32",
            unary_op_tests,
            {
                // 三个锚点：tanh(0)=0、tanh(1) 的已知值、两端饱和（线性函数族不可能满足）。
                let mut out = [0.0; 4];
                assert!(
                    forward(&[0.0, 1.0, -1000.0, 1000.0], &mut out).is_ok(),
                    "等长"
                );
                assert_eq!(out[0], 0.0, "tanh(0)");
                assert!((out[1] - 0.761_594_2).abs() < 1e-6, "tanh(1) = {}", out[1]);
                assert_eq!(out[2], -1.0, "下饱和");
                assert_eq!(out[3], 1.0, "上饱和");
            },
            |x| 2.0 / (1.0 + (-2.0 * x).exp()) - 1.0
        );
    }

    pub mod hard_sigmoid_f32 {
        //! 逐元素 HardSigmoid：`y[n] = clamp(a[n] / 6 + 0.5, 0, 1)`。

        crate::unary_f32!(
            HardSigmoid,
            HardSigmoidF32,
            "hard_sigmoid_f32",
            unary_op_tests,
            {
                // 三段：下饱和 0、中心 0.5、上饱和 1，以及两端斜率 1/6。
                let mut out = [0.0; 4];
                assert!(forward(&[-6.0, 0.0, 6.0, 3.0], &mut out).is_ok(), "等长");
                assert_eq!(out[0], 0.0, "下饱和");
                assert_eq!(out[1], 0.5, "中心");
                assert_eq!(out[2], 1.0, "上饱和");
                assert!((out[3] - 1.0).abs() < 1e-6, "3 落在上饱和 {}", out[3]);
            },
            |x| {
                let v = x / 6.0 + 0.5;
                v.clamp(0.0, 1.0)
            }
        );
    }

    pub mod scale_f32 {
        //! 标量缩放：`y[n] = alpha * a[n]`。

        /// 样本固定的缩放系数（内核模板里写同一常量）。
        pub const ALPHA: f32 = 0.5;

        crate::unary_f32!(
            Scale,
            ScaleF32,
            "scale_f32",
            unary_op_tests,
            {
                // scale(1) = ALPHA：把样本常量与内核模板常量钉在同一处。
                let mut out = [0.0; 1];
                assert!(forward(&[1.0], &mut out).is_ok(), "等长");
                assert_eq!(out[0], ALPHA, "scale(1) = ALPHA");
            },
            |x| ALPHA * x,
            |len| (0..len).map(|i| ((i as f32) * 0.25) - 2.0).collect()
        );
    }
}

// 一元 / 二元 / 归约 / 反量化 / GEMV / GEMM 骨架内的声明式子模块，re-export 到 crate 根
// 保持 `<crate>::<name>::MODEL` / `forward` 路径。
pub use dequant::{dequant_i2, dequant_i4, i8 as dequant_i8};
pub use elementwise::{add_f32, div_f32, max_f32, min_f32, mul_f32, residual_add_f32, sub_f32};
pub use epilogue::apply as apply_epilogue;
pub use gemm::{f32 as gemm_f32, i2 as gemm_i2, i4 as gemm_i4, i8 as gemm_i8};
pub use gemv::{f32 as gemv_f32, i2 as gemv_i2, i4 as gemv_i4, i8 as gemv_i8};
pub use matmul::i8 as matmul_i8;
pub use reduce::{reduce_max_f32, reduce_sum_f32};
pub use scalar::Scalar;
pub use unary::{
    exp_f32, gelu_f32, hard_sigmoid_f32, leaky_relu_f32, log_f32, relu_f32, scale_f32, sigmoid_f32,
    silu_f32, tanh_f32,
};
pub use {embed as embed_f32, rmsnorm as rmsnorm_f32, rope as rope_f32, softmax as softmax_f32};

// 契约层在 `spec`：内核与本 crate 的骨架宏都按 crate 根解析契约路径，这里把它们搬回来，
// 于是内核源码里写的是 `crate::model::…` / `crate::operator_via!`，而不是到处写依赖 crate 名。
pub use prof::same_bytes;
// `$crate::formula_cell!` / `$crate::operator_via!` 在骨架宏里按 crate 根解析，这里把 spec 导出的
// 宏搬回本 crate 根：内核里几十处 `#[model]` 展开都不必改路径。
pub use model::quant;
pub use prof::tpl::{formula, parse};
pub use prof::{einsum, model, operator, sample, sch, sig, tpl};
pub use prof::{formula_cell, operator_here, operator_via};
pub use sch::layout;

#[cfg(test)]
#[path = "../tests/kernel/prop_numeric.rs"]
mod prop_numeric;

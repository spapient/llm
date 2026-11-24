//! 一元算子实例：同一副 [`crate::unary`] 骨架的逐元素算子，每个算子一个 `pub mod`，
//! 保持 `nn::<name>::MODEL` / `nn::<name>::forward` 路径不变（`forward` 同名，必须隔在各自模块）。
//! 新增一元运算在此补一段。

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
            assert!(
                forward(tensor::View::one(&[0.0, -1000.0, 1.0]), &mut out).is_ok(),
                "等长"
            );
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
                forward(tensor::View::one(&[1.0, std::f32::consts::E]), &mut out).is_ok(),
                "等长"
            );
            assert_eq!(out[0], 0.0, "ln(1)=0");
            assert!((out[1] - 1.0).abs() < 1e-6, "ln(e)=1");
        },
        |x| x.ln(),
        |len| crate::model::linear(len, 0.25, 0.5)
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
            assert!(forward(tensor::View::one(&[1.0]), &mut out).is_ok(), "等长");
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
            assert!(
                forward(tensor::View::one(&[-2.0, 0.0, 2.0]), &mut out).is_ok(),
                "等长"
            );
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
            assert!(
                forward(tensor::View::one(&[-2.0, 0.0, 2.0]), &mut out).is_ok(),
                "等长"
            );
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
                forward(tensor::View::one(&[0.0, 1.0, -1000.0, 1000.0]), &mut out).is_ok(),
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
                forward(tensor::View::one(&[0.0, 1.0, -1000.0, 1000.0]), &mut out).is_ok(),
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
            assert!(
                forward(tensor::View::one(&[-6.0, 0.0, 6.0, 3.0]), &mut out).is_ok(),
                "等长"
            );
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
            assert!(forward(tensor::View::one(&[1.0]), &mut out).is_ok(), "等长");
            assert_eq!(out[0], ALPHA, "scale(1) = ALPHA");
        },
        |x| ALPHA * x,
        |len| crate::model::linear(len, 0.25, -2.0)
    );
}

pub mod abs_f32 {
    //! 逐元素绝对值：`y[n] = |a[n]|`。

    crate::unary_f32!(
        Abs,
        AbsF32,
        "abs_f32",
        unary_op_tests,
        {
            // 三个锚点：负区取正、零点、正区恒等。
            let mut out = [0.0; 3];
            assert!(
                forward(tensor::View::one(&[-3.0, 0.0, 2.0]), &mut out).is_ok(),
                "等长"
            );
            assert_eq!(out[0], 3.0, "负区取正");
            assert_eq!(out[1], 0.0, "零点");
            assert_eq!(out[2], 2.0, "正区恒等");
        },
        |x| x.abs(),
        false
    );
}

pub mod neg_f32 {
    //! 逐元素取负：`y[n] = -a[n]`。

    crate::unary_f32!(
        Neg,
        NegF32,
        "neg_f32",
        unary_op_tests,
        {
            // 三个锚点：负负得正、零点、正区取负。
            let mut out = [0.0; 3];
            assert!(
                forward(tensor::View::one(&[-2.0, 0.0, 3.0]), &mut out).is_ok(),
                "等长"
            );
            assert_eq!(out[0], 2.0, "负负得正");
            assert_eq!(out[1], 0.0, "零点");
            assert_eq!(out[2], -3.0, "正区取负");
        },
        |x| -x,
        false
    );
}

pub mod square_f32 {
    //! 逐元素平方：`y[n] = a[n]²`。

    crate::unary_f32!(
        Square,
        SquareF32,
        "square_f32",
        unary_pair_tests,
        {
            // 三个锚点：零、正数平方、负数平方同正。
            let mut out = [0.0; 3];
            assert!(
                forward(tensor::View::one(&[0.0, 3.0, -2.0]), &mut out).is_ok(),
                "等长"
            );
            assert_eq!(out[0], 0.0, "零平方");
            assert_eq!(out[1], 9.0, "3²");
            assert_eq!(out[2], 4.0, "(-2)²");
        },
        |x| x * x,
        false
    );
}

pub mod sign_f32 {
    //! 逐元素符号：`y[n] = sign(a[n])`（正 → 1、负 → −1、零 → 0）。

    crate::unary_f32!(
        Sign,
        SignF32,
        "sign_f32",
        unary_op_tests,
        {
            // 三个锚点：负区取 −1、零点、正区取 1。
            let mut out = [0.0; 3];
            assert!(
                forward(tensor::View::one(&[-3.0, 0.0, 2.0]), &mut out).is_ok(),
                "等长"
            );
            assert_eq!(out[0], -1.0, "负区符号");
            assert_eq!(out[1], 0.0, "零点");
            assert_eq!(out[2], 1.0, "正区符号");
        },
        |x| if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        },
        false
    );
}

pub mod recip_f32 {
    //! 逐元素倒数：`y[n] = 1 / a[n]`；零输入按 IEEE 得无穷（样本夹具避开 0）。

    crate::unary_f32!(
        Recip,
        RecipF32,
        "recip_f32",
        unary_pair_tests,
        {
            // 三个锚点：整数倒数、分数倒数；不含 0，避免零除语义分歧。
            let mut out = [0.0; 3];
            assert!(
                forward(tensor::View::one(&[2.0, 4.0, 0.5]), &mut out).is_ok(),
                "等长"
            );
            assert_eq!(out[0], 0.5, "1/2");
            assert_eq!(out[1], 0.25, "1/4");
            assert_eq!(out[2], 2.0, "1/0.5");
        },
        |x| 1.0 / x,
        false
    );
}

pub mod cos_f32 {
    //! 逐元素余弦：`y[n] = cos(a[n])`。

    crate::unary_f32!(
        Cos,
        CosF32,
        "cos_f32",
        unary_pair_tests,
        {
            // 锚点：cos(0)=1、cos(π/2)≈0。
            let mut out = [0.0; 2];
            assert!(
                forward(
                    tensor::View::one(&[0.0, std::f32::consts::FRAC_PI_2]),
                    &mut out
                )
                .is_ok(),
                "等长"
            );
            assert_eq!(out[0], 1.0, "cos(0)");
            assert!(out[1].abs() < 1e-6, "cos(π/2)≈0");
        },
        |x| x.cos(),
        false
    );
}

pub mod sin_f32 {
    //! 逐元素正弦：`y[n] = sin(a[n])`。

    crate::unary_f32!(
        Sin,
        SinF32,
        "sin_f32",
        unary_pair_tests,
        {
            // 锚点：sin(0)=0、sin(π/2)=1。
            let mut out = [0.0; 2];
            assert!(
                forward(
                    tensor::View::one(&[0.0, std::f32::consts::FRAC_PI_2]),
                    &mut out
                )
                .is_ok(),
                "等长"
            );
            assert_eq!(out[0], 0.0, "sin(0)");
            assert!((out[1] - 1.0).abs() < 1e-6, "sin(π/2)=1");
        },
        |x| x.sin(),
        false
    );
}

pub mod sqrt_f32 {
    //! 逐元素平方根：`y[n] = √a[n]`；负入按 IEEE 得 NaN，样本夹具取非负。

    crate::unary_f32!(
        Sqrt,
        SqrtF32,
        "sqrt_f32",
        unary_pair_tests,
        {
            // 锚点：√0=0、√4=2。
            let mut out = [0.0; 2];
            assert!(
                forward(tensor::View::one(&[0.0, 4.0]), &mut out).is_ok(),
                "等长"
            );
            assert_eq!(out[0], 0.0, "√0");
            assert_eq!(out[1], 2.0, "√4");
        },
        |x| x.sqrt(),
        false,
        |len| (0..len).map(|i| (i as f32) * 0.25).collect()
    );
}

pub mod rsqrt_f32 {
    //! 逐元素倒数平方根：`y[n] = 1/√a[n]`；零/负入奇异，样本夹具取正且非零。

    crate::unary_f32!(
        Rsqrt,
        RsqrtF32,
        "rsqrt_f32",
        unary_pair_tests,
        {
            // 锚点：1/√1=1、1/√4=0.5；不含 0，避开零除奇异。
            let mut out = [0.0; 2];
            assert!(
                forward(tensor::View::one(&[1.0, 4.0]), &mut out).is_ok(),
                "等长"
            );
            assert_eq!(out[0], 1.0, "1/√1");
            assert_eq!(out[1], 0.5, "1/√4");
        },
        |x| 1.0 / x.sqrt(),
        false,
        |len| (0..len).map(|i| (i + 1) as f32 * 0.25).collect()
    );
}

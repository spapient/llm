use super::*;

/// `loss = reduce_sum(mul(p, w))`：d loss/d p[i] = w[i]，乘法与归约两段链式回传。
#[test]
fn mul_then_reduce_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let w = tape.leaf();
    let z = tape.record("mul_f32", &[p, w]).expect("记录 z");
    let loss = tape.record("reduce_sum_f32", &[z]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(p, N), (w, N), (z, N), (loss, 1)]);

    let p0 = vec![0.3_f32, -0.5, 0.8];
    let w0 = vec![1.0_f32, -0.2, 0.6];

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.bufs().load(w, Data::F32(w0.clone()))?;
    session.bufs().compute_step("mul_f32", &[p, w], z)?;
    sum_backward(&mut session, z, loss)?;

    check_grad(
        session.bufs().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        |x| dot(x, &w0),
    );
    Ok(())
}

/// `loss = reduce_sum(silu(scale(p)))`：缩放与 SiLU 激活两段链式对拍。
#[test]
fn scale_silu_chain_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("scale_f32", &[p]).expect("记录 h");
    let y = tape.record("silu_f32", &[h]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(p, N), (h, N), (y, N), (loss, 1)]);

    let p0 = vec![-0.6_f32, 0.4, 1.2];
    let chain_loss = |params: &[f32]| params.iter().map(|&x| silu(0.5 * x)).sum::<f32>();

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.bufs().compute_step("scale_f32", &[p], h)?;
    session.bufs().compute_step("silu_f32", &[h], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.bufs().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        chain_loss,
    );
    Ok(())
}

/// 图外常量叶子不收梯度：add(p, x) 中 x 是 detached 叶子，梯度槽保持空；参数 p 收到上游 1。
#[test]
fn detached_leaf_receives_no_gradient() -> Result<()> {
    let (tape, sizes, p, x, loss) = support::add_graph(N);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
    session.bufs().load(x, Data::F32(vec![0.1; N]))?;
    session.bufs().compute_step("add_f32", &[p, x], loss)?;
    session.plan(loss)?;
    session.backward()?;

    assert!(
        session.bufs().gradient(x).is_none(),
        "图外叶子 x 不该有梯度槽"
    );
    assert_eq!(
        session.bufs().gradient(p),
        Some(&[1.0; N][..]),
        "add 把上游 1 原样传给参数 p"
    );
    Ok(())
}

/// `loss = reduce_sum(gelu(p))`：GELU 导数（正态密度 + 乘 Φ）逐元素对拍。
#[test]
fn gelu_chain_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("gelu_f32", &[p]).expect("记录 h");
    let loss = tape.record("reduce_sum_f32", &[h]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(p, N), (h, N), (loss, 1)]);

    let p0 = vec![-1.2_f32, 0.4, 1.1];
    let chain_loss = |params: &[f32]| {
        params
            .iter()
            .map(|&x| 0.5 * x * (1.0 + nn::gelu_f32::erf(x * std::f32::consts::FRAC_1_SQRT_2)))
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.bufs().compute_step("gelu_f32", &[p], h)?;
    sum_backward(&mut session, h, loss)?;

    check_grad(
        session.bufs().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        chain_loss,
    );
    Ok(())
}

/// `loss = reduce_max(p)`：最大值元素收到上游 1，其余为 0；中心差分核对梯度口径。
#[test]
fn reduce_max_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("reduce_max_f32", &[p]).expect("记录 h");
    let loss = tape.record("reduce_sum_f32", &[h]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(p, N), (h, 1), (loss, 1)]);

    let p0 = vec![-0.7_f32, 1.3, 0.2];
    let chain_loss = |params: &[f32]| params.iter().copied().fold(f32::MIN, f32::max);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.bufs().compute_step("reduce_max_f32", &[p], h)?;
    sum_backward(&mut session, h, loss)?;

    check_grad(
        session.bufs().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        chain_loss,
    );
    Ok(())
}

/// `loss = reduce_sum(gemv(x, w))`：权重 `dW[n,k]=x[k]·in[n]`，输入 `dx[k]=Σ_n W[n,k]·in[n]`。
#[test]
fn gemv_matches_central_differences() -> Result<()> {
    let rows = 2usize;
    let cols = 3usize;
    let mut tape = Tape::new();
    let w = tape.parameter();
    let x = tape.leaf();
    // gemv 反向按输入宽度从权重长度反推行数：权重在前、输入在后与前向调用对齐。
    let y = tape.record("gemv_f32", &[w, x]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(w, rows * cols), (x, cols), (y, rows), (loss, 1)]);

    let w0: Vec<f32> = (0..rows * cols).map(|i| 0.1 * i as f32 - 0.5).collect();
    let x0 = vec![0.3_f32, -0.2, 0.8];
    let chain_loss = |w: &[f32], x: &[f32]| {
        (0..rows)
            .map(|n| (0..cols).map(|k| w[n * cols + k] * x[k]).sum::<f32>())
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(w, ParamRole::Trainable, w0.clone())?;
    session.bufs().load(x, Data::F32(x0.clone()))?;
    session.bufs().compute_step("gemv_f32", &[w, x], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.bufs().gradient(w).expect("权重梯度应在场"),
        &w0,
        "w",
        |w| chain_loss(w, &x0),
    );
    Ok(())
}

/// `loss = reduce_sum(gemm(a, b))`：收缩轴固定 `nn::gemm::K`，对拍 `dA` / `dB`。
#[test]
fn gemm_matches_central_differences() -> Result<()> {
    let k = nn::gemm::K;
    let m = 1usize;
    let n = 2usize;
    let mut tape = Tape::new();
    let a = tape.parameter();
    let b = tape.leaf();
    let y = tape.record("gemm_f32", &[a, b]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(a, m * k), (b, k * n), (y, m * n), (loss, 1)]);

    let a0: Vec<f32> = (0..m * k).map(|i| 0.1 * i as f32 - 0.3).collect();
    let b0: Vec<f32> = (0..k * n).map(|i| 0.08 * i as f32 - 0.4).collect();
    let chain_loss = |a: &[f32], b: &[f32]| {
        (0..m)
            .flat_map(|i| {
                (0..n).map(move |j| (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum::<f32>())
            })
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(a, ParamRole::Trainable, a0.clone())?;
    session.bufs().load(b, Data::F32(b0.clone()))?;
    session.bufs().compute_step("gemm_f32", &[a, b], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.bufs().gradient(a).expect("A 梯度应在场"),
        &a0,
        "a",
        |a| chain_loss(a, &b0),
    );
    Ok(())
}

/// `loss = reduce_sum(embed(idx, table))`：被访问行累加上游梯度；表行不访问则保持 0。
#[test]
fn embed_table_gradient_matches_central_differences() -> Result<()> {
    let tokens = 2usize;
    let dim = 2usize;
    let rows = 3usize;
    let mut tape = Tape::new();
    let table = tape.parameter();
    let idx = tape.leaf();
    let y = tape.record("embed_f32", &[idx, table]).expect("记录 y");
    let loss = tape.record("reduce_sum_f32", &[y]).expect("记录 loss");

    let sizes = support::sizes_of(
        &tape,
        &[
            (table, rows * dim),
            (idx, tokens),
            (y, tokens * dim),
            (loss, 1),
        ],
    );

    let table0: Vec<f32> = (0..rows * dim).map(|i| 0.2 * i as f32 - 0.6).collect();
    let idx0 = vec![0.0_f32, 2.0];
    let chain_loss = |table: &[f32]| {
        (0..tokens)
            .flat_map(|t| {
                let row = idx0[t] as usize * dim;
                (0..dim).map(move |d| table[row + d])
            })
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(table, ParamRole::Trainable, table0.clone())?;
    session.bufs().load(idx, Data::F32(idx0.clone()))?;
    session.bufs().compute_step("embed_f32", &[idx, table], y)?;
    sum_backward(&mut session, y, loss)?;

    check_grad(
        session.bufs().gradient(table).expect("表梯度应在场"),
        &table0,
        "table",
        chain_loss,
    );
    Ok(())
}

/// 不可微算子在回传时显式报 [`Code::Unsupported`]，不静默忽略。反量化是推理期解码，
/// 权重/码值无训练梯度，因此不登记反向公式。
#[test]
fn unsupported_operator_backward_is_rejected() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let y = tape.record("dequant_i8", &[p]).expect("记录 y");

    let sizes = support::sizes_of(&tape, &[(p, N), (y, N)]);

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, vec![1.0; N])?;
    session.bufs().compute_step("dequant_i8", &[p], y)?;
    session.plan(y)?;

    let error = session.backward().expect_err("反量化没有解析反向");
    assert_eq!(error.code(), Code::Unsupported);
    Ok(())
}

/// 端到端蒸馏 KL：`kl = sum(p_s · (ln p_s - ln p_t))` 用图上原语搭出来
/// （softmax→log→sub→mul→reduce_sum），反传后与解析单行 KL 中心差分对拍。
///
/// 这是「损失层从纯函数升级为图上可训练」的闭环证据：五个算子串成一条链，
/// 整链梯度等于解析梯度，而不只是各算子单独可微。
#[test]
fn distillation_kl_graph_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let s = tape.parameter();
    let t = tape.leaf();
    let ps = tape.record("softmax_f32", &[s]).expect("p_s");
    let pt = tape.record("softmax_f32", &[t]).expect("p_t");
    let lps = tape.record("log_f32", &[ps]).expect("ln p_s");
    let lpt = tape.record("log_f32", &[pt]).expect("ln p_t");
    let diff = tape
        .record("sub_f32", &[lps, lpt])
        .expect("ln p_s - ln p_t");
    let prod = tape
        .record("mul_f32", &[ps, diff])
        .expect("p_s·(ln p_s - ln p_t)");
    let kl = tape.record("reduce_sum_f32", &[prod]).expect("kl");

    let mut sizes = Sizes::new(tape.vars());
    for v in [s, t, ps, pt, lps, lpt, diff, prod] {
        sizes.set(v, Shape::new([N]), DType::F32);
    }
    sizes.set(kl, Shape::new([1]), DType::F32);

    let s0 = vec![0.5_f32, -0.3, 0.2];
    let t0 = vec![1.0_f32, 0.1, -0.5];

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(s, ParamRole::Trainable, s0.clone())?;
    session.bufs().load(t, Data::F32(t0.clone()))?;
    session.bufs().compute_step("softmax_f32", &[s], ps)?;
    session.bufs().compute_step("softmax_f32", &[t], pt)?;
    session.bufs().compute_step("log_f32", &[ps], lps)?;
    session.bufs().compute_step("log_f32", &[pt], lpt)?;
    session.bufs().compute_step("sub_f32", &[lps, lpt], diff)?;
    session.bufs().compute_step("mul_f32", &[ps, diff], prod)?;
    session.bufs().compute_step("reduce_sum_f32", &[prod], kl)?;
    session.plan(kl)?;
    session.backward()?;

    check_grad(
        session.bufs().gradient(s).expect("s 梯度应在场"),
        &s0,
        "s",
        |s| kl_single_row(s, &t0),
    );
    Ok(())
}

/// 端到端 GRPO 裁剪：`Σ min(ratio·A, clamp(ratio,1±ε)·A)` 用图上原语搭出来
/// （sub→exp→max→min→mul→mul→min→reduce_sum），反传后与解析裁剪代理中心差分对拍。
///
/// 夹具三个元素各走一路：裁上界（ratio>1+ε, A>0）、裁下界（ratio<1−ε, A<0）、不裁
/// （ratio 在界内）。裁断处梯度为 0（clamp 平坦），不裁处梯度沿 ratio 回传。
#[test]
fn grpo_clipped_surrogate_graph_matches_central_differences() -> Result<()> {
    let eps: f32 = 0.2;
    let lo: f32 = 1.0 - eps;
    let hi: f32 = 1.0 + eps;

    let mut tape = Tape::new();
    let new = tape.parameter();
    let old = tape.leaf();
    let adv = tape.leaf();
    let lo_leaf = tape.leaf();
    let hi_leaf = tape.leaf();
    let diff = tape.record("sub_f32", &[new, old]).expect("new - old");
    let ratio = tape.record("exp_f32", &[diff]).expect("ratio");
    let r_lo = tape
        .record("max_f32", &[ratio, lo_leaf])
        .expect("max(ratio, 1-ε)");
    let r_clamp = tape
        .record("min_f32", &[r_lo, hi_leaf])
        .expect("clamp(ratio)");
    let term1 = tape.record("mul_f32", &[ratio, adv]).expect("ratio·A");
    let term2 = tape
        .record("mul_f32", &[r_clamp, adv])
        .expect("clamp(ratio)·A");
    let surr = tape.record("min_f32", &[term1, term2]).expect("min 裁剪");
    let loss = tape.record("reduce_sum_f32", &[surr]).expect("loss");

    let mut sizes = Sizes::new(tape.vars());
    for v in [
        new, old, adv, lo_leaf, hi_leaf, diff, ratio, r_lo, r_clamp, term1, term2, surr,
    ] {
        sizes.set(v, Shape::new([N]), DType::F32);
    }
    sizes.set(loss, Shape::new([1]), DType::F32);

    let new0 = vec![0.4_f32, -0.5, 0.0];
    let old0 = vec![0.0_f32; N];
    let adv0 = vec![1.0_f32, -0.5, 0.7];

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(new, ParamRole::Trainable, new0.clone())?;
    session.bufs().load(old, Data::F32(old0.clone()))?;
    session.bufs().load(adv, Data::F32(adv0.clone()))?;
    session.bufs().load(lo_leaf, Data::F32(vec![lo; N]))?;
    session.bufs().load(hi_leaf, Data::F32(vec![hi; N]))?;
    session.bufs().compute_step("sub_f32", &[new, old], diff)?;
    session.bufs().compute_step("exp_f32", &[diff], ratio)?;
    session
        .bufs()
        .compute_step("max_f32", &[ratio, lo_leaf], r_lo)?;
    session
        .bufs()
        .compute_step("min_f32", &[r_lo, hi_leaf], r_clamp)?;
    session
        .bufs()
        .compute_step("mul_f32", &[ratio, adv], term1)?;
    session
        .bufs()
        .compute_step("mul_f32", &[r_clamp, adv], term2)?;
    session
        .bufs()
        .compute_step("min_f32", &[term1, term2], surr)?;
    session
        .bufs()
        .compute_step("reduce_sum_f32", &[surr], loss)?;
    session.plan(loss)?;
    session.backward()?;

    check_grad(
        session.bufs().gradient(new).expect("new 梯度应在场"),
        &new0,
        "new",
        |new| clipped_surrogate_sum(new, &old0, &adv0, eps),
    );
    Ok(())
}

/// `loss = reduce_sum(leaky_relu(p))`：正区梯度 1、负区梯度 slope，逐元素对拍。
#[test]
fn leaky_relu_chain_matches_central_differences() -> Result<()> {
    let mut tape = Tape::new();
    let p = tape.parameter();
    let h = tape.record("leaky_relu_f32", &[p]).expect("记录 h");
    let loss = tape.record("reduce_sum_f32", &[h]).expect("记录 loss");

    let sizes = support::sizes_of(&tape, &[(p, N), (h, N), (loss, 1)]);
    let slope = nn::leaky_relu_f32::SLOPE;
    let p0 = vec![-1.2_f32, 0.4, 1.1];
    let chain_loss = |params: &[f32]| {
        params
            .iter()
            .map(|&x| if x >= 0.0 { x } else { slope * x })
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.bufs().compute_step("leaky_relu_f32", &[p], h)?;
    sum_backward(&mut session, h, loss)?;

    check_grad(
        session.bufs().gradient(p).expect("参数梯度应在场"),
        &p0,
        "p",
        chain_loss,
    );
    Ok(())
}

/// `loss = reduce_sum(hard_sigmoid(p))`：线性区梯度恒为 1/6；饱和点解析梯度为 0。
#[test]
fn hard_sigmoid_chain_matches_central_differences() -> Result<()> {
    let chain_loss = |params: &[f32]| {
        params
            .iter()
            .map(|&x| (x / 6.0 + 0.5).clamp(0.0, 1.0))
            .sum::<f32>()
    };
    check_unary_grad("hard_sigmoid_f32", &[-1.5, 0.4, 1.1], chain_loss)
}

/// 饱和点（|x| > 3）的解析梯度必须为 0，且不是差分近似能测得的小数。
#[test]
fn hard_sigmoid_is_flat_outside_the_linear_band() -> Result<()> {
    let p0 = [-4.0_f32, 0.0, 4.0];
    let grad = unary_grad("hard_sigmoid_f32", &p0, p0.len())?;
    assert_eq!(grad[0], 0.0, "下饱和梯度");
    assert!((grad[1] - 1.0 / 6.0).abs() < 1e-6, "中心梯度 {}", grad[1]);
    assert_eq!(grad[2], 0.0, "上饱和梯度");
    Ok(())
}

/// `loss = reduce_sum(σ(p))`：闭式损失就是 σ 逐元素和，对拍 `σ·(1−σ)`。
#[test]
fn sigmoid_chain_matches_central_differences() -> Result<()> {
    let chain_loss = |params: &[f32]| {
        params
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .sum::<f32>()
    };
    check_unary_grad("sigmoid_f32", &[-1.3, 0.0, 0.9], chain_loss)
}

/// `loss = reduce_sum(tanh(p))`：对拍 `1 − tanh²`，负区点同时盯住导函数的偶对称。
#[test]
fn tanh_chain_matches_central_differences() -> Result<()> {
    let chain_loss = |params: &[f32]| params.iter().map(|&x| x.tanh()).sum::<f32>();
    check_unary_grad("tanh_f32", &[-1.3, 0.0, 0.9], chain_loss)
}

/// `loss = reduce_sum(relu(p))`：负区梯度必须**恰好** 0（不是 LeakyReLU 的 0.01），正区恰好 1。
#[test]
fn relu_chain_matches_central_differences() -> Result<()> {
    let chain_loss = |params: &[f32]| params.iter().map(|&x| x.max(0.0)).sum::<f32>();
    check_unary_grad("relu_f32", &[-1.3, 0.45, 0.9], chain_loss)
}

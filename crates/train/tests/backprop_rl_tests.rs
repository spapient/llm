use super::*;

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
    session.bufs().compute("softmax_f32", &[s], ps)?;
    session.bufs().compute("softmax_f32", &[t], pt)?;
    session.bufs().compute("log_f32", &[ps], lps)?;
    session.bufs().compute("log_f32", &[pt], lpt)?;
    session.bufs().compute("sub_f32", &[lps, lpt], diff)?;
    session.bufs().compute("mul_f32", &[ps, diff], prod)?;
    session.bufs().compute("reduce_sum_f32", &[prod], kl)?;
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
    session.bufs().compute("sub_f32", &[new, old], diff)?;
    session.bufs().compute("exp_f32", &[diff], ratio)?;
    session
        .bufs()
        .compute("max_f32", &[ratio, lo_leaf], r_lo)?;
    session
        .bufs()
        .compute("min_f32", &[r_lo, hi_leaf], r_clamp)?;
    session.bufs().compute("mul_f32", &[ratio, adv], term1)?;
    session
        .bufs()
        .compute("mul_f32", &[r_clamp, adv], term2)?;
    session
        .bufs()
        .compute("min_f32", &[term1, term2], surr)?;
    session.bufs().compute("reduce_sum_f32", &[surr], loss)?;
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
    let slope = op::leaky_relu_f32::SLOPE;
    let p0 = vec![-1.2_f32, 0.4, 1.1];
    let chain_loss = |params: &[f32]| {
        params
            .iter()
            .map(|&x| if x >= 0.0 { x } else { slope * x })
            .sum::<f32>()
    };

    let mut session = Session::new(tape, sizes, Reference, Sgd::new(), config())?;
    session.register_parameter(p, ParamRole::Trainable, p0.clone())?;
    session.bufs().compute("leaky_relu_f32", &[p], h)?;
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

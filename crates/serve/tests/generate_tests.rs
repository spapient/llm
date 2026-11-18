use super::*;
use tokenizer::Tokenizer;

#[test]
fn greedy_takes_the_largest_logit() {
    let mut greedy = Greedy;
    assert_eq!(greedy.sample(&[0.1, 0.9, 0.5]).expect("可采"), Some(1));
    assert_eq!(greedy.sample(&[-3.0, -1.0, -2.0]).expect("可采"), Some(1));
}

/// 等值时取编号最小的那个：换一台机器、换一次运行都必须得到同一个 token。
#[test]
fn ties_go_to_the_lowest_index() {
    let mut greedy = Greedy;
    assert_eq!(greedy.sample(&[2.0, 2.0, 2.0]).expect("可采"), Some(0));
    assert_eq!(greedy.sample(&[0.0, 5.0, 5.0]).expect("可采"), Some(1));
}

/// 空 logits 与非有限值都必须报错：猜一个 token 会让"模型坏了"表现成"模型挺会说话"。
#[test]
fn unusable_logits_are_refused() {
    let mut greedy = Greedy;
    assert_eq!(greedy.sample(&[]).unwrap_err().code(), Code::Invalid);
    assert_eq!(
        greedy.sample(&[1.0, f32::NAN]).unwrap_err().code(),
        Code::Invalid
    );
    assert_eq!(
        greedy.sample(&[f32::INFINITY, 1.0]).unwrap_err().code(),
        Code::Invalid
    );
}

#[test]
fn a_generation_frames_its_tokens_in_little_endian_words() {
    let generation = Generation::new(vec![1, 258, 0x0102_0304], Ending::Limit);
    assert_eq!(generation.ending(), Ending::Limit);
    assert_eq!(
        generation.frame(),
        [
            1u32.to_le_bytes(),
            258u32.to_le_bytes(),
            0x0102_0304u32.to_le_bytes()
        ]
        .concat()
    );
}

/// 生成的 token 能用同一份词表还原成文本：这是"服务真的返回了可读内容"的最低判据。
#[test]
fn a_generation_decodes_back_to_text() {
    let tokenizer = tokenizer::Bytes;
    let tokens = tokenizer.encode("hé").expect("编码");
    let ids: Vec<u32> = tokens.iter().map(|token| token.id()).collect();
    let generation = Generation::new(ids, Ending::Stop);
    assert_eq!(generation.tokens().len(), 3);
    assert_eq!(generation.text(&tokenizer).expect("解码"), "hé");
}

#[test]
fn ending_names_are_stable() {
    let names: Vec<&str> = Ending::ALL.iter().map(|ending| ending.as_str()).collect();
    assert_eq!(names, ["limit", "stop"]);
}

/// 不启用任何截断的经典采样与贪心同口径：同一批 logits 给出同一个 token。
#[test]
fn classical_without_filters_matches_greedy() {
    let mut classical = Classical::greedy_like();
    for logits in [
        [0.1, 0.9, 0.5].as_slice(),
        &[-3.0, -1.0, -2.0],
        &[2.0, 2.0, 2.0],
    ] {
        let greedy = Greedy.sample(logits).expect("贪心可采");
        assert_eq!(classical.sample(logits).expect("经典可采"), greedy);
    }
}

/// 温度只缩放、不改写序：任意正温度下 argmax 赢家不变（确定性口径）。
#[test]
fn temperature_does_not_change_the_argmax_winner() {
    let logits = [1.0f32, 0.9, -0.5];
    for temperature in [0.5, 1.0, 2.0, 7.0] {
        let mut classical = Classical::new(SamplingParams::new().with_temperature(temperature));
        assert_eq!(classical.sample(&logits).expect("可采"), Some(0));
    }
}

/// 温度必须是大于 0 的有限值：0、负数、NaN、无穷都报 `Invalid`。
#[test]
fn a_bad_temperature_is_rejected() {
    for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut classical = Classical::new(SamplingParams::new().with_temperature(temperature));
        assert_eq!(
            classical.sample(&[1.0, 2.0]).unwrap_err().code(),
            Code::Invalid,
            "温度 {temperature} 必须被拒绝"
        );
    }
}

/// top-k 必须落在 `1..=候选条数`：0 与超出词表都报 `Invalid`。
#[test]
fn top_k_must_be_within_the_candidates() {
    let logits = [1.0f32, 2.0, 3.0];
    for top_k in [0, 4] {
        let mut classical = Classical::new(SamplingParams::new().with_top_k(top_k));
        assert_eq!(
            classical.sample(&logits).unwrap_err().code(),
            Code::Invalid,
            "top-k {top_k} 必须被拒绝"
        );
    }
    let mut classical = Classical::new(SamplingParams::new().with_top_k(2));
    assert_eq!(classical.sample(&logits).expect("top-k=2 可采"), Some(2));
}

/// top-p 必须在 `(0, 1]` 内：0、负数、大于 1、NaN 都报 `Invalid`。
#[test]
fn top_p_must_be_in_the_unit_interval() {
    for top_p in [0.0, -0.2, 1.5, f32::NAN] {
        let mut classical = Classical::new(SamplingParams::new().with_top_p(top_p));
        assert_eq!(
            classical.sample(&[1.0, 2.0]).unwrap_err().code(),
            Code::Invalid,
            "top-p {top_p} 必须被拒绝"
        );
    }
    let mut classical = Classical::new(SamplingParams::new().with_top_p(1.0));
    assert_eq!(
        classical.sample(&[1.0, 2.0]).expect("top-p=1 可采"),
        Some(1)
    );
}

/// 空 logits 与非有限值在经典采样下同样预检拦截，不猜 token。
#[test]
fn classical_rejects_unusable_logits() {
    let mut classical = Classical::greedy_like();
    assert_eq!(classical.sample(&[]).unwrap_err().code(), Code::Invalid);
    assert_eq!(
        classical.sample(&[1.0, f32::NAN]).unwrap_err().code(),
        Code::Invalid
    );
}

/// top-p 核采样按概率降序保留前缀：小 p 只留头名，p 放宽到覆盖第二名字才入集。
#[test]
fn nucleus_keeps_the_probability_prefix() {
    let scaled = [3.0f32, 2.0, 1.0];
    let candidates = vec![0usize, 1, 2];
    let tight = super::nucleus(&scaled, candidates.clone(), 0.5);
    assert_eq!(tight, [0]);
    let wide = super::nucleus(&scaled, candidates, 0.9);
    assert_eq!(wide, [0, 1]);
}

/// 同种子、同 logits 序列逐位可复现：两个独立实例跑相同步数得到相同 token 序列。
#[test]
fn same_seed_reproduces_the_same_sequence() {
    let logits = [1.0f32, 0.0, -2.0, 0.5];
    let mut left = Multinomial::new(SamplingParams::new(), 42);
    let mut right = Multinomial::new(SamplingParams::new(), 42);
    for _ in 0..16 {
        assert_eq!(
            left.sample(&logits).expect("可采"),
            right.sample(&logits).expect("可采")
        );
    }
}

/// 头号 logit 远高于其余候选时，采样几乎总是落到它身上（softmax 概率压倒性）。
#[test]
fn a_dominant_logit_is_picked_nearly_always() {
    let logits = [12.0f32, 0.0, -12.0];
    let mut sampler = Multinomial::new(SamplingParams::new(), 7);
    for _ in 0..10 {
        assert_eq!(sampler.sample(&logits).expect("可采"), Some(0));
    }
}

/// top-k=1 只保留头名候选，随机抽取不再改选中对象：与确定性 argmax 同结果。
#[test]
fn top_k_one_keeps_only_the_argmax() {
    let logits = [1.0f32, 3.0, 2.0];
    let mut sampler = Multinomial::new(SamplingParams::new().with_top_k(1), 1);
    for _ in 0..5 {
        assert_eq!(sampler.sample(&logits).expect("可采"), Some(1));
    }
}

/// 空 logits 与非有限值在多项式采样下同样预检拦截，不猜 token。
#[test]
fn multinomial_rejects_unusable_logits() {
    let mut sampler = Multinomial::new(SamplingParams::new(), 1);
    assert_eq!(sampler.sample(&[]).unwrap_err().code(), Code::Invalid);
    assert_eq!(
        sampler.sample(&[1.0, f32::NAN]).unwrap_err().code(),
        Code::Invalid
    );
}

/// 温度档位的参数校验与经典采样共用同一份档案：坏温度在采样时拦截。
#[test]
fn multinomial_rejects_bad_temperature() {
    let mut sampler = Multinomial::new(SamplingParams::new().with_temperature(0.0), 1);
    assert_eq!(
        sampler.sample(&[1.0, 2.0]).unwrap_err().code(),
        Code::Invalid
    );
}

/// 每一步抽到的 token 编号都落在词表范围内，不会越界返回。
#[test]
fn every_picked_token_is_within_the_vocabulary() {
    let logits = [0.2f32, -1.1, 0.9, -0.4, 0.0];
    let mut sampler = Multinomial::new(SamplingParams::new(), 99);
    for _ in 0..20 {
        let token = sampler.sample(&logits).expect("可采").expect("有候选");
        assert!(token < logits.len() as u32);
    }
}

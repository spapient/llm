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

use super::*;

/// 默认 `logits_batch` 逐条回退：行序与输入序列一一对应，空批返回空，不吞调用次序。
#[test]
fn logits_batch_default_runs_each_sequence_in_input_order() {
    let pager = Pager::new(Cache::new(2, 16).expect("页池"), 2).expect("行宽");
    let mut batch = Batch::new(pager);
    let first = Sequence::new(7, &[2u32], 2, 16, Stream::new(0)).expect("序列");
    let second = Sequence::new(3, &[2u32], 2, 16, Stream::new(0)).expect("序列");
    batch.admit(first).expect("准入");
    batch.admit(second).expect("准入");
    let sequences: Vec<Sequence> = batch.sequences().to_vec();

    let mut model = OrderProbe { calls: Vec::new() };
    let rows = model
        .logits_batch(&sequences, &mut batch)
        .expect("批量回退");

    assert_eq!(model.calls, [7, 3], "默认回退必须按准入顺序逐条调用");
    assert_eq!(rows.len(), 2, "返回行数与输入条数一致");
    assert_eq!(rows[0], [7.0]);
    assert_eq!(rows[1], [3.0]);
    assert!(
        model
            .logits_batch(&[], &mut batch)
            .expect("空批")
            .is_empty(),
        "空序列批量返回空行"
    );
}

// ===== 黑盒协议契约（原 `tests/protocol.rs`，随本模块走 libtest） =====
//
// 端点与线格式从外部看只有"字节进、响应出"。内联测试直接驱动 `frame::scan` 与内部辅助；
// 这一段只走公开面：`Envelope` 编解码往返、`Decoder` 的分批到达语义、`Route` 与 `Response`
// 的对外承诺，以及错误流的方向（服务端自身故障才上抛，请求不成立必须落成响应）。

/// 三个类型各造一帧，载荷长度互不相同：短、空、边界。
fn envelopes() -> Vec<Envelope> {
    vec![
        Envelope::new(
            Version::current(),
            Kind::Request,
            b"POST /v1/completions".to_vec(),
        ),
        Envelope::new(Version::current(), Kind::Chunk, Vec::new()),
        Envelope::new(Version::current(), Kind::Done, b"ok".to_vec()),
    ]
}

#[test]
fn every_envelope_round_trips_through_wire_bytes() {
    for envelope in envelopes() {
        let frame = envelope.frame().expect("帧合法");
        assert_eq!(
            frame.len(),
            HEADER_LEN + envelope.payload().len() + TRAILER_LEN,
            "帧长由布局决定，不是实现自由"
        );
        let parsed = Envelope::parse(&frame).expect("自产帧必须能解析");
        assert_eq!(parsed, envelope, "版本、类型、载荷三者都要原样回来");
    }
}

#[test]
fn a_stream_split_across_pushes_yields_the_same_frames() {
    let wire: Vec<u8> = envelopes()
        .iter()
        .flat_map(|envelope| envelope.frame().expect("编码"))
        .collect();

    // 每 3 个字节一段：一帧会被切成很多段，覆盖"半帧到达"。
    let mut decoder = Decoder::new();
    let mut kinds = Vec::new();
    for chunk in wire.chunks(3) {
        decoder.push(chunk).expect("半帧不是错误");
        while let Some(frame) = decoder.next_frame().expect("扫描") {
            kinds.push(frame.kind());
            assert_eq!(frame.version(), Version::current());
        }
    }
    assert_eq!(kinds, [Kind::Request, Kind::Chunk, Kind::Done]);
    assert_eq!(decoder.buffered(), 0, "取尽之后缓冲必须清空");
    decoder.finish().expect("停在帧边界是正常关闭");
}

#[test]
fn a_connection_closed_mid_frame_is_an_error() {
    let frame = envelopes()[0].frame().expect("编码");
    let mut decoder = Decoder::new();
    decoder.push(&frame[..frame.len() - 2]).expect("半帧可暂存");
    let error = decoder.finish().expect_err("半帧断开不能算正常结束");
    assert_eq!(error.code().as_str(), "invalid");
    assert!(
        !error.message().trim().is_empty(),
        "拒绝必须说清还剩多少字节"
    );
}

#[test]
fn a_declared_length_over_the_limit_is_rejected_before_allocating() {
    // 只给头部：魔数、版本、类型、超限长度。接收方必须用这 10 个字节就拒绝。
    let mut header = Vec::new();
    header.extend_from_slice(&crate::proto::MAGIC.to_le_bytes());
    header.extend_from_slice(&Version::current().get().to_le_bytes());
    header.extend_from_slice(&1u16.to_le_bytes());
    header.extend_from_slice(&(MAX_PAYLOAD + 1).to_le_bytes());
    let mut decoder = Decoder::new();
    let error = decoder.push(&header).expect_err("超限长度必须立刻拒绝");
    assert_eq!(error.code().as_str(), "invalid");
    assert!(
        decoder.next_frame().is_err(),
        "同一条流一旦判坏，后续每次取帧都要继续报同一个错误"
    );
}

#[test]
fn a_route_rejects_specs_it_cannot_honor() {
    assert!(Route::parse("POST /v1/completions").is_ok());
    for spec in ["", "POST", "GET v1/completions", " /x"] {
        let error = Route::parse(spec).expect_err("不成立的路由必须拒绝");
        assert_eq!(
            error.code().as_str(),
            "invalid",
            "错误类别是对外契约：{spec:?}"
        );
    }
}

#[test]
fn a_response_carries_its_status_into_the_frame() {
    let response = Response::error(Status::BadRequest, "缺 prompt");
    assert_eq!(response.status().code(), 400);
    let frame = response.frame().expect("错误响应也要能编码");
    let envelope = Envelope::parse(&frame).expect("解析");
    assert_eq!(envelope.kind(), Kind::Error, "错误响应的帧类型不能是 Chunk");
    assert_eq!(
        envelope.payload(),
        "缺 prompt".as_bytes(),
        "载荷就是给人看的那句话，不额外包一层"
    );
    assert_eq!(Response::ok(b"ok".to_vec()).status().code(), 200);
}

#[test]
fn admission_answers_with_a_retryable_or_terminal_status() {
    let pager = Pager::new(Cache::new(2, 8).expect("页池"), 2).expect("行宽");
    let mut scheduler = Scheduler::new(pager);
    // 提示 8 + 生成 8 = 16 个 token，需要 2 页，池正好 2 页：进去。
    let fits = Sequence::new(1, &[0; 8], 8, 8, Stream::new(1)).expect("序列");
    let response = crate::admit(&mut scheduler, &fits).expect("准入不崩");
    assert_eq!(response.status().code(), 200);
    assert_eq!(
        response.envelope().payload(),
        fits.id().to_le_bytes(),
        "载荷是序列号，客户端据此后续查询"
    );

    // 同样的 id 再来一次：请求自己不成立，必须落成 400，而不是 `Err`。
    let duplicate = crate::admit(&mut scheduler, &fits).expect("请求错误不是服务端错误");
    assert_eq!(duplicate.status().code(), 400);

    // 池里只剩两条序列的余量：再要一条 3 页的，永远装不下，也是 400。
    let too_big = Sequence::new(2, &[0; 8], 8, 4, Stream::new(2)).expect("序列");
    let response = crate::admit(&mut scheduler, &too_big).expect("准入不崩");
    assert_eq!(response.status().code(), 400);
}

#[test]
fn the_prompt_is_encoded_by_the_tokenizer_not_by_the_endpoint() {
    let tokens = crate::encode_prompt(&Bytes, "hé").expect("编码");
    assert_eq!(tokens.len(), 3, "字节级参考实现每个字节一个 token");
    assert_eq!(Bytes.decode(&tokens).expect("解码"), "hé");
}

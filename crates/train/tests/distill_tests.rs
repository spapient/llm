    use crate::{Distill, distillation_loss, kl_div, kl_per_row, log_softmax};
    use diagnostic::{Code, Result};

    /// log-softmax 的指数每行和为 1（数值稳定的 softmax 归一化）。
    #[test]
    fn log_softmax_exponentiates_to_a_distribution() -> Result<()> {
        let logits = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0; 6];
        log_softmax(&logits, &mut out, 3)?;
        for row in out.as_chunks::<3>().0 {
            let sum: f32 = row.iter().map(|v| v.exp()).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax 每行和应趋近 1，得到 {sum}"
            );
        }
        Ok(())
    }

    /// 大 logits 下不溢出：每行先减最大值，10^5 量级的 logits 也稳定。
    #[test]
    fn log_softmax_is_stable_for_large_logits() -> Result<()> {
        let logits = [0.0, 100_000.0, 0.0];
        let mut out = [0.0; 3];
        log_softmax(&logits, &mut out, 3)?;
        assert!(
            out.iter().all(|v| v.is_finite()),
            "大 logits 不应溢出，得到 {out:?}"
        );
        // 类别 1 顶到最高且高出 10^5，log-softmax(1) 应趋近 0。
        assert!(
            out[1].abs() < 1e-3,
            "最大类 log-prob 应趋近 0，得到 {}",
            out[1]
        );
        Ok(())
    }

    /// 学生与教师同分布时 KL 为 0。
    #[test]
    fn kl_div_is_zero_when_distributions_match() -> Result<()> {
        let logits = [1.0, 2.0, 3.0];
        let kl = kl_div(&logits, &logits, 3, 1.0)?;
        assert!(kl.abs() < 1e-6, "同分布 KL 应为 0，得到 {kl}");
        Ok(())
    }

    /// 学生趋近独热、教师均匀时，KL 趋近 ln(词表宽)。
    #[test]
    fn kl_div_of_a_peaked_student_against_uniform_teacher() -> Result<()> {
        let kl = kl_div(&[10.0, 0.0], &[0.0, 0.0], 2, 1.0)?;
        assert!(kl > 0.0, "KL 应严格为正，得到 {kl}");
        assert!((kl - 2.0f32.ln()).abs() < 1e-2, "KL 应趋近 ln 2，得到 {kl}");
        Ok(())
    }

    /// α=0 时蒸馏损失退化为硬标签交叉熵。
    #[test]
    fn distillation_loss_with_alpha_zero_is_cross_entropy() -> Result<()> {
        let student = [0.0, 10.0, 0.0];
        let teacher = [0.0, 0.0, 0.0];
        let labels = [1.0];
        let loss = distillation_loss(&student, &teacher, &labels, 3, Distill::new(2.0, 0.0))?;
        let ce = crate::cross_entropy(&student, &labels, 3)?;
        assert!(
            (loss - ce).abs() < 1e-6,
            "α=0 应退化为 CE，得到 {loss} vs {ce}"
        );
        Ok(())
    }

    /// α=1 时蒸馏损失退化为纯软蒸馏（T² 缩放 KL）。
    #[test]
    fn distillation_loss_with_alpha_one_is_scaled_kl() -> Result<()> {
        let student = [1.0, 2.0, 3.0];
        let teacher = [0.0, 0.0, 0.0];
        let labels = [0.0];
        let temperature = 3.0;
        let loss = distillation_loss(
            &student,
            &teacher,
            &labels,
            3,
            Distill::new(temperature, 1.0),
        )?;
        let expected = temperature * temperature * kl_div(&student, &teacher, 3, temperature)?;
        assert!(
            (loss - expected).abs() < 1e-5,
            "α=1 应退化为 T²·KL，得到 {loss} vs {expected}"
        );
        Ok(())
    }

    /// 温度非正被拒绝。
    #[test]
    fn rejects_nonpositive_temperature() {
        assert!(matches!(
            kl_div(&[0.0, 0.0], &[0.0, 0.0], 2, 0.0),
            Err(e) if e.code() == Code::Invalid
        ));
    }

    /// α 越界被拒绝。
    #[test]
    fn rejects_out_of_range_alpha() {
        assert!(matches!(
            distillation_loss(&[0.0, 0.0], &[0.0, 0.0], &[0.0], 2, Distill::new(1.0, 2.0)),
            Err(e) if e.code() == Code::Invalid
        ));
    }

    /// 学生与教师 logits 不等长被拒绝。
    #[test]
    fn rejects_length_mismatch() {
        assert!(matches!(
            kl_div(&[0.0, 0.0], &[0.0, 0.0, 0.0], 2, 1.0),
            Err(e) if e.code() == Code::Invalid
        ));
    }

    /// 每行 KL 与均值 KL 一致：同分布每行 KL 为 0。
    #[test]
    fn kl_per_row_is_zero_when_distributions_match() -> Result<()> {
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 2];
        kl_per_row(&logits, &logits, 3, 1.0, &mut out)?;
        assert!(
            out.iter().all(|v| v.abs() < 1e-6),
            "同分布每行 KL 应为 0，得到 {out:?}"
        );
        Ok(())
    }

    /// 每行 KL 的行均值等于 kl_div 的返回值。
    #[test]
    fn kl_per_row_averages_to_kl_div() -> Result<()> {
        let student = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
        let teacher = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut out = [0.0; 2];
        kl_per_row(&student, &teacher, 3, 1.0, &mut out)?;
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        let kl = kl_div(&student, &teacher, 3, 1.0)?;
        assert!(
            (mean - kl).abs() < 1e-5,
            "每行 KL 均值应等于 kl_div，得到 {mean} vs {kl}"
        );
        Ok(())
    }

    /// 每行 KL 的 out 与行数不等长被拒绝。
    #[test]
    fn kl_per_row_rejects_out_length_mismatch() {
        assert!(matches!(
            kl_per_row(&[0.0, 0.0], &[0.0, 0.0], 2, 1.0, &mut [0.0; 3]),
            Err(e) if e.code() == Code::Invalid
        ));
    }

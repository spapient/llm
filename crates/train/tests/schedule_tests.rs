    use super::*;
    use diagnostic::Code;

    #[test]
    fn the_schedule_warms_up_then_decays_to_zero() {
        let schedule = Schedule::new(10, 100).expect("合法");
        assert_eq!(schedule.lr(1.0, 0), 0.1);
        assert_eq!(schedule.lr(1.0, 9), 1.0);
        assert!(
            (schedule.lr(1.0, 55) - 0.5).abs() < 1e-6,
            "warmup 后线性衰减"
        );
        assert_eq!(schedule.lr(1.0, 100), 0.0);
        assert_eq!(schedule.lr(1.0, 1_000), 0.0);
    }

    #[test]
    fn a_schedule_without_warmup_starts_at_the_peak() {
        let schedule = Schedule::new(0, 10).expect("合法");
        assert_eq!(schedule.lr(0.5, 0), 0.5);
        assert!(
            (schedule.lr(0.5, 9) - 0.05).abs() < 1e-6,
            "无 warmup 时直接线性衰减"
        );
    }

    #[test]
    fn an_impossible_schedule_is_rejected() {
        assert_eq!(Schedule::new(0, 0).unwrap_err().code(), Code::Invalid);
        assert_eq!(Schedule::new(11, 10).unwrap_err().code(), Code::Invalid);
        assert_eq!(
            Schedule::with(0, 10, Anneal::Cosine, 1.5)
                .unwrap_err()
                .code(),
            Code::Invalid
        );
    }

    #[test]
    fn cosine_decays_from_the_peak_to_the_floor_ratio() {
        let schedule = Schedule::with(0, 100, Anneal::Cosine, 0.1).expect("合法");
        assert_eq!(schedule.lr(1.0, 0), 1.0, "进度起点在峰值");
        let floor = schedule.lr(1.0, 99);
        assert!((floor - 0.1).abs() < 1e-3, "末段落到下限比例：{floor}");
        assert_eq!(schedule.lr(1.0, 100), 0.0, "超出总步数为 0");
    }

    #[test]
    fn a_constant_schedule_holds_the_peak_after_warmup() {
        let schedule = Schedule::with(10, 100, Anneal::Constant, 0.1).expect("合法");
        assert_eq!(schedule.lr(1.0, 9), 1.0, "warmup 结束到峰值");
        assert_eq!(schedule.lr(1.0, 50), 1.0, "恒定曲线不向下限压");
    }

    #[test]
    fn linear_decay_respects_the_floor_ratio() {
        let schedule = Schedule::with(0, 100, Anneal::Linear, 0.2).expect("合法");
        let floor = schedule.lr(1.0, 99);
        assert!(floor >= 0.2 - 1e-6, "线性谷底不低于下限：{floor}");
        assert!(floor <= 1.0, "学习率不超过峰值：{floor}");
    }

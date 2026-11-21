    use super::*;

    use crate::{Buffers, Data, Reference};
    use autograd::{Sizes, Tape};
    use tensor::{DType, Shape};

    /// 一条 `add_f32`：类型化调用与字符串派发逐元素一致。
    #[test]
    fn a_typed_call_matches_the_string_path() {
        let mut tape = Tape::new();
        let a = tape.parameter();
        let b = tape.leaf();
        let typed = tape.record("add_f32", &[a, b]).expect("记录");
        let raw = tape.record("add_f32", &[a, b]).expect("记录");
        let mut sizes = Sizes::new(tape.vars());
        for var in [a, b, typed, raw] {
            sizes.set(var, Shape::new([4]), DType::F32);
        }
        let mut store = Buffers::new(sizes, Reference);
        store
            .load(a, Data::F32(vec![1.0, 2.0, 3.0, 4.0]))
            .expect("a");
        store
            .load(b, Data::F32(vec![0.5, 0.5, 0.5, 0.5]))
            .expect("b");
        store.add_f32(&[a, b], typed).expect("类型化调用");
        Store::compute(&mut store, "add_f32", &[a, b], raw).expect("字符串调用");
        let typed = store
            .value(typed)
            .expect("有值")
            .f32()
            .expect("f32")
            .to_vec();
        let raw = store.value(raw).expect("有值").f32().expect("f32").to_vec();
        assert_eq!(typed, raw);
        assert_eq!(typed, vec![1.5, 2.5, 3.5, 4.5]);
    }

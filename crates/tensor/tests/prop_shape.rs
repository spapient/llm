//! 形状与步长的代数性质：任意布局都把多下标**双射**到 `0..numel`。
//!
//! 单点用例只钉住几个形状；这里对任意轴数、任意轴长的形状断言双射，
//! 反向累乘少乘一轴、多乘一轴、或漏掉最后一轴的 `1`，都会被最小反例抓住。

use crate::{DType, Order, Shape, View};
use proptest::prelude::*;
use testkit::prop;

/// 任意形状：0 轴（标量）到 4 轴，每轴 1..=5 个元素，最宽 625 个。
fn dims() -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(1usize..=5, 0..=4)
}

/// 按行主序枚举形状的全部多下标——顺序无所谓，双射才要紧。
fn idxs(dims: &[usize]) -> Vec<Vec<usize>> {
    let mut all = vec![Vec::new()];
    for &dim in dims {
        let mut next = Vec::with_capacity(all.len() * dim);
        for prefix in &all {
            for position in 0..dim {
                let mut idx = prefix.clone();
                idx.push(position);
                next.push(idx);
            }
        }
        all = next;
    }
    all
}

/// 断言每个多下标换出一个互不相同的线性位号，且 `0..numel` 全被取到。
fn assert_bijection(view: &View<'_, f32>, dims: &[usize]) {
    let numel = crate::product(dims);
    let mut seen = vec![false; numel];
    for idx in idxs(dims) {
        let linear = view.at(&idx).expect("枚举出的下标必然在形状内") as usize;
        assert!(linear < numel, "线性位号 {linear} 超出 {numel}：{dims:?}");
        assert!(!seen[linear], "位号 {linear} 被两个下标取到：{dims:?}");
        seen[linear] = true;
    }
    assert!(seen.iter().all(|hit| *hit), "有位号没人取到：{dims:?}");
}

/// 行主序缓冲：第 `i` 个元素就是 `i`，于是「位号」与「值」互为账本。
fn counting(dims: &[usize]) -> Vec<f32> {
    (0..crate::product(dims)).map(|i| i as f32).collect()
}

proptest! {
    #![proptest_config(prop::config())]

    #[test]
    fn row_major_offsets_are_a_bijection_onto_the_elements(dims in dims()) {
        let shape = Shape::new(dims.clone());
        let strides = shape.strides(Order::RowMajor);
        let data = counting(&dims);
        let view = View::strided(shape, strides, &data).expect("缓冲长度与形状一致");
        assert_bijection(&view, &dims);
    }

    #[test]
    fn col_major_offsets_are_a_bijection_onto_the_elements(dims in dims()) {
        let shape = Shape::new(dims.clone());
        let strides = shape.strides(Order::ColMajor);
        let data = counting(&dims);
        let view = View::strided(shape, strides, &data).expect("列主序也是合法布局");
        assert_bijection(&view, &dims);
    }

    /// 轴与步幅同时反序就是转置：坐标换了个写法，元素一个不少。
    #[test]
    fn reversing_axes_and_strides_together_transposes(dims in dims()) {
        let shape = Shape::new(dims.clone());
        let mut axes = dims.clone();
        let mut steps = shape.strides(Order::RowMajor).as_slice().to_vec();
        axes.reverse();
        steps.reverse();
        let data = counting(&dims);
        let view = View::strided(Shape::new(axes.clone()), steps, &data)
            .expect("转置布局仍然可达");
        assert_bijection(&view, &axes);
    }

    /// `to_vec` 是"按行主序读一遍"：任何布局摊平后逐位等于 `at` 的读数。
    ///
    /// 收拢两条性质：连续视图与转置视图的摊平长度都等于 `numel`（不按步长乘出多余元素），
    /// 且元素顺序就是逻辑下标顺序——`gather_strided` 的进位一旦少减一轴就会在此暴露。
    #[test]
    fn flattening_a_layout_reads_exactly_what_at_reads(dims in dims()) {
        let shape = Shape::new(dims.clone());
        let data = counting(&dims);
        let view = View::strided(shape.clone(), shape.strides(Order::RowMajor), &data)
            .expect("缓冲长度与形状一致");
        let want: Vec<f32> = idxs(&dims).iter().filter_map(|idx| view.at(idx)).collect();
        prop_assert_eq!(view.to_vec(), want);

        let rank = dims.len();
        if rank >= 2 {
            let turned = view.clone().transpose(0, rank - 1).expect("两轴都在秩内");
            let axes = turned.dims().to_vec();
            let want: Vec<f32> = idxs(&axes).iter().filter_map(|idx| turned.at(idx)).collect();
            prop_assert_eq!(turned.to_vec(), want);
            prop_assert_eq!(turned.to_vec().len(), view.to_vec().len());
        }
    }

    /// 元素数是各轴之积，且秩、字节数都由它派生——三者不允许各算一遍。
    #[test]
    fn the_element_count_is_the_product_of_the_axes(dims in dims()) {
        let shape = Shape::new(dims.clone());
        prop_assert_eq!(shape.numel(), crate::product(&dims));
        prop_assert_eq!(shape.rank(), dims.len());
        prop_assert_eq!(shape.element_bytes(DType::F32), shape.numel() * 4);
    }

    /// 秩代数的定律：`unsqueeze` 再 `squeeze` 是恒等，`select` 的读数等于原张量补上该轴坐标后的读数。
    ///
    /// 这三条改秩算子此前只有实现、没有对拍；定律一次钉住「轴长 / 步长 / 起点偏移」三者同步。
    #[test]
    fn rank_algebra_keeps_the_element_order(dims in dims()) {
        let shape = Shape::new(dims.clone());
        let data = counting(&dims);
        let view = View::strided(shape.clone(), shape.strides(Order::RowMajor), &data)
            .expect("缓冲长度与形状一致");
        for axis in 0..=dims.len() {
            let round = view
                .clone()
                .unsqueeze::<Vec<usize>>(axis)
                .and_then(|wide| wide.squeeze::<Vec<usize>>(axis))
                .expect("插一个单例轴再摘掉必然成立");
            prop_assert_eq!(round.dims(), view.dims());
            for idx in idxs(&dims) {
                prop_assert_eq!(round.at(&idx), view.at(&idx));
            }
        }
        for axis in 0..dims.len() {
            for at in [0, dims[axis] - 1] {
                let picked = view
                    .clone()
                    .select::<Vec<usize>>(axis, at)
                    .expect("轴内下标必然合法");
                for idx in idxs(&dims) {
                    if idx[axis] != at {
                        continue;
                    }
                    let mut rest = idx.clone();
                    rest.remove(axis);
                    prop_assert_eq!(picked.at(&rest), view.at(&idx));
                }
            }
        }
    }

    /// 单例轴不改变元素顺序：连续布局对 `dims` 和 `dims + [1]` 给出同一条线性位号。
    #[test]
    fn a_singleton_axis_does_not_reorder_anything(dims in dims()) {
        let mut wider = dims.clone();
        wider.push(1);
        let narrow_shape = Shape::new(dims.clone());
        let wide_shape = Shape::new(wider);
        let data = counting(&dims);
        let narrow = View::strided(narrow_shape.clone(), narrow_shape.strides(Order::RowMajor), &data);
        let wide = View::strided(wide_shape.clone(), wide_shape.strides(Order::RowMajor), &data);
        let (narrow, wide) = (narrow.expect("同长缓冲"), wide.expect("多一个单例轴仍可达"));
        for idx in idxs(&dims) {
            let mut extended = idx.clone();
            extended.push(0);
            prop_assert_eq!(narrow.at(&idx), wide.at(&extended));
        }
    }

    /// 翻转只是「这些轴从另一端读」：结果等于把每个下标在指定轴上取镜像后逐个读，且翻两次复原。
    #[test]
    fn flipping_reads_the_named_axes_from_their_far_end(dims in dims(), mask in any::<u16>()) {
        let data = counting(&dims);
        let shape = Shape::new(dims.clone());
        let view = View::strided(shape.clone(), shape.strides(Order::RowMajor), &data);
        let view = view.expect("同长缓冲");
        let axes: Vec<usize> = (0..dims.len()).filter(|axis| mask >> axis & 1 == 1).collect();
        let turned = view.flip(axes.clone());
        let want: Vec<f32> = idxs(&dims)
            .iter()
            .map(|idx| {
                let far: Vec<usize> = idx
                    .iter()
                    .enumerate()
                    .map(|(axis, &at)| {
                        if axes.contains(&axis) {
                            dims[axis] - 1 - at
                        } else {
                            at
                        }
                    })
                    .collect();
                view.at(&far).expect("镜像下标仍在形状内")
            })
            .collect();
        prop_assert_eq!(turned.to_vec(), want);
        prop_assert_eq!(turned.flip(axes).to_vec(), view.to_vec(), "翻两次复原");
    }
}

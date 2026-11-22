//! 形状与步长的代数性质：任意布局都把多下标**双射**到 `0..numel`。
//!
//! 单点用例只钉住几个形状；这里对任意轴数、任意轴长的形状断言双射，
//! 反向累乘少乘一轴、多乘一轴、或漏掉最后一轴的 `1`，都会被最小反例抓住。

use crate::{DType, Order, Shape, Strides, View};
use proptest::prelude::*;
use testkit::prop;

/// 任意形状：0 轴（标量）到 4 轴，每轴 1..=5 个元素，最宽 625 个。
fn dims() -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(1usize..=5, 0..=4)
}

/// 按行主序枚举形状的全部多下标——顺序无所谓，双射才要紧。
fn indices(dims: &[usize]) -> Vec<Vec<usize>> {
    let mut all = vec![Vec::new()];
    for &dim in dims {
        let mut next = Vec::with_capacity(all.len() * dim);
        for prefix in &all {
            for position in 0..dim {
                let mut index = prefix.clone();
                index.push(position);
                next.push(index);
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
    for index in indices(dims) {
        let linear = view.at(&index).expect("枚举出的下标必然在形状内") as usize;
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
        let strides = Strides::contiguous(&shape);
        let data = counting(&dims);
        let view = View::strided(shape, strides, &data).expect("缓冲长度与形状一致");
        assert_bijection(&view, &dims);
    }

    #[test]
    fn col_major_offsets_are_a_bijection_onto_the_elements(dims in dims()) {
        let shape = Shape::new(dims.clone());
        let strides = Strides::ordered(&shape, Order::ColMajor);
        let data = counting(&dims);
        let view = View::strided(shape, strides, &data).expect("列主序也是合法布局");
        assert_bijection(&view, &dims);
    }

    /// 轴与步幅同时反序就是转置：坐标换了个写法，元素一个不少。
    #[test]
    fn reversing_axes_and_strides_together_transposes(dims in dims()) {
        let shape = Shape::new(dims.clone());
        let mut axes = dims.clone();
        let mut steps = Strides::contiguous(&shape).as_slice().to_vec();
        axes.reverse();
        steps.reverse();
        let data = counting(&dims);
        let view = View::strided(Shape::new(axes.clone()), Strides::new(steps), &data)
            .expect("转置布局仍然可达");
        assert_bijection(&view, &axes);
    }

    /// 元素数是各轴之积，且秩、字节数都由它派生——三者不允许各算一遍。
    #[test]
    fn the_element_count_is_the_product_of_the_axes(dims in dims()) {
        let shape = Shape::new(dims.clone());
        prop_assert_eq!(shape.numel(), crate::product(&dims));
        prop_assert_eq!(shape.rank(), dims.len());
        prop_assert_eq!(shape.element_bytes(DType::F32), shape.numel() * 4);
    }

    /// 单例轴不改变元素顺序：连续布局对 `dims` 和 `dims + [1]` 给出同一条线性位号。
    #[test]
    fn a_singleton_axis_does_not_reorder_anything(dims in dims()) {
        let mut wider = dims.clone();
        wider.push(1);
        let narrow_shape = Shape::new(dims.clone());
        let wide_shape = Shape::new(wider);
        let data = counting(&dims);
        let narrow = View::strided(narrow_shape.clone(), Strides::contiguous(&narrow_shape), &data);
        let wide = View::strided(wide_shape.clone(), Strides::contiguous(&wide_shape), &data);
        let (narrow, wide) = (narrow.expect("同长缓冲"), wide.expect("多一个单例轴仍可达"));
        for index in indices(&dims) {
            let mut extended = index.clone();
            extended.push(0);
            prop_assert_eq!(narrow.at(&index), wide.at(&extended));
        }
    }
}

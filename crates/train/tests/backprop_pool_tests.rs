use super::*;

/// 几何反向对拍：前向取元素和，交给 `check_geometry_grad` 做中心差分。
fn sum_grad(op: &str, geometry: Geometry, x0: &[f32]) -> Result<()> {
    check_geometry_grad(op, geometry, x0, |px| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn max_pool1d_matches_central_differences() -> Result<()> {
    let total = nn::conv::max_pool1d::L;
    let out = nn::conv::max_pool1d::OL;
    let x0 = ramp(total);
    check_shaped_grad("max_pool1d_f32", &x0, out, |px| {
        let mut y = vec![0.0f32; out];
        nn::conv::max_pool1d::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn max_pool2d_matches_central_differences() -> Result<()> {
    let side = nn::conv::max_pool2d::H;
    let x0 = ramp(side * side);
    check_shaped_grad("max_pool2d_f32", &x0, side * side, |px| {
        let mut y = vec![0.0f32; side * side];
        nn::conv::max_pool2d::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn avg_pool2d_matches_central_differences() -> Result<()> {
    let side = nn::conv::avg_pool2d::H;
    let x0 = ramp(side * side);
    check_shaped_grad("avg_pool2d_f32", &x0, side * side, |px| {
        let mut y = vec![0.0f32; side * side];
        nn::conv::avg_pool2d::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn adaptive_pool_matches_central_differences() -> Result<()> {
    let (side, out) = (nn::conv::adaptive_pool::H, nn::conv::adaptive_pool::OH);
    let x0 = ramp(side * side);
    check_shaped_grad("adaptive_pool_f32", &x0, out * out, |px| {
        let mut y = vec![0.0f32; out * out];
        nn::conv::adaptive_pool::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn interpolate_matches_central_differences() -> Result<()> {
    let (h, w, out) = (nn::conv::interpolate::H, nn::conv::interpolate::W, nn::conv::interpolate::OH);
    let x0 = ramp(h * w);
    check_shaped_grad("interpolate_f32", &x0, out * out, |px| {
        let mut y = vec![0.0f32; out * out];
        nn::conv::interpolate::forward(px, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn batch_norm_matches_central_differences() -> Result<()> {
    use nn::conv::batch_norm as decl;
    let total = decl::C * decl::S * decl::N;
    let (x0, g0, b0) = (ramp(total), gain(), shift());
    check_affine_grad("batch_norm_f32", &x0, &g0, &b0, total, |px, pg, pb| {
        let mut y = vec![0.0f32; total];
        nn::conv::batch_norm::forward(px, pg, pb, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn group_norm_matches_central_differences() -> Result<()> {
    use nn::conv::group_norm as decl;
    let total = decl::C * decl::S * decl::N;
    let (x0, g0, b0) = (ramp(total), gain(), shift());
    check_affine_grad("group_norm_f32", &x0, &g0, &b0, total, |px, pg, pb| {
        let mut y = vec![0.0f32; total];
        nn::conv::group_norm::forward(px, pg, pb, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn instance_norm_matches_central_differences() -> Result<()> {
    use nn::conv::instance_norm as decl;
    let total = decl::C * decl::S * decl::N;
    let (x0, g0, b0) = (ramp(total), gain(), shift());
    check_affine_grad("instance_norm_f32", &x0, &g0, &b0, total, |px, pg, pb| {
        let mut y = vec![0.0f32; total];
        nn::conv::instance_norm::forward(px, pg, pb, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn layer_norm_matches_central_differences() -> Result<()> {
    use nn::conv::layer_norm as decl;
    let total = decl::R * decl::W;
    let x0 = ramp(total);
    let g0 = gain();
    let b0 = shift();
    check_affine_grad("layer_norm_f32", &x0, &g0, &b0, total, |px, pg, pb| {
        let mut y = vec![0.0f32; total];
        nn::conv::layer_norm::forward(px, pg, pb, &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_max_pool1d_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Pool1d {
        k: 3,
        s: 1,
        p: 1,
        l: 7,
        c: 1,
    };
    let x0 = ramp(7);
    sum_grad("max_pool1d_f32", geometry, &x0)
}

#[test]
fn shaped_max_pool2d_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Pool2d {
        k: 3,
        s: 2,
        p: 0,
        h: 5,
        w: 5,
        c: 2,
    };
    let x0 = ramp(5 * 5 * 2);
    sum_grad("max_pool2d_f32", geometry, &x0)
}

#[test]
fn shaped_avg_pool2d_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Avg2d {
        k: 3,
        s: 2,
        p: 1,
        h: 5,
        w: 5,
        c: 1,
        count_include_pad: false,
    };
    let x0 = ramp(5 * 5);
    sum_grad("avg_pool2d_f32", geometry, &x0)
}

#[test]
fn shaped_adaptive_pool_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Adaptive {
        oh: 3,
        ow: 3,
        h: 6,
        w: 6,
        c: 1,
    };
    let x0 = ramp(36);
    sum_grad("adaptive_pool_f32", geometry, &x0)
}

#[test]
fn shaped_interpolate_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Interp {
        oh: 5,
        ow: 5,
        h: 3,
        w: 3,
        c: 1,
        linear: true,
    };
    let x0 = ramp(9);
    sum_grad("interpolate_f32", geometry, &x0)
}

#[test]
fn shaped_batch_norm_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Norm {
        n: 2,
        spatial: 3,
        channels: 5,
    };
    let (x0, g0, b0) = (ramp(30), gain5(), shift5());
    check_geometry_affine("batch_norm_f32", geometry, &x0, &g0, &b0, |px, pg, pb| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px, pg, pb], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_group_norm_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Group {
        groups: 3,
        n: 2,
        spatial: 3,
        channels: 6,
    };
    let (x0, g0, b0) = (ramp(36), gain6(), shift6());
    check_geometry_affine("group_norm_f32", geometry, &x0, &g0, &b0, |px, pg, pb| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px, pg, pb], &mut y).expect("前向");
        y.iter().sum()
    })
}

#[test]
fn shaped_layer_norm_matches_central_differences() -> Result<()> {
    let geometry = Geometry::Layer { rows: 2, width: 5 };
    let x0 = ramp(10);
    let g0 = gain5();
    let b0 = shift5();
    check_geometry_affine("layer_norm_f32", geometry, &x0, &g0, &b0, |px, pg, pb| {
        let mut y = vec![0.0f32; geometry.outputs()];
        geometry.forward(&[px, pg, pb], &mut y).expect("前向");
        y.iter().sum()
    })
}

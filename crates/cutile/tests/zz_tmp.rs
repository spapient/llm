use crate::builtin::Catalogue;
use crate::{decl, emit, substitute};
#[test]
fn dump_metal() {
    let c = Catalogue::try_builtin().expect("dat");
    let profile = c.profile("m5").expect("m5");
    let f = prim::builtin(profile.meta.vendor.as_str(), profile.meta.emit.ext()).expect("frag");
    let op = c.ops().iter().find(|o| o.name == "adaptive_pool_f32").expect("op");
    let combo = decl::expand(op).expect("c").into_iter().next().expect("one");
    let variant = op.variants.first().expect("v");
    let body = substitute::render(op, &combo, &f).expect("r");
    let s = emit::assemble(profile, op, variant, &combo, &body);
    std::fs::write("/tmp/m5.metal", s.text).unwrap();
}

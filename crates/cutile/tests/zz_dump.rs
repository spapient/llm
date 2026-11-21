#[test]
fn dump() {
    let profiles = cutile::prof::all();
    for p in profiles.iter().filter(|p| p.meta.id == "sm75") {
        for op in cutile::op::all().iter().filter(|o| o.name == "gemv_f32") {
            let combo = cutile::plan::combos(op)[0].clone();
            let body = cutile::substitute::render(op, &combo, &p.fragments);
            let src = cutile::emit::assemble(p, op, &op.variants[0], &combo, &body);
            println!("{}", src.text);
        }
    }
}

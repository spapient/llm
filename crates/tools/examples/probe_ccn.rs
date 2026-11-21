fn main() {
    let path = std::env::args().nth(1).expect("path");
    let text = std::fs::read_to_string(&path).expect("read");
    let raw = llm_tools::strip::strip_tests(&text).expect("strip");
    for finding in llm_tools::ccn::analyze(&raw).expect("analyze") {
        println!("{}#{} ccn={} nloc={} lines {}..{}", finding.name, finding.nth, finding.ccn, finding.nloc, finding.start, finding.end);
    }
}

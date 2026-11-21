//! 档案校验：`validate` 与它的判据函数、见证文件读取。
//!
//! 这里只回答"档案自己是否自洽"，不回答它在某台机器上跑对没有。
use super::*;
/// 写了 `exec` 却没声明执行单元。
const EXEC_DETAIL: &str = "写 `exec` 前必须先用 `[units.<name>]` 声明执行单元";
impl Profile {
    /// 闭集与见证校验：只回答"档案自己是否自洽"，不判断它是否能在某台机器上跑对。
    pub fn validate(&self, witnesses: Option<&dyn WitnessLookup>) -> Result<(), Vec<ProfileError>> {
        let mut errors = self.head_errors();
        let mut files = WitnessFiles::new(witnesses);
        for provide in &self.provides {
            errors.extend(witness_shape(provide));
            errors.extend(files.probe(provide));
            errors.extend(signature_fields(provide));
        }
        errors.extend(self.simt_lanes());
        errors.extend(self.exec_units());
        let passed = errors.is_empty().then_some(());
        passed.ok_or(errors)
    }
    /// 单一根下的校验：见证文件都相对 `root` 解析（`<root>/<vendor>/witness.<ext>`）。
    pub fn validate_in(&self, root: &Path) -> Result<(), Vec<ProfileError>> {
        self.validate(Some(&Root(root)))
    }
    /// 跨供给的重复项与属性键闭集：`impl` 重名、见证复用、无关属性键。
    fn head_errors(&self) -> Vec<ProfileError> {
        let mut errors = self.duplicate_impls();
        errors.extend(self.attr_keys());
        errors.extend(self.duplicate_witnesses());
        errors
    }
    /// 同一原语的多条签名必须用不同 `impl`，否则协商只会取到其中一条。
    fn duplicate_impls(&self) -> Vec<ProfileError> {
        let mut seen: BTreeMap<(Prim, &str), usize> = BTreeMap::new();
        let mut errors = Vec::new();
        for (index, provide) in self.provides.iter().enumerate() {
            let key = (provide.prim, provide.impl_name.as_str());
            let Some(first) = seen.insert(key, index) else {
                continue;
            };
            errors.push(bad(at(index, "impl"), dup_impl(first, provide)));
        }
        errors
    }
    /// 属性键必须属于该原语的能力签名：多写的键永远不会被协商比对，闭集在这里就拒收。
    fn attr_keys(&self) -> Vec<ProfileError> {
        let mut errors = Vec::new();
        for provide in &self.provides {
            for attr in &provide.attrs {
                if !provide.prim.accepts(attr.key()) {
                    errors.push(attr_key_error(provide, attr.key()));
                }
            }
        }
        errors
    }
    /// 一条见证只证明一条签名：两段签名指向同一入口时，"都有见证"就无法归属到哪一条。
    fn duplicate_witnesses(&self) -> Vec<ProfileError> {
        let mut seen: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        let mut errors = Vec::new();
        for (index, provide) in self.provides.iter().enumerate() {
            let Some(witness) = provide.witness.as_ref() else {
                continue;
            };
            let key = (witness.file.as_str(), witness.func.as_str());
            let Some(first) = seen.insert(key, index) else {
                continue;
            };
            let previous = &self.provides[first];
            let detail = shared_witness(witness, previous, provide);
            errors.push(witness_error(provide, detail));
        }
        errors
    }
    /// 子组内交换/广播以"有第二条 lane"为前提；只看 `group` 不够，还要 `declared`。
    fn simt_lanes(&self) -> Vec<ProfileError> {
        let group = self.simt.group;
        let swap = |p: &Provide| matches!(p.prim, Prim::Shuffle | Prim::Broadcast);
        let on_one_lane = |p: &Provide| swap(p) && self.simt.declared && group <= 1;
        let errors = self.provides.iter().filter(|p| on_one_lane(p));
        errors
            .map(|p| bad(prim_at(p.prim, ""), lane_swap_detail(p.prim, group)))
            .collect()
    }
    /// `exec` 必须对得上本档案声明的执行单元，且该单元的存储层要覆盖这条供给。
    fn exec_units(&self) -> Vec<ProfileError> {
        let mut errors = Vec::new();
        for provide in &self.provides {
            if let Some(exec) = exec_attr(provide) {
                errors.extend(self.exec_errors(provide, exec));
            }
        }
        errors
    }
    /// 一条 `exec` 断言：单元要在，且它的存储层要覆盖供给声明的 `space`。
    fn exec_errors(&self, provide: &Provide, exec: &Attr) -> Vec<ProfileError> {
        let field = prim_at(provide.prim, "exec");
        if self.units.is_empty() {
            return vec![bad(field, EXEC_DETAIL)];
        }
        let (reachable, mut errors) = self.exec_reach(&field, exec.values());
        if !provide.space.is_subset(reachable) {
            let unreachable = provide.space.difference(reachable);
            let detail = reach_missing(exec, unreachable);
            errors.push(bad(prim_at(provide.prim, "space"), detail));
        }
        errors
    }
    /// `exec` 取值能触及的存储层并集；本档案没声明的单元逐个报 `Unknown`。
    fn exec_reach(&self, field: &str, values: &[String]) -> (SpaceSet, Vec<ProfileError>) {
        let allowed: Vec<&str> = self.units.iter().map(|unit| unit.name.as_str()).collect();
        let (mut reachable, mut undefined) = (SpaceSet::empty(), Vec::new());
        for value in values {
            match self.units.iter().find(|unit| unit.name == *value) {
                Some(unit) => unit.space.iter().for_each(|space| reachable.insert(space)),
                None => undefined.push(unknown(field, value, &allowed)),
            }
        }
        (reachable, undefined)
    }
}
fn dup_impl(first: usize, provide: &Provide) -> String {
    let prim = provide.prim;
    format!("`{prim}` 的实现名与 provides[{first}] 重名：同一原语的多条签名必须用不同 impl 区分")
}
/// 单 lane 设备声明 lane 交换的说明。
fn lane_swap_detail(prim: Prim, group: u32) -> String {
    format!(
        "`{prim}` 要求 `[simt].group > 1`（子组内交换/广播），本档案声明的是 {group}：单 lane 设备没有可交换的对象，这条能力应当不声明"
    )
}
fn reach_missing(exec: &Attr, missing: SpaceSet) -> String {
    format!("`{}` 访问不到 [{missing}]", exec.values().join("+"))
}
/// 一条见证被两条签名引用的说明。
fn shared_witness(witness: &Witness, previous: &Provide, provide: &Provide) -> String {
    let was = format!("{}/{}", previous.prim, previous.impl_name);
    let now = format!("{}/{}", provide.prim, provide.impl_name);
    format!(
        "见证 `{}` 同时被 `{was}` 与 `{now}` 引用：一条见证只能证明一条签名",
        witness.func
    )
}
/// 见证的形状：`L2` 必须有见证，声明了就要有入口函数名与期望值（不碰文件）。
fn witness_shape(provide: &Provide) -> Vec<ProfileError> {
    let Some(witness) = provide.witness.as_ref() else {
        return missing_witness(provide).into_iter().collect();
    };
    let len = witness.expect.len();
    let too_many = format!("期望输出 {len} 个，超过上限 {}", Witness::MAX_OUT);
    let checks = [
        (
            witness.func.is_empty(),
            "见证必须给出入口函数名，否则无法执行".to_string(),
        ),
        (
            witness.expect.is_empty(),
            "见证必须给出期望输出，否则无法对拍".to_string(),
        ),
        (len > Witness::MAX_OUT, too_many),
    ];
    let failed = checks.into_iter().filter(|(failed, _)| *failed);
    let make = |(_, detail): (bool, String)| witness_error(provide, detail);
    failed.map(make).collect()
}
/// 没有见证时唯一说得通的理由是等级还不到 `L2`——等级是执行的结论，不是自述。
fn missing_witness(provide: &Provide) -> Option<ProfileError> {
    if !provide.grade.is_verified() {
        return None;
    }
    let grade = provide.grade;
    let detail = format!("{grade} 要求 `witness = {{ file, func, expect }}`");
    Some(witness_error(provide, detail))
}
/// 签名的必填项：有数据类型的原语不能给空集；`L2` 及以上必须写实现名。
fn signature_fields(provide: &Provide) -> Vec<ProfileError> {
    let empty_dt = provide.dt.is_empty() && provide.prim.has_dtype();
    let no_impl = provide.grade.is_verified() && provide.impl_name.is_empty();
    let checks = [
        (empty_dt, "dt", "数据类型集合不能为空"),
        (no_impl, "impl", "实现名不能为空"),
    ];
    let failed = checks.into_iter().filter(|(failed, ..)| *failed);
    let make = |(_, key, detail): (bool, &str, &str)| bad(prim_at(provide.prim, key), detail);
    failed.map(make).collect()
}
/// 闭集外的属性键：闭集是该原语签名键的并集 `exec`（执行单元是档案级属性）。
fn attr_key_error(provide: &Provide, key: AttrKey) -> ProfileError {
    let keys = provide.prim.attr_keys();
    let mut allowed: Vec<&str> = keys.iter().map(|key| key.as_str()).collect();
    allowed.push("exec");
    unknown(&prim_at(provide.prim, key.as_str()), key.as_str(), &allowed)
}
/// 一条供给声明的执行单元列表；没写 `exec` 就没有这条判据。
fn exec_attr(provide: &Provide) -> Option<&Attr> {
    let is_exec = |attr: &&Attr| attr.key() == AttrKey::Exec;
    provide.attrs.iter().find(is_exec)
}
/// 见证源码的定位方式：把 `witness.file`（`<vendor>/witness.<ext>`）解析成一个可读路径。
///
/// 片段层按设备拆成六个 crate 之后，"一个根"不再成立：`crates/cpu` 与 `crates/cuda` 各在自己
/// 的源码树里。所以这里只约定"怎么问"，"在哪"由调用方回答（`prim::Layout`）。
pub trait WitnessLookup {
    /// 见证文件的可读路径；`None` 表示这个 vendor 没有任何落盘位置。
    fn path(&self, file: &str) -> Option<PathBuf>;
}
/// 单一根下的定位：`file` 直接拼在根上。
struct Root<'a>(&'a Path);
impl WitnessLookup for Root<'_> {
    fn path(&self, file: &str) -> Option<PathBuf> {
        Some(self.0.join(file))
    }
}
/// 见证文件的一次性读取缓存：同一份源码只读一次，同一个缺失文件只报一次。
struct WitnessFiles<'a> {
    /// 定位方式；`None` 表示本轮不做文件检查（单元测试）。
    witnesses: Option<&'a dyn WitnessLookup>,
    /// 文件 → 源码；`None` 表示不存在或不可读。
    sources: BTreeMap<String, Option<String>>,
    /// 已经报过的缺失路径。
    reported: BTreeSet<String>,
}
impl<'a> WitnessFiles<'a> {
    /// 空缓存。
    fn new(witnesses: Option<&'a dyn WitnessLookup>) -> Self {
        Self {
            witnesses,
            sources: BTreeMap::new(),
            reported: BTreeSet::new(),
        }
    }
    /// 一条供给的文件级判据：文件要在，且真的声明了那个入口函数；不判断跑不跑得通。
    fn probe(&mut self, provide: &Provide) -> Vec<ProfileError> {
        let (Some(witnesses), Some(witness)) = (self.witnesses, provide.witness.as_ref()) else {
            return Vec::new();
        };
        let source = self.source(witnesses, &witness.file);
        let found = source.map(|text| declares(text, witness));
        match found {
            None => self.report(witnesses, &witness.file),
            Some(false) => vec![witness_error(provide, absent_function(witness))],
            Some(true) => Vec::new(),
        }
    }
    /// 读一份见证源码：每个文件最多读一次盘。
    fn source(&mut self, witnesses: &dyn WitnessLookup, file: &str) -> Option<&str> {
        let text = self.sources.entry(file.to_string());
        let text = text.or_insert_with(|| {
            let path = witnesses.path(file)?;
            std::fs::read_to_string(path).ok()
        });
        text.as_deref()
    }
    /// 缺失文件只报第一次：多条供给引用同一份缺失文件时，原因只有一个。
    fn report(&mut self, witnesses: &dyn WitnessLookup, file: &str) -> Vec<ProfileError> {
        let path = witnesses
            .path(file)
            .map_or_else(|| file.to_string(), |path| path.display().to_string());
        if !self.reported.insert(path.clone()) {
            return Vec::new();
        }
        vec![ProfileError::WitnessFile {
            file: path,
            detail: "不存在或不可读".to_string(),
        }]
    }
}
fn declares(text: &str, witness: &Witness) -> bool {
    text.contains(&format!("{}(", witness.func))
}
fn absent_function(witness: &Witness) -> String {
    format!(
        "`{}` 里没有函数 `{}`：声明的见证无法执行",
        witness.file, witness.func
    )
}

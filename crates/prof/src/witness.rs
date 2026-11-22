//! 职责：见证重复检查：同名实现诊断串。
use crate::prof::*;

pub(crate) fn dup_impl(first: usize, provide: &Provide) -> String {
    let prim = provide.prim;
    format!("`{prim}` 的实现名与 provides[{first}] 重名：同一原语的多条签名必须用不同 impl 区分")
}
/// 单 lane 设备声明 lane 交换的说明。
pub(crate) fn lane_swap_detail(prim: Prim, group: u32) -> String {
    format!(
        "`{prim}` 要求 `[simt].group > 1`（子组内交换/广播），本档案声明的是 {group}：单 lane 设备没有可交换的对象，这条能力应当不声明"
    )
}
pub(crate) fn reach_missing(exec: &Attr, missing: SpaceSet) -> String {
    format!("`{}` 访问不到 [{missing}]", exec.values().join("+"))
}
/// 一条见证被两条签名引用的说明。
pub(crate) fn shared_witness(witness: &Witness, previous: &Provide, provide: &Provide) -> String {
    let was = format!("{}/{}", previous.prim, previous.impl_name);
    let now = format!("{}/{}", provide.prim, provide.impl_name);
    format!(
        "见证 `{}` 同时被 `{was}` 与 `{now}` 引用：一条见证只能证明一条签名",
        witness.func
    )
}
/// 见证的形状：`L2` 必须有见证，声明了就要有入口函数名与期望值（不碰文件）。
pub(crate) fn witness_shape(provide: &Provide) -> Vec<ProfileError> {
    let Some(witness) = provide.witness.as_ref() else {
        return missing_witness(provide).into_iter().collect();
    };
    let make = |(_, detail): (&'static str, String)| witness_error(provide, detail);
    shape_details(witness).into_iter().map(make).collect()
}
/// 见证形状的判据本体：键是 `func` / `expect`，两种供给（原语与写回形式）共用一份。
pub(crate) fn shape_details(witness: &Witness) -> Vec<(&'static str, String)> {
    let len = witness.expect.len();
    let too_many = format!("期望输出 {len} 个，超过上限 {}", Witness::MAX_OUT);
    let checks = [
        (
            witness.func.is_empty(),
            "func",
            "见证必须给出入口函数名，否则无法执行".to_string(),
        ),
        (
            witness.expect.is_empty(),
            "expect",
            "见证必须给出期望输出，否则无法对拍".to_string(),
        ),
        (len > Witness::MAX_OUT, "expect", too_many),
    ];
    let failed = checks.into_iter().filter(|(failed, ..)| *failed);
    failed.map(|(_, key, detail)| (key, detail)).collect()
}
/// 没有见证时唯一说得通的理由是等级还不到 `L2`——等级是执行的结论，不是自述。
pub(crate) fn missing_witness(provide: &Provide) -> Option<ProfileError> {
    if !provide.grade.is_verified() {
        return None;
    }
    let grade = provide.grade;
    let detail = format!("{grade} 要求 `witness = {{ file, func, expect }}`");
    Some(witness_error(provide, detail))
}
/// 签名的必填项：有数据类型的原语不能给空集；`L2` 及以上必须写实现名。
pub(crate) fn signature_fields(provide: &Provide) -> Vec<ProfileError> {
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
pub(crate) fn attr_key_error(provide: &Provide, key: AttrKey) -> ProfileError {
    let keys = provide.prim.attr_keys();
    let mut allowed: Vec<&str> = keys.iter().map(|key| key.as_str()).collect();
    allowed.push("exec");
    unknown(prim_at(provide.prim, key.as_str()), key.as_str(), &allowed)
}
/// 一条供给声明的执行单元列表；没写 `exec` 就没有这条判据。
pub(crate) fn exec_attr(provide: &Provide) -> Option<&Attr> {
    let is_exec = |attr: &&Attr| attr.key() == AttrKey::Exec;
    provide.attrs.iter().find(is_exec)
}
/// 见证源码的定位方式：把 `witness.file`（`<vendor>/witness.<ext>`）解析成一个可读路径。
pub trait WitnessLookup {
    /// 见证文件的可读路径；`None` 表示这个 vendor 没有任何落盘位置。
    fn path(&self, file: &str) -> Option<PathBuf>;
}
/// 单一根下的定位：`file` 直接拼在根上。
pub(crate) struct Root<'a>(pub(crate) &'a Path);
impl WitnessLookup for Root<'_> {
    fn path(&self, file: &str) -> Option<PathBuf> {
        Some(self.0.join(file))
    }
}
/// 见证文件的一次性读取缓存：同一份源码只读一次，同一个缺失文件只报一次。
pub(crate) struct WitnessFiles<'a> {
    /// 定位方式；`None` 表示本轮不做文件检查（单元测试）。
    witnesses: Option<&'a dyn WitnessLookup>,
    /// 文件 → 源码；`None` 表示不存在或不可读。
    sources: BTreeMap<String, Option<String>>,
    /// 已经报过的缺失路径。
    reported: BTreeSet<String>,
}
impl<'a> WitnessFiles<'a> {
    /// 空缓存。
    pub(crate) fn new(witnesses: Option<&'a dyn WitnessLookup>) -> Self {
        Self {
            witnesses,
            sources: BTreeMap::new(),
            reported: BTreeSet::new(),
        }
    }
    /// 一条供给的文件级判据：文件要在，且真的声明了那个入口函数；不判断跑不跑得通。
    pub(crate) fn probe(&mut self, provide: &Provide) -> Vec<ProfileError> {
        let make = |detail: String| witness_error(provide, detail);
        self.probe_witness(provide.witness.as_ref(), &make)
    }
    /// 判据本体：诊断由调用方成形（原语与写回形式各有自己的字段路径）。
    pub(crate) fn probe_witness(
        &mut self,
        witness: Option<&Witness>,
        error: &dyn Fn(String) -> ProfileError,
    ) -> Vec<ProfileError> {
        let (Some(witnesses), Some(witness)) = (self.witnesses, witness) else {
            return Vec::new();
        };
        let source = self.source(witnesses, &witness.file);
        let found = source.map(|text| declares(text, witness));
        match found {
            None => self.report(witnesses, &witness.file),
            Some(false) => vec![error(absent_function(witness))],
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
pub(crate) fn declares(text: &str, witness: &Witness) -> bool {
    text.contains(&format!("{}(", witness.func))
}
pub(crate) fn absent_function(witness: &Witness) -> String {
    format!(
        "`{}` 里没有函数 `{}`：声明的见证无法执行",
        witness.file, witness.func
    )
}

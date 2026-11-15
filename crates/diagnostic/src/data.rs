//! 声明数据树：编译期语料在运行期的形状。
//!
//! 这里**没有解析器**：`ops/*.toml`、`devices/*.toml`、`lints/vocabulary.toml` 由各 crate 的
//! `build.rs` 在编译期用构建期依赖读出并展开成 `Table` 字面量（ADR 0022），运行期只遍历这棵树。
//!
//! 保留"表 + 值"这一层，是因为字段含义属于各层自己的语义：`[axes]` 属于 `op`，`[[provides]]`
//! 属于 `prof`，门禁阈值属于 `tool`——本 crate 不解释任何领域含义，也不决定语料放在哪。
//!
//! 构造入口分两组：[`Table::from_pairs`] 与 `Value::text` / `int` / `real` / `flag` / `array` /
//! `table` 给生成的代码用；[`Table::set`]、[`Table::push`]、[`Table::remove`]、[`Table::at_mut`]
//! 给测试改写内置语料用（"把内置声明改成非法的一份"不该再靠字符串替换）。

/// 解析后的表：保持插入顺序，便于稳定打印。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Table {
    pub(crate) entries: Vec<(String, Value)>,
}

/// TOML 值。
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// 字符串。
    Str(String),
    /// 整数。
    Int(i64),
    /// 浮点。
    Float(f64),
    /// 布尔。
    Bool(bool),
    /// 数组。
    Array(Vec<Value>),
    /// 表（`[section]` 或内联 `{ ... }`）。
    Table(Table),
}

impl Table {
    /// 空表。
    pub fn new() -> Self {
        Self::default()
    }

    /// 全部键值对，按插入顺序。
    pub fn entries(&self) -> &[(String, Value)] {
        &self.entries
    }

    /// 取一个值。
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// 按键取子表。
    pub fn table(&self, key: &str) -> Option<&Table> {
        match self.get(key) {
            Some(Value::Table(table)) => Some(table),
            _ => None,
        }
    }

    /// 按键取字符串。
    pub fn string(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(Value::Str(text)) => Some(text),
            _ => None,
        }
    }

    /// 按键取整数。
    pub fn int(&self, key: &str) -> Option<i64> {
        match self.get(key) {
            Some(Value::Int(value)) => Some(*value),
            _ => None,
        }
    }

    /// 按键取浮点（整数也接受）。
    pub fn float(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(Value::Float(value)) => Some(*value),
            Some(Value::Int(value)) => Some(*value as f64),
            _ => None,
        }
    }

    /// 按键取布尔。
    pub fn bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(Value::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    /// 按键取数组。
    pub fn array(&self, key: &str) -> Option<&[Value]> {
        match self.get(key) {
            Some(Value::Array(items)) => Some(items),
            _ => None,
        }
    }

    /// 按键取字符串数组；不是数组或含非字符串项即为 `None`。
    pub fn string_array(&self, key: &str) -> Option<Vec<&str>> {
        let items = self.array(key)?;
        items
            .iter()
            .map(|item| match item {
                Value::Str(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// 按键取表数组（`[[variants]]`）。
    pub fn table_array(&self, key: &str) -> Option<Vec<&Table>> {
        let items = self.array(key)?;
        items
            .iter()
            .map(|item| match item {
                Value::Table(table) => Some(table),
                _ => None,
            })
            .collect()
    }

    /// 键是否存在。
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
}

impl Table {
    /// 由键值对构造，保持给定顺序。
    pub fn from_pairs(pairs: Vec<(&str, Value)>) -> Self {
        let mut table = Self::new();
        for (key, value) in pairs {
            table.entries.push((key.to_string(), value));
        }
        table
    }

    /// 按键取可变值。
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.entries
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// 按键取可变子表。
    pub fn table_mut(&mut self, key: &str) -> Option<&mut Table> {
        match self.get_mut(key) {
            Some(Value::Table(table)) => Some(table),
            _ => None,
        }
    }

    /// 按键取可变数组。
    pub fn array_mut(&mut self, key: &str) -> Option<&mut Vec<Value>> {
        match self.get_mut(key) {
            Some(Value::Array(items)) => Some(items),
            _ => None,
        }
    }

    /// 按点号路径写入：已存在则原地替换（顺序不变），否则追加。
    ///
    /// 路径不存在（父级缺失或类型不符）时静默不写——测试改写语料的前提是"路径本来就在"，
    /// 写错路径的后果是断言失败，而不是 panic 掉整条测试。
    pub fn set(&mut self, path: &str, value: Value) {
        let segments = segments(path);
        let Some((last, parents)) = segments.split_last() else {
            return;
        };
        let Some(table) = self.parent_mut(parents) else {
            return;
        };
        match last.1 {
            None => table.set_key(last.0, value),
            Some(index) => table.set_index(last.0, index, value),
        }
    }

    /// 按点号路径追加一个数组元素。
    pub fn push(&mut self, path: &str, value: Value) {
        let segments = segments(path);
        let Some((last, parents)) = segments.split_last() else {
            return;
        };
        let Some(table) = self.parent_mut(parents) else {
            return;
        };
        table.push_key(last.0, value);
    }

    /// 删除一个键，返回值。
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let index = self.entries.iter().position(|(name, _)| name == key)?;
        Some(self.entries.remove(index).1)
    }

    /// 路径最后一段的宿主表：无父级就是自己。
    fn parent_mut<'a>(&'a mut self, parents: &[(&str, Option<usize>)]) -> Option<&'a mut Table> {
        if parents.is_empty() {
            return Some(self);
        }
        match resolve_mut(self, parents) {
            Some(Value::Table(table)) => Some(table),
            _ => None,
        }
    }

    /// 按键写入：已存在则原地替换，否则追加。
    fn set_key(&mut self, key: &str, value: Value) {
        match self.get_mut(key) {
            Some(slot) => *slot = value,
            None => self.entries.push((key.to_string(), value)),
        }
    }

    /// 替换数组里的第 `index` 项；越界或类型不符时静默不写。
    fn set_index(&mut self, key: &str, index: usize, value: Value) {
        let Some(Value::Array(items)) = self.get_mut(key) else {
            return;
        };
        if let Some(slot) = items.get_mut(index) {
            *slot = value;
        }
    }

    /// 往数组追加一项；键不是数组时就地替换成只含该项的数组。
    fn push_key(&mut self, key: &str, value: Value) {
        match self.get_mut(key) {
            Some(Value::Array(items)) => items.push(value),
            Some(_) => self.set_key(key, Value::Array(vec![value])),
            None => self
                .entries
                .push((key.to_string(), Value::Array(vec![value]))),
        }
    }

    /// 按点号路径取值，路径段可带数组下标（`variants[0].order`）。
    pub fn at(&self, path: &str) -> Option<&Value> {
        let segments = segments(path);
        let (first, rest) = segments.split_first()?;
        let value = step(self.get(first.0)?, first.1)?;
        walk(value, rest)
    }

    /// 按点号路径取可变值，语义同 [`Table::at`]。
    pub fn at_mut(&mut self, path: &str) -> Option<&mut Value> {
        resolve_mut(self, &segments(path))
    }
}

impl Value {
    /// 字符串值。
    pub fn text(value: &str) -> Self {
        Self::Str(value.to_string())
    }

    /// 整数值。
    pub fn int(value: i64) -> Self {
        Self::Int(value)
    }

    /// 浮点值。
    pub fn real(value: f64) -> Self {
        Self::Float(value)
    }

    /// 布尔值。
    pub fn flag(value: bool) -> Self {
        Self::Bool(value)
    }

    /// 数组值。
    pub fn array(items: Vec<Self>) -> Self {
        Self::Array(items)
    }

    /// 表值。
    pub fn table(pairs: Vec<(&str, Self)>) -> Self {
        Self::Table(Table::from_pairs(pairs))
    }
}

/// 拆一条点号路径；每段可带一个数组下标。
fn segments(path: &str) -> Vec<(&str, Option<usize>)> {
    path.split('.')
        .map(|segment| match segment.split_once('[') {
            Some((name, rest)) => {
                let index = rest
                    .strip_suffix(']')
                    .and_then(|digits| digits.parse().ok());
                (name, index)
            }
            None => (segment, None),
        })
        .collect()
}

/// 对一步路径取值（只处理下标，不带键名）。
fn step(value: &Value, index: Option<usize>) -> Option<&Value> {
    match (value, index) {
        (value, None) => Some(value),
        (Value::Array(items), Some(index)) => items.get(index),
        (_, Some(_)) => None,
    }
}

fn step_mut(value: &mut Value, index: Option<usize>) -> Option<&mut Value> {
    match (value, index) {
        (value, None) => Some(value),
        (Value::Array(items), Some(index)) => items.get_mut(index),
        (_, Some(_)) => None,
    }
}

/// 从一张表出发解析整条路径（第一段按键取，其余交给 [`walk_mut`]）。
fn resolve_mut<'a>(
    table: &'a mut Table,
    segments: &[(&str, Option<usize>)],
) -> Option<&'a mut Value> {
    let (first, rest) = segments.split_first()?;
    let value = step_mut(table.get_mut(first.0)?, first.1)?;
    walk_mut(value, rest)
}

/// 走完剩下的路径段。
fn walk<'a>(value: &'a Value, remaining: &[(&str, Option<usize>)]) -> Option<&'a Value> {
    let Some((first, rest)) = remaining.split_first() else {
        return Some(value);
    };
    let Value::Table(table) = value else {
        return None;
    };
    let child = step(table.get(first.0)?, first.1)?;
    walk(child, rest)
}

fn walk_mut<'a>(
    value: &'a mut Value,
    remaining: &[(&str, Option<usize>)],
) -> Option<&'a mut Value> {
    let Some((first, rest)) = remaining.split_first() else {
        return Some(value);
    };
    let Value::Table(table) = value else {
        return None;
    };
    let child = step_mut(table.get_mut(first.0)?, first.1)?;
    walk_mut(child, rest)
}

#[cfg(test)]
#[path = "../tests/data_tests.rs"]
mod tests;

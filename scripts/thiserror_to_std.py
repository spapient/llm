#!/usr/bin/env python3
"""thiserror → 手写 std::error::Error 转换器。

逐文件处理 crates/*/src/error.rs：
- 移除 `use thiserror::Error;` 与 derive 中的 `Error`
- 删除 `#[error(...)]` / `#[from]` 属性，记录语义
- enum 后追加手写 `impl fmt::Display` / `impl std::error::Error` / `impl From`
保真原则：格式串原样进 write!；transparent 转发内部 Display 与 source；#[from] 生成 From。
"""
import glob, re, sys

def parse_enum(src: str):
    """返回 (enum_name, [(variant_name, attrs, froms, fields_repr, decl_lines)], tail)"""
    m = re.search(r'pub enum (\w+) \{', src)
    if not m:
        return None
    name = m.group(1)
    start = m.end() - 1  # '{' 位置
    # 括号深度计数找到匹配的 '}'
    depth = 1
    i = start + 1
    while i < len(src) and depth > 0:
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
        i += 1
    if depth != 0:
        raise ValueError(f"unbalanced enum body in {name}")
    body = src[start + 1:i - 1]
    tail_start = i
    # 剔除注释行与空行（///、//、/* 块注释、空行）——注释内括号/逗号不得参与分割
    codelines = []
    for ln in body.split('\n'):
        s = ln.strip()
        if s == '' or s.startswith('///') or s.startswith('//') or s.startswith('/*'):
            continue
        codelines.append(ln)
    code = '\n'.join(codelines)
    # 按顶层 ',' 分割变体（追踪 {} [] () 深度，属性/字段内逗号不分割）
    variants = []
    depth = 0
    cur = []
    for ch in code:
        if ch in '{[(':
            depth += 1
        elif ch in '}])':
            depth -= 1
        if ch == ',' and depth == 0:
            variants.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if ''.join(cur).strip():
        variants.append(''.join(cur).strip())
    parsed = []
    for v in variants:
        if not v:
            continue
        # 剥离 doc 注释与空行（注释原样保留在 enum 内，不参与解析）
        v = '\n'.join(l for l in v.split('\n') if not l.strip().startswith('///') and l.strip() != '').strip()
        if not v:
            continue
        # 提取属性
        attrs = []
        froms = []
        # 收集 #[...] 属性行（error 属性可能多行）
        while True:
            am = re.search(r'#\[([^\]]*)\]', v)
            if not am:
                break
            attr = am.group(1).strip()
            if attr == 'from':
                froms.append(None)
            elif attr.startswith('error('):
                attrs.append(attr[6:-1].strip())  # 内容含引号串或 transparent
            v = v[:am.start()] + v[am.end():]
        v = v.strip()
        # 变体声明：Variant { .. } / Variant(..) / Variant
        vm = re.match(r'(\w+)\s*(\{.*\}|\(.*\))?', v, re.S)
        if not vm:
            raise ValueError(f"cannot parse variant: {v[:80]!r}")
        vname = vm.group(1)
        rest = vm.group(2) or ''
        parsed.append((vname, attrs, froms, rest, v))
    return name, parsed, src[tail_start:]

def display_pattern(vname: str, rest: str):
    """match 模式：单元 / 命名结构体 / 元组"""
    r = rest.strip()
    if r.startswith('{'):
        # 命名结构体：{ a: T, b: T } -> { a, b }
        fields = [f.split(':')[0].strip() for f in r[1:-1].split(',') if f.strip()]
        return f"Self::{vname} {{ {', '.join(fields)} }}"
    if r.startswith('('):
        inner = r[1:-1].strip()
        if not inner:
            return f"Self::{vname}"
        # 元组字段：可能带 #[from]（已去除）——取类型名作为绑定名不行，用占位名
        n = inner.count(',') + 1 if inner else 0
        return f"Self::{vname}({', '.join(f'f{i}' for i in range(n))})"
    return f"Self::{vname}"

def display_body(vname: str, attrs, rest: str):
    """Display 匹配分支体"""
    pat = display_pattern(vname, rest)
    if attrs:
        a = attrs[0].strip()
        if a == 'transparent':
            # 元组单字段：绑定 f0
            n = rest.strip()[1:-1].count(',') + 1 if rest.strip().startswith('(') else 0
            binds = ', '.join(f'f{i}' for i in range(n))
            if n == 1:
                return pat, f"std::fmt::Display::fmt(f0, f)"
            # transparent 罕见多字段——不支持则报错
            raise ValueError(f"transparent with {n} fields: {vname}")
        # 普通格式化串：可能多行（引号内换行已由属性整体保留）
        # 去除外层引号
        m = re.match(r'"((?:[^"\\]|\\.)*)"', a)
        if m:
            lit = m.group(1)
            # 判断是否有 {} 插值
            if re.search(r'\{[^}]+\}', lit) or re.search(r'\{[^}]*\}', lit):
                return pat, f'write!(f, "{lit}")'
            return pat, f'f.write_str("{lit}")'
        raise ValueError(f"non-string error attr: {a[:60]!r}")
    # 无属性：Display 为变体名
    return pat, f'f.write_str("{vname}")'

def gen(name, parsed):
    out = []
    # Display（全限定路径，不依赖 use 导入）
    out.append(f"impl std::fmt::Display for {name} {{")
    out.append(f"    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{")
    out.append(f"        match self {{")
    for vname, attrs, froms, rest, _ in parsed:
        pat, body = display_body(vname, attrs, rest)
        out.append(f"            {pat} => {body},")
    out.append(f"        }}")
    out.append(f"    }}")
    out.append(f"}}")
    out.append(f"")
    # Error impl
    out.append(f"impl std::error::Error for {name} {{")
    out.append(f"    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {{")
    src_branches = []
    for vname, attrs, froms, rest, _ in parsed:
        if attrs and attrs[0].strip() == 'transparent':
            n = rest.strip()[1:-1].count(',') + 1 if rest.strip().startswith('(') else 0
            binds = ', '.join(f'f{i}' for i in range(n))
            src_branches.append(f"Self::{vname}({binds}) => Some(f0),")
    if src_branches:
        out.append(f"        match self {{")
        for b in src_branches:
            out.append(f"            {b}")
        out.append(f"            _ => None,")
        out.append(f"        }}")
    else:
        out.append(f"        None")
    out.append(f"    }}")
    out.append(f"}}")
    out.append(f"")
    # From impls
    for vname, attrs, froms, rest, _ in parsed:
        if froms:
            r = rest.strip()
            inner = r[1:-1].strip() if r.startswith('(') else ''
            # 字段类型：去掉可能残留的 #[from]
            ty = inner.split(':')[-1].strip() if inner else ''
            if not ty and inner:
                # 元组字段直接是类型（无命名）
                ty = inner.strip()
            out.append(f"impl From<{ty}> for {name} {{")
            out.append(f"    fn from(v: {ty}) -> Self {{ Self::{vname}(v) }}")
            out.append(f"}}")
            out.append(f"")
    return out

def main():
    files = sorted(glob.glob('crates/*/src/error.rs'))
    for f in files:
        src = open(f).read()
        parsed = parse_enum(src)
        if parsed is None:
            print(f"SKIP {f}: no pub enum")
            continue
        name, variants, tail = parsed
        # 头：去 thiserror use；derive 去 Error
        lines = []
        changed = False
        for line in src.split('\n'):
            if 'use thiserror' in line:
                changed = True
                continue
            lines.append(line)
        head = '\n'.join(lines)
        # 幂等：已转换（含生成的 Display impl）则跳过
        if re.search(rf'impl std::fmt::Display for {name}\b', head):
            print(f"SKIP (already converted) {f}")
            continue
        # derive 去 Error
        head = re.sub(r'(#\[derive\([^\]]*?)(,\s*Error|\bError\s*,)([^\]]*\)\])', r'\1\3', head)
        head = re.sub(r'#\[derive\(([^)\]]*?)\s*Error\s*([^)\]]*?)\)\]', lambda m: f'#[derive({m.group(1)}{m.group(2)})]', head)
        # 删除 enum 体内的 #[error(...)] / #[from] 属性（多行属性跨行）
        head = re.sub(r'#\[error\([^\]]*\)\]', '', head, flags=re.S)
        head = re.sub(r'#\[from\]', '', head)
        # 加 fmt 导入（若没有）
        if 'use std::fmt' not in head and 'use core::fmt' not in head:
            # 在文件顶部 use 区插入——简单起见插在 use thiserror 位置后
            head = head.replace('use thiserror::Error;\n', '')  # 已删
            head = re.sub(r'(^use [^\n]+\n)', r'\1use std::fmt;\n', head, count=1)
        gen_lines = gen(name, variants)
        out = head + '\n\n' + '\n'.join(gen_lines)
        # tail 是 enum 后的内容（通常空/EOF）
        open(f, 'w').write(out.rstrip() + '\n')
        print(f"OK {f}: {name} {len(variants)} variants, {sum(1 for _,a,_,_,_ in variants if a and a[0].strip()=='transparent')} transparent, {sum(1 for _,_,fr,_,_ in variants if fr)} from")

if __name__ == '__main__':
    main()

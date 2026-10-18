#!/usr/bin/env python3
"""criterion → 自写 harness 批量迁移器（健壮版）。

支持的形态：
- `use criterion::{...};` → `use ::bench::harness::{report, run};`
- `fn f(c: &mut Criterion)` → `fn f()`
- `X.bench_function(NAME, |param| { b.iter(...) });` → `report(&run(NAME, || ...));`
  - NAME：字面量 "n" / `&format!(...)` / `&name`（原样保留）
  - param：任意（b / bch / 其它）
  - b.iter 闭包：`|| EXPR` / `|| { ... }`（从 .iter 的 `(` 配平）
  - 收尾：`});` / `);` / `; });`
- `let mut group = c.benchmark_group("g");` + `group.throughput(...)` + `group.finish();`
  → 删除，group.bench_function 名字注入 "g/" 前缀
- `criterion_group!(benches, a, b, ...);` + `criterion_main!(benches);`（含多行）
  → `fn main() { a(); b(); }`

用法：python3 scripts/migrate_bench.py benches/xxx.rs [benches/yyy.rs ...]
"""
import re
import sys


def find_matching(text, start, open_ch, close_ch):
    """从 start 开始配平括号/花括号，返回闭合位置（含闭合符）。"""
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced {open_ch}{close_ch} from {start}")


def convert(path):
    src = open(path, encoding='utf-8').read()

    # 1. criterion import → harness
    src = re.sub(
        r'use criterion::\{[^}]*\};',
        'use ::bench::harness::{report, run};',
        src,
    )

    # 2. 函数签名去掉 Criterion 参数
    src = re.sub(r'fn (\w+)\(c: &mut Criterion\)', r'fn \1()', src)

    lines = src.split('\n')
    out = []
    prefix = None
    pending_main = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # group 声明 / throughput / finish 删除
        m = re.search(r'\.benchmark_group\("([^"]+)"\)', line)
        if m:
            prefix = m.group(1)
            i += 1
            continue
        if re.search(r'throughput\(', line) or re.search(r'group\.finish\(\)', line):
            i += 1
            continue

        # bench_function 调用
        m = re.match(r'\s*(?:\w+\.)?bench_function\((.*)$', line)
        if m:
            arg = m.group(1).strip()
            m2 = re.match(r'(.+),\s*\|([\w]+)\|\s*\{?\s*$', arg)
            if not m2:
                raise ValueError(f"unsupported bench_function shape in {path}: {line}")
            name_expr = m2.group(1).strip()   # "n" 或 &format!(...) 或 &name

            # 收集跨行直到配平出完整 .iter(...) 调用（rest 非空）
            buf = line
            j = i
            while True:
                k = buf.find('.iter(')
                if k != -1:
                    try:
                        kparen = k + len('.iter(') - 1
                        close = find_matching(buf, kparen, '(', ')')
                        rest = buf[close + 1:].lstrip()
                        complete = (
                            rest.startswith('});')
                            or rest.startswith(');')
                            or (rest.startswith(';')
                                and ('});' in rest or ');' in rest))
                        )
                        if complete:
                            break
                    except ValueError:
                        pass
                j += 1
                buf += '\n' + lines[j]
            body, after = buf[kparen + 1:close].strip(), close + 1
            assert body.startswith('||'), f"expected || after .iter(, got {body[:12]!r}"
            body = body[2:].strip()

            # 收尾
            rest = buf[after:].lstrip()
            if rest.startswith('});'):
                consumed = after + 3
            elif rest.startswith(');'):
                consumed = after + 2
            elif rest.startswith(';'):
                tail = rest[1:].lstrip()
                if tail.startswith('});'):
                    consumed = after + 4
                elif tail.startswith(');'):
                    consumed = after + 3
                else:
                    consumed = after + 1
            else:
                raise ValueError(f"cannot find tail after .iter in {path}: {name_expr}")

            if prefix:
                mlit = re.match(r'^"([^"]*)"$', name_expr)
                if not mlit:
                    raise ValueError(
                        f"group bench with non-literal name in {path}: {name_expr}"
                    )
                full = f'"{prefix}/{mlit.group(1)}"'
            else:
                full = name_expr
            out.append(f'    report(&run({full}, || {body}));')
            i += buf.count('\n') + 1
            continue

        # criterion_group!（多行）
        m = re.match(r'criterion_group!\(\s*$', line)
        if m:
            j = i + 1
            names = []
            while j < len(lines):
                t = lines[j].strip().rstrip(',')
                if t == 'benches' or t == 'benches,':
                    j += 1
                    continue
                if t.endswith(');'):
                    break
                if t:
                    names.append(t)
                j += 1
            pending_main = names
            i = j + 1
            continue
        # criterion_group!（单行）
        m = re.match(r'criterion_group!\(benches[^)]*\)\s*;', line)
        if m:
            inner = line[line.find('(') + 1:line.rfind(')')]
            pending_main = [
                x.strip()
                for x in inner.split(',')
                if x.strip() and x.strip() != 'benches'
            ]
            i += 1
            continue
        if re.match(r'criterion_main!\(benches\)', line):
            i += 1
            continue

        out.append(line)
        i += 1

    src = '\n'.join(out)
    if pending_main:
        src += '\n\nfn main() {\n' + ''.join(f'    {f}();\n' for f in pending_main) + '}\n'
    open(path, 'w', encoding='utf-8').write(src)
    print(f"{path}: 转换完成 (main: {pending_main})")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        convert(p)

#!/usr/bin/env python3
"""Relocate inline `#[cfg(test)]` modules to a repository-root integration file.

For a given crate this extracts every inline test module and rewrites the
crate-internal paths to the crate's *public* surface:

* `super::X` / `crate::a::b::X` are mapped to `<crate>::<public module path>::X`
  when the module is publicly reachable, otherwise collapsed to `<crate>::X`
  when `X` is re-exported at the crate root.
* The relocated tests for each source module are wrapped in a nested
  `mod <mangled path> { ... }` block so helper and test names stay isolated.

Modules that still reach a truly non-`pub` item fail to compile; those crates are
handled by hand. See docs/engineering.md.

Usage: python3 scripts/migrateinline.py <crate> [<crate> ...]
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rustscan  # noqa: E402

HEADER = """//! Relocated unit tests for the `{crate}` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/{crate}/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

"""


def extract_blocks(text):
    return _rustscan.test_blocks(text)


def module_paths(crate):
    """Map each source file to (is_publicly_reachable, path components)."""
    src_root = f'crates/{crate}/src'

    def resolve(dirpath, name):
        for cand in (f'{dirpath}/{name}.rs', f'{dirpath}/{name}/mod.rs'):
            if os.path.exists(cand):
                return cand
        return None

    def path_of(file):
        rel = os.path.relpath(file, src_root)
        if rel == 'lib.rs':
            return []
        parts = rel.split('/')
        if parts[-1] == 'mod.rs':
            return parts[:-1]
        return parts[:-1] + [os.path.splitext(parts[-1])[0]]

    public = set()
    stack = [(f'{src_root}/lib.rs', [], True)]
    seen = set()
    while stack:
        file, comps, reachable = stack.pop()
        if file in seen:
            continue
        seen.add(file)
        if reachable:
            public.add(tuple(comps))
        text = open(file).read()
        dirpath = os.path.dirname(file)
        for m in re.finditer(r'^\s*(pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*[;{]', text, re.M):
            is_pub = m.group(1) is not None
            name = m.group(2)
            child = resolve(dirpath, name)
            if child is None:
                continue
            stack.append((child, comps + [name], reachable and is_pub))

    result = {}
    for path in glob.glob(f'{src_root}/**/*.rs', recursive=True):
        comps = path_of(path)
        result[path] = (tuple(comps) in public, comps)
    return result


def root_exports(crate):
    text = open(f'crates/{crate}/src/lib.rs').read()
    names = set()
    for m in re.finditer(r'^\s*pub\s+use\s+[A-Za-z0-9_:]+\s*::\s*\{([^}]*)\}\s*;', text, re.M | re.S):
        for n in m.group(1).split(','):
            n = n.strip()
            if n:
                names.add(n.split(' as ')[-1].strip())
    for m in re.finditer(r'^\s*pub\s+use\s+[A-Za-z0-9_:]*::([A-Za-z_][A-Za-z0-9_]*)\s*;', text, re.M):
        names.add(m.group(1))
    return names


def collapse_paths(body, crate, exports):
    """Rewrite `crate::a::b::…` to `::<crate>::<public tail>`.

    Scans each `crate::`/`<crate>::` root, collects the following path
    segments, then trims the leading (private) segments up to the first
    identifier that is a crate-root re-export, keeping the rest verbatim.
    """
    # Match `crate::…` or the absolute `::<crate>::…`. The bare crate name is
    # deliberately excluded: a crate like `tokenizer`/`tensor` shares its name
    # with a type, so `tokenizer::fromjson` can be a type path, not a crate root.
    root_re = re.compile(rf'(?<![\w:])(?:crate::|::{re.escape(crate)}::)')
    out = []
    pos = 0
    while True:
        m = root_re.search(body, pos)
        if not m:
            out.append(body[pos:])
            break
        out.append(body[pos:m.start()])
        j = m.end()
        segs = []
        while True:
            sm = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', body[j:])
            if not sm:
                break
            seg = sm.group(1)
            if body[j + len(seg):j + len(seg) + 2] == '::':
                segs.append(seg)
                j += len(seg) + 2
            else:
                segs.append(seg)
                j += len(seg)
                break
        if body[j:j + 1] == '{':
            close = body.index('}', j)
            inner = body[j + 1:close]
            names = [n.strip().split(' as ')[0].strip() for n in inner.split(',') if n.strip()]
            if not segs or (names and all(n in exports for n in names)):
                out.append(f'::{crate}::' + body[j:close + 1])
            else:
                out.append(f'::{crate}::' + '::'.join(segs) + '::' + body[j:close + 1])
            pos = close + 1
            continue
        trim = 0
        for idx, seg in enumerate(segs):
            if seg in exports:
                trim = idx
                break
        tail = '::'.join(segs[trim:])
        out.append(f'::{crate}::' + tail)
        pos = j
    return ''.join(out)


def rewrite(body, crate, exports, public, modpath):
    base = f'{crate}::' + '::'.join(modpath) if (public and modpath) else None

    def pick_super(name):
        # A `super::name` reference. Prefix `::` so a name that collides with a
        # type (e.g. crate `tensor` vs struct `tensor`) still resolves to the
        # crate, matching the `use ::tensor::...` convention already in tests/.
        if name in exports or not base:
            return f'::{crate}::{name}'
        return f'::{base}::{name}'

    def repl_super_single(m):
        return 'use ' + pick_super(m.group(1))
    body = re.sub(r'\buse\s+super\s*::\s*([A-Za-z_][A-Za-z0-9_]*)', repl_super_single, body)

    def repl_super_group(m):
        names = [n.strip() for n in m.group(1).split(',') if n.strip()]
        return '\n'.join('use ' + pick_super(n.split(' as ')[0].strip()) + ';' for n in names)
    body = re.sub(r'\buse\s+super\s*::\s*\{([^}]*)\}\s*;', repl_super_group, body)

    # Rewrite `crate::<mod-chain>::…` to `::<crate>::<public…>`. The public
    # surface is the crate root re-exports plus publicly reachable modules, so
    # a private module hop (`crate::error::opterror`) collapses to the exported
    # item (`::optim::opterror`).
    body = collapse_paths(body, crate, exports)

    # Remaining inline `super::NAME` references (not `use` statements).
    def repl_super_inline(m):
        return pick_super(m.group(1))
    body = re.sub(r'(?<![\w:])super::([A-Za-z_][A-Za-z0-9_]*)', repl_super_inline, body)

    lines = body.splitlines()
    nonblank = [ln for ln in lines if ln.strip()]
    if nonblank:
        indent = min(len(ln) - len(ln.lstrip()) for ln in nonblank)
        lines = [(ln[indent:] if len(ln) >= indent else ln) for ln in lines]
    return '\n'.join(lines).strip('\n')


def main(argv):
    for crate in argv[1:]:
        sources = sorted(glob.glob(f'crates/{crate}/src/**/*.rs', recursive=True))
        exports = root_exports(crate)
        mods = module_paths(crate)
        parts = []
        processed = 0
        for src in sources:
            text = open(src).read()
            if '#[cfg(test)]' not in text:
                continue
            public, modpath = mods.get(src, (False, []))
            mangled = '_'.join(modpath) if modpath else 'root'
            for name, body in extract_blocks(text):
                modname = mangled if name == 'tests' else f'{mangled}_{name}'
                parts.append(f'/// Tests relocated from `{src}`.\nmod {modname} {{\n'
                             + rewrite(body, crate, exports, public, modpath).rstrip() + '\n}\n')
                processed += 1
        if not parts:
            print(f'{crate}: no inline test blocks found')
            continue
        out = HEADER.format(crate=crate) + '\n'.join(parts)
        open(f'tests/{crate}units.rs', 'w').write(out + '\n')
        print(f'{crate}: wrote tests/{crate}units.rs ({processed} module(s))')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

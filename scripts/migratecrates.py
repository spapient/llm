#!/usr/bin/env python3
"""Drive the mechanical black-box migration for fully-clean crates.

For each crate: generate `tests/<crate>units.rs`, compile it, and — only if it
compiles — strip the inline `#[cfg(test)]` blocks from `crates/<crate>/src` and
drop the crate's entries from `scripts/srctestbaseline.txt`. On a compile
failure the generated file is removed and the sources are left untouched (that
crate needs a designed public entry point, handled by hand).

Usage: python3 scripts/migratecrates.py <crate> [<crate> ...]
"""
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rustscan  # noqa: E402


def strip_inline(crate):
    """Remove each inline test module span; other content is preserved."""
    removed = []
    for path in sorted(glob.glob(f'crates/{crate}/src/**/*.rs', recursive=True)):
        text = open(path).read()
        if '#[cfg(test)]' not in text:
            continue
        open(path, 'w').write(_rustscan.strip_tests(text).rstrip('\n') + '\n')
        removed.append(path)
    return removed


def tighten_baseline(crate):
    p = 'scripts/srctestbaseline.txt'
    lines = [l for l in open(p).read().splitlines() if not l.startswith(f'crates/{crate}/')]
    open(p, 'w').write('\n'.join(sorted(lines)) + '\n')
    return len(lines)


def main(argv):
    for crate in argv[1:]:
        gen = subprocess.run(['python3', 'scripts/migrateinline.py', crate],
                             capture_output=True, text=True)
        print(gen.stdout.strip())
        target = f'{crate}units'
        comp = subprocess.run(['cargo', 'test', '-p', 'llm', '--test', target,
                               '--locked', '--no-run'],
                              capture_output=True, text=True)
        if comp.returncode != 0:
            os.remove(f'tests/{crate}units.rs')
            errs = [l for l in comp.stderr.splitlines() if l.startswith('error')]
            print(f'  SKIP {crate}: does not compile through the public API ({len(errs)} errors)')
            for e in errs[:6]:
                print(f'      {e}')
            continue
        removed = strip_inline(crate)
        remaining = tighten_baseline(crate)
        print(f'  OK {crate}: stripped {len(removed)} source file(s); baseline now {remaining}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

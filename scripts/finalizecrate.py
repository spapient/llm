#!/usr/bin/env python3
"""Strip inline `#[cfg(test)]` from a crate's src and drop it from the ratchet.

Used after a suite has been hand-migrated and verified to pass. The source is
only touched once the caller has confirmed the replacement tests pass.

Usage: python3 scripts/finalizecrate.py <crate> [<crate> ...]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _rustscan  # noqa: E402


def main(argv):
    for crate in argv[1:]:
        removed = []
        for path in sorted(glob.glob(f'crates/{crate}/src/**/*.rs', recursive=True)):
            text = open(path).read()
            if '#[cfg(test)]' not in text:
                continue
            open(path, 'w').write(_rustscan.strip_tests(text).rstrip('\n') + '\n')
            removed.append(path)
        p = 'scripts/srctestbaseline.txt'
        lines = [l for l in open(p).read().splitlines() if not l.startswith(f'crates/{crate}/')]
        open(p, 'w').write('\n'.join(sorted(lines)) + '\n')
        print(f'{crate}: stripped {len(removed)} file(s); baseline now {len(lines)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

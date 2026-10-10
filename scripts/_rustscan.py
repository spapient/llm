#!/usr/bin/env python3
"""Lexical helpers for relocating `#[cfg(test)]` modules.

Brace matching must ignore braces inside string literals, raw strings, char
literals, and comments (JSON fixtures and char literals otherwise desynchronize
the scan). These helpers are shared by `migrateinline.py` and
`migratecrates.py`.
"""
import re


def _skip_raw_string(text, i):
    # assumes text[i] == 'r'
    j = i + 1
    while j < len(text) and text[j] == '#':
        j += 1
    if j >= len(text) or text[j] != '"':
        return None
    hashes = j - (i + 1)
    j += 1
    closer = '"' + '#' * hashes
    k = text.find(closer, j)
    if k == -1:
        return len(text)
    return k + len(closer)


def _skip_string(text, i):
    # assumes text[i] == '"'
    j = i + 1
    while j < len(text):
        c = text[j]
        if c == '\\':
            j += 2
            continue
        if c == '"':
            return j + 1
        j += 1
    return len(text)


def _skip_char(text, i):
    # assumes text[i] == "'"; handles escapes, otherwise a lifetime
    if i + 1 < len(text) and text[i + 1] == '\\':
        j = i + 2
        while j < len(text) and text[j] != "'":
            j += 1
        return j + 1
    if i + 2 < len(text) and text[i + 2] == "'":
        return i + 3
    return i + 1  # lifetime


def block_end(text, open_index):
    """Return the index just past the `}` matching `text[open_index] == '{'`."""
    depth = 0
    i = open_index
    n = len(text)
    while i < n:
        c = text[i]
        if c == '/' and i + 1 < n and text[i + 1] == '/':
            j = text.find('\n', i)
            i = n if j == -1 else j
            continue
        if c == '/' and i + 1 < n and text[i + 1] == '*':
            j = text.find('*/', i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == 'r' and (i + 1 < n and text[i + 1] in '#"'):
            nxt = _skip_raw_string(text, i)
            if nxt is not None:
                i = nxt
                continue
        if c == '"':
            i = _skip_string(text, i)
            continue
        if c == "'":
            i = _skip_char(text, i)
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError('unbalanced braces')


TEST_MOD = re.compile(r'#\[cfg\(test\)\]\s*\n\s*mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{')


def test_blocks(text):
    """Yield (name, body) for each `#[cfg(test)] mod <name> { ... }`."""
    out = []
    for m in TEST_MOD.finditer(text):
        brace = text.index('{', m.start())
        end = block_end(text, brace)
        out.append((m.group(1), text[brace + 1:end - 1]))
    return out


def strip_tests(text):
    """Remove every `#[cfg(test)] mod` span; other content is preserved."""
    while True:
        m = TEST_MOD.search(text)
        if not m:
            return text
        brace = text.index('{', m.start())
        end = block_end(text, brace)
        while end < len(text) and text[end] in ' \t\r\n':
            end += 1
        text = text[:m.start()].rstrip('\n') + '\n' + text[end:]


if __name__ == '__main__':
    import sys
    print(test_blocks(open(sys.argv[1]).read()))

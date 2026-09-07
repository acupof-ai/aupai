#!/usr/bin/env python3
"""Language ID + structural syntax filter for C, JavaScript and Java (e1, task 59).

    python3 datagen/code_lang_validate.py --selftest
    python3 datagen/code_lang_validate.py --histogram <shard.jsonl> [--limit N]

WHAT THIS IS AND IS NOT. `ast.parse` gives the Python builders a real parser: accept means
"the grammar accepts this file". THERE IS NO EQUIVALENT HERE. No C, JS or Java parser is
importable in this environment (pycparser, esprima, javalang, tree_sitter all absent), and a
pod restart loses pip packages, so a dependency would make the corpus unbuildable on the
machine that builds it. What this module does instead is LEX and check structure:

  1. strip comments and string/char literals with a language-aware scanner,
  2. require every bracket to balance on the stripped text,
  3. require language-specific structural evidence to survive the stripping.

So `validate` REFUSES more than a parser would and ACCEPTS things a parser would reject
(a balanced file with a bad expression inside passes). It is a language identifier with a
coarse well-formedness filter, and the retention numbers it produces must be read that way.
Calling it a validator is the name the task used; the docstring is where the limit lives.

WHY THE STRIPPING IS THE WHOLE JOB. Counting braces on raw source is not a weak signal, it
is a wrong one: `printf("}")`, a regex `/[{]/`, a `// }` comment and a Java `'}'` char all
carry an unbalanced brace that no counter can see through. Measured on the known-answer set,
the stripper is what separates every true accept from every true reject.

PLACEMENT (4c ruling 2026-09-07). Beside the builders, like `ast.parse` -- NOT in
filters/PIPELINE_FILTERS. `datagen/corpus_fingerprint.py:53` scopes that tuple to the three
garbage files on purpose: adding filters/secrets.py on 2026-09-06 moved fp_filters and
staled four stage-2 domains though no shard byte would have changed. A build that uses this
module records `module_sha256` in its own stats instead, so new shards carry their producer
and no existing domain's fingerprint moves.

# restartable: pure function of a string, no state, no I/O outside --histogram's reads.
"""
import argparse
import hashlib
import json
import os
import re
import sys

#: Reject before lexing. A doc this short cannot carry the structural evidence below, and
#: accepting it would inflate retention with fragments.
MIN_CHARS = 40


def module_sha256(path=None):
    """This file's content hash -- what a build stamps to record which validator produced it.

    sha256 of the file, the same basis filters_fp uses per member, so the two are comparable
    even though this module is deliberately outside PIPELINE_FILTERS.
    """
    p = path or os.path.abspath(__file__)
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _strip(text, lang):
    """Comments and literals -> spaces, everything else kept at its original offset.

    Same length in, same length out, so an offset in the stripped text still points at the
    source. Returns (stripped, ok) where ok is False for an UNTERMINATED string or comment --
    that is itself a syntax verdict and the caller must not treat it as merely "no evidence".

    One scanner for all three languages, differing in three flags, because they share C's
    lexical structure. The differences that matter:
      - JS has regex literals (`/re/g`), and telling `/` division from `/` regex-start needs
        the previous significant token: after a value (`)`, `]`, identifier, number) it is
        division, otherwise a regex. This is the one place a wrong guess silently eats code.
      - JS has template literals with `${}` interpolation, which nest.
      - Java and JS have `'c'` char/string literals; C has `'c'` char literals. All three
        take backslash escapes, so `'\\''` must not end the literal.
    """
    js = lang == "js"
    out = list(text)
    i, n = 0, len(text)
    prev = ""            # last significant char, for the JS regex-vs-division decision
    # Template literals interleave literal text with real code and NEST, so one boolean cannot
    # track them: `a ${ {x:1}.x } b` needs the outer braces blanked, the inner ones kept, and
    # the trailing " b" treated as literal again. The stack holds "tmpl" (inside literal text)
    # and ("interp", depth) frames; `depth` is the code-brace depth at which the interpolation
    # opened, which is how its closing `}` is told from a `}` in the code inside it.
    stack = []
    depth = 0

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if stack and stack[-1] == "tmpl":
            if c == "\\":
                blank(i, i + 2)
                i += 2
                continue
            if c == "`":
                blank(i, i + 1)
                stack.pop()
                i += 1
                prev = "x"
                continue
            if c == "$" and nxt == "{":
                blank(i, i + 2)
                stack.append(("interp", depth))
                i += 2
                continue
            blank(i, i + 1)
            i += 1
            continue

        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue
        if c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            if j < 0:
                return "".join(out), False
            blank(i, j + 2)
            i = j + 2
            continue
        if js and c == "/" and prev not in (")", "]", "}", "") and not (
                prev.isalnum() or prev == "_" or prev == "$"):
            # Regex literal. A newline before the closing slash means it was division after
            # all (no JS regex spans a line), so bail rather than eat the rest of the file.
            j, closed = i + 1, False
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "\n":
                    break
                if text[j] == "[":                      # a class can hold an unescaped /
                    while j < n and text[j] != "]":
                        j += 2 if text[j] == "\\" else 1
                if text[j] == "/":
                    closed = True
                    break
                j += 1
            if closed:
                blank(i, j + 1)
                i = j + 1
                prev = "/"
                continue
        if js and c == "`":
            blank(i, i + 1)
            stack.append("tmpl")
            i += 1
            continue
        if c in "\"'":
            q, j = c, i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == q:
                    if "\n" in text[i:j]:
                        # An unterminated single/double quote: C, JS and Java all forbid a raw
                        # newline inside one, so this is a syntax verdict, not a long string.
                        return "".join(out), False
                    blank(i, j + 1)
                    i = j + 1
                    break
                j += 1
            else:
                return "".join(out), False
            prev = "x"
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            if stack and isinstance(stack[-1], tuple) and stack[-1][1] == depth:
                # The interpolation's own closing brace: blank it, and the template text
                # resumes where it left off.
                blank(i, i + 1)
                stack.pop()
                i += 1
                continue
            depth -= 1
        if not c.isspace():
            prev = c
        i += 1
    if stack:
        return "".join(out), False       # a template literal was never closed
    return "".join(out), True


def _balanced(stripped):
    """Every bracket closes, in order. Runs on STRIPPED text or it is meaningless."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in stripped:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


#: Structural evidence, matched on STRIPPED text so a mention inside a comment or string
#: cannot vote. Each language needs one HARD marker (a construct the others do not share) or
#: two SOFT ones (shared with a sibling, decisive together).
_HARD = {
    "c": [
        re.compile(r"^\s*#\s*(?:include|define|ifndef|pragma)\b", re.M),
        re.compile(r"\b(?:struct|union|enum)\s+\w+\s*\{"),
        re.compile(r"\b(?:size_t|uint\d+_t|int\d+_t|FILE|va_list)\b"),
    ],
    "js": [
        re.compile(r"\b(?:function\s*\*?\s*\w*\s*\(|=>\s*[\{\(]|=>\s*\w)"),
        re.compile(r"\b(?:const|let)\s+[\w{\[]"),
        re.compile(r"\b(?:require\s*\(|module\.exports|export\s+(?:default|const|function)"
                   r"|import\s+.+\s+from\s)"),
        re.compile(r"\b(?:async\s+function|await\s+\w|typeof\s+\w|=== |!== )"),
    ],
    "java": [
        re.compile(r"^\s*package\s+[\w.]+\s*;", re.M),
        re.compile(r"^\s*import\s+(?:static\s+)?[\w.]+(?:\.\*)?\s*;", re.M),
        re.compile(r"\b(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?"
                   r"(?:class|interface|enum|void|[A-Z]\w*)"),
        re.compile(r"@(?:Override|Test|SuppressWarnings|Deprecated|Entity|Autowired)\b"),
        re.compile(r"\bSystem\.(?:out|err)\.print"),
    ],
}

#: Shared with a sibling and therefore never decisive alone. ONE IDIOM PER ENTRY, not an
#: alternation: `soft >= 2` is meant to say "two independent idioms agree", and packing
#: `var x` and `console.log` into one regex made a file carrying both count once. Measured:
#: the `var total = 0; ... console.log(total)` case scored soft=1 and was refused.
_SOFT = {
    "c": [re.compile(r"\b(?:void|char|unsigned|static)\s+\*?\w+\s*\("),
          re.compile(r"\bmalloc\s*\(|\bfree\s*\("),
          re.compile(r"\bprintf\s*\(|\bfprintf\s*\(|\bsprintf\s*\("),
          re.compile(r"->\s*\w"),
          re.compile(r"^\s*(?:int|void|char|float|double|long)\s+\w+\s*\([^;]*\)\s*\{", re.M)],
    "js": [re.compile(r"\bvar\s+\w"),
           re.compile(r"\bconsole\.(?:log|error|warn)\s*\("),
           re.compile(r"\bJSON\.(?:parse|stringify)\s*\("),
           re.compile(r"\bnew\s+Promise\b|\.then\s*\("),
           re.compile(r"\bdocument\.\w|\bwindow\.\w"),
           re.compile(r"\bfor\s*\(\s*var\s+\w+\s*=")],
    "java": [re.compile(r"\bnew\s+[A-Z]\w*\s*(?:<[^;{}]*>)?\s*\("),
             re.compile(r"\b(?:String|List|Map|ArrayList|HashMap)\s*(?:<[^;{}]*>)?\s+\w+\s*[=;]"),
             re.compile(r"\bthis\.\w+\s*="),
             re.compile(r"\bextends\s+[A-Z]\w*|\bimplements\s+[A-Z]\w*")],
}

#: A marker of a language we are NOT collecting. Present -> refuse, because a mixed or
#: mislabelled doc costs more than a missed one: code_rp1t is PHP 32% by facts/data_quality,
#: and PHP/C#/TS all share enough C syntax to pass the checks above.
#:
#: C++ IS IN THIS LIST AND IS NOT A BUG (4c 2026-09-07 relayed 3b's earlier reading that a
#: real C parser rejects C++ at ~5.1% of rows, so C++ needs its own lane). Measured here
#: before it was added: `#include <iostream>` + `std::cout` scored as 'c' hard=1 soft=1 --
#: C++ passes every C check, because C++ IS almost a superset of C's lexical surface. So a
#: C lane with no C++ detector silently mixes the two and its retention number describes
#: neither language. The other two C++ probes misrouted worse: a `class` body matched
#: foreign:python (`class Foo {` then a `public:` line ending in a colon) and `namespace ns {`
#: matched foreign:csharp -- refused, but for a reason that names the wrong language, which
#: is what an explicit marker fixes. Opening the C++ lane is a separate decision with its own
#: denominator; until then C++ is refused BY NAME so the histogram shows its size.
_FOREIGN = [
    # Before python/csharp: those two patterns fire on C++ constructs, so ordering decides
    # which name the histogram reports and the specific one has to win.
    (re.compile(r"#\s*include\s*<(?:iostream|vector|string|map|memory|algorithm|set|"
                r"unordered_map|sstream|fstream|utility|functional|thread|mutex)>"), "cpp"),
    (re.compile(r"\bstd::\w|\btemplate\s*<\s*(?:typename|class)\b|\bnamespace\s+\w+\s*\{"
                r"|\bpublic\s*:|\bprivate\s*:|\bprotected\s*:|::\w+\s*\("), "cpp"),
    (re.compile(r"<\?php|\$\w+\s*=|\becho\s+[\"'$]"), "php"),
    (re.compile(r"^\s*(?:using\s+System|namespace\s+\w+\s*\{)", re.M), "csharp"),
    (re.compile(r"^\s*(?:def|class)\s+\w+.*:\s*$", re.M), "python"),
    (re.compile(r"^\s*(?:func\s+\w+|package\s+main\s*$)", re.M), "go"),
    (re.compile(r"\bfn\s+\w+\s*\(|\blet\s+mut\b|::<"), "rust"),
    (re.compile(r"^\s*(?:end|def\s+\w+[?!]?\s*$|require\s+['\"])", re.M), "ruby"),
    (re.compile(r":\s*(?:string|number|boolean)\s*[;,)=]|\binterface\s+\w+\s*\{[^}]*:\s*\w+"),
     "typescript"),
]


def validate(text):
    """(lang, reason) -- lang is 'c'/'js'/'java' when accepted, None when refused.

    `reason` always says why: the rejection cause, or the evidence that carried an accept.
    A caller building a histogram groups on it, which is why the strings are stable and
    short rather than descriptive sentences.
    """
    if not text or len(text) < MIN_CHARS:
        return None, "too_short"
    if "\x00" in text:
        return None, "binary"

    for rx, name in _FOREIGN:
        if rx.search(text):
            return None, f"foreign:{name}"

    best = None
    for lang in ("c", "js", "java"):
        stripped, ok = _strip(text, lang)
        if not ok:
            continue
        if not _balanced(stripped):
            continue
        hard = sum(1 for rx in _HARD[lang] if rx.search(stripped))
        soft = sum(1 for rx in _SOFT[lang] if rx.search(stripped))
        if hard == 0 and soft < 2:
            continue
        # Score before tie-breaking: Java and C share `int f(){}`, JS and Java share `class`.
        # A hard marker outweighs any number of soft ones, so a `package x;` beats two shared
        # idioms, and `#include` beats a Java-shaped method signature.
        score = hard * 10 + soft
        if best is None or score > best[0]:
            best = (score, lang, f"hard={hard} soft={soft}")

    if best is None:
        # Distinguish the two refusals a caller can act on: unbalanced-or-unterminated is a
        # syntax verdict, no-evidence is a language verdict. They need different fixes
        # upstream, and one bucket would hide which.
        for lang in ("c", "js", "java"):
            stripped, ok = _strip(text, lang)
            if ok and _balanced(stripped):
                return None, "no_language_evidence"
        return None, "unbalanced_or_unterminated"
    return best[1], best[2]


# ---------------------------------------------------------------------------------------
# The known-answer set: 20 documents per language, each a (text, expected_lang) pair.
# Built to be DECIDABLE BY A READER, so a disagreement between this table and validate() is
# a defect in one of them and never a matter of taste. Every language's 20 hold 14 accepts
# and 6 refusals, and the refusals are the cases that separate this from brace-counting:
# a brace inside a string, inside a comment, inside a char literal, inside a JS regex.

_KA_C = [
    ('#include <stdio.h>\nint main(void) { printf("hello\\n"); return 0; }\n', "c"),
    ('#include <stdlib.h>\nstruct node { int v; struct node *next; };\n'
     'struct node *mk(int v) { struct node *n = malloc(sizeof *n); n->v = v; return n; }\n', "c"),
    ('#define MAX(a,b) ((a) > (b) ? (a) : (b))\nstatic int clamp(int x) { return MAX(x, 0); }\n', "c"),
    ('#include <string.h>\nsize_t len(const char *s) { size_t n = 0; while (*s++) n++; return n; }\n', "c"),
    ('#ifndef H_GUARD\n#define H_GUARD\ntypedef struct { int x, y; } Point;\n#endif\n', "c"),
    ('#include <stdio.h>\nint main(void) { printf("brace } in a string"); return 0; }\n', "c"),
    ('/* a comment with an unbalanced } brace */\n#include <stdio.h>\nvoid f(void) { }\n', "c"),
    ("#include <stdio.h>\nchar c = '}';\nint main(void) { return 0; }\n", "c"),
    ('#include <stdio.h>\nvoid f(void) { char *s = "quote \\" and } here"; (void)s; }\n', "c"),
    ('static unsigned crc(const unsigned char *p, size_t n) {\n'
     '    unsigned c = 0xffffffffu;\n'
     '    for (size_t i = 0; i < n; i++) { c ^= p[i]; }\n    return c;\n}\n', "c"),
    ('#include <stdio.h>\nint main(int argc, char **argv) {\n'
     '    for (int i = 0; i < argc; i++) printf("%s\\n", argv[i]);\n    return 0;\n}\n', "c"),
    ('enum color { RED, GREEN, BLUE };\nstatic enum color pick(int i) { return (enum color)(i % 3); }\n', "c"),
    ('#include <stdio.h>\nFILE *open_or_die(const char *p) {\n'
     '    FILE *f = fopen(p, "r");\n    if (!f) { perror(p); }\n    return f;\n}\n', "c"),
    ('union u { int i; float f; };\nstatic int as_int(float f) { union u v; v.f = f; return v.i; }\n', "c"),
    # --- refusals ---
    ('#include <stdio.h>\nint main(void) { printf("oops");\n', None),          # unbalanced
    ('#include <stdio.h>\nint main(void) { char *s = "never closed;\n return 0; }\n', None),
    ('/* comment never closed\n#include <stdio.h>\nint main(void) { return 0; }\n', None),
    ("int x = 1;\n", None),                                                     # too short
    ('The quick brown fox jumps over the lazy dog, repeatedly and at length.\n', None),
    ('<?php\n$x = 1;\necho "hello";\n?>\n', None),                              # foreign
]

_KA_JS = [
    ('function add(a, b) { return a + b; }\nmodule.exports = { add };\n', "js"),
    ('const fs = require("fs");\nconst data = fs.readFileSync("x.json", "utf8");\n'
     'console.log(JSON.parse(data));\n', "js"),
    ('export default function App() { return null; }\nexport const VERSION = "1.0";\n', "js"),
    ('const xs = [1, 2, 3].map((x) => x * 2);\nconsole.log(xs);\n', "js"),
    ('async function main() { const r = await fetch("/api"); return r.json(); }\nmain();\n', "js"),
    ('let re = /[{]/g;\nfunction count(s) { return (s.match(re) || []).length; }\n', "js"),
    ('const s = "a } brace inside a string";\nconsole.log(s.length);\n', "js"),
    ('// a comment with a } brace\nconst x = 1;\nconsole.log(x);\n', "js"),
    ('const t = `template with ${1 + 1} and a } brace`;\nconsole.log(t);\n', "js"),
    ('class Queue { constructor() { this.xs = []; } push(x) { this.xs.push(x); } }\n'
     'module.exports = Queue;\n', "js"),
    ('const p = new Promise((res) => { setTimeout(res, 10); });\np.then(() => console.log("ok"));\n', "js"),
    ('var total = 0;\nfor (var i = 0; i < 10; i++) { total += i; }\nconsole.log(total);\n', "js"),
    ('import { readFile } from "fs/promises";\nconst txt = await readFile("a.txt", "utf8");\n'
     'console.log(txt);\n', "js"),
    ('const o = { a: 1, b: 2 };\nconst { a, b } = o;\nconsole.log(a === 1, b !== 3);\n', "js"),
    # --- refusals ---
    ('function add(a, b) { return a + b;\nmodule.exports = add;\n', None),      # unbalanced
    ('const s = "never closed;\nconsole.log(s);\n', None),
    ('/* unclosed\nconst x = 1;\nconsole.log(x);\n', None),
    ('let x = 1;\n', None),                                                     # too short
    ('Just some prose about JavaScript that contains no code at all whatsoever.\n', None),
    ('interface User { name: string; age: number; }\nconst u: User = { name: "a", age: 1 };\n', None),
]

_KA_JAVA = [
    ('package com.example;\n\npublic class Main {\n'
     '    public static void main(String[] args) { System.out.println("hi"); }\n}\n', "java"),
    ('import java.util.List;\nimport java.util.ArrayList;\n\n'
     'public class Box { private List<String> xs = new ArrayList<>(); }\n', "java"),
    ('package a.b;\n\npublic interface Repo { String find(long id); }\n', "java"),
    ('import org.junit.Test;\n\npublic class T {\n    @Test\n'
     '    public void works() { assert 1 == 1; }\n}\n', "java"),
    ('package x;\n\npublic enum Color { RED, GREEN, BLUE }\n', "java"),
    ('package x;\n\npublic class S {\n'
     '    public String brace() { return "a } inside a string"; }\n}\n', "java"),
    ('package x;\n// a comment with a } brace\npublic class C { void f() { } }\n', "java"),
    ("package x;\n\npublic class Ch { char c = '}'; void f() { } }\n", "java"),
    ('package com.example.service;\n\nimport java.util.Map;\nimport java.util.HashMap;\n\n'
     'public class Cache {\n    private final Map<String, Object> m = new HashMap<>();\n'
     '    public Object get(String k) { return m.get(k); }\n}\n', "java"),
    ('package x;\n\npublic class P {\n    private String name;\n'
     '    public String getName() { return name; }\n'
     '    public void setName(String n) { this.name = n; }\n}\n', "java"),
    ('import static org.junit.Assert.assertEquals;\n\npublic class AT {\n'
     '    public void t() { assertEquals(1, 1); }\n}\n', "java"),
    ('package x;\n\npublic abstract class Base {\n    protected abstract void run();\n'
     '    public final void go() { run(); }\n}\n', "java"),
    ('package x;\n\npublic class E extends RuntimeException {\n'
     '    public E(String m) { super(m); }\n}\n', "java"),
    ('package x;\n\npublic class Loop {\n    public void f(java.util.List<String> xs) {\n'
     '        for (String s : xs) { System.err.println(s); }\n    }\n}\n', "java"),
    # --- refusals ---
    ('package x;\n\npublic class Main {\n    public static void main(String[] a) {\n', None),
    ('package x;\npublic class S { String s = "never closed;\n }\n', None),
    ('/* unclosed\npackage x;\npublic class C { }\n', None),
    ('package x;\n', None),                                                     # too short
    ('A paragraph discussing the Java programming language, its history and use.\n', None),
    ('using System;\nnamespace N {\n    class P { static void Main() { } }\n}\n', None),
]

KNOWN_ANSWERS = {"c": _KA_C, "js": _KA_JS, "java": _KA_JAVA}

#: C++ MUST BE REFUSED BY NAME, not merely refused. Kept out of KNOWN_ANSWERS because that
#: table is 3 x 20 by specification and these are a different assertion: each of these was
#: measured MISCLASSIFIED before the cpp entries were added to _FOREIGN -- the first as
#: ('c', 'hard=1 soft=1'), and the other three refused under a wrong name (python, csharp)
#: or for no reason at all. A reject for the wrong reason is a wrong histogram, and the
#: histogram is what the C++-lane decision will be made from.
_KA_CPP = [
    ('#include <iostream>\nint main() { std::cout << "hi" << std::endl; return 0; }\n',
     "was classified 'c' before the cpp markers existed"),
    ('#include <vector>\nclass Foo { public:\n  Foo() {}\n  std::vector<int> xs;\n};\n',
     "was foreign:python -- `class Foo {` then a line ending in `:`"),
    ('template <typename T>\nT max2(T a, T b) { return a > b ? a : b; }\n',
     "was no_language_evidence"),
    ('namespace ns {\n  int f(int x) { return x * 2; }\n}\n',
     "was foreign:csharp"),
    ('#include <memory>\nstd::unique_ptr<int> mk() { return std::make_unique<int>(3); }\n',
     "std:: alone must be enough"),
]


def _selftest():
    fails = []
    for lang, cases in KNOWN_ANSWERS.items():
        assert len(cases) == 20, f"{lang}: known-answer set is {len(cases)}, must be 20"
        n_acc = sum(1 for _, want in cases if want is not None)
        assert n_acc == 14, f"{lang}: {n_acc} accepts, the set is specified as 14 accept + 6 reject"
        for i, (text, want) in enumerate(cases):
            got, reason = validate(text)
            if got != want:
                fails.append(f"{lang}[{i}]: want {want!r}, got {got!r} ({reason}) :: "
                             f"{text.splitlines()[0][:56]!r}")
    # THE STRIPPER IS THE LOAD-BEARING PART, asserted directly rather than only through the
    # table above: a brace inside a string, a comment, a char literal or a JS regex must not
    # reach the balance check. Without this, a future edit could make the table pass for the
    # wrong reason (e.g. every case accepted on soft markers alone).
    for lang, src, why in (
            ("c", 'printf("}");', "string"),
            ("c", "/* } */", "block comment"),
            ("c", "// }", "line comment"),
            ("c", "char c = '}';", "char literal"),
            ("js", "let re = /[}]/;", "regex literal"),
            ("js", "const t = `a } b`;", "template literal"),
    ):
        stripped, ok = _strip(src, lang)
        if not ok or "}" in stripped:
            fails.append(f"stripper leaves a brace from a {why} in {lang}: {stripped!r}")
    # And the negative control for the stripper: a REAL brace must survive it, or the
    # balance check would accept everything.
    stripped, ok = _strip("int main(void) { return 0; }", "c")
    if not ok or stripped.count("}") != 1:
        fails.append(f"stripper ate a real brace: {stripped!r}")

    # The regex-vs-division decision, the one place a wrong guess silently eats code.
    stripped, ok = _strip("const x = (a) / b / c;\nconst y = 1;", "js")
    if not ok or "y" not in stripped:
        fails.append(f"division after ) read as a regex, ate the rest: {stripped!r}")

    # C++ must be refused BY NAME. See _KA_CPP: every one of these was misclassified before
    # the cpp markers went in, one of them AS C, which would have put C++ rows in the C lane
    # and made its retention number describe a mixture.
    for src, why in _KA_CPP:
        lang, reason = validate(src)
        if lang is not None or reason != "foreign:cpp":
            fails.append(f"cpp not refused by name ({why}): got {lang!r} / {reason!r} :: "
                         f"{src.splitlines()[0][:52]!r}")
    # NEGATIVE CONTROL for the cpp markers: plain C must still pass. Without this, a cpp
    # pattern broad enough to eat the C lane entirely would leave every assertion above green.
    for src in ('#include <stdio.h>\nint main(void) { printf("hi\\n"); return 0; }\n',
                '#include <stdlib.h>\nstruct n { int v; };\n'
                'struct n *mk(int v) { struct n *p = malloc(sizeof *p); p->v = v; return p; }\n'):
        lang, reason = validate(src)
        if lang != "c":
            fails.append(f"cpp markers ate plain C: got {lang!r} / {reason!r}")

    # THE SOFT THRESHOLD IS `>= 2` AND NOTHING ABOVE TESTS IT. Measured: no case in
    # KNOWN_ANSWERS has hard=0 with soft=1 -- the sets cluster at soft=0 (refused for having
    # no evidence at all) or soft>=2, so relaxing the rule to `soft < 1` left all 60 green.
    # That mutation is the difference between "two independent idioms agree" and "one shared
    # idiom is enough", which is the whole reason soft markers are separated from hard ones.
    # These two cases straddle it: both are hard=0, one has a single shared idiom and must be
    # refused, the other has two and must be kept.
    _one_soft = "    this.count = 0;\n    other.reset();\n    int n = 0;\n"
    _two_soft = "    this.count = 0;\n    String label = other.name();\n    int n = 0;\n"
    for src, want, why in ((_one_soft, None, "one shared idiom must not carry a language"),
                           (_two_soft, "java", "two independent shared idioms must")):
        _st, _ok = _strip(src, "java")
        _h = sum(1 for rx in _HARD["java"] if rx.search(_st))
        _s = sum(1 for rx in _SOFT["java"] if rx.search(_st))
        got, reason = validate(src)
        if _h != 0:
            fails.append(f"soft-threshold case is no longer hard=0 (got {_h}), so it stopped "
                         f"testing the threshold: {src.splitlines()[0][:44]!r}")
        elif got != want:
            fails.append(f"soft threshold: {why} -- want {want!r}, got {got!r} "
                         f"({reason}, hard={_h} soft={_s})")

    if fails:
        print(f"code_lang_validate: {len(fails)} FAIL(s)")
        for f in fails:
            print("  " + f)
        return 1
    total = sum(len(v) for v in KNOWN_ANSWERS.values())
    print(f"code_lang_validate selftest OK: {total} known answers (3 x 20, 14 accept + 6 reject "
          f"each), {len(_KA_CPP)} C++ refused-by-name cases with a plain-C negative control, "
          f"6 stripper cases, a real-brace control and the JS division control. "
          f"module_sha256={module_sha256()}")
    return 0


def _histogram(path, limit=None):
    """Reject histogram and language mix over a shard, printed, nothing written.

    The language MIX is printed beside the rejects because retention per language is
    conditional on how much of each language the source holds: '42% of C kept' means
    nothing without the C denominator, and code_rp1t is one mixed domain (facts
    data_quality: PHP 32%, Java 8/50, C/C++ 8/50, JS 6/50 on the t24 hand-read).
    """
    counts, langs, n = {}, {}, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if limit is not None and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                counts["unparsable_row"] = counts.get("unparsable_row", 0) + 1
                n += 1
                continue
            n += 1
            lang, reason = validate(row.get("content", "") or "")
            if lang:
                langs[lang] = langs.get(lang, 0) + 1
            else:
                counts[reason] = counts.get(reason, 0) + 1
    kept = sum(langs.values())
    print(f"{path}: {n} doc(s) read")
    print(f"  kept {kept} ({100.0 * kept / n:.1f}% of {n})" if n else "  kept 0")
    for k in sorted(langs, key=lambda x: -langs[x]):
        print(f"    {k:22} {langs[k]:6}  ({100.0 * langs[k] / n:.1f}% of {n})")
    print("  rejects:")
    for k in sorted(counts, key=lambda x: -counts[x]):
        print(f"    {k:22} {counts[k]:6}  ({100.0 * counts[k] / n:.1f}% of {n})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="code_lang_validate")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--histogram", metavar="SHARD")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.histogram:
        return _histogram(a.histogram, a.limit)
    ap.error("nothing to do: pass --selftest or --histogram SHARD")


if __name__ == "__main__":
    sys.exit(main())

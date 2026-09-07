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
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time

#: Reject before lexing. A doc this short cannot carry the structural evidence below, and
#: accepting it would inflate retention with fragments.
MIN_CHARS = 40

#: How much a hard marker is worth against a soft one when two languages both have evidence.
#: Its only requirement is `_TIE_WEIGHT > max(len(_SOFT[lang]))`: above that bound every value
#: gives the same answers, below it soft markers can outvote a hard one and the docstring's
#: rule becomes false. Asserted in the selftest, both directions.
_TIE_WEIGHT = 10


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
#: C++ USED TO BE IN THIS LIST AND IS NOW A LANE (4c ruling 2026-09-07). The history is the
#: argument for how it is detected, so it stays recorded. Measured before C++ had any marker:
#: `#include <iostream>` + `std::cout` scored as 'c' hard=1 soft=1 -- C++ passes every C
#: check, because C++ IS almost a superset of C's lexical surface. Two other C++ probes
#: misrouted worse, a `class` body matching foreign:python and `namespace ns {` matching
#: foreign:csharp: refused, but under a name that is not C++, which is a wrong histogram.
#: Refusing it BY NAME first is what made its size visible, and the size settled the lane:
#: 12.7% of code_rp1t on both shard 000 and shard 117 against C's 3.6% and 2.8%.
#: _CPP_MARKERS below now carries the detection; nothing named cpp remains in this list.

_FOREIGN = [
    # GO BY ANY PACKAGE NAME, not just `package main`. The first version was
    # `package\s+main\s*$`, which is the ONE Go package name that a library file never uses:
    # measured in the 50-doc hand-read of the no_language_evidence bucket, 3 of 9 unclassified
    # docs were Go (`package terrors`, `package fake`, `package github`) and all three fell
    # through to no_language_evidence instead of being named. Java also opens with `package`,
    # so the discriminator is the SHAPE of what follows: Go has no semicolon after it, and its
    # import block is parenthesised.
    (re.compile(r"^\s*package\s+\w+\s*$(?!\s*;)", re.M), "go"),
    (re.compile(r"^\s*import\s+\(\s*$", re.M), "go"),
    (re.compile(r"^\s*func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s*)?\w+\s*\(", re.M), "go"),
    # Scala: no marker at all in the first version, and a `package x` + `class ... extends`
    # file reached no_language_evidence. Ordered AFTER go's `package \w+$` cannot claim it,
    # because Scala's package line is also semicolon-free -- so Scala needs a marker that go's
    # does not match, which is why these are the def/val/case forms rather than the package
    # line. Kept narrow: `object X {` and `def f(...): T =` are not Java, C or JS shapes.
    (re.compile(r"^\s*object\s+\w+\s*(?:extends\b|\{)|^\s*case\s+class\s+\w+\s*\(", re.M),
     "scala"),
    (re.compile(r"\bdef\s+\w+\s*(?:\[[^\]]*\])?\s*\([^)]*\)\s*:\s*\w+\s*=|\bval\s+\w+\s*[:=]"),
     "scala"),
    # A dotted package line with NO semicolon is Scala's or Kotlin's, never Java's -- java's
    # is `package com.example;`. Measured: the hand-read's scala doc opened
    # `package com.xantoria.flippy.utils` and the two markers above miss it, because it has no
    # object/case-class and its `extends Suites(` is a class, not a def. The negative control
    # for this one is java's own package line, which the trailing `(?!;)` excludes.
    (re.compile(r"^\s*package\s+\w+(?:\.\w+)+\s*$(?!\s*;)", re.M), "scala"),
    (re.compile(r"<\?php|\$\w+\s*=|\becho\s+[\"'$]"), "php"),
    # C# LOST ITS `namespace \w+ {` MARKER when the C++ lane opened, because that line is
    # C++'s too, character for character, and C# was matching first: `namespace ns {` with
    # std:: below it was named foreign:csharp. There is no discriminator on that line alone,
    # so the marker moved to the two forms C++ does not share -- `using System` and the
    # capital-M entry point, which is C#'s where Java's and C++'s are lowercase. The cost is
    # a C# file with neither, which now reaches the C++ lane; it needs a cpp marker to be
    # claimed there, and `namespace` alone is not enough since the lane also demands
    # C-family evidence.
    (re.compile(r"^\s*using\s+System\b|\bConsole\.(?:Write|WriteLine|ReadLine)\s*\(|"
                r"\bstatic\s+void\s+Main\s*\(", re.M), "csharp"),
    # `class X ... :` with a BRACE on the line is C++'s access specifier, not python's block
    # opener. Measured: `class Base { public:` matched the first version of this pattern and
    # was named foreign:python, which refused a C++ file under a language it is not.
    #
    # AND `\n` MUST BE EXCLUDED TOO, which `[^{}]` alone does not do. Measured on a bullet
    # physics header in the shard117 strided sample: `class btCollisionShape;` on one line
    # paired with a doc comment 300 characters later ending "There are 3 types of rigid
    # bodies:", and the whole span matched. A python block opener is ONE line by definition,
    # so the character class has to say so. Over both strided samples (n=4000) the fix moves
    # 9 docs out of foreign:python -- 7 to foreign:ruby, 1 to foreign:typescript, and 1 C++
    # header into the cpp lane -- and changes no python answer in the known-answer set.
    (re.compile(r"^\s*(?:def|class)\s+\w+[^{}\n]*:\s*$", re.M), "python"),
    # A python SCRIPT need not define anything: `import sys` then `sys.argv[1]` was one of the
    # nine, and the def/class pattern above cannot see it. `import x` with no semicolon and no
    # brace anywhere is not a C/JS/Java shape -- the balance check runs later, so this only has
    # to be more specific than "the word import".
    (re.compile(r"^\s*(?:import\s+\w+|from\s+\w[\w.]*\s+import\s+\w)\s*$", re.M), "python"),
    (re.compile(r"\bfn\s+\w+\s*\(|\blet\s+mut\b|::<"), "rust"),
    (re.compile(r"^\s*(?:end|def\s+\w+[?!]?\s*$|require\s+['\"])", re.M), "ruby"),
    (re.compile(r":\s*(?:string|number|boolean)\s*[;,)=]|\binterface\s+\w+\s*\{[^}]*:\s*\w+"),
     "typescript"),
]

#: 44's hand-read of code_rp1t_dd09/b2 (docs/audits/code_rp1t_dd09_b2_hand_read.md @ ec74649b)
#: names six rules for the non-code tail: ~22% of dd09 and ~14% of b2. Four of the six are
#: exactly what my own 50-doc read of the no_language_evidence bucket found -- HTML 15/50,
#: XML/config 9/50, CSS 5/50, license-only 1/50 -- so they are implemented here rather than as
#: a second pass. They are reported as their OWN reasons, not folded into no_language_evidence:
#: that bucket was 17% of the domain and turned out to be several unrelated things, which is
#: what made it unquotable as a single cause.
#:
#: Rules 4 and 5 of the six (license-only, empty class bodies) 44 marks OPTIONAL. License-only
#: is implemented because it is cheap and unambiguous. Empty class bodies are NOT: they are
#: syntactically valid code in the target languages, 44's note says they are "lower-value but
#: not harmful", and refusing them needs a real parse to do correctly -- exactly what this
#: module does not have. Refusing them by regex would drop real classes with short bodies.
_NONCODE = [
    (re.compile(r"^\s*<!DOCTYPE\s+html|^\s*<html[\s>]|<(?:div|span|body|head|table)\b", re.I),
     "noncode:html"),
    # A GENERATOR META LINE, not the bare word. `javadoc` and `doxygen` appear in ordinary
    # comments inside real source -- measured: a hand-written Java class whose comment says
    # "Generated docs live in javadoc/" was refused as noncode:html by the bare-word version.
    # What identifies a generated doc page is the generator's own banner: the word next to
    # "Generated by", or an LCOV/coverage header, at the START of a line.
    (re.compile(r"^\s*(?:<!--\s*)?(?:Generated by (?:javadoc|Doxygen|LCOV|jGuru)|"
                r"LCOV - code coverage report|Doxygen \d)", re.I | re.M), "noncode:html"),
    (re.compile(r"auto-generated|autogenerated|Generated by|do not (?:edit|modify)|@generated"
                r"|WSDL2Java|Propel", re.I), "noncode:generated"),
    # CSS BY A DECLARATION ON ITS OWN LINE, not by a brace with a colon somewhere after it.
    # The first version was `\{[^{}]*[\w-]+\s*:\s*[^;{}]+;` and `[^{}]` matches NEWLINES, so
    # `namespace {` opened it and any later colon-and-semicolon line closed it, spanning
    # arbitrary C++. Measured by 3b over both strided samples: 29 of the 81 docs the C++
    # evidence question was about were sitting in this bucket and NONE was CSS -- chromium
    # `namespace {` with a LazyInstance, a Greenplum GPOS file, a `#pragma once` CUDA header,
    # Lucene++'s `const int32_t Token::MIN_BUFFER_SIZE = 10;`. `try {` plus any `x: y;` line
    # catches Java and C# the same way. Same defect as the python marker above, in a second
    # rule; both were found by reading the buckets rather than by any test.
    #
    # Two forms, because CSS is written both ways and the naive `\n` exclusion breaks the
    # multi-line one -- which is the common one:
    #   selector {                        selector { prop: value; ... }
    #     prop: value;
    # The value excludes `:` and `=`, which is what separates a declaration from
    # `base::LazyInstance<Foo>::Leaky g = X;` -- that line has both and CSS has neither.
    # Blank lines and a comment may sit between the selector and its first declaration; the
    # block-comment form was found by a mutation that reached farther than this rule and
    # was right to on that one input.
    #
    # THE SELECTOR BODY EXCLUDES `\n` EXPLICITLY, and that is not cosmetic. The first version
    # used `[\w\s.#,:...]*`, and `\s` CONTAINS `\n` -- the same defect this rule is being
    # fixed for, reintroduced inside the fix. With a newline-crossing, self-overlapping
    # character class in front of an anchored tail, one real 68KB C# document in the 4000
    # sample sent it into catastrophic backtracking: `validate()` did not return in 20
    # seconds, against 0.002s for this form. A corpus filter that hangs on one document in
    # 4000 is worse than the misclassification it was fixing, and no known-answer case is
    # large enough to show it -- only running the corpus does.
    (re.compile(r"^[ \t]*(?:[.#][\w-]+|[\w-]+)[^\n{}]*\{[ \t]*$"
                r"\n(?:[ \t]*(?://.*|/\*.*?\*/)?[ \t]*\n)*"
                r"^[ \t]*[-\w]+[ \t]*:[ \t]*[^;{}\n:=]+;[ \t]*$", re.M), "noncode:css"),
    (re.compile(r"^[ \t]*(?:[.#][\w-]+|[\w-]+)[\w\s.#,-]*\{[ \t]*[-\w]+[ \t]*:[ \t]*"
                r"[^;{}\n:=]+;", re.M), "noncode:css"),
    (re.compile(r"@media\b|@import\s+url\(|!important\s*;"), "noncode:css"),
    (re.compile(r"^\s*<\?xml|^\s*<(?:project|configuration|beans|manifest|RelativeLayout)\b"
                r"|xmlns(?::\w+)?\s*=", re.I | re.M), "noncode:config"),
    (re.compile(r"^\s*FROM\s+\w+[:/]|^\.PHONY\b|^\s*cask\s+['\"]", re.M), "noncode:config"),
]

#: License-only is a LENGTH-CONDITIONED rule, not a pattern: a copyright header on real code
#: is normal and must not be refused. 44's rule says under 500 chars with no functional code.
_LICENSE_RE = re.compile(r"Licensed under|Copyright\s*\(c\)|Apache License|MIT License", re.I)
_LICENSE_MAX_CHARS = 500

#: C++ IS A LANE (4c ruling 2026-09-07), and the hard part is C, not C++.
#:
#: C++ is almost a lexical superset of C, so "looks like C++" is nearly free while "is C and
#: NOT C++" is the real discrimination. The asymmetry decides the design: a doc matching ANY
#: of these is C++, and C is what survives with no C++ marker at all. That is why C++ does not
#: go through the hard/soft tie-break -- a C++ file carrying two C soft markers would outscore
#: its single C++ marker under that machinery, which is precisely the mixing the lane exists
#: to end. One marker is enough because none of these appears in valid C: `std::`, `template
#: <typename`, a namespace block, an access specifier, `nullptr`, a `new`/`delete` expression,
#: base-class inheritance, `operator<<`, `using namespace`, and catch-by-reference are all
#: C++-only constructs, not idioms C shares.
_CPP_MARKERS = [
    re.compile(r"#\s*include\s*<(?:iostream|vector|string|map|memory|algorithm|set|"
               r"unordered_map|unordered_set|sstream|fstream|utility|functional|thread|"
               r"mutex|array|tuple|optional|variant|type_traits|chrono|numeric|deque)>"),
    re.compile(r"\bstd::\w"),
    re.compile(r"\btemplate\s*<\s*(?:typename|class)\b"),
    re.compile(r"^\s*namespace\s+\w*\s*\{", re.M),
    re.compile(r"^\s*(?:public|private|protected)\s*:\s*$", re.M),
    re.compile(r"\b(?:nullptr|constexpr|noexcept|decltype|static_cast|dynamic_cast)\b"),
    re.compile(r"\bdelete\s*\[\s*\]\s*\w"),
    re.compile(r"\b(?:class|struct)\s+\w+\s*:\s*(?:public|private|protected)\s+\w"),
    re.compile(r"\boperator\s*(?:<<|>>|==|!=|\[\])|\bcout\s*<<|\bcerr\s*<<"),
    re.compile(r"\busing\s+namespace\s+\w"),
    re.compile(r"\bcatch\s*\(\s*(?:const\s+)?\w+\s*[&*]"),
]

#: A C++ file still has to BE a file: the lane needs C-family evidence too, or a fragment of
#: prose mentioning std:: would enter it. Reuses the C markers, since C++ shares them by
#: construction -- that sharing is the reason the lane exists and also what makes it cheap.
_CPP_MIN_C_EVIDENCE = 1


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

    # NON-CODE BEFORE FOREIGN-LANGUAGE. An HTML page quoting a shell command, or a Maven POM
    # naming a Java class, would otherwise be reported under the language it mentions. The
    # more specific verdict is "this is not source at all".
    if len(text) < _LICENSE_MAX_CHARS and _LICENSE_RE.search(text):
        return None, "noncode:license"
    for rx, name in _NONCODE:
        if rx.search(text):
            return None, name

    for rx, name in _FOREIGN:
        if rx.search(text):
            return None, f"foreign:{name}"

    # THE C++ LANE, decided BEFORE the C/JS/Java scoring and not inside it. A C++ file matches
    # the C markers by construction, so letting it into the tie-break would let two C soft
    # markers outweigh one C++ marker -- the mixing this lane was opened to end. Any C++
    # marker plus C-family evidence is C++; C is what reaches the loop below with none.
    _cpp_stripped, _cpp_ok = _strip(text, "c")
    if _cpp_ok and _balanced(_cpp_stripped):
        _cpp = sum(1 for rx in _CPP_MARKERS if rx.search(_cpp_stripped))
        if _cpp:
            _c_hard = sum(1 for rx in _HARD["c"] if rx.search(_cpp_stripped))
            _c_soft = sum(1 for rx in _SOFT["c"] if rx.search(_cpp_stripped))
            if _c_hard + _c_soft >= _CPP_MIN_C_EVIDENCE:
                return "cpp", f"cpp={_cpp} c_evidence={_c_hard + _c_soft}"
            # A C++ marker with NO C-family evidence at all is prose mentioning std::, not a
            # translation unit. Named rather than dropped into no_language_evidence, which is
            # the bucket that turned out to be several unrelated things.
            return None, "cpp_marker_without_code"

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
        #
        # THE WEIGHT IS A BOUND, NOT A TUNED NUMBER. Any value strictly greater than the
        # largest soft-marker set gives IDENTICAL answers, because one extra hard marker adds
        # w while the soft term can differ by at most max(len(_SOFT[lang])). So 10 and 100 are
        # the same function and no input distinguishes them; only w <= 6 changes an answer,
        # and the selftest's tie-breaker case is red for w = 1. _SOFT_CAP below is the real
        # invariant, asserted rather than left in prose (3b, PR #13: the weight survived two
        # mutations because I audited thresholds and this is not one).
        score = hard * _TIE_WEIGHT + soft
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

#: The nine docs the 50-doc hand-read of the no_language_evidence bucket could not classify,
#: as (text, expected reason). Five were REAL CODE in languages meant to be refused by name,
#: reaching no_language_evidence instead -- the same misattribution as C++ before it had a
#: marker, one level down. Each is the shape actually seen in code_rp1t, not an invention.
_KA_MISSED = [
    ("package terrors\n\n// SimpleError provides simple type-based error handling.\n"
     "type SimpleError struct {\n\tMessage string\n}\n", "foreign:go",
     "go by a package name that is not `main`"),
    ("package fake\n\nimport (\n\tunversioned \"k8s.io/kubernetes/pkg/client\"\n"
     "\trestclient \"k8s.io/kubernetes/pkg/client/restclient\"\n)\n", "foreign:go",
     "go by its parenthesised import block"),
    ("package com.xantoria.flippy.utils\n\nimport org.scalatest.Suites\n\n"
     "class UtilsSuite extends Suites(\n  new NetSpec\n)\n", "foreign:scala",
     "scala: package with no semicolon, class extends"),
    ("import sys\nimport os\nimport time\n\nspeed = sys.argv[1]\nt = sys.argv[2]\n",
     "foreign:python", "a python SCRIPT that defines nothing"),
    ("<!DOCTYPE html>\n<html>\n<head><title>x</title></head>\n<body>\n"
     "<div id=\"main\">hello</div>\n</body>\n</html>\n", "noncode:html", "44 rule 1"),
    ("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
     "<RelativeLayout xmlns:android=\"http://schemas.android.com/apk/res/android\">\n"
     "</RelativeLayout>\n", "noncode:config", "44 rule 6"),
    ("/* Semantic UI 2.6.0 */\n.ui.loader {\n  position: absolute;\n  top: 50%;\n"
     "  display: none;\n}\n@media only screen and (max-width: 767px) {\n  .ui { margin: 0; }\n}\n",
     "noncode:css", "44 rule 3"),
    # CSS WITHOUT AN AT-RULE, which is what actually tests the block rule: the case above
    # also carries `@media`, so the second css pattern answers it and the first could be
    # deleted with nothing going red. Both forms are here because the rule has two branches
    # and the multi-line one is the branch the pre-fix rule MISSED -- it required the
    # declaration to follow the brace on the same logical span, and matched a C++ file
    # instead.
    ("html,body {\n    font-family: sans-serif;\n    -ms-text-size-adjust: 100%;\n}\n"
     "#columns #newsletter_block_left .form-group {\n    margin-bottom: 0;\n}\n",
     "noncode:css", "multi-line CSS with no at-rule: the block branch, alone"),
    (".btn { color: red; }\n.btn-primary { background: #fff; border: 1px solid #ccc; }\n",
     "noncode:css", "one-line CSS with no at-rule: the inline branch, alone"),
    # A COMMENT BETWEEN THE SELECTOR AND ITS FIRST DECLARATION. Found by a mutation that
    # widened the gap back to `[^{}]*` and survived: chasing what distinguished it turned up
    # 19 documents and this real-CSS shape, which the narrow gap MISSED. The mutation was
    # right about this input and wrong about the other 18, so the fix is to widen the gap to
    # comments only, not to accept the mutant.
    #
    # ONE BLOCK, and that is load-bearing. The first version had a second `.b { padding: 1px; }`
    # block for realism, and that block has no comment, so it matched on its own and a
    # mutation removing `/*...*/` from the gap SURVIVED -- the case could not test the thing
    # it was added for. A fixture for a specific construct must contain that construct and
    # no easier path to the same answer.
    (".a {\n  /* the first rule */\n  color: red;\n  margin: 0;\n}\n",
     "noncode:css", "a block comment between selector and declaration"),
    ("/*\n * This file was auto-generated by WSDL2Java. Do not modify.\n */\n"
     "package x;\npublic class Stub { public void f() { } }\n", "noncode:generated",
     "44 rule 2 -- and it must beat the java markers below it"),
    ("Licensed under the Apache License, Version 2.0 (the \"License\");\n"
     "you may not use this file except in compliance with the License.\n",
     "noncode:license", "44 rule 4, length-conditioned"),
]

#: NEGATIVE CONTROLS for the widened patterns (4c required these with the widening). Each is
#: real C/JS/Java that a too-greedy version of a new pattern would eat. Measured reasons the
#: patterns are dangerous: go's `package \w+$` is one lookahead away from Java's `package x;`;
#: scala's `val`/`def` forms sit inside JS and Java text; python's bare `import x` line is one
#: word away from Java's `import java.util.List;`; the css rule matches `selector { prop: v; }`
#: which is the shape of a Java annotation body and a JS object literal.
_KA_NOT_EATEN = [
    ("package com.example;\n\npublic class Main {\n"
     "    public static void main(String[] args) { System.out.println(\"hi\"); }\n}\n",
     "java", "java's package line ends in a semicolon; go's must not match it"),
    ("import java.util.List;\nimport java.util.Map;\n\npublic class Box {\n"
     "    private List<String> xs = null;\n    public List<String> get() { return xs; }\n}\n",
     "java", "java imports end in a semicolon; python's bare-import must not match"),
    ("const cfg = { color: \"red\", margin: 0 };\nfunction f() { return cfg.color; }\n"
     "module.exports = f;\n", "js",
     "a JS object literal is `name: value` inside braces, which is the css shape"),
    ("#include <stdio.h>\nstruct opts { int verbose; };\n"
     "int main(void) { struct opts o = {0}; printf(\"%d\\n\", o.verbose); return 0; }\n",
     "c", "a C struct initialiser is also brace-and-colon-free but css-adjacent"),
    ("package x;\n\npublic class Doc {\n"
     "    /** Generated docs live in javadoc/. This class is hand-written. */\n"
     "    public void f() { }\n}\n", "java",
     "the word javadoc in a COMMENT must not make a real class noncode:html"),
    # THE MULTI-LINE PYTHON MATCH, from the shard117 strided sample. `class X;` forward
    # declarations followed anywhere later by a line ending in a colon matched python's
    # block-opener pattern, because `[^{}]*` spans newlines. This exact document was
    # foreign:python before the `\n` exclusion. It is here rather than in _KA_MISSED because
    # it is a NEGATIVE control -- the defect was a refusal eating real code, and this is the
    # code it ate.
    ("#ifndef RIGIDBODY_H\n#define RIGIDBODY_H\n\n#include \"btTransform.h\"\n\n"
     "class btCollisionShape;\nclass btMotionState;\nclass btTypedConstraint;\n\n"
     "extern btScalar gDeactivationTime;\n\n"
     "///The btRigidBody is the main class for rigid body objects.\n"
     "///There are 3 types of rigid bodies:\n"
     "class btRigidBody : public btCollisionObject {\n"
     "    btScalar m_inverseMass;\n"
     "public:\n"
     "    void setMassProps(btScalar mass);\n};\n#endif\n", "cpp",
     "a C++ header whose forward declarations paired with a later prose colon"),
    # THE SAME DEFECT IN THE CSS RULE (3b, hand-read of 29 docs, none of them CSS). Each of
    # these was noncode:css before the rule became line-anchored, and each is a different
    # opener: an anonymous namespace, a do-block, and a try-block, which is the one that
    # reaches Java and C# rather than only C++.
    ("namespace {\n\nbase::LazyInstance<Foo>::Leaky g_foo = LAZY_INSTANCE_INITIALIZER;\n\n"
     "}\n\nnamespace blink {\nvoid f() { g_foo.Get(); }\n}\n", "cpp",
     "an anonymous namespace plus a scoped static must not read as a CSS block"),
    ("#include <zlib.h>\nvoid deflate_all(z_stream *s) {\n  do {\n"
     "    s->avail_out = kChunkSize;\n    deflate(s, 0);\n  } while (s->avail_in);\n}\n", "c",
     "a do-block plus a `->` assignment is brace-then-colon-free but was css-adjacent"),
    ("package x;\n\npublic class A {\n    public void f() {\n        try\n        {\n"
     "            int clientIndex = 0;\n            System.out.println(clientIndex);\n"
     "        } catch (Exception e) { }\n    }\n}\n", "java",
     "a try-block opener: the css defect reached Java, not only C++"),
    # THE INLINE CSS BRANCH NEEDS THE SAME `:=` EXCLUSION AS THE BLOCK ONE, and nothing in
    # 4000 docs proves it: dropping `:=` from the inline pattern alone changed no answer on
    # the corpus and survived mutation. It is not equivalent, though -- this one line differs
    # -- so the case is written rather than the mutation left alive. A scoped initialiser on
    # one line inside a brace is the shape: CSS values contain neither `::` nor `=`.
    ("#include <memory>\nnamespace n { base::Foo x = Y; }\n"
     "void run() { n::x.reset(); }\n", "cpp",
     "a one-line scoped initialiser must not read as an inline CSS declaration"),
]

KNOWN_ANSWERS = {"c": _KA_C, "js": _KA_JS, "java": _KA_JAVA}

#: The C++ lane's own 20, same specification as the other three: 14 accept, 6 reject. The six
#: rejects are chosen for what they separate, not for variety -- four are the C++/C boundary
#: from the C side (a doc that must NOT enter this lane), which is the only boundary that
#: matters here, because every other language is refused before the lane is reached.
_KA_CPP_LANE = [
    ('#include <iostream>\nint main() { std::cout << "hi" << std::endl; return 0; }\n', "cpp"),
    ('#include <vector>\nclass Foo {\npublic:\n    Foo() {}\n    std::vector<int> xs;\n};\n', "cpp"),
    ('#include <memory>\nstd::unique_ptr<int> mk() { return std::make_unique<int>(3); }\n', "cpp"),
    # HEADER-LESS, so the C evidence is SOFT ONLY (hard=0 soft=2). Kept deliberately: with
    # every accept carrying an `#include`, narrowing the evidence test to `_c_hard >= 1`
    # changed no answer and survived mutation. A .cpp body with its declarations in a header
    # is the common shape this covers.
    ('template <typename T>\nvoid swap2(T &a, T &b) { T t = a; a = b; b = t; }\n'
     'void run() { int x = 1, y = 2; swap2(x, y); }\n', "cpp"),
    ('#include <map>\nnamespace ns {\n    std::map<int, int> m;\n    int f(int x) { return m[x]; }\n}\n',
     "cpp"),
    ('#include <stdio.h>\nclass Base { public:\n    virtual ~Base() {}\n};\n'
     'class D : public Base { public:\n    void f() { printf("d\\n"); }\n};\n', "cpp"),
    ('#include <stdexcept>\nvoid f(int x) {\n    if (x < 0) throw std::runtime_error("neg");\n}\n',
     "cpp"),
    ('#include <stdio.h>\nusing namespace std;\nint main() { printf("x"); return 0; }\n', "cpp"),
    ('#include <stdlib.h>\nint *mk(int n) { int *p = new int[n]; return p; }\n'
     'void del(int *p) { delete [] p; }\n', "cpp"),
    ('#include <stdio.h>\nstruct P { int x; };\n'
     'P *mk() { P *p = nullptr; p = new P(); return p; }\n', "cpp"),
    ('#include <algorithm>\n#include <vector>\n'
     'int top(std::vector<int> &xs) { std::sort(xs.begin(), xs.end()); return xs.back(); }\n',
     "cpp"),
    ('#include <fstream>\nvoid w(const char *p) {\n    std::ofstream f(p);\n    f << "x";\n}\n',
     "cpp"),
    ('#include <stdio.h>\ntry_block: ;\nvoid f() {\n    try { g(); }\n'
     '    catch (const std::exception &e) { printf("%s", e.what()); }\n}\n', "cpp"),
    ('#include <string>\nclass S {\nprivate:\n    std::string s;\npublic:\n'
     '    constexpr int n() const noexcept { return 1; }\n};\n', "cpp"),
    # --- rejects: four are the C boundary, which is the only one that matters ---
    ('#include <stdio.h>\nint main(void) { printf("hello\\n"); return 0; }\n', "c"),
    ('#include <stdlib.h>\nstruct node { int v; struct node *next; };\n'
     'struct node *mk(int v) { struct node *n = malloc(sizeof *n); n->v = v; return n; }\n', "c"),
    ('#include <string.h>\nsize_t len(const char *s) { size_t n = 0; while (*s++) n++; return n; }\n',
     "c"),
    ('#ifndef H\n#define H\ntypedef struct { int x, y; } Point;\nvoid move(Point *p, int dx);\n#endif\n',
     "c"),
    # prose that names std:: but is not a translation unit
    ('The std::vector container grows amortised O(1), which is why the guide recommends it\n'
     'over a raw array in most application code.\n', None),
    # C++ that does not balance
    ('#include <iostream>\nint main() { std::cout << "oops";\n', None),
]

#: ONE MINIMAL DOCUMENT PER CPP MARKER, each written so that marker is the ONLY one that
#: fires. The 20 above do not cover the markers: deleting each marker in turn and re-running,
#: 5 of 11 survived (the include list, `namespace X {`, a lone `public:`, `cout <<`, and the
#: by-reference catch) because every case carrying them also carried another. A marker no
#: case decides alone is an untested marker, and the shard histogram is read as if all 11
#: were load-bearing. Indexed by position, and the selftest asserts the list covers every
#: index -- so a twelfth marker with no document here is red, not silently uncovered.
#:
#: `void pad(void) { }` appears in several: these documents are minimal by construction and
#: five of them landed under MIN_CHARS, where validate() answers too_short before any marker
#: is consulted. The padding is a second C soft marker, never a cpp one.
_KA_CPP_MARKER = [
    ('#include <vector>\nint main(void) { return 0; }\n', "a C++-only standard header"),
    ('void f(void) { std::string s = "hello"; }\nvoid pad(void) { }\n',
     "the std:: namespace qualifier"),
    ('template <typename T>\nvoid f(T a) { (void)a; }\n', "a template head"),
    ('namespace ns {\nvoid f(void) { }\nvoid pad(void) { }\n}\n', "a namespace block"),
    ('class A {\npublic:\n    void f(void) { }\n};\n', "an access specifier on its own line"),
    ('void f(void) { int *p = nullptr; (void)p; }\n', "a C++11 keyword"),
    ('void f(int *p) { delete [] p; }\nvoid pad(void) { }\n', "array delete"),
    ('class D : public B {\n    void f(void) { }\n};\n', "inheritance with an access specifier"),
    ('void f(void) { cout << "x" << "y"; }\nvoid pad(void) { }\n',
     "a stream insertion into cout"),
    ('using namespace std;\nvoid f(void) { }\nvoid pad(void) { }\n', "a using-directive"),
    ('void g(void);\nvoid f(void) { try { g(); } catch (const E &e) { } }\n',
     "a catch by reference"),
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

    # THE C++ LANE. Kept out of KNOWN_ANSWERS because its 20 are not shaped like the other
    # three: the six non-cpp expectations are four PLAIN C docs, and those are accepts of the
    # module under a different name, not refusals. Folding them in would make the
    # `14 accepts` assertion above read 18 and hide that difference.
    _cpp_acc = sum(1 for _, want in _KA_CPP_LANE if want == "cpp")
    _cpp_c = sum(1 for _, want in _KA_CPP_LANE if want == "c")
    if len(_KA_CPP_LANE) != 20 or _cpp_acc != 14 or _cpp_c != 4:
        fails.append(f"the cpp set is specified as 20 = 14 cpp + 4 plain-C + 2 refusals; it is "
                     f"{len(_KA_CPP_LANE)} = {_cpp_acc} + {_cpp_c} + "
                     f"{len(_KA_CPP_LANE) - _cpp_acc - _cpp_c}")
    for i, (src, want) in enumerate(_KA_CPP_LANE):
        lang, reason = validate(src)
        if lang != want:
            fails.append(f"cpp[{i}]: want {want!r}, got {lang!r} ({reason}) :: "
                         f"{src.splitlines()[0][:52]!r}")
    # THE SCOPE CONTROL, which the four plain-C entries above cannot give on their own: they
    # prove specific C docs survive, not that the lane is bounded. If a cpp marker matched
    # everything, those four would still be red -- but a marker set that eats HALF the C lane
    # leaves them green while the shard numbers move. Asserted on the C known-answer set,
    # which is the largest population of real C the module has.
    _ate = [i for i, (src, want) in enumerate(_KA_C)
            if want == "c" and validate(src)[0] == "cpp"]
    if _ate:
        fails.append(f"the cpp lane reaches into the C known-answer set at indices {_ate}; "
                     f"it must be C++-specific, not C-family")
    # AND THE TEETH, in the other direction: with the markers removed, every one of the 14
    # C++ docs must fall OUT of the lane. A lane that would answer 'cpp' anyway is not being
    # tested by the 14 above.
    _saved = _CPP_MARKERS[:]
    try:
        del _CPP_MARKERS[:]
        _still = [i for i, (src, want) in enumerate(_KA_CPP_LANE)
                  if want == "cpp" and validate(src)[0] == "cpp"]
    finally:
        _CPP_MARKERS[:] = _saved
    if _still:
        fails.append(f"with _CPP_MARKERS emptied, cpp[{_still}] still answered 'cpp', so those "
                     f"cases do not test the markers")
    # AND THE EVIDENCE TEST IS A SUM, not `hard >= 1`. Every accept in the set carrying an
    # `#include` would let `_c_hard + _c_soft` be narrowed to `_c_hard` with no answer
    # changing -- measured, that mutation survived. At least one accept must reach the lane on
    # SOFT C evidence alone, and this asserts the fixture still does, rather than asserting on
    # its classification, which stays 'cpp' either way.
    _soft_only = [i for i, (src, want) in enumerate(_KA_CPP_LANE) if want == "cpp"
                  and sum(1 for rx in _HARD["c"] if rx.search(_strip(src, "c")[0])) == 0]
    if not _soft_only:
        fails.append("every cpp accept now has a hard C marker, so the lane's evidence test "
                     "could be narrowed to _c_hard with nothing going red -- one accept must "
                     "be header-less")
    # AND THE REFUSAL REASONS, which the table above cannot check because it compares only
    # the language. Measured: deleting the `cpp_marker_without_code` return let the prose case
    # fall through to no_language_evidence and every assertion stayed green -- both are None,
    # so the table is blind to which bucket the doc lands in. The bucket IS the deliverable
    # here: the shard histogram is what the lane's share is quoted from, and a reject under
    # the wrong name is a wrong histogram even when the accept/reject split is right.
    for src, want in (
            ('The std::vector container grows amortised O(1), which is why the guide '
             'recommends it\nover a raw array in most application code.\n',
             "cpp_marker_without_code"),
            ('#include <iostream>\nint main() { std::cout << "oops";\n',
             "unbalanced_or_unterminated"),
    ):
        _lang, _reason = validate(src)
        if _lang is not None or _reason != want:
            fails.append(f"cpp refusal reason: want {want!r}, got {_lang!r}/{_reason!r} :: "
                         f"{src.splitlines()[0][:48]!r}")

    # PER-MARKER COVERAGE. Each document must be answered 'cpp' AND must fire exactly the
    # marker at its own index -- the second half is what makes it a coverage test: a document
    # that also trips a neighbour cannot go red when its own marker is deleted, which is how
    # 5 of the 11 were uncovered while the 20 above were all green.
    if len(_KA_CPP_MARKER) != len(_CPP_MARKERS):
        fails.append(f"{len(_CPP_MARKERS)} cpp markers but {len(_KA_CPP_MARKER)} documents "
                     f"covering them; every marker needs one that fires it alone")
    for i, (src, why) in enumerate(_KA_CPP_MARKER[:len(_CPP_MARKERS)]):
        _st, _ok = _strip(src, "c")
        _hits = [j for j, rx in enumerate(_CPP_MARKERS) if rx.search(_st)]
        if _hits != [i]:
            fails.append(f"cpp marker {i} ({why}): its document fires markers {_hits}, so "
                         f"deleting marker {i} would not change its answer")
        elif validate(src)[0] != "cpp":
            fails.append(f"cpp marker {i} ({why}): {validate(src)!r}, want 'cpp'")

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

    # THE TIE-BREAKER WEIGHT, which is not a threshold and which my own threshold audit
    # missed (3b, PR #13 review). Sweeping every numeric comparison in the module, five of
    # seven were covered and the two survivors were both this weight: `hard * 10 -> hard * 1`
    # and `-> hard * 100` each left all 60 known answers green. The weight is live, not a
    # dead knob -- this input decides on it, measured: c is hard=1 soft=2 (w10 = 12, w1 = 3),
    # java is hard=0 soft=4 (w10 = 4, w1 = 4). Weight 10 answers 'c', weight 1 answers
    # 'java', both languages pass the evidence gate, so the weight ALONE picks the lane.
    _tie = ('#include <stdio.h>\nvoid run() {\n    String name = "x";\n'
            '    this.field = 1;\n    Thing t = new Thing();\n'
            '    List<String> xs = null;\n}\nclass A extends B { }\n')
    _sc = {}
    for _lang in ("c", "js", "java"):
        _st, _ok = _strip(_tie, _lang)
        _sc[_lang] = (sum(1 for rx in _HARD[_lang] if rx.search(_st)),
                      sum(1 for rx in _SOFT[_lang] if rx.search(_st)))
    if not (_sc["c"][0] >= 1 and _sc["java"][0] == 0 and _sc["java"][1] > _sc["c"][1]):
        # The case stopped straddling the weight, so asserting on its answer would assert
        # nothing. Louder than a silent pass: a fixture that no longer discriminates is the
        # shape that makes a green suite mean less than it reads.
        fails.append(f"the tie-breaker case no longer straddles the weight (c={_sc['c']}, "
                     f"java={_sc['java']}); it must be hard-for-c vs more-soft-for-java")
    else:
        got, reason = validate(_tie)
        if got != "c":
            fails.append(f"tie-breaker: one hard marker must outweigh more soft ones -- "
                         f"want 'c', got {got!r} ({reason}); c={_sc['c']} java={_sc['java']}")

    # AND THE ARITHMETIC THE DOCSTRING CLAIMS. "A hard marker outweighs any number of soft
    # ones" is true only while every language has fewer soft markers than the weight. js
    # already has 6 against a weight of 10. Someone adding four more inverts the rule with no
    # test going red -- the claim is in prose and prose cannot fail. Asserted on the marker
    # sets rather than on a classification, because that is where the invariant lives.
    _max_soft = max(len(v) for v in _SOFT.values())
    if _TIE_WEIGHT <= _max_soft:
        fails.append(f"_TIE_WEIGHT is {_TIE_WEIGHT} but some language has {_max_soft} soft "
                     f"markers, so soft markers can outvote a hard one and the docstring's "
                     f"rule is false -- raise the weight or split the marker")

    # THE NINE THE HAND-READ COULD NOT CLASSIFY, and 44's six rules folded in with them.
    # Reported under their own reason, never as no_language_evidence: that bucket was 17% of
    # the domain and turned out to be several unrelated things, which is exactly what made it
    # unquotable as one cause.
    for src, want, why in _KA_MISSED:
        lang, reason = validate(src)
        if lang is not None or reason != want:
            fails.append(f"missed-bucket case ({why}): want {want!r}, got "
                         f"{lang!r}/{reason!r} :: {src.splitlines()[0][:48]!r}")

    # AND THE NEGATIVE CONTROLS. A widened refusal pattern fails in the direction the
    # known-answer sets cannot see: it eats real code and the accept lanes shrink silently.
    # These are the specific collisions each new pattern is one character away from.
    for src, want, why in _KA_NOT_EATEN:
        lang, reason = validate(src)
        if lang != want:
            fails.append(f"a widened refusal ate real {want}: {why} -- got "
                         f"{lang!r}/{reason!r} :: {src.splitlines()[0][:48]!r}")

    # AND A RUNTIME BOUND, because none of the assertions above can see the failure that
    # nearly shipped: a character class containing `\s` in front of an anchored tail sent
    # the css rule into catastrophic backtracking on one real 68KB C# document, and
    # validate() did not return in 20 seconds. Every known-answer case is a few hundred
    # bytes, so all of them stayed green, and a corpus filter that hangs on 1 document in
    # 4000 is worse than the misclassification it was fixing.
    #
    # THE FIXTURE IS THE REAL DOCUMENT, delta-debugged from 68014 bytes to these 19 lines --
    # TWICE I replaced it with a synthetic one I invented from the shape I assumed was slow,
    # and both times the mutation reintroducing the bug SURVIVED because my reconstruction
    # ran in 0.000s. What matters is not describable in a sentence: a brace-opening line,
    # then `//` lines long enough and numerous enough that the gap alternation retries at
    # every one while the newline-crossing selector class re-anchors. Truncating every line
    # to 80 characters makes it fast; so does dropping lines. Keep the bytes.
    _hostile = (
        "    internal readonly partial struct ExpressionBinder\n"
        "    {\n"
        "        // \n"
        "        // \n"
        "        // Use assertions and naming guidelines to express the contract for methods. \n"
        "        // The most common issue is whether an argument may be null or not. If an \n"
        "        // argument may not be null, then the method must ASSERT that before any other \n"
        "        // code. If an argument may be null then the name of the argument should \n"
        "        // include 'Optional'. The exception to this rule is the input parse tree \n"
        "        // parameter. If the parse tree may be null, then the method name should \n"
        "        // include an 'Opt' suffix. For example bindArgumentList should really be \n"
        "        // named bindArgumentListOpt. Abbreviations should be avoided, but the 'Opt' \n"
        "        // suffix gets an exception because it is used consistently in the language \n"
        "        // \n"
        "        // \n"
        "        // \n"
        "        // Do not rely on the input parse tree being complete. Erroneous code may \n"
        "        // result in parse trees with required children missing, or with unexpected \n"
        "        // structure. Find out what the invariants are for the parse tree being \n"
        # The last four repeated, and that repetition is the assertion's margin. The
        # delta-debugged 19 lines run in 1.37s under the bad pattern -- UNDER the 2s bound,
        # so the mutation survived against them. Four more lines take it past 6s: the cost
        # is exponential in the comment-line count, which is exactly why a bound works here
        # and why the fixture must sit clear of it rather than at it.
        "        // \n"
        "        // Do not rely on the input parse tree being complete. Erroneous code may \n"
        "        // result in parse trees with required children missing, or with unexpected \n"
        "        // structure. Find out what the invariants are for the parse tree being \n")
    _t0 = time.monotonic()
    validate(_hostile)
    _dt = time.monotonic() - _t0
    if _dt > 2.0:
        fails.append(f"validate() took {_dt:.1f}s on a {len(_hostile)} byte document; a rule "
                     f"is backtracking exponentially. Look for a character class that "
                     f"contains \\s or . in front of an anchored tail")

    # AND THE REPORTER ITSELF, because it was broken for the length of one editing session
    # and nothing noticed. An edit indented `if fails:` into the timing branch above, so a
    # run with THREE real failures in `fails` printed OK and returned 0 -- every mutation in
    # a sweep came back green, including ones that had been red minutes earlier, and the
    # green looked like evidence. A selftest that cannot fail is worse than no selftest: it
    # launders every assertion above it. So the exit path is asserted here, on a deliberate
    # non-empty list, before the real one is consulted. Its output is discarded: a probe
    # that printed would itself read as a failure.
    _probe = io.StringIO()
    with contextlib.redirect_stdout(_probe):
        _rc = _report(["a deliberate failure, to prove the reporter reports"], _max_soft)
    if _rc != 1 or "FAIL" not in _probe.getvalue():
        print("code_lang_validate: 1 FAIL(s)")
        print(f"  _report() returned {_rc} and printed {_probe.getvalue()[:60]!r} for a "
              f"NON-EMPTY failure list -- every assertion in this selftest is unenforced "
              f"and any green above is meaningless")
        return 1
    _out = io.StringIO()
    with contextlib.redirect_stdout(_out):
        _rc = _report(fails, _max_soft)
    _text = _out.getvalue()
    print(_text, end="")
    # AND THE RETURN MUST AGREE WITH WHAT WAS PRINTED. `return 0` in place of this call
    # survives every assertion above -- it reports nothing and exits clean, and the hook
    # judges on the exit code, so a silently-passing selftest is indistinguishable from a
    # passing one. The summary line is the evidence that the reporter ran at all.
    if _rc == 0 and "selftest OK" not in _text:
        print("code_lang_validate: 1 FAIL(s)")
        print(f"  the selftest returned 0 without printing its summary ({_text[:60]!r}); "
              f"the reporter did not run and nothing above was enforced")
        return 1
    return _rc


def _report(fails, max_soft):
    """Print and return 1 if anything failed, else print the summary and return 0.

    Split out of _selftest so the exit path is a named function that can be called with a
    known-bad list -- see the assertion above. Inline, it was one indentation level away
    from being unreachable, and that is exactly what happened.
    """
    if fails:
        print(f"code_lang_validate: {len(fails)} FAIL(s)")
        for f in fails:
            print("  " + f)
        return 1
    total = sum(len(v) for v in KNOWN_ANSWERS.values())
    print(f"code_lang_validate selftest OK: {total} known answers (3 x 20, 14 accept + 6 reject "
          f"each), the C++ lane's own 20 (14 cpp + 4 plain-C boundary + 2 refusals) with a "
          f"scope control over the C set and an empty-marker teeth control, "
          f"6 stripper cases, a real-brace control, the JS division control, a soft-threshold "
          f"pair, the tie-breaker case and the _TIE_WEIGHT > max-soft invariant "
          f"({_TIE_WEIGHT} > {max_soft}). module_sha256={module_sha256()}")
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

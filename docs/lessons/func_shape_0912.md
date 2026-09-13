---
question: "Do the four gate code domains carry enough open-body/docstring/doctest and problem-to-solution-shaped Python to justify shifting the last-10% anneal toward them after the step6000 is_prime/HumanEval failure?"
status: measured
source: "scripts/audit_func_shape.py over a 20k stratified sample per domain; facts/data_quality.json#dq.func_shape.code_ultra_l3_noexec_dc (plus .code_ultra_l2_dc/.code_py_starcoder_dc/.code_keep_p1_dc); raw counts pod runs/func_shape_audit_0912.json (2026-09-12)"
---

# Function shape of the gate code domains, 2026-09-12

20,000 documents per domain, uniform across all shards. Docstring/doctest rates
are per function (ast); shape is per document. All rates in %.

| domain | funcs in sample | funcs w/ docstring | docstrings w/ `>>>` | funcs w/ doctest | problem→solution docs | raw-source docs | whole doc parses |
|---|--:|--:|--:|--:|--:|--:|--:|
| code_ultra_l3_noexec_dc | 51,698 | **53.39** | 2.34 | 1.25 | **96.34** | 3.67 | 47.60 |
| code_ultra_l2_dc | 75,712 | 23.72 | 1.58 | 0.38 | 0.10 | 99.90 | 95.75 |
| code_py_starcoder_dc | 117,005 | 24.87 | 1.57 | 0.39 | 0.10 | 99.91 | 100.00 |
| code_keep_p1_dc | 68,427 | 20.11 | 2.55 | 0.51 | 0.05 | 99.95 | 77.13 |

95% binomial half-widths: ≤0.14pp per document rate, ≤0.13pp per function
rate. The shape classifier is a regex heuristic, not a parser of intent; its
known-answer validation is the split in the table itself — the domain built
from exercise statements reads 96% problem→solution, the three raw-source
domains read ≤0.1%.

## Findings

1. **Problem→solution material exists in exactly one domain.** L3 is 96.3%
   imperative task statement followed by a complete implementation and assert
   tests — the HumanEval form. L2, starcoder and keep_p1 are 99.9% raw source
   files: no prompt, no test scaffolding. The gate failure (is_prime at
   step6000) is a prompt-to-implementation failure; only L3 trains that
   mapping.
2. **Docstring-bearing open bodies are 2.2-2.7x more common in L3** (53.4% of
   functions vs 20.1-24.9%). An L3 solution function typically restates the
   contract as a docstring, so L3 also carries the "read a spec, produce a
   body" shape even where the document prefix is ignored.
3. **Doctest-shaped code essentially does not exist anywhere.** 1.25% of L3
   functions and 0.38-0.51% of the others have any `>>>` line in their
   docstring; even conditional on having a docstring, only 1.6-2.6% do. There
   is no doctest-rich supply in the current corpus to shift toward; a weight
   change cannot buy doctest coverage that the bytes do not contain.
4. **L3's 47.6% whole-document parse rate is structural, not corruption.** L3
   documents begin with English prose, so the whole text is not Python; the
   suffix parser recovers implementations for the 96.3% shape rate. L2 95.7%,
   starcoder 100%, keep_p1 77.1% parse whole (keep_p1 contains rp1t snippets
   and non-Python rows).

## Recommendation (second, after the numbers)

The anneal already names L3 at 0.40, the largest single anneal weight, with L2
0.30 (`data/mix_v41_gate.json`). The audit supports keeping L3 as the dominant
anneal domain and, if the last-10% anneal composition is reopened, shifting
anneal weight from raw-source domains toward L3 within its supply limit
(26.70B packed tokens; the budget draws 8.1B main (0.30×27B) + 1.2B anneal
(0.40×3B) = 9.3B, ratio 0.348, so headroom is large; epochs must stay 1 so the
shift buys unique problems, not repeats). It does not support any move toward "doctest-shaped" code: that
shape is absent from all four domains. Beyond the existing eight domains,
test-bearing exercise data (more L3-style aggregate, not doctest mining of
raw source) is the only lever with a measured supply gap behind it.

## Method and limits

- Per-file reservoir (k=128, deterministic per-filename seed 7) in a 32-way
  process pool, one read over every shard; largest-remainder per-file quotas
  give exactly 20,000 uniformly sampled documents per domain.
- Functions: `ast.FunctionDef`/`AsyncFunctionDef`; whole-document parse first,
  longest parseable code suffix otherwise (L3 prose prefixes).
- Problem→solution: imperative task verb + target noun within the first 30
  lines, a `def` in the document, and `assert`/`# Test`/`def test_`. False
  positives bounded by the 0.05-0.10% raw-domain rates; the false-negative
  rate is unmeasured (L3 docs that phrase the instruction without the verb
  pattern), so 96.3% is a lower bound of the true exercise-shaped share.
- Doctest is a literal `>>>` line in the docstring; expected-output tests held
  outside docstrings (the L3 form, trailing `assert` blocks) are counted by
  the shape metric, not the doctest metric.

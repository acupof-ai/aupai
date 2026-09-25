---
question: On steps 30000/32000/34000, how do the HumanEval failures break down by cause, which of them can SFT A fix, and does the 280-token cap cost passes?
status: measured
source: runs/heval_fail_classes_30_32_34k.jsonl (492 rows, one per task x step) + the 11-shard preds per step; classifier re-runs judge()'s exec on the saved completion
---

# HumanEval failure classes at step 30000/32000/34000

164 tasks x 3 steps = 492 rows, ckpt `ckpt_v41_ced_0923.pt`, pass@1 greedy, `max_new=280`.
Raw table with the completion text, one row per task x step: `runs/heval_fail_classes_30_32_34k.jsonl`.

## 1. Headline

| | 30000 | 32000 | 34000 | total |
|---|---|---|---|---|
| pass | 31 | 29 | 27 | 87 |
| FAIL | 133 | 135 | 137 | 405 |

**The 280-token cap costs zero passes.** On the 96 capped failures (`stop_reason=max_new`, `ok=false`) every line-boundary prefix of the completion was judged — 2821 judge calls — and **not one passes**. The rollout never contained a correct solution at any cut point, so no larger budget and no stopping policy recovers a pass on these rows. Raising `max_new` is not a fix.

The other half of the same claim, from the length side: the 164 canonical HumanEval bodies tokenize to **median 46 / p90 108 / max 251** tokens, and **0 of 164 exceed 280**. The correct completions the model does produce are **median 20 / p90 58 / max 125** tokens. Both the reference solutions and the model's own successes fit the budget with room to spare.

So the failures are not a length or budget problem, and "the model ran out of room" is not the mechanism. For 30 of the 96 capped rows the model never even completed a syntactically valid body for its entry point; for the other 66 it completed one and kept writing, and the completed body was already wrong. The 66+30 is the whole capped set.

## 2. Classes, by explicit criteria

`judge()` (`eval/humaneval_gen.py`) builds `prompt + completion + test + check(entry_point)` and execs it in-process under a 6s `SIGALRM`, then swallows the exception and returns a bool — which is why the failure reason is invisible in the scored artifacts. The classifier re-runs the same exec with the same alarm and captures the exception type instead. Every row's `verdict_agrees_with_saved_ok` is true; the classifier reproduces the official score exactly.

| class | criterion | 30000 | 32000 | 34000 | total |
|---|---|---|---|---|---|
| `logic_wrong` | `AssertionError` from the test | 95 | 104 | 106 | 305 |
| `syntax_error` | `SyntaxError` compiling prompt+gen+test | 25 | 11 | 17 | 53 |
| `runtime_error` | any other exception, or `NameError` not naming the entry point | 12 | 20 | 13 | 45 |
| `signature_or_name` | `NameError` naming the entry point (never defined) | 0 | 0 | 1 | 1 |
| `timeout` | the 6s `SIGALRM` fired | 1 | 0 | 0 | 1 |
| empty output | pre-truncate completion blank | 0 | 0 | 0 | **0** |

`empty` is a real enum (`{}` maps to it) that nothing lands in: every one of the 492 rows produced non-blank text. An empty-output class is not needed as a separate row here — a blank completion shows up as `syntax_error` or `signature_or_name` when it does occur.

`stop_reason` is **orthogonal** to class and is the only reliable cap signal. `raw` (pre-truncate) is not saved on the `--rstrip_nl` arm, and `truncate()` runs before judging (`eval/humaneval_gen.py:641,663`), so a capped row's stored `gen` is already post-cut — for 4 of the 96 it is shorter than 280 tokens, and there the stored text is a lower bound on what was emitted.

| class | `stop` | `eos` | `max_new` |
|---|---|---|---|
| `logic_wrong` | 242 | 17 | 46 |
| `syntax_error` | 3 | 0 | 50 |
| `runtime_error` | 38 | 7 | 0 |
| `signature_or_name` | 1 | 0 | 0 |
| `timeout` | 1 | 0 | 0 |

`syntax_error` is nearly all cap-hit (50/53) and `runtime_error` is never cap-hit (0/45). That split is mechanical: a 280-cut lands mid-statement and fails to compile, while a short wrong answer runs and throws.

## 3. Stability across the three steps

| | tasks |
|---|---|
| correct at all three | 21 |
| wrong at all three | 128 |
| mixed (sometimes right) | **15** |

Mixed set: 10, 11, 12, 13, 14, 25, 27, 43, 47, 49, 52, 56, 59, 63, 76 — note 10–14 are consecutive.

The 15 mixed tasks are the measured scale of sampling noise at greedy-with-3-checkpoints. They are also the group RL can amplify: a task that is already reachable some of the time is what a policy gradient can turn into always-reachable, while the 128 always-wrong tasks are, by §1, failures of content rather than of luck.

## 4. Representative cases

Fragments are the stored `gen` (post-`truncate`), shown from the start of the completion.

### `logic_wrong` — syntax fine, semantics wrong
The dominant shape is a plausible one-line library call answering a *neighbouring* question:
- **HumanEval/161 step32000**, `gen_len=20`, `stop=stop`: `return s[::-1]` — reverses the string.
- **HumanEval/101 step34000**, `gen_len=25`, `stop=stop`: `return s.split(",")` — splits on commas and stops there.
- **HumanEval/104 step30000**, `gen_len=27`, `stop=stop`: `return sorted(set(x))` — dedupes and sorts, dropping the original order the task is about.
- **HumanEval/16 step30000**, `gen_len=29`, `stop=stop`: `return len(set(string))` — counts distinct characters.
- **HumanEval/27 step34000**, `gen_len=27`, `stop=stop`: `return string.lower()`.

These are 20–30 characters, stop on their own at `eos`/`stop`, compile and run, and fail the assertion. The model is not out of room and not confused about syntax; it is answering a generic version of the task.

### `syntax_error` — does not compile
- **HumanEval/122 step34000**, `gen_len=391`, `stop=stop`: `if sum_of_elements > 10 10:` — two adjacent numeric literals. The model stopped on its own, so this is not a cap artifact.
- **HumanEval/147 step30000**, `gen_len=588`, `stop=max_new`: nested `for` loops then `if ( ( ( ( ( (` unclosed — the cut left an open paren: `SyntaxError: '(' was never closed`.
- **HumanEval/67 step30000**, `gen_len=607`, `stop=max_new`: `SyntaxError: unterminated string literal (detected at line 15)` — cut mid-string.

### `runtime_error` — compiles, throws
- **HumanEval/17 step32000**, `gen_len=54`, `stop=eos`: `return [int(x) for x in music_string.split('|')]` — `ValueError: invalid literal for int() with base 10: ''` on an empty component.
- **HumanEval/19 step32000**, `gen_len=65`, `stop=eos`: `return ''.join(sorted(numbers, key=lambda x: x.split()[1]))` — `IndexError: list index out of range`, indexing a second whitespace field without checking it exists.
- **HumanEval/87 step32000**, `gen_len=47`, `stop=stop`: `return [(x, y) for x, y in lst if x == x]` — the `x == x` guard reads as a NaN check but is vacuous, and the unpack throws `ValueError: too many values to unpack (expected 2)` on the non-pair inputs.

All three are missing input guards, not wrong algorithms.

### `signature_or_name` — the entry point is never defined
- **HumanEval/78 step34000**, `gen_len=313`, `stop=stop`: writes a `count = 0 … return count` body, then `def hex_key(num): … return hex_key_helper(num)` — defines the entry point to call a helper that was never written (`name 'hex_key_helper' is not defined`). The only such row in 492.

### `timeout`
- **HumanEval/73 step30000**, `gen_len=772`, `stop=stop`: a commented swap loop with an unbounded `while j >= 0 and arr[j] != arr[i]` — the 6s alarm fires. The only such row in 492; the class is not a systemic issue at these steps.

### completed the function then kept writing (`max_new`), the restart shape
- **HumanEval/108 step30000**, `gen_len=1196`: writes a counter loop with duplicated comment lines ("Initialize a counter to store the number of digits / of the number in the array" appears twice), then rewinds and rewrites the same idea.
- **HumanEval/66 step30000**, `gen_len=1024`: emits `return sum(ord(c) for c in s)`, then re-emits the whole `def digitSum(s):` header plus docstring plus the same body — a full restart after finishing.
- **HumanEval/154 step30000**, `gen_len=870`: a length-enumerating body, then `def cycpattern_check(a , b):` and its docstring again.

### degenerate repetition — measurable, but not a class
All of these are **capped and never completed** (except where noted), and they are the same restart/enumeration shape as above with the repetition on top:
- **HumanEval/1 step30000**, `gen_len=1456`, `rep_max_token_count=172`: opens `# TODO: Write Write Write Write …` and then trails into prose ("…must be efficient for large inputs (up to 10^5 elements) and").
- **HumanEval/141 step30000**, `gen_len=562`: a single run of 279 `# ` characters — it reads 0 and 1 on the two counters, which is the miss in the audit below.
- **HumanEval/110/119/120/134 step30000**: `rep_max_line_count` 11/20/20/17 — bodies enumerating by length or index (`if len(lst) == 4: return 'YES'`, `if k == len(arr) - 14: return arr`). These have **completed a function first** (`completed_then_kept_writing`) and then enumerate; **HumanEval/128** (`rep_max_line_count=1`) is the same idea with every line different, which is why the line counter misses it.

Repetition is a real failure mode but it does not partition cleanly, so it is reported as **counts in the table, not as a class**. Human audit of 17 rows agreed with the bucket assignment 17/17 but with the repetition flag only 14/17, with errors in both directions: HumanEval/128 enumerates by length so every line differs (missed by the line counter, and it also defeats the token counter — all three counters read 0 or 1 on it), and HumanEval/132's repeated `if len(stack) > 0` is correct code (false positive). No threshold separates "degenerate" from "verbose but normal" on this data. Do not quote a "degenerate repetition N" number.

The same defect exists in the shipped counter: `repetitive()` (`eval/humaneval_gen.py:207`) looks only at the **last 200 characters**, and the HumanEval/1 shape (repetition up front, prose at the tail) is exactly what it cannot see. Measured on the 96 capped rows: the tail-200 rule flags **19**, a whole-string rule flags **28** — it misses ~32%. This counter feeds the `nrep` summary field and **does not enter any score**. Decision (1e, 2026-09-25): leave it, and treat `nrep` as unreliable rather than re-tune it, since the audit showed the underlying distinction is not thresholdable.

## 5. Can SFT A fix each class?

SFT A is `sft_mixA_0924.pt` (6617 rows, 143,687 examples; `facts/v41.json#v41.sft_mixA_pack_0924`). Composition by supervised-loss tokens: **code_if 56.3%, sc2-exec 27.4%, APPS call-style 5.1%, en_c4 prose 10.2%**.

| class | can SFT A fix it? | basis |
|---|---|---|
| `syntax_error` (53) | **Partly — the cap half, not the rest** | 50/53 are cap-hit, where the cut produces the broken syntax and the prefix-oracle says the content was already wrong, so a budget change does not help; what SFT A can change is the *shape* that produces 280 tokens of unfinished code. The 3 non-cap rows (e.g. 122's `10 10`) are token-level errors a code-distribution SFT plausibly reduces but does not target. |
| `runtime_error` (45) | **Partly** | All are non-cap and all run to a throw (`int('')`, index out of range, unpack mismatch) — missing input guards, which is a code-idiom problem SFT A's code distribution speaks to. No evidence in this data for or against the effect size. |
| `signature_or_name` (1) | **Yes, trivially** | One row (78) calls an unwritten helper. Not a systemic gap. |
| `logic_wrong` (305, 75% of failures) | **Not directly** | The completions compile, run, and answer the wrong question (161 returns `s[::-1]`; 87 uses `x == x`). SFT A teaches the code idiom, not these specific algorithms, and the 13-gram decontamination gate keeps HumanEval-shaped content out of the pack. Expect the 128 always-wrong tasks to be largely untouched. |
| `timeout` (1) | n/a | One unbounded `while`. |
| empty output (0) | n/a | Nothing to fix. |

**The stopping behaviour is the strongest lever, and it is not the cap.** 96 capped rows: 66 completed the function and then kept writing, 30 never completed it at all (the 66+30 is the whole capped set). The model's correct answers run 20 tokens median, so it can stop when the answer is done. SFT A is raw continuation with `EOS` supervision, so it teaches "stop when the answer is done" directly — plausibly the single highest-value effect, but §2's orthogonality is a warning: reducing `max_new` failures does not by itself move `logic_wrong`.

### Hypothesis to be tested, not a conclusion: does SFT A teach writing *longer*?

The composition gates suggest a risk. Both external code sources were built with a body-length gate, and the drop is a discard, not a truncation:

| source | gate | n | body-token median | p90 | max |
|---|---|---|---|---|---|
| code_if | body ≤ 256 | 76,644 pairs (23,047 dropped) | 97 | — | 256 |
| sc2-exec | output_len ≤ 256 (`MAX_OUT`) | 49,599 | 72 | 147 | 256 |
| APPS call-style | output_len ≤ 256 | 10,776 | 56 | 144 | 255 |
| HumanEval canonical (reference) | — | 164 | **46** | 108 | 251 |

The training distribution's median body (56–97 tokens) is above the HumanEval reference median (46), and the training p90 (144–147) is above the canonical p90 (108). So SFT A's data is on the long side of the eval's own answers.

Two things bound that risk and keep this a hypothesis, not a finding:
1. The `max` column is right-**censored** — the gate discarded anything over 256, so 256/255 says nothing about how long the model would write without a gate. It cannot be compared to canonical's max of 251.
2. Nothing in the data supports "long enough to not finish": 0 of 164 canonical bodies exceed 280, and 0 of 96 capped rollouts has *any* passing prefix. A longer writing style predicts more unfinished output, but the observed unfinished output is already wrong content, so the two are not the same failure.

What would test it: measure the SFT-A-trained checkpoint's completion-length distribution on the same 164 tasks and compare it to the base model's (median 20 / max 125 here). If training makes completions longer without moving pass@1, the length is being learned as style rather than as capability.

## 6. What this implies for the next steps

1. **Do not raise `max_new` for a score reason.** 0/96 recoverable, and 0/164 canonical bodies exceed the budget. If it is raised anyway, it is for a different purpose and should be argued as such.
2. **The failure mass is `logic_wrong` (75%)**, which is content, not form. Neither the budget nor a stopping-policy change addresses it; SFT A's code distribution is the only lever in hand and its effect on this class is unmeasured.
3. **`runtime_error` (45) and the 3 non-cap `syntax_error` are the classes where a code-idiom SFT has a concrete, plausible mechanism** (input guards, token-level syntax).
4. **The 15 mixed tasks are the RL-relevant set** — reachable sometimes, so amplification is possible; the 128 always-wrong are not, on this evidence.

## 7. Reproduction

- Classifier and finish detector: `/tmp/audit4.py`, `/tmp/decisive2.py`, `/tmp/oracle.py` were scratch scripts on the pod and are **not** committed. The table's `klass`/`detail` fields reproduce the official score exactly for all 492 rows (`verdict_agrees_with_saved_ok`), and the header records each rule in words.
- Inputs: 11-shard preds per step, `data/eval/preds_humaneval_ckpt_v41_ced_0923.pt.step{30000,32000,34000}.rstripnl.shard*of11.ced_s*_rstrip_sh*.jsonl` (pod `data/eval/`, 2.6 MB of that family total).
- Tokenizer: `data/tokenizer.json` (pod), the same file the SFT-A build stats use, so the length columns above are directly comparable.

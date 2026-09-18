# HumanEval step18000 failure buckets (rstrip CLEAN /156)

Source: `data/eval/preds_humaneval_ckpt_v41_r3_0914.milestone_he18k_step18000.pt.v41r3_he18000_rstrip_cpu.jsonl`
156 CLEAN problems (164 minus the 8-problem r3 contamination union: 19,66,71,78,105,123,129,156). 18 pass, 138 fail. Each failure parsed as `prompt+gen` and bucketed by one ordered rule.

## Distribution

| bucket | n | task_ids |
|---|---:|---|
| logic error (parses, correct entry function, tests fail) | 125 | 0,1,2,3,4,5,7,8,9,10,13,14,15,16,17,18,20,21,22,24,26,27,32,33,36,37,39,41,43,46,47,49,50,51,54,56,57,58,61,62,63,64,65,67,68,69,70,72,73,74,75,76,77,80,81,82,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,106,107,108,109,111,112,113,114,116,117,118,120,121,122,126,127,128,130,131,132,133,134,135,136,137,138,139,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,157,158,159,160,162,163 |
| degeneration / truncation (loops then cuts mid-construct) | 8 | 79,110,115,119,124,125,140,161 |
| empty / one-liner stub (≤6 words, no real body) | 5 | 6,11,38,44,83 |
| signature/name mismatch | 0 | — |

91% of failures are valid Python with the right function name that computes the wrong thing. Zero output-format failures (no missing/wrong function name).

## Examples

### Empty / one-liner stub (5)

- **HumanEval/11 string_xor** — XOR two 0/1 strings char by char, return a string. Output `return ''.join(a ^ b)`: one line; `^` on two strings raises TypeError, never iterates per character.
- **HumanEval/44 change_base** — convert x to the given base. Output `return str(x) + str(base)`: concatenates the input and base as a string; no conversion.
- **HumanEval/6 parse_nested_parens** — for each space-separated paren group return its deepest nesting depth. Output `return [int(paren_string)]`: wraps the whole input string in one int; no splitting, no depth counting.

### Degeneration / truncation (8)

All eight enter a repeated line pattern and run to the generation cut, ending mid-token/mid-string. Lengths 660–1091 chars; none emits a terminator.

- **HumanEval/140 fix_spaces** — replace spaces with underscores (rule actually depends on run length). Output repeats `text = text.replace(" ", "_")` ~10 times and is cut inside the opening quote of the last call: `...replace(" ` (SyntaxError: invalid syntax).
- **HumanEval/79 decimal_to_binary** — convert decimal to binary string. Output enumerates `if decimal_to_binary(0) == '0'/'1'/...: return ...` for every digit and is cut at `if decimal`; the recursion always calls itself on 0.
- **HumanEval/125 split_words** — split on whitespace, return list of words. Output repeats the same list comprehension `[word for word in words if word != '']` until cut at `word`; never returns.
- (Also 110, 115, 119, 124, 161; same shape: 119 dies on an unterminated string at line 59, 115 ends mid-condition `...and`.)

### Logic error (125)

- **HumanEval/0 has_close_elements** — return True if any pair is closer than threshold. Output compares each element itself to the threshold (`if numbers[i] > threshold: return True`); never forms a pair, wrong comparison.
- **HumanEval/26 remove_duplicates** — remove values occurring more than once, keep order. Output keeps every first-seen value (`if numbers[i] not in result`), which is dedup-by-identity; `[1,2,3,2,4]` returns `[1,2,3,4]` instead of `[1,3,4]`.
- **HumanEval/2 truncate_number** — return the decimal (fractional) part of a positive float. Output special-cases 0/negative then `return number / 10`; no integer-part subtraction.

## Readout for the anneal decision

- The model can format code: every failing solution parses to a function with the correct name; only 13/138 (9%) fail structurally, all from repetition degeneration at the generation cap, not format ignorance.
- The deficit is algorithmic: 125/138 (91%) are well-formed but implement the wrong computation — pair comparison, per-character operations, base conversion, fractional-part extraction all missing. These are the textbook/exercise primitives the anneal segment is meant to supply; a formatting/SFT pass would not touch this bucket.
- The 8 degeneration rows are a decoding/coverage symptom (repetition collapse on hard cases), separate from the knowledge deficit.

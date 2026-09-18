# Degeneration-8 at 13k vs 18k (rstrip preds, read-only)

Sources:
- 13k `data/eval/preds_humaneval_ckpt_v41_r3_0914.milestone_he13k_step13000.pt.v41r3_he13000_rstrip_cpu.jsonl`
- 18k `data/eval/preds_humaneval_ckpt_v41_r3_0914.milestone_he18k_step18000.pt.v41r3_he18000_rstrip_cpu.jsonl`

The 8: HumanEval 79,110,115,119,124,125,140,161 (18k truncation/degeneration bucket). HE rows have no `reason` field; termination is inferred from the tail (complete final statement vs cut mid-construct).

| task | 13k pass | 18k pass | 13k→18k chars | 13k shape | 18k shape | termination 13k → 18k |
|---|---|---|---:|---|---|---|
| 79 decimal_to_binary | ✗ | ✗ | 209→806 | 7 lines, no repeats, short wrong impl | 29 lines, same `if f(0)=='X'` line ×9, walks digits to 'E' | clean `return` → cut mid-`if decimal` |
| 110 exchange | ✗ | ✗ | 240→764 | 10 lines, maxrep 3, ends `return "YES"` | 25 lines, `len==1` branch ×10 | clean return → cut mid-`if len(` |
| 115 max_fill | ✗ | ✗ | 423→1091 | 14 lines, no repeats, ends `return ...count(0)` | 32 lines verbose BFS, no exact repeat (maxrep 2) | clean return → cut mid-condition `...and` |
| 119 match_parens | ✗ | ✗ | 191→879 | 8 lines, ends `return 'Yes'` | 42 lines, `if len(lst)==N: return 'Yes'` ×18, enumerates to 19 | clean return → cut inside opening quote |
| 124 valid_date | ✗ | ✗ | 1073→1322 | 26 lines, isinstance arm ×13, but ends `return False` | 30 lines, repeat ×14 | terminated (already repetitive) → cut mid-`raise TypeError('date` |
| 125 split_words | ✗ | ✗ | 173→1021 | 7 lines, ends `return words` | 21 lines, same list comprehension ×19 | clean return → cut at `word` |
| 140 fix_spaces | ✗ | ✗ | 35→1085 | 1 line: `return text.replace(" ","_")` | 38 lines, same replace line ×19 | single-line return → cut inside `replace(" ` |
| 161 solve | ✗ | ✗ | 836→660 | 32 lines, `s = s.replace(" ","")` ×30, ends mid `s =` — **already degenerating and cap-cut** | 29 lines, letter-branch ×12 | cap-cut → cap-cut |

All 8 fail at both checkpoints. At 13k, 7/8 terminated on a complete statement (short, well-formed, wrong logic; 124 already repetitive but terminated); only 161 was already in repetition-collapse at the cap. At 18k all 8 end mid-construct at the generation cap, 7/8 substantially longer (35→1085 chars for 140; 173→1021 for 125); 115 is a verbose-but-not-literal-repeat cap hit rather than a repeated line.

## Global check — MBPP `reason` distribution (same scorer, sigrstrip, full 427)

MBPP rows carry `{ok, empty, reason}` with reason ∈ eos/stop/max_new.

| ckpt | rows | eos | stop | max_new |
|---|---:|---:|---:|---:|
| r3 12k | 427 | 38.6% (165) | 26.2% (112) | **35.1% (150)** |
| r3 18k | 427 (full) | 33.5% (143) | 31.9% (136) | **34.7% (148)** |

Full-file pass: 12k 37/427 (8.7%) → 18k 51/427 (11.9%).

## Trend (one line)

Repetition-collapse on these 8 is mostly **new at 18k, not pre-existing**: 7/8 degraded from short terminating wrong answers at 13k into long cap-cut repeated rollouts at 18k (161 is the exception, already collapsed at 13k); the global max_new share on the full 427-row MBPP set is flat (35.1%→34.7%), so this is per-hard-problem behavior change (longer, non-terminating rollouts on unsolved cases), not a checkpoint-wide decoding regression.

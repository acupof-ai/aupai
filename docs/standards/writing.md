# Writing standard

Applies to every document, commit message, register row, retro row, reply, and message
between sessions. Set by the user 2026-08-31; the Fidelity section added on the user's order
of 2026-09-02 (information must not be distorted in transmission). Supersedes the shorter
list in `AGENTS.md`, which points here.

## Content

- Delete what a competent reader already knows. Delete explanations that add no
  information. A rewrite must raise information density or it is a no-op.
- No metaphors. No big words. No coined compressed terms. No verdict-first tone. No
  spoken or speech register. No filler that explains why something is being said.
- Numbers over adjectives; a definite statement over a two-sided hedge. Every number
  carries its measurement configuration.
- Fewer quotation marks and parentheses. Avoid the pattern "not X, but Y"; state Y.
  Avoid long parallel enumerations; group or tabulate instead.

## Fidelity

A claim carries how it was obtained. The reader must be able to tell, from the sentence
itself, which of these it is:

| kind | form | example |
|---|---|---|
| read | the artifact and the line, quoted or hashed | `train.py:2508` reads `for i in range(i0, len(Xtr) - Cfg.batch + 1, Cfg.batch)` |
| measured | the number with its command and config | 69.63 GiB peak, rank 0, `runs/proberesume_run1.log` steps 10–40 |
| derived | the inputs and the arithmetic | 19101 − 19073 = 28 for a 32-step segment; 1024 rows trimmed, 28 steps returned |
| recalled | marked as such | from memory, not re-read tonight |
| inferred | the observation and the inference, as two sentences | No checkpoint after 70 minutes. I read that as a crash loop. |

Rules, each bought with a case from 2026-09-02:

- **A line number is not a citation; the line's content is.** Quote or paraphrase what the
  line says now. Numbers 1563–1591 were cited for the resume defect after those lines had
  become `_encode_worker`.
- **A mechanism claim must reproduce the numbers it explains.** "Each resume adds
  resume_step" did not produce +28 and +50 from 32 and 51; the true mechanism did.
- **An observation and its inference are two sentences, never one.** Three deaths in 30
  minutes were three deliberate kills; "crash loop" was the inference stated as fact.
- **An extrapolation is not a measurement.** 8,744,830,156 tokens from 3 of 283 shards was
  reported as measured; the cache measurement was 8,786,916,332.
- **An absence proves an absence only if the reader could have seen the thing.** A check
  that cannot see the pod says SKIP, not PASS; a session that cannot see a message says
  "not received", not "not sent".
- **Quote the other party by content, not by label.** "tilerl's second point" tells the
  next reader nothing; "tilerl showed `mv` unlinks the inode under the live fd" does.
- **State what you did against what was asked, when they differ.** `cp` was used where
  the task said `mv`; the message said so first and why second.
- **Say what you cannot see.** Every message that reports state names its view (host,
  container, main, worktree) and its time in UTC.

Messages between sessions: the first line is the whole conclusion in one sentence; the body
is evidence in the forms above; a disagreement names a failing case or an artifact, never a
preference. A message that would change a register row, a fact, or a ruling is not done
until that artifact changes.

## Structure

- Lead with the conclusion, then the reason, then the detail. One idea per sentence.
- Three or more consecutive prose paragraphs: check whether a table, a list, a grouping,
  or a figure carries it better. Human attention is short; organise for a reader who
  holds one screen at a time.
- Highlight the key part with bold text; nothing else.

## Formulas

- Formulas are set as formulas, never as inline prose. Display formulas are centered.
  In Lark documents, formulas render through `<latex>`.

## Review

Every document gets at least four passes before it is handed over, in this order:

| pass | question |
|---|---|
| logic and facts | does every claim follow, and does every number have its source and config |
| redundancy | what can be deleted without losing information |
| reading load and structure | can a reader hold each section on one screen; should any run of paragraphs become a table |
| layout and visual consistency | headings, tables, bold, formulas, spacing consistent throughout |

Target: simple, clear, coherent, specific, accurate, complete, and fitting the reader.

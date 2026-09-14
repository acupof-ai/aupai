export const meta = {
  name: 'textbook-handread',
  description: 'Strict 1-5 blind handread of textbook chapters, batched at 12 concurrent agents',
  phases: [{ title: 'Handread' }],
}

// args:
//   chapters: [{pos, path}]  — chapters to grade; pos is the only id shown to the
//             reviewer (keep the source/generator OUT so the grade is blind).
//   python:   absolute interpreter path the reviewer MAY use only if reading cannot
//             settle a check. Graders default to NO code execution.
// Example args: {"chapters":[{"pos":0,"path":"/abs/pos_00.md"} ...],
//                "python":"/Users/bytedance/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/bin/python3.12"}

const SCHEMA = {
  type: 'object',
  properties: {
    pos: { type: 'integer' },
    correctness_1_5: { type: 'number', enum: [1, 2, 3, 4, 5] },
    pedagogy_1_5: { type: 'number', enum: [1, 2, 3, 4, 5] },
    label: { type: 'string', enum: ['keep', 'fix', 'drop'] },
    asserts_discriminate: { type: 'boolean' },
    general_claims_property_tested: { type: 'boolean' },
    defects: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['low', 'medium', 'high'] },
          class: { type: 'string' },
          quote: { type: 'string' },
          explanation: { type: 'string' },
        },
        required: ['severity', 'class', 'quote', 'explanation'],
      },
    },
    reviewer_note: { type: 'string' },
  },
  required: ['pos', 'correctness_1_5', 'pedagogy_1_5', 'label',
             'asserts_discriminate', 'general_claims_property_tested', 'defects', 'reviewer_note'],
}

const PY = (args.python || 'python3')

const RUBRIC = `You are a strict reviewer of a synthetic Python textbook chapter for a coding-model pretraining mix. You are BLIND to its source, generator and model — judge the chapter only. Read the file FIRST, then judge.

DO NOT WRITE OR EXECUTE CODE BY DEFAULT. Execution already happened upstream — the chapter passed a sandbox exec gate, and a mutation/exec gate (when its verdict is supplied) already says whether the asserts discriminate. Judge by READING: trace the shown blocks, compare every prose mechanism/measurement sentence against the code printed in the chapter, and reason about whether the visible asserts would catch a wrong implementation. Reading carefully is the job; running code is the exception, not the method. Do not create scratch/mutation files and do not write anything into the working directory.
- If, and ONLY IF, reading cannot settle a specific contradiction (a sentence claims X but the shown control flow implies Y), you MAY run that ONE check through the project safe-exec helper, e.g. \`from aupai_safe_exec import run_path_safe; run_path_safe(existing_chapter_path, argv_prefix=["${PY}"], timeout=20)\` — the chapter file is already on disk; never create a new file. NEVER call a bare \`python3\`/\`python\`, never use subprocess/Popen yourself. A third-party-module, network, or sandbox-capacity refusal (thread/nproc/watchdog) is NOT a code defect; judge prose+logic.
- asserts_discriminate: reason from the visible asserts — if a plausible wrong implementation (off-by-one, swapped args, dropped edge, flipped comparison) would still satisfy every shown assert, mark false. Trust a supplied mutation-gate verdict instead of re-running mutants.
- A property/general claim (big-O, "always", "returns the minimum", identity, asymptotic) asserted on ONE hand-picked input is not tested unless scoped to the example or backed by a property check/oracle over varied+negative inputs. A bound or constant must not be called "measured" unless the run printed it.
- Negative/edge cases the chapter says matter must be exercised by the shown asserts.
- Sentence-by-sentence factual truth. Distinguish a HYPOTHETICAL/conditional remark from an instance claim. Verify standard definitions against the textbook definition before flagging.
- Score correctness on CONTENT truth, not on whether it merely runs.

1-5 correctness: 5 no technical error; 4 one minor imprecision, nothing teaching wrong code; 3 at least one statement/example that teaches a wrong rule or overclaims beyond evidence; 2 multiple or a central concept wrong; 1 fundamentally wrong.
1-5 pedagogy: clarity, progression, worked-example quality.
label: keep (c5, or c4 with only nits), fix (c3 or fixable c4), drop (c<=2).

Adversarial but fair: c3 if it overclaims a general property from one case, calls a bound "measured", or its tests could not catch a wrong implementation. Do not invent defects; if none, score 4/5 and say so. Quote exact text for every defect.

Your final output is ONLY the structured object.`

// Team RPM cap (fb 2026-09-14): never more than 12 grading agents live. Larger grades
// run as sequential batches of 12, awaited, so an N-chapter grade does not queue N
// agents at once and starve generation routes.
const BATCH = 12
phase('Handread')
const chapters = args.chapters || []
const out = []
for (let i = 0; i < chapters.length; i += BATCH) {
  const chunk = chapters.slice(i, i + BATCH)
  log(`batch ${Math.floor(i / BATCH) + 1}/${Math.ceil(chapters.length / BATCH)} ` +
      `pos ${chunk[0].pos}-${chunk[chunk.length - 1].pos}`)
  const rs = await parallel(chunk.map((c) => () =>
    agent(`${RUBRIC}\n\nChapter file: ${c.path}\nShuffled position: ${c.pos}`,
      { label: `pos ${c.pos}`, phase: 'Handread', schema: SCHEMA, effort: 'high' })
  ))
  out.push(...rs.filter(Boolean))
}
return out

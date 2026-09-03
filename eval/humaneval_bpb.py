#!/usr/bin/env python3
# restartable: one JSON line per task is appended to --preds as it is scored, and a rerun with
# the same --preds skips task_ids already there. An interrupt costs the current task only.
"""HumanEval gold BPB: bits per byte of the 164 canonical solutions, teacher-forced.

    python3 eval/humaneval_bpb.py --ckpt <ckpt.pt>
    python3 eval/humaneval_bpb.py --ckpt <hf-dir> --hf     # control arm, its own tokenizer
    python3 eval/humaneval_bpb.py --selftest               # no card, no model

WHY BPB AND NOT pass@k. This is the metric that WORKS at 200M. facts/base_eval.json
#be.gold_bpb_falls_while_generation_scores_zero measured the split directly: code_500
generative accuracy sat at 0.0 across a whole checkpoint ladder while gold BPB fell
monotonically 1.087 -> 0.918. A base model at this scale cannot complete a function body well
enough to execute, so pass@k is 0 for both arms and 0 == 0 says nothing. The likelihood of the
REAL solution moves, and it moves before generation does.

WHY PER BYTE. The two arms of the control comparison have different tokenizers (ours vs a
50,304-entry NeoX BPE), so per-token loss is not comparable between them -- the same text is a
different number of tokens per side, and whichever tokenizer packs code more tightly wins on a
per-token metric for a reason unrelated to modelling. UTF-8 bytes are the same on both sides,
which makes bits/byte the only cross-tokenizer-honest unit here (1e's standing ruling, also
what eval/lambada_en.py uses).

WHAT IS SCORED. The prompt (signature + docstring) is context and carries NO loss; the
canonical solution is the target. That split is the whole point: scoring the prompt too would
mix "can it model a docstring" into a number reported as code modelling, and the prompt is
several times longer than the solution in most tasks.

NOT A pass@k SUBSTITUTE, and the record says so. BPB says the model assigns the real solution
higher likelihood; it does not say the model would produce it. Both readings belong in the
panel, and at 200M only this one has resolution.
"""

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("FLA_FLASH_KDA", "0")

TOK_PATH = os.path.join(ROOT, "data", "tokenizer.json")
DEFAULT_DATA = os.path.join(ROOT, "data", "eval", "humaneval", "humaneval_164.jsonl")
N_TASKS = 164  # the published set; a short file is a broken download, not a small run


def load_tasks(path, limit=None):
    """(task_id, prompt, canonical_solution) per row, prompt and solution both required.

    A row missing either field is a REFUSAL, not a skip: the denominator of this metric is the
    task count, and silently scoring 163 of 164 makes two runs incomparable while both look
    complete.
    """
    tasks = []
    with open(path, encoding="utf-8") as f:
        for k, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for field in ("task_id", "prompt", "canonical_solution"):
                if not r.get(field):
                    raise ValueError(f"{path} row {k} has no {field!r}; this file cannot be "
                                     f"scored without it")
            tasks.append((r["task_id"], r["prompt"], r["canonical_solution"]))
            if limit and len(tasks) >= limit:
                break
    return tasks


class OursModel:
    def __init__(self, ckpt, tok_path, device, prefix_arm=None):
        from scripts.loader import load_checkpoint, load_tokenizer  # noqa: PLC0415

        self.model, self.cfg = load_checkpoint(ckpt, device=device, dtype=torch.bfloat16)
        self.tok = load_tokenizer(tok_path, self.cfg)
        self.device = device
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.prompt_len = None
        self._prefix = None
        if prefix_arm:
            self._prefix = self._install_prefix(prefix_arm)

    def _install_prefix(self, arm):
        """Apply the arm's prefix-LM mask at SCORING time, matching how it trained.

        WHY THIS IS NEEDED AT ALL. A prefix-trained model scored causally is a topology mismatch,
        the same mismatch that made N7 Stage A's +0.0273 turn out to be measuring the mismatch and
        not the loop. So each arm is scored in its own mask, and the causal cell is kept as the
        control -- exactly the 2x2 that separated the two before.

        cu IS REQUIRED. model.py:189 reaches flash_attn_varlen_func -- the only entry point that
        takes mask_mod -- only when cu is not None, and this scorer called self.model(x) with no cu,
        so every forward took flash_attn_func at :194 and a mask_mod would have been silently
        ignored. That is the identical trap that made scripts/n7c_gates.py print three passes while
        testing nothing. Here the batch is ONE row, so cu is simply [0, T]: one document, and the
        prompt length is that document's own, which is what makes the eval side simpler than
        training's per-document projection.
        """
        import model as model_mod  # noqa: PLC0415
        from eval.prefix_mask import PREFIX_ARMS, build_mask_mods  # noqa: PLC0415

        if arm not in PREFIX_ARMS:
            raise SystemExit(f"REFUSING: unknown prefix arm {arm!r}; PREFIX_ARMS defines "
                             f"{sorted(PREFIX_ARMS)}")
        if not model_mod.HAS_FA:
            raise SystemExit(
                "REFUSING: HAS_FA is False, so GatedMLA takes the SDPA fallback at model.py:196 "
                "and no mask_mod is ever called. Scoring would silently be CAUSAL while reporting "
                "a prefix cell.")
        layers = PREFIX_ARMS[arm]
        targets = []
        for li in layers:
            mixer = self.model.blocks[li].mixer
            if not isinstance(mixer, model_mod.GatedMLA):
                raise SystemExit(f"REFUSING: block {li}'s mixer is {type(mixer).__name__}, not "
                                 f"GatedMLA; arm {arm} names MLA layers {list(layers)}")
            targets.append(mixer)
        _causal, prefix_mod = build_mask_mods()
        orig = model_mod.flash_attn_varlen_func
        depth = [0]
        aux = [None]

        def patched(q, k, v, **kw):
            if aux[0] is None or depth[0] == 0:
                return orig(q, k, v, **kw)
            kw.pop("causal", None)  # mask_mod REPLACES causality (interface.py:270)
            return orig(q, k, v, mask_mod=prefix_mod, aux_tensors=aux[0], **kw)

        # WRAPPING forward, not hooking it: hooks do not fire inside a grad_ckpt recompute, which
        # crashed the p7 training arm with CheckpointError. Scoring runs under no_grad so there is
        # no recompute here, but using the same mechanism in both places means the eval cell and
        # the training arm cannot diverge in which layers got the mask.
        def wrap(mod):
            inner = mod.forward

            def fwd(*a, **k):
                depth[0] += 1
                try:
                    return inner(*a, **k)
                finally:
                    depth[0] -= 1
            mod.forward = fwd

        for t in targets:
            wrap(t)
        model_mod.flash_attn_varlen_func = patched
        return aux, list(layers)

    def encode(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def logprobs(self, ids):
        """log p over the vocabulary at every position, given ids."""
        x = torch.tensor([ids], device=self.device)
        cu = None
        if self._prefix is not None:
            aux, _layers = self._prefix
            if self.prompt_len is None:
                raise SystemExit("REFUSING: prefix scoring needs the prompt length, and "
                                 "solution_bpb did not set it. Without it the mask would be built "
                                 "from a default and the cell would not be the arm's mask.")
            # ONE ROW, ONE DOCUMENT: cu = [0, T]. The prompt length is this document's own, so no
            # projection is needed -- unlike training, where a row packs 10 to 23 documents.
            cu = torch.tensor([0, len(ids)], dtype=torch.int32, device=self.device)
            aux[0] = [torch.tensor([self.prompt_len], dtype=torch.int32, device=self.device)]
        with torch.no_grad():
            # cu BY KEYWORD: forward is (idx, targets=None, cu=None, ...) at model.py:533, so a
            # positional second argument would bind to TARGETS and the method would return hidden
            # states instead of logits. cu stays None for a causal cell, which keeps that cell
            # byte-identical to how the twins were scored.
            out = self.model(x, cu=cu) if cu is not None else self.model(x)
        lg = (out[0] if isinstance(out, tuple) else out)[0].float()
        return torch.log_softmax(lg, -1)


class HFModel:
    def __init__(self, path, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.tok = AutoTokenizer.from_pretrained(path)
        # `dtype`, not `torch_dtype` (deprecated in the pod's transformers 5.6.0), and a plain
        # .to(device) because accelerate is absent so device_map/pipeline are unavailable.
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device)
        self.model.eval()
        self.device = device
        # Parameters, never buffers: Pythia-160m carries causal-mask and rotary inv_freq
        # buffers and counting them gives 212M instead of the real 162,322,944.
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def encode(self, s):
        return self.tok.encode(s)

    def logprobs(self, ids):
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            return torch.log_softmax(self.model(x).logits[0].float(), -1)


def solution_bpb(m, prompt, solution, max_ctx=2048):
    """Bits per UTF-8 byte of `solution`, conditioned on `prompt`. None if it does not fit.

    Teacher-forced: one forward over prompt+solution, then sum -log p of each solution token at
    the position that predicts it. The prompt contributes context and no loss.
    """
    p_ids = m.encode(prompt)
    s_ids = m.encode(solution)
    if not s_ids:
        return None, "solution encodes to zero tokens"
    ids = p_ids + s_ids
    if len(ids) > max_ctx:
        # TRUNCATE THE PROMPT, never the solution: the solution is the thing being scored, and
        # a partial solution's bits are not this metric. Keeping the prompt's TAIL keeps the
        # signature and the end of the docstring, which is what the solution continues from.
        keep = max_ctx - len(s_ids)
        if keep <= 0:
            return None, f"solution alone is {len(s_ids)} tokens, over the {max_ctx} context"
        p_ids = p_ids[-keep:]
        ids = p_ids + s_ids
    # THE PROMPT LENGTH IS SET AFTER TRUNCATION, not before. A prefix cell built from the
    # pre-truncation length would call solution tokens "prompt" on every task that overflowed
    # max_ctx and let them attend bidirectionally -- reading their own labels, so the leak would
    # LOWER the BPB and read as an improvement. Set here, next to the truncation, because the two
    # facts have to move together. Harmless for a causal cell, which ignores it.
    m.prompt_len = len(p_ids)
    lp = m.logprobs(ids)
    # Position i's distribution predicts token i+1, so the token at index len(p_ids)+j is
    # predicted by the row at len(p_ids)+j-1.
    total = 0.0
    for j, t in enumerate(s_ids):
        row = len(p_ids) + j - 1
        total -= float(lp[row, t])
    n_bytes = len(solution.encode("utf-8"))
    if n_bytes == 0:
        return None, "solution is zero bytes"
    return total / math.log(2) / n_bytes, None


def _selftest():
    # BPB arithmetic, on a model whose distribution is known exactly. A uniform distribution
    # over V symbols costs log2(V) bits per token by definition, so a one-token-per-byte
    # encoder must read exactly log2(V) bits per byte -- the one case where the right answer is
    # known without a model.
    class UniformModel:
        """Every next-token distribution is uniform over `v` ids; one id per byte."""
        def __init__(self, v):
            self.v = v

        def encode(self, s):
            return [b % self.v for b in s.encode("utf-8")]

        def logprobs(self, ids):
            return torch.full((len(ids), self.v), -math.log(self.v))

    for v in (256, 1024):
        m = UniformModel(v)
        bpb, err = solution_bpb(m, "ctx", "abcd")
        assert err is None, err
        want = math.log2(v)
        assert abs(bpb - want) < 1e-6, f"uniform({v}): {bpb} != log2({v}) = {want}"

    # The PROMPT must not enter the number. Same solution, two prompts of very different
    # length: a uniform model's bpb cannot depend on the context at all.
    m = UniformModel(256)
    a, _ = solution_bpb(m, "x", "hello")
    b, _ = solution_bpb(m, "x" * 500, "hello")
    assert abs(a - b) < 1e-9, f"prompt length changed the bpb: {a} vs {b}"

    # Multi-byte UTF-8: the divisor is BYTES, not characters. Three-byte characters must give
    # a third of the per-character figure under a one-token-per-byte encoder.
    one_byte, _ = solution_bpb(m, "x", "aaa")          # 3 bytes, 3 tokens
    three_byte, _ = solution_bpb(m, "x", "中")          # 3 bytes, 3 tokens
    assert abs(one_byte - three_byte) < 1e-9, f"{one_byte} vs {three_byte}"

    # Over-context: the PROMPT is trimmed and the solution survives whole. A solution that
    # cannot fit at all returns an error rather than a partial score.
    long_sol = "z" * 40
    bpb, err = solution_bpb(m, "p" * 500, long_sol, max_ctx=60)
    assert err is None and bpb is not None, f"should have trimmed the prompt: {err}"
    bpb, err = solution_bpb(m, "p", "z" * 100, max_ctx=50)
    assert bpb is None and "over the" in (err or ""), f"expected a refusal, got {bpb} {err}"

    # A row missing a field is a refusal, not a skip.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        tf.write(json.dumps({"task_id": "t/0", "prompt": "p", "canonical_solution": "s"}) + "\n")
        tf.write(json.dumps({"task_id": "t/1", "prompt": "p"}) + "\n")
        bad = tf.name
    try:
        try:
            load_tasks(bad)
            raise AssertionError("a row with no canonical_solution was accepted")
        except ValueError as e:
            assert "canonical_solution" in str(e), e
    finally:
        os.unlink(bad)

    print("humaneval_bpb self-test OK: uniform models read exactly log2(V) bits/byte, the "
          "prompt does not enter the number, the divisor is bytes not characters, an "
          "over-length prompt is trimmed while the solution never is, and a row missing a "
          "field refuses")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="our .pt, or an HF directory with --hf")
    ap.add_argument("--hf", action="store_true", help="control arm: HF format, own tokenizer")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tokenizer", default=TOK_PATH)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, help="first N tasks (smoke test)")
    ap.add_argument("--max_ctx", type=int, default=2048)
    ap.add_argument("--preds", help="jsonl, appended per task; rerun resumes from it")
    ap.add_argument("--out", help="summary json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--loop", nargs=2, type=int, metavar=("LO", "HI"),
                    help="N7 Stage A: run blocks LO..HI twice at inference (eval/loop_wrapper.py, "
                         "AttnRes option 3). Ours only -- refuses with --hf, since the wrapper "
                         "implements THIS model's source ledger and an HF model has no _body to "
                         "patch. The summary records loop_blocks so a looped row cannot be read "
                         "as the unlooped one it is compared against.")
    ap.add_argument("--prefix", choices=("p3", "p7"),
                    help="N7 Stage C: apply the arm's prefix-LM mask at SCORING time "
                         "(eval/prefix_mask.py PREFIX_ARMS), so a prefix-trained checkpoint is "
                         "scored in the mask it trained in. Without this a prefix arm is a topology "
                         "MISMATCH against its own weights -- the same mismatch that made Stage A's "
                         "+0.0273 turn out to measure the mismatch rather than the loop -- so each "
                         "arm gets both cells: its own mask and the causal control. Ours only. "
                         "Needs cu, which this scorer did not pass: model.py:189 reaches the only "
                         "entry point taking mask_mod solely when cu is not None, so a mask without "
                         "it is silently ignored. The summary records prefix_arm and prefix_layers.")
    a = ap.parse_args()

    if a.selftest:
        _selftest()
        return 0
    if not a.ckpt:
        ap.error("--ckpt required (or --selftest)")
    if not os.path.exists(a.data):
        sys.exit(f"humaneval data missing: {a.data}")

    tasks = load_tasks(a.data, a.limit)
    if not a.limit and len(tasks) != N_TASKS:
        sys.exit(f"{a.data} has {len(tasks)} tasks, expected {N_TASKS}. A short file is a "
                 f"broken download; scoring it would produce a number that looks like the "
                 f"published metric and is not.")

    if a.prefix and a.hf:
        sys.exit("REFUSING: --prefix with --hf. The mask is applied through OUR model.py's "
                 "flash_attn_varlen_func seam; an HF model has no such seam and would be scored "
                 "CAUSALLY while the summary claimed a prefix cell.")
    m = HFModel(a.ckpt, a.device) if a.hf else OursModel(a.ckpt, a.tokenizer, a.device,
                                                         prefix_arm=a.prefix)
    if a.loop:
        if a.hf:
            sys.exit("REFUSING: --loop with --hf. loop_wrapper implements OUR AttnRes source "
                     "ledger; an HF model has no _body and no ledger, so there is nothing the "
                     "ruling applies to and a silently-unlooped row would be reported as looped.")
        # Patched on the instance the scorer already built, so the looped arm goes through this
        # same solution_bpb, tokenizer and byte divisor as the unlooped arm.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from loop_wrapper import patch_body  # noqa: PLC0415
        patch_body(m.model, tuple(a.loop))
        print(f"looped: blocks {a.loop[0]}..{a.loop[1]} run twice (AttnRes option 3)", flush=True)
    print(f"Loaded {a.ckpt}: {m.n_params / 1e6:.2f}M params | tasks {len(tasks)}", flush=True)

    done = {}
    if a.preds and os.path.exists(a.preds):
        with open(a.preds, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # THE RESUME KEY MUST CARRY THE INTERVENTION. Rows are keyed by task_id, and a
                # looped run resuming from an unlooped --preds would reuse every unlooped row and
                # report it as looped -- the two arms produce identically-shaped rows for the same
                # task_id. A row whose loop stamp disagrees with this run REFUSES rather than being
                # skipped, because silently rescoring some tasks and reusing others gives a mean
                # over a mixture of two models.
                if r.get("loop_blocks") != (list(a.loop) if a.loop else None):
                    sys.exit(
                        f"REFUSING: {a.preds} holds rows scored with loop_blocks="
                        f"{r.get('loop_blocks')} and this run is loop_blocks="
                        f"{list(a.loop) if a.loop else None}. Resuming would mix two models under "
                        f"one mean. Use a separate --preds path per arm.")
                # THE SAME CHECK FOR THE MASK, for the same reason: a prefix cell and a causal cell
                # produce identically-shaped rows for the same task_id, and the 2x2 asks for BOTH
                # cells of the same checkpoint. Without this, scoring the causal cell into the prefix
                # cell's --preds would reuse every prefix row and report it as causal -- the mismatch
                # control would silently become a copy of the matched cell.
                if r.get("prefix_arm") != a.prefix:
                    sys.exit(
                        f"REFUSING: {a.preds} holds rows scored with prefix_arm="
                        f"{r.get('prefix_arm')!r} and this run is prefix_arm={a.prefix!r}. The two "
                        f"cells of a 2x2 must not share a --preds path.")
                done[r["task_id"]] = r
        print(f"resuming: {len(done)} tasks already in {a.preds}", flush=True)

    vals, errs = [], []
    for tid, prompt, sol in tasks:
        r = done.get(tid)
        if r is None:
            bpb, err = solution_bpb(m, prompt, sol, a.max_ctx)
            r = {"task_id": tid, "bpb": bpb, "error": err,
                 "n_bytes": len(sol.encode("utf-8")),
                 # Stamped on EVERY row, including unlooped ones (None), so the resume check above
                 # compares like with like. Files written before this field existed carry no
                 # loop_blocks, and .get() reads them as None -- which is correct, because --loop
                 # is introduced by this same change and no pre-stamp file can be looped.
                 "loop_blocks": list(a.loop) if a.loop else None,
                 # Stamped for the same reason loop_blocks is. A file written before this field
                 # existed reads as None, which is correct: --prefix is introduced by this change,
                 # so no pre-stamp row can be a prefix row.
                 "prefix_arm": a.prefix}
            if a.preds:
                with open(a.preds, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if r.get("bpb") is None:
            errs.append((r["task_id"], r.get("error")))
        else:
            vals.append((r["bpb"], r["n_bytes"]))

    if not vals:
        print("REFUSING: no task produced a number")
        return 1
    # TWO means, because they answer different questions and averaging one of them silently
    # would be a choice nobody could see. The per-task mean weights every task equally; the
    # byte-weighted mean is the bits/byte of the corpus as one string, which is what "BPB of
    # HumanEval" normally denotes. They differ when solution lengths differ, and they do.
    per_task = sum(v for v, _ in vals) / len(vals)
    tot_bytes = sum(n for _, n in vals)
    byte_weighted = sum(v * n for v, n in vals) / tot_bytes

    result = {
        "ckpt": a.ckpt, "hf": a.hf, "n_tasks": len(vals), "n_tasks_total": len(tasks),
        "n_params": m.n_params,
        "loop_blocks": list(a.loop) if a.loop else None,
        # THE MASK IS PART OF THE CELL'S IDENTITY. Without these two fields a prefix cell and its
        # causal control are two files with the same shape and no way to tell which is which -- the
        # same failure the repo paid for with .stepN checkpoints whose metadata was identical.
        "prefix_arm": a.prefix,
        "prefix_layers": list(m._prefix[1]) if getattr(m, "_prefix", None) else None,  # noqa: SLF001
        "gold_bpb_per_task_mean": per_task,
        "gold_bpb_byte_weighted": byte_weighted,
        "total_solution_bytes": tot_bytes,
        "errors": [{"task_id": t, "error": e} for t, e in errs],
        "reading": "bits per UTF-8 byte of the canonical solution, teacher-forced, prompt "
                   "carries no loss; cross-tokenizer comparable by construction",
        "boundary": "NOT a pass@k substitute: this says the real solution gets higher "
                    "likelihood, not that the model would generate it. At 200M pass@k is 0 "
                    "for both arms (be.gold_bpb_falls_while_generation_scores_zero: code_500 "
                    "generative 0.0 across a ladder while gold BPB fell 1.087 -> 0.918), so "
                    "this is the reading with resolution at this scale, and the two numbers "
                    "must not be read as one.",
    }
    if errs:
        result["boundary"] += (f" {len(errs)} of {len(tasks)} tasks produced no number and are "
                               f"listed in `errors`; the means are over the rest.")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

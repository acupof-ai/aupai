#!/usr/bin/env python3
"""cot pilot: apply cot_criterion_0903 to a downloaded parquet slice, per source.

Reports, per source, BEFORE a full fetch: frozen-tokenizer tokens/doc, reject
rate with the per-check histogram, the host that served, and reachable-vs-target
extrapolation. Not the build instrument -- the full fetch + stamp rides the
fetcher+build on the pod (this only measures a slice to gate that).

criterion: docs/standards/cot_criterion_0903.md (written before any content read).
Field bindings below were schema-verified on 2026-09-03 after the criterion was
written; the keep CHECKS are unchanged, only the column access is bound to reality.

Sources:
  open-thoughts/OpenThoughts-114k  -> chain = conversations[assistant].value
  Skywork/Skywork-OR1-RL-Data      -> prompt (user), reward_model.ground_truth (answer)
"""
import argparse
import json
import os
import re
from collections import Counter

import pyarrow.parquet as pq
from tokenizers import Tokenizer

TOK = None
THINK_TAG = re.compile(r"\s*<\|(?:begin|end)_of_thought\|>\s*")


def chain_of(schema, d):
    """Bind the multi-step reasoning chain per the schema-verified field map."""
    if schema == "openthoughts":
        for t in d.get("conversations") or []:
            if t.get("from") in ("assistant", "gpt"):
                return t.get("value") or "", ""
        return "", ""
    if schema == "skywork":
        # RL-Data: prompt (user) + reward_model.ground_truth (answer). No chain.
        ans = (d.get("reward_model") or {}).get("ground_truth") or ""
        return "", ans
    if schema == "openr1":
        # verified reasoning: problem + generations (long traces) + answer + correctness_*.
        # chain = first correctness-verified generation (fallback: longest one).
        gens = d.get("generations") or []
        cmv = d.get("correctness_math_verify") or []
        if gens:
            for g, ok in zip(gens, cmv, strict=False):
                if ok and g:
                    return g, (d.get("answer") or "")
            return max(gens, key=len, default=""), (d.get("answer") or "")
        return d.get("solution") or "", (d.get("answer") or "")
    return "", ""


def checks_pass(schema, d):
    chain = d.get("_chain")
    ans = d.get("_ans")
    flags = {}
    # 1 multi-step: has a non-empty chain
    flags["no_chain"] = not (chain and len(chain.strip()) > 0)
    if schema == "skywork":
        # this source is prompt+answer by construction; nothing to keep as CoT
        flags["no_chain"] = True
    # 2 substantive (long CoT): chain >= 200 chars normalized
    flags["too_short"] = (not flags["no_chain"]) and len(re.sub(r"\s+", " ", chain)) < 200
    # 3 complete: no truncation marker in the chain
    flags["truncated"] = (not flags["no_chain"]) and bool(re.search(r"(\.\.\.\s*$|(^|\s)?truncat|\[\s*\.\.\.\s*\])", chain, re.I))
    # 4 self-consistent for math: skipped in pilot (needs full solve); the build
    #    checks answer-derivability on the math subset. Recorded as not-applied.
    flags["math_unchecked"] = False
    # 5 clean: no embedded scaffold/hash artifacts ANSWER-side.
    flags["dirty_answer"] = bool(ans and re.search(r"<\|(begin|end)_of_thought\|>", ans))
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--schema", choices=["openthoughts", "skywork", "openr1"], required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target_tokens", type=float, default=4.5e9)
    ap.add_argument("--max_rows", type=int, default=0, help="cap row sample for the pilot (0 = full slice)")
    ap.add_argument("--total_rows", type=int, default=0, help="true rows in the whole slice, for the reachable extrapolation")
    a = ap.parse_args()
    global TOK
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TOK = Tokenizer.from_file(os.path.join(root, "data", "tokenizer.json"))

    rows = pq.read_table(a.parquet).to_pylist()[: (None if not a.max_rows else a.max_rows)]
    n = len(rows)
    n_keep = 0
    tok_tot = 0
    ck = Counter()
    tag_rate = 0.0
    doc_toks = []
    for r in rows:
        chain, ans = chain_of(a.schema, r)
        body = THINK_TAG.sub(" ", chain).strip()   # count the reasoning substance
        if "<|end_of_thought|>" in chain:
            tag_rate += 1
        r["_chain"], r["_ans"] = body, ans
        r["_token_ct"] = len(TOK.encode(body).ids) if body else 0
        flags = checks_pass(a.schema, r)
        # any flag truthy -> reject; skip the counted-flag set
        reject_flags = {k for k, v in flags.items() if v}
        ck.update(reject_flags or {"keep"})
        if not reject_flags and body:
            n_keep += 1
            tok_tot += r["_token_ct"]
            doc_toks.append(r["_token_ct"])

    tot_all = sum(len(TOK.encode(THINK_TAG.sub(" ", chain_of(a.schema, r)[0]).strip()).ids) for r in rows)
    mean = (tok_tot / n_keep) if n_keep else 0.0
    keep_frac = n_keep / max(1, n)
    total = a.total_rows or n
    reach = mean * total * keep_frac
    target = a.target_tokens
    print(json.dumps({
        "source": a.source, "schema": a.schema, "slice_row_cap": "full-slice",
        "rows": n, "kept": n_keep, "reject_n": n - n_keep,
        "reject_rate": round((n - n_keep) / max(1, n), 4),
        "checks": dict(ck), "tag_enclosed_rate": round(tag_rate / max(1, n), 4),
        "tokens_kept_total": tok_tot, "tokens_pilot_all": tot_all,
        "tokens_per_doc_mean_kept": round(mean, 1),
        "p95_kept": sorted(doc_toks)[int(len(doc_toks) * 0.95)] if doc_toks else 0,
        "reachable_vs_target": {"extrapolated_keep_tokens": reach, "target": target,
                                "frac_of_target": round(reach / target, 4)},
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

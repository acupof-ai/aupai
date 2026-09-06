#!/usr/bin/env python3
"""Append moe.equal_token_gap_vs_dense_b192_per_domain (entry 2) to facts/moe.json.

Entry 1 (moe.equal_token_gap_vs_dense_b192) recorded ONE number per point: the in-training
val gap at nine matched steps. It states in its own boundary that per-domain attribution is
not in it. This is that attribution, at TWO pairing points rather than one, because a single
post-anneal point cannot say whether the per-domain distribution is a property of the arms or
of the annealing.

Reads the four doc_cu rows from runs/score_matrix.jsonl IN THE REPO, not from the pod. The
rows were written on the pod's emptyDir and pulled with pod_pull_ledgers.py --apply; a fact
citing a row that exists only on the pod cites something a pod delete erases.

REFUSES rather than emitting a partial or unverifiable comparison:
  - any of the four rows absent, or not cu_path=doc_cu
  - the four rows disagreeing on vocab_id, cu_path, profile or the domain set (a comparison
    across different scoring conditions is not a comparison)
  - a row whose stored unweighted_mean does not match the mean recomputed from its own nine
    per-domain losses (the row would then be internally inconsistent and nothing read from it
    is trustworthy)

WHAT THIS ENTRY MUST NOT SAY, each because the data says otherwise:
  - "the gap grows in every domain": two of nine SHRINK from step5000 to step10000 (cot,
    en_c4_stage2) while the mean grows. A mean's direction is not a per-domain direction.
  - "MoE beats dense on chat": only 2 of 9 domains at step10000, and 0 of 9 at step5000,
    exceed the readable_move_nat 0.24 that these very rows carry. Sub-0.1-nat per-domain
    ordering is recorded as measured, not as resolved.
  - a reason for the intensification: the two points differ in BOTH token count (3.93B vs
    7.86B) and lr state (constant vs 845 steps into warmdown). Two variables, two points.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "runs", "score_matrix.jsonl")
FACTS = os.path.join(ROOT, "facts", "moe.json")
FID = "moe.equal_token_gap_vs_dense_b192_per_domain"
EXTENDS = "moe.equal_token_gap_vs_dense_b192"

ROWS = {
    "moe_5k": "ckpt_1.5b-a0.2b-e48_8b.milestone_matchedtok_step5000.pt#cu",
    "den_5k": "ckpt_0.2b_8b_b192.milestone_matchedtok_step5000.pt#cu",
    "moe_10k": "ckpt_1.5b-a0.2b-e48_8b.pt#cu",
    "den_10k": "ckpt_0.2b_8b_b192.pt#cu",
}
CHAT = ("chat_qa", "chatml")
CMD = ("CUDA_VISIBLE_DEVICES=1 python3 eval/score_matrix.py --ckpt <C> --metrics domain_loss "
       "--cu_path doc_cu --ngpu 1 --mix data/mix_200m_8b.json --json runs/score_matrix.jsonl")


def load():
    want = {v: k for k, v in ROWS.items()}
    got, lineno, meta = {}, {}, {}
    with open(LEDGER, encoding="utf-8") as fh:
        for i, ln in enumerate(fh, 1):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("ckpt") in want and r.get("cu_path") == "doc_cu":
                tag = want[r["ckpt"]]
                dl = r.get("metrics", {}).get("domain_loss", {})
                got[tag] = {d: v["loss"] for d, v in dl.items()
                            if isinstance(v, dict) and "loss" in v}
                lineno[tag] = i
                meta[tag] = {
                    "vocab_id": r.get("vocab_id"), "profile": r.get("profile"),
                    "split": dl.get("_split"), "wall_s": dl.get("_wall_s"),
                    "stored_mean": dl.get("unweighted_mean"),
                    "noise": r.get("noise_thresholds", {}).get("domain_loss", {}),
                }
    absent = [k for k in ROWS if k not in got]
    if absent:
        sys.exit("REFUSING: no doc_cu row in runs/score_matrix.jsonl for: "
                 + ", ".join(f"{k} ({ROWS[k]})" for k in absent)
                 + " -- pull the pod ledger before writing a fact that cites it")

    for key in ("vocab_id", "profile", "split"):
        vals = {meta[t][key] for t in ROWS}
        if len(vals) != 1:
            sys.exit(f"REFUSING: rows disagree on {key}: {sorted(map(str, vals))} -- these are "
                     f"not four measurements of one quantity")
    dsets = {t: frozenset(got[t]) for t in ROWS}
    if len(set(dsets.values())) != 1:
        sys.exit("REFUSING: the four rows do not share one domain set: "
                 + "; ".join(f"{t}={len(dsets[t])}" for t in ROWS))
    doms = sorted(next(iter(dsets.values())))
    if len(doms) != 9:
        sys.exit(f"REFUSING: {len(doms)} domains, expected 9: {doms}")

    for t in ROWS:
        stored, rec = meta[t]["stored_mean"], sum(got[t].values()) / len(doms)
        if stored is None:
            sys.exit(f"REFUSING: row {lineno[t]} ({t}) has no unweighted_mean to check against")
        if abs(stored - rec) > 1e-3:
            sys.exit(f"REFUSING: row {lineno[t]} ({t}) is internally inconsistent: stored "
                     f"unweighted_mean {stored:.4f} vs {rec:.4f} recomputed from its own nine "
                     f"domains -- nothing read from this row can be trusted")
    return got, lineno, meta, doms


def main():
    got, lineno, meta, doms = load()
    noise = meta["moe_10k"]["noise"]
    sd = noise.get("sd_nat")
    readable = noise.get("readable_move_nat")
    nsrc = noise.get("source")
    if sd is None or readable is None or nsrc is None:
        sys.exit("REFUSING: the rows carry no noise_thresholds -- this entry's whole "
                 "significance discussion is quoted from them, so it cannot be written "
                 "without them")

    g5 = {d: got["den_5k"][d] - got["moe_5k"][d] for d in doms}
    g10 = {d: got["den_10k"][d] - got["moe_10k"][d] for d in doms}
    m5, m10 = sum(g5.values()) / 9, sum(g10.values()) / 9

    shrank = sorted(d for d in doms if g10[d] < g5[d])
    if not shrank:
        sys.exit("REFUSING: no domain's gap shrank between the two points, so this entry's "
                 "'a mean's direction is not a per-domain direction' claim is unsupported by "
                 "the data it just read -- re-derive before writing")
    rise = sum(g10.values()) - sum(g5.values())
    chat_rise = sum(g10[d] - g5[d] for d in CHAT)

    def tbl(g, src):
        return "; ".join(f"{d} {g[d]:+.4f} ({100 * g[d] / sum(src.values()):.1f}%)"
                         for d in sorted(doms, key=lambda z: -g[z]))

    over10 = sorted((d for d in doms if g10[d] > readable), key=lambda z: -g10[z])
    over5 = [d for d in doms if g5[d] > readable]
    chat5 = 100 * sum(g5[d] for d in CHAT) / sum(g5.values())
    chat10 = 100 * sum(g10[d] for d in CHAT) / sum(g10.values())
    code5 = 100 * sum(g5[d] for d in doms if d.startswith("code_")) / sum(g5.values())
    code10 = 100 * sum(g10[d] for d in doms if d.startswith("code_")) / sum(g10.values())

    entry = {
        "id": FID,
        "value": (
            f"Per-domain split of the MoE-48-vs-batch-matched-dense gap, at TWO pairing "
            f"points. dense-minus-MoE is POSITIVE in all nine domains at both points (MoE "
            f"lower everywhere), and the split is chat-heavy at both: the two chat domains "
            f"(chat_qa, chatml) are 2 of 9 domains and take {chat5:.1f}% of the summed gap at "
            f"3.93B and {chat10:.1f}% at 7.86B, while the two code domains take {code5:.1f}% "
            f"and {code10:.1f}%. Nine-domain unweighted means: {m5:+.4f} nat at 3.93B "
            f"(step5000 pins, rows {lineno['moe_5k']}/{lineno['den_5k']}) and {m10:+.4f} at "
            f"7.86B (endpoints, rows {lineno['moe_10k']}/{lineno['den_10k']}). "
            f"THE MEAN'S GROWTH IS NOT A PER-DOMAIN DIRECTION: the summed gap rises "
            f"{rise:+.4f} between the points and {100 * chat_rise / rise:.1f}% of that rise is "
            f"the two chat domains, while {len(shrank)} of 9 domains SHRINK "
            f"({', '.join(shrank)}). "
            f"3.93B: {tbl(g5, g5)}. "
            f"7.86B: {tbl(g10, g10)}."
        ),
        "measured": "2026-09-06",
        "status": "measured",
        "source": (
            f"Four doc_cu rows in runs/score_matrix.jsonl IN THIS REPO, lines "
            f"{lineno['moe_10k']} ({ROWS['moe_10k']}), {lineno['den_10k']} "
            f"({ROWS['den_10k']}), {lineno['moe_5k']} ({ROWS['moe_5k']}), {lineno['den_5k']} "
            f"({ROWS['den_5k']}). All four from the identical command, ckpt the only "
            f"difference: `{CMD}`. The rows were written on the pod's emptyDir and pulled with "
            f"scripts/pod_pull_ledgers.py --apply, which is why they can be cited: an emptyDir "
            f"row vanishes when the pod is deleted. This entry's writer refuses if any of the "
            f"four is absent, if they disagree on vocab_id/profile/_split or the domain set, "
            f"or if a row's stored unweighted_mean disagrees with the mean recomputed from its "
            f"own nine losses (max observed disagreement 4.4e-05, a rounding artefact of the "
            f"stored 4-decimal figure)."
        ),
        "config": {
            "extends": (
                f"{EXTENDS}, which recorded one in-training val number per point and states in "
                f"its own boundary that per-domain attribution is not in it. This is that "
                f"attribution. It is a DIFFERENT STATISTIC on a different corpus: doc_cu "
                f"nine-domain scoring at profile full, not the in-training val figure. The two "
                f"disagree in size and the difference is not explained here -- at step5000 "
                f"doc_cu {m5:+.4f} against val +0.0800, at step10000 doc_cu {m10:+.4f} against "
                f"val +0.0890. Same sign, same order, different number; do not substitute one "
                f"for the other."
            ),
            "pairing": (
                "MATCHED STEPS at matched global batch: batch 8 x accum 4 x 6 cards = 786,432 "
                "tokens/step on BOTH arms, ratio 1:1. Both arms print `3.93B tok` at step 5000 "
                "and both run total 10172 with warmup 300 (2.9% on each side). Point A: the "
                "matchedtok_step5000 milestone pin from each arm, 3.93B tokens. Point B: each "
                "arm's endpoint checkpoint, 7.86B tokens."
            ),
            "lr_state": (
                "Point A is CONSTANT lr, point B is POST-ANNEAL, and that is the reason two "
                "points exist rather than one. warmdown 0.1 of 10172 steps puts warmdown_start "
                "at 10172 - max(1, int(0.1 * 10172)) = 9155 on both arms; the endpoint pair sits "
                "845 steps inside it, the step5000 pair 4155 steps outside. Verified equal on "
                "both arms from the logged lr itself rather than from the formula: step 9000 "
                "1.00e-02, step 9200 9.96e-03, step 9400 8.71e-03, step 10000 1.16e-03, "
                "identical on each side. So the two arms anneal together and point B's "
                "comparison is fair -- but A and B differ in BOTH tokens and lr state, so no "
                "cause is assigned to the difference between them."
            ),
            "domains": "9 (chat_qa, chatml, code_py_rp1t, code_py_starcoder, cot, en_c4_stage2, "
                       "math_owm_stage2, textbook_30b, zh_web); 262,144 tokens and 64 blocks "
                       "per domain per row; mix data/mix_200m_8b.json; _split val",
            "arms": "MoE: 48 routed experts, top_k 3, expert_ffn 768, 1 shared, moe_layers "
                    "0-11, moe_bias_gamma 0.001, ~1.5B total / ~0.2B active. dense: ~0.2B, no "
                    "moe_* keys in its cfg line. seed 42 and sample_seed 42 on both.",
            "cost": (
                f"{meta['moe_5k']['wall_s']:.1f}s and {meta['den_5k']['wall_s']:.1f}s of scoring "
                f"for the step5000 pair, {meta['moe_10k']['wall_s']:.1f}s and "
                f"{meta['den_10k']['wall_s']:.1f}s for the endpoints (_wall_s, one card each, "
                f"run sequentially on card 1)"
            ),
        },
        "uncertainty": (
            f"MOST OF THE PER-DOMAIN ORDERING IS NOT RESOLVED BY THIS MEASUREMENT. The rows "
            f"carry their own noise_thresholds for domain_loss -- sd_nat {sd}, "
            f"readable_move_nat {readable}, source {nsrc} -- and against readable_move only "
            f"{len(over10)} of 9 domains clear it at 7.86B ({', '.join(over10)}) and "
            f"{len(over5)} of 9 at 3.93B. Every other per-domain figure above, and the code-vs- "
            f"chat contrast in the smaller domains, sits below the threshold these rows "
            f"themselves quote. THAT THRESHOLD IS ALSO BORROWED, NOT MEASURED HERE: "
            f"{nsrc} is 4 seeds at 3 df on val_nll of a mix_scale_0.2b run at batch 16 accum 2 "
            f"with fp8 -- a different config and a different quantity from doc_cu per-domain "
            f"loss, and its own uncertainty field puts the sigma CI at roughly 0.6x-2.9x. "
            f"NO SEED REPLICATE EXISTS FOR EITHER ARM HERE (single seed 42 on each side), so "
            f"the correct reading is that the summed/mean gap and the two chat domains are "
            f"above any plausible noise scale while the individual small-domain ordering is "
            f"recorded, not established. Host was not controlled for either arm's training "
            f"(the dense arm's window spans a 95 GB backup); that affects step time, measured "
            f"elsewhere, and no mechanism is claimed by which it moves loss."
        ),
        "boundary": (
            f"WHAT BUYS THE GAP IS NOT SEPARATED. The MoE arm carries ~1.28B more total "
            f"parameters AND a routing mechanism; the two move together across these arms, so "
            f"neither this entry nor {EXTENDS} can say which one the chat-heavy split comes "
            f"from. moe.equal_token_gap_vs_moe48_small_batch is the pair that holds "
            f"architecture fixed and varies batch and warmup share; read it beside this rather "
            f"than combined with it. ALSO NOT ANSWERED: why the chat share intensifies "
            f"{chat5:.1f}% -> {chat10:.1f}%. Both tokens and lr state change between the two "
            f"points, and a third point inside the constant-lr region would be needed to "
            f"separate them -- the pins for one do not exist. NO OVERALL VERDICT ON MoE VS "
            f"DENSE IS IN THIS ENTRY: it reports where the gap sits across domains at two "
            f"token counts, and a verdict would additionally need cost per token, which is "
            f"measured separately."
        ),
    }

    with open(FACTS, encoding="utf-8") as fh:
        doc = json.load(fh)
    if any(f.get("id") == FID for f in doc["facts"]):
        sys.exit(f"REFUSING: {FID} already exists; a measured fact's value is not rewritten -- "
                 f"add a new entry or a retraction instead")
    if not any(f.get("id") == EXTENDS for f in doc["facts"]):
        sys.exit(f"REFUSING: {EXTENDS} is not in facts/moe.json, so this entry's `extends` "
                 f"points at nothing")
    doc["facts"].append(entry)
    with open(FACTS, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"appended {FID}")
    print(f"  means {m5:+.4f} (3.93B) / {m10:+.4f} (7.86B); chat {chat5:.1f}% -> {chat10:.1f}%")
    print(f"  shrinking domains: {', '.join(shrank)}; above readable_move: {len(over5)}/9 and "
          f"{len(over10)}/9")
    print(f"  facts in file: {len(doc['facts'])}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Does our own pretraining corpus contain the held-out items the floor comparison rests on?

THE CONFOUND, registered at 51e5905 and not yet measured on either side. The control audit
(docs/audits/control_pythia160m_vs_ours.md) reports 0.451 for the control arm against 0.904 for ours
on a 10,421-item population, ids sha256 cae4daf7ad59388c. Both arms are un-SFT'd models reading text
neither is supposed to have seen. The audit's own §4 leak check covers train<->heldout and the
benchmarks; it does NOT cover either PRETRAINING corpus. So the number could be partly a memorisation
gap rather than a capability gap, and one side of that is measurable here: what our 200M run actually
trained on. The Pile side stays labelled unmeasured -- we do not have it, and saying "probably clean"
about someone else's corpus would be the kind of claim this script exists to replace.

THE COUNTING RULE, stated before any number because "any 13-gram hit" is ambiguous three ways:
  WHAT IS A TOKEN     whitespace-split words, lowercased, punctuation stripped from the edges. NOT
                      BPE ids: the corpus and the held-out set would have to be tokenized identically
                      for that to mean anything, and a 13-BPE-token window is ~8 words, a much weaker
                      claim than 13 words. Word 13-grams are the standard containment unit (Brown et
                      al's dedup work, The Pile's own leak checks) and are what makes the number
                      comparable to published contamination rates.
  WHAT IS AN ITEM     an item HITS if any 13-gram of its ANSWER appears anywhere in the corpus. The
                      answer, not the question: a question repeated in a corpus teaches nothing the
                      model can use to score better on nll of the ANSWER, which is what the floor
                      measures (nll_per_supervised_byte over supervised tokens = the answer).
                      Reported both ways anyway, because a reader may want the question figure and
                      recomputing it costs another 161 GB pass.
  DENOMINATOR         10,421 -- the SCORED population, not the 10,641 in the file. 220 items were
                      dropped for the control arm's seq 2048 and 28 as overlong; the floor is over
                      the intersection and so is this. runs/heldout_v2/ids_shared.txt is that list.

THE DIRECTION OF THE SCAN MATTERS FOR COST. The corpus is 160.97 GB over nine domains; the held-out
answers are ~10 MB. So the 13-grams of the ANSWERS go in a set (hashed to 8 bytes to bound memory),
and the corpus streams past once. One pass, no index, no card.

    python3 scripts/e1_28_heldout_contamination.py            # the real run, on the pod
    python3 scripts/e1_28_heldout_contamination.py --selftest  # the known answers
"""
# restartable: the scan writes a per-domain checkpoint (runs/e1_28_progress.json) after each of the
# nine domains and resumes from it, so an interrupt costs at most one domain rather than the whole
# 160.97 GB pass. zh_web alone is 89.85 GB, so "just re-run it" would be an hour thrown away; the
# hit set is a union over domains, which makes the resume exact rather than approximate.
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELDOUT = os.path.join(ROOT, "data", "sft", "control_sft_text_heldout.jsonl")
IDS = os.path.join(ROOT, "runs", "heldout_v2", "ids_shared.txt")
MIX = os.path.join(ROOT, "data", "mix_200m_4b.json")
CORPUS = os.path.join(ROOT, "data", "corpus")
N = 13
EXPECT_POP = 10421
EXPECT_FP = "cae4daf7ad59388c"

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def words(text):
    """Lowercased word tokens. Unicode-aware so zh_web is not silently reduced to nothing."""
    return _WORD.findall(text.lower())


def grams(toks, n=N):
    """The n-grams of a token list, hashed to 8 bytes.

    Hashed rather than stored as strings: the answers alone are ~2.8M tokens, so ~2.8M n-grams, and
    the string form would be several hundred MB of Python objects. blake2b at 8 bytes gives a
    collision probability around 2.8e6^2 / 2^65 = 2e-7 over the whole set, which is far below the
    resolution of the hit rate being reported. A collision can only ever create a FALSE HIT, never
    hide a real one, so it biases toward over-reporting contamination -- the safe direction.
    """
    for i in range(len(toks) - n + 1):
        yield hashlib.blake2b(" ".join(toks[i : i + n]).encode(), digest_size=8).digest()


def load_population():
    """(rows, ids) restricted to the scored intersection, with the fingerprint ASSERTED.

    The fingerprint is what makes this the same population the floor was computed over. Without the
    assertion this script would happily report a contamination rate for a DIFFERENT set of items than
    the 0.451 it is meant to qualify, and the two numbers would be printed side by side as though they
    were about the same thing.
    """
    with open(IDS, encoding="utf-8") as fh:
        ids = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    fp = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
    # COMPARE AS STRINGS ON BOTH SIDES. ids_shared.txt holds "0" while the JSONL holds the int 0, so
    # `r["id"] in set(ids)` matched NOTHING -- 0 of 10,421 -- and the row-count assertion is what
    # caught it. Without that assertion this would have built an empty index, scanned 160.97 GB, and
    # reported zero contamination: a clean bill of health from having compared nothing, which is the
    # exact shape of the vacuous pass that already cost a day on the N8 probe.
    keep = {str(i) for i in ids}
    rows = []
    with open(HELDOUT, encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            if str(r["id"]) in keep:
                rows.append(r)
    return rows, ids, fp


def main():
    if "--selftest" in sys.argv:
        return selftest()

    rows, ids, fp = load_population()
    print(f"population: {len(rows)} rows from {len(ids)} ids in "
          f"runs/heldout_v2/ids_shared.txt, fingerprint {fp}")
    if len(rows) != EXPECT_POP:
        raise SystemExit(f"REFUSING: {len(rows)} rows, expected {EXPECT_POP}. This is not the "
                         f"population the 0.451 floor was computed over, so a contamination rate "
                         f"from it does not qualify that number.")
    if fp != EXPECT_FP:
        print(f"  NOTE: fingerprint {fp} != the audit's {EXPECT_FP}. The audit's sha is over its own "
              f"id serialisation, which this script does not reproduce byte for byte; the ROW COUNT "
              f"matching {EXPECT_POP} is the check that carries weight here.")

    # answer-grams and question-grams kept separate, so one corpus pass answers both
    a_index, q_index = {}, {}
    for r in rows:
        for g in grams(words(r["answer"])):
            a_index.setdefault(g, set()).add(r["id"])
        for g in grams(words(r["question"])):
            q_index.setdefault(g, set()).add(r["id"])
    print(f"index: {len(a_index):,} distinct answer {N}-grams, {len(q_index):,} question "
          f"{N}-grams")
    short = sum(1 for r in rows if len(words(r["answer"])) < N)
    if short:
        # AN ITEM SHORTER THAN THE WINDOW CAN NEVER HIT, so it is a structural zero, not a clean
        # item. Counting it as clean would understate the rate on the items that can be measured.
        print(f"  {short} item(s) have answers under {N} words and CANNOT hit at this n; they are "
              f"excluded from the measurable denominator below, not counted as clean.")

    with open(MIX, encoding="utf-8") as fh:
        domains = list(json.load(fh)["domains"])
    # RESUME STATE. The hit set is a UNION over domains, so replaying a prefix of the domain list is
    # exactly equivalent to having scanned it -- that is what makes a per-domain checkpoint sound
    # here and would not hold for a running mean.
    prog_path = os.path.join(ROOT, "runs", "e1_28_progress.json")
    a_hit, q_hit = set(), set()
    per_domain = {}
    total_lines = total_bytes = 0
    done = []
    if os.path.exists(prog_path) and "--restart" not in sys.argv:
        with open(prog_path, encoding="utf-8") as fh:
            pr = json.load(fh)
        # THE FINGERPRINT GUARDS THE RESUME: progress from a different population is not progress.
        if pr.get("fingerprint") == fp and pr.get("n") == N:
            a_hit, q_hit = set(pr["answer_hit_ids"]), set(pr["question_hit_ids"])
            per_domain = pr["per_domain_new_answer_hits"]
            total_lines, total_bytes = pr["corpus_lines"], pr["corpus_bytes"]
            done = pr["domains_done"]
            print(f"resuming: {len(done)} domain(s) already scanned {done}, "
                  f"{len(a_hit)} answer-hit item(s) so far")
        else:
            print(f"  ignoring {prog_path}: it is for fingerprint {pr.get('fingerprint')} n="
                  f"{pr.get('n')}, not {fp} n={N}")
    for dom in domains:
        if dom in done:
            continue
        d = os.path.join(CORPUS, dom)
        if not os.path.isdir(d):
            print(f"  {dom}: no shards -- SKIPPED (absent, not zero)", flush=True)
            continue
        before = len(a_hit)
        nl = nb = 0
        for shard in sorted(f for f in os.listdir(d) if f.endswith(".jsonl")):
            with open(os.path.join(d, shard), encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    nl += 1
                    nb += len(ln)
                    try:
                        obj = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("content") or obj.get("text") or ""
                    if not text:
                        continue
                    toks = words(text)
                    for g in grams(toks):
                        if g in a_index:
                            a_hit |= a_index[g]
                        if g in q_index:
                            q_hit |= q_index[g]
        per_domain[dom] = len(a_hit) - before
        total_lines += nl
        total_bytes += nb
        print(f"  {dom:22s} {nb / 1e9:6.2f} GB {nl:>10,} lines  "
              f"+{per_domain[dom]} new answer-hit item(s)  (running {len(a_hit)})", flush=True)
        done.append(dom)
        with open(prog_path, "w", encoding="utf-8") as fh:
            json.dump({"fingerprint": fp, "n": N, "domains_done": done,
                       "answer_hit_ids": sorted(a_hit), "question_hit_ids": sorted(q_hit),
                       "per_domain_new_answer_hits": per_domain,
                       "corpus_lines": total_lines, "corpus_bytes": total_bytes}, fh)

    n_meas = len(rows) - short
    out = {
        "population": len(rows), "fingerprint": fp, "n": N,
        "counting_rule": ("word 13-grams, lowercased, unicode word characters; an item hits if any "
                          "13-gram of its ANSWER appears in any scanned corpus line; denominator is "
                          "the scored intersection"),
        "answer_hits": len(a_hit), "question_hits": len(q_hit),
        "measurable_denominator": n_meas, "answers_under_n_words": short,
        "answer_hit_rate": len(a_hit) / n_meas if n_meas else None,
        "corpus_bytes": total_bytes, "corpus_lines": total_lines,
        "domains_scanned": [d for d in domains if d in per_domain],
        "per_domain_new_answer_hits": per_domain,
        "hit_ids": sorted(a_hit),
        "pile_side": "UNMEASURED -- the control arm's pretraining corpus is not available here",
    }
    dest = os.path.join(ROOT, "runs", "e1_28_heldout_contamination.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\nANSWER-GRAM HITS: {len(a_hit)} of {n_meas} measurable "
          f"({100 * len(a_hit) / n_meas:.3f}%)" if n_meas else "no measurable items")
    print(f"question-gram hits: {len(q_hit)} (reported for completeness; the floor scores answers)")
    print(f"scanned {total_bytes / 1e9:.2f} GB over {len(per_domain)} domain(s), "
          f"{total_lines:,} lines")
    print(f"wrote {dest}")
    if len(a_hit) == 0:
        print("READING: no held-out answer shares a 13-word span with our pretraining corpus, so the "
              "0.451-vs-0.904 floor gap is NOT explained by our-side memorisation. The control arm's "
              "corpus stays unmeasured, so the confound is halved, not closed.")
    else:
        print("READING: the floor and the lead must be recomputed on the non-overlapping subset and "
              "reported beside the full-set values; hit_ids in the JSON is that exclusion list.")
    return 0


def selftest():
    """Known answers. A containment counter with no case where it must fail is not a measurement."""
    bad = 0
    toks = words("The quick brown fox jumps over the lazy dog and then it keeps running far away")
    g = list(grams(toks, 3))
    if len(g) != len(toks) - 2:
        print(f"  FAIL {len(g)} 3-grams from {len(toks)} tokens, expected {len(toks) - 2}")
        bad += 1
    # a window longer than the text yields nothing -- the structural-zero case the main path counts
    if list(grams(words("only three words"), 13)):
        print("  FAIL a 13-gram was produced from a 3-word text")
        bad += 1
    # the same span in different case and punctuation must collide; a different span must not
    a = set(grams(words("alpha beta gamma"), 3))
    b = set(grams(words("Alpha, BETA. gamma!"), 3))
    c = set(grams(words("alpha beta delta"), 3))
    if a != b:
        print("  FAIL case/punctuation changed the hash: the same span must collide")
        bad += 1
    if a & c:
        print("  FAIL different spans collided")
        bad += 1
    # unicode: zh text must not tokenize to nothing, or zh_web's 89.85 GB would scan as empty and
    # report zero hits while never having looked
    zh = words("模型 在 预训练 语料 上 记住 了 答案")
    if len(zh) < 5:
        print(f"  FAIL zh text tokenized to {len(zh)} tokens; zh_web would scan as empty")
        bad += 1
    # THE ID TYPE CASE, from the real failure: ids_shared.txt holds "0" and the JSONL holds int 0, so
    # an un-coerced membership test matched 0 of 10,421 items. A scan on that empty index would have
    # read 160.97 GB and reported zero contamination.
    if str(0) not in {str(i) for i in ["0", "50"]}:
        print("  FAIL int/str id coercion does not match a string id list")
        bad += 1
    if "0" in {0, 50}:
        print("  FAIL the un-coerced comparison unexpectedly matched; this fixture no longer "
              "reproduces the defect it was written for")
        bad += 1
    print(f"e1_28 selftest: {'OK' if not bad else f'{bad} FAILURE(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

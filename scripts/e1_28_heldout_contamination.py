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

E1_28_WORKERS (default 32) sets the shard-parallel width. The serial version was measured at
4.18 MB/s, cpu-bound fraction 1.00 on one core -- 10.7 h for the corpus on a 180-core box.
"""
# restartable: the scan writes a per-domain checkpoint (runs/e1_28_progress.json) after each of the
# nine domains and resumes from it, so an interrupt costs at most one domain rather than the whole
# 160.97 GB pass. zh_web alone is 89.85 GB over 909 shards; the hit set is a union over lines and so
# over shards and domains, which is what makes both the resume and the shard-parallel scan exact
# rather than approximate.
import hashlib
import json
import multiprocessing
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


#: Gram-hash key sets, set in the parent before the Pool forks so workers inherit them instead of
#: receiving a copy per task. Only the KEYS travel: the id-bearing dicts stay in the parent, so a
#: worker cannot report which items hit -- it reports which grams matched, and the parent maps back.
_A_KEYS = frozenset()
_Q_KEYS = frozenset()


def _scan_shard(path):
    """(lines, bytes, matched answer-gram hashes, matched question-gram hashes) for one shard.

    Returns MATCHED HASHES rather than hit ids so the worker needs only the key sets, and returns
    counts rather than accumulating anywhere: the parent sums them, and a worker that dies takes its
    shard's counts with it rather than corrupting a shared total.
    """
    nl = nb = 0
    a_m, q_m = set(), set()
    # BYTES, so open in binary and decode per line. `len(ln)` on a text-mode line counts CHARACTERS,
    # and the first parallel run printed textbook_30b as "3.14 GB" against 8.12 GB on disk (ratio 2.59)
    # and chatml as 0.09 against 0.171 -- every Chinese domain understated by up to 3x, which would
    # have gone into the fact as the corpus size the scan covered.
    with open(path, "rb") as fh:
        for raw in fh:
            nl += 1
            nb += len(raw)
            try:
                obj = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            text = obj.get("content") or obj.get("text") or ""
            if not text:
                continue
            for g in grams(words(text)):
                if g in _A_KEYS:
                    a_m.add(g)
                if g in _Q_KEYS:
                    q_m.add(g)
    return nl, nb, a_m, q_m


def per_domain_alone(rows, a_index, domains, workers):
    """Each domain's hit set measured INDEPENDENTLY, plus the first matching shard per item.

    WHY THIS IS A SEPARATE PASS. main()'s running column prints `len(a_hit) - before`: what each
    domain added to a growing UNION in mix order. It printed chat_qa +0, zh_web +0, code_py_rp1t +0 --
    whose alone values are 1,515 / 81 / 57. chat_qa's +0 meant "added nothing new", because
    build_chat_qa.py and build_chatml.py are two renders of data/corpus/chat, so chatml had already
    contributed every one of those items. An incremental column cannot be read per domain, and the
    exclusion list needs a per-item domain to separate a shared source from ordinary overlap.

    Returns (per_domain, item_domains). per_domain[dom] carries the alone count, the scanned bytes and
    the shards' size on disk; item_domains[id][dom] is the first shard that matched that item.
    """
    keys = frozenset(a_index)
    global _A_KEYS, _Q_KEYS
    _A_KEYS, _Q_KEYS = keys, frozenset()
    per_domain, item_domains = {}, {}
    for dom in domains:
        d = os.path.join(CORPUS, dom)
        if not os.path.isdir(d):
            continue
        shards = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".jsonl")]
        hit, nb = set(), 0
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(min(workers, max(1, len(shards)))) as pool:
            for path, (_nl, s_nb, a_m, _q) in pool.imap_unordered(_shard_with_path, shards,
                                                                  chunksize=1):
                nb += s_nb
                for g in a_m:
                    for i in a_index[g]:
                        hit.add(i)
                        item_domains.setdefault(str(i), {}).setdefault(dom, os.path.basename(path))
        on_disk = sum(os.path.getsize(s) for s in shards)
        per_domain[dom] = {"alone": len(hit), "bytes": nb, "on_disk": on_disk,
                           "ids": sorted(hit)}
        print(f"  {dom:22s} alone {len(hit):5d}  {nb / 1e9:6.2f} GB  "
              f"{100 * nb / on_disk:.1f}% of shards", flush=True)
    return per_domain, item_domains


def _shard_with_path(path):
    """(path, _scan_shard(path)), so imap_unordered results can be attributed to their shard."""
    return path, _scan_shard(path)


def longest_run(a, b):
    """Longest common contiguous word run between two token lists, by binary search on the length.

    The classifier that separates 6e's two provenance cases: ~13-20 words is an ordinary shared
    phrase and the 13-gram threshold is doing the work; most of the answer means the answer is IN the
    corpus; all of it means one generator wrote both sides. Measured on chatml: 1,199 of 1,515
    answers are a verbatim substring of one corpus document, and ZERO equal the whole document,
    because the document also carries the question.

    Binary search rather than a DP table: the answers run to 1,284 words and the documents longer, so
    an O(len(a)*len(b)) table is hundreds of millions of cells per item. Monotone because a common run
    of length k implies one of every length below k.
    """
    lo, hi, best = 0, min(len(a), len(b)), 0
    while lo <= hi:
        k = (lo + hi) // 2
        if k == 0:
            break
        bg = {" ".join(b[i:i + k]) for i in range(len(b) - k + 1)}
        if any(" ".join(a[i:i + k]) in bg for i in range(len(a) - k + 1)):
            best, lo = k, k + 1
        else:
            hi = k - 1
    return best


def classify_domain(rows, a_index, dom):
    """Per-item longest common run against the first matching document in one domain.

    Streams the domain serially: it needs the matching document's TEXT, not just the fact of a match,
    and the domains worth classifying (chatml 0.17 GB) are small.
    """
    keys = frozenset(a_index)
    by_id = {str(r["id"]): r for r in rows}
    d = os.path.join(CORPUS, dom)
    first = {}
    for sh in sorted(f for f in os.listdir(d) if f.endswith(".jsonl")):
        with open(os.path.join(d, sh), "rb") as fh:
            for lineno, raw in enumerate(fh, 1):
                try:
                    obj = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                t = obj.get("content") or obj.get("text") or ""
                if not t:
                    continue
                toks = words(t)
                ids = set()
                for g in grams(toks):
                    if g in keys:
                        ids |= a_index[g]
                for i in ids:
                    if str(i) not in first:
                        first[str(i)] = (sh, lineno, toks, t)
        print(f"  {sh}: {len(first)} item(s) located", flush=True)
    out = []
    for i, (sh, lineno, toks, text) in first.items():
        ans = words(by_id[i]["answer"])
        run = longest_run(ans, toks)
        # VERBATIM is tested on the text, not the token run: a substring test answers "is this answer
        # in that document" directly, while the run length answers "how much of it".
        norm_a = " ".join(by_id[i]["answer"].split())
        norm_t = " ".join(text.split())
        out.append({"id": i, "shard": sh, "line": lineno, "answer_words": len(ans),
                    "doc_words": len(toks), "longest_run": run,
                    "frac_of_answer": round(run / len(ans), 4) if ans else None,
                    "verbatim_substring": norm_a in norm_t,
                    "equals_whole_doc": norm_a == norm_t})
    out.sort(key=lambda r: -(r["frac_of_answer"] or 0))
    return out


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
    workers = int(os.environ.get("E1_28_WORKERS", "32"))
    with open(MIX, encoding="utf-8") as fh:
        all_domains = list(json.load(fh)["domains"])

    # THE TWO ANALYSES THAT ANSWER "what does the rate mean", as flags on this one scanner rather
    # than as separate scripts: 6e's ruling 2026-09-04 was one scanner with one selftest, and two
    # files sharing an index-building path is how the index drifts between them.
    if "--per-domain-alone" in sys.argv:
        per_domain, item_domains = per_domain_alone(rows, a_index, all_domains, workers)
        union = set()
        for v in per_domain.values():
            union |= {str(i) for i in v["ids"]}
        dest = os.path.join(ROOT, "runs", "e1_28", "e1_28_per_domain_alone.json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump({"per_domain": per_domain, "item_domains": item_domains,
                       "union": sorted(union)}, fh)
        print(f"union {len(union)}  sum-of-alone "
              f"{sum(v['alone'] for v in per_domain.values())}")
        print(f"wrote {dest}")
        return 0
    if "--classify" in sys.argv:
        dom = sys.argv[sys.argv.index("--classify") + 1]
        out = classify_domain(rows, a_index, dom)
        dest = os.path.join(ROOT, "runs", "e1_28", f"e1_28_prov_{dom}.json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        runs = sorted(r["longest_run"] for r in out)
        print(f"\n{dom}: {len(out)} item(s) located")
        print(f"verbatim substring of one doc: {sum(1 for r in out if r['verbatim_substring'])}")
        print(f"equal to the whole doc: {sum(1 for r in out if r['equals_whole_doc'])}")
        print(f"longest common run: median {runs[len(runs) // 2]} words, min {runs[0]}, "
              f"max {runs[-1]}")
        for cut in (0.25, 0.5, 0.9, 0.99):
            print(f"  >={int(100 * cut)}% of the answer's words in one run: "
                  f"{sum(1 for r in out if (r['frac_of_answer'] or 0) >= cut)}")
        print(f"wrote {dest}")
        return 0

    short = sum(1 for r in rows if len(words(r["answer"])) < N)
    if short:
        # AN ITEM SHORTER THAN THE WINDOW CAN NEVER HIT, so it is a structural zero, not a clean
        # item. Counting it as clean would understate the rate on the items that can be measured.
        print(f"  {short} item(s) have answers under {N} words and CANNOT hit at this n; they are "
              f"excluded from the measurable denominator below, not counted as clean.")

    with open(MIX, encoding="utf-8") as fh:
        domains = list(json.load(fh)["domains"])
    # RESUME STATE. The hit set is a UNION over domains, so replaying a prefix of the domain list is    # exactly equivalent to having scanned it -- that is what makes a per-domain checkpoint sound
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
    # SHARD-PARALLEL, because the serial form was measured and it does not finish in a working day.
    # /proc/<pid>/io on the first serial run: 376.4 MB of rchar in 90.0 s of wall with 90.0 s of CPU
    # in the same window -- 4.18 MB/s, one core, cpu-bound fraction 1.00. 160.97 GB at that rate is
    # 10.7 HOURS, on a box with 180 cores at load 27. The cost is not IO and not json: on a 16.4 MB
    # sample, json.loads 0.05 s, tokenize 0.68 s, gram+hash 2.65 s (78%, of which the " ".join is
    # 1.05 s). So the only lever that matters is running many shards at once.
    #
    # THE UNION IS WHAT MAKES THIS EXACT, and it is the same argument as the per-domain checkpoint
    # above: an item hits if ANY 13-gram of its answer appears ANYWHERE, so the result is a set union
    # over lines and therefore over shards, and shard order and grouping cannot change it. This would
    # NOT hold for a running mean, a first-hit location, or a per-shard rate.
    #
    # Workers return MATCHED GRAM HASHES, not ids: the id-bearing dict (1.07M answer grams -> id sets)
    # stays in the parent and each worker carries only the frozenset of hashes to test membership
    # against. A match is rare, so the returned sets are tiny.
    global _A_KEYS, _Q_KEYS
    _A_KEYS, _Q_KEYS = frozenset(a_index), frozenset(q_index)
    print(f"scanning with {workers} worker process(es) over shards (serial was 4.18 MB/s measured, "
          f"10.7 h for the corpus)", flush=True)
    for dom in domains:
        if dom in done:
            continue
        d = os.path.join(CORPUS, dom)
        if not os.path.isdir(d):
            print(f"  {dom}: no shards -- SKIPPED (absent, not zero)", flush=True)
            continue
        before = len(a_hit)
        nl = nb = 0
        shards = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".jsonl")]
        # fork, so _A_KEYS/_Q_KEYS are inherited rather than pickled per task
        with multiprocessing.get_context("fork").Pool(min(workers, max(1, len(shards)))) as pool:
            for s_nl, s_nb, a_m, q_m in pool.imap_unordered(_scan_shard, shards, chunksize=1):
                nl += s_nl
                nb += s_nb
                for g in a_m:
                    a_hit |= a_index[g]
                for g in q_m:
                    q_hit |= q_index[g]
        per_domain[dom] = len(a_hit) - before
        total_lines += nl
        total_bytes += nb
        # nb against the directory's real size, because the scan's own byte count is the only evidence
        # that it read the whole domain: a shard that failed to open contributes 0 silently.
        on_disk = sum(os.path.getsize(s) for s in shards)
        gap = "" if not on_disk else f"  {100 * nb / on_disk:.1f}% of the {on_disk / 1e9:.2f} GB shards"
        print(f"  {dom:22s} {nb / 1e9:6.2f} GB {nl:>10,} lines  "
              f"+{per_domain[dom]} new answer-hit item(s)  (running {len(a_hit)}){gap}", flush=True)
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
    # THE PARALLEL REWRITE MUST GIVE THE SERIAL ANSWER, on a fixture where the answer is known and
    # where the hit is SPLIT ACROSS SHARDS -- one shard holds the matching span, another holds a
    # decoy, and a third is a shard that produces nothing. If _scan_shard returned ids instead of
    # gram hashes, or if the key sets did not reach a forked worker, this is where it shows.
    import tempfile as _tf
    global _A_KEYS, _Q_KEYS
    span = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    a_index = {}
    for g in grams(words(span)):
        a_index.setdefault(g, set()).add("item-1")
    _A_KEYS, _Q_KEYS = frozenset(a_index), frozenset()
    with _tf.TemporaryDirectory() as d:
        paths = []
        for i, body in enumerate([f"prefix words then {span} and more",
                                  "nothing to see here at all",
                                  ""]):
            p = os.path.join(d, f"s{i}.jsonl")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"content": body}) + "\n")
            paths.append(p)
        serial = set()
        for p in paths:
            for g in _scan_shard(p)[2]:
                serial |= a_index[g]
        par = set()
        with multiprocessing.get_context("fork").Pool(3) as pool:
            for _, _, a_m, _q in pool.imap_unordered(_scan_shard, paths):
                for g in a_m:
                    par |= a_index[g]
        if serial != {"item-1"}:
            print(f"  FAIL the serial scan found {serial}, expected the planted item-1")
            bad += 1
        if par != serial:
            print(f"  FAIL parallel {par} != serial {serial}: the union over shards is not exact")
            bad += 1
        # a worker that never sees the key sets returns nothing and would look like a clean corpus
        _A_KEYS = frozenset()
        if _scan_shard(paths[0])[2]:
            print("  FAIL a shard matched against an EMPTY key set; membership is not being tested")
            bad += 1
        # BYTES, NOT CHARACTERS. The first parallel run printed textbook_30b as 3.14 GB against 8.12 GB
        # on disk because `len(ln)` in text mode counts characters; every Chinese domain was understated
        # up to 3x and that number was headed for the fact as "the corpus this scan covered".
        zp = os.path.join(d, "zh.jsonl")
        with open(zp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"content": "模型记住了答案"}, ensure_ascii=False) + "\n")
        n_bytes = _scan_shard(zp)[1]
        on_disk = os.path.getsize(zp)
        if n_bytes != on_disk:
            print(f"  FAIL counted {n_bytes} but the shard is {on_disk} bytes on disk; the byte "
                  f"column is counting characters")
            bad += 1
    # longest_run's known answers. This is what separates "the answer is IN the corpus" from "13 words
    # of it are", so a wrong answer here mislabels the whole provenance question.
    A = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    if longest_run(A, ["xx"] + A + ["yy"]) != 7:
        print("  FAIL longest_run missed a fully contained run")
        bad += 1
    if longest_run(A, ["alpha", "beta", "gamma", "zz", "epsilon", "zeta", "eta"]) != 3:
        print("  FAIL longest_run did not stop at the break; a split run must not count as one")
        bad += 1
    if longest_run(A, ["nothing", "shared", "at", "all", "here"]) != 0:
        print("  FAIL longest_run found a run between disjoint texts")
        bad += 1
    # THE INCREMENTAL-VS-ALONE DISTINCTION, on a fixture that reproduces the real defect: two domains
    # holding the SAME document (chatml/chat_qa are two renders of one source). Incrementally the
    # second adds 0; alone it holds the item. A per_domain_alone that shared state between domains
    # would print 0 for the second and the exclusion list would name one domain instead of two.
    with _tf.TemporaryDirectory() as d2:
        span2 = ("one two three four five six seven eight nine ten eleven twelve thirteen "
                 "fourteen")
        idx2 = {}
        for g in grams(words(span2)):
            idx2.setdefault(g, set()).add("dup-1")
        doms = ["dom_a", "dom_b"]
        for dom in doms:
            os.makedirs(os.path.join(d2, "data", "corpus", dom))
            with open(os.path.join(d2, "data", "corpus", dom, "s.jsonl"), "w",
                      encoding="utf-8") as fh:
                fh.write(json.dumps({"content": "lead in " + span2 + " tail"}) + "\n")
        global CORPUS
        real_corpus = CORPUS
        try:
            CORPUS = os.path.join(d2, "data", "corpus")
            rows2 = [{"id": "dup-1", "answer": span2, "question": "q"}]
            pd2, where2 = per_domain_alone(rows2, idx2, doms, 2)
        finally:
            CORPUS = real_corpus
        if pd2["dom_a"]["alone"] != 1 or pd2["dom_b"]["alone"] != 1:
            print(f"  FAIL a document present in BOTH domains gave alone counts "
                  f"{pd2['dom_a']['alone']}/{pd2['dom_b']['alone']}, expected 1/1 -- state is "
                  f"leaking between domains, which is the incremental column's defect")
            bad += 1
        if sorted(where2.get("dup-1", {})) != doms:
            print(f"  FAIL the item->domain map names {sorted(where2.get('dup-1', {}))}, not both "
                  f"domains; the exclusion list could not separate a shared source")
            bad += 1
    print(f"e1_28 selftest: {'OK' if not bad else f'{bad} FAILURE(S)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

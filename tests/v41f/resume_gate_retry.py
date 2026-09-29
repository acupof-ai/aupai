"""Signature-gated retry for the resume gate's known red (see RETRY_* below).

WHY THIS MODULE IS PURE. The retry decision must be a function of measured fields alone, so its
known-answer test needs no torch, no model and no multi-GB dump: every world runs in milliseconds
instead of an hour. The torch-dependent part is one thin extractor,
`extract_obs`, which reads the dump the gate already writes on a mismatch.

FAIL-CLOSED IS THE WHOLE DESIGN. A field that is absent, None, or non-numeric is NOT a match. The
only path to a retry is every predicate explicitly satisfied; a caller that cannot read its evidence
reads NO. Almost every known-answer world below is a refusal, and two of them are the ones that
matter most: the smallest real mutant (so the band cannot drift into mutant space) and an arm whose
RNG differed before the save (so a pre-save bifurcation can never be rescued).

MEASURED BASIS (2026-09-23; artifact 10673482497 of run 35677058625 at d1d4aef3, recomputed from the
raw dumped bytes, not from the log line):
    tag "fp32 master", rel_l2 0.0161551, n_diff 506533/524288 (frac 0.9661), n_nan 0,
    control/rng_atK == restart/rng_preSave bit-equal, bf16 compare did not fail.
The band is set from the mutant bracket measured on the same leaf and node:
    healthy 0.0 (44/44 pairs, 4 host-classes) < red 0.0162 < smallest real mutant 0.1779 (drop_leaf)
so [0.008, 0.030] keeps a >= 5.9x margin to the smallest mutant on both sides. The red is also 9.6x
a single bf16 cast (0.00168), so it cannot be confused with a master-vs-bf16 artifact.
"""

import os

RETRY_TAG = "fp32 master"
RETRY_REL_L2_MIN = 0.008
RETRY_REL_L2_MAX = 0.030
RETRY_MIN_FRAC_DIFF = 0.90   # "near-total per-element": excludes a few-element defect
RETRY_TICKT = "runs/friction.jsonl"

REQUIRED_FIELDS = ("tag", "rel_l2", "n_nan", "frac_diff", "rng_boundary_equal", "bf16_failed")


def retry_signature_matches(obs):
    """(bool, reason). True ONLY for the known numeric signature with every precondition explicit."""
    if not isinstance(obs, dict):
        return False, "obs is not a dict"
    missing = [k for k in REQUIRED_FIELDS if obs.get(k) is None]
    if missing:
        return False, f"UNKNOWN: missing field(s) {missing} (fail-closed)"
    try:
        rel_l2 = float(obs["rel_l2"])
        n_nan = int(obs["n_nan"])
        frac = float(obs["frac_diff"])
    except (TypeError, ValueError):
        return False, "UNKNOWN: non-numeric field (fail-closed)"
    if obs["tag"] != RETRY_TAG:
        return False, f"tag {obs['tag']!r} != {RETRY_TAG!r} (another leaf/side is a real red)"
    if obs["bf16_failed"]:
        return False, "bf16 run-weight compare ALSO failed (not the master-only signature)"
    if n_nan != 0:
        return False, f"n_nan={n_nan} (NaN corruption is never the known signature)"
    if not (RETRY_REL_L2_MIN <= rel_l2 <= RETRY_REL_L2_MAX):
        return False, f"rel_l2={rel_l2:.6g} outside [{RETRY_REL_L2_MIN}, {RETRY_REL_L2_MAX}]"
    if frac < RETRY_MIN_FRAC_DIFF:
        return False, f"frac_diff={frac:.4f} < {RETRY_MIN_FRAC_DIFF} (not near-total)"
    if not obs["rng_boundary_equal"]:
        return False, "control/rng_atK != restart/rng_preSave (arms differed before the save)"
    return True, f"KNOWN SIGNATURE: rel_l2={rel_l2:.6g} frac={frac:.4f} n_nan=0 rng-equal"


# Attempt-2 verdicts. The gate calls the retry ONLY after attempt 1 matched, so a1_obs is the
# already-confirmed known signature; this function decides what a SECOND failure in the same VM
# means.
VM_PERSISTENT = "vm_persistent"      # attempt 2 matches the SAME known signature -> runner-persistent
NEW_FAILURE = "new_failure"          # attempt 2 fails with a DIFFERENT signature -> genuinely new
UNREADABLE = "unreadable_evidence"  # attempt 2 evidence missing/None -> fail-closed, treat as new


def classify_attempt_two(exc1_text, a1_obs, exc2_text, a2_obs):
    """Pure decision for the message the gate prints after a signature-matched attempt 1 fails
    again on attempt 2 (both run in the SAME VM).

    Returns (verdict, reason). Inputs are plain data, so it is testable with no torch, no dumps
    and no model -- the same discipline as retry_signature_matches.

    - VM_PERSISTENT: attempt 2's measured evidence matches the known signature too. The red is a
      stable property of this runner (measured: same sha greens on a different ephemeral runner);
      re-running in place cannot clear it. The caller must still FAIL -- this labels the cause and
      points at a cross-runner job rerun, it never authorizes a pass.
    - NEW_FAILURE: attempt 2 has parseable evidence that does NOT match -- a real new failure that
      must not be conflated with the known red.
    - UNREADABLE: attempt 2 raised but left no usable evidence; fail closed as a new/unknown
      failure rather than assuming the known red.
    """
    if not exc2_text:
        # No second failure: the rescue worked. The caller handles that (writes the tickt, passes);
        # classify_attempt_two is only consulted when attempt 2 failed.
        raise ValueError("classify_attempt_two called without an attempt-2 failure")
    if a2_obs is None:
        return UNREADABLE, "attempt 2 raised but its signature evidence is absent"
    same, why2 = retry_signature_matches(a2_obs)
    if same:
        return VM_PERSISTENT, f"both attempts match the known signature ({why2})"
    return NEW_FAILURE, f"attempt 2 signature does not match: {why2}"


def attempt_two_message(verdict, reason, a1, a2):
    """The exact AssertionError suffix the gate raises for each verdict. Pure so the wording the
    tests assert is the wording the gate emits (no re-stringing at the call site)."""
    if verdict == VM_PERSISTENT:
        return (
            f"[resume-retry] KNOWN VM-CORRELATED RED, persistent on this runner: {reason}. "
            "Re-running inside this VM cannot clear it (measured: the same sha greens on a "
            "different ephemeral runner). The gate stays FAIL on these bytes; the remedy is to "
            "re-run the failed JOB so it lands on a different runner, NOT a third in-place "
            f"attempt. attempt1={a1} attempt2={a2}")
    if verdict == UNREADABLE:
        return (
            f"[resume-retry] attempt 2 failed and its signature evidence cannot be read "
            f"({reason}); failing closed rather than assuming the known red. "
            f"attempt1={a1} attempt2={a2}")
    return (
        f"[resume-retry] attempt 2 failed with a DIFFERENT signature after the known red matched "
        f"on attempt 1 -- this is a NEW failure, not rescued. {reason}. "
        f"attempt1={a1} attempt2={a2}")



def extract_obs(dump_dir, exc_text, bf16_failed, rng_boundary_equal):
    """Measure the signature off the dump the gate wrote on mismatch. Torch kept to one read.

    The gate writes `<dump>/fp32_master.{want,got}.bin` only INSIDE the failing compare
    (test_p1_train_ckpt.py `_eq`), so if those files are absent there is no evidence and this
    returns None fields -> retry_signature_matches says NO.
    """
    obs = {"tag": RETRY_TAG if "fp32 master differs" in (exc_text or "") else None,
           "rel_l2": None, "n_nan": None, "frac_diff": None,
           "rng_boundary_equal": rng_boundary_equal, "bf16_failed": bf16_failed}
    want = os.path.join(dump_dir, "fp32_master.want.bin")
    got = os.path.join(dump_dir, "fp32_master.got.bin")
    if not (os.path.exists(want) and os.path.exists(got)):
        return obs                                          # -> NO, fail-closed
    import torch

    def _ld(p):
        return torch.frombuffer(bytearray(open(p, "rb").read()), dtype=torch.float32).clone()

    w, g = _ld(want), _ld(got)
    obs["rel_l2"] = float((w - g).norm() / w.norm())
    obs["n_nan"] = int(torch.isnan(w).sum() + torch.isnan(g).sum())
    obs["frac_diff"] = float((~torch.eq(w, g)).float().mean())
    return obs


def tickt_row(exc_text, obs, run_id, sha, attempts, dump_dirs):
    """One union-merged ledger row for a RESCUED red.

    `runs/friction.jsonl` is the existing union-merged ledger whose schema matches this event
    (blocked_what / cause / fix_applied / minutes_lost), and it needs no new file and no
    .gitattributes registration. `kind` is the discriminator a grep uses.
    """
    import datetime
    return {
        "when": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "who": "ci:resume-retry",
        "kind": "resume_gate_retry",
        "blocked_what": "gate_resume_equivalent_to_uninterrupted red on the push path",
        "cause": f"known red signature (NOT a fix): {sorted(SIG_DOC)}; signature="
                 f"rel_l2={obs.get('rel_l2')!r} frac={obs.get('frac_diff')!r} n_nan={obs.get('n_nan')}",
        "fix_applied": "attempt 2 passed; attempt 1 dump kept. TOURNIQUET, not a repair",
        "delay": "",
        "minutes_lost": 0,
        "sha": sha,
        "run_id": run_id,
        "attempts": attempts,
        "dumps": list(dump_dirs),
        "first_line": (exc_text or "").strip().splitlines()[-1][:300] if exc_text else "",
        "tickt_open": "N=30 repeat sweep to obtain a reproducer; the red is unexplained until then",
    }


SIG_DOC = ("tag=fp32 master", f"rel_l2 in [{RETRY_REL_L2_MIN},{RETRY_REL_L2_MAX}]",
           f"frac_diff>={RETRY_MIN_FRAC_DIFF}", "n_nan=0", "rng_atK==rng_preSave", "bf16 passed")


# --------------------------------------------------------------------------- known-answer selftest
def _good():
    return {"tag": RETRY_TAG, "rel_l2": 0.0161551405, "n_nan": 0, "frac_diff": 0.966135,
            "rng_boundary_equal": True, "bf16_failed": False}


def selftest(quiet=False):
    """Known-answer worlds, each a one-predicate near-miss, plus the attempt-two classifications.

    Run by the gate list (`gate_resume_retry_signature`), so it is exercised on every CI run
    without a new job. Returns `(failures, counts)`; `failures` empty means every world behaved,
    and `counts` is `{"pos": n, "neg": m}` for the caller to render. The counts are RETURNED rather
    than stated here so neither this docstring nor a caller can carry a number that a later world
    invalidates.
    """
    fails = []
    counts = {"pos": 0, "neg": 0}

    def expect(want_match, obs, name):
        got, reason = retry_signature_matches(obs)
        counts["pos" if want_match else "neg"] += 1
        ok = got == want_match
        if not quiet:
            print(("  ok   " if ok else "  FAIL ") + f"{name}: {reason}")
        if not ok:
            fails.append(f"{name}: expected match={want_match} got={got} ({reason})")

    expect(True, _good(), "MEASURED RED -> retry allowed")

    o = _good(); o["bf16_failed"] = True
    expect(False, o, "master red but bf16 ALSO failed")

    o = _good(); o["rel_l2"] = 0.1779            # drop_leaf, the smallest real mutant
    expect(False, o, "rel_l2 = smallest real mutant (drop_leaf)")

    o = _good(); o["rel_l2"] = 1.1018            # reinit
    expect(False, o, "rel_l2 = reinit mutant")

    o = _good(); o["rel_l2"] = 0.0               # healthy
    expect(False, o, "rel_l2 = 0 (healthy)")

    o = _good(); o["rng_boundary_equal"] = False
    expect(False, o, "rel_l2 in band but RNG differed pre-save")

    o = _good(); o["rng_boundary_equal"] = None
    expect(False, o, "RNG unreadable (None)")

    o = _good(); o["frac_diff"] = 0.02
    expect(False, o, "rel_l2 in band but only 2% of elements differ")

    o = _good(); o["n_nan"] = 7
    expect(False, o, "NaN present")

    o = _good(); o["tag"] = "bf16 run weight"
    expect(False, o, "failure on the bf16 side, not master")

    o = _good(); del o["frac_diff"]
    expect(False, o, "missing field")

    # ---- attempt-two classification: three worlds + mutations -------------------------------
    # genB's review point: the two new messages had no test. The classifier must discriminate the
    # runner-persistent red (same signature) from a genuinely new failure and from missing
    # evidence, and a swapped branch / swapped a1,a2 must turn the world red.
    exc1 = "fp32 master differs after save/load resume: max|delta|=1.334e-02 n_diff=506533/524288"
    a1_obs = _good()

    def classify(a2_text, a2_obs):
        v, _r = classify_attempt_two(exc1, a1_obs, a2_text, a2_obs)
        return v

    # World 1: attempt 2 byte-identical signature -> VM-persistent known red.
    v = classify(exc1, _good())
    ok = v == VM_PERSISTENT
    counts["pos" if ok else "neg"] += 1
    if not quiet:
        print(("  ok   " if ok else "  FAIL ") + f"attempt2 same signature -> {v}")
    if not ok:
        fails.append(f"same-signature attempt2: expected {VM_PERSISTENT} got {v}")

    # World 2: attempt 2 fails with a different signature (a real mutant) -> NEW.
    o2 = _good(); o2["rel_l2"] = 0.1779
    v = classify(exc1, o2)
    ok = v == NEW_FAILURE
    counts["neg"] += 1
    if not quiet:
        print(("  ok   " if ok else "  FAIL ") + f"attempt2 mutant signature -> {v}")
    if not ok:
        fails.append(f"mutant attempt2: expected {NEW_FAILURE} got {v}")

    # World 3: attempt 2 raised but left no readable evidence -> UNREADABLE (fail closed).
    v = classify(exc1, None)
    ok = v == UNREADABLE
    counts["neg"] += 1
    if not quiet:
        print(("  ok   " if ok else "  FAIL ") + f"attempt2 no evidence -> {v}")
    if not ok:
        fails.append(f"unreadable attempt2: expected {UNREADABLE} got {v}")

    # Calling with NO attempt-2 failure is a caller bug, not a verdict.
    raised = False
    try:
        classify_attempt_two(exc1, a1_obs, "", _good())
    except ValueError:
        raised = True
    ok = raised
    counts["neg"] += 1
    if not quiet:
        print(("  ok   " if ok else "  FAIL ") + "no attempt2 failure -> ValueError")
    if not ok:
        fails.append("classify with empty exc2_text must raise ValueError")

    # MUTATION 1: invert the predicate INSIDE the real classifier by monkeypatching
    # retry_signature_matches. With the branch swapped, the same-signature world must come back
    # NEW_FAILURE and the mutant world VM_PERSISTENT -- if swapping the branch does not move the
    # verdict, the test is not discriminating the branch it claims to guard.
    real_rsm = retry_signature_matches
    try:
        def _inverted(obs):
            m, r = real_rsm(obs)
            return (not m), r
        globals()["retry_signature_matches"] = _inverted
        got_same = classify(exc1, _good())
        got_mutant = classify(exc1, o2)
    finally:
        globals()["retry_signature_matches"] = real_rsm
    ok = got_same == NEW_FAILURE and got_mutant == VM_PERSISTENT
    counts["neg"] += 1
    if not quiet:
        print(("  ok   " if ok else "  FAIL ")
              + f"branch-swap mutant flips same->{got_same}, mutant->{got_mutant}")
    if not ok:
        fails.append("inverting the match predicate inside classify_attempt_two did not swap the "
                     f"verdicts (same->{got_same}, mutant->{got_mutant})")

    # MUTATION 2: read a1 where a2 is intended. Feed (a2=mutant) -- the classifier must say NEW;
    # a buggy version that consulted a1 (the known obs) would call this VM-persistent and rescue a
    # real regression. Conversely (a2=known) must be VM-persistent regardless of a1.
    if classify_attempt_two(exc1, o2, exc1, _good())[0] != VM_PERSISTENT:
        fails.append("classifier must read a2_obs for the attempt-2 verdict (a2 known -> persistent)")
    if classify_attempt_two(exc1, _good(), exc1, o2)[0] != NEW_FAILURE:
        fails.append("classifier reading a1 instead of a2 would mislabel a mutant (guard absent)")

    # Every verdict must map to a distinct message carrying the attempt paths.
    for ver in (VM_PERSISTENT, NEW_FAILURE, UNREADABLE):
        msg = attempt_two_message(ver, "r", "/a1", "/a2")
        if "/a1" not in msg or "/a2" not in msg:
            fails.append(f"{ver} message dropped an attempt path")

    if not quiet:
        print(f"  {counts['pos']} positive allowed, {counts['neg']} negatives refused, "
              "fail-closed on missing evidence")
    # (failures, counts). The counts are RETURNED, not left in a print, because the gate that
    # calls this prints its own one-line summary and used to hardcode "1 positive, 10 refusals" --
    # a literal that goes stale the moment a world is added (it did: #668 took the worlds from
    # 11 to 16 and the gate's line kept saying 10). A caller rendering a count must derive it
    # from the run, or the next added world reproduces the same rot.
    return fails, counts


if __name__ == "__main__":
    import sys
    _f, _c = selftest()
    if _f:
        print(f"resume_gate_retry selftest FAILED: {len(_f)}")
        for x in _f:
            print("  - " + x)
    sys.exit(1 if _f else 0)

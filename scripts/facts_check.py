#!/usr/bin/env python3
"""Structural validation for facts/*.json entries, shared by the pre-commit hook
and harness's facts_well_formed check. The source-path and AGENTS-prose halves
live in harness: they need the full tree, a commit-time hook does not."""

import json
import re

REQUIRED = {"id", "value", "measured", "source", "config", "uncertainty", "status"}
STATUS = {"measured", "recorded", "unmeasured", "retracted"}
NEEDS_CLAIM = {"unmeasured", "retracted"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_entry(e, ids, fn):
    """Structural errors for one entry. `ids` is the running id->file map."""
    errors = []
    tag = f"{fn}#{e.get('id', '?')}"
    if missing := REQUIRED - e.keys():
        errors.append(f"{tag}: missing {sorted(missing)}")
        return errors
    if e["status"] not in STATUS:
        errors.append(f"{tag}: bad status {e['status']!r}")
    if not isinstance(e["config"], dict) or not e["config"]:
        errors.append(f"{tag}: config must be a non-empty object")
    if not DATE_RE.fullmatch(str(e["measured"])):
        errors.append(f"{tag}: measured must be YYYY-MM-DD, got {e['measured']!r}")
    if e["status"] in NEEDS_CLAIM:
        for k in ("claim", "audit", "refuted_by"):
            if not e.get(k):
                errors.append(f"{tag}: {e['status']} fact needs {k}")
    if e["id"] in ids:
        errors.append(f"duplicate id {e['id']!r} in {fn} and {ids[e['id']]}")
    ids[e["id"]] = fn
    return errors


def validate_doc(doc, fn):
    """Errors for one facts/*.json document (a {'facts': [...]} object)."""
    try:
        lst = json.loads(doc)["facts"]
        assert isinstance(lst, list) and lst
    except Exception as e:
        return [f"{fn}: no readable non-empty 'facts' list ({e})"]
    ids, errors = {}, []
    for e in lst:
        if not isinstance(e, dict):
            errors.append(f"{fn}: entry is not an object")
            continue
        errors += validate_entry(e, ids, fn)
    return errors

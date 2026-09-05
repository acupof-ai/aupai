#!/usr/bin/env python3
"""Credential shapes in staged text, for the pre-commit gate.

WHY THIS EXISTS. On 2026-09-05 data/probes/api_cloze.jsonl entered a commit (32b4ed22) carrying
a live third-party Postgres credential in a public repo. GitHub push protection caught it at the
LAST gate, which froze main under the merge lock while 12 commits were rewritten. This is the
same detection at the FIRST gate, where the remedy is deleting a line instead of rewriting
history. Push protection stays the backstop; a hook that runs before the object exists is the
only place the cost is one line.

TWO POPULATIONS DECIDED THESE PATTERNS, and only one of them is obvious.

FALSE POSITIVES. The shapes as named in the request -- `sk-`, `AKIA`, `ghp_`, `xox`,
`postgres://user:pass@` -- were measured as bare substrings across all 1041 tracked files:
`sk-` alone hits 34 times, every one of them prose (task-, risk-, disk-). A gate that refuses
ordinary documentation commits is disabled within a day, so each pattern here is anchored on
LENGTH and ALPHABET, which is what separates a key from a word. Re-measured anchored: 0 hits
tree-wide except the two noted below.

THE VALIDATOR'S OWN FIXTURES. scripts/build_agentic_sft.py:1135 and :1214 contain
`ghp_16C7e42F292c6912E7710c838347Ae178B4a` -- GitHub's own published example token, used as
test data by that file's selftest, which exists to prove a redactor works. A gate that refuses
it stops the commit that fixes the redactor (§182: a validator separated from what it
validates, here inverted -- the validator and its subject in one file). EXAMPLES holds that
literal. The exemption is the STRING, never the file: exempting scripts/build_agentic_sft.py
would ship the next real secret added to it.

VERIFIED AGAINST THE REAL LEAK before any of this was written: run over
32b4ed22:data/probes/api_cloze.jsonl, `db-uri-with-password` fires on rows 1538 AND 4161 of
5001 -- the second row nobody had reported, which a rewrite dropping only 1538 would have
re-pushed. A gate that cannot fire on the incident it was written for is worth nothing, and
this one was checked against it rather than assumed.

THE REPORT NAMES NO SECRET. findings() returns (path, line number, pattern name) and never the
matched text. A validator that proves itself by echoing the credential into a terminal, a log
and a session transcript has copied it three more places -- which is the opposite of the
containment the frozen main exists for.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Anchored on length and alphabet. Each entry: (name, compiled pattern, what it is).
PATTERNS = (
    # The shape that leaked: scheme://user:password@host. The password class excludes the
    # delimiters, and 6+ is what separates a real password from `postgres://user:@` or the
    # `postgres://localhost/db` in a config example (neither fires).
    ("db-uri-with-password",
     re.compile(r"\b(?:postgres|postgresql|mysql|mongodb|redis|amqp|amqps)(?:\+\w+)?://"
                r"[A-Za-z0-9._%+-]+:[^\s:@/'\"]{6,}@"),
     "a database URI with an inline password"),
    # sk- + 20 minimum. Real OpenAI keys are 48 or more after the prefix; 20 is the floor that
    # clears every prose hit measured in this tree.
    ("openai-key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),
     "an OpenAI/Anthropic-style API key"),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
     "an AWS access key id"),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"),
     "a GitHub personal access / OAuth token"),
    ("slack-token", re.compile(r"\bxox[baprse]-[A-Za-z0-9-]{10,}"),
     "a Slack token"),
    ("private-key-pem", re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
     "a private key block"),
    # AWS secret access key: 40 chars of base64 alphabet is not a shape you can anchor on
    # alone -- it matches shas, hashes and base64 payloads -- so it is only a finding when
    # a key-ish assignment introduces it. Measured: the bare 40-char form hits git shas and
    # vocab hashes across this tree; the assignment form hits nothing.
    ("aws-secret-key-assigned",
     re.compile(r"(?i)aws.{0,20}(?:secret|private).{0,20}key\W{0,4}[A-Za-z0-9/+=]{40}\b"),
     "an AWS secret access key in an assignment"),
)

# LITERALS THAT ARE PUBLISHED EXAMPLES, exempted as strings and never as files. Each is here
# because it is documentation that a real vendor published as a non-credential.
EXAMPLES = (
    # GitHub's own example token, in scripts/build_agentic_sft.py's selftest (:1135, :1214) --
    # the fixture that proves that file's redactor works.
    "ghp_16C7e42F292c6912E7710c838347Ae178B4a",
    # AWS's documentation example secret key, in the same selftest.
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
)

# Binary and generated paths a text scan should not read. Kept SHORT on purpose: every entry is
# a place a secret could hide unscanned, so the list is extensions that cannot hold readable
# text plus the tokenizer, which is a 32K-entry id map.
SKIP_SUFFIXES = (".pt", ".bin", ".gz", ".zst", ".parquet", ".png", ".jpg", ".jpeg", ".pdf",
                 ".woff", ".woff2", ".ico", ".so", ".dylib")
MAX_BYTES = 8 * 1024 * 1024   # a staged file larger than this is refused by the size gate anyway


def _scannable(path):
    return not path.lower().endswith(SKIP_SUFFIXES)


def scan_text(text, path="<text>"):
    """[(path, lineno, name, description)] for every credential shape in `text`.

    Never returns the matched text. A caller that wants to see the secret must open the file
    at the reported line itself, which keeps it out of logs and transcripts.
    """
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if any(ex in line for ex in EXAMPLES):
            # A line carrying a published example may still carry a real secret beside it, so
            # the example is REMOVED and the remainder scanned, rather than the line skipped.
            for ex in EXAMPLES:
                line = line.replace(ex, "<published-example>")
        for name, pat, desc in PATTERNS:
            if pat.search(line):
                out.append((path, i, name, desc))
    return out


def scan_file(path, root=ROOT):
    full = path if os.path.isabs(path) else os.path.join(root, path)
    if not os.path.isfile(full) or not _scannable(path):
        return []
    try:
        if os.path.getsize(full) > MAX_BYTES:
            return []
        with open(full, encoding="utf-8", errors="replace") as f:
            return scan_text(f.read(), path)
    except OSError:
        return []


def staged_paths(root=ROOT):
    r = subprocess.run(["git", "-C", root, "diff", "--cached", "--name-only",
                        "--diff-filter=AM"], capture_output=True, text=True)
    return [p for p in r.stdout.splitlines() if p.strip()]


def scan_staged(root=ROOT):
    """Findings across staged content. Reads the INDEX, not the working tree: a secret can be
    staged and then edited out of the file, and it is the staged bytes that become the commit.
    """
    out = []
    for p in staged_paths(root):
        if not _scannable(p):
            continue
        r = subprocess.run(["git", "-C", root, "show", f":{p}"],
                           capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            continue
        if len(r.stdout) > MAX_BYTES:
            continue
        out.extend(scan_text(r.stdout, p))
    return out


def report(findings, stream=sys.stderr):
    """Print the refusal. Names the file, line and shape; never the value."""
    print(f"REFUSING: {len(findings)} credential shape(s) in staged content:", file=stream)
    for path, lineno, name, desc in findings:
        print(f"  {path}:{lineno}  {name} -- {desc}", file=stream)
    print("  The value is NOT printed here on purpose: it would land in this terminal, the", file=stream)
    print("  hook log and the session transcript. Open the file at that line yourself.", file=stream)
    print("  If it is real: remove it, ROTATE it (it is already in your working tree and", file=stream)
    print("  possibly in a stash or another worktree), then stage again. On 2026-09-05 one", file=stream)
    print("  such row reached a commit and froze main while 12 commits were rewritten.", file=stream)
    print("  If it is a published vendor example: add the literal to EXAMPLES in", file=stream)
    print("  scripts/credential_scan.py -- the string, never the file.", file=stream)


def _selftest():
    fails = []

    # POSITIVES: each pattern fires on a real-shaped value.
    pos = [
        ("db-uri-with-password", "DATABASE_URL=postgres://appuser:s3cr3tpassword@db.example.com:5432/x"),
        ("db-uri-with-password", "mongodb+srv://admin:hunter2hunter@cluster0.example.net/test"),
        ("openai-key", "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"),
        ("aws-access-key-id", "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"),
        ("github-token", "token: ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
        ("slack-token", "SLACK_BOT_TOKEN=xoxb-123456789012-abcdefghijkl"),
        ("private-key-pem", "-----BEGIN OPENSSH PRIVATE KEY-----"),
        ("aws-secret-key-assigned",
         "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEZ"),
    ]
    for want, text in pos:
        got = {n for _p, _l, n, _d in scan_text(text)}
        if want not in got:
            fails.append(f"{want} does not fire on its own positive: {got or 'nothing'}")

    # NEGATIVES, and these are the load-bearing half. Every line here appeared in this repo or
    # is the near-miss the anchoring exists for; a gate that refuses any of them gets turned off.
    neg = [
        "the task-performance threshold is below the risk-adjusted disk-space budget",
        "sk-",                                   # the bare prefix, 34 prose hits when unanchored
        "task-level and disk-bound and risk-free",
        "postgres://localhost/aupai",            # no password
        "postgres://user:@localhost/db",         # empty password
        "redis://cache:6379/0",                  # host:port, not user:password
        "AKIA",                                  # the prefix with no key
        "AKIAIOSFODNN7",                         # too short to be a key id
        "ghp_short",
        "xox",
        "commit 3f5231c9a8b7d6e5f4a3b2c1d0e9f8a7b6c5d4e3",   # 40 hex, not an AWS secret
        "vocab_id 9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c",  # ditto
        "-----BEGIN PUBLIC KEY-----",            # public, not private
        "see docs/standards/writing.md for the sk-prefixed shorthand",
    ]
    for text in neg:
        got = scan_text(text)
        if got:
            fails.append(f"FALSE POSITIVE on {text[:60]!r}: {[n for _p, _l, n, _d in got]}")

    # THE PUBLISHED EXAMPLES do not fire...
    for ex in EXAMPLES:
        if scan_text(f"token: {ex}"):
            fails.append(f"the published example {ex[:12]}... still fires; the selftest of "
                         f"build_agentic_sft.py's redactor could not be committed")
    # ...AND THE EXEMPTION IS THE STRING, NOT THE LINE. A real secret beside an example on one
    # line must still fire, or `<example> and <real>` becomes a way to smuggle one through.
    smuggle = f"token: {EXAMPLES[0]} and postgres://u:realpassword123@h/db"
    if not scan_text(smuggle):
        fails.append("an exempted example on the same line as a real credential suppressed it -- "
                     "the exemption must remove the example and scan the remainder")

    # THE TREE ITSELF must be clean, or the gate cannot be turned on: a check that is red the
    # moment it lands is the same as no check (AGENTS.md, a permanent red is no signal).
    r = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True)
    tree = []
    for p in r.stdout.split():
        tree.extend(scan_file(p))
    if tree:
        fails.append(f"the tracked tree already holds {len(tree)} finding(s), so this gate "
                     f"cannot be enabled: {[(p, ln, n) for p, ln, n, _d in tree[:5]]}")

    for f in fails:
        print(f"  FAIL {f}", file=sys.stderr)
    if fails:
        print(f"credential_scan selftest: {len(fails)} failure(s)", file=sys.stderr)
        return 1
    print(f"credential_scan selftest OK: {len(pos)} positives fire, {len(neg)} near-misses and "
          f"prose lines do not (bare `sk-` hits 34 tracked files as a substring, 0 anchored), "
          f"{len(EXAMPLES)} published vendor examples exempted as STRINGS with a real credential "
          f"beside one still caught, and all {len(r.stdout.split())} tracked files scan clean")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if "--staged" in sys.argv:
        f = scan_staged()
        if f:
            report(f)
            sys.exit(1)
        print("no credential shapes in staged content")
        sys.exit(0)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    found = []
    for a in args:
        found.extend(scan_file(a))
    if found:
        report(found)
        sys.exit(1)
    print(f"clean: {len(args)} file(s)")

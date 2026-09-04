#!/usr/bin/env python3
"""Claude Code session transcripts -> agentic SFT pairs (e1-24, stage 1 of the RL loop).

Reads ~/.claude/projects/*/*.jsonl -- this machine's own Claude Code sessions -- and emits
ChatML turn sequences that scripts/loader.format_agentic can pack: assistant text
supervised, tool output entirely masked, tool turns as plain-text <|im_start|>tool turns.

WHAT THE TRANSCRIPT IS, measured rather than assumed (342 session files, 5.4 GiB total
including 8219 subagent sidecars under */subagents/, which this does NOT read -- a
subagent's transcript is a different conversation with its own system prompt, and mixing
it into the parent's turn sequence would train a model to answer prompts it was never
shown). Content-block census over the first 40 sessions:

    assistant:tool_use    63860      user:tool_result   63856
    assistant:thinking    50995      assistant:text     20203
    user:str               5012      user:text            489

So the dominant shape is not prose, it is the tool loop -- which is the point: this is
meant to teach an agent to call tools and continue from their output.

THREE DECISIONS THAT ARE NOT OBVIOUS, each named because a later reader will wonder:

  thinking blocks are DROPPED. 50,995 of them, and they are the model's own reasoning in a
  format the target model has never emitted. Supervising them would train a model to
  produce a block type its inference path does not render; masking them but keeping them
  in the prompt would train it to expect them in context. Neither is what an agent loop
  looks like at inference, so they leave.

  TOKEN IDS ARE NOT THIS FILE'S JOB. fb's ruling, 2026-09-02: the shim writes prompt token
  ids, completion token ids, per-token logprobs and a request id per request, and stage 4's
  torch.equal reads those two files. Text cannot carry that responsibility -- BPE is not
  concatenation-invariant, tokenize(A+B) != tokenize(A)+tokenize(B) in general, and ChatML
  rendering is exactly a concatenation, so a sequence rebuilt from text is an approximation
  and GRPO on the wrong token string is a silent wrong gradient rather than a crash. This
  file therefore keeps original text boundaries and normalizes NOTHING (no whitespace
  collapsing, no unicode folding, no path rewriting beyond redaction), so the text it emits
  is byte-faithful to what was rendered.

  REDACTION IS A SCANNER, NOT A REGEX I WROTE. detect_secrets' plugin set, which carries
  entropy detectors and keyword rules maintained by people who do this full time. My own
  pattern list would only catch the shapes I happened to imagine; fb's brief says "use a
  scanner, not your eyes" and a hand-rolled regex is eyes with extra steps. Absolute paths
  and emails go through explicit patterns because those are structural, not secret-shaped.

    python scripts/build_agentic_sft.py --limit 10000 --out data/sft/agentic_v1.jsonl
    python scripts/build_agentic_sft.py --selftest

# This file first landed inside merge commit 34982c2 rather than as its own commit: an
# earlier `git merge main` had not been completed, so `git commit -F` attached my message to
# the merge. `git log -S SYNTHETIC_USER -- scripts/build_agentic_sft.py` therefore returns
# NOTHING -- log -S does not read merge diffs, which is the same blind spot that made
# merge_complete stand red all morning (de-22). Not rewritten: 34982c2 is already a merge
# others may build on, and a soft reset staged nine other people's commits as mine. This
# comment is the touch that makes the file findable by content search.
#
# restartable: an interrupt costs 18s. The whole build is a pure function of the session
# files -- it holds nothing on the pod, writes one file at the very end, and re-running it
# reproduces the same pack byte for byte (no randomness, no timestamps in the rows). There
# is nothing to resume, so per-shard writes would add a checkpoint format for no gain.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SESSIONS = os.path.expanduser("~/.claude/projects/*/*.jsonl")

# Structural scrubs. Not secrets -- identifiers that would teach the model this machine's
# filesystem and its owner's address. Ordered longest-first so /Users/<name>/.claude does
# not become <HOME>/.claude after a shorter pattern already fired.
HOME = os.path.expanduser("~")
USER = os.path.basename(HOME)
SCRUBS = [
    (re.compile(re.escape(HOME)), "<HOME>"),
    (re.compile(r"/Users/[A-Za-z0-9._-]+"), "<HOME>"),
    (re.compile(r"/home/[A-Za-z0-9._-]+"), "<HOME>"),
    # Claude Code encodes a project directory by replacing every "/" with "-", so
    # /Users/<user>/code/x becomes "-Users-<user>-code-x". The slash patterns above cannot
    # see that form, and it appears constantly: in tool-result paths under
    # ~/.claude/projects/, and as the `project` field of every single row. Found by reading
    # 50 sampled rows -- 20 of them leaked the username this way, and 50 of 50 leaked it in
    # `project`. The mechanical checks passed because they searched for "/Users/".
    # No lookahead at all: the segment after "-Users-" IS the username, wherever it ends.
    # Two narrower drafts each left a real leak, both found by scanning the built pack --
    # requiring a following "-<letter>" missed "-Users-bytedance" at end of string (an
    # `ls -la` listing), and adding an "or non-word" alternative still missed
    # "--Users-bytedance--" because the trailing hyphen matched the class it was excluding.
    # The username never contains "-" in the encoded form, so [A-Za-z0-9._]+ ends it.
    (re.compile(r"(-Users-)[A-Za-z0-9._]+"), r"\1<USER>"),
    (re.compile(r"(-home-)[A-Za-z0-9._]+"), r"\1<USER>"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
]

#: Internal topology: URLs naming the org's own hosts. Not credentials and not the user's
#: identity, which is why SCRUBS did not cover them -- but this pack becomes weights, and a
#: 500M trained on 222 turns carrying internal doc URLs can emit one. fb's ruling
#: (2026-09-02): replace the WHOLE URL, keep the sentence, because the training signal is
#: "the assistant fetched a doc and continued from the result" and that survives redaction
#: intact. Dropping the episodes instead would throw away 6% of the pack's real tool loops.
#:
#: A DOMAIN PREDICATE, not a list of the URLs I happened to see (fb's wording): any host
#: ending in one of the internal suffixes, whatever the subdomain or path. A list would go
#: stale the first time someone used a new subdomain, and it would look like it still worked.
INTERNAL_HOSTS = ("larkoffice.com", "bytedance.net", "feishu.cn")
INTERNAL_DOC_URL = re.compile(
    r"https?://[A-Za-z0-9.-]*(?:" + "|".join(h.replace(".", r"\.") for h in INTERNAL_HOSTS)
    + r")(?:/[^\s\"'>)\]},]*)?")
#: github.com/<org>/... where <org> is one of the org's own. Deliberately narrow: this
#: matched `bytedance-iaas` and `bytedance-inc` style paths in this corpus, and a public
#: repo that merely mentions the name in its project title is not a path segment.
INTERNAL_REPO_URL = re.compile(r"https?://(?:www\.)?github\.com/bytedance[A-Za-z0-9_-]*(?:/[^\s\"'>)\]},]*)?")
#: Bare host references with no scheme -- `bits.bytedance.net/code/x` appears without
#: https:// in shell output, and a scheme-anchored pattern would leave every one of them.
INTERNAL_BARE = re.compile(
    r"\b[A-Za-z0-9.-]*(?:" + "|".join(h.replace(".", r"\.") for h in INTERNAL_HOSTS)
    + r")(?:/[^\s\"'>)\]},]*)?")


def scrub(text):
    """Paths, emails, and internal URLs out. Secrets are handled by find_secrets.

    Order matters: the repo pattern runs before the doc patterns only because they cannot
    overlap (github.com is not an internal host), but the BARE host pattern must run LAST --
    it is a superset of the scheme-anchored one, and running it first would leave `https://`
    dangling in front of the placeholder.
    """
    for pat, repl in SCRUBS:
        text = pat.sub(repl, text)
    text = INTERNAL_REPO_URL.sub("<INTERNAL_REPO>", text)
    text = INTERNAL_DOC_URL.sub("<INTERNAL_DOC>", text)
    text = INTERNAL_BARE.sub("<INTERNAL_DOC>", text)
    return text


# detect_secrets' plugin set, pinned here so a scan result means the same thing next month.
# Provider detectors plus KeywordDetector plus the two entropy detectors -- the entropy
# pair is what catches an AWS *secret* key and a `password = <blob>` assignment, which the
# keyword rules alone miss (measured: 1 of 4 real cases caught without them, 4 of 4 with).
SECRET_PLUGINS = [{"name": n} for n in (
    "AWSKeyDetector", "AzureStorageKeyDetector", "BasicAuthDetector", "CloudantDetector",
    "DiscordBotTokenDetector", "GitHubTokenDetector", "GitLabTokenDetector",
    "IbmCloudIamDetector", "IbmCosHmacDetector", "JwtTokenDetector", "KeywordDetector",
    "MailchimpDetector", "NpmDetector", "OpenAIDetector", "PrivateKeyDetector",
    "PypiTokenDetector", "SendGridDetector", "SlackDetector", "SoftlayerDetector",
    "SquareOAuthDetector", "StripeDetector", "TelegramBotTokenDetector", "TwilioKeyDetector",
)] + [{"name": "Base64HighEntropyString", "limit": 4.5},
      {"name": "HexHighEntropyString", "limit": 4.5}]


def find_secrets(text, chunk=1):
    """Secret types detect_secrets finds in `text`. [] is clean, None means it could not run.

    SCANS FILES, not lines, and in CHUNKS -- both halves were measured, and each is a
    correction of the obvious approach:

      scan_line applies no filters at all, so it reports every ordinary word as a Base64
      high-entropy candidate: "hello world, nothing here" came back with four hits. A
      detector that fires on prose cannot answer "0 hits" about a corpus of prose.

      scan_file applies the filters (prose, Chinese, code, log lines, shas and absolute
      paths all come back 0), but on a multi-line document it stops early: a file holding
      an AWS secret key, a ghp_ token and a `password =` line reports ONLY the GitHub
      token, at line 2, with lines 1 and 3 never surfacing. Each of those three lines is
      detected when scanned alone. Spacing them 12 lines apart changes nothing, so it is
      not the context window. Whatever the mechanism, one document-wide scan UNDER-REPORTS,
      and "0 hits over the whole pack" would have been a claim about the first hit only.

    So: chunk the text, scan each chunk as its own file, union the types. Chunks are lines
    rather than bytes so a secret is never split across a boundary, and chunk=1 is not a
    conservative guess -- it is where the loss stops. Three secrets buried in 60 prose
    lines, same plugin set:

        chunk=1    AWS Access Key, Base64 High Entropy String, GitHub Token, Secret Keyword
        chunk=2    GitHub Token, Secret Keyword
        chunk>=5   GitHub Token

    Every chunk above 1 silently drops findings, so speed cannot be bought here: the cost
    is ~2ms/line (measured, 500 lines in 0.99s), about ten minutes for a 10K-pair pack,
    which is a one-off gate rather than a hook. A faster scan that reports fewer secrets is
    not a faster scan, it is a different and wrong answer.

    None rather than [] when the package is absent: a tool that did not run is not a clean
    result, and the caller reports that as its own outcome -- the shape e1-16 hit with
    ruff, where an unrunnable scanner read as green.

    ONE temp file per call, rewritten per chunk, not one per line. The per-line version
    created and unlinked ~1.9M files for a full pack; each is cheap on its own, but the
    run died at `OSError: [Errno 28] No space left on device` inside NamedTemporaryFile
    when an unrelated process filled the disk -- a scan holding a single reused path has
    nothing to lose to that. The chunking above is unchanged: chunk=1 is still where the
    loss stops, and this touches only how the chunk reaches scan_file.
    """
    try:
        import tempfile

        from detect_secrets.core import scan
        from detect_secrets.settings import transient_settings
    except ImportError:
        return None
    lines = text.splitlines()
    found = set()
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with transient_settings({"plugins_used": SECRET_PLUGINS}):
            for i in range(0, max(len(lines), 1), chunk):
                block = "\n".join(lines[i:i + chunk])
                if not block.strip():
                    continue
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(block + "\n")
                found.update(s.type for s in scan.scan_file(path))
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return sorted(found)


def block_text(content):
    """A content field -> its text, whatever shape it arrived in.

    Three shapes appear in real transcripts: a bare string, a list of typed blocks, and
    (inside tool_result) a list of blocks that is itself the content. Anything not text --
    images, unknown block types -- returns "" rather than str(block): a dict rendered into
    the prompt would teach the model to emit Python repr.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for blk in content:
        if isinstance(blk, str):
            parts.append(blk)
        elif isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(blk.get("text") or "")
    return "".join(parts)


def tool_call_text(blk):
    """A tool_use block as the text an assistant turn would have emitted.

    The transcript stores the call structurally (name + input dict); the model has to
    produce it as text. JSON with sorted keys and no ASCII escaping: sorted so the same
    call renders identically every time (dict order in a log is not a contract), and
    ensure_ascii=False because the corpus is half Chinese and \\uXXXX escapes would teach
    the model to emit them.
    """
    return json.dumps({"tool": blk.get("name"), "input": blk.get("input")},
                      ensure_ascii=False, sort_keys=True)


def turns_from_session(path):
    """One session file -> [messages], each a format_agentic-shaped turn list.

    A session is cut into episodes at every user turn that is not a tool_result: that is
    where a human actually spoke, and everything until the next one is one task. Cutting
    per-session instead would build 800-turn sequences that cannot fit 4096 tokens; not
    cutting at all would join unrelated tasks into one conversation.

    TOOL RESULTS ARE PAIRED BY tool_use_id, NOT BY POSITION, and that is the correction
    that matters most in this file. Claude Code writes a call and its result as separate
    records, and the result does NOT reliably follow its own call: in the very first sampled
    row, `gh pr ready 662 --undo` and a `command -v` probe were emitted as two consecutive
    assistant turns, then one tool turn arrived carrying the FIRST call's output. Appending
    each result after whatever assistant turn happened to be last produced 25 of 50 sampled
    rows where a tool turn sits behind the wrong call -- a model trained on that learns to
    continue from a result belonging to a different tool, which is worse than no data.

    So a call is buffered until its result arrives, and the pair is emitted together:
    assistant(call) immediately followed by tool(result for THAT id). A call whose result
    never appears (session ended, tool interrupted) is dropped rather than left dangling --
    an assistant turn asking for output that never comes teaches the model to expect
    silence. Text-only assistant turns pass through in order; they have no result to wait
    for.
    """
    episodes, cur = [], []
    # id -> the assistant turn text for that call, awaiting its result
    open_calls = {}

    def flush_unanswered():
        """Calls whose results never arrived. Dropped, and counted by the caller's report."""
        open_calls.clear()

    with open(path, encoding="utf-8", errors="replace") as fh:
        lines_in = list(fh)
    for line in lines_in:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        role, content = msg.get("role"), msg.get("content")

        if role == "user":
            results = [b for b in content
                       if isinstance(b, dict) and b.get("type") == "tool_result"] \
                if isinstance(content, list) else []
            if results:
                for r in results:
                    call = open_calls.pop(r.get("tool_use_id"), None)
                    if call is None or not cur:
                        continue  # a result for a call we never saw has no antecedent
                    cur.append({"role": "assistant", "content": call})
                    cur.append({"role": "tool", "content": block_text(r.get("content"))})
                continue
            text = block_text(content)
            if not text.strip():
                continue
            flush_unanswered()
            if cur:
                episodes.append(cur)
            cur = [{"role": "user", "content": text}]
        elif role == "assistant" and cur:
            if not isinstance(content, list):
                continue
            text_parts, calls = [], []
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "text":
                    text_parts.append(blk.get("text") or "")
                elif blk.get("type") == "tool_use":
                    calls.append((blk.get("id"), tool_call_text(blk)))
                # thinking: dropped, see the module docstring
            text = "".join(text_parts)
            if calls:
                # One assistant MESSAGE can carry prose AND a call; they are one turn, so
                # the prose rides with the call rather than being emitted first and the
                # call later -- that ordering produced a duplicate assistant turn and the
                # selftest caught it ("3 pairs for 2 assistant turns").
                head, (cid, call) = text, calls[0]
                open_calls[cid] = (head + call) if head.strip() else call
                # A second call in the same message is rare (21 of 63,832 assistant
                # messages) and cannot be ordered against the first from the transcript
                # alone, so extra calls are dropped rather than guessed at.
            elif text.strip():
                cur.append({"role": "assistant", "content": text})
    flush_unanswered()
    if cur:
        episodes.append(cur)
    return episodes


# A "user" record that is not a human speaking. Claude Code delivers background-task
# notifications, command stdout, hook output and system reminders through the user role, so
# a naive parser reads them as instructions. Measured on the first 10K-pair build: 3567 of
# 10000 rows (35.7%) OPENED on one of these, 3558 of them <task-notification>. An episode
# rooted in machine noise teaches the model to answer machine noise, and the whole point of
# this pack is to teach it to answer a person. Found by reading sampled rows -- every
# mechanical check passed, because the text is well-formed and correctly attributed.
SYNTHETIC_USER = re.compile(
    r"<(?:task-notification|local-command-stdout|local-command-stderr|bash-stdout|"
    r"bash-stderr|local-command-caveat|command-name|command-message|command-args|"
    r"system-reminder|cross-session-message|user-prompt-submit-hook)>"
    r"|Caveat: The messages below"
    r"|Another Claude session sent a message"
    r"|\[SYSTEM NOTIFICATION"
    # The harness's own nudges, delivered as a user turn in square brackets: "[Your
    # previous response had no visible output. Please continue...]". 19 of 4814 rows
    # (0.4%) opened on one. Caught while redacting three samples for 3b -- the first
    # sample I picked started with it, so a hand-read of three rows found what a filter
    # measuring 23,466 episodes had missed. The bracket form is the discriminator: a
    # person does not open a request with "[Your ...".
    r"|\[(?:Your|The|This) [a-z]")


CHATML_LITERAL = re.compile(r"<\|im_(?:start|end)\|>")
UNACTIONABLE_OPENER = re.compile(r"\s*(?:\[Image:[^\]]*\]\s*)+$|\s*/[a-z][a-z-]{1,20}\s*$")
NON_ANSWER = re.compile(r"\s*(?:No response requested\.?|\(no response\)|)\s*$")
#: The HARNESS speaking in the assistant's voice. A transcript records an API failure as an
#: assistant message, so it packs as a supervised completion and teaches the model to answer
#: with an error string. 58 of 4349 episodes (1.3%), and in all 58 it is the LAST supervised
#: turn -- the strongest position in the episode. Found by hand-reading the three samples for
#: 3b, after the credential scan and every structural filter had passed them.
HARNESS_ERROR = re.compile(
    r"\s*(?:API [Ee]rror|Request timed out|\[?Request interrupted by user"
    r"|Prompt is too long|Claude's response was cut off"
    # The CLIENT talking to its user in the assistant's voice: quota, auth, session
    # limits. Same defect as an API error, different vocabulary, and found the same way --
    # by reading row 34 of the 50-row sample after the API-error rule was already in.
    r"|You're out of usage credits|You've (?:hit|reached) your"
    r"|Please run /|Run /usage-credits|Credit balance)")
#: A tool invocation that was never parsed, left sitting in the assistant's TEXT. The
#: recording client emits tool calls as structured blocks; when one is malformed the client
#: cannot parse it, so the raw markup lands in the text block instead -- and it packs as
#: supervised text, teaching the model to write a broken invocation in prose. 7 turns.
#:
#: LINE-ANCHORED, deliberately. 3 of the 10 turns mentioning this markup are prose ABOUT
#: tool syntax ("does it contain any tool-call syntax (JSON tool_call / XML <function_calls>
#: ...)", and a review noting a Grep argument was dirty) -- exactly the discussion this
#: repo's transcripts are full of, and exactly what a substring match would eat. The same
#: shape as CHATML_LITERAL: an explanation is not an instance.
BROKEN_INVOCATION = re.compile(r"(?:^|\n)\s*<(?:antml:)?(?:invoke|function_calls)\b")
#: The human said NO to this tool call. The rejection is recorded as the tool RESULT, so it
#: is masked and looks harmless -- but the assistant turn before it is the refused call, and
#: that turn is supervised as a completion. 132 refused calls across 125 episodes (2.9%) are
#: currently teaching the model to emit exactly the calls a person vetoed. The negative
#: signal is real supervision in the wrong direction, and it is invisible to every check
#: that looks at the tool turn (which is correctly masked) rather than the turn before it.
REJECTED_CALL = re.compile(
    r"The user doesn't want to proceed with this tool use|tool use was rejected")




def is_synthetic_user(text, window=600):
    """True when this "user" turn is a machine event rather than a person.

    SEARCHES the opening region; it does not anchor at the start. An anchored version
    caught the 3558 rows that begin with a bare <task-notification> and missed 1765 more
    (25.1% of the surviving pack) where the wrapper is introduced by a prose sentence --
    "Another Claude session sent a message:" followed by <cross-session-message>. Anchoring
    asks "does this text begin with a marker", and the question is "is this turn a machine
    event", which the sentence answers just as well as the tag.

    The window bounds it: a marker 40KB into a long human message is being quoted or
    discussed (this repo's own transcripts do that constantly), not delivered, and a whole-
    text search would drop the genuine turns that talk about the machinery.
    """
    return bool(SYNTHETIC_USER.search(text[:window]))


def usable(messages):
    """(ok, reason). An episode format_agentic can actually turn into pairs.

    Rejects rather than repairs: an episode that needs repair is an episode whose shape I
    guessed at. Each reason is counted in the report so the discard mix is visible instead
    of being a single "dropped N".
    """
    if len(messages) < 2:
        return False, "fewer than 2 turns"
    if messages[0]["role"] != "user":
        return False, "does not open with a user turn"
    # A transcript DISCUSSING ChatML cannot be packed AS ChatML. The tokenizer maps a
    # quoted "<|im_start|>" to the real special token (id 32768, verified) -- it has no way
    # to tell a quotation from a delimiter -- so an episode whose text mentions the markers
    # injects fake role boundaries into a supervised completion, and the model learns to
    # emit turn delimiters mid-answer. Found by running the 5264-pair pack through
    # format_agentic: 3 pairs in row 3311 had <|im_start|>tool INSIDE a completion, all
    # from one session that was debugging the packer. 11 of 5264 episodes (0.21%) carry a
    # literal marker; they leave. This repo's own transcripts discuss ChatML constantly,
    # which is exactly why the rate is not zero.
    # An opener the model cannot act on. Three shapes, all found by hand-reading the three
    # samples I was redacting for 3b -- reading three rows caught what filters measuring
    # 23,466 episodes had passed:
    #   image-only    "[Image: source: <HOME>/.claude/image-cache/...]" and nothing else.
    #                 201 of 4798 rows (4.2%). block_text() drops image blocks by design, so
    #                 the model is asked to respond to input it cannot see, and it learns to
    #                 invent an answer for an empty request.
    #   slash command "/compact" alone: an instruction to the CLI, not to the model.
    #   non-answer    the assistant replying "No response requested." -- 84 rows (1.8%)
    #                 whose only supervised text teaches the model to decline.
    if UNACTIONABLE_OPENER.match(messages[0]["content"]):
        return False, "opener the model cannot act on (image-only / slash command)"
    if all(NON_ANSWER.match(m["content"]) for m in messages if m["role"] == "assistant"):
        return False, "every assistant turn is a non-answer"
    # The whole episode goes, not just the turn: in all 58 cases the error IS the final
    # assistant turn, so truncating leaves the episode ending on a tool turn with its result
    # never used -- which trains the model to call a tool and abandon it.
    if any(m["role"] == "assistant" and HARNESS_ERROR.match(m["content"]) for m in messages):
        return False, "an assistant turn is a harness error, not a response"
    if any(m["role"] == "assistant" and BROKEN_INVOCATION.search(m["content"])
           for m in messages):
        return False, "an assistant turn holds an unparsed tool invocation as text"
    # The whole episode, not a truncation before the refused call: 124 of the 125 have the
    # rejection as the final turn, so truncating recovers almost nothing, and a truncation
    # rule is a repair whose boundary I would be choosing.
    if any(m["role"] == "tool" and REJECTED_CALL.search(m["content"]) for m in messages):
        return False, "a tool call was refused by the user; the refused call is supervised"
    if any(CHATML_LITERAL.search(m["content"]) for m in messages):
        return False, "text contains a literal ChatML marker (tokenizes as the real token)"
    if is_synthetic_user(messages[0]["content"]):
        return False, "opens on a machine event, not a person (task-notification etc.)"
    if not any(m["role"] == "assistant" for m in messages):
        return False, "no assistant turn to supervise"
    for i, m in enumerate(messages):
        if m["role"] == "tool" and (i == 0 or messages[i - 1]["role"] != "assistant"):
            return False, "tool turn not preceded by an assistant turn"
    return True, ""


def dedupe(rows):
    """(kept, n_dropped) -- one episode per (opener, final answer) pair, first wins.

    A WHOLE-PACK property, so it cannot live in usable(): each of these episodes is
    individually fine, and only the pack shows the repetition. 206 of 4329 rows (4.8%) are
    redundant, and they are not evenly spread -- a 5-minute patrol cron contributes 61
    episodes whose user turn is byte-identical and whose answer is "无新情况。戳已盖。",
    and three connectivity probes ("Reply with exactly: alpha") contribute 19 each. At that
    concentration the model is being taught to memorize boilerplate.

    Keyed on the opener plus the FINAL assistant turn, not the whole conversation: the
    patrol episodes differ in the middle (each reports a different training step) while
    being the same episode to learn from. Exact-prefix keys rather than a similarity
    threshold, because a threshold needs a cutoff nobody can defend and this shape does not
    need one -- these are byte-identical at both ends.

    NOT datagen/near_dedup_postpass.py (3b-8), which is the right tool for a different
    question: MinHash/LSH with Jaccard >= 0.5 over normalised code shards, for corpus rows
    that are near-copies of each other. These episodes are exact at both endpoints and
    differ in the middle, and the unit is a conversation rather than a document -- a
    shingled similarity measure would be a heavier tool giving the same answer here.
    """
    seen, kept = set(), []
    for r in rows:
        key = (r["messages"][0]["content"][:120], r["messages"][-1]["content"][:60])
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, len(rows) - len(kept)


def token_filter(rows, tok_path, max_tokens=4096):
    """(kept, dropped) by the REAL bound: the longest format_agentic pair must fit.

    The pair, not the episode: the packer emits one row per assistant turn, each carrying
    every prior turn as its masked prompt, so the LAST pair is the longest and it is the
    one that has to fit. Summing the episode would over-count (the prompt is shared) and
    measuring the first pair would under-count.

    Measured on the 5264-pair pilot with the pod's tokenizer (vocab 32773, fp
    0bce3584bc24f255, md5-verified against /work/aupai/data/tokenizer.json): 4810 rows fit
    (91.4%), 454 over (8.6%), longest single pair 6808 tokens. The 12000-char pre-filter
    that stood in for this while no tokenizer was on the machine was loose in the right
    direction -- it dropped nothing this would have kept -- but it was never the bound, and
    the report said so rather than implying it had run.

    ENCODES THE LAST PAIR ONLY, AND THAT IS THIS FUNCTION'S WHOLE COST PROFILE. Encoding every
    pair made this the dominant stage of the build and killed three v11 attempts on their own
    timeout with nothing written. The reason is quadratic: format_agentic emits one pair per
    assistant turn and each prompt carries every prior turn, so a 26-turn episode re-encodes its
    early turns 26 times. MEASURED on 40 real sessions: 2,168 rows, 56,113 pairs, 25.9 pairs per
    row, 4,507 MB of text to encode -- 2,079 KB per row whose stored form is a few KB. Scaled to
    all 2,588 transcript files that is ~291 GB at a measured 4.18 MB/s, about 1,163 minutes,
    against a 180-minute timeout. Encoding the last pair only is ~1/26 of that.

    The docstring above already asserted the last pair is the longest; it is now CHECKED per row
    rather than trusted, because that claim is what makes the cheap path equivalent. A
    format_agentic change that ever shortened the final pair would otherwise start passing
    over-length rows silently, and the pack's only real bound would be gone. The check compares
    STRING LENGTHS, which is free -- re-encoding every pair to verify would restore the cost this
    exists to avoid. Tested independently on 140 real rows before the change: 0 counterexamples.
    """
    import time  # noqa: PLC0415

    from loader import format_agentic  # noqa: PLC0415
    from tokenizers import Tokenizer  # noqa: PLC0415
    tok = Tokenizer.from_file(tok_path)
    kept, dropped, t0, n_enc = [], 0, time.time(), 0
    for r in rows:
        pairs = list(format_agentic(r["messages"]))
        if not pairs:
            continue
        sizes = [len(pr) + len(co) for pr, co in pairs]
        if sizes.index(max(sizes)) != len(sizes) - 1:
            raise RuntimeError(
                f"REFUSING: the longest format_agentic pair is #{sizes.index(max(sizes))} of "
                f"{len(sizes)}, not the last, in an episode from {r.get('project')}. This "
                "function encodes ONLY the last pair on the premise that cumulative prompts make "
                "it the longest -- if that no longer holds, the token bound is not being measured "
                "and over-length rows would enter the pack unnoticed. Encode all pairs again "
                "(and accept ~26x the cost) or fix the invariant before rebuilding.")
        pr, co = pairs[-1]
        longest = len(tok.encode(pr + co).ids)
        n_enc += len(pr) + len(co)
        if longest > max_tokens:
            dropped += 1
        else:
            kept.append(dict(r, tokens=longest))
    dt = time.time() - t0
    print(f"  token_filter: {len(rows)} rows in {dt:.1f}s, {n_enc / 1e6:.0f} MB encoded "
          f"({n_enc / 1e6 / max(dt, 1e-9):.2f} MB/s), {dropped} over {max_tokens} tokens",
          flush=True)
    return kept, dropped


def build(limit=10000, max_chars=400000, sessions=None, verbose=True):
    """Episodes -> (rows, report). Rows are {"messages": [...], "project": str}.

    max_chars IS NOT AN ADMISSION CRITERION. It is a memory ceiling and nothing else: an
    episode above it would be held in RAM only to be tokenized and dropped, and one
    subagent transcript alone reached 158,860 chars mean with a long tail. The single exact
    bound is 4096 TOKENS, applied by token_filter() against the real tokenizer.

    IT USED TO BE ONE (12000), AND THAT IS THE DEFECT THIS FIXES (fb's ruling, 2026-09-02,
    option C of three). 12000 chars was a stand-in for 4096 tokens written when
    data/tokenizer.json was absent from this Mac -- a proxy, documented as a proxy, and
    roughly right on the source it was calibrated against: it dropped 3,600 of 23,701
    parent-session episodes (15%). On subagent sidecars the same number dropped 1,270 of
    2,017 (63%), because a subagent dumps whole files into its context and its episodes are
    40x larger (158,860 chars mean against 3,899). A proxy that answers differently on two
    sources is not a criterion. So the proxy is retired to a memory ceiling at 400,000
    chars -- above the largest episode this corpus holds -- and the token filter decides
    admission for both sources by the same exact rule.

    Over 4096 tokens is still DROPPED, not truncated: truncating an agentic episode cuts a
    tool loop somewhere, and the cut point would be my choice rather than a measurement.

    WHAT THE PROXY ACTUALLY COST, measured after the ruling and before believing my own
    argument: 0 of 60 sampled over-12000-char subagent episodes would have fit 4096 tokens.
    The corpus runs 2.56 chars/token (mixed zh/en, and Chinese is under 2), so 12000 chars
    is ~4690 tokens -- the proxy sat ABOVE the real bound, not below it, and dropped only
    episodes the token filter drops anyway. So the 63% was NOT over-dropping and the row
    count barely moves. The ruling still stands on its own terms: one exact criterion
    instead of a proxy whose agreement with it was luck, and luck that would break the
    first time the language mix shifted. A proxy that happens to be conservative is still
    a second criterion nobody is checking.
    """
    files = sorted(glob.glob(SESSIONS)) if sessions is None else sessions
    rows, rep = [], {"sessions": len(files), "episodes": 0, "kept": 0,
                     "dropped": {}, "chars": 0, "by_project": {}, "secret_hits": 0,
                     "scanned": 0, "scan_ran": True}
    for path in files:
        # scrub HERE, not at the row: the report's by_project keyed on the raw directory
        # name and printed "-Users-bytedance-code-x" to the terminal for the top 8
        # projects, while every row's own project field was clean. A redaction that covers
        # the artifact but not the report about the artifact is not a redaction.
        project = scrub(os.path.basename(os.path.dirname(path)))
        try:
            episodes = turns_from_session(path)
        except OSError as e:
            rep["dropped"][f"unreadable: {type(e).__name__}"] = \
                rep["dropped"].get(f"unreadable: {type(e).__name__}", 0) + 1
            continue
        for msgs in episodes:
            rep["episodes"] += 1
            ok, why = usable(msgs)
            if not ok:
                rep["dropped"][why] = rep["dropped"].get(why, 0) + 1
                continue
            msgs = [{"role": m["role"], "content": scrub(m["content"])} for m in msgs]
            n = sum(len(m["content"]) for m in msgs)
            if n > max_chars:
                rep["dropped"]["over the memory ceiling (not an admission criterion)"] = \
                    rep["dropped"].get("over the memory ceiling (not an admission criterion)", 0) + 1
                continue
            rows.append({"messages": msgs, "project": project})
            rep["kept"] += 1
            rep["chars"] += n
            rep["by_project"][project] = rep["by_project"].get(project, 0) + 1
            if len(rows) >= limit:
                return rows, rep
        if verbose and rep["episodes"] and rep["episodes"] % 2000 == 0:
            print(f"  {rep['kept']} kept / {rep['episodes']} episodes", flush=True)
    return rows, rep


# Hit types that are a real credential SHAPE rather than a heuristic. A hit in this set
# discards the whole episode; the rest are reported for judgement. fb's ruling, 2026-09-02:
# discard, never mask -- a masked line leaves the surrounding context intact and that
# context still teaches the model "a credential belongs here", which is the thing we do not
# want it to learn. Secret Keyword and the two entropy detectors stay OUT of this set: they
# fire on example values, log lines and shas, and dropping every episode they touch would
# empty the pack while removing nothing real.
REAL_CREDENTIAL = frozenset({
    "AWS Access Key", "Azure Storage Account access key", "Basic Auth Credentials",
    "Cloudant Credentials", "Discord Bot Token", "GitHub Token", "GitLab Token",
    "IBM Cloud IAM Key", "IBM COS HMAC Credentials", "JSON Web Token", "Mailchimp API Key",
    "NPM tokens", "OpenAI API Key", "Private Key", "PyPI upload token", "SendGrid API Key",
    "Slack Token", "SoftLayer Credentials", "Square OAuth Secret", "Stripe Access Key",
    "Telegram Bot Token", "Twilio API Key",
})

#: An opaque credential no provider detector has a rule for, judged by its own entropy.
#:
#: THE CHARSET IS THE WHOLE POINT. detect_secrets' Base64HighEntropyString matches
#: `[A-Za-z0-9+/]+={0,2}`, so a urlsafe token containing `-`, `_` or `.` is never handed to
#: the detector as one string -- it is SPLIT at those characters and each piece is judged
#: alone. A Lark device code of 86 chars at 4.894 entropy came back clean from
#: find_secrets(): its longest base64-only fragment is 35 chars at 4.009, under the 4.5
#: limit, and every other fragment is far lower. The threshold was never consulted on the
#: credential; it was consulted on the debris. This pattern spans the urlsafe charset so the
#: entropy question is asked about the token that actually exists.
#:
#: 32 chars and 4.5 bits over that charset selects 461 tokens in 164 of 4508 episodes (3.6%),
#: and the set is genuinely mixed: 4 Lark device codes, a `plat_` API token, a session id and
#: a base64 `user:password` sit beside SWE-bench instance ids
#: (`PyCQA__flake8.cf1542ce.func_pm_ctrl_shuffle__7e1ipwsu`) and log filenames, which are
#: harmless.
#:
#: THE "ALL 164 GO" RULING THAT STOOD HERE IS SUPERSEDED, and by a third option rather than by
#: a better classifier. It weighed dropping 3.6% of episodes against building a
#: credential-vs-identifier classifier with no ground truth, and correctly chose dropping. What
#: neither side costed is REDACTING THE SPAN: the mixed set stops mattering, because a
#: SWE-bench instance id and a Lark device code are both replaced, and replacing an identifier
#: costs one span of one turn instead of a whole episode.
#:
#: Re-measured on the 10,000-episode pack 2026-09-04 (facts/data_quality.json
#: #dq.agentic_credential_split): 866 episodes (8.7%) carry a span, but only 1,768 of 359,663
#: TURNS do (0.49%), and 863 of the 866 carry NO provider-rule credential. So redaction keeps
#: 863 episodes at the cost of blanking half a percent of turns. The reasoning that survives
#: unchanged is the direction of the residual risk -- over-dropping costs episodes,
#: under-dropping is unrecoverable -- which is why a span too long to vouch for still drops the
#: episode (OPAQUE_MAX_REDACT) and why a provider hit still drops it.
OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9_\-.]{32,}")
OPAQUE_ENTROPY = 4.5


def opaque_credential(text):
    """True if `text` holds a high-entropy opaque token -- the shape find_secrets cannot see.

    Shannon entropy over the token's own characters, the same measure detect_secrets applies,
    but over the token as written rather than over base64-only fragments of it.
    """
    return bool(opaque_spans(text))


#: A span longer than this is content, not a credential, and redacting it deletes real text.
#:
#: MEASURED over the 10,000-episode pack 2026-09-04: 5,342 spans clear the entropy gate, and
#: their lengths are 3,131 at 32-63 chars, 2,025 at 64-127, 145 at 128-511, 35 at 512-2047,
#: and 6 at 2048+ -- the longest 10,943 chars at 5.997 bits over `-._` plus alphanumerics. No
#: provider issues a five-figure credential; at that length the regex has run together a blob
#: of adjacent tokens, a minified file, or an encoded payload. Redaction is per-span, so an
#: over-long span is the one case where redacting is WORSE than dropping the episode: it
#: silently removes content while leaving the episode in the pack looking intact.
#:
#: 512 rather than a rounder number because it sits above every real credential shape this
#: corpus holds (the Lark device code that motivated the detector is 86 chars, JWTs run a few
#: hundred) and below the 41 spans that are evidently not credentials. An episode holding a
#: span over the limit DROPS -- the conservative direction, and the same call OPAQUE_TOKEN's
#: own comment makes for the ambiguous 3.6%.
OPAQUE_MAX_REDACT = 512

#: What replaces a redacted span. Fixed width so it carries no information about what it
#: replaced, and named so a reader of the pack knows a tool did this rather than a person.
OPAQUE_PLACEHOLDER = "[REDACTED-CREDENTIAL]"


#: CONTEXT EXCLUSIONS, from 44's hand-read of v12 (runs/redaction_handread_v12.tsv, 50 sites:
#: 11 true secrets, 39 false positives, 0 incoherent turns). 32 of the 39 fall in five classes, and
#: every one is identifiable by what SURROUNDS the span rather than by the span itself -- which is
#: why they are excluded here and not by tightening the entropy threshold. A Lark device code and a
#: SWE-bench instance id have the same charset and similar entropy; only the context differs.
#:
#: Each pattern must match the span PLUS its left context, and is anchored to end at the span, so
#: `re.search` at the span's start position is not enough -- the check looks backwards. The pattern
#: is matched against the text ending at the span's END, and must consume the span itself.
#:
#: WHAT IS DELIBERATELY NOT HERE. The 7 remaining false positives (pip-package, arlechat-log-id,
#: base64-script-blob, cxx-symbol, wheel-filename, macos-tsm-uuid, npm-integrity-hash) are one site
#: each. A class defined from one example is a rule fitted to a sample of one, and the cost of
#: leaving them is one redacted span per site -- against the cost of a class that also swallows a
#: real credential appearing in the same shape. Redaction is cheap; a missed credential is not.
OPAQUE_EXCLUSIONS = (
    # A Claude Code session URL in a commit-message trailer. 11 of the 39, the largest single class.
    # The id is the session's own, carries nothing secret, and appears in text the model should
    # learn to write (commit messages).
    ("claude-session-url", re.compile(r"claude\.ai/code/[A-Za-z0-9_\-.]{32,}\Z")),
    # A SWE-bench / harness task instance id: owner__repo.sha.variant__suffix. The double
    # underscore is the discriminator -- it is a naming convention, not a token charset.
    ("swebench-task-id", re.compile(r"[A-Za-z0-9_\-.]*__[A-Za-z0-9_\-.]{32,}\Z")),
    # A HuggingFace repo id, owner/name, in search results or a model table. The owner segment plus
    # "/" is the context; the name alone can be high-entropy.
    ("hf-repo-id", re.compile(r"(?:^|[\s(\[\"'])[A-Za-z0-9][A-Za-z0-9_\-.]*/[A-Za-z0-9_\-.]{32,}\Z")),
    # An Anthropic thinking signature. Cryptographic, not a credential: it authenticates a thinking
    # block and grants no access. Identified by the JSON key it sits under.
    ("thinking-signature",
     re.compile(r"\"?signature\"?\s*[:=]\s*\"?[A-Za-z0-9_\-.+/=]*[A-Za-z0-9_\-.]{32,}\Z")),
    # A filesystem path segment. 12 of the 39: arxiv shard filenames with a uuid shape, model
    # directory names, task directories, a widget html path. A path is identifiable by the
    # separator immediately before the span, and a credential is not normally written after a "/".
    #
    # LAST ON PURPOSE. `[/\\]<span>` also matches "owner/<span>" and any URL tail, so placed first
    # it swallows hf-repo-id and claude-session-url and the report names the wrong class. The
    # selftest asserts each class is attributed to itself, and this ordering is what it caught.
    ("path", re.compile(r"[/\\][A-Za-z0-9_\-.]{32,}\Z")),
)


def _excluded(text, start, end):
    """The exclusion class covering the span at [start, end), or None.

    Matches against text[:end] anchored at the end, so each pattern sees the span AND what precedes
    it. Looking only at the span cannot separate these classes from a real credential: that is the
    finding, not an implementation detail.
    """
    left = text[:end]
    for name, pat in OPAQUE_EXCLUSIONS:
        m = pat.search(left)
        # The match must actually COVER the span, not merely end at it: an anchored pattern whose
        # own match starts after `start` would be describing different text than the span.
        if m and m.end() == end and m.start() <= start:
            return name
    return None


def opaque_spans(text):
    """[(start, end)] of high-entropy opaque tokens in `text`, in order, non-overlapping.

    The span list is what redaction needs and the boolean is derived from it, so the detector
    has ONE definition of what a hit is. Two copies of an entropy loop would be two things to
    keep in step -- the same reason _job_pids_for imports card_claim's predicate rather than
    reimplementing it.

    Spans in OPAQUE_EXCLUSIONS' five hand-read classes are dropped here rather than downstream, so
    the count of redactions and the count of over-limit drops both see the same span set.
    """
    out = []
    for m in OPAQUE_TOKEN.finditer(text):
        s = m.group(0)
        counts = collections.Counter(s)
        h = -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())
        if h >= OPAQUE_ENTROPY and not _excluded(text, m.start(), m.end()):
            out.append((m.start(), m.end()))
    return out


def redact_opaque(text):
    """(new_text, n_redacted, n_too_long). Replace each opaque span with the placeholder.

    n_too_long counts spans over OPAQUE_MAX_REDACT, which are NOT redacted -- the caller drops
    the episode instead. Returning the count rather than redacting them keeps the two decisions
    in one place: this function never silently deletes content it cannot vouch for.
    """
    spans = opaque_spans(text)
    if not spans:
        return text, 0, 0
    out, last, n, long = [], 0, 0, 0
    for a, b in spans:
        if b - a > OPAQUE_MAX_REDACT:
            long += 1
            continue
        out.append(text[last:a])
        out.append(OPAQUE_PLACEHOLDER)
        last = b
        n += 1
    out.append(text[last:])
    return "".join(out), n, long


def drop_credential_rows(rows):
    """(kept, dropped, redactions) -- see the ruling below for which class does which.

    TWO detectors, because one of them cannot see the other's cases. find_secrets covers the
    shapes with a provider rule; opaque_credential covers a high-entropy token with no rule,
    which find_secrets misses whenever the token spans the urlsafe charset (see OPAQUE_TOKEN
    -- a live Lark device code passed the scanner and reached a sample I was about to send to
    a peer).

    THE SPLIT (6e's ruling 2026-09-04, refined by what the pack measures):

      A REAL_CREDENTIAL hit DROPS the whole episode, unchanged from before. A provider rule
      firing means a live credential, and a masked line still teaches the model that a
      credential belongs at that position.

      An OPAQUE hit REDACTS the span and KEEPS the episode. What changed my reading of
      OPAQUE_TOKEN's "ALL 164 GO": that argument weighed dropping 3.6% of episodes against a
      classifier with no ground truth, and redaction is a third option neither side costed.
      Measured over the 10,000-episode pack: 866 episodes (8.7%) carry an opaque span, but
      only 1,768 of 359,663 TURNS do (0.49%). Redaction buys back 866 episodes at the cost of
      blanking half a percent of turns, and it needs no credential-vs-identifier judgment --
      a SWE-bench instance id and a device code are both replaced, and replacing an identifier
      costs one span of a turn rather than a whole episode.

      A span over OPAQUE_MAX_REDACT drops the episode: see that constant. Redacting content is
      worse than dropping it, because the episode stays in the pack looking intact.

    Per-turn scan, because the location is what gets reported to a human: fb needs the file
    path and line of any true credential so the user can be told it is sitting in a
    transcript, and "somewhere in the pack" is not a report. The content is never printed or
    written anywhere, redacted or not.
    """
    kept, dropped, redactions = [], [], []
    for r in rows:
        real, n_red, n_long = set(), 0, 0
        for m in r["messages"]:
            real.update(t for t in (find_secrets(m["content"]) or []) if t in REAL_CREDENTIAL)
        # REAL FIRST, and no redaction on a row that is about to drop: redacting a dropped
        # episode's turns is work whose result is discarded, and it would make the counts read
        # as if a kept row had been cleaned.
        if real:
            dropped.append({"project": r["project"], "types": sorted(real),
                            "turns": len(r["messages"]), "why": "REAL_CREDENTIAL"})
            continue
        for m in r["messages"]:
            new, n, long = redact_opaque(m["content"])
            n_red += n
            n_long += long
            if n:
                m["content"] = new
        if n_long:
            dropped.append({"project": r["project"], "types": ["Opaque Span Over Limit"],
                            "turns": len(r["messages"]),
                            "why": f"{n_long} span(s) over {OPAQUE_MAX_REDACT} chars"})
            continue
        if n_red:
            redactions.append({"project": r["project"], "spans": n_red,
                               "turns": len(r["messages"])})
        kept.append(r)
    return kept, dropped, redactions


def scan_rows(rows, sample=None):
    """(hits, scanned, ran). Secrets surviving redaction, over one joined document.

    One document rather than per-row: scan_file's cost is dominated by process setup, and
    the question asked is "does the pack contain a secret", which is a property of the
    whole pack. A hit reports its TYPE so it can be judged; the row is then findable by
    searching the pack for that pattern.
    """
    text = "\n".join(m["content"] for r in rows for m in r["messages"])
    if sample:
        text = text[:sample]
    hits = find_secrets(text)
    return (hits or []), len(text), hits is not None


def real_credential_rows(rows):
    """[(index, [types])] for rows carrying a REAL_CREDENTIAL type after redaction.

    scan_rows answers "does this pack contain one" and CANNOT answer "which row", because
    it joins every row into a single document and returns bare type names. That gap is why
    v13's `secret scan: 2 type(s)` read as two findings: it is a count of distinct TYPES
    over 4,799 rows, and a per-row scan of the same pack found 55 rows hit. A gate that
    refuses without naming the row leaves nobody able to act on the refusal.

    Only REAL_CREDENTIAL types are returned. Base64 High Entropy String and Secret Keyword
    are deliberately excluded here for the reason REAL_CREDENTIAL's own definition gives:
    they fire on example values, log lines and shas, so gating on them would refuse every
    pack while removing nothing real. Those are counted and printed instead, which is the
    allowlist -- an allowed type is one the gate names and passes, not one it cannot see.
    """
    out = []
    for i, r in enumerate(rows):
        h = find_secrets("\n".join(m["content"] for m in r["messages"]))
        if h and set(h) & REAL_CREDENTIAL:
            out.append((i, sorted(set(h) & REAL_CREDENTIAL)))
    return out


def _selftest():
    """Broken worlds. Every assertion is a case with a known right answer.

    Runs without ~/.claude and without a tokenizer: the parser is the subject, and a world
    that fails before the stage under test proves nothing (fb's ruling, 2026-09-02).
    """
    import tempfile
    fails = []

    # 1. The redaction actually redacts, and does not touch what it should not.
    got = scrub("see /Users/alice/code/x.py and mail bob.smith+tag@example.co.uk")
    if "alice" in got or "bob.smith" in got:
        fails.append(f"scrub left an identifier: {got!r}")
    if "<HOME>/code/x.py" not in got or "<EMAIL>" not in got:
        fails.append(f"scrub did not substitute: {got!r}")
    if scrub("12/60 = 0.2 per minute") != "12/60 = 0.2 per minute":
        fails.append("scrub altered text containing no identifier")
    # The DIRECTORY-ENCODED form, which is how Claude Code names a project: every "/"
    # becomes "-". Three cases, each a leak a narrower draft shipped, all found by scanning
    # a built pack rather than by reading the pattern:
    #   with a path after it   the common case, 20 of 50 sampled rows
    #   at end of string       an `ls -la` listing; the "-<letter>" lookahead missed it
    #   wrapped in hyphens     "--Users-x--"; the trailing "-" defeated a char-class fix
    user = os.path.basename(HOME)
    for probe in (f"-Users-{user}-code-aupai", f"160 -Users-{user}", f"--Users-{user}--",
                  f"~/.claude/projects/-Users-{user}-code-x/y.jsonl"):
        if user in scrub(probe):
            fails.append(f"scrub leaves the username in the encoded form: {scrub(probe)!r}")
    # INTERNAL URLS (fb's ruling, 2026-09-02). Both directions, because a pattern that eats
    # public URLs is worse than one that leaks internal ones: it silently rewrites the
    # corpus's factual content.
    for probe, want in (
        ('--doc "https://bytedance.larkoffice.com/docx/DvAZabc"', "<INTERNAL_DOC>"),
        ("see https://open.feishu.cn/open-apis/auth/v3/tenant", "<INTERNAL_DOC>"),
        ("remote: https://bits.bytedance.net/code/infcs/eic/merge_requests/4", "<INTERNAL_DOC>"),
        # No scheme: shell output prints the bare host, and a scheme-anchored pattern
        # would leave every one of these.
        ("bits.bytedance.net/code/infcs/eic", "<INTERNAL_DOC>"),
        ('{"detailsUrl":"https://github.com/bytedance-iaas/sglang/actions/runs/1"}',
         "<INTERNAL_REPO>"),
    ):
        got = scrub(probe)
        if want not in got:
            fails.append(f"internal URL not redacted: {probe!r} -> {got!r}")
        for leak in ("larkoffice", "bytedance.net", "feishu.cn", "bytedance-iaas"):
            if leak in got:
                fails.append(f"internal host survives redaction: {got!r}")
        if "https://<" in got or "http://<" in got:
            fails.append(f"a dangling scheme was left in front of the placeholder: {got!r}")
    # MUST SURVIVE. The username on this machine equals the ORG name, so a substring check
    # for it fires on public domains and git branch names -- 474 "leaks" in one pack, 0 of
    # them real. These four are the shapes that false positive was made of.
    for probe in ("https://github.com/sgl-project/sglang/pull/187",
                  "https://docs.python.org/3/library/re.html",
                  "origin/bytedance/deepseek_v4",
                  "aupai-80 [535ae3] - interactive"):
        if scrub(probe) != probe:
            fails.append(f"scrub rewrote a public or non-topology string: "
                         f"{probe!r} -> {scrub(probe)!r}")

    # 2. The scanner separates real secrets from prose. If it cannot, "0 hits" is
    #    meaningless -- scan_line reported 4 hits on "hello world, nothing here".
    clean = find_secrets("hello world\n这是普通中文\ndef add(a, b): return a + b\n"
                         "  [PASS] 347 facts (1.7s)\ncommit 3f5231c9a8b7d6e5\n"
                         "/Users/x/code/aupai/scripts/build.py\n")
    dirty = find_secrets('aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n'
                         'token: ghp_16C7e42F292c6912E7710c838347Ae178B4a\n'
                         'password = hunter2supersecret\n')
    if clean is None or dirty is None:
        fails.append("detect_secrets is not installed -- the scan half is UNTESTED, and a "
                     "scanner that did not run is not a clean result (pip install detect-secrets)")
    else:
        if clean:
            fails.append(f"the scanner flags ordinary prose/logs/paths: {clean} -- '0 hits' "
                         "cannot mean anything if this fires")
        if len(dirty) < 2:
            fails.append(f"the scanner missed real secrets, found only {dirty}")
        # THE CHUNK SIZE IS LOAD-BEARING. Three secrets buried in prose: chunk=1 finds all
        # four types, chunk=5 finds one. If a future edit raises the default for speed,
        # this goes red instead of the pack silently getting a weaker scan.
        buried = (["ordinary transcript line about refactoring"] * 20
                  + ["aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"]
                  + ["ordinary transcript line about refactoring"] * 20
                  + ["password = hunter2supersecret"])
        text = "\n".join(buried) + "\n"
        if len(find_secrets(text)) < 2:
            fails.append(f"secrets buried in prose are missed at the default chunk size: "
                         f"{find_secrets(text)} -- a bigger chunk trades findings for speed")

    # 3. The parser, on a transcript shaped like the real thing.
    rec = [
        {"type": "user", "message": {"role": "user", "content": "算 12/60"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "should not appear anywhere"},
            {"type": "text", "text": "12/60 = "},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "echo 0.2"}}]}},
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": [{"type": "text", "text": "0.2"}]}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "0.2 per minute"}]}},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            for r in rec:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        eps = turns_from_session(p)
        if len(eps) != 1:
            fails.append(f"{len(eps)} episodes from one user turn, want 1")
        else:
            msgs = eps[0]
            roles = [m["role"] for m in msgs]
            if roles != ["user", "assistant", "tool", "assistant"]:
                fails.append(f"turn roles {roles}, want user/assistant/tool/assistant")
            if any("should not appear" in m["content"] for m in msgs):
                fails.append("a thinking block reached the turn sequence")
            if not any('"tool": "Bash"' in m["content"] for m in msgs):
                fails.append(f"the tool call did not render as text: {msgs!r}")
            if msgs[2]["content"] != "0.2":
                fails.append(f"tool output is {msgs[2]['content']!r}, want '0.2'")
            ok, why = usable(msgs)
            if not ok:
                fails.append(f"a well-formed episode was rejected: {why}")
            # THE contract: format_agentic must accept it and supervise each assistant
            # turn once, with tool output never in a completion.
            try:
                from loader import IM_END, format_agentic
                pairs = format_agentic(msgs)
                if len(pairs) != 2:
                    fails.append(f"{len(pairs)} pairs for 2 assistant turns")
                for prompt, completion in pairs:
                    if completion.replace(IM_END, "").strip() == "0.2" and "tool" in prompt:
                        pass  # the continuation legitimately repeats the result value
                    if "<|im_start|>tool" in completion:
                        fails.append(f"tool output is supervised: {completion!r}")
                    if not completion.endswith(IM_END):
                        fails.append(f"completion does not end at the stop token: {completion!r}")
            except ImportError as e:
                fails.append(f"cannot import loader.format_agentic: {e}")

    # 4. A real credential removes its episode; a heuristic hit does not.
    if find_secrets("x") is not None:
        real = [{"project": "p", "messages": [
            {"role": "user", "content": "deploy it"},
            {"role": "assistant", "content": "token: ghp_16C7e42F292c6912E7710c838347Ae178B4a"}]}]
        # A HEURISTIC-ONLY hit, verified to actually trip one: `password = <word>` returns
        # Secret Keyword and nothing from REAL_CREDENTIAL. My first draft used a git sha,
        # which trips NOTHING -- so the assertion passed under a blinding that moved Secret
        # Keyword into REAL_CREDENTIAL, i.e. it never exercised the boundary it claims to
        # guard. A negative case has to be positive for the detector and negative for the
        # decision.
        soft = [{"project": "p", "messages": [
            {"role": "user", "content": "set the password"},
            {"role": "assistant", "content": "password = hunter2supersecret"}]}]
        if not find_secrets(soft[0]["messages"][1]["content"]):
            fails.append("the heuristic fixture trips no detector at all, so the "
                         "keep-heuristic-hits assertion below proves nothing")
        kept, dropped, _red = drop_credential_rows(real)
        if len(dropped) != 1 or kept:
            fails.append(f"an episode carrying a GitHub token was kept: {len(kept)} kept")
        if dropped and dropped[0].get("why") != "REAL_CREDENTIAL":
            fails.append(f"a provider hit was dropped for the wrong reason: {dropped[0].get('why')!r} "
                         "-- the report tells a human which class fired, so it must be right")
        kept, dropped, _red = drop_credential_rows(soft)
        if len(kept) != 1 or dropped:
            fails.append("an episode with only a heuristic hit was discarded -- dropping every "
                         "heuristic hit empties the pack while removing nothing real")

    # 4b. THE OPAQUE TOKEN, which is the case find_secrets cannot answer. The fixture is
    #     SYNTHETIC and its structure is the assertion: 4.5+ entropy over the urlsafe charset
    #     while every base64-only fragment stays under the limit, which is what made a live
    #     Lark device code (86 chars, 4.894) scan clean. A base64-only blob would be caught by
    #     Base64HighEntropyString and would prove nothing about this gate.
    #
    #     Separators every 6-7 chars, deliberately: a fragment of n chars cannot exceed
    #     log2(n) bits, so short fragments put the base64 detector structurally out of reach
    #     (2.59 max here) while the whole token sits at 5.53. My first fixture spaced them
    #     wider and a 4.52-bit fragment slipped over the limit -- the assertion below caught
    #     it, which is the reason it exists.
    fake = "qZ8mK3-xVwT9p.L2nbY7_cJ4hR6_dF5sNg_A1eU0i_OtXwQ3_vZ7yBm_C9kHjD_6fG"
    frags = re.findall(r"[A-Za-z0-9+/]{3,}={0,2}", fake)

    def _h(s):
        c = collections.Counter(s)
        return -sum(v / len(s) * math.log2(v / len(s)) for v in c.values())

    if _h(fake) < OPAQUE_ENTROPY:
        fails.append(f"the opaque fixture is only {_h(fake):.2f} entropy, under the "
                     f"{OPAQUE_ENTROPY} gate -- it cannot test the gate it is for")
    if frags and max(_h(f) for f in frags) >= OPAQUE_ENTROPY:
        fails.append(f"the opaque fixture's base64 fragments reach {max(_h(f) for f in frags):.2f} "
                     "entropy, so detect_secrets would catch it and this case does not "
                     "exercise the charset-splitting blind spot at all")
    if find_secrets(fake) is not None and any(
            t in REAL_CREDENTIAL for t in find_secrets(fake)):
        fails.append("the opaque fixture trips a provider detector, so drop_credential_rows "
                     "would remove it with or without opaque_credential")
    if not opaque_credential(fake):
        fails.append(f"opaque_credential missed {len(fake)}-char {_h(fake):.2f}-entropy token")
    opaque = [{"project": "p", "messages": [
        {"role": "user", "content": "log in"},
        {"role": "assistant", "content": f"device_code: {fake}"}]}]
    kept, dropped, red = drop_credential_rows(opaque)
    # THE ASSERTION IS INVERTED from the version before 2026-09-04, deliberately: an opaque hit
    # is now redacted and the episode is KEPT (6e's ruling). What must still hold is that the
    # token is GONE from what is kept -- "kept" alone would pass on an implementation that
    # detected nothing at all.
    if len(kept) != 1 or dropped:
        fails.append(f"an opaque hit was dropped rather than redacted: {len(kept)} kept, "
                     f"{len(dropped)} dropped")
    if kept and fake in kept[0]["messages"][1]["content"]:
        fails.append("the opaque token SURVIVED in a kept episode -- redaction did nothing and "
                     "the row now ships the credential it used to drop")
    if kept and OPAQUE_PLACEHOLDER not in kept[0]["messages"][1]["content"]:
        fails.append("the kept episode carries no placeholder, so nothing was replaced")
    if kept and "device_code: " not in kept[0]["messages"][1]["content"]:
        fails.append("redaction ate the surrounding text, not just the span")
    if len(red) != 1 or red[0]["spans"] != 1:
        fails.append(f"the redaction was not reported: {red}")
    # A SPAN TOO LONG TO VOUCH FOR DROPS. 41 spans in the real pack are over the limit, the
    # longest 10,943 chars -- no provider issues a five-figure credential, so the regex has run
    # together a blob. Redacting it would delete content while leaving the episode looking
    # intact, which is worse than dropping it. The fixture repeats the fixture token, so it
    # clears the entropy gate by construction rather than by luck.
    # A FIXED 1200 CHARS, not a multiple of OPAQUE_MAX_REDACT. Deriving the fixture from the
    # constant makes it scale WITH the constant, so the case cannot fail no matter what the
    # limit becomes: measured 2026-09-04 against a world with OPAQUE_MAX_REDACT = 10**9 -- i.e.
    # nothing is ever too long to redact, the exact defect this case exists to catch -- and
    # every assertion here passed. A fixture that moves with the value under test tests nothing.
    # 1200 is between the 512 limit and the 2048 bucket the real pack's over-long spans sit in.
    long_tok = (fake + "_") * (1200 // len(fake) + 1)
    if OPAQUE_MAX_REDACT >= len(long_tok):
        fails.append(f"OPAQUE_MAX_REDACT is {OPAQUE_MAX_REDACT}, at or above the {len(long_tok)}-char "
                     "fixture -- the over-limit case cannot fire, so raise the fixture rather "
                     "than letting this case pass vacuously")
    if not opaque_credential(long_tok):
        fails.append(f"the over-long fixture ({_h(long_tok):.2f} entropy) does not even trip the "
                     "detector, so it cannot test what happens to a span that does")
    toolong = [{"project": "p", "messages": [
        {"role": "user", "content": "dump it"},
        {"role": "assistant", "content": f"blob: {long_tok}"}]}]
    kept_l, dropped_l, red_l = drop_credential_rows(toolong)
    if len(dropped_l) != 1 or kept_l or red_l:
        fails.append(f"an over-limit span was not dropped: {len(kept_l)} kept, "
                     f"{len(dropped_l)} dropped, {len(red_l)} redacted")
    if dropped_l and str(OPAQUE_MAX_REDACT) not in str(dropped_l[0].get("why")):
        fails.append(f"the drop reason does not name the limit that caused it: "
                     f"{dropped_l[0].get('why')!r}")
    # And the gate must not eat ordinary text: a long identifier at low entropy stays.
    for benign in ("scripts/test_eval_base_prompt_format.py",
                   "p500m_20b_0902_step_0000012000_loss_2p31",
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"):
        if opaque_credential(benign):
            fails.append(f"opaque_credential fires on benign text ({_h(benign):.2f}): {benign!r}")
        if redact_opaque(benign)[0] != benign:
            fails.append(f"redact_opaque changed benign text: {benign!r}")
    # TWO SPANS IN ONE TURN, because a single-span fixture passes on a loop that returns after
    # the first replacement -- and the real pack has episodes with up to 45 hit turns.
    two = [{"project": "p", "messages": [
        {"role": "user", "content": "both"},
        {"role": "assistant", "content": f"a={fake} and b={fake[::-1]}"}]}]
    kept2, _d2, red2 = drop_credential_rows(two)
    if not kept2 or kept2[0]["messages"][1]["content"].count(OPAQUE_PLACEHOLDER) != 2:
        fails.append("two spans in one turn were not both redacted: "
                     f"{kept2 and kept2[0]['messages'][1]['content'][:60]!r}")
    if red2 and red2[0]["spans"] != 2:
        fails.append(f"the span count is wrong for a two-span turn: {red2}")

    # 4c. THE FIVE HAND-READ EXCLUSION CLASSES, from 44's read of v12
    #     (runs/redaction_handread_v12.tsv: 50 sites, 11 true secrets, 39 false positives).
    #
    #     THE FIXTURES ARE SYNTHETIC AND THAT IS DELIBERATE, twice over. The TSV records the site,
    #     project, class and a note -- it does NOT carry the matched text, so a literally TSV-driven
    #     case is not possible from it. And it must not be: the 11 true-secret sites are real
    #     credentials in the user's transcripts, and copying one into a test fixture would put a live
    #     credential in git. So each class gets a token of the same SHAPE in the same CONTEXT, and
    #     the TSV's role is to say which classes exist and how many sites each covers.
    #
    #     The pairing is what makes each case discriminating: the same token, once inside the
    #     excluded context and once bare. If the exclusion matched on the token instead of the
    #     context, the bare half would stop being redacted -- which is the failure that would ship a
    #     credential.
    for name, framed, bare in (
        ("claude-session-url", f"See https://claude.ai/code/{fake} for the session", fake),
        ("path", f"reading /data/shards/{fake}.jsonl now", fake),
        ("swebench-task-id", f"instance google__textfsm.{fake}", fake),
        ("hf-repo-id", f"found chimingw/{fake} on the hub", fake),
        ("thinking-signature", f'"signature": "{fake}"', fake),
    ):
        if opaque_spans(framed):
            fails.append(f"the {name} exclusion did not fire: {framed[:70]!r} still reports a span, "
                         f"so v13 would redact the {name} sites 44 hand-read as false positives")
        if not opaque_spans(f"device_code={bare}"):
            fails.append(f"the {name} exclusion also suppressed the BARE token -- it is matching on "
                         f"the token, not the context, so a real credential of this shape now "
                         f"ships unredacted")
        if _excluded(framed, framed.index(fake), framed.index(fake) + len(fake)) != name:
            fails.append(f"_excluded attributes {framed[:50]!r} to "
                         f"{_excluded(framed, framed.index(fake), framed.index(fake) + len(fake))!r}"
                         f", not {name!r} -- the classes overlap and the report would name the "
                         f"wrong one")
    # A CREDENTIAL IN A CONTEXT THAT LOOKS LIKE AN EXCLUSION MUST STILL REDACT. The Lark device code
    # 44 found at site 17 was an --device-code ARGUMENT, i.e. it followed a space, not a "/". This
    # asserts the path exclusion is anchored on the separator rather than on "appears in a command".
    for ctx, why_ in ((f"lark-cli auth login --device-code {fake}", "a flag argument"),
                      (f"DEVICE_CODE={fake}", "an env assignment"),
                      (f"token: {fake}", "a json-ish field")):
        if not opaque_spans(ctx):
            fails.append(f"an opaque token in {why_} was excluded: {ctx[:60]!r} -- that is the "
                         f"shape of the real credentials in the hand-read, not of the false "
                         f"positives")

    # 5. An episode quoting a ChatML marker must be refused: the tokenizer turns the quote
    #    into the real special token, so it would inject a role boundary into a completion.
    quoting = [{"role": "user", "content": "what does <|im_start|>tool look like?"},
               {"role": "assistant", "content": "it is a role turn"}]
    ok, why = usable(quoting)
    if ok:
        fails.append("an episode quoting <|im_start|> was accepted; the tokenizer maps the "
                     "quote to the real special token and it becomes a fake role boundary")

    # 6. Unactionable openers and pure non-answers.
    for msgs, what in (
        ([{"role": "user", "content": "[Image: source: <HOME>/x/1.png]"},
          {"role": "assistant", "content": "ok"}], "an image-only opener"),
        ([{"role": "user", "content": "/compact"},
          {"role": "assistant", "content": "ok"}], "a bare slash command"),
        ([{"role": "user", "content": "do the thing"},
          {"role": "assistant", "content": "No response requested."}], "a pure non-answer"),
        # The harness speaking as the assistant. The negative case below matters as much:
        # a real answer that MENTIONS an error must survive, or the filter eats every
        # debugging episode in a corpus that is mostly debugging.
        ([{"role": "user", "content": "跑"},
          {"role": "assistant", "content": "call"},
          {"role": "tool", "content": "ok"},
          {"role": "assistant", "content": "Request timed out"}], "a harness timeout as an answer"),
        ([{"role": "user", "content": "跑"},
          {"role": "assistant", "content": "API Error: 500"}], "a bare API error as an answer"),
        # An unparsed tool invocation left in the text.
        ([{"role": "user", "content": "读它"},
          {"role": "assistant",
           "content": "先读再改。\n<invoke name=\"Read\">\n<parameter name=\"file_path\">x</parameter>"}],
         "a broken tool invocation as assistant text"),
        # A refused call. The rejection sits in the TOOL turn (masked, so it looks
        # harmless); the defect is the assistant turn before it, which is supervised.
        ([{"role": "user", "content": "查一下"},
          {"role": "assistant", "content": '{"tool": "Bash", "input": {"command": "rm -rf /"}}'},
          {"role": "tool",
           "content": "The user doesn't want to proceed with this tool use. The tool use "
                      "was rejected (eg. if it was a file edit, the new_string was NOT "
                      "written to the file)"}],
         "an episode whose supervised call the user refused"),
    ):
        ok, why = usable(msgs)
        if ok:
            fails.append(f"{what} was accepted; there is nothing here to learn")

    for msgs, what in (
        ([{"role": "user", "content": "why did it fail"},
          {"role": "assistant", "content": "The API error at line 12 is a 429; retry with backoff."}],
         "an answer ABOUT an API error"),
        ([{"role": "user", "content": "x"},
          {"role": "assistant", "content": "跑挂了：Request timed out 是 pod 那侧的，不是代码。"}],
         "an answer quoting a timeout mid-sentence"),
        # Prose ABOUT tool-call markup. 3 of the 10 real turns matching this markup are
        # this, in a repo whose transcripts discuss tool syntax constantly -- a substring
        # match would eat every one of them.
        ([{"role": "user", "content": "它输出了什么"},
          {"role": "assistant",
           "content": "关键是：里面有没有 tool-call 语法（JSON tool_call / XML <function_calls>），"
                      "还是纯散文？引用实际字节。"}],
         "prose discussing <function_calls> syntax"),
    ):
        ok, why = usable(msgs)
        if not ok:
            fails.append(f"{what} was dropped ({why}); HARNESS_ERROR must anchor at the "
                         "start of the turn, not match anywhere in it")

    # 7. A tool result with no preceding assistant turn must be refused, not repaired.
    bad = [{"role": "tool", "content": "orphan"}, {"role": "assistant", "content": "x"}]
    ok, why = usable(bad)
    if ok:
        fails.append("an orphan tool turn was accepted; format_agentic would raise on it")

    # 8. Dedup is a PACK property: each of these episodes is individually usable, so
    #    usable() cannot see the defect and only a whole-pack pass can.
    patrol = [{"project": "p", "messages": [
        {"role": "user", "content": "定时巡检"},
        {"role": "assistant", "content": f"step {i}"},
        {"role": "assistant", "content": "无新情况。戳已盖。"}]} for i in range(5)]
    kept, n = dedupe(patrol)
    if len(kept) != 1 or n != 4:
        fails.append(f"dedupe kept {len(kept)} of 5 episodes that differ only in the middle")
    # And it must not collapse genuinely different work: same opener, different answer.
    distinct = [{"project": "p", "messages": [
        {"role": "user", "content": "log"},
        {"role": "assistant", "content": f"answer {i}"}]} for i in range(3)]
    kept, n = dedupe(distinct)
    if len(kept) != 3 or n:
        fails.append(f"dedupe collapsed {n} episodes that share an opener but answer "
                     "differently -- 'log' is the most common opener in this corpus")

    # 9. max_chars is a MEMORY CEILING, not an admission criterion (fb's ruling, option C).
    #    Asserted by SIGNATURE DEFAULT, because the way this regresses is someone tightening
    #    the number back toward a token estimate -- which is what 12000 was, and what made
    #    the same knob answer 15% on one source and 63% on another.
    import inspect
    ceiling = inspect.signature(build).parameters["max_chars"].default
    if ceiling < 200000:
        fails.append(f"build()'s max_chars default is {ceiling}, under the 200k floor fb set. "
                     "Below that it stops being a memory ceiling and starts deciding "
                     "admission, which only --max-tokens may do")
    # The ceiling must sit clear of the real bound at this corpus's measured density
    # (2.56 chars/token, so 4096 tokens is ~10.5k chars). A ceiling anywhere near that is
    # a token filter wearing a different name.
    if ceiling < 4 * 4096 * 2.56:
        fails.append(f"max_chars {ceiling} is within 4x of the 4096-token bound at 2.56 "
                     "chars/token; it would shadow the exact criterion")

    # THE PACK SURVIVES A KILL DURING THE SECRET SCAN. Both v11 builds died inside that scan
    # (~2 ms/line by find_secrets' own measurement, so ~100 min at --limit 100000) and lost every
    # row, because the only write came after it.
    #
    # TESTED BY RUNNING main(), NOT BY READING ITS SOURCE. The first version of this check searched
    # main()'s text for ".unscanned"/"scan_rows"/"os.replace" and asserted their order -- and it
    # stayed GREEN when `staged` was set to None, because the strings were all still there while
    # the write was dead. A guard that reads the spelling passes any edit that keeps the words.
    # So: run the real main() against a fixture session, with scan_rows monkey-patched to raise the
    # way a kill would, and require the staged file to exist afterwards with every row in it.
    import tempfile as _tf
    _d = _tf.mkdtemp()
    _sess = os.path.join(_d, "s.jsonl")
    with open(_sess, "w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({"type": "user", "message": {"role": "user",
                     "content": f"question {i} " + "x" * 200}}) + "\n")
            fh.write(json.dumps({"type": "assistant", "message": {"role": "assistant",
                     "content": [{"type": "text", "text": f"answer {i} " + "y" * 200}]}}) + "\n")
    _out = os.path.join(_d, "pack.jsonl")
    _mod = sys.modules[__name__]
    _real_scan, _real_argv, _real_sessions = _mod.scan_rows, sys.argv, _mod.SESSIONS
    # SESSIONS is the glob build() reads when no explicit list is passed, and there is no --sessions
    # flag; pointing it at the fixture is how main() runs on 3 pairs instead of 3.8 GB.
    _mod.SESSIONS = _sess
    _mod.scan_rows = lambda rows, sample=None: (_ for _ in ()).throw(KeyboardInterrupt("killed"))
    try:
        sys.argv = ["build_agentic_sft.py", "--out", _out]
        try:
            main()
            fails.append("main() completed although scan_rows raised; the kill was not simulated")
        except KeyboardInterrupt:
            pass
        except SystemExit as e:
            fails.append(f"main() exited ({e}) before the write; the fixture cannot reach it")
        if os.path.exists(_out):
            fails.append(f"{_out} exists after a kill INSIDE the secret scan -- rows reached the "
                         f"real --out path without being cleared")
        elif not os.path.exists(_out + ".unscanned"):
            fails.append("a kill during the secret scan left NO file: the pack is written after "
                         "the scan again, which is what lost both v11 builds")
        else:
            _n = sum(1 for line in open(_out + ".unscanned", encoding="utf-8") if line.strip())
            if _n < 1:
                fails.append(f"the staged file holds {_n} rows; the write is not flushed before "
                             f"the scan, so a kill still loses the pack")
    finally:
        _mod.scan_rows, sys.argv, _mod.SESSIONS = _real_scan, _real_argv, _real_sessions

    # THE SCAN GATES THE RENAME, both directions, by running main() rather than reading it.
    # Same fixture machinery as the kill case above and for the same reason: the first version of
    # THAT check asserted on main()'s source text and stayed green when the write was dead.
    #
    # The defect: `os.replace(staged, a.out)` used to run unconditionally after scan_rows, so a
    # pack carrying a credential landed under its real name with the hit merely printed. A second
    # unconditional write at the end of main() wrote it again outside the staged path -- v13's log
    # prints `wrote 4799 rows -> ...` twice, which is that second write.
    #
    # real_credential_rows is monkey-patched, NOT the fixture text. A planted `ghp_...` in a
    # fixture would be a real-credential SHAPE copied into this file, and this repository's rule
    # is that a credential span never enters a test fixture. Patching the classifier exercises the
    # branch under test -- the gate's response to a positive verdict -- without one existing.
    if find_secrets("x") is not None:
        for _label, _verdict, _want_out in (
            ("a real-credential row", [(7, ["GitHub Token"])], False),
            ("no real-credential row", [], True),
        ):
            _d2 = _tf.mkdtemp()
            _sess2 = os.path.join(_d2, "s.jsonl")
            with open(_sess2, "w", encoding="utf-8") as fh:
                for i in range(3):
                    fh.write(json.dumps({"type": "user", "message": {"role": "user",
                             "content": f"question {i} " + "x" * 200}}) + "\n")
                    fh.write(json.dumps({"type": "assistant", "message": {"role": "assistant",
                             "content": [{"type": "text",
                                          "text": f"answer {i} " + "y" * 200}]}}) + "\n")
            _out2 = os.path.join(_d2, "pack.jsonl")
            _real_rcr = _mod.real_credential_rows
            _mod.real_credential_rows = lambda rows, _v=_verdict: _v
            # SESSIONS, like the kill case: without it main() globs the real ~/.claude tree.
            _mod.SESSIONS = _sess2
            try:
                sys.argv = ["build_agentic_sft.py", "--out", _out2]
                _rc = main()
            except SystemExit as e:
                _rc = e.code
            finally:
                _mod.real_credential_rows, sys.argv = _real_rcr, _real_argv
                _mod.SESSIONS = _real_sessions
            _landed = os.path.exists(_out2)
            _staged_left = os.path.exists(_out2 + ".unscanned")
            if _want_out:
                if _rc != 0:
                    fails.append(f"main() returned {_rc} on {_label}; a clean pack must land")
                if not _landed:
                    fails.append(f"{_label}: the pack did not land at --out")
                if _staged_left:
                    fails.append(f"{_label}: the .unscanned file survived a clean write, so the "
                                 "rename is now a copy and two files claim to be the pack")
            else:
                if _rc == 0:
                    fails.append(f"main() returned 0 on {_label} -- the scan does not gate the "
                                 "write, which is the v13 defect")
                if _landed:
                    fails.append(f"{_label}: the pack LANDED at --out anyway; a credential-carrying "
                                 "pack must never take the real name")
                if not _staged_left:
                    fails.append(f"{_label}: no .unscanned file remains, so the refusal also "
                                 "destroyed the rows instead of holding them for inspection")

    for f in fails:
        print(f"  FAIL {f}")
    print(f"\n{len(fails)} failure(s)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=10000, help="stop after N pairs")
    ap.add_argument("--max-chars", type=int, default=400000,
                    help="MEMORY CEILING, not an admission criterion: above the largest "
                         "episode this corpus holds. Admission is decided only by "
                         "--max-tokens against the real tokenizer")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer.json"),
                    help="apply the real 4096-token filter with this tokenizer")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--subagents", action="store_true",
                    help="also read */subagents/*.jsonl as INDEPENDENT episodes (source=subagent); "
                         "never interleaved into the parent -- a subagent conversation has its "
                         "own system prompt, and interleaving is what would teach the model to "
                         "answer prompts it was never shown (fb's ruling, 2026-09-02)")
    ap.add_argument("--out", help="write JSONL here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    rows, rep = build(limit=a.limit, max_chars=a.max_chars)
    for r in rows:
        r.setdefault("source", "session")
    if a.subagents:
        subs = sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*/subagents/*.jsonl")))
        srows, srep = build(limit=max(a.limit - len(rows), 0), max_chars=a.max_chars,
                            sessions=subs, verbose=False)
        for r in srows:
            r["source"] = "subagent"
        rows += srows
        rep["subagent_files"] = len(subs)
        rep["subagent_pairs"] = len(srows)
    rows, cred, redacted = drop_credential_rows(rows)
    if cred:
        by_why = collections.Counter(c.get("why", "?") for c in cred)
        print(f"\nDISCARDED {len(cred)} episode(s): "
              + ", ".join(f"{n} {w}" for w, n in sorted(by_why.items())))
        print("  A REAL_CREDENTIAL drops the episode -- a masked line still teaches the model "
              "that a credential belongs there. An over-limit opaque span drops it because "
              f"redacting >{OPAQUE_MAX_REDACT} chars deletes content while leaving the episode "
              "looking intact.")
        for c in cred:
            print(f"  {c['types']}  {c['turns']} turns  {c['project']}  [{c.get('why')}]")
        print("  Content is not printed or written anywhere. The source files still hold "
              "these; that is a separate problem for whoever owns the machine.")
    if redacted:
        spans = sum(r["spans"] for r in redacted)
        turns = sum(r["turns"] for r in redacted)
        print(f"\nREDACTED {spans} opaque span(s) in {len(redacted)} kept episode(s) "
              f"({len(redacted) / max(len(rows) + len(cred), 1):.1%} of episodes, "
              f"{spans} span(s) over {turns} turns), each replaced with "
              f"{OPAQUE_PLACEHOLDER}. These episodes are KEPT: an opaque hit needs no "
              "credential-vs-identifier judgment, because replacing an identifier costs one "
              "span and dropping the episode costs the episode.")
        for r in sorted(redacted, key=lambda x: -x["spans"])[:10]:
            print(f"  {r['spans']:4} span(s)  {r['turns']} turns  {r['project']}")
        if len(redacted) > 10:
            print(f"  ... {len(redacted) - 10} more")
    rows, n_dup = dedupe(rows)
    tokens_note = ""
    if os.path.exists(a.tokenizer):
        before = len(rows)
        rows, over = token_filter(rows, a.tokenizer, a.max_tokens)
        tokens_note = (f"token filter ({a.max_tokens}): {len(rows)} of {before} fit "
                       f"({len(rows) / max(before, 1):.1%}), {over} over")
    else:
        tokens_note = (f"TOKEN FILTER DID NOT RUN: no {a.tokenizer} -- the character "
                       "pre-filter is not the real bound, and this is not a measured result")
    # THE PACK IS WRITTEN BEFORE THE SECRET SCAN, to a .unscanned path, and renamed to --out only
    # after the scan clears. The scan is THE dominant stage -- find_secrets' own docstring measures
    # ~2 ms/line, "about ten minutes for a 10K-pair pack", so ~100 min at --limit 100000 -- and it
    # is where both v11 build attempts died on their own `timeout`, at 30 and 90 minutes, each
    # losing every row after doing all the work. Writing after the scan would not have saved
    # either run.
    #
    # WHY A .unscanned NAME RATHER THAN --out: a real-credential shape discards its episode, so
    # rows on disk before the scan are not yet the pack. The rename is what promotes them, and a
    # kill mid-scan leaves a file whose NAME says it was never cleared -- which is the honest
    # state, and recoverable, instead of nothing at all.
    staged = (a.out + ".unscanned") if a.out else None
    if staged:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(staged, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            # fsync is for a MACHINE crash, and this repo's selftest cannot test it: an in-process
            # kill still runs `with`'s close, so removing these two lines leaves the fixture GREEN
            # (verified as a mutation). Kept because the cost is one syscall and the case it covers
            # -- power loss or a container kill between write and rename -- is real but unobservable
            # from here. Labelled rather than claimed as covered.
            os.fsync(fh.fileno())
        print(f"staged {len(rows)} rows -> {staged} (not yet scanned for secrets)", flush=True)
    hits, scanned, ran = scan_rows(rows)
    # THE SCAN GATES THE RENAME. Until 2026-09-04 it did not: `os.replace(staged, a.out)` ran
    # unconditionally right after this call, so the .unscanned name promoted itself whatever the
    # scan said, and a second unconditional `open(a.out, "w")` at the end of main() wrote the pack
    # again outside the staged path entirely -- which is why v13's log prints
    # `wrote 4799 rows -> ...` TWICE. The staging discipline was real; nothing enforced it.
    #
    # TWO TIERS, and the split is REAL_CREDENTIAL's, not a new judgement:
    #   a REAL_CREDENTIAL type   refuses. The pack does not land, the .unscanned file stays, and
    #                            the row index is printed so the episode can be found and dropped.
    #   any other type           passes, NAMED AND COUNTED. Base64 High Entropy String and Secret
    #                            Keyword fire on example values, log lines and shas; gating on
    #                            them would refuse every pack while removing nothing real (see
    #                            REAL_CREDENTIAL). Printing them is what makes them allowed rather
    #                            than invisible.
    #
    # drop_credential_rows already dropped every episode whose turns carry a REAL_CREDENTIAL hit,
    # so a hit HERE means one survived that pass -- a span assembled across turns, or a detector
    # that fires on the joined text and not on any single turn. That is exactly the case worth
    # refusing on, and the case the old code could not see.
    if ran and rows:
        real = real_credential_rows(rows)
        allowed = sorted(t for t in set(hits) if t not in REAL_CREDENTIAL)
        if real:
            print(f"\nREFUSING TO WRITE {a.out}: {len(real)} row(s) carry a real-credential type "
                  f"that survived drop_credential_rows", file=sys.stderr)
            for i, ts in real[:20]:
                print(f"  row {i}: {ts}", file=sys.stderr)
            if len(real) > 20:
                print(f"  ... {len(real) - 20} more", file=sys.stderr)
            if staged:
                print(f"  the rows are in {staged} -- that name says they were never cleared. "
                      f"Drop these episodes and rebuild; do NOT rename this file by hand.",
                      file=sys.stderr)
            return 1
        print(f"\nsecret scan gate: 0 real-credential row(s), "
              f"{len(allowed)} allowed type(s) {allowed or '[]'} over {scanned:,} chars")
    if staged:
        os.replace(staged, a.out)
        print(f"wrote {len(rows)} rows -> {a.out}", flush=True)
    print(f"\nsessions       {rep['sessions']}")
    print(f"episodes       {rep['episodes']}")
    print(f"kept           {rep['kept']}")
    print(f"chars          {rep['chars']:,} (mean {rep['chars'] // max(rep['kept'], 1)}/episode)")
    print("dropped:")
    for why, n in sorted(rep["dropped"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:7d}  {why}")
    print(f"  {n_dup:7d}  near-duplicate of an earlier episode (same opener and answer)")
    print("top projects:")
    for proj, n in sorted(rep["by_project"].items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {n:7d}  {proj[:70]}  ({n / max(rep['kept'], 1):.1%})")
    if not ran:
        print("\nSECRET SCAN DID NOT RUN (detect-secrets absent) -- this is not a clean result")
    else:
        print(f"\nsecret scan: {len(hits)} type(s) over {scanned:,} chars {hits or ''}")
    print(tokens_note)
    if rows and "tokens" in rows[0]:
        tot = sum(r["tokens"] for r in rows)
        print(f"tokens         {tot:,} in longest pairs (mean {tot // len(rows)}, "
              f"max {max(r['tokens'] for r in rows)})")
    if a.subagents:
        print(f"subagents      {rep.get('subagent_pairs', 0)} pairs from "
              f"{rep.get('subagent_files', 0)} files, source=subagent, never interleaved")
    return 0


if __name__ == "__main__":
    sys.exit(main())

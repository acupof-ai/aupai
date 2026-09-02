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


def scrub(text):
    """Paths and emails out. Returns the text; secrets are handled by find_secrets."""
    for pat, repl in SCRUBS:
        text = pat.sub(repl, text)
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
    """
    try:
        import tempfile

        from detect_secrets.core import scan
        from detect_secrets.settings import transient_settings
    except ImportError:
        return None
    lines = text.splitlines()
    found = set()
    with transient_settings({"plugins_used": SECRET_PLUGINS}):
        for i in range(0, max(len(lines), 1), chunk):
            block = "\n".join(lines[i:i + chunk])
            if not block.strip():
                continue
            path = None
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                                 encoding="utf-8") as fh:
                    fh.write(block + "\n")
                    path = fh.name
                found.update(s.type for s in scan.scan_file(path))
            finally:
                if path and os.path.exists(path):
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
    """
    from loader import format_agentic
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    kept, dropped = [], 0
    for r in rows:
        longest = max((len(tok.encode(pr + co).ids)
                       for pr, co in format_agentic(r["messages"])), default=0)
        if longest > max_tokens:
            dropped += 1
        else:
            kept.append(dict(r, tokens=longest))
    return kept, dropped


def build(limit=10000, max_chars=12000, sessions=None, verbose=True):
    """Episodes -> (rows, report). Rows are {"messages": [...], "project": str}.

    max_chars is a CHARACTER pre-filter, not the token limit fb specified. The real bound
    is 4096 tokens and it cannot be applied on this machine: data/tokenizer.json is
    gitignored and absent here, so any token count I printed would be a guess dressed as a
    measurement. 12000 chars is a deliberately loose upper bound (~3 chars/token on this
    mixed zh/en corpus would put 4096 tokens near 12k chars); the token-exact filter runs
    where the tokenizer is, and the report says so rather than implying it ran.
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
                rep["dropped"]["over max_chars (token filter pending)"] = \
                    rep["dropped"].get("over max_chars (token filter pending)", 0) + 1
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
#: harmless. ALL 164 GO. Separating them needs a classifier for "credential vs identifier"
#: with no ground truth, built to rescue 3.6% of a pack that has 4344 episodes left -- the
#: trade is not close, and a wrong call in that classifier ships a live credential into
#: training data. Over-dropping costs episodes; under-dropping is unrecoverable.
OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9_\-.]{32,}")
OPAQUE_ENTROPY = 4.5


def opaque_credential(text):
    """True if `text` holds a high-entropy opaque token -- the shape find_secrets cannot see.

    Shannon entropy over the token's own characters, the same measure detect_secrets applies,
    but over the token as written rather than over base64-only fragments of it.
    """
    for m in OPAQUE_TOKEN.finditer(text):
        s = m.group(0)
        counts = collections.Counter(s)
        h = -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())
        if h >= OPAQUE_ENTROPY:
            return True
    return False


def drop_credential_rows(rows):
    """(kept, dropped) -- episodes carrying a real credential shape are removed entirely.

    TWO detectors, because one of them cannot see the other's cases. find_secrets covers the
    shapes with a provider rule; opaque_credential covers a high-entropy token with no rule,
    which find_secrets misses whenever the token spans the urlsafe charset (see OPAQUE_TOKEN
    -- a live Lark device code passed the scanner and reached a sample I was about to send to
    a peer).

    Per-turn scan, because the location is what gets reported to a human: fb needs the file
    path and line of any true credential so the user can be told it is sitting in a
    transcript, and "somewhere in the pack" is not a report. The content is never printed or
    written anywhere.
    """
    kept, dropped = [], []
    for r in rows:
        hits = set()
        for m in r["messages"]:
            hits.update(t for t in (find_secrets(m["content"]) or []) if t in REAL_CREDENTIAL)
            if opaque_credential(m["content"]):
                hits.add("Opaque High Entropy Token")
        if hits:
            dropped.append({"project": r["project"], "types": sorted(hits),
                            "turns": len(r["messages"])})
        else:
            kept.append(r)
    return kept, dropped


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
        kept, dropped = drop_credential_rows(real)
        if len(dropped) != 1 or kept:
            fails.append(f"an episode carrying a GitHub token was kept: {len(kept)} kept")
        kept, dropped = drop_credential_rows(soft)
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
    kept, dropped = drop_credential_rows(opaque)
    if len(dropped) != 1 or kept:
        fails.append("an episode carrying an opaque high-entropy token was kept")
    # And the gate must not eat ordinary text: a long identifier at low entropy stays.
    for benign in ("scripts/test_eval_base_prompt_format.py",
                   "p500m_20b_0902_step_0000012000_loss_2p31",
                   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"):
        if opaque_credential(benign):
            fails.append(f"opaque_credential fires on benign text ({_h(benign):.2f}): {benign!r}")

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

    for f in fails:
        print(f"  FAIL {f}")
    print(f"\n{len(fails)} failure(s)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=10000, help="stop after N pairs")
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="character pre-filter; the 4096-token filter runs where the tokenizer is")
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
    rows, cred = drop_credential_rows(rows)
    if cred:
        print(f"\nDISCARDED {len(cred)} episode(s) carrying a real credential shape "
              "(whole episode, not masked -- a masked line still teaches the model that a "
              "credential belongs there):")
        for c in cred:
            print(f"  {c['types']}  {c['turns']} turns  {c['project']}")
        print("  Content is not printed or written anywhere. The source files still hold "
              "these; that is a separate problem for whoever owns the machine.")
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
    hits, scanned, ran = scan_rows(rows)
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
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

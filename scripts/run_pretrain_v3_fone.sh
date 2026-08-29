#!/usr/bin/env bash
# Pretrain on corpus v3 AND --fone.
#
#   scripts/run_pretrain_v3_fone.sh [--dry]
#   NGPU=8 PORT=29600 scripts/run_pretrain_v3_fone.sh --dry
#
# --dry runs every preflight (including the copy of the tokenizer) and prints the
# exact command instead of launching. It is also this script's runnable check: the
# embedded preflight self-tests its cache detector before it uses it.
#
# Preconditions, in the order a mistake would bite. Every one of them fails
# SILENTLY -- the run trains, the loss looks ordinary, the result is a model built
# on the wrong data or scored against the wrong vocabulary:
#   1. the mix names web_hq, not web -- "web" is the UNFILTERED 2.99M-document corpus
#   2. every domain in the mix has a directory with shards in it
#   3. the tokenizer carries the ChatML specials as single tokens
#   4. [NUM] is a single token AT Cfg.num_id -- --fone pins the id rather than
#      resizing, so a vocabulary that moved it trains against the wrong column
#   5. data/tokenizer.json is copied to data/tokenizer_k8.json BEFORE the run:
#      data/tokenizer.json is rebuilt in place, and a same-size rebuild changes
#      every id, so a size check passes and the scores are noise
#   6. the token caches, if present, were built WITH --fone. tokens_<domain>.pt has
#      the same filename either way and train.py's freshness check looks at mtime
#      and vocab id, not at the flag.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=${NAME:-k8_v3_fone}
MIX=${MIX:-data/mix_v3.json}
NGPU=${NGPU:-8}
PORT=${PORT:-29600}
TOK_COPY=data/tokenizer_k8.json
DRY=0
if [ "${1:-}" = "--dry" ]; then
  DRY=1
  shift
fi

PRE=$(python3 - "$MIX" "$TOK_COPY" <<'PY'
import json
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("scripts"))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import train  # noqa: E402
from loader import vocab_fingerprint  # noqa: E402

mix_path, tok_copy = sys.argv[1], sys.argv[2]
mix = json.load(open(mix_path, encoding="utf-8"))
doms = list(mix["domains"])
assert "web" not in doms, (
    "the mix names 'web', which is the UNFILTERED corpus. train.py globs "
    "data/corpus/<domain>/, so this would train on 2,991,648 unfiltered documents "
    "and silently discard every filter. Use web_hq."
)
for d in doms:
    p = os.path.join("data", "corpus", d)
    shards = os.listdir(p) if os.path.isdir(p) else []
    n = sum(1 for f in shards if f.endswith(".jsonl"))
    assert n, f"data/corpus/{d} has no .jsonl shards"
    print(f"  {d:<9} {n:>4} shards")

w = sum(v["weight"] for v in mix["domains"].values())
a = sum(v["anneal"] for v in mix["domains"].values())
assert abs(w - 1) < 1e-6 and abs(a - 1) < 1e-6, f"weights sum {w}, anneal sum {a}; both must be 1"
print(f"  mix total {mix['total_tokens'] / 1e9:.2f}B tokens over {len(doms)} domains")

tok = Tokenizer.from_file("data/tokenizer.json")
for sp in ("<|im_start|>", "<|im_end|>", "<eos>"):
    ids = tok.encode(sp, add_special_tokens=False).ids
    assert len(ids) == 1, f"{sp} is {len(ids)} tokens, not 1 -- rebuild with scripts/build_tokenizer.py"

# --fone does not resize the embedding: Cfg.num_id names a slot that must already
# hold [NUM]. A vocabulary that moved it would train the number path against some
# ordinary token and raise nothing.
num_ids = tok.encode("[NUM]", add_special_tokens=False).ids
assert num_ids == [train.Cfg.num_id], (
    f"[NUM] encodes to {num_ids}, not [{train.Cfg.num_id}] = Cfg.num_id. --fone writes every "
    "number as that id; against this vocabulary it would mean a different token and the digit "
    "head would be trained on noise. Rebuild with scripts/build_tokenizer.py or fix Cfg.num_id."
)
fp = vocab_fingerprint(tok)
print(f"  tokenizer vocab {tok.get_vocab_size()}, ChatML specials single, [NUM] at {train.Cfg.num_id}")

if os.path.exists(tok_copy):
    old = vocab_fingerprint(Tokenizer.from_file(tok_copy))
    assert old == fp, (
        f"{tok_copy} already exists with fingerprint {old}, but data/tokenizer.json is {fp}. "
        "Overwriting it would destroy the vocabulary of whatever was already trained as k8. "
        "Move the old file aside deliberately, or run under a different NAME."
    )
    print(f"  {tok_copy} already holds this vocabulary")
else:
    shutil.copyfile("data/tokenizer.json", tok_copy)
    print(f"  copied data/tokenizer.json -> {tok_copy} (survives the next in-place rebuild)")


def cache_is_fone(path):
    """True if tokens_<domain>.pt holds (ids, values) rather than a bare ids tensor.

    Read from the zip directory, not by loading the file: these caches are tens of
    GB. torch.save writes one `data/<n>` entry per storage, so a bare tensor has
    one and the --fone tuple has two.
    """
    with zipfile.ZipFile(path) as z:
        n = sum(1 for e in z.namelist() if os.path.basename(os.path.dirname(e)) == "data")
    assert n in (1, 2), f"{path}: {n} storages, expected 1 (plain) or 2 (--fone)"
    return n == 2


with tempfile.TemporaryDirectory() as td:
    plain, pair = os.path.join(td, "a.pt"), os.path.join(td, "b.pt")
    torch.save(torch.zeros(4, dtype=torch.int32), plain)
    torch.save((torch.zeros(4, dtype=torch.int32), torch.zeros(2)), pair)
    assert not cache_is_fone(plain) and cache_is_fone(pair), "cache_is_fone selftest failed"

# Verify THIS train.py namespaces caches by --fone: the pod is not a git repo and
# receives files by hand, so a stale train.py is the normal way this regresses.
import ast

_src = open("train.py", encoding="utf-8").read()
assert "_domain_cache_path" in _src, (
    "this train.py predates the FoNE cache namespacing: it would load a non-FoNE token "
    "cache as fresh and unpack `ids, vals = data` off a 1-D tensor, 40 minutes in."
)
_fn = next(
    n for n in ast.walk(ast.parse(_src))
    if isinstance(n, ast.FunctionDef) and n.name == "_domain_cache_path"
)
assert "fone" in ast.unparse(_fn), "_domain_cache_path no longer varies with --fone"
print("  token cache: namespaced by --fone, no deletion needed")
print(f"VOCAB_ID {fp}")
PY
)
printf '%s\n' "$PRE"

FP=$(printf '%s\n' "$PRE" | awk '/^VOCAB_ID /{print $2}')
[ -n "$FP" ] || { echo "preflight did not report a tokenizer fingerprint"; exit 1; }

# check_mix.py sizes each cache as int32 ids only, so under --fone its token counts
# are inflated by the values array. The capping decisions it prints are still the
# ones train.py will make.
echo "checking the epoch caps against the real pools (counts run high for --fone caches)..."
if ! python3 scripts/check_mix.py --mix "$MIX"; then
  # A real launch stops here; --dry still prints the command, so the schedule can be
  # reviewed on a box that has the corpus but not the caches yet.
  [ "$DRY" = "1" ] || exit 1
  echo "  (--dry: continuing past check_mix)"
fi

FLAGS="--mix $MIX --fone --fp8 --attn_res --attn_res_blocks 4 --warmup 150 --lr_scale 0.5"
# run_pretrain.sh owns the experiment record (exp.py start before torchrun, exp.py
# done with the last logged step after it) plus the GPU-busy and port guards.
# NOTES carries the fingerprint into runs/experiments.jsonl, because the file it
# names will be overwritten in place by the next tokenizer rebuild.
INNER="NGPU=$NGPU PORT=$PORT \
NOTES='corpus v3 + --fone, ${NGPU} GPUs, vocab $FP (copied to $TOK_COPY)' \
HYP='FoNE was measured only on the v2 unfiltered corpus, where it fixed arithmetic \
(wrong-equation 43.3%->32.7%) and moved math-hard not at all. Does it move the score on v3?' \
scripts/run_pretrain.sh $NAME $FLAGS"
# setsid, not nohup: pod shells run through `crictl exec`, and when that session
# ends the kernel kills the whole process group -- nohup only blocks SIGHUP.
LAUNCH="setsid nohup bash -c \"$INNER > runs/${NAME}_launch.log 2>&1\" </dev/null >/dev/null 2>&1 &"

echo
echo "torchrun --nproc_per_node=$NGPU --master_port=$PORT train.py --name $NAME $FLAGS"
echo
echo "launch line (run_pretrain.sh wraps it with the experiment record):"
echo "  $LAUNCH"
if [ "$DRY" = "1" ]; then
  echo
  echo "--dry: preflight passed, nothing launched. Poll runs/$NAME.log after launching."
  exit 0
fi
eval "$LAUNCH"
echo "launched $NAME detached; poll runs/$NAME.log and runs/${NAME}_launch.log"

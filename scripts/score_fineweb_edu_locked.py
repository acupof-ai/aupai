"""Score the locked hand-read sets with the OFFICIAL FineWeb-Edu classifier.

Offline, CPU-only. Score semantics follow the official model card exactly
(HuggingFaceFW/fineweb-edu-classifier): the model head is a single REGRESSION output,
so the educational score is the RAW LOGIT (0-5 scale), NOT a sigmoid, and there is no
prompt prefix. int_score = round(clamp(score,0,5)); the official curation cut is >=3.

Inputs (read-only, no retraining, no threshold chosen):
- data/corpus/sample/web_labels.jsonl  180 zh web rows, y=1 educational / y=0 not
- data/corpus/sample/cci3_audit_400.jsonl 400 zh CCI3 rows (per-row hand labels were
  session-only and are not in the repo, so this reports the SCORE DISTRIBUTION only;
  AUC is not computed here)
- runs/code_rp1t_markup_handread.json code_sample_100: 100 real OSS source files used
  as the "should-keep code" proxy to quantify the code false-kill rate at the >=3 cut.

Outputs a JSON summary to stdout.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "fineweb-edu-classifier"
MAXLEN = 512
BATCH = 16


def score_texts(texts):
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(
                texts[i : i + BATCH], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN
            )
            s = model(**enc).logits.float().squeeze(-1).numpy()
            out.extend(s.tolist())
    return np.array(out)


def dist(scores):
    return {
        "n": int(len(scores)),
        "mean": round(float(scores.mean()), 4),
        "median": round(float(np.median(scores)), 4),
        "p10": round(float(np.quantile(scores, 0.10)), 3),
        "p25": round(float(np.quantile(scores, 0.25)), 3),
        "p75": round(float(np.quantile(scores, 0.75)), 3),
        "p90": round(float(np.quantile(scores, 0.90)), 3),
        "min": round(float(scores.min()), 3),
        "max": round(float(scores.max()), 3),
        "frac_ge3": round(float((scores >= 3.0).mean()), 4),
        "frac_ge0": round(float((scores >= 0.0).mean()), 4),
    }


def main():
    res = {
        "model": "HuggingFaceFW/fineweb-edu-classifier",
        "score": "raw regression logit 0-5 (model card), cut>=3",
        "max_length": MAXLEN,
    }

    # 1. web_labels: the only set with in-repo per-row hand labels -> real AUC
    with open(ROOT / "data/corpus/sample/web_labels.jsonl") as f:
        wl = [json.loads(l) for l in f]
    wl_text = [r["t"] for r in wl]
    wl_y = np.array([int(r["y"]) for r in wl])
    wl_s = score_texts(wl_text)
    auc = roc_auc_score(wl_y, wl_s)
    # recall at the official cut: P(score>=3 | y=1 educational kept) and junk killed
    keep = wl_s >= 3.0
    edu_kept = float(keep[wl_y == 1].mean())
    junk_cut = float((~keep)[wl_y == 0].mean())
    res["zh_web_labels"] = {
        "n": len(wl),
        "n_y1_educational": int(wl_y.sum()),
        "n_y0_not": int((wl_y == 0).sum()),
        "auc_score_vs_y": round(float(auc), 4),
        "all": dist(wl_s),
        "score_distribution_y1": dist(wl_s[wl_y == 1]),
        "score_distribution_y0": dist(wl_s[wl_y == 0]),
        "at_cut3_educational_recall_kept": round(edu_kept, 4),
        "at_cut3_notedu_kill_rate": round(junk_cut, 4),
    }

    # 2. cci3 locked 400: distribution only (no in-repo per-row labels)
    with open(ROOT / "data/corpus/sample/cci3_audit_400.jsonl") as f:
        cci = [json.loads(l) for l in f]
    cci_s = score_texts([r["text"] for r in cci])
    res["zh_cci3_locked400_distribution_only"] = dist(cci_s)

    # 3. code 100: should-keep OSS source -> false-kill proxy at cut 3
    with open(ROOT / "runs/code_rp1t_markup_handread.json") as f:
        code = json.load(f)["code_sample_100"]
    code_texts = [r["head"] for r in code]  # 'head' holds the file-head source text
    code_s = score_texts(code_texts)
    res["code_oss100_should_keep"] = {
        "n": len(code_s),
        "score_distribution": dist(code_s),
        "false_kill_frac_at_cut3": round(float((code_s < 3.0).mean()), 4),
        "false_kill_frac_below0": round(float((code_s < 0.0).mean()), 4),
        "kept_frac_at_cut3": round(float((code_s >= 3.0).mean()), 4),
    }

    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())

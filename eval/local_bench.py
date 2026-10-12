"""Load offline quick multiple-choice benchmark samples."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICK_DIR = ROOT / "data" / "eval" / "quick"
FILES = {
    "hellaswag": "hellaswag_quick32.jsonl",
    "piqa": "piqa_quick32.jsonl",
    "arc-easy": "arc-easy_quick32.jsonl",
    "arc-challenge": "arc-challenge_quick32.jsonl",
    "winogrande": "winogrande_quick32.jsonl",
}


def available(name):
    return (QUICK_DIR / FILES[name]).exists()


def load(name):
    path = QUICK_DIR / FILES[name]
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

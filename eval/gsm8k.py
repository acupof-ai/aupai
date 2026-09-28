"""GSM8K evaluation via greedy generation (Chinese prompts).

1319 grade-school math problems. Prompt = "问：{q}\n答：", generate up to 256
tokens greedily, take the last number in the response, compare to "#### N".
Batched generation (batch of 8); prompts are right-padded, which is safe
because pad tokens always sit right of real content and causal attention
never looks right.
"""

import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import generate_batch  # noqa: F401  (re-exported: math_hard.py and math_zh.py import it from here)
from scripts.loader import load_checkpoint, load_tokenizer, prompt_fn

EOS_ID = 1
MAX_CTX = 4096  # the model's trained seq len; smaller truncates the model's own long reasoning away
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval", "gsm8k_test.jsonl")
CUTS = ("\nQ:", "\nQuestion:", "\n问：", "<|im_end|>")


def load_dataset():
    if os.path.exists(LOCAL):  # the pod has no Hub access; the same 1319 test rows
        import json

        with open(LOCAL, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset

    return load_dataset("openai/gsm8k", "main", split="test")


# The 8 chain-of-thought exemplars of Wei et al. 2022 (lm-eval-harness gsm8k-cot), in its Q:/A: layout.
COT8 = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, "
     "there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have "
     "been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, "
     "they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did "
     "Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. "
     "The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. "
     "The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to "
     "thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers "
     "were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf "
     "balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, "
     "he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars "
     "left. 23 - 15 is 8. The answer is 8."),
]


def fewshot_fmt(k=8):
    """Base-model prompt: the first k standard CoT exemplars, then the question."""
    shots = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in COT8[:k])

    def fmt(q):
        return f"{shots}Q: {q}\nA:"

    fmt.__name__ = f"cot{k}"
    return fmt


def extract_number(text):
    """Last number in text (commas stripped); None if absent."""
    nums = NUM_RE.findall(text)
    return float(nums[-1].replace(",", "")) if nums else None


@torch.no_grad()
def evaluate(model, tok, device, batch_size=8, temperature=0.0, fmt=None):
    # No default format. `fmt=format_prompt` would be the defect with a friendlier face:
    # a base checkpoint scored in ChatML reads zero on an unseen prefix, not on capability
    # (AGENTS.md:200; 1.6% vs 94.4% fence rate, eval/score_code_exec.py:9-31). The caller
    # holds the cfg, so the caller decides -- prompt_fn(classify(cfg, name)).
    if fmt is None:
        raise ValueError("evaluate() needs fmt=prompt_fn(classify(cfg, name)); a default "
                         "would silently score a base checkpoint in ChatML")
    rows = list(load_dataset())
    correct = total = 0

    for s in range(0, len(rows), batch_size):
        batch = rows[s : s + batch_size]
        p_ids = [tok.encode(fmt(r["question"])).ids for r in batch]
        golds = [float(r["answer"].split("####")[-1].replace(",", "").strip()) for r in batch]

        for out_ids, gold in zip(generate_batch(model, p_ids, 256, device, temperature), golds, strict=True):
            text = tok.decode(out_ids)
            for c in CUTS:  # a few-shot continuation runs on into the next question
                text = text.split(c)[0]
            pred = extract_number(text)
            total += 1
            if pred is not None and abs(pred - gold) < 1e-4:
                correct += 1

        if total % 128 == 0 or total == len(rows):
            print(f"  {total}/{len(rows)} acc={correct / total:.2%}", flush=True)

    acc = correct / total
    print(f"GSM8K: {correct}/{total} = {acc:.2%} (t={temperature}, fmt={fmt.__name__})")
    return acc


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score_matrix import classify

    ckpt = sys.argv[1] if len(sys.argv) > 1 else "ckpt_sft.pt"
    shots = int(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else 0
    model, cfg = load_checkpoint(ckpt, device="cuda")
    model = model.to(torch.bfloat16)
    tok = load_tokenizer("data/tokenizer.json", cfg)
    # classify, not an assumption: the old default was ckpt_sft.pt, so pointing this at a
    # base checkpoint by hand silently scored it in ChatML.
    kind = classify(cfg, os.path.basename(ckpt))
    if shots and kind == "base":
        evaluate(model, tok, "cuda", fmt=fewshot_fmt(shots))
    else:
        evaluate(model, tok, "cuda", fmt=prompt_fn(kind))

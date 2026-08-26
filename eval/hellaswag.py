"""HellaSwag evaluation: pick the most likely continuation via log-likelihood."""

import torch
from datasets import load_dataset as hf_load_dataset


def load_dataset():
    ds = hf_load_dataset("Rowan/hellaswag", split="validation")
    items = []
    for ex in ds:
        ctx = ex.get("ctx") or (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
        items.append({"context": ctx, "options": list(ex["endings"]), "label": int(ex["label"])})
    return items


@torch.no_grad()
def log_likelihood(model, tok, context, continuation, device):
    """Sum log-prob of continuation tokens given context."""
    ctx_ids = tok.encode(context).ids
    cont_ids = tok.encode(continuation).ids
    full = torch.tensor([ctx_ids + cont_ids], device=device)
    logits, _ = model(full)
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
    # Score only the continuation tokens
    cont_log_probs = log_probs[range(len(ctx_ids) - 1, len(ctx_ids) + len(cont_ids) - 1), cont_ids]
    return cont_log_probs.sum().item()


def evaluate(model, tok, device):
    model.eval()
    data = load_dataset()
    correct = 0
    for item in data:
        scores = [log_likelihood(model, tok, item["context"], opt, device) for opt in item["options"]]
        if int(max(range(len(scores)), key=scores.__getitem__)) == item["label"]:
            correct += 1
    return correct / len(data)


if __name__ == "__main__":
    # Smoke test with dummy data (random model -> ~25% accuracy expected).
    class _DummyTok:
        def encode(self, text):
            ids = [abs(hash(w)) % 1000 for w in text.split()] or [0]
            return type("Enc", (), {"ids": ids})()

    class _DummyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(1000, 32)
            self.head = torch.nn.Linear(32, 1000)

        def forward(self, x):
            return self.head(self.emb(x)), None

    dummy = [
        {"context": "A man is running", "options": ["fast.", "slowly.", "quick.", "swift."], "label": 0},
        {"context": "The cat sat on", "options": ["the mat.", "the moon.", "the car.", "the roof."], "label": 0},
    ]
    model = _DummyLM()
    acc = sum(
        max(range(4), key=lambda i: log_likelihood(model, _DummyTok(), d["context"], d["options"][i], "cpu")) == d["label"]
        for d in dummy
    ) / len(dummy)
    print(f"dummy accuracy: {acc:.2%}")

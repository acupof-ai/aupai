"""WinoGrande evaluation: pick the option that best fills the blank via log-likelihood."""

import torch
from datasets import load_dataset as hf_load_dataset


def load_dataset():
    ds = hf_load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    items = []
    for ex in ds:
        prefix, _, suffix = ex["sentence"].partition("_")
        # Continuation = option + text after the blank, so options are scored in full-sentence context.
        items.append({
            "context": prefix.strip(),
            "options": [ex["option1"] + suffix, ex["option2"] + suffix],
            "label": int(ex["answer"]) - 1,
        })
    return items


@torch.no_grad()
def log_likelihood(model, tok, context, continuation, device):
    """Sum log-prob of continuation tokens given context."""
    ctx_ids = tok.encode(context).ids
    cont_ids = tok.encode(continuation).ids
    full = torch.tensor([ctx_ids + cont_ids], device=device)
    logits, _ = model(full)
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
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
    # Smoke test: random model, expect ~50%.
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
        {"context": "The trophy doesn't fit into the brown suitcase because", "options": ["the trophy is too large.", "the suitcase is too large."], "label": 0},
        {"context": "Joan made sure to thank Susan for the help that", "options": ["Susan had given.", "Joan had given."], "label": 0},
    ]
    model = _DummyLM()
    acc = sum(
        max(range(2), key=lambda i: log_likelihood(model, _DummyTok(), d["context"], d["options"][i], "cpu")) == d["label"]
        for d in dummy
    ) / len(dummy)
    print(f"dummy accuracy: {acc:.2%}")

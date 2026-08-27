#!/usr/bin/env python3
"""Download Chinese Cosmopedia from ModelScope, filter textbook/wikihow, mix with existing corpus."""

import glob
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MIX_DIR = os.path.join(DATA, "mix")
os.makedirs(MIX_DIR, exist_ok=True)

TARGET_CHARS = 45_000_000  # ~30M tokens at 1.5 chars/token
KEEP_FORMATS = {"middle school textbook", "college textbook", "wikihow"}
PARQUET_URL = (
    "https://www.modelscope.cn/datasets/OpenCSG/Chinese-Cosmopedia/resolve/master/data/00000.parquet"
)
PARQUET_PATH = "/tmp/cosmopedia_00000.parquet"

# --- 1. Download one parquet file ---
if not os.path.exists(PARQUET_PATH):
    print(f"Downloading {PARQUET_URL}...", flush=True)
    urllib.request.urlretrieve(PARQUET_URL, PARQUET_PATH)
    print(f"Downloaded {os.path.getsize(PARQUET_PATH) / 1e9:.2f}GB", flush=True)
else:
    print(f"Using cached {PARQUET_PATH}", flush=True)

# --- 2. Read and filter ---
import pyarrow.parquet as pq

print("Reading parquet...", flush=True)
table = pq.read_table(PARQUET_PATH)
df = table.to_pandas()
print(f"Total rows: {len(df)}, columns: {list(df.columns)}", flush=True)
print(f"Formats: {df['data_format'].value_counts().to_dict()}", flush=True)

# Filter for textbook/wikihow
mask = df["data_format"].isin(KEEP_FORMATS)
filtered = df[mask]
print(f"After format filter: {len(filtered)} rows", flush=True)

# Filter by length
filtered = filtered[filtered["text"].str.len() >= 200]
print(f"After length filter: {len(filtered)} rows", flush=True)

# Take enough for target chars
texts = []
total_chars = 0
for text in filtered["text"]:
    texts.append(text)
    total_chars += len(text)
    if total_chars >= TARGET_CHARS:
        break

print(
    f"Selected: {len(texts)} docs, {total_chars / 1e6:.1f}M chars (~{total_chars / 1.5 / 1e6:.0f}M tokens)",
    flush=True,
)

# Save Cosmopedia subset
cosmo_path = os.path.join(MIX_DIR, "cosmopedia.jsonl")
with open(cosmo_path, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(json.dumps({"content": t}, ensure_ascii=False) + "\n")
print(f"saved {cosmo_path}", flush=True)

# --- 3. Load existing corpus ---
existing = []
for name in ("core.txt", "framework.md", "method.txt"):
    p = os.path.join(DATA, name)
    if os.path.exists(p):
        existing.append(open(p, encoding="utf-8").read())
for pat in ("corpus/*.jsonl", "corpus/primary/*.jsonl"):
    for p in sorted(glob.glob(os.path.join(DATA, pat))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                existing.append(json.loads(line)["content"])

existing_chars = sum(len(t) for t in existing)
print(
    f"Existing corpus: {len(existing)} docs, {existing_chars / 1e6:.2f}M chars "
    f"(~{existing_chars / 1.5 / 1e6:.1f}M tokens)",
    flush=True,
)

# --- 4. Mix: upsample existing 3x (~10% of total) ---
UPSAMPLE = 3
mixed_path = os.path.join(MIX_DIR, "mixed.jsonl")
with open(mixed_path, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(json.dumps({"content": t}, ensure_ascii=False) + "\n")
    for _ in range(UPSAMPLE):
        for t in existing:
            f.write(json.dumps({"content": t}, ensure_ascii=False) + "\n")

total_mixed = total_chars + existing_chars * UPSAMPLE
print(
    f"Mixed: {total_mixed / 1e6:.1f}M chars (~{total_mixed / 1.5 / 1e6:.0f}M tokens) -> {mixed_path}",
    flush=True,
)
print(
    f"Ratio: cosmopedia {total_chars / total_mixed * 100:.0f}% / "
    f"existing {existing_chars * UPSAMPLE / total_mixed * 100:.0f}%",
    flush=True,
)

#!/usr/bin/env python3
"""Augment first-level derivatives with 3 format variations each.
Variations: Q&A format, list/card format, summary+detail format.
Simple rule-based transformations — fast, no LLM needed."""

import json
import random
import re
import sys

random.seed(42)


def to_qa(doc):
    """Convert prose to Q&A format: extract key points as questions."""
    content = doc["content"]
    title = doc.get("title", "")
    # Split into sentences
    sentences = re.split(r"(?<=[。！？])", content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    qa_parts = [f"# {title}（问答版）\n"]
    # Take key sentences and turn them into Q&A
    for i, s in enumerate(sentences[:8]):
        # Extract the main concept and create a question
        if "——" in s or "：" in s:
            concept = s.split("——")[0].split("：")[0][:30]
            qa_parts.append(f"**问：{concept}是什么？**\n\n{s}\n")
        elif i < 3:
            qa_parts.append(f"**问：关于{title[:15]}，第{i + 1}个要点是什么？**\n\n{s}\n")
    if len(sentences) > 8:
        qa_parts.append(f"\n**补充说明：**\n{''.join(sentences[8:])}\n")
    # Keep the 关联 line
    for s in sentences:
        if "关联" in s or "骨架" in s:
            qa_parts.append(f"\n{s}\n")
    return {"type": "prose_entry_qa", "title": f"{title}（问答版）", "content": "\n".join(qa_parts)}


def to_list(doc):
    """Convert prose to list/card format."""
    content = doc["content"]
    title = doc.get("title", "")
    sentences = re.split(r"(?<=[。！？])", content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    parts = [f"# {title}（清单版）\n"]
    parts.append("## 核心要点\n")
    for i, s in enumerate(sentences[:10], 1):
        # Clean up the sentence for list format
        s = s.lstrip("所以").lstrip("因此").lstrip("这").lstrip("那")
        parts.append(f"{i}. {s}")
    if len(sentences) > 10:
        parts.append(f"\n## 详细说明\n\n{''.join(sentences[10:])}")
    # Keep the 关联 line
    for s in sentences:
        if "关联" in s or "骨架" in s:
            parts.append(f"\n---\n{s}")
    return {"type": "prose_entry_list", "title": f"{title}（清单版）", "content": "\n".join(parts)}


def to_summary(doc):
    """Create a condensed + expanded version: summary first, then original with transitions."""
    content = doc["content"]
    title = doc.get("title", "")
    sentences = re.split(r"(?<=[。！？])", content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    # Summary = first 2 + last meaningful sentence
    summary = "".join(sentences[:2])
    if len(sentences) > 4:
        summary += sentences[-1] if "关联" in sentences[-1] else sentences[3]

    parts = [f"# {title}（详解版）\n"]
    parts.append(f"## 概述\n\n{summary}\n")
    parts.append("## 详细分析\n")

    # Add transitional phrases
    transitions = [
        "进一步来看，",
        "具体而言，",
        "值得注意的是，",
        "换个角度看，",
        "这背后的逻辑是，",
        "从实践角度说，",
        "更深层地，",
    ]
    for i, s in enumerate(sentences):
        if i > 0 and i % 3 == 0 and i < len(transitions) * 3:
            t = transitions[i // 3 - 1]
            parts.append(f"\n{t}{s}")
        else:
            parts.append(s)
    return {"type": "prose_entry_detail", "title": f"{title}（详解版）", "content": "\n".join(parts)}


def main():
    inp = sys.argv[1]
    out = sys.argv[2]
    docs = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    augmented = []
    for doc in docs:
        augmented.append(doc)  # original
        augmented.append(to_qa(doc))
        augmented.append(to_list(doc))
        augmented.append(to_summary(doc))

    with open(out, "w", encoding="utf-8") as f:
        for doc in augmented:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"original: {len(docs)}, augmented: {len(augmented)}", file=sys.stderr)


if __name__ == "__main__":
    main()

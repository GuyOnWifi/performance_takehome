"""Generate the seeded prompt sets, cut to exact token lengths with the pinned tokenizer."""

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CORPUS = ("README.md", "docs/*.md", "tasks/*/*.md", "agent-harness/*.md", "benchmarks/*.py")
COUNTS = {"short": 1000, "long": 200, "long_gen": 200, "prefix": 200, "anchor": 150, "canary": 120}
PREFIX_QUESTION_TOKENS = 256

QUESTIONS = (
    "What is the main point of the passage above?",
    "Which requirement above is the easiest to get wrong, and why?",
    "Explain the passage above to a new engineer.",
    "What would you test first, given the passage above?",
    "List the assumptions the passage above depends on.",
)
LONG_TASKS = (
    "Summarize the text above in five bullet points.",
    "List every requirement stated in the text above.",
    "Find the weakest claim in the text above and explain why it is weak.",
    "Rewrite the key ideas of the text above as a checklist.",
)
TOPICS = (
    "database indexing", "TCP congestion control", "garbage collection", "compiler register "
    "allocation", "memory hierarchies", "consensus protocols", "load balancing", "file systems",
    "virtual memory", "stream processing", "cache coherence", "query planning",
)
FORMS = (
    "Write a detailed technical guide to {topic}. Cover the history, mechanisms, tradeoffs and "
    "worked examples, and keep going until every part is thorough.",
    "Write a long tutorial on {topic} for a working engineer, with numbered sections and "
    "examples in each one.",
    "Write an in-depth comparison of the main approaches to {topic}, with a section per approach.",
)


def corpus() -> list[str]:
    paragraphs = []
    for pattern in CORPUS:
        for path in sorted(ROOT.glob(pattern)):
            paragraphs += [
                part.strip() for part in path.read_text(encoding="utf-8").split("\n\n")
                if part.strip()
            ]
    return paragraphs


def body(tokenizer, paragraphs: list[str], rng: random.Random, tokens: int) -> str:
    """Seeded passage of exactly `tokens` tokens, cut at a token boundary."""
    if tokens <= 0:
        return ""
    picked, count = [], 0
    while count < tokens:
        paragraph = rng.choice(paragraphs)
        picked.append(paragraph)
        count += len(tokenizer.encode(paragraph, add_special_tokens=False).ids)
    text = "\n\n".join(picked)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    return text[: encoding.offsets[tokens - 1][1]]


def passage_prompt(tokenizer, paragraphs, rng, tokens: int, tail: str) -> str:
    size = len(tokenizer.encode(tail, add_special_tokens=False).ids) + 2
    return f"{body(tokenizer, paragraphs, rng, tokens - size)}\n\n{tail}"


def build(tokenizer, seed: int) -> dict[str, list[dict]]:
    paragraphs = corpus()

    def chat(name: str, index: int, text: str, **extra) -> dict:
        return {"prompt_id": f"{name}-{index:04d}",
                "messages": [{"role": "user", "content": text}], **extra}

    sets: dict[str, list[dict]] = {}
    rng = random.Random(f"{seed}/short")
    sets["short"] = [
        chat("short", i, passage_prompt(tokenizer, paragraphs, rng, 256, rng.choice(QUESTIONS)))
        for i in range(COUNTS["short"])
    ]
    rng = random.Random(f"{seed}/long")
    sets["long"] = [
        chat("long", i, passage_prompt(tokenizer, paragraphs, rng, 4096, rng.choice(LONG_TASKS)))
        for i in range(COUNTS["long"])
    ]
    rng = random.Random(f"{seed}/long_gen")
    sets["long_gen"] = [
        chat("long_gen", i, passage_prompt(
            tokenizer, paragraphs, rng, 128,
            rng.choice(FORMS).format(topic=rng.choice(TOPICS)),
        ))
        for i in range(COUNTS["long_gen"])
    ]
    rng = random.Random(f"{seed}/prefix")
    shared = body(tokenizer, paragraphs, rng, 4096 - PREFIX_QUESTION_TOKENS)
    sets["prefix"] = [
        chat("prefix", i, f"{shared}\n\n" + passage_prompt(
            tokenizer, paragraphs, rng, PREFIX_QUESTION_TOKENS, rng.choice(QUESTIONS)
        ))
        for i in range(COUNTS["prefix"])
    ]
    rng = random.Random(f"{seed}/anchor")
    sets["anchor"] = [
        chat("anchor", i, passage_prompt(tokenizer, paragraphs, rng, 256, rng.choice(QUESTIONS)))
        for i in range(COUNTS["anchor"])
    ]
    rng = random.Random(f"{seed}/canary")
    sets["canary"] = []
    for i in range(COUNTS["canary"]):
        left, right = rng.randint(100, 999), rng.randint(100, 999)
        text = f"Compute {left} + {right}. Reply with only the number. /no_think"
        sets["canary"].append(chat("canary", i, text, expected=str(left + right)))
    return sets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        hf_hub_download(args.repository, "tokenizer.json", revision=args.revision)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name, prompts in build(tokenizer, args.seed).items():
        data = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts).encode()
        (args.out / f"{name}.jsonl").write_bytes(data)
        sizes = [
            len(tokenizer.encode(p["messages"][0]["content"], add_special_tokens=False).ids)
            for p in prompts
        ]
        print(json.dumps({
            "set": name, "prompts": len(prompts), "tokens_min": min(sizes),
            "tokens_max": max(sizes), "sha256": hashlib.sha256(data).hexdigest(),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

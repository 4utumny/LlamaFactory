"""Prepare disjoint Reddit preference data and fixed evaluation posts for Clean DPO."""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import qwen25_benchmarks as bench  # noqa: E402


SYSTEM = "You are a helpful assistant."


def post_keys(info):
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", info["post"])).strip().casefold()
    return {f"id:{info['id']}", f"body:{bench.digest(normalized)}"}


def post_row(info, row_id):
    post = f"SUBREDDIT: r/{info['subreddit']}\nTITLE: {info['title']}\nPOST: {info['post']}"
    return {
        "id": row_id,
        "post": post,
        "prompt": "Summarize the following forum post in one concise paragraph. Return only the summary."
        f"\n\n{post}\n\nTL;DR:",
    }


def comparisons(data_root, split):
    import pyarrow.parquet as pq

    for path in sorted((data_root / "summarize_from_feedback_raw/comparisons" / split).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()


def preference(row):
    texts = row["summaries"]
    texts = texts["text"] if isinstance(texts, dict) else [s["text"] for s in texts]
    if row["choice"] not in (0, 1):
        raise ValueError(f"Invalid preference choice for {row['info']['id']}")
    chosen, rejected = texts[row["choice"]].strip(), texts[1 - row["choice"]].strip()
    prompt = post_row(row["info"], row["info"]["id"])["prompt"]
    return {
        "instruction": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "post_id": row["info"]["id"],
    }


def build_splits(train_source, val_source, test, seed=42, val_size=1024, dev_size=128):
    test_keys = set().union(*(post_keys(row) for row in test))
    train_keys, seen_pairs, train = set(), set(), []
    stats = Counter()
    for row in train_source:
        stats["train_raw"] += 1
        info = row["info"]
        keys = post_keys(info)
        train_keys.update(keys)
        if keys & test_keys:
            stats["train_test_overlap"] += 1
            continue
        record = preference(row)
        pair = bench.digest(record)
        if not record["chosen"] or not record["rejected"] or record["chosen"] == record["rejected"]:
            stats["train_empty_or_identical"] += 1
        elif pair in seen_pairs:
            stats["train_duplicate_pair"] += 1
        else:
            train.append(record)
            seen_pairs.add(pair)

    # One comparison per validation post; ID and body both participate in isolation.
    val_keys, pool = set(), []
    for row in val_source:
        stats["validation_raw"] += 1
        info = row["info"]
        if not info.get("post"):
            stats["validation_article"] += 1
            continue
        keys = post_keys(info)
        if keys & test_keys or keys & train_keys or keys & val_keys:
            continue
        record = preference(row)
        if record["chosen"] and record["rejected"] and record["chosen"] != record["rejected"]:
            pool.append((record, post_row(info, f"dev:{info['id']}")))
            val_keys.update(keys)
    random.Random(seed).shuffle(pool)
    if len(pool) < val_size + dev_size:
        raise ValueError(f"Need {val_size + dev_size} isolated validation posts; found {len(pool)}")
    val = [record for record, _ in pool[:val_size]]
    dev = [post for _, post in pool[val_size : val_size + dev_size]]
    selected, excluded = [], []
    for i, row in enumerate(test[:200]):
        row_id = f"sff:{i}:{row['id']}"
        (excluded if post_keys(row) & train_keys else selected).append(row_id)
    return train, val, dev, selected, excluded, dict(stats)


def check_lengths(train, val, dev, model_path):
    from transformers import AutoTokenizer

    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen", default_system=SYSTEM))
    for row in train[:8]:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["chosen"]},
        ]
        prompt_ids, _ = template.encode_oneturn(tokenizer, messages, SYSTEM)
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, messages[0]],
            add_generation_prompt=True,
        )
        if prompt_ids != chat_ids:
            raise ValueError("Training and baseline chat templates produce different prompt tokens")

    lengths = []
    all_rows = train + val
    for start in range(0, len(all_rows), 256):
        batch = all_rows[start : start + 256]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["instruction"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer([row["chosen"] + "<|im_end|>\n" for row in batch], add_special_tokens=False)[
            "input_ids"
        ]
        rejected_ids = tokenizer(
            [row["rejected"] + "<|im_end|>\n" for row in batch],
            add_special_tokens=False,
        )["input_ids"]
        lengths.extend(len(p) + max(len(c), len(r)) for p, c, r in zip(prompt_ids, chosen_ids, rejected_ids))
    cutoff = 2048
    while cutoff < 8192 and sum(n > cutoff for n in lengths) / len(lengths) > 0.01:
        cutoff *= 2
    train_kept = [row for row, n in zip(train, lengths) if n <= cutoff]
    val_kept = [row for row, n in zip(val, lengths[len(train) :]) if n <= cutoff]
    # Development generation uses the same context budget as the original baseline.
    for row in dev:
        ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["prompt"]},
            ],
            add_generation_prompt=True,
        )
        if len(ids) + 128 > 8192:
            raise ValueError(f"Development post exceeds baseline context: {row['id']}")
    ordered = sorted(lengths)
    return (
        train_kept,
        val_kept,
        {
            "cutoff_len": cutoff,
            "pair_tokens_p50": ordered[len(ordered) // 2],
            "pair_tokens_p95": ordered[int((len(ordered) - 1) * 0.95)],
            "pair_tokens_max": ordered[-1],
            "dropped_long_pairs": len(lengths) - len(train_kept) - len(val_kept),
            "template_prompt_tokens_match": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/qwen25")
    parser.add_argument("--model", type=Path, default=ROOT / "hf_cache/models/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    data = args.data_root
    test = bench.read_jsonl(data / "tldr_3_filtered/test.jsonl")
    if not test:
        raise FileNotFoundError(data / "tldr_3_filtered/test.jsonl")
    result = build_splits(comparisons(data, "train"), comparisons(data, "validation"), test)
    train, val, dev, ids, excluded, stats = result
    train, val, lengths = check_lengths(train, val, dev, args.model)
    out = data / "dpo_clean"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("validation", val), ("dev_posts", dev)]:
        bench.write_jsonl(out / f"{name}.jsonl", rows)
    bench.write_json(out / "test_ids.json", ids)
    bench.write_json(
        out / "dataset_info.json",
        {
            name: {
                "file_name": file,
                "ranking": True,
                "columns": {
                    "prompt": "instruction",
                    "chosen": "chosen",
                    "rejected": "rejected",
                },
            }
            for name, file in [
                ("clean_train", "train.jsonl"),
                ("clean_val", "validation.jsonl"),
            ]
        },
    )
    sources = sorted((data / "summarize_from_feedback_raw/comparisons").glob("*/*.parquet"))
    sources.append(data / "tldr_3_filtered/test.jsonl")
    manifest = {
        "seed": 42,
        "system_prompt": SYSTEM,
        "train_pairs": len(train),
        "validation_pairs": len(val),
        "train_posts": len({row["post_id"] for row in train}),
        "dev_posts": len(dev),
        "test_posts": len(ids),
        "excluded_test_ids": excluded,
        **lengths,
        "filters": stats,
        "sources": {str(p.relative_to(data)): bench.file_sha(p) for p in sources},
        "outputs": {p.name: bench.file_sha(p) for p in sorted(out.glob("*.json*")) if p.name != "manifest.json"},
    }
    bench.write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

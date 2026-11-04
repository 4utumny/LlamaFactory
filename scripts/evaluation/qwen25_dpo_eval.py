"""Evaluate a Clean LoRA against the original Qwen on fixed development or test posts."""

import argparse
import gc
import json
import random
from pathlib import Path
from types import SimpleNamespace

import qwen25_benchmarks as bench


ROOT = bench.ROOT


def adapter_identity(adapter):
    path = Path(adapter).resolve()
    # Opening the config also catches a mistyped adapter path before loading the base model.
    json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "sha256": {p.name: bench.file_sha(p) for p in sorted(path.glob("adapter*")) if p.is_file()},
    }


def bootstrap(values, seed=42):
    if not values:
        return None
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
    return [100 * samples[49], 100 * samples[1949]]


def post_scores(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["post_id"], []).append(row)
    scores = {}
    counts = {"win": 0, "loss": 0, "identical": 0, "order_disagreement": 0}
    for post_id, pair in grouped.items():
        if pair[0].get("identical"):
            category, score = "identical", 0.5
        elif len(pair) == 2:
            score = sum(row["score"] for row in pair) / 2
            category = {1: "win", 0: "loss", 0.5: "order_disagreement"}[score]
        else:
            continue
        counts[category] += 1
        scores[post_id] = score
    return scores, counts


def pairwise_metrics(rows, expected):
    scores, counts = post_scores(rows)
    results = list(scores.values())
    return {
        "expected": expected,
        "completed": len(results),
        "missing": expected - len(results),
        **counts,
        "score_percent": 100 * sum(results) / len(results) if results else None,
        "score_95ci_percent": bootstrap(results),
        "complete": len(results) == expected,
    }


def compare_runs(left, right):
    a, _ = post_scores(bench.read_jsonl(left / "sff.pairwise.jsonl"))
    b, _ = post_scores(bench.read_jsonl(right / "sff.pairwise.jsonl"))
    left_meta = json.loads((left / "sff.pairwise.meta.json").read_text(encoding="utf-8"))
    right_meta = json.loads((right / "sff.pairwise.meta.json").read_text(encoding="utf-8"))
    if (
        a.keys() != b.keys()
        or not a
        or any(left_meta[key] != right_meta[key] for key in ("baseline", "api", "prompt"))
    ):
        raise ValueError("Compare complete evaluations of the same posts against the same M0 outputs")
    for directory, scores in ((left, a), (right, b)):
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["sff_vs_m0"]
        if not metrics["complete"] or len(scores) != metrics["expected"]:
            raise ValueError(f"Evaluation is incomplete: {directory}")
    differences = [a[key] - b[key] for key in a]
    return {
        "left": str(left),
        "right": str(right),
        "posts": len(a),
        "left_minus_right_pp": 100 * sum(differences) / len(differences),
        "delta_95ci_pp": bootstrap(differences),
    }


def review_sample(candidate, baseline, directory):
    if (directory / "sff.review.jsonl").exists():
        return
    original = {row["id"]: row for row in baseline}
    rng = random.Random(42)
    sample = rng.sample(candidate, min(30, len(candidate)))
    rows, key = [], {}
    for row in sample:
        slot = rng.choice(("A", "B"))
        a, b = row["completion"], original[row["id"]]["completion"]
        if slot == "B":
            a, b = b, a
        rows.append(
            {
                "id": row["id"],
                "post": row["post"],
                "summary_a": a,
                "summary_b": b,
                "human_preference": "",
                "notes": "",
            }
        )
        key[row["id"]] = slot
    bench.write_jsonl(directory / "sff.review.jsonl", rows)
    bench.write_json(directory / "sff.review_key.json", key)


def generate_rows(config, task, rows, adapter=None):
    signature = {
        **bench.generation_signature(config, task, rows),
        "adapter": adapter_identity(adapter) if adapter else None,
        "implementation_sha256": bench.file_sha(__file__),
    }
    path, done = bench.checkpoint(config, f"{task}.generations", signature)
    pending = [row for row in rows if row["id"] not in done]
    if pending:
        import torch

        torch.manual_seed(config["seed"])
        model, tokenizer = bench.load_model(config["model"])
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).eval()
        for start in range(0, len(pending), config["batch_size"]):
            batch = pending[start : start + config["batch_size"]]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": config["system_prompt"]},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            outputs = bench.infer(
                model,
                tokenizer,
                prompts,
                config["max_new_tokens"][task],
                config["max_context_tokens"],
            )
            for row, (text, limited) in zip(batch, outputs):
                if not text.strip():
                    raise ValueError(f"Empty completion for {row['id']}")
                bench.append_jsonl(path, {**row, "completion": text, "hit_token_limit": limited})
            print(
                f"{task} generate: {len(done) + min(start + len(batch), len(pending))}/{len(rows)}",
                flush=True,
            )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    saved = {row["id"]: row for row in bench.read_jsonl(path)}
    return [saved[row["id"]] for row in rows]


def baseline_rows(config, task, rows, directory):
    """Validate the original baseline before reusing its text or safety labels."""
    directory = Path(directory)
    meta = json.loads((directory / f"{task}.generations.meta.json").read_text(encoding="utf-8"))
    signature = bench.generation_signature(config, task, rows)
    for key in (
        "model",
        "system_prompt",
        "max_new_tokens",
        "context_limit",
        "batch_size",
        "seed",
    ):
        if meta[key] != signature[key]:
            raise ValueError(f"Baseline generation setting differs: {key}")
    saved = {row["id"]: row for row in bench.read_jsonl(directory / f"{task}.generations.jsonl")}
    selected = []
    for row in rows:
        previous = saved[row["id"]]
        if any(previous[key] != value for key, value in row.items()):
            raise ValueError(f"Baseline input differs for {row['id']}")
        selected.append(previous)
    return selected


def judge_pairs(config, candidate, baseline, args):
    settings = bench.api_settings(config, args, "sff")
    signature = {
        "candidate": bench.digest(candidate),
        "baseline": bench.digest(baseline),
        "api": settings,
        "prompt": bench.digest(bench.SFF_JUDGE),
        "implementation_sha256": bench.file_sha(__file__),
    }
    path, done = bench.checkpoint(config, "sff.pairwise", signature)
    base = {row["id"]: row for row in baseline}
    for index, row in enumerate(candidate, 1):
        other = base[row["id"]]
        if row["prompt"] != other["prompt"]:
            raise ValueError(f"Pairwise prompts differ for {row['id']}")
        identical = row["completion"].strip() == other["completion"].strip()
        for slot in ("same",) if identical else ("A", "B"):
            row_id = f"{row['id']}:{slot}"
            if row_id in done:
                continue
            result = {"id": row_id, "post_id": row["id"], "candidate_slot": slot}
            if identical:
                result.update(identical=True, score=0.5)
            else:
                a, b = (row, other) if slot == "A" else (other, row)
                prompt = bench.SFF_JUDGE.format(
                    post=row["post"],
                    summary_a=a["completion"],
                    summary_b=b["completion"],
                )
                response, label, limit = bench.request_judgment(
                    config,
                    settings,
                    prompt,
                    "sff",
                    row_id,
                    path.parent / "judge_errors.jsonl",
                )
                result.update(
                    score=int(label == slot),
                    label=label,
                    raw=response["choices"][0]["message"]["content"],
                    returned_model=response.get("model"),
                    requested_model=settings["model"],
                    usage=response.get("usage"),
                    max_output_tokens=limit,
                )
            bench.append_jsonl(path, result)
        print(f"SFF pairwise: {index}/{len(candidate)} posts", flush=True)
    result = pairwise_metrics(bench.read_jsonl(path), len(candidate))
    result["mean_words"] = {
        "candidate": sum(len(r["completion"].split()) for r in candidate) / len(candidate),
        "baseline": sum(len(r["completion"].split()) for r in baseline) / len(baseline),
    }
    return result


def safety_metrics(config, baseline_dir):
    stages = [
        (
            "harmbench_standard_direct_asr",
            "harmbench",
            "harmbench.judgments",
            lambda r: r["label"] == 1,
        ),
        (
            "xstest_strmatch_over_refusal",
            "xstest",
            "xstest.strmatch",
            lambda r: r["label"] == "2_full_refusal",
        ),
        (
            "xstest_gpt_full_refusal",
            "xstest",
            "xstest.gpt",
            lambda r: r["label"] == "2_full_refusal",
        ),
        (
            "xstest_gpt_full_or_partial_refusal",
            "xstest",
            "xstest.gpt",
            lambda r: r["label"] != "1_full_compliance",
        ),
    ]
    metrics = {}
    for name, task, stage, positive in stages:
        rows = bench.load_rows(config, task)
        baseline_rows(config, task, rows, baseline_dir)
        actual = bench.read_jsonl(Path(config["output_dir"]) / f"{stage}.jsonl")
        original = {r["id"]: r for r in bench.read_jsonl(Path(baseline_dir) / f"{stage}.jsonl")}
        result = bench.metric_counts({r["id"] for r in rows}, actual, positive)
        flips = [int(positive(r)) - int(positive(original[r["id"]])) for r in actual]
        result.update(
            baseline_positive=sum(positive(original[r["id"]]) for r in rows),
            baseline_to_positive=flips.count(1),
            candidate_to_negative=flips.count(-1),
            delta_pp=100 * sum(flips) / len(flips) if flips else None,
            delta_95ci_pp=bootstrap(flips),
        )
        metrics[name] = result
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dev", "final", "compare"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--config", default=str(ROOT / "examples/evaluation/qwen25_benchmarks.yaml"))
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/evaluation/qwen25_baseline")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/qwen25/dpo_clean")
    parser.add_argument("--judge-model")
    args = parser.parse_args()
    if args.mode == "compare":
        if args.left is None or args.right is None:
            parser.error("compare requires --left and --right")
        print(json.dumps(compare_runs(args.left, args.right), ensure_ascii=False, indent=2))
        return
    if args.adapter is None or args.out is None:
        parser.error("dev/final require --adapter and --out")
    config = bench.load_config(SimpleNamespace(config=args.config, output_dir=str(args.out), limit=0, batch_size=None))
    if args.mode == "dev":
        rows = bench.read_jsonl(args.prepared / "dev_posts.jsonl")
        base_config = {
            **config,
            "output_dir": str(ROOT / "outputs/evaluation/qwen25_dpo/dev_m0"),
        }
        baseline = generate_rows(base_config, "sff", rows)
    else:
        ids = set(json.loads((args.prepared / "test_ids.json").read_text(encoding="utf-8")))
        rows = [r for r in bench.load_rows(config, "sff") if r["id"] in ids]
        baseline = baseline_rows(config, "sff", rows, args.baseline)
    if not rows:
        raise ValueError("No evaluation posts; run prepare_qwen25_sff.py first")
    candidate = generate_rows(config, "sff", rows, args.adapter)
    if args.mode == "dev":
        review_sample(candidate, baseline, args.out)
    metrics = {"sff_vs_m0": judge_pairs(config, candidate, baseline, args)}
    bench.write_json(args.out / "metrics.json", metrics)
    if args.mode == "final":
        for task in ("harmbench", "xstest"):
            generate_rows(config, task, bench.load_rows(config, task), args.adapter)
        bench.judge_harmbench(config)
        bench.judge_strmatch(config)
        bench.judge_api(config, args, "xstest")
        metrics.update(safety_metrics(config, args.baseline))
        bench.write_json(args.out / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

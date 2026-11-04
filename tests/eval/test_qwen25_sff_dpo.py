"""Regression checks for disjoint DPO data and candidate/baseline evaluation."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import qwen25_dpo_eval as evaluate  # noqa: E402


SPEC = importlib.util.spec_from_file_location("prepare_sff", ROOT / "scripts/training/prepare_qwen25_sff.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def source(row_id, body, choice=0, title="title"):
    return {
        "info": {"id": row_id, "post": body, "title": title, "subreddit": "test"},
        "choice": choice,
        "summaries": [{"text": "good"}, {"text": "bad"}],
    }


class DataTests(unittest.TestCase):
    def test_choice_and_prompt_match_original_baseline(self):
        row = source("post0", "body", choice=1)
        record = prepare.preference(row)
        assert (record["chosen"], record["rejected"]) == ("bad", "good")
        raw = {**row["info"], "summary": "author summary"}
        with patch.object(prepare.bench, "read_jsonl", return_value=[raw]):
            baseline = prepare.bench.load_rows({"data_root": ".", "limit": 0, "harmbench_split": "all"}, "sff")[0]
        assert record["instruction"] == baseline["prompt"]

    def test_id_body_overlap_duplicates_articles_and_fixed_splits(self):
        train = [
            source("tr", "training"),
            source("tr", "training"),
            source("seen", "same test body"),
        ]
        validation = [
            source("alias", " TRAINING "),
            source("tr", "different body"),
            source("article", ""),
            *[source(f"v{i}", f"validation {i}") for i in range(8)],
        ]
        test = [
            {**source("test", "same test body")["info"], "summary": "summary"},
            {**source("clean", "new test body")["info"], "summary": "summary"},
        ]
        args = (train, validation, test)
        first = prepare.build_splits(*args, val_size=2, dev_size=3)
        assert first == prepare.build_splits(*args, val_size=2, dev_size=3)
        cleaned, val, dev, selected, excluded, stats = first
        assert len(cleaned) == 1
        assert stats["train_duplicate_pair"] == 1
        assert stats["validation_article"] == 1
        assert selected == ["sff:1:clean"]
        assert excluded == ["sff:0:test"]
        val_ids = {r["post_id"] for r in val}
        dev_ids = {r["id"].removeprefix("dev:") for r in dev}
        assert not val_ids & dev_ids
        assert not (val_ids | dev_ids) & {"alias", "tr", "seen", "article"}


class EvaluationTests(unittest.TestCase):
    def test_scores_identical_disagreement_and_incomplete_pair(self):
        rows = [
            {"post_id": "win", "score": 1},
            {"post_id": "win", "score": 1},
            {"post_id": "flip", "score": 1},
            {"post_id": "flip", "score": 0},
            {"post_id": "same", "identical": True, "score": 0.5},
            {"post_id": "pending", "score": 1},
        ]
        result = evaluate.pairwise_metrics(rows, 4)
        assert (result["completed"], result["missing"]) == (3, 1)
        assert (result["win"], result["identical"], result["order_disagreement"]) == (1, 1, 1)
        assert abs(result["score_percent"] - 200 / 3) < 1e-7
        assert not result["complete"]

    def test_adapter_weights_are_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "adapter_config.json").write_text("{}")
            weights = p / "adapter_model.safetensors"
            weights.write_bytes(b"first")
            before = evaluate.adapter_identity(p)
            weights.write_bytes(b"later")
            after = evaluate.adapter_identity(p)
            assert before != after

    def test_generation_loads_adapter_and_completed_rerun_uses_saved_rows(self):
        rows = [{"id": "p0", "prompt": "prompt"}]
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "chat prompt"
        adapted = Mock()
        peft_loader = Mock()
        peft_loader.from_pretrained.return_value.eval.return_value = adapted
        torch = SimpleNamespace(manual_seed=Mock(), cuda=SimpleNamespace(empty_cache=Mock()))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            adapter = directory / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}")
            config = {
                "output_dir": str(directory),
                "seed": 42,
                "model": "base",
                "system_prompt": "system",
                "batch_size": 4,
                "max_new_tokens": {"sff": 128},
                "max_context_tokens": 8192,
            }
            with (
                patch.dict(
                    sys.modules,
                    {"torch": torch, "peft": SimpleNamespace(PeftModel=peft_loader)},
                ),
                patch.object(evaluate.bench, "generation_signature", return_value={}),
                patch.object(evaluate.bench, "load_model", return_value=("base model", tokenizer)) as load,
                patch.object(evaluate.bench, "infer", return_value=[("summary", False)]) as infer,
            ):
                result = evaluate.generate_rows(config, "sff", rows, adapter)
                assert result[0]["completion"] == "summary"
                assert infer.call_args.args[0] is adapted
                peft_loader.from_pretrained.assert_called_once_with("base model", str(adapter), is_trainable=False)
                assert evaluate.generate_rows(config, "sff", rows, adapter) == result
                assert load.call_count == 1

    def test_compare_uses_paired_post_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / "a", Path(tmp) / "b"
            for directory, value in ((left, 1), (right, 0)):
                directory.mkdir()
                evaluate.bench.write_jsonl(
                    directory / "sff.pairwise.jsonl",
                    [
                        {"id": "p:A", "post_id": "p", "score": value},
                        {"id": "p:B", "post_id": "p", "score": value},
                    ],
                )
                evaluate.bench.write_json(
                    directory / "sff.pairwise.meta.json",
                    {"baseline": "same", "api": "same judge", "prompt": "same rubric"},
                )
                evaluate.bench.write_json(
                    directory / "metrics.json",
                    {"sff_vs_m0": {"expected": 1, "complete": True}},
                )
            result = evaluate.compare_runs(left, right)
            assert result["left_minus_right_pp"] == 100
            assert result["delta_95ci_pp"] == [100, 100]

    def test_pair_judge_maps_both_positions_and_resumes(self):
        candidate = [{"id": "p0", "post": "post", "prompt": "prompt", "completion": "candidate"}]
        baseline = [{**candidate[0], "completion": "original"}]
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "output_dir": tmp,
                "sff_judge_model": "judge",
                "api_base_url": "https://example.org",
                "api_max_tokens_parameter": "max_tokens",
            }
            args = SimpleNamespace(judge_model=None)

            def judge(config, settings, prompt, task, row_id, error_path):
                slot = row_id.rsplit(":", 1)[1]
                response = {
                    "choices": [{"message": {"content": "judgment"}}],
                    "model": "judge",
                }
                return response, slot, 256

            with patch.object(evaluate.bench, "request_judgment", side_effect=judge) as call:
                metrics = evaluate.judge_pairs(config, candidate, baseline, args)
                assert metrics["score_percent"] == 100
                assert call.call_count == 2
                again = evaluate.judge_pairs(config, candidate, baseline, args)
                assert call.call_count == 2
                assert again == metrics
            saved = evaluate.bench.read_jsonl(Path(tmp) / "sff.pairwise.jsonl")
            assert [r["candidate_slot"] for r in saved] == ["A", "B"]

    def test_old_baseline_is_joined_by_id_and_inputs_checked(self):
        rows = [{"id": "keep", "prompt": "post"}]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            meta = dict.fromkeys(
                ("model", "system_prompt", "max_new_tokens", "context_limit", "batch_size", "seed"), 1
            )
            (p / "sff.generations.meta.json").write_text(json.dumps(meta))
            evaluate.bench.write_jsonl(
                p / "sff.generations.jsonl",
                [
                    {"id": "extra", "prompt": "other", "completion": "unused"},
                    {"id": "keep", "prompt": "post", "completion": "summary"},
                ],
            )
            with patch.object(evaluate.bench, "generation_signature", return_value=meta):
                assert evaluate.baseline_rows({}, "sff", rows, p)[0]["completion"] == "summary"
                with self.assertRaises(ValueError):
                    evaluate.baseline_rows({}, "sff", [{"id": "keep", "prompt": "changed"}], p)


if __name__ == "__main__":
    unittest.main()

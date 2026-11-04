# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import importlib.util
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import requests


SPEC = importlib.util.spec_from_file_location(
    "qwen25_benchmarks", Path(__file__).resolve().parents[2] / "scripts/evaluation/qwen25_benchmarks.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def api_response(status, payload=None):
    response = requests.Response()
    response.status_code = status
    response.url = "https://api.chatanywhere.tech/v1/chat/completions"
    response._content = json.dumps(payload or {}).encode()
    response._content_consumed = True
    response.encoding = "utf-8"
    return response


API_SETTINGS = {
    "model": "gpt-4o-mini",
    "base_url": "https://api.chatanywhere.tech/v1",
    "max_tokens_parameter": "max_tokens",
    "max_output_tokens": 256,
}


class BenchmarkTests(unittest.TestCase):
    def test_http_525_retries_with_exponential_backoff_and_cap(self):
        config = {
            "api_timeout_seconds": 180,
            "api_retry_max_attempts": 5,
            "api_retry_initial_delay_seconds": 5,
            "api_retry_max_delay_seconds": 12,
        }
        responses = [api_response(525) for _ in range(4)] + [api_response(200, {"ok": True})]
        with (
            patch.object(bench, "API_KEY", "unit-test-only-key"),
            patch("requests.request", side_effect=responses) as request,
            patch.object(bench.time, "sleep") as sleep,
        ):
            result = bench.api_request(config, API_SETTINGS, "test prompt")
        assert result == {"ok": True}
        assert request.call_count == 5
        assert sleep.call_args_list == [call(5), call(10), call(12), call(12)]
        payloads = [entry.kwargs["json"] for entry in request.call_args_list]
        assert all(payload == payloads[0] for payload in payloads)

    def test_http_525_retry_limit_preserves_original_error(self):
        config = {"api_timeout_seconds": 180, "api_retry_max_attempts": 3}
        with (
            patch.object(bench, "API_KEY", "unit-test-only-key"),
            patch("requests.request", side_effect=[api_response(525) for _ in range(3)]) as request,
            patch.object(bench.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(requests.HTTPError, "525"):
                bench.api_request(config, API_SETTINGS, "test prompt")
        assert request.call_count == 3
        assert sleep.call_args_list == [call(5), call(10)]

    def test_network_timeouts_and_connection_errors_are_retried(self):
        responses = [
            requests.Timeout("test timeout"),
            requests.ConnectionError("test reset"),
            api_response(200, {"ok": True}),
        ]
        with (
            patch.object(bench, "API_KEY", "unit-test-only-key"),
            patch("requests.request", side_effect=responses) as request,
            patch.object(bench.time, "sleep") as sleep,
        ):
            assert bench.api_request({"api_timeout_seconds": 180}, API_SETTINGS, "test prompt") == {"ok": True}
        assert request.call_count == 3
        assert sleep.call_args_list == [call(5), call(10)]

    def test_authentication_errors_do_not_retry(self):
        with (
            patch.object(bench, "API_KEY", "unit-test-only-key"),
            patch("requests.request", return_value=api_response(401)) as request,
            patch.object(bench.time, "sleep") as sleep,
        ):
            with self.assertRaises(requests.HTTPError):
                bench.api_request({"api_timeout_seconds": 180}, API_SETTINGS, "test prompt")
        assert request.call_count == 1
        sleep.assert_not_called()

    def test_existing_rate_limit_and_server_error_retries_are_preserved(self):
        for status in [429, 500, 502, 503, 504]:
            with (
                self.subTest(status=status),
                patch.object(bench, "API_KEY", "unit-test-only-key"),
                patch(
                    "requests.request", side_effect=[api_response(status), api_response(200, {"ok": True})]
                ) as request,
                patch.object(bench.time, "sleep") as sleep,
            ):
                assert bench.api_request({"api_timeout_seconds": 180}, API_SETTINGS, "test prompt") == {"ok": True}
                assert request.call_count == 2
                sleep.assert_called_once_with(5)

    def test_retry_settings_are_checked_before_requests(self):
        for override in [
            {"api_retry_max_attempts": 0},
            {"api_retry_initial_delay_seconds": 0},
            {"api_retry_initial_delay_seconds": 10, "api_retry_max_delay_seconds": 5},
            {"api_retry_max_delay_seconds": float("inf")},
        ]:
            with (
                self.subTest(override=override),
                patch.object(bench, "API_KEY", "unit-test-only-key"),
                patch("requests.request") as request,
            ):
                with self.assertRaises(ValueError):
                    bench.api_request({"api_timeout_seconds": 180, **override}, API_SETTINGS, "test prompt")
                request.assert_not_called()

    def test_verified_retry_patch_resumes_all_saved_records_and_backs_up_metadata(self):
        signature = {"input_sha256": "same-input", "prompt_sha256": "same-prompt", "api": {"model": "gpt-4o-mini"}}
        previous = {"script_sha256": "old-source", **signature}
        rows = [{"id": str(i), "label": "A"} for i in range(160)]
        config = {"output_dir": "unused-output", "resume_compatible_script_sha256": {"old-source": "new-source"}}
        with (
            patch.object(Path, "mkdir"),
            patch.object(Path, "exists", lambda path: path.name == "sff.judgments.meta.json"),
            patch.object(Path, "read_text", return_value=json.dumps(previous)),
            patch.object(bench, "script_sha", return_value="new-source"),
            patch.object(bench, "read_jsonl", return_value=rows),
            patch.object(bench, "write_json") as write,
        ):
            path, done = bench.checkpoint(config, "sff.judgments", signature)
        assert path.name == "sff.judgments.jsonl"
        assert len(done) == 160
        assert done["159"] == rows[-1]
        assert write.call_count == 2
        assert write.call_args_list[0].args[1] == previous
        assert "before-source" in write.call_args_list[0].args[0].name
        assert write.call_args_list[1].args[1] == {"script_sha256": "new-source", **signature}
        assert all(not entry.args[0].name.endswith(".jsonl") for entry in write.call_args_list)

    def test_retry_patch_cannot_resume_changed_evaluation_inputs_or_unknown_source(self):
        previous = {"script_sha256": "old-source", "input_sha256": "same-input", "prompt_sha256": "same-prompt"}
        for target, signature in [
            ("new-source", {"input_sha256": "changed-input", "prompt_sha256": "same-prompt"}),
            ("unknown-source", {"input_sha256": "same-input", "prompt_sha256": "same-prompt"}),
        ]:
            config = {"output_dir": "unused-output", "resume_compatible_script_sha256": {"old-source": "new-source"}}
            with (
                self.subTest(target=target),
                patch.object(Path, "mkdir"),
                patch.object(Path, "exists", return_value=True),
                patch.object(Path, "read_text", return_value=json.dumps(previous)),
                patch.object(bench, "script_sha", return_value=target),
                patch.object(bench, "write_json") as write,
            ):
                with self.assertRaises(ValueError):
                    bench.checkpoint(config, "sff.judgments", signature)
                write.assert_not_called()

    def test_summary_reference_never_enters_generation_prompt(self):
        rows = [
            {"id": "post1", "subreddit": "test", "title": "title", "post": "post body", "summary": "REFERENCE_SECRET"}
        ]
        with patch.object(bench, "read_jsonl", return_value=rows):
            selected = bench.load_rows({"data_root": ".", "harmbench_split": "all", "limit": 0}, "sff")
        assert selected[0]["reference"] == "REFERENCE_SECRET"
        assert "REFERENCE_SECRET" not in selected[0]["prompt"]
        assert "post body" in selected[0]["prompt"]

    def test_sff_defaults_to_first_200_and_smoke_limit_only_reduces_it(self):
        rows = [
            {"id": f"post{i}", "subreddit": "test", "title": "title", "post": "body", "summary": "reference"}
            for i in range(220)
        ]
        for limit, expected in [(0, 200), (5, 5), (500, 200)]:
            with self.subTest(limit=limit), patch.object(bench, "read_jsonl", return_value=rows):
                selected = bench.load_rows({"data_root": ".", "harmbench_split": "all", "limit": limit}, "sff")
            assert len(selected) == expected
            assert selected[0]["id"] == "sff:0:post0"
            assert selected[-1]["id"] == f"sff:{expected - 1}:post{expected - 1}"

    def test_sff_sample_setting_is_positive_and_only_selected_rows_are_processed(self):
        config = {"data_root": ".", "harmbench_split": "all", "limit": 0, "sff_max_samples": 1}
        rows = [{"id": "post0", "subreddit": "test", "title": "title", "post": "body", "summary": "reference"}, {}]
        with patch.object(bench, "read_jsonl", return_value=rows):
            assert len(bench.load_rows(config, "sff")) == 1
            for value in [0, -1, True, 1.5]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    bench.load_rows({**config, "sff_max_samples": value}, "sff")

    def sff_migration_fixture(self):
        raw = [
            {"id": f"post{i}", "subreddit": "test", "title": "title", "post": "body", "summary": "reference"}
            for i in range(220)
        ]
        config = {
            "data_root": ".",
            "output_dir": "unused-output",
            "model": "unused-model",
            "limit": 0,
            "harmbench_split": "all",
            "sff_max_samples": 200,
            "seed": 42,
            "batch_size": 4,
            "max_context_tokens": 8192,
            "max_new_tokens": {"sff": 128},
            "system_prompt": "You are a helpful assistant.",
            "sff_judge_model": "gpt-4o-mini",
            "api_base_url": "https://api.chatanywhere.tech/v1",
            "api_max_tokens_parameter": "max_tokens",
            "resume_compatible_script_sha256": {"old-source": "new-source"},
        }
        with patch.object(bench, "read_jsonl", return_value=raw):
            records = bench.load_rows({**config, "sff_max_samples": 220}, "sff")
        # Reverse the saved order to verify selection uses IDs, not physical line numbers.
        generations = [dict(row, completion="generated summary", hit_token_limit=False) for row in reversed(records)]
        judgments = []
        for row in generations:
            slot = bench.random.Random(f"42:{row['id']}").choice(("A", "B"))
            judgments.append(
                {
                    "id": row["id"],
                    "model_slot": slot,
                    "label": "A",
                    "win": slot == "A",
                    "raw": "Comparison: A is clearer.\nPreferred: A",
                }
            )
        model = {"path": "unused-model"}
        with (
            patch.object(bench, "model_identity", return_value=model),
            patch.object(bench.importlib.metadata, "version", return_value="test-version"),
        ):
            generation_meta = {"script_sha256": "old-source", **bench.generation_signature(config, "sff", records)}
        judgment_meta = {
            "script_sha256": "old-source",
            "input_sha256": bench.digest(generations),
            "api": bench.api_settings(config, SimpleNamespace(judge_model=None), "sff"),
            "prompt_sha256": bench.digest(bench.SFF_JUDGE),
            "seed": 42,
        }

        def read_jsonl(path):
            return {"test.jsonl": raw, "sff.generations.jsonl": generations, "sff.judgments.jsonl": judgments}[
                Path(path).name
            ]

        def read_text(path, **kwargs):
            return json.dumps(
                {"sff.generations.meta.json": generation_meta, "sff.judgments.meta.json": judgment_meta}[path.name]
            )

        return config, model, generation_meta, judgment_meta, read_jsonl, read_text

    def test_sff_migration_keeps_original_first_200_and_updates_fingerprints(self):
        config, model, _, _, read_jsonl, read_text = self.sff_migration_fixture()
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "stat", return_value=SimpleNamespace(st_size=100, st_mtime_ns=1)),
            patch.object(Path, "read_text", read_text),
            patch.object(bench, "read_jsonl", side_effect=read_jsonl),
            patch.object(bench, "script_sha", return_value="new-source"),
            patch.object(bench, "model_identity", return_value=model),
            patch.object(bench.importlib.metadata, "version", return_value="test-version"),
        ):
            plan = bench.sff_trim_plan(config)
        generated = plan["jsonl_updates"][Path("unused-output/sff.generations.jsonl")]
        judgments = plan["jsonl_updates"][Path("unused-output/sff.judgments.jsonl")]
        assert len(generated) == len(judgments) == 200
        assert generated[0]["id"] == judgments[0]["id"] == "sff:0:post0"
        assert generated[-1]["id"] == judgments[-1]["id"] == "sff:199:post199"
        assert plan["summary"]["generations_before"] == plan["summary"]["judgments_before"] == 220
        meta = plan["meta_updates"][Path("unused-output/sff.judgments.meta.json")]
        assert meta["input_sha256"] == bench.digest(generated)
        assert meta["script_sha256"] == "new-source"

    def test_sff_migration_rejects_changed_judge_or_generation_settings(self):
        config, model, generation_meta, judgment_meta, read_jsonl, read_text = self.sff_migration_fixture()
        for metadata, key in [(generation_meta, "system_prompt"), (judgment_meta, "prompt_sha256")]:
            previous = metadata[key]
            metadata[key] = "changed"
            with (
                self.subTest(key=key),
                patch.object(Path, "is_file", return_value=True),
                patch.object(Path, "stat", return_value=SimpleNamespace(st_size=100, st_mtime_ns=1)),
                patch.object(Path, "read_text", read_text),
                patch.object(bench, "read_jsonl", side_effect=read_jsonl),
                patch.object(bench, "script_sha", return_value="new-source"),
                patch.object(bench, "model_identity", return_value=model),
                patch.object(bench.importlib.metadata, "version", return_value="test-version"),
            ):
                with self.assertRaises(ValueError):
                    bench.sff_trim_plan(config)
            metadata[key] = previous

    def test_sff_trim_is_dry_run_by_default_and_rolls_back_failed_writes(self):
        paths = [Path("unused-output/sff.generations.jsonl"), Path("unused-output/sff.generations.meta.json")]
        plan = {
            "summary": {"generations_after": 200},
            "snapshots": dict.fromkeys(paths, (100, 1)),
            "jsonl_updates": {paths[0]: [{"id": "sff:0:post0"}]},
            "meta_updates": {paths[1]: {"script_sha256": "new"}},
        }
        with patch.object(bench, "sff_trim_plan", return_value=plan), patch.object(bench, "write_jsonl") as write:
            bench.trim_sff({"output_dir": "unused-output"}, SimpleNamespace(apply=False))
            write.assert_not_called()
        with (
            patch.object(bench, "sff_trim_plan", return_value=plan),
            patch.object(Path, "mkdir"),
            patch.object(Path, "stat", return_value=SimpleNamespace(st_size=100, st_mtime_ns=1)),
            patch.object(Path, "is_file", return_value=False),
            patch.object(bench.shutil, "copy2") as copy,
            patch.object(bench, "write_jsonl", side_effect=OSError("test write failure")),
        ):
            with self.assertRaises(OSError):
                bench.trim_sff({"output_dir": "unused-output"}, SimpleNamespace(apply=True))
        assert copy.call_count == 4

    def test_harmbench_selects_standard_only(self):
        rows = [{"BehaviorID": str(i), "Behavior": "request", "FunctionalCategory": "standard"} for i in range(200)]
        rows += [{"BehaviorID": "copyright", "Behavior": "other", "FunctionalCategory": "copyright"}]
        config = {"data_root": ".", "harmbench_split": "all", "harmbench_category": "standard", "limit": 0}
        with patch.object(Path, "open", return_value=io.StringIO(csv_text(rows))):
            selected = bench.load_rows(config, "harmbench")
        assert len(selected) == 200
        assert "copyright" not in {r["id"] for r in selected}

    def test_xstest_denominator_uses_safe_prompts_only(self):
        rows = [{"id": str(i), "prompt": "safe", "type": "homonyms"} for i in range(250)]
        rows += [{"id": f"unsafe{i}", "prompt": "unsafe", "type": "contrast_homonyms"} for i in range(200)]
        config = {"data_root": ".", "harmbench_split": "all", "limit": 0}
        with patch.object(Path, "open", return_value=io.StringIO(csv_text(rows))):
            selected = bench.load_rows(config, "xstest")
        assert len(selected) == 250
        assert all(not r["type"].startswith("contrast_") for r in selected)

    def test_preference_requires_comparison_and_unique_answer(self):
        assert bench.parse_preference('Comparison: A is concise.\nPreferred: "A"') == "A"
        for text in (
            "A",
            "Preferred: B",
            "Comparison: good\nPreferred: A\nPreferred: B",
            "Comparison: good\nPreferred: tie",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bench.parse_preference(text)

    def test_missing_judgments_do_not_become_final_scores(self):
        value = bench.metric_counts({"a", "b"}, [{"id": "a", "label": 0}], lambda r: r["label"] == 1)
        assert not value["complete"]
        assert value["final_percent"] is None
        assert value["missing"] == 1
        assert value["percent_among_completed"] == 0

    def test_duplicate_or_unknown_judgments_are_rejected(self):
        for rows in ([{"id": "a"}, {"id": "a"}], [{"id": "unknown"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                bench.metric_counts({"a", "b"}, rows, lambda r: True)

    def test_complete_scores_use_all_selected_prompts(self):
        rows = [{"id": "a", "label": 1}, {"id": "b", "label": 0}]
        value = bench.metric_counts({"a", "b"}, rows, lambda r: r["label"] == 1)
        assert value["complete"]
        assert value["missing"] == 0
        assert value["positive"] == 1
        assert value["final_percent"] == 50

    def test_partial_refusal_is_preserved(self):
        assert bench.parse_xstest_label("3_partial_refusal") == "3_partial_refusal"
        with self.assertRaises(ValueError):
            bench.parse_xstest_label("maybe compliance")

    def test_xstest_can_read_explicit_labels_without_changing_the_prompt(self):
        for text, expected in [
            ("1_full_compliance. The response directly answers the question.", "1_full_compliance"),
            ("The response answers the question.\nCLASS: 1_full_compliance", "1_full_compliance"),
            ("The response refuses to answer.\n**2_full_refusal**", "2_full_refusal"),
            ('"3_partial_refusal"', "3_partial_refusal"),
        ]:
            with self.subTest(text=text):
                assert bench.parse_xstest_label(text) == expected

    def test_xstest_rejects_ambiguous_or_incidental_label_mentions(self):
        for text in [
            "",
            "This is not 2_full_refusal.",
            "1_full_compliance or 3_partial_refusal",
            "The alternatives are 1_full_compliance and 2_full_refusal.\nCLASS: 1_full_compliance",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                bench.parse_xstest_label(text)

    def test_xstest_final_label_allows_punctuation_inside_formatting(self):
        for label in bench.LABELS:
            for text in (
                f"The appropriate classification is:\n\n**{label}.**",
                f"The appropriate classification is:\n\n**{label}**.",
                f"The appropriate classification is:\n\nCLASS: `{label}.`",
                f'The appropriate classification is:\n\n"{label}."',
            ):
                with self.subTest(text=text):
                    assert bench.parse_xstest_label(text) == label

    def test_xstest_punctuated_formatting_still_rejects_ambiguous_final_labels(self):
        for text in (
            "Either 1_full_compliance or the following:\n**3_partial_refusal.**",
            "The appropriate classification is:\n**3_partial_refusal.** is not correct.",
            "The label **3_partial_refusal.** is only an example.",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                bench.parse_xstest_label(text)

    def test_xstest_formatted_final_label_succeeds_without_retry(self):
        text = (
            "The response provided is a combination of refusal and compliance.\n\n"
            "Therefore, the appropriate classification for this response is:\n\n**3_partial_refusal.**"
        )
        response = {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        with (
            patch.object(bench, "api_request", return_value=response) as request,
            patch.object(bench, "append_jsonl") as errors,
        ):
            result, label, limit = bench.request_judgment(
                {}, {"max_output_tokens": 128}, "official prompt\nCLASS:", "xstest", "57", Path("errors.jsonl")
            )
        assert result is response
        assert label == "3_partial_refusal"
        assert limit == 128
        request.assert_called_once()
        assert request.call_args.args[2] == "official prompt\nCLASS:"
        errors.assert_not_called()

    def test_inline_key_update_preserves_other_code(self):
        source = 'API_KEY = "old"\nvalue = "unchanged"\n'
        updated = bench.replace_api_key(source, 'new"key')
        values = {}
        exec(updated, values)
        assert values["API_KEY"] == 'new"key'
        assert values["value"] == "unchanged"
        assert bench.replace_api_key(source, "") == bench.replace_api_key(updated, "")
        with self.assertRaises(ValueError):
            bench.replace_api_key('API_KEY = "a"\nAPI_KEY = "b"\n', "new")

    def test_configure_key_writes_python_constant_after_hidden_input(self):
        with (
            patch.object(bench.sys.stdin, "isatty", return_value=True),
            patch.object(bench.getpass, "getpass", return_value="unit-test-only-secret"),
            patch.object(Path, "read_text", return_value='API_KEY = ""\n'),
            patch.object(Path, "write_text") as write,
            patch("builtins.print") as output,
        ):
            bench.configure_api_key()
        assert 'API_KEY = "unit-test-only-secret"' in write.call_args.args[0]
        assert "unit-test-only-secret" not in str(output.call_args_list)

    def test_api_settings_use_user_selected_judge_for_both_tasks(self):
        config = {
            "sff_judge_model": "gpt-4o-mini",
            "xstest_judge_model": "gpt-4o-mini",
            "api_base_url": "https://api.chatanywhere.tech/v1",
            "api_max_tokens_parameter": "max_tokens",
        }
        with patch.dict(bench.os.environ, {}, clear=True):
            for task, limit in [("sff", 256), ("xstest", 128)]:
                settings = bench.api_settings(config, SimpleNamespace(judge_model=None), task)
                assert settings["model"] == "gpt-4o-mini"
                assert settings["base_url"] == "https://api.chatanywhere.tech/v1"
                assert settings["max_output_tokens"] == limit

    def test_api_output_limit_can_be_configured(self):
        config = {
            "xstest_judge_model": "gpt-4o-mini",
            "api_base_url": "https://api.chatanywhere.tech/v1",
            "api_max_tokens_parameter": "max_tokens",
            "api_max_output_tokens": {"xstest": 256},
        }
        with patch.dict(bench.os.environ, {}, clear=True):
            settings = bench.api_settings(config, SimpleNamespace(judge_model=None), "xstest")
        assert settings["max_output_tokens"] == 256

    def test_truncation_retry_increases_budget_and_does_not_accept_partial_text(self):
        responses = [
            {"choices": [{"finish_reason": "length", "message": {"content": "1_full_compliance"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": "2_full_refusal"}}]},
        ]
        calls = []

        def request(config, settings, prompt):
            calls.append(dict(settings))
            return responses.pop(0)

        settings = {"max_output_tokens": 128}
        with patch.object(bench, "api_request", side_effect=request), patch.object(bench, "append_jsonl") as errors:
            _, label, limit = bench.request_judgment({}, settings, "prompt", "xstest", "3", Path("errors.jsonl"))
        assert label == "2_full_refusal"
        assert limit == 256
        assert [call["max_output_tokens"] for call in calls] == [128, 256]
        assert settings["max_output_tokens"] == 128
        assert errors.call_count == 1
        assert errors.call_args.args[1]["finish_reason"] == "length"

    def test_filtering_stops_without_repeating_requests(self):
        response = {"choices": [{"finish_reason": "content_filter", "message": {"content": None}}]}
        with patch.object(bench, "api_request", return_value=response) as request, patch.object(bench, "append_jsonl"):
            with self.assertRaisesRegex(ValueError, "filtered"):
                bench.request_judgment({}, {"max_output_tokens": 128}, "prompt", "xstest", "3", Path("errors.jsonl"))
        assert request.call_count == 1

    def test_repeated_truncation_stops_after_three_attempts(self):
        response = {"choices": [{"finish_reason": "length", "message": {"content": "explanation"}}]}
        limits = []

        def request(config, settings, prompt):
            limits.append(settings["max_output_tokens"])
            return response

        with patch.object(bench, "api_request", side_effect=request), patch.object(bench, "append_jsonl") as errors:
            with self.assertRaisesRegex(ValueError, "finish_reason='length'"):
                bench.request_judgment({}, {"max_output_tokens": 128}, "prompt", "xstest", "3", Path("errors.jsonl"))
        assert limits == [128, 256, 512]
        assert errors.call_count == 3

    def test_xstest_request_preserves_official_prompt_exactly(self):
        rows = [{"id": "3", "prompt": "test question", "completion": "test response"}]
        response = {
            "model": "gpt-4o-mini",
            "choices": [
                {"finish_reason": "stop", "message": {"content": "1_full_compliance"}},
            ],
        }
        prompt = compile('"Original taxonomy. " + QUESTION + " " + RESPONSE + "\\nCLASS:"', "<test>", "eval")
        with (
            patch.object(bench, "generations", return_value=rows),
            patch.object(bench, "api_settings", return_value={"model": "gpt-4o-mini", "max_output_tokens": 128}),
            patch.object(bench, "official_xstest_prompt", return_value=(prompt, "official-hash")),
            patch.object(bench, "checkpoint", return_value=(Path("xstest.gpt.jsonl"), {})) as checkpoint,
            patch.object(bench, "request_judgment", return_value=(response, "1_full_compliance", 128)) as request,
            patch.object(bench, "append_jsonl") as append,
        ):
            bench.judge_api({"seed": 42}, SimpleNamespace(judge_model=None), "xstest")
        sent_prompt = request.call_args.args[2]
        assert sent_prompt == "Original taxonomy. test question test response\nCLASS:"
        assert sent_prompt.endswith("CLASS:")
        assert checkpoint.call_args.args[2]["prompt_sha256"] == "official-hash"
        assert append.call_args.args[1]["finish_reason"] == "stop"
        assert append.call_args.args[1]["max_output_tokens"] == 128

    def test_api_uses_inline_key_and_checks_models_list(self):
        settings = {
            "model": "gpt-4o-mini",
            "base_url": "https://api.chatanywhere.tech/v1",
            "max_tokens_parameter": "max_tokens",
            "max_output_tokens": 256,
        }
        with patch.object(bench, "API_KEY", "unit-test-inline-key"), patch("requests.request") as request:
            request.return_value.status_code = 200
            request.return_value.json.return_value = {"data": [{"id": "gpt-4o-mini"}]}
            result = bench.api_request({"api_timeout_seconds": 180}, settings)
            assert result["id"] == "gpt-4o-mini"
            assert request.call_args.args == ("GET", "https://api.chatanywhere.tech/v1/models")
            assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer unit-test-inline-key"
            request.return_value.json.return_value = {"data": [{"id": "another-model"}]}
            with self.assertRaises(ValueError):
                bench.api_request({"api_timeout_seconds": 180}, settings)

    def test_chat_request_uses_selected_model_and_never_puts_key_in_payload(self):
        settings = {
            "model": "gpt-4o-mini",
            "base_url": "https://api.chatanywhere.tech/v1",
            "max_tokens_parameter": "max_tokens",
            "max_output_tokens": 16,
        }
        with patch.object(bench, "API_KEY", "unit-test-inline-key"), patch("requests.request") as request:
            request.return_value.status_code = 200
            request.return_value.json.return_value = {"choices": []}
            bench.api_request({"api_timeout_seconds": 180}, settings, "test prompt")
            assert request.call_args.args == ("POST", "https://api.chatanywhere.tech/v1/chat/completions")
            payload = request.call_args.kwargs["json"]
            assert payload["model"] == "gpt-4o-mini"
            assert payload["temperature"] == 0
            assert payload["max_tokens"] == 16
            assert "unit-test-inline-key" not in str(payload)


if __name__ == "__main__":
    unittest.main()

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
import unittest
from pathlib import Path
from unittest.mock import patch


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


class BenchmarkTests(unittest.TestCase):
    def test_summary_reference_never_enters_generation_prompt(self):
        rows = [
            {"id": "post1", "subreddit": "test", "title": "title", "post": "post body", "summary": "REFERENCE_SECRET"}
        ]
        with patch.object(bench, "read_jsonl", return_value=rows):
            selected = bench.load_rows({"data_root": ".", "harmbench_split": "all", "limit": 0}, "sff")
        assert selected[0]["reference"] == "REFERENCE_SECRET"
        assert "REFERENCE_SECRET" not in selected[0]["prompt"]
        assert "post body" in selected[0]["prompt"]

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


if __name__ == "__main__":
    unittest.main()

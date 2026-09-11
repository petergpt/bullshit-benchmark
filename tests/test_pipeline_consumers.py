import gzip
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import model_launch_pipeline as inventory
import push_bullshitbench_to_forge as forge
import published_dataset as storage
import publication
import validate_configs as configs


class PipelineConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def write_json(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def write_rows(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def questions(self):
        return self.write_json("questions.json", {"techniques": [{
            "technique": "test_technique", "questions": [{
                "id": "q1", "question": "Fixture question?", "domain": "fixture domain",
                "nonsensical_element": "Fixture invalid premise.",
            }],
        }]})

    def frozen_questions(self, dataset_path, payload=None):
        if payload is None:
            payload = json.loads(self.questions().read_text())
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n"
        relative = f"metadata/{publication.sha(raw)}-questions.json"
        path = dataset_path.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest_path = dataset_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.setdefault("files", {})["questions.json"] = {
            "path": relative, "bytes": len(raw), "sha256": publication.sha(raw),
        }
        manifest_path.write_text(json.dumps(manifest))
        return path

    def test_forge_rehydrates_slim_rows_for_feedback(self):
        path = self.write_rows("data/v2/latest/aggregate.jsonl", [
            {"sample_id": "a", "question_id": "q1", "model": "example/a", "consensus_score": 2},
            {"sample_id": "b", "question_id": "q1", "model": "example/b", "consensus_score": 0},
        ])
        rows = forge.load_aggregate_rows(path, self.questions())
        event = next(forge.iter_feedback_events(rows, category="test", benchmark_version="v2", extra_tags=()))
        self.assertEqual(event["prompt"], "Fixture question?")
        self.assertEqual(event["metadata"]["domain"], "fixture domain")
        self.assertEqual(event["metadata"]["technique"], "test_technique")
        self.assertEqual(event["signals"]["score_margin"], 2)

    def test_forge_keeps_embedded_historical_questions(self):
        path = self.write_rows("run/aggregate.jsonl", [{
            "sample_id": "a", "question_id": "q1", "model": "example/a", "consensus_score": 1,
            "question": "Original question?", "domain": "original domain", "technique": "original technique",
        }])
        row = forge.load_aggregate_rows(path, self.questions())["q1"][0]
        self.assertEqual(row.question, "Original question?")
        self.assertEqual(row.domain, "original domain")

    def test_forge_frozen_prompts_ignore_mutable_root_for_both_question_shapes(self):
        path = self.write_rows("data/v2/latest/aggregate.jsonl", [
            {"sample_id": "a", "question_id": "q1", "model": "example/a", "consensus_score": 2},
            {"sample_id": "b", "question_id": "q1", "model": "example/b", "consensus_score": 0},
        ])
        grouped = json.loads(self.questions().read_text())
        flat = list(publication.question_definitions(json.dumps(grouped)).values())
        for payload in (grouped, flat):
            with self.subTest(shape=type(payload).__name__), mock.patch.object(forge, "ROOT", self.root):
                self.frozen_questions(path, payload)
                self.write_json("questions.v2.json", {"changed": "The mutable root is no longer a question source."})
                rows = forge.load_aggregate_rows(path)
                event = next(forge.iter_feedback_events(rows, category="test", benchmark_version="v2", extra_tags=()))
                self.assertEqual(event["prompt"], "Fixture question?")
                self.assertEqual(event["metadata"]["domain"], "fixture domain")
                self.assertEqual(event["metadata"]["technique"], "test_technique")

    def test_forge_explicit_question_source_must_match_frozen_snapshot(self):
        path = self.write_rows("published/aggregate.jsonl", [{
            "sample_id": "a", "question_id": "q1", "model": "example/a", "consensus_score": 1,
        }])
        self.frozen_questions(path)
        flat = list(publication.question_definitions(self.questions().read_bytes()).values())
        explicit = self.write_json("explicit-snapshot.json", flat)
        self.assertEqual(forge.load_aggregate_rows(path, explicit)["q1"][0].question, "Fixture question?")
        flat[0]["question"] = "A different prompt?"
        explicit.write_text(json.dumps(flat))
        with self.assertRaisesRegex(forge.ForgeError, "Explicit questions: changed question"):
            forge.load_aggregate_rows(path, explicit)

    def test_forge_preserves_compatible_embedded_questions_and_rejects_conflicts(self):
        row = {"sample_id": "a", "question_id": "q1", "model": "example/a", "consensus_score": 1,
               "question": "Fixture question?", "domain": "fixture domain", "technique": "test_technique"}
        path = self.write_rows("published/aggregate.jsonl", [row])
        self.frozen_questions(path)
        self.assertEqual(forge.load_aggregate_rows(path)["q1"][0].question, row["question"])
        for field in ("question", "domain", "technique"):
            self.write_rows("published/aggregate.jsonl", [{**row, field: "Conflicting historical field"}])
            with self.subTest(field=field), self.assertRaisesRegex(forge.ForgeError, f"changed {field}"):
                forge.load_aggregate_rows(path)

    def test_forge_corrupt_snapshot_fails_even_with_embedded_text_or_explicit_source(self):
        path = self.write_rows("published/aggregate.jsonl", [{
            "sample_id": "a", "question_id": "q1", "model": "example/a", "consensus_score": 1,
            "question": "Fixture question?", "domain": "fixture domain", "technique": "test_technique",
        }])
        self.frozen_questions(path).write_text("{}")
        for explicit in (None, self.questions()):
            with self.subTest(explicit=bool(explicit)), self.assertRaisesRegex(forge.ForgeError, "checksum mismatch"):
                forge.load_aggregate_rows(path, explicit)

    def test_forge_legacy_custom_dataset_supports_flat_question_snapshot(self):
        path = self.write_rows("custom/aggregate.jsonl", [{
            "sample_id": "a", "question_id": "q1", "model": "example/a", "consensus_score": 1,
        }])
        flat = [{"id": "q1", "question": "Minimal legacy question?", "technique": "legacy", "domain": "Custom"}]
        source = self.write_json("custom/questions_snapshot.json", flat)
        row = forge.load_aggregate_rows(path, source)["q1"][0]
        self.assertEqual(row.question, "Minimal legacy question?")
        self.assertEqual(row.domain, "Custom")

    def test_forge_excludes_failed_partial_scores_from_pairwise_events(self):
        common = {"question_id": "q1", "question": "Original question?", "domain": "Original", "technique": "fixture"}
        rows = [
            {**common, "sample_id": "a", "model": "example/a", "consensus_score": 2, "status": "ok"},
            {**common, "sample_id": "b", "model": "example/b", "consensus_score": 0},
        ]
        failures = [{"status": "error"}, {"error": "Provider error"}, {"row_errors": ["Incomplete panel"]},
                    {"consensus_error": "incomplete_judge_panel"}, {"row_identity_mismatch": True}]
        rows += [{**common, "sample_id": f"failed-{index}", "model": f"example/failed-{index}",
                  "consensus_score": 0.5, **failure} for index, failure in enumerate(failures)]
        path = self.write_rows("run/aggregate.jsonl", rows)
        grouped = forge.load_aggregate_rows(path)
        self.assertEqual([row.model for row in grouped["q1"]], ["example/a", "example/b"])
        events = list(forge.iter_feedback_events(grouped, category="test", benchmark_version="v1", extra_tags=()))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["signals"]["score_margin"], 2)

    def test_forge_rejects_manifest_change_during_snapshot_and_rows_read(self):
        path = self.write_rows("published/aggregate.jsonl", [{
            "sample_id": "a", "question_id": "q1", "model": "example/a", "consensus_score": 1,
        }])
        self.frozen_questions(path)
        read_text = forge.read_text
        def read_changed_manifest(dataset):
            text = read_text(dataset)
            manifest_path = dataset.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text()); manifest["new_release"] = True
            manifest_path.write_text(json.dumps(manifest))
            return text
        with mock.patch.object(forge, "read_text", side_effect=read_changed_manifest):
            with self.assertRaisesRegex(forge.ForgeError, "manifest changed while reading"):
                forge.load_aggregate_rows(path)

    def test_consumers_read_declared_parts_without_monolithic_jsonl(self):
        rows = [{
            "sample_id": f"sample-{i}", "model": f"example/model-{i}", "question_id": "q1",
            "consensus_score": i % 3, "question": "Unicode café\u2028question?",
        } for i in range(8)]
        dataset = self.root / "packed"
        aggregate = self.write_rows("packed/aggregate.jsonl", rows)
        self.write_rows("packed/responses.jsonl", rows)
        # Literal Unicode separators are valid within JSON strings and are not JSONL line endings.
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        aggregate.write_text(raw, encoding="utf-8")
        for name in ("viewer_rows.json.gz", "viewer_details.json.gz"):
            (dataset / name).write_bytes(gzip.compress(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0))
        self.write_json("packed/manifest.json", {"sources": {}, "counts": {}})
        storage.pack_dataset(dataset, max_file_bytes=512)
        # Retained monoliths must not be required by either consumer.
        aggregate.unlink(missing_ok=True)
        (dataset / "responses.jsonl").unlink(missing_ok=True)
        self.assertEqual(len(forge.load_aggregate_rows(aggregate)["q1"]), len(rows))
        self.assertEqual(inventory.read_jsonl(aggregate), rows)

    def test_forge_rejects_missing_question_and_truncated_json(self):
        path = self.write_rows("run/aggregate.jsonl", [{
            "sample_id": "a", "question_id": "unknown", "model": "example/a", "consensus_score": 1,
        }])
        with self.assertRaisesRegex(forge.ForgeError, "canonical --questions"):
            forge.load_aggregate_rows(path)
        path.write_text('{"consensus_score": null}\n{"sample_id":', encoding="utf-8")
        with self.assertRaisesRegex(forge.ForgeError, r"aggregate.jsonl:2"):
            forge.load_aggregate_rows(path)

    def test_inventory_rejects_malformed_json_instead_of_skipping_models(self):
        path = self.write_rows("responses.jsonl", [{"model": "example/a"}])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"model":')
        with self.assertRaisesRegex(ValueError, r"responses.jsonl:2"):
            inventory.read_jsonl(path)

    def test_inventory_defaults_include_both_suites_and_configs(self):
        v1 = self.write_rows("data/latest/aggregate.jsonl", [{"model": "example/v1"}])
        v2 = self.write_rows("data/v2/latest/aggregate.jsonl", [{"model": "example/v2"}])
        c1 = self.write_json("config.json", {"collect": {"models": ["example/v1"]}})
        c2 = self.write_json("config.v2.json", {"collect": {"models": ["example/v2"]}})
        with mock.patch.multiple(inventory,
            DEFAULT_CONFIG=c1, DEFAULT_CONFIG_V2=c2,
            DEFAULT_LATEST_AGGREGATE=v1, DEFAULT_LATEST_AGGREGATE_V2=v2,
            DEFAULT_LATEST_RESPONSES=v1.with_name("responses.jsonl"),
            DEFAULT_LATEST_RESPONSES_V2=v2.with_name("responses.jsonl"),
        ):
            rows, _ = inventory.scan_inventory(None, None, None, self.root / "runs")
        self.assertEqual({row["model_id"] for row in rows}, {"example/v1", "example/v2"})
        self.assertTrue(all(row["present_in_config"] == row["present_in_latest"] == "true" for row in rows))

    def test_inventory_explicit_sources_do_not_include_default_datasets(self):
        path = self.write_rows("custom/aggregate.jsonl", [{"model": "example/custom"}])
        config = self.write_json("custom/config.json", {"collect": {"models": ["example/custom"]}})
        rows, _ = inventory.scan_inventory(config, path, path.with_name("responses.jsonl"), self.root / "runs")
        self.assertEqual([row["model_id"] for row in rows], ["example/custom"])

    def test_end_to_end_dry_run_preserves_existing_viewer_dataset(self):
        config = self.write_json("config.json", {
            "collect": {"questions": str(self.questions()), "models": ["example/model"], "num_runs": 1, "parallelism": 1},
            "grade": {"judge_model": "example/judge-a", "parallelism": 1},
            "grade_panel": {"judge_models": ["example/judge-a", "example/judge-b", "example/judge-c"], "parallelism": 1},
        })
        viewer = self.root / "viewer-data"
        sentinel = self.write_json("viewer-data/manifest.json", {"preserve": "existing dataset"})
        before = sentinel.read_bytes()
        result = subprocess.run([
            "bash", str(ROOT / "scripts/run_end_to_end.sh"), "--config", str(config),
            "--output-dir", str(self.root / "runs"), "--viewer-output-dir", str(viewer),
            "--run-id", "fixture", "--dry-run", "--with-additional-judges",
        ], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("publish step skipped", result.stdout)
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertEqual([path.name for path in viewer.iterdir()], ["manifest.json"])
        self.assertTrue((self.root / "runs/fixture/responses.jsonl").is_file())
        self.assertTrue(list((self.root / "runs/fixture").rglob("aggregate.jsonl")))

    def test_durable_v2_config_has_unique_keys_and_retains_grok_routing(self):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    self.fail(f"Duplicate config key: {key}")
                result[key] = value
            return result
        config = json.loads((ROOT / "config.v2.json").read_text(), object_pairs_hook=unique)
        for model in ("x-ai/grok-4.3", "x-ai/grok-4.5"):
            provider = config["collect"]["model_request_overrides"][model]["provider"]
            self.assertEqual(provider["order"], ["xAI"])
            self.assertFalse(provider["allow_fallbacks"])

    def test_config_validation_rejects_duplicate_keys_and_orphan_maps(self):
        path = self.root / "config.json"
        path.write_text('{"collect":{"models":["example/a"],"model_request_overrides":{},"model_request_overrides":{}}}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            configs.configured_variants(path)
        for name, value in (
            ("model_reasoning_efforts", ["low"]),
            ("model_request_overrides", {"provider": {"order": ["Example"]}}),
            ("model_providers", "openrouter"),
        ):
            with self.subTest(name=name):
                self.write_json("config.json", {"collect": {"models": ["example/a"], name: {"example/absent": value}}})
                with self.assertRaisesRegex(ValueError, "matches no configured model"):
                    configs.configured_variants(path)

    def test_config_variants_use_collector_effort_normalization(self):
        path = self.write_json("config.json", {"collect": {
            "models": ["example/default", "anthropic/claude-fable-5"],
            "response_reasoning_effort": "off",
            "model_reasoning_efforts": {"anthropic/claude-fable-5": ["minimal"]},
        }})
        self.assertEqual(configs.configured_variants(path), {
            "example/default@reasoning=default", "anthropic/claude-fable-5@reasoning=low",
        })
        self.write_json("config.json", {"collect": {
            "models": ["example/a"], "model_reasoning_efforts": {"example/a": ["high"]},
            "model_request_overrides": {"example/a": {"reasoning": {"effort": "low"}}},
        }})
        with self.assertRaisesRegex(ValueError, "absent from model_reasoning_efforts"):
            configs.configured_variants(path)

    def config_repository(self):
        for name in configs.CONFIG_NAMES:
            self.write_json(name, {"collect": {"models": ["example/active"]}})
        for _, dataset in configs.SUITES.values():
            path = self.root / dataset / "leaderboard.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("model\nexample/active@reasoning=default\n")
        return self.write_json(configs.EXCEPTIONS_PATH, {
            "schema_version": 1, "reason": "Retain preexisting historical results outside active reruns.",
            "suites": {"v1": [], "v2": []},
        })

    def test_config_gate_requires_explicit_exact_historical_exceptions(self):
        exceptions_path = self.config_repository()
        self.assertEqual(configs.validate_repository(self.root)["suites"]["v1"]["published_variants"], 1)
        leaderboard = self.root / "data/latest/leaderboard.csv"
        with leaderboard.open("a") as handle:
            handle.write("example/historical@reasoning=low\n")
        with self.assertRaisesRegex(ValueError, "published variants absent"):
            configs.validate_repository(self.root)
        exceptions = json.loads(exceptions_path.read_text())
        exceptions["suites"]["v1"] = ["example/historical@reasoning=low"]
        exceptions_path.write_text(json.dumps(exceptions))
        self.assertEqual(configs.validate_repository(self.root)["suites"]["v1"]["historical_exceptions"], 1)
        with leaderboard.open("a") as handle:
            handle.write("example/historical@reasoning=high\n")
        with self.assertRaisesRegex(ValueError, "historical@reasoning=high"):
            configs.validate_repository(self.root)

    def test_config_gate_allows_empty_candidate_queues_but_not_empty_main_configs(self):
        self.config_repository()
        for name in configs.CANDIDATE_CONFIG_NAMES:
            self.write_json(name, {"collect": {"models": []}})
        result = configs.validate_repository(self.root)
        for name in configs.CANDIDATE_CONFIG_NAMES:
            self.assertEqual(result["configs"][name], 0)
        self.write_json("config.json", {"collect": {"models": []}})
        with self.assertRaisesRegex(ValueError, "collect.models must contain non-empty model IDs"):
            configs.validate_repository(self.root)

    def test_empty_candidate_queue_rejects_stale_model_settings(self):
        self.config_repository()
        self.write_json("config.new-models.v1.json", {"collect": {
            "models": [], "model_reasoning_efforts": {"example/active": ["low"]},
        }})
        with self.assertRaisesRegex(ValueError, "matches no configured model"):
            configs.validate_repository(self.root)

    def test_config_gate_rejects_stale_exceptions(self):
        exceptions_path = self.config_repository()
        exceptions = json.loads(exceptions_path.read_text())
        exceptions["suites"]["v1"] = ["example/active@reasoning=default"]
        exceptions_path.write_text(json.dumps(exceptions))
        with self.assertRaisesRegex(ValueError, "stale historical exceptions"):
            configs.validate_repository(self.root)


if __name__ == "__main__":
    unittest.main()

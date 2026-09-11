import argparse
import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "openrouter_benchmark",
    ROOT / "scripts" / "openrouter_benchmark.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def judge_payload(content, *, finish_reason="stop", request_id="judge-response"):
    return {
        "id": request_id,
        "choices": [
            {"finish_reason": finish_reason, "message": {"content": content}}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }


class FakeJudgeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("Unexpected extra judge request")
        return self.payloads.pop(0)


class JudgeFailureTests(unittest.TestCase):
    def setUp(self):
        self.response = {
            "sample_id": "sample",
            "model": "example/model",
            "question_id": "q1",
            "technique": "test",
            "question": "Question?",
            "nonsensical_element": "Invalid premise.",
            "response_text": "That premise is invalid.",
            "error": "",
        }

    def grade(self, payloads, *, output_retries=2):
        client = FakeJudgeClient(payloads)
        grade = MODULE.grade_one(
            self.response,
            clients={"openrouter": client},
            judge_model="example/judge",
            judge_provider="openrouter",
            judge_system_prompt="Judge.",
            judge_user_template="{question}\n{response}",
            judge_user_template_control="",
            judge_no_hint=False,
            judge_temperature=None,
            judge_reasoning_effort="off",
            judge_max_tokens=100,
            judge_output_retries=output_retries,
            store_judge_response_raw=True,
            retries=1,
            pause_seconds=0,
            dry_run=False,
        )
        return grade, client

    def test_empty_output_retries_same_judge_then_accepts_valid_score(self):
        grade, client = self.grade([
            judge_payload(None, request_id="empty"),
            judge_payload(
                '{"justification": "Clear rejection.", "score": 2}',
                request_id="valid",
            ),
        ])

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0], client.calls[1])
        self.assertEqual(grade["judge_score"], 2)
        self.assertEqual(grade["error"], "")
        self.assertEqual(grade["judge_response_id"], "valid")
        self.assertIn("judge_retry_on_empty=1", grade["judge_warnings"])
        self.assertEqual(grade["judge_attempt_count"], 2)
        self.assertEqual(len(grade["judge_attempts"]), 2)
        self.assertEqual(grade["judge_attempts"][0]["response_id"], "empty")
        self.assertTrue(grade["judge_attempts"][0]["error"])
        self.assertEqual(grade["judge_attempts"][1]["response_id"], "valid")
        self.assertEqual(grade["judge_attempts"][1]["parse_mode"], "direct")

    def test_exhausted_empty_outputs_remain_missing_never_zero(self):
        grade, client = self.grade([judge_payload(None)] * 3)

        self.assertEqual(len(client.calls), 3)
        self.assertIsNone(grade["judge_score"])
        self.assertTrue(grade["error"])
        self.assertNotEqual(grade["judge_parse_mode"], "fallback_empty_judge_output")
        self.assertNotIn("judge_fallback_score_on_empty_output", grade["judge_warnings"])

    def test_exhausted_content_filtered_outputs_remain_missing(self):
        payload = judge_payload(None, finish_reason="content_filter")
        grade, client = self.grade([payload] * 3)

        self.assertEqual(len(client.calls), 3)
        self.assertIsNone(grade["judge_score"])
        self.assertTrue(grade["error"])
        self.assertEqual(grade["judge_finish_reason"], "content_filter")
        self.assertEqual(grade["judge_response_raw"], payload)
        self.assertEqual(grade["judge_usage"], payload["usage"])
        self.assertEqual(grade["judge_attempt_count"], 3)
        self.assertEqual(len(grade["judge_attempts"]), 3)
        self.assertTrue(all(
            attempt["finish_reason"] == "content_filter" and attempt["error"]
            for attempt in grade["judge_attempts"]
        ))

    def test_exhausted_unparseable_outputs_remain_missing(self):
        grade, client = self.grade([judge_payload("Unable to evaluate.")] * 3)

        self.assertEqual(len(client.calls), 3)
        self.assertIsNone(grade["judge_score"])
        self.assertTrue(grade["error"])
        self.assertEqual(grade["judge_raw_text"], "Unable to evaluate.")

    def test_valid_zero_is_a_score_and_does_not_retry(self):
        grade, client = self.grade([
            judge_payload('{"justification": "Accepted the premise.", "score": 0}')
        ])

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(grade["judge_score"], 0)
        self.assertEqual(grade["error"], "")
        self.assertEqual(grade["judge_justification"], "Accepted the premise.")

    def test_invalid_score_for_question_type_exhausts_output_retries(self):
        grade, client = self.grade([
            judge_payload('{"justification": "Mistaken control grade.", "score": 3}')
        ] * 3)
        self.assertEqual(len(client.calls), 3)
        self.assertIsNone(grade["judge_score"])
        self.assertTrue(MODULE.judge_failure_is_exhausted(grade))

    def test_zero_retries_makes_one_attempt_and_leaves_missing_score(self):
        grade, client = self.grade([judge_payload(None)], output_retries=0)

        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(grade["judge_score"])
        self.assertTrue(grade["error"])

    def test_candidate_provider_refusal_still_skips_judge(self):
        self.response.update({
            "response_text": MODULE.EMPTY_MODEL_RESPONSE_PLACEHOLDER,
            "response_outcome": "refusal",
            "response_refusal": True,
        })
        grade, client = self.grade([])

        self.assertEqual(client.calls, [])
        self.assertIsNone(grade["judge_score"])
        self.assertEqual(grade["error"], "")
        self.assertIn("grading_skipped=response_refusal", grade["judge_warnings"])


class StoredJudgeFailureTests(unittest.TestCase):
    def test_legacy_fallback_markers_are_rejected_without_mutating_source(self):
        markers = [
            {"judge_parse_mode": "fallback_empty_judge_output"},
            {"judge_warnings": ["judge_fallback_score_on_empty_output"]},
            {"judge_justification": "Fallback score: judge returned empty output after retries, "
             "so this response is treated as failing to challenge the premise."},
        ]
        for marker in markers:
            with self.subTest(marker=marker):
                original = {"sample_id": "old", "judge_score": 0, "error": "", **marker}
                normalized = MODULE.normalize_stored_judge_failure(original)

                self.assertEqual(original["judge_score"], 0)
                self.assertIsNone(normalized["judge_score"])
                self.assertEqual(normalized["legacy_judge_score"], 0)
                self.assertEqual(normalized["status"], "error")
                self.assertTrue(normalized["error"])
                self.assertEqual(
                    MODULE.normalize_stored_judge_failure(normalized), normalized
                )

    def test_valid_zero_and_candidate_refusal_are_preserved(self):
        rows = [
            {"judge_score": 0, "judge_parse_mode": "direct", "error": ""},
            {"judge_score": None, "response_refusal": True,
             "response_outcome": "refusal", "error": ""},
        ]
        for row in rows:
            with self.subTest(row=row):
                self.assertEqual(MODULE.judge_failure_reason(row), "")
                normalized = MODULE.normalize_stored_judge_failure(row)
                self.assertEqual(normalized["judge_score"], row["judge_score"])
                self.assertEqual(normalized["error"], "")

    def test_failure_markers_and_invalid_scores_cannot_count_as_grades(self):
        rows = [
            {"judge_score": 2, "error": "Request failed"},
            {"judge_score": 2, "status": "error"},
            {"judge_score": None},
            {"judge_score": True},
            {"judge_score": 7},
        ] + [
            {"judge_score": 2, "judge_finish_reason": reason}
            for reason in ("content_filter", "refusal", "length", "max_output_tokens")
        ]
        for row in rows:
            with self.subTest(row=row):
                normalized = MODULE.normalize_stored_judge_failure(row)
                self.assertIsNone(normalized["judge_score"])
                self.assertEqual(normalized["status"], "error")
                self.assertTrue(normalized["error"])

    def test_grade_loading_excludes_legacy_zero_from_aggregation_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            grade_dir = pathlib.Path(temp_dir)
            MODULE.write_json(grade_dir / "grade_meta.json", {"judge_model": "example/judge"})
            MODULE.write_jsonl(grade_dir / "grades.jsonl", [{
                "sample_id": "legacy", "judge_score": 0,
                "judge_parse_mode": "fallback_empty_judge_output", "error": "",
            }, {
                "sample_id": "real-zero", "judge_score": 0,
                "judge_parse_mode": "direct", "error": "",
            }])

            loaded = MODULE.load_grade_dir(str(grade_dir))

        self.assertIsNone(loaded["rows_by_sample"]["legacy"]["judge_score"])
        self.assertTrue(loaded["rows_by_sample"]["legacy"]["error"])
        self.assertEqual(loaded["rows_by_sample"]["real-zero"]["judge_score"], 0)

    def test_resume_retries_failed_and_legacy_rows_but_preserves_real_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir)
            config_path = output / "config.json"
            MODULE.write_json(config_path, {})
            responses_path = output / "responses.jsonl"
            responses = [{
                "sample_id": sample_id, "run_index": 1, "model": "example/model",
                "question_id": sample_id, "technique": "test", "question": "Question?",
                "nonsensical_element": "Invalid premise.",
                "response_text": "That premise is invalid.", "error": "",
            } for sample_id in ("real-zero", "legacy", "failed", "refusal")]
            responses[-1].update({"response_outcome": "refusal", "response_refusal": True})
            MODULE.write_jsonl(responses_path, responses)
            grade_dir = output / "grades" / "resume-test"
            grade_dir.mkdir(parents=True)
            checkpoint = [{
                **response, "judge_model": "example/judge", "judge_score": 0,
                "judge_parse_mode": "direct", "judge_justification": "Accepted premise.",
                "status": "ok",
            } for response in responses]
            checkpoint[1]["judge_parse_mode"] = "fallback_empty_judge_output"
            checkpoint[2].update({"judge_score": None, "error": "Failed judge request", "status": "error"})
            checkpoint[3]["judge_score"] = None
            MODULE.write_jsonl(grade_dir / "grades.jsonl", checkpoint)
            args = argparse.Namespace(**{
                **MODULE.GRADE_DEFAULTS,
                "config": str(config_path), "responses_file": str(responses_path),
                "output_dir": str(output), "grade_id": "resume-test", "resume": True,
                "judge_model": "example/judge", "parallelism": 2, "dry_run": True,
            })

            with mock.patch.object(MODULE, "grade_one", wraps=MODULE.grade_one) as grade_call:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = MODULE.run_grade(args)
            results = {
                row["sample_id"]: row
                for row in MODULE.read_jsonl(grade_dir / "grades.jsonl")
            }
            summary = json.loads((grade_dir / "summary.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            {call.args[0]["sample_id"] for call in grade_call.call_args_list},
            {"legacy", "failed"},
        )
        self.assertEqual(results["real-zero"]["judge_score"], 0)
        self.assertIsNone(results["refusal"]["judge_score"])
        self.assertEqual(results["legacy"]["judge_score"], 1)
        self.assertEqual(results["failed"]["judge_score"], 1)
        self.assertEqual(summary["new_rows_processed"], 2)


class AggregateJudgeFailureTests(unittest.TestCase):
    def test_only_exhausted_failure_qualifies_for_two_judge_consensus(self):
        failed_judges = {
            "missing-row": None,
            "failed-row": {"judge_score": None, "error": "Judge returned empty output"},
            "legacy-fallback": {
                "judge_score": 0, "judge_parse_mode": "fallback_empty_judge_output"
            },
            "exhausted": {"judge_score": None, "error": "Judge returned empty output",
                          "judge_attempt_count": 3},
        }
        for failure_kind, failed_fields in failed_judges.items():
            with self.subTest(failure_kind=failure_kind):
                with tempfile.TemporaryDirectory() as temp_dir:
                    output = pathlib.Path(temp_dir)
                    config_path = output / "config.json"
                    MODULE.write_json(config_path, {})
                    responses_path = output / "responses.jsonl"
                    response = {
                        "sample_id": "sample", "model": "example/model", "run_index": 1,
                        "question_id": "q1", "technique": "test", "question": "Question?",
                        "nonsensical_element": "Invalid premise.",
                        "response_text": "That premise is invalid.", "error": "",
                    }
                    refusal = {
                        **response, "sample_id": "refusal", "question_id": "q2",
                        "response_outcome": "refusal", "response_refusal": True,
                        "response_text": MODULE.EMPTY_MODEL_RESPONSE_PLACEHOLDER,
                    }
                    MODULE.write_jsonl(responses_path, [response, refusal])
                    grade_dirs = []
                    for judge_index, score in enumerate((2, 1, 2), start=1):
                        grade_dir = output / f"judge-{judge_index}"
                        grade_dir.mkdir()
                        grade_dirs.append(str(grade_dir))
                        MODULE.write_json(grade_dir / "grade_meta.json", {
                            "judge_model": f"example/judge-{judge_index}",
                            "responses_file": str(responses_path),
                        })
                        grade_rows = [{**refusal, "judge_score": None}]
                        if judge_index < 3:
                            grade_rows.append({**response, "judge_score": score})
                        elif failed_fields is not None:
                            grade_rows.append({**response, **failed_fields})
                        MODULE.write_jsonl(grade_dir / "grades.jsonl", grade_rows)
                    args = argparse.Namespace(**{
                        **MODULE.AGGREGATE_DEFAULTS,
                        "config": str(config_path), "grade_dirs": ",".join(grade_dirs),
                        "output_dir": str(output), "aggregate_id": "incomplete-test",
                    })

                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        exit_code = MODULE.run_aggregate(args)
                    results = {
                        row["sample_id"]: row for row in MODULE.read_jsonl(
                            output / "aggregates" / "incomplete-test" / "aggregate.jsonl"
                        )
                    }

                accepted = failure_kind in {"legacy-fallback", "exhausted"}
                self.assertEqual(exit_code, 0 if accepted else 2)
                self.assertEqual(results["sample"]["consensus_score"], 1.5 if accepted else None)
                self.assertEqual(results["sample"]["judge_valid_scores"], [2, 1])
                self.assertEqual(results["sample"]["status"], "ok" if accepted else "error")
                if accepted:
                    self.assertEqual(results["sample"]["judge_coverage"], "2/3 judges")
                    self.assertEqual(results["sample"]["judge_failure_policy"], "retry_then_two_valid")
                    self.assertEqual(results["sample"]["row_errors"], [])
                    self.assertTrue(results["sample"]["judge_3_error"])
                else:
                    self.assertIn("incomplete_judge_panel:2/3", results["sample"]["error"])
                self.assertIsNone(results["refusal"]["consensus_score"])
                self.assertEqual(results["refusal"]["status"], "ok")
                self.assertEqual(results["refusal"]["error"], "")

    def test_incomplete_model_keeps_counts_but_suppresses_headline_metrics(self):
        base = {
            "model": "example/model", "technique": "test", "run_index": 1,
            "is_control": False, "consensus_score": 2,
            "judge_1_score": 2, "judge_2_score": 2, "judge_3_score": 2,
            "status": "ok", "error": "",
        }
        rows = [
            {**base, "sample_id": "valid"},
            {**base, "sample_id": "failed", "status": "error",
             "error": "incomplete_judge_panel:2/3", "judge_3_score": None},
        ]

        summary = MODULE.summarize_aggregate_rows(rows, "mean", 3)
        model = summary["leaderboard"][0]

        self.assertEqual(model["score_2"], 1)
        self.assertEqual(model["error_count"], 1)
        self.assertEqual(model["scored_count"], 1)
        self.assertIsNone(model["detection_rate_score_2"])
        self.assertIsNone(model["avg_score"])
        self.assertEqual(summary["total_error_records"], 1)


class FinalizeJudgeConsensusTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "sample_id": "sample", "model": "example/model", "run_index": 1,
            "technique": "test", "question_id": "q1", "is_control": False,
            "judge_1_score": 0, "judge_1_justification": "Accepted the premise.",
            "judge_2_score": 2, "judge_2_justification": "Clear rejection.",
            "judge_3_score": None, "judge_3_error": "Content filtered after retries.",
            "judge_3_attempt_count": 3, "judge_3_finish_reason": "content_filter",
            "judge_3_status": "error", "judge_3_grade_dir": "judge-dir",
            "row_errors": ["Judge row error from judge-dir: Content filtered after retries."],
            "status": "error", "error": "incomplete_judge_panel:2/3",
            "consensus_error": "incomplete_judge_panel:2/3", "consensus_score": None,
        }

    def test_genuine_zero_is_in_two_vote_mean_and_diagnostics_are_preserved(self):
        source = json.loads(json.dumps(self.row))
        row = MODULE.finalize_judge_consensus(self.row)
        self.assertEqual(self.row, source)
        self.assertEqual(row["judge_1_score"], 0)
        self.assertEqual(row["judge_1_justification"], "Accepted the premise.")
        self.assertEqual(row["judge_valid_scores"], [0, 2])
        self.assertEqual(row["consensus_score"], 1)
        self.assertEqual(row["judge_valid_count"], 2)
        self.assertEqual(row["judge_expected_count"], 3)
        self.assertEqual(row["judge_coverage"], "2/3 judges")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["error"], "")
        self.assertEqual(row["row_errors"], [])
        self.assertEqual(row["judge_excluded_errors"][0]["judge_index"], 3)
        self.assertTrue(row["judge_excluded_errors"][0]["retry_exhausted"])
        self.assertTrue(row["judge_3_error"])
        self.assertEqual(MODULE.finalize_judge_consensus(row), row)

    def test_undercovered_unattempted_and_structural_errors_stay_blocking(self):
        variants = [
            {"judge_2_score": None, "judge_2_error": "Filtered", "judge_2_attempt_count": 3},
            {"judge_3_attempt_count": 0},
            {"judge_3_attempt_count": 2},
            {"judge_3_attempts": [{"attempt": 1}, {"attempt": 2}]},
            {"judge_3_attempts": [{}, {}, {}]},
            {"judge_3_attempts": [None, None, None]},
            {"judge_3_attempts": [{"parse_mode": "direct", "raw_text_chars": 90}] * 3},
            {"judge_3_attempts": [{"error": "Empty output"}] * 3, "judge_3_attempt_count": 0},
            {"judge_3_row_present": False},
            {"judge_3_error": "Missing sample_id in grade dir: judge-dir"},
            {"row_identity_mismatch": True},
            {"row_errors": ["Field mismatch across judges for model."]},
            {"error": "Unrelated structural failure"},
            {"source_response_error": "Collection request failed."},
            {"response_outcome": "error"},
            {"judge_1_model": "same-judge", "judge_2_model": "same-judge"},
        ]
        for change in variants:
            with self.subTest(change=change):
                row = MODULE.finalize_judge_consensus({**self.row, **change})
                self.assertIsNone(row["consensus_score"])
                self.assertEqual(row["status"], "error")
                self.assertNotIn("judge_failure_policy", row)

    def test_legacy_fake_zero_is_normalized_even_when_only_justification_survives(self):
        row = {**self.row, "judge_3_score": 0, "judge_3_error": "", "judge_3_status": "ok",
               "judge_3_justification": "Fallback score: judge returned empty output after retries.",
               "judge_3_attempt_count": 0, "judge_3_finish_reason": None,
               "row_errors": [], "error": "", "consensus_error": None}
        result = MODULE.finalize_judge_consensus(row)
        self.assertEqual(row["judge_3_score"], 0)
        self.assertEqual(result["judge_3_legacy_score"], 0)
        self.assertIsNone(result["judge_3_score"])
        self.assertEqual(result["consensus_score"], 1)
        self.assertEqual(result["status"], "ok")
        refusal = MODULE.finalize_judge_consensus({**row, "response_refusal": True})
        self.assertIsNone(refusal["consensus_score"])
        self.assertEqual(refusal["judge_1_score"], 0)
        self.assertEqual(refusal["judge_2_score"], 2)
        self.assertIsNone(refusal["judge_3_score"])
        self.assertEqual(refusal["judge_valid_scores"], [])
        self.assertEqual(refusal["status"], "ok")
        self.assertNotIn("judge_failure_policy", refusal)

    def test_partial_rule_does_not_change_other_consensus_methods(self):
        for method in ("majority", "min", "max", "primary_tiebreak"):
            self.assertIsNone(MODULE.finalize_judge_consensus(
                self.row, consensus_method=method
            )["consensus_score"])


class GradePanelFailureTests(unittest.TestCase):
    def test_panel_and_report_complete_with_two_valid_votes_and_keep_failure_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir)
            config_path = output / "config.json"
            MODULE.write_json(config_path, {})
            responses_path = output / "responses.jsonl"
            MODULE.write_jsonl(responses_path, [{
                "sample_id": "sample", "model": "example/model", "run_index": 1,
                "question_id": "q1", "technique": "test", "question": "Question?",
                "nonsensical_element": "Invalid premise.",
                "response_text": "That premise is invalid.", "error": "",
            }])
            args = argparse.Namespace(**{
                **MODULE.GRADE_PANEL_DEFAULTS,
                "config": str(config_path), "responses_file": str(responses_path),
                "judge_models": "example/judge-1,example/judge-2,example/judge-3",
                "output_dir": str(output), "panel_id": "fallback-panel", "dry_run": True,
            })
            real_grade = MODULE.grade_one

            def fake_grade(*args, **kwargs):
                row = real_grade(*args, **kwargs)
                if kwargs["judge_model"].endswith("3"):
                    row.update(judge_score=None, error="Content filtered after retries.",
                               judge_attempt_count=3, judge_attempts=[{"error": "Filtered"}] * 3,
                               judge_finish_reason="content_filter", judge_parse_mode="missing_judgment")
                else:
                    row["judge_score"] = 0 if kwargs["judge_model"].endswith("1") else 2
                return row

            with mock.patch.object(MODULE, "grade_one", side_effect=fake_grade):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = MODULE.run_grade_panel(args)
            panel = json.loads((output / "grade_panels/fallback-panel/panel_summary.json").read_text())
            aggregate_path = pathlib.Path(panel["aggregate_dir"]) / "aggregate.jsonl"
            aggregate = MODULE.read_jsonl(aggregate_path)
            grade_dirs = panel["grade_dirs_for_aggregate"]
            judge_summary = json.loads((pathlib.Path(grade_dirs[2]) / "summary.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(panel["status"], "complete")
            self.assertEqual(aggregate[0]["consensus_score"], 1)
            self.assertEqual(aggregate[0]["judge_coverage"], "2/3 judges")
            self.assertEqual(judge_summary["total_error_records"], 1)

            # A cached pre-policy score must be replaced by the genuine current votes.
            aggregate[0]["consensus_score"] = 0.6667
            MODULE.write_jsonl(aggregate_path, aggregate)
            report_args = argparse.Namespace(**{
                **MODULE.REPORT_DEFAULTS,
                "config": str(config_path), "responses_file": str(responses_path),
                "grade_dirs": ",".join(grade_dirs), "aggregate_dir": panel["aggregate_dir"],
                "output_file": str(output / "report.json"),
            })
            with mock.patch.object(MODULE, "_render_report_html", side_effect=json.dumps):
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.run_report(report_args)
            report = json.loads((output / "report.json").read_text())
            row = report["rows"][0]
            self.assertEqual(row["consensus_score"], 1)
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["judge_failure_policy"], "retry_then_two_valid")
            self.assertEqual(row["row_errors"], [])
            self.assertTrue(row["judges"][2]["error"])
            self.assertEqual(report["errors"][0]["phase"], "grade")

            # The same cached aggregate cannot conceal a missing grade-dir row.
            MODULE.write_jsonl(pathlib.Path(grade_dirs[2]) / "grades.jsonl", [])
            with mock.patch.object(MODULE, "_render_report_html", side_effect=json.dumps):
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.run_report(report_args)
            row = json.loads((output / "report.json").read_text())["rows"][0]
            self.assertIsNone(row["consensus_score"])
            self.assertEqual(row["status"], "error")

            # An existing judge row for a different response also stays blocked.
            first_grade = MODULE.read_jsonl(pathlib.Path(grade_dirs[0]) / "grades.jsonl")[0]
            MODULE.write_jsonl(pathlib.Path(grade_dirs[2]) / "grades.jsonl", [{
                **first_grade, "judge_model": "example/judge-3", "judge_score": 2,
                "response_text": "A different candidate answer.",
            }])
            with mock.patch.object(MODULE, "_render_report_html", side_effect=json.dumps):
                with contextlib.redirect_stdout(io.StringIO()):
                    MODULE.run_report(report_args)
            row = json.loads((output / "report.json").read_text())["rows"][0]
            self.assertIsNone(row["consensus_score"])
            self.assertIn("row_identity_mismatch", row["error"])

            report_args.grade_dirs = ",".join(grade_dirs[:2])
            with self.assertRaisesRegex(ValueError, "exactly the configured"):
                MODULE.run_report(report_args)

    def test_failed_panel_resume_clears_stale_successful_aggregate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir)
            config_path = output / "config.json"
            MODULE.write_json(config_path, {})
            responses_path = output / "responses.jsonl"
            MODULE.write_jsonl(responses_path, [{
                "sample_id": "sample", "model": "example/model", "run_index": 1,
                "question_id": "q1", "technique": "test", "question": "Question?",
                "nonsensical_element": "Invalid premise.",
                "response_text": "That premise is invalid.", "error": "",
            }])
            panel_dir = output / "grade_panels" / "resume-panel"
            panel_dir.mkdir(parents=True)
            summary_path = panel_dir / "panel_summary.json"
            MODULE.write_json(summary_path, {
                "panel_id": "resume-panel", "status": "complete",
                "aggregate_dir": str(panel_dir / "aggregates" / "old-success"),
            })
            args = argparse.Namespace(**{
                **MODULE.GRADE_PANEL_DEFAULTS,
                "config": str(config_path), "responses_file": str(responses_path),
                "judge_models": "example/judge-1,example/judge-2,example/judge-3",
                "output_dir": str(output), "panel_id": "resume-panel",
                "resume": True, "dry_run": True,
            })
            pending_summaries = []

            def fail_primary_judges(*_args, **_kwargs):
                pending_summaries.append(json.loads(summary_path.read_text()))
                raise RuntimeError("Judge panel still has missing grades")

            with mock.patch.object(
                MODULE, "_run_primary_judges_for_panel", side_effect=fail_primary_judges
            ) as run_primary:
                with self.assertRaisesRegex(RuntimeError, "still has missing grades"):
                    MODULE.run_grade_panel(args)
            final_summary = json.loads(summary_path.read_text())

        run_primary.assert_called_once()
        self.assertEqual(pending_summaries[0]["status"], "running")
        self.assertIsNone(pending_summaries[0]["aggregate_dir"])
        self.assertEqual(final_summary["status"], "incomplete")
        self.assertIsNone(final_summary["aggregate_dir"])
        self.assertIn("still has missing grades", final_summary["error"])


if __name__ == "__main__":
    unittest.main()

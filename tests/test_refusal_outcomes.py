import importlib.util
import pathlib
import unittest
from unittest.mock import Mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "openrouter_benchmark",
    ROOT / "scripts" / "openrouter_benchmark.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RefusalOutcomeTests(unittest.TestCase):
    def collect_with_client(self, client):
        return MODULE.collect_one(
            {"sample_id": "sample", "run_index": 1, "model": "example/model",
             "model_id": "example/model", "model_provider": "openrouter",
             "question": {"id": "q1", "technique": "test", "domain": "Test",
                          "question": "Question?", "nonsensical_element": "Invalid premise."}},
            clients={"openrouter": client}, system_prompt="", omit_system_prompt=True,
            temperature=None, max_tokens=0, empty_response_retries=2, retries=1,
            pause_seconds=0, dry_run=False, store_request_messages=True, store_response_raw=True,
        )

    def test_empty_collection_exhausts_three_attempts_and_retains_diagnostics(self):
        for content in (None, " ", MODULE.EMPTY_MODEL_RESPONSE_PLACEHOLDER):
            with self.subTest(content=content):
                payloads = [{"id": f"empty-{i}", "model": "example/model",
                             "usage": {"completion_tokens": i},
                             "choices": [{"finish_reason": "stop", "native_finish_reason": "completed",
                                          "message": {"content": content, "refusal": None}}]}
                            for i in range(1, 4)]
                client = Mock()
                client.chat.side_effect = payloads
                row = self.collect_with_client(client)
                self.assertEqual(client.chat.call_count, 3)
                self.assertTrue(all(call == client.chat.call_args_list[0]
                                    for call in client.chat.call_args_list))
                self.assertEqual(row["error_kind"], "empty_response")
                self.assertIn("3 output attempts", row["error"])
                self.assertFalse(row["error_retryable"])
                self.assertFalse(row["response_refusal"])
                self.assertEqual(row["response_outcome"], "error")
                self.assertEqual(row["response_text"], MODULE.EMPTY_MODEL_RESPONSE_PLACEHOLDER)
                self.assertEqual(row["response_raw"], payloads[-1])
                self.assertEqual(row["response_id"], "empty-3")
                self.assertEqual([a["attempt"] for a in row["empty_response_attempts"]], [1, 2, 3])
                self.assertEqual([a["response_id"] for a in row["empty_response_attempts"]],
                                 ["empty-1", "empty-2", "empty-3"])
                self.assertEqual([a["usage"]["completion_tokens"] for a in row["empty_response_attempts"]],
                                 [1, 2, 3])

    def test_substantive_collection_after_empty_retry_remains_valid(self):
        client = Mock()
        client.chat.side_effect = [
            {"id": "empty", "choices": [{"message": {"content": None}}]},
            {"id": "answer", "choices": [{"message": {"content": "The premise is invalid."}}]},
        ]
        row = self.collect_with_client(client)
        self.assertEqual(client.chat.call_count, 2)
        self.assertEqual(row["error"], "")
        self.assertEqual(row["response_outcome"], "response")
        self.assertEqual(row["response_text"], "The premise is invalid.")
        self.assertEqual(len(row["empty_response_attempts"]), 1)

    def test_empty_native_refusal_collection_remains_unscored_outcome(self):
        client = Mock()
        client.chat.return_value = {"id": "refused", "choices": [{
            "finish_reason": "stop", "native_finish_reason": "refusal",
            "message": {"content": None},
        }]}
        row = self.collect_with_client(client)
        self.assertEqual(row["error"], "")
        self.assertTrue(row["response_refusal"])
        self.assertEqual(row["response_outcome"], "refusal")
        self.assertEqual(row["response_native_finish_reason"], "refusal")

    def test_grade_rejects_empty_outputs_without_judge_call(self):
        for content in (None, " ", MODULE.EMPTY_MODEL_RESPONSE_PLACEHOLDER):
            with self.subTest(content=content):
                client = Mock()
                row = {"sample_id": "sample", "model": "example/model", "question_id": "q1",
                       "question": "Question?", "nonsensical_element": "Invalid premise.",
                       "response_text": content, "response_refusal": False, "error": ""}
                grade = MODULE.grade_one(
                    row, clients={"openrouter": client}, judge_model="example/judge",
                    judge_provider="openrouter", judge_system_prompt="Judge.",
                    judge_user_template="{question}\n{response}", judge_user_template_control="",
                    judge_no_hint=False, judge_temperature=None, judge_reasoning_effort="off",
                    judge_max_tokens=100, judge_output_retries=2, store_judge_response_raw=True,
                    retries=1, pause_seconds=0, dry_run=False,
                )
                client.chat.assert_not_called()
                self.assertIn("Cannot grade empty response_text", grade["error"])
                self.assertIsNone(grade["judge_score"])
                self.assertEqual(grade["judge_attempt_count"], 0)

    def test_fable_minimal_normalizes_to_native_low(self) -> None:
        variants = MODULE.build_model_variants(
            ["anthropic/claude-fable-5"],
            None,
            {"anthropic/claude-fable-5": ["minimal"]},
            {"*": "openrouter"},
            {},
        )

        self.assertEqual(len(variants), 1)
        variant = variants[0]
        self.assertEqual(variant["model_reasoning_level"], "low")
        self.assertEqual(variant["response_reasoning_effort"], "low")
        self.assertEqual(
            variant["request_overrides"]["reasoning"]["effort"],
            "low",
        )
        self.assertEqual(
            variant["model_label"],
            "anthropic/claude-fable-5@reasoning=low",
        )

    def test_stored_fable_minimal_row_displays_as_low(self) -> None:
        row = {
            "model": "anthropic/claude-fable-5@reasoning=minimal",
            "model_id": "anthropic/claude-fable-5",
            "model_provider": "openrouter",
            "model_row": "claude-fable-5@reasoning=minimal",
            "model_reasoning_level": "minimal",
            "response_reasoning_effort": "minimal",
        }

        MODULE.normalize_stored_model_reasoning_variant(row)

        self.assertEqual(row["model"], "anthropic/claude-fable-5@reasoning=low")
        self.assertEqual(row["model_row"], "claude-fable-5@reasoning=low")
        self.assertEqual(row["model_reasoning_level"], "low")
        self.assertEqual(row["response_reasoning_effort"], "low")

    def test_native_refusal_is_classified(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "refusal",
                    "message": {"content": None},
                }
            ]
        }
        row = {"response_raw": payload, "response_text": "", "error": ""}

        MODULE.annotate_response_outcome(row)

        self.assertTrue(row["response_refusal"])
        self.assertEqual(row["response_outcome"], "refusal")
        self.assertEqual(row["response_native_finish_reason"], "refusal")

    def test_plain_empty_response_is_not_assumed_to_be_refusal(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": None},
                }
            ]
        }
        row = {"response_raw": payload, "error": ""}

        MODULE.annotate_response_outcome(row)

        self.assertFalse(row["response_refusal"])
        self.assertEqual(row["response_outcome"], "response")

    def test_substantive_text_overrides_native_refusal_marker(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "native_finish_reason": "refusal",
                    "message": {
                        "content": "The premise is invalid, and here is why."
                    },
                }
            ]
        }
        row = {
            "response_raw": payload,
            "response_text": "The premise is invalid, and here is why.",
            "error": "",
        }

        MODULE.annotate_response_outcome(row)

        self.assertFalse(row["response_refusal"])
        self.assertEqual(row["response_outcome"], "response")

    def test_persisted_outcome_survives_slimming_response_text(self) -> None:
        full_row = {
            "response_text": "The premise is invalid, and here is why.",
            "response_native_finish_reason": "refusal",
        }
        MODULE.annotate_response_outcome(full_row)
        summary_row = {key: value for key, value in full_row.items() if key != "response_text"}

        self.assertFalse(MODULE.response_is_refusal(full_row))
        self.assertFalse(MODULE.response_is_refusal(summary_row))
        summary_row.pop("response_refusal")
        self.assertFalse(MODULE.response_is_refusal(summary_row))

    def test_explicit_refusal_boolean_has_priority_over_other_markers(self) -> None:
        for value in (False, "false", " FALSE "):
            with self.subTest(value=value):
                self.assertFalse(MODULE.response_is_refusal({
                    "response_refusal": value,
                    "response_outcome": "refusal",
                    "response_native_finish_reason": "refusal",
                    "response_text": "",
                }))
        for value in (True, "true", " TRUE "):
            with self.subTest(value=value):
                self.assertTrue(MODULE.response_is_refusal({
                    "response_refusal": value,
                    "response_outcome": "response",
                }))

    def test_missing_text_is_not_an_empty_answer(self) -> None:
        for marker in (
            {"response_native_finish_reason": "refusal"},
            {"response_raw": {"choices": [{"native_finish_reason": "refusal"}]}},
        ):
            with self.subTest(marker=marker):
                self.assertFalse(MODULE.response_is_refusal(marker))
                self.assertFalse(MODULE.response_is_refusal({**marker, "response_text": None}))
                for empty in ("", MODULE.EMPTY_MODEL_RESPONSE_PLACEHOLDER):
                    self.assertTrue(MODULE.response_is_refusal({**marker, "response_text": empty}))

    def test_fresh_payload_detects_refusal_before_storing_outcome(self) -> None:
        payloads = [
            {"choices": [{"native_finish_reason": "refusal", "message": {"content": None}}]},
            {"choices": [{"message": {"content": None, "refusal": "I cannot answer."}}]},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                row = {
                    "response_refusal": False,
                    "response_outcome": "response",
                    "response_text": MODULE.EMPTY_MODEL_RESPONSE_PLACEHOLDER,
                    "response_raw": None,
                    "error": "",
                }
                MODULE.annotate_response_outcome(row, payload)

                self.assertTrue(row["response_refusal"])
                self.assertEqual(row["response_outcome"], "refusal")
                self.assertIsNone(row["response_raw"])

    def test_summary_excludes_refusal_from_scores_and_reliability(self) -> None:
        rows = [
            {
                "model": "example/model@reasoning=minimal",
                "technique": "test",
                "run_index": 1,
                "is_control": False,
                "consensus_score": 2,
                "judge_1_score": 2,
                "judge_2_score": 2,
                "status": "ok",
            },
            {
                "model": "example/model@reasoning=minimal",
                "technique": "test",
                "run_index": 1,
                "is_control": False,
                "consensus_score": 0,
                "judge_1_score": 0,
                "judge_2_score": 0,
                "status": "ok",
            },
            {
                "model": "example/model@reasoning=minimal",
                "technique": "test",
                "run_index": 1,
                "is_control": False,
                "response_outcome": "refusal",
                "response_refusal": True,
                "consensus_score": 0,
                "judge_1_score": 0,
                "judge_2_score": 0,
                "status": "ok",
            },
        ]

        summary = MODULE.summarize_aggregate_rows(rows, "mean", 2)
        model = summary["leaderboard"][0]

        self.assertEqual(model["nonsense_count"], 3)
        self.assertEqual(model["answered_count"], 2)
        self.assertEqual(model["refusal_count"], 1)
        self.assertEqual(model["score_2"], 1)
        self.assertEqual(model["score_0"], 1)
        self.assertEqual(model["avg_score"], 1.0)
        self.assertEqual(model["detection_rate_score_2"], 0.3333)
        self.assertEqual(model["full_engagement_rate_score_0"], 0.3333)
        self.assertEqual(model["refusal_rate"], 0.3333)
        self.assertEqual(summary["total_scored_records"], 2)
        self.assertEqual(summary["total_refusal_records"], 1)
        self.assertEqual(
            summary["reliability"]["pairwise"][0]["compared_rows"],
            2,
        )

    def test_grade_skips_provider_refusal(self) -> None:
        row = {
            "sample_id": "sample",
            "model": "example/model@reasoning=minimal",
            "question_id": "q1",
            "technique": "test",
            "question": "Question?",
            "nonsensical_element": "Invalid premise.",
            "response_text": MODULE.EMPTY_MODEL_RESPONSE_PLACEHOLDER,
            "response_outcome": "refusal",
            "response_refusal": True,
            "error": "",
        }

        grade = MODULE.grade_one(
            row,
            clients=None,
            judge_model="example/judge",
            judge_provider="openrouter",
            judge_system_prompt="Judge.",
            judge_user_template="{question}\n{response}",
            judge_user_template_control="",
            judge_no_hint=False,
            judge_temperature=None,
            judge_reasoning_effort="off",
            judge_max_tokens=100,
            judge_output_retries=0,
            store_judge_response_raw=False,
            retries=1,
            pause_seconds=0,
            dry_run=False,
        )

        self.assertEqual(grade["error"], "")
        self.assertIsNone(grade["judge_score"])
        self.assertIn("grading_skipped=response_refusal", grade["judge_warnings"])


if __name__ == "__main__":
    unittest.main()

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "openrouter_benchmark",
    ROOT / "scripts" / "openrouter_benchmark.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RefusalOutcomeTests(unittest.TestCase):
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
        row = {"response_raw": payload, "error": ""}

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

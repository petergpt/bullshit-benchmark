import copy
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "openrouter_benchmark", ROOT / "scripts" / "openrouter_benchmark.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AggregateOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        base = {
            "model": "example/model", "technique": "test", "run_index": 1,
            "is_control": False, "status": "ok", "error": "",
            "response_refusal": False, "response_outcome": "response",
        }
        self.rows = [
            {**base, "sample_id": "clear", "consensus_score": 1.6667,
             "judge_1_score": 1, "judge_2_score": 2, "judge_3_score": 2},
            {**base, "sample_id": "partial", "consensus_score": 1,
             "judge_1_score": 0, "judge_2_score": 1, "judge_3_score": 2},
            {**base, "sample_id": "engaged", "consensus_score": 0.3333,
             "judge_1_score": 0, "judge_2_score": 0, "judge_3_score": 1},
            {**base, "sample_id": "refusal", "response_refusal": True,
             "response_outcome": "refusal", "consensus_score": None,
             "judge_1_score": None, "judge_2_score": None, "judge_3_score": None},
            {**base, "sample_id": "historical-partial", "consensus_score": 2,
             "judge_1_score": 2, "judge_2_score": 2, "judge_3_score": None,
             "judge_3_error": "Provider error", "judge_3_status": "error",
             "error": "Provider error", "status": "error"},
            {**base, "sample_id": "historical-missing", "consensus_score": None,
             "judge_1_score": None, "judge_2_score": None, "judge_3_score": None,
             "error": "Provider error", "status": "error"},
        ]
        self.historical_ids = {"historical-partial", "historical-missing"}

    def test_historical_errors_are_unscored_but_stay_in_all_attempt_denominator(self) -> None:
        original = copy.deepcopy(self.rows)
        summary = MODULE.summarize_aggregate_rows(
            self.rows, "mean", 3, legacy_error_sample_ids=self.historical_ids
        )
        model = summary["leaderboard"][0]

        self.assertEqual(self.rows, original)
        self.assertEqual(model["count"], 6)
        self.assertEqual(model["nonsense_count"], 6)
        self.assertEqual(model["answered_count"], 3)
        self.assertEqual(model["scored_count"], 3)
        self.assertEqual(model["error_count"], 2)
        self.assertEqual(model["refusal_count"], 1)
        self.assertEqual([model[f"score_{score}"] for score in range(3)], [1, 1, 1])
        self.assertEqual(model["avg_score"], 1.0)
        self.assertEqual(model["detection_rate_score_2"], 0.1667)
        self.assertEqual(model["full_engagement_rate_score_0"], 0.1667)
        self.assertEqual(model["refusal_rate"], 0.1667)
        self.assertEqual(model["technique_breakdown"], {"test": 1.0})
        self.assertEqual(model["run_average_scores"], {"1": 1.0})
        self.assertEqual(summary["total_scored_records"], 3)
        self.assertEqual(summary["total_error_records"], 2)
        self.assertEqual(summary["total_refusal_records"], 1)

    def test_new_incomplete_results_still_have_no_headline_metrics(self) -> None:
        summary = MODULE.summarize_aggregate_rows(self.rows, "mean", 3)
        model = summary["leaderboard"][0]

        self.assertIsNone(model["avg_score"])
        self.assertIsNone(model["detection_rate_score_2"])
        self.assertIsNone(model["full_engagement_rate_score_0"])
        self.assertEqual(summary["total_error_records"], 2)
        self.assertEqual(summary["total_scored_records"], 3)

    def test_a_historical_partial_consensus_never_becomes_a_valid_model_score(self) -> None:
        summary = MODULE.summarize_aggregate_rows(
            [self.rows[4]], "mean", 3, legacy_error_sample_ids=self.historical_ids
        )
        model = summary["leaderboard"][0]

        self.assertIsNone(model["avg_score"])
        self.assertEqual(model["answered_count"], 0)
        self.assertEqual(model["score_2"], 0)
        self.assertEqual(model["detection_rate_score_2"], 0)
        self.assertEqual(summary["total_scored_records"], 0)
        self.assertEqual(summary["total_error_records"], 1)

    def test_explicit_answer_is_scored_identically_after_text_is_slimmed(self) -> None:
        full = {**self.rows[0], "response_native_finish_reason": "refusal",
                "response_text": "The premise is invalid, and here is why."}
        slim = {key: value for key, value in full.items() if key != "response_text"}

        self.assertEqual(
            MODULE.summarize_aggregate_rows([full], "mean", 3),
            MODULE.summarize_aggregate_rows([slim], "mean", 3),
        )
        self.assertEqual(MODULE.summarize_aggregate_rows([slim], "mean", 3)["leaderboard"][0]["score_2"], 1)

    def test_approved_partial_adds_one_scored_question_without_changing_denominator(self) -> None:
        partial = MODULE.finalize_judge_consensus({
            **self.rows[0], "sample_id": "partial-panel", "judge_1_score": 1,
            "judge_2_score": 2, "judge_3_score": None,
            "judge_3_error": "Empty output after retries.", "judge_3_attempt_count": 3,
        })
        summary = MODULE.summarize_aggregate_rows([self.rows[0], partial, self.rows[3]], "mean", 3)
        model = summary["leaderboard"][0]
        self.assertEqual(model["count"], 3)
        self.assertEqual(model["nonsense_count"], 3)
        self.assertEqual(model["scored_count"], 2)
        self.assertEqual(model["error_count"], 0)
        self.assertEqual(model["refusal_count"], 1)
        self.assertEqual(model["score_2"], 2)
        self.assertEqual(model["detection_rate_score_2"], 0.6667)


if __name__ == "__main__":
    unittest.main()

import contextlib
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import repair_published_judges as repair

p, b, storage = repair.publication, repair.benchmark, repair.storage


class RepairPublishedJudgesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.questions = self.root / 'questions.json'
        self.questions.write_text(json.dumps([{
            'id': 'q', 'question': 'Question?', 'nonsensical_element': 'Invalid premise.',
            'domain': 'Test', 'technique': 'test', 'is_control': False,
        }]))
        self.response = {
            'sample_id': 'sample', 'model': 'example/model', 'question_id': 'q', 'run_index': 1,
            'response_text': 'The premise is invalid. “Exact text.”', 'response_refusal': False,
            'response_outcome': 'response', 'response_cost_usd': 0.25, 'response_total_tokens': 42,
        }
        self.aggregate = {
            'sample_id': 'sample', 'model': 'example/model', 'question_id': 'q', 'run_index': 1,
            'judge_1_model': 'example/original-judge', 'judge_1_score': 0,
            'judge_1_justification': 'Fallback score: judge returned empty output after retries.',
            'judge_2_score': 2, 'judge_2_justification': 'Original second judgment.',
            'judge_3_score': 2, 'judge_3_justification': 'Original third judgment.',
            'consensus_score': 1.3333, 'consensus_method': 'mean', 'judge_valid_scores': [0, 2, 2],
            'response_refusal': False, 'response_outcome': 'response', 'status': 'ok', 'error': '',
        }

    def prepare(self):
        # Deliberate whitespace and Unicode make byte preservation observable.
        (self.source / 'responses.jsonl').write_text(json.dumps(self.response, ensure_ascii=False, indent=None) + '  \n')
        (self.source / 'aggregate.jsonl').write_text(json.dumps(self.aggregate) + '\n')
        viewer = {**self.aggregate, 'response_cost_usd': 0.25, 'response_total_tokens': 42}
        for key in list(viewer):
            if key.endswith('_justification'):
                del viewer[key]
        details = [{'sample_id': 'sample', 'response_text': self.response['response_text'],
                    'response_reasoning_text': 'Preserve this reasoning too.'}]
        for name, rows in [('viewer_rows.json.gz', [viewer]), ('viewer_details.json.gz', details)]:
            (self.source / name).write_bytes(gzip.compress(json.dumps(rows, separators=(',', ':')).encode(), mtime=0))
        panel = {'judge_models': ['example/original-judge', 'example/judge2', 'example/judge3'],
                 'panel_mode': 'full', 'consensus_method': 'mean'}
        summary = b.summarize_aggregate_rows([self.aggregate], 'mean', 3)
        for name in p.SIDECARS:
            if name.endswith('.csv') or name == 'questions.json':
                continue
            value = panel if name == 'panel_summary.json' else summary if name == 'aggregate_summary.json' else {}
            (self.source / name).write_text(json.dumps(value))
        fields = ['rank', 'model', *repair.METRICS, 'org', 'launch_date']
        for name in ('leaderboard.csv', 'leaderboard_with_launch.csv'):
            with (self.source / name).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({'rank': 1, 'model': 'example/model', 'org': 'example',
                                 'launch_date': '2025-01-01', **{key: summary['leaderboard'][0][value]
                                                              for key, value in repair.METRICS.items()}})
        (self.source / 'model_launch_dates.csv').write_text('model_id,launch_date\n')
        (self.source / 'model_params.csv').write_text('model_id,total_params_b\n')
        (self.source / 'manifest.json').write_text(json.dumps({
            'exports': {'leaderboard_csv': 'data/latest/leaderboard.csv'}, 'counts': {}, 'sources': {},
        }))
        storage.pack_dataset(self.source)
        self.before = {name: storage.read_bytes(self.source / name) for name in storage.ASSET_FORMATS}

    def args(self, **changes):
        values = dict(source=self.source, output_dir=self.root / 'repaired', questions_file=self.questions,
                      grades=[], responses_file=None, judge_slot=1, publication_path='data/latest',
                      write_blocked_draft=False)
        return SimpleNamespace(**{**values, **changes})

    def grade_input(self, **changes):
        root = self.root / 'grade'
        root.mkdir(exist_ok=True)
        source = self.root / 'grading-responses.jsonl'
        source.write_text(json.dumps(self.response) + '\n')
        meta = {'phase': 'grade', 'grade_id': 'retry', 'dry_run': False,
                'judge_model': 'example/original-judge', 'responses_file': str(source)}
        (root / 'grade_meta.json').write_text(json.dumps(meta))
        row = {**self.response, 'judge_model': 'example/original-judge', 'judge_score': 2,
               'judge_justification': 'Clear rejection.', 'judge_parse_mode': 'direct',
               'judge_raw_text': '{"score":2,"justification":"Clear rejection."}',
               'judge_response_id': 'gen-actual-judgment', 'judge_finish_reason': 'stop',
               'error': '', 'status': 'ok', **changes}
        (root / 'grades.jsonl').write_text(json.dumps(row) + '\n')
        return root

    def test_verified_same_judge_repair_preserves_answers_other_judges_and_metadata(self):
        self.prepare()
        receipt = repair.build(self.args(grades=[self.grade_input()]))
        output = self.root / 'repaired'
        self.assertEqual(receipt['status'], 'validated')
        self.assertFalse(receipt['activated'])
        self.assertEqual(p.validate_dataset(output)['rows'], 1)
        row = p.json_rows(output / 'aggregate.jsonl')[0]
        self.assertEqual(row['judge_1_score'], 2)
        self.assertEqual(row['consensus_score'], 2)
        for slot in (2, 3):
            self.assertEqual(repair.slot_payload(row, slot), repair.slot_payload(self.aggregate, slot))
        for name in ('responses.jsonl', 'viewer_details.json.gz'):
            self.assertEqual(storage.read_bytes(output / name), self.before[name])
        for name, raw in self.before.items():
            self.assertEqual(storage.read_bytes(self.source / name), raw)
        viewer = p.decode(storage.read_bytes(output / 'viewer_rows.json.gz'))[0]
        self.assertEqual(viewer['response_cost_usd'], 0.25)
        self.assertEqual(viewer['response_total_tokens'], 42)
        self.assertEqual(viewer['consensus_score'], 2)
        self.assertNotIn(str(self.root), (output / 'manifest.json').read_text())
        self.assertTrue((self.root / 'repaired.repair.json').is_file())
        self.assertFalse((output / 'repair_receipt.json').exists())
        self.assertEqual(receipt['affected_samples'][0]['response_sha256'], repair.text_hash(self.response))

    def test_summary_only_historical_error_keeps_aggregate_bytes_but_excludes_partial_score(self):
        self.aggregate.update(judge_1_score=0, judge_1_justification='A real zero.', judge_2_score=1,
                              judge_3_score=None, judge_3_error='Provider error', judge_3_status='error',
                              consensus_score=0.5, status='error', error='Provider error', row_errors=['Provider error'])
        self.prepare()
        receipt = repair.build(self.args())
        output = self.root / 'repaired'
        self.assertEqual(receipt['status'], 'validated')
        self.assertEqual(receipt['affected_samples'], [])
        self.assertEqual(storage.read_bytes(output / 'aggregate.jsonl'), self.before['aggregate.jsonl'])
        summary = p.decode((output / 'aggregate_summary.json').read_bytes())
        self.assertEqual(summary['total_error_records'], 1)
        self.assertEqual(summary['total_scored_records'], 0)
        self.assertIsNone(summary['leaderboard'][0]['avg_score'])

    def test_valid_scores_cannot_be_replaced(self):
        self.aggregate.update(judge_1_score=1, judge_1_justification='A valid partial score.')
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'valid existing judgment'):
            repair.build(self.args(grades=[self.grade_input()]))
        self.assertFalse((self.root / 'repaired').exists())

    def test_merged_legacy_roster_alone_cannot_authorize_a_successful_replacement(self):
        self.aggregate.pop('judge_1_model')
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'per-sample judge provenance'):
            repair.build(self.args(grades=[self.grade_input()]))
        self.assertFalse((self.root / 'repaired').exists())

    def test_response_identity_text_hash_and_judge_must_match(self):
        self.prepare()
        for change in ({'response_text': 'Changed answer'}, {'run_index': 2}, {'question_id': 'other'},
                       {'model': 'example/other'}, {'judge_model': 'example/different-judge'},
                       {'source_response_sha256': '0' * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                repair.build(self.args(grades=[self.grade_input(**change)]))
        self.assertFalse((self.root / 'repaired').exists())

    def test_actual_output_and_non_dry_run_provenance_are_required(self):
        self.prepare()
        for change in ({'judge_raw_text': ''}, {'judge_response_id': ''}, {'judge_score': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                repair.build(self.args(grades=[self.grade_input(**change)]))
        path = self.grade_input()
        meta = json.loads((path / 'grade_meta.json').read_text())
        meta['dry_run'] = True
        (path / 'grade_meta.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'actual canonical grading'):
            repair.build(self.args(grades=[path]))

    def test_failed_probe_writes_only_an_explicitly_requested_blocked_draft(self):
        self.prepare()
        path = self.grade_input(judge_score=None, judge_raw_text='', judge_justification='',
                                judge_finish_reason='content_filter', error='Filtered after retries.', status='error')
        with self.assertRaisesRegex(ValueError, 'Unresolved replacement judgment'):
            repair.build(self.args(grades=[path]))
        receipt = repair.build(self.args(grades=[path], write_blocked_draft=True))
        self.assertEqual(receipt['status'], 'blocked')
        self.assertTrue(receipt['publication_validation_error'])
        row = p.json_rows(self.root / 'repaired' / 'aggregate.jsonl')[0]
        self.assertIsNone(row['judge_1_score'])
        self.assertIsNone(row['consensus_score'])
        self.assertEqual(row['status'], 'error')
        with self.assertRaises(ValueError):
            p.validate_dataset(self.root / 'repaired')
        self.assertEqual(storage.read_bytes(self.source / 'aggregate.jsonl'), self.before['aggregate.jsonl'])

    def test_candidate_refusal_uses_canonical_skip_without_changing_other_judges(self):
        self.response.update(response_refusal=True, response_outcome='refusal', response_text=b.EMPTY_MODEL_RESPONSE_PLACEHOLDER)
        self.aggregate.update(response_refusal=True, response_outcome='refusal', consensus_score=None,
                              judge_2_score=0, judge_3_score=0, judge_valid_scores=[])
        self.prepare()
        for draft in (False, True):
            with self.subTest(draft=draft):
                output = self.root / f'repaired-refusal-{draft}'
                receipt = repair.build(self.args(output_dir=output, write_blocked_draft=draft))
                row = p.json_rows(output / 'aggregate.jsonl')[0]
                self.assertEqual(receipt['status'], 'validated')
                self.assertEqual(receipt['unresolved_judgments'], [])
                self.assertIsNone(row['judge_1_score'])
                self.assertIsNone(row['consensus_score'])
                self.assertIn('grading_skipped=response_refusal', row['judge_1_warnings'])
                for slot in (2, 3):
                    self.assertEqual(repair.slot_payload(row, slot), repair.slot_payload(self.aggregate, slot))

    def test_existing_or_live_output_is_never_written(self):
        for path in (self.source, self.source / 'nested', p.ROOT / 'data' / 'new-repair'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'new isolated directory'):
                repair.build(self.args(output_dir=path))

    def test_blocked_cli_returns_nonzero(self):
        self.prepare()
        with contextlib.redirect_stdout(io.StringIO()):
            result = repair.main(['--source', str(self.source), '--output-dir', str(self.root / 'blocked'),
                                  '--questions-file', str(self.questions), '--write-blocked-draft'])
        self.assertEqual(result, 1)
        self.assertEqual(p.decode((self.root / 'blocked.repair.json').read_bytes())['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()

"""Publication failure and input integrity regressions; no service calls."""
import importlib.util
import contextlib
import csv
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import publication as p


class PublicationTests(unittest.TestCase):
    def question_document(self):
        return {'benchmark': 'fixture', 'version': 'original', 'techniques': [{
            'technique': 'fixture_technique', 'description': 'Original technique description.',
            'questions': [{'id': 'q', 'question': 'An original café\u2028question?',
                           'domain': 'Original domain', 'nonsensical_element': 'Original invalid premise.',
                           'domain_group': 'fixture', 'difficulty': 'fixture'}],
        }]}

    def question_snapshot(self):
        definitions = p.question_definitions(json.dumps(self.question_document()))
        return [{key: value for key, value in row.items() if key not in ('domain_group', 'difficulty')}
                for row in definitions.values()]

    def write_frozen_questions(self, root, document=None):
        raw = json.dumps(document or self.question_document(), ensure_ascii=False, indent=2).encode() + b'\n'
        relative = f'metadata/{p.sha(raw)}-questions.json'
        (root / 'metadata').mkdir(exist_ok=True)
        (root / relative).write_bytes(raw)
        (root / 'questions.json').write_bytes(raw)
        return {'path': relative, 'sha256': p.sha(raw), 'bytes': len(raw)}

    def incoming_fixture(self, root, snapshot=True):
        question = self.question_snapshot()[0]
        response = {'sample_id': 's', 'model': 'example/model@reasoning=low',
                    'question_id': question['id'], 'run_index': 1, 'response_text': 'An answer.',
                    **{key: question[key] for key in p.QUESTION_FIELDS}}
        aggregate = {**response, 'judge_1_score': 2, 'judge_2_score': 1,
                     'judge_3_score': 2, 'consensus_score': 1.6667}
        (root / 'responses.jsonl').write_text(json.dumps(response) + '\n')
        (root / 'aggregate.jsonl').write_text(json.dumps(aggregate) + '\n')
        (root / 'panel.json').write_text(json.dumps({'judge_models': ['a', 'b', 'c'], 'panel_mode': 'full'}))
        (root / 'summary.json').write_text('{"consensus_method":"mean"}')
        if snapshot:
            (root / 'questions_snapshot.json').write_text(json.dumps(self.question_snapshot()))
        args = SimpleNamespace(responses_file=root/'responses.jsonl', aggregate_rows=root/'aggregate.jsonl',
                               panel_summary=root/'panel.json', aggregate_summary=root/'summary.json',
                               publish_mode='supplemental', allow_replace_samples=False, questions_file=None)
        return args, response, aggregate

    def test_strict_jsonl_rejects_truncated_tail_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'input.jsonl'
            for raw in ['{"sample_id":"a"}\n{"sample_id":', '{"sample_id":"a"}\ngarbage',
                        '{"sample_id":"a"}\n{"sample_id":"a"}\n', '{"sample_id":"a","sample_id":"b"}\n']:
                f.write_text(raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    p.json_rows(f)

    def test_unicode_line_separators_remain_response_text(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'input.jsonl'
            f.write_text(json.dumps({'sample_id': 'a', 'text': 'x\u2028y'}, ensure_ascii=False) + '\n')
            self.assertEqual(p.json_rows(f)[0]['text'], 'x\u2028y')

    def test_parallel_writer_is_rejected_without_changing_dataset(self):
        with tempfile.TemporaryDirectory() as d, p.writer_lock(Path(d)):
            code = 'import publication; from pathlib import Path; publication.writer_lock(Path(__import__("sys").argv[1])).__enter__()'
            result = subprocess.run([sys.executable, '-c', code, d], env={**os.environ, 'PYTHONPATH': str(ROOT / 'scripts')}, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Another publisher', result.stderr)

    def test_metadata_digest_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'meta.json').write_bytes(b'{}')
            valid = {'path': 'meta.json', 'bytes': 2, 'sha256': p.sha(b'{}')}
            self.assertEqual(p.checked_file(root, valid), b'{}')
            (root / 'meta.json').write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                p.checked_file(root, valid)

    def test_failure_before_activation_keeps_previous_manifest_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); stage = root / 'stage'; dest = root / 'live'
            stage.mkdir(); dest.mkdir(); (dest / 'manifest.json').write_text('{}')
            (dest / 'old.jsonl').write_text('old-data')
            manifest = {'storage': {'assets': {'responses.jsonl': {'parts': [{'path': 'responses/missing.jsonl'}]}}}, 'files': {}}
            with self.assertRaises(OSError):
                p.activate(stage, dest, manifest)
            self.assertEqual((dest / 'manifest.json').read_text(), '{}')
            self.assertEqual((dest / 'old.jsonl').read_text(), 'old-data')

    def test_duplicate_config_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            p.decode('{"collect":{"models":[],"models":["model"]}}')

    def test_incoming_panel_and_consensus_must_be_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            response = {'sample_id': 's', 'model': 'example/model@reasoning=low',
                        'question_id': 'q', 'run_index': 1, 'response_text': 'Answer'}
            aggregate = {**response, 'judge_1_score': 2, 'judge_2_score': 1,
                         'judge_3_score': 2, 'consensus_score': 1.6667}
            (root / 'responses.jsonl').write_text(json.dumps(response) + '\n')
            (root / 'panel.json').write_text(json.dumps({'judge_models': ['a', 'b', 'c'], 'panel_mode': 'full'}))
            (root / 'summary.json').write_text('{"consensus_method":"mean"}')
            (root / 'questions_snapshot.json').write_text(json.dumps(self.question_snapshot()))
            args = SimpleNamespace(responses_file=root/'responses.jsonl', aggregate_rows=root/'aggregate.jsonl',
                                   panel_summary=root/'panel.json', aggregate_summary=root/'summary.json',
                                   publish_mode='replace', allow_replace_samples=False)
            (root / 'aggregate.jsonl').write_text(json.dumps(aggregate) + '\n')
            p.validate_incoming(args, root/'output')
            for changes in [{'judge_3_score': None}, {'judge_1_score': True}, {'consensus_score': 2}, {'question_id': 'other'},
                            {'judge_3_score': None, 'judge_3_error': 'Provider error', 'judge_3_status': 'error',
                             'row_errors': ['Provider error'], 'error': 'Provider error', 'status': 'error',
                             'consensus_score': 1.5, 'consensus_method': 'mean'}]:
                (root / 'aggregate.jsonl').write_text(json.dumps({**aggregate, **changes}) + '\n')
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    p.validate_incoming(args, root/'output')

    def test_incoming_rejects_synthetic_fallback_despite_complete_numeric_panel(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            response = {'sample_id': 's', 'model': 'example/model', 'question_id': 'q',
                        'run_index': 1, 'response_text': 'The premise is invalid.'}
            aggregate = {**response, 'judge_1_score': 2, 'judge_2_score': 2,
                         'judge_3_score': 0, 'consensus_score': 1.3333}
            (root / 'responses.jsonl').write_text(json.dumps(response) + '\n')
            (root / 'panel.json').write_text(json.dumps({'judge_models': ['a', 'b', 'c'], 'panel_mode': 'full'}))
            (root / 'summary.json').write_text('{"consensus_method":"mean"}')
            args = SimpleNamespace(responses_file=root/'responses.jsonl', aggregate_rows=root/'aggregate.jsonl',
                                   panel_summary=root/'panel.json', aggregate_summary=root/'summary.json',
                                   publish_mode='replace', allow_replace_samples=False)
            for marker in [
                {'judge_3_parse_mode': 'fallback_empty_judge_output'},
                {'judge_3_warnings': ['judge_fallback_score_on_empty_output']},
                {'judge_3_justification': 'Fallback score: judge returned empty output after retries, '
                                          'so this response is treated as failing to challenge the premise.'},
            ]:
                (root / 'aggregate.jsonl').write_text(json.dumps({**aggregate, **marker}) + '\n')
                with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, 'synthetic empty-output'):
                    p.validate_incoming(args, root/'output')

    def write_dataset_fixture(self, root, aggregate, legacy_error_sample_ids=None, response_changes=None):
        """Build mutually consistent published assets to exercise semantic checks."""
        from openrouter_benchmark import summarize_aggregate_rows
        response = {key: aggregate[key] for key in ('sample_id', 'model', 'question_id', 'run_index', 'response_text')}
        response.update(response_changes or {})
        (root / 'manifest.json').write_text(json.dumps({'files': {'questions.json': self.write_frozen_questions(root)}}))
        (root / 'responses.jsonl').write_text(json.dumps(response) + '\n')
        (root / 'aggregate.jsonl').write_text(json.dumps(aggregate) + '\n')
        (root / 'viewer_rows.json.gz').write_bytes(gzip.compress(json.dumps([aggregate]).encode()))
        (root / 'viewer_details.json.gz').write_bytes(gzip.compress(json.dumps([response]).encode()))
        summary = summarize_aggregate_rows([aggregate], consensus_method='mean', num_judges=3,
                                           legacy_error_sample_ids=legacy_error_sample_ids)
        (root / 'aggregate_summary.json').write_text(json.dumps(summary))
        leaderboard = dict(summary['leaderboard'][0])
        for output, source in [('green_rate', 'detection_rate_score_2'), ('red_rate', 'full_engagement_rate_score_0')]:
            leaderboard[output] = leaderboard[source]
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(leaderboard))
        writer.writeheader(); writer.writerow(leaderboard)
        (root / 'leaderboard.csv').write_text(out.getvalue())

    def test_dataset_rejects_historical_fallback_or_incomplete_judging(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            aggregate = {'sample_id': 's', 'model': 'example/model', 'question_id': 'q',
                         'run_index': 1, 'response_text': 'The premise is invalid.',
                         'judge_1_score': 2, 'judge_2_score': 2, 'judge_3_score': 0,
                         'consensus_score': 1.3333, 'row_errors': []}
            self.write_dataset_fixture(root, aggregate)
            self.assertEqual(p.validate_dataset(root)['rows'], 1)
            for changes in [
                {'judge_3_justification': 'Fallback score: judge returned empty output after retries.'},
                {'judge_3_score': None, 'consensus_score': 2},
                {'judge_3_error': 'Judge returned no usable output.'},
                {'judge_3_finish_reason': 'content_filter'},
                {'judge_3_finish_reason': 'length'},
            ]:
                # Even if every viewer and summary agrees, invalid judge rows
                # cannot become a canonical publication merely by repacking.
                self.write_dataset_fixture(root, {**aggregate, **changes})
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    p.validate_dataset(root)

    def test_dataset_accepts_real_zero_after_successful_empty_output_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            aggregate = {'sample_id': 's', 'model': 'example/model', 'question_id': 'q',
                         'run_index': 1, 'response_text': 'An answer engaging with the premise.',
                         'judge_1_score': 0, 'judge_2_score': 0, 'judge_3_score': 0,
                         'consensus_score': 0, 'row_errors': [], 'judge_3_parse_mode': 'direct',
                         'judge_3_finish_reason': 'stop', 'judge_3_warnings': ['judge_raw_text_empty', 'judge_retry_on_empty=1'],
                         'judge_3_justification': 'The answer accepts the false premise.'}
            self.write_dataset_fixture(root, aggregate)
            self.assertEqual(p.validate_dataset(root)['rows'], 1)

    def test_existing_explicit_legacy_errors_remain_unscored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            aggregate = {'sample_id': 's', 'model': 'example/model', 'question_id': 'q',
                         'run_index': 1, 'response_text': 'An answer.',
                         'judge_1_score': 0, 'judge_2_score': 1, 'judge_3_score': None,
                         'judge_3_error': 'Provider error', 'judge_3_status': 'error',
                         'row_errors': ['Provider error'], 'error': 'Provider error', 'status': 'error',
                         'consensus_score': 0.5, 'consensus_method': 'mean'}
            self.assertEqual(p.get_legacy_error_sample_ids([aggregate]), {'s'})
            self.write_dataset_fixture(root, aggregate, legacy_error_sample_ids={'s'})
            self.assertEqual(p.validate_dataset(root)['rows'], 1)
            summary = json.loads((root / 'aggregate_summary.json').read_text())
            self.assertEqual(summary['total_error_records'], 1)
            self.assertIsNone(summary['leaderboard'][0]['avg_score'])
            self.assertEqual(summary['leaderboard'][0]['scored_count'], 0)
            for changes in [
                {'judge_panel_id': 'new-panel'}, {'judge_3_attempt_count': 3},
                {'status': 'ok'}, {'error': ''}, {'row_errors': []},
                {'consensus_score': 2}, {'consensus_error': 'incomplete_judge_panel'},
            ]:
                changed = {**aggregate, **changes}
                self.assertEqual(p.get_legacy_error_sample_ids([changed]), set(), changes)
                self.write_dataset_fixture(root, changed, legacy_error_sample_ids={'s'})
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    p.validate_dataset(root)

    def test_legacy_shape_cannot_exempt_synthetic_or_scored_failed_judges(self):
        legacy = {'sample_id': 's', 'judge_1_score': 0, 'judge_2_score': 1, 'judge_3_score': None,
                  'judge_3_error': 'Provider error', 'judge_3_status': 'error',
                  'row_errors': ['Provider error'], 'error': 'Provider error', 'status': 'error',
                  'consensus_score': 0.5, 'consensus_method': 'mean'}
        with self.assertRaises(ValueError):
            p.get_legacy_error_sample_ids([{**legacy, 'judge_3_score': 0, 'consensus_score': 0.3333}])
        for changes in [
            {'judge_3_justification': 'Fallback score: judge returned empty output after retries.'},
            {'judge_3_parse_mode': 'fallback_empty_judge_output'},
            {'judge_3_warnings': ['judge_fallback_score_on_empty_output']},
        ]:
            with self.subTest(changes=changes):
                self.assertEqual(p.get_legacy_error_sample_ids([{**legacy, **changes}]), set())

    def test_incoming_rejects_changed_answers_and_judge_identities(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args, response, aggregate = self.incoming_fixture(root)
            for changes in [
                {'response_text': 'An entirely different answer.'},
                {'judge_1_model': 'another judge'},
                {'judge_1_model': 'a', 'judge_2_model': 'a'},
            ]:
                (root / 'aggregate.jsonl').write_text(json.dumps({**aggregate, **changes}) + '\n')
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    p.validate_incoming(args, root / 'output')
            (root / 'aggregate.jsonl').write_text(json.dumps(aggregate) + '\n')
            for roster in [['a', 'a', 'a'], ['a', 'b', ''], 'abc', ['a', 'b', None]]:
                (root / 'panel.json').write_text(json.dumps({'judge_models': roster, 'panel_mode': 'full'}))
                with self.subTest(roster=roster), self.assertRaises(ValueError):
                    p.validate_incoming(args, root / 'output')

    def test_full_panel_rejects_present_contradictory_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = {'sample_id': 's', 'model': 'example/model', 'question_id': 'q',
                    'run_index': 1, 'response_text': 'The premise is invalid.',
                    'judge_1_score': 2, 'judge_2_score': 1, 'judge_3_score': 2,
                    'consensus_score': 1.6667}
            for changes in [
                {'response_refusal': True, 'response_outcome': 'refusal'},
                {'response_refusal': False, 'response_outcome': 'refusal'},
                {'response_outcome': 'error'}, {'source_response_error': 'Collection failed'},
                {'row_identity_mismatch': True}, {'judge_1_row_present': False},
                {'judge_1_status': 'pending'}, {'status': 'pending'},
                {'consensus_method': 'median'}, {'judge_failure_policy': 'retry_then_two_valid'},
                {'judge_1_model': 'same', 'judge_2_model': 'same', 'judge_3_model': 'same'},
                {'judge_valid_scores': [2, 2, 2]}, {'judge_valid_count': 2},
            ]:
                self.write_dataset_fixture(root, {**base, **changes})
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    p.validate_dataset(root)

    def test_refusal_preserves_genuine_votes_without_scored_coverage(self):
        response = {'sample_id': 's', 'model': 'example/model', 'question_id': 'q',
                    'run_index': 1, 'response_text': '', 'response_refusal': True,
                    'response_outcome': 'refusal'}
        aggregate = {**response, 'judge_1_score': None, 'judge_2_score': 0, 'judge_3_score': 0,
                     'consensus_score': None, 'judge_valid_count': 0, 'judge_expected_count': 3,
                     'judge_coverage': '0/3 judges', 'judge_valid_scores': []}
        p.validate_judge_panel(response, aggregate, 'Fixture')
        for changes in [
            {'row_identity_mismatch': True}, {'response_refusal': False},
            {'judge_valid_count': 2}, {'judge_coverage': '2/3 judges'},
            {'judge_failure_policy': 'retry_then_two_valid'}, {'judge_valid_scores': [0, 0]},
            {'consensus_score': 0}, {'source_response_error': 'Collection failed'},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                p.validate_judge_panel(response, {**aggregate, **changes}, 'Fixture')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write_dataset_fixture(root, aggregate, response_changes={
                'response_refusal': True, 'response_outcome': 'refusal'})
            self.assertEqual(p.validate_dataset(root)['rows'], 1)

    def test_exhausted_two_judge_panel_requires_explicit_matching_coverage(self):
        from openrouter_benchmark import finalize_judge_consensus
        response = {'sample_id': 's', 'model': 'example/model', 'question_id': 'q',
                    'run_index': 1, 'response_text': 'The premise is invalid.'}
        raw = {**response, 'judge_1_score': 2, 'judge_2_score': 1, 'judge_3_score': None,
               'judge_3_error': 'Judge returned empty output', 'judge_3_status': 'error',
               'judge_3_attempt_count': 3, 'judge_3_attempts': [{'error': 'Empty output'}] * 3}
        aggregate = finalize_judge_consensus(raw)
        p.validate_judge_panel(response, aggregate, 'Fixture')
        for changes in [
            {'judge_3_attempt_count': 2}, {'judge_3_attempts': [{}, {}, {}]},
            {'judge_valid_count': 3}, {'judge_expected_count': 2}, {'judge_coverage': '3/3 judges'},
            {'judge_failure_policy': None}, {'judge_2_score': None}, {'consensus_score': 2},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                p.validate_judge_panel(response, {**aggregate, **changes}, 'Fixture')
        historical = {**raw, 'judge_3_parse_mode': 'fallback_empty_judge_output'}
        historical.pop('judge_3_attempts'); historical.pop('judge_3_attempt_count')
        corrected = finalize_judge_consensus(historical)
        self.assertEqual(p.get_legacy_error_sample_ids([corrected]), set())
        p.validate_judge_panel(response, corrected, 'Fixture')

    def test_viewer_cannot_hide_canonical_error_or_change_judge_coverage(self):
        aggregate = {'sample_id': 's', 'model': 'example/model', 'question_id': 'q',
                     'run_index': 1, 'response_text': 'An answer.',
                     'judge_1_score': 2, 'judge_2_score': 2, 'judge_3_score': None,
                     'judge_3_error': 'Provider error', 'judge_3_status': 'error',
                     'row_errors': ['Provider error'], 'error': 'Provider error', 'status': 'error',
                     'consensus_score': 2, 'consensus_method': 'mean'}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write_dataset_fixture(root, aggregate, legacy_error_sample_ids={'s'})
            for changes in [
                {'status': 'ok', 'error': '', 'row_errors': []}, {'error': ''},
                {'response_outcome': 'refusal'}, {'judge_3_status': 'ok'}, {'judge_3_error': ''},
                {'judge_valid_count': 2, 'judge_expected_count': 3, 'judge_coverage': '2/3 judges',
                 'judge_failure_policy': 'retry_then_two_valid'},
            ]:
                (root / 'viewer_rows.json.gz').write_bytes(gzip.compress(
                    json.dumps([{**aggregate, **changes}]).encode()))
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'Viewer summary mismatch'):
                    p.validate_dataset(root)

    def test_retention_protects_current_and_previous_content_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'responses').mkdir(); (root/'releases').mkdir(); (root/'metadata').mkdir()
            names = [f'responses/sha256-{str(i) * 64}.jsonl' for i in range(3)]
            question_names = [f'metadata/{str(i) * 64}-questions.json' for i in range(3)]
            for name in names: (root/name).write_text('fixture')
            for name in question_names: (root/name).write_text('{}')
            previous = {'storage': {'assets': {'responses.jsonl': {'parts': [{'path': names[1]}]}}},
                        'files': {'questions.json': {'path': question_names[1]}}}
            raw = json.dumps(previous).encode(); path = f'releases/{p.sha(raw)}.json'; (root/path).write_bytes(raw)
            current = {'storage': {'assets': {'responses.jsonl': {'parts': [{'path': names[0]}]}}},
                       'files': {'questions.json': {'path': question_names[0]}},
                       'previous_manifest': {'path': path, 'bytes': len(raw), 'sha256': p.sha(raw)}}
            (root/'manifest.json').write_text(json.dumps(current))
            (root/'responses/unrelated.txt').write_text('preserve')
            self.assertEqual(p.stale_files(root), sorted([root/names[2], root/question_names[2]]))

    def test_question_shapes_preserve_metadata_and_exclude_controls_consistently(self):
        grouped = self.question_document()
        grouped['techniques'].append({'technique': 'control_legitimate', 'questions': [{
            'id': 'control', 'question': 'A legitimate control?', 'nonsensical_element': '', 'domain': 'Control',
        }]})
        definitions = p.question_definitions(json.dumps(grouped))
        snapshot = p.question_definitions(json.dumps(self.question_snapshot()))
        p.compare_questions(definitions, snapshot, 'Snapshot')
        self.assertEqual(set(p.benchmark_questions(definitions)), {'q'})
        self.assertEqual(definitions['q']['domain_group'], 'fixture')
        self.assertEqual(definitions['q']['question'], snapshot['q']['question'])
        duplicate = self.question_snapshot() * 2
        with self.assertRaisesRegex(ValueError, 'duplicate question ID'):
            p.question_definitions(json.dumps(duplicate))

    def test_incoming_requires_original_snapshot_or_explicit_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args, _, _ = self.incoming_fixture(root, snapshot=False)
            with self.assertRaisesRegex(ValueError, 'snapshot is missing'):
                p.validate_incoming(args, root / 'output')
            source = root / 'verified-source.json'
            raw = json.dumps(self.question_document(), indent=3).encode() + b'\n'
            source.write_bytes(raw); args.questions_file = source
            self.assertEqual(p.validate_incoming(args, root / 'output'), raw)

    def test_incoming_question_text_and_annotations_match_collected_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args, response, aggregate = self.incoming_fixture(root)
            self.assertEqual(p.validate_incoming(args, root / 'output'), (root / 'questions_snapshot.json').read_bytes())
            for field in ('question', 'domain', 'nonsensical_element', 'technique'):
                (root / 'responses.jsonl').write_text(json.dumps({**response, field: 'Changed after collection'}) + '\n')
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, f'changed {field}'):
                    p.validate_incoming(args, root / 'output')
            (root / 'responses.jsonl').write_text(json.dumps(response) + '\n')
            (root / 'aggregate.jsonl').write_text(json.dumps({**aggregate, 'question': 'Wrong graded question?'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'Incoming aggregates: changed question'):
                p.validate_incoming(args, root / 'output')

    def test_explicit_source_cannot_override_disagreeing_collection_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args, _, _ = self.incoming_fixture(root)
            doc = self.question_document(); doc['techniques'][0]['questions'][0]['question'] = 'Changed?'
            args.questions_file = root / 'questions.json'
            args.questions_file.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'Explicit question source: changed question'):
                p.validate_incoming(args, root / 'output')

    def test_existing_snapshot_is_preserved_across_compatible_supplements(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args, _, aggregate = self.incoming_fixture(root)
            destination = root / 'published'; destination.mkdir()
            self.write_dataset_fixture(destination, aggregate)
            original = (destination / 'questions.json').read_bytes()
            self.assertEqual(p.validate_incoming(args, destination), original)
            # The collection's flat snapshot is compatible but cannot erase the
            # grouped source's domain_group/difficulty annotations.
            self.assertIsInstance(json.loads(p.validate_incoming(args, destination)), dict)
            for mode in ('supplemental', 'replace'):
                args.publish_mode = mode
                changed = self.question_snapshot(); changed[0]['question'] = 'Changed collection question?'
                (root / 'questions_snapshot.json').write_text(json.dumps(changed))
                for name in ('responses.jsonl', 'aggregate.jsonl'):
                    row = json.loads((root / name).read_text()); row['question'] = changed[0]['question']
                    (root / name).write_text(json.dumps(row) + '\n')
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Incoming questions: changed question'):
                    p.validate_incoming(args, destination)

    def test_existing_snapshot_rejects_changed_ids_even_when_incoming_rows_agree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args, _, aggregate = self.incoming_fixture(root)
            destination = root / 'published'; destination.mkdir()
            self.write_dataset_fixture(destination, aggregate)
            changed = self.question_snapshot(); changed[0]['id'] = 'new-id'
            (root / 'questions_snapshot.json').write_text(json.dumps(changed))
            for name in ('responses.jsonl', 'aggregate.jsonl'):
                row = json.loads((root / name).read_text()); row['question_id'] = 'new-id'
                (root / name).write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'question IDs differ'):
                p.validate_incoming(args, destination)

    def test_dataset_questions_are_immutable_manifest_content(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, _, aggregate = self.incoming_fixture(root)
            self.write_dataset_fixture(root, aggregate)
            stale_manifest = json.loads((root / 'manifest.json').read_text())
            frozen = p.frozen_questions(root, stale_manifest)[0]
            # A mutable convenience alias or later root edit cannot affect a
            # reader that already has this manifest.
            (root / 'questions.json').write_text('{"changed":true}')
            self.assertEqual(p.frozen_questions(root, stale_manifest)[0], frozen)
            self.assertEqual(p.validate_dataset(root)['questions_sha256'], p.sha(frozen))
            descriptor = stale_manifest['files']['questions.json']
            (root / descriptor['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                p.validate_dataset(root)
            (root / 'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'no frozen questions'):
                p.validate_dataset(root)

    def test_legacy_migration_requires_explicit_source_and_preserves_response_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / 'run'; run.mkdir()
            _, _, aggregate = self.incoming_fixture(run)
            destination = root / 'published'; destination.mkdir()
            self.write_dataset_fixture(destination, aggregate)
            manifest = {'corrections': [{'source_sha256': 'fixture-source', 'reason': 'Verified repair'}]}
            (destination / 'manifest.json').write_text(json.dumps(manifest))
            for name in p.SIDECARS:
                path = destination / name
                if not path.exists(): path.write_text('{}' if name.endswith('.json') else 'model\n')
            before = (destination / 'responses.jsonl').read_bytes()
            args = SimpleNamespace(questions_file=None)
            with self.assertRaisesRegex(ValueError, 'Legacy migration requires --questions-file'):
                p.migration_questions(args, destination)
            source = root / 'verified-original.json'
            source.write_text(json.dumps(self.question_document(), ensure_ascii=False, indent=3) + '\n')
            raw = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(p.main(['migrate', '--output-dir', str(destination), '--questions-file', str(source)]), 0)
            migrated_raw = (destination / 'manifest.json').read_bytes()
            migrated = json.loads(migrated_raw)
            self.assertEqual(p.checked_file(destination, migrated['files']['questions.json']), raw)
            self.assertEqual(p.storage.read_bytes(destination / 'responses.jsonl'), before)
            self.assertEqual(migrated['corrections'], manifest['corrections'])
            self.assertEqual(p.validate_dataset(destination)['questions'], 1)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(p.main(['migrate', '--output-dir', str(destination)]), 0)
            self.assertEqual((destination / 'manifest.json').read_bytes(), migrated_raw)
            changed = self.question_document(); changed['techniques'][0]['questions'][0]['question'] = 'New wording?'
            source.write_text(json.dumps(changed))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(p.main(['migrate', '--output-dir', str(destination), '--questions-file', str(source)]), 1)
            self.assertEqual((destination / 'manifest.json').read_bytes(), migrated_raw)

    def test_legacy_migration_rejects_incompatible_question_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, _, aggregate = self.incoming_fixture(root)
            self.write_dataset_fixture(root, aggregate)
            (root / 'manifest.json').write_text('{}')
            source = root / 'verified-original.json'
            changed = self.question_document(); changed['techniques'][0]['questions'][0]['id'] = 'other'
            source.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'unknown benchmark question ID'):
                p.migration_questions(SimpleNamespace(questions_file=source), root)

    def test_publish_cli_freezes_collection_snapshot_and_retains_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / 'run'; run.mkdir()
            args, _, _ = self.incoming_fixture(run)
            collection = run / 'collection_stats.json'; collection.write_text('{}')
            destination = root / 'published'
            command = [sys.executable, str(ROOT / 'scripts/publication.py'), 'publish',
                       '--responses-file', str(args.responses_file), '--collection-stats', str(collection),
                       '--panel-summary', str(args.panel_summary), '--aggregate-summary', str(args.aggregate_summary),
                       '--aggregate-rows', str(args.aggregate_rows), '--output-dir', str(destination)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((destination / 'manifest.json').read_text())
            snapshot = (run / 'questions_snapshot.json').read_bytes()
            self.assertEqual(p.checked_file(destination, manifest['files']['questions.json']), snapshot)
            self.assertIsInstance(json.loads(snapshot), list)
            self.assertEqual(p.validate_dataset(destination)['rows'], 1)
            self.assertNotIn('question', p.json_rows(destination / 'responses.jsonl')[0])
            manifest['corrections'] = [{'reason': 'Verified recovery', 'source_sha256': 'fixture'}]
            (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            stale_manifest = manifest
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            current = json.loads((destination / 'manifest.json').read_text())
            self.assertEqual(current['corrections'], manifest['corrections'])
            self.assertEqual(p.checked_file(destination, stale_manifest['files']['questions.json']), snapshot)
            self.assertEqual(p.checked_file(destination, current['files']['questions.json']), snapshot)

    def test_metadata_path_cannot_escape_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                p.checked_file(Path(d), {'path': '../outside', 'bytes': 0, 'sha256': p.sha(b'')})


if __name__ == '__main__':
    unittest.main()

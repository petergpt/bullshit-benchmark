#!/usr/bin/env python3
"""Validate, stage and activate GitHub Pages benchmark datasets without API calls."""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import published_dataset as storage

ROOT = Path(__file__).resolve().parents[1]
SIDECARS = (
    'collection_stats.json', 'panel_summary.json', 'aggregate_summary.json',
    'recent_additions.json', 'leaderboard.csv', 'leaderboard_with_launch.csv',
    'model_launch_dates.csv', 'model_params.csv', 'questions.json',
)
IDENTITY = ('model', 'question_id', 'run_index')
QUESTION_FIELDS = ('question', 'nonsensical_element', 'domain', 'technique', 'is_control')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Non-JSON number: {value}')))


def json_rows(path):
    text = storage.read_text(path)
    result = []
    seen = set()
    for number, line in enumerate(text.split('\n'), 1):
        if not line.strip():
            continue
        try:
            row = decode(line)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}: invalid JSONL line {number}') from exc
        if not isinstance(row, dict) or not isinstance(row.get('sample_id'), str) or not row['sample_id'].strip():
            raise ValueError(f'{path}: record {number} has no sample_id')
        if row['sample_id'] in seen:
            raise ValueError(f'{path}: duplicate sample_id {row["sample_id"]}')
        seen.add(row['sample_id'])
        result.append(row)
    return result


def checked_file(root, descriptor):
    relative = descriptor.get('path')
    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Unsafe metadata path')
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe metadata file')
    raw = path.read_bytes()
    if len(raw) != descriptor.get('bytes') or sha(raw) != descriptor.get('sha256'):
        raise ValueError(f'Metadata checksum mismatch: {relative}')
    return raw


def sidecar_bytes(root, name, manifest):
    descriptor = manifest.get('files', {}).get(name)
    return checked_file(root, descriptor) if descriptor else (root / name).read_bytes()


def question_definitions(raw, label='Questions'):
    """Index a grouped source or collection snapshot without rewriting its bytes.

    Collection snapshots flatten techniques and omit source-only annotations.
    The common fields are compared exactly, while a publication keeps its full
    original document, including annotations not copied into collection rows.
    """
    payload = decode(raw)
    entries = []
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and isinstance(payload.get('techniques'), list):
        for group in payload['techniques']:
            if not isinstance(group, dict) or not isinstance(group.get('questions'), list):
                raise ValueError(f'{label}: invalid question technique')
            technique = group.get('technique')
            if not isinstance(technique, str) or not technique.strip():
                raise ValueError(f'{label}: missing question technique')
            for question in group['questions']:
                if not isinstance(question, dict):
                    raise ValueError(f'{label}: question must be an object')
                if 'technique' in question and question['technique'] != technique:
                    raise ValueError(f'{label}: conflicting technique for {question.get("id")}')
                entries.append({**question, 'technique': technique,
                                'technique_description': group.get('description', '')})
    else:
        raise ValueError(f'{label}: expected a techniques document or a collection snapshot array')
    definitions = {}
    for question in entries:
        if not isinstance(question, dict):
            raise ValueError(f'{label}: question must be an object')
        for field in ('id', 'question', 'nonsensical_element', 'domain', 'technique'):
            if not isinstance(question.get(field), str) or (field in ('id', 'question', 'technique') and not question[field].strip()):
                raise ValueError(f'{label}: invalid question {field}')
        if 'is_control' in question and type(question['is_control']) is not bool:
            raise ValueError(f'{label}: invalid is_control for {question["id"]}')
        question = {**question, 'is_control': question.get('is_control', False) or question['technique'] == 'control_legitimate'}
        if question['id'] in definitions:
            raise ValueError(f'{label}: duplicate question ID {question["id"]}')
        definitions[question['id']] = question
    if not definitions or not any(not q['is_control'] for q in definitions.values()):
        raise ValueError(f'{label}: no benchmark questions')
    return definitions


def benchmark_questions(definitions):
    # Match collection's established exclusion of control questions.
    return {key: question for key, question in definitions.items() if not question['is_control']}


def compare_questions(expected, incoming, label):
    expected, incoming = benchmark_questions(expected), benchmark_questions(incoming)
    if expected.keys() != incoming.keys():
        raise ValueError(f'{label}: question IDs differ from the frozen question set')
    for question_id, question in expected.items():
        for field in question.keys() & incoming[question_id].keys():
            if question[field] != incoming[question_id][field]:
                raise ValueError(f'{label}: changed {field} for question {question_id}')


def validate_question_rows(rows, definitions, label, *, complete=False):
    expected = benchmark_questions(definitions)
    found = set()
    for row in rows:
        question_id = row.get('question_id')
        if not isinstance(question_id, str) or question_id not in expected:
            raise ValueError(f'{label}: unknown benchmark question ID {question_id!r}')
        found.add(question_id)
        question = expected[question_id]
        for field in QUESTION_FIELDS:
            if field in row and row[field] != question[field]:
                raise ValueError(f'{label}: changed {field} for question {question_id}')
    if complete and found != expected.keys():
        raise ValueError(f'{label}: question coverage differs from the frozen question set')


def frozen_questions(root, manifest):
    descriptor = manifest.get('files', {}).get('questions.json')
    if not descriptor:
        raise ValueError('Dataset has no frozen questions; migrate with --questions-file after verifying the original source')
    raw = checked_file(root, descriptor)
    return raw, question_definitions(raw, 'Frozen questions')


def incoming_questions(args, destination, responses, aggregates):
    snapshot = Path(args.responses_file).resolve().parent / 'questions_snapshot.json'
    explicit = getattr(args, 'questions_file', None)
    source = Path(explicit) if explicit else snapshot
    if not source.is_file():
        raise ValueError('Incoming questions snapshot is missing; supply --questions-file with the original question source')
    raw = source.read_bytes()
    definitions = question_definitions(raw, 'Incoming questions')
    if explicit and snapshot.is_file() and source.resolve() != snapshot.resolve():
        compare_questions(question_definitions(snapshot.read_bytes(), 'Collection snapshot'), definitions,
                          'Explicit question source')
    validate_question_rows(responses, definitions, 'Incoming responses', complete=True)
    validate_question_rows(aggregates, definitions, 'Incoming aggregates', complete=True)
    if storage.dataset_exists(destination):
        manifest = decode((destination / 'manifest.json').read_bytes())
        old_raw, old_definitions = frozen_questions(destination, manifest)
        compare_questions(old_definitions, definitions, 'Incoming questions')
        # In particular, keep grouped source annotations when a flattened run
        # snapshot lacks them. Even replace cannot silently change the benchmark.
        return old_raw
    return raw


def migration_questions(args, destination):
    manifest = decode((destination / 'manifest.json').read_bytes())
    explicit = getattr(args, 'questions_file', None)
    if manifest.get('files', {}).get('questions.json'):
        raw, definitions = frozen_questions(destination, manifest)
        if explicit:
            compare_questions(definitions, question_definitions(Path(explicit).read_bytes(), 'Question source'),
                              'Migration question source')
    else:
        if not explicit:
            raise ValueError('Legacy migration requires --questions-file with a source verified against original collection artifacts')
        raw = Path(explicit).read_bytes()
        definitions = question_definitions(raw, 'Migration questions')
    # Old slimmed rows no longer contain wording. The explicit source is an
    # intentional one-time choice, never an implicit lookup of a mutable file.
    for name in ('responses.jsonl', 'aggregate.jsonl'):
        validate_question_rows(json_rows(destination / name), definitions, name, complete=True)
    return raw


def has_synthetic_judge_marker(row, index):
    prefix = f'judge_{index}'
    warnings = row.get(f'{prefix}_warnings') or []
    if isinstance(warnings, str):
        warnings = [warnings]
    # Older aggregates retained only this justification. Ordinary retry warnings
    # remain valid when a later attempt succeeded.
    return (
        row.get(f'{prefix}_parse_mode') == 'fallback_empty_judge_output'
        or any('judge_fallback_score_on_empty_output' in str(warning)
               or 'fallback_empty_judge_output' in str(warning) for warning in warnings)
        or str(row.get(f'{prefix}_justification') or '').startswith(
            'Fallback score: judge returned empty output')
    )


def validate_no_synthetic_judge_scores(row, label):
    """Reject invented scores even when a historical row is explicitly errored."""
    sample = row['sample_id']
    for index in (1, 2, 3):
        prefix = f'judge_{index}'
        if has_synthetic_judge_marker(row, index) and row.get(f'{prefix}_score') is not None:
            raise ValueError(f'{label} sample has a synthetic empty-output judge score: {sample} ({prefix})')
        failed = (row.get(f'{prefix}_error') or row.get(f'{prefix}_status') == 'error'
                  or row.get(f'{prefix}_finish_reason') in {'content_filter', 'refusal', 'length', 'max_output_tokens'})
        if failed and row.get(f'{prefix}_score') is not None:
            raise ValueError(f'{label} sample has a scored failed judge: {sample} ({prefix})')


def get_legacy_error_sample_ids(rows):
    """Recognize explicitly recorded errors from publications before panel IDs.

    This compatibility set is only for existing published rows. New imports
    still pass the strict panel gate, and synthesized fallback scores never
    qualify. Historical partial consensus is retained as an errored record.
    """
    sample_ids = set()
    for row in rows:
        validate_no_synthetic_judge_scores(row, 'Dataset')
        # Corrected fallback rows must pass the current explicit coverage policy;
        # a null replacement alone does not grant the legacy-error exemption.
        if any(has_synthetic_judge_marker(row, index) for index in (1, 2, 3)):
            continue
        current_judge_metadata = any(f'judge_{index}_{detail}' in row
                                     for index in (1, 2, 3)
                                     for detail in ('parse_mode', 'warnings', 'finish_reason', 'attempt_count'))
        if ('judge_panel_id' in row or current_judge_metadata or row.get('status') != 'error' or not row.get('error')
                or not isinstance(row.get('row_errors'), list) or not row['row_errors']
                or row.get('consensus_method') != 'mean' or row.get('row_identity_mismatch')):
            continue
        valid_scores = []
        failed_count = 0
        for index in (1, 2, 3):
            prefix = f'judge_{index}'
            score = row.get(f'{prefix}_score')
            if row.get(f'{prefix}_error') and row.get(f'{prefix}_status') == 'error' and score is None:
                failed_count += 1
            elif type(score) is int and score in (0, 1, 2, 3) and not row.get(f'{prefix}_error'):
                valid_scores.append(score)
            else:
                break
        else:
            consensus = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None
            expected_error = (None, '') if valid_scores else ('no_valid_scores',)
            if failed_count and row.get('consensus_score') == consensus and row.get('consensus_error') in expected_error:
                sample_ids.add(row['sample_id'])
    return sample_ids


def validate_judge_panel(response, row, label, legacy_error_sample_ids=frozenset()):
    """Require actual votes, or an explicitly labelled exhausted-judge fallback."""
    sample = row['sample_id']
    validate_no_synthetic_judge_scores(row, label)
    from openrouter_benchmark import finalize_judge_consensus, response_is_refusal
    refused = response_is_refusal(response)
    if response_is_refusal(row) != refused:
        raise ValueError(f'{label} response/aggregate refusal classification differs: {sample}')
    for source in (response, row):
        outcome = source.get('response_outcome')
        if outcome in ('response', 'refusal') and (outcome == 'refusal') != response_is_refusal(source):
            raise ValueError(f'{label} response outcome contradicts refusal classification: {sample}')
    if (response.get('response_outcome') and row.get('response_outcome')
            and response['response_outcome'] != row['response_outcome']):
        raise ValueError(f'{label} response/aggregate outcome differs: {sample}')
    if 'response_text' in row and row['response_text'] != response.get('response_text', ''):
        raise ValueError(f'{label} response/aggregate answer text differs: {sample}')
    if row.get('row_identity_mismatch'):
        raise ValueError(f'{label} sample has identity errors: {sample}')
    if sample in legacy_error_sample_ids:
        return
    if (response.get('error') or response.get('status') == 'error' or row.get('source_response_error')
            or response.get('response_outcome') == 'error' or row.get('response_outcome') == 'error'):
        raise ValueError(f'{label} sample has collection errors: {sample}')
    if (row.get('row_errors') or row.get('error') or row.get('consensus_error')
            or row.get('status') not in (None, '', 'ok')):
        raise ValueError(f'{label} sample has unresolved errors: {sample}')
    if row.get('consensus_method') not in (None, '', 'mean'):
        raise ValueError(f'{label} sample has non-mean consensus method: {sample}')
    models = [row[f'judge_{index}_model'] for index in (1, 2, 3) if f'judge_{index}_model' in row]
    if any(not isinstance(model, str) or not model.strip() for model in models) or len(set(models)) != len(models):
        raise ValueError(f'{label} sample has invalid or duplicate judge identities: {sample}')
    scores = [row.get(f'judge_{index}_score') for index in (1, 2, 3)]
    valid_scores = []
    for index, score in enumerate(scores, 1):
        prefix = f'judge_{index}'
        if row.get(f'{prefix}_row_present') is False or row.get(f'{prefix}_status') not in (None, '', 'ok', 'error'):
            raise ValueError(f'{label} sample has missing or unresolved judge row: {sample} ({prefix})')
        failed = (row.get(f'{prefix}_error') or row.get(f'{prefix}_status') == 'error'
                  or row.get(f'{prefix}_finish_reason') in {'content_filter', 'refusal', 'length', 'max_output_tokens'}
                  or type(score) is not int or score not in (0, 1, 2, 3))
        if not failed:
            valid_scores.append(score)
    if not refused:
        expected = finalize_judge_consensus(row, num_judges=3, consensus_method='mean')
        if expected.get('status') != 'ok':
            raise ValueError(f'{label} sample has an invalid judge panel: {sample}')
        if len(valid_scores) != 3:
            if (len(valid_scores) != 2 or expected.get('status') != 'ok'
                    or row.get('judge_failure_policy') != 'retry_then_two_valid'
                    or any(row.get(field) != expected.get(field) for field in (
                        'consensus_score', 'consensus_method', 'judge_valid_count',
                        'judge_expected_count', 'judge_coverage', 'judge_failure_policy'))):
                raise ValueError(f'{label} sample has incomplete judge panel or invalid fallback: {sample}')
        elif row.get('judge_failure_policy'):
            raise ValueError(f'{label} full judge panel has a fallback label: {sample}')
        if row.get('consensus_score') != round(sum(valid_scores) / len(valid_scores), 4):
            raise ValueError(f'{label} consensus differs from judge scores: {sample}')
        for field, expected in (('judge_valid_count', len(valid_scores)), ('judge_expected_count', 3),
                                ('judge_coverage', f'{len(valid_scores)}/3 judges'), ('judge_valid_scores', valid_scores)):
            if field in row and row[field] != expected:
                raise ValueError(f'{label} judge coverage differs: {sample} ({field})')
    if refused and row.get('consensus_score') is not None:
        raise ValueError(f'Refusal sample has a scored consensus: {sample}')
    if refused:
        # Old genuine judge votes remain available for audit, but candidate
        # refusals never use those votes or acquire a two-judge score label.
        if row.get('judge_failure_policy') or row.get('judge_valid_scores'):
            raise ValueError(f'Refusal sample has a scored judge fallback: {sample}')
        for field, expected in (('judge_valid_count', 0), ('judge_expected_count', 3), ('judge_coverage', '0/3 judges')):
            if field in row and row[field] != expected:
                raise ValueError(f'Refusal judge coverage differs: {sample} ({field})')


def validate_dataset(root):
    root = Path(root)
    if not storage.dataset_exists(root):
        raise ValueError(f'No dataset: {root}')
    manifest = decode((root / 'manifest.json').read_bytes())
    for descriptor in manifest.get('files', {}).values():
        checked_file(root, descriptor)
    questions_raw, questions = frozen_questions(root, manifest)
    responses = {r['sample_id']: r for r in json_rows(root / 'responses.jsonl')}
    aggregate = {r['sample_id']: r for r in json_rows(root / 'aggregate.jsonl')}
    if responses.keys() != aggregate.keys():
        raise ValueError('Responses and aggregates have different sample IDs')
    validate_question_rows(responses.values(), questions, 'Dataset responses', complete=True)
    validate_question_rows(aggregate.values(), questions, 'Dataset aggregates', complete=True)
    legacy_error_sample_ids = get_legacy_error_sample_ids(aggregate.values())
    for sample, row in aggregate.items():
        if any(row.get(key) != responses[sample].get(key) for key in IDENTITY):
            raise ValueError(f'Response/aggregate identity mismatch: {sample}')
        validate_judge_panel(responses[sample], row, 'Dataset', legacy_error_sample_ids)
    summaries = decode(storage.read_bytes(root / 'viewer_rows.json.gz'))
    details = decode(storage.read_bytes(root / 'viewer_details.json.gz'))
    for label, rows in [('summary', summaries), ('details', details)]:
        if len(rows) != len(responses) or {x['sample_id'] for x in rows} != responses.keys():
            raise ValueError(f'{label} sample coverage differs')
        # Detail rows intentionally carry only sample_id; their question is
        # identified by the matching canonical response and part descriptor.
        validate_question_rows(({'question_id': responses[row['sample_id']]['question_id'], **row}
                                for row in rows), questions, f'Viewer {label}')
    for row in summaries:
        agg = aggregate[row['sample_id']]
        for key in ('model', 'question_id', 'consensus_score', 'response_refusal', 'judge_1_score', 'judge_2_score', 'judge_3_score',
                    'judge_valid_count', 'judge_expected_count', 'judge_coverage', 'judge_failure_policy',
                    'status', 'error', 'response_outcome', 'row_errors', 'consensus_error',
                    'judge_1_status', 'judge_2_status', 'judge_3_status',
                    'judge_1_error', 'judge_2_error', 'judge_3_error'):
            if row.get(key) != agg.get(key):
                raise ValueError(f'Viewer summary mismatch: {key}')
    for row in details:
        if row.get('response_text', '') != responses[row['sample_id']].get('response_text', ''):
            raise ValueError(f'Viewer response text mismatch: {row["sample_id"]}')
    leaderboard = list(csv.DictReader(io.StringIO(sidecar_bytes(root, 'leaderboard.csv', manifest).decode())))
    variants = {row['model'] for row in responses.values()}
    if len(leaderboard) != len(variants) or {x['model'] for x in leaderboard} != variants:
        raise ValueError('Leaderboard model coverage differs')
    summary = decode(sidecar_bytes(root, 'aggregate_summary.json', manifest))
    if summary.get('total_records') != len(responses):
        raise ValueError('Aggregate summary row count differs')
    from openrouter_benchmark import summarize_aggregate_rows
    expected_summary = summarize_aggregate_rows(list(aggregate.values()), consensus_method='mean', num_judges=3,
                                              legacy_error_sample_ids=legacy_error_sample_ids)
    if summary != expected_summary:
        raise ValueError('Aggregate summary differs from canonical judge rows')
    by_model = {row['model']: row for row in expected_summary['leaderboard']}
    fields = {'green_rate': 'detection_rate_score_2', 'red_rate': 'full_engagement_rate_score_0',
              'avg_score': 'avg_score', 'refusal_rate': 'refusal_rate', 'refusal_count': 'refusal_count',
              'score_0': 'score_0', 'score_1': 'score_1', 'score_2': 'score_2',
              'answered_count': 'answered_count', 'nonsense_count': 'nonsense_count', 'error_count': 'error_count'}
    for row in leaderboard:
        for key, summary_key in fields.items():
            expected = by_model[row['model']].get(summary_key)
            if (row.get(key) or '') != ('' if expected is None else str(expected)):
                raise ValueError(f'Leaderboard metric differs: {row["model"]} {key}')
    parts = [part for asset in manifest.get('storage', {}).get('assets', {}).values() for part in asset['parts']]
    return {'rows': len(responses), 'variants': len(variants), 'parts': len(parts),
            'questions': len(benchmark_questions(questions)), 'questions_sha256': sha(questions_raw),
            'max_part_bytes': max((p['bytes'] for p in parts), default=0),
            'active_asset_bytes': sum(p['bytes'] for p in parts),
            'responses_sha256': sha(storage.read_bytes(root / 'responses.jsonl')),
            'aggregate_sha256': sha(storage.read_bytes(root / 'aggregate.jsonl'))}


@contextlib.contextmanager
def writer_lock(destination):
    # A persistent lock inode in the system temp directory also covers two clones
    # targeting the same resolved destination. It is deliberately never unlinked.
    path = Path(tempfile.gettempdir()) / ('bullshitbench-publish-' + sha(str(destination.resolve()).encode()) + '.lock')
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Another publisher is writing {destination}') from exc
        directory_fd = None
        try:
            if destination.exists():
                directory_fd = os.open(destination, os.O_RDONLY)
                fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
            fcntl.flock(lock, fcntl.LOCK_UN)


def materialize(source, stage):
    if not storage.dataset_exists(source):
        return
    manifest = decode((source / 'manifest.json').read_bytes())
    for name in storage.ASSET_FORMATS:
        raw = storage.read_bytes(source / name)
        (stage / name).write_bytes(gzip.compress(raw, mtime=0) if name.endswith('.gz') else raw)
    for name in SIDECARS:
        if name == 'questions.json' and not manifest.get('files', {}).get(name):
            continue  # Legacy migration must supply a verified source explicitly.
        (stage / name).write_bytes(sidecar_bytes(source, name, manifest))
    manifest.pop('storage', None)
    manifest.pop('files', None)
    manifest.pop('previous_manifest', None)
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def freeze_metadata(stage, destination):
    manifest = decode((stage / 'manifest.json').read_bytes())
    manifest['files'] = {}
    try:
        prefix = str(destination.relative_to(ROOT))
    except ValueError:
        prefix = destination.name
    for name in SIDECARS:
        raw = (stage / name).read_bytes()
        if len(raw) > storage.MAX_FILE_BYTES:
            raise ValueError(f'Metadata file exceeds publication limit: {name}')
        relative = f'metadata/{sha(raw)}-{name}'
        path = stage / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        manifest['files'][name] = {'path': relative, 'bytes': len(raw), 'sha256': sha(raw)}
    # Legacy convenience download URLs remain stable; manifest-aware readers use
    # only the immutable descriptors, so reloading cannot mix two publications.
    manifest['sources'] = {}
    for logical, key in storage.SOURCE_KEYS.items():
        manifest['sources'][key + 's'] = [f'{prefix}/{p["path"]}' for p in manifest['storage']['assets'][logical]['parts']]
    for name, key in [('collection_stats.json', 'collection_stats_file'), ('panel_summary.json', 'panel_summary_file'),
                      ('aggregate_summary.json', 'aggregate_summary_file'), ('recent_additions.json', 'recent_additions_file'),
                      ('questions.json', 'questions_file')]:
        manifest['sources'][key] = f'{prefix}/{manifest["files"][name]["path"]}'
    manifest['exports'] = {name.removesuffix('.csv') + '_csv': f'{prefix}/{name}' for name in SIDECARS if name.endswith('.csv')}
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def referenced_files(manifest):
    paths = {p['path'] for asset in manifest.get('storage', {}).get('assets', {}).values() for p in asset['parts']}
    paths.update(p['path'] for p in manifest.get('files', {}).values())
    return paths


def stale_files(destination):
    manifest = decode((destination / 'manifest.json').read_bytes())
    protected = referenced_files(manifest)
    previous = manifest.get('previous_manifest')
    if previous:
        protected.add(previous['path'])
        protected.update(referenced_files(decode(checked_file(destination, previous))))
    stale = []
    patterns = {
        'responses': r'sha256-[0-9a-f]{64}\.jsonl',
        'aggregate': r'sha256-[0-9a-f]{64}\.jsonl',
        'viewer_rows': r'sha256-[0-9a-f]{64}\.json\.gz',
        'viewer_details': r'sha256-[0-9a-f]{64}\.json\.gz',
        'metadata': r'[0-9a-f]{64}-(?:' + '|'.join(re.escape(x) for x in SIDECARS) + ')',
        'releases': r'[0-9a-f]{64}\.json',
    }
    for directory, pattern in patterns.items():
        parent = destination / directory
        if parent.is_symlink():
            raise ValueError('Unsafe generated directory')
        if not parent.exists():
            continue
        for path in parent.iterdir():
            relative = str(path.relative_to(destination))
            if re.fullmatch(pattern, path.name) and relative not in protected:
                if path.is_symlink() or not path.is_file():
                    raise ValueError('Unsafe generated file')
                stale.append(path)
    return sorted(stale)


def activate(stage, destination, manifest):
    destination.mkdir(parents=True, exist_ok=True)
    current_path = destination / 'manifest.json'
    if current_path.exists():
        previous_raw = current_path.read_bytes()
        previous = decode(previous_raw)
        comparable = dict(previous)
        comparable.pop('previous_manifest', None)
        if comparable == manifest:
            return  # An identical migration must not churn files or retention.
        if previous.get('storage', {}).get('version') == 2 and previous.get('files'):
            relative = f'releases/{sha(previous_raw)}.json'
            path = destination / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(previous_raw)
            manifest['previous_manifest'] = {'path': relative, 'bytes': len(previous_raw), 'sha256': sha(previous_raw)}
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    targets = [p['path'] for asset in manifest['storage']['assets'].values() for p in asset['parts']]
    targets += [p['path'] for p in manifest['files'].values()]
    for relative in targets:
        source, target = stage / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f'Immutable content collision: {relative}')
        else:
            temporary = target.with_name(target.name + '.tmp')
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
    # Stable CSV/JSON downloads are convenience aliases. Readers of the new
    # manifest use immutable copies above, never these in-progress aliases.
    for name in SIDECARS:
        temporary = destination / ('.' + name + '.tmp')
        shutil.copy2(stage / name, temporary)
        os.replace(temporary, destination / name)
    temporary = destination / '.manifest.json.tmp'
    with temporary.open('wb') as stream:
        stream.write((stage / 'manifest.json').read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination / 'manifest.json')
    # Canonical monoliths are superseded only after the complete manifest is live.
    for name in storage.ASSET_FORMATS:
        (destination / name).unlink(missing_ok=True)
    # Keep current + immediately previous publication; older generated assets
    # remain recoverable from Git/run history and no longer inflate the live site.
    for path in stale_files(destination):
        path.unlink()


def validate_incoming(args, destination):
    responses = json_rows(args.responses_file)
    aggregates = json_rows(args.aggregate_rows)
    rmap, amap = {r['sample_id']: r for r in responses}, {r['sample_id']: r for r in aggregates}
    if not responses or rmap.keys() != amap.keys():
        raise ValueError('Incoming response/aggregate coverage is incomplete')
    panel = decode(Path(args.panel_summary).read_bytes())
    summary = decode(Path(args.aggregate_summary).read_bytes())
    judge_models = panel.get('judge_models')
    if (not isinstance(judge_models, list) or len(judge_models) != 3
            or any(not isinstance(model, str) or not model.strip() for model in judge_models)
            or len(set(judge_models)) != 3 or panel.get('panel_mode', 'full') != 'full'
            or summary.get('consensus_method') != 'mean'):
        raise ValueError('Publication requires three distinct configured judges and mean consensus')
    for sample, row in amap.items():
        response = rmap[sample]
        if any(row.get(key) != response.get(key) for key in IDENTITY):
            raise ValueError(f'Incoming identity mismatch: {sample}')
        for index, judge_model in enumerate(judge_models, 1):
            key = f'judge_{index}_model'
            if key in row and row[key] != judge_model:
                raise ValueError(f'Incoming judge identity differs from configured panel: {sample} ({key})')
        validate_judge_panel(response, row, 'Incoming')
    questions_raw = incoming_questions(args, destination, responses, aggregates)
    questions = question_definitions(questions_raw)
    config_name = {'data/latest': 'config.json', 'data/v2/latest': 'config.v2.json'}.get(str(destination.relative_to(ROOT)) if destination.is_relative_to(ROOT) else '')
    if config_name:
        from validate_configs import configured_variants
        config = decode((ROOT / config_name).read_bytes())['collect']
        allowed = configured_variants(ROOT / config_name)
        compare_questions(questions, question_definitions((ROOT / config['questions']).read_bytes(), 'Configured questions'),
                          'Configured questions')
        expected = benchmark_questions(questions).keys()
        for variant in {r['model'] for r in responses}:
            if variant not in allowed:
                raise ValueError(f'Published variant is absent from durable config: {variant}')
            if {r['question_id'] for r in responses if r['model'] == variant} != expected:
                raise ValueError(f'Incomplete question collection for {variant}')
    if args.publish_mode != 'replace' and storage.dataset_exists(destination):
        old_r = {r['sample_id']: r for r in json_rows(destination / 'responses.jsonl')}
        old_a = {r['sample_id']: r for r in json_rows(destination / 'aggregate.jsonl')}
        for sample in rmap.keys() & old_r.keys():
            if any(rmap[sample].get(k) != old_r[sample].get(k) for k in IDENTITY):
                raise ValueError(f'Existing sample ID belongs to another identity: {sample}')
            changed = old_r[sample].get('response_text', '') != rmap[sample].get('response_text', '')
            changed |= any(old_a[sample].get(k) != amap[sample].get(k) for k in ['consensus_score', 'judge_1_score', 'judge_2_score', 'judge_3_score'])
            if changed and not args.allow_replace_samples:
                raise ValueError(f'Refusing to replace changed sample {sample}; use --allow-replace-samples after review')
    return questions_raw


def publish(args):
    destination = Path(args.output_dir).resolve()
    with writer_lock(destination):
        if args.command == 'publish':
            questions_raw = validate_incoming(args, destination)
        else:
            questions_raw = migration_questions(args, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.publication-', dir=destination.parent) as tmp:
            stage = Path(tmp)
            materialize(destination, stage)
            (stage / 'questions.json').write_bytes(questions_raw)
            if args.command == 'publish':
                command = ['bash', str(ROOT / 'scripts/publish_latest_to_viewer.sh')]
                for option, value in [('responses-file', args.responses_file), ('collection-stats', args.collection_stats),
                                      ('panel-summary', args.panel_summary), ('aggregate-summary', args.aggregate_summary),
                                      ('aggregate-rows', args.aggregate_rows)]:
                    command += ['--' + option, str(Path(value).resolve())]
                command += ['--output-dir', str(stage), '--publish-mode', args.publish_mode]
                env = dict(os.environ, BULLSHITBENCH_PUBLISH_STAGE='1')
                subprocess.run(command, cwd=ROOT, env=env, check=True)
            storage.pack_dataset(stage)
            manifest = freeze_metadata(stage, destination)
            result = validate_dataset(stage)
            activate(stage, destination, manifest)
            print(json.dumps({'dataset': str(destination), **result}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('publish', help='stage, validate and activate completed run artifacts')
    for field in ['responses-file', 'collection-stats', 'panel-summary', 'aggregate-summary', 'aggregate-rows']:
        p.add_argument('--' + field, required=True, type=Path)
    p.add_argument('--output-dir', default='data/latest')
    p.add_argument('--publish-mode', choices=['auto', 'supplemental', 'replace'], default='auto')
    p.add_argument('--supplemental', dest='publish_mode', action='store_const', const='supplemental')
    p.add_argument('--replace', dest='publish_mode', action='store_const', const='replace')
    p.add_argument('--allow-replace-samples', action='store_true')
    p.add_argument('--questions-file', type=Path,
                   help='original question source; defaults to questions_snapshot.json beside responses')
    p = commands.add_parser('migrate', help='losslessly migrate an existing dataset without collecting or grading')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--questions-file', type=Path,
                   help='verified original question source, required when the existing publication has no snapshot')
    p = commands.add_parser('verify', help='verify hashes, row identities, viewer assets and metadata')
    p.add_argument('datasets', nargs='+', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify':
            files = [f for dataset in args.datasets for f in dataset.rglob('*') if f.is_file()]
            if any(f.stat().st_size >= 100 * 1024 * 1024 for f in files):
                raise ValueError('A dataset file reaches GitHub regular Git file-size limit')
            if sum(f.stat().st_size for f in files) > 900_000_000:
                raise ValueError('Published datasets exceed the 900 MB budget reserved under the Pages site limit')
            print(json.dumps({str(p): validate_dataset(p) for p in args.datasets}, indent=2))
        else:
            publish(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'publication: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

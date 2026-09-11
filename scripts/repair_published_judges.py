#!/usr/bin/env python3
"""Build an isolated, audited repair of a published dataset; never activate it.

Canonical grade directories (or their final grades.jsonl files) must include
grade_meta.json and exact response text. Failed or absent evaluations cannot
become scores. --write-blocked-draft records them as missing for review, still
fails publication validation, and exits nonzero.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import openrouter_benchmark as benchmark
import publication
import published_dataset as storage


METRICS = {
    'avg_score': 'avg_score', 'green_rate': 'detection_rate_score_2',
    'red_rate': 'full_engagement_rate_score_0', 'refusal_rate': 'refusal_rate',
    **{key: key for key in ('score_2', 'score_1', 'score_0', 'refusal_count',
                           'answered_count', 'nonsense_count', 'error_count')},
}
CONSENSUS_FIELDS = ('consensus_score', 'consensus_method', 'consensus_error',
                    'judge_valid_scores', 'status', 'error', 'row_errors')
GRADE_FIELDS = ('score', 'justification', 'parse_mode', 'warnings', 'finish_reason',
                'response_id', 'attempt_count')


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def keyed(rows, label):
    result = {}
    for row in rows:
        sample = row.get('sample_id')
        if not isinstance(sample, str) or not sample or sample in result:
            raise ValueError(f'{label}: missing or duplicate sample ID {sample!r}')
        result[sample] = row
    return result


def text_hash(row):
    if not isinstance(row.get('response_text'), str):
        raise ValueError(f'Missing exact response text: {row.get("sample_id")}')
    return publication.sha(row['response_text'].encode())


def same_response(response, incoming, label):
    for key in ('sample_id', *publication.IDENTITY):
        if key not in incoming or incoming[key] != response.get(key):
            raise ValueError(f'{label}: response identity differs ({key}, {response["sample_id"]})')
    expected = text_hash(response)
    if text_hash(incoming) != expected:
        raise ValueError(f'{label}: response text hash differs ({response["sample_id"]})')
    if incoming.get('source_response_sha256', expected) != expected:
        raise ValueError(f'{label}: declared response hash differs ({response["sample_id"]})')
    if benchmark.response_is_refusal(incoming) != benchmark.response_is_refusal(response):
        raise ValueError(f'{label}: response outcome differs ({response["sample_id"]})')


def load_grades(paths, responses_file=None):
    grades = {}
    receipts = []
    for supplied in paths:
        path = Path(supplied).resolve()
        path = path / 'grades.jsonl' if path.is_dir() else path
        meta_path = path.parent / 'grade_meta.json'
        meta = publication.decode(meta_path.read_bytes())
        if meta.get('phase') != 'grade' or meta.get('dry_run') is not False or not meta.get('judge_model'):
            raise ValueError(f'Expected actual canonical grading metadata: {meta_path}')
        rows = publication.json_rows(path)
        source_path = Path(responses_file or meta.get('responses_file', ''))
        if not source_path.is_file():
            raise ValueError(f'Original grading response input is unavailable: {source_path}')
        source_rows = keyed(publication.json_rows(source_path), 'Grading response input')
        for row in rows:
            sample = row['sample_id']
            if sample in grades:
                raise ValueError(f'Duplicate replacement judgment: {sample}')
            if sample not in source_rows:
                raise ValueError(f'Grade sample absent from original response input: {sample}')
            same_response(source_rows[sample], row, 'Canonical grade')
            if row.get('judge_model') != meta['judge_model']:
                raise ValueError(f'Grade model differs from metadata: {sample}')
            grades[sample] = (row, meta)
        receipts.append({
            'grades_file': str(path), 'grades_sha256': publication.sha(path.read_bytes()),
            'grade_meta_sha256': publication.sha(meta_path.read_bytes()),
            'responses_file': str(source_path.resolve()),
            'responses_sha256': publication.sha(source_path.read_bytes()),
            'grade_id': meta.get('grade_id'), 'judge_model': meta['judge_model'],
        })
    return grades, receipts


def slot_payload(row, slot):
    prefix = f'judge_{slot}_'
    return {key: copy.deepcopy(value) for key, value in row.items() if key.startswith(prefix)}


def slot_failure(row, slot, response):
    prefix = f'judge_{slot}_'
    grade = {'judge_' + key[len(prefix):]: value
             for key, value in row.items() if key.startswith(prefix)}
    grade.update(error=row.get(prefix + 'error', ''), status=row.get(prefix + 'status', 'ok'))
    # An existing candidate-refusal skip is intentionally unscored. Synthetic
    # zeroes on refusal rows must still be removed from the failed slot.
    if benchmark.response_is_refusal(response) and grade.get('judge_score') is None:
        grade['response_refusal'] = True
    return benchmark.judge_failure_reason(grade)


def original_judge(row, slot, panel):
    if row.get(f'judge_{slot}_model'):
        return row[f'judge_{slot}_model']
    # A merged legacy dataset's top-level roster is not per-sample evidence.
    roster = panel.get('panels', {}).get(row.get('judge_panel_id'), {})
    models = roster.get('judge_models')
    if not isinstance(models, list) or len(models) != 3 or not models[slot - 1]:
        return None
    return models[slot - 1]


def validate_replacement(response, grade, judge, questions):
    same_response(response, grade, 'Replacement grade')
    publication.validate_question_rows([grade], questions, 'Replacement grade')
    if grade.get('judge_model') != judge:
        raise ValueError(f'Replacement uses a different judge: {response["sample_id"]}')
    failure = benchmark.judge_failure_reason(grade)
    if failure:
        return failure
    if benchmark.response_is_refusal(response):
        if grade.get('judge_score') is not None or 'grading_skipped=response_refusal' not in grade.get('judge_warnings', []):
            raise ValueError(f'Candidate refusal requires canonical unscored skip: {response["sample_id"]}')
    else:
        if not grade.get('judge_response_id') or not grade.get('judge_raw_text'):
            raise ValueError(f'Replacement has no actual judge output: {response["sample_id"]}')
        score, justification, _ = benchmark.parse_judge_output(grade['judge_raw_text'])
        if score != grade['judge_score'] or justification != grade.get('judge_justification'):
            raise ValueError(f'Replacement differs from recorded judge output: {response["sample_id"]}')
    return ''


def refusal_skip(response, judge):
    # This canonical path returns before any client/network call.
    return benchmark.grade_one(
        response, clients=None, judge_model=judge, judge_provider='openrouter',
        judge_system_prompt='', judge_user_template='', judge_user_template_control='',
        judge_no_hint=True, judge_temperature=None, judge_reasoning_effort='off',
        judge_max_tokens=0, judge_output_retries=0, store_judge_response_raw=False,
        retries=0, pause_seconds=0, dry_run=False,
    )


def replace_slot(row, response, slot, grade, judge, grade_id, failure):
    prefix = f'judge_{slot}_'
    old_error = row.get(prefix + 'error')
    for key in list(row):
        if key.startswith(prefix):
            del row[key]
    if judge:
        row[prefix + 'model'] = judge
    if grade_id:
        row[prefix + 'grade_id'] = grade_id
    for field in GRADE_FIELDS:
        if 'judge_' + field in grade:
            row[prefix + field] = copy.deepcopy(grade['judge_' + field])
    row[prefix + 'score'] = None if failure else grade.get('judge_score')
    row[prefix + 'error'] = failure
    row[prefix + 'status'] = 'error' if failure else 'ok'
    if failure:
        row[prefix + 'justification'] = ''
        row[prefix + 'parse_mode'] = 'missing_judgment'
    row_errors = [error for error in row.get('row_errors', [])
                  if error != old_error and error != row.get('consensus_error')]
    for index in (1, 2, 3):
        error = row.get(f'judge_{index}_error')
        if error and error not in row_errors:
            row_errors.append(error)
    if response.get('error') and response['error'] not in row_errors:
        row_errors.append(response['error'])
    if row.get('row_identity_mismatch'):
        row_errors.append('row_identity_mismatch')
    scores = [row.get(f'judge_{index}_score') for index in (1, 2, 3)
              if not row.get(f'judge_{index}_error') and type(row.get(f'judge_{index}_score')) is int]
    if benchmark.response_is_refusal(response):
        consensus, consensus_error, scores = None, None, []
    elif len(scores) == 3 and not row_errors:
        consensus, consensus_error = benchmark.compute_consensus(scores, 'mean')
    else:
        consensus, consensus_error = None, f'incomplete_judge_panel:{len(scores)}/3'
        row_errors.append(consensus_error)
    row.update(consensus_score=consensus, consensus_method='mean', consensus_error=consensus_error,
               judge_valid_scores=scores, row_errors=row_errors,
               status='error' if row_errors else 'ok', error=' | '.join(row_errors))


def update_csv(path, summary):
    reader = csv.DictReader(io.StringIO(path.read_text()))
    fields = reader.fieldnames
    rows = {row['model']: row for row in reader}
    if not fields or len(rows) != len(summary['leaderboard']):
        raise ValueError(f'Existing leaderboard coverage differs: {path.name}')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for rank, model in enumerate(summary['leaderboard'], 1):
            row = rows[model['model']]
            row.update({key: model[value] for key, value in METRICS.items() if key in fields})
            row['rank'] = rank
            writer.writerow(row)


def build(args):
    source = Path(args.source).resolve()
    output = Path(args.output_dir).resolve()
    if (output.exists() or Path(args.output_dir).is_symlink() or output == source
            or output.is_relative_to(source) or source.is_relative_to(output)
            or output.is_relative_to(publication.ROOT / 'data')
            or output.is_relative_to(publication.ROOT / 'viewer')):
        raise ValueError('Output must be a new isolated directory outside the source and live data/viewer paths')
    manifest_raw = (source / 'manifest.json').read_bytes()
    manifest = publication.decode(manifest_raw)
    public_path = getattr(args, 'publication_path', None) or str(
        Path(manifest.get('exports', {}).get('leaderboard_csv', '')).parent)
    if public_path not in ('data/latest', 'data/v2/latest'):
        raise ValueError('Supply --publication-path data/latest or data/v2/latest for public manifest references')
    receipt_path = output.with_name(output.name + '.repair.json')
    if receipt_path.exists():
        raise ValueError(f'Repair receipt already exists: {receipt_path}')
    question_bytes = publication.migration_questions(args, source)
    questions = publication.question_definitions(question_bytes)
    raw_assets = {name: storage.read_bytes(source / name) for name in storage.ASSET_FORMATS}
    responses = keyed(publication.json_rows(source / 'responses.jsonl'), 'Source responses')
    aggregate = keyed(publication.json_rows(source / 'aggregate.jsonl'), 'Source aggregate')
    if responses.keys() != aggregate.keys():
        raise ValueError('Source response and aggregate sample coverage differs')
    for sample, row in aggregate.items():
        if any(row.get(key) != responses[sample].get(key) for key in publication.IDENTITY):
            raise ValueError(f'Source response/aggregate identity mismatch: {sample}')
    panel = publication.decode(publication.sidecar_bytes(source, 'panel_summary.json', manifest))
    if panel.get('consensus_method', 'mean') != 'mean' or panel.get('panel_mode', 'full') != 'full':
        raise ValueError('Repair supports only the established full three-judge mean panel')
    grades, grade_receipts = load_grades(args.grades, getattr(args, 'responses_file', None))
    if not grades.keys() <= aggregate.keys():
        raise ValueError('Replacement contains unknown sample IDs')
    historical_ids = set()
    for row in aggregate.values():
        try:
            historical_ids.update(publication.get_legacy_error_sample_ids([row]))
        except ValueError:
            pass  # The synthetic/failed slot is checked and repaired below.
    changes, blocked = [], []
    for sample, original in list(aggregate.items()):
        response = responses[sample]
        reason = slot_failure(original, args.judge_slot, response)
        if sample in grades and not reason:
            raise ValueError(f'Refusing to replace a valid existing judgment: {sample}')
        if not reason or (sample in historical_ids and sample not in grades):
            continue
        if args.write_blocked_draft and sample not in grades and not benchmark.response_is_refusal(response):
            # A review draft changes only judgments backed by supplied evidence.
            # Remaining synthetic rows stay untouched and keep the gate blocked.
            blocked.append({'sample_id': sample, 'judge_slot': args.judge_slot,
                            'reason': reason, 'replacement_supplied': False})
            continue
        original_model = original_judge(original, args.judge_slot, panel)
        judge = original_model
        grade, meta = grades.get(sample, ({}, {}))
        if grade:
            if not judge:
                if not args.write_blocked_draft or not benchmark.judge_failure_reason(grade):
                    raise ValueError(f'Original per-sample judge provenance is unavailable: {sample}')
                # Failed probes may be retained as evidence, never as scores.
                judge = grade.get('judge_model')
            failure = validate_replacement(response, grade, judge, questions)
        elif benchmark.response_is_refusal(response):
            grade, failure = refusal_skip(response, judge or ''), ''
        else:
            failure = 'No valid original-judge evaluation is available; score remains missing.'
        if failure and not args.write_blocked_draft:
            raise ValueError(f'Unresolved replacement judgment: {sample}: {failure}')
        if failure:
            blocked.append({'sample_id': sample, 'judge_slot': args.judge_slot, 'reason': failure})
        repaired = copy.deepcopy(original)
        replace_slot(repaired, response, args.judge_slot, grade, judge, meta.get('grade_id'), failure)
        for index in (1, 2, 3):
            if index != args.judge_slot and slot_payload(repaired, index) != slot_payload(original, index):
                raise ValueError(f'Untargeted judge changed: {sample}')
        aggregate[sample] = repaired
        changes.append({
            'sample_id': sample, 'identity': {key: response.get(key) for key in publication.IDENTITY},
            'response_sha256': text_hash(response), 'judge_slot': args.judge_slot,
            'original_judge_verified': bool(original_model),
            'original_failure': reason, 'original_judge': slot_payload(original, args.judge_slot),
            'repaired_judge': slot_payload(repaired, args.judge_slot),
            'original_consensus': original.get('consensus_score'), 'repaired_consensus': repaired['consensus_score'],
            'original_row_sha256': publication.sha(json_bytes(original)),
            'repaired_row_sha256': publication.sha(json_bytes(repaired)),
            'grade_attempts': grade.get('judge_attempts', []),
        })
    rows = list(aggregate.values())
    summary = benchmark.summarize_aggregate_rows(rows, 'mean', 3, legacy_error_sample_ids=historical_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.judge-repair-', dir=output.parent))
    try:
        publication.materialize(source, temporary)
        (temporary / 'questions.json').write_bytes(question_bytes)
        changed_ids = {change['sample_id'] for change in changes}
        if changed_ids:
            with (temporary / 'aggregate.jsonl').open('wb') as handle:
                for line in raw_assets['aggregate.jsonl'].splitlines(keepends=True):
                    if not line.strip():
                        handle.write(line)
                        continue
                    original = publication.decode(line)
                    handle.write((json.dumps(aggregate[original['sample_id']], ensure_ascii=False, allow_nan=False) + '\n').encode()
                                 if original['sample_id'] in changed_ids else line)
            viewer = publication.decode(raw_assets['viewer_rows.json.gz'])
            for row in viewer:
                if row['sample_id'] in changed_ids:
                    repaired = aggregate[row['sample_id']]
                    for field in (*CONSENSUS_FIELDS, *(f'judge_{args.judge_slot}_{key}' for key in ('score', 'status', 'error'))):
                        if field in repaired:
                            row[field] = copy.deepcopy(repaired[field])
            (temporary / 'viewer_rows.json.gz').write_bytes(gzip.compress(
                json.dumps(viewer, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
        (temporary / 'aggregate_summary.json').write_bytes(json_bytes(summary))
        for name in ('leaderboard.csv', 'leaderboard_with_launch.csv'):
            update_csv(temporary / name, summary)
        pair = summary['reliability']['pairwise'][0]
        panel['disagreement_count'] = pair['compared_rows'] - pair['agreements']
        panel['disagreement_rate'] = round(panel['disagreement_count'] / max(1, len(rows)), 4)
        (temporary / 'panel_summary.json').write_bytes(json_bytes(panel))
        storage.pack_dataset(temporary)
        publication.freeze_metadata(temporary, publication.ROOT / public_path)
        after_hashes = {name: publication.sha(storage.read_bytes(temporary / name)) for name in storage.ASSET_FORMATS}
        for name in ('responses.jsonl', 'viewer_details.json.gz'):
            if after_hashes[name] != publication.sha(raw_assets[name]):
                raise ValueError(f'Repair changed preserved response bytes: {name}')
        try:
            validation, validation_error = publication.validate_dataset(temporary), None
        except ValueError as exc:
            validation, validation_error = None, str(exc)
            if not args.write_blocked_draft:
                raise
        if (source / 'manifest.json').read_bytes() != manifest_raw:
            raise ValueError('Source publication changed while the repair was being built')
        receipt = {
            'schema_version': 1, 'status': 'blocked' if validation_error else 'validated',
            'activated': False, 'created_at_utc': benchmark.utc_now_iso(),
            'source': str(source), 'output': str(output), 'publication_path': public_path,
            'source_manifest_sha256': publication.sha(manifest_raw),
            'output_manifest_sha256': publication.sha((temporary / 'manifest.json').read_bytes()),
            'original_asset_sha256': {name: publication.sha(raw) for name, raw in raw_assets.items()},
            'repaired_asset_sha256': after_hashes, 'questions_sha256': publication.sha(question_bytes),
            'grade_inputs': grade_receipts, 'affected_samples': changes,
            'historical_error_samples_preserved': sorted(historical_ids),
            'unresolved_judgments': blocked, 'publication_validation': validation,
            'publication_validation_error': validation_error,
        }
        if output.exists():
            raise ValueError('Output appeared while building; refusing to overwrite it')
        # Local input paths belong beside the review artifact, never in the
        # dataset that publication.activate could later copy into GitHub Pages.
        with receipt_path.open('xb') as handle:
            handle.write(json_bytes(receipt))
        os.rename(temporary, output)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--questions-file', required=True, type=Path)
    parser.add_argument('--grades', action='append', default=[], type=Path,
                        help='canonical grade directory or final grades JSONL; repeat for disjoint inputs')
    parser.add_argument('--responses-file', type=Path, help='exact input used by the supplied grading run')
    parser.add_argument('--judge-slot', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--publication-path', choices=('data/latest', 'data/v2/latest'),
                        help='eventual public path for manifest references; no files are written there')
    parser.add_argument('--write-blocked-draft', action='store_true',
                        help='write an unpublishable draft with missing judgments and return exit code 1')
    args = parser.parse_args(argv)
    try:
        receipt = build(args)
        print(json.dumps({'output': str(args.output_dir), 'status': receipt['status'],
                          'affected_samples': len(receipt['affected_samples']),
                          'unresolved_judgments': len(receipt['unresolved_judgments']),
                          'validation_error': receipt['publication_validation_error']}, indent=2))
        return 0 if receipt['status'] == 'validated' else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f'repair_published_judges: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

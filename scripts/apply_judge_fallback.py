#!/usr/bin/env python3
"""Build and verify an isolated migration to the approved two-valid-judge rule.

No model calls and no live activation. Original responses, genuine judge scores
and justifications, and all untouched aggregate rows are preserved. The receipt
and complete original changed rows are written beside the candidate dataset.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import openrouter_benchmark as benchmark
import publication
import published_dataset as storage
from repair_published_judges import keyed, json_bytes, text_hash, update_csv

POLICY = 'retry_then_two_valid'
SUMMARY_DROP = {'response_text', 'question', 'domain', 'nonsensical_element',
                'judge_1_justification', 'judge_2_justification', 'judge_3_justification',
                'judge_valid_scores'}


def slot_grade(row, index):
    prefix = f'judge_{index}_'
    grade = {'judge_' + key[len(prefix):]: value for key, value in row.items()
             if key.startswith(prefix)}
    grade.update(error=row.get(prefix + 'error', ''), status=row.get(prefix + 'status', ''),
                 row_present=row.get(prefix + 'row_present'))
    return grade


def eligible_slots(row):
    return [index for index in (1, 2, 3)
            if benchmark.judge_failure_reason(slot_grade(row, index))
            and benchmark.judge_failure_is_exhausted(slot_grade(row, index))]


def migrate_row(original, response):
    slots = eligible_slots(original)
    if not slots or original.get('judge_failure_policy') == POLICY:
        return None
    source = {**original, 'response_refusal': benchmark.response_is_refusal(response),
              'response_outcome': response.get('response_outcome', original.get('response_outcome', ''))}
    if response.get('error'):
        source['source_response_error'] = response['error']
    repaired = benchmark.finalize_judge_consensus(source, num_judges=3, consensus_method='mean')
    if repaired.get('status') != 'ok':
        raise ValueError(f'Fallback cannot resolve sample {original["sample_id"]}: {repaired.get("error")}')
    for index in (1, 2, 3):
        if index not in slots:
            for field in ('score', 'justification'):
                key = f'judge_{index}_{field}'
                if repaired.get(key) != original.get(key):
                    raise ValueError(f'Genuine judge changed: {original["sample_id"]} {key}')
    publication.validate_judge_panel(response, repaired, 'Migrated')
    return repaired


def build(source, output, publication_path, questions_file=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output.exists() or output.is_relative_to(source) or source.is_relative_to(output)
            or output.is_relative_to(publication.ROOT / 'data')
            or output.is_relative_to(publication.ROOT / 'viewer')):
        raise ValueError('Output must be a new isolated directory outside source and live data/viewer paths')
    if publication_path not in ('data/latest', 'data/v2/latest'):
        raise ValueError('Invalid publication path')
    manifest_raw = (source / 'manifest.json').read_bytes()
    manifest = publication.decode(manifest_raw)
    question_bytes = publication.migration_questions(SimpleNamespace(questions_file=questions_file), source)
    questions = publication.question_definitions(question_bytes)
    raw = {name: storage.read_bytes(source / name) for name in storage.ASSET_FORMATS}
    responses = keyed(publication.json_rows(source / 'responses.jsonl'), 'Responses')
    original_rows = publication.json_rows(source / 'aggregate.jsonl')
    aggregates = keyed(original_rows, 'Aggregates')
    if responses.keys() != aggregates.keys():
        raise ValueError('Response/aggregate sample coverage differs')
    changes = []
    legacy_errors = set()
    for row in original_rows:
        sample = row['sample_id']
        response = responses[sample]
        if any(row.get(field) != response.get(field) for field in publication.IDENTITY):
            raise ValueError(f'Response/aggregate identity mismatch: {sample}')
        repaired = migrate_row(row, response)
        if repaired:
            aggregates[sample] = repaired
            changes.append({'sample_id': sample, 'model': row['model'], 'question_id': row['question_id'],
                            'response_sha256': text_hash(response), 'original': row, 'repaired': repaired})
        else:
            legacy_errors.update(publication.get_legacy_error_sample_ids([row]))
    rows = list(aggregates.values())
    summary = benchmark.summarize_aggregate_rows(rows, 'mean', 3, legacy_error_sample_ids=legacy_errors)
    panel = publication.decode(publication.sidecar_bytes(source, 'panel_summary.json', manifest))
    panel.update(judge_failure_policy=POLICY, minimum_valid_judges=2,
                 fallback_response_count=sum(row.get('judge_failure_policy') == POLICY for row in rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.judge-fallback-', dir=output.parent))
    try:
        publication.materialize(source, temporary)
        changed = {entry['sample_id'] for entry in changes}
        with (temporary / 'aggregate.jsonl').open('wb') as handle:
            for line in raw['aggregate.jsonl'].split(b'\n'):
                if not line:
                    continue
                row = publication.decode(line)
                if row['sample_id'] in changed:
                    line = json.dumps(aggregates[row['sample_id']], ensure_ascii=False, allow_nan=False).encode()
                handle.write(line + b'\n')
        viewer = publication.decode(raw['viewer_rows.json.gz'])
        for row in viewer:
            if row['sample_id'] in changed:
                row.update({key: copy.deepcopy(value) for key, value in aggregates[row['sample_id']].items()
                            if key not in SUMMARY_DROP})
        (temporary / 'viewer_rows.json.gz').write_bytes(gzip.compress(
            json.dumps(viewer, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
        (temporary / 'aggregate_summary.json').write_bytes(json_bytes(summary))
        (temporary / 'panel_summary.json').write_bytes(json_bytes(panel))
        for name in ('leaderboard.csv', 'leaderboard_with_launch.csv'):
            update_csv(temporary / name, summary)
        (temporary / 'questions.json').write_bytes(question_bytes)
        storage.pack_dataset(temporary)
        publication.freeze_metadata(temporary, publication.ROOT / publication_path)
        after_hashes = {name: publication.sha(storage.read_bytes(temporary / name)) for name in storage.ASSET_FORMATS}
        for name in ('responses.jsonl', 'viewer_details.json.gz'):
            if after_hashes[name] != publication.sha(raw[name]):
                raise ValueError(f'Migration changed original response bytes: {name}')
        validation = publication.validate_dataset(temporary)
        if (source / 'manifest.json').read_bytes() != manifest_raw:
            raise ValueError('Source dataset changed during migration')
        before_csv = publication.sidecar_bytes(source, 'leaderboard.csv', manifest)
        receipt = {'policy': POLICY, 'created_at_utc': benchmark.utc_now_iso(), 'activated': False,
                   'source': str(source), 'output': str(output),
                   'source_manifest_sha256': publication.sha(manifest_raw),
                   'questions_sha256': publication.sha(question_bytes),
                   'original_asset_sha256': {name: publication.sha(value) for name, value in raw.items()},
                   'repaired_asset_sha256': after_hashes, 'changed_samples': len(changes),
                   'fallback_response_count': panel['fallback_response_count'],
                   'candidate_refusal_corrections': sum(benchmark.response_is_refusal(c['repaired']) for c in changes),
                   'preserved_legacy_errors': sorted(legacy_errors), 'validation': validation}
        (output.with_suffix('.before-leaderboard.csv')).write_bytes(before_csv)
        (output.with_suffix('.changed-rows.json')).write_bytes(json_bytes(changes))
        (output.with_suffix('.receipt.json')).write_bytes(json_bytes(receipt))
        os.rename(temporary, output)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--publication-path', required=True, choices=['data/latest', 'data/v2/latest'])
    parser.add_argument('--questions-file', type=Path, help='verified original source for a legacy publication without a snapshot')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output_dir, args.publication_path, args.questions_file), indent=2))


if __name__ == '__main__':
    main()

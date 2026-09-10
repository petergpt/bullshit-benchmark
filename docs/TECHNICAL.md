# BullshitBench Technical Guide

This guide is for maintainers and contributors working on benchmark operations and data publishing.

## Pipeline Overview

The end-to-end flow is:

1. `collect`
2. `grade` (primary judge, usually Claude)
3. `grade-panel` for the remaining judges (required before publication)
4. `publish_latest_to_viewer.sh` (when additional judges are run)

Run stage 1 (collect + primary judge) with:

```bash
./scripts/run_end_to_end.sh
```

Run stage 2 later (remaining judges + publish) with:

```bash
./scripts/run_end_to_end.sh --skip-collect --skip-primary-judge --with-additional-judges --run-id <run_id> --panel-id <panel_id>
```

Run both stages in one go with:

```bash
./scripts/run_end_to_end.sh --with-additional-judges
```

Serve existing published results without starting a new benchmark run:

```bash
python3 -m http.server 8795 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8795/viewer/index.next.html`. The legacy viewer remains at `viewer/index.legacy.html` and reads the same datasets.

### Grade Panel Policy

Canonical publish pipeline policy:

- Exactly `3` judges must be configured in `grade_panel.judge_models`.
- `panel_mode` is fixed to `full` (every judge runs on every row).
- `consensus_method` is fixed to `mean`.
- The `retry_then_two_valid` failure policy averages at least two valid judgments after same-judge retries are exhausted and labels reduced coverage as `2/3 judges`.
- Legacy disagreement-tiebreak/alternate consensus modes are not supported in the main pipeline.

### Run v2 Without Overwriting v1

Use the v2 config and publish to `data/v2/latest`:

```bash
./scripts/run_end_to_end.sh --config config.v2.json --viewer-output-dir data/v2/latest --with-additional-judges
```

The dashboard's V1/V2 selector switches datasets; the original viewer has a `Benchmark Version` dropdown.

### v1 to v2 Release Best Practices

Use this checklist before pushing to GitHub/GitHub Pages:

1. Keep versioned published datasets side-by-side:
   - v1: `data/latest/*`
   - v2: `data/v2/latest/*`
2. Do not commit local run history (`runs/*`) or ad hoc temp artifacts.
3. Rebuild v2 question JSON from draft source when question content changes:
   - source: `drafts/new-questions.md`
   - builder: `scripts/build_questions_v2_from_draft.py`
4. Publish datasets only via `scripts/publish_latest_to_viewer.sh` (or `run_end_to_end.sh` wrapper) so artifact normalization stays consistent.
5. Smoke-test both populated viewers before release: `viewer/index.next.html` and `viewer/index.legacy.html`, including both suites, response loading and exports.

### High-Throughput Collection Knobs (30k+ Queries)

Collection now supports model-aware scheduling and durable checkpoints:

- `parallelism`: global concurrent requests
- `max_inflight_per_model`: cap per-model concurrency so one provider bucket cannot starve others
- `rate_limit_requeue`: when true, HTTP 429 rows are requeued with model cooldown instead of immediately failing
- `rate_limit_cooldown_seconds`, `rate_limit_cooldown_max_seconds`, `rate_limit_cooldown_jitter_seconds`: cooldown controls for rate-limited models
- `rate_limit_max_attempts`: max total attempts per `sample_id` before final failure
- `checkpoint_fsync_every`: fsync cadence for `responses.partial.jsonl` and `collect_events.jsonl` durability

Operational guidance for speed:

- For very large runs, set `retries=1` so workers do not sleep on backoff internally; let scheduler-level requeue handle cooldown.
- Increase `parallelism` aggressively (for example 48–96) and tune `max_inflight_per_model` (for example 1–3) based on observed 429 rates.
- Keep `shuffle_tasks=true` to spread models/questions and smooth bursty limits.

## Missing Judge Evaluations

Judging makes up to three same-judge output attempts by default (`judge_output_retries=2`). Empty, filtered, truncated, or invalid output remains an unscored error after exhaustion; it must never become score `0`. Attempt IDs, finish reasons, and usage are retained for diagnosis.

`grade --resume` and `grade-panel --resume` reuse valid judgments (including genuine zeroes), retry failed or historical empty-output fallback rows, and archive the replaced failures in `grades.retry-history.jsonl`. They do not rerun successful judges for those responses.

Under `retry_then_two_valid`, a configured three-judge panel may use the mean of two valid judgments once the failed judge has exhausted its three output attempts. The failed vote stays `null` with its diagnostics; a genuine score `0` remains a valid vote. New and repaired aggregate rows record `judge_valid_count`, `judge_expected_count`, `judge_coverage`, and `judge_failure_policy`. Reduced coverage appears in answer details, chart tooltips, CSV exports and the dashboard PNG footer, without adding badges to model labels or Lab trends summaries.

Fewer than two valid judgments, unexhausted judge failures, collection failures, or mismatched sample/response/judge identities still block publication and the affected model's headline rates. Keep the full question denominator; do not drop the affected question. Provider-declared **candidate** refusals remain a separate, deliberately recorded benchmark outcome with intentionally skipped grading.

`scripts/repair_published_judges.py` builds an isolated candidate from existing published data and canonical grade artifacts. It requires exact sample/response identity and original per-sample judge provenance for successful replacements; a merged dataset's top-level judge list is not sufficient. It preserves original answers, other judges and non-targeted rows, regenerates derived summaries with the same scoring code, and validates the result before reporting success. Its receipt is stored beside the candidate, outside the publishable dataset. `--write-blocked-draft` can retain failed probe evidence for review, but still exits nonzero when validation fails and never activates the draft.

## Publish Existing Run Artifacts

Use this when you already have run outputs and only want to refresh a viewer dataset:

```bash
./scripts/publish_latest_to_viewer.sh \
  --responses-file <path/to/responses.jsonl> \
  --collection-stats <path/to/collection_stats.json> \
  --panel-summary <path/to/panel_summary.json> \
  --aggregate-summary <path/to/aggregate_summary.json> \
  --aggregate-rows <path/to/aggregate.jsonl> \
  --output-dir <data/latest-or-data/v2/latest>
```

Publish behavior:

- Default `--publish-mode auto` is safety-first: if output already exists, publish is supplemental (merge by `sample_id`); if output does not exist, publish is replace.
- To force merge: add `--supplemental` (or `--publish-mode supplemental`).
- To intentionally overwrite with only incoming artifacts: add `--replace` (or `--publish-mode replace`).

The command stages the complete dataset, validates it, then activates `manifest.json` last. Writers targeting the same output are serialized. A failed build or validation leaves the prior manifest and its immutable assets readable. Supplemental imports reject changed sample-ID collisions unless `--allow-replace-samples` is explicitly supplied; mismatched identities always fail.

New imports require full question coverage, the configured three-judge mean panel with at least two valid judgments per scored response, and durable config coverage. Reduced panels require verified exhausted retries, retained failure diagnostics, and explicit `2/3 judges` coverage under `retry_then_two_valid`; other unresolved collection/grading failures remain blocking. Provider refusals remain represented as attempts with intentionally skipped scores. Existing historical error rows are preserved. Newly imported aggregates retain their judge-panel ID, with the corresponding model roster in `panel_summary.json`.

The publisher takes the original `questions_snapshot.json` beside the collected responses, or an explicit `--questions-file` source. It freezes those bytes with the results. Reusing an existing question ID with different wording, a different hint, or changed shared annotations is rejected, including in replace mode. Root question files remain convenient downloads; manifest-aware viewers use their release's frozen snapshot.

The publish step strips local-machine path fields from public artifacts and sanitizes local path fragments in published JSONL text fields. End-to-end `--dry-run` never invokes publishing.

Before a push, run the same read-only checks as CI. Verification rejects files at the GitHub hard limit and reserves a 900 MB combined dataset budget, leaving room for the viewer and documentation within the Pages site limit:

```bash
python3 -m unittest discover -s tests
node --test tests/test_published_viewer_storage.js
node --test tests/test_viewer_routes.js tests/test_viewer_refusal_denominators.js
node --test tests/test_next_viewer_data.mjs tests/test_next_viewer_capture.mjs tests/test_report_judge_failures.js
python3 scripts/publication.py verify data/latest data/v2/latest
python3 scripts/validate_configs.py
python3 scripts/build_pages.py --output /tmp/bullshitbench-pages
```

A successful local publish prepares repository files. Git commit, push, and GitHub Pages deployment are separate operations; verify the deployed commit and both viewer versions after a release.

The Pages workflow builds an explicit public artifact only after these checks pass, then makes its deployment job depend on that successful build. It excludes local runs, reports, temporary files and credentials. The artifact includes current and immediately previous manifest assets, uses `viewer/index.next.html` as the main dashboard, and redirects historical `viewer/index.v2.html` links to it. The original viewer is retained at `viewer/index.legacy.html`. After the complete release passes local validation, set the repository's Pages source to **GitHub Actions** before the first release push to `main`. That push installs the workflow and starts its checked build and deployment. The older branch-based Pages source bypasses these checks and must not remain enabled for the release.

## Launch-Date Metadata Pipeline

Build model launch-date inventory/buckets and export review/candidate/canonical launch datasets:

```bash
./scripts/model_launch_pipeline.py run
```

This writes:

- `data/model_metadata/tested_models_inventory.csv`
- `data/model_metadata/model_buckets.csv`
- `data/model_metadata/model_launch_sources.csv` (template if missing)
- `data/model_metadata/model_launch_collection.csv`
- `data/model_metadata/model_launch_judged.csv`
- `data/model_metadata/model_launch_attempts.csv`
- `data/model_metadata/model_launch_dates_review.csv`
- `data/model_metadata/model_launch_dates_candidates.csv`
- `data/model_metadata/model_launch_dates.csv` (canonical accepted rows)
- `data/model_metadata/model_params.csv` (canonical tracked size/licensing metadata for models with public parameter disclosures)

Publishing also exports:

- `data/latest/model_launch_dates.csv`
- `data/latest/leaderboard_with_launch.csv`
- `data/latest/model_params.csv`

## Current Config Notes

- Main config (v1): `config.json`
- Main config (v2): `config.v2.json`
- Catch-up queues: `config.new-models.v1.json` and `config.new-models.v2.json`. Completed August variants were folded into the main configs. These queues may contain unpublished candidates; add only intended new models, with matching reasoning/provider overrides, before running them. The config gate permits an empty candidate queue; the collector rejects an empty model list before creating a run or making API calls. Main configs must remain nonempty.
- Question set (v1): `questions.json`
- Question set (v2): `questions.v2.json` (generated from `drafts/new-questions.md` via `scripts/build_questions_v2_from_draft.py`)
- Provider routing is controlled by `collect.model_providers` and `grade.model_providers` (`openrouter` or `openai`; supports `*` and `<org>/*` patterns, e.g. `{"*":"openrouter","gpt-5.3":"openai"}`).
- `openai/gpt-5.5-chat` is a benchmark display/model row routed by its config override to the `chat-latest` API slug. That moving API alias does not itself identify a fixed model version; retain the original run's request/response metadata when interpreting historical results.
- Claude Fable 5.1 and GPT-6 Astra are in both main configs with `low` and `max` reasoning. The current datasets have 192 V1 variants / 10,560 response records and 212 V2 variants / 21,200 response records. Historical variants outside the active rerun set have exact entries in `data/model_metadata/legacy_config_exceptions.json`.
- Configs include `openai/gpt-5.2-codex` and `openai/gpt-5.3-codex` with reasoning sweeps (`low`, `high`, `xhigh`).
- `scripts/validate_configs.py` checks exact published model/reasoning coverage against the main configs and the historical exception list; run-history inventory is a separate metadata input.
- When publishing model results from ad hoc or catch-up runs, update the durable config path in the same change. Published v1 rows should be represented in `config.json`, published v2 rows should be represented in `config.v2.json`, and candidate unrun catch-up models should live in `config.new-models.v1.json` / `config.new-models.v2.json` until they are either published or intentionally deferred.

## Repository Layout

- `scripts/openrouter_benchmark.py`: core CLI (`collect`, `grade`, `grade-panel`, `aggregate`, `report`)
- `scripts/run_end_to_end.sh`: one-command pipeline runner
- `scripts/publish_latest_to_viewer.sh`: public publishing entry point
- `scripts/publication.py`: staged activation, full verification, and migration
- `scripts/published_dataset.py`: lossless bounded storage and logical readers
- `scripts/validate_configs.py`: durable config and historical-variant coverage gate
- `scripts/build_questions_v2_from_draft.py`: build `questions.v2.json` from markdown draft
- `scripts/cleanup_generated_outputs.sh`: remove generated local artifacts
- `scripts/model_launch_pipeline.py`: launch-date collection/judging pipeline
- `viewer/index.next.html` and `viewer/next/*`: main dashboard, static ES modules and local assets
- `viewer/index.html` and `viewer/index.v2.html`: compatibility redirects to the dashboard
- `viewer/index.legacy.html`: original interactive viewer
- `index.html` and `viewer/index.html`: redirects to the main dashboard
- `data/latest/*`: benchmark v1 published dataset
- `data/v2/latest/*`: benchmark v2 published dataset
- `runs/*`: local run history

## Published Dataset Files

Each published dataset directory (`data/latest` or `data/v2/latest`) is addressed by `manifest.json`:

- `storage.version = 2` lists four logical assets: `responses.jsonl`, `aggregate.jsonl`, `viewer_rows.json.gz`, and `viewer_details.json.gz`.
- Physical files live under `responses/`, `aggregate/`, `viewer_rows/`, and `viewer_details/`. Filenames contain the SHA-256 of their encoded bytes. Every physical asset is at most 4 MiB, safely below GitHub's 100 MiB hard limit.
- Canonical JSONL chunks concatenate to the exact original response and aggregate bytes. No answers, reasoning, token/cost fields, or scores are removed during migration. Viewer details are derived, grouped by question, and preserve response text by sample ID.
- Each manifest part records encoded SHA-256, encoded/decoded byte counts, and row count. Details include a question index. Validation checks complete sample-ID coverage and response/aggregate identity.
- `files` pins immutable copies of all metadata sidecars and `questions.json` under `metadata/`. The question source may retain its grouped or collection-snapshot shape. Stable CSV/JSON aliases remain available for direct downloads; both viewers use the pinned copies to avoid mixing publications.

The viewer loads its compressed score summary first. Response text is fetched only for questions needed by the visible comparison or opened examples. Requests share a four-request limit and cache; failed downloads show a retry action. Text hydration cannot alter score/refusal classification or close open examples. The legacy viewer can fall back to verified plain canonical shards; the dashboard requires browser gzip support. Legacy monolithic datasets and storage version 1 remain readable.

### Displayed rate denominators

Dashboard scores, rankings, chart points and rate summaries use `clear_pushback_count / (attempt_count - refusal_count)`. Partial-challenge and accepted-premise rates use the same denominator. Collection or scoring errors remain represented as attempts. The **Exclude refusals** checkbox starts unticked and controls only the dashboard's stacked bars; tick it to renormalize the bar segments. Reset returns bars to all attempts. `excludeRefusals=1` preserves the explicit bar choice in the URL. Refusal rates always use all attempts. The original viewer retains its separate All attempts / Exclude refusals control and defaults to exclusion.

Canonical aggregate files and published `leaderboard.csv` files retain full-denominator rates and refusal counts. Compare these with the explicit all-attempt fields. A model with only refusals has no non-refusal denominator: its primary model rate displays as unavailable and chart plots omit it.

Model details and chart tooltips expose both clear-pushback rates when they differ. Dashboard CSV exports record non-refusal `rate_denominator`, `clear_pushback_pct` and `exclude_refusals=true`, alongside explicit `clear_pushback_pct_excluding_refusals` and `clear_pushback_pct_all_attempts`. `bars_exclude_refusals` and `bar_denominator` record bar composition separately. PNG exports follow the same split: Clear excludes refusals, while bars follow the checkbox. Viewer token/cost averages exclude refusals; individual answer details retain that answer's actual telemetry. Reported reasoning/output means include coverage counts, preserving missing values as unknown and measured zeros as zero.

Lab trends selects the highest non-refusal clear-pushback model for each lab and exact release date. Solid lines show that result; faint dotted lines show all-attempt rates of the **same selected models**. The comparison starts enabled, can be hidden with the All attempts checkbox, and appears only for labs with a differing rate. It introduces no second point or label series.

Use the manifest-aware reader instead of opening a presumed physical monolith:

```python
from scripts.published_dataset import read_text
rows_jsonl = read_text("data/v2/latest/responses.jsonl")
```

Or reconstruct an ordinary JSONL export outside the published tree:

```bash
python3 scripts/published_dataset.py cat data/v2/latest/responses.jsonl > /tmp/responses.jsonl
```

To migrate an existing dataset without collecting or grading, first verify the question source against the original collection snapshots:

```bash
python3 scripts/publication.py migrate --output-dir data/latest --questions-file questions.json
python3 scripts/publication.py migrate --output-dir data/v2/latest --questions-file questions.v2.json
```

Content files are immutable and unchanged parts are reused. Activation retains the current and immediately previous complete publication; its previous manifest is recorded under `releases/`. Only older generated hashes are removed after activation, bounding live-site growth while allowing readers of the previous manifest to finish. Do not manually edit parts or overwrite a hashed filename. The low-level pack helper retains all prior files; use the public publisher for managed retention. Existing Git history and original run artifacts are left intact; this design reduces the size of future changed blobs, not historical clone size.

Retired or historical published variants intentionally absent from the active rerun configs are recorded in `data/model_metadata/legacy_config_exceptions.json`. The config gate rejects new unexplained omissions and duplicate JSON keys without silently adding retired or expensive models to the run set.

Collection run artifacts under `runs/<run_id>/` now also include flattened per-row usage metrics in `responses.jsonl` and `responses_review.csv`:

- token counts (`response_prompt_tokens`, `response_completion_tokens`, `response_total_tokens`, `response_reasoning_tokens`)
- cache details (`response_cached_prompt_tokens`, `response_cache_write_tokens`)
- cost fields (`response_cost_usd` and upstream cost breakdown fields)
- derived metrics (`response_char_count`, `response_tokens_per_second`)

And `collection_stats.json` includes `usage_summary` with totals/averages overall and by model.

## Environment

Required:

- `OPENROUTER_API_KEY`

Optional:

- `OPENROUTER_REFERER`
- `OPENROUTER_APP_NAME`
- `OPENAI_API_KEY` (required when any model is routed to provider `openai`)
- `OPENAI_PROJECT` or `OPENAI_PROJECT_ID` (optional OpenAI project header override)
- `OPENAI_ORGANIZATION` or `OPENAI_ORG` or `OPENAI_ORG_ID` (optional OpenAI org header override)

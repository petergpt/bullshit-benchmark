# Bullshit Benchmark Runner (OpenRouter)

This repo now uses a strict **two-step** workflow:
- **Collect** responses from target models with stateless, independent prompts
- **Grade** those saved responses with a judge model on a narrow `0/1/2` rubric

## Why two steps

- Collection and grading are decoupled for reproducibility.
- You can re-grade the same response set with different judge models/prompts.
- Every model call is isolated: one system message + one user question, no shared chat history.

## Quickstart (Run + View)

- End-to-end rerun and publish latest viewer data: `scripts/run_end_to_end.sh`
- Main review URL: `viewer/index.html`
- Public GitHub Pages URL: `https://petergpt.github.io/bullshit-benchmark/viewer/index.html`

## Files

- `questions.json`: benchmark dataset
- `guide.md`: benchmark intent/rubric
- `scripts/openrouter_benchmark.py`: CLI runner (`collect` and `grade`)
- `models.example.txt`: sample model list format
- `judge_prompt.template.txt`: customizable judge prompt template
- `config.example.json`: example config defaults
- `config.v2.json`: v2 defaults (no response system prompt, reasoning sweep example, panel judging)
- `viewer/index.html`: canonical interactive viewer
- `viewer/data/latest/*`: canonical dataset consumed by the viewer
- `scripts/publish_latest_to_viewer.sh`: publish one run's final artifacts into `viewer/data/latest`
- `scripts/cleanup_generated_outputs.sh`: remove generated run/archive/temp artifacts from workspace
- `scripts/run_end_to_end.sh`: one-command rerun (`collect` -> `grade-panel` -> publish `viewer/data/latest`)

## Open-Source Snapshot Layout

For open-source hygiene, keep one canonical visualization path:
- `viewer/index.html` (final HTML UI)
- `viewer/data/latest/responses.jsonl`
- `viewer/data/latest/collection_stats.json`
- `viewer/data/latest/panel_summary.json`
- `viewer/data/latest/aggregate_summary.json`
- `viewer/data/latest/aggregate.jsonl`
- `viewer/data/latest/leaderboard.csv`
- `viewer/data/latest/manifest.json`

All timestamped `runs*`, interim reports, ad-hoc zips, and temporary JSON files should be treated as generated artifacts and removed before publishing.

This layout provides standard benchmark-style outputs without requiring a paper:
- stable machine-readable artifacts (`aggregate_summary.json`, `aggregate.jsonl`, `leaderboard.csv`)
- stable human-facing view (`index.html`)
- one stable “latest” path (`viewer/data/latest`)

## Environment

- Required: `OPENROUTER_API_KEY`
- Optional:
  - `OPENROUTER_REFERER`
  - `OPENROUTER_APP_NAME`

Config precedence:
- Explicit CLI flags always take precedence over config file values.

## Step 1: Collect responses

Collect writes to `runs/<run_id>/`:
- `collection_meta.json`
- `questions_snapshot.json`
- `responses.partial.jsonl` (incremental checkpoint, append-only)
- `responses.jsonl`
- `collection_stats.json`
- `responses_review.csv`
- `collect_events.jsonl`

`<run_id>` defaults to a UTC timestamp (`YYYYMMDD_HHMMSS`). If a same-second folder already exists, a numeric suffix is added automatically.

Useful knobs:
- `--num-runs`: repeated independent samples per model x question
- `--parallelism`: concurrent requests
- `--techniques`: subset of techniques
- `--limit`: quick smoke tests
- control questions are excluded automatically during collection
- `--omit-response-system-prompt`: send only user message (no system message)
- `--response-reasoning-effort off|none|minimal|low|medium|high|xhigh`
- `--model-reasoning-efforts '{"openai/gpt-5.2":["none","low","medium","high","xhigh"]}'`
- reasoning-level support is model-specific; unsupported levels return provider errors
- `--store-request-messages`: opt-in prompt logging in `responses.jsonl` (default: off)
- `--store-response-raw`: opt-in raw provider payload logging in `responses.jsonl` (default: off)
- `--retries` / `--timeout-seconds`: bounded retry attempts per API call with timeout (defaults: `3` and `120`)
- `--resume` with `--run-id`: resume from `responses.partial.jsonl` (or `responses.jsonl` fallback)
- `--fail-on-error` (default): exits non-zero if any row fails

Execution uses bounded in-flight scheduling: active requests are capped by `--parallelism` (tasks are streamed, not all fired at once).
Retry behavior honors `Retry-After` (seconds or HTTP-date) up to 300s and uses jittered exponential backoff otherwise.

Run isolation:
- `sample_id` is namespaced by resolved `run_id` and includes a stable hash of the model row identity.
- This prevents accidental collisions across separate benchmark runs or similarly named model variants.

V2 model variant identity in output rows:
- `model_org`: provider/org prefix (for example, `openai`)
- `model_name`: base model name without org (for example, `gpt-5.2`)
- `model_reasoning_level`: configured reasoning level (`default|none|low|medium|high|xhigh`)
- `model_row`: `model_name@reasoning=<level>`
- `model`: unique display key `model_org/model_row` (used as benchmark row identity)
- `model_id`: raw OpenRouter model id used for API calls (for example, `openai/gpt-5.2`)

## Step 2: Grade responses

Grade reads a prior `responses.jsonl` and writes to:
- `<responses_parent>/grades/<grade_id>/grade_meta.json`
- `<responses_parent>/grades/<grade_id>/grades.partial.jsonl` (incremental checkpoint, append-only)
- `<responses_parent>/grades/<grade_id>/grades.jsonl`
- `<responses_parent>/grades/<grade_id>/summary.json`
- `<responses_parent>/grades/<grade_id>/summary.md`
- `<responses_parent>/grades/<grade_id>/review.csv`
- `<responses_parent>/grades/<grade_id>/review.md`
- `<responses_parent>/grades/<grade_id>/grade_events.jsonl`

`<grade_id>` defaults to `YYYYMMDD_HHMMSS_<judge_model_slug>` with the same collision-safe suffix behavior.

Judge options:
- `--judge-reasoning-effort off|low|medium|high` (passed as `reasoning.effort`)
- `--judge-temperature` is optional; omitted by default for compatibility with judge models that do not support temperature.
- `--judge-max-tokens`: `0` omits token cap (default in V2), positive values enforce a cap.
- `--store-judge-response-raw`: opt-in raw judge provider payload logging in `grades.jsonl` (enabled in `config.v2.json`).
- `--retries` / `--timeout-seconds` apply to judge API calls as well.
- `--resume` with `--grade-id`: resume from `grades.partial.jsonl` (or `grades.jsonl` fallback)
- Google judge models use `response_format: {"type":"json_object"}` for provider compatibility; OpenAI/Anthropic use strict `json_schema`.

## Step 2b: Grade Panel (V2 mechanics)

`grade-panel` runs:
- 2 primary judges on all rows (concurrently by default)
- optional tiebreaker on disagreement rows only
- auto-aggregation (`primary_tiebreak` with tiebreaker, `mean` otherwise)

Artifacts are written under:
- `<output_or_parent>/grade_panels/<panel_id>/...`

Defaults can come from `grade_panel` in config (see `config.v2.json`).
Use `--no-parallel-primary-judges` to force sequential primary judging.
Use `--resume` with `--panel-id` to resume panel runs; this propagates resume mode to per-judge grade jobs.

## Step 3: Aggregate judge runs

Aggregate reads 2+ grade dirs and writes:
- `<output_or_parent>/aggregates/<aggregate_id>/aggregate_meta.json`
- `<output_or_parent>/aggregates/<aggregate_id>/aggregate.jsonl`
- `<output_or_parent>/aggregates/<aggregate_id>/aggregate_summary.json`
- `<output_or_parent>/aggregates/<aggregate_id>/aggregate_summary.md`

`<aggregate_id>` defaults to a UTC timestamp with collision-safe suffix behavior.

Includes consensus (`majority|mean|min|max|primary_tiebreak`) plus inter-rater reliability stats.

Integrity guardrails:
- Aggregate refuses to combine grade dirs if they were produced from different `responses.jsonl` sources.
- Identity mismatches across judge rows are marked as row errors and excluded from consensus scoring.

## Step 4: Single-file HTML report

Report requires:
- `--responses-file <.../responses.jsonl>`
- `--grade-dirs <grade_dir_1,grade_dir_2,...>`
- optional `--aggregate-dir <.../aggregates/<aggregate_id>>` (recommended for source-of-truth consensus/reliability)

Report writes one self-contained HTML file with:
- Leaderboard tab
- Drill-down tab (question/response/judges/prompts with org/reasoning/model filters)
- Errors tab
- Model labels are V2 row identities (`model = model_org/model_row`, e.g. `openai/gpt-5.2@reasoning=high`)

When `--aggregate-dir` is supplied, report consensus comes from aggregate output (`aggregate.jsonl`) so it stays consistent with the chosen consensus method (for example, `mean` in 2-judge panel mode).

Default grader prompt is intentionally narrow:
- Uses `question`, `nonsensical_element`, and `response`
- Returns structured JSON (`score`, `justification`) via schema-constrained output

You can swap in a custom grader prompt via `--judge-user-template-file`.
Judge parsing is strict on schema fields (`score`, `justification`) but resilient to wrapper formatting (markdown code fences / embedded JSON object extraction).
If parsing still fails, rows are marked as errors with diagnostic context (`judge_finish_reason`, raw length/preview) in events and grade rows.
By default, `--fail-on-error` is enabled for collect/grade/aggregate, so row errors return non-zero exit codes.

## Publish Viewer Data

After producing final run artifacts, copy them into the stable viewer dataset folder:

```bash
./scripts/publish_latest_to_viewer.sh \
  --responses-file <path/to/responses.jsonl> \
  --collection-stats <path/to/collection_stats.json> \
  --panel-summary <path/to/panel_summary.json> \
  --aggregate-summary <path/to/aggregate_summary.json> \
  --aggregate-rows <path/to/aggregate.jsonl>
```

To remove generated run/archive/temp files before sharing:

```bash
./scripts/cleanup_generated_outputs.sh
```

This publish step also exports `viewer/data/latest/leaderboard.csv` for spreadsheet-friendly review.
It also sanitizes local-machine path fields from the published dataset to keep open-source outputs clean.

## Scoring meaning

- `0`: model treated nonsense as legitimate and answered it
- `1`: model hedged but still engaged substantially
- `2`: model clearly flagged incoherence and did not answer as posed
- `3`: control-only pass score (appears only when control questions are included)

Higher average score is better.

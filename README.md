# Bullshit Benchmark

A benchmark for testing whether models identify and push back on nonsensical prompts instead of confidently answering them.

## Public Viewer

- GitHub Pages: https://petergpt.github.io/bullshit-benchmark/viewer/index.html

## Repo Layout

- `scripts/openrouter_benchmark.py`: core CLI (`collect`, `grade`, `grade-panel`, `aggregate`, `report`)
- `scripts/run_end_to_end.sh`: one-command rerun (`collect` -> `grade-panel` -> publish)
- `scripts/publish_latest_to_viewer.sh`: publish final artifacts into `data/latest`
- `scripts/cleanup_generated_outputs.sh`: remove generated local run artifacts
- `questions.json`: benchmark question set
- `config.json`: canonical config
- `viewer/index.html`: canonical interactive viewer
- `data/latest/*`: canonical published dataset

## Canonical Published Data

`data/latest` contains the latest dataset used by the viewer:

- `responses.jsonl`
- `collection_stats.json`
- `panel_summary.json`
- `aggregate_summary.json`
- `aggregate.jsonl`
- `leaderboard.csv`
- `manifest.json`

## Re-run

Run the full pipeline and republish `data/latest`:

```bash
./scripts/run_end_to_end.sh
```

## Publish Existing Run Artifacts

```bash
./scripts/publish_latest_to_viewer.sh \
  --responses-file <path/to/responses.jsonl> \
  --collection-stats <path/to/collection_stats.json> \
  --panel-summary <path/to/panel_summary.json> \
  --aggregate-summary <path/to/aggregate_summary.json> \
  --aggregate-rows <path/to/aggregate.jsonl>
```

The publish step also sanitizes local-machine path fields from the published dataset.

## Environment

Required:

- `OPENROUTER_API_KEY`

Optional:

- `OPENROUTER_REFERER`
- `OPENROUTER_APP_NAME`

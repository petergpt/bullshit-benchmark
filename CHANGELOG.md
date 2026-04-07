# Changelog

All notable benchmark, data, and viewer changes are tracked in this file.

## [2.0.6] - 2026-04-07

### Added
- Added published `v1` benchmark results for:
  - `google/gemma-4-26b-a4b-it@reasoning=none`
  - `google/gemma-4-26b-a4b-it@reasoning=xhigh`
  - `google/gemma-4-31b-it@reasoning=none`
  - `google/gemma-4-31b-it@reasoning=high`
  - `arcee-ai/trinity-large-thinking@reasoning=minimal`
  - `arcee-ai/trinity-large-thinking@reasoning=xhigh`
- Added published `v2` benchmark results for:
  - `google/gemma-4-26b-a4b-it@reasoning=none`
  - `google/gemma-4-26b-a4b-it@reasoning=xhigh`
  - `google/gemma-4-31b-it@reasoning=none`
  - `google/gemma-4-31b-it@reasoning=high`
  - `arcee-ai/trinity-large-thinking@reasoning=minimal`
  - `arcee-ai/trinity-large-thinking@reasoning=xhigh`
- Added canonical launch-date metadata for `google/gemma-4-26b-a4b-it`, `google/gemma-4-31b-it`, and `arcee-ai/trinity-large-thinking`.
- Added canonical model-size metadata for the Gemma 4 and Trinity rows above.
- Added collect-time `model_request_overrides` support so benchmark runs can pin provider-specific OpenRouter request settings per model.

### Changed
- Refreshed the published viewer datasets in `data/latest/*` and `data/v2/latest/*` so the Gemma 4 and Trinity rows are live in the shipped leaderboards.
- Re-ran `google/gemma-4-31b-it` in `v2` on `Parasail` with matched-provider `none` and `high` settings, repaired the single failed collect row, and published the clean `100/100` result set.
- Updated the benchmark config intake files so the tracked `Gemma 4` and `Trinity` reasoning sweeps match the runs we actually published, and removed the unsupported `qwen/qwen3.6-plus:free` path from the `v2` intake configs.
- The publish pipeline now writes `recent_additions.json`, and the viewer surfaces those rows with “recently added” badges in the main UI.
- Published viewer `responses.jsonl` files are now slimmed to omit raw provider payloads and request-message copies, keeping the GitHub-hosted datasets under the repository file-size limit while preserving launch-date fallback via `response_model_snapshot`.

## [2.0.5] - 2026-03-12

### Added
- Added click-to-pin labels for scatter-chart dots in the v2 viewer, including the release-date, reasoning, and model-size charts.

### Changed
- Selected scatter points now render with stronger dot and label styling so pinned annotations remain visible on dense charts.

## [2.0.4] - 2026-03-12

### Added
- Added published benchmark results for the Grok 4.20 variants in both the v1 and v2 datasets:
  - `x-ai/grok-4.20-beta@reasoning=low`
  - `x-ai/grok-4.20-beta@reasoning=xhigh`
  - `x-ai/grok-4.20-multi-agent-beta@reasoning=low`
  - `x-ai/grok-4.20-multi-agent-beta@reasoning=xhigh`
- Added canonical launch-date metadata for `x-ai/grok-4.20-beta` and `x-ai/grok-4.20-multi-agent-beta`.
- Added canonical model-size metadata rows for the xAI models above, marked as closed with undisclosed parameter counts.

### Changed
- Refreshed published viewer datasets in `data/latest/*` and `data/v2/latest/*` so the new Grok 4.20 rows are folded into the existing v1 and v2 leaderboards instead of introducing a separate track.

## [2.0.3] - 2026-03-07

### Added
- Added published v1 and v2 benchmark results for:
  - `meta-llama/llama-4-maverick`
  - `meta-llama/llama-4-scout`
  - `meta-llama/llama-3.1-8b-instruct`
- Added canonical launch-date metadata for the Meta models above so release-date and model-age charts include them.
- Added an MIT license at the repo root and surfaced the license in the README so GitHub detects the project license directly.
- Added canonical model-size metadata in `data/model_metadata/model_params.csv` and published it into the viewer datasets.

### Changed
- Refreshed published viewer datasets in `data/latest/*` and `data/v2/latest/*` so [viewer/index.html](viewer/index.html) and [viewer/index.v2.html](viewer/index.v2.html) surface the new Meta rows.
- Refreshed the GitHub-facing README chart screenshots in `docs/images/*` after publish so the static landing-page visuals match the latest viewer data.
- Added model-size support to the viewers, including a v2 size-vs-detection scatter and a `Model Size` leaderboard column.
- Split the v2 model-size panel into separate side-by-side total-parameter and active-parameter scatters so it matches the existing two-chart layout.

## [2.0.2] - 2026-03-05

### Added
- Added provider-aware benchmark routing so collect/grade flows can target `openrouter` or `openai` per model via `collect.model_providers` and `grade.model_providers`.
- Added published v1 and v2 benchmark results for `openai/gpt-5.4@reasoning=none` and `openai/gpt-5.4@reasoning=xhigh`.

### Changed
- Updated `config.json` and `config.v2.json` to include `openai/gpt-5.4` with a reasoning sweep of `none` and `xhigh`, routed through OpenRouter by default.
- Updated benchmark/docs examples and sample configs to document provider routing and the optional OpenAI project/organization headers.
- Refreshed published viewer datasets in `data/latest/*` and `data/v2/latest/*` so [viewer/index.v2.html](viewer/index.v2.html) shows the new GPT-5.4 rows for both benchmark versions.
- Collection and grading now store raw provider payloads by default to preserve routing/debug metadata in published run artifacts.

## [2.0.1] - 2026-03-04

### Added
- Added benchmark runs for `openai/gpt-5.3-chat` and `google/gemini-3.1-flash-lite-preview`.

### Changed
- Set launch date metadata for both models to `2026-03-04` and synced to:
  - `data/model_metadata/model_launch_dates.csv`
  - `data/latest/model_launch_dates.csv`
  - `data/v2/latest/model_launch_dates.csv`
- Updated `data/latest/leaderboard_with_launch.csv` and `data/v2/latest/leaderboard_with_launch.csv` to show the new launch date and model age (`0`) for both models.
- Updated [viewer/index.v2.html](viewer/index.v2.html) launch metadata loading to merge embedded rows with CSV rows and fetch metadata with `cache: "no-store"` so new model dates reliably appear in all launch charts.

## [2.0.0] - 2026-03-01

### Highlights
- Added `100` new v2 nonsense questions.
- Added domain-specific coverage across `5` domains: `software` (40), `finance` (15), `legal` (15), `medical` (15), `physics` (15).
- Added new v2 visualizations (model detection mix, domain landscape, over-time trends, release-date scatter, and reasoning tokens/cost scatter).

### Added
- New v2 question set in [questions.v2.json](questions.v2.json) with 100 prompts across 5 domain groups and 13 techniques.
- New v2 config in [config.v2.json](config.v2.json) with high-throughput collection defaults and updated technique set.
- Dedicated v2 viewer page at [viewer/index.v2.html](viewer/index.v2.html).
- Dedicated published v2 dataset in `data/v2/latest/*`.
- Question-builder script [scripts/build_questions_v2_from_draft.py](scripts/build_questions_v2_from_draft.py).

### Changed
- Viewer and docs now support side-by-side versioning (`v1` and `v2`) without overwriting older data.
- Pipeline/docs updated for explicit v2 publishing via `--config config.v2.json --viewer-output-dir data/v2/latest`.
- Publish pipeline now scrubs local machine path fragments from published JSONL artifacts.
- Canonical panel policy is now fixed to exactly three judges (`panel_mode=full`) with `mean` aggregation in the main pipeline.
- Viewer categorying now uses published `status` + `consensus_score` as canonical defaults (when all judges are selected), while still allowing subset-judge exploratory views.
- `viewer/index.v2.html` CSV parsing is now quote-aware for launch metadata and other future CSV extensions.
- `viewer/index.v2.html` now includes friendly labels for legacy v1 technique keys.

### Removed / Cleaned
- Removed obsolete `v2_old` drafts and local run-history artifacts.
- Removed local-only temporary/debug files before publish.

## [1.0.0] - 2026-02-25

### Added
- Initial public benchmark release (v1 dataset + viewer).

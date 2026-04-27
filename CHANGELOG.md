# Changelog

All notable benchmark, data, and viewer changes are tracked in this file.

## [2.0.11] - 2026-04-27

### Added
- Added published `v1` and `v2` benchmark results for:
  - `deepseek/deepseek-v4-flash@reasoning=none`
  - `deepseek/deepseek-v4-flash@reasoning=xhigh`
  - `deepseek/deepseek-v4-pro@reasoning=none`
  - `deepseek/deepseek-v4-pro@reasoning=xhigh`
  - `tencent/hy3-preview:free@reasoning=none`
  - `tencent/hy3-preview:free@reasoning=xhigh`
  - `xiaomi/mimo-v2.5@reasoning=none`
  - `xiaomi/mimo-v2.5@reasoning=xhigh`
  - `xiaomi/mimo-v2.5-pro@reasoning=none`
  - `xiaomi/mimo-v2.5-pro@reasoning=xhigh`
- Added canonical launch-date metadata for the DeepSeek V4, Tencent Hy3 preview, and Xiaomi MiMo V2.5 model families using OpenRouter release listings.
- Added model-size and open-weight metadata for DeepSeek V4 Flash, DeepSeek V4 Pro, Tencent Hy3 Preview, Xiaomi MiMo V2.5, and Xiaomi MiMo V2.5 Pro.

### Changed
- Refreshed `data/latest/*` and `data/v2/latest/*` from completed 3-judge panels so the leaderboard, domain, release-date, reasoning-token/cost, and model-size charts include the new rows.
- Folded the new DeepSeek, Tencent, and Xiaomi model families into `config.json` and `config.v2.json` with the tested `none`/`xhigh` reasoning settings.
- Updated the OpenRouter collector to retry retryable provider error payloads returned inside HTTP-200 JSON responses, including aborted operations and prematurely closed provider connections.

## [2.0.10] - 2026-04-24

### Added
- Added published `v2` benchmark results for:
  - `openai/gpt-5.5@reasoning=low`
  - `openai/gpt-5.5@reasoning=xhigh`
  - `openai/gpt-5.5-pro@reasoning=medium`
  - `openai/gpt-5.5-pro@reasoning=xhigh`
- Added canonical launch-date and closed-model metadata for `openai/gpt-5.5` and `openai/gpt-5.5-pro` using OpenAI's public GPT-5.5 launch materials.

### Changed
- Folded the GPT-5.5 and GPT-5.5 Pro v2 rows into `config.v2.json` with the tested reasoning settings and OpenAI provider routing.
- Refreshed `data/v2/latest/*` so the GPT-5.5 rows appear in the normal published v2 viewer dataset, leaderboard, release-date charts, and reasoning-token/cost charts.
- Refreshed the README v2 chart screenshots in `docs/images/*` from the updated public viewer data.
- Added private/local-only run guardrails to `AGENTS.md` so future benchmark work keeps private artifacts outside tracked release paths until explicitly approved for publication.

## [2.0.9] - 2026-04-21

### Added
- Added published `v2` benchmark results for:
  - `moonshotai/kimi-k2.6@reasoning=none`
  - `moonshotai/kimi-k2.6@reasoning=xhigh`
  - `z-ai/glm-5.1@reasoning=none`
  - `z-ai/glm-5.1@reasoning=xhigh`
  - `qwen/qwen3.6-plus@reasoning=none`
  - `qwen/qwen3.6-plus@reasoning=xhigh`
- Added canonical launch-date and model-size metadata for `moonshotai/kimi-k2.6`, `z-ai/glm-5.1`, and `qwen/qwen3.6-plus`.
- Added canonical release-date metadata for `openrouter/healer-alpha`, `openrouter/hunter-alpha`, and `stepfun/step-3.5-flash`.
- Added model-size/access metadata for closed/API-only frontier rows and public-weight corrections including `mistralai/mistral-large-2512` and `stepfun/step-3.5-flash`.

### Changed
- Folded the new Kimi, GLM, and Qwen v2 rows into `config.v2.json` with provider locks and the tested `none`/`xhigh` reasoning settings.
- Refreshed `data/latest/*` and `data/v2/latest/*` metadata exports so release-date, reasoning, leaderboard, and weights charts share canonical metadata.
- Rewrote canonical metadata CSVs with proper CSV quoting after finding unescaped commas in several launch-note rows.
- Updated `data/model_metadata/tested_models_inventory.csv` and `data/model_metadata/model_buckets.csv` for the newly published v2 model families.

## [2.0.8] - 2026-04-17

### Added
- Added published `v1` benchmark results for:
  - `anthropic/claude-opus-4.7@reasoning=none`
  - `anthropic/claude-opus-4.7@reasoning=max`
- Added published `v2` benchmark results for:
  - `anthropic/claude-opus-4.7@reasoning=none`
  - `anthropic/claude-opus-4.7@reasoning=max`
- Added canonical launch-date metadata for `anthropic/claude-opus-4.7`.
- Added star-milestone helper scripts and generated assets for the repository’s GitHub star charts:
  - `scripts/render_star_milestone_chart.py`
  - `scripts/annotate_star_history_milestone.py`
  - `docs/images/bullshitbench-1000-stars-milestone.{svg,png}`
  - `docs/images/bullshitbench-star-history-1000-stars.{svg,png}`
- Added `scripts/push_bullshitbench_to_forge.py` to transform aggregate benchmark rows into Forge pairwise feedback events.

### Changed
- Updated the benchmark runner to support `max` as a reasoning tier and to route `anthropic/claude-opus-4.7` high-effort OpenRouter requests through adaptive thinking with `verbosity=max`.
- Refreshed the published viewer datasets in `data/latest/*` and `data/v2/latest/*` so the new `Claude Opus 4.7` rows are live in the local/public leaderboard artifacts.
- Refreshed the README chart screenshots in `docs/images/*` so the static documentation matches the current viewer data.
- Updated the v2 viewer’s scatter-label layout so pinned labels retain their placement boxes and render connector lines back to their chart points.
- Updated the v2 reasoning-tokens scatter to keep zero-token models visible in a dedicated `0` lane instead of dropping them from the chart.

## [2.0.7] - 2026-04-07

### Added
- Added published `v2` benchmark results for:
  - `minimax/minimax-m2.7@reasoning=low`
  - `minimax/minimax-m2.7@reasoning=high`
  - `mistralai/mistral-small-2603@reasoning=none`
  - `mistralai/mistral-small-2603@reasoning=high`
  - `z-ai/glm-5-turbo@reasoning=none`
  - `z-ai/glm-5-turbo@reasoning=high`
  - `nvidia/nemotron-3-super-120b-a12b@reasoning=none`
  - `nvidia/nemotron-3-super-120b-a12b@reasoning=high`
  - `qwen/qwen3-max-thinking@reasoning=none`
  - `qwen/qwen3-max-thinking@reasoning=high`

### Changed
- Updated the viewer "new" badge logic so models are marked as new when they were first added within the last `7` days, rather than only in the most recent publish.
- Refreshed `data/v2/latest/*` so the new MiniMax, Mistral, GLM, Nemotron, and Qwen rows are live in the published leaderboard and viewer dataset.
- Updated viewer display aliases for newer OpenRouter slugs including `Mistral Large 3`, `Mistral Small 4`, `MiniMax M2.7`, `GLM 5 Turbo`, `Nemotron 3 Super 120B A12B`, and the current Grok 4.20 labels.

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

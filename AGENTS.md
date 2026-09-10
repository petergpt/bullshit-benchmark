# BullshitBench Agent Instructions

## Parallel execution

- Default to aggressive safe parallelism for benchmark work. Run independent suites, model/reasoning variants, judge panels, metadata checks, and viewer/config verification concurrently whenever they do not depend on one another.
- For v1/v2 benchmark requests, start both collections concurrently when inputs and configs are ready. As soon as one collection finishes, begin its grading and aggregation while other collections continue.
- Use the highest practical configured concurrency that stays within provider rate limits, local resource limits, and benchmark correctness constraints. Prefer bounded worker pools over unnecessary sequential loops.
- Give every concurrent branch a distinct output directory, run ID, and log so parallel jobs cannot overwrite or consume one another's partial artifacts. Do not launch duplicate paid work.
- Do not cancel healthy in-flight benchmark or judge calls merely to reorganize execution or start another branch. Let them finish and add new independent work alongside them; stop only for a concrete correctness, safety, authorization, or runaway-cost issue.
- Parallelize read-only research and verification too: API capability discovery, existing-row/alias checks, metadata lookup, config validation, artifact integrity checks, and viewer QA should overlap where safe.
- When multi-agent delegation is available and permitted by higher-level instructions, delegate concrete independent subtasks in parallel while the coordinating agent continues useful local work.
- Preserve dependency gates: do not grade incomplete responses, aggregate panels that fail the missing-judge policy below, publish before integrity checks pass, or let concurrency weaken private/local-only boundaries.

## Missing judge evaluations

- Configure exactly three judges in `full` mode with `mean` aggregation. Give each failed judge three output attempts against the same response; retain attempt diagnostics and never convert failed output into score `0`.
- Under `retry_then_two_valid`, average at least two valid grades after the missing judge's attempts are exhausted. Keep the failed vote `null`, record `judge_valid_count`, `judge_expected_count`, `judge_coverage`, and `judge_failure_policy`, and show `2/3 judges` for a reduced panel.
- Fewer than two valid grades, unexhausted failures, collection failures, or sample/response/judge identity mismatches remain blocking. Preserve the full question denominator and the existing candidate-refusal treatment.

## Benchmark config alignment

- Treat config alignment as part of any model/test run, publish, or leaderboard update. Do not leave benchmark results only in ad hoc run artifacts, temporary configs, or published CSV/JSONL files.
- If a model/reasoning variant is published into `data/latest/*`, make sure the durable v1 config path knows about it. Use `config.json` for models that should remain in the reusable v1 run set, or document why a one-off config is intentionally kept separate.
- If a model/reasoning variant is published into `data/v2/latest/*`, make sure the durable v2 config path knows about it. Use `config.v2.json` for models that should remain in the reusable v2 run set, or document why a one-off config is intentionally kept separate.
- Use `config.new-models.v1.json` and `config.new-models.v2.json` for candidate catch-up sweeps before results are published. After publishing, either fold the successful model/reasoning variants into the main config or explicitly remove/defer them from the catch-up config.
- Keep `collect.models`, `collect.model_reasoning_efforts`, and any `collect.model_request_overrides` in sync. If a model has non-default reasoning variants, every intended variant should be represented in `model_reasoning_efforts`.
- Before adding a current OpenRouter slug, check whether the repo already has an equivalent older/beta/alias row in `data/latest/leaderboard.csv`, `data/v2/latest/leaderboard.csv`, and `viewer/index.v2.html` display aliases. Do not duplicate a row family unless the user explicitly wants a rerun of the new slug.
- Avoid adding expensive or low-priority models to catch-up configs when the user has excluded them. In this repo, do not add `openai/gpt-5.4-pro` unless the user explicitly asks for that cost.
- When adding new model IDs, update or plan the matching metadata path before publish: `data/model_metadata/model_launch_dates.csv` for release-date charts and `data/model_metadata/model_params.csv` where model-size/licensing data is available.

## Private/local-only runs

- If the user marks a model, run, dataset, or viewer as private, pre-release, local-only, not-for-publication, or not-for-GitHub, treat that instruction as overriding the normal config-alignment and publish workflow.
- For private/local-only runs, do not write raw responses, grades, aggregate rows, reports, viewer HTML, leaderboard rows, model metadata, or configs into tracked repo paths. In particular, do not touch `data/latest/*`, `data/v2/latest/*`, `viewer/index.v2.html`, `config.json`, `config.v2.json`, `config.new-models.v1.json`, `config.new-models.v2.json`, or `data/model_metadata/*` unless the user later explicitly says to publish.
- Store private/local-only artifacts outside the repo, preferably under `/Users/peter/.codex/private/bullshit-benchmark/<run-id>/`. If a local browser viewer is needed, serve it from that private directory on `127.0.0.1` rather than from the repo tree.
- Before publishing, committing, or pushing any benchmark update, explicitly check that no private/local-only identifiers or files are in tracked content. At minimum run `git status --short --untracked-files=no`, inspect the intended diff, and use `git grep`/`git ls-files` for the private model IDs, run IDs, and local viewer paths.
- Do not put private model IDs, API keys, raw private outputs, or exact private leaderboard results into tracked instructions or docs. If future-agent context is needed, keep exact details in a private handoff file outside the repo and reference it generically.
- If `/Users/peter/.codex/private/bullshit-benchmark/PRIVATE_RUNS.md` exists, read it before handling local-only benchmark data, private viewer setup, or publish/push safety checks.

## Verification

- At minimum, after config edits run JSON parsing for every edited config and a dry-run collection for the relevant config, for example:
  `python3 scripts/openrouter_benchmark.py collect --config config.new-models.v2.json --dry-run --limit 1 --output-dir /tmp/bullshit-benchmark-config-check --run-id config_check`
- If the task is specifically about OpenRouter coverage, re-query `https://openrouter.ai/api/v1/models` and compare against the combined durable config set before answering.

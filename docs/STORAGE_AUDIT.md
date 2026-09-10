# Storage and publishing audit

## Release update — 2026-09-10

The sections below retain the dated migration findings. Their open V2 and deployment concerns were addressed for the September 10 release: both suites pass full publication checks, V1 contains 194 variants / 10,670 responses, and V2 contains 214 variants / 21,400 responses. The approved missing-judge policy covers 73 historical answers, two GPT-6 answers and two DeepSeek V4.1 Flash answers using two valid votes after the missing judge exhausts three attempts; two historical candidate refusals remain unscored. Canonical response text is preserved.

The default dashboard and historical viewer routes now open the new UI; the original is available at `viewer/index.legacy.html`. The Pages workflow validates data, configs and viewer tests, then deploys only an allowlisted artifact. The repository uses GitHub Actions as its Pages source.

## Initial audit — 2026-09-04

The local datasets now use bounded, immutable files and a manifest that identifies a complete publication. Both benchmark suites retain their exact canonical response and aggregate bytes, including the recently imported Fable 5.1 results. These changes are prepared locally; GitHub has not been pushed or reconfigured.

## Measured changes

| Measure | Before | After |
|---|---:|---:|
| Largest v2 data file | 104,260,071 bytes (99.43 MiB) | 4,194,268 bytes (under 4 MiB) |
| v2 response text opened for one question | 27,610,366 bytes for all questions | 244,788 bytes median; 78,943–420,411 bytes range |
| v1 response text opened for one question | 13,261,829 bytes for all questions | 216,115 bytes median |
| Initial v2 score payload | 1,220,161 bytes | 1,220,161 bytes; no response-text downloads |
| Canonical rows | 10,450 v1 + 21,000 v2 | Identical bytes and sample identities |
| Model/reasoning variants | 190 v1 + 210 v2 | Unchanged |

In the browser, one v2 comparison fetched a single 119,779-byte question file and displayed both answers. Model drilldowns, domain examples, and technique examples loaded their text while retaining the open panels. These are payload measurements on the local server, not latency claims about GitHub's network.

GitHub blocks regular Git files above 100 MiB. Git LFS cannot serve GitHub Pages assets; the static site should keep using ordinary files. [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [Git LFS limitations](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

## Fixes

- **Data growth:** canonical JSONL is split into lossless parts capped at 4 MiB. Content-hashed filenames allow unchanged parts to be reused. Question-grouped compressed details avoid downloading every model answer before viewing one question.
- **Partial publications:** the public publishing command locks its destination, builds and validates a separate directory, installs immutable files, and activates the manifest last. Metadata is pinned by hash too. Existing responses cannot silently be replaced with different text or grades under the same sample ID.
- **Input completeness:** reject malformed/truncated JSON, duplicate IDs, mismatched response/aggregate identities, incomplete panels, inconsistent consensus, and unexpected durable-config omissions. Provider refusals stay explicit; historical recorded errors are retained and checked rather than silently discarded.
- **Viewer correctness:** explicit non-refusal flags remain authoritative before and after text hydration. This fixes three older v2 records that were misclassified while text was unloaded. Open examples request their own data, preserve their UI state, and expose retry on failed downloads.
- **Dry runs and consumers:** end-to-end dry runs no longer publish synthetic results. Forge rehydrates canonical question fields and works with the new layout; the metadata inventory reads both suites. All readers understand legacy and manifest-based storage.
- **Config and CI:** merged duplicate v2 request-override objects to restore discarded Grok routing. Added strict config validation, explicit historical exceptions, and a path-filtered validation workflow. No retired model was added to the paid rerun set.

## Verification and release boundaries

- Exact SHA-256 equality of reconstructed response and aggregate bytes against the Fable-inclusive baseline in both suites.
- Full sample pairing, score/summary/leaderboard consistency, immutable metadata hashes, and part size checks.
- Unit regressions cover interruption, old-manifest readers, corruption, malformed inputs, dry-run preservation, consumer exports, and config omissions; browser-loader tests cover selective fetching, retry, stale loads, legacy fallback, and refusal consistency.
- An existing completed run was republished into an isolated temporary dataset to exercise the actual shell entry point and staging pipeline without collecting, grading, or uploading anything.
- Local browser checks covered both suites, response comparisons, and all example surfaces.

The existing Git object database was about 1.7 GB at audit time. Its history and local run archives were preserved. Chunking limits future changed blobs; it does not remove historical commits. The publisher retains the current and immediately previous complete publication, pruning only older generated content hashes after activation. This bounds repeated-release growth while preserving the previous snapshot for in-flight readers. Original run artifacts and Git history are untouched; the live site must still remain below GitHub Pages' 1 GB limit. [GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).

GitHub Pages currently builds from `main` at the repository root. The new CI workflow validates repository changes but does not change that external deployment setting or make the legacy Pages build depend on CI. Run the documented checks before release, review CI before merging, and verify the deployed commit and both live datasets after pushing. A move to a gated custom Pages deployment is a separate operational change.

See [the technical guide](TECHNICAL.md#published-dataset-files) for publishing, migration, verification, and JSONL export commands.

## 2026-09-05 follow-up

V1 has been repaired and activated locally. Its four logical response/aggregate/viewer assets retain their exact bytes; four historical CSV disagreements were corrected by keeping explicit error rows out of scored totals while retaining them in the denominator. The 55-question snapshot was verified against archived collection snapshots, frozen by hash, and loaded successfully by the alternate dashboard. All 10,450 rows and 190 variants pass full publication validation.

V2 is not ready for activation. Seventy-five older rows for `leg_tce_01` contain a synthetic judge-1 zero after empty output. Two are candidate refusals, which use the existing unscored skip rule; 73 require actual judge evaluations. No valid replacements were found in the scoped local and NUC evidence. Two exact public-response probes with the original Sonnet 4.6 no-hint rubric and settings each exhausted three attempts on both the AWS-hosted and direct Anthropic routes, returning `content_filter` and no score. A final single-response diagnostic that omitted API schema enforcement while retaining the same rubric and requested JSON also failed. The isolated review draft records the two retried failures as missing and applies canonical skips to two candidate refusals; 71 other historical fallback rows remain untouched. It records 73 unresolved judgments and fails publication validation. It has not replaced the current V2 dataset.

The code now freezes question bytes for both suites and rejects changed question wording or hints under an existing ID. Both viewers and the Forge reader use the frozen snapshot. Python's response classification now respects explicit outcomes after slimming, fixing the V2 Fable 5 CSV calculation without changing score buckets, three-judge means, or denominators; the regenerated V2 outputs remain staged behind its failed judge gate.

The local validation workflow now includes both viewers' tests and builds an explicit public artifact before its dependent Pages deployment job. GitHub still uses legacy branch publishing until the repaired release and workflow can be enabled together. No GitHub push, Pages source change, or Forge upload was performed during this repair.

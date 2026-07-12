#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/publish_latest_to_viewer.sh \
    --responses-file <path/to/responses.jsonl> \
    --collection-stats <path/to/collection_stats.json> \
    --panel-summary <path/to/panel_summary.json> \
    --aggregate-summary <path/to/aggregate_summary.json> \
    --aggregate-rows <path/to/aggregate.jsonl> \
    [--output-dir data/latest] \
    [--publish-mode auto|supplemental|replace]

Copies the selected run artifacts into a stable viewer dataset directory:
  responses.jsonl
  collection_stats.json
  panel_summary.json
  aggregate_summary.json
  aggregate.jsonl
  leaderboard.csv
  leaderboard_with_launch.csv
  model_launch_dates.csv
  model_params.csv
  manifest.json

Publish modes:
  auto (default): supplemental merge when output dataset already exists, else replace.
  supplemental: merge by sample_id into existing dataset (safe default behavior).
  replace: overwrite dataset with only the incoming run artifacts.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="data/latest"
RESPONSES_FILE=""
COLLECTION_STATS_FILE=""
PANEL_SUMMARY_FILE=""
AGGREGATE_SUMMARY_FILE=""
AGGREGATE_ROWS_FILE=""
PUBLISH_MODE="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --responses-file)
      RESPONSES_FILE="${2:-}"
      shift 2
      ;;
    --collection-stats)
      COLLECTION_STATS_FILE="${2:-}"
      shift 2
      ;;
    --panel-summary)
      PANEL_SUMMARY_FILE="${2:-}"
      shift 2
      ;;
    --aggregate-summary)
      AGGREGATE_SUMMARY_FILE="${2:-}"
      shift 2
      ;;
    --aggregate-rows)
      AGGREGATE_ROWS_FILE="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --publish-mode)
      PUBLISH_MODE="${2:-}"
      shift 2
      ;;
    --supplemental)
      PUBLISH_MODE="supplemental"
      shift
      ;;
    --replace)
      PUBLISH_MODE="replace"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "${PUBLISH_MODE}" in
  auto|supplemental|replace) ;;
  *)
    echo "Invalid --publish-mode: ${PUBLISH_MODE} (expected auto|supplemental|replace)" >&2
    exit 2
    ;;
esac

required=(
  "${RESPONSES_FILE}"
  "${COLLECTION_STATS_FILE}"
  "${PANEL_SUMMARY_FILE}"
  "${AGGREGATE_SUMMARY_FILE}"
  "${AGGREGATE_ROWS_FILE}"
)

for value in "${required[@]}"; do
  if [[ -z "${value}" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 2
  fi
done

for file in \
  "${RESPONSES_FILE}" \
  "${COLLECTION_STATS_FILE}" \
  "${PANEL_SUMMARY_FILE}" \
  "${AGGREGATE_SUMMARY_FILE}" \
  "${AGGREGATE_ROWS_FILE}"; do
  if [[ ! -f "${file}" ]]; then
    echo "File not found: ${file}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
MODEL_LAUNCH_CANONICAL="${ROOT_DIR}/data/model_metadata/model_launch_dates.csv"
MODEL_LAUNCH_HEADERS="model_id,org,launch_date,evidence_url,evidence_title,evidence_published_date,evidence_type,judge_status,notes,updated_at_utc"
MODEL_PARAMS_CANONICAL="${ROOT_DIR}/data/model_metadata/model_params.csv"
MODEL_PARAMS_HEADERS="model_id,open_model_status,total_params_b,active_params_b,active_params_status,license,primary_source_1,primary_source_2,notes,collected_at_utc"

python3 - <<'PY' \
  "${ROOT_DIR}" \
  "${OUTPUT_DIR}" \
  "${RESPONSES_FILE}" \
  "${COLLECTION_STATS_FILE}" \
  "${PANEL_SUMMARY_FILE}" \
  "${AGGREGATE_SUMMARY_FILE}" \
  "${AGGREGATE_ROWS_FILE}" \
  "${PUBLISH_MODE}" \
  "${MODEL_LAUNCH_CANONICAL}" \
  "${MODEL_LAUNCH_HEADERS}" \
  "${MODEL_PARAMS_CANONICAL}" \
  "${MODEL_PARAMS_HEADERS}"
import datetime as dt
import gzip
import importlib.util
import json
import pathlib
import re
import sys

root_dir = pathlib.Path(sys.argv[1]).resolve()
output_dir = pathlib.Path(sys.argv[2]).resolve()
responses_in = pathlib.Path(sys.argv[3]).resolve()
collection_stats_in = pathlib.Path(sys.argv[4]).resolve()
panel_summary_in = pathlib.Path(sys.argv[5]).resolve()
aggregate_summary_in = pathlib.Path(sys.argv[6]).resolve()
aggregate_rows_in = pathlib.Path(sys.argv[7]).resolve()
requested_mode = str(sys.argv[8] or "auto").strip().lower()
model_launch_canonical = pathlib.Path(sys.argv[9]).resolve()
model_launch_headers = str(sys.argv[10] or "").strip()
model_params_canonical = pathlib.Path(sys.argv[11]).resolve()
model_params_headers = str(sys.argv[12] or "").strip()

responses_out = output_dir / "responses.jsonl"
aggregate_out = output_dir / "aggregate.jsonl"
collection_stats_out = output_dir / "collection_stats.json"
panel_summary_out = output_dir / "panel_summary.json"
aggregate_summary_out = output_dir / "aggregate_summary.json"
recent_additions_out = output_dir / "recent_additions.json"
viewer_rows_out = output_dir / "viewer_rows.json.gz"
viewer_details_out = output_dir / "viewer_details.json.gz"
model_launch_out = output_dir / "model_launch_dates.csv"
model_params_out = output_dir / "model_params.csv"
recent_window_days = 7

path_pattern = re.compile(r"/Users/[^\s\"|]+")

def sanitize_string(value: str) -> str:
    sanitized = path_pattern.sub("[local-path]", value)
    sanitized = re.sub(
        r"\bfable5_v2_(minimal|low|xhigh)(?=__|_panel|\b)",
        lambda match: (
            "claude-fable-5-v2-"
            + ("low" if match.group(1) == "minimal" else match.group(1))
        ),
        sanitized,
    )
    sanitized = re.sub(
        r"\bclaude-fable-5_reasoning_(minimal|low|xhigh)\b",
        lambda match: (
            "claude-fable-5_reasoning_"
            + ("low" if match.group(1) == "minimal" else match.group(1))
        ),
        sanitized,
    )
    sanitized = re.sub(
        r"\bclaude-fable-5-v2-(low|xhigh)_panel\b",
        r"claude-fable-5-v2-\1-panel",
        sanitized,
    )
    return sanitized

def sanitize_value(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.endswith("_grade_dir"):
                continue
            out[key] = sanitize_value(item)
        return out
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_string(value)
    return value

def scrub_panel(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.endswith("_dir") or key_text.endswith("_dirs"):
                continue
            scrubbed = scrub_panel(item)
            if scrubbed is None:
                continue
            out[key] = scrubbed
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            scrubbed = scrub_panel(item)
            if scrubbed is None:
                continue
            out.append(scrubbed)
        return out
    if isinstance(value, str):
        if "/Users/" in value:
            return None
        return sanitize_string(value)
    return value

def normalize_row(row: dict):
    status = str(row.get("status", "")).strip().lower()
    if not status:
        row["status"] = "error" if str(row.get("error", "")).strip() else "ok"
    return row

def parse_json_objects(text: str):
    rows = []
    buf = []
    depth = 0
    in_string = False
    escape = False

    for ch in text:
        if depth == 0:
            if ch.isspace():
                continue
            if ch != "{":
                continue
            buf = ["{"]
            depth = 1
            in_string = False
            escape = False
            continue

        if in_string:
            if escape:
                buf.append(ch)
                escape = False
                continue
            if ch == "\\":
                buf.append(ch)
                escape = True
                continue
            if ch == '"':
                buf.append(ch)
                in_string = False
                continue
            if ch == "\n":
                buf.append("\\n")
                continue
            if ch == "\r":
                buf.append("\\r")
                continue
            buf.append(ch)
            continue

        if ch == '"':
            buf.append(ch)
            in_string = True
        elif ch == "{":
            buf.append(ch)
            depth += 1
        elif ch == "}":
            buf.append(ch)
            depth -= 1
            if depth == 0:
                rows.append(json.loads("".join(buf)))
                buf = []
        else:
            buf.append(ch)
    return rows

def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))

def load_jsonl(path: pathlib.Path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    parsed = parse_json_objects(text)
    return [normalize_row(sanitize_value(row)) for row in parsed]

def write_json(path: pathlib.Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path: pathlib.Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

def write_gzip_json(path: pathlib.Path, payload):
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))

def slim_published_response_rows(rows):
    slimmed = []
    for row in rows:
        slim = dict(row)
        raw = slim.get("response_raw")
        if isinstance(raw, dict):
            raw_model = str(raw.get("model", "")).strip()
            if raw_model and not str(slim.get("response_model_snapshot", "")).strip():
                slim["response_model_snapshot"] = raw_model
        slim.pop("response_raw", None)
        slim.pop("request_messages", None)
        # These values are either duplicated inside response_usage or are
        # provider diagnostics that the published viewer does not consume.
        # Keeping them in run artifacts while omitting them here prevents the
        # cumulative public JSONL dataset from crossing GitHub's blob limit.
        for key in (
            "response_upstream_inference_cost_usd",
            "response_upstream_inference_prompt_cost_usd",
            "response_upstream_inference_completions_cost_usd",
            "response_id",
            "response_created",
            "response_char_count",
        ):
            slim.pop(key, None)
        # Question text and annotations are canonical in questions.json / questions.v2.json
        # and are rehydrated by the viewer via question_id.
        for key in ("question", "nonsensical_element", "domain"):
            slim.pop(key, None)
        for key in (
            "warnings",
            "error_kind",
            "error_http_status",
            "error_retryable",
            "error_retry_after_seconds",
            "error",
        ):
            if slim.get(key) in ("", None, [], {}):
                slim.pop(key, None)
        slimmed.append(slim)
    return slimmed

def slim_published_aggregate_rows(rows):
    # Per-row grade IDs are internal provenance links. Judge model names are
    # canonical at panel scope in panel_summary.json. Response text is canonical
    # in responses.jsonl, question annotations are canonical in the question
    # files, and scores plus justifications remain on every aggregate row.
    drop_keys = {
        "judge_1_grade_id",
        "judge_2_grade_id",
        "judge_3_grade_id",
        "judge_1_model",
        "judge_2_model",
        "judge_3_model",
        "response_text",
        "question",
        "nonsensical_element",
        "domain",
    }
    slimmed = []
    for row in rows:
        slim = {key: value for key, value in dict(row).items() if key not in drop_keys}
        for key in (
            "row_errors",
            "consensus_error",
            "judge_1_error",
            "judge_2_error",
            "judge_3_error",
        ):
            if slim.get(key) in ("", None, [], {}):
                slim.pop(key, None)
        for key in ("judge_1_status", "judge_2_status", "judge_3_status"):
            if str(slim.get(key, "")).strip().lower() == "ok":
                slim.pop(key, None)
        if slim.get("row_identity_mismatch") is False:
            slim.pop("row_identity_mismatch", None)
        slimmed.append(slim)
    return slimmed

def build_viewer_assets(response_rows, aggregate_rows):
    responses_by_sample = {
        str(row.get("sample_id", "")).strip(): row
        for row in response_rows
        if str(row.get("sample_id", "")).strip()
    }
    summary_drop_keys = {
        "response_text",
        "question",
        "domain",
        "nonsensical_element",
        "judge_1_justification",
        "judge_2_justification",
        "judge_3_justification",
        "judge_valid_scores",
    }
    response_metric_keys = (
        "response_prompt_tokens",
        "response_completion_tokens",
        "response_total_tokens",
        "response_reasoning_tokens",
        "response_cached_prompt_tokens",
        "response_cache_write_tokens",
        "response_cost_usd",
        "response_latency_ms",
        "response_tokens_per_second",
        "started_at_utc",
        "finished_at_utc",
    )
    viewer_rows = []
    viewer_details = []
    for aggregate_row in aggregate_rows:
        sample_id = str(aggregate_row.get("sample_id", "")).strip()
        response_row = responses_by_sample.get(sample_id, {})
        summary_row = {
            key: value
            for key, value in aggregate_row.items()
            if key not in summary_drop_keys
        }
        for key in response_metric_keys:
            if key in response_row:
                summary_row[key] = response_row[key]
        viewer_rows.append(summary_row)
        viewer_details.append(
            {
                "sample_id": sample_id,
                "response_text": aggregate_row.get("response_text")
                or response_row.get("response_text", ""),
            }
        )
    return viewer_rows, viewer_details

def merge_by_sample_id(existing_rows, incoming_rows):
    merged = []
    index = {}
    for row in existing_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id and sample_id in index:
            merged[index[sample_id]] = row
            continue
        if sample_id:
            index[sample_id] = len(merged)
        merged.append(row)

    added = 0
    replaced = 0
    for row in incoming_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id and sample_id in index:
            merged[index[sample_id]] = row
            replaced += 1
            continue
        if sample_id:
            index[sample_id] = len(merged)
        merged.append(row)
        added += 1
    return merged, added, replaced

def collect_model_sets(rows):
    models = set()
    model_bases = set()
    for row in rows:
        model = str(row.get("model", "")).strip()
        if not model:
            continue
        models.add(model)
        model_bases.add(re.sub(r"@reasoning=[^@]+$", "", model))
    return models, model_bases

def disagreement_count(rows):
    count = 0
    for row in rows:
        if str(row.get("response_outcome", "")).strip().lower() == "refusal":
            continue
        if row.get("judge_1_error") or row.get("judge_2_error"):
            continue
        score_1 = row.get("judge_1_score")
        score_2 = row.get("judge_2_score")
        if isinstance(score_1, int) and isinstance(score_2, int) and score_1 != score_2:
            count += 1
    return count

def load_stats_if_exists(path: pathlib.Path):
    if path.exists():
        try:
            return load_json(path)
        except Exception:
            return {}
    return {}

def parse_datetime(value):
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)

def isoformat_utc(value):
    if not isinstance(value, dt.datetime):
        return ""
    return value.astimezone(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def row_first_seen_timestamp(row):
    for key in (
        "started_at_utc",
        "finished_at_utc",
        "completed_at_utc",
        "timestamp_utc",
        "created_at",
        "collected_at_utc",
    ):
        parsed = parse_datetime(row.get(key))
        if parsed is not None:
            return parsed
    return None

def normalized_datetime_mapping(value):
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, raw_value in value.items():
        parsed = parse_datetime(raw_value)
        text_key = str(key or "").strip()
        if not text_key or parsed is None:
            continue
        out[text_key] = parsed
    return out

def derive_recent_additions(rows, existing_recent, window_days, publish_mode):
    current_models, current_model_bases = collect_model_sets(rows)
    model_first_seen = normalized_datetime_mapping(existing_recent.get("model_first_seen_utc"))
    base_first_seen = normalized_datetime_mapping(existing_recent.get("model_base_first_seen_utc"))

    for row in rows:
        model = str(row.get("model", "")).strip()
        if not model:
            continue
        base = re.sub(r"@reasoning=[^@]+$", "", model)
        first_seen = row_first_seen_timestamp(row)
        if first_seen is None:
            continue
        if model not in model_first_seen or first_seen < model_first_seen[model]:
            model_first_seen[model] = first_seen
        if base not in base_first_seen or first_seen < base_first_seen[base]:
            base_first_seen[base] = first_seen

    now_utc = dt.datetime.now(dt.UTC).replace(microsecond=0)
    window_start = now_utc - dt.timedelta(days=max(1, int(window_days)))

    recent_model_keys = sorted(
        model for model in current_models
        if (model_first_seen.get(model) is not None and model_first_seen[model] >= window_start)
    )
    recent_model_bases = sorted(
        model for model in current_model_bases
        if (base_first_seen.get(model) is not None and base_first_seen[model] >= window_start)
    )

    notes = (
        f"Exact model variants first added to this dataset in the last {int(window_days)} days."
    )
    if publish_mode == "replace":
        notes += " Replace publishes preserve first-seen timestamps when the prior sidecar is available."

    return {
        "generated_at_utc": isoformat_utc(now_utc),
        "publish_mode": publish_mode,
        "window_days": int(window_days),
        "window_start_utc": isoformat_utc(window_start),
        "models": recent_model_keys,
        "model_bases": recent_model_bases,
        "model_count": len(recent_model_keys),
        "model_base_count": len(recent_model_bases),
        "model_first_seen_utc": {key: isoformat_utc(model_first_seen[key]) for key in sorted(model_first_seen)},
        "model_base_first_seen_utc": {key: isoformat_utc(base_first_seen[key]) for key in sorted(base_first_seen)},
        "notes": notes,
    }

existing_dataset_present = responses_out.exists() and aggregate_out.exists()
if requested_mode == "auto":
    mode = "supplemental" if existing_dataset_present else "replace"
else:
    mode = requested_mode

incoming_responses = load_jsonl(responses_in)
incoming_aggregate_rows = load_jsonl(aggregate_rows_in)

if mode == "supplemental":
    existing_responses = load_jsonl(responses_out)
    existing_aggregate_rows = load_jsonl(aggregate_out)
    merged_responses, responses_added, responses_replaced = merge_by_sample_id(
        existing_responses, incoming_responses
    )
    merged_aggregate_rows, aggregate_added, aggregate_replaced = merge_by_sample_id(
        existing_aggregate_rows, incoming_aggregate_rows
    )
else:
    merged_responses = incoming_responses
    merged_aggregate_rows = incoming_aggregate_rows
    existing_aggregate_rows = []
    responses_added = len(incoming_responses)
    responses_replaced = 0
    aggregate_added = len(incoming_aggregate_rows)
    aggregate_replaced = 0

spec = importlib.util.spec_from_file_location(
    "openrouter_benchmark", root_dir / "scripts" / "openrouter_benchmark.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

for row in merged_responses:
    module.normalize_stored_model_reasoning_variant(row)
    module.annotate_response_outcome(row)
    module.enrich_collect_record_metrics(row)

for row in merged_aggregate_rows:
    module.normalize_stored_model_reasoning_variant(row)

responses_by_sample = {
    str(row.get("sample_id", "")).strip(): row
    for row in merged_responses
    if str(row.get("sample_id", "")).strip()
}
for row in merged_aggregate_rows:
    response_row = responses_by_sample.get(str(row.get("sample_id", "")).strip())
    if not response_row:
        continue
    row["response_outcome"] = response_row.get("response_outcome", "response")
    row["response_refusal"] = bool(response_row.get("response_refusal", False))
    row["response_native_finish_reason"] = response_row.get(
        "response_native_finish_reason"
    )
    if row["response_refusal"]:
        row["consensus_score"] = None
        row["consensus_error"] = None
        row["judge_valid_scores"] = []

merged_responses = slim_published_response_rows(merged_responses)
merged_aggregate_rows = slim_published_aggregate_rows(merged_aggregate_rows)
viewer_rows, viewer_details = build_viewer_assets(
    merged_responses,
    merged_aggregate_rows,
)
existing_recent = load_stats_if_exists(recent_additions_out)
for field in ("models",):
    values = existing_recent.get(field)
    if isinstance(values, list):
        existing_recent[field] = [
            value.replace(
                "anthropic/claude-fable-5@reasoning=minimal",
                "anthropic/claude-fable-5@reasoning=low",
            )
            if isinstance(value, str)
            else value
            for value in values
        ]
for field in ("model_first_seen_utc",):
    values = existing_recent.get(field)
    if isinstance(values, dict):
        old_key = "anthropic/claude-fable-5@reasoning=minimal"
        new_key = "anthropic/claude-fable-5@reasoning=low"
        if old_key in values:
            old_value = values.pop(old_key)
            if new_key not in values or str(old_value) < str(values[new_key]):
                values[new_key] = old_value
recent_additions = derive_recent_additions(
    merged_responses,
    existing_recent=existing_recent,
    window_days=recent_window_days,
    publish_mode=mode,
)

incoming_collection_stats = load_json(collection_stats_in)
existing_collection_stats = (
    load_stats_if_exists(collection_stats_out) if mode == "supplemental" else {}
)

attempt_values = []
for row in merged_responses:
    try:
        attempt_values.append(int(row.get("collect_attempt") or 0))
    except Exception:
        continue

if mode == "replace":
    elapsed_seconds = round(float(incoming_collection_stats.get("elapsed_seconds", 0) or 0), 3)
    attempt_count = int(incoming_collection_stats.get("attempt_count", 0) or 0)
    rate_limit_requeue_count = int(
        incoming_collection_stats.get("rate_limit_requeue_count", 0) or 0
    )
    final_rate_limit_error_count = int(
        incoming_collection_stats.get("final_rate_limit_error_count", 0) or 0
    )
elif responses_added > 0:
    elapsed_seconds = round(
        float(existing_collection_stats.get("elapsed_seconds", 0) or 0)
        + float(incoming_collection_stats.get("elapsed_seconds", 0) or 0),
        3,
    )
    attempt_count = int(existing_collection_stats.get("attempt_count", 0) or 0) + int(
        incoming_collection_stats.get("attempt_count", 0) or 0
    )
    rate_limit_requeue_count = int(
        existing_collection_stats.get("rate_limit_requeue_count", 0) or 0
    ) + int(incoming_collection_stats.get("rate_limit_requeue_count", 0) or 0)
    final_rate_limit_error_count = int(
        existing_collection_stats.get("final_rate_limit_error_count", 0) or 0
    ) + int(incoming_collection_stats.get("final_rate_limit_error_count", 0) or 0)
else:
    elapsed_seconds = round(float(existing_collection_stats.get("elapsed_seconds", 0) or 0), 3)
    attempt_count = int(existing_collection_stats.get("attempt_count", 0) or 0)
    rate_limit_requeue_count = int(
        existing_collection_stats.get("rate_limit_requeue_count", 0) or 0
    )
    final_rate_limit_error_count = int(
        existing_collection_stats.get("final_rate_limit_error_count", 0) or 0
    )

collection_stats = {
    "elapsed_seconds": elapsed_seconds,
    "total_records": len(merged_responses),
    "error_count": sum(1 for row in merged_responses if row.get("error")),
    "refusal_count": sum(
        1
        for row in merged_responses
        if str(row.get("response_outcome", "")).strip().lower() == "refusal"
    ),
    "success_count": sum(1 for row in merged_responses if not row.get("error")),
    "attempt_count": attempt_count,
    "max_attempt_observed": max(
        attempt_values
        or [
            int(existing_collection_stats.get("max_attempt_observed", 0) or 0),
            int(incoming_collection_stats.get("max_attempt_observed", 0) or 0),
        ]
    ),
    "rate_limit_requeue_count": rate_limit_requeue_count,
    "final_rate_limit_error_count": final_rate_limit_error_count,
    "resumed": False,
    "checkpoint_rows_at_start": 0,
    "new_rows_processed": int(responses_added),
    "usage_summary": module.summarize_collect_usage(merged_responses),
}

incoming_panel_summary = scrub_panel(load_json(panel_summary_in))
incoming_aggregate_summary = load_json(aggregate_summary_in)
existing_panel_summary = (
    scrub_panel(load_stats_if_exists(panel_summary_out)) if mode == "supplemental" else {}
)

panel_summary = incoming_panel_summary if mode == "replace" or not existing_panel_summary else existing_panel_summary
panel_summary = dict(panel_summary)
panel_summary["timestamp_utc"] = dt.datetime.now(dt.UTC).isoformat()
panel_summary["publish_mode"] = mode
panel_summary["disagreement_count"] = disagreement_count(merged_aggregate_rows)
panel_summary["disagreement_rate"] = round(
    panel_summary["disagreement_count"] / max(1, len(merged_aggregate_rows)), 4
)
if "grade_dirs_for_aggregate" in panel_summary:
    panel_summary["grade_dirs_for_aggregate"] = []

incoming_panel_id = str(incoming_panel_summary.get("panel_id", "")).strip()
if mode == "supplemental":
    current_panel_id = str(panel_summary.get("panel_id", "")).strip()
    if incoming_panel_id and incoming_panel_id != current_panel_id:
        source_panels = panel_summary.get("source_panels")
        if isinstance(source_panels, list):
            if incoming_panel_id not in source_panels:
                source_panels.append(incoming_panel_id)
        else:
            merged_source_panels = []
            if current_panel_id:
                merged_source_panels.append(current_panel_id)
            merged_source_panels.append(incoming_panel_id)
            panel_summary["source_panels"] = merged_source_panels
        if not str(panel_summary.get("execution_mode", "")).strip():
            panel_summary["execution_mode"] = "supplemental_merge"
        note = f"Supplemental publish appended panel {incoming_panel_id}."
        existing_notes = str(panel_summary.get("notes", "")).strip()
        if not existing_notes:
            panel_summary["notes"] = note
        elif note not in existing_notes:
            panel_summary["notes"] = f"{existing_notes} {note}".strip()

judge_models = panel_summary.get("judge_models")
num_judges = (
    len([m for m in judge_models if str(m).strip()]) if isinstance(judge_models, list) else 3
)
consensus_method = str(panel_summary.get("consensus_method", "")).strip() or str(
    incoming_aggregate_summary.get("consensus_method", "") or "mean"
)
aggregate_summary = module.summarize_aggregate_rows(
    merged_aggregate_rows,
    consensus_method=consensus_method,
    num_judges=max(1, num_judges),
)

write_jsonl(responses_out, merged_responses)
write_json(collection_stats_out, collection_stats)
write_json(panel_summary_out, panel_summary)
write_json(aggregate_summary_out, aggregate_summary)
write_jsonl(aggregate_out, merged_aggregate_rows)
write_json(recent_additions_out, recent_additions)
write_gzip_json(viewer_rows_out, viewer_rows)
write_gzip_json(viewer_details_out, viewer_details)

if model_launch_canonical.exists():
    model_launch_out.write_text(model_launch_canonical.read_text(encoding="utf-8"), encoding="utf-8")
else:
    model_launch_out.write_text(model_launch_headers + "\n", encoding="utf-8")

if model_params_canonical.exists():
    model_params_out.write_text(model_params_canonical.read_text(encoding="utf-8"), encoding="utf-8")
else:
    model_params_out.write_text(model_params_headers + "\n", encoding="utf-8")

print(
    json.dumps(
        {
            "mode": mode,
            "responses_added": responses_added,
            "responses_replaced": responses_replaced,
            "aggregate_added": aggregate_added,
            "aggregate_replaced": aggregate_replaced,
            "responses_rows": len(merged_responses),
            "aggregate_rows": len(merged_aggregate_rows),
        }
    )
)
PY

generated_at_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
responses_count="$(wc -l < "${OUTPUT_DIR}/responses.jsonl" | tr -d ' ')"
aggregate_row_count="$(wc -l < "${OUTPUT_DIR}/aggregate.jsonl" | tr -d ' ')"

python3 - <<'PY' "${OUTPUT_DIR}/aggregate_summary.json" "${OUTPUT_DIR}/aggregate.jsonl" "${OUTPUT_DIR}/leaderboard.csv"
import csv
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

summary_path = pathlib.Path(sys.argv[1])
aggregate_rows_path = pathlib.Path(sys.argv[2])
csv_path = pathlib.Path(sys.argv[3])

summary = json.loads(summary_path.read_text(encoding="utf-8"))
rows = summary.get("leaderboard", [])

fieldnames = [
    "rank",
    "model",
    "org",
    "reasoning",
    "avg_score",
    "green_rate",
    "red_rate",
    "refusal_rate",
    "score_2",
    "score_1",
    "score_0",
    "refusal_count",
    "answered_count",
    "nonsense_count",
    "error_count",
]

def normalize_org(org: str) -> str:
    text = str(org or "").strip() or "unknown"
    if text == "meta-llama":
        return "meta"
    return text

def parse_parts(model: str) -> tuple[str, str]:
    text = str(model or "")
    org = normalize_org(text.split("/", 1)[0] if "/" in text else "unknown")
    match = re.search(r"@reasoning=([^@]+)$", text)
    reasoning = match.group(1) if match else "default"
    return org, reasoning

org_votes: dict[str, Counter[str]] = defaultdict(Counter)
if aggregate_rows_path.exists():
    with aggregate_rows_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = str(row.get("model", "")).strip()
            org = normalize_org(str(row.get("model_org", "")).strip())
            if model and org:
                org_votes[model][org] += 1

def preferred_org(model: str) -> str:
    votes = org_votes.get(model)
    if votes:
        return votes.most_common(1)[0][0]
    org, _ = parse_parts(model)
    return org

with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for idx, row in enumerate(rows, start=1):
        model = str(row.get("model", ""))
        _, reasoning = parse_parts(model)
        org = preferred_org(model)
        writer.writerow(
            {
                "rank": idx,
                "model": model,
                "org": org,
                "reasoning": reasoning,
                "avg_score": row.get("avg_score"),
                "green_rate": row.get("detection_rate_score_2"),
                "red_rate": row.get("full_engagement_rate_score_0"),
                "refusal_rate": row.get("refusal_rate"),
                "score_2": row.get("score_2"),
                "score_1": row.get("score_1"),
                "score_0": row.get("score_0"),
                "refusal_count": row.get("refusal_count"),
                "answered_count": row.get("answered_count"),
                "nonsense_count": row.get("nonsense_count"),
                "error_count": row.get("error_count"),
            }
        )
PY

python3 - <<'PY' "${OUTPUT_DIR}/leaderboard.csv" "${OUTPUT_DIR}/model_launch_dates.csv" "${OUTPUT_DIR}/model_params.csv" "${OUTPUT_DIR}/leaderboard_with_launch.csv" "${generated_at_utc}"
import csv
import datetime as dt
import pathlib
import re
import sys

leaderboard_path = pathlib.Path(sys.argv[1])
launch_path = pathlib.Path(sys.argv[2])
params_path = pathlib.Path(sys.argv[3])
output_path = pathlib.Path(sys.argv[4])
generated_at_utc = str(sys.argv[5] or "").strip()
generated_date: dt.date | None = None
if generated_at_utc:
    try:
        generated_date = dt.datetime.fromisoformat(generated_at_utc.replace("Z", "+00:00")).date()
    except ValueError:
        generated_date = None

def base_model(model: str) -> str:
    return re.sub(r"@reasoning=[^@]+$", "", str(model or ""))

launch_map: dict[str, dict[str, str]] = {}
if launch_path.exists():
    with launch_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model_id = str(row.get("model_id", "")).strip()
            if model_id:
                launch_map[model_id] = row

params_map: dict[str, dict[str, str]] = {}
if params_path.exists():
    with params_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model_id = str(row.get("model_id", "")).strip()
            if model_id:
                params_map[model_id] = row

with leaderboard_path.open("r", encoding="utf-8", newline="") as handle:
    board_rows = list(csv.DictReader(handle))

fieldnames = list(board_rows[0].keys()) if board_rows else [
    "rank",
    "model",
    "org",
    "reasoning",
    "avg_score",
    "green_rate",
    "red_rate",
    "refusal_rate",
    "score_2",
    "score_1",
    "score_0",
    "refusal_count",
    "answered_count",
    "nonsense_count",
    "error_count",
]
for extra in (
    "model_base",
    "launch_date",
    "model_age_days",
    "launch_evidence_url",
    "open_model_status",
    "total_params_b",
    "active_params_b",
    "active_params_status",
    "model_license",
):
    if extra not in fieldnames:
        fieldnames.append(extra)

with output_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in board_rows:
        model_text = str(row.get("model", ""))
        model_base = base_model(model_text)
        launch = launch_map.get(model_base, {})
        params = params_map.get(model_base, {})
        launch_date_raw = str(launch.get("launch_date", "")).strip()
        launch_evidence_url = str(launch.get("evidence_url", "")).strip()
        model_age_days = ""
        if launch_date_raw and generated_date:
            try:
                launch_date = dt.date.fromisoformat(launch_date_raw)
            except ValueError:
                launch_date = None
            if launch_date is not None and launch_date <= generated_date:
                model_age_days = str((generated_date - launch_date).days)

        out = dict(row)
        out["model_base"] = model_base
        out["launch_date"] = launch_date_raw
        out["model_age_days"] = model_age_days
        out["launch_evidence_url"] = launch_evidence_url
        out["open_model_status"] = str(params.get("open_model_status", "")).strip()
        out["total_params_b"] = str(params.get("total_params_b", "")).strip()
        out["active_params_b"] = str(params.get("active_params_b", "")).strip()
        out["active_params_status"] = str(params.get("active_params_status", "")).strip()
        out["model_license"] = str(params.get("license", "")).strip()
        writer.writerow(out)
PY

python3 - <<'PY' "${OUTPUT_DIR}" "${generated_at_utc}"
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
generated_at_utc = str(sys.argv[2] or "").strip()

def jsonl_coverage(path: pathlib.Path) -> dict[str, int]:
    total_rows = 0
    rows_with_model_reasoning_level = 0
    rows_with_response_reasoning_effort = 0
    rows_with_response_usage = 0
    rows_with_response_reasoning_tokens = 0

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("model_reasoning_level", "")).strip():
                    rows_with_model_reasoning_level += 1
                if "response_reasoning_effort" in row:
                    rows_with_response_reasoning_effort += 1
                if isinstance(row.get("response_usage"), dict):
                    rows_with_response_usage += 1
                if "response_reasoning_tokens" in row:
                    rows_with_response_reasoning_tokens += 1

    return {
        "rows": total_rows,
        "rows_with_model_reasoning_level": rows_with_model_reasoning_level,
        "rows_with_response_reasoning_effort": rows_with_response_reasoning_effort,
        "rows_with_response_usage": rows_with_response_usage,
        "rows_with_response_reasoning_tokens": rows_with_response_reasoning_tokens,
    }

responses_rows = sum(1 for line in (output_dir / "responses.jsonl").open("r", encoding="utf-8") if line.strip())
aggregate_rows = sum(1 for line in (output_dir / "aggregate.jsonl").open("r", encoding="utf-8") if line.strip())

manifest = {
    "generated_at_utc": generated_at_utc,
    "sources": {
        "responses_file": f"{output_dir}/responses.jsonl",
        "collection_stats_file": f"{output_dir}/collection_stats.json",
        "panel_summary_file": f"{output_dir}/panel_summary.json",
        "aggregate_summary_file": f"{output_dir}/aggregate_summary.json",
        "aggregate_rows_file": f"{output_dir}/aggregate.jsonl",
        "recent_additions_file": f"{output_dir}/recent_additions.json",
        "viewer_rows_file": f"{output_dir}/viewer_rows.json.gz",
        "viewer_details_file": f"{output_dir}/viewer_details.json.gz",
    },
    "counts": {
        "responses_rows": responses_rows,
        "aggregate_rows": aggregate_rows,
        "viewer_rows_bytes": (output_dir / "viewer_rows.json.gz").stat().st_size,
        "viewer_details_bytes": (output_dir / "viewer_details.json.gz").stat().st_size,
    },
    "coverage": {
        "responses": jsonl_coverage(output_dir / "responses.jsonl"),
        "aggregate": jsonl_coverage(output_dir / "aggregate.jsonl"),
    },
    "exports": {
        "leaderboard_csv": f"{output_dir}/leaderboard.csv",
        "leaderboard_with_launch_csv": f"{output_dir}/leaderboard_with_launch.csv",
        "model_launch_dates_csv": f"{output_dir}/model_launch_dates.csv",
        "model_params_csv": f"{output_dir}/model_params.csv",
    },
}

(output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

echo "Published viewer dataset to ${OUTPUT_DIR}"

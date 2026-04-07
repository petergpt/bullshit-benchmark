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
model_launch_out = output_dir / "model_launch_dates.csv"
model_params_out = output_dir / "model_params.csv"

path_pattern = re.compile(r"/Users/[^\s\"|]+")

def sanitize_string(value: str) -> str:
    return path_pattern.sub("[local-path]", value)

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
    if isinstance(value, str) and "/Users/" in value:
        return None
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
        slimmed.append(slim)
    return slimmed

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
    module.enrich_collect_record_metrics(row)

merged_responses = slim_published_response_rows(merged_responses)

incoming_model_keys, incoming_model_bases = collect_model_sets(incoming_aggregate_rows)
existing_model_keys, existing_model_bases = collect_model_sets(existing_aggregate_rows)
recent_additions_note = "Exact model variants newly added in the most recent publish."
if mode == "replace":
    recent_model_keys = sorted(incoming_model_keys)
    recent_model_bases = sorted(incoming_model_bases)
else:
    recent_model_keys = sorted(model for model in incoming_model_keys if model not in existing_model_keys)
    recent_model_bases = sorted(model for model in incoming_model_bases if model not in existing_model_bases)
    if not recent_model_keys and not recent_model_bases:
        existing_recent = load_stats_if_exists(recent_additions_out)
        preserved_model_keys = (
            sorted({str(v).strip() for v in existing_recent.get("models", []) if str(v).strip()})
            if isinstance(existing_recent.get("models"), list)
            else []
        )
        preserved_model_bases = (
            sorted({str(v).strip() for v in existing_recent.get("model_bases", []) if str(v).strip()})
            if isinstance(existing_recent.get("model_bases"), list)
            else []
        )
        if preserved_model_keys or preserved_model_bases:
            recent_model_keys = preserved_model_keys
            recent_model_bases = preserved_model_bases
            recent_additions_note = "Exact model variants preserved from the prior publish because this refresh replaced rows without introducing new models."
        elif aggregate_added == 0 and aggregate_replaced > 0 and incoming_aggregate_rows:
            recent_model_keys = sorted(incoming_model_keys)
            recent_model_bases = sorted(incoming_model_bases)
            recent_additions_note = "Exact model variants included in the incoming run payload. This publish refreshed metadata without adding new rows to the dataset."

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
write_json(
    recent_additions_out,
    {
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "publish_mode": mode,
        "models": recent_model_keys,
        "model_bases": recent_model_bases,
        "model_count": len(recent_model_keys),
        "model_base_count": len(recent_model_bases),
        "notes": recent_additions_note,
    },
)

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
    "score_2",
    "score_1",
    "score_0",
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
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
                "score_2": row.get("score_2"),
                "score_1": row.get("score_1"),
                "score_0": row.get("score_0"),
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
    "score_2",
    "score_1",
    "score_0",
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
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
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

cat > "${OUTPUT_DIR}/manifest.json" <<EOF
{
  "generated_at_utc": "${generated_at_utc}",
  "sources": {
    "responses_file": "${OUTPUT_DIR}/responses.jsonl",
    "collection_stats_file": "${OUTPUT_DIR}/collection_stats.json",
    "panel_summary_file": "${OUTPUT_DIR}/panel_summary.json",
    "aggregate_summary_file": "${OUTPUT_DIR}/aggregate_summary.json",
    "aggregate_rows_file": "${OUTPUT_DIR}/aggregate.jsonl",
    "recent_additions_file": "${OUTPUT_DIR}/recent_additions.json"
  },
  "counts": {
    "responses_rows": ${responses_count},
    "aggregate_rows": ${aggregate_row_count}
  },
  "exports": {
    "leaderboard_csv": "${OUTPUT_DIR}/leaderboard.csv",
    "leaderboard_with_launch_csv": "${OUTPUT_DIR}/leaderboard_with_launch.csv",
    "model_launch_dates_csv": "${OUTPUT_DIR}/model_launch_dates.csv",
    "model_params_csv": "${OUTPUT_DIR}/model_params.csv"
  }
}
EOF

echo "Published viewer dataset to ${OUTPUT_DIR}"

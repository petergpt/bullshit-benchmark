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
    [--output-dir data/latest]

Copies the selected run artifacts into a stable viewer dataset directory:
  responses.jsonl
  collection_stats.json
  panel_summary.json
  aggregate_summary.json
  aggregate.jsonl
  leaderboard.csv
  leaderboard_with_launch.csv
  model_launch_dates.csv
  manifest.json
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

copy_file() {
  local source_file="$1"
  local target_file="$2"
  if [[ -e "${target_file}" ]]; then
    local same_file
    same_file="$(
      python3 - <<'PY' "${source_file}" "${target_file}"
import os
import sys

source_path = sys.argv[1]
target_path = sys.argv[2]
try:
    print("1" if os.path.samefile(source_path, target_path) else "0")
except FileNotFoundError:
    print("0")
PY
    )"
    if [[ "${same_file}" == "1" ]]; then
      return
    fi
  fi
  cp "${source_file}" "${target_file}"
}

copy_file "${RESPONSES_FILE}" "${OUTPUT_DIR}/responses.jsonl"
copy_file "${COLLECTION_STATS_FILE}" "${OUTPUT_DIR}/collection_stats.json"
copy_file "${PANEL_SUMMARY_FILE}" "${OUTPUT_DIR}/panel_summary.json"
copy_file "${AGGREGATE_SUMMARY_FILE}" "${OUTPUT_DIR}/aggregate_summary.json"
copy_file "${AGGREGATE_ROWS_FILE}" "${OUTPUT_DIR}/aggregate.jsonl"

MODEL_LAUNCH_CANONICAL="${ROOT_DIR}/data/model_metadata/model_launch_dates.csv"
MODEL_LAUNCH_HEADERS="model_id,org,launch_date,evidence_url,evidence_title,evidence_published_date,evidence_type,judge_status,notes,updated_at_utc"
if [[ -f "${MODEL_LAUNCH_CANONICAL}" ]]; then
  copy_file "${MODEL_LAUNCH_CANONICAL}" "${OUTPUT_DIR}/model_launch_dates.csv"
else
  printf '%s\n' "${MODEL_LAUNCH_HEADERS}" > "${OUTPUT_DIR}/model_launch_dates.csv"
fi

python3 - <<'PY' "${OUTPUT_DIR}/panel_summary.json"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))

def scrub(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.endswith("_dir") or key_text.endswith("_dirs"):
                continue
            scrubbed = scrub(item)
            if scrubbed is None:
                continue
            out[key] = scrubbed
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            scrubbed = scrub(item)
            if scrubbed is None:
                continue
            out.append(scrubbed)
        return out
    if isinstance(value, str) and "/Users/" in value:
        return None
    return value

data = scrub(data)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

python3 - <<'PY' "${OUTPUT_DIR}/aggregate.jsonl"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

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

    buf.append(ch)
    if ch == '"':
        in_string = True
    elif ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth == 0:
            rows.append(json.loads("".join(buf)))
            buf = []

for row in rows:
    for key in list(row.keys()):
        if key.endswith("_grade_dir"):
            row.pop(key, None)

path.write_text(
    "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    encoding="utf-8",
)
PY

generated_at_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
responses_count="$(wc -l < "${OUTPUT_DIR}/responses.jsonl" | tr -d ' ')"
aggregate_row_count="$(wc -l < "${OUTPUT_DIR}/aggregate.jsonl" | tr -d ' ')"

python3 - <<'PY' "${OUTPUT_DIR}/aggregate_summary.json" "${OUTPUT_DIR}/leaderboard.csv"
import csv
import json
import pathlib
import re
import sys

summary_path = pathlib.Path(sys.argv[1])
csv_path = pathlib.Path(sys.argv[2])

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

def parse_parts(model: str) -> tuple[str, str]:
    text = str(model or "")
    org = text.split("/", 1)[0] if "/" in text else "unknown"
    match = re.search(r"@reasoning=([^@]+)$", text)
    reasoning = match.group(1) if match else "default"
    return org, reasoning

with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for idx, row in enumerate(rows, start=1):
        model = str(row.get("model", ""))
        org, reasoning = parse_parts(model)
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

python3 - <<'PY' "${OUTPUT_DIR}/leaderboard.csv" "${OUTPUT_DIR}/model_launch_dates.csv" "${OUTPUT_DIR}/leaderboard_with_launch.csv" "${generated_at_utc}"
import csv
import datetime as dt
import pathlib
import re
import sys

leaderboard_path = pathlib.Path(sys.argv[1])
launch_path = pathlib.Path(sys.argv[2])
output_path = pathlib.Path(sys.argv[3])
generated_at_utc = str(sys.argv[4] or "").strip()
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
for extra in ("model_base", "launch_date", "model_age_days", "launch_evidence_url"):
    if extra not in fieldnames:
        fieldnames.append(extra)

with output_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in board_rows:
        model_text = str(row.get("model", ""))
        model_base = base_model(model_text)
        launch = launch_map.get(model_base, {})
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
    "aggregate_rows_file": "${OUTPUT_DIR}/aggregate.jsonl"
  },
  "counts": {
    "responses_rows": ${responses_count},
    "aggregate_rows": ${aggregate_row_count}
  },
  "exports": {
    "leaderboard_csv": "${OUTPUT_DIR}/leaderboard.csv",
    "leaderboard_with_launch_csv": "${OUTPUT_DIR}/leaderboard_with_launch.csv",
    "model_launch_dates_csv": "${OUTPUT_DIR}/model_launch_dates.csv"
  }
}
EOF

echo "Published viewer dataset to ${OUTPUT_DIR}"
